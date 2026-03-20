#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║           SynthGen  —  Synthetic Conversation Generator  ║
║   Streams · Rotates models · Deduplicates · Per-cat JSONL ║
╚══════════════════════════════════════════════════════════╝

Usage (PowerShell — all one line):
    python generate.py generate --models "model-id-1" "model-id-2" --count 50000 --rotate-every 200 --output-dir ./convos

    python generate.py stats  --output-dir ./convos
    python generate.py sample --output-dir ./convos --category rust_ownership

Requirements:
    pip install requests rich
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import signal
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Callable, Optional

import requests
from rich import box
from rich.console import Console
from rich.table import Table
from rich.theme import Theme
from rich.panel import Panel   # used only in sample_cmd

# ─────────────────────────────────────────────────────────
#  THEME & CONSOLE
# ─────────────────────────────────────────────────────────

THEME = Theme({
    "info":    "dim cyan",
    "success": "bold green",
    "warn":    "bold yellow",
    "error":   "bold red",
    "model":   "bold #7c5cbf",
    "cat":     "italic #5fafd7",
    "token":   "dim white",
    "dedup":   "bold #ff8c00",
    "think":   "yellow",
    "dim":     "dim",
})

console = Console(theme=THEME, highlight=False)
log = logging.getLogger("synthgen")

# Enable ANSI escape processing on Windows
if sys.platform == "win32":
    import os; os.system("")

# ─────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────

LM_BASE      = "http://localhost:1234"
API_V1       = f"{LM_BASE}/api/v1"
API_OAI      = f"{LM_BASE}/v1"
TEMPERATURE  = 0.87
MAX_TOKENS   = 2048
CTX_LENGTH   = 4096
REQ_TIMEOUT  = 180
LOAD_TIMEOUT = 120
MAX_RETRIES  = 3
RETRY_DELAY  = 6

# ─────────────────────────────────────────────────────────
#  GRACEFUL SHUTDOWN
# ─────────────────────────────────────────────────────────

_shutdown = Event()

def _handle_sigint(sig, frame):
    if not _shutdown.is_set():
        console.print("\n[warn]Ctrl+C — finishing current conversation then exiting...[/warn]")
        _shutdown.set()

signal.signal(signal.SIGINT,  _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)

# ─────────────────────────────────────────────────────────
#  SESSION STATS  (plain mutable object, no threading needed)
# ─────────────────────────────────────────────────────────

@dataclass
class SessionStats:
    generated:      int   = 0
    parse_fail:     int   = 0
    api_fail:       int   = 0
    skip:           int   = 0
    dedup_hit:      int   = 0
    total_tokens:   int   = 0
    think_tokens:   int   = 0
    start_time:     float = field(default_factory=time.time)
    model_counts:   Counter = field(default_factory=Counter)
    current_model:  str   = ""
    current_cat:    str   = ""
    current_topic:  str   = ""
    tokens_per_sec: float = 0.0
    # Stream display state
    stream_buf:     str   = ""   # last ~200 chars of output tokens
    stream_phase:   str   = "idle"  # "idle" | "thinking" | "generating"

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def conv_rate(self) -> str:
        r = self.generated / self.elapsed * 60 if self.elapsed > 0 else 0
        return f"{r:.1f}/min" if self.generated > 0 else "—"

# ─────────────────────────────────────────────────────────
#  DEDUPLICATION STORE
# ─────────────────────────────────────────────────────────

class DedupStore:
    """SHA-256 of the first user turn, persisted per category."""

    def __init__(self, output_dir: str):
        self._dir = Path(output_dir) / ".hashes"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, set] = {}

    def _path(self, cat_id: str) -> Path:
        return self._dir / f"{cat_id}.txt"

    def _load(self, cat_id: str) -> None:
        if cat_id in self._store:
            return
        s: set = set()
        p = self._path(cat_id)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                s = {l.strip() for l in f if l.strip()}
        self._store[cat_id] = s

    def load_all(self, cat_ids: list) -> int:
        for cid in cat_ids:
            self._load(cid)
        return self.total()

    @staticmethod
    def _fp(conv: dict) -> str:
        first = next(
            (t["content"] for t in conv.get("turns", []) if t.get("role") == "user"),
            ""
        )
        norm = re.sub(r"\s+", " ", first.lower().strip())
        return hashlib.sha256(norm.encode()).hexdigest()

    def seen(self, cat_id: str, conv: dict) -> bool:
        self._load(cat_id)
        return self._fp(conv) in self._store[cat_id]

    def register(self, cat_id: str, conv: dict) -> None:
        self._load(cat_id)
        fp = self._fp(conv)
        if fp not in self._store[cat_id]:
            self._store[cat_id].add(fp)
            with open(self._path(cat_id), "a", encoding="utf-8") as f:
                f.write(fp + "\n")

    def count(self, cat_id: str) -> int:
        self._load(cat_id)
        return len(self._store.get(cat_id, set()))

    def total(self) -> int:
        return sum(len(v) for v in self._store.values())

# ─────────────────────────────────────────────────────────
#  PROMPT VARIATOR
# ─────────────────────────────────────────────────────────

_PERSONAS = [
    "a complete beginner who has never programmed before",
    "a self-taught hobbyist programmer",
    "a Python developer learning Rust for the first time",
    "a junior developer (6 months experience) at a startup",
    "a mid-level developer with 3 years of experience in C++",
    "a senior backend engineer who is new to this specific topic",
    "a computer science student in their second year",
    "a data scientist who writes mostly Python and R",
    "a DevOps engineer learning to write tooling in Rust",
    "an experienced Java developer exploring systems programming",
    "a bootcamp graduate on their first job",
    "a developer returning to coding after a 5-year break",
    "a technical lead reviewing new technology for their team",
    "a game developer exploring Rust for performance-critical code",
    "a frontend developer curious about low-level programming",
]

_SCENARIOS = [
    "working on a personal side project over the weekend",
    "debugging a production issue at work under time pressure",
    "going through an online tutorial and getting stuck",
    "preparing for a technical interview",
    "reviewing a colleague's code during a code review",
    "reading official documentation and finding it confusing",
    "trying to port an existing project to a new language",
    "building a proof-of-concept to evaluate a technology",
    "following along with a conference talk or blog post",
    "exploring the topic out of pure curiosity",
    "working through exercises in a programming book",
    "trying to fix a failing CI/CD pipeline",
    "onboarding to a new codebase at a new job",
    "building a tool to automate a repetitive task",
    "contributing to an open-source project for the first time",
]

_OPENING_STYLES = [
    "The user opens with a direct, specific technical question.",
    "The user opens by describing a problem they're stuck on and sharing a code snippet.",
    "The user opens with a broad conceptual question, then narrows down as they learn.",
    "The user opens by admitting they've read about this but still don't fully understand it.",
    "The user opens with an incorrect assumption that gets gently corrected.",
    "The user opens by comparing this concept to something they already know.",
    "The user opens with a 'why does this even work this way?' frustration.",
    "The user opens with a 'what's the difference between X and Y?' question.",
    "The user opens by pasting an error message and asking what went wrong.",
    "The user opens with a working solution but asks if there's a better way.",
    "The user asks a simple question, then keeps asking follow-ups that go deeper.",
    "The user opens with a hypothetical scenario to test their understanding.",
]

_CATEGORY_ANGLES: dict[str, list] = {
    "rust_ownership": [
        "moving a value into a function and not being able to use it afterward",
        "trying to have two mutable references at the same time",
        "understanding why Clone is sometimes needed",
        "the difference between moving and copying types",
        "how ownership works with structs and their fields",
        "passing owned data into a thread with move closures",
        "why returning references from functions causes lifetime issues",
        "the mental model: ownership as responsibility for cleanup",
    ],
    "rust_lifetimes": [
        "why the compiler needs explicit lifetime annotations",
        "lifetime elision rules and when they apply",
        "lifetimes in struct definitions",
        "'static lifetime and when to use it",
        "lifetime bounds on trait implementations",
        "why two references with different lifetimes can't be returned",
        "the relationship between lifetimes and the borrow checker",
    ],
    "rust_error_handling": [
        "the ? operator and how it propagates errors",
        "defining custom error types with thiserror",
        "when to use anyhow vs thiserror",
        "converting between different error types",
        "the difference between unwrap, expect, and proper handling",
        "error handling in async code",
        "wrapping third-party errors in your own type",
    ],
    "rust_async": [
        "what the Future trait actually is under the hood",
        "why you need a runtime like Tokio",
        "tokio::spawn vs async blocks",
        "sharing state between async tasks",
        "select! for racing multiple futures",
        "async in traits (the challenges and workarounds)",
        "cancellation and why it matters",
        "blocking code inside async context",
    ],
    "rust_traits": [
        "the difference between trait objects (dyn) and generics",
        "associated types vs generic type parameters",
        "implementing Display and Debug",
        "the From and Into conversion traits",
        "operator overloading with std traits",
        "trait bounds in where clauses",
        "blanket implementations",
        "object safety rules for dyn Trait",
    ],
    "rust_concurrency": [
        "why Arc<Mutex<T>> is the standard pattern",
        "the difference between Mutex and RwLock",
        "channel-based message passing with mpsc",
        "what Send and Sync actually mean",
        "deadlock scenarios and how to avoid them",
        "rayon for data parallelism",
        "atomic types and when to use them",
    ],
    "rust_debugging": [
        "a confusing lifetime error with multiple borrows",
        "cannot borrow as mutable because it's also borrowed as immutable",
        "value used after move error",
        "type mismatch with generic constraints",
        "a stack overflow from accidental recursion",
        "unexpected behavior from integer overflow in debug vs release",
        "a subtle bug with shadowing inside a loop",
    ],
    "rust_code_review": [
        "unnecessary cloning that hurts performance",
        "using unwrap() everywhere instead of proper error handling",
        "non-idiomatic use of match where if let would be cleaner",
        "inefficient string concatenation in a loop",
        "missing documentation on public API",
        "overly complex lifetime annotations that can be simplified",
        "mutable state that should be refactored into a cleaner design",
    ],
    "algorithms_general": [
        "implementing binary search from scratch",
        "understanding why quicksort is O(n log n) average case",
        "BFS vs DFS and when to use each",
        "dynamic programming bottom-up vs top-down",
        "hash map collision handling strategies",
        "graph shortest path algorithms",
        "amortized analysis of dynamic arrays",
    ],
    "systems_programming": [
        "why stack vs heap allocation matters for performance",
        "what a segfault actually is at the hardware level",
        "how virtual memory works",
        "what happens when you dereference a null pointer",
        "the difference between a process and a thread",
        "why context switching is expensive",
        "cache lines and why data layout matters",
    ],
    "linux_cli": [
        "piping commands together to process text",
        "writing a bash script to automate a repetitive task",
        "understanding file permissions and chmod",
        "using grep, awk, and sed for log analysis",
        "process management with ps, kill, and signals",
        "setting up environment variables and .bashrc",
        "ssh port forwarding for accessing remote services",
    ],
    "code_debug_session": [
        "an off-by-one error in array indexing",
        "a race condition in multithreaded code",
        "a memory leak from a missing cleanup step",
        "incorrect regex that matches too broadly",
        "a subtle floating point precision issue",
        "unexpected behavior from implicit type conversion",
        "a bug that only appears in release builds",
        "wrong assumption about API return values",
    ],
    "casual_chat": [
        "two developers chatting about weekend plans",
        "colleagues discussing a recent movie or TV show",
        "friends debating the best pizza toppings",
        "two people comparing their morning routines",
        "coworkers chatting about a local sports team",
        "friends planning a hiking trip",
        "discussing a book one of them just finished",
    ],
    "learning_programming": [
        "understanding what a variable actually is",
        "the difference between a function and a method",
        "why do we even need loops?",
        "what recursion is and when to use it",
        "understanding why types matter",
        "what an API is in simple terms",
        "debugging their first Hello World that won't run",
    ],
}

_GENERIC_ANGLES = [
    "a foundational concept in this area",
    "a common mistake beginners make",
    "a performance or efficiency consideration",
    "a real-world use case or example",
    "comparing two different approaches",
    "best practices and idiomatic style",
    "a specific edge case or gotcha",
    "how this connects to something else they know",
]

_REQUEST_PHRASINGS = [
    "Generate a realistic multi-turn conversation with exactly {n} total turns.",
    "Write a natural {n}-turn dialogue between a user and a knowledgeable assistant.",
    "Create a {n}-turn back-and-forth conversation that feels like a real chat.",
    "Produce a realistic {n}-turn exchange that could appear in a developer forum or Slack.",
    "Simulate a {n}-turn conversation a developer might have while learning this topic.",
    "Write a {n}-turn Q&A conversation that reads like a natural human interaction.",
    "Generate a {n}-turn dialogue — make it feel authentic, not like a textbook.",
    "Craft a {n}-turn conversation between someone learning and someone teaching.",
]


class PromptVariator:
    def build(self, cat: dict, num_turns: int) -> str:
        persona  = random.choice(_PERSONAS)
        scenario = random.choice(_SCENARIOS)
        opening  = random.choice(_OPENING_STYLES)
        phrasing = random.choice(_REQUEST_PHRASINGS).format(n=num_turns)
        angles   = _CATEGORY_ANGLES.get(cat["id"], _GENERIC_ANGLES)
        angle    = random.choice(angles)
        mods     = random.sample([
            "Include at least one code snippet.",
            "Include a realistic error message.",
            "The user makes a wrong assumption early on.",
            "The assistant corrects a misconception gently.",
            "The user asks a follow-up that goes deeper than expected.",
            "The conversation ends with the user having a clear aha-moment.",
            "The assistant uses an analogy to explain a concept.",
            "The user pushes back on the first explanation.",
            "The exchange has a slightly informal, friendly tone.",
            "The user pastes code and asks what's wrong with it.",
            "The assistant asks a clarifying question mid-conversation.",
            "The user tries something, it doesn't work, and they report back.",
        ], k=random.randint(2, 3))

        return (
            f"{phrasing}\n\n"
            f"User persona: {persona}\n"
            f"Situation: {scenario}\n"
            f"Specific angle to focus on: {angle}\n"
            f"Opening style: {opening}\n"
            f"Additional requirements: {' '.join(mods)}\n\n"
            f"Topic: {cat['topic']}\n"
            f"Description: {cat['description']}\n"
            f"Category guidance: {cat['system_hint']}\n\n"
            f"Output raw JSON only:\n"
            f'{{"category":"{cat["subcategory"]}","topic":"{cat["topic"]}","turns":[...]}}'
        )

# ─────────────────────────────────────────────────────────
#  CATEGORY / OUTPUT HELPERS
# ─────────────────────────────────────────────────────────

def load_categories(path: str) -> list:
    cats = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cats.append(json.loads(line))
    if not cats:
        console.print("[error]No categories found.[/error]")
        sys.exit(1)
    return cats

def build_weighted_pool(cats: list) -> list:
    pool = []
    for c in cats:
        pool.extend([c] * c.get("weight", 1))
    return pool

def output_path(output_dir: str, cat_id: str) -> Path:
    return Path(output_dir) / f"{cat_id}.jsonl"

def count_existing(output_dir: str, cat_id: str) -> int:
    p = output_path(output_dir, cat_id)
    if not p.exists():
        return 0
    with open(p, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

def total_existing(output_dir: str, cats: list) -> int:
    return sum(count_existing(output_dir, c["id"]) for c in cats)

def append_conversation(output_dir: str, cat_id: str, conv: dict) -> None:
    p = output_path(output_dir, cat_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(conv, ensure_ascii=False) + "\n")

# ─────────────────────────────────────────────────────────
#  LM STUDIO API
# ─────────────────────────────────────────────────────────

def list_loaded_models() -> list:
    try:
        r = requests.get(f"{LM_BASE}/api/v0/models", timeout=10)
        r.raise_for_status()
        return [m for m in r.json().get("data", []) if m.get("state") == "loaded"]
    except requests.exceptions.ConnectionError:
        console.print("[error]Cannot connect to LM Studio on port 1234.[/error]")
        sys.exit(1)
    except Exception as e:
        log.warning(f"list_models: {e}")
        return []

def unload_model(instance_id: str) -> bool:
    try:
        r = requests.post(f"{API_V1}/models/unload",
                          json={"instance_id": instance_id}, timeout=30)
        r.raise_for_status()
        return True
    except Exception as e:
        log.warning(f"unload {instance_id}: {e}")
        return False

def unload_all() -> None:
    for m in list_loaded_models():
        mid = m.get("id") or m.get("instance_id", "")
        if mid:
            unload_model(mid)

def load_model(model_id: str) -> Optional[str]:
    console.print(f"[model]Loading: {model_id}[/model]")
    try:
        r = requests.post(
            f"{API_V1}/models/load",
            json={"model": model_id, "context_length": CTX_LENGTH,
                  "flash_attention": True, "echo_load_config": True},
            timeout=LOAD_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        iid  = data.get("instance_id", model_id)
        secs = data.get("load_time_seconds")
        t    = f" ({secs:.1f}s)" if secs else ""
        console.print(f"[success]  Loaded{t} → {iid}[/success]")
        return iid
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else 0
        if code == 409:
            console.print(f"[warn]  Already loaded — reusing[/warn]")
            for m in list_loaded_models():
                if model_id in (m.get("id", ""), m.get("instance_id", "")):
                    return m.get("instance_id") or m.get("id", model_id)
            return model_id
        log.warning(f"HTTP {code} loading {model_id}")
        return None
    except Exception as e:
        log.warning(f"load_model: {e}")
        return None

# ─────────────────────────────────────────────────────────
#  SYSTEM PROMPT
# ─────────────────────────────────────────────────────────

SYS_PROMPT = (
    "You are a synthetic training data generator. "
    "Produce realistic multi-turn conversations for training a language model.\n\n"
    "RULES:\n"
    "- Output ONLY raw JSON. No markdown, no code fences, no preamble.\n"
    '- Schema: {"category":"...","topic":"...","turns":['
    '{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}\n'
    "- Turns strictly alternate user/assistant, starting with user.\n"
    "- For technical topics: use real code and realistic error messages.\n"
    "- For casual topics: use natural informal language.\n"
    "- Every conversation must feel genuinely unique.\n"
    "- Raw JSON ONLY. No ```json``` wrappers ever."
)

# ─────────────────────────────────────────────────────────
#  STREAMING CALL
#
#  The on_token callback is called on every output token so the
#  display can update in real time while the HTTP stream is open.
#  Thinking tokens are shown but stripped from the returned string.
# ─────────────────────────────────────────────────────────

_THINK_OPEN  = re.compile(r"<think>",   re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>",  re.IGNORECASE)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def stream_completion(
    instance_id: str,
    prompt: str,
    stats: SessionStats,
    on_think: Callable[[str], None],
    on_output: Callable[[str], None],
) -> Optional[str]:
    """
    Stream from LM Studio's OpenAI-compatible endpoint.
    Updates stats.stream_buf and stats.stream_phase live.
    Calls on_token() periodically so the caller can refresh the display.
    Returns the response text with <think> blocks stripped, or None on error.
    """
    payload = {
        "model":    instance_id,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "temperature": TEMPERATURE,
        "max_tokens":  MAX_TOKENS,
        "stream":      True,
    }

    output: list[str] = []
    token_count  = 0
    think_count  = 0
    t_start      = time.time()
    in_think     = False

    stats.stream_phase = "thinking"

    def _append_output(text: str) -> None:
        output.append(text)
        stats.stream_phase = "generating"
        on_output(text)

    def _append_think(text: str) -> None:
        nonlocal think_count
        think_count += 1
        stats.think_tokens = think_count
        stats.stream_phase = "thinking"
        on_think(text)

    try:
        with requests.post(
            f"{API_OAI}/chat/completions",
            json=payload,
            stream=True,
            timeout=REQ_TIMEOUT,
        ) as resp:
            resp.raise_for_status()
            stats.stream_phase = "thinking"

            for raw_line in resp.iter_lines():
                if _shutdown.is_set():
                    return None
                if not raw_line:
                    continue

                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue
                payload_str = line[6:].strip()
                if payload_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue

                delta = chunk.get("choices", [{}])[0].get("delta", {})

                # Pattern 1: explicit reasoning_content field (LM Studio)
                reasoning = delta.get("reasoning_content")
                if reasoning:
                    _append_think(reasoning)
                    token_count += 1
                    continue

                content = delta.get("content")
                if not content:
                    continue

                token_count += 1

                # Pattern 2: <think>...</think> tags in content
                if in_think:
                    if _THINK_CLOSE.search(content):
                        parts = _THINK_CLOSE.split(content, maxsplit=1)
                        _append_think(parts[0])
                        in_think = False
                        remainder = parts[-1] if len(parts) > 1 else ""
                        if remainder:
                            _append_output(remainder)
                    else:
                        _append_think(content)
                    continue

                if _THINK_OPEN.search(content):
                    if _THINK_CLOSE.search(content):
                        # Self-contained <think>...</think>
                        clean = _THINK_BLOCK.sub("", content)
                        _append_think(content)
                        if clean.strip():
                            _append_output(clean)
                    else:
                        in_think = True
                        _append_think(content)
                    continue

                # Normal output
                _append_output(content)

    except requests.exceptions.ConnectionError:
        console.print("[error]Lost connection to LM Studio.[/error]")
        stats.stream_phase = "idle"
        return None
    except requests.exceptions.Timeout:
        log.warning("Request timed out.")
        stats.stream_phase = "idle"
        return None
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        log.warning(f"HTTP {code} during stream.")
        stats.stream_phase = "idle"
        return None
    except Exception as e:
        log.warning(f"stream error: {e}")
        stats.stream_phase = "idle"
        return None

    elapsed = time.time() - t_start
    stats.total_tokens   += token_count
    stats.tokens_per_sec  = token_count / elapsed if elapsed > 0 else 0.0
    stats.stream_phase    = "idle"

    return "".join(output) if output else None

# ─────────────────────────────────────────────────────────
#  PARSE & VALIDATE
# ─────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────
#  PARSE & REPAIR
#
#  Small models produce broken JSON in predictable ways.
#  Rather than retrying the whole generation, we try to fix it.
#
#  Repair pipeline (applied in order, stops at first success):
#
#  1. Strip markdown fences and leading/trailing prose
#  2. Straight json.loads
#  3. Single-quote → double-quote coercion
#  4. Truncated JSON — try closing open arrays/objects
#  5. Unescaped inner quotes inside string values
#  6. Wrong role names (Human/Assistant/user_turn/etc.)
#  7. Alternation fix — drop or relabel turns so they strictly alternate
#  8. Odd-turn fix — drop the trailing user turn
#  9. Nested wrapper — unwrap {"conversation":{"turns":[...]}}
# ─────────────────────────────────────────────────────────

_FENCE        = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_SINGLE_QUOTE = re.compile(r"(?<!\\)'")   # naive single-quote swap

# Role name aliases the model might use instead of "user"/"assistant"
_ROLE_ALIASES = {
    "human":       "user",
    "user_turn":   "user",
    "user turn":   "user",
    "person":      "user",
    "questioner":  "user",
    "student":     "user",
    "developer":   "user",
    "programmer":  "user",
    "bot":         "assistant",
    "ai":          "assistant",
    "gpt":         "assistant",
    "claude":      "assistant",
    "assistant_turn": "assistant",
    "assistant turn":  "assistant",
    "response":    "assistant",
    "answer":      "assistant",
    "teacher":     "assistant",
    "expert":      "assistant",
    "system":      "assistant",   # some models use system for the reply
}

def _normalize_role(role: str) -> str:
    return _ROLE_ALIASES.get(role.lower().strip(), role.lower().strip())

def _try_parse(text: str) -> Optional[dict]:
    """Attempt json.loads, return dict or None."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def _extract_json_object(text: str) -> str:
    """
    Slice from first { to last } to strip leading/trailing prose.

    Two guards prevent over-aggressive trimming:

    1. Structural truncation guard: if there is meaningful content AFTER
       the last } (more than a few chars of whitespace/prose punctuation),
       the last } is an interior close brace, not the root close — the
       stream is still mid-structure. Return raw text so truncation repair
       sees the full incomplete stream.

    2. Volume guard: only trim if we keep at least 60% of the original —
       prevents clipping when the last } belongs to an inner turn.
    """
    start = text.find("{")
    last_close = text.rfind("}")

    if start == -1 or last_close == -1 or last_close <= start:
        return text

    after_close = text[last_close + 1:]

    # If there is meaningful content after the last } the stream is
    # mid-structure — don't extract, let truncation repair handle it
    # "Meaningful" = more than trailing whitespace or simple prose punctuation
    after_stripped = after_close.strip()
    if len(after_stripped) > 3 and not re.match(r'^[\w\s!?.,:;]*$', after_stripped):
        return text

    extracted = text[start:last_close + 1]
    if len(extracted) < len(text) * 0.6:
        return text
    return extracted

def _repair_truncated(text: str) -> Optional[dict]:
    """
    Try progressively more aggressive closings for truncated JSON.
    Ordered from least to most invasive — first successful parse wins.
    """
    suffixes = [
        ']}',        # cut after last complete turn
        '}]}',       # cut after closed string, still inside turn object
        '"]}',       # cut mid-string, object already closed
        '"}]}',      # cut mid-string inside a turn content field
        '..."}]}',   # same, with ellipsis to signal truncation
        '"}, {"role": "assistant", "content": "..."}]}',  # cut mid-user-turn
        ']}}',
        '}}',
    ]
    for suffix in suffixes:
        d = _try_parse(text + suffix)
        if d:
            return d
    return None

def _repair_single_quotes(text: str) -> Optional[dict]:
    """Swap unescaped single quotes to double quotes (model used Python dict syntax)."""
    if '": ' in text or '":' in text:
        return None  # already double-quoted, don't mangle
    swapped = text.replace("'", '"')
    return _try_parse(swapped)


def _repair_trailing_comma(text: str) -> Optional[dict]:
    """
    Remove trailing commas before ] or } — JavaScript habit, invalid JSON.
    Handles:
      [..., "last item",]
      {..., "key": "val",}
      also the truncation variant: content ends with ,"  before closing
    """
    # Remove trailing commas before closing bracket/brace
    fixed = re.sub(r",\s*([}\]])", r"\1", text)
    # Also handle trailing comma+quote patterns from truncated strings
    # e.g.  "content": "some text,"  }  →  "content": "some text"  }
    fixed = re.sub(r',"\s*([}\]])', r'"\1', fixed)
    if fixed != text:
        return _try_parse(fixed)
    return None


def _repair_mixed_escaping(text: str) -> Optional[dict]:
    """
    Fix partially-escaped JSON where some delimiters are \" and others are
    raw ".  This happens when the model starts with escaped quotes then
    switches style mid-output.

    Strategy: if straight parse fails and the text contains both \" and
    unescaped-quote patterns, try normalising all string boundaries to use
    unescaped quotes by decoding the \" sequences first, then re-encoding
    only the interior ones via the unescaped-inner-quotes repair.
    """
    if '\\"' not in text:
        return None   # no mixed escaping present

    # Decode all \" → " then re-run the unescaped inner quotes repair
    decoded = text.replace('\\"', '"')
    result  = _try_parse(decoded)
    if result:
        return result

    # Also try stripping the outer wrapper that sometimes appears when the
    # model double-encodes the whole payload as a JSON string
    if decoded.startswith('"') and decoded.endswith('"'):
        inner = decoded[1:-1]
        result = _try_parse(inner)
        if result:
            return result

    return _repair_unescaped_inner_quotes(decoded)


def _repair_missing_turn_close(text: str) -> Optional[dict]:
    """
    Fix turn objects missing their closing brace and/or comma, e.g.:
      {"role":"user","content":"text"}{"role":"assistant",...}  ← missing comma
      {"role":"user","content":"text"{"role":"assistant",...}   ← missing } and comma

    Two passes:
      1. Inject missing comma between adjacent turn objects: }{ → },{
      2. Inject missing } before turn openers where content isn't closed
    """
    # Pass 1: missing comma between objects (}{ → ,{)
    fixed = re.sub(r'\}\s*(\{"(?:role|content)")', r'},\1', text)

    # Pass 2: missing close brace before a new turn opener
    fixed = re.sub(
        r'("content"\s*:\s*"(?:[^"\\]|\\.)*")\s*,?\s*(\{"role")',
        r'\1},\2',
        fixed,
    )

    if fixed != text:
        r = _try_parse(fixed)
        if r:
            return r
        return _repair_truncated(fixed)
    return None


def _repair_unescaped_inner_quotes(text: str) -> Optional[dict]:
    """
    Fix: "content": "She said "hello" to him"
    Escapes unescaped double quotes found inside content/role string values.
    """
    fixed = re.sub(
        r'("(?:content|role)"\s*:\s*")(.*?)("(?:\s*[,}\]]))',
        lambda m: m.group(1) + m.group(2).replace('"', '\\"') + m.group(3),
        text,
        flags=re.DOTALL,
    )
    if fixed != text:
        return _try_parse(fixed)
    return None


def _repair_embedded_json_in_content(text: str) -> Optional[dict]:
    """
    Fix content strings that contain unescaped JSON objects, e.g.:
      "content": "Here is code: {"key": "value"} try it"

    The model embedded a JSON object literal inside a string value without
    escaping the inner braces/quotes.  We escape the inner quotes so the
    outer JSON parses correctly.

    This is a targeted version of _repair_unescaped_inner_quotes that also
    handles the brace characters by working on the raw text between the
    content field opener and the next structural comma/brace.
    """
    # Find content values and escape any unescaped { } inside them
    def _escape_content(m: re.Match) -> str:
        prefix  = m.group(1)   # "content": "
        body    = m.group(2)   # the inner text
        suffix  = m.group(3)   # closing "  followed by , or }
        # Escape inner double-quotes that aren't already escaped
        escaped = re.sub(r'(?<!\\)"', '\\"', body)
        return prefix + escaped + suffix

    fixed = re.sub(
        r'("content"\s*:\s*")(.*?)("(?:\s*[,}\]]))',
        _escape_content,
        text,
        flags=re.DOTALL,
    )
    if fixed != text:
        return _try_parse(fixed)
    return None

def _validate_and_fix_turns(turns: list) -> Optional[list]:
    """
    Given a list of raw turn dicts:
    1. Normalize role names
    2. Drop turns with missing/empty content
    3. Fix alternation — if two same-role turns in a row, merge them
    4. Ensure starts with user, ends with assistant
    5. Require at least 2 turns
    Returns cleaned list or None if unrecoverable.
    """
    if not isinstance(turns, list):
        return None

    # Normalize roles and filter empties
    cleaned = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        role    = _normalize_role(str(t.get("role", "")))
        content = str(t.get("content", "")).strip()
        if role not in ("user", "assistant"):
            continue
        if len(content) < 4:
            continue
        cleaned.append({"role": role, "content": content})

    if len(cleaned) < 2:
        return None

    # Ensure starts with user
    if cleaned[0]["role"] != "user":
        # Try dropping a leading assistant turn
        if len(cleaned) >= 3 and cleaned[1]["role"] == "user":
            cleaned = cleaned[1:]
        else:
            return None

    # Fix alternation — merge consecutive same-role turns
    merged = [cleaned[0]]
    for turn in cleaned[1:]:
        if turn["role"] == merged[-1]["role"]:
            # Same role back-to-back: merge content with a newline
            merged[-1]["content"] += "\n\n" + turn["content"]
        else:
            merged.append(turn)

    # Must end on assistant
    if merged[-1]["role"] == "user":
        if len(merged) >= 3:
            merged = merged[:-1]   # drop the trailing user turn
        else:
            return None

    # Must strictly alternate now
    for i, t in enumerate(merged):
        expected = "user" if i % 2 == 0 else "assistant"
        if t["role"] != expected:
            return None

    if len(merged) < 2:
        return None

    return merged


def _repair_fused_turns(text: str) -> Optional[dict]:
    """
    Fixes the most common small-model structural failure: two turns fused
    into one JSON object with duplicate keys, e.g.:
      {"role":"user","content":"...","role":"assistant","content":"..."}

    Uses object_pairs_hook to capture ALL key-value pairs before dedup.
    Inner objects arrive depth-first so all "role"-keyed entries are turns.
    """
    pairs_log: list = []

    def _hook(pairs: list) -> dict:
        pairs_log.append(pairs)
        return dict(pairs)

    try:
        root = json.loads(text, object_pairs_hook=_hook)
    except json.JSONDecodeError:
        return None

    if not pairs_log:
        return None

    fixed_turns: list = []
    for pairs in pairs_log:
        keys = [k for k, _ in pairs]
        if "role" not in keys:
            continue
        role_vals    = [v for k, v in pairs if k == "role"]
        content_vals = [v for k, v in pairs if k == "content"]
        if len(role_vals) >= 2 and len(content_vals) >= 2:
            for role, content in zip(role_vals, content_vals):
                fixed_turns.append({"role": role, "content": content})
        else:
            role    = role_vals[0]    if role_vals    else ""
            content = content_vals[0] if content_vals else ""
            fixed_turns.append({"role": role, "content": content})

    if len(fixed_turns) < 2:
        return None

    result = {k: v for k, v in root.items() if k != "turns"}
    result["turns"] = fixed_turns
    return result


def _unwrap_nested(data: dict) -> dict:
    """
    Unwrap {"conversation": {"turns": [...]}} or {"dialogue": {"turns": [...]}}
    one level if top-level has no "turns" key.
    """
    if "turns" in data:
        return data
    for val in data.values():
        if isinstance(val, dict) and "turns" in val:
            return val
        if isinstance(val, list):
            data["turns"] = val
            return data
    return data


def _repair_outer_wrapper(text: str) -> Optional[dict]:
    """
    Fix: { {"turns":[...]} }
    Some models wrap the whole payload in an extra set of braces.
    Uses bracket depth tracking to find the inner object correctly
    rather than rfind which grabs the wrong closing brace.
    """
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
            if depth == 2:
                start = i
        elif ch == "}":
            depth -= 1
            if depth == 1 and start is not None:
                inner = text[start:i + 1]
                d = _try_parse(inner)
                if d and "turns" in d:
                    return d
                start = None
    return None


def repair_and_parse(raw: str) -> Optional[dict]:
    """
    Full repair pipeline. Returns a dict with a valid "turns" list or None.

    Steps (applied in order, stops at first successful parse):
      1.  Strip markdown fences, extract outermost JSON object
      2.  Straight json.loads
      3.  Trailing comma removal  (JS habit: [...,] or {...,})
      4.  Single-quote coercion   (Python dict style)
      5.  Mixed escaping fix      (partial \" sequences)
      6.  Missing turn-close brace (with or without preceding comma)
      7.  Truncation repair       (hit max_tokens mid-output)
      8.  Unescaped inner quotes  (content contains raw ")
      9.  Embedded JSON in content (unescaped {} inside strings)
      10. Outer wrapper           ({ {"turns":[...]} })
      Post-parse:
      11. Fused-turn expansion    (duplicate role/content keys)
      12. Nested wrapper unwrap   ({"conversation":{"turns":[...]}})
      13. Turn validation + alternation fix
    """
    text = _FENCE.sub("", raw.strip())
    text = _extract_json_object(text)
    if not text:
        return None

    data = _try_parse(text)

    if data is None:
        data = _repair_trailing_comma(text)
        if data: log.debug("repair: trailing comma")

    if data is None:
        data = _repair_single_quotes(text)
        if data: log.debug("repair: single quotes")

    if data is None:
        data = _repair_mixed_escaping(text)
        if data: log.debug("repair: mixed escaping")

    if data is None:
        data = _repair_truncated(text)
        if data: log.debug("repair: truncated JSON")

    if data is None:
        data = _repair_missing_turn_close(text)
        if data: log.debug("repair: missing turn close brace")

    if data is None:
        data = _repair_unescaped_inner_quotes(text)
        if data: log.debug("repair: unescaped inner quotes")

    if data is None:
        data = _repair_embedded_json_in_content(text)
        if data: log.debug("repair: embedded JSON in content")

    if data is None:
        data = _repair_outer_wrapper(text)
        if data: log.debug("repair: outer wrapper")

    if data is None:
        return None

    # Fused turn expansion — always run, only adopt if it found more turns
    turns_before = data.get("turns", [])
    data_fused   = _repair_fused_turns(text)
    if data_fused is not None:
        fused = data_fused.get("turns", [])
        if isinstance(turns_before, list) and len(fused) > len(turns_before):
            data = data_fused
            log.debug(f"repair: fused turns ({len(turns_before)} → {len(fused)})")

    data  = _unwrap_nested(data)
    turns = _validate_and_fix_turns(data.get("turns"))
    if turns is None:
        return None

    data["turns"] = turns
    return data


def parse_conversation(raw: str, cat: dict) -> Optional[dict]:
    data = repair_and_parse(raw)
    if data is None:
        return None

    turns = data["turns"]   # already validated by repair_and_parse

    return {
        "category_id":  cat["id"],
        "category":     cat["category"],
        "subcategory":  cat["subcategory"],
        "topic":        cat["topic"],
        "style":        cat["style"],
        "turns":        turns,
        "turn_count":   len(turns),
        "generated_at": int(time.time()),
    }

# ─────────────────────────────────────────────────────────
#  MODEL ROTATOR
# ─────────────────────────────────────────────────────────

class ModelRotator:
    def __init__(self, model_ids: list):
        if not model_ids:
            console.print("[error]No --models specified.[/error]")
            sys.exit(1)
        self.model_ids   = model_ids
        self.idx         = 0
        self.instance_id: Optional[str] = None
        self.current_id  = ""

    def current(self) -> str:
        return self.current_id

    def load_initial(self) -> bool:
        return self._load(self.idx)

    def rotate(self) -> bool:
        if self.instance_id:
            console.print(f"[warn]Rotating — unloading {self.current_id}[/warn]")
            unload_model(self.instance_id)
            self.instance_id = None
            time.sleep(2)
        self.idx = (self.idx + 1) % len(self.model_ids)
        return self._load(self.idx)

    def _load(self, idx: int) -> bool:
        mid = self.model_ids[idx]
        self.current_id = mid
        iid = load_model(mid)
        if iid is None:
            return False
        self.instance_id = iid
        time.sleep(1)
        return True

    def cleanup(self) -> None:
        if self.instance_id:
            console.print(f"[warn]Unloading {self.current_id} on exit...[/warn]")
            unload_model(self.instance_id)
            self.instance_id = None

# ─────────────────────────────────────────────────────────
#  DISPLAY PANELS
# ─────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────
#  DISPLAY  — plain scrolling output, zero redraws, zero flicker
#
#  Design:
#    • Each conversation gets a section header printed once
#    • Thinking tokens overwrite a single line with \r  (you see
#      activity without flooding the screen)
#    • Output tokens stream directly to stdout, no buffering
#    • After each conversation a one-line summary is printed
#    • A summary bar prints every SUMMARY_EVERY conversations
#
#  Nothing is ever redrawn.  The terminal just scrolls naturally.
# ─────────────────────────────────────────────────────────

SUMMARY_EVERY = 10   # print a stats bar every N conversations
_TERM_WIDTH   = 72   # assumed terminal width for dividers

def _div(char: str = "─") -> str:
    return char * _TERM_WIDTH

def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"

def _print_section_header(
    stats: SessionStats,
    cat: dict,
    model: str,
    num_turns: int,
) -> None:
    """Print the divider + info line before each conversation starts."""
    n      = stats.generated + 1
    mshort = model.split("/")[-1] if "/" in model else model
    console.print(f"[dim]{_div()}[/dim]")
    console.print(
        f"[dim]#{n}[/dim]  "
        f"[cat]{_trunc(cat['subcategory'], 18)}  ›  {_trunc(cat['topic'], 32)}[/cat]"
        f"  [model]{_trunc(mshort, 20)}[/model]"
        f"  [dim]{num_turns}t[/dim]"
    )


def _print_summary_bar(stats: SessionStats, dedup: DedupStore) -> None:
    """Print a stats line every SUMMARY_EVERY conversations."""
    elapsed = time.strftime("%H:%M:%S", time.gmtime(stats.elapsed))
    tps     = f"{stats.tokens_per_sec:.0f} tok/s" if stats.tokens_per_sec else "—"
    console.print(
        f"\n[dim]{_div('═')}[/dim]\n"
        f"  [success]✓ {stats.generated:,}[/success] written  "
        f"[dedup]⊘ {stats.dedup_hit:,} dupes  {dedup.total():,} fp[/dedup]  "
        f"[warn]⚠ {stats.parse_fail}[/warn] parse  "
        f"[token]{tps}[/token]  "
        f"[dim]{elapsed}  {stats.conv_rate}[/dim]\n"
        f"[dim]{_div('═')}[/dim]\n"
    )


# ─── token writers ───────────────────────────────────────

# These write directly to stdout (bypassing Rich's buffering)
# so tokens appear the instant they arrive from the model.

_think_buf    = [""]   # mutable so nested fn can write to it
_in_think_display = [False]

def _write_think_token(text: str) -> None:
    """
    Thinking tokens overwrite a single line with \\r.
    The line shows a spinner char + last ~60 chars of thinking.
    No newline is printed, so it never scrolls.
    """
    _think_buf[0] = (_think_buf[0] + text)[-80:]
    display = _think_buf[0].replace("\n", " ")
    line    = f"\r\033[2K\033[33m🧠 {display}\033[0m"
    sys.stdout.write(line)
    sys.stdout.flush()


def _write_output_token(text: str, first_output: list) -> None:
    """
    Output tokens stream directly.  On the very first output token
    after thinking, print a newline to end the thinking line cleanly,
    then a dim prefix marker.
    """
    if first_output[0]:
        # End the thinking line and start output on a fresh line
        sys.stdout.write("\n\033[2m✍  \033[0m")
        first_output[0] = False
    sys.stdout.write(text)
    sys.stdout.flush()


def _end_stream_line() -> None:
    """Ensure we're on a fresh line after streaming ends."""
    sys.stdout.write("\n")
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────
#  PARSE FAIL LOGGER
# ─────────────────────────────────────────────────────────

def _log_parse_fail(output_dir: str, raw: str, cat: dict, attempt: int) -> None:
    """
    Append every failed raw response to parse_failures.log so patterns
    can be reviewed and new repairs added.

    Format:
    ════ #N  2024-01-01 12:00:00  attempt=1  category=rust_ownership ════
    <full raw response>
    ════ END ════
    """
    log_path = Path(output_dir) / "parse_failures.log"
    ts       = time.strftime("%Y-%m-%d %H:%M:%S")
    sep      = "═" * 72

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{sep}\n")
            f.write(f"  attempt={attempt}  cat={cat['id']}  {ts}\n")
            f.write(f"{sep}\n")
            f.write(raw)
            f.write(f"\n{sep} END {sep}\n")
    except Exception as e:
        log.warning(f"Could not write parse_failures.log: {e}")

    # Also print a short snippet to the terminal so you know it happened
    snippet = raw[:100].replace("\n", " ")
    console.print(f"[warn]  ✗ Parse fail (attempt {attempt}) — {snippet}…[/warn]")
    console.print(f"[dim]    Full output logged to {log_path}[/dim]")


# ─────────────────────────────────────────────────────────
#  GENERATE COMMAND
# ─────────────────────────────────────────────────────────

def generate(args: argparse.Namespace) -> None:
    cats   = load_categories(args.categories)
    pool   = build_weighted_pool(cats)
    outdir = args.output_dir
    Path(outdir).mkdir(parents=True, exist_ok=True)

    dedup = DedupStore(outdir)
    console.print("[info]Loading dedup fingerprints...[/info]")
    n_fp = dedup.load_all([c["id"] for c in cats])
    console.print(f"[info]  {n_fp:,} fingerprints loaded[/info]")

    already = total_existing(outdir, cats)
    remain  = args.count - already
    if remain <= 0:
        console.print(f"[success]Already at {already:,} conversations. Done.[/success]")
        return
    if already > 0:
        console.print(f"[info]Resuming — {already:,} done, {remain:,} to go[/info]")

    if not args.no_unload_existing:
        loaded = list_loaded_models()
        if loaded:
            console.print(f"[warn]Unloading {len(loaded)} model(s)...[/warn]")
            unload_all()

    rotator  = ModelRotator(args.models)
    variator = PromptVariator()
    cat_to_sc = {c["id"]: c["subcategory"] for c in cats}

    if not rotator.load_initial():
        for _ in range(len(args.models) - 1):
            if rotator.rotate():
                break
        else:
            console.print("[error]Could not load any model.[/error]")
            sys.exit(1)

    stats          = SessionStats(current_model=rotator.current())
    since_rotation = 0

    console.print(f"[dim]{_div('═')}[/dim]")
    console.print(f"  [success]SynthGen running[/success]  target=[bold]{args.count:,}[/bold]  models={len(args.models)}")
    console.print(f"[dim]{_div('═')}[/dim]\n")

    try:
        while stats.generated < remain and not _shutdown.is_set():

            # Model rotation
            if since_rotation >= args.rotate_every and len(args.models) > 1:
                since_rotation = 0
                if rotator.rotate():
                    stats.current_model = rotator.current()
                    console.print(f"\n[model]↺ Rotated to: {rotator.current()}[/model]\n")
                time.sleep(1)

            # Pick category
            cat       = random.choice(pool)
            num_turns = random.randint(cat["turns_min"], cat["turns_max"])
            if num_turns % 2 != 0:
                num_turns += 1

            stats.current_cat   = cat["subcategory"]
            stats.current_topic = cat["topic"]
            stats.think_tokens  = 0
            stats.stream_buf    = ""

            _print_section_header(stats, cat, rotator.current(), num_turns)

            prompt = variator.build(cat, num_turns)
            conv   = None

            # These are passed into stream_completion as the token callback.
            # Using mutable lists so the closure can share state cleanly.
            first_output = [True]   # flips False on first non-thinking token
            _think_buf[0] = ""

            def on_token_think(text: str) -> None:
                _write_think_token(text)
                stats.think_tokens += 1

            def on_token_out(text: str) -> None:
                _write_output_token(text, first_output)

            for attempt in range(MAX_RETRIES):
                if _shutdown.is_set():
                    break

                first_output[0] = True
                _think_buf[0]   = ""

                raw = stream_completion(
                    rotator.instance_id or rotator.current(),
                    prompt,
                    stats,
                    on_think=on_token_think,
                    on_output=on_token_out,
                )

                _end_stream_line()   # ensure newline after streaming

                if raw is None:
                    stats.api_fail += 1
                    wait = RETRY_DELAY * (attempt + 1)
                    console.print(f"[warn]  API fail ({attempt+1}/{MAX_RETRIES}) — retry in {wait}s[/warn]")
                    time.sleep(wait)
                    continue

                conv = parse_conversation(raw, cat)
                if conv is not None:
                    break

                stats.parse_fail += 1
                _log_parse_fail(outdir, raw, cat, attempt + 1)
                first_output[0] = True
                _think_buf[0]   = ""

            if conv is None:
                stats.skip += 1
                console.print("[warn]  ✗ Skipped after all retries[/warn]")
                continue

            if dedup.seen(cat["id"], conv):
                stats.dedup_hit += 1
                console.print("[dedup]  ⊘ Duplicate — regenerating[/dedup]")
                continue

            # Save
            append_conversation(outdir, cat["id"], conv)
            dedup.register(cat["id"], conv)
            stats.generated                       += 1
            since_rotation                        += 1
            stats.model_counts[rotator.current()] += 1

            # One-line completion summary
            tps = f"{stats.tokens_per_sec:.0f} tok/s" if stats.tokens_per_sec else "—"
            console.print(
                f"[success]  ✓[/success] [dim]saved  "
                f"{conv['turn_count']} turns  "
                f"{stats.total_tokens:,} total tokens  "
                f"{tps}[/dim]"
            )

            # Periodic stats bar
            if stats.generated % SUMMARY_EVERY == 0:
                _print_summary_bar(stats, dedup)

            if args.delay > 0:
                time.sleep(args.delay)

    finally:
        rotator.cleanup()

    final = total_existing(outdir, cats)
    _print_summary_bar(stats, dedup)
    console.print(f"[success]Done! {stats.generated:,} new  |  {final:,} total on disk[/success]")

# ─────────────────────────────────────────────────────────
#  STATS COMMAND
# ─────────────────────────────────────────────────────────

def stats_cmd(args: argparse.Namespace) -> None:
    cats   = load_categories(args.categories)
    outdir = args.output_dir
    dedup  = DedupStore(outdir)

    by_sc: dict = defaultdict(int)
    total = 0
    for c in cats:
        n = count_existing(outdir, c["id"])
        by_sc[c["subcategory"]] += n
        total += n

    total_fp = dedup.load_all([c["id"] for c in cats])

    tbl = Table(title=f"Dataset — {outdir}", box=box.ROUNDED,
                show_footer=True, header_style="bold cyan")
    tbl.add_column("Subcategory", footer="TOTAL",     style="cat",    width=28)
    tbl.add_column("Convs",       footer=f"{total:,}",justify="right",style="green")
    tbl.add_column("Share",       footer="",          justify="right",style="dim")

    for sc, cnt in sorted(by_sc.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100 if total else 0
        bar = "█" * int(pct / 3)
        tbl.add_row(sc, f"{cnt:,}", f"{pct:4.1f}%  {bar}")

    console.print(tbl)
    console.print(f"\n[dedup]Fingerprints on disk: {total_fp:,}[/dedup]")

    if getattr(args, "verbose", False):
        tbl2 = Table(title="Per-Category", box=box.SIMPLE)
        tbl2.add_column("ID",     style="dim",   width=28)
        tbl2.add_column("Topic",  style="cat",   width=32)
        tbl2.add_column("Convs",  justify="right", style="green")
        tbl2.add_column("Hashes", justify="right", style="dedup")
        for c in sorted(cats, key=lambda x: -count_existing(outdir, x["id"])):
            n = count_existing(outdir, c["id"])
            if n > 0 or getattr(args, "show_empty", False):
                tbl2.add_row(c["id"], c["topic"][:32], f"{n:,}", f"{dedup.count(c['id']):,}")
        console.print(tbl2)

# ─────────────────────────────────────────────────────────
#  SAMPLE COMMAND
# ─────────────────────────────────────────────────────────

def sample_cmd(args: argparse.Namespace) -> None:
    cats   = load_categories(args.categories)
    outdir = args.output_dir

    if args.category:
        matched = [c for c in cats
                   if c["id"] == args.category or c["subcategory"] == args.category]
        if not matched:
            console.print(f"[error]Category '{args.category}' not found.[/error]")
            return
        files = [output_path(outdir, c["id"]) for c in matched]
    else:
        files = [output_path(outdir, c["id"]) for c in cats]

    lines: list = []
    for f in files:
        if f.exists():
            with open(f, encoding="utf-8") as fh:
                lines.extend(ln for ln in fh if ln.strip())

    if not lines:
        console.print("[warn]No conversations found.[/warn]")
        return

    for i, s in enumerate(random.sample(lines, min(args.n, len(lines))), 1):
        d = json.loads(s)
        console.print(Panel(
            f"[bold]{d.get('subcategory')}[/bold]  →  [cat]{d.get('topic')}[/cat]",
            title=f"Sample {i}", border_style="cyan",
        ))
        for turn in d.get("turns", []):
            role  = turn["role"].upper()
            color = "green" if role == "ASSISTANT" else "blue"
            console.print(f"[bold {color}][{role}][/bold {color}]")
            console.print(turn["content"])
            console.print()

# ─────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        prog="synthgen",
        description="Synthetic conversation generator — streams from LM Studio",
    )
    sub = p.add_subparsers(dest="command")

    g = sub.add_parser("generate", help="Generate conversations")
    g.add_argument("--models",             nargs="+", required=True)
    g.add_argument("--count",              type=int,   default=10_000)
    g.add_argument("--rotate-every",       type=int,   default=150)
    g.add_argument("--output-dir",         type=str,   default="./output")
    g.add_argument("--categories",         type=str,   default="categories.jsonl")
    g.add_argument("--delay",              type=float, default=0.0)
    g.add_argument("--no-unload-existing", action="store_true")

    s = sub.add_parser("stats", help="Print dataset statistics")
    s.add_argument("--output-dir",  default="./output")
    s.add_argument("--categories",  default="categories.jsonl")
    s.add_argument("--verbose",     action="store_true")
    s.add_argument("--show-empty",  action="store_true")

    sp = sub.add_parser("sample", help="Print random sample conversations")
    sp.add_argument("--output-dir",  default="./output")
    sp.add_argument("--categories",  default="categories.jsonl")
    sp.add_argument("--category",    default=None)
    sp.add_argument("--n",           type=int, default=2)

    args = p.parse_args()

    if args.command == "generate":
        generate(args)
    elif args.command == "stats":
        stats_cmd(args)
    elif args.command == "sample":
        sample_cmd(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()