#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║       LectureGen  —  Synthetic Lecture Generator         ║
║   Streams · Rotates models · Deduplicates · Per-cat JSONL ║
╚══════════════════════════════════════════════════════════╝

Generates single-turn monologue lectures (not multi-turn conversations).
Each record is one authoritative, in-depth explanation by the assistant.

Usage (PowerShell — all one line):
    python generate_lecture.py generate --models "qwen3.5-4b@q3_k_m" "qwen3.5-4b@q3_k_m" --count 5000 --rotate-every 50 --output-dir ./convos/lectures

    python generate_lecture.py stats  --output-dir ./convos/lectures
    python generate_lecture.py sample --output-dir ./convos/lectures

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
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

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
log = logging.getLogger("lecturegen")

if sys.platform == "win32":
    import os; os.system("")

# ─────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────

LM_BASE      = "http://localhost:1234"
API_V1       = f"{LM_BASE}/api/v1"
API_OAI      = f"{LM_BASE}/v1"
TEMPERATURE  = 0.6     # slightly higher than convos — more voice variety
CPU_THREADS  = 8
MAX_TOKENS   = 8192
CTX_LENGTH   = 8192
REQ_TIMEOUT  = 360
LOAD_TIMEOUT = 120
MAX_RETRIES  = 3
RETRY_DELAY  = 6
MIN_CONTENT  = 150     # minimum chars for a usable lecture

# ─────────────────────────────────────────────────────────
#  GRACEFUL SHUTDOWN
# ─────────────────────────────────────────────────────────

_shutdown = Event()

def _handle_sigint(sig, frame):
    if not _shutdown.is_set():
        console.print("\n[warn]Ctrl+C — finishing current lecture then exiting...[/warn]")
        _shutdown.set()

signal.signal(signal.SIGINT,  _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)

# ─────────────────────────────────────────────────────────
#  SESSION STATS
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
    stream_buf:     str   = ""
    stream_phase:   str   = "idle"

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def conv_rate(self) -> str:
        r = self.generated / self.elapsed * 60 if self.elapsed > 0 else 0
        return f"{r:.1f}/min" if self.generated > 0 else "—"

# ─────────────────────────────────────────────────────────
#  DEDUPLICATION STORE  (keyed on topic + first 200 chars of content)
# ─────────────────────────────────────────────────────────

class DedupStore:
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
    def _fp(lecture: dict) -> str:
        topic   = lecture.get("topic", "")
        content = lecture.get("content", "")
        norm    = re.sub(r"\s+", " ", (topic + "|" + content[:200]).lower().strip())
        return hashlib.sha256(norm.encode()).hexdigest()

    def seen(self, cat_id: str, lecture: dict) -> bool:
        self._load(cat_id)
        return self._fp(lecture) in self._store[cat_id]

    def register(self, cat_id: str, lecture: dict) -> None:
        self._load(cat_id)
        fp = self._fp(lecture)
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

_LECTURE_STYLES = [
    "a dense, no-fluff technical lecture aimed at working engineers",
    "a thorough conceptual walkthrough — the kind found in an excellent programming book",
    "a practical tutorial that reasons through the concept using actual Rust types and functions",
    "a mental model explainer — build the intuition and explain the why behind the design decision",
    "a common mistakes and misconceptions guide — what goes wrong, why, and how to think correctly",
    "a deep dive covering edge cases, performance implications, and non-obvious interactions",
    "a live annotated walkthrough — narrate through the concept as if pairing with a developer",
    "a conference talk transcript — engaging, builds from first principles to a satisfying insight",
    "a rapid-fire FAQ — answer the five most important questions developers have about this topic",
    "a first-principles derivation — explain it from basics without assuming prior knowledge of the topic",
    "a 'before and after' explanation — show wrong thinking, then correct it with real examples",
    "a senior engineer's 'things I wish I knew earlier' reflection on this topic",
]

_DEPTH_ANGLES = [
    "Emphasize the why — don't just say what it is, explain the reasoning behind the design.",
    "Focus on the mental model. What's the right way to think about this to avoid confusion?",
    "Highlight common pitfalls and the subtle misconceptions that cause them.",
    "Cover performance and memory implications where relevant.",
    "Connect this to related Rust concepts the reader likely already knows.",
    "Use analogies where they genuinely help, but always ground them in actual Rust behavior.",
    "Cover both the happy path and the cases where things go wrong.",
    "Be opinionated — give concrete recommendations, not just neutral descriptions.",
    "Show the progression: naive approach → why it fails → correct approach.",
    "Explain what the compiler is actually doing, not just what the programmer must do.",
]

_TONE_MODS = [
    "Write as if the reader is smart but new to this specific topic.",
    "Assume the reader has hit a wall and needs the mental block removed.",
    "Be direct and authoritative — no hedging, no 'it depends' without explanation.",
    "Make it memorable — use a running example or analogy throughout.",
    "Include the kind of detail that only comes from real experience with this topic.",
    "Anticipate follow-up questions and answer them before they're asked.",
    "Treat the reader as a colleague, not a student.",
    "Go beyond the documentation — explain what the docs assume you already know.",
]

_CATEGORY_ANGLES: dict[str, list] = {
    "rust_ownership": [
        "moving a value into a function and why it can't be used afterward",
        "why you can't have two mutable references simultaneously — the aliasing XOR mutation rule",
        "how to think about ownership as responsibility for cleanup, not just access control",
        "the difference between moving and copying types — what makes a type Copy?",
        "how ownership interacts with struct fields and partial moves",
        "passing owned data into threads with move closures and why the compiler demands it",
        "why returning references from functions triggers lifetime errors",
        "how the drop order works and why it matters for RAII patterns",
    ],
    "rust_lifetimes": [
        "why the compiler needs explicit lifetime annotations — what it cannot infer",
        "lifetime elision rules: the three cases where annotations are implicit",
        "lifetimes in struct definitions — what it means to hold a reference",
        "the 'static lifetime — what it actually means and when to use it",
        "lifetime bounds on generics and trait implementations",
        "why you can't return a reference to a local variable",
        "the relationship between lifetimes, borrows, and stack frames",
        "named lifetimes and multiple lifetime parameters in function signatures",
    ],
    "rust_error_handling": [
        "the ? operator — how it expands and why it only works in Result-returning functions",
        "designing custom error types with thiserror — derive vs manual impl",
        "anyhow for applications vs thiserror for libraries — when to use which",
        "converting between error types with From and Into",
        "the difference between unwrap, expect, and ? — and when each is appropriate",
        "error handling patterns in async Rust — pitfalls and solutions",
        "wrapping third-party errors in your own type — the newtype pattern",
        "error context and chaining — adding meaning at each layer of the call stack",
    ],
    "rust_async": [
        "what a Future actually is — the state machine model under the hood",
        "why you need an async runtime like Tokio — what the executor does",
        "tokio::spawn vs async blocks — ownership, Send bounds, and when to use each",
        "sharing state between async tasks — Arc, Mutex, channels, and the tradeoffs",
        "select! for racing multiple futures — cancellation semantics and pitfalls",
        "async in traits — the challenges, object safety, and async-trait workaround",
        "structured cancellation — why dropping a future cancels it and what that means",
        "blocking code in async context — why it matters and spawn_blocking",
    ],
    "rust_traits": [
        "trait objects (dyn Trait) vs generics — monomorphization vs dynamic dispatch",
        "associated types vs generic type parameters — when each is the right choice",
        "implementing Display and Debug — the difference and when to derive vs implement",
        "the From and Into conversion traits — the blanket impl and how to use them",
        "operator overloading with std traits — Add, Index, Deref, and their semantics",
        "trait bounds in where clauses — readability and what they enable",
        "blanket implementations — how std uses them and how to write your own",
        "object safety rules — why not all traits can be used as dyn Trait",
    ],
    "rust_concurrency": [
        "why Arc<Mutex<T>> is the standard pattern — and when it's the wrong choice",
        "Mutex vs RwLock — the tradeoff and the pitfall of reader starvation",
        "channel-based message passing with mpsc — when to prefer it over shared state",
        "what Send and Sync actually mean — automatic implementation and unsafe impls",
        "deadlock scenarios — how they happen and structural ways to avoid them",
        "rayon for data parallelism — the work-stealing model and when to use it",
        "atomic types — the memory ordering model and which ordering to choose",
        "the actor model in Rust — channels, tasks, and why it scales",
    ],
    "rust_debugging": [
        "reading lifetime error messages — what the compiler is actually telling you",
        "cannot borrow as mutable because it's also borrowed as immutable — the mental model",
        "value used after move — how to spot it early and the patterns that avoid it",
        "type mismatch with generic constraints — reading the full error chain",
        "integer overflow in debug vs release — why the behavior differs",
        "shadowing inside loops — the subtle bugs it causes and how to catch them",
        "using dbg!, eprintln!, and tracing for progressive diagnosis",
        "cargo check vs cargo build — what each catches and when to use which",
    ],
    "algorithms_general": [
        "binary search — why the off-by-one is so persistent and how to think past it",
        "why quicksort is O(n log n) average but O(n²) worst — and why we still use it",
        "BFS vs DFS — the structural difference and which problems each solves",
        "dynamic programming — how to recognize a DP problem and choose top-down vs bottom-up",
        "hash map internals — load factor, collision handling, and why amortized O(1) holds",
        "graph shortest path — Dijkstra vs BFS vs Bellman-Ford and what each requires",
        "amortized analysis — the accounting method and why Vec::push is O(1) amortized",
    ],
    "systems_programming": [
        "stack vs heap — what the hardware is actually doing and why it matters for performance",
        "what a segfault is at the hardware level — page tables, virtual memory, and the trap",
        "virtual memory — address spaces, pages, and how the OS makes each process feel alone",
        "the difference between a process and a thread — shared vs private memory",
        "why context switching is expensive — what the CPU has to flush and reload",
        "cache lines and data layout — why struct field order can double your throughput",
        "what happens at a function call — stack frames, the call convention, and the ABI",
    ],
    "linux_cli": [
        "Unix pipes — how they work at the kernel level and why the interface is so composable",
        "file permissions — the nine bits, setuid, and why chmod 777 is almost always wrong",
        "grep, awk, and sed — when to use each and the one-liners worth memorizing",
        "process management — fork/exec model, signals, and what kill actually does",
        "environment variables — how they're inherited, scoped, and why they matter for CLI tools",
        "ssh port forwarding — local, remote, and dynamic forwarding with practical examples",
        "shell scripting pitfalls — word splitting, quoting, and why bash bites the unwary",
    ],
    "learning_programming": [
        "what a variable actually is — names, values, and memory locations",
        "functions vs methods — the difference and why OOP makes it confusing",
        "why loops exist — the connection between repetition and computation",
        "recursion — the base case, the recursive case, and the call stack",
        "why types matter — preventing an entire class of bugs at compile time",
        "what an API is — the contract between code that calls and code that implements",
        "debugging your first program — the mental model of reading error messages",
    ],
}

_GENERIC_ANGLES = [
    "the foundational concept that everything else builds on",
    "the most common mistake beginners make and why it happens",
    "the performance or memory implications that experts watch for",
    "a real-world use case that makes the motivation concrete",
    "the comparison between two approaches and when to choose each",
    "the idiomatic Rust way to do this — and why it's different from other languages",
    "the specific edge case or gotcha that trips up experienced developers",
    "the underlying mechanism that explains the surface-level behavior",
]


class LecturePromptVariator:
    def build(self, cat: dict) -> str:
        style  = random.choice(_LECTURE_STYLES)
        angles = _CATEGORY_ANGLES.get(cat["id"], _GENERIC_ANGLES)
        angle  = random.choice(angles)
        depth  = random.choice(_DEPTH_ANGLES)
        mods   = random.sample(_TONE_MODS, k=random.randint(1, 2))

        return (
            f"Write {style}.\n\n"
            f"Topic: {cat['topic']}\n"
            f"Description: {cat['description']}\n"
            f"Category guidance: {cat['system_hint']}\n\n"
            f"Specific focus angle: {angle}\n"
            f"Depth guidance: {depth}\n"
            f"Additional tone: {' '.join(mods)}\n\n"
            f"IMPORTANT: No multi-line code blocks or triple backticks ever. "
            f"Name specific Rust types, traits, crates, and functions naturally inline in sentences. "
            f"Be concrete and specific — vague explanations are useless for training. "
            f"Write a complete, self-contained explanation — not a stub or outline.\n\n"
            f"Output raw JSON only:\n"
            f'{{"category":"{cat["subcategory"]}","topic":"{cat["topic"]}","content":"...full lecture here..."}}'
        )

# ─────────────────────────────────────────────────────────
#  CATEGORY / OUTPUT HELPERS
# ─────────────────────────────────────────────────────────

def _normalize_category(cat: dict) -> dict:
    if "prompt_focus" not in cat:
        return cat

    raw_name = cat["category"]
    parts    = raw_name.split(" - ", 1)
    subcategory = parts[0].strip() if len(parts) == 2 else raw_name
    slug = re.sub(r"[^a-z0-9]+", "_", raw_name.lower()).strip("_")
    tags = cat.get("tags", [])
    system_hint = (
        f"Difficulty: {cat.get('difficulty', 'intermediate')}. "
        f"Focus: {cat['prompt_focus']}."
        + (f" Key concepts: {', '.join(tags)}." if tags else "")
    )
    return {
        "id":          slug,
        "category":    "Rust",
        "subcategory": subcategory,
        "topic":       raw_name,
        "description": cat["prompt_focus"],
        "system_hint": system_hint,
        "weight":      cat.get("weight", 1),
        "style":       "technical",
    }


def load_categories(path: str) -> list:
    cats = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cats.append(_normalize_category(json.loads(line)))
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


def append_lecture(output_dir: str, cat_id: str, lecture: dict) -> None:
    p = output_path(output_dir, cat_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(lecture, ensure_ascii=False) + "\n")

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
    "Produce authoritative, in-depth technical lectures for training a language model.\n\n"
    "RULES:\n"
    "- Output ONLY raw JSON. No markdown, no code fences, no preamble.\n"
    '- Schema: {"category":"...","topic":"...","content":"..."}\n'
    "- The content field is one complete, flowing lecture monologue.\n"
    "- NEVER include multi-line code blocks or triple backticks. No fenced code blocks ever.\n"
    "- You MAY naturally name Rust types, traits, crates, and identifiers inline in sentences "
    "(e.g. 'the Arc<Mutex<T>> pattern', 'impl Serialize', 'Vec::push') without formatting them.\n"
    "- This is a Rust lecture. Keep all discussion focused on Rust. Brief comparisons to other "
    "languages are acceptable but never the main explanation.\n"
    "- Be authoritative and concrete — vague hand-waving is useless for training.\n"
    "- Every lecture must feel genuinely unique — different angle, voice, or structure each time.\n"
    "- Write a complete, self-contained explanation. Do not produce an outline or stub.\n"
    "- Raw JSON ONLY. No ```json``` wrappers ever."
)

# ─────────────────────────────────────────────────────────
#  STREAMING CALL
# ─────────────────────────────────────────────────────────

_THINK_OPEN  = re.compile(r"<think>",  re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def stream_completion(
    instance_id: str,
    prompt: str,
    stats: SessionStats,
    on_think: Callable[[str], None],
    on_output: Callable[[str], None],
) -> Optional[str]:
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

    output:      list[str] = []
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

                reasoning = delta.get("reasoning_content")
                if reasoning:
                    _append_think(reasoning)
                    token_count += 1
                    continue

                content = delta.get("content")
                if not content:
                    continue

                token_count += 1

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
                        clean = _THINK_BLOCK.sub("", content)
                        _append_think(content)
                        if clean.strip():
                            _append_output(clean)
                    else:
                        in_think = True
                        _append_think(content)
                    continue

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
#  PARSE & REPAIR  (lecture-specific — no turn validation)
#
#  A lecture response has this structure:
#    {"category":"...","topic":"...","content":"...long text..."}
#
#  The content field is a single long string — the main failure modes are:
#    1. Literal newlines inside the content string
#    2. Truncation (content is long — hits max_tokens mid-string)
#    3. Unescaped inner double-quotes inside content
#    4. Mixed escaping (partial \" sequences)
#    5. Trailing commas / stray punctuation
# ─────────────────────────────────────────────────────────

_FENCE     = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_ROOT_KEYS = frozenset({"category", "topic", "content"})


def _try_parse(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_json_object(text: str) -> str:
    start      = text.find("{")
    last_close = text.rfind("}")
    if start == -1 or last_close == -1 or last_close <= start:
        return text
    after_stripped = text[last_close + 1:].strip()
    if len(after_stripped) > 3 and not re.match(r'^[\w\s!?.,:;]*$', after_stripped):
        return text
    extracted = text[start:last_close + 1]
    if len(extracted) < len(text) * 0.6:
        return text
    return extracted


def _close_json(text: str) -> str:
    """Compute the minimal suffix to close an incomplete JSON document."""
    stack:  list = []
    in_str = False
    i      = 0
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                stack.append("}")
            elif ch == "}":
                if stack and stack[-1] == "}":
                    stack.pop()
            elif ch == "[":
                stack.append("]")
            elif ch == "]":
                if stack and stack[-1] == "]":
                    stack.pop()
        i += 1
    close = '"' if in_str else ""
    for ch in reversed(stack):
        close += ch
    return close


def _repair_truncated(text: str) -> Optional[dict]:
    """Close an incomplete JSON document using the minimum suffix."""
    suffix = _close_json(text)
    if not suffix:
        return None
    d = _try_parse(text + suffix)
    if d:
        return d
    # Also try appending "..." to signal truncated content before closing
    d = _try_parse(text + "..." + suffix)
    return d


def _repair_literal_newlines(text: str) -> Optional[dict]:
    """Escape bare newlines inside JSON string values."""
    result:  list = []
    in_str   = False
    i        = 0
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == '\\':
                result.append(ch)
                i += 1
                if i < len(text):
                    result.append(text[i])
                    i += 1
                continue
            elif ch == '"':
                in_str = False
                result.append(ch)
            elif ch == '\n':
                result.append('\\n')
            elif ch == '\r':
                result.append('\\r')
            else:
                result.append(ch)
        else:
            if ch == '"':
                in_str = True
                result.append(ch)
            else:
                result.append(ch)
        i += 1
    fixed = ''.join(result)
    if fixed == text:
        return None
    return _try_parse(fixed) or _repair_truncated(fixed)


def _repair_unescaped_inner_quotes(text: str) -> Optional[dict]:
    """Escape unescaped double-quotes inside content/topic/category values."""
    def _normalize(m: re.Match) -> str:
        prefix = m.group(1)
        body   = m.group(2)
        suffix = m.group(3)
        body_clean   = body.replace('\\"', '"')
        body_escaped = body_clean.replace('"', '\\"')
        return prefix + body_escaped + suffix

    # Non-DOTALL pass (content has no literal newlines — i.e. after _repair_literal_newlines)
    fixed = re.sub(
        r'("(?:content|topic|category)"\s*:\s*")(.*)((?:\s*"(?:\s*[,}\]]))+)',
        _normalize, text,
    )
    if fixed != text:
        r = _try_parse(fixed)
        if r: return r
        r = _repair_truncated(fixed)
        if r: return r

    # DOTALL pass (content still contains literal newlines)
    fixed2 = re.sub(
        r'("(?:content|topic|category)"\s*:\s*")(.*)((?:\s*"(?:\s*[,}\]]))+)',
        _normalize, text, flags=re.DOTALL,
    )
    if fixed2 != text and fixed2 != fixed:
        r = _try_parse(fixed2)
        if r: return r
        return _repair_truncated(fixed2)

    return None


def _repair_mixed_escaping(text: str) -> Optional[dict]:
    if '\\"' not in text:
        return None
    decoded = text.replace('\\"', '"')
    r = _try_parse(decoded)
    if r: return r
    return _repair_unescaped_inner_quotes(decoded)


def _repair_trailing_comma(text: str) -> Optional[dict]:
    fixed = re.sub(r",\s*([}\]])", r"\1", text)
    fixed = re.sub(r',"\s*([}\]])', r'"\1', fixed)
    if fixed != text:
        return _try_parse(fixed)
    return None


def _repair_stray_punctuation(text: str) -> Optional[dict]:
    fixed = re.sub(r'"([.!?>])(\s*[,}\]])', r'"\2', text)
    if fixed != text:
        return _try_parse(fixed)
    return None


def _repair_invalid_escapes(text: str) -> Optional[dict]:
    fixed = re.sub(r'\\([^"\\\\/bfnrtu\n\r])', r'\1', text)
    if fixed != text:
        return _try_parse(fixed)
    return None


def _repair_content_section_quotes(text: str) -> Optional[dict]:
    """
    Fix content strings with spurious bare section-closing quotes and/or literal newlines.
    Scanner tracks string boundaries, treats " followed by structural delimiter as real close.
    """
    result:  list = []
    i        = 0
    n        = len(text)
    changed  = False
    OPENER   = re.compile(r'"(?:content|topic|category)"\s*:\s*"')
    IS_DELIM = re.compile(r'\s*[,}\]]')

    while i < n:
        m = OPENER.match(text, i)
        if m:
            result.append(text[i:m.end()])
            i = m.end()
            while i < n:
                ch = text[i]
                if ch == '\\':
                    result.append(ch); i += 1
                    if i < n: result.append(text[i]); i += 1
                    continue
                elif ch == '"':
                    if IS_DELIM.match(text, i + 1):
                        result.append(ch); i += 1; break
                    else:
                        result.append('\\"'); i += 1; changed = True
                elif ch == '\n':
                    result.append('\\n'); i += 1; changed = True
                elif ch == '\r':
                    result.append('\\r'); i += 1; changed = True
                else:
                    result.append(ch); i += 1
        else:
            result.append(text[i]); i += 1

    if not changed:
        return None
    fixed = ''.join(result)
    return _try_parse(fixed) or _repair_truncated(fixed)


def _repair_multi_pass(text: str) -> Optional[dict]:
    t = re.sub(r",(\s*[}\]])", r"\1", text)             # trailing commas
    t = re.sub(r'"([.!?>])(\s*[,}\]])', r'"\2', t)     # stray punctuation
    t = re.sub(r'(`{3,}(?:\w*)\s*)\\"', r'\1"', t)     # fence escaped quote
    t = re.sub(r'\\([^"\\\\/bfnrtu\n\r])', r'\1', t)   # invalid escapes
    t = re.sub(r'"""(\s*[,}\]])', r'"\1', t)            # triple-quote closer
    if t == text:
        return None
    return _try_parse(t) or _repair_unescaped_inner_quotes(t)


def repair_and_parse_lecture(raw: str) -> Optional[dict]:
    """
    Repair pipeline for lecture JSON responses.

    A lecture is: {"category":"...","topic":"...","content":"...long text..."}

    Steps (stops at first success):
      1.  Strip markdown fences, extract outermost JSON object
      2.  Straight json.loads
      3.  Literal newlines in strings (bare \\n inside JSON values)
      4.  Content section quotes + literal newlines (spurious " terminators)
      5.  Mixed escaping (partial \\" sequences)
      6.  Invalid escape sequences
      7.  Trailing comma removal
      8.  Stray sentence punctuation after string close
      9.  Multi-pass combined fixes
      10. Unescaped inner quotes in content
      11. Truncation (hit max_tokens mid-content)
    """
    text = _FENCE.sub("", raw.strip())
    text = _extract_json_object(text)
    if not text:
        return None

    data = _try_parse(text)

    if data is None:
        data = _repair_literal_newlines(text)

    if data is None:
        data = _repair_content_section_quotes(text)

    if data is None:
        data = _repair_mixed_escaping(text)

    if data is None:
        data = _repair_invalid_escapes(text)

    if data is None:
        data = _repair_trailing_comma(text)

    if data is None:
        data = _repair_stray_punctuation(text)

    if data is None:
        data = _repair_multi_pass(text)

    if data is None:
        data = _repair_unescaped_inner_quotes(text)

    if data is None:
        data = _repair_truncated(text)

    if data is None:
        return None

    # Strip unknown keys and validate
    data = {k: v for k, v in data.items() if k in _ROOT_KEYS}
    content = str(data.get("content", "")).strip()
    if len(content) < MIN_CONTENT:
        return None

    data["content"] = content
    return data


def parse_lecture(raw: str, cat: dict) -> Optional[dict]:
    data = repair_and_parse_lecture(raw)
    if data is None:
        return None
    return {
        "category_id":  cat["id"],
        "category":     cat["category"],
        "subcategory":  cat["subcategory"],
        "topic":        cat["topic"],
        "style":        "lecture",
        "content":      data["content"],
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
#  DISPLAY
# ─────────────────────────────────────────────────────────

SUMMARY_EVERY = 10
_TERM_WIDTH   = 72

def _div(char: str = "─") -> str:
    return char * _TERM_WIDTH

def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"

def _print_section_header(stats: SessionStats, cat: dict, model: str) -> None:
    n      = stats.generated + 1
    mshort = model.split("/")[-1] if "/" in model else model
    console.print(f"[dim]{_div()}[/dim]")
    console.print(
        f"[dim]#{n}[/dim]  "
        f"[cat]{_trunc(cat['subcategory'], 18)}  ›  {_trunc(cat['topic'], 36)}[/cat]"
        f"  [model]{_trunc(mshort, 20)}[/model]"
    )


def _print_summary_bar(stats: SessionStats, dedup: DedupStore) -> None:
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


_think_buf         = [""]
_in_think_display  = [False]

def _write_think_token(text: str) -> None:
    _think_buf[0] = (_think_buf[0] + text)[-80:]
    display = _think_buf[0].replace("\n", " ")
    sys.stdout.write(f"\r\033[2K\033[33m🧠 {display}\033[0m")
    sys.stdout.flush()

def _write_output_token(text: str, first_output: list) -> None:
    if first_output[0]:
        sys.stdout.write("\n\033[2m✍  \033[0m")
        first_output[0] = False
    sys.stdout.write(text)
    sys.stdout.flush()

def _end_stream_line() -> None:
    sys.stdout.write("\n")
    sys.stdout.flush()

# ─────────────────────────────────────────────────────────
#  PARSE FAIL LOGGER
# ─────────────────────────────────────────────────────────

def _log_parse_fail(raw: str, cat: dict, attempt: int) -> None:
    log_path = Path(__file__).parent / "lecture_parse_failures.log"
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
        log.warning(f"Could not write lecture_parse_failures.log: {e}")
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
        console.print(f"[success]Already at {already:,} lectures. Done.[/success]")
        return
    if already > 0:
        console.print(f"[info]Resuming — {already:,} done, {remain:,} to go[/info]")

    if not args.no_unload_existing:
        loaded = list_loaded_models()
        if loaded:
            console.print(f"[warn]Unloading {len(loaded)} model(s)...[/warn]")
            unload_all()

    rotator  = ModelRotator(args.models)
    variator = LecturePromptVariator()

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
    console.print(f"  [success]LectureGen running[/success]  target=[bold]{args.count:,}[/bold]  models={len(args.models)}")
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

            cat = random.choice(pool)
            stats.current_cat   = cat["subcategory"]
            stats.current_topic = cat["topic"]
            stats.think_tokens  = 0
            stats.stream_buf    = ""

            _print_section_header(stats, cat, rotator.current())

            prompt  = variator.build(cat)
            lecture = None

            first_output = [True]
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

                _end_stream_line()

                if raw is None:
                    stats.api_fail += 1
                    wait = RETRY_DELAY * (attempt + 1)
                    console.print(f"[warn]  API fail ({attempt+1}/{MAX_RETRIES}) — retry in {wait}s[/warn]")
                    time.sleep(wait)
                    continue

                lecture = parse_lecture(raw, cat)
                if lecture is not None:
                    break

                stats.parse_fail += 1
                _log_parse_fail(raw, cat, attempt + 1)
                first_output[0] = True
                _think_buf[0]   = ""

            if lecture is None:
                stats.skip += 1
                console.print("[warn]  ✗ Skipped after all retries[/warn]")
                continue

            if dedup.seen(cat["id"], lecture):
                stats.dedup_hit += 1
                console.print("[dedup]  ⊘ Duplicate — regenerating[/dedup]")
                continue

            append_lecture(outdir, cat["id"], lecture)
            dedup.register(cat["id"], lecture)
            stats.generated                       += 1
            since_rotation                        += 1
            stats.model_counts[rotator.current()] += 1

            chars = len(lecture["content"])
            tps   = f"{stats.tokens_per_sec:.0f} tok/s" if stats.tokens_per_sec else "—"
            console.print(
                f"[success]  ✓[/success] [dim]saved  "
                f"{chars:,} chars  "
                f"{stats.total_tokens:,} total tokens  "
                f"{tps}[/dim]"
            )

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

    tbl = Table(title=f"Lectures — {outdir}", box=box.ROUNDED,
                show_footer=True, header_style="bold cyan")
    tbl.add_column("Subcategory", footer="TOTAL",     style="cat",    width=28)
    tbl.add_column("Lectures",    footer=f"{total:,}", justify="right", style="green")
    tbl.add_column("Share",       footer="",           justify="right", style="dim")

    for sc, cnt in sorted(by_sc.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100 if total else 0
        bar = "█" * int(pct / 3)
        tbl.add_row(sc, f"{cnt:,}", f"{pct:4.1f}%  {bar}")

    console.print(tbl)
    console.print(f"\n[dedup]Fingerprints on disk: {total_fp:,}[/dedup]")

    if getattr(args, "verbose", False):
        tbl2 = Table(title="Per-Category", box=box.SIMPLE)
        tbl2.add_column("ID",      style="dim",    width=28)
        tbl2.add_column("Topic",   style="cat",    width=36)
        tbl2.add_column("Lectures", justify="right", style="green")
        for c in sorted(cats, key=lambda x: -count_existing(outdir, x["id"])):
            n = count_existing(outdir, c["id"])
            if n > 0 or getattr(args, "show_empty", False):
                tbl2.add_row(c["id"], c["topic"][:36], f"{n:,}")
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
        console.print("[warn]No lectures found.[/warn]")
        return

    for i, s in enumerate(random.sample(lines, min(args.n, len(lines))), 1):
        d = json.loads(s)
        console.print(Panel(
            f"[bold]{d.get('subcategory')}[/bold]  →  [cat]{d.get('topic')}[/cat]",
            title=f"Lecture {i}", border_style="cyan",
        ))
        console.print(d.get("content", ""))
        console.print()

# ─────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        prog="lecturegen",
        description="Synthetic lecture generator — streams from LM Studio",
    )
    sub = p.add_subparsers(dest="command")

    g = sub.add_parser("generate", help="Generate lectures")
    g.add_argument("--models",             nargs="+", required=True)
    g.add_argument("--count",              type=int,   default=5_000)
    g.add_argument("--rotate-every",       type=int,   default=150)
    g.add_argument("--output-dir",         type=str,   default="./convos/lectures")
    g.add_argument("--categories",         type=str,   default="rust_categories.jsonl")
    g.add_argument("--delay",              type=float, default=0.0)
    g.add_argument("--no-unload-existing", action="store_true")

    s = sub.add_parser("stats", help="Print dataset statistics")
    s.add_argument("--output-dir",  default="./convos/lectures")
    s.add_argument("--categories",  default="rust_categories.jsonl")
    s.add_argument("--verbose",     action="store_true")
    s.add_argument("--show-empty",  action="store_true")

    sp = sub.add_parser("sample", help="Print random sample lectures")
    sp.add_argument("--output-dir",  default="./convos/lectures")
    sp.add_argument("--categories",  default="rust_categories.jsonl")
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
