#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║   FrontierGen  —  Rust Code Example Generator                ║
║   Generates broken→fixed pairs, validates with cargo tools   ║
╚══════════════════════════════════════════════════════════════╝

Reads rust_categories.jsonl, prompts LM Studio for frontier-format
Rust examples, validates fixed_code with cargo check / clippy / fmt /
msrv, and writes verified examples to datasets/local/<category>.jsonl.

Usage:
    python generate_frontier.py --model "model-id"
    python generate_frontier.py --model "model-id" --count 3
    python generate_frontier.py --model "model-id" --category "Core - Move Semantics on Function Call"
    python generate_frontier.py --model "model-id" --cooldown-every 50 --cooldown-secs 60
    python generate_frontier.py --categories rust_categories.jsonl --output-dir datasets/local
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import requests
from rich.console import Console
from rich.theme import Theme

# ─────────────────────────────────────────────────────────
#  CONSOLE
# ─────────────────────────────────────────────────────────

console = Console(theme=Theme({
    "ok":      "bold green",
    "fail":    "bold red",
    "warn":    "bold yellow",
    "info":    "dim cyan",
    "cat":     "italic #5fafd7",
    "model":   "bold #7c5cbf",
    "dim":     "dim",
}), highlight=False)

if sys.platform == "win32":
    os.system("")  # enable ANSI on Windows

# ─────────────────────────────────────────────────────────
#  SHUTDOWN
# ─────────────────────────────────────────────────────────

_shutdown       = False
_active_iid: Optional[str] = None   # instance_id currently loaded by us


def _handle_sigint(_sig, _frame) -> None:
    global _shutdown
    if not _shutdown:
        _shutdown = True
        console.print("\n[warn]Ctrl+C — unloading model and exiting…[/warn]")
        if _active_iid:
            unload_model(_active_iid)
        sys.exit(0)


signal.signal(signal.SIGINT, _handle_sigint)

# ─────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────

LM_BASE      = "http://localhost:1234"
API_OAI      = f"{LM_BASE}/v1"
API_V1       = f"{LM_BASE}/api/v1"
TEMPERATURE  = 0.35
MAX_TOKENS   = 8192
REQ_TIMEOUT  = 240
CTX_LENGTH   = 8192
LOAD_TIMEOUT = 120
EDITION      = "2021"
LICENSE      = "Apache-2.0"

# ─────────────────────────────────────────────────────────
#  CRATE → Cargo.toml DEPENDENCY MAP
# ─────────────────────────────────────────────────────────

# Keys are the crate root names as they appear in `use` statements.
# Values are the full Cargo.toml dependency line.
CRATE_MAP: dict[str, str] = {
    "tokio":              'tokio = { version = "1", features = ["full"] }',
    "tokio_util":         'tokio-util = { version = "0.7", features = ["full"] }',
    "tokio_stream":       'tokio-stream = "0.1"',
    "futures":            'futures = "0.3"',
    "futures_util":       'futures-util = "0.3"',
    "async_trait":        'async-trait = "0.1"',
    "pin_project":        'pin-project = "1"',
    "pin_project_lite":   'pin-project-lite = "0.2"',
    "serde":              'serde = { version = "1", features = ["derive"] }',
    "serde_json":         'serde_json = "1"',
    "serde_derive":       'serde_derive = "1"',
    "thiserror":          'thiserror = "2"',
    "anyhow":             'anyhow = "1"',
    "rayon":              'rayon = "1"',
    "crossbeam":          'crossbeam = "0.8"',
    "crossbeam_channel":  'crossbeam-channel = "0.5"',
    "crossbeam_utils":    'crossbeam-utils = "0.8"',
    "parking_lot":        'parking_lot = "0.12"',
    "dashmap":            'dashmap = "6"',
    "once_cell":          'once_cell = "1"',
    "lazy_static":        'lazy_static = "1"',
    "rand":               'rand = "0.8"',
    "rand_core":          'rand_core = "0.6"',
    "regex":              'regex = "1"',
    "tracing":            'tracing = "0.1"',
    "tracing_subscriber": 'tracing-subscriber = { version = "0.3", features = ["env-filter"] }',
    "log":                'log = "0.4"',
    "env_logger":         'env_logger = "0.11"',
    "tower":              'tower = { version = "0.4", features = ["full"] }',
    "tower_service":      'tower = { version = "0.4", features = ["full"] }',
    "tower_layer":        'tower = { version = "0.4", features = ["full"] }',
    "hyper":              'hyper = { version = "1", features = ["full"] }',
    "axum":               'axum = "0.7"',
    "reqwest":            'reqwest = { version = "0.12", features = ["json"] }',
    "bytes":              'bytes = "1"',
    "byteorder":          'byteorder = "1"',
    "itertools":          'itertools = "0.13"',
    "num_traits":         'num-traits = "0.2"',
    "num_derive":         'num-derive = "0.4"',
    "indexmap":           'indexmap = "2"',
    "typed_arena":        'typed-arena = "2"',
    "derive_more":        'derive_more = { version = "1", features = ["full"] }',
    "strum":              'strum = { version = "0.26", features = ["derive"] }',
    "strum_macros":       'strum = { version = "0.26", features = ["derive"] }',
    "clap":               'clap = { version = "4", features = ["derive"] }',
    "chrono":             'chrono = "0.4"',
    "uuid":               'uuid = { version = "1", features = ["v4"] }',
    "libc":               'libc = "0.2"',
}

# ─────────────────────────────────────────────────────────
#  LLM PROMPTS
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a Rust expert generating training data. Output ONLY raw JSON — \
no markdown, no code fences, no preamble or explanation outside the JSON.

Generate exactly one JSON object:
{
  "category": "<category name as given>",
  "difficulty": "<beginner|intermediate|advanced>",
  "prompt": "<realistic developer question describing a bug/confusion — 1-3 sentences>",
  "broken_code": "<complete self-contained Rust source — fn main, use stmts, everything needed>",
  "error_message": "<exact compiler error text OR description of wrong runtime behaviour>",
  "error_code": "<compile-error|logic-bug|design-issue|runtime-panic>",
  "fixed_code": "<complete, idiomatic, working Rust source that fixes the issue>",
  "explanation": "<2-4 sentences: why it was wrong and what the fix does>",
  "concepts": ["<tag>", "..."],
  "crates": ["<crate-name>", "..."]
}

Rules:
- Both broken_code and fixed_code must be COMPLETE Rust files \
(fn main or #[tokio::main] async fn main, all use statements included).
- Use edition 2021.
- broken_code must illustrate a REAL, common mistake — not a trivial typo.
- fixed_code must be idiomatic and pass clippy without warnings.
- If no external crates are needed, set "crates" to [].
- Output ONLY the JSON object. Nothing else."""


def build_user_prompt(cat: dict) -> str:
    parts = [
        f'Category: "{cat["category"]}"',
        f'Focus: {cat.get("prompt_focus", cat.get("description", ""))}',
        f'Difficulty: {cat["difficulty"]}',
    ]
    tags = cat.get("tags", [])
    if tags:
        parts.append(f'Tags: {", ".join(tags)}')
    parts.append("\nGenerate ONE realistic broken → fixed example in the required JSON format.")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────
#  LM STUDIO API
# ─────────────────────────────────────────────────────────

def list_loaded_models() -> list[dict]:
    """Return list of currently loaded model records from LM Studio."""
    r = requests.get(f"{LM_BASE}/api/v0/models", timeout=10)
    r.raise_for_status()
    return [m for m in r.json().get("data", []) if m.get("state") == "loaded"]


def get_loaded_model() -> Optional[str]:
    """Return the first loaded model's identifier, or None."""
    try:
        loaded = list_loaded_models()
        return loaded[0]["id"] if loaded else None
    except Exception:
        return None


def load_model(model_id: str) -> Optional[str]:
    """
    Ask LM Studio to load *model_id*.  Returns the instance_id on success,
    None on failure.  Handles 409 Already Loaded gracefully.
    """
    try:
        r = requests.post(
            f"{API_V1}/models/load",
            json={
                "model":            model_id,
                "context_length":   CTX_LENGTH,
                "flash_attention":  True,
                "echo_load_config": True,
            },
            timeout=LOAD_TIMEOUT,
        )
        if r.status_code == 409:
            # Already loaded — fetch the instance_id from the running list
            loaded = list_loaded_models()
            for m in loaded:
                if m.get("id") == model_id or model_id in m.get("id", ""):
                    return m.get("instance_id") or m.get("id")
            return model_id  # best-effort fallback
        r.raise_for_status()
        data = r.json()
        return data.get("instance_id") or data.get("id") or model_id
    except Exception as e:
        console.print(f"[fail]load_model error: {e}[/fail]")
        return None


def unload_model(instance_id: str) -> bool:
    """Ask LM Studio to unload the model with the given instance_id."""
    try:
        r = requests.post(
            f"{API_V1}/models/unload",
            json={"instance_id": instance_id},
            timeout=30,
        )
        r.raise_for_status()
        return True
    except Exception as e:
        console.print(f"[warn]unload_model error: {e}[/warn]")
        return False


def call_llm(model_id: str, user_prompt: str) -> Optional[str]:
    """
    Stream a completion from LM Studio.  Returns the full response text
    with <think> blocks stripped, or None on error.

    Thinking tokens are shown live on a single overwriting line.
    Output tokens stream directly to stdout.
    """
    payload = {
        "model":       model_id,
        "messages":    [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": TEMPERATURE,
        "max_tokens":  MAX_TOKENS,
        "stream":      True,
    }
    output: list[str] = []
    think_buf = ""
    first_out = True   # flips False on first real output token
    in_think  = False  # True while inside a <think>...</think> span

    def _show_think(tok: str) -> None:
        nonlocal think_buf
        think_buf = (think_buf + tok)[-80:]
        display   = think_buf.replace("\n", " ")
        sys.stdout.write(f"\r\033[2K\033[33m  🧠 {display}\033[0m")
        sys.stdout.flush()

    def _emit_output(tok: str) -> None:
        nonlocal first_out
        if first_out:
            sys.stdout.write("\n\033[2m  ✍  \033[0m")
            first_out = False
        output.append(tok)
        sys.stdout.write(tok)
        sys.stdout.flush()

    try:
        with requests.post(
            f"{API_OAI}/chat/completions",
            json=payload,
            stream=True,
            timeout=REQ_TIMEOUT,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
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

                # ── Pattern 1: explicit reasoning_content field ───────
                reasoning = delta.get("reasoning_content") or ""
                if reasoning:
                    _show_think(reasoning)
                    continue

                content = delta.get("content") or ""
                if not content:
                    continue

                # ── Pattern 2: <think> tags spanning chunks ───────────
                if in_think:
                    if _THINK_CLOSE.search(content):
                        parts    = _THINK_CLOSE.split(content, maxsplit=1)
                        in_think = False
                        _show_think(parts[0])
                        if len(parts) > 1 and parts[1]:
                            _emit_output(parts[1])
                    else:
                        _show_think(content)
                    continue

                if _THINK_OPEN.search(content):
                    if _THINK_CLOSE.search(content):
                        # Self-contained <think>…</think> in one chunk
                        clean = _THINK_BLOCK.sub("", content)
                        _show_think(content)
                        if clean.strip():
                            _emit_output(clean)
                    else:
                        in_think = True
                        _show_think(content)
                    continue

                # ── Normal output token ───────────────────────────────
                _emit_output(content)

        sys.stdout.write("\n")
        sys.stdout.flush()

        text = "".join(output)
        # Final safety strip for any think blocks that slipped through
        text = _THINK_BLOCK.sub("", text).strip()
        return text if text else None

    except requests.exceptions.ConnectionError:
        sys.stdout.write("\n")
        console.print("[fail]Cannot connect to LM Studio on port 1234.[/fail]")
        return None
    except Exception as e:
        sys.stdout.write("\n")
        console.print(f"[fail]API error: {e}[/fail]")
        return None


# ─────────────────────────────────────────────────────────
#  JSON PARSING
# ─────────────────────────────────────────────────────────

_THINK_OPEN  = re.compile(r"<think>",  re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

REQUIRED_FIELDS = {"category", "difficulty", "prompt", "broken_code",
                   "error_message", "error_code", "fixed_code", "explanation"}


# ── JSON repair helpers ───────────────────────────────────────────────────────

def _try_parse(text: str) -> Optional[dict]:
    try:
        d = json.loads(text)
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None


def _apply_escapes(text: str) -> tuple[str, bool]:
    """
    Walk JSON character by character and fix common string-value issues:
      • literal \\n / \\r / \\t → proper escape sequences
      • \\\\\" (double-backslash + quote) → \\\" (escaped quote)
    Returns (fixed_text, changed).
    """
    out: list[str] = []
    i = 0
    n = len(text)
    changed  = False
    in_str   = False

    while i < n:
        ch = text[i]
        if not in_str:
            out.append(ch)
            if ch == '"':
                in_str = True
            i += 1
            continue
        # ── inside a string value ────────────────────────────────────
        if ch == '\\':
            nxt = text[i + 1] if i + 1 < n else ''
            if nxt == '\\' and i + 2 < n and text[i + 2] == '"':
                # \\" → \" (double-backslash before quote → escaped quote)
                out.append('\\"')
                i += 3
                changed = True
            else:
                out.append(ch); i += 1
                if i < n:
                    out.append(text[i]); i += 1
        elif ch == '"':
            out.append(ch); in_str = False; i += 1
        elif ch == '\n':
            out.append('\\n'); changed = True; i += 1
        elif ch == '\r':
            out.append('\\r'); changed = True; i += 1
        elif ch == '\t':
            out.append('\\t'); changed = True; i += 1
        else:
            out.append(ch); i += 1

    return ''.join(out), changed


def _repair_escapes(text: str) -> Optional[dict]:
    fixed, changed = _apply_escapes(text)
    return _try_parse(fixed) if changed else None


def _repair_trailing_commas(text: str) -> Optional[dict]:
    fixed = re.sub(r",\s*([}\]])", r"\1", text)
    return _try_parse(fixed) if fixed != text else None


def _repair_truncated(text: str) -> Optional[dict]:
    """
    Close any unclosed string / array / object at the end of a truncated
    response so json.loads can still recover a valid object.
    """
    stack: list[str] = []
    in_str = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '\\' and in_str:
            i += 2; continue
        if ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch in '{[':
                stack.append(ch)
            elif ch == '}' and stack and stack[-1] == '{':
                stack.pop()
            elif ch == ']' and stack and stack[-1] == '[':
                stack.pop()
        i += 1

    if not stack and not in_str:
        return None  # nothing to close

    suffix = ('"' if in_str else '') + ''.join(
        '}' if c == '{' else ']' for c in reversed(stack)
    )
    return _try_parse(text + suffix)


def _repair_and_parse(text: str) -> Optional[dict]:
    """Try a chain of lightweight repairs before giving up."""
    if (d := _try_parse(text)):               return d
    if (d := _repair_trailing_commas(text)):  return d
    if (d := _repair_escapes(text)):          return d
    # trailing commas + escapes combined
    tc = re.sub(r",\s*([}\]])", r"\1", text)
    fixed, changed = _apply_escapes(tc)
    if changed and (d := _try_parse(fixed)):  return d
    # truncation (try on raw and escape-fixed variants)
    if (d := _repair_truncated(text)):        return d
    esc, changed = _apply_escapes(text)
    if changed and (d := _repair_truncated(esc)): return d
    return None


# ─────────────────────────────────────────────────────────────────────────────

def parse_example(text: str) -> Optional[dict]:
    """
    Parse LLM output as a frontier JSON example.
    Returns the dict or None if parsing or validation fails.
    """
    # Strip markdown fences if present
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()

    # Find outermost { }
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        # Might be truncated before the closing brace
        start = text.find("{")
        if start == -1:
            return None
        text = text[start:]
    else:
        text = text[start:end + 1]

    data = _repair_and_parse(text)

    if not isinstance(data, dict):
        return None

    missing = REQUIRED_FIELDS - data.keys()
    if missing:
        return None

    # Normalise list fields
    if isinstance(data.get("concepts"), str):
        data["concepts"] = [t.strip() for t in data["concepts"].split(",") if t.strip()]
    if isinstance(data.get("crates"), str):
        data["crates"] = [c.strip() for c in data["crates"].split(",") if c.strip()]
    if not isinstance(data.get("concepts"), list):
        data["concepts"] = []
    if not isinstance(data.get("crates"), list):
        data["crates"] = []

    return data


# ─────────────────────────────────────────────────────────
#  CRATE DETECTION
# ─────────────────────────────────────────────────────────

_STD_CRATES = frozenset({"std", "core", "alloc", "proc_macro", "test"})


def detect_crates(code: str) -> list[str]:
    """
    Extract external crate root names from use / extern crate statements.
    Returns only names present in CRATE_MAP.
    """
    found: set[str] = set()
    for m in re.finditer(r"^use\s+([a-zA-Z_][a-zA-Z0-9_]*)", code, re.MULTILINE):
        found.add(m.group(1))
    for m in re.finditer(r"^extern\s+crate\s+([a-zA-Z_][a-zA-Z0-9_]*)", code, re.MULTILINE):
        found.add(m.group(1))
    # Also catch attribute-style crates: #[tokio::main]
    for m in re.finditer(r"#\[([a-zA-Z_][a-zA-Z0-9_]*)::", code):
        found.add(m.group(1))
    # derive macros that pull in crates
    for m in re.finditer(r"derive\([^)]*\b(Serialize|Deserialize)\b", code):
        found.add("serde")
    found -= _STD_CRATES
    return sorted(c for c in found if c in CRATE_MAP)


def make_cargo_toml(crates: list[str]) -> str:
    lines = [
        '[package]',
        'name = "rust-example"',
        'version = "0.1.0"',
        f'edition = "{EDITION}"',
        '',
        '[dependencies]',
    ]
    for crate in sorted(set(crates)):
        if crate in CRATE_MAP:
            lines.append(CRATE_MAP[crate])
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────
#  CODE WRAPPING
# ─────────────────────────────────────────────────────────

_HAS_MAIN    = re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+main\s*\(", re.MULTILINE)
_ASYNC_MAIN  = re.compile(r"^\s*(?:pub\s+)?async\s+fn\s+main\s*\(", re.MULTILINE)
_TOKIO_ATTR  = re.compile(r"#\[tokio::main\]")
_TOP_LEVEL   = re.compile(
    r"^(?:pub\s+)?(?:async\s+)?(?:fn|struct|enum|trait|type|impl|const|static|mod|use|extern)\s",
    re.MULTILINE,
)


_PYTHON_COMMENT = re.compile(r"^#(?!\[)\s.*$", re.MULTILINE)


def wrap_for_check(code: str) -> str:
    """
    Ensure code can be compiled as a binary crate by cargo check.

    • Has fn main (sync or async) → return as-is (add #[tokio::main] if async
      main exists but attribute is missing).
    • Has top-level definitions but no main → append an empty fn main().
    • Looks like bare statements → wrap in fn main() { ... }.
    """
    # Strip Python-style comment lines the model occasionally emits (e.g. "# use ...")
    # but preserve valid Rust attributes which start with "#[".
    code = _PYTHON_COMMENT.sub("", code)

    has_main   = bool(_HAS_MAIN.search(code))
    async_main = bool(_ASYNC_MAIN.search(code))
    tokio_attr = bool(_TOKIO_ATTR.search(code))
    uses_tokio = "tokio" in code

    if has_main:
        if async_main and not tokio_attr:
            # Insert the attribute immediately before `async fn main`, not at
            # the top of the file (which would attach it to the wrong item).
            return _ASYNC_MAIN.sub(
                lambda m: "#[tokio::main]\n" + m.group(0),
                code,
                count=1,
            )
        return code

    has_top = bool(_TOP_LEVEL.search(code))
    if has_top:
        if uses_tokio:
            return code + "\n#[tokio::main]\nasync fn main() {}\n"
        return code + "\nfn main() {}\n"
    else:
        if uses_tokio:
            return f"#[tokio::main]\nasync fn main() {{\n{code}\n}}\n"
        return f"fn main() {{\n{code}\n}}\n"


# ─────────────────────────────────────────────────────────
#  CARGO VALIDATION
# ─────────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _detect_msrv(code: str) -> str:
    """
    Return a minimum Rust version string based on feature patterns in the code.
    Edition 2021 requires 1.56 as the baseline.
    """
    # (major, minor, patch) tuples — take the max across all detected features
    msrv = (1, 56, 0)

    checks = [
        # let-else: 1.65
        (r"\blet\b[^=\n]+=[^=\n]+\belse\b\s*\{", (1, 65, 0)),
        # Generic Associated Types (GATs): 1.65
        (r"\btype\s+\w+\s*<[^>]*'[a-z]", (1, 65, 0)),
        # std::backtrace: 1.65
        (r"\bbacktrace\b", (1, 65, 0)),
        # impl Trait in associated type position: 1.75
        (r"type\s+\w+\s*=\s*impl\b", (1, 75, 0)),
        # async closures (async || {}): 1.85
        (r"\basync\s*\|", (1, 85, 0)),
        # Return-position impl Trait in traits (RPITIT): 1.75
        (r"fn\s+\w+[^{;]*->\s*impl\s+\w+", (1, 75, 0)),
        # LazyCell / LazyLock: 1.80
        (r"\bLazyCell\b|\bLazyLock\b", (1, 80, 0)),
    ]
    for pattern, version in checks:
        if re.search(pattern, code):
            msrv = max(msrv, version)

    return f"{msrv[0]}.{msrv[1]}.{msrv[2]}"


def validate(fixed_code: str, declared_crates: list[str]) -> dict:
    """
    Create a temp cargo project, run check / clippy / fmt / msrv.

    Returns a result dict:
      {
        "build":        bool,
        "clippy":       bool,
        "fmt":          bool,
        "msrv":         bool,
        "clippy_lints": [str],   # non-allow warnings
        "min_rust_version": str, # e.g. "1.56.0"
        "compiler_ver": str,     # e.g. "rustc 1.93.1"
        "has_unsafe":   bool,
        "wrapped_code": str,     # the code actually compiled
        "errors":       [str],   # cargo check stderr if build failed
      }
    """
    result = {
        "build": False, "clippy": False, "fmt": False, "msrv": False,
        "clippy_lints": [], "clippy_output": "",
        "min_rust_version": "1.56.0",
        "compiler_ver": "", "has_unsafe": False,
        "wrapped_code": fixed_code, "errors": [],
    }

    # Detect compiler version once
    try:
        rv = subprocess.run(["rustc", "--version"], capture_output=True, text=True, timeout=10)
        result["compiler_ver"] = rv.stdout.strip()
    except Exception:
        pass

    wrapped = wrap_for_check(fixed_code)
    # Detect crates from the wrapped code so anything injected by wrap_for_check
    # (e.g. #[tokio::main]) is included in the dependency list.
    detected = detect_crates(wrapped)
    all_crates = sorted(set(declared_crates + detected))
    result["wrapped_code"]  = wrapped
    result["has_unsafe"]    = bool(re.search(r"\bunsafe\b", wrapped))

    with tempfile.TemporaryDirectory(prefix="frontier_") as tmp:
        proj = Path(tmp) / "rust-example"
        proj.mkdir()
        (proj / "Cargo.toml").write_text(make_cargo_toml(all_crates), encoding="utf-8")
        src = proj / "src"
        src.mkdir()
        (src / "main.rs").write_text(wrapped, encoding="utf-8")

        # ── cargo check ──────────────────────────────────────────────
        r = _run(["cargo", "check", "--quiet", "--color=never"], proj)
        if r.returncode != 0:
            result["errors"] = [r.stderr.strip()]
            return result  # no point continuing
        result["build"] = True

        # ── cargo fmt --check ────────────────────────────────────────
        r = _run(["cargo", "fmt", "--", "--check"], proj)
        result["fmt"] = (r.returncode == 0)

        # ── cargo clippy ─────────────────────────────────────────────
        r = _run(
            ["cargo", "clippy", "--quiet", "--color=never",
             "--", "-D", "warnings"],
            proj,
        )
        result["clippy"] = (r.returncode == 0)
        # Collect all lint names from clippy/rustc output.
        # Matches both [clippy::lint_name] and plain rustc lints like [unused_mut].
        if not result["clippy"]:
            lints: list[str] = []
            for line in r.stderr.splitlines():
                m = re.search(r"\[([a-z_:]+)\]", line)
                if m and m.group(1) not in ("E", "W"):
                    lints.append(m.group(1))
            result["clippy_lints"]  = sorted(set(lints))
            result["clippy_output"] = r.stderr.strip()

        # ── MSRV: pattern-based detection (edition 2021 baseline = 1.56) ──
        result["min_rust_version"] = _detect_msrv(wrapped)
        result["msrv"] = True

    return result


# ─────────────────────────────────────────────────────────
#  OUTPUT
# ─────────────────────────────────────────────────────────

_SCRIPT_DIR     = Path(__file__).resolve().parent
_CARGO_FAIL_LOG  = _SCRIPT_DIR / "frontier_cargo_failures.log"
_PARSE_FAIL_LOG  = _SCRIPT_DIR / "frontier_parse_failures.log"
_CLIPPY_FAIL_LOG = _SCRIPT_DIR / "frontier_clippy_failures.log"
_SEP = "═" * 72


def log_cargo_failure(
    cat_name: str,
    attempt: int,
    model: str,
    raw: str,
    fixed_code: str,
    wrapped_code: str,
    errors: list[str],
) -> None:
    """Append a cargo check failure record to frontier_cargo_failures.log."""
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(_CARGO_FAIL_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n{_SEP}\n")
        f.write(f"  {ts}  cat={cat_name!r}  attempt={attempt}  model={model}\n")
        f.write(f"{_SEP}\n")
        f.write("── raw LLM output ──\n")
        f.write(raw if raw else "<empty>")
        if raw and not raw.endswith("\n"):
            f.write("\n")
        f.write("── fixed_code (as submitted) ──\n")
        f.write(fixed_code)
        if not fixed_code.endswith("\n"):
            f.write("\n")
        f.write("── wrapped_code (as compiled) ──\n")
        f.write(wrapped_code)
        if not wrapped_code.endswith("\n"):
            f.write("\n")
        f.write("── cargo check output ──\n")
        for err in errors:
            f.write(err)
            if not err.endswith("\n"):
                f.write("\n")
        f.write(f"{_SEP} END {_SEP}\n")


def log_clippy_failure(
    cat_name: str, attempt: int, model: str, lints: list[str], fixed_code: str
) -> None:
    """Append a clippy warning record to frontier_clippy_failures.log."""
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(_CLIPPY_FAIL_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n{_SEP}\n")
        f.write(f"  {ts}  cat={cat_name!r}  attempt={attempt}  model={model}\n")
        f.write(f"  lints: {', '.join(lints) if lints else 'unspecified'}\n")
        f.write(f"{_SEP}\n")
        f.write(fixed_code)
        if not fixed_code.endswith("\n"):
            f.write("\n")
        f.write(f"{_SEP} END {_SEP}\n")


def log_parse_failure(cat_name: str, attempt: int, model: str, raw: str) -> None:
    """Append a JSON parse failure record to frontier_parse_failures.log."""
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(_PARSE_FAIL_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n{_SEP}\n")
        f.write(f"  {ts}  cat={cat_name!r}  attempt={attempt}  model={model}\n")
        f.write(f"{_SEP}\n")
        f.write(raw if raw else "<empty response>")
        if raw and not raw.endswith("\n"):
            f.write("\n")
        f.write(f"{_SEP} END {_SEP}\n")


def safe_filename(name: str) -> str:
    """Sanitise a category name for use as a filename on all platforms."""
    # Remove characters invalid on Windows
    invalid = r'\/:*?"<>|'
    for ch in invalid:
        name = name.replace(ch, "")
    return name.strip()


def output_path(output_dir: Path, category: str) -> Path:
    return output_dir / f"{safe_filename(category)}.jsonl"


def write_example(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────
#  CATEGORIES
# ─────────────────────────────────────────────────────────

def load_categories(path: Path) -> list[dict]:
    cats = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cats.append(json.loads(line))
    return cats


# ─────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    cats_path  = Path(args.categories)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = load_categories(cats_path)
    if args.category:
        categories = [c for c in categories if c["category"] == args.category]
        if not categories:
            console.print(f"[fail]Category not found: {args.category!r}[/fail]")
            sys.exit(1)

    # ── Model loading ────────────────────────────────────────────────
    global _active_iid

    # Unload anything already in memory before we start
    try:
        already_loaded = list_loaded_models()
    except Exception:
        already_loaded = []
    if already_loaded:
        console.print(f"[warn]Unloading {len(already_loaded)} model(s) already in memory…[/warn]")
        for m in already_loaded:
            unload_model(m.get("instance_id") or m.get("id", ""))

    if args.model:
        console.print(f"[model]Loading model: {args.model}[/model]")
        instance_id = load_model(args.model)
        if instance_id is None:
            console.print("[fail]Failed to load model.[/fail]")
            sys.exit(1)
        _active_iid = instance_id
        model = args.model
    else:
        model = get_loaded_model()
        if not model:
            console.print("[fail]No model loaded in LM Studio (or use --model to specify one).[/fail]")
            sys.exit(1)
        instance_id = model
        _active_iid = instance_id
    console.print(f"[model]Model: {model}[/model]")
    console.print(f"[info]Output: {output_dir.resolve()}[/info]")
    console.print(
        f"[info]Cooldown: every {args.cooldown_every} generations, "
        f"pause {args.cooldown_secs}s[/info]\n"
    )

    total_ok   = 0
    total_fail = 0
    cooldown_every = args.cooldown_every
    cooldown_secs  = args.cooldown_secs

    for cat in categories:
        cat_name   = cat["category"]
        target     = cat.get("target_entries", args.count)
        count      = args.count if args.count is not None else target
        out_path   = output_path(output_dir, cat_name)

        # Count already-existing entries for this category
        existing = 0
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                existing = sum(1 for line in f if line.strip())
        remaining = max(0, count - existing)

        if remaining == 0:
            console.print(f"[dim]Skipping {cat_name!r} (already has {existing}/{count})[/dim]")
            continue

        console.print(f"[cat]{cat_name}[/cat]  [dim]({existing} existing, {remaining} to generate)[/dim]")

        generated_this_cat = 0
        attempt = 0

        while generated_this_cat < remaining:
            attempt += 1
            if attempt > remaining * 4:
                console.print(f"  [warn]Too many attempts for {cat_name!r}, moving on.[/warn]")
                break

            console.print(f"  [dim]attempt {attempt}…[/dim]")

            # ── LLM call ─────────────────────────────────────────────
            user_prompt = build_user_prompt(cat)
            raw = call_llm(model, user_prompt)
            if raw is None:
                total_fail += 1
                continue

            # ── JSON parse ────────────────────────────────────────────
            example = parse_example(raw)
            if example is None:
                console.print(f"  [warn]JSON parse failed (attempt {attempt})[/warn]")
                log_parse_failure(cat_name, attempt, model, raw)
                total_fail += 1
                continue

            fixed_code = _PYTHON_COMMENT.sub("", example.get("fixed_code", "")).strip()
            if not fixed_code:
                console.print(f"  [warn]Empty fixed_code (attempt {attempt})[/warn]")
                total_fail += 1
                continue

            # ── Cargo validation ──────────────────────────────────────
            console.print(f"  [dim]Validating with cargo…[/dim]")
            val = validate(fixed_code, example.get("crates", []))

            if not val["build"]:
                err_preview = val["errors"][0][:120] if val["errors"] else "unknown"
                console.print(
                    f"  [fail]✗ cargo check failed (attempt {attempt})[/fail]  "
                    f"[dim]{err_preview}[/dim]"
                )
                log_cargo_failure(
                    cat_name, attempt, model,
                    raw, fixed_code, val["wrapped_code"], val["errors"],
                )
                total_fail += 1
                continue

            # ── Build the output record ───────────────────────────────
            record = {
                "category":         cat_name,
                "difficulty":       example.get("difficulty", cat.get("difficulty", "intermediate")),
                "prompt":           example["prompt"],
                "broken_code":      _PYTHON_COMMENT.sub("", example.get("broken_code", "")),
                "error_message":    example.get("error_message", ""),
                "error_code":       example.get("error_code", ""),
                "fixed_code":       fixed_code,
                "explanation":      example.get("explanation", ""),
                "concepts":         example.get("concepts", []),
                "crates":           example.get("crates", []),
                "edition":          EDITION,
                "source_model":     model,
                "validation": {
                    "fmt":           val["fmt"],
                    "clippy":        val["clippy"],
                    "build":         val["build"],
                    "test":          False,  # not run by default
                    "clippy_lints":  val["clippy_lints"],
                    "clippy_output": val["clippy_output"],
                },
                "verified_at":       str(int(time.time())),
                "compiler_ver":      val["compiler_ver"],
                "min_rust_version":  val["min_rust_version"],
                "has_unsafe":        val["has_unsafe"],
                "license":           LICENSE,
            }

            if not val["clippy"]:
                log_clippy_failure(
                    cat_name, attempt, model,
                    val["clippy_lints"], fixed_code,
                )

            write_example(out_path, record)
            generated_this_cat += 1
            total_ok           += 1

            # ── Cooldown ──────────────────────────────────────────────
            if cooldown_every > 0 and total_ok % cooldown_every == 0:
                console.print(
                    f"\n[warn]Cooldown after {total_ok} generations — "
                    f"unloading model for {cooldown_secs}s…[/warn]"
                )
                unload_model(instance_id)
                for remaining_s in range(cooldown_secs, 0, -5):
                    console.print(f"  [dim]{remaining_s}s remaining…[/dim]", end="\r")
                    time.sleep(min(5, remaining_s))
                console.print(f"  [dim]Reloading model…[/dim]" + " " * 20, end="\r")
                new_iid = load_model(model) if args.model else None
                if new_iid:
                    instance_id = new_iid
                    _active_iid = new_iid
                console.print(f"[ok]Cooldown done. Resuming.[/ok]" + " " * 20)

            clippy_badge = "[ok]✓ clippy[/ok]" if val["clippy"] else "[warn]~ clippy warn[/warn]"
            fmt_badge    = "[ok]✓ fmt[/ok]"    if val["fmt"]    else "[warn]~ fmt[/warn]"
            msrv_info    = f"  msrv={val['min_rust_version']}" if val["msrv"] else ""
            console.print(
                f"  [ok]✓ build[/ok]  {clippy_badge}  {fmt_badge}"
                f"{msrv_info}"
                f"  [dim]{cat_name} #{existing + generated_this_cat}[/dim]"
            )

        console.print()

    console.print(
        f"[ok]Done.[/ok]  Generated: [ok]{total_ok}[/ok]  "
        f"Discarded: [fail]{total_fail}[/fail]"
    )

    # Unload the model we loaded so it doesn't sit in memory
    if _active_iid:
        console.print(f"[info]Unloading model…[/info]")
        unload_model(_active_iid)
        _active_iid = None


# ─────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate frontier-format Rust examples using LM Studio",
    )
    p.add_argument(
        "--categories",
        default="rust_categories.jsonl",
        help="Path to categories JSONL (default: rust_categories.jsonl)",
    )
    p.add_argument(
        "--output-dir",
        default="datasets/local",
        help="Output directory (default: datasets/local)",
    )
    p.add_argument(
        "--category",
        default=None,
        help="Generate only for this specific category name",
    )
    p.add_argument(
        "--count",
        type=int,
        default=None,
        help="Examples per category (overrides target_entries in the JSONL)",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Force a specific model ID (default: use whatever LM Studio has loaded)",
    )
    p.add_argument(
        "--cooldown-every",
        type=int,
        default=50,
        metavar="N",
        help="Unload/reload model after every N successful generations (default: 50)",
    )
    p.add_argument(
        "--cooldown-secs",
        type=int,
        default=60,
        metavar="S",
        help="Seconds to pause during cooldown (default: 60)",
    )
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
