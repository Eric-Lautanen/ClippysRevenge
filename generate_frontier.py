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
    python generate_frontier.py --model "tesslate_tessa-rust-t1-7b" --categories rust_categories.jsonl --output-dir datasets/local
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
import uuid
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
TEMPERATURE  = 0.2
MAX_TOKENS   = 8192
REQ_TIMEOUT  = 240
CTX_LENGTH   = 8192
LOAD_TIMEOUT = 120
EDITION      = "2021"
LICENSE      = "Apache-2.0"
SHARED_TARGET_DIR = Path(__file__).parent / "cargo_target"  # shared build cache

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
Rust expert generating training data. Output ONE raw JSON object — no fences, no prose, nothing else.

{
  "category": "<as given>",
  "difficulty": "<beginner|intermediate|advanced>",
  "prompt": "<1-3 sentence developer question about the bug>",
  "broken_code": "<complete Rust file — fn main + all use stmts — that FAILS to compile>",
  "error_message": "<exact compiler error or runtime behaviour>",
  "error_code": "<compile-error|logic-bug|design-issue|runtime-panic>",
  "fixed_code": "<complete, idiomatic Rust file that compiles and passes clippy>",
  "explanation": "<2-4 sentences: root cause and what the fix does>",
  "concepts": ["<tag>"],
  "crates": []
}

Rules:
- Both code fields must be complete Rust files (fn main or async fn main, all use stmts), edition 2021.
- broken_code MUST fail to compile with the stated error — not a trivial typo.
- fixed_code MUST differ from broken_code semantically, not just whitespace or comments.
- No external crates. crates is always [].
- Never use placeholder or stub function bodies — every function must have a real, compilable implementation.
- Only use stdlib APIs you are certain exist. Verify method names (e.g. VecDeque uses push_back, not push).
- Vec, String, Option, Result are in the prelude — never import them with use.
- If a function is generic over T, do not push or return concrete literals of a specific type."""

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
# Matches any { … } block, even if preceded by prose / preamble text.
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

REQUIRED_FIELDS = {"category", "difficulty", "prompt", "broken_code",
                   "error_message", "error_code", "fixed_code", "explanation"}

_VALID_ERROR_CODES = {"compile-error", "logic-bug", "design-issue", "runtime-panic"}
_VALID_DIFFICULTIES = {"beginner", "intermediate", "advanced"}
_MIN_CODE_LEN = 20   # characters — reject suspiciously tiny code snippets


# ── JSON repair helpers ───────────────────────────────────────────────────────

def _try_parse(text: str) -> Optional[dict]:
    try:
        d = json.loads(text)
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None


def _preprocess(text: str) -> str:
    """
    Normalise raw LLM output before any JSON repair attempts.
      • Strip UTF-8 BOM
      • Remove null bytes
      • Normalise Windows line endings (\\r\\n → \\n) and bare \\r → \\n
        so that literal CR/CRLF inside JSON strings are handled uniformly
        by _apply_escapes rather than surfacing as 'Invalid control character'.
    """
    text = text.lstrip('\ufeff')
    text = text.replace('\x00', '')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text


# Valid single-char JSON escape followers: " \\ / b f n r t u
_VALID_JSON_ESC = frozenset('"\\/ bfnrtu')

# Rust raw-string pattern: r#"..."# with any number of # delimiters
_RUST_RAW_STR = re.compile(r'r(#+)"(.*?)\1"', re.DOTALL)


def _fix_rust_raw_strings(text: str) -> tuple[str, bool]:
    """
    Convert Rust raw-string literals (r#"..."#, r##"..."##) that the model
    sometimes emits directly as JSON field values into proper JSON strings.

    In a Rust raw string every character is literal — backslashes are NOT
    escape characters — so we must escape everything for JSON.
    """
    if 'r#"' not in text:
        return text, False

    def _to_json_str(m: re.Match) -> str:
        content = m.group(2)
        # Backslashes in raw strings are literal; escape them first
        content = content.replace('\\', '\\\\')
        content = content.replace('"',  '\\"')
        content = content.replace('\n', '\\n')
        content = content.replace('\r', '\\r')
        content = content.replace('\t', '\\t')
        # Escape remaining control chars
        out: list[str] = []
        for ch in content:
            cp = ord(ch)
            out.append(f'\\u{cp:04x}' if cp < 0x20 else ch)
        return '"' + ''.join(out) + '"'

    fixed = _RUST_RAW_STR.sub(_to_json_str, text)
    return fixed, fixed != text


def _fix_json_comments(text: str) -> tuple[str, bool]:
    """
    Strip // line-comments that appear outside JSON string values.
    Models occasionally add them (e.g. after trailing commas or at EOF).
    Also strips /* … */ block comments.
    This is deliberately conservative: we only strip when we can confirm
    the // sits outside a quoted string by tracking quote state per line.
    """
    changed = False
    out_lines: list[str] = []
    for line in text.split('\n'):
        in_str = False
        result: list[str] = []
        i = 0
        while i < len(line):
            ch = line[i]
            if in_str:
                if ch == '\\' and i + 1 < len(line):
                    result.append(ch)
                    result.append(line[i + 1])
                    i += 2
                elif ch == '"':
                    result.append(ch)
                    in_str = False
                    i += 1
                else:
                    result.append(ch)
                    i += 1
            else:
                if ch == '"':
                    result.append(ch)
                    in_str = True
                    i += 1
                elif ch == '/' and i + 1 < len(line) and line[i + 1] == '/':
                    changed = True
                    break  # drop the rest of the line
                else:
                    result.append(ch)
                    i += 1
        out_lines.append(''.join(result).rstrip())

    result_text = '\n'.join(out_lines)
    # Block comments (approximate — not string-aware, but models rarely nest them)
    stripped = re.sub(r'/\*.*?\*/', '', result_text, flags=re.DOTALL)
    if stripped != result_text:
        changed = True
        result_text = stripped
    return result_text, changed


def _apply_escapes(text: str) -> tuple[str, bool]:
    """
    Walk JSON character by character and fix string-value issues:
      • Literal control chars (\\n \\r \\t, 0x00–0x1F) → proper escape sequences
      • Bare backslash not followed by a valid JSON escape char → \\\\
      • \\\\\\\" (double-backslash + quote) → \\" (fix double-escaped quotes)
    Returns (fixed_text, changed).

    NOTE: This function tracks in-string state with a simple quote toggle so
    it works reliably only when the string boundaries are well-formed.  Use
    _repair_unescaped_quotes for the harder case of unescaped inner quotes.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    changed = False
    in_str  = False

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
                # \\\" → \" — fix double-escaped quotes the model emits
                out.append('\\"')
                i += 3
                changed = True
            elif nxt in _VALID_JSON_ESC:
                # Legitimate JSON escape sequence — pass through verbatim
                out.append(ch)
                out.append(nxt)
                i += 2
            else:
                # Bare backslash not starting a valid escape → escape it
                out.append('\\\\')
                changed = True
                i += 1
        elif ch == '"':
            out.append(ch)
            in_str = False
            i += 1
        elif ch == '\n':
            out.append('\\n');                  changed = True; i += 1
        elif ch == '\r':
            out.append('\\r');                  changed = True; i += 1
        elif ch == '\t':
            out.append('\\t');                  changed = True; i += 1
        elif ord(ch) < 0x20:
            out.append(f'\\u{ord(ch):04x}');   changed = True; i += 1
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


def _repair_unescaped_quotes(text: str) -> Optional[dict]:
    """
    Fix JSON where string values contain unescaped double-quote characters —
    the most common failure mode when the model emits Rust code with format
    strings like println!(\"{}\", x) but forgets to escape the inner quotes.

    Strategy: iteratively try json.loads, use the JSONDecodeError position to
    locate the most recent unescaped '"' that caused the parser to exit string
    context prematurely, insert a backslash there, and retry.

    The backward search from the error position looks for the last '"' that
    is not already preceded by an odd run of backslashes (i.e. not already
    escaped).  We search at most _UQ_LOOKBACK chars back to avoid clobbering
    quotes from earlier, correctly-bounded fields.

    Cap at _UQ_MAX_FIXES iterations to prevent infinite loops on truly
    malformed input.
    """
    _UQ_MAX_FIXES  = 80   # max quotes to escape in one call
    _UQ_LOOKBACK   = 400  # chars to search back from the error position

    fixed = text

    for _ in range(_UQ_MAX_FIXES):
        try:
            result = json.loads(fixed)
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError as e:
            pos = e.pos
            # Scan backward from the error position for the last unescaped '"'
            candidate: Optional[int] = None
            limit = max(0, pos - _UQ_LOOKBACK)
            for i in range(pos - 1, limit - 1, -1):
                if fixed[i] != '"':
                    continue
                # Count preceding backslashes to determine if already escaped
                bs = 0
                j  = i - 1
                while j >= 0 and fixed[j] == '\\':
                    bs += 1
                    j  -= 1
                if bs % 2 == 0:   # even number of backslashes → unescaped
                    candidate = i
                    break

            if candidate is None:
                return None   # can't find a fixable quote

            # Insert a backslash before the unescaped '"'
            fixed = fixed[:candidate] + '\\"' + fixed[candidate + 1:]

    return None   # exceeded iteration limit


def _repair_stray_closing_braces(text: str) -> Optional[dict]:
    """
    Remove stray },  that appear directly after a string field value.

    Some models emit an extra closing brace after multi-line code fields:
        "broken_code": "...",
        },               ← spurious — the object is not closed here
        "error_message": "..."

    The pattern is unambiguous: a `},` that follows a closing string quote
    (`",`) cannot be a legitimate nested-object close (which would follow the
    LAST field of the nested object, with no comma on the preceding line).
    """
    fixed = re.sub(r'",(\s*\n\s*)\},', r'",\1', text)
    if fixed == text:
        return None
    return _repair_and_parse_base(fixed)


def _repair_overescaped_quotes(text: str) -> Optional[dict]:
    """
    Fix JSON where the model emits backslash-escaped quotes *outside* string
    context, most commonly in array literals:
        "concepts": [\"two-phase borrows\", \"borrow checker\"]
    instead of:
        "concepts": ["two-phase borrows", "borrow checker"]

    Walk the text tracking in/out-of-string state.  When outside a string,
    a `\"` sequence is invalid JSON; drop the backslash and treat the `"` as
    opening a new string.

    When a string IS opened by a stripped `\"` delimiter we set a flag so
    that the matching closing `\"` also has its backslash dropped (rather than
    being misread as an escaped inner quote).  Strings opened normally with `"`
    are unaffected.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    changed = False
    in_str = False
    escaped_open = False   # True when current string was opened by \"

    while i < n:
        ch = text[i]
        if in_str:
            if ch == '\\' and i + 1 < n:
                nxt = text[i + 1]
                if nxt == '"' and escaped_open:
                    # Closing delimiter of a \"...\"-delimited string
                    out.append('"')
                    in_str = False
                    escaped_open = False
                    i += 2
                else:
                    out.append(ch)
                    out.append(nxt)
                    i += 2
            elif ch == '"':
                out.append(ch)
                in_str = False
                escaped_open = False
                i += 1
            else:
                out.append(ch)
                i += 1
        else:
            if ch == '\\' and i + 1 < n and text[i + 1] == '"':
                # Backslash before quote outside a string — drop the backslash
                out.append('"')
                in_str = True
                escaped_open = True
                changed = True
                i += 2
            elif ch == '"':
                out.append(ch)
                in_str = True
                escaped_open = False
                i += 1
            else:
                out.append(ch)
                i += 1

    return _try_parse(''.join(out)) if changed else None


def _repair_and_parse(text: str) -> Optional[dict]:
    """
    Try a chain of repairs on *text*, from cheapest to most invasive.

    Order:
      1. Direct parse (no repair needed)
      2. Trailing-comma removal
      3. Control-char / escape normalisation (_apply_escapes)
      4. Combined trailing-commas + escape fix
      5. Truncation recovery (close unclosed strings/objects)
      6. Rust raw-string conversion  (r#"..."# → JSON string)
      7. JSON comment stripping  (// and /* */ comments)
      8. Stray closing-brace removal  (},  after string field values)
      9. Over-escaped quotes in array context  ([\"tag\"] → ["tag"])
     10. Unescaped inner-quote repair (iterative backslash insertion)
     11. Combined: rust-raw + unescaped-quote
     12. Combined: comments + unescaped-quote
     13. Combined: escape fix + unescaped-quote (for literal-CRLF + bare-quote)
    """
    # ── fast path ─────────────────────────────────────────────────────
    if (d := _try_parse(text)):                          return d
    if (d := _repair_trailing_commas(text)):             return d

    esc, esc_changed = _apply_escapes(text)
    if esc_changed:
        if (d := _try_parse(esc)):                       return d

    tc = re.sub(r",\s*([}\]])", r"\1", text)
    tc_esc, tc_esc_changed = _apply_escapes(tc)
    if tc_esc_changed and (d := _try_parse(tc_esc)):     return d

    # ── truncation ────────────────────────────────────────────────────
    if (d := _repair_truncated(text)):                   return d
    if esc_changed and (d := _repair_truncated(esc)):    return d

    # ── Rust raw strings ──────────────────────────────────────────────
    rrs, rrs_changed = _fix_rust_raw_strings(text)
    if rrs_changed:
        if (d := _repair_and_parse_base(rrs)):           return d

    # ── comment stripping ─────────────────────────────────────────────
    nc, nc_changed = _fix_json_comments(text)
    if nc_changed:
        if (d := _repair_and_parse_base(nc)):            return d

    # ── stray closing braces (},  after a string value mid-object) ────
    if (d := _repair_stray_closing_braces(text)):        return d

    # ── over-escaped quotes outside string context ([\"tag\"] pattern) ─
    if (d := _repair_overescaped_quotes(text)):          return d

    # ── unescaped inner quotes (the println!("{}", x) problem) ────────
    if (d := _repair_unescaped_quotes(text)):            return d
    if esc_changed and (d := _repair_unescaped_quotes(esc)): return d

    # ── combined: rust-raw → unescaped-quote ─────────────────────────
    if rrs_changed and (d := _repair_unescaped_quotes(rrs)): return d

    # ── combined: comments → unescaped-quote ─────────────────────────
    if nc_changed and (d := _repair_unescaped_quotes(nc)): return d

    return None


def _repair_and_parse_base(text: str) -> Optional[dict]:
    """
    Reduced repair chain used recursively (no Rust-raw or comment steps,
    to avoid infinite recursion when those fixes call back into repair).
    """
    if (d := _try_parse(text)):                          return d
    if (d := _repair_trailing_commas(text)):             return d
    esc, changed = _apply_escapes(text)
    if changed and (d := _try_parse(esc)):               return d
    if (d := _repair_truncated(text)):                   return d
    if changed and (d := _repair_truncated(esc)):        return d
    if (d := _repair_unescaped_quotes(text)):            return d
    if changed and (d := _repair_unescaped_quotes(esc)): return d
    return None


def _extract_json_candidates(text: str) -> list[str]:
    """
    Return candidate JSON substrings from *text*, from most to least specific.

    Strategy order:
      1. Content inside a ```json ... ``` or ``` ... ``` fence
      2. Outermost { … } span (greedy: first { to last })
      3. Any { … } match found by regex (handles leading prose + trailing junk)
      4. Slice from first { to end (for truncated responses)
    """
    candidates: list[str] = []

    # 1. Fenced block
    m = _FENCE.search(text)
    if m:
        candidates.append(m.group(1).strip())

    # 2. Outermost braces
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start:end + 1])

    # 3. Regex-located object (catches preamble + trailing garbage)
    m2 = _JSON_OBJECT.search(text)
    if m2 and m2.group(0) not in candidates:
        candidates.append(m2.group(0))

    # 4. Truncated: everything from the first brace onwards
    if start != -1:
        tail = text[start:]
        if tail not in candidates:
            candidates.append(tail)

    return candidates


# ─────────────────────────────────────────────────────────────────────────────

# Stores the reason the most recent parse_example call was rejected.
# Read immediately after a None return to get a loggable label.
_last_reject_reason: str = "json-parse"

# Collapse line comments, block comments, and all whitespace runs so that
# code blocks that differ only cosmetically still compare equal.
_STRIP_LINE_COMMENT  = re.compile(r"//[^\n]*")
_STRIP_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_COLLAPSE_WS         = re.compile(r"\s+")


def _normalize_code(code: str) -> str:
    """
    Normalise a Rust snippet for fuzzy equality comparison.
    Strips // and /* */ comments, then collapses all whitespace runs to a
    single space.  Used to catch "no-fix" cases where the model only made
    cosmetic (whitespace / comment) changes between broken_code and fixed_code.
    """
    code = _STRIP_LINE_COMMENT.sub("", code)
    code = _STRIP_BLOCK_COMMENT.sub("", code)
    return _COLLAPSE_WS.sub(" ", code).strip()


def parse_example(text: str) -> Optional[dict]:
    """
    Parse LLM output as a frontier JSON example.
    Returns the dict or None if parsing or validation fails.

    Extraction strategy: tries multiple candidate substrings (fenced block,
    outermost braces, regex-located object, truncation fallback) and applies
    the full repair chain to each before giving up.

    Content validation (post-parse):
      • All REQUIRED_FIELDS present
      • broken_code and fixed_code are not identical (no fix was applied)
      • Neither code is suspiciously short
      • error_code is one of the four canonical values
    """
    global _last_reject_reason
    _last_reject_reason = "json-parse"   # default if we can't even extract JSON

    data: Optional[dict] = None

    for candidate in _extract_json_candidates(text):
        if not candidate:
            continue
        data = _repair_and_parse(candidate)
        if isinstance(data, dict):
            break

    if not isinstance(data, dict):
        return None

    # ── Required-field check ──────────────────────────────────────────
    missing = REQUIRED_FIELDS - data.keys()
    if missing:
        _last_reject_reason = "missing-fields"
        data["_reject_reason"] = "missing-fields"
        return None

    # ── Normalise list fields ─────────────────────────────────────────
    if isinstance(data.get("concepts"), str):
        data["concepts"] = [t.strip() for t in data["concepts"].split(",") if t.strip()]
    if isinstance(data.get("crates"), str):
        data["crates"] = [c.strip() for c in data["crates"].split(",") if c.strip()]
    if not isinstance(data.get("concepts"), list):
        data["concepts"] = []
    if not isinstance(data.get("crates"), list):
        data["crates"] = []

    # ── Content quality checks ────────────────────────────────────────
    broken  = data.get("broken_code",  "").strip()
    fixed   = data.get("fixed_code",   "").strip()
    error_c = data.get("error_code",   "")

    # Reject if the model returned the same code for both fields (exact or fuzzy).
    # Fuzzy comparison strips comments and collapses whitespace so that purely
    # cosmetic differences (reformatting, added/removed blank lines, comment-only
    # changes) are still caught as no-fix.
    if broken == fixed or _normalize_code(broken) == _normalize_code(fixed):
        _last_reject_reason = "no-fix"
        return None

    # Reject trivially short code (almost certainly a placeholder)
    if len(broken) < _MIN_CODE_LEN or len(fixed) < _MIN_CODE_LEN:
        _last_reject_reason = "too-short"
        return None

    # Normalise error_code to the canonical set; reject unknown values
    if error_c not in _VALID_ERROR_CODES:
        lower_map = {v.lower(): v for v in _VALID_ERROR_CODES}
        normalised = lower_map.get(error_c.lower())
        if normalised:
            data["error_code"] = normalised
        else:
            _last_reject_reason = "bad-error-code"
            return None

    # Normalise difficulty
    diff = data.get("difficulty", "")
    if diff not in _VALID_DIFFICULTIES:
        lower_diff = {v.lower(): v for v in _VALID_DIFFICULTIES}
        data["difficulty"] = lower_diff.get(diff.lower(), "intermediate")

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

def _run(cmd: list[str], cwd: Path, timeout: int = 120, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
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

    SHARED_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    cargo_env = {**os.environ, "CARGO_TARGET_DIR": str(SHARED_TARGET_DIR)}

    with tempfile.TemporaryDirectory(prefix="frontier_") as tmp:
        proj = Path(tmp) / "rust-example"
        proj.mkdir()
        (proj / "Cargo.toml").write_text(make_cargo_toml(all_crates), encoding="utf-8")
        src = proj / "src"
        src.mkdir()
        (src / "main.rs").write_text(wrapped, encoding="utf-8")

        # ── cargo check ──────────────────────────────────────────────
        r = _run(["cargo", "check", "--quiet", "--color=never"], proj, env=cargo_env)
        if r.returncode != 0:
            result["errors"] = [r.stderr.strip()]
            return result  # no point continuing
        result["build"] = True

        # ── cargo fmt --check ────────────────────────────────────────
        r = _run(["cargo", "fmt", "--", "--check"], proj, env=cargo_env)
        result["fmt"] = (r.returncode == 0)

        # ── cargo clippy ─────────────────────────────────────────────
        r = _run(
            ["cargo", "clippy", "--quiet", "--color=never",
             "--", "-D", "warnings"],
            proj,
            env=cargo_env,
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


def check_broken_compiles(broken_code: str, declared_crates: list[str]) -> bool:
    """
    Return True if *broken_code* successfully passes cargo check — meaning the
    model generated code that isn't actually broken, making the example invalid.

    Only the error path is interesting here, so we skip clippy/fmt/msrv and
    return as soon as cargo check finishes.  The shared target dir is reused
    so incremental compilation keeps this fast.
    """
    wrapped = wrap_for_check(broken_code)
    detected = detect_crates(wrapped)
    all_crates = sorted(set(declared_crates + detected))

    SHARED_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    cargo_env = {**os.environ, "CARGO_TARGET_DIR": str(SHARED_TARGET_DIR)}

    try:
        with tempfile.TemporaryDirectory(prefix="frontier_brok_") as tmp:
            proj = Path(tmp) / "rust-example"
            proj.mkdir()
            (proj / "Cargo.toml").write_text(make_cargo_toml(all_crates), encoding="utf-8")
            src = proj / "src"
            src.mkdir()
            (src / "main.rs").write_text(wrapped, encoding="utf-8")
            r = _run(["cargo", "check", "--quiet", "--color=never"], proj, env=cargo_env)
            return r.returncode == 0
    except Exception:
        return False  # if cargo itself errors, don't block the attempt


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


def log_parse_failure(
    cat_name: str,
    attempt: int,
    model: str,
    raw: str,
    reason: str = "json-parse",
) -> None:
    """Append a JSON parse failure record to frontier_parse_failures.log.

    *reason* is a short label for why parse_example returned None:
      json-parse      — could not extract / repair valid JSON
      no-fix          — broken_code identical to fixed_code (exact or fuzzy)
      too-short       — one of the code fields is suspiciously short
      bad-error-code  — error_code not in the canonical set
      missing-fields  — one or more REQUIRED_FIELDS absent
      broken-compiles — broken_code passes cargo check (not actually broken)
    """
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(_PARSE_FAIL_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n{_SEP}\n")
        f.write(f"  {ts}  cat={cat_name!r}  attempt={attempt}  model={model}  reason={reason}\n")
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
        consecutive_same_reason = 0
        last_seen_reason: str = ""
        _CONSEC_BAIL = 4   # bail after this many identical rejection reasons in a row

        while generated_this_cat < remaining:
            attempt += 1
            if attempt > remaining * 4:
                console.print(f"  [warn]Too many attempts for {cat_name!r}, moving on.[/warn]")
                break

            console.print(f"  [dim]attempt {attempt}…[/dim]")

            def _fail(reason: str) -> None:
                """Increment counters and track consecutive same-reason runs."""
                nonlocal total_fail, consecutive_same_reason, last_seen_reason
                total_fail += 1
                if reason == last_seen_reason:
                    consecutive_same_reason += 1
                else:
                    consecutive_same_reason = 1
                    last_seen_reason = reason

            # ── LLM call ─────────────────────────────────────────────
            user_prompt = build_user_prompt(cat)
            raw = call_llm(model, user_prompt)
            if raw is None:
                _fail("llm-error")
                if consecutive_same_reason >= _CONSEC_BAIL:
                    console.print(f"  [warn]{_CONSEC_BAIL} consecutive '{last_seen_reason}' — skipping category.[/warn]")
                    break
                continue

            # ── JSON parse ────────────────────────────────────────────
            example = parse_example(raw)
            if example is None:
                reason = _last_reject_reason
                console.print(f"  [warn]JSON parse failed (attempt {attempt})  [{reason}][/warn]")
                log_parse_failure(cat_name, attempt, model, raw, reason=reason)
                _fail(reason)
                if consecutive_same_reason >= _CONSEC_BAIL:
                    console.print(f"  [warn]{_CONSEC_BAIL} consecutive '{last_seen_reason}' — skipping category.[/warn]")
                    break
                continue

            fixed_code = _PYTHON_COMMENT.sub("", example.get("fixed_code", "")).strip()
            if not fixed_code:
                console.print(f"  [warn]Empty fixed_code (attempt {attempt})[/warn]")
                _fail("empty-fixed-code")
                if consecutive_same_reason >= _CONSEC_BAIL:
                    console.print(f"  [warn]{_CONSEC_BAIL} consecutive '{last_seen_reason}' — skipping category.[/warn]")
                    break
                continue

            # ── Broken-code sanity check ──────────────────────────────
            # For compile-error examples, verify broken_code actually fails
            # to compile.  If it passes, the model hallucinated the error —
            # reject cheaply before spending cargo time on fixed_code.
            if example.get("error_code") == "compile-error":
                broken_code = _PYTHON_COMMENT.sub("", example.get("broken_code", "")).strip()
                if broken_code and check_broken_compiles(broken_code, example.get("crates", [])):
                    console.print(
                        f"  [warn]broken_code compiles cleanly (attempt {attempt}) "
                        f"[broken-compiles][/warn]"
                    )
                    log_parse_failure(cat_name, attempt, model,
                                      raw, reason="broken-compiles")
                    _fail("broken-compiles")
                    if consecutive_same_reason >= _CONSEC_BAIL:
                        console.print(f"  [warn]{_CONSEC_BAIL} consecutive '{last_seen_reason}' — skipping category.[/warn]")
                        break
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
                _fail("cargo-check")
                if consecutive_same_reason >= _CONSEC_BAIL:
                    console.print(f"  [warn]{_CONSEC_BAIL} consecutive '{last_seen_reason}' — skipping category.[/warn]")
                    break
                continue

            # ── Build the output record ───────────────────────────────
            record = {
                "id":               str(uuid.uuid4()),
                "schema_variant":   "bug_fix",
                "source":           "frontier",
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
                    "test":          False,
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
            consecutive_same_reason = 0   # success resets the streak
            last_seen_reason        = ""

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