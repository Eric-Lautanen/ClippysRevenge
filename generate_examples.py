#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║   FrontierGen  —  Rust Code Example Generator                ║
║   Generates idiomatic Rust examples, validates with cargo    ║
╚══════════════════════════════════════════════════════════════╝

Reads rust_categories.jsonl, prompts LM Studio for frontier-format
Rust examples, validates code with cargo check / clippy / fmt /
msrv, and writes verified examples to datasets/local/<category>.jsonl.

Usage:
    python generate_frontier.py --model "model-id"
    python generate_frontier.py --model "model-id" --count 3
    python generate_frontier.py --model "model-id" --category "Core - Move Semantics on Function Call"
    python generate_frontier.py --model "model-id" --cooldown-every 50 --cooldown-secs 60
    python generate_frontier.py --categories rust_categories.jsonl --output-dir datasets/local
    python generate_frontier.py --model "tesslate_tessa-rust-t1-7b" --categories rust_categories.jsonl --output-dir data/examples
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import requests  # type: ignore[import-untyped]
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
TEMPERATURE  = 0.75
MAX_TOKENS   = 8192
REQ_TIMEOUT  = 240
CTX_LENGTH   = 8192
LOAD_TIMEOUT = 120
EDITION      = "2021"
LICENSE      = "Apache-2.0"
MAX_FAILURES     = 3   # skip a category after this many *consecutive* failures
MAX_CAT_FAILURES = 5  # skip a category after this many *total* failures (never resets)

# Set to True to inject the category's prompt_focus field into every user prompt.
# This gives the model precise guidance on which sub-concepts to demonstrate.
# Set to False to test baseline generation quality without the extra context.
USE_PROMPT_FOCUS: bool = False

# Set to True to run `cargo fmt`, `cargo fix`, and `cargo clippy --fix` on
# the model's code before validation.  All three tools rewrite main.rs in
# place; the cleaned code is what gets stored in the dataset.
# fmt        — fixes all formatting (indentation, spacing, trailing commas …)
# fix        — applies safe automatic compiler lint suggestions
# clippy fix — applies Clippy's machine-applicable suggestions on top
# cargo check / clippy still run afterward so genuinely broken code is still
# rejected.  Set to False to store the model's output exactly as produced.
AUTO_FIX_CODE: bool = True

# Set to True to write a rustfmt.toml with stable-only options into each
# temp project before running cargo fmt.  This normalises import ordering,
# derive merging, trailing commas, and line width across all examples for a
# more consistent dataset.  Only stable rustfmt options are used -- no
# nightly required.  Set to False if the toml causes unexpected fmt failures
# or if you prefer vanilla rustfmt defaults.
AUTO_FMT_TOML: bool = True

# Content written to rustfmt.toml in each temp project when AUTO_FMT_TOML is True.
# All options below are stable on stable rustfmt as of April 2026.
# Unstable options (imports_granularity, group_imports) are deliberately excluded —
# they are still nightly-only and have known non-idempotency bugs.
_RUSTFMT_TOML: str = """# rustfmt.toml -- stable options only (rustfmt stable, April 2026)
# Injected by generate_examples.py when AUTO_FMT_TOML = True.

# Merge multiple #[derive(...)] attributes into one line.
merge_derives = true

# Sort use statements alphabetically within each group.
reorder_imports = true

# Sort mod declarations alphabetically.
reorder_modules = true

# Collapse empty impl/fn bodies onto a single line where they fit.
empty_item_single_line = true

# Maximum line width before wrapping (100 is common modern Rust style).
max_width = 100

# Trailing comma in function args / struct literals when wrapped vertically.
trailing_comma = "Vertical"

# Use Unix-style newlines regardless of platform.
newline_style = "Unix"
"""

# Set to True to enable Stage 2 MCP web-search assisted fixing.
# When True, cargo check errors that aren't trivial syntax failures are sent to
# LM Studio's /api/v1/chat endpoint with the 'mcp/web-search' integration so
# the model can look up the correct fix online before regenerating.
# Set to False to skip MCP entirely (Stage 1 self-fix only).
# Prerequisites when True:
#   • mcp.py registered in LM Studio's mcp.json as "web-search"
#   • "Allow calling servers from mcp.json" ON in Server Settings
USE_MCP: bool = False

# How many times Stage 1 (self-fix from cargo check output) will retry before
# giving up or escalating to Stage 2 MCP search.
# Each retry calls the model again with the same errors; increasing this gives
# the model more chances to correct itself at the cost of extra inference time.
SELF_FIX_RETRIES: int = 3
SHARED_TARGET_DIR = Path(__file__).parent / "cargo_target"  # shared build cache

# LM Studio API token — set LM_API_TOKEN in the environment if auth is enabled.
# When auth is disabled in LM Studio Server Settings this value is ignored.
LM_API_TOKEN: str = os.environ.get("LM_API_TOKEN", "")

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
    # ── Additional crates required by specific categories ──────────────────
    "walkdir":            'walkdir = "2"',
    "tempfile":           'tempfile = "3"',
    "smallvec":           'smallvec = { version = "1", features = ["union"] }',
    "arrayvec":           'arrayvec = "0.7"',
    "bumpalo":            'bumpalo = { version = "3", features = ["collections"] }',
    "smolstr":            'smolstr = "0.2"',
    "compact_str":        'compact_str = "0.8"',
    "unicode_segmentation": 'unicode-segmentation = "1"',
    "miette":             'miette = { version = "5", features = ["fancy"] }',
    "eyre":               'eyre = "0.6"',
    "color_eyre":         'color-eyre = "0.6"',
    "itertools":          'itertools = "0.13"',
}

# ─────────────────────────────────────────────────────────
#  LLM PROMPTS
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Rust expert, senior dev. Your ONLY output must be a single, raw JSON object. Do not include markdown formatting, code fences (```), or any conversational text.

OUTPUT SCHEMA:
{
  "category": "Broad topic (e.g., File I/O, Data Structures)",
  "difficulty": "beginner|intermediate|advanced",
  "prompt": "Max 2 sentences. Write as a non-Rust programmer Googling a generic problem. Use plain English. (Example: 'I need to group items by their name' instead of 'How do I use a HashMap').",
  "code": "Complete, working Rust 2021 code. 15-50 lines. Must compile!",
  "explanation": "Brief reasoning for why this code is idiomatic.",
  "concepts": ["concept1", "concept2"],
  "crates": ["crate_name"] // Leave empty if stdlib only
}

STRICT CODE RULES:
1. NO COMMENTS ALLOWED: The "code" string must contain zero comments. Remove all //, /*, ///, and //!.
2. MUST COMPILE: Code must be idiomatic, complete, and warning-free. Do not use #[allow(...)].
3. COMPLETE: Include all necessary `use` statements. No stubbed functions.
4. MUST BE ACTUAL CODE. NO CONCEPTS OR EXAMPLES.
5. DOMAIN-SPECIFIC: The code must be an example of a real world application.  No "Hello World" or toy scripts."""

SYSTEM_PROMPT_FIX = """\
You are a Rust compiler expert. Your only job is to fix broken Rust code.

OUTPUT RULES — follow these exactly:
- Output ONLY the corrected Rust source code.
- No JSON. No markdown fences. No explanation. No commentary.
- Do not wrap the code in ```rust or ``` blocks.
- Do not output anything before or after the code.
- Fix every compiler error shown. Do not suppress warnings with #[allow(…)].
- No comments of any kind — no //, no /* */, no ///, no //!. Remove any that exist in the broken code."""

# ── 100 domain-specific coding project contexts ───────────────────────────────
# Injected randomly into user prompts to force variety in the generated examples.
# Each entry describes a real-world project where the Rust category would matter.

_CODING_CONTEXTS: list[str] = [
    # Systems / Infrastructure
    "a Linux process monitor that streams CPU and memory stats in real time",
    "a custom memory allocator for a game engine",
    "a lightweight container runtime (like a tiny Docker)",
    "a cross-platform system tray application",
    "a kernel module loader that validates ELF binaries before exec",
    "a packet sniffer for a home network security tool",
    "a bare-metal embedded firmware for an IoT sensor node",
    "a USB HID driver for a custom input device",
    "a bootloader for a small RISC-V dev board",
    "a hypervisor that runs sandboxed WebAssembly modules",
    # CLI Tools
    "a CLI password manager that encrypts vaults locally",
    "a git-log analyzer that produces burndown charts in the terminal",
    "a fast recursive file search tool (like ripgrep but smaller)",
    "a command-line CSV transformer with regex column filters",
    "a dotfile manager that symlinks configs across machines",
    "a terminal Pomodoro timer with desktop notifications",
    "a CLI tool that batch-renames photos based on EXIF date",
    "a cross-platform port scanner with colorized output",
    "a local secret scanner that checks git history for leaked keys",
    "a diff-and-patch tool for binary firmware images",
    # Networking / Servers
    "a high-throughput HTTP/2 reverse proxy",
    "a WebSocket chat server handling thousands of concurrent rooms",
    "a DNS resolver with a local cache and DNSSEC validation",
    "a gRPC gateway that translates REST calls to protobuf messages",
    "a SOCKS5 proxy with per-connection traffic shaping",
    "a toy SMTP server that queues and delivers email locally",
    "an mTLS API gateway for a microservice mesh",
    "a rate-limiter middleware for an Axum-based REST API",
    "a lightweight load balancer using consistent hashing",
    "a pub/sub message broker inspired by NATS",
    # Databases / Storage
    "a write-ahead log (WAL) engine for a custom key-value store",
    "a time-series database optimized for IoT telemetry",
    "a columnar storage engine with run-length encoding",
    "a B-tree index implementation for an embedded database",
    "a log-structured merge tree (LSM) storage backend",
    "a distributed consensus module using Raft",
    "a Redis-compatible in-memory cache server",
    "a SQLite extension that adds vector similarity search",
    "a backup tool that deduplicates file chunks with a rolling hash",
    "a graph database with BFS/DFS query primitives",
    # Parsers / Compilers / Languages
    "a Markdown-to-HTML renderer with custom extension syntax",
    "a JSON5 parser that preserves comments for config files",
    "a TOML linter and auto-formatter",
    "a simple Lisp interpreter with a REPL",
    "a stack-based bytecode VM for a scripting language",
    "a Pratt parser for a toy expression language",
    "a CSS minifier and property sorter",
    "a template engine with Jinja2-compatible syntax",
    "a source map generator for a transpiler",
    "a protobuf schema validator and diff tool",
    # Web / APIs
    "a server-side rendered blog engine with an Axum backend",
    "a REST API for a task management app with JWT auth",
    "a GraphQL server with DataLoader-style batching",
    "a webhook dispatcher that retries failed deliveries with backoff",
    "an OpenAPI spec validator and code-gen tool",
    "a static site generator that hot-reloads on file change",
    "a URL shortener service backed by a local SQLite DB",
    "a headless CMS API with role-based access control",
    "a server-sent events (SSE) feed for a live sports scoreboard",
    "a multipart file upload handler with virus scanning integration",
    # Games / Simulation
    "a 2D tile-based game engine with an ECS architecture",
    "a physics simulation for rigid body collisions",
    "a procedural dungeon generator using cellular automata",
    "a real-time strategy game server handling unit pathfinding",
    "a chess engine with alpha-beta pruning",
    "a Conway's Game of Life simulation running on the GPU via WGPU",
    "a Tetris clone with a pluggable AI bot interface",
    "a particle system renderer for a game special effects library",
    "a replay recorder and playback system for a multiplayer game",
    "a voxel world engine with greedy meshing",
    # Data / ML / Science
    "a fast CSV ingestion pipeline for a data warehouse",
    "a neural network inference engine for ONNX models",
    "a distributed map-reduce job runner",
    "a data frame library with lazy evaluation",
    "a feature store that computes embeddings on the fly",
    "a time-series anomaly detector using z-score sliding windows",
    "a Monte Carlo simulation for options pricing",
    "a bioinformatics tool that aligns DNA sequences with Smith-Waterman",
    "a geospatial index using an R-tree for spatial queries",
    "a streaming statistics library (mean, variance, quantiles) over sensor data",
    # Security / Crypto
    "a certificate transparency log monitor",
    "a hardware security key (FIDO2/WebAuthn) library",
    "a fuzzing harness for a file format parser",
    "a zero-knowledge proof verifier for a toy protocol",
    "an end-to-end encrypted messaging protocol (Signal-style)",
    "a TLS 1.3 handshake implementation from scratch",
    "a JWT library with pluggable signing algorithms",
    "a secure multi-party computation (MPC) demo for private set intersection",
    "a post-quantum key encapsulation module (Kyber)",
    "a memory-safe sandbox for executing untrusted Wasm plugins",
    # Tooling / DevOps
    "a build system with incremental compilation and dependency tracking",
    "a CI pipeline runner that executes jobs in isolated namespaces",
    "a log aggregator that tails multiple files and ships to Elasticsearch",
    "a distributed tracing exporter (OpenTelemetry compatible)",
    "a config file watcher that hot-reloads app settings without restart",
    "a Kubernetes operator that reconciles custom resource definitions",
    "a container image layer analyzer and size optimizer",
    "a chaos engineering agent that randomly kills processes under load",
    "a benchmark harness for comparing serialization formats",
    "a plugin system where third-party extensions are loaded as shared libraries",
]

# ── 30 rotating user prompts ──────────────────────────────────────────────────
# Each entry is a format string. {category} and {context} are substituted at
# call time. {context} is drawn randomly from _CODING_CONTEXTS so each
# generation gets a different real-world angle even within the same category.
#
# VOICE RULE: Every prompt must read as someone who has NEVER written Rust
# before. They are coming from Python, JavaScript, Go, or no background at
# all. They do not know Rust terms, idioms, or concepts. They only know what
# they want the program to DO.

_USER_PROMPTS: list[str] = [
    # 1 — Python dev picking up Rust for the first time
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as a Python developer trying Rust for "
    "the first time. They know how to do this in Python but have no idea how Rust works. "
    "They describe what they want the program to do for their project in plain Python-brained "
    "terms. No Rust vocabulary whatsoever.",

    # 2 — JavaScript dev completely lost
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as a JavaScript developer who just "
    "installed Rust for the first time and is completely lost. They describe what they "
    "want to happen in their project using JavaScript concepts (callbacks, objects, arrays) "
    "and ask for help achieving it. No Rust terms.",

    # 3 — total beginner, never programmed in a systems language
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as someone who has only ever used "
    "high-level scripting languages (Python, Ruby, PHP) and is trying Rust for the first "
    "time because they heard it was fast. They describe the problem in their project "
    "in plain everyday English. Zero systems programming vocabulary.",

    # 4 — Go developer who expected Rust to be similar
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as a Go developer who assumed Rust would "
    "work the same way — they describe how they'd solve it in Go and ask why Rust seems "
    "so different for their project. No Rust jargon. They only speak in Go terms.",

    # 5 — heard Rust was fast, decided to rewrite everything
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as someone who decided to rewrite their "
    "Python project in Rust because they read it was fast, and now they're completely "
    "stuck on the first real thing they need to do. Describe the stuck moment in plain "
    "English. No Rust terminology.",

    # 6 — copy-pasting from tutorials but it's not working
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as a Rust newcomer who has been "
    "copy-pasting tutorial code and modifying it for their project, but it stopped "
    "working and they don't understand why. They describe what they're trying to do "
    "in plain English. No Rust vocabulary.",

    # 7 — asking a friend who knows Rust
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as someone texting a friend who knows "
    "Rust — very casual, describes what they need for their project in everyday language, "
    "admits they just started learning. The friend speaks Rust; they don't. No Rust terms.",

    # 8 — data scientist trying to speed up their pipeline
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as a data scientist who knows Python and "
    "NumPy but was told to try Rust for performance in their project. They describe the "
    "data problem in plain terms. No Rust vocabulary — only data/math language.",

    # 9 — frontend dev building their first backend in Rust
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as a frontend developer (React, TypeScript) "
    "writing their first backend service in Rust for their project. They describe what "
    "the server needs to do using web/frontend vocabulary. No Rust terms.",

    # 10 — sysadmin replacing a bash script
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as a sysadmin who has been told to stop "
    "writing bash scripts and use Rust instead. They describe what their script did for "
    "their project in shell-script terms (pipes, grep, files) and ask how to do the "
    "same thing. No Rust vocabulary.",

    # 11 — student working on a class project
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as a CS student assigned to use Rust "
    "for their class project for the first time. They know C or Java from class but "
    "describe what they need in plain terms. Slightly stressed. No Rust jargon.",

    # 12 — game modder who wants to write native code
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as a game modder or Lua scripter who "
    "wants to move into native code with Rust for their project. They describe game "
    "logic or mod behavior in plain gamer/scripter terms. No Rust vocabulary.",

    # 13 — 'the Rust book confused me'
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as someone who tried reading The Rust "
    "Book, got confused, closed it, and is now just asking for working code for their "
    "project. They describe what they want to happen in plain outcome-focused English. "
    "No Rust terms.",

    # 14 — DevOps engineer writing their first CLI tool in Rust
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as a DevOps engineer who usually writes "
    "Python scripts and Terraform configs but is trying Rust for a CLI tool in their "
    "project. They describe the tool's behavior in infrastructure/ops language. "
    "No Rust vocabulary.",

    # 15 — mobile developer trying desktop/systems for the first time
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as an iOS or Android developer "
    "branching out into desktop or systems programming with Rust for their project. "
    "They describe the feature in mobile app terms (background task, state, events). "
    "No Rust terminology.",

    # 16 — frustrated that Rust won't let them do what Python would
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as a Python developer frustrated that "
    "Rust won't let them do something that 'just works' in Python. They describe the "
    "thing they're trying to do for their project and complain that Rust keeps "
    "blocking them. No Rust jargon — only Python comparisons.",

    # 17 — read a Hacker News post and got excited
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as someone who got hyped reading a "
    "Hacker News post about Rust being fast/safe and decided to try it for their "
    "project right away, with zero preparation. They describe what they want to build "
    "in enthusiastic but non-technical terms. No Rust vocabulary.",

    # 18 — switching from C, scared of memory bugs
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as a C developer who is tired of "
    "memory bugs and segfaults in their project and has heard Rust prevents them. "
    "They describe what they were doing in C terms (malloc, pointer, struct) and "
    "ask how to do the same thing safely. No Rust-specific vocabulary.",

    # 19 — building a side project to learn Rust
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as someone who picked Rust as a "
    "weekend learning project and chose something fun to build. They describe what "
    "feature they want to add next in plain terms and admit they're still figuring "
    "out the basics. Casual and curious. No Rust jargon.",

    # 20 — 'my coworker said just use Rust'
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as someone whose coworker told them "
    "'just rewrite it in Rust' without explaining anything. They describe what their "
    "project does and what they need the code to accomplish, completely blind to Rust "
    "conventions. Mildly exasperated. No Rust terms.",

    # 21 — Java developer baffled by no garbage collector
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as a Java or C# developer confused "
    "that Rust doesn't have a garbage collector. They describe what they want to do "
    "for their project using OOP terms (class, object, method, interface) and ask "
    "how it works. No Rust vocabulary.",

    # 22 — 'I just need it to not crash when two things happen at once'
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as a newcomer whose project falls "
    "apart when multiple things happen at the same time. They don't know what "
    "concurrency means — they just describe the symptom in plain English and "
    "ask for code that handles it. No Rust terms.",

    # 23 — scientist who knows MATLAB and nothing else
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as a scientist or engineer who only "
    "knows MATLAB or R and is trying Rust for a performance-critical part of their "
    "project. They describe the computation in math/matrix terms. No Rust vocabulary.",

    # 24 — just wants a small program to do one thing
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as someone who just wants a small "
    "standalone program that does exactly one useful thing for their project. "
    "They describe the input, the output, and the behavior. No theory, no Rust terms, "
    "just a plain description of what they want.",

    # 25 — non-technical founder who learned to code
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as a non-technical founder who learned "
    "to code via bootcamp (JavaScript/Python) and is attempting Rust for the performance "
    "needs of their startup project. Plain outcome-focused language. No Rust vocabulary.",

    # 26 — embedded/hardware person coming from C
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as a hardware engineer who writes "
    "firmware in C and is trying Rust on a new embedded project. They describe what "
    "the hardware needs to do in electronics/firmware terms (registers, interrupts, GPIO). "
    "No Rust vocabulary.",

    # 27 — first Rust project after finishing a tutorial
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as someone who just finished a '30 "
    "minutes of Rust' tutorial and is now trying to build something real. They hit "
    "the first wall immediately and describe what they want in plain English. "
    "Slightly overwhelmed. No Rust jargon.",

    # 28 — PHP/WordPress developer going low-level
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as a PHP or WordPress developer who "
    "wants to build something fast and native for their project. They describe what "
    "they want using web/server terms (request, response, database, query). "
    "No Rust vocabulary.",

    # 29 — 'I followed a YouTube tutorial and now I'm stuck'
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as someone who followed a YouTube "
    "Rust tutorial until it ended and then tried to go further on their own project "
    "and got stuck immediately. They describe what they're trying to do next in "
    "plain English. No Rust terms.",

    # 30 — Ruby developer who thinks everything should be magic
    "TOPIC: {category}\n"
    "PROJECT: {context}\n\n"
    "TASK: For the JSON 'prompt' field, write it as a Ruby or Rails developer confused "
    "that Rust doesn't just 'figure it out.' They describe what they want their project "
    "to do in Ruby-speak (block, method, hash, symbol) and ask why it's so hard. "
    "No Rust vocabulary.",
]

# ── Category → required crates ────────────────────────────────────────────────
# Maps tag names (lowercased, hyphens stripped) to CRATE_MAP keys.
# Built from the tags field so it stays in sync with whatever tags exist.
_TAG_TO_CRATE: dict[str, str] = {
    "thiserror":            "thiserror",
    "anyhow":               "anyhow",
    "regex":                "regex",
    "indexmap":             "indexmap",
    "dashmap":              "dashmap",
    "serde":                "serde",
    "serde_json":           "serde_json",
    "rayon":                "rayon",
    "rand":                 "rand",
    "chrono":               "chrono",
    "uuid":                 "uuid",
    "clap":                 "clap",
    "tracing":              "tracing",
    "arena":                "typed_arena",
    "bumpalo":              "bumpalo",
    "walkdir":              "walkdir",
    "tempfile":             "tempfile",
    "smolstr":              "smolstr",
    "compactstring":        "compact_str",   # tag is "CompactString"
    "smallvec":             "smallvec",
    "arrayvec":             "arrayvec",
    "miette":               "miette",
    "eyre":                 "eyre",
    "coloreyre":            "color_eyre",    # tag is "color-eyre"
    "itertools":            "itertools",
    "unicodesegmentation":  "unicode_segmentation",
}


def _crates_for_category(cat: dict) -> list[str]:
    """
    Return the external crate names this category's examples should use,
    derived from the category's tags. Only returns crates present in CRATE_MAP.
    """
    found: list[str] = []
    seen: set[str] = set()
    for tag in cat.get("tags", []):
        key = tag.lower().replace("-", "").replace("_", "")
        crate = _TAG_TO_CRATE.get(key)
        if crate and crate not in seen and crate in CRATE_MAP:
            found.append(crate)
            seen.add(crate)
    return found


def build_user_prompt(
    cat: dict,
    attempt: int = 1,
) -> str:
    """
    Build a user prompt for the given category + attempt number.

    Picks a random prompt template AND a random domain-specific project
    context from _CODING_CONTEXTS, so every generation has a different
    voice and real-world angle even within the same category.

    When USE_PROMPT_FOCUS is True and the category has a prompt_focus field,
    that text is appended so the model knows exactly which sub-concepts to
    demonstrate.  Toggle USE_PROMPT_FOCUS in constants to compare quality.
    """
    category = cat["category"]

    # Random selection — no deterministic cycling so consecutive retries
    # on the same category always land on a different template+context pair.
    template = random.choice(_USER_PROMPTS)
    context  = random.choice(_CODING_CONTEXTS)
    prompt   = template.format(category=category, context=context)

    # ── Optional prompt_focus injection ──────────────────────────────────────
    if USE_PROMPT_FOCUS:
        focus = cat.get("prompt_focus", "").strip()
        if focus:
            prompt += (
                f'\n\nFOCUS: {focus}\n'
            )

    cat_crates = _crates_for_category(cat)
    if cat_crates:
        crate_json = "[" + ", ".join(f'"{c}"' for c in cat_crates) + "]"
        prompt += (
            f'\nRequired crates: {", ".join(cat_crates)}. '
            f'Set "crates": {crate_json} and import/use these crates in the code.'
        )
    else:
        prompt += '\n'

    return prompt


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
    Load model via lms CLI so --gpu max and --context-length are honoured.
    Unloads any currently loaded models first to avoid stacking instances.
    Returns model_id as the instance handle, or None on failure.
    """
    # Always clear any loaded models first so we never stack instances
    try:
        subprocess.run(
            ["lms", "unload", "--all"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
    except Exception:
        pass  # Non-fatal — proceed with load regardless

    try:
        result = subprocess.run(
            [
                "lms", "load", model_id,
                "--gpu",            "max",
                "--context-length", str(CTX_LENGTH),
                "--identifier",     model_id,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=LOAD_TIMEOUT,
        )
        if result.returncode == 0:
            console.print(f"[ok]Model loaded via CLI (gpu=max, ctx={CTX_LENGTH})[/ok]")
            return model_id
        combined = (result.stdout + result.stderr).lower()
        if "already" in combined:
            console.print(f"[warn]Model already loaded — reusing.[/warn]")
            return model_id
        console.print(f"[fail]lms load failed: {result.stderr.strip()}[/fail]")
        return None
    except FileNotFoundError:
        console.print("[fail]`lms` CLI not found — is it on your PATH?[/fail]")
        return None
    except subprocess.TimeoutExpired:
        console.print("[fail]lms load timed out.[/fail]")
        return None
    except Exception as e:
        console.print(f"[fail]load_model error: {e}[/fail]")
        return None


def unload_model(instance_id: str) -> bool:
    """Unload model via lms CLI."""
    try:
        result = subprocess.run(
            ["lms", "unload", instance_id],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        if result.returncode == 0:
            return True
        console.print(f"[warn]lms unload warning: {result.stderr.strip()}[/warn]")
        return False
    except Exception as e:
        console.print(f"[warn]unload_model error: {e}[/warn]")
        return False


def call_llm(
    model_id: str,
    user_prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> Optional[str]:
    """
    Stream a completion from LM Studio.  Returns the full response text
    with <think> blocks stripped, or None on error.

    Thinking tokens are shown live on a single overwriting line.
    Output tokens stream directly to stdout.

    Pass system_prompt=SYSTEM_PROMPT_FIX for fix-stage calls so the model
    outputs only corrected Rust code instead of the full JSON schema.
    """
    payload = {
        "model":       model_id,
        "messages":    [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": TEMPERATURE,
        "max_tokens":  MAX_TOKENS,
        "stream":      True,
        "reasoning":   {"effort": "none"}, 
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
        _auth_headers: dict[str, str] = {}
        if LM_API_TOKEN:
            _auth_headers["Authorization"] = f"Bearer {LM_API_TOKEN}"
        with requests.post(
            f"{API_OAI}/chat/completions",
            json=payload,
            headers=_auth_headers,
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
#  MCP-ASSISTED CARGO FIX
# ─────────────────────────────────────────────────────────

# Errors that are pure codegen failures — mismatched braces, unclosed strings,
# etc. Web search cannot help with these; skip MCP and go straight to self-fix
# or discard.
_TRIVIAL_ERROR_PATTERNS = re.compile(
    r"unexpected (closing|opening) delimiter"
    r"|this file contains an un-closed delimiter"
    r"|unterminated (string|character|block comment)"
    r"|expected expression, found `\}`"
    r"|aborting due to \d+ previous errors?"
    r"|expected one of .* found end of file",
    re.IGNORECASE,
)


def _is_trivial_error(errors: list[str]) -> bool:
    """Return True if ALL errors are pure syntax failures that web search can't fix."""
    if not errors:
        return False
    return all(_TRIVIAL_ERROR_PATTERNS.search(e) for e in errors)


def search_cargo_fix_via_mcp(
    errors: list[str],
    failed_code: str,
    model_id: str,
) -> Optional[str]:
    """
    Send cargo check errors + failed code to LM Studio's /api/v1/chat endpoint
    with the 'mcp/web-search' MCP server attached.  The model will call the
    web-search tools in mcp.py to find how to fix the Rust compiler errors and
    return a plain-text summary of the fix.

    Returns the model's text response (fix guidance), or None on any error.

    Prerequisites:
      • mcp.py must be registered in LM Studio's mcp.json as "web-search"
      • "Allow calling servers from mcp.json" must be ON in Server Settings
    """
    error_text   = "\n".join(errors)[:2000]
    code_preview = failed_code[:1500]

    prompt = (
        "I have Rust code that fails `cargo check` with the following errors:\n\n"
        f"```\n{error_text}\n```\n\n"
        f"Failed code:\n```rust\n{code_preview}\n```\n\n"
        "Please search the web to find the correct fix for these specific Rust compiler "
        "errors.  Focus on:\n"
        "1. The root cause of each error code shown above.\n"
        "2. The idiomatic Rust way to fix it (correct API usage, missing imports, "
        "wrong types, borrow-checker fixes, etc.).\n"
        "3. A concrete corrected code snippet if available.\n\n"
        "Return a concise, actionable fix summary."
    )

    payload = {
        "model":          model_id,
        "input":          prompt,
        "integrations":   ["mcp/web-search"],
        "context_length": CTX_LENGTH,
        "temperature":    0.1,
    }

    # Only send Authorization if a non-empty token is configured.
    # If LM Studio auth is disabled, omit the header entirely.
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if LM_API_TOKEN:
        headers["Authorization"] = f"Bearer {LM_API_TOKEN}"

    try:
        console.print(f"  [info]🔍 Querying MCP web-search for compiler fix…[/info]")
        r = requests.post(
            f"{API_V1}/chat",
            json=payload,
            headers=headers,
            timeout=REQ_TIMEOUT,
        )
        if r.status_code == 403:
            console.print(
                "  [warn]MCP fix: 403 Forbidden — check two things in LM Studio Server Settings:[/warn]\n"
                "  [warn]  1. 'Allow calling servers from mcp.json' must be ON[/warn]\n"
                "  [warn]  2. Set LM_API_TOKEN env var to match your API token (or disable auth)[/warn]"
            )
            return None
        r.raise_for_status()
        data = r.json()

        # /api/v1/chat returns {"output": [{type, content}, …]}
        output_blocks = data.get("output", [])
        text_parts = [
            b["content"]
            for b in output_blocks
            if b.get("type") == "message" and b.get("content", "").strip()
        ]
        result = "\n\n".join(text_parts).strip()
        if result:
            console.print(f"  [info]✓ MCP returned fix guidance ({len(result)} chars)[/info]")
        return result or None

    except requests.exceptions.ConnectionError:
        console.print("  [warn]MCP fix search: cannot connect to LM Studio API v1.[/warn]")
        return None
    except Exception as e:
        console.print(f"  [warn]MCP fix search error: {e}[/warn]")
        return None


def call_llm_self_fix(
    model_id: str,
    example: dict,
    errors: list[str],
) -> Optional[str]:
    """
    Stage 1 fix — no MCP needed.

    Asks the model to return ONLY corrected Rust code (no JSON).
    The caller merges the fixed code back into the original example,
    preserving prompt, explanation, and all other fields unchanged.
    """
    error_text = "\n".join(errors)[:3000]
    code       = example.get("code", "")

    prompt = (
        "The Rust code below fails `cargo check` with these errors:\n\n"
        f"```\n{error_text}\n```\n\n"
        "Broken code:\n"
        f"```rust\n{code}\n```\n\n"
        "Fix every error shown above."
    )

    console.print("  [info]↻ Stage 1: self-fix using cargo error output…[/info]")
    return call_llm(model_id, prompt, system_prompt=SYSTEM_PROMPT_FIX)


def call_llm_with_fix(
    model_id: str,
    example: dict,
    errors: list[str],
    fix_info: str,
) -> Optional[str]:
    """
    Stage 2 fix — uses MCP web-search results.

    Asks the model to return ONLY corrected Rust code (no JSON).
    The caller merges the fixed code back into the original example,
    preserving prompt, explanation, and all other fields unchanged.
    """
    error_text  = "\n".join(errors)[:2000]
    code        = example.get("code", "")
    fix_preview = fix_info[:2500]

    prompt = (
        "The Rust code below still fails `cargo check`:\n\n"
        f"```\n{error_text}\n```\n\n"
        "Broken code:\n"
        f"```rust\n{code}\n```\n\n"
        "Research on how to fix these errors:\n"
        f"{fix_preview}\n\n"
        "Fix every error shown above."
    )

    console.print("  [info]↻ Stage 2: regenerating with MCP web-search context…[/info]")
    return call_llm(model_id, prompt, system_prompt=SYSTEM_PROMPT_FIX)


def _extract_fix_code(response: str) -> Optional[str]:
    """
    Pull Rust code out of a fix-stage response that should contain code only.
    Handles: raw code, ```rust fences, ``` fences, code preceded by a prose line.
    """
    if not response:
        return None
    # Strip think blocks if present
    response = _THINK_BLOCK.sub("", response).strip()
    # Prefer a fenced rust block
    m = re.search(r"```(?:rust)?\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fall back: if the response looks like raw Rust (starts with use/fn/struct/pub etc.)
    if re.match(r"\s*(use |fn |pub |struct |enum |impl |//|#\[)", response):
        return response.strip()
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

REQUIRED_FIELDS = {"category", "difficulty", "prompt", "code", "explanation"}

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
        "code": "...",
        },               ← spurious — the object is not closed here
        "explanation": "..."

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
    Strips allow attributes, // and /* */ comments, then collapses all
    whitespace runs to a single space.  Used for deduplication hashing.
    """
    code = _ALLOW_ATTR.sub("", code)
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
      • code is not suspiciously short
    """
    global _last_reject_reason
    _last_reject_reason = "json-parse"   # default if we can't even extract JSON

    # Normalise raw LLM output before any extraction or repair attempt:
    # strips BOM, removes null bytes, normalises \r\n / \r → \n so that
    # _apply_escapes sees a consistent newline character inside strings.
    text = _preprocess(text)

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
    code = data.get("code", "").strip()

    # Reject trivially short code (almost certainly a placeholder)
    if len(code) < _MIN_CODE_LEN:
        _last_reject_reason = "too-short"
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


_PYTHON_COMMENT  = re.compile(r"^#(?!\[)\s.*$", re.MULTILINE)
_ALLOW_ATTR      = re.compile(r"^[ \t]*#!?\[allow\([^\)]*\)\][ \t]*\n?", re.MULTILINE)
# Strips backslashes the model sometimes emits before Rust keywords inside JSON strings.
# e.g. "\nimpl" in a JSON string can parse as "\n" + "\impl" (invalid escape → stray backslash).
_STRAY_BACKSLASH = re.compile(r"\\(?=[a-zA-Z_])")
_MISSING_DEBUG   = re.compile(r"E0277|doesn't implement `Debug`")
_STRUCT_ENUM_DEF = re.compile(r"^(\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum)\s)", re.MULTILINE)


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
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        # Process is killed by subprocess.run before re-raising; return a
        # synthetic failed result so callers never see an unhandled exception.
        return subprocess.CompletedProcess(
            cmd, returncode=-1, stdout="",
            stderr=f"[timeout: command exceeded {timeout}s]",
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


def _autopatch_debug(code: str) -> str:
    """
    Insert #[derive(Debug)] before every struct/enum definition that
    doesn't already have Debug in any derive attribute in the preceding
    attribute block (handles multi-attribute structs correctly).
    """
    lines = code.splitlines(keepends=True)
    out: list[str] = []
    for line in lines:
        if _STRUCT_ENUM_DEF.match(line):
            # Walk backward through the accumulated output to check the
            # full preceding attribute block, not just the immediate line.
            already_has_debug = False
            for prev_line in reversed(out):
                stripped = prev_line.strip()
                if not stripped or stripped.startswith("#["):
                    if "Debug" in stripped:
                        already_has_debug = True
                        break
                else:
                    break  # hit a non-attribute line — stop looking
            if not already_has_debug:
                out.append("#[derive(Debug)]\n")
        out.append(line)
    return "".join(out)


def validate(code: str, declared_crates: list[str]) -> dict:
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
        "wrapped_code": code, "errors": [],
        "auto_fixed": False,   # True when cargo fmt / fix rewrote the code
    }

    # Detect compiler version once
    try:
        rv = subprocess.run(["rustc", "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
        result["compiler_ver"] = rv.stdout.strip()
    except Exception:
        pass

    wrapped = wrap_for_check(code)
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

        # ── Auto-fix: cargo fmt + cargo fix + cargo clippy --fix ─────
        # All three tools run before cargo check so stored code is already clean.
        # Each rewrites main.rs in place; we read it back once at the end.
        if AUTO_FIX_CODE:
            # ── Optional rustfmt.toml ───────────────────────────
            # Inject a rustfmt.toml with stable-only options before running
            # cargo fmt.  All options below are on stable rustfmt as of 2026.
            # Note: imports_granularity / group_imports are still UNSTABLE and
            # have known non-idempotency bugs -- deliberately excluded.
            # Set AUTO_FMT_TOML = False at the top of this file to disable.
            if AUTO_FMT_TOML:
                (proj / "rustfmt.toml").write_text(_RUSTFMT_TOML, encoding="utf-8")

            # cargo fmt — reformat unconditionally (always safe)
            _run(["cargo", "fmt"], proj, env=cargo_env)

            # cargo fix -- applies safe automatic lint suggestions.
            # --allow-dirty because the working tree isn't a git repo.
            # --allow-no-vcs for the same reason.
            _run(
                ["cargo", "fix", "--allow-dirty", "--allow-no-vcs"],
                proj, env=cargo_env,
            )

            # cargo clippy --fix -- applies Clippy's machine-applicable lint
            # suggestions automatically (needless clones, redundant closures,
            # map_or patterns, etc.) that cargo fix alone doesn't touch.
            # Runs after cargo fix so both passes compose cleanly.
            # We intentionally ignore the return code -- clippy --fix exits
            # non-zero when it applies suggestions it can't fully verify;
            # cargo check below is the real pass/fail gate.
            _run(
                [
                    "cargo", "clippy", "--fix",
                    "--allow-dirty", "--allow-no-vcs",
                    "--", "-W", "clippy::all",
                ],
                proj, env=cargo_env,
            )

            # Read back whatever the tools produced
            fixed_code = (src / "main.rs").read_text(encoding="utf-8")
            if fixed_code != wrapped:
                result["auto_fixed"] = True
                wrapped = fixed_code
                result["wrapped_code"] = wrapped

        # ── cargo check ──────────────────────────────────────────────
        r = _run(["cargo", "check", "--quiet", "--color=never"], proj, env=cargo_env)
        if r.returncode != 0:
            # ── Auto-patch: insert #[derive(Debug)] for E0277 failures ──
            if _MISSING_DEBUG.search(r.stderr):
                patched = _autopatch_debug(wrapped)
                if patched != wrapped:
                    (src / "main.rs").write_text(patched, encoding="utf-8")
                    r2 = _run(["cargo", "check", "--quiet", "--color=never"], proj, env=cargo_env)
                    if r2.returncode == 0:
                        wrapped = patched
                        result["wrapped_code"] = wrapped
                        # Re-run fmt so the injected #[derive(Debug)] is formatted
                        # consistently and fmt --check passes in the validation step.
                        _run(["cargo", "fmt"], proj, env=cargo_env)
                        patched_fmt = (src / "main.rs").read_text(encoding="utf-8")
                        if patched_fmt != wrapped:
                            wrapped = patched_fmt
                            result["wrapped_code"] = wrapped
                    else:
                        result["errors"] = [r.stderr.strip()]
                        return result
                else:
                    result["errors"] = [r.stderr.strip()]
                    return result
            else:
                result["errors"] = [r.stderr.strip()]
                return result
        result["build"] = True

        # ── cargo fmt --check ────────────────────────────────────────
        r = _run(["cargo", "fmt", "--", "--check"], proj, env=cargo_env)
        result["fmt"] = (r.returncode == 0)

        # ── cargo clippy ─────────────────────────────────────────────
        r = _run(
            ["cargo", "clippy", "--color=never",
             "--", "-D", "warnings"],
            proj,
            env=cargo_env,
        )
        result["clippy"] = (r.returncode == 0)
        # Collect all lint names from clippy/rustc output.
        # Matches both [clippy::lint_name] and plain rustc lints like [unused_mut].
        # NOTE: --quiet is intentionally absent — it suppresses warning output
        # even when the build fails due to -D warnings, leaving clippy_lints empty.
        if not result["clippy"]:
            lints: list[str] = []
            combined = r.stderr + "\n" + r.stdout
            for line in combined.splitlines():
                m = re.search(r"\[([a-z_:]+)\]", line)
                if m and m.group(1) not in ("E", "W"):
                    lints.append(m.group(1))
            result["clippy_lints"]  = sorted(set(lints))
            result["clippy_output"] = (r.stderr + r.stdout).strip()

        # ── MSRV: pattern-based detection (edition 2021 baseline = 1.56) ──
        result["min_rust_version"] = _detect_msrv(wrapped)
        result["msrv"] = True

    return result



# ─────────────────────────────────────────────────────────
#  OUTPUT
# ─────────────────────────────────────────────────────────

_CAT_FAILURE_LOG = Path(__file__).resolve().parent / "category_failures.jsonl"


def log_category_failure(cat: dict, model: str) -> None:
    """Append a JSONL record when MAX_CAT_FAILURES is hit for a category."""
    import datetime
    record = {
        "timestamp":   datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "category":    cat["category"],
        "category_id": cat["id"],
        "model":       model,
    }
    with open(_CAT_FAILURE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
#  DEDUPLICATION HASHES
# ─────────────────────────────────────────────────────────

def _hash_dir(output_dir: Path) -> Path:
    """Return the .hashes sub-directory, creating it if necessary."""
    d = output_dir / ".hashes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _hash_file(output_dir: Path, category: str) -> Path:
    return _hash_dir(output_dir) / f"{safe_filename(category)}.txt"


def example_hash(record: dict) -> str:
    """
    Compute a stable SHA-256 fingerprint for an example.
    We hash code after normalisation so that purely cosmetic differences
    (whitespace, comments, allow-attrs) are ignored.
    """
    code    = _normalize_code(record.get("code", ""))
    payload = code.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_hashes(output_dir: Path, category: str) -> set[str]:
    """Load the set of known hashes for *category* from disk."""
    hf = _hash_file(output_dir, category)
    if not hf.exists():
        return set()
    with open(hf, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def save_hash(output_dir: Path, category: str, h: str) -> None:
    """Append a single hash to the category hash file."""
    with open(_hash_file(output_dir, category), "a", encoding="utf-8") as f:
        f.write(h + "\n")


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

        # Load deduplication hashes for this category.
        # Also seed from any existing JSONL records that pre-date the hash files.
        known_hashes: set[str] = load_hashes(output_dir, cat_name)
        if out_path.exists() and not known_hashes:
            with open(out_path, encoding="utf-8") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line:
                        try:
                            _h = example_hash(json.loads(_line))
                            known_hashes.add(_h)
                        except Exception:
                            pass
            # Persist the seeded hashes so future runs skip the JSONL scan
            if known_hashes:
                hf = _hash_file(output_dir, cat_name)
                with open(hf, "w", encoding="utf-8") as _f:
                    _f.write("\n".join(known_hashes) + "\n")

        generated_this_cat = 0
        attempt = 0
        consecutive_failures = 0
        total_cat_fail       = 0   # total failures this category — never resets on success

        while generated_this_cat < remaining:
            attempt += 1

            console.print(f"  [dim]attempt {attempt}…[/dim]")

            def _fail(reason: str) -> None:
                """Increment fail counters; caller checks consecutive_failures to bail."""
                nonlocal total_fail, consecutive_failures, total_cat_fail
                total_fail += 1
                consecutive_failures += 1
                total_cat_fail += 1
                console.print(
                    f"  [dim]↳ run total: [ok]{total_ok}[/ok] ok  "
                    f"[fail]{total_fail}[/fail] failed[/dim]"
                )

            def _bail_if_stuck() -> bool:
                """Return True (and print a warning) when either failure limit is hit."""
                if consecutive_failures >= MAX_FAILURES:
                    console.print(
                        f"  [warn]{MAX_FAILURES} consecutive failures — "
                        f"skipping category.[/warn]"
                    )
                    log_category_failure(cat, model)
                    return True
                if total_cat_fail >= MAX_CAT_FAILURES:
                    console.print(
                        f"  [warn]{MAX_CAT_FAILURES} total failures this category — "
                        f"skipping category.[/warn]"
                    )
                    log_category_failure(cat, model)
                    return True
                return False

            # ── LLM call ─────────────────────────────────────────────
            user_prompt = build_user_prompt(cat, attempt=attempt)
            raw = call_llm(model, user_prompt)
            if raw is None:
                _fail("llm-error")
                if _bail_if_stuck(): break
                continue

            # ── JSON parse ────────────────────────────────────────────
            example = parse_example(raw)
            if example is None:
                reason = _last_reject_reason
                console.print(f"  [warn]JSON parse failed (attempt {attempt})  [{reason}][/warn]")
                _fail(reason)
                if _bail_if_stuck(): break
                continue

            code = _STRAY_BACKSLASH.sub("", _ALLOW_ATTR.sub("", _PYTHON_COMMENT.sub("", example.get("code", "")))).strip().lstrip("\n")
            if not code:
                console.print(f"  [warn]Empty code (attempt {attempt})[/warn]")
                _fail("empty-code")
                if _bail_if_stuck(): break
                continue

            # ── Cargo validation ──────────────────────────────────────
            console.print(f"  [dim]Validating with cargo…[/dim]")
            try:
                val = validate(code, example.get("crates", []))
            except Exception as e:
                console.print(f"  [fail]✗ validate() crashed (attempt {attempt}): {e}[/fail]")
                _fail("validate-crash")
                if _bail_if_stuck(): break
                continue

            if not val["build"]:
                err_preview = val["errors"][0][:120] if val["errors"] else "unknown"
                console.print(
                    f"  [fail]✗ cargo check failed (attempt {attempt})[/fail]  "
                    f"[dim]{err_preview}[/dim]"
                )

                # ── 3-stage fix cascade ────────────────────────────────
                # Stage 1: send full JSONL + cargo output back — model
                #          self-corrects from rustc's own error messages.
                # Stage 2: MCP web search for the error, then regenerate
                #          with the full JSONL + errors + web fix info.
                # Stage 3: give up, log, count as failure.

                def _try_fix(fix_raw: Optional[str], label: str) -> bool:
                    """Extract fixed code, validate it, merge into original example.
                    All fields other than code are preserved from the original."""
                    nonlocal code, val, example
                    fix_code = _extract_fix_code(fix_raw or "")
                    if not fix_code:
                        return False
                    fix_code = _STRAY_BACKSLASH.sub("", _ALLOW_ATTR.sub(
                        "", _PYTHON_COMMENT.sub("", fix_code)
                    )).strip().lstrip("\n")
                    if not fix_code:
                        return False
                    try:
                        fix_val = validate(fix_code, example.get("crates", []))
                    except Exception as _e:
                        console.print(f"  [warn]{label} validate() error: {_e}[/warn]")
                        return False
                    if fix_val["build"]:
                        console.print(f"  [ok]✓ {label} succeeded[/ok]")
                        code    = fix_code
                        val     = fix_val
                        example = {**example, "code": fix_code}
                        return True
                    console.print(f"  [warn]{label} still failed cargo check[/warn]")
                    return False

                _fixed = False

                # Stage 1 — self-fix from cargo output (no MCP), up to SELF_FIX_RETRIES attempts
                for _retry in range(1, SELF_FIX_RETRIES + 1):
                    label = f"Stage 1 self-fix (attempt {_retry}/{SELF_FIX_RETRIES})"
                    _fixed = _try_fix(
                        call_llm_self_fix(model, example, val["errors"]),
                        label,
                    )
                    if _fixed:
                        break

                # Stage 2 — MCP web search + regenerate (skip when USE_MCP is False or
                # errors are pure syntax failures that web search cannot help with)
                if not _fixed and USE_MCP and not _is_trivial_error(val["errors"]):
                    fix_info = search_cargo_fix_via_mcp(
                        val["errors"], val["wrapped_code"], model
                    )
                    if fix_info:
                        _fixed = _try_fix(
                            call_llm_with_fix(model, example, val["errors"], fix_info),
                            "Stage 2 MCP-fix",
                        )

                # Stage 3 — give up
                if not _fixed:
                    _fail("cargo-check")
                    if _bail_if_stuck(): break
                    continue
                # _fixed == True → fall through to build the output record

            # ── Build the output record ───────────────────────────────
            record = {
                "id":               str(uuid.uuid4()),
                "schema_variant":   "good_code",
                "source":           "opensource",
                "category_id":      cat["id"],
                "category":         cat_name,
                "difficulty":       example.get("difficulty", cat.get("difficulty", "intermediate")),
                "prompt":           example["prompt"],
                "code":             code,
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
                    "auto_fixed":    val.get("auto_fixed", False),
                },
                "verified_at":       str(int(time.time())),
                "compiler_ver":      val["compiler_ver"],
                "min_rust_version":  val["min_rust_version"],
                "has_unsafe":        val["has_unsafe"],
                "license":           LICENSE,
            }

            # ── Deduplication check ───────────────────────────────────
            h = example_hash(record)
            if h in known_hashes:
                console.print(f"  [warn]Duplicate example detected — skipping (attempt {attempt})[/warn]")
                _fail("duplicate")
                if _bail_if_stuck(): break
                continue

            write_example(out_path, record)
            save_hash(output_dir, cat_name, h)
            known_hashes.add(h)
            generated_this_cat  += 1
            total_ok            += 1
            consecutive_failures   = 0  # successful example clears the streak
            # total_cat_fail is intentionally NOT reset — it is a hard budget
            # for the whole category so a model that barely scrapes by can't
            # loop forever on a category it has little knowledge of.
            # ── Cooldown ──────────────────────────────────────────────
            if cooldown_every > 0 and total_ok % cooldown_every == 0:
                console.print(
                    f"\n[warn]Cooldown after {total_ok} generations — "
                    f"unloading model for {cooldown_secs}s…[/warn]"
                )
                unload_model(instance_id)
                _active_iid = None   # clear immediately so signal handler won't try a ghost unload
                for remaining_s in range(cooldown_secs, 0, -5):
                    console.print(f"  [dim]{remaining_s}s remaining…[/dim]", end="\r")
                    time.sleep(min(5, remaining_s))
                console.print(f"  [dim]Reloading model…[/dim]" + " " * 20, end="\r")
                new_iid = load_model(model) if args.model else None
                if new_iid:
                    instance_id = new_iid
                    _active_iid = new_iid
                else:
                    # Load failed — keep running with the existing model id but
                    # leave _active_iid as None so we don't try to unload a ghost.
                    console.print(f"[warn]Model reload failed — continuing without reloading.[/warn]")
                console.print(f"[ok]Cooldown done. Resuming.[/ok]" + " " * 20)

            clippy_badge = "[ok]✓ clippy[/ok]" if val["clippy"] else "[warn]~ clippy warn[/warn]"
            fmt_badge    = "[ok]✓ fmt[/ok]"    if val["fmt"]    else "[warn]~ fmt[/warn]"
            fix_badge    = "[info]~ auto-fixed[/info]" if val.get("auto_fixed") else ""
            msrv_info    = f"  msrv={val['min_rust_version']}" if val["msrv"] else ""
            console.print(
                f"  [ok]✓ build[/ok]  {clippy_badge}  {fmt_badge}"
                + (f"  {fix_badge}" if fix_badge else "")
                + f"{msrv_info}"
                f"  [dim]{cat_name} #{existing + generated_this_cat}[/dim]"
                f"  [dim]↳ run total: [ok]{total_ok}[/ok] ok  "
                f"[fail]{total_fail}[/fail] failed[/dim]"
            )

        console.print()

    # ── Continuous weighted loop ───────────────────────────────────────────────
    # After the initial pass fills every category to its target_entries, if
    # --continuous is set we keep generating indefinitely.  Categories are chosen
    # randomly weighted by their `weight` field (falls back to `target_entries`,
    # then 1).  This naturally gives important categories more examples over time
    # while ensuring every category grows.
    if args.continuous and not args.category:
        console.print(
            "\n[warn]Initial pass complete — entering continuous weighted mode. "
            "Press Ctrl+C to stop.[/warn]\n"
        )

        # Build a stable weight list (recalculated once per outer iteration so
        # editing the JSONL mid-run takes effect on the next pass through here,
        # though in practice we never reload the file — weights are fixed at
        # startup which is fine for a weeks-long run).
        cat_weights = [
            float(c.get("weight", c.get("target_entries", 1)) or 1)
            for c in categories
        ]

        # Per-category consecutive-failure and total-failure counters survive
        # across the continuous loop so a persistently broken category is skipped.
        cont_consec_fail: dict[str, int] = {c["category"]: 0 for c in categories}
        cont_total_fail:  dict[str, int] = {c["category"]: 0 for c in categories}

        def _try_fix_cont(fix_raw: Optional[str], label: str) -> bool:
            """Extract fixed code, validate it, merge into the current example.
            Defined once outside the loop — captures code/val/example via nonlocal
            on each invocation so the closure always sees the live variables."""
            nonlocal code, val, example
            fix_code = _extract_fix_code(fix_raw or "")
            if not fix_code:
                return False
            fix_code = _STRAY_BACKSLASH.sub("", _ALLOW_ATTR.sub(
                "", _PYTHON_COMMENT.sub("", fix_code)
            )).strip().lstrip("\n")
            if not fix_code:
                return False
            try:
                fix_val = validate(fix_code, example.get("crates", []))
            except Exception:
                return False
            if fix_val["build"]:
                console.print(f"  [ok]✓ {label} succeeded[/ok]")
                code    = fix_code
                val     = fix_val
                example = {**example, "code": fix_code}
                return True
            console.print(f"  [warn]{label} still failed cargo check[/warn]")
            return False

        continuous_pass = 0
        while True:
            continuous_pass += 1

            # Weighted random pick — respects keyboard interrupt
            cat = random.choices(categories, weights=cat_weights, k=1)[0]
            cat_name = cat["category"]

            # Skip categories that have burned through their failure budgets
            if cont_consec_fail[cat_name] >= MAX_FAILURES:
                console.print(
                    f"[dim]Skipping {cat_name!r} "
                    f"({MAX_FAILURES} consecutive failures in continuous mode)[/dim]"
                )
                log_category_failure(cat, model)
                # Guard: if every category is locked out, exit rather than spin forever
                active = [
                    c["category"] for c in categories
                    if cont_consec_fail[c["category"]] < MAX_FAILURES
                    and cont_total_fail[c["category"]] < MAX_CAT_FAILURES
                ]
                if not active:
                    console.print(
                        "[fail]All categories are locked out by failure limits "
                        "— exiting continuous mode.[/fail]"
                    )
                    break
                continue
            if cont_total_fail[cat_name] >= MAX_CAT_FAILURES:
                console.print(
                    f"[dim]Skipping {cat_name!r} "
                    f"({MAX_CAT_FAILURES} total failures in continuous mode)[/dim]"
                )
                log_category_failure(cat, model)
                # Same guard as above
                active = [
                    c["category"] for c in categories
                    if cont_consec_fail[c["category"]] < MAX_FAILURES
                    and cont_total_fail[c["category"]] < MAX_CAT_FAILURES
                ]
                if not active:
                    console.print(
                        "[fail]All categories are locked out by failure limits "
                        "— exiting continuous mode.[/fail]"
                    )
                    break
                continue

            out_path = output_path(output_dir, cat_name)

            # Count existing so the summary line is accurate
            existing_cont = 0
            if out_path.exists():
                with open(out_path, encoding="utf-8") as _f:
                    existing_cont = sum(1 for ln in _f if ln.strip())

            known_hashes_cont: set[str] = load_hashes(output_dir, cat_name)

            console.print(
                f"  [dim][continuous #{continuous_pass}][/dim]  "
                f"[cat]{cat_name}[/cat]  "
                f"[dim]({existing_cont} existing)[/dim]"
            )

            user_prompt = build_user_prompt(cat, attempt=continuous_pass)
            raw = call_llm(model, user_prompt)
            if raw is None:
                cont_consec_fail[cat_name] += 1
                cont_total_fail[cat_name]  += 1
                total_fail += 1
                continue

            example = parse_example(raw)
            if example is None:
                cont_consec_fail[cat_name] += 1
                cont_total_fail[cat_name]  += 1
                total_fail += 1
                continue

            code = _STRAY_BACKSLASH.sub("", _ALLOW_ATTR.sub(
                "", _PYTHON_COMMENT.sub("", example.get("code", ""))
            )).strip().lstrip("\n")
            if not code:
                cont_consec_fail[cat_name] += 1
                cont_total_fail[cat_name]  += 1
                total_fail += 1
                continue

            try:
                val = validate(code, example.get("crates", []))
            except Exception as _e:
                console.print(f"  [fail]validate() crashed: {_e}[/fail]")
                cont_consec_fail[cat_name] += 1
                cont_total_fail[cat_name]  += 1
                total_fail += 1
                continue

            if not val["build"]:
                err_preview = val["errors"][0][:120] if val["errors"] else "unknown"
                console.print(
                    f"  [fail]✗ cargo check failed[/fail]  [dim]{err_preview}[/dim]"
                )
                # ── 3-stage fix cascade (mirrors initial pass) ────
                _fixed_cont = False

                # Stage 1 — self-fix from cargo output (no MCP), up to SELF_FIX_RETRIES attempts
                for _retry in range(1, SELF_FIX_RETRIES + 1):
                    label = f"Stage 1 self-fix (attempt {_retry}/{SELF_FIX_RETRIES})"
                    _fixed_cont = _try_fix_cont(
                        call_llm_self_fix(model, example, val["errors"]),
                        label,
                    )
                    if _fixed_cont:
                        break

                # Stage 2 — MCP web search + regenerate (skip when USE_MCP is False or
                # errors are pure syntax failures that web search cannot help with)
                if not _fixed_cont and USE_MCP and not _is_trivial_error(val["errors"]):
                    fix_info = search_cargo_fix_via_mcp(
                        val["errors"], val["wrapped_code"], model
                    )
                    if fix_info:
                        _fixed_cont = _try_fix_cont(
                            call_llm_with_fix(model, example, val["errors"], fix_info),
                            "Stage 2 MCP-fix",
                        )

                if not _fixed_cont:
                    cont_consec_fail[cat_name] += 1
                    cont_total_fail[cat_name]  += 1
                    total_fail += 1
                    continue

            # Hash before building the full record (same key used by example_hash internally).
            h = example_hash({"code": code})
            if h in known_hashes_cont:
                console.print(f"  [warn]Duplicate — skipping[/warn]")
                cont_consec_fail[cat_name] += 1
                cont_total_fail[cat_name]  += 1
                total_fail += 1
                continue

            record = {
                "id":               str(uuid.uuid4()),
                "schema_variant":   "good_code",
                "source":           "opensource",
                "category_id":      cat["id"],
                "category":         cat_name,
                "difficulty":       example.get("difficulty", cat.get("difficulty", "intermediate")),
                "prompt":           example["prompt"],
                "code":             code,
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
                    "auto_fixed":    val.get("auto_fixed", False),
                },
                "verified_at":       str(int(time.time())),
                "compiler_ver":      val["compiler_ver"],
                "min_rust_version":  val["min_rust_version"],
                "has_unsafe":        val["has_unsafe"],
                "license":           LICENSE,
            }

            write_example(out_path, record)
            save_hash(output_dir, cat_name, h)
            known_hashes_cont.add(h)
            total_ok += 1
            cont_consec_fail[cat_name] = 0   # success resets consecutive streak

            clippy_badge = "[ok]✓ clippy[/ok]" if val["clippy"] else "[warn]~ clippy warn[/warn]"
            fmt_badge    = "[ok]✓ fmt[/ok]"    if val["fmt"]    else "[warn]~ fmt[/warn]"
            console.print(
                f"  [ok]✓ build[/ok]  {clippy_badge}  {fmt_badge}"
                f"  [dim]{cat_name} #{existing_cont + 1}[/dim]"
                f"  [dim]↳ run total: [ok]{total_ok}[/ok] ok  "
                f"[fail]{total_fail}[/fail] failed[/dim]"
            )

            # Cooldown applies in continuous mode too
            if cooldown_every > 0 and total_ok % cooldown_every == 0:
                console.print(
                    f"\n[warn]Cooldown after {total_ok} generations — "
                    f"unloading model for {cooldown_secs}s…[/warn]"
                )
                unload_model(instance_id)
                _active_iid = None
                for remaining_s in range(cooldown_secs, 0, -5):
                    console.print(f"  [dim]{remaining_s}s remaining…[/dim]", end="\r")
                    time.sleep(min(5, remaining_s))
                console.print(f"  [dim]Reloading model…[/dim]" + " " * 20, end="\r")
                new_iid = load_model(model) if args.model else None
                if new_iid:
                    instance_id = new_iid
                    _active_iid = new_iid
                else:
                    console.print(f"[warn]Model reload failed — continuing.[/warn]")
                console.print(f"[ok]Cooldown done. Resuming.[/ok]" + " " * 20)

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
    p.add_argument(
        "--continuous",
        action="store_true",
        help=(
            "Never stop after filling initial targets. "
            "Keep generating indefinitely using weighted category selection "
            "(weight = category\'s target_entries, or \'weight\' field if present). "
            "Stop with Ctrl+C."
        ),
    )
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()