#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  validate_dataset.py  —  Rust JSONL Dataset Batch Validator      ║
║  Runs cargo check / clippy / fmt on every record's fixed_code,   ║
║  writes passing records to 50 MB-capped output JSONL files.      ║
╚══════════════════════════════════════════════════════════════════╝

For each record in every .jsonl file in --input-dir:
  1. Extract fixed_code; try several wrapping strategies for snippets
  2. Run: cargo check → cargo fmt --check → cargo clippy
  3. Write passing records to output/ (rotated at 50 MB per file)
  4. Log failures with details to validation_failures.log

Usage:
    python validate_dataset.py --input-dir raw/ --output-dir validated/
    python validate_dataset.py --input-dir raw/ --output-dir validated/ --workers 8
    python validate_dataset.py --input-dir raw/ --output-dir validated/ --limit 500
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────

EDITION           = "2021"
LICENSE           = "Apache-2.0"
MAX_FILE_BYTES    = 50 * 1024 * 1024   # 50 MB per output file
CARGO_TIMEOUT     = 180                 # seconds per cargo invocation
CARGO_ADD_TIMEOUT = 120                 # seconds for `cargo add` (network)
MAX_ADD_ROUNDS    = 8                   # max rounds of "add missing crate → retry"
CRATE_CACHE_FILE  = Path("crate_add_cache.json")  # persists across runs
SHARED_TARGET_DIR = Path(__file__).parent / "cargo_target"  # shared build cache

# Target output schema — all records are normalised to this shape (null = unknown)
_SCHEMA_DEFAULTS: dict = {
    "category":         None,
    "difficulty":       None,
    "prompt":           None,
    "broken_code":      None,
    "error_message":    None,
    "error_code":       None,
    "fixed_code":       None,
    "explanation":      None,
    "concepts":         None,
    "crates":           [],
    "edition":          None,
    "source_model":     None,
    "validation":       None,
    "verified_at":      None,
    "compiler_ver":     None,
    "min_rust_version": None,
    "has_unsafe":       None,
    "license":          None,
}

# ─────────────────────────────────────────────────────────
#  CRATE MAP  —  known crates we can wire into Cargo.toml
# ─────────────────────────────────────────────────────────

CRATE_MAP: dict[str, str] = {
    # Async runtime
    "tokio":              'tokio = { version = "1", features = ["full"] }',
    "tokio_util":         'tokio-util = { version = "0.7", features = ["full"] }',
    "tokio_stream":       'tokio-stream = "0.1"',
    "futures":            'futures = "0.3"',
    "futures_util":       'futures-util = "0.3"',
    "async_trait":        'async-trait = "0.1"',
    "pin_project":        'pin-project = "1"',
    "pin_project_lite":   'pin-project-lite = "0.2"',
    # Serialization
    "serde":              'serde = { version = "1", features = ["derive"] }',
    "serde_json":         'serde_json = "1"',
    "json_canon":         'json-canon = "0.1"',
    "serde_derive":       'serde_derive = "1"',
    "toml":               'toml = "0.8"',
    "csv":                'csv = "1"',
    # Error handling
    "thiserror":          'thiserror = "2"',
    "anyhow":             'anyhow = "1"',
    # Concurrency / collections
    "rayon":              'rayon = "1"',
    "crossbeam":          'crossbeam = "0.8"',
    "crossbeam_channel":  'crossbeam-channel = "0.5"',
    "crossbeam_utils":    'crossbeam-utils = "0.8"',
    "parking_lot":        'parking_lot = "0.12"',
    "dashmap":            'dashmap = "6"',
    "once_cell":          'once_cell = "1"',
    "lazy_static":        'lazy_static = "1"',
    "smallvec":           'smallvec = "1"',
    "ahash":              'ahash = "0.8"',
    "hashbrown":          'hashbrown = "0.14"',
    "indexmap":           'indexmap = "2"',
    "typed_arena":        'typed-arena = "2"',
    # Random
    "rand":               'rand = "0.8"',
    "rand_core":          'rand_core = "0.6"',
    # Text / regex
    "regex":              'regex = "1"',
    "memchr":             'memchr = "2"',
    "nom":                'nom = "7"',
    "pest":               'pest = "2"',
    "pest_derive":        'pest_derive = "2"',
    # Logging / tracing
    "tracing":            'tracing = "0.1"',
    "tracing_subscriber": 'tracing-subscriber = { version = "0.3", features = ["env-filter"] }',
    "log":                'log = "0.4"',
    "env_logger":         'env_logger = "0.11"',
    # Derive macros
    "derive_more":        'derive_more = { version = "1", features = ["full"] }',
    "strum":              'strum = { version = "0.26", features = ["derive"] }',
    "strum_macros":       'strum = { version = "0.26", features = ["derive"] }',
    "num_traits":         'num-traits = "0.2"',
    "num_derive":         'num-derive = "0.4"',
    "bitflags":           'bitflags = "2"',
    # CLI
    "clap":               'clap = { version = "4", features = ["derive"] }',
    # Web / networking
    "tower":              'tower = { version = "0.4", features = ["full"] }',
    "tower_service":      'tower = { version = "0.4", features = ["full"] }',
    "tower_layer":        'tower = { version = "0.4", features = ["full"] }',
    "hyper":              'hyper = { version = "1", features = ["full"] }',
    "axum":               'axum = "0.7"',
    "reqwest":            'reqwest = { version = "0.12", features = ["json"] }',
    "actix_web":          'actix-web = "4"',
    "warp":               'warp = "0.3"',
    "rocket":             'rocket = "0.5"',
    "tonic":              'tonic = "0.11"',
    "prost":              'prost = "0.12"',
    # Bytes / encoding
    "bytes":              'bytes = "1"',
    "byteorder":          'byteorder = "1"',
    "base64":             'base64 = "0.22"',
    "hex":                'hex = "0.4"',
    "url":                'url = "2"',
    "flate2":             'flate2 = "1"',
    "tar":                'tar = "0.4"',
    "zip":                'zip = "2"',
    # Iterators / numeric
    "itertools":          'itertools = "0.13"',
    # Date / time / id
    "chrono":             'chrono = "0.4"',
    "uuid":               'uuid = { version = "1", features = ["v1", "v3", "v4", "v5", "v6", "v7", "v8", "serde"] }',
    # System
    "libc":               'libc = "0.2"',
    "tempfile":           'tempfile = "3"',
    "walkdir":            'walkdir = "1"',
    "glob":               'glob = "0.3"',
    "dirs":               'dirs = "5"',
    "directories":        'directories = "5"',
    "config":             'config = "0.14"',
    "dotenv":             'dotenv = "0.15"',
    # Database
    "rusqlite":           'rusqlite = { version = "0.31", features = ["bundled"] }',
    "sqlx":               'sqlx = { version = "0.7", features = ["sqlite", "runtime-tokio"] }',
    # Graphics / GUI (compile-check only, heavy but valid)
    "image":              'image = "0.25"',
    "nalgebra":           'nalgebra = "0.32"',
    "glam":               'glam = "0.27"',
    # Testing
    "criterion":          'criterion = "0.5"',
    "proptest":           'proptest = "1"',
    "quickcheck":         'quickcheck = "1"',
    "mockall":            'mockall = "0.12"',
    "static_assertions":  'static_assertions = "1"',
    "insta":              'insta = "1"',
    "assert2":            'assert2 = "0.3"',
    # Hashing / crypto
    "siphasher":          'siphasher = "1"',
    "base62":             'base62 = "2"',
    "md5":                'md5 = "0.7"',
    "sha2":               'sha2 = "0.10"',
    "sha1":               'sha1 = "0.10"',
    "blake2":             'blake2 = "0.10"',
    "blake3":             'blake3 = "1"',
    "hmac":               'hmac = "0.12"',
    "pbkdf2":             'pbkdf2 = "0.12"',
    "argon2":             'argon2 = "0.5"',
    "bcrypt":             'bcrypt = "0.15"',
    "aes":                'aes = "0.8"',
    "rsa":                'rsa = "0.9"',
    "ed25519_dalek":      'ed25519-dalek = "2"',
    "curve25519_dalek":   'curve25519-dalek = "4"',
    "x25519_dalek":       'x25519-dalek = "2"',
    "ring":               'ring = "0.17"',
    "rustls":             'rustls = "0.23"',
    "native_tls":         'native-tls = "0.2"',
    "openssl":            'openssl = "0.10"',
    # HTTP / networking extras
    "http":               'http = "1"',
    "http_body":          'http-body = "1"',
    "http_body_util":     'http-body-util = "0.1"',
    "headers":            'headers = "0.4"',
    "hyper_util":         'hyper-util = "0.1"',
    "mime":               'mime = "0.3"',
    "tower_http":         'tower-http = { version = "0.5", features = ["full"] }',
    "h2":                 'h2 = "0.4"',
    "quinn":              'quinn = "0.11"',
    "socket2":            'socket2 = "0.5"',
    "tokio_tungstenite":  'tokio-tungstenite = "0.21"',
    # Proc-macro / codegen
    "proc_macro2":        'proc-macro2 = "1"',
    "quote":              'quote = "1"',
    "syn":                'syn = { version = "2", features = ["full"] }',
    "darling":            'darling = "0.20"',
    "paste":              'paste = "1"',
    # Data / formats
    "arrow":              'arrow = "53"',
    "arrow2":             'arrow2 = "0.18"',
    "parquet":            'parquet = "53"',
    "datafusion":         'datafusion = "37"',
    "polars":             'polars = { version = "0.39", features = ["lazy"] }',
    "serde_yaml":         'serde_yaml = "0.9"',
    "serde_cbor":         'serde_cbor = "0.11"',
    "serde_urlencoded":   'serde_urlencoded = "0.7"',
    "bincode":            'bincode = "1"',
    "postcard":           'postcard = { version = "1", features = ["alloc"] }',
    "rmp":                'rmp = "0.8"',
    "rmp_serde":          'rmp-serde = "1"',
    "prost":              'prost = "0.12"',
    "prost_types":        'prost-types = "0.12"',
    # Byte / bit manipulation
    "byte_slice_cast":    'byte-slice-cast = "1"',
    "bytemuck":           'bytemuck = "1"',
    "bitvec":             'bitvec = "1"',
    "bit_vec":            'bit-vec = "0.6"',
    "bytes_utils":        'bytes-utils = "0.1"',
    "zerocopy":           'zerocopy = { version = "0.7", features = ["derive"] }',
    # Concurrency extras
    "arc_swap":           'arc-swap = "1"',
    "flume":              'flume = "0.11"',
    "kanal":              'kanal = "0.1"',
    "tokio_rayon":        'tokio-rayon = "2"',
    "async_channel":      'async-channel = "2"',
    "async_lock":         'async-lock = "3"',
    "event_listener":     'event-listener = "5"',
    # Derive / macro extras
    "derive_builder":     'derive_builder = "0.20"',
    "typed_builder":      'typed-builder = "0.18"',
    "bon":                'bon = "2"',
    "getset":             'getset = "0.1"',
    "delegate":           'delegate = "0.12"',
    "ambassador":         'ambassador = "0.4"',
    "newtype_derive":     'newtype-derive = "0.1"',
    "shrinkwraprs":       'shrinkwraprs = "0.3"',
    "nutype":             'nutype = { version = "0.5", features = ["serde"] }',
    "smart_default":      'smart-default = "0.7"',
    "educe":              'educe = "0.6"',
    "variantly":          'variantly = "0.2"',
    "superstruct":        'superstruct = "0.7"',
    "deriving_via":       'deriving_via = "2"',
    "typesize":           'typesize = "0.1"',
    # String / text extras
    "smol_str":           'smol-str = "0.2"',
    "compact_str":        'compact_str = "0.8"',
    "aho_corasick":       'aho-corasick = "1"',
    "fancy_regex":        'fancy-regex = "0.13"',
    "unicode_segmentation": 'unicode-segmentation = "1"',
    "unicode_normalization": 'unicode-normalization = "0.1"',
    "unicode_width":      'unicode-width = "0.1"',
    "unicode_xid":        'unicode-xid = "0.2"',
    "percent_encoding":   'percent-encoding = "2"',
    "form_urlencoded":    'form_urlencoded = "1"',
    "chardet":            'chardet = "0.2"',
    # CLI / terminal
    "indicatif":          'indicatif = "0.17"',
    "console":            'console = "0.15"',
    "dialoguer":          'dialoguer = "0.11"',
    "termcolor":          'termcolor = "1"',
    "colored":            'colored = "2"',
    "crossterm":          'crossterm = "0.27"',
    "ratatui":            'ratatui = "0.26"',
    # Config / env
    "dotenvy":            'dotenvy = "0.15"',
    "figment":            'figment = { version = "0.10", features = ["toml", "env"] }',
    "confique":           'confique = { version = "0.2", features = ["toml"] }',
    # File system
    "notify":             'notify = "6"',
    "ignore":             'ignore = "0.4"',
    "jwalk":              'jwalk = "0.8"',
    "camino":             'camino = "1"',
    # Numeric / math
    "num":                'num = "0.4"',
    "num_bigint":         'num-bigint = "0.4"',
    "num_rational":       'num-rational = "0.4"',
    "num_complex":        'num-complex = "0.4"',
    "num_integer":        'num-integer = "0.1"',
    "ordered_float":      'ordered-float = "4"',
    "rust_decimal":       'rust_decimal = "1"',
    "bigdecimal":         'bigdecimal = "0.4"',
    "statrs":             'statrs = "0.17"',
    "ndarray":            'ndarray = "0.15"',
    "approx":             'approx = "0.5"',
    # Time
    "time":               'time = "0.3"',
    "jiff":               'jiff = "0.1"',
    # ID generation
    "ulid":               'ulid = "1"',
    "nanoid":             'nanoid = "0.4"',
    "ksuid":              'ksuid = "0.1"',
    # Serialization extras
    "serde_with":         'serde_with = "3"',
    "serde_tokenstream":  'serde_tokenstream = "0.2"',
    "serde_repr":         'serde_repr = "0.1"',
    # Observability
    "opentelemetry":      'opentelemetry = "0.23"',
    "metrics":            'metrics = "0.23"',
    "miette":             'miette = { version = "7", features = ["fancy"] }',
    "color_eyre":         'color-eyre = "0.6"',
    "eyre":               'eyre = "0.6"',
    # State machine / actor
    "tokio_actor":        'tokio-actor = "0.2"',
    "actix":              'actix = "0.13"',
    # DB extras
    "redis":              'redis = "0.26"',
    "mongodb":            'mongodb = "3"',
    "sea_orm":            'sea-orm = "1"',
    "diesel":             'diesel = { version = "2", features = ["sqlite"] }',
    "surrealdb":          'surrealdb = "1"',
    # Misc utilities
    "strum_macros":       'strum = { version = "0.26", features = ["derive"] }',
    "humantime":          'humantime = "2"',
    "either":             'either = "1"',
    "tap":                'tap = "1"',
    "scopeguard":         'scopeguard = "1"',
    "fragile":            'fragile = "2"',
    "owning_ref":         'owning-ref = "0.4"',
    "tinyvec":            'tinyvec = { version = "1", features = ["alloc"] }',
    "arrayvec":           'arrayvec = "0.7"',
    "thin_vec":           'thin-vec = "0.2"',
    "vec_map":            'vec_map = "0.8"',
    "fixedbitset":        'fixedbitset = "0.4"',
    "petgraph":           'petgraph = "0.6"',
    "id_arena":           'id-arena = "2"',
    "slotmap":            'slotmap = "1"',
    "slab":               'slab = "0.4"',
    "generational_arena": 'generational-arena = "0.2"',
}

# ─────────────────────────────────────────────────────────
#  DYNAMIC CRATE CACHE
#  Maps `code_name` (underscore) → cargo package name (hyphenated)
#  or None if cargo add failed. Persisted to disk between runs.
# ─────────────────────────────────────────────────────────

_cache_lock: threading.Lock = threading.Lock()
_crate_cache: dict[str, Optional[str]] = {}   # None = known-unfetchable
_cargo_add_failures: dict[str, str] = {}       # pkg → reason cargo add failed


def _load_crate_cache(path: Path = CRATE_CACHE_FILE) -> None:
    global _crate_cache
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                _crate_cache = json.load(f)
        except Exception:
            _crate_cache = {}


def _save_crate_cache(path: Path = CRATE_CACHE_FILE) -> None:
    with _cache_lock:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(_crate_cache, f, indent=2, sort_keys=True)
        except Exception:
            pass


def _record_cache(code_name: str, pkg_name: Optional[str]) -> None:
    with _cache_lock:
        _crate_cache[code_name] = pkg_name
    _save_crate_cache()


_STD_CRATES = frozenset({"std", "core", "alloc", "proc_macro", "test"})

# ─────────────────────────────────────────────────────────
#  REGEX PATTERNS
# ─────────────────────────────────────────────────────────

_HAS_MAIN       = re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+main\s*\(", re.MULTILINE)
_ASYNC_MAIN     = re.compile(r"^\s*(?:pub\s+)?async\s+fn\s+main\s*\(", re.MULTILINE)
_TOKIO_ATTR     = re.compile(r"#\[tokio::main\]")
_TOP_LEVEL      = re.compile(
    r"^(?:pub\s+)?(?:async\s+)?(?:fn|struct|enum|trait|type|impl|const|static|mod|use|extern)\s",
    re.MULTILINE,
)
_PYTHON_COMMENT = re.compile(r"^#(?!\[)\s.*$", re.MULTILINE)
# `mod foo;` → convert to inline `mod foo {}` so we don't need separate files
_MOD_FILE_DECL  = re.compile(r"^(\s*(?:pub(?:\(crate\))?\s+)?mod\s+\w+)\s*;", re.MULTILINE)
# Heuristic: does text look like Rust source?
_LOOKS_LIKE_RUST = re.compile(
    r"^\s*(?:fn\s|struct\s|enum\s|impl\s|trait\s|pub\s|use\s|mod\s|let\s|const\s|"
    r"static\s|type\s|#\[|//|/\*)",
    re.MULTILINE,
)
# #[cfg(test)] + mod tests block — strip it for snippets to avoid unresolved imports
_TEST_MOD = re.compile(
    r"\n?\s*#\[cfg\(test\)\]\s*\n\s*mod\s+\w+\s*\{[^}]*(?:\{[^}]*\}[^}]*)?\}",
    re.DOTALL,
)

# ─────────────────────────────────────────────────────────
#  CRATE DETECTION
# ─────────────────────────────────────────────────────────

def detect_crates(code: str) -> list[str]:
    """Extract external crate root names used in the code that are in CRATE_MAP."""
    found: set[str] = set()
    # use-statement imports
    for m in re.finditer(r"^use\s+([a-zA-Z_][a-zA-Z0-9_]*)", code, re.MULTILINE):
        found.add(m.group(1))
    # extern crate declarations
    for m in re.finditer(r"^extern\s+crate\s+([a-zA-Z_][a-zA-Z0-9_]*)", code, re.MULTILINE):
        found.add(m.group(1))
    # attribute-style: #[tokio::main], #[derive(serde::Serialize)], ::prost::Enumeration
    for m in re.finditer(r"(?:^|[^a-zA-Z0-9_])([a-zA-Z_][a-zA-Z0-9_]*)::", code):
        found.add(m.group(1))
    # derive macros that pull in crates
    for m in re.finditer(r"derive\([^)]*\b(Serialize|Deserialize)\b", code):
        found.add("serde")
    found -= _STD_CRATES
    return sorted(c for c in found if c in CRATE_MAP)


def _pkg_name_candidates(code_name: str) -> list[str]:
    """
    Return cargo package name candidates to try for a given in-code crate identifier.
    Rust uses underscores in code; cargo package names typically use hyphens.
    """
    hyph = code_name.replace("_", "-")
    candidates = [code_name]
    if hyph != code_name:
        candidates.append(hyph)
    return candidates


def _try_cargo_add(proj: Path, code_name: str, env: dict | None = None) -> Optional[str]:
    """
    Attempt `cargo add <pkg>` for each candidate package name.
    Returns the package name that succeeded, or None.
    Consults and updates _crate_cache.
    """
    with _cache_lock:
        if code_name in _crate_cache:
            return _crate_cache[code_name]

    for pkg in _pkg_name_candidates(code_name):
        try:
            r = subprocess.run(
                ["cargo", "add", pkg],
                cwd=str(proj),
                capture_output=True,
                text=True,
                timeout=CARGO_ADD_TIMEOUT,
                env=env,
            )
            if r.returncode == 0:
                _record_cache(code_name, pkg)
                return pkg
            # Log why cargo add failed so we can diagnose missing crates
            _cargo_add_failures[pkg] = r.stderr.strip() or r.stdout.strip()
        except subprocess.TimeoutExpired:
            _cargo_add_failures[pkg] = "timeout"
            break
        except Exception as exc:
            _cargo_add_failures[pkg] = str(exc)

    _record_cache(code_name, None)
    return None


# Regex to extract crate names from E0432/E0433 error lines
_E_UNRESOLVED_CRATE = re.compile(
    r"(?:use of unresolved module or unlinked crate|unresolved import)\s+`([a-zA-Z_][a-zA-Z0-9_]*)",
)
_E_NO_CRATE = re.compile(
    r"no external crate `([a-zA-Z_][a-zA-Z0-9_]*)`",
)


def _extract_missing_crates(stderr: str) -> list[str]:
    """Parse cargo stderr for unknown crate names (E0432 / E0433)."""
    found: set[str] = set()
    for pat in (_E_UNRESOLVED_CRATE, _E_NO_CRATE):
        for m in pat.finditer(stderr):
            name = m.group(1)
            if name not in _STD_CRATES:
                found.add(name)
    return sorted(found)


def make_cargo_toml(crates: list[str], edition: str = EDITION) -> str:
    lines = [
        '[package]',
        'name = "rust-example"',
        'version = "0.1.0"',
        f'edition = "{edition}"',
        '',
        '[dependencies]',
    ]
    seen: set[str] = set()
    for crate in sorted(set(crates)):
        dep = CRATE_MAP.get(crate)
        if dep and dep not in seen:
            lines.append(dep)
            seen.add(dep)
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────
#  CODE WRAPPING
# ─────────────────────────────────────────────────────────

_PROSE_LINE = re.compile(
    r"^[A-Z`'\u2018\u2019\u201c\u201d][^\n]{20,}$"
)

def _strip_leading_prose(code: str) -> str:
    """
    Remove leading natural-language lines that were accidentally prepended
    to fixed_code in some source records (e.g. 'The original implementation...').
    Stops as soon as a line looks like Rust code.
    """
    lines = code.splitlines()
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if _PROSE_LINE.match(stripped):
            start = i + 1
        else:
            break
    return "\n".join(lines[start:])


def _preprocess(code: str) -> str:
    """Strip Python-style comments, prose prefixes, and convert `mod foo;` to inline modules."""
    code = _strip_leading_prose(code)
    code = _PYTHON_COMMENT.sub("", code)
    code = _MOD_FILE_DECL.sub(r"\1 {}", code)
    return code


def wrap_for_check(code: str) -> str:
    """
    Ensure `code` can be compiled as a binary crate.

    - Has fn main (sync or async) → return as-is; fix missing #[tokio::main] if needed.
    - Has top-level definitions but no main → append an empty fn main().
    - Looks like bare statements/expressions → wrap in fn main() { ... }.
    """
    code = _preprocess(code)

    has_main   = bool(_HAS_MAIN.search(code))
    async_main = bool(_ASYNC_MAIN.search(code))
    tokio_attr = bool(_TOKIO_ATTR.search(code))
    uses_tokio = "tokio" in code

    if has_main:
        if async_main and not tokio_attr:
            return _ASYNC_MAIN.sub(
                lambda m: "#[tokio::main]\n" + m.group(0),
                code, count=1,
            )
        return code

    has_top = bool(_TOP_LEVEL.search(code))
    if has_top:
        return (code + "\n#[tokio::main]\nasync fn main() {}\n") if uses_tokio \
               else (code + "\nfn main() {}\n")
    else:
        return (f"#[tokio::main]\nasync fn main() {{\n{code}\n}}\n") if uses_tokio \
               else (f"fn main() {{\n{code}\n}}\n")


def _looks_like_rust(text: str) -> bool:
    return bool(text and _LOOKS_LIKE_RUST.search(text))


def build_candidates(record: dict) -> list[tuple[str, str]]:
    """
    Return ordered list of (label, candidate_source) to try compiling.
    Each candidate is the raw source before wrapping (wrapping is done in validate()).

    Strategy order:
      1. fixed_code alone
      2. explanation (context structs) prepended to fixed_code  (for snippets)
      3. fixed_code prepended to explanation  (fixed_code defines, explanation uses)
      4. explanation alone (sometimes it is the complete program)
    """
    fixed       = (record.get("fixed_code") or "").strip()
    explanation = (record.get("explanation") or "").strip()
    schema      = record.get("schema_variant") or ""

    if not fixed:
        return []

    candidates: list[tuple[str, str]] = [("fixed_code", fixed)]

    if _looks_like_rust(explanation) and explanation != fixed:
        if schema == "snippet":
            # Most common for snippets: explanation holds the type/impl context
            candidates.append(("explanation+fixed", explanation + "\n\n" + fixed))
            candidates.append(("fixed+explanation", fixed + "\n\n" + explanation))
        # Always try explanation alone — it may be the complete program
        candidates.append(("explanation", explanation))

    return candidates


# ─────────────────────────────────────────────────────────
#  MSRV DETECTION  (edition 2021 baseline = 1.56)
# ─────────────────────────────────────────────────────────

def _detect_msrv(code: str) -> str:
    msrv = (1, 56, 0)
    checks = [
        (r"\blet\b[^=\n]+=[^=\n]+\belse\b\s*\{",   (1, 65, 0)),   # let-else
        (r"\btype\s+\w+\s*<[^>]*'[a-z]",            (1, 65, 0)),   # GATs
        (r"\bbacktrace\b",                           (1, 65, 0)),   # std::backtrace
        (r"type\s+\w+\s*=\s*impl\b",                (1, 75, 0)),   # impl-in-assoc-type
        (r"\basync\s*\|",                            (1, 85, 0)),   # async closures
        (r"fn\s+\w+[^{;]*->\s*impl\s+\w+",          (1, 75, 0)),   # RPITIT
        (r"\bLazyCell\b|\bLazyLock\b",              (1, 80, 0)),   # LazyCell/LazyLock
        (r"\bstd::sync::Mutex::new\b.*const",        (1, 63, 0)),   # const Mutex::new
        (r"#\[derive\(.*\bDefault\b",               (1, 56, 0)),   # already baseline
    ]
    for pattern, version in checks:
        if re.search(pattern, code):
            msrv = max(msrv, version)
    return f"{msrv[0]}.{msrv[1]}.{msrv[2]}"


# ─────────────────────────────────────────────────────────
#  CARGO INVOCATION
# ─────────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: Path, timeout: int = CARGO_TIMEOUT, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout, env=env,
    )


def _cargo_validate(wrapped: str, crates: list[str], edition: str, run_clippy: bool = True) -> dict:
    """
    Spin up a temp Cargo project, compile `wrapped`, return a result dict.

    When cargo check fails with E0432/E0433 (unresolved crate), the function
    attempts `cargo add <crate>` up to MAX_ADD_ROUNDS times so that obscure or
    project-specific crates are resolved automatically.
    """
    result = {
        "build": False, "clippy": False, "fmt": False, "msrv": False,
        "clippy_lints": [], "clippy_output": "",
        "min_rust_version": "1.56.0",
        "compiler_ver": "", "has_unsafe": bool(re.search(r"\bunsafe\b", wrapped)),
        "wrapped_code": wrapped, "errors": [],
        "crates_added": [],   # crates resolved via dynamic cargo add
    }

    SHARED_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    cargo_env = {**os.environ, "CARGO_TARGET_DIR": str(SHARED_TARGET_DIR)}

    with tempfile.TemporaryDirectory(prefix="rustval_") as tmp:
        proj = Path(tmp) / "rust-example"
        proj.mkdir()
        (proj / "Cargo.toml").write_text(make_cargo_toml(crates, edition), encoding="utf-8")
        src = proj / "src"
        src.mkdir()
        (src / "main.rs").write_text(wrapped, encoding="utf-8")

        # Detect compiler version once
        try:
            rv = subprocess.run(["rustc", "--version"], capture_output=True, text=True, timeout=10)
            result["compiler_ver"] = rv.stdout.strip()
        except Exception:
            pass

        # ── cargo check with dynamic-add retry loop ───────────────────
        added_this_run: set[str] = set()
        for _round in range(MAX_ADD_ROUNDS + 1):
            r = _run(["cargo", "check", "--quiet", "--color=never"], proj, env=cargo_env)
            if r.returncode == 0:
                break

            # On last allowed round, record errors and bail
            if _round == MAX_ADD_ROUNDS:
                result["errors"] = [r.stderr.strip()]
                return result

            missing = _extract_missing_crates(r.stderr)
            new_missing = [c for c in missing if c not in added_this_run]
            if not new_missing:
                # Failure is not a missing-crate error — no point retrying
                result["errors"] = [r.stderr.strip()]
                return result

            any_added = False
            for crate_name in new_missing:
                if crate_name in added_this_run:
                    continue
                pkg = _try_cargo_add(proj, crate_name, env=cargo_env)
                if pkg:
                    added_this_run.add(crate_name)
                    result["crates_added"].append(pkg)
                    any_added = True

            if not any_added:
                # Could not add any of the missing crates — hard fail
                result["errors"] = [r.stderr.strip()]
                return result
        else:
            # Exhausted rounds
            result["errors"] = [r.stderr.strip()]
            return result

        result["build"] = True

        # ── cargo fmt --check ─────────────────────────────────────────
        r = _run(["cargo", "fmt", "--", "--check"], proj, env=cargo_env)
        result["fmt"] = (r.returncode == 0)

        # ── cargo clippy ──────────────────────────────────────────────
        if run_clippy:
            r = _run(
                ["cargo", "clippy", "--quiet", "--color=never", "--", "-D", "warnings"],
                proj,
                env=cargo_env,
            )
            result["clippy"] = (r.returncode == 0)
            if not result["clippy"]:
                lints: list[str] = []
                for line in r.stderr.splitlines():
                    m = re.search(r"\[([a-z_:]+)\]", line)
                    if m and m.group(1) not in ("E", "W"):
                        lints.append(m.group(1))
                result["clippy_lints"]  = sorted(set(lints))
                result["clippy_output"] = r.stderr.strip()

        result["min_rust_version"] = _detect_msrv(wrapped)
        result["msrv"] = True

    return result


# ─────────────────────────────────────────────────────────
#  RECORD VALIDATION  (tries multiple candidate sources)
# ─────────────────────────────────────────────────────────

def _apply_schema(record: dict) -> dict:
    """Return a new dict with all _SCHEMA_DEFAULTS keys, overlaid with record values."""
    out = dict(_SCHEMA_DEFAULTS)
    for k in _SCHEMA_DEFAULTS:
        if k in record and record[k] is not None:
            out[k] = record[k]
    return out


def _make_failure_record(record: dict, reason: str) -> dict:
    """Build a schema-conformant record for a snippet that failed to build."""
    out = _apply_schema(record)
    out["validation"] = {
        "build":        False,
        "clippy":       None,
        "fmt":          None,
        "test":         False,
        "clippy_lints": [],
        "clippy_output": "",
        "build_error":  reason,
    }
    out["verified_at"] = str(int(time.time()))
    return out


def validate_record(record: dict) -> tuple[bool, dict, str]:
    """
    Try to validate a record's Rust code.

    Returns:
        (passed: bool, updated_record: dict, failure_reason: str)

    On success the updated_record has its validation/versioning fields filled.
    The fixed_code field is updated to whichever candidate compiled successfully.
    On failure the updated_record is a schema-conformant failure record.
    """
    declared = record.get("crates") or []
    if not isinstance(declared, list):
        declared = [str(declared)] if declared else []
    known_crates = [c for c in declared if c in CRATE_MAP]
    edition = record.get("edition") or EDITION

    candidates = build_candidates(record)
    if not candidates:
        reason = "empty fixed_code"
        return False, _make_failure_record(record, reason), reason

    last_errors: list[str] = []

    for label, raw_code in candidates:
        wrapped = wrap_for_check(raw_code)
        detected = detect_crates(wrapped)
        all_crates = sorted(set(known_crates + detected))

        try:
            val = _cargo_validate(wrapped, all_crates, edition, run_clippy=_RUN_CLIPPY)
        except subprocess.TimeoutExpired:
            reason = f"cargo timeout on candidate '{label}'"
            return False, _make_failure_record(record, reason), reason
        except Exception as exc:
            reason = f"exception on candidate '{label}': {exc}"
            return False, _make_failure_record(record, reason), reason

        if val["build"]:
            # Start from schema defaults, overlay known input fields, then validation results
            updated = _apply_schema(record)
            updated["fixed_code"] = raw_code.strip()
            updated["validation"] = {
                "fmt":           val["fmt"],
                "clippy":        val["clippy"],
                "build":         True,
                "test":          False,
                "clippy_lints":  val["clippy_lints"],
                "clippy_output": val["clippy_output"],
            }
            updated["verified_at"]      = str(int(time.time()))
            updated["compiler_ver"]     = val["compiler_ver"]
            updated["min_rust_version"] = val["min_rust_version"]
            updated["has_unsafe"]       = val["has_unsafe"]
            # Fold dynamically resolved crates into the crates field
            crates_added = sorted(set(val.get("crates_added", [])))
            if crates_added:
                existing = updated.get("crates") or []
                updated["crates"] = sorted(set(list(existing) + crates_added))
            return True, updated, ""

        last_errors = val.get("errors", [])

    reason = (last_errors[0][:300] if last_errors else "cargo check failed on all candidates")
    return False, _make_failure_record(record, reason), reason


# ─────────────────────────────────────────────────────────
#  TOP-LEVEL WORKER  (must be module-level for pickling)
# ─────────────────────────────────────────────────────────

_RUN_CLIPPY: bool = True  # overridden by --no-clippy flag

def _pool_init(run_clippy: bool) -> None:
    global _RUN_CLIPPY
    _RUN_CLIPPY = run_clippy

def _worker(record: dict) -> tuple[bool, dict, str]:
    return validate_record(record)


# ─────────────────────────────────────────────────────────
#  ROTATING OUTPUT WRITER
# ─────────────────────────────────────────────────────────

class RotatingWriter:
    """Writes JSONL lines to sequentially numbered files, rotating at max_bytes."""

    def __init__(self, output_dir: Path, prefix: str = "validated",
                 max_bytes: int = MAX_FILE_BYTES):
        self.output_dir = output_dir
        self.prefix     = prefix
        self.max_bytes  = max_bytes
        self._file      = None
        self._bytes     = 0
        self._idx       = 0
        self.files_written: list[Path] = []
        self._open_next()

    def _open_next(self) -> None:
        if self._file:
            self._file.close()
        self._idx += 1
        path = self.output_dir / f"{self.prefix}_{self._idx:04d}.jsonl"
        self._file = open(path, "a", encoding="utf-8")
        self._bytes = path.stat().st_size if path.exists() else 0
        self.files_written.append(path)

    def write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        data = line.encode("utf-8")
        if self._bytes + len(data) > self.max_bytes and self._bytes > 0:
            self._open_next()
        self._file.write(line)
        self._file.flush()
        self._bytes += len(data)

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


# ─────────────────────────────────────────────────────────
#  FAILURE LOG
# ─────────────────────────────────────────────────────────

_SEP = "═" * 72


def _log_failure(log_path: Path, record: dict, reason: str) -> None:
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rid = record.get("id", "?")
    src = record.get("source", "?")
    code_snippet = (record.get("fixed_code") or "")[:400]

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{_SEP}\n")
        f.write(f"  {ts}  id={rid}  source={src}\n")
        f.write(f"  reason: {reason}\n")
        f.write(f"{_SEP}\n")
        f.write(code_snippet)
        if code_snippet and not code_snippet.endswith("\n"):
            f.write("\n")
        f.write(f"{_SEP} END {_SEP}\n")


# ─────────────────────────────────────────────────────────
#  RECORD STREAMING
# ─────────────────────────────────────────────────────────

def iter_records(input_dir: Path, limit: Optional[int] = None):
    """Yield (source_file, record_dict) for every valid record in input_dir."""
    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        print(f"[ERROR] No .jsonl files found in: {input_dir}", file=sys.stderr)
        return

    total_yielded = 0
    for fpath in files:
        print(f"  Scanning {fpath.name} …", flush=True)
        with open(fpath, encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"  [WARN] {fpath.name}:{lineno} — bad JSON: {exc}", file=sys.stderr)
                    continue
                if not isinstance(record, dict):
                    continue
                # Skip records that have no code to validate
                if not (record.get("fixed_code") or "").strip():
                    continue

                yield fpath, record
                total_yielded += 1
                if limit and total_yielded >= limit:
                    return


# ─────────────────────────────────────────────────────────
#  PROGRESS DISPLAY
# ─────────────────────────────────────────────────────────

def _progress(i: int, total: int, ok: int, fail: int, t0: float) -> None:
    elapsed = time.time() - t0
    rate    = i / elapsed if elapsed else 0
    eta_s   = (total - i) / rate if rate else 0
    pct     = i / total * 100 if total else 0
    bar_len = 30
    filled  = int(bar_len * i / total) if total else 0
    bar     = "█" * filled + "░" * (bar_len - filled)
    print(
        f"\r  [{bar}] {pct:5.1f}%  {i:>{len(str(total))}}/{total}"
        f"  ✓ {ok}  ✗ {fail}"
        f"  {rate:5.1f} rec/s  ETA {eta_s/60:.1f}m   ",
        end="", flush=True,
    )


# ─────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the persistent crate-add cache so we skip known-good/bad lookups
    global CRATE_CACHE_FILE
    cache_path = output_dir / CRATE_CACHE_FILE.name
    _load_crate_cache(cache_path)
    # Monkey-patch save to use the output dir
    CRATE_CACHE_FILE = cache_path

    log_path     = output_dir / "validation_failures.log"
    writer       = RotatingWriter(output_dir, prefix="validated", max_bytes=MAX_FILE_BYTES)
    failures_dir = output_dir / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)
    fail_writer  = RotatingWriter(failures_dir, prefix="failed", max_bytes=MAX_FILE_BYTES)

    # Load all records upfront so we know the total for the progress bar.
    # 160 K records × ~3 KB avg ≈ ~480 MB — acceptable.
    print("\nScanning input files…", flush=True)
    all_records = list(iter_records(input_dir, limit=args.limit))
    total = len(all_records)
    print(
        f"\nLoaded {total:,} records from {input_dir.resolve()}\n"
        f"Workers: {args.workers}   Output: {output_dir.resolve()}\n",
        flush=True,
    )

    ok = fail = 0
    t0 = time.time()

    global _RUN_CLIPPY
    _RUN_CLIPPY = not args.no_clippy

    if args.workers == 1:
        # ── Single-process (easy to debug, same logic) ────────────────
        for i, (_fp, record) in enumerate(all_records, 1):
            passed, updated, reason = validate_record(record)
            if passed:
                writer.write(updated)
                ok += 1
            else:
                _log_failure(log_path, record, reason)
                fail_writer.write(updated)
                fail += 1
            if i % 50 == 0 or i == total:
                _progress(i, total, ok, fail, t0)

    else:
        # ── Multi-process ─────────────────────────────────────────────
        # Submit all futures; collect in completion order.
        # The writer runs in the main process — no race conditions.
        records_only = [r for _, r in all_records]
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_pool_init,
            initargs=(not args.no_clippy,),
        ) as pool:
            futs = {pool.submit(_worker, rec): rec for rec in records_only}
            i = 0
            for fut in as_completed(futs):
                record = futs[fut]
                i += 1
                try:
                    passed, updated, reason = fut.result()
                except Exception as exc:
                    passed, updated, reason = False, _make_failure_record(record, str(exc)), str(exc)

                if passed:
                    writer.write(updated)
                    ok += 1
                else:
                    _log_failure(log_path, record, reason)
                    fail_writer.write(updated)
                    fail += 1

                if i % 50 == 0 or i == total:
                    _progress(i, total, ok, fail, t0)

    writer.close()
    fail_writer.close()
    elapsed = time.time() - t0

    print(f"\n\n{'─' * 60}")
    print(f"  Done in {elapsed/60:.1f} min")
    print(f"  Passed : {ok:>8,}  ({ok/total*100:.1f}%)")
    print(f"  Failed : {fail:>8,}  ({fail/total*100:.1f}%)")
    print(f"  Total  : {total:>8,}")
    print(f"{'─' * 60}")
    print(f"  Output files  : {output_dir.resolve()}")
    print(f"  Files written : {len(writer.files_written)}")
    for p in writer.files_written:
        size_mb = p.stat().st_size / 1024 / 1024 if p.exists() else 0
        print(f"    {p.name}  ({size_mb:.1f} MB)")
    if fail:
        print(f"  Failure log   : {log_path}")
        print(f"  Failure JSONL : {failures_dir.resolve()}  ({len(fail_writer.files_written)} file(s))")
    if _cargo_add_failures:
        add_fail_path = output_dir / "cargo_add_failures.log"
        with open(add_fail_path, "w", encoding="utf-8") as f:
            for pkg, reason in sorted(_cargo_add_failures.items()):
                f.write(f"{pkg}\n  {reason}\n\n")
        print(f"  cargo add failures ({len(_cargo_add_failures)}): {add_fail_path}")
    print()


# ─────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Batch-validate a folder of Rust JSONL datasets with cargo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir", required=True,
        help="Directory containing .jsonl files to process",
    )
    p.add_argument(
        "--output-dir", default="validated",
        help="Output directory for cleaned JSONL files",
    )
    p.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
        help="Parallel worker processes (default: CPU count − 1)",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N records (useful for smoke-testing)",
    )
    p.add_argument(
        "--no-clippy", action="store_true", default=False,
        help="Skip cargo clippy (faster; clippy field will be null in output)",
    )
    args = p.parse_args()

    if not Path(args.input_dir).is_dir():
        print(f"[ERROR] --input-dir does not exist: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    run(args)


if __name__ == "__main__":
    main()