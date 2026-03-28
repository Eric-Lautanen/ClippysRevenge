#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  validate_frontier.py  —  Frontier Dataset Validator             ║
║  Validates datasets/frontier/*.jsonl with cargo check/fmt/clippy ║
║  Resolves labeled crates; writes passing records to              ║
║  datasets/rust/ (one file per category, 50 MB cap).              ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python validate_frontier.py
    python validate_frontier.py --input-dir datasets/frontier --output-dir datasets/rust
    python validate_frontier.py --workers 2
    python validate_frontier.py --no-clippy
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
import uuid
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────

EDITION           = "2021"
LICENSE           = "Apache-2.0"
MAX_FILE_BYTES    = 50 * 1024 * 1024
CARGO_TIMEOUT     = 180
CARGO_ADD_TIMEOUT = 120
MAX_ADD_ROUNDS    = 6
SHARED_TARGET     = Path(__file__).parent / "cargo_target"

# ─────────────────────────────────────────────────────────
#  CRATE MAP  (code-name → Cargo.toml dep line)
# ─────────────────────────────────────────────────────────

CRATE_MAP: dict[str, str] = {
    "tokio":              'tokio = { version = "1", features = ["full"] }',
    "tokio_util":         'tokio-util = { version = "0.7", features = ["full"] }',
    "tokio_stream":       'tokio-stream = "0.1"',
    "futures":            'futures = "0.3"',
    "futures_util":       'futures-util = "0.3"',
    "async_trait":        'async-trait = "0.1"',
    "async_stream":       'async-stream = "0.1"',
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
    "hyper_util":         'hyper-util = "0.1"',
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
    "smallvec":           'smallvec = "1"',
    "ahash":              'ahash = "0.8"',
    "hashbrown":          'hashbrown = "0.14"',
    "bitflags":           'bitflags = "2"',
    "bytes_utils":        'bytes-utils = "0.1"',
    "http":               'http = "1"',
    "http_body":          'http-body = "1"',
    "http_body_util":     'http-body-util = "0.1"',
    "tower_http":         'tower-http = { version = "0.5", features = ["full"] }',
    "syn":                'syn = { version = "2", features = ["full"] }',
    "quote":              'quote = "1"',
    "proc_macro2":        'proc-macro2 = "1"',
    "tempfile":           'tempfile = "3"',
    "walkdir":            'walkdir = "1"',
    "tokio_tungstenite":  'tokio-tungstenite = "0.21"',
    "flume":              'flume = "0.11"',
    "kanal":              'kanal = "0.1"',
    "arc_swap":           'arc-swap = "1"',
}

# ─────────────────────────────────────────────────────────
#  CARGO.TOML GENERATION
# ─────────────────────────────────────────────────────────

def _crate_dep_line(pkg_name: str) -> Optional[str]:
    """Return the Cargo.toml dep line for a package name, or None if not in CRATE_MAP."""
    # pkg_name may be hyphenated (Cargo) or underscored (Rust code) — try both
    code_name = pkg_name.replace("-", "_")
    return CRATE_MAP.get(code_name) or CRATE_MAP.get(pkg_name)


def make_cargo_toml(pkg_names: list[str], edition: str = EDITION) -> str:
    lines = [
        '[package]',
        'name = "rust-example"',
        'version = "0.1.0"',
        f'edition = "{edition}"',
        '',
        '[dependencies]',
    ]
    seen: set[str] = set()
    for name in sorted(set(pkg_names)):
        dep = _crate_dep_line(name)
        if dep and dep not in seen:
            lines.append(dep)
            seen.add(dep)
        # Crates not in CRATE_MAP are added dynamically via cargo add in _cargo_validate
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────
#  REGEX / CODE WRAPPING
# ─────────────────────────────────────────────────────────

_HAS_MAIN    = re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+main\s*\(", re.MULTILINE)
_ASYNC_MAIN  = re.compile(r"^\s*(?:pub\s+)?async\s+fn\s+main\s*\(", re.MULTILINE)
_TOKIO_ATTR  = re.compile(r"#\[tokio::main\]")
_TOP_LEVEL   = re.compile(
    r"^(?:pub\s+)?(?:async\s+)?(?:fn|struct|enum|trait|type|impl|const|static|mod|use|extern)\s",
    re.MULTILINE,
)
_PYTHON_COMMENT = re.compile(r"^#(?!\[)\s.*$", re.MULTILINE)
_MOD_FILE_DECL  = re.compile(r"^(\s*(?:pub(?:\(crate\))?\s+)?mod\s+\w+)\s*;", re.MULTILINE)
_E_UNRESOLVED   = re.compile(
    r"(?:use of unresolved module or unlinked crate"
    r"|unresolved import"
    r"|no external crate"
    r"|could not find `.+?` in the list of imported crates"
    r"|failed to resolve: could not find)"
)
_E_CRATE_NAME   = re.compile(
    r"(?:unresolved import|use of unresolved module or unlinked crate|"
    r"no external crate)\s+`([a-zA-Z_][a-zA-Z0-9_]*)"
)


def wrap_for_check(code: str) -> str:
    code = _PYTHON_COMMENT.sub("", code)
    code = _MOD_FILE_DECL.sub(r"\1 {}", code)

    has_main   = bool(_HAS_MAIN.search(code))
    async_main = bool(_ASYNC_MAIN.search(code))
    tokio_attr = bool(_TOKIO_ATTR.search(code))
    uses_tokio = "tokio" in code

    if has_main:
        if async_main and not tokio_attr:
            return _ASYNC_MAIN.sub(lambda m: "#[tokio::main]\n" + m.group(0), code, count=1)
        return code

    has_top = bool(_TOP_LEVEL.search(code))
    if has_top:
        return (code + "\n#[tokio::main]\nasync fn main() {}\n") if uses_tokio \
               else (code + "\nfn main() {}\n")
    else:
        return (f"#[tokio::main]\nasync fn main() {{\n{code}\n}}\n") if uses_tokio \
               else (f"fn main() {{\n{code}\n}}\n")


def _detect_msrv(code: str) -> str:
    msrv = (1, 56, 0)
    checks = [
        (r"\blet\b[^=\n]+=[^=\n]+\belse\b\s*\{", (1, 65, 0)),
        (r"\btype\s+\w+\s*<[^>]*'[a-z]",          (1, 65, 0)),
        (r"\bbacktrace\b",                         (1, 65, 0)),
        (r"type\s+\w+\s*=\s*impl\b",              (1, 75, 0)),
        (r"\basync\s*\|",                          (1, 85, 0)),
        (r"fn\s+\w+[^{;]*->\s*impl\s+\w+",        (1, 75, 0)),
        (r"\bLazyCell\b|\bLazyLock\b",            (1, 80, 0)),
        (r"\bstd::sync::Mutex::new\b.*const",      (1, 63, 0)),
    ]
    for pattern, version in checks:
        if re.search(pattern, code):
            msrv = max(msrv, version)
    return f"{msrv[0]}.{msrv[1]}.{msrv[2]}"


def _run(cmd: list[str], cwd: Path, env: dict | None = None,
         timeout: int = CARGO_TIMEOUT) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout, env=env,
    )


def _extract_missing_crates(stderr: str) -> list[str]:
    found: set[str] = set()
    for m in _E_CRATE_NAME.finditer(stderr):
        found.add(m.group(1))
    return sorted(found)


# ─────────────────────────────────────────────────────────
#  CARGO VALIDATION
# ─────────────────────────────────────────────────────────

def cargo_validate(fixed_code: str, pkg_names: list[str], edition: str,
                   run_clippy: bool = True) -> dict:
    """
    Validate fixed_code with cargo check / fmt / clippy.

    Uses labeled pkg_names to build Cargo.toml. For any crates not in CRATE_MAP,
    attempts `cargo add` dynamically (up to MAX_ADD_ROUNDS times).
    """
    result = {
        "build": False, "clippy": False, "fmt": False,
        "clippy_lints": [], "clippy_output": "",
        "min_rust_version": "1.56.0",
        "compiler_ver": "", "has_unsafe": bool(re.search(r"\bunsafe\b", fixed_code)),
        "errors": [], "crates_added": [],
    }

    wrapped = wrap_for_check(fixed_code)
    result["has_unsafe"] = bool(re.search(r"\bunsafe\b", wrapped))

    SHARED_TARGET.mkdir(parents=True, exist_ok=True)
    cargo_env = {**os.environ, "CARGO_TARGET_DIR": str(SHARED_TARGET)}

    try:
        rv = subprocess.run(["rustc", "--version"], capture_output=True, text=True, timeout=10)
        result["compiler_ver"] = rv.stdout.strip()
    except Exception:
        pass

    with tempfile.TemporaryDirectory(prefix="rustfrontier_") as tmp:
        proj = Path(tmp) / "rust-example"
        proj.mkdir()
        (proj / "Cargo.toml").write_text(make_cargo_toml(pkg_names, edition), encoding="utf-8")
        src = proj / "src"
        src.mkdir()
        (src / "main.rs").write_text(wrapped, encoding="utf-8")

        # ── cargo check with dynamic-add retry ───────────────────────
        added: set[str] = set()
        for round_ in range(MAX_ADD_ROUNDS + 1):
            r = _run(["cargo", "check", "--quiet", "--color=never"], proj, env=cargo_env)
            if r.returncode == 0:
                break

            if round_ == MAX_ADD_ROUNDS:
                result["errors"] = [r.stderr.strip()]
                return result

            if not _E_UNRESOLVED.search(r.stderr):
                result["errors"] = [r.stderr.strip()]
                return result

            missing = [c for c in _extract_missing_crates(r.stderr) if c not in added]
            if not missing:
                # E0432/E0433 but can't parse names — try the labeled pkg_names directly
                unresolved_pkgs = [p for p in pkg_names
                                   if _crate_dep_line(p) is None and p not in added]
                if not unresolved_pkgs:
                    result["errors"] = [r.stderr.strip()]
                    return result
                missing = unresolved_pkgs

            any_added = False
            for crate in missing:
                pkg = crate.replace("_", "-")
                try:
                    add_r = subprocess.run(
                        ["cargo", "add", pkg],
                        cwd=str(proj), env=cargo_env,
                        capture_output=True, text=True, timeout=CARGO_ADD_TIMEOUT,
                    )
                    if add_r.returncode == 0:
                        added.add(crate)
                        result["crates_added"].append(pkg)
                        any_added = True
                except subprocess.TimeoutExpired:
                    pass

            if not any_added:
                result["errors"] = [r.stderr.strip()]
                return result

        result["build"] = True

        # ── cargo fmt ────────────────────────────────────────────────
        r = _run(["cargo", "fmt", "--", "--check"], proj, env=cargo_env)
        result["fmt"] = (r.returncode == 0)

        # ── cargo clippy ─────────────────────────────────────────────
        if run_clippy:
            r = _run(
                ["cargo", "clippy", "--quiet", "--color=never", "--", "-D", "warnings"],
                proj, env=cargo_env,
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

    return result


# ─────────────────────────────────────────────────────────
#  SCHEMA
# ─────────────────────────────────────────────────────────

def _normalise_record(record: dict, val: dict) -> dict:
    """
    Merge original record fields with fresh validation results.
    Ensures all schema fields are present; preserves unknown fields.
    """
    out = {k: v for k, v in record.items()}

    # Fill in fields that old frontier records may be missing
    if not out.get("id"):
        out["id"] = str(uuid.uuid4())
    if not out.get("schema_variant"):
        out["schema_variant"] = "bug_fix"
    if not out.get("source"):
        out["source"] = "frontier"
    if not out.get("license"):
        out["license"] = LICENSE

    # Overwrite validation metadata with fresh results
    out["validation"] = {
        "build":         val["build"],
        "fmt":           val["fmt"],
        "clippy":        val["clippy"],
        "test":          False,
        "clippy_lints":  val["clippy_lints"],
        "clippy_output": val["clippy_output"],
    }
    out["verified_at"]      = str(int(time.time()))
    out["compiler_ver"]     = val["compiler_ver"]
    out["min_rust_version"] = val["min_rust_version"]
    out["has_unsafe"]       = val["has_unsafe"]

    # Fold in any dynamically resolved crates
    if val.get("crates_added"):
        existing = out.get("crates") or []
        out["crates"] = sorted(set(list(existing) + val["crates_added"]))

    return out


# ─────────────────────────────────────────────────────────
#  PER-CATEGORY ROTATING WRITER
# ─────────────────────────────────────────────────────────

class CategoryWriter:
    """Writes to per-category JSONL files, rotating at max_bytes."""

    def __init__(self, output_dir: Path, max_bytes: int = MAX_FILE_BYTES):
        self.output_dir = output_dir
        self.max_bytes  = max_bytes
        self._handles:  dict[str, object] = {}
        self._sizes:    dict[str, int]    = {}
        self._counters: dict[str, int]    = {}
        self.files_written: list[Path]    = []

    def _safe_name(self, category: str) -> str:
        for ch in r'\/:*?"<>|':
            category = category.replace(ch, "")
        return category.strip()

    def _path(self, category: str, idx: int) -> Path:
        base = self._safe_name(category)
        return self.output_dir / (f"{base}.jsonl" if idx == 1 else f"{base}_{idx:04d}.jsonl")

    def write(self, category: str, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        data = line.encode("utf-8")

        if category not in self._handles:
            self._counters[category] = 1
            p = self._path(category, 1)
            self._handles[category] = open(p, "a", encoding="utf-8")
            self._sizes[category]   = p.stat().st_size if p.exists() else 0
            self.files_written.append(p)

        if self._sizes[category] + len(data) > self.max_bytes and self._sizes[category] > 0:
            self._handles[category].close()
            self._counters[category] += 1
            p = self._path(category, self._counters[category])
            self._handles[category] = open(p, "a", encoding="utf-8")
            self._sizes[category]   = 0
            self.files_written.append(p)

        self._handles[category].write(line)
        self._handles[category].flush()
        self._sizes[category] += len(data)

    def close(self) -> None:
        for fh in self._handles.values():
            fh.close()
        self._handles.clear()


# ─────────────────────────────────────────────────────────
#  FAILURE LOG
# ─────────────────────────────────────────────────────────

_SEP = "═" * 72


def _log_failure(log_path: Path, record: dict, reason: str) -> None:
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rid = record.get("id", "?")
    cat = record.get("category", "?")
    snippet = (record.get("fixed_code") or "")[:300]
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{_SEP}\n")
        f.write(f"  {ts}  id={rid}  cat={cat!r}\n")
        f.write(f"  reason: {reason[:300]}\n")
        f.write(f"{_SEP}\n")
        f.write(snippet)
        if snippet and not snippet.endswith("\n"):
            f.write("\n")
        f.write(f"{_SEP} END {_SEP}\n")


# ─────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        print(f"[ERROR] No .jsonl files in {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Count total records upfront for progress
    total = 0
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            total += sum(1 for l in fh if l.strip())
    print(f"Found {total:,} records across {len(files)} file(s) in {input_dir.resolve()}")
    print(f"Output → {output_dir.resolve()}\n")

    log_path    = output_dir / "validation_failures.log"
    writer      = CategoryWriter(output_dir)
    fail_dir    = output_dir / "failures"
    fail_dir.mkdir(exist_ok=True)
    fail_writer = CategoryWriter(fail_dir)

    ok = fail = i = 0
    t0 = time.time()

    for src_file in files:
        cat_hint = src_file.stem   # use filename as category hint for progress
        with open(src_file, encoding="utf-8", errors="replace") as fh:
            lines = [l.strip() for l in fh if l.strip()]

        for lineno, line in enumerate(lines, 1):
            i += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] {src_file.name}:{lineno} bad JSON: {e}", file=sys.stderr)
                fail += 1
                continue

            fixed_code = (record.get("fixed_code") or "").strip()
            if not fixed_code:
                fail += 1
                continue

            pkg_names = record.get("crates") or []
            edition   = str(record.get("edition") or EDITION)
            category  = record.get("category") or src_file.stem

            print(f"  [{i:>{len(str(total))}}/{total}]  {category[:50]}", end="  ", flush=True)

            try:
                val = cargo_validate(
                    fixed_code, pkg_names, edition,
                    run_clippy=(not args.no_clippy),
                )
            except subprocess.TimeoutExpired:
                reason = "cargo timeout"
                print(f"TIMEOUT")
                _log_failure(log_path, record, reason)
                fail_rec = {k: v for k, v in record.items()}
                fail_rec.setdefault("id", str(uuid.uuid4()))
                fail_rec["validation"] = {"build": False, "clippy": None, "fmt": None,
                                          "test": False, "clippy_lints": [], "clippy_output": "",
                                          "build_error": reason}
                fail_rec["verified_at"] = str(int(time.time()))
                fail_writer.write(category, fail_rec)
                fail += 1
                continue
            except Exception as exc:
                reason = str(exc)
                print(f"ERROR: {reason[:60]}")
                fail += 1
                continue

            if val["build"]:
                out = _normalise_record(record, val)
                writer.write(category, out)
                ok += 1
                badges = []
                if val["clippy"]:  badges.append("✓clippy")
                else:              badges.append(f"~clippy({len(val['clippy_lints'])} lints)")
                if val["fmt"]:     badges.append("✓fmt")
                else:              badges.append("~fmt")
                print(f"✓ build  {' '.join(badges)}")
            else:
                reason = (val["errors"][0][:200] if val["errors"] else "cargo check failed")
                print(f"✗ {reason[:80]}")
                _log_failure(log_path, record, reason)
                fail_rec = _normalise_record(record, val)
                fail_rec["validation"]["build_error"] = reason
                fail_writer.write(category, fail_rec)
                fail += 1

    writer.close()
    fail_writer.close()

    elapsed = time.time() - t0
    print(f"\n{'─' * 60}")
    print(f"  Done in {elapsed/60:.1f} min")
    print(f"  Passed : {ok:>6,}  ({ok/max(i,1)*100:.1f}%)")
    print(f"  Failed : {fail:>6,}  ({fail/max(i,1)*100:.1f}%)")
    print(f"  Total  : {i:>6,}")
    print(f"{'─' * 60}")
    print(f"  Output files ({len(writer.files_written)}):")
    for p in writer.files_written:
        mb = p.stat().st_size / 1024 / 1024 if p.exists() else 0
        print(f"    {p.name}  ({mb:.2f} MB)")
    if fail:
        print(f"  Failure log : {log_path}")
    print()


# ─────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Validate frontier Rust examples (with crates) → datasets/rust/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir",  default="datasets/frontier",
                   help="Directory of frontier .jsonl files")
    p.add_argument("--output-dir", default="datasets/rust",
                   help="Output directory (one .jsonl per category)")
    p.add_argument("--no-clippy",  action="store_true", default=False,
                   help="Skip clippy (faster)")
    args = p.parse_args()

    if not Path(args.input_dir).is_dir():
        print(f"[ERROR] --input-dir not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    run(args)


if __name__ == "__main__":
    main()
