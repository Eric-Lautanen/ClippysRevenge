#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  validate_dataset.py  —  Rust JSONL Dataset Batch Validator      ║
║  Runs cargo check / clippy / fmt on every record's fixed_code.   ║
║  Only pure-std Rust passes; snippets requiring external crates   ║
║  are rejected. Passing records are written to 50 MB-capped JSONL.║
╚══════════════════════════════════════════════════════════════════╝

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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────

EDITION        = "2021"
MAX_FILE_BYTES = 50 * 1024 * 1024   # 50 MB per output file
CARGO_TIMEOUT  = 180                 # seconds per cargo invocation
SHARED_TARGET  = Path(__file__).parent / "cargo_target"  # shared build cache

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

_STD_MODULES = frozenset({
    "std", "core", "alloc", "proc_macro", "test",
    # common std re-exports used bare
    "self", "super", "crate",
})

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
_MOD_FILE_DECL  = re.compile(r"^(\s*(?:pub(?:\(crate\))?\s+)?mod\s+\w+)\s*;", re.MULTILINE)
_LOOKS_LIKE_RUST = re.compile(
    r"^\s*(?:fn\s|struct\s|enum\s|impl\s|trait\s|pub\s|use\s|mod\s|let\s|const\s|"
    r"static\s|type\s|#\[|//|/\*)",
    re.MULTILINE,
)
_PROSE_LINE = re.compile(r"^[A-Z`'\u2018\u2019\u201c\u201d][^\n]{20,}$")

# E0432 / E0433 — unresolved crate reference
_E_UNRESOLVED = re.compile(
    r"(?:use of unresolved module or unlinked crate|unresolved import|"
    r"no external crate)\s+`([a-zA-Z_][a-zA-Z0-9_]*)",
)

# ─────────────────────────────────────────────────────────
#  EXTERNAL-CRATE DETECTION
# ─────────────────────────────────────────────────────────

def _uses_external_crates(code: str) -> list[str]:
    """
    Return a sorted list of crate root names that appear to be external
    (i.e. not std/core/alloc/self/super/crate).
    An empty list means the code is pure-std.
    """
    found: set[str] = set()
    for m in re.finditer(r"^use\s+([a-zA-Z_][a-zA-Z0-9_]*)", code, re.MULTILINE):
        found.add(m.group(1))
    for m in re.finditer(r"^extern\s+crate\s+([a-zA-Z_][a-zA-Z0-9_]*)", code, re.MULTILINE):
        found.add(m.group(1))
    for m in re.finditer(r"(?:^|[^a-zA-Z0-9_])([a-zA-Z_][a-zA-Z0-9_]*)::", code):
        found.add(m.group(1))
    found -= _STD_MODULES
    return sorted(found)


def _cargo_mentions_external_crates(stderr: str) -> bool:
    """True if cargo stderr contains E0432/E0433 (unresolved external crate)."""
    return bool(_E_UNRESOLVED.search(stderr))


# ─────────────────────────────────────────────────────────
#  CODE WRAPPING
# ─────────────────────────────────────────────────────────

def _strip_leading_prose(code: str) -> str:
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
    code = _strip_leading_prose(code)
    code = _PYTHON_COMMENT.sub("", code)
    code = _MOD_FILE_DECL.sub(r"\1 {}", code)
    return code


def wrap_for_check(code: str) -> str:
    """Ensure code can be compiled as a binary crate (adds fn main if needed)."""
    code = _preprocess(code)

    has_main   = bool(_HAS_MAIN.search(code))
    async_main = bool(_ASYNC_MAIN.search(code))
    tokio_attr = bool(_TOKIO_ATTR.search(code))

    if has_main:
        if async_main and not tokio_attr:
            return _ASYNC_MAIN.sub(lambda m: "#[tokio::main]\n" + m.group(0), code, count=1)
        return code

    has_top = bool(_TOP_LEVEL.search(code))
    if has_top:
        return code + "\nfn main() {}\n"
    else:
        return f"fn main() {{\n{code}\n}}\n"


def _looks_like_rust(text: str) -> bool:
    return bool(text and _LOOKS_LIKE_RUST.search(text))


def build_candidates(record: dict) -> list[tuple[str, str]]:
    """Return ordered (label, source) pairs to try compiling."""
    fixed       = (record.get("fixed_code") or "").strip()
    explanation = (record.get("explanation") or "").strip()
    schema      = record.get("schema_variant") or ""

    if not fixed:
        return []

    candidates: list[tuple[str, str]] = [("fixed_code", fixed)]
    if _looks_like_rust(explanation) and explanation != fixed:
        if schema == "snippet":
            candidates.append(("explanation+fixed", explanation + "\n\n" + fixed))
            candidates.append(("fixed+explanation", fixed + "\n\n" + explanation))
        candidates.append(("explanation", explanation))
    return candidates


# ─────────────────────────────────────────────────────────
#  MSRV DETECTION
# ─────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────
#  CARGO INVOCATION
# ─────────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True,
        timeout=CARGO_TIMEOUT, env=env,
    )


_MINIMAL_CARGO_TOML = """\
[package]
name = "rust-example"
version = "0.1.0"
edition = "{edition}"

[dependencies]
"""


def _cargo_validate(wrapped: str, edition: str, run_clippy: bool = True) -> dict:
    """
    Compile `wrapped` in an isolated temp project with no external dependencies.
    Returns a result dict; build=False if it needs external crates or has errors.
    """
    result = {
        "build": False, "clippy": False, "fmt": False,
        "clippy_lints": [], "clippy_output": "",
        "min_rust_version": "1.56.0",
        "compiler_ver": "", "has_unsafe": bool(re.search(r"\bunsafe\b", wrapped)),
        "errors": [], "needs_crates": False,
    }

    SHARED_TARGET.mkdir(parents=True, exist_ok=True)
    cargo_env = {**os.environ, "CARGO_TARGET_DIR": str(SHARED_TARGET)}

    with tempfile.TemporaryDirectory(prefix="rustval_") as tmp:
        proj = Path(tmp) / "rust-example"
        proj.mkdir()
        (proj / "Cargo.toml").write_text(
            _MINIMAL_CARGO_TOML.format(edition=edition), encoding="utf-8"
        )
        src = proj / "src"
        src.mkdir()
        (src / "main.rs").write_text(wrapped, encoding="utf-8")

        try:
            rv = subprocess.run(["rustc", "--version"], capture_output=True, text=True, timeout=10)
            result["compiler_ver"] = rv.stdout.strip()
        except Exception:
            pass

        r = _run(["cargo", "check", "--quiet", "--color=never"], proj, env=cargo_env)
        if r.returncode != 0:
            if _cargo_mentions_external_crates(r.stderr):
                result["needs_crates"] = True
                result["errors"] = ["requires external crates"]
            else:
                result["errors"] = [r.stderr.strip()]
            return result

        result["build"] = True

        r = _run(["cargo", "fmt", "--", "--check"], proj, env=cargo_env)
        result["fmt"] = (r.returncode == 0)

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
#  RECORD VALIDATION
# ─────────────────────────────────────────────────────────

def _apply_schema(record: dict) -> dict:
    """Preserve all original fields; fill in any missing schema defaults."""
    out = {k: v for k, v in record.items()}
    for k, default in _SCHEMA_DEFAULTS.items():
        if k not in out or out[k] is None:
            out[k] = default
    return out


def _make_failure_record(record: dict, reason: str) -> dict:
    out = _apply_schema(record)
    out["validation"] = {
        "build": False, "clippy": None, "fmt": None, "test": False,
        "clippy_lints": [], "clippy_output": "", "build_error": reason,
    }
    out["verified_at"] = str(int(time.time()))
    return out


def validate_record(record: dict) -> tuple[bool, dict, str]:
    edition    = str(record.get("edition") or EDITION)
    candidates = build_candidates(record)
    if not candidates:
        reason = "empty fixed_code"
        return False, _make_failure_record(record, reason), reason

    # Fast pre-check: reject immediately if the raw code references external crates
    fixed = (record.get("fixed_code") or "").strip()
    external = _uses_external_crates(fixed)
    if external:
        reason = f"requires external crates: {', '.join(external[:5])}"
        return False, _make_failure_record(record, reason), reason

    last_errors: list[str] = []

    for label, raw_code in candidates:
        wrapped = wrap_for_check(raw_code)

        try:
            val = _cargo_validate(wrapped, edition, run_clippy=_RUN_CLIPPY)
        except subprocess.TimeoutExpired:
            reason = f"cargo timeout on candidate '{label}'"
            return False, _make_failure_record(record, reason), reason
        except Exception as exc:
            reason = f"exception on candidate '{label}': {exc}"
            return False, _make_failure_record(record, reason), reason

        if val.get("needs_crates"):
            reason = "requires external crates"
            return False, _make_failure_record(record, reason), reason

        if val["build"]:
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
            return True, updated, ""

        last_errors = val.get("errors", [])

    reason = last_errors[0][:300] if last_errors else "cargo check failed on all candidates"
    return False, _make_failure_record(record, reason), reason


# ─────────────────────────────────────────────────────────
#  WORKER
# ─────────────────────────────────────────────────────────

_RUN_CLIPPY: bool = True


def _pool_init(run_clippy: bool) -> None:
    global _RUN_CLIPPY
    _RUN_CLIPPY = run_clippy


def _worker(record: dict) -> tuple[bool, dict, str]:
    return validate_record(record)


# ─────────────────────────────────────────────────────────
#  ROTATING OUTPUT WRITER
# ─────────────────────────────────────────────────────────

class RotatingWriter:
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
    code_snippet = (record.get("fixed_code") or "")[:400]
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{_SEP}\n")
        f.write(f"  {ts}  id={rid}\n")
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
                if not (record.get("fixed_code") or "").strip():
                    continue
                yield fpath, record
                total_yielded += 1
                if limit and total_yielded >= limit:
                    return


# ─────────────────────────────────────────────────────────
#  PROGRESS
# ─────────────────────────────────────────────────────────

def _progress(i: int, total: int, ok: int, fail: int, t0: float) -> None:
    elapsed = time.time() - t0
    rate    = i / elapsed if elapsed else 0
    eta_s   = (total - i) / rate if rate else 0
    pct     = i / total * 100 if total else 0
    filled  = int(30 * i / total) if total else 0
    bar     = "█" * filled + "░" * (30 - filled)
    print(
        f"\r  [{bar}] {pct:5.1f}%  {i:>{len(str(total))}}/{total}"
        f"  ✓ {ok}  ✗ {fail}  {rate:5.1f} rec/s  ETA {eta_s/60:.1f}m   ",
        end="", flush=True,
    )


# ─────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path     = output_dir / "validation_failures.log"
    writer       = RotatingWriter(output_dir, prefix="validated")
    failures_dir = output_dir / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)
    fail_writer  = RotatingWriter(failures_dir, prefix="failed")

    print("\nScanning input files…", flush=True)
    all_records = list(iter_records(input_dir, limit=args.limit))
    total = len(all_records)
    print(f"\nLoaded {total:,} records  |  workers: {args.workers}  |  output: {output_dir.resolve()}\n")

    ok = fail = 0
    t0 = time.time()

    global _RUN_CLIPPY
    _RUN_CLIPPY = not args.no_clippy

    if args.workers == 1:
        for i, (_fp, record) in enumerate(all_records, 1):
            passed, updated, reason = validate_record(record)
            if passed:
                writer.write(updated); ok += 1
            else:
                _log_failure(log_path, record, reason)
                fail_writer.write(updated); fail += 1
            if i % 50 == 0 or i == total:
                _progress(i, total, ok, fail, t0)
    else:
        CHUNK = args.workers * 4
        records_only = [r for _, r in all_records]
        i = 0
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_pool_init,
            initargs=(not args.no_clippy,),
        ) as pool:
            offset = 0
            while offset < len(records_only):
                chunk = records_only[offset:offset + CHUNK]
                offset += CHUNK
                futs = {pool.submit(_worker, rec): rec for rec in chunk}
                for fut in as_completed(futs):
                    record = futs[fut]
                    i += 1
                    try:
                        passed, updated, reason = fut.result()
                    except Exception as exc:
                        passed, updated, reason = False, _make_failure_record(record, str(exc)), str(exc)
                    if passed:
                        writer.write(updated); ok += 1
                    else:
                        _log_failure(log_path, record, reason)
                        fail_writer.write(updated); fail += 1
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
    print(f"  Output  : {output_dir.resolve()}")
    for p in writer.files_written:
        size_mb = p.stat().st_size / 1024 / 1024 if p.exists() else 0
        print(f"    {p.name}  ({size_mb:.1f} MB)")
    if fail:
        print(f"  Failures: {failures_dir.resolve()}")
    print()


# ─────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Batch-validate Rust JSONL dataset — pure std only, no external crates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir",  required=True,  help="Directory containing .jsonl files")
    p.add_argument("--output-dir", default="validated", help="Output directory")
    p.add_argument("--workers",    type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--limit",      type=int, default=None, help="Process at most N records")
    p.add_argument("--no-clippy",  action="store_true", default=False, help="Skip clippy")
    args = p.parse_args()

    if not Path(args.input_dir).is_dir():
        print(f"[ERROR] --input-dir not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    run(args)


if __name__ == "__main__":
    main()
