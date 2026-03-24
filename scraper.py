#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║         RustScrape  —  Rust Training Data Collector      ║
║        crates.io · rust-lang GitHub · Official Docs      ║
╚══════════════════════════════════════════════════════════╝

Collects high-quality Rust source data from three sources:

  1. crates.io  — Top N crates by downloads, extract all .rs files
  2. GitHub     — rust-lang org repos + curated ecosystem crates
  3. Docs       — The Rust Book, Reference, Nomicon, RFCs (markdown)

Output: JSONL files per source in <output-dir>/
  ./rust_data/crates/    — one JSONL per crate
  ./rust_data/github/    — one JSONL per repo
  ./rust_data/docs/      — one JSONL per doc source

Each record:
  {
    "source":    "crates.io" | "github" | "docs",
    "origin":    "tokio",
    "path":      "src/runtime/mod.rs",
    "content":   "...",
    "tokens_est": 1842,
    "scraped_at": 1234567890
  }

Requirements:
    pip install requests rich

Usage:
    python scraper.py crates  --top 500 --output-dir ./rust_data
    python scraper.py github  --output-dir ./rust_data [--token ghp_xxx]
    python scraper.py docs    --output-dir ./rust_data
    python scraper.py all     --output-dir ./rust_data [--token ghp_xxx]
    python scraper.py stats   --output-dir ./rust_data

GitHub token (optional but strongly recommended — raises limit from 60 to 5000 req/hr):
    export GITHUB_TOKEN=ghp_yourtoken
    python scraper.py github --output-dir ./rust_data
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import os
import re
import signal
import sys
import tarfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Iterator, Optional

import requests
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
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
    "source":  "bold #7c5cbf",
    "file":    "italic #5fafd7",
    "count":   "bold white",
    "header":  "bold white on #1a1a2e",
    "skip":    "dim yellow",
})
console = Console(theme=THEME, highlight=False)
log = logging.getLogger("rustscrape")

# ─────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────

CRATES_API       = "https://crates.io/api/v1"
CRATES_CDN       = "https://static.crates.io/crates"
GITHUB_API       = "https://api.github.com"
USER_AGENT       = "rust-training-scraper/1.0 (research; contact via github)"
CRATES_API_DELAY = 1.1      # seconds between crates.io API calls (crawlers policy)
GITHUB_DELAY     = 0.5      # seconds between GitHub API calls
CDN_DELAY        = 0.5      # seconds between crate tarball downloads — be polite
DOWNLOAD_TIMEOUT = 60
API_TIMEOUT      = 20
MAX_FILE_BYTES   = 500_000  # skip files >500KB (usually generated or data files)
MIN_FILE_BYTES   = 100      # skip tiny files (empty mods, just re-exports)

# Rough token estimate: ~4 chars per token for code
def estimate_tokens(text: str) -> int:
    return len(text) // 4

# ─────────────────────────────────────────────────────────
#  SHUTDOWN
# ─────────────────────────────────────────────────────────

_shutdown = Event()

def _handle_sig(sig, frame):
    if not _shutdown.is_set():
        console.print("\n[warn]Interrupt received — finishing current item then stopping cleanly.[/warn]")
        _shutdown.set()

signal.signal(signal.SIGINT,  _handle_sig)
signal.signal(signal.SIGTERM, _handle_sig)

# ─────────────────────────────────────────────────────────
#  SESSION STATS
# ─────────────────────────────────────────────────────────

@dataclass
class Stats:
    files_written:  int   = 0
    files_skipped:  int   = 0
    bytes_written:  int   = 0
    tokens_est:     int   = 0
    errors:         int   = 0
    current_item:   str   = ""
    current_file:   str   = ""
    start_time:     float = field(default_factory=time.time)

    @property
    def elapsed(self) -> str:
        return time.strftime("%H:%M:%S", time.gmtime(time.time() - self.start_time))

    @property
    def mb_written(self) -> float:
        return self.bytes_written / 1_048_576

# ─────────────────────────────────────────────────────────
#  MANIFEST — resume support
# ─────────────────────────────────────────────────────────

class Manifest:
    """
    Tracks which items (crates, repos, doc pages) have been fully scraped.
    Stored as a simple newline-delimited text file of completed item IDs.
    On resume, already-completed items are skipped instantly.
    """

    def __init__(self, output_dir: str, source: str):
        self._path = Path(output_dir) / f".manifest_{source}.txt"
        self._done: set = set()
        self._load()

    def _load(self):
        if self._path.exists():
            with open(self._path, encoding="utf-8") as f:
                self._done = {line.strip() for line in f if line.strip()}

    def is_done(self, item_id: str) -> bool:
        return item_id in self._done

    def mark_done(self, item_id: str) -> None:
        if item_id not in self._done:
            self._done.add(item_id)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(item_id + "\n")

    def count(self) -> int:
        return len(self._done)

# ─────────────────────────────────────────────────────────
#  OUTPUT HELPERS
# ─────────────────────────────────────────────────────────

def output_path(output_dir: str, source: str, name: str) -> Path:
    p = Path(output_dir) / source / f"{name}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def write_record(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")

def make_record(
    source: str, origin: str, path: str, content: str, **extra
) -> dict:
    return {
        "source":     source,
        "origin":     origin,
        "path":       path,
        "content":    content,
        "tokens_est": estimate_tokens(content),
        "scraped_at": int(time.time()),
        **extra,
    }

# ─────────────────────────────────────────────────────────
#  HTTP SESSION
# ─────────────────────────────────────────────────────────

def make_session(github_token: Optional[str] = None) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    if github_token:
        s.headers.update({"Authorization": f"Bearer {github_token}"})
    return s

# ─────────────────────────────────────────────────────────
#  FILE FILTERS
# ─────────────────────────────────────────────────────────

# Paths to skip — generated code, fixtures, test data, vendor dirs
_SKIP_PATH_PARTS = {
    "vendor", "node_modules", ".git", ".github", "target",
    "generated", "gen", "proto",
}
_SKIP_PATH_SUFFIXES = {
    ".pb.rs",          # protobuf generated
    "_generated.rs",   # common generated pattern
}
_SKIP_FILENAMES = {
    "bindgen_output.rs", "bindings.rs",
}

def should_skip_path(path_str: str) -> bool:
    parts = set(Path(path_str).parts)
    if parts & _SKIP_PATH_PARTS:
        return True
    name = Path(path_str).name
    if name in _SKIP_FILENAMES:
        return True
    for suffix in _SKIP_PATH_SUFFIXES:
        if path_str.endswith(suffix):
            return True
    return False

def is_quality_rust(content: str) -> bool:
    """
    Basic quality filter — reject files that are just
    auto-generated, empty, or contain no real code.
    """
    if not content.strip():
        return False
    # Must have at least one fn, struct, impl, trait, enum, or use
    if not re.search(r"\b(fn |struct |impl |trait |enum |use |pub |mod )\b", content):
        return False
    # Reject files that are >80% non-ASCII (binary snuck through)
    ascii_ratio = sum(1 for c in content if ord(c) < 128) / max(len(content), 1)
    if ascii_ratio < 0.8:
        return False
    return True

# ─────────────────────────────────────────────────────────
#  LIVE DISPLAY
# ─────────────────────────────────────────────────────────

def _header_panel(stats: Stats, source_name: str) -> Panel:
    g = Table.grid(padding=(0, 3))
    g.add_column(); g.add_column(); g.add_column(); g.add_column()
    g.add_row(
        f"[source]◈ {source_name}[/source]",
        f"[success]✓ {stats.files_written:,}[/success] files",
        f"[count]{stats.mb_written:.1f} MB[/count]  ~{stats.tokens_est:,} tokens",
        f"[info]{stats.elapsed}[/info]  [skip]{stats.files_skipped:,} skipped[/skip]",
    )
    return Panel(g, style="header", padding=(0, 1))

def _item_panel(stats: Stats) -> Panel:
    item = stats.current_item or "[dim]—[/dim]"
    f    = stats.current_file or ""
    return Panel(
        f"[source]{item}[/source]\n[file]{f}[/file]",
        title="[bold]Current[/bold]",
        border_style="dim blue",
        height=5,
    )

def _make_layout() -> Layout:
    l = Layout(name="root")
    l.split_column(Layout(name="header", size=3), Layout(name="item", size=5))
    return l

# ─────────────────────────────────────────────────────────
#  SOURCE 1: crates.io
# ─────────────────────────────────────────────────────────

# Crates to explicitly exclude (pure proc-macro wrappers, test scaffolding, etc.)
_CRATES_EXCLUDE = {
    "winapi", "winapi-i686-pc-windows-gnu", "winapi-x86_64-pc-windows-gnu",
    "windows-sys", "windows-targets",  # mostly generated FFI bindings
}

def fetch_top_crates(session: requests.Session, top: int) -> list[dict]:
    """Return metadata for the top N crates by all-time downloads."""
    crates = []
    per_page = 100
    page = 1
    console.print(f"[info]Fetching top {top} crates from crates.io API...[/info]")

    while len(crates) < top:
        url = f"{CRATES_API}/crates?sort=downloads&per_page={per_page}&page={page}"
        try:
            r = session.get(url, timeout=API_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            batch = data.get("crates", [])
            if not batch:
                break
            crates.extend(batch)
            console.print(f"[info]  Page {page}: {len(batch)} crates ({len(crates)} total)[/info]")
            page += 1
            time.sleep(CRATES_API_DELAY)  # respect crawlers policy
        except Exception as e:
            log.warning(f"API error page {page}: {e}")
            time.sleep(CRATES_API_DELAY * 3)

    return crates[:top]

def download_crate_tarball(
    session: requests.Session, name: str, version: str
) -> Optional[bytes]:
    """Download a .crate tarball from the CDN. Returns raw bytes or None."""
    url = f"{CRATES_CDN}/{name}/{version}/download"
    try:
        r = session.get(url, timeout=DOWNLOAD_TIMEOUT, stream=False)
        r.raise_for_status()
        return r.content
    except Exception as e:
        log.warning(f"Download failed {name} {version}: {e}")
        return None

# Doc files worth keeping from crate tarballs
_DOC_STEMS = frozenset({"readme", "changelog", "changes", "contributing", "architecture", "design"})
_DOC_EXTS  = frozenset({".md", ".rst", ".txt"})

def _is_crate_doc(rel_path: str) -> bool:
    """
    True for README/CHANGELOG-style files and any markdown inside a docs/ or doc/ dir.
    Skips .github/, benches/, test data, etc.
    """
    p    = Path(rel_path)
    stem = p.stem.lower()
    ext  = p.suffix.lower()
    # Named documentation files at any depth (README.md, CHANGELOG.rst, etc.)
    if stem in _DOC_STEMS and (ext in _DOC_EXTS or ext == ""):
        return True
    # Any markdown in a top-level docs/ or doc/ directory
    if ext == ".md" and len(p.parts) > 1 and p.parts[0].lower() in {"docs", "doc"}:
        return True
    return False

def extract_rs_files(tarball_bytes: bytes) -> Iterator[tuple[str, str]]:
    """
    Yield (path, content) for .rs source files AND documentation files
    (README, CHANGELOG, docs/*.md) found in a .crate tarball.
    """
    try:
        with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                is_rs  = member.name.endswith(".rs")
                # Strip the crate-version prefix (e.g. tokio-1.35.0/src/lib.rs → src/lib.rs)
                parts    = member.name.split("/", 1)
                rel_path = parts[1] if len(parts) > 1 else member.name
                is_doc = _is_crate_doc(rel_path)
                if not (is_rs or is_doc):
                    continue
                if member.size > MAX_FILE_BYTES or member.size < MIN_FILE_BYTES:
                    continue
                if should_skip_path(rel_path):
                    continue
                try:
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    content = f.read().decode("utf-8", errors="replace")
                    if is_rs and not is_quality_rust(content):
                        continue
                    yield rel_path, content
                except Exception:
                    continue
    except Exception as e:
        log.warning(f"Tarball extraction error: {e}")

def scrape_crates(args: argparse.Namespace) -> None:
    outdir  = args.output_dir
    session = make_session()
    manifest = Manifest(outdir, "crates")
    stats   = Stats()
    layout  = _make_layout()

    crates_meta = fetch_top_crates(session, args.top)
    to_scrape   = [
        c for c in crates_meta
        if c["id"] not in _CRATES_EXCLUDE and not manifest.is_done(c["id"])
    ]
    already = manifest.count()

    console.print(
        f"[info]Top {args.top} crates fetched. "
        f"{already} already done, {len(to_scrape)} to scrape.[/info]"
    )

    with Live(layout, console=console, refresh_per_second=4, screen=False):

        def _refresh():
            layout["header"].update(_header_panel(stats, "crates.io"))
            layout["item"].update(_item_panel(stats))

        _refresh()

        for crate in to_scrape:
            if _shutdown.is_set():
                break

            name    = crate["id"]
            version = crate["newest_version"]
            stats.current_item = f"{name}  v{version}"
            stats.current_file = "downloading tarball..."
            _refresh()

            tarball = download_crate_tarball(session, name, version)
            if tarball is None:
                stats.errors += 1
                _refresh()
                time.sleep(CDN_DELAY)
                continue

            out_path = output_path(outdir, "crates", name)
            file_count = 0

            for rel_path, content in extract_rs_files(tarball):
                if _shutdown.is_set():
                    break
                stats.current_file = rel_path
                record = make_record(
                    source="crates.io", origin=name, path=rel_path,
                    content=content, version=version,
                    downloads=crate.get("downloads", 0),
                )
                write_record(out_path, record)
                stats.files_written += 1
                stats.bytes_written += len(content.encode("utf-8"))
                stats.tokens_est    += record["tokens_est"]
                file_count          += 1
                _refresh()

            if file_count == 0:
                stats.files_skipped += 1
            manifest.mark_done(name)
            time.sleep(CDN_DELAY)
            _refresh()

    _print_final_stats(stats, "crates.io", manifest)

# ─────────────────────────────────────────────────────────
#  SOURCE 2: GitHub — rust-lang org + curated ecosystem
# ─────────────────────────────────────────────────────────

# Core rust-lang repos
_RUSTLANG_REPOS = [
    "rust",           # the compiler & stdlib — highest value
    "cargo",          # package manager
    "rustfmt",
    "clippy",
    "rust-analyzer",
    "book",           # The Rust Book
    "reference",      # The Reference
    "nomicon",        # The Nomicon (unsafe book)
    "rfcs",           # design decisions
    "async-book",
    "rustlings",      # exercises
    "rust-by-example",
]

# Curated ecosystem crates — the ones every Rust dev uses
_ECOSYSTEM_REPOS = [
    # ── Async runtime & networking ───────────────────────────────────────────
    ("tokio-rs",         "tokio"),
    ("tokio-rs",         "axum"),
    ("tokio-rs",         "tracing"),
    ("tokio-rs",         "mio"),
    ("hyperium",         "hyper"),
    ("hyperium",         "http"),
    ("hyperium",         "tonic"),           # gRPC
    ("seanmonstar",      "reqwest"),          # most-used HTTP client
    ("seanmonstar",      "warp"),
    ("tower-rs",         "tower"),
    ("tower-rs",         "tower-http"),
    ("quinn-rs",         "quinn"),            # QUIC protocol
    ("smol-rs",          "smol"),
    ("actix",            "actix-web"),
    ("cloudflare",       "quiche"),           # QUIC/HTTP3
    # ── Serialization & parsing ──────────────────────────────────────────────
    ("serde-rs",         "serde"),
    ("serde-rs",         "serde_json"),
    ("toml-rs",          "toml"),             # TOML (used everywhere)
    ("Geal",             "nom"),              # foundational parser combinator
    ("pest-parser",      "pest"),             # PEG parser
    ("dtolnay",          "syn"),
    ("dtolnay",          "quote"),
    ("dtolnay",          "proc-macro2"),
    ("dtolnay",          "serde-yaml"),
    ("dtolnay",          "semver"),
    ("messagerie-rs",    "postcard"),         # embedded serialization
    # ── Error handling & diagnostics ────────────────────────────────────────
    ("dtolnay",          "thiserror"),
    ("dtolnay",          "anyhow"),
    ("zkat",             "miette"),           # great diagnostic patterns
    ("eyre-rs",          "eyre"),
    # ── Concurrency & data structures ────────────────────────────────────────
    ("rayon-rs",         "rayon"),
    ("crossbeam-rs",     "crossbeam"),
    ("tokio-rs",         "bytes"),            # widely used byte buffer
    ("indexmap-rs",      "indexmap"),         # ordered hash map
    ("contain-rs",       "fixedbitset"),
    ("bluss",            "petgraph"),         # graph data structures
    ("servo",            "rust-smallvec"),    # small vec optimisation
    # ── Proc macros & codegen ────────────────────────────────────────────────
    ("clap-rs",          "clap"),
    ("bitflags",         "bitflags"),
    ("rust-num",         "num"),
    ("rust-itertools",   "itertools"),
    ("matklad",          "once_cell"),
    # ── Databases & storage ──────────────────────────────────────────────────
    ("launchbadge",      "sqlx"),
    ("diesel-rs",        "diesel"),
    ("rusqlite",         "rusqlite"),
    ("spacejam",         "sled"),             # embedded database
    ("cberner",          "redb"),             # embedded database
    # ── Crypto & security ────────────────────────────────────────────────────
    ("rustls",           "rustls"),
    ("sfackler",         "rust-openssl"),
    ("dalek-cryptography","curve25519-dalek"),
    ("RustCrypto",       "hashes"),
    ("RustCrypto",       "signatures"),
    # ── CLI tools — exemplary real-world Rust ────────────────────────────────
    ("BurntSushi",       "ripgrep"),
    ("BurntSushi",       "xsv"),             # CSV toolkit — clean idiomatic code
    ("BurntSushi",       "regex"),
    ("rust-lang",        "regex"),
    ("sharkdp",          "bat"),
    ("sharkdp",          "fd"),              # modern find replacement
    ("sharkdp",          "hyperfine"),       # benchmarking tool
    ("dandavison",       "delta"),           # git diff viewer
    ("ajeetdsouza",      "zoxide"),          # smart cd
    ("bootandy",         "dust"),            # du replacement
    ("imsnif",           "bandwhich"),       # network utilisation
    ("casey",            "just"),
    ("XAMPPRocky",       "tokei"),
    ("ogham",            "exa"),
    ("crate-ci",         "typos"),           # spell checker — good string handling patterns
    ("orhun",            "git-cliff"),       # changelog generator
    # ── TUI & terminal ───────────────────────────────────────────────────────
    ("extrawurst",       "gitui"),           # full git TUI application
    ("zellij-org",       "zellij"),          # terminal multiplexer
    ("ratatui-org",      "ratatui"),         # TUI framework (tui-rs successor)
    # ── Editors & large applications ─────────────────────────────────────────
    ("helix-editor",     "helix"),           # production editor — ~100k lines of clean Rust
    ("starship",         "starship"),        # cross-platform shell prompt — tons of patterns
    # ── Build tools & dev infrastructure ─────────────────────────────────────
    ("LukeMathWalker",   "cargo-chef"),
    ("mozilla",          "sccache"),         # shared compilation cache
    ("rust-lang",        "crates.io"),       # the crates.io site itself is Rust
    ("cargo-bins",       "cargo-binstall"),
    # ── WebAssembly ──────────────────────────────────────────────────────────
    ("bytecodealliance", "wasmtime"),
    ("rustwasm",         "wasm-bindgen"),
    ("rustwasm",         "wasm-pack"),
    # ── Miscellaneous high-quality crates ────────────────────────────────────
    ("image-rs",         "image"),
    ("rust-random",      "rand"),
    ("chronotope",       "chrono"),
    ("uuid-rs",          "uuid"),
    ("mitsuhiko",        "minijinja"),
    ("nix-rust",         "nix"),
    ("nickel-org",       "nickel.rs"),
    ("Byron",            "gitoxide"),        # pure-Rust git implementation — very large, very clean
    ("vectordotdev",     "vector"),          # data pipeline — enterprise-grade Rust
    ("meilisearch",      "meilisearch"),     # search engine — large real-world app
    ("alacritty",        "alacritty"),       # terminal emulator
    ("rust-embedded",    "embedded-hal"),    # embedded systems abstraction
    ("embassy-rs",       "embassy"),         # async embedded framework
    ("smoltcp-rs",       "smoltcp"),         # bare-metal network stack
]

def gh_headers_ok(session: requests.Session) -> bool:
    """Check GitHub rate limit and warn if low."""
    try:
        r = session.get(f"{GITHUB_API}/rate_limit", timeout=API_TIMEOUT)
        r.raise_for_status()
        remaining = r.json()["resources"]["core"]["remaining"]
        reset_at  = r.json()["resources"]["core"]["reset"]
        if remaining < 50:
            wait = max(0, reset_at - time.time())
            console.print(
                f"[warn]GitHub rate limit low ({remaining} remaining). "
                f"Resets in {int(wait/60)}m. Consider a --token.[/warn]"
            )
        else:
            console.print(f"[info]GitHub rate limit: {remaining} requests remaining[/info]")
        return remaining > 0
    except Exception as e:
        log.warning(f"Rate limit check failed: {e}")
        return True

def gh_get_default_branch(session: requests.Session, owner: str, repo: str) -> str:
    try:
        r = session.get(f"{GITHUB_API}/repos/{owner}/{repo}", timeout=API_TIMEOUT)
        r.raise_for_status()
        return r.json().get("default_branch", "main")
    except Exception:
        return "main"

def gh_download_repo_archive(
    session: requests.Session, owner: str, repo: str, branch: str
) -> Optional[bytes]:
    """Download the entire repo as a tar.gz archive."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/tarball/{branch}"
    try:
        r = session.get(url, timeout=DOWNLOAD_TIMEOUT, allow_redirects=True)
        r.raise_for_status()
        return r.content
    except Exception as e:
        log.warning(f"Archive download failed {owner}/{repo}: {e}")
        return None

def extract_rs_from_github_archive(
    tarball_bytes: bytes,
    extensions: tuple = (".rs",),
    also_markdown: bool = False,
) -> Iterator[tuple[str, str]]:
    """
    Yield (path, content) from a GitHub archive tarball.
    Optionally includes .md files too (for docs repos).
    """
    if also_markdown:
        extensions = extensions + (".md",)
    try:
        with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                if not any(member.name.endswith(ext) for ext in extensions):
                    continue
                if member.size > MAX_FILE_BYTES or member.size < MIN_FILE_BYTES:
                    continue
                # Strip the GitHub archive prefix (owner-repo-sha/...)
                parts = member.name.split("/", 1)
                rel_path = parts[1] if len(parts) > 1 else member.name
                if should_skip_path(rel_path):
                    continue
                try:
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    content = f.read().decode("utf-8", errors="replace")
                    if member.name.endswith(".rs") and not is_quality_rust(content):
                        continue
                    if len(content.strip()) >= MIN_FILE_BYTES:
                        yield rel_path, content
                except Exception:
                    continue
    except Exception as e:
        log.warning(f"GitHub archive extraction error: {e}")

def _scrape_one_repo(
    session: requests.Session,
    owner: str,
    repo: str,
    out_path: Path,
    stats: Stats,
    also_markdown: bool = False,
) -> int:
    """Scrape a single GitHub repo. Returns number of files written."""
    branch = gh_get_default_branch(session, owner, repo)
    time.sleep(GITHUB_DELAY)

    tarball = gh_download_repo_archive(session, owner, repo, branch)
    if tarball is None:
        stats.errors += 1
        return 0

    count = 0
    for rel_path, content in extract_rs_from_github_archive(
        tarball, also_markdown=also_markdown
    ):
        if _shutdown.is_set():
            break
        stats.current_file = rel_path
        record = make_record(
            source="github", origin=f"{owner}/{repo}",
            path=rel_path, content=content,
            owner=owner, repo=repo,
        )
        write_record(out_path, record)
        stats.files_written += 1
        stats.bytes_written += len(content.encode("utf-8"))
        stats.tokens_est    += record["tokens_est"]
        count               += 1

    return count

def scrape_github(args: argparse.Namespace) -> None:
    outdir   = args.output_dir
    token    = getattr(args, "token", None) or os.environ.get("GITHUB_TOKEN")
    session  = make_session(token)
    manifest = Manifest(outdir, "github")
    stats    = Stats()
    layout   = _make_layout()

    if not token:
        console.print(
            "[warn]No GitHub token — rate limited to 60 req/hr. "
            "Set GITHUB_TOKEN env var or pass --token for 5000/hr.[/warn]"
        )

    gh_headers_ok(session)

    # Build full repo list
    all_repos: list[tuple[str, str, bool]] = []   # (owner, repo, also_markdown)
    for repo in _RUSTLANG_REPOS:
        all_repos.append(("rust-lang", repo, True))
    for owner, repo in _ECOSYSTEM_REPOS:
        all_repos.append((owner, repo, True))   # include README/docs markdown for all repos

    to_scrape = [
        (o, r, md) for o, r, md in all_repos
        if not manifest.is_done(f"{o}/{r}")
    ]
    already = manifest.count()

    console.print(
        f"[info]{len(all_repos)} repos total. "
        f"{already} already done, {len(to_scrape)} to scrape.[/info]"
    )

    with Live(layout, console=console, refresh_per_second=4, screen=False):

        def _refresh():
            layout["header"].update(_header_panel(stats, "GitHub"))
            layout["item"].update(_item_panel(stats))

        _refresh()

        for owner, repo, also_md in to_scrape:
            if _shutdown.is_set():
                break

            stats.current_item = f"{owner}/{repo}"
            stats.current_file = "downloading archive..."
            _refresh()

            repo_id  = f"{owner}/{repo}"
            out_path = output_path(outdir, "github", repo_id.replace("/", "__"))

            count = _scrape_one_repo(
                session, owner, repo, out_path, stats, also_markdown=also_md
            )

            if count == 0:
                stats.files_skipped += 1
            manifest.mark_done(repo_id)
            time.sleep(GITHUB_DELAY)
            _refresh()

    _print_final_stats(stats, "GitHub", manifest)

# ─────────────────────────────────────────────────────────
#  SOURCE 3: Official Rust Docs (direct URL scrape)
# ─────────────────────────────────────────────────────────

# Each entry: (label, base_url, is_single_page)
_DOC_SOURCES = [
    # ── Official rust-lang docs ──────────────────────────────────────────────
    ("rust-book",          "https://doc.rust-lang.org/book/",                    False),
    ("rust-reference",     "https://doc.rust-lang.org/reference/",               False),
    ("rust-nomicon",       "https://doc.rust-lang.org/nomicon/",                 False),
    ("rust-async-book",    "https://rust-lang.github.io/async-book/",            False),
    ("rust-edition-guide", "https://doc.rust-lang.org/edition-guide/",           False),
    ("rust-cargo",         "https://doc.rust-lang.org/cargo/",                   False),
    ("rust-clippy",        "https://doc.rust-lang.org/clippy/",                  False),
    ("rust-error-index",   "https://doc.rust-lang.org/error_codes/error-index.html", True),
    ("rust-std-docs",      "https://doc.rust-lang.org/std/index.html",           True),
    # ── High-quality community books ────────────────────────────────────────
    ("rust-patterns",      "https://rust-unofficial.github.io/patterns/",        False),
    ("rust-too-many-lists","https://rust-unofficial.github.io/too-many-lists/",  False),
    ("rust-tlborm",        "https://veykril.github.io/tlborm/",                  False),
    ("rust-perf-book",     "https://nnethercote.github.io/perf-book/",           False),
    ("rust-comprehensive", "https://google.github.io/comprehensive-rust/",       False),
    ("rust-cookbook",      "https://rust-lang-nursery.github.io/rust-cookbook/", False),
]

# Known chapter/section paths for each book.
# Books NOT listed here get auto-discovered from their index page.
_BOOK_CHAPTERS: dict[str, list[str]] = {
    "rust-book": [
        "title-page.html","foreword.html","ch00-00-introduction.html",
        "ch01-00-getting-started.html","ch01-01-installation.html","ch01-02-hello-world.html","ch01-03-hello-cargo.html",
        "ch02-00-guessing-game-tutorial.html",
        "ch03-00-common-programming-concepts.html","ch03-01-variables-and-mutability.html",
        "ch03-02-data-types.html","ch03-03-how-functions-work.html","ch03-04-comments.html",
        "ch03-05-control-flow.html",
        "ch04-00-understanding-ownership.html","ch04-01-what-is-ownership.html",
        "ch04-02-references-and-borrowing.html","ch04-03-slices.html",
        "ch05-00-structs.html","ch05-01-defining-structs.html","ch05-02-example-structs.html","ch05-03-method-syntax.html",
        "ch06-00-enums.html","ch06-01-defining-an-enum.html","ch06-02-match.html","ch06-03-if-let.html",
        "ch07-00-managing-growing-projects-with-packages-crates-and-modules.html",
        "ch07-01-packages-and-crates.html","ch07-02-defining-modules-to-control-scope-and-privacy.html",
        "ch07-03-paths-for-referring-to-an-item-in-the-module-tree.html",
        "ch07-04-bringing-paths-into-scope-with-the-use-keyword.html","ch07-05-separating-modules-into-different-files.html",
        "ch08-00-common-collections.html","ch08-01-vectors.html","ch08-02-strings.html","ch08-03-hash-maps.html",
        "ch09-00-error-handling.html","ch09-01-unrecoverable-errors-with-panic.html",
        "ch09-02-recoverable-errors-with-result.html","ch09-03-to-panic-or-not-to-panic.html",
        "ch10-00-generics.html","ch10-01-syntax.html","ch10-02-traits.html","ch10-03-lifetime-syntax.html",
        "ch11-00-testing.html","ch11-01-writing-tests.html","ch11-02-running-tests.html","ch11-03-test-organization.html",
        "ch12-00-an-io-project.html","ch12-01-accepting-command-line-arguments.html",
        "ch12-02-reading-a-file.html","ch12-03-improving-error-handling-and-modularity.html",
        "ch12-04-testing-the-librarys-functionality.html","ch12-05-working-with-environment-variables.html",
        "ch12-06-writing-to-stderr-instead-of-stdout.html",
        "ch13-00-functional-features.html","ch13-01-closures.html","ch13-02-iterators.html",
        "ch13-03-improving-our-io-project.html","ch13-04-performance.html",
        "ch14-00-more-about-cargo.html","ch14-01-release-profiles.html","ch14-02-publishing-to-crates-io.html",
        "ch14-03-cargo-workspaces.html","ch14-04-installing-binaries.html","ch14-05-extending-cargo.html",
        "ch15-00-smart-pointers.html","ch15-01-box.html","ch15-02-deref.html","ch15-03-drop.html",
        "ch15-04-rc.html","ch15-05-interior-mutability.html","ch15-06-reference-cycles.html",
        "ch16-00-concurrency.html","ch16-01-threads.html","ch16-02-message-passing.html",
        "ch16-03-shared-state.html","ch16-04-extensible-concurrency-sync-and-send.html",
        "ch17-00-oop.html","ch17-01-what-is-oo.html","ch17-02-trait-objects.html","ch17-03-oo-design-patterns.html",
        "ch18-00-patterns.html","ch18-01-all-the-places-for-patterns.html",
        "ch18-02-refutability.html","ch18-03-pattern-syntax.html",
        "ch19-00-advanced-features.html","ch19-01-unsafe-rust.html","ch19-02-advanced-traits.html",
        "ch19-03-advanced-types.html","ch19-04-advanced-functions-and-closures.html","ch19-05-macros.html",
        "ch20-00-final-project-a-web-server.html","ch20-01-single-threaded.html",
        "ch20-02-multithreaded.html","ch20-03-graceful-shutdown-and-cleanup.html",
        "appendix-00.html","appendix-01-keywords.html","appendix-02-operators.html",
        "appendix-03-derivable-traits.html","appendix-04-useful-development-tools.html",
        "appendix-05-editions.html","appendix-06-translation.html","appendix-07-nightly-rust.html",
    ],
    "rust-nomicon": [
        "intro.html","meet-safe-and-unsafe.html","data.html","ownership.html","references.html",
        "lifetimes.html","lifetime-mismatch.html","lifetime-elision.html","lifetime-advanced.html",
        "subtyping.html","drop-flags.html","destructor-madness.html","unbounded-lifetimes.html",
        "phantom-data.html","splitting-borrows.html","vec/vec.html","vec/vec-layout.html",
        "vec/vec-alloc.html","vec/vec-push-pop.html","vec/vec-dealloc.html",
        "vec/vec-final.html","arc-mutex/arc-mutex.html","arc-mutex/arc.html",
        "arc-mutex/arc-layout.html","arc-mutex/arc-base.html","arc-mutex/arc-clone.html",
        "arc-mutex/arc-drop.html","arc-mutex/arc-final.html",
        "other-reprs.html","atomics.html","send-and-sync.html","races.html",
        "working-with-unsafe.html","safe-unsafe-meaning.html","what-unsafe-does.html",
        "hrtb.html","exotic-sizes.html","conversions.html","coercions.html","dot-operator.html",
        "casts.html","transmutes.html","checked.html","uninitialized.html","ptr.html",
    ],
    # The Reference — all major sections and sub-pages
    "rust-reference": [
        "introduction.html","notation.html","keywords.html","identifiers.html",
        "comments.html","whitespace.html","tokens.html",
        "macros.html","macros-by-example.html","procedural-macros.html",
        "crates-and-source-files.html","conditional-compilation.html",
        "items.html",
        "items/modules.html","items/extern-crates.html","items/use-declarations.html",
        "items/functions.html","items/type-aliases.html","items/structs.html",
        "items/enumerations.html","items/unions.html","items/constant-items.html",
        "items/static-items.html","items/traits.html","items/implementations.html",
        "items/external-blocks.html","items/generics.html","items/associated-items.html",
        "items/visibility-and-privacy.html",
        "attributes.html",
        "attributes/testing.html","attributes/derive.html","attributes/diagnostics.html",
        "attributes/code-generation.html","attributes/limits.html","attributes/type_system.html",
        "statements-and-expressions.html","statements.html","expressions.html",
        "expressions/literal-expr.html","expressions/path-expr.html","expressions/block-expr.html",
        "expressions/operator-expr.html","expressions/grouped-expr.html","expressions/array-expr.html",
        "expressions/tuple-expr.html","expressions/struct-expr.html","expressions/call-expr.html",
        "expressions/method-call-expr.html","expressions/field-expr.html","expressions/index-expr.html",
        "expressions/range-expr.html","expressions/closure-expr.html","expressions/loop-expr.html",
        "expressions/if-expr.html","expressions/match-expr.html","expressions/return-expr.html",
        "expressions/await-expr.html","expressions/underscore-expr.html",
        "patterns.html","type-system.html","types.html",
        "types/boolean.html","types/numeric.html","types/textual.html","types/never.html",
        "types/tuple.html","types/array.html","types/slice.html","types/struct.html",
        "types/enum.html","types/union.html","types/function-item.html","types/closure.html",
        "types/pointer.html","types/reference.html","types/fn.html",
        "types/trait-object.html","types/impl-trait.html","types/parameters.html",
        "dynamically-sized-types.html","type-layout.html","interior-mutability.html",
        "subtyping.html","trait-bounds.html","type-coercions.html","destructors.html",
        "lifetime-elision.html","special-types-and-traits.html",
        "names.html","names/namespaces.html","names/scopes.html","names/name-resolution.html",
        "names/paths.html","names/identifier-resolution.html","names/preludes.html",
        "names/extern-prelude.html","names/tool-prelude.html",
        "memory-model.html","linkage.html","inline-assembly.html","unsafety.html",
        "behavior-considered-undefined.html","behavior-not-considered-unsafe.html",
        "const-eval.html","application-binary-interface.html",
        "influences.html","glossary.html",
    ],
    # Async Book
    "rust-async-book": [
        "01_getting_started/01_chapter.html","01_getting_started/02_why_async.html",
        "01_getting_started/03_state_of_async_rust.html","01_getting_started/04_async_await_primer.html",
        "02_execution/01_chapter.html","02_execution/02_future.html","02_execution/03_wakeups.html",
        "02_execution/04_executor.html","02_execution/05_io.html",
        "03_async_await/01_chapter.html",
        "04_pinning/01_chapter.html",
        "05_streams/01_chapter.html","05_streams/02_iteration_and_concurrency.html",
        "06_multiple_futures/01_chapter.html","06_multiple_futures/02_join.html",
        "06_multiple_futures/03_select.html","06_multiple_futures/04_spawning.html",
        "07_workarounds/01_chapter.html","07_workarounds/02_return_type.html",
        "07_workarounds/03_recursion.html","07_workarounds/04_async_in_traits.html",
    ],
    # Cargo Book — guide + reference + all commands
    "rust-cargo": [
        "index.html",
        "getting-started/index.html","getting-started/installation.html","getting-started/first-steps.html",
        "guide/index.html","guide/why-cargo-exists.html","guide/creating-a-new-project.html",
        "guide/working-on-an-existing-project.html","guide/dependencies.html",
        "guide/package-layout.html","guide/cargo-toml-vs-cargo-lock.html","guide/tests.html",
        "guide/continuous-integration.html","guide/cargo-home.html","guide/build-cache.html",
        "reference/index.html","reference/specifying-dependencies.html",
        "reference/overriding-dependencies.html","reference/manifest.html",
        "reference/workspaces.html","reference/features.html","reference/profiles.html",
        "reference/configuration.html","reference/environment-variables.html",
        "reference/build-scripts.html","reference/build-script-examples.html",
        "reference/publishing.html","reference/package-id-spec.html",
        "reference/source-replacement.html","reference/external-tools.html",
        "reference/registries.html","reference/resolver.html","reference/semver.html",
        "reference/future-incompat-report.html","reference/unstable.html",
        "commands/index.html","commands/cargo.html",
        "commands/cargo-bench.html","commands/cargo-build.html","commands/cargo-check.html",
        "commands/cargo-clean.html","commands/cargo-doc.html","commands/cargo-fetch.html",
        "commands/cargo-fix.html","commands/cargo-generate-lockfile.html","commands/cargo-help.html",
        "commands/cargo-init.html","commands/cargo-install.html","commands/cargo-locate-project.html",
        "commands/cargo-login.html","commands/cargo-logout.html","commands/cargo-metadata.html",
        "commands/cargo-new.html","commands/cargo-owner.html","commands/cargo-package.html",
        "commands/cargo-publish.html","commands/cargo-remove.html","commands/cargo-report.html",
        "commands/cargo-run.html","commands/cargo-rustc.html","commands/cargo-rustdoc.html",
        "commands/cargo-search.html","commands/cargo-test.html","commands/cargo-tree.html",
        "commands/cargo-update.html","commands/cargo-vendor.html",
        "commands/cargo-verify-project.html","commands/cargo-version.html","commands/cargo-yank.html",
        "faq.html",
    ],
    # Rust Design Patterns — idioms, patterns, anti-patterns, functional
    "rust-patterns": [
        "intro.html",
        "idioms/index.html","idioms/coerce-with-target.html","idioms/concat-format.html",
        "idioms/ctor.html","idioms/default.html","idioms/deref.html","idioms/dtor-finally.html",
        "idioms/ffi/index.html","idioms/ffi/accepting-strings.html","idioms/ffi/passing-strings.html",
        "idioms/mem-replace.html","idioms/on-stack-dyn-dispatch.html","idioms/option-iter.html",
        "idioms/pass-var-to-closure.html","idioms/priv-extend.html",
        "idioms/return-consumed-arg-on-error.html","idioms/rustdoc-init.html",
        "idioms/temporary-mutability.html","idioms/unsafe-guidelines.html",
        "patterns/index.html",
        "patterns/behavioural/index.html","patterns/behavioural/command.html",
        "patterns/behavioural/interpreter.html","patterns/behavioural/newtype.html",
        "patterns/behavioural/RAII.html","patterns/behavioural/strategy.html",
        "patterns/behavioural/visitor.html","patterns/behavioural/fold.html",
        "patterns/creational/index.html","patterns/creational/builder.html",
        "patterns/structural/index.html","patterns/structural/compose-structs.html",
        "patterns/structural/small-crates.html","patterns/structural/unsafe-mods.html",
        "anti_patterns/index.html","anti_patterns/borrow_clone.html","anti_patterns/deref.html",
        "anti_patterns/deny-warnings.html","anti_patterns/wildcard-imports.html",
        "functional/index.html","functional/generics-type-classes.html",
        "functional/optics.html","functional/paradigms.html",
    ],
}

def _discover_chapters(session: requests.Session, base_url: str) -> list[str]:
    """
    Auto-discover chapter URLs from a book's index page.
    Parses relative href="*.html" links — works for any mdBook-style site.
    Falls back to ["index.html"] on failure or if nothing is found.
    """
    for try_url in [base_url + "index.html", base_url]:
        try:
            r = session.get(try_url, timeout=API_TIMEOUT)
            if r.status_code == 200:
                break
        except Exception:
            pass
    else:
        return ["index.html"]

    hrefs = re.findall(r'href="([^"#?]+\.html)"', r.text)
    seen: set[str] = set()
    chapters: list[str] = []
    for h in hrefs:
        if h.startswith("http") or h.startswith("//") or h.startswith("/"):
            continue
        h = h.lstrip("./")          # strip leading "./"
        if not h or h in seen:
            continue
        seen.add(h)
        chapters.append(h)
    return chapters if chapters else ["index.html"]


def _html_to_text(html: str) -> str:
    """Very light HTML → text: strip tags, decode entities, normalize whitespace."""
    # Remove script and style blocks
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Strip all tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Decode common HTML entities
    entities = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
                 "&#39;": "'", "&apos;": "'", "&nbsp;": " ", "&#x27;": "'"}
    for ent, char in entities.items():
        text = text.replace(ent, char)
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text))
    return text.strip()

def scrape_docs(args: argparse.Namespace) -> None:
    outdir   = args.output_dir
    session  = make_session()
    manifest = Manifest(outdir, "docs")
    stats    = Stats()
    layout   = _make_layout()

    with Live(layout, console=console, refresh_per_second=4, screen=False):

        def _refresh():
            layout["header"].update(_header_panel(stats, "Official Docs"))
            layout["item"].update(_item_panel(stats))

        _refresh()

        for label, base_url, is_single in _DOC_SOURCES:
            if _shutdown.is_set():
                break
            if manifest.is_done(label):
                console.print(f"[skip]Skipping {label} (already done)[/skip]")
                continue

            stats.current_item = label
            out_path = output_path(outdir, "docs", label)
            count = 0

            if is_single:
                # Single-page doc (error index, std index)
                stats.current_file = base_url
                _refresh()
                try:
                    r = session.get(base_url, timeout=API_TIMEOUT)
                    r.raise_for_status()
                    text = _html_to_text(r.text)
                    if len(text) > MIN_FILE_BYTES:
                        record = make_record(
                            source="docs", origin=label,
                            path=base_url, content=text,
                        )
                        write_record(out_path, record)
                        stats.files_written += 1
                        stats.bytes_written += len(text.encode("utf-8"))
                        stats.tokens_est    += record["tokens_est"]
                        count += 1
                except Exception as e:
                    log.warning(f"Docs fetch failed {base_url}: {e}")
                    stats.errors += 1
                time.sleep(CRATES_API_DELAY)

            else:
                # Chapter-by-chapter
                chapters = _BOOK_CHAPTERS.get(label, [])
                if not chapters:
                    console.print(f"[info]  Auto-discovering chapters for {label}...[/info]")
                    chapters = _discover_chapters(session, base_url)
                    console.print(f"[info]  Found {len(chapters)} chapters[/info]")

                for chapter in chapters:
                    if _shutdown.is_set():
                        break
                    url = base_url + chapter
                    stats.current_file = chapter
                    _refresh()
                    try:
                        r = session.get(url, timeout=API_TIMEOUT)
                        r.raise_for_status()
                        text = _html_to_text(r.text)
                        if len(text) > MIN_FILE_BYTES:
                            record = make_record(
                                source="docs", origin=label,
                                path=url, content=text,
                            )
                            write_record(out_path, record)
                            stats.files_written += 1
                            stats.bytes_written += len(text.encode("utf-8"))
                            stats.tokens_est    += record["tokens_est"]
                            count += 1
                    except Exception as e:
                        log.warning(f"Chapter fetch failed {url}: {e}")
                        stats.errors += 1
                    time.sleep(0.5)

            if count == 0:
                stats.files_skipped += 1
            manifest.mark_done(label)
            _refresh()

    _print_final_stats(stats, "Docs", manifest)

# ─────────────────────────────────────────────────────────
#  STATS COMMAND
# ─────────────────────────────────────────────────────────

def stats_cmd(args: argparse.Namespace) -> None:
    outdir = args.output_dir
    base   = Path(outdir)

    if not base.exists():
        console.print(f"[error]Output directory not found: {outdir}[/error]")
        return

    sources = ["crates", "github", "docs"]
    totals  = defaultdict(lambda: {"files": 0, "records": 0, "tokens": 0, "mb": 0.0})

    for source in sources:
        src_dir = base / source
        if not src_dir.exists():
            continue
        for jsonl_file in src_dir.glob("*.jsonl"):
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        totals[source]["records"] += 1
                        totals[source]["tokens"]  += rec.get("tokens_est", 0)
                        totals[source]["mb"]      += len(line.encode()) / 1_048_576
                    except Exception:
                        pass
            totals[source]["files"] += 1

    tbl = Table(
        title=f"Rust Data — {outdir}",
        box=box.ROUNDED, show_footer=True, header_style="bold cyan",
    )
    total_rec = sum(v["records"] for v in totals.values())
    total_tok = sum(v["tokens"]  for v in totals.values())
    total_mb  = sum(v["mb"]      for v in totals.values())

    tbl.add_column("Source",  footer="TOTAL",             style="source", width=12)
    tbl.add_column("Files",   footer="",                  justify="right")
    tbl.add_column("Records", footer=f"{total_rec:,}",    justify="right", style="green")
    tbl.add_column("~Tokens", footer=f"~{total_tok:,}",   justify="right", style="count")
    tbl.add_column("MB",      footer=f"{total_mb:.1f}",   justify="right", style="dim")

    for source, v in sorted(totals.items()):
        tbl.add_row(
            source,
            str(v["files"]),
            f"{v['records']:,}",
            f"~{v['tokens']:,}",
            f"{v['mb']:.1f}",
        )

    console.print(tbl)

    # Manifest counts
    console.print()
    for source in ["crates", "github", "docs"]:
        m = Manifest(outdir, source)
        console.print(f"  [source]{source:<8}[/source] manifest: {m.count():,} items completed")

def _print_final_stats(stats: Stats, source_name: str, manifest: Manifest) -> None:
    console.print(
        f"\n[success]✓ {source_name} done.  "
        f"{stats.files_written:,} files  |  "
        f"{stats.mb_written:.1f} MB  |  "
        f"~{stats.tokens_est:,} tokens[/success]"
    )
    console.print(
        f"  Skipped: [skip]{stats.files_skipped:,}[/skip]  "
        f"Errors: [error]{stats.errors}[/error]  "
        f"Manifest: {manifest.count():,} items"
    )

# ─────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        prog="rustscrape",
        description="Rust training data scraper — crates.io · GitHub · Docs",
    )
    sub = p.add_subparsers(dest="command")

    # crates
    c = sub.add_parser("crates", help="Scrape top crates from crates.io")
    c.add_argument("--top",        type=int, default=500,       help="Number of top crates (default: 500)")
    c.add_argument("--output-dir", type=str, default="./rust_data")

    # github
    g = sub.add_parser("github", help="Scrape rust-lang org and ecosystem repos")
    g.add_argument("--token",      type=str, default=None,
                   help="GitHub personal access token (or set GITHUB_TOKEN env var)")
    g.add_argument("--output-dir", type=str, default="./rust_data")

    # docs
    d = sub.add_parser("docs", help="Scrape official Rust documentation")
    d.add_argument("--output-dir", type=str, default="./rust_data")

    # all
    a = sub.add_parser("all", help="Run all scrapers in sequence")
    a.add_argument("--token",      type=str, default=None)
    a.add_argument("--top",        type=int, default=500)
    a.add_argument("--output-dir", type=str, default="./rust_data")

    # stats
    s = sub.add_parser("stats", help="Print dataset statistics")
    s.add_argument("--output-dir", type=str, default="./rust_data")

    args = p.parse_args()

    if args.command == "crates":
        scrape_crates(args)
    elif args.command == "github":
        scrape_github(args)
    elif args.command == "docs":
        scrape_docs(args)
    elif args.command == "all":
        scrape_crates(args)
        if not _shutdown.is_set():
            scrape_github(args)
        if not _shutdown.is_set():
            scrape_docs(args)
    elif args.command == "stats":
        stats_cmd(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()