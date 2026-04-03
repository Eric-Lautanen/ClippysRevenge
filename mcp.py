# ── MUST be first: stops mcp.py from shadowing the installed `mcp` package.
# ── Python inserts the script's own directory into sys.path[0] at launch;
# ── removing it lets `from mcp.server.fastmcp import FastMCP` find the real package.
import sys as _sys, os as _os
_sys.path = [p for p in _sys.path
             if _os.path.abspath(p) != _os.path.dirname(_os.path.abspath(__file__))]

"""
mcp.py — Hardened MCP web search server for LM Studio headless
Optimized for finding latest Rust crate versions, APIs, and docs.

Requirements:
    pip install mcp ddgs requests beautifulsoup4

mcp.json entry:
    {
      "mcpServers": {
        "rust-search": {
          "command": "python",
          "args": ["C:\\LLM\\mcp.py"]
        }
      }
    }
"""

# typing imports handle py3.9+ compat (List, Optional, Set, Dict, Callable)

# ── stdlib ────────────────────────────────────────────────────────────────────
import functools
import json
import logging
import re
import time
import urllib.parse
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
    TimeoutError as FuturesTimeoutError,  # aliased — does NOT shadow builtin
)
from typing import Callable, Dict, List, Optional, Set

# ── third-party ───────────────────────────────────────────────────────────────
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

try:
    from ddgs import DDGS           # pip install ddgs  (new name)
except ImportError:
    from duckduckgo_search import DDGS  # legacy fallback

from mcp.server.fastmcp import FastMCP

# ── logging — goes to stderr, visible in LM Studio's plugin log ───────────────
logging.basicConfig(
    stream=_sys.stderr,
    level=logging.INFO,
    format="[mcp-search] %(levelname)s %(message)s",
)
log = logging.getLogger("mcp-search")

# Silence the internal HTTP request logs produced by DDGS and its dependencies
# (httpx, httpcore, urllib3).  These log every outbound request to search-engine
# backends (Google, Yahoo, Brave, DuckDuckGo, Wikipedia autocomplete, etc.),
# which is expected DDGS behaviour but creates confusing noise in the plugin log.
# We keep WARNING+ so genuine errors (timeouts, SSL failures) still surface.
for _noisy_logger in (
    "ddgs", "duckduckgo_search",
    "httpx", "httpcore",
    "urllib3", "urllib3.connectionpool",
    "requests.packages.urllib3",
    "charset_normalizer",
):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# Constants / limits
# ─────────────────────────────────────────────────────────────────────────────

MAX_QUERY_LEN    = 300        # chars — truncate absurdly long queries
MAX_QUERIES      = 6          # max parallel queries in multi_search
MAX_URLS         = 6          # max parallel URLs in fetch_pages
MAX_RESULTS      = 15         # hard cap on DDG results per query
MAX_HTML_BYTES   = 2_000_000  # 2 MB — don't feed BS4 bigger pages than this
MAX_CHARS_PAGE   = 8_000      # max content chars returned from a single page
MAX_CHARS_TOTAL  = 24_000     # hard cap on any tool's total output
HTTP_TIMEOUT     = 12         # seconds for individual HTTP calls
FUTURES_TIMEOUT  = 20         # seconds for thread-pool completion
DDG_RETRIES      = 3          # attempts before giving up on a DDG query
DDG_BACKOFF      = [1, 2, 4]  # sleep seconds between retries

# ─────────────────────────────────────────────────────────────────────────────
# Simple in-process TTL cache — prevents duplicate network hits when the model
# calls web_search multiple times with identical or near-identical queries.
# ─────────────────────────────────────────────────────────────────────────────

_CACHE_TTL_SECS = 300   # 5 minutes — covers any single fix-cascade session
_search_cache: Dict[str, tuple] = {}  # key -> (expires_at, result_str)


def _cache_get(key: str) -> Optional[str]:
    entry = _search_cache.get(key)
    if entry and time.time() < entry[0]:
        log.info("Cache hit for query: %s", key[:80])
        return entry[1]
    return None


def _cache_set(key: str, value: str) -> None:
    _search_cache[key] = (time.time() + _CACHE_TTL_SECS, value)


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

SKIP_TAGS = frozenset({
    "script", "style", "nav", "footer", "header", "aside",
    "noscript", "iframe", "svg", "form", "menu", "meta", "link",
})

# ─────────────────────────────────────────────────────────────────────────────
# Domain whitelist — controls which domains our SESSION may connect to AND
# which DDG result URLs are surfaced to the model.
#
# Matching is suffix-based: "rust-lang.org" covers doc.rust-lang.org,
# blog.rust-lang.org, internals.rust-lang.org, users.rust-lang.org, …
#
# DDGS uses its own internal HTTP client (httpx/requests) to query search
# engines (Google, Yahoo, Brave, DuckDuckGo, Mojeek) — that traffic is
# external to this SESSION and cannot be blocked without breaking the library.
# Those internal DDGS requests are silenced in the log below; only OUR fetch
# calls (fetch_page / get_crates_package / get_github_release) are enforced.
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_DOMAINS: frozenset = frozenset({
    # ── Official Rust infrastructure ──────────────────────────────────────────
    "rust-lang.org",            # doc.*, blog.*, internals.*, users.*, play.*, …
    "rustup.rs",
    "crates.io",
    "docs.rs",                  # official hosted crate documentation
    "lib.rs",                   # alternative crate search / stats

    # ── Source hosting ────────────────────────────────────────────────────────
    "github.com",
    "githubusercontent.com",    # raw.githubusercontent.com
    "github.io",                # *.github.io — rust-lang.github.io, matklad.*, …
    "gitlab.com",
    "gitlab.io",                # *.gitlab.io — GitLab Pages

    # ── Community Q&A ─────────────────────────────────────────────────────────
    "stackoverflow.com",
    "stackexchange.com",        # softwareengineering.stackexchange.com, etc.

    # ── Official crate / framework sites ──────────────────────────────────────
    "tokio.rs",
    "rocket.rs",
    "actix.rs",
    "diesel.rs",
    "serde.rs",
    "hyper.rs",
    "leptos.dev",
    "dioxuslabs.com",
    "bevy.rs",
    "linebender.org",
    "tauri.app",

    # ── Hosted documentation platforms ────────────────────────────────────────
    "readthedocs.io",           # *.readthedocs.io
    "readthedocs.org",

    # ── Rust tooling ──────────────────────────────────────────────────────────
    "cheats.rs",
    "deps.rs",
    "rustfmt.rs",
    "clippy.rs",

    # ── High-quality Rust-focused blogs & guides ──────────────────────────────
    "fasterthanli.me",
    "without.boats",
    "smallcultfollowing.com",   # Niko Matsakis
    "corrode.dev",
    "pretzelhammer.com",
    "this-week-in-rust.org",
    "llogiq.github.io",         # covered by github.io
    "manishearth.github.io",    # covered by github.io
    "burntsushi.net",           # Andrew Gallant (ripgrep, regex, csv)
    "fitzgeraldnick.com",
    "blog.adamchalmers.com",
    "hegdenu.net",
    "lobste.rs",

    # ── Learning & exercises ──────────────────────────────────────────────────
    "exercism.org",
    "tourofrust.com",
    "rustlings.cool",
    "rust-exercises.com",

    # ── Official Rust books (separately hosted) ───────────────────────────────
    "rust-book.cs.brown.edu",
    "rustwasm.github.io",       # covered by github.io
    "rust-embedded.github.io",  # covered by github.io
    "rust-unofficial.github.io",# covered by github.io
})


def _is_allowed_domain(url: str) -> bool:
    """
    Return True if the URL's hostname equals or is a subdomain of any entry
    in ALLOWED_DOMAINS.  Strips leading "www." and port numbers first.

        "https://doc.rust-lang.org/std/"  -> True  (suffix of rust-lang.org)
        "https://tokio.rs/docs"           -> True  (exact match)
        "https://wikipedia.org/wiki/Rust" -> False
        "https://medium.com/..."          -> False
    """
    try:
        host = urllib.parse.urlparse(url).netloc.lower().split(":")[0]
        if host.startswith("www."):
            host = host[4:]
        if not host:
            return False
        for domain in ALLOWED_DOMAINS:
            if host == domain or host.endswith("." + domain):
                return True
        return False
    except Exception:
        return False


def _filter_allowed(results: List[dict]) -> List[dict]:
    """
    Remove DDG result dicts whose href is not in ALLOWED_DOMAINS.
    Rejected domains are logged so they are visible in the MCP server log.
    """
    allowed, rejected = [], []
    for r in results:
        href = r.get("href", "")
        if _is_allowed_domain(href):
            allowed.append(r)
        else:
            rejected.append(href)
    if rejected:
        log.info(
            "Domain filter: dropped %d result(s) not in whitelist: %s",
            len(rejected),
            ", ".join(urllib.parse.urlparse(u).netloc for u in rejected if u)[:200],
        )
    return allowed

# ─────────────────────────────────────────────────────────────────────────────
# Shared HTTP session (connection pooling + automatic retry on 5xx/network err)
# ─────────────────────────────────────────────────────────────────────────────

class _BlockingAdapter(HTTPAdapter):
    """
    HTTPAdapter subclass that enforces ALLOWED_DOMAINS at the socket level.
    The check fires in send() — before any TCP connection is opened — so a
    non-whitelisted hostname is never contacted, not even for a DNS lookup.

    This covers all requests made through our shared SESSION (fetch_page,
    get_crates_package, get_github_release).  DDGS uses its own internal
    HTTP client and is not subject to this adapter; its results are filtered
    separately by _filter_allowed().
    """
    def send(self, request, **kwargs):
        if not _is_allowed_domain(request.url):
            host = urllib.parse.urlparse(request.url).netloc
            log.warning("SESSION blocked outbound request to non-whitelisted domain: %s", host)
            raise requests.exceptions.ConnectionError(
                f"Blocked by domain whitelist: '{host}' is not a trusted Rust source. "
                f"Add it to ALLOWED_DOMAINS in mcp.py if it should be permitted."
            )
        return super().send(request, **kwargs)


def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist={500, 502, 503, 504},
        allowed_methods={"GET"},
        raise_on_status=False,
    )
    adapter = _BlockingAdapter(max_retries=retry, pool_connections=8, pool_maxsize=16)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(HEADERS)
    return session

SESSION = _make_session()

# ─────────────────────────────────────────────────────────────────────────────
# Safety decorator — every tool is wrapped so exceptions NEVER reach FastMCP
# ─────────────────────────────────────────────────────────────────────────────

def _safe_tool(fn: Callable) -> Callable:
    """
    Ensures every tool always returns a plain string.
    Catches ALL exceptions and returns them as readable error messages
    instead of crashing the MCP server process.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> str:
        try:
            result = fn(*args, **kwargs)
            if not isinstance(result, str):
                result = str(result)
            if not result.strip():
                return "No results returned."
            # Hard cap on total output size
            if len(result) > MAX_CHARS_TOTAL:
                result = result[:MAX_CHARS_TOTAL] + "\n\n[... output truncated to protect context window]"
            return result
        except Exception as exc:
            log.exception("Tool '%s' raised an unhandled exception", fn.__name__)
            return f"[Tool error — {fn.__name__}] {type(exc).__name__}: {exc}"
    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize_query(q: str) -> str:
    """Strip control chars and enforce length limit."""
    q = re.sub(r"[\x00-\x1f\x7f]", " ", q).strip()
    return q[:MAX_QUERY_LEN] if q else ""


def _validate_url(url: str) -> Optional[str]:
    """Return None if URL looks safe, or an error string if not."""
    url = url.strip()
    if not url:
        return "URL is empty."
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return f"Unsupported URL scheme '{parsed.scheme}'. Only http/https allowed."
        if not parsed.netloc:
            return "URL has no hostname."
    except Exception as e:
        return f"Invalid URL: {e}"
    return None


def _clean_html(html: str, max_chars: int = MAX_CHARS_PAGE) -> str:
    """
    Parse HTML into readable text, preserving code blocks.
    Enforces MAX_HTML_BYTES before even touching BeautifulSoup.
    """
    # Trim before parsing — avoids feeding BS4 multi-MB pages
    if len(html) > MAX_HTML_BYTES:
        html = html[:MAX_HTML_BYTES]
        log.info("HTML truncated to %d bytes before parsing", MAX_HTML_BYTES)

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as e:
        log.warning("BeautifulSoup parse error: %s", e)
        # Fallback: strip all tags with regex
        return re.sub(r"<[^>]+>", " ", html)[:max_chars]

    for tag in soup(list(SKIP_TAGS)):
        tag.decompose()

    # Mark code blocks before text extraction
    for code in soup.find_all(["pre", "code"]):
        try:
            classes = code.get("class") or []
            lang = next(
                (c.replace("language-", "") for c in classes if c.startswith("language-")),
                ""
            )
            code.replace_with(f"\n```{lang}\n{code.get_text()}\n```\n")
        except Exception:
            pass  # if this fails just leave the tag alone

    for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "li", "tr"]):
        try:
            tag.insert_before("\n")
        except Exception:
            pass

    text = soup.get_text(separator=" ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:max_chars]


def _ddg_search(query: str, max_results: int = 8) -> List[dict]:
    """
    DuckDuckGo search with capped inputs, exponential backoff, and per-attempt
    timeouts. Always returns a list — never raises.
    """
    query = _sanitize_query(query)
    if not query:
        return [{"title": "Empty query", "href": "", "body": "Query was empty or invalid."}]

    max_results = min(max(1, max_results), MAX_RESULTS)

    last_err: str = ""
    for attempt, sleep_secs in enumerate(DDG_BACKOFF[:DDG_RETRIES]):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            results = _filter_allowed(results)
            if results:
                return results
            # Empty result (after domain filtering) is not an error
            return []
        except Exception as e:
            last_err = str(e)
            log.warning("DDG attempt %d/%d failed for '%s': %s",
                        attempt + 1, DDG_RETRIES, query[:60], e)
            if attempt < DDG_RETRIES - 1:
                time.sleep(sleep_secs)

    return [{"title": "Search failed", "href": "",
             "body": f"DuckDuckGo search failed after {DDG_RETRIES} attempts: {last_err}"}]


def _fmt_results(results: list[dict]) -> str:
    parts = []
    for r in results:
        title = r.get("title") or "No title"
        href  = r.get("href") or ""
        body  = (r.get("body") or "").strip()
        parts.append(f"**{title}**\nURL: {href}\n{body}")
    return "\n\n---\n\n".join(parts)


def _run_parallel(
    tasks: dict[str, Callable[[], str]],
    timeout: float = FUTURES_TIMEOUT,
    max_workers: int = 5,
) -> dict[str, str]:
    """
    Run a dict of {label: callable} in parallel.
    Returns {label: result_or_error_string}.
    Never raises — timeout and exceptions become error strings.
    """
    results: dict[str, str] = {}
    if not tasks:
        return results

    workers = min(len(tasks), max_workers)
    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mcp-search") as pool:
            future_map: dict = {pool.submit(fn): label for label, fn in tasks.items()}
            try:
                for future in as_completed(future_map, timeout=timeout):
                    label = future_map[future]
                    try:
                        results[label] = future.result()
                    except Exception as e:
                        log.warning("Task '%s' raised: %s", label, e)
                        results[label] = f"Error: {type(e).__name__}: {e}"
            except FuturesTimeoutError:
                # Collect whatever finished; mark the rest as timed out
                for future, label in future_map.items():
                    if label not in results:
                        results[label] = f"Timed out after {timeout}s."
                log.warning("Parallel tasks timed out; partial results returned.")
    except Exception as e:
        log.error("Thread pool creation failed: %s", e)
        for label in tasks:
            if label not in results:
                results[label] = f"Thread pool error: {e}"

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MCP server
# ─────────────────────────────────────────────────────────────────────────────

mcp = FastMCP("Rust Search — Dev Edition")


# ── Tool 1 ────────────────────────────────────────────────────────────────────

@mcp.tool()
@_safe_tool
def web_search(query: str, max_results: int = 8) -> str:
    """
    Search the web with DuckDuckGo. Returns titles, URLs, and snippets.
    For researching multiple angles simultaneously, prefer multi_search.
    Repeated identical queries within 5 minutes are served from cache.

    Args:
        query: Search query string.
        max_results: Number of results to return (1-15, default 8).
    """
    cache_key = f"ws:{_sanitize_query(query)}:{max_results}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    results = _ddg_search(query, max_results)
    out = _fmt_results(results) if results else f"No results found for: {query}"
    _cache_set(cache_key, out)
    return out


# ── Tool 2 ────────────────────────────────────────────────────────────────────

@mcp.tool()
@_safe_tool
def multi_search(queries: List[str], max_results_each: int = 5) -> str:
    """
    Run up to 6 search queries IN PARALLEL and merge unique results.
    Deduplicates by URL. Use when you need multiple search angles at once.

    Example queries list:
        ["tokio 1.0 changelog", "tokio 1.0 breaking changes", "tokio migrate 0.x 1.0"]

    Args:
        queries: List of search queries (max 6).
        max_results_each: Results per query (1-10, default 5).
    """
    if not queries:
        return "No queries provided."

    # Sanitize and cap inputs
    queries = [_sanitize_query(q) for q in queries[:MAX_QUERIES]]
    queries = [q for q in queries if q]  # drop empty after sanitization
    if not queries:
        return "All queries were empty after sanitization."

    max_results_each = min(max(1, max_results_each), 10)

    cache_key = f"ms:{','.join(sorted(queries))}:{max_results_each}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    seen_urls: set[str] = set()
    tasks = {q: (lambda q=q: _ddg_search(q, max_results_each)) for q in queries}
    raw = _run_parallel(tasks)

    sections: List[str] = []
    for query_label, result in raw.items():
        if isinstance(result, str):
            # Error string from _run_parallel
            sections.append(f"### Query: {query_label}\n\n{result}")
            continue
        unique = [r for r in result if r.get("href") and r["href"] not in seen_urls]
        seen_urls.update(r["href"] for r in unique)
        if unique:
            sections.append(f"### Query: {query_label}\n\n" + _fmt_results(unique))

    out = "\n\n═══════════════════\n\n".join(sections) if sections else "No results found."
    _cache_set(cache_key, out)
    return out


# ── Tool 3 ────────────────────────────────────────────────────────────────────

@mcp.tool()
@_safe_tool
def fetch_page(url: str, max_chars: int = 6000) -> str:
    """
    Fetch a URL and return clean readable text. Strips ads/nav/scripts.
    Preserves code blocks. Use after web_search to read a specific result.

    Args:
        url: Full URL to fetch (http/https only).
        max_chars: Max characters returned (default 6000, max 8000).
    """
    err = _validate_url(url)
    if err:
        return f"Invalid URL: {err}"

    if not _is_allowed_domain(url):
        host = urllib.parse.urlparse(url).netloc
        log.warning("fetch_page blocked non-whitelisted domain: %s", host)
        return (
            f"Blocked: '{host}' is not in the trusted domain whitelist. "
            f"Only Rust-focused sources (docs.rs, rust-lang.org, github.com, "
            f"stackoverflow.com, crates.io, etc.) may be fetched."
        )

    max_chars = min(max(100, max_chars), MAX_CHARS_PAGE)

    try:
        resp = SESSION.get(url, timeout=HTTP_TIMEOUT, stream=True)
        resp.raise_for_status()

        ct = resp.headers.get("content-type", "").lower()

        # Refuse to download binary files
        if not any(t in ct for t in ("text", "json", "xml")):
            return f"Skipped — unsupported content type: {ct or 'unknown'}"

        # Stream read with size cap to avoid pulling huge responses into RAM
        chunks: list[bytes] = []
        size = 0
        for chunk in resp.iter_content(chunk_size=65536):
            chunks.append(chunk)
            size += len(chunk)
            if size >= MAX_HTML_BYTES:
                log.info("fetch_page: response truncated at %d bytes for %s", size, url[:80])
                break
        resp.close()

        raw = b"".join(chunks).decode("utf-8", errors="replace")

        if "json" in ct:
            try:
                return json.dumps(json.loads(raw), indent=2)[:max_chars]
            except json.JSONDecodeError:
                return raw[:max_chars]

        return _clean_html(raw, max_chars)

    except requests.exceptions.Timeout:
        return f"Request timed out after {HTTP_TIMEOUT}s: {url}"
    except requests.exceptions.ConnectionError as e:
        return f"Connection error for {url}: {e}"
    except requests.exceptions.HTTPError as e:
        return f"HTTP {e.response.status_code} error for {url}: {e}"
    except Exception as e:
        return f"Unexpected error fetching {url}: {type(e).__name__}: {e}"


# ── Tool 4 ────────────────────────────────────────────────────────────────────

@mcp.tool()
@_safe_tool
def fetch_pages(urls: List[str], max_chars_each: int = 3000) -> str:
    """
    Fetch multiple URLs in parallel and return their cleaned text.
    Use after multi_search to read several results at once.

    Args:
        urls: List of URLs to fetch (max 6).
        max_chars_each: Max chars per page (default 3000).
    """
    if not urls:
        return "No URLs provided."

    urls = [u.strip() for u in urls[:MAX_URLS]]
    max_chars_each = min(max(100, max_chars_each), MAX_CHARS_PAGE)

    # Validate all URLs upfront
    tasks: dict[str, Callable] = {}
    invalid: List[str] = []
    for url in urls:
        err = _validate_url(url)
        if err:
            invalid.append(f"Skipped {url}: {err}")
        else:
            tasks[url] = (lambda u=url: fetch_page(u, max_chars_each))

    raw = _run_parallel(tasks)

    sections: List[str] = invalid[:]
    for url, content in raw.items():
        sections.append(f"### {url}\n\n{content}")

    return "\n\n═══════════════════\n\n".join(sections) if sections else "No content retrieved."


# ── Tool 5 ────────────────────────────────────────────────────────────────────

@mcp.tool()
@_safe_tool
def get_crates_package(crate_name: str) -> str:
    """
    Get latest version and recent history for a Rust crate from crates.io.
    Always current.

    Args:
        crate_name: Exact crate name e.g. "tokio", "serde", "axum".
    """
    crate_name = crate_name.strip()
    if not crate_name:
        return "Crate name is required."

    try:
        resp = SESSION.get(
            f"https://crates.io/api/v1/crates/{crate_name}",
            headers={"User-Agent": "mcp-search-server/2.0 (github contact preferred)"},
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code == 404:
            return f"Crate '{crate_name}' not found on crates.io."
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        return f"crates.io request timed out for '{crate_name}'."
    except Exception as e:
        return f"crates.io API error for '{crate_name}': {type(e).__name__}: {e}"

    krate    = data.get("crate") or {}
    versions = [v for v in (data.get("versions") or []) if not v.get("yanked")][:10]
    recent   = [f"{v['num']} ({(v.get('created_at') or '')[:10]})" for v in versions]
    license_ = versions[0].get("license", "N/A") if versions else "N/A"

    return "\n".join(filter(None, [
        f"**{krate.get('name', crate_name)}** v{krate.get('newest_version', '?')}",
        f"Description:   {krate.get('description') or 'N/A'}",
        f"License:       {license_}",
        f"Downloads:     {krate.get('downloads', 0):,} total",
        f"Homepage:      {krate.get('homepage') or 'N/A'}",
        f"Repository:    {krate.get('repository') or 'N/A'}",
        f"Documentation: {krate.get('documentation') or 'N/A'}",
        "",
        f"Recent non-yanked versions: {', '.join(recent[:8])}",
        f"crates.io URL: https://crates.io/crates/{crate_name}",
    ]))


# ── Tool 8 ────────────────────────────────────────────────────────────────────

@mcp.tool()
@_safe_tool
def get_github_release(owner_repo: str) -> str:
    """
    Get the latest release info from a public GitHub repository.
    Falls back to listing tags if the repo doesn't use formal releases.
    No auth token needed for public repos (60 req/hr unauthenticated).

    Args:
        owner_repo: "owner/repo" e.g. "microsoft/typescript", "tokio-rs/tokio".
    """
    owner_repo = owner_repo.strip().strip("/")
    if not owner_repo or owner_repo.count("/") != 1:
        return "owner_repo must be in 'owner/repo' format e.g. 'microsoft/typescript'."

    gh_headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    base = f"https://api.github.com/repos/{owner_repo}"

    # Fetch latest release and recent releases list in parallel
    def _get_latest():
        r = SESSION.get(f"{base}/releases/latest", headers=gh_headers, timeout=HTTP_TIMEOUT)
        return r

    def _get_all():
        r = SESSION.get(f"{base}/releases?per_page=10", headers=gh_headers, timeout=HTTP_TIMEOUT)
        return r

    def _get_tags():
        r = SESSION.get(f"{base}/tags?per_page=10", headers=gh_headers, timeout=HTTP_TIMEOUT)
        return r

    tasks = {"latest": _get_latest, "all": _get_all}
    raw = _run_parallel(tasks, timeout=15)

    latest_resp = raw.get("latest")
    all_resp    = raw.get("all")

    # Handle timeout/error cases
    if isinstance(latest_resp, str):
        return f"GitHub API error: {latest_resp}"

    try:
        if latest_resp.status_code == 404:
            # Repo has no releases — try tags
            tags_raw = _run_parallel({"tags": _get_tags}, timeout=10)
            tags_resp = tags_raw.get("tags")
            if isinstance(tags_resp, str) or not tags_resp.ok:
                return f"No releases or tags found for '{owner_repo}'."
            tags = tags_resp.json()
            if not tags:
                return f"Repository '{owner_repo}' has no releases or tags."
            names = [t["name"] for t in tags[:10]]
            return (
                f"**{owner_repo}** — no formal releases\n"
                f"Latest tags: {', '.join(names)}\n"
                f"URL: https://github.com/{owner_repo}/tags"
            )

        if latest_resp.status_code == 403:
            return "GitHub API rate limit reached (60 req/hr for unauthenticated). Try again later."

        latest_resp.raise_for_status()
        rel      = latest_resp.json()
        pub_date = (rel.get("published_at") or "")[:10]
        body     = (rel.get("body") or "No release notes provided.").strip()
        if len(body) > 1500:
            body = body[:1500] + "\n\n[... truncated — use fetch_page for full notes]"

        recent: List[str] = []
        if not isinstance(all_resp, str) and all_resp and all_resp.ok:
            recent = [
                f"{r['tag_name']} ({(r.get('published_at') or '')[:10]})"
                for r in all_resp.json()
            ]

        return "\n".join(filter(None, [
            f"**{owner_repo}**",
            f"Latest release: {rel.get('tag_name', '?')} — published {pub_date}",
            f"Pre-release:    {rel.get('prerelease', False)}",
            f"URL:            {rel.get('html_url', '')}",
            "",
            f"Release notes:\n{body}",
            "",
            f"Recent releases: {', '.join(recent)}" if recent else "",
        ]))

    except Exception as e:
        return f"GitHub error for '{owner_repo}': {type(e).__name__}: {e}"


# ── Tool 9 ────────────────────────────────────────────────────────────────────

@mcp.tool()
@_safe_tool
def find_latest_version(name: str, ecosystem: Optional[str] = None) -> str:
    """
    Find the latest version of a Rust crate by checking crates.io and/or GitHub.
    Filters out irrelevant or error results automatically.

    Args:
        name: Crate name, or "owner/repo" for GitHub.
        ecosystem: Optional hint — "rust", "github".
                   If omitted, both crates.io and GitHub are checked in parallel.

    Examples:
        find_latest_version("tokio", "rust")
        find_latest_version("serde")
        find_latest_version("rust-lang/rust", "github")
    """
    name = name.strip()
    if not name:
        return "Crate name is required."

    eco = (ecosystem or "").lower().strip()

    tasks: dict[str, Callable] = {}

    if eco == "rust":
        tasks["crates.io"] = lambda: get_crates_package(name)
    elif eco == "github" or "/" in name:
        tasks["GitHub"] = lambda: get_github_release(name)
    else:
        tasks["crates.io"] = lambda: get_crates_package(name)
        tasks["Web"]       = lambda: _fmt_results(
            _ddg_search(f"{name} rust crate latest version release site:github.com OR site:crates.io", 3)
        )

    raw = _run_parallel(tasks, timeout=15)

    NOISE = (
        "not found on crates",
        "security holding package", "security placeholder",
    )

    sections: List[str] = []
    for label, content in raw.items():
        low = content.lower()
        if any(n in low for n in NOISE):
            continue
        sections.append(f"## {label}\n\n{content}")

    return "\n\n═══════════════════\n\n".join(sections) if sections else \
           f"'{name}' not found on crates.io or GitHub."


# ── Tool 10 ───────────────────────────────────────────────────────────────────

@mcp.tool()
@_safe_tool
def find_changelog(
    library: str,
    from_version: Optional[str] = None,
    to_version: Optional[str] = None,
) -> str:
    """
    Find a Rust crate's changelog, migration guide, or breaking changes.
    Runs multiple targeted searches in parallel.

    Args:
        library: Crate name e.g. "tokio", "serde", "axum".
        from_version: Starting version e.g. "0.1", "1.0".
        to_version: Target version e.g. "1.0", "2.0".

    Examples:
        find_changelog("tokio", "0.x", "1.0")
        find_changelog("serde", "1.0", "2.0")
        find_changelog("axum")
    """
    library = library.strip()
    if not library:
        return "Library name is required."

    ver_ctx = ""
    if from_version and to_version:
        ver_ctx = f" {from_version.strip()} to {to_version.strip()}"
    elif to_version:
        ver_ctx = f" {to_version.strip()}"

    queries = [
        f"{library}{ver_ctx} changelog",
        f"{library}{ver_ctx} breaking changes",
        f"{library}{ver_ctx} migration guide",
        f"{library}{ver_ctx} release notes what's new",
    ]
    return multi_search(queries, max_results_each=4)


# ── Tool 11 ───────────────────────────────────────────────────────────────────

@mcp.tool()
@_safe_tool
def search_api_docs(
    library: str,
    topic: str,
    version: Optional[str] = None,
) -> str:
    """
    Search for API documentation on a specific topic within a Rust crate.
    Runs multiple targeted searches in parallel for best coverage.

    Args:
        library: Crate name e.g. "tokio", "serde", "axum".
        topic: Specific API topic e.g. "async runtime", "derive macros".
        version: Optional version to scope results e.g. "1.0", "0.9".

    Examples:
        search_api_docs("tokio", "select macro")
        search_api_docs("serde", "derive macros", "1.0")
        search_api_docs("axum", "middleware")
    """
    library = library.strip()
    topic   = topic.strip()
    if not library or not topic:
        return "Both 'library' and 'topic' are required."

    ver = f" {version.strip()}" if version else ""
    queries = [
        f"{library}{ver} {topic} documentation",
        f"{library}{ver} {topic} example how to",
        f"{library}{ver} {topic} API reference",
        f"{library} {topic} official docs",
    ]
    return multi_search(queries, max_results_each=5)


# ── Tool 12 — health check ────────────────────────────────────────────────────

@mcp.tool()
@_safe_tool
def ping() -> str:
    """
    Health check. Confirms the MCP server is running and shows loaded tool names.
    Use this first to verify the server started correctly.
    """
    tool_names = [t.name for t in mcp._tool_manager.list_tools()]
    return (
        f"MCP search server is running.\n"
        f"Python {_sys.version.split()[0]} | "
        f"requests {requests.__version__}\n"
        f"Tools ({len(tool_names)}): {', '.join(tool_names)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Starting MCP web search server (stdio transport)...")
    try:
        mcp.run()  # stdio — LM Studio communicates over stdin/stdout
    except KeyboardInterrupt:
        log.info("Server stopped by user.")
    except Exception as e:
        log.critical("Server crashed: %s", e, exc_info=True)
        _sys.exit(1)