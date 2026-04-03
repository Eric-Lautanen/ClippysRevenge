# ── MUST be first: stops mcp.py from shadowing the installed `mcp` package.
# ── Python inserts the script's own directory into sys.path[0] at launch;
# ── removing it lets `from mcp.server.fastmcp import FastMCP` find the real package.
import sys as _sys, os as _os
_sys.path = [p for p in _sys.path
             if _os.path.abspath(p) != _os.path.dirname(_os.path.abspath(__file__))]

"""
mcp.py — Hardened MCP web search server for LM Studio headless
Optimized for finding latest library versions, APIs, and docs.

Requirements:
    pip install mcp ddgs requests beautifulsoup4

mcp.json entry:
    {
      "mcpServers": {
        "web-search": {
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
# Shared HTTP session (connection pooling + automatic retry on 5xx/network err)
# ─────────────────────────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist={500, 502, 503, 504},
        allowed_methods={"GET"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=16)
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
            if results:
                return results
            # Empty result is not an error — just return it
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

mcp = FastMCP("Web Search — Dev Edition")


# ── Tool 1 ────────────────────────────────────────────────────────────────────

@mcp.tool()
@_safe_tool
def web_search(query: str, max_results: int = 8) -> str:
    """
    Search the web with DuckDuckGo. Returns titles, URLs, and snippets.
    For researching multiple angles simultaneously, prefer multi_search.

    Args:
        query: Search query string.
        max_results: Number of results to return (1-15, default 8).
    """
    results = _ddg_search(query, max_results)
    if not results:
        return f"No results found for: {query}"
    return _fmt_results(results)


# ── Tool 2 ────────────────────────────────────────────────────────────────────

@mcp.tool()
@_safe_tool
def multi_search(queries: List[str], max_results_each: int = 5) -> str:
    """
    Run up to 6 search queries IN PARALLEL and merge unique results.
    Deduplicates by URL. Use when you need multiple search angles at once.

    Example queries list:
        ["pandas 2.2 changelog", "pandas 2.2 breaking changes", "pandas migrate 1.x 2.x"]

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

    return "\n\n═══════════════════\n\n".join(sections) if sections else "No results found."


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
def get_pypi_package(package_name: str) -> str:
    """
    Get latest version, release date, summary, and recent version history
    from the PyPI JSON API. Always current — no caching.

    Args:
        package_name: Exact PyPI package name e.g. "fastapi", "numpy".
    """
    package_name = package_name.strip().lower()
    if not package_name:
        return "Package name is required."

    try:
        resp = SESSION.get(f"https://pypi.org/pypi/{package_name}/json", timeout=HTTP_TIMEOUT)
        if resp.status_code == 404:
            return f"Package '{package_name}' not found on PyPI."
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        return f"PyPI request timed out for '{package_name}'."
    except Exception as e:
        return f"PyPI API error for '{package_name}': {type(e).__name__}: {e}"

    info = data.get("info", {})
    urls  = data.get("urls", [])
    upload_date = urls[0].get("upload_time", "unknown")[:10] if urls else "unknown"

    # Get recent non-prerelease versions
    all_versions = sorted(data.get("releases", {}).keys(), reverse=True)
    stable = [v for v in all_versions if not re.search(r"[a-zA-Z]", v)][:10]
    recent_display = stable or all_versions[:10]

    return "\n".join(filter(None, [
        f"**{info.get('name', package_name)}** v{info.get('version', '?')}",
        f"Released:        {upload_date}",
        f"Summary:         {info.get('summary') or 'N/A'}",
        f"License:         {info.get('license') or 'N/A'}",
        f"Requires Python: {info.get('requires_python') or 'N/A'}",
        f"Homepage:        {info.get('home_page') or info.get('project_url') or 'N/A'}",
        f"Docs:            {info.get('docs_url') or 'N/A'}",
        "",
        f"Recent stable versions: {', '.join(recent_display)}",
        f"PyPI URL: https://pypi.org/project/{info.get('name', package_name)}/",
    ]))


# ── Tool 6 ────────────────────────────────────────────────────────────────────

@mcp.tool()
@_safe_tool
def get_npm_package(package_name: str) -> str:
    """
    Get latest version, release date, and recent history from the npm registry.
    Always current. Handles scoped packages (@org/pkg).

    Args:
        package_name: npm package name e.g. "react", "@types/node".
    """
    package_name = package_name.strip()
    if not package_name:
        return "Package name is required."

    encoded = urllib.parse.quote(package_name, safe="@/")
    try:
        resp = SESSION.get(f"https://registry.npmjs.org/{encoded}", timeout=HTTP_TIMEOUT)
        if resp.status_code == 404:
            return f"Package '{package_name}' not found on npm."
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        return f"npm registry request timed out for '{package_name}'."
    except Exception as e:
        return f"npm API error for '{package_name}': {type(e).__name__}: {e}"

    latest_tag = (data.get("dist-tags") or {}).get("latest", "")
    if not latest_tag:
        return f"Package '{package_name}' exists on npm but has no 'latest' tag."

    # Security placeholder check
    description = data.get("description", "") or ""
    if "security holding package" in description.lower():
        return (
            f"'{package_name}' is an npm security placeholder — "
            "the real package likely has a different name or ecosystem."
        )

    latest_info = (data.get("versions") or {}).get(latest_tag, {})
    time_data   = data.get("time") or {}
    release_date = (time_data.get(latest_tag) or "unknown")[:10]

    # Recent versions sorted newest-first
    skip = {"created", "modified"}
    recent = sorted(
        [v for v in time_data if v not in skip],
        key=lambda v: time_data.get(v, ""),
        reverse=True,
    )[:10]

    deps      = latest_info.get("dependencies") or {}
    peer_deps = latest_info.get("peerDependencies") or {}
    repo      = data.get("repository") or {}
    repo_url  = repo.get("url", "N/A") if isinstance(repo, dict) else str(repo)

    lines = [
        f"**{data.get('name', package_name)}** v{latest_tag}",
        f"Released:    {release_date}",
        f"Description: {description or 'N/A'}",
        f"License:     {latest_info.get('license') or 'N/A'}",
        f"Homepage:    {data.get('homepage') or 'N/A'}",
        f"Repository:  {repo_url}",
        "",
        f"Recent versions: {', '.join(recent)}",
    ]
    if deps:
        lines.append(f"Dependencies ({len(deps)}): {', '.join(list(deps)[:15])}")
    if peer_deps:
        lines.append(f"Peer deps: {', '.join(list(peer_deps)[:10])}")
    lines.append(f"\nnpm URL: https://www.npmjs.com/package/{package_name}")

    return "\n".join(lines)


# ── Tool 7 ────────────────────────────────────────────────────────────────────

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
        owner_repo: "owner/repo" e.g. "microsoft/typescript", "pydantic/pydantic".
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
    Find the latest version of a package by checking all relevant registries
    simultaneously (PyPI, npm, crates.io, and/or GitHub).
    Filters out irrelevant or error results automatically.

    Args:
        name: Package/library name, or "owner/repo" for GitHub.
        ecosystem: Optional hint — "python", "node", "rust", "github".
                   If omitted, all registries are checked in parallel.

    Examples:
        find_latest_version("pydantic")
        find_latest_version("react", "node")
        find_latest_version("tokio", "rust")
        find_latest_version("microsoft/typescript", "github")
    """
    name = name.strip()
    if not name:
        return "Package name is required."

    eco = (ecosystem or "").lower().strip()

    tasks: dict[str, Callable] = {}

    if eco == "python":
        tasks["PyPI"] = lambda: get_pypi_package(name)
    elif eco == "node":
        tasks["npm"] = lambda: get_npm_package(name)
    elif eco == "rust":
        tasks["crates.io"] = lambda: get_crates_package(name)
    elif eco == "github" or "/" in name:
        tasks["GitHub"] = lambda: get_github_release(name)
    else:
        tasks["PyPI"]     = lambda: get_pypi_package(name)
        tasks["npm"]      = lambda: get_npm_package(name)
        tasks["crates.io"] = lambda: get_crates_package(name)
        tasks["Web"]      = lambda: _fmt_results(
            _ddg_search(f"{name} latest version release site:github.com OR site:pypi.org", 3)
        )

    raw = _run_parallel(tasks, timeout=15)

    NOISE = (
        "not found on pypi", "not found on npm", "not found on crates",
        "security holding package", "security placeholder",
    )

    sections: List[str] = []
    for label, content in raw.items():
        low = content.lower()
        if any(n in low for n in NOISE):
            continue
        sections.append(f"## {label}\n\n{content}")

    return "\n\n═══════════════════\n\n".join(sections) if sections else \
           f"'{name}' not found in any registry."


# ── Tool 10 ───────────────────────────────────────────────────────────────────

@mcp.tool()
@_safe_tool
def find_changelog(
    library: str,
    from_version: Optional[str] = None,
    to_version: Optional[str] = None,
) -> str:
    """
    Find a library's changelog, migration guide, or breaking changes.
    Runs multiple targeted searches in parallel.

    Args:
        library: Library name e.g. "pydantic", "react", "sqlalchemy".
        from_version: Starting version e.g. "v1", "17".
        to_version: Target version e.g. "v2", "18".

    Examples:
        find_changelog("pydantic", "v1", "v2")
        find_changelog("react", "17", "18")
        find_changelog("sqlalchemy")
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
    Search for API documentation on a specific topic within a library.
    Runs multiple targeted searches in parallel for best coverage.

    Args:
        library: Library name e.g. "fastapi", "pandas", "tokio".
        topic: Specific API topic e.g. "dependency injection", "groupby agg".
        version: Optional version to scope results e.g. "2.2", "0.9".

    Examples:
        search_api_docs("fastapi", "dependency injection")
        search_api_docs("pandas", "groupby agg", "2.2")
        search_api_docs("tokio", "select macro")
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