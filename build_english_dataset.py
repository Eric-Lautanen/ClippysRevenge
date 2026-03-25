#!/usr/bin/env python3
"""
=============================================================================
 English Conversation Dataset Builder  — Open Source Community Edition
=============================================================================
 Downloads, normalises, filters, and deduplicates the best open English
 conversation / instruction datasets available as of March 2026.
 Outputs chunked JSONL files to convos/english/ ready for GitHub / training.

 OUTPUT FORMAT (one JSON object per line):
   {
     "conversations": [
       {"role": "user",      "content": "..."},
       {"role": "assistant", "content": "..."},
       ...
     ],
     "source": "<dataset_name>"
   }

 DATASETS INCLUDED:
   ── Classic (proven quality) ──────────────────────────────────────────
   lmsys/lmsys-chat-1m             Real users x 25 LLMs         ~1 M convos
   allenai/WildChat-1M             Real ChatGPT logs             ~1 M convos
   Open-Orca/OpenOrca              GPT-4/3.5 FLAN reasoning      ~4 M rows
   HuggingFaceH4/ultrachat_200k    Clean multi-turn              ~200 k
   OpenAssistant/oasst2            Human-annotated trees         ~600 k turns
   databricks/databricks-dolly-15k Human-written intents         ~15 k
   Anthropic/hh-rlhf               Helpful/harmless pairs        ~170 k

   ── 2024-2025 (newer, SLM-optimised) ──────────────────────────────────
   HuggingFaceTB/smoltalk          SmolLM2 SFT mix               ~1 M
   allenai/tulu-3-sft-mixture      Tulu 3 SFT mix (AllenAI)      ~939 k
   magpie-align/Magpie-Llama-3.1-Pro-1M-v0.1  Magpie pipeline   ~1 M
   HuggingFaceTB/everyday-conversations-llama3.1-2k  Greetings   ~2 k
   microsoft/orca-agentinstruct-1M-v1  AgentInstruct (MS)        ~1 M
   lmsys/chatbot_arena_conversations  Chatbot Arena              ~33 k

 FILTERS APPLIED:
   * English-only  (langdetect, deterministic seed=42)
   * Drop turns shorter than --min_tokens tokens
   * Drop conversations with fewer than --min_turns turn pairs
   * Drop turns that are pure whitespace / URL-only / repeated-char garbage
   * Drop turns with >40% non-ASCII characters (likely garbled encoding)
   * SHA-256 exact deduplication on normalised full conversation text

 RESUME SUPPORT:
   Completed datasets are recorded in <output_dir>/progress.json.
   Re-running the script automatically skips completed datasets.
   Delete progress.json (or use --no_resume) to start fresh.

 INSTALL:
   pip install datasets langdetect tqdm huggingface_hub

 USAGE:
   # Full run (all datasets, default filters):
   python build_english_dataset.py

   # Test run -- 50k rows per dataset:
   python build_english_dataset.py --limit 50000

   # Skip specific datasets:
   python build_english_dataset.py --skip lmsys_chat,wildchat

   # Run only specific datasets:
   python build_english_dataset.py --only dolly,oasst2

   # Tighter quality filters:
   python build_english_dataset.py --min_tokens 15 --min_turns 2

   # Custom chunk size (default 50 MB for GitHub):
   python build_english_dataset.py --chunk_mb 100

   # List available dataset keys:
   python build_english_dataset.py --list
=============================================================================
"""

import os
import re
import sys
import json
import time
import signal
import shutil
import hashlib
import logging
import argparse
import traceback
from pathlib import Path
from typing import Optional

# =============================================================================
# USER CONFIGURATION
# Set your HuggingFace token here to access gated datasets
# (lmsys/lmsys-chat-1m, lmsys/chatbot_arena_conversations, etc.)
# Get your token at: https://huggingface.co/settings/tokens
# You can also pass it via --hf_token on the command line, or set the
# HF_TOKEN environment variable.
# =============================================================================
HF_TOKEN = ""   # <-- paste your HF token here, e.g. "hf_xxxxxxxxxxxxxxxxxxxx"

# =============================================================================
# DEPENDENCY CHECKS  (fail fast with clear messages)
# =============================================================================
try:
    from langdetect import detect, LangDetectException
    from langdetect import DetectorFactory
    # Fix seed IMMEDIATELY — langdetect is non-deterministic without this.
    # Same text can flip between 'en' / 'unknown' between runs without it.
    DetectorFactory.seed = 42
except ImportError:
    sys.exit("ERROR: 'langdetect' not installed.  Run:  pip install langdetect")

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("ERROR: 'datasets' not installed.  Run:  pip install datasets")

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("ERROR: 'tqdm' not installed.  Run:  pip install tqdm")

# Check for the dill/multiprocess version bug that breaks load_dataset on Python 3.13+.
# Symptom: "Pickler._batch_setitems() takes 2 positional arguments but 3 were given"
# Fix:     pip install --upgrade dill multiprocess
import sys as _sys
if _sys.version_info >= (3, 13):
    try:
        import dill as _dill
        from packaging.version import Version as _V
        if _V(_dill.__version__) < _V("0.3.9"):
            print(
                "WARNING: Python 3.13+ detected with dill < 0.3.9. "
                "You will likely get 'Pickler._batch_setitems()' errors.\n"
                "Fix:  pip install --upgrade dill multiprocess\n",
                file=sys.stderr,
            )
    except ImportError:
        pass  # packaging not available — skip the check silently


# =============================================================================
# LOGGING  (stdout + persistent log file)
# =============================================================================

def setup_logging(log_path: Path) -> logging.Logger:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logger = logging.getLogger("dataset_builder")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        logger.addHandler(sh)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    return logger


# Module-level logger — replaced by setup_logging() in main()
log = logging.getLogger("dataset_builder")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# =============================================================================
# DATASET REGISTRY
# Tuple: (hf_path, config_name_or_None, split, description)
# =============================================================================
REGISTRY: dict[str, tuple] = {
    # Classic
    "lmsys_chat": (
        "lmsys/lmsys-chat-1m", None, "train",
        "Real user x LLM chats (1M). Broad intent diversity.",
    ),
    "wildchat": (
        # Upgraded: WildChat-1M -> WildChat-4.8M (free/ungated, 3.2M non-toxic
        # convos through Aug 2025, includes o1-preview/o1-mini reasoning turns)
        "allenai/WildChat-4.8M", None, "train",
        "Real ChatGPT/o1 logs (3.2M non-toxic). Upgraded from 1M to 4.8M.",
    ),
    "openorca": (
        "Open-Orca/OpenOrca", None, "train",
        "GPT-4/3.5 FLAN completions (4M). Reasoning & instructions.",
    ),
    "ultrachat": (
        "HuggingFaceH4/ultrachat_200k", None, "train_sft",
        "Filtered multi-turn (200k). Clean structure.",
    ),
    "oasst2": (
        "OpenAssistant/oasst2", None, "train",
        "Human-annotated tree conversations (best quality signal).",
    ),
    "dolly": (
        "databricks/databricks-dolly-15k", None, "train",
        "Human-written categorical intents (15k).",
    ),
    "hh_rlhf": (
        # NOTE: hh-rlhf uses data_dir= (not a named config). "helpful-base"
        # is passed via load_kwargs below, not as the config positional arg.
        "Anthropic/hh-rlhf", None, "train",
        "Helpful/harmless preference pairs (170k).",
    ),
    # 2024-2025
    "smoltalk": (
        "HuggingFaceTB/smoltalk", "all", "train",
        "SmolLM2 SFT mix (1M). Purpose-built for SLMs. Apache 2.0.",
    ),
    "tulu3": (
        "allenai/tulu-3-sft-mixture", None, "train",
        "Tulu 3 SFT mix (939k). AllenAI state-of-the-art post-training.",
    ),
    "magpie_pro": (
        "magpie-align/Magpie-Llama-3.1-Pro-1M-v0.1", None, "train",
        "Magpie pipeline on Llama-3.1-70B (1M). High quality synth.",
    ),
    "everyday_convos": (
        "HuggingFaceTB/everyday-conversations-llama3.1-2k", None, "train",
        "Basic everyday conversations (2k). Critical for SLM greetings.",
    ),
    "orca_agentinstruct": (
        # NOTE: this dataset has NO "train" split — it uses named splits like
        # "creative_content", "rc", "rag", "mcq", "follow_up", "code_", etc.
        # process_dataset detects the sentinel "_ALL_SPLITS_" and iterates them.
        "microsoft/orca-agentinstruct-1M-v1", None, "_ALL_SPLITS_",
        "AgentInstruct 1M (Microsoft). Diverse web-grounded tasks.",
    ),
    "chatbot_arena": (
        "lmsys/chatbot_arena_conversations", None, "train",
        "Chatbot Arena conversations with GPT-4/Claude/etc (33k).",
    ),
    # ── New 2024-2025 additions ────────────────────────────────────────────
    "openhermes": (
        "teknium/OpenHermes-2.5", None, "train",
        "1M GPT-4 generated instruction/chat samples. Widely used SFT mix.",
    ),
    "magpie_llama33": (
        "Magpie-Align/Magpie-Llama-3.3-Pro-1M-v0.1", None, "train",
        "Magpie pipeline on Llama-3.3-70B (1M). Newer & higher quality synth.",
    ),
}


# =============================================================================
# TEXT UTILITIES
# =============================================================================

RE_WHITESPACE_ONLY = re.compile(r"^\s*$")
RE_REPEATED_CHAR   = re.compile(r"(.)\1{19,}")   # 20+ identical chars in a row
RE_URL_ONLY        = re.compile(r"^(https?://\S+\s*)+$")
RE_NON_ASCII       = re.compile(r"[^\x00-\x7F]")
RE_MULTI_NEWLINE   = re.compile(r"\n{4,}")


def clean_text(text: str) -> str:
    """Strip leading/trailing whitespace and collapse 4+ blank lines to 2."""
    text = text.strip()
    text = RE_MULTI_NEWLINE.sub("\n\n", text)
    return text


def is_garbage_turn(text: str, min_tokens: int) -> bool:
    """Return True if a turn is junk and should be dropped."""
    if not text or RE_WHITESPACE_ONLY.match(text):
        return True
    if RE_URL_ONLY.match(text.strip()):
        return True
    if RE_REPEATED_CHAR.search(text):
        return True
    # > 40% non-ASCII usually means wrong encoding or non-Latin script
    if len(text) > 0 and (len(RE_NON_ASCII.findall(text)) / len(text)) > 0.40:
        return True
    if len(text.split()) < min_tokens:
        return True
    return False


def is_english(text: str) -> bool:
    """
    Deterministic English detection (seed=42 set at import).
    Samples first 500 chars for speed.
    Very short text (<20 chars) is passed through — langdetect is unreliable
    on tiny samples and short greetings like "Hi!" should be kept.
    """
    sample = text[:500].strip()
    if not sample:
        return False
    if len(sample) < 20:
        return True
    try:
        return detect(sample) == "en"
    except LangDetectException:
        return False


def conversation_hash(turns: list[dict]) -> str:
    """
    SHA-256 dedup hash. Turns are separated by null bytes to prevent
    false collisions between e.g. ["hello world","foo"] and ["hello","world foo"].
    """
    parts = [t.get("content", "").lower().strip() for t in turns]
    flat  = "\x00".join(" ".join(p.split()) for p in parts)
    return hashlib.sha256(flat.encode("utf-8")).hexdigest()


# =============================================================================
# PER-DATASET PARSERS
# Return a normalised dict or None (skip this row entirely).
# ALL field accesses use .get() with safe defaults — no KeyError crashes.
# =============================================================================

def _make_conv(turns: list[dict], source: str) -> dict:
    return {"conversations": turns, "source": source}


def _norm_messages(messages: object, source: str) -> Optional[dict]:
    """Generic normaliser for datasets already in chat/messages format."""
    if not messages or not isinstance(messages, list):
        return None
    turns: list[dict] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role    = str(m.get("role") or "").lower().strip()
        content = clean_text(str(m.get("content") or ""))
        if role not in ("user", "assistant", "system") or not content:
            continue
        turns.append({"role": role, "content": content})
    return _make_conv(turns, source) if turns else None


def parse_lmsys_chat(row: dict) -> Optional[dict]:
    if row.get("language", "English") != "English":
        return None
    return _norm_messages(row.get("conversation"), "lmsys_chat")


def parse_wildchat(row: dict) -> Optional[dict]:
    if row.get("language", "English") != "English":
        return None
    return _norm_messages(row.get("conversation"), "wildchat")


def parse_openorca(row: dict) -> Optional[dict]:
    q     = clean_text(str(row.get("question")      or ""))
    r     = clean_text(str(row.get("response")      or ""))
    sys_p = clean_text(str(row.get("system_prompt") or ""))
    if not q or not r:
        return None
    turns: list[dict] = []
    if sys_p:
        turns.append({"role": "system", "content": sys_p})
    turns.append({"role": "user",      "content": q})
    turns.append({"role": "assistant", "content": r})
    return _make_conv(turns, "openorca")


def parse_ultrachat(row: dict) -> Optional[dict]:
    return _norm_messages(row.get("messages"), "ultrachat")


def build_oasst2_conversations(dataset, source: str = "oasst2") -> list[dict]:
    """
    Walk OASST2's tree structure using ITERATIVE DFS (no recursion limit risk).
    Extracts full conversation chains, English-only, best-ranked path.
    """
    nodes:    dict[str, dict] = {}
    children: dict[str, list] = {}

    log.info("  Building OASST2 node index...")
    for row in dataset:
        mid  = row["message_id"]
        pid  = row.get("parent_id")
        nodes[mid] = {
            "parent_id": pid,
            "role":      "user" if row.get("role") == "prompter" else "assistant",
            "content":   clean_text(str(row.get("text") or "")),
            "rank":      row.get("rank") if row.get("rank") is not None else 99,
            "lang":      row.get("lang", "en"),
        }
        if pid:
            children.setdefault(pid, []).append(mid)

    roots = [mid for mid, n in nodes.items() if n["parent_id"] is None]
    log.info(f"  OASST2: {len(nodes):,} nodes, {len(roots):,} root threads")

    conversations: list[dict] = []

    for root in tqdm(roots, desc="  OASST2 tree walk", unit="thread"):
        # Stack: (message_id, chain_accumulated_so_far)
        stack: list[tuple] = [(root, [])]
        while stack:
            mid, chain = stack.pop()
            node = nodes[mid]
            if node["lang"] not in ("en", ""):
                continue
            new_chain = chain + [{"role": node["role"], "content": node["content"]}]
            kids = children.get(mid, [])
            if not kids:
                if len(new_chain) >= 2:
                    conversations.append(_make_conv(new_chain, source))
            else:
                # Reverse-sort so lowest rank (best) is processed first off the stack
                kids_sorted = sorted(kids, key=lambda k: nodes[k].get("rank", 99), reverse=True)
                for kid in kids_sorted:
                    stack.append((kid, new_chain))

    log.info(f"  OASST2: extracted {len(conversations):,} conversation chains")
    return conversations


def parse_dolly(row: dict) -> Optional[dict]:
    instruction = clean_text(str(row.get("instruction") or ""))
    context     = clean_text(str(row.get("context")     or ""))
    response    = clean_text(str(row.get("response")    or ""))
    if not instruction or not response:
        return None
    user_content = f"{instruction}\n\nContext:\n{context}" if context else instruction
    return _make_conv(
        [{"role": "user",      "content": user_content},
         {"role": "assistant", "content": response}],
        "dolly",
    )


_HH_RE = re.compile(r"\n\n(Human|Assistant): ")

def parse_hh_rlhf(row: dict) -> Optional[dict]:
    """Parse the 'chosen' (highest-quality) branch of HH-RLHF text format."""
    text = str(row.get("chosen") or "").strip()
    if not text:
        return None
    parts = _HH_RE.split(text)
    turns: list[dict] = []
    i = 1
    while i + 1 < len(parts):
        speaker = parts[i].strip()
        content = clean_text(parts[i + 1])
        if content:
            turns.append({
                "role":    "user" if speaker == "Human" else "assistant",
                "content": content,
            })
        i += 2
    return _make_conv(turns, "hh_rlhf") if len(turns) >= 2 else None


def parse_smoltalk(row: dict) -> Optional[dict]:
    return _norm_messages(row.get("messages"), "smoltalk")


def parse_tulu3(row: dict) -> Optional[dict]:
    return _norm_messages(row.get("messages"), "tulu3")


def parse_magpie_pro(row: dict) -> Optional[dict]:
    conv = row.get("conversations") or row.get("conversation")
    if conv:
        return _norm_messages(conv, "magpie_pro")
    instruction = clean_text(str(row.get("instruction") or ""))
    response    = clean_text(str(row.get("response")    or ""))
    if instruction and response:
        return _make_conv(
            [{"role": "user",      "content": instruction},
             {"role": "assistant", "content": response}],
            "magpie_pro",
        )
    return None


def parse_everyday_convos(row: dict) -> Optional[dict]:
    return _norm_messages(row.get("messages"), "everyday_convos")


def parse_orca_agentinstruct(row: dict) -> Optional[dict]:
    messages = row.get("messages") or row.get("conversations")
    if messages:
        return _norm_messages(messages, "orca_agentinstruct")
    instruction = clean_text(str(row.get("instruction") or ""))
    response    = clean_text(str(row.get("response")    or ""))
    if instruction and response:
        return _make_conv(
            [{"role": "user",      "content": instruction},
             {"role": "assistant", "content": response}],
            "orca_agentinstruct",
        )
    return None


def parse_chatbot_arena(row: dict) -> Optional[dict]:
    if row.get("language", "English") != "English":
        return None
    conv = row.get("conversation_a") or row.get("conversation")
    return _norm_messages(conv, "chatbot_arena") if conv else None




def parse_openhermes(row: dict) -> Optional[dict]:
    # OpenHermes-2.5 uses ShareGPT-style "conversations" with from/value keys
    conv = row.get("conversations")
    if not conv or not isinstance(conv, list):
        return None
    turns: list[dict] = []
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    for m in conv:
        if not isinstance(m, dict):
            continue
        role    = role_map.get(str(m.get("from") or "").lower(), None)
        content = clean_text(str(m.get("value") or ""))
        if not role or not content:
            continue
        turns.append({"role": role, "content": content})
    return _make_conv(turns, "openhermes") if turns else None


def parse_magpie_llama33(row: dict) -> Optional[dict]:
    # Same structure as magpie_pro: instruction/response or conversations field
    return parse_magpie_pro(row) and _make_conv(
        parse_magpie_pro(row)["conversations"], "magpie_llama33"
    ) if parse_magpie_pro(row) else None

PARSERS: dict = {
    "lmsys_chat":         parse_lmsys_chat,
    "wildchat":           parse_wildchat,
    "openorca":           parse_openorca,
    "ultrachat":          parse_ultrachat,
    "dolly":              parse_dolly,
    "hh_rlhf":            parse_hh_rlhf,
    "smoltalk":           parse_smoltalk,
    "tulu3":              parse_tulu3,
    "magpie_pro":         parse_magpie_pro,
    "everyday_convos":    parse_everyday_convos,
    "orca_agentinstruct": parse_orca_agentinstruct,
    "chatbot_arena":      parse_chatbot_arena,
    "openhermes":         parse_openhermes,
    "magpie_llama33":     parse_magpie_llama33,
}


# =============================================================================
# QUALITY FILTER  (applied after per-dataset normalisation)
# =============================================================================

def quality_filter(conv: dict, min_tokens: int, min_turns: int) -> Optional[dict]:
    """Returns cleaned conversation dict or None if quality checks fail."""
    turns = conv.get("conversations", [])
    if not turns:
        return None

    clean_turns: list[dict] = []
    for t in turns:
        content = clean_text(str(t.get("content") or ""))
        role    = str(t.get("role") or "")
        if not content or role not in ("user", "assistant", "system"):
            continue
        # System prompts are kept as-is (they can legitimately be short)
        if role != "system" and is_garbage_turn(content, min_tokens):
            continue
        clean_turns.append({"role": role, "content": content})

    # Require min_turns complete user+assistant exchanges (system doesn't count)
    exchange = [t for t in clean_turns if t["role"] != "system"]
    if len(exchange) < min_turns * 2:
        return None

    # English check on first user message only (fast path)
    first_user = next((t["content"] for t in clean_turns if t["role"] == "user"), None)
    if first_user and not is_english(first_user):
        return None

    conv["conversations"] = clean_turns
    return conv


# =============================================================================
# CHUNKED WRITER
# Writes JSONL lines, rotating to a new file every chunk_bytes bytes.
# Files: english_conversations_001.jsonl, _002.jsonl, ...
# =============================================================================

class ChunkedWriter:
    def __init__(self, out_dir: Path, base_name: str, chunk_bytes: int):
        self.out_dir      = out_dir
        self.base_name    = base_name
        self.chunk_bytes  = chunk_bytes
        self.chunk_index  = 1
        self.current_size = 0
        self.files_written: list[Path] = []
        self._fh = None
        self._open_next()

    def _open_next(self):
        if self._fh:
            self._fh.flush()
            self._fh.close()
        path = self.out_dir / f"{self.base_name}_{self.chunk_index:03d}.jsonl"
        self._fh = open(path, "w", encoding="utf-8")
        self.files_written.append(path)
        log.info(f"   -> Opened chunk {self.chunk_index:03d}: {path.name}")
        self.chunk_index  += 1   # increment AFTER logging so number matches filename
        self.current_size  = 0

    def write(self, line: str):
        encoded = line if line.endswith("\n") else line + "\n"
        size    = len(encoded.encode("utf-8"))
        if self.current_size + size > self.chunk_bytes and self.current_size > 0:
            self._open_next()
        self._fh.write(encoded)
        self.current_size += size

    def flush(self):
        if self._fh:
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None


# =============================================================================
# RESUME / CHECKPOINT
# =============================================================================
PROGRESS_FILENAME = "progress.json"


def load_progress(out_dir: Path) -> dict:
    p = out_dir / PROGRESS_FILENAME
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"Could not read progress file ({e}) — starting fresh.")
    return {}


def save_progress(out_dir: Path, progress: dict):
    p = out_dir / PROGRESS_FILENAME
    try:
        p.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning(f"Could not save progress file: {e}")


# =============================================================================
# SUMMARY  (called on both normal completion and interrupted exit)
# =============================================================================

def print_summary(
    t0: float,
    keys_done: list[str],
    grand_total: int,
    grand_kept: int,
    grand_written: int,
    grand_errors: int,
    writer: ChunkedWriter,
    interrupted: bool = False,
):
    elapsed     = time.time() - t0
    total_bytes = sum(f.stat().st_size for f in writer.files_written if f.exists())
    status      = "INTERRUPTED -- partial data saved" if interrupted else "COMPLETE"

    log.info("=" * 60)
    log.info(f" {status}")
    log.info(f" Elapsed           : {elapsed / 60:.1f} min")
    log.info(f" Datasets finished : {', '.join(keys_done) if keys_done else 'none'}")
    log.info(f" Rows processed    : {grand_total:,}")
    log.info(f" Passed filters    : {grand_kept:,}")
    log.info(f" Written (deduped) : {grand_written:,}")
    log.info(f" Parse errors      : {grand_errors:,}")
    log.info(f" Chunks written    : {len(writer.files_written)}")
    log.info(f" Total size        : {total_bytes / 1e9:.3f} GB")
    log.info(f" Dedup RAM (approx): ~{(grand_written * 72) / 1e6:.1f} MB")
    log.info("=" * 60)


# =============================================================================
# DATASET PROCESSOR
# =============================================================================

def process_dataset(
    key: str,
    args: argparse.Namespace,
    seen_hashes: set,
    writer: ChunkedWriter,
    hf_token: str | None = None,
) -> tuple[int, int, int, int]:
    """
    Load, parse, filter, dedup, write one dataset.
    Returns: (total_rows, kept, written, parse_errors)
    """
    path, config, split, desc = REGISTRY[key]
    log.info(f"{'─' * 60}")
    log.info(f"  {key}  ({path})")
    log.info(f"   {desc}")

    # ── OASST2: special tree-walk, no streaming ───────────────────────────
    if key == "oasst2":
        try:
            ds_full = load_dataset(
                path, split=split, trust_remote_code=False,
                **({"token": hf_token} if hf_token else {}),
            )
        except Exception as e:
            log.warning(f"   SKIP -- could not load oasst2: {e}")
            return 0, 0, 0, 0

        conversations = build_oasst2_conversations(ds_full)
        total = len(conversations)
        kept = written = errors = 0

        for conv in tqdm(conversations, desc="  oasst2 filter", unit="conv"):
            try:
                filtered = quality_filter(conv, args.min_tokens, args.min_turns)
            except Exception:
                errors += 1
                continue
            if not filtered:
                continue
            kept += 1
            h = conversation_hash(filtered["conversations"])
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            writer.write(json.dumps(filtered, ensure_ascii=False))
            written += 1

        if errors:
            log.warning(f"   {errors:,} filter errors (skipped)")
        log.info(f"   total={total:,}  kept={kept:,}  written={written:,}  errors={errors:,}")
        return total, kept, written, errors

    # ── All other datasets: streaming row-by-row ──────────────────────────

    # orca_agentinstruct has no "train" split — it uses many named splits.
    # We detect the sentinel and concatenate all available splits.
    if split == "_ALL_SPLITS_":
        try:
            from datasets import get_dataset_split_names, concatenate_datasets
            split_names = get_dataset_split_names(path)
            log.info(f"   Loading {len(split_names)} splits: {', '.join(split_names)}")
            base_kwargs: dict = dict(streaming=True, trust_remote_code=False)
            if hf_token:
                base_kwargs["token"] = hf_token
            split_datasets = []
            for sn in split_names:
                try:
                    split_datasets.append(load_dataset(path, split=sn, **base_kwargs))
                except Exception as e:
                    log.warning(f"   Could not load split '{sn}': {e}")
            if not split_datasets:
                log.warning(f"   SKIP -- no splits could be loaded for '{key}'")
                return 0, 0, 0, 0
            from datasets import interleave_datasets
            ds = interleave_datasets(split_datasets)
        except Exception as e:
            log.warning(f"   SKIP -- could not load multi-split dataset: {e}")
            return 0, 0, 0, 0
    else:
        try:
            load_kwargs: dict = dict(split=split, streaming=True, trust_remote_code=False)
            if config:
                load_kwargs["name"] = config
            # hh-rlhf uses data_dir= to select the subset, not a named config
            if key == "hh_rlhf":
                load_kwargs["data_dir"] = "helpful-base"
            if hf_token:
                load_kwargs["token"] = hf_token
            ds = load_dataset(path, **load_kwargs)
        except Exception as e:
            log.warning(f"   SKIP -- could not load: {e}")
            return 0, 0, 0, 0

    parser = PARSERS.get(key)
    if parser is None:
        log.warning(f"   SKIP -- no parser registered for '{key}'")
        return 0, 0, 0, 0

    total = kept = written = errors = 0
    limit = args.limit

    with tqdm(desc=f"  {key}", unit="row", dynamic_ncols=True) as pbar:
        for row in ds:
            total += 1
            if limit and total > limit:
                break
            pbar.update(1)

            try:
                parsed = parser(row)
                if parsed is None:
                    continue
                filtered = quality_filter(parsed, args.min_tokens, args.min_turns)
                if filtered is None:
                    continue
            except Exception:
                errors += 1
                continue

            kept += 1
            h = conversation_hash(filtered["conversations"])
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            writer.write(json.dumps(filtered, ensure_ascii=False))
            written += 1

            # Flush every 5000 written rows so data survives a hard kill
            if written % 5000 == 0:
                writer.flush()

    if errors:
        log.warning(f"   {errors:,} rows raised parse/filter errors (skipped)")
    log.info(f"   total={total:,}  kept={kept:,}  written={written:,}  errors={errors:,}")
    return total, kept, written, errors


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download & build clean English conversation JSONL dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--output_dir",  default="./convos/english",
        help="Output directory (default: ./convos/english)",
    )
    p.add_argument(
        "--output_file", default="english_conversations",
        help="Base filename without extension (default: english_conversations)",
    )
    p.add_argument(
        "--chunk_mb", type=int, default=50,
        help="Max MB per output chunk for GitHub (default: 50)",
    )
    p.add_argument(
        "--min_tokens", type=int, default=8,
        help="Min whitespace-split tokens per turn (default: 8)",
    )
    p.add_argument(
        "--min_turns", type=int, default=1,
        help="Min user/assistant exchange pairs per convo (default: 1)",
    )
    p.add_argument(
        "--limit", type=int, default=0,
        help="Max rows per dataset, 0=unlimited (default: 0)",
    )
    p.add_argument(
        "--skip", default="",
        help="Comma-separated dataset keys to skip",
    )
    p.add_argument(
        "--only", default="",
        help="Comma-separated dataset keys to run exclusively",
    )
    p.add_argument(
        "--no_resume", action="store_true",
        help="Ignore progress.json and reprocess all datasets from scratch",
    )
    p.add_argument(
        "--list", action="store_true",
        help="Print available dataset keys and exit",
    )
    p.add_argument(
        "--hf_token", default=None,
        help="HuggingFace token for gated datasets (or set HF_TOKEN env var)",
    )
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    if args.list:
        print("\nAvailable dataset keys:\n")
        for k, (path, cfg, split, desc) in REGISTRY.items():
            print(f"  {k:<24} {path}")
            print(f"  {'':24} {desc}\n")
        return

    # ── Output dir + log file ─────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    global log
    log = setup_logging(out_dir / "build.log")
    log.info(f"Log file: {out_dir / 'build.log'}")

    # ── HuggingFace auth ──────────────────────────────────────────────────
    hf_token = args.hf_token or HF_TOKEN or os.environ.get("HF_TOKEN") or None
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            log.info("Logged in to HuggingFace Hub.")
        except ImportError:
            log.warning("huggingface_hub not installed; HF token login skipped.")

    # ── Disk space pre-check ──────────────────────────────────────────────
    free_gb = shutil.disk_usage(out_dir).free / 1e9
    log.info(f"Free disk on output volume: {free_gb:.1f} GB")
    if free_gb < 10:
        log.warning(
            f"Only {free_gb:.1f} GB free. A full run needs ~50-80 GB. "
            "Consider --only or --limit to reduce output."
        )

    # ── Dataset selection ─────────────────────────────────────────────────
    skip_keys = {k.strip() for k in args.skip.split(",") if k.strip()}
    only_keys = {k.strip() for k in args.only.split(",") if k.strip()}
    if only_keys:
        keys_to_run = [k for k in REGISTRY if k in only_keys]
    else:
        keys_to_run = [k for k in REGISTRY if k not in skip_keys]

    if not keys_to_run:
        log.error("No datasets selected. Check --skip / --only arguments.")
        sys.exit(1)

    # ── Resume support ────────────────────────────────────────────────────
    progress = {} if args.no_resume else load_progress(out_dir)
    already_done: set = set(progress.get("completed", []))
    if already_done:
        resuming = already_done & set(keys_to_run)
        if resuming:
            log.info(f"Resuming run -- skipping already completed: {', '.join(sorted(resuming))}")
            log.info("  Use --no_resume to reprocess everything from scratch.")
        keys_to_run = [k for k in keys_to_run if k not in already_done]

    if not keys_to_run:
        log.info("All selected datasets already completed. Nothing to do.")
        log.info("Use --no_resume to reprocess from scratch.")
        return

    # ── RAM advisory ──────────────────────────────────────────────────────
    log.info(
        "Note: dedup hash table grows in RAM (~72 bytes per unique convo). "
        "Full run (~7M convos) uses ~500 MB RAM for hashes alone."
    )

    log.info("=" * 60)
    log.info(" English Conversation Dataset Builder")
    log.info("=" * 60)
    log.info(f" Output dir  : {out_dir}")
    log.info(f" Chunk size  : {args.chunk_mb} MB per file")
    log.info(f" Datasets    : {', '.join(keys_to_run)}")
    log.info(f" Filters     : min_tokens={args.min_tokens}  min_turns={args.min_turns}")
    if args.limit:
        log.info(f" Row cap     : {args.limit:,} per dataset")
    log.info("=" * 60)

    # ── State ─────────────────────────────────────────────────────────────
    seen_hashes:   set[str] = set()
    chunk_bytes              = args.chunk_mb * 1024 * 1024
    writer                   = ChunkedWriter(out_dir, args.output_file, chunk_bytes)
    t0                       = time.time()
    grand_total              = 0
    grand_kept               = 0
    grand_written            = 0
    grand_errors             = 0
    keys_done:  list[str]    = sorted(already_done)
    interrupted              = False

    # ── Signal handlers ───────────────────────────────────────────────────
    def _graceful_exit(signum, frame):
        nonlocal interrupted
        interrupted = True
        sig_name = "SIGINT (Ctrl+C)" if signum == signal.SIGINT else "SIGTERM (kill)"
        # Print directly to stderr so it shows even if stdout is redirected
        print(f"\n  {sig_name} received. Finishing current row then exiting cleanly...",
              file=sys.stderr, flush=True)

    signal.signal(signal.SIGINT,  _graceful_exit)
    signal.signal(signal.SIGTERM, _graceful_exit)

    # ── Main processing loop ──────────────────────────────────────────────
    try:
        for key in keys_to_run:
            if interrupted:
                log.warning("Interrupt flag set — stopping before next dataset.")
                break

            try:
                total, kept, written, errors = process_dataset(
                    key, args, seen_hashes, writer, hf_token=hf_token
                )
            except KeyboardInterrupt:
                # Secondary catch for platforms where signal handler may not fire
                interrupted = True
                log.warning(f"KeyboardInterrupt during '{key}' — stopping.")
                break
            except MemoryError:
                log.error(
                    "MemoryError — dedup hash table too large. "
                    "Use --only to split the run across sessions, or add more RAM."
                )
                interrupted = True
                break
            except Exception as e:
                log.error(f"Unexpected error processing '{key}': {e}")
                log.error(traceback.format_exc())
                log.warning(f"Skipping '{key}' and continuing with the next dataset.")
                continue

            grand_total   += total
            grand_kept    += kept
            grand_written += written
            grand_errors  += errors
            keys_done.append(key)

            # Save checkpoint after every completed dataset
            progress["completed"] = keys_done
            save_progress(out_dir, progress)
            log.info(f"  Checkpoint saved. ({len(keys_done)} dataset(s) complete)")

    finally:
        # This block ALWAYS runs — normal exit, Ctrl+C, kill, or crash
        writer.close()
        print_summary(
            t0, keys_done, grand_total, grand_kept,
            grand_written, grand_errors, writer, interrupted,
        )
        if interrupted:
            remaining = [k for k in keys_to_run if k not in keys_done]
            log.info(f"Remaining datasets: {', '.join(remaining) if remaining else 'none'}")
            log.info("Re-run this script to resume from where it left off.")
            print(f"\n  Interrupted. Partial output is safe in: {out_dir}", file=sys.stderr)
            print( "  Re-run to resume (progress.json tracks completed datasets).", file=sys.stderr)
        else:
            print(f"\nDone. {len(writer.files_written)} chunk(s) written to: {out_dir}")


if __name__ == "__main__":
    main()