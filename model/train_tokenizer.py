"""
train_tokenizer.py
==================
Trains a BPE tokenizer on your Rust SLM dataset.

Handles all four schema variants:
  - snippets      (strandset-rust-v1)
  - docs/crates   (source: docs | crates.io | github)
  - lectures      (style: lecture | technical)
  - conversations (has 'turns' array)

Usage:
    python train_tokenizer.py --data_dir C:/llm/data --output_dir C:/llm/tokenizer

Optional flags:
    --vocab_size    BPE vocabulary size         (default: 40960)
    --min_frequency Minimum BPE merge frequency (default: 5)
    --min_chars     Minimum characters per record to keep (default: 50)

Requirements:
    pip install tokenizers transformers tqdm beautifulsoup4 lxml
"""

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from tqdm import tqdm
from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFC
from tokenizers.processors import TemplateProcessing

# Suppress XMLParsedAsHTMLWarning — some scraped docs are XML-flavoured;
# html.parser handles them fine and we don't need the noise on every record.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


# ---------------------------------------------------------------------------
# Special tokens — baked in now, never need to retrain
# ---------------------------------------------------------------------------
SPECIAL_TOKENS = [
    "<|pad|>",        # 0  — padding
    "<|bos|>",        # 1  — beginning of sequence
    "<|eos|>",        # 2  — end of sequence
    "<|unk|>",        # 3  — unknown fallback
    "<|system|>",     # 4  — system prompt start
    "<|user|>",       # 5  — user turn start
    "<|assistant|>",  # 6  — assistant turn start
    "<|endturn|>",    # 7  — end of any turn
    "<|code|>",       # 8  — code block start
    "<|endcode|>",    # 9  — code block end
]

# ---------------------------------------------------------------------------
# Defaults — all overridable via CLI flags (see __main__ at the bottom).
# train_tokenizer() also accepts these as explicit parameters so it can be
# called programmatically without going through argparse.
# ---------------------------------------------------------------------------
DEFAULT_VOCAB_SIZE    = 40_960   # bumped from 32K; Rust generics/lifetimes need room
DEFAULT_MIN_FREQUENCY = 5        # was 2; filters low-frequency noise pairs from vocab
DEFAULT_MIN_CHARS     = 50       # records with fewer extracted chars are skipped

# ---------------------------------------------------------------------------
# Sanity-check idioms used in the efficiency test after training.
# Covers keywords, stdlib types, crate names, and realistic patterns.
# After a good training run, aim for 1-3 tokens each.
# ---------------------------------------------------------------------------
SANITY_IDIOMS = [
    # Core keywords / common patterns
    "fn",
    "pub async fn",
    "impl Trait",
    "async fn",
    # Snake_case identifiers — the main win from the new pre-tokenizer
    "unwrap_or_else",
    "unwrap_or_default",
    "from_utf8_unchecked",
    # Generic types
    "Vec<String>",
    "Result<T, E>",
    "Arc<Mutex<T>>",
    "HashMap<String, Vec<u8>>",
    "Option<Arc<Mutex<T>>>",
    # Path syntax
    "tokio::spawn",
    "std::sync::Arc",
    "serde_json::Value",
    # Attributes
    "#[derive(Debug, Clone)]",
    "#[tokio::main]",
    # Iterator chains
    ".iter().map(|x|",
    ".collect::<Vec<_>>()",
    # Lifetime syntax
    "&'static str",
    "fn foo<'a>(x: &'a str)",
]


# ---------------------------------------------------------------------------
# Code-fence language tags we recognise and wrap with <|code|>/<|endcode|>.
# Fences with missing or unrecognised tags are also matched (the group is
# optional in the regex below).
# ---------------------------------------------------------------------------
_FENCE_LANGS = (
    "rust", "rs", "toml", "cargo",
    "sh", "bash", "zsh", "console", "terminal",
    "json", "yaml", "yml", "xml",
    "text", "plain", "md", "markdown",
    "c", "cpp", "python", "py",
)
_CODE_FENCE_RE = re.compile(
    r"```(?:" + "|".join(_FENCE_LANGS) + r")?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# HTML stripper — handles noisy doc scrapes like the async-book
# ---------------------------------------------------------------------------
def strip_html(text: str) -> str:
    """Remove HTML tags and collapse excessive whitespace."""
    if "<" not in text:
        return text
    try:
        soup  = BeautifulSoup(text, "html.parser")
        clean = soup.get_text(separator=" ")
    except Exception as exc:
        warnings.warn(f"strip_html: BeautifulSoup failed ({exc!r}), falling back to regex strip")
        # fallback: crude regex tag strip
        clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s{3,}", "\n\n", clean)
    return clean.strip()


# ---------------------------------------------------------------------------
# Code-block wrapper
# Replaces markdown ``` fences with <|code|>…<|endcode|> special tokens.
# This teaches the model that those tokens precede a Rust-specific token
# distribution — pays off meaningfully during SFT.
# ---------------------------------------------------------------------------
def wrap_code_blocks(text: str) -> str:
    """Replace markdown code fences with <|code|>…<|endcode|> markers."""
    return _CODE_FENCE_RE.sub(
        lambda m: f"<|code|>{m.group(1).rstrip()}<|endcode|>",
        text,
    )


# ---------------------------------------------------------------------------
# Schema-aware text extractors
# ---------------------------------------------------------------------------
def extract_snippet(record: dict) -> str:
    """
    strandset-rust-v1 snippets.
    Pulls every useful field from the record.
    Raw code fields (fixed_code, broken_code) are emitted as-is;
    explanation text may contain markdown fences so we wrap those.
    """
    parts = []

    # Raw code — no fence wrapping, already bare code
    for field in ("fixed_code", "broken_code"):
        val = record.get(field)
        if val and isinstance(val, str) and val.strip():
            parts.append(val.strip())

    # Explanation may contain markdown code fences — wrap them
    explanation = record.get("explanation")
    if explanation and isinstance(explanation, str) and explanation.strip():
        parts.append(wrap_code_blocks(explanation.strip()))

    # Validation block — clippy output is gold for a Rust model
    seen_error_message: str | None = None
    validation = record.get("validation")
    if isinstance(validation, dict):
        for field in ("clippy_output", "error_message"):
            val = validation.get(field)
            if val and isinstance(val, str) and val.strip():
                parts.append(val.strip())
                if field == "error_message":
                    seen_error_message = val.strip()

    # Top-level error fields (older schema variants).
    # Skip error_message if we already emitted the identical text from validation.
    for field in ("error_message", "error_code"):
        val = record.get(field)
        if not (val and isinstance(val, str) and val.strip()):
            continue
        if field == "error_message" and val.strip() == seen_error_message:
            continue
        parts.append(val.strip())

    return "\n\n".join(parts)


def extract_content(record: dict) -> str:
    """
    docs / crates.io / github records — text lives in 'content'.
    Order: strip_html first (backtick fences survive plain-text extraction),
    then wrap_code_blocks to convert them to special tokens.
    """
    content = record.get("content", "")
    if not content or not isinstance(content, str):
        return ""
    return wrap_code_blocks(strip_html(content.strip()))


def extract_lecture(record: dict) -> str:
    """Lecture records — 'content' is usually clean; wrap any code fences."""
    content = record.get("content", "")
    if not content or not isinstance(content, str):
        return ""
    return wrap_code_blocks(content.strip())


def extract_conversation(record: dict) -> str:
    """
    Multi-turn conversations. Format with role markers so the tokenizer
    learns the turn structure — critical for Phase 2 SFT.
    Code fences inside turns are wrapped with <|code|>/<|endcode|>.
    """
    turns = record.get("turns", [])
    if not turns or not isinstance(turns, list):
        return ""

    parts = []
    for turn in turns:
        role    = turn.get("role", "user").strip().lower()
        content = turn.get("content", "").strip()
        if not content:
            continue
        content = wrap_code_blocks(content)
        if role == "assistant":
            parts.append(f"<|assistant|>{content}<|endturn|>")
        elif role == "system":
            parts.append(f"<|system|>{content}<|endturn|>")
        else:
            parts.append(f"<|user|>{content}<|endturn|>")

    return "\n".join(parts)


def dispatch(record: dict) -> str:
    """
    Route a record to the right extractor based on its schema.
    Returns empty string if nothing useful can be extracted.
    """
    # Conversations: has a 'turns' list
    if "turns" in record and isinstance(record.get("turns"), list):
        return extract_conversation(record)

    # Lectures: has a 'style' field
    if "style" in record:
        return extract_lecture(record)

    # Snippets: schema_variant == 'snippet', or has 'fixed_code', or known source
    source         = record.get("source", "")
    schema_variant = record.get("schema_variant", "")
    if (
        schema_variant == "snippet"
        or "fixed_code" in record
        or source == "strandset-rust-v1"
    ):
        return extract_snippet(record)

    # Docs / Crates / GitHub
    if source in ("docs", "crates.io", "github") or "content" in record:
        return extract_content(record)

    # Fallback: try 'text' (a record with 'content' would have been caught above)
    val = record.get("text")
    if val and isinstance(val, str) and val.strip():
        return wrap_code_blocks(strip_html(val.strip()))

    return ""


# ---------------------------------------------------------------------------
# Extraction statistics — populated by iter_jsonl_files / text_iterator,
# printed by train_tokenizer after the training loop completes.
# Separating dispatch_empty from too_short lets you distinguish "schema not
# recognised at all" failures from "record too small to be useful" skips.
# ---------------------------------------------------------------------------
@dataclass
class ExtractionStats:
    total:          int = 0   # records seen
    dispatch_empty: int = 0   # dispatch() returned "" — schema unrecognised
    too_short:      int = 0   # extracted text was non-empty but < min_chars
    json_errors:    int = 0   # malformed JSON lines skipped

    @property
    def usable(self) -> int:
        return self.total - self.dispatch_empty - self.too_short


# ---------------------------------------------------------------------------
# JSONL reader — streams records without loading all files into RAM
# ---------------------------------------------------------------------------
def iter_jsonl_files(
    data_dir: Path,
    stats: "ExtractionStats",
) -> Generator[dict, None, None]:
    """Recursively find and stream every .jsonl file under data_dir."""
    files = sorted(data_dir.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found under {data_dir}")
    print(f"\nFound {len(files)} JSONL file(s):")
    for f in files:
        print(f"  {f}")
    print()
    for filepath in tqdm(files, desc="Files", unit="file"):
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    stats.json_errors += 1
                    continue


def text_iterator(
    data_dir: Path,
    min_chars: int = DEFAULT_MIN_CHARS,
    stats: "ExtractionStats | None" = None,
) -> Generator[str, None, None]:
    """
    Yields extracted text strings for tokenizer training.

    Populates *stats* (an ExtractionStats instance) with per-category counts
    so the caller can print a meaningful summary after the iterator is drained.
    Separates dispatch_empty (unrecognised schema) from too_short (extracted
    text present but below min_chars) so failures are distinguishable.
    """
    if stats is None:
        stats = ExtractionStats()
    for record in iter_jsonl_files(data_dir, stats):
        stats.total += 1
        text = dispatch(record)
        if not text:
            stats.dispatch_empty += 1
            continue
        if len(text) < min_chars:
            stats.too_short += 1
            continue
        yield text


# ---------------------------------------------------------------------------
# Tokenizer construction
# ---------------------------------------------------------------------------
def _build_tokenizer(vocab_size: int, min_frequency: int) -> tuple[Tokenizer, BpeTrainer]:
    """
    Construct and return a configured (but untrained) BPE tokenizer and its
    trainer.  Separated from training so the setup can be tested independently
    and re-run without touching the filesystem.
    """
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))

    # NFC normalization — consistent unicode representation across platforms
    tokenizer.normalizer = NFC()

    # ------------------------------------------------------------------
    # Pre-tokenizer: Rust-aware Regex → ByteLevel
    #
    # Plain ByteLevel alone splits on every non-alphanumeric character,
    # destroying Rust snake_case identifiers:
    #   unwrap_or_else → ['unwrap', '_', 'or', '_', 'else']  (5 tokens — bad)
    #
    # The Rust regex runs FIRST to preserve identifiers, numeric literals,
    # and important multi-char operators as atomic units before BPE sees them.
    # ByteLevel then handles unicode safety on whatever remains.
    #
    # Expected token counts after retraining:
    #   unwrap_or_else → 1 token   (snake_case stays atomic)
    #   pub async fn   → 3 tokens  (pub / Ġasync / Ġfn — no standalone Ġ)
    #   tokio::spawn   → 3 tokens  (tokio / :: / spawn)
    #   Result<T, E>   → 6 tokens  (Result / < / T / , / ĠE / >)
    # ------------------------------------------------------------------
    rust_regex = Regex(
        r"""(?x)
        # Space-prefixed identifier patterns come FIRST so that a single space
        # before a word is captured together with it (GPT-2 style: " async" →
        # one piece → ByteLevel → "Ġasync").  This prevents whitespace from
        # becoming a standalone "Ġ" token between every pair of keywords.
        #
        # Without this:  pub async fn  →  ['pub','Ġ','async','Ġ','fn']  (5 tokens)
        # With this:     pub async fn  →  ['pub','Ġasync','Ġfn']        (3 tokens)
        #
        # Multi-space runs (indentation, blank lines) still fall through to
        # the \s+ rule at the bottom and become their own whitespace piece.
        [ ]r\#[a-zA-Z_][a-zA-Z0-9_]*\#     # single space + raw identifier: ' r#async#'
        |[ ][a-zA-Z_][a-zA-Z0-9_]*           # single space + identifier:     ' unwrap_or_else'
        |r\#[a-zA-Z_][a-zA-Z0-9_]*\#         # raw identifier (no leading space)
        |[a-zA-Z_][a-zA-Z0-9_]*              # identifier / keyword (no leading space)
        |0x[0-9a-fA-F][0-9a-fA-F_]*         # hex literals:    0xFF, 0xDEAD_BEEF
        |0b[01][01_]*                         # binary literals: 0b1010_1010
        |0o[0-7][0-7_]*                      # octal literals:  0o777
        |[0-9][0-9_]*(?:\.[0-9][0-9_]*)?    # decimal / float: 1_000_000, 3.14
        |::                                   # path separator
        |->                                   # return type arrow
        |=>                                   # match arm arrow
        |\.\.=                                # inclusive range ..=
        |\.\.                                 # exclusive range / struct update ..
        |<<=|>>=                              # shift-assign
        |&&=|\|\|=                            # logical-assign
        |<<|>>                                # bit shift
        |&&|\|\|                              # logical and / or
        |[+\-*/%&|^]=                        # op-assign: +=, -=, *=, /=, etc.
        |[(){}\[\];:,.<>=!?@#~$^&|]         # single-char punctuation
        |\s+                                  # remaining whitespace (indentation, newlines)
        """
    )

    tokenizer.pre_tokenizer = Sequence([
        Split(rust_regex, behavior="isolated"),
        ByteLevel(add_prefix_space=False, use_regex=False),
    ])

    # ByteLevel decoder correctly reconstructs the original bytes
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=min_frequency,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
    )

    return tokenizer, trainer


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def _save_tokenizer(tokenizer: Tokenizer, output_dir: Path, vocab_size: int) -> Path:
    """
    Save all files that AutoTokenizer.from_pretrained() expects.
    Returns the path to tokenizer.json.
    Separated from training so a save failure can be retried without retraining.
    """
    # 1. Main tokenizer weights
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"\nTokenizer saved        → {tokenizer_path}")

    # 2. tokenizer_config.json
    config = {
        "tokenizer_class":      "PreTrainedTokenizerFast",
        "bos_token":            "<|bos|>",
        "eos_token":            "<|eos|>",
        "unk_token":            "<|unk|>",
        "pad_token":            "<|pad|>",
        "model_max_length":     2048,
        "special_tokens_map_file": None,
    }
    config_path = output_dir / "tokenizer_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved           → {config_path}")

    # 3. special_tokens_map.json
    # AutoTokenizer.from_pretrained() warns (and may error in strict mode)
    # when this file is absent. Saving it explicitly prevents that.
    special_tokens_map = {
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "unk_token": "<|unk|>",
        "pad_token": "<|pad|>",
        "additional_special_tokens": [
            t for t in SPECIAL_TOKENS
            if t not in ("<|bos|>", "<|eos|>", "<|unk|>", "<|pad|>")
        ],
    }
    spm_path = output_dir / "special_tokens_map.json"
    with open(spm_path, "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, indent=2)
    print(f"Special tokens map     → {spm_path}")

    # 4. vocab.txt — human-readable ID → token for debugging merges.
    #    Examples:
    #      grep "unwrap" C:/llm/tokenizer/vocab.txt
    #      grep "tokio"  C:/llm/tokenizer/vocab.txt
    vocab        = tokenizer.get_vocab()
    vocab_sorted = sorted(vocab.items(), key=lambda kv: kv[1])
    vocab_path   = output_dir / "vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for tok, idx in vocab_sorted:
            f.write(f"{idx}\t{tok}\n")
    print(f"Vocab saved            → {vocab_path}")

    return tokenizer_path


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
def _run_sanity_checks(tokenizer: Tokenizer, vocab_size: int) -> bool:
    """
    Run post-training sanity checks.  Returns True if all checks pass.
    Separated so it can be called independently on a saved tokenizer.
    """
    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    bos_eos_ids = {bos_id, eos_id}

    print("\n" + "=" * 60)
    print("  SANITY CHECKS")
    print("=" * 60)

    # 1. Confirm every special token got the expected ID
    print("\nSpecial token IDs:")
    id_ok = True
    for expected_id, tok in enumerate(SPECIAL_TOKENS):
        actual_id = tokenizer.token_to_id(tok)
        match = "✓" if actual_id == expected_id else "✗"
        if actual_id != expected_id:
            id_ok = False
        print(f"  {match} {tok:<20} → {actual_id}  (expected {expected_id})")
    if not id_ok:
        print("  !! WARNING: Special token IDs do not match expected positions !!")

    # 2. Round-trip encode → decode
    #
    # Why we filter by ID and not string-replace:
    #   The post-processor added bos_id / eos_id to encoded.ids.
    #   We strip them by ID before calling decode, so the decoder never sees
    #   them and we don't have to guess how they'll be rendered as strings.
    #   String replace ("<|bos|> ", "") was the previous approach and it was
    #   fragile — any whitespace variation would cause a false FAIL.
    test_cases = [
        "fn main() {\n    println!(\"Hello, Rust!\");\n}",
        "impl<T: Clone> Iterator for MyStruct<T> { type Item = T; }",
        "use std::sync::{Arc, Mutex};\nuse tokio::runtime::Runtime;",
        # Special tokens in the stream — previously always failed
        (
            "<|user|>How do lifetimes work in Rust?<|endturn|>\n"
            "<|assistant|>Lifetimes ensure references are valid.<|endturn|>"
        ),
        # <|code|> / <|endcode|> tokens
        (
            "<|user|>Show me an example<|endturn|>\n"
            "<|assistant|><|code|>fn add(a: i32, b: i32) -> i32 { a + b }"
            "<|endcode|><|endturn|>"
        ),
    ]

    print("\nRound-trip encode → decode tests:")
    all_passed = True
    for i, text in enumerate(test_cases):
        encoded   = tokenizer.encode(text)
        # Strip BOS/EOS by ID — the only robust approach
        inner_ids = [id_ for id_ in encoded.ids if id_ not in bos_eos_ids]
        decoded   = tokenizer.decode(inner_ids, skip_special_tokens=False)
        ok        = decoded == text
        all_passed = all_passed and ok
        status    = "✓ PASS" if ok else "✗ FAIL"
        total_toks = len(encoded.ids)
        print(f"  [{status}] Test {i+1}: {total_toks} tokens | {repr(text[:55])}")
        if not ok:
            print(f"           Original : {repr(text)}")
            print(f"           Decoded  : {repr(decoded)}")

    # 3. Rust idiom tokenization efficiency
    print("\nRust idiom token efficiency (fewer = better compression):")
    for idiom in SANITY_IDIOMS:
        enc            = tokenizer.encode(idiom)
        content_tokens = [
            t for id_, t in zip(enc.ids, enc.tokens)
            if id_ not in bos_eos_ids
        ]
        print(f"  {idiom:<42} → {len(content_tokens):>2} token(s): {content_tokens}")

    # 4. Vocab stats + corpus-size warning
    vocab = tokenizer.get_vocab()
    print(f"\nVocabulary stats:")
    print(f"  Final vocab size : {len(vocab):,} / {vocab_size:,}")
    print(f"  Special tokens   : {len(SPECIAL_TOKENS)}")
    if len(vocab) < vocab_size:
        shortfall = vocab_size - len(vocab)
        print(f"  !! Vocab is {shortfall:,} tokens short of the requested size.")
        print(f"     This means your corpus didn't have enough unique pairs to fill it.")
        print(f"     Add more training data, or lower --vocab_size for next run.")

    # Final verdict
    print("\n" + "=" * 60)
    passed = all_passed and id_ok
    if passed:
        print("  All checks PASSED — tokenizer is healthy!")
    else:
        print("  WARNING: One or more checks FAILED — review output above.")
    print("=" * 60)

    return passed


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def train_tokenizer(
    data_dir: Path,
    output_dir: Path,
    vocab_size: int    = DEFAULT_VOCAB_SIZE,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
    min_chars: int     = DEFAULT_MIN_CHARS,
) -> None:
    """
    Train a BPE tokenizer and save it in HuggingFace format.

    Parameters
    ----------
    data_dir      : directory containing .jsonl training files (searched recursively)
    output_dir    : where to write tokenizer.json, tokenizer_config.json,
                    special_tokens_map.json, and vocab.txt
    vocab_size    : total BPE vocabulary size including special tokens
    min_frequency : minimum pair frequency required for a BPE merge
    min_chars     : minimum extracted-text length to include a record
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Rust SLM Tokenizer Trainer")
    print("=" * 60)
    print(f"  Data dir      : {data_dir}")
    print(f"  Output dir    : {output_dir}")
    print(f"  Vocab size    : {vocab_size:,}")
    print(f"  Min frequency : {min_frequency}")
    print(f"  Min chars     : {min_chars}")
    print(f"  Special tokens: {len(SPECIAL_TOKENS)}")
    print("=" * 60)

    tokenizer, trainer = _build_tokenizer(vocab_size, min_frequency)

    # Stream training data, collecting extraction stats for the summary below.
    stats = ExtractionStats()
    print("\nStreaming training data...")
    texts = text_iterator(data_dir, min_chars=min_chars, stats=stats)

    print("Training BPE tokenizer (the tokenizers library is Rust — it's fast!)...")
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Print extraction summary here, after the iterator is fully drained,
    # so the counts are complete and appear at a predictable point in the log.
    print(
        f"\nExtraction complete: {stats.usable:,} usable / {stats.total:,} total records"
        f"\n  {stats.too_short:,} skipped (too short)"
        f"\n  {stats.dispatch_empty:,} skipped (schema unrecognised / empty extract)"
        + (f"\n  !! {stats.json_errors:,} malformed JSON lines skipped" if stats.json_errors else "")
    )

    # ------------------------------------------------------------------
    # Post-processor: wrap every encoded sequence with <|bos|>…<|eos|>.
    #
    # Wired up AFTER training because we need the final token IDs.
    #
    # IMPORTANT — spaces in the template are REQUIRED syntax.
    # TemplateProcessing uses spaces as delimiters between pieces in the
    # template DSL. They are NOT inserted as literal space tokens into the
    # encoded sequence — the special tokens are added as atomic units.
    #   "<|bos|> $A <|eos|>"  means: [BOS_ID] + sequence_tokens + [EOS_ID]
    # ------------------------------------------------------------------
    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    tokenizer.post_processor = TemplateProcessing(
        single="<|bos|> $A <|eos|>",
        pair="<|bos|> $A <|eos|> $B:1 <|eos|>:1",
        special_tokens=[
            ("<|bos|>", bos_id),
            ("<|eos|>", eos_id),
        ],
    )

    tokenizer_path = _save_tokenizer(tokenizer, output_dir, vocab_size)
    _run_sanity_checks(tokenizer, vocab_size)

    print(f"\nLoad with tokenizers:")
    print(f"  from tokenizers import Tokenizer")
    print(f"  tok = Tokenizer.from_file(r'{tokenizer_path}')")
    print(f"\nLoad with HuggingFace transformers:")
    print(f"  from transformers import AutoTokenizer")
    print(f"  tok = AutoTokenizer.from_pretrained(r'{output_dir}')")
    vocab_path = output_dir / "vocab.txt"
    print(f"\nInspect token merges (Windows):")
    print(f"  findstr unwrap {vocab_path}")
    print(f"\nInspect token merges (Linux/Mac):")
    print(f"  grep unwrap {vocab_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Rust-aware BPE tokenizer on your SLM dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="C:/llm/data",
        help="Directory containing .jsonl files (searched recursively)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="C:/llm/tokenizer",
        help="Where to save tokenizer.json, tokenizer_config.json, "
             "special_tokens_map.json, and vocab.txt",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help="BPE vocabulary size (includes all special tokens)",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=DEFAULT_MIN_FREQUENCY,
        help="Minimum pair frequency required for a BPE merge to be kept",
    )
    parser.add_argument(
        "--min_chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help="Skip records whose extracted text is shorter than this many characters",
    )
    args = parser.parse_args()

    # All three values are now passed explicitly into train_tokenizer().
    # Previously vocab_size and min_frequency were silently ignored:
    #   - vocab_size was assigned to a local variable instead of the module constant
    #   - min_frequency was parsed but never forwarded to BpeTrainer
    train_tokenizer(
        data_dir      = Path(args.data_dir),
        output_dir    = Path(args.output_dir),
        vocab_size    = args.vocab_size,
        min_frequency = args.min_frequency,
        min_chars     = args.min_chars,
    )