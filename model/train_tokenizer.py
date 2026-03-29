"""
train_tokenizer.py
==================
Trains a BPE tokenizer on your Rust SLM dataset.

Handles all four schema variants:
  - snippets     (strandset-rust-v1)
  - docs/crates  (source: docs | crates.io | github)
  - lectures     (style: lecture | technical)
  - conversations (has 'turns' array)

Usage:
    python train_tokenizer.py --data_dir C:/llm/data --output_dir C:/llm/tokenizer

Requirements:
    pip install tokenizers transformers tqdm beautifulsoup4
"""

import argparse
import json
import re
import os
from pathlib import Path
from typing import Generator
from tqdm import tqdm
from bs4 import BeautifulSoup
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFC


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

VOCAB_SIZE = 32_768  # 32K — sweet spot for a 50M model


# ---------------------------------------------------------------------------
# HTML stripper — handles noisy doc scrapes like the async-book
# ---------------------------------------------------------------------------
def strip_html(text: str) -> str:
    """Remove HTML tags and collapse excessive whitespace."""
    if "<" not in text:
        return text
    try:
        soup = BeautifulSoup(text, "html.parser")
        clean = soup.get_text(separator=" ")
    except Exception:
        # fallback: crude tag strip
        clean = re.sub(r"<[^>]+>", " ", text)
    # collapse whitespace runs
    clean = re.sub(r"\s{3,}", "\n\n", clean)
    return clean.strip()


# ---------------------------------------------------------------------------
# Schema-aware text extractors
# ---------------------------------------------------------------------------
def extract_snippet(record: dict) -> str:
    """
    strandset-rust-v1 snippets.
    Pulls every useful text field that exists on the record.
    """
    parts = []

    # Core code fields
    for field in ("fixed_code", "broken_code", "explanation"):
        val = record.get(field)
        if val and isinstance(val, str) and val.strip():
            parts.append(val.strip())

    # Validation output — clippy explanations are gold for a Rust model
    validation = record.get("validation")
    if isinstance(validation, dict):
        for field in ("clippy_output", "error_message"):
            val = validation.get(field)
            if val and isinstance(val, str) and val.strip():
                parts.append(val.strip())

    # Top-level error fields (older schema variants)
    for field in ("error_message", "error_code"):
        val = record.get(field)
        if val and isinstance(val, str) and val.strip():
            parts.append(val.strip())

    return "\n\n".join(parts)


def extract_content(record: dict) -> str:
    """
    docs / crates.io / github records — text lives in 'content'.
    Strip HTML noise from web-scraped docs.
    """
    content = record.get("content", "")
    if not content or not isinstance(content, str):
        return ""
    return strip_html(content.strip())


def extract_lecture(record: dict) -> str:
    """
    Lecture records — single 'content' string, usually clean.
    """
    content = record.get("content", "")
    if not content or not isinstance(content, str):
        return ""
    return content.strip()


def extract_conversation(record: dict) -> str:
    """
    Multi-turn conversations. Format with role markers so the tokenizer
    learns the turn structure — important for Phase 2 SFT.
    """
    turns = record.get("turns", [])
    if not turns or not isinstance(turns, list):
        return ""

    parts = []
    for turn in turns:
        role = turn.get("role", "user").strip().lower()
        content = turn.get("content", "").strip()
        if not content:
            continue
        # Map role to special token prefix
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
    # --- Conversations: has a 'turns' list ---
    if "turns" in record and isinstance(record.get("turns"), list):
        return extract_conversation(record)

    # --- Lectures: has 'style' field ---
    if "style" in record:
        return extract_lecture(record)

    # --- Snippets: has 'schema_variant' or 'fixed_code' or 'source_model' ---
    source = record.get("source", "")
    schema_variant = record.get("schema_variant", "")
    if (
        schema_variant == "snippet"
        or "fixed_code" in record
        or source == "strandset-rust-v1"
    ):
        return extract_snippet(record)

    # --- Docs / Crates / GitHub: has 'content' and known source values ---
    if source in ("docs", "crates.io", "github") or "content" in record:
        return extract_content(record)

    # --- Fallback: try 'content' then 'text' ---
    for field in ("content", "text"):
        val = record.get(field)
        if val and isinstance(val, str) and val.strip():
            return strip_html(val.strip())

    return ""


# ---------------------------------------------------------------------------
# JSONL reader — streams records without loading all files into RAM
# ---------------------------------------------------------------------------
def iter_jsonl_files(data_dir: Path) -> Generator[dict, None, None]:
    """Recursively find and stream all .jsonl files in data_dir."""
    files = sorted(data_dir.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found under {data_dir}")
    print(f"\nFound {len(files)} JSONL file(s):")
    for f in files:
        print(f"  {f}")
    print()
    for filepath in files:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip malformed lines silently


def text_iterator(data_dir: Path, min_chars: int = 50) -> Generator[str, None, None]:
    """
    Yields extracted text strings for tokenizer training.
    Skips anything shorter than min_chars (noise/empty records).
    """
    total = skipped = 0
    for record in iter_jsonl_files(data_dir):
        total += 1
        text = dispatch(record)
        if len(text) < min_chars:
            skipped += 1
            continue
        yield text

    print(f"\nExtraction complete: {total - skipped:,} usable / {total:,} total records "
          f"({skipped:,} skipped as too short)")


# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------
def train_tokenizer(data_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Rust SLM Tokenizer Trainer")
    print("=" * 60)
    print(f"  Data dir   : {data_dir}")
    print(f"  Output dir : {output_dir}")
    print(f"  Vocab size : {VOCAB_SIZE:,}")
    print(f"  Special tks: {len(SPECIAL_TOKENS)}")
    print("=" * 60)

    # Build tokenizer with BPE model
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))

    # ByteLevel pre-tokenizer — handles ALL unicode safely, no OOV ever
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # NFC normalization — consistent unicode representation
    tokenizer.normalizer = NFC()

    # Trainer config
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,          # token pair must appear at least twice
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
    )

    print("\nStreaming training data...")
    texts = text_iterator(data_dir)

    print("Training BPE tokenizer (this runs in Rust under the hood — fast!)...")
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # -----------------------------------------------------------------------
    # Save in HuggingFace format — works with both tokenizers and transformers
    # -----------------------------------------------------------------------
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"\nTokenizer saved → {tokenizer_path}")

    # Also save a tokenizer_config.json so transformers.AutoTokenizer can load it
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "unk_token": "<|unk|>",
        "pad_token": "<|pad|>",
        "model_max_length": 2048,
        "special_tokens_map_file": None,
    }
    config_path = output_dir / "tokenizer_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved    → {config_path}")

    # -----------------------------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SANITY CHECKS")
    print("=" * 60)

    # 1. Special token IDs
    print("\nSpecial token IDs:")
    for tok in SPECIAL_TOKENS:
        tok_id = tokenizer.token_to_id(tok)
        print(f"  {tok:<20} → {tok_id}")

    # 2. Encode / decode round trip
    test_cases = [
        "fn main() {\n    println!(\"Hello, Rust!\");\n}",
        "impl<T: Clone> Iterator for MyStruct<T> { type Item = T; }",
        "use std::sync::{Arc, Mutex};\nuse tokio::runtime::Runtime;",
        "<|user|>How do lifetimes work in Rust?<|endturn|>\n<|assistant|>Lifetimes ensure references are valid.<|endturn|>",
    ]

    print("\nRound-trip encode → decode tests:")
    all_passed = True
    for i, text in enumerate(test_cases):
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        ok = decoded == text
        all_passed = all_passed and ok
        token_count = len(encoded.ids)
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  [{status}] Test {i+1}: {token_count} tokens | {repr(text[:50])}")
        if not ok:
            print(f"         Original : {repr(text)}")
            print(f"         Decoded  : {repr(decoded)}")

    # 3. Rust idiom tokenization efficiency
    print("\nRust idiom token efficiency (fewer = better):")
    idioms = [
        "unwrap_or_else",
        "Vec<String>",
        "impl Trait",
        "Result<T, E>",
        "Arc<Mutex<T>>",
        "async fn",
        "#[derive(Debug, Clone)]",
        "tokio::spawn",
    ]
    for idiom in idioms:
        ids = tokenizer.encode(idiom).ids
        tokens = tokenizer.encode(idiom).tokens
        print(f"  {idiom:<30} → {len(ids)} token(s): {tokens}")

    # 4. Vocab stats
    vocab = tokenizer.get_vocab()
    print(f"\nVocabulary stats:")
    print(f"  Final vocab size : {len(vocab):,} / {VOCAB_SIZE:,}")
    print(f"  Special tokens   : {len(SPECIAL_TOKENS)}")

    print("\n" + "=" * 60)
    if all_passed:
        print("  All round-trip tests PASSED — tokenizer is healthy!")
    else:
        print("  WARNING: Some round-trip tests FAILED — check output above")
    print("=" * 60)
    print(f"\nDone! Load your tokenizer with:")
    print(f"  from tokenizers import Tokenizer")
    print(f"  tok = Tokenizer.from_file(r'{tokenizer_path}')")
    print(f"\nOr with HuggingFace transformers:")
    print(f"  from transformers import AutoTokenizer")
    print(f"  tok = AutoTokenizer.from_pretrained(r'{output_dir}')")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on Rust SLM dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="C:/llm/data",
        help="Directory containing your .jsonl files (searched recursively)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="C:/llm/tokenizer",
        help="Where to save the trained tokenizer",
    )
    args = parser.parse_args()

    train_tokenizer(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
    )