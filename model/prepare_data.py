"""
prepare_data.py
===============
Tokenises the Rust SLM corpus and writes memory-mapped binary files
ready for the training loop.

Imports the schema-dispatch and extraction logic directly from
train_tokenizer.py to avoid duplication — keep both scripts in the
same directory (e.g. C:\\llm\\model\\).

Output layout in --output_dir:
  train.bin    uint16 numpy memmap — flat token-ID stream, training split
  val.bin      uint16 numpy memmap — flat token-ID stream, validation split
  meta.json    vocab size, seq_len, split sizes, token counts, timestamp

Binary format:
  Documents are concatenated as:
      [doc tokens…] <|eos|> [doc tokens…] <|eos|> …
  No padding.  The training loop slices this into (seq_len + 1)-token
  windows and uses window[:-1] as input, window[1:] as target.

Usage:
    python prepare_data.py --data_dir C:/llm/data --tok_dir C:/llm/tokenizer --output_dir C:/llm/data/prepared

Optional flags:
    --seq_len        Context window length (tokens)      (default: 2048)
    --val_fraction   Fraction of docs held out for val   (default: 0.005)
    --seed           Document-shuffle seed                (default: 42)
    --min_chars      Minimum chars per record to keep    (default: 50)
    --encode_batch   Docs per tokeniser batch            (default: 512)

Requirements:
    pip install tokenizers numpy tqdm
    train_tokenizer.py must be importable from the same directory.
"""

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer

# ---------------------------------------------------------------------------
# Import extraction helpers from train_tokenizer.
# Both scripts live in the same directory; train_tokenizer has no side-effects
# at import time — all training is guarded by `if __name__ == "__main__"`.
# ---------------------------------------------------------------------------
try:
    from train_tokenizer import (
        dispatch,
        iter_jsonl_files,
        ExtractionStats,
        DEFAULT_MIN_CHARS,
    )
except ImportError as exc:
    sys.exit(
        f"Cannot import from train_tokenizer.py: {exc}\n"
        "Ensure train_tokenizer.py is in the same directory as prepare_data.py."
    )

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SEQ_LEN      = 2048
DEFAULT_VAL_FRACTION = 0.005   # 0.5 % ≈ 1 600 docs from 320 K corpus
DEFAULT_SEED         = 42
DEFAULT_ENCODE_BATCH = 512


# ---------------------------------------------------------------------------
# Tokeniser loader
# ---------------------------------------------------------------------------
def load_tokenizer(tok_dir: Path) -> tuple[Tokenizer, int, int, int]:
    """
    Load tokenizer.json and return (tokenizer, bos_id, eos_id, vocab_size).
    Uses the raw tokenizers.Tokenizer — faster than the transformers wrapper
    and gives direct access to encode_batch().
    """
    tok_path = tok_dir / "tokenizer.json"
    if not tok_path.exists():
        sys.exit(f"tokenizer.json not found at {tok_path}")

    tokenizer  = Tokenizer.from_file(str(tok_path))
    bos_id     = tokenizer.token_to_id("<|bos|>")
    eos_id     = tokenizer.token_to_id("<|eos|>")
    vocab_size = tokenizer.get_vocab_size()

    print(f"Tokenizer loaded  →  vocab={vocab_size:,}   bos={bos_id}   eos={eos_id}")
    return tokenizer, bos_id, eos_id, vocab_size


# ---------------------------------------------------------------------------
# Tokenisation pass
# ---------------------------------------------------------------------------
def tokenize_corpus(
    data_dir:     Path,
    tokenizer:    Tokenizer,
    bos_id:       int,
    eos_id:       int,
    min_chars:    int,
    encode_batch: int,
) -> tuple[list[np.ndarray], ExtractionStats]:
    """
    Stream every JSONL record, extract text via dispatch(), batch-encode,
    and return a list of uint16 numpy arrays — one per document.

    Each array ends with a single <|eos|> token that acts as the document
    separator in the packed binary stream.  BOS and EOS tokens added by the
    tokenizer's post-processor are stripped before the separator is appended,
    so no document starts or ends with a spurious BOS/EOS pair.

    Batch encoding (encode_batch_size docs at a time) is meaningfully faster
    than encoding one doc at a time because the Rust tokenizers library
    parallelises across the batch.
    """
    bos_eos = {bos_id, eos_id}
    stats   = ExtractionStats()
    docs:   list[np.ndarray] = []

    # Buffers for the current batch
    pending_texts: list[str] = []

    def flush_batch() -> None:
        """Encode pending_texts, filter tokens, append to docs."""
        if not pending_texts:
            return
        encodings = tokenizer.encode_batch(pending_texts)
        for enc in encodings:
            ids = [id_ for id_ in enc.ids if id_ not in bos_eos]
            if ids:
                ids.append(eos_id)   # document separator
                docs.append(np.array(ids, dtype=np.uint16))
        pending_texts.clear()

    print("\nStreaming and tokenising corpus...")
    for record in iter_jsonl_files(data_dir, stats):
        stats.total += 1

        text = dispatch(record)
        if not text:
            stats.dispatch_empty += 1
            continue
        if len(text) < min_chars:
            stats.too_short += 1
            continue

        pending_texts.append(text)
        if len(pending_texts) >= encode_batch:
            flush_batch()

    flush_batch()   # remainder

    print(
        f"\nTokenisation complete:"
        f"\n  {stats.usable:,} usable / {stats.total:,} total records"
        f"\n  {stats.too_short:,} skipped (too short)"
        f"\n  {stats.dispatch_empty:,} skipped (schema unrecognised / empty)"
        + (f"\n  !! {stats.json_errors:,} malformed JSON lines skipped"
           if stats.json_errors else "")
    )
    return docs, stats


# ---------------------------------------------------------------------------
# Split, pack, and save
# ---------------------------------------------------------------------------
def save_split(
    split_name: str,
    split_docs: list[np.ndarray],
    output_dir: Path,
    seq_len:    int,
) -> dict:
    """
    Concatenate doc arrays, trim to a (seq_len + 1)-aligned boundary so
    every training sequence is a complete window, and write a uint16 memmap.

    Returns a dict of stats for meta.json.
    """
    if not split_docs:
        print(f"  WARNING: {split_name} split has no documents — skipping.")
        return {}

    print(f"\nPacking {split_name} split ({len(split_docs):,} docs)...")
    tokens = np.concatenate(split_docs)   # single allocation

    # Trim to exact multiple of (seq_len + 1).
    # +1 because each window is input[seq_len] + target[seq_len] = seq_len+1 tokens.
    chunk_len   = seq_len + 1
    n_complete  = (len(tokens) // chunk_len) * chunk_len
    n_trimmed   = len(tokens) - n_complete
    tokens      = tokens[:n_complete]
    n_sequences = n_complete // chunk_len

    if n_trimmed:
        print(f"  Trimmed {n_trimmed:,} tokens from tail to align to seq_len+1 boundary.")

    bin_path = output_dir / f"{split_name}.bin"
    fp = np.memmap(bin_path, dtype=np.uint16, mode="w+", shape=(len(tokens),))
    fp[:] = tokens
    fp.flush()
    del fp   # close the memmap

    size_mb = bin_path.stat().st_size / 1_048_576
    print(
        f"  {split_name}.bin  →  {len(tokens):,} tokens  "
        f"{n_sequences:,} sequences  {size_mb:.1f} MB"
    )

    return {
        "n_docs":      len(split_docs),
        "n_tokens":    int(len(tokens)),
        "n_sequences": int(n_sequences),
        "size_mb":     round(size_mb, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def prepare_data(
    data_dir:     Path,
    tok_dir:      Path,
    output_dir:   Path,
    seq_len:      int   = DEFAULT_SEQ_LEN,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed:         int   = DEFAULT_SEED,
    min_chars:    int   = DEFAULT_MIN_CHARS,
    encode_batch: int   = DEFAULT_ENCODE_BATCH,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Rust SLM Data Preparation")
    print("=" * 60)
    print(f"  Data dir      : {data_dir}")
    print(f"  Tokenizer dir : {tok_dir}")
    print(f"  Output dir    : {output_dir}")
    print(f"  Seq length    : {seq_len}")
    print(f"  Val fraction  : {val_fraction}")
    print(f"  Shuffle seed  : {seed}")
    print(f"  Min chars     : {min_chars}")
    print(f"  Encode batch  : {encode_batch}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load tokenizer
    # ------------------------------------------------------------------
    tokenizer, bos_id, eos_id, vocab_size = load_tokenizer(tok_dir)

    # ------------------------------------------------------------------
    # 2. Tokenise corpus
    # ------------------------------------------------------------------
    docs, stats = tokenize_corpus(
        data_dir, tokenizer, bos_id, eos_id, min_chars, encode_batch
    )

    total_raw_tokens = sum(len(d) for d in docs)
    print(f"\n  {len(docs):,} documents  →  {total_raw_tokens:,} raw tokens before splitting")

    if not docs:
        sys.exit("No documents produced — check data_dir and min_chars.")

    # ------------------------------------------------------------------
    # 3. Shuffle at document level, then split
    # ------------------------------------------------------------------
    print(f"\nShuffling {len(docs):,} documents (seed={seed})...")
    rng = random.Random(seed)
    rng.shuffle(docs)

    n_val      = max(1, round(len(docs) * val_fraction))
    val_docs   = docs[:n_val]
    train_docs = docs[n_val:]

    print(f"  Train: {len(train_docs):,} docs")
    print(f"  Val  : {len(val_docs):,} docs  ({val_fraction*100:.1f}%)")

    # ------------------------------------------------------------------
    # 4. Pack and save each split
    # ------------------------------------------------------------------
    train_stats = save_split("train", train_docs, output_dir, seq_len)
    val_stats   = save_split("val",   val_docs,   output_dir, seq_len)

    # ------------------------------------------------------------------
    # 5. Write metadata
    # ------------------------------------------------------------------
    meta = {
        "vocab_size":  vocab_size,
        "seq_len":     seq_len,
        "eos_id":      eos_id,
        "bos_id":      bos_id,
        "dtype":       "uint16",
        "seed":        seed,
        "val_fraction": val_fraction,
        "train":       train_stats,
        "val":         val_stats,
        "corpus": {
            "n_docs_total":      len(docs),
            "n_tokens_total":    int(total_raw_tokens),
            "n_json_errors":     stats.json_errors,
            "n_dispatch_empty":  stats.dispatch_empty,
            "n_too_short":       stats.too_short,
        },
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }

    meta_path = output_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved  →  {meta_path}")

    # ------------------------------------------------------------------
    # 6. Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Train sequences : {train_stats.get('n_sequences', 0):>10,}")
    print(f"  Val sequences   : {val_stats.get('n_sequences',   0):>10,}")
    print(f"  Train tokens    : {train_stats.get('n_tokens',    0):>10,}")
    print(f"  Val tokens      : {val_stats.get('n_tokens',      0):>10,}")
    print(f"  Vocab size      : {vocab_size:>10,}")
    print(f"  Sequence length : {seq_len:>10,}")
    print("=" * 60)

    n_train_seq = train_stats.get("n_sequences", 0)
    if n_train_seq > 0:
        print(
            f"\nLoad in training loop:\n"
            f"  import numpy as np\n"
            f"  train = np.memmap(r'{output_dir / 'train.bin'}', "
            f"dtype='uint16', mode='r')\n"
            f"  # Each sequence: train[i*(seq_len+1) : (i+1)*(seq_len+1)]\n"
            f"  # input  = seq[:-1]   shape ({seq_len},)\n"
            f"  # target = seq[1:]    shape ({seq_len},)"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenise and pack the Rust SLM corpus into training binaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="C:/llm/data",
        help="Directory containing .jsonl source files (searched recursively)",
    )
    parser.add_argument(
        "--tok_dir",
        type=str,
        default="C:/llm/tokenizer",
        help="Directory containing tokenizer.json (output of train_tokenizer.py)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="C:/llm/data/prepared",
        help="Where to write train.bin, val.bin, and meta.json",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help="Context window length in tokens — must match model_max_length",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help="Fraction of documents held out for validation (0.005 = 0.5%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for document shuffle before train/val split",
    )
    parser.add_argument(
        "--min_chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help="Skip records whose extracted text is shorter than this many characters",
    )
    parser.add_argument(
        "--encode_batch",
        type=int,
        default=DEFAULT_ENCODE_BATCH,
        help="Number of documents per tokeniser batch (larger = faster, more RAM)",
    )
    args = parser.parse_args()

    prepare_data(
        data_dir     = Path(args.data_dir),
        tok_dir      = Path(args.tok_dir),
        output_dir   = Path(args.output_dir),
        seq_len      = args.seq_len,
        val_fraction = args.val_fraction,
        seed         = args.seed,
        min_chars    = args.min_chars,
        encode_batch = args.encode_batch,
    )