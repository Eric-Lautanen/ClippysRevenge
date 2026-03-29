"""
prepare_sft_data.py
===================
Converts structured Rust SLM instruction JSONL files into masked binary
training data for supervised fine-tuning (SFT).

Each JSONL record is formatted into a conversation using the special tokens
already baked into the tokenizer:

    <|user|>
    {prompt}

    <|code|>{broken_code}<|endcode|>
    <|endturn|><|assistant|>
    {explanation}

    <|code|>{fixed_code}<|endcode|>
    <|endturn|><|eos|>

Loss masking: the model only trains on assistant tokens. All tokens up to
and including <|assistant|> are masked out in the target (set to -1,
which cross_entropy ignores). This teaches the model to respond, not to
repeat the prompt.

Output layout in --output_dir:
    sft_train.bin       uint16 numpy memmap — flat token stream, train split
    sft_train_mask.bin  uint8  numpy memmap — parallel loss mask (1=train, 0=ignore)
    sft_val.bin         uint16 — val split tokens
    sft_val_mask.bin    uint8  — val split mask
    sft_meta.json       vocab size, seq_len, split sizes, token counts, timestamp

Records longer than seq_len are truncated from the left of the prompt
(keeping the full assistant response). Records where the assistant response
alone exceeds seq_len are skipped.

Usage:
    python prepare_sft_data.py \\
        --data_dir  C:/llm/data/sft \\
        --tok_dir   C:/llm/tokenizer \\
        --output_dir C:/llm/data/sft_prepared

Optional flags:
    --seq_len        Context window length (default: 2048, must match model)
    --val_fraction   Fraction held out for validation (default: 0.05)
    --seed           Shuffle seed (default: 42)

Requirements:
    pip install tokenizers numpy tqdm
"""

from __future__ import annotations

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
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SEQ_LEN      = 2048
DEFAULT_VAL_FRACTION = 0.05   # 5% — SFT datasets are smaller, want more val coverage
DEFAULT_SEED         = 42


# ---------------------------------------------------------------------------
# Tokeniser loader (mirrors prepare_data.py)
# ---------------------------------------------------------------------------
def load_tokenizer(tok_dir: Path):
    tok_path = tok_dir / "tokenizer.json"
    if not tok_path.exists():
        sys.exit(f"tokenizer.json not found at {tok_path}")

    tokenizer  = Tokenizer.from_file(str(tok_path))
    vocab_size = tokenizer.get_vocab_size()

    # Resolve all special token IDs we'll need
    def tid(name):
        t = tokenizer.token_to_id(name)
        if t is None:
            sys.exit(f"Special token '{name}' not found in tokenizer vocab.")
        return t

    ids = {
        "bos":       tid("<|bos|>"),
        "eos":       tid("<|eos|>"),
        "user":      tid("<|user|>"),
        "assistant": tid("<|assistant|>"),
        "endturn":   tid("<|endturn|>"),
        "code":      tid("<|code|>"),
        "endcode":   tid("<|endcode|>"),
    }

    print(f"Tokenizer loaded  →  vocab={vocab_size:,}")
    print(f"  Special token IDs: { {k: v for k, v in ids.items()} }")
    return tokenizer, ids, vocab_size


# ---------------------------------------------------------------------------
# Record → conversation formatter
# ---------------------------------------------------------------------------
def format_record(record: dict) -> tuple[str, str] | None:
    """
    Returns (prompt_text, response_text) for a record, or None if unusable.

    prompt_text  — everything the user says (will be masked in loss)
    response_text — everything the assistant says (loss computed here)

    Handles two schemas:
      1. Structured: has prompt + broken_code + fixed_code + explanation
      2. Simple: has prompt + response (fallback)
    """
    prompt   = record.get("prompt", "").strip()
    if not prompt:
        return None

    # --- Build prompt side ---
    parts = [prompt]

    broken = record.get("broken_code", "").strip()
    if broken:
        parts.append(f"\nBroken code:\n```rust\n{broken}\n```")

    error_msg = record.get("error_message", "").strip()
    if error_msg:
        # Strip the leading comment markers if present
        error_clean = "\n".join(
            line.lstrip("/ ") for line in error_msg.splitlines()
        ).strip()
        if error_clean:
            parts.append(f"\nCompiler / runtime note:\n{error_clean}")

    prompt_text = "\n".join(parts)

    # --- Build response side ---
    explanation = record.get("explanation", "").strip()
    fixed       = record.get("fixed_code", "").strip()
    response    = record.get("response", "").strip()   # simple schema fallback

    if explanation and fixed:
        response_text = f"{explanation}\n\nFixed:\n```rust\n{fixed}\n```"
    elif fixed:
        response_text = f"```rust\n{fixed}\n```"
    elif explanation:
        response_text = explanation
    elif response:
        response_text = response
    else:
        return None

    # Append concepts / crates as a footer if present (useful context for the model)
    concepts = record.get("concepts", [])
    crates   = record.get("crates", [])
    if concepts:
        response_text += f"\n\nKey concepts: {', '.join(concepts)}"
    if crates:
        response_text += f"\nCrates: {', '.join(crates)}"

    return prompt_text, response_text


# ---------------------------------------------------------------------------
# Tokenise a single conversation into (token_ids, mask)
# ---------------------------------------------------------------------------
def tokenize_conversation(
    prompt_text:   str,
    response_text: str,
    tokenizer:     Tokenizer,
    ids:           dict,
    seq_len:       int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Builds the full token sequence and a parallel loss mask.

    Token layout:
        [user_id] [prompt tokens] [endturn_id] [assistant_id]
        [response tokens] [endturn_id] [eos_id]

    Loss mask:
        0 for everything up to and including assistant_id
        1 for response tokens + endturn + eos

    Returns (tokens uint16, mask uint8) both shape (seq_len+1,)
    padded/truncated to exactly seq_len+1 tokens, or None if the
    response alone is longer than seq_len.
    """
    special = {ids["bos"], ids["eos"]}

    def encode_clean(text: str) -> list[int]:
        """Encode and strip auto-added BOS/EOS from post-processor."""
        return [t for t in tokenizer.encode(text).ids if t not in special]

    prompt_tokens   = encode_clean(prompt_text)
    response_tokens = encode_clean(response_text)

    # Build the assistant half (these tokens have loss=1)
    assistant_half  = response_tokens + [ids["endturn"], ids["eos"]]

    if len(assistant_half) > seq_len:
        return None   # response alone too long — skip

    # Build the user half (these tokens have loss=0)
    user_half = [ids["user"]] + prompt_tokens + [ids["endturn"], ids["assistant"]]

    # Truncate user half from the LEFT if combined exceeds seq_len
    # (keep full response — that's the valuable signal)
    max_user = seq_len - len(assistant_half)
    if len(user_half) > max_user:
        user_half = user_half[-max_user:]   # keep the end (closest to response)

    full_tokens = user_half + assistant_half

    # Masks: 0 for user half, 1 for assistant half
    mask = [0] * len(user_half) + [1] * len(assistant_half)

    # Pad to exactly seq_len + 1  (pad token = 0)
    target_len = seq_len + 1
    if len(full_tokens) < target_len:
        pad_len    = target_len - len(full_tokens)
        full_tokens = full_tokens + [0] * pad_len
        mask       = mask + [0] * pad_len

    # Trim to target_len (shouldn't be needed but be safe)
    full_tokens = full_tokens[:target_len]
    mask        = mask[:target_len]

    return (
        np.array(full_tokens, dtype=np.uint16),
        np.array(mask,        dtype=np.uint8),
    )


# ---------------------------------------------------------------------------
# Scan and convert all JSONL files
# ---------------------------------------------------------------------------
def load_all_records(data_dir: Path) -> list[dict]:
    """Walk data_dir recursively and collect all JSONL records."""
    records = []
    jsonl_files = sorted(data_dir.rglob("*.jsonl"))
    if not jsonl_files:
        sys.exit(f"No .jsonl files found under {data_dir}")

    print(f"Found {len(jsonl_files)} JSONL file(s) under {data_dir}")
    for fpath in tqdm(jsonl_files, desc="Reading files"):
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"  {len(records):,} records loaded")
    return records


# ---------------------------------------------------------------------------
# Save a split to binary files
# ---------------------------------------------------------------------------
def save_split(
    name:       str,
    sequences:  list[np.ndarray],
    masks:      list[np.ndarray],
    output_dir: Path,
) -> dict:
    if not sequences:
        print(f"  WARNING: {name} split is empty — skipping.")
        return {}

    tokens_arr = np.stack(sequences)   # (N, seq_len+1)
    masks_arr  = np.stack(masks)       # (N, seq_len+1)

    flat_tokens = tokens_arr.reshape(-1)
    flat_masks  = masks_arr.reshape(-1)

    n_seq       = len(sequences)
    seq_len_p1  = sequences[0].shape[0]

    tok_path  = output_dir / f"sft_{name}.bin"
    mask_path = output_dir / f"sft_{name}_mask.bin"

    fp_tok  = np.memmap(tok_path,  dtype=np.uint16, mode="w+", shape=(len(flat_tokens),))
    fp_mask = np.memmap(mask_path, dtype=np.uint8,  mode="w+", shape=(len(flat_masks),))
    fp_tok[:]  = flat_tokens;  fp_tok.flush();  del fp_tok
    fp_mask[:] = flat_masks;   fp_mask.flush(); del fp_mask

    tok_mb  = tok_path.stat().st_size  / 1_048_576
    mask_mb = mask_path.stat().st_size / 1_048_576

    # Compute what fraction of tokens are actually trained on
    loss_frac = flat_masks.mean() * 100

    print(
        f"  sft_{name}.bin  →  {n_seq:,} sequences  "
        f"{tok_mb:.1f} MB  ({loss_frac:.1f}% loss tokens)"
    )

    return {
        "n_sequences": n_seq,
        "n_tokens":    int(len(flat_tokens)),
        "loss_fraction": round(float(loss_frac) / 100, 4),
        "size_mb":     round(tok_mb, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def prepare_sft_data(
    data_dir:     Path,
    tok_dir:      Path,
    output_dir:   Path,
    seq_len:      int   = DEFAULT_SEQ_LEN,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed:         int   = DEFAULT_SEED,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Rust SLM SFT Data Preparation")
    print("=" * 60)
    print(f"  Data dir      : {data_dir}")
    print(f"  Tokenizer dir : {tok_dir}")
    print(f"  Output dir    : {output_dir}")
    print(f"  Seq length    : {seq_len}")
    print(f"  Val fraction  : {val_fraction}")
    print("=" * 60)

    tokenizer, ids, vocab_size = load_tokenizer(tok_dir)

    # ------------------------------------------------------------------
    # 1. Load records
    # ------------------------------------------------------------------
    records = load_all_records(data_dir)

    # ------------------------------------------------------------------
    # 2. Format and tokenise
    # ------------------------------------------------------------------
    sequences, masks = [], []
    n_skipped_format = 0
    n_skipped_len    = 0

    print(f"\nFormatting and tokenising {len(records):,} records...")
    for record in tqdm(records, desc="Tokenising"):
        result = format_record(record)
        if result is None:
            n_skipped_format += 1
            continue

        prompt_text, response_text = result
        encoded = tokenize_conversation(
            prompt_text, response_text, tokenizer, ids, seq_len
        )
        if encoded is None:
            n_skipped_len += 1
            continue

        tokens, mask = encoded
        sequences.append(tokens)
        masks.append(mask)

    print(f"\n  {len(sequences):,} sequences produced")
    print(f"  {n_skipped_format:,} skipped (no usable text)")
    print(f"  {n_skipped_len:,} skipped (response > seq_len)")

    if not sequences:
        sys.exit("No sequences produced — check data_dir and JSONL format.")

    # ------------------------------------------------------------------
    # 3. Shuffle and split
    # ------------------------------------------------------------------
    rng     = random.Random(seed)
    indices = list(range(len(sequences)))
    rng.shuffle(indices)

    n_val   = max(1, round(len(indices) * val_fraction))
    val_idx = indices[:n_val]
    trn_idx = indices[n_val:]

    print(f"\nSplit: train={len(trn_idx):,}  val={len(val_idx):,}")

    train_seqs  = [sequences[i] for i in trn_idx]
    train_masks = [masks[i]     for i in trn_idx]
    val_seqs    = [sequences[i] for i in val_idx]
    val_masks   = [masks[i]     for i in val_idx]

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    print()
    train_stats = save_split("train", train_seqs, train_masks, output_dir)
    val_stats   = save_split("val",   val_seqs,   val_masks,   output_dir)

    # ------------------------------------------------------------------
    # 5. Meta
    # ------------------------------------------------------------------
    meta = {
        "vocab_size":    vocab_size,
        "seq_len":       seq_len,
        "eos_id":        ids["eos"],
        "bos_id":        ids["bos"],
        "assistant_id":  ids["assistant"],
        "dtype":         "uint16",
        "mask_dtype":    "uint8",
        "seed":          seed,
        "val_fraction":  val_fraction,
        "train":         train_stats,
        "val":           val_stats,
        "corpus": {
            "n_records_total":   len(records),
            "n_skipped_format":  n_skipped_format,
            "n_skipped_len":     n_skipped_len,
            "n_sequences_total": len(sequences),
        },
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = output_dir / "sft_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved  →  {meta_path}")

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Train sequences : {train_stats.get('n_sequences', 0):>8,}")
    print(f"  Val sequences   : {val_stats.get('n_sequences',   0):>8,}")
    print(f"  Loss fraction   : {train_stats.get('loss_fraction', 0)*100:>7.1f}%  (assistant tokens only)")
    print(f"  Vocab size      : {vocab_size:>8,}")
    print(f"  Sequence length : {seq_len:>8,}")
    print("=" * 60)
    print(
        f"\nRun SFT with:\n"
        f"  python sft.py \\\n"
        f"    --data_dir  {output_dir} \\\n"
        f"    --base_ckpt C:/llm/runs/run1/ckpt_best.pt \\\n"
        f"    --out_dir   C:/llm/runs/sft1"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert structured Rust SLM JSONL into masked SFT binaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir",     type=str, required=True,
                        help="Directory containing structured .jsonl SFT files")
    parser.add_argument("--tok_dir",      type=str, default="C:/llm/tokenizer",
                        help="Directory containing tokenizer.json")
    parser.add_argument("--output_dir",   type=str, default="C:/llm/data/sft_prepared",
                        help="Where to write sft_train.bin, sft_val.bin, etc.")
    parser.add_argument("--seq_len",      type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--val_fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--seed",         type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    prepare_sft_data(
        data_dir     = Path(args.data_dir),
        tok_dir      = Path(args.tok_dir),
        output_dir   = Path(args.output_dir),
        seq_len      = args.seq_len,
        val_fraction = args.val_fraction,
        seed         = args.seed,
    )