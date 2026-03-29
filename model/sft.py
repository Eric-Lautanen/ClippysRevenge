"""
sft.py
======
Supervised fine-tuning (continued pretraining) for the Rust SLM.

Since the SFT corpus uses the same JSONL schema as pretraining, this script
runs full-sequence language modelling loss (no prompt masking) at a lower
learning rate on a fresh pass over the data.  This biases the model toward
the target domain distribution without catastrophic forgetting of pretraining.

Differences from train.py:
  - Loads weights from a pretrained checkpoint (--base_ckpt)
  - Optimizer state is reset (new LR regime — don't carry over Adam moments)
  - Peak LR 3e-5  (10× lower than pretraining)
  - Warmup 100 steps  (shorter — weights already converged)
  - Default 10,000 steps  (≈ 2.6B tokens, ~3 passes over the SFT corpus)

Usage:
    python sft.py \\
        --data_dir  C:/llm/data/prepared \\
        --base_ckpt C:/llm/runs/run1/ckpt_best.pt \\
        --out_dir   C:/llm/runs/sft1

Resume a stopped SFT run:
    python sft.py ... --resume

Key flags:
    --base_ckpt      Pretrained checkpoint to start from   (required)
    --micro_batch    Sequences per GPU step                (default: 1)
    --grad_accum     Accumulation steps                    (default: 256)
    --max_steps      SFT training steps                    (default: 10000)
    --peak_lr        Peak learning rate                    (default: 3e-5)
    --warmup_steps   LR warmup steps                       (default: 100)
    --val_every      Validate every N steps                (default: 250)
    --save_every     Save periodic checkpoint every N      (default: 500)

Requirements:
    pip install torch numpy tqdm
    model.py and train.py must be in the same directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

# Reuse all data loading, LR schedule, checkpointing, and eval from train.py
from train import (
    BinaryDataset,
    InfiniteLoader,
    get_lr,
    save_checkpoint,
    load_checkpoint,
    evaluate,
    DEFAULT_GRAD_CLIP,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_BETA1,
    DEFAULT_BETA2,
)
from model import RustSLM, ModelConfig, count_parameters

import json
import time
import numpy as np


# ---------------------------------------------------------------------------
# SFT-specific defaults (everything else inherited from train.py)
# ---------------------------------------------------------------------------
SFT_MICRO_BATCH   = 1
SFT_GRAD_ACCUM    = 256     # eff. batch = 256 sequences — same as pretraining
SFT_MAX_STEPS     = 10_000  # ≈ 2.6B tokens at eff. batch 256 × seq_len 2048
SFT_PEAK_LR       = 3e-5   # 10× lower than pretraining
SFT_MIN_LR_RATIO  = 0.1
SFT_WARMUP_STEPS  = 100
SFT_VAL_EVERY     = 250
SFT_SAVE_EVERY    = 500


# ---------------------------------------------------------------------------
# SFT loop
# ---------------------------------------------------------------------------

def sft(
    data_dir:     Path,
    base_ckpt:    Path,
    out_dir:      Path,
    micro_batch:  int   = SFT_MICRO_BATCH,
    grad_accum:   int   = SFT_GRAD_ACCUM,
    max_steps:    int   = SFT_MAX_STEPS,
    peak_lr:      float = SFT_PEAK_LR,
    warmup_steps: int   = SFT_WARMUP_STEPS,
    val_every:    int   = SFT_VAL_EVERY,
    save_every:   int   = SFT_SAVE_EVERY,
    grad_clip:    float = DEFAULT_GRAD_CLIP,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    resume:       bool  = False,
    seed:         int   = 42,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    torch.manual_seed(seed)

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found at {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)

    vocab_size = meta["vocab_size"]
    seq_len    = meta["seq_len"]
    eos_id     = meta["eos_id"]
    print(f"\nMeta: vocab={vocab_size:,}  seq_len={seq_len}  eos={eos_id}")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    train_ds = BinaryDataset(data_dir / "train.bin", seq_len, device)
    val_ds   = BinaryDataset(data_dir / "val.bin",   seq_len, device)
    print(f"Train: {len(train_ds):,} sequences   Val: {len(val_ds):,} sequences")

    loader   = InfiniteLoader(train_ds, micro_batch, seed=seed)
    eff_batch = micro_batch * grad_accum

    print(f"\nBatch config:")
    print(f"  micro_batch  : {micro_batch}")
    print(f"  grad_accum   : {grad_accum}")
    print(f"  eff_batch    : {eff_batch} sequences = {eff_batch * seq_len:,} tokens")
    print(f"  max_steps    : {max_steps:,}")
    print(f"  Total tokens : {max_steps * eff_batch * seq_len / 1e9:.2f}B")

    # ------------------------------------------------------------------
    # Model — build from config then load pretrained weights
    # ------------------------------------------------------------------
    cfg = ModelConfig(
        vocab_size  = vocab_size,
        max_seq_len = seq_len,
        use_gradient_checkpointing = True,
    )
    model = RustSLM(cfg).to(device)

    if not base_ckpt.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {base_ckpt}")

    print(f"\nLoading pretrained weights from {base_ckpt}...")
    ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    pretrain_step = ckpt.get("step", "?")
    pretrain_val  = ckpt.get("best_val_loss", float("inf"))
    print(f"  Pretrained at step {pretrain_step}  val_loss={pretrain_val:.4f}")

    model.train()
    print(f"  {count_parameters(model) / 1e6:.1f}M parameters")

    # ------------------------------------------------------------------
    # Optimizer — fresh state (don't inherit pretrain Adam moments)
    # ------------------------------------------------------------------
    decay_params   = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() <  2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,   "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr    = peak_lr,
        betas = (DEFAULT_BETA1, DEFAULT_BETA2),
        eps   = 1e-8,
        fused = True if device.type == "cuda" else False,
    )

    min_lr     = peak_lr * SFT_MIN_LR_RATIO
    start_step = 0
    best_val   = float("inf")

    # ------------------------------------------------------------------
    # Resume SFT checkpoint (optimizer state preserved across SFT steps)
    # ------------------------------------------------------------------
    latest_ckpt = out_dir / "ckpt_latest.pt"
    if resume and latest_ckpt.exists():
        print(f"\nResuming SFT from {latest_ckpt}...")
        start_step, best_val = load_checkpoint(latest_ckpt, model, optimizer)
        loader.pos = (start_step * grad_accum * micro_batch) % len(loader.order)
    elif resume:
        print("  --resume set but no SFT checkpoint found — starting from base weights.")

    # ------------------------------------------------------------------
    # Training loop  (identical structure to train.py)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  SFT from step {start_step:,} to {max_steps:,}  (lr={peak_lr:.0e})")
    print(f"{'=' * 60}\n")

    running_loss = 0.0
    t0           = time.perf_counter()

    for step in range(start_step, max_steps):
        lr = get_lr(step, warmup_steps, max_steps, peak_lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(grad_accum):
            x, y = loader.next_batch()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, targets=y)
                loss    = loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        running_loss += accum_loss

        if (step + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            avg_loss = running_loss / 10
            tok_s    = (10 * eff_batch * seq_len) / elapsed
            print(
                f"step {step+1:>6,} | loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | grad_norm {grad_norm:.3f} | "
                f"{tok_s/1000:.1f}K tok/s"
            )
            running_loss = 0.0
            t0           = time.perf_counter()

        if (step + 1) % val_every == 0:
            val_loss  = evaluate(model, val_ds, micro_batch)
            improved  = val_loss < best_val
            marker    = " ★ best" if improved else ""
            print(f"\n  [val] step {step+1:,}  val_loss={val_loss:.4f}{marker}\n")
            if improved:
                best_val = val_loss
                save_checkpoint(
                    out_dir / "ckpt_best.pt",
                    model, optimizer, step + 1, best_val, cfg.__dict__,
                )

        if (step + 1) % save_every == 0:
            ckpt_path = out_dir / f"ckpt_{step+1:07d}.pt"
            save_checkpoint(ckpt_path, model, optimizer, step + 1, best_val, cfg.__dict__)
            save_checkpoint(latest_ckpt, model, optimizer, step + 1, best_val, cfg.__dict__)
            print(f"  [ckpt] saved {ckpt_path.name}")

    final_path = out_dir / "ckpt_final.pt"
    save_checkpoint(final_path, model, optimizer, max_steps, best_val, cfg.__dict__)
    print(f"\nSFT complete. Final checkpoint: {final_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune the Rust SLM from a pretrained checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir",     type=str, required=True,
                        help="Directory containing train.bin, val.bin, meta.json")
    parser.add_argument("--base_ckpt",    type=str, required=True,
                        help="Pretrained checkpoint to start from (e.g. ckpt_best.pt)")
    parser.add_argument("--out_dir",      type=str, default="C:/llm/runs/sft1",
                        help="Output directory for SFT checkpoints")
    parser.add_argument("--micro_batch",  type=int,   default=SFT_MICRO_BATCH)
    parser.add_argument("--grad_accum",   type=int,   default=SFT_GRAD_ACCUM)
    parser.add_argument("--max_steps",    type=int,   default=SFT_MAX_STEPS)
    parser.add_argument("--peak_lr",      type=float, default=SFT_PEAK_LR)
    parser.add_argument("--warmup_steps", type=int,   default=SFT_WARMUP_STEPS)
    parser.add_argument("--val_every",    type=int,   default=SFT_VAL_EVERY)
    parser.add_argument("--save_every",   type=int,   default=SFT_SAVE_EVERY)
    parser.add_argument("--grad_clip",    type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--resume",       action="store_true",
                        help="Resume from ckpt_latest.pt in out_dir")
    args = parser.parse_args()

    sft(
        data_dir     = Path(args.data_dir),
        base_ckpt    = Path(args.base_ckpt),
        out_dir      = Path(args.out_dir),
        micro_batch  = args.micro_batch,
        grad_accum   = args.grad_accum,
        max_steps    = args.max_steps,
        peak_lr      = args.peak_lr,
        warmup_steps = args.warmup_steps,
        val_every    = args.val_every,
        save_every   = args.save_every,
        grad_clip    = args.grad_clip,
        weight_decay = args.weight_decay,
        resume       = args.resume,
        seed         = args.seed,
    )