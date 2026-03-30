"""
train.py
========
Pretraining loop for the Rust SLM (~15M parameters).

Fixes vs previous version:
  - fused=False in AdamW (fused kernel crashes on gfx1103 ROCm nightlies)
  - Removed expandable_segments env var (not supported on this platform)
  - tqdm progress bar inside gradient accumulation — no more silent hangs
  - Per-step VRAM reporting
  - Tuned defaults for 15M model on 329M token dataset

Features:
  BF16 mixed precision, gradient checkpointing, gradient accumulation,
  AdamW with cosine LR + linear warmup, gradient clipping,
  resumable checkpoints, periodic validation.

Checkpoints saved to out_dir:
  ckpt_NNNNNNN.pt  — every --save_every steps
  ckpt_best.pt     — whenever val loss improves
  ckpt_latest.pt   — always the most recent (for --resume)

Usage:
    # Fresh run
    python train.py --data_dir C:/llm/data/prepared --out_dir C:/llm/runs/run1

    # Resume
    python train.py --data_dir C:/llm/data/prepared --out_dir C:/llm/runs/run1 --resume
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ROCm experimental attention kernels — safe on all platforms, ignored if not ROCm
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
# NOTE: Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on gfx1103
#       — it is not supported and triggers a UserWarning flood.

from model import RustSLM, ModelConfig, count_parameters


# ---------------------------------------------------------------------------
# Defaults — tuned for 15M model on ~329M token dataset, 780M iGPU
# ---------------------------------------------------------------------------
DEFAULT_MICRO_BATCH   = 2        # 2 sequences per GPU step
DEFAULT_GRAD_ACCUM    = 64       # effective batch = 2 × 64 = 128 sequences
DEFAULT_MAX_STEPS     = 12_500   # 128 seq × 2048 tok × 12500 steps ≈ 3.3B tokens (~10 epochs)
DEFAULT_PEAK_LR       = 3e-4
DEFAULT_MIN_LR_RATIO  = 0.1
DEFAULT_WARMUP_STEPS  = 200
DEFAULT_VAL_EVERY     = 100
DEFAULT_SAVE_EVERY    = 500
DEFAULT_GRAD_CLIP     = 1.0
DEFAULT_WEIGHT_DECAY  = 0.1
DEFAULT_BETA1         = 0.9
DEFAULT_BETA2         = 0.95


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

class BinaryDataset:
    """
    Wraps a uint16 numpy memmap of packed token IDs.
    Each sequence is a (seq_len+1)-token window:
        input  = window[:-1]  shape (seq_len,)
        target = window[1:]   shape (seq_len,)
    """
    def __init__(self, bin_path: Path, seq_len: int, device: torch.device) -> None:
        self.seq_len = seq_len
        self.device  = device
        self.data    = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
        self.chunk   = seq_len + 1
        self.n_seq   = len(self.data) // self.chunk
        if self.n_seq == 0:
            raise ValueError(f"No complete sequences in {bin_path}")

    def __len__(self) -> int:
        return self.n_seq

    def get_batch(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        chunks = [self.data[i * self.chunk : (i + 1) * self.chunk] for i in indices]
        arr    = np.stack(chunks)
        t      = torch.from_numpy(arr.astype(np.int64)).to(self.device, non_blocking=True)
        return t[:, :-1], t[:, 1:]


class InfiniteLoader:
    def __init__(self, dataset: BinaryDataset, micro_batch: int, seed: int = 42) -> None:
        self.ds          = dataset
        self.micro_batch = micro_batch
        self.rng         = np.random.default_rng(seed)
        self._reset()

    def _reset(self) -> None:
        self.order = self.rng.permutation(len(self.ds))
        self.pos   = 0

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pos + self.micro_batch > len(self.order):
            self._reset()
        idx       = self.order[self.pos : self.pos + self.micro_batch].tolist()
        self.pos += self.micro_batch
        return self.ds.get_batch(idx)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay → floor
# ---------------------------------------------------------------------------

def get_lr(step, warmup_steps, max_steps, peak_lr, min_lr) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (1.0 + np.cos(np.pi * progress)) * (peak_lr - min_lr)


# ---------------------------------------------------------------------------
# VRAM reporting helper
# ---------------------------------------------------------------------------

def vram_str(device: torch.device) -> str:
    if device.type != "cuda":
        return ""
    used  = torch.cuda.memory_allocated(device) / 1_073_741_824
    total = torch.cuda.get_device_properties(device).total_memory / 1_073_741_824
    return f"VRAM {used:.2f}/{total:.1f}GB"


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, step, best_val, cfg_dict) -> None:
    torch.save({
        "step":            step,
        "best_val_loss":   best_val,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config":    cfg_dict,
    }, path)


def load_checkpoint(path, model, optimizer) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["step"], ckpt["best_val_loss"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: RustSLM, val_ds: BinaryDataset, micro_batch: int) -> float:
    model.eval()
    total_loss = 0.0
    n_batches  = min(50, len(val_ds) // micro_batch)
    indices    = list(range(n_batches * micro_batch))

    for i in range(n_batches):
        idx  = indices[i * micro_batch : (i + 1) * micro_batch]
        x, y = val_ds.get_batch(idx)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, targets=y)
        total_loss += loss.item()

    model.train()
    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_dir:     Path,
    out_dir:      Path,
    micro_batch:  int   = DEFAULT_MICRO_BATCH,
    grad_accum:   int   = DEFAULT_GRAD_ACCUM,
    max_steps:    int   = DEFAULT_MAX_STEPS,
    peak_lr:      float = DEFAULT_PEAK_LR,
    warmup_steps: int   = DEFAULT_WARMUP_STEPS,
    val_every:    int   = DEFAULT_VAL_EVERY,
    save_every:   int   = DEFAULT_SAVE_EVERY,
    grad_clip:    float = DEFAULT_GRAD_CLIP,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    resume:       bool  = False,
    seed:         int   = 42,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props  = torch.cuda.get_device_properties(device)
        vram_gb = props.total_memory / 1_073_741_824
        print(f"Device: cuda")
        print(f"  {props.name}")
        print(f"  VRAM available: {vram_gb:.1f} GB")
    else:
        device = torch.device("cpu")
        print(f"Device: cpu  (no GPU detected — training will be slow)")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {data_dir}")
    with open(meta_path) as f:
        meta = json.load(f)

    vocab_size = meta["vocab_size"]
    seq_len    = meta["seq_len"]
    eos_id     = meta["eos_id"]
    print(f"\nMeta: vocab={vocab_size:,}  seq_len={seq_len}  eos={eos_id}")

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
    total_tokens = max_steps * eff_batch * seq_len
    print(f"  Total tokens : {total_tokens / 1e9:.2f}B")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    cfg = ModelConfig(
        vocab_size  = vocab_size,
        max_seq_len = seq_len,
        use_gradient_checkpointing = False,
    )
    model = RustSLM(cfg).to(device)
    model.train()
    n_params = count_parameters(model)
    print(f"\nModel: {n_params / 1e6:.1f}M parameters")

    # ------------------------------------------------------------------
    # Optimizer
    # NOTE: fused=False — fused AdamW crashes on gfx1103 ROCm nightlies.
    #       The non-fused path is stable and only ~5% slower on iGPU.
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
        fused = False,   # ← critical fix: fused=True crashes on gfx1103
    )

    min_lr     = peak_lr * DEFAULT_MIN_LR_RATIO
    start_step = 0
    best_val   = float("inf")

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    latest_ckpt = out_dir / "ckpt_latest.pt"
    if resume and latest_ckpt.exists():
        print(f"\nResuming from {latest_ckpt}...")
        start_step, best_val = load_checkpoint(latest_ckpt, model, optimizer)
        loader.pos = (start_step * grad_accum * micro_batch) % len(loader.order)
        print(f"  Resumed at step {start_step:,}  best_val={best_val:.4f}")
    elif resume:
        print("  --resume set but no checkpoint found — starting fresh.")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Training: step {start_step:,} → {max_steps:,}")
    print(f"  Logging every step | Val every {val_every} | Save every {save_every}")
    print(f"{'=' * 60}\n")

    running_loss = 0.0
    t_step_start = time.perf_counter()
    tokens_since_log = 0

    for step in range(start_step, max_steps):
        lr = get_lr(step, warmup_steps, max_steps, peak_lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ---- Gradient accumulation with live progress bar ------------
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        pbar = tqdm(
            range(grad_accum),
            desc=f"step {step+1:>6,}/{max_steps:,}",
            leave=False,
            ncols=88,
            unit="μbatch",
        )
        for micro_step in pbar:
            x, y = loader.next_batch()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, targets=y)
                loss    = loss / grad_accum
            loss.backward()
            accum_loss += loss.item()
            pbar.set_postfix(loss=f"{accum_loss * grad_accum / (micro_step + 1):.4f}")

        # ---- Clip + step ---------------------------------------------
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss     += accum_loss
        tokens_since_log += eff_batch * seq_len

        # ---- Per-step log --------------------------------------------
        elapsed          = time.perf_counter() - t_step_start
        tokens_per_sec   = tokens_since_log / elapsed
        ms_per_step      = elapsed * 1000
        vram             = vram_str(device)
        eta_hours        = (max_steps - step - 1) * elapsed / 3600

        print(
            f"step {step+1:>6,}/{max_steps:,} | "
            f"loss {accum_loss:.4f} | "
            f"lr {lr:.2e} | "
            f"grad {grad_norm:.3f} | "
            f"{tokens_per_sec/1000:.1f}K tok/s | "
            f"{ms_per_step:.0f}ms | "
            f"ETA {eta_hours:.1f}h"
            + (f" | {vram}" if vram else "")
        )

        running_loss     = 0.0
        tokens_since_log = 0
        t_step_start     = time.perf_counter()

        # ---- Validation ----------------------------------------------
        if (step + 1) % val_every == 0:
            val_loss = evaluate(model, val_ds, micro_batch)
            improved = val_loss < best_val
            marker   = " ★ NEW BEST" if improved else ""
            print(f"\n{'─'*60}")
            print(f"  [VAL] step {step+1:,}  val_loss={val_loss:.4f}{marker}")
            print(f"{'─'*60}\n")
            if improved:
                best_val = val_loss
                save_checkpoint(out_dir / "ckpt_best.pt",
                                model, optimizer, step+1, best_val, cfg.__dict__)

        # ---- Periodic checkpoint ------------------------------------
        if (step + 1) % save_every == 0:
            ckpt_path = out_dir / f"ckpt_{step+1:07d}.pt"
            save_checkpoint(ckpt_path, model, optimizer, step+1, best_val, cfg.__dict__)
            save_checkpoint(latest_ckpt, model, optimizer, step+1, best_val, cfg.__dict__)
            print(f"  [CKPT] saved → {ckpt_path.name}")

    # ------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------
    final_path = out_dir / "ckpt_final.pt"
    save_checkpoint(final_path, model, optimizer, max_steps, best_val, cfg.__dict__)
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Final checkpoint : {final_path}")
    print(f"  Best val loss    : {best_val:.4f}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrain the Rust SLM (15M) on packed binary data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir",     type=str,   default="C:/llm/data/prepared")
    parser.add_argument("--out_dir",      type=str,   default="C:/llm/runs/run1")
    parser.add_argument("--micro_batch",  type=int,   default=DEFAULT_MICRO_BATCH)
    parser.add_argument("--grad_accum",   type=int,   default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--max_steps",    type=int,   default=DEFAULT_MAX_STEPS)
    parser.add_argument("--peak_lr",      type=float, default=DEFAULT_PEAK_LR)
    parser.add_argument("--warmup_steps", type=int,   default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--val_every",    type=int,   default=DEFAULT_VAL_EVERY)
    parser.add_argument("--save_every",   type=int,   default=DEFAULT_SAVE_EVERY)
    parser.add_argument("--grad_clip",    type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--resume",       action="store_true")
    args = parser.parse_args()

    train(
        data_dir     = Path(args.data_dir),
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