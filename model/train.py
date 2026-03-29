"""
train.py
========
Pretraining loop for the Rust SLM (~504M parameters).

Features:
  BF16 mixed precision (torch.autocast), gradient checkpointing (in model),
  gradient accumulation, AdamW with cosine LR + linear warmup,
  gradient clipping, resumable checkpoints, periodic validation.

Checkpoints:
  ckpt_every_N.pt   — saved every --save_every steps (default 500)
  ckpt_best.pt      — overwritten whenever val loss improves
  Both contain full training state so training can resume exactly.

Usage:
    # Fresh run
    python train.py --data_dir C:/llm/data/prepared --out_dir C:/llm/runs/run1

    # Resume from checkpoint
    python train.py --data_dir C:/llm/data/prepared --out_dir C:/llm/runs/run1 --resume

Key flags:
    --micro_batch    Sequences per GPU step           (default: 2)
    --grad_accum     Accumulation steps               (default: 128)  → eff. batch 256
    --max_steps      Training steps                   (default: 160000 ≈ 1 epoch)
    --peak_lr        Peak learning rate               (default: 3e-4)
    --warmup_steps   LR warmup steps                  (default: 500)
    --val_every      Validate every N steps           (default: 250)
    --save_every     Save periodic checkpoint every N (default: 500)
    --grad_clip      Gradient norm clip               (default: 1.0)

Requirements:
    pip install torch numpy tqdm
    model.py must be in the same directory.
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

# Enable experimental ROCm attention kernels before importing model
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

from model import RustSLM, ModelConfig, count_parameters


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MICRO_BATCH   = 2
DEFAULT_GRAD_ACCUM    = 128      # effective batch = micro_batch × grad_accum = 256
DEFAULT_MAX_STEPS     = 160_000  # ≈ 1 epoch over 160,779 sequences at eff. batch 256
DEFAULT_PEAK_LR       = 3e-4
DEFAULT_MIN_LR_RATIO  = 0.1      # min_lr = peak_lr × this
DEFAULT_WARMUP_STEPS  = 500
DEFAULT_VAL_EVERY     = 250
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
    Wraps a uint16 numpy memmap and yields (input, target) token tensors.

    The bin file is a flat sequence of token IDs packed as:
        [doc tokens…] <eos> [doc tokens…] <eos> …
    trimmed to an exact multiple of (seq_len + 1).

    Each "sequence" is a (seq_len + 1)-token window:
        input  = window[:-1]   shape (seq_len,)
        target = window[1:]    shape (seq_len,)
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
        """
        Load a batch of sequences by index.
        Returns (input, target) both of shape (B, seq_len) on self.device.
        """
        chunks = [self.data[i * self.chunk : (i + 1) * self.chunk] for i in indices]
        arr    = np.stack(chunks)                                  # (B, seq_len+1)
        t      = torch.from_numpy(arr.astype(np.int64))           # int64 for embedding
        t      = t.to(self.device, non_blocking=True)
        return t[:, :-1], t[:, 1:]                                # input, target


class InfiniteLoader:
    """
    Shuffles sequence indices each epoch and yields micro-batches forever.
    Tracks the global step count for resuming.
    """
    def __init__(
        self,
        dataset:     BinaryDataset,
        micro_batch: int,
        seed:        int = 42,
    ) -> None:
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
# LR schedule: linear warmup → cosine decay → min_lr floor
# ---------------------------------------------------------------------------

def get_lr(
    step:         int,
    warmup_steps: int,
    max_steps:    int,
    peak_lr:      float,
    min_lr:       float,
) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff    = 0.5 * (1.0 + np.cos(np.pi * progress))
    return min_lr + coeff * (peak_lr - min_lr)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path:      Path,
    model:     RustSLM,
    optimizer: torch.optim.Optimizer,
    step:      int,
    best_val:  float,
    cfg_dict:  dict,
) -> None:
    torch.save(
        {
            "step":            step,
            "best_val_loss":   best_val,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config":    cfg_dict,
        },
        path,
    )


def load_checkpoint(
    path:      Path,
    model:     RustSLM,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, float]:
    """Returns (step, best_val_loss)."""
    ckpt      = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    step     = ckpt["step"]
    best_val = ckpt.get("best_val_loss", float("inf"))
    print(f"  Resumed from step {step:,}   best_val={best_val:.4f}")
    return step, best_val


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:      RustSLM,
    val_ds:     BinaryDataset,
    micro_batch: int,
    max_batches: int = 20,
) -> float:
    """
    Evaluate on up to max_batches random val batches.
    Returns mean cross-entropy loss.
    """
    model.eval()
    rng     = np.random.default_rng(0)
    indices = rng.permutation(len(val_ds)).tolist()
    total, n = 0.0, 0

    for start in range(0, min(len(indices), max_batches * micro_batch), micro_batch):
        batch_idx = indices[start : start + micro_batch]
        if len(batch_idx) == 0:
            break
        x, y      = val_ds.get_batch(batch_idx)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, targets=y)
        total += loss.item()
        n     += 1

    model.train()
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Main training loop
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")
        print(f"  VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    torch.manual_seed(seed)

    # ------------------------------------------------------------------
    # Load meta.json
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

    loader = InfiniteLoader(train_ds, micro_batch, seed=seed)

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
        use_gradient_checkpointing = True,
    )
    model = RustSLM(cfg).to(device)
    model.train()
    print(f"\nModel: {count_parameters(model) / 1e6:.1f}M parameters")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    # Separate weight decay: apply to 2-D params (weight matrices) only,
    # skip 1-D params (norms, biases, embeddings).
    decay_params   = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() <  2]
    optim_groups   = [
        {"params": decay_params,   "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=peak_lr,
        betas=(DEFAULT_BETA1, DEFAULT_BETA2),
        eps=1e-8,
        fused=True if device.type == "cuda" else False,
    )

    min_lr   = peak_lr * DEFAULT_MIN_LR_RATIO
    start_step = 0
    best_val   = float("inf")

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    latest_ckpt = out_dir / "ckpt_latest.pt"
    if resume and latest_ckpt.exists():
        print(f"\nResuming from {latest_ckpt}...")
        start_step, best_val = load_checkpoint(latest_ckpt, model, optimizer)
        # Fast-forward the loader past already-seen data
        steps_to_skip = start_step * grad_accum
        loader.pos    = (steps_to_skip * micro_batch) % len(loader.order)
    elif resume:
        print("  --resume set but no checkpoint found — starting fresh.")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Training from step {start_step:,} to {max_steps:,}")
    print(f"{'=' * 60}\n")

    scaler = None   # BF16 doesn't need a GradScaler (unlike FP16)

    running_loss = 0.0
    t0           = time.perf_counter()

    for step in range(start_step, max_steps):
        # ---------- LR update ----------------------------------------
        lr = get_lr(step, warmup_steps, max_steps, peak_lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ---------- Gradient accumulation ----------------------------
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(grad_accum):
            x, y = loader.next_batch()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, targets=y)
                # Divide by grad_accum so the accumulated gradient is the
                # mean over the effective batch, not the sum.
                loss = loss / grad_accum

            loss.backward()
            accum_loss += loss.item()

        # ---------- Gradient clip + optimizer step -------------------
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += accum_loss   # accum_loss is already mean over eff. batch

        # ---------- Logging ------------------------------------------
        if (step + 1) % 10 == 0:
            elapsed  = time.perf_counter() - t0
            avg_loss = running_loss / 10
            tokens_per_sec = (10 * eff_batch * seq_len) / elapsed
            print(
                f"step {step+1:>7,} | loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | grad_norm {grad_norm:.3f} | "
                f"{tokens_per_sec/1000:.1f}K tok/s"
            )
            running_loss = 0.0
            t0           = time.perf_counter()

        # ---------- Validation ---------------------------------------
        if (step + 1) % val_every == 0:
            val_loss = evaluate(model, val_ds, micro_batch)
            improved = val_loss < best_val
            marker   = " ★ best" if improved else ""
            print(f"\n  [val] step {step+1:,}  val_loss={val_loss:.4f}{marker}\n")

            if improved:
                best_val = val_loss
                save_checkpoint(
                    out_dir / "ckpt_best.pt",
                    model, optimizer, step + 1, best_val,
                    cfg.__dict__,
                )

        # ---------- Periodic save ------------------------------------
        if (step + 1) % save_every == 0:
            ckpt_path = out_dir / f"ckpt_{step+1:07d}.pt"
            save_checkpoint(ckpt_path, model, optimizer, step + 1, best_val, cfg.__dict__)
            # Overwrite latest so --resume always finds it
            save_checkpoint(latest_ckpt, model, optimizer, step + 1, best_val, cfg.__dict__)
            print(f"  [ckpt] saved {ckpt_path.name}")

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    final_path = out_dir / "ckpt_final.pt"
    save_checkpoint(final_path, model, optimizer, max_steps, best_val, cfg.__dict__)
    print(f"\nTraining complete. Final checkpoint: {final_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrain the Rust SLM on packed binary data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir",     type=str, default="C:/llm/data/prepared",
                        help="Directory containing train.bin, val.bin, meta.json")
    parser.add_argument("--out_dir",      type=str, default="C:/llm/runs/run1",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--micro_batch",  type=int, default=DEFAULT_MICRO_BATCH)
    parser.add_argument("--grad_accum",   type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--max_steps",    type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--peak_lr",      type=float, default=DEFAULT_PEAK_LR)
    parser.add_argument("--warmup_steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--val_every",    type=int, default=DEFAULT_VAL_EVERY)
    parser.add_argument("--save_every",   type=int, default=DEFAULT_SAVE_EVERY)
    parser.add_argument("--grad_clip",    type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--resume",       action="store_true",
                        help="Resume from ckpt_latest.pt in out_dir")
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