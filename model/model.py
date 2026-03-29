"""
model.py
========
Decoder-only transformer for the Rust SLM project (~504M parameters).

Architecture:
  36 transformer layers, d_model=1024, n_heads=16, head_dim=64
  SwiGLU FFN (hidden_dim=2816), RMSNorm (pre-norm), RoPE positional embeddings
  Tied input/output embeddings, F.scaled_dot_product_attention (ROCm-safe)
  Gradient checkpointing via torch.utils.checkpoint

Approximate parameter breakdown:
  Embedding      :  41.9M  (40960 × 1024)
  Attention      : 150.7M  (4 × 1024² × 36)
  FFN (SwiGLU)   : 311.6M  (3 × 1024 × 2816 × 36)
  Norms + misc   :   ~0.3M
  Total          : ~504M   (lm_head tied to embedding — no extra params)

ROCm / gfx1103 notes:
  - No flash-attn: flash_attn only supports gfx90a/940 datacenter cards.
    F.scaled_dot_product_attention works correctly through ROCm.
  - No torch.compile(): Triton backend has patchy gfx1103 support.
  - BF16 is natively supported on RDNA3 — use that for mixed precision.

Usage:
    from model import RustSLM, ModelConfig
    cfg   = ModelConfig()
    model = RustSLM(cfg).to("cuda")
    logits, loss = model(input_ids, targets=targets)

    # Parameter count
    python model.py   # runs smoke test + prints param count

Requirements:
    pip install torch   (PyTorch ≥ 2.1 for F.scaled_dot_product_attention)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

# Enable experimental ROCm attention kernels (Flash + Mem-Efficient) on gfx1103.
# Safe to set unconditionally — ignored on non-ROCm builds.
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    # Vocabulary — must match tokenizer vocab_size in meta.json
    vocab_size:  int   = 40_960

    # Architecture
    d_model:     int   = 1_024   # embedding / residual stream width
    n_heads:     int   = 16      # attention heads  →  head_dim = d_model // n_heads = 64
    n_layers:    int   = 36      # transformer blocks  →  ~504M total params
    ffn_hidden:  int   = 2_816   # SwiGLU intermediate dim  (≈ 8/3 × d_model, rounded to 128×)
    max_seq_len: int   = 2_048   # must match seq_len used in prepare_data.py

    # Regularisation (0 for pretraining; small value e.g. 0.1 for SFT)
    dropout:     float = 0.0

    # Rotary position embedding base frequency
    rope_theta:  float = 10_000.0

    # Training
    use_gradient_checkpointing: bool = True

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, \
            f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
        return self.d_model // self.n_heads


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10_000.0,
) -> torch.Tensor:
    """
    Precompute complex RoPE frequency tensor.

    Returns:
        freqs: complex64 tensor of shape (max_seq_len, head_dim // 2)

    Stored as a non-trainable buffer on the model so it automatically
    moves with .to(device) / .to(dtype).
    """
    half  = head_dim // 2
    inv_f = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t     = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_f)                        # (T, half)
    return torch.polar(torch.ones_like(freqs), freqs)    # complex64  (T, half)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to a query or key tensor.

    Args:
        x:      (B, T, n_heads, head_dim)   real tensor (BF16 / FP32)
        freqs:  (max_T, head_dim // 2)      complex64 tensor

    Returns:
        Tensor of same shape and dtype as x with RoPE applied.
    """
    T = x.shape[1]
    # Cast to float32 before view_as_complex (complex ops need float)
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Broadcast freqs over batch and head dims: (1, T, 1, head_dim//2)
    f  = freqs[:T].unsqueeze(0).unsqueeze(2)
    # Rotate, convert back to real, restore original dtype
    return torch.view_as_real(xc * f).flatten(-2).to(x.dtype)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalisation.
    Simpler than LayerNorm (no mean subtraction), same empirical quality.
    """
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in FP32, return in input dtype
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.scale


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward block.

        FFN(x) = dropout( (SiLU(x W_gate) ⊙ x W_up) W_down )

    Three bias-free linear layers matching LLaMA / Mistral convention.
    """
    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up   = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden,  d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Attention(nn.Module):
    """
    Multi-head self-attention with RoPE.

    Uses F.scaled_dot_product_attention which dispatches to PyTorch's
    built-in memory-efficient attention kernel — ROCm-compatible without
    requiring the flash-attn package.

    Includes an optional KV cache dict for autoregressive inference;
    pass kv_cache=None (default) during training.
    """
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.dropout  = cfg.dropout

        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x:        torch.Tensor,              # (B, T, d_model)
        freqs:    torch.Tensor,              # (max_T, head_dim // 2) complex
        mask:     Optional[torch.Tensor],    # explicit attention mask or None
        kv_cache: Optional[dict] = None,     # {'k': Tensor, 'v': Tensor} inference only
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # Project → (B, T, n_heads, head_dim)
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim)

        # Apply rotary embeddings
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # Append to KV cache if provided (inference path)
        if kv_cache is not None:
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=1)
                v = torch.cat([kv_cache["v"], v], dim=1)
            kv_cache["k"] = k
            kv_cache["v"] = v

        # (B, n_heads, T, head_dim) — layout expected by SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_drop = self.dropout if self.training else 0.0
        # is_causal=True when mask is None: lets SDPA use its built-in
        # causal kernel (faster, no explicit mask allocation needed).
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=attn_drop,
            is_causal=(mask is None),
        )

        # Merge heads → (B, T, d_model) and project
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class TransformerBlock(nn.Module):
    """
    Standard pre-norm transformer block:
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    """
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attn      = Attention(cfg)
        self.norm_ffn  = RMSNorm(cfg.d_model)
        self.ffn       = SwiGLU(cfg.d_model, cfg.ffn_hidden, cfg.dropout)

    def forward(
        self,
        x:        torch.Tensor,
        freqs:    torch.Tensor,
        mask:     Optional[torch.Tensor],
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), freqs, mask, kv_cache)
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class RustSLM(nn.Module):
    """
    Decoder-only language model for Rust SLM pretraining.

    Forward returns (logits, loss):
      logits : (B, T, vocab_size)  — always returned
      loss   : scalar              — only when targets is not None
    """
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embed  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop   = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm   = RMSNorm(cfg.d_model)

        # Tied output projection: shares weights with embedding — zero extra params
        self.lm_head        = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        # RoPE buffer — moves with model.to(device) automatically
        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """
        GPT-2 style weight initialisation.
        Residual projections (wo, FFN down) are scaled by 1/√(2·n_layers)
        so the variance of residual additions stays constant with depth.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        residual_scale = (2 * self.cfg.n_layers) ** -0.5
        for name, param in self.named_parameters():
            if name.endswith(("wo.weight", "down.weight")):
                param.data.mul_(residual_scale)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,                    # (B, T)
        targets:   Optional[torch.Tensor] = None,   # (B, T)  labels for LM loss
        kv_caches: Optional[list[dict]]   = None,   # per-layer KV cache (inference)
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, (
            f"Input length {T} exceeds max_seq_len {self.cfg.max_seq_len}"
        )

        x = self.drop(self.embed(input_ids))   # (B, T, d_model)

        use_ckpt = self.cfg.use_gradient_checkpointing and self.training

        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            if use_ckpt:
                # Recompute activations on backward pass instead of storing them.
                # Trades ~30% extra compute for a ~40% reduction in activation memory.
                x = checkpoint(layer, x, self.rope_freqs, None, cache, use_reentrant=False)
            else:
                x = layer(x, self.rope_freqs, None, cache)

        x      = self.norm(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten for cross-entropy; ignore_index=-1 lets callers mask tokens
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids:   torch.Tensor,      # (1, T_prompt)
        max_new:     int   = 200,
        temperature: float = 1.0,
        top_k:       int   = 50,
        eos_id:      int   = -1,        # stop on this token id; -1 = disabled
    ) -> torch.Tensor:
        """
        Top-k sampler with KV cache for O(n) autoregressive generation.
        Batch size must be 1.
        """
        assert input_ids.shape[0] == 1, "generate() only supports batch_size=1"
        self.eval()
        kv_caches = [{} for _ in range(self.cfg.n_layers)]
        ids = input_ids.clone()

        for _ in range(max_new):
            # First step: feed full prompt; subsequent steps: one new token
            ctx    = ids if not kv_caches[0] else ids[:, -1:]
            logits, _ = self(ctx, kv_caches=kv_caches)
            logits = logits[:, -1, :] / max(temperature, 1e-8)  # (1, vocab)

            if top_k > 0:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < topk_vals[:, -1:]] = float("-inf")

            probs   = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)   # (1, 1)
            ids     = torch.cat([ids, next_id], dim=1)

            if eos_id >= 0 and next_id.item() == eos_id:
                break

        return ids   # (1, T_prompt + n_generated)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module, dtype: torch.dtype = torch.bfloat16) -> float:
    """Estimate model weight size in MB for a given dtype."""
    n     = count_parameters(model)
    bytes_ = n * torch.finfo(dtype).bits // 8
    return bytes_ / 1_048_576


# ---------------------------------------------------------------------------
# Smoke test / parameter count
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg   = ModelConfig()
    model = RustSLM(cfg)
    n     = count_parameters(model)
    size  = model_size_mb(model)

    print(f"RustSLM")
    print(f"  Parameters : {n:,}  ({n / 1e6:.1f}M)")
    print(f"  BF16 size  : {size:.1f} MB")
    print(f"  Layers     : {cfg.n_layers}")
    print(f"  d_model    : {cfg.d_model}")
    print(f"  n_heads    : {cfg.n_heads}")
    print(f"  head_dim   : {cfg.head_dim}")
    print(f"  ffn_hidden : {cfg.ffn_hidden}")
    print(f"  vocab_size : {cfg.vocab_size}")
    print(f"  max_seq_len: {cfg.max_seq_len}")
    print(f"  grad_ckpt  : {cfg.use_gradient_checkpointing}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nSmoke test on {device}...")
    model = model.to(device)

    B, T = 2, 64
    ids  = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    tgt  = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    model.train()
    logits, loss = model(ids, targets=tgt)
    print(f"  Train forward  : logits={tuple(logits.shape)}  loss={loss.item():.4f}")

    loss.backward()
    print(f"  Backward pass  : OK")

    model.eval()
    out = model.generate(ids[:1, :8], max_new=10, eos_id=-1)
    print(f"  Generate       : {tuple(out.shape)}  (prompt=8, new=10)")

    print("\nAll checks passed.")