"""
model.py
========
Decoder-only transformer for the Rust SLM project (~15M parameters).

Architecture:
  6 transformer layers, d_model=256, n_heads=4, head_dim=64
  SwiGLU FFN (hidden_dim=704), RMSNorm (pre-norm), RoPE positional embeddings
  Tied input/output embeddings, F.scaled_dot_product_attention (ROCm-safe)
  Gradient checkpointing via torch.utils.checkpoint

Approximate parameter breakdown:
  Embedding (tied): 10.5M  (40960 × 256)
  Attention       :  1.6M  (4 × 256² × 6 layers)
  FFN (SwiGLU)    :  3.2M  (3 × 256 × 704 × 6 layers)
  Norms + misc    :  ~0.1M
  Total           : ~15.4M (lm_head tied to embedding — no extra params)

ROCm / gfx1103 notes:
  - No flash-attn: only supports gfx90a/940 datacenter cards.
    F.scaled_dot_product_attention works correctly on ROCm.
  - No torch.compile(): Triton backend has patchy gfx1103 support.
  - BF16 natively supported on RDNA3.
  - fused=False in AdamW — fused kernel crashes on gfx1103 ROCm nightlies.
  - expandable_segments not supported on this platform — don't set it.

Usage:
    from model import RustSLM, ModelConfig
    cfg   = ModelConfig()
    model = RustSLM(cfg).to("cuda")
    logits, loss = model(input_ids, targets=targets)

    python model.py    # smoke test + prints param count
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

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
    vocab_size:  int   = 40_960   # must match tokenizer vocab_size in meta.json

    # 15M parameter config
    d_model:     int   = 256      # embedding / residual stream width
    n_heads:     int   = 4        # head_dim = 256 // 4 = 64
    n_layers:    int   = 6
    ffn_hidden:  int   = 704      # SwiGLU intermediate (≈ 8/3 × 256, rounded to 64×)
    max_seq_len: int   = 2_048

    dropout:     float = 0.0
    rope_theta:  float = 10_000.0
    use_gradient_checkpointing: bool = True

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, \
            f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
        return self.d_model // self.n_heads


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> torch.Tensor:
    half  = head_dim // 2
    inv_f = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t     = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_f)
    return torch.polar(torch.ones_like(freqs), freqs)   # complex64 (T, half)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    T  = x.shape[1]
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    f  = freqs[:T].unsqueeze(0).unsqueeze(2)
    return torch.view_as_real(xc * f).flatten(-2).to(x.dtype)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.scale


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up   = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden,  d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Attention(nn.Module):
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
        x:        torch.Tensor,
        freqs:    torch.Tensor,
        mask:     Optional[torch.Tensor],
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        if kv_cache is not None:
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=1)
                v = torch.cat([kv_cache["v"], v], dim=1)
            kv_cache["k"] = k
            kv_cache["v"] = v

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        drop_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v,
              attn_mask=mask, dropout_p=drop_p, is_causal=(mask is None))
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, -1))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attn      = Attention(cfg)
        self.norm_ffn  = RMSNorm(cfg.d_model)
        self.ffn       = SwiGLU(cfg.d_model, cfg.ffn_hidden, cfg.dropout)

    def forward(self, x, freqs, mask, kv_cache=None):
        x = x + self.attn(self.norm_attn(x), freqs, mask, kv_cache)
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class RustSLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg    = cfg
        self.embed  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop   = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm   = RMSNorm(cfg.d_model)

        self.lm_head        = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight   # tied weights

        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta),
        )
        self._init_weights()

    def _init_weights(self) -> None:
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

    def forward(
        self,
        input_ids: torch.Tensor,
        targets:   Optional[torch.Tensor] = None,
        kv_caches: Optional[list[dict]]   = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len
        x = self.drop(self.embed(input_ids))
        use_ckpt = self.cfg.use_gradient_checkpointing and self.training

        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            if use_ckpt:
                x = checkpoint(layer, x, self.rope_freqs, None, cache, use_reentrant=False)
            else:
                x = layer(x, self.rope_freqs, None, cache)

        logits = self.lm_head(self.norm(x))
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new=200, temperature=1.0, top_k=50, eos_id=-1):
        assert input_ids.shape[0] == 1
        self.eval()
        kv_caches = [{} for _ in range(self.cfg.n_layers)]
        ids = input_ids.clone()
        for _ in range(max_new):
            ctx    = ids if not kv_caches[0] else ids[:, -1:]
            logits, _ = self(ctx, kv_caches=kv_caches)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k > 0:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < topk_vals[:, -1:]] = float("-inf")
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            ids     = torch.cat([ids, next_id], dim=1)
            if eos_id >= 0 and next_id.item() == eos_id:
                break
        return ids


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size_mb(model: nn.Module, dtype: torch.dtype = torch.bfloat16) -> float:
    return count_parameters(model) * torch.finfo(dtype).bits // 8 / 1_048_576


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg   = ModelConfig()
    model = RustSLM(cfg)
    n     = count_parameters(model)

    print(f"RustSLM — 15M config")
    print(f"  Parameters  : {n:,}  ({n/1e6:.1f}M)")
    print(f"  BF16 size   : {model_size_mb(model):.1f} MB")
    print(f"  d_model     : {cfg.d_model}  |  n_heads: {cfg.n_heads}  |  head_dim: {cfg.head_dim}")
    print(f"  n_layers    : {cfg.n_layers}  |  ffn_hidden: {cfg.ffn_hidden}")
    print(f"  vocab_size  : {cfg.vocab_size:,}  |  max_seq_len: {cfg.max_seq_len}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nSmoke test on {device}...")
    model = model.to(device)

    B, T = 2, 64
    ids  = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    tgt  = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    model.train()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=dtype):
        logits, loss = model(ids, targets=tgt)
    print(f"  Train forward  : logits={tuple(logits.shape)}  loss={loss.item():.4f}")
    loss.backward()
    print(f"  Backward pass  : OK")

    model.eval()
    out = model.generate(ids[:1, :8], max_new=10)
    print(f"  Generate       : {tuple(out.shape)}")
    print(f"\nAll checks passed.")