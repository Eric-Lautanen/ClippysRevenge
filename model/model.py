"""
model.py
========
Decoder-only transformer for the Rust SLM project.
Supports ~15M and ~50M configs with optional 1.58-bit ternary weight
quantization (BitLinear), validated against the BitNet b1.58 literature.

Architecture:
  Decoder-only transformer, pre-norm + sub-layer norm (SubLN), SwiGLU FFN,
  RoPE positional embeddings, tied input/output embeddings,
  F.scaled_dot_product_attention (ROCm-safe), gradient checkpointing.

Model configs:
  15M  — d_model=256,  n_heads=4, n_layers=6,  ffn_hidden=704
  50M  — d_model=512,  n_heads=8, n_layers=9,  ffn_hidden=1408

Effective capacity note (IMPORTANT):
  1.58-bit ternary quantization reduces representational capacity.
  Research (Nielsen & Schneider-Kamp, 2024) shows that sub-50M 1.58-bit
  models need approximately DOUBLE the hidden size to match 16-bit
  perplexity.  The 50M config here is intentionally larger than what you
  would need for a 50M full-precision model to compensate.

BitLinear — ternary quantization (BitNet b1.58 standard):
  - Weights quantized to {-scale, 0, +scale} via absmean + round().clamp()
  - NOT binary sign() — the zero value enables feature filtering and
    provides implicit regularisation (42-51% of weights become 0 in practice)
  - Activations quantized per-token to int8 range via absmax scaling
  - Straight-through estimator (STE) for gradient flow through both steps
  - Embedding + lm_head stay full precision (tying keeps them coupled;
    binarising the lookup table hurts quality at sub-100M scale)
  - SubLN: extra RMSNorm after each Attention and FFN output improves
    training stability in the quantized regime (from BitNet b1.58 2B4T)

References:
  BitNet b1.58 original paper: https://arxiv.org/abs/2402.17764
  BitNet b1.58 2B4T report:    https://arxiv.org/abs/2504.12285
  BitNet b1.58 Reloaded (sub-50M): https://arxiv.org/abs/2407.09527

Training tips (from research, important for this model size):
  - weight_decay: use 0.05, NOT 0.1 — high weight decay at high LR distorts
    1-bit training by reducing confidence in shadow weights too aggressively.
    Consider zeroing weight_decay for the second half of training.
  - learning_rate: do NOT raise LR for small 1-bit models — research shows
    SLMs do NOT benefit from higher LRs the way large 1-bit LLMs do.
    Keep peak_lr at 3e-4 as you have it.
  - The loss curve in 1-bit training is S-shaped (slow start, fast middle,
    slow convergence) — this is expected, not a bug.

ROCm / gfx1103 notes:
  - No flash-attn: only supports gfx90a/940 datacenter cards.
    F.scaled_dot_product_attention works correctly on ROCm.
  - No torch.compile(): Triton backend has patchy gfx1103 support.
  - BF16 natively supported on RDNA3.
  - fused=False in AdamW — fused kernel crashes on gfx1103 ROCm nightlies.
  - expandable_segments not supported on this platform — don't set it.
  - For 50M on iGPU: use micro_batch=1, grad_accum=128,
    use_gradient_checkpointing=True.

Usage:
    # 50M 1.58-bit model (default)
    from model import RustSLM, ModelConfig
    cfg   = ModelConfig()
    model = RustSLM(cfg).to("cuda")
    logits, loss = model(input_ids, targets=targets)

    # 15M full-precision (original)
    cfg = ModelConfig(d_model=256, n_heads=4, n_layers=6, ffn_hidden=704,
                      use_bitlinear=False, use_subln=False)

    python model.py    # smoke test — runs all four combos, prints param counts
"""

from __future__ import annotations

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

    # 50M parameter config (default)
    # For 15M: d_model=256, n_heads=4, n_layers=6, ffn_hidden=704
    d_model:     int   = 512      # embedding / residual stream width
    n_heads:     int   = 8        # head_dim = 512 // 8 = 64
    n_layers:    int   = 9
    ffn_hidden:  int   = 1408     # SwiGLU intermediate (~8/3 x 512, rounded to 64x)
    max_seq_len: int   = 2_048

    dropout:     float = 0.0
    rope_theta:  float = 10_000.0
    use_gradient_checkpointing: bool = True

    # 1.58-bit ternary quantization.
    # Set False to train a standard full-precision model.
    use_bitlinear: bool = True

    # Sub-layer norm: extra RMSNorm after each attention and FFN output.
    # Recommended True when use_bitlinear=True (improves stability).
    # Set False for full-precision runs to match the original architecture.
    use_subln: bool = True

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, \
            f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
        return self.d_model // self.n_heads


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def precompute_rope_freqs(
    head_dim: int, max_seq_len: int, theta: float = 10_000.0
) -> torch.Tensor:
    inv_f = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t     = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_f)
    return torch.polar(torch.ones_like(freqs), freqs)   # complex64 (T, head_dim//2)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    T  = x.shape[1]
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    f  = freqs[:T].unsqueeze(0).unsqueeze(2)
    return torch.view_as_real(xc * f).flatten(-2).to(x.dtype)


# ---------------------------------------------------------------------------
# BitLinear — 1.58-bit ternary quantization with straight-through estimator
# ---------------------------------------------------------------------------

class BitLinear(nn.Linear):
    """
    Drop-in replacement for nn.Linear implementing BitNet b1.58 ternary QAT.

    Weight quantization — absmean ternary {-1, 0, +1}:
      scale   = mean(|w|)                          # per-tensor L1 scale
      w_quant = round(w / scale).clamp(-1, 1) * scale  -> {-scale, 0, +scale}

      IMPORTANT: do NOT centre weights before this step. Centering (subtracting
      mean) is PrismML's binary approach and eliminates the zero value.
      The zero is what makes this 1.58-bit rather than 1-bit — in practice
      ~42-51% of weights land on 0, acting as learned feature filters and
      providing implicit L0 regularisation.

    Activation quantization — absmax int8 per token:
      x_scale = 127 / max(|x|)   (per token, last dim)
      x_quant = round(x * x_scale).clamp(-128, 127) / x_scale

    Both use the straight-through estimator (STE):
      quantized = full_precision + (quantized - full_precision).detach()
      Gradient flows as if quantization is identity. No optimizer changes needed.

    Shadow weights (the BF16 .weight tensor) are what the optimizer updates.
    Only the on-the-fly quantized values are used in the forward pass.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight

        # ---- Weight quantization (absmean ternary) ----------------------
        scale   = w.abs().mean().clamp(min=1e-5)
        w_quant = (w / scale).round().clamp(-1, 1) * scale
        # STE: backward sees the unquantized gradient
        w_quant = w + (w_quant - w).detach()

        # ---- Activation quantization (absmax per token, int8) ----------
        x_max   = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        x_scale = 127.0 / x_max
        x_quant = (x * x_scale).round().clamp(-128, 127) / x_scale
        # STE
        x_quant = x + (x_quant - x).detach()

        return F.linear(x_quant, w_quant, self.bias)


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
    def __init__(self, d_model: int, hidden: int, dropout: float,
                 use_bitlinear: bool = False) -> None:
        super().__init__()
        Linear    = BitLinear if use_bitlinear else nn.Linear
        self.gate = Linear(d_model, hidden, bias=False)
        self.up   = Linear(d_model, hidden, bias=False)
        self.down = Linear(hidden,  d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.dropout  = cfg.dropout
        Linear  = BitLinear if cfg.use_bitlinear else nn.Linear
        self.wq = Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wv = Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wo = Linear(cfg.d_model, cfg.d_model, bias=False)

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
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=drop_p, is_causal=(mask is None)
        )
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, -1))


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block with optional sub-layer norm (SubLN).

    With SubLN (use_subln=True, recommended for 1.58-bit training):
        x = x + subln_attn( attn( prenorm(x) ) )
        x = x + subln_ffn(  ffn(  prenorm(x) ) )

    Standard pre-norm (use_subln=False, original 15M architecture):
        x = x + attn( prenorm(x) )
        x = x + ffn(  prenorm(x) )

    SubLN normalises the output of each sub-layer before the residual add,
    preventing variance spikes from quantised matmuls from accumulating
    across layers. Used in the official BitNet b1.58 2B4T implementation.
    """
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attn      = Attention(cfg)
        self.norm_ffn  = RMSNorm(cfg.d_model)
        self.ffn       = SwiGLU(cfg.d_model, cfg.ffn_hidden, cfg.dropout,
                                use_bitlinear=cfg.use_bitlinear)
        self.use_subln = cfg.use_subln
        if cfg.use_subln:
            self.subln_attn = RMSNorm(cfg.d_model)
            self.subln_ffn  = RMSNorm(cfg.d_model)

    def forward(self, x, freqs, mask, kv_cache=None):
        attn_out = self.attn(self.norm_attn(x), freqs, mask, kv_cache)
        if self.use_subln:
            attn_out = self.subln_attn(attn_out)
        x = x + attn_out

        ffn_out = self.ffn(self.norm_ffn(x))
        if self.use_subln:
            ffn_out = self.subln_ffn(ffn_out)
        x = x + ffn_out
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class RustSLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Embedding and lm_head always stay full-precision (BF16).
        # Ternary quantisation of the embedding/unembedding table hurts
        # quality significantly at sub-100M scale (confirmed in literature).
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
        # Scale residual projections at init to keep residual stream variance
        # stable — GPT-2 technique, still valid for quantised models.
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
        assert T <= self.cfg.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"
        x        = self.drop(self.embed(input_ids))
        use_ckpt = self.cfg.use_gradient_checkpointing and self.training

        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            if use_ckpt:
                x = checkpoint(layer, x, self.rope_freqs, None, cache,
                               use_reentrant=False)
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
    def generate(
        self,
        input_ids:   torch.Tensor,
        max_new:     int   = 200,
        temperature: float = 1.0,
        top_k:       int   = 50,
        eos_id:      int   = -1,
    ) -> torch.Tensor:
        assert input_ids.shape[0] == 1, "generate() only supports batch size 1"
        self.eval()
        kv_caches = [{} for _ in range(self.cfg.n_layers)]
        ids = input_ids.clone()
        for _ in range(max_new):
            ctx       = ids if not kv_caches[0] else ids[:, -1:]
            logits, _ = self(ctx, kv_caches=kv_caches)
            logits    = logits[:, -1, :] / max(temperature, 1e-8)
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
    bits = torch.finfo(dtype).bits
    return count_parameters(model) * bits // 8 / 1_048_576

def ternary_inference_size_mb(model: nn.Module) -> float:
    """
    Estimates the inference footprint using 2-bit ternary packing
    (bitnet.cpp / BitNet b1.58 format: 4 ternary values packed per byte).
    Non-BitLinear params (embedding, norms) stay BF16 (16 bits).
    """
    ternary_params = 0
    full_params    = 0
    seen           = set()
    for name, module in model.named_modules():
        if id(module) in seen:
            continue
        seen.add(id(module))
        if isinstance(module, BitLinear):
            ternary_params += module.weight.numel()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            full_params += sum(p.numel() for p in module.parameters())
        elif isinstance(module, RMSNorm):
            full_params += module.scale.numel()
    total_bits = ternary_params * 2 + full_params * 16
    return total_bits / 8 / 1_048_576


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Device : {device}  |  autocast dtype: {dtype}\n")

    configs = [
        # label              d_model  n_heads  n_layers  ffn   bit    subln
        ("15M  fp         ", 256,     4,       6,        704,  False, False),
        ("15M  1.58-bit   ", 256,     4,       6,        704,  True,  True),
        ("50M  fp         ", 512,     8,       9,        1408, False, False),
        ("50M  1.58-bit   ", 512,     8,       9,        1408, True,  True),
    ]

    for label, d_model, n_heads, n_layers, ffn_hidden, use_bit, use_sub in configs:
        cfg = ModelConfig(
            d_model    = d_model,
            n_heads    = n_heads,
            n_layers   = n_layers,
            ffn_hidden = ffn_hidden,
            use_bitlinear = use_bit,
            use_subln     = use_sub,
            use_gradient_checkpointing = False,
        )
        model = RustSLM(cfg).to(device)
        n     = count_parameters(model)
        bf16  = model_size_mb(model)
        tern  = ternary_inference_size_mb(model) if use_bit else bf16

        print(f"RustSLM {label}| params={n/1e6:.2f}M  BF16={bf16:.0f}MB"
              + (f"  2-bit-packed~={tern:.0f}MB" if use_bit else ""))

        B, T = 1, 64
        ids  = torch.randint(0, cfg.vocab_size, (B, T), device=device)
        tgt  = torch.randint(0, cfg.vocab_size, (B, T), device=device)

        model.train()
        with torch.autocast(device_type=device, dtype=dtype):
            logits, loss = model(ids, targets=tgt)
        loss.backward()
        print(f"  train fwd : logits={tuple(logits.shape)}  loss={loss.item():.4f}  ✓")

        model.eval()
        out = model.generate(ids[:1, :8], max_new=5)
        print(f"  generate  : {tuple(out.shape)}  ✓\n")

    print("All checks passed.")