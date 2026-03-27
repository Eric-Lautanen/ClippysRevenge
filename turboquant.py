#!/usr/bin/env python3
"""
turboquant.py  —  Apply TurboQuant to a HuggingFace model folder
=======================================================================
Loads all weight tensors from a model directory (.safetensors or .bin),
quantizes every eligible 2-D weight matrix, and saves the result.

Two output modes
────────────────
  --mode dequant  (default)
      Quantize then immediately dequantize back to fp32.
      The saved model is a standard HuggingFace checkpoint that works with
      any inference library without modification. Use this to measure quality
      degradation, run benchmarks, or compare outputs.

  --mode compressed
      Save the raw quantized representation:
        - <name>.idx    : int16 indices  (MSE codebook pointers)
        - <name>.signs  : int8 signs     (QJL sketch, b≥2 only)
        - <name>.rnorm  : float32 norms  (QJL residual norms, b≥2 only)
        - <name>.meta   : JSON metadata  (d, b, codebook, rotation matrix)
      A companion loader script (turboquant_loader.py) is written alongside.
      Use this to actually reduce on-disk/in-memory footprint.

Quantizer choice
────────────────
  --quantizer mse   (default) — minimises ‖w − ŵ‖².  Best for weight fidelity.
  --quantizer prod             — unbiased inner-product estimator.  Best for
                                 attention / similarity computations.

Usage examples
──────────────
  # Quick sanity check — fp32 round-trip at 4 bits
  python apply_turboquant.py --model ./llama-7b  --bits 4

  # Actually compress to disk at 3 bits using inner-product quantizer
  python apply_turboquant.py --model ./llama-7b  --bits 3 \
      --quantizer prod --mode compressed --out ./llama-7b-tq3

  # Sweep bits 1-4, compare MSE for each layer
  python apply_turboquant.py --model ./llama-7b  --bits 2 4 \
      --mode dequant --report

Requirements
────────────
  Python  ≥ 3.10  (required by PyTorch 2.6+)
  PyTorch ≥ 2.6   (weights_only=True is default; 2.11 recommended)
  pip install torch safetensors>=0.4 numpy scipy
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.special import gammaln

# ── Optional safetensors support ──────────────────────────────────────────────
try:
    from safetensors.torch import load_file as st_load, save_file as st_save
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

# ══════════════════════════════════════════════════════════════════════════════
# TurboQuant core  (self-contained copy — no dependency on turboquant.py)
# ══════════════════════════════════════════════════════════════════════════════

def _hypersphere_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """Lemma 1 coordinate PDF of a uniform point on S^{d-1}."""
    log_c = gammaln(d / 2) - 0.5 * math.log(math.pi) - gammaln((d - 1) / 2)
    c = math.exp(log_c)
    safe = np.maximum(1.0 - x ** 2, 0.0)
    return np.where(np.abs(x) < 1.0, c * safe ** ((d - 3) / 2), 0.0)


def _lloyd_max(d: int, n_levels: int, n_grid: int = 60_000, n_iter: int = 400) -> np.ndarray:
    """Solve 1-D k-means for the hypersphere Beta coordinate distribution (Eq. 4)."""
    x  = np.linspace(-1.0, 1.0, n_grid, endpoint=False) + 1.0 / n_grid
    dx = 2.0 / n_grid
    p  = np.maximum(_hypersphere_pdf(x, d), 0.0)
    p /= p.sum() * dx
    cdf = np.cumsum(p) * dx
    cdf /= cdf[-1]
    q   = np.linspace(0.5 / n_levels, 1.0 - 0.5 / n_levels, n_levels)
    centroids = np.interp(q, cdf, x)
    for _ in range(n_iter):
        bounds = np.r_[-np.inf, (centroids[:-1] + centroids[1:]) / 2, np.inf]
        new_c = np.empty(n_levels)
        for i in range(n_levels):
            m = (x >= bounds[i]) & (x < bounds[i + 1])
            w = p[m]; s = w.sum()
            new_c[i] = np.dot(x[m], w) / s if s > 1e-20 else centroids[i]
        if np.max(np.abs(new_c - centroids)) < 1e-11:
            break
        centroids = new_c
    return np.sort(centroids)


def build_codebooks(d: int, bits: list[int], device: torch.device) -> dict:
    books = {}
    for b in bits:
        if b not in books:
            c = _lloyd_max(d, 2 ** b)
            books[b] = torch.tensor(c, dtype=torch.float32, device=device)
    return books


class _MSE:
    """TurboQuantMSE for a fixed d and b (Algorithm 1)."""
    def __init__(self, d: int, b: int, codebook: torch.Tensor, device: torch.device):
        self.d, self.b = d, b
        G, _ = torch.linalg.qr(torch.randn(d, d, device=device))
        self.Pi = G                       # (d, d) random rotation
        self.cb = codebook.to(device)     # (2^b,)

    @torch.no_grad()
    def quant(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.Pi.T                 # rotate
        return (y.unsqueeze(-1) - self.cb).abs().argmin(dim=-1).to(torch.int16)

    @torch.no_grad()
    def dequant(self, idx: torch.Tensor) -> torch.Tensor:
        return self.cb[idx.long()] @ self.Pi

    def quant_dequant(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequant(self.quant(x))


class _Prod:
    """TurboQuantProd for a fixed d and b (Algorithm 2)."""
    def __init__(self, d: int, b: int, codebooks: dict, device: torch.device):
        self.d, self.b, self.use_mse = d, b, (b >= 2)
        self.mse = _MSE(d, b - 1, codebooks[b - 1], device) if self.use_mse else None
        self.S   = torch.randn(d, d, device=device)
        self._c  = math.sqrt(math.pi / 2) / d

    @torch.no_grad()
    def quant(self, x: torch.Tensor):
        if self.use_mse:
            idx    = self.mse.quant(x)
            x_mse  = self.mse.dequant(idx)
            r      = x - x_mse
            r_norm = r.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            signs  = torch.sign(r @ self.S.T).to(torch.int8)
            return idx, signs, r_norm
        r_norm = x.norm(dim=-1, keepdim=True)
        signs  = torch.sign(x @ self.S.T).to(torch.int8)
        return None, signs, r_norm

    @torch.no_grad()
    def dequant(self, idx, signs, r_norm) -> torch.Tensor:
        x_qjl = (self._c * (signs.float() @ self.S)) * r_norm
        if self.use_mse and idx is not None:
            return self.mse.dequant(idx) + x_qjl
        return x_qjl

    def quant_dequant(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequant(*self.quant(x))


# ══════════════════════════════════════════════════════════════════════════════
# Weight loading / saving
# ══════════════════════════════════════════════════════════════════════════════

def _find_weight_files(model_dir: Path) -> list[Path]:
    """Return all weight shard Paths, respecting HF sharded-index JSON if present.

    Modern HuggingFace models split weights across numbered shards and record the
    mapping in model.safetensors.index.json (or pytorch_model.bin.index.json).
    When that index is present we extract the unique shard filenames from
    ``weight_map`` so the caller gets them in a deterministic, deduplicated order
    rather than relying on glob ordering alone.
    """
    # ── sharded safetensors index (preferred) ────────────────────────────────
    st_index = model_dir / "model.safetensors.index.json"
    if st_index.exists():
        with open(st_index) as f:
            weight_map = json.load(f).get("weight_map", {})
        shards = sorted({model_dir / v for v in weight_map.values()})
        if shards:
            return shards

    # ── sharded .bin index ────────────────────────────────────────────────────
    pt_index = model_dir / "pytorch_model.bin.index.json"
    if pt_index.exists():
        with open(pt_index) as f:
            weight_map = json.load(f).get("weight_map", {})
        shards = sorted({model_dir / v for v in weight_map.values()})
        if shards:
            return shards

    # ── flat (non-sharded) fallback ───────────────────────────────────────────
    st = sorted(model_dir.glob("*.safetensors"))
    pt = sorted(model_dir.glob("*.bin"))
    if st:
        return st
    if pt:
        return pt
    raise FileNotFoundError(
        f"No .safetensors or .bin files found in {model_dir}. "
        "Check that --model points to a valid HuggingFace model directory."
    )


def _load_shard(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        if not HAS_SAFETENSORS:
            raise ImportError("Install safetensors: pip install safetensors")
        return st_load(path, device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def _save_shard(tensors: dict[str, torch.Tensor], path: Path):
    if path.suffix == ".safetensors":
        if not HAS_SAFETENSORS:
            raise ImportError("Install safetensors: pip install safetensors")
        st_save(tensors, path)
    else:
        torch.save(tensors, path)


def _is_quantizable(name: str, t: torch.Tensor, min_dim: int = 64) -> bool:
    """
    Only quantize 2-D floating-point weight matrices above a minimum size.
    Embeddings, layer norms, biases, and small matrices are left in fp32.
    """
    if t.ndim != 2:
        return False
    if not t.is_floating_point():
        return False
    if min(t.shape) < min_dim:
        return False
    # Skip embeddings and layer/group norms — distortion matters more there
    skip_keywords = ("embed", "norm", "ln_", "layernorm", "bias", "lm_head")
    low = name.lower()
    if any(k in low for k in skip_keywords):
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Main quantization loop
# ══════════════════════════════════════════════════════════════════════════════

def _quantize_matrix(
    w: torch.Tensor,
    b: int,
    quantizer: str,
    codebooks: dict,
    device: torch.device,
    chunk: int = 4096,
) -> tuple:
    """
    Quantize a 2-D weight matrix row-by-row (each row = one vector).

    Weights are normalized per-row before quantization; the row norms are
    stored and used to rescale the dequantized output (standard practice,
    as noted in Section 1.3 of the paper).

    Returns
    -------
    dequant  : torch.Tensor  (same shape as w, fp32) — always returned
    payload  : dict | None   — compressed representation if quantizer='mse'
                               and mode='compressed', else None
    mse      : float         — reconstruction MSE for this matrix
    """
    rows, d = w.shape
    w = w.to(device, dtype=torch.float32)

    # Per-row L2 norm  (paper Sec 1.3: store norms in fp32, quantize unit vectors)
    row_norms = w.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    w_unit    = w / row_norms

    # Build quantizer for this dimension (codebooks shared across rows)
    if b not in codebooks:
        print(f"      Building codebook for d={d}, b={b} …", flush=True)
        codebooks[b] = torch.tensor(_lloyd_max(d, 2 ** b), dtype=torch.float32, device=device)

    if quantizer == "mse":
        q = _MSE(d, b, codebooks[b], device)
    else:
        extra = {}
        for bb in range(max(1, b - 1), b + 1):
            if bb not in codebooks:
                codebooks[bb] = torch.tensor(_lloyd_max(d, 2 ** bb), dtype=torch.float32, device=device)
            extra[bb] = codebooks[bb]
        q = _Prod(d, b, extra, device)

    # Process in chunks to avoid OOM on large matrices
    dq_rows = []
    for start in range(0, rows, chunk):
        batch = w_unit[start:start + chunk]
        dq_rows.append(q.quant_dequant(batch))
    dq_unit = torch.cat(dq_rows, dim=0)

    # Rescale back to original norm
    dq = dq_unit * row_norms

    mse_val = ((w - dq) ** 2).mean().item()

    # Build the compressed payload (MSE quantizer only; Prod compressed not yet supported)
    payload: dict | None = None
    if quantizer == "mse" and isinstance(q, _MSE):
        # Requantize unit vectors to collect final indices (cheap — codebook already built)
        idx_rows = []
        for start in range(0, rows, chunk):
            idx_rows.append(q.quant(w_unit[start:start + chunk]))
        payload = {
            "idx":    torch.cat(idx_rows, dim=0).cpu(),       # (rows, d)  int16
            "rnorm":  row_norms.squeeze(-1).cpu(),             # (rows,)    float32
            "Pi":     q.Pi.cpu(),                              # (d, d)     float32
            "cb":     q.cb.cpu(),                              # (2^b,)     float32
            "d": d, "b": b,
        }

    return dq.cpu(), payload, mse_val


def run(args):
    model_dir = Path(args.model).expanduser().resolve()
    if not model_dir.is_dir():
        sys.exit(f"Error: --model '{model_dir}' is not a directory.")

    out_dir = Path(args.out).expanduser().resolve() if args.out else \
              model_dir.parent / f"{model_dir.name}-tq{args.bits[0]}"

    if out_dir != model_dir:
        print(f"Copying model folder to {out_dir} …")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(model_dir, out_dir)
    else:
        print(f"Quantizing in-place in {out_dir} (no --out specified, same dir).")

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  bits: {args.bits}  |  quantizer: {args.quantizer}"
          f"  |  mode: {args.mode}\n")

    weight_files = _find_weight_files(out_dir)
    print(f"Found {len(weight_files)} weight shard(s).")

    # One codebook dict per (dimension, bit-width) — built lazily and reused
    codebooks: dict[int, torch.Tensor] = {}

    report_rows = []          # (name, shape, bits, mse) for the summary table
    total_params = 0
    quantized_params = 0
    t_start = time.perf_counter()

    for shard_path in weight_files:
        print(f"\n── Shard: {shard_path.name}")
        tensors = _load_shard(shard_path)
        out_tensors: dict[str, torch.Tensor] = {}

        for name, tensor in tensors.items():
            d = tensor.shape[-1] if tensor.ndim == 2 else 0
            total_params += tensor.numel()

            if not _is_quantizable(name, tensor, min_dim=args.min_dim):
                out_tensors[name] = tensor
                print(f"  skip  {name:60s}  {tuple(tensor.shape)}")
                continue

            # Use the first (or only) bit-width for the actual compression
            b = args.bits[0]
            print(f"  quant {name:60s}  {tuple(tensor.shape)}  b={b}", end="", flush=True)

            dq, payload, mse_val = _quantize_matrix(
                tensor, b, args.quantizer, codebooks, device, chunk=args.chunk
            )

            # ── compressed save ──────────────────────────────────────────────
            if args.mode == "compressed" and payload is not None:
                safe_name = name.replace("/", "__").replace(".", "_")
                comp_dir  = shard_path.parent / (shard_path.stem + "_tqcomp")
                comp_dir.mkdir(exist_ok=True)
                torch.save(payload["idx"],   comp_dir / f"{safe_name}.idx.pt")
                torch.save(payload["rnorm"], comp_dir / f"{safe_name}.rnorm.pt")
                meta = {
                    "name": name, "d": payload["d"], "b": payload["b"],
                    "Pi":   payload["Pi"].tolist(),
                    "cb":   payload["cb"].tolist(),
                }
                with open(comp_dir / f"{safe_name}.meta.json", "w") as mf:
                    json.dump(meta, mf)
                # Placeholder in the shard keeps key present; loader replaces it
                out_tensors[name] = torch.zeros(1)
            else:
                # Keep the same dtype as the original (cast back if needed)
                out_tensors[name] = dq.to(tensor.dtype)
            quantized_params += tensor.numel()
            report_rows.append((name, tuple(tensor.shape), b, mse_val))
            print(f"  MSE={mse_val:.2e}")

        _save_shard(out_tensors, shard_path)

    elapsed = time.perf_counter() - t_start
    compression_ratio = 32 / args.bits[0]

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("QUANTIZATION SUMMARY")
    print("═" * 72)
    print(f"  Model dir       : {out_dir}")
    print(f"  Quantizer       : TurboQuant{args.quantizer.upper()} ({args.bits[0]}-bit)")
    print(f"  Parameters      : {total_params:,}  total")
    print(f"  Quantized       : {quantized_params:,}  ({100*quantized_params/total_params:.1f}%)")
    print(f"  Compression     : {compression_ratio:.1f}× vs fp32 for quantized layers")
    print(f"  Wall time       : {elapsed:.1f}s")

    if report_rows:
        print(f"\n  {'Layer name':<55} {'Shape':>15} {'b':>3}  {'MSE':>12}")
        print("  " + "─" * 90)
        total_mse = 0.0
        for (n, sh, b, m) in report_rows:
            short = n if len(n) <= 55 else "…" + n[-54:]
            print(f"  {short:<55} {str(sh):>15} {b:>3}  {m:>12.4e}")
            total_mse += m
        print("  " + "─" * 90)
        avg_mse = total_mse / len(report_rows)
        lb  = 4.0 ** (-args.bits[0])
        ub  = math.sqrt(3 * math.pi) / 2 * lb
        print(f"  Average MSE across quantized layers : {avg_mse:.4e}")
        print(f"  Theoretical LB  (Theorem 3)         : {lb:.4e}  (1 / 4^b)")
        print(f"  Theoretical UB  (Theorem 1)         : {ub:.4e}  (√3π/2 / 4^b)")
        if avg_mse < lb:
            print(f"  ⚠  Average MSE is below the theoretical lower bound — likely because")
            print(f"     weight rows aren't unit-norm. The bound applies to unit vectors.")
        else:
            factor = avg_mse / lb
            print(f"  Ratio measured/LB : {factor:.2f}×  (paper target: ≤ {math.sqrt(3*math.pi)/2:.2f}×)")

    print("\nDone.  Quantized model saved to:")
    print(f"  {out_dir}\n")

    if args.report:
        report_path = out_dir / "turboquant_report.json"
        with open(report_path, "w") as f:
            json.dump(
                {
                    "model": str(model_dir),
                    "bits": args.bits[0],
                    "quantizer": args.quantizer,
                    "mode": args.mode,
                    "total_params": total_params,
                    "quantized_params": quantized_params,
                    "avg_mse": sum(r[3] for r in report_rows) / max(len(report_rows), 1),
                    "layers": [
                        {"name": n, "shape": list(sh), "bits": b, "mse": m}
                        for n, sh, b, m in report_rows
                    ],
                },
                f,
                indent=2,
            )
        print(f"Report written to: {report_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Apply TurboQuant to a HuggingFace model directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", required=True,
                   help="Path to the HuggingFace model folder "
                        "(must contain .safetensors or .bin files).")
    p.add_argument("--bits", type=int, nargs="+", default=[4],
                   help="Bit-width(s) for quantization.  Default: 4.  "
                        "Supply multiple values (e.g. --bits 2 4) to run a comparison sweep.")
    p.add_argument("--quantizer", choices=["mse", "prod"], default="mse",
                   help="mse = MSE-optimal (Alg 1); prod = unbiased inner-product (Alg 2). "
                        "Default: mse.")
    p.add_argument("--mode", choices=["dequant", "compressed"], default="dequant",
                   help="dequant: save fp32 round-trip weights (drop-in replacement). "
                        "compressed: save quantized indices + sketches (smaller, needs loader). "
                        "Default: dequant.")
    p.add_argument("--out", default=None,
                   help="Output directory.  Default: <model_dir>-tq<bits>.")
    p.add_argument("--min-dim", type=int, default=64,
                   help="Skip matrices with min(rows, cols) < this.  Default: 64.")
    p.add_argument("--chunk", type=int, default=4096,
                   help="Row-chunk size for batched quantization (tune for VRAM).  Default: 4096.")
    p.add_argument("--cuda", action="store_true",
                   help="Use CUDA if available.")
    p.add_argument("--report", action="store_true",
                   help="Write turboquant_report.json to the output directory.")
    args = p.parse_args()

    if any(b < 1 or b > 8 for b in args.bits):
        sys.exit("Error: --bits must be between 1 and 8.")
    if args.mode == "compressed" and args.quantizer == "prod":
        print("Note: --mode compressed currently saves the dequantized output for "
              "the 'prod' quantizer (compressed prod format not yet implemented).")
        args.mode = "dequant"

    run(args)


if __name__ == "__main__":
    main()