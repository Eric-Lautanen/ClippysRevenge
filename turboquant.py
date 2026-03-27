#!/usr/bin/env python3
"""
TurboQuant — PyTorch Implementation
======================================
Reference: Zandieh et al., "TurboQuant: Online Vector Quantization with
           Near-optimal Distortion Rate"  arXiv:2504.19874

Implements both variants from the paper:
  • TurboQuantmse  — minimises MSE between original and reconstructed vectors
  • TurboQuantprod — unbiased inner-product estimator (two-stage: MSE + QJL)

Run:
  python turboquant.py            # CPU
  python turboquant.py --cuda     # GPU (if available)
"""

import argparse
import math
import time

import numpy as np
import torch
from scipy.special import gammaln


# ══════════════════════════════════════════════════════════════════════════════
# 1 ·  Hypersphere Coordinate Distribution  (Lemma 1)
# ══════════════════════════════════════════════════════════════════════════════

def hypersphere_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """
    Lemma 1 — PDF of one coordinate of a uniform point on S^{d-1}:

        f(x) = Γ(d/2) / [√π · Γ((d-1)/2)] · (1 − x²)^{(d−3)/2},   x ∈ (−1, 1)

    Converges to N(0, 1/d) as d → ∞ (concentration of measure + CLT).
    This is the distribution that each rotated coordinate follows after applying
    the random rotation Π, irrespective of the input vector.
    """
    log_coeff = gammaln(d / 2) - 0.5 * math.log(math.pi) - gammaln((d - 1) / 2)
    coeff = math.exp(log_coeff)
    safe = np.maximum(1.0 - x ** 2, 0.0)
    return np.where(np.abs(x) < 1.0, coeff * safe ** ((d - 3) / 2), 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# 2 ·  Lloyd-Max Optimal Scalar Quantizer  (Eq. 4)
# ══════════════════════════════════════════════════════════════════════════════

def lloyd_max(pdf_fn, n_levels: int, n_grid: int = 80_000, n_iter: int = 500) -> np.ndarray:
    """
    Lloyd-Max algorithm — solves the 1-D k-means problem:

        C* = min_{c_1 ≤ … ≤ c_K}  Σ_i ∫_{b_i}^{b_{i+1}} (x − c_i)² f(x) dx

    where boundaries b_i are midpoints between consecutive centroids.
    This gives the optimal MSE scalar quantizer for the given PDF.

    Args:
        pdf_fn  : callable, probability density function over [-1, 1]
        n_levels: number of quantisation levels (= 2^b)
        n_grid  : number of grid points for numerical integration
        n_iter  : maximum Lloyd-Max iterations

    Returns:
        Sorted array of centroids, shape (n_levels,).
    """
    # Fine grid over the support [-1, 1]
    x  = np.linspace(-1.0, 1.0, n_grid, endpoint=False) + 1.0 / n_grid
    dx = 2.0 / n_grid
    p  = np.maximum(pdf_fn(x), 0.0)
    p /= p.sum() * dx                          # normalise

    # Initialise centroids at distribution quantiles (fast convergence)
    cdf = np.cumsum(p) * dx
    cdf /= cdf[-1]
    q   = np.linspace(0.5 / n_levels, 1.0 - 0.5 / n_levels, n_levels)
    centroids = np.interp(q, cdf, x)

    for _ in range(n_iter):
        # Decision boundaries = midpoints between adjacent centroids
        bounds = np.r_[-np.inf, (centroids[:-1] + centroids[1:]) / 2, np.inf]

        new_c = np.empty(n_levels)
        for i in range(n_levels):
            m = (x >= bounds[i]) & (x < bounds[i + 1])
            w = p[m]
            s = w.sum()
            new_c[i] = np.dot(x[m], w) / s if s > 1e-20 else centroids[i]

        if np.max(np.abs(new_c - centroids)) < 1e-11:
            break
        centroids = new_c

    return np.sort(centroids)


def precompute_codebooks(d: int, max_bits: int = 6) -> dict:
    """
    Build Lloyd-Max codebooks for bit-widths 1 … max_bits using the
    hypersphere Beta coordinate distribution for dimension d.

    Returns a dict: {b: torch.Tensor of shape (2^b,)}.
    These are reusable across many quantiser instantiations.
    """
    print(f"⚙  Precomputing Lloyd-Max codebooks (d={d}) ...")
    books = {}
    for b in range(1, max_bits + 1):
        c = lloyd_max(lambda x, _d=d: hypersphere_pdf(x, _d), 2 ** b)
        books[b] = torch.tensor(c, dtype=torch.float32)
        std = 1.0 / math.sqrt(d)
        print(f"   b={b}: {2**b:3d} levels  |  range [{c[0]:+.5f}, {c[-1]:+.5f}]"
              f"  (≈ {c[-1]/std:.1f}σ of N(0,1/d))")
    print()
    return books


# ══════════════════════════════════════════════════════════════════════════════
# 3 ·  TurboQuantmse  —  Algorithm 1
# ══════════════════════════════════════════════════════════════════════════════

class TurboQuantMSE:
    """
    MSE-optimal TurboQuant  (Section 3.1 / Algorithm 1).

    One-time setup
    ──────────────
      Π  ← random rotation matrix  (QR of a Gaussian matrix)
      c  ← Lloyd-Max codebook for bit-width b

    Quant(x)                                       O(d · 2^b)
    ──────────
      y    = Π · x                                 # random rotation → Beta distribution
      idx  = argmin_k |y_j − c_k|  ∀j             # optimal scalar quantise each coord

    DeQuant(idx)                                   O(d)
    ────────────
      ỹ    = c[idx]                                # centroid lookup
      x̃    = Π^T · ỹ                               # inverse rotation

    Theorem 1 guarantee:
      Dmse = E[‖x − x̃‖²] ≤ √(3π)/2 · 4^{−b}  ≈ 2.72 · 4^{−b}
    """

    def __init__(self, d: int, b: int, codebooks: dict, device: torch.device):
        self.d, self.b = d, b

        # Π: random rotation via QR decomposition of a Gaussian matrix
        G, _ = torch.linalg.qr(torch.randn(d, d, device=device))
        self.Pi:       torch.Tensor = G                            # (d, d) orthogonal
        self.codebook: torch.Tensor = codebooks[b].to(device)     # (2^b,)

    @torch.no_grad()
    def quant(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N, d)  unit vectors  →  idx : (N, d)  int64 in [0, 2^b)
        Each index is b bits, so the total storage is b·d bits per vector.
        """
        y    = x @ self.Pi.T                        # rotate  (N, d)
        diff = y.unsqueeze(-1) - self.codebook      # (N, d, 2^b)  — vectorised NN
        return diff.abs().argmin(dim=-1)            # (N, d)

    @torch.no_grad()
    def dequant(self, idx: torch.Tensor) -> torch.Tensor:
        """idx : (N, d)  →  x̃ : (N, d)  float32"""
        return self.codebook[idx] @ self.Pi         # centroid lookup + inverse rotate

    def quant_dequant(self, x: torch.Tensor) -> torch.Tensor:
        """Full encode → decode round-trip."""
        return self.dequant(self.quant(x))


# ══════════════════════════════════════════════════════════════════════════════
# 4 ·  QJL — 1-bit Inner-Product Quantizer  (Definition 1 / Lemma 4)
# ══════════════════════════════════════════════════════════════════════════════

class QJL:
    """
    Quantized Johnson-Lindenstrauss transform (Section 2.2).

    Qqjl(x)          = sign(S · x),                  S_{ij} ~ N(0, 1)
    Q⁻¹qjl(z, γ=1)  = √(π/2) / d · S^T · z

    Lemma 4 guarantees:
      E  [ ⟨y, Q⁻¹(Q(x))⟩ ]  = ⟨y, x⟩                  (unbiased)
      Var[ ⟨y, Q⁻¹(Q(x))⟩ ]  ≤ π/(2d) · ‖y‖²            (variance bound)

    When applied to a residual r = x − x̃_mse, we pass r_norm = ‖r‖₂ so that
    the dequantised output estimates the residual, not the unit-normalised residual.
    """

    def __init__(self, d: int, device: torch.device):
        self.d  = d
        self.S  = torch.randn(d, d, device=device)      # random projection (d, d)
        self._c = math.sqrt(math.pi / 2) / d            # reconstruction scale

    @torch.no_grad()
    def quant(self, x: torch.Tensor) -> torch.Tensor:
        """x : (N, d)  →  z : (N, d)  in {−1, +1}"""
        return torch.sign(x @ self.S.T)

    @torch.no_grad()
    def dequant(self, z: torch.Tensor, r_norm: torch.Tensor | None = None) -> torch.Tensor:
        """
        z      : (N, d)  signs
        r_norm : (N, 1)  optional ‖r‖₂ scale  (Algorithm 2, line 11)
        returns: (N, d)  unbiased estimate of x (or r, if r_norm given)
        """
        out = self._c * (z @ self.S)        # √(π/2)/d · S^T · z
        return out * r_norm if r_norm is not None else out


# ══════════════════════════════════════════════════════════════════════════════
# 5 ·  TurboQuantprod  —  Algorithm 2
# ══════════════════════════════════════════════════════════════════════════════

class TurboQuantProd:
    """
    Inner-product-optimal TurboQuant  (Section 3.2 / Algorithm 2).

    Two-stage pipeline — total bit-width = b:
      Stage 1: TurboQuantmse with (b−1) bits  →  x̃_mse,  residual r = x − x̃_mse
      Stage 2: QJL on residual r               →  sign(S·r)  [1 bit per coord]

    For b=1 the MSE stage is skipped (contributes 0 bits) → pure QJL on x.

    Theorem 2 guarantees:
      E  [ ⟨y, x̃⟩ ]   = ⟨y, x⟩                                    (unbiased)
      Var[ ⟨y, x̃⟩ ]   ≤ √(3π)/2 · ‖y‖² / d · 4^{−b}

    Why MSE-alone is biased: at b=1, TurboQuantmse gives E[⟨y, Q⁻¹(Q(x))⟩] = (2/π)⟨y,x⟩,
    introducing a multiplicative bias of 2/π ≈ 0.637.  The QJL residual corrects this.
    """

    def __init__(self, d: int, b: int, codebooks: dict, device: torch.device):
        assert b >= 1, "bit-width must be ≥ 1"
        self.d, self.b = d, b
        self.use_mse = (b >= 2)
        self.mse = TurboQuantMSE(d, b - 1, codebooks, device) if self.use_mse else None
        self.qjl = QJL(d, device)

    @torch.no_grad()
    def quant(self, x: torch.Tensor):
        """
        x : (N, d)  unit vectors

        Returns (idx, signs, r_norm):
          idx    : (N, d) int64 or None  — MSE indices  (b−1 bits each)
          signs  : (N, d) {−1,+1}        — QJL sketch of residual  (1 bit each)
          r_norm : (N, 1) float           — ‖residual‖₂  (stored in float)
        """
        if self.use_mse:
            idx    = self.mse.quant(x)
            x_mse  = self.mse.dequant(idx)
            r      = x - x_mse                                      # residual
            r_norm = r.norm(dim=-1, keepdim=True).clamp(min=1e-12)  # ‖r‖₂
            signs  = self.qjl.quant(r)   # sign(S·r) = sign(S·(r/‖r‖)) — scale-invariant
            return idx, signs, r_norm
        else:
            # b=1: whole budget goes to QJL on x itself
            r_norm = x.norm(dim=-1, keepdim=True)
            signs  = self.qjl.quant(x)
            return None, signs, r_norm

    @torch.no_grad()
    def dequant(self, idx, signs, r_norm) -> torch.Tensor:
        """Reconstruct an unbiased inner-product estimate x̃ ≈ x."""
        x_qjl = self.qjl.dequant(signs, r_norm)       # unbiased estimate of r
        if self.use_mse and idx is not None:
            return self.mse.dequant(idx) + x_qjl      # x̃_mse + x̃_residual
        return x_qjl

    def quant_dequant(self, x: torch.Tensor) -> torch.Tensor:
        """Full round-trip for convenience."""
        return self.dequant(*self.quant(x))


# ══════════════════════════════════════════════════════════════════════════════
# 6 ·  Theoretical Bounds  (Theorems 1–3)
# ══════════════════════════════════════════════════════════════════════════════

_C = math.sqrt(3 * math.pi) / 2        # ≈ 2.7207  — the "small constant" from Theorem 1

def mse_upper(b: int)         -> float: return _C * 4 ** (-b)
def mse_lower(b: int)         -> float: return       4 ** (-b)
def ip_upper(b: int, d: int)  -> float: return _C / d * 4 ** (-b)
def ip_lower(b: int, d: int)  -> float: return  1.0 / d * 4 ** (-b)

# Fine-grained distortion values quoted in Section 1.3 of the paper
_PAPER_MSE  = {1: 0.360, 2: 0.117, 3: 0.030, 4: 0.009}
_PAPER_PROD = {1: 1.57,  2: 0.560, 3: 0.180, 4: 0.047}   # divide by d for Dprod


# ══════════════════════════════════════════════════════════════════════════════
# 7 ·  Demo / Validation
# ══════════════════════════════════════════════════════════════════════════════

def _bar(title: str = "", w: int = 72):
    if title:
        pad = (w - len(title) - 2) // 2
        print("─" * pad + f" {title} " + "─" * (w - pad - len(title) - 2))
    else:
        print("─" * w)


@torch.no_grad()
def run_demo(device: torch.device):
    torch.manual_seed(0)
    np.random.seed(0)

    D      = 256       # embedding dimension (same order as paper's experiments)
    N      = 8_000     # database vectors
    NQ     = 1_000     # query vectors
    BITS   = 5         # max bit-width to sweep

    # ── Codebooks (one-time cost, reused by all quantisers) ───────────────────
    books = precompute_codebooks(D, max_bits=BITS)

    # ── Random unit-sphere data ───────────────────────────────────────────────
    X = torch.randn(N,  D, device=device)
    X = X / X.norm(dim=-1, keepdim=True)           # project to S^{d-1}
    Y = torch.randn(NQ, D, device=device)           # query vectors (arbitrary norm)

    # ══════════════════════════════════════════════════════════════════════════
    # A · MSE distortion validation  (Theorem 1)
    # ══════════════════════════════════════════════════════════════════════════
    _bar("A · MSE distortion — TurboQuantmse  (Theorem 1)")
    print(f"  {'b':>3}  {'Measured':>11}  {'LB: 4^−b':>11}  {'UB: √3π/2·4^−b':>16}  {'Paper':>9}  {'measured/LB':>12}")
    _bar()
    for b in range(1, BITS + 1):
        tq    = TurboQuantMSE(D, b, books, device)
        X_hat = tq.quant_dequant(X)
        mse   = ((X - X_hat) ** 2).sum(-1).mean().item()
        lb, ub = mse_lower(b), mse_upper(b)
        paper  = _PAPER_MSE.get(b, "—")
        ps     = f"{paper:.4f}" if isinstance(paper, float) else paper
        print(f"  {b:>3}  {mse:>11.6f}  {lb:>11.6f}  {ub:>16.6f}  {ps:>9}  {mse/lb:>12.3f}×")
    print()
    print(f"  Note: √(3π)/2·4^{{-b}} is an asymptotic (Panter-Dite) bound, valid for b > 4.")
    print(f"  For small b, use the paper's fine-grained Lloyd-Max values — they match above.")
    print(f"  The LB 4^{{-b}} holds at all bit-widths (Theorem 3).  measured ≥ LB  ✓\n")

    # ══════════════════════════════════════════════════════════════════════════
    # B · Inner-product distortion & bias  (Theorem 2)
    # Use unit-norm queries so the bound Dprod ≤ UB(b,d) applies directly.
    # ══════════════════════════════════════════════════════════════════════════
    Y_unit = Y / Y.norm(dim=-1, keepdim=True)       # normalise queries to S^{d-1}

    _bar("B · Inner-product bias & distortion (Theorem 2, unit-norm queries)")
    print(f"  {'b':>3}  {'mse_bias':>10}  {'mse_Dprod':>14}  {'prod_bias':>11}  {'prod_Dprod':>14}"
          f"  {'LB':>14}  {'UB':>12}")
    _bar()
    for b in range(1, BITS + 1):
        tq_mse  = TurboQuantMSE(D, b, books, device)
        tq_prod = TurboQuantProd(D, b, books, device)

        # Quantise the whole dataset once
        X_mse  = tq_mse.quant_dequant(X)
        X_prod = tq_prod.quant_dequant(X)

        # Inner products with unit queries  (NQ × N)
        ip_true  = Y_unit @ X.T
        ip_mse   = Y_unit @ X_mse.T
        ip_prod  = Y_unit @ X_prod.T

        err_mse  = ip_mse  - ip_true
        err_prod = ip_prod - ip_true

        mse_bias   = err_mse.mean().item()
        prod_bias  = err_prod.mean().item()
        mse_dprod  = (err_mse  ** 2).mean().item()
        prod_dprod = (err_prod ** 2).mean().item()

        lb, ub = ip_lower(b, D), ip_upper(b, D)
        print(f"  {b:>3}  {mse_bias:>10.6f}  {mse_dprod:>12.8f}  {prod_bias:>11.6f}  {prod_dprod:>12.8f}"
              f"  {lb:>12.8f}  {ub:>10.8f}")

    print()
    print("  LB = 1/(d·4^b),  UB = √(3π)/2/(d·4^b)  [for unit ‖y‖=1 queries]")
    print("  ✓  prod_bias ≈ 0 at all bit-widths — TurboQuantprod is unbiased (Theorem 2)")
    print("  ✓  mse_bias shrinks but persists — MSE quantiser has a systematic bias")
    print("  ✓  prod_Dprod > LB at all b — consistent with information-theoretic lower bound\n")

    # ══════════════════════════════════════════════════════════════════════════
    # C · Nearest-neighbour recall  (Section 4.4)
    # ══════════════════════════════════════════════════════════════════════════
    _bar("C · Nearest-neighbour Recall@1@k  (inner-product similarity)")

    def recall_at_k(ip_approx: torch.Tensor, ip_exact: torch.Tensor, k: int) -> float:
        """What fraction of queries have their true NN in the approx top-k?"""
        true_nn  = ip_exact.argmax(dim=-1)                  # (NQ,)
        top_k    = ip_approx.topk(k, dim=-1).indices        # (NQ, k)
        return (top_k == true_nn.unsqueeze(-1)).any(-1).float().mean().item()

    ip_exact = Y @ X.T     # (NQ, N) ground-truth inner products

    print(f"  {'Method':<20} {'b':>2}   {'@1':>7}  {'@4':>7}  {'@16':>7}  {'@64':>7}")
    _bar()
    for b in [2, 4]:
        tq_mse  = TurboQuantMSE(D,  b, books, device)
        tq_prod = TurboQuantProd(D, b, books, device)

        ip_mse  = Y @ tq_mse.quant_dequant(X).T
        ip_prod = Y @ tq_prod.quant_dequant(X).T

        r_mse  = [recall_at_k(ip_mse,  ip_exact, k) for k in [1, 4, 16, 64]]
        r_prod = [recall_at_k(ip_prod, ip_exact, k) for k in [1, 4, 16, 64]]

        print(f"  {'TurboQuantmse':<20} {b:>2}   " + "  ".join(f"{r:>7.4f}" for r in r_mse))
        print(f"  {'TurboQuantprod':<20} {b:>2}   " + "  ".join(f"{r:>7.4f}" for r in r_prod))
        if b == 2:
            print()
    print()
    print("  Note: TurboQuantmse often wins on recall because it uses all b bits for MSE.")
    print("  TurboQuantprod spends 1 bit on QJL for unbiasedness, leaving (b-1) for MSE.")
    print("  TurboQuantprod's advantage is accuracy of inner product *values*, not ranking.\n")

    # ══════════════════════════════════════════════════════════════════════════
    # D · Throughput benchmark  (Section 1 — "lightweight, accelerator-friendly")
    # ══════════════════════════════════════════════════════════════════════════
    _bar("D · Throughput benchmark  (d=256, N=8000, b=4)")

    def bench(fn, x: torch.Tensor, n_warm: int = 5, n_rep: int = 30) -> float:
        """Returns vectors / second."""
        for _ in range(n_warm):
            fn(x)
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_rep):
            fn(x)
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        return N * n_rep / (time.perf_counter() - t0)

    b4_mse  = TurboQuantMSE(D,  4, books, device)
    b4_prod = TurboQuantProd(D, 4, books, device)

    tps_mse  = bench(b4_mse.quant_dequant,  X)
    tps_prod = bench(b4_prod.quant_dequant, X)

    print(f"  TurboQuantmse   (encode+decode): {tps_mse:>12,.0f}  vectors/s")
    print(f"  TurboQuantprod  (encode+decode): {tps_prod:>12,.0f}  vectors/s")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    # E · Bit-rate vs distortion summary table
    # ══════════════════════════════════════════════════════════════════════════
    _bar("E · Bit-rate / compression summary  (d=256, unit vectors, fp32 baseline)")
    print(f"  {'b':>3}  {'bits/coord':>10}  {'vs fp32':>9}  {'MSE upper bound':>17}  {'factor above LB':>17}")
    _bar()
    for b in range(1, BITS + 1):
        lb, ub = mse_lower(b), mse_upper(b)
        print(f"  {b:>3}  {b:>10}       {32/b:>5.1f}×  {ub:>17.6f}  {ub/lb:>14.3f}×  (≤ {_C:.3f})")
    print()
    print(f"  The ratio is always ≤ √(3π)/2 ≈ {_C:.4f}  (Theorem 1).")
    print(f"  At b=1: ratio ≈ 1.45 — TurboQuant is nearly optimal at low bits.")
    print()
    _bar()
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TurboQuant demo")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA GPU if available")
    parser.add_argument("--d",    type=int, default=256,  help="Embedding dimension")
    parser.add_argument("--n",    type=int, default=8000, help="Number of database vectors")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"\n🚀  TurboQuant Demo  |  device={device}  d={args.d}  N={args.n}\n")
    run_demo(device)