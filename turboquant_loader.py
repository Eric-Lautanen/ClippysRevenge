#!/usr/bin/env python3
"""
turboquant_loader.py  —  Load a TurboQuant compressed model checkpoint
=======================================================================
Companion to turboquant.py --mode compressed.

For every layer saved in the ``<shard>_tqcomp/`` directory the loader
reconstructs the full fp32 weight matrix on-the-fly by dequantizing the
stored int16 indices, rotation matrix, and codebook.  Non-quantized
tensors (embeddings, norms, biases …) are loaded directly from the
original shard files.

Quick usage
───────────
  # As a drop-in state_dict (works with any HuggingFace model class)
  from load_compressed import load_state_dict
  sd = load_state_dict("./llama-7b-tq3", device="cuda")
  model.load_state_dict(sd)

  # Or get a generator that yields (name, tensor) pairs — handy for
  # very large models where you want to load one layer at a time:
  from load_compressed import iter_tensors
  for name, weight in iter_tensors("./llama-7b-tq3"):
      model.get_parameter(name).data.copy_(weight)

  # Command-line: verify all layers reconstruct without error
  python turboquant_loader.py --model ./llama-7b-tq3 [--device cpu]

Requirements
────────────
  Python  ≥ 3.10
  PyTorch ≥ 2.6
  pip install torch safetensors>=0.4 numpy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Generator

import torch

# ── Optional safetensors support ─────────────────────────────────────────────
try:
    from safetensors.torch import load_file as st_load
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


# ══════════════════════════════════════════════════════════════════════════════
# Low-level dequantization  (mirrors _MSE.dequant in turboquant.py)
# ══════════════════════════════════════════════════════════════════════════════

def _dequant_mse(
    idx:   torch.Tensor,   # (rows, d)  int16 — codebook indices
    rnorm: torch.Tensor,   # (rows,)    float32 — per-row L2 norms
    Pi:    torch.Tensor,   # (d, d)     float32 — random rotation matrix
    cb:    torch.Tensor,   # (2^b,)     float32 — Lloyd-Max codebook
) -> torch.Tensor:
    """Reconstruct a weight matrix from its MSE-quantized representation."""
    # cb[idx] → (rows, d) unit-norm coordinates in the rotated basis
    # @ Pi    → rotate back to the original basis
    # * rnorm → restore per-row scale
    unit = cb[idx.long()] @ Pi                    # (rows, d)
    return unit * rnorm.unsqueeze(-1)             # (rows, d)  fp32


# ══════════════════════════════════════════════════════════════════════════════
# Shard / comp-dir discovery
# ══════════════════════════════════════════════════════════════════════════════

def _find_shards(model_dir: Path) -> list[Path]:
    """Return weight shard paths, honouring HF sharded-index JSON if present."""
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = model_dir / index_name
        if index_path.exists():
            with open(index_path) as f:
                weight_map = json.load(f).get("weight_map", {})
            shards = sorted({model_dir / v for v in weight_map.values()})
            if shards:
                return shards

    st = sorted(model_dir.glob("*.safetensors"))
    pt = sorted(model_dir.glob("*.bin"))
    if st:
        return st
    if pt:
        return pt
    raise FileNotFoundError(
        f"No weight files found in {model_dir}.  "
        "Expected .safetensors, .bin, or a sharded index JSON."
    )


def _load_shard_raw(path: Path) -> dict[str, torch.Tensor]:
    """Load all tensors from one shard (safetensors or legacy pickle .bin)."""
    if path.suffix == ".safetensors":
        if not HAS_SAFETENSORS:
            raise ImportError(
                "safetensors is required to load .safetensors files.  "
                "pip install 'safetensors>=0.4'"
            )
        return st_load(path, device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def _comp_dir_for(shard_path: Path) -> Path:
    """Return the companion _tqcomp directory for a given shard file."""
    return shard_path.parent / (shard_path.stem + "_tqcomp")


def _load_compressed_layer(
    safe_name: str,
    comp_dir: Path,
    device: torch.device,
) -> torch.Tensor:
    """Reconstruct one weight matrix from its compressed representation."""
    meta_path = comp_dir / f"{safe_name}.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    Pi  = torch.tensor(meta["Pi"], dtype=torch.float32, device=device)
    cb  = torch.tensor(meta["cb"], dtype=torch.float32, device=device)
    idx = torch.load(
        comp_dir / f"{safe_name}.idx.pt",
        map_location=device,
        weights_only=True,
    )
    rnorm = torch.load(
        comp_dir / f"{safe_name}.rnorm.pt",
        map_location=device,
        weights_only=True,
    )

    return _dequant_mse(idx, rnorm, Pi, cb)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def iter_tensors(
    model_dir: str | Path,
    device: str | torch.device = "cpu",
    target_dtype: torch.dtype | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield ``(name, tensor)`` pairs for every weight in the model.

    Compressed layers are dequantized on-the-fly; uncompressed layers are
    yielded directly from the shard.  Tensors are moved to ``device`` and,
    if ``target_dtype`` is given, cast to that dtype before yielding.

    Parameters
    ----------
    model_dir:
        Path to the TurboQuant output directory (same ``--out`` value used
        when running turboquant.py).
    device:
        Target device for all returned tensors.  Default: ``"cpu"``.
    target_dtype:
        Optional dtype to cast all tensors to (e.g. ``torch.bfloat16``).
        If *None*, fp32 is returned for dequantized layers; original dtype
        is preserved for uncompressed layers.
    """
    model_dir = Path(model_dir).expanduser().resolve()
    device    = torch.device(device)
    shards    = _find_shards(model_dir)

    for shard_path in shards:
        raw      = _load_shard_raw(shard_path)
        comp_dir = _comp_dir_for(shard_path)

        for name, raw_tensor in raw.items():
            # Build the safe filename the compressor would have used
            safe_name = name.replace("/", "__").replace(".", "_")
            meta_path = comp_dir / f"{safe_name}.meta.json"

            if comp_dir.exists() and meta_path.exists():
                weight = _load_compressed_layer(safe_name, comp_dir, device)
            else:
                # Not compressed — yield as-is (norm, embed, bias, etc.)
                weight = raw_tensor.to(device=device, dtype=torch.float32)

            if target_dtype is not None:
                weight = weight.to(dtype=target_dtype)

            yield name, weight


def load_state_dict(
    model_dir: str | Path,
    device: str | torch.device = "cpu",
    target_dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Return a complete ``state_dict`` with all weights reconstructed.

    This is the simplest way to use a compressed checkpoint::

        sd    = load_state_dict("./llama-7b-tq3")
        model.load_state_dict(sd, strict=False)

    For very large models, prefer :func:`iter_tensors` to avoid holding every
    layer in memory simultaneously.
    """
    return {name: tensor for name, tensor in iter_tensors(model_dir, device, target_dtype)}


# ══════════════════════════════════════════════════════════════════════════════
# CLI  —  smoke-test / inspection
# ══════════════════════════════════════════════════════════════════════════════

def _cli_main():
    p = argparse.ArgumentParser(
        description="Load and verify a TurboQuant compressed checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", required=True,
                   help="Path to the TurboQuant output directory.")
    p.add_argument("--device", default="cpu",
                   help="Device to dequantize on (cpu / cuda).  Default: cpu.")
    p.add_argument("--dtype", default=None,
                   help="Cast output tensors to this dtype "
                        "(float32 / float16 / bfloat16).  Default: float32.")
    p.add_argument("--save", default=None,
                   help="If given, save the reconstructed state_dict to this "
                        ".safetensors file (requires safetensors).")
    args = p.parse_args()

    dtype_map = {
        "float32":  torch.float32,
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        None:       None,
    }
    if args.dtype not in dtype_map:
        sys.exit(f"Error: --dtype must be one of {list(dtype_map)[:-1]}")
    target_dtype = dtype_map[args.dtype]

    model_dir = Path(args.model)
    device    = torch.device(args.device)

    print(f"Loading compressed model from: {model_dir}")
    print(f"Device: {device}  |  dtype: {args.dtype or 'float32 (default)'}\n")

    t0       = time.perf_counter()
    n_layers = 0
    n_comp   = 0
    total_params = 0
    state_dict: dict[str, torch.Tensor] = {}

    shards   = _find_shards(model_dir)
    for shard_path in shards:
        raw      = _load_shard_raw(shard_path)
        comp_dir = _comp_dir_for(shard_path)
        print(f"── Shard: {shard_path.name}")

        for name, raw_tensor in raw.items():
            safe_name = name.replace("/", "__").replace(".", "_")
            meta_path = comp_dir / f"{safe_name}.meta.json"
            is_comp   = comp_dir.exists() and meta_path.exists()

            if is_comp:
                weight = _load_compressed_layer(safe_name, comp_dir, device)
                n_comp += 1
                tag = "dequant"
            else:
                weight = raw_tensor.to(device=device, dtype=torch.float32)
                tag = "passthru"

            if target_dtype is not None:
                weight = weight.to(dtype=target_dtype)

            state_dict[name] = weight
            total_params    += weight.numel()
            n_layers        += 1
            print(f"  [{tag:8s}]  {name:60s}  {tuple(weight.shape)}")

    elapsed = time.perf_counter() - t0

    print(f"\n{'═'*68}")
    print(f"  Layers loaded   : {n_layers}  ({n_comp} dequantized, {n_layers-n_comp} passthru)")
    print(f"  Total params    : {total_params:,}")
    print(f"  Wall time       : {elapsed:.2f}s")

    if args.save:
        if not HAS_SAFETENSORS:
            sys.exit("Error: --save requires safetensors.  pip install safetensors")
        from safetensors.torch import save_file
        out_path = Path(args.save)
        # safetensors requires all tensors to share the same dtype
        save_dtype = target_dtype or torch.float32
        sd_cast = {k: v.to(dtype=save_dtype) for k, v in state_dict.items()}
        save_file(sd_cast, out_path)
        print(f"\n  Saved reconstructed checkpoint to: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    _cli_main()