#!/usr/bin/env python3
"""What actually fits, parametrically, with the KV cache counted rather than waved at.

The first version of this arithmetic reserved a flat 25% of HBM for "KV cache, activations and
scratch", which is the kind of hand-wave that hides an order of magnitude. KV cache is not a
percentage: it is 2 x layers x kv_heads x head_dim x dtype x context x batch, and at 262k context
it can exceed the weights. So it gets computed.

Two directions of use:

    what fits now      given v6e at 32 GB per chip and the slices we can actually get
    what fits later    the same models against Ironwood at 192 GB and TPU 8i at 288 GB, which is
                       how to tell whether a model is permanently out of reach or just out of
                       reach for us

    python3 capacity_calculator.py                      # the table
    python3 capacity_calculator.py --chips 64 --context 32768
    python3 capacity_calculator.py --json out.json      # machine-readable, for the logbook
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GB = 1e9

# HBM per chip, in GB. v6e measured on the chip; the rest from vendor specifications.
CHIPS = {"v5e": 16, "v6e": 32, "v7x-ironwood": 192, "8i-zebrafish": 288}

# Model geometry. kv_heads matters far more than total heads: grouped-query attention is the
# reason a 671B model can be served at all. full_attn_layers is for hybrid models where only some
# blocks keep quadratic attention, which is the entire point of Qwen3.8's Gated DeltaNet design.
#   (name, params, active_params, layers, full_attn_layers, kv_heads, head_dim)
MODELS = [
    ("Qwen3-4B",              4.0e9,   4.0e9,  36, 36,  8, 128),
    ("Qwen3.8-27B",          27.0e9,  27.0e9,  64, 16, 24, 128),
    ("Qwen3-30B-A3B",        30.5e9,   3.3e9,  48, 48,  4, 128),
    ("Llama-3.3-70B",        70.0e9,  70.0e9,  80, 80,  8, 128),
    ("Qwen3-235B-A22B",     235.0e9,  22.0e9,  94, 94,  4, 128),
    ("DeepSeek-V3 class",   671.0e9,  37.0e9,  61, 61,  8, 128),
    ("Kimi K2",               1.0e12,  32.0e9,  61, 61,  8, 128),
    ("Kimi K3",               2.8e12,  32.0e9,  80, 80,  8, 128),
]

RESERVE = 0.10  # compile scratch and fragmentation, on top of computed weights and KV


def weights_gb(params: float, bits: int) -> float:
    return params * (bits / 8) / GB


def kv_gb(layers: int, full_attn: int, kv_heads: int, head_dim: int,
          context: int, batch: int, bits: int) -> float:
    """Only the full-attention layers hold a growing KV cache."""
    per_token = 2 * full_attn * kv_heads * head_dim * (bits / 8)
    return per_token * context * batch / GB


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", default="v6e", choices=list(CHIPS))
    ap.add_argument("--chips", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--kv-bits", type=int, default=16)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    per_chip = CHIPS[args.chip]
    rows = []
    print(f"chip {args.chip} at {per_chip} GB | context {args.context:,} | batch {args.batch} | "
          f"KV in {args.kv_bits}-bit | {int(RESERVE*100)}% reserved for scratch\n")
    header = f"{'model':22s} {'params':>7s} {'act':>6s} {'w bf16':>8s} {'w fp8':>7s} {'KV':>8s}"
    for n in args.chips:
        header += f" {str(n) + ' chips':>16s}"
    print(header)
    print("-" * len(header))

    for name, params, active, layers, full_attn, kv_heads, head_dim in MODELS:
        w16, w8 = weights_gb(params, 16), weights_gb(params, 8)
        kv = kv_gb(layers, full_attn, kv_heads, head_dim, args.context, args.batch, args.kv_bits)
        rec = {"model": name, "params": params, "active_params": active,
               "weights_bf16_gb": round(w16, 1), "weights_fp8_gb": round(w8, 1),
               "kv_gb": round(kv, 1), "chip": args.chip, "context": args.context,
               "batch": args.batch, "fits": {}}
        line = (f"{name:22s} {params/1e9:6.0f}B {active/1e9:5.0f}B "
                f"{w16:7.0f}G {w8:6.0f}G {kv:7.1f}G")
        for n in args.chips:
            usable = n * per_chip * (1 - RESERVE)
            if w16 + kv <= usable:
                v, pct = "bf16", (w16 + kv) / (n * per_chip) * 100
            elif w8 + kv <= usable:
                v, pct = "fp8", (w8 + kv) / (n * per_chip) * 100
            else:
                need = (w8 + kv) / (per_chip * (1 - RESERVE))
                v, pct = f"no, {need:.0f}ch", 0.0
            rec["fits"][str(n)] = {"verdict": v, "hbm_percent": round(pct, 1)}
            line += f" {v + (f' {pct:.0f}%' if pct else ''):>16s}"
        rows.append(rec)
        print(line)

    print(f"\nKV cache is {rows[5]['kv_gb']:.0f} GB for the 671B model at this context and batch, "
          f"which is {rows[5]['kv_gb']/rows[5]['weights_fp8_gb']*100:.0f}% of its fp8 weights.")
    print("Raise --context to see why 8i puts 384 MB of SRAM on the chip.")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"chip": args.chip, "per_chip_gb": per_chip, "context": args.context,
             "batch": args.batch, "kv_bits": args.kv_bits, "reserve": RESERVE,
             "rows": rows}, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
