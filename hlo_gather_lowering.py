#!/usr/bin/env python3.11
"""Why does a gather get 3.3x slower when the buffer crosses a size threshold? Read the HLO.

experiment10 measured a cliff: on v5p a gather runs at 256 GB/s out of a 40 MiB buffer and 77 GB/s
out of a 48 MiB one, flat on both sides, and independent of which addresses the indices touch. On
v6e the same cliff sits between 80 and 96 MiB. Correlations are where this kind of investigation
usually stops. It does not have to stop here, because XLA will hand over the compiled program.

This compiles the identical gather at a buffer just below and just above the threshold and diffs the
optimised HLO. If the compiler is choosing a different lowering, the diff says so in words.

Two further questions the same tool answers cheaply:

    is the threshold in bytes or in rows?   compile at dim=128 and dim=256 and see whether the
                                           cliff stays at the same MiB or moves to half of it
    does it depend on the index count?      compile at n=16384 and n=1024

    python3.11 hlo_gather_lowering.py --out hlo_lowering.json
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp

N = 16384
INNER = 1          # one gather, so the HLO is small enough to read


def compiled_text(rows: int, dim: int, n: int = N) -> str:
    table = jax.ShapeDtypeStruct((rows, dim), jnp.float32)
    idx = jax.ShapeDtypeStruct((n,), jnp.int32)
    fn = jax.jit(lambda t, i: jnp.take(t, i, axis=0))
    return fn.lower(table, idx).compile().as_text()


def fingerprint(text: str) -> dict:
    """The handful of things in an HLO module that distinguish one gather lowering from another."""
    ops = re.findall(r"=\s*\S+\s+([a-z][a-z0-9-]*)\(", text)
    counts: dict[str, int] = {}
    for op in ops:
        counts[op] = counts.get(op, 0) + 1
    interesting = {k: v for k, v in counts.items() if k in {
        "gather", "dynamic-slice", "fusion", "while", "dot", "convert", "copy", "bitcast",
        "custom-call", "dynamic-update-slice", "iota", "compare", "select", "reduce", "transpose",
        "all-gather", "concatenate", "reshape", "broadcast"}}
    return {"total_ops": len(ops), "lines": text.count("\n") + 1,
            "ops": dict(sorted(interesting.items(), key=lambda kv: -kv[1])),
            "has_while_loop": "while(" in text or "while_body" in text,
            "windowed": "windowed" in text.lower(),
            "bytes_in_text": len(text)}


def promoted(rows: int, dim: int, n: int = N) -> bool:
    """Did XLA put the gather's source table in memory space S(1), i.e. on chip?

    This is the whole mechanism behind the cliff, and it is visible in the compiled module without
    running anything. So the exact threshold can be bisected for the price of a few compiles.
    """
    text = compiled_text(rows, dim, n)
    return f"f32[{rows},{dim}]" in text and re.search(
        rf"%param_0\.\d+ = f32\[{rows},{dim}\]\{{[^}}]*S\(1\)\}} parameter\(0\)", text) is not None


def param_line(rows: int, dim: int) -> str:
    """The parameter-0 declaration, so an odd bisection result can be told from a real one."""
    for line in compiled_text(rows, dim).splitlines():
        if "parameter(0)" in line and f"[{rows},{dim}]" in line:
            return line.strip()[:150]
    return "(no parameter(0) line mentioning that shape)"


def bisect_threshold(dim: int, lo_rows: int = 8, hi_rows: int = 1 << 21, n: int = N) -> dict:
    """Largest table still promoted to S(1), by binary search over the row count.

    Rows are kept a multiple of 8 because that is the tile height for f32, so every candidate is a
    shape XLA lays out the same way.
    """
    step = 8
    lo, hi = lo_rows, hi_rows
    if not promoted(lo, dim, n):
        return {"dim": dim, "n": n, "error": f"even {lo} rows is not promoted",
                "param_line": param_line(lo, dim)}
    if promoted(hi, dim, n):
        return {"dim": dim, "n": n, "error": f"{hi} rows is still promoted"}
    while hi - lo > step:
        mid = ((lo + hi) // 2 // step) * step
        mid = max(lo + step, min(mid, hi - step))
        if promoted(mid, dim, n):
            lo = mid
        else:
            hi = mid
    return {"dim": dim, "n": n, "last_promoted_rows": lo, "first_hbm_rows": hi,
            "last_promoted_bytes": lo * dim * 4, "first_hbm_bytes": hi * dim * 4,
            "last_promoted_mib": round(lo * dim * 4 / 2**20, 3),
            "first_hbm_mib": round(hi * dim * 4 / 2**20, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hlo_lowering.json")
    ap.add_argument("--dump", action="store_true", help="write both HLO modules to disk")
    ap.add_argument("--bisect", action="store_true",
                    help="find the exact S(1) promotion threshold, compile-only")
    args = ap.parse_args()

    if args.bisect:
        kind = jax.devices()[0].device_kind
        print(f"{kind}: bisecting the S(1) promotion threshold, no kernels run\n", flush=True)
        rows = []
        for dim in (64, 128, 256, 512):
            r = bisect_threshold(dim)
            rows.append(r)
            if "error" in r:
                print(f"dim={dim:4d}  {r['error']}", flush=True)
                print(f"          {r.get('param_line', '')}", flush=True)
            else:
                print(f"dim={dim:4d}  promoted up to {r['last_promoted_rows']:8d} rows "
                      f"({r['last_promoted_mib']:8.3f} MiB = {r['last_promoted_bytes']:>11,} B), "
                      f"HBM from {r['first_hbm_rows']:8d} rows", flush=True)
        good = [r for r in rows if "last_promoted_bytes" in r]
        byts = {r["last_promoted_bytes"] for r in good}
        if len(good) > 1:
            print(f"\nthreshold constant in BYTES across row widths: {len(byts) == 1}")
            print(f"  byte thresholds seen: {sorted(byts)}")

        # The prediction. If one on-chip budget holds the table AND the index vector, then growing
        # the index vector must shrink the table's allowance by exactly the bytes it added.
        print(f"\nprediction test: budget = table + indices, so threshold + 4n should be constant",
              flush=True)
        idx_rows = []
        for n in (1024, 4096, 16384, 65536):
            r = bisect_threshold(128, n=n)
            if "last_promoted_bytes" not in r:
                print(f"  n={n:6d}  {r['error']}", flush=True)
                continue
            total = r["last_promoted_bytes"] + n * 4
            r["index_bytes"] = n * 4
            r["table_plus_indices"] = total
            idx_rows.append(r)
            print(f"  n={n:6d}  indices {n * 4:>9,} B  table {r['last_promoted_bytes']:>12,} B  "
                  f"sum {total:>12,} B  ({total / 2**20:.4f} MiB)", flush=True)
        if len(idx_rows) > 1:
            sums = {r["table_plus_indices"] for r in idx_rows}
            print(f"\n  sum constant across a 64x range of index count: {len(sums) == 1}")
            if len(sums) == 1:
                b = sums.pop()
                print(f"  the budget is {b:,} bytes = {b / 2**20:.4f} MiB = "
                      f"{b / 1024:,.0f} KiB. Confirmed.")
            else:
                print(f"  sums seen: {sorted(sums)}; spread "
                      f"{max(sums) - min(sums):,} B, so something else is in the budget too.")

        Path(args.out).write_text(json.dumps(
            {"device_kind": kind, "jax": jax.__version__, "thresholds": rows,
             "index_scaling": idx_rows}, indent=2))
        print(f"wrote {args.out}")
        return

    dev = jax.devices()[0]
    kind = dev.device_kind
    # Buffers straddling each generation's measured threshold, in rows at dim=128 float32.
    below, above = (81920, 98304) if "v5" in kind else (163840, 196608)
    print(f"{kind}: comparing {below * 128 * 4 / 2**20:.0f} MiB (fast) against "
          f"{above * 128 * 4 / 2**20:.0f} MiB (slow)\n", flush=True)

    lo, hi = compiled_text(below, 128), compiled_text(above, 128)
    flo, fhi = fingerprint(lo), fingerprint(hi)
    print(f"below threshold: {flo['total_ops']:4d} ops, {flo['lines']:4d} lines, "
          f"while-loop={flo['has_while_loop']}")
    print(f"  {flo['ops']}")
    print(f"above threshold: {fhi['total_ops']:4d} ops, {fhi['lines']:4d} lines, "
          f"while-loop={fhi['has_while_loop']}")
    print(f"  {fhi['ops']}")
    same = flo["ops"] == fhi["ops"] and flo["total_ops"] == fhi["total_ops"]
    print(f"\nsame op mix on both sides of the cliff: {same}")
    if same:
        print("  So the 3.3x is NOT a different HLO lowering. Whatever changes is below HLO:")
        print("  the LLO/Mosaic schedule, a tiling choice, or a runtime decision.")

    # Is the threshold counted in bytes or in rows? At dim=256 the same row count is twice the
    # bytes, so a byte threshold halves the row count at which the cliff appears.
    probes = []
    for dim in (128, 256):
        for rows in (below, above):
            t = fingerprint(compiled_text(rows, dim))
            probes.append({"dim": dim, "rows": rows,
                           "mib": round(rows * dim * 4 / 2**20, 1),
                           "total_ops": t["total_ops"], "ops": t["ops"]})

    diff = list(difflib.unified_diff(lo.splitlines(), hi.splitlines(),
                                     "below_threshold", "above_threshold", n=1, lineterm=""))
    trimmed = [d for d in diff if not d.startswith(("---", "+++", "@@"))][:60]
    print(f"\nHLO text diff: {len([d for d in diff if d.startswith(('+', '-'))])} changed lines")
    for line in trimmed[:24]:
        print("  " + line[:150])

    payload = {"device_kind": kind, "jax": jax.__version__,
               "below_rows": below, "above_rows": above,
               "below": flo, "above": fhi, "identical_op_mix": same,
               "dim_probes": probes, "diff_sample": trimmed}
    Path(args.out).write_text(json.dumps(payload, indent=2))
    if args.dump:
        Path("hlo_below.txt").write_text(lo)
        Path("hlo_above.txt").write_text(hi)
        print("\nwrote hlo_below.txt and hlo_above.txt")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
