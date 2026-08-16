#!/usr/bin/env python3.11
"""What exactly makes a TPU gather slow? Allocation size, touched span, or index order?

experiment9 found a cliff. A gather of 16384 rows of 128 float32 runs at ~190 GB/s out of a 32 MiB
table and ~70 GB/s out of a 128 MiB table, with the index-reuse ratio held constant. The same
cliff appears on both generations at the same factor:

    v5p   192.5 -> 69.4 GB/s   2.77x
    v6e   160.5 -> 58.0 GB/s   2.77x

An identical factor on two different chips rules out the obvious explanation, that v5p's on-chip
CMEM was absorbing the small table and v6e's smaller cache was not. Something structural is going
on instead, and there are three candidates tangled together in that comparison:

    allocation   how large the table buffer is, independent of what gets read
    span         how wide an address range the indices actually touch
    order        whether the indices walk that range in ascending order or at random

This separates them. The table is allocated at 262144 rows (128 MiB) for every measurement, and
only the indices change, so allocation is held fixed while span varies. Two extra configurations
allocate a table exactly as large as the span, which is the control: if a 32 MiB span reads at the
same rate out of a 128 MiB table as out of a 32 MiB table, allocation is irrelevant and the answer
is span.

A contiguous read of the same byte count is measured alongside, as the reference every gather is
losing against. Both paths carry the same accumulator, so the gather-to-contiguous ratio is free of
the accumulator's own traffic in a way that an absolute GB/s figure is not.

    python3.11 experiment10_gather_locality.py --out gather_locality.json
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPEATS = 12
WARMUP = 3
INNER = 32                # experiment9's convergence sweep showed 8 was still 2x off the asymptote
ALLOC = 262144            # rows; 128 MiB at dim=128 float32
DIM = 128
N = 16384                 # indices per gather; 8 MiB delivered per op
WINDOW = 256              # the SparseCore out-block that compiles on both generations at dim=128
HBM_GBS = {"TPU v5": 2765.0, "TPU v5p": 2765.0, "TPU v6 lite": 1638.0, "TPU v6e": 1638.0}

# (alloc_rows, span_rows, order). Allocation is fixed at ALLOC except for the two controls,
# which shrink the buffer to exactly the span to test whether allocation matters at all.
CONFIGS = [
    (ALLOC,   4096, "random"),
    (ALLOC,  16384, "random"),
    (ALLOC,  65536, "random"),
    (ALLOC, 262144, "random"),
    (ALLOC, 262144, "sorted"),
    (ALLOC, 262144, "blocked"),
    (65536,  65536, "random"),   # control: same span as row 3, smaller buffer
    (4096,    4096, "random"),   # control: same span as row 1, smaller buffer
]


# --sweep replaces the configuration list above with an allocation ladder, span always equal to
# allocation and the order always random. The first run showed the cliff sits between a 32 MiB and a
# 128 MiB buffer; this finds where, and a threshold that lands on a round number of bytes is
# evidence for a compiler heuristic rather than anything about the memory system.
SWEEP_ROWS = [16384, 32768, 49152, 65536, 81920, 98304, 131072, 163840, 196608, 262144, 393216]


def timed(fn, *args) -> float:
    jax.block_until_ready(fn(*args))
    for _ in range(WARMUP):
        jax.block_until_ready(fn(*args))
    s = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        s.append(time.perf_counter() - t0)
    return statistics.median(s)


def make_indices(span: int, order: str, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if order == "random":
        idx = rng.integers(0, span, size=N, dtype=np.int32)
    elif order == "sorted":
        idx = np.sort(rng.integers(0, span, size=N, dtype=np.int32))
    elif order == "blocked":
        # what a batch sorted by expert looks like: random blocks of consecutive rows
        block = 32
        starts = rng.integers(0, max(1, span - block), size=N // block, dtype=np.int32)
        idx = np.concatenate([np.arange(s, s + block, dtype=np.int32) for s in starts])
    else:
        raise ValueError(order)
    return idx.astype(np.int32)


def tc_gather(span: int, inner: int):
    @jax.jit
    def f(table, idx):
        acc = jnp.zeros((N, DIM), table.dtype)
        for i in range(inner):
            acc = acc + jnp.take(table, (idx + i) % span, axis=0)
        return acc
    return f


def sc_gather(span: int, inner: int, window: int = WINDOW):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.experimental.pallas import tpu_sc as plsc

    mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

    @jax.jit
    def f(table, idx):
        @pl.kernel(out_type=jax.ShapeDtypeStruct((N, DIM), table.dtype), mesh=mesh)
        def kernel(x_hbm, i_hbm, o_hbm):
            def body(i_vmem, o_vmem):
                pltpu.sync_copy(x_hbm.at[i_vmem.at[0]], o_vmem)

            pltpu.emit_pipeline(
                body,
                grid=(N // window,),
                in_specs=[pl.BlockSpec((1, window), index_map=lambda i: (0, i))],
                out_specs=[pl.BlockSpec((window, DIM), index_map=lambda i: (i, 0))],
                core_axis_name="subcore",
                dimension_semantics=(pltpu.PARALLEL,),
            )(i_hbm, o_hbm)

        acc = jnp.zeros((N, DIM), table.dtype)
        for i in range(inner):
            acc = acc + kernel(table, ((idx + i) % span).reshape((1, N)))
        return acc
    return f


def contiguous(alloc: int, inner: int):
    """The reference: the same byte count read as one block, same accumulator, same chip."""
    stride = max(1, (alloc - N) // max(1, inner))

    @jax.jit
    def f(table):
        acc = jnp.zeros((N, DIM), table.dtype)
        for i in range(inner):
            acc = acc + jax.lax.dynamic_slice(table, (i * stride, 0), (N, DIM))
        return acc
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gather_locality.json")
    ap.add_argument("--inner", type=int, default=INNER)
    ap.add_argument("--sweep", action="store_true",
                    help="sweep the allocation size instead of the span/order grid")
    args = ap.parse_args()
    inner = args.inner
    configs = ([(r, r, "random") for r in SWEEP_ROWS] if args.sweep else CONFIGS)

    dev = jax.devices()[0]
    peak = HBM_GBS.get(dev.device_kind, 2765.0)
    meta = {"jax": jax.__version__, "device_kind": dev.device_kind, "host": platform.node(),
            "inner": inner, "repeats": REPEATS, "dim": DIM, "indices": N, "window": WINDOW,
            "alloc_rows": ALLOC, "hbm_gbs_assumed": peak,
            "delivered_mib": round(N * DIM * 4 / 2**20, 3)}
    try:
        from jax.experimental.pallas import tpu_sc as plsc
        meta["sparsecore"] = str(plsc.get_sparse_core_info())
    except Exception as exc:
        meta["sparsecore"] = f"unavailable: {exc}"
    print(json.dumps(meta, indent=2), flush=True)

    delivered = N * DIM * 4
    big = jax.random.normal(jax.random.key(0), (ALLOC, DIM), jnp.float32)

    ref = timed(contiguous(ALLOC, inner), big) / inner
    ref_gbs = delivered / ref / 1e9
    print(f"\ncontiguous reference: {ref * 1e3:.4f} ms/op  {ref_gbs:8.2f} GB/s  "
          f"{ref_gbs / peak * 100:5.2f}% of HBM  (same accumulator as every gather below)\n",
          flush=True)

    records = []
    for alloc, span, order in configs:
        table = big if alloc == ALLOC else jax.random.normal(
            jax.random.key(1), (alloc, DIM), jnp.float32)
        idx = jnp.asarray(make_indices(span, order))
        rec = {"alloc_rows": alloc, "span_rows": span, "order": order,
               "alloc_mib": round(alloc * DIM * 4 / 2**20, 1),
               "span_mib": round(span * DIM * 4 / 2**20, 1)}

        tc = timed(tc_gather(span, inner), table, idx) / inner
        rec["tc_ms"] = round(tc * 1e3, 4)
        rec["tc_gbs"] = round(delivered / tc / 1e9, 2)
        rec["tc_pct_hbm"] = round(delivered / tc / 1e9 / peak * 100, 3)
        rec["tc_vs_contiguous"] = round((delivered / tc / 1e9) / ref_gbs, 3)

        try:
            fn = sc_gather(span, inner)
            sc = timed(fn, table, idx) / inner
            rec["sc_ms"] = round(sc * 1e3, 4)
            rec["sc_gbs"] = round(delivered / sc / 1e9, 2)
            rec["sc_pct_hbm"] = round(delivered / sc / 1e9 / peak * 100, 3)
            rec["sc_vs_contiguous"] = round((delivered / sc / 1e9) / ref_gbs, 3)
            rec["sc_over_tc"] = round(tc / sc, 3)
            want = jnp.take(table, idx, axis=0)
            got = sc_gather(span, 1)(table, idx)
            rec["max_abs_err"] = float(jnp.max(jnp.abs(want - got)))
            rec["correct"] = bool(rec["max_abs_err"] == 0.0)
        except Exception as exc:
            rec["sc_error"] = f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')}"[:200]

        records.append(rec)
        Path(args.out).write_text(json.dumps(
            {"meta": meta, "contiguous": {"ms": round(ref * 1e3, 4), "gbs": round(ref_gbs, 2)},
             "records": records}, indent=2))
        tail = (f"SC {rec['sc_gbs']:8.2f} GB/s  x{rec['sc_over_tc']:5.2f} vs TC"
                if "sc_gbs" in rec else f"SC failed: {rec['sc_error'][:44]}")
        print(f"alloc {rec['alloc_mib']:6.1f} MiB  span {rec['span_mib']:6.1f} MiB  "
              f"{order:9s}  TC {rec['tc_gbs']:8.2f} GB/s "
              f"({rec['tc_vs_contiguous'] * 100:5.1f}% of contiguous)  {tail}", flush=True)

    if args.sweep:
        # Locate the cliff: the largest buffer still on the fast path, and the smallest on the slow.
        fast = [r for r in records if r["tc_gbs"] > 150]
        slow = [r for r in records if r["tc_gbs"] <= 150]
        if fast and slow:
            lo = max(r["alloc_mib"] for r in fast)
            hi = min(r["alloc_mib"] for r in slow)
            print(f"\nTensorCore gather cliff is between a {lo:.0f} MiB and a {hi:.0f} MiB buffer.",
                  flush=True)
            print(f"  fast path {statistics.median(r['tc_gbs'] for r in fast):.1f} GB/s, "
                  f"slow path {statistics.median(r['tc_gbs'] for r in slow):.1f} GB/s, "
                  f"a {statistics.median(r['tc_gbs'] for r in fast) / statistics.median(r['tc_gbs'] for r in slow):.2f}x step")
        else:
            print("\nno cliff inside the swept range: every buffer took the same path.", flush=True)
        sc = [r["sc_gbs"] for r in records if "sc_gbs" in r]
        if sc:
            print(f"  SparseCore over the same range: {min(sc):.1f} to {max(sc):.1f} GB/s "
                  f"({(max(sc) / min(sc) - 1) * 100:.1f}% spread), i.e. buffer size is irrelevant to it.")
        return

    print("\nreading this table:", flush=True)
    print("  rows 1-4 hold allocation at 128 MiB and widen the span. If bandwidth tracks span,")
    print("  the cliff is about the address range touched, not the buffer.")
    print("  rows 7-8 repeat spans 32 MiB and 2 MiB with a buffer exactly that size. If they")
    print("  match rows 3 and 1, allocation is irrelevant and span is the whole story.")
    print("  rows 5-6 keep the span at 128 MiB and change only the order of the indices.")


if __name__ == "__main__":
    main()
