#!/usr/bin/env python3.11
"""Does the SparseCore actually beat the TensorCore at a gather? Measured on v5p, where it compiles.

The study's original question was whether irregular access belongs on the TensorCore or on the
gather engine sitting next to it. On v6e the question could not be asked: every SparseCore Pallas
kernel we tried failed to compile with `'tpu.enqueue_indirect_dma' op Not implemented`. On v5p the
same kernel compiles at some shapes, so the question is finally answerable on real hardware.

Two halves, in order:

    envelope    which (rows, dim, indices, window) shapes compile at all. The failures are a
                result in their own right, because the documentation implies all of them work.
    timing      for every shape that compiles, the SparseCore kernel against `jnp.take` on the
                same chip, same table, same indices.

Two measurement rules the earlier experiments in this repo learned the hard way:

  * A single small kernel launch measures host dispatch, not hardware. Every timing chains INNER
    gathers inside one jit so the per-op cost is dispatch-free, and reports the single-call time
    separately so the difference is visible.
  * Chained gathers must not be common-subexpression-eliminated. Each iteration shifts the index
    vector by `i`, so no two gathers in the chain are the same computation, and the results are
    accumulated into a full-size array so neither path can skip materialising its output.

The accumulator costs both paths one read and one write of the output per iteration, so the
absolute GB/s printed here is a lower bound on the gather's own rate. The SparseCore-to-TensorCore
ratio is unaffected, and the ratio is the answer being sought.

    python3.11 experiment9_sparsecore_v5p.py --out sparsecore_v5p.json
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

REPEATS = 15
WARMUP = 3
INNER = 8                 # gathers chained inside one jit, to amortise host dispatch
HBM_GBS = {"TPU v5": 2765.0, "TPU v5p": 2765.0, "TPU v6 lite": 1638.0, "TPU v6e": 1638.0}

# The shapes to probe. Chosen around the two that were already known to compile
# (dim=128/window=256 and dim=256/window=128, both float32), to find the actual
# constraint rather than to confirm two lucky points.
CASES = [
    # rows,  dim,     n, window, dtype
    (  4096,   8,  1024,   256, jnp.float32),
    (  4096,  32,  1024,   256, jnp.float32),
    (  4096, 128,  1024,   256, jnp.float32),   # known good
    (  4096, 128,  4096,   256, jnp.float32),
    (  4096, 128, 16384,   256, jnp.float32),
    ( 65536, 128, 16384,   256, jnp.float32),
    (262144, 128, 16384,   256, jnp.float32),
    (262144, 128, 65536,   256, jnp.float32),
    (  4096, 128,  1024,   128, jnp.float32),
    (  4096, 128,  1024,    64, jnp.float32),
    (  4096, 128,  2048,   512, jnp.float32),   # 256 KiB out-block, expected to fail
    (  4096, 256,  1024,   128, jnp.float32),   # known good
    (  4096, 256,  4096,   128, jnp.float32),
    (  4096, 512,  1024,    64, jnp.float32),
    (  4096, 512,  1024,   128, jnp.float32),   # 256 KiB out-block, expected to fail
    (  4096, 128,  1024,   256, jnp.bfloat16),  # expected to fail: dtype
]


def timed(fn, *args) -> float:
    """Median wall time of a jitted call in seconds, compile excluded."""
    jax.block_until_ready(fn(*args))
    for _ in range(WARMUP):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def tensorcore(n: int, dim: int, rows: int, inner: int):
    """What XLA gives you for free, chained the same way as the kernel below."""
    @jax.jit
    def f(table, idx):
        acc = jnp.zeros((n, dim), table.dtype)
        for i in range(inner):
            acc = acc + jnp.take(table, (idx + i) % rows, axis=0)
        return acc
    return f


def sparsecore(n: int, dim: int, rows: int, inner: int):
    """The documented SparseCore gather: an indirect DMA per window, pipelined over subcores."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.experimental.pallas import tpu_sc as plsc

    mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

    def build(window: int):
        @jax.jit
        def f(table, idx):
            @pl.kernel(out_type=jax.ShapeDtypeStruct((n, dim), table.dtype), mesh=mesh)
            def kernel(x_hbm, i_hbm, o_hbm):
                def body(i_vmem, o_vmem):
                    pltpu.sync_copy(x_hbm.at[i_vmem.at[0]], o_vmem)

                pltpu.emit_pipeline(
                    body,
                    grid=(n // window,),
                    in_specs=[pl.BlockSpec((1, window), index_map=lambda i: (0, i))],
                    out_specs=[pl.BlockSpec((window, dim), index_map=lambda i: (i, 0))],
                    core_axis_name="subcore",
                    dimension_semantics=(pltpu.PARALLEL,),
                )(i_hbm, o_hbm)

            acc = jnp.zeros((n, dim), table.dtype)
            for i in range(inner):
                acc = acc + kernel(table, ((idx + i) % rows).reshape((1, n)))
            return acc
        return f
    return build


def one_case(rows: int, dim: int, n: int, window: int, dtype, peak_gbs: float) -> dict:
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    table = jax.random.normal(k1, (rows, dim), dtype)
    idx = jax.random.randint(k2, (n,), 0, rows, dtype=jnp.int32)
    itemsize = jnp.dtype(dtype).itemsize
    delivered = n * dim * itemsize            # bytes the gather must produce, per op

    rec = {"rows": rows, "dim": dim, "indices": n, "window": window, "dtype": dtype.__name__,
           "table_mib": round(rows * dim * itemsize / 2**20, 2),
           "delivered_mib": round(delivered / 2**20, 3),
           "out_block_kib": round(window * dim * itemsize / 1024, 1)}

    tc_chain = timed(tensorcore(n, dim, rows, INNER), table, idx) / INNER
    tc_one = timed(tensorcore(n, dim, rows, 1), table, idx)
    rec["tc_ms"] = round(tc_chain * 1e3, 4)
    rec["tc_ms_with_dispatch"] = round(tc_one * 1e3, 4)
    rec["tc_gbs"] = round(delivered / tc_chain / 1e9, 2)
    rec["tc_pct_hbm"] = round(delivered / tc_chain / 1e9 / peak_gbs * 100, 3)

    try:
        build = sparsecore(n, dim, rows, INNER)
        fn = build(window)
        sc_chain = timed(fn, table, idx) / INNER
        sc_one = timed(sparsecore(n, dim, rows, 1)(window), table, idx)
        rec["sc_ms"] = round(sc_chain * 1e3, 4)
        rec["sc_ms_with_dispatch"] = round(sc_one * 1e3, 4)
        rec["sc_gbs"] = round(delivered / sc_chain / 1e9, 2)
        rec["sc_pct_hbm"] = round(delivered / sc_chain / 1e9 / peak_gbs * 100, 3)
        rec["speedup"] = round(tc_chain / sc_chain, 3)
        # A fast wrong kernel is worth nothing.
        want = jnp.take(table, idx, axis=0)
        single = sparsecore(n, dim, rows, 1)(window)(table, idx)
        rec["max_abs_err"] = float(jnp.max(jnp.abs(want - single)))
        rec["correct"] = bool(rec["max_abs_err"] == 0.0)
    except Exception as exc:
        rec["sc_error"] = f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')}"[:300]
    return rec


def convergence(rows: int, dim: int, n: int, window: int) -> list[dict]:
    """Is INNER=8 enough to have amortised dispatch? Sweep it and watch the per-op cost settle."""
    out = []
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    table = jax.random.normal(k1, (rows, dim), jnp.float32)
    idx = jax.random.randint(k2, (n,), 0, rows, dtype=jnp.int32)
    for inner in (1, 2, 4, 8, 16, 32):
        row = {"inner": inner}
        row["tc_ms"] = round(timed(tensorcore(n, dim, rows, inner), table, idx) / inner * 1e3, 4)
        try:
            row["sc_ms"] = round(
                timed(sparsecore(n, dim, rows, inner)(window), table, idx) / inner * 1e3, 4)
        except Exception as exc:
            row["sc_error"] = f"{type(exc).__name__}"
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sparsecore_v5p.json")
    args = ap.parse_args()

    dev = jax.devices()[0]
    meta = {"jax": jax.__version__, "device_kind": dev.device_kind,
            "num_devices": jax.device_count(), "python": platform.python_version(),
            "host": platform.node(), "inner": INNER, "repeats": REPEATS}
    peak = HBM_GBS.get(dev.device_kind, 2765.0)
    meta["hbm_gbs_assumed"] = peak
    try:
        from jax.experimental.pallas import tpu_sc as plsc
        meta["sparsecore"] = str(plsc.get_sparse_core_info())
    except Exception as exc:
        meta["sparsecore"] = f"unavailable: {exc}"
    print(json.dumps(meta, indent=2), flush=True)

    out = Path(args.out)
    records = []
    for case in CASES:
        rec = one_case(*case, peak_gbs=peak)
        records.append(rec)
        out.write_text(json.dumps({"meta": meta, "records": records}, indent=2))
        head = (f"dim={rec['dim']:4d} n={rec['indices']:6d} win={rec['window']:4d} "
                f"rows={rec['rows']:7d} {rec['dtype']:9s} out_blk={rec['out_block_kib']:6.0f}KiB")
        if "sc_gbs" in rec:
            flag = "" if rec["correct"] else f"  WRONG err={rec['max_abs_err']:.2e}"
            print(f"{head}  TC {rec['tc_gbs']:8.2f} GB/s  SC {rec['sc_gbs']:8.2f} GB/s  "
                  f"x{rec['speedup']:6.2f}{flag}", flush=True)
        else:
            print(f"{head}  TC {rec['tc_gbs']:8.2f} GB/s  SC failed: "
                  f"{rec['sc_error'][:70]}", flush=True)

    print("\nconvergence check, dim=128 n=4096 win=256:", flush=True)
    conv = convergence(65536, 128, 4096, 256)
    for row in conv:
        print(f"  inner={row['inner']:3d}  TC {row['tc_ms']:8.4f} ms/op  "
              f"SC {row.get('sc_ms', float('nan')):8.4f} ms/op", flush=True)

    ok = [r for r in records if "sc_gbs" in r]
    payload = {"meta": meta, "records": records, "convergence": conv}
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n{len(ok)}/{len(records)} shapes compiled on the SparseCore; wrote {out}")
    if ok:
        best = max(ok, key=lambda r: r["speedup"])
        print(f"best SparseCore advantage: x{best['speedup']:.2f} at dim={best['dim']} "
              f"n={best['indices']} window={best['window']} "
              f"({best['sc_gbs']:.1f} vs {best['tc_gbs']:.1f} GB/s, "
              f"{best['sc_pct_hbm']:.2f}% of HBM)")


if __name__ == "__main__":
    main()
