#!/usr/bin/env python3.11
"""Where does irregular access belong: the TensorCore, or the SparseCore next to it?

One question, measured rather than argued. A gather of `num_indices` rows out of a
`num_rows x value_dim` table, run two ways on the same v6e chip:

    TensorCore   jnp.take, which is what XLA gives you for free
    SparseCore   a Pallas kernel over plsc.VectorSubcoreMesh, per the JAX docs

Sweeping the row width is the whole point. A one-element row is pure pointer chasing and should
favour the SparseCore badly; a 512-element row is really a strided copy and should favour whoever
has the bandwidth. Somewhere between them is a crossover, and that crossover is the number a
performance model needs and does not currently have for TPUs.

Writes a JSON record per configuration so the results survive a preemption.

    python3.11 bench_gather.py --out results.json
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

REPEATS = 20
WARMUP = 3


def timed(fn, *args) -> float:
    """Median wall time of a jitted call, in seconds, with the compile excluded."""
    out = fn(*args)
    jax.block_until_ready(out)
    for _ in range(WARMUP):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def tensorcore_gather(value_dim: int):
    if value_dim == 1:
        return jax.jit(lambda table, idx: jnp.take(table, idx, axis=0))
    return jax.jit(lambda table, idx: jnp.take(table, idx, axis=0))


def sparsecore_gather(num_indices: int, value_dim: int, window: int):
    """The doc's gather kernel, closed over its shapes."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.experimental.pallas import tpu_sc as plsc

    mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

    @jax.jit
    def gather(table, idx):
        idx2 = idx.reshape((1, num_indices))

        @pl.kernel(out_type=jax.ShapeDtypeStruct((num_indices, value_dim), table.dtype),
                   mesh=mesh)
        def kernel(x_hbm, i_hbm, o_hbm):
            def body(i_vmem, o_vmem):
                pltpu.sync_copy(x_hbm.at[i_vmem.at[0]], o_vmem)

            pltpu.emit_pipeline(
                body,
                grid=(num_indices // window,),
                in_specs=[pl.BlockSpec((1, window), index_map=lambda i: (0, i))],
                out_specs=[pl.BlockSpec((window, value_dim), index_map=lambda i: (i, 0))],
                core_axis_name="subcore",
                dimension_semantics=(pltpu.PARALLEL,),
            )(i_hbm, o_hbm)

        return kernel(table, idx2)

    return gather


def run_case(num_rows: int, value_dim: int, num_indices: int, window: int) -> dict:
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    table = jax.random.normal(k1, (num_rows, value_dim), jnp.float32)
    idx = jax.random.randint(k2, (num_indices,), 0, num_rows, dtype=jnp.int32)
    delivered = num_indices * value_dim * 4  # bytes the gather actually has to produce

    record = {
        "num_rows": num_rows, "value_dim": value_dim, "num_indices": num_indices,
        "window": window, "table_mib": round(num_rows * value_dim * 4 / 2**20, 2),
        "delivered_mib": round(delivered / 2**20, 3),
    }

    tc = timed(tensorcore_gather(value_dim), table, idx)
    record["tensorcore_ms"] = round(tc * 1e3, 4)
    record["tensorcore_gbs"] = round(delivered / tc / 1e9, 3)

    try:
        sc_fn = sparsecore_gather(num_indices, value_dim, window)
        sc = timed(sc_fn, table, idx)
        record["sparsecore_ms"] = round(sc * 1e3, 4)
        record["sparsecore_gbs"] = round(delivered / sc / 1e9, 3)
        record["speedup"] = round(tc / sc, 3)
        # Correctness is not optional: a fast wrong kernel is worth nothing.
        want = jnp.take(table, idx, axis=0)
        got = sc_fn(table, idx)
        record["max_abs_err"] = float(jnp.max(jnp.abs(want - got)))
        record["correct"] = bool(record["max_abs_err"] == 0.0)
    except Exception as exc:  # keep the TensorCore half of the dataset either way
        record["sparsecore_error"] = f"{type(exc).__name__}: {exc}"[:400]

    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--rows", type=int, default=1 << 20)
    args = ap.parse_args()

    device = jax.devices()[0]
    meta = {
        "jax": jax.__version__, "device_kind": device.device_kind,
        "num_devices": jax.device_count(), "python": platform.python_version(),
        "host": platform.node(),
    }
    try:
        from jax.experimental.pallas import tpu_sc as plsc
        meta["sparsecore"] = str(plsc.get_sparse_core_info())
    except Exception as exc:
        meta["sparsecore"] = f"unavailable: {exc}"
    print(json.dumps(meta, indent=2))

    out = Path(args.out)
    records = []
    # Row width is the axis that should decide the winner. Index count is swept too, because a
    # gather small enough to sit in VMEM is a different regime from one that is not.
    for value_dim in (1, 8, 32, 128, 512):
        for num_indices in (1 << 14, 1 << 16, 1 << 18):
            window = min(512, num_indices)
            rec = run_case(args.rows, value_dim, num_indices, window)
            records.append(rec)
            out.write_text(json.dumps({"meta": meta, "records": records}, indent=2))
            flag = ""
            if "speedup" in rec:
                flag = f"  SC {rec['sparsecore_gbs']:8.2f} GB/s  x{rec['speedup']:.2f}"
                if not rec.get("correct"):
                    flag += f"  WRONG(err={rec['max_abs_err']:.2e})"
            elif "sparsecore_error" in rec:
                flag = f"  SC failed: {rec['sparsecore_error'][:60]}"
            print(f"dim={value_dim:4d} n={num_indices:7d}  "
                  f"TC {rec['tensorcore_gbs']:8.2f} GB/s{flag}", flush=True)
    print(f"\nwrote {out} with {len(records)} records")


if __name__ == "__main__":
    main()
