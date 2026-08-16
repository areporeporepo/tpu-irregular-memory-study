#!/usr/bin/env python3.11
"""Three measurements that matter to someone running thousands of chips, not thirty-two.

A clean-room collective latency is close to useless at scale, for two reasons a small benchmark
usually misses:

    interference  In a real step the fabric and the MXU run at once and share memory bandwidth. A
                  collective measured on an idle chip is a best case that never happens.
    tail          At 1,000 chips the step waits for the slowest participant, so p99.9 sets the
                  step time and the mean is decoration.

And one control, because the obvious reading of our own earlier data is probably wrong:

    subcomm       Run the collective on a 16-chip subset of a 32-chip slice. If 16 chips are still
                  fast here, the cost we saw at 32 is an N-scaling or bisection-congestion effect.
                  If they are slow, it is the slice's physical shape. Same payload either way.

    python3.11 experiment3_interference_and_tail.py
"""
from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map

INNER = 32
REPEATS = 60          # enough samples that a p99 means something
WARMUP = 5
MATMUL = 4096         # square matmul per chip, sized to occupy the MXU for a while


def samples(fn, *args) -> list[float]:
    out = fn(*args)
    jax.block_until_ready(out)
    for _ in range(WARMUP):
        jax.block_until_ready(fn(*args))
    got = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        got.append(time.perf_counter() - start)
    return got


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q / 100 * (len(ordered) - 1))))
    return ordered[idx]


def stats(values: list[float], inner: int) -> dict:
    per = [v / inner for v in values]
    return {"p50_ms": round(statistics.median(per) * 1e3, 4),
            "p99_ms": round(pct(per, 99) * 1e3, 4),
            "max_ms": round(max(per) * 1e3, 4),
            "jitter_ratio": round(pct(per, 99) / statistics.median(per), 3),
            "samples": len(per)}


def build(mesh, chips: int, per: int, with_matmul: bool):
    """all_to_all chained INNER times, optionally with an MXU-saturating matmul alongside."""
    @jax.jit
    def run(v, w):
        def body(v, w):
            acc = w
            for _ in range(INNER):
                v = jax.lax.all_to_all(v, "chips", split_axis=0, concat_axis=0, tiled=True) * 0.5
                if with_matmul:
                    # Real work on the same chip, competing for memory bandwidth. Kept in the
                    # dependency chain so XLA cannot delete it.
                    acc = jnp.dot(acc, w, precision=jax.lax.Precision.DEFAULT) * 0.5
            return v, acc
        return shard_map(body, mesh=mesh,
                         in_specs=(P("chips", None), P(None, None)),
                         out_specs=(P("chips", None), P(None, None)))(v, w)
    x = jnp.ones((chips * chips, per), jnp.float32)
    w = jnp.ones((MATMUL, MATMUL), jnp.float32) / MATMUL
    return run, x, w


def main() -> None:
    devices = jax.devices()
    meta = {"jax": jax.__version__, "device_kind": devices[0].device_kind,
            "num_devices": len(devices), "process_count": jax.process_count(),
            "host": platform.node(), "inner": INNER, "matmul": MATMUL}
    if jax.process_index() == 0:
        print(json.dumps(meta, indent=2), flush=True)

    records = []
    per_chip_mib = 16.0
    for chips in [c for c in (8, 16, 32) if c <= len(devices)]:
        per = int(per_chip_mib * 2**20 // 4 // chips)
        mesh = Mesh(np.asarray(devices[:chips]), ("chips",))
        for with_matmul in (False, True):
            try:
                run, x, w = build(mesh, chips, per, with_matmul)
                vals = samples(run, x, w)
                rec = {"case": "interference" if with_matmul else "clean",
                       "chips": chips, "per_chip_mib": per_chip_mib,
                       "subset_of": len(devices), **stats(vals, INNER)}
            except Exception as exc:
                rec = {"case": "interference" if with_matmul else "clean", "chips": chips,
                       "error": f"{type(exc).__name__}: {exc}"[:240]}
            records.append(rec)
            if jax.process_index() == 0:
                if "error" in rec:
                    print(f"{rec['case']:13s} chips={chips:3d}  {rec['error'][:70]}", flush=True)
                else:
                    print(f"{rec['case']:13s} chips={chips:3d}  p50 {rec['p50_ms']:7.4f} ms  "
                          f"p99 {rec['p99_ms']:7.4f} ms  jitter x{rec['jitter_ratio']:.2f}",
                          flush=True)

    if jax.process_index() == 0:
        # The comparison the whole file exists for.
        by = {(r.get("case"), r.get("chips")): r for r in records if "p50_ms" in r}
        for chips in sorted({c for _, c in by}):
            clean, hot = by.get(("clean", chips)), by.get(("interference", chips))
            if clean and hot:
                ratio = hot["p50_ms"] / clean["p50_ms"]
                print(f"  chips={chips:3d}: a collective costs {ratio:.2f}x more "
                      f"while the MXU is busy", flush=True)
        out = Path("interference_results.json")
        out.write_text(json.dumps({"meta": meta, "records": records}, indent=2))
        print(f"\nwrote {out} with {len(records)} records")


if __name__ == "__main__":
    main()
