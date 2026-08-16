#!/usr/bin/env python3.11
"""Does it matter WHICH chips you were given, holding how many fixed?

The accidental result that started this: the same 16 chips cost 0.239 ms carved from a 32-chip
slice and 0.421 ms carved from a 256-chip slice. That says geometry, not count, sets the price.
This tests it deliberately and cheaply, on one slice, by choosing different subsets of the same
devices:

    contiguous   devices[0:n]        the layout a scheduler gives you if it packs
    strided      devices[::step]     what you get when a pod is fragmented across jobs
    reversed     devices[::-1][0:n]  same set as contiguous when n = all, different order
    shuffled     a fixed permutation  the pathological case, deterministic seed

If cost tracks the subset shape at constant n, then allocation shape is a first-class performance
parameter, and a scheduler that ignores it is leaving multiples on the table. If all four are the
same, the earlier 1.76x came from the parent slice's own size and not from placement, which is a
different and also publishable answer.

    python3.11 experiment4_geometry.py
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
REPEATS = 20
WARMUP = 3
PAYLOAD_MIB = 16.0


def timed(fn, *args) -> float:
    out = fn(*args)
    jax.block_until_ready(out)
    for _ in range(WARMUP):
        jax.block_until_ready(fn(*args))
    s = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        s.append(time.perf_counter() - start)
    return statistics.median(s)


def subsets(devices: list, n: int) -> dict:
    """Four ways to pick n devices out of the slice, all the same size."""
    total = len(devices)
    out = {"contiguous": devices[:n]}
    if n < total:
        step = total // n
        out["strided"] = devices[::step][:n]
    out["reversed"] = devices[::-1][:n]
    rng = np.random.default_rng(0)          # fixed, so the pathological case is reproducible
    perm = list(rng.permutation(total))
    out["shuffled"] = [devices[i] for i in perm[:n]]
    return {k: v for k, v in out.items() if len(v) == n}


def measure(mesh, chips: int, per: int, op: str) -> float:
    x = jnp.ones((chips * chips, per), jnp.float32) if op == "all_to_all" \
        else jnp.ones((chips, per * chips), jnp.float32)

    @jax.jit
    def run(v):
        def body(v):
            for _ in range(INNER):
                if op == "all_to_all":
                    v = jax.lax.all_to_all(v, "chips", split_axis=0, concat_axis=0,
                                           tiled=True) * 0.5
                else:
                    v = jax.lax.psum(v, "chips") * 0.5
            return v
        return shard_map(body, mesh=mesh, in_specs=(P("chips", None),),
                         out_specs=P("chips", None))(v)

    return timed(run, x) / INNER


def main() -> None:
    devices = jax.devices()
    meta = {"jax": jax.__version__, "device_kind": devices[0].device_kind,
            "num_devices": len(devices), "process_count": jax.process_count(),
            "host": platform.node(), "payload_mib": PAYLOAD_MIB, "inner": INNER}
    if jax.process_index() == 0:
        print(json.dumps(meta, indent=2), flush=True)

    records = []
    for chips in [c for c in (8, 16) if c <= len(devices)]:
        per = int(PAYLOAD_MIB * 2**20 // 4 // chips)
        for label, subset in subsets(devices, chips).items():
            mesh = Mesh(np.asarray(subset), ("chips",))
            for op in ("all_to_all", "all_reduce"):
                try:
                    ms = measure(mesh, chips, per, op) * 1e3
                    rec = {"op": op, "chips": chips, "layout": label,
                           "parent_slice": len(devices), "ms": round(ms, 4),
                           "device_ids": [int(d.id) for d in subset][:16]}
                except Exception as exc:
                    rec = {"op": op, "chips": chips, "layout": label,
                           "parent_slice": len(devices),
                           "error": f"{type(exc).__name__}: {exc}"[:200]}
                records.append(rec)
                if jax.process_index() == 0:
                    if "error" in rec:
                        print(f"{op:11s} n={chips:3d} {label:11s} {rec['error'][:60]}", flush=True)
                    else:
                        print(f"{op:11s} n={chips:3d} {label:11s} {rec['ms']:8.4f} ms  "
                              f"ids={rec['device_ids'][:8]}", flush=True)

    if jax.process_index() == 0:
        # The comparison the file exists for: spread between best and worst layout at fixed n.
        for chips in sorted({r["chips"] for r in records}):
            for op in ("all_to_all", "all_reduce"):
                vals = {r["layout"]: r["ms"] for r in records
                        if r["chips"] == chips and r["op"] == op and "ms" in r}
                if len(vals) > 1:
                    lo, hi = min(vals.values()), max(vals.values())
                    worst = max(vals, key=vals.get)
                    print(f"  n={chips:3d} {op:11s}: {hi/lo:.2f}x spread across layouts "
                          f"(worst: {worst})", flush=True)
        out = Path("geometry_results.json")
        out.write_text(json.dumps({"meta": meta, "records": records}, indent=2))
        print(f"\nwrote {out} with {len(records)} records")


if __name__ == "__main__":
    main()
