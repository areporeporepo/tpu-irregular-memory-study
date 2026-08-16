#!/usr/bin/env python3.11
"""The other half of the question: what does irregular work cost when it goes on the wire?

TPU 8i answers this by deleting SparseCores and adding a collectives engine, then cutting network
diameter from 16 hops to 7. To have any opinion about whether that was the right trade, you need
the current cost curve. This measures it on a v6e slice:

    all_to_all    the MoE dispatch primitive, cost against payload size and against chip count
    all_reduce    the training primitive, for contrast, since its payload is fixed by the model
    permute       a routing-shaped pattern: every chip sends a different slice to every other

Sweeping the participating chip count is the part that matters. A curve over 8, 16 and 32 chips on
one ICI slice is what a model of a 7-hop fabric has to be calibrated against.

Run on every host of the slice at once:
    gcloud compute tpus tpu-vm ssh NAME --worker=all --command='python3.11 bench_fabric.py'
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

# shard_map was promoted out of jax.experimental during the 0.x series; accept either location.
try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map

REPEATS = 20
WARMUP = 3
# Every jitted call pays a host dispatch cost, and on this machine that turned out to be about
# 0.55 ms, which is the same order as the collectives being measured. The first version of this
# file reported that floor as though it were fabric latency. So each measurement now chains
# INNER collectives inside a single jit and divides, which amortises dispatch away. The
# unamortised number is kept too, because the difference between them *is* the dispatch cost.
INNER = 32


def timed(fn, *args) -> float:
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


def make_mesh(devices, axis="chips") -> Mesh:
    return Mesh(np.asarray(devices), (axis,))


def bench_all_to_all(mesh, chips: int, per_chip_mib: float) -> dict:
    """The dispatch half of MoE routing: every chip hands every other chip a share.

    The global array is (chips*chips, per) so that each shard is (chips, per) and the axis being
    split is exactly the number of participants. Getting this wrong is what an earlier version
    did: sharding (chips, per) leaves each shard with a leading axis of 1, which cannot be split
    across `chips` participants.
    """
    per = int(per_chip_mib * 2**20 // 4 // chips)
    if per <= 0:
        return {}
    x = jnp.ones((chips * chips, per), jnp.float32)

    def make(inner: int):
        @jax.jit
        def run(v):
            def body(v):
                for _ in range(inner):
                    # The 0.5 scaling keeps values bounded and stops XLA from folding the chain
                    # of identical collectives into one.
                    v = jax.lax.all_to_all(v, "chips", split_axis=0, concat_axis=0,
                                           tiled=True) * 0.5
                return v
            return shard_map(body, mesh=mesh, in_specs=(P("chips", None),),
                             out_specs=P("chips", None))(v)
        return run

    dt_1 = timed(make(1), x)
    dt_n = timed(make(INNER), x) / INNER
    held = per * chips * 4                       # bytes this chip starts with
    moved = held * (chips - 1) / chips           # bytes that actually leave the chip
    return {"op": "all_to_all", "chips": chips, "per_chip_mib": round(held / 2**20, 3),
            "ms": round(dt_n * 1e3, 4), "ms_with_dispatch": round(dt_1 * 1e3, 4),
            "dispatch_ms": round((dt_1 - dt_n) * 1e3, 4),
            "bus_gbs": round(moved / dt_n / 1e9, 2),
            "per_peer_kib": round(per * 4 / 1024, 3), "inner": INNER}


def bench_all_reduce(mesh, chips: int, per_chip_mib: float) -> dict:
    """For contrast: the gradient primitive, whose payload does not grow with the batch."""
    elems = int(per_chip_mib * 2**20 // 4)
    x = jnp.ones((chips, elems), jnp.float32)

    def make(inner: int):
        @jax.jit
        def run(v):
            def body(v):
                for _ in range(inner):
                    v = jax.lax.psum(v, "chips") * 0.5
                return v
            return shard_map(body, mesh=mesh, in_specs=(P("chips", None),),
                             out_specs=P("chips", None))(v)
        return run

    dt_1 = timed(make(1), x)
    dt_n = timed(make(INNER), x) / INNER
    moved = elems * 4 * 2 * (chips - 1) / chips
    return {"op": "all_reduce", "chips": chips, "per_chip_mib": round(elems * 4 / 2**20, 3),
            "ms": round(dt_n * 1e3, 4), "ms_with_dispatch": round(dt_1 * 1e3, 4),
            "dispatch_ms": round((dt_1 - dt_n) * 1e3, 4),
            "bus_gbs": round(moved / dt_n / 1e9, 2), "inner": INNER}


def main() -> None:
    devices = jax.devices()
    meta = {"jax": jax.__version__, "device_kind": devices[0].device_kind,
            "num_devices": len(devices), "process_index": jax.process_index(),
            "process_count": jax.process_count(), "host": platform.node(),
            "python": platform.python_version()}
    if jax.process_index() == 0:
        print(json.dumps(meta, indent=2), flush=True)

    records = []
    counts = [c for c in (8, 16, 32, 64, 128, 256) if c <= len(devices)]
    for chips in counts:
        mesh = make_mesh(devices[:chips])
        for mib in (0.25, 1.0, 4.0, 16.0):
            for fn in (bench_all_to_all, bench_all_reduce):
                try:
                    rec = fn(mesh, chips, mib)
                except Exception as exc:
                    rec = {"op": fn.__name__, "chips": chips, "per_chip_mib": mib,
                           "error": f"{type(exc).__name__}: {exc}"[:300]}
                if not rec:
                    continue
                records.append(rec)
                if jax.process_index() == 0:
                    if "error" in rec:
                        print(f"{rec['op']:11s} chips={chips:3d} {mib:6.2f} MiB  {rec['error'][:70]}",
                              flush=True)
                    else:
                        print(f"{rec['op']:11s} chips={chips:3d} {rec['per_chip_mib']:6.2f} MiB  "
                              f"{rec['ms']:9.4f} ms/op  (+{rec['dispatch_ms']:.3f} dispatch)  "
                              f"{rec['bus_gbs']:9.2f} GB/s", flush=True)

    if jax.process_index() == 0:
        out = Path("fabric_results.json")
        out.write_text(json.dumps({"meta": meta, "records": records}, indent=2))
        print(f"\nwrote {out} with {len(records)} records")


if __name__ == "__main__":
    main()
