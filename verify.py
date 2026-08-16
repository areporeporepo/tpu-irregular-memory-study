#!/usr/bin/env python3.11
"""Verification, run before any number from this study is quoted anywhere.

Three checks, each aimed at a way this study could be quietly wrong:

    1. correctness    A fast collective that computes the wrong thing is worthless, and nothing in
                      experiment2 or experiment4 checks the result. all_to_all and psum are
                      compared against a numpy reference on the same data.
    2. timing method  Every latency here comes from wall-clock timing with dispatch amortised over
                      32 chained ops. If that method is sound, a different INNER should give the
                      same per-op cost. Disagreement means the chain is being optimised or the
                      amortisation is wrong.
    3. topology       The geometry claim rests on which physical chips a subset holds. Every TPU
                      device reports coords, so the bounding box and diameter of each subset can be
                      printed rather than assumed. This turns "strided is slower" into a statement
                      about distance.

    python3.11 verify.py
"""
from __future__ import annotations

import itertools
import json
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


def gather_global(x):
    """Bring a globally-sharded array to every host.

    A plain np.asarray on a multi-process array raises: the buffers live on devices this process
    cannot address. The first version of this file died here, which is why correctness went
    unchecked on the very meshes that produced impossible timings.
    """
    from jax.experimental import multihost_utils
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))


def check_correctness(devices) -> list[dict]:
    """Do the collectives actually compute what their names claim?"""
    out = []
    chips = min(8, len(devices))
    mesh = Mesh(np.asarray(devices[:chips]), ("chips",))
    per = 4

    # all_to_all: chip i holds rows [i*chips:(i+1)*chips]; afterwards chip i must hold row i of
    # every chip's original block. Distinct values make a wrong permutation visible.
    x = jnp.arange(chips * chips * per, dtype=jnp.float32).reshape(chips * chips, per)

    @jax.jit
    def a2a(v):
        return shard_map(lambda v: jax.lax.all_to_all(v, "chips", split_axis=0, concat_axis=0,
                                                      tiled=True),
                         mesh=mesh, in_specs=(P("chips", None),), out_specs=P("chips", None))(v)

    got = gather_global(a2a(x))
    src = np.asarray(x).reshape(chips, chips, per)      # [sender, slot, :]
    want = src.transpose(1, 0, 2).reshape(chips * chips, per)
    out.append({"check": "all_to_all_permutation", "max_abs_err": float(np.abs(got - want).max()),
                "ok": bool(np.array_equal(got, want)), "chips": chips})

    @jax.jit
    def psum(v):
        return shard_map(lambda v: jax.lax.psum(v, "chips"), mesh=mesh,
                         in_specs=(P("chips", None),), out_specs=P("chips", None))(v)

    y = jnp.arange(chips * per, dtype=jnp.float32).reshape(chips, per)
    got = gather_global(psum(y))
    want = np.tile(np.asarray(y).sum(axis=0), (chips, 1))
    out.append({"check": "psum_value", "max_abs_err": float(np.abs(got - want).max()),
                "ok": bool(np.allclose(got, want)), "chips": chips})
    return out


def check_timing(devices) -> list[dict]:
    """Does the per-op cost survive changing how many ops are chained?"""
    chips = min(8, len(devices))
    mesh = Mesh(np.asarray(devices[:chips]), ("chips",))
    per = int(4 * 2**20 // 4 // chips)
    x = jnp.ones((chips * chips, per), jnp.float32)

    def build(inner):
        @jax.jit
        def run(v):
            def body(v):
                for _ in range(inner):
                    v = jax.lax.all_to_all(v, "chips", split_axis=0, concat_axis=0,
                                           tiled=True) * 0.5
                return v
            return shard_map(body, mesh=mesh, in_specs=(P("chips", None),),
                             out_specs=P("chips", None))(v)
        return run

    out = []
    for inner in (8, 32, 64, 128, 256):
        fn = build(inner)
        jax.block_until_ready(fn(x))
        samples = []
        for _ in range(12):
            start = time.perf_counter()
            jax.block_until_ready(fn(x))
            samples.append(time.perf_counter() - start)
        out.append({"check": "timing_method", "inner": inner,
                    "ms_per_op": round(statistics.median(samples) / inner * 1e3, 4)})
    spread = max(r["ms_per_op"] for r in out) / min(r["ms_per_op"] for r in out)
    out.append({"check": "timing_method_spread", "spread": round(spread, 3),
                "ok": bool(spread < 1.15), "note": "per-op cost should stop moving once dispatch is amortised"})
    return out


def check_topology(devices) -> list[dict]:
    """What does each subset actually look like in the torus?"""
    def describe(subset) -> dict:
        coords = [tuple(int(c) for c in getattr(d, "coords", (d.id, 0, 0))) for d in subset]
        arr = np.asarray(coords)
        extent = (arr.max(axis=0) - arr.min(axis=0) + 1).tolist()
        # Manhattan diameter: the worst-case hop count in a mesh without wraparound.
        diameter = max(sum(abs(a - b) for a, b in zip(p, q))
                       for p, q in itertools.combinations(coords, 2)) if len(coords) > 1 else 0
        return {"extent": extent, "manhattan_diameter": diameter,
                "coords": coords[:8], "n": len(coords)}

    total = len(devices)
    out = []
    for n in [c for c in (8, 16) if c <= total]:
        layouts = {"contiguous": devices[:n], "reversed": devices[::-1][:n]}
        if n < total:
            layouts["strided"] = devices[::(total // n)][:n]
        rng = np.random.default_rng(0)
        layouts["shuffled"] = [devices[i] for i in rng.permutation(total)[:n]]
        for label, subset in layouts.items():
            if len(subset) != n:
                continue
            out.append({"check": "topology", "layout": label, **describe(subset)})
    return out


def main() -> None:
    devices = jax.devices()
    print(f"verifying on {len(devices)} x {devices[0].device_kind}, "
          f"{jax.process_count()} processes\n")
    results = []
    for name, fn in (("correctness", check_correctness), ("timing", check_timing),
                     ("topology", check_topology)):
        try:
            got = fn(devices)
        except Exception as exc:
            got = [{"check": name, "error": f"{type(exc).__name__}: {exc}"[:200]}]
        results += got
        if jax.process_index() == 0:
            for r in got:
                print(json.dumps(r), flush=True)

    if jax.process_index() == 0:
        failed = [r for r in results if r.get("ok") is False or "error" in r]
        Path("verify_results.json").write_text(json.dumps(results, indent=2))
        print(f"\n{len(results)} checks, {len(failed)} problems")
        for f in failed:
            print(f"  PROBLEM: {json.dumps(f)[:180]}")


if __name__ == "__main__":
    main()
