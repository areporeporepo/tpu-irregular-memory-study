#!/usr/bin/env python3.11
"""Does the shape of a sub-mesh predict its collective speed? A 3D torus is where you can ask.

The study's central claim is that all_to_all cost is set by bisection bandwidth, B = 2*min(X,Y)*b
for a 2D torus. On v6e that was only ever testable in two dimensions, and the model was fitted to
chip *count* rather than to shape, because on a 2D torus of a fixed slice you cannot hold the count
constant and change the geometry much.

v5p is a 3D torus. Our v5p-32 reports dims [2, 2, 4], which makes a sharper experiment possible:
take eight chips four different ways out of the same sixteen, so chip count, payload, code and
compiler are all identical and only the geometry differs.

    cube_2x2x2     z in {0,1}      a contiguous cube, what a good scheduler gives you
    slab_1x2x4     x == 0          contiguous, but long and thin, and the long axis wraps
    slab_2x1x4     y == 0          the same thing rotated, a consistency check on the model
    split_z_0_2    z in {0,2}      two planes that are not adjacent, so the set is disconnected

The last one is the interesting case. Its chips own no link to each other along z, so traffic has to
transit chips the mesh does not include. If the bisection model is right that should be markedly
slower than the cube despite being the same eight chips, and if placement is irrelevant it will not
be.

Bisection here is computed from the coordinates rather than assumed: build the torus edge set for
the chips actually selected, then take the minimum edge count over balanced axis-aligned cuts. A
wraparound link is only counted as a second, distinct link when the axis is longer than two, since
on a length-two axis the "two" neighbours are the same pair of chips.

    python3.11 experiment11_torus_shape.py
"""
from __future__ import annotations

import itertools
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

INNER = 32          # chained collectives per jit, to amortise the host dispatch floor
REPEATS = 12
WARMUP = 3
PAYLOADS_MIB = (1.0, 16.0)


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


def torus_edges(coords: list[tuple], dims: list[int]) -> list[tuple[int, int]]:
    """Links that exist between the selected chips, given the torus wiring."""
    idx = {c: i for i, c in enumerate(coords)}
    edges = []
    for c, i in idx.items():
        for a in range(len(dims)):
            if dims[a] < 2:
                continue
            for step in ({1} if dims[a] == 2 else {1, -1}):
                n = list(c)
                n[a] = (c[a] + step) % dims[a]
                j = idx.get(tuple(n))
                if j is not None and i < j:
                    edges.append((i, j))
    return edges


def bisection(coords: list[tuple], dims: list[int]) -> int:
    """Fewest links crossing a balanced axis-aligned cut of the selected chips."""
    edges = torus_edges(coords, dims)
    if not edges:
        return 0
    best = None
    for a in range(len(dims)):
        vals = sorted({c[a] for c in coords})
        for t in vals[:-1]:
            left = {i for i, c in enumerate(coords) if c[a] <= t}
            if not left or len(left) == len(coords):
                continue
            # only balanced cuts: a 1-vs-7 split is not a bisection
            if abs(len(left) * 2 - len(coords)) > len(coords) // 2:
                continue
            cross = sum(1 for i, j in edges if (i in left) != (j in left))
            best = cross if best is None else min(best, cross)
    return best if best is not None else 0


def bench_all_to_all(mesh, chips: int, per_chip_mib: float) -> dict:
    """Identical to experiment2's, so the numbers are directly comparable."""
    per = int(per_chip_mib * 2**20 // 4 // chips)
    if per <= 0:
        return {}
    x = jnp.ones((chips * chips, per), jnp.float32)

    def make(inner: int):
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

    dt = timed(make(INNER), x) / INNER
    held = per * chips * 4
    moved = held * (chips - 1) / chips
    return {"ms": round(dt * 1e3, 4), "per_chip_mib": round(held / 2**20, 3),
            "bus_gbs": round(moved / dt / 1e9, 2)}


def main() -> None:
    devs = jax.devices()
    coords_of = {d: tuple(d.coords) for d in devs}
    dims = [max(c[i] for c in coords_of.values()) + 1 for i in range(len(next(iter(coords_of.values()))))]
    local = {tuple(d.coords) for d in jax.local_devices()}
    me = jax.process_index()

    meta = {"jax": jax.__version__, "device_kind": devs[0].device_kind, "chips": len(devs),
            "dims": dims, "processes": jax.process_count(), "host": platform.node(),
            "inner": INNER, "repeats": REPEATS}
    if me == 0:
        print(json.dumps(meta, indent=2), flush=True)
        print(f"\nprocess 0 owns coords {sorted(local)}\n", flush=True)

    # Added after the first run, which falsified the bisection reading: the two fast 8-chip sets
    # were each two whole z-planes and the two slow ones took a slice from every plane. Each host
    # owns one 2x2 z-plane, so what actually differed was the number of hosts spanned. These four
    # hold the chip count fixed and vary only the host count, which separates the two explanations.
    shapes = {
        "1host_4chips": lambda c: c[2] == 0,
        "2hosts_4chips": lambda c: c[2] in (0, 1) and c[1] == 0,
        "4hosts_4chips": lambda c: c[0] == 0 and c[1] == 0,
        "cube_2x2x2": lambda c: c[2] in (0, 1),
        "slab_1x2x4": lambda c: c[0] == 0,
        "slab_2x1x4": lambda c: c[1] == 0,
        "split_z_0_2": lambda c: c[2] in (0, 2),
        "full_2x2x4": lambda c: True,
    }

    records = []
    for name, pred in shapes.items():
        sel = [d for d in devs if pred(coords_of[d])]
        sel_coords = [coords_of[d] for d in sel]
        # The timing process must own devices inside the mesh, or it measures a no-op. That mistake
        # produced an impossible 3.5 TB/s figure earlier in this study.
        owns = sum(1 for c in sel_coords if c in local)
        b = bisection(sel_coords, dims)
        rec = {"shape": name, "chips": len(sel), "bisection_links": b,
               "coords": sorted(sel_coords), "process0_owns": owns}
        if owns == 0:
            rec["skipped"] = "this process owns no chip in the mesh"
            records.append(rec)
            continue
        mesh = Mesh(np.asarray(sel), ("chips",))
        for mib in PAYLOADS_MIB:
            try:
                r = bench_all_to_all(mesh, len(sel), mib)
                rec[f"a2a_{mib:g}mib_ms"] = r.get("ms")
                rec[f"a2a_{mib:g}mib_gbs"] = r.get("bus_gbs")
            except Exception as exc:
                rec[f"a2a_{mib:g}mib_error"] = f"{type(exc).__name__}: {exc}"[:160]
        records.append(rec)
        if me == 0:
            print(f"{name:13s} chips={len(sel):3d} bisection={b:3d} links  "
                  f"1MiB {rec.get('a2a_1mib_ms', float('nan')):8.4f} ms  "
                  f"16MiB {rec.get('a2a_16mib_ms', float('nan')):8.4f} ms  "
                  f"({rec.get('a2a_16mib_gbs', float('nan')):7.2f} GB/s)", flush=True)

    if me == 0:
        Path("torus_shape.json").write_text(json.dumps({"meta": meta, "records": records}, indent=2))
        eight = [r for r in records if r["chips"] == 8 and "a2a_16mib_ms" in r]
        if len(eight) > 1:
            print("\nsame eight chips, four geometries, 16 MiB payload:", flush=True)
            base = min(eight, key=lambda r: r["a2a_16mib_ms"])
            for r in sorted(eight, key=lambda r: r["a2a_16mib_ms"]):
                print(f"  {r['shape']:13s} bisection {r['bisection_links']:2d} links  "
                      f"{r['a2a_16mib_ms']:8.4f} ms  "
                      f"{r['a2a_16mib_ms'] / base['a2a_16mib_ms']:5.2f}x the best")
            print("\nIf bisection sets the cost, the ordering here follows the link counts and the")
            print("disconnected set is the slowest. If placement is irrelevant, they are all equal.")


if __name__ == "__main__":
    main()
