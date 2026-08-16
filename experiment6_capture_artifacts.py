#!/usr/bin/env python3.11
"""Bank the raw artifacts, because the hardware is temporary and the files are not.

Access to this project ends when the course credit does. Summary numbers are cheap to regret: the
moment a reviewer asks "is that XLA choosing a ring algorithm or a hardware limit", the answer is
in a compiled HLO module or a profiler trace, and by then there is no TPU to produce one.

So this captures, for a set of representative configurations:

    HLO       the optimised module XLA actually ran, as text. This is where the collective
              algorithm choice, the fusion decisions and the layout assignments are visible, and
              it explains results long after the chips are gone.
    trace     an xprof profiler trace, which carries device-side timings and therefore settles
              the dispatch-versus-fabric question without any wall-clock arithmetic.
    metadata  device coords for every chip, so any later topology claim can be checked.

Everything lands in artifacts/ and gets pulled to the laptop by the supervisor.

    python3.11 experiment6_capture_artifacts.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map

OUT = Path("artifacts")
INNER = 32


def build(mesh, chips: int, per: int, op: str):
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
    return run


def main() -> None:
    devices = jax.devices()
    OUT.mkdir(exist_ok=True)
    lead = jax.process_index() == 0

    # Topology, so any later claim about distance can be re-derived rather than remembered.
    if lead:
        topo = [{"id": int(d.id), "coords": [int(c) for c in getattr(d, "coords", ())],
                 "process": int(d.process_index), "kind": d.device_kind} for d in devices]
        (OUT / "topology.json").write_text(json.dumps(topo, indent=2))
        print(f"banked topology for {len(topo)} devices", flush=True)

    captured = []
    for chips in [c for c in (8, 16, 32, 64, 128, 256) if c <= len(devices)]:
        mesh = Mesh(np.asarray(devices[:chips]), ("chips",))
        for mib in (0.25, 16.0):
            per = int(mib * 2**20 // 4 // chips)
            if per <= 0:
                continue
            for op in ("all_to_all", "all_reduce"):
                tag = f"{op}_{chips}chips_{mib:g}MiB"
                x = jnp.ones((chips * chips, per), jnp.float32) if op == "all_to_all" \
                    else jnp.ones((chips, per * chips), jnp.float32)
                fn = build(mesh, chips, per, op)
                try:
                    # The compiled module: algorithm choice and fusion, in text, forever.
                    if lead:
                        text = fn.lower(x).compile().as_text()
                        (OUT / f"hlo_{tag}.txt").write_text(text)
                    # A device-side trace of the same thing.
                    trace_dir = OUT / f"trace_{tag}"
                    with jax.profiler.trace(str(trace_dir)):
                        for _ in range(3):
                            jax.block_until_ready(fn(x))
                    captured.append({"tag": tag, "ok": True})
                    if lead:
                        print(f"banked {tag}", flush=True)
                except Exception as exc:
                    captured.append({"tag": tag, "ok": False,
                                     "error": f"{type(exc).__name__}: {exc}"[:200]})
                    if lead:
                        print(f"failed {tag}: {captured[-1]['error'][:80]}", flush=True)

    if lead:
        (OUT / "captured.json").write_text(json.dumps(captured, indent=2))
        size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
        print(f"\n{sum(1 for c in captured if c['ok'])}/{len(captured)} captured, "
              f"{size / 2**20:.1f} MiB in {OUT}/")


if __name__ == "__main__":
    main()
