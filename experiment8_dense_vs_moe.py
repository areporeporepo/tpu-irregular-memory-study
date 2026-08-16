#!/usr/bin/env python3.11
"""What does the MoE all-to-all actually cost, measured against a dense layer doing the same work?

The whole study points here. A dense feed-forward layer, tensor-parallel across chips, pays one
all_reduce per layer. An MoE layer with the same number of *active* parameters pays two all_to_all
exchanges instead, to dispatch tokens to experts and gather the results back. Everything else about
the two is comparable, so the difference is the routing tax.

Both are built at matched active FLOPs: the MoE has E experts of the dense hidden size divided by
E, with top-k routing, so a token touches the same amount of arithmetic either way. If the MoE is
slower, it is paying for movement, not for maths.

The bisection result predicts the shape of the answer. all_to_all was flat in chip count above
32 chips, so the routing tax should be roughly constant as the slice grows, while a dense
all_reduce grows with participants. That is a falsifiable prediction, and this measures it.

    python3.11 experiment8_dense_vs_moe.py
"""
from __future__ import annotations

import json
import platform
import statistics
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map

D = 4096          # model dim
FF = 4 * D        # dense hidden
TOKENS = 8192     # tokens per step, global
EXPERTS = 8
TOPK = 1          # top-1 routing, so active FLOPs match the dense layer exactly
INNER = 8
REPEATS = 15
WARMUP = 3


def timed(fn, *args) -> float:
    out = fn(*args)
    jax.block_until_ready(out)
    for _ in range(WARMUP):
        jax.block_until_ready(fn(*args))
    s = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        s.append(time.perf_counter() - t0)
    return statistics.median(s)


def dense(mesh, chips: int):
    """Tensor-parallel FFN: shard the hidden dimension, one all_reduce to rejoin."""
    per = FF // chips
    x = jnp.ones((TOKENS, D), jnp.bfloat16)
    w1 = jnp.ones((chips, D, per), jnp.bfloat16) * 0.01
    w2 = jnp.ones((chips, per, D), jnp.bfloat16) * 0.01

    @jax.jit
    def run(x, w1, w2):
        def body(x, w1, w2):
            for _ in range(INNER):
                h = jnp.dot(x, w1[0], preferred_element_type=jnp.bfloat16)
                h = jax.nn.relu(h)
                y = jnp.dot(h, w2[0], preferred_element_type=jnp.bfloat16)
                x = jax.lax.psum(y, "chips") * 0.5      # the dense layer's only collective
            return x
        return shard_map(body, mesh=mesh,
                         in_specs=(P(None, None), P("chips", None, None), P("chips", None, None)),
                         out_specs=P(None, None))(x, w1, w2)
    return run, (x, w1, w2)


def moe(mesh, chips: int):
    """Expert-parallel FFN: one expert per chip, two all_to_all exchanges per layer.

    Matched active FLOPs: each expert has FF/EXPERTS hidden units and top-1 routing sends each
    token to exactly one expert, so a token does the same multiply-accumulate count as the dense
    layer above.
    """
    e_hidden = FF          # an expert is as WIDE as the dense FFN; top-1 routing
                           # then makes per-token active FLOPs identical. Using
                           # FF//EXPERTS was the bug: it did 1/E of the work.
    local_tokens = TOKENS // chips
    x = jnp.ones((chips, local_tokens, D), jnp.bfloat16)
    w1 = jnp.ones((chips, D, e_hidden), jnp.bfloat16) * 0.01
    w2 = jnp.ones((chips, e_hidden, D), jnp.bfloat16) * 0.01

    @jax.jit
    def run(x, w1, w2):
        def body(x, w1, w2):
            xl = x[0]
            for _ in range(INNER):
                # Dispatch: every chip hands every other chip the tokens routed to its expert.
                # Reshaped so the leading axis is the participant count, which is what all_to_all
                # splits. This is the padded-dense form of routing, i.e. what XLA gives you
                # without a custom kernel.
                packed = xl.reshape(chips, local_tokens // chips, D)
                recv = jax.lax.all_to_all(packed, "chips", split_axis=0, concat_axis=0, tiled=True)
                h = jnp.dot(recv.reshape(-1, D), w1[0], preferred_element_type=jnp.bfloat16)
                h = jax.nn.relu(h)
                out = jnp.dot(h, w2[0], preferred_element_type=jnp.bfloat16)
                # Combine: send expert outputs back to the chips their tokens came from.
                back = jax.lax.all_to_all(out.reshape(chips, -1, D), "chips",
                                          split_axis=0, concat_axis=0, tiled=True)
                xl = back.reshape(local_tokens, D) * 0.5
            return xl[None]
        return shard_map(body, mesh=mesh,
                         in_specs=(P("chips", None, None), P("chips", None, None),
                                   P("chips", None, None)),
                         out_specs=P("chips", None, None))(x, w1, w2)
    return run, (x, w1, w2)


def main() -> None:
    devices = jax.devices()
    meta = {"jax": jax.__version__, "device_kind": devices[0].device_kind,
            "num_devices": len(devices), "process_count": jax.process_count(),
            "host": platform.node(), "d_model": D, "ff": FF, "tokens": TOKENS,
            "experts": EXPERTS, "topk": TOPK, "inner": INNER}
    if jax.process_index() == 0:
        print(json.dumps(meta, indent=2), flush=True)

    # Active FLOPs per layer are identical for the two designs by construction.
    # Global FLOPs per layer. Dense TP: every chip sees all tokens, holds FF/chips
    # hidden. MoE EP: every chip sees TOKENS/chips tokens, holds a full-width expert.
    # With top-1 routing these are equal, which is the point of the comparison.
    flops = 2 * 2 * TOKENS * D * FF
    records = []
    for chips in [c for c in (8, 16, 32) if c <= len(devices)]:
        mesh = Mesh(np.asarray(devices[:chips]), ("chips",))
        row = {"chips": chips}
        for name, build in (("dense", dense), ("moe", moe)):
            try:
                fn, args = build(mesh, chips)
                dt = timed(fn, *args) / INNER
                row[name] = {"ms": round(dt * 1e3, 4),
                             "tflops": round(flops / dt / 1e12, 1),
                             "pct_peak": round(flops / dt / 1e12 / (918 * chips) * 100, 1)}
            except Exception as exc:
                row[name] = {"error": f"{type(exc).__name__}: {exc}"[:220]}
        if "ms" in row.get("dense", {}) and "ms" in row.get("moe", {}):
            row["moe_tax"] = round(row["moe"]["ms"] / row["dense"]["ms"], 3)
        records.append(row)
        if jax.process_index() == 0:
            d, m = row.get("dense", {}), row.get("moe", {})
            if "ms" in d and "ms" in m:
                print(f"chips={chips:3d}  dense {d['ms']:7.3f} ms ({d['pct_peak']:4.1f}% peak)  "
                      f"moe {m['ms']:7.3f} ms ({m['pct_peak']:4.1f}% peak)  "
                      f"routing tax {row['moe_tax']:.2f}x", flush=True)
            else:
                print(f"chips={chips:3d}  dense={d.get('error','?')[:60]} "
                      f"moe={m.get('error','?')[:60]}", flush=True)

    if jax.process_index() == 0:
        Path("dense_vs_moe_results.json").write_text(
            json.dumps({"meta": meta, "records": records}, indent=2))
        taxes = [r["moe_tax"] for r in records if "moe_tax" in r]
        if len(taxes) > 1:
            print(f"\nrouting tax across {len(taxes)} slice sizes: "
                  f"{', '.join(f'{t:.2f}x' for t in taxes)}")
            print("The bisection result predicts this stays roughly flat above 32 chips.")


if __name__ == "__main__":
    main()
