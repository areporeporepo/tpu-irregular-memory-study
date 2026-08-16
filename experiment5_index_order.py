#!/usr/bin/env python3.11
"""Is a TPU gather slow because it is irregular, or slow because it is unsorted?

The 1.3 GB/s figure for a random row gather is 0.08% of HBM bandwidth. Gemini's reading is that
HBM transactions are wide, so a random small-row read pulls a full line and discards most of it,
and it predicted sorting the indices could recover roughly an order of magnitude. That is a cheap
and falsifiable claim, and it matters: MoE routing and embedding lookups can often afford to sort,
and if sorting is what buys the bandwidth back then "irregular access is slow on TPU" is the wrong
lesson to publish.

Four index orders, same count, same table, one chip:

    random      uniform random rows, the worst case and the usual benchmark
    sorted      the same rows, sorted ascending
    blocked     random blocks of consecutive rows, which is what a sorted-by-expert batch looks like
    sequential  rows 0..n, the best case, effectively a strided copy

    python3.11 experiment5_index_order.py
"""
from __future__ import annotations

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


def orders(key, num_rows: int, n: int, block: int = 64) -> dict:
    rnd = jax.random.randint(key, (n,), 0, num_rows, dtype=jnp.int32)
    starts = jax.random.randint(key, (max(1, n // block),), 0, num_rows - block, dtype=jnp.int32)
    blocked = (starts[:, None] + jnp.arange(block, dtype=jnp.int32)[None, :]).reshape(-1)[:n]
    return {"random": rnd,
            "sorted": jnp.sort(rnd),
            "blocked": blocked,
            "sequential": jnp.arange(n, dtype=jnp.int32) % num_rows}


def main() -> None:
    device = jax.devices()[0]
    meta = {"jax": jax.__version__, "device_kind": device.device_kind,
            "host": platform.node(), "python": platform.python_version()}
    print(json.dumps(meta, indent=2), flush=True)

    num_rows = 1 << 20
    gather = jax.jit(lambda t, i: jnp.take(t, i, axis=0))
    records = []
    for value_dim in (1, 8, 32, 128, 512):
        key = jax.random.key(0)
        table = jax.random.normal(key, (num_rows, value_dim), jnp.float32)
        for n in (1 << 16, 1 << 18):
            base = None
            for label, idx in orders(jax.random.key(1), num_rows, n).items():
                idx = jax.device_put(idx)
                dt = timed(gather, table, idx)
                gbs = n * value_dim * 4 / dt / 1e9
                base = base or gbs
                rec = {"order": label, "value_dim": value_dim, "num_indices": n,
                       "ms": round(dt * 1e3, 4), "gbs": round(gbs, 3),
                       "vs_random": round(gbs / base, 3)}
                records.append(rec)
                print(f"dim={value_dim:4d} n={n:7d} {label:11s} {rec['gbs']:8.2f} GB/s  "
                      f"x{rec['vs_random']:.2f} vs random", flush=True)
    out = Path("index_order_results.json")
    out.write_text(json.dumps({"meta": meta, "records": records}, indent=2))
    print(f"\nwrote {out} with {len(records)} records")


if __name__ == "__main__":
    main()
