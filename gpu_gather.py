#!/usr/bin/env python3
"""The same gather measurement as experiment10, on an NVIDIA GPU, so the TPU numbers have a peer.

experiment10 found something odd on TPU: the speed of `jnp.take` depends on how large the indexed
buffer is *declared* to be, and not at all on which addresses the indices touch or what order they
arrive in. On v6e the rate steps from 154 GB/s to 42 GB/s between an 80 MiB and a 96 MiB buffer,
flat on both sides. A cliff that sharp, with no dependence on the access footprint, looks like a
compile-time lowering choice rather than anything about the memory system.

That claim needs an outside comparison, because "gathers are slow" is not a TPU-specific fact. A GPU
runs the same gather with thousands of resident threads hiding the latency, and its L2 is a real
cache, so the prediction is the mirror image:

    GPU rate should depend on the FOOTPRINT the indices touch, because that is what fits in L2
    GPU rate should NOT depend on the buffer's declared size, because nothing decides anything
        at compile time

If that is what happens, then the two architectures fail at irregular access for different reasons,
and the TPU's reason is fixable in a compiler.

Everything is matched to the TPU harness on purpose: 16384 indices, 128-wide float32 rows, 8 MiB
delivered per gather, 32 gathers chained per measurement, an accumulator that forces both the
gather and the contiguous reference to materialise their output, and the same index orders. Timing
uses a captured CUDA graph so the per-op cost excludes launch overhead, which is the closest
analogue to putting the whole chain inside one `jax.jit`.

    python3 gpu_gather.py                # the span/order/allocation grid
    python3 gpu_gather.py --sweep        # the allocation ladder, to look for a cliff
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
from pathlib import Path

import torch

REPEATS = 12
WARMUP = 3
INNER = 32
DIM = 128
N = 16384
ALLOC = 262144            # rows; 128 MiB at dim=128 float32

# Published HBM bandwidth, GB/s, for the % -of-peak column only. The contiguous reference measured
# on the same device is the number the gathers are actually judged against.
PEAK = {"L4": 300.0, "A100-SXM4-40GB": 1555.0, "A100-SXM4-80GB": 2039.0, "A100-PCIE-40GB": 1555.0,
        "A100 80GB PCIe": 1935.0, "H100 80GB HBM3": 3350.0, "H100 PCIe": 2000.0, "T4": 320.0,
        "V100-SXM2-16GB": 900.0, "A10G": 600.0, "L40S": 864.0, "H200": 4800.0, "B200": 8000.0}

CONFIGS = [
    (ALLOC,   4096, "random"),
    (ALLOC,  16384, "random"),
    (ALLOC,  65536, "random"),
    (ALLOC, 262144, "random"),
    (ALLOC, 262144, "sorted"),
    (ALLOC, 262144, "blocked"),
    (65536,  65536, "random"),
    (4096,    4096, "random"),
]
SWEEP_ROWS = [16384, 32768, 49152, 65536, 81920, 98304, 131072, 163840, 196608, 262144, 393216]


def peak_for(name: str) -> float:
    for key, val in PEAK.items():
        if key in name:
            return val
    return float("nan")


def make_indices(span: int, order: str, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if order == "random":
        idx = torch.randint(0, span, (N,), generator=g)
    elif order == "sorted":
        idx = torch.sort(torch.randint(0, span, (N,), generator=g)).values
    elif order == "blocked":
        block = 32
        starts = torch.randint(0, max(1, span - block), (N // block,), generator=g)
        idx = torch.cat([torch.arange(s, s + block) for s in starts])
    else:
        raise ValueError(order)
    return idx.to(torch.int64).cuda()          # torch indexing wants int64; 128 KB, immaterial


def graph_timed(build, *tensors) -> float:
    """Median seconds per replay of a captured graph, launch overhead excluded."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(WARMUP):
            build(*tensors)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        build(*tensors)

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / 1e3)
    return statistics.median(samples)


def gather_chain(span: int, inner: int):
    def build(table, idx):
        acc = torch.zeros((N, DIM), device="cuda", dtype=table.dtype)
        for i in range(inner):
            acc = acc + table[(idx + i) % span]
        return acc
    return build


def contiguous_chain(alloc: int, inner: int):
    stride = max(1, (alloc - N) // max(1, inner))

    def build(table):
        acc = torch.zeros((N, DIM), device="cuda", dtype=table.dtype)
        for i in range(inner):
            acc = acc + table[i * stride: i * stride + N]
        return acc
    return build


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gpu_gather.json")
    ap.add_argument("--inner", type=int, default=INNER)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    inner = args.inner

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    peak = peak_for(name)
    meta = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": name,
            "sm": f"{props.major}.{props.minor}", "hbm_gib": round(props.total_memory / 2**30, 1),
            "l2_mib": round(getattr(props, "L2_cache_size", 0) / 2**20, 1),
            "sms": props.multi_processor_count, "host": platform.node(),
            "inner": inner, "repeats": REPEATS, "dim": DIM, "indices": N,
            "hbm_gbs_assumed": peak, "delivered_mib": round(N * DIM * 4 / 2**20, 3)}
    print(json.dumps(meta, indent=2), flush=True)

    delivered = N * DIM * 4
    big = torch.randn((ALLOC, DIM), device="cuda", dtype=torch.float32)

    ref = graph_timed(contiguous_chain(ALLOC, inner), big) / inner
    ref_gbs = delivered / ref / 1e9
    print(f"\ncontiguous reference: {ref * 1e3:.4f} ms/op  {ref_gbs:8.2f} GB/s  "
          f"{ref_gbs / peak * 100:5.2f}% of HBM  (same accumulator as every gather below)\n",
          flush=True)

    configs = [(r, r, "random") for r in SWEEP_ROWS] if args.sweep else CONFIGS
    records = []
    for alloc, span, order in configs:
        table = big if alloc == ALLOC else torch.randn((alloc, DIM), device="cuda",
                                                       dtype=torch.float32)
        idx = make_indices(span, order)
        gbs = delivered / (graph_timed(gather_chain(span, inner), table, idx) / inner) / 1e9
        rec = {"alloc_rows": alloc, "span_rows": span, "order": order,
               "alloc_mib": round(alloc * DIM * 4 / 2**20, 1),
               "span_mib": round(span * DIM * 4 / 2**20, 1),
               "gather_gbs": round(gbs, 2), "pct_hbm": round(gbs / peak * 100, 3),
               "pct_contiguous": round(gbs / ref_gbs * 100, 2)}
        # A fast wrong gather is worth nothing: check the chain's first term against index_select.
        want = torch.index_select(table, 0, idx)
        got = table[idx]
        rec["max_abs_err"] = float((want - got).abs().max().item())
        records.append(rec)
        Path(args.out).write_text(json.dumps(
            {"meta": meta, "contiguous": {"ms": round(ref * 1e3, 4), "gbs": round(ref_gbs, 2)},
             "records": records}, indent=2))
        print(f"alloc {rec['alloc_mib']:7.1f} MiB  span {rec['span_mib']:7.1f} MiB  {order:9s}  "
              f"gather {rec['gather_gbs']:8.2f} GB/s  ({rec['pct_contiguous']:5.1f}% of contiguous, "
              f"{rec['pct_hbm']:5.2f}% of HBM)", flush=True)
        if table is not big:
            del table
            torch.cuda.empty_cache()

    if args.sweep:
        rates = [r["gather_gbs"] for r in records]
        print(f"\nallocation ladder: {min(rates):.1f} to {max(rates):.1f} GB/s, "
              f"a {max(rates) / min(rates):.2f}x spread across a 24x range of buffer size.")
        print("  A flat line here means the GPU makes no compile-time decision on buffer size,")
        print("  which is the TPU behaviour experiment10 found. A step means it does.")
    else:
        span_rows = [r for r in records if r["alloc_rows"] == ALLOC and r["order"] == "random"]
        if len(span_rows) > 1:
            lo, hi = span_rows[0]["gather_gbs"], span_rows[-1]["gather_gbs"]
            print(f"\nspan 2 MiB -> 128 MiB at fixed allocation: {lo:.1f} -> {hi:.1f} GB/s "
                  f"({lo / hi:.2f}x). On TPU this was flat; a fall here means the GPU is "
                  f"footprint-limited, i.e. its L2 is doing the work.")
        ctl = [r for r in records if r["alloc_rows"] == r["span_rows"]]
        for r in ctl:
            match = [s for s in span_rows if s["span_rows"] == r["span_rows"]]
            if match:
                print(f"  allocation control at span {r['span_mib']:.0f} MiB: "
                      f"{match[0]['gather_gbs']:.1f} GB/s in a 128 MiB buffer vs "
                      f"{r['gather_gbs']:.1f} GB/s in a {r['alloc_mib']:.0f} MiB buffer")


if __name__ == "__main__":
    main()
