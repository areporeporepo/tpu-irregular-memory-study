# Collective and irregular-access measurements, 8 to 256 accelerator chips

How data movement behaves on a multi-host slice of 918 TFLOPS (bf16), 1.64 TB/s HBM chips (Google TPU v6e), from 8 to 256 chips, in JAX/XLA. Dated August 2026. Every number below links to the raw file it came from.

## Hardware, by the numbers that matter

| Chip (label) | Peak bf16 | HBM bandwidth | Ridge point, peak FLOP ÷ HBM bytes |
|---|---:|---:|---:|
| TPU v5e | 197 TFLOPS | 819 GB/s | 241 FLOP/byte |
| TPU v6e | 918 TFLOPS | 1,640 GB/s | 560 FLOP/byte |
| TPU v5p | 459 TFLOPS | 2,765 GB/s | 166 FLOP/byte |

Vendor peak figures. A kernel whose arithmetic intensity sits below the ridge point is limited by memory bandwidth, not compute.

## Results

**1. All-to-all bandwidth stays flat with scale; all-reduce does not.** 16 MiB per chip, 918 TFLOPS chips, 64 host processes ([`data/fabric256_20260816T110009Z.json`](data/fabric256_20260816T110009Z.json)):

| Chips | all-to-all | all-reduce | all-to-all ÷ all-reduce time |
|---:|---:|---:|---:|
| 32 | 0.424 ms, 38.3 GB/s bus | 0.243 ms, 133.7 GB/s bus | 1.74× |
| 64 | 0.431 ms, 38.3 GB/s bus | 0.246 ms, 134.5 GB/s bus | 1.75× |
| 128 | 0.438 ms, 38.0 GB/s bus | 0.295 ms, 112.9 GB/s bus | 1.49× |
| 256 | 0.438 ms, 38.2 GB/s bus | 0.212 ms, 158.0 GB/s bus | 2.07× |

All-to-all is the collective that routes tokens to experts in a mixture-of-experts layer, so this is the routing cost a model pays that dense data parallelism (all-reduce) does not.

In the 256-chip run, all-reduce hit an XLA internal error (`RET_CHECK`) at 8 and 16 chips; 8- and 16-chip all-reduce comes from separate runs on the same chip type (e.g. [`data/fabric_20260816T1055Z.json`](data/fabric_20260816T1055Z.json): 0.223 ms at 8 chips and 0.240 ms at 16 chips, 16 MiB).

**2. At these message sizes, per-op dispatch dominates.** The script times dispatch overhead separately: 0.51 to 1.28 ms per op, larger than the collective itself in all 40 measured rows (0.25 to 16 MiB). Batching small collectives matters more than the fabric at this size.

**3. Companion training benchmark** (in [tpu-capacity-hunter](https://github.com/areporeporepo/tpu-capacity-hunter/blob/main/benchmarks/RESULTS-2026-08-14.md)), a 4B-parameter transformer training step:

- 97.1% strong-scaling efficiency from 8 to 16 chips of 918 TFLOPS each (DP2 × TP8, fixed global batch).
- Memory-bandwidth-bound: going from the 197 TFLOPS / 819 GB/s chip to the 918 TFLOPS / 1,640 GB/s chip gave a 2.16× speedup, matching the 2.00× bandwidth ratio rather than the 4.66× compute ratio.
- MFU follows HBM bytes per FLOP: 31.7% on the 166 FLOP/byte chip, 25.1% at 241, 11.6% at 560. The chip with the most compute had the lowest MFU.

## What else is here

- Sparse-gather throughput on the dedicated gather units (SparseCore), HBM-promotion thresholds, torus-shape and interference effects on collective tail latency: `experiment1` to `experiment12`, raw output in `data/`.
- Interactive pages (GitHub Pages): the logbook ([`index.html`](index.html)), a cross-vendor glossary ([`glossary.html`](glossary.html)), and the gather-cliff and torus pages.

## Not yet measured

`experiment8_dense_vs_moe.py` (a dense layer against an MoE layer at matched FLOPs) is defined but has not produced output. No dense-vs-MoE claim is made here.

## Reproduce

Measured with Python 3.11.13 and JAX 0.10.2. Each `experiment*.py` writes a timestamped JSON to `data/`; `burst_256.sh` is the 256-chip launcher.

Author: Anh Q. Nguyen
