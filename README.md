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

**1. All-to-all cost on whole slices fits "mesh below the full pod, torus at the full pod".** All-to-all is the collective that routes tokens to experts in a mixture-of-experts layer. 16 MiB per chip, 918 TFLOPS chips on a 2D-torus interconnect, each row a complete slice (not a subset carved out of a larger one):

| Slice | Chips | all-to-all | ÷ 16-chip time | Model: torus everywhere | Model: mesh below 256, torus at 256 | Source |
|---|---:|---:|---:|---:|---:|---|
| 4×4 | 16 | 0.237 ms | 1.00 | 1 | 1 | [`fabric_20260816T120856Z`](data/fabric_20260816T120856Z.json) |
| 4×8 | 32 | 0.423 ms | 1.78 | 2 | 2 | [`fabric_20260816T1055Z`](data/fabric_20260816T1055Z.json) |
| 8×8 | 64 | 0.466 ms | 1.96 | 2 | 2 | [`fabric_results_64chip_20260816T113730Z`](data/fabric_results_64chip_20260816T113730Z.json) |
| 16×16 | 256 | 0.438 ms | 1.84 | **4** | **2** | [`fabric256_20260816T110009Z`](data/fabric256_20260816T110009Z.json) |

Model: uniform all-to-all moves N·M/4 bytes across the bisection, so T ≥ (N/4)·M / (W·b), with W the links crossing the cut: W = min(X,Y) for a mesh, 2·min(X,Y) for a torus with wraparound. If every slice were a torus, the 256-chip pod should cost 4× the 16-chip slice; it costs 1.84×. The data fits only if sub-pod slices are meshes without wraparound and the full 16×16 pod is a torus. Google's v6e documentation does not state this; its v4 documentation states the analogous rule (sub-cube slices are meshes). **Falsifiable prediction:** a whole 8×16 slice (128 chips) should take ~0.95 ms if it is a mesh, ~0.44 ms if it is a torus. Not yet measured.

Related: carving a subset out of a larger slice is penalised. The same 16 chips took 0.239 ms as part of a 32-chip slice and 0.421 ms as part of the 256-chip slice; small all-reduce on 128 of 256 chips cost 3.7× the full slice. All-reduce also hit an XLA internal error (`RET_CHECK`) on 8- and 16-chip subsets of the 256-chip slice, while all-to-all ran on the same subsets.

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
