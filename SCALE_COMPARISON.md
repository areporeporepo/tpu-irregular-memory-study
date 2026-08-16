# How much less we have, stated honestly

Verified specs, August 2026. The point of this table is not to feel small. It is to decide what
kind of contribution is even possible from here, and the answer it gives is clear: not scale.

## Per chip

| | v6e (ours) | Ironwood v7x | TPU 8t (late 2027) | TPU 8i (late 2027) | A100 40GB (ours) |
|---|---|---|---|---|---|
| HBM per chip | 32 GB | 192 GB | 8 stacks, 12-high HBM3e | 288 GB | 40 GB |
| Bandwidth | ~1.6 TB/s | **7.37 TB/s** | ~30% over Ironwood | not published | 1.56 TB/s |
| Compute | 918 BF16 TFLOPS | 4,614 FP8 TFLOPS | not published | not published | 312 BF16 TFLOPS |
| On-chip SRAM | small | baseline | not published | 384 MB, ~3x Ironwood | 40 MB L2 |
| Irregular-access unit | SparseCore (2 cores, 16 subcores, 8 lanes) | SparseCore | SparseCore kept | **replaced by CAE** | none |
| Fabric | ICI, 3D torus | ICI 9.6 Tb/s, 3D torus | 3D torus, larger | **Boardfly, 7 hops, OCS** | NVLink/IB |

Ironwood has **4.6x our memory bandwidth per chip** and 6x our HBM capacity. 8i then triples
on-chip SRAM again so KV caches stop spilling. Every generation on this table spends its increment
on memory and interconnect, not on arithmetic.

## Per system, and this is where it gets absurd

| | Chips | HBM total | Aggregate bandwidth |
|---|---|---|---|
| **What we actually obtained** | 32 v6e | 1.0 TB | ~51 TB/s |
| Our quota if capacity existed | 512 v6e | 16 TB | ~820 TB/s |
| One Ironwood pod | 9,216 | 1.77 PB | ~68 PB/s |
| One 8t superpod (2027) | 9,600 | 2 PB | higher again |
| One 8i group (2027) | 1,152 | 332 TB | not published |
| One NVL144 CPX rack | 144 + 144 | 100 TB fast memory | **1.7 PB/s** |

Our 32-chip slice is roughly **1/1,700th of one Ironwood pod** by HBM capacity, and about
**1/1,300th** by aggregate bandwidth. A single NVIDIA rack has 100x our memory. Google's 2026 capex
is $175 to $185 billion; our credit is $20,000, which is about **one nine-millionth** of it.

## So the strategy follows from the arithmetic

We cannot contribute at scale, and any result of the form "we ran a big thing" is worthless from
here. Three things are still open to us, and all three are things scale does not buy:

1. **Measurement of a unit nobody publishes numbers for.** SparseCore is in v4, v5p, v6e and v7x,
   and there is no public benchmark of it. One chip is enough to produce that, and one chip is what
   capacity will reliably give us.
2. **Calibration for models that currently guess.** ASTRA-sim, SCALE-Sim and Calculon are all
   tuned against NVIDIA parts. A TPU calibration set for irregular primitives is missing, and
   producing it needs care rather than chips.
3. **The prediction that becomes checkable.** 8i claims 5x lower on-chip collective latency and a
   diameter cut from 16 hops to 7. Measuring the current cost curve on ICI now means that when 8i
   is reachable, the claim can be checked against a number that was published before the chip
   existed. That is the most valuable artifact available to someone with 32 chips.

## What our own measurements say so far, 16 August 2026

| Measurement | Value | Note |
|---|---|---|
| TensorCore row gather, 1M indices of 4.2M rows | 6.25 ms, **1.3 GB/s** | 0.08% of the chip's own HBM bandwidth |
| all_to_all, 8 to 32 chips, 0.25 MiB | ~0.55 ms, flat in chip count | first pass, dispatch-contaminated |
| all_reduce, 8 to 32 chips, 0.25 MiB | ~0.55 ms, flat in chip count | same floor, so the floor was the harness |
| all_to_all, 16 MiB | 1.20 ms at 8 chips, 1.40 ms at 32 | +17% for 4x the participants |
| Spot survival, v6e-1 in us-east5-a | preempted in under 27 minutes | why the campaign is built from short cycles |
| Multi-chip capacity, 9 zones swept | only us-east1-d and asia-northeast1-b | quota 1536, obtainable 32 |

The flat-in-chip-count result is the interesting one and it needs the correction described in
`experiment2_fabric_collectives.py`: a ~0.55 ms floor appearing identically for two different
collectives at two different payload sizes is the signature of host dispatch overhead, not of a
fabric. The corrected version chains 32 collectives inside one jit to amortise it away. **Do not
cite the 0.55 ms figure until that rerun lands.**
