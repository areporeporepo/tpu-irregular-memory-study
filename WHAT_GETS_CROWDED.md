# What is hot now, what gets crowded next, and where the gap stays open

Written 2026-08-16 from three sources that mostly agree: the citation record, the silicon
roadmaps, and where the money is committed. The purpose is to avoid spending three weeks
duplicating work while still landing in a field people care about.

## What the citation record says today

Queried through OpenAlex, works from 2025 onward, sorted by citations.

| Area | Top recent work | Cites | Read |
|---|---|---|---|
| MoE inference and serving | MegaScale-Infer: disaggregated expert parallelism | 14 | **hot and filling up** |
| | D2MoE, BrownoutServe, expert swapping on edge | 2-9 | a crowd forming |
| Collective communication for accelerators | SuperMesh: energy-efficient collectives | 3 | **nearly empty** |
| | On Topology's Role in ML Training Performance (2026) | **0** | brand new, nobody has cited it |
| LLM-generated kernels | TritonBench, GPU Kernel Scientist, CuAsmRL | 1-10 | active, and entirely GPU |

Two things fall out. MoE serving is the crowded room. Collective and topology behaviour on real
accelerators is an empty one, and the single 2026 paper closest to our question has no citations
yet, which means the question is recognised but unanswered.

## The gap, stated precisely

**Everyone is working on MoE serving. Almost nobody has measured collectives on a TPU.** Not
because it is uninteresting, but because it needs hardware that cannot be rented outside Google
Cloud and cannot be resold, which is the subject of MARKET_MECHANISMS.md. The intersection is
where a small study can matter: *fabric behaviour, measured, in service of the workload everyone
already cares about.*

## What gets crowded next, and why

Ranked by confidence, with the evidence rather than the vibe.

**1. Disaggregated prefill and decode. Very high confidence.**
NVIDIA shipped separate silicon for it (Rubin CPX with GDDR7 for prefill, VR200 with HBM4 for
decode) and Google split its entire line (8t training, 8i inference). The most-cited recent MoE
paper is about disaggregating expert parallelism. When two competitors commit silicon and the
literature's leading paper commits to the same idea, the research follows within a year. Expect
this to be thoroughly crowded by mid-2027.

**2. KV cache under memory scarcity. High confidence.**
HBM is sold out through 2026 at SK hynix, Samsung and Micron. 8i answers with 288 GB of HBM and
384 MB of on-chip SRAM, roughly 3x Ironwood, specifically so long-context KV caches stop spilling.
Anything that reduces bytes per token served is worth real money in 2027, and scarcity makes it
worth more.

**3. Fabric-aware scheduling and placement. Medium confidence, and the most open.**
8i cuts network diameter from 16 hops to 7 and adds a collectives engine, and Google named MoE
serving as the reason. That is a bet that placement and collective latency dominate. The
literature has essentially nothing on it for TPUs. If the bet is right, this becomes crowded in
2027; if it is wrong, the measurement still explains why. Either way the data is missing now.

**4. Cross-architecture cost models. Medium, and rising in 2027.**
Google begins selling TPUs into other people's datacenters in 2027. The moment a buyer can choose
between a TPU and a GPU on their own floor, public comparative models stop being academic. Today
those models are calibrated on NVIDIA parts only.

## What is already too crowded to enter

- **Another LLM-generates-CUDA-kernels benchmark.** TritonBench, KernelBench, JAXBench, GPU Kernel
  Scientist and AlphaEvolve are all in this space, several from large labs with compute we cannot
  match. Contributing an irregular or SparseCore track to JAXBench would be additive; starting a
  competing suite would not.
- **Generic MoE routing algorithms.** Optimised to death inside XLA and MaxText.
- **Anything whose headline is a v6e throughput number.** 8t and 8i land late 2027 and stale it.

## So the study aims at the intersection

**Measure the collective and irregular-access behaviour that MoE serving depends on, on TPU
hardware nobody else can rent, and express the result in tokens rather than microseconds.**

Tokens matter because that is the unit the market prices, the unit a serving operator budgets in,
and the unit that survives a hardware generation. A microsecond of all-to-all is a curiosity; the
same number expressed as "this is what a v6e chip-hour buys you in served MoE tokens, and here is
what the same money buys on the GPU spot market" is a decision input.

That framing also merges the two halves of this study: the collectives work is the mechanism, the
serving measurement is the consequence, and the market analysis is what makes the consequence
legible to someone who does not care about fabrics.
