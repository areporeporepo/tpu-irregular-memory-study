# How accelerator capacity changes hands, August 2026

A study section, written because the question "could this capacity be resold" has a precise and
surprising answer, and because the answer explains something about TPU economics that is not
otherwise written down.

## The short version

GPU capacity has a liquid secondary market. **TPU capacity has none, and the barrier is
contractual rather than technical.** Google's response to that gap is not to build a marketplace
but to start selling the chips outright.

## What exists for GPUs

| Venue | Model | Indicative price, mid-2026 |
|---|---|---|
| **Vast.ai** | peer to peer, providers list hardware they control, no uptime guarantee | H100 SXM ~$2.00/hr spot, A100 80GB ~$0.51-0.73/hr, RTX 4090 ~$0.35/hr |
| **RunPod** | split between a Community Cloud of third-party hosts and its own SLA-backed datacenters | between Vast and the hyperscalers |
| **SF Compute** | a spot market for large clusters; owns no GPUs, brokers over $100M of third-party hardware | cluster-scale, market-clearing |
| **Prime Intellect** | an exchange aggregating roughly a dozen clouds | H100 at competitive rates, no long contracts |

Marketplace pricing runs 50 to 70% below the hyperscalers. That discount is the whole reason the
venues exist.

## What exists for TPUs

Nothing equivalent, and the reasons compound:

1. **TPUs are only obtainable through Google Cloud.** There is no hardware to list, because nobody
   outside Google owns any. Vast.ai's model requires a provider who controls a machine.
2. **The terms forbid it.** Google Cloud's terms state a customer may not "sell, resell,
   sublicense, transfer, or distribute any or all of the Services". Reselling a TPU VM is not a
   grey area, and doing it from an institutional billing account puts that account at risk.
3. **Lock-in is the product, not a side effect.** The favourable price per FLOP exists to capture
   the workload. A secondary market would leak exactly the value the pricing is buying.

## So the arbitrage is real and unexecutable

Our measured cost is **$1.4033 per v6e chip-hour** on spot. Against marketplace rates that is
roughly 2x an A100 80GB and about 0.7x an H100 SXM. On paper spec, v6e is 918 BF16 TFLOPS against
A100's 312, so per unit of dense compute the TPU is priced well below what the open market charges
for equivalent GPU throughput.

That gap would ordinarily be closed by arbitrage. It is not closed, and cannot be, because the
only party permitted to sell TPU time is Google. **The absence of the market is the finding**: TPU
pricing can stay structurally below GPU market rates precisely because no mechanism exists to
carry that price into the open market.

## What Google is doing instead, and it changed this year

Rather than allow a secondary market, Google moved the primary one:

- **Direct hardware sales.** On the Q1 FY2026 earnings call Google said it will supply TPUs to a
  select group of customers for deployment in their own datacenters, targeting AI labs, capital
  markets firms and HPC. Alphabet management has cited **2027** as when third-party hardware sales
  begin in earnest, with an ambition of taking up to 10% of NVIDIA's datacenter revenue.
- **A sanctioned third-party TPU cloud.** Google and Blackstone announced a TPU cloud service in
  May 2026, which is a licensed operator rather than a resale market.
- **Very large rental contracts.** Meta signed a multi-billion dollar deal to rent Google TPUs in
  March 2026, and Anthropic has 3.5 GW coming online in 2027.
- **Deliberate lock-in reduction on the 8th generation.** TPU 8t and 8i are hosted by Axion CPUs
  and support native PyTorch through TorchTPU in preview, plus vLLM, SGLang and bare-metal access.

Read together: Google is unbundling the chip from the cloud, on its own terms and its own
schedule. A resale market would have done the same thing without the terms or the schedule, which
is presumably why it is prohibited.

## What this means for a study on this budget

The useful conversion of credit is not into resold hours, it is into **measurements nobody else
can take**. Only Google, and the handful of firms buying chips from 2027, will have TPU hardware.
Everyone else who wants to reason about these parts, including the people writing the open
performance models this study is calibrating, is working from vendor slides.

The mining analogy inverts cleanly. Mining converts electricity into a fungible token whose value
is set by a market. This converts credit into non-fungible public knowledge whose value is set by
scarcity of access, and access is exactly what the contractual structure above makes scarce.

## Sources

Vendor pricing and marketplace structure from public rate cards and comparisons, mid-2026.
Terms language from Google Cloud Platform Terms of Service. TPU sales strategy from Google's
Q1 FY2026 earnings call coverage, the Google and Blackstone announcement, May 2026, and reporting
on the Meta rental agreement, March 2026. Retrieved 2026-08-16.
