# Three weeks, $20k, and 512 chips of quota that mostly do not exist

Written 2026-08-16. Every number below was measured today on `soe-hpccenter`, not looked up.

## 1. What is actually true right now

| Thing | Measured | How |
|---|---|---|
| v6e quota | **512 on-demand, 1536 spot**, project-global | `services quota list --service=tpu.googleapis.com` |
| v7 / Ironwood quota | **none**, no line item exists | same |
| v6e-1 spot, us-east5-a | **READY**, external IP, SSH works | created `tpu-v6e-probe` |
| v6e-4 / v6e-8 / v6e-32 spot | **"no more capacity in the zone"** | four create attempts, us-east5-a and -b |
| VPN needed | **No.** `default-allow-ssh` is `0.0.0.0/0 tcp:22` | SSH'd in from this Mac, no VPN |
| v6e price, us-east5 | **$2.70/chip-hr** on demand, **$1.4033** spot | Cloud pricing |
| A100, us-west4 | **32 on-demand, 64 preemptible** | region quota |
| SparseCore per v6e chip | **2 cores, 16 subcores, 8 lanes**, 256 KB VMEM, 32 B DMA granule | `plsc.get_sparse_core_info()` on the chip |
| Pallas SparseCore API | **works**, needs Python 3.11 and JAX 0.10.2 | `tpu_sc` imports, exposes `load_gather`, `addupdate_scatter`, `cumsum`, `fetch_and_add`, `parallel_loop` |
| Default image trap | `v2-alpha-tpuv6e` defaults to py3.10, which pins **JAX 0.6.2**, which has no `tpu_sc` and no `pl.kernel` | use `python3.11` explicitly |

**The number that decides the project:** a TensorCore gather of 1M indices out of a 4.2M-row
table takes 6.25 ms on v6e. That is **1.3 GB/s effective against 1,600 GB/s of HBM**, about 0.08%
of the chip's bandwidth. Irregular access is where v6e is worst, and SparseCore is the silicon
that exists to fix it.

## 2. The budget is tighter than the quota implies

504 hours in three weeks. At spot $1.4033/chip-hr:

| Fleet | $/hour | $/day | Burns $20k in |
|---|---|---|---|
| 8 chips | $11.23 | $269 | 74 days |
| 16 chips | $22.45 | $539 | 37 days |
| **28 chips** | **$39.29** | **$943** | **21 days** |
| 32 chips | $44.91 | $1,078 | 19 days |
| a full v6e-256 pod | $359 | $8,616 | 55 hours |

So $20k is *not* "512 chips for three weeks". It is **28 single chips running continuously**, or
two days of a full pod. And since multi-chip slices will not provision anyway, the fleet of
singles is the only shape that both fits the money and exists.

## 2.5 Correction, same day: multi-chip capacity exists, just not where we looked

A nine-zone sweep found **`us-east1-d` provisions a v6e-8 spot slice** while all three us-east5
zones refuse even v6e-4. So the earlier conclusion, singles only, was a *us-east5* fact and not a
project fact. This changes what is possible:

- Multi-chip work is back on the table, which matters because JAXBench's own future-work section
  names **multi-TPU kernel evaluation** as open.
- The zone matters more than the quota. Capacity is per zone and it churns, so the harvester
  should be sweeping continuously rather than settling on a home zone.
- us-central2-b returned a different error entirely, worth classifying separately.

## 2.6 The objection worth taking seriously

Put to Gemini as an open question, the strongest counterargument came back as: **SparseCore is a
deprecating branch.** TPU 8i deletes four SparseCores and puts a Collectives Acceleration Engine
in their place, so Google is signalling that the future of irregular routing at inference is fast
fabric, not a local gather engine. Three weeks spent on SparseCore could make you an expert in a
dead end.

Half right, and the half that is wrong is the useful half:

- **8t keeps SparseCore.** The unit is not deleted, it is *split by role*: training keeps local
  gather, inference moves the work onto the fabric. SparseCore is also present on v7x.
- So the real question is not "is SparseCore good", it is **where the boundary sits between local
  gather and fabric collectives, and why it lands in different places for training and
  inference.** That is a more interesting question than the one I started with, and measuring both
  sides of it needs exactly what is now available: a SparseCore-bearing chip *and* a multi-chip
  slice.

The objection sharpens the project rather than killing it. It does kill the framing "SparseCore
benchmark" and replaces it with "where does irregular work belong, locally or on the wire".

## 3. What that rules out, and what it rules in

Ruled out by capacity, not by taste:

- Multi-pod ICI/DCN scaling studies. Needs slices. There are none.
- Training anything from scratch. 28 chips for 3 weeks is not a training run worth publishing.
- Anything whose headline result is a v6e throughput number. 8t/8i lands and it is stale.

Ruled in: **work whose unit of progress is one kernel on one chip**, run thousands of times.
That is embarrassingly parallel, preemption-tolerant, and exactly what a flaky fleet of spot
singles is good for.

## 3.5 The pattern actually worth betting on

Look at what shipped in 2025-2026 and the shape repeats:

| Work | Agent | Evaluator in the loop | Search budget |
|---|---|---|---|
| AlphaChip | RL policy | placement cost proxy, then commercial P&R | Google-scale |
| AlphaEvolve | LLM | correctness + benchmark timing | Google-scale |
| JAXBench (Jul 2026) | LLM | JAX profiler on a real TPU, `np.allclose` | 50 workloads, one chip each |
| MLCAD 2026 contest | agentic LLM flow | EDA timing optimization | contest-scale |

**The method is commoditised. What is scarce is the evaluator.** Anyone can call a model; almost
nobody can score its output against a v6e SparseCore, and almost nobody scoring EDA output has a
commercial sign-off tool. As a Stanford EE student with Cadence, Synopsys and Siemens Catapult,
plus this TPU grant, you hold **two scarce evaluators at once**. That is the actual asset, and it
is the thing to build a resume around: not "I used an LLM", but "I own the loop that tells the LLM
it is wrong".

So: build **one** agentic search harness, and point it at two evaluators.

## 3.6 Track B: the AlphaChip question, with the baseline nobody has

Worth knowing before you commit, because the field is unresolved rather than closed:

- Synopsys's VP for AI/ML said in 2026 that RL in core EDA algorithms "hasn't really panned out".
- Neither Google nor Ricursive has published AlphaChip results on modern public benchmarks.
- UCSD/TILOS `MacroPlacement` re-implemented it and built the Cadence Genus + Innovus reference
  flows, including scaled 7nm Ariane, precisely because the commercial baseline was missing.

The reason the argument never resolves is that the people with the RL do not publish tool-based
baselines, and the people with the tools mostly do not have the compute. You have both. A careful
head-to-head on public benchmarks, commercial P&R against RL against OpenROAD, is a contribution
whether the answer is yes or no, and you have already contributed to OpenROAD once.

Two hard constraints on this track:

1. **The licence.** Cadence and Synopsys EULAs typically forbid publishing benchmark results.
   MacroPlacement navigates this carefully. Before any of it is public, this must be checked with
   the Stanford EE CAD administrator, in writing. If the answer is no, the numbers get reported as
   "a leading commercial tool", which is still publishable.
2. **The calendar.** MLCAD 2026's contest already closed, results announced 15 June 2026. The next
   windows are MLCAD 2027 and ISPD 2027 (31 March - 2 April 2027, Taipei), with contests usually
   announced late 2026. That is the right target, and it is **outside** the three-week credit
   window. So Track B is not what the credits buy; it is what the credits' harness gets reused for.

## 3.7 Simulating hardware you cannot buy: yes, this is a real method

The instinct is right and it is not a consolation prize. Nobody outside the vendor has Vera Rubin
or 8t/8i either. Architecture research has always been done by people without the chip, and there
is a live 2026 toolchain for exactly this:

| Tool | What it models | Status |
|---|---|---|
| **ASTRA-sim 3.0** | distributed training, hierarchical networks, collectives including all-to-all | arXiv 2606.10440, June 2026; ISCA 2026 tutorial |
| **SCALE-Sim** | systolic arrays cycle-accurate, pre-silicon | ISCA 2026 tutorial; this is the MXU shape |
| **Calculon** | analytical LLM training co-design, hardware/software joint space | SC'23, open source |
| **StableHLO cross-arch modeling** | predicting distributed ML workloads across architectures from the XLA IR | arXiv 2604.12090, April 2026 |

The method is: **measure primitives on hardware you have, fit the model, substitute the announced
specs of the hardware you do not have, and report sensitivities rather than absolutes.** That is
how vendor performance teams work pre-silicon, and it is how DOE labs decide what to buy. NERSC
and the ECP co-design centres exist to do this. Being the person who can calibrate a model on real
silicon and project it forward is a literal DOE job description, not a workaround.

Where the gap is, and why you are positioned for it: **these simulators are calibrated against
NVIDIA GPUs.** TPU-side calibration is thin and SparseCore is essentially unmodelled. You have a
v6e to calibrate against and 32 A100s for the GPU anchor. Producing the missing TPU/irregular
calibration set, openly, is a contribution the tool authors themselves would want.

Honest limits, state them in the paper before a reviewer does:

- You get **relative** trends and crossover points, not absolute 8i throughput.
- Vendor internal models are better than yours. Your edge is that yours is public and reproducible.
- Announced 8t/8i specs are partial, so every assumption must be a named, swappable parameter.
  When the real numbers land, the model gets rerun rather than rewritten.

The workload to aim it at is **MoE**, because that is where the next generation's design choices
actually differ: 8i drops the 3D torus to cut diameter from 16 hops to 7, and Google named MoE
serving as the reason. MoE is all-to-all plus irregular gather/scatter routing, which is the same
primitive set as Layer 1. The three layers stop being three projects and become one argument.

## 4. The project

**Irregular-access kernels on SparseCore, aimed at 8t/8i, validated on v6e and A100.**

Three layers, each shippable on its own.

### Layer 1: the benchmark nobody has built
A suite of *irregular* primitives, the ones that make GPUs and TensorCores fall over: gather,
scatter-add, segment-sum, histogram, top-k expert routing, sparse message passing. For each,
three implementations: v6e TensorCore (XLA), v6e SparseCore (Pallas `tpu_sc`), A100 (Triton or
cuSPARSE). JAXBench, published July 2026 by Google/Harvard/Berkeley/DeepMind, covers 50 dense
MXU-bound workloads and is explicitly TensorCore. **The irregular axis is empty.**

### Layer 2: the agent that tunes them
JAXBench's own numbers: LLM agents reach 1.36x geomean over XLA, hand-tuned Pallas reaches 2.08x.
That gap is the open problem, and their harness (`python -m JAXBench evaluate --workload X
--kernel Y --json`, single chip, correctness-checked, device-timed) is a ready-made scoring
function. Point an agent at SparseCore kernels instead of MXU kernels and let the spot fleet run
the search. This is what burns the $20k productively: every chip-hour is search throughput.

### Layer 3: the forward model for 8t/8i and Vera Rubin
This is the part that answers "what is it for", and per 3.7 it is a real method rather than a
spreadsheet. Feed Layer 1's measured per-primitive costs into ASTRA-sim 3.0 for the collectives
and a SCALE-Sim-style MXU model for the dense part, then substitute announced next-generation
specs. The prediction to make: **which irregular primitives stop being the bottleneck on 8i and
which do not**, given that 8i cuts network diameter from 16 hops to 7 specifically for MoE
serving. Publish the calibration set alongside the model, because the calibration is the scarce
part and it is what makes the prediction checkable the day 8i is reachable.

SparseCore is on v4, v5p, v6e **and v7x**, per the JAX docs. Kernels written now run on
Ironwood-class hardware. That is why this survives the generation change instead of being
invalidated by it.

## 5. Why this is the right resume artifact

- It is *first*, not *better*. "First public SparseCore benchmark for irregular scientific
  workloads" is a sentence no one else can currently write.
- It plugs into a live Google/DeepMind benchmark. A PR to `AI-Hypercomputer/accelerator-agents`
  adding an irregular track is a citable, visible contribution with named maintainers.
- It is DOE-shaped. Irregular memory access on accelerators is the thing exascale people complain
  about. GNN tracking (Exa.TrkX), sparse solvers and histogramming are all in this shape.
- It is AI-lab-shaped too. AI-generated kernels is a hot area right now, and this is a new axis
  in it rather than another KernelBench score.
- Three artifacts fall out: the repo, a short arXiv-able writeup with real measurements, and the
  capacity harvester story below.

## 6. Free side artifact: the capacity data itself

`tpu-capacity-hunter` already classifies quota-vs-capacity failures. Three weeks of running a
spot fleet produces a dataset nobody publishes: **granted 1536 chips, actually obtainable N**,
by zone, by hour, by slice size. That is a short, honest, genuinely useful systems note, and it
writes itself as a byproduct.

## 7. Schedule

| Days | Phase | Spend | Exit condition |
|---|---|---|---|
| 1-2 | Bake-off. One chip. Port 3 primitives to `tpu_sc`, measure against TensorCore. Run JAXBench baseline on v6e. | ~$70 | A SparseCore gather beats 1.3 GB/s by >3x, or the project changes |
| 3-5 | Fleet. Extend the hunter to hold N singles, autoscale, checkpoint to GCS, survive preemption. | ~$500 | 15+ chips held for 6 hours unattended |
| 6-14 | Main burn. Agent search across the suite, A100 baselines in parallel on the 32-GPU grant. | ~$14k | Suite complete, all three implementations, all correctness-checked |
| 15-18 | Forward model + writeup. | ~$2k | Draft with numbers, repo public |
| 19-21 | Buffer, PRs, publish. | ~$2k | PR opened, post out |

Reserve is deliberate: spot preemption and re-runs always cost more than planned.

## 8. Go/no-go tests

Passed today:
- SparseCore Pallas imports and reports hardware on v6e. **Pass.**
- Single-chip spot capacity exists and is SSH-able without VPN. **Pass.**
- TensorCore irregular access is genuinely bad, so there is headroom to win. **Pass, 1.3 GB/s.**

Still to test, days 1-2:
- Can a `tpu_sc` gather kernel actually be written and beat TensorCore? The docs claim 4-5x.
- How many single chips can be held at once? Unknown; this sets the whole burn rate.
- Does JAXBench run clean on v6e with JAX 0.10.2?

## 9. Rejected, with reasons

| Idea | Why not |
|---|---|
| Multi-pod DCN scaling study | No slices. Capacity refuses even 4 chips. |
| AlphaChip-style RL floorplanning | Crowded, contested, and the TPU is not the interesting part of it. |
| Train an LLM on MaxText | 28 chips proves nothing anyone wants to read. |
| Generic MoE routing work | XLA and MaxText already optimized it. Only worth it via the irregular-primitive angle. |
| Scaling the JAX cosmology code | Good science, but it wants big slices, and it is a v6e number that 8t/8i stales. |
