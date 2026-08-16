#!/usr/bin/env python3
"""Build glossary.html: the same ideas, in each vendor's dialect, with our own numbers attached.

Most of the difficulty in reading accelerator documentation is not conceptual, it is that three
companies invented different words for the same wire and nobody writes the translation down. NVLink
and ICI are the same idea with different topologies. A Streaming Multiprocessor and a TensorCore
occupy the same slot in the story. `TP8` is written as though everyone already knows.

So this page is two tables and a warning list. What makes it worth publishing rather than being one
more generic glossary is the third column: wherever this study measured the cost of one of these
things, the measurement is next to the term instead of a hand-wave.

    python3 build_glossary_page.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "glossary.html"

# concept, NVIDIA's word, Google's word, what it actually is
DIALECT = [
    ("chip-to-chip fabric", "NVLink, switched by NVSwitch", "ICI, direct-wired as a 2D or 3D torus",
     "The private high-bandwidth network between accelerators that skips PCIe and the host. "
     "NVLink is <em>switched</em>, so within a domain every pair is equidistant. ICI is a "
     "<em>torus</em>, so traffic hops through intermediate chips and geometry matters."),
    ("scale-out fabric", "InfiniBand or RoCE, once you leave the NVLink domain",
     "still ICI, all the way to the edge of the pod",
     "This is the real structural difference. A GPU cluster has two tiers with a bandwidth cliff "
     "between them at 8 or 72 GPUs. A TPU pod has one fabric to thousands of chips."),
    ("collectives library", "NCCL", "XLA collectives, inside libtpu",
     "The thing that implements all_reduce and friends. On GPU you can call it directly; on TPU it "
     "is emitted by the compiler, which is why a TPU collective is harder to hand-tune."),
    ("kernel language", "CUDA, or Triton", "Pallas, lowering through Mosaic",
     "How you write something the built-in ops do not cover. Both are the escape hatch."),
    ("the compute unit", "SM, Streaming Multiprocessor", "TensorCore, plus its VPU",
     "The scheduling unit. An H100 has 132 SMs; a v6e chip has one TensorCore. That difference in "
     "count is why a GPU hides memory latency with thread parallelism and a TPU cannot."),
    ("the matrix unit", "Tensor Core (confusingly, inside an SM)", "MXU",
     "The systolic array that does the multiply-accumulate. NVIDIA's 'Tensor Core' and Google's "
     "'TensorCore' are <em>not</em> the same level of the hierarchy, which is a genuinely bad "
     "collision of names."),
    ("on-chip scratch", "shared memory and L1, plus a shared L2", "VMEM and CMEM",
     "Fast memory that is not HBM. On a GPU the L2 is a cache and fills on demand; on a TPU the "
     "compiler decides statically what lives there, which is the whole subject of our gather "
     "result."),
    ("gather engine", "none, the SMs do it", "SparseCore",
     "Dedicated hardware for chasing pointers, present on v4, v5p and v6e. Nothing on the NVIDIA "
     "side corresponds to it."),
    ("graph compiler", "TensorRT, or torch.compile with Inductor", "XLA",
     "Turns your model into something the chip runs. XLA is not optional on TPU, which means its "
     "decisions are your performance whether you asked or not."),
    ("intermediate representation", "PTX, then SASS", "StableHLO, then LLO",
     "What to read when you need to know what the compiler actually did. Our S(1) finding came out "
     "of reading the optimised HLO."),
    ("how a slice is named", "8&times;H100 means eight GPUs",
     "v5p-16 means <em>eight chips</em>",
     "A trap that cost us real money. For v5p and v4 the number counts TensorCores and there are "
     "two per chip, so v5p-32 is 16 chips. For v5e and v6e the number counts chips. Four scripts "
     "in this repository had this wrong and billed our v5p at double."),
]

# name, notation, what it shards, what it costs in collectives, our measurement if we have one
PARALLELISM = [
    ("Data parallel", "DP8", "the batch; every device keeps a full copy of the weights",
     "one all_reduce of gradients per step",
     "Cheapest to reason about, most expensive in memory. Nothing is sharded except work."),
    ("Tensor parallel", "TP8", "each weight matrix, across devices",
     "two all_reduce per transformer block, on the critical path",
     "Wants the fattest links you have. On v6e a host holds 4 chips, so single-host TP caps at "
     "TP4 &mdash; we tried TP8 on one host and it cannot work."),
    ("Pipeline parallel", "PP4", "the layers; each stage owns a contiguous block",
     "point-to-point activation sends, no collective",
     "Cheap on bandwidth, costly in idle time: the bubble is (stages-1)/microbatches."),
    ("Expert parallel", "EP8", "the experts of an MoE layer",
     "two all_to_all per MoE layer, to dispatch and to combine",
     "We measured the routing tax against a dense layer at matched active FLOPs: 1.07&times; and "
     "0.98&times;. The movement is close to free at these sizes; the earlier 175%-of-peak figure "
     "was our own bug."),
    ("Sequence or context parallel", "SP, CP", "the sequence dimension of the activations",
     "all_gather or a ring exchange inside attention",
     "How long-context training fits at all. Turns attention into a fabric problem."),
    ("Fully sharded data parallel", "FSDP, or ZeRO-3",
     "weights, gradients <em>and</em> optimiser state across the data-parallel ranks",
     "all_gather the weights of each layer just before using it, reduce_scatter the gradients after",
     "The memory win is 1/N of the model instead of a full copy. The price is two collectives per "
     "layer per step instead of one per step, so it trades bandwidth for capacity."),
    ("ZeRO stages", "ZeRO-1, -2, -3",
     "stage 1 the optimiser state, stage 2 also gradients, stage 3 also the weights",
     "increasing amounts of all_gather and reduce_scatter",
     "ZeRO-3 and FSDP are the same idea from two codebases. If someone says FSDP they mean ZeRO-3."),
    ("Combined", "TP8 DP4, or 3D / 4D parallelism",
     "multiply them: TP8&nbsp;DP4 is 32 devices, 8-way tensor inside 4-way data",
     "the union of their collectives",
     "Read right to left for locality: the innermost factor should be the one needing the fattest "
     "links, which means TP inside a host or a small hop count, DP outermost."),
]


# layer, Google TPU, NVIDIA, AMD, Intel Gaudi, whether this study has actually touched it
STACKS = [
    ("the framework", "JAX, or PyTorch/XLA", "PyTorch, or JAX on CUDA",
     "PyTorch, ROCm build", "PyTorch + habana_frameworks", "both"),
    ("the compiler", "XLA, not optional", "Inductor, TensorRT, nvcc",
     "Inductor on ROCm, hipcc", "SynapseAI graph compiler", "both"),
    ("the IR you read when confused", "StableHLO, then LLO", "PTX, then SASS",
     "LLVM IR, then GCN ISA", "not public", "TPU only"),
    ("writing your own kernel", "Pallas, lowering via Mosaic", "CUDA C++, Triton, CUTLASS",
     "HIP, Triton, composable_kernel", "TPC-C", "both"),
    ("collectives", "XLA collectives in libtpu", "NCCL", "RCCL", "HCCL", "both"),
    ("the runtime", "libtpu", "CUDA driver", "ROCm and HSA", "SynapseAI runtime", "both"),
    ("what is on this box?", "jax.devices(), gcloud tpus list", "nvidia-smi", "rocm-smi", "hl-smi",
     "both"),
    ("the profiler", "jax.profiler, xprof", "Nsight Systems and Compute",
     "rocprof, Omnitrace", "Habana profiler", "neither yet"),
    ("getting one at all", "gcloud compute tpus tpu-vm create", "GKE device plugin, nvidia-docker",
     "k8s AMD device plugin", "k8s Habana plugin", "both"),
    ("the async copy engine", "SparseCore, and DMA via sync_copy",
     "TMA on Hopper, cp.async before it", "no direct equivalent", "DMA engines", "TPU only"),
]

GOTCHAS = [
    ("v5p-8 is four chips, not eight",
     "For v5p and v4 the suffix counts TensorCores, two per chip. Our budget guard billed a v5p-8 as "
     "eight v6e chips, $11.23/hr against a real $5.04, which is about $2,400 of imaginary spend "
     "across three weeks and enough to trigger a false shutdown."),
    ("There is no v6p",
     "The line is v5p, then v6e, then v7 Ironwood. Generation six shipped only the efficiency part, "
     "so v5p's 95 GB per chip is still the most memory a student can get."),
    ("JAX process 0 is not gcloud worker 0",
     "On a multi-host slice the process that writes your results file is whichever host JAX called "
     "rank 0, and it moves. Our results landed on worker 2 and then worker 3 on consecutive runs."),
    ("Quota, capacity, permission and service account are four separate gates",
     "We had 32 A100s of quota, no ability to call compute.instances.create, and got an A100 anyway "
     "by asking GKE Autopilot for one, because the node is created by Google's service agent rather "
     "than by us. Each gate has to be tested separately; none implies the others."),
    ("A collective that works on a whole slice can fail on a subset of it",
     "jax.lax.psum under shard_map returned INTERNAL: RET_CHECK failure for 8- and 16-chip "
     "sub-meshes of a v6e-256, on 8 of 48 configurations, while all_to_all succeeded on exactly the "
     "same meshes."),
    ("The timing process must own a chip in the mesh",
     "Selecting a sub-mesh the calling process owns nothing in measures a no-op. That produced an "
     "impossible 3.5 TB/s reading here before we caught it."),
]


def load(name: str) -> dict | None:
    p = DATA / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def measured_costs() -> tuple[list[tuple[str, str, str]], str]:
    """Pull the real collective numbers out of the newest fabric sweep, so this page cannot drift.

    Sorted by modification time, not by name. Sorting these filenames lexically picks
    `fabric_results_64chip_...` over `fabric_20260816T192359Z...`, because "r" sorts after "2", so
    the newest sweep loses to a six-hour-old one from a different generation. That is how this page
    first shipped a v6e-64 measurement labelled as v5p.
    """
    files = sorted(DATA.glob("fabric*.json"), key=lambda p: p.stat().st_mtime)
    for path in reversed(files):
        doc = load(path.name)
        if not doc:
            continue
        rows = [r for r in doc.get("records", []) if "ms" in r]
        if not rows:
            continue
        biggest = max(r["per_chip_mib"] for r in rows)
        most = max(r["chips"] for r in rows)
        out = []
        for op in ("all_reduce", "all_to_all"):
            hit = [r for r in rows if r["op"] == op and r["chips"] == most
                   and r["per_chip_mib"] == biggest]
            if hit:
                r = hit[0]
                out.append((op, f"{r['ms']:.4f} ms", f"{r.get('bus_gbs', 0):.1f} GB/s"))
        if out:
            kind = doc.get("meta", {}).get("device_kind", "unknown chip")
            return ([(f"{op} at {most} chips, {biggest:g} MiB per chip", ms, gbs)
                     for op, ms, gbs in out], kind)
    return [], "no sweep yet"


def main() -> None:
    costs, kind = measured_costs()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    dialect = "".join(
        f"<tr><td class='k'>{c}</td><td>{n}</td><td>{g}</td><td class='w'>{w}</td></tr>"
        for c, n, g, w in DIALECT)
    par = "".join(
        f"<tr><td class='k'>{n}<br><span class='sub'>{notation}</span></td><td>{shards}</td>"
        f"<td>{cost}</td><td class='w'>{note}</td></tr>"
        for n, notation, shards, cost, note in PARALLELISM)
    got = "".join(f"<li><b>{t}.</b> {d}</li>" for t, d in GOTCHAS)
    stacks = "".join(
        f"<tr><td class='k'>{l}</td><td>{g}</td><td>{n}</td><td>{a}</td><td>{i}</td>"
        f"<td class='w'>{t}</td></tr>" for l, g, n, a, i, t in STACKS)
    cost_rows = "".join(f"<tr><td>{what}</td><td class='n'>{ms}</td><td class='n'>{gbs}</td></tr>"
                        for what, ms, gbs in costs) or \
        "<tr><td colspan='3'>no fabric sweep collected yet</td></tr>"

    OUT.write_text(f"""<!doctype html><meta charset="utf-8">
<title>The same idea in three dialects</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 :root{{--bg:#fcfcfb;--surface:#fff;--ink:#1c1c20;--ink-2:#42424a;--ink-3:#75757f;--line:#dcd9d4;
   --band:#efece6;--accent:#b4531f;
   --mono:ui-monospace,"SF Mono",Menlo,monospace;
   --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
 @media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#111113;--surface:#191919;
   --ink:#f1f0ed;--ink-2:#b8b7b2;--ink-3:#86858d;--line:#2f2f34;--band:#212127;--accent:#e08a4c}}}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 var(--sans)}}
 .wrap{{max-width:1080px;margin:0 auto;padding:0 22px}}
 header{{padding:54px 0 26px;border-bottom:1px solid var(--line)}}
 h1{{font-size:clamp(27px,4.4vw,40px);letter-spacing:-.03em;margin:0 0 12px;font-weight:680;
   line-height:1.13}}
 h2{{font-size:15px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-3);
   margin:46px 0 12px;font-weight:620}}
 .sub{{color:var(--ink-3);font:500 12px var(--mono)}}
 p{{margin:0 0 14px;max-width:76ch;color:var(--ink-2)}}
 .lede{{font-size:18px;color:var(--ink)}}
 table{{width:100%;border-collapse:collapse;font-size:14px;background:var(--surface);
   border:1px solid var(--line);border-radius:12px;overflow:hidden;margin:14px 0}}
 th,td{{padding:11px 13px;border-bottom:1px solid var(--line);text-align:left;
   vertical-align:top}}
 th{{font:600 10px var(--mono);letter-spacing:.07em;text-transform:uppercase;color:var(--ink-3);
   background:var(--band)}}
 td.k{{font:600 13px var(--sans);color:var(--ink);white-space:nowrap}}
 td.w{{color:var(--ink-3);font-size:13.5px}}
 td.n{{text-align:right;font-family:var(--mono)}}
 .scroll{{overflow-x:auto}}
 ul{{max-width:76ch;color:var(--ink-2)}} li{{margin-bottom:11px}}
 li b{{color:var(--ink)}}
 footer{{margin-top:50px;padding:26px 0 64px;border-top:1px solid var(--line);color:var(--ink-3);
   font-size:13.5px}}
 a{{color:var(--ink)}}
</style>
<header><div class="wrap">
 <h1>The same idea in three dialects, and what each one costs</h1>
 <p class="sub">{now} &middot; measurements from this study, not from a datasheet</p>
 <p class="lede" style="margin-top:18px">Most of the difficulty in reading accelerator
 documentation is not conceptual. It is that NVIDIA and Google invented different words for the same
 wire, that <span class="sub">Tensor Core</span> and <span class="sub">TensorCore</span> are
 different levels of the hierarchy, and that notation like <span class="sub">TP8 DP4</span> is
 written as though you already know. This is the translation, with our own numbers attached wherever
 we measured the thing being named.</p>
</div></header>
<div class="wrap">

<h2>1. The same component, in each vendor's dialect</h2>
<div class="scroll"><table>
 <thead><tr><th>the thing</th><th>NVIDIA calls it</th><th>Google calls it</th>
  <th>what it actually is</th></tr></thead>
 <tbody>{dialect}</tbody></table></div>
<p><b>Intel is a third answer to the fabric question, and worth knowing about.</b> A Gaudi 3 puts
<span class="sub">24&times;200 GbE RoCE</span> ports on the die itself, so its scale-up network is
<em>standard Ethernet</em> rather than a proprietary fabric: no NVSwitch, no torus, just RoCE v2 out
of the chip. That is architecturally the most interesting of the three, because it means the fabric
is commodity and replaceable. On paper the part is competitive too: 128 GB of HBM2e at 3.7 TB/s is
more memory <em>and</em> more bandwidth than a v5p's 95 GB at 2.77 TB/s. The problem is not the
silicon. Falcon Shores, which was to merge Gaudi with Intel's GPU line, was cancelled; Gaudi itself
is slated for discontinuation across 2026&ndash;27; and its replacement, Jaguar Shores, is not
expected until 2027. So Intel is the cautionary case in this table: a genuinely good chip is not
enough without a roadmap and a software ecosystem, and the reason nobody in this class is running on
one is not that it is slow.</p>

<p>The row worth re-reading is the second one. Because NVLink is switched and ICI is a torus, a GPU
cluster has two fabrics with a cliff between them, while a TPU pod has one fabric all the way up.
That is also why geometry matters on a TPU and barely does inside an NVLink domain: on a torus your
traffic physically hops through chips you may not even own.</p>

<h2>2. The parallelism notation, and what each strategy pays for</h2>
<div class="scroll"><table>
 <thead><tr><th>strategy</th><th>what it shards</th><th>what it costs</th>
  <th>the part people leave out</th></tr></thead>
 <tbody>{par}</tbody></table></div>
<p>Read a combined spec right to left for locality. In <span class="sub">TP8 DP4</span> the tensor
groups are innermost, which is correct, because tensor parallel puts two all_reduce on the critical
path of every block and therefore wants the fattest, shortest links available. Data parallel
tolerates distance because it synchronises once per step.</p>

<h2>3. What those collectives actually cost, on {kind}</h2>
<div class="scroll"><table>
 <thead><tr><th>measured on {kind}</th><th class="n">per op</th>
  <th class="n">bus bandwidth</th></tr></thead>
 <tbody>{cost_rows}</tbody></table></div>
<p>Two things this study learned that change how you should place these groups. First, an
<span class="sub">all_reduce</span> gets <em>faster</em> as chips are added at fixed per-chip payload,
because a ring gains links, while <span class="sub">all_to_all</span> gets slower. So tensor
parallel and expert parallel scale in opposite directions and should not be sized by the same rule
of thumb. Second, and more practically: a mesh spanning four hosts ran 1.44&times; slower than the
same number of chips inside two hosts, independent of chip count. Where your groups land on host
boundaries is worth as much as which strategy you chose.</p>

<h2>4. The same stack, layer by layer, in four vendors' words</h2>
<p>This is the table we wish had existed when this study started. Each row is one layer you have to
touch to get a measurement out of a chip, and each column is what you actually type. The last column
is honest about which of these we have run on real hardware rather than read about.</p>
<div class="scroll"><table>
 <thead><tr><th>the layer</th><th>Google TPU</th><th>NVIDIA</th><th>AMD</th><th>Intel Gaudi</th>
  <th>touched here?</th></tr></thead>
 <tbody>{stacks}</tbody></table></div>
<p>Reading down the "touched here" column is the useful part. We have been through the whole Google
stack and most of NVIDIA's, and have touched neither AMD's nor Intel's at all. That is the actual gap
in this study, and it is a bigger gap than the difference between an A100 and an H100, which share
every row of this table. If the goal is to know the frontier rather than to collect numbers, one
MI300X run through ROCm teaches more than a second NVIDIA generation would.</p>
<p>The last row is the one that ties back to our own result. A SparseCore and Hopper's TMA are both
answers to "should moving data have its own engine, separate from the units doing arithmetic". Google
built a whole coprocessor; NVIDIA built a descriptor-driven copy engine inside the SM. Our gather
measurement is a way of asking which of those bets pays, and on the third architecture, AMD's, the
answer is currently "neither".</p>

<h2>5. Terminology that has actually cost us something</h2>
<ul>{got}</ul>
<p>Every one of these is in here because it wasted our time or our money first, which is the only
reason a list like this is worth writing down.</p>

<footer><div class="wrap">
 Generated by <span class="sub">build_glossary_page.py</span>; the collective costs are read from the
 newest sweep in <span class="sub">data/</span> so they cannot go stale. Companion pages:
 <a href="./">the measurement log</a>, <a href="./gather-cliff.html">the gather cliff</a>,
 <a href="./torus-shape.html">the wrong variable</a>, <a href="./roadmap.html">chips we can get</a>,
 <a href="./models.html">what we can run</a>.
</div></footer>
</div>
""", encoding="utf-8")
    print(f"wrote {OUT}: {len(DIALECT)} dialect rows, {len(PARALLELISM)} strategies, "
          f"{len(GOTCHAS)} gotchas, {len(costs)} measured collective costs")


if __name__ == "__main__":
    main()
