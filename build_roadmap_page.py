#!/usr/bin/env python3
"""Build roadmap.html: the chips we measured, placed on the roadmap they belong to.

The study runs on v5e, v5p and v6e, which are the chips a student can actually get. Those are two
and three generations behind what the hyperscalers are installing, and four behind what is being
taped out. This page connects them, because the point of measuring an old chip carefully is to be
able to read a new one's spec sheet and know which numbers matter.

Every figure carries a source and a date, and figures that come from analyst modelling rather than
a vendor are marked as such. Two of the claims in here could not be re-verified today and say so.
The only numbers in this file that are ours are the ones the study measured.

    python3 build_roadmap_page.py
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "roadmap.html"

W, PAD_L, PAD_R = 900, 168, 118

# vendor: G google, N nvidia. `est` marks a figure from analyst modelling or unconfirmed reporting
# rather than a vendor datasheet. `ours` marks something this study measured.
CHIPS = [
    # name, vendor, year, hbm_gb, hbm_gbs, dense_tflops, precision, sparsecores, fabric, est
    ("TPU v5e",      "G", 2023,  16.0,   819.0,   197.0, "bf16", 0, "2D torus",  False),
    ("TPU v5p",      "G", 2023,  95.0,  2765.0,   459.0, "bf16", 4, "3D torus",  False),
    ("TPU v6e",      "G", 2024,  32.0,  1638.0,   918.0, "bf16", 2, "2D torus",  False),
    ("TPU v7 Ironwood", "G", 2026, 192.0, 7370.0, 4614.0, "fp8", 4, "3D torus",  False),
    ("TPU 8t Sunfish",  "G", 2027, 288.0, 9600.0,    0.0, "fp8", 4, "3D torus",  True),
    ("TPU 8i Zebrafish","G", 2027, 288.0, 7400.0,    0.0, "fp8", 0, "Boardfly",  True),
    ("A100 80GB",    "N", 2020,  80.0,  2039.0,   312.0, "bf16", 0, "NVLink 3",  False),
    ("H100 SXM",     "N", 2022,  80.0,  3350.0,   989.0, "bf16", 0, "NVLink 4",  False),
    ("B200",         "N", 2024, 192.0,  8000.0,  2250.0, "fp8",  0, "NVLink 5",  False),
    ("R200 Rubin",   "N", 2026, 288.0, 22000.0, 50000.0, "fp4",  0, "NVLink 6",  False),
]

# How much memory sits on the fast side of the cliff, per architecture, in MiB.
#
# This is the table that connects this study's own measurement to the rest of the industry. We found
# that a TPU gather runs 3.3x faster when its source table fits inside a fixed on-chip budget, which
# we measured at 48 MiB on v5p and 96 MiB on v6e. Two vendors have answered that same question by
# building chips whose entire memory is the fast tier: Groq with 230 MB of SRAM and no DRAM at all,
# Cerebras with 44 GB of it on one wafer. A GPU's equivalent is its L2, which is why the A100 in our
# measurements is mildly footprint-sensitive rather than blind to it.
#
# `measured` marks the two numbers this study produced. Everything else is a vendor figure.
FAST_TIER = [
    ("TPU v5p, promoted budget", 47.94, True, "measured by HLO bisection, this study"),
    ("TPU v6e, promoted budget", 95.94, True, "measured by HLO bisection, this study"),
    ("A100, L2 cache", 40.0, False, "NVIDIA datasheet"),
    ("H100, L2 cache", 50.0, False, "NVIDIA datasheet"),
    ("Groq LPU, SRAM (whole chip)", 230.0, False, "no DRAM on the part at all"),
    ("Cerebras WSE-3, SRAM (whole wafer)", 44.0 * 1024, False, "44 GB, at ~21 PB/s"),
]

SOURCES = [
    ("TPU v5e / v5p / v6e capacity and bandwidth", "Google Cloud TPU documentation",
     "cloud.google.com/tpu/docs/system-architecture-tpu-vm", "checked 2026-08-16", "vendor"),
    ("v6e SparseCore count, 2 cores, 256 KiB VMEM", "measured, plsc.get_sparse_core_info()",
     "this study", "2026-08-16", "ours"),
    ("v5p SparseCore count, 4 cores, 512 KiB VMEM", "measured, plsc.get_sparse_core_info()",
     "this study", "2026-08-16", "ours"),
    ("On-chip promotion budget, 48 MiB v5p and 96 MiB v6e", "measured by HLO bisection",
     "this study", "2026-08-16", "ours"),
    ("TPU v7 Ironwood, 192 GB, 7.37 TB/s, GA at Cloud Next",
     "Google Cloud / SemiAnalysis TPUv7 analysis", "newsletter.semianalysis.com/p/tpuv7-google-takes-a-swing-at-the",
     "April 2026", "vendor + analyst"),
    ("TPU 8t Sunfish: Broadcom, 2 compute dies, 8 stacks of 12-high HBM3e, ~30% more bandwidth "
     "than Ironwood, 2.7x perf/$ for training, TSMC 2nm, late 2027",
     "Google Cloud blog and press coverage of Cloud Next 2026",
     "cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive",
     "April 2026", "vendor"),
    ("TPU 8i Zebrafish: MediaTek, 1 compute die, 6 stacks HBM3e, 20-30% lower inference cost, "
     "80% better perf/$ at low latency for MoE, 2x perf/W", "same",
     "cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive",
     "April 2026", "vendor"),
    ("R200 Rubin: 288 GB HBM4 per package at 22 TB/s, 50 PFLOPS NVFP4 per GPU, 2H2026",
     "Glenn Lockwood's processor notes, collating NVIDIA GTC material",
     "glennklockwood.com/garden/processors/r200", "checked 2026-08-16", "vendor, collated"),
    ("VR200 NVL144: 144 R200 GPUs as 72 Vera Rubin superchips, 3.6 EF NVFP4, ~20.7 TB HBM4/rack",
     "NVIDIA GTC 2026 material as reported", "multiple, see above", "2026", "vendor"),
    ("A100 / H100 / B200 capacity and bandwidth", "NVIDIA datasheets",
     "nvidia.com", "checked 2026-08-16", "vendor"),
    ("A100 gather bandwidth, 222-236 GB/s flat across a 24x buffer range",
     "measured on a2-highgpu-1g via GKE Autopilot", "this study", "2026-08-16", "ours"),
    ("Cerebras WSE-3: 44 GB on-wafer SRAM, ~21 PB/s, 900,000 cores, 125 PFLOPS peak",
     "Cerebras and press coverage of the CS-3", "cerebras.ai/blog/cerebras-cs-3-vs-groq-lpu",
     "checked 2026-08-16", "vendor"),
    ("Groq LPU: 230 MB SRAM per chip, ~80 TB/s on-chip, no DRAM",
     "Cerebras' own CS-3 vs LPU comparison, and Groq material",
     "cerebras.ai/blog/cerebras-cs-3-vs-groq-lpu", "checked 2026-08-16", "vendor, partisan source"),
    ("A100 L2 40 MB, H100 L2 50 MB", "NVIDIA datasheets", "nvidia.com", "checked 2026-08-16",
     "vendor"),
]

UNVERIFIED = [
    "That TPU 8i removes the SparseCores in favour of a collectives acceleration engine. This was "
    "in earlier reporting we read but we could not re-confirm it from a vendor source today, so the "
    "SparseCore count for 8i in the table below is our reading of that reporting, not a fact.",
    "TPU 8t and 8i bandwidth figures. Google published a relative claim, roughly 30% above "
    "Ironwood for 8t, and not an absolute number. The absolute figures here are derived from that "
    "percentage and should be treated as arithmetic on a rounded claim.",
]


def load(name: str) -> dict | None:
    p = DATA / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def bars(title: str, key, fmt, note: str, logscale: bool = True) -> str:
    """One horizontal bar per chip, Google and NVIDIA distinguished by fill and by texture."""
    vals = [(c[0], key(c), c[1], c[9]) for c in CHIPS if key(c) > 0]
    bar_h, gap = 24, 7
    h = len(vals) * (bar_h + gap) + 46
    vmax = max(v for _, v, _, _ in vals)
    vmin = min(v for _, v, _, _ in vals)

    def px(v: float) -> float:
        span = W - PAD_L - PAD_R
        if not logscale:
            return PAD_L + v / vmax * span
        lo = vmin / 1.6
        return PAD_L + (math.log10(v) - math.log10(lo)) / (math.log10(vmax * 1.15) -
                                                           math.log10(lo)) * span

    out = [f'<svg viewBox="0 0 {W} {h}" width="100%" role="img" aria-label="{title}">']
    for i, (name, v, vendor, est) in enumerate(vals):
        y = 14 + i * (bar_h + gap)
        cls = "bar-g" if vendor == "G" else "bar-n"
        out.append(f'<text class="cat" x="{PAD_L - 12}" y="{y + bar_h * 0.72:.0f}" '
                   f'text-anchor="end">{name}</text>')
        out.append(f'<rect class="{cls}" x="{PAD_L}" y="{y}" '
                   f'width="{max(px(v) - PAD_L, 2):.1f}" height="{bar_h}" rx="4">'
                   f'<title>{name}: {fmt(v)}{" (estimated)" if est else ""}</title></rect>')
        if est:
            out.append(f'<rect class="hatch" x="{PAD_L}" y="{y}" '
                       f'width="{max(px(v) - PAD_L, 2):.1f}" height="{bar_h}" rx="4"/>')
        out.append(f'<text class="val" x="{px(v) + 9:.1f}" y="{y + bar_h * 0.72:.0f}">'
                   f'{fmt(v)}{" est" if est else ""}</text>')
    out.append(f'<text class="ax-t" x="{PAD_L}" y="{h - 8}">{note}</text>')
    out.append("</svg>")
    return "\n".join(out)


def chart_fast_tier() -> str:
    """On-chip memory across architectures, on a log scale, in the units our measurement produced."""
    bar_h, gap, pad_l = 26, 9, 250
    h = len(FAST_TIER) * (bar_h + gap) + 52
    vmax = max(v for _, v, _, _ in FAST_TIER) * 1.6
    vmin = min(v for _, v, _, _ in FAST_TIER) / 1.6

    def px(v: float) -> float:
        return pad_l + (math.log10(v) - math.log10(vmin)) / (math.log10(vmax) - math.log10(vmin)) \
            * (W - pad_l - PAD_R)

    def fmt(v: float) -> str:
        return f"{v / 1024:.0f} GB" if v >= 1024 else f"{v:.0f} MiB"

    out = [f'<svg viewBox="0 0 {W} {h}" width="100%" role="img" aria-label="On-chip fast-tier '
           f'memory by architecture">']
    for i, (name, mib, measured, note) in enumerate(FAST_TIER):
        y = 14 + i * (bar_h + gap)
        out.append(f'<text class="cat" x="{pad_l - 12}" y="{y + bar_h * 0.72:.0f}" '
                   f'text-anchor="end">{name}</text>')
        out.append(f'<rect class="{"bar-g" if measured else "bar-n"}" x="{pad_l}" y="{y}" '
                   f'width="{max(px(mib) - pad_l, 2):.1f}" height="{bar_h}" rx="4">'
                   f'<title>{name}: {fmt(mib)} &mdash; {note}</title></rect>')
        out.append(f'<text class="val" x="{px(mib) + 9:.1f}" y="{y + bar_h * 0.72:.0f}">'
                   f'{fmt(mib)}{" measured" if measured else ""}</text>')
    out.append(f'<text class="ax-t" x="{pad_l}" y="{h - 10}">log scale. dark: measured here. '
               f'light: vendor figures.</text>')
    out.append("</svg>")
    return "\n".join(out)


def chart_promotable(t5: float, t6: float) -> str:
    """The fraction of a chip's memory the compiler is willing to promote on chip, over time.

    This is the study's own measurement projected onto the roadmap. The budget is a fixed number of
    MiB; HBM per chip is growing by an order of magnitude a generation. So the share of a model that
    can live on the fast side of the cliff is collapsing, whether or not anyone intended that.
    """
    rows = [("TPU v5p", 95.0, t5), ("TPU v6e", 32.0, t6),
            ("TPU v7 Ironwood", 192.0, t6), ("TPU 8i Zebrafish", 288.0, t6)]
    bar_h, gap = 30, 12
    h = len(rows) * (bar_h + gap) + 52
    fracs = [(n, budget_mib / (hbm_gb * 1024) * 100, hbm_gb, budget_mib) for n, hbm_gb, budget_mib
             in rows]
    fmax = max(f for _, f, _, _ in fracs) * 1.25

    def px(v: float) -> float:
        return PAD_L + v / fmax * (W - PAD_L - PAD_R)

    out = [f'<svg viewBox="0 0 {W} {h}" width="100%" role="img" aria-label="Share of chip memory '
           f'eligible for on-chip promotion, by generation">']
    for i, (name, frac, hbm, budget) in enumerate(fracs):
        y = 14 + i * (bar_h + gap)
        measured = name in ("TPU v5p", "TPU v6e")
        out.append(f'<text class="cat" x="{PAD_L - 12}" y="{y + bar_h * 0.7:.0f}" '
                   f'text-anchor="end">{name}</text>')
        out.append(f'<rect class="{"bar-g" if measured else "bar-n"}" x="{PAD_L}" y="{y}" '
                   f'width="{max(px(frac) - PAD_L, 2):.1f}" height="{bar_h}" rx="4">'
                   f'<title>{name}: {budget:.0f} MiB promotable out of {hbm:.0f} GB HBM, '
                   f'{frac:.3f}%</title></rect>')
        if not measured:
            out.append(f'<rect class="hatch" x="{PAD_L}" y="{y}" '
                       f'width="{max(px(frac) - PAD_L, 2):.1f}" height="{bar_h}" rx="4"/>')
        out.append(f'<text class="val" x="{px(frac) + 9:.1f}" y="{y + bar_h * 0.7:.0f}">'
                   f'{frac:.3f}% of HBM{"" if measured else ", if the budget does not move"}</text>')
    out.append(f'<text class="ax-t" x="{PAD_L}" y="{h - 10}">solid: budget measured on that chip. '
               f'hatched: the same budget assumed forward.</text>')
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    bis5, bis6 = load("hlo_bisect_v5p.json"), load("hlo_bisect_v6e.json")

    def thr(doc, default):
        if not doc:
            return default
        hits = [t["last_promoted_bytes"] for t in doc.get("thresholds", [])
                if t.get("dim") in (128, 256, 512) and "last_promoted_bytes" in t]
        return max(hits) / 2**20 if hits else default

    t5, t6 = thr(bis5, 47.94), thr(bis6, 95.94)

    cap = bars("HBM capacity per chip or package", lambda c: c[3], lambda v: f"{v:,.0f} GB",
               "log scale. 8t and 8i are per-package figures from reported stack counts.")
    bw = bars("HBM bandwidth per chip or package", lambda c: c[4],
              lambda v: f"{v / 1000:.2f} TB/s" if v >= 1000 else f"{v:.0f} GB/s",
              "log scale. R200 is the 22 TB/s per-package figure; some sources say 13.")
    # FLOPs per byte at the roofline corner: peak arithmetic divided by peak bandwidth. Stated this
    # way round because it is the number a kernel has to beat, and because it reproduces the 560
    # FLOPs/byte corner this study already measured for v6e, which is a useful check on the table.
    bpf = bars("Arithmetic intensity needed to saturate the chip",
               lambda c: c[5] / c[4] * 1000 if c[5] else 0,
               lambda v: f"{v:,.0f} FLOPs/byte",
               "peak FLOPs divided by peak bandwidth, the roofline corner. mixed precisions down "
               "the list, so read the trend. higher means a kernel must find more arithmetic per "
               "byte it moves before the compute units can be kept busy.")
    promo = chart_promotable(t5, t6)
    fast = chart_fast_tier()

    src = "".join(
        f"<tr><td>{what}</td><td>{who}</td><td class='sub'>{where}</td>"
        f"<td class='sub'>{when}</td><td>{kind}</td></tr>"
        for what, who, where, when, kind in SOURCES)
    spec = "".join(
        f"<tr><td>{c[0]}</td><td>{'Google' if c[1] == 'G' else 'NVIDIA'}</td>"
        f"<td class='n'>{c[2]}</td><td class='n'>{c[3]:,.0f}</td>"
        f"<td class='n'>{c[4] / 1000:.2f}</td>"
        f"<td class='n'>{c[5]:,.0f} {c[6]}</td>"
        f"<td class='n'>{c[7] if c[7] else '&mdash;'}</td><td>{c[8]}</td>"
        f"<td>{'reported' if c[9] else 'datasheet'}</td></tr>" for c in CHIPS)
    unv = "".join(f"<li>{u}</li>" for u in UNVERIFIED)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    OUT.write_text(f"""<!doctype html><meta charset="utf-8">
<title>The chips we can get, and the chips that are coming</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 :root{{--bg:#fcfcfb;--surface:#fff;--ink:#1c1c20;--ink-2:#42424a;--ink-3:#75757f;--line:#dcd9d4;
   --band:#efece6;--g:#1c1c20;--n:#8a8a92;
   --mono:ui-monospace,"SF Mono",Menlo,monospace;
   --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
 @media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#111113;--surface:#191919;
   --ink:#f1f0ed;--ink-2:#b8b7b2;--ink-3:#86858d;--line:#2f2f34;--band:#212127;--g:#f1f0ed;
   --n:#8a8992}}}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 var(--sans)}}
 .wrap{{max-width:940px;margin:0 auto;padding:0 22px}}
 header{{padding:54px 0 26px;border-bottom:1px solid var(--line)}}
 h1{{font-size:clamp(27px,4.4vw,40px);letter-spacing:-.03em;margin:0 0 12px;font-weight:680;
   line-height:1.13}}
 h2{{font-size:15px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-3);
   margin:46px 0 12px;font-weight:620}}
 h3{{font-size:17px;margin:28px 0 8px;font-weight:640}}
 .sub{{color:var(--ink-3);font:500 12.5px var(--mono)}}
 p{{margin:0 0 14px;max-width:74ch;color:var(--ink-2)}}
 .lede{{font-size:18px;color:var(--ink)}}
 figure{{margin:22px 0 8px;background:var(--surface);border:1px solid var(--line);
   border-radius:14px;padding:18px 16px 6px}}
 figcaption{{font:13px/1.55 var(--sans);color:var(--ink-3);padding:4px 6px 10px;max-width:80ch}}
 .cat{{font:12px var(--mono);fill:var(--ink-2)}}
 .val{{font:600 11.5px var(--mono);fill:var(--ink)}}
 .ax-t{{font:600 10px var(--mono);fill:var(--ink-3);letter-spacing:.05em}}
 .bar-g{{fill:var(--g)}} .bar-n{{fill:var(--n)}}
 .hatch{{fill:url(#hatch)}}
 table{{width:100%;border-collapse:collapse;font:12.5px var(--mono);background:var(--surface);
   border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:10px 0}}
 th,td{{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
 th{{font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3)}}
 td.n{{text-align:right}}
 .scroll{{overflow-x:auto}}
 ul{{max-width:74ch;color:var(--ink-2)}} li{{margin-bottom:8px}}
 .keys{{display:flex;gap:18px;list-style:none;padding:0 6px;margin:2px 0 0}}
 .keys li{{font:12px var(--mono);color:var(--ink-2);display:flex;align-items:center;gap:7px}}
 .keys b{{width:16px;height:11px;border-radius:2px;display:inline-block}}
 footer{{margin-top:52px;padding:26px 0 64px;border-top:1px solid var(--line);color:var(--ink-3);
   font-size:13.5px}}
 a{{color:var(--ink)}}
</style>
<svg width="0" height="0" style="position:absolute"><defs>
 <pattern id="hatch" width="7" height="7" patternTransform="rotate(45)"
   patternUnits="userSpaceOnUse">
  <line x1="0" y1="0" x2="0" y2="7" stroke="var(--bg)" stroke-width="2.4" opacity=".55"/>
 </pattern>
</defs></svg>
<header><div class="wrap">
 <h1>The chips we can get, and the chips everyone is actually buying</h1>
 <p class="sub">{now} &middot; v5e, v5p, v6e measured here &middot; v7, 8t, 8i, A100 and Rubin
 from published figures</p>
 <p class="lede" style="margin-top:18px">This study runs on the accelerators a student can get hold
 of, which are two and three generations behind what is being installed and four behind what is
 being taped out. That is fine, and it is the reason to be careful: the value of measuring an old
 chip properly is being able to read a new chip's spec sheet and know which line matters. This page
 puts our measurements on the same axes as the roadmap.</p>
</div></header>
<div class="wrap">

<h2>1. The two lines everyone quotes</h2>
<figure>{cap}
 <ul class="keys"><li><b style="background:var(--g)"></b>Google</li>
  <li><b style="background:var(--n)"></b>NVIDIA</li>
  <li><b style="background:var(--n);opacity:.55"></b>hatched: reported, not a datasheet</li></ul>
 <figcaption>Memory per chip, or per package where the vendor sells packages. This is the number
 that decides whether a model fits at all, and it is the reason a 95 GB v5p is a better fine-tuning
 chip than a 32 GB v6e despite v6e having twice the arithmetic.</figcaption>
</figure>
<figure>{bw}
 <figcaption>Bandwidth per chip or package. Rubin's figure is the one to be careful with: 22 TB/s
 per package is what the collated NVIDIA material says, while some coverage carries 13 TB/s. Both
 numbers are in circulation and we have not seen the datasheet.</figcaption>
</figure>

<h2>2. The line nobody quotes, which is the one that bites</h2>
<figure>{bpf}
 <figcaption>Peak compute divided by peak bandwidth: the roofline corner, in FLOPs per byte. A
 kernel below its chip's number is bandwidth-bound no matter how well written it is. Precisions
 differ down the list so this is a trend rather than a like-for-like ratio, and the trend is the
 point: the bar for v6e lands on 560 FLOPs/byte, which is the corner this study measured
 independently, and Rubin's is four times higher again.</figcaption>
</figure>
<p>An irregular gather is the worst case for this trend, because it has no reuse to give. A random
row read is one row's worth of arithmetic per row's worth of traffic, forever. Which is why where
the gather runs, and out of which memory, is a question that gets more expensive with every
generation rather than less.</p>

<h2>3. Where our own measurement lands on the roadmap</h2>
<p>The study found that XLA promotes a gather's source table into on-chip memory when it fits inside
a fixed budget, and leaves it in HBM when it does not, with about a 3.3&times; difference in
delivered bandwidth. We measured that budget at {t5:.2f} MiB on v5p and {t6:.2f} MiB on v6e. It is a
fixed number of mebibytes. HBM per chip is not fixed; it is going up by roughly an order of
magnitude across the generations on this page.</p>
<figure>{promo}
 <figcaption>The share of a chip's memory that is eligible for the fast path, if the budget stays
 where we measured it. The first two bars are measured. The last two assume the budget does not
 move, which is exactly the assumption worth checking on an Ironwood the moment anyone can get
 one.</figcaption>
</figure>
<p>We cannot test v7, 8t or 8i, so those two bars are an extrapolation and are drawn as one. But the
extrapolation is falsifiable in a single afternoon by anyone with Ironwood access and the bisection
script in this repository, which needs no benchmark and no allocation, only a compile.</p>

<h2>4. Two vendors answered the same question by building the whole chip out of the fast tier</h2>
<figure>{fast}
 <figcaption>How much memory sits on the fast side of the cliff, per architecture, in the units this
 study's own measurement produced. The first two bars are ours, bisected out of the compiler. The
 last two are vendors who decided the answer to "what if the table always fits" is to build a part
 with no DRAM at all.</figcaption>
</figure>
<p>This is the comparison worth sitting with. Our measurement says a TPU gather is fast when its
table fits inside roughly 96 MiB and 3.3&times; slower when it does not. A Groq LPU is
<em>230 MB of SRAM and nothing else</em>, which is the same order of magnitude as the TPU's
promoted budget and is the whole chip rather than a compiler's discretionary allowance. Cerebras
went further and put 44 GB of SRAM on a single wafer at around 21 PB/s, which is roughly ten
thousand times the bandwidth of a v6e's HBM. Both are betting that the way to win at
memory-bound serving is to make the slow tier not exist.</p>
<p>The cost of that bet is capacity, and it shows up as a different failure. A 230 MB part cannot
hold a model at all on its own, so a Groq deployment is many chips wired together to hold weights
that a single v5p chip fits in its 95 GB, and the fabric becomes the constraint instead of the
memory. That is the same trade the TPU roadmap is making from the other direction, and it is why
the interesting number on a serving chip is not capacity or bandwidth alone but which of the two
runs out first for the model you actually have.</p>

<h2>5. What each design is betting on for irregular work</h2>
<p>Read down the SparseCore column of the table below and the disagreement is visible. Google put
four gather engines on v5p, two on v6e, and by the reporting on the eighth generation the training
chip keeps them while the inference chip spends that area on collectives instead. NVIDIA has never
shipped a gather engine at all, and does not need one: an A100 held 222&ndash;236 GB/s on the same
gather across a 24&times; range of table size, flat, because thousands of resident threads hide the
latency that a TPU has to schedule around.</p>
<p>So the two architectures fail at irregular access for different reasons, and only one of those
reasons is fixable in a compiler. That is the sharpest thing this study has to say, and it is only
sayable because the same script ran on both.</p>

<div class="scroll"><table>
 <thead><tr><th>chip</th><th>vendor</th><th class="n">year</th><th class="n">HBM GB</th>
  <th class="n">TB/s</th><th class="n">dense peak</th><th class="n">SparseCores</th>
  <th>fabric</th><th>figures</th></tr></thead>
 <tbody>{spec}</tbody></table></div>

<h2>6. What in here we are not sure about</h2>
<ul>{unv}</ul>
<p>Both are the kind of claim that is easy to repeat and hard to check, which is why they are
flagged here rather than buried. The measured numbers in this study do not depend on either.</p>

<h2>7. Sources</h2>
<div class="scroll"><table>
 <thead><tr><th>figure</th><th>source</th><th>where</th><th>as of</th><th>kind</th></tr></thead>
 <tbody>{src}</tbody></table></div>
<p>Analyst material is marked as such. SemiAnalysis in particular is well sourced but a large part
of any given post is its own modelling, and their eighth-generation numbers should be read as
informed estimates rather than as specifications. Anything marked <span class="sub">ours</span> was
measured on hardware this week and can be reproduced from this repository.</p>

<footer><div class="wrap">
 Generated by <span class="sub">build_roadmap_page.py</span>. Companion pages:
 <a href="./">the measurement log</a>, <a href="./gather-cliff.html">the gather cliff</a>,
 <a href="./models.html">what we can run</a>, <a href="./cluster.html">who is using the cluster</a>,
 <a href="./stacks-and-physics.html">the physics underneath</a>.
</div></footer>
</div>
""", encoding="utf-8")
    print(f"wrote {OUT}: {len(CHIPS)} chips, {len(SOURCES)} sourced figures, "
          f"budgets {t5:.2f}/{t6:.2f} MiB")


if __name__ == "__main__":
    main()
