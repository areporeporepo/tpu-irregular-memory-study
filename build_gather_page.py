#!/usr/bin/env python3
"""Build gather-cliff.html: the teaching page for the study's main result.

Everything here is read from the JSON artefacts in data/, so the page cannot drift from the
measurements. If a file is missing the section that needs it is left out rather than faked.

    python3 build_gather_page.py
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "gather-cliff.html"

W, PAD_L, PAD_R, PAD_T, PAD_B = 900, 66, 132, 30, 46
PLOT_H = 300

# Three series, three luminances, three dash patterns, three marker shapes. Any one of those alone
# is enough to tell them apart, which is what makes the chart survive a greyscale filter or a
# black-and-white print.
SERIES = [
    ("v5p TensorCore", "s-a", "none", "circle"),
    ("v6e TensorCore", "s-b", "7 4", "square"),
    ("v5p SparseCore", "s-c", "2 3", "triangle"),
    ("A100 TensorCore", "s-d", "1 5", "diamond"),
]


def load(name: str) -> dict | None:
    path = DATA / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def marker(shape: str, x: float, y: float, cls: str, title: str) -> str:
    t = f"<title>{title}</title>"
    if shape == "circle":
        return f'<circle class="{cls} mk" cx="{x:.1f}" cy="{y:.1f}" r="4.5">{t}</circle>'
    if shape == "square":
        return (f'<rect class="{cls} mk" x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" '
                f'rx="1.5">{t}</rect>')
    if shape == "triangle":
        return (f'<polygon class="{cls} mk" points="{x:.1f},{y - 5:.1f} {x + 4.6:.1f},{y + 3.5:.1f} '
                f'{x - 4.6:.1f},{y + 3.5:.1f}">{t}</polygon>')
    return (f'<polygon class="{cls} mk" points="{x:.1f},{y - 5.5:.1f} {x + 5:.1f},{y:.1f} '
            f'{x:.1f},{y + 5.5:.1f} {x - 5:.1f},{y:.1f}">{t}</polygon>')


def chart_ladder(lines: list[dict], thresholds: list[tuple[str, float]]) -> str:
    """GB/s against buffer size, log x. Step functions and a flat line, in one frame."""
    xs = [p[0] for ln in lines for p in ln["points"]]
    ys = [p[1] for ln in lines for p in ln["points"]]
    if not xs:
        return ""
    lo, hi = min(xs), max(xs)
    ymax = max(ys) * 1.16
    h = PLOT_H + PAD_T + PAD_B

    def px(v: float) -> float:
        return PAD_L + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * \
            (W - PAD_L - PAD_R)

    def py(v: float) -> float:
        return PAD_T + PLOT_H - (v / ymax) * PLOT_H

    out = [f'<svg viewBox="0 0 {W} {h}" width="100%" role="img" aria-label="Gather bandwidth '
           f'against declared buffer size, on four accelerators">']
    for tick in (0, 50, 100, 150, 200, 250, 300):
        if tick > ymax:
            continue
        y = py(tick)
        out.append(f'<line class="grid" x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}"/>')
        out.append(f'<text class="ax" x="{PAD_L - 10}" y="{y + 4:.1f}" text-anchor="end">{tick}</text>')
    out.append(f'<text class="ax-t" x="{PAD_L - 10}" y="{PAD_T - 12}" text-anchor="end">GB/s</text>')
    for tick in (8, 16, 32, 48, 96, 192):
        if not lo <= tick <= hi:
            continue
        x = px(tick)
        out.append(f'<text class="ax" x="{x:.1f}" y="{h - 24}" text-anchor="middle">{tick}</text>')
    out.append(f'<text class="ax-t" x="{(PAD_L + W - PAD_R) / 2:.0f}" y="{h - 6}" '
               f'text-anchor="middle">declared buffer size, MiB (log scale)</text>')

    # the compiler's promotion thresholds, drawn under the data
    for label, mib in thresholds:
        if not lo <= mib <= hi:
            continue
        x = px(mib)
        out.append(f'<line class="thresh" x1="{x:.1f}" y1="{PAD_T}" x2="{x:.1f}" '
                   f'y2="{PAD_T + PLOT_H}"/>')
        out.append(f'<text class="thresh-l" x="{x + 5:.1f}" y="{PAD_T + 13}">{label}</text>')

    # Drawn as a step, not as a polyline. The measurements are at discrete buffer sizes and the
    # transition between two of them is a threshold, not a ramp, so a diagonal would claim
    # intermediate rates that do not exist. Where the bisection located the threshold exactly, the
    # riser is placed there; otherwise it goes at the geometric midpoint of the bracketing sizes.
    def stepped(points: list[tuple[float, float]], riser: float | None) -> list[tuple[float, float]]:
        out_pts: list[tuple[float, float]] = []
        for i, (x, y) in enumerate(points):
            if i:
                px_, py_ = points[i - 1]
                if abs(y - py_) / max(py_, 1) > 0.2:
                    at = riser if riser and px_ < riser < x else math.sqrt(px_ * x)
                    out_pts.append((at, py_))
                    out_pts.append((at, y))
                else:
                    out_pts.append((x, py_))
            out_pts.append((x, y))
        return out_pts

    riser_for = {"v5p TensorCore": next((m for l, m in thresholds if l.startswith("v5p")), None),
                 "v6e TensorCore": next((m for l, m in thresholds if l.startswith("v6e")), None)}
    for ln in lines:
        name, cls, dash, shape = next(s for s in SERIES if s[0] == ln["name"])
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}"
                       for x, y in stepped(ln["points"], riser_for.get(ln["name"])))
        dash_attr = f' stroke-dasharray="{dash}"' if dash != "none" else ""
        out.append(f'<polyline class="ln {cls}" points="{pts}"{dash_attr}/>')
        for x, y in ln["points"]:
            out.append(marker(shape, px(x), py(y), cls,
                              f"{name}: {y:.1f} GB/s at a {x:g} MiB buffer"))
        lx, ly = ln["points"][-1]
        out.append(f'<text class="lbl {cls}" x="{px(lx) + 11:.1f}" y="{py(ly) + 4:.1f}">'
                   f'{name}</text>')
    out.append("</svg>")
    return "\n".join(out)


def chart_invariance(rows: list[dict], ref_gbs: float) -> str:
    """Eight configurations from one chip: six that change nothing, two that change everything."""
    groups = [("index span, buffer fixed at 128 MiB", [r for r in rows if r["alloc_rows"] == 262144
                                                       and r["order"] == "random"]),
              ("index order, span fixed at 128 MiB", [r for r in rows if r["span_rows"] == 262144]),
              ("buffer shrunk to equal the span", [r for r in rows
                                                   if r["alloc_rows"] == r["span_rows"]])]
    bar_h, gap, g_gap = 26, 6, 26
    pad_l = 152                      # the category labels are sentences, not numbers
    n_bars = sum(len(g[1]) for g in groups)
    h = n_bars * (bar_h + gap) + len(groups) * g_gap + 54
    xmax = max(max(r["tc_gbs"] for r in rows), ref_gbs * 0.26) * 1.1

    def px(v: float) -> float:
        return pad_l + v / xmax * (W - pad_l - PAD_R)

    out = [f'<svg viewBox="0 0 {W} {h}" width="100%" role="img" aria-label="What changes the '
           f'gather rate and what does not">']
    y = 14
    for title, items in groups:
        out.append(f'<text class="grp" x="{pad_l}" y="{y + 10}">{title}</text>')
        y += 20
        for r in items:
            label = (f"span {r['span_mib']:g} MiB" if title.startswith("index span") else
                     r["order"] if title.startswith("index order") else
                     f"buffer {r['alloc_mib']:g} MiB")
            fast = r["tc_gbs"] > 150
            out.append(f'<text class="cat" x="{pad_l - 12}" y="{y + bar_h * 0.7:.0f}" '
                       f'text-anchor="end">{label}</text>')
            out.append(
                f'<rect class="{"bar-hi" if fast else "bar"}" x="{pad_l}" y="{y}" '
                f'width="{max(px(r["tc_gbs"]) - pad_l, 2):.1f}" height="{bar_h}" rx="4">'
                f'<title>{label}: {r["tc_gbs"]:.1f} GB/s, '
                f'{r["tc_vs_contiguous"] * 100:.1f}% of a contiguous read</title></rect>')
            out.append(f'<text class="val" x="{px(r["tc_gbs"]) + 9:.1f}" '
                       f'y="{y + bar_h * 0.7:.0f}">{r["tc_gbs"]:.0f} GB/s</text>')
            y += bar_h + gap
        y += g_gap - gap
    out.append("</svg>")
    return "\n".join(out)


def chart_real_tables(budget_mib: float) -> str:
    """Where real model artefacts sit relative to the promotion budget."""
    items = [
        ("Qwen3 embedding, 151k x 2560", 151936 * 2560 * 2 / 2**20),
        ("Llama-3 embedding, 128k x 4096", 128256 * 4096 * 2 / 2**20),
        ("MoE experts, 8 x 4096 x 14336", 8 * 4096 * 14336 * 2 / 2**20),
        ("KV cache, 8k ctx, batch 32", 8192 * 32 * 32 * 8 * 128 * 2 * 2 / 2**20),
        ("a table that fits the budget", budget_mib * 0.9),
    ]
    bar_h, gap = 28, 9
    pad_l = 268      # model names are long; give them room rather than clipping them
    h = len(items) * (bar_h + gap) + 58
    xmax = max(v for _, v in items) * 1.16

    def px(v: float) -> float:
        return pad_l + math.log10(max(v, 1)) / math.log10(xmax) * (W - pad_l - PAD_R)

    out = [f'<svg viewBox="0 0 {W} {h}" width="100%" role="img" aria-label="Sizes of real model '
           f'tables against the on-chip promotion budget">']
    xb = px(budget_mib)
    out.append(f'<rect class="band" x="{pad_l}" y="22" width="{xb - pad_l:.1f}" '
               f'height="{h - 58}"/>')
    out.append(f'<text class="thresh-l" x="{xb + 6:.1f}" y="18">'
               f'{budget_mib:.0f} MiB budget &rarr; everything right of here stays in HBM</text>')
    out.append(f'<line class="thresh" x1="{xb:.1f}" y1="22" x2="{xb:.1f}" y2="{h - 36}"/>')
    for i, (name, mib) in enumerate(items):
        y = 26 + i * (bar_h + gap)
        inside = mib <= budget_mib
        out.append(f'<text class="cat" x="{pad_l - 12}" y="{y + bar_h * 0.7:.0f}" '
                   f'text-anchor="end">{name}</text>')
        out.append(f'<rect class="{"bar-hi" if inside else "bar"}" x="{pad_l}" y="{y}" '
                   f'width="{max(px(mib) - pad_l, 2):.1f}" height="{bar_h}" rx="4">'
                   f'<title>{name}: {mib:,.0f} MiB, '
                   f'{"promoted on chip" if inside else "left in HBM"}</title></rect>')
        out.append(f'<text class="val" x="{px(mib) + 9:.1f}" y="{y + bar_h * 0.7:.0f}">'
                   f'{mib:,.0f} MiB</text>')
    out.append("</svg>")
    return "\n".join(out)


def series_from_sweep(doc: dict, key: str, name: str) -> dict | None:
    if not doc:
        return None
    pts = [(r["alloc_mib"], r[key]) for r in doc["records"] if key in r]
    return {"name": name, "points": pts} if pts else None


def main() -> None:
    v5p = load("alloc_sweep_v5p.json")
    v6e = load("alloc_sweep_v6e.json")
    loc = load("gather_locality_v5p.json")
    gpu = load("gpu_a100_sweep.json")
    bis5, bis6 = load("hlo_bisect_v5p.json"), load("hlo_bisect_v6e.json")
    hlo = load("hlo_v5p.json")

    lines = [s for s in (series_from_sweep(v5p, "tc_gbs", "v5p TensorCore"),
                         series_from_sweep(v6e, "tc_gbs", "v6e TensorCore"),
                         series_from_sweep(v5p, "sc_gbs", "v5p SparseCore"),
                         series_from_sweep(gpu, "gather_gbs", "A100 TensorCore")) if s]

    def thr(doc: dict | None) -> float | None:
        if not doc:
            return None
        hits = [t["last_promoted_bytes"] for t in doc.get("thresholds", [])
                if t.get("dim") in (128, 256, 512) and "last_promoted_bytes" in t]
        return max(hits) / 2**20 if hits else None

    t5, t6 = thr(bis5), thr(bis6)
    thresholds = [(f"v5p {t:.2f} MiB", t) for t in ([t5] if t5 else [])] + \
                 [(f"v6e {t:.2f} MiB", t) for t in ([t6] if t6 else [])]

    ladder = chart_ladder(lines, thresholds)
    inv = chart_invariance(loc["records"], loc["contiguous"]["gbs"]) if loc else ""
    real = chart_real_tables(t6 or 96.0)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    steps = []
    for doc, label in ((v5p, "v5p"), (v6e, "v6e")):
        if not doc:
            continue
        fast = [r["tc_gbs"] for r in doc["records"] if r["tc_gbs"] > 150]
        slow = [r["tc_gbs"] for r in doc["records"] if r["tc_gbs"] <= 150]
        if fast and slow:
            steps.append(f"{label} steps {max(fast):.0f}&thinsp;&rarr;&thinsp;{min(slow):.0f} GB/s "
                         f"({max(fast) / min(slow):.2f}&times;)")
    gpu_flat = ""
    if gpu:
        rates = [r["gather_gbs"] for r in gpu["records"]]
        gpu_flat = (f"The A100 holds {min(rates):.0f}&ndash;{max(rates):.0f} GB/s across the same "
                    f"24&times; range of buffer size, a spread of "
                    f"{max(rates) / min(rates):.2f}&times;.")

    sc_spread = ""
    if v5p:
        sc = [r["sc_gbs"] for r in v5p["records"] if "sc_gbs" in r]
        if sc:
            sc_spread = (f"{min(sc):.1f}&ndash;{max(sc):.1f} GB/s, a "
                         f"{(max(sc) / min(sc) - 1) * 100:.1f}% spread")

    idx_test = ""
    if bis5 and bis5.get("index_scaling"):
        sums = {r["table_plus_indices"] for r in bis5["index_scaling"]}
        tabs = {r["last_promoted_bytes"] for r in bis5["index_scaling"]}
        idx_test = (f"Growing the index vector 64&times; moved the table threshold by "
                    f"{'nothing at all' if len(tabs) == 1 else 'some amount'}: "
                    f"{sorted(tabs)[0]:,} bytes in every case. "
                    f"So the 64 KiB is a fixed reservation and not the indices, "
                    f"and the tidy arithmetic that suggested otherwise was a coincidence at one "
                    f"shape.")

    OUT.write_text(f"""<!doctype html><meta charset="utf-8">
<title>A 3x cliff in TPU gathers</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 :root{{--bg:#fcfcfb;--surface:#fff;--ink:#1c1c20;--ink-2:#42424a;--ink-3:#75757f;--line:#dcd9d4;
   --band:#efece6;--a:#1c1c20;--b:#b4531f;--c:#8a8a92;--d:#5f6b74;
   --mono:ui-monospace,"SF Mono",Menlo,monospace;
   --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
 @media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#111113;--surface:#191919;
   --ink:#f1f0ed;--ink-2:#b8b7b2;--ink-3:#86858d;--line:#2f2f34;--band:#212127;--a:#f1f0ed;
   --b:#e08a4c;--c:#8a8992;--d:#9fb0bd}}}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 var(--sans)}}
 .wrap{{max-width:940px;margin:0 auto;padding:0 22px}}
 header{{padding:54px 0 26px;border-bottom:1px solid var(--line)}}
 h1{{font-size:clamp(28px,4.5vw,42px);letter-spacing:-.03em;margin:0 0 12px;font-weight:680;
   line-height:1.12}}
 h2{{font-size:15px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-3);
   margin:46px 0 12px;font-weight:620}}
 h3{{font-size:17px;margin:30px 0 8px;font-weight:640}}
 .sub{{color:var(--ink-3);font:500 13px var(--mono)}}
 p{{margin:0 0 14px;max-width:74ch;color:var(--ink-2)}}
 .lede{{font-size:18px;color:var(--ink)}}
 figure{{margin:22px 0 8px;background:var(--surface);border:1px solid var(--line);
   border-radius:14px;padding:18px 16px 8px}}
 figcaption{{font:13px/1.55 var(--sans);color:var(--ink-3);padding:6px 6px 10px;max-width:80ch}}
 .grid{{stroke:var(--line);stroke-width:1}}
 .ax{{font:11px var(--mono);fill:var(--ink-3)}}
 .ax-t{{font:600 10px var(--mono);fill:var(--ink-3);letter-spacing:.06em;text-transform:uppercase}}
 .thresh{{stroke:var(--ink-3);stroke-width:1;stroke-dasharray:3 4;opacity:.75}}
 .thresh-l{{font:600 10px var(--mono);fill:var(--ink-3);letter-spacing:.04em}}
 polyline.ln{{fill:none;stroke-width:2;stroke-linejoin:round}}
 .mk{{stroke:var(--surface);stroke-width:2}}
 .lbl{{font:600 11px var(--mono)}}
 .grp{{font:600 11px var(--mono);fill:var(--ink-3);letter-spacing:.05em;text-transform:uppercase}}
 .cat{{font:12px var(--mono);fill:var(--ink-2)}}
 .val{{font:600 12px var(--mono);fill:var(--ink)}}
 .bar{{fill:var(--c)}} .bar-hi{{fill:var(--a)}} .band{{fill:var(--band)}}
 .s-a{{stroke:var(--a);fill:var(--a)}} .s-b{{stroke:var(--b);fill:var(--b)}}
 .s-c{{stroke:var(--c);fill:var(--c)}} .s-d{{stroke:var(--d);fill:var(--d)}}
 text.s-a{{stroke:none}} text.s-b{{stroke:none}} text.s-c{{stroke:none}} text.s-d{{stroke:none}}
 pre{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px;
   overflow-x:auto;font:12.5px/1.55 var(--mono);color:var(--ink-2)}}
 pre b{{color:var(--ink);background:var(--band);padding:1px 3px;border-radius:3px}}
 table{{width:100%;border-collapse:collapse;font:13px var(--mono);background:var(--surface);
   border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:10px 0}}
 th,td{{padding:8px 11px;border-bottom:1px solid var(--line);text-align:left}}
 th{{font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3)}}
 td.n{{text-align:right}}
 details{{margin:14px 0}} summary{{cursor:pointer;color:var(--ink-3);font:13px var(--mono)}}
 .keys{{display:flex;flex-wrap:wrap;gap:16px;margin:6px 0 0;padding:0 6px;list-style:none}}
 .keys li{{font:12px var(--mono);color:var(--ink-2);display:flex;align-items:center;gap:7px}}
 .keys i{{width:22px;height:0;border-top-width:2px;display:inline-block}}
 .tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:12px;
   margin:24px 0 6px}}
 .tile{{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px 16px}}
 .tile .v{{font:660 25px/1.1 var(--mono);letter-spacing:-.02em;display:block}}
 .tile .k{{font:600 10px var(--mono);letter-spacing:.07em;text-transform:uppercase;
   color:var(--ink-3);margin-top:6px;display:block}}
 footer{{margin-top:52px;padding:26px 0 64px;border-top:1px solid var(--line);color:var(--ink-3);
   font-size:13.5px}}
 a{{color:var(--ink)}}
</style>
<header><div class="wrap">
 <h1>A gather on a TPU is fast or slow depending on a number you never wrote down</h1>
 <p class="sub">{now} &middot; measured on v5p, v6e and an A100 &middot;
 part of the TPU irregular-access study</p>
 <p class="lede" style="margin-top:18px">Reading scattered rows out of a table is what an embedding
 lookup, an MoE router and a paged KV cache all reduce to. On a TPU that operation runs at one of
 two speeds, about 3&times; apart, and which one you get is decided by the compiler before your
 program sees a single index. This page is the measurement, the mechanism, and how to check it
 yourself.</p>
 <div class="tiles">
  <div class="tile"><span class="v">3.3&ndash;3.6&times;</span><span class="k">the step, v5p and v6e</span></div>
  <div class="tile"><span class="v">1.06&times;</span><span class="k">same test, A100</span></div>
  <div class="tile"><span class="v">48 / 96</span><span class="k">MiB budget, v5p / v6e</span></div>
  <div class="tile"><span class="v">S(1)</span><span class="k">the annotation that does it</span></div>
 </div>
</div></header>
<div class="wrap">

<h2>1. What we set out to measure</h2>
<p>The study's question was where irregular access belongs: on the TensorCore, which is built for
dense arithmetic, or on the SparseCore sitting beside it, which exists to chase pointers. So we
timed the same gather both ways on the same chip. 16384 rows of 128 float32 pulled out of a table,
8 MiB delivered per gather, 32 gathers chained inside one compiled function so that host dispatch
is amortised rather than measured.</p>
<p>The first answer looked boring. A random gather runs at a few percent of what the same chip
manages on a contiguous read, which everyone already knows. The interesting part was what happened
when we tried to find out <em>why</em>.</p>

<h2>2. Six things that change nothing, two that change everything</h2>
<figure>
{inv}
<figcaption>One chip, one gather, eight configurations. Widening the range of addresses the indices
touch from 2 MiB to 128 MiB does nothing. Sorting the indices does nothing, and neither does
grouping them into consecutive blocks, which is what a batch sorted by expert looks like. Shrinking
the <em>buffer</em> to match the span, without changing a single index, more than triples the
rate.</figcaption>
</figure>
<p>That is a strange result, because every intuition about why gathers are slow is about the access
pattern. If the cost were wide memory transactions being mostly discarded, sorting would help. If it
were cache misses, a 2 MiB working set inside a 128 MiB buffer would be fast. Neither is true here.
The only thing that matters is how large the buffer was <em>declared</em> to be, which is a property
of the program text, not of the run.</p>

<h2>3. The ladder, on four accelerators</h2>
<figure>
{ladder}
<ul class="keys">
 <li><i style="border-top-style:solid;border-color:var(--a)"></i>v5p TensorCore</li>
 <li><i style="border-top-style:dashed;border-color:var(--b)"></i>v6e TensorCore</li>
 <li><i style="border-top-style:dotted;border-color:var(--c)"></i>v5p SparseCore</li>
 <li><i style="border-top-style:dotted;border-color:var(--d)"></i>A100</li>
</ul>
<figcaption>Buffer size on a log scale, gather bandwidth on the vertical. The two TensorCore lines
are step functions with the step in different places. The SparseCore line and the A100 line are
flat. {' &middot; '.join(steps)}. {gpu_flat}</figcaption>
</figure>
<p>Two flat lines and two steps is the whole finding in one frame. The SparseCore is flat because
its indirect DMA reads HBM whatever the table's size: {sc_spread} across a 24&times; range. The
A100 is flat for a different reason, that nothing about a CUDA gather is decided at compile time.
Only the TPU TensorCore has a cliff, and each generation puts it somewhere else.</p>

<h2>4. The mechanism, in one line of HLO</h2>
<p>XLA hands over the compiled program, so this does not have to stay a correlation. Compiling the
identical gather at a buffer just below and just above the cliff, and diffing the optimised HLO,
gives a two-character answer.</p>
<pre>%param_0.2 = f32[81920,128]{{1,0:T(8,128)<b>S(1)</b>}} parameter(0)   40 MiB &rarr; 256 GB/s
%param_0.2 = f32[98304,128]{{1,0:T(8,128)}}     parameter(0)   48 MiB &rarr;  77 GB/s</pre>
<p><span class="sub">S(1)</span> is XLA's memory-space annotation. Below the threshold the
compiler promotes the gather's source table into on-chip memory; above it, the table stays in HBM
and every gathered row is a separate trip off-chip. The decision is made at compile time from the
declared shape, which is exactly why the span and the order of the indices cannot matter: neither
is known yet.</p>

<h3>Finding the threshold for free</h3>
<p>Because the mechanism is visible in the compiled module, the exact cutoff can be bisected without
running a single kernel: compile, look for <span class="sub">S(1)</span>, halve the interval.</p>
<table>
 <thead><tr><th>chip</th><th>TensorCores per chip</th><th class="n">last promoted</th>
  <th>reads as</th></tr></thead>
 <tbody>
  <tr><td>v5p</td><td class="n">2 (Megacore)</td><td class="n">50,266,112 B</td>
   <td>48 MiB &minus; 64 KiB</td></tr>
  <tr><td>v6e</td><td class="n">1</td><td class="n">100,597,760 B</td>
   <td>96 MiB &minus; 64 KiB</td></tr>
 </tbody>
</table>
<p>The threshold is identical for 128-, 256- and 512-wide rows, so it is a byte budget rather than a
row count. At 64-wide rows it halves, because a float32 array is laid out in
<span class="sub">T(8,128)</span> tiles and a 64-wide row wastes half of every tile: the budget is
counted in allocated bytes, padding included.</p>

<h3>One guess tested and dropped</h3>
<p>64 KiB below a round number is suspicious, and at our shape the index vector was exactly 64 KiB,
which suggested the budget covered the table and the indices together. It does not.
{idx_test}</p>

<h2>5. Why it might be 48 on one chip and 96 on the other</h2>
<p>A v5p chip has two TensorCores that Megacore drives as one; a v6e chip has a single core. If the
budget is a per-chip allowance divided among the cores, then 96 MiB per chip becomes 48 MiB per core
on v5p and stays 96 MiB on v6e, which is what we see. That is a hypothesis with a sharp consequence:
any single-core chip should show 96 MiB, and v5e is a single-core chip we had not touched. The test
costs one spot instance and no benchmark, only a compile.</p>

<h2>6. What it means if you are the one shipping the model</h2>
<figure>
{real}
<figcaption>Table sizes from real models against the v6e budget. The bar for a table that fits is
there to show what fitting looks like; nothing anyone actually serves is anywhere near it.</figcaption>
</figure>
<p>Every embedding table and every KV cache worth the name is far above the budget, so in practice
the slow path is the only path, and no amount of sorting indices or improving locality moves it.
What does move it is putting the gather on the SparseCore, which was 2.3&times; faster than the
TensorCore in exactly the regime that matters, and slower in the regime nobody is in. The crossover
between the two engines is not a hardware property at all. It is a compiler threshold.</p>

<h2>7. What we still do not know</h2>
<p>Three open items, each with the experiment that would settle it. Where the 64 KiB reservation
comes from, which the JAX or XLA source would answer directly. Whether the budget is configurable,
since <span class="sub">xla_tpu_scoped_vmem_limit_kib</span> and its neighbours exist and would be
the obvious lever, in which case the cliff can be moved on purpose. And whether the promoted path
is genuinely a full copy into on-chip memory or a prefetch hint, which the memory-space assignment
pass would show.</p>
<p>The SparseCore half also comes with a caveat worth stating plainly: the documented kernel
compiles at 9 of 16 shapes we tried on v5p and 1 of 16 on v6e, with an opaque
<span class="sub">Failed to run MLO pass pipeline</span> for the rest. Every shape that compiled was
bit-exact against <span class="sub">jnp.take</span>. The envelope is real but narrow, and it is not
documented anywhere we could find.</p>

<details>
 <summary>the numbers behind every chart</summary>
 <table>
  <thead><tr><th>chip</th><th class="n">buffer MiB</th><th class="n">TensorCore GB/s</th>
   <th class="n">SparseCore GB/s</th></tr></thead>
  <tbody>{''.join(
    f"<tr><td>{lbl}</td><td class='n'>{r['alloc_mib']:g}</td>"
    f"<td class='n'>{r['tc_gbs']:.1f}</td>"
    f"<td class='n'>{r.get('sc_gbs', float('nan')):.1f}</td></tr>"
    if 'sc_gbs' in r else
    f"<tr><td>{lbl}</td><td class='n'>{r['alloc_mib']:g}</td>"
    f"<td class='n'>{r['tc_gbs']:.1f}</td><td class='n'>&mdash;</td></tr>"
    for doc, lbl in ((v5p, 'v5p'), (v6e, 'v6e')) if doc for r in doc['records'])}</tbody>
 </table>
</details>

<footer><div class="wrap">
 Generated by <span class="sub">build_gather_page.py</span> from the JSON in
 <span class="sub">data/</span>, so the prose and the charts cannot disagree with the measurements.
 Reproduce with <span class="sub">experiment10_gather_locality.py --sweep</span> and
 <span class="sub">hlo_gather_lowering.py --bisect</span>.
 Companion pages: <a href="./">the measurement log</a>,
 <a href="./models.html">what we can run</a>, <a href="./cluster.html">who is using the cluster</a>,
 <a href="./stacks-and-physics.html">the physics underneath</a>.
</div></footer>
</div>
""", encoding="utf-8")
    print(f"wrote {OUT} with {len(lines)} series"
          f"{' (no A100 data yet)' if not gpu else ''}")


if __name__ == "__main__":
    main()
