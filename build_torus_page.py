#!/usr/bin/env python3
"""Build torus-shape.html: the wrong variable and the right one, side by side.

This page exists because a measurement falsified the study's own headline mechanism. The claim was
that all_to_all cost is set by bisection bandwidth. On a 3D torus that claim is testable in a way it
never was on v6e, because you can hold the chip count fixed and change only the geometry. When we
did, bisection did not order the results and one configuration ruled it out outright: a set of eight
chips whose computed bisection is zero ran exactly as fast as a contiguous cube.

Two panels, because that is what the result is: the same eight measurements plotted against the
variable that does not explain them, and against the one that does.

    python3 build_torus_page.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "torus-shape.html"

W, PAD_L, PAD_R, PAD_T, PAD_B = 880, 80, 168, 26, 52
PLOT_H = 280


def load() -> dict | None:
    p = DATA / "torus_shape_v5p.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def hosts_of(rec: dict, long_axis: int) -> int:
    """Each host on this slice owns one plane of the long axis, so distinct values count hosts."""
    return len({tuple(c)[long_axis] for c in rec["coords"]})


def scatter(rows: list[dict], xkey: str, xlabel: str, xticks: list[int], caption_hint: str) -> str:
    """Time against one candidate explanation. Marker shape encodes chip count, so the two panels
    can be read against each other without a legend lookup."""
    ys = [r["ms"] for r in rows]
    ymin, ymax = min(ys) * 0.88, max(ys) * 1.08
    xs = [r[xkey] for r in rows]
    xlo, xhi = min(xs) - 0.6, max(xs) + 0.6
    h = PLOT_H + PAD_T + PAD_B

    def px(v: float) -> float:
        return PAD_L + (v - xlo) / (xhi - xlo) * (W - PAD_L - PAD_R)

    def py(v: float) -> float:
        return PAD_T + PLOT_H - (v - ymin) / (ymax - ymin) * PLOT_H

    out = [f'<svg viewBox="0 0 {W} {h}" width="100%" role="img" '
           f'aria-label="all_to_all time against {xlabel}">']
    for t in (0.18, 0.20, 0.22, 0.24, 0.26, 0.28):
        if not ymin <= t <= ymax:
            continue
        y = py(t)
        out.append(f'<line class="grid" x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}"/>')
        out.append(f'<text class="ax" x="{PAD_L - 9}" y="{y + 4:.1f}" text-anchor="end">'
                   f'{t:.2f}</text>')
    out.append(f'<text class="ax-t" x="{PAD_L - 9}" y="{PAD_T - 10}" text-anchor="end">ms</text>')
    for t in xticks:
        out.append(f'<text class="ax" x="{px(t):.1f}" y="{h - 30}" text-anchor="middle">{t}</text>')
    out.append(f'<text class="ax-t" x="{(PAD_L + W - PAD_R) / 2:.0f}" y="{h - 10}" '
               f'text-anchor="middle">{xlabel}</text>')

    # Several configurations share an x value and land within a pixel of each other in y, so their
    # labels overlap into an unreadable smear and one point can hide another entirely. Stagger the
    # label positions within each column and draw a leader line wherever a label had to move.
    groups: dict[float, list[dict]] = {}
    for r in rows:
        groups.setdefault(r[xkey], []).append(r)
    for grp in groups.values():
        grp.sort(key=lambda r: r["ms"])
        last = None
        for r in grp:
            ly = py(r["ms"])
            if last is not None and ly - last < 15:
                ly = last + 15
            r["_ly"] = ly
            last = ly

    for r in rows:
        x, y = px(r[xkey]), py(r["ms"])
        cls = "fast" if r["ms"] < 0.22 else "slow"
        n = r["chips"]
        if n == 4:
            out.append(f'<circle class="pt {cls}" cx="{x:.1f}" cy="{y:.1f}" r="6">'
                       f'<title>{r["shape"]}: {r["chips"]} chips, {r["hosts"]} hosts, '
                       f'bisection {r["bisection"]}, {r["ms"]:.4f} ms</title></circle>')
        elif n == 8:
            out.append(f'<rect class="pt {cls}" x="{x - 5.5:.1f}" y="{y - 5.5:.1f}" width="11" '
                       f'height="11" rx="2"><title>{r["shape"]}: {r["chips"]} chips, '
                       f'{r["hosts"]} hosts, bisection {r["bisection"]}, {r["ms"]:.4f} ms</title>'
                       f'</rect>')
        else:
            out.append(f'<polygon class="pt {cls}" points="{x:.1f},{y - 7:.1f} '
                       f'{x + 6.4:.1f},{y + 4.6:.1f} {x - 6.4:.1f},{y + 4.6:.1f}">'
                       f'<title>{r["shape"]}: {r["chips"]} chips, {r["hosts"]} hosts, '
                       f'bisection {r["bisection"]}, {r["ms"]:.4f} ms</title></polygon>')
        ly = r.get("_ly", y)
        if abs(ly - y) > 3:
            out.append(f'<line class="leader" x1="{x + 8:.1f}" y1="{y:.1f}" '
                       f'x2="{x + 15:.1f}" y2="{ly:.1f}"/>')
        out.append(f'<text class="lbl" x="{x + 17:.1f}" y="{ly + 4:.1f}">{r["shape"]}</text>')
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    doc = load()
    if not doc:
        print("no data/torus_shape_v5p.json yet; nothing written")
        return
    dims = doc["meta"]["dims"]
    long_axis = dims.index(max(dims))

    rows = []
    for r in doc["records"]:
        ms = r.get("a2a_16mib_ms")
        if ms is None:
            continue
        rows.append({"shape": r["shape"], "chips": r["chips"], "ms": ms,
                     "bisection": r["bisection_links"], "hosts": hosts_of(r, long_axis),
                     "gbs": r.get("a2a_16mib_gbs")})
    if not rows:
        print("no timed records; nothing written")
        return

    by_bis = scatter(rows, "bisection", "computed bisection of the selected chips, in links",
                     sorted({r["bisection"] for r in rows}), "")
    by_hosts = scatter(rows, "hosts", "hosts the mesh spans",
                       sorted({r["hosts"] for r in rows}), "")

    fast = [r for r in rows if r["ms"] < 0.22]
    slow = [r for r in rows if r["ms"] >= 0.22]
    zero = next((r for r in rows if r["bisection"] == 0), None)
    cube = next((r for r in rows if r["shape"] == "cube_2x2x2"), None)

    table = "".join(
        f"<tr><td>{r['shape']}</td><td class='n'>{r['chips']}</td><td class='n'>{r['hosts']}</td>"
        f"<td class='n'>{r['bisection']}</td><td class='n'>{r['ms']:.4f}</td>"
        f"<td class='n'>{r['gbs']:.1f}</td></tr>"
        for r in sorted(rows, key=lambda r: r["ms"]))

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    OUT.write_text(f"""<!doctype html><meta charset="utf-8">
<title>The wrong variable</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 :root{{--bg:#fcfcfb;--surface:#fff;--ink:#1c1c20;--ink-2:#42424a;--ink-3:#75757f;--line:#dcd9d4;
   --band:#efece6;--fast:#1c1c20;--slow:#b4531f;
   --mono:ui-monospace,"SF Mono",Menlo,monospace;
   --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
 @media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#111113;--surface:#191919;
   --ink:#f1f0ed;--ink-2:#b8b7b2;--ink-3:#86858d;--line:#2f2f34;--band:#212127;--fast:#f1f0ed;
   --slow:#e08a4c}}}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 var(--sans)}}
 .wrap{{max-width:920px;margin:0 auto;padding:0 22px}}
 header{{padding:54px 0 26px;border-bottom:1px solid var(--line)}}
 h1{{font-size:clamp(27px,4.4vw,40px);letter-spacing:-.03em;margin:0 0 12px;font-weight:680;
   line-height:1.13}}
 h2{{font-size:15px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-3);
   margin:44px 0 12px;font-weight:620}}
 .sub{{color:var(--ink-3);font:500 12.5px var(--mono)}}
 p{{margin:0 0 14px;max-width:74ch;color:var(--ink-2)}}
 .lede{{font-size:18px;color:var(--ink)}}
 figure{{margin:20px 0 8px;background:var(--surface);border:1px solid var(--line);
   border-radius:14px;padding:16px 14px 6px}}
 figcaption{{font:13px/1.55 var(--sans);color:var(--ink-3);padding:4px 6px 10px;max-width:80ch}}
 .grid{{stroke:var(--line);stroke-width:1}}
 .ax{{font:11px var(--mono);fill:var(--ink-3)}}
 .ax-t{{font:600 10px var(--mono);fill:var(--ink-3);letter-spacing:.05em;text-transform:uppercase}}
 .pt{{stroke:var(--surface);stroke-width:2}}
 .pt.fast{{fill:var(--fast)}} .pt.slow{{fill:var(--slow)}}
 .lbl{{font:11px var(--mono);fill:var(--ink-2)}}
 .leader{{stroke:var(--ink-3);stroke-width:1;opacity:.5}}
 table{{width:100%;border-collapse:collapse;font:12.5px var(--mono);background:var(--surface);
   border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:12px 0}}
 th,td{{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left}}
 th{{font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3)}}
 td.n{{text-align:right}}
 .keys{{display:flex;gap:16px;list-style:none;padding:0 6px;margin:2px 0 0;flex-wrap:wrap}}
 .keys li{{font:12px var(--mono);color:var(--ink-2)}}
 footer{{margin-top:50px;padding:26px 0 64px;border-top:1px solid var(--line);color:var(--ink-3);
   font-size:13.5px}}
 a{{color:var(--ink)}}
</style>
<header><div class="wrap">
 <h1>A measurement that killed our own explanation</h1>
 <p class="sub">{now} &middot; TPU v5p, dims {dims[0]}&times;{dims[1]}&times;{dims[2]} &middot;
 all_to_all, 16 MiB per chip, 32 collectives chained per timing</p>
 <p class="lede" style="margin-top:18px">This study had been explaining collective cost with
 bisection bandwidth, the number of links crossing a cut through the mesh. A 3D torus lets you test
 that properly, because you can hold the chip count fixed and change only the shape. We did, and the
 explanation failed: a set of eight chips with a computed bisection of <em>zero</em> ran exactly as
 fast as a contiguous cube.</p>
</div></header>
<div class="wrap">

<h2>1. The variable that does not explain it</h2>
<figure>{by_bis}
 <ul class="keys"><li>circle: 4 chips</li><li>square: 8 chips</li><li>triangle: 16 chips</li>
  <li>dark: fast group</li><li>orange: slow group</li></ul>
 <figcaption>Time against bisection computed from the actual coordinates of the selected chips. If
 bisection set the cost, this would slope. Instead the points at bisection 4 straddle both groups,
 and the point at bisection 0 sits with the fastest.</figcaption>
</figure>
<p>{f'The set with bisection zero is <span class="sub">{zero["shape"]}</span>, two 2&times;2 planes '
   f'that are not adjacent, so it owns no link at all between its two halves. It ran in '
   f'{zero["ms"]:.4f} ms against the contiguous cube&rsquo;s {cube["ms"]:.4f} ms, a difference of '
   f'{abs(zero["ms"] - cube["ms"]) / cube["ms"] * 100:.1f}%. A model that predicts a set with no '
   f'internal links should be catastrophic, and measures it as free, is the wrong model.'
   if zero and cube else ''}</p>

<h2>2. The variable that does</h2>
<figure>{by_hosts}
 <figcaption>The same measurements against the number of hosts the mesh spans. Two groups, cleanly
 separated, with no overlap: {min(r['ms'] for r in fast):.3f}&ndash;{max(r['ms'] for r in fast):.3f} ms
 at one or two hosts against {min(r['ms'] for r in slow):.3f}&ndash;{max(r['ms'] for r in slow):.3f} ms
 at four. Chip count varies from 4 to 16 within each group and does not move it.</figcaption>
</figure>
<p>The step is between two hosts and four, not a smooth function of host count: one host and two
hosts are indistinguishable. Holding bisection at 2 and chip count at 4, spanning one host took
0.1872 ms, two hosts 0.1798 ms, and four hosts 0.2565 ms.</p>

<h2>3. What we cannot separate, and are not pretending to</h2>
<p>On this slice each host owns exactly one plane of the length-4 axis. So the number of hosts, the
span along the long axis, and the fraction of the mesh the timing process owns all move together,
and this experiment cannot tell them apart. Separating them needs a slice whose hosts are not
planes, which means more chips per host or a different generation.</p>
<p>What the experiment does settle is narrower and still useful: chip-adjacency bisection is not the
mechanism, and every earlier comparison in this study that varied chip count from 8 to 16 was also
varying host count from 2 to 4 without saying so. The honest version of the earlier finding is that
those numbers describe a host-boundary effect, and the bisection arithmetic that accompanied them
was a coincidence of the shapes we happened to be given.</p>

<h2>4. Every configuration</h2>
<table>
 <thead><tr><th>shape</th><th class="n">chips</th><th class="n">hosts</th>
  <th class="n">bisection</th><th class="n">ms</th><th class="n">GB/s</th></tr></thead>
 <tbody>{table}</tbody>
</table>
<p>Reproduce with <span class="sub">experiment11_torus_shape.py</span>, which computes the bisection
from the device coordinates rather than assuming a formula, and refuses to time a mesh the calling
process owns no chip in, because that mistake produced an impossible 3.5 TB/s reading earlier in this
study.</p>

<footer><div class="wrap">
 Generated by <span class="sub">build_torus_page.py</span> from
 <span class="sub">data/torus_shape_v5p.json</span>. Companion pages:
 <a href="./">the measurement log</a>, <a href="./gather-cliff.html">the gather cliff</a>,
 <a href="./roadmap.html">chips we can get</a>, <a href="./cluster.html">who is using the cluster</a>.
</div></footer>
</div>
""", encoding="utf-8")
    print(f"wrote {OUT}: {len(rows)} configurations, "
          f"{len(fast)} fast / {len(slow)} slow, long axis = index {long_axis}")


if __name__ == "__main__":
    main()
