#!/usr/bin/env python3
"""Which models can we actually run, and what would a token cost against the market?

Two filters. A candidate must be **listed on OpenRouter**, so there is a published price to measure
our cost against rather than a vendor claim, and it must **fit on chips we can obtain**. Popularity
on HuggingFace is deliberately not a criterion: the current trending flagship is a 2.4-trillion
parameter mixture needing roughly ninety v6e chips.

Output is two charts rather than tables, because both questions are magnitude comparisons against a
threshold, which is what a bar chart with a reference line is for.

Colour note: this is read on a display running a grayscale filter, so no hue carries meaning.
Weights and KV are two parts of one total, encoded as a two-step sequential ramp (monotonic
lightness) plus a 45-degree texture on the lighter step. The one accent colour is used only for
annotation, always beside a text label. Every segment is directly labelled and a table view is
included, which is the required relief for the lighter step's contrast.

    python3 build_model_page.py            # writes models.html
"""
from __future__ import annotations

import json
import math
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "models.html"
SPOT_RATE = 1.4033      # USD per v6e chip-hour, spot, us-east1, measured
HBM_PER_CHIP = 32       # GB
RESERVE = 0.10
CONTEXT, BATCH = 8192, 32
CHIP_COUNTS = [4, 16, 32, 64]

# Published v6e serving throughput, tokens/sec/chip, for models of this class. The band rather
# than a point, because the figure depends on batch size and sequence length.
PUBLISHED_LO, PUBLISHED_HI = 300, 1000

#   openrouter id -> (display, params, active, layers, full_attn, kv_heads, head_dim)
KNOWN = {
    "openai/gpt-oss-20b":          ("gpt-oss-20b",       21.0e9,  3.6e9, 24, 24, 8,  64),
    "openai/gpt-oss-120b":         ("gpt-oss-120b",     117.0e9,  5.1e9, 36, 36, 8,  64),
    "meta-llama/llama-4-scout":    ("Llama-4-Scout",    109.0e9, 17.0e9, 48, 48, 8, 128),
    "meta-llama/llama-4-maverick": ("Llama-4-Maverick", 400.0e9, 17.0e9, 48, 48, 8, 128),
    "deepseek/deepseek-v3.2":      ("DeepSeek-V3.2",    671.0e9, 37.0e9, 61, 61, 8, 128),
}


def openrouter() -> dict:
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=60) as r:
        return {m["id"]: m for m in json.load(r)["data"]}


def kv_gb(layers, full_attn, kv_heads, head_dim, context, batch, bits=16) -> float:
    return 2 * full_attn * kv_heads * head_dim * (bits / 8) * context * batch / 1e9


def collect() -> list[dict]:
    cat = openrouter()
    rows = []
    for rid, (name, params, active, layers, fa, kvh, hd) in KNOWN.items():
        m = cat.get(rid)
        if not m:
            continue
        pr = m.get("pricing", {})
        out_price = float(pr.get("completion") or 0) * 1e6
        w16 = params * 2 / 1e9
        kv = kv_gb(layers, fa, kvh, hd, CONTEXT, BATCH)
        smallest = next((c for c in CHIP_COUNTS
                         if w16 + kv <= c * HBM_PER_CHIP * (1 - RESERVE)), None)
        rows.append({
            "name": name, "params": params, "active": active,
            "in_price": float(pr.get("prompt") or 0) * 1e6, "out_price": out_price,
            "ctx": m.get("context_length") or 0, "w16": w16, "kv": kv, "total": w16 + kv,
            "smallest": smallest,
            # Throughput per chip needed to match the market price at our spot rate. Independent
            # of chip count: both cost and output scale linearly with chips.
            "breakeven": (SPOT_RATE / 3600) / (out_price / 1e6) if out_price else None,
        })
    return sorted(rows, key=lambda r: r["params"])


# ----------------------------------------------------------------------- chart helpers
W, PAD_L, PAD_R = 900, 168, 92
ROW_H, GAP = 34, 10


def logx(v: float, lo: float, hi: float) -> float:
    v = max(v, lo)
    return PAD_L + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * \
        (W - PAD_L - PAD_R)


def fmt_gb(v: float) -> str:
    return f"{v/1000:.1f} TB" if v >= 1000 else f"{v:.0f} GB"


def chart_memory(rows: list[dict]) -> str:
    """Stacked weights + KV against the capacity of 4, 16, 32 and 64 chips."""
    lo, hi = 10, 4000
    h = len(rows) * (ROW_H + GAP) + 92
    out = [f'<svg viewBox="0 0 {W} {h}" width="100%" role="img" '
           f'aria-label="Memory footprint of each model against chip capacity">']

    # capacity thresholds: recessive, labelled, drawn under the marks
    for c in CHIP_COUNTS:
        cap = c * HBM_PER_CHIP
        x = logx(cap, lo, hi)
        out.append(f'<line class="thresh" x1="{x:.1f}" y1="34" x2="{x:.1f}" y2="{h-46}"/>')
        out.append(f'<text class="thresh-l" x="{x:.1f}" y="26" text-anchor="middle">'
                   f'{c} chips</text>')
        out.append(f'<text class="thresh-v" x="{x:.1f}" y="{h-30}" text-anchor="middle">'
                   f'{fmt_gb(cap)}</text>')

    for i, r in enumerate(rows):
        y = 44 + i * (ROW_H + GAP)
        x0 = logx(lo, lo, hi)
        xw = logx(r["w16"], lo, hi)
        xt = logx(r["total"], lo, hi)
        out.append(f'<text class="cat" x="{PAD_L-12}" y="{y+ROW_H*0.68:.0f}" '
                   f'text-anchor="end">{r["name"]}</text>')
        # weights: dark step, square at the baseline end, rounded at the data end
        out.append(
            f'<rect class="w" x="{x0:.1f}" y="{y}" width="{max(xw-x0,2):.1f}" height="{ROW_H}" '
            f'rx="4"><title>{r["name"]}: weights {fmt_gb(r["w16"])} in bf16</title></rect>')
        # KV: lighter step + texture, separated from the weights by a 2px surface gap
        out.append(
            f'<rect class="kv" x="{xw+2:.1f}" y="{y}" width="{max(xt-xw-2,2):.1f}" '
            f'height="{ROW_H}" rx="4"><title>{r["name"]}: KV cache {fmt_gb(r["kv"])} at '
            f'{CONTEXT:,} context, batch {BATCH}</title></rect>')
        out.append(f'<text class="val" x="{xt+8:.1f}" y="{y+ROW_H*0.68:.0f}">'
                   f'{fmt_gb(r["total"])}</text>')
    out.append('</svg>')
    return "\n".join(out)


def chart_breakeven(rows: list[dict]) -> str:
    """Tokens/sec/chip required to match the market price, against published v6e throughput."""
    lo, hi = 100, 20000
    h = len(rows) * (ROW_H + GAP) + 92
    out = [f'<svg viewBox="0 0 {W} {h}" width="100%" role="img" '
           f'aria-label="Throughput per chip required to match market price, per model">']
    # the achievable band, as annotation
    xa, xb = logx(PUBLISHED_LO, lo, hi), logx(PUBLISHED_HI, lo, hi)
    out.append(f'<rect class="band" x="{xa:.1f}" y="34" width="{xb-xa:.1f}" height="{h-80}"/>')
    out.append(f'<text class="band-l" x="{(xa+xb)/2:.1f}" y="26" text-anchor="middle">'
               f'published v6e range</text>')
    for tick in (100, 1000, 10000):
        x = logx(tick, lo, hi)
        out.append(f'<text class="thresh-v" x="{x:.1f}" y="{h-30}" text-anchor="middle">'
                   f'{tick:,}</text>')
    for i, r in enumerate(rows):
        y = 44 + i * (ROW_H + GAP)
        if not r["breakeven"]:
            continue
        x0, xv = logx(lo, lo, hi), logx(r["breakeven"], lo, hi)
        reach = r["breakeven"] <= PUBLISHED_HI
        out.append(f'<text class="cat" x="{PAD_L-12}" y="{y+ROW_H*0.68:.0f}" '
                   f'text-anchor="end">{r["name"]}</text>')
        out.append(
            f'<rect class="{"w" if reach else "kv"}" x="{x0:.1f}" y="{y}" '
            f'width="{max(xv-x0,2):.1f}" height="{ROW_H}" rx="4">'
            f'<title>{r["name"]}: needs {r["breakeven"]:,.0f} tok/s/chip to match '
            f'${r["out_price"]:.3f} per million output tokens</title></rect>')
        label = f'{r["breakeven"]:,.0f} tok/s/chip'
        out.append(f'<text class="val" x="{xv+8:.1f}" y="{y+ROW_H*0.68:.0f}">{label}</text>')
    out.append('</svg>')
    return "\n".join(out)


def table_view(rows: list[dict]) -> str:
    trs = "".join(
        f"<tr><td>{r['name']}</td><td class='n'>{r['params']/1e9:.0f}B</td>"
        f"<td class='n'>{r['active']/1e9:.0f}B</td><td class='n'>{r['ctx']:,}</td>"
        f"<td class='n'>${r['in_price']:.3f}</td><td class='n'>${r['out_price']:.3f}</td>"
        f"<td class='n'>{fmt_gb(r['w16'])}</td><td class='n'>{fmt_gb(r['kv'])}</td>"
        f"<td class='n'>{r['smallest'] or '&mdash;'}</td>"
        f"<td class='n'>{f'{r['breakeven']:,.0f}' if r['breakeven'] else '&mdash;'}</td></tr>"
        for r in rows)
    return f"""<details><summary>Table view of the same data</summary>
<div class="scroll"><table>
<thead><tr><th>model</th><th class="n">params</th><th class="n">active</th><th class="n">context</th>
<th class="n">in $/M</th><th class="n">out $/M</th><th class="n">weights</th><th class="n">KV</th>
<th class="n">min chips</th><th class="n">break-even tok/s/chip</th></tr></thead>
<tbody>{trs}</tbody></table></div></details>"""


def main() -> None:
    rows = collect()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    OUT.write_text(f"""<title>What We Can Actually Run</title>
<style>
 :root{{--bg:#fcfcfb;--surface:#fff;--ink:#1c1c20;--ink-2:#4a4a52;--ink-3:#75757f;--line:#d8d5d0;
   --step-1:#1c1c20;--step-2:#b9b3aa;--accent:#b4531f;--grid:#e6e3de;
   --mono:ui-monospace,"SF Mono",Menlo,monospace;
   --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
 /* Dark mode is selected, not flipped: its own steps from the same ramp against a dark surface. */
 @media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
   --bg:#111113;--surface:#1a1a19;--ink:#f0efec;--ink-2:#b6b5b0;--ink-3:#85848c;--line:#2e2e33;
   --step-1:#f0efec;--step-2:#5c5952;--accent:#e08a4c;--grid:#26262b}}}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 var(--sans)}}
 .wrap{{max-width:1000px;margin:0 auto;padding:0 22px}}
 header{{padding:58px 0 28px;border-bottom:1px solid var(--line)}}
 h1{{font-size:clamp(30px,5vw,48px);line-height:1.05;letter-spacing:-.03em;margin:0 0 12px;font-weight:680}}
 h2{{font-size:19px;letter-spacing:-.01em;margin:44px 0 4px;font-weight:660}}
 .sub{{color:var(--ink-3);font:500 13px var(--mono)}}
 p{{margin:0 0 14px;max-width:74ch;color:var(--ink-2)}}
 strong{{color:var(--ink);font-weight:640}}
 figure{{margin:18px 0 8px;background:var(--surface);border:1px solid var(--line);
   border-radius:12px;padding:14px 16px 6px}}
 figcaption{{font-size:13.5px;color:var(--ink-3);margin-top:6px}}
 .legend{{display:flex;gap:18px;align-items:center;font:600 12px var(--mono);
   color:var(--ink-3);margin:2px 0 10px}}
 .key{{display:inline-block;width:13px;height:13px;border-radius:3px;vertical-align:-2px;margin-right:6px}}
 .k1{{background:var(--step-1)}}
 .k2{{background:var(--step-2)}}
 svg text{{font-family:var(--mono)}}
 .cat{{font-size:12.5px;fill:var(--ink-2)}}
 .val{{font-size:12px;fill:var(--ink);font-weight:600}}
 .thresh{{stroke:var(--accent);stroke-width:1.5;stroke-dasharray:3 4}}
 .thresh-l{{font-size:11px;fill:var(--accent);font-weight:700}}
 .thresh-v{{font-size:10.5px;fill:var(--ink-3)}}
 .band{{fill:var(--accent);opacity:.10}}
 .band-l{{font-size:11px;fill:var(--accent);font-weight:700}}
 rect.w{{fill:var(--step-1)}}
 rect.kv{{fill:var(--step-2)}}
 rect.w:hover,rect.kv:hover{{stroke:var(--surface);stroke-width:2}}
 details{{margin:14px 0 0}} summary{{cursor:pointer;font:600 13px var(--mono);color:var(--ink-3)}}
 table{{width:100%;border-collapse:collapse;font:13px var(--mono);margin-top:12px}}
 th,td{{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left}}
 th{{font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3)}}
 .n{{text-align:right;white-space:nowrap}} .scroll{{overflow-x:auto}}
 footer{{padding:34px 0 60px;color:var(--ink-3);font-size:13px}} a{{color:var(--ink)}}
</style>
<header><div class="wrap">
 <h1>What We Can Actually Run</h1>
 <p class="sub">generated {now} &middot; live OpenRouter prices &middot; v6e spot ${SPOT_RATE}/chip-hour</p>
</div></header>
<div class="wrap">
 <p>Two filters, in order. A model must be <strong>listed on OpenRouter</strong>, so it has a
 published price to measure our cost against instead of a vendor claim, and it must
 <strong>fit on chips we can obtain</strong>. Trending on HuggingFace is not a criterion: the
 current trending flagship is a 2.4-trillion-parameter mixture needing about ninety v6e chips.</p>

 <h2>Does it fit?</h2>
 <div class="legend">
   <span><span class="key k1"></span>weights, bf16</span>
   <span><span class="key k2"></span>KV cache at {CONTEXT:,} context, batch {BATCH}</span>
 </div>
 <figure>{chart_memory(rows)}
  <figcaption>Log scale. Dashed lines are the total HBM of 4, 16, 32 and 64 v6e chips at 32 GB
  each. A bar ending left of a line fits in that many chips, before the 10% reserved for compile
  scratch.</figcaption>
 </figure>
 <p>KV cache is the part people forget. For <strong>gpt-oss-20b</strong> it is small because the
 model uses 64-wide heads, but the ratio grows with context: raise the context to 262k and KV
 overtakes the weights for every model here.</p>

 <h2>Would a token pay for itself?</h2>
 <figure>{chart_breakeven(rows)}
  <figcaption>Throughput per chip we would need to match each model's market price for output
  tokens, at our measured spot rate. The shaded band is published v6e throughput for models of this
  class, roughly {PUBLISHED_LO} to {PUBLISHED_HI:,} tokens per second per chip. A bar extending
  past the band means serving is not economic at our scale.</figcaption>
 </figure>
 <p><strong>The answer splits by model size, and not in the direction you would guess.</strong>
 The cheap small models are hopeless: gpt-oss-20b needs about 3,000 tokens per second per chip to
 match its market price, roughly three times the top of the published band, because its price has
 been competed down to a tenth of a cent per million. The large mixtures are the opposite.
 Llama-4-Maverick needs only <strong>487</strong> and DeepSeek-V3.2 about <strong>975</strong>,
 both inside the achievable range, because the market charges several times more for them while
 their active parameter counts stay modest at 17B and 37B.</p>
 <p><strong>And then the cruel part.</strong> The two models where serving could plausibly pay for
 itself are exactly the two we cannot serve: Maverick needs 32 chips and DeepSeek-V3.2 needs 64,
 while our working serving path reaches four. The models that fit do not pay, and the models that
 would pay do not fit.</p>

 {table_view(rows)}

 <h2>What we chose, and why</h2>
 <p><strong>gpt-oss-20b.</strong> On OpenRouter with a published price, a mixture of experts at 21B
 total and 3.6B active which is where this study's questions live, and small enough to fit the
 four-chip path with room for context. A v6e host carries four chips, so single-host tensor
 parallelism caps at four; reaching more needs multi-host serving through Ray, which is not working
 here yet. That gap between what memory could hold and what the serving stack can address is the
 recurring lesson: 32 chips is a terabyte of HBM, enough for a 671B model in fp8, and the serving
 path reaches an eighth of it.</p>
</div>
<footer><div class="wrap">
 Regenerated by <span class="mono">build_model_page.py</span>, which reads the OpenRouter catalogue
 live. Charts use a two-step sequential ramp with monotonic lightness plus direct labels, so nothing
 depends on hue; the one accent colour appears only as annotation beside its own text label.
 Companion pages: <a href="./">the measurement log</a>,
 <a href="./stacks-and-physics.html">Two Stacks, One Roofline</a>,
 <a href="./teaching-accelerators.html">Where Does the Data Live?</a>
</div></footer>
""", encoding="utf-8")
    print(f"wrote {OUT} with {len(rows)} candidates")
    for r in rows:
        be = f"{r['breakeven']:,.0f}" if r["breakeven"] else "n/a"
        print(f"  {r['name']:20s} {r['params']/1e9:5.0f}B  total {r['total']:6.0f} GB  "
              f"min {r['smallest'] or '-'} chips  break-even {be} tok/s/chip")


if __name__ == "__main__":
    main()
