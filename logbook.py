#!/usr/bin/env python3
"""The lab notebook, kept as data and rendered as a page.

Markdown is fine for prose but wrong for this, because most of what accumulates here is not
prose: it is fleet state, spend, and numbers that change every twenty minutes. So entries are
appended as JSON lines and the page is generated from them together with the live campaign state.
Nothing is hand-edited, which means nothing goes stale.

    python3 logbook.py add finding "all_to_all is free to 16 chips, then costs 77%"
    python3 logbook.py add decision "burst a 256-chip slice for the third point" --detail "..."
    python3 logbook.py build            # regenerate index.html
    python3 logbook.py build --open     # and open it

Entry kinds, chosen so the feed can be skimmed for what matters:
    finding     a measurement that says something
    correction  a previous number was wrong, and why
    decision    a choice made, with the reason
    plan        the plan changed because of data
    event       infrastructure happened (preemption, capacity, spend)
"""
from __future__ import annotations

import html
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENTRIES = HERE / "logbook.jsonl"
DATA = HERE / "data"
LEDGER = HERE / "budget_ledger.json"
OUT = HERE / "index.html"

KINDS = {
    "finding": ("Finding", "◆"),
    "correction": ("Correction", "✕"),
    "decision": ("Decision", "▲"),
    "plan": ("Plan change", "❖"),
    "event": ("Event", "●"),
}


def add(kind: str, text: str, detail: str = "") -> None:
    if kind not in KINDS:
        sys.exit(f"kind must be one of {', '.join(KINDS)}")
    entry = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "kind": kind, "text": text, "detail": detail}
    with ENTRIES.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print(f"logged {kind}: {text[:70]}")


def read_entries() -> list[dict]:
    if not ENTRIES.is_file():
        return []
    out = []
    for line in ENTRIES.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def fleet_state() -> tuple[list[dict], float]:
    """Ask the budget guard rather than duplicating its logic."""
    try:
        raw = subprocess.run([sys.executable, str(HERE / "budget_guard.py"), "--report"],
                             capture_output=True, text=True, timeout=300).stdout
    except Exception:
        return [], 0.0
    rows, burn = [], 0.0
    for line in raw.splitlines():
        if "/hr" in line and "fleet" in line:
            for part in line.split("|"):
                if "/hr" in part:
                    try:
                        burn = float(part.split("$")[-1].split("/hr")[0].replace(",", ""))
                    except ValueError:
                        pass
        elif line.startswith("  "):
            bits = line.split()
            if len(bits) >= 4:
                rows.append({"name": bits[0], "accel": bits[1], "zone": bits[2], "state": bits[3]})
    return rows, burn


def chips_for(accel: str) -> int:
    """Chips in an accelerator type, which is not always the number in its name.

    v5e and v6e count chips in the suffix. v5p and v4 count TensorCores and put two on a chip, so
    `v5p-32` is sixteen chips, which JAX confirms. The same helper lives in budget_guard.py,
    observe_contention.py and build_cluster_dashboard.py; all four read this wrong at first and
    doubled every v5p slice.
    """
    tail = accel.rsplit("-", 1)[-1]
    if not tail.isdigit():
        return 1
    n = int(tail)
    return n // 2 if accel.startswith(("v5p-", "v4-")) else n


def publish_health() -> dict:
    """Is what you are reading actually current?

    A page that regenerates every twenty minutes but cannot push is worse than a page with a date on
    it, because it looks live and is not. On 2026-08-16 the stored GitHub credential disappeared and
    every cycle logged "push failed" while the published site stayed frozen for hours. This puts that
    state on the page itself rather than leaving it in a log nobody reads.
    """
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(HERE), *args], capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except Exception:
            return ""

    unpushed = git("rev-list", "--count", "origin/main..HEAD")
    return {"unpushed": int(unpushed) if unpushed.isdigit() else 0,
            "last_published": git("log", "-1", "--format=%cI", "origin/main"),
            "head": git("log", "-1", "--format=%h", "HEAD")}


def latest_fabric() -> tuple[dict, str]:
    """The most recent complete fabric sweep, keyed by (op, chips, payload)."""
    files = sorted(DATA.glob("fabric*.json"))
    for path in reversed(files):
        try:
            blob = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        table = {}
        for rec in blob.get("records", []):
            if "ms" not in rec:
                continue
            table[(rec["op"], rec["chips"], rec["per_chip_mib"])] = rec
        if table:
            return table, path.name
    return {}, ""


def render() -> str:
    entries = sorted(read_entries(), key=lambda e: e["at"], reverse=True)
    fleet, burn = fleet_state()
    ledger = json.loads(LEDGER.read_text()) if LEDGER.is_file() else {}
    spent = ledger.get("spent", 0.0)
    spendable = ledger.get("total", 20000) - ledger.get("reserve", 3000)
    table, source = latest_fabric()
    health = publish_health()
    chips = sum(chips_for(f["accel"]) for f in fleet)
    runway = (spendable - spent) / burn if burn else 0.0
    cycles = len(list(DATA.glob("fabric*.json")))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ---------------------------------------------------------------- measurement table
    ops = sorted({k[0] for k in table})
    chip_counts = sorted({k[1] for k in table})
    payloads = sorted({k[2] for k in table})
    rows = []
    for op in ops:
        for mib in payloads:
            cells = []
            for c in chip_counts:
                rec = table.get((op, c, mib))
                cells.append(f"<td>{rec['ms']:.4f}</td>" if rec else "<td class='na'>&mdash;</td>")
            rows.append(f"<tr><td class='op'>{html.escape(op)}</td>"
                        f"<td class='num'>{mib:g} MiB</td>{''.join(cells)}</tr>")
    head = "".join(f"<th>{c} chips</th>" for c in chip_counts)

    feed = []
    for e in entries:
        label, glyph = KINDS.get(e["kind"], ("Note", "•"))
        stamp = e["at"].replace("T", " ").replace("+00:00", "")
        detail = (f"<div class='detail'>{html.escape(e['detail'])}</div>"
                  if e.get("detail") else "")
        feed.append(
            f"<article class='entry {e['kind']}'>"
            f"<div class='meta'><span class='glyph'>{glyph}</span>"
            f"<span class='kind'>{label}</span><time>{stamp}</time></div>"
            f"<p>{html.escape(e['text'])}</p>{detail}</article>")

    fleet_rows = "".join(
        f"<tr><td class='op'>{html.escape(f['name'])}</td><td>{html.escape(f['accel'])}</td>"
        f"<td>{html.escape(f['zone'])}</td><td>{html.escape(f['state'])}</td></tr>"
        for f in fleet) or "<tr><td colspan='4' class='na'>no live TPUs</td></tr>"

    stale = ""
    if health["unpushed"]:
        when = health["last_published"][:16].replace("T", " ") or "unknown"
        stale = (f'<p class="stale">This page is behind. {health["unpushed"]} '
                 f'{"commit" if health["unpushed"] == 1 else "commits"} of measurements exist '
                 f'locally that have not reached GitHub, so what you are reading was published at '
                 f'{when} and the numbers above are newer than the ones on the site. The campaign '
                 f'is still collecting; only publishing is stuck.</p>')

    return f"""<title>TPU Irregular Memory Study &middot; Logbook</title>
<style>
  :root {{
    --bg:#f6f5f3; --surface:#fff; --surface-2:#efedea; --ink:#16161a; --ink-2:#4a4a52;
    --ink-3:#75757f; --line:#d8d5d0; --accent:#b4531f; --accent-soft:#e8d9cf;
    --mono:ui-monospace,"SF Mono",Menlo,monospace;
    --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
    --bg:#0e0e11; --surface:#17171b; --surface-2:#1e1e24; --ink:#f2f1ee; --ink-2:#b6b5b0;
    --ink-3:#85848c; --line:#2c2c34; --accent:#e08a4c; --accent-soft:#3a2a1d;
  }} }}
  :root[data-theme="dark"] {{
    --bg:#0e0e11; --surface:#17171b; --surface-2:#1e1e24; --ink:#f2f1ee; --ink-2:#b6b5b0;
    --ink-3:#85848c; --line:#2c2c34; --accent:#e08a4c; --accent-soft:#3a2a1d;
  }}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 var(--sans)}}
  .wrap{{max-width:1000px;margin:0 auto;padding:0 22px}}
  header{{padding:44px 0 26px;border-bottom:1px solid var(--line)}}
  h1{{font-size:clamp(26px,4vw,40px);letter-spacing:-.025em;margin:0 0 6px;font-weight:670}}
  .sub{{color:var(--ink-3);font:500 13px var(--mono);letter-spacing:.03em}}
  h2{{font-size:14px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3);
      margin:38px 0 14px;font-weight:620}}
  .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:24px}}
  .kpi{{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px 16px}}
  .kpi .v{{font:660 25px/1.1 var(--mono);letter-spacing:-.02em;display:block}}
  .kpi .k{{font:600 10.5px var(--mono);letter-spacing:.08em;text-transform:uppercase;
           color:var(--ink-3);margin-top:6px;display:block}}
  table{{width:100%;border-collapse:collapse;font:14px var(--mono);background:var(--surface);
         border:1px solid var(--line);border-radius:10px;overflow:hidden}}
  th,td{{padding:8px 11px;text-align:right;border-bottom:1px solid var(--line)}}
  th{{font:600 10.5px var(--mono);letter-spacing:.07em;text-transform:uppercase;
      color:var(--ink-3);background:var(--surface-2)}}
  td.op,th:first-child,td.num{{text-align:left}}
  td.op{{font-weight:600}}
  .na{{color:var(--ink-3)}}
  .scroll{{overflow-x:auto}}
  .entry{{background:var(--surface);border:1px solid var(--line);border-left:3px solid var(--ink-3);
          border-radius:0 11px 11px 0;padding:13px 16px;margin-bottom:11px}}
  .entry.finding{{border-left-color:var(--ink)}}
  .entry.correction{{border-left-color:var(--accent)}}
  .entry.decision{{border-left-style:double;border-left-width:5px}}
  .entry.plan{{border-left-style:dashed}}
  .entry .meta{{display:flex;gap:9px;align-items:baseline;margin-bottom:5px}}
  .glyph{{font-size:12px}}
  .kind{{font:600 10.5px var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3)}}
  time{{font:500 11.5px var(--mono);color:var(--ink-3);margin-left:auto}}
  .entry p{{margin:0;color:var(--ink)}}
  .detail{{margin-top:7px;font:13px/1.55 var(--mono);color:var(--ink-2);
           background:var(--surface-2);padding:9px 11px;border-radius:7px;white-space:pre-wrap}}
  footer{{padding:36px 0 60px;color:var(--ink-3);font-size:13px}}
  .nav{{line-height:2.1;max-width:82ch}}\n  .stale{{margin:18px 0 0;padding:11px 14px;border:1px solid var(--accent);\n    border-radius:10px;background:var(--accent-soft);color:var(--ink);\n    font:13.5px/1.55 var(--sans);max-width:82ch}}
  .nav a{{font-weight:620}}
  a{{color:var(--ink)}}
</style>
<header><div class="wrap">
  <h1>Irregular Memory Access on TPU v6e</h1>
  <div class="sub">local gather versus fabric collectives &middot; generated {now}</div>
  <div class="kpis">
    <div class="kpi"><span class="v">{chips}</span><span class="k">chips live</span></div>
    <div class="kpi"><span class="v">${burn:,.0f}</span><span class="k">per hour</span></div>
    <div class="kpi"><span class="v">${spent:,.0f}</span><span class="k">of ${spendable:,.0f} spent</span></div>
    <div class="kpi"><span class="v">{runway:,.0f} h</span><span class="k">runway left</span></div>
    <div class="kpi"><span class="v">{cycles}</span><span class="k">sweeps collected</span></div>
    <div class="kpi"><span class="v">{len(entries)}</span><span class="k">log entries</span></div>
  </div>
  {stale}
</div></header>
<div class="wrap">
  <h2>Pages</h2>
  <p class="nav">
   <a href="./gather-cliff.html">The gather cliff</a> &middot; the study's main result: a 3x step in
   gather bandwidth caused by one compile-time placement decision, measured on v5p, v6e and an A100.
   <br><a href="./roadmap.html">Chips we can get, chips being bought</a> &middot; v5e through Rubin
   on the same axes, with every figure sourced and the estimates marked.
   <br><a href="./models.html">What we can actually run</a> &middot; live model catalogue against
   the capacity we hold.
   <br><a href="./cluster.html">Who is using the cluster</a> &middot; refreshed every cycle.
   <br><a href="./stacks-and-physics.html">The physics underneath</a> &middot;
   <a href="./background-2026-2027.html">where the hardware is going</a> &middot;
   <a href="./teaching-accelerators.html">where does the data live</a>
  </p>

  <h2>Latest sweep &middot; milliseconds per collective, dispatch amortised</h2>
  <div class="scroll"><table>
    <thead><tr><th>op</th><th>per-chip payload</th>{head}</tr></thead>
    <tbody>{''.join(rows) or "<tr><td colspan='6' class='na'>no sweep collected yet</td></tr>"}</tbody>
  </table></div>
  <p class="sub" style="margin-top:9px">source: {html.escape(source) or "none"}</p>

  <h2>Fleet</h2>
  <div class="scroll"><table>
    <thead><tr><th>name</th><th>type</th><th>zone</th><th>state</th></tr></thead>
    <tbody>{fleet_rows}</tbody>
  </table></div>

  <h2>Log</h2>
  {''.join(feed) or "<p class='na'>nothing logged yet</p>"}
</div>
<footer><div class="wrap">
  Regenerated by <span style="font-family:var(--mono)">logbook.py build</span> at the end of every
  supervisor cycle. Entries are appended as JSON lines in
  <span style="font-family:var(--mono)">logbook.jsonl</span>, so this page is never hand-edited.
</div></footer>
"""


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    if cmd == "add":
        if len(sys.argv) < 4:
            sys.exit("usage: logbook.py add KIND TEXT [--detail TEXT]")
        detail = ""
        if "--detail" in sys.argv:
            i = sys.argv.index("--detail")
            detail = sys.argv[i + 1] if len(sys.argv) > i + 1 else ""
            args = sys.argv[2:i]
        else:
            args = sys.argv[2:]
        add(args[0], " ".join(args[1:]), detail)
    elif cmd == "build":
        OUT.write_text(render(), encoding="utf-8")
        print(f"wrote {OUT}")
        if "--open" in sys.argv:
            subprocess.run(["open", str(OUT)])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
