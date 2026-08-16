#!/usr/bin/env python3
"""Who is using the class hardware right now, refreshed every cycle.

`soe-hpccenter` is shared: four GKE Autopilot clusters and a TPU quota that every member draws on.
Nothing tells a student what is currently in use, so they guess, and the guess is usually "try
us-east5 and see what happens". This surveys everything visible and writes a page.

Two populations are reported separately because they behave differently:

    GKE workloads   pods in non-system namespaces, which is what a student assignment looks like
    TPU VMs         direct `tpu-vm create` allocations, which is what everyone actually uses

Peer names are hashed. The point is to show a student whether the hardware is busy, not who to
blame for it.

    python3 build_cluster_dashboard.py        # writes cluster.html
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "cluster.html"
SERIES = HERE / "data" / "cluster_state.jsonl"

CLUSTERS = [("class-tpu-cluster-east5", "us-east5"), ("class-tpu-cluster-south1", "us-south1"),
            ("class-tpu-cluster-west4", "us-west4"), ("class-tpu-cluster-central1", "us-central1")]
TPU_ZONES = ["us-east1-d", "us-east5-a", "us-east5-b", "us-east5-c", "us-south1-a",
             "us-central1-a", "europe-west4-a", "asia-northeast1-b", "us-west4-a", "us-west4-b"]
OURS_PREFIX = ("anh-", "probe-", "sweep-")
SYSTEM_NS = ("kube-system", "kube-node-lease", "kube-public", "gke-gmp-system", "gke-managed-cim",
             "gmp-system", "jobset-system", "gke-managed-system", "composer-system")


def sh(args: list[str], timeout: int = 120) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        return ""


def anon(name: str) -> str:
    return "peer-" + hashlib.sha256(name.encode()).hexdigest()[:6]


def survey_clusters() -> list[dict]:
    out = []
    for name, loc in CLUSTERS:
        sh(["gcloud", "container", "clusters", "get-credentials", name, f"--region={loc}"])
        nodes = [l for l in sh(["kubectl", "get", "nodes", "--no-headers"]).splitlines() if l.strip()]
        tpu_nodes = [l for l in sh(["kubectl", "get", "nodes", "-l",
                                    "cloud.google.com/gke-tpu-accelerator",
                                    "--no-headers"]).splitlines() if l.strip()]
        raw = sh(["kubectl", "get", "pods", "--all-namespaces", "--no-headers"])
        user_pods = []
        for line in raw.splitlines():
            parts = line.split()
            if len(parts) < 4 or parts[0] in SYSTEM_NS:
                continue
            user_pods.append({"ns": parts[0], "pod": parts[1], "status": parts[3]})
        out.append({"cluster": name, "region": loc, "nodes": len(nodes),
                    "tpu_nodes": len(tpu_nodes), "user_pods": user_pods,
                    "system_pods": len([l for l in raw.splitlines() if l.strip()]) - len(user_pods)})
    return out


def survey_tpus() -> list[dict]:
    rows = []
    for zone in TPU_ZONES:
        raw = sh(["gcloud", "compute", "tpus", "tpu-vm", "list", f"--zone={zone}",
                  "--format=value(name,acceleratorType,state)"])
        for line in raw.strip().splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            name, accel, state = parts[0], parts[1], parts[2]
            tail = accel.rsplit("-", 1)[-1]
            mine = name.startswith(OURS_PREFIX)
            rows.append({"zone": zone, "accel": accel, "state": state, "mine": mine,
                         "who": "us" if mine else anon(name),
                         "chips": int(tail) if tail.isdigit() else 1})
    return rows


def main() -> None:
    now = datetime.now(timezone.utc)
    clusters, tpus = survey_clusters(), survey_tpus()
    ours = sum(t["chips"] for t in tpus if t["mine"])
    theirs = sum(t["chips"] for t in tpus if not t["mine"])
    peers = len({t["who"] for t in tpus if not t["mine"]})
    user_pods = sum(len(c["user_pods"]) for c in clusters)

    SERIES.parent.mkdir(exist_ok=True)
    with SERIES.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"at": now.isoformat(timespec="seconds"), "our_chips": ours,
                             "peer_chips": theirs, "peers": peers,
                             "user_pods": user_pods}) + "\n")

    ctr = "".join(
        f"<tr><td>{c['cluster'].replace('class-tpu-cluster-','')}</td><td>{c['region']}</td>"
        f"<td class='n'>{c['nodes']}</td><td class='n'>{c['tpu_nodes'] or '&mdash;'}</td>"
        f"<td class='n'>{len(c['user_pods']) or '&mdash;'}</td>"
        f"<td class='n'>{c['system_pods']}</td></tr>" for c in clusters)
    tpur = "".join(
        f"<tr><td>{t['who']}</td><td>{t['accel']}</td><td>{t['zone']}</td>"
        f"<td>{t['state']}</td><td class='n'>{t['chips']}</td></tr>"
        for t in sorted(tpus, key=lambda t: (not t["mine"], t["zone"]))) or \
        "<tr><td colspan='5'>no TPU VMs anywhere in the project</td></tr>"
    pods = "".join(
        f"<tr><td>{p['ns']}</td><td>{p['pod'][:44]}</td><td>{p['status']}</td></tr>"
        for c in clusters for p in c["user_pods"]) or \
        "<tr><td colspan='3'>no workloads in any non-system namespace</td></tr>"

    OUT.write_text(f"""<title>Class Hardware Right Now</title>
<style>
 :root{{--bg:#fcfcfb;--surface:#fff;--ink:#1c1c20;--ink-2:#4a4a52;--ink-3:#75757f;--line:#d8d5d0;
   --accent:#b4531f;--mono:ui-monospace,"SF Mono",Menlo,monospace;
   --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
 @media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#111113;--surface:#1a1a19;
   --ink:#f0efec;--ink-2:#b6b5b0;--ink-3:#85848c;--line:#2e2e33;--accent:#e08a4c}}}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 var(--sans)}}
 .wrap{{max-width:940px;margin:0 auto;padding:0 22px}}
 header{{padding:50px 0 24px;border-bottom:1px solid var(--line)}}
 h1{{font-size:clamp(28px,4.6vw,42px);letter-spacing:-.03em;margin:0 0 10px;font-weight:680}}
 h2{{font-size:15px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-3);
   margin:38px 0 10px;font-weight:620}}
 .sub{{color:var(--ink-3);font:500 13px var(--mono)}}
 p{{margin:0 0 13px;max-width:74ch;color:var(--ink-2)}}
 .tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:22px}}
 .tile{{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px 16px}}
 .tile .v{{font:660 27px/1.1 var(--mono);letter-spacing:-.02em;display:block}}
 .tile .k{{font:600 10px var(--mono);letter-spacing:.08em;text-transform:uppercase;
   color:var(--ink-3);margin-top:6px;display:block}}
 table{{width:100%;border-collapse:collapse;font:13px var(--mono);background:var(--surface);
   border:1px solid var(--line);border-radius:10px;overflow:hidden;margin-top:8px}}
 th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left}}
 th{{font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3)}}
 .n{{text-align:right}} .scroll{{overflow-x:auto}}
 footer{{padding:32px 0 60px;color:var(--ink-3);font-size:13px}} a{{color:var(--ink)}}
</style>
<header><div class="wrap">
 <h1>Class Hardware Right Now</h1>
 <p class="sub">{now.strftime('%Y-%m-%d %H:%M UTC')} &middot; project soe-hpccenter &middot;
 regenerated every campaign cycle</p>
 <div class="tiles">
  <div class="tile"><span class="v">{ours}</span><span class="k">chips, ours</span></div>
  <div class="tile"><span class="v">{theirs}</span><span class="k">chips, peers</span></div>
  <div class="tile"><span class="v">{peers}</span><span class="k">other holders</span></div>
  <div class="tile"><span class="v">{user_pods}</span><span class="k">GKE workloads</span></div>
  <div class="tile"><span class="v">{sum(c['nodes'] for c in clusters)}</span><span class="k">cluster nodes</span></div>
 </div>
</div></header>
<div class="wrap">
 <p>Two ways to get hardware here, and they behave differently. Direct <span class="mono">tpu-vm
 create</span> is what everyone uses and what shows up in the second table. The four GKE Autopilot
 clusters are the multi-tenant path, and a student workload there would appear in the third table.
 Both are visible to every project member, which is what makes this page possible.</p>

 <h2>GKE clusters</h2>
 <div class="scroll"><table>
  <thead><tr><th>cluster</th><th>region</th><th class="n">nodes</th><th class="n">TPU nodes</th>
   <th class="n">user pods</th><th class="n">system pods</th></tr></thead>
  <tbody>{ctr}</tbody></table></div>
 <p class="sub" style="margin-top:8px">System pods are GKE's own: metrics collectors, CNI, logging.
 A cluster with many nodes and no user pods is idle, not busy.</p>

 <h2>TPU VMs across every zone</h2>
 <div class="scroll"><table>
  <thead><tr><th>holder</th><th>type</th><th>zone</th><th>state</th><th class="n">chips</th></tr></thead>
  <tbody>{tpur}</tbody></table></div>

 <h2>GKE workloads in non-system namespaces</h2>
 <div class="scroll"><table>
  <thead><tr><th>namespace</th><th>pod</th><th>status</th></tr></thead>
  <tbody>{pods}</tbody></table></div>
</div>
<footer><div class="wrap">
 Written by <span class="mono">build_cluster_dashboard.py</span> from the same cycle that collects
 measurements, so this page is never older than twenty minutes. Peer names are hashed; the history
 accumulates in <span class="mono">data/cluster_state.jsonl</span>. Companion pages:
 <a href="./">the measurement log</a>, <a href="./models.html">what we can run</a>,
 <a href="./teaching-accelerators.html">where does the data live</a>.
</div></footer>
""", encoding="utf-8")
    print(f"wrote {OUT}: ours={ours} chips, peers={theirs} chips across {peers} holders, "
          f"{user_pods} GKE workloads, {sum(c['nodes'] for c in clusters)} nodes")


if __name__ == "__main__":
    main()
