#!/usr/bin/env python3
"""Who else is holding the chips, and does it explain what we can get?

`soe-hpccenter` is a shared class project. Every TPU in it is visible to every member, which makes
it a natural experiment nobody has published: roughly fifty students competing for one quota grant,
observed continuously for three weeks. That series is the missing half of the capacity story. Our
own sweep found multi-chip v6e in only two zones out of nine, and the obvious question is whether
that is Google's capacity or our classmates'.

Names are hashed. The signal is aggregate demand, not who is running what, and publishing other
students' VM names is not ours to do.

    python3 observe_contention.py           # one observation, appended to data/contention.jsonl
    python3 observe_contention.py --report  # summarise the series so far
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERIES = HERE / "data" / "contention.jsonl"

# Every zone where this project has ever been seen to offer TPUs, v5e and v6e families both.
ZONES = ["us-east1-d", "us-east5-a", "us-east5-b", "us-east5-c", "us-south1-a",
         "us-central1-a", "us-central2-b", "europe-west4-a", "asia-northeast1-b",
         "us-west4-a", "us-west4-b"]
OURS = {"anh-v6e-32", "anh-dev1", "anh-big-256", "anh-big-128", "anh-big-64"}


def anon(name: str) -> str:
    """Stable pseudonym. Enough to count distinct holders across time, not enough to identify."""
    return "peer-" + hashlib.sha256(name.encode()).hexdigest()[:8]


def observe() -> dict:
    at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for zone in ZONES:
        try:
            raw = subprocess.run(
                ["gcloud", "compute", "tpus", "tpu-vm", "list", f"--zone={zone}",
                 "--format=value(name,acceleratorType,state)"],
                capture_output=True, text=True, timeout=90).stdout
        except subprocess.TimeoutExpired:
            continue
        for line in raw.strip().splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            name, accel, state = parts[0], parts[1], parts[2]
            tail = accel.rsplit("-", 1)[-1]
            rows.append({"zone": zone, "accel": accel, "state": state,
                         "chips": int(tail) if tail.isdigit() else 1,
                         "mine": name in OURS, "who": "us" if name in OURS else anon(name)})
    theirs = [r for r in rows if not r["mine"]]
    return {"at": at, "total_vms": len(rows), "total_chips": sum(r["chips"] for r in rows),
            "peer_vms": len(theirs), "peer_chips": sum(r["chips"] for r in theirs),
            "distinct_peers": len({r["who"] for r in theirs}),
            "by_zone": dict(Counter(r["zone"] for r in rows)),
            "by_type": dict(Counter(r["accel"] for r in rows)),
            "rows": rows}


def report() -> None:
    if not SERIES.is_file():
        print("no observations yet")
        return
    obs = [json.loads(l) for l in SERIES.read_text().splitlines() if l.strip()]
    print(f"{len(obs)} observations, {obs[0]['at']} to {obs[-1]['at']}")
    peak = max(obs, key=lambda o: o["total_chips"])
    print(f"peak project-wide: {peak['total_chips']} chips in {peak['total_vms']} VMs "
          f"at {peak['at']}")
    print(f"peak peers: {max(o['distinct_peers'] for o in obs)} distinct holders")
    zones = Counter()
    for o in obs:
        for z, n in o["by_zone"].items():
            zones[z] += n
    print("VM-observations by zone: " + ", ".join(f"{z}={n}" for z, n in zones.most_common()))
    types = Counter()
    for o in obs:
        for t, n in o["by_type"].items():
            types[t] += n
    print("by accelerator type:    " + ", ".join(f"{t}={n}" for t, n in types.most_common(8)))


def main() -> None:
    if "--report" in sys.argv:
        report()
        return
    rec = observe()
    SERIES.parent.mkdir(exist_ok=True)
    with SERIES.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"{rec['at']}: {rec['total_chips']} chips project-wide, "
          f"{rec['peer_chips']} held by {rec['distinct_peers']} peers, "
          f"zones={list(rec['by_zone'])}")


if __name__ == "__main__":
    main()
