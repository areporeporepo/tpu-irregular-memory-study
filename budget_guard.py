#!/usr/bin/env python3
"""Keep the campaign inside its credit, without anyone having to remember to check.

The rule this enforces: never let the estimated spend cross (total - reserve). The reserve exists
because the last week of a study is when reruns are needed most, and running out of credit at that
point is what turns a dataset into a waste of three weeks.

Called at the start of every supervisor cycle. It bills the wall time of every live TPU since the
previous cycle, appends a ledger line, and prints one of:

    OK       spend is fine, claim what you like
    THROTTLE past the soft line, do not claim anything new, keep what is running
    STOP     past the hard line, delete everything and stop claiming

    python3 budget_guard.py            # bill this cycle and print a verdict
    python3 budget_guard.py --report   # human summary, bills nothing
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "budget_ledger.json"

TOTAL = 20_000.0     # credit at the start of this campaign
RESERVE = 3_000.0    # never spend into this
SOFT_FRACTION = 0.80  # stop *claiming* at 80% of the spendable pool

# Spot price per chip-hour, us-east5 and us-east1, August 2026. v5e is the least certain of the
# three and is deliberately rounded up, because a rate that is too low here spends real credit.
RATE = {"v6e": 1.4033, "v5p": 1.26, "v5litepod": 0.60, "v4": 0.97}


def chips_for(accel: str) -> int:
    """Chips in an accelerator type, which is not always the number in its name.

    v5e and v6e count chips in the suffix. v5p and v4 count TensorCores and put two on a chip, so
    `v5p-8` is four chips, which JAX confirms by reporting 4 devices on it. Before this was fixed
    the guard billed our v5p-8 as eight v6e chips, $11.23/hr against a real $5.04, which is about
    $2,400 of imaginary spend between now and September and enough to trigger a false STOP.
    """
    tail = accel.rsplit("-", 1)[-1]
    if not tail.isdigit():
        return 1
    n = int(tail)
    return n // 2 if accel.startswith(("v5p-", "v4-")) else n
ZONES = ["us-east1-d", "asia-northeast1-b", "us-east5-a", "us-east5-b", "us-east5-c"]


def live_chips() -> list[dict]:
    """Every TPU this project currently has, with its chip count."""
    out = []
    for zone in ZONES:
        try:
            raw = subprocess.run(
                ["gcloud", "compute", "tpus", "tpu-vm", "list", f"--zone={zone}",
                 "--format=value(name,acceleratorType,state)"],
                capture_output=True, text=True, timeout=120).stdout
        except subprocess.TimeoutExpired:
            continue
        for line in raw.strip().splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            name, accel, state = parts[0], parts[1], parts[2]
            family, _, count = accel.partition("-")
            if state not in ("READY", "CREATING"):
                continue
            out.append({"name": name, "zone": zone, "accel": accel,
                        "chips": chips_for(accel),
                        "rate": RATE.get(family, 1.4033), "state": state})
    return out


def load() -> dict:
    if LEDGER.is_file():
        return json.loads(LEDGER.read_text())
    return {"total": TOTAL, "reserve": RESERVE, "spent": 0.0, "cycles": [], "last": None}


def main() -> None:
    report_only = "--report" in sys.argv
    now = datetime.now(timezone.utc)
    state = load()
    fleet = live_chips()
    chips = sum(f["chips"] for f in fleet)
    burn_per_hour = sum(f["chips"] * f["rate"] for f in fleet)

    if not report_only:
        last = state.get("last")
        if last:
            hours = (now - datetime.fromisoformat(last)).total_seconds() / 3600
            # Bill the current fleet for the gap since the last cycle. This over-bills a slice
            # that was preempted mid-gap, which is the safe direction to be wrong in.
            charge = min(hours, 2.0) * burn_per_hour
            state["spent"] = round(state["spent"] + charge, 2)
            state["cycles"].append({"at": now.isoformat(), "hours": round(hours, 4),
                                    "chips": chips, "charge": round(charge, 2)})
            state["cycles"] = state["cycles"][-500:]
        state["last"] = now.isoformat()
        LEDGER.write_text(json.dumps(state, indent=2))

    spendable = state["total"] - state["reserve"]
    spent = state["spent"]
    soft = spendable * SOFT_FRACTION
    verdict = "OK" if spent < soft else ("THROTTLE" if spent < spendable else "STOP")
    hours_left = (spendable - spent) / burn_per_hour if burn_per_hour else float("inf")

    print(f"spent≈${spent:,.0f} of ${spendable:,.0f} spendable "
          f"(${state['reserve']:,.0f} reserved) | fleet {chips} chips "
          f"= ${burn_per_hour:,.2f}/hr | {hours_left:,.1f} hr of runway | {verdict}")
    for f in fleet:
        print(f"  {f['name']:16s} {f['accel']:9s} {f['zone']:18s} {f['state']}")
    if report_only:
        return
    print(verdict)  # last line is what the shell reads


if __name__ == "__main__":
    main()
