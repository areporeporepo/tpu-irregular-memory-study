#!/bin/bash
# One cycle of the measurement campaign. Designed to be run on a schedule, forever, and to be
# safe to run when everything is already fine or everything is already broken.
#
# The campaign is deliberately built as many short cycles rather than one long-lived job on the
# TPU, because every slice here is spot and will be preempted. A cycle that takes minutes and
# writes its results back to this Mac loses at most one cycle when that happens. It also means a
# three-week campaign produces repeated measurements of the same configuration on *different*
# physical slices, which turns preemption from a nuisance into a variance study.
#
#   ./supervisor.sh            # one cycle
#   ./supervisor.sh --status   # what is alive, what has been collected
set -uo pipefail

STUDY="$HOME/tpu-irregular-memory-study"
DATA="$STUDY/data"
LOG="$STUDY/campaign.log"
SLICE="anh-steady-16"     # steady fleet: 16 chips at $22.45/hr fits a Sept 1 runway
DEV="anh-dev1"
# Only these two zones had multi-chip v6e capacity in the 2026-08-16 sweep. Ordered by
# preference: us-east1-d is closer to everything else we own.
ZONES="us-east1-d asia-northeast1-b"
ACCEL="v6e-16"
RUNTIME="v2-alpha-tpuv6e"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

mkdir -p "$DATA"
say() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }

# Two JAX programs on one TPU slice fight over the accelerator and both fail, so a cycle that
# runs long must not have the next one started on top of it. A stale lock older than two hours
# is assumed dead, because the longest legitimate cycle is a jax reinstall plus two experiments.
LOCK="$STUDY/.cycle.lock"
if [ -d "$LOCK" ]; then
  if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +120 2>/dev/null)" ]; then
    say "clearing stale lock"; rm -rf "$LOCK"
  elif [ "${1:-}" != "--status" ]; then
    say "another cycle is still running, skipping"; exit 0
  fi
fi
if [ "${1:-}" != "--status" ]; then
  mkdir "$LOCK" 2>/dev/null || { say "could not take lock"; exit 0; }
  trap 'rm -rf "$LOCK"' EXIT
fi

find_slice() {
  for z in $ZONES; do
    state=$(gcloud compute tpus tpu-vm describe "$SLICE" --zone="$z" \
              --format="value(state)" 2>/dev/null | head -1)
    if [ -n "$state" ]; then echo "$z $state"; return 0; fi
  done
  echo "- MISSING"
}

if [ "${1:-}" = "--status" ]; then
  read -r zone state <<< "$(find_slice)"
  say "status: slice=$SLICE zone=$zone state=$state"
  say "status: $(ls -1 "$DATA" 2>/dev/null | wc -l | tr -d ' ') result files, $(du -sh "$DATA" 2>/dev/null | cut -f1) on disk"
  tail -5 "$LOG" 2>/dev/null
  exit 0
fi

# ---------------------------------------------------------------- money first
# Bill the fleet for the time since the last cycle before deciding anything. A campaign that
# spends its reserve is worse than one that collects less data, because reruns happen at the end.
VERDICT=$(python3 "$STUDY/budget_guard.py" 2>/dev/null | tail -1)
case "$VERDICT" in
  STOP)
    say "budget guard says STOP: deleting the fleet and standing down"
    for z in $ZONES; do
      for n in $(gcloud compute tpus tpu-vm list --zone="$z" --format="value(name)" 2>/dev/null); do
        gcloud compute tpus tpu-vm delete "$n" --zone="$z" --quiet >/dev/null 2>&1
        say "  deleted $n in $z"
      done
    done
    exit 0 ;;
  THROTTLE)
    say "budget guard says THROTTLE: measuring what is already running, claiming nothing new" ;;
esac

read -r zone state <<< "$(find_slice)"

# ---------------------------------------------------------------- keep a slice alive
if [ "$state" = "MISSING" ] || [ "$state" = "PREEMPTED" ] || [ "$state" = "TERMINATED" ]; then
  say "slice $state, hunting capacity"
  if [ "$state" != "MISSING" ]; then
    gcloud compute tpus tpu-vm delete "$SLICE" --zone="$zone" --quiet >/dev/null 2>&1
  fi
  got=""
  for z in $ZONES; do
    for size in 16 8 4 1; do   # take the largest the zone will give
      out=$(gcloud compute tpus tpu-vm create "$SLICE" --zone="$z" \
            --accelerator-type="v6e-$size" --version="$RUNTIME" --spot \
            --labels=owner=anh,study=irregular-memory 2>&1)
      if echo "$out" | grep -qi "no more capacity\|Insufficient\|RESOURCE_EXHAUSTED"; then
        say "  $z v6e-$size: no capacity"; continue
      fi
      if echo "$out" | grep -qi "error"; then
        say "  $z v6e-$size: error"; continue
      fi
      say "  claimed v6e-$size in $z"; got="$z"; zone="$z"; break
    done
    [ -n "$got" ] && break
  done
  if [ -z "$got" ]; then say "no capacity anywhere this cycle, will retry"; exit 0; fi
  state=READY
fi

if [ "$state" != "READY" ]; then say "slice is $state, skipping this cycle"; exit 0; fi

# ---------------------------------------------------------------- make sure it can run jax
have_jax=$(gcloud compute tpus tpu-vm ssh "$SLICE" --zone="$zone" --worker=0 \
             --command='python3.11 -c "import jax" 2>/dev/null && echo yes || echo no' \
             2>/dev/null | tail -1)
if [ "$have_jax" != "yes" ]; then
  say "installing jax on all workers"
  gcloud compute tpus tpu-vm ssh "$SLICE" --zone="$zone" --worker=all \
    --command='python3.11 -m pip install -q -U "jax[tpu]"' >/dev/null 2>&1
fi

# ---------------------------------------------------------------- measure
say "cycle $STAMP: running fabric collectives on $SLICE in $zone"
gcloud compute tpus tpu-vm scp "$STUDY/experiment2_fabric_collectives.py" \
  "$SLICE:~/experiment2_fabric_collectives.py" --zone="$zone" --worker=all >/dev/null 2>&1
gcloud compute tpus tpu-vm ssh "$SLICE" --zone="$zone" --worker=all \
  --command='cd ~ && python3.11 experiment2_fabric_collectives.py' \
  > "$DATA/fabric_${STAMP}.stdout" 2>&1

# JAX process 0 is not necessarily gcloud worker 0: on this slice the file landed on worker 3.
# So try every worker and take the first one that has it.
workers=$(gcloud compute tpus tpu-vm describe "$SLICE" --zone="$zone" \
            --format="value(networkEndpoints.len())" 2>/dev/null | head -1)
workers=${workers:-8}
got_file=""
for w in $(seq 0 $((workers - 1))); do
  if gcloud compute tpus tpu-vm scp "$SLICE:~/fabric_results.json" \
       "$DATA/fabric_${STAMP}.json" --zone="$zone" --worker="$w" >/dev/null 2>&1; then
    got_file="w$w"; break
  fi
done
if [ -n "$got_file" ]; then
  n=$(grep -c '"op"' "$DATA/fabric_${STAMP}.json" 2>/dev/null || echo 0)
  say "cycle $STAMP: collected $n fabric records from $got_file"
else
  say "cycle $STAMP: no fabric_results.json came back, stdout kept for diagnosis"
fi

# ---------------------------------------------------------------- single chip, if one exists
for z in $ZONES; do
  dstate=$(gcloud compute tpus tpu-vm describe "$DEV" --zone="$z" --format="value(state)" 2>/dev/null | head -1)
  [ "$dstate" = "READY" ] || continue
  say "cycle $STAMP: running local gather on $DEV in $z"
  gcloud compute tpus tpu-vm ssh "$DEV" --zone="$z" \
    --command='python3.11 -c "import jax" 2>/dev/null || python3.11 -m pip install -q -U "jax[tpu]"' >/dev/null 2>&1
  gcloud compute tpus tpu-vm scp "$STUDY/experiment1_local_gather.py" \
    "$DEV:~/experiment1_local_gather.py" --zone="$z" >/dev/null 2>&1
  gcloud compute tpus tpu-vm ssh "$DEV" --zone="$z" \
    --command="cd ~ && python3.11 experiment1_local_gather.py --out gather_${STAMP}.json" \
    > "$DATA/gather_${STAMP}.stdout" 2>&1
  gcloud compute tpus tpu-vm scp "$DEV:~/gather_${STAMP}.json" \
    "$DATA/gather_${STAMP}.json" --zone="$z" >/dev/null 2>&1 \
    && say "cycle $STAMP: collected local gather results"
  break
done

# ---------------------------------------------------------------- the shared-project view
# soe-hpccenter is a class project, so every cycle also records what everyone else is holding.
# That series is what distinguishes "Google has no capacity" from "our classmates have it all",
# and the first observation already showed we hold 33 of 41 chips project-wide.
python3 "$STUDY/observe_contention.py" >> "$LOG" 2>&1

# Availability probing, once an hour rather than every cycle. Everything we know about capacity
# was measured at 04:00 on a Sunday; this is the instrument that tells us whether that generalises.
if [ "$(date -u +%M)" -lt 20 ]; then
  say "cycle $STAMP: probing availability across zones"
  bash "$STUDY/probe_availability.sh" >> "$LOG" 2>&1
fi

# ---------------------------------------------------------------- publish
# The logbook page is generated, never hand-edited, so it can be rebuilt and pushed every cycle.
python3 "$STUDY/logbook.py" build >/dev/null 2>&1
# Who is holding the class hardware, refreshed on the same cycle as the measurements.
python3 "$STUDY/build_cluster_dashboard.py" >> "$LOG" 2>&1
if [ -n "$(git -C "$STUDY" status --porcelain data index.html logbook.jsonl budget_ledger.json 2>/dev/null)" ]; then
  git -C "$STUDY" add data index.html cluster.html logbook.jsonl budget_ledger.json >/dev/null 2>&1
  git -C "$STUDY" -c user.name="anh nguyen" -c user.email="qanh@stanford.edu" \
    commit -q -m "campaign cycle $STAMP" >/dev/null 2>&1
  if git -C "$STUDY" push -q origin main >/dev/null 2>&1; then
    say "cycle $STAMP: published"
  else
    say "cycle $STAMP: push failed, will retry next cycle"
  fi
fi

say "cycle $STAMP: done"
