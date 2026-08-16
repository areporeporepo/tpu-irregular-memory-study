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

# Every command that touches the network gets a deadline. This is not defensive dressing: on
# 2026-08-16 a `--worker=all` ssh wedged for four hours because one worker's JAX process was waiting
# on a peer that had been preempted. launchd will not start a second copy of a job while the first
# is still alive, so that one hung ssh silently stopped the entire campaign, and every scheduled
# cycle after it exited with "another cycle is still running". A cycle that fails is recoverable.
# A cycle that hangs is not.
#
# macOS ships no timeout(1) and no coreutils here, so this is the bash equivalent: run the command,
# race it against a sleeper, kill whichever loses.
# Two things in here are not obvious and both were found by testing rather than by reasoning:
#
#   >/dev/null 2>&1 on the killer      Without it, the killer inherits the write end of any command
#                                      substitution wrapping this function, so `x=$(t 120 cmd)`
#                                      blocks for the full 120 seconds waiting for EOF on the pipe
#                                      even after cmd has already finished and exited.
#   polling instead of one long sleep   A killer that outlives its target and then fires kill -9 at
#                                      a recycled PID is a loaded gun. Checking once a second means
#                                      it exits as soon as the target is gone.
t() {
  local secs="$1"; shift
  "$@" &
  local pid=$!
  ( waited=0
    while [ "$waited" -lt "$secs" ]; do
      sleep 1
      kill -0 "$pid" 2>/dev/null || exit 0
      waited=$((waited + 1))
    done
    kill -9 "$pid" 2>/dev/null ) >/dev/null 2>&1 &
  local killer=$!
  disown "$killer" 2>/dev/null || true   # else bash announces "Terminated" into the campaign log
  wait "$pid" 2>/dev/null
  local rc=$?
  kill "$killer" 2>/dev/null
  return "$rc"
}

# A gcloud left over from a previous cycle that outlived its deadline holds nothing useful and may
# hold the TPU. Half an hour is well beyond any healthy call and well short of a wedge.
for p in $(pgrep -f "gcloud.py compute tpus" 2>/dev/null); do
  age=$(ps -o etimes= -p "$p" 2>/dev/null | tr -d ' ')
  if [ -n "$age" ] && [ "$age" -gt 1800 ]; then
    kill -9 "$p" 2>/dev/null && say "reaped orphaned gcloud pid $p, ${age}s old"
  fi
done

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
  # Last line of defence: even with a deadline on every call, the whole cycle gets one too. Shorter
  # than two schedule intervals, so a killed cycle is replaced rather than accumulating.
  ( sleep 1500; pkill -P $$ 2>/dev/null; kill -TERM $$ 2>/dev/null ) &
  WATCHDOG=$!
  trap 'rm -rf "$LOCK"; kill "$WATCHDOG" 2>/dev/null' EXIT
fi

find_slice() {
  for z in $ZONES; do
    state=$(t 120 gcloud compute tpus tpu-vm describe "$SLICE" --zone="$z" \
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
      for n in $(t 120 gcloud compute tpus tpu-vm list --zone="$z" --format="value(name)" 2>/dev/null); do
        t 300 gcloud compute tpus tpu-vm delete "$n" --zone="$z" --quiet >/dev/null 2>&1
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
    t 300 gcloud compute tpus tpu-vm delete "$SLICE" --zone="$zone" --quiet >/dev/null 2>&1
  fi
  got=""
  for z in $ZONES; do
    for size in 16 8 4 1; do   # take the largest the zone will give
      out=$(t 420 gcloud compute tpus tpu-vm create "$SLICE" --zone="$z" \
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
have_jax=$(t 240 gcloud compute tpus tpu-vm ssh "$SLICE" --zone="$zone" --worker=0 \
             --command='python3.11 -c "import jax" 2>/dev/null && echo yes || echo no' \
             2>/dev/null | tail -1)
if [ "$have_jax" != "yes" ]; then
  say "installing jax on all workers"
  t 900 gcloud compute tpus tpu-vm ssh "$SLICE" --zone="$zone" --worker=all \
    --command='timeout 600 python3.11 -m pip install -q -U "jax[tpu]"' >/dev/null 2>&1
fi

# ---------------------------------------------------------------- measure
say "cycle $STAMP: running fabric collectives on $SLICE in $zone"
t 240 gcloud compute tpus tpu-vm scp "$STUDY/experiment2_fabric_collectives.py" \
  "$SLICE:~/experiment2_fabric_collectives.py" --zone="$zone" --worker=all >/dev/null 2>&1
t 900 gcloud compute tpus tpu-vm ssh "$SLICE" --zone="$zone" --worker=all \
  --command='cd ~ && timeout 780 python3.11 experiment2_fabric_collectives.py' \
  > "$DATA/fabric_${STAMP}.stdout" 2>&1

# JAX process 0 is not necessarily gcloud worker 0: on this slice the file landed on worker 3.
# So try every worker and take the first one that has it.
workers=$(t 120 gcloud compute tpus tpu-vm describe "$SLICE" --zone="$zone" \
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
  dstate=$(t 120 gcloud compute tpus tpu-vm describe "$DEV" --zone="$z" --format="value(state)" 2>/dev/null | head -1)
  [ "$dstate" = "READY" ] || continue
  say "cycle $STAMP: running local gather on $DEV in $z"
  t 900 gcloud compute tpus tpu-vm ssh "$DEV" --zone="$z" \
    --command='python3.11 -c "import jax" 2>/dev/null || python3.11 -m pip install -q -U "jax[tpu]"' >/dev/null 2>&1
  t 240 gcloud compute tpus tpu-vm scp "$STUDY/experiment1_local_gather.py" \
    "$DEV:~/experiment1_local_gather.py" --zone="$z" >/dev/null 2>&1
  t 900 gcloud compute tpus tpu-vm ssh "$DEV" --zone="$z" \
    --command="cd ~ && timeout 780 python3.11 experiment1_local_gather.py --out gather_${STAMP}.json" \
    > "$DATA/gather_${STAMP}.stdout" 2>&1
  t 240 gcloud compute tpus tpu-vm scp "$DEV:~/gather_${STAMP}.json" \
    "$DATA/gather_${STAMP}.json" --zone="$z" >/dev/null 2>&1 \
    && say "cycle $STAMP: collected local gather results"
  break
done

# ---------------------------------------------------------------- v5p, the SparseCore chip
# v5p is the only chip in the fleet where the documented SparseCore gather kernel compiles, so it is
# the only place the allocation cliff can be measured against a working SparseCore baseline. Two of
# the three artefacts here cost no kernel time at all: the HLO bisection only compiles. Run every
# second hour rather than every cycle, so one chip does not eat the cycle.
V5P="anh-v5p-8"
V5P_ZONE="us-east5-a"
if [ "$(( $(date -u +%H) % 2 ))" -eq 0 ] && [ "$(date -u +%M)" -lt 20 ]; then
  vstate=$(t 120 gcloud compute tpus tpu-vm describe "$V5P" --zone="$V5P_ZONE" \
             --format="value(state)" 2>/dev/null | head -1)
  if [ "$vstate" = "READY" ]; then
    say "cycle $STAMP: allocation ladder and HLO bisection on $V5P"
    for f in experiment10_gather_locality.py hlo_gather_lowering.py; do
      t 240 gcloud compute tpus tpu-vm scp "$STUDY/$f" "$V5P:~/$f" \
        --zone="$V5P_ZONE" --worker=0 >/dev/null 2>&1
    done
    t 900 gcloud compute tpus tpu-vm ssh "$V5P" --zone="$V5P_ZONE" --worker=0 \
      --command="cd ~ && timeout 780 python3.11 experiment10_gather_locality.py --sweep --out alloc_sweep_v5p.json" \
      > "$DATA/alloc_sweep_${STAMP}.stdout" 2>&1
    t 900 gcloud compute tpus tpu-vm ssh "$V5P" --zone="$V5P_ZONE" --worker=0 \
      --command="cd ~ && timeout 780 python3.11 hlo_gather_lowering.py --bisect --out hlo_bisect_v5p.json" \
      > "$DATA/hlo_bisect_${STAMP}.stdout" 2>&1
    for f in alloc_sweep_v5p.json hlo_bisect_v5p.json; do
      t 240 gcloud compute tpus tpu-vm scp "$V5P:~/$f" "$DATA/$f" \
        --zone="$V5P_ZONE" --worker=0 >/dev/null 2>&1
    done
    say "cycle $STAMP: v5p artefacts refreshed"
  else
    say "cycle $STAMP: $V5P is $vstate, skipping the v5p artefacts"
  fi
fi

# The v6e single chip carries the cross-generation half of the same measurement. Two chips minimum
# for every claim: the identical script on a second architecture is what killed the first two
# explanations we had for the cliff.
if [ "$(( $(date -u +%H) % 2 ))" -eq 1 ] && [ "$(date -u +%M)" -lt 20 ]; then
  for z in $ZONES; do
    dstate=$(t 120 gcloud compute tpus tpu-vm describe "$DEV" --zone="$z" \
               --format="value(state)" 2>/dev/null | head -1)
    [ "$dstate" = "READY" ] || continue
    say "cycle $STAMP: allocation ladder on $DEV, the v6e side of the comparison"
    for f in experiment10_gather_locality.py hlo_gather_lowering.py; do
      t 240 gcloud compute tpus tpu-vm scp "$STUDY/$f" "$DEV:~/$f" --zone="$z" >/dev/null 2>&1
    done
    t 900 gcloud compute tpus tpu-vm ssh "$DEV" --zone="$z" \
      --command="cd ~ && timeout 780 python3.11 experiment10_gather_locality.py --sweep --out alloc_sweep_v6e.json" \
      > "$DATA/alloc_sweep_v6e_${STAMP}.stdout" 2>&1
    t 900 gcloud compute tpus tpu-vm ssh "$DEV" --zone="$z" \
      --command="cd ~ && timeout 780 python3.11 hlo_gather_lowering.py --bisect --out hlo_bisect_v6e.json" \
      > "$DATA/hlo_bisect_v6e_${STAMP}.stdout" 2>&1
    for f in alloc_sweep_v6e.json hlo_bisect_v6e.json; do
      t 240 gcloud compute tpus tpu-vm scp "$DEV:~/$f" "$DATA/$f" --zone="$z" >/dev/null 2>&1
    done
    break
  done
fi

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
# The result pages are generated from the JSON in data/, so they cannot drift from the numbers.
# Cheap and local: no TPU, no network. Failures must not take the cycle down with them.
python3 "$STUDY/build_gather_page.py" >> "$LOG" 2>&1 || say "cycle $STAMP: gather page build failed"
python3 "$STUDY/build_roadmap_page.py" >> "$LOG" 2>&1 || say "cycle $STAMP: roadmap page build failed"
# The model page scrapes a live catalogue, so it goes stale fastest and is also the most likely to
# fail. Hourly, and never fatal.
if [ "$(date -u +%M)" -lt 20 ]; then
  python3 "$STUDY/build_model_page.py" >> "$LOG" 2>&1 || \
    say "cycle $STAMP: model page build failed, keeping the last good copy"
fi
ahead=$(git -C "$STUDY" rev-list --count origin/main..HEAD 2>/dev/null || echo 0)
if [ -n "$(git -C "$STUDY" status --porcelain data index.html logbook.jsonl budget_ledger.json gather-cliff.html roadmap.html models.html 2>/dev/null)" ] || [ "${ahead:-0}" -gt 0 ]; then
  git -C "$STUDY" add data index.html cluster.html gather-cliff.html roadmap.html models.html \
    logbook.jsonl budget_ledger.json >/dev/null 2>&1
  git -C "$STUDY" -c user.name="anh nguyen" -c user.email="qanh@stanford.edu" \
    commit -q -m "campaign cycle $STAMP" >/dev/null 2>&1
  if git -C "$STUDY" push -q origin main >/dev/null 2>&1; then
    say "cycle $STAMP: published"
  else
    say "cycle $STAMP: push failed, will retry next cycle"
  fi
fi

say "cycle $STAMP: done"
