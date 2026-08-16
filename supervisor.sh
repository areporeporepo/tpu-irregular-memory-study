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
SLICE="anh-v6e-32"
DEV="anh-dev1"
# Only these two zones had multi-chip v6e capacity in the 2026-08-16 sweep. Ordered by
# preference: us-east1-d is closer to everything else we own.
ZONES="us-east1-d asia-northeast1-b"
ACCEL="v6e-32"
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

read -r zone state <<< "$(find_slice)"

# ---------------------------------------------------------------- keep a slice alive
if [ "$state" = "MISSING" ] || [ "$state" = "PREEMPTED" ] || [ "$state" = "TERMINATED" ]; then
  say "slice $state, hunting capacity"
  if [ "$state" != "MISSING" ]; then
    gcloud compute tpus tpu-vm delete "$SLICE" --zone="$zone" --quiet >/dev/null 2>&1
  fi
  got=""
  for z in $ZONES; do
    for size in 32 16 8; do
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

if gcloud compute tpus tpu-vm scp "$SLICE:~/fabric_results.json" \
     "$DATA/fabric_${STAMP}.json" --zone="$zone" --worker=0 >/dev/null 2>&1; then
  n=$(grep -c '"op"' "$DATA/fabric_${STAMP}.json" 2>/dev/null || echo 0)
  say "cycle $STAMP: collected $n fabric records"
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

say "cycle $STAMP: done"
