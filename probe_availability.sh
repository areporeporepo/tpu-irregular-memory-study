#!/bin/bash
# When can a student actually get TPU capacity?
#
# Everything we know about availability was measured at 04:00 on a Sunday, which is exactly when
# enterprise demand is lowest. That makes every capacity claim in this study provisional until it
# is measured across hours and across weekdays. So: once an hour, for the next two weeks, try to
# claim a small slice in each candidate zone, record the outcome, and delete anything that
# succeeds within seconds.
#
# Cost: a failed probe is free. A successful one lives about ninety seconds, which is a few cents.
# The resulting series is a dataset nobody publishes and every student on a shared project wants.
#
#   ./probe_availability.sh          # one round over all zones
set -uo pipefail

STUDY="$HOME/tpu-irregular-memory-study"
SERIES="$STUDY/data/availability.jsonl"
SIZE="${PROBE_SIZE:-8}"
ZONES="us-east1-d us-east5-a us-east5-b us-east5-c us-south1-a us-central1-a europe-west4-a asia-northeast1-b"
STAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
DOW=$(date -u +%u)          # 1=Monday, 7=Sunday
HOUR=$(date -u +%H)

mkdir -p "$STUDY/data"

# Reap anything a previous run left behind, before claiming anything new. This replaces the `wait`
# that used to sit at the end of this script: a probe slice in asia-northeast1-b took over twelve
# minutes to delete on 2026-08-16, the wait blocked on it, and the cycle hit its watchdog and was
# killed before it could publish. Deletes are still issued immediately below; they simply are not
# waited for. Anything that slips through is caught here on the next run, which is a better
# guarantee than blocking, because it also catches slices left by a cycle that was killed.
for z in $ZONES; do
  for stale in $(gcloud compute tpus tpu-vm list --zone="$z" \
                   --format="value(name)" 2>/dev/null | grep '^probe-' || true); do
    gcloud compute tpus tpu-vm delete "$stale" --zone="$z" --quiet >/dev/null 2>&1 &
    echo "  reaping leftover $stale in $z"
  done
done

for z in $ZONES; do
  name="probe-${SIZE}-${z##*-}-$(date -u +%H%M%S)"
  start=$(date +%s)
  out=$(gcloud compute tpus tpu-vm create "$name" --zone="$z" \
        --accelerator-type="v6e-$SIZE" --version=v2-alpha-tpuv6e --spot \
        --labels=purpose=availability-probe 2>&1)
  elapsed=$(( $(date +%s) - start ))

  if echo "$out" | grep -qi "no more capacity\|Insufficient capacity"; then
    verdict=no_capacity
  elif echo "$out" | grep -qi "does not have permission"; then
    verdict=no_permission
  elif echo "$out" | grep -qi "Quota\|exceeded"; then
    verdict=quota
  elif echo "$out" | grep -qi "error"; then
    verdict=other_error
  else
    verdict=available
    # Delete immediately. The measurement is whether it could be claimed, not what it can compute.
    gcloud compute tpus tpu-vm delete "$name" --zone="$z" --quiet >/dev/null 2>&1 &
  fi

  printf '{"at":"%s","dow":%s,"hour":%s,"zone":"%s","size":%s,"verdict":"%s","seconds":%s}\n' \
    "$STAMP" "$DOW" "$HOUR" "$z" "$SIZE" "$verdict" "$elapsed" >> "$SERIES"
  printf '  %-18s v6e-%-3s %-14s %ss\n' "$z" "$SIZE" "$verdict" "$elapsed"
done

# Deliberately no `wait` here. See the reaper at the top of this file for why, and for what
# guarantees the probe slices actually go away.
n=$(wc -l < "$SERIES" | tr -d ' ')
echo "availability series now $n observations"
