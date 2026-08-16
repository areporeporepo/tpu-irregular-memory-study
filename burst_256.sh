#!/bin/bash
# Use a large slice once, hard, and then delete it no matter what happens.
#
# A v6e-256 costs about $359/hour on spot. The scaling curve it produces is the most valuable
# measurement available to this study, and leaving it running by accident is the most expensive
# mistake available. So the delete is in a trap, and there is a wall-clock cap.
set -uo pipefail

NAME="${1:-anh-big-256}"
ZONE="${2:-us-east1-d}"
STUDY="$HOME/tpu-irregular-memory-study"
DATA="$STUDY/data"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
CAP_MINUTES=50

mkdir -p "$DATA"
say() { printf '%s %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$STUDY/campaign.log"; }

cleanup() {
  say "deleting $NAME (this is the expensive part, so it happens unconditionally)"
  gcloud compute tpus tpu-vm delete "$NAME" --zone="$ZONE" --quiet >/dev/null 2>&1
  say "deleted $NAME"
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------ wait for it
for i in $(seq 1 60); do
  state=$(gcloud compute tpus tpu-vm describe "$NAME" --zone="$ZONE" --format="value(state)" 2>/dev/null | head -1)
  [ -z "$state" ] && { say "$NAME vanished before it was ready"; exit 0; }
  [ "$state" = "READY" ] && break
  [ "$state" = "PREEMPTED" ] && { say "$NAME preempted before use"; exit 0; }
  sleep 20
done
[ "$state" = "READY" ] || { say "$NAME never became READY (last state $state)"; exit 0; }
say "$NAME is READY, capped at ${CAP_MINUTES} minutes from here"

# ------------------------------------------------------------------ set up
say "installing jax across all hosts"
gcloud compute tpus tpu-vm ssh "$NAME" --zone="$ZONE" --worker=all \
  --command='python3.11 -m pip install -q -U "jax[tpu]"' >/dev/null 2>&1
gcloud compute tpus tpu-vm scp "$STUDY/experiment2_fabric_collectives.py" \
  "$NAME:~/experiment2_fabric_collectives.py" --zone="$ZONE" --worker=all >/dev/null 2>&1

# ------------------------------------------------------------------ measure
say "running the scaling curve out to 256 chips"
gcloud compute tpus tpu-vm ssh "$NAME" --zone="$ZONE" --worker=all \
  --command="cd ~ && timeout $((CAP_MINUTES * 60 - 300)) python3.11 experiment2_fabric_collectives.py" \
  > "$DATA/fabric256_${STAMP}.stdout" 2>&1
grep -E "all_to_all|all_reduce" "$DATA/fabric256_${STAMP}.stdout" | tail -60

# ------------------------------------------------------------------ collect from whichever host wrote it
workers=$(gcloud compute tpus tpu-vm describe "$NAME" --zone="$ZONE" \
            --format="value(networkEndpoints.len())" 2>/dev/null | head -1)
for w in $(seq 0 $(( ${workers:-32} - 1 ))); do
  if gcloud compute tpus tpu-vm scp "$NAME:~/fabric_results.json" \
       "$DATA/fabric256_${STAMP}.json" --zone="$ZONE" --worker="$w" >/dev/null 2>&1; then
    say "collected the 256-chip curve from worker $w"
    break
  fi
done
say "burst complete"
