#!/bin/bash
# Claim a slice, run everything that needs that slice size, delete it from a trap.
#
# Generalised from burst_256.sh, because the geometry question needs the SAME subset size measured
# from several different parent slices: 16 chips carved from a 32, a 64, a 128 and a 256 behave
# differently, and that difference is the finding. One parent per burst.
#
#   ./burst.sh 64 us-east1-d
set -uo pipefail

SIZE="${1:?usage: burst.sh SIZE ZONE}"
ZONE="${2:-us-east1-d}"
NAME="anh-burst-$SIZE"
STUDY="$HOME/tpu-irregular-memory-study"
DATA="$STUDY/data"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
WAIT_MINUTES=25

mkdir -p "$DATA"
say() { printf '%s %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$STUDY/campaign.log"; }

cleanup() {
  say "deleting $NAME unconditionally"
  gcloud compute tpus tpu-vm delete "$NAME" --zone="$ZONE" --quiet >/dev/null 2>&1
  say "deleted $NAME"
}

say "claiming v6e-$SIZE in $ZONE"
out=$(gcloud compute tpus tpu-vm create "$NAME" --zone="$ZONE" --accelerator-type="v6e-$SIZE" \
      --version=v2-alpha-tpuv6e --spot --labels=owner=anh,study=irregular-memory 2>&1)
if echo "$out" | grep -qi "no more capacity\|Insufficient\|RESOURCE_EXHAUSTED"; then
  say "v6e-$SIZE in $ZONE: no capacity"; exit 0
fi
if echo "$out" | grep -qi "error"; then
  say "v6e-$SIZE in $ZONE: $(echo "$out" | grep -i -m1 message | cut -c1-90)"; exit 0
fi
trap cleanup EXIT INT TERM

for i in $(seq 1 $((WAIT_MINUTES * 3))); do
  state=$(gcloud compute tpus tpu-vm describe "$NAME" --zone="$ZONE" --format="value(state)" 2>/dev/null | head -1)
  [ -z "$state" ] && { say "$NAME vanished"; exit 0; }
  [ "$state" = "READY" ] && break
  [ "$state" = "PREEMPTED" ] && { say "$NAME preempted before use"; exit 0; }
  sleep 20
done
[ "$state" = "READY" ] || { say "$NAME never ready (last: $state)"; exit 0; }
say "$NAME READY"

gcloud compute tpus tpu-vm ssh "$NAME" --zone="$ZONE" --worker=all \
  --command='python3.11 -c "import jax" 2>/dev/null || python3.11 -m pip install -q -U "jax[tpu]"' \
  >/dev/null 2>&1
say "jax ready on $NAME"

# Everything that wants this parent size. The sweep gives the scaling curve; the geometry run
# gives subset-shape effects at this parent; the capture banks HLO and traces we keep forever.
for exp in experiment2_fabric_collectives experiment4_geometry experiment6_capture_artifacts; do
  gcloud compute tpus tpu-vm scp "$STUDY/$exp.py" "$NAME:~/$exp.py" --zone="$ZONE" --worker=all >/dev/null 2>&1
  say "running $exp on v6e-$SIZE"
  gcloud compute tpus tpu-vm ssh "$NAME" --zone="$ZONE" --worker=all \
    --command="cd ~ && python3.11 $exp.py" > "$DATA/${exp}_${SIZE}chip_${STAMP}.stdout" 2>&1
done

# JAX process 0 is not necessarily worker 0, so try them all.
workers=$(gcloud compute tpus tpu-vm describe "$NAME" --zone="$ZONE" \
            --format="value(networkEndpoints.len())" 2>/dev/null | head -1)
for f in fabric_results.json geometry_results.json; do
  for w in $(seq 0 $(( ${workers:-8} - 1 ))); do
    if gcloud compute tpus tpu-vm scp "$NAME:~/$f" \
         "$DATA/${f%.json}_${SIZE}chip_${STAMP}.json" --zone="$ZONE" --worker="$w" >/dev/null 2>&1; then
      say "collected $f from worker $w"; break
    fi
  done
done
say "burst on v6e-$SIZE complete"
