#!/bin/bash
# Run this ON Sherlock (or Marlowe) after logging in. It changes nothing and allocates nothing: it
# only reports what is available, so the benchmark job can be written against the real environment
# instead of a guess at module names.
#
#   ssh qanh@login.sherlock.stanford.edu
#   bash sherlock_probe.sh 2>&1 | tee sherlock_probe.txt
#
# Then paste sherlock_probe.txt back. Everything below is read-only.

echo "=== cluster ==="
hostname
scontrol show config 2>/dev/null | grep -i clustername

echo
echo "=== partitions I can submit to ==="
sinfo -o "%20P %10a %10l %6D %10T %N" 2>/dev/null | head -25

echo
echo "=== GPU types present, by partition ==="
sinfo -o "%20P %30G %10D %10T" 2>/dev/null | grep -i gpu | head -30

echo
echo "=== my accounts and any GPU limits ==="
sacctmgr -n show assoc user="$USER" format=Account,Partition,QOS,GrpTRES,MaxTRES 2>/dev/null | head -15
sshare -U 2>/dev/null | head -8

echo
echo "=== what H100 / A100 / L40S nodes exist and whether any are idle ==="
for g in h100 a100 l40s v100; do
  printf '%-6s ' "$g"
  sinfo -N -o "%N %G %t" 2>/dev/null | grep -i "$g" | awk '{print $3}' | sort | uniq -c \
    | tr '\n' ' '
  echo
done

echo
echo "=== pytorch availability, in order of preference ==="
module --version 2>&1 | head -2
for m in py-pytorch pytorch python cuda; do
  printf '%-12s ' "$m"
  (module avail "$m" 2>&1 | grep -oE "${m}[/-][0-9][^ ]*" | sort -u | tr '\n' ' ') || true
  echo
done
python3 -c 'import torch;print("system torch", torch.__version__, "cuda", torch.version.cuda)' 2>&1 | tail -1

echo
echo "=== scratch space for the job ==="
for d in "$SCRATCH" "$GROUP_SCRATCH" "$HOME"; do
  [ -n "$d" ] && printf '%-28s %s\n' "$d" "$(df -h "$d" 2>/dev/null | awk 'NR==2{print $4" free"}')"
done
