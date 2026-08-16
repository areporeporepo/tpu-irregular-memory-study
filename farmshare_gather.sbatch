#!/bin/bash
#SBATCH --job-name=gather-l40s
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=gather-l40s-%j.out
#
# The same gather measurement this study ran on TPU v5p, v6e and an A100, on a FarmShare L40S.
#
# Why an L40S is worth a slot rather than being a consolation prize: its L2 cache is 48 MB, and the
# on-chip promotion budget we bisected out of the TPU compiler on v5p is 47.94 MiB. The same size,
# for entirely different reasons. On the TPU it is a compile-time placement decision keyed on the
# declared size of the buffer, so the address span the indices touch is irrelevant. On the L40S it
# is a real cache, so the span should matter and the declared size should not. That is a falsifiable
# prediction and this job tests it.
#
#   scp farmshare_gather.sbatch gpu_gather.py <you>@rice.stanford.edu:~/
#   ssh <you>@rice.stanford.edu
#   sbatch farmshare_gather.sbatch
#   # when it finishes:  cat gather-l40s-<jobid>.out
#
# Nothing here needs sudo, and it allocates one GPU for at most half an hour.

set -uo pipefail
echo "=== node and GPU ==="
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 || true

echo
echo "=== finding a torch ==="
# FarmShare uses Lmod. Try the most likely module names, then fall back to a user venv. Whichever
# path works, print the torch and CUDA versions so the result is attributable.
FOUND=""
for m in py-pytorch pytorch python3 python; do
  if module load "$m" 2>/dev/null; then
    if python3 -c 'import torch' 2>/dev/null; then FOUND="module:$m"; break; fi
  fi
done
if [ -z "$FOUND" ] && python3 -c 'import torch' 2>/dev/null; then
  FOUND="system python3"
fi
if [ -z "$FOUND" ]; then
  echo "no torch from modules; building a user venv (a few minutes, once)"
  python3 -m venv "$HOME/.venv-gather" 2>/dev/null || true
  # shellcheck disable=SC1091
  . "$HOME/.venv-gather/bin/activate"
  pip install -q --upgrade pip
  pip install -q torch --index-url https://download.pytorch.org/whl/cu121 \
    || pip install -q torch
  FOUND="venv $HOME/.venv-gather"
fi
echo "torch from: $FOUND"
python3 -c 'import torch;print("torch",torch.__version__,"cuda",torch.version.cuda,
  "device",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")' 2>&1 | tail -2

echo
echo "=== allocation ladder: does the declared buffer size matter? ==="
echo "    TPU: yes, a 3.3x step. prediction for a real cache: no."
python3 gpu_gather.py --sweep --out "gather_l40s_sweep_${SLURM_JOB_ID:-local}.json"

echo
echo "=== span, order and allocation grid: does the touched footprint matter? ==="
echo "    TPU: no, flat to 0.2%. prediction for a 48 MB L2: yes, and near 48 MB."
python3 gpu_gather.py --out "gather_l40s_grid_${SLURM_JOB_ID:-local}.json"

echo
echo "=== the two JSON files to send back ==="
ls -la gather_l40s_*_"${SLURM_JOB_ID:-local}".json 2>/dev/null
echo "--- sweep json ---"; cat "gather_l40s_sweep_${SLURM_JOB_ID:-local}.json" 2>/dev/null
echo "--- grid json ---";  cat "gather_l40s_grid_${SLURM_JOB_ID:-local}.json" 2>/dev/null
