#!/bin/bash
# Can a current open model actually be served on a TPU v6e, and what does a token cost?
#
# This is the question that converts every microsecond in this study into the unit the market
# prices. Qwen3.8-27B is 27B dense, BF16, about 56 GB of weights, so it needs roughly four v6e
# chips for weights alone and comfortably fits eight. Its model card recommends vLLM, SGLang or
# TokenSpeed and does not mention JAX or TPU, which is exactly the risk worth measuring: TPU
# software support for new architectures (this one uses Gated DeltaNet linear attention) tends to
# lag, and if it does, that lag is itself the finding.
#
# Strategy: establish the pipeline on a small known-good model first, then attempt the 27B. A
# failure on the 27B with a success on the small model is a clean result about ecosystem lag. A
# failure on both is a setup problem and says nothing.
#
#   ./experiment7_serving.sh SLICE ZONE
set -uo pipefail

NAME="${1:-anh-steady-16}"
ZONE="${2:-us-east1-d}"
STUDY="$HOME/tpu-irregular-memory-study"
DATA="$STUDY/data"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SMALL="Qwen/Qwen3-4B"
BIG="Qwen/Qwen3.8-27B"

mkdir -p "$DATA"
say() { printf '%s %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$STUDY/campaign.log"; }

say "serving test on $NAME: installing vllm and the TPU backend on worker 0"
gcloud compute tpus tpu-vm ssh "$NAME" --zone="$ZONE" --worker=0 --command='
  python3.11 -m pip install -q "vllm==0.25.0" "tpu-inference==0.25.0" 2>&1 | tail -2
  python3.11 -c "import vllm; print(\"vllm\", vllm.__version__)"
  python3.11 -c "import tpu_inference; print(\"tpu_inference present\")" 2>&1 | tail -1
' > "$DATA/serving_install_${STAMP}.log" 2>&1
tail -3 "$DATA/serving_install_${STAMP}.log"

for MODEL in "$SMALL" "$BIG"; do
  tag=$(echo "$MODEL" | tr '/' '_')
  say "attempting to serve $MODEL"
  gcloud compute tpus tpu-vm ssh "$NAME" --zone="$ZONE" --worker=0 --command="
    export VLLM_TARGET_DEVICE=tpu HF_HUB_ENABLE_HF_TRANSFER=0
    timeout 1800 python3.11 - <<'PYEOF'
import json, time, os
try:
    from vllm import LLM, SamplingParams
except Exception as e:
    print(json.dumps({'model': '$MODEL', 'stage': 'import', 'error': str(e)[:300]})); raise SystemExit
try:
    # tensor_parallel_size 8 uses one host's chips, which avoids multi-host serving entirely.
    llm = LLM(model='$MODEL', tensor_parallel_size=8, max_model_len=4096,
              download_dir='/tmp/hf', enforce_eager=False)
except Exception as e:
    print(json.dumps({'model': '$MODEL', 'stage': 'load', 'error': str(e)[:600]})); raise SystemExit
prompts = ['Explain why irregular memory access is slow on systolic-array accelerators.'] * 32
sp = SamplingParams(temperature=0.0, max_tokens=128)
llm.generate(prompts[:2], sp)                     # warm the compile out of the measurement
t0 = time.perf_counter()
outs = llm.generate(prompts, sp)
dt = time.perf_counter() - t0
tok = sum(len(o.outputs[0].token_ids) for o in outs)
print(json.dumps({'model': '$MODEL', 'stage': 'ok', 'prompts': len(prompts),
                  'output_tokens': tok, 'seconds': round(dt, 3),
                  'tokens_per_sec': round(tok / dt, 1),
                  'chips': 8, 'tokens_per_sec_per_chip': round(tok / dt / 8, 2)}))
PYEOF
  " > "$DATA/serving_${tag}_${STAMP}.log" 2>&1
  grep -o '{.*}' "$DATA/serving_${tag}_${STAMP}.log" | tail -1 | tee -a "$DATA/serving_results.jsonl"
done
say "serving test complete"
