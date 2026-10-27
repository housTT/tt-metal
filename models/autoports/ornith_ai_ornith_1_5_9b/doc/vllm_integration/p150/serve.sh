#!/usr/bin/env bash
# Launch the shared readiness runner for the p150 (one chip) profile.
# usage: serve.sh <label> <max_num_seqs> <chip> [extra runner args...]
# Environment knobs: HOST_SAMPLING=1 enables the explicit host-sampling compatibility mode;
# TRACKED=1 sets TT_METAL_TRACE_ALLOC_TRACKING=1 in the server environment.
set -euo pipefail
LABEL=$1; SEQS=$2; CHIP=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/p150_env.sh"
export TT_VISIBLE_DEVICES=$CHIP
export ORNITH_VLLM_ALLOW_HOST_SAMPLING=${HOST_SAMPLING:-0}
if [ "${TRACKED:-0}" = "1" ]; then
  export TT_METAL_TRACE_ALLOC_TRACKING=1
  export TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
fi
cd /home/hous/dev/tt-metal-wt-ornith
mkdir -p "$HERE/$LABEL"
{
  echo "label=$LABEL max_num_seqs=$SEQS chip=$CHIP host_sampling=$ORNITH_VLLM_ALLOW_HOST_SAMPLING tracked=${TRACKED:-0} started=$(date -u +%FT%TZ)"
  env | grep -E "^(ORNITH_|TT_|VLLM_|EXTRA_MODELS_DIR|HF_|ARCH_NAME|PYTHONPATH)" | sort
  echo "git_head=$(git rev-parse HEAD)"
  echo "plugin_head=$(git -C $VLLM_TT_PLUGIN_ROOT rev-parse HEAD)"
} > "$HERE/$LABEL/launch_env.txt"
exec "$PY" -m readiness_check.run_vllm_server \
  --stages serve \
  --model-dir "$MODEL_DIR" \
  --hf-model "$HF_MODEL_ID" \
  --mesh-device N150 \
  --max-num-seqs "$SEQS" \
  --max-model-len 262144 \
  --block-size 64 \
  --no-enable-prefix-caching \
  --tt-config '{"trace_region_size": 100000000, "l1_small_size": 32768}' \
  --additional-server-args "--hf-overrides '{\"architectures\": [\"TTOrnithForCausalLM\"]}' --async-scheduling" \
  "$@"
