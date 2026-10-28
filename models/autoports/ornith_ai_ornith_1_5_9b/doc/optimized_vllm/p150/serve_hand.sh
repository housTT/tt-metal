#!/usr/bin/env bash
# Hand launch of a server on a given chip and port with the runner's exact server command, so a second server can run beside the runner-launched one. usage: serve_hand.sh <label> <max_num_seqs> <chip> <port>
# (max_num_seqs 1) so it can run beside the B32 server on chip 2. Logs to b1_primary/server.log.
set -euo pipefail
LABEL=$1; SEQS=$2; CHIP=$3; PORT=$4
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../vllm_integration/p150/p150_env.sh"
export ORNITH_BATCHED_PREFILL=${BATCHED:-1}
export ORNITH_BATCHED_PREFILL_MAX_LEN=${BATCHED_MAX_LEN:-512}
export TT_VISIBLE_DEVICES=$CHIP
export ORNITH_VLLM_ALLOW_HOST_SAMPLING=${HOST_SAMPLING:-0}
mkdir -p "$HERE/$LABEL"
cd /home/hous/dev/tt-metal-wt-ornith
{
  echo "label=$LABEL max_num_seqs=$SEQS chip=$CHIP port=$PORT host_sampling=$ORNITH_VLLM_ALLOW_HOST_SAMPLING tracked=0 batched=$ORNITH_BATCHED_PREFILL batched_max_len=$ORNITH_BATCHED_PREFILL_MAX_LEN started=$(date -u +%FT%TZ)"
  env | grep -E "^(ORNITH_|TT_|VLLM_|EXTRA_MODELS_DIR|HF_|ARCH_NAME|PYTHONPATH)" | sort
  echo "git_head=$(git rev-parse HEAD)"
  echo "git_dirty_files=$(git status --short | wc -l)"
  sha256sum "$MODEL_DIR"/tt/*.py "$MODEL_DIR"/vllm_bundle/*/vllm_metadata.json
  echo "plugin_head=$(git -C $VLLM_TT_PLUGIN_ROOT rev-parse HEAD)"
  echo "server_command=see exec line of serve_hand.sh"
} > "$HERE/$LABEL/launch_env.txt"
exec $PY -m vllm.entrypoints.openai.api_server --model ornith-ai/Ornith-1.5-9B --block_size 64 --max_num_seqs "$SEQS" --port "$PORT" --max_model_len 262144 --no-enable-prefix-caching --additional-config '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 100000000, "l1_small_size": 32768}}' --hf-overrides '{"architectures": ["TTOrnithForCausalLM"]}' --async-scheduling
