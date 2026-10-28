#!/usr/bin/env bash
# Hand launch of the B1 primary server on chip 3, port 8001, with the runner's exact server command
# (max_num_seqs 1) so it can run beside the B32 server on chip 2. Logs to b1_primary/server.log.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../vllm_integration/p150/p150_env.sh"
export ORNITH_BATCHED_PREFILL=${BATCHED:-1}
export ORNITH_BATCHED_PREFILL_MAX_LEN=${BATCHED_MAX_LEN:-512}
export TT_VISIBLE_DEVICES=3
export ORNITH_VLLM_ALLOW_HOST_SAMPLING=0
cd /home/hous/dev/tt-metal-wt-ornith
{
  echo "label=b1_primary max_num_seqs=1 chip=3 port=8001 host_sampling=0 tracked=0 batched=$ORNITH_BATCHED_PREFILL batched_max_len=$ORNITH_BATCHED_PREFILL_MAX_LEN started=$(date -u +%FT%TZ)"
  env | grep -E "^(ORNITH_|TT_|VLLM_|EXTRA_MODELS_DIR|HF_|ARCH_NAME|PYTHONPATH)" | sort
  echo "git_head=$(git rev-parse HEAD)"
  echo "git_dirty_files=$(git status --short | wc -l)"
  sha256sum "$MODEL_DIR"/tt/*.py "$MODEL_DIR"/vllm_bundle/*/vllm_metadata.json
  echo "plugin_head=$(git -C $VLLM_TT_PLUGIN_ROOT rev-parse HEAD)"
  echo "server_command=$(cat "$HERE/b1_primary_server_command.txt")"
} > "$HERE/b1_primary/launch_env.txt"
exec $PY -m vllm.entrypoints.openai.api_server --model ornith-ai/Ornith-1.5-9B --block_size 64 --max_num_seqs 1 --port 8001 --max_model_len 262144 --no-enable-prefix-caching --additional-config '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 100000000, "l1_small_size": 32768}}' --hf-overrides '{"architectures": ["TTOrnithForCausalLM"]}' --async-scheduling
