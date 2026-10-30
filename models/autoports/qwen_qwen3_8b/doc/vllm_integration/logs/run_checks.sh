#!/usr/bin/env bash
# usage: run_checks.sh <port> <stage> [extra runner args...]
# Attaches one packaged readiness runner check stage to the live server on <port> with the stage environment.
port="$1"; stage="$2"; shift 2
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
exec env HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=/home/hous/dev/tt-metal-wt-qwen3:/home/hous/dev/tt-metal:/home/hous/dev/skills/plugins/tt-model-bringup/runtime \
  VLLM_TT_PLUGIN_ROOT=/home/hous/dev/vllm/plugins/vllm-tt-plugin \
  timeout 7200 /home/hous/dev/tt-metal/python_env/bin/python -m readiness_check.run_vllm_server \
  --stages "$stage" --server-url "http://localhost:$port" --model-dir models/autoports/qwen_qwen3_8b --hf-model Qwen/Qwen3-8B \
  --max-num-seqs 32 --max-model-len 40960 --block-size 32 \
  --server-log models/autoports/qwen_qwen3_8b/readiness_vllm/server.log "$@"
