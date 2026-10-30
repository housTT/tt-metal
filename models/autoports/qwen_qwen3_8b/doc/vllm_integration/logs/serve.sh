#!/usr/bin/env bash
# usage: serve.sh <chip> <port> <timeout_s> [extra env assignments...]
# Launches the vLLM server through the in-tree readiness runner (serve stage only) with the stage environment.
chip="$1"; port="$2"; limit="$3"; shift 3
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
exec timeout -s INT "$limit" env HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TT_METAL_HOME=/home/hous/dev/tt-metal \
  PYTHONPATH=/home/hous/dev/tt-metal-wt-qwen3:/home/hous/dev/tt-metal:/home/hous/dev/skills/plugins/tt-model-bringup/runtime \
  TT_MESH_GRAPH_DESC_PATH=/home/hous/dev/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  VLLM_TT_PLUGIN_ROOT=/home/hous/dev/vllm/plugins/vllm-tt-plugin VLLM_SERVER_DEV_MODE=1 \
  EXTRA_MODELS_DIR=/home/hous/dev/tt-metal-wt-qwen3/models/autoports/qwen_qwen3_8b/vllm \
  TT_VISIBLE_DEVICES="$chip" "$@" /home/hous/dev/tt-metal/python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages serve --model-dir models/autoports/qwen_qwen3_8b --hf-model Qwen/Qwen3-8B --mesh-device N150 \
  --max-num-seqs 32 --max-model-len 40960 --block-size 32 --port "$port" \
  --tt-config '{"trace_region_size": 536870912}' \
  --additional-server-args "--enable-prefix-caching --async-scheduling"
