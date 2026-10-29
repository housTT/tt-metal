#!/usr/bin/env bash
# Stage environment from the operator brief; first argument is the chip, the rest is the command.
chip="$1"; shift
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
exec env HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TT_METAL_HOME=/home/hous/dev/tt-metal \
  PYTHONPATH=/home/hous/dev/tt-metal-wt-qwen3:/home/hous/dev/tt-metal:/home/hous/dev/skills/plugins/tt-model-bringup/runtime \
  TT_MESH_GRAPH_DESC_PATH=/home/hous/dev/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  VLLM_TT_PLUGIN_ROOT=/home/hous/dev/vllm/plugins/vllm-tt-plugin VLLM_SERVER_DEV_MODE=1 \
  TT_VISIBLE_DEVICES="$chip" "$@"
