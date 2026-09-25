#!/usr/bin/env bash
# Start the local batch-one Qwen3.8-Flash-Next vLLM endpoint from tt-metal.

set -euo pipefail

model_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tt_metal_root="$(cd "${model_dir}/../../.." && pwd)"
workspace_root="$(dirname "${tt_metal_root}")"
plugin_src="${VLLM_TT_PLUGIN_SRC:-${workspace_root}/vllm-tt-plugin-a48857a/src}"
vllm_bin="${VLLM_BIN:-/home/ttuser/.tenstorrent-venv/bin/vllm}"

if [[ ! -x "${vllm_bin}" ]]; then
    echo "vLLM executable not found: ${vllm_bin}; set VLLM_BIN" >&2
    exit 1
fi
if [[ ! -d "${plugin_src}/vllm_tt_plugin" ]]; then
    echo "vLLM TT plugin source not found: ${plugin_src}; set VLLM_TT_PLUGIN_SRC" >&2
    exit 1
fi

export TT_METAL_HOME="${tt_metal_root}"
export TT_METAL_RUNTIME_ROOT="${tt_metal_root}"
export PYTHONPATH="${tt_metal_root}:${tt_metal_root}/ttnn:${tt_metal_root}/tools:${plugin_src}${PYTHONPATH:+:${PYTHONPATH}}"
export EXTRA_MODELS_DIR="${model_dir}/vllm_bundle"
export TT_VISIBLE_DEVICES="${TT_VISIBLE_DEVICES:-0,1,2,3}"
export MESH_DEVICE="${MESH_DEVICE:-(4,1)}"
export VLLM_PLUGINS="tt,tt_model_registry"
export TT_VLLM_BUILTIN_MODELS="1"
export QWEN38_VLLM_LOG_METRICS="${QWEN38_VLLM_LOG_METRICS:-1}"
export QWEN38_PREFILL_SCHEDULE="${QWEN38_PREFILL_SCHEDULE:-stack_major}"
export QWEN38_DECODE_TRACE_SCHEDULE="${QWEN38_DECODE_TRACE_SCHEDULE:-resident_stack}"
export QWEN38_MOE_KERNEL="${QWEN38_MOE_KERNEL:-sparse_bank}"
# Prefill microchunk plan (doc/prefill_sweep/).  128-row microchunks are the
# accuracy baseline and the packaged default.  QWEN38_PREFILL_CHUNK=512
# QWEN38_PREFILL_CHUNK_ADAPTIVE=1 QWEN38_MOE_PREFILL_SLABS=1 prefills 1.45x
# faster (16k at 262k capacity: 58.2 s -> 39.6 s) but lowers decode top-1
# agreement after long documents by 6-19 points (teacher_ab_20260916/), so it
# is opt-in only.
export QWEN38_PREFILL_CHUNK="${QWEN38_PREFILL_CHUNK:-512}"
export QWEN38_PREFILL_CHUNK_ADAPTIVE="${QWEN38_PREFILL_CHUNK_ADAPTIVE:-1}"
export QWEN38_MOE_PREFILL_SLABS="${QWEN38_MOE_PREFILL_SLABS:-1}"
# Match the packaged long-context profile.  Since the dense (< 2,048-token)
# decode variant became capacity-independent, the only capacity cost is the
# selector variant above 2,048 tokens (~77 ms/token at 262k, ~67 ms at 131k).
export QWEN38_MAX_MODEL_LEN="${QWEN38_MAX_MODEL_LEN:-262144}"
if [[ "${QWEN38_EXPERT_WEIGHT_CACHE:-}" == "off" ]]; then
    unset QWEN38_EXPERT_WEIGHT_CACHE
else
    # tt-model's per-model weight cache (shared with the packaged container);
    # the older ~/.cache/ttnn/models copy is byte-identical and can be removed.
    export QWEN38_EXPERT_WEIGHT_CACHE="${QWEN38_EXPERT_WEIGHT_CACHE:-${HOME}/.cache/tt-model/qwen3.8-flash-next-p300x2/weights/resident-experts-f5d08274-p300-tp4-ep4-bfp4-v1}"
fi
unset TT_MESH_GRAPH_DESC_PATH TT_METAL_KERNEL_PATH

exec "${vllm_bin}" serve Qwen/Qwen3.8-Flash-Next \
    --port "${PORT:-8000}" \
    --max-num-seqs 1 \
    --block-size 64 \
    --max-model-len "${QWEN38_MAX_MODEL_LEN}" \
    --enable-chunked-prefill \
    --max-num-batched-tokens 1024 \
    --async-scheduling \
    --served-model-name Qwen/Qwen3.8-Flash-Next \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --additional-config '{"tt":{"sample_on_device_mode":"all","trace_region_size":1073741824,"fabric_config":"FABRIC_1D","fabric_packet_payload_bytes":8192,"l1_small_size":24576,"trace_mode":"decode_only"}}' \
    "$@"
