# Single-chip (p150) serving environment for the Ornith-1.5-9B autoport on this host.
# Source this file, then set TT_VISIBLE_DEVICES to the chip the process may use.
export HF_HOME=/home/hous/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TT_METAL_HOME=/home/hous/dev/ornith-1.5-9b/tt-metal
export PYTHONPATH=/home/hous/dev/tt-metal-wt-ornith:/home/hous/dev/ornith-1.5-9b/tt-metal:/home/hous/dev/skills/plugins/tt-model-bringup/runtime
export TT_MESH_GRAPH_DESC_PATH=/home/hous/dev/ornith-1.5-9b/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export ARCH_NAME=blackhole
export ORNITH_PRECISION_CONFIG=/home/hous/dev/tt-metal-wt-ornith/models/autoports/ornith_ai_ornith_1_5_9b/doc/datatype_sweep/p150_head8_hifi2_kv8_policy.json
export ORNITH_VLLM_POOL_TOKENS=262208
export ORNITH_PREFIX_CACHE_BYTES=0
export ORNITH_VLLM_ALLOW_HOST_SAMPLING=${ORNITH_VLLM_ALLOW_HOST_SAMPLING:-0}
export ORNITH_ACTIVE8_MODE=active8
export ORNITH_MODEL_PATH=/home/hous/dev/ornith-1.5-9b/upstream
export VLLM_TT_PLUGIN_ROOT=/home/hous/dev/ornith-1.5-9b/vllm-tt-plugin
export VLLM_SERVER_DEV_MODE=1
export EXTRA_MODELS_DIR=/home/hous/dev/tt-metal-wt-ornith/models/autoports/ornith_ai_ornith_1_5_9b/vllm_bundle
export TT_VLLM_BUILTIN_MODELS=0
export PY=/home/hous/dev/ornith-1.5-9b/tt-metal/python_env/bin/python
export MODEL_DIR=/home/hous/dev/tt-metal-wt-ornith/models/autoports/ornith_ai_ornith_1_5_9b
export HF_MODEL_PATH=/home/hous/dev/ornith-1.5-9b/upstream
export HF_MODEL_ID=ornith-ai/Ornith-1.5-9B
export TT_MODEL_BRINGUP_ROOT=/home/hous/dev/skills/plugins/tt-model-bringup
