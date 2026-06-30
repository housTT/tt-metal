#!/usr/bin/env bash
# Run inference with HuggingFaceTB/SmolLM2-1.7B-Instruct on a SINGLE Blackhole p150 chip
# of the QB2, using the official tt-inference-server image (proven compatible with this
# host's driver) and this host's local HuggingFace cache.
#
# Usage:
#   ./run_smollm2_p150.sh                      # default sample prompt (batch-1)
#   ./run_smollm2_p150.sh /path/to/prompts.json
#
# Prompt file format (JSON array):
#   [ { "prompt": "your prompt here" }, ... ]
#
# Notes:
# - Uses MESH_DEVICE=P150 -> a 1x1 mesh = one chip.
# - Disables the demo's hardcoded fabric_config (single chip needs no inter-chip ethernet
#   fabric; leaving it on tries to bring up fabric across all 4 chips and can fail).
# - HF cache is mounted read-write under the container user's home (uid 1000 matches the
#   host 'ttuser', so no root-owned files leak into the cache). HF offline mode is on, so
#   it uses the already-downloaded weights and never hits the network.
# - If device open fails with an ethernet/sysmem/"reset the board" error (e.g. after a
#   crashed job), reset first:   tt-smi -r
set -euo pipefail

IMG="ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.17.0-8c48a10-f52987a"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
MODEL="HuggingFaceTB/SmolLM2-1.7B-Instruct"
PROMPTS_HOST="${1:-}"

DOCKER_PROMPT_ARGS=()
INNER_PROMPT_OPT=""
if [[ -n "$PROMPTS_HOST" ]]; then
  PROMPTS_HOST="$(readlink -f "$PROMPTS_HOST")"
  DOCKER_PROMPT_ARGS=(-v "$PROMPTS_HOST:/home/container_app_user/prompts.json:ro")
  INNER_PROMPT_OPT="--input_prompts /home/container_app_user/prompts.json"
fi

docker run --rm --name smollm2-p150 \
  --device /dev/tenstorrent \
  --mount type=bind,source=/dev/hugepages-1G,target=/dev/hugepages-1G \
  -v "$HF_CACHE:/home/container_app_user/hf_cache" \
  "${DOCKER_PROMPT_ARGS[@]}" \
  -e HF_HOME=/home/container_app_user/hf_cache \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e TT_METAL_CACHE=/tmp/ttmetal_cache \
  -e MAX_PREFILL_CHUNK_SIZE=128 \
  -e HF_MODEL="$MODEL" -e MESH_DEVICE=P150 \
  --entrypoint bash "$IMG" -c "
    set -e
    cd /home/container_app_user/tt-metal
    source python_env/bin/activate
    mkdir -p /tmp/ttmetal_cache
    # single chip => no fabric
    sed -i 's/\"fabric_config\": True/\"fabric_config\": False/' models/tt_transformers/demo/simple_text_demo.py
    python3 -m pytest models/tt_transformers/demo/simple_text_demo.py \
      -k 'performance and batch-1' -s $INNER_PROMPT_OPT
  "
