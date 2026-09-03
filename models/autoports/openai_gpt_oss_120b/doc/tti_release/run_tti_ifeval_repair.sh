#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT=/home/ttuser/dev/gpt-oss-20b
TT_METAL_ROOT=${WORKSPACE_ROOT}/tt-metal
TTI_WORK_ROOT=${WORKSPACE_ROOT}/tti-release/openai_gpt_oss_120b
TTI_ROOT=${TTI_WORK_ROOT}/tt-inference-server
REPAIR_SPEC=${TTI_WORK_ROOT}/specs/autoport_ifeval_repair_spec.json
REPAIR_LOG=${TTI_WORK_ROOT}/evidence/tti_ifeval_repair.log

cd "${TT_METAL_ROOT}"
source .agents/scripts/gpt_oss_workspace_env.sh

export CACHE_ROOT=${TTI_WORK_ROOT}/tti_cache
export PERSISTENT_VOLUME_ROOT=${TTI_WORK_ROOT}/persistent_volume
export SERVICE_PORT=8000
export HF_HOME=${WORKSPACE_ROOT}/.cache/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN

cd "${TTI_ROOT}"
python3 run.py \
  --model gpt-oss-120b \
  --runtime-model-spec-json "${REPAIR_SPEC}" \
  --tt-device p150x4 \
  --engine vllm \
  --workflow evals \
  --server-url http://127.0.0.1:8000 \
  --service-port 8000 \
  --no-auth \
  --skip-system-sw-validation \
  --disable-trace-capture \
  --eval-tasks meta_ifeval \
  2>&1 | tee "${REPAIR_LOG}"
