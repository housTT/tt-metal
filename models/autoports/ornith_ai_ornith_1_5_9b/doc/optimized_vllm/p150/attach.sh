#!/usr/bin/env bash
# Attach readiness-runner stages and the stage 10 serving clients to the live server on port 8000.
# usage: attach.sh <label> <max_num_seqs> <stage-or-client> [extra runner args]
set -euo pipefail
LABEL=$1; SEQS=$2; WHAT=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../vllm_integration/p150/p150_env.sh"
cd /home/hous/dev/tt-metal-wt-ornith
OUT=$HERE/$LABEL
mkdir -p "$OUT"
case "$WHAT" in
  mixed)
    $PY "$HERE/mixed_shape_check.py" --model "$HF_MODEL_ID" --server-log "$MODEL_DIR/readiness_vllm/server.log" \
      --output "$OUT/mixed_shape_check.json" 2>&1 | tee "$OUT/mixed_shape_check.log"
    $PY "$HERE/trace_reuse_summary.py" --server-log "$MODEL_DIR/readiness_vllm/server.log" \
      --mixed-shape "$OUT/mixed_shape_check.json" --output "$OUT/trace_reuse_summary.json"
    ;;
  nonaligned)
    $PY "$HERE/nonaligned_requests.py" --max-num-seqs "$SEQS" --output "$OUT/nonaligned_requests.json" 2>&1 \
      | tee "$OUT/nonaligned_requests.log"
    ;;
  *)
    $PY -m readiness_check.run_vllm_server --stages "$WHAT" --server-url http://localhost:8000 \
      --model-dir "$MODEL_DIR" --hf-model "$HF_MODEL_ID" --mesh-device N150 --max-num-seqs "$SEQS" \
      --max-model-len 262144 --block-size 64 --server-log "$MODEL_DIR/readiness_vllm/server.log" "$@" 2>&1 \
      | tee "$OUT/runner_${WHAT//,/_}.log"
    ;;
esac
