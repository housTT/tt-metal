#!/usr/bin/env bash
# B1: long-prompt prefill sweep, one pytest process per (chunk, slabs) point.
# Usage: doc/prefill_sweep/run_sweep.sh [ISL=16384] [MAX_SEQ_LEN=262144] [CHUNKS="128 256 512"] [SLABS="0 1"]
set -u
ISL="${1:-16384}"; MAXLEN="${2:-262144}"; CHUNKS="${3:-128 256 512}"; SLABS="${4:-0 1}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model_dir="$(cd "$here/../.." && pwd)"
tt_metal="$(cd "$model_dir/../../.." && pwd)"
out_dir="$here/results_isl${ISL}_len${MAXLEN}"; mkdir -p "$out_dir"
export TT_METAL_HOME="$tt_metal" PYTHONPATH="$tt_metal:$tt_metal/ttnn:$tt_metal/tools" PYTHONUNBUFFERED=1
export QWEN38_MOE_KERNEL=sparse_bank QWEN38_PREFILL_SCHEDULE=stack_major QWEN38_DECODE_TRACE_SCHEDULE=resident_stack
export QWEN38_EXPERT_WEIGHT_CACHE="${QWEN38_EXPERT_WEIGHT_CACHE:-/home/ttuser/.cache/tt-model/qwen3.8-flash-next-p300x2/weights/resident-experts-f5d08274-p300-tp4-ep4-bfp4-v1}"
export RUN_QWEN38_PREFILL_SWEEP=1 QWEN38_SWEEP_ISL="$ISL" QWEN38_SWEEP_MAX_SEQ_LEN="$MAXLEN" QWEN38_PREFILL_TIMING_SYNC="${QWEN38_PREFILL_TIMING_SYNC:-1}"
cd "$tt_metal"
for chunk in $CHUNKS; do for slabs in $SLABS; do
  tag="chunk${chunk}_slabs${slabs}"
  echo "[$(date +%H:%M:%S)] === sweep point $tag (ISL=$ISL, max_seq_len=$MAXLEN)"
  QWEN38_PREFILL_CHUNK="$chunk" QWEN38_MOE_PREFILL_SLABS="$slabs" QWEN38_SWEEP_OUT="$out_dir/$tag.json" \
    /home/ttuser/.tenstorrent-venv/bin/python -m pytest -q -s -x \
      models/autoports/qwen_qwen3_8_flash_next/tests/test_full_model_perf.py::test_long_prompt_prefill_sweep_point \
      2>&1 | grep -E "prefill_sweep_point|PASSED|FAILED|passed|failed|Error|assert" | cut -c1-600
  echo "[$(date +%H:%M:%S)] === done $tag rc=${PIPESTATUS[0]}"
done; done
echo "[$(date +%H:%M:%S)] sweep complete: $out_dir"
