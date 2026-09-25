#!/usr/bin/env bash
# C0: generate the HF bf16 greedy reference for every teacher-set prompt (one process each, CPU).
# Usage: doc/correctness/teacher_set/generate_refs.sh [THREADS=16] [IDS...]
set -u
THREADS="${1:-16}"; shift || true
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model_dir="$(cd "$here/../../.." && pwd)"
tt_metal="$(cd "$model_dir/../../.." && pwd)"
refs="$here/refs"; mkdir -p "$refs"
export TT_METAL_HOME="$tt_metal" PYTHONPATH="$tt_metal:$tt_metal/ttnn:$tt_metal/tools" PYTHONUNBUFFERED=1
ids="$*"
if [ -z "$ids" ]; then ids=$(python3 -c "import json;print(' '.join(p['id'] for p in json.load(open('$here/manifest.json'))['prompts']))"); fi
cd "$tt_metal"
for id in $ids; do
  out="$refs/$id.refpt"
  if [ -s "$out" ]; then echo "[$(date +%H:%M:%S)] skip $id (exists)"; continue; fi
  echo "[$(date +%H:%M:%S)] === generating $id"
  /home/ttuser/.tenstorrent-venv/bin/python -m models.autoports.qwen_qwen3_8_flash_next.demo.generate_hf_reference \
    --output "$out" --manifest "$here/manifest.json" --prompt-id "$id" --threads "$THREADS" 2>&1 \
    | grep -E "hf-reference step=(1|25|50|75|100)/|Error|Traceback|rror:" | cut -c1-200
  echo "[$(date +%H:%M:%S)] === done $id rc=${PIPESTATUS[0]} $(ls -la "$out" 2>/dev/null | awk '{print $5" bytes"}')"
done
echo "[$(date +%H:%M:%S)] all references done: $(ls "$refs" | wc -l) files"
