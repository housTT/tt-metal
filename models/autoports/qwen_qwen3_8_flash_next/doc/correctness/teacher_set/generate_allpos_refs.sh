#!/usr/bin/env bash
# C0: per-position HF prefill references for the teacher-set prompts up to MAX_TOKENS (default 3300;
# the 6k/12k documents are skipped: one CPU forward over 12k tokens is too slow/large for routine use).
# Run ALONE (the HF oracle + a TT model load exceed host RAM). Usage: generate_allpos_refs.sh [THREADS=12] [MAX_TOKENS=3300]
set -u
THREADS="${1:-12}"; MAX_TOKENS="${2:-3300}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tt_metal="$(cd "$here/../../../../../.." && pwd)"
refs="$here/refs"; mkdir -p "$refs"
export TT_METAL_HOME="$tt_metal" PYTHONPATH="$tt_metal:$tt_metal/ttnn:$tt_metal/tools" PYTHONUNBUFFERED=1
cd "$tt_metal"
for id in $(python3 -c "import json;print(' '.join(p['id'] for p in json.load(open('$here/manifest.json'))['prompts']))"); do
  out="$refs/$id.allpos.pt"
  if [ -s "$out" ]; then echo "[$(date +%H:%M:%S)] skip $id (exists)"; continue; fi
  ref="$refs/$id.refpt"
  if [ -s "$ref" ]; then
    ntok=$(/home/ttuser/.tenstorrent-venv/bin/python -c "import torch;print(torch.load('$ref',weights_only=False)['prompt_tokens'].numel())" 2>/dev/null)
    if [ -n "$ntok" ] && [ "$ntok" -gt "$MAX_TOKENS" ]; then echo "[$(date +%H:%M:%S)] skip $id ($ntok tokens > $MAX_TOKENS)"; continue; fi
  fi
  echo "[$(date +%H:%M:%S)] === allpos $id"
  /home/ttuser/.tenstorrent-venv/bin/python -m models.autoports.qwen_qwen3_8_flash_next.demo.generate_prefill_allpos_reference \
    --manifest "$here/manifest.json" --prompt-id "$id" --output "$out" --threads "$THREADS" 2>&1 | grep -E "allpos-reference|Error|Traceback|rror:" | cut -c1-200
  echo "[$(date +%H:%M:%S)] === done $id rc=${PIPESTATUS[0]}"
done
echo "[$(date +%H:%M:%S)] all-position references done: $(ls "$refs"/*.allpos.pt 2>/dev/null | wc -l) files"
