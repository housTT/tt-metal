#!/usr/bin/env bash
# The real-weight half of the full-context evidence, plus the model-free SDPA reproducer it needs.
#
# Neither stage 1 nor stage 2 ran the advertised 262143-token context on the **real checkpoint** -
# `real_weights=True` appears only in their 8192-token tests - so this stage is the first to, and it
# found two scale failures the stand-in weights never showed. This runs, strictly one at a time:
#
#   1. the `full_attention` precision bisection on real weights (which group costs the prefill tail
#      its scale, walking from the fused control that passes to the shipped policy that does not);
#   2. the `linear_attention` attribution on real weights;
#   3. `probe_sdpa_peakiness.sh`, which tests stage 1's untested claim that the chunked-SDPA defect is
#      "far smaller" on peaked attention than on the flat synthetic case.
#
#   bash models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/run_realweight_longcontext.sh
#
# Watch it with:
#   tail -f models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/logs/realweight_longcontext.log
set -uo pipefail
REPO=/home/ttuser/dev/qwen/tt-metal
cd "$REPO"
ART="$REPO/models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder"
P="$ART/probes"
L="$ART/logs"
S="$L/realweight_longcontext.log"
source "$REPO/models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/ttenv.sh" > /dev/null

run() {  # run <label> <logfile> <command...>
  local label="$1" log="$2"
  shift 2
  echo "== $(date -Is) $label" >> "$S"
  "$@" > "$log" 2>&1
  local rc=$?
  grep -E "^  " "$log" >> "$S" || true
  echo "== $(date -Is) $label exit $rc" >> "$S"
}

run "full_attention precision bisection, real weights, 262143 tokens" \
  "$L/probe_long_context_precision_real.log" \
  python "$P/probe_long_context_precision.py" --real-weights
run "linear_attention attribution, real weights, 262143 tokens" \
  "$L/probe_long_context_linear_real.log" \
  python "$P/probe_long_context_linear.py" --real-weights
echo "== $(date -Is) model-free SDPA peakiness sweep" >> "$S"
bash "$P/probe_sdpa_peakiness.sh"
grep -E "^== SCALE|tail_pcc" "$L/probe_sdpa_peakiness.log" >> "$S" || true
echo "== $(date -Is) REAL-WEIGHT LONG-CONTEXT DONE" >> "$S"
