#!/usr/bin/env bash
# Warmed prefill + traced decode profiling for one (layer kind, phase, implementation) triple.
#   usage: run_perf.sh <linear_attention|full_attention> <prefill|decode|decode_batch32> [fused|functional]
#
# ``decode_batch32`` is the same traced-decode window at the advertised max_batch, which is a
# different graph (see the --perf-batch fixture), so it gets its own artifacts.
#
# This checkout is built in-tree with ENABLE_TRACY=ON, so the ordinary stage environment is
# also the profiling environment - there is no separate profiler tree to switch to.
#
# Artifacts land under doc/fused_decoder/tracy/<impl>/<kind>/ so the before (functional) and
# after (fused) tables sit side by side and were produced by the same script.
set -euo pipefail
REPO=/home/ttuser/dev/qwen/tt-metal
cd "$REPO"
source "$REPO/models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/ttenv.sh" > /dev/null

KIND="$1"
PHASE="$2"
IMPL="${3:-fused}"
ART="$REPO/models/autoports/qwen_qwen3_6_27b/doc/fused_decoder"
OUT="$ART/tracy/$IMPL/$KIND"
mkdir -p "$OUT" "$ART/logs"

# `python -m tracy` re-joins the wrapped command without shell quoting, so address the test by
# node id (no spaces) rather than with a -k selector.
BATCH=1
case "$PHASE" in
  prefill)        NODE="test_perf_prefill[$KIND]"; SIGN=PERF_PREFILL ;;
  decode)         NODE="test_perf_decode_traced[$KIND]"; SIGN=PERF_DECODE ;;
  decode_batch32) NODE="test_perf_decode_traced[$KIND]"; SIGN=PERF_DECODE; BATCH=32 ;;
  *) echo "bad phase $PHASE"; exit 2 ;;
esac

LOG="$ART/logs/tracy_${IMPL}_${KIND}_${PHASE}.log"
echo "== tracy run: $IMPL $KIND $PHASE -> $LOG"
python -m tracy -r -p -v -m pytest \
  "$REPO/models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder_perf.py::$NODE" \
  -s -q --impl "$IMPL" --perf-batch "$BATCH" > "$LOG" 2>&1

# The post-processed ops CSV lands under $TT_METAL_HOME, not the working directory.
CSV=$(find "$TT_METAL_HOME/generated/profiler/reports" "$PWD/generated/profiler/reports" \
        -name 'ops_perf_results_*.csv' -printf '%T@ %p\n' 2>/dev/null \
      | sort -rn | head -1 | cut -d' ' -f2- || true)
if [ -z "$CSV" ]; then echo "no ops_perf_results CSV found"; exit 3; fi
echo "newest ops csv: $CSV"
cp "$CSV" "$OUT/${PHASE}_ops.csv"
echo "$CSV" > "$OUT/${PHASE}_ops.csv.provenance"
date -Is >> "$OUT/${PHASE}_ops.csv.provenance"
gzip -9 -c "$OUT/${PHASE}_ops.csv" > "$OUT/${PHASE}_ops.csv.gz"

tt-perf-report "$OUT/${PHASE}_ops.csv" \
  --start-signpost "$SIGN" --end-signpost "${SIGN}_END" \
  --csv "$OUT/${PHASE}_perf_report.csv" --no-advice \
  > "$OUT/${PHASE}_perf_report.console.log" 2>&1
tt-perf-report "$OUT/${PHASE}_ops.csv" \
  --start-signpost "$SIGN" --end-signpost "${SIGN}_END" \
  --no-summary --no-advice \
  > "$OUT/${PHASE}_perf_report.txt" 2>&1
echo "== wrote $OUT/${PHASE}_perf_report.{txt,csv}"
grep -a "^PERF " "$LOG" || true
