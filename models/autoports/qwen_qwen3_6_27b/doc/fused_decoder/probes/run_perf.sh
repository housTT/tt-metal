#!/usr/bin/env bash
# Warmed prefill + traced decode profiling for one (layer kind, phase) pair of a decoder stage.
#   usage: run_perf.sh <kind> <prefill|decode> [fused|functional] [artifact_dir]
# Writes the ops CSV, the tt-perf-report table (txt + csv) and the run log under
#   doc/<stage>_decoder/tracy/<kind>/ and doc/<stage>_decoder/logs/.
set -euo pipefail
cd /home/ttuser/dev/qwen/rundir
source ./ttenv_profiler.sh

KIND="$1"
PHASE="$2"
STAGE="${3:-fused}"
# Optional 4th argument: where to write the artifacts, when it is not the stage that owns the
# code under test.  The functional baseline is re-measured into doc/fused_decoder/baseline so
# the before/after pair comes from one sitting without overwriting the earlier stage's evidence.
DEST="${4:-$REPO/models/autoports/qwen_qwen3_6_27b/doc/${STAGE}_decoder}"
ART="$DEST"
OUT="$ART/tracy/$KIND"
mkdir -p "$OUT" "$ART/logs"

case "$PHASE" in
  prefill) NODE="test_perf_prefill[$KIND]"; SIGN=PERF_PREFILL ;;
  decode)  NODE="test_perf_decode_traced[$KIND]"; SIGN=PERF_DECODE ;;
  *) echo "bad phase $PHASE"; exit 2 ;;
esac

LOG="$ART/logs/tracy_${KIND}_${PHASE}.log"
echo "== tracy run: $STAGE $KIND $PHASE -> $LOG"
python -m tracy -r -p -v -m pytest \
  "$REPO/models/autoports/qwen_qwen3_6_27b/tests/test_${STAGE}_decoder_perf.py::$NODE" \
  -s -q > "$LOG" 2>&1

CSV=$(find "$TT_METAL_HOME/generated/profiler/reports" "$PWD/generated/profiler/reports" \
        -name 'ops_perf_results_*.csv' -printf '%T@ %p\n' 2>/dev/null \
      | sort -rn | head -1 | cut -d' ' -f2- || true)
if [ -z "$CSV" ]; then echo "no ops_perf_results CSV found"; exit 3; fi
echo "newest ops csv: $CSV"
cp "$CSV" "$OUT/${PHASE}_ops.csv"
echo "$CSV" > "$OUT/${PHASE}_ops.csv.provenance"
date -Is >> "$OUT/${PHASE}_ops.csv.provenance"

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
