#!/usr/bin/env bash
# Warmed prefill + traced decode profiling for one (layer kind, phase) pair.
#   usage: run_perf.sh <linear_attention|full_attention> <prefill|decode>
set -euo pipefail
cd /home/ttuser/dev/qwen/rundir
source ./ttenv_profiler.sh

KIND="$1"
PHASE="$2"
ART="$REPO/models/autoports/qwen_qwen3_6_27b/doc/functional_decoder"
OUT="$ART/tracy/$KIND"
mkdir -p "$OUT"

# `python -m tracy` re-joins the wrapped command without shell quoting, so a `-k "a and b"`
# selector arrives as three arguments and pytest fails with "file or directory not found: and".
# Address the test by node id instead; it contains no spaces.
case "$PHASE" in
  prefill) NODE="test_perf_prefill[$KIND]"; SIGN=PERF_PREFILL ;;
  decode)  NODE="test_perf_decode_traced[$KIND]"; SIGN=PERF_DECODE ;;
  *) echo "bad phase $PHASE"; exit 2 ;;
esac

LOG="$ART/logs/tracy_${KIND}_${PHASE}.log"
echo "== tracy run: $KIND $PHASE -> $LOG"
python -m tracy -r -p -v -m pytest \
  "$REPO/models/autoports/qwen_qwen3_6_27b/tests/test_functional_decoder_perf.py::$NODE" \
  -s -q > "$LOG" 2>&1

# The post-processed ops CSV lands under $TT_METAL_HOME, not the working directory.
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
