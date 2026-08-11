#!/usr/bin/env bash
# Warmed prefill + traced decode profiling for one (layer kind, phase, configuration) tuple.
#   usage: run_perf.sh <linear_attention|full_attention> <prefill|decode|decode_batch32> \
#                      [impl] [policy] [geometry]
#
# ``decode_batch32`` is the same traced-decode window at the advertised max_batch, which is a
# different graph (the group-reduction gated norm, and every recurrence op 32x wider), so it gets
# its own artifacts.
#
# ``impl`` is one of fused / optimized / functional.  For ``optimized`` the policy and geometry name
# a configuration from tests/test_optimized_decoder_perf.py POLICIES / GEOMETRIES, so the whole
# before/after matrix - including "the same code at the old precision" - is produced by one script
# with different arguments, on the same machine, against the same build.
#
# This checkout is built in-tree with ENABLE_TRACY=ON, so the ordinary stage environment is also the
# profiling environment.  Advice is left ON in the perf report: the optimize skill requires the
# guiding report to carry it.  A compact no-advice table is written alongside it.
#
# Artifacts land under doc/optimized_decoder/tracy/<impl>[-<policy>-<geometry>]/<kind>/.
set -euo pipefail
REPO=/home/ttuser/dev/qwen/tt-metal
cd "$REPO"
source "$REPO/models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/ttenv.sh" > /dev/null

KIND="$1"
PHASE="$2"
IMPL="${3:-optimized}"
POLICY="${4:-opt-v1}"
GEOMETRY="${5:-opt-v1}"
ART="$REPO/models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder"
if [ "$IMPL" = "optimized" ] && { [ "$POLICY" != "opt-v1" ] || [ "$GEOMETRY" != "opt-v1" ]; }; then
  TAG="optimized-${POLICY}-${GEOMETRY}"
else
  TAG="$IMPL"
fi
OUT="$ART/tracy/$TAG/$KIND"
mkdir -p "$OUT" "$ART/logs"

# `python -m tracy` re-joins the wrapped command without shell quoting, so address the test by node
# id (no spaces) rather than with a -k selector.
BATCH=1
case "$PHASE" in
  prefill)        NODE="test_perf_prefill[$KIND]"; SIGN=PERF_PREFILL ;;
  decode)         NODE="test_perf_decode_traced[$KIND]"; SIGN=PERF_DECODE ;;
  decode_batch32) NODE="test_perf_decode_traced[$KIND]"; SIGN=PERF_DECODE; BATCH=32 ;;
  *) echo "bad phase $PHASE"; exit 2 ;;
esac

LOG="$ART/logs/tracy_${TAG}_${KIND}_${PHASE}.log"
echo "== tracy run: $TAG $KIND $PHASE -> $LOG"
python -m tracy -r -p -v -m pytest \
  "$REPO/models/autoports/qwen_qwen3_6_27b/tests/test_optimized_decoder_perf.py::$NODE" \
  -s -q --impl "$IMPL" --policy "$POLICY" --geometry "$GEOMETRY" --perf-batch "$BATCH" > "$LOG" 2>&1

# The post-processed ops CSV lands under $TT_METAL_HOME, not the working directory.
CSV=$(find "$TT_METAL_HOME/generated/profiler/reports" "$PWD/generated/profiler/reports" \
        -name 'ops_perf_results_*.csv' -printf '%T@ %p\n' 2>/dev/null \
      | sort -rn | head -1 | cut -d' ' -f2- || true)
if [ -z "$CSV" ]; then echo "no ops_perf_results CSV found"; exit 3; fi
echo "newest ops csv: $CSV"
cp "$CSV" "$OUT/${PHASE}_ops.csv"
echo "$CSV" > "$OUT/${PHASE}_ops.csv.provenance"
date -Is >> "$OUT/${PHASE}_ops.csv.provenance"
python -m models.autoports.qwen_qwen3_6_27b.tt.build_fingerprint >> "$OUT/${PHASE}_ops.csv.provenance"
gzip -9 -f -c "$OUT/${PHASE}_ops.csv" > "$OUT/${PHASE}_ops.csv.gz"

# Advice ON - this is the report the optimization decisions are made against.
tt-perf-report "$OUT/${PHASE}_ops.csv" \
  --start-signpost "$SIGN" --end-signpost "${SIGN}_END" \
  --csv "$OUT/${PHASE}_perf_report.csv" \
  > "$OUT/${PHASE}_perf_report.console.log" 2>&1
tt-perf-report "$OUT/${PHASE}_ops.csv" \
  --start-signpost "$SIGN" --end-signpost "${SIGN}_END" \
  --no-summary \
  > "$OUT/${PHASE}_perf_report.txt" 2>&1
# Compact table without advice, for the diff-friendly artifact.
tt-perf-report "$OUT/${PHASE}_ops.csv" \
  --start-signpost "$SIGN" --end-signpost "${SIGN}_END" \
  --no-summary --no-advice \
  > "$OUT/${PHASE}_perf_report.noadvice.txt" 2>&1
echo "== wrote $OUT/${PHASE}_perf_report.{txt,csv,noadvice.txt}"
grep -a "^PERF \|^PERFCONFIG " "$LOG" || true
