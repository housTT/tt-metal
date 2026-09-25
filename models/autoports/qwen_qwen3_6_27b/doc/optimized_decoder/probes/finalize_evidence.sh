#!/usr/bin/env bash
# Turn a finished campaign into committed artifacts, then say whether the documents agree with them.
#
# Everything here reads committed logs and writes committed artifacts; it opens no device, so it is safe
# to run while nothing else is scheduled.  The order matters:
#
#   1. perf_summary.json      <- the twelve tt-perf-report CSVs
#   2. pcc_evidence.json      <- the suite, long-context and watcher logs
#   3. the context contract's optimized_decoder block <- the policy, geometry and the evidence above
#   4. every <!-- GENERATED --> block in README.md and work_log.md <- all of the above plus probe logs
#   5. the document/artifact gate, which re-derives 1-4 and fails if a committed figure disagrees
#
#   bash models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/finalize_evidence.sh
set -uo pipefail
REPO=/home/ttuser/dev/qwen/tt-metal
cd "$REPO"
source "$REPO/models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/ttenv.sh" > /dev/null
ART="$REPO/models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder"
LOGS="$ART/logs"
OUT="$LOGS/finalize.log"
: > "$OUT"

step() {
  echo "== $(date -Is) $1" | tee -a "$OUT"
  shift
  "$@" >> "$OUT" 2>&1
  local rc=$?
  echo "   exit $rc" | tee -a "$OUT"
  return $rc
}

step "perf_summary.json from the committed reports" python "$ART/probes/make_perf_summary.py"
step "pcc_evidence.json from the suite logs" python -m models.autoports.qwen_qwen3_6_27b.scripts.collect_evidence \
  "$LOGS/suite_main.log" "$LOGS/long_context.log" "$LOGS/watcher_run.log" --out "$ART/pcc_evidence.json"
step "context_contract.json optimized_decoder block" python "$ART/probes/make_contract_block.py"
step "generated document blocks" python "$ART/probes/make_doc_tables.py"

echo "== $(date -Is) document/artifact gate" | tee -a "$OUT"
python -m pytest models/autoports/qwen_qwen3_6_27b/tests/test_optimized_decoder_docs.py -v >> "$OUT" 2>&1
gate=$?
echo "   exit $gate" | tee -a "$OUT"

echo
echo "==== summary ===="
grep -E "^== |^   exit" "$OUT" | tail -20
grep -E "^FAILED|passed|failed" "$OUT" | tail -6
exit "$gate"
