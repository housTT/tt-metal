#!/usr/bin/env bash
# Regenerate every committed artifact of this stage, in the only order that is correct.
#
# Device-facing steps run strictly one at a time - never a watcher run and a profiler run together -
# and the two generators at the end read only committed files.
#
#   usage: regenerate_evidence.sh [tracy|suite|probes|docs|all]
#
# `tracy` is the long pole (12 profiled windows, a few minutes each). Watch it with:
#   tail -f models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/logs/regenerate.log
set -euo pipefail
REPO=/home/ttuser/dev/qwen/tt-metal
cd "$REPO"
ART="$REPO/models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder"
PROBES="$ART/probes"
LOGS="$ART/logs"
mkdir -p "$LOGS"
WHAT="${1:-all}"

run_tracy() {
  for kind in linear_attention full_attention; do
    for phase in prefill decode decode_batch32; do
      for impl in fused optimized; do
        echo "== $(date -Is) tracy $impl $kind $phase"
        "$PROBES/run_perf.sh" "$kind" "$phase" "$impl"
      done
    done
  done
}

run_probes() {
  source "$REPO/models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/ttenv.sh" > /dev/null
  python "$PROBES/probe_blockfloat_distribution.py" > "$LOGS/probe_blockfloat_distribution.log" 2>&1
  python "$PROBES/probe_matmul_policy.py" > "$LOGS/probe_matmul_policy.log" 2>&1
  python "$PROBES/probe_optimized.py" policy > "$LOGS/probe_optimized_policy.log" 2>&1
  python "$PROBES/probe_optimized.py" geometry > "$LOGS/probe_optimized_geometry.log" 2>&1
  python "$PROBES/probe_optimized.py" prefill > "$LOGS/probe_optimized_prefill.log" 2>&1
  python "$PROBES/probe_optimized.py" isolation > "$LOGS/probe_optimized_isolation.log" 2>&1
  python "$PROBES/probe_projection_packing.py" > "$LOGS/probe_projection_packing.log" 2>&1
  python "$PROBES/probe_real_weight_policy.py" > "$LOGS/probe_real_weight_policy.log" 2>&1
}

run_suite() {
  source "$REPO/models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/ttenv.sh" > /dev/null
  python -m pytest models/autoports/qwen_qwen3_6_27b/tests/test_optimized_decoder.py -v -s \
    > "$LOGS/suite_main.log" 2>&1 || true
  python -m pytest models/autoports/qwen_qwen3_6_27b/tests/test_optimized_decoder.py \
    -k test_full_advertised_context --long-context -v -s > "$LOGS/long_context.log" 2>&1 || true
  # Watcher LAST, and never together with a profiler run.
  TT_METAL_WATCHER=10 python -m pytest models/autoports/qwen_qwen3_6_27b/tests/test_optimized_decoder.py \
    -k "traced_decode or repeated_runs or decode_pcc or batched_users or linear_state or conv_state" -v -s \
    > "$LOGS/watcher_run.log" 2>&1 || true
}

run_docs() {
  source "$REPO/models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/ttenv.sh" > /dev/null
  python "$PROBES/make_perf_summary.py"
  python -m models.autoports.qwen_qwen3_6_27b.scripts.collect_evidence \
    "$LOGS/suite_main.log" "$LOGS/long_context.log" "$LOGS/watcher_run.log" \
    --out "$ART/pcc_evidence.json"
  python "$PROBES/make_doc_tables.py"
  python -m pytest models/autoports/qwen_qwen3_6_27b/tests/test_optimized_decoder_docs.py -v
}

case "$WHAT" in
  tracy) run_tracy ;;
  probes) run_probes ;;
  suite) run_suite ;;
  docs) run_docs ;;
  all) run_probes; run_tracy; run_suite; run_docs ;;
  *) echo "bad argument $WHAT"; exit 2 ;;
esac
echo "== $(date -Is) done: $WHAT"
