#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Regenerate every committed artifact of the optimized-decoder stage, in order, from a clean tree.
# Run from the tt-metal root on an idle 1x1 Blackhole device:
#
#     bash models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_decoder/logs/run_evidence.sh
#
# The phases are deliberately separate device runs. Watcher and the device profiler are never
# combined, and no phase overlaps another: every command below opens and closes the device on its
# own, which is what `$tt-device-usage` asks for.
#
# NOT run here, on purpose: `probe_sparse_matmul.py --nnz`. That arm wedged the device (work_log
# §4.8) and needs a `tt-smi -r` afterwards, so it is opt-in only.
set -euo pipefail

ROOT="models/autoports/ornith_ai_ornith_1_0_35b"
ART="$ROOT/doc/optimized_decoder"
LOGS="$ART/logs"

echo "=== 1/7  correctness suite (no watcher, no profiler) ==="
python -m pytest "$ROOT/tests/test_optimized_decoder.py" -v -p no:randomly > "$LOGS/pytest_full_suite.txt" 2>&1

echo "=== 2/7  before/after benchmark, fused and optimized in one process ==="
{
  python "$LOGS/bench.py" --impl fused     --layers 0,3 --phase both --iters 32 --tag before
  python "$LOGS/bench.py" --impl optimized --layers 0,3 --phase both --iters 32 --tag after --policy optimized
} 2>&1 | grep -aE "^BENCH" > "$LOGS/ab_fused_vs_optimized.txt"

echo "=== 3/7  precision policy sweep, one tensor group at a time ==="
{
  for set in "" "proj_dtype=bfloat4_b" "proj_fidelity=LoFi" "expert_fidelity=HiFi2" \
             "expert_gate_up_dtype=bfloat8_b" "shared_dtype=bfloat4_b,shared_fidelity=LoFi" \
             "expert_down_dtype=bfloat8_b" "kv_cache_dtype=bfloat16" "expert_act_dtype=bfloat16"; do
    python "$LOGS/bench.py" --impl optimized --policy optimized --set "$set" \
      --layers 3,0 --phase both --pcc --iters 32 --tag policy-sweep
  done
} 2>&1 | grep -aE "^BENCH" > "$LOGS/ab_precision_policy.txt"

echo "=== 4/7  op-level candidate sweeps ==="
{
  echo "# ttnn.sparse_matmul geometry sweep for the two routed-expert projections."
  for active in 8 32 64 162; do
    python "$LOGS/probe_sparse_matmul.py" --weight-dtype bfloat4_b --act-dtype bfloat8_b \
      --fidelity LoFi --iters 8 --active "$active"
  done
} 2>&1 | grep -aE "^SPARSE|^#" > "$LOGS/probe_sparse_matmul.txt"

{
  echo "# Dense decode matmul family sweep."
  python "$LOGS/probe_dense_matmul.py" --weight-dtype bfloat8_b --fidelity HiFi2 --iters 15
  python "$LOGS/probe_dense_matmul.py" --weight-dtype bfloat8_b --fidelity HiFi2 --iters 15 \
    --families dram_sharded
  python "$LOGS/probe_dense_matmul.py" --weight-dtype bfloat8_b --fidelity HiFi2 --iters 15 \
    --families mcast1d --in0-blocks 8,16,64,128
  python "$LOGS/probe_dense_matmul.py" --weight-dtype bfloat16 --fidelity HiFi4 --fp32-acc \
    --iters 15 --roles router --in0-blocks 16,32,64
} 2>&1 | grep -aE "^DENSE|^#" > "$LOGS/probe_dense_matmul.txt"

{
  echo "# Explicit 2D program configs for the dense PREFILL projections vs ttnn's heuristic."
  python "$LOGS/probe_prefill_matmul.py"
} 2>&1 | grep -aE "^PREFILLMM|^#" > "$LOGS/probe_prefill_matmul.txt"

{
  echo "# Non-matmul decode micro-probes."
  python "$LOGS/probe_decode_micro.py" --section sdpa
  python "$LOGS/probe_decode_micro.py" --section all
  python "$LOGS/probe_decode_micro.py" --section state
} 2>&1 | grep -aE "^NORM|^TOPK|^GATE|^SDPA|^STATE|^SPLIT|^#" > "$LOGS/probe_decode_micro.txt"

echo "=== 5/7  OPT-007: BFP4 vs BFP8 projections on the real-weight PCC ladder ==="
{
  echo "# OPT-007: BFP4 vs BFP8 dense projection weights, real-weight HF-golden PCC ladder."
  python "$LOGS/probe_projection_dtype.py"
} 2>&1 | grep -aE "^PROJDTYPE|^#" > "$LOGS/probe_projection_dtype.txt"

echo "=== 6/7  Tracy / tt-perf-report captures (separate runs, advice enabled) ==="
bash "$ART/tracy/run_profiling.sh"
python "$ART/tracy/perf_accounting.py" | tee "$ART/tracy/perf_accounting.txt"

echo "=== 7/7  watcher (separate run, never combined with the profiler) ==="
rm -f generated/watcher/watcher.log
TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 \
python -m pytest "$ROOT/tests/test_optimized_decoder.py" -v -p no:randomly \
  -k "(decode_pcc and 130) or traced_decode or determinism or permuted or poisoned or ragged \
or unaligned_max_context or repeated_run_stress or optimized_path_is_used or layout_churn \
or rope_mode_equivalence or masked_chunk_length or batched_paged_fill or above_head_split_limit \
or prefill_continuation or batch_smaller_than_allocated_state or batched_prefill_decode \
or lazy_allocation or optimized_matches_fused or padded_rows or tuned_program_configs \
or precision_policy" \
  > "$LOGS/watcher_pytest.txt" 2>&1
cp generated/watcher/watcher.log "$ART/watcher/watcher_log.txt"
python "$ART/watcher/census.py" > "$ART/watcher/census_summary.txt"

echo "=== gzip the logs this repo's 500 KB file limit rejects ==="
for big in "$LOGS/pytest_full_suite.txt" "$LOGS/watcher_pytest.txt" "$ART/watcher/watcher_log.txt"; do
  if [ -f "$big" ] && [ "$(stat -c%s "$big")" -gt 500000 ]; then gzip -9 -f "$big"; fi
done

echo "=== fill the README's generated blocks from the artifacts above ==="
python "$LOGS/make_readme.py"
python "$LOGS/make_readme.py" --check

echo "=== prove the generators reproduce from the COMMITTED tree, not just this worktree ==="
# Review round 3's P1: `make_readme.py --check` passed locally while one of its inputs was matched by
# the repo's blanket `*.csv` ignore rule and had never been committed. Checking against
# `git archive HEAD` is what catches that class of hole. Run it after committing.
tree=$(mktemp -d)
git archive HEAD | tar -x -C "$tree"
( cd "$tree" \
  && python "$LOGS/make_readme.py" --check \
  && python "$ART/tracy/perf_accounting.py" > /dev/null \
  && python "$ART/watcher/census.py" > /dev/null )
rm -rf "$tree"
echo "committed-tree reproduction OK"

echo "=== done ==="
