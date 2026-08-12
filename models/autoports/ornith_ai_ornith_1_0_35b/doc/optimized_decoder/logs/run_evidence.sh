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

# Record WHICH BYTES every artifact below was produced from, before producing any of them. mtimes alone
# cannot prove this (a touch defeats them), and review round 4 found six probe artifacts predating a
# source edit with nothing to show whether the edit was inert. `audit_figures.py` re-hashes these three
# files and fails if the shipped bytes differ from the run's.
# Normalise the sources with the repo's own pre-commit hooks BEFORE hashing them. The hooks rewrite
# whitespace, line endings and `black` formatting at commit time, so a file hashed before they run is not
# the file that gets committed — and `audit_figures.py` then correctly refuses the evidence as having been
# produced by different bytes. Doing it here makes the run idempotent: hash what will actually ship.
echo "=== 0a/9  normalise the sources with the pre-commit hooks, so the hashes are the shipped bytes ==="
pre-commit run --files "$ROOT/tt/optimized_decoder.py" "$ROOT/tests/test_optimized_decoder.py" \
  "$ROOT/tests/conftest.py" >/dev/null 2>&1 || true

echo "=== 0/9  record the source hashes the evidence is produced from ==="
{
  echo "# sha256 of the code every artifact in this directory was produced from."
  echo "# Written by logs/run_evidence.sh before the first device run; checked by audit_figures.py."
  sha256sum "$ROOT/tt/optimized_decoder.py" "$ROOT/tests/test_optimized_decoder.py" "$ROOT/tests/conftest.py"
} > "$LOGS/source_manifest.txt"

echo "=== 1/9  correctness suite (no watcher, no profiler) ==="
python -m pytest "$ROOT/tests/test_optimized_decoder.py" -v -p no:randomly > "$LOGS/pytest_full_suite.txt" 2>&1

echo "=== 2/9  before/after benchmark, fused and optimized in one process ==="
{
  python "$LOGS/bench.py" --impl fused     --layers 0,3 --phase both --iters 32 --tag before
  python "$LOGS/bench.py" --impl optimized --layers 0,3 --phase both --iters 32 --tag after --policy optimized
} 2>&1 | grep -aE "^BENCH" > "$LOGS/ab_fused_vs_optimized.txt"

echo "=== 3/9  precision policy sweep, one tensor group at a time ==="
{
  for set in "" "proj_dtype=bfloat4_b" "proj_fidelity=LoFi" "expert_fidelity=HiFi2" \
             "expert_gate_up_dtype=bfloat8_b" "shared_dtype=bfloat4_b,shared_fidelity=LoFi" \
             "expert_down_dtype=bfloat8_b" "kv_cache_dtype=bfloat16" "expert_act_dtype=bfloat16"; do
    python "$LOGS/bench.py" --impl optimized --policy optimized --set "$set" \
      --layers 3,0 --phase both --pcc --iters 32 --tag policy-sweep
  done
} 2>&1 | grep -aE "^BENCH" > "$LOGS/ab_precision_policy.txt"

echo "=== 4/9  op-level candidate sweeps ==="
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
  echo "# Explicit 2D program configs for the dense PREFILL projections, and the 'place input 0 in L1'"
  echo "# advice that both committed prefill reports still raise on three rows."
  echo "# in0=DRAM and in0=L1 are SEPARATE passes; the L1 copy exists only during its own pass, so the"
  echo "# DRAM rows are measured under the L1 state the shipped graph actually has (review round 4)."
  python "$LOGS/probe_prefill_matmul.py"
} 2>&1 | grep -aE "^PREFILLMM|^#" > "$LOGS/probe_prefill_matmul.txt"

{
  echo "# Non-matmul decode micro-probes."
  python "$LOGS/probe_decode_micro.py" --section sdpa
  python "$LOGS/probe_decode_micro.py" --section all
  python "$LOGS/probe_decode_micro.py" --section state
} 2>&1 | grep -aE "^NORM|^TOPK|^GATE|^SDPA|^STATE|^SPLIT|^#" > "$LOGS/probe_decode_micro.txt"

echo "=== 4b/9  whole-layer A/B for the sharded-norm core count ==="
# Separate from the micro-probe because the op-level ladder and the layer-level result disagree: the
# conversions each shard pays scale with the shard count and cancel the op gain (review round 6).
{
  echo "# Whole-layer traced decode against OptimizedDecoder.NORM_SHARD_CORES."
  echo "# Command: python doc/optimized_decoder/logs/ab_norm_shard_cores.py"
  echo "#"
  echo "# Why this exists: the isolated NORM rows of probe_decode_micro.txt say 4 cores is fastest, and"
  echo "# review round 6 found the shipped choice of 8 defended by a monotonicity claim the artifact"
  echo "# contradicted and by an A/B that varies a different knob. Each sharded norm also pays a"
  echo "# to_memory_config in and a sharded_to_interleaved out, and those scale with the shard count, so"
  echo "# the op-level winner need not be the layer-level one. This measures the layer."
  python "$LOGS/ab_norm_shard_cores.py"
} 2>&1 | grep -aE "^NORMCORES|^#" > "$LOGS/ab_norm_shard_cores.txt"

echo "=== 4c/9  whole-layer A/B for the routed gate/up in0_block_w cap ==="
# Regenerable, unlike the other one-off ab_*.txt files: both arms are reachable at runtime by swapping the
# cap table, so no variant of the implementation has to exist to measure it (work_log section 4.15).
{
  echo "# Whole-layer A/B for the routed gate/up in0_block_w cap (work_log section 4.15)."
  echo "# Command: python doc/optimized_decoder/logs/ab_gate_up_in0_block_w.py"
  python "$LOGS/ab_gate_up_in0_block_w.py"
} 2>&1 | grep -aE "^GATEUPIBW|^#" > "$LOGS/ab_gate_up_in0_block_w.txt"

echo "=== 5/9  OPT-007: BFP4 vs BFP8 projections on the real-weight PCC ladder ==="
{
  echo "# OPT-007: BFP4 vs BFP8 dense projection weights, real-weight HF-golden PCC ladder."
  python "$LOGS/probe_projection_dtype.py"
} 2>&1 | grep -aE "^PROJDTYPE|^#" > "$LOGS/probe_projection_dtype.txt"

echo "=== 6/9  measured per-layer device footprint, both policies, full context ==="
{
  echo "# Measured per-layer device footprint; sources doc/context_contract.json's footprint_change."
  echo "# Command: python doc/optimized_decoder/logs/probe_footprint.py"
  python "$LOGS/probe_footprint.py"
} 2>&1 | grep -aE "^FOOTPRINT|^DEVICE|^#" > "$LOGS/probe_footprint.txt"

echo "=== 7/9  Tracy / tt-perf-report captures (separate runs, advice enabled) ==="
bash "$ART/tracy/run_profiling.sh"
python "$ART/tracy/perf_accounting.py" | tee "$ART/tracy/perf_accounting.txt"

echo "=== 8/9  watcher (separate run, never combined with the profiler) ==="
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

echo "=== 9/9  fill the README's generated blocks from the artifacts above ==="
python "$LOGS/make_readme.py"
python "$LOGS/make_readme.py" --check

echo "=== assert every figure quoted in any document exists in a committed artifact ==="
# Review rounds 2, 3 and 4 each closed on "the figures are re-derived" and each next round found more
# that were not - round 4 found a whole search table quoting a superseded run of its own probe. This is
# the mechanical gate for that class, ported from the fused stage. It also re-hashes the sources against
# source_manifest.txt, so a code edit after the evidence run fails here.
# The audit measuring its own strength first: how often it ACCEPTS an arbitrary value of each figure class.
# Committed, because round 5's finding was about the check rather than about any figure, and a rate that
# rises later is the signal that the matching rules were weakened. It runs BEFORE the audit, because the
# audit diffs this artifact and would otherwise report drift against a refresh the sweep had not done yet.
python "$ART/audit_figures.py" --selftest > /dev/null

python "$ART/audit_figures.py"


echo "=== prove the generators reproduce from the COMMITTED tree, not just this worktree ==="
# NOTE what this second run does and does not prove. `git archive` stamps every extracted file with the
# commit time, so audit_figures.py's freshness check (artifact newer than source) cannot fire there. This
# run proves REPRODUCTION. Freshness is proved by the in-tree run above and, independently of any
# timestamp, by the sha256 comparison against logs/source_manifest.txt, which the archive run also does.
# Review round 3's P1: `make_readme.py --check` passed locally while one of its inputs was matched by
# the repo's blanket `*.csv` ignore rule and had never been committed. Checking against
# `git archive HEAD` is what catches that class of hole. Run it after committing.
tree=$(mktemp -d)
git archive HEAD | tar -x -C "$tree"
( cd "$tree" \
  && python "$LOGS/make_readme.py" --check \
  && python "$ART/tracy/perf_accounting.py" > /dev/null \
  && python "$ART/watcher/census.py" > /dev/null \
  && python "$ART/audit_figures.py" )
rm -rf "$tree"
echo "committed-tree reproduction OK"

echo "=== done ==="
