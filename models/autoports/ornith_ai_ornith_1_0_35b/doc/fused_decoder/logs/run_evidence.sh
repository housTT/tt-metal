#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Regenerates every committed piece of fused-decoder evidence, in the order the
# $tt-device-usage skill requires: one device-facing command at a time, and watcher and profiler
# runs in separate sessions that never overlap.
#
#     bash models/autoports/ornith_ai_ornith_1_0_35b/doc/fused_decoder/logs/run_evidence.sh
#
# Everything here reads the *current* source tree, so it must be re-run after any source edit —
# that is the whole point of having it as one script.
set -euo pipefail

ROOT="models/autoports/ornith_ai_ornith_1_0_35b"
L="$ROOT/doc/fused_decoder/logs"
W="$ROOT/doc/fused_decoder/watcher"
T="$ROOT/doc/fused_decoder/tracy"
TEST="$ROOT/tests/test_fused_decoder.py"

echo "=== 0/6 source manifest (what every artifact below measures) ==="
sha256sum "$ROOT/tt/fused_decoder.py" "$ROOT/tests/test_fused_decoder.py" "$ROOT/tests/conftest.py" \
  | tee "$L/source_manifest.txt"

echo "=== 1/6 full test suite (no watcher, no profiler) ==="
pytest "$TEST" -v -p no:randomly > "$L/pytest_full_suite.txt" 2>&1
python "$L/summarise_pcc.py" "$L/pytest_full_suite.txt"
# Account for the critical-level lines the suite's own conv probing prints. The document generators
# do NOT run here: make_readme_tables.py also owns README §7's watcher figures, so it has to run
# after stage 2, not before it. All three run together at 5/5 once every input exists.
python "$L/classify_suite_criticals.py"

echo "=== 2/6 watcher run (separate session; TT_METAL_WATCHER_APPEND keeps every device open) ==="
rm -f generated/watcher/watcher.log
TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 \
  pytest "$TEST" -v -p no:randomly \
  -k "(decode_pcc and 130) or traced_decode or determinism or permuted or poisoned or ragged \
or unaligned_max_context or repeated_run_stress or fused_path_is_used or layout_churn \
or rope_mode_equivalence or masked_chunk_length or batched_paged_fill or above_head_split_limit \
or prefill_continuation or batch_smaller_than_allocated_state or batched_prefill_decode \
or lazy_allocation or fused_matches_functional" \
  > "$L/watcher_pytest.txt" 2>&1
cp generated/watcher/watcher.log "$W/watcher_log.txt"
# watcher pads the index ("k_id[  0]: blank"), so the bracket contents must allow spaces.
grep -oE "k_id\[ *[0-9]+\]: .*" "$W/watcher_log.txt" | sed -E "s/^k_id\[ *([0-9]+)\]/k_id[\\1]/" \
  | sort -u > "$W/kernel_names.txt"
test -s "$W/kernel_names.txt" || { echo "!!! kernel_names.txt is empty - the k_id grep matched nothing" >&2; exit 1; }
# census.py writes watcher/census_summary.txt itself; redirecting stdout over the same path as well
# only worked because the two were byte-identical.
python "$W/census.py" > /dev/null

echo "=== 3/6 tracy captures (separate session from watcher) ==="
bash "$T/run_profiling.sh"
python "$T/summarise_perf.py"       > "$T/perf_summary.txt"
python "$T/summarise_slow_ops.py"   > "$T/slow_ops_summary.txt"
# Splits the UnaryDeviceOperation aggregate into its FILL and non-FILL parts from the raw capture;
# writes tracy/fill_summary.txt itself.
python "$T/summarise_fill.py" > /dev/null

echo "=== 4/6 A/B sweeps ==="
bash "$L/run_sweeps.sh"

echo "=== 5/6 regenerate every table in the documents ==="
# Every measured figure in README.md and work_log.md is spliced in from one of these three, so they
# run after the last measurement and before the audit. Ordering matters only in that all of them
# must follow all of the measurements: make_readme_tables.py reads the suite log *and* the watcher
# census, make_readme_perf.py the tracy captures *and* the A/B bench.
python "$L/make_readme_tables.py"  --write
python "$L/make_readme_perf.py"    --write
python "$L/make_worklog_tables.py" --write

echo "=== 6/6 figure audit (re-checks all three generators) ==="
python "$ROOT/doc/fused_decoder/audit_figures.py"
python "$ROOT/doc/functional_decoder/audit_figures.py"
