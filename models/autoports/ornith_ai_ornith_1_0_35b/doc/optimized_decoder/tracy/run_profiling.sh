#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Warmed prefill / warmed traced decode profiling for the Ornith-1.0-35B OPTIMIZED decoder.
# Prefill and decode are captured in SEPARATE runs, one per layer kind, so each Tracy capture holds
# exactly one signposted window and one device session.
#
# Unlike the fused stage's script this one keeps `tt-perf-report`'s advice ENABLED in the committed
# table — the optimize skill requires the advice-backed report to be the one the stage is guided by
# and reports from. A compact no-advice table is written alongside it under a distinct name.
#
# Tests are selected by full node id, not by `-k "a and b"`: `python -m tracy -m pytest` re-splits
# its argv, so a quoted -k expression containing spaces reaches pytest as separate arguments and
# fails with `file or directory not found: and`.
#
# Run from the tt-metal root, with the device idle and NO watcher enabled (watcher and profiler
# evidence are separate runs):
#     bash models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_decoder/tracy/run_profiling.sh
set -euo pipefail

ROOT="models/autoports/ornith_ai_ornith_1_0_35b"
ART="$ROOT/doc/optimized_decoder/tracy"
TEST="$ROOT/tests/test_optimized_decoder.py"

# `--active-experts`: `tt-perf-report` cannot model a `sparse_matmul` row whose `nnz` is `std::nullopt`, and
# `generate_matmul_advice` EARLY-RETURNS on such a row - so without this flag the two routed matmuls, 31 % of the
# decode window and ~82 % of prefill, carry no Bound, no DRAM %, no FLOPs % and **no advice at all**, while the
# report prints a warning saying so. Review round 14 found every committed report in that state and one advice
# item ("place input 0 in L1", on the largest op of the prefill window) consequently untried. The count is the
# active experts per input batch group: exactly 8 for a batch-1 decode step (one token, top-8), and 162 for a
# 32-token prefill group - the expected distinct union of 256 draws from 256 experts, 256*(1-(1-1/256)^256),
# which is the same figure `probe_sparse_matmul.py` tunes at.
run_one() {
  local kind="$1" phase="$2" node="$3" sign="$4" active="$5"
  local out="$ART/$kind/$phase"
  rm -rf "$out"
  mkdir -p "$out" "$ART/$kind"

  echo "=== tracy: $kind / $phase ==="
  # --op-support-count: the profiler's default program budget is 1000 (tools/tracy/common.py);
  # a warmed 2048-token prefill plus its two warmup passes, and 32 traced decode replays, both
  # exceed it and post-processing then asserts "Device data missing".
  python -m tracy -r -p -v --op-support-count 50000 -o "$out" -m pytest "$TEST::$node" 2>&1 \
    | tee "$out/tracy_run.txt"

  local ops_csv
  ops_csv=$(find "$out" -name 'ops_perf_results_*.csv' | sort | tail -1)
  if [ -z "$ops_csv" ]; then
    echo "!!! no ops_perf_results_*.csv produced for $kind/$phase" >&2
    return 1
  fi
  cp "$ops_csv" "$ART/$kind/${phase}_ops.csv"

  # Advice-enabled human-readable table: this is the report the stage is guided by.
  tt-perf-report "$ART/$kind/${phase}_ops.csv" \
    --start-signpost "$sign" --end-signpost "${sign}_END" \
    --active-experts "$active" \
    --no-summary \
    > "$ART/$kind/${phase}_perf_report.txt"

  # Machine-readable rows for the same window (--csv mode prints status chatter, not the table).
  tt-perf-report "$ART/$kind/${phase}_ops.csv" \
    --start-signpost "$sign" --end-signpost "${sign}_END" \
    --active-experts "$active" \
    --csv "$ART/$kind/${phase}_perf_report.csv" \
    > "$ART/$kind/${phase}_perf_report.console.txt"

  # Roofline summary + per-op-code stack, advice enabled.
  tt-perf-report "$ART/$kind/${phase}_ops.csv" \
    --start-signpost "$sign" --end-signpost "${sign}_END" \
    --active-experts "$active" \
    --group-by op \
    > "$ART/$kind/${phase}_perf_report.summary.txt"

  grep -vE "^\s*(\||[0-9]+ \||~|\^|In file included|from |note:|  *\^~*$)" "$out/tracy_run.txt" \
    | grep -viE "generating|compil|\.cc:|\.cpp:|\.hpp:|DeviceZone" \
    > "$ART/$kind/${phase}_tracy_run.txt"
  rm -rf "$out"
  # The raw ops dump is 620-720 KB gzipped for a decode window, over this repo's 500 KB file limit
  # and on `$optimize`'s list of artifacts not worth copying back. Keep it for prefill (small) and
  # drop it for decode; the per-op `_perf_report.csv` is what every table and the accounting read.
  gzip -f "$ART/$kind/${phase}_ops.csv"
  if [ "$phase" = "decode" ]; then rm -f "$ART/$kind/${phase}_ops.csv.gz"; fi
  # Every CSV is gzipped UNCONDITIONALLY, not just the ones over the 500 KB hook limit: the repo's
  # .gitignore has a blanket `*.csv`, so a report CSV left uncompressed is silently not committed and
  # the generators that read it stop reproducing from a clean checkout. Review round 3 found exactly
  # that on `full_attention/decode_perf_report.csv` (477 600 B - under the size threshold, so it was
  # never gzipped, so it was never committed). `.csv.gz` is not matched by the ignore rule.
  for f in "$ART/$kind"/*.csv; do
    [ -f "$f" ] && gzip -9 -f "$f"
  done
  # Same 500 KB limit for the human-readable tables; every generator here reads `foo.txt` or
  # `foo.txt.gz`.
  for big in "$ART/$kind/${phase}_perf_report.txt" "$ART/$kind/${phase}_perf_report.summary.txt"; do
    if [ -f "$big" ] && [ "$(stat -c%s "$big")" -gt 500000 ]; then gzip -9 -f "$big"; fi
  done

  echo "wrote $ART/$kind/${phase}_perf_report.* (gzipped above 500 KB) for $kind/$phase"
}

run_one linear_attention prefill \
  'test_perf_prefill[blackhole-2048-linear_attention-mesh_device0-device_params0]' PERF_PREFILL 162
run_one linear_attention decode \
  'test_perf_decode_traced[blackhole-linear_attention-mesh_device0-device_params0]' PERF_DECODE 8
run_one full_attention prefill \
  'test_perf_prefill[blackhole-2048-full_attention-mesh_device0-device_params0]' PERF_PREFILL 162
run_one full_attention decode \
  'test_perf_decode_traced[blackhole-full_attention-mesh_device0-device_params0]' PERF_DECODE 8
