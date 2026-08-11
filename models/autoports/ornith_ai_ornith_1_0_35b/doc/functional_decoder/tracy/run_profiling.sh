#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Warmed prefill / warmed traced decode profiling for the Ornith-1.0-35B functional decoder.
# Prefill and decode are captured in SEPARATE runs, one per layer kind, so each Tracy capture
# holds exactly one signposted window and one device session.
#
# Tests are selected by full node id, not by `-k "a and b"`: `python -m tracy -m pytest` re-splits
# its argv, so a quoted -k expression containing spaces reaches pytest as separate arguments and
# fails with `file or directory not found: and`.
#
# Run from the tt-metal root, with the device idle and no watcher enabled:
#     bash models/autoports/ornith_ai_ornith_1_0_35b/doc/functional_decoder/tracy/run_profiling.sh
set -euo pipefail

ROOT="models/autoports/ornith_ai_ornith_1_0_35b"
ART="$ROOT/doc/functional_decoder/tracy"
TEST="$ROOT/tests/test_functional_decoder.py"

run_one() {
  local kind="$1" phase="$2" node="$3" sign="$4"
  local out="$ART/$kind/$phase"
  rm -rf "$out"
  mkdir -p "$out"

  echo "=== tracy: $kind / $phase ==="
  # --op-support-count: the profiler's default program budget is 1000 (tools/tracy/common.py);
  # a warmed 2048-token prefill plus its two warmup passes, and 32 traced decode replays, both
  # exceed it and the post-processing then asserts "Device data missing: Op ... not present in
  # cpp_device_perf_report.csv".
  python -m tracy -r -p -v --op-support-count 50000 -o "$out" -m pytest "$TEST::$node" 2>&1 \
    | tee "$out/tracy_run.txt"

  local ops_csv
  ops_csv=$(find "$out" -name 'ops_perf_results_*.csv' | sort | tail -1)
  if [ -z "$ops_csv" ]; then
    echo "!!! no ops_perf_results_*.csv produced for $kind/$phase" >&2
    return 1
  fi
  cp "$ops_csv" "$ART/$kind/${phase}_ops.csv"

  tt-perf-report "$ART/$kind/${phase}_ops.csv" \
    --start-signpost "$sign" --end-signpost "${sign}_END" \
    --csv "$ART/$kind/${phase}_perf_report.csv" --no-advice \
    > "$ART/$kind/${phase}_perf_report.console.txt"

  tt-perf-report "$ART/$kind/${phase}_ops.csv" \
    --start-signpost "$sign" --end-signpost "${sign}_END" \
    --no-summary --no-advice \
    > "$ART/$kind/${phase}_perf_report.txt"

  # Keep only what is evidence: the console log (kernel-compilation spam stripped) and a gzipped
  # ops CSV. The raw capture directory is several hundred MB to a couple of GB per run.
  grep -vE "^\s*(\||[0-9]+ \||~|\^|In file included|from |note:|  *\^~*$)" "$out/tracy_run.txt" \
    | grep -viE "generating|compil|\.cc:|\.cpp:|\.hpp:|DeviceZone" \
    > "$ART/$kind/${phase}_tracy_run.txt"
  rm -rf "$out"
  gzip -f "$ART/$kind/${phase}_ops.csv"

  echo "wrote $ART/$kind/${phase}_perf_report.{txt,csv} + ${phase}_ops.csv.gz from $(basename "$ops_csv")"
}

run_one linear_attention prefill \
  'test_perf_prefill[blackhole-2048-linear_attention-mesh_device0-device_params0]' PERF_PREFILL
run_one linear_attention decode \
  'test_perf_decode_traced[blackhole-linear_attention-mesh_device0-device_params0]' PERF_DECODE
run_one full_attention prefill \
  'test_perf_prefill[blackhole-2048-full_attention-mesh_device0-device_params0]' PERF_PREFILL
run_one full_attention decode \
  'test_perf_decode_traced[blackhole-full_attention-mesh_device0-device_params0]' PERF_DECODE
