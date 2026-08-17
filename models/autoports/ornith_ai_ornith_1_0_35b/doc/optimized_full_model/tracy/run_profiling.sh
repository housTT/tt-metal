#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# `tt-perf-report` for the full model's **reduced profiling variant**: one real linear_attention
# layer, one real full_attention layer, real weights, real cache/page-table shapes, and the real
# terminal path (final norm, LM head, split sampling, traced decode). `$full-model` asks for this
# shape rather than the 40-layer stack, and for good reason: a decode step of the whole stack is
# ~3300 device ops, which overruns Tracy's buffers and writes multi-gigabyte dumps, while every op
# kind in the window already appears here. The all-layer numbers come from
# `logs/bench_full_model.py`, which is wall-clock and needs no profiler.
#
# Prefill and decode are captured in SEPARATE runs so each capture holds one signposted window and
# one device session. NO watcher: watcher and profiler evidence are separate runs.
#
#     bash models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_full_model/tracy/run_profiling.sh
set -euo pipefail

ROOT="models/autoports/ornith_ai_ornith_1_0_35b"
ART="$ROOT/doc/optimized_full_model/tracy"
SCRIPT="$ROOT/doc/optimized_full_model/logs/profile_reduced.py"

# --active-experts: the per-device, per-sparsity-group active expert count, unchanged from the
# decoder stage because the MoE geometry is unchanged - 4 at batch-1 decode, 41 for a 32-token
# prefill group. See doc/optimized_multichip_decoder/tracy/run_profiling.sh for the derivation.
run_one() {
  local phase="$1" sign="$2" active="$3"
  local out="$ART/$phase.capture"
  rm -rf "$out"; mkdir -p "$out" "$ART"

  echo "=== tracy: full-model reduced / $phase ==="
  python -m tracy -r -p -v --op-support-count 50000 -o "$out" "$SCRIPT" --phase "$phase" 2>&1 \
    | tee "$out/tracy_run.txt"

  local ops_csv
  ops_csv=$(find "$out" -name 'ops_perf_results_*.csv' | sort | tail -1)
  if [ -z "$ops_csv" ]; then
    echo "!!! no ops_perf_results_*.csv produced for $phase" >&2
    return 1
  fi
  cp "$ops_csv" "$ART/${phase}_ops.csv"

  tt-perf-report "$ART/${phase}_ops.csv" \
    --start-signpost "$sign" --end-signpost "${sign}_END" \
    --active-experts "$active" --no-summary \
    > "$ART/${phase}_perf_report.txt"

  tt-perf-report "$ART/${phase}_ops.csv" \
    --start-signpost "$sign" --end-signpost "${sign}_END" \
    --active-experts "$active" \
    --csv "$ART/${phase}_perf_report.csv" \
    > "$ART/${phase}_perf_report.console.txt"

  tt-perf-report "$ART/${phase}_ops.csv" \
    --start-signpost "$sign" --end-signpost "${sign}_END" \
    --active-experts "$active" --group-by op \
    > "$ART/${phase}_perf_report.summary.txt"

  grep -vE "^\s*(\||[0-9]+ \||~|\^|In file included|from |note:|  *\^~*$)" "$out/tracy_run.txt" \
    | grep -viE "generating|compil|\.cc:|\.cpp:|\.hpp:|DeviceZone" \
    > "$ART/${phase}_tracy_run.txt"
  rm -rf "$out"

  # The repo's .gitignore carries a blanket *.csv and pre-commit rejects files over 500 KB, so the
  # committed tables are gzipped and the raw multi-device op CSV is regenerated rather than kept.
  rm -f "$ART/${phase}_ops.csv"
  for f in "$ART"/${phase}*.csv; do [ -f "$f" ] && gzip -9 -f "$f"; done
  for big in "$ART/${phase}_perf_report.txt" "$ART/${phase}_perf_report.summary.txt" \
             "$ART/${phase}_tracy_run.txt"; do
    if [ -f "$big" ] && [ "$(stat -c%s "$big")" -gt 500000 ]; then gzip -9 -f "$big"; fi
  done
  echo "wrote $ART/${phase}_perf_report.*"
}

run_one decode  PERF_DECODE 4
run_one prefill PERF_PREFILL 41
