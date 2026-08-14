#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Warmed prefill / warmed traced decode profiling for the Ornith-1.0-35B MULTICHIP decoder, on the
# 4-chip Blackhole ring. Prefill and decode are captured in SEPARATE runs, one per layer kind, so
# each Tracy capture holds exactly one signposted window and one device session.
#
# `tt-perf-report` merges the four devices' rows; the per-op tables below are therefore the mesh's
# view, which is what a decode step's latency actually is (every device runs the same program and
# the collectives synchronise them).
#
# Run from the tt-metal root, with the device idle and NO watcher enabled (watcher and profiler
# evidence are separate runs):
#     bash models/autoports/ornith_ai_ornith_1_0_35b/doc/multichip_decoder/tracy/run_profiling.sh
set -euo pipefail

ROOT="models/autoports/ornith_ai_ornith_1_0_35b"
ART="$ROOT/doc/multichip_decoder/tracy"
TEST="$ROOT/tests/test_multichip_decoder.py"

# `--active-experts`: tt-perf-report cannot model a `sparse_matmul` row whose `nnz` is
# `std::nullopt` and early-returns out of its advice for such a row, so without this flag the two
# routed matmuls carry no Bound, no DRAM %, no FLOPs % and no advice. The count is the active
# experts **per device**, which is what changes under expert parallelism: at batch-1 decode the
# expected maximum over the four devices is 3.54 of the global 8 (probe_expert_parallel.txt), and
# for a 32-token prefill group it is the expected distinct union of 256 draws over 64 local experts,
# 64*(1-(1-1/64)^256) = 63.
run_one() {
  local kind="$1" phase="$2" node="$3" sign="$4" active="$5"
  local out="$ART/$kind/$phase"
  rm -rf "$out"
  mkdir -p "$out" "$ART/$kind"

  echo "=== tracy: $kind / $phase ==="
  python -m tracy -r -p -v --op-support-count 50000 -o "$out" -m pytest "$TEST::$node" 2>&1 \
    | tee "$out/tracy_run.txt"

  local ops_csv
  ops_csv=$(find "$out" -name 'ops_perf_results_*.csv' | sort | tail -1)
  if [ -z "$ops_csv" ]; then
    echo "!!! no ops_perf_results_*.csv produced for $kind/$phase" >&2
    return 1
  fi
  cp "$ops_csv" "$ART/$kind/${phase}_ops.csv"

  # Advice-enabled human-readable table: the report this stage is guided by.
  tt-perf-report "$ART/$kind/${phase}_ops.csv" \
    --start-signpost "$sign" --end-signpost "${sign}_END" \
    --active-experts "$active" \
    --no-summary \
    > "$ART/$kind/${phase}_perf_report.txt"

  # Machine-readable rows for the same window.
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

  # `.gitignore` carries a blanket `*.csv`, so every CSV here is gzipped unconditionally — a plain
  # `.csv` is silently not committed at all. Same for text tables over the 500 KB pre-commit limit.
  #
  # The RAW tt-metal op CSV is regenerated locally by this script but is NOT kept as a committed
  # artifact, for either phase. This capture merges four devices, so the raw file is ~10 000 rows x
  # ~420 columns: 1.42 MB gzipped for full_attention prefill and 1.52 MB for linear_attention,
  # against the repo's hard 500 KB pre-commit limit (`scripts/check_file_size.py`). Trimming it to
  # the signposted window fits (454 KB / 468 KB) but does **not** reproduce the committed report
  # byte for byte: `tt-perf-report` merges the four devices' rows and its choice of representative
  # device for an op is sensitive to what precedes the window, so a trimmed input silently relabels
  # the Device column and shifts the first Op-to-Op Gap. A provenance file that does not reproduce
  # the report is worse than none. What IS committed for every capture is the full machine-readable
  # per-op table for the exact window (`${phase}_perf_report.csv.gz`, written by `tt-perf-report`
  # from the raw CSV), the stacked-by-op-code CSV, the two human tables, and the capture's own
  # console log — plus this script, which is the exact command.
  rm -f "$ART/$kind/${phase}_ops.csv"
  for f in "$ART/$kind"/*.csv; do
    [ -f "$f" ] && gzip -9 -f "$f"
  done
  for big in "$ART/$kind/${phase}_perf_report.txt" "$ART/$kind/${phase}_perf_report.summary.txt" \
             "$ART/$kind/${phase}_tracy_run.txt"; do
    if [ -f "$big" ] && [ "$(stat -c%s "$big")" -gt 500000 ]; then gzip -9 -f "$big"; fi
  done

  echo "wrote $ART/$kind/${phase}_perf_report.* for $kind/$phase"
}

run_one linear_attention prefill \
  'test_perf_prefill[blackhole-2048-linear_attention-mesh_device0-device_params0]' PERF_PREFILL 63
run_one linear_attention decode \
  'test_perf_decode_traced[blackhole-linear_attention-mesh_device0-device_params0]' PERF_DECODE 4
run_one full_attention prefill \
  'test_perf_prefill[blackhole-2048-full_attention-mesh_device0-device_params0]' PERF_PREFILL 63
run_one full_attention decode \
  'test_perf_decode_traced[blackhole-full_attention-mesh_device0-device_params0]' PERF_DECODE 4
