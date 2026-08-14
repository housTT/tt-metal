#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Regenerate every committed artifact of the Ornith-1.0-35B MULTICHIP decoder stage, in the order
# the hardware discipline requires: correctness and A/B first, then the profiler run, then the
# watcher run **last and alone** (watcher and Tracy must never share a process).
#
# Run from the tt-metal root with the 4-chip mesh idle:
#     bash models/autoports/ornith_ai_ornith_1_0_35b/doc/multichip_decoder/logs/run_evidence.sh
#
# Steps can be selected with STEPS="suite bench ab probes tracy watcher" (default: all).
set -euo pipefail

ROOT="models/autoports/ornith_ai_ornith_1_0_35b"
DOC="$ROOT/doc/multichip_decoder"
LOGS="$DOC/logs"
TEST="$ROOT/tests/test_multichip_decoder.py"
STEPS="${STEPS:-suite bench ab probes tracy watcher}"

has() { [[ " $STEPS " == *" $1 "* ]]; }

# ---------------------------------------------------------------- 1. full pytest suite
if has suite; then
  echo "=== step 1/6: full pytest suite (including the advertised-context cases) ==="
  python -m pytest "$TEST" -v -p no:randomly > "$LOGS/pytest_full_suite.txt" 2>&1 || {
    echo "!!! suite failed; see $LOGS/pytest_full_suite.txt" >&2; exit 1;
  }
  tail -1 "$LOGS/pytest_full_suite.txt"
fi

# ---------------------------------------------------------------- 2. before/after benchmark
# Three arms, three processes, same weights and same harness:
#   * the single-chip baseline on a 1x1 mesh  -> the "before" of every speedup number;
#   * the single-chip decoder replicated on the 4-chip mesh -> the replication control, which
#     separates "the mesh changed dispatch" from "the parallelisation helped";
#   * this stage on the 4-chip mesh.
if has bench; then
  echo "=== step 2/6: warmed prefill / traced decode, three arms ==="
  {
    echo "# bench.py, real Ornith-1.0-35B weights, batch 1, 2048-token warmed prefill, 32 traced decode replays"
    python "$LOGS/bench.py" --impl optimized  --mesh 1x1 --layers 0,3 --weights real --tag single-chip-baseline
    python "$LOGS/bench.py" --impl optimized  --mesh 1x4 --layers 0,3 --weights real --tag replication-control
    python "$LOGS/bench.py" --impl multichip --mesh 1x4 --layers 0,3 --weights real --tag multichip
  } 2>/dev/null | grep -E "^BENCH|^#" > "$LOGS/ab_single_vs_multichip.txt"
  cat "$LOGS/ab_single_vs_multichip.txt"
fi

# ---------------------------------------------------------------- 3. whole-layer A/B knobs
if has ab; then
  echo "=== step 3/6: whole-layer A/B for the CCL spelling and the retuned decode geometry ==="
  python "$LOGS/ab_layer_knobs.py" 2>/dev/null | grep -E "^ABLAYER|^#" > "$LOGS/ab_layer_knobs.txt"
  cat "$LOGS/ab_layer_knobs.txt"
fi

# ---------------------------------------------------------------- 4. isolated probes
if has probes; then
  echo "=== step 4/6: isolated op probes (CCL topology, dense geometry, expert parallelism) ==="
  python "$LOGS/probe_ccl.py" 2>/dev/null | grep -E "^CCL|^#" > "$LOGS/probe_ccl.txt"
  python "$LOGS/probe_dense_matmul.py" 2>/dev/null | grep -E "^DENSE|^#" > "$LOGS/probe_dense_matmul.txt"
  {
    python "$LOGS/probe_expert_parallel.py" --phase decode 2>/dev/null | grep -E "^MOEPAR|^#"
    python "$LOGS/probe_expert_parallel.py" --phase prefill 2>/dev/null | grep -E "^MOEPAR"
  } > "$LOGS/probe_expert_parallel.txt"
  wc -l "$LOGS/probe_ccl.txt" "$LOGS/probe_dense_matmul.txt" "$LOGS/probe_expert_parallel.txt"
fi

# ---------------------------------------------------------------- 5. tt-perf-report via Tracy
if has tracy; then
  echo "=== step 5/6: device profiling + tt-perf-report (NO watcher in this process) ==="
  bash "$DOC/tracy/run_profiling.sh"
fi

# ---------------------------------------------------------------- 6. watcher, last and alone
# `TT_METAL_WATCHER_APPEND=1`: watcher truncates its log on each device open and this subset opens
# the mesh once per test, so without it the committed log would hold only the last test's session.
# `TT_METAL_WATCHER_DISABLE_ETH=1` is REQUIRED here and is a hard tool limit, not a choice. With
# watcher instrumenting the ACTIVE_ETH cores, the 1D-fabric ERISC program grows to 29040 B against a
# 25600 B ACTIVE_ETH kernel config buffer on Blackhole, so *every* test in this subset fails at
# `mesh_device` setup, before any model code runs:
#     TT_FATAL: Program size (29040) too large for kernel config buffer (25600) on ACTIVE_ETH
#              (assert.hpp:104)
# That signature is reproduced in `logs/watcher_pytest_eth_enabled.txt` (30/30 errors in 15 s, zero
# tests executed). There is no environment knob that grows that buffer, so the choice is watcher
# coverage on the 110 Tensix worker cores per chip or no watcher coverage at all. The Tensix cores
# are where every op in this stage runs; the uninstrumented cores are the fabric routers, which this
# stage does not author (they are stock `ttnn` 1D-fabric kernels). Recorded as a limitation in
# README §8.
if has watcher; then
  echo "=== step 6/6: watcher run over the state-, trace- and collective-critical subset ==="
  rm -f generated/watcher/watcher.log
  TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
  python -m pytest "$TEST" -v -p no:randomly \
    -k "traced_decode or determinism or stress or collectives or zero_local_active or kv_cache_is_local or ccl_modes or ragged or batched or output_is_identical or permuted_page_table or continuation" \
    > "$LOGS/watcher_pytest.txt" 2>&1 || {
      echo "!!! watcher subset failed; see $LOGS/watcher_pytest.txt" >&2; exit 1;
    }
  tail -1 "$LOGS/watcher_pytest.txt"
  cp generated/watcher/watcher.log "$DOC/watcher/watcher_log.txt"
  python "$DOC/watcher/census.py" > /dev/null
  grep -E "fatal-class|TOTAL|dumps:" "$DOC/watcher/census_summary.txt"
fi

# ---------------------------------------------------------------- committed-size policy
# The repo's pre-commit hook rejects files over 500 KB and `.gitignore` carries a blanket `*.csv`,
# so the big text logs and every CSV are committed gzipped. Every generator here reads `foo.ext` or
# `foo.ext.gz` transparently.
for f in "$LOGS/pytest_full_suite.txt" "$LOGS/watcher_pytest.txt" \
         "$LOGS/watcher_pytest_eth_enabled.txt" "$DOC/watcher/watcher_log.txt"; do
  if [ -f "$f" ] && [ "$(stat -c%s "$f")" -gt 500000 ]; then gzip -9 -f "$f"; fi
done

echo "=== done ==="
