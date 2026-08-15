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
  echo "=== step 1/7: full pytest suite (including the advertised-context cases) ==="
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
  echo "=== step 2/7: warmed prefill / traced decode, three arms ==="
  {
    echo "# bench.py, real Ornith-1.0-35B weights, batch 1, 2048-token warmed prefill, 32 traced decode replays"
    python "$LOGS/bench.py" --impl optimized  --mesh 1x1 --layers 0,3 --weights real --tag single-chip-baseline
    python "$LOGS/bench.py" --impl optimized  --mesh 1x4 --layers 0,3 --weights real --tag replication-control
    python "$LOGS/bench.py" --impl multichip --mesh 1x4 --layers 0,3 --weights real --tag multichip
    # The packet-size pair, **three builds each**: the layer-level difference is smaller than the
    # build-to-build spread of a single prefill measurement, so one build per arm cannot resolve it
    # and review round 6 was right to say so. Three builds each makes the decode difference (which is
    # repeatable) and the prefill non-difference (which is not) both visible in the artifact.
    for build in 1 2 3; do
      python "$LOGS/bench.py" --impl multichip --mesh 1x4 --layers 0,3 --weights real \
        --tag "multichip-packet-8192-build$build"
      python "$LOGS/bench.py" --impl multichip --mesh 1x4 --layers 0,3 --weights real \
        --packet-bytes 0 --tag "multichip-build-default-packet-build$build"
    done
  } 2>/dev/null | grep -E "^BENCH|^#" > "$LOGS/ab_single_vs_multichip.txt"
  cat "$LOGS/ab_single_vs_multichip.txt"
fi

# ---------------------------------------------------------------- 3. whole-layer A/B knobs
if has ab; then
  echo "=== step 3/7: whole-layer A/B: CCL spelling, decode geometry, routing narrowing, sparse cores ==="
  python "$LOGS/ab_layer_knobs.py" 2>/dev/null | grep -E "^ABLAYER|^#" > "$LOGS/ab_layer_knobs.txt"
  cat "$LOGS/ab_layer_knobs.txt"
fi

# ---------------------------------------------------------------- 4. isolated probes
if has probes; then
  echo "=== step 4/7: isolated op probes (CCL, dense + sparse geometry, EP, footprint, decode batch) ==="
  # Five processes: `set_fabric_config` is a before-open_mesh_device setting, so neither the line
  # fabric nor a packet size can be an arm inside the ring-fabric run; line-fabric rows carry the
  # `CCLFAB` tag and an extra fabric column. The layer's two collectives carry different dtypes
  # (bf16 from the token mixer, bfloat8_b from the MoE) whose ideal packet sizes differ, and review
  # round 6 found the packet decision made on bf16 rows alone. Runtime warnings are censused from the
  # suite and watcher logs in step 7 (`warning_census.py`), which is where the runtime prints them --
  # rounds 5 and 6 each turned on a warning that no artifact carried.
  {
    python "$LOGS/probe_ccl.py" 2>/dev/null | grep -E "^CCL |^#"
    python "$LOGS/probe_ccl.py" --fabric line 2>/dev/null | grep -E "^CCLFAB|^#"
    python "$LOGS/probe_ccl.py" --packet-bytes 0 2>/dev/null | grep -E "^CCLPKT|^#"
    python "$LOGS/probe_ccl.py" --dtype bfloat8_b 2>/dev/null | grep -E "^CCLBF8 |^#"
    python "$LOGS/probe_ccl.py" --dtype bfloat8_b --packet-bytes 0 2>/dev/null \
      | grep -E "^CCLBF8PKT|^#"
  } > "$LOGS/probe_ccl.txt"
  python "$LOGS/probe_dense_matmul.py" 2>/dev/null | grep -E "^DENSE|^#" > "$LOGS/probe_dense_matmul.txt"
  {
    python "$LOGS/probe_expert_parallel.py" --phase decode 2>/dev/null | grep -E "^MOEPAR|^#"
    python "$LOGS/probe_expert_parallel.py" --phase prefill 2>/dev/null | grep -E "^MOEPAR"
  } > "$LOGS/probe_expert_parallel.txt"
  python "$LOGS/probe_footprint_local.py" 2>/dev/null | grep -E "^LOCALFOOTPRINT|^DEVICE|^#" \
    > "$LOGS/probe_footprint_local.txt"
  # The routed sparse-matmul ladder at the two per-device operating points EP creates. Two processes:
  # the sweep builds a few hundred kernels per run and a single process exhausts L1_SMALL.
  # 4 is the measured batch-1 decode count and 41 the corrected prefill-group expectation; 8/16/32
  # fill in the decode range that `_active_expert_bound` walks as batch grows, and 63 is kept so the
  # rows the earlier (over-counted) calibration used are still in the artifact.
  : > "$LOGS/probe_sparse_matmul_local.txt"
  for a in 4 8 16 32 41 63; do
    python "$LOGS/probe_sparse_matmul_local.py" --experts 64 --active "$a" 2>/dev/null \
      | grep -E "^SPARSEL|^#" >> "$LOGS/probe_sparse_matmul_local.txt"
  done
  # The fused matmul+CCL family, one process per shape under `timeout`: `all_gather_matmul_async`
  # hung the mesh once (doc/multichip_decoder/triage/), and although that arm is off by default,
  # process isolation means a fused-CCL hang cannot take the rest of the sweep with it.
  : > "$LOGS/.fused.tmp"
  for sh in decode decode_b32 prefill_2048; do
    timeout 900 python "$LOGS/probe_fused_ccl.py" --shapes "$sh" 2>/dev/null \
      | grep -E "^FUSED|^#" >> "$LOGS/.fused.tmp" || echo "# shape $sh did not complete" >> "$LOGS/.fused.tmp"
  done
  mv "$LOGS/.fused.tmp" "$LOGS/probe_fused_ccl.txt"
  # One process per batch, for the same reason: this probe builds ~7 decoders per batch and the CCL
  # semaphores they allocate out of L1_SMALL are not reclaimed while the mesh stays open.
  : > "$LOGS/.decode_batch.tmp"
  for b in 1 2 4 8 13 16 32; do
    python "$LOGS/probe_decode_batch.py" --batches "$b" 2>/dev/null \
      | grep -E "^SHAPE|^DECODEB|^SPARSEB|^#" >> "$LOGS/.decode_batch.tmp"
  done
  grep -E "^#" "$LOGS/.decode_batch.tmp" | head -4 > "$LOGS/probe_decode_batch.txt"
  grep -E "^SHAPE|^DECODEB|^SPARSEB" "$LOGS/.decode_batch.tmp" >> "$LOGS/probe_decode_batch.txt"
  rm -f "$LOGS/.decode_batch.tmp"
  wc -l "$LOGS/probe_ccl.txt" "$LOGS/probe_dense_matmul.txt" "$LOGS/probe_expert_parallel.txt" \
        "$LOGS/probe_footprint_local.txt" "$LOGS/probe_sparse_matmul_local.txt" \
        "$LOGS/probe_decode_batch.txt" "$LOGS/probe_fused_ccl.txt"
fi

# ---------------------------------------------------------------- 5. tt-perf-report via Tracy
if has tracy; then
  echo "=== step 5/7: device profiling + tt-perf-report (NO watcher in this process) ==="
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
# That signature is reproduced in `logs/watcher_pytest_eth_enabled.txt.gz` (every selected test errors in ~15 s, zero
# tests executed). There is no environment knob that grows that buffer, so the choice is watcher
# coverage on the 110 Tensix worker cores per chip or no watcher coverage at all. The Tensix cores
# are where every op in this stage runs; the uninstrumented cores are the fabric routers, which this
# stage does not author (they are stock `ttnn` 1D-fabric kernels). Recorded as a limitation in
# README §8.
if has watcher; then
  echo "=== step 6/7: watcher run over the state-, trace- and collective-critical subset ==="
  rm -f generated/watcher/watcher.log
  TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
  python -m pytest "$TEST" -v -p no:randomly \
    -k "traced_decode or traced_replay or determinism or stress or collectives or zero_local_active or kv_cache_is_local or ccl_modes or ragged or batched or output_is_identical or permuted_page_table or continuation" \
    > "$LOGS/watcher_pytest.txt" 2>&1 || {
      echo "!!! watcher subset failed; see $LOGS/watcher_pytest.txt" >&2; exit 1;
    }
  tail -1 "$LOGS/watcher_pytest.txt"
  cp generated/watcher/watcher.log "$DOC/watcher/watcher_log.txt"
  # The control for the paragraph above, re-run from the committed bytes every sweep: the same
  # subset with ACTIVE_ETH instrumentation left on. Every test is expected to error at setup, so a
  # non-zero exit here is the result, not a failure of the sweep. Review round 3 noted that this
  # artifact used to predate the sweep it sat next to.
  TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 \
  python -m pytest "$TEST" -v -p no:randomly \
    -k "traced_decode or traced_replay or determinism or stress or collectives or zero_local_active or kv_cache_is_local or ccl_modes or ragged or batched or output_is_identical or permuted_page_table or continuation" \
    > "$LOGS/watcher_pytest_eth_enabled.txt" 2>&1 || true
  tail -1 "$LOGS/watcher_pytest_eth_enabled.txt"
  rm -f generated/watcher/watcher.log
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

# ---------------------------------------------------------------- 7. tables, then the figure audit
# Every table in README.md and work_log.md is generated from the artifacts this sweep just wrote, so
# a re-measurement cannot leave a stale table behind; `make_tables.py` reports which ones moved.
echo "=== step 7/7: regenerate tables, then audit every quoted figure ==="
python "$LOGS/make_tables.py"
# One line per distinct runtime warning class in the committed logs, with counts. Two review rounds in
# a row turned on a warning that was sitting unread in a log, so the classes are an artifact now.
python "$LOGS/warning_census.py"
# The stamp certifies "these artifacts measured this code", so it is only honest after a run that
# regenerated *all* of them. A partial run (STEPS="suite") leaves the rest untouched, and review
# round 4 pointed out that stamping there would certify artifacts nothing had re-measured.
FULL_SWEEP=1
for step in suite bench ab probes tracy watcher; do has "$step" || FULL_SWEEP=0; done
# Every measured figure the documents quote must exist in a committed artifact. Ported in review
# round 3, which asked for it after rounds 1, 2 and 3 each found quoted-figure errors that no
# hard check could see. It runs last because it checks the artifacts the steps above just wrote.
python "$DOC/audit_figures.py" --selftest
# The behavioural fingerprint of every source these artifacts measure. `check_freshness` compares it
# instead of mtimes when it exists, so a later documentation-only edit to the decoder or the suite
# does not read as a stale sweep -- and a change to the *code* still does, decided by hashing the
# comment- and docstring-stripped AST rather than by whoever is holding the pen.
if [ "$FULL_SWEEP" = 1 ]; then
  python "$DOC/audit_figures.py" --stamp
else
  echo "partial sweep (STEPS=\"$STEPS\"): source stamp NOT refreshed, freshness stays on mtimes" >&2
fi
python "$DOC/audit_figures.py"

echo "=== done ==="
