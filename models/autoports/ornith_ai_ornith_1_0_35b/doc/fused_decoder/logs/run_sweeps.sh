#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Regenerates every A/B log under doc/fused_decoder/logs/.
# Run from the tt-metal root with the device idle and no watcher or profiler enabled:
#     bash models/autoports/ornith_ai_ornith_1_0_35b/doc/fused_decoder/logs/run_sweeps.sh
set -euo pipefail

L="models/autoports/ornith_ai_ornith_1_0_35b/doc/fused_decoder/logs"
BENCH="$L/bench_ab.py"

echo "=== op-feasibility probes ==="
python "$L/probe_fused_ops.py"           > "$L/probe_fused_ops.txt"        2>&1 || true
python "$L/probe_router_and_reduce.py"   > "$L/probe_router_and_reduce.txt" 2>&1 || true
python "$L/probe_gate_up_pack.py"        > "$L/probe_gate_up_pack.txt"      2>&1 || true
python "$L/probe_conv1d_and_norm.py"     > "$L/probe_conv1d_and_norm.txt"   2>&1 || true
python "$L/probe_decode_micro.py"        > "$L/probe_decode_micro.txt"      2>&1 || true
python "$L/probe_conv_tail.py"           > "$L/probe_conv_tail.txt"         2>&1 || true

echo "=== functional-vs-fused A/B (the headline before/after) ==="
python "$BENCH" --impl both              > "$L/ab_functional_vs_fused.txt"  2>&1

{
  echo "# MoE expert-group granularity sweep (fused decoder, 2048-token warmed prefill)"
  echo "# One sparsity entry covers one 32-token group, so the group size sets how much of the"
  echo "# 256-expert axis the DOWN projection can skip."
  for g in 32 64 128 256 512; do
    python "$BENCH" --impl fused --phase prefill --moe-group-tokens "$g" 2>&1 | grep '^BENCH' || true
  done
} > "$L/ab_moe_group_tokens.txt"

{
  echo "# Partial-RoPE lowering: 'partial' = slice + rotary_embedding_hf + slice + concat (4 ops, 64-wide"
  echo "# table); 'full' = head-dim permutation + one full-width rotate-half (1 op, 256-wide table)."
  for m in partial full; do
    python "$BENCH" --impl fused --layers 3 --rope-mode "$m" 2>&1 | grep '^BENCH' || true
  done
} > "$L/ab_rope_mode.txt"

echo "wrote A/B logs under $L"
