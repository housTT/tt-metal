#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# The LM head's decode GEOMETRY, re-measured under the SELECTED precision policy.
#
# `$stage-review` is explicit that a core-count / program-config / in0_block_w result obtained under
# BFP8/HiFi does not validate or reject geometry under a selected BFP4/LoFi policy. The optimized
# full-model stage chose `mcast1d` on 110 cores with a width-sharded terminal norm, and it measured
# every alternative at `HiFi2 BF16 x BFP8`. This stage moved that row to `LoFi BF16 x BFP4`, and its
# own tt-perf-report capture shows the row's bound class changing with it (DRAM at 69.0 % -> SLOW at
# 48.6 %). So the ladder is re-run here, unchanged in every other respect, with **no policy
# argument** - i.e. on the selected config.
#
# Same harness, same reduced two-layer variant, one process per arm, serialized
# (`doc/optimized_full_model/logs/ab_terminal.py`). Rows are absolute milliseconds, so a difference
# here is the same difference on 40 layers.
#
#     bash models/autoports/ornith_ai_ornith_1_0_35b/doc/datatype_sweep/logs/ab_terminal_geometry.sh
#
# `dram-sharded-c64` is `tt-perf-report`'s own advice under the BASELINE policy and the only legal
# DRAM-sharded core count (32 and 16 exceed L1 - the previous stage measured both failing, and this
# ladder re-checks 32 to confirm the blocker is a property of L1 rather than of the dtype).
set -u
R=models/autoports/ornith_ai_ornith_1_0_35b
OUT="$R/doc/datatype_sweep/logs/ab_terminal_geometry.txt"
P="$R/doc/optimized_full_model/logs/ab_terminal.py"
: > "$OUT"

run() {
  local name="$1"; shift
  echo "=== arm $name : $* ===" >> "$OUT"
  timeout 900 python "$P" --arm "$name" "$@" >> "$OUT" 2>&1
  echo "--- exit $? ---" >> "$OUT"
}

# The shipped geometry, on the selected policy. This is the row every other arm is compared against.
run sel-mcast1d-c110-nsh-align32  --lm-head-program mcast1d --lm-head-cores 110 --terminal-norm-sharded 1 --vocab-align-tiles 32
run sel-mcast1d-c110-repeat       --lm-head-program mcast1d --lm-head-cores 110 --terminal-norm-sharded 1 --vocab-align-tiles 32

# The alternative core counts, same norm layout and vocabulary alignment.
run sel-mcast1d-c88-nsh-align32   --lm-head-program mcast1d --lm-head-cores 88  --terminal-norm-sharded 1 --vocab-align-tiles 32
run sel-mcast1d-c64-nsh-align32   --lm-head-program mcast1d --lm-head-cores 64  --terminal-norm-sharded 1 --vocab-align-tiles 32

# tt-perf-report's DRAM-sharded advice, at the only legal core count and at one that should not build.
run sel-dram-sharded-c64          --lm-head-program dram_sharded --lm-head-cores 64 --terminal-norm-sharded 1 --vocab-align-tiles 32
run sel-dram-sharded-c32          --lm-head-program dram_sharded --lm-head-cores 32 --terminal-norm-sharded 1 --vocab-align-tiles 32

# The K block, which is the other geometry knob the previous stage swept at BFP8.
run sel-mcast1d-c110-k4           --lm-head-program mcast1d --lm-head-cores 110 --terminal-norm-sharded 1 --vocab-align-tiles 32 --lm-head-in0-block-w 4
run sel-mcast1d-c110-k16          --lm-head-program mcast1d --lm-head-cores 110 --terminal-norm-sharded 1 --vocab-align-tiles 32 --lm-head-in0-block-w 16

# The inherited spelling, as the floor.
run sel-interleaved               --lm-head-program interleaved --terminal-norm-sharded 0 --vocab-align-tiles 32
echo "=== done ===" >> "$OUT"
