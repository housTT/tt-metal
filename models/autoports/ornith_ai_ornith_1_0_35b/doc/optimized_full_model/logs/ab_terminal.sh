#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# The optimized-full-model terminal-path ladder: every arm behind README §3 and §9, in one place.
# One process per arm, serialized, so the arms share nothing but the checkout - `$tt-device-usage`
# asks for one hardware-facing command at a time. Every arm runs on the reduced two-layer variant, so
# a build costs ~15 s instead of ~200 s and the rows are absolute milliseconds.
#
#     bash models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_full_model/logs/ab_terminal.sh
#
# Output: logs/ab_terminal.txt (one `ARM_JSON {...}` line per arm). Regenerate the table with:
#     python - <<'PY'  ... see README §3 ... PY   (logs/ab_terminal_table.md is the committed table)
#
# Two arms are EXPECTED to fail and their failures are the evidence for README §3.2: at 32 and 16
# cores the DRAM-sharded head's circular buffers exceed L1.
set -u
R=models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_full_model
OUT="$R/logs/ab_terminal.txt"
P="$R/logs/ab_terminal.py"
: > "$OUT"

run() {
  local name="$1"; shift
  echo "=== arm $name : $* ===" >> "$OUT"
  timeout 900 python "$P" --arm "$name" "$@" >> "$OUT" 2>&1
  echo "--- exit $? ---" >> "$OUT"
}

# --- the inherited spelling, and the two knobs on their own ---------------------------------------
run baseline                  --lm-head-program interleaved  --terminal-norm-sharded 0
run norm-sharded              --lm-head-program interleaved  --terminal-norm-sharded 1
run groups-auto               --lm-head-program interleaved  --terminal-norm-sharded 0
run align32                   --lm-head-program interleaved  --terminal-norm-sharded 0 --vocab-align-tiles 32
run align32-repeat            --lm-head-program interleaved  --terminal-norm-sharded 0 --vocab-align-tiles 32
run interleaved-nsh-align32   --lm-head-program interleaved  --terminal-norm-sharded 1 --vocab-align-tiles 32

# --- the 1D mcast decode geometry, across core counts and both norm layouts -----------------------
run mcast1d-c64               --lm-head-program mcast1d --lm-head-cores 64  --terminal-norm-sharded 1
run mcast1d-c88               --lm-head-program mcast1d --lm-head-cores 88  --terminal-norm-sharded 0
run mcast1d-c110-plain        --lm-head-program mcast1d --lm-head-cores 110 --terminal-norm-sharded 0
run mcast1d-c110-nsh          --lm-head-program mcast1d --lm-head-cores 110 --terminal-norm-sharded 1
run mcast1d-c110-align32      --lm-head-program mcast1d --lm-head-cores 110 --terminal-norm-sharded 0 --vocab-align-tiles 32
run mcast1d-c110-align32-b    --lm-head-program mcast1d --lm-head-cores 110 --terminal-norm-sharded 0 --vocab-align-tiles 32
run mcast1d-c110-nsh-align32  --lm-head-program mcast1d --lm-head-cores 110 --terminal-norm-sharded 1 --vocab-align-tiles 32

# --- tt-perf-report's own advice: DRAM-sharded. c32 and c16 are the expected L1 failures ----------
run dram-sharded-c64          --lm-head-program dram_sharded --lm-head-cores 64 --terminal-norm-sharded 1
run dram-sharded-c32          --lm-head-program dram_sharded --lm-head-cores 32 --terminal-norm-sharded 1
run dram-sharded-c16          --lm-head-program dram_sharded --lm-head-cores 16 --terminal-norm-sharded 1

# --- the LM-head weight dtype trial (rejected on real-weight accuracy, README §3.4) ---------------
run bfp4-head-plain           --lm-head-program interleaved  --terminal-norm-sharded 0 --lm-head-dtype bfp4
run bfp4-head-interleaved     --lm-head-program interleaved  --terminal-norm-sharded 1 --lm-head-dtype bfp4
run mcast1d-c110-bfp4         --lm-head-program mcast1d --lm-head-cores 110 --terminal-norm-sharded 0 --lm-head-dtype bfp4
run mcast1d-c110-nsh-align32-bfp4 \
                              --lm-head-program mcast1d --lm-head-cores 110 --terminal-norm-sharded 1 \
                              --vocab-align-tiles 32 --lm-head-dtype bfp4
echo "=== done ===" >> "$OUT"
