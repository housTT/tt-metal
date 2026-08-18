#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Everything the stage owes *after* the selected precision config has been written to
# doc/datatype_sweep/selected_precision_config.json. One device job at a time; each step goes
# through the normal construction path with no policy argument, so what is measured is what a
# later full-model or vLLM build gets by default.
#
#   1. token-out benchmark for the selected config          (the serving headline)
#   2. token-out benchmark for the pre-sweep baseline policy (the like-for-like comparison)
#   3. DRAM capacity at the full advertised context, per KV-cache dtype candidate
#   4. non-aligned prompt walk on the full stack at the full advertised cache
#   5. the shared qualitative prompt suite, with the HF control
#   6. the delivered pytest suite, including the `long` cases
set -u
cd "$(git rev-parse --show-toplevel)"
R=models/autoports/ornith_ai_ornith_1_0_35b
D=$R/doc/datatype_sweep
OFM=$R/doc/optimized_full_model
mkdir -p "$D/logs/post" "$D/capacity"
STATUS="$D/logs/post_status.txt"
: > "$STATUS"

step() {  # step <name> <logfile> -- <command...>
  local name=$1 log=$2; shift 3
  echo "=== $name $(date -Is) ===" | tee -a "$STATUS"
  "$@" > "$log" 2>&1
  local rc=$?
  echo "$name rc=$rc  ($log)" | tee -a "$STATUS"
  return 0
}

# 1 - the selected config's token-out benchmark, the same harness and workload the optimized
#     full-model stage used for its headline (prompt 128 / generate 128, warmed, nine repeats).
step token_out_selected "$D/logs/post/bench_selected.txt" -- \
  python "$OFM/logs/bench_full_model.py" --repeats 9 --output "$D/post_selection_token_out.json"

# 2 - the same benchmark on the pre-sweep policy, so the token-out comparison is like for like and
#     any conflict with the teacher-forcing ranking is resolvable rather than assumed away.
step token_out_baseline "$D/logs/post/bench_baseline_policy.txt" -- \
  env ORNITH_PRECISION_POLICY=optimized \
  python "$OFM/logs/bench_full_model.py" --repeats 9 --output "$D/post_selection_token_out_baseline.json"

# 3 - capacity at the advertised 262144-token context for every KV-cache dtype the sweep evaluated.
for kv in selected optimized C11-kv-bfp4 C12-kv-bf16; do
  case "$kv" in
    selected)  arg=(--config "$D/selected_precision_config.json") ;;
    optimized) arg=(--policy optimized) ;;
    *)         arg=(--config "$D/candidates/$kv.json") ;;
  esac
  step "capacity_$kv" "$D/logs/post/capacity_$kv.txt" -- \
    python "$D/logs/probe_capacity.py" "${arg[@]}" --output "$D/capacity/$kv.json"
done

# 4 - non-aligned prompt lengths on the full stack with the full advertised cache allocated.
step non_aligned_prompts "$D/logs/post/long_prompt.txt" -- \
  python "$OFM/logs/probe_long_prompt.py" --budget-s 2400 --output "$D/long_prompt.json"

# 5 - the shared qualitative suite on the selected config, with the HF control.
step qualitative "$D/logs/post/qualitative.txt" -- \
  python "$OFM/logs/run_readiness.py" --check qualitative --out-dir "$D"

# 6 - the delivered suite, short cases then the long ones.
step pytest_short "$D/logs/post/pytest_short.txt" -- \
  python -m pytest "$R/tests/test_full_model.py" -q -m "not long"
step pytest_long "$D/logs/post/pytest_long.txt" -- \
  python -m pytest "$R/tests/test_full_model.py" -q -m long

echo "=== done ===" | tee -a "$STATUS"
cat "$STATUS"
