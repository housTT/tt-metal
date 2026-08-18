#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# The full-model sweep: one process per candidate, one device job at a time.
# Each process opens the 1x4 ring, builds the 40-layer model at that candidate's precision config,
# runs the prefill readiness check and three traced teacher-forcing repeats, and closes the mesh.
#
#   run_sweep.sh                 # every candidate in doc/datatype_sweep/candidates
#   run_sweep.sh C05 C06         # only the candidates whose id contains one of these
#
# TF_REPEATS sets the teacher-forcing repeat count (default 10; the first is normally cold, so that
# is nine warm windows per candidate).
set -u
cd "$(git rev-parse --show-toplevel)"
R=models/autoports/ornith_ai_ornith_1_0_35b
D=$R/doc/datatype_sweep
mkdir -p "$D/logs/runs" "$D/runs"
TF_REPEATS=${TF_REPEATS:-10}
STATUS="$D/logs/sweep_status.txt"
: > "$STATUS"

configs=()
for cfg in "$D"/candidates/*.json; do
  id=$(basename "$cfg" .json)
  [ "$id" = "index" ] && continue
  if [ "$#" -gt 0 ]; then
    keep=0
    for want in "$@"; do case "$id" in *"$want"*) keep=1;; esac; done
    [ "$keep" = 1 ] || continue
  fi
  configs+=("$cfg")
done

echo "sweeping ${#configs[@]} candidate(s)" | tee -a "$STATUS"
for cfg in "${configs[@]}"; do
  id=$(basename "$cfg" .json)
  echo "=== $id $(date -Is) ===" | tee -a "$STATUS"
  extra=()
  # The baseline additionally runs the official readiness runners (each building its own generator)
  # as the control that this driver's shared-generator shortcut reports the same numbers.
  [ "$id" = "S00-baseline-optimized" ] && extra=(--verify-against-runners)
  python "$D/logs/sweep_one.py" --config "$cfg" --output "$D/runs/$id.json" --tf-repeats "$TF_REPEATS" "${extra[@]+"${extra[@]}"}" \
      > "$D/logs/runs/$id.txt" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    line=$(python - "$D/runs/$id.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))["result"]
print(f"{r['status']:>4}  prefill top1={r['prefill_top1']:.3f} top5={r['prefill_top5']:.3f}  "
      f"teacher top1={r['teacher_top1']:.3f} top5={r['teacher_top5']:.3f} top100={r['teacher_top100']:.3f}  "
      f"decode={r['teacher_decode_t/s/u']:.3f} t/s/u (warm {r['warm_repeats']}/{r['total_repeats']}, spread {r['teacher_decode_spread_pct']:.2f}%)  "
      f"TTFT={r['ttft_ms']:.1f} ms")
PY
)
    echo "$id $line" | tee -a "$STATUS"
  else
    echo "$id RUN_FAILED rc=$rc (see $D/logs/runs/$id.txt)" | tee -a "$STATUS"
  fi
done
echo "=== done ===" | tee -a "$STATUS"
