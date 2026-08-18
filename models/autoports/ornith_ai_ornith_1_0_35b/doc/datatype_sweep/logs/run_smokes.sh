#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Reduced two-layer smoke for every candidate, one process at a time.
# `$datatype-sweep`: "when changing datatypes first run a quick one-decoder smoketest to check it
# works correctly ... before using it or rejecting it in a full model pareto sweep".
set -u
cd "$(git rev-parse --show-toplevel)"
R=models/autoports/ornith_ai_ornith_1_0_35b
D=$R/doc/datatype_sweep
mkdir -p "$D/logs/smoke"
STATUS="$D/logs/smoke_status.txt"
: > "$STATUS"

for cfg in "$D"/candidates/*.json; do
  id=$(basename "$cfg" .json)
  [ "$id" = "index" ] && continue
  echo "=== smoke $id ==="
  python "$D/logs/smoke_policy.py" --config "$cfg" \
      --output "$D/logs/smoke/$id.json" > "$D/logs/smoke/$id.txt" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "$id OK" >> "$STATUS"
  else
    echo "$id FAILED rc=$rc" >> "$STATUS"
  fi
done
echo "=== done ===" >> "$STATUS"
cat "$STATUS"
