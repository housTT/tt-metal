#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Watcher run for the full-model suite. SEPARATE from the profiler capture, as $tt-device-usage
# requires: TT_METAL_WATCHER and the device profiler must never share a run.
#
# TT_METAL_WATCHER_DISABLE_ETH=1 is inherited from the decoder stage's known limitation 2: watcher
# does not cover ACTIVE_ETH cores on this configuration. Every worker-core assert stays armed.
#
#     bash models/autoports/ornith_ai_ornith_1_0_35b/doc/full_model/logs/run_watcher.sh
set -uo pipefail

ROOT="models/autoports/ornith_ai_ornith_1_0_35b"
OUT="$ROOT/doc/full_model/watcher"
mkdir -p "$OUT"

TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
  python -m pytest "$ROOT/tests/test_full_model.py" -m "not long" -q --timeout=3600 \
  > "$OUT/watcher_pytest.txt" 2>&1
echo "pytest exit: $?" >> "$OUT/watcher_pytest.txt"

WATCHER_LOG="generated/watcher/watcher.log"
if [ -f "$WATCHER_LOG" ]; then
  cp "$WATCHER_LOG" "$OUT/watcher.log"
fi

{
  echo "watcher artifacts for the full-model suite"
  echo "pytest log : $OUT/watcher_pytest.txt.gz"
  echo "watcher log: ${WATCHER_LOG}"
  for f in "$OUT/watcher_pytest.txt" "$OUT/watcher.log"; do
    [ -f "$f" ] || continue
    echo "--- $f ---"
    echo "error/assert/hang lines: $(grep -ciE 'watcher.*(error|assert|hang|detected)|\bWATCHER\b.*(ERROR|ASSERT)' "$f" || true)"
    grep -iE "watcher.*(error|assert|hang|detected)" "$f" | head -20 || true
  done
} > "$OUT/watcher_error_count.txt" 2>&1

gzip -9 -f "$OUT/watcher_pytest.txt" 2>/dev/null || true
[ -f "$OUT/watcher.log" ] && gzip -9 -f "$OUT/watcher.log"
cat "$OUT/watcher_error_count.txt"
