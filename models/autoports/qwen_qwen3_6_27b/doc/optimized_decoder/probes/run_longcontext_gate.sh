#!/usr/bin/env bash
# The four full-context cases, on their own, as a gate before spending an hour on a full campaign.
#
# `test_full_advertised_context` is parametrised over both layer kinds and both weight sources, and it is
# the only test in the suite that asserts a *scale* rather than only a PCC - so it is the one that catches
# a systematic gain error, and the one a precision change has to clear first.  Running it alone costs
# about 25 minutes against a campaign's hour and a half.
#
#   bash models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/run_longcontext_gate.sh
#
# Watch it with:
#   tail -f models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/logs/long_context.log
set -uo pipefail
REPO=/home/ttuser/dev/qwen/tt-metal
cd "$REPO"
source "$REPO/models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/ttenv.sh" > /dev/null
LOGS="$REPO/models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/logs"

python -m pytest models/autoports/qwen_qwen3_6_27b/tests/test_optimized_decoder.py \
  -k test_full_advertised_context --long-context -v -s > "$LOGS/long_context.log" 2>&1
rc=$?
echo "== $(date -Is) long-context gate exit $rc"
grep -E "^FAILED|passed|failed|scaled by" "$LOGS/long_context.log" | tail -12
exit "$rc"
