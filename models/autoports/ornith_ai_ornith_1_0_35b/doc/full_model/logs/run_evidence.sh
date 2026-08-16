#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Regenerates every artifact under doc/full_model/, in order, on the 4-chip Blackhole ring.
# One device-facing command at a time, as $tt-device-usage requires. Run from the tt-metal root:
#
#     bash models/autoports/ornith_ai_ornith_1_0_35b/doc/full_model/logs/run_evidence.sh
#
# The profiler capture (tracy/run_profiling.sh) and the watcher run (logs/run_watcher.sh) are
# deliberately NOT part of this script: profiler and watcher evidence must come from separate runs.
set -uo pipefail

ROOT="models/autoports/ornith_ai_ornith_1_0_35b"
LOGS="$ROOT/doc/full_model/logs"
DRIVER="$LOGS/run_readiness.py"
STATUS="$LOGS/run_evidence_status.txt"
: > "$STATUS"
FAILURES=0

step() {
  local name="$1"; shift
  echo "=== $name ===" | tee -a "$STATUS"
  local started=$SECONDS
  if "$@" > "$LOGS/$name.txt" 2>&1; then
    echo "  ok   ($((SECONDS - started)) s)  -> $LOGS/$name.txt" | tee -a "$STATUS"
  else
    echo "  FAIL ($((SECONDS - started)) s)  -> $LOGS/$name.txt" | tee -a "$STATUS"
    FAILURES=$((FAILURES + 1))
  fi
}

# 1-2. Readiness accuracy gates against the fresh AIME24 chat-template reference.
step readiness_prefill      python "$DRIVER" --check prefill
step readiness_teacher      python "$DRIVER" --check teacher

# 3-4. Free-running generation: the raw continuation prompt the shared runner ships (labelled
#      continuation stress coverage for an instruct model) and a chat-template prompt.
step readiness_autoregressive python "$DRIVER" --check autoregressive --max-new-tokens 128
step readiness_autoregressive_chat python "$DRIVER" --check autoregressive --max-new-tokens 128 \
  --prompt-file "$ROOT/doc/full_model/autoregressive_chat_prompt.txt" \
  --output-dir "$ROOT/readiness_autoregressive_chat" --suffix _chat

# 5. The shared qualitative prompt suite, HF control plus TT, rendered with the chat template.
step readiness_qualitative  python "$DRIVER" --check qualitative --max-new-tokens 128

# 6. Warmed performance at the vLLM primary single-user profile.
step bench_full_model       python "$LOGS/bench_full_model.py"

# 7. Multi-request corruption regression: six prompts of different lengths, twice, one generator.
step probe_multi_prompt     python "$LOGS/probe_multi_prompt.py" --arms plain,plain --gen-len 24

# 8. The minimal repro for the post-capture compilation hazard, both orders. The `after` arm is
#    EXPECTED to report `A stable=False`: it calls the trace directly and bypasses the generator's
#    `_ensure_traces_replay_safe` guard on purpose, which is what makes it the repro.
step probe_bisect_after     python "$LOGS/probe_bisect.py" --order after
step probe_bisect_before    python "$LOGS/probe_bisect.py" --order before

# 8b. The batch>1 slot contract: prefill state reaching slot 0, and what batch geometry alone moves.
step probe_batch_slots      python "$LOGS/probe_batch_slots.py"

# 9. Terminal-cost breakdown and the sampler A/B that chose the grouped local top-k.
step probe_terminal_grouped python "$LOGS/probe_terminal.py" --topk-groups 20
step probe_terminal_single  python "$LOGS/probe_terminal.py" --topk-groups 1

# 10. The runner-side degeneracy gate over everything generated above.
step check_degenerate       python models/common/readiness_check/check_degenerate_output.py \
  --model-dir "$ROOT" --missing-artifacts critical --scope autoregressive \
  --json "$ROOT/doc/full_model/degenerate_report.json"

# 11. The suite.
step pytest_full_model      python -m pytest "$ROOT/tests/test_full_model.py" -m "not long" -q --timeout=2400

echo "=== done ($FAILURES failed) ===" | tee -a "$STATUS"
cat "$STATUS"
# A partially failed evidence run must not look like a clean one to a caller or a gate.
exit $((FAILURES > 0))
