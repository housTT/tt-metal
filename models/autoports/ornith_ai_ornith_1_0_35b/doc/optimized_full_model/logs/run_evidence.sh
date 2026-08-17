#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Regenerates every behavioural artifact under doc/optimized_full_model/, in order, on the 4-chip
# Blackhole ring. One device-facing command at a time, as $tt-device-usage requires. Run from the
# tt-metal root:
#
#     bash models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_full_model/logs/run_evidence.sh
#
# NOT part of this script, deliberately, because each needs a device session of its own:
#   * tracy/run_profiling.sh   - profiler capture (never with watcher)
#   * logs/run_watcher.sh      - watcher run     (never with the profiler)
#   * logs/ab_terminal.sh      - the terminal-path candidate ladder (many builds)
#   * logs/probe_footprint.py, logs/probe_long_prompt.py, logs/update_context_contract.py
#   * logs/check_prose_figures.py - deliberately NOT a step here. It asserts README.md and work_log.md
#     against the artifacts this script produces, so running it inside the same sweep would always fail
#     on the run that generates new numbers (the documents can only be refreshed afterwards). Run it
#     after refreshing them; its committed output is logs/check_prose_figures.txt.
set -uo pipefail

ROOT="models/autoports/ornith_ai_ornith_1_0_35b"
LOGS="$ROOT/doc/optimized_full_model/logs"
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

# 1-2. Readiness accuracy gates against the same fresh AIME24 chat-template reference the
#      full-model stage generated (its provenance is readiness_aime24_chat.meta.json).
step readiness_prefill      python "$DRIVER" --check prefill
step readiness_teacher      python "$DRIVER" --check teacher

# 3-4. Free-running generation: the raw continuation prompt the shared runner ships (labelled
#      continuation stress coverage for an instruct model) and a chat-template prompt.
step readiness_autoregressive python "$DRIVER" --check autoregressive --max-new-tokens 128
step readiness_autoregressive_chat python "$DRIVER" --check autoregressive --max-new-tokens 128 \
  --prompt-file "$ROOT/doc/optimized_full_model/autoregressive_chat_prompt.txt" \
  --output-dir "$ROOT/readiness_autoregressive_chat" --suffix _chat

# 5. The shared qualitative prompt suite, HF control plus TT, rendered with the chat template.
step readiness_qualitative  python "$DRIVER" --check qualitative --max-new-tokens 128

# 6-7. Warmed performance at the vLLM primary single-user profile, BOTH arms, nine repeats each, back
#      to back. The `inherited` arm rebuilds the pre-optimization path from constructor knobs, so
#      before/after is one script version and one sweep; nine repeats because TTFT's spread on this host
#      is wider than the effect. Each arm also measures its own serial-loop row, TTFT breakdown and
#      cold-length cost.
step bench_inherited        python "$LOGS/bench_full_model.py" --arm inherited --repeats 9 \
  --output "$ROOT/doc/optimized_full_model/perf_summary_before.json"
step bench_full_model       python "$LOGS/bench_full_model.py" --repeats 9

# 7. Where TTFT goes: the warmed prefill ladder, its slope/intercept split and the logging A/B.
step probe_prefill          python "$LOGS/probe_prefill.py"

# 8. Multi-request corruption regression: six prompts of different lengths, twice, one generator.
step probe_multi_prompt     python "$LOGS/probe_multi_prompt.py" --arms plain,plain --gen-len 24

# 9. The `before` arm of the minimal repro for the post-capture compilation hazard: the safe order,
#    which must reproduce both prompts.
#
#    The `after` arm is NOT run here any more. It deliberately replays a trace whose kernel binaries
#    were overwritten - that is the whole point of the repro - and on the optimized terminal path
#    that no longer merely corrupts the output: it wedged the mesh (Ethernet-core training timeouts,
#    tt-triage itself unable to attach; see doc/optimized_full_model/triage/bisect_after/ and
#    README §Rejected). Run it deliberately, on its own, when the hazard needs demonstrating:
#        python "$LOGS/probe_bisect.py" --order after
step probe_bisect_before    python "$LOGS/probe_bisect.py" --order before

# 10. The batch>1 slot contract: prefill state reaching slot 0, and what batch geometry alone moves.
step probe_batch_slots      python "$LOGS/probe_batch_slots.py"

# 11. Terminal-cost breakdown on the delivered configuration.
step probe_terminal         python "$LOGS/probe_terminal.py"

# 12. The runner-side degeneracy gate over everything generated above.
step check_degenerate       python models/common/readiness_check/check_degenerate_output.py \
  --model-dir "$ROOT" --missing-artifacts critical --scope autoregressive \
  --json "$ROOT/doc/optimized_full_model/degenerate_report.json"

# 13. The suite.
step pytest_full_model      python -m pytest "$ROOT/tests/test_full_model.py" -m "not long" -q --timeout=2400

echo "=== done ($FAILURES failed) ===" | tee -a "$STATUS"
cat "$STATUS"
# A partially failed evidence run must not look like a clean one to a caller or a gate.
exit $((FAILURES > 0))
