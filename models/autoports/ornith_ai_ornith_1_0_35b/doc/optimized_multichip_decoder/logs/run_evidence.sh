#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Regenerate every committed artifact of the Ornith-1.0-35B OPTIMIZED MULTICHIP decoder stage, in the
# order the hardware discipline requires: correctness and A/B first, then the profiler run, then the
# watcher run **last and alone** (watcher and Tracy must never share a process).
#
# Run from the tt-metal root with the 4-chip mesh idle:
#     bash models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_multichip_decoder/logs/run_evidence.sh
#
# Steps can be selected with STEPS="suite bench ab probes tracy watcher" (default: all).
set -euo pipefail

ROOT="models/autoports/ornith_ai_ornith_1_0_35b"
DOC="$ROOT/doc/optimized_multichip_decoder"
LOGS="$DOC/logs"
TEST="$ROOT/tests/test_multichip_decoder.py"
STEPS="${STEPS:-suite bench ab probes tracy watcher}"

step() { case " $STEPS " in *" $1 "*) return 0;; *) return 1;; esac; }

if step suite; then
  echo "=== full suite (the correctness floor, on the final default path) ==="
  # A failing log is preserved under its own name before anything can overwrite it. This stage lost
  # the full log of the one cross-device-divergence failure it saw (README section 8 limitation 3)
  # because a later run of this script rewrote the fixed filename.
  python -m pytest "$TEST" -v -p no:randomly > "$LOGS/pytest_full_suite.txt" 2>&1 || {
    cp "$LOGS/pytest_full_suite.txt" "$LOGS/pytest_full_suite_FAILED.txt"
    gzip -9 -f "$LOGS/pytest_full_suite_FAILED.txt"
    tail -40 "$LOGS/pytest_full_suite.txt"; exit 1; }
  tail -1 "$LOGS/pytest_full_suite.txt"
  gzip -9 -f "$LOGS/pytest_full_suite.txt"
fi

if step bench; then
  echo "=== before/after warmed prefill and traced decode, same harness ==="
  # `before` is the multichip stage's shipped path, reached by turning BOTH of this stage's
  # behavioural knobs back to what it inherited: its router (`--router-mode topk`) and its collective
  # (`--ccl-mode auto --auto-stack-sum stack_sum`: the stack-sum crossover, spelled with the
  # deprecated ttnn.all_gather). Same binary, same process shape,
  # same weights - the only honest way to get a before/after for an in-place optimization, and it has
  # to restore both or the "before" arm is not the path stage 4 shipped.
  {
    python "$LOGS/bench.py" --impl multichip --mesh 1x4 --layers 0,3 --router-mode topk \
      --ccl-mode auto --auto-stack-sum stack_sum \
      --weights real --tag before-optimized-multichip --json "$LOGS/bench_before.json"
    python "$LOGS/bench.py" --impl multichip --mesh 1x4 --layers 0,3 \
      --weights real --tag after-optimized-multichip --json "$LOGS/bench_after.json"
    python "$LOGS/bench.py" --impl optimized --mesh 1x1 --layers 0,3 \
      --weights real --tag single-chip-baseline --json "$LOGS/bench_single_chip.json"
  } 2>/dev/null | grep -E "^BENCH" > "$LOGS/ab_before_after.txt"
  cat "$LOGS/ab_before_after.txt"
fi

if step ab; then
  echo "=== whole-layer A/B for every knob, under the final default ==="
  python "$LOGS/ab_layer_knobs.py" 2>/dev/null | grep -E "^ABLAYER|^#" > "$LOGS/ab_layer_knobs.txt"
  # Guard: the shipped default must be the fastest arm of every knob, or the difference must be a
  # deliberate correctness trade recorded in the work log. Review round 2 found the stage shipping a
  # config 14 us/step slower than an arm it had itself measured, so this is checked rather than read.
  python - "$LOGS/ab_layer_knobs.txt" <<'PYEOF' 2>/dev/null | grep "^ABBEST" | tee "$LOGS/ab_best_arms.txt"
import sys, collections
best = collections.defaultdict(lambda: (1e9, None))
for line in open(sys.argv[1]):
    if not line.startswith("ABLAYER "):
        continue
    _, knob, arm, _idx, kind, _b, dec, _pre, _f = line.split()
    key = (knob, kind)
    if float(dec) < best[key][0]:
        best[key] = (float(dec), arm)
# Read the shipped arm from the module rather than restating it here, so the guard cannot drift
# from the defaults it is guarding. Only the arms whose name is not literally the module value need
# a mapping.
from models.autoports.ornith_ai_ornith_1_0_35b.tt import multichip_decoder as MC

shipped = {
    "router": MC.ROUTER_MODE,
    "collective": MC.CCL_MODE if MC.CCL_MODE != "auto" else MC.AUTO_STACK_SUM_MODE,
    "router_fidelity": "hifi2" if MC.ROUTER_DECODE_FIDELITY is not None else "hifi4-inherited",
    "residual": "l1" if MC.DECODE_RESIDUAL_MEMORY is not None else "dram-interleaved-inherited",
    "state_fidelity": "hifi4-inherited",
    "policy": "optimized",
    "geometry": "multichip-retuned",
    "sparse": "tp-rescaled" if MC.SPARSE_SCALE_CORES_BY_TP else "single-chip-inherited",
    "cast": "bf16" if MC.CCL_CAST_BLOCKFLOAT else "block-float",
}
# Three arms are EXPECTED to be faster than what ships, each for a reason recorded in the work log.
expected = {
    "collective": "the fast spelling is the one that diverges across devices (work_log section 11)",
    "policy": "bfp4-projections is faster and rejected on accuracy (work_log section 7.1)",
}
TIE = 0.0015  # the harness's own resolution; anything inside it is a tie, not a faster arm
for (knob, kind), (value, arm) in sorted(best.items()):
    ship = shipped.get(knob)
    ship_value = min(
        (float(l.split()[6]) for l in open(sys.argv[1])
         if l.startswith("ABLAYER ") and l.split()[1] == knob and l.split()[2] == ship
         and l.split()[4] == kind),
        default=None,
    )
    if arm == ship or (ship_value is not None and ship_value - value <= TIE):
        flag = "" if arm == ship else f"  (tie with {ship}, {ship_value:.3f})"
    elif knob in expected:
        flag = f"  (expected: {expected[knob]})"
    else:
        flag = "  <-- UNEXPLAINED: a non-shipped arm is faster"
    print(f"ABBEST {knob} {kind} fastest={arm} {value:.3f}{flag}")
PYEOF
fi

if step probes; then
  echo "=== router-gate spellings ==="
  {
    python "$LOGS/probe_gate.py" --rows 32 --valid 1 2>/dev/null | grep -E "^GATE|^TOPKW|^#"
    python "$LOGS/probe_gate.py" --rows 32 --valid 32 2>/dev/null | grep -E "^GATE|^TOPKW|^#"
  } > "$LOGS/probe_gate.txt"
  echo "=== traced-replay cross-device divergence stress ==="
  # Chases the anomaly this stage started from. Every block below runs both layer kinds and both
  # replay patterns: `--sync-every-replay` reproduces what test_traced_replay_does_not_leak does (a
  # synchronize after every replay), which is different fabric pressure from a back-to-back burst.
  #
  # The ATTRIBUTION control: the deprecated collective under the router the MULTICHIP stage shipped.
  # If the divergence reproduces here it is inherited, not introduced by this stage's router change.
  # Same round count as each arm of the A/B below (600), because at ~1.5 % of rounds a 240-round
  # sample can read zero by chance - review round 4 caught exactly that.
  {
    python "$LOGS/probe_replay_divergence.py" --rounds 150 --replays 256 --router-mode topk \
      --ccl-mode auto --auto-stack-sum stack_sum 2>/dev/null | grep -E "^REPLAYDIV|^#"
    python "$LOGS/probe_replay_divergence.py" --rounds 150 --replays 128 --sync-every-replay \
      --router-mode topk --ccl-mode auto --auto-stack-sum stack_sum 2>/dev/null \
      | grep -E "^REPLAYDIV|^#"
  } > "$LOGS/probe_replay_divergence.txt"
  # The A/B that decided the shipped collective: 150 rounds per (layer kind x replay pattern) for the
  # deprecated ttnn.all_gather and for the barrier-semaphore all_gather_async, 600 rounds each.
  {
    for mode in stack_sum stack_sum_async; do
      python "$LOGS/probe_replay_divergence.py" --rounds 150 --replays 256 --ccl-mode auto \
        --auto-stack-sum "$mode" 2>/dev/null | grep -E "^REPLAYDIV|^#"
      python "$LOGS/probe_replay_divergence.py" --rounds 150 --replays 128 --sync-every-replay \
        --ccl-mode auto --auto-stack-sum "$mode" 2>/dev/null | grep -E "^REPLAYDIV|^#"
    done
  } > "$LOGS/probe_replay_divergence_ab.txt"
  # The shipped collective, on the same harness at the same round count: `ttnn.all_reduce` at every
  # shape. This is the arm the stage ships, so it carries the largest sample.
  {
    python "$LOGS/probe_replay_divergence.py" --rounds 150 --replays 256 --ccl-mode all_reduce \
      2>/dev/null | grep -E "^REPLAYDIV|^#"
    python "$LOGS/probe_replay_divergence.py" --rounds 150 --replays 128 --sync-every-replay \
      --ccl-mode all_reduce 2>/dev/null | grep -E "^REPLAYDIV|^#"
  } > "$LOGS/probe_replay_divergence_allreduce.txt"
  echo "=== persistent-buffer collectives (OPT-009) ==="
  {
    python "$LOGS/probe_ccl_persistent.py" --dtype bfloat16 2>/dev/null | grep -E "^CCLPERS|^#"
    python "$LOGS/probe_ccl_persistent.py" --dtype bfloat8_b 2>/dev/null | grep -E "^CCLPERS|^#"
  } > "$LOGS/probe_ccl_persistent.txt"
fi

if step tracy; then
  echo "=== tt-perf-report captures (NO watcher in this process) ==="
  bash "$DOC/tracy/run_profiling.sh"
fi

if step watcher; then
  echo "=== watcher, last and alone ==="
  # `TT_METAL_WATCHER_DISABLE_ETH=1` for the reason the multichip stage recorded: watcher's
  # ACTIVE_ETH kernel config buffer overflows on this 4-chip configuration. Scoped limitation, not a
  # skipped run - every worker-core assert is still armed.
  mkdir -p "$DOC/watcher"
  TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
    python -m pytest "$TEST" -v -p no:randomly \
      -k "pcc or trace or determinism or stress or router or collectives or ccl or sparse or routing" \
      > "$DOC/watcher/watcher_pytest.txt" 2>&1 || {
        tail -40 "$DOC/watcher/watcher_pytest.txt"; exit 1; }
  tail -1 "$DOC/watcher/watcher_pytest.txt"
  grep -ciE "watcher.*(error|assert|hang|corrupt)" "$DOC/watcher/watcher_pytest.txt" \
    > "$DOC/watcher/watcher_error_count.txt" || true
  gzip -9 -f "$DOC/watcher/watcher_pytest.txt"
fi

# Staleness guard: every artifact that describes the measured path must post-date the code that path
# is. Review round 3 found the suite, Tracy and watcher artifacts predating a (comment-only, as it
# turned out) edit to the implementation, which is exactly the doubt this removes.
echo "=== document checks ==="
# Both guards are cheap and both exist because a review round found what they now catch:
# make_tables --check owns the numbers inside the TABLE markers, check_prose_figures owns the ones
# outside them.
python "$LOGS/make_tables.py" --check
python "$LOGS/check_prose_figures.py"

echo "=== artifact staleness check ==="
CODE="$ROOT/tt/multichip_decoder.py $ROOT/tests/test_multichip_decoder.py"
STALE=0
# Globbed rather than listed: review round 6 found five probe artifacts outside a hand-maintained
# list, so the guard reported "all evidence post-dates the implementation" while they did not.
for art in "$LOGS"/*.txt "$LOGS"/*.txt.gz "$LOGS"/*.json "$DOC"/watcher/*.gz "$DOC"/tracy/*/*.gz \
           "$DOC"/tracy/*/*.txt; do
  # Scripts are not evidence, and `traced_replay_divergence_failure.txt` is a HISTORICAL artifact:
  # it preserves a failure from an earlier code state and by definition cannot post-date the code.
  case "$art" in
    *run_evidence*|*make_tables*|*check_prose*|*traced_replay_divergence_failure*) continue;;
  esac
  [ -e "$art" ] || { echo "MISSING $art"; STALE=1; continue; }
  for src in $CODE; do
    if [ "$src" -nt "$art" ]; then echo "STALE $art is older than $src"; STALE=1; fi
  done
done
[ "$STALE" -eq 0 ] && echo "all evidence post-dates the implementation" || echo "!!! stale evidence above"

echo "=== done ==="
