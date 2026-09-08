# Stage Review

Verdict: clean-pass

Independent resumed review of Stage 09 vLLM integration for
`google/gemma-4-26B-A4B-it`, dated 2026-09-08. Reviewed the live workspace at
tt-metal `4c3ae53b496` plus the context-checker repair and documentation changes.
The vLLM checkout remains clean at
`2f81f493b969da7ce3cd64c0f6bc1895b8f229cb`.

The runner failure is a context-checker scoping error. The repaired checker
preserves the recorded profile limits, and the retained serving evidence
continues to support Stage 09. One inaccurate description in the previous
review was corrected during this review; its evidentiary limit is explicit
below. No hardware, server, reset, profiler, or new model experiment was run.

## Required Work

None. The stage owner should complete the normal local post-review checkpoint
and record its SHA. This is the administrative step ordered after review,
not an outstanding serving check.

## Other Concerns

None requiring remediation in this repair. The following conclusions were
independently derived rather than inferred from the prior pass label.

### Checker behavior

- Executing the original committed checker against the current artifacts
  produces exactly one critical finding:
  `readiness_vllm/unified_gate_manifest.json:profiles.P150.max_model_len` is
  compared with 262,144. Its existing `profile_context_limits()` already
  resolves P150/P150x2/P150x4 to 50,624/262,144/262,144, and its existing
  directory handling accepts the same P150 value in profile-local artifacts.
- The repair applies that existing profile floor to the explicit top-level
  `profiles` section of JSON files directly under `readiness_vllm/`. It does
  not skip the manifest, remove context keys, modify the contract, or alter
  serving configuration. Unscoped settings retain the global floor.
- All nine added host tests pass. Additional independent temporary-artifact
  checks produced seven required failures: P150 below 50,624; TP2, TP4, and
  unknown profiles below 262,144; an unscoped root setting; a nested
  `settings.profiles` section; and a profile-shaped document outside root
  `readiness_vllm/`. Directory-scoped behavior remains covered by the tests.
- An independent invocation of the complete
  `.agents/prompts/model_bringup_multigoal/09-vllm.check.sh` exits 0, reporting
  no degenerate output and a valid context contract. The existing free-text
  advisory for the documented P150 command remains visible.

### Context and retained serving evidence

Paths here are relative to `models/autoports/google_gemma_4_26b_a4b_it/`.

- `doc/context_contract.json` and the selected precision file are unchanged.
  The precision SHA256 is still
  `58125b7d377e77ab0f0edad5b81d5bb2b1252df3ceff962158624b53925df2ca`.
  Every recorded profile section agrees on the limits used by the repaired
  checker.
- The inherited P150 reduction has an analytical physical-capacity basis in
  `doc/multichip_decoder/capacity_projection.json`: 56,398,546,944 projected
  bytes at advertised context versus 34,359,738,368 physical bytes. That
  source-lifetime projection records 50,624 as its largest contiguous
  analytical contract and 50,625 as the first rejected candidate. The latter
  is an analytical/host-guard rejection, not a measured hardware OOM.
  Subsequent contract sections preserve the limit and link their capacity
  and boundary results. This repair introduces no further reduction.
- The selected P150 boundary artifact
  `doc/optimized_full_model/final/full_stack_context_tp1/lifetime_fix_50624/long_context_probe_tp1.json`
  records all 30 real-weight layers, BF16 KV, a 50,623-token prefill, and a
  traced boundary replay ending at position 50,624.
- The selected TP2/TP4 boundary artifacts under
  `doc/optimized_full_model/final/nonaligned/` record layers `[0, 5]`,
  262,143-token prefills, and final position 262,144. Their separate
  `final/capacity/capacity_tp2.json` and `capacity_tp4.json` records cover
  construction of all 30 layers at the advertised capacity. These are
  complementary observations, not all-layer maximum-length serving runs.
- All 39 SHA256 entries in `readiness_vllm/unified_gate_manifest.json` match
  the current files. Decompressed final server logs and retained `/v1/models`
  responses agree on 50,624/262,144/262,144, block size 64, 32 sequences, async
  scheduling, device sampling `all`, and the `gemma4` reasoning/tool parsers.
  Every profile retains a successful 29-token chat request, a parsed
  `get_weather` call with `city=Paris`, and parsed reasoning with answer 5.
- The three full-sampling logs each finish with 72 passed and one expected
  all-vocabulary-logprobs skip. Each final server log contains 1,580 POST
  responses with HTTP 200, the one expected HTTP 400, zero HTTP 500, and
  completed device close. These results belong to the unchanged final
  lifetimes identified in the manifest.
- The raw primary benchmarks complete 128/128/1 at concurrency one and
  reproduce headline TTFT P50 values 235.121/235.547/203.296 ms and
  TPOT-derived decode rates 36.206/43.958/48.512 t/s/u. The raw secondary
  benchmarks complete 100/100/32, with output throughput
  100.548/122.205/129.489 tok/s. This repair makes no new performance claim.

### Output inspection

Read all 36 retained greedy/sampled serving completions. The six prompts
produce coherent haiku, labeled-fruit explanations, inventor stories,
thermodynamics explanations, French translations, and valid iterative
Fibonacci implementations. No mechanical repetition, gibberish, wrong-language
drift, or visible request contamination appears. Long answers stop at the
recorded token cap; this does not establish complete instruction coverage.

Loaded the exact checkpoint tokenizer locally with `local_files_only=True`.
It is `GemmaTokenizer` with a chat template. Re-rendered all 18 profile/prompt
combinations and matched both rendered text and token IDs exactly. Verified
the retained HF/standalone-TT control and assessment hashes, decoded all six
HF and TT control outputs, and compared their task, language, and coherent
prefixes with serving. Controls use 64 greedy tokens while serving uses a
256-token cap and includes sampling; exact full-length text equivalence is
not claimed.

## Hard-Check Gaps

- The checker consumes recorded profile limits. It does not independently
  validate every capacity projection or prove its physical assumptions; that
  remains the contract/stage-review responsibility. This pre-existing division
  of responsibility is unchanged by the aggregate-manifest repair.
- TP2/TP4 maximum-length boundary execution is representative-layer evidence,
  with separate full-stack capacity and ordinary live serving evidence. No
  all-layer maximum-length TP2/TP4 serving request is established here. The
  original Stage 09 contract requires matching advertised serving context and
  a valid non-aligned serving request, both present; it does not independently
  prescribe such a new maximum-length serving experiment.

## Anomaly Ledger

- Observed anomaly: the resumed runner rejected P150's aggregate-manifest
  context despite accepting the same profile-local context.
  Evidence: original `09-09-vllm.resume-2.check-1.log`, fresh
  `09-vllm-context-reproduce.log`, and retained
  `context_gate_repair/reproduced_check.log.xz`.
  Affected path: runner-side JSON context scan.
  Control or comparison: original/repaired scans over identical artifacts;
  nine regression tests and seven additional negative checks.
  Likely subsystem: profile scope resolution in `scan_caps()`.
  Investigation performed: read the complete checker, rederived profile
  limits, reproduced the one original critical finding, and ran the complete
  repaired stage check.
  Resolution: fixed; aggregate/profile-local floors agree, and unsupported
  reductions beneath each recorded floor still fail.

- Observed anomaly: the earlier `STAGE_REVIEW.md` called all three selected
  boundary probes complete-model runs.
  Evidence: TP2/TP4 `long_context_probe_tp*.json` list `[0, 5]`; P150 lists all
  30 layers. Earlier full-model TP2/TP4 boundary artifacts also list `[0, 5]`.
  Affected path: evidence description, not a changed serving limit.
  Control or comparison: separate all-30-layer capacity artifacts, final
  serving lifetimes, and direct non-aligned API responses.
  Likely subsystem: documentation overstatement.
  Investigation performed: independently inspected both generations of raw
  boundary artifacts and notified the stage owner. Reread the corrected
  earlier review row, README, and repair work-log section.
  Resolution: fixed with an explicit historical correction; representative
  boundary scope remains a documented evidence limitation.

## Scope Inspected

- Goal: `bringup/artifacts/multigoal-runs/gemma4-26b-a4b-p150/09-09-vllm.prompt.txt`.
- Skills: `.agents/skills/stage-review/SKILL.md`,
  `vllm-integration/SKILL.md`, `tt-device-usage/SKILL.md`, and
  `qualitative-check/SKILL.md`.
- Repair: complete `.agents/scripts/check_context_contract.py`, original
  committed version, complete `.agents/tests/test_check_context_contract.py`,
  and the stage shell gate.
- Artifacts: original/reproduced/fixed runner logs; README/work log/previous
  review and their corrections; context and precision contracts; referenced
  projection, capacity, and boundary records; unified manifest; all final
  compressed server logs and sampling logs; feature responses; raw benchmark
  results; all serving qualitative text and retained HF/TT controls.
- Code spot checks: adapter construction, cache pass-through, changed-only
  page refresh, canonical decode delegation, sampling compatibility branch,
  and unchanged model context constants. This focused resumed review does
  not claim to repeat the previous review's complete implementation audit.

Commands run include:

```bash
env PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s .agents/tests -p test_check_context_contract.py -v
# 9 tests passed

env PYTHONDONTWRITEBYTECODE=1 python3 .agents/scripts/check_context_contract.py \
  --model-dir models/autoports/google_gemma_4_26b_a4b_it \
  --stage vllm --require-contract
# exit 0

MODEL_DIR=models/autoports/google_gemma_4_26b_a4b_it \
HF_MODEL=google/gemma-4-26B-A4B-it PYTHONDONTWRITEBYTECODE=1 \
bash .agents/prompts/model_bringup_multigoal/09-vllm.check.sh
# exit 0, invoked through a host subprocess that captured the output

git diff --check
# passed
```

Additional small host scripts compared original/repaired scans, exercised
negative scope cases in a temporary directory, recomputed hashes and metrics,
decompressed logs, inspected JSON, and loaded the cached tokenizer. No
implementation or retained evidence file was changed by this reviewer. This
report is the reviewer's only authored file. Python/documentation changes do
not require a C++/CMake build.

## Residual Risk

- This is a repair and retained-evidence review, not new hardware validation
  or release certification. The inherited P150 analytical capacity basis and
  TP2/TP4 representative-layer maximum-length coverage retain the scope
  described above.
- The prior review's controlled B1/B32 distribution sensitivity, truncated
  qualitative evidence, optional host compatibility modes, disabled prefix
  caching, text-only scope, and narrowly classified recovered tokenizer warning
  are not expanded or erased by this checker repair.
- Local post-review commit provenance must be recorded by the stage owner;
  nothing should be pushed as part of autonomous stage closure.
