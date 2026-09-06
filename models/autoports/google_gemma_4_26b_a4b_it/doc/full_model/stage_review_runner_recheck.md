# Stage Review

Verdict: more-work-needed

Stage: 06 full-model, runner-side failed-check repair. Independent host-only
review on 2026-09-06 of the live `hous/gemma-4-26b-a4b-it` worktree at
`2fb82badbec`, with the checker repair uncommitted.

## Required Work

- P1: Missing TT completion evidence can pass the critical missing-artifact gate.
  Evidence: `models/common/readiness_check/check_degenerate_output.py:202`
  accepts a metadata object with no `tt` entry, or an empty `tt.token_ids`
  array, when `tt_completion.txt` is absent. At lines 120-122,
  `check_completion` records `source: null` and returns without a finding.
  Calling the real CLI `main` with mocked discovery/artifact reads and
  `--missing-artifacts critical --scope autoregressive` returned exit 0 for
  both `{}` and `{"tt":{"token_ids":[]}}`. The current five tests do not
  exercise either case.
  Why this matters: the runner explicitly requires free-running TT evidence
  to exist; discovering a metadata filename does not establish that any
  generation took place. A partial or damaged retained artifact can silently
  satisfy that required gate.
  Required next step: require usable TT completion evidence after discovery,
  validate its expected structure, and make missing content honor the
  missing-artifact policy or fail as an artifact error. Add regression cases
  for absent and empty TT payloads, retain the valid token-ID-only case, and
  rerun the exact runner check.

- P1: Text-first measurement bypasses the promised token-collapse check.
  Evidence: `models/common/readiness_check/check_degenerate_output.py:96`
  always selects regex words when nonblank text exists and ignores supplied
  token IDs. Direct host probes with 64 identical token IDs and text equal
  to `"ha" * 64` or `"." * 64` both returned exit 0. The reported word counts
  were respectively 1 and 0, although the token stream had 100% adjacent
  duplication. Fused subwords and punctuation can produce exactly this
  difference between decoded words and model tokens.
  Why this matters: single-token collapse is explicitly named in the runner
  and full-model contract. The normal autoregressive artifact supplies token
  IDs, but the repair discards that evidence whenever a text file is present.
  It also lets stale clean text mask collapsed current token IDs.
  Required next step: measure available token IDs for mechanical duplication
  even when text is present, while retaining useful word/phrase checks. Add
  non-whitespace token-collapse and simultaneous text/token-ID regressions;
  verify the retained model artifacts still pass.

## Other Concerns

- The missing-entry-point diagnosis is supported. The failed runner log is
  precisely Python's missing-file error; the work log's earlier invocation
  used `/tmp/gemma4-readiness.T6WBn5/.../check_degenerate_output.py`. The new
  standalone standard-library entry point fixes that packaging failure.
- The exact runner command currently exits 0 on real evidence. It discovers
  exactly the target model's chat and raw-story autoregressive artifacts,
  reports 64 HF/TT IDs in each, and passes the context check. All five focused
  tests pass. These positive results do not cover the two false-pass cases.
- No new contradiction was found in the preserved model implementation or
  hardware evidence. All six provenance hashes match the live model,
  generator, tests, decoder implementations, and precision loader. The
  retained profile JUnit has three tests and zero failures/errors/skips;
  TP1/2/4 prefill and teacher-forcing JSON each report top-5/top-100 of 1.0.
  Profile capacity and long-context artifacts remain present. This repair
  does not change those model/device paths.

## Hard-Check Gaps

- The runner checks the two `autoregressive_meta.json` artifacts, not the
  six-prompt suite's separate output schema. I inspected the six HF/TT
  output pairs directly; their metadata records the pinned Gemma tokenizer,
  chat template, greedy generation, and per-request reset. The retained
  prompt-source hash matches. Missing automatic suite discovery alone does
  not require a new model run.
- The maximum-common-trigram metric does not detect every long phrase loop:
  the visibly looping raw TT story scores only 0.2941. This case is controlled
  by its equally repetitive HF raw-completion output and is explicitly
  excluded from the instruct-model quality verdict. Broader phrase detection
  is a possible improvement, not an additional required repair here.
- The context subcheck reports the top-level 262,144-token capability; the
  per-profile 50,624/262,144/262,144 limits and TP1 physical-capacity reasoning
  reside in the retained context contract and stage evidence. A passing
  scalar check alone does not validate all profile construction results.

## Anomaly Ledger

- Observed anomaly: runner could not open its checker.
  Evidence: `06-06-full-model.check-1.log` and prior temporary-checkout command.
  Affected path: runner entry point.
  Control or comparison: exact in-tree runner command now exits 0.
  Likely subsystem: readiness checker packaging.
  Investigation performed: compared runner path, failed log, repair, and
  current execution.
  Resolution: fixed; checker semantics still need the two repairs above.

- Observed anomaly: metadata without generated TT content passes; repeated
  model token IDs can be hidden by the text-first path.
  Evidence: the host reproductions recorded in Required Work.
  Affected path: critical generation-evidence and mechanical-collapse gate.
  Control or comparison: valid artifacts and the existing doubled-word test
  pass/fail as expected, isolating the uncovered cases.
  Likely subsystem: artifact validation and measurement-source selection.
  Investigation performed: real checker functions and CLI control flow with
  in-memory artifact inputs; no model or device execution.
  Resolution: more-work-needed.

- Observed anomaly: raw story repeatedly emits the oak-tree sentence;
  several chat answers end mid-sentence and haiku includes `<turn|>`.
  Evidence: direct HF/TT completion inspection under `artifacts/qualitative`.
  Affected path: raw continuation stress and capped chat comparison outputs.
  Control or comparison: HF raw story also loops; both sides use the same
  pinned checkpoint and 64-token budget. Both haiku controls include the
  end-of-turn marker. Main chat agrees through token 43, then remains
  coherent; it matches the shared suite's corresponding TT token array.
  Likely subsystem: prompt mode, generation budget, and token decoding.
  Investigation performed: inspected prompts, rendered chat templates,
  metadata, actual text, token counts, and control outputs.
  Resolution: controlled; no new model-quality repair required.

## Scope Inspected

- Goal/skill paths: `bringup/artifacts/multigoal-runs/gemma4-26b-a4b-p150/06-06-full-model.prompt.txt`;
  `.agents/skills/{stage-review,full-model,tt-device-usage,qualitative-check}/SKILL.md`.
- Artifact paths: original failed-check log; full-model README, work log,
  previous review, provenance, profile JUnit, readiness metrics, capacity and
  long-context JSON; `doc/context_contract.json`; chat/story metadata and
  completions; six-prompt HF/TT outputs, prompt metadata, assessments, and
  retained prompt source.
- Code paths: repaired checker and focused tests;
  `.agents/prompts/model_bringup_multigoal/06-full-model.check.sh`;
  `.agents/scripts/check_context_contract.py`; model-source provenance hashes.
- Reviewed checker SHA-256:
  `55f86651f750f267c0bddeb970b099f3d5e5a6b07eb7d6e7363d125db45cffd6`.
  Reviewed focused-test SHA-256:
  `59be37757b2a220a959ed4ec58b9697bff8a7f0375fad9a2b89814f9ea242495`.
- Commands run: read-only `git status`, `git diff`, `git log`, `rg`, `sed`,
  `nl`, `jq`, and small Python artifact/hash/XML analyses; exact runner via
  `env PYTHONDONTWRITEBYTECODE=1 MODEL_DIR=models/autoports/google_gemma_4_26b_a4b_it HF_MODEL=google/gemma-4-26B-A4B-it bash .agents/prompts/model_bringup_multigoal/06-full-model.check.sh`;
  `env PYTHONDONTWRITEBYTECODE=1 python_env/bin/pytest -q -p no:cacheprovider --noconftest models/common/readiness_check/test_check_degenerate_output.py`
  (5 passed); in-memory `unittest.mock` artifact probes and direct
  `check_completion` token-collapse probes.

## Residual Risk

- No hardware, server, or vLLM experiment was run by this reviewer. Existing
  profile and trace evidence was not reopened without a concrete
  contradiction. The two required repairs are host-only checker changes.
- Retained model outputs are short qualitative comparisons, not broad model
  quality certification. Their coherence and controls support the narrow
  full-model evidence claim.
- After the checker regressions are fixed, rerun the focused tests and exact
  gate, obtain a clean rereview, and perform the stage's already-required
  local checkpoint/logging step.
