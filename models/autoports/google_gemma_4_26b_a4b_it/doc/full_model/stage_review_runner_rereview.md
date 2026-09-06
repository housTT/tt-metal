# Stage Review

Verdict: clean-pass

Stage: 06 full-model, runner-side failed-check repair. Independent host-only
review on 2026-09-06 of the live `hous/gemma-4-26b-a4b-it` worktree at
`2fb82badbecc6048760038adcea171c92d299de1`, including the uncommitted checker
repair. This verdict applies to the final hashes recorded below, after the
stage owner repaired an additional schema-validation issue found during review.

## Required Work

- None. Both first-review findings and the additional reproduced malformed-
  artifact false passes are fixed in the final reviewed implementation.

## Other Concerns

- Missing/empty TT evidence fails the runner's critical policy. Independent
  CLI-control-flow probes return exit 2 for missing `tt`, an empty `tt`
  object, and empty `tt.token_ids` with no text. Empty content returns exit 1
  under the advisory policy. Valid token-ID-only and text-only artifacts
  still pass when their enclosing metadata has the expected structure.
- Token IDs are measured even when nonblank decoded text exists. Sixty-four
  identical IDs return exit 2 with fused `ha` text, punctuation-only text,
  and otherwise clean varied words. The retained chat/story artifacts each
  produce both word and token-ID measurements, with zero adjacent duplication.
- Malformed JSON schemas no longer masquerade as evidence. JSON `null`,
  string/object token IDs, and arrays containing null, boolean, negative, or
  fractional IDs all return exit 2 in independent host probes. Invalid JSON
  returns exit 2; an invalid CLI argument returns exit 3. A distinct load-
  failure sentinel prevents JSON null from taking the successful empty-return
  path that was observed earlier in this review.
- The final focused suite passes all 11 tests, and the exact stage-06 runner
  exits 0. Model-directory discovery finds exactly the target chat and raw
  story metadata; HF-model resolution selects exactly this autoport. The
  context subcheck also passes.
- The shared vLLM schema changes preserve a valid completion's clean result;
  independent probes reject null/invalid schemas and missing completions.
  Blank-string output retains an advisory result. This inspection found no
  unintended regression relevant to the autoregressive stage gate.
- No new contradiction was found in the settled model evidence. All six
  implementation/test/precision hashes in `artifacts/provenance.json` match
  live files. Retained profile JUnit records three tests with no failures,
  errors, or skips. All six TP1/2/4 prefill and teacher-forcing metric files
  report top-5/top-100 of 1.0. Each full-model context profile's referenced
  capacity, stack, long-context, and applicable batch artifact exists.

## Hard-Check Gaps

- The runner discovers `autoregressive_meta.json`, not the shared six-prompt
  suite's separate schema. Direct review of all six HF/TT output pairs and
  their pinned tokenizer/chat-template metadata supplies that qualitative
  evidence. The retained prompt-source hash matches its metadata.
- Trigram frequency is a limited phrase-loop heuristic. The raw story visibly
  repeats a longer phrase despite remaining below the advisory threshold;
  its HF raw-completion control also loops, and the stage explicitly treats
  this as continuation stress rather than the instruct-model quality verdict.
- The scalar context checker reports 262,144 tokens. The retained contract
  separately records the P150 physical-capacity limit of 50,624 and full
  262,144-token support on P150x2/P150x4. The scalar pass alone is not proof
  of those distinct profile construction results.

## Anomaly Ledger

- Observed anomaly: the runner could not open its checker entry point.
  Evidence: `06-06-full-model.check-1.log` records Python's missing-file error;
  the previous work log command used a temporary historical checkout.
  Affected path: runner-side checker packaging.
  Control or comparison: the unchanged exact runner now executes the in-tree
  standalone checker and exits 0 on retained evidence.
  Likely subsystem: removed readiness-package entry point.
  Investigation performed: inspected the goal, runner, failed log, restored
  file, live diff, and final exact invocation.
  Resolution: fixed.

- Observed anomaly: empty TT payloads passed and decoded text hid collapsed IDs.
  Evidence: first review, current regression tests, and independent missing-
  content/fused-text/punctuation/clean-text probes.
  Affected path: required free-running evidence and token-collapse validation.
  Control or comparison: valid ID-only/text-only artifacts pass; absent content
  and identical-ID collapse return the required nonzero status.
  Likely subsystem: missing-content validation and measurement selection.
  Investigation performed: repeated the first-review cases against final code.
  Resolution: fixed.

- Observed anomaly: JSON null and malformed nonempty token-ID values passed.
  Evidence: during this review, checker hash
  `498b8f365c885ab99c91c5bfc94fb58985e440e6bdc215f30b2953c6c4413425`
  returned exit 0 for null, string/object IDs, and `[null]` with critical
  missing-artifact policy and no completion text.
  Affected path: autoregressive artifact schema validation.
  Control or comparison: final code returns exit 2 for every reproduced case;
  valid evidence still passes. Four dedicated regression cases were added.
  Likely subsystem: ambiguous load-failure sentinel and unchecked JSON types.
  Investigation performed: reported the cases to the stage owner, inspected
  the owner's remediation, then reran focused tests, independent type probes,
  exit-code controls, and the exact runner against the final hashes.
  Resolution: fixed.

- Observed anomaly: raw story repetition, capped mid-sentence chat answers,
  and the haiku's `<turn|>` marker.
  Evidence: actual HF/TT chat/story and shared-suite completion files.
  Affected path: raw continuation and 64-token qualitative comparisons.
  Control or comparison: HF raw output also loops; both sides share the pinned
  checkpoint, prompt format, and generation cap. Both haiku outputs include
  the marker. Chat responses remain coherent and on task after divergence.
  Likely subsystem: prompt mode, generation budget, and decoded special tokens.
  Investigation performed: inspected actual text, token counts, rendered chat
  prompt, shared-suite metadata, and prompt-source provenance.
  Resolution: controlled; no new model-quality repair required.

## Scope Inspected

- Goal/skill paths: `bringup/artifacts/multigoal-runs/gemma4-26b-a4b-p150/06-06-full-model.prompt.txt`;
  `.agents/skills/{stage-review,full-model,tt-device-usage,qualitative-check}/SKILL.md`.
- Artifact paths: original failed runner log; full-model README and work log;
  `stage_review_runner_recheck.md`; provenance, final profile JUnit, six readiness
  metric JSONs; `doc/context_contract.json` and referenced artifact existence;
  chat/story metadata and HF/TT completions; six shared-suite output pairs,
  prompt-format metadata, and retained prompt source.
- Code paths: `models/common/readiness_check/check_degenerate_output.py`;
  `models/common/readiness_check/test_check_degenerate_output.py`;
  `.agents/prompts/model_bringup_multigoal/06-full-model.check.sh`; provenance
  hashes of the model, generator, full-model tests, decoder implementations,
  and precision loader.
- Final checker SHA-256:
  `3833298564cfc70b997d8055d8c6916219cd811ef5ecb1e363288d15ae3a0fee`.
- Final focused-test SHA-256:
  `0651e536bcded3a7f0bdb6b9bf6302df4304c830edbd519eb32b973a317d42af`.
- Commands run: read-only git status/diff/branch/revision checks, file searches
  and reads, SHA-256/JSON/XML analysis, and in-memory `unittest.mock` CLI probes.
  Final focused command:
  `env PYTHONDONTWRITEBYTECODE=1 python_env/bin/pytest -q -o addopts= -p no:cacheprovider --noconftest models/common/readiness_check/test_check_degenerate_output.py`
  (`11 passed`). Exact runner:
  `env PYTHONDONTWRITEBYTECODE=1 MODEL_DIR=models/autoports/google_gemma_4_26b_a4b_it HF_MODEL=google/gemma-4-26B-A4B-it bash .agents/prompts/model_bringup_multigoal/06-full-model.check.sh`
  (exit 0). `git diff --check` passed. An earlier seven-test invocation used
  repository pytest defaults, including its generated JUnit report; no
  implementation file was modified by the reviewer.

## Residual Risk

- No TT device, hardware test, reset, server, reservation, or vLLM experiment
  was run by this reviewer. The repair changes host-side validation; preserved
  device and performance evidence remains the applicable model record.
- The heuristic checker and short qualitative controls do not certify broad
  model quality. They support the narrow free-running evidence requirement
  together with the retained profile, trace, accuracy, and capacity artifacts.
- The stage owner still performs the prescribed local checkpoint and records
  its SHA after this clean review; no push is authorized by this report.
