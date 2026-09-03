clean-pass

# Stage Review

Verdict: clean-pass

## Required Work

- None. The sole final-review finding—RUN_NOTES' incorrect 64-token smoke
  cap—was corrected to 8 tokens in tt-metal commit
  `076a5e46ff7fd470e2f54f98614edbd563abd069` and rereviewed against the
  request and response evidence.

## Other Concerns

- None requiring action.

## Hard-Check Gaps

- The checker did not catch the smoke-cap documentation mismatch; direct
  artifact comparison did. The corrected note now agrees with
  `smoke_chat_request.json` and `smoke_chat_response_summary.json`.
- The final report records SHA-256 bindings for GPQA's report/raw/runtime
  triplet but not the equivalent IFEval triplet. For this exact state, manual
  inspection closed that gap: the IFEval source report references the supplied
  runtime spec; the handoff copy is semantically identical apart from a trailing
  newline; run timestamps, raw aggregate, runtime settings, and server log
  agree.

## Anomaly Ledger

- Observed anomaly: stale IFEval source-report timing used the old 28-sample
  denominator.
  Evidence: authoritative raw duration is `13045.45777320399`; corrected mean
  is `24.11360032015525` over 541 samples.
  Resolution: stale source report removed from customer handoff; corrected
  report, aggregate, runtime spec, and explanatory notes retained.

- Observed anomaly: GPQA raw metadata reports the 198-row dataset size despite
  runtime selection of IDs `0..6`.
  Evidence: runtime spec and log show seven exact IDs and seven completed
  requests; duration `2246.9087665929983 / 7 = 320.9869666561426`.
  Resolution: final report uses the selected-request denominator and
  fail-closed raw/runtime/API/autoport/context/policy validation.

- Observed anomaly: one benchmark row has five visible target failures.
  Evidence: exact set is `functional.tput`, `complete.tput`,
  `target.tput`, `complete.ttft`, `target.ttft`.
  Resolution: three throughput failures are malformed B32-scaled aggregate
  targets attached to B1; two TTFT tiers are genuinely unmet and disclosed.
  The waiver is limited to that row, all requests completed, and unrestricted
  performance readiness is explicitly false.

- Observed anomaly: IFEval aggregate has top-level `max_length=131071`.
  Evidence: installed lm-eval deliberately stores configured `131072 - 1` to
  reserve one generation token; raw model arguments, canonical task metadata,
  TTI specs, server, and context contract all remain 131072.
  Resolution: benign harness bookkeeping, not a context reduction.

- Observed anomaly: generated logs contain routine TT/vLLM warnings, including
  trace-allocation caution and parser recovery.
  Resolution: no traceback or fatal condition occurred; workflows completed,
  health remained good, cleanup recorded graceful shutdown, and the post-run
  P150x4 mesh smoke passed.

## Scope Inspected

- Exact revisions:
  - tt-metal `076a5e46ff7fd470e2f54f98614edbd563abd069`
  - TTI `8459ba8dc6e690e7987235182a0d87c67bacc4a7`
  - official vLLM `54dea57d98ccfaef072908f085d9296d544ba1fe`
  - plugin `053c0782aa11028924c21cb061ffa76576705cad`
- Final report: acceptance true, no blockers; evals 4/4, spec tests passing,
  benchmarks 20 NA plus one narrowly waived row.
- Accuracy:
  - AIME `13/15`
  - GPQA `7/7`
  - MMLU `84.6732487070992%` over 2,127 effective examples
  - canonical full IFEval `463/541 = 85.58225508317929%`; the cited public
    comparison is 78.2% over the same 541-prompt benchmark.
- Benchmarks: all 21 raw artifacts uniquely match their aggregate rows, exact
  prompt/output lengths and counts, numeric zero errors, including both
  non-aligned ISL 10000 rows.
- Context: 131072 preserved throughout; `130944 + 128` boundary completed
  twice.
- API/spec: Logger Fork Safety and all 22 chat-conformance cases passed.
- Autoport proof: specs, runtime metadata, import origins, server
  initialization, and adapter imports identify
  `models/autoports/openai_gpt_oss_120b`; no stock implementation was
  selected.
- Handoff: 36 committed files, tracked-clean, no pycache, secrets, weights,
  caches, binary dumps, or raw sample/response payloads. Authoritative and
  copied reports are JSON-identical, differing only by a trailing newline.
- Test evidence: TTI 308 tests, final touched-surface 304 tests, merger 36
  tests, and focused vLLM repair 5 tests.
- Cleanup: no port-8000 listener or owned release process remains; only the
  documented pre-existing tmux session remains.
- Review used read-only filesystem/process inspection only; no hardware
  execution, server start, Docker use, edits, or commits.

## Residual Risk

- This is deliberately `release-readiness-ci-subset-pass`, not unrestricted
  full-set or performance readiness.
- The two higher TTFT tiers remain unmet and visible.
- Raw IFEval includes expected metric failures, including some responses
  without a final-answer channel; these are counted in the 78 strict failures
  rather than hidden as successful samples. The full canonical gate still
  exceeds its configured threshold.
- Hardware results were not rerun during this review; they were independently
  checked through committed reports, raw aggregates, logs, and cleanup
  evidence.
- tt-metal retains unrelated pre-existing untracked agent/cache and
  earlier-stage artifacts outside the clean committed TTI handoff; the TTI,
  vLLM, and plugin repositories are clean.
