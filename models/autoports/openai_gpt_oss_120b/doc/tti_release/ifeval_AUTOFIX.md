# IFEval release-gate AutoFix

## Problem

The initial Stage 11 handoff omitted the mandatory `meta_ifeval` gate. A
literal Meta recipe was not a valid repair: Meta documents that recipe as
Llama-3-only, and its prompts embed Llama-specific tokens. The installed
common lm-eval environment instead provides canonical task `ifeval` over
`google/IFEval` (541 samples).

TTI also coupled task selection to sample limiting and used one task name for
both the public release identity and lm-eval execution identity. Those issues
made it impossible to request only the canonical full gate while truthfully
reporting `meta_ifeval`.

## Harness repair

- Added explicit `lm_eval_task_name` plumbing: the TTI selector/report key is
  `meta_ifeval`, while lm-eval executes `ifeval`; the report records both.
- Added independent `--eval-tasks` selection. It can select `meta_ifeval`
  without implicitly adding either `--limit` or `--samples`.
- Made the configured prompt-level strict metric fail closed for this gate;
  an unrelated numeric metric cannot replace it.
- Made a nonzero lm-eval subprocess exit authoritative even if a stale or
  partial result file contains a pass-looking block.
- Extended the release merger to require canonical task/dataset provenance,
  full 541/541 counts, a null limit, exact score agreement, and the intended
  generation/context policy without loading or copying sample text.
- Made the merger reject both `NA` and `FAIL` accuracy rows; only an
  explicit `PASS` can satisfy mandatory IFEval.
- Standardized the generated readiness value to
  `release-readiness-ci-subset-pass`.

Focused argument, routing, scoring, subprocess-failure, and merger tests pass.
An independent AutoFix rereview returned clean-pass on the
metric/exit/limit repairs. The final host regressions passed 299 TTI tests and
17 merger tests.

## Configuration-parity repair

The first full canonical run completed 541/541 in 1h54m58s, but its row failed:

- prompt-level strict: 397/541 = 73.3826%;
- required threshold: 74.29% (78.2% public reference with 5% tolerance);
- configuration: explicit low reasoning and `max_gen_toks=1280`.

The workflow-level experimental acceptance mask was rejected; this row was
treated as blocking and was not waived.

The cited public reference is also full 541-sample canonical IFEval, but uses
`max_gen_toks=4096` and no reasoning-effort override. Both the official GPT-OSS
checkpoint template and official vLLM Harmony path default an absent effort to
medium. The corrected candidate therefore uses explicit medium effort and
4,096 output tokens while retaining deterministic sampling, seed 42, and
`max_length=131072`.

A fixed 20-ID subset of strict failures isolated the change without printing
or copying responses:

| Configuration | Strict passes | Nonempty finals |
| --- | ---: | ---: |
| low + 1,280 (original full-run rows) | 0/20 | 11/20 |
| low + 4,096 (budget isolation) | 6/20 | 20/20 |
| medium + 4,096 (reference-parity candidate) | 11/20 | 16/20 |

The budget-only probe recovered six strict failures—enough to demonstrate that
1,280 changed the gate result—and the reference-parity candidate recovered
eleven. The earlier eight-way concurrency probe was separately refuted: its
first eight requests took 7m24s, materially slower than serial execution, so
the owned client was stopped safely and the server remained healthy.

## Final confirmation

The authoritative full medium+4,096 run completed 541/541 requests in
3h37m20s and exited 0:

- prompt-level strict: 463/541 = 85.58225508317929%;
- required threshold: 74.29%;
- TTI row: explicit `PASS`, logical task `meta_ifeval`, canonical task
  `ifeval`;
- release merger: `acceptance=PASS`, `blocker_keys=none`.

The response-free lm-eval aggregate is `ifeval_aggregate_results.json`; the
matching runtime spec is `runtime_model_spec_validation.json`; and the final
four-gate report is `report_data_release_ci_nightly.json`. The pre-repair TTI
source report remains in the bounded TTI work area rather than this customer
handoff because its original harness timing used the stale 28-sample
denominator. The merged report recomputes timing from the authoritative
aggregate total and 541-sample count. Raw per-sample JSONL data remains only in
the TTI cache and is not part of this handoff.
