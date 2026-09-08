# AutoFix: first-token variety assertion

Date: 2026-09-08.

## Starting evidence

`AUTODEBUG_first_token_variety.md` and the 71-pass/1-fail/1-skip final-server
sampling log. The failed seven-request check compared seven leading spaces
while describing them as first generated tokens. Full-output variety passed.
The original generated IDs were not logged and cannot be recovered from those
seven characters.

## Hypothesis experiment

Hypothesis: the canonical tests use first decoded characters where their
contract requires first tokenizer token IDs.

The CPU regression executes the actual canonical temperature test with seven
responses whose actual first IDs differ but whose decoded text begins with
the same space. Before the correction, it fails the same first-token assertion;
the initial run reports 1 failed/1 passed. Artifact:
`first_token_variety_cpu_before.log`.

Verdict: source assertion defect verified independently of the unknown token
IDs in the original serving failure.

Fix: request `return_token_ids` response metadata only for tests requiring
first-token comparisons and check the actual first ID. Preserve decoded text
for existing full-output checks. Preserve five conceptual variety checks,
thresholds, request counts, sampling parameters, seed checks, and paired
comparisons; add response-count/nonempty-output/metadata checks and explicit
first-ID seed consistency. The existing chat metadata option is unchanged.

Verification command from the tt-metal root:

```sh
USER=hous ../state/serving-env/bin/python -m pytest ../vllm/plugins/vllm-tt-plugin/tests/test_first_token_variety.py -q
```

Result: **11 passed**, recorded in `first_token_variety_cpu_after.log`.
Negative controls reject identical first IDs despite varied later text,
identical first IDs across top-k reruns despite pairwise full-text changes,
changed first IDs under a repeated seed, and missing/empty response data.
The metadata option is checked both enabled and disabled. Source formatting,
byte-compilation, and diff whitespace checks pass.

## Parent-owned live verification

One diagnostic run, with the exact original seven request parameters and only
token-ID response metadata added:

```sh
USER=hous ../state/serving-env/bin/python models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/first_token_variety_request_probe.py --output models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/first_token_variety_original_probe.json
```

The parent ran this diagnostic exactly once. It completed with exit0 and
recorded actual IDs `[326,710,318,318,357,15,23]`, whose individual decoded
pieces are `[" S"," K"," ("," ("," A","0","8"]`. Five leading spaces
therefore represented four distinct actual tokens. This live batch passed
both character and token variety because two other outputs began with `0`
and `8`. It did **not** reproduce the original all-seven-space failure shape;
that original run's token IDs remain unknown. The source defect is established
by the CPU reproduction and actual token/character distinction, without a
retry-until-pass strategy. Evidence:
`first_token_variety_original_probe.json` and `.log`.

The script has no retry loop and does not ask for logprobs or change sampling.
The parent is running the corrected original canonical node and broader suite.

## Continuation: terminal EOS has valid empty display text

The subsequent full suite reported 69 passed, 3 failed, 1 skipped. All failures
were in the new helper's added nonempty-display assertion, not the retained
variety checks. The failed responses had `finish_reason='stop'`, `text=''`, and
actual IDs `[248046]`. This assertion was stronger than the original test and
incorrectly rejected valid immediate EOS. The complete failure is archived in
`first_token_eos_sampling_failure.log`.

CPU tokenizer inspection confirmed 248046 is the pinned tokenizer's EOS and
decodes to an empty string when special tokens are skipped
(`first_token_eos_tokenizer_control.json`). A new regression through the actual
canonical top-k method reproduced the helper's response9 failure before the
repair: 1 failed, 11 passed (`first_token_eos_cpu_before.log`).

The helper now requires a response, string text, and nonempty actual token IDs;
an empty string additionally requires `finish_reason='stop'` and a terminal
actual ID equal to the model tokenizer's configured EOS. It uses cached
tokenizer lookup and has no model-specific token constant. No request sampling
parameters, prompts, thresholds, or retries were changed. EOS remains in the
actual first-token variety calculation and empty decoded strings remain in
the original full-output comparison.

Afterward, **17 CPU tests pass** (`first_token_eos_cpu_after.log`). Negative
controls reject empty non-EOS output, wrong terminal reason, missing metadata
or text, and an all-EOS batch with insufficient variety. Positive controls
accept mixed EOS/text outputs and a different configured tokenizer EOS.
Formatting, byte-compilation, and diff checks pass.

## Final status

Source/CPU assertion defect fixed; live token metadata confirms the semantic
distinction. Canonical rerun is pending. This report does not classify the
original seven actual token IDs as varied. The newly introduced empty-display
assertion defect is fixed with focused CPU evidence; the parent owns the
targeted top-k and full-suite serving reruns.
No runtime model, adapter, sampler, or precision changes were made by this
repair. No tests were skipped or thresholds reduced.
