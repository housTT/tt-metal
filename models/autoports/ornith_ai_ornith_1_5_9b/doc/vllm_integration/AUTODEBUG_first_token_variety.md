# AutoDebug: first-token variety checks inspect characters

Date: 2026-09-08. Source/CPU-only investigation; parent serializes all serving
requests. This report precedes implementation changes.

## Starting evidence

The current final-server sampling run reports 71 passed, 1 failed, 1 skipped.
Its failure is
`TestSeedingAndVariety.test_temperature_varied_in_batch[7]` in
`readiness_vllm/sampling_tests.log`. The test sends seven concurrent legacy
completion requests with prompt `Random letter: `, maximum ten output tokens,
temperature 5, and no seed or explicit top-k. The full-output variety assertion
passes. The purported first-token assertion fails on seven string values
equal to one ASCII space.

## Verified source defect

`run_concurrent_batch` returns decoded strings by default.
`test_seeding_and_variety.py` then builds its purported first-token results
using `x[:1]`. On a Python string, this extracts the first character, not the
first generated tokenizer token. Multiple distinct vocabulary tokens can
decode to strings beginning with a space, so this cannot test prefill token
variety. The same mismatch occurs in five conceptual checks: seeding,
temperature within a batch, temperature across batches, top-k within a batch,
and top-k across batches (two list expressions in the last check).

The original failing log does not contain generated token IDs. Therefore it
does not establish whether the original seven first tokens differed. The
source assertion is independently invalid regardless of that unknown.

## Focused experiment and proposed repair

The parent will run `first_token_variety_request_probe.py` once. It repeats
the original seven request parameters and adds only `return_token_ids=true`
response metadata. It saves all raw responses, actual first token IDs,
individual-token decoded pieces, first characters, and both variety verdicts.
No logprob request or other sampling change is added. Do not infer generated
token IDs by re-encoding returned text, and do not retry until a random output
happens to pass.

If source/CPU and live metadata evidence confirm the mismatch, use actual API
IDs for all five first-token assertions. Preserve existing full-output
variety, deterministic seed, paired variety, batch shape, and thresholds.
Require the expected response count, nonempty output, and nonempty actual
token-ID metadata. Add CPU negative controls showing that identical actual
first IDs fail even when later text varies, while distinct IDs with identical
leading characters pass the intended token-variety check.

## CPU confirmation and applied correction

A CPU regression compiles the actual canonical test class and substitutes only
the HTTP-request function. Seven responses with distinct supplied first token
IDs and the same leading space reproduce the original assertion failure:
**1 failed, 1 passed** before the fix (`first_token_variety_cpu_before.log`).
The second test is a negative control whose first IDs are identical while
later output text varies.

The parent authorized the source-verified correction while unrelated serving
benchmarks were running. All five conceptual first-token checks now use actual
API `token_ids[0]`. The helper preserves original decoded strings for existing
full-output checks, requires response count/nonempty text/nonempty token
metadata, and changes only response metadata. All sampling parameters,
thresholds, batch sizes, trial counts, paired comparisons, and full-output seed
checks remain. The seeded check additionally verifies identical actual first
IDs across the same seed's positions and runs.

Expanded CPU controls exercise the actual temperature, top-k, and seeded test
methods, reject same first tokens despite varied later output, reject a
pairwise first-token mismatch failure being hidden by varied full text, reject
missing metadata/empty completions, and verify token metadata is explicit and
optional in the completion helper. **11 tests pass**
(`first_token_variety_cpu_after.log`). Black, byte-compilation, and diff
whitespace checks pass for the touched code.

## One-shot live diagnostic

The parent ran the diagnostic once on the real server. It completed with exit0
and saved `first_token_variety_original_probe.json`/`.log`. The seven actual
first IDs were `[326,710,318,318,357,15,23]`, decoding individually as
`[" S"," K"," ("," ("," A","0","8"]`.

Thus five requests shared the same leading space while containing **four
distinct first token IDs**. This confirms the character/token distinction in
actual serving output. Both character and token variety passed in this new
batch because two outputs began with `0` and `8`; the original all-seven-space
failure shape was **not** exactly reproduced. The diagnostic was not repeated
to obtain a preferred result, and the original failing run's token IDs remain
unknown. Its source assertion defect was already established independently by
the CPU reproduction.

Status: source/CPU defect fixed and the distinction verified with actual live
token metadata. The parent is running the corrected original canonical node,
followed by the broader sampling gate. No runtime implementation or
sampling-policy change was made.

## Continuation: valid immediate EOS rejected by the new helper

The next full suite reported 69 passed, 3 failed, 1 skipped. All three failures
are `test_topk[15,19,32]` in the new helper's `assert choice.text`, before the
existing full-output or first-token variety checks. Each inspected failed
response has `text=''`, `finish_reason='stop'`, and actual
`token_ids=[248046]`. The complete failure is preserved as
`first_token_eos_sampling_failure.log`.

This nonempty-display assertion was introduced with the token-ID helper; the
original top-k test did not impose it. CPU inspection of the pinned tokenizer
confirms its EOS ID is 248046 (`<|im_end|>`) and decoding that token with the
default special-token skipping produces the empty string. Evidence:
`first_token_eos_tokenizer_control.json`. Thus an actual token was sampled and
the request terminated correctly; the helper incorrectly treats its empty
display text as absent generation.

Hypothesis: accepting an empty display **only with terminal EOS evidence**
repairs this helper regression without weakening token/output variety checks.
Proposed minimal repair: retain one response per request and nonempty token-ID
metadata; require string text; permit an empty string only when
`finish_reason == 'stop'` and the final actual generated ID equals the served
model tokenizer's configured EOS ID. Obtain the EOS via cached tokenizer lookup,
not a model-specific hardcoded number. Keep empty non-EOS, missing metadata,
and inconsistent finish reasons as failures. Do not suppress EOS generation,
change temperature, or lower variety thresholds.

This continuation diagnosis is recorded before changing the helper. Focused
CPU controls will reproduce valid EOS rejection, then test mixed EOS/text
results and negative cases through the actual canonical methods. The parent
will rerun targeted top-k nodes and the full serving suite after the repair.

The focused mixed-EOS CPU control reproduced the exact helper rejection at
response9 before the repair: **1 failed, 11 passed**
(`first_token_eos_cpu_before.log`). The helper now accepts empty display text
only with `finish_reason='stop'` and a final actual ID equal to
`cached_get_tokenizer(tt_model_name).eos_token_id`. Response count, string text,
and nonempty token IDs remain mandatory. No token number is hardcoded in the
helper; a CPU control with a different configured EOS proves that boundary.

Expanded controls now report **17 passed** (`first_token_eos_cpu_after.log`).
They exercise the actual canonical methods, including mixed immediate-EOS/text
top-k batches, rejection of empty non-EOS outputs, wrong finish reasons,
missing text/metadata, and rejection of an all-EOS batch by the original
variety assertion. Existing seeded, pairwise, and first-token negative controls
remain. Formatting, byte-compilation, and diff checks pass. The parent owns
the targeted top-k and full-suite reruns; no runtime/model changes were made.
