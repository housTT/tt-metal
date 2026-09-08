# AutoDebug: presence-penalty output variation

Initial source-only diagnosis, 2026-09-08, written before runtime or test
changes. Subsequent measured controls are recorded below.

## Failure

`full_b32_sampling_full_v1.log` records 69 passing tests, three failures, and
one skip. This investigation owns the two presence failures in
`tests/tt/test_tt_penalties.py`; the bad-words failure is separate.
The prompt `a b c a b c a b c`, greedy decoding, and 40 output tokens produced
the same continuing ` a b c ...` sequence for penalties -1.5 through 2.0 and
for the mixed 0/2 batch. Repetition and frequency suites passed.

## Source findings and hypotheses

The real common sampler enables its penalty path when presence is nonzero.
`SamplingGenerator._run_sampling` applies penalties before sampling and updates
generated-token counts afterward. `TTPenalties.apply_penalties` subtracts
`presence * output_mask`; its output mask comes from accumulated generated
counts, not prompt frequency. This matches the pinned vLLM host implementation.
No source branch obviously drops presence while retaining frequency.

H1 remains possible: device presence parameters/history are wrong in the
serving path. An exact-shape synthetic sampler probe can force a known top-1
crossing after one generated occurrence and verify row-specific subtraction.

H2 remains possible: this prompt does not expose the intended behavior. Once
all three cycle tokens have appeared, presence subtracts the same constant
from all three. Their internal ordering stays unchanged. A positive penalty
of 2 only lets an unseen token win if its unpenalized logit lies within 2 of
the cycle winner. Frequency penalties keep growing and can change output even
when presence is correctly constant. This is a mathematical possibility, not
yet a finding about the model or a reason to waive the failing assertions.

## Required focused controls

`presence_distribution_probe.py` sends the exact failing prompt through both
device sampling and explicit host-logprobs compatibility, at neutral, -1.5,
and 2.0 presence. The host sampler is constructed as `Sampler()` and therefore
returns raw logprobs by default. The probe records token IDs and raw top-20
logprobs, reapplies the presence formula on CPU for each actual generated
history, and bounds every unreturned token by the raw top-20 cutoff. It can
therefore certify when a constant penalty cannot change the argmax, rather
than inferring that from repeated text alone. Host/device disagreements stay
visible and fail the control; inconclusive bounds remain explicitly unknown.

If the original prompt is mathematically insensitive in the host control,
test a shorter version (`a b c`) with the same controls before proposing a
prompt correction. Any test change must retain deterministic cohorts,
mixed-row isolation, and an actual observed penalty-driven change. No runtime
or existing test was changed during the initial diagnosis.

Prompt mode: raw `/v1/completions`, deliberately labeled continuation stress.
The checkpoint is `ornith-ai/Ornith-1.5-9B`, pinned revision
`489cb97981b8654bcfcf30ce1f94ed1b62e07b53`, whose chat template is present.
These raw probes provide sampling-mechanism controls, not chat-quality verdicts.
All HTTP/hardware execution remains in the supervising lane.

## Measured original-prompt control

The supervising all-layer B32 server ran
[`presence_distribution_v1.json`](presence_distribution_v1.json) and
[`presence_distribution_v1.log`](presence_distribution_v1.log). All six
40-token requests succeeded. Device token IDs exactly matched host sampling
at presence 0, -1.5, and 2.0; all outputs followed the same three-token cycle
`[264, 292, 272]`. Every one of the 120 host decode choices obeyed the CPU
presence formula, and every choice was certified against the bound on all
unreturned vocabulary tokens. No presence setting changed the raw argmax.

At the narrowest positive-penalty decision, output position 3 (zero based),
the repeated raw winner exceeded the best unseen token by **3.31250003**
logits. Subtracting 2 therefore left a **1.31250003** advantage. This was the
smallest adjusted winner margin over any returned competitor across all 40
positive-penalty steps. The smallest margin over the upper bound for every
unreturned candidate was **3.99999979**. These are observed distribution
margins, not a claim inferred from repeated text.

Thus H2 is confirmed for this checkpoint and prompt: the assertion that
presence must vary this continuation is false even under the mathematically
correct host implementation. A device presence implementation error is not
established by this failure. Host/device equality on an insensitive prompt
alone does not establish that the device subtracts the penalty correctly;
the sensitive prompt and synthetic canonical sampler controls below address
that remaining distinction.

The shorter `a b c` control also remained insensitive:
[`presence_distribution_short_v1.json`](presence_distribution_short_v1.json).
All six requests matched host/device; all 120 host choices were formula
correct and certified, with no adjusted argmax changes. Its alphabet/digit
continuation therefore does not repair the test stimulus.

## Sensitive stimulus and proposed correction

The supervising lane screened four natural continuations at presence 0 and
2 using concurrent device requests, 40 tokens, and greedy sampling:
[`presence_prompt_sensitivity.json`](presence_prompt_sensitivity.json).
All four showed differing token IDs. `Once upon a time` is the simplest
fixture and uses the same raw completion mode as the original tests. Its
two outputs share the first 16 tokens, then differ at position 16: neutral
chooses token 264 (` a`), while presence 2 chooses token 279 (` the`).

The authorized minimal correction changes only the two presence test prompt
strings to this measured-sensitive continuation. Penalty values, output
length, temperature, varied-output assertions, and deterministic mixed-row
cohort assertions stay intact. This preserves an observable penalty effect
and request isolation, without a model-specific skip or weaker assertion.
The original prompt and its certified no-change evidence remain archived
as a regression control in this report and distribution probe.

The full replacement-prompt control then passed:
[`presence_distribution_sensitive_v1.json`](presence_distribution_sensitive_v1.json)
and [`presence_distribution_sensitive_v1.log`](presence_distribution_sensitive_v1.log).
Device and host token IDs match at all three penalties, and both paths vary
their output. All 120 host choices are formula correct and full-vocabulary
certified. Presence 2 changes the raw argmax at positions **16, 18, 21, 33**;
presence -1.5 changes it at position **20**. At the first positive-penalty
crossing, repeated token 264 exceeds unseen token 279 by only **0.75000012**
raw logits; subtracting 2 correctly makes token 279 win. Thus the replacement
fixture exposes a verified mathematical presence effect in both serving
sampling modes, rather than merely a nondeterministic output difference.

The supervising lane then ran `tests/presence_sampler_device_probe.py`:
[`presence_sampler_exact.json`](presence_sampler_exact.json) and
[`presence_sampler_exact.log`](presence_sampler_exact.log), exit 0. The
actual traced split canonical sampler used 32 mixed lanes, real vocabulary
248320 padded to 262144, and candidates on all four shards. Both generated
count 1 and count 7 produced exact row-specific logit subtraction and the
expected two-step top-1 crossings: presence-2 lanes chose candidate B then C;
neutral, -1.5, and 0.5 lanes chose A twice. The two initial counts yielded
identical outputs, proving presence applies once. Trace-bound logits,
feedback, history, and seed addresses stayed fixed, and reset operations
introduced no new compiled programs after capture.

Together, the source, actual serving distribution, and forced canonical
sampler controls refute H1 for this observed failure and identify an
insensitive test stimulus as the cause. Both corrected canonical presence
tests then passed in [`full_b32_targeted_final.log`](full_b32_targeted_final.log)
(three-test run, 3 passed in 32.13 seconds). The full B32 canonical suite also
passed, 72 passed and one skipped in 329.46 seconds:
[`full_b32_sampling_before_penalty_order.log`](full_b32_sampling_before_penalty_order.log).
That run predates a separately diagnosed combined-penalty ordering repair;
the latter has its own post-fix verification requirements.
