# AUTODEBUG: residual full-36 vLLM sampling failures

Date: 2026-08-31

## Scope and disposition

This is a read-only source-and-artifact investigation of exactly the two
failures in `readiness_vllm/sampling_tests.log`. No TT device, live server, or
running process was touched, and no implementation file was changed.

The inspected snapshots are:

- tt-metal working tree at the current workspace state. Relevant common
  sampling/generator files are modified and the GPT-OSS adapter, readiness
  artifacts, and integration documentation are untracked, so all conclusions
  below describe this exact working tree rather than a clean commit.
- vLLM commit `568afb3a13806beb53bb2e6bd518269357b237c0`, clean.
- vLLM TT plugin commit `d7a6008b03c7afba001444f2d7a4cfde9ef6d498` with unrelated/current workspace
  modifications in `README.md`, `src/vllm_tt_plugin/platform.py`, and one
  registration test.

### Executive verdict

| Failure | Classification | Leading cause | Confidence |
| --- | --- | --- | --- |
| `TestBatchIsolation::test_mixed_params_batch` | Product reproducibility bug, localized but not yet proven to one instruction | Per-row presence-penalty state/logits are reconstructed incorrectly at a heterogeneous request-removal reset; a B>8 physical-row model/logit defect is the main alternative | Medium |
| Full-capacity structured choice (`blue<|return|>`) | Server/configuration bug, not TT device sampling | `enable_in_reasoning=true` applies the choice grammar before GPT-OSS emits a Harmony header, producing a bare choice plus the Harmony return token; the parser intentionally recovers that malformed sequence verbatim | High |

The shared choice assertion is a reasonable API expectation and should not be
weakened first. The mixed-params exact-equality assertion is also required by
the seeded vLLM contract. Neither failure justifies disabling or replacing the
normal traced on-device sampling path.

## Starting evidence

The full profile has only these two failures: 71 passed and one skipped
(`sampling_tests.log:269-271`). In particular:

- all logprob cases pass (`sampling_tests.log:35-45`);
- all dedicated seeded cases pass, including seeds across shuffled batch
  positions and uniform seeded batches at B32 (`sampling_tests.log:48-75`);
- all dedicated repetition, presence, and frequency penalty cases pass
  (`sampling_tests.log:78-83`).

The current server is vLLM 0.26.0 and advertises `max_num_seqs=32`, async
scheduling, `sample_on_device_mode=all`, the GPT-OSS reasoning parser, and
`enable_in_reasoning=true` (`readiness_vllm/server.log:12-16,42`). The adapter
also explicitly accepts up to 32 sequences
(`tt/generator_vllm.py:33-36,185-188,249-252`).

This matters because the integration README is stale: it still says the
server is capped at eight because B>8 decode was corrupt
(`doc/vllm_integration/README.md:33,172-173`) and records an older 72-pass
result (`README.md:18-20`). The stale statement is not proof that the present
B32 path is broken, but it makes B9/B10/B32 row-isolation a required gate.

## Failure 1: mixed parameters, seed 42, presence penalty

### Exact symptom

The only failing logical request is:

```text
prompt="List: ", max_tokens=10, temperature=0.5,
presence_penalty=2.0, seed=42
```

The recorded outputs are (`sampling_tests.log:111-118,175-186`):

```text
run 1: 1 2\n\nBut we need to be careful
run 2: 1 2\n\nBut we need to check if
```

A host-only Harmony tokenizer probe established the exact token boundary:

```text
1: "1"       6: " we"
2: " "       7: " need"
3: "2"       8: " to"
4: "\n\n"    9: " be"       vs " check"
5: "But"     10: " careful" vs " if"
```

Thus the saved failure is identical through output token 8 and first differs
at token 9. This is not merely a vague "late" divergence: the batch contains
two other requests with `max_tokens=8`, after shorter peers finish at tokens 3
and 5 (`sampling_tests.log:96-151`). Immediately before the failing request's
ninth sample, those last eight-token peers are removed and the persistent
layout is marked changed.

Focused live evidence supplied with this investigation is consistent with the
same boundary-sensitive defect: one isolated run passed, then a five-run
repeat failed three times, always on this presence-penalty request and only
after an identical prefix of at least five tokens. Stable lane slots do not
move during a request, but the shuffle assigns this logical request a different
physical row in the second batch.

### Current execution path

This request uses the product device-sampling path:

1. Structured requests alone are forced to host sampling; an ordinary seeded
   penalty request remains eligible for device sampling
   (`vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py:1851-1879`).
2. Lane mode pins each live request to one persistent physical row and leaves
   gaps instead of condensing (`input_batch.py:730-758,843-872`).
3. Finishing/removing any request sets `layout_changed`
   (`input_batch.py:905-966`), which becomes
   `_decode_layout_changed_since_last_decode=True` and drains pending async
   work (`model_runner.py:1482-1501`). Penalty requests already disable the
   steady overlapped decode fast path (`async_decode.py:208-246`).
4. On that next decode, the plugin rebuilds full prompt/output history tensors
   and sends `reset_batch=True` (`input_batch.py:1181-1236`). Output history is
   copied from the host token table using each row's prompt and current total
   lengths (`input_batch.py:644-678`).
5. The adapter forwards the reset, histories, positions, and stable slot map
   into the canonical generator (`tt/generator_vllm.py:621-647`). Explicit
   seeds and penalties make sampling-trace state non-reusable
   (`tt/generator_vllm.py:464-475`), and explicit request seeds run the device
   sampler eagerly rather than replaying a stale sampler trace
   (`models/common/sampling/generator.py:450-462`). The model decode trace may
   remain active; this distinction is important.
6. `apply_decode_state()` resets all penalty state from the host histories at
   a reset boundary (`models/common/sampling/generator.py:212-256`).
   `TTPenalties.reset_output_tokens()` zeros every row and rebuilds counts and
   masks (`models/common/sampling/tt_penalties.py:294-311`). The current token is
   then sampled from the penalized logits and counted exactly once on the
   device (`models/common/sampling/generator.py:318-337`).
7. The seed manager is explicitly configured with duplicate-seed salting off
   (`tt/generator_vllm.py:266-275`), and each explicit-seed counter is aligned
   to the absolute decode position before every new device seed
   (`models/tt_transformers/tt/generator.py:2194-2237` and
   `models/common/sampling/generator.py:983-1026,1104-1156`).

### Ranked hypotheses

#### H1 — penalty history/mask reconstruction is wrong for the continuing row at a removal reset (leading)

Prediction: the first mismatch will appear in the `List` row's reconstructed
`output_tokens`, presence mask, or post-penalty logits on the first decode after
a 3/5/8-token peer is removed. Before that boundary, the two runs' logical-row
state and sampled tokens will match.

Why it ranks first:

- the saved divergence is exactly token 9, immediately after both eight-token
  peers finish;
- repeat failures diverge only after a shared prefix and always involve the
  sole seeded presence-penalty request;
- uniform-lifetime B32 seed tests and uniform-lifetime B32 penalty tests pass;
- this exact test uniquely combines explicit seed, presence penalty,
  heterogeneous lifetimes, repeated layout resets, and a changed physical row;
- resets rebuild every row's counts wholesale even though stable slots make a
  slot-scoped rebuild possible.

Possible mechanisms within H1 are an off-by-one host history at the drained
reset, a wrong physical row when the full history tensor is copied, or a
reset/update sequence that loses or double-counts the boundary token. Presence
penalty only depends on a token's presence, so the first unique token omitted
or spuriously added is enough to change the candidate distribution.

#### H2 — B>8 or physical-row-dependent model/post-penalty logits

Prediction: with identical logical history, seed value, and presence mask, the
same request has different pre-penalty or post-penalty logits when placed in
different rows, especially rows at or above the old B8 boundary.

The physical row changes across the shuffled batches, and the stale integration
report records past B>8 corruption. A 36-layer standalone artifact is evidence
against a broad two-row model defect: duplicated prompts at B2 produced
bit-identical prefill and decode logits with maximum absolute difference 0.0
(`doc/full_model/artifacts/logit_reproducibility.json:7-18,19-33,74-96`). It
does not cover B10/B32, different physical rows, heterogeneous prompts, or the
penalty operator. This remains the principal alternative to H1.

#### H3 — device RNG seed/counter consumption is row-dependent at resets

Prediction: the logical request's derived 32-bit device seed or uniform draw
will first differ at a reset boundary even though its output position and
request seed match.

This is possible but lower-ranked. The current implementation disables
duplicate salting, aligns the counter from the logical absolute position, and
all dedicated cross-position/B32 seeded tests pass. If the derived seed is
identical but the draw differs, the fault is below `SeedManager` in the device
sampler's row mapping. If the derived seed differs, fix counter/reset plumbing.

#### H4 — stale async host state feeds the reset

Prediction: the host token table is exactly one sampled token behind at the
boundary, and disabling async scheduling makes the target deterministic.

This is lower-ranked because penalty requests are excluded from steady decode
overlap and layout changes explicitly drain pending steps. The current code was
written to address precisely this hazard (`async_decode.py:208-246,335-419`;
`model_runner.py:1434-1449,1492-1501`). It still warrants one control because a
missed drain/apply ordering would feed H1.

#### Refuted as the present root cause — duplicate-seed salting

The earlier reduced-model diagnosis was valid for the previous code, but the
fix is present now. The host isolate showed order-dependent device seeds with
salting on and identical seeds with salting off, then the exact reduced
mixed-params target passed
(`autofix/AUTOFIX_workspace_reduced_mixed_params.md:69-96`). Current full-36
outputs also share a long prefix; salting would change the first RNG draw. Do
not reapply or broaden that fix.

### Isolate experiments, in order

The first live experiment should instrument one logical request without
changing sampling behavior. At every token, record:

- request ID/prompt label, physical row, absolute position, `reset_batch`, and
  which peers were removed;
- host output-token history (or a collision-resistant hash plus the final 16
  token IDs), prompt history hash, and reconstructed presence-mask hash;
- request seed, salt, aligned counter, derived device seed, and if available
  the sampler uniform;
- top candidates from logits before penalties and after penalties, plus the
  sampled token.

Compare two runs by logical request, not physical row. The earliest unequal
field decides the next fix:

```text
history/mask differs -> H1 host/reset plumbing
history same, pre-penalty logits differ -> H2 model/KV/row path
pre logits same, post-penalty logits differ -> H1/H2 penalty operator row path
post logits same, derived seed differs -> H3 seed plumbing
post logits and seed same, sample differs -> H3 device sampler RNG mapping
```

Then run these narrow A/B controls; do not rerun the whole 22-minute profile
between each one:

1. Exact ten-request target with a fixed shuffle and forced `List` rows
   0, 7, 8, 15, and 31, repeated at least ten times.
2. Keep the same request and seed but set presence penalty to zero. A pass
   isolates the penalty path from the base model/RNG path.
3. Keep presence penalty but make all peers live for ten tokens. A pass
   isolates the removal/reset boundary.
4. Construct one peer group that finishes at exactly 3, 5, or 8 tokens, one
   boundary at a time. Correlate the first divergent token with the removal.
5. Run B8, B9, B10, and B32 with the target forced above and below row 8. This
   distinguishes a generic B>8 defect from reset history.
6. Disable async scheduling only as a diagnostic control. Penalty requests
   should already drain; a changed result proves the invariant is incomplete.
7. Explicit host-sampling compatibility mode for only this target. If the same
   TT model logits plus vLLM's host sampler reproduce across rows, localize the
   defect to TT device penalty/sampling. This is an experiment or opt-in
   compatibility containment, not the default production fix.

### Minimal fix, conditional on the isolate

- If H1 is confirmed, add a slot-scoped output-state reset analogous to the
  existing slot-scoped prompt merge (`tt_penalties.py:258-292`): after draining,
  clear/rebuild only removed/new/continuing rows whose authoritative history
  changed. Preserve the device-side counts of unaffected stable rows. Add an
  assertion/test that the boundary token is counted once. This work occurs at
  layout changes and need not slow steady traced decode.
- If H2 is confirmed, fix the first row-dependent model/KV/penalty operation.
  Do not restore the B8 cap unless a measured physical limit, rather than a
  software defect, is proved.
- If H3 is confirmed, fix the logical-request-to-row seed mapping/counter. Do
  not relax exact seeded equality.
- A default full-logits host fallback for all seeded/penalty requests is not an
  acceptable fix: it would hide the defect and weaken the advertised device
  path. An explicit compatibility option is acceptable as containment while a
  device fix is developed.

## Failure 2: structured choice returns `blue<|return|>`

### Exact symptom and path classification

The full-capacity test starts 32 mixed requests; every fourth request is one of
eight choice requests. Each choice uses chat completions, temperature zero,
eight completion tokens, and `structured_outputs.choice`; it expects exact
membership in `['red', 'green', 'blue', 'yellow']`
(`vllm-tt-plugin/tests/tt/test_structured_output_dp1.py:9,22-37,104-131`). The
observed content is `blue<|return|>` (`sampling_tests.log:239-259`).

This request cannot exercise TT on-device sampling. The plugin explicitly
returns false for device sampling whenever structured output is present
(`model_runner.py:1876-1879`), then applies the grammar bitmask to host logits
and calls vLLM's host sampler (`input_batch.py:1405-1419`). Therefore this is
not evidence against traced on-device performance sampling.

The server log supplies a one-to-one causal signature: exactly eight
`Harmony parser ended in a non-terminal state; returning the recovered raw
output` warnings appear together (`server.log:4398-4405`), followed by eight
choice chat responses (`server.log:4406-4413`). There are exactly eight choice
tasks in the 32-request mix.

### Why `enable_in_reasoning=true` causes it

Current vLLM structured output behavior is explicit:

- the configuration default is false (`vllm/config/structured_outputs.py:35-42`);
- when true, the structured grammar bitmask is filled unconditionally, before
  the GPT-OSS reasoner has detected the final channel
  (`vllm/v1/structured_output/__init__.py:362-380`);
- GPT-OSS's reasoner expects `<|channel|>final ... <|message|>` and only detects
  the end of reasoning; Harmony itself parses the message
  (`vllm/reasoning/gptoss_reasoning_parser.py:64-83`);
- on malformed/non-terminal Harmony, `flush()` intentionally decodes the raw
  pending token IDs and returns them as final content
  (`vllm/parser/harmony.py:113-154`), which chat serving exposes
  (`vllm/entrypoints/openai/chat_completion/serving.py:892-907`).

With the choice grammar active at token one, the model is constrained to a
bare choice rather than allowed to emit the required Harmony assistant/channel
header. A host-only tokenizer/parser probe produced:

```text
"blue"             -> [18789]          (ordinary token)
"<|return|>"       -> [200002]         (special token)
"blue<|return|>"   -> [18789, 200002]
bare parser EOS     -> HarmonyError: Unexpected EOS while waiting for message header
```

This precisely explains both the suffix and the non-terminal warnings. The
upstream unit test also treats raw recovery as intentional: it feeds malformed
Harmony ending in `<|return|>` and asserts that the recovered string, including
that token, is returned verbatim (`vllm/tests/parser/test_harmony.py:53-59,
167-185`). A focused pytest invocation ran the assertion body successfully and
emitted the expected warning, but the command ended in a teardown error because
vLLM's global cleanup called `torch.accelerator.empty_cache()` on this
device-free host; it is not reported as a passing command.

Historical workspace evidence provides a useful, though version-confounded,
control: a B32 suite had the structured test pass
(`readiness_vllm/sampling_tests_failed_seed_only_max32.log:47-82`) while its
server used `enable_in_reasoning=false`
(`readiness_vllm/server_failed_seed_only_max32.log:16`). That server used an
older vLLM development revision, so repeat the A/B on current vLLM before
calling it final.

### Ranked hypotheses

#### S1 — incorrect launch configuration (`enable_in_reasoning=true`) (leading)

Prediction: on current vLLM, removing the explicit option or setting it false
allows GPT-OSS to emit a valid Harmony header, applies the choice grammar only
to final content, removes all eight non-terminal warnings, and returns an exact
choice. Historical false/default evidence and the current source semantics
both support this.

#### S2 — insufficient eight-token budget after configuration is corrected

Prediction: false/default removes the bare-choice behavior but eight tokens can
end before GPT-OSS reaches a final Harmony message. If so, raise only this
test's completion budget to the minimum empirically sufficient value. Do not
use a larger budget as a substitute for correcting `enable_in_reasoning`.

#### S3 — shared-test normalization is required for an intentionally raw mode

Only if the product intentionally keeps grammar-in-reasoning true, the shared
test's plain-string expectation is incompatible with that malformed raw-output
mode. Tokenizer `skip_special_tokens` can decode `[blue, return]` to `blue`, but
global string stripping in the TT plugin/API is unsafe: upstream deliberately
preserves malformed output so callers can see parser failure, and stripping
could conceal malformed tool calls or Harmony envelopes.

### Isolate experiments and minimal fix

1. On current vLLM, issue one choice request with current true, then false/default,
   collecting response content, output token IDs, finish reason, and Harmony
   warnings. Repeat with completion budgets 8 and 32.
2. Confirm that false/default yields a completed Harmony final message and that
   the grammar begins advancing only after the final marker.
3. Rerun the 32-request mixed structured target three times. Require zero
   non-terminal warnings and exact choice strings, in addition to regex/JSON
   validity.

The minimal product fix is to remove the explicit
`"enable_in_reasoning": true` from the GPT-OSS server command or set it false,
which is also vLLM's default. Update the stale launch documentation at
`doc/vllm_integration/README.md:59-74`. If current-vLLM evidence proves an
eight-token length issue after that change, increase only the choice test's
budget.

Do not add a TT sampler workaround and do not globally call
`.removesuffix("<|return|>")`. If product requirements explicitly demand
grammar-in-reasoning mode, a narrowly model-aware test normalization may remove
exactly one trailing special return token at the token-ID level while rejecting
all other extra content, but that is the fallback, not the first fix.

## Verification gates after fixes

1. Mixed-params fixed-shuffle row matrix and boundary-isolation tests pass
   repeatedly (at least 10/10 each).
2. Existing B32 seed, penalty, logprob, and structured tests remain green.
3. Current-vLLM structured false/default A/B has no Harmony recovery warnings.
4. Complete full-36 sampling profile returns zero failures.
5. Production on-device trace counters/performance are remeasured separately.
   No performance claim is made by this report, and the structured path is host
   sampled by design.
