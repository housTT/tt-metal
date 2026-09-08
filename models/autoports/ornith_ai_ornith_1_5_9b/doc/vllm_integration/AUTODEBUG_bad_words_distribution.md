# AutoDebug: constrained greeting quality and distribution control

Date: 2026-09-08. The investigator prepares source/CPU-only diagnostics; the
parent serializes all live requests.

## Observed quality problems

Inspection of `bad_words_original_token_ids.json` confirms that seed2 answers
the English greeting request in Chinese, and seed3 incorrectly describes the
request as asking about netcat and continues on that topic. These outputs are
wrong-language and off-topic respectively. They are not marked coherent
greeting passes merely because their forbidden token IDs are absent.

That test bans all six `hello`/`Hello`/`hi`/`Hi`/`hey`/`Hey` variants while
prompting `Say hello to me`, temperature1/top-p1/max100. The exact token mask
can remove high-probability continuations. Whether that explains these poor
outputs, or whether an additional model/serving defect exists, needs evidence.
Absence of forbidden IDs alone does not answer the quality question. Wrong
topic alone also does not establish cross-request contamination.

## Focused control

`bad_words_distribution_control.py` performs six sequential requests when both
seeds diverge. It requests the same seeds2/3, prompt, and generation parameters
with and without the six bad words. Both controls request actual token IDs and
raw logprobs, forcing the same explicit optional host sampler on both paths.
This avoids comparing different host/device random-number algorithms.

For each seed, it finds the first differing **actual generated token ID**,
records both full outputs, and constructs the exact neutral prefix from the
server-returned chat `prompt_token_ids` plus common generated IDs. A one-token
neutral legacy completion at that exact prefix requests the top20 raw
logprobs. Source inspection confirms `TTModelRunner` constructs `Sampler()`;
its default `raw_logprobs` are computed before bad-word masking. The script
records source hashes and explicit client/server request IDs.

The script executes the actual pinned `SamplingParams.update_from_tokenizer`
method and actual upstream bad-word masking function on CPU. It shows which
observed high-probability greeting IDs are masked, confirms observed allowed
logits are unchanged, and reports observed removed probability mass as a
**lower bound** because only top20 logprobs are collected.

Exact parent-owned command from the tt-metal root:

```sh
USER=hous ../state/serving-env/bin/python models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/bad_words_distribution_control.py --label bad-words-quality-control --output models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/bad_words_distribution_control.json
```

The original bad-word output can differ from these diagnostic calls because
they explicitly expose logprobs and run sequentially on the current final
server. The report must describe what actually reproduces. This script has no
retry loop and does not claim an automatic quality pass.

## Verification and limitations

A CPU dry run substituted only the network client and exercised all six
request/response branches, exact-prefix construction, actual tokenizer
variants, first-divergence indexing, and actual CPU masking. It passed with
zero network requests or TTNN imports:
`bad_words_distribution_control_cpu_dryrun.log`. Synthetic responses were not
saved as serving evidence. Formatting and byte-compilation pass.

Raw-distribution agreement at the first divergent prefix can localize the
effect of the mask. It does not numerically validate every later token in the
autoregressive path. Fresh exact-prefix prefill can also differ numerically
from the original decode sequence. The parent is coordinating an independent
standalone numerical control separately. Read and classify the paired full
outputs and inspect both controls before deciding whether the observed poor
quality is attributable to constrained sampling or remains an integration
defect.

## Live control and complete prefix audit

The parent ran the six-request command once; it completed with exit0.
`bad_words_distribution_control.json` and `.log` contain every request and raw
response. Request IDs use `bad-words-quality-control-seed{2,3}-` followed by
`unbanned`, `banned`, or `neutral-prefix`; the summary records the corresponding
server response IDs. The paired request parameters are exactly identical
except for bad words and request ID. Both use explicit host logprobs, the same
seed, temperature1/top-p1, and maximum100 output tokens.

The banned responses reproduce **all original generated IDs and text exactly**:
98 IDs for seed2 and 100 IDs for seed3. The prompt IDs also match the original
requests exactly. Their unbanned counterparts produce English greetings
(85 and79 IDs respectively).

The investigator independently compared every raw top20 map at output indices
**0 through7 inclusive** for each seed. All16 paired comparisons match exactly,
with zero logprob difference after disregarding list order. The same maps also
match across seeds, so all four host controls have identical observed raw
distributions throughout their shared prefix and at the first divergent step.
No generated IDs are inferred by re-tokenizing text.

Both controls first diverge at output index7: the unbanned request samples
23066 (` hello`), while the banned request samples328 (` "`). The former is a
forbidden singleton; the latter is allowed. The exact rendered prefix is the
original chat prompt followed by `The user is asking me to say`. The raw model
distribution at this point is unchanged between banned and unbanned controls;
the requested token mask changes the available next-token distribution.

In the fresh neutral-prefix prefill control, the observed forbidden greeting
IDs carry **at least97.39995128%** of the raw probability mass. The dominant
` hello` token alone has probability0.97266542; the allowed quotation token
has probability0.02592065. The actual pinned CPU mask sets all12 forbidden
IDs to negative infinity while leaving observed allowed logits unchanged.
The investigator reran that exact source-mask check and rechecked all source
hashes. This is a mask-write control, not a full-distribution sampling oracle.

The original decode distribution has a removed-mass lower bound of
97.40094292%. Fresh prefill and original decode contain the same top20 IDs but
are **not numerically identical**: the largest observed difference is
0.25006390 nats on low-probability ` a` and ` the` entries. The dominant
` hello` logprob differs by approximately0.00006331 nats. These cross-mode
differences are preserved in the summary and are not conflated with the exact
banned/unbanned comparisons within the original decode path.

## Anomaly ledger and control verdict

The investigator read all four paired full outputs.

| Seed / condition | Observed output and qualitative disposition |
| --- | --- |
| 2, unbanned | Coherent English greeting on topic, without a pathological repetition loop or unreadable token output. It adds the unsolicited self-identification `I'm Qwen`, which is recorded rather than silently omitted. |
| 2, bad words | Exactly reproduces the original Chinese final greeting after invented `helo`/`helto` typo reasoning. Greeting topic broadly remains, but the English prompt receives a Chinese answer and user wording is misrepresented. Wrong-language quality anomaly retained. |
| 3, unbanned | Coherent English greeting on topic; no language drift, pathological loop, or unreadable token output observed. |
| 3, bad words | Exactly reproduces the original netcat/411 digression. It fabricates a different user request, gives a semantically confused explanation, and reaches the100-token budget before a normal greeting answer. Off-topic quality anomaly retained. |

The two bad-word outputs are **not qualitative passes**. Their exact
reproduction under the ban, English unbanned controls, equal prompt IDs, and
identical observed raw prefix distributions localize the first branch change
to the requested token exclusion. There is no evidence of request mixing in
the inspected prompt IDs or prefix distributions. This does not claim a
numerical validation of every later token or categorically rule out every
possible later-path defect. The independent standalone numerical baseline has
now passed on the different, meaningful prose workloads described below.

The inspected [standalone report](logit_determinism_standalone.json) and
[log](logit_determinism_standalone.log) cover a131-token community-library
narrative and65-token exercise paragraph, batch capacity32, all32 layers,
selected precision, and device rows0/1/31. All9 standalone full-vocabulary
raw-logit comparisons are exact across four steps and248320 vocabulary
entries. All11 comparisons to the [vLLM API control](logit_determinism_vllm.json)
match generated IDs and sampled/top20 raw-logprob signatures exactly. The
standalone report records `cleanup_completed=true`; the log confirms device
close completion. These API comparisons concern token/logprob signatures,
not a full-vocabulary API readback.

This completed baseline supports repeatability, request-slot isolation, and
standalone/vLLM agreement on those prose workloads. It is **not** a numerical
comparison of the same greeting prefixes or all98/100 tokens of the constrained
greeting outputs. The bad-word first-divergence evidence and this independent
numerical control retain their distinct scopes.

Control verdict: **the mask-induced first branch change is verified, and the
poor constrained outputs remain explicitly recorded quality anomalies**.
No runtime implementation, sampling policy, canonical threshold, or benchmark
path was changed to obtain this result.

Machine-readable audit: `bad_words_distribution_summary.json`, with input and
source hashes, every prefix-map comparison, original-output identity checks,
mask-input proof, numerical limits, and the full per-output anomaly ledger.
Its SHA256 is recorded in `bad_words_distribution_summary.sha256`:

```text
58e99e6309fdb1163b37f0b3b543620a078f80342f6499da6f5987f7b08716ce
```
