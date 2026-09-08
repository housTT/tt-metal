# AutoFix: combined penalty order

2026-09-08. Status: shared fix verified by CPU, exact hardware, and the live
combined-penalty serving control; final full-suite rerun pending.

The final serving-path source audit found that the common TT sampler applied
presence/frequency before repetition, unlike pinned vLLM. The focused test
source-executed the real common function: six combined checks failed while
seven neutral/individual controls passed. The actual pinned host bin-count,
penalty wrapper, CPU repetition dispatcher and Torch helper matched the
independent oracle in all 12 cases; five combined configurations disagreed
with the common sampler. See
[`AUTODEBUG_combined_penalty_order.md`](AUTODEBUG_combined_penalty_order.md).

The only runtime change reorders the existing operation blocks in
`models/common/sampling/tt_penalties.py` to repetition, frequency, presence.
This preserves original-logit sign selection for repetition and matches the
host additive order. Dtypes, parameters, persistent output addresses,
sampler implementation, trace API, and model precision are unchanged.

After the fix, `tests/test_sampling_penalty_order.py` passes all **13** CPU
checks and the actual pinned host control matches all **12** cases exactly:
[`combined_penalty_order_after.log`](combined_penalty_order_after.log) and
[`combined_penalty_host_after.json`](combined_penalty_host_after.json).
Cases cover positive/negative logits, zero crossings, positive/negative
additive penalties, repetition above/below one, repeated output counts,
prompt-only/generated/both/unseen history, and an argmax crossing. The tests
also require the shared function to return its original logits buffer.

The existing exact TP4 probe now accepts `--case combined`, with 32 mixed
lanes, candidates on every vocabulary shard, four score patterns, prompt-only
history, generated counts 1 and 7, and two captured steps. It asserts exact
candidate scores and tokens, stable trace addresses, and no new reset
programs after capture. Default `--case presence` keeps the previous isolated
presence proof. Device execution remains exclusively in the supervising lane.

The supervising hardware execution passed (exit 0):
[`combined_penalty_sampler_exact.json`](combined_penalty_sampler_exact.json)
and [`combined_penalty_sampler_exact.log`](combined_penalty_sampler_exact.log).
All 32 mixed lanes, four vocabulary shards, both initial history counts and
both replays produced exactly the expected candidate scores and tokens.
Stable buffer addresses and unchanged reset program-cache counts also passed.

The bounded live control uses `presence_distribution_probe.py --combined
--prompt 'Once upon a time' --max-tokens 20` to compare neutral and
presence-2/frequency-0.5/repetition-2 device outputs against explicit host
sampling. It requires exact token IDs and an observable effect, and records
actual prompt IDs and generated counts. It does not infer repetition math
from normalized logprobs: without the omitted logsumexp, absolute logit signs
and values cannot be recovered. The separate CPU and exact hardware scores
provide the arithmetic proof.

This live control passed (exit 0) on the restarted all-layer B32 server:
[`combined_penalty_live.json`](combined_penalty_live.json) and
[`combined_penalty_live.log`](combined_penalty_live.log). All four requests
returned 20 completion tokens. Device and host token IDs matched exactly for
both neutral and combined settings, and both paths showed an observable
penalty effect. The report explicitly marks combined absolute-logit formula
certification unavailable from raw logprobs; it is not claimed as a second
arithmetic proof.

Commands (repository root):

```bash
USER=hous ../state/serving-env/bin/python -m pytest -q --noconftest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_sampling_penalty_order.py
USER=hous ../state/serving-env/bin/python models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/combined_penalty_host_control.py --vllm-root ../vllm --output models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/combined_penalty_host_after.json
```

The complete canonical suite passed 72 tests with one skip immediately
**before** this order correction; that run cannot validate the new runtime
ordering. The post-fix full-suite rerun remains required. All five Python
files touched in this repair passed Black and compilation. No C++
build is needed for this Python-only change.
