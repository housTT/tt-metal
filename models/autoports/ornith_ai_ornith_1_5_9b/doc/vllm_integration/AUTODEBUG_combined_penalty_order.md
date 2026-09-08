# AutoDebug: combined presence/frequency and repetition order

2026-09-08, source-only finding written before implementation changes.

The final decode audit found a possible shared sampler contract mismatch.
`models/common/sampling/tt_penalties.py::apply_penalties` currently subtracts
presence and frequency before applying repetition scaling. The pinned vLLM
`vllm/model_executor/layers/utils.py::apply_penalties` first calls
`apply_repetition_penalties`, then subtracts frequency and presence. Its
actual CPU branch is `vllm/_custom_ops.py::apply_repetition_penalties_torch`:
scale positive logits by reciprocal repetition, nonpositive logits by
repetition, only for tokens in the prompt or generated history.

Hypothesis: the ordering changes combined-penalty scores and can change the
sampled token even when each individual penalty is correct. For generated
token A with raw logit 4, presence 2, repetition 2, and unseen token B at
0.5, the shared device formula yields A=1, while the host formula yields
A=0. The two paths therefore choose different tokens. Negative logits and
subtractions crossing zero can also select the wrong repetition branch.

The planned focused proof executes the actual common function against CPU
TTNN-operation boundaries, and executes the pinned host helper plus its real
Torch repetition implementation without importing TTNN or initializing vLLM.
Cases cover neutral and individual penalties, combined positive/negative
coefficients, positive/negative logits, zero crossings, repeated generated
counts, and prompt-only versus generated history. Only after this proves the
hypothesis may the canonical shared function be reordered minimally.

The existing all-layer serving process must finish with its already imported
source. The supervising lane alone will run an exact TP4 canonical sampler
probe after shutdown, then rerun serving coverage. No dtype, sampler
replacement, or model change is proposed.

## Focused CPU result before the fix

`test_sampling_penalty_order.py` source-executed the unmodified shared
function: **6 failed, 7 passed** (`combined_penalty_order_before.log`). All
neutral/individual controls passed; five combined score cases and the forced
top-1 crossing failed. Histories distinguish generated once/three times,
prompt only, both histories, and unseen tokens; logits include both signs
and additive zero crossings.

`combined_penalty_host_control.py --vllm-root ../vllm --expect-difference`
then executed the pinned host bin-count helper, penalty wrapper, repetition
dispatcher, and its actual Torch implementation. **All 12 host cases matched
the independent reference exactly; all five combined configurations
disagreed with the unmodified shared function.** See
[`combined_penalty_host_before.json`](combined_penalty_host_before.json),
including host source hashes and exact scores. H1 is verified before any
implementation change.

After the minimal order correction, all 13 CPU regressions and all 12 actual
host controls pass. The exact supervising TP4 sampler execution also passes:
[`combined_penalty_sampler_exact.json`](combined_penalty_sampler_exact.json).
It confirms exact candidate scores and sampled tokens for 32 mixed lanes,
four shards, positive/negative/zero-crossing logits, prompt/generated history,
and count 1 versus 7 across two replays. See the corresponding
[`AUTOFIX_combined_penalty_order.md`](AUTOFIX_combined_penalty_order.md)
for final serving regression status. The restarted all-layer B32 live
neutral/combined control also passed exact host/device token-ID and effect
parity for four 20-token requests:
[`combined_penalty_live.json`](combined_penalty_live.json).
