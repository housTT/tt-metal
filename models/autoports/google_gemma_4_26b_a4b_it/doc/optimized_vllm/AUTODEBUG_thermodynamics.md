# AutoDebug: sampled thermodynamics claim

Scope: source/log inspection only. Hardware and adapter-swap execution belong to the coordinating agent. No implementation change is justified by the source evidence alone.

## Finding and control gap

Entry 3 of [the retained P150x4 sampled output](../../readiness_vllm/P150x4/optimized_vllm/after_warmed/vllm_qualitative_outputs.json) says all energy transfers are never 100% efficient because some energy is always wasted as heat. This is a factual overgeneralization. The request used the model's rendered chat template, temperature 0.7, top-p 0.9, top-k 32, and a 256-token limit, without an explicit seed. The previous sampled answer does not reach the same sentence. Other prompts' logit equality, the matching greedy answer, and a 64-token HF control do not control this late sampled claim.

## Source evidence

- The diff from `6eb0427423392d7c6a7f87a511be892b8bf677ae` changes [the adapter](../../tt/generator_vllm.py): greedy-temperature normalization, selective updates into stable page-table buffers, and decode trace retention/invalidation. Positive-temperature top-k/top-p/temperature values and the seed formula are unchanged. `git diff 6eb042 -- models/autoports/google_gemma_4_26b_a4b_it/tt/generator.py models/common/modules/sampling/sampling_1d.py` is empty.
- `Gemma4ForCausalLM._sampling_values` computes an unseeded row's device seed as `(prefill_epoch * 104729 + row) % 2147483647`. The epoch starts at zero and advances on each `prefill_forward`, so unrelated previous requests and prefill chunk histories can change a sample even with identical public sampling parameters.
- [The generator](../../tt/generator.py) resets seeded slots on prefill, includes sampled seeds in `SamplingSpec.key`, seeds newly active slots, and restores request seeds after capture. [Sampling1D](../../../../common/modules/sampling/sampling_1d.py) calls `ttnn.manual_seed` before sampling. This describes intended state handling; changed trace lifetime still warrants a prompt-specific A/B check.
- An explicit API seed changes the tested path: the adapter advertises `supports_device_seeded_sampling=False`, and the sibling plugin's `model_runner.py:2648` routes such requests to host sampling. Requested top-logprobs also fall back for this model (`model_runner.py:2672`). Keep the device-path comparison unseeded; a seeded host control is supplementary evidence only.
- Since each unseeded positive-temperature request gets a new semantic seed key, repeated requests exercise warmed programs/cache history, not reuse of one stochastic trace across request boundaries.

## Focused experiment and predictions

The coordinating agent's [control script](run_thermodynamics_control.py) compares the original adapter with the selected adapter using otherwise identical fresh TP4 servers and three serial, unseeded requests each. It retains every response. Its prompt is the exact retained rendered chat prompt followed by the sampled answer prefix ending `processes. `, immediately before the disputed claim. It uses `/v1/completions` to continue that already-rendered prefix with the same 0.7/0.9/32 sampling settings and 256-token limit. The [manifest](thermodynamics_control/manifest.json) retains checkpoint revision, adapter hashes, launch arguments, request bodies, and response hashes.

Identical serial admission and prefill histories align the internal seed calculation without forcing host sampling. Compare each original/current continuation directly. If the original also produces the disputed substantive claim, that claim is already reproducible in the baseline TT serving implementation under this prefix. If paired continuations match, this additionally refutes a behavior change for these particular controlled requests. If they diverge, localize the first divergence with identical prefix and actual thermodynamics logits before changing math, cache, or RNG code; preserve device sampling in the initial A/B.

This experiment conditions on a sampled prefix. It cannot establish equivalence of the original full 256-token stochastic trajectory, the frequency of this error across seeds, or whether HF produces the same error. A broader attribution would require the original prompt and matched full request history/internal seed, or a matched HF continuation. No such broader claim is required to classify an error demonstrably reproduced by the old TT adapter as an existing TT-serving limitation.

## Status

All three original-adapter continuations reproduce the substantive claim that conversion always wastes energy as heat. Each matched optimized continuation has identical completion text and usage. Both runners exited zero and the selected adapter was restored. See [the AutoFix result](AUTOFIX_thermodynamics.md) for evidence and interpretation. Keep the factual limitation visible; no runtime fix is supported by this conditional control.
