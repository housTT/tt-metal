# Stage 6: full model (encoder) on p150

Mapping from the plugin's decoder stage to this encoder-only model is recorded in
`/home/hous/dev/clm-v0.1-8B/PLAN.md` section 4. There is no LM head, no sampling, no autoregressive loop,
so the plugin's AIME24 teacher-forcing and `check_degenerate_output.py --scope autoregressive` gates are not
applicable. Their substitute is an embedding fidelity gate against a CPU fp32 reference, below.

## Implementation

`tt/encoder.py`, class `TtQwen3Encoder`:

- builds Qwen3-8B with `models.tt_transformers` (`create_tt_model`, paged KV cache, 1x1 mesh, bfp8 weights);
- `embed_ids` groups inputs by padded prefill bucket (128, 1024, 2048), right-pads, and calls
  `Generator.prefill_forward_text(..., return_hidden_states=True)`, which slices the last real token per user,
  applies the final RMSNorm on device and returns `[batch, 4096]`; the encoder L2-normalizes;
- `embed(texts)` tokenizes with `add_special_tokens=False`, keeps the last `max_tokens` tokens (vLLM
  `truncate_prompt_tokens` semantics used by upstream `clm-serve`), and returns the encoder tokens spent;
- the Qwen3-8B entries added to `models/tt_transformers/tt/model_config.py` enable traced prefill at 128, 1024
  and 2048 tokens on P150, P300 and P150x4;
- `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0` is set when the mesh is opened (see `doc/probe/README.md`).

## Fidelity gate and result (accuracy policy)

Reference: `/home/hous/dev/clm-v0.1-8B/reference/hf_embeddings.npy`, HF `AutoModel` fp32 on CPU, same
tokenization and pooling, 308 texts (221 states, 87 candidates; README examples, Typed Decisions states and
options, T-Rex states, tale-of-two-cities excerpts at 32 to 2048 tokens). Script: `tests/run_fidelity.py`.
Result file: `fidelity_accuracy.json`; raw vectors `fidelity_accuracy_tt_single.npy`, `fidelity_accuracy_tt_batched.npy`.

| metric | gate | measured |
|---|---|---|
| cosine TT vs HF fp32, mean | >= 0.99 | 0.99909 |
| cosine TT vs HF fp32, min / p05 | min >= 0.97 | 0.99588 / 0.99753 |
| cosine by length bucket (<=128 / <=1024 / >1024), mean | | 0.99910 / 0.99908 / 0.99965 |
| centered cosine (corpus mean removed), mean / min | report | 0.99434 / 0.98588 |
| head projection cosine, state head, mean / min (221) | min >= 0.95 | 0.99744 / 0.99429 |
| head projection cosine, action head, mean / min (87) | min >= 0.95 | 0.99870 / 0.99573 |
| same text alone vs inside a mixed batch, mean / min | mean >= 0.999 | 0.99934 / 0.99643 |
| run-to-run determinism (32 texts), min | 1.0 - 1e-4 | 1.0000 |
| NaN count | 0 | 0 |

Why these numbers and not a token-accuracy gate: the model's output is the pooled vector itself. The raw Qwen3
embedding space is anisotropic (mean pairwise cosine between unrelated corpus texts is 0.846), so the centered
cosine row is included to show the agreement is not an artifact of the shared component.

The batched-vs-single minimum (0.9964) is below the single-vs-HF minimum because tt_transformers' batched prefill
on Blackhole is batch-variant in its float reduction order (noted in `model_config.py` next to
`disable_batched_prefill`, tt-metal issue 47238). It is numerical noise, not cross-request leakage: the
run-to-run cosine of the same input is 1.0, and different inputs in the trace replay check give cosine far below 1.

## Qualitative check substitute

`$qualitative-check` is for generated text. The substitute recorded here: the README's two worked examples
through both encoders and the CLM heads (`doc/probe/probe_full_encoder.json`): tides ranking Moon 0.992 (HF 0.993),
customer routing billing 0.990 (HF 0.990); argmax decisions identical. Decision-level agreement on the Typed
Decisions benchmark is measured against the served package in `doc/release/`.

## Context contract

`doc/context_contract.json`: Qwen3-8B advertises 40960 positions; upstream `clm-serve` serves 2048 by default;
this port serves 2048 (traced buckets 128 / 1024 / 2048, 2047 and 2048-token inputs verified in the corpus).
Longer inputs are truncated to their last 2048 tokens, matching upstream.
