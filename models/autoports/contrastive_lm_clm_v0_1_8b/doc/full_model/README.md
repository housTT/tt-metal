# Stage 6: full model (encoder) on p150

Mapping from the plugin's decoder stage to this encoder-only model is recorded in
`/home/hous/dev/clm-v0.1-8B/PLAN.md` section 4. There is no LM head, no sampling, no autoregressive loop,
so the plugin's AIME24 teacher-forcing and `check_degenerate_output.py --scope autoregressive` gates are not
applicable. Their substitute is an embedding fidelity gate against a CPU fp32 reference, below.

## Implementation

`tt/encoder.py`, class `TtQwen3Encoder`:

- builds Qwen3-8B with `models.tt_transformers` (`create_tt_model`, paged KV cache, 1x1 mesh, bfp8 weights);
- `embed_ids` groups inputs by padded prefill bucket (the smallest of 128, 256, 512, 1024, 2048 that fits the
  longest text in the group; `CLM_TRACE_LENS` overrides the list), right-pads to the bucket and to a batch of 1, 4 or
  8, replays the matching prefill trace and reads the pre-norm residual back; the last real token per sequence is
  picked on the host, the final RMSNorm is applied in fp32 on the host, and the result is L2-normalized;
- `embed(texts)` tokenizes with `add_special_tokens=False`, keeps the last `max_tokens` tokens (vLLM
  `truncate_prompt_tokens` semantics used by upstream `clm-serve`), and returns the encoder tokens spent;
- the Qwen3-8B entries added to `models/tt_transformers/tt/model_config.py` enable traced prefill on P150, P300 and
  P150x4; the encoder then replaces the instance's `trace_prefill_supported_seq_lens` with its own bucket list;
- `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0` is set when the mesh is opened (see `doc/probe/README.md`).

## Fidelity gate and result (accuracy policy)

Reference: `/home/hous/dev/clm-v0.1-8B/reference/hf_embeddings.npy`, HF `AutoModel` fp32 on CPU, same
tokenization and pooling, 308 texts (221 states, 87 candidates; README examples, Typed Decisions states and
options, T-Rex states, tale-of-two-cities excerpts at 32 to 2048 tokens). Script: `tests/run_fidelity.py`.
Result files: `fidelity_accuracy.json` (final encoder code, 2026 Oct 1 23:10 UTC, nine trace variants; byte-identical
to `../datatype_sweep/fidelity_accuracy.json`) and `../optimized_full_model/fidelity_accuracy_buckets5.json` (the
shipped fifteen-variant encoder; single-text vectors bit-identical to the nine-variant run). Raw vectors
`fidelity_accuracy_tt_single.npy`, `fidelity_accuracy_tt_batched.npy` (not committed).

| metric | gate | measured |
|---|---|---|
| cosine TT vs HF fp32, mean | >= 0.99 | 0.99910 |
| cosine TT vs HF fp32, min / p05 | min >= 0.97 | 0.99596 / 0.99756 |
| cosine by length bucket (<=128 / <=1024 / >1024), mean (min) | | 0.99910 (0.99600) / 0.99910 (0.99596) / 0.99965 (0.99961) |
| PCC TT vs HF fp32, mean / min | report | 0.99914 / 0.99604 |
| centered cosine (corpus mean removed), mean / min | report | 0.99549 / 0.98845 |
| head projection cosine, state head, mean / min (221) | min >= 0.95 | 0.99746 / 0.99434 |
| head projection cosine, action head, mean / min (87) | min >= 0.95 | 0.99871 / 0.99571 |
| same text alone vs inside a mixed batch, mean / min | mean >= 0.999 (plan amendment 2026 Oct 2) | 0.99934 / 0.99644 |
| run-to-run determinism (32 texts), mean / min | 1.0 - 1e-4 | 1.0000 / 0.9999998 |
| Typed Decisions argmax agreement with the fp32 reference, 200 subset decisions | report | 95.5 percent (191 of 200) |
| same, over the 188 decisions whose reference top-2 margin is >= 0.10 | >= 98 percent (plan row 6, margin-aware form) | 98.9 percent (186 of 188) |
| NaN count | 0 | 0 |

Why the agreement gate is margin-aware: 12 of the 200 reference decisions are ties within 0.10 (8 within 0.05), and a
verifier running in bf16 and bfp8 arithmetic cannot be expected to reproduce the argmax of a tie; the gate therefore
counts the decisions the reference itself is confident about and reports the plain number next to it. The two
confident disagreements (`invoice_processing_000063` and `_000096`, question `discrepancy_severity`) have reference
margins 0.13 and 0.32 and total-variation distances 0.24 and 0.28. The gate is evaluated on the single-text vectors;
on batched vectors the shipped policy scores 185 to 186 of 188 (`../datatype_sweep/README.md`).

Why these numbers and not a token-accuracy gate: the model's output is the pooled vector itself. The raw Qwen3
embedding space is anisotropic (mean pairwise cosine between unrelated corpus texts is 0.846), so the centered
cosine row is included to show the agreement is not an artifact of the shared component.

The batched-vs-single minimum (0.9964) is below the single-vs-HF minimum because tt_transformers' batched prefill
on Blackhole is batch-variant in its float reduction order (noted in `model_config.py` next to
`disable_batched_prefill`, tt-metal issue 47238). It is numerical noise, not cross-request leakage: the
run-to-run cosine of the same input is 1.0, and different inputs in the trace replay check give cosine far below 1.

## Serving nondeterminism to know about

The encoder output for a text depends slightly on which other texts share its prefill batch (cosine 0.9993 mean,
0.9964 min between alone and in a batch; 60 of 308 texts below 0.999). On the Typed Decisions subset this flips the
argmax of 4 to 5 of 200 decisions between the two modes, all near-ties. Cause: batch-variant float reduction order
in the prefill kernels (tt-metal 47238). The server groups a request's texts into batches, so repeated identical
requests are deterministic (replay check), but the same text in a different request mix can differ at this level.
Recorded in the card's limitations.

## Qualitative check substitute

`$qualitative-check` is for generated text. The substitute: the README's worked examples through the encoder and the
CLM heads, computed from the final-path single-text vectors (`../optimized_full_model/fidelity_accuracy_buckets5_tt_single.npy`)
and from the fp32 reference vectors with the same head code (`tests/decision_agreement.py` method):

| question | TT, single-text path | fp32 reference | served package (request batched) |
|---|---|---|---|
| tides ranking, P(Moon) | 0.9922 | 0.9935 | 0.9948 (`/v1/rank`, clean pull check) |
| urgency, noul P(true) | 0.852 | 0.842 | 0.816 |
| department, P(billing) | 0.991 | 0.988 | 0.993 |
| frustration, score on the 0 to 2 scale | 2.000 | 2.000 | 2.000 |

Argmax decisions are identical in all three columns. The served column differs from the single-text column because
the server embeds a request's state and option texts in one batch (batch-variant reduction order, next section).
The README's printed values for the same example (0.410 / 0.939 / 1.984) are not reproduced by the published head and
code on any hardware (`/home/hous/dev/clm-v0.1-8B/reference/README.md`). Decision-level agreement on the Typed
Decisions benchmark is measured against the served package in `../release/RUN_NOTES.md`.

## Context contract

`doc/context_contract.json`: Qwen3-8B advertises 40960 positions; upstream `clm-serve` serves 2048 by default;
this port serves 2048 (traced buckets 128 / 256 / 512 / 1024 / 2048, 2047 and 2048-token inputs verified in the
corpus). The 256 and 512 buckets were added after the first served evaluation showed that five 130 to 300 token
texts per Typed Decisions case cost one batch-8 1024-token pass; `../optimized_full_model/README.md` has the evidence.
Longer inputs are truncated to their last 2048 tokens, matching upstream.
