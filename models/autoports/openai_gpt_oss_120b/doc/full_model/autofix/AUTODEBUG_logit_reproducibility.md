# AutoDebug: batch-position decode-logit reproducibility

## Scope and starting evidence

This is a fresh, source-only AutoDebug pass. No TT hardware command was run,
and no implementation or test file was edited. The failed command was the
P150x4 reduced real-weight gate
`test_real_weight_two_layer_batch2_logit_reproducibility`. Its delayed-assert
diagnostic payload is
`artifacts/logit_reproducibility_probe.json`.

The artifact localizes the observed failure more precisely than the original
first assertion:

| Boundary | Across identical batch rows | Across reset/reuse runs | Other evidence |
| --- | --- | --- | --- |
| Prefill full logits | bitwise equal | bitwise equal | finite; max diff 0; identical raw and top-100 hashes; token 3282 |
| Decode full logits | **not equal** | bitwise equal per row | finite; 94,910/201,088 values differ; max diff 0.375; same argmax 45427; different top-100 hashes |

Thus this is not run-to-run nondeterminism. It is a stable, logical-row or
physical-page-dependent difference introduced on the decode path. The
two-layer gate must remain failed until the first differing device boundary is
identified and made bitwise equal; the result must not be relabeled
deterministic merely because its greedy token matches.

## Findings

### 1. Sequential prefill output aliasing is refuted for the returned logits

`Generator._gather_prefill_logits` performs a blocking device-to-host transfer
and constructs a new Torch allocation with `torch.cat`; `prefill_forward` then
constructs the batch result with `torch.stack`
(`tt/generator.py:166`-`tt/generator.py:171` and
`tt/generator.py:216`-`tt/generator.py:235`). The test additionally clones the
returned tensor before the second prefill (`tests/test_full_model.py:591`-
`tests/test_full_model.py:596`). More importantly, all four prefill row/run raw
hashes in the diagnostic artifact are identical. A later prefill call is not
overwriting or aliasing the earlier host logit row.

This does **not** yet prove that the two physical KV page ranges contain
bitwise-identical K/V. Prefill attention computes from its live Q/K/V and only
writes the quantized K/V to the paged cache; its logits can be identical even
if a page-fill or cache-storage boundary differs. The implementation fills
each prompt sequentially with `batch_size=1`, `user_id=0`, and a one-row page
table (`tt/generator.py:173`-`tt/generator.py:190`). The attention prefill then
uses `paged_fill_cache` with that explicit page table
(`models/demos/gpt_oss/tt/attention/prefill.py:98`-`prefill.py:126`). The
physical cache contents must therefore be inspected directly before paging is
cleared as a hypothesis.

Verdict: **returned-output aliasing refuted; physical KV equality still
unproven**.

### 2. Random device output, NaN semantics, and reset failure are refuted

All recorded tensors are finite, so `torch.equal` is not failing merely
because equal-position NaNs compare unequal. Each decode row has the same raw
SHA-256 in both reset/reuse runs and `max_abs_diff_between_runs` is zero. This
rules out an intermittently incomplete read or ordinary random kernel output
for this reproduction.

`Generator.reset` clears every layer's K/V after a dirty run and resets the
shared generator's mode, prior page table, sampling mode, and slot state
(`tt/generator.py:468`-`tt/generator.py:481`). The exact per-row repetition
after that path is evidence against stale cache data as the source of
run-to-run drift. It does not rule out a deterministic difference between the
two physical page ranges or batch lanes.

Verdict: **refuted as the headline cause**.

### 3. Host token, position, and page-table refresh is source-consistent, but trace must be A/B isolated

The failing call supplies identical tokens and positions for both rows and
uses `sampling_mode="host"`, `reset_batch=True`, and
`force_host_tokens=True` (`tests/test_full_model.py:597`-
`tests/test_full_model.py:608`). The autoport marks both slots freshly supplied
and forwards the complete host state (`tt/generator.py:298`-
`tt/generator.py:322`). The shared generator treats host sampling as a full
input refresh on every call, and its trace replay copies tokens, current
position/RoPE indices, and the page table before execution
(`models/tt_transformers/tt/generator.py:1942`-
`generator.py:2011`). There is no async-ahead device token selection in this
host-sampling call.

That inspection makes stale host state unlikely, but the current artifact
contains only traced decode. One eager `enable_trace=False` control is still
needed. If eager decode is equal and traced decode is not, the defect is in
trace-input preparation/copy/capture. If both differ with the same row hashes,
trace staging is refuted and localization should move into the model graph.

Verdict: **unlikely; one focused eager/traced A/B remains**.

### 4. The first proven differing boundary is full decode logits, not yet decoder versus terminal

The full model processes every layer before applying a decode-sharded final
RMSNorm and the TP-column-sharded BFP8 LM head
(`models/demos/gpt_oss/tt/model.py:374`-`model.py:425` and
`tt/model.py:455`-`tt/model.py:487`). Consequently, 94,910 differing logits do
not prove that the decoder hidden states differ. A small hidden-state delta can
be amplified or quantized into many BFP8 logit differences, while a terminal
batch-lane defect can also create the whole symptom from identical decoder
states.

Within each optimized multichip decoder, paged attention handles both users in
one batched operation and indexes their distinct page-table rows. The sparse
MoE explicitly splits B>1 into one device-only execution per logical user and
concatenates the outputs
(`tt/multichip_decoder.py:1516`-`tt/multichip_decoder.py:1537`). Therefore the
strongest source-level suspects, pending boundary data, are:

1. different K/V contents or physical-page behavior in paged decode attention;
2. a decode attention/norm batch-lane difference before the per-user MoE;
3. final decode RMSNorm or LM-head batch-lane behavior;
4. trace staging, only if the eager control passes.

No evidence currently selects one of the first three. Editing any of them now
would be speculative.

## Minimal verify/refute experiment

Use the same two-layer model, identical 214-token prompt, and disjoint page
ranges. Delay assertions until all compact hashes are written. The following
ladder avoids a broad full-stack rerun and keeps every comparison exact.

### A. Separate trace, logical-row, and physical-page effects

1. Repeat the current decode once with `enable_trace=False` and once with
   `enable_trace=True`, rebuilding the same prefill before each. Record raw
   hashes and max differences.
2. Repeat after swapping the two complete page-table rows at both prefill and
   decode. If a decode hash follows the physical page range, investigate cache
   fill/update/read. If it stays attached to logical row 0 or 1, investigate a
   batch-lane operator.
3. Before decode, read only the four physical blocks covering tokens 0-213 from
   K and V for both users, for every reduced layer and TP rank. Compare logical
   K/V values bitwise and record the first `(layer, K-or-V, rank, block,
   element)` difference. Do not compare unused bytes outside the logical
   prompt.
4. Record exact hashes of the staged token, current-position, RoPE-index, and
   page-table trace inputs. The two token and position entries must be equal;
   only page IDs may differ.

Interpretation:

- K/V differs before decode: the first boundary is sequential prefill/cache
  fill, even though prefill logits match.
- K/V is equal and the output follows physical pages: the first suspect is
  `paged_scaled_dot_product_attention_decode` page lookup/read behavior.
- K/V is equal and the output follows logical rows in eager and trace modes:
  the first suspect is a batched decode operator or terminal batch lane.
- Only trace differs: fix the persistent trace-input copy/capture contract.

### B. Split decoder hidden state from the terminal head

For each diagnostic boundary, start from a fresh reset+prefill so a host read
cannot perturb the next comparison.

1. Reproduce `ttnn_decode_forward` through embedding, RoPE lookup, and
   `_forward_layers_and_head(..., skip_lm_head=True)`. Read the replicated
   logical `[2, 2880]` decoder output and compare the two rows bitwise.
2. On that exact hidden state, run final RMSNorm separately and compare rows.
3. On the exact normalized state, run the existing BFP8 LM head separately and
   compare each TP vocab shard before host concatenation.
4. If decoder hidden already differs, repeat with one layer, then capture the
   first layer's input-norm output, attention output/post-attention residual,
   and post-MoE output in separate fresh executions. This finds the first
   differing component without changing production precision.
5. At the first differing component, use an exact CPU/reference tail
   substitution only as a localization control. Do not keep host work in the
   model.

This is the smallest experiment that distinguishes a decoder/cache error from
a terminal projection error. Comparing only argmax or PCC cannot do so.

## Smallest correct fix, conditional on the experiment

- If the trace-only control fails, fix the row mapping or in-place refresh of
  the persistent token/position/RoPE/page-table tensor. Keep the model graph
  unchanged.
- If physical K/V differs, fix the first page-fill ownership, synchronization,
  or page-row mapping error. Do not normalize the resulting logits afterward.
- If paged attention is first, fix its batch/page-table layout or program
  contract so identical logical inputs and cache values produce identical
  rows. A sequential host decode or single-user fallback is not acceptable.
- If a decoder norm/attention/MoE boundary is first, change only that operator's
  batch-safe memory/program configuration and re-run the established optimized
  policy gates. Do not change weight, activation, KV, or CCL dtype merely to
  hide the symptom.
- If the final RMSNorm or LM head is first, keep the sharded/device-resident
  terminal path and select a batch-safe sharding/program configuration. Do not
  fall back to host logits, a replicated host head, or a less optimized
  production path.

After the focused two-layer experiment passes, rerun the original reduced gate
unchanged and then the 36-layer acceptance gate. Both must retain finite
logits, bitwise equality across rows and reset/reuse runs, identical raw and
top-100 hashes, and the explicit validation-only host-logit boundary.

## Final status

**Still failing.** The evidence establishes a deterministic decode
batch-position divergence and refutes returned prefill aliasing, NaNs, random
replay behavior, and reset/reuse drift. It does not yet identify whether the
first differing value is in paged KV state, a decoder layer, final RMSNorm, or
the LM head. The A/B and boundary ladder above is required before any fix is
kept.
