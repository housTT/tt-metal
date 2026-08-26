# QSA functional-decoder AutoDebug report

Date: 2026-08-26

Final-status note: this fresh-context diagnosis preceded the independent stage
review. The underfilled fixed-`k` risk described below was subsequently
verified, repaired with separate complete-block/tail validity masks, and closed
by `topk_multiset.log`, post-fix real PCC, traced replay, and the final watcher
run. The report below is retained as the point-in-time diagnostic record.

Scope: Qwen/Qwen3.8-Flash-Next layer-3 Qwen sparse attention (QSA), with the
reported real-weight prefill PCC of `0.27623364`. Layers 0 and 1 passed above
`0.998`. This investigation did not run TT hardware. The post-repair device
results quoted below were supplied by the parent repair run.

## Executive diagnosis

The catastrophic layer-3 failure was caused by QSA-specific index/address
preparation, not by the Qwen attention equations or the shared hyperconnection
and MoE implementation.

1. **Verified root cause: integer page decomposition was not exact on device.**
   The former int32 scalar `floor_div` path returned zero for these operands.
   QSA uses the quotient and remainder to map virtual token and compressed-block
   indices through a shuffled page table. A zero quotient aliases virtual pages
   to page-table entry zero, so the indexer and main K/V gather consume the
   wrong physical cache rows. The repaired code uses exact power-of-two shifts
   and masks in both `_physical_compressed_ids` and
   `_virtual_to_physical_tokens`.

2. **Verified root cause: multi-row `ttnn.embedding` indices did not preserve
   row order for this tiled path.** The main cache gather formerly presented a
   two-dimensional index matrix whose leading dimension was greater than one.
   Rows were reordered, attaching gathered K/V vectors to the wrong
   query/token rows. The repaired path flattens indices to one row before
   `ttnn.embedding`, then reshapes the result back. Rotation-table lookups use
   the same safe convention.

3. **Verified additional batched fix: index keys are explicitly replicated to
   all four indexer heads.** This is required for the batched QSA path. It is not
   the explanation for the original `max_batch=1` prefill PCC, but removes a
   separate batch-dependent broadcast ambiguity.

After these repairs, the same real layer-3 traced test reports:

- prefill PCC: `0.99679226`
- decode PCC: `0.99988198`

Both exceed the `0.995` gate. That before/after result closes the causal chain
for the reported catastrophic failure.

There was also a real source-level discrepancy in the underfilled QSA top-k
logic: TT always requests 512 blocks while HF requests only the number of
complete visible blocks. It was not the cause of the observed `0.27623364`, but
the later stage-review remediation fixed it rather than accepting the residual
risk.

## Why the verified fixes explain the observation matrix

Only layer 3 enters `_qsa_prefill`, paged K/V and indexer caches, virtual-to-
physical mapping, and gathered sparse attention. Layers 0 and 1 use the linear
attention path, so their greater-than-`0.998` PCC is compatible with the shared
hyperconnection and MoE code being sound while QSA fails.

For the failing test, the page geometry is:

- logical sequence: 33 tokens, physically padded to 128 for prefill;
- page size: 64 tokens;
- configured QSA context: 4096 tokens, hence 64 physical pages;
- compression ratio: 4, hence 16 compressed blocks per page;
- main gather width: 2080 entries per query row.

The intended address equations are exact integer operations:

```text
token virtual page       = token_id >> 6
token offset in page     = token_id & 63
physical token row       = page_table[virtual_page] * 64 + offset

compressed virtual page  = compressed_block_id >> 4
compressed offset        = compressed_block_id & 15
physical pooled-key row  = page_table[virtual_page] * 16 + offset
```

If the quotient becomes zero, every virtual page consults
`page_table[..., 0]`. With the shuffled page table used by the real test, this
is data aliasing, not a small numerical approximation. It corrupts both the
block-indexer key source and the K/V source that attention ultimately consumes.

The second error occurs after the addresses are computed. The gathered cache
indices logically have shape `[batch, tokens, width]`; for this prefill that is
effectively 128 rows by 2080 columns. The affected tiled embedding path can
reorder a two-dimensional index input when its leading dimension exceeds one.
Flattening to `[1, batch * tokens * width]` makes the operator's row convention
unambiguous. The output can then be reshaped to
`[batch, tokens, width, kv_heads, head_dim]` without changing logical order.

Together, these defects predict broad K/V permutation/aliasing and therefore a
catastrophic attention error. They also predict the layer specificity and the
large jump after the localized repairs. In contrast, ordinary BF16 error would
not plausibly explain PCC `0.27623364` while adjacent real-weight layers exceed
`0.998`.

## Current repaired dataflow

The current implementation's relevant chain is:

1. `_qsa_projections` creates Q, K, V, the sigmoid gate, index Q, and raw index
   K, applies per-head RMSNorm and 64-lane partial RoPE.
2. `paged_fill_cache` stores K, V, and raw index K under the chunk page table.
3. `_physical_compressed_ids` uses `>> 4` and `& 15` to retrieve the pooled
   raw-index keys through the full shuffled page table.
4. `_selected_virtual_tokens` scores complete blocks and constructs virtual
   token indices and validity lanes.
5. `_virtual_to_physical_tokens` uses `>> 6` and `& 63` for exact physical K/V
   rows.
6. `_gathered_qsa_attention` flattens all physical indices into one embedding
   row, reshapes gathered values back to token order, expands 2 KV heads to 24
   query heads, then performs scaled attention.

The exact-power-of-two arithmetic appears at
`functional_decoder.py:1203-1232` and `functional_decoder.py:1320-1335`. The
single-row embedding workaround appears at `functional_decoder.py:1144-1157`
and `functional_decoder.py:1337-1367`.

## Ranked remaining hypotheses and risks

### 1. Underfilled fixed-`k` top-k can admit masked filler blocks and duplicate the tail

**Final status: fixed; not the verified cause of the original catastrophic
failure.**

HF computes:

```python
scores.topk(min(self.block_topk, num_complete_blocks))
```

and appends only the incomplete visible tail. At sequence length 33, each query
has at most eight complete blocks, far below the 512-block budget. Therefore HF
selects every visible token exactly once; QSA is semantically ordinary dense
causal attention for this test. See the HF 5.16 source at
`modeling_qwen4_exp.py:669-702`.

TT masks invisible blocks to `-1e4` but always executes `topk(k=512)`, expands
all returned blocks to 2048 token indices, and appends a four-position tail.
Validity is later recomputed only as `selected <= current_pos`; it does not
retain whether a top-k block was complete and visible. See
`functional_decoder.py:1281-1318`.

If a tied filler block equals `complete = floor((position + 1) / 4)`, its
already-visible tokens occur once through that block and again through the
appended tail. HF's boolean mask does not have duplicate multiplicity.

A CPU-only actual-weight control that deliberately duplicated exactly this
incomplete tail measured:

- QSA attention-output PCC: approximately `0.994305`;
- QSA attention relative L2 error: approximately `0.1077`;
- complete real layer-3 output PCC: approximately `0.992716`.

This proves that such duplication could fail the `0.995` bar. It does **not**
prove that the current Blackhole top-k returns the necessary filler indices.
The routed `topk_large_indices` path documents deterministic but unspecified
tie order for `stable=false`, and the repaired device result of `0.99679226`
passes. Consequently this issue is not part of the established explanation for
the old `0.27623364` result.

**Decisive check completed:** `topk_multiset.log` reads back `selected` and
`valid` around block/page/tile/budget boundaries and requires the exact HF
multiset `{0, ..., p}` with multiplicity one. The implementation now propagates
complete-block eligibility separately from incomplete-tail validity and does
not rely on masked-value tie order.

### 2. QSA cache/gather geometry should retain an exact-value regression

**Status: fixed, but high-value regression coverage.**

The existing page-table permutation test checks end-output invariance, which
can miss two consistently wrong mappings. Add a device microtest whose cache
values encode virtual token, physical page, KV head, and feature lane. With a
shuffled page table, compare exact values at these boundaries:

- virtual tokens 0, 63, 64, and 127;
- compressed blocks 0, 15, 16, and 31;
- KV heads before expansion and query heads after the 12-way expansion;
- at least two query rows in one embedding invocation.

Expected query heads 0-11 must use KV head 0, and heads 12-23 must use KV head
1. This should be exact equality, not PCC.

### 3. Batched index-head replication and per-user paging need an exact tagged test

**Status: repaired; not implicated in the single-batch failure.**

Use two batch rows with distinct shuffled page tables and distinct current
positions. Tag raw index keys per batch and verify that all four indexer heads
receive the correct batch's same key tensor. Then verify selected physical IDs
remain within that user's pages. This isolates the new explicit head
replication from K/V attention and MoE.

### 4. Long-context indexer arithmetic is not exercised by the 33-token failure

**Status: demoted for the reported symptom.**

At length 33, all complete visible blocks fit in the budget, so pooled-key
scores cannot change the HF selected set. Differences such as TT BF16 pooling
versus HF float pooling and top-k ordering matter only once there are more than
512 complete visible blocks. They cannot explain the initial short-context PCC
collapse, but need separate long-context selection-quality coverage.

## Focused verify/refute sequence for any recurrence

If layer-3 PCC regresses again, locate the earliest divergence rather than
starting from the final layer output:

1. **Exact page arithmetic:** dump virtual page, in-page offset, page-table
   result, and final physical row for the token and compressed-block boundary
   cases above.
2. **Embedding row order:** feed the same indices once as a multi-row matrix and
   once as one flattened row. After reshaping, require exact equality. The
   flattened path is the expected one.
3. **Projection boundary:** compare TT and HF mixed input, Q, gate, K, V, index
   Q, and raw index K before caches. This also distinguishes upstream
   hyperconnection error from QSA error.
4. **Cache boundary:** inverse-map the shuffled page table and compare cached K,
   V, and raw index K token-by-token and head-by-head.
5. **Gather boundary:** compare selected virtual IDs, physical IDs, valid-token
   multiplicities, K/V before GQA expansion, and K/V after expansion.
6. **Attention boundary:** compare logits, additive mask, softmax probabilities,
   attention values, sigmoid gating, and output projection. Run a dense causal
   gather of `{0, ..., p}` once to bypass the indexer.
7. **Paging contrast:** repeat only after intermediates exist with identity and
   shuffled page tables. Failure only under shuffling points back to address
   mapping; failure under both points to row layout or attention math.

Readbacks for diagnostics should occur outside `ForbidHostFallback`; the
measured execution should retain the no-host-fallback gate.

## Refuted or strongly demoted explanations

- **Q/gate split:** HF views each q-projection head as `2 * head_dim` and chunks
  that final dimension. TT reshapes the raw weight as
  `[num_heads, 2, head_dim, hidden]`, which is the corresponding per-head split.
- **Zero-centred RMSNorm:** HF multiplies by `1 + weight`; TT adds one during
  weight upload for Q, K, and indexer norms.
- **Partial RoPE:** both paths rotate the first 64 of 256 head lanes with the
  same theta and preserve the remaining lanes.
- **128-row prefill padding:** zero-padded future rows cannot influence logical
  rows 0-32 under a correct causal validity mask. CPU emulation of padded and
  unpadded logical rows confirmed this.
- **Indexer score quality at length 33:** HF selects all 0-8 complete blocks, so
  scores only permute an identical key set.
- **Shared MoE as the primary cause:** layers 0 and 1 exercise the same real
  hyperconnection/MoE machinery and pass. MoE can amplify a bad QSA block, but
  the repaired QSA address/layout changes are sufficient to restore the full
  layer above the gate.

## Conclusion

The old `0.27623364` result is accounted for by two QSA-specific contract
violations at the virtual-index to physical-cache boundary: inexact integer
page decomposition and corrupted multi-row embedding index order. Exact
bitwise page math plus single-row embedding indices restore real prefill and
traced decode correctness. The fixed-512 underfilled top-k behavior was a
separate semantic bug, not the established catastrophic cause; the later
validity-mask repair and exact multiplicity regression close it without
changing that diagnosis.
