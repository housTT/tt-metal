# AutoDebug: batch-32 traced-replay nondeterminism

## Headline

The failure is localized to an optimized-decoder delta, not to the accepted
paged-cache/trace harness: optimized batch 32 sliding attention completes
prefill, eager decode, capture, and both replays but the second replay is not
bitwise equal to the first. The same-environment fused-decoder batch-32 control
passes for sliding and full attention, and optimized batch 1 and batch 2 replay
determinism pass.

No hardware experiment was run in this source-only investigation. The most
useful next step is a one-delta-at-a-time A/B in the existing batch-32 harness.

## Prioritized hypotheses and focused experiments

### H1: optimized router residency/path is corrupted or trace-unsafe at 32 users

`_OptimizedMLP` replaces the inherited router with `_L1Router`, deallocating the
old router weights and creating persistent L1-resident weight and bias tensors
(`optimized_decoder.py:319-332`). Decode batch 32 is then split into 32
single-user calls by `_FusedMLP.__call__` (`fused_decoder.py:874-888`), so the
same L1 router tensors are consumed repeatedly inside one captured graph. This
is the only router delta from the passing fused control and fits the batch
boundary: batch 1 and 2 exercise only one or two repetitions.

The generic `TopKRouter` has a special fused branch when
`actual_tokens == 32` (`topk.py:119-128`), while `_L1Router` hard-disables it
(`optimized_decoder.py:313-316`). However, the current inherited `_FusedMLP`
also explicitly sets its original router's `use_fused_op = False`
(`fused_decoder.py:281-286`). Therefore the definite optimized delta is L1
residency/replacement; a fused-selector difference must be confirmed from a
runtime op trace rather than assumed.

Focused A/B:

1. Keep optimized attention and both optimized norms unchanged, but retain the
   inherited DRAM `TopKRouter` instead of replacing it with `_L1Router`. Run the
   exact failing sliding node twice. A pass isolates H1.
2. Log the router op names during capture and assert `use_fused_op` immediately
   before capture for both fused and optimized controls. If the graphs differ,
   force both to the same non-fused path before comparing DRAM versus L1.
3. If DRAM passes, test L1 weight-only, L1 bias-only, then both. Compare router
   indices and scores after replay 1 and replay 2 before running expert
   matmuls. This distinguishes persistent-tensor corruption from downstream
   indexed-expert behavior.

### H2: the optimized sharded post-attention norm triggers a trace-unsafe split fallback

`_DecodeShardedRMSNorm` returns a ten-way L1 width-sharded tensor
(`optimized_decoder.py:335-371`). The batch-32 MLP immediately splits that
tensor into single-user views. The failing log emits, during eager and capture,
`ttnn.split: derived shard [1, 288] not tile-aligned; DRAM downgrade` and
`migrating L1 input ... to DRAM before slice fallback`. The passing fused path
uses canonical interleaved RMSNorm and does not show this optimized-only
layout boundary.

Focused A/B:

1. Replace only `post_attention_layernorm` with canonical `RMSNorm`; leave the
   optimized input norm, attention, router, and MLP unchanged. Re-run the
   sliding batch-32 node and verify that the split warnings disappear.
2. If still failing, restore the post-attention optimized norm and replace only
   the input norm. If post-only passes but input-only does not, the causal edge
   is the sharded-norm-to-split conversion, not RMSNorm arithmetic.
3. Before full-decoder replay, capture `post_attention_layernorm -> split ->
   concat` alone at batch 32 and compare every split output bitwise across two
   replays. Sweep batch 31/32/33 to identify a tile-boundary defect.

### H3: custom decode QKV linear/compute configuration is unstable at batch 32

The optimized attention replaces canonical `ttnn.matmul` plus in-place bias
add (`attention/decode.py:63-65`) with `ttnn.linear` carrying bias and an
explicit projection compute configuration (`optimized_decoder.py:89-102,
488-496`). The remainder of paged cache update and SDPA is structurally the
same. Max-batch configurations automatically select BF16 attention weights
(`optimized_decoder.py:427-429`), so a large replay delta would suggest a
buffer/program race rather than ordinary reduced-precision noise.

Focused A/B:

1. Keep all other optimized components and replace only QKV `ttnn.linear` with
   the canonical matmul plus in-place bias add. Run the exact batch-32 node.
2. If that passes, restore `linear` and independently remove the explicit
   compute-kernel config, then remove the fused bias. This isolates factory,
   compute config, and bias fusion.
3. Capture attention alone with a fresh external KV cache, snapshot Q/K/V and
   attention output after each replay, and identify the first differing tensor.

## Lower-priority explanations

- **Trace-input alias/lifetime:** `decode_forward` clones caller hidden state to
  DRAM before `_forward` (`fused_decoder.py:1211-1220`), while current position,
  RoPE tensors, and page table remain live across capture and both replays. The
  same contract passes in fused batch 32 and optimized batch 1/2. Still verify
  their host snapshots before and after each replay if H1-H3 fail.
- **Paged KV mutation:** both paths perform the same K/V paged updates before
  SDPA. Replaying identical hidden state at identical positions should write
  identical values. The test performs eager decode before capture, so position
  33 is initialized before replay; visible differences also include a user at
  an already-prefilled position. Snapshot only the physical cache pages touched
  by `current_position` after eager, capture, replay 1, and replay 2 to falsify
  this cleanly.
- **Precision instability:** fixed BF16/BFP8 arithmetic may differ from a
  reference but cannot by itself explain two executions of the same captured
  program producing different bits. Treat it as a symptom amplifier unless an
  upstream state or scheduling difference is first demonstrated.
- **FullLocal MoE:** it is not selected for `max_batch_size=32`; the allowlist is
  capped at batch 2. Its separate batch-boundary behavior is not causal here.

## Decisive result matrix

Run H1 DRAM-router, H2 canonical-post-norm, and H3 canonical-QKV as separate
single-delta variants under the exact same seed and node. For each, record
replay PCC, differing element count, maximum absolute delta, differing user
indices, captured op list, and the two `ttnn.split` warnings. The first variant
that restores bitwise equality should then be repeated three times and checked
for both sliding and full attention before changing production code.
