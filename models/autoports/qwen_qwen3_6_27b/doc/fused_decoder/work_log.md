# Qwen/Qwen3.6-27B — fused decoder work log

Stage 2 of the repo-local TTNN autoport pipeline: take the completed functional decoder
(`tt/functional_decoder.py`, stage 1) and fuse its op graph, preserving semantics, the context
contract, and the PCC bar.

Hardware, environment and device policy are unchanged from stage 1 — one Blackhole chip
(`/dev/tenstorrent/2`, `TT_VISIBLE_DEVICES=2`) of the intact p300c board, 1x1 mesh, this
checkout's own `python_env`, sourced through
`doc/functional_decoder/ttenv.sh`. A mesh smoke (`ttnn.open_mesh_device` /
`close_mesh_device`) was run before starting and returned `MESH_SMOKE_OK`; no reset, no ARC or
Ethernet fault, and no hang occurred at any point in this stage, so `$autotriage` / `$autofix`
were never needed for hardware.

---

## 0. Method

`$graph-fusing` in the order the skill sets out, iterated:

1. **Step 1** — inventory the dedicated ops this tree has that could replace parts of the
   graph.
2. **Step 2** — write the op table out of the stage-1 `tt-perf-report` CSVs (device time per op
   for one warmed prefill and one traced decode replay, per layer kind).
3. **Step 3** — classify each candidate subgraph, dedicated-op rewrites first, then graph
   rewrites, then op merging.
4. **Step 4** — apply one, prove it on device (op-level probe where a model-free reproducer is
   cheaper, layer-level PCC against HF always), keep only if it is both correct and faster.
5. **Step 5** — re-profile and go back to step 2.

Four full passes of 2→5 were run. Each pass's table is below.

---

## 1. Step 1 — dedicated ops found in this tree

Searched `ttnn/cpp/ttnn/operations/**`, `ttnn/ttnn/operations/*.py`, `tests/ttnn/**`,
`models/common/modules/`, and the model-family references the skill points at. The relevant
find is that **this checkout already contains a Qwen3.6 demo** — `models/demos/blackhole/qwen36/`
— plus the op library it is built on (`models/experimental/gated_attention_gated_deltanet/`),
so the idiomatic fused op sequence for both of this model's mixers is available to read.

| op | what it replaces here | verdict |
|---|---|---|
| `ttnn.transformer.chunk_gated_delta_rule` | the whole `linear_attention` prefill delta-rule core | **taken** (§3.1) |
| `ttnn.transformer.gated_delta_attn_seq` | the same core, but as a scan kernel fed by python/ttnn preprocessing | rejected: strictly more ops than the above, which is the fully-fused successor (`fused_chunk.py` in the demo says so and uses the fused op for prefill) |
| `ttnn.experimental.rotary_embedding_hf` | prefill partial RoPE | **taken** (§3.2) |
| `ttnn.experimental.rotate_half` | decode partial RoPE's slice/slice/neg/concat | **taken** (§3.2) |
| `ttnn.experimental.rotary_embedding` / `rotary_embedding_llama` / `rotary_embedding_llama_fused_qk` | ditto | rejected: rotate-half over the **whole** head_dim, and this model's rotary factor is 0.25, so they cannot express the partial rotation without permuting head channels (§3.2) |
| `ttnn.experimental.paged_fused_update_cache` | the two decode `paged_update_cache` calls | rejected with an exact op-contract blocker (§3.5) |
| `ttnn.experimental.group_attn_matmul` | the decode recurrent-state read | rejected with an exact op-contract blocker (§3.6) |
| `ttnn.rms_norm` + `LayerNormShardedMultiCoreProgramConfig` | the decode RMS norms, which run on one core interleaved | **taken** (§3.3) |
| `ttnn.rms_norm` | the decode GatedDeltaNet Q/K L2 norm's `mul/sum/rsqrt/mul` | **taken** (§3.6) |
| `ttnn.experimental.nlp_create_qkv_heads_decode` / `nlp_concat_heads_decode` / `nlp_create_qkv_heads` / `nlp_concat_heads` / `paged_fill_cache` / `paged_update_cache` / `chunked_scaled_dot_product_attention` / `paged_scaled_dot_product_attention_decode` | — | already used by stage 1; kept |
| `ttnn.conv1d` / depthwise `conv2d` | the GatedDeltaNet causal conv1d | rejected: `conv_dim` is 10240 and the tt-metal conv CBs overflow L1 past ~2048 channels (the same reason `models/experimental/gated_attention_gated_deltanet` falls back to an FIR at `D > 2048`) |
| `ttnn.experimental.hc_sum_reduce` / `repeat_and_interleave_eltwise_mul` | the per-head gated RMS norm's group reduce/expand | rejected: SSM ops fixed to 32-wide groups; this model's group is `head_v_dim` = 128. The same algebra is expressed with two constant matmuls instead (§3.4) |

---

## 2. Step 2 — the stage-1 op table (the "before")

From `doc/functional_decoder/tracy/*/{prefill,decode}_perf_report.csv`, warmed, batch 1,
2048-token prefill, traced decode at position 2048 (mean of 8 replays):

<!-- GENERATED:before_breakdown -->
| bucket | `linear_attention` prefill | `linear_attention` decode | `full_attention` prefill | `full_attention` decode |
|---|---|---|---|---|
| `matmul` | 14.387 ms | 1.985 ms | 13.465 ms | 1.812 ms |
| `batched_matmul` (the spelled-out delta rule / recurrence) | 78.985 ms | 0.259 ms | — | — |
| `sdpa` | — | — | 1.279 ms | 0.104 ms |
| `layout` (tilize/untilize/reshape/permute/concat/slice/shard) | 33.429 ms | 0.288 ms | 1.142 ms | 0.038 ms |
| `elementwise` | 23.501 ms | 0.290 ms | 1.853 ms | 0.053 ms |
| `norm` | 0.599 ms | 0.213 ms | 0.536 ms | 0.219 ms |
| `heads_and_cache` | — | — | 0.313 ms | 0.045 ms |
| `other` | 0.072 ms | — | — | — |
| **total** | **150.971 ms** | **3.034 ms** | **18.588 ms** | **2.270 ms** |
| ops in one pass | 801 | 92 | 44 | 50 |
| op-to-op gap | 8.115 ms | 0.359 ms | 0.022 ms | 0.046 ms |
<!-- END GENERATED:before_breakdown -->

Reading of the table: the `linear_attention` prefill is entirely op-count bound - 801 ops, and
its `batched_matmul` bucket alone (the spelled-out delta rule, single-core `32x32x32` blocks) is
more than twice the whole fused pass - and it is a spelled-out version of an op this tree already
has. Both decodes are dominated by the MLP's two matmuls, which the profiler's own DRAM column
puts at the roofline for bfloat16 weights and which no graph rewrite can improve. And both pay
for two RMS norms that the interleaved kernel runs on a single core.

---

## 3. Step 3/4 — the rewrites, in the order they were applied

### 3.1 `linear_attention` prefill delta rule → `ttnn.transformer.chunk_gated_delta_rule`

**Kind:** dedicated fused op.

The functional layer spells out `torch_chunk_gated_delta_rule`: head split, L2 norms, GQA
head expansion, `1/sqrt(head_k_dim)` scale, decay cumsum and mask, the WY (unit-triangular)
inverse by recursive 2x2 block inversion, and a python loop over the 32 sub-chunks of a
2048-token chunk. `ttnn.transformer.chunk_gated_delta_rule` is a single op that does all of it.

**Op-level probe first**, because the op has two call shapes and one of them is numerically
much better: `doc/fused_decoder/probes/probe_chunk_gdr.py` builds this model's exact shapes
(`Nk` 16, `Nv` 48, `Dk` = `Dv` 128) with q/k/v/beta/g in the ranges the real layer produces and
compares against HF's `torch_chunk_gated_delta_rule` in float32
(`logs/probe_chunk_gdr.log`):

<!-- GENERATED:gdr_call_shapes -->
| call shape | chunk | seq | output PCC | final-state PCC | best wall |
|---|---|---|---|---|---|
| flat rank-3 `[1, T, H*D]` | 32 | 64 | 0.999994 | 0.999995 | 0.37 ms |
| flat rank-3 `[1, T, H*D]` | 32 | 2048 | **0.999994** | 0.999994 | **3.79 ms** |
| split rank-4 `[1, T, H, D]` | 64 | 64 | 0.999992 | 0.999992 | 0.55 ms |
| split rank-4 `[1, T, H, D]` | 64 | 2048 | **0.903635** | 0.996086 | **6.98 ms** |
<!-- END GENERATED:gdr_call_shapes -->

The rank-4/chunk-64 row is the documented failure mode: at chunk 64 each per-chunk WY matrix is
a 2x2 tile block whose bottom-right 32x32 sub-block can be ill-conditioned enough for the
float32 block inverse to lose precision. The flat path at chunk 32 is both exact-to-PCC and
faster, and it additionally L2-normalises Q/K and folds the scale in-kernel (`qk_norm = flat_qk
&& C == 32` in `chunk_gated_delta_rule.cpp`) and does the GQA head expansion itself — so the
`l2norm`, `repeat_interleave`, head permute and scale multiply all disappear from the graph as
well. It requires `T % 32 == 0`; the layer's padded chunk length is always a multiple of
`DELTA_CHUNK` = 64, so that always holds.

Kept. `use_qk_l2norm=True` is **not** usable (`TT_FATAL: use_qk_l2norm not yet supported`) — the
in-kernel norm comes from the flat path, not from that flag.

Constant tiles (`eye`/`tril`/`ones`/`masks`) are built in `from_state_dict` and passed in: the
op's own fallback build does a host upload, which is illegal under trace capture.

### 3.2 partial RoPE

**Kind:** dedicated fused op (prefill), dedicated fused op for a sub-pattern (decode).

`partial_rotary_factor` is 0.25, so only the leading 64 of 256 head channels rotate. The
functional layer spells the rotation out in ~9 ops per tensor.

*Prefill* — `ttnn.experimental.rotary_embedding_hf` in prefill mode takes exactly this layer's
`[1, heads, seq, D]` input with `[1, 1, seq, D]` cos/sin, so applying it to the 64-wide rotary
slice collapses `slice/slice/neg/concat/mul/mul/add` to one op; only the rotary/passthrough
slice and the re-concat remain. Probe `probes/probe_small_ops.py`: PCC 0.999997 against the
spelled-out form on `[1, 24, 2048, 256]`.

*Decode* — the same op's decode mode needs a HEIGHT_SHARDED input **and** sharded per-user
cos/sin (`rotary_embedding_hf_device_operation.cpp` lines 55-68), and its prefill mode
broadcasts cos/sin over dim 1, which is the *batch* axis in the decode layout — so it cannot
serve per-user positions. Rejected for decode with that exact contract blocker; the rotate-half
itself is still replaced by `ttnn.experimental.rotate_half`, taking the decode RoPE from 10 ops
to 7.

**Also considered, built as a probe, and rejected on the measurement:** permuting head channels
at load time so that the rotary pairs land at `(c, c+128)` and the whole 256-wide head can go
through one `rotary_embedding_hf` with `cos = 1, sin = 0` on the passthrough lanes. That works
arithmetically — attention is a dot product over the channel axis, so a consistent permutation of
Q and K is invisible — and it removes the slice, the passthrough slice and the concat, which are
a couple of percent of the fused `full_attention` prefill. But it also rotates **four times** as
much data.
Measured at the real prefill shapes (`probes/probe_output_paths.py`, `logs/probe_output_paths.log`):

<!-- GENERATED:rope_width -->
| tensor | slice + 64-wide RoPE + slice + concat | permuted, one 256-wide RoPE |
|---|---|---|
| q, 24 heads | 0.443 ms | **0.361 ms** |
| k, 4 heads | 0.127 ms | **0.114 ms** |

so the whole permutation is worth **95 us of a 17.778 ms prefill, 0.5 %**.
<!-- END GENERATED:rope_width -->
 — and it costs three
things: the public `rot_mats` contract widens from `[…, rotary_dim]` to `[…, head_dim]` (4x the
cos/sin upload, which in *decode* is per-step host work inside the traced input buffers), the
paged K cache then holds permuted channels so the stage-1 comparison against HF's own cache
object needs an un-permute, and the decode RoPE gets 4x more data for no structural saving.
Rejected: measured upside too small for a public-contract and cache-content change.

### 3.3 decode RMS norms → width-sharded `ttnn.rms_norm`

**Kind:** graph rewrite (memory layout).

The decode hidden state is `[1, 1, batch, 5120]` — one tile row — so the interleaved
`ttnn.rms_norm` parallelises over exactly **one core**, and in the stage-1 report it is the
third-largest op of the step, twice
per decode step (three times counting the gated norm's own). Width-sharding the 5120 channels
across the grid and using `LayerNormShardedMultiCoreProgramConfig` fixes it. Measured
(`probes/probe_small_ops.py`, `logs/probe_small_ops.log`), wall for
interleaved→shard→`rms_norm`→interleaved:

<!-- GENERATED:norm_cores -->
| cores | 16 | 20 | 32 | 40 | 80 | interleaved |
|---|---|---|---|---|---|---|
| ms | 0.045 | **0.038** | 0.044 | 0.049 | 0.068 | 0.099 |
<!-- END GENERATED:norm_cores -->

16 and 20 swap places between runs by about the run-to-run spread; from 32 upwards the
shard/unshard overhead starts to dominate. `NORM_SHARD_CORES` is 20, with a fallback that picks
the largest divisor of `hidden_size / TILE_WIDTH` the device grid allows. PCC of the sharded
norm against torch: 0.999990, against 0.999928 interleaved. In the committed reports the sharded norm is an order of magnitude below the interleaved one it
replaces.

### 3.4 per-head gated RMS norm → group reduction, no relayout

**Kind:** graph rewrite (algebraic).

`Qwen3_5RMSNormGated` normalises over `head_v_dim` = 128, i.e. per head. The functional layer
therefore reshapes the flat `[1, 1, L, 6144]` tensors to `[1, L, 48, 128]`, norms, gates, and
reshapes back. In TILE layout those two reshapes are **full relayouts** — the tile grid of
`[L, 6144]` and of `[L*48, 128]` have nothing in common — and the first fused profile in the first fused profile they were the two most expensive ops of the whole pass, together more
than the fused delta-rule op itself.

The identical arithmetic is a group reduction that stays in the flat layout:

```
mean_h(core^2)        == core^2 @ gdn_group_mean      # [value_dim, 64], 1/head_v_dim in the entries
rsqrt(... + eps) * w  == rsqrt(...) @ gdn_scale_expand # [64, value_dim], the norm weight in the entries
```

Both constant matrices are exactly representable in bfloat16 (the mean factor is a power of
two; the scale entries are the norm weight, itself bfloat16), and both are built with explicit
zero rows/columns out to a tile (`_GDN_GROUP_PAD` = 64) rather than relying on tile padding, so
the padded lanes provably contribute nothing. The delta-rule op returns its output token-major
in ROW_MAJOR, where merging the trailing `[48, 128]` into `[6144]` is contiguous. That merge is
still a real op in the report - a ROW_MAJOR page-size change is a copy, not a restride - but it
is one copy instead of the two full TILE relayouts it replaces, and §3.13 measures the whole
alternative epilogue at about twice this one's cost.

Measured effect: the two relayouts - the two most expensive ops of the first fused profile - and
the head-axis tilize they needed
are gone from the profile entirely, replaced by two skinny constant matmuls that together cost
a fraction of one of them. Layer PCC is unchanged to the fourth decimal (§4).

### 3.5 decode K/V cache writes

**Kind:** graph rewrite (drop a reshard) + a rejected op merge.

`nlp_create_qkv_heads_decode` already emits exactly the height-sharded memory config
`paged_update_cache` wants — `probes/probe_decode_heads.py` prints the two configs side by side
and they compare equal for batch 1 and 4 — so the functional layer's
sharded→interleaved→sharded round trip on V is pure waste. Removed; V now goes straight from
the head op into the cache write. Q and K still take the interleaved trip because the norm and
the partial-RoPE slice need it.

`ttnn.experimental.paged_fused_update_cache` would merge the two cache writes into one op. It
was tried and **fails its own validation** on this graph:
`paged_fused_update_cache_device_operation.cpp:342` requires the two input tensors' shard core
sets to be *disjoint* (`TT_FATAL(!is_overlap, ...)`), and K and V come off the head op on the
same cores. Satisfying it means resharding V onto a second core range — trading one small op
for one reshard plus the fused op, i.e. a few microseconds on a ~2 ms step, at the cost
of a hand-built second core range that has to track the head op's own grid (which differs at
batch 32, as the same probe shows). Rejected with the contract blocker recorded.

### 3.6 decode recurrence matmuls → explicit core grid

**Kind:** graph rewrite (parallelisation, not a new op).

At batch 1 the single-token delta rule is 48 independent `[1,128] x [128,128]` state reads and
one `[128,1] x [1,128]` outer product per head. `ttnn.matmul`'s default batched program factory
puts them on 4 and 16 cores. `probes/probe_decode_recurrence.py`
(`logs/probe_decode_recurrence.log`), best-of-20 wall:

<!-- GENERATED:recurrence_grid -->
| shape | default | 1x4 | 1x8 | 1x11 | 2x4 | 2x8 | 2x11 | 4x4 | 4x8 | 4x11 | 6x4 | 6x8 | 6x11 | selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| state read | 114.7 (14.0) | 72.2 (3.8) | 56.3 (3.4) | 56.9 (6.1) | 50.9 (2.7) | 46.3 (3.5) | 48.5 (3.8) | 41.3 (2.7) | 44.8 (7.2) | 47.9 (2.7) | 39.1 (2.8) | 43.0 (4.2) | 43.2 (4.2) | 6x4 |
| outer product | 107.4 (10.6) | — | — | — | — | 43.7 (8.7) | 46.8 (10.9) | — | 45.1 (11.1) | 45.1 (7.1) | — | 43.2 (7.0) | 42.3 (4.9) | 6x8 |

Median and (stdev) in microseconds over 30 repeats. The default program factory is several times slower than any explicit grid; the explicit grids sit within about a stdev of each other, so the two selected are a representative pick from that flat region rather than a unique optimum.
<!-- END GENERATED:recurrence_grid -->

All exact (PCC 1.000000 against torch); the grid only changes how independent per-head problems
are distributed. Taken, clamped to the device's real grid. Device time for the three recurrence
matmuls in the final decode profile: about a fifth of what the functional stage's three take; both are rows of the committed
decode reports.

`ttnn.experimental.group_attn_matmul` was tried for the state read and rejected: its contract
ties the batch dimension to the number of users (`TT_FATAL: Num of users must match!`), and here
that dimension is `batch * num_v_heads`.

### 3.7 GatedDeltaNet causal conv1d

**Kind:** graph rewrite (dtype/layout of an existing sequence), plus one algebraic fix.

After §3.1 the 4-tap depthwise FIR over 10240 channels was the largest remaining prefill cost
(then the single largest entry of the prefill). It has two structural problems: every tap slices the
concatenated window at row 1/2/3, i.e. **off** a tile boundary, which makes `ttnn.slice` an
`untilize_with_unpadding` + `tilize_with_val_padding` sandwich; and the per-channel tap multiply
is a height-broadcast binary op.

Microbenchmark in `probes/probe_causal_conv.py` (`logs/probe_causal_conv.log`), on
`[1, 1, 2051, 10240]`:

<!-- GENERATED:broadcast_bandwidth -->
| multiply | float32 | bfloat16 |
|---|---|---|
| height-broadcast | 2.282 ms — **74 GB/s** | 0.441 ms — 191 GB/s |
| same-shape | 0.709 ms — 355 GB/s | 0.438 ms — 288 GB/s |
<!-- END GENERATED:broadcast_bandwidth -->

The float32 height-broadcast multiply is the outlier: 5x below what the same op reaches on
same-shape float32 operands. Four whole formulations of the FIR were measured against torch
(all PCC 1.000000 in float32, 0.999990 in bfloat16):

<!-- GENERATED:conv_formulations -->
| formulation | float32 median (stdev) ms | bfloat16 median (stdev) ms |
|---|---|---|
| all-TILE slices (what the functional layer does) | 17.061 (0.136) | 5.626 (0.046) |
| untilize once, ROW_MAJOR shift, tilize per tap (TILE concat) | 16.782 (0.077) | 5.494 (0.043) |
| ROW_MAJOR concat *and* shift, SiLU folded into the last add - **shipped** | 15.325 (0.049) | **4.714** (0.033) |
| untilize once, whole FIR in ROW_MAJOR, tilize once | **14.268** (0.048) | 7.425 (0.031) |
| one pre-padded window per tap so every slice is tile-aligned | 19.149 (0.032) | 6.834 (0.031) |
<!-- END GENERATED:conv_formulations -->

So the win is the dtype first - every bfloat16 row is about 3x its float32 twin - and the
layout second: building *and* shifting the window in ROW_MAJOR is the fastest row, by several
times the run-to-run spread and bit-identically. The FIR now runs in bfloat16 while the carried conv **state** stays
float32 and is taken from the float32 inputs — nothing downstream can use more than bfloat16
anyway (the conv output feeds `chunk_gated_delta_rule`, whose contract casts q/k/v to bfloat16),
and the checkpoint stores `conv1d.weight` in bfloat16 to begin with, so the bfloat16 taps are
that weight at its native precision rather than a reduction of it.

One algebraic fix on top: cutting the K state rows straight out of `mixed_qkv` untilizes the
*whole* 84 MB tensor, because neither end of the cut is on a tile boundary. The
cut now takes a tile-aligned two-tile block first — a plain tile copy — and does the ragged cut
inside it. This introduced a real aliasing bug on the first try (for short chunks the aligned
block is a full-range slice, i.e. a *view* of `mixed_qkv`, and freeing it freed the input); the
`_free()` guard the functional stage built for exactly this hazard fixes it, and
`test_prefill_decode_pad_below_one_tile` at 735..768 is the regression.

### 3.8 SwiGLU MLP — SiLU folded into the multiply

**Kind:** op merging.

`ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])` removes the
separate `silu` dispatch. Measured at the real shape (`probes/probe_mlp_variants.py`,
`logs/probe_mlp_variants.log`), all three variants PCC 0.999988:

<!-- GENERATED:mlp_variants -->
| variant | prefill 2048 | decode 32 |
|---|---|---|
| fused gate/up matmul + 2 slices + `silu` + `multiply` (functional) | 8.441 ms | 0.950 ms |
| fused gate/up matmul + 2 slices + `multiply(act=SILU)` | **7.991 ms** | 0.935 ms |
| split gate/up matmuls, `silu` on the gate matmul's `activation=` epilogue, `multiply` | 10.485 ms | **0.924 ms** |
<!-- END GENERATED:mlp_variants -->

The split variant is the textbook "matmul + activation" merge and it removes both slices, but at
prefill it is **the slowest of the three** in the table above — two `[hidden, intermediate]`
matmuls lose more than the slices cost — and at decode the difference is inside run-to-run noise. Rejected on
measurement; the SiLU fold is kept.

The same input-activation merge applies twice more, and both were taken: the attention output
gate's `sigmoid` — two ops in the functional prefill report, one in the fused one — and the
gated delta net's `silu(z)`.

### 3.9 `in_proj_b` + `in_proj_a` → one matmul

**Kind:** graph rewrite (shared-LHS peer merge).

Both read the normed hidden state, both are `num_v_heads` = 48 wide, and both are
float32-weighted (stage 1 kept them float32 deliberately: the softplus/exp gating is
numerically sensitive). At decode each was a `32 x 5120 x 64` matmul costing ~60 us on 2 cores —
dispatch-bound, not bandwidth-bound — so packing them into one `[hidden, 128]` weight halves
that. The second block starts at a tile-aligned column (`_AB_STRIDE` = 64) so both halves come
back out with a plain slice. In the committed decode reports the functional layer's two
`32 x 5120 x 64` matmuls become one `32 x 5120 x 128`, at half the cost. The two unpacked
weights are freed in `from_state_dict` rather than left on device.

Packing `in_proj_qkv` and `in_proj_z` in as well was considered and rejected: those two matmuls
are DRAM-bound on their weights (5120x10240 and 5120x6144 bfloat16), so merging them moves the
same bytes and saves only one dispatch, and it would force the float32 `a`/`b` weights to
bfloat16 (a precision change stage 1 explicitly avoided) or the bfloat16 qkv/z weights to
float32 (doubling 168 MB of weight traffic).

### 3.10 recurrent-state write-back folded into the add

**Kind:** op merging.

`new_state = state * decay + update` followed by `ttnn.copy(new_state, self.recurrent_state)`
becomes `ttnn.add(state, update, output_tensor=self.recurrent_state)`, removing a 3 MB float32
copy from every decode step while keeping the persistent buffer address the traced
decode requires.

### 3.11 decode causal conv state → batch-major tap buffers

**Kind:** graph rewrite (memory layout).

The functional decode keeps the conv state as one `[1, batch, K, conv_dim]` tensor and slices
row `j` out of it per tap. That puts the shift on the **tile-height** axis, so each tap costs an
`untilize_with_unpadding` + `tilize_with_val_padding` pair, and in the stage-1 profile those
pairs also carried tens of microseconds of op-to-op gap each - in the stage-1 report the conv's
gap is larger than its device time. The fused layer keeps `K-1` separate `[1, 1, batch, conv_dim]` buffers
instead: every tap is a whole-tensor elementwise read, and the shift is an in-place `ttnn.copy`
chain. The untilize/tilize pairs and their op-to-op gaps are gone from the committed decode
report entirely.

### 3.12 decode Q/K L2 norm → `ttnn.rms_norm`

**Kind:** dedicated fused op.

`FunctionalDecoder._l2norm` is `mul → sum → add → rsqrt → mul`, twice per decode step, plus the
query's separate `1/sqrt(head_k_dim)` multiply. With `mean(x^2) = sum(x^2)/D`,

```
rms_norm(x, eps/D) == x * sqrt(D) / sqrt(sum(x^2) + eps) == sqrt(D) * l2norm(x)
```

so `l2norm(x) == rms_norm(x, eps/D) / sqrt(D)`, and the query's extra scale folds into the same
constant (`1/D`, since `head_k_dim == head_v_dim`). Eleven ops become four. It is applied on
`[1, batch, heads, head_dim]`, before the reshape into the recurrence layout, because there the
norm axis is the last axis and the head axis is the tile height — the shape the norm kernel
wants. Layer PCC is unchanged to the fourth decimal.

### 3.13 `chunk_gated_delta_rule`'s output layout

**Kind:** dedicated-op option, assessed and rejected on the measurement.

The op takes `output_head_major`. With it off (the default) it returns token-major
`[B, T, HV, V]` ROW_MAJOR after an internal untilize + permute; with it on it returns
`[B*HV, T, V]` TILE for free, because that is what the kernel already produces. Head-major looks
strictly better in isolation — it also turns the per-head gated norm back into a plain
`ttnn.rms_norm` with no constant matrices — but the rest of the consumer chain (`z`, the z gate,
`out_proj`) is token-major-flat, so head-major buys the skipped epilogue back as a relayout of
`z` on the way in and of the gated result on the way out.

Both whole paths were built and timed from the op call to the flat `[1, 1, T, value_dim]` tensor
`out_proj` consumes, at the real shapes (`probes/probe_output_paths.py`,
`logs/probe_output_paths.log`):

<!-- GENERATED:gdn_epilogue -->
| path | 2048-token chunk |
|---|---|
| token-major output + group-reduction norm (§3.4) — **shipped** | **5.43 ms** |
| `output_head_major=True` + per-head `ttnn.rms_norm` + z/result relayouts | 10.74 ms |

PCC between the two outputs: 0.999994.
<!-- END GENERATED:gdn_epilogue -->

The shipped path is 2x faster; rejected on measurement.

### 3.14 causal-conv SiLU folded into the last tap's add

**Kind:** op merging.

The FIR ended with `acc = add(acc, term)` and then a standalone `ttnn.silu(acc)` — a full
read+write of a ~42 MB bfloat16 tensor on every prefill chunk. `ttnn.add` takes an
**output** activation list, so the SiLU rides the last tap's add instead. Same merge as the MLP's
SiLU, the output gate's sigmoid and the z gate's SiLU, and the same one the in-tree reference
makes (`models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py`'s
`ttnn.add(out, bias, activations=...)`). Applied to the decode FIR too.

`ttnn.addcmul` was considered for the whole tap (`acc + tap * w` in one op) and rejected: in this
checkout it is a *composite* that decomposes to multiply + multiply + add
(`ttnn/cpp/ttnn/operations/eltwise/ternary/ternary_composite_op.cpp`), i.e. one op worse than the
multiply + add it would replace.

### 3.15 KV-cache typecasts guarded on dtype

**Kind:** graph rewrite (remove a no-op).

`_full_attention_prefill` cast K and V to the cache dtype unconditionally. At the default
`cache_dtype = bfloat16` they are already bfloat16, so both casts were `bfloat16 -> bfloat16`
no-ops dispatched on 110 cores on every prefill chunk. Now guarded on
`k.dtype != k_cache.dtype`; `test_bfloat8_kv_cache` covers the branch that still casts.

### 3.17 decode z-gated norm: two forms, picked by batch

**Kind:** graph rewrite (the same algebraic one as §3.4, applied conditionally).

Round 1's §3.4 replaced the per-head gated norm's two tile relayouts with a group reduction in
*prefill*, and §6 recorded the decode side as "about the same" - an estimate, which a stage
review correctly refused. Measured at the real decode shapes, median over 25 repeats
(microseconds):

<!-- GENERATED:gated_norm_batches -->
| batch | 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| reshape + `ttnn.rms_norm` (us) | 73.5 | 81.4 | 92.4 | 151.8 | 238.2 |
| group reduction (us) | 149.4 | 148.6 | 150.6 | 151.7 | 151.5 |

Median over 25 repeats. Lowest PCC between the two forms' outputs, over all batches measured: 0.999993.
<!-- END GENERATED:gated_norm_batches -->

The group form is two skinny constant matmuls and barely moves with the row count; the reshape
form's two tile relayouts grow with it. They cross between 16 and 32, so the layer picks by
`max_batch` (`_GATED_NORM_GROUP_BATCH`) rather than committing to one, and
`test_batched_users[32-linear_attention]` covers the branch the batch-1 perf runs do not take.
Probe: `probes/probe_gated_norm_batch.py`, log `logs/probe_gated_norm_batch.log`.

### 3.16 the fused graph's own peak

After all of the above, more than half of the `linear_attention` prefill is the `matmul` bucket
— the four projections, the MLP's two matmuls and the two gated-norm constant matmuls — and a
tenth is `chunk_gated_delta_rule` itself. The largest remaining non-matmul cost is the causal
conv, which the probe measures in isolation and which §3.7 records four formulations for. Both
decodes are four fifths and seven eighths matmul time at the roofline the profiler's own `DRAM` column reports — the device's DRAM
roofline for bfloat16 weights, and the point at which graph fusing stops being the lever and
weight dtype starts.

The full bucket breakdown of all four passes is the `breakdown_ms` block of
`perf_summary.json`, reproduced in [`README.md`](README.md); it is derived from the report's own
op codes rather than added up by hand, and its `other` bucket is empty.

---

## 4. Correctness

Acceptance bar is unchanged: **PCC >= 0.995**, the same bar and the same HF reference harness as
stage 1. `tests/test_fused_decoder.py` is the stage-1 contract re-run against `FusedDecoder` —
same sequence-length coverage, same paged-cache and state comparisons, same batch, traced-decode,
determinism, block-size, BFP8-cache and full-context tests — plus five tests that only exist
because this is a fusing stage:

* `test_fused_ops_are_dispatched` — asserts each op in `FusedDecoder.FUSED_OPS` is actually
  called on a real pass, so a shape or dtype that silently missed a fused path fails loudly
  instead of quietly costing performance;
* `test_fused_graph_is_smaller` — the fused graph dispatches strictly fewer `ttnn` ops than the
  functional one for the same prefill and the same decode;
* `test_fused_matches_functional` — fused and functional outputs agree with **each other**, from
  identical weights and inputs, not only with HF;
* `test_repeated_runs_stable` — six prefill+decode cycles, bit-identical across cycles, with the
  device's per-bank allocated bytes required to be unchanged from cycle 1 onward (a per-cycle
  device leak in the `_free` aliasing rules is a failure);
* `test_no_relayout_or_host_ops_in_measured_decode` — the layer itself asks for no
  `tilize`/`untilize`/`to_layout` in a measured decode, and its reshard count stays inside a
  budget that each remaining reshard's op contract justifies.

Final results, all against the shipped code:

* `logs/suite_main.log` — **71 passed, 2 skipped** (the two `--long-context` cases);
* `logs/long_context.log` — **2 passed**, 262143-token prefill and decode at position 262143 for
  both layer kinds;
* `logs/watcher_run.log` — **11 passed** under `TT_METAL_WATCHER=10`, watcher log clean
  (`watcher/WATCHER_AUDIT.md`);
* `pcc_evidence.json` — 285 records, 264 of them PCC, **minimum 0.998030**, none below the bar;
  four full-context scale ratios, range 0.99577 to 0.99759, inside the ±2 % tolerance.

The per-measurement table and the delta against the functional stage are in
[`README.md`](README.md), generated from the two stages' evidence files. In short:
`linear_attention` moves by a few times 1e-5, and every bit of that traces to the deliberate
bfloat16 causal-conv FIR of §3.7; `full_attention` prefill is unchanged to within 1e-6 and its
full-context decode moves *up*.

`tests/test_fused_decoder_docs.py` is the document gate for this stage: it re-derives every perf
figure in these documents from the committed `tt-perf-report` CSVs, resolves every cited path,
re-checks `pcc_evidence.json` against its own records and the bar, re-runs the watcher grep, and
binds each quoted pass count to the log it is attributed to. `scripts/check_docs.py` (the
functional stage's equivalent) still passes too — its artifact corpus now spans both stages,
because `doc/context_contract.json` is a document both stages write to.

---

## 5. Performance — before and after

Both implementations profiled by the same script, `probes/run_perf.sh <kind> <phase> <impl>`, on
the same machine against the same build; `<impl>` is the only argument that differs between the
two halves of each row, and the `.provenance` file next to each CSV carries its run timestamp. Prefill is one warmed 2048-token pass; decode is traced — capture once, then
replay `execute_trace` 8x inside the signposted window — and the figure is the mean replay.
Device time is the sum of the `Device Time` column of the `tt-perf-report --csv` output, and
`perf_summary.json` re-derives all of it from the committed CSVs (as does
`tests/test_fused_decoder_docs.py`).

<!-- GENERATED:before_after -->
| layer kind | phase | device time before | device time after | speed-up | ops before | ops after |
|---|---|---|---|---|---|---|
| `linear_attention` | prefill, 2048 tokens | 150.971 ms | **26.144 ms** | **5.78x** | 801 | 68 |
| `linear_attention` | traced decode, 1 token | 3.034 ms | **2.395 ms** | **1.27x** | 92 | 67 |
| `full_attention` | prefill, 2048 tokens | 18.588 ms | **17.778 ms** | **1.05x** | 44 | 28 |
| `full_attention` | traced decode, 1 token | 2.270 ms | **2.061 ms** | **1.10x** | 50 | 44 |
<!-- END GENERATED:before_after -->

Every row is faster *and* smaller. The stage contract is the first of those — "fewer ops or
cleaner topology is not enough" — and `test_speedup_block_is_consistent` asserts it directly out
of `perf_summary.json`.

The `full_attention` prefill moves least because it was already a fused graph in stage 1:
`nlp_create_qkv_heads`, `chunked_scaled_dot_product_attention`, `paged_fill_cache` and
`nlp_concat_heads` were all in place, so what was left to fuse there was the partial RoPE, the
MLP's SiLU and the output gate's sigmoid, against a pass whose `matmul` and `sdpa` buckets
are almost all of it and which no graph rewrite touches.5 ms is five matmuls
and one SDPA call, both at the DRAM roofline.

## 6. What was assessed and not taken

Recorded here so "no remaining fusing" is a claim with evidence behind it, not an assertion.

| candidate | why not |
|---|---|
| `gated_delta_attn_seq` instead of `chunk_gated_delta_rule` | strictly more ops for the same result; the fused op is its successor |
| `chunk_gated_delta_rule` on the **decode** path as well as prefill | an exact op-contract blocker at the advertised batch: the phased program factory asserts `TT_FATAL(BH <= ncores, ...)` (`chunk_gdn_phased_program_factory.cpp:137`) with `BH = batch * num_v_heads`. At `max_batch` 32 that is 1536 against this device's 110 compute cores, so it fits only up to batch 2. It is also the wrong shape for decode: the op's flat path needs `T % 32 == 0` and decode's `T` is 1, so it would have to be called at chunk 32 with 31 masked positions per step |
| the decode group-reduction gated norm **at small batch** | measured at 1/4/8/16/32 (§3.17). The two forms cross between 16 and 32, so the layer picks by `max_batch`: the reshape form below 32, the group reduction at and above it. Both were measured, neither is estimated |
| `ttnn.conv1d` on 2048-channel groups of the causal conv | tried, exact blocker: `TT_FATAL @ ttnn/cpp/ttnn/operations/sliding_window/op_slicing/op_slicing.cpp:266: found_valid_config` - the sliding-window op finds no valid slicing for a 2051x1 depthwise conv over 2048 channels, which is why the in-tree reference falls back to an FIR past 2048 channels rather than grouping |
| width-sharding the `full_attention` decode Q/K head norms | tried at 4 and 8 cores, exact blocker: `TT_FATAL @ tt_metal/impl/tensor/spec/tensor_spec.cpp:161: !shard_grid_fit_error` - a `[1, batch, heads, 256]` tensor's shard grid does not fit. It is two small ops of a ~2 ms step in any case |
| `use_qk_l2norm=True` on the fused op | `TT_FATAL: use_qk_l2norm not yet supported`; the flat rank-3 path gives the in-kernel norm anyway |
| chunk size 64 on the fused op | measured PCC 0.903635 at 2048 tokens against 0.999994 at 32 (§3.1) |
| `rotary_embedding_hf` decode mode | needs a HEIGHT_SHARDED input and sharded per-user cos/sin; prefill mode broadcasts cos/sin over the batch axis (§3.2) |
| `paged_fused_update_cache` | requires disjoint shard core sets for the two inputs; K and V share cores (§3.5) |
| `group_attn_matmul` for the recurrence | batch dim must equal the number of users; here it is `batch * num_v_heads` (§3.6) |
| `hc_sum_reduce` / `repeat_and_interleave_eltwise_mul` for the gated norm | fixed 32-wide groups; this model's group is 128 |
| `output_head_major=True` on `chunk_gated_delta_rule` | measured about 2x the shipped token-major path, PCC 0.999994 between the two outputs (§3.13) |
| the RoPE head-channel permutation | measured worth well under 1 % of the prefill, and it widens the public `rot_mats` contract and permutes what the KV cache holds (§3.2) |
| `ttnn.addcmul` for the conv taps | a composite in this checkout: decomposes to multiply + multiply + add, i.e. one op worse than what it replaces (§3.14) |
| emitting bfloat16 from `in_proj_qkv` to remove the conv's float32 -> bfloat16 cast | it would also make the carried conv state bfloat16. That state is compared against HF's cache object at 0.999995 and stage 1 chose float32 for it deliberately; changing a *carried state's* dtype is a precision-policy decision the datatype-sweep stage owns, not a graph rewrite. The cast that remains is a real dtype conversion, not a no-op (§3.15 removed the ones that were) |
| `ttnn.conv1d` / depthwise `conv2d` for the causal conv | conv CBs overflow L1 at 10240 channels |
| the pre-aligned-window, all-ROW_MAJOR-arithmetic and all-TILE causal-conv formulations | measured slower than the shipped ROW_MAJOR-concat-and-shift form (§3.7) |
| `transpose_a=True` on the decode outer product | **taken**, not rejected: exact (PCC 1.000000) and one dispatch fewer (§3.6) |
| a matmul program config for the packed `a`/`b` projection | it is the one row this stage created that the profiler still calls dispatch-bound: at `32 x 5120 x 128` its N is 4 tiles, so output parallelism caps at 4 cores and §3.6's `core_grid` lever does not apply. The untried candidate is a K-split or DRAM-sharded program config, which is matmul *scheduling* rather than graph fusing and belongs to the optimized-decoder stage; recorded here so it is not lost |
| split gate/up MLP matmuls with `activation="silu"` | slowest of the three MLP forms at prefill (§3.8) |
| packing `in_proj_qkv`/`in_proj_z` into the `a`/`b` matmul | DRAM-bound already; would force a weight-dtype change (§3.9) |
| removing `repeat_interleave` from the decode GQA head expansion | it is a relayout **inside** a dedicated op — 2 `untilize_with_unpadding` + 2 `tilize_with_val_padding` in the committed decode report, about 1 % of the step. The two alternatives are a `[key_dim, value_dim]` 0/1-matrix matmul, whose weight alone is 25 MB against a percent of the step in headroom, or a recurrent-state layout in which v-head `h` maps to k-head `h % num_k_heads` instead of `h // v_per_k`, which would break the direct comparison of the on-device state against HF's cache object |
| widening the decode SDPA beyond one core per head | not a graph property: stage 1 pins `max_cores_per_head_batch = 1` to work around an upstream cross-core tree-reduction defect in `sdpa_decode`, documented with a model-free reproducer, and hands the kernel fix to the optimization stage. It is the whole `sdpa` bucket of the fused decode breakdown, about 5 % of the step, and this stage does not touch it |
| bfloat8/bfloat4 weights, lower math fidelity | precision policy, owned by the datatype-sweep stage; this stage changes the graph at a fixed precision policy, with the one exception in §3.7 where the arithmetic dtype *is* the graph property being measured |

---

## 7. Commands

```bash
cd /home/ttuser/dev/qwen/tt-metal
source models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/ttenv.sh

# fused functional suite
python -m pytest models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py -v -s

# full advertised context, 262143-token prompt + decode at 262143
python -m pytest models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py \
    -k test_full_advertised_context --long-context -v -s

# watcher-clean fused correctness run (the exact command doc/fused_decoder/watcher/WATCHER_AUDIT.md
# reports, and the one tests/test_fused_decoder_docs.py binds that audit to)
export ART=$REPO/models/autoports/qwen_qwen3_6_27b/doc/fused_decoder
export TT_METAL_LOGS_PATH=$ART/watcher TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=0 \
       TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1
python -m pytest $REPO/models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py \
    -k "test_traced_decode_pcc or (test_decode_pcc and 2049) or test_bfloat8_kv_cache or test_traced_decode_batched or test_alternate_page_block_size or test_repeated_runs_stable" \
    -v -s

# op-level probes (each is model-free and self-checking)
python models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/probes/probe_chunk_gdr.py
python models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/probes/probe_small_ops.py
python models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/probes/probe_causal_conv.py
python models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/probes/probe_decode_heads.py
python models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/probes/probe_decode_recurrence.py
python models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/probes/probe_mlp_variants.py

# before/after profiling, one (kind, phase, impl) triple at a time
doc/fused_decoder/probes/run_perf.sh <linear_attention|full_attention> <prefill|decode> <functional|fused>

# collect every recorded number
python -m models.autoports.qwen_qwen3_6_27b.scripts.collect_evidence \
    models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/logs/*.log
```

---

## 8. Stage review and checkpoint commits

`$stage-review` was run as a fresh independent subagent against the live worktree, with the goal
contract, the two stage skills and the artifact roots passed in. Round 1 returned
**more-work-needed** with six P2 findings; every one was taken as work, not argued with:

| finding | what was done |
|---|---|
| `WATCHER_AUDIT.md` described an earlier run than the committed watcher log | the audit is now **generated** from the committed artifacts by `probes/make_watcher_audit.py`, and `tests/test_fused_decoder_docs.py::test_watcher_audit_matches_its_artifacts` re-derives its line count, dump count, histogram and pass/deselect counts. The README's and the work log's watcher command are now the audit's, byte for byte, and that is asserted too |
| record count said 281 in the work log and the context contract, 283 in the evidence | corrected, and `::test_context_contract_matches_the_evidence` now pins the contract's whole acceptance block, its long-context results and its measured byte delta against `pcc_evidence.json` field by field |
| `chunk_gated_delta_rule`'s `output_head_major` was never assessed | built as a whole alternative path and measured: §3.13. Rejected at 2x the cost |
| the causal-conv FIR's trailing SiLU was never folded into the last tap's add | folded, in prefill and decode: §3.14 |
| two `bfloat16 -> bfloat16` no-op typecasts in the measured `full_attention` prefill | guarded on dtype: §3.15 |
| the RoPE head-channel permutation was rejected on test-comparison grounds, not on measurement | built as a probe and measured: §3.2. Rejected on the measurement plus the public-contract cost |

The reviewer's *Other Concerns* were taken the same way: the self-contradictory sigmoid figures
in the `_attn_epilogue` docstring, the "costs nothing" claim about a reshape that the report shows is a copy, the
77-78 %/58 % prose approximations and the `scripts/check_docs.py` corpus widening are all fixed
(the last by stripping later-stage blocks out of the shared `context_contract.json` when the
functional gate reads it, rather than by widening that gate's corpus). The volatile figures in
these documents now live between `<!-- GENERATED:... -->` markers and are written by
`probes/make_doc_tables.py` out of `perf_summary.json` and the probe logs, so a re-measurement
cannot leave them stale - which is what kept happening.

Round 2 re-reviewed the fixed stage and returned **more-work-needed** again, with five P2
findings and eight further concerns. Every one was taken:

| finding | what was done |
|---|---|
| the causal-conv rejection was contradicted by its own probe log - `rm_shift` was the fastest bfloat16 row, not `tile` | re-measured with 12 repeats and median/stdev instead of best-of-3: `rm_shift` really is faster, by several times the spread. **Adopted** (§3.7); the shipped FIR now untilizes once and shifts in ROW_MAJOR. `make_doc_tables.py` derives which cell it bolds from the measurement instead of hard-coding it |
| a `tilize` → `typecast` op-merge in the measured prefill was never assessed | `ttnn.tilize` takes the output dtype, so the two are now one op (§3.4). The rest of the typecast class is assessed too: the remaining ones are real float32 -> bfloat16 conversions, and removing the largest would mean making the *carried conv state* bfloat16, which §6 records as a datatype-sweep-stage decision, not a graph rewrite |
| the selected outer-product recurrence grid was the slowest measured one | re-measured with 30 repeats and stdev: the explicit grids are all within about a stdev of each other and all ~3x the default. The table now reports median (stdev), labels which grid is *selected* rather than pretending it is a unique minimum, and §3.6 says so (§3.6) |
| the README still quoted an earlier run's watcher line count | removed; `::test_readme_watcher_claims_match_the_audit` now binds any watcher number the README quotes to the generated audit |
| eleven stale perf figures survived outside the generated blocks, four of them in `tt/fused_decoder.py` and the tests | every transcribed figure in the implementation, the tests and the prose was either deleted or replaced by a pointer to the generated table that carries it. The gate now also scans `tt/fused_decoder.py` and the three test files, matches microsecond figures and bare integers as well as decimals, and binds every correlation-shaped figure to a committed artifact. The README's correctness and delta tables are generated too, from the two stages' evidence files |

Concerns from that round were taken the same way: the pad-alias test's attribution now names the
range that actually covers the fused conv's view hazard (`seq_len <= 64`, i.e. the `1` and `17`
cases), the inherited packed `conv_state` is documented as a template the fused decode does not
read, the `QWEN_GDN_PHASED` guard now matches the op's own `e[0] != '0'` test and says what it
cannot catch, the before/after pairing claim is stated as "same machine, same build" with the
`.provenance` timestamps rather than "same session", and the evidence script now gzips and
removes the raw watcher log so the committed `.gz` is the only copy the audit and the gate can
read.

Round 3 re-reviewed again and returned **more-work-needed** with one P1 and four P2 findings.
The P1 was a real graph defect all three rounds' probes had been blind to:

| finding | what was done |
|---|---|
| **P1** a `tilize` -> `untilize` round trip over the whole ~42 MB conv window survived in the measured prefill: `ttnn.concat` on TILE operands untilizes them, concatenates and re-tilizes, and the ROW_MAJOR tap loop then threw that tilize away. Every one of the four measured FIR formulations inherited the same TILE concat, so re-measuring them could not expose it | a fifth formulation that concatenates in ROW_MAJOR was built, measured over 12 repeats - faster by several times the spread, and bit-identical - and **shipped** (§3.7). `tests/test_fused_decoder.py::test_no_redundant_relayout_in_measured_prefill` was added to gate the class the goal contract names but nothing checked. Round 4 showed that python-level trap cannot see relayouts made *inside* `ttnn.concat`, so it is now backed by `test_fused_decoder_docs.py::test_no_layout_round_trip_in_the_measured_prefill`, which reads the committed device report instead |
| the work log and the probes README still said the all-TILE conv formulation was the fastest, three lines under the generated table that says otherwise | both rewritten to match the measurement |
| `tt/fused_decoder.py` quoted three pre-round-2 GB/s figures | deleted; the docstring points at the generated table |
| the README quoted a pre-round-2 python-boundary op count | the whole claim is a generated block now, read from `pcc_evidence.json`'s own record |
| `perf_summary.json` still claimed the before/after pair was measured "in the same session" | corrected in the generator, so it is written from the artifacts' actual provenance |

Its concerns were taken too: `transpose_a=True` folds the decode outer product's transpose into
its matmul (exact, one dispatch fewer - §3.12); the packed `a`/`b` matmul being dispatch-bound at
4 cores is recorded in §6 as matmul *scheduling* for the optimized-decoder stage rather than
graph fusing; the generated recurrence table now shows every measured grid rather than a
hand-picked subset; `_AB_STRIDE` is asserted against `num_v_heads` at load time; and the README's
two overstated gate descriptions ("`git diff` shows only one file", "pins the call counts") now
say what the artifacts actually support.

Round 4 returned **more-work-needed** with four P2 findings and eight concerns, and its first
two were about the *fix mechanism* rather than the graph:

| finding | what was done |
|---|---|
| the README's generated op-count block still held the generator's unfilled placeholder text: its markers were inline in a table cell, and the generator's regex needs them on their own lines, so the block was silently skipped forever | the block is multi-line now, and `::test_generated_blocks_are_current` re-runs every generator and requires the committed text to match byte for byte, and rejects any block still holding the generator's placeholder text. That single test is what retires this whole recurring class - it also caught the evidence script generating the tables *before* rebuilding the evidence they read, which is now ordered the other way |
| `test_no_redundant_relayout_in_measured_prefill` could not see the defect it was written for: it traps python-level calls, and `ttnn.concat` relayouts *inside* itself | replaced by `::test_no_layout_round_trip_in_the_measured_prefill`, which reads the committed `tt-perf-report` op sequence - the device-level truth - and fails on any `Tilize*` immediately followed by an `Untilize*`. It immediately found two more round trips in the state cut, which is now done in ROW_MAJOR end to end |
| §8's own round-3 row quoted two numbers from a superseded run, and "bit-identical" was asserted but never measured | re-quoted from the generated table, and `probe_causal_conv.py` now checks every variant against the first as well as against torch and prints the max absolute difference, so "the formulations agree" is a measurement |
| `chunk_gated_delta_rule` on the **decode** path had no row in §6 | recorded with its exact blocker: the phased program factory asserts `BH <= ncores` with `BH = batch * num_v_heads`, which is 1536 against 110 cores at `max_batch` 32 |

Its concerns closed the last three "rejected without a measurement" gaps by measuring them:
the decode group-reduction gated norm at five batch sizes (§3.17, and the layer now picks the
cheaper form by `max_batch` instead of committing to one), `ttnn.conv1d` on 2048-channel groups
(an exact `found_valid_config` blocker), and width-sharding the `full_attention` decode head norms
(an exact `shard_grid_fit_error` blocker). The "free reshape" wording, the off-by-one in the
conv-state comment, the two unfreed typecast intermediates, the capacity percentage and the probe
docstring's stale variant list are all corrected.

Round 5 returned **more-work-needed** with two P2 findings and five concerns, all of them
about evidence rather than the graph:

| finding | what was done |
|---|---|
| `_GATED_NORM_GROUP_BATCH = 32` was chosen from a threshold no committed artifact measured - the round-4 fix had added the *mechanism* for picking between the two gated-norm forms without a probe behind the number | `probes/probe_gated_norm_batch.py` was written and run, and its log is committed and generated into §3.17. It backs the threshold: the reshape form wins at batch 1 and the group form from batch 16 up, and they agree at PCC 0.999994 |
| `test_prose_perf_figures_match_the_summary` was vacuous: it accepted any figure appearing *as a substring anywhere* in the corpus, so a stale figure passed as long as some unrelated number contained its digits | rewritten around `_allowed_figures()`, which derives the allowed set by value from `perf_summary.json` (device time, gap, breakdown, top ops), the stage-1 summary, and every number the probe logs actually printed. Generated blocks are excluded from the scan and covered by `::test_generated_blocks_are_current` instead. It immediately found the ten remaining hand-transcribed figures, all now either generated or reworded to quote the table |

Its concerns were taken too: two skill patterns the earlier rounds had not assessed are now
**taken** - `dt_bias` folds into the packed `a`/`b` projection as `ttnn.linear(bias=...)` and the
gated norm's epsilon add carries `RSQRT` as its output activation, two dispatches fewer per
`linear_attention` layer per phase at unchanged PCC (§3.9, §3.17); §6's contradictory
conv-formulation row was corrected; the README now credits the device-level relayout gate rather
than the python-level one; and the three figures that no longer matched the regenerated perf
summary were replaced with the generated tables.

Checkpoint commits on `agentic-research/hous/qwen3.6-27b-v2` (local only; never pushed):

| SHA | what |
|---|---|
| `8c0f31b7421` | Qwen3.6-27B fused decoder: graph-fuse both layer kinds |
| `1c8d2c9ca18` | Qwen3.6-27B fused decoder: second-review fixes |
| `85ffa523178` | Qwen3.6-27B fused decoder: third-review fixes |
| `3cc755545c1` | Qwen3.6-27B fused decoder: fourth-review fixes |

Unrelated dirty state in the worktree - `.agents/notes/gdn.md`, two
`.agents/prompts/model_bringup_multigoal/*.txt` and `scripts/check_agent_prompt_lengths.py` -
predates this stage and was deliberately left out of the checkpoint.
