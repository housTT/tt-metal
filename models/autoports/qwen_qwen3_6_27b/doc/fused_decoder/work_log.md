# Qwen3.6-27B — fused decoder work log

Narrative for the fusing stage: what the profile said, what was tried, what landed, what was
measured and thrown away. The index of the landed rewrites is in [`README.md`](README.md); this
file is the reasoning and the raw numbers behind each one.

Everything ran on **device 2** (a Blackhole chip of the intact p300c board, compute grid 11x10),
1x1 mesh, from `/home/ttuser/dev/qwen/rundir` with `ttenv.sh` sourced — `ttenv_profiler.sh` for
the Tracy runs, never both at once, and never watcher alongside the profiler.

## 0. Starting point

`tt/functional_decoder.py` at commit `206df66` — correct, deliberately unoptimised: BF16, tile
layout, DRAM interleaved everywhere. Re-measured in this stage as the baseline
(`baseline/tracy/`):

| run | device time | ops in window |
|---|---|---|
| `linear_attention` prefill 2048 | 150.23 ms | 805 |
| `linear_attention` traced decode | 3032.2 µs/token | 96 |
| `full_attention` prefill 2048 | 18.60 ms | 44 |
| `full_attention` traced decode | 2425.2 µs/token | 50 |

Re-aggregating `baseline/tracy/*/*_perf_report.csv` by op code — step 2 of the graph-fusing
skill. The full op tables are in those CSVs; this is the shape of the problem:

```
linear_attention/prefill  150.2 ms  Matmul b={6144} 32x32x32       38.3 ms  25.5 %  recursive triangular inverse, 6 calls
                                    BinaryNg                       21.0 ms  13.9 %  195 elementwise
                                    ReshapeView                    12.4 ms   8.2 %  5 head splits
                                    Matmul b={48} 64x128x128        9.5 ms   6.3 %  per-chunk recurrence loop
                                    Matmul b={3072} 32x32x32        6.4 ms   4.3 %  triangular inverse, level 2
                                    Slice                           5.8 ms   3.8 %  278
linear_attention/decode   3032 µs   MLP gate_up 32x5120x34816        866 µs  28.6 %
                                    MLP down    32x17408x5120        433 µs  14.3 %
                                    in_proj_qkv 32x5120x10240        261 µs   8.6 %
                                    LayerNorm (3/iter)               209 µs   6.9 %
                                    BinaryNg (24/iter)               203 µs   6.7 %
full_attention/prefill    18.6 ms   5 matmuls at 80-83 % of roofline 13.5 ms  73 %
                                    partial-RoPE slice/neg/concat     636 µs   3.4 %  (ops 81-100)
                                    MLP epilogue slice/silu/multiply 1724 µs   9.3 %  (ops 113-116)
full_attention/decode     2425 µs   MLP gate_up + down              1300 µs  54 %
                                    SdpaDecode                       258 µs  10.7 %
                                    LayerNorm (4/iter)               216 µs   8.9 %
                                    wqkv + wgate + o_proj            507 µs  21 %
```

Two things shaped the whole stage:

1. `linear_attention` prefill is **not** FLOP-bound. A quarter of it is a triangular inverse made
   of thousands of one-tile matmuls, and another fifth is layout churn.
2. `full_attention` is already at the DRAM roofline in both phases; its matmuls are 73 % of
   prefill and 75 % of decode. The only headroom is the op count around them.

## 1. Exploring the op library first (skill step 1)

Everything the tt-metal tree offers for these subgraphs, before writing any code:

* `ttnn.swiglu` / `glu` / `geglu` / `reglu` — the MLP.
* `ttnn.experimental.rotary_embedding_hf` — HF-style rotate-half RoPE, prefill and decode.
  Also `rotary_embedding`, `rotary_embedding_llama`, `rotary_embedding_llama_fused_qk`.
* `ttnn.transformer.gated_delta_attn_seq` — a **dedicated chunked gated-delta-rule kernel**,
  exactly this model's linear-attention mixer.
* `ttnn.experimental.nlp_create_qkv_heads{,_decode}`, `nlp_concat_heads{,_decode}`,
  `create_qkv_heads_from_separate_tensors`.
* `ttnn.experimental.paged_fused_update_cache` — K and V cache update in one dispatch.
* `ttnn.conv1d` (depthwise via `groups`), `ttnn.experimental.ssm.*` (Mamba-specific),
  `ttnn.addcmul`, `ttnn.rms_norm` with a sharded program config, and the binary ops'
  `input_tensor_a_activations` / `input_tensor_b_activations`.

Existing implementations read for idiom: `models/demos/blackhole/qwen36/tt/` (a Qwen3.5/3.6 port
in the installed tree) and `models/experimental/gated_attention_gated_deltanet/tt/`, which is
what drives `gated_delta_attn_seq`; `models/tt_transformers/tt/{attention,rope}.py` for the
decode sharding idioms.

## 2. F2 — partial RoPE as one op, via a host-side channel permutation

`ttnn.experimental.rotary_embedding_hf` applies rotate-half over the **whole** head. Qwen3.5
rotates only the leading `rotary_dim` = 64 of `head_dim` = 256 channels and pairs channel `j`
with `j + 32`; the op pairs `j` with `j + 128`. So the op cannot express partial RoPE — unless
the channels move.

Permute the head channels to

```
new[0:32]     = old[0:32]        # rotary, first half
new[32:128]   = old[64:160]      # pass-through
new[128:160]  = old[32:64]       # rotary, second half  -> now at the rotate-half midpoint
new[160:256]  = old[160:256]     # pass-through
```

and give the pass-through channels `cos = 1`, `sin = 0`. Then a full-width rotate-half is exactly
Qwen3.5's partial RoPE. The permutation is applied host-side, at load time, to the rows of the
`q`/`k` projection weights and to the `q_norm`/`k_norm` weights, so nothing moves at runtime.

Why it is safe: RMSNorm over the head is permutation-invariant in its reduction and per-channel
in its weight, so permuting both input and weight permutes the output; and `q·k` is invariant
under a permutation applied to **both**. `v`, the output gate and `o_proj` never see it.
`cos = 1` and `sin = 0` are exact in bfloat16, so the pass-through channels are bit-exact.

Verified model-free first (`probes/probe_fused_ops.py`, `logs/probe_fused_ops.log`):

```
prefill permuted rotary_embedding_hf: pcc=0.999998
```

Decode mode additionally requires the input **and** `cos`/`sin` height-sharded one user per core;
the first attempt with a `(1, head_dim)` shard shape failed
(`tensor_layout.cpp:162: !shard_align_error.has_value()`) — the shard height must be a whole
tile, i.e. `(32, head_dim)`, which is what `models/tt_transformers/tt/rope.py` uses. With that
fixed the decode op is exact at every batch the layer supports
(`probes/probe_candidates.py ropebatch`, `logs/probe_ropebatch.log`):

```
batch=1  overall=0.999998  min_user=0.999998  bad=[]
batch=4  overall=0.999998  min_user=0.999998  bad=[]
batch=8  overall=0.999998  min_user=0.999997  bad=[]
batch=16 overall=0.999998  min_user=0.999998  bad=[]
batch=32 overall=0.999998  min_user=0.999997  bad=[]
```

Payoff, from the op-by-op reports rather than the aggregate: baseline ops 81–100 (the twenty
RoPE ops) total **636 µs**; the fused replacement is two `RotaryEmbeddingHfDeviceOperation`s
totalling **367 µs**. **−269 µs and −18 ops** of an 18.6 ms prefill, plus 20 ops of a decode
step. Not a large time win — the honest accounting matters, because this is the riskiest change
in the stage: it makes `rot_mats` `head_dim`-wide and the paged K cache permuted, which every
later stage has to honour. It is documented in the module docstring, in `README.md`, in
`context_contract.json`, and asserted at the API boundary.

## 3. F6/F7 — the triangular inverse: L1, not fewer ops

`_unit_tri_inverse` was 47.9 ms of the 150 ms baseline (32 %), and the profile said its batched
`32x32x32` matmuls ran on **one core**. The obvious reading — "the op does not parallelise over
batch" — is wrong. Timing the same matmul out of different memory
(`probes/probe_fused_ops.py`, `probes/probe_fused_ops2.py`; `logs/probe_fused_ops.log`,
`logs/probe_fused_ops2.log`):

```
matmul [768, 1, 32, 32]  DRAM  0.813 ms/call   -> 1.06 µs per batch element
matmul [768, 32, 32]     DRAM  0.806 ms/call
matmul [6144, 1, 32, 32] DRAM  6.408 ms/call   -> 1.04 µs per batch element (perfectly linear)
matmul [3072, 32, 32]    DRAM  3.209 ms/call
matmul [768, 32, 32]     L1    0.033 ms/call   -> 0.043 µs per batch element  (24x faster)
```

The cost is per-batch-element DRAM latency, not compute and not core count. So the fix is
residency, not restructuring.

L1 is finite, though. A first attempt that put the *whole* gated-delta-rule working set in L1
died with

```
Out of Memory: Not enough space to allocate 25165824 B L1 buffer across 110 banks,
where each bank needs to store 229376 B, but bank size is 1461504 B
```

110 banks x 1.43 MB = 157 MB total. The landed version therefore chooses per region:

* `_unit_tri_inverse` picks a memory config **per recursion level** from that level's own tensor
  size (`_tri_mem`, budget 5 blocks against `L1_BUDGET_BYTES` = 96 MB), so the base case —
  thousands of `32x32x32` matmuls — is in L1 while the wider top level, whose few matmuls are
  large, stays in DRAM;
* the per-chunk recurrence loop slices one chunk at a time (~1.5 MB each) and runs in L1;
* the full-length tensors (decay mask, `kk`, `k_beta`, the projections) stay in DRAM, where one
  big streaming pass costs nothing extra.

`TRI_INV_BASE` went 16 → 32 at the same time (**F7**): the recursion stores its blocks in 32x32
tiles, so base 16 wastes three quarters of every tile, and 32 is one level shallower. The
functional stage had already measured identical layer PCC at bases 8/16/32 and recurrent-state
PCC 0.999986 (32) vs 0.999991 (16); this stage's end-to-end PCC confirms it (`linear_attention`
prefill 0.999940, unchanged to six decimals).

Result: the `b={6144}` and `b={3072}` `32x32x32` matmuls, 44.7 ms of the baseline's top ops, do
not appear in the fused profile's top ops at all.

## 4. F15 — the decode norms were running on one core

The first fused `full_attention` decode profile still put `LayerNormDeviceOperation` at 9 % with
four calls per step. Reading the per-op rows rather than the aggregate showed why:

```
168  LayerNormDeviceOperation   102 µs   Cores 1     <- input_layernorm   [1,1,32,5120]
177  LayerNormDeviceOperation     6 µs   Cores 1     <- q_norm            [1,B,24,256]
178  LayerNormDeviceOperation     6 µs   Cores 1     <- k_norm            [1,B,4,256]
194  LayerNormDeviceOperation   102 µs   Cores 1     <- post_attention_layernorm
```

The interleaved layernorm kernel parallelises over tile **rows**, and a decode activation has
exactly one row of tiles, so both full-width norms ran single-core: 204 µs of a 2425 µs step, in
both layer kinds. Width-sharding fixes it. Sweeping core grids and `subblock_w` on the real shape
(`probes/probe_candidates.py norm`, `logs/probe_norm.log`):

```
interleaved (baseline)                 103.8 µs/call, 1 core
sharded 5x2  block_w=16 subblock_w=4    24.6 µs/call   max|err| 3.1e-2 (one bf16 ULP here)
sharded 5x2  block_w=16 subblock_w=2    23.7 µs/call
sharded 8x4  block_w=5  subblock_w=1    30.8 µs/call
sharded 8x5  block_w=4  subblock_w=1    31.9 µs/call
sharded 10x4 block_w=4  subblock_w=1    32.0 µs/call
grid 8x8                                skipped: 160 tiles not divisible by 64
```

`_decode_norm_config` picks the first grid in `_DECODE_NORM_GRIDS` whose core count divides the
hidden size in tiles (5x2 for 5120) and falls back to the interleaved kernel when none does. The
two resharding dispatches it adds cost ~1 µs each.

## 5. The composite trap — and the merges that actually work

This is the part the stage got wrong first and had to redo, so it is written out in full.

**`ttnn.swiglu` is not a fusion.** It is a composite: `unary_composite_op.cpp:293` does
`split_tensor_for_glu` (two slices) → `swish` → `multiply` — exactly the sequence it appears to
replace. The profiler confirmed it: the functional MLP epilogue is
`Slice 351 µs, Slice 363 µs, Unary 555 µs, BinaryNg 455 µs` and the `swiglu` version was the same
four ops at the same cost. In decode it was **worse**, because `swiglu` reports the tile-padded
height as its logical height, so a batch < 32 step came back 32 rows tall and needed an extra
`slice` to trim.

**`ttnn.linear(activation=...)` is not a fusion either**, unless a program config or `core_grid`
is supplied: `matmul.cpp:295` applies `user_fused_activation` as a separate `ttnn::unary_chain`
dispatch. The profiler showed the same 192 µs `UnaryDeviceOperation` after the gate matmul,
merely moved earlier in the stream.

**`ttnn.matmul(transpose_a=/transpose_b=True)` is not a fusion either**: ttnn lowers the flag to
a separate `TransposeDeviceOperation`, 75 of them before and 75 after. It is kept because it
reads better, but it is not counted as a win.

What *does* fuse is the skill's own listed pattern — an activation as an **input argument of an
eltwise binary** the graph already contains. `ttnn.graph` capture makes the difference visible:

```
ttnn.swiglu(x)                                          -> Slice, Slice, Unary, BinaryNg  (4 device ops)
ttnn.multiply(a, b, input_tensor_b_activations=[SILU])  -> BinaryNg                       (1 device op)
```

So the landed merges are:

* **F1** SwiGLU MLP: `matmul → slice ×2 → multiply(SiLU on b)`. `full_attention` prefill ops
  113–116 (1724 µs, 4 ops) become ops 74–76 (1253 µs, 3 ops): **−471 µs**.
* **F9** attention output gate: the sigmoid becomes an input activation of the multiply that was
  already there. Baseline `Unary 191 µs + BinaryNg 175 µs` becomes `BinaryNg 200 µs`: **−166 µs**.
* **F10** gated-delta-net `z`: same shape of change, at the gated-norm multiply.
* **F11** `dt_bias` as `ttnn.linear(bias=...)` — this one *is* a genuine matmul feature; the
  profiler shows no add after the `32 x 5120 x 128` matmul.
* **F13** decay mask: a constant strictly-upper `-1e9` added before a single `exp` replaces
  `tril → exp → tril`.
* **F14** `attn0`: the `neg` and the `tril(-1)` fold into a multiply by a precomputed
  `-tril(ones, -1)` mask.

Those three prefill savings — 269 (F2) + 471 (F1) + 166 (F9) = 906 µs — account for the whole
`full_attention` prefill delta of 897 µs, which is the cross-check that the accounting is real.

**F16** is the same idea one level down: the causal conv is a 4-tap FIR, four `slice → multiply`
plus three `add`, and `addcmul(acc, tap, w)` merges each multiply-add. On the real shape,
`[1, 1, 2051, 10240]` fp32 (`logs/probe_conv.log`):

```
multiply + add : 15.38 ms   max|err| 3.0e-7
addcmul        : 13.73 ms   max|err| 2.4e-7
```

**F3** replaces the hand-rolled L2 norm with one `ttnn.rms_norm`:
`rms_norm(x, eps') = x·√D / √(Σx² + D·eps')`, so with `eps' = 1e-6/D` it *is* HF's
`l2norm(x, eps=1e-6)` up to the constant `√D`, which folds into the norm weight together with
the `1/√head_k_dim` query scale. Five ops become one, twice per pass.

## 6. F8/F18/F19/F20 — the smaller structural rewrites

**F8, the decode conv state.** The window used to be one `[1, batch, K, conv_dim]` tile-layout
buffer, and each tap sliced a row out of it at a non-tile-aligned offset — which TTNN implements
as untilize → slice → retilize, visible in the profile as
`UntilizeWithUnpadding 5 µs → Slice 2 µs → TilizeWithValPadding 10 µs` per tap per step. Holding
the `K − 1` history rows as **separate `[1, 1, batch, conv_dim]` buffers** removes all of it: a
tap reads its buffer directly, and the write-back is `K − 1` `ttnn.copy`s that keep the buffers'
addresses stable for trace replay. Putting the batch on the tile-height axis is what keeps the
memory sane — three buffers, 3.9 MB total at batch 32, where a per-tap `[1, batch, 1, W]` layout
would have been 126 MB. The whole decode path then stays in the `[1, 1, batch, ...]` layout, so
the `x_rows` reshape the unfused decoder needed is gone too.

**F18, the small `b|a` projection.** F5's fused `b|a` matmul is `32 x 5120 x 128` at decode —
only four output tiles, so the default program picks four cores and `tt-perf-report` flags it
`SLOW` at 9.5 % of DRAM roofline. Sweeping (`logs/probe_review_followups.log`):

```
N=128  default 63.1 µs   grid 8x8 62.2 µs   grid 4x8 35.3 µs
N=256  default 65.4 µs   grid 8x8 63.1 µs   grid 4x8 35.6 µs
N=512  default 75.1 µs   grid 8x8 75.0 µs   grid 4x8 41.7 µs
N=1024 default 79.4 µs   grid 8x8 79.7 µs   grid 4x8 70.5 µs
N=128 with bf16 weights instead of fp32: 65.3 µs
```

An explicit 4x8 core grid at the natural width wins; padding N wider does not help and bf16
weights do not either. Prefill has 2048 rows and does not want the restriction, so the grid is
applied only when the activation is one tile row tall.

**F19/F20, the output path.** The per-chunk recurrence outputs are `[1, nv, chunk, Dv]` and chunk
`i` holds tokens `[i·chunk, (i+1)·chunk)`, so concatenating along the **sequence** axis lands
directly in `[1, nv, L, Dv]`; the unfused code concatenated along the chunk axis and then needed
a permute and a 2.3 ms reshape to get there. Similarly, the gated norm's output can be flattened
back to `[1, 1, L, value_dim]` and multiplied against `z` where it already is — reshaping `z`
into head shape instead cost 3.1 ms, while this direction is a view. Together **−5.2 ms** of
`linear_attention` prefill.

## 7. F17 — the head split, and why the obvious fix was the wrong one

After F6 the largest remaining non-matmul block in `linear_attention` prefill was
`ReshapeViewDeviceOperation`, 12.3 ms / 15.3 %: three of the five calls are the `q`/`k`/`v` head
split, `[1, 1, L, W] → [1, L, H, D]`, a last-dim change that TTNN implements as untilize +
retilize.

The dedicated op for that pattern is `ttnn.experimental.nlp_create_qkv_heads`, and it does not
obviously apply: it requires K and V to have the same head count, while this mixer has 16 key
heads and 48 value heads (`create_qkv_heads_from_separate_tensors` has the same constraint).

Two candidates were built and measured (`logs/probe_review_followups.log`,
`logs/probe_headsplit_narrow.log`):

**(a) Widen the projection.** Fold the 3x key-head repeat into `in_proj_qkv` so all three tensors
have 48 heads, then one `nlp_create_qkv_heads` call:

```
prefill L=2048  : current 13.68 ms  ->  widened 5.15 ms
decode  32 rows : current  0.55 ms  ->  widened 0.74 ms
```

Good for prefill, but the projection goes 10240 → 18432 columns, which is **+189 MB of weights
per layer** and **+205 µs/token** in decode (that matmul is weight-bandwidth bound: 105 → 189 MB
at ~410 GB/s). Decode is the stage's gate, so this is a regression where it counts, on top of a
memory cost that would be 12 GB over 64 layers. Rejected.

**(b) Two overlapping calls on the *narrow* output.** 48 value heads are three consecutive groups
of 16, so:

```
call A over columns [0, 6144)      -> q,     k,       v heads 0-15
call B over columns [4096, 10240)  -> (v0),  v 16-31, v 32-47
```

then `concat` the three v groups and `repeat_interleave` q and k by 3, as before. The duplicated
`v0` output of call B is the only waste, one third of one of the two calls. Measured:

```
L=2048  current 11.25 ms  ->  two-call nlp_create_qkv_heads 3.14 ms
q, k, v all bit-identical to the reshape path (maxdiff = 0)
```

**−8.1 ms, no extra weights, no decode cost.** Landed as F17. The `[nv, nc, chunk, D]` reshape and
permute that follow are leading-dimension operations and stay.

## 8. What is left in `linear_attention` prefill, and why

After everything, 67.4 ms:

```
BinaryNg                            9.1 ms  13.6 %  180 ops - the delta rule's own elementwise work
Ternary (addcmul, causal conv)      6.3 ms   9.3 %    3 ops
Matmul 2048x5120x34816 (MLP)        6.1 ms   9.1 %  at roofline
Matmul b={1536} 64x64x128           5.7 ms   8.5 %  inv @ v_beta, inv @ k_beta_decay
Slice                               4.8 ms   7.2 %  274 ops - the per-chunk loop's eight slices x 32
Matmul b={1536} 64x128x128          4.6 ms   6.8 %  kk
Matmul 2048x17408x5120 (MLP down)   3.3 ms   5.0 %  at roofline
Matmul b={1536} 32x32x32            3.2 ms   4.7 %  triangular inverse, top level (DRAM by budget)
```

The remaining bookkeeping is gone: `ReshapeView` has dropped out of the top ops entirely, the
conv taps no longer untilize, and the output path no longer permutes. What is left is the
arithmetic of the gated delta rule itself — bandwidth on 25–50 MB float32 tensors — plus the
32-iteration loop that the only dedicated alternative (`gated_delta_attn_seq`, §9.1) is both
slower and less accurate than.

The one identified but untaken lever is the triangular inverse's **top** level, still in DRAM
because `_tri_mem` estimates its live set at 126 MB against a 96 MB budget. Raising the budget
risks the L1 exhaustion quoted in §3, which is a hard failure rather than a slow path, so it was
left alone; it is 4.7 % of the layer.

## 9. Rejected, with the measurement

### 9.1 `ttnn.transformer.gated_delta_attn_seq` — the dedicated kernel is slower *and* less accurate

The most promising candidate on paper: a hand-written kernel for exactly this model's chunked
gated delta rule, with a validated Python wrapper in the installed tree
(`models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_seq.py`). It would
replace the 32-iteration recurrence loop, the triangular inverse and most of the elementwise
work.

Probed against HF's `torch_chunk_gated_delta_rule` at the real head geometry (48 v-heads,
`head_k_dim` = `head_v_dim` = 128 — the only geometry the kernel accepts: it hard-requires
`chunk_size == key_dim == val_dim == 128`, `gated_delta_attn_seq_device_operation.cpp:48`):

```
seq=256   out pcc=0.984615   state pcc=0.987376
seq=2048  out pcc=0.987237   state pcc=0.987776
seq=2048  whole chunked delta rule wall = 167.9 ms
```

Both halves fail. **Accuracy**: 0.987 against a 0.995 bar, because the wrapper's diagonal-block
inverse is a Neumann series plus two Newton–Schulz steps at HiFi2, where the functional decoder's
recursive block inversion reaches ~1e-3 absolute error. **Speed**: 167.9 ms for the delta rule
*alone* versus 150 ms for the entire baseline layer and 67.4 ms for the entire fused one.
Rejected on both counts; making the kernel's `L_inv` as accurate as the recursion would only make
it slower.

### 9.2 Shared-LHS `wqkv` + `wgate`

The classic ≥2-matmuls-sharing-an-LHS rewrite (`logs/probe_sharedlhs.log`):

```
rows=32    two matmuls  364.2 µs    one 5120x14336 matmul + 2 slices  373.3 µs
rows=2048  two matmuls 3510.0 µs    one 5120x14336 matmul + 2 slices 5182.1 µs
```

Slower in both phases. The two matmuls already run at 82–83 % of DRAM roofline, so merging them
buys one dispatch and pays for two slices and a worse-shaped matmul.

### 9.3 `ttnn.experimental.paged_fused_update_cache`

Would merge the two decode `paged_update_cache` dispatches into one. It requires its two inputs
on **disjoint** core ranges:

```
TT_FATAL: input_tensor1 ({[0-0 - 0-0]}) and input_tensor2 ({[0-0 - 0-0]}) must not overlap
          (paged_fused_update_cache_device_operation.cpp:227)
```

`nlp_create_qkv_heads_decode` puts K and V on the same batch cores, so using it needs a reshard of
V onto a second core range — exactly the dispatch the fusion would save.

### 9.4 Keeping `q` sharded through the decode SDPA

`rotary_embedding_hf` decode emits a height-sharded `q`, and feeding it straight to
`paged_scaled_dot_product_attention_decode` would drop a reshard. It is **wrong**, and it is the
one bug this stage produced that batch-1 PCC would never have caught. Per-user PCC at batch 32:

```
users 0-7   0.9995 0.9995 0.9996 0.9995 0.9995 0.9995 0.9994 0.9994
users 8-31  0.0116 0.0185 0.0008 0.0384 0.0206 0.0364 0.0493 ... -0.0007
```

The boundary at 8 is the width of the `CoreCoord(8, 8)` core range the shard was built on, while
this part's compute grid is **11x10** — the decode SDPA kernel derives each user's core from the
real grid width, so it reads users 8+ off the wrong core. Rebuilding the shard on the device grid
fixes SDPA but breaks `nlp_concat_heads_decode` (`RuntimeError: bad optional access`), and asking
the rotary op for an interleaved output fails in `circular_buffer_config.cpp:222`. The landed code
reshards `q` to DRAM before the SDPA, as the functional decoder did. `test_batched_users[32-*]`
caught this, which is why the fused suite re-runs the whole functional suite rather than a subset.
Bisect log: `logs/bisect_sharded_q_decode_batch32.log`.

### 9.5 A sharded SDPA-decode output

Would feed `nlp_concat_heads_decode` directly. Rejected by the op:
`TT_FATAL: Sharded output not supported for GQA (sdpa_decode_device_operation.cpp:405)`.

### 9.6 `ttnn.conv1d` for the causal conv

The depthwise `groups = conv_dim` conv1d is the dedicated op for the 4-tap FIR. The auto slicing
config search fails at this width:

```
op_slicing.cpp:266: found_valid_config   ("DRAM Auto slice could not find valid slice config")
```

That is an auto-config failure, not an op contract, so it was retried with an explicit
`Conv2dSliceConfig(slice_type=Conv2dDRAMSliceWidth, num_slices=N)` for N in 2, 4, 8, 16, 32
(`logs/probe_review_followups.log`). The config search then succeeds and every one of them
exhausts the allocator instead:

```
num_slices=2/4/8/16/32: bank_manager.cpp:462
```

1 batch, length 2051, 10240 channels, kernel 4, groups 10240 does not fit this op at any slicing
this build offers. Rejected with the adapted retry on record.

### 9.7 The composites

`ttnn.swiglu`, `ttnn.linear(activation=…)` and `ttnn.matmul(transpose_a/transpose_b=True)` — all
three look like fusions and none of them is one on this build. See §5.

## 10. Measured but deliberately not taken: a bfloat16 causal conv

The causal conv is 13.7 ms of the fused `linear_attention` prefill's 67.4 ms — the largest
remaining single block. Almost all of it is bandwidth on a float32 `[1, 1, 2051, 10240]` tensor
(84 MB per tap). In bfloat16 (`logs/probe_conv.log`):

```
FIR fp32 multiply+add: 15.38 ms   max|err| 3.0e-7
FIR fp32 addcmul     : 13.73 ms   max|err| 2.4e-7     <- landed
FIR bf16 multiply+add:  4.68 ms   max|err| 1.7e-2
FIR bf16 addcmul     :  4.11 ms   max|err| 1.8e-2
```

A further ~9.6 ms, about 14 % of the layer. It is **not** taken here because it is a precision
trade, not a graph rewrite: `mixed_qkv` is float32 by a deliberate decision of the functional
stage ("the recurrence and the softplus/exp gating are numerically sensitive"), the conv state is
compared against HF's cache at 0.999995, and changing compute dtype is the optimization stage's
job. The numbers are recorded here so that stage does not have to re-derive them.

## 11. Verification and evidence

Commands, all from `/home/ttuser/dev/qwen/rundir` with `ttenv.sh` sourced
(`ART=$REPO/models/autoports/qwen_qwen3_6_27b/doc/fused_decoder`):

```bash
# full fused suite - 62 passed, 2 skipped (the long-context pair), 470.92 s
python -m pytest $REPO/models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py -v -s
#   -> logs/suite_fused_final.log

# functional regression - 55 passed, 2 skipped, unchanged from the previous stage
python -m pytest $REPO/models/autoports/qwen_qwen3_6_27b/tests/test_functional_decoder.py -q
#   -> logs/suite_functional_regression.log

# full advertised context, 262143 tokens
python -m pytest $REPO/models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py \
    -k test_full_advertised_context --long-context -v -s
#   -> logs/long_context.log   (1 passed, 1 failed - the inherited SDPA-decode gap)

# watcher, separate run, profiler off
export TT_METAL_LOGS_PATH=$ART/watcher TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=0 \
       TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1
python -m pytest $REPO/models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py \
  -k "test_traced_decode_pcc or (test_decode_pcc and 2049) or test_bfloat8_kv_cache \
      or test_fused_graph_is_smaller" -v -s
#   -> logs/watcher_run.log, watcher/WATCHER_AUDIT.md  (7 passed, log clean)

# perf, one at a time, profiler build
cd $REPO/models/autoports/qwen_qwen3_6_27b/doc/fused_decoder
./probes/run_perf.sh <kind> <phase> fused
./probes/run_perf.sh <kind> <phase> functional "$PWD/baseline"

# evidence
python -m models.autoports.qwen_qwen3_6_27b.scripts.collect_evidence $ART/logs/*.log \
    --out $ART/pcc_evidence.json
#   -> 282 records, 276 numeric, exactly one under the bar (the inherited gap)
```

Record-by-record comparison against the functional stage's `pcc_evidence.json`: **258 shared
numeric measurements, none worse by more than 1e-4**, mean change +7.5e-6, worst −9.2e-5, best
+9.8e-4. Per-class minima are in [`README.md`](README.md).

Device op counts from `ttnn.graph` capture, asserted by `test_fused_graph_is_smaller` (these cover
the whole call including the input upload, so they are larger than the signposted perf window's
counts): `linear_attention` prefill 996 → 916, decode 98 → 70; `full_attention` prefill 111 → 71,
decode 56 → 42.

## 12. What the stage review changed

The first pass of this stage was reviewed and returned `more-work-needed`. The findings and what
was done about each:

| finding | action |
|---|---|
| `ttnn.swiglu` is a composite; F1 was not a fusion and was a net regression in decode | replaced with `multiply(..., input_tensor_b_activations=[SILU])`, a real merge; §5 rewritten from the profiler rows |
| `ttnn.linear(activation=…)` does not fuse; the skill's `input_*_activations` pattern was never assessed | assessed and landed for F9/F10; F11's bias verified as a genuine matmul feature; F12 already recorded as not-a-fusion |
| the head-split rejection was arithmetic only, and compared a prefill cost against a decode benefit | both candidates built and measured; the two-call `nlp_create_qkv_heads` variant landed as F17 (−8.1 ms, bit-identical) |
| ~5 ms of untilize/tilize left in the measured path, and F8's docstring did not match the code | F8 implemented as documented (per-tap row buffers); the conv-tap untilize/retilize is gone |
| `ttnn.conv1d` rejected on a single auto-slicing failure | retried with explicit slicing at five slice counts; every one exhausts the allocator (§9.6) |
| a `SLOW`, 4-core, 9.5 %-DRAM matmul introduced by F5 was unmentioned | swept and fixed with an explicit 4x8 core grid (F18), 63 → 35 µs |
| the RoPE saving was stated as 2.9 ms and the shared-record count as 262 | corrected to 269 µs and 258, from the CSVs and the two evidence files |
| a Python-level op spy cannot prove a fusion | replaced with `ttnn.graph` device-op counting (`test_fused_graph_is_smaller`) |

Net effect of the second pass, on top of the first: `linear_attention` prefill 78.7 → 67.4 ms,
its decode 2395 → 2255 µs/token; `full_attention` prefill 18.28 → 17.70 ms, its decode
2214 → 2201 µs/token.

## 13. Hardware notes

`tt-smi -ls --local` still hangs on this host (chip 0's ARC is wedged, as the functional stage
recorded); it is not a fault of this stage and does not block it, because the stage only ever
opens device 2. The 1x1 mesh smoke test passes:

```bash
python - <<'PY'
import ttnn
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
ttnn.close_mesh_device(mesh)
print("MESH_SMOKE_OK")
PY
```

No reset, no hang and no `tt-triage` capture was needed at any point: every failure hit was a
deterministic op-validation error or a wrong-answer bug, all reproduced and fixed from ordinary
logs. One second-order effect is worth recording: when a `linear_attention` test failed
mid-decode it leaked its L1 intermediates, and the *next* test then failed with
`Statically allocated circular buffers ... clash with L1 buffers`. That cascade was collateral
damage, not an independent bug — fixing the first failure cleared all of them.
