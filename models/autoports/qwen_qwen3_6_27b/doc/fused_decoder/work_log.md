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
| `ttnn.experimental.rotate_half` | decode partial RoPE's slice/slice/neg/concat | taken in §3.2, then **reverted on measurement** in §3.22: single-core by construction |
| `ttnn.addcmul` | decode recurrent-state update's multiply + add | **taken** (§3.21) |
| `ttnn.experimental.rotary_embedding` / `rotary_embedding_llama` / `rotary_embedding_llama_fused_qk` | ditto | rejected: rotate-half over the **whole** head_dim, and this model's rotary factor is 0.25, so they cannot express the partial rotation without permuting head channels (§3.2) |
| `ttnn.experimental.paged_fused_update_cache` | the two decode `paged_update_cache` calls | rejected with an exact op-contract blocker (§3.5) |
| `ttnn.experimental.group_attn_matmul` | the decode recurrent-state read | rejected with a quantified *resource* blocker, not a contract one: it overflows L1 (§3.6) (§3.6) |
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
| `matmul` | 14.396 ms | 1.990 ms | 13.468 ms | 1.812 ms |
| `batched_matmul` (the spelled-out delta rule / recurrence) | 79.024 ms | 0.259 ms | — | — |
| `sdpa` | — | — | 1.281 ms | 0.104 ms |
| `layout` (tilize/untilize/reshape/permute/concat/slice/shard) | 33.401 ms | 0.289 ms | 1.138 ms | 0.038 ms |
| `elementwise` | 23.541 ms | 0.290 ms | 1.799 ms | 0.053 ms |
| `norm` | 0.605 ms | 0.213 ms | 0.538 ms | 0.219 ms |
| `heads_and_cache` | — | — | 0.312 ms | 0.045 ms |
| **total** | **150.966 ms** | **3.040 ms** | **18.536 ms** | **2.271 ms** |
| ops in one pass | 801 | 92 | 44 | 50 |
| op-to-op gap | 8.115 ms | 0.359 ms | 0.023 ms | 0.046 ms |
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
| flat rank-3 `[1, T, H*D]` | 32 | 2048 | 0.999994 | 0.999994 | 3.79 ms |
| split rank-4 `[1, T, H, D]` | 64 | 64 | 0.999992 | 0.999992 | 0.55 ms |
| split rank-4 `[1, T, H, D]` | 64 | 2048 | 0.903635 | 0.996086 | 6.97 ms |
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
serve per-user positions. Rejected for decode with that exact contract blocker. The rotate-half
itself was replaced by `ttnn.experimental.rotate_half` here, which took the decode RoPE from 10
ops to 7 — and §3.22 later measured that substitution and **reverted** it, because the dedicated
op is single-core by construction and the op count was the wrong thing to have optimised.

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
| q, 24 heads | 0.444 ms | 0.365 ms (shipped) |
| k, 4 heads | 0.120 ms | 0.112 ms (shipped) |

so the whole permutation is worth **87 us of a 17.807 ms prefill, 0.5 %**.
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
| ms (stdev) | 0.038 (0.009) | 0.039 (0.010) | 0.045 (0.009) | 0.050 (0.009) | 0.072 (0.020) | 0.100 (0.007) |

Median and (stdev) over the sweep. A cell is bolded only when it beats the runner-up by more than the two spreads together: no width here does, so the choice is made on the shipped fallback rule.
<!-- END GENERATED:norm_cores -->

16 and 20 swap places between runs by about the run-to-run spread; from 32 upwards the
shard/unshard overhead starts to dominate. `NORM_SHARD_CORES` is 20, with a fallback that picks
the largest divisor of `hidden_size / TILE_WIDTH` the device grid allows. PCC of the sharded
norm against torch: 0.999990, against 0.999928 interleaved. In the committed reports the sharded norm is far below the interleaved one it
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
(`logs/probe_decode_recurrence.log`), median and spread over 30 repeats:

<!-- GENERATED:recurrence_grid -->
**48 head problems** (batch 1)

| shape | default | 1x4 | 1x8 | 1x11 | 2x4 | 2x8 | 2x11 | 4x4 | 4x8 | 4x11 | 6x4 | 6x8 | 6x11 | 8x4 | 8x8 | 8x11 | 10x4 | 10x8 | 10x11 | selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| state read | 113.1 (12.9) | 71.5 (3.7) | 55.6 (3.9) | 57.3 (4.4) | 52.7 (3.3) | 46.5 (4.7) | 50.3 (4.3) | 42.8 (3.6) | 46.3 (4.5) | 47.3 (2.8) | 39.4 (2.7) | 43.6 (3.1) | 43.8 (3.3) | 42.6 (2.6) | 44.0 (9.3) | 44.3 (6.3) | 45.0 (2.7) | 44.1 (2.8) | 43.9 (2.9) | 6x4 |
| outer product (`transpose` + `matmul`) | 108.8 (9.4) | 88.5 (13.6) | 58.1 (6.3) | 53.5 (9.2) | 56.8 (4.4) | 43.7 (7.0) | 45.5 (6.0) | 45.9 (13.4) | 45.3 (7.3) | 46.1 (7.3) | 45.9 (6.3) | 44.7 (6.2) | 43.6 (6.6) | 48.2 (9.3) | 44.7 (5.6) | 44.9 (7.4) | 49.8 (5.8) | 46.3 (7.1) | 47.6 (7.7) | — |
| outer product (`transpose_a=True`, shipped) | 107.5 (6.6) | 87.6 (9.3) | 58.6 (11.0) | 55.4 (7.6) | 58.1 (4.9) | 45.1 (11.3) | 47.7 (8.7) | 44.9 (4.7) | 46.3 (9.2) | 45.8 (4.3) | 44.4 (5.9) | 45.0 (12.2) | 43.4 (5.3) | 48.5 (9.9) | 44.5 (7.5) | 45.2 (6.5) | 51.9 (6.9) | 46.7 (5.4) | 47.8 (7.1) | 2x11 |

**1536 head problems** (batch 32, the advertised `max_batch`)

| shape | default | 1x4 | 1x8 | 1x11 | 2x4 | 2x8 | 2x11 | 4x4 | 4x8 | 4x11 | 6x4 | 6x8 | 6x11 | 8x4 | 8x8 | 8x11 | 10x4 | 10x8 | 10x11 | selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| state read | 2957.5 (6.9) | 1704.3 (10.2) | 1177.0 (10.2) | 1087.0 (9.7) | 978.8 (11.9) | 821.2 (11.7) | 828.6 (12.1) | 654.9 (15.2) | 664.6 (10.7) | 673.1 (12.4) | 591.7 (8.6) | 602.9 (10.6) | 608.8 (8.8) | 590.5 (10.7) | 603.7 (10.3) | 605.0 (8.9) | 549.4 (8.0) | 563.9 (9.4) | 555.5 (7.6) | 10x4 |
| outer product (`transpose` + `matmul`) | 2618.0 (12.9) | 2109.4 (8.7) | 1126.8 (10.0) | 858.2 (9.6) | 1128.9 (12.6) | 645.7 (23.7) | 517.4 (9.9) | 650.5 (14.1) | 569.8 (12.6) | 560.9 (14.3) | 666.1 (11.9) | 576.3 (10.5) | 557.1 (12.4) | 650.2 (11.9) | 566.1 (12.3) | 555.1 (11.3) | 645.5 (13.1) | 564.3 (11.0) | 565.0 (12.3) | — |
| outer product (`transpose_a=True`, shipped) | 2604.5 (19.5) | 2115.2 (13.4) | 1129.3 (14.9) | 869.1 (15.3) | 1124.9 (11.7) | 645.6 (11.8) | 513.1 (10.0) | 652.9 (14.1) | 570.2 (13.3) | 555.4 (11.2) | 668.5 (10.9) | 581.4 (13.9) | 557.5 (12.9) | 646.9 (10.8) | 564.3 (12.7) | 549.0 (9.6) | 641.6 (11.4) | 558.1 (10.4) | 557.0 (9.7) | 2x11 |

Median and (stdev) in microseconds over 30 repeats, at both decode regimes, over 18 explicit grids plus the program factory's own choice. The default is the slowest row of every sweep here. Read from the log: at 48 head problems the state read's 6x4 is the fastest measured; at 48 head problems the outer product's 2x11 is inside the combined spread of the fastest, 6x11 at 43.4 us; at 1536 head problems the state read's 10x4 is the fastest measured; at 1536 head problems the outer product's 2x11 is the fastest measured. The shipped form of the outer product folds its transpose into the matmul, which is one dispatch fewer and bit-exact, and has its own row so that choice is a measurement rather than an argument.
<!-- END GENERATED:recurrence_grid -->

All exact (PCC 1.000000 against torch); the grid only changes how independent per-head problems
are distributed. Taken, clamped to the device's real grid. Device time for the three recurrence
matmuls in the final decode profile: a fraction of what the functional stage's three take, which §6.1's generated table quantifies; both are rows of the committed
decode reports.

`ttnn.experimental.group_attn_matmul` was tried for the state read twice. The first attempt put
the flattened `batch * num_v_heads` axis in the op's batch slot and recorded
`TT_FATAL: Num of users must match!` as the blocker; a stage review showed that was a mapping
mistake, not the op's contract. Mapped the way the op wants -
`a = [1, num_v_heads, batch, head_dim]`, `b = [batch, num_v_heads, head_k_dim, head_v_dim]`, which
the user-major recurrent state provides as a free leading-dimension view - every shape assertion
passes at `max_batch` 32, and this is what happens instead:

<!-- GENERATED:group_attn_matmul -->
| input dtype | outcome |
|---|---|
| fp32 | `rejected: RuntimeError: TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/program/program.cpp:1722: tt::exception` |
| bf16 | `rejected: RuntimeError: TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/program/program.cpp:1722: tt::exception` |

The overflow is the whole of it: 6484864 B of circular buffers against 1572864 B of L1; 3298176 B of circular buffers against 1572864 B of L1.
<!-- END GENERATED:group_attn_matmul -->

That is the real blocker and it is quantified: the op's circular buffers do not fit in this
device's L1, by a factor of four in float32 and still a factor of two in bfloat16, because it
wants the whole `[32, 48, 128, 128]` state resident per core range. Probe:
`probes/probe_group_attn_matmul.py`, log `logs/probe_group_attn_matmul.log`.

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
| height-broadcast | 2.279 ms — **74 GB/s** | 0.449 ms — 187 GB/s |
| same-shape | 0.716 ms — 352 GB/s | 0.435 ms — 290 GB/s |
<!-- END GENERATED:broadcast_bandwidth -->

The float32 height-broadcast multiply is the outlier: 5x below what the same op reaches on
same-shape float32 operands. Every formulation of the FIR in the table below was measured against torch
(all PCC 1.000000 in float32, 0.999990 in bfloat16):

<!-- GENERATED:conv_formulations -->
| formulation | float32 median (stdev) ms | bfloat16 median (stdev) ms |
|---|---|---|
| all-TILE slices (what the functional layer does) | 15.957 (0.059) | 5.253 (0.032) |
| untilize once, ROW_MAJOR shift, tilize per tap (TILE concat) | 15.725 (0.041) | 5.138 (0.037) |
| ROW_MAJOR concat *and* shift, SiLU folded into the last add - **shipped** | 14.251 (0.042) | **4.341** (0.026) |
| untilize once, whole FIR in ROW_MAJOR, tilize once | 14.260 (0.039) | 7.424 (0.040) |
| one pre-padded window per tap so every slice is tile-aligned | 18.062 (0.044) | 6.460 (0.047) |
| scale on the TILE tensor first, then untilize per tap and shift-and-add in ROW_MAJOR | 17.379 (0.072) | 5.682 (0.031) |

A bolded median is a minimum outside the two spreads together; a column with no bold has its two fastest rows inside each other's spread.
<!-- END GENERATED:conv_formulations -->

So the win is the dtype first - the generated table pairs every formulation with its float32 twin -
and the layout second: at bfloat16, the dtype the FIR ships in, building *and* shifting the window
in ROW_MAJOR is the fastest row outside the spreads, bit-identically. At float32 it is nominally
first but inside its runner-up's spread, which is why the table bolds only the bfloat16 cell. The FIR now runs in bfloat16 while the carried conv **state** stays
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
| fused gate/up matmul + 2 slices + `silu` + `multiply` (functional) | 8.473 ms | 0.952 ms |
| fused gate/up matmul + 2 slices + `multiply(act=SILU)` — **shipped** | 7.987 ms | 0.932 ms |
| split gate/up matmuls, `silu` on the gate matmul's `activation=` epilogue, `multiply` | 10.479 ms | 0.925 ms |

`probe_mlp_variants.py` reports best-of-N wall time and prints no spread, so no cell here is marked a win - the shipped variant is labelled instead. The prefill column separates the three by margins far larger than any spread this stage has measured on that shape; the decode column does not, and §6 records the split variant as rejected on prefill. Re-running this probe with median and stdev, as every other probe reports, is listed as a limitation.
<!-- END GENERATED:mlp_variants -->

The split variant is the textbook "matmul + activation" merge and it removes both slices, but at
prefill it is **the slowest of the three** in the table above — two `[hidden, intermediate]`
matmuls lose more than the slices cost. The decode column decides nothing either way: this probe
reports a best-of-N wall time with no spread, and the three variants land within a hair of each
other there, so no cell in it is marked a win. Rejected on the prefill
measurement; the SiLU fold is kept.

The same input-activation merge applies twice more, and both were taken: the attention output
gate's `sigmoid` — two ops in the functional prefill report, one in the fused one — and the
gated delta net's `silu(z)`.

### 3.9 `in_proj_b` + `in_proj_a` → one matmul

**Kind:** graph rewrite (shared-LHS peer merge).

Both read the normed hidden state, both are `num_v_heads` = 48 wide, and both are
float32-weighted (stage 1 kept them float32 deliberately: the softplus/exp gating is
numerically sensitive). At decode each was its own `32 x 5120 x 64` matmul on a handful of cores —
dispatch-bound, not bandwidth-bound — so packing them into one `[hidden, 128]` weight halves
that. The second block starts at a tile-aligned column (`_AB_STRIDE` = 64) so both halves come
back out with a plain slice. In the committed decode reports the functional layer's two
`32 x 5120 x 64` matmuls become one `32 x 5120 x 128`; the committed decode reports carry both,
and §6.1's generated table carries the fused row's share of the step. The two unpacked
weights are freed in `from_state_dict` rather than left on device.

Packing `in_proj_qkv` and `in_proj_z` in as well was considered and rejected: those two matmuls
are DRAM-bound on their weights (5120x10240 and 5120x6144 bfloat16), so merging them moves the
same bytes and saves only one dispatch, and it would force the float32 `a`/`b` weights to
bfloat16 (a precision change stage 1 explicitly avoided) or the bfloat16 qkv/z weights to
float32 (doubling 168 MB of weight traffic).

`dt_bias` is added to `a` immediately after this matmul, and a bias row is exactly what
`ttnn.linear` already takes, so it rides on the same call: the packed bias is zero under the `b`
block and `dt_bias` under the `a` block, and the separate `ttnn.add` is gone. One dispatch fewer
per `linear_attention` layer per phase, at unchanged PCC (§4's table is the after).

The same shared-LHS merge applied to `full_attention`'s `wqkv` and `wgate` — the largest such
pair left, and the one HF itself ships merged — and **lost**; the measurement is in §6.

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

A stage review then found the cost of that layout: the functional layer rewrites the packed
`conv_state` every decode step, and the fused one did not, so anything reading `conv_state`
mid-generation got the post-prefill window. The fix is one more buffer, not a
copy-back: there is now one buffer per *packed row* (`K`, not `K - 1`), the FIR reads rows
`1..K-1`, and the shift rotates all of them, so `FusedDecoder.current_conv_state()` folds them
into the packed buffer exactly. `test_conv_state_after_decode_matches_reference` runs 1 and 5
decode steps and compares that against HF's own cache object.

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
| token-major output + group-reduction norm (§3.4) — **shipped** | 5.40 ms |
| `output_head_major=True` + per-head `ttnn.rms_norm` + z/result relayouts | 10.78 ms |

PCC between the two outputs: 0.999994. `probe_output_paths.py` reports a single wall time with no spread, so the shipped path is labelled rather than bolded as a measured win.
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

`ttnn.addcmul` was considered for the whole tap (`acc + tap * w` in one op). An earlier round
rejected it as "a composite in this checkout", citing `ternary_composite_op.cpp`, and a stage
review showed that reading was **wrong**: `ttnn::addcmul` dispatches
`ttnn::prim::ternary(TernaryOpType::ADDCMUL, ...)`, a single LLK device op
(`ttnn/cpp/ttnn/operations/eltwise/ternary/ternary.cpp`), and only decomposes for invalid or
subtile block-float broadcasts. §3.21 takes it where it pays - the recurrent-state update - and
this tap case is measured there too: the taps are already one multiply and one add with the SiLU
riding on the add, so `addcmul` would replace two ops with one *and* lose the fused activation.

### 3.15 KV-cache typecasts guarded on dtype

**Kind:** graph rewrite (remove a no-op).

`_full_attention_prefill` cast K and V to the cache dtype unconditionally. At the default
`cache_dtype = bfloat16` they are already bfloat16, so both casts were `bfloat16 -> bfloat16`
no-ops dispatched on 110 cores on every prefill chunk. Now guarded on
`k.dtype != k_cache.dtype`; `test_bfloat8_kv_cache` covers the branch that still casts.

### 3.16 the decode recurrence grids, re-measured at the advertised batch

**Kind:** dispatch/parallelism (§3.6, re-opened).

§3.6 chose the two recurrence core grids from a sweep at 48 head problems, which is batch 1. At
`max_batch` 32 the same two matmuls are 1536 head problems and, per the generated table there,
a large share of the whole step, which §6.1's generated table quantifies - so the sweep was
re-run at both counts rather than assumed to transfer. The state read's `6x4` is the fastest measured at *both*. The outer product's grids sit
inside a stdev of each other at batch 1 but not at batch 32, so it now takes the grid that wins
where the op is ten times as expensive - the minimum at 1536 head problems and inside the
combined spread of the best at 48. The state read is the one lever where no single grid wins at
both: once the sweep ran to the edge of the device grid, `6x4` was fastest at 48 head problems and
`10x4` at 1536, each distinguishably faster than the other in its own regime, so it is **keyed by
regime** and the layer picks by its own head-problem count. Every figure and every "selected" cell
in §3.6's tables is read from the log and from the shipped constants. Both tables are in §3.6, generated from the same log.

### 3.17 decode z-gated norm: two forms, picked by batch

**Kind:** graph rewrite (the same algebraic one as §3.4, applied conditionally).

Round 1's §3.4 replaced the per-head gated norm's two tile relayouts with a group reduction in
*prefill*, and §6 recorded the decode side as "about the same" - an estimate, which a stage
review correctly refused. Measured at the real decode shapes, median over 25 repeats
(microseconds):

<!-- GENERATED:gated_norm_batches -->
| batch | 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| reshape + `ttnn.rms_norm` (us) | 73.0 (23.1) | 99.2 (19.3) | 91.5 (3.1) | 150.1 (0.8) | 238.5 (0.7) |
| group reduction (us) | 145.4 (4.8) | 146.6 (16.6) | 146.0 (25.8) | 166.1 (2.9) | 150.1 (19.1) |

Median and (stdev) in microseconds over 25 repeats. Lowest PCC between the two forms' outputs, over all batches measured: 0.999993. The group form first becomes distinguishably faster at batch 32, which is where the shipped threshold sits. Per batch, by the same rule: batch 1 reshape wins, batch 4 reshape wins, batch 8 reshape wins, batch 16 reshape wins, batch 32 group wins.
<!-- END GENERATED:gated_norm_batches -->

The group form is two skinny constant matmuls and barely moves with the row count; the reshape
form's two tile relayouts grow with it. The table above puts the crossing between **16 and 32** -
the reshape form wins every batch up to and including 16, and the group form wins at 32 - and the
per-batch verdicts in the caption are derived from the medians and their spreads rather than
asserted here. The threshold is that crossing, which
the generated caption derives from the log and `::test_selected_constants_are_the_measured_best`
binds to it. The layer picks by
`max_batch` (`_GATED_NORM_GROUP_BATCH`) rather than committing to one, and
`test_batched_users[32-linear_attention]` covers the branch the batch-1 perf runs do not take.
Probe: `probes/probe_gated_norm_batch.py`, log `logs/probe_gated_norm_batch.log`.

Both forms share the group reduction's arithmetic tail, and one more op merged out of it here:
the epsilon add carries `RSQRT` as its **output** activation (`activations=[UnaryOpType.RSQRT]`),
so `rsqrt(mean_square + eps)` is one dispatch rather than two. There is no `UnaryDeviceOperation`
rsqrt row left in either committed `linear_attention` report.

### 3.18 small-N matmuls: name the core grid

**Kind:** dispatch/parallelism (the same lever as §3.6, applied to the rows this stage created).

Three of the matmuls in the fused graph have an N of 2, 4 or 192 tiles: the packed `a`/`b`
projection and the two gated-norm constant matmuls. The default 1D program factory spreads
output columns across the whole 110-core grid, so a row with four output tiles broadcasts its
activation to every core and then leaves almost all of them idle. Every grid from the full
device down to `1x2` was measured at both row counts:

<!-- GENERATED:matmul_grids -->
| row | shape | default | 10x11 | 8x8 | 4x8 | 2x8 | 2x4 | 1x4 | 1x2 |
|---|---|---|---|---|---|---|---|---|---|
| `in_proj_qkv` prefill | 2048x5120x10240 | 2543.0 | 2690.0 | 3745.8 | 5940.6 | 11016.5 | 21863.6 | 43628.3 | 87190.5 |
| `gated_norm_sum` prefill | 2048x6144x64 | 177.4 | 177.6 | 144.8 | 174.0 | 207.3 | 229.1 | 373.8 | 851.8 |
| `in_proj_ab` prefill | 2048x5120x128 | 169.8 | 169.2 | 162.1 | 155.0 | 183.9 | 310.6 | 586.4 | 1428.1 |
| `gated_norm_exp` prefill | 2048x64x6144 | 136.6 | 132.7 | 137.9 | 183.8 | 221.8 | 337.0 | 632.6 | 1210.2 |
| `in_proj_ab` decode | 32x5120x128 | 79.8 | 78.2 | 78.5 | 47.4 | 37.1 | 33.1 | 31.3 | 40.1 |
| `gated_norm_sum` decode | 32x6144x64 | 91.8 | 89.6 | 69.3 | 48.4 | 38.9 | 34.8 | 34.4 | 35.5 |
| `gated_norm_exp` decode | 32x64x6144 | 33.8 | 31.4 | 27.3 | 25.4 | 25.4 | 26.6 | 30.2 | 43.3 |

Median microseconds over 25 repeats; `default` is the program factory's own choice at the shipped output dtype. The rows whose N is 2 or 4 tiles are 2-3x faster on a small explicit grid, because the default spreads output columns over the whole device and then broadcasts the activation to cores that have nothing to do.
<!-- END GENERATED:matmul_grids -->

The shipped grids are `in_proj_ab` 4x8 at prefill and 1x4 at decode, the gated-norm sum matmul
8x8 at prefill and 1x4 at decode, and the gated-norm expand matmul the default at prefill and
2x8 at decode. The decode rows are the ones that matter: two of them are 2-3x faster than the
default. The grid is a function of the row count as well as the shape, which is why prefill and
decode are separate entries rather than one constant.

The other two levers on the same rows — the output dtype and `fp32_dest_acc_en` — were measured
on the same shapes and are not worth taking, and both are precision policy rather than graph
shape:

<!-- GENERATED:matmul_dtype_levers -->
| row | shape | shipped | output dtype swapped | `fp32_dest_acc_en=False` |
|---|---|---|---|---|
| `in_proj_qkv` prefill | 2048x5120x10240 | 2543.0 us (fp32) | 2626.0 us (bf16) | 2721.2 us |
| `gated_norm_sum` prefill | 2048x6144x64 | 177.4 us (fp32) | 176.5 us (bf16) | 178.1 us |
| `in_proj_ab` prefill | 2048x5120x128 | 169.8 us (fp32) | 166.8 us (bf16) | 167.1 us |
| `gated_norm_exp` prefill | 2048x64x6144 | 136.6 us (bf16) | 238.3 us (fp32) | 141.3 us |
| `in_proj_ab` decode | 32x5120x128 | 79.8 us (fp32) | 78.1 us (bf16) | 78.1 us |
| `gated_norm_sum` decode | 32x6144x64 | 91.8 us (fp32) | 92.0 us (bf16) | 89.5 us |
| `gated_norm_exp` decode | 32x64x6144 | 33.8 us (bf16) | 32.3 us (fp32) | 30.0 us |

Median microseconds over 25 repeats. Neither lever is worth taking here, and both are precision policy rather than graph shape: the output dtype of these rows is what the next op consumes, and `fp32_dest_acc_en` is the stage-1 compute-kernel policy.
<!-- END GENERATED:matmul_dtype_levers -->

That table is also the answer for `in_proj_qkv`, the largest `Bound=SLOW` row in the report
(and one this stage inherited rather than created): no grid beats the default — every smaller
grid is worse, monotonically — and swapping its float32 output for bfloat16 measures neutral-to-slower at
this shape (the generated dtype-lever table above), quite apart from changing what the causal conv
carries. It is bound by neither DRAM nor FLOPs at the full
grid, which makes it a matmul-scheduling question (K-split or DRAM-sharded program config) for
the optimized-decoder stage, recorded in §6.

Probe: `probes/probe_matmul_bound.py`, log `logs/probe_matmul_bound.log`.

### 3.19 the last two unary-into-binary folds

**Kind:** op merging.

Decode computed `exp(g)` and then multiplied the recurrent state by it, and `sigmoid(b)` and
then multiplied the delta residual by it. Both unaries ride on their consumer's
`input_tensor_b_activations` instead, even though both operands are height-and-width broadcast:

<!-- GENERATED:input_folds -->
| fold | batch | separate unary | folded into the binary | agreement |
|---|---|---|---|---|
| `exp(g)` into the recurrent-state multiply | 1 | 69.4 us | 58.4 us | PCC 1.000000, max abs diff 0.000e+00 |
| `exp(g)` into the recurrent-state multiply | 32 | 598.7 us | 580.5 us | PCC 1.000000, max abs diff 0.000e+00 |
| `sigmoid(b)` into the `delta` multiply | 1 | 80.9 us | **69.6 us** | PCC 1.000000, max abs diff 0.000e+00 |
| `sigmoid(b)` into the `delta` multiply | 32 | 427.7 us | 420.4 us | PCC 1.000000, max abs diff 0.000e+00 |
| rank-3 before the slices instead of after (not taken) | 2048 rows | 262.2 us (shipped) | 259.1 us | PCC 1.000000, max abs diff 0.000e+00 |

Median microseconds over 25 repeats (9 for the rank-3 row). Both folds are bit-exact and both were taken; on time, 1 of the 4 rows is a win outside the spreads and the rest are ties, so what a fold buys for certain is a dispatch, not microseconds. A bolded cell is a win outside the two spreads together. Moving the rank change ahead of the slices removes two float32 reshapes and adds one, and measures as a tie, so the shipped order stands.
<!-- END GENERATED:input_folds -->

Both are bit-exact (maximum absolute difference 0.0), and both were taken: two dispatches fewer
per `linear_attention` decode step. `sigmoid` can only fold on the decode path — in prefill
`beta` is an *input* of `chunk_gated_delta_rule`, and an op input has no activation slot — so
`_gdn_inputs` takes a `raw_beta` flag rather than the decode path re-deriving it.

The last row of that table is the rank-3 slice order, which was measured and **not** taken: the
same table records it as a tie.

Probe: `probes/probe_gdn_input_folds.py`, log `logs/probe_gdn_input_folds.log`.

### 3.20 the fused graph's own peak, at both decode batches

After all of the above, more than half of the `linear_attention` prefill is the `matmul` bucket
— the four projections, the MLP's two matmuls and the two gated-norm constant matmuls — and a
tenth is `chunk_gated_delta_rule` itself. The largest remaining non-matmul cost is the causal
conv, which the probe measures in isolation and which §3.7 records every formulation of. Both
decodes are four fifths and seven eighths matmul time **at batch 1**, at the roofline the
profiler's own `DRAM` column reports — the device's DRAM roofline for bfloat16 weights, and the
point at which graph fusing stops being the lever and weight dtype starts.

At the advertised `max_batch` of 32 the weights are the same bytes but the state is not: the
`linear_attention` step's recurrence buckets grow with the batch while the weight-bound ones do
not - the growth table in [`README.md`](README.md) carries every bucket at both batches with its
multiple and its share, and those are the numbers to read rather than any fraction restated here -
because the carried recurrent state is 100 MB at
that batch and a step decays, reads and writes it. §6.1 records every `Bound=SLOW` row of both regimes and what
each was measured against; the levers that exist there — the two recurrence core grids, the
`exp`/`sigmoid` folds, the per-head norm/expand order and the decode FIR's dtype — were all
swept at 1536 head problems as well as 48, and the shipped choices are the ones that win there.
What is left at batch 32 is the state's own bandwidth and the tile padding of a one-row-per-head
layout, which is the recurrent state's *shape* rather than the graph over it: changing it would
change what `prepare_decode_state` carries and what the HF cache comparison reads, so it is
recorded in §6 rather than taken here.

The full bucket breakdown of all six measured passes is the `breakdown_ms` block of
`perf_summary.json`, reproduced in [`README.md`](README.md); it is derived from the report's own
op codes rather than added up by hand, and its `other` bucket is empty.

### 3.21 the recurrent-state update → one `ttnn.addcmul`

**Kind:** dedicated op (the skill's first-priority rewrite), found by re-reading a rejected one.

Decode's state update is `state * exp(g) + update`. It shipped as two full-size passes over the
carried state - a multiply with `exp` riding on it (§3.19) and an add writing back into the
persistent buffer (§3.10). At the advertised `max_batch` that state is 100 MB of float32, so the
pair reads 300 MB and writes 200 MB per step, and it was the largest non-matmul cost of the step.

`ttnn.addcmul(a, b, c) = a + b * c` is exactly that arithmetic in one op. §3.14 had recorded
`addcmul` as "a composite in this checkout"; a stage review checked the source and showed it
dispatches a single LLK ternary op here. Measured at the real shapes:

<!-- GENERATED:addcmul_state -->
| batch | multiply + add (was) | `addcmul` | `addcmul` in place | agreement |
|---|---|---|---|---|
| 1 | 84.8 us | 65.3 us | 57.7 us | PCC 1.000000 in place, 1.000000 against torch, max abs diff 2.980e-08 |
| 32 | 1276.7 us | 772.2 us | 770.1 us | PCC 1.000000 in place, 1.000000 against torch, max abs diff 5.960e-08 |

Median and spread over 15 repeats, state uploaded once outside the timed region. A bolded cell is a minimum outside the spreads; where none is bolded, the two `addcmul` forms are inside each other's spread and both are decisively below the two-op form. The in-place form is what ships, for a reason the timing does not carry: it lands at the persistent buffer's address, which the traced decode needs. It is bit-exact against both the two-op form and torch.
<!-- END GENERATED:addcmul_state -->

Taken. Two consequences, both in the code's comments: `exp(g)` becomes its own tiny op again
(the multiply it used to ride on no longer exists), and the state *read* now consumes the
undecayed state with the decay applied to its `[1, BH, 1, head_v_dim]` result instead - which is
exact, because `g` is one scalar per head, so `k @ (state * g) == (k @ state) * g`. The in-place
form (`output_tensor=` aliasing the state) is what the traced decode needs, and the probe checks
it against torch as well as against the two-op form.

Probe: `probes/probe_addcmul_state.py`, log `logs/probe_addcmul_state.log`.

### 3.22 decode rotate-half → back to the spelled-out form

**Kind:** a dedicated-op rewrite **reverted on measurement**.

§3.2 replaced the functional layer's `slice`/`slice`/`neg`/`concat` rotate-half with
`ttnn.experimental.rotate_half` on both paths and recorded it as an op-count win, with no timing.
A stage review read the committed reports and found the dedicated op is the largest layout-ish row
of the batch-32 `full_attention` decode, on **one core**: its program factory pins
`CoreCoord({0, 0})`, so it cannot parallelise and its cost grows with the batch, while the four ops
it replaced run on 64 to 110 cores.

Both forms were then measured two ways, because they disagree:

<!-- GENERATED:rope_half -->
| batch | `ttnn.experimental.rotate_half` | spelled out (shipped) | agreement |
|---|---|---|---|
| 1 | 13.9 us | 41.3 us | PCC 1.000000, max abs diff 0.000e+00 |
| 32 | 41.3 us | 60.3 us | PCC 1.000000, max abs diff 0.000e+00 |

Median over 25 repeats, *wall clock*, so dispatch is on the critical path - which is why this table favours the dedicated op and the traced pass measurement does not.
<!-- END GENERATED:rope_half -->

Wall-clock (above) favours the dedicated op, because four dispatches cost more than one when
dispatch is on the critical path. Under **trace**, which is how decode actually runs, dispatch is
not on the critical path and device time is - so the whole traced `full_attention` decode pass was
profiled with each form, and **both runs are committed**:

<!-- GENERATED:rope_half_traced -->
| pass | dedicated `rotate_half` (rejected) | spelled out (shipped) | difference |
|---|---|---|---|
| `full_attention` decode, batch 1 | 2.067 ms | **2.070 ms** | -0.1 % |
| `full_attention` decode, batch 32 | 2.911 ms | **2.869 ms** | +1.4 % |

Device time per trace replay, summed over the signposted window of each committed report. The two runs differ only in this one op - the rejected one's provenance carries a different `FUSED_BUILD` fingerprint, which is how it is identifiable as the alternative.
<!-- END GENERATED:rope_half_traced -->

The spelled-out form ships. Prefill keeps `rotary_embedding_hf`, which is a different op on a
different shape and a measured win (§3.2).

This is the one place in the stage where the graph has *more* python-level ops than it could have,
which is why `test_fused_graph_is_smaller` asserts `<=` rather than `<` for decode and says so.

Probe: `probes/probe_decode_rope_half.py`, log `logs/probe_decode_rope_half.log`.

### 3.23 the causal conv's non-final taps → `ttnn.addcmul`

**Kind:** dedicated op, on the taps §3.14 had excluded.

§3.14 rejected `addcmul` for the FIR because the SiLU rides on the last tap's add. That is true of
*that* add and of no other: the FIR's other taps are a plain multiply and a plain add, and there
are two of them in each pass. §3.21 established that `addcmul` is a single LLK ternary op here, so
those two pairs become two ops:

<!-- GENERATED:conv_tap_addcmul -->
| pass | rows | dtype | `multiply` + `add` | `addcmul` | agreement |
|---|---|---|---|---|---|
| prefill | 2048 | bf16 | 605.1 us | **405.8 us** | PCC 0.999999, max abs diff 7.812e-03 |
| decode | 32 | fp32 | 86.1 us | **61.7 us** | PCC 1.000000, max abs diff 1.192e-07 |

Median over 15 repeats, per tap, each at the dtype its path runs (§3.7 makes the prefill FIR bfloat16, §3.25 keeps the decode one float32). Where the two forms differ at all it is rounding of the *intermediate*: the two-op form rounds `state * w` to the tensor dtype before the add and the fused one keeps it in the accumulator, so the fused result is the closer of the two to exact arithmetic, not the further.
<!-- END GENERATED:conv_tap_addcmul -->

Taken in both FIRs. The last tap keeps its multiply-and-add so the SiLU still rides on it. Note the
direction of the small difference: the two-op form rounds `state * w` to bfloat16 *before* the add,
and the fused form keeps it in the accumulator, so the fused result is the closer of the two to
float32 arithmetic.

Probe: `probes/probe_addcmul_state.py` (its `conv_tap` rows), log `logs/probe_addcmul_state.log`.

### 3.24 the recurrence's transients → dense, not one padded row per head

**Kind:** graph rewrite (layout), on the cost §6 had excluded as a state-format change.

The decode recurrence works in `[1, batch * num_v_heads, 1, head_dim]`, and TILE layout pads that
logical height of 1 to 32 rows - so 31 of every 32 bytes moved are padding. §6 recorded the whole
of that cost as a *state-format* change and handed it on. A stage review pointed out that this is
true only of the three matmuls: `v`, `b`, `kv_mem`, `residual` and `delta` are transients that no
matmul reads in that shape, and they can be dense.

<!-- GENERATED:dense_recurrence -->
| batch | one padded row per head | dense `[1, batch, heads, dim]` | agreement |
|---|---|---|---|
| 1 | 100.0 us | 101.9 us | PCC 1.000000, max abs diff 0.000e+00 |
| 32 | 741.0 us | **408.6 us** | PCC 1.000000, max abs diff 0.000e+00 |

Median over 25 repeats over the `subtract` / `sigmoid`-multiply chain and the rank changes each form needs. Bit-identical. At batch 1 a tie; at batch 32 dense wins - at the advertised batch the dense form is far ahead, because a one-row-per-head TILE tensor carries 31 padding rows for every real one. A bolded cell is a win outside the combined spread; a row with none is a tie.
<!-- END GENERATED:dense_recurrence -->

Taken: the state read's per-head-row result is turned dense once, the subtract and the gated
multiply run there, and only `delta` is turned back into rows for the outer product. Bit-identical
at both regimes. What remains padded is the *state* itself and the two matmul boundaries, which is
the part that really is a state-format change, and §6 now says only that.

Probe: `probes/probe_dense_recurrence.py`, log `logs/probe_dense_recurrence.log`.

### 3.25 the decode FIR's dtype: faster in bfloat16, and reverted

**Kind:** a rewrite that was measured, **built**, and then rejected by the suite rather than by a
probe - the one case in this stage where those two disagree.

§3.7 moved the *prefill* FIR to bfloat16 for a measured 3x. The decode FIR stayed float32, and
§6 recorded that as "the casts cancel the win". With the shipped tap arithmetic (§3.23) that is no
longer what the probe says:

<!-- GENERATED:decode_conv_dtype -->
| batch | float32 (shipped) | bfloat16 | PCC of each against torch |
|---|---|---|---|
| 1 | **112.5 us** | 137.2 us | 1.000000 / 0.999990 (float32 faster) |
| 4 | 209.0 us | **139.6 us** | 1.000000 / 0.999990 (bfloat16 faster) |
| 8 | 214.6 us | **140.1 us** | 1.000000 / 0.999989 (bfloat16 faster) |
| 16 | 213.8 us | **141.6 us** | 1.000000 / 0.999989 (bfloat16 faster) |
| 32 | 212.6 us | **140.8 us** | 1.000000 / 0.999989 (bfloat16 faster) |

Median over 25 repeats, both forms accumulating the way the shipped FIR does. The bolded cell is the faster of the pair where they are outside their combined spread.
<!-- END GENERATED:decode_conv_dtype -->

From batch 4 up the bfloat16 form is clearly faster. It was implemented behind a threshold
constant and the suite rejected it, and **that run is committed** - `logs/rejected_bf16_decode_fir.log`,
produced by the same suite against a build whose only difference is this constant:

<!-- GENERATED:rejected_bf16_fir -->
| case | PCC against HF |
|---|---|
| the run | **2 failed**, 4 passed |
| batched traced decode, batch 4, user 0 (the 64-token prefill), replay 1 | **0.976326** |
| batched traced decode, batch 32, user 0 (the 64-token prefill), replay 1 | **0.980567** |
| unbatched decode after a 2049-token prefill, steps 0/1/2/3 | 0.999888 / 0.999872 / 0.999914 / 0.999900 |
| synthetic carried state after 1 / 8 steps, fp32 FIR | 1.000000 / 1.000000 |
| synthetic carried state after 1 / 8 steps, bf16 FIR | 0.999987 / 0.999978 |

Read the rows together. The failures are at the **shortest** prefill in the batch, at the first replay the test checks; the same build's *unbatched* decode after a 2049-token prefill holds four steps with no downward trend. So what separates pass from fail here is the size of the carried state, not the number of steps - and the synthetic per-step probe, written to show compounding, does not reproduce the failure at all. That negative result is why this table states the pattern rather than a mechanism: the rejection rests on the committed failing run, and the *why* is left as the open question it is.
<!-- END GENERATED:rejected_bf16_fir -->

What the artifacts *do not* support is the mechanism an earlier draft of this section gave. That
draft said the per-step error compounds, and `probes/probe_fir_dtype_compounding.py` was written to
show it - it does not: eight steps of the synthetic recurrence move the carried state by 9e-6,
against the 0.02 the failing run shows, i.e. nowhere near it (both figures are in the table above). The failing run points somewhere else: both
failures are the **shortest-prefill user** in the batch at the first replay checked, while the same
build's unbatched decode after a 2049-token prefill is flat over four steps. The pattern is the
size of the carried state, not the number of steps, and the honest position is that the *why* is
open - the plausible reading, which this stage did not confirm, is that with a small state
`delta = beta * (v - k @ S)` is a difference of two close quantities and a bfloat16 FIR perturbs it
relatively more.

What is settled is the decision and its evidence: the decode FIR stays float32,
`_DECODE_CONV_BF16_BATCH` is `None`, the failing run and the negative probe result are both
committed, and the constant's docstring carries the same summary so the datatype-sweep stage - which
owns precision policy - inherits the measurement rather than a story. The prefill FIR is not
affected either way: its output feeds `chunk_gated_delta_rule`, which casts to bfloat16 anyway and
accumulates the state in kernel.

Probe: `probes/probe_decode_conv_dtype.py`, log `logs/probe_decode_conv_dtype.log`.

---

## 4. Correctness

Acceptance bar is unchanged: **PCC >= 0.995**, the same bar and the same HF reference harness as
stage 1. `tests/test_fused_decoder.py` is the stage-1 contract re-run against `FusedDecoder` —
same sequence-length coverage, same paged-cache and state comparisons, same batch, traced-decode,
determinism, block-size, BFP8-cache and full-context tests — plus the ones that only exist
because this is a fusing stage:

* `test_fused_ops_are_dispatched` — asserts each op in `FusedDecoder.FUSED_OPS` is actually
  called on a real pass, so a shape or dtype that silently missed a fused path fails loudly
  instead of quietly costing performance;
* `test_fused_graph_is_smaller` — the fused graph dispatches strictly fewer `ttnn` ops than the
  functional one for the same prefill, and no more for the same decode (§3.22 trades six of them
  for device time);
* `test_fused_matches_functional` — fused and functional outputs agree with **each other**, from
  identical weights and inputs, not only with HF;
* `test_repeated_runs_stable` — six prefill+decode cycles, bit-identical across cycles, with the
  device's per-bank allocated bytes required to be unchanged from cycle 1 onward (a per-cycle
  device leak in the `_free` aliasing rules is a failure);
* `test_no_relayout_or_host_ops_in_measured_decode` — the layer itself asks for no
  `tilize`/`untilize`/`to_layout` in a measured decode, and its reshard count is exactly the set
  each remaining reshard's op contract justifies;
* `test_merged_unaries_are_not_dispatched` — the unaries this stage merged into their consumers
  (`rsqrt`, `sigmoid`, `exp`, `silu`) are not dispatched on their own in a measured decode, and
  in prefill only `linear_attention`'s `beta` sigmoid remains, because it feeds an op input.

Final results, all against the shipped code:

<!-- GENERATED:run_totals -->
* `logs/suite_main.log` — **87 passed, 2 skipped**
* `logs/long_context.log` — **2 passed**
* `logs/watcher_run.log` — **23 passed**
* `pcc_evidence.json` — 493 records, 460 of them PCC, **minimum 0.998030**, none below the 0.995 bar; 4 full-context scale ratios, range 0.99577 to 0.99750, inside the ±2 % tolerance.
<!-- END GENERATED:run_totals -->

`long_context.log` is the 262143-token prefill and the decode at position 262143 for both layer
kinds; `watcher_run.log` is the same suite under `TT_METAL_WATCHER=10`, with the log audited
clean by `watcher/WATCHER_AUDIT.md`.

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
| `linear_attention` | prefill, 2048 tokens | 150.966 ms | **25.681 ms** | **5.88x** | 801 | 66 |
| `linear_attention` | traced decode, 1 token, batch 1 | 3.040 ms | **2.352 ms** | **1.29x** | 92 | 67 |
| `linear_attention` | traced decode, 1 token, batch 32 (advertised `max_batch`) | 36.627 ms | **5.166 ms** | **7.09x** | 93 | 70 |
| `full_attention` | prefill, 2048 tokens | 18.536 ms | **17.807 ms** | **1.04x** | 44 | 28 |
| `full_attention` | traced decode, 1 token, batch 1 | 2.271 ms | **2.070 ms** | **1.10x** | 50 | 50 |
| `full_attention` | traced decode, 1 token, batch 32 (advertised `max_batch`) | 3.065 ms | **2.869 ms** | **1.07x** | 49 | 49 |
<!-- END GENERATED:before_after -->

Every row is faster, and none is larger. The stage contract is the first of those — "fewer ops or
cleaner topology is not enough" — and `test_speedup_block_is_consistent` asserts it directly out
of `perf_summary.json`.

One row deserves a note rather than only a number. The `linear_attention` batch-32 "before" is
twelve times its own batch-1 figure, because the stage-1 decode reshapes the hidden state to
`[1, batch, 1, hidden]` and so runs each user's projections as a separate batched matmul with one
padded tile row — 32x the projection work at batch 32. The fused layer keeps
`[1, 1, batch, hidden]`, which is where most of that speed-up comes from. It is a real property
of the baseline, measured by the same script minutes apart against a byte-identical
`tt/functional_decoder.py`, not a mis-measurement — but it is a layout win rather than a fusing
win, and the table should be read that way.

The `full_attention` prefill moves least because it was already a fused graph in stage 1:
`nlp_create_qkv_heads`, `chunked_scaled_dot_product_attention`, `paged_fill_cache` and
`nlp_concat_heads` were all in place, so what was left to fuse there was the partial RoPE, the
MLP's SiLU and the output gate's sigmoid, against a pass whose `matmul` and `sdpa` buckets are
almost all of it and which no graph rewrite touches: five matmuls and one SDPA call, and the
generated breakdown above is where those two buckets' share comes from. Those five matmuls are
not bandwidth-starved either — the committed report classifies every one of them `Bound=FLOP`,
and its console log puts the whole pass well under the DRAM roofline — so what is left there is
matmul scheduling and precision policy, which later stages own, not graph fusing. The `decode`
passes are the opposite: the report classifies every one of their large matmuls `Bound=DRAM`,
close to the bandwidth roofline, which is why decode moves on op count, core grid and dispatch
rather than on arithmetic.

## 6. What was assessed and not taken

Recorded here so "no remaining fusing" is a claim with evidence behind it, not an assertion.

| candidate | why not |
|---|---|
| `gated_delta_attn_seq` instead of `chunk_gated_delta_rule` | strictly more ops for the same result, and an exact contract blocker for the decode step: `gated_delta_attn_device_operation.cpp` requires `C == head_k_dim == head_v_dim == 128` with all-float32 inputs *and* eight pre-built intermediates, so it cannot express a one-token step at all |
| `ttnn.swiglu` for the MLP | a composite in this checkout (split + silu + multiply, `unary_composite_op.cpp`), and it applies the SiLU to the *second* half where this model needs it on the first - so it is both more ops than §3.8's shipped form and the wrong arithmetic |
| `chunk_gated_delta_rule` on the **decode** path as well as prefill | an exact op-contract blocker at the advertised batch: the phased program factory asserts `TT_FATAL(BH <= ncores, ...)` (`chunk_gdn_phased_program_factory.cpp:137`) with `BH = batch * num_v_heads`. At `max_batch` 32 that is 1536 against this device's 110 compute cores, so it fits only up to batch 2. It is also the wrong shape for decode: the op's flat path needs `T % 32 == 0` and decode's `T` is 1, so it would have to be called at chunk 32 with 31 masked positions per step |
| the decode group-reduction gated norm **at small batch** | measured at 1/4/8/16/32 (§3.17). The crossing is between 16 and 32, so the layer picks by `max_batch`: the reshape form below 32, the group reduction at and above it, and `test_batched_users` covers 4, 16 and 32 - both sides and the boundary. Both were measured, neither is estimated |
| `ttnn.conv1d` on 2048-channel groups of the causal conv | tried, exact blocker: `TT_FATAL @ ttnn/cpp/ttnn/operations/sliding_window/op_slicing/op_slicing.cpp:266: found_valid_config` - the sliding-window op finds no valid slicing for a 2051x1 depthwise conv over 2048 channels, which is why the in-tree reference falls back to an FIR past 2048 channels rather than grouping |
| width-sharding the `full_attention` decode Q/K head norms | tried at 4 and 8 cores, exact blocker: `TT_FATAL @ tt_metal/impl/tensor/spec/tensor_spec.cpp:161: !shard_grid_fit_error` - a `[1, batch, heads, 256]` tensor's shard grid does not fit. It is two small ops of a ~2 ms step in any case |
| `use_qk_l2norm=True` on the fused op | `TT_FATAL: use_qk_l2norm not yet supported`; the flat rank-3 path gives the in-kernel norm anyway |
| chunk size 64 on the fused op | measured PCC 0.903635 at 2048 tokens against 0.999994 at 32 (§3.1) |
| `rotary_embedding_hf` decode mode | needs a HEIGHT_SHARDED input and sharded per-user cos/sin; prefill mode broadcasts cos/sin over the batch axis (§3.2) |
| `paged_fused_update_cache` | requires disjoint shard core sets for the two inputs; K and V share cores (§3.5) |
| `group_attn_matmul` for the recurrence | **not** a contract blocker: mapped the way the op wants, every shape assertion passes at `max_batch` 32 and the op overflows L1 - 6484864 B of circular buffers in float32, 3298176 B in bfloat16, against 1572864 B (§3.6). A resource blocker, so a later stage with a different core range, a chunked state or more chips may revisit it |
| `hc_sum_reduce` / `repeat_and_interleave_eltwise_mul` for the gated norm | fixed 32-wide groups; this model's group is 128 |
| `output_head_major=True` on `chunk_gated_delta_rule` | measured about 2x the shipped token-major path, PCC 0.999994 between the two outputs (§3.13) |
| the RoPE head-channel permutation | measured worth well under 1 % of the prefill, and it widens the public `rot_mats` contract and permutes what the KV cache holds (§3.2) |
| `ttnn.transformer.split_query_key_value_and_split_heads` / `nlp_create_qkv_heads` for `linear_attention`'s `_split_qkv` | those ops split a packed projection into *equal* q/k/v head sets; here q and k are 16 heads of 128 and v is 48 of 128, which none of their signatures can express. The three slices stay, and the row above measures the alternative that does not need them - three projections and three FIRs - and finds it slower |
| `ttnn.addcmul` for the conv taps | **taken for every tap it fits** (§3.23): only the *last* tap's add carries the SiLU, so only that one keeps a separate multiply and add. The "composite in this checkout" blocker an earlier round recorded here was factually wrong and is corrected in §3.14; `addcmul` is also what §3.21 uses for the recurrent-state update |
| emitting bfloat16 from `in_proj_qkv` to remove the conv's float32 -> bfloat16 cast | it would also make the carried conv state bfloat16. That state is compared against HF's cache object at 0.999995 and stage 1 chose float32 for it deliberately; changing a *carried state's* dtype is a precision-policy decision the datatype-sweep stage owns, not a graph rewrite. The cast that remains is a real dtype conversion, not a no-op (§3.15 removed the ones that were) |
| `ttnn.conv1d` / depthwise `conv2d` for the causal conv | conv CBs overflow L1 at 10240 channels |
| the pre-aligned-window, all-ROW_MAJOR-arithmetic, untilize-once-and-shift, and all-TILE causal-conv formulations | all measured slower than the shipped ROW_MAJOR-concat-and-shift form; the generated table in §3.7 is the whole comparison, every formulation against torch and against each other |
| the **decode** causal-conv FIR in bfloat16, as §3.7 made the prefill one | **measured, built, and rejected by the suite** (§3.25): from batch 4 up the bfloat16 form is clearly faster, and it fails correctness - batched traced decode falls below the bar for the shortest-prefill user, in a run committed as `logs/rejected_bf16_decode_fir.log`. The mechanism is *open*: a per-step compounding probe does not reproduce it, and what separates pass from fail in the artifacts is the size of the carried state |
| normalising the decode Q/K **before** the GQA expansion instead of after | **measured and rejected**: it is a third of the norm work, but `repeat_interleave` on the flattened head axis of a one-row-per-head tensor costs more than the norm saves - decisively at batch 32, and at batch 1 the two are inside the merged form's own spread, i.e. a tie rather than a win either way. Outputs identical. The table is below |
| a batch-dependent core grid for the two decode recurrence matmuls | **taken for the state read, not needed for the outer product**: eighteen grids swept at 48 *and* 1536 head problems (§3.6). No single grid wins the state read at both regimes, so it is keyed by regime (`6x4` small, `10x4` large); the outer product's `2x11` is the fastest measured at 1536 and inside the combined spread of the fastest at 48, so one constant holds. Both are re-derived from the log by `::test_selected_grids_are_the_measured_best`, and the tables' "selected" cells are read out of the shipped constants rather than typed |
| a recurrent-state layout with more than one row per head | after §3.24 the only tensors still in the one-row-per-head layout are the *state* and the two matmul boundaries; the transients around them are dense. Packing several heads into one tile row would need a block-diagonal state - the heads share no columns, so a shared tile row multiplies the state's bytes rather than saving them - and it would change the shape of the carried state, which `prepare_decode_state` writes, `test_traced_decode_*` reads and the HF cache comparison validates. That is a state-format change, not a graph rewrite, and it belongs to the stage that owns the decode state |
| `transpose_a=True` on the decode outer product | **taken**, not rejected: exact (PCC 1.000000) and one dispatch fewer (§3.6) |
| packing `wqkv` and `wgate` into one matmul | **measured and rejected**, not assumed: it is the largest shared-LHS pair left and HF itself ships the two as one weight, but the merged output has to be cut back apart, and at 2048 rows those two slices of a 14336-wide TILE tensor cost more than the activation re-read and the dispatch they save. At decode the two forms are within a stdev. The table is §6's `qkv_gate_pack` block below (§3.9) |
| moving the rank change ahead of the `a`/`b` slices | **measured and rejected**: it removes two float32 rank-changing reshapes and adds one, and measures as a tie at 2048 rows (§3.19) |
| a matmul program config for the remaining `Bound=SLOW` rows | every `SLOW` row in the committed reports is enumerated with its measurement below. What is left after §3.18's `core_grid` sweep is a K-split or DRAM-sharded *program config*, which is matmul scheduling rather than graph fusing and belongs to the optimized-decoder stage; recorded here so it is not lost |
| split gate/up MLP matmuls with `activation="silu"` | slowest of the three MLP forms at prefill (§3.8) |
| packing `in_proj_qkv`/`in_proj_z` into the `a`/`b` matmul | DRAM-bound already; would force a weight-dtype change (§3.9) |
| packing `in_proj_qkv` and `in_proj_z` into one matmul (the pair on its own, no dtype objection - both are bfloat16) | **measured and rejected**: it is the largest shared-LHS pair in the `linear_attention` graph, and the merged output has to be cut apart again. At 2048 rows the packed form is about 44 % slower; at decode the two are inside a stdev. Same table as §6.2 (`probes/probe_qkv_gate_pack.py`) |
| removing `repeat_interleave` from the decode GQA head expansion | it is a relayout **inside** a dedicated op — 2 `untilize_with_unpadding` + 2 `tilize_with_val_padding` in the committed decode report, a little over 1 % of the step at batch 1 and under 1 % at batch 32. The two alternatives are a `[key_dim, value_dim]` 0/1-matrix matmul, whose weight alone is 25 MB against a percent of the step in headroom, or a recurrent-state layout in which v-head `h` maps to k-head `h % num_k_heads` instead of `h // v_per_k`, which would break the direct comparison of the on-device state against HF's cache object. A third variant a stage review raised - concatenating the `v_per_k` value-heads of one k-head along the value axis, `[batch * num_k_heads, head_k_dim, v_per_k * head_v_dim]`, which needs no reordering because HF's mapping is already `h // v_per_k` - is the same *state-format* change as the padding one two rows above: it changes what `prepare_decode_state` writes and what the cache comparison reads, so it belongs to the same owner |
| widening the decode SDPA beyond one core per head | not a graph property: stage 1 pins `max_cores_per_head_batch = 1` to work around an upstream cross-core tree-reduction defect in `sdpa_decode`, documented with a model-free reproducer, and hands the kernel fix to the optimization stage. It is the whole `sdpa` bucket of the fused decode breakdown, and the README's growth table carries its share at both batches; this stage does not touch it |
| bfloat8/bfloat4 weights, lower math fidelity | precision policy, owned by the datatype-sweep stage; this stage changes the graph at a fixed precision policy, with the one exception in §3.7 where the arithmetic dtype *is* the graph property being measured |
| merging the decode Q and K head chains into one norm, scale and rank change | **measured and rejected**: they are the same shape and take the same path, so one `rms_norm` over `2 * num_k_heads` heads with a per-head scale column replaces two - but concatenating them and cutting the result apart costs more than the shared norm saves, at both batches and with identical outputs (`probes/probe_decode_qk_pair.py`). §3.24 reaches the rest of what a graph rewrite can: the transients around the recurrence are dense now, and what stays padded is the state itself and the two matmul boundaries |
| splitting `in_proj_qkv` into three projections and three FIRs, to remove `_split_qkv`'s three slices | **measured and rejected**: the three slices are among the larger `layout` rows of the `linear_attention` prefill, and §6.2 found the analogous cut expensive - but splitting the projection also splits the depthwise conv, and the whole chain measures slower than the packed one (`probes/probe_prefill_qkv_split.py`) |

<!-- GENERATED:rejected_shared_work -->
| candidate | shape | shipped | merged | agreement |
|---|---|---|---|---|
| decode Q and K through one norm/scale/rank-change chain | batch 1 | **197.8 us** (separate) | 212.1 us (merged) | PCC 1.000000 / 1.000000 |
| decode Q and K through one norm/scale/rank-change chain | batch 32 | **509.0 us** (separate) | 527.4 us (merged) | PCC 1.000000 / 1.000000 |
| prefill `in_proj_qkv` as three projections and three FIRs instead of one and three slices | 2048 tokens | **6.931 ms** (packed) | 7.229 ms (split) | PCC 1.000000 |

Median over 25 repeats (9 for the prefill row). Both merges are the same arithmetic as what ships and both measure slower: cutting a wide TILE tensor apart, or concatenating one, costs more than the shared work it enables - the same result §6.2 found for `wqkv`/`wgate`. A bolded cell is a win outside the two spreads together; a row with no bold is a tie.
<!-- END GENERATED:rejected_shared_work -->

<!-- GENERATED:rejected_decode_variants -->
| variant | batch | shipped | alternative | agreement |
|---|---|---|---|---|
| decode causal-conv FIR in bfloat16 instead of float32 | 1 | **112.5 us** (float32) | 137.2 us (bfloat16) | PCC 0.999990 between them, 0.999990 against torch |
| decode causal-conv FIR in bfloat16 instead of float32 | 32 | 212.6 us (float32) | **140.8 us** (bfloat16) | PCC 0.999989 between them, 0.999989 against torch |
| Q/K L2 norm before the GQA expansion instead of after | 1 | 114.3 us (expand, then norm) | 137.2 us (norm, then expand) | PCC 1.000000 between them |
| Q/K L2 norm before the GQA expansion instead of after | 32 | **291.2 us** (expand, then norm) | 338.1 us (norm, then expand) | PCC 1.000000 between them |

Median microseconds over 25 repeats. Both alternatives are the same arithmetic as what ships. The bolded cell in each row is the faster of the pair as measured, and a row with no bold is one where the two are inside their combined spread.
<!-- END GENERATED:rejected_decode_variants -->

Probes: `probes/probe_decode_conv_dtype.py` and `probes/probe_gdn_decode_heads.py`; logs
`logs/probe_decode_conv_dtype.log` and `logs/probe_gdn_decode_heads.log`.

### 6.1 every `Bound=SLOW` row in the committed reports

`tt-perf-report` labels a row `SLOW` when it reaches neither the DRAM nor the FLOP roofline. The
label is not a diagnosis, so each one was measured against the levers that are graph properties —
output dtype, DEST precision, and core grid — at its real shape. The enumeration is generated
from all six committed fused reports, because the first hand-written version of it was taken from
four of them and went stale the same day:

<!-- GENERATED:slow_rows -->
| pass | op | ops per pass | device time per pass | share | cores | DRAM % |
|---|---|---|---|---|---|---|
| `linear_attention` prefill | `MatmulDeviceOperation 2048 x 5120 x 10240` | 1 | 2242.5 us | 8.7 % | 110 | 18 % |
| `linear_attention` prefill | `MatmulDeviceOperation 2048 x 5120 x 128` | 1 | 135.0 us | 0.5 % | 32 | 36 % |
| `linear_attention` prefill | `MatmulDeviceOperation 2048 x 6144 x 64` | 1 | 120.2 us | 0.5 % | 64 | 43 % |
| `linear_attention` prefill | `MatmulDeviceOperation 2048 x 64 x 6144` | 1 | 109.7 us | 0.4 % | 110 | 47 % |
| `linear_attention` decode | `MatmulDeviceOperation b={48} x 32 x 128 x 128` | 3 | 60.5 us | 2.6 % | 22/24 | 41-50 % |
| `linear_attention` decode | `MatmulDeviceOperation 32 x 5120 x 128` | 1 | 15.1 us | 0.6 % | 4 | 38-39 % |
| `linear_attention` decode_batch32 | `MatmulDeviceOperation 32 x 6144 x 64` | 1 | 16.4 us | 0.3 % | 2 | 14 % |
| `linear_attention` decode_batch32 | `MatmulDeviceOperation 32 x 5120 x 128` | 1 | 15.1 us | 0.3 % | 4 | 38-39 % |
| `linear_attention` decode_batch32 | `MatmulDeviceOperation 32 x 64 x 6144` | 1 | 7.3 us | 0.1 % | 16 | 31-32 % |

9 `Bound=SLOW` op groups across the six committed fused reports, every one of them a `linear_attention` row. Device time and share are per pass — per trace replay for the two decode windows — and `cores` is what the profiler reports each instance ran on.
<!-- END GENERATED:slow_rows -->

| row | where | what the measurement says |
|---|---|---|
| `in_proj_qkv` `2048 x 5120 x 10240` | `linear_attention` prefill, the largest `SLOW` row and the only one that is a material share of the pass | inherited from stage 1, not created here. No grid beats the default (§3.18's sweep is monotonically worse below the full grid) and the dtype levers measure neutral-to-slower at these shapes, quite apart from changing what the conv carries. Matmul scheduling, handed to the optimized-decoder stage |
| `in_proj_ab` `2048 x 5120 x 128` and `32 x 5120 x 128` | both `linear_attention` phases | created by this stage (§3.9). The grid sweep **took** 4x8 at prefill and 1x4 at decode; the decode row is 2-3x faster than the default. It stays `SLOW` because 4 output tiles cannot saturate the device however they are placed |
| `gated_norm_sum` `2048 x 6144 x 64` | `linear_attention` prefill | created by this stage (§3.4). Grid 8x8 **taken**; N is 2 tiles, so the same ceiling applies |
| `gated_norm_expand` `2048 x 64 x 6144` | `linear_attention` prefill | created by this stage (§3.4). K is 2 tiles, which is what makes it `SLOW`; no grid or dtype lever moves it, and it is the cheapest of the three constant-matmul rows |
| the batched `b={48}` recurrence matmuls | `linear_attention` decode at batch 1 only — the generated table above carries no `b={1536}` row, because at the advertised batch the profiler classifies those instances `Bound=DRAM` at two thirds of the roofline and up | §3.6 swept eighteen grids for exactly these at both head counts, and the shipped grids are the fastest measured in their regime or inside the run-to-run spread of it, which §3.6's generated caption says row by row. The batch-1 instances stay `SLOW` because one token against a `[128, 128]` state is a single tile of work per head — the shape's own floor, not a placement mistake |
| `gated_norm_sum` / `gated_norm_expand` at `32 x ...` | `linear_attention` decode at `max_batch` 32 | created by this stage (§3.4); the grid sweep took 1x4 and 2x8 for them (§3.18). Together they are well under a percent of that step |

None of these is a fusing opportunity that was left: each is either taken, or a program-config
question that the next stage owns. The projection and constant-matmul rows are measured in
`logs/probe_matmul_bound.log`; the batched recurrence rows are measured in
`logs/probe_decode_recurrence.log`, at both head counts.

### 6.2 the shared-LHS merge that lost

`q_proj` in the checkpoint emits `[num_heads, 2 * head_dim]` and the per-head second half is the
output gate, so HF ships q and the gate as one weight and stage 1 split them into `wqkv` and
`wgate` — the two halves take different paths, and `nlp_create_qkv_heads*` needs exactly
`q|k|v`. That makes them the largest shared-LHS pair in the graph, so the skill's merge was
measured rather than argued about:

<!-- GENERATED:qkv_gate_pack -->
| pair | rows | output dtypes | two matmuls (shipped) | one packed matmul + 2 slices | verdict |
|---|---|---|---|---|---|
| `full_attention` `wqkv` + `wgate` | 2048 (prefill) | bf16/bf16 split, bf16 packed | **3171.4 us** | 4781.8 us | shipped wins |
| `full_attention` `wqkv` + `wgate` | 32 (decode) | bf16/bf16 split, bf16 packed | 424.8 us | 424.5 us | tie |
| `linear_attention` `in_proj_qkv` + `in_proj_z` | 2048 (prefill) | fp32/bf16 split, fp32 packed | **3944.0 us** | 6248.8 us | shipped wins |
| `linear_attention` `in_proj_qkv` + `in_proj_z` | 32 (decode) | fp32/bf16 split, fp32 packed | 498.4 us | 491.0 us | tie |

Median over 25 repeats (9 at 2048 rows), outputs identical to the precision the dtypes allow. A row is a tie when the two medians are inside their combined spread - that is the case for qkv_gate at 32 rows, qkv_z at 32 rows. The output dtype matters and is measured, not assumed: `in_proj_qkv` emits float32 because the causal conv carries float32 state, and a packed matmul has one output dtype, so the merge would push `in_proj_z` to float32 as well.
<!-- END GENERATED:qkv_gate_pack -->

Rejected on the measurement: at prefill the packed form is more than a third slower, because
cutting the merged output back apart is two full copies of a 14336-wide TILE tensor. Probe:
`probes/probe_qkv_gate_pack.py`, log `logs/probe_qkv_gate_pack.log`.

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
    -k "test_traced_decode_pcc or (test_decode_pcc and 2049) or test_bfloat8_kv_cache or test_traced_decode_batched or test_alternate_page_block_size or test_repeated_runs_stable or (test_batched_users and 32) or (test_merged_unaries_are_not_dispatched and 32) or test_conv_state_after_decode_matches_reference" \
    -v -s

# op-level probes (each is model-free and self-checking); probes/README.md lists what each answers
for p in models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/probes/probe_*.py; do python "$p"; done

# before/after profiling, one (kind, phase, impl) triple at a time.  ``decode_batch32`` is the
# same traced-decode window at the advertised max_batch, which takes a different graph branch.
doc/fused_decoder/probes/run_perf.sh <linear_attention|full_attention> <prefill|decode|decode_batch32> <functional|fused>

# ...or all of the above, in the only order that is correct, into the committed artifacts:
bash models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/probes/regenerate_evidence.sh

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
| **P1** a `tilize` -> `untilize` round trip over the whole ~42 MB conv window survived in the measured prefill: `ttnn.concat` on TILE operands untilizes them, concatenates and re-tilizes, and the ROW_MAJOR tap loop then threw that tilize away. Every one of the four measured FIR formulations inherited the same TILE concat, so re-measuring them could not expose it | a fifth formulation that concatenates in ROW_MAJOR was built, measured over 12 repeats - faster by several times the spread, and bit-identical - and **shipped** (§3.7). `tests/test_fused_decoder.py::test_no_redundant_relayout_in_measured_prefill` was added to gate the class the goal contract names but nothing checked. Round 4 showed that python-level trap cannot see relayouts made *inside* `ttnn.concat`, so it is now backed by `test_fused_decoder_docs.py::test_no_layout_round_trip_in_the_measured_pass`, which reads the committed device report instead |
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
| `test_no_redundant_relayout_in_measured_prefill` could not see the defect it was written for: it traps python-level calls, and `ttnn.concat` relayouts *inside* itself | replaced by `::test_no_layout_round_trip_in_the_measured_pass`, which reads the committed `tt-perf-report` op sequence - the device-level truth - and fails on any `Tilize*` immediately followed by an `Untilize*`. It immediately found two more round trips in the state cut, which is now done in ROW_MAJOR end to end |
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
| `_GATED_NORM_GROUP_BATCH = 32` was chosen from a threshold no committed artifact measured - the round-4 fix had added the *mechanism* for picking between the two gated-norm forms without a probe behind the number | `probes/probe_gated_norm_batch.py` was written and run, and its log is committed and generated into §3.17. It backs the threshold, and the two forms agree at PCC 0.999994. (Round 5 read the crossing off that log as "the group form from batch 16 up"; the committed log puts it between 16 and 32, which §3.17's caption derives - the misreading is what made round 10 move the constant to 16, and round 16 moved it back) |
| `test_prose_perf_figures_match_the_summary` was vacuous: it accepted any figure appearing *as a substring anywhere* in the corpus, so a stale figure passed as long as some unrelated number contained its digits | rewritten around `_allowed_figures()`, which derives the allowed set by value from `perf_summary.json` (device time, gap, breakdown, top ops), the stage-1 summary, and every number the probe logs actually printed. Generated blocks are excluded from the scan and covered by `::test_generated_blocks_are_current` instead. It immediately found the ten remaining hand-transcribed figures, all now either generated or reworded to quote the table |

Its concerns were taken too: two skill patterns the earlier rounds had not assessed are now
**taken** - `dt_bias` folds into the packed `a`/`b` projection as `ttnn.linear(bias=...)` and the
gated norm's epsilon add carries `RSQRT` as its output activation, two dispatches fewer per
`linear_attention` layer per phase at unchanged PCC (§3.9, §3.17); §6's contradictory
conv-formulation row was corrected; the README now credits the device-level relayout gate rather
than the python-level one; and the three figures that no longer matched the regenerated perf
summary were replaced with the generated tables.

Round 6 returned **more-work-needed** with four P2 findings and seven concerns. Two of the four
were "you never measured this", and measuring both changed the shipped code:

| finding | what was done |
|---|---|
| the committed reports label six matmul rows `Bound=SLOW`, §6 asserted there was exactly one, and the largest of them - 8.6 % of the `linear_attention` prefill - appeared in no document | `probes/probe_matmul_bound.py` measures **every** `SLOW` row against the three levers that are graph properties, and §6.1 enumerates all of them with what the measurement says. The sweep found a real win the earlier rounds had missed: the rows whose N is 2 or 4 tiles are 2-3x faster on a small explicit `core_grid`, so the packed `a`/`b` projection and both gated-norm constant matmuls now name a grid per phase (§3.18) |
| `wqkv` and `wgate` are the largest shared-LHS matmul pair in the graph, HF ships them as one weight, and no artifact said whether merging them was tried | measured at both real shapes (`probes/probe_qkv_gate_pack.py`). **Rejected on the measurement**: cutting the merged output apart costs more at prefill than the merge saves, and it is a wash at decode. §6.2 carries the table |
| the fused contract's `capacity_conclusion` quoted a byte count three rounds stale, because the document gate covers the stage documents and the contract's *fields*, but not the contract's own prose | corrected, and `::test_contract_prose_matches_the_evidence` now holds every number in the contract's sentences to the same rule as the documents' |
| §5's perf conclusion was a sentence a previous round's edit had broken in half, and the DRAM-roofline claim that survived was contradicted by the report it cited | rewritten from the committed report: the prefill matmuls are `Bound=FLOP`, the decode matmuls are `Bound=DRAM`, and neither figure is now transcribed by hand |

Its concerns were taken too: the two remaining unary-into-binary folds (`exp` into the recurrent
state multiply, `sigmoid` into the delta multiply) are measured, bit-exact and **shipped**
(§3.19); the rank-3 slice reorder was measured, tied and recorded as not taken; the recurrence
probe now sweeps the shipped `transpose_a` form over every grid so it is a row of the generated
table rather than a footnote; decode at the advertised `max_batch` 32 - a *different* graph
branch, not a wider tensor - has its own before/after Tracy pair (`run_perf.sh <kind>
decode_batch32 <impl>`); the decode reshard gate is an equality against the enumerated set
instead of a budget with slack in it; `dt_bias` and the `RSQRT` fold are now recorded in the §3
sections that round 5 claimed carried them; and `probes/regenerate_evidence.sh` makes the
artifact rebuild order a committed script rather than a list in §7.

Round 7 returned **more-work-needed** with one P1 and five P2s, and the P1 was the sharpest
finding of the seven rounds: round 6 *measured* the advertised-`max_batch` decode and then
analysed nothing in it, so every "nothing left to fuse" sentence in the stage was a batch-1
sentence sitting next to a batch-32 report that contradicted it.

| finding | what was done |
|---|---|
| the batch-32 regime was measured and never analysed; the `matmul` bucket is under a third of that step, not the four fifths the documents claimed, and the rest of it is exactly the elementwise/batched-matmul/layout work fusing is supposed to attack | the breakdown table now carries all six measured passes, §3.20 and §5 state the batch-32 conclusion separately from the batch-1 one, and every lever the batch-32 profile exposes was measured: the two recurrence grids at 1536 head problems (§3.16), the decode FIR's dtype (§6), the norm/expand order (§6), and the recurrent state's one-row-per-head padding (§6, handed on as a state-format change with the reason) |
| the recurrence core grids were batch-1 constants shipped unchanged at 1536 head problems | swept at both counts. The state read's 6x4 is fastest at both; the outer product moves to the grid that wins at batch 32 and ties at batch 1 (§3.16). `probe_decode_recurrence.py` now sweeps both regimes, and the generated table has one block per regime |
| §6.1's "every `Bound=SLOW` row" was enumerated from four of the six reports, and its supporting-log sentence was wrong | the enumeration is a **generated** block over all six committed fused reports, grouped by op code, and the sentence now names the right log for each family |
| the decode causal-conv FIR still ran in float32, the pathology §3.7 measured and fixed in prefill | measured (`probe_decode_conv_dtype.py`): at decode the taps read the carried float32 state, so a bfloat16 FIR pays `K - 1` typecasts per step and comes out *slower* at batch 1 and level at batch 32. Rejected with the measurement, recorded in §6 |
| `test_watcher_audit_matches_its_artifacts` could not fail on the histogram it was credited with binding - a `continue` made the assertion unreachable | rewritten to collect the missing lines and assert on the collection. It immediately caught a second stale figure: the audit's "eight Tracy runs", now twelve and derived by counting the reports |
| four count statements contradicted the generated table beside them (conv formulations, probe count), and one measured-and-rejected conv formulation was missing from §6 | all corrected, the missing row added, and `::test_probe_readme_covers_every_probe` now checks the probe README against the directory so this class cannot come back |

Its concerns were taken too: the SDPA hand-off is quoted at both batches since
the optimization stage should plan against the larger one; the graph-shape gates
(`test_no_relayout_or_host_ops_in_measured_decode`, `test_merged_unaries_are_not_dispatched`) run
at `max_batch` 1 **and** 32; the device-level layout round-trip gate reads all six fused reports
rather than the two prefills; the merged-unary evidence records carry a `phase` tag so
`collect_evidence` keeps both halves; and §3's section order ends with the summary section
instead of burying it mid-list.

Round 8 returned **more-work-needed** with two P1s and three P2s, and both P1s were about the
gap between what an artifact *says* and what it *is*:

| finding | what was done |
|---|---|
| the whole batch-32 evidence set and six probe logs were never committed - the repository ignores `*.log` and `*.csv`, earlier rounds' artifacts had been force-added by hand and rounds 5-7's had not, so every "committed artifact" sentence about the batch-32 analysis was false and the document gate could not run on a fresh clone | all of it force-added; `probes/regenerate_evidence.sh` now force-adds after every regeneration; and `::test_every_artifact_the_gate_reads_is_tracked_by_git` derives the artifact list from the probe directory and the phase/impl/kind matrix and asserts each one is in `git ls-files` |
| the shipped outer-product recurrence grid was **not** the one that wins at batch 32 - its own probe log put another grid several stdevs ahead of the shipped one at 1536 head problems - and six places said it was, including a "selected" column inside a GENERATED block, because that cell was a literal in the generator | the grid is `2x11` now, and `::test_selected_grids_are_the_measured_best` re-derives every shipped `core_grid` constant from its probe log at every regime measured and fails if a grid is distinguishably slower than the log's own minimum. It rejects the `6x11` that shipped one commit ago, which is the point |
| the fused decode never wrote the packed `conv_state`, so anything reading it mid-generation got the post-prefill window, while the functional layer keeps it current - a silent divergence with no test | fixed structurally rather than papered over: there is one tap buffer per *packed row* now (`K`, not `K - 1`), the FIR reads rows 1..K-1, the shift rotates all of them, and `current_conv_state()` folds them back exactly. `test_conv_state_after_decode_matches_reference` runs 1 and 5 decode steps and compares the *post-decode* packed state against HF's cache object (a different quantity from the post-prefill one the correctness table carries) |
| four citations named the round-trip gate by its pre-round-7 name | corrected, and `::test_every_cited_test_name_exists` parses the test modules and fails on any cited `test_...` identifier that is not defined |
| §6 called the rejected bfloat16 decode FIR "level at batch 32"; the generated table under it says 9.8 % slower | restated from the table |

Its concerns were taken too: the SDPA hand-off no longer quotes a figure from the wrong
implementation; "more than an order of magnitude" is the two real multiples; the watcher run now
includes the advertised-`max_batch` branch and the new state test; `perf_summary.json`'s note no
longer says "batch 1" when the file carries batch-32 rows; and the four stray empty directories
and the stale `generated/test_reports/` tree are gone.

Round 9 returned **more-work-needed** with one P1 and four P2s. Every measurement in the stage
re-derived correctly this round; what did not hold was the agreement between artifacts:

| finding | what was done |
|---|---|
| the shipped outer-product grid was fixed in round 8 but the GENERATED table's "selected" cell and three prose passages still named the *old* grid, because that cell was a literal in the generator and the new gate only compared the constant to the log | the "selected" cells are read out of `tt/fused_decoder.py`'s constants now (`_shipped_grid`), so the table states what the layer ships by construction; the three prose passages no longer name a grid at all, they name the rule |
| the published watcher `-k` selected eleven of the seventeen tests the committed run ran - it was a literal in the audit generator and did not grow when the run did | `SELECTOR` is read out of `regenerate_evidence.sh`, and `::test_watcher_command_selects_the_run_it_documents` evaluates the published expression against every test id in the run log and against the documents' copy of it |
| `kernel_elf_paths.txt.gz` was untracked while the generated audit said it was committed - the same class as round 8's P1, on a path the new tracked-by-git gate did not cover | all three watcher artifacts are force-added by the regeneration script and are in the gate's required list; the audit sentence now describes what is actually committed |
| two bare-integer multipliers in two documents were stale against the regenerated summary, and the prose gate exempted that shape of figure by construction | both restated so no multiplier is transcribed, and the gate now reads `Nx` as a figure. Its limit for one- and two-digit values is written into the test: small integers are in every artifact, so the documents point at the table instead of typing ratios |
| the README's "only the op graph changes" claim did not mention the packed `conv_state` divergence, and the recovery accessor exists only on the subclass | the claim carries the exception, Known limitations describes it and why the accessor is not on the base class (that file is stage 1's, outside this stage's scope), and a serving stage is told what to call |

Its concerns were taken too: the two shared-work merges the profiles suggested - Q/K through one
norm chain at decode, and `in_proj_qkv` as three projections at prefill - were **measured** and
both are slower than what ships (§6, generated table); traced decode and the post-decode
conv-state check now run at the advertised `max_batch` of 32 as well as 4 and 1; and
`current_conv_state`'s docstring gives the real reason it is not trace-safe.

Round 10 returned **more-work-needed** with one P1 and one P2, and closed the numeric audit:
every figure in the stage re-derived, so both findings were about hand-written prose again.

| finding | what was done |
|---|---|
| three README coverage claims were stale against the artifacts rounds 8-9 expanded - the watcher count, the batches trace capture runs at, and how many cases are one-layer-kind-only - and the gate written to bind the first was exempt by construction (it only checked numbers of three digits or more) | the three claims are one **generated** block now, `coverage_claims`, read out of `suite_main.log` and `watcher_run.log`: selected cases, test functions, pass count, the batches traced decode runs at, and the layer-kind split. The watcher gate checks *every* integer on a line naming the audit, in both documents |
| the README credited the batch-32 speed-up to fusing the recurrence, while its own §5 says most of it is removing a stage-1 decode layout | the README carries §5's disclosure at the point it states the number |

Its concerns were taken as work rather than noted:

* the recurrence grid sweep stopped at `y = 6` while the trend was still improving. Widened to the
  edge of the device grid in both axes - and it moved the shipped state-read grid, which is now
  `10x4`: the fastest measured at 1536 head problems and inside the spread of the fastest at 48.
  `::test_selected_grids_are_the_measured_best` rejected the old constant against the new log,
  which is what that gate is for.
* `_GATED_NORM_GROUP_BATCH` was pinned at the advertised batch rather than the measured crossing,
  so any `max_batch` in 17..31 took the slower form. Round 10 read the crossing off the probe as
  "between 8 and 16" and moved the threshold to **16**; that reading is wrong against the
  committed log, which puts it between 16 and 32, so round 16 moved the constant back to 32 and
  bound it to the log. `test_batched_users[16-*]` still covers the boundary.
* the `in_proj_qkv`/`in_proj_z` pair - the largest shared-LHS pair with no dtype objection, both
  weights bfloat16 - had no row of its own. Measured: the packed form is about 44 % slower at
  prefill and a tie at decode (§6, same generated table as §6.2).
* a sixth causal-conv formulation, "scale on the TILE tensor first, then untilize per tap and
  shift-and-add in ROW_MAJOR", is the one ordering that moves the per-tap tilize off the critical
  path. Measured and rejected: slower than the shipped form in both dtypes, bit-identical output.
* the Q/K-pair rejection is restated against its log rather than asserted. Round 10 wrote that as
  "a tie at batch 1", which round 22 found the log denies - the shipped separate form leads at
  both batches, narrowly at 1 and decisively at 32, which is why §6's generated table bolds it in
  both rows; §8's round-7 row no longer transcribes a figure of the current artifact; and the
  `conv_state` divergence is described as being about the packed buffer only, which is the only
  buffer the functional layer writes during decode.

Round 11 returned **more-work-needed** with two P1s and two P2s, and the first P1 is the most
useful finding of the eleven rounds:

| finding | what was done |
|---|---|
| the committed correctness, long-context and watcher runs had been produced *before* round 10's shipped decode-configuration change: the four decode perf windows were re-profiled after it and the correctness runs were not, so the watcher-clean claim was about a build that no longer existed. No gate could see it, because every gate reads artifacts and none tied an artifact to the source | every pytest run in this stage now prints a SHA-256 of `tt/fused_decoder.py` at session start (`tests/conftest.py`), `probes/run_perf.sh` appends the same hash to each profiler provenance file, and `::test_every_run_was_made_against_the_shipped_build` asserts every run log and every provenance names the **current** hash. Then the whole evidence set was regenerated against the shipped build, so every committed record is of the code that ships |
| §3.6's generated table still listed thirteen grid columns after the sweep grew to eighteen, so its "selected" cell named a column the table did not contain, and the caption inside the block asserted "the state read's 6x4 is the fastest measured at both" - which its own log contradicts | the column set is read from the log now, and the caption is *derived*: for each family and regime it names the measured minimum and says whether the shipped grid is it, is inside the combined spread, or is slower. Four prose passages and one docstring line that named a grid or a count are corrected, and §6's row no longer records the batch-dependent grid as rejected - it is shipped for the state read |
| §3.17 and §6 still described a gated-norm threshold of 32 after round 10 moved it to 16 | restated from the shipped constant and the probe table, and the cited test id is the one that covers the boundary |
| §6.1's row said the batched recurrence matmuls "stay `SLOW`" at both batches; at the advertised batch the profiler classifies them `Bound=DRAM` | restated from the generated table, which carries no `b={1536}` row for exactly that reason |

Its concerns were taken too: the implementation comment that still described `K - 1` conv state
buffers says `K`; the README no longer credits §6.1 with §3.19's folds, no longer claims a
measurement for the state-layout hand-off, and no longer reads the MLP pair's 534 MB as the whole
step's weight traffic; and the state-layout rejection gives the real reason (a shared tile row
needs a block-diagonal state, which multiplies the state's bytes) rather than only the scope one.

Round 12 returned **more-work-needed** with two P1s and two P2s, and both P1s were real
optimisation defects rather than documentation ones - the first two in several rounds:

| finding | what was done |
|---|---|
| `ttnn.addcmul` had been rejected in §3.14 as "a composite in this checkout", and the source says otherwise: it dispatches a single LLK ternary op here. That made it an unassessed dedicated-op candidate sitting on the largest non-matmul cost of the advertised-batch decode - the two full-size passes over the 100 MB carried state | measured (`probes/probe_addcmul_state.py`) and **taken** (§3.21): one pass instead of two, in place at the persistent buffer's address, bit-exact against torch and against the two-op form. The state read now consumes the undecayed state with the decay moved onto its tiny result, which is exact because `g` is one scalar per head. The wrong blocker in §3.14 and §6 is corrected |
| the decode `rotate_half` substitution was recorded as an op-count win with no timing, and the committed reports show the dedicated op is single-core by construction and the largest layout-ish row of the batch-32 `full_attention` decode | measured both ways (`probes/probe_decode_rope_half.py`) and **reverted on the decode path** (§3.22). Wall clock favours the dedicated op; traced device time - which is how decode runs - favours the four ops it replaced, and the whole traced pass is about one and a half percent faster at batch 32 with them. Prefill keeps `rotary_embedding_hf`. `test_fused_graph_is_smaller` now asserts `<=` for decode and explains why a rewrite deliberately trades python-level ops for device time |
| "each shipped grid is the measured best in its regime" overstated what §3.6's own caption says about the outer product at 48 head problems | both sentences restated to the rule the code follows |
| three row labels in the generated correctness block named batches 4 and 32 after batch 16 was added, and one named only batch 4 for a row that aggregates 4 and 32 | the labels take their batches from the records they aggregate |

Its concerns were taken too: the watcher audit's histogram now buckets the way the command it
prints does; and §6 records why `nlp_create_qkv_heads`-style ops cannot express
`linear_attention`'s `_split_qkv` (q and k are 16 heads of 128 and v is 48, which those signatures
cannot describe).

Round 13 returned **more-work-needed** with two P1s and two P2s, all of them round 12's own
changes not having been propagated - and it confirmed the round-12 work itself: the reviewer
re-derived the `addcmul` algebra, the in-place aliasing and every perf figure and found no defect.

| finding | what was done |
|---|---|
| an eighth of the advertised-batch decode was sitting in the breakdown's `other` bucket - the stage's own new ternary op, which the bucket predicates never named - while both documents asserted `other` was empty | the recurrent-state update has its own `state_update` bucket, the last unclassified op (`Accumulation*`) joins `elementwise`, the "is it empty" sentence is **generated** from the summary rather than asserted, and `::test_no_device_time_is_unclassified` fails if anything lands in `other` again |
| five artifacts still claimed `ttnn.experimental.rotate_half` is dispatched after §3.22 reverted it - including the implementation's own module docstring and a README row describing a test guarantee that did not exist | all five corrected, `addcmul` added to the docstring's dedicated-op list, and `::test_documented_dedicated_ops_are_the_ones_shipped` now binds every dedicated-op name the documents use to what the layer actually dispatches, exempting only the paragraphs that record the revert |
| the batch-32 growth multipliers and the SDPA hand-off percentages were pre-round-12 numbers, true only if the addcmul's time is counted back into `elementwise` | replaced by a generated growth table carrying every bucket at both batches, and the prose points at it |
| §3.22's decisive measurement - the traced pass profiled with the dedicated op - was described but not committed, and the only committed measurement (wall clock) points the other way | both traced runs are committed now: the rejected alternative's reports live under `tracy/rejected/rotate_half_dedicated/` with their own `FUSED_BUILD` fingerprint, which is what identifies them as the alternative, and §3.22 carries a generated table over the pair |

Its concerns were taken too: "all four passes" is six; `build_fingerprint.py` states the limit of
what it covers; and the two conv-state PCC figures are named as the different quantities they are.

Round 14 returned **more-work-needed** with one P1 and four P2s. The P1 was another real
optimisation, found by reading an exclusion the stage had written too broadly:

| finding | what was done |
|---|---|
| about an eighth of the advertised-batch decode is tile padding on the recurrence's *transient* vectors, and §6 excluded the whole of that cost as "a state-format change" - which is true of the three matmuls and of nothing else. `v`, `b`, `kv_mem`, `residual` and `delta` are read by no matmul in that shape | measured (`probes/probe_dense_recurrence.py`) and **taken** (§3.24): the transient chain runs dense, one rank change in and one out, bit-identical, and at the advertised batch the chain measures a little over half what it did. §6's claim is narrowed to what is actually a state-format change |
| `addcmul` for the conv taps was rejected because the SiLU rides on the tap's add - true of the *last* tap and of no other, and the FIR has two more in each pass | measured and **taken** for the non-final taps in both FIRs (§3.23). The fused form is also the more accurate one: the two-op form rounds the intermediate to bfloat16 before the add and the fused one keeps it in the accumulator |
| the generated `rejected_decode_variants` table bolded a winner its own log contradicts at batch 32, and its caption asserted "slower at both batches" | the bolding and the caption are derived from the measured medians now, and §6's row says what the log says - a tie at the advertised batch |
| `WATCHER_AUDIT.md` was stale against its own generator (it counts the Tracy runs, and round 13 added two) | regenerated, and `::test_generated_blocks_are_current` re-runs the watcher generator too and requires byte equality |
| three hand-written figures contradicted the generated blocks beside them | all three restated or removed; where a share is meant, the prose points at the generated growth table |

Its concerns were taken too: the README's garbled clause and its wrong attribution of the
`elementwise` growth are fixed (that bucket is the transients, not the state, and §3.24 is why);
§6's table is one table again; and the `repeat_interleave` share is stated as the profile shows it.

Round 15 returned **more-work-needed** with one P1 and three P2s, and its P1 was the sharpest
methodological finding of the fifteen rounds: *the metric*.

| finding | what was done |
|---|---|
| every decode-path accept/reject rests on wall-clock probes, and §3.22 had already shown that metric can point the wrong way under trace - with one rejection (`in_proj_qkv` + `in_proj_z`) contradicted by its own log, measured at a dtype the layer does not use (both halves bfloat16, where the shipped `in_proj_qkv` emits float32) | the probe measures the **shipped** output dtypes now, and at those dtypes the packed form loses at prefill and ties at decode - the rejection holds and is finally measured against the trade the layer would actually make. The table's verdict cell and its closing sentence are derived from the measured medians and spreads instead of asserted, the way §6's other tables already were |
| `probe_causal_conv.py`'s "what ships" variant still used the pre-§3.23 tap arithmetic, so §3.7's five-way layout comparison was made against a FIR the layer no longer runs | every formulation accumulates the way the shipped FIR does now, through one shared helper, and the probe was re-run. The ranking is unchanged - the shipped ROW_MAJOR-concat form still wins in bfloat16 - and the re-run surfaced an exact op-contract blocker worth recording: `addcmul` is TILE-only, so the all-ROW_MAJOR formulation cannot use it |
| six statements said "five formulations" / "four alternatives" against a table that carries six | all of them reworded to point at the table rather than count it |
| "Every row is faster **and** smaller" is contradicted by the block above it: `full_attention` decode is 50 -> 50 and 49 -> 49 ops, deliberately, since §3.22 | restated as "faster, and none is larger", and the gate table's row says where the tie comes from |

Its concerns were taken as work too, and one of them turned into a rewrite that was built and then
**reverted by the suite**: with the shipped tap arithmetic the bfloat16 decode FIR is clearly
faster from batch 4 up, and it fails correctness - batched traced decode falls to PCC 0.98,
below the bar for the shortest-prefill user in the batch. §3.25
records the measurement, the implementation, the failure and the asymmetry with prefill that
explains it.

Round 16 returned **more-work-needed** with one P1 and two P2s. The P1 was the best-argued
rejection challenge of the sixteen rounds, and it was right:

| finding | what was done |
|---|---|
| the `group_attn_matmul` rejection was not earned: the probe had put the *flattened* `batch * num_v_heads` axis in the op's batch slot, so `TT_FATAL: Num of users must match!` was a mapping mistake rather than the op's contract, and the review showed the mapping the op wants satisfies every assertion at `max_batch` 32 | re-tried with that mapping (`probes/probe_group_attn_matmul.py`). Every shape assertion does pass - and the op then overflows L1: 6484864 B of circular buffers in float32 and 3298176 B in bfloat16 against 1572864 B. That is the real blocker, quantified, and out by a factor of two even at half precision. §3.6 records it with the numbers; §6, §1, the probes README and `probe_decode_recurrence.py`'s docstring were corrected in round 18, when a review found the propagation had stopped at §3.6 |
| `_GATED_NORM_GROUP_BATCH = 16` shipped the *slower* of the two measured forms for `max_batch` 16..31: the re-measurement puts the reshape form ahead at 16 by several times the spread | the threshold is 32 again, the generated table carries the **spreads** so a "within the spread" claim is checkable, its caption derives the crossing from the log, and `::test_selected_constants_are_the_measured_best` binds the constant to that log - the same tie the core grids have had since round 10 |
| the `rejected_decode_variants` caption still ended in a hard-coded clause asserting a tie its own row contradicts | the clause is derived: a row with no bold is one where the two are inside their combined spread |

Its concerns were taken too: `gated_delta_attn_seq`'s rejection carries its exact contract blocker
(`C == Dk == Dv == 128` and eight pre-built intermediates, so it cannot express a one-token step);
`ttnn.swiglu` is recorded as assessed (a composite, and it applies SiLU to the wrong half for this
model); and the unfreed typecast on the small-batch decode branch is routed through `_free` like
every other intermediate.

Round 17 returned **more-work-needed** with one P1 and two P2s, all of one shape: a probe that
measures something the layer does not run.

| finding | what was done |
|---|---|
| `probe_gated_norm_batch.py` measured the group form with the **default** program factory on both constant matmuls, while the layer ships explicit decode grids for them (§3.18) - so the threshold the log chooses, and the gate round 16 added to bind it, were both certified against a configuration that is not the shipped one | the probe passes the shipped `core_grid` for both matmuls now and was re-run. The crossing is unchanged - the reshape form still wins at 16 by many times the spread and the group form wins at 32 - so `_GATED_NORM_GROUP_BATCH = 32` stands, but it now stands on a measurement of the code that ships |
| `NORM_SHARD_CORES = 20` is not the measured minimum (16 is), and its "the two swap places between runs" defence rested on a best-of probe with no spread at all | `probe_small_ops.py` reports median and stdev like every other probe now. With spreads the two are inside each other's, which is what the docstring said and could not show; and `::test_selected_constants_are_the_measured_best` covers this constant too, under the same within-the-combined-spread rule as the core grids |
| three hand-written sentences described the batch-16 comparison in terms the then-committed log did not support | restated, and they now match the re-measured log |

Its concerns were taken too: the README's two `§5` citations for the rejected-options table point
at §6; and §6's `repeat_interleave` row records the third variant a review raised - concatenating
one k-head's value-heads along the value axis - as the state-format change it is, with the same
owner as the padding row above it.

Round 18 returned **more-work-needed** with three P2s and no P1, and said explicitly that it
found no unearned fusing rejection and that every number re-derives. All three were rejection
*reasons* - the one thing the gates had never bound:

| finding | what was done |
|---|---|
| §6's row for `group_attn_matmul` still stated the blocker §3.6 retracted two rounds earlier, §1 still called it a contract blocker, the probes README said the same, and §8's round-16 row claimed the propagation had happened | all four corrected from the quantified L1 overflow, and §8 says where the propagation actually stopped. The distinction matters: a contract blocker is final, an L1 one is revisitable with a different core range, a chunked state or more chips - which is what §6 now says |
| the probes README described the bfloat16 decode FIR as rejected for a typecast wash, when §3.25 rejected it on **correctness** after measuring it faster | restated from §3.25, so a later stage reads the durable insight rather than an invitation to re-take it |
| §3.23's decode conv-tap row measured **bfloat16** while the shipped decode FIR is float32 | each row is measured at the dtype its own path runs now. The shipped choice stands - one op instead of two at float32 as well - and at float32 the two forms agree to 1.2e-07 rather than bfloat16's 7.8e-03 |

And the class itself is closed rather than the instances: `::test_quoted_blockers_appear_in_a_committed_log`
takes every `TT_FATAL`/`TT_THROW`/named-assertion string the documents quote and requires it to
appear in a committed log **or** in this checkout's own source, so a retracted or mis-transcribed
blocker fails the gate. It rejects an invented assertion, and it is what would have caught this
round's P1 two rounds earlier.

Its concerns were taken too: the fusing-stage test list no longer miscounts itself, and
`test_repeated_runs_stable` - the only per-cycle DRAM-leak check - runs at the advertised
`max_batch` as well as at 1, which is the branch it had never covered.

Round 19 returned **more-work-needed** with three P2s and no P1, and said the stage has no
correctness, capability or performance defect and no unearned rejection. All three findings were
the last unbound figure shape: a measured ratio written in **words**.

| finding | what was done |
|---|---|
| §3.20's batch-32 conclusion said two buckets "grow by more than an order of magnitude … to well over half the pass"; the generated growth table beside it says otherwise, because §3.24 halved one of those buckets and round 13 split the other out | §3.20, §3.16 and the README now name the table instead of restating fractions from it |
| three artifacts said the output-dtype lever on the `Bound=SLOW` rows "buys a few percent"; the generated table measures it neutral-to-slower | all three restated from the table, which is also what its own caption said |
| two more word-figures were stale: the norm-shard docstring's "16 is nominally the faster" (the re-run says 20) and §3.9's "at half the cost" (the reports say an eighth) | both restated, and the docstring now defers to the within-the-spread rule the gate applies rather than naming a winner |

And the class is closed: `::test_no_unbound_comparatives_in_the_documents` fails on a list of
ratio-in-words phrases anywhere outside a generated block, so a sentence that wants to state a
ratio has to take it from a table or point at one. That is the shape every one of the last four
rounds' stale claims had.

Round 19 also named the one accept/reject whose decisive evidence was prose: §3.25's rejection of
the *measured-faster* bfloat16 decode FIR rested on a suite failure that no committed artifact
recorded. That run is committed now - `logs/rejected_bf16_decode_fir.log`, the same suite against
a build differing only in that constant - and §3.25 carries a generated block over it, together
with a new probe (`probes/probe_fir_dtype_compounding.py`) that shows the *mechanism*: the FIR is
accurate to five decimals per step, and the state it feeds compounds that error monotonically.

Its concerns were taken too: the probes README's tooling table has a row for
`regenerate_evidence.sh` and the gate checks the table against the directory; and the README's
batch-32 `layout` attribution names the flat-to-dense changes §3.24 introduced as well as the two
matmul boundaries.

Round 20 returned **more-work-needed** with a single P2 and no other findings: it confirmed no
correctness defect, no capability reduction, no unearned rejection and no like-for-like problem in
the perf comparison, and re-derived every figure in the stage exactly.

| finding | what was done |
|---|---|
| §3.25's *mechanism* for rejecting the bfloat16 decode FIR - "the per-step error compounds through the carried state" - is contradicted by the two artifacts it cites: the compounding probe moves the state by 9e-6 over eight steps, and the failing run's own records show four unbatched decode steps at a 2049-token prefill holding flat while the failures are the **shortest-prefill user** at the first replay checked | §3.25's block is generated from the failing run's records now, and states the pattern the artifacts show rather than a mechanism they do not: the size of the carried state separates pass from fail, not the step count. The compounding probe is kept and relabelled as the **negative result** it is, which is what rules compounding out. §6 and the constant's docstring say the same; the probe index was corrected in round 21, when a review found the propagation had stopped short of it, and the *why* is recorded as open - with the plausible reading (a small state makes `delta = beta * (v - k @ S)` a difference of close quantities) marked as unconfirmed |

The decision itself was already artifact-backed from round 19 and does not change: the bfloat16
decode FIR is faster from batch 4 up and fails the bar, so `_DECODE_CONV_BF16_BATCH` stays `None`.
What changed is that the datatype-sweep stage now inherits a measurement and an open question
instead of a story that its own evidence refutes.

Its concerns were taken too: the watcher audit's trace-capture sentence is derived from the run it
prints rather than naming two batches.

Round 21 returned **more-work-needed** with three P2s, all of one class the last rounds have been
narrowing: a hand-written justification for a shipped decision that the stage's own log
contradicts. It found no correctness, capability or performance defect and no unearned rejection,
and re-derived every published figure.

| finding | what was done |
|---|---|
| the probe index still gave §3.25's *retracted* mechanism ("the error compounds") in two rows, and §8's round-20 row claimed the index had been corrected | both rows restated from §3.25 - faster from batch 4 up, rejected on the committed failing run, pass/fail tracks the size of the carried state, compounding ruled out by a committed negative result - and the round-20 row now says where the propagation actually stopped |
| `_GATED_NORM_GROUP_BATCH`'s prose stated the batch-16 comparison in words rather than from the table | restated in both places - but **wrongly**, as a tie the log denies; round 22 found and fixed that, and the per-batch verdicts are generated now |
| the recurrence-grid docstring described "best-of-20 wall time", "roughly 3x slower than any explicit grid" and a "flat region … within run-to-run spread", none of which the current sweep shows - and it contradicted its own next paragraph, which is why the grid is keyed by regime | rewritten to say what the sweep shows and to point at §3.6's derived caption; the caption's own "several times slower than any explicit grid" literal is gone too |

Two structural gaps it named are closed rather than noted. Probe logs now carry the `FUSED_BUILD`
stamp - three shipped constants and every rejection are certified from them, and they were the one
artifact class with no tie to the source - and `::test_every_run_was_made_against_the_shipped_build`
checks all twenty of them. And the two configuration constants that are measured at their ends
rather than across their range are stated as a limitation in the README, where a later stage will
read it.

Round 22 returned **more-work-needed** with one P2, and it is the previous round's own fix: round
21 restated `_GATED_NORM_GROUP_BATCH`'s justification as "the two forms tie at 16", which the log
it cites denies - §3.17's table has the reshape form ahead there by several times the two spreads
put together, which is the same rule every other verdict in this document uses. The
shipped constant was never wrong (32 is the crossing either way, and
`::test_selected_constants_are_the_measured_best` derives it), but the sentence a later stage
would read to decide whether to move that threshold was, and it is the same false-tie reading
that made round 16 ship the slower form for `max_batch` 16..31. Round 22 re-derived every
published figure and found no correctness, capability or performance defect, no unearned
rejection and no stale artifact.

| finding | what was done |
|---|---|
| the batch-16 gated-norm comparison was called a tie in the constant's docstring, in §3.17 and in §8's round-21 row, against a log whose gap there is several times the two spreads put together | all three restated from the log, and the per-batch verdict is **generated** now: §3.17's caption names the winner of every batch it measured, derived from the medians and their spreads, so the sentence has nothing left to assert |
| the same rule caught two more: the probe index called the decode Q/K pair "a tie at batch 1" and §8's round-10 bullet said the same "which is what its log says", where the shipped separate form actually leads at both batches; and §3.24's generated table bolded the dense cell at batch 1, where the row is a tie and the *rows* form is nominally ahead | the index and the bullet say what the log says; the dense-recurrence generator bolds a cell only when the gap is outside the combined spread, and its caption states each batch's verdict |

The gap under all three is that no gate could see a *verdict*. `test_prose_perf_figures_match_the_summary`
binds figures carrying a unit and `test_no_unbound_comparatives_in_the_documents` binds ratios
written in words; a claim that two medians tie carries neither.
`::test_qualitative_verdicts_match_their_logs` closes it: for every sentence in the documents or
the shipped sources that names a batch and calls the comparison a tie or a win, it re-derives the
verdict from the raw probe log - independently of `make_doc_tables.py`, so a generator that
misclassified a row fails there too - and requires the two to agree. It is scoped to sections
citing exactly one probe log with a row at that batch; a section citing several is skipped rather
than guessed at.

Round 23 returned **more-work-needed** with two P2s, both of them holes in round 22's own fix. It
re-derived every perf row, every op count, the PCC minimum and all 34 generated blocks, and found
no correctness, capability or performance defect, no unearned rejection and no stale figure.

| finding | what was done |
|---|---|
| round 22 gated *sentences* but not table **markup**: four generators still bolded "the faster of the pair" by comparing medians while their caption promised the spread rule, so §6's Q/K-norm-order row read as a measured win at batch 1 where the two are inside their combined spread - contradicting §6's own prose thirty lines above it | every generator that bolds a measured cell routes through `_verdict()` now (`rejected_decode_variants`, `rejected_shared_work`, `input_folds`), so a tied row carries no bold, and `::test_generated_table_bolding_marks_a_measured_win` re-derives it from the log for **every** generated table row that names two timings - no per-table knowledge, so a table added later is covered by construction |
| `logs/doc_gate.log` was a round stale - a 43-test run committed beside a 44-test gate - and it was the one run log bound by neither the prose-count gate nor the build-stamp gate, so the stage's newest gate had no committed passing artifact | the log is regenerated and committed, `doc_gate` is in the build-stamp list, and `::test_doc_gate_log_is_of_the_shipped_gates` asserts the committed log ran every `def test_` this file defines |

Its other concerns were taken as work rather than noted: §8's round-5 and round-10 rows read the
gated-norm crossing off the log as 16 and are annotated with what the log says and how the
misreading propagated; the probe index no longer calls the two input folds "cheaper" (one of the
four rows is a win outside the spreads, the rest are ties, so what a fold buys for certain is a
dispatch); the verdict gate's skipped-claim count is capped and listed in its failure message
rather than computed and dropped; and the constant's docstring records the bimodality of the
group arm across re-runs, which is the mechanism behind the threshold moving twice.

Round 24 returned **more-work-needed** with one P2, and it is round 23's closure claim rather than
a new class: round 23 said "every generator that bolds a measured cell routes through
`_verdict()`", and three tables it never reached still bolded a tie - §3.3's core-count sweep
(where the bolded cell is not even the shipped constant, and the prose three lines below it says
the two are a tie), §3.7's float32 column, and §3.8's decode column, which bolded the **rejected**
variant. Round 24 re-derived every perf row, op count, PCC figure and generated block and found no
correctness, capability or performance defect and no unearned rejection.

| finding | what was done |
|---|---|
| three generated tables bolded a cell their own log makes a tie, and the new bolding gate reached only 8 of the 34 blocks because its matcher required a ``us``/``ms`` suffix and a probe-log pair line | `_distinguishable_min()` gives the sweep the same rule `_verdict()` gives a pair, and §3.3, §3.7 and §3.21 use it (§3.3's table carries the spreads now, so the tie is checkable from the table). The gate reads in-cell spreads as well as log pairs, and checks each bolded cell **down its column** as well as across its row, which is the direction §3.7's bold lives in |
| the gate's docstring claimed tables were covered "by construction" when three probes print a best-of-N wall time with no spread at all, so no tie rule can apply to them | those generators bold no timing cell now - they label the shipped row instead (§3.1, §3.2, §3.8, §3.13) - the docstring names the exclusion and the reason, and the README carries it as a limitation with the concrete fix: re-run those three probes with median and stdev, which needs the board |

Two prose claims the regenerated tables no longer support were restated: §3.7's "the fastest row by
several times the spread" is true at bfloat16, the dtype the FIR ships in, and a tie at float32;
and §3.8's "at decode the difference is inside run-to-run noise" asserted a spread its probe does
not print, so it now says the decode column decides nothing. §3.21's table stops bolding the
in-place column as though it beat plain `addcmul` - they are inside each other's spread, and the
in-place form ships for the trace-address reason its caption gives, not for the microseconds.

Checkpoint commits on `agentic-research/hous/qwen3.6-27b-v2` (local only; never pushed):

| SHA | what |
|---|---|
| `8c0f31b7421` | Qwen3.6-27B fused decoder: graph-fuse both layer kinds |
| `1c8d2c9ca18` | Qwen3.6-27B fused decoder: second-review fixes |
| `85ffa523178` | Qwen3.6-27B fused decoder: third-review fixes |
| `3cc755545c1` | Qwen3.6-27B fused decoder: fourth-review fixes |
| `9d9a41a151d` | Qwen3.6-27B fused decoder: fifth-review fixes |
| `9acf5e0f5e7` | Qwen3.6-27B fused decoder: sixth-review fixes |
| `ead8e05310b` | Qwen3.6-27B fused decoder: seventh-review fixes |
| `bbaf92d8e37` | Qwen3.6-27B fused decoder: eighth-review fixes |
| `2241c099254` | Qwen3.6-27B fused decoder: ninth-review fixes |
| `8f412308861` | Qwen3.6-27B fused decoder: tenth-review fixes |
| `306c4991604` | Qwen3.6-27B fused decoder: eleventh-review fixes |
| `60178c2fb06` | Qwen3.6-27B fused decoder: twelfth-review fixes |
| `1f665ee911a` | Qwen3.6-27B fused decoder: thirteenth-review fixes |
| `cd76dcbde6d` | Qwen3.6-27B fused decoder: fourteenth-review fixes |
| `9abd74847de` | Qwen3.6-27B fused decoder: fifteenth-review fixes |
| `7e934a81872` | Qwen3.6-27B fused decoder: sixteenth-review fixes |
| `0847bcb41b1` | Qwen3.6-27B fused decoder: seventeenth-review fixes |
| `6fb4a7b8f49` | Qwen3.6-27B fused decoder: eighteenth-review fixes |
| `a3d205bf519` | Qwen3.6-27B fused decoder: nineteenth-review fixes |
| `c36b7d6ac85` | Qwen3.6-27B fused decoder: twentieth-review fixes |
| `782fb23fc88` | Qwen3.6-27B fused decoder: twenty-first-review fixes |
| `c9a93716645` | Qwen3.6-27B fused decoder: twenty-second-review fixes |
| `781c0fc935a` | Qwen3.6-27B fused decoder: twenty-third-review fixes |

Unrelated dirty state in the worktree - `.agents/notes/gdn.md`, two
`.agents/prompts/model_bringup_multigoal/*.txt` and `scripts/check_agent_prompt_lengths.py` -
predates this stage and was deliberately left out of the checkpoint.
