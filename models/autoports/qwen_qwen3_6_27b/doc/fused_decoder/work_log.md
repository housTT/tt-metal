# Qwen3.6-27B — fused decoder work log

Narrative for the fusing stage: what the profile said, what was tried, what landed, what was
measured and thrown away. The index of the landed rewrites is in [`README.md`](README.md);
this file is the reasoning and the raw numbers behind each one.

Everything ran on **device 2** (a Blackhole chip of the intact p300c board), 1x1 mesh, from
`/home/ttuser/dev/qwen/rundir` with `ttenv.sh` sourced — `ttenv_profiler.sh` for the four
Tracy runs, never both at once, and never watcher alongside the profiler.

## 0. Starting point

`tt/functional_decoder.py` at commit `206df66` — correct, deliberately unoptimised: BF16, tile
layout, DRAM interleaved everywhere. Its own profile (re-measured in this stage as the
baseline, `baseline/tracy/`):

| run | device time | ops |
|---|---|---|
| `linear_attention` prefill 2048 | 150.23 ms | 805 |
| `linear_attention` traced decode | 3032.2 µs/token | 96 |
| `full_attention` prefill 2048 | 18.60 ms | 44 |
| `full_attention` traced decode | 2425.2 µs/token | 50 |

Re-aggregating `baseline/tracy/*/**_perf_report.csv` by op code (step 2 of the graph-fusing
skill — the full op tables are in those CSVs; this is the shape of the problem):

```
linear_attention/prefill  150.4 ms   MatmulDeviceOperation b={6144} 32x32x32   38.3 ms  25.5 %  (recursive triangular inverse, 6 calls)
                                     BinaryNgDeviceOperation                   21.0 ms  13.9 %  (195 elementwise)
                                     ReshapeViewDeviceOperation                12.4 ms   8.2 %  (5 head splits)
                                     MatmulDeviceOperation b={48} 64x128x128    9.5 ms   6.3 %  (per-chunk recurrence loop)
                                     MatmulDeviceOperation b={3072} 32x32x32    6.4 ms   4.3 %  (triangular inverse, level 2)
                                     SliceDeviceOperation                       5.8 ms   3.8 %  (278)
linear_attention/decode   3032 µs    MLP gate_up 32x5120x34816                  866 µs  28.6 %
                                     MLP down 32x17408x5120                     433 µs  14.3 %
                                     in_proj_qkv 32x5120x10240                  261 µs   8.6 %
                                     LayerNormDeviceOperation (3/iter)          209 µs   6.9 %
                                     BinaryNgDeviceOperation (24/iter)          203 µs   6.7 %
full_attention/prefill    18.6 ms    5 matmuls at 80-83 % of DRAM roofline     13.5 ms  73 %
                                     RoPE slice/neg/concat/mul/add              2.9 ms  15 %
full_attention/decode     2425 µs    MLP gate_up + down                        1300 µs  54 %
                                     SdpaDecodeDeviceOperation                  258 µs  10.7 %
                                     LayerNormDeviceOperation (4/iter)          216 µs   8.9 %
                                     wqkv + wgate + o_proj                      507 µs  21 %
```

Two things stand out immediately and shaped the whole stage:

1. `linear_attention` prefill is **not** FLOP-bound. Half of it is a triangular inverse made of
   thousands of one-tile matmuls, and another fifth is layout churn.
2. `full_attention` is already at the DRAM roofline in both phases. Its matmuls are 73 % of
   prefill and 75 % of decode and no graph rewrite can move them; the only headroom is the
   op count around them.

## 1. Exploring the op library first (skill step 1)

Everything the tt-metal tree offers for these subgraphs, before writing any code:

* `ttnn.swiglu` / `glu` / `geglu` / `reglu` — the MLP.
* `ttnn.experimental.rotary_embedding_hf` — HF-style rotate-half RoPE, prefill and decode.
  Also `rotary_embedding`, `rotary_embedding_llama`, `rotary_embedding_llama_fused_qk`.
* `ttnn.transformer.gated_delta_attn_seq` — a **dedicated chunked gated-delta-rule kernel**,
  exactly this model's linear-attention mixer.
* `ttnn.experimental.paged_fused_update_cache` — K and V cache update in one dispatch.
* `ttnn.experimental.nlp_create_qkv_heads{,_decode}`, `nlp_concat_heads{,_decode}` — already
  used by the functional decoder.
* `ttnn.conv1d` (depthwise via `groups`), `ttnn.experimental.ssm.*` (Mamba-specific), 
  `ttnn.addcmul`, `ttnn.rms_norm` with a sharded program config.

Existing implementations read for idiom: `models/demos/blackhole/qwen36/tt/` (a Qwen3.5/3.6
port in the installed tree) and `models/experimental/gated_attention_gated_deltanet/tt/`,
which is what drives `gated_delta_attn_seq`; `models/tt_transformers/tt/{attention,rope}.py`
for the decode sharding idioms.

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

and give the pass-through channels `cos = 1`, `sin = 0`. Then a full-width rotate-half is
exactly Qwen3.5's partial RoPE. The permutation is applied host-side, at load time, to the
rows of the `q`/`k` projection weights and to the `q_norm`/`k_norm` weights, so nothing moves
at runtime.

Why it is safe: RMSNorm over the head is permutation-invariant in its reduction and per-channel
in its weight, so permuting both input and weight permutes the output; and `q·k` is invariant
under a permutation applied to **both**. `v`, the output gate and `o_proj` never see it.

Verified model-free first (`probes/probe_fused_ops.py`, `logs/probe_fused_ops.log`):

```
prefill permuted rotary_embedding_hf: pcc=0.999998
```

Decode mode additionally requires the input **and** `cos`/`sin` height-sharded one user per
core; the first attempt with a `(1, head_dim)` shard shape failed
(`tensor_layout.cpp:162: !shard_align_error.has_value()`) — the shard height must be a whole
tile, i.e. `(32, head_dim)`, which is what `models/tt_transformers/tt/rope.py` uses. With that
fixed the decode op is exact at every batch tried:

```
batch=1  overall=0.999998  min_user=0.999998  bad=[]
batch=4  overall=0.999998  min_user=0.999998  bad=[]
batch=8  overall=0.999998  min_user=0.999997  bad=[]
batch=16 overall=0.999998  min_user=0.999998  bad=[]
batch=32 overall=0.999998  min_user=0.999997  bad=[]
```
(`probes/probe_candidates.py ropebatch`, `logs/probe_ropebatch.log`)

Cost: `full_attention` prefill loses 10 ops per tensor (20 per pass) and 2.9 ms; decode loses
20 ops. The visible consequence is that `rot_mats` are now `head_dim`-wide and the paged K
cache holds permuted channels — both documented in the module docstring and in
`README.md`, and the layer asserts the `rot_mats` width so a functional-shaped call fails
loudly.

## 3. F6/F7 — the triangular inverse: L1, not fewer ops

`_unit_tri_inverse` was 47.9 ms of the 150 ms baseline (32 %), and the profile said its
batched `32x32x32` matmuls ran on **one core**. The obvious reading — "the op does not
parallelise over batch" — is wrong. Timing the same matmul from different memory
(`probes/probe_fused_ops.py`, `probes/probe_fused_ops2.py`; `logs/probe_fused_ops.log`, `logs/probe_fused_ops2.log`):

```
matmul [768, 1, 32, 32] DRAM   0.813 ms/call        -> 1.06 µs per batch element
matmul [768, 32, 32]    DRAM   0.806 ms/call
matmul [6144, 1, 32, 32] DRAM  6.408 ms/call        -> 1.04 µs per batch element (perfectly linear)
matmul [768, 32, 32]    L1     0.033 ms/call        -> 0.043 µs per batch element  (24x faster)
matmul [3072, 32, 32]   DRAM   3.209 ms/call
```

The cost is per-batch-element DRAM latency, not compute and not core count: `[768,32,32]` in
L1 is 24x faster than the same shape in DRAM, and 4x the batch costs exactly 4x the time
either way. So the fix is residency, not restructuring.

L1 is finite, though: a first attempt that put the *whole* gated-delta-rule working set in L1
died with

```
Out of Memory: Not enough space to allocate 25165824 B L1 buffer across 110 banks,
where each bank needs to store 229376 B, but bank size is 1461504 B
```

110 banks x 1.43 MB = 157 MB total. The landed version therefore chooses per region:

* `_unit_tri_inverse` picks a memory config **per recursion level** from that level's own
  tensor size (`_tri_mem`, budget 5 x one block against `L1_BUDGET_BYTES` = 96 MB), so the
  base case — thousands of `32x32x32` matmuls — is in L1 while the wider top level, whose few
  matmuls are large, stays in DRAM;
* the per-chunk recurrence loop slices one chunk at a time (~1.5 MB each) and runs in L1;
* the full-length tensors (decay mask, `kk`, `k_beta`, the projections) stay in DRAM, where
  one big streaming pass costs nothing extra.

`TRI_INV_BASE` went 16 → 32 at the same time (**F7**): the recursion stores its blocks in
32x32 tiles, so a base of 16 wastes three quarters of every tile, and 32 is one level
shallower. The functional stage had already measured identical layer PCC at bases 8/16/32 and
recurrent-state PCC 0.999986 (32) vs 0.999991 (16), both far above the bar; this stage's
end-to-end PCC confirms it (`linear_attention` prefill 0.999940, unchanged to six decimals).

Result: the `b={6144}` and `b={3072}` `32x32x32` matmuls, 44.7 ms of the baseline's top ops,
do not appear in the fused profile's top ops at all.

## 4. F15 — the decode norms were running on one core

The fused `full_attention` decode profile put `LayerNormDeviceOperation` at 9 % with four
calls per step. Reading the per-op rows rather than the aggregate
(`tracy/full_attention/decode_perf_report.txt`) showed why:

```
168  LayerNormDeviceOperation   102 µs   Cores 1     <- input_layernorm   [1,1,32,5120]
177  LayerNormDeviceOperation     6 µs   Cores 1     <- q_norm            [1,B,24,256]
178  LayerNormDeviceOperation     6 µs   Cores 1     <- k_norm            [1,B,4,256]
194  LayerNormDeviceOperation   102 µs   Cores 1     <- post_attention_layernorm
```

The interleaved layernorm kernel parallelises over tile **rows**, and a decode activation has
exactly one row of tiles, so the two full-width norms ran single-core: 204 µs of a 2425 µs
step, in both layer kinds. Width-sharding fixes it. Sweeping core grids and `subblock_w` on
the real shape (`logs/probe_norm.log`):

```
interleaved (baseline)                 103.8 µs/call, 1 core
sharded 5x2  block_w=16 subblock_w=4     24.6 µs/call   max|err| 3.1e-2 (one bf16 ULP at these magnitudes)
sharded 5x2  block_w=16 subblock_w=2     23.7 µs/call
sharded 8x4  block_w=5  subblock_w=1     30.8 µs/call
sharded 8x5  block_w=4  subblock_w=1     31.9 µs/call
sharded 10x4 block_w=4  subblock_w=1     32.0 µs/call
grid 8x8                                 skipped: 160 tiles not divisible by 64
```

`_decode_norm_config` picks the first grid in `_DECODE_NORM_GRIDS` whose core count divides the
hidden size in tiles (5x2 for 5120) and falls back to the interleaved kernel when none does.
The two resharding dispatches it adds cost ~1 µs each.

## 5. F1/F3/F16 and the op merges

**F1 — `ttnn.swiglu`.** Probed for semantics before use, because the two halves are not
symmetric (`logs/probe_fused_ops.log`):

```
swiglu vs first*silu(second): pcc=0.999993
swiglu vs silu(first)*second: pcc=0.668415
```

So it is `x[:n] * silu(x[n:])`, and the gate/up weights are concatenated **up first** — the
opposite of the functional decoder's order. One gotcha: `ttnn.swiglu` reports the *tile-padded*
height as its logical height, so a decode pass with batch < 32 comes back 32 rows tall and
would broadcast against the residual. `_mlp` trims it; prefill chunks are always a multiple of
a tile, so the trim only ever fires on decode.

**F3 — L2 norm as an `rms_norm`.** `rms_norm(x, eps') = x·√D / √(Σx² + D·eps')`, so with
`eps' = 1e-6/D` it *is* HF's `l2norm(x, eps=1e-6)` up to the constant `√D`. That constant, and
the `1/√head_k_dim` query scale, fold into the norm weight (`1/D` for `q`, `1/√D` for `k`).
Five ops become one, twice per pass. Visible in the decode profile as BinaryNg 1624 → 728 µs.

**F16 — `ttnn.addcmul` for the causal conv.** The depthwise conv is a 4-tap FIR: four
`slice → multiply` plus three `add`. `addcmul(acc, tap, w)` merges the multiply and the add.
Measured on the real shape, `[1, 1, 2051, 10240]` fp32 (`logs/probe_conv.log`):

```
multiply + add : 15.38 ms   max|err| 3.0e-7
addcmul        : 13.73 ms   max|err| 2.4e-7
```

1.65 ms and three fewer 84 MB temporaries.

**F9/F10/F11** fold the output gate's `sigmoid`, `z`'s `silu` and `dt_bias`'s add into the
matmul that was already running (`ttnn.linear(activation=...)`, `ttnn.linear(bias=...)`).
**F13** replaces `tril → exp → tril` on the decay mask with a constant strictly-upper `-1e9`
added before a single `exp`. **F14** folds the `neg` and the `tril(-1)` of `attn0` into a
multiply by a precomputed `-tril(ones, -1)` mask. **F5** merges the `b` and `a` projections,
which share their LHS, into one matmul (the `b` half padded up to a tile so both output slices
start on a tile boundary). **F8** reads the decode conv taps straight out of the state buffer
instead of rebuilding the window with a slice and a concat.

## 6. Rejected, with the measurement

### 6.1 `ttnn.transformer.gated_delta_attn_seq` — the dedicated kernel is slower *and* less accurate

This was the most promising candidate on paper: a hand-written kernel for exactly this
model's chunked gated delta rule, with a validated Python wrapper in the installed tree
(`models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_seq.py`). It would
replace the 32-iteration Python recurrence loop, the triangular inverse and most of the
elementwise work.

Probed against HF's `torch_chunk_gated_delta_rule` at the real head geometry
(48 v-heads, `head_k_dim` = `head_v_dim` = 128, which is the only geometry the kernel accepts —
it hard-requires `chunk_size == key_dim == val_dim == 128`,
`gated_delta_attn_seq_device_operation.cpp:48`):

```
seq=256   out pcc=0.984615   state pcc=0.987376
seq=2048  out pcc=0.987237   state pcc=0.987776
seq=2048  whole chunked delta rule wall = 167.9 ms
```

Both halves fail. **Accuracy**: 0.987 against a 0.995 bar, because the wrapper's diagonal-block
inverse is a Neumann series plus two Newton–Schulz steps at HiFi2, where the functional
decoder's recursive block inversion reaches ~1e-3 absolute error. **Speed**: 167.9 ms for the
delta rule *alone* versus 150 ms for the entire baseline layer, and 78.7 ms for the entire
fused layer. Rejected on both counts; the alternative — making the kernel's `L_inv` as accurate
as the recursion — would only make it slower still.

### 6.2 Shared-LHS `wqkv` + `wgate`

The classic ≥2-matmuls-sharing-an-LHS rewrite. Measured on the real shapes (`logs/probe_sharedlhs.log`):

```
rows=32    two matmuls  364.2 µs    one 5120x14336 matmul + 2 slices  373.3 µs
rows=2048  two matmuls 3510.0 µs    one 5120x14336 matmul + 2 slices 5182.1 µs
```

Slower in both phases. The two matmuls already run at 82–83 % of DRAM roofline, so merging
them buys one dispatch and pays for two slices and a worse-shaped matmul. Rejected.

### 6.3 `ttnn.experimental.paged_fused_update_cache`

Would merge the two decode `paged_update_cache` dispatches into one. It requires its two
inputs on **disjoint** core ranges:

```
TT_FATAL: input_tensor1 ({[0-0 - 0-0]}) and input_tensor2 ({[0-0 - 0-0]}) must not overlap
          (paged_fused_update_cache_device_operation.cpp:227)
```

`nlp_create_qkv_heads_decode` puts K and V on the same batch cores, so using it needs a
reshard of V onto a second core range — exactly the dispatch the fusion would save. Rejected.

### 6.4 Keeping `q` sharded through the decode SDPA

Tempting: `rotary_embedding_hf` decode emits a height-sharded `q`, and feeding it straight to
`paged_scaled_dot_product_attention_decode` would drop a reshard. It is **wrong**, and it is
the one bug this stage produced that PCC at batch 1 would never have caught. Per-user PCC at
batch 32:

```
users 0-7   0.9995 0.9995 0.9996 0.9995 0.9995 0.9995 0.9994 0.9994
users 8-31  0.0116 0.0185 0.0008 0.0384 0.0206 0.0364 0.0493 ... -0.0007
```

The boundary at 8 is the width of the `CoreCoord(8, 8)` core range the shard was built on,
while this Blackhole part's compute grid is **11x10** — the decode SDPA kernel derives each
user's core from the real grid width, so it reads users 8+ off the wrong core. Rebuilding the
shard on the device grid instead fixes SDPA but breaks `nlp_concat_heads_decode`
(`RuntimeError: bad optional access`), and asking the rotary op for an interleaved output
fails in `circular_buffer_config.cpp:222`. The landed code therefore reshards `q` to DRAM
before the SDPA, which is what the functional decoder did. `test_batched_users[32-*]` is what
caught this, and it is why the fused suite re-runs the whole functional suite rather than a
subset.  Bisect log: `logs/bisect_sharded_q_decode_batch32.log`.

### 6.5 A sharded SDPA-decode output

Would feed `nlp_concat_heads_decode` directly. Rejected by the op:
`TT_FATAL: Sharded output not supported for GQA (sdpa_decode_device_operation.cpp:405)`.

### 6.6 `ttnn.conv1d` for the causal conv

The depthwise `groups = conv_dim` conv1d is the dedicated op for the 4-tap FIR. At this
model's width it cannot be built:

```
TT_FATAL @ ttnn/cpp/ttnn/operations/sliding_window/op_slicing/op_slicing.cpp:266: found_valid_config
```

(1 batch, length 2051, 10240 channels, kernel 4, groups 10240). Rejected — no valid slicing
configuration.

### 6.7 `ttnn.matmul(transpose_a=/transpose_b=True)`

Kept in the code because it is clearer, but it is **not** a fusion on this build: the transpose
count is 75 before and 75 after, i.e. ttnn lowers the flag to a separate
`TransposeDeviceOperation`. Recorded so nobody re-counts it as a win.

## 7. Measured but deliberately not taken: a bfloat16 causal conv

The causal conv is 13.7 ms of the fused `linear_attention` prefill's 78.7 ms — the largest
remaining single block. Almost all of it is bandwidth on a float32 `[1, 1, 2051, 10240]`
tensor (84 MB per tap). In bfloat16 (`logs/probe_conv.log`):

```
FIR fp32 multiply+add: 15.38 ms   max|err| 3.0e-7
FIR fp32 addcmul     : 13.73 ms   max|err| 2.4e-7     <- landed
FIR bf16 multiply+add:  4.68 ms   max|err| 1.7e-2
FIR bf16 addcmul     :  4.11 ms   max|err| 1.8e-2
```

A further ~9.6 ms, about 12 % of the layer. It is **not** taken here because it is a precision
trade, not a graph rewrite: `mixed_qkv` is float32 by a deliberate decision of the functional
stage ("the recurrence and the softplus/exp gating are numerically sensitive"), the conv state
is compared against HF's cache at 0.999995, and changing compute dtype is the optimization
stage's job, not the fusing stage's. The numbers are recorded here so that stage does not have
to re-derive them.

## 8. What is left in `linear_attention` prefill, and why

After fusing, 78.7 ms:

```
BinaryNgDeviceOperation                17.2 ms  21.4 %  187 ops
ReshapeViewDeviceOperation             12.3 ms  15.3 %    5 ops
MatmulDeviceOperation 2048x5120x34816   6.1 ms   7.6 %  (MLP gate|up, at roofline)
MatmulDeviceOperation b={1536} ...      5.7 ms   7.1 %  (inv @ v_beta, inv @ k_beta_decay)
SliceDeviceOperation                    4.8 ms   5.9 %  275 ops
TransposeDeviceOperation                4.1 ms   5.1 %   75 ops
```

The 12.3 ms of `ReshapeView` is five head splits — `[1, 1, L, W] → [1, L, H, D]`, a last-dim
change that TTNN implements as untilize + retilize. The dedicated op for that pattern is
`nlp_create_qkv_heads`, and it does **not** apply here: it requires K and V to have the same
head count, while this mixer has 16 key heads and 48 value heads with the value heads three
times as many. `create_qkv_heads_from_separate_tensors` has the same constraint. Folding the
3x key-head repeat into the projection weight instead would make `in_proj_qkv` 80 % wider
(10240 → 18432 columns), costing ~1.8 ms of matmul in prefill and ~208 µs in decode against
~204 µs of layout saved in decode — a wash at best, a regression for decode. Left alone, and
recorded as the next thing to look at if a suitable head-split op appears.

The `BinaryNg` total is the elementwise cost of the delta rule itself (`k_beta`, `v_beta`,
`k_decayed`, `q_decayed`, the decay mask, the per-chunk loop's five binaries x 32 chunks). It
is bandwidth on 25–50 MB float32 tensors, and F13/F14/F16 already removed the ones that were
pure bookkeeping.

## 9. Verification and evidence

Commands, all from `/home/ttuser/dev/qwen/rundir` with `ttenv.sh` sourced:

```bash
# full fused suite - 62 passed, 2 skipped (the long-context pair), 473.95 s
python -m pytest $REPO/models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py -v -s
#   -> logs/suite_fused_final.log

# full advertised context, 262143 tokens
python -m pytest $REPO/models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py \
    -k test_full_advertised_context --long-context -v -s
#   -> logs/long_context.log   (1 passed, 1 failed - the inherited SDPA-decode gap)

# watcher, separate run, profiler off
export TT_METAL_LOGS_PATH=$ART/watcher TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=0 \
       TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1
python -m pytest $REPO/models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py \
  -k "test_traced_decode_pcc or (test_decode_pcc and 2049) or test_bfloat8_kv_cache or test_fused_ops_are_used" -v -s
#   -> logs/watcher_run.log, watcher/WATCHER_AUDIT.md  (7 passed, log clean)

# perf, one at a time, profiler build
cd $REPO/models/autoports/qwen_qwen3_6_27b/doc/fused_decoder
./probes/run_perf.sh <kind> <phase> fused
./probes/run_perf.sh <kind> <phase> functional "$PWD/baseline"

# evidence
python -m models.autoports.qwen_qwen3_6_27b.scripts.collect_evidence \
    $ART/logs/*.log --out $ART/pcc_evidence.json
#   -> 278 records, 268 numeric, exactly one under the bar (the inherited gap)
```

Record-by-record comparison against the functional stage's `pcc_evidence.json`: **262 shared
measurements, none worse by more than 1e-4**, mean change +7.5e-6, worst −9.2e-5, best
+9.8e-4. Per-class minima are in [`README.md`](README.md).

## 10. Hardware notes

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

No reset, no hang and no `tt-triage` capture was needed at any point in this stage: every
failure hit was a deterministic op-validation error or a wrong-answer bug, all reproduced and
fixed from ordinary logs.
