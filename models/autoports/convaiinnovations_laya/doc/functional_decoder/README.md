# Stage 1: functional decoder (ModernBERT-large encoder plus the Laya head on one Blackhole p150)

Plugin stage "functional-decoder" mapped to Laya (PLAN.md section 5, row 1): functional TTNN blocks with the real
`convaiinnovations/laya` weights and real typed-decisions inputs on chip 0, each compared with an fp32 torch reference,
paired with negative controls, one watcher-clean run and one Tracy profile. Device: p150 (Blackhole, 11x10 worker grid,
`l1_small_size` 79104). Software: tt-metal `/home/hous/dev/ornith-1.5-9b/tt-metal` (v0.79.0-dev base), transformers 5.12.1.

## What was built (files under `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/convaiinnovations_laya/`)

| file | role |
|---|---|
| `tt/model_config.py` | `PrecisionPolicy` (Appendix A.7 table, six policies, default `bf8w_hifi3`), `PortConfig` (placement and program-config levers), Blackhole program-config rules, bucket helpers |
| `tt/weights.py` | safetensors load, key map (170 encoder tensors, head, scorer, `type_emb`, `act_head`, `temperature`), Q-scale fold, padded GeGLU blocks, device upload with optional mesh mapper |
| `tt/modernbert_masks.py` | runtime padding-mask builder (Appendix A.3) plus the host builder used by the tests |
| `tt/modernbert_rope.py` | cos/sin caches per layer type and sequence length, height-sharded round trip with a per-core byte bound |
| `tt/modernbert_attention.py` | Wqkv linear, head split, rotary, SDPA with the baked mask, head concat, Wo linear; optional biases (head layers) and `minimal_matmul` alternative |
| `tt/modernbert_mlp.py` | GeGLU, block-sharded plan or interleaved |
| `tt/modernbert_layer.py`, `tt/modernbert_embeddings.py`, `tt/modernbert_model.py` | pre-norm layer (resident L1 residual stream when the plan is sharded), embeddings, 28-layer encoder with a per-layer hook |
| `tt/laya_head.py` | `type_emb` add, two biased pre-LN head layers (ReLU fused into `linear1`), scorer with an fp32 final linear, CLS rows |
| `tt/laya_model.py` | `TtnnLayaModel`: weights once, one sub-model per (rows, seq) bucket, preallocated inputs, `device_forward`, eager `forward` |
| `tests/laya_inputs.py`, `tests/head_reference.py`, `tests/conftest.py` | real inputs from typed-decisions prompts through the vendored `build_sequence` and `collate_items`; explicit-math head reference; shared fixtures (6 torch threads) |
| `tests/test_model_config.py`, `tests/test_weights.py` | no device: 54 tests |
| `tests/test_ttnn_{embeddings,rope,masks,mlp,attention,layer,encoder,head}.py` | device tests, section "Results" |
| `tests/profile_layer.py` | one layer (optionally plus one head layer) under Tracy |

## Generalisation from the upstream base port (Appendix A.2), what changed and why

- Config comes from `encoder/config.json` through `transformers.AutoConfig` (`ModernBertConfig`): hidden 1024, 28 layers,
  16 heads, intermediate 2624, `layer_types` full on 0, 3, ..., 27 (10 full, 18 sliding), `local_attention` 128 (band
  +/-64), `rope_parameters` 160000 (full) and 10000 (sliding), `norm_eps` 1e-5, no biases, pad 50283. The checkpoint
  holds 206 tensors in float16 (PLAN.md section 1 says bfloat16; the file dtype is F16), 170 of them `encoder.*`.
- Compute configs: `ttnn.init_device_compute_kernel_config(device.arch(), ...)` driven by `PrecisionPolicy`
  (`math_approx_mode=False`, `fp32_dest_acc_en=True`, `packer_l1_acc=True`, `dst_full_sync_en=False`). Groups: `attn`
  (Wqkv, Wo, SDPA), `mlp` (Wi, Wo), `head`, `scorer`, `norm` (HiFi4, fp32 accumulation, every LayerNorm). The dest
  register cap is derived (`dest_tiles` = 16 / 2 without full sync / 2 with fp32 accumulation = 4) and every
  `out_subblock_h x out_subblock_w` obeys it.
- Grid policy: `select_down_projection_grid` accepts 11x10 and returns the tuned 8x8 for attn Wo, mlp Wo (interleaved
  path), the head linears and the scorer dense; `PortConfig.down_grid="minimal_11x10"` and `qkv_mode="minimal_11x10"`
  switch Wo and Wqkv to `ttnn.experimental.minimal_matmul` with `MinimalMatmulConfig(8,8,8, CoreCoord(11,10))` (stage 3 A/B).
- Wqkv program config: a rule, not a table. `in0_block_w` = 8 (32 K tiles divisible by 8), grid 8x8,
  `per_core_M = rows/256`, `per_core_N = 12`, `out_subblock` 1x4, and `out_block_h = largest divisor of per_core_M <= 8`
  so the per-core in0 and output blocks stay bounded at every row bucket (B 1 to 64 at S 512 and 1024 all get a config;
  the upstream table missed B >= 8 at S 512).
- GeGLU: default plan block-sharded on 8x8 with the intermediate zero-padded 2624 -> 2816 (88 tiles, `per_core_N` 11,
  `out_subblock_w` 1); 3072 (96 tiles, `per_core_N` 12, `out_subblock_w` 4) is the measured alternative; the interleaved
  path uses the unpadded 2624 weights and ttnn's automatic matmul choice (an 8x8 mcast config exists for padded widths).
  Both widths are uploaded when the sharded plan is active (`prepare_weights(intermediate_pads=(None, 2816))`) because
  one process serves buckets on both paths. Sharding window: at least 12 tiles per core (B >= 2 at S 512) and at most
  `shard_max_rows` 2048 rows (an estimate of the per-core L1 budget: at 4096 rows the three live intermediates of width
  2816 plus the hidden shards exceed 1.3 MiB per core); so B 2 and 4 at S 512 and B 1 and 2 at S 1024 run resident,
  B 1 and B >= 8 interleaved.
- Attention chain memory: L1 interleaved up to `l1_attention_max_rows` and DRAM above. The plan's 8192 was wrong on this
  box: at 4096 rows (B 8, S 512) the L1 buffers of the chain and the static circular buffers of the 8x8 matmul clash
  (`program.cpp:1932`, "L1 buffer allocated at 1030912 and static circular buffer region ends at 1049600"). The
  threshold shipped by this stage is 2048 rows (B <= 4 at S 512, B <= 2 at S 1024); stage 3 measures the alternative
  of a smaller `out_block_h` to keep L1 at 4096 rows.
- SDPA: `SDPAProgramConfig` on the full device grid, `q_chunk = k_chunk = 128` for rows < 768 and 256 above, for S in
  {256, 512, 768, 1024}; `exp_approx_mode=False`. `sliding_window_size` is never passed (this build rejects it together
  with `attn_mask`); the band lives in the mask.
- Rotary: `rotary_embedding_hf` with HF rotate-half caches built on host per (layer type, S); height-sharded 8xY round
  trip when rows >= 24576 and the per-core shard is at most 512 KiB (so B 8 and 16 at S 512 shard, B 64 does not).
- `TtnnModernBertModel(parameters, config, device, seq_len, batch_size, policy, port)`; `__call__(input_ids, masks,
  layer_hook=None, final_norm=True)`; the masks are an argument, not construction state.

Stage 3 moved the shipped `PortConfig` and policy defaults (`../optimized_decoder/README.md`); the stage 1 placement
above is kept as `model_config.STAGE1_PORT` and the stage 1 policy as `POLICIES["bf8w_hifi3"]`.

## Runtime padding-mask design (Appendix A.3), as built

`TtnnMaskBuilder(config, device, seq_len, batch_size)` holds two static DRAM tensors per bucket, `band` (B,1,S,S) with
0 inside |i-j| <= 64 and -1e30 outside, and `zeros` (B,1,S,S). The per-call device input is `pad_row` (B,1,1,S) bf16
TILE (0 real, -1e30 pad), preallocated with `allocate_tensor_on_device` and rewritten with `copy_host_to_device_tensor`.
`build(pad_row)` returns `{sliding: band + pad_row, full: zeros + pad_row}` through two `ttnn.add` broadcasts, both
(B,1,S,S) bf16 in DRAM, which is the only mask form this SDPA accepts (tiled, on device, DRAM, batch 1 or B, heads 1 or H).
`build_masks_host` is the host twin; `tests/test_ttnn_masks.py` asserts the device masks equal it for random lengths at
(1,512), (4,512) and (2,1024), that they are finite, DRAM and TILE, and that rewriting `pad_row` changes SDPA's output
on the padded row while leaving the unpadded row unchanged. Padding rows of a bucket (rows beyond the request) carry an
all-zero attention row; SDPA stays finite because the masked value is -1e30, not -inf.

## Head on device (Appendix A.4), as built

`TtnnLayaHead`: `ttnn.embedding(qtype (B,1) uint32, type_emb)` -> (B,1,1024) broadcast-added to the encoder output;
two `TtnnHeadLayer`s, each `x + out_proj(SDPA(split(in_proj(LN1(x))), full_mask))` and `x + linear2(relu(linear1(LN2(x))))`
with the 1/8 scale folded into the Q third of `in_proj_weight` and `in_proj_bias`, biases passed to `ttnn.linear`, ReLU
fused as `activation=UnaryWithParam(RELU)`, `ttnn.layer_norm(weight, bias)`, the encoder attention module reused with
`rotary=None`; scorer `LN(w,b) -> linear(1024,1024)+b with fused erf GELU -> typecast fp32 -> linear(1024,1)+b in fp32`
over every position; CLS rows by a tile-aligned `ttnn.slice(h, [0,0,0], [B,32,1024])`. Readbacks: logits (B,S,1) fp32
and CLS (B,32,1024) bf16, both through `ttnn.to_torch`. The host side (gather at marker positions, `-1e4` fill, softmax,
features, `act_head`, temperatures) belongs to stage 6 (`tt/engine.py`).

## Device API exposed to stage 6 and the server

```
from models.autoports.convaiinnovations_laya.tt.laya_model import TtnnLayaModel, open_device
from models.autoports.convaiinnovations_laya.tt.model_config import policy_from_name, DEFAULT_PORT
device = open_device(device_id=0, l1_small_size=79104, trace_region_size=512 * 1024 * 1024)
model = TtnnLayaModel(device, config, state_dict=sd, policy=policy_from_name("bf8w_hifi3"), port=DEFAULT_PORT,
                      row_buckets=(1, 2, 4, 8, 16, 32, 64), seq_buckets=(512,))
out = model.forward(input_ids, attention_mask, qtype)      # eager; dict(logits (n, L) fp32, cls (n, 1024) fp32, bucket, device_ms)
```
`input_ids` (n, L) long, `attention_mask` (n, L) long with 1 on real tokens and 0 on padding, `qtype` (n,) long in
{0 choice, 1 score, 2 noul}. The model pads n up to the row bucket (pad id, all-zero attention, qtype 0) and L up to the
seq bucket, and slices the outputs back to (n, L). Marker gather happens on host from `out["logits"]`.
`model.build_bucket(B, S)` preallocates a bucket; `model.device_forward(bucket)` is the traced body (stage 2,
`tt/runner.py: LayaTraceRunner`).

## How to run

```
source /home/hous/dev/laya/bin/ttenv.sh; export TT_METAL_VISIBLE_DEVICES=0; cd $TT_METAL_HOME
A=models/autoports/convaiinnovations_laya/tests
/home/hous/dev/laya/bin/devlock python -m pytest $A/test_model_config.py $A/test_weights.py -q            # no device
/home/hous/dev/laya/bin/devlock python -m pytest $A/test_ttnn_embeddings.py $A/test_ttnn_rope.py $A/test_ttnn_masks.py \
    $A/test_ttnn_mlp.py $A/test_ttnn_attention.py $A/test_ttnn_layer.py $A/test_ttnn_encoder.py $A/test_ttnn_head.py -q -rA
```
`LAYA_PCC_LOG=<file>` appends every measured PCC row as JSON; `test_ttnn_encoder.py` writes the per-layer trace to
`layer_pcc_<policy>_<port label>.json` here (`LAYA_POLICY` selects the policy, `LAYA_PORT=stage1` selects `STAGE1_PORT`,
`LAYA_PORT_OVERRIDES` adds `PortConfig` fields), so a later run with another policy or port cannot overwrite stage evidence.
The stage 1 file is `layer_pcc_bf8w_hifi3_stage1port.json`, regenerated after review R1 by one encoder test run
(`/home/hous/dev/laya/logs/p3_s1_regen_layer_pcc_20261005T230058Z.log`); the shipped configuration's trace is
`../optimized_decoder/layer_pcc_shipped.json`.

## Results (chip 0, real weights, real typed-decisions inputs; every number from `pcc_rows.json`, each row names its source log)

Inputs: `tests/laya_inputs.build_inputs` takes the first questions of the typed-decisions test parquet through the
vendored `build_sequence` (max_len 512, head_max_len 192) and `collate_items`, pads to the bucket, and for batches above
one extends the last row with state text to 512 real tokens ("fill" row) so every batch has one unpadded row. Row 0 is
always the same 165-token `agent_trace_observability_000000/action` question. Real positions only, unless stated.

### Components versus fp32 (gate 0.999, floor 0.995)

| component | shape | PCC | note |
|---|---|---|---|
| embeddings | B 1, S 512 | 0.999995 | max abs channel 13.2 |
| embeddings | B 8, S 512 | 0.999995 | max abs channel 13.2 |
| rope full_attention | B 1, S 512, interleaved | q 0.999995, k 0.999995 | real Q, K of layer 0 |
| rope sliding_attention | B 1, S 512, interleaved | q 0.999994, k 0.999994 | real Q, K of layer 0 |
| rope full_attention | B 8, S 512, sharded | q 0.999995, k 0.999995 | real Q, K of layer 0 |
| rope sliding_attention | B 8, S 512, sharded | q 0.999994, k 0.999994 | real Q, K of layer 0 |
| rope full_attention | B 1, S 1024, interleaved | q 0.999995, k 0.999995 | real Q, K of layer 0 |
| rope sliding_attention | B 1, S 1024, interleaved | q 0.999994, k 0.999994 | real Q, K of layer 0 |
| rope full_attention | B 2, S 1024, sharded | q 0.999995, k 0.999995 | real Q, K of layer 0 |
| rope sliding_attention | B 2, S 1024, sharded | q 0.999994, k 0.999994 | real Q, K of layer 0 |
| GeGLU layer 0 | B 1, S 512, interleaved 2624 | 0.999648 | max abs err 0.807, input max abs 5.6 |
| GeGLU layer 16 | B 1, S 512, interleaved 2624 | 0.999568 | max abs err 1.559, input max abs 12.7 |
| GeGLU layer 0 | B 2, S 512, sharded 2816 | 0.999660 | max abs err 0.682, input max abs 5.6 |
| GeGLU layer 16 | B 2, S 512, sharded 2816 | 0.999470 | max abs err 1.568, input max abs 12.7 |
| GeGLU layer 16 width variant interleaved_2624 | B 2, S 512 | 0.999471 | max abs err 1.568 |
| GeGLU layer 16 width variant sharded_2816 | B 2, S 512 | 0.999470 | max abs err 1.568 |
| GeGLU layer 16 width variant sharded_3072 | B 2, S 512 | 0.999470 | max abs err 1.568 |
| attention full_attention (layer 0) | B 2, S 512, lengths [165, 512] | 0.999856 real positions | 0.999863 all positions |
| attention sliding_attention (layer 1) | B 2, S 512, lengths [165, 512] | 0.999845 real positions | 0.999885 all positions |
| layer 0 (full_attention) | B 1, S 512, interleaved | 0.999813 | max abs err 0.74, output max abs 47 |
| layer 1 (sliding_attention) | B 1, S 512, interleaved | 0.999883 | max abs err 0.64, output max abs 86 |
| layer 16 (sliding_attention) | B 1, S 512, interleaved | 0.999998 | max abs err 23.31, output max abs 10819 |
| layer 27 (full_attention) | B 1, S 512, interleaved | 0.999999 | max abs err 47.33, output max abs 27165 |
| layer 0 (full_attention) | B 2, S 512, resident L1 | 0.999793 | max abs err 0.75, output max abs 48 |
| layer 1 (sliding_attention) | B 2, S 512, resident L1 | 0.999858 | max abs err 0.64, output max abs 86 |
| layer 16 (sliding_attention) | B 2, S 512, resident L1 | 0.999996 | max abs err 23.31, output max abs 10819 |
| layer 27 (full_attention) | B 2, S 512, resident L1 | 0.999998 | max abs err 60.49, output max abs 31078 |
| head layers (two) | B 2, S 512 | 0.999996 raw, 0.999996 standardized | max abs err 8.74 |
| scorer, policy bf8w_hifi3 | B 2, S 512, fp32 input | 0.999988 | max abs err 0.0062, marker max abs err 0.0041 |
| scorer, policy bf8w_hifi3_head_bf16 | B 2, S 512, fp32 input | 0.999988 | max abs err 0.0062, marker max abs err 0.0026 |

### Encoder, 28 layers versus fp32 (gate 0.99)

| shape | variant | PCC after final norm | per row | worst layer | max abs err |
|---|---|---|---|---|---|
| B 1, S 512 | default | 0.997114 | 0.9971 | 13 at 0.9980 | 2.14 |
| B 1, S 512 | fill_only | 0.992017 | 0.9920 | 20 at 0.9665 | 26.35 |
| B 2, S 512 | default | 0.994863 | 0.9973, 0.9940 | 20 at 0.9805 | 22.02 |
| B 2, S 512 | interleaved | 0.995009 | 0.9974, 0.9942 | 20 at 0.9832 | 22.08 |
| B 4, S 512 | default | 0.992006 | 0.9973, 0.9899, 0.9948, 0.9897 | 20 at 0.9610 | 28.22 |
| B 8, S 512 | default | 0.992817 | 0.9974, 0.9900, 0.9945, 0.9972, 0.9963, 0.9973, 0.9896, 0.9878 | 19 at 0.9838 | 29.74 |
| B 1, S 1024 | default | 0.997302 | 0.9973 | 13 at 0.9974 | 2.06 |
| B 1, S 512 | vs bf16 CPU reference (informational) | 0.983844 | | | |

### Per-layer PCC trace (stage 1 baseline policy `bf8w_hifi3` and `STAGE1_PORT`, from `layer_pcc_bf8w_hifi3_stage1port.json`)

Residual stream after each layer against the fp32 reference, real positions only. F marks a full-attention layer, S a
sliding one. "max abs" is the largest element error of the B 1 run and "ref max abs channel" the reference's largest
activation after that layer.

| layer (type) | B 1 S 512 PCC | max abs | ref max abs channel | B 1 fill row PCC | B 8 S 512 PCC | B 1 S 1024 PCC |
|---|---|---|---|---|---|---|
| 0 (F) | 0.9998 | 0.7 | 47 | 0.9998 | 0.9998 | 0.9998 |
| 1 (S) | 0.9997 | 1.9 | 86 | 0.9996 | 0.9997 | 0.9997 |
| 2 (S) | 0.9996 | 2.1 | 85 | 0.9995 | 0.9996 | 0.9996 |
| 3 (F) | 0.9995 | 2.2 | 78 | 0.9993 | 0.9994 | 0.9995 |
| 4 (S) | 0.9994 | 2.1 | 77 | 0.9993 | 0.9993 | 0.9994 |
| 5 (S) | 0.9997 | 40.9 | 1527 | 0.9991 | 0.9996 | 0.9997 |
| 6 (F) | 0.9998 | 45.5 | 1650 | 0.9995 | 0.9992 | 0.9998 |
| 7 (S) | 0.9989 | 189.2 | 2484 | 0.9994 | 0.9989 | 0.9985 |
| 8 (S) | 0.9987 | 215.7 | 2595 | 0.9992 | 0.9989 | 0.9982 |
| 9 (F) | 0.9987 | 224.2 | 2656 | 0.9991 | 0.9990 | 0.9982 |
| 10 (S) | 0.9986 | 237.4 | 2720 | 0.9990 | 0.9989 | 0.9981 |
| 11 (S) | 0.9985 | 241.2 | 2711 | 0.9990 | 0.9989 | 0.9980 |
| 12 (F) | 0.9981 | 242.2 | 2696 | 0.9989 | 0.9959 | 0.9975 |
| 13 (S) | 0.9980 | 244.0 | 2693 | 0.9988 | 0.9958 | 0.9974 |
| 14 (S) | 0.9990 | 659.8 | 10820 | 0.9968 | 0.9976 | 0.9990 |
| 15 (F) | 0.9990 | 661.9 | 10823 | 0.9968 | 0.9976 | 0.9990 |
| 16 (S) | 0.9990 | 658.1 | 10819 | 0.9968 | 0.9976 | 0.9990 |
| 17 (S) | 0.9990 | 657.1 | 10818 | 0.9968 | 0.9976 | 0.9990 |
| 18 (F) | 0.9986 | 652.2 | 10814 | 0.9940 | 0.9970 | 0.9987 |
| 19 (S) | 0.9990 | 2850.6 | 27145 | 0.9666 | 0.9838 | 0.9996 |
| 20 (S) | 0.9990 | 2848.3 | 27152 | 0.9665 | 0.9838 | 0.9996 |
| 21 (F) | 0.9990 | 2857.3 | 27151 | 0.9666 | 0.9838 | 0.9996 |
| 22 (S) | 0.9990 | 2852.3 | 27159 | 0.9667 | 0.9839 | 0.9996 |
| 23 (S) | 0.9990 | 2856.3 | 27159 | 0.9668 | 0.9839 | 0.9996 |
| 24 (F) | 0.9990 | 2856.8 | 27161 | 0.9670 | 0.9840 | 0.9996 |
| 25 (S) | 0.9990 | 2855.1 | 27165 | 0.9670 | 0.9841 | 0.9996 |
| 26 (S) | 0.9990 | 2857.0 | 27167 | 0.9671 | 0.9841 | 0.9996 |
| final norm | 0.9971 | 2.14 | | 0.9920 | 0.9928 | 0.9973 |


### Negative controls (must fall below the gate)

| control | PCC | gate | detected |
|---|---|---|---|
| NC embeddings without layernorm | 0.796371 | 0.999 | yes |
| NC rope wrong theta | 0.530009 | 0.999 | yes |
| NC rope not applied | 0.654763 | 0.999 | yes |
| NC geglu gate swapped | 0.803305 | 0.999 | yes |
| NC sliding without band | 0.591428 | 0.999 | yes |
| NC band +/-32 | 0.978272 | 0.999 | yes |
| NC theta swapped | 0.885771 | 0.999 | yes |
| NC pad mask dropped (padded row) | 0.571780 | 0.999 | yes |
| NC Q/K permuted | 0.383288 | 0.999 | yes |
| NC norm applied at layer 0 | 0.947958 | 0.999 | yes |
| NC head layers skipped (reference intrinsic 0.904760) | 0.904848 | 0.99 | yes |
| NC wrong question type (reference intrinsic 0.760552) | 0.760217 | 0.99 | yes |
| band removed at S 64 (must stay above the gate) | 0.999740 | 0.999 | as expected |
| pad_row rewritten (padded row, SDPA output) | 0.572416 | 0.999 | yes |
| relu_to_gelu (recorded, not gated) | device 0.999921, reference intrinsic 0.999943, marker max abs 0.0136; device vs reference under the same control 0.999983 | | n/a |
| head_pad_mask_dropped (recorded, not gated) | device 0.999820, reference intrinsic 0.999844, marker max abs 0.0091; device vs reference under the same control 0.999982 | | n/a |

### End to end (TtnnLayaModel.forward versus reference encoder plus explicit-math head)

| rows | bucket | scorer logits PCC (real positions) | marker logits PCC | marker max abs err | CLS PCC | NaN |
|---|---|---|---|---|---|---|
| 1 | [1, 512] | 0.990241 | 0.994245 | 0.344 | 0.999961 | False |
| 8 | [8, 512] | 0.990675 | 0.992061 | 0.379 | 0.999571 | False |


### Gates of the stage table (PLAN.md section 5, row 1)

| gate | result |
|---|---|
| component PCC >= 0.999 (floor 0.995) versus fp32 | pass: embeddings 0.999995, rope 0.999994 to 0.999995 (both thetas, S 512 and 1024, interleaved and sharded), GeGLU 0.999470 to 0.999660 (layers 0 and 16, both paths), attention 0.999845 and 0.999856 on a padded batch, layers 0, 1, 16, 27 at B 1 and 2 0.999793 to 0.999999, head layers 0.999996, scorer 0.999988 |
| 28-layer encoder versus fp32 >= 0.99 | pass at every shape run: 0.9971 (1,512), 0.9949 (2,512), 0.9920 (4,512), 0.9928 (8,512), 0.9973 (1,1024); per-layer PCC and outliers in `layer_pcc_bf8w_hifi3_stage1port.json` |
| negative controls below threshold | pass for all ten encoder-side controls (GeGLU order 0.803, Q/K permuted 0.383, theta swapped 0.886 and 0.530, band removed 0.591, band +/-32 0.978, layer-0 norm 0.948, pad mask dropped 0.572, embeddings without norm 0.796, rope not applied 0.655). The plan's two head controls cannot be met by any implementation: on the fp32 reference itself ReLU->GELU moves the scorer logits to PCC 0.999943 (marker max abs 0.0136) and dropping the head pad mask to 0.999844 (0.0091); the device reproduces those moves (0.999921, 0.999820) and matches the reference under the same control at 0.99998. They are recorded, not gated; the gated head controls are "head layers skipped" (0.9048) and "wrong question type" (0.7602), both reproducing the reference's own intrinsic numbers |
| one watcher-clean run of the layer test | pass: `TT_METAL_WATCHER=10`, 10 passed in 37 s (`/home/hous/dev/laya/logs/p3_s1_watcher_20261005T212635Z.log`), `watcher_layer_run.log` has 8 periodic dumps and no error, assert, hang or stall lines |
| Tracy profile of one layer | pass: layer 1 (sliding, with attention norm) at B 8, S 512, `python -m tracy -r -p -v -o doc/functional_decoder/tracy/layer1_b8s512 tests/profile_layer.py --batch 8 --seq 512 --layer 1 --repeats 5` then `tt-perf-report` (`perf_report.csv`, `perf_report_stacked.csv/.png`, `perf_report.console.log`; the raw `.tracy` under `reports/` is kept locally, not committed). Host 1-minute load at start 7.09. Last of five passes: 20 device ops, 2751.5 us of device kernel time (host-timed pass with dispatch 2.93 ms). Table below; B 1 and B 64 profiles in `tracy/layer1_b1s512` and `tracy/layer1_b64s512` are analysed in `../optimized_decoder/README.md` |

### Per-op device profile of layer 1 at B 8, S 512 (DRAM attention chain, interleaved GeGLU)

| op | calls | device us | share | cores | fidelity, dtypes |
|---|---|---|---|---|---|
| MatmulDeviceOperation b={8} x 512 x 1024 x 2624 | 2 | 699.5 | 25.4 % | 88 | HiFi3 BF16 x BFP8 => BF16 BFLOAT16xBFLOAT8_B |
| SDPAOperation | 1 | 406.5 | 14.8 % | 110 | BF16, BF16 => BF16 BFLOAT16xBFLOAT16 |
| MatmulDeviceOperation b={8} x 512 x 1024 x 3072 | 1 | 301.1 | 10.9 % | 64 | HiFi3 BF16 x BFP8 => BF16 BFLOAT16xBFLOAT8_B |
| BinaryNgDeviceOperation | 3 | 265.6 | 9.7 % | 110 | BF16, BF16 => BF16 BFLOAT16xBFLOAT16 |
| MatmulDeviceOperation b={8} x 512 x 2624 x 1024 | 1 | 261.4 | 9.5 % | 64 | HiFi3 BF16 x BFP8 => BF16 BFLOAT16xBFLOAT8_B |
| RotaryEmbeddingHfDeviceOperation | 2 | 159.2 | 5.8 % | 64 | BF16, BF16 => BF16 BFLOAT16xBFLOAT16 |
| LayerNormDeviceOperation | 2 | 132.8 | 4.8 % | 110 | BF16, BF16 => BF16 BFLOAT16xBFLOAT16 |
| NlpCreateHeadsDeviceOperation | 1 | 130.7 | 4.8 % | 110 | BF16 => BF16 BFLOAT16xnan |
| MatmulDeviceOperation b={8} x 512 x 1024 x 1024 | 1 | 119.6 | 4.3 % | 64 | HiFi3 BF16 x BFP8 => BF16 BFLOAT16xBFLOAT8_B |
| UnaryDeviceOperation | 1 | 115.5 | 4.2 % | 110 | BF16 => BF16 BFLOAT16xnan |
| ShardedToInterleavedDeviceOperation | 2 | 56.8 | 2.1 % | 64 | BF16 => BF16 BFLOAT16xnan |
| NLPConcatHeadsDeviceOperation | 1 | 54.4 | 2.0 % | 110 | BF16 => BF16 BFLOAT16xnan |
| InterleavedToShardedDeviceOperation | 2 | 48.4 | 1.8 % | 64 | BF16 => BF16 BFLOAT16xnan |

Reading: the two Wi matmuls (ttnn's automatic config, 88 cores) and SDPA are 40 percent of the layer; the sharded rotary
costs its two reshards (105 us) on top of its 159 us; the separate `UnaryDeviceOperation` is the GELU, which the
automatic matmul path does not fuse (stage 3 fixes this with an explicit program config); LayerNorm and the head
reshapes run on 110 cores at 4096 rows.

### Findings that matter downstream

- Massive activations: ModernBERT-large develops channel outliers at channels 379, 382, 963, 195, 270 (max abs 1500 at
  layer 5, 2500 at layer 7, 10800 at layer 14, 27000 at layer 19 and after) against a median channel max of 60. In bf16
  the residual stream resolves 27000 only to 128, so the attention and MLP contributions to those channels are lost and
  per-layer PCC is dominated by whether the outlier channels agree. Per-layer PCC never falls below 0.9974 for the
  165-token row, but some sequences lose more: the synthetic 512-token fill row scores 0.9920 after the final norm (0.9665
  after layer 19) and one real 127-token question in the B 8 batch scores 0.9878 (0.908 after layer 19). The loss is a
  property of the sequence, not of the batch or the placement: the same row scores the same in every batch, and the
  interleaved and sharded B 2 runs agree to 1e-4.
- End to end the scorer logits at marker positions reach PCC 0.9942 (B 1) and 0.9921 (B 8) with a maximum error of 0.34
  to 0.38 logits, while the isolated scorer on an fp32 input errs by at most 0.006. The error therefore comes from the
  encoder plus head hidden state under the upstream default policy `bf8w_hifi3`. Stage 6 measures what this does to
  probabilities on the 200-decision gate corpus; the stage 8 sweep has `bf16_hifi4`, `bf8w_hifi3_head_bf16` and, if
  needed, the fp32 residual fallback of PLAN.md section 12 item 4. The scorer's fp32 final linear (typecast of the
  dense output then an fp32 matmul) costs nothing visible and is kept.
- The L1 attention chain does not fit at 4096 rows next to the 8x8 Wqkv program config (circular buffers end at
  1049600 bytes, L1 buffers start at 1030912). Shipped threshold 2048 rows; stage 3 A/B.
- The bf16 CPU reference scores 0.9838 against the device: with 27000-magnitude outliers the bf16 CPU model carries
  its own large error and is not a useful yardstick for -large; it is recorded as informational.
- Padded widths: 2816 and 3072 agree with each other to 0.03 (bf16 rounding on an output scale of 58) and with the
  interleaved 2624 path to 0.25 (different accumulation blocking); all three score 0.99947 against fp32.
- The checkpoint stores float16, not bfloat16 (PLAN.md section 1); `transformers` 5.12.1 `ModernBertConfig` exposes
  `sliding_window == 64` and `rope_parameters[layer_type]["rope_theta"]` as the plan expects.

### Not met or open

- Per-row encoder PCC below 0.99 exists for some real sequences (0.9878 and 0.9896 in the B 8 batch) although every
  batch aggregate is above 0.99; the decision-level gates of stage 6 decide whether the default policy ships.
- Performance numbers (eager `device_ms` in the logs) were taken with a host load of 25 to 47 and are not reported.
