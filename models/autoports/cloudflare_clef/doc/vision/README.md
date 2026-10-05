# Clef stage 2 (vision tower on device): host side

Date: 2026 Oct 05. Host side only. No Tenstorrent device was opened and `devrun` was not used. All paths are absolute. Times are ET (UTC-4); the host clock is UTC, so log lines are UTC.

Snapshot: `/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c`. Interpreter: `/home/hous/dev/clef/bin/hostrun python` (torch 2.11.0+cpu, transformers 5.12.1) with `OMP_NUM_THREADS=8`, `CLEF_MODEL` and `HF_MODEL` set to the snapshot.

## Files

All under `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/`:

- `tt/vision.py`: `ClefVisionArgs(VisionModelArgs)`, `ClefVision`, `load_hf_visual`, `read_vision_state_dict`, `tt_vision_state_dict`, `expected_tt_keys`, `padded_rows`, `vision_shape_report`.
- `scripts/vision_reference.py`: the HF reference for the vision tower (CPU, bf16, vision module only).
- `tests/test_vision.py`: 4 host tests (`-k host`) and 8 device tests (`-k device`): the 4 image cases, the video case, `test_device_upstream_defaults_reproduce_saved_rows`, `test_device_precision_improves_80_token_image` (these 7 are the `7 passed` of the 12:10 ET run, `/home/hous/dev/clef/logs/stage2_vision_device_precision.log`), and `test_device_sensitive_record_all_precisions` (added in the stage 2 remediation, see "Stage 2 remediation" at the end).
- `doc/vision/work_log.md`: one entry per run.

Reference data: `/home/hous/dev/clef/reports/reference/vision_ref/<record_id>.pt` (24 image records plus `video_2a7ddcfe4724ee1403a6291d21347162.pt`), index `/home/hous/dev/clef/reports/reference/vision_ref/index.json`, log `/home/hous/dev/clef/logs/stage2_vision_reference.log`. 830 MB on disk; `*.pt` is in the autoport `.gitignore`.

## What the upstream tower is, and how Clef fits it

The upstream tower is `/home/hous/dev/clef/tt-metal/models/demos/blackhole/qwen36/tt/vision/` (`model.py`, `vision_block.py`, `vision_attention.py`, `vision_mlp.py`, `patch_merger.py`, `vision_distributed_layernorm.py`, `vision_layernorm.py`, `vision_model_config.py`, `functional.py`).

`DropInVisionTransformer(reference_model, model_args, dtype, tt_ccl)` (`model.py` lines 154 to 329) takes the HF `Qwen3_5VisionModel` instance (`model.model.visual`) and uses it for two host steps per image: `patch_embed` (the Conv3d over `[S, 1536]` pixel patches, `model.py` line 243) and `fast_pos_embed_interpolate` (bilinear interpolation of the 48 x 48 learned position table, line 244). Everything else runs on device: 27 `VisionBlock`s (DistributedLayerNorm, tensor-parallel attention with 2D rotary embedding, DistributedLayerNorm, tensor-parallel MLP) and the `PatchMerger` (DistributedLayerNorm, 2 x 2 spatial merge by reshape, fc1 + GELU, fc2 with a reduce-scatter). The tower is always tensor-parallel over mesh column axis 1 (`VisionModelArgs`, `vision_model_config.py` lines 23 to 86). Block input and output are fractured along the hidden dim; the merger output is fractured along `out_hidden_size`, so each device holds `[1, 1, N, 2560]` of the `[1, 1, N, 5120]` result.

Weights come from `standardize_hf_keys_multimodal(reference_model.state_dict())` then `convert_hf_to_meta(state_dict, head_dim)` then a `visual.` prefix (`model.py` lines 183 to 186). The upstream tests (`tests/test_vision_attention.py`, `tests/test_vision_block.py`, `tests/test_patch_merger.py`) load `HF_MODEL` (default `Qwen/Qwen3.6-27B`) with `dummy_weights=True` and compare one module at a time against the HF module on random inputs: attention PCC 0.99 at grid (1, 16, 16) padded to 384 rows in bf16; the 27 blocks at grid (1, 98, 146) (14308 rows padded to 14336) in bfp8 with PCC 0.99 for blocks 0 to 23 and 0.85 for blocks 24 to 26; the merger at 14308 rows in bfp8 with PCC 0.99. No upstream test runs the whole tower against HF; `demo/vision_demo.py` runs it end to end and checks only that generation is not degenerate. `vision_demo.py` line 50 maps `MESH_DEVICE=N300` to a (1, 2) mesh, so a two-device tower was intended, but the tower was validated upstream on P150x4 (TP=4) only (plan, Stage 2).

Clef vision config (`config.json` `vision_config`) against the values the TT code derives at TP=2 (`tt/vision.py` `vision_shape_report`, checked by `tests/test_vision.py::test_host_clef_vision_config_fits_tp2`):

| Field | Clef | TT derivation at TP=2 | Source of the constraint |
|---|---|---|---|
| `depth` | 27 | 27 blocks; upstream block bars assume 24 as the bar boundary | `test_vision_block.py` line 38 |
| `hidden_size` | 1152 | `dim` 1152, 576 per device | `vision_model_config.py` line 77 |
| `num_heads` | 16 | 8 local heads per device | line 75 |
| `head_dim` | 72 | padded to 96 (`padded_head_dim`), `qkv_size` 4608, 2304 per device | lines 47, 52, 76; `vision_attention.py` lines 101 to 103 |
| `intermediate_size` | 4304 | `hidden_dim` padded to 4352 (multiple of 32 x 2), 2176 per device | lines 38 to 42, 78 |
| `spatial_merge_size` | 2 | merger `mlp_size` 4608, 2304 per device | lines 83, 85 |
| `out_hidden_size` | 5120 | 2560 per device | line 86 |
| `patch_size`, `temporal_patch_size`, `in_channels` | 16, 2, 3 | host only (`patch_embed` input width 3 x 2 x 16 x 16 = 1536) | HF `Qwen3_5VisionPatchEmbed` |
| `num_position_embeddings` | 2304 | host only (`fast_pos_embed_interpolate`, 48 x 48 grid) | HF `Qwen3_5VisionModel` |
| `deepstack_visual_indexes` | `[]` | not used; the TT `PatchMerger` asserts `postshuffle_norm=False` and ships no deepstack merger | `patch_merger.py` line 54 |
| `hidden_act` | `gelu_pytorch_tanh` | TT MLP uses `activation="gelu"` in `ttnn.linear` | `vision_mlp.py` line 136 |

Every divisibility assert at `vision_model_config.py` lines 71 to 86 holds at TP=2 (`divisible` in the shape report is all true). The Qwen3.5-9B-Base vision config in the HF cache (`/home/hous/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B-Base/snapshots/*/config.json`) is identical to Clef's except `out_hidden_size` 4096, and it also has `deepstack_visual_indexes: []`, so "no deepstack" is a property of the Qwen3.5 family here and not a Clef deviation. The Qwen3.6-27B config that the upstream tests load is not in the HF cache on this box (offline), so its exact values are unverified; the upstream code's hard-coded derivations (`in0_block_w = dim // 1024 = 1` in `VISION_WO_PREFILL_PROGCFG`, `vision_model_config.py` line 67) fit `hidden_size` 1152.

Things in the upstream tower that do not fit Clef's use, and what the wrapper does about them:

1. `DropInVisionTransformer.forward` pads every image to a multiple of 2048 rows (`model.py` line 232) and runs the attention without any padding mask: `VisionTransformer.forward` does not pass `cu_window_seqlens` to the blocks, so `VisionAttention.forward_prefill` calls SDPA with `is_causal=False` and no mask (`vision_attention.py` lines 380 to 389). The pad rows are zeros; after `norm1` they equal `norm1.bias`, so their keys and values are a constant nonzero vector that every real query attends to. The reference cartoons have 320 to 2000 patches (see the token table below), so with a 2048 pad the junk keys are 2 % to 84 % of the sequence. The upstream unit tests pass under this because PCC is scale invariant and their inputs are random; this is not evidence that real images survive it. The wrapper masks the pad rows through the existing windowed SDPA: it drives the blocks itself and passes `cu_window_seqlens = [0, n_patches, rows]` (int32, row major, replicated), which the SDPA kernel turns into a block-diagonal mask (`/home/hous/dev/clef/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp` line 337; validation at `device/sdpa_device_operation.cpp` lines 435 to 475: on device, int32 or uint32, 1-D, 2 to 1024 entries, no tile alignment requirement). Real tokens then attend only to real tokens, and the pad rows attend among themselves and are sliced off before the merger. `VisionBlock.forward` and `VisionAttention.forward` already accept `cu_window_seqlens` (`vision_block.py` lines 89 and 110, `vision_attention.py` lines 45 and 293), so no qwen36 file changes. `CLEF_VISION_PAD_MASK=0` turns the mask off; `ClefVision.upstream_features_torch` runs the unmodified `DropInVisionTransformer.forward` for comparison, and the device test records both PCCs per image.
2. The pad length. `padded_rows(n, granule)` defaults to the upstream 2048 granule. Smaller granules are allowed by the ops only in this pattern: multiples of 128 up to 1024 (`vision_attention.py` line 296 asserts `seq_len % 128 == 0`), then 1024 or 2048 (`vision_mlp.py` line 127 reshapes to 1024-row slabs when `seq_len >= 1024`), then multiples of 2048 (`vision_attention.py` lines 299 to 302, `MAX_QKV_MM_SEQ_LEN = 2048`). `CLEF_VISION_PAD=128` selects the tight rule; it is untested on device and each distinct row count compiles its own programs.
3. Several images per request. `DropInVisionTransformer.forward` concatenates per-image outputs along dim 1 (`model.py` line 326), which only works when all images have the same token count. `ClefVision.forward` concatenates along the row dim (dim 2), so `image_features` is `[sum N_i, 5120]` in request order, which is the order the placeholders appear in `input_ids`.
4. Weight dtype. `dtype` reaches only `wo`, `wo.bias` and the merger `fc1`, `fc2`; `wqkv`, `wqkv.bias` and both MLP matrices are `bfloat8_b` regardless (`vision_attention.py` lines 187, 213; `vision_mlp.py` lines 73, 100). `ClefVision` defaults to `bf16` for the configurable part (`CLEF_VISION_DTYPE=bfp8` for the upstream default). Tower weights are 461 M parameters, about 0.9 GB in bf16 over the two devices, so DRAM is not a concern.
5. Weight cache. `DropInVisionTransformer` passes `model_args.weight_cache_path(dtype)` to every module. Stage 0 Finding 3 (cached mesh-tensor reload spins on the (1, 2) submesh, `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/doc/stage0/README.md`) applies to the tower too, so `ClefVisionArgs.weight_cache_path` returns `None` unless `CLEF_VISION_WEIGHT_CACHE=1`; every upstream vision module takes `None` as "no cache" (`vision_attention.py` line 134, `vision_mlp.py` line 61, `patch_merger.py` line 102, `vision_layernorm.py` line 32). With caching on, the path is `<TT_CACHE_PATH>/<device_name>/vision_cache_<bf16|bfp8>_mesh1x2_clef_2f3de3dd`.
6. `q_norm` and `k_norm`. `VisionAttention` raises `NotImplementedError` if the state dict has them (`vision_attention.py` lines 225 and 227). Clef's `model.visual.*` has none (checked by `test_host_key_mapping_matches_upstream_modules`).
7. The HF reference module itself. `VisionModelArgs.reference_vision_model` loads the whole `Qwen3_5ForConditionalGeneration` (55 GB, memory mapped) to get `model.visual`. `load_hf_visual` builds `Qwen3_5VisionModel(config.vision_config)` directly under `torch.set_default_dtype(bfloat16)` (so parameters are bf16 and the rotary `inv_freq` buffer stays float32 exactly as `from_pretrained(dtype=bfloat16)` leaves it), sets `_attn_implementation = "sdpa"` (the stage 0 release model also runs sdpa), and loads the 333 `model.visual.*` tensors from shard `model-00012-of-00012.safetensors` with `strict=True` (6.6 s, `index.json` `visual_build_seconds`).
8. `fast_pos_embed_interpolate` is deprecated in transformers 5.12.1 (FutureWarning, "removed in v5.11", still present). `DropInVisionTransformer` and `ClefVision._host_patches` call it; the wrapper silences the warning. The pinned transformers version keeps it working.
9. Rounding order of the position add: HF casts the interpolated position embedding to bf16 before adding it to the bf16 patch embedding; `DropInVisionTransformer` (and `ClefVision`) add in float32 and cast to bf16 in `prepare_input`. Expected to be below PCC resolution; recorded here because it is a difference in the host step.

## Weight key mapping

`ClefModelArgs.vision_state_dict()` (`tt/loader.py`) and `read_vision_state_dict()` (`tt/vision.py`) return the 333 `model.visual.*` tensors with the prefix stripped, which is exactly `Qwen3_5VisionModel.state_dict()` (`load_hf_visual` loads it with `strict=True`). `tt_vision_state_dict(hf_state_dict, head_dim=72)` applies the same three steps as `DropInVisionTransformer.__init__`:

| HF key (prefix `model.visual.` stripped) | After `standardize_hf_keys_multimodal` | After `convert_hf_to_meta(head_dim=72)` plus `visual.` | Read by |
|---|---|---|---|
| `blocks.N.attn.qkv.{weight,bias}` `[3456, 1152]`, `[3456]` | `blocks.N.self_attn.qkv_proj.*` | `visual.blocks.N.attention.{wq,wk,wv}.*`: `split_hf_keys` cuts equal thirds, `convert_hf_qkv_to_meta_format` applies `reverse_permute` (16 heads of 72) to `wq`, `wk` and their biases; `wv` is unchanged | `VisionAttention` (`vision_attention.py` lines 140 to 143, 169 to 207) |
| `blocks.N.attn.proj.{weight,bias}` | `blocks.N.self_attn.o_proj.*` | `visual.blocks.N.attention.wo.*` | `vision_attention.py` lines 232, 254 |
| `blocks.N.mlp.linear_fc{1,2}.{weight,bias}` | unchanged | `visual.blocks.N.feed_forward.linear_fc{1,2}.*` | `vision_mlp.py` lines 58, 59 |
| `blocks.N.norm{1,2}.{weight,bias}` | unchanged | `visual.blocks.N.norm{1,2}.*` | `vision_layernorm.py` lines 28 to 31 through `get_state_dict_prefix("norm1", N)` |
| `merger.norm.*`, `merger.linear_fc{1,2}.*` | unchanged | `visual.merger.*` | `patch_merger.py` lines 87, 99, 100 |
| `patch_embed.proj.{weight,bias}` `[1152, 3, 2, 16, 16]` | `patch_embed.o_proj.*` | `visual.patch_embed.wo.*` | nobody on device; the HF module runs `patch_embed` on host |
| `pos_embed.weight` `[2304, 1152]` | unchanged | `visual.pos_embed.weight` | nobody on device; host interpolation |

Result (`/home/hous/dev/clef/logs/stage2_vision_host_tests.log`, `test_host_key_mapping_matches_upstream_modules`): 333 HF tensors map to 441 keys; the 438 keys that `expected_tt_keys(27)` derives from the module code are all present; the 3 extra keys are the host-only `visual.patch_embed.wo.{weight,bias}` and `visual.pos_embed.weight`; no `q_norm` or `k_norm`; `wq` equals `reverse_permute(q_rows, 16, 1152, 1152)` and `wv` equals the last 1152 rows of `qkv.weight` bit for bit. `VisionAttention` pads each 72-wide head to 96 at load time and pads the rotary tables to 96 at run time (`vision_attention.py` lines 151 to 163, 343 to 346).

## HF reference (CPU)

Command (2026 Oct 05, 08:46 ET; log `/home/hous/dev/clef/logs/stage2_vision_reference.log`):

```
cd /home/hous/dev/clef/tt-metal && nohup env OMP_NUM_THREADS=8 CLEF_MODEL=<snapshot> HF_MODEL=<snapshot> /home/hous/dev/clef/bin/hostrun python models/autoports/cloudflare_clef/scripts/vision_reference.py --records /home/hous/dev/clef/reports/reference/records_image.jsonl --records /home/hous/dev/clef/reports/reference/dev16_image.jsonl --threads 8 > /home/hous/dev/clef/logs/stage2_vision_reference.log 2>&1 &
```

Per record the script builds the request exactly as `scripts/cpu_reference.py` does (PIL `convert("RGB")`), calls the release `encode_record(tokenizer, record, processor=processor)` through `tt/encode.py`, takes `pixel_values` and `image_grid_thw` from `encoded.media`, counts the `image_token_id` (248056) placeholders in `input_ids`, runs `Qwen3_5VisionModel` with forward hooks on blocks 0, 12, 23 and 26, and saves to `<record_id>.pt`: `pixel_values` (float32 `[S, 1536]`, kept so the device test feeds the identical input), `image_grid_thw`, `merged` (float32 `[N, 5120]`, the `pooler_output` that `Qwen3_5Model.get_image_features` scatters into the text embeddings), `blocks` (float32 `[S, 1152]` per tap), token counts and timing. It raises if the placeholder count differs from `N` or from `prod(grid) / 4`; it did not. The video file comes from the 16 x 20 cartoon as 4 frames (frame k rotated by k degrees, white fill) through `processor.video_processor(videos=[frames], do_sample_frames=False)`: grid (2, 16, 20), 640 patches, 160 tokens.

Timings: processor 0.88 s, visual build and load 6.6 s, 24 image forwards 23.4 s in total (0.14 s for 320 patches to 2.2 s for 2100 patches), video 0.27 s. Source: `index.json`.

Token counts of the 8 reference records (`records_image.jsonl`; `input_tokens` is the whole encoded request, `media_token_offset` 36 is the prefix length before the `<|vision_start|>` block):

| Record | Grid (t, h, w) | Patches S | Image tokens N | Request tokens | HF forward (s) | In the device test |
|---|---|---|---|---|---|---|
| `6900e8fa480aa890209946c139b55693` | (1, 22, 38) | 836 | 209 | 428 | 0.58 | yes |
| `eb937eba821eb5eea0503d07ff3ce79f` | (1, 28, 36) | 1008 | 252 | 482 | 0.75 | |
| `2a92d6c1c2ddd393b4b9d8bd4f9881eb` | (1, 28, 38) | 1064 | 266 | 510 | 0.89 | |
| `848089ea8a254944feaa3c3456e331d5` | (1, 40, 50) | 2000 | 500 | 736 | 2.11 | yes |
| `c4ea611f295ae54b86e78c4fea823f4e` | (1, 22, 38) | 836 | 209 | 461 | 0.61 | |
| `6c1478b40fa4ab7f72933d7e3fe58d0a` | (1, 28, 38) | 1064 | 266 | 511 | 0.89 | |
| `2a7ddcfe4724ee1403a6291d21347162` | (1, 16, 20) | 320 | 80 | 330 | 0.20 | yes, also the video source |
| `e000b5c0a8358750424503bd92a9a656` | (1, 26, 36) | 936 | 234 | 478 | 0.71 | yes |

The grids and token counts equal the stage 0 ablation (`/home/hous/dev/clef/reports/reference/image_ablation.json`). The 16 dev records (`dev16_image.jsonl`) range from 308 patches (77 tokens) to 2100 patches (525 tokens); two of them (`f3edd3ef...`, `c3828c64...`) exceed 2048 patches and pad to 4096 rows under the 2048 rule. Every image is one `t = 1` grid; `S = 4 N` always.

## Merge and M-RoPE upstream (what the engine must pass)

- Tower call: `Qwen36Model.get_image_features(pixel_values, image_grid_thw)` (`/home/hous/dev/clef/tt-metal/models/demos/blackhole/qwen36/tt/model.py` lines 237 to 264) stashes the grid in `self._req_image_grid_thw`, clears `_req_video_grid_thw`, calls `self.vision_model.forward(pixel_values, grid_thw=...)` and returns `ttnn.reshape(out, (-1, hidden))`: `[N, 5120]` global, `[N, 2560]` per device, fractured along the hidden dim like the embedding output. `get_video_features` (lines 266 to 290) is the same call with the grid stashed as a video grid, which switches the placeholder id to `video_token_id` 248057 and the M-RoPE to the per-frame video rule (`_vision_placeholder_token_id`, lines 292 to 300).
- Eager merge: `_scatter_vision_tokens(x, token_ids, vision_tokens)` (lines 686 to 784) finds `token_ids == image_token_id` on host, asserts the count equals `vision_tokens.shape[0]`, uploads a `[N, H]` int32 scatter index (sharded `dims=(None, 1)`) and a `[rows, 1]` predicate (replicated), then `ttnn.scatter` into a zero buffer and `ttnn.where`. Used by `prefill_tp` (line 602), `prefill` (line 799), `prefill_layer_chunked` (line 823) and `_prefill_paged_tp` (line 2802).
- Trace-safe merge: `_alloc_vision_merge_buffers(device, chunk_size)` (lines 312 to 375) allocates `_vis_buf` `[1, 1, chunk, H]` hidden-sharded and `_vis_mask_buf` `[1, 1, chunk, 1]` replicated; `_set_vision_merge(ids_host, vision_tokens, vis_row_offset)` (lines 408 to 470) reads the fractured rows back with `ConcatMeshToTensor(dim=1)`, places the segment's slice at its placeholder rows on host and copies both buffers to device; `_apply_vision_merge(x, length)` (lines 377 to 406) is the captured `ttnn.where`; `_vis_row_offset_for(token_ids, chunk_start)` (lines 485 to 492) counts placeholders before a chunk. `_forward_prefill_chunk_masked_tp` applies `_apply_vision_merge(x, length=bucket)` right after the embedding (line 2048); `prefill_masked_bucket` calls `_build_request_rope` at `chunk_start == 0` (line 2146) and `_set_vision_merge(token_buf, vision_tokens, vis_row_offset)` before the forward (line 2159).
- M-RoPE: `_build_request_rope(token_ids, vision_tokens)` (lines 302 to 310) passes the stashed grid to `Qwen36RoPESetup.build_request_rope` (`tt/rope.py` lines 153 to 191), which derives `mm_token_type_ids` from the placeholder ids, calls `get_rope_index(input_ids, mm, image_grid_thw=..., video_grid_thw=..., spatial_merge_size=2)` and `get_rot_mats(inv_freq, position_ids, mrope_section=[11, 11, 10], attention_scaling=1.0)` from `tt/attention/rope_tp.py` (lines 96 and 217), and stores a sequence-indexed cos/sin table plus `rope_delta`. `prefill_cos_sin_torch(start, length)` (`rope.py` line 206) serves every prefill chunk from that table; text-only requests clear it and use 1D RoPE. The engine does not compute M-RoPE itself.

Contract for `tt/engine.py` (owned by the device agent):

1. After `Qwen36Model(...)` is built and before any trace capture: `vision = ClefVision(mesh, clef_args)` (reads the 333 tensors from the safetensors and builds the HF visual on host, then the TT blocks and merger on the mesh), then `vision.attach(model)` sets `model.vision_model = vision` and `model.vision_args`. `Qwen36Model.init_vision_model()` then returns the attached tower and does not load the 55 GB checkpoint. The duck type `get_image_features` needs is `forward(pixel_values, grid_thw=...)` returning `[1, 1, N, out/TP]`.
2. Per request with images: `encoded = encode(tokenizer, record, processor=processor)`; `media = encoded.media`; `vision_tokens = model.get_image_features(media["pixel_values"], media["image_grid_thw"])`. For videos: `model.get_video_features(media["pixel_values_videos"], media["video_grid_thw"])`. The `encode_record` sequence is `prefix_ids + media_ids + state_ids + ...` with `media["token_offset"] = len(prefix_ids) = 36`, so the image rows sit at positions 37 to 36 + N inside the prefix-cache head (`split_for_cache` keeps them in the first piece). A prefix-cache key for an image request must include the image content, not only the token ids, because the `<|image_pad|>` ids are identical for every image of the same grid.
3. Pass `vision_tokens` to the prefill entry the engine uses. With `prefill_masked_bucket(..., vision_tokens=vision_tokens, vis_row_offset=...)` the model stages M-RoPE and the merge itself. If the engine calls `_forward_prefill_chunk_masked_tp` directly, it must first call `model._build_request_rope(token_ids[:, :actual_len], vision_tokens)` once per request and `model._set_vision_merge(token_buf, vision_tokens, vis_row_offset)` per segment (trace-safe buffers allocated by `_alloc_vision_merge_buffers`), or use `_scatter_vision_tokens` on the embedding output in an eager path. One image per `get_image_features` call pads to 2048 rows (4096 above 2048 patches); the result rows are exactly `N`, with no padding, which `_scatter_vision_tokens` asserts.
4. Run the tower eagerly before capturing prefill traces, as `vision_demo.py` `_compute_vision_tokens` does (lines 234 to 270), so its program compiles do not land while a trace is parked. The tower has no trace of its own.

## Tests

Host (ran, `/home/hous/dev/clef/logs/stage2_vision_host_tests.log`): `4 passed, 5 deselected in 8.15s`.

```
cd /home/hous/dev/clef/tt-metal && OMP_NUM_THREADS=8 CLEF_MODEL=<snapshot> HF_MODEL=<snapshot> /home/hous/dev/clef/bin/hostrun pytest models/autoports/cloudflare_clef/tests/test_vision.py -k host --timeout=600
```

- `test_host_key_mapping_matches_upstream_modules`: the mapping result above.
- `test_host_clef_vision_config_fits_tp2`: the shape table and the `padded_rows` rule.
- `test_host_hf_visual_loads_standalone`: bf16 parameters, float32 `inv_freq`, sdpa, 27 blocks, 460,730,096 parameters.
- `test_host_reference_index_lists_the_test_images`: the 4 reference files exist and `placeholders == n_tokens`.

Collection dry run (`/home/hous/dev/clef/logs/stage2_vision_collect_only.log`): 9 tests collected, no import error.

Device (not run here; the device agent runs it):

```
cd /home/hous/dev/clef/tt-metal && nohup /home/hous/dev/clef/bin/devrun timeout 1800 env OMP_NUM_THREADS=8 CLEF_MODEL=/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c HF_MODEL=/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c pytest models/autoports/cloudflare_clef/tests/test_vision.py --timeout=1500 > /home/hous/dev/clef/logs/stage2_vision_device.log 2>&1 &
```

Fixture: mesh path A as in `tests/test_tp2_sanity.py` (`CLEF_PARENT=1x4`, `FABRIC_1D`, (1, 4) parent, `create_submesh((1, 2), (0, 0))`, `l1_small_size=24576`, two command queues; `CLEF_PARENT=2x2` for path B); `ClefVisionArgs(submesh, snapshot, max_seq_len=4096)`; `ClefVision(submesh, args, load_hf_visual(snapshot))` with the env defaults (`bf16`, pad 2048, mask on, no weight cache). Results JSON: `/home/hous/dev/clef/reports/stage2_vision_tower.json` (mesh facts, tower build seconds, CCL topology and link count, per image PCCs and timings).

- `test_device_image_tower_matches_hf[<record>]` for `2a7ddcfe` (80 tokens), `6900e8fa` (209), `e000b5c0` (234), `848089ea` (500): one tapped pass (blocks 0, 12, 23, 26 read back and sliced to the real rows), one warm pass, the `image_features` ttnn tensor (asserts global shape `[N, 5120]` and that `ConcatMeshToTensor(dim=1)` of it equals the torch result, which is what `_set_vision_merge` relies on), and one pass through the unmodified `DropInVisionTransformer.forward` (recorded as `pcc_upstream_unmasked`, not asserted). Bars: merged PCC >= 0.97 (plan), blocks 0, 12, 23 >= 0.99 and block 26 >= 0.85 (upstream `test_vision_block.py`).
- `test_device_video_tower_matches_hf`: the 4-frame reference through `video_features_torch`, merged PCC >= 0.97. Skips if the reference file is missing.
- `test_device_upstream_defaults_reproduce_saved_rows` and `test_device_precision_improves_80_token_image`: see "Plan amendment (12:10 ET)".
- `test_device_sensitive_record_all_precisions[7b7ef383...]`: the dev16 record `7b7ef383` through three towers built in the same process (accuracy, the env default; upstream defaults with `activation_bf16=False`; upstream weights with `activation_bf16=True`), merged and block PCCs per tower, the device rows of each tower saved to `CLEF_VISION_ROWS_DIR` (default `/home/hous/dev/clef/reports/stage2r_tower_rows/<record>_<tower>.pt`); the bars apply to the accuracy tower.

Environment knobs read by `tt/vision.py`: `CLEF_VISION_DTYPE` (`bf16` default, `bfp8`), `CLEF_VISION_PAD` (2048 default, 128 for the tight rule), `CLEF_VISION_PAD_MASK` (`1` default), `CLEF_VISION_WEIGHT_CACHE` (`0` default), `CLEF_VISION_ACT_BF16` (`1` default since 12:10 ET, see "Tower precision knobs"; the gate run of 11:00 ET and the 11:06 ET parity run predate the knob and ran with bfp8 attention output).

## Open items for the device agent

1. The padding mask through `cu_window_seqlens` is the one behaviour that differs from the upstream forward; the device test records masked and unmasked PCC per image. If the windowed SDPA fails on the (1, 2) mesh, set `CLEF_VISION_PAD_MASK=0` to fall back to the upstream behaviour and read `pcc_upstream_unmasked` to judge the cost.
2. `ClefVisionArgs` runs the base `tt_transformers.ModelArgs.__init__` on the Clef config (not the `Qwen36ModelArgs` override that stage 0 proved on the submesh). It is the same path `Qwen36Model.init_vision_model` takes for Qwen3.6-27B. Unverified on this box until the device test runs.
3. Tower weights convert from host tensors on every start (no cache). Build time is recorded in the results JSON as `build_seconds`.

## Device results

Date: 2026 Oct 05, device side (10:56 to 11:01 ET). Mesh path A: `FABRIC_1D`, (1, 4) parent, `create_submesh((1, 2), (0, 0))`, chips [1, 0], cluster type `P300_X2`, `Topology.Ring`, 2 links (stock `qwen36`, as stage 0 and 1). `ClefVisionArgs` on the submesh builds in 1.7 s (open item 2 of the host side is closed: the base `tt_transformers` `ModelArgs.__init__` runs on the Clef config on this mesh), the HF visual loads in 7.4 to 9.4 s, the TT blocks and merger convert from host tensors in 1.9 to 3.6 s (no weight cache). Results: `/home/hous/dev/clef/reports/stage2_vision_tower.json` (mask on, the gate run), `/home/hous/dev/clef/reports/stage2_vision_tower_nomask.json` (`CLEF_VISION_PAD_MASK=0`); logs `/home/hous/dev/clef/logs/stage2_vision_device.log` (run 1), `stage2_vision_device_run2.log` (gate run, `5 passed in 19.34 s`), `stage2_vision_device_nomask.log`.

Command (the gate run):

```
cd /home/hous/dev/clef/tt-metal && /home/hous/dev/clef/bin/devrun timeout 1800 env OMP_NUM_THREADS=8 CLEF_MODEL=<snapshot> HF_MODEL=<snapshot> pytest models/autoports/cloudflare_clef/tests/test_vision.py -k device --timeout=1500 -p no:cacheprovider
```

Tower PCC against the HF CPU bf16 reference (`vision_ref/<record>.pt`), bf16 configurable weights, pad 2048, one image per pass:

| Record | Grid | Patches / tokens | Window (mask on) | Merged, mask on | Block 0 / 12 / 23 / 26, mask on | Merged, mask off (= upstream forward) | Block 0 / 12 / 23 / 26, mask off | Warm TT s | HF CPU s |
|---|---|---|---|---|---|---|---|---|---|
| `2a7ddcfe` | (1, 16, 20) | 320 / 80 | `[0, 320, 2048]` | 0.98901 | 0.99965 / 0.99638 / 0.99762 / 0.99051 | 0.89452 | 0.94673 / 0.74469 / 0.93467 / 0.80906 | 0.081 | 0.203 |
| `6900e8fa` | (1, 22, 38) | 836 / 209 | `[0, 836, 2048]` | 0.99564 | 0.99956 / 0.99719 / 0.99804 / 0.99486 | 0.94975 | 0.98344 / 0.95258 / 0.97326 / 0.91264 | 0.094 | 0.581 |
| `e000b5c0` | (1, 26, 36) | 936 / 234 | `[0, 936, 2048]` | 0.99789 | 0.99948 / 0.99913 / 0.99844 / 0.99578 | 0.96154 | 0.98754 / 0.86657 / 0.97214 / 0.92037 | 0.097 | 0.709 |
| `848089ea` | (1, 40, 50) | 2000 / 500 | `[0, 2000, 2048]` | 0.99780 | 0.99945 / 0.99748 / 0.99837 / 0.99462 | 0.99462 | 0.99815 / 0.98942 / 0.99605 / 0.98812 | 0.125 | 2.114 |
| `video_2a7ddcfe` (4 frames) | (2, 16, 20) | 640 / 160 | `[0, 320, 640, 2048]` | 0.99427 | not tapped | 0.92233 (per-frame window, pad rows in the last frame) | not tapped | 0.088 | 0.269 |

Bars: merged >= 0.97 (plan), blocks 0, 12, 23 >= 0.99 and block 26 >= 0.85 (upstream `test_vision_block.py`). All five cases pass with the mask on. The mask-off column equals `pcc_upstream_unmasked` (the unmodified `DropInVisionTransformer.forward`) to six decimals, so the wrapper with `CLEF_VISION_PAD_MASK=0` is the upstream tower; without the mask three of the four images fail the 0.97 bar and the cost grows as the pad fraction grows (84 % pad rows: 0.895; 2 % pad rows: 0.995). The `image_features` mesh tensor has the per-device shape `[N, 2560]`; `ConcatMeshToTensor(dim=1)` returns `[N, 5120]` equal to the torch result (round trip max abs 0.0), which is what `_set_vision_merge` reads. The first tapped call of a fresh process compiles the kernels (11.3 s in run 1; 0.2 s in later runs with the kernel binaries cached on disk); a warm pass is 0.08 to 0.13 s per image, 2.5 to 17 times faster than the HF CPU forward at 8 threads.

Video finding (run 1, 0.934 before the fix): HF `Qwen3_5VisionModel.forward` attends within each temporal patch group (`cu_seqlens = repeat_interleave(h * w, t).cumsum`, `transformers/models/qwen3_5/modeling_qwen3_5.py` line 1096 through `get_vision_cu_seqlens`); the host-side wrapper passed one window over all 640 patches. `window_bounds(cu_seqlens, rows, mask_padding)` in `tt/vision.py` now takes the per-frame boundaries that `qwen3_5_vision_transformer_preprocess` already returns and appends the pad segment, so the SDPA mask is `[0, hw, 2hw, ..., n_patches, rows]`. Images (t = 1) are unchanged. With the mask off the last frame absorbs the pad rows, and a one-frame image with the mask off gets no window (the upstream behaviour).
## Engine integration

Date: 2026 Oct 05, device side. Files: `tt/engine.py` (the media path), `tt/vision.py` (`window_bounds`), `scripts/image_parity.py`, `scripts/cpu_reference.py` (video frames). No `qwen36` file changed.

### What the engine does with media

- Build: after `Qwen36Model(...)` and before any slot allocation, `ClefEngine.__init__` builds `ClefVisionArgs(mesh, snapshot, max_batch_size=1, max_seq_len=4096)` and `ClefVision(mesh, vision_args)` (reads the 333 `model.visual.*` tensors from the safetensors, builds the HF visual on host, the 27 blocks and the merger on the mesh), calls `vision.attach(model)` (sets `model.vision_model` and `model.vision_args`, so `Qwen36Model.get_image_features` and `get_video_features` work and `init_vision_model()` never loads the 55 GB checkpoint), and allocates the trace-safe merge buffers once with `model._alloc_vision_merge_buffers(mesh, 1024)` (`_vis_buf` `[1, 1, 1024, 5120]` hidden-sharded bf16, `_vis_mask_buf` `[1, 1, 1024, 1]` replicated). `CLEF_VISION=0` (or `ClefEngine(..., vision=False)`) skips all of it and leaves the stage 1 engine as it was. `engine.timings["vision_build_s"]` and `engine.dram_free_after_text_weights` (before the tower) are recorded next to `dram_free_after_weights` (after the tower), which `_fit_slots` uses.
- Per request: `prefill_state(state_ids, slot=0, key=None, media=None)`, `prefill_hidden(token_ids, slot=0, media=None)` and `cached_hidden(state_ids, tail_ids, slot=0, media=None)` accept `EncodedRecord.media` from the release `encode_record(..., processor=processor)`. `_vision_request(media)` runs `model.get_image_features(pixel_values, image_grid_thw)` or `model.get_video_features(pixel_values_videos, video_grid_thw)` (one media kind per request; the backbone stages one grid) and keeps the `[N, 2560]` per-device rows, the grid, the modality and the tower seconds in a `VisionRequest`. `_stage_rope(ids, vision)` restores the grid stash (`model._req_image_grid_thw` or `_req_video_grid_thw`) and calls `model._build_request_rope(ids, vision.tokens)`, which builds the 3D M-RoPE table (`get_rope_index` with `mrope_section [11, 11, 10]`) from the full token ids; text requests call it with `None` and get 1D RoPE as before. `_run_chunk` stages the merge per 1024-token chunk with `model._set_vision_merge(token_buf, vision.tokens, model._vis_row_offset_for(full_ids, chunk_start))` (host placement of the chunk's slice of the packed rows at its `<|image_pad|>` or `<|video_pad|>` positions, then two host-to-device copies); `_forward_prefill_chunk_masked_tp` applies `ttnn.where(mask, vision, text)` right after the embedding. A text chunk, or a chunk past the media rows, stages the zero mask, so the `where` is the identity.
- Prefix cache: the media rows sit at positions 37 to 36 + N (`token_offset` 36), inside the 128-aligned cached prefix `[0, S0)` for every record here. `prefill_state` runs the prefix with the merge and the M-RoPE, snapshots the GDN state as before, and keeps the `VisionRequest` and the full state ids on the `StateHandle`. `schema_hidden` rebuilds the M-RoPE table from `state_ids[:S0] + suffix + schema` (positions after an image are shifted by `rope_delta`, so the tail cannot use 1D RoPE) and runs the tail with the handle's rows, which also covers a short state whose media rows spill past `S0`. The slot releases the device rows when it is reused (`_release_slot`). `cached_hidden` keys the slot on `cache_key(state_ids) + media_key(media)` (SHA-1 over the pixel tensors and grids), because the placeholder ids are identical for every image of the same grid; the server passes its own `key`.
- `probs_for_request(record, mode, slot)`: `load_media_paths` opens `images` (paths or PIL images) and `videos` (lists of frame paths or images) with PIL in RGB, `encode(tokenizer, request, processor=processor)` (the processor is loaded lazily), then the full or the cached path with `media=encoded.media`. The result carries `timing.vision_s` and a `vision` block (`grid`, `n_patches`, `n_rows`, `seconds`, `tower` with the padded rows and the window).

### Text-only behaviour is byte-identical

`tests/test_engine.py -k l4` ran twice on 2026 Oct 05 (11:03 and 11:04 ET): with the tower attached (`/home/hous/dev/clef/logs/stage2_engine_l4_vision.log`, `/home/hous/dev/clef/reports/stage2_engine_l4_vision_on.json`) and with `CLEF_VISION=0` (`stage2_engine_l4_novision.log`, `stage2_engine_l4_vision_off.json`), both `8 passed`. Every `test_hidden_vs_hf` PCC field is identical (min per-position 0.99993, 0.999824, 0.999806, 0.999915 at T=300, 1500, 2300, 8192), both tail cases are identical (0.999999), and the 3 records x 2 modes `probs` rows are bit-equal (`stage2_tt_text_l4_{full,cached}_vision_on.jsonl` against the `CLEF_VISION=0` rows: max |dp| 0.0). The `ttnn.where` with a zero mask after the embedding is an exact identity. Warmed `prefill_hidden` 0.012 / 0.014 / 0.019 / 0.036 s at 128 / 256 / 512 / 1024 tokens in both runs; 8192-token state 0.272 s (on) and 0.277 s (off). The stage 1 `stage1_engine_l4.json` of 08:54 ET predates the `QWEN36_GDN_GATE_FP32=1` change and is not a byte-identity baseline (max |dp| 0.00089 against either run). The stage 1 files were restored after the runs.

### Cached path equals the full path for media (orchestrator note from the stage 1 review)

`tests/test_engine_media.py` (4 layers, `/home/hous/dev/clef/reports/stage2_engine_media_l4.json`, logs `/home/hous/dev/clef/logs/stage2_engine_media_l4.log` and `..._run2.log`): for the image record `2a7ddcfe` (80 image rows at positions 37 to 116, state S=133, S0=128) and the video record (160 rows, S=236, S0=128, rows spanning past S0 because timestamps interleave the frames) the cached path gives bit-equal probabilities to the full path on the miss, on the hit, and on a hit after another image request and a text request ran in other slots (max |dp| 0.0 in all six comparisons). `rope_delta` after the cached schema call is -70 (image) and -140 (video): the tail ran on the request's M-RoPE table rebuilt from `state_ids[:S0] + tail` with the handle's grid. At 64 layers the parity run gives the same result on the 8 reference records and the video: cached versus full max |dp| 0.000000 (`stage2_image_parity.json`, `cached_vs_full_max_dp`).

### Image parity (64 layers, stage 1 precision knobs, eager)

Run of 11:06 ET: `/home/hous/dev/clef/logs/stage2_image_parity.log`, summary `/home/hous/dev/clef/reports/stage2_image_parity.json`, candidate rows `/home/hous/dev/clef/reports/stage2_tt_{reference,dev16,video}_{full,cached}.jsonl`, `parity_compare.py` output `/home/hous/dev/clef/reports/stage2_parity_<set>_<mode>.{json,md}`. Command:

```
cd /home/hous/dev/clef/tt-metal && /home/hous/dev/clef/bin/devrun timeout 3600 env OMP_NUM_THREADS=8 CLEF_MODEL=<snapshot> HF_MODEL=<snapshot> python models/autoports/cloudflare_clef/scripts/image_parity.py --sets reference,dev16,video --cached reference,video
```

| Set (records / questions) | Path | max dp | mean dp | median dp | argmax flips | flips at margin >= 0.05 | acc ref / cand | Gate (max dp <= 0.10, 0 flips at margin) |
|---|---|---|---|---|---|---|---|---|
| reference image (8 / 8) | full | 0.2195 | 0.0549 | 0.0280 | 0 | 0 | 2/8 / 2/8 | not met (max dp) |
| reference image (8 / 8) | cached | 0.2195 | 0.0549 | 0.0280 | 0 | 0 | 2/8 / 2/8 | not met (max dp); identical to full (max dp 0.000000) |
| dev16 image (16 / 16) | full | 0.1024 | 0.0316 | 0.0265 | 1 (`97974bf7`, reference margin 0.047) | 0 | 7/16 / 7/16 | not met (max dp, by 0.0024) |
| video (1 / 1) | full | 0.0671 | 0.0671 | 0.0671 | 0 | 0 | 1/1 / 1/1 | met |
| video (1 / 1) | cached | 0.0671 | | | 0 | 0 | | met; identical to full (0.000000) |

Per reference record (TT full minus CPU bf16, `scripts/parity_compare.py` per question; the TT rows carry the per-record timing):

| Record | Tokens (request / image rows) | max dp | Options that move | Reference argmax and margin | Tower s (warm or first) | Device s | End to end s |
|---|---|---|---|---|---|---|---|
| `6900e8fa` | 428 / 209 | 0.024 | D +0.024, A -0.023 | E, 0.357 | 0.125 (first of grid) | 0.648 | 2.79 (first record: head and processor load) |
| `eb937eba` | 482 / 252 | 0.006 | | A, 0.897 | 1.020 (first of grid) | 1.261 | 1.31 |
| `2a92d6c1` | 510 / 266 | 0.220 | C -0.220 (0.768 to 0.548), A +0.213 | C, 0.624 | 2.270 (first of grid) | 2.512 | 2.57 |
| `848089ea` | 736 / 500 | 0.059 | D -0.059, B +0.046 | D, 0.196 | 0.099 | 0.511 | 0.58 |
| `c4ea611f` | 461 / 209 | 0.059 | C +0.059, D -0.031 | C, 0.118 | 0.080 | 0.320 | 0.38 |
| `6c1478b4` | 511 / 266 | 0.032 | B -0.032, D +0.025 | B, 0.337 | 0.082 | 0.324 | 0.39 |
| `2a7ddcfe` | 330 / 80 | 0.024 | D -0.024 | D, 0.526 | 0.076 | 0.311 | 0.36 |
| `e000b5c0` | 478 / 234 | 0.015 | D -0.015, A +0.012 | D, 0.494 | 0.082 | 0.321 | 0.38 |

dev16: 13 of 16 records within 0.065; `7b7ef383` 0.1024 (A, margin 0.355, argmax kept), `97974bf7` 0.0717 (C to E on a 0.047 margin), `c5abc159` 0.0643. The video record: D 0.647 (CPU) versus 0.714 (TT), label D, correct on both.

### Attribution: the deltas come from the vision tower, not from the backbone

Run of 11:11 ET (`/home/hous/dev/clef/logs/stage2_image_parity_reftower.log`, `/home/hous/dev/clef/reports/stage2_reftower_image_parity.json`, `stage2_reftower_parity_<set>_full.md`): the same engine with `--vision-source reference`, which replaces the device tower output of each record by the author's CPU bf16 image features (`vision_ref/<record_id>.pt`, `merged`) before the merge, so the device runs only the backbone and the head on the reference features.

| Set | Device tower (11:06 ET) max / mean dp | Reference tower rows on the device backbone max / mean dp | Gate with reference rows |
|---|---|---|---|
| reference image (8) | 0.2195 / 0.0549 | 0.0388 / 0.0181 | met |
| dev16 image (16) | 0.1024 / 0.0316 | 0.0883 / 0.0217 | met |
| video (1) | 0.0671 | 0.0035 | met |

Per record: `2a92d6c1` 0.2195 to 0.0327, `7b7ef383` 0.1024 to 0.0341, `848089ea` 0.0592 to 0.0120, `c4ea611f` 0.0590 to 0.0157; the device tower rows of those records have PCC 0.99709, 0.99681, 0.99830 and 0.99581 against the reference rows (`tower_pcc_vs_reference` in the summary). The device backbone on reference image features is within the stage 1 text parity envelope (text: max dp 0.0722, mean 0.0109). The HF tower itself is stable: rebuilt in float32 it matches the bf16 reference rows at PCC 0.9995 to 1.0000 on six images (`/home/hous/dev/clef/reports/stage2_hf_tower_floor.json`), while the device rows are at 0.9896 to 0.9983 against either, 10 to 100 times the bf16 floor. The reverse experiment (device tower rows into the CPU bf16 backbone) is below.

Where the tower precision is fixed upstream (`/home/hous/dev/clef/tt-metal/models/demos/blackhole/qwen36/tt/vision/`): `wqkv` and its bias are `bfloat8_b` (`vision_attention.py` lines 187, 213); q and v are typecast to `bfloat8_b` before the SDPA (lines 370, 376; k follows `kv_cache_dtype`, bf16 under the accuracy preset); the `wo` output takes `activation_dtype or bfloat8_b` (line 412; `activation_dtype` is None under the default accuracy preset, so bfp8 enters the residual stream after every attention); `linear_fc1` and `linear_fc2` are `bfloat8_b` (`vision_mlp.py` lines 73, 100) with `HIFI2_FP16` (lines 137 to 150); the merger `fc1`, `fc2` and `wo` take the `dtype` argument (bf16 here). The SDPA and QKV matmuls already run `HIFI4` under the accuracy preset for this model name (`models/tt_transformers/tt/model_config.py` lines 275 to 290). The only wrapper-side lever is the `ACTIVATION` tensor group of the preset (`CLEF_VISION_ACT_BF16=1`, `tower_precision()` in `tt/vision.py`: bf16 `wo` output and QKV output); its result is in the subsection below. Lifting the rest (bf16 `wqkv`, `fc1`, `fc2`, no bfp8 typecast of q and v) is a `qwen36` vision change and is not authorized without a report back; this README is that report.

### Timing and memory (64 layers, `/home/hous/dev/clef/reports/stage2_image_parity.json`)

| Item | Value |
|---|---|
| Engine load: host state dict / 64-layer build / vision tower (HF visual on host, TT blocks and merger, merge buffers) / total | 2.9 s / 90.0 s / 9.5 s / 102.4 s |
| DRAM free per device: after the text weights / after the tower / after 4 snapshot slots / at the end of the run | 15.27 GiB / 14.96 GiB (tower 0.31 GiB) / 11.90 GiB / 11.85 GiB |
| Snapshot slots | 4 of 4 requested (max fit 18 at a 2 GiB reserve) |
| Vision tower per image, warm (same grid seen before in the process): 80 / 209 / 234 / 266 / 500 / 525 image tokens (320 / 836 / 936 / 1064 / 2000 / 2100 patches, 2048 or 4096 rows) | 0.076 / 0.080 / 0.082 / 0.082 / 0.098 / 0.202 s |
| Vision tower per image, first image of a grid in the process (program compile for that window) | 1.0 to 2.3 s at 2048 rows; 9.4 s for the first 4096-row image |
| Video (4 frames, 160 tokens, 640 patches) | 0.073 s |
| `probs_for_request` device part (tower, backbone, read-out) per image record, warm | 0.31 to 0.65 s (330 to 778 tokens) |
| End-to-end image record latency, warm (encode, device, head) | 0.35 to 0.65 s |
| Cached path on a hit (4 layers, `stage2_engine_media_l4.json`): device / end to end | 0.018 s / 0.060 s (image), 0.041 s / 0.087 s (video) |
| HF CPU bf16 reference per image record, 8 threads | 7.9 to 27.4 s |

The per-grid compile is the one timing finding: a new `cu_window_seqlens` value set compiles the windowed SDPA program (and the slice programs) once per process; `tests/test_vision.py` shows the same (first call 11.3 s in a cold process, 0.2 s with the kernel binaries cached on disk, 0.08 s warm). The server (stage 3) can warm the grids of its expected image sizes or keep the compile as a first-request cost; the upstream unmasked forward has no window and compiles once per padded row count, but fails the PCC bars (see "Device results").

Reverse experiment result (11:16 to 11:53 ET, work log; `scripts/vision_swap_cpu.py`, `/home/hous/dev/clef/reports/stage2_cpuswap_<set>.jsonl`): the device tower rows fed into the CPU bf16 backbone reproduce the device deltas against the CPU reference (reference set max |dp| 0.2601 on `2a92d6c1`, mean 0.0609; dev16 0.1390 on `7b7ef383`, mean 0.0398; video 0.0871), and differ from the device run by max 0.0406 / 0.0806 / 0.0199 (mean 0.0204 / 0.0250), which is the backbone's own contribution. Both directions agree on all three sets.

### Open items

1. Superseded by the amendment (12:10 ET): the reference set is met (0.0463, 0 flips) with the `qwen36` vision precision arguments. Remaining: dev16 record `7b7ef383` at max dp 0.1754 (A kept, reference margin 0.355) after the change (0.1024 before, 0.0591 with the act-bf16 knob alone, 0.0341 with the author's image features); 15 of 16 dev16 records are at 0.052 or below. Next step (not run, per the orchestrator): measure that record's tower PCC after the change (`vision_ref/7b7ef383...pt` exists; the pre-change value was 0.99681) and, if the tower rows are at the floor, treat it as a sensitive record like the stage 1 text residual. With the author's image features on the device backbone both sets pass (0.0388 / 0.0883). The cause is the fixed bfp8 precision inside the upstream vision modules (`wqkv`, `fc1`, `fc2` weights; q and v typecast to bfp8 before the SDPA; bfp8 `wo` output under the default preset; `HIFI2_FP16` MLP), which puts the tower at PCC 0.9896 to 0.9983 against a reference whose own bf16 floor is 0.9995 or better. Proposed `qwen36` vision change for authorization (not made): a dtype argument on `VisionAttention` (`wqkv`, the q and v typecasts) and `VisionMLP` (`fc1`, `fc2`, fidelity) that `ClefVisionArgs` can set to bf16 and HiFi4, defaults unchanged for Qwen3.6; the tower is 461 M parameters, so bf16 weights cost 0.45 GiB more per device and about 2x the warm time (0.08 to 0.2 s per image). `CLEF_VISION_ACT_BF16=1` (wrapper only) is the partial lever; see the precision-knob subsection.
2. Per-grid program compile: the first image of a grid costs 1.0 to 2.3 s (9.4 s at 4096 rows; 1.0 to 1.3 s and 6.2 s after the dtype change, which compiled new programs) in a process. Stage 3 should warm the common grids or document the first-request cost. The compile count is bounded by the number of distinct `cu_window_seqlens` value sets (one per grid and per frame count).
3. One media kind per request (images or videos, not both) because `Qwen36Model` stashes one grid; several images of different grids in one request go through `ClefVision.forward` as one `[sum N_i, 5120]` row block. Exercised in the stage 2 remediation on a two-image record (grids (1, 16, 20) and (1, 22, 38)): parity max dp 0.034, cached equals full; see "P2: several images per request" below.
4. `ClefVisionArgs(max_seq_len=4096)`: measured in the stage 2 remediation ("P2: image size capability" below); no vision op reads `max_seq_len`, every size up to 16384 patches (4096 image tokens) passes the tower, the full and the cached forward with 4 slots, and the limit above that is the engine's 16384-token request bound, not the tower.
5. The per-grid compile cost of open item 2 is a per-box first-time cost, not a per-process cost: the 12:02 ET parity run, in a fresh process after the kernel binaries of the earlier runs were on disk, saw 0.08 to 0.22 s on every image and 0.90 s on the 4096-row image. A fresh box (or a cleared kernel cache) pays the 1 to 9 s once per grid.

### Tower precision knobs (device results per knob, 12:01 ET)

Three bounded tower runs of `tests/test_vision.py -k device` with the same reference files. Results: `/home/hous/dev/clef/reports/stage2_vision_tower.json` (bf16, the gate run), `stage2_vision_tower_bfp8.json` (`CLEF_VISION_DTYPE=bfp8`), `stage2_vision_tower_actbf16.json` (`CLEF_VISION_ACT_BF16=1`); logs `/home/hous/dev/clef/logs/stage2_vision_device_{run2,bfp8,actbf16}.log`. In every run the attention K is bf16 and the SDPA and QKV matmuls run HiFi4 (the `tt_transformers` accuracy preset for this model name); `wqkv`, `fc1`, `fc2` are bfp8 and q, v are typecast to bfp8 before the SDPA in every run (hard-coded upstream).

| Knob | Configurable weights (`wo`, merger `fc1`, `fc2`) | Attention output dtype (`wo` output, QKV output) | Merged PCC: 80 / 209 / 234 / 500 tokens, video 160 | Block 0 / 12 / 23 / 26 PCC, worst of the 4 images | Warm s per image | Bars |
|---|---|---|---|---|---|---|
| bf16 (stage 2 default so far) | bf16 | bfp8 (`activation_dtype` None) | 0.98901 / 0.99564 / 0.99789 / 0.99780, 0.99427 | 0.99945 / 0.99638 / 0.99762 / 0.99051 | 0.081 to 0.125 | met |
| `CLEF_VISION_DTYPE=bfp8` (upstream default) | bfp8 | bfp8 | 0.99627 / 0.99623 / 0.99821 / 0.99764, 0.99611 | 0.99927 / 0.99204 / 0.99716 / 0.99020 | 0.102 to 0.134 | met |
| `CLEF_VISION_ACT_BF16=1` (selected) | bf16 | bf16 | 0.99655 / 0.99632 / 0.99890 / 0.99868, 0.99859 | 0.99982 / 0.99614 / 0.99887 / 0.99675 | 0.103 to 0.153 | met |

`CLEF_VISION_ACT_BF16=1` raises every tap and every merged row and is the knob of the final parity run below. It is wrapper-only: `tower_precision()` in `tt/vision.py` hands `VisionModelArgs` a `DecodersPrecision` whose `ACTIVATION` group is bf16, which `VisionAttention` reads at `vision_attention.py` lines 117, 307 and 412. The bfp8 row shows a second effect: with bfp8 `wo` and merger the block taps are lower (block 12 0.9920 versus 0.9964 on the 80-token image) but the merged rows are higher on three of four images and the video, so the bf16 merger loses precision after the blocks. The merger ops are `patch_merger.py` lines 56 to 74 of `forward`: `ttnn.linear(x_norm, w1, bias=b1, activation="gelu", compute_kernel_config=compute_kernel_config_hifi2_fp16)` then `ttnn.linear(w1_out, w2, bias=b2, compute_kernel_config=compute_kernel_config_hifi2_fp16)` plus the reduce-scatter; `compute_kernel_config_hifi2_fp16` is HiFi2 with `fp32_dest_acc_en=False` and `packer_l1_acc=True` (`models/tt_transformers/tt/model_config.py` lines 936 to 941). The suspect is therefore the fp16 destination accumulation of the two merger matmuls over K = 4608 (fc1) and K = 2304 per device (fc2), not the weight dtype. Not changed (qwen36 code); open item 1 lists it with the proposed change.

### Final parity run with the selected knob (64 layers, `CLEF_VISION_ACT_BF16=1`, 12:02 ET)

Command: as the 11:06 ET run with `CLEF_VISION_ACT_BF16=1 ... --cached reference,dev16,video --tag stage2_actbf16`. Log `/home/hous/dev/clef/logs/stage2_image_parity_actbf16.log`; summary `/home/hous/dev/clef/reports/stage2_actbf16_image_parity.json`; rows `stage2_actbf16_tt_<set>_<mode>.jsonl`; parity `stage2_actbf16_parity_<set>_<mode>.{json,md}`.

| Set | Path | Before the knob (11:06 ET): max / mean dp, flips at margin | With the knob: max / mean / median dp, argmax flips, flips at margin | Cached versus full | Gate (max dp <= 0.10, 0 flips at margin) |
|---|---|---|---|---|---|
| reference image (8) | full and cached | 0.2195 / 0.0549, 0 | 0.1107 / 0.0449 / 0.0438, 0, 0 | identical (0.000000) | not met: max dp by 0.0107 on one record; flips met |
| dev16 image (16) | full and cached | 0.1024 / 0.0316, 0 | 0.0810 / 0.0220 / 0.0159, 1 (`97974bf7`, reference margin 0.047), 0 | identical (0.000000) | met |
| video (1) | full and cached | 0.0671, 0 | 0.0312, 0, 0 | identical (0.000000) | met |

Per reference record, before and with the knob (max dp): `6900e8fa` 0.024 to 0.036, `eb937eba` 0.006 to 0.011, `2a92d6c1` 0.220 to 0.111, `848089ea` 0.059 to 0.054, `c4ea611f` 0.059 to 0.077, `6c1478b4` 0.032 to 0.014, `2a7ddcfe` 0.025 to 0.052, `e000b5c0` 0.015 to 0.006. dev16: `7b7ef383` 0.102 to 0.059, `c5abc159` 0.064 to 0.013, `97974bf7` 0.072 to 0.081 (argmax C to E on a 0.047 margin in both runs). Accuracy against the labels is unchanged (reference 2/8, dev16 7/16, video 1/1, the CPU values). Timing in this run (host under load from the stage 1 remediation jobs): engine load 171 s (build 150 s, tower 15.8 s), tower 0.08 to 0.22 s per image (0.90 s at 4096 rows), device 0.36 to 0.95 s per record, end to end 0.45 to 1.5 s; DRAM free per device 15.27 / 14.96 / 11.90 GiB after the text weights / the tower / 4 slots, 11.85 GiB at the end.

### Gate statement (plan, stage 2)

| Bar | Result | Status |
|---|---|---|
| Tower merged PCC >= 0.97 per image (4 images) | 0.9966, 0.9963, 0.9989, 0.9987 with the selected knob (0.9890 to 0.9978 before) | met |
| Tower blocks 0, 12, 23 >= 0.99 and block 26 >= 0.85 | worst 0.99982 / 0.99614 / 0.99887 / 0.99675 with the knob (0.99945 / 0.99638 / 0.99762 / 0.99051 before) | met |
| Video through the same path, image bars | merged 0.9986 (0.9943 before); parity 0.0312, 0 flips | met |
| Image records, 0 flips at margin >= 0.05 | reference 0, dev16 0, video 0 | met |
| Image records, max dp <= 0.10 | reference 0.1107 (one record; 0.2195 before), dev16 0.0810 (0.1024 before), video 0.0312 | not met on the reference set by 0.0107; met on dev16 and the video |
| Cached path equals the full path for media | 0.000000 on all three sets (64 layers) and in `tests/test_engine_media.py` (4 layers, miss, hit, hit after other media) | met |
| Text-only behaviour unchanged | bit-equal at 4 layers with and without the tower | met |

The tower is on device, all tower PCC bars and the flip rule are met, and the remaining gap is a single reference record 0.0107 above the max dp bar, attributed to the tower's fixed bfp8 precision (open item 1) and recoverable with the proposed `qwen36` change: with the author's image features the same backbone gives 0.0388 on that set.

## Plan amendment (12:10 ET): tower precision arguments in `qwen36` vision

Authorized by the orchestrator after the attribution above. Diff (`git -C /home/hous/dev/clef/tt-metal diff --stat -- models/demos/blackhole/qwen36/tt/vision`): `vision_model_config.py` +4, `vision_attention.py`, `vision_mlp.py`, `patch_merger.py`; 50 insertions, 16 deletions, no comments. Defaults keep the upstream behaviour byte-identical:

| Module | New argument (constructor) | Falls back to | Default value (upstream behaviour) |
|---|---|---|---|
| `VisionAttention` | `weight_dtype`, `sdpa_dtype` | `args.vision_weight_dtype`, `args.vision_sdpa_dtype` | bfp8 `wqkv` and bias; q and v typecast to bfp8 before the SDPA (`_to_sdpa_dtype`, no op when the dtype already matches) |
| `MLP` (vision) | `weight_dtype`, `compute_kernel_config` | `args.vision_weight_dtype`, `args.vision_mlp_compute_kernel_config` | bfp8 `fc1` (bfp4 under `bfp4_mlp`) and `fc2`; `lofi` for bfp4 else `hifi2_fp16` |
| `PatchMerger` | `compute_kernel_config` | `args.vision_merger_compute_kernel_config` | `hifi2_fp16` for both merger matmuls |
| `VisionModelArgs` | the four `vision_*` attributes | | bfp8, bfp8, None, None |

`ClefVisionArgs(precision="accuracy")` (env `CLEF_VISION_PRECISION`, default `accuracy`; `upstream` restores the old values) sets bf16 weights, bf16 q and v into the SDPA (no typecast), and `compute_kernel_config_hifi4` (HiFi4, `fp32_dest_acc_en=True`) for the MLP and merger matmuls; `CLEF_VISION_ACT_BF16=1` (bf16 attention output) stays on. Two tests in `tests/test_vision.py`: `test_device_upstream_defaults_reproduce_saved_rows` (a tower with `precision="upstream", activation_bf16=False` reproduces the 80-token rows dumped by the pre-change code at 11:11 ET: bit-equal, max abs 0.0, PCC 0.98901) and `test_device_precision_improves_80_token_image` (0.998804 against 0.98901). Run of 12:10 ET: `7 passed in 48.16 s` (`/home/hous/dev/clef/logs/stage2_vision_device_precision.log`, `/home/hous/dev/clef/reports/stage2_vision_tower_precision.json`).

Tower PCC before and after (same reference files):

| Record | bf16 run (11:00 ET, pre-change code) | act-bf16 knob only (12:01 ET) | precision change (12:10 ET) | Blocks 0 / 12 / 23 / 26 after | Warm s after (before) |
|---|---|---|---|---|---|
| `2a7ddcfe` (80 tokens) | 0.98901 | 0.99655 | 0.99880 | 0.99993 / 0.99977 / 0.99962 / 0.99853 | 0.093 (0.081) |
| `6900e8fa` (209) | 0.99564 | 0.99632 | 0.99822 | 0.99995 / 0.99856 / 0.99939 / 0.99694 | 0.109 (0.094) |
| `e000b5c0` (234) | 0.99789 | 0.99890 | 0.99909 | 0.99993 / 0.99911 / 0.99934 / 0.99831 | 0.113 (0.097) |
| `848089ea` (500) | 0.99780 | 0.99868 | 0.99898 | 0.99988 / 0.99959 / 0.99942 / 0.99798 | 0.160 (0.125) |
| video (160, 4 frames) | 0.99427 | 0.99859 | 0.99839 | | 1.076 first video call of the process |

The upstream-defaults tower built in the same process reads bfp8 `wqkv`, bfp8 SDPA inputs, bfp8 `fc1`, HiFi2 MLP and merger, attention activation None, exactly the pre-change configuration.

### Image parity before and after the change (64 layers, full and cached)

Run of 12:12 ET: `/home/hous/dev/clef/logs/stage2_image_parity_precision.log`, `/home/hous/dev/clef/reports/stage2_precision_image_parity.json`, rows `stage2_precision_tt_<set>_<mode>.jsonl`, parity `stage2_precision_parity_<set>_<mode>.{json,md}`. Same command as 11:06 ET with `--cached reference,dev16,video --tag stage2_precision` (the accuracy precision is the default).

| Set | pre-change code (11:06 ET) max / mean dp, argmax flips, flips at margin | act-bf16 knob only (12:02 ET) | precision change (12:12 ET) | Cached vs full (12:12 ET) | Gate (max dp <= 0.10, 0 flips at margin) |
|---|---|---|---|---|---|
| reference image (8) | 0.2195 / 0.0549, 0, 0 | 0.1107 / 0.0449, 0, 0 | 0.0463 / 0.0200, 0, 0 | 0.000000 | met |
| dev16 image (16) | 0.1024 / 0.0316, 1, 0 | 0.0810 / 0.0220, 1, 0 | 0.1754 / 0.0318, 0, 0 | 0.000000 | not met: max dp on one record (`7b7ef383`); flips met |
| video (1) | 0.0671, 0, 0 | 0.0312, 0, 0 | 0.0126, 0, 0 | 0.000000 | met |

Per reference record (max dp, pre-change / act-bf16 / precision change): `6900e8fa` 0.024 / 0.036 / 0.037, `eb937eba` 0.006 / 0.011 / 0.004, `2a92d6c1` 0.220 / 0.111 / 0.030, `848089ea` 0.059 / 0.054 / 0.025, `c4ea611f` 0.059 / 0.077 / 0.046, `6c1478b4` 0.032 / 0.014 / 0.009, `2a7ddcfe` 0.025 / 0.052 / 0.007, `e000b5c0` 0.015 / 0.006 / 0.003. The reference set now sits where the author's own image features put the device backbone (0.0388 at 11:11 ET). dev16: 14 of 16 records at 0.052 or below; `97974bf7` (the 0.047-margin record) 0.072 / 0.081 / 0.033 and now keeps the reference argmax; `7b7ef383` (A, reference margin 0.355, argmax kept) 0.102 / 0.059 / 0.175 is the one record above the bar; with the author's image features that record gave 0.034, so its device tower rows are the remaining difference; its tower PCC was 0.99681 with the pre-change code and is not measured after the change (the orchestrator's instruction was to stop after this run). Accuracy against the labels is unchanged on every set.

Deltas of the change (12:12 ET run against 11:06 ET; the host was under load from stage 1 jobs in both 12:xx runs, so the build time is not comparable):

| Item | Pre-change | After the change |
|---|---|---|
| DRAM free per device after the tower (tower cost) | 14.96 GiB (0.31 GiB) | 14.77 GiB (0.50 GiB) |
| DRAM free after 4 slots | 11.90 GiB | 11.72 GiB |
| Tower build at engine start | 9.5 s | 12.0 s |
| Tower per image, warm, 80 to 525 tokens | 0.076 to 0.202 s | 0.088 to 0.219 s |
| First image of a grid in the process (new programs after the dtype change) | 1.0 to 2.3 s; 9.4 s at 4096 rows | 1.0 to 1.3 s; 6.2 s at 4096 rows |
| Device time per image record, warm | 0.31 to 0.65 s | 0.38 to 0.74 s |
| End to end per image record, warm | 0.35 to 0.65 s | 0.45 to 0.83 s |
| Video record (160 tokens) tower / device / end to end | 0.073 / 0.31 / 0.36 s | 0.082 / 0.39 / 0.54 s |

### Gate statement after the amendment (plan, stage 2)

| Bar | Result | Status |
|---|---|---|
| Tower merged PCC >= 0.97 (4 images) | 0.99880, 0.99822, 0.99909, 0.99898 | met |
| Tower blocks 0, 12, 23 >= 0.99 and block 26 >= 0.85 | worst 0.99988 / 0.99856 / 0.99934 / 0.99694 | met |
| Video through the same path | tower 0.99839; parity 0.0126, 0 flips, cached identical | met |
| Image records, 0 flips at margin >= 0.05 | reference 0, dev16 0, video 0 | met |
| Image records, max dp <= 0.10: reference set | 0.0463 (mean 0.0200) | met |
| Image records, max dp <= 0.10: dev16 (reported, not the plan's gate set) | 0.1754 on `7b7ef383`, 15 of 16 records <= 0.052 | not met on one record |
| Image records, max dp <= 0.10: video | 0.0126 | met |
| Cached path equals the full path for media | 0.000000 on all 25 records, both runs | met |
| Upstream behaviour byte-identical with default arguments | pre-change rows reproduced bit for bit (test) | met |
| Text-only behaviour unchanged | l4 with the tower attached (12:17 ET): all PCC fields and all 6 record rows bit-equal to the `CLEF_VISION=0` baseline (max dp 0.0), `8 passed` | met |

## Stage 2 remediation (2026 Oct 05, from 16:55 ET)

Tasks from `/home/hous/dev/clef/reports/review_stage2.md`: P1-1 (the dev16 `7b7ef383` regression), P2 (image size capability), P2 (several images per request), and the "Other concerns". Logs `/home/hous/dev/clef/logs/stage2r_*.log`; reports `/home/hous/dev/clef/reports/stage2r_*`. `tt/engine.py` and the `qwen36` files were not changed. Every device run used `/home/hous/dev/clef/bin/devrun timeout <s> ...` with `CLEF_TRACED=0`, `OMP_NUM_THREADS=8`, the snapshot in `CLEF_MODEL` and `HF_MODEL`.

### `describe()` field (Other concerns)

`ClefVision.describe()` no longer reports the stale constant list `fixed_bfp8`. The new field `bfp8_weights` is derived from the module tensors at call time (`ClefVision.bfp8_weights()` in `tt/vision.py`): it lists which of `attention_wqkv`, `attention_wqkv_bias`, `attention_wo`, `feed_forward_linear_fc1`, `feed_forward_linear_fc2`, `merger_fc1`, `merger_fc2` have dtype `bfloat8_b` on block 0 and the merger. Under the accuracy precision the list is empty; under `precision="upstream"` it lists the four fixed-bfp8 tensors of the pre-change code. Earlier JSON files (`stage2_vision_tower_precision.json`, `stage2_precision_image_parity.json`) still carry the old `fixed_bfp8` key next to `wqkv_dtype: BFLOAT16`; the per-tensor dtype fields in those files are the correct ones.

### Upstream handoff: pad rows attend in the unmodified `qwen36` vision tower (Other concerns)

Defect. The unmodified upstream forward lets every real patch attend to zero-padded rows. Files (worktree `/home/hous/dev/clef/tt-metal`, branch `hous/clef-bringup`, under `models/demos/blackhole/qwen36/tt/vision/`):

- `model.py` line 232 (`DropInVisionTransformer.forward`): `seq_len = ((unpadded_seq_len // 2048) + 1) * 2048` pads every image to the next multiple of 2048 rows (a 2048-patch image pads to 4096, an off-by-one that only costs time). The pad rows are zeros (`prepare_input`, line 115).
- `model.py` lines 282 to 286: `self.tt_model(tt_input, unpadded_seq_len=..., rot_mats=rot_mats)` passes no `cu_window_seqlens`.
- `model.py` lines 139 to 143 (`VisionTransformer.forward`): `x = block(x, rot_mats=rot_mats)` for every block, again without `cu_window_seqlens`, although `VisionBlock.forward` (`vision_block.py` lines 85 to 110) and `VisionAttention.forward` (`vision_attention.py` lines 37 to 55) accept and forward it.
- `vision_attention.py` lines 393 to 402 (`forward_prefill`): `ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False, cu_window_seqlens=cu_window_seqlens, ...)` therefore runs with `cu_window_seqlens=None`, that is unmasked over all padded rows. After `norm1` (LayerNorm) a zero row equals `norm1.bias`, so the pad rows contribute a constant nonzero key and value that every real query attends to; the error grows with the pad fraction.

Numbers (HF CPU bf16 tower rows as the reference, PCC on the merged `[N, 5120]` rows; `/home/hous/dev/clef/reports/stage2_vision_tower_nomask.json` with `CLEF_VISION_PAD_MASK=0`, equal to six decimals to the `pcc_upstream_unmasked` field of `stage2_vision_tower.json`, which runs the unmodified `DropInVisionTransformer.forward` itself):

| Image | Patches / rows | Pad rows | Upstream forward (unmasked) merged PCC | Unmasked block 0 / 12 / 23 / 26 | Masked wrapper merged PCC |
|---|---|---|---|---|---|
| `2a7ddcfe` | 320 / 2048 | 84 % | 0.894523 | 0.94673 / 0.74469 / 0.93467 / 0.80906 | 0.98901 |
| `6900e8fa` | 836 / 2048 | 59 % | 0.949746 | 0.98344 / 0.95258 / 0.97326 / 0.91264 | 0.99564 |
| `e000b5c0` | 936 / 2048 | 54 % | 0.961537 | 0.98754 / 0.86657 / 0.97214 / 0.92037 | 0.99789 |
| `848089ea` | 2000 / 2048 | 2 % | 0.99462 | 0.99815 / 0.98942 / 0.99605 / 0.98812 | 0.99780 |

Three of four images fall below the 0.97 merged bar and block 12 falls to 0.74 on the smallest image. The upstream unit tests (`tests/test_vision_block.py`, random inputs, PCC per block) cannot see this because PCC is scale invariant and the random inputs have no pad rows. The masked column is the pre-change wrapper (bf16 configurable weights, upstream bfp8 fixed weights); with the precision arguments the masked tower is at 0.9988 to 0.9991 on these images.

Proposed upstream fix (not made here; `qwen36` is outside the stage's allowed changes apart from the authorized precision arguments): in `DropInVisionTransformer.forward` build `cu_window_seqlens` as the per-frame `cu_seqlens` that `qwen3_5_vision_transformer_preprocess` already returns (`functional.py` lines 167 to 175) plus the pad bound, that is `[0, h*w, 2*h*w, ..., unpadded_seq_len, seq_len]` as an int32 row-major replicated tensor, and pass it through `VisionTransformer.forward` to every block. The SDPA kernel accepts it (`/home/hous/dev/clef/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp` line 337; validation in `device/sdpa_device_operation.cpp` lines 435 to 475: int32 or uint32, 1-D, 2 to 1024 entries, no tile alignment). This is exactly what `window_bounds` in `tt/vision.py` (lines 114 to 124) does, and it also fixes the video case (HF attends within each temporal patch group, `transformers/vision_utils.py` `get_vision_cu_seqlens`; one window over all frames gave 0.934 on the 4-frame clip, per-frame windows give 0.994). Cost: one extra tensor upload per image and one windowed SDPA program per distinct `cu_window_seqlens` value (first compile 1 to 2 s per grid at 2048 rows, 6 to 9 s at 4096). Bounded repro for the owners: `tests/test_vision.py -k device` in this autoport with `CLEF_VISION_PAD_MASK=0` (the five cases fail the bars, 15 s) against the default run (they pass), the two JSON files above, and the HF reference files `/home/hous/dev/clef/reports/reference/vision_ref/<record>.pt` (reproducible from `scripts/vision_reference.py` on the 8 reference images).

### P1-1: dev16 record `7b7ef383`, classification

Record: grid (1, 24, 32), 768 patches, 192 image tokens, 447 request tokens, reference A 0.5778 / C 0.2228 (margin 0.355, label A). Runs: tower test with three towers in one process (`/home/hous/dev/clef/reports/stage2r_vision_tower.json`, log `stage2r_vision_device.log`, `8 passed`), dev16 parity at 64 layers under the accuracy and the upstream precision with the device rows dumped (`stage2r_dev16_{accuracy,upstream}_image_parity.json`, rows dirs `stage2r_device_vision_rows_{accuracy,upstream}/`), the CPU swaps and the perturbation series (`stage2r_cpuswap_7b7ef383_*.jsonl`), the fp32 CPU reference (`stage2r_ref_7b7ef383_fp32.jsonl`), and the collected report `/home/hous/dev/clef/reports/stage2r_7b7ef383_classification.json`. Details and commands: work log, 12:57 to 13:17 ET entries.

Tower on this image (PCC against the HF CPU bf16 rows; `comp_pcc`; blocks 0 / 12 / 23 / 26):

| Tower | Merged PCC | Blocks | Against the HF fp32 tower | Offset norm (row mean of the error) |
|---|---|---|---|---|
| accuracy (default) | 0.999055 | 0.99993 / 0.99916 / 0.99949 / 0.99756 | 0.99905 | 1.45 |
| act-bf16 only (upstream weights) | 0.997728 | 0.99986 / 0.99909 / 0.99916 / 0.99608 | 0.99775 | |
| upstream defaults | 0.996658 | 0.99947 / 0.99811 / 0.99816 / 0.99046 | 0.99647 | 2.46 |
| HF bf16 against HF fp32 | 0.99990 | | | 0.08 |

The accuracy tower is at or above the four gate images on this record (0.99822 to 0.99909) and the three configurations improve monotonically; the device rows are bit-identical between the tower test and the 64-layer parity process. The tower is not the low element.

Backbone (probabilities for A / C; max |dp| against the bf16 reference):

| Rows into which backbone | A / C | max dp |
|---|---|---|
| HF bf16 rows, CPU bf16 backbone (reference) | 0.578 / 0.223 | 0 |
| HF bf16 rows, CPU fp32 backbone and fp32 tower | 0.542 / 0.253 | 0.036 |
| HF bf16 rows, device backbone (11:11 ET) | 0.544 / 0.248 | 0.034 |
| accuracy device rows, device backbone (12:12 ET, reproduced bit for bit at 12:59 ET) | 0.436 / 0.398 | 0.175 |
| accuracy device rows, CPU bf16 backbone | 0.499 / 0.325 | 0.102 |
| upstream device rows, device backbone (11:06 ET, reproduced bit for bit at 13:03 ET) | 0.680 / 0.156 | 0.102 |
| upstream device rows, CPU bf16 backbone | 0.717 / 0.125 | 0.139 |
| HF rows plus Gaussian noise at the device PCC (3 seeds), CPU backbone | | 0.003 to 0.017 |
| HF rows plus the device error, half, reversed | | 0.102, 0.088, 0.121 (A 0.699) |
| HF rows plus the row-mean offset of the device error only (18 % of the error energy) | 0.451 / 0.361 | 0.139 |
| HF rows plus the zero-mean residual only | 0.560 / 0.272 | 0.049 |
| HF rows plus the mean offset of the other 15 dev16 records | 0.507 / 0.284 | 0.071 |
| HF rows plus the upstream run's offset only | 0.553 / 0.244 | 0.025 |

Classification: backbone sensitivity to a structured tower artifact, not a tower-precision shortfall and not a bug. (1) The CPU bf16 backbone reproduces the direction and most of the magnitude of the shift from the device rows in both configurations, so the device backbone adds only its usual 0.03 to 0.07 (text parity envelope). (2) Random noise of the device's size moves this record by at most 0.017; the device error moves it by 0.10, linearly (half gives 0.088, the reverse 0.121 the other way), because it contains a per-channel offset that is the same on every output row and nearly the same on every image (cosine 0.73 to 0.95 with the other 15 records' offsets; the other records' mean offset alone gives 0.071 here). The offset is absent from the HF bf16 tower (0.08 against fp32) and is halved by the precision change (mean norm over 16 records 1.46 to 1.00), so it is a device compute artifact. (3) The sign reversal the review flagged is explained: the upstream configuration's larger zero-mean error (bfp8 weights, bfp8 q and v, HiFi2) pushed A up on this record (its offset alone gives 0.025), the precision change removed that component, and the remaining offset pushes A down; both effects are reproduced on the CPU backbone from the rows alone. (4) The fp32 reference puts this record at A 0.542 / C 0.253, so the bf16 reference is itself 0.036 off here; against fp32 the accuracy run is at 0.145 and the author's rows on the device backbone at 0.005.

Knob: none in the wrapper. The offset source is inside the `qwen36` vision modules under exact bf16 weights; excluded by measurement or reading: the host position-add order (0.022), the HF GELU variant (an offset of 0.12 at cosine 0.56), ttnn's fused `"gelu"` mode (accurate, `unary_op_utils.cpp` line 997), `ccl_dtype` bfp8 (not applied on a (1, 2) mesh, `ccl.py` lines 168 to 190), LayerNorm weights (bf16). Remaining candidates are the SDPA kernel numerics (bf16 inputs, exp and reciprocal on the SFPU), the LayerNorm kernel statistics in bf16, and the bf16 rounding of the two reduce-scatter partials at the merger output (channel 3994 at -52 mean, bf16 ulp 0.25). The per-block taps (step 9, `stage2r_tower_rows/7b7ef383..._<tower>_taps.pt`, work log 13:31 ET) localize the offset to block 0: after the first block the device error is already 57 % a constant per-channel vector (PCC 0.999979, error std 0.0075 on a hidden std of 0.614, offset norm 0.19), it keeps its direction through blocks 12 and 23 (cosine 0.74 to 0.88 between towers) and at block 26 it sits on the hidden massive channel 514 (HF mean 1680; device offset -15 accuracy, -39 upstream), which the merger maps onto channel 3994 of the merged rows. The pad rows are excluded: with `CLEF_VISION_PAD=128` (768 rows, no window; work log 13:33 ET) the block 0 offset is 0.1985 against 0.1920 at cosine 0.998. Every bias tensor is bf16 under the accuracy config. So the offset comes from the block compute itself on exact weights (the LayerNorm kernel, the QKV matmul and rotary with the 72-to-96 head padding, the SDPA, `wo`, the MLP with the SFPU GELU); which op is not determined here. The next step is a block-0 op swap on the device (HF intermediate into the device block after each op, offset measured after each) and is stage 4 work (datatype sweep). It affects every image equally (the mean offset over the 16 dev16 records has norm 1.00 and cosine 0.86 to 0.95 with each record's own), so removing it would lower the dp of every record; on the 8-record reference set the current dp is already within the bf16 reference's own noise against fp32. Until then `7b7ef383` stays a reported, sensitive record: the dev16 max dp is 0.1754 on it under the accuracy config, 15 of 16 records are at 0.052 or below, the reference set (8 records) is at 0.0463, and the record keeps its argmax with the label.

### P2: image size capability (64 layers, 4 slots)

Script `scripts/vision_capacity.py`; JSON `/home/hous/dev/clef/reports/stage2r_default_vision_capacity.json` (1024 to 16384 patches), `stage2r_large_vision_capacity.json` (20480 to 65536 patches, step 8; see the work log for the result). The engine's `VISION_MAX_SEQ_LEN` 4096 stayed as it is; no vision op reads `max_seq_len`, so it does not bound anything. Synthetic images (gradient plus noise) at the processor's native multiples of 32 px.

| Patches (rows) | Grid | Image tokens | Request tokens | Tower warm s | Tower first-of-grid s (compile) | Full request s (device s) | Cached = full | Min DRAM free GiB |
|---|---|---|---|---|---|---|---|---|
| 1024 (2048) | 32 x 32 | 256 | 420 | 0.084 | 0.54 | 0.75 (0.41) | 0.0 | 11.70 |
| 2048 (2048) | 32 x 64 | 512 | 676 | 0.118 | 0.54 | 0.55 (0.50) | 0.0 | 11.67 |
| 4096 (4096) | 64 x 64 | 1024 | 1188 | 0.284 | 2.73 | 0.92 (0.84) | 0.0 | 11.67 |
| 6144 (6144) | 64 x 96 | 1536 | 1700 | 0.497 | 7.52 | 1.33 (1.21) | 0.0 | 11.67 |
| 8192 (8192) | 64 x 128 | 2048 | 2212 | 0.753 | 8.20 | 1.86 (1.72) | 0.0 | 11.66 |
| 10240 | 80 x 128 | 2560 | 2724 | 1.064 | 7.69 | 2.39 (2.20) | 0.0 | 11.66 |
| 12288 | 96 x 128 | 3072 | 3236 | 1.405 | 8.00 | 3.08 (2.87) | 0.0 | 11.66 |
| 14336 | 112 x 128 | 3584 | 3748 | 1.841 | 9.74 | 4.69 (4.15) | 0.0 | 11.65 |
| 16384 | 128 x 128 | 4096 | 4260 | 2.483 | 10.84 | 5.69 (5.03) | 0.0 | 11.65 |

No failure up to 16384 patches (2048 x 2048 px, 4096 image tokens): the tower, the full forward and the cached forward all pass, cached equals full, and the DRAM headroom with 4 slots never drops below 11.65 GiB (the 64-layer weights plus the tower plus 4 slots leave 11.72 GiB; a 16384-row pass uses about 0.07 GiB). The warm tower time is about 0.15 s per 1024 patches above 4096 rows; the first image of a new row count compiles once per box (7.5 to 10.8 s at 6144 rows and above). The processor permits up to 65,536 patches (16,384 tokens, 4096 x 4096 px). Step 8 (`stage2r_large_vision_capacity.json`, work log 13:21 ET) continued in 8192-patch steps:

| Patches (rows) | Grid | Image tokens | Request tokens | Tower warm s | Tower first-of-grid s | Full request s (device s) | Cached = full | Min DRAM free GiB |
|---|---|---|---|---|---|---|---|---|
| 20480 | 128 x 160 | 5120 | 5284 | 3.27 | 9.45 | 6.77 (6.07) | 0.0 | 11.66 |
| 24576 | 128 x 192 | 6144 | 6308 | 4.46 | 10.31 | 8.31 (7.82) | 0.0 | 11.66 |
| 32768 | 128 x 256 | 8192 | 8356 | 7.41 | 13.49 | 13.07 (12.43) | 0.0 | 11.65 |
| 40960 | 160 x 256 | 10240 | 10404 | 11.08 | 16.93 | 18.82 (18.02) | 0.0 | 11.64 |
| 49152 | 192 x 256 | 12288 | 12452 | 15.43 | 20.76 | 25.49 (24.56) | 0.0 | 11.63 |
| 57344 | 224 x 256 | 14336 | 14500 | 20.53 | 26.09 | 33.20 (32.08) | 0.0 | 11.62 |
| 65536 | 256 x 256 | 16384 | | fails on the host before the device: `ValueError: schema requires 16536 tokens before state; maximum is 16384` (release `encode_record`, `max_length` 16384) | | | | |

Largest passing grid: 57,344 patches ((1, 224, 256), 3584 x 4096 px, 14,336 image tokens), through the tower, the full forward and the cached forward with 4 slots, DRAM never below 11.62 GiB, no L1 or op limit reached. The first and only failure mode is the release's 16,384-token request limit (`joint_schema_model.encode_record`, the value `probs_for_request` and the release `systemone` pass): the image tokens plus the 36-token prefix, the state and the schema must fit in 16,384 tokens, so about 16,000 image tokens (64,000 patches) is the ceiling for a minimal request and less for a longer state or schema. `VISION_MAX_SEQ_LEN` (4096) is not a bound.

Recommendation for the server (stage 3): do not add an image-pixel bound as a device limit; map the encoder's `ValueError` on the token budget to 422 (it already fires on the host before any device work) and, if a latency policy is wanted, bound the image tokens per request at 4096 (16,384 patches, 2048 x 2048 px): a warm request there takes 5.7 s against 0.4 to 0.9 s for the reference cartoons (320 to 2100 patches), and above it the tower time grows quadratically (20.5 s at 57,344 rows). State the bound next to `max_len` 16,384 tokens in `doc/context_contract.json`. The first image of a new row count compiles once per box (7.5 s at 6144 rows, 26 s at 57,344), which the server should warm for the grids it admits or document as a first-request cost.

### P2: several images per request

Record `/home/hous/dev/clef/reports/reference/records_two_image.jsonl` (`two_image_2a7ddcfe_6900e8fa`): the `2a7ddcfe` cartoon (grid (1, 16, 20), 80 tokens) and the `6900e8fa` cartoon ((1, 22, 38), 209 tokens), two caption questions, labels D and B. CPU references `ref_two_image_bf16.jsonl` (15.4 s) and `ref_two_image_fp32.jsonl`; engine runs `tests/test_engine_media.py::test_device_two_image_cached_matches_full` (4 layers, `stage2_engine_media_l4.json`) and `image_parity.py --sets two_image --cached two_image` (64 layers, `stage2r_two_image_image_parity.json`).

| Item | Result |
|---|---|
| Rows and order | 289 rows at positions 37 to 327 (80 then 209, request order); `media_rows` equals the sum of the single-image counts |
| M-RoPE | two-row `image_grid_thw` through `get_rope_index`; `rope_delta` -260 after the cached schema call (4 layers) |
| Cached equals full | 0.0 on the miss, the hit, the hit after other media (4 layers); 0.000000 at 64 layers |
| Parity at 64 layers (bf16 reference) | `caption_1` D 0.856 (reference 0.844), max dp 0.012; `caption_2` B 0.295, A 0.231 (reference A 0.2615, B 0.2615, a tie), max dp 0.034; mean 0.023; 1 near-tie flip (margin below 0.0001), 0 flips at margin; against fp32 0.020 and 0.037, fp32 also puts B first |
| One-media-kind rule | not hit: two images are one kind |

The multi-image path passes the stage 2 bars; no xfail. The second question is a near tie in the joint context (the single-image reference gives E first with B second on `6900e8fa`), which the margin rule handles.

### fp32 CPU control for the image records (Other concerns)

`scripts/cpu_reference.py --dtype float32 --threads 8` on `records_image.jsonl` (plus the `7b7ef383` record and the two-image record in the same process): `/home/hous/dev/clef/reports/reference/ref_image_fp32.jsonl`, log `/home/hous/dev/clef/logs/stage2r_cpu_reference_fp32.log`, 157 GB RSS, 36 to 67 s per record, 8:52 wall for 10 records. `parity_compare.py` outputs `/home/hous/dev/clef/reports/stage2r_image_{bf16_vs_fp32,tt_precision_vs_fp32,tt_precision_vs_bf16}.{json,md}` and `stage2r_two_image_bf16_vs_fp32.{json,md}`:

| Comparison (8 reference image records, 8 questions) | max dp | mean dp | median dp | argmax flips | flips at margin |
|---|---|---|---|---|---|
| CPU bf16 against CPU fp32 | 0.0572 | 0.0265 | 0.0219 | 0 | 0 |
| TT (12:12 ET precision run) against CPU fp32 | 0.0383 | 0.0189 | 0.0176 | 0 | 0 |
| TT (12:12 ET precision run) against CPU bf16 | 0.0463 | 0.0200 | 0.0167 | 0 | 0 |

Per record, bf16-vs-fp32 / TT-vs-fp32 / TT-vs-bf16: `6900e8fa` 0.057 / 0.038 / 0.037, `eb937eba` 0.001 / 0.004 / 0.004, `2a92d6c1` 0.054 / 0.038 / 0.030, `848089ea` 0.036 / 0.018 / 0.025, `c4ea611f` 0.029 / 0.017 / 0.046, `6c1478b4` 0.013 / 0.022 / 0.009, `2a7ddcfe` 0.015 / 0.009 / 0.007, `e000b5c0` 0.008 / 0.006 / 0.003. The TT engine is closer to fp32 than the bf16 reference is on 6 of 8 records, and the bf16 reference's own noise (0.057 max) is above the TT-vs-bf16 max (0.046) on this set. `7b7ef383` (dev16): bf16 against fp32 0.036; TT accuracy against fp32 0.145. Two-image record: bf16 against fp32 0.017, the near-tie question flips (margin below 0.0001).
