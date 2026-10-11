# Prefill dispatch source diagnosis

AutoFix source investigation, 2026-10-10; not a stage review. No TTNN import, device use, or implementation edits. `dispatch_maps.json` records passing hot/random/boundary map tests at T=128 and 512. The parent reports a passing 512-token end-to-end PCC and about 17.4 ms with BFP4/4x4 versus 32.5 ms packed; this report does not independently qualify that measurement or its runtime provenance.

## Larger chunks do not require larger native activation CBs

`input_m_tiles` is a **per-expert maximum**, not the dispatched-buffer height. The reader computes `m_t=min(ceil(count[e]/32), M_T_MAX)`, then `m_blocks=ceil(m_t/8)` and loops those blocks. Compute and writer receive exactly this runtime count through the mailbox. Reducing the maximum silently truncates a hot expert; the op does not continue the omitted rows in a later call. See `moe_fused_swiglu/device/kernels/moe_fused_swiglu_reader.cpp:490-618` and `moe_fused_swiglu_compute.cpp:505-559` under `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/`.

The geometry already bounds its working set with `M_BLOCK=8` (256 token rows), `DEPTH_X=2`, and fixed weight/hidden/output FIFO depths. `m_t_max` only changes `depth_x` from 1 to 2 when the maximum exceeds 8 tiles; it does not multiply CB storage by the number of M blocks. Thus T=512/1024/2048, with `input_m_tiles=16/32/64`, have the **same CB geometry** for a fixed grid, activation layout and weight/output dtype. A smaller maximum above 8 saves no CB memory; a maximum of 8 saves one input slot but is incorrect whenever any expert has more than 256 tokens. Source: `moe_fused_swiglu_geometry.hpp:14-33`, `.cpp:184-240,319-349`.

For unique top-8 routing, `count[e]<=T`, so the safe default remains T/32. A separately dispatched sequence of input chunks is correct if each chunk recomputes counts/maps, but reloads expert weights and repeats routing/dispatch overhead; it is a performance alternative, not an L1 requirement.

| T | Routes | Safe capacity C=8T+31*256 | Per-expert tiles | Maximum native M blocks | One BF16 Cx2048 buffer |
|---:|---:|---:|---:|---:|---:|
| 512 | 4096 | 12032 | 16 | 2 | 47 MiB |
| 1024 | 8192 | 16128 | 32 | 4 | 63 MiB |
| 2048 | 16384 | 24320 | 64 | 8 | 95 MiB |

All maps, offsets and route ordinals remain exactly representable through the UINT16 layout bridge and FP32 scan at these sizes. T=1024/2048 also satisfy masked_bincount's T%64 requirement; sort widths 8192/16384 are powers of two. The prefill_chunk setup prepares these native routing buffers and new positional index vectors. Capacity, output allocations and map traffic grow; CB storage does not.

## Exact native geometry and BF8 activation adaptation

Inputs must be BF16 ROW_MAJOR or BFP8 TILE, DRAM interleaved, rank 2 or rank 4 with leading dimensions (1,1), and capacity divisible by 32. Gate/up/down weights must be matching tiled BF4/BF8/BF16 tensors, with uniform per-role accessor layouts across experts. Grid columns must be >= rows >=2 and fit the device; the factory uses the rectangle beginning at (0,0). With hidden=2048/intermediate=512, `kr_pad=ceil(64/grid_y)`, hidden tiles are split over grid_x, and `ec_max=ceil(64/(grid_x*grid_y))` must be <=8. Gate/up width must also admit the geometry's scatter lattice and <=8 DEST-tile subblocks; `choose_hn_pad()` performs that check. The familiar 4x4 and 8x2 geometries are legal. FP32 DEST, packer L1 accumulation and full DEST synchronization must remain disabled. See device validator lines79-226,270-310 and geometry lines106-182,243-288.

For 4x4, `kr_pad=16`, `hn_pad=4`, `gu_chunks=2`, `ec_max=4`, `slice_pages=8`, `gather_pages=32`; M-row/group schedules are off. For 8x2, the corresponding values are 32,2,2,4,8,16. Both keep depth_x=2, depth_h=3 and resident weights for the ordinary Blackhole budget.

Calculated CB payload per core for T>=512 and BF16 output, including two actual 1024-byte metadata scratch pages:

| Grid | Weights | BF16 RM input | BFP8 TILE input |
|---|---|---:|---:|
| 4x4 | BFP4 | 940608 B | 908928 B |
| 4x4 | BFP8 | 1038912 B | 1007232 B |
| 8x2 | BFP4 | 1112640 B | 1048192 B |
| 8x2 | BFP8 | 1210944 B | 1146496 B |

These are source-derived CB sums, not measured free L1 or allocator peaks. `program_factory.cpp:396-435` uses budget `hal::get_max_worker_l1_unreserved_size()-70656` and rejects an oversized geometry. On Blackhole that is `1536*1024 - MEM_MAP_END - 70656` (`tt_metal/hal.cpp:50`, `bh_hal_tensix.cpp:53`). The factory's preliminary `l1_bytes()` estimate uses 64-byte scratch pages, undercounting this 256-expert case by 1920 bytes; actual CB allocation uses the real page sizes. Persistent L1 tensors and actual allocation placement still need the ordinary runtime check. No shared-source change is proposed.

The parent's new adaptation is legal:

```python
dispatched = ttnn.to_layout(dispatched, ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
# Same counts, offsets, read_x_at_offset=True, T//32 limit, and BF16 output.
```

Reader lines569-599 rebase tiled input by `(offset/32)*64` tile pages. The tiled branch consumes BFP8 directly and bypasses compute's BF16-row tilization (`compute.cpp:580-602`). The RM path already packs into a BFP8 `cb_x_tiles`, so this moves quantization/tilization earlier rather than introducing the first BFP8 activation boundary. Exact numerical equivalence still needs measurement. Keep native output BF16: the inverse embedding requires a BF16 ROW_MAJOR table. BFP8 output would require conversion before it. Include the added dispatch conversion and its transient buffers in end-to-end timing. Existing native tests explicitly cover both activation layouts and larger allocated token counts; their presence is not Qwen hardware qualification (`test_moe_fused_swiglu.py:182-208`, sweep at line69).

## Tiled planner and trace lifetime

The first tiled planner correctly removed logical padding before each stateful layer call and preserved the inherited absolute-position/chunk-page-table logic. Its direct slices and variable-length `concat(pieces)` nevertheless broke the prior physical-bucket program-stability argument. Warming logical35 in a physical128 tensor does not prepare the 36-input concat for logical36. Slice hashes also contain host starts/ends and input shape (`slice_device_operation.cpp:348-377`); concat hashes its tensor sequence and selected factory (`concat_device_operation.cpp:101-134`). Nonaligned slices may additionally untilize/retilize, and concat with token-padding may use an RM fallback; the original tiled-path comment was not true for tails (`slice.cpp:235-237`, `concat.cpp:93-139`).

The parent has now added fallback to `FusedDecoder.prefill_forward` when the logical length differs from the input length, length is not divisible by128, or start_pos is not divisible by128. This addresses unseen padded valid lengths and ordinary nonaligned continuations. The inherited output assembly keeps the physical output bucket and changes only logical view metadata, avoiding variable concat signatures (`fused_decoder.py:223-281`). For page128, aligned equal physical/logical lengths produce only real-token chunks and correct continued state.

The second identified corner was larger/nondividing page geometry: page_size256/start_pos128 and page_size256/length128 could still enter the tiled path and produce token tails; page96 has no compatible 128/512/1024/2048 prefill bucket. The current code requires nonempty `aligned_buckets`, a page-aligned start, and a zero greedy remainder: beginning with `remainder=length`, it applies `remainder %= count` for each eligible bucket in descending order. This matches the actual chunk planner and also covers arbitrary prefill_chunk768/page256/length1024, whose remainder256 now selects the stable fused planner. That fallback retains the optimized layer modules. The identified planner caveat is fixed by source inspection; device qualification remains pending. Exact physical shape, batch size, page geometry and optimization policy still need prewarming before a live trace; a new physical bucket is not automatically covered.

`run_reuse.py` deliberately withholds logical36 and the continuation cut at36, then disables program-cache misses for a decode trace's entire lifetime. During this investigation the parent added `--optimization-json`, fixing the previous gap where `--decoder optimized` alone retained default `prefill_tiled=False`. Run it with the actual selected policy. Prefill outputs must be released before replay as the harness already does; allowing cache misses or exempting persistent program allocations would not prove the lifetime contract.

## Discriminating next checks

1. Extend exact maps and real-weight contributions to T=1024/2048, including hot counts T and boundaries 255/256/257, 511/512/513. This exercises native M-block loops and rules out an accidentally small maximum.
2. Compare BF16 RM versus BFP8 TILE dispatch with identical maps/weights and a changed-routing cache hit; grade intermediate contributions and complete layer PCC, then include all conversion/map/dispatch work in warmed latency.
3. Run reuse with the selected optimization JSON and allocation tracking: unseen logical36 in physical128, cut-at36 continuation, changed page ownership, and batch/request-slot changes. Add page256/start128 and page256/length128 if larger page sizes remain supported by the tiled policy. Preserve source identities before/after each accepted run.
