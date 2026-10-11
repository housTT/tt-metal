# Single-device prefill dispatch recipe

Source-only investigation, 2026-10-10. No devices, TTNN imports, or tests were run. No shared TTNN source was changed. This is an executable candidate recipe and source diagnosis, not correctness or performance evidence.

The native fused expert op supports the requested dynamic per-expert regions. The public DeepSeek `dispatch`, `combine`, and `offset_cumsum` wrappers do not supply a usable fabric-free 1x1 path in this checkout. A local device-only candidate is therefore: **masked_bincount → local prefix sums → sort/scatter row maps → embedding gather → moe_fused_swiglu → embedding gather → post_combine_reduce**. The maps replace dispatch/combine without changing expert math, routing, or shared TTNN source.

## Native API contracts and concrete 1x1 blockers

`D = ttnn.experimental.deepseek_prefill`.

| Call | Exact relevant contract |
| --- | --- |
| `D.masked_bincount(indices, mask, 256, 8)` | indices UINT16 TILE **2D `[T,8]`**, default 32x32 tiles, interleaved L1/DRAM; T divisible by 64. Mask INT32 ROW_MAJOR `[256]` or `[1,256]`. Output UINT32 ROW_MAJOR DRAM `[256]`. Uses fixed 8x8 workers; do not restrict the loaded subdevice below this grid. All-local mask is zero for every expert. Counts are actual route occurrences, not boolean activity times T. |
| `D.offset_cumsum(counts, cluster_axis=0, num_links=1, experts_per_chip=256, memory_config=DRAM)` | Unconditionally calls `all_gather`; `all_gather/device/all_gather_device_operation.cpp:41` rejects `num_devices == 1`. Its primitive kernel would accept one histogram row, but that primitive is not exposed by this Python binding. Use the local scan below. |
| `D.dispatch(x, indices_rm, offsets, table, dispatch_group_size=1, experts_per_chip=256, num_routed_experts=256, num_experts_per_tok=8, metadata_len=3, max_dispatch_buffer_token_size=C, memory_config=DRAM, cluster_axis=0, num_links=1, topology=ttnn.Topology.Linear)` | x BF16 TILE or ROW_MAJOR; indices UINT16 ROW_MAJOR; offsets UINT32/INT32 ROW_MAJOR; table INT32 ROW_MAJOR mapping every expert to device **0**, not expert ID. Produces BF16 ROW_MAJOR `[1,1,C,2048]` and INT32 metadata `[1,1,C,3]`. |
| `D.combine(y, metadata, counts, offsets, dispatch_group_size=1, experts_per_chip=256, num_experts_per_tok=8, seq_len_per_chip=T, memory_config=DRAM, cluster_axis=0, num_links=1, topology=ttnn.Topology.Linear)` | y BF16 TILE/ROW_MAJOR or BFP8 TILE; counts and offsets identical shapes, UINT32/INT32 ROW_MAJOR. Produces BF16 ROW_MAJOR `[1,1,T,8,2048]`. Despite the docstring, no router-weight tensor is an argument: weights are applied in post-combine reduce. |

Both dispatch/combine wrappers require `num_links` in 1..4. Their factories call `get_neighbors`, whose `ccl/common/host/moe_utils.cpp` implementation rejects an empty neighbor set (`"No neighbors found"`). A 1x1 Linear mesh has none. They also unconditionally call fabric payload/header accessors, which dereference `ControlPlane::get_fabric_context()` and fail if it is uninitialized (`control_plane.cpp:2654`). Setting all destination IDs to zero does not bypass this setup. Passing zero links is rejected earlier. Ring is not a supported bypass: `get_usable_topology` normalizes it to the physically usable topology; self-neighbor directions would not provide real local fabric connections. Do not enable another device to work around these checks.

Dispatch allocates sender plus worker cores on the first row: defaults need at least three same-row cores for one sender. Combine supports a full worker grid or a one-row/one-column subdevice, rejects partial 2D subdevices, and needs at least one untilizer per sender. The local recipe needs none of that fabric setup or subdevice mutation. Existing dispatch/combine integration tests are multi-device; `test_masked_bincount.py`, `test_moe_fused_swiglu.py`, and `test_deepseek_moe_post_combine_reduce.py` provide individual single-device coverage, not evidence for this newly composed candidate.

## Counts, offsets, and capacity

Let `c[e]` be the actual histogram of top-8 expert IDs, `a[e]=32*ceil(c[e]/32)`, and `r[e]=sum(a[:e])`. Every selected route appears exactly once; `sum(c)=8*T`, `0<=c[e]<=T` because top-k selects distinct experts. A one-device offset_cumsum would return `(r,c,r,r)` with shapes `[1,256]`. Compute these locally on device. FP32 represents every integer in this recipe exactly; all are below 2^24. Do not convert counts or offsets through BF16.

A routing-independent safe capacity is `C=8*T+31*256`. The occupied region is `sum(a)<=C`; inactive experts occupy zero rows. Static capacity keeps allocation shapes independent of device counts.

| T | Routes N=8T | Safe C rows | BF16 `[C,2048]` buffer | Maximum per-expert `input_m_tiles` |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 8960 | 35 MiB | 4 |
| 512 | 4096 | 12032 | 47 MiB | 16 |
| 1024 | 8192 | 16128 | 63 MiB | 32 |

The recipe can temporarily hold dispatch, TILE expert output, its ROW_MAJOR copy, and gathered contributions: roughly `3*C*4096 + N*4096` bytes, respectively 109/157/221 MiB, excluding weights and small maps. Releasing intermediates can reduce peaks. This is an allocation estimate, not measured memory.

`input_m_tiles` is the **per-expert processing ceiling**, not the total buffer height or average load. The reader computes `m_t=min(ceil(c[e]/32), input_m_tiles)`, so setting it below the hottest expert silently truncates work. Set it to T/32 for these chunks. With `expert_region_offsets=r` and `read_x_at_offset=True`, both input reads and output writes begin at row r[e]. Output must be a separate BF16/BFP8 TILE DRAM tensor, at least as tall as input. Both tensors must have rank 2 or rank 4 with leading `[1,1]`. Offsets must be aligned and `r[e]+32*m_t<=C`; the reader skips invalid regions. The writer emits only the ceil32(count) rows; unused regions need not be read or initialized by the combine gather.

`_CompactExperts` inherits 256 rank-2 `gate`, `up`, and `down` tensors from `_DedicatedExperts`: `[2048,512]`, `[2048,512]`, `[512,2048]`. The loader packs `[up,gate]`, and these lists already separate the correct halves. The lists use matching BFP8 dtype after the existing conversion. Reuse these weights and `expert_ids` (UINT32 ROW_MAJOR DRAM identity map); do not recast from an unrelated checkpoint. Native fused compute requires `fp32_dest_acc_en=False`, `packer_l1_acc=False`, `dst_full_sync_en=False`; the inherited compute config satisfies these. Start with its 8x2 grid, then measure alternatives.

## Device-only candidate skeleton

This can live in the model's optimized Python file. Inputs `hidden` and `(values, indices)` must describe exactly the same T tokens and ordered top-8 routes. Setup-only constants may use torch; no route-dependent host readback occurs in forward.

```python
import torch
import ttnn

D = ttnn.experimental.deepseek_prefill
DRAM, RM, TILE = ttnn.DRAM_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT

# Setup once per static chunk T and existing single-device mesh.
N, C = 8 * T, 8 * T + 31 * 256
def put(x, dtype, layout=RM):
    return ttnn.from_torch(x, dtype=dtype, layout=layout, device=mesh,
        memory_config=DRAM, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
table = put(torch.zeros(256, dtype=torch.int32), ttnn.int32)
zero_C = put(torch.zeros((1, 1, 1, C), dtype=torch.int32), ttnn.uint32)
zero_N = put(torch.zeros((1, 1, 1, N), dtype=torch.int32), ttnn.uint32)
ordinal = put(torch.arange(N, dtype=torch.float32).reshape(1, 1, 1, N), ttnn.float32, TILE)

# All numeric inputs to these helpers are nonnegative integers <= 16128.
# Use UINT16 across the layout boundary. scatter explicitly rejects FP32 TILE
# and long (>256) INT32/UINT32 TILE rows; direct long UINT32 untilize is unsafe
# according to its support guard. RM typecast has a dedicated chunked factory.
def tile_f32(x):
    x = ttnn.typecast(x, ttnn.uint16, memory_config=DRAM)
    x = ttnn.to_layout(x, TILE)
    return ttnn.typecast(ttnn.reshape(x, (1, 1, 1, x.shape[-1])), ttnn.float32)

def rm_u16(x):
    return ttnn.to_memory_config(ttnn.to_layout(ttnn.typecast(x, ttnn.uint16), RM), DRAM)

def rm_u32(x):
    return ttnn.typecast(rm_u16(x), ttnn.uint32, memory_config=DRAM)

# Forward: preserve top-k order when flattening by untilizing first.
idx2 = ttnn.reshape(ttnn.typecast(indices, ttnn.uint16), (T, 8))
idx2 = ttnn.to_layout(idx2, TILE)
counts = D.masked_bincount(idx2, table, 256, 8)   # actual UINT32 [256]
cf = tile_f32(ttnn.reshape(counts, (1, 1, 1, 256)))
aligned = ttnn.multiply(ttnn.floor(ttnn.multiply(ttnn.add(cf, 31.0), 0.03125)), 32.0)
raw_start = ttnn.subtract(ttnn.cumsum(cf, dim=-1), cf)
region_start = ttnn.subtract(ttnn.cumsum(aligned, dim=-1), aligned)
offsets = ttnn.reshape(rm_u32(region_start), (256,))

# Changing the last dimension invokes reshape_view's device kernel, which
# rejects UINT16. Cross this reshape as UINT32 ROW_MAJOR, then restore UINT16.
flat_ids = ttnn.typecast(ttnn.reshape(ttnn.typecast(ttnn.to_layout(idx2, RM), ttnn.uint32),
    (1, 1, 1, N)), ttnn.uint16)
sorted_ids, permutation = ttnn.sort(flat_ids, dim=-1, descending=False, stable=True)
# Both sort outputs remain ROW_MAJOR; UINT16 input gives UINT32 permutation.
gap_table = rm_u16(ttnn.subtract(region_start, raw_start))
gap_by_route = ttnn.gather(gap_table, -1, sorted_ids, memory_config=DRAM)
packed_pos = rm_u32(ttnn.add(ordinal, tile_f32(gap_by_route)))
token_ids = rm_u32(ttnn.floor(ttnn.multiply(tile_f32(permutation), 0.125)))

# Scatter only small integer maps, never the 2048-wide hidden vectors.
dispatch_rows = ttnn.scatter(zero_C, -1, packed_pos, token_ids, memory_config=DRAM)
combine_rows = ttnn.scatter(zero_N, -1, permutation, packed_pos, memory_config=DRAM)
x_rm = ttnn.to_memory_config(ttnn.to_layout(ttnn.reshape(hidden, (T, 2048)), RM), DRAM)
x_dispatch = ttnn.embedding(ttnn.reshape(dispatch_rows, (C,)), x_rm,
    layout=RM, memory_config=DRAM)  # [C,2048], BF16; padding rows safely use token 0
y = ttnn.empty((C, 2048), device=mesh, dtype=ttnn.bfloat16, layout=TILE, memory_config=DRAM)
y = D.moe_fused_swiglu(x_dispatch, experts.gate, experts.up, experts.down,
    counts, experts.expert_ids, input_m_tiles=T // 32,
    dtype=ttnn.bfloat16, core_grid=ttnn.CoreCoord(8, 2), output=y,
    expert_region_offsets=offsets, read_x_at_offset=True,
    compute_kernel_config=experts.compute)
contributions = ttnn.embedding(ttnn.reshape(combine_rows, (N,)), ttnn.to_layout(y, RM),
    layout=RM, memory_config=DRAM)
contributions = ttnn.reshape(contributions, (1, 1, T, 8, 2048))
weights_rm = ttnn.reshape(ttnn.to_layout(values, RM), (1, 1, T, 8, 1))
result = D.post_combine_reduce(contributions, weights_rm,
    expert_dim=3, output_memory_config=ttnn.L1_MEMORY_CONFIG)
```

The maps are inverses on real routes: sorting returns `(expert_id[j], original_route[j])`; `packed_pos[j]=j+r[e]-raw_start[e]` is unique and tile-region aligned by construction. Scatter maps this packed row to token `original_route//8` and maps each original route back to its packed row. Sort tie order is immaterial if values and returned permutation agree. Only real route rows are gathered after the expert op. Duplicate scatter destinations are not used.

`post_combine_reduce` requires BF16 ROW_MAJOR inputs, **weights of the same rank as contributions with a trailing singleton**, and T divisible by 32; its docstring's rank-4 weight example is stale. Omit both optional indices/table for this all-local path; it skips weight-zero contributions. Supplying them instead requires UINT16 ROW_MAJOR indices and INT32 ROW_MAJOR table (the docstring's INT32 index claim is stale). Output is BF16 TILE `[1,1,T,2048]`. Embedding requires BF16 ROW_MAJOR weights and UINT32/BF16 indices; explicit output layout above avoids an unintended tiled gather result.

`routed_expert_ffn` processes an already prepared dense expert input. `unified_routed_expert_moe` and `hybrid_routed_expert_moe` take the same contiguous dispatched buffer, counts, expert IDs, and region offsets; none exposes a token-index-gather argument. They fuse extract/insert at a contiguous region, so they are possible later FFN substitutions, not a dispatch substitute.

## First validation and material limits

Validate map generation before expert math, including a routing case with many ties, zero-count experts, counts 31/32/33, and a hot expert with count T. Read back only in the diagnostic harness: compare exact counts, offsets, dispatch rows at every real route, and inverse map to CPU. Verify `sum(counts)==8*T`, every map bound, and `offsets[e]+ceil32(counts[e])<=C`. Then compare fused expert contributions and weighted output with the existing real-weight layer fixture. Change routing between two same-shape calls to check cache-hit addresses; qualify traced replay separately.

No speedup follows from lower arithmetic alone: almost every expert can remain active, each active region rounds to at least 32 rows, and native fused SwiGLU still loops all 256 weight triples. At T=128, one tile per active expert can consume ~8192 rows of compute for 1024 real routes. Sort, scan, embedding bandwidth, and per-expert synchronization must be included in warmed end-to-end timing. This candidate resolves the material dispatch opportunity with source-supported local APIs; passing correctness and measured latency remain the parent's device work.

Source review of the parent's first `_LocalDispatch` draft found two changes required before its smoke: the existing router emits UINT16 **ROW_MAJOR** indices, so reshape to `[T,8]` must be followed by TILE conversion for `masked_bincount`; and flattening width-8 UINT16 ROW_MAJOR indices to width `8*T` must use the UINT32 reshape bridge above. The masked_bincount keyword names are correct. The default `input_m_tiles=T//32`, safe output capacity, same-rank native output, and rank-5 post-combine weights match their validators. A nonzero `dispatch_m_tiles` override below `T//32` is not a safe general policy because the native reader silently clamps real counts; require a bound that covers actual maxima or use the default.

The current implementation incorporates both layout corrections. Its FP32 TILE gap table and UINT16 TILE gather indices match the gather contract. Scatter creates a new ROW_MAJOR output, so reusing the setup-only zero templates does not carry route state between calls. Embedding indices `[1,C]` produce `[1,C,2048]`, which can be reshaped to the required `[1,1,C,2048]`; the inverse embedding and rank-5 reduce similarly match the implementation. No further definite source-level validation rejection was found. This review did not import TTNN or run hardware.
