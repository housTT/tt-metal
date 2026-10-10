# Routed FFN hypothesis investigation

Inspection: 2026-10-10, stage 02. This investigator used the AutoFix hypothesis workflow and wrote only this report; the owning agent controls implementation and hardware. No device was opened, TTNN imported, reset performed, or implementation edited here.

**The original failure had a concrete Python adapter cause: almost all routed experts were skipped by the fused kernel's region-capacity guard. The owning agent's repaired real-weight controls now pass .995 against both HF and the functional decoder.** Internal precision was not an unavoidable obstacle for that workload. The subsequent compact eight-region implementation passes real-weight checks and improves blocking host decode replay from approximately 1,479 to 960 microseconds. Direct router-to-compact-expert handoff also passes 4096-prefix/128-decode checks. The packed prefill split adapter required a separate 256-output stack repair, now validated with eight groups of 32 outputs.

## Evidence identity and observed failures

The inspected repository HEAD is `53274a46856a80e3872967fd0e79d2d9758f48a7`. At 01:53 UTC, `tt/fused_decoder.py` has SHA256 `f1ee2671f8f658c97ed2b3f5572f9b2b18c9178aa53f2937a30f5aaf84491085`, matching all four JSONs in [ffn_controls129](ffn_controls129/commands.json). The relevant adapter is `_DedicatedExperts`, initially lines 238–292. Source is live; any subsequent repair needs its own evidence identity.

These are actual real-checkpoint layer-3 runs, B1, prefix 129, eight decode tokens, with trace replay and runtime audit:

| Variant | HF prefill PCC | HF minimum decode PCC | Functional minimum decode PCC | Median blocking trace replay, microseconds |
|---|---:|---:|---:|---:|
| Functional | .998847961 | .998132229 | — | 1478.8145 |
| Router + fused FFN, LoFi | .997186422 | .991438806 | .992748320 | 2162.9210 |
| Router + fused FFN, HiFi4 | .997186363 | .991438806 | .992748320 | 2162.9765 |
| Full-minimal candidate, inherited routed experts | .998835623 | .998121917 | .999948978 | 1243.8845 |

LoFi and HiFi4 have identical eight-element decode PCC arrays. The older [dedicated trial log](dedicated_trials129/fused_ffn.log) records prefill PCC .997186422 and fails the functional decode-equivalence assertion. The newer runner writes its failing JSON before asserting, making the current evidence more informative.

The separate [router control](full_trials129_adapted/router.json) passed with HF minimum decode PCC .998132229 and functional minimum decode PCC .999999702. That control has an older source identity; it supports the router rewrite's feasibility but is not a replacement for a fixed-source one-variable trial.

## Proven adapter failure: input capacity also bounds output region offsets

The adapter creates:

```text
P = max(32, logical_token_count)
input shape       = [1,1,P,2048]
output shape      = [1,1,256*P,2048]
expert_ids[e]     = e
counts[e]         = logical_token_count if any token selects e, otherwise 0
region_offsets[e] = e*P
read_x_at_offset  = False
input_m_tiles    = P/32
```

The intended equation is sound: every active expert evaluates the same full token chunk, and its dense routing mask removes unselected token/expert contributions afterward. Gate/up order is also correct: qwen36 `moe/weights.py` stores `[up|gate]`, while the adapter passes the second half as gate and the first half as up.

However, this C++ source chain contradicts the assumed region-capacity contract:

1. `moe_fused_swiglu/device/moe_fused_swiglu_program_factory.cpp:179` sets the reader's `NEED_START` whenever direct-write offsets are present, independently of `read_x_at_offset`.
2. Factory line 513 passes **`activations.padded_shape()[-2]`** as `ROW_CAPACITY`. It does not pass output height.
3. `device/kernels/moe_fused_swiglu_reader.cpp:525–540` always checks the offset when `NEED_START` is true:

   ```cpp
   region_ok = (start_row % TILE_H == 0) && (start_row <= ROW_CAPACITY) &&
               (m_t <= (ROW_CAPACITY - start_row) / TILE_H);
   ```

4. Reader lines 546–549 set count and `m_t` to zero when the check fails. This is an unconditional release-mode guard; the adjacent `ASSERT` is not the only enforcement.
5. Only afterward, lines 571–572 apply `READ_X_AT_OFFSET` to the input read base. Setting it false prevents input rebasing but **does not disable the earlier capacity guard**.

For any active expert `e >= 1`, `start=e*P`, `ROW_CAPACITY=P`, and `m_t>=1`. Expert 1 fails because zero rows remain after its start; higher expert IDs fail the start bound. Only expert 0 can run. The caller zero-initializes output, so rejected experts contribute zeros without creating an obvious NaN or garbage-output failure. The residual/shared path can keep full-layer PCC close to one even when routed contributions are missing.

This causal chain proves the current adapter drops active experts under the inspected source. It explains why improving matmul fidelity cannot repair these observations. It does **not** yet measure the exact PCC contribution from missing routing versus arithmetic errors in expert 0.

## Smallest permitted repair and decisive check

Keep `read_x_at_offset=False`, `input_m_tiles=P//32`, counts, expert ordering, output offsets, and the existing final routing reduction. Enlarge only the **physical input backing height** to `256*P`, with the existing `P` rows at its beginning:

```python
# After the existing conversion/padding to P rows:
inputs = ttnn.pad(inputs, ((0,0), (0,0), (0,255*P), (0,0)), 0)
```

This is the simplest diagnostic adapter. Every output region then passes the reader guard, while all experts continue reading the common first `P` rows. Output and input must remain distinct buffers; output aliasing is rejected by validation.

For lower overhead after correctness is proven, construct a separate persistent RM BF16 input backing tensor per bucket, shape `[1,1,256*P,2048]`, and use `experimental.slice_write` to refresh only its first `P` rows. Retain the existing zero padding inside the first region for decode. The remaining capacity is never read when `read_x_at_offset=False`, so it need not be filled each step. The additional backing is 32 MiB, 128 MiB, or 512 MiB for P=32,128,512 respectively; report this memory cost.

The decisive experiment is the identical real-weight LoFi/HiFi4 layer control after **only** this capacity fix. Gate both HF and functional per-token PCC at .995. If needed, capture the routed output before reduction for an active expert with ID >0: before the fix it must be all zero; after the fix it should match a direct single-expert call using offsets zero and the same quantized weights. This isolates addressing from numerical quality. No synthetic-only rejection is justified by the old results.

## Internal precision remains a real limitation, not the established failure cause

The fused kernel has fixed precision boundaries beyond Python's requested output dtype:

| Internal value | Source-enforced format |
|---|---|
| Row-major BF16 input after internal tilize | BF8 |
| Gate/up partials and gathered partials | BF8 |
| Reduced gate/up slice and SiLU gate slice | BF16 |
| SwiGLU H, communicated H, local H | BF8 |
| Down intermediate accumulation spill | BF16 |
| Final output | Caller-selectable BF8 or BF16 |

These come from `moe_fused_swiglu_geometry.cpp:326–351` and `format_for` in the factory. `moe_fused_swiglu_device_operation.cpp:283–290` explicitly rejects FP32 destination accumulation, caller-managed packer accumulation, and full destination synchronization. HiFi4 changes math fidelity but cannot remove BF8 buffers.

After fixing addressing, a still-failing case has another small diagnostic lever: `core_grid=CoreCoord(8,2)` reduces gate/up K partitions from eight to two (`geometry.cpp:116–117`, K groups are grid rows). That can change partial-quantization/reduction error; it does not remove BF8 input/H formats or prove .995 correctness. This experiment should follow, not precede, the capacity repair.

## Other exposed kernels

- **`unified_routed_expert_moe`** uses a distinct program but does not offer a high-precision escape. Its factory hardcodes BF8 input tilize and activated intermediates, fixed `kFp32DestAccEn=false`, and LoFi compute (`unified_routed_expert_ffn_program_factory.cpp:314,384–390,1161–1165`). The public RM path also returns BF8. It requires an actual dispatched input buffer with per-expert regions. A mere name/config swap cannot preserve the current graph's precision policy.
- **`hybrid_routed_expert_moe`** combines the fused and unified branches; both retain the same relevant BF8 boundaries. It is not evidence that this capacity issue or accuracy gate is solved.
- **`minimal_matmul(..., fuse_swiglu=True)`** accepts BF16 activations and configurable FP32 destination accumulation/intermediate storage. The same repository already has real-checkpoint evidence for the shared-expert adapter: [shared_minimal](dedicated_trials129/shared_minimal.json) passes HF minimum decode PCC .998098195 and functional minimum decode PCC .999987304. This validates a useful existing fused FFN subgraph, not routed-expert equivalence.

For a routed **prefill-only** alternative, minimal_matmul can take one horizontal packed weight `[2048,256*2*512]` with adjacent 32-column `(gate,up)` tile pairs within each expert. Its BF16 result `[1,1,T,256*512]` reshapes to `[1,T,256,512]` and permutes to `[1,256,T,512]`, feeding the inherited sparse down projection and routing reduction. **Correction to the initial investigation:** current qwen36 prefill already uses per-32-token expert-union sparsity for gate/up; only down uses an all-expert mask. A dense packed matmul adds gate/up arithmetic for unselected expert/tile pairs. Keep inherited sparse decode. `minimal_matmul` rejects batched weight dimensions, making direct `[1,256,2048,1024]` use invalid. This packing variant has no routed real-weight result yet and must not be claimed passing.

## Follow-up: repaired real-weight results

The owning agent implemented the capacity repair and ran [ffn_repaired129](ffn_repaired129/commands.json). All three JSONs identify fused-source SHA256 `448558ee090e360928e7ff4103ccbc97d9462b855e20d4e97f828e64937b87d5`:

| Variant | HF prefill PCC | HF minimum decode PCC | Functional minimum decode PCC | Median blocking trace replay, microseconds |
|---|---:|---:|---:|---:|
| Functional | .998847961 | .998132229 | — | 1479.0995 |
| Repaired fused FFN, grid 8x8 | .998846769 | .998107731 | .999990046 | 2425.5235 |
| Repaired fused FFN, grid 8x2 | .998841107 | .998089314 | .999980330 | 1869.2775 |

This validates the capacity diagnosis on the real failing workload. The grid-2 variant improves the measured replay time relative to grid-8 but still loses to the functional decoder. These are blocking host replay measurements, not isolated device-kernel latency. They do not qualify longer prefixes or other layers.

## Exact routed-prefill packing, preserving existing BFP4 values

The live `_MinimalExperts` implementation added by the owning agent follows the correct orientation. Construction should use the **already quantized** `original.weights.gate_up_proj`, not newly selected checkpoint precision:

```python
# Setup only. One-device mesh; existing qwen36 tensor is [1,E,H,2I]=[up|gate].
q = ttnn.to_torch(original.weights.gate_up_proj,
                 mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[0]
E, H, two_I = q.shape
I = two_I // 2
up = q[..., :I].reshape(E, H, I // 32, 32)
gate = q[..., I:].reshape(E, H, I // 32, 32)
packed = (torch.stack((gate, up), dim=3)
          .permute(1, 0, 2, 3, 4).contiguous().reshape(H, E * 2 * I))
packed_tt = ttnn.from_torch(
    packed, device=mesh, dtype=original.weights.gate_up_proj.dtype,
    layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
```

The exact mapping is `packed[k,e*2I+2*32*j+c] = gate[e,k,32*j+c]` and `packed[k,e*2I+(2*j+1)*32+c] = up[e,k,32*j+c]`, with `0<=c<32`. Thus expert e is contiguous along N; within each expert the first tile of each pair is gate and the second is up. The output pair index is `e*(I/32)+j`.

This moves whole 32-column tiles while retaining K-row membership. BFP4's shared-exponent groups are 16 contiguous values inside a tile face (`tt_metal/impl/data_format/blockfloat_common.cpp:27–33`; BFP4 unpack in `bfloat4.cpp`). Those groups are unchanged, so unpacking and repacking the existing finite BFP4 values retains their numerical values. A construction-only `to_torch(packed_tt)` equality check against `packed` is the appropriate evidence if the implementation needs to assert that preservation; no such new device check was run by this investigator. Do not transpose K with I, interleave individual scalars, or swap `(gate,up)` to `(up,gate)`.

Runtime needs BF16 output explicitly:

```python
h = ttnn.experimental.minimal_matmul(
    hidden, packed_tt, fuse_swiglu=True, dtype=ttnn.bfloat16,
    compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
h = ttnn.reshape(h, (1, T, E, I))
h = ttnn.permute(h, (0, 2, 1, 3))  # [1,E,T,I]
# Inherited sparse down: self.weights.down_proj, self.prefill_sparsity,
# nnz=E, is_input_a_sparse=True, _build_sparse_matmul_config(T,H), BF16.
```

All dimensions in prefill are tile-aligned. No explicit RM conversion is required: BF16 tiled reshape has a native data-movement factory (`data_movement/reshape_view`), and the following permutation is tiled. The first reshape changes tile order and is **not** a free view. The final weighted reduction retains routing `[1,1,T,E] -> [1,E,T,1]` and sums expert axis 1.

`minimal_matmul`'s `swiglu_sfpu.hpp` distinguishes BF16 output from block-float output. BF16 avoids the cheap block-float sigmoid approximation; FP32 destination accumulation selects its accurate FP32 sigmoid branch. These precision choices still change matmul/activation rounding relative to inherited sparse matmul and must pass real-layer evidence.

The native split alternative emits each expert as `[1,1,T,I]`, which concatenates directly to `[1,E,T,I]`. Although the public validator accepts `chunks=E=256`, the real trial exposed a NCRISC stack overflow in its output-accessor tuple. [AutoTriage report](triage/prefill_split/AUTOTRIAGE.md) separates captured waits from the compiled stack/counter-overlap inference. The owner repaired this in Python by slicing the packed BFP4 matrix into eight construction-time groups of 32 experts and calling `minimal_matmul_split(..., chunks=32, fuse_swiglu=True, dtype=BF16)` per group. Real129 prefill passes HF PCC .998835087 and functional PCC .999959826; warmed blocking host prefill is 26272.256 microseconds, versus unsplit 29727.359 and functional 56824.796. See [repaired evidence](prefill_split_repaired129/prefill_split.json). This workaround retains existing quantized values and expert order; the dense packed gate/up path still computes inactive tile/expert pairs.

## Compact eight-region decode with all 256 weight lists

**The source supports this without dynamically selecting host weight objects.** Keep 256 entries in each gate/up/down list and 256 entries in `global_expert_idx_table`. Use that index table to map local weight-list expert IDs to compact output slots, rather than shortening the list:

```text
For each selected expert e_s, s in [0,7]: idx[e_s] = s
For every unselected expert:            idx[e]   = 8
counts  = [1,1,1,1,1,1,1,1,0]                  # UINT32 RM, length 9
offsets = [0,32,64,96,128,160,192,224,0]         # UINT32 RM, length 9
input backing shape = [1,1,256,2048]           # real token in row 0
output shape        = [1,1,256,2048]
input_m_tiles=1; read_x_at_offset=False
```

The eight selected expert IDs and scores can come from device top-k of the existing dense routing, or be retained from a specialized router. Build `idx` on device by scattering slot numbers 0–7 into an array initially filled with 8. If UINT32 scatter is unsupported in the chosen path, BF16 values 0–8 are exact; scatter BF16 then cast UINT32 and convert RM. Counts/offsets are static resident tensors.

Why the mapping is valid:

- Public `experts_per_chip` is the weight-list length. Validation requires **idx length >= experts_per_chip**, so an eight-element idx table with 256 weights is invalid.
- `moe_fused_swiglu_reader.cpp:429–431` chooses gate/down base addresses using **local_expert_id**; the writer does the same for up/down at lines 269–271. The dynamic idx value does not choose weights.
- Reader lines 502–512 use `global_expert_id=idx[local_expert_id]` only to index counts; offsets use the same index. The bound is **counts length**, so sentinel 8 is in range for length 9 and has count zero.
- Validation requires offsets length == counts length. It does not require either length to equal weight-list length, and direct-write output need only cover the supplied regions.
- Each of the eight unique selected experts writes exactly one compact region. Unselected experts have zero loop count and write nothing.

Because every logical output slot is written, an uninitialized output allocation can replace the 256-region `zeros_like` buffer for this decode path. Preserve input row padding where required, and consume only logical row 0 from each output region. Reshape to `[1,8,32,2048]`, select the first token row, weight by the correspondingly ordered eight scores, and reduce axis 1. Both enlarged input/output backing drop from 32 MiB to 1 MiB at decode. All 256 experts are still inspected inside the fused program; compact regions remove large outer zero-fill/multiply/reduce costs, not that internal loop.

The owner implemented this recommendation and produced the measured evidence below; this investigator did not execute hardware. The alternate identity-idx formulation also works by source: counts length 256, offsets[active_expert]=slot*32, inactive count zero. The nine-entry sentinel version makes the compact-slot contract explicit.

### Compact real-weight evidence and ownership review

The [compact129 trials](compact_trials129/commands.json) record source SHA256 `212c9adbb385c110ff9d20770c5eec7faf43bfa9540fb803b20207fe57a6387d`. All are real layer3 B1, prefix129, eight decode tokens, runtime audit and trace replay:

| Variant | HF minimum decode PCC | Functional minimum decode PCC | Median blocking host replay, microseconds |
|---|---:|---:|---:|
| Functional | .998132229 | reference | 1478.855 |
| Compact FFN | .998110175 | .999946237 | 962.5045 |
| Compact FFN with fused weighted reduction | .998110294 | .999946594 | 960.095 |

The direct top8 pair handoff subsequently passes prefix4096 and 128 decode tokens at source SHA256 `2cef42c3bdb8c65a55ab4da301bb6d35b85c6f98cbc71d9b45bf5279b23917bb`:

| Layer/result | HF minimum decode PCC | Functional minimum decode PCC | Median blocking host replay, microseconds |
|---|---:|---:|---:|
| [Full attention, compact routing](compact_trials4096/compact_routing.json) | .995726228 | .999637008 | 924.6975 |
| [Linear attention, complete fusion set](complete_linear4096/linear_complete.json) | .997841656 | .999664128 | 1279.162 |

These configurations include earlier graph fusions, so their improvements are not isolated FFN-kernel timings. Their successful B1 checks do not by themselves qualify B32 or every context case. Full-attention margin to .995 is modest and should be retained explicitly in final acceptance.

The formatted live source was rechecked at SHA256 `bc78fb8d141f5e67e1cb26c1198f5a088da49db648c2e09da6cf7bbde3efe32a`. Source review found no stale-routing or output-ownership defect in the measured composition:

- `scatter` allocates a new tensor (`scatter_device_operation.cpp:66–68`), so repeated compact-table construction does not mutate the persistent all-8 sentinel template or leave earlier selected experts active.
- Gate/up/down lists, sentinel counts/offsets, and mapping tensors remain owned by the module. All eight selected topk IDs are unique; each maps to its matching score slot and writes that output region. Only logical row0 is consumed after the native 32-row write region is selected.
- The router returns the `(values, indices)` pair as call-local tensors. `Qwen36MoE.forward` passes routing directly to the expert consumer, and `_PackedPrefillExperts` forwards decode to its wrapped `_CompactExperts`. There is no mutable Python `last_topk` cache. The gate module copies per-chunk outputs out of its reusable internal buffers (`tt_moe_gate.py:581–582`).
- Decode still uses a one-token expert call. `FunctionalDecoder.decode_forward` iterates request slots and invokes each row independently, so the compact counts of one are compatible with outer B32 semantics; B32 hardware qualification remains the owner's separate acceptance task.
- `tests/run_layer.py:248–255` copies each successive token and position into the captured trace input before replay. Thus the multi-token PCC series exercises changing runtime routing inputs; `repeated_identical` records a separate deterministic repeat. The test does not log exact expert-ID transitions, but the trace has no host-derived expert selection.
- Direct handoff removes dense scatter and a redundant second topk. It keeps normalized scores paired with their IDs; changes in reduction ordering/rounding are measured by the real functional comparisons above. Ties or zero-weight ordering do not justify introducing another host selection step.

The review identified one optional-candidate dependency: selecting `moe_gate,compact_routing` without either `compact_ffn` or `compact_reduce` returns a tuple to an incompatible dense expert consumer. The owner subsequently added the constructor guard, confirmed in the later live source at SHA256 `7d0d59a808a6e8351101ba13af2769c301a1243dd4223ce2564b33dd40de9a3f`. The measured/default combinations already included the compact consumer.

## Additional dedicated fusions missing from the current ledger

1. **Fused routed weighting and reduction, now implemented and measured:** `ttnn.experimental.deepseek_moe_fast_reduce_nc_fused` directly matches the multiply+reduce subgraph. For compact decode, reshape activation to `[8,1,1,2048]`, supply BF16 RM scores `[1,1,1,8]`, valid UINT16 expert indices of that shape, and an all-zero UINT16 expert-to-device map on this single-device mesh. Invoke with `reduce_dim=0`, `split_size=2048`, `cluster_axis=0`; the sole returned tensor is `[1,1,1,2048]`. Its reader fuses score layout/tilization and its compute kernel accumulates `activation*score` without materializing weighted activation. The compute config supports FP32 destination accumulation. The real compact129 and subsequent compact-routing4096 results above validate the removed BF16 product boundary in those workloads.

   This op's score tiles consume `ceil(T/32)*E*2048` bytes **per worker core**. Compact E=8,T=1 is 16 KiB; dense E=256,T=1 is 512 KiB; dense E=256,T=128 is 2 MiB before any other buffers. Therefore dense prefill is not a direct full-chunk application. The dataflow uses tensor accessors and host validation does not enforce L1 activation placement, despite the public doc's L1 wording; the owner's measured compact path successfully consumes DRAM activation.

2. **Projection plus residual:** `ttnn.experimental.dit_minimal_matmul_addcmul_fused` computes `base + scalar*(x@W)*gate` in one program. With a persistent all-ones gate, it can fuse attention output projection and residual add if a local layer wrapper passes the residual into the projection adapter. It can also fuse shared-expert down projection, expanded sigmoid gate, and routed-output addition. The public validator accepts gate `[1,H]` or full `[T,H]`; the live Metal2 descriptor/compute supports row-by-row full gates. The shared gate is `[T,1]`, so it must first be expanded over H. A deeper factory check also requires the residual/base dtype to match the weight dtype; the full-attention BF8 weight therefore needs construction-time BF16 expansion preserving its existing quantized values. Details below. These change intermediate rounding and have no new real-weight evidence in this report. The base/residual must not then be added a second time by the inherited layer.

3. **Indexed partial-width RoPE is available but is not a direct adapter substitution.** `deepseek_prefill.rotary_embedding_indexed` can retain the non-rotary 192 channels and consume resident position metadata, but its reused llama compute rotates within 32-channel tiles using a transformation matrix. The current target uses a half-split 64-channel rotation crossing a tile boundary. Matching it requires channel/weight/table permutation, and metadata positions are tile-aligned. Keep this behind the exact compact-FFN and weighting/reduction experiments; a bare `rotary_dim=64` call would not prove equivalence.

`sigmoid_gated_rms_norm` alone is not GDN's SiLU gate, but the final source scan found a valid two-op adapter: multiply its result by the original gate. The earlier name-only mismatch is therefore not sufficient to reject the fusion. Plain fused pre-add RMSNorm still returns only normalized output; the layer needs its residual sum for the later skip, so using that argument alone does not remove the separate residual producer.

## Final source-only graph scan

This scan used the graph-fusing skill checklist and the current graph inventory. It proposes graph transformations only; no geometry or datatype sweeps or optimized-decoder work were requested. These source-backed candidates were handed to the owner for isolated real-weight experiments. The owner has now staged `kda_tuple` and `kda_norm` candidates; staging alone is not measured qualification.

### GDN native convolution output tuple: exact structural cancellation

`experimental/kda/qkv_causal_conv1d_silu/device/qkv_causal_conv1d_silu_device_operation.cpp:126–141` returns three independent BF16 tiled tensors `[1,T,Q]`, `[1,T,K]`, `[1,T,V]`. `_FusedGDN._convolve` previously concatenated these along width, and `forward` immediately sliced the same Q/K/V ranges apart. Return the three cropped outputs directly and consume the tuple in `forward`. Keep cropping native 32-row decode output to the one logical token, and keep the existing convolution-state tail copy. This removes a concat and three slices without changing arithmetic or FP32 recurrent state. The later live source contains this candidate as `kda_tuple`.

There is also a small exact prefill-only state-layout cancellation: for `length>=3`, the new convolution history is `current[:,length-3:length,:]`, so concatenating old history with the entire current chunk only to slice its last three tokens is unnecessary. Decode still needs two old tokens plus the current token. This is a graph simplification, not a state reset or padding change.

### GDN gated RMSNorm: source-supported two-op adapter

For aligned prefill, preserve the existing BF16 recurrence-output boundary, retain head-major output `[32,T,128]`, and supply BF16 flat Z `[1,T,4096]` plus BF16 weight `[128]` to:

```python
g = ttnn.experimental.kda.sigmoid_gated_rms_norm(
    output_bf16_head_major, z_bf16_flat, weight_bf16_vector, 32,
    epsilon=norm_eps, compute_kernel_config=norm_compute,
    memory_config=ttnn.DRAM_MEMORY_CONFIG, output_dtype=ttnn.float32)
gated = ttnn.typecast(
    ttnn.multiply(g, ttnn.typecast(z_bf16_flat, ttnn.float32)),
    ttnn.bfloat16)
```

The algebra is `RMSNorm(x)*weight*sigmoid(Z)*Z = RMSNorm(x)*weight*SiLU(Z)`. The validator accepts BF16/FP32 input, BF16 gate/weight, BF16/FP32 output, interleaved TILE storage and `T%32==0`. The factory allocates FP32 norm/tmp/stats/inverse buffers. Use the existing HiFi4/FP32 destination settings with `packer_l1_acc=False`, required by this op's validator. No persistent state enters this operation. Native output is already flat `[1,T,4096]`, so the original head-layout conversion and flattening can also disappear. The owner has staged this candidate as `kda_norm`.

This still changes the intermediate BF16 norm/weight rounding boundaries relative to the reference graph, and the kernel uses FPU arithmetic with FP32 buffers rather than promising bit-identical FP32 SFPU calculations. Real PCC is required. For decode T1, pad the stateless input and gate to32, consume row0, and leave state updates at one real token; extra movement makes that a separate candidate. A sigmoid-only substitution without the final Z multiply is mathematically wrong.

### Projection plus retained residual: exact API and adapter contract

```python
h = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
    u, weight_adapter, 1.0, residual_x, ones,
    bias_tensor=None, memory_config=mc, dtype=ttnn.bfloat16,
    compute_kernel_config=projection_compute)
```

The positional formula is **argument4 + argument3 * (argument1 @ argument2) * argument5**. For both attention kinds, use `u=[1,T,4096]`, weight `[4096,2048]` (leading singleton dimensions also accepted), residual `[1,T,2048]`, and persistent ones `[1,1,2048]`. All are allocated on the same device in TILE layout; interleaved storage is the existing path. Output h is BF16 `[1,T,2048]` and is the retained post-add residual tensor required later.

The validator supports BF16/BF8/BF4/FP32 activation and weight, without requiring activation dtype==weight dtype. **The factory additionally requires residual/base data format and tile size equal to weight format** (`minimal_matmul_program_descriptor.cpp:454–458`). GDN's BF16 output weight already matches BF16 residual. Full attention needs a construction-time `ttnn.typecast(existing_o_proj, ttnn.bfloat16)`; this preserves the already-quantized BF8 values while satisfying the ABI. Casting the residual to BF8 would alter the residual stream and is not the proposed adapter. The ones tensor may stay BF16.

The factory forwards the compute config, supports LoFi through HiFi4 and `fp32_dest_acc_en`, and chooses Float32 intermediate CBs when FP32 destination is enabled (`minimal_matmul_program_descriptor.cpp:229–234,790–808`). Thus GDN may retain HiFi4/FP32 accumulation and full attention may retain its existing fidelity/accumulation policy. FP32 accumulation support is not a guarantee of exact old rounding: the fused ternary stage uses FPU multiply/add and removes the former BF16 matmul-output boundary. It does not touch recurrent state, but real-layer PCC remains necessary.

The smallest safe integration is an instance-local layer forward plus optional `residual=None` on `_FusedAttention.forward` and `_FusedGDN.forward`. Reuse the original norm, attention and `_RankThreeMoE` objects. Compute attention_norm(x), call attention with `residual=x` so its terminal projection emits h directly, normalize h, invoke the unchanged `_RankThreeMoE.forward`, and finally add h to the MoE output. Skip the inherited first `ttnn.add(x,attn_output)` and retain h until the final add. Preserve all page/position/chunk metadata, GDN chunk/recurrent mode, and `valid_len`. `_select_state` must still refer to that same attention object. There is no need to modify `_RankThreeMoE`: it already bridges rank3 `[1,T,2048]` to the MoE's rank4 input and back. Never return h to the unchanged inherited layer forward, which would add x twice.

### Contracts that do not match the requested fusion

- `rms_norm`, `dit_rms_norm_unary_fused`, and `fused_rms_minimal` each return one normalized tensor. None returns a separately retained pre-normalization residual sum. In-place normalized output likewise cannot preserve that sum. `fused_rms_minimal` additionally requires width-sharded L1 input/output, persistent stats and semaphore setup; those do not add a second output.
- `dit_rms_norm_unary_fused` applies SiLU/GELU to the normalized tensor itself. GDN gates with a separate Z tensor, so this unary epilogue alone is the wrong graph. The sigmoid-RMS-times-Z adapter above is the applicable alternative.
- KDA `prepare_chunk_recurrence` requires BF16 per-key-channel log gates with the same flat shape as Q/K; the current GDN uses FP32 scalar-per-value-head log decay. Its scans use FPU matmuls. Converting those gates and expanding heads does not preserve the existing FP32 recurrence contract merely because final state storage is FP32. `gated_delta_attn` likewise requires a precomputed 128-token chunk algebra; it does not replace the one-token FP32 SFPU update. The already adopted exact chunk kernel remains the matching prefill dedicated operation.

## Owner follow-through

Later adapted experiments resolve this report's candidate suggestions. See
`graph_inventory.md`: compact FFN, direct routing, grouped32 prefill, direct KDA
Q/K/V, prefill tail and prefill norm/gate are selected. Corrected projection
residual and padded decode KDA norm lose on measured latency (full residual also
fails headline PCC). Final gates and device profiles are in README.md.
