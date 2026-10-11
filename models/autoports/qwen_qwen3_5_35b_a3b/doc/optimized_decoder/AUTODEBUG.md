# AutoDebug: compact top-8 expert FFN capacity and alternatives

Investigation date: 2026-10-10. Inspection only: no TTNN import, device execution, reset, implementation edit, or AutoDebug relaunch. The hardware owner retains all execution. Only this report was written by the investigator. Paths below are relative to the `tt-metal` checkout unless stated otherwise.

## Findings

1. **BF8 weights / BF16 row-major input / 4×2 is a real live-L1 overlap, reproduced exactly by static arithmetic.** The native planner admits against nominal architectural capacity, whereas Metal checks against the lowest live L1 allocation. The actual CB region ends at **1,559,808**, above the observed live frontier **1,556,416**, an overlap of **3,392 bytes**. There is no need to hypothesize device corruption to explain this exception.
2. **2×2 is a different failure: unsupported output geometry.** Each core would own 16 output tiles, exceeding this kernel's fixed eight-tile DEST limit. Neither BF4 weights nor BF8 input can repair that grid.
3. **BF4 weights / 4×2 already passes in the supplied evidence.** The problem description overgeneralizes the failure matrix. BF4 activations, however, are explicitly unsupported. All three native weight roles must share the same dtype.
4. **The smallest promising Python-only adaptation for BF8 / 4×2 is BF8 TILE input, keeping the existing 256-row capacity, DRAM placement, BF16 output, and routing.** Source-recomputed CB end is **1,530,176**, giving **26,240 bytes** of margin against the old frontier. This is a source prediction, not a hardware result. The net saving is **29,632 bytes**, not 64,448: the planner uses some of the input-buffer saving to increase hidden-buffer depth.
5. **Several recorded sparse failures never execute `sparse_matmul`.** They stop at an old Python name collision, `self.config(...)` calling the inherited `MoEConfig` object. The inspected implementation uses `program_config(...)`. Those failed rows do not reject their proposed device geometries.

The narrow intervention boundary is `_OptimizedCompactExperts.__call__` and policy validation in `tt/optimized_decoder.py`. No native implementation change is required to try the supported input layout or select an already passing grid. Native admission should eventually account for live capacity, but that is outside this requested patch boundary.

## Evidence and revision limits

Base HEAD is `1d857aa39b50774bbf35fbd09a2dedf53450a3bc`; `optimized_decoder.py` is a working-tree addition. The hardware owner continued adding trial results and editing this file while this investigation ran. Consequently the table below describes **the named saved artifacts**, not a claim that every row ran identical source.

The compact BF8 successes record optimized source hash `a0ea32e87f68d833865f8e32c87fc16a40aa3450c7cc79e6e1454886588e4f0f`; compact BF4 successes record `10772514e9c38c96d4087ca3063a156a34e7a53eaa0a73db6d1d28b5926095a3`. Successful nonsplit sparse trials record `67527e2994de59f77972a16857d349928f4f256ed915653f501815396e24d2cd`. A later inspected optimized snapshot was `6ed7e27fa9f7fb601e99760942c460f34f642c4ea82719c6aadf3232ddc601a1`. Use symbol names as well as line numbers when navigating this changing file. The saved error, emitted compile arguments, stable C++ formulas, and exact address match provide stronger evidence for the capacity finding than historical Python source identity alone.

Stable inspected sources:

| File | SHA256 |
|---|---|
| `tt/fused_decoder.py` in the autoport | `e570144c58707bb0846be80075c0b26787e2b3cff8539e9a245bca739c1a168d` |
| native `device/moe_fused_swiglu_geometry.cpp` | `0cb31c59b59e014c55d4241c0dcccd8b46f3b32b6a6a94876caccc3b8d725344` |
| native `device/moe_fused_swiglu_program_factory.cpp` | `2bfd8183727bb8390e434693a1b9fcf1abe421948db374751fa2d02878a9b608` |

For brevity, **N/** below means `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/`; **A/** means `models/autoports/qwen_qwen3_5_35b_a3b/`.

### Direct observations

`expert_real/commands.json` records layer 3, length 128, 16 decode steps, trace, runtime audit, warm prefill, real checkpoint weights, and recorded HF activations from `runs/main/stage03_inputs/layer3.pt`. Successful JSON files record `repeated_identical=true`. Latency below is the median **whole-layer blocking host trace replay**, not the native FFN device duration. PCC is the minimum per-token decode PCC.

| Native weight dtype | Grid x×y | Observed result | Median replay µs | Min decode PCC |
|---|---:|---|---:|---:|
| BF8 | 2×2 | DEST: 16 tiles > 8 | — | — |
| BF8 | 4×2 | CB end 1,559,808 > L1 frontier 1,556,416 | — | — |
| BF8 | 4×4 | pass | 894.815 | 0.9994128 |
| BF8 | 8×2 | pass | 928.199 | 0.9994024 |
| BF4 | 2×2 | DEST: 16 tiles > 8 | — | — |
| BF4 | 4×2 | pass | 880.111 | 0.9988191 |
| BF4 | 4×4 | pass | 873.163 | 0.9987903 |
| BF4 | 8×2 | pass | 909.493 | 0.9988134 |

Error evidence: `expert_real/compact_bfp8_4x2.log:222,255`; both `compact_bfp*_2x2.log:36,69`. Passing evidence is the correspondingly named JSON. The default BF8 8×2 control is 926.901 µs, minimum PCC 0.9994024. These are short layer screens; they do not qualify the whole model, other layer types, or final workload.

### Synchronization boundary

The failures occur during the first eager decode (`A/tests/run_layer.py:220–223`), before the test can enter trace capture. `ProgramImpl::validate_circular_buffer_region` compares addresses on the host before this program executes (`tt_metal/impl/program/program.cpp:2433–2542`). The exception is an allocation conflict, not an asynchronous hang inferred from teardown. Passing cases also copy eager results to the host and execute blocking trace replays (`run_layer.py:247–269`); their success is stronger than merely reaching the Python return.

## Lowered call and geometry

The production compact adapter supplies all 256 gate/up/down descriptors, each with gate/up dimensions 2048×512 and down dimensions 512×2048. Hidden decode input is logically `(1,1,1,2048)`. It is untilized and zero-padded to `(1,1,256,2048)` in DRAM. Output is preallocated BF16 TILE DRAM with the same shape. The native call explicitly uses `input_m_tiles=1`, selected `(x,y)` grid, LoFi, `fp32_dest_acc_en=False`, `packer_l1_acc=False`, no bias, and `read_x_at_offset=False`.

`_CompactExperts` builds nine-entry counts `[1]*8+[0]`, offsets `[0,32,...,224,0]`, and a 256-entry expert-to-slot table whose default slot 8 is inactive (`A/tt/fused_decoder.py:551–555`). Device scatter assigns each of the eight selected experts its slot 0–7. An active expert therefore sees count 1 and writes its one physical tile-row to its slot. The final reduction uses the corresponding original route scores. Unsorted top-k is harmless when IDs, slot labels, and scores stay paired.

The wrapper does not swap operands or choose a different implementation for these grids. It preserves explicit `input_m_tiles`, grid and supplied output dtype; a conflicting `dtype` and preallocated output is rejected (`N/moe_fused_swiglu.cpp:53–78`). Device validation accepts only BF16 ROW_MAJOR or BF8 TILE activations, both DRAM interleaved; weights can be BF4/BF8/BF16 TILE but all roles and experts must have the same dtype (`N/device/moe_fused_swiglu_device_operation.cpp:95–142`).

For this model, embedding tiles `E=64`, hidden tiles `H=16`; grid columns split H, rows split E. `M_BLOCK=8`, `DEST_LIMIT=8`, output CB depth 2, and resident weights are fixed native constants (`N/device/moe_fused_swiglu_geometry.hpp:14–45`).

| Grid | Cores | K tiles per row `kr_pad` | Hidden tiles per column `hn_pad` | Output tiles per core `ec_max` | Gate/up chunks | Down hidden rounds | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| 2×2 | 4 | 32 | 8 | 16 | 2 | 2 | rejected before CB placement |
| 4×2 | 8 | 32 | 4 | 8 | 2 | 4 | DEST legal; L1 depends on representation |
| 4×4 | 16 | 16 | 4 | 4 | 2 | 4 | passes saved native trials |
| 8×2 | 16 | 32 | 2 | 4 | 2 | 8 | passes saved native trials |

`ec_max=ceil(64/(x*y))`; the rejection is `ec_max <= 8` (`geometry.cpp:154–174`). A caller guard can reject 2×2 clearly or a policy can select 4×4/8×2. For these dimensions, `x*y>=8` is necessary, not a general proof of validity: native restrictions also include `x>=y>=2`, device grid bounds, scatter geometry, and L1 capacity.

The count-1 runtime path has `m_t=1`, `m_blocks=1`, `m_eff=1`, no full-eight-row grouped down path and no cross-block prefetch. It computes one physical tile-row, not eight. Nevertheless **CB capacities retain `M_BLOCK=8`**; `input_m_tiles=1` only already reduces `depth_x` to 1. Neither reducing input capacity nor repeating `input_m_tiles=1` reduces these fixed buffers (`geometry.cpp:184–238,319–348`; reader:514–555,614–634).

## Exact resource reconciliation

### Planner versus final placement

The factory takes `get_max_worker_l1_unreserved_size() - 70656`, an architectural budget, without consulting live tensor allocations (`N/device/moe_fused_swiglu_program_factory.cpp:393–432`). HAL derives the maximum from L1 end minus kernel-config base (`tt_metal/impl/context/metal_env.cpp:744–747`). On the observed Blackhole map, L1 end is 1,572,864, kernel-config base is 41,088, and the default reserved config region is 70,656, yielding CB base **111,744** and nominal CB budget **1,461,120**. The address construction is in `tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal_tensix.cpp:53–69`.

The planner also estimates scratch pages as 64/64 unconditionally (`geometry.cpp:441–445`). Actual CB creation uses tensor page sizes (`program_factory.cpp:418–435`): the 256-entry UINT32 table needs **1,024 bytes**, while counts and offsets each fit the reused **64-byte** page. Thus the estimate undercounts **960 bytes**. This is a real adjacent discrepancy, but fixing it alone would still admit the failing configuration.

| BF8 weights, BF16 RM, 4×2 | Bytes |
|---|---:|
| Planner estimate after depth tuning | 1,447,104 |
| Actual scratch-corrected CB allocation | 1,448,064 |
| Nominal CB capacity | 1,461,120 |
| Nominal remaining capacity after actual CBs | 13,056 |
| Live L1 frontier below physical end: 1,572,864 − 1,556,416 | 16,448 |
| Actual capacity up to live frontier: 1,556,416 − 111,744 | 1,444,672 |
| Overlap: 1,448,064 − 1,444,672 | **3,392** |

This completes the causal chain: nominal admission succeeds; actual CB allocation is slightly larger than estimated; the live allocation frontier reduces available space by more than the nominal headroom; the final validator correctly refuses overlap. The earliest decisive mismatch is the capacity used by the op planner, not the final validator. No future native output L1 allocation is missing from this particular budget: output is already preallocated in DRAM.

### Per-core CB ledger for the failing 4×2 configuration

The planner reduces down-weight depth from 4 to 3 and disables down residency; it then reduces hidden-buffer depth from 3 to 2. `depth_x=depth_w=1`. Gate/up scatter has `gather_pages=32`, `slice_pages=16`. BF8 tile size is 1,088 bytes, BF4 576, BF16 2,048, including block-float exponent bytes.

| CB index / role | Actual bytes |
|---|---:|
| 0: row-major X input, 32 × 2,048 | 65,536 |
| 1: tiled X, 8 × 32 × 1,088 | 278,528 |
| 2/20/21: shared stage/mailbox backing | 64 |
| 3: gate weights | 139,264 |
| 4: up weights | 139,264 |
| 5: down weights, depth 3 | 104,448 |
| 6: hidden, depth 2 | 69,632 |
| 7: index scratch | 1,024 |
| 8: counts/offset scratch | 64 |
| 9 / 10: gate / up gather | 34,816 each |
| 11 / 12: gate / up BF16 slices | 32,768 each |
| 13: hidden slice | 17,408 |
| 14: BF16 output, 2 × 8 × 8 tiles | 262,144 |
| 15 / 16: gate / up accumulation | 34,816 each |
| 17/19: aliased SiLU / output-intermediate backing | 131,072 |
| 18: local hidden | 34,816 |
| **Total** | **1,448,064** |
| **Base + total** | **1,559,808** |

The allocation formulas and alias rules are `geometry.cpp:319–348,401–426`. Mailbox views must not be counted three times. SiLU/output intermediates share their LCM-sized allocation. BF16 output disables the other optional phase alias.

### Passing-grid contrasts and layout matrix

The following are **calculated allocations**, not new device results. They include actual scratch pages and planner depth changes; all use BF16 output, `depth_x=depth_w=1`, and no down mrow/mgroup path. A separate standard-library-only arithmetic pass reproduced every total from the CB expressions and alias LCMs.

| Grid | Weights | Input | Down depth / resident | H depth | CB bytes | CB end |
|---|---|---|---|---:|---:|---:|
| 4×2 | BF8 | BF16 RM | 3 / no | 2 | 1,448,064 | 1,559,808 |
| 4×2 | BF8 | BF8 TILE | 3 / no | 3 | 1,418,432 | 1,530,176 |
| 4×2 | BF4 | BF16 RM | 4 / yes | 3 | 1,321,088 | 1,432,832 |
| 4×2 | BF4 | BF8 TILE | 4 / yes | 3 | 1,256,640 | 1,368,384 |
| 4×4 | BF8 | BF16 RM | 4 / yes | 3 | 898,688 | 1,010,432 |
| 4×4 | BF8 | BF8 TILE | 4 / yes | 3 | 867,008 | 978,752 |
| 4×4 | BF4 | BF16 RM | 4 / yes | 3 | 800,384 | 912,128 |
| 4×4 | BF4 | BF8 TILE | 4 / yes | 3 | 768,704 | 880,448 |
| 8×2 | BF8 | BF16 RM | 8 / yes | 3 | 931,456 | 1,043,200 |
| 8×2 | BF8 | BF8 TILE | 8 / yes | 3 | 867,008 | 978,752 |
| 8×2 | BF4 | BF16 RM | 8 / yes | 3 | 833,152 | 944,896 |
| 8×2 | BF4 | BF8 TILE | 8 / yes | 3 | 768,704 | 880,448 |

These totals explain the full named contrast: 4×4 and 8×2 sit hundreds of KiB below the observed frontier, while BF8 RM 4×2 is on its edge. No CB total is reported for 2×2 because its earlier DEST assertion prevents a supported allocation from being formed.

### Live tensor consumers and ownership limits

The compact native input, table, counts, offsets, all weights and output are DRAM by the operation's validated contract. Moving those inputs to DRAM again cannot explain or repair this L1 clash. The winning BF8 input proposal changes the **native staging CB layout**, not the input tensor's existing memory tier.

A concrete persistent consumer is the native router. `TTMoEGate` creates four HEIGHT_SHARDED L1 templates: BF16 tiled bias, UINT16 tiled input indices, BF16 row-major output, UINT16 row-major output indices. Each reserves a 32×32 two-byte shard, **2,048 bytes per participating core**, or **8,192 bytes per core together** (`models/common/modules/moe/tt_moe_gate.py:267–340,458–465`). The inherited router configuration does not set `batch_per_device`, so these templates cover the device compute cores even for one-token decode (`A/tt/fused_decoder.py:613–622`). Router-produced scores/indices and other upstream allocations can coexist with them. Future sharded residual policies add different live consumers and require their own budget.

The observed frontier is **not** a named-tensor census. Available logs do not identify which root buffer owns address 1,556,416 or fully explain the remaining reservation. Do not assign that address to one particular tensor by arithmetic alone. Router templates establish real competing occupancy, not exclusive ownership of the entire 16,448-byte frontier displacement. Allocation order, bank rounding, padding, and other live buffers still matter.

Do not count every router temporary as live through FFN: `TTMoEGate.forward` explicitly deallocates `cur` and `logits` before returning (`tt_moe_gate.py:583–584`), and its frame-local working slices leave scope. The compact adapter retains the route tuple and creates a small UINT32 indices copy; route scores and IDs have only 16/32 bytes of logical payload, respectively, although allocation rounding is larger. Removing that redundant cast alone is not evidence of recovering 3,392 bytes per bank. The unsharded fused RMSNorm also returns DRAM BF16 (`A/tt/fused_decoder.py:288–295`); do not assume its hidden tensor is L1 in the failing policy.

## Minimal Python-only intervention

### Preferred discriminating change: BF8 tiled native input

After the existing row-major pad in `_OptimizedCompactExperts.__call__`, add this candidate transformation, optionally behind an explicit per-role policy:

```python
# `inputs` is the existing BF16 ROW_MAJOR DRAM tensor, shape (1, 1, 256, 2048).
inputs = ttnn.to_layout(
    inputs,
    ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat8_b,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

Keep `input_m_tiles=1`, `read_x_at_offset=False`, all routing descriptors, 256-row capacity, and the BF16 preallocated output unchanged. This is a proposed patch, **not applied**. `to_layout` forwards the dtype to tilize when the input is row-major (`ttnn/cpp/ttnn/operations/core/to_layout/to_layout_op.cpp:129–165`). Do not move this call onto an already tiled tensor and expect dtype conversion: same-layout `to_layout` returns its input and ignores a requested dtype/memory change (`:62–80`); use explicit `typecast`/`to_memory_config` for such an alternative construction.

On 4×2, `CB_X_IN` shrinks from 32 × 2,048 = 65,536 bytes to one BF8 tile = 1,088 bytes. **Re-run planner tuning in the calculation:** BF8 TILE retains down depth 3/nonresident but hidden depth increases back to 3, adding 34,816 bytes. Therefore:

```text
new total = 1,448,064 − 64,448 + 34,816 = 1,418,432
new end   = 111,744 + 1,418,432         = 1,530,176
margin against recorded frontier       =    26,240
```

The tiled case's down residency remains disabled because the early pressure checks still call `l1_bytes(true, ...)`, with its 1,088-byte `x_stick` treated as 32 row-major pages; later checks and final allocation use the actual tiled layout (`geometry.cpp:193–233`). This conservative early check is part of the prediction. Changing it in C++ would change the chosen depths and requires recomputing the live-capacity result.

The row-major native path already converts X to BF8 internally (`geometry.cpp:328`; compute:580–600; `moe_fused_swiglu_compute_helpers.hpp:214–222`). External conversion therefore retains the kernel's internal activation storage precision and removes repeated native tilize work, but exact rounding equivalence and total latency still need measurement. The conservative patch tilizes all eight capacity tile-rows once; only the first is read by active experts. Reducing that extra conversion work is a later optimization.

**Do not reduce the tensor to 32 logical rows.** Even with `read_x_at_offset=False`, the reader checks output region offsets against activation `ROW_CAPACITY` (`reader.cpp:523–540`). The last active slot starts at row 224 and needs one tile-row; capacity must remain at least 256. The actual X address remains zero-based (`:571–572`). A small input with the old offsets can skip selected experts or assert.

### Alternatives, in order of scope

1. **Use BF8 4×4**, already passing and faster than saved BF8 8×2 in this screen; this is the lowest-risk precision-preserving policy choice. Treat selection as measured configuration, not a silent override of an explicitly requested unsupported grid.
2. **Use homogeneous BF4 weights on 4×2 or 4×4**, already passing. The current native BF4 policy casts gate, up and down. On 4×2 the planner restores resident down depth 4 and hidden depth 3; total CBs are 1,321,088 bytes, end 1,432,832. Its actual saving over failing BF8 is 126,976 bytes, not the raw 196,608-byte weight saving at fixed buffer depths. Qualify accuracy beyond the short layer fixture.
3. **BF8 output is supported**, but changes another precision boundary. On 4×2 with BF8 weights/RM input, its smaller output tiles and legal phase alias yield predicted total 1,342,592/end 1,454,336. Both `ttnn.empty(dtype=...)` and the native call's `dtype` must agree. The native fast reducer accepts BF8 input and returns that dtype (`.../reduction/deepseek_moe_fast_reduce_nc_fused/device/*device_operation.cpp:59,140`). Explicit BF16 conversion afterward can preserve the external dtype but cannot recover BF8 rounding. Lower priority than the input-only test.
4. **Reduce persistent router L1 occupancy** if preserving RM input becomes necessary. For example, local adapter changes can keep router output templates in DRAM and stage sharded working buffers only during routing. Two 2,048-byte/core templates exceed the nominal 3,392-byte deficit. This requires a measured allocation/lifetime check: freeing an upper-address buffer need not raise the lowest occupied address, and reducing `batch_per_device` alone does not guarantee a lockstep per-bank frontier improvement. Do not force-deallocate live router aliases or traced buffers. This is more invasive than the input-layout fix.

BF4 activation, BF16 TILE native activation, an L1 native input/output, or a mixed BF4 gate/up plus BF8 down **in this fused op** violate validation. Do not try them as capacity workarounds. The sparse backend is a different operation and can choose weight precision independently per call.

## Why the native FFN remains expensive

The approximately 410 µs claim is supported by existing Stage-02 device profiler CSVs, not inferred from the whole-layer screen. Re-reading the 128 native-op rows between `PERF_DECODE` and `PERF_DECODE_END` gives:

| Existing Stage-02 artifact under `A/doc/fused_decoder/` | Mean kernel µs | Mean firmware µs |
|---|---:|---:|
| `profile_final/tracy/linear_attention/ops.csv.gz` | 408.653 | 411.022 |
| `profile_placement_final/tracy/full_attention/ops.csv.gz` | 411.402 | 414.171 |

These figures describe the earlier 4096-input/128-decode profile. They are not device measurements of the new 4×4/BF4 candidates. Op intervals can overlap; do not add profiler rows as a substitute for the complete-layer denominator.

Source explains several fixed costs:

- The native reader loops over **all 256 local expert descriptors** on every core. For each it reads the 1,024-byte index page and 64-byte counts page, then the 64-byte offsets page even when the selected count is zero, and publishes writer/compute mailboxes (`reader.cpp:427,490–568`). That is 294,912 bytes of requested metadata reads per core per invocation, before cache/transaction effects. Only eight experts execute matrix work; this is not 256 FFNs.
- For each of eight active experts, runtime M is one physical 32-row tile, although one row is useful. Gate/up requires row-split partial reductions, hidden scatter/gather and cross-core transfers before down projection. X is reread and, on RM input, converted for each expert.
- Per active expert, 4×2 has 128/128/128 gate/up/down tile products per core; both 16-core grids have 64/64/64. The latter share arithmetic totals but differ in communication: 4×4 has four down hidden rounds and four row contributors; 8×2 has eight down rounds and two row contributors. Smaller grid is not automatically faster.
- Useful selected weight payload is 8 × 3 × 1,024 BF8 tiles = 26,738,688 bytes (25.5 MiB). Dividing by about 411 µs gives approximately 65 GB/s of *effective selected-weight payload*, not a measured DRAM counter or proof of bandwidth saturation. Control-plane, NoC, reduction, tilize and compute time are not separated by the existing aggregate row.

Native stage profiling (`MOE_FUSED_SWIGLU_STAGE_PROFILE`, factory:30–36) can distinguish X staging/tilize, gate/up reduction, and down phases under the hardware owner's diagnostic run, followed by normal unprofiled whole-layer timing. **The existing zones do not directly time the descriptor scan/count/offset/mailbox phase** at reader:490–568; the first active-block reader zones are later. Remaining time is unattributed, not proof that descriptor scanning dominates. Direct scan attribution would require additional diagnostic instrumentation outside the requested Python-only fix. No claim here establishes which phase dominates the 410 µs.

## Sparse-matmul comparison

The inspected `_OptimizedSparseExperts` uses broadcast dense A for packed gate/up, then sparse A for down. Logical decode M=1 is padded to one 32-row tile. Packed gate/up has K=2048 (64 tiles), N=1024 (32 tiles); down has K=512 (16 tiles), N=2048 (64 tiles). Its masks contain exactly eight literal ones from distinct top-k IDs. Keep that mask independent of route scores: a low-precision score becoming zero must not disagree with `nnz=8` and the kernels' loop counts.

Four older logs (`sparse_bfp8_c32_k2`, `c32_k8`, `c16_k16`, `c8_k16`) end with `TypeError: 'MoEConfig' object is not callable` at `program_config=self.config(...)`. The inherited instance field shadowed the helper. They are **Python failures, not invalid geometry or L1 evidence**. The current helper is named `program_config`; rerun those cells before interpreting them as rejected candidates.

The effective mask-mode geometries are below. `c` is the number of N-partitioned cores; eight selected experts execute as successive batches on that grid, not as eight independent grids. All use output M block/subblock one tile. Packed gate/up N is 32 tiles and down N is 64 tiles.

| Family | Gate grid | Gate N block / subblock | Gate K block / iterations | Down grid | Down N block / subblock | Down K block / iterations |
|---|---|---|---|---|---|---|
| c32/k2 | 8×4 | 1 / 1 | 2 / 32 | 8×4 | 2 / 2 | 16 / 1 |
| c32/k8 | 8×4 | 1 / 1 | 8 / 8 | 8×4 | 2 / 2 | 16 / 1 |
| c16/k16 | 8×2 | 2 / 2 | 16 / 4 | 8×2 | 4 / 4 | 16 / 1 |
| c8/k16 | 8×1 | 4 / 4 | 16 / 4 | 8×1 | 8 / 4 | 16 / 1 |
| c8/k32 | 8×1 | 4 / 4 | 32 / 2 | 8×1 | 8 / 4 | 16 / 1 |

Thus the default sparse gate K block 2 produces 32 K iterations per selected expert, whereas K block 16 produces four. Larger K blocks trade CB space for fewer iterations; reducing core count also reduces N parallelism. The measured c32/k2 versus c32/k8 improvement is consistent with this tradeoff, but is not an isolated measurement of spill cost.

Sparse source references: **S/** means `ttnn/cpp/ttnn/operations/matmul/device/sparse/`. Geometry and buffering are `S/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp:179–330`; shape/block validation is `S/sparse_matmul_device_operation.cpp:403–432`.

Saved successful nonsplit candidates include:

| Candidate | Whole-layer median µs | Minimum token PCC |
|---|---:|---:|
| BF8 gate/up, c8 k32 | 944.570 | 0.9993940 |
| BF4 gate/up, c32 k2 | 1043.023 | 0.9993940 |
| BF4 gate/up, c32 k8 | 883.473 | 0.9993940 |
| BF4 gate/up, c16 k16 | 874.686 | 0.9993940 |
| BF4 gate/up, c8 k16 | 943.157 | 0.9993940 |

These sparse BF4 trials retain default BF8 down weights. Native compact BF4 lowers down as well. In addition, inherited gate/up weights were originally loaded as BF4 (`models/demos/blackhole/qwen36/tt/moe/weights.py:57–58,108–142`); widening them to BF8 cannot recover lost checkpoint precision. This accounts for unequal precision contracts and cautions against attributing PCC differences solely to backend.

The current mask mode skips unselected expert matrix work but materializes the 256-expert output axis. At BF16 TILE with padded M=32, packed gate/up is **16 MiB**, activated intermediate **8 MiB**, and down **32 MiB**, before other temporaries. `both`, `activated`, `down`, and weighted results can coexist in the Python frame. Output clearing, SwiGLU, weighting and reduction still traverse large materializations. Geometry alone does not eliminate this cost.

### Compact indexed sparse mode: stronger Python-only candidate

The existing indexed `sparse_matmul` path is a supported alternative to the dense expert mask output. Its in-tree contract is exercised by `tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul_indexed.py:68–144`:

1. Convert the eight route IDs to same-device **UINT16 ROW_MAJOR**, a single stick `(1,1,1,8)`. Pass `indices=...`, **omit `nnz`**, and retain a valid sparsity tensor with logical volume 256 (indexed mode obtains its selection from indices).
2. Gate/up: `is_input_a_sparse=False`, `is_input_b_sparse=True`; broadcast the token across selected weights. Output order is route-slot order and compact shape becomes `(1,8,1,1024)` after reshape.
3. Apply SwiGLU to obtain `(1,8,1,512)`.
4. Down: `is_input_a_sparse=True`, `is_input_b_sparse=True`; A is read by compact ordinal slot i, while B expert selection uses `indices[i]`. Output is `(1,8,1,2048)`.
5. Multiply directly by `permute(values,(0,3,2,1))`, then reduce expert dimension 1. Do not re-expand to 256 experts or independently sort IDs and scores.

This changes the corresponding outputs to **0.5/0.25/1 MiB**, a 32× reduction in materialized expert extent. Current output creation still zero-initializes indexed outputs; do not claim it eliminates zero-fill altogether. It avoids the native 256-descriptor scan and preserves eight selected experts, but adds distinct matmul/activation/reduction launches. It remains an unmeasured candidate in this report.

The lowered mapping is explicit: index validation is `S/sparse_matmul_device_operation.cpp:230–289`, compact output shape `:455–468`, and output clearing `:491–540`. Input-A compact-slot addressing is in `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding_metal2.cpp:127–135,455–461`; selected B-expert addressing is in the sibling `reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp:349–368`.

After an indexed baseline using the measured c16/k16 geometry, source-legal candidate tuples `(cores, Kblock, Nsubblock)` are:

| Gate tuple | Down tuple | Discriminating change |
|---|---|---|
| (16,32,2) | (16,16,4) | Gate K iterations 4 → 2 |
| (16,64,2) | (16,16,4) | One gate K iteration |
| (32,32,1) | (32,16,2) | More N parallelism, two gate K iterations |
| (32,64,1) | (32,16,2) | More N parallelism, one gate K iteration |
| (16,32,2) | (32,16,2), then (64,16,1) | Tune down independently of gate |

These satisfy K divisibility, nonzero/divisible block/subblock sizes and rectangular grids in the source; they are not hardware-qualified. For BF16 A/BF4 B at gate c16, the leading double-buffered A/B plus output CB estimate grows from **104 KiB/core at Kblock16** to **404 KiB/core at Kblock64** (factory:251–271), excluding other CBs and live allocations. Compact intermediate outputs make the larger-block trials more plausible, but capacity must still be checked.

## Prioritized experiments for the hardware owner

All experiments below remain unexecuted by AutoDebug. Keep source and command manifests per run, fresh artifact names, recorded activations, route pairing, serialized device access, and explicit completion boundaries. No broad matrix is needed before the first small discriminating cells.

| Priority | Change / control | Prediction and decision |
|---|---|---|
| 1 | Same-source native BF8 4×2 RM failure control, then the single BF8 TILE input conversion | Current failure should match the reconstructed end; tiled native end should be 1,530,176 with the stated planner configuration. Record actual frontier as it may change. A different failing end falsifies the assumed lowering; a lower frontier identifies additional live pressure. |
| 2 | Same-source BF8 TILE 4×4 and RM 4×4, plus BF4 RM 4×2/4×4 | Separate capacity repair from latency benefit. Compare minimum per-token PCC, repeated replay equality and complete-layer time, then both full-attention and GDN layers. The old timing rank is a screening result, not a universal winner. |
| 3 | Snapshot L1 immediately before native launch in a diagnostic run | Record allocator bank/frontier information and root buffer IDs for router templates, scores/indices and upstream tensors. Reconcile the 16,448-byte displacement; establish which owner must move before considering lifetime changes. No forced release of referenced/traced allocations. |
| 4 | Rerun previously TypeError-blocked sparse geometries with fixed `program_config` | Distinguish device acceptance/correctness from old Python failure. Use larger K blocks as controlled contrasts and separate gate/down role choices. |
| 5 | Indexed compact sparse, initially packed gate/up BF4 and down BF8, then controlled geometry sweep | Verify compact shape, ID order and per-slot outputs before whole-layer reduction. Measure whether reduced materialization beats native fusion. |
| 6 | Diagnostic native stage profile on accepted RM and tiled configurations | Compare the instrumented conversion, weight transport and reduction phases; leave the unzoned descriptor phase unattributed. Remove profiling from final timing. |
| 7 | BF8 output or router template migration only if needed | Additional precision/lifetime changes need their own matched baseline; do not combine them into the first BF8 input experiment. |

For acceptance, use the existing per-token PCC ≥0.995 check without relaxing it, compare replayed results after changing tokens/positions and expert selections, and expand to required stage workloads after a candidate passes the small screen. The provided test's host timing blocks on each trace replay; no extra synchronizations belong inside accepted latency windows.

## Claim review and other potential issues

After the first draft, the report was rechecked against the lowered source, including independent geometry and sparse-path reviews. That review corrected an overstatement about stage profiling: descriptor scanning is unzoned and remains unattributed. A separate CPU-only calculation verified all twelve supported-layout CB totals. No target or device test was run. The principal falsification checks were:

| Candidate explanation | Adjudication |
|---|---|
| All small-grid failures are L1 exhaustion | **False.** 2×2 is a DEST geometry rejection; BF4 4×2 succeeds. |
| The final L1 validator is overly conservative or corrupt | **Unsupported.** Its comparison exactly matches the actual CB sum and observed frontier. |
| Fixing the 960-byte scratch underestimate alone fixes 4×2 | **False.** Actual CBs still fit nominal capacity by 13,056 bytes; live capacity is the decisive missing constraint. |
| BF8 input saves 64,448 bytes net | **False.** That is the X buffer delta; hidden depth retuning recovers 34,816 bytes, leaving 29,632 net. |
| BF4 weights save 196,608 bytes net on failing 4×2 | **False at the final layout.** Restored buffer depths leave 126,976 net. |
| A 32-row input is sufficient because X is read at row zero | **False under this reader contract.** Region validation still needs 256 rows for output slot 7. |
| Padding capacity 256 means eight tile-rows of FFN arithmetic | **False.** Runtime count/input_m_tiles select one tile-row; static buffers remain sized for eight. |
| A particular router buffer owns the logged lowest address | **Unproven.** Only a live allocation census can establish that identity. |
| The 410 µs native duration is established for every Stage-03 candidate | **False.** It is supported for the cited Stage-02 profiles only. |
| Failed sparse log names prove bad matmul geometries | **False for four BF8 rows.** Python raises before invoking the operation. |
| Lower top-k order itself causes bad reduction | **No contradiction found.** Current slot/count/output/score mapping preserves pairing. |

Other issues are separate from the reported allocation failure:

- The generic native tests cover BF8 tiled input, but their main helper uses one expert and default output, rather than this 256-descriptor compact-offset/BF16-output path (`test_moe_fused_swiglu.py:185–214,314–327`). This is a coverage gap; it does not refute source support for the candidate.
- The saved `split_bfp4.json` / `split_bfp8.json` have minimum decode PCC about **0.332661**, despite deterministic replay and approximately 871–873 µs host timing. They are failed correctness trials. The old candidate requested fused SiLU in its matmul program config, but the sparse factory hardcodes `FUSE_ACTIVATION="0"` (`S/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp:525`) and descriptor construction supplies `user_fused_activation=nullopt` (`S/sparse_matmul_device_operation.cpp:654`). The selected compute kernel applies SFPU activation only under `SFPU_ACTIVATION`, which that sparse factory does not define (`ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp:288,467,654`). The omitted SiLU is a concrete explanation for those old split results, separate from native L1. The hardware owner was editing the Python activation consumer during investigation; the artifacts must stay bound to their recorded `4a86f49d...` source rather than being assigned to later code. Explicitly applied SiLU needs its own evidence.
- A robust native admission fix would use actual scratch pages and live allocator headroom, while accounting for guaranteed allocations after admission and cache behavior. That would be a C++ change outside the requested scope. The supported layout/configuration adaptation is the practical intervention here.

No hardware claim above is promoted from a predicted byte count or source-supported API alone. The report distinguishes the observed matrix, exact reconstruction of the failure, and adaptations requiring parent-owned execution.
