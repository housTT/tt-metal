# AUTODEBUG — Qwen3.6-27B datatype sweep: `canonical_bfp8_hifi2_kv_bf16` traced-decode warmup L1/CB clash

**Scope.** Source-only diagnosis. No Tenstorrent hardware, device tests, `tt-smi`, or
device-initializing commands were run. Every quantity below is derived from checked-in
source, and the derivation reproduces the reported failure numbers **exactly** — and
independently reproduces the numbers in two other checked-in failure logs.

**Working tree inspected.** Branch `agentic-research/hous/qwen3.8-27b` at `b7b52f83305`
plus the uncommitted precision-policy refactor and the untracked
`models/autoports/qwen_qwen3_6_27b/doc/datatype_sweep/` and `tt/precision.py`. The tree
was being edited concurrently while this ran; §7.1 notes one finding remediated
mid-investigation.

**Failure under diagnosis** (primary artifact:
`doc/datatype_sweep/evidence/candidates/canonical_bfp8_hifi2_kv_bf16/failure.json`):

```
phase:     warmup traced decode before accuracy measurement
exception: RuntimeError: TT_THROW
signature: Statically allocated circular buffers in program 606 clash with L1 buffers
           on core range [0-0 - 7-9]. L1 buffer allocated at 928000 and static
           circular buffer region ends at 1333760
source:    models/autoports/qwen_qwen3_6_27b/tt/optimized_decoder.py:_mlp down ttnn.linear
```

---

## TL;DR

**This is a known, already-diagnosed, already-fixed failure that was only half-fixed.**

`doc/multichip_decoder/autofix/full_decode_l1/AUTOFIX.md` records the *identical*
`TT_THROW` at the *identical* call site with the *identical* CB region end
**`1,333,760`**. Its accepted fix (H3) padded the TP-local MLP intermediate
`4,352 → 4,608` so 16 compute cores become legal — and applied it to **full-attention
layers only**, stating explicitly:

> “The linear-attention 4,352/8-core path and public 5,120-wide residual/output contract
> are unchanged.” — `AUTOFIX.md:104-106`

That was safe **only because linear-attention MLP weights are BFP4**
(`doc/optimized_full_model/README.md:80`, “Linear-attention MLP: BFP4 gate/up/down”).
The sweep candidate flips exactly that field:

```json
"mlp_down": {"linear_attention": "bfp8", "full_attention": "bfp8"}
```
`doc/datatype_sweep/candidates/canonical_bfp8_hifi2_kv_bf16.json:5`

The linear down projection's weight CB is triple-buffered
`per_core_N_in1_sender(20) × in0_block_w(17) = 340` tiles, so it scales directly with the
weight tile size:

| `mlp_down.linear_attention` | in1 tile | in1 CB | static CB region end |
|---|---:|---:|---:|
| `bfp4` (baseline selected policy) | 576 B | 587,520 B | **811,520** |
| `bfp8` (canonical candidate) | 1,088 B | 1,109,760 B | **1,333,760** ← reported |

`1,333,760` is reproduced byte-for-byte from source (§2). The bundled `hifi2`,
`kv bf16`, and CCL changes are **not** causal — each is independently exonerated by a
sibling candidate that ran successfully (§4).

---

## 1. Observations vs. interpretations

**Direct observations** (report + checked-in artifacts):

1. The `TT_THROW` above: program 606, core range `[0-0 - 7-9]`, lowest L1 buffer
   `928000`, CB region end `1333760`, at `_mlp`'s down `ttnn.linear`, during warmup
   traced decode.
2. `evidence/candidates/canonical_bfp8_hifi2_kv_bf16/` contains only `failure.json`. The
   **five other candidates all have `teacher_forcing_metrics.json` and passed.**
3. The same failure class has occurred **three times** in this model's recorded history,
   twice with the identical region end:
   * `doc/multichip_decoder/autofix/full_decode_l1/AUTOFIX.md:36-38` — full-attention
     decode, program 204, `_mlp` down `ttnn.linear`, **CB end `1,333,760`**.
   * `doc/optimized_decoder/correctness/final_sharded/run.log:868` — single device,
     program 18590, `_mlp` down, `[17408, 5120]` **BFLOAT8_B** weight,
     **CB end `1,333,760`**.
   * `doc/optimized_decoder/candidates/dram_sharded_mlp/geometry/16_linear_attention.log:37`
     — `_mlp` gate, BFP4, `in0_block_w=10`, CB end `1,474,816`.
4. `doc/optimized_multichip_decoder/README.md:143-152` records the decode geometry table;
   every linear-attention MLP row is tuned under **“BFP4 / LoFi”**, and the down row lists
   its only two legal `in0_block_w` values as `17, 1`, selecting 17 on latency
   (60.470 µs vs 114.866 µs).

Everything else below is derivation, marked as proven or as hypothesis.

---

## 2. PROVEN — the arithmetic that produces 1,333,760 exactly

### 2.1 The lowered configuration

`_mlp` issues the down projection at `tt/optimized_decoder.py:965-981`. For a
`linear_attention` layer on the 1x4 mesh:

| quantity | value | source |
|---|---|---|
| `hidden_size` | 5120 | `tt/multichip_decoder.py:196` |
| local intermediate | **4352** (`17408/4`, *not* padded) | `tt/multichip_decoder.py:206-211` |
| `dram_sharded_mlp_cores["down"]` | **8** (16 only for `full_attention`) | `tt/multichip_decoder.py:486-489` |
| `dram_sharded_mlp_blocks["down"]` | **17** (`QWEN36_MC_MLP_DOWN_BLOCK` default) | `tt/multichip_decoder.py:490-501` |
| `in0_block_w` / `per_core_M` / `per_core_N` | 17 / 1 / 20 | `tt/optimized_decoder.py:890-912` |
| `packer_l1_acc` / `fp32_dest_acc_en` | True / False | `tt/multichip_decoder.py:466-481` |

### 2.2 What the DRAM-sharded factory actually does

`ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp`:

* Worker cores are the **DRAM readers**, not the model's `cores` value.
  `num_dram_banks = 8` on Blackhole (`tt_metal/soc_descriptors/blackhole_140_arch.yaml`,
  8 `dram_views`; `tt_metal/impl/device/device.cpp:587`), so
  `per_core_N_compute = div_up(Nt=160, 8) = 20` (factory:144). **The program config's
  `per_core_N=20` is not this number** — it is `per_core_N_storage`, used only for the
  globally-allocated output-reshard CB. (They coincide here at 8 cores; for
  `full_attention` down the config says 10 while the CB is sized at 20.)
* `per_core_N_in1_sender = 20` is captured **before** the sub-block widening loop
  (factory:145). The loop (factory:147-170) takes
  `get_matmul_subblock_params(1,20,…) = (1,5)`
  (`config/matmul_program_config.cpp:16-25, 946-981`) → `out_subblock_w = 7`,
  `per_core_N_compute = 21`.
* `num_blocks = Kt/in0_block_w = 136/17 = 8`; `packer_l1_acc_en = true`;
  `interm0 = Float16_b == output_data_format`, so out and interm **share** one CB
  (factory:181-186, 554, 578-591).
* CB sizes (factory:203-216):
  * `in0_CB  = 2 · per_core_M · in0_block_w · 2048 = 2·1·17·2048 = 69,632`
  * `in1_CB  = 3 · per_core_N_in1_sender · in0_block_w · tile(in1) = 3·20·17·tile`
    (the `3×` is hard-coded, factory:209-212)
  * `out/interm CB = per_core_M · per_core_N_compute · 2048 = 21·2048 = 43,008`
* CB2 (in0) and CB6 (out-reshard) set `CBDescriptor.tensor`, so they are **globally
  allocated** and contribute **0** to the region
  (`tt_metal/impl/buffers/circular_buffer_config.cpp:65-93`,
  `tt_metal/impl/program/program.cpp:1331-1342`).
* All CBs are created on `all_cores_in_rect_grid` = the **bounding box** of
  (in0 storage cores ∪ DRAM readers) (factory:277-281, 517-614) — the core range printed
  in the throw.

### 2.3 The base address

`DEFAULT_UNRESERVED = ((MEM_MAP_END + 69·1024 − 1) | 63) + 1` with `MEM_MAP_END = 40,704`
(`MaxDMProcessorsPerCoreType = 2`,
`tt_metal/hw/inc/internal/tt-1xx/blackhole/core_config.h:35`) and
`max_alignment = max(DRAM_ALIGNMENT 64, L1_ALIGNMENT 16) = 64`
→ **`l1_unreserved_base = 111,360`**
(`tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal_tensix.cpp:68-69`; round-trips through
`tt_metal/impl/allocator/l1_banking_allocator.cpp:228-229`).
`MEM_L1_SIZE = 1,572,864`. Each CB *start* is 64-aligned
(`tt_metal/impl/program/program.cpp:1355`); all three sizes and the base are already
64-multiples, so alignment contributes **exactly 0**.

### 2.4 Result

```
bfp8:  111,360 + 69,632 + (3·20·17·1088 = 1,109,760) + 43,008 = 1,333,760   ← reported
bfp4:  111,360 + 69,632 + (3·20·17·  576 =   587,520) + 43,008 =   811,520
```

Two independent cross-checks validate the whole model against checked-in logs:

* `final_sharded/run.log:868` — same op, single device, `K/core = 17` tiles, `N = 5120`,
  BFLOAT8_B, `in0_block_w=17`, `per_core_N=5` → same **1,333,760**. (`per_core_N` differs
  yet the total is identical — itself proof that `per_core_N` is `per_core_N_storage`
  only.)
* `geometry/16_linear_attention.log:37` — gate projection, `Kt=160`, `Nt=544`, BFP4,
  `in0_block_w=10`, `per_core_N_compute` widened 68→72:
  `111,360 + 40,960 + (3·68·10·576 = 1,175,040) + (72·2048 = 147,456) = 1,474,816` ✔

### 2.5 Why `[0-0 - 7-9]` pins this to a *linear-attention* layer

`CoreRange::str()` prints `"[x-y - x-y]"` (`tt_metal/common/core_coord.cpp:115`;
`tt_metal/third_party/umd/device/types/xy_pair.cpp:11`). On Blackhole the 8 DRAM reader
cores map to logical `{(0,0),(0,3),(0,7),(0,9),(7,1),(7,4),(7,6),(7,9)}` → max x = 7,
max y = 9 (`tt_metal/common/core_assignment.cpp:160-196` + `blackhole_140_arch.yaml`).

The down projection's in0 is `activated`, on `_decode_l1_memory(32, intermediate, cores)`
(`tt/optimized_decoder.py:62-84`):

* `cores = 8` (linear_attention) → `CoreRange((0,0),(7,0))` → bounding box **`[0-0 - 7-9]`** ✔
* `cores = 16` (full_attention) → `(0,0)-(10,0) ∪ (0,1)-(4,1)` → bounding box `[0-0 - 10-9]` ✘

So the failing layer is `linear_attention` — consistent with `mlp_down.full_attention`
already being `bfp8` in the baseline and working.

---

## 3. PROVEN — the candidate cannot fit under any allocation ordering

`lowest_address` in the throw is device-wide, read from **bank 0**
(`tt_metal/impl/sub_device/sub_device_manager_tracker.cpp:141-152`,
`tt_metal/impl/program/program.cpp:1448`), and an 8-core width-sharded tensor reserves its
shard size on **every** bank. That is provable, not assumed:
`AllocatorImpl::allocate_buffer` passes `num_cores` as `num_shards`
(`tt_metal/impl/allocator/allocator.cpp:170-177`); `BankManager::allocate_buffer` computes
`size_per_bank = calculate_bank_size_spread(...)` = exactly one shard
(`bank_manager.cpp:421-434`, `allocator.cpp:587-597`); and that one address comes from a
single `FreeListOpt` shared by all L1 banks, each with `bank_offset = 0`
(`bank_manager.cpp:445`, `l1_banking_allocator.cpp:101`). L1 buffers grow **top-down**
(`tt_metal/impl/buffers/buffer.cpp:289`).

* CB budget actually available: `928,000 − 111,360 = 816,640 B`.
* `bfp4` needs `700,160 B` → fits, **116,480 B** spare.
* `bfp8` needs `1,222,400 B` → overflows by **405,760 B**.

Note the raw `cb_region_end > max_l1_size` branch (`program.cpp:1486`) is **never** hit:
`1,333,760 < 1,572,864`. This is purely a **co-residency** failure, which is exactly why
it only surfaces in traced-decode warmup once the persistent CCL pool and the sharded
activations are live.

### 3.1 The observed `928,000` is reconstructed from source to within 2.7%

Census of every live L1 buffer at the failing enqueue, per bank:

| group | items | B/bank |
|---|---|---:|
| Persistent CCL pool (never freed) | `attention_16` 81,920 + `linear_mlp_8` **163,840** (bf16) + `full_mlp_16` 81,920 (`tt/multichip_decoder.py:627-651`) | **327,680** |
| 8-core MLP working set | `hidden_states`, `mlp_input`, `down` output @ 40,960; `gate`, `up`, `activated` @ 34,816 | 227,328 |
| 16-core attention output | `mixed` pre- and post-all-reduce @ 20,480 (`tt/optimized_decoder.py:1460-1465`, `tt/multichip_decoder.py:680-693`) | 40,960 |
| 32-core residual/norm | `residual`, `normalized` @ 10,240 (`tt/optimized_decoder.py:436-450`) | 20,480 |
| RoPE `cos`/`sin` | 32-core height-sharded @ 4,096 | 8,192 |
| 39 global semaphores | `tt_ccl.py:91-104` (36) + `tt/multichip_decoder.py:620-625` (3), 64 B each | 2,496 |
| **total live** | | **627,136** |
| observed span `1,572,864 − 928,000` | | 644,864 |
| residual | allocator fragmentation (top-down first-fit, `bank_manager.cpp:492-500`) | 17,728 (2.7%) |

Confirmed **not** in L1: all weights including the DRAM-sharded decode copies
(`tt/multichip_decoder.py:64-107`), paged KV cache and DeltaNet conv/recurrent state, the
inter-layer residual (`tt/optimized_decoder.py:1033` forces DRAM), the trace buffer
(`BufferType::TRACE`, reported as DRAM at `buffer.cpp:553`), the whole sampling path, and
the sub-device reservation (`local_l1_size = 0`,
`sub_device_manager_tracker.cpp:101-103`).

### 3.2 No allocation ordering can rescue it

A CB region ending at `1,333,760` leaves `1,572,864 − 1,333,760 = 239,104 B` for all L1
buffers device-wide. The **irreducible** live set at that instant — assuming perfect
ordering and deallocating everything not semantically required — is:

| must be live | B/bank |
|---|---:|
| persistent CCL pool (the very next op is the MLP all-reduce that consumes it) | 327,680 |
| `activated` — the matmul's own in0 | 34,816 |
| `down` output | 40,960 |
| `hidden_states` — needed for the final residual add (`tt/optimized_decoder.py:1033`) | 40,960 |
| **minimum possible** | **444,416** |

`444,416 > 239,104` — **over by 205,312 B (1.86×)**. Even with the baseline's bfp8 CCL
pool the minimum is `367,616`, still over by 128,512. Even if literally everything but the
pool were freed, `327,680 > 239,104`. **Shrinking or re-typing the CCL pool cannot fix
this**; only shrinking the CB region can.

(For scale: `bf16` linear-down weights would give `in1_CB = 2,088,960` and a CB region end
of `2,312,960` — above total L1, i.e. rejected outright by the
`cb_region_end > max_l1_size` branch.)

---

## 4. PROVEN — fidelity, KV dtype, and CCL dtype are *not* causal

The candidate changes four things at once. The sweep's own evidence isolates three:

| candidate | `mlp_*.linear` | fidelity | kv | CCL | outcome |
|---|---|---|---|---|---|
| baseline `baseline_optimized_mixed` | bfp4 | lofi | bfp8 | linear-mlp bfp8 | pass, 97% top-1 |
| `projection_hifi2` | bfp4 | **projection hifi2** | bfp8 | — | pass, 96% |
| `full_down_bfp8_hifi2` | bfp4 | **mlp_down full hifi2** | bfp8 | — | pass, 97% |
| `kv_bf16_control` | bfp4 | lofi | **bf16** | — | pass, 96% |
| `ccl_all_bfp8` | bfp4 | lofi | bfp8 | **all bfp8** | pass, 97% |
| `full_down_bfp4_lofi` | bfp4 | lofi | bfp8 | — | pass, 95% |
| **`canonical_bfp8_hifi2_kv_bf16`** | **bfp8** | hifi2 | bf16 | all bf16 | **crash** |

`MathFidelity` appears in no CB-size expression in the factory (page sizes come from
tensor data formats; `interm0` depends only on `packer_l1_acc`/`fp32_dest_acc_en`), so it
cannot change L1 occupancy — only numerics and speed. The KV cache is DRAM-paged
(`kv_cache.layout = tile_dram_paged`) and never occupies L1.

**The single field distinguishing the crashing candidate from every passing one is
`weight_groups.mlp_*.linear_attention: bfp4 → bfp8`.**

Sweeping every DRAM-sharded decode matmul under both weight dtypes shows exactly one row
can clash:

| matmul (M×K×N, cores, block) | bfp4 CB end | bfp8 CB end |
|---|---:|---:|
| linear gate/up 32×5120×4352, 8c, blk10 | 482,944 | 744,064 |
| **linear down 32×4352×5120, 8c, blk17** | **811,520** | **1,333,760 ← clash** |
| full gate/up 32×5120×4608, 16c, blk10 | 500,224 | 776,704 |
| full down 32×4608×5120, 16c, blk9 | 502,272 | 778,752 |
| linear packed input 32×5120×4352, 8c, blk10 | 482,944 | 744,064 |
| linear/attn output 32×1536×5120, 16c, blk3 | 270,336 | 362,496 |
| full packed QKV 32×5120×3584, 16c, blk10 | 422,912 | 637,952 |

Under the canonical policy the down projection is simply the **first** to cross: six other
matmuls sit within ~150–185 KB of the floor. This configuration is at the edge
everywhere, not just at one call site.

---

## 5. Headline findings

### F1 (root cause) — the accepted fix for this exact failure was applied to full-attention layers only; linear-attention layers still run the geometry that works only with BFP4

`doc/multichip_decoder/autofix/full_decode_l1/AUTOFIX.md` diagnosed and fixed this precise
`TT_THROW` (same `_mlp` down `ttnn.linear`, same CB region end `1,333,760`,
`AUTOFIX.md:36-38`). The retained fix, H3, was:

> “padding each **full-attention** TP-local MLP intermediate shard from 4,352 (136 tiles)
> to 4,608 (144 tiles) permits 16 compute cores … **The linear-attention 4,352/8-core
> path … [is] unchanged.**” — `AUTOFIX.md:95-106`

That is exactly the shape of today's code: `FULL_LOCAL_INTERMEDIATE_SIZE = 4608`
(`tt/multichip_decoder.py:39`) is applied only when `layer_kind == "full_attention"`
(`:206-211`), and `mlp_cores = 16 if layer_kind == "full_attention" else 8` (`:486`).

Linear layers therefore keep `K_tiles/cores = 136/8 = 17`, which is **prime**, so the only
legal `in0_block_w` values are `{1, 17}` — verified against both the Python constraint
(`tt/optimized_decoder.py:905-906`) and the kernel contract
(`matmul_device_operation.cpp:1377, 1382`), and matching the hard-coded set in
`tests/test_multichip_decoder.py:756`. Block 17 was selected on latency and hard-coded
(`tt/multichip_decoder.py:490-501`); even with the default removed, the fallback picks the
**largest** legal divisor (`next(d for d in (68,34,20,17,…) if blocks % d == 0)` → 17), and
the sibling helper's docstring states the policy outright: *“Run a decode projection with
the largest legal common K/N core factor”* (`tt/optimized_decoder.py:482-483, 495-500`).

Because `in1_CB = 3 · 20 · in0_block_w · tile_size`, **the block size that is optimal for
BFP4 is precisely the one that makes BFP8 unschedulable**, and the geometry code never
reads `self.mlp_weight_dtypes[role]`. The precision refactor made the weight dtype
user-selectable per layer kind (`tt/multichip_decoder.py:248-257`, `tt/precision.py:118-131`)
while leaving the geometry frozen at its BFP4-tuned constants. The prior remediation was a
point workaround for one layer kind, not a guard, so the next dtype change re-opened it.

### F2 — TTNN owns an L1 admission check for matmul, and the DRAM-sharded path is excluded from it at three separate points

TTNN already computes the exact quantity that later throws:
`utilities::get_max_l1_space()` reads **the same** `lowest_occupied_compute_l1_address` and
subtracts the allocator base
(`ttnn/cpp/ttnn/operations/matmul/device/utilities/matmul_utilities.cpp:79-85` →
`928,000 − 111,360 = 816,640`), and `can_cbs_fit_in_l1()` compares it against a summed CB
estimate (`config/matmul_program_config.cpp:179-202`). It never runs here:

1. **Single unreachable call site.** `can_cbs_fit_in_l1` is called only at
   `matmul_program_config.cpp:460`, inside `create_matmul_program_config`, on the
   `input_b_is_batched && !a_is_sharded && !b_is_sharded` branch.
2. **User-config early return.** `get_program_config` returns the caller's config
   immediately (`matmul_program_config.cpp:992-994`), so `generate_matmul_program_config`
   — and every `can_cbs_fit_in_l1` call — is skipped. `_mlp` and `_dram_projection`
   **always** pass a program config.
3. **Explicit type opt-out in the generic sanity checks.**
   `matmul_program_config.cpp:1009-1016` guards with
   `not std::is_same_v<ProgramConfigType, MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>`.

Op `validate` for this config type (`matmul_device_operation.cpp:1339-1392`) checks shard
layout, `M == per_core_M`, `M == 1`, `K % in0_block_w == 0` and
`(shard_w_tiles) % in0_block_w == 0` — **nothing about L1, in1 dtype, or CB bytes**. The
factory computes `in0_CB_size`/`in1_CB_size`/`out_CB_size` (factory:200-217) and asserts
nothing about them. The first and only real check is
`validate_circular_buffer_region` at `tt_metal/impl/program/program.cpp:1444-1549`, inside
`MeshWorkloadImpl::compile` → `EnqueueMeshWorkload`.

Notably, **even TTNN's deliberately imprecise estimator would have caught this**:
`get_estimated_size_of_cbs` (`matmul_utilities.cpp:18-64`) uses
`MCAST_INPUT_BUFFERING_DEPTH = 2` rather than 3 and the unaligned tile size, and still
yields ≈930,304 B > 816,640 for this case. Wiring `can_cbs_fit_in_l1` into the
DRAM-sharded path with **no accuracy improvement at all** would have rejected the config.

Per DBG-008/DBG-009 the allocator is the messenger, not the owner. The **earliest stage
with all the inputs** is `MultichipDecoder.from_state_dict`
(`tt/multichip_decoder.py:485-505`), which materializes `mlp_dtypes` (`:249-257`),
`mlp_cores`, and `dram_sharded_mlp_blocks` **in the same function** and never crosses
them. A ~15-line host-side computation there reproduces `1,333,760` exactly and could
reject the config at weight-load time, naming the offending JSON field.

### F3 — The only test that walks this geometry pins the dtype to BFP4

`test_linear_attention_bfp4_mlp_geometry_sweep`
(`tests/test_multichip_decoder.py:744-800`) sweeps exactly these two shapes and even
hard-codes the legal block set `("down", 4352, 5120, (17, 1))` at `:756` — but pins
`dtype=ttnn.bfloat4_b` at `:783`, and is gated behind `QWEN36_RUN_BFP4_GEOMETRY_SWEEP=1`.
The one regression test that would have caught a dtype-driven CB overflow on this exact
program is dtype-blind by construction and off by default.

Related: the linear-attention MLP down projection is the **tightest program in the whole
decode graph even under the shipping baseline** — `700,160` of `816,640 B`, i.e. **86%**,
leaving 116,480 B. Nothing records that. `doc/optimized_multichip_decoder/README.md:143-152`
gives the geometry table with latency justification only, no L1-occupancy column and no
note that block 17 is dtype-conditional; `doc/optimized_full_model/README.md:80` presents
“Linear-attention MLP: BFP4 gate/up/down” as a numerical choice rather than a feasibility
constraint; and `tt/precision.py` — the new gatekeeper for exactly this dimension — has no
concept of it. `grep -riE 'l1_size|1572864|circular_buffer|cb_size|max_l1|fits_in_l1'`
across `models/autoports/qwen_qwen3_6_27b` and `models/common` returns **zero hits**.

---

## 6. Ranked experiments and fixes

Target: the linear `mlp_down` region end must drop below **928,000**, or the floor must
rise to meet it.

### Rank 1 — keep `mlp_down.linear_attention: bfp4` in the candidate — config-only, zero code

```json
"mlp_down": {"linear_attention": "bfp4", "full_attention": "bfp8"}
```

The layer-kind-keyed schema already supports this (`tt/precision.py:79-86`,
`tt/multichip_decoder.py:249-257`). Region end → **811,520**, slack **+116,480**; every
other canonical matmul then fits (worst remaining: full gate/up at 776,704, +151,296).
**This is the only change that makes the whole canonical candidate run with no
implementation edit.** Accuracy risk is low — `full_down_bfp4_lofi` already measured
0.95–0.97 top-1 with BFP4 down on both kinds. The cost is honesty about the sweep point:
it is no longer “BFP8 everywhere”.

### Rank 2 — `QWEN36_MC_MLP_DOWN_BLOCK=1` — zero-edit discriminating probe, heavy perf cost

Region end → **223,744**, slack **+704,256**. The env var is still read at
`tt/multichip_decoder.py:495-501`, so this needs no code change and **decisively confirms
the diagnosis**.

Caveats: block 1 means 136 K-blocks instead of 8 — 136 in0 mcast rounds and 136 weight-block
DRAM reads per down projection, with the pipeline drained each block; block 1 measured
114.866 µs vs 60.470 µs at BFP4 (`doc/optimized_multichip_decoder/README.md:149`), ×48
linear layers ≈ +2.6 ms/token. The variable is also **not layer-kind scoped**, so it forces
`w=1` on the full-attention down projection too (legal: `9 % 1 == 0`, 144 blocks).
Correctness is unaffected — accumulation order only. Use as a probe, not a shipping config.

### Rank 3 — pad the linear-attention intermediate 4352 → 4608 and use 16 MLP cores — code change, best structural answer

This is prior-art H3 finally applied to the other layer kind. Resulting region ends:
linear gate/up **776,704** (+151,296), linear down **778,752** (+149,248). All canonical
matmuls fit. Required edits:

* `tt/multichip_decoder.py:206-211` — apply `FULL_LOCAL_INTERMEDIATE_SIZE` to
  `linear_attention` too; the pad block at `:263-285` generalizes as written.
* `tt/multichip_decoder.py:486` — `mlp_cores = 16` unconditionally.
* `tt/multichip_decoder.py:495-497` — the `down: 17` default becomes **illegal**
  (144/16 = 9 blocks/core, `9 % 17 ≠ 0` → `ValueError` at `tt/optimized_decoder.py:906`);
  must become 9 or 0 (auto).
* **Hard blocker:** the persistent CCL slot. `_all_reduce_partial` raises if the linear MLP
  output is not on 8 cores (`tt/multichip_decoder.py:675`) and the pool allocates
  `linear_mlp_8` at 8 cores (`:618-648`). It must become a 16-core slot — which as a bonus
  **halves** that buffer, 163,840 → 81,920 B, raising the L1 floor by another 81,920.
* Sharded RMSNorm grid becomes `[8, 2]` (`tt/optimized_decoder.py:1019-1030`) — already
  exercised by full-attention.
* `tests/test_multichip_decoder.py:211` (`assert linear.intermediate_size == 4352`) and
  `:755-756` must be updated.

Cost: 5.9% padded MLP FLOPs and weight bytes on 3/4 of the layers; the padded rows are
algebraic zeros so numerics are exact. Already validated once at BFP4:
`doc/optimized_multichip_decoder/README.md:150` records the padded 16-core alternative as
*“real-layer PCC passed”*, rejected only on perf (45.7/45.8/46.3 µs vs ~43 µs).

### Rank 4 — make `in0_block_w` dtype-aware in `dram_program()`, and wire `can_cbs_fit_in_l1` into the DRAM-sharded path

The general fixes behind F1 and F2. Subsumes Ranks 1–2 and turns the next occurrence into
a named constructor-time error rather than program 606.

### Rank 5 — split the down projection into two N-halves (2 × N=2560, each a multiple of 32·8)

`per_core_N_in1_sender` 20 → 10 ⇒ in1_CB 554,880, out_CB 20,480, region end **756,352**,
slack **+171,648**. Same total math and weight traffic; costs an extra program launch and
a second pass over in0. A middle ground if Rank 1 is unacceptable and Rank 3 too invasive.

### Ruled out

| option | result |
|---|---|
| Shrink the persistent CCL pool (`linear_attention_mlp: bfp8`, or all four slots bfp8) | **Insufficient** — §3.2: the irreducible live set is 444,416 B (367,616 with a bfp8 pool) against a 239,104 B budget; even the pool alone exceeds it. |
| `packer_l1_acc = False` | No change: `interm0` is already `Float16_b` = output, so out/interm still share one buffer (factory:182-186, 554, 577). |
| `fp32_dest_acc_en = True` | **Worse**: interm0 → Float32 ≠ output ⇒ separate CBs, and `max_subblock_w` drops to 4 → region end 1,413,632. |
| Output dtype bfp8 instead of bf16 | **Worse by 22,848**: interm0 ≠ output ⇒ separate CBs. |
| Raise `dram_sharded_mlp_cores["down"]` to 16 | No effect (only resizes the tensor-backed CB c_6, which is outside the static region) — and illegal anyway, `136 % 16 ≠ 0` (`tt/optimized_decoder.py:894`). |
| in1 double instead of triple buffering | Region end 963,840 — still short of 928,000 alone. Requires editing factory:210; **not exposed through the ttnn API**, and it would reduce reader/compute overlap for every DRAM-sharded matmul in the repo. |
| `QWEN36_OPT_DRAM_SHARDED_MLP=0` | **Ineffective** — `tt/multichip_decoder.py:485` sets `dram_sharded_mlp = True` unconditionally, overriding the env read at `tt/optimized_decoder.py:161`. |

---

## 7. Other potential issues found (real, but **not** causes of this crash)

Each was checked against the failure and cannot produce it. Reported because they are
defects or evidence-integrity gaps in the same change set.

### 7.1 Already remediated during this investigation
When first inspected, `doc/datatype_sweep/build_artifacts.py` built its ledger by globbing
`evidence/candidates/*/teacher_forcing_metrics.json` with `run_status` hardcoded to
`"completed"`, so a crashed candidate vanished from `sweep_results.*` and both Pareto
plots. The current tree adds a `failure.json` branch (`build_artifacts.py:158-195`,
`run_status: "failed"`, `selection_status: "rejected_runtime_failure"`) and the canonical
candidate now has one. No action needed; noted so the timeline is reproducible.

### 7.2 `hifi4` is silently downgraded to `HiFi2` for projections
The refactor deleted the multichip guard
`if decoder.projection_fidelity not in ("auto","lofi","hifi2"): raise`. `PrecisionPolicy`
accepts `"hifi4"` (`tt/precision.py:20, 63-64`), but the consumer is a two-way branch:

```python
math_fidelity=(ttnn.MathFidelity.LoFi if self.projection_fidelity == "lofi"
               else ttnn.MathFidelity.HiFi2)      # tt/optimized_decoder.py:505-511
```

The MLP (`tt/multichip_decoder.py:469-477`) and LM head (`tt/model.py:175-180`) map
`hifi4` correctly. A future `hifi4` projection candidate would record `"projection":
"hifi4"` in `precision_summary` while the hardware ran HiFi2 — **silently falsified sweep
evidence**.

### 7.3 `QWEN36_OPT_KV_CACHE_DTYPE` is dead on the single-chip path, and the policy is re-read from disk per call
`allocate_paged_kv_cache` became
`policy = getattr(self, "precision_policy", None) or load_precision_policy()`
(`tt/optimized_decoder.py:574-581`). `precision_policy` is set only by `MultichipDecoder`
(`tt/multichip_decoder.py:550`), so on the single-chip `OptimizedDecoder` path this
(a) ignores `QWEN36_OPT_KV_CACHE_DTYPE`, which the recorded commands under
`doc/optimized_decoder/candidates/mlp_mixed/*/correctness.log` still use, (b) leaves that
path incoherent — weights from env, KV cache from JSON — and (c) re-parses the JSON from
disk on **every** call.

### 7.4 `precision_config_path` is unreachable from the readiness harness
`Generator` and `QwenFullModel` now accept `precision_config_path`
(`tt/generator.py:42,52`; `tt/model.py:267,272`) and `build_generator` forwards it
(`tt/generator.py:770-774`), but neither `run_prefill_check._main` nor
`run_teacher_forcing._main` exposes a CLI flag or populates `build_kwargs`
(`run_teacher_forcing.py:333-336`). The only working selector is the
`QWEN36_PRECISION_CONFIG` env var (`tt/precision.py:17, 159`), which appears nowhere in
the repo except its own definition and `build_artifacts.py`'s command string, and
`doc/datatype_sweep/` has no README or runner script.
`models/common/readiness_check/run_autoregressive.py:161` was not updated, so
autoregressive evidence carries no `precision_summary`.

### 7.5 The code hard-depends on an untracked directory
`DEFAULT_PRECISION_CONFIG` points at `doc/datatype_sweep/selected_precision_config.json`
and `load_precision_policy` raises `FileNotFoundError` if it is missing
(`tt/precision.py:16, 158-163`). `git status` lists `doc/datatype_sweep/` as untracked
(`??`) and not gitignored. Committing the refactor without that directory breaks **every**
`QwenFullModel` / `MultichipDecoder.from_state_dict` / `allocate_paged_kv_cache`.

### 7.6 `precision.py` validation gaps
* `ccl` and `kv_cache` values are consumed **raw**, not through `_dtype()`
  (`tt/multichip_decoder.py:515-518`, `tt/optimized_decoder.py:578-580`), while `_validate`
  only lowercases for the *check*. A config with `"BFP8"` passes validation, then raises
  `KeyError` at KV allocation.
* `layer_exceptions` keys are never validated, so `"mlp_donw"` or `"mlp_down_fidelty"` are
  silently ignored; and `weight_dtype`'s override path calls `_dtype()` before per-kind
  handling (`tt/precision.py:121-123`), so an exception cannot be a
  `{linear_attention, full_attention}` dict. No config on disk exercises this path.
* `kv_cache.page_block_size`, `kv_cache.layout`, and the `logits_sampling.*` extras are
  validated nowhere and consumed nowhere (`PAGE_BLOCK_SIZE` is hardcoded at
  `tt/model.py:46`), despite the docstring at `tt/precision.py:87-89` claiming the
  validator rejects ignored fields.
* The layer range is hardcoded `0 <= layer_idx < 64` (`tt/precision.py:112`) rather than
  read from `config.num_hidden_layers` (`tt/model.py:286`). 64 is correct for this
  checkpoint, so this is latent only.
* There is no unit test for `precision.py`.

### 7.7 Orphaned environment variables
`QWEN36_MC_MLP_{GATE,UP,DOWN}_{DTYPE,FIDELITY}`, `QWEN36_MC_{INPUT,OUTPUT}_PROJ_DTYPE`,
`QWEN36_MC_PROJECTION_FIDELITY`, `QWEN36_MC_CCL_DTYPE`, and the `QWEN36_MC_*_CCL_DTYPE`
family are removed from code but still appear in recorded commands and in
`doc/optimized_multichip_decoder/autofix/fused_persistent/AUTOFIX.md:147` and
`doc/optimized_multichip_decoder/work_log.md:69`. Setting them is now silently ignored, so
several recorded sweeps are no longer reproducible from their own command lines.

### 7.8 Baseline sweep row is not self-describing
`doc/datatype_sweep/evidence/baseline/{prefill,teacher_forcing}_metrics.json` are the only
artifacts without a `precision_summary` key — they predate the refactor. The baseline
config was verified field-by-field to reproduce the old env defaults exactly (including
`projection_fidelity "auto" ≡ "lofi"`, which is bit-identical here because
`has_program_config == true` forces `LoFi` and `default_l1_acc = true` at
`matmul_device_operation.cpp:2205-2206, 2251-2257`), so the baseline row is *correct* — but
it is not provable from its own artifact and it spans a code change. Separately,
candidates have teacher-forcing evidence only, no prefill.

### 7.9 Sweep resolution is below the effect size
On a 100-token reference, 1 token = 1% top-1. `kv_bf16_control` (0.96) and
`projection_hifi2` (0.96) both score *below* the 0.97 baseline despite being strictly more
precise numerically. The sweep as configured cannot resolve the differences it measures.

### 7.10 Recorded geometry contradicts the code for one row
`doc/optimized_multichip_decoder/README.md:155` records the linear packed input projection
as “selected block 20”, but `linear_input_dram_block = 0` (`tt/multichip_decoder.py:568`)
routes to the auto list `(17,10,8,6,5,4,3,2,1)`, which for `blocks = 20` yields **10**.
Block 20 would give `111,360 + 81,920 + 1,109,760 + 36,864 = 1,339,904` — it would already
clash under the baseline, which it demonstrably does not. The doc is wrong; the code is
right.

### 7.11 Latent L1 hazard: the CCL pool grows per distinct dtype and is never freed
`persistent_ccl_slot_dtypes["attention_16"]` is
`tuple(dict.fromkeys((linear_attention_attention, full_attention_attention)))`
(`tt/multichip_decoder.py:528-533`), so a candidate that sets those two CCL policies to
*different* dtypes allocates **two** buffers in that slot — an extra 43,520–81,920 B of
permanently-held L1. Separately, `_PERSISTENT_CCL_POOLS` (`tt/multichip_decoder.py:164-167`)
is a class dict keyed by `id(mesh_device)` with **no removal path anywhere in the repo**,
and the population loop skips only dtypes already present — so a process that builds
models under two different CCL policies accumulates both buffers. Neither case is
exercised by the six configs on disk; both would eat directly into the §3.2 budget.

### 7.12 Watch item: DRAM footprint under the canonical policy
Not the observed failure (this is an L1 problem), but worth checking once L1 is fixed. MLP
weights are stored **twice** — interleaved plus DRAM-sharded
(`tt/multichip_decoder.py:110-142`) — so `bfp4 → bfp8` adds ≈1.84 GiB **per copy**
(≈3.7 GiB per device), and `kv_cache bf16` adds ≈1.88 GiB at the advertised 262,144-token
pool. Against the recorded 19.535 GiB/device plan
(`models/autoports/qwen_qwen3_6_27b/README.md`) that is ≈25 GiB of 32 GiB — tight but not
obviously fatal. Estimated, not measured.

---

## 8. Unproven / would need hardware

* **The last 17,728 B of the 644,864 B below `928,000`.** §3.1 accounts for 627,136 B
  (97.3%) from source. The residual is attributed to allocator fragmentation — top-down
  first-fit over holes left by `_linear_token`'s temporaries
  (`tt/optimized_decoder.py:1383-1465`, `bank_manager.cpp:492-500`) — because
  `lowest_occupied_address` reports the floor of the occupied *span*, not the sum of live
  bytes. It is not a missing tensor: 17,728 is a multiple of neither 1,024 (bf16 tile
  rows) nor 1,088 (bfp8 tiles). This does not affect any conclusion; §3.2's bound uses
  only source-proven irreducible allocations.
* **`program 606`.** Consistent with the first `linear_attention` decode MLP-down program
  created by the untraced warmup decode in `Generator._capture_split_traces`
  (`tt/generator.py:480-489`, reached from `decode_forward` at `:579-580`) after prefill
  has populated the program cache. Program ids are process-global and were not
  independently reconstructed.
* **Whether prefill would also clash.** `use_sharded` requires `m <= 32`
  (`tt/optimized_decoder.py:876-879`) and `_finish_layer_chunked` slices by
  `LINEAR_CHUNK_SIZE = 64` (`tt/functional_decoder.py:39, 272-287`), so the 161-token AIME
  prompt never yields a ≤32-row chunk. A prompt of ≤32 tokens would take the same path in
  prefill and should hit the identical throw — the
  `geometry/16_linear_attention.log` failure was in fact raised from `prefill_forward`.
  Untested here.
* **`l1_unreserved_base` is not immutable.** It is
  `align(L1_BASE + L1_SIZE − worker_l1_size, 64)`
  (`tt_metal/impl/allocator/l1_banking_allocator.cpp:228-229`), which round-trips to
  111,360 only for the default `worker_l1_size`. A non-default
  `ttnn.open_device(worker_l1_size=…)` would shift every number in §2 by the same
  constant. The readiness harness does not pass one.
* **Blackhole DRAM-reader logical coordinates assume no Tensix column harvesting in
  x ∈ 2..7** (`tt_metal/common/core_assignment.cpp:146-157`). The reported `7-9` is itself
  the evidence that this holds on the machine that produced the failure.
