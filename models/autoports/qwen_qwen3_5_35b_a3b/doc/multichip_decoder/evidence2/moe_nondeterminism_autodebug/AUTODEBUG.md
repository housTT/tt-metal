# AutoDebug: `moe_fused_swiglu` non-repeatable output on a 1x4 Blackhole mesh

Inspection only. No device job, no build, no source edit. Prepared Oct 11, about 02:00 ET.

Acronyms:

- ULP: unit in the last place.
- BF16: bfloat16. BFP8/BFP4: Tenstorrent block floating point, 8-bit and 4-bit mantissa.
- CB: circular buffer. L1: core-local SRAM. DEST: Tensix destination register file.
- NoC: network on chip. DRAM: device memory.
- TRISC: Tensix compute RISC-V (T0 unpack, T1 math, T2 pack). NCRISC/BRISC: data-movement RISC-Vs.
- FPU: Tensix matrix/elementwise unit. SFPU: Tensix vector unit.
- RMWCIB: Tensix byte-masked config read-modify-write instruction (atomic per byte).
- CCL: collective communication library. ET: Eastern Time.

All file references below are relative to `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/` unless they are absolute.
Short names:

- `K/` = `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/`
- `R` = `K/kernels/moe_fused_swiglu_reader.cpp`, `W` = `K/kernels/moe_fused_swiglu_writer.cpp`
- `C` = `K/kernels/moe_fused_swiglu_compute.cpp`, `CH` = `K/kernels/moe_fused_swiglu_compute_helpers.hpp`
- `D` = `K/kernels/moe_fused_swiglu_dataflow.hpp`, `G` = `K/moe_fused_swiglu_geometry.cpp`
- `RUN` = `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage04/micro/nondeterminism/`

## 1. Verdict

- **No source-level defect in `moe_fused_swiglu` explains the symptom.**
  - The reduction order is fixed by grid position. It does not depend on arrival order.
  - Every handshake named in the request is correct for this configuration.
  - If any of those handshakes failed, the op would read another expert's, block's or round's data.
    That gives a gross error, not a 1-2 ULP error.
  - No compute-thread config race can change rounding at run time in this configuration.
- **The probe evidence points outside the op's source logic.**
  - The alternate output is discrete and reproducible. The same call gives the same alternate output in two
    processes with different weight layouts (section 2.3).
  - This excludes stochastic rounding, random stray writes, and weight corruption as the mechanism.
  - Each op runs on idle chips: the probe does a blocking read before every call (section 2.4).
  - `post_combine_reduce` also drifts in the same runs, with identical captured inputs.
  - Earlier, `post_combine_reduce` and `fast_reduce_nc` drifted on a single chip (Stage 3).
- **The alternate output most likely starts upstream of the down matmul (interpretation).**
  - One core's down matmul owns only 1/16 of the output columns (128 columns, about 0.26M-0.38M elements per
    call here).
  - The alternate output changes 2.4M-3.9M elements. So at least one core computes a slightly different
    X tile, gate/up partial, reduced slice or H slice, or at least seven cores' down matmuls differ.
  - Every core's down matmul reads the full H, so one perturbed H slice reaches every output column.
- **No minimal source fix is supported by the evidence.**
  - There is no order dependence in the op to remove.
  - Section 6 lists the device experiments that can localize the cause. The owning agent must approve them.

## 2. Observations versus interpretations

### 2.1 Direct observations (logs and JSON, read as data)

| # | Observation | Source |
|---|---|---|
| O1 | 4-chip probe: `moe_output` differs between repeats on one device at a time, while the captured `moe_input` and the route-map outputs (counts, offsets, dispatch map, target) are identical. | `RUN/x4_full_b32_moe.json`, `RUN/x4_full_b32_lists.json` |
| O2 | A drifting call changes 2.41M-3.92M elements. Max abs is 0.0034-0.0088. | same |
| O3 | For one (call, device), all runs that differ from run 0 show the same element count and max abs. Example: lists process, group 16, device 1: 2915924 elements, 0.0039, in runs 1, 2 and 3. | `RUN/x4_full_b32_lists.json` |
| O4 | Group 20, device 1 gives 2652603 elements, max 0.00341796875, in the stacked process (run 2) and in the lists process (run 1). | both JSON files |
| O5 | Odd-run counts, first 21 groups (84 device-calls per run): stacked process run0 5, run1 2, run2 2; lists process run0 2, run1 1, run2 1, run3 4. That is about 1-6% of device-calls per run. | section 2.3 table |
| O6 | `post_combine_output` differs while `post_combine_input_rows` and `post_combine_input_weights` are identical in the same call (example: stacked process, run 1, group 0, device 3: 64 elements). This happens in many groups. | `RUN/x4_full_b32_moe.json` |
| O7 | Single chip, same probe, 3 repeats x 32 calls: no differing capture. | `RUN/x1_multichip_full_b32.json`, `RUN/x1_optimized_full_b32.json` |
| O8 | Standalone single chip (physical device 3), 64 or 256 experts, 128 tokens, 4x4 grid, three weight modes, 20 repeats: bitwise repeatable. Routing is uniform random. Only valid rows are compared. | `RUN/moe_repeat.py`, `RUN/moe_repeat.json` |
| O9 | Stage 3 single chip: one 32-row block of the final output differs (rows 608-639, 2948 elements, max 0.0078). | `models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/review2/bfp8_grid/full_1024/small4x4_large11x8.json` |
| O10 | Stage 3: `fast_reduce_nc` drifts in model context with an identical captured input; standalone is repeatable. | `models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/review1/post_combine_autodebug/AUTODEBUG.md`, O8-O9 there |
| O11 | The probe keeps the previous run's output `y` alive while the next run allocates. Device buffer addresses can therefore differ between run 0 and later runs. | `RUN/probe.py:120-127` |
| O12 | Lists process: `post_combine_output` group 9, device 1 (6725 elements) differs in runs 1 and 3, not in run 2. But `moe_output` groups 10, 11, 14 and 19 differ only in run 3. | `RUN/x4_full_b32_lists.json` |

### 2.2 Interpretations (testable)

- I1: O3 and O4 show a small set of fixed outcomes per call. Each outcome is a deterministic function of the
  inputs. It is not random noise.
- I2: O4 crosses two processes and two weight layouts (`stacked` BFP4/BFP8 tensors versus newly created
  per-expert BFP8 lists). Weight corruption or a random late write would not reproduce the same output.
- I3: O6 and O10 show that ops other than `moe_fused_swiglu` lose repeatability in the same context, with
  identical captured inputs. A shared platform or state cause is more likely than three separate op bugs.
  This is not proven.
- I4: O8 is not a matched control. Its routing is uniform, so every expert has `m_t = 1`. The model's routing
  is concentrated: up to about 587 local routes on one device (O1 detail, 1.2M differing elements over
  2048 columns), so some experts have `m_t` 2-4. Its inputs, preceding ops and mesh state also differ.
  - At the observed 1-6% rate, 60 clean calls with the matching shape (64 experts, 4x4, three modes) have a
    probability of about 0.09-0.5. O8 is weak evidence.
- I5: O11 and O12 suggest that buffer placement (allocation history) changes the odds of the alternate
  outcome. A strict placement rule is contradicted: under it, run 3 would match run 1. Some events occur in one
  run only. Placement is at most one factor; a timing component remains.

### 2.3 Outcome classification (from the saved JSON, first 21 groups)

The probe saves only the first 40 differing captures per comparison. The table uses groups 0-20, which all
comparisons cover. "run0 odd" means every later run differs from run 0 in the same way.

| Process | Group, device | Runs that differ from run 0 | Elements, max abs | Class |
|---|---|---|---|---|
| stacked | 1, 0 | 1, 2 | 3426145, 0.0078 | run 0 odd |
| stacked | 12, 0 / 12, 1 | 1, 2 | 3498527 / 3195379 | run 0 odd |
| stacked | 15, 2 / 15, 3 | 1, 2 | 3817994 / 2754971 | run 0 odd |
| stacked | 0, 1 | 2 | 3920267, 0.0039 | run 2 odd |
| stacked | 5, 1 / 6, 1 | 1 | 3366256 / 2412330 | run 1 odd |
| stacked | 20, 1 | 2 | 2652603, 0.0034 | run 2 odd |
| lists | 16, 1 / 20, 2 | 1, 2, 3 | 2915924 / 3262593 | run 0 odd |
| lists | 13, 3 | 2 | 2780581, 0.0068 | run 2 odd |
| lists | 20, 1 | 1 | 2652603, 0.0034 | run 1 odd |
| lists | 10, 1 / 11, 2 / 14, 3 / 19, 0 | 3 | 3293106 / 3389780 / 3572760 / 3562892 | run 3 odd |

### 2.4 Synchronization boundaries in the probe

- `RUN/probe.py` wraps the op. It calls `snap("moe_input", args[0])` before the op.
- `snap` calls `ttnn.to_torch` on every device shard. That is a blocking command-queue read.
- A command-queue read waits for all earlier programs on that device. So when `moe_fused_swiglu` starts,
  every earlier op on every chip has completed. The same holds for `post_combine_reduce` (two snaps first).
- The kernel that runs just before the op on its cores is the probe's `zeros_like` fill (the probe replaces
  `ttnn.empty`) or `ttnn.embedding`. Both are the same in every run.
- Consequence: an overlap with a CCL op or a late fabric write must survive a millisecond-scale host
  round trip. I rank that as unlikely (section 5.6).

## 3. Lowered path and instantiated configuration

### 3.1 Call site

- `models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py:1898-1938` (`_LocalDispatch.__call__`).
- `_StackedDispatch` inherits it (`models/autoports/qwen_qwen3_5_35b_a3b/tt/multichip_decoder.py`).
- Per call: 128 tokens, so `input_m_tiles = 4`, grid 4x4, 64 local experts, intermediate 512.
- `capacity = ((1024 + 64*31 + 32 + 31) // 32) * 32 = 3040` rows (`multichip_decoder.py`, `prepare_maps`).

### 3.2 Host-derived constants (verified in source)

| Constant | Value | Source |
|---|---|---|
| HGROUPS, KGROUPS | 4, 4 | grid |
| M_BLOCK, M_T_MAX | 8, 4 | `K/moe_fused_swiglu_geometry.hpp:15`, `input_m_tiles` |
| M_EFF_MIN | 1 | `G:97` (`OUT_SUBBLOCK_H_GU = 1`) |
| HN_PAD, KR_PAD, EC_MAX | 4, 16, 4 | 16 hidden tiles / 4, 64 emb tiles / 4, 64 tiles / 16 cores |
| WD_MROW_ROUNDS | false | `G:136`: needs `kgroups == M_BLOCK` |
| DEPTH_X | 1 | `G:186-187`: one M-block maximum |
| CAN_PREFETCH_X | false | `R:879`: needs `HGROUPS >= M_BLOCK` |
| DEPTH_H, HACK_AHEAD | 3, 2 | `G:190-191` |
| WD_SPLIT | 3 (writer reads 1 of 4 rows per K-block) | `G:239` |
| PHASE_CB_ALIAS | false (BF16 output tile 2048 B) | `G:379-381` |
| SCATTER_ONE_SIGNAL, H_MCAST_POSTED | true, false | `K/moe_fused_swiglu_geometry.hpp:35,42` |
| Compute config | LoFi, 16-bit DEST, half sync, `bfp8_pack_precise = true` | `K/moe_fused_swiglu_program_factory.cpp:615-627` |

- The L1 fallback ladder (`G:140-237`) can lower DEPTH_H, HACK_AHEAD and WD_SPLIT. It is assumed not to fire:
  the subagent estimate of CB need is about 877 KiB with BFP8 weights. The exact budget is unverified.
- Every runtime `m_eff` (1, 2, 4) takes the hidden-slice phase-2 rounds, not the M-row rounds.
- All four grid rows are reduce-scatter workers, with 1, 2 or 4 tiles each.

## 4. Candidate ledger

"Explains" means the candidate predicts O1-O5: discrete outcomes, ULP-only magnitude, millions of elements,
and mesh-only occurrence.

### 4.1 Reduce-scatter accumulation order depends on arrival: refuted

- A contributor writes its slice at `dst + my_row * slice_tiles * tile_bytes` (`D:169`).
- The landing CB base is the same on every core. Each block pushes the whole capacity, so the write pointer
  returns to the base (`R:957-959`, `R:1028-1029`).
- Compute folds slots in a fixed order: `copy`/`add` of slots 0-1, then pairs (`C:165-225`).
- The last gate slot is added in `add_silu_elementwise` at a fixed offset (`C:716-728`, `CH:182-211`).
- The up fold is `fold_chain(KGROUPS, ...)`, also fixed (`C:730`).
- Result: the sum order is a function of grid row. Arrival order cannot change it.

### 4.2 Early read or early overwrite of a landing slot: refuted

- Each contributor's data is acked before its signal:
  - Up payload: acked write barrier (`D:183`), fence, then the mailbox word `MBOX_UP_SCATTER_DONE` (`R:999-1006`).
  - Gate payload: acked barrier, then the writer polls the mailbox word with `invalidate_l1_cache` each
    iteration, then sends one `SEM_DATA` per destination (`W:488-490`).
- The destination waits for `SEM_DATA >= 4(gb+1)` before it pushes (`R:1020-1030`).
- An invite (`SEM_GO`) is sent only after the destination reserves the whole landing CB (`R:957-983`).
  That reserve needs compute to pop the previous block (`C:716-762`).
- `MBOX_UP_SCATTER_DONE` is zeroed at reader start (`R:378-379`). The writer reads it only after the mailbox
  magic, which the reader publishes after a fence (`K/kernels/moe_fused_swiglu_common.hpp:83-92`).
- Any failure here would read the previous block's or expert's partials: a gross error.

### 4.3 Semaphore reuse across experts or calls: refuted

- Program semaphores are re-initialized to 0 at each launch (`K/moe_fused_swiglu_program_factory.cpp:462-469`).
- `SEM_GO`, `SEM_DATA`, `SEM_HSLICE`, `SEM_H_FREE`, `SEM_WDSPLIT` are monotone counters over the global block
  index `gb`. Reader and writer advance them on the same grid-uniform predicates (`R:644-654`).
- Peer counts cannot run ahead: a peer invites block k+1 only after its block-k data wait, which needs this
  core's block-k signal.

### 4.4 H multicast slot and flag reuse: refuted

- Round g uses flag cell `g % 3` (`R:1232`, `R:1242`).
- A receiver acks at most `HACK_AHEAD = 2` rounds ahead (`R:1150-1194`). Cell reuse is therefore always on a
  cell the receiver already consumed and reset (`R:1244-1245`).
- The sender self-copies and multicasts with source excluded. Payload and flag are linked and non-posted
  (`R:232-286`).
- `h_cursor` / `h_pad` realignment is identical in reader and compute (`R:650-654`, `R:1317-1321`,
  `C:540-544`, `C:932-936`).
- Any failure would deliver another hidden slice or round: a gross error.

### 4.5 W_down split and X multicast: refuted

- A W_down block is pushed only after the reader's barrier and `wd_split_gate`, which waits for the writer's
  per-block publish (`R:326-333`, `R:1043-1048`, `R:1211-1221`, `R:1247-1258`).
- X: compute tilizes only row `my_col`; peers write other rows (`C:584-601`).
  Receivers ack only after they reserve the slot (`R:709`, `R:833`).
- A failure would use another expert's weights or tokens: gross.

### 4.6 CB or mailbox state assumed initialized: refuted

- The mailbox magic carries the expert index (`R:585-590`). Consumers wait on a CB front first
  (`C:507-520`, `W:297-300`).
- Pad rows past `m_rows` stay stale by design (`K/kernels/moe_fused_swiglu_common.hpp:120-146`).
  Each output row depends only on its own input row. BFP8 shares an exponent per 16 values of one row.
  Stale pad rows cannot reach real rows. The writer does not write tile rows at or beyond `m_t`.
- DEST and srcA/srcB are zeroed at every kernel start: `run_kernel()` calls `zeroacc()` on MATH and
  `zerosrc()` on UNPACK (`tt_metal/hw/ckernels/blackhole/metal/common/chlkc_list.h:29-36`).
  This corrects item 7.4 of the earlier `post_combine_reduce` report.
- Firmware does not reset Tensix semaphores, config or SFPU per launch, and the three TRISCs start without a
  barrier (`tt_metal/hw/firmware/src/tt-1xx/trisc.cc:133-223`, `brisc.cc:282-292`). Section 4.8 covers the
  config fields this kernel consumes.

### 4.7 Packer L1 accumulate toggling: refuted

- Down matmul, 4 K-blocks: block 0 packs with L1 accumulate off, blocks 1-2 on (`CH:152`, `CH:159`).
- `configure_pack` clears `Pack_L1_Acc` at kernel start (`tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h:439`).
- The kernel clears it after gate/up and after down (`C:696`, `C:907`).
- The final K-block reloads the BF16 scratch after `PACK_DONE` (`CH:106-110`, `CH:167-171`).
  - `PACK_DONE` is seeded at boot with value 0, max 1 (`tt_metal/hw/inc/internal/tt-1xx/blackhole/c_tensix_core.h:421`,
    called from `tt_metal/hw/firmware/src/tt-1xx/brisc.cc:262`).
  - The only other users post and get in balance (subagent search; streaming SDPA re-seeds it).
- A missed toggle or an early reload loses one K-block's contribution (one quarter of the contraction): gross.

### 4.8 Compute-thread config overlap or runtime rounding change: refuted

- With `bfp8_pack_precise = true` and 16-bit DEST, every pack source format is Float16_b
  (`tt_metal/jit_build/data_format.cpp:330-331`). Dstacc never changes.
- Shared config words are written with RMWCIB on disjoint byte masks. Their values never change during the
  kernel (subagent ledger: words 1, 2, 18, 70, 71, 220).
- Stochastic rounding bits are cleared once per launch (`tt_metal/tt-llk/tt_llk_blackhole/common/inc/cunpack_common.h:839-847`).
  No production caller enables them (`_llk_unpack_configure_stoch_rnd_` has no caller).
- SiLU rounds to nearest BF16 explicitly
  (`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h:23-25`).
- Only the PACK thread runs SFPU init here (`C:450`, `tt_metal/hw/inc/api/compute/compute_kernel_api.h:709`).
  No other thread writes the SFPU constant it uses.
- The only runtime-toggled field read by the FPU is `Zero_Flag_disabled_src`
  (`tt_metal/tt-llk/tt_llk_blackhole/common/inc/cmath_common.h:62-66,99-107`). Its effect is limited to
  denormals. It cannot give 0.004-0.009 differences.
- One real cross-thread timing exposure exists, and it is numerically inert here:
  - The PACK-thread SiLU's implied SFPU load/store format follows the SrcB format.
  - UNPACK reprograms SrcB for the next fold while PACK may still run SiLU (`C:716-730`, `CH:182-211`).
  - Every SrcB format in this kernel is B-exponent family (Float16_b, Bfp8_b, Bfp4_b), so the implied format
    does not change. Uncertain, low.

### 4.9 Summary

| Candidate | Code contradiction found | Predicts discrete outcomes | Predicts ULP-only | Verdict |
|---|---|---|---|---|
| Arrival-order reduction | no (order fixed) | no | yes | refuted |
| Handshake/semaphore race | no | yes | no (gross) | refuted |
| H slot/flag reuse | no | yes | no (gross) | refuted |
| Packer L1 acc / `PACK_DONE` | no (balanced) | yes | no (gross) | refuted |
| Config race, rounding mode | no | yes | yes | refuted by source |
| Shared platform/state cause outside the op | not identified | yes | yes | open |

## 5. Other potential issues (real, but they do not explain the symptom)

### 5.1 Pad X rows clamp to `M_T_MAX - 1`, not to the expert region

- `R:724-727` and `R:928-931`: a pad tile row is clamped to `M_T_MAX - 1`, then rebased by `start_row`.
- When `m_eff > m_t` for the last region in the buffer, the read can pass `ROW_CAPACITY` by up to 32 sticks.
- `region_ok` (`R:561-562`) checks only `m_t`, not `m_eff`.
- The data feeds pad rows only. Real rows are not affected. In this model the last 32 rows are a spare slot.

### 5.2 `PACK_DONE` is not seeded by the kernel

- The reload handshake depends on boot-time SEMINIT and on every other kernel staying balanced.
- Streaming SDPA re-seeds it to max 2 (`ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp:343`).
- A local `t6_semaphore_init` at kernel start would remove the dependency. It does not change this symptom.

### 5.3 DEST remap/swizzle (config word 220) stays set after fast tilize

- Fast tilize sets it and never clears it (`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_common.h:72-81`).
- In one launch, only injector cores (column `t % HGROUPS` for `t < m_eff`) run fast tilize.
  Which cores carry the bit is therefore data dependent.
- The value is deterministic for a fixed input and op sequence. It cannot give run-to-run drift on its own.

### 5.4 Probe gaps

- Counts, offsets and expert ids are captured when `route_maps` returns, not at the op call.
  Weights are never captured. A changed count either keeps the same `m_t` (no output change) or changes the
  written rows (a gross change). O4 argues against weight changes.
- The probe saves only the first 40 differing captures per comparison and no raw tensors. The column and row
  pattern of the alternate output cannot be read from the saved data.

### 5.5 Standalone repro is not a matched control (I4)

- Uniform routing, `m_t = 1` for every expert, random inputs, no preceding model ops, no fabric.
- Its comparison covers valid rows only. The probe compares the whole output.

### 5.6 Late CCL or fabric writes into worker L1, and other mesh differences

- The all-reduce kernels were not traced to completion. The first audit was interrupted.
- The blocking read before each call (section 2.4) and the reproducible alternate (O4) both argue against it.
- Fabric regions in worker L1 are fixed below `MEM_MAP_END`, so the CB base does not depend on fabric
  (subagent: `dev_mem_map.h:131-182`).
- `FabricTensixConfig` defaults to disabled, so no fabric kernel runs on worker cores
  (subagent: `ttnn/cpp/ttnn-nanobind/fabric.cpp:138`).
- Not checked: per-chip harvesting and logical-to-physical core mapping, dispatch core placement. These can
  change NoC paths and timing, not results.

### 5.7 Probe keeps the previous output alive

- `RUN/probe.py:120-127` assigns `y` only after `prefill_forward` returns, so run r allocates with run r-1's
  output live. Run 0 sees a different device layout from later runs (O11, I5).
- This is a probe artifact, not an op defect. It must be removed before any repeat-rate statistic is trusted.

## 6. Discriminating experiments (device jobs; the owning agent must approve)

Ranked by information per run.

- **E0. Fix the probe first.**
  - Deallocate `y` and synchronize before the next run, so every run has the same layout (section 5.7).
  - Save raw tensors for every differing capture, not only the first 40 summaries.
  - Snap counts, offsets and expert ids inside the op wrapper (`args[4]`, `args[5]`, the
    `expert_region_offsets` keyword).
  - If run 0 stops being the odd one, allocation history is one factor (I5).
- **E1. Localize the alternate output (one capture, host analysis).**
  - Save the raw `moe_output` for both outcomes of a recurring call (stacked process, group 20, device 1).
  - Map the diff by output column band (16 bands of 128 columns, one per core) and by expert region.
  - All bands of a token row: upstream of the down matmul (X tile, gate/up, fold, SiLU, H).
  - One band only: that core's down matmul.
  - Some experts only: per-expert state.
- **E2. Replay the captured inputs standalone.**
  - Save `moe_input`, counts, offsets and expert ids at the call for group 20, device 1.
  - Replay them at least 200 times: (a) single chip, no fabric; (b) 1x4 mesh with `FABRIC_1D_RING`, no model,
    no CCL.
  - (a) drifts: data-dependent op behavior. Only (b) drifts: the mesh environment enables it.
    Neither drifts: preceding model ops are needed.
- **E3. Back-to-back calls in model context.**
  - In the 4-chip probe, call the op twice in a row with the same inputs, then once more after a neutral op.
  - If the second call can differ from the first, the cause is inside the call. If not, carried state.
- **E4. Make it frequent.**
  - Use the `perturb` skill (`tt_metal/tt-llk/.claude/skills/perturb/`) to inject NOPs per TRISC and per
    data-movement RISC-V on the E2(b) setup. A rate increase points at the actor.
- **E5. Read hardware state on the moe cores.**
  - With `tt-exalens`, read config words 1, 2, 18, 70, 71, 220, the Tensix semaphores, and SFPU programmable
    constants at kernel entry, for both outcomes.
  - On a drifting call, read the H-slice CB (`CB_H_SLICE`) and gate/up landing CBs of all 16 cores. The core
    whose values differ is the source.
- **E6. Move SiLU off the PACK thread.**
  - Build a test variant that runs the SiLU on MATH (as the `SWIGLU_OAI` path already does for its activation).
  - If the alternate disappears, the PACK-thread SFPU timing (section 4.8, last item) is involved.

## 7. Minimal fix

- None is supported. The op has no order-dependent reduction to make order-independent.
- Do not loosen tolerances or claim a fix until E1-E3 localize the stage.

## 8. Confidence and open gaps

- High: the op's reduction order is fixed, and handshake failures would be gross (sections 4.1-4.7).
- Medium-high: no compute config race changes rounding (section 4.8). It relies partly on subagent ledgers.
- Medium: the cause is outside the op's source logic. It rests on O4, O6 and section 2.4, not on a found defect.
- Not checked: Blackhole hardware behavior not described in source (packer, DEST, NoC). Raw ISA text was not read.
- Subagent claims I did not re-read are marked "subagent" above.
