# AutoDebug: `post_combine_reduce` bitwise nondeterminism in Qwen3.5-35B-A3B prefill (Blackhole)

Inspection only. No device job, no build, no source edit. Prepared Oct 10, about 12:00 ET.

Acronyms: ULP = unit in the last place. CB = circular buffer. LLK = low-level kernel library. BH = Blackhole.
DEST = Tensix destination register file. L1 = core-local SRAM. NoC = network on chip.
TRISC = Tensix RISC-V core (T0 unpack, T1 math, T2 pack). FPU = Tensix floating-point unit.
RM = row-major. RMW = read-modify-write. RMWCIB = masked per-byte config RMW instruction.
CT/RT = compile-time/runtime kernel arguments. ISA = instruction set architecture. ET = Eastern Time.

## 1. Verdict

- **No source defect in `post_combine_reduce` explains the symptom.**
- Every suspect named in the request was traced. None produces it.
- The symptom is: ULP-level drift on all 32 rows of one core, model context only.
- **Newer evidence points outside this op.** `fast_reduce_nc` also drifts in the same model context.
  - Its captured input is bitwise identical across runs.
  - It shares no op-specific kernel code with `post_combine_reduce`.
  - Source: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_claude/micro/flake/TP_*.log`, `NT_*.log`.
  - These logs are from 11:30 to 11:33 ET, after the request was written.
- **The root cause is unresolved.** Section 6 lists the discriminating experiments.
- **No minimal fix for the symptom is supported by evidence.**
  - Section 7 lists real `post_combine_reduce` defects. Each one is numerically inert here.
  - Fixing them will probably not remove the drift.

## 2. Observations versus interpretations

### 2.1 Direct observations (from logs, read as untrusted data)

| # | Observation | Source |
|---|---|---|
| O1 | `post_combine_reduce` path: `d_out` differs in exactly 32 or 64 rows. Each block starts at a multiple of 32. | `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_claude/micro/flake/PD_*.log`, `PS_*.log` |
| O2 | The differing chunk changes between processes: rows 1024, 256, 1728, 288, 640, 1376, 608. It is not always the last chunk. | same |
| O3 | `d_out` maxdiff is 2^-13, 2^-12 or 2^-11. All are exact powers of two. | same |
| O4 | `router_values` and `d_restored` show no diff across runs. | same (`PROBE_DONE` lists them; no `run*` line) |
| O5 | `d_result` differs in 11 rows with NaN in every process. `d_restored` does not differ. | same |
| O6 | The standalone `post_combine_reduce` repro is deterministic. It ran 40 calls on each of 4 devices, with host weights (`pcr_*`) and device weights (`pcrw1_*`, `pcrw2_*`). | `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_claude/micro/flake/pcr*.log` |
| O7 | The repro weights are in DRAM (`BufferType::DRAM` printed). | `pcrw1_d0.log` |
| O8 | Tiles-combine path: `t_experts`, `t_scores` and `t_prod` show no diff. `d_out = fast_reduce_nc(t_prod)` differs in 1128 to 2046 of 2048 rows, in 14 of 16 processes. | `TP_*.log`, `NT_*.log` (`PROBE_DONE` lists `t_prod`) |
| O9 | Standalone `fast_reduce_nc` with the same shape is deterministic over 40 calls on each of 4 devices. | `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_claude/micro/flake/fred_d*.log`, script `fred_repro.py` |
| O10 | `NEUTRAL=1` inserts an unrelated `ttnn.add` before `fast_reduce_nc`. The drift stays. | `NT_*.log`, `prefill_probe.py:73-75` |

The commands for the other log families (`A`, `B`, `NP`, `NX`, `MX`, `SZ`, `T*`, `PL`, `TL`, `DBG*`, `PQ`, `PR`, `TC`) are not recorded.
I did not use them as evidence.

### 2.2 Interpretations (mine, testable)

- I1: O1 + O3 indicate a rounding-level change. They do not indicate a lost or doubled expert term (see 4.3).
- I2: One output chunk equals one core here (see 3.2). O1 is therefore a per-core event.
- I3: O8 + O9 + O10 show that a second reduction kernel loses determinism in the same context. Shared upstream state or a platform effect is more likely than a defect inside `post_combine_reduce`.
- I4: The O8 pattern (most runs about 1174-1216 rows, 57-59%) differs from O1 (one core, every row). One shared mechanism is plausible but not proven.
- The request says "usually the first call differs from later calls". I could not confirm that run1 equals run2. The probe compares only against run0 (`prefill_probe.py`, final loop).

## 3. Lowered path for the failing call

### 3.1 Call site and operands

Call site: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py:1650-1653`.

| Operand | Producer | Layout / memory (from source) | Page bytes |
|---|---|---|---|
| `combine_output` | `ttnn.embedding` result, viewed as `(1,1,T,8,2048)` | bf16 RM, DRAM | 4096 |
| `weights` | `reshape(to_layout(values, RM), (1,1,T,8,1))` | bf16 RM, **probably L1 interleaved** | 2 (aligned 16 in L1, 64 in DRAM) |

- `values` comes from `_OptimizedGateModule.route`. It is a view of RM tensors moved to `L1_MEMORY_CONFIG`.
  - Source: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py:600-609`.
  - Source: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/common/modules/moe/tt_moe_gate.py:327-332`.
  - So `to_layout(values, RM)` does nothing, and the reshape output stays in L1.
  - Not verified at runtime.
- The reshape changes the last dimension, so it is a real copy.
  - It runs `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/device/rm_reshape_interleaved.cpp`.
  - The copy writes 16 B per 2-B page. Bytes 2-15 are stale staging bytes.
  - The kernel ends with `noc.async_write_barrier()` (`rm_reshape_interleaved.cpp:170-172`).

### 3.2 Work split instantiated (T = 2048)

From `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/post_combine_reduce_program_factory.cpp:41-94`:

- `emb_dim=2048`, `num_experts=8`, `emb_dim_cb_tiles=2`, `emb_dim_out_tiles=64`.
- `total_chunks=64`. The grid must have at least 64 cores; the exact BH grid is not verified.
- So `num_cores=64` and `chunks_this_core=1`. Chunk c is tokens 32c to 32c+31.
- CB sizes: c_0 = 4096 B, c_1 = 2048 B, c_16 = c_17 = 131072 B.
- Compute config: HiFi4, `fp32_dest_acc_en=false`, `dst_full_sync_en=false` (`program_factory.cpp:265-269`).

### 3.3 Per-core CB ledger (one chunk)

| CB | Producer actions | Consumer actions | Balanced |
|---|---|---|---|
| c_0 | reader: 256 x (reserve 2, read 4096 B, barrier, push 2) | compute: 256 x (wait 2, pop 2) | yes |
| c_1 | writer: 256 x (reserve 1, read 2048 B, barrier, push 1) | compute: 256 x (wait 1, `read_tile_value`, pop 1) | yes |
| c_17 | compute PACK: reserve 64 once, 512 `pack_tile<true>`, push 64 | tilize UNPACK: wait 64, pop 64 | yes |
| c_16 | tilize PACK: reserve 64, push 64 | writer: wait 64, write 64 tiles, barrier, pop 64 | yes |

Every chunk on every core runs the same instruction sequence. No branch or count differs between the failing core and a passing core.

## 4. Candidate ledger

Each suspect named in the request, plus others found. "Explains" means it can produce O1 to O3 and the model-only, first-call-biased pattern.

### 4.1 Uninitialized bytes in the weight tile (c_1): refuted

- The writer reads 2048 B from a 2-B page (`deepseek_moe_post_combine_reduce_writer.cpp:70,163-166`). Bytes 2-2047 are neighbour pages or stale data.
- Only datum 0 reaches the FPU:
  - `llk_unpack_AB_init<SCALAR>`: srcB gets one face, once per call (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB.h:119-127`).
  - Math uses `SRCB_BCAST_ALL`. The srcB counter never moves (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h:42-63,136-142`).
  - The ISA semantics (srcB[0][0]) come from a DeepWiki summary. Unverified against the raw ISA text.
- `read_tile_value` reads 4 bytes (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/hw/inc/api/compute/cb_api.h:202-217`). It only gates the skip of a `+0.0` weight.
  - A processed zero weight adds plus or minus 0 through L1 accumulate. Finite results are bit-identical.
- Even if garbage were consumed, it would change one token's row. O1 shows all 32 rows of a core.

### 4.2 Writer weight page layout: refuted as cause, real defect (7.1)

- Wrong reads would move the multiplier by much more than a ULP, on single tokens.
- The read is out of bounds but numerically inert (4.1).

### 4.3 Packer L1-accumulate reconfig race: refuted

- `reconfigure_packer_l1_acc` waits with `STALLWAIT(STALL_CFG, PACK)`, then does one RMWCIB on word 71, bit 19 (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h:327-332`).
- Expert-0 packs always see 0. `deepseek_moe_post_combine_reduce_compute.cpp:147` writes 0 at the end of every token, and hw startup also writes 0 (`cpack_common.h:438-439`). A late write of 0 is a no-op.
- The only real transition is expert 0 to expert 1 (0 to 1). If missed, expert 0's term is lost.
- Magnitude: the term w_e * x_e is about 5-30% of the sum. That is far above 1-2 ULP, and per token, not uniform.
- Same-address RAW between experts: two packer drains separate them (the release stall and the reconfig stall). Sibling kernels use the same bare `pack_reconfig_l1_acc`:
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp:420-445`.
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/kernels/moe_fused_swiglu_compute_helpers.hpp:152,159`.
- No cross-thread whole-word write to word 71 exists. `set_packer_config` skips word 71 on purpose (`cpack_common.h:51-61,434-436`).

### 4.4 CB synchronization and pack-to-tilize handoff: refuted

- Ledger 3.3 balances.
- If tilize read c_17 early, only the last packs would be stale: token 31, missing w7*x7. That is one row and a gross error.
- `llk_pop_tiles` stalls until unpack finishes, so c_0 and c_1 cannot be overwritten mid-unpack (agent-reported: `llk_io_unpack.h:50`).

### 4.5 Stale or racing hardware config: no source-level hit

The per-thread ledger covers `compute_kernel_hw_startup`, the bcast init, pack, L1 accumulate and fast tilize. It found:

| Field | Written each launch | Writer / mechanism | Fits O1-O3 |
|---|---|---|---|
| Stochastic rounding bits (word 1, byte 0) | yes, cleared | T0, RMWCIB in mutex (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/common/inc/cunpack_common.h:839-845`) | no: no production code sets them |
| `Fp32_enabled`, INT8, SFPU fp32 (word 1, byte 3) | yes | T1, RMWCIB, no mutex (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_common.h:31-59`) | only if a hardware lost update happens (4.6) |
| `Dstacc` (word 1, byte 3) | yes | T2, RMWCIB in mutex | no |
| `PCK_DEST_RD_CTRL` (word 18) | yes | T2, whole-word `cfg[]` store (`cpack_common.h:462`); single owner | no |
| Exp threshold, `Pack_L1_Acc`, LF8 bits (word 71) | yes | T0/T2 RMWCIB, disjoint masks | no |
| DEST `remap_addrs`/`swizzle_32b` (word 220) | **no** | never reset in production code (`llk_math_common.h:69-82`); set by pack_untilize, fast tilize, `deepseek_compute_kernel_init` | addressing only; see 6 |
| GS_LF, Bfp8_HF (word 1, bits 13-14) | no | no writer exists | constant power-on value |
| DEST contents | not cleared at start | ELWMUL accumulates into DEST (`llk_math_eltwise_binary.h:534-541`) | token 0 only; see 7.4 |

- All three TRISCs start without a barrier (agent-reported: `firmware_common.h:206-214`).
- Each TRISC ends with `tensix_sync()` (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/hw/firmware/src/tt-1xx/trisc.cc:222`), so kernels do not overlap on a core.
- `state_configure` is a no-op unless the sentinel is enabled (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/hw/inc/api/compute/sentinel/compute_kernel_sentinel.h:280-292`).

### 4.6 Hardware lost update on word 1, byte 3: unproven hypothesis

- Story:
  - T1 (`Fp32_enabled=0`) and T2 (`Dstacc`) RMWCIB the same byte at kernel start, with no barrier between them.
  - If the hardware lost T1's update, `Fp32_enabled=1` would survive from an fp32-DEST predecessor while `Read_32b_data=0`.
  - The FPU would then keep fp32 partial results. The packer would read the upper 16 bits, which truncates instead of rounding.
  - That gives 0-1 ULP per product, on one core, on every row. This matches O1 to O3.
- Contradictions:
  - The ISA states that RMWCIB is atomic per byte (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/tt-llk/.claude/skills/cfg-word-overlap-audit/SKILL.md:50-53,109`, citing `RMWCIB.md`; I did not read the ISA text).
  - It does not explain O8, because `fast_reduce_nc` writes the same byte the same way.
  - The 32-bit DEST layout claim comes from code comments only (agent-reported: `llk_math_eltwise_unary_datacopy.h:115,125`).
- Status: attractive but unproven. A readback of word 1 on the failing core would decide it (6, E4).

### 4.7 Input bytes differ from the captured tensors: weakened, not closed

- The kernel consumes:
  - every byte of each 4096-B `restored` page (`deepseek_moe_post_combine_reduce_reader.cpp:38-44`);
  - datum 0 of each weight page.
- The probe captures both logically: `d_restored`, and `router_values` before the reshape copy.
- Data movement cannot create a 1-ULP change. A late or corrupt write would give gross per-token errors.
- Gap: the reshape output itself is not captured.

## 5. Cross-op evidence

| Property | `post_combine_reduce` | `fast_reduce_nc` (tiles path) |
|---|---|---|
| Kernel | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/kernels/deepseek_moe_post_combine_reduce_compute.cpp` | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/kernels/reduce_nc.cpp` |
| Math | ELWMUL SCALAR bcast, HiFi4 | ELWADD, `acc_to_dest=true` |
| Accumulation | packer L1 accumulate (bf16) | 16-bit DEST |
| `read_tile_value`, tilize, L1 acc | yes | no |
| Shared | `compute_kernel_hw_startup`, 16-bit DEST that relies on zero DEST, 1-tile pack, same preceding model ops | same |
| Captured input identical | yes (O4) | yes (O8) |
| Standalone deterministic | yes (O6) | yes (O9) |
| Model context drifts | one core per process (O1) | most rows (O8) |

- Every suspect in 4.1 to 4.4 is code that `fast_reduce_nc` does not have.
- A shared cause must sit in the common LLK startup, in DEST handling, in state left by preceding model ops, or in the platform.

## 6. What to run next (device jobs; the owning agent must approve)

Ranked by information per run. The smallest decisive test comes first.

- **E1, host only, no device.** Load the saved `d_out` run0, run1 and run2 tensors and diff them per element.
  - Count differing elements per row and the signed ULP delta.
  - Check whether run1 equals run2.
  - Check which of the 64 output tiles in the chunk are hit.
  - Every element of the core points to a mode change (4.6). Sparse elements point to a rare per-element event.
- **E2.** Run the standalone repro with L1-interleaved weights, as in the model.
  - Steps: `from_torch(..., memory_config=L1)` for `values`, then reshape.
  - This tests the O7 placement gap. Its kernel path is identical.
- **E3.** Bisect the preceding model ops. Replay the dispatch chain standalone, then call `fast_reduce_nc` and `post_combine_reduce`.
  - The chain: router (`generalized_moe_gate`), `moe_fused_swiglu`, untilize, embeddings.
  - Drop one op at a time. Both kernels drift, so the first op whose removal restores determinism is the lead.
- **E4.** After a mismatch, read back config words 0, 1, 2, 18, 71 and 220, plus DEST, on the failing core (`tt-exalens`).
  - This decides 4.6 directly.
- **E5.** Call `post_combine_reduce` twice back-to-back in the model with the same inputs, and compare.
  - If the second call differs, preceding-op state is implicated. Otherwise the state is set within the call.

## 7. Other potential issues (real, but they do not explain the symptom)

### 7.1 Weight over-read and missing validation

- File: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/kernels/deepseek_moe_post_combine_reduce_writer.cpp:70,153-166`.
- The writer reads `weight_tile_size` (2048 B) per weight page. The page holds 2 B.
- DRAM case:
  - Page 64 B, 8 banks. Page p = 8t + e lies in bank e at slot t.
  - The read covers tokens t to t+31 of expert e.
  - For t > 2016 it runs up to 1984 B past the buffer.
- L1 case: page 16 B. The read spans 128 slots and runs into neighbour allocations.
- Cost: about 32x (DRAM) or 128x (L1) read amplification per weight.
- Validation gap:
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/post_combine_reduce_device_operation.cpp:22-73,85-88` never checks that `weights` last dim == 1.
  - It never checks page size against the read size.
- Minimal fix:
  - Pass `weights.buffer()->aligned_page_size()` as a writer CT argument and read that many bytes.
  - Add `TT_FATAL(weights_shape[-1] == 1)`.
  - Only datum 0 is consumed (4.1), so the unpacked face keeps working.

### 7.2 `read_tile_value` zero test reads 2 garbage bytes

- File: `deepseek_moe_post_combine_reduce_compute.cpp:112-113`.
- `weight_val` includes bytes 2-3 (reshape staging bytes). A `+0.0` weight may not be skipped.
- The result is the same for finite inputs. It differs if a skipped row holds Inf or NaN.
- Minimal fix: test `(weight_val & 0x7FFF) == 0`. This also covers `-0.0`.

### 7.3 Repro does not match the model placement

- `pcr_repro.py` uses DRAM weights (O7). The model weights are probably in L1 (3.1).
- The DRAM-only "deterministic" result does not cover the model configuration. See E2.

### 7.4 DEST is not zeroed at kernel start

- ELWMUL accumulates into DEST (`llk_math_eltwise_binary.h:534-541`).
- `_llk_math_pack_sync_init_` and `_llk_pack_dest_init_` do not run ZEROACC:
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_common.h:150-169`.
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack_common.h:129-136`.
- Token 0 of each core therefore relies on the previous kernel leaving DEST clean.
- Standard kernels clear on release. The metal API always uses `clear_dest=true` (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_common_api.h:123`).
- Latent only. It affects one row, not 32.

### 7.5 Garbage rows in the `moe_fused_swiglu` output (O5)

- Capacity padding rows of the `ttnn.empty` output are never written. They hold NaN that changes between runs.
- `restored` does not reference them (O4). No effect now.
- They become a hazard if `inverse` ever points into the padding.

## 8. Confidence and open gaps

- High: 4.1 to 4.4 cannot produce O1 to O3. The source chain is complete.
- Medium: the model weights live in L1. This comes from source; no runtime print exists.
- Medium: O8 shows the same mechanism class as O1. The patterns differ (I4).
- Low: 4.6. It needs a hardware atomicity violation.
- Not checked: the exact BH grid and L1 bank count, the raw ISA text, and the meaning of undocumented log families.
- Subagent claims marked "agent-reported" were not re-read by me.
