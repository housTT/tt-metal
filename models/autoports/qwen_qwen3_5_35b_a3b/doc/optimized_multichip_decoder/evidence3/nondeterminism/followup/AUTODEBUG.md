# AutoDebug follow-up: run-level ULP drift in `moe_fused_swiglu` and `post_combine_reduce` (Blackhole 1x2 and 1x4)

Inspection only. No device job, no build, no source edit. Prepared Oct 11 (ET).

## 0. Acronyms and names

- ULP: unit in the last place. BF16: bfloat16. BFP4/BFP8: Tenstorrent block floating point (4-bit, 8-bit mantissa).
- MoE: mixture of experts. PCR: `post_combine_reduce`. CCL: collective communication library.
- RS: reduce-scatter. AG: all-gather. ARC: Blackhole on-chip management controller (power, clocks).
- L1: core-local SRAM. CB: circular buffer. DRAM: device memory. NoC: network on chip.
- TRISC: Tensix RISC-V for compute (T0 unpack, T1 math, T2 pack). BRISC/NCRISC: data-movement RISC-V.
- FPU: Tensix matrix/eltwise unit. SFPU: Tensix vector unit. LREG: SFPU register. DEST: destination register file.
- LLK: low-level kernel library. RTA: runtime argument. MOP: Tensix macro-op program. MVMUL: FPU matrix multiply instruction.
- ADDR_MOD: Tensix address-modifier register. RWC: register read/write counter. GPR: Tensix general-purpose register.
- PRNG: pseudo-random number generator. AICLK: Tensix clock. ET: Eastern Time.
- "Band b": output columns `[128b, 128b+128)` of the MoE output (16 bands, 2048 columns).
- "Pair": one (call, device) output. "Event": a pair whose output differs from run 0 of the same process.
- "Recall": the in-wrapper repeats B, C, D, E of call A.

## 1. Verdict

- **No source defect found that produces the observed drift.** This holds for both ops and for the ops around them.
- **New finding from the saved JSON: each MoE event perturbs exactly one Tensix core, always in MoE grid column 3.**
  - 165 of 165 events on 1x4 have one "hot" band out of 16. It is always band 3, 7, 11 or 15.
  - Source maps band b to core `(b % 4, b / 4)`. So the hot core is `(3, y)` (section 2.2).
  - Inference (I2): that core's H contribution and its own down output are both perturbed. The rest is propagation.
- **New finding: the perturbed output is a deterministic function.** The same alternate (same element count, same max
  difference, same hot band) recurs in separate processes for the same call and device (O8).
- **Two statements in the request are incorrect.**
  - The weights were **not** hashed. The digest skips Python-list arguments, and gate/up/down are lists (O4).
  - The MoE call has **256 rows**, not 129. Rows 129-255 are zero padding and are routed (section 3).
- **Fabric and CCL are not necessary for drift with this signature.**
  - Stage 3 single-chip runs showed one-core, ULP-level, run-stable PCR drift with no fabric (O12, section 2.3).
  - CCL programs never run on `(3,1)`, `(3,2)` or `(3,3)`: 150 of 165 hot-core events (section 4.6).
- **Refuted by evidence plus source:** kernel config ring and RTAs, CB-region L1, program and Tensix semaphores,
  arrival-order reduction or NoC timing, DRAM read path, input values and addresses, CCL residue.
- **One run-varying, per-core input that the MoE reads and never writes does exist:** the ARC matmul-throttle word
  at L1 byte address 16, below the CB region (section 4.8).
  - Every `matmul_block` on Blackhole reads it and switches MOPs on it.
  - Source shows the throttled LoFi MOP issues the same MVMUL and ADDR_MOD sequence as the normal one. So it is
    numerically inert by source. It is not a headline cause. It is direct evidence that ARC power management acts during runs.
- **Remaining candidates (none proven):**
  - R1: per-core Tensix state that neither op programs, left by an earlier program. No consumed field found (section 4.7).
  - R2: a per-core hardware operating condition (clock, voltage, temperature, ARC throttling) that changes FPU or packer results.
- **Recommended single experiment: E7, an in-window reset-and-wait test (section 5).**
  - It separates R1 from R2 in one probe run. It needs no op source change.
- **No minimal source fix is supported.** Do not change tolerances to "fix" this.
- **Unresolved dimension:** why column 3, and why one row of it per event. No candidate explains this.

## 2. Observations versus interpretations

### 2.1 Direct observations

Logs and JSON are read as data.

| # | Observation | Source |
|---|---|---|
| O1 | Recalls never differ from A: 0 of 11008 MoE recall comparisons, 0 of 8832 PCR recall comparisons, over all seven probe files. | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder/evidence3/nondeterminism/followup/e3b_x4.json`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder/evidence3/nondeterminism/followup/e4_x4.json`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder/evidence3/nondeterminism/followup/e5_x4.json`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder/evidence3/nondeterminism/followup/e6_x4.json`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder/evidence3/nondeterminism/followup/e3b_x2.json`; `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder/evidence3/nondeterminism/e3e1_x4_full_b32.json`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder/evidence3/nondeterminism/e3e1_x2_full_b32.json` |
| O2 | 1x4 MoE: 165 events in 2048 pair comparisons (8.1%). Per device: 28, 50, 43, 44. | the five 1x4 JSON files |
| O3 | 1x2 MoE: 0 events in 512 comparisons. 1x1: 0 differing captures in 3 runs of 385 captures. | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder/evidence3/nondeterminism/e3e1_x2_full_b32.json`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder/evidence3/nondeterminism/followup/e3b_x2.json`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder/evidence3/nondeterminism/x1_full_b32.json` |
| O4 | The digest hashes only `ttnn.Tensor` arguments. Digest keys are `arg0`, `arg4`, `arg5`, `expert_region_offsets`. Gate, up and down are passed as lists, so they are skipped. | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder/evidence3/nondeterminism/followup/e4.py:35-45`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder/evidence3/nondeterminism/followup/e5.py:35-45`; call `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py:2051-2052`; lists at `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py:2102` |
| O5 | In every MoE event all 16 bands differ. Exactly one band has 3-7% more differing elements (max/median 1.031-1.068). | band maps in the 1x4 JSON files |
| O6 | The hot band is always 3, 7, 11 or 15. Counts: 15: 58, 7: 50, 11: 42, 3: 15. Device 2 never shows band 3. | same |
| O7 | In every event every valid row of every expert differs. About 2.2-2.6 M differing elements lie in pad rows. | same (`rows == differing_rows` for each expert, 165 of 165) |
| O8 | Identical alternates recur across processes: call 4/dev 3 (3341113 elements, max 0.0078125, band 11) in e3e1 and e5; call 12/dev 3 (2914688, 0.00390625, band 3) in e6 and e3b; call 21/dev 1 (3499515, 0.017578125, band 11) in e3e1 and e3b; call 1/dev 0 (3575393, 0.0078125, band 15) in e3e1 and e4. | same |
| O9 | One pair can have several alternates. e6 call 19/dev 3: 3289310 (run 1) and 3005058 (runs 2-3), both band 7. e4 call 24/dev 1: band 11 in run 1, band 15 in run 2. | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder/evidence3/nondeterminism/followup/e6_x4.json`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder/evidence3/nondeterminism/followup/e4_x4.json` |
| O10 | About 3-6 of 128 pairs per run are perturbed. Run 0 is not special: "run 0 odd" pairs occur at the single-run rate. Pairs drift in one run and return in the next (e5 call 1/dev 0: run 1 only). | same |
| O11 | PCR drifts on 1x2 (31 events) and on 1x4 apart from MoE propagation (132 events in 2048). Both PCR inputs are tensors and hash equal. | `pcr_calls[*].inputs` in e4/e5; x2 files |
| O12 | Stage 3, single chip, no fabric: PCR output differs in one 32-row chunk (once two adjacent chunks) per event. Max difference is 2^-13, 2^-12 or 2^-11. | `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_claude/micro/flake/PD_1_d0.log`, `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_claude/micro/flake/PD_2_d2.log`, `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_claude/micro/flake/PS_1_d1.log`, `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_claude/micro/flake/PS_2_d1.log`, `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_claude/micro/flake/PS_2_d2.log`, `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_claude/micro/flake/PS_3_d0.log`, `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_claude/micro/flake/PS_3_d2.log` |
| O13 | A scrub call of the op (E5) and a 999424-byte CB-region zero fill (E6) before A do not remove the drift. | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder/evidence3/nondeterminism/followup/e5_x4.log.gz`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder/evidence3/nondeterminism/followup/e6_x4.log.gz` |
| O14 | One process logs "AICLK settled at 1343 MHz, within 5% of the requested 1350 MHz". | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder/evidence3/nondeterminism/followup/e4_x4.log.gz` (UMD warning) |

### 2.2 Interpretations (testable)

- I1, band to core (proven from source):
  - Core index is `y * HGROUPS + x` (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/moe_fused_swiglu_program_factory.cpp:681`).
  - The 64 output tiles split 4 per core in index order (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/moe_fused_swiglu_geometry.cpp:156`; passed at `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/moe_fused_swiglu_program_factory.cpp:692,715`).
  - So bands 3/7/11/15 are cores `(3,0)`, `(3,1)`, `(3,2)`, `(3,3)`.
- I2, one perturbed core per event:
  - A down-matmul perturbation alone changes one band only. O5 contradicts that.
  - An H perturbation alone changes all bands equally. O5 contradicts that too (no hot band).
  - O5 + O7 fit one core whose H contribution and own down output are both perturbed.
- I3, deterministic alternate: O1 + O8 make the alternate a function of (inputs, core state). It is not noise.
- I4, state lifetime: the state exists before A. It survives the recalls, `ttnn.add`, `clone`, `zeros_like`,
  the scrub call and the L1 zero fill (O1, O13). It changes between calls, because only some pairs drift.
- I5, core choice depends partly on call data (O8) and partly on the run (O9).
- I6, magnitude is ULP-level. A dropped or doubled term would be 5-30% of a value (O5, O12).

### 2.3 Stage 3 single-chip cores (inference, depends on grid width)

- PCR maps 32-token chunk c to core `(c % W, c / W)`, row-major
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/post_combine_reduce_program_factory.cpp:86-94`).
- W = 11 is supported by "16 of 110 cores" (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/stage_review.md:135`). Not recorded in the flake logs.

| Log | First row | Chunk | Core if W = 11 |
|---|---|---|---|
| PD_2_d2 | 256 | 8 | (8,0) |
| PD_1_d0 | 1024 | 32 | (10,2) |
| PS_3_d0 | 1376 | 43 | (10,3) |
| PS_2_d2 | 640 | 20 | (9,1) |
| PS_3_d2 | 608 | 19 | (8,1) |
| PS_1_d1 | 1728 | 54 | (10,4) |
| PS_2_d1 | 288 (64 rows) | 9, 10 | (9,0), (10,0) |

- All seven events sit in logical columns 8-10 if W = 11. With W = 13 they scatter. Treat as unverified.
- Interpretation: one-core events also occur on a single chip. More chips raise the rate (O2, O3).

## 3. Instantiated configuration (corrections to the request)

| Item | Value | Source |
|---|---|---|
| Rows per MoE call | 256: 129 real, 127 zero | chunk plan `(0, 256, 129)`: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py:276-331`; zero padding `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py:394-398` |
| Rows masked from routing | none: `valid_rows = rows_here = 256` | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py:452,460`; mask only when `valid_rows < length`: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/multichip_decoder.py:261-262` |
| Capacity | 4064 rows (matches `total_rows` in the JSON) | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/multichip_decoder.py:221-222` |
| `input_m_tiles` (M_T_MAX) | 8 | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py:2053` |
| Activations | row-major BF16; compute runs `fast_tilize` on injector cores | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/kernels/moe_fused_swiglu_compute.cpp:581-602` |
| Weights | stacked gate/up BFP4, down BFP8, 64 local experts | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/multichip_decoder.py:186-189` |
| Compute config | LoFi, 16-bit DEST, half sync, `bfp8_pack_precise=true`, SiLU on the PACK thread | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/moe_fused_swiglu_program_factory.cpp:619-627`; `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/kernels/moe_fused_swiglu_compute.cpp:446-451` |
| Geometry | KR_PAD 16, HN_PAD 4, GU_CHUNKS 2, HN_BLOCK 2, EC 4, M_BLOCK 8, DEPTH_H 3, HACK_AHEAD 2, WD_SPLIT 3, DEPTH_X 1, no L1 fallback | subagent instantiation of the geometry file; CB use 832192 B of a 1461120 B budget |
| Down matmul | 4 K-blocks. K-blocks 0-2 accumulate through packer L1 accumulate (BF16). K-block 3 reloads the partial into DEST. | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/kernels/moe_fused_swiglu_compute_helpers.hpp:94-172` |
| PCR | 256 tokens, 8 chunks, cores `(0..7, 0)`, HiFi4, packer L1 accumulate | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/post_combine_reduce_program_factory.cpp:86-94,258-269` |

Column-3 roles (subagent trace, spot-checked):

- Compute uses `my_col` only in `inject_first` and `my_row` only in `slice_assigned`
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/kernels/moe_fused_swiglu_compute.cpp:584,702`).
- All compile-time args, and all other compute runtime args, are equal on the 16 cores.
- Columns 2 and 3 run the same `fast_tilize` schedule (`t ≡ x mod 4`, `t < m_eff`, `m_eff` a power of 2).
- Column 3 differs only in data movement:
  - It injects X tile rows 3 and 7, the last row-multicast round.
  - Its reduce root is `(3,3)`, from `x % KGROUPS` (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/kernels/moe_fused_swiglu_writer.cpp:241-245`).
  - `(3,3)` sends the last H round, which is K-block 3 of every core's down matmul.
  - It reads stacked gate tiles 28-31 and up tiles 12-15.
- Result: **no source path makes a column-3 core do different arithmetic.** Its roles differ in timing and data only.

## 4. Candidate ledger

"Explains" means: one core, column 3, ULP-level, stable over all recalls, varies between runs, also on single chip.

### 4.1 Kernel config ring, RTAs, CB config: refuted

- Kernel text, RTAs, CB config and program semaphores are written into a ring for every launch
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/impl/dispatch/worker_config_buffer.cpp:43-146`;
  read at `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/hw/firmware/src/tt-1xx/trisc.cc:153-217`).
- Each launch lands at a different ring offset. A, B, C, D, E are five launches.
- Stale ring content would change between recalls. O1 shows no change.

### 4.2 L1 above the zeroed range: refuted for these ops

- The MoE uses 832192 B of CB. E6 zeroed 999424 B from the CB base. All MoE CBs were covered (O13).
- The MoE reads L1 only through its CBs, its ring semaphores and RTAs, firmware mailboxes, and L1 word 16 (section 4.8).
- Above the CB region sit allocator buffers: CCL global semaphores
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/multichip_decoder.py:320-328`)
  and the decode all-reduce L1 buffer (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/multichip_decoder.py:296-317`). No MoE or PCR kernel reads them.

### 4.3 Semaphores: refuted as the ULP source

- Program semaphores start at 0 on every launch (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/moe_fused_swiglu_program_factory.cpp:462-469`).
- CCL global semaphores change only on CCL worker cores in row 0 (section 4.6).
- Tensix semaphores persist (seeded at boot: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/hw/inc/internal/tt-1xx/blackhole/c_tensix_core.h:415-422`).
  - `PACK_DONE` is never seeded by the op. A stale value of 1 or more makes the K-block-3 reload read early.
  - That drops a quarter of the contraction: a gross error, not 1 ULP (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/kernels/moe_fused_swiglu_compute_helpers.hpp:106-110,167-171`).

### 4.4 Arrival-order reduction or NoC timing: refuted

- Fold order is fixed by slot index (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/kernels/moe_fused_swiglu_compute.cpp:165-225`). Down K-block order is fixed (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/kernels/moe_fused_swiglu_compute_helpers.hpp:94-179`).
- Every handshake failure in this op reads another block, expert or round: a gross error.
- A timing race would make some recalls differ. 0 of 11008 differ (O1).

### 4.5 DRAM read path, input bytes and addresses: refuted

- Activations, counts, expert ids and offsets hash equal for drifting calls (O4 keys).
- D and E move the activation to new addresses; the output does not change (O1).
- Weights were not hashed (O4). Persistent weight corruption is still ruled out:
  - A corrupted weight stays corrupted, so later runs would keep the alternate.
  - O10 shows pairs that drift in one run and return in the next.
- Pad X rows can read past the activation buffer (section 6.3). Those rows are never multiplied, and D/E show no effect.

### 4.6 CCL residue on the hot cores: refuted for `(3,1..3)`

- `all_reduce_async` runs RS, then AG (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.cpp:402-457`).
- Worker cores are taken row-major from `(0,0)` (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:480-541`).
- At this size RS and AG use 12 cores in row 0, plus at most `(0,1)`. Mux on `(0,0)`, `(3,0)`, `(6,0)`, `(9,0)`; workers on the rest
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp:436-449`).
- `(3,1)`, `(3,2)`, `(3,3)` run no CCL kernel and receive no fabric write. They hold 150 of 165 hot-core events.
- The RS compute kernel leaves only state that every later `compute_kernel_hw_startup` rewrites.
- `(3,0)` is a mux core, and it is the least frequent hot core (15 of 165).

### 4.7 Tensix or SFPU state consumed but not programmed (R1): open, no hit

Persistence across programs on a Tensix core:

| State | Reset per launch | Source |
|---|---|---|
| CB interfaces and counters, RTA bases, kernel `.data/.bss`, instruction cache | yes | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/hw/firmware/src/tt-1xx/trisc.cc:97-106,157-183`; `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/hw/firmware/src/tt-1xx/brisc.cc:458,504-556` |
| DEST, srcA, srcB contents | yes, at kernel start | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/hw/ckernels/blackhole/metal/common/chlkc_list.h:29-36` |
| Tensix config registers, SETC16 registers, ADDR_MODs, MOPs, replay buffers | no | each kernel's init must reprogram |
| SFPU LREGs and programmable constants, SFPU config | no (LREG11 = -1 at boot only) | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/hw/inc/internal/tt-1xx/blackhole/c_tensix_core.h:261-267` |
| Tensix semaphores | no (boot only) | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/hw/inc/internal/tt-1xx/blackhole/c_tensix_core.h:415-422` |
| GPRs, `cfg_state_id`, PRNG seed | no (boot only) | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/hw/firmware/src/tt-1xx/trisc.cc:114-127` |
| L1 word 16 (ARC scratch) | no; written only by ARC firmware | section 4.8 |

#### 4.7.1 Ledger for the MoE compute path (subagent, spot-checked by me where a line is cited)

| Consumed state | Programmed by the op | Note |
|---|---|---|
| Unpack/pack formats, tile descriptors | yes, startup and every reconfig | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_common.h:66-81`; `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/common/inc/cunpack_common.h:761-848` |
| Stochastic-rounding enables (FPU, gasket, packer) | yes, cleared | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/common/inc/cunpack_common.h:839-847` |
| `Fp32_enabled`, SFPU FP32, INT8, src zero-flag baseline | yes | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_common.h:31-58`; matmul init also re-establishes the zero flag (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_matmul.h:663-697`) |
| Packer L1 accumulate, exponent threshold, edge masks, ReLU, `PCK_DEST_RD_CTRL` | yes | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h:346-462` |
| Matmul MOP, replay entries 16-31, ADDR_MOD 0-5, fidelity counter | yes, every `matmul_block_init` | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_matmul.h:296-447,663-697` |
| SiLU (PACK thread): ADDR_MOD_7, DEST offset, D RWC, SFPU config, `vConstFloatPrgm0 = 2.0` | yes | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h:13-37`; `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid.h:18-66`; `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_recip.h:23-53,380-384` |
| SiLU rounding | encoded in the instruction (round to nearest), no PRNG | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h:24` |
| PACK-thread DEST base | consistent: both helpers read the PACK copy of `dest_offset_id` | `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel.h:413-416`; `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h:657-660` |
| SFPU implied load/store format | follows SrcB format; UNPACK may change it while PACK runs SiLU | all formats used are 8-bit-exponent formats; expected neutral (uncertain) |
| LREG11, LREG13, LREG14 | no | only experimental softmax_k/sinkhorn kernels overwrite LREG11; none run in this model. Whether compiled SiLU reads them needs the PACK ELF (uncertain) |
| PACK ADDR_MOD_1 used by pack-release ZEROACC | not by this op (subagent) | a wrong value would move DEST addressing: gross, not ULP (uncertain) |
| DEST `remap_addrs`/`swizzle_32b`, `CLR_DVALID_SrcA_Disable` | set by `fast_tilize`, never cleared | same schedule on columns 2 and 3; scrub call sets them identically (O13) |
| `GS_LF`, `Bfp8_HF`, `FIDELITY_BASE_Phase`, packer downsample mask | no writer anywhere | keep power-on values; cannot change per run |
| `Zero_Flag_disabled_dst` | no | only experimental `topk_xl` writes it and restores it |
| ARC throttle word (L1 byte 16) | no; read on every `matmul_block` | section 4.8 |

Constraint every R1 candidate must meet:

- Not written by the op on any thread or path, nor by `ttnn.add`, `clone`, `zeros_like` or the zero fill.
- Written per call, run-dependently, by some earlier program on that core.
- Consumed by both the H path and the down path of the core, with ULP-level effect.
- No consumed field meets all four.

### 4.8 ARC matmul-throttle word: run-varying input, numerically inert by source

- `matmul_block` on Blackhole calls `matmul_block_math_dynamic_throttle`
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/hw/inc/api/compute/matmul.h:273-276`).
- It reads `MEM_L1_ARC_FW_SCRATCH` (L1 byte 16) on every call. Odd means throttle (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/hw/inc/api/compute/matmul.h:25-31,65`;
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h:97`).
- On a change it re-runs `llk_math_matmul_init` with throttle level 5 or 0, mid-kernel (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/hw/inc/api/compute/matmul.h:66-79`).
- The word is written only by ARC firmware. No kernel writes it. E6 did not zero it (it is below the CB base).
- So it is exactly a "run-level state the op reads but does not write". It also matches 1x4 > 1x2 > 1x1 if ARC throttles on power.
- Why it is not the headline cause:
  - Normal LoFi MOP: ADDR_MOD order `[0,1,0,2][0,1,0,4][0,1,0,2][0,1,0,5+CLR]` from one 16-entry replay (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_matmul.h:388-447`).
  - Throttled level 5: 2 outer x 2 inner loops of replay `[0,1,0]` plus loop ops 2, 4, 2, `5+CLR`, with NOPs between (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_matmul.h:450-637`).
  - The ADDR_MOD values are the same in both modes; only ADDR_MOD_6 is added, and LoFi does not use it (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_matmul.h:63-88`).
  - Same MVMUL order and same address walk give the same DEST result. The MOP rewrite waits on `mop_sync`
    (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_template.h:354-369`).
  - PCR and Stage 3 `fast_reduce_nc` have no matmul, yet they drift too (O11, O12).
- What would make it causal: a hardware difference between NOP-spaced and back-to-back MVMULs, or ARC
  actions beyond this word (clock or voltage change). Neither is visible in source.

### 4.9 Hardware operating condition (R2): open

- Fits: per core, ULP-level, stable for tens of milliseconds, varies across runs (O1, O5, O10).
- Fits: the rate rises with active chips: 1x4 MoE 8.1%; 1x2 and 1x1 zero (O2, O3). ARC throttling exists (4.8).
- Fits: single-chip events still occur (O12).
- Against: the hot core always has the column-3 role on four chips. A fixed physical defect would not follow a logical role.
  - Logical x = 3 is translated x = 4 on every chip (subagent, UMD coordinate manager). Physical placement depends on
    each chip's harvesting mask, which the run artifacts do not record.
- Against: O8 needs the same failing operations in two processes.
- Gap: no per-call AICLK, voltage, temperature or throttle-word telemetry exists in the evidence. O14 is the only clock datum.

### 4.10 Summary

| Candidate | Code contradiction | One core | Column 3 | Stable over recalls | Varies by run | Verdict |
|---|---|---|---|---|---|---|
| Config ring / RTAs | none | n/a | no | no (rewritten) | yes | refuted |
| L1 above CB region | none (not read) | n/a | no | yes | yes | refuted |
| Semaphores | none | yes | no | yes | yes | refuted (gross, not ULP) |
| Fabric/NoC timing, fold order | none | no | timing role only | no | yes | refuted |
| DRAM read path | none | no | no | no | yes | refuted |
| CCL residue | none | yes | no (`(3,1..3)` untouched) | yes | yes | refuted |
| ARC throttle word | none (same MVMUL sequence) | possible | no | yes | yes | inert by source; keep as a probe |
| Unprogrammed Tensix/SFPU state (R1) | none found | yes | unexplained | yes | yes | open |
| Operating condition (R2) | n/a | yes | unexplained | yes | yes | open |

## 5. Discriminating experiment (device job; the owning agent must approve)

### 5.1 E7: in-window reset-and-wait test (recommended single experiment)

Setup: the E6 probe on 1x4, 6 runs, same seed. Inside the MoE wrapper, after B, add:

1. **Telemetry at A.** Read L1 byte 16 on all 16 MoE cores (tt-exalens) and ARC telemetry (AICLK, ASIC temperature, voltage).
2. **F: reset, then recall.**
   - Run one `generic_op` compute kernel on `(0,0)-(3,3)` that re-establishes persistent Tensix state:
     - full unpack/math/pack configure in both config banks; the existing
       `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/hw/inc/api/compute/experimental/compute_kernel_hw_cleanup.h:66-70` covers this part;
     - `t6_semaphore_init` of every Tensix semaphore to its boot value;
     - SFPU config reset, all programmable constants (LREG11 = -1), LREG0-7 zero;
     - DEST zero in both halves; clear `remap_addrs`/`swizzle_32b`.
   - Recall the MoE with the same arguments. Save `F`.
3. **G: wait, then recall.** Idle the device for 3 s. Recall. Save `G`. Repeat the telemetry.
4. **Control run.** One extra run applies the reset kernel before A on every call.

Read-out per pair, against the per-pair majority output over all runs:

| F | G | Conclusion |
|---|---|---|
| reverts on drifting pairs only | equals A | R1, state inside the reset scope. Next: split the reset into parts. |
| equals A | reverts | R2, a time-varying condition. Correlate with the telemetry and the throttle word. |
| equals A | equals A | State outside the reset scope, stable for seconds. Next: tt-exalens diff of the hot core (5.2). |
| differs on every pair | any | The reset kernel leaves its own state. Use the control run as the new reference. |

Why this one:

- O1 shows the output is constant across all recalls. So any change in F or G is caused by the step before it.
- One run separates R1 from R2. The telemetry tests the only run-varying input found in source (4.8).

### 5.2 Follow-ups (after E7)

- Localise the stage on the hot core. Add `hash_cb_trisc` probes, gated on `DEBUG_CB_HASH`, after gate/up, fold, SiLU,
  H slice and down on all 16 cores (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/hw/inc/api/compute/debug/cb_hash.h:25-79`).
  The first CB hash that differs between runs names the stage.
- If E7 points to R1: dump config registers (both banks), SETC16 registers, GPRs, Tensix semaphores and SFPU constants on
  the hot core and on `(2, y)` with tt-exalens, between B and C. Diff them.
- If E7 points to R2: repeat at a lower AICLK or with a 30 s cool-down between runs. Compare event rates per device.
- Map PCR events to cores: save PCR output row indices. PCR runs on `(0..7, 0)`, the row that also hosts CCL cores.

## 6. Other potential issues (real, but they do not explain the drift)

### 6.1 Probe gaps

- The digest skips list arguments (O4). Hash `args[1][0]`, `args[2][0]`, `args[3][0]` too.
- The JSON keeps no raw tensors, no PCR row positions, and no per-call telemetry.

### 6.2 Zero padding rows are routed through the MoE

- The plan pads 129 rows to 256 and the model passes `valid_rows = 256`
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py:452,460`).
- So the row mask at `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/multichip_decoder.py:261-262` never runs.
- 127 zero tokens route to the same few experts. Those experts carry 128-142 rows (band-map `rows`).
- Effect: about twice the MoE work for this prompt length. Real-token values do not change (rows are independent).

### 6.3 MoE pad X rows can read past the activation buffer

- `row` is clamped to `M_T_MAX - 1`, not to the region
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/kernels/moe_fused_swiglu_reader.cpp:722-726`).
- `region_ok` checks `m_t`, not `m_eff` (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device/kernels/moe_fused_swiglu_reader.cpp:561-562`).
- For a region near row 4064, tile row 7 reads up to 32 sticks past the buffer. Those rows are never multiplied (`m_rows`).

### 6.4 PCR weight over-read

- The writer reads 2048 B per weight page. The page holds 2 B and is aligned to 64 B in DRAM
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/kernels/deepseek_moe_post_combine_reduce_writer.cpp:162-166`).
- For late tokens the read passes the buffer end by up to 1984 B.
- Only bytes 0-511 are unpacked, and math uses srcB row 0 with `SRCB_BCAST_ALL`
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB.h:119-127`;
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h:42,136-139`).
- PCR recall D clones both inputs; the output is unchanged (O1). So the garbage is not consumed.
- `read_tile_value` includes 2 padding bytes in the zero-weight skip test
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/kernels/deepseek_moe_post_combine_reduce_compute.cpp:110-114`). Value-neutral for finite inputs.

### 6.5 CCL semaphore double buffer alternates per wrapper, not per call

- Attention and FFN wrappers share one `_CCL_SEMAPHORES[mesh]` list, each with its own toggle
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/multichip_decoder.py:320-328,367-368,394-395`).
- Row i runs the attention and the FFN all-reduce back to back on the same set (i % 2).
- A device that runs ahead could send increments into a set that a lagging device still uses.
- The probe syncs every call, so it cannot show this. Unverified in production.

### 6.6 `fast_tilize` leaves DEST remap and `CLR_DVALID_SrcA_Disable` set

- Set in `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_math_fast_tilize.h:30-50` and not restored by uninit.
- Later kernels on that core inherit them. `deepseek_compute_kernel_init` (router) also sets remap
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/hw/inc/api/compute/experimental/deepseek_compute_kernel_hw_startup.h:36-38`).
- Deterministic per op sequence. It cannot give run-to-run drift on its own (O13).

### 6.7 Dynamic matmul throttle rewrites ADDR_MOD_6 mid-kernel

- The throttled init programs MATH ADDR_MOD_6 (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_matmul.h:79-88`).
- MATH-thread SFPU code also uses ADDR_MOD_6 (reciprocal: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_recip.h:403-409`).
- A kernel that runs `recip_init` once, then interleaves `matmul_block` and MATH-thread SFPU, gets a wrong ADDR_MOD_6
  only when ARC throttles. Not this op (its SFPU runs on PACK). Worth an audit of SDPA-style kernels. Unverified.

## 7. Confidence and open gaps

- High: one core per MoE event, always column 3 (165/165; band-to-core mapping proven from source).
- High: weights were not hashed; the MoE sees 256 rows with 127 routed zero rows.
- High: config ring, CB-region L1, timing races and DRAM reads cannot produce O1 with O5.
- High: the throttled and normal LoFi MOPs issue the same MVMUL sequence (read by me).
- Medium: CCL leaves no state on `(3,1..3)` (subagent core trace; grid width not recorded at runtime).
- Medium: no consumed Tensix field stays unprogrammed (ledger from LLK source, not from ISA text).
- Low: Stage 3 hot cores in columns 8-10 (depends on W = 11).
- Unresolved: why column 3, and why one row of it per event.
- Not checked: Blackhole ISA text; per-chip harvesting masks; what ARC does besides writing L1 byte 16.
