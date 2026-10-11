# Stage Review

Stage 03, optimized decoder (single-device decoder-layer optimization), re-review 4.
Previous reviews:
`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/stage_review.md`,
`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/stage_review_2.md`,
`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/stage_review_3.md`
(review 3: more-work-needed, 1 P1 and 2 P2).
Target `Qwen/Qwen3.5-35B-A3B` revision `59d61f3ce65a6d9863b86d2e96597125219dc754`. One Blackhole
ASIC (application-specific integrated circuit), 1x1 mesh. Reviewer: fresh independent subagent,
2026 Oct 10. I ran no device, server or hardware job.

Abbreviations: PCC (Pearson correlation coefficient), GDN (Gated DeltaNet, the linear-attention
layer), MoE (mixture of experts), KV (key/value cache), HF (Hugging Face reference), BF16/FP32
(16/32-bit float), BFP8/BFP4 (8/4-bit block floating point), SDPA (scaled dot-product attention),
RoPE (rotary position embedding), RM (row-major layout), DRAM (device main memory), L1 (per-core
SRAM, static random-access memory), CPU (host processor), ND (N-dimensional sharding), ULP (unit
in the last place), us/ms (micro/milliseconds), MiB/GiB (2^20/2^30 bytes), SHA-256 (secure hash
algorithm, 256 bit).

Source identity recheck (reviewer): all 37 `tt/*.py` and `tests/*.py` entries of
`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/final_evidence13/source_manifest_start.json`
match the live files by SHA-256. No live file is missing or extra. The start and end manifests are
equal. `git rev-parse HEAD` is `1d857aa3`. The SHA-256 of `git diff HEAD` outside the autoport is
the empty-input hash, as recorded. Every final-chain result JSON that I opened records
`optimized_decoder.py` hash `c974884f`.

Verdict: more-work-needed

## Required Work

- P2: The prefill reproducibility figures in the contract, README and telemetry are still wrong. The final chain contradicts them.
  Evidence:
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/context_contract.json`
    (`optimized_decoder_validation.prefill_reproducibility`), the README "Determinism" paragraph,
    the work log ("one to three 32-row blocks") and telemetry note 4 in
    `/home/hous/dev/qwen3.5-35b-a3b/runs/main/telemetry/packets/9ba51029-cb68-4989-ae12-8f9ed3691bdd.json`
    all say "one to three 32-row blocks".
  - Reviewer scan of all 80 result JSON files under
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/final_evidence13/`
    that carry `prefill_repeat_identical`. Four runs are not identical:

    | Run | Length | Differing 32-row blocks | Rows | Max abs | Repeat PCC |
    |---|---:|---:|---:|---:|---:|
    | `control_grid_linear_2048/default.json` | 2048 | 38 of 64 | 1165 | 9.8e-4 | 1.0 |
    | `control_grid_linear_2048/control_grid_4x4_only.json` | 2048 | 38 of 64 | 1161 | 9.8e-4 | 1.0 |
    | `prefill_full_1100/default.json` | 1100 | 21 of 35 | 578 | 9.8e-4 | 0.99999988 |
    | `prefill_linear_1100/default.json` | 1100 | 21 of 35 | 616 | 4.9e-4 | 1.0 |

  - All four use a MoE group of 2048 rows (1100 tokens plan as 128 + masked 1024, padded to the
    2048 dispatch bucket). The k-major tile combine is active from 2048 rows
    (`dispatch_tiles_from: 2048`, new in this round). Under the review-3 source, drift was confined
    to 1 to 3 blocks. With the tile combine, the drift now spreads over more than half of the rows.
    The magnitude bound still holds (at most 9.8e-4 here, 0.0078 largest anywhere).
  Why this matters: review-3 P2 asked for these figures to be corrected. Later stages read the
  contract and telemetry as measured facts. The drift pattern changed with a new default (the tile
  combine), and the stated root-cause experiments (`review2/drift/`) only covered the
  `post_combine_reduce` path.
  Required next step: Correct the block-count wording in the contract, README, work log and
  telemetry, using the final-chain range and its source. State that the tile-combine path drifts
  over many rows. No hardware rerun is needed for the wording.

- P2: An unclassified per-token prefill anomaly. Linear-layer token 950 falls to 0.9922 for some lengths. The stage does not mention it.
  Evidence (field `prefill_token_pcc`, physical device 3 unless noted):
  - Token 950 is 0.99220 in `final_evidence13/prefill_linear_2600/default.json`,
    `prefill_linear_4097/default.json` and `prefill_linear_4098/default.json`. These runs report
    `prefill_tokens_below_0995: 2`.
  - The same value appears in every earlier 4097 and 4098 run (`final_evidence10`, `11`, `12`,
    `review3/planner_check*`, on devices 1, 2 and 3 and on six different sources). It also appears
    in all three 2600 runs (`review3/plan_calibration/linear_2600/force_ladder.json`,
    `force_masked.json`, `review3/plan_calibration_default/linear_2600/default.json`).
  - Token 950 is never among the eight lowest tokens for 2048, 3072, 4000, 4095, 4096 or 4100.
  - For 4097 and 4100, the TT plan's first group is the same 4096-row computation (reviewer
    simulation of `_chunk_plan`: `[(0,4096,4096)]`, then one decode step or one masked 32-row
    group). The TT output at token 950 should therefore be the same in both runs. This points at
    the HF BF16 reference changing with sequence length (for example, a near-tie router decision
    on the CPU). This is unverified.
  - `grep` finds no mention of token 950 or 0.9922 in the README, work log, checklist or contract.
    The README per-token row covers 4096 tokens only (minimum 0.9940).
  Why this matters: The stage-review anomaly rule requires classification before pass. 0.9922 is
  below the 0.99402 token that the stage already classified as a TT-side error.
  Required next step: Classify the anomaly. A CPU-only control is enough: compare the HF BF16
  and FP32 reference output at token 950 for lengths 4096, 4097, 4100 and 2600 (same recorded
  input). If the reference moves, record it as reference noise, as for full-layer token 911.
  Otherwise, compare saved TT outputs (`run_layer.py --save-tensors`) for 4097 and 4100.

- P2: The `hybrid_routed_expert_moe` rejection is not earned. The recorded cause is wrong, and the claimed threshold sweep does not exist.
  Evidence:
  - The README says "TT_FATAL L1 out of memory (128.6 MB static buffers)". The work log says "the
    fused half sizes its circular buffers for the maximum tokens per expert".
  - The raw error (all six `hybrid_64_nd.log` files under
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/review3/routed_ffn/`)
    says that each bank needs 1461120 B and the bank size is 1461120 B, with 8192 B already
    allocated. The op needs exactly one full L1 bank per core, and the run fails by 8 KiB.
  - Source:
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/hybrid_routed_expert_ffn/device/hybrid_program_factory.cpp`
    (`arena_bytes_for`, around line 2883) allocates "everything above the allocator base, and
    nothing held back" as one arena. Any live L1 tensor at the call blocks it. The failure is an
    8 KiB L1 tensor that the layer keeps alive during prefill (unverified which tensor), not a
    token-count sizing limit.
  - The work log table says "threshold 64 or 128". A search of the stage documents and
    `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_claude/` finds only `hybrid_64_nd`, and no
    configuration with threshold 128 or non-ND weights.
  - `unified_routed_expert_moe` was measured slower (+903 to +3065 us). I re-derived all six rows
    from raw JSON, and they match. That part of review-3 P2 is resolved.
  Why this matters: `moe_fused_swiglu` is still the largest prefill op (7.5 ms at 4096 tokens). The
  skill does not accept a rejection based on the first API error without an adapted retry. Here
  the adaptation looks small: free or move one 8 KiB L1 tensor around the MoE call.
  Required next step: Find the live L1 allocation at the MoE call (allocator view or buffer
  report), move it to DRAM or free it, and run hybrid at two or more thresholds. Alternatively,
  record the exact reason that tensor must stay in L1. Correct the README and work-log cause and
  threshold claims.

- P2: The scratch-page KV path has no ownership coverage, has a context-edge bug, and is missing from the contract.
  Evidence:
  - Coverage. Reviewer simulation of `_chunk_plan` with each gate's page table: the reuse gate
    (2048 tokens per row, pages 32 and 128) and the B32 positions gate (32 pages per row, longest
    prompt 1006) never plan a masked chunk past the page table. Only the batch-1 `run_layer` runs
    write the scratch page (full 1100: 32 rows, 1500: 544, 2600: 448, 4000: 64). `run_layer.py`
    allocates three guard pages but never checks them (`grep guard` finds only arguments). The
    watcher runs (`final_evidence13/watcher_long/layer3_{4096,4095,1000}.json`) use tables that
    cover the masked chunk. As a result, no guard-checked, multi-request, trace-alive or watcher
    run covers a scratch-page write. The work-log statement "so the trace-alive gates exercise
    this path" holds only for decode-path steps with a scratch page bound, not for scratch writes.
  - Context-edge bug. The cost model may plan a masked 4096-row chunk whose padded positions pass
    262144. Reviewer simulation found start 258176 with length 3952 and a scratch page bound: the
    plan ends with masked chunk `(258176, 4096, 3952)`, so padded positions reach 262271.
    `FunctionalDecoder._rotary` gathers these positions with `ttnn.embedding` from the
    262144-row RoPE tables, so the gather reads past the end of the table (out-of-bounds device
    read, padded rows only). `masked()` in `OptimizedDecoder._chunk_plan` checks only page
    capacity, not `max_seq_len`. `tests/test_prefill_plan.py` (starts 0, 128, 1000 and 4096) does
    not reach this case.
  - Contract. `masked_chunk` in `optimized_decoder_validation.prefill_chunk_buckets` still says
    "masked chunk only when ... the page table covers the padded rows", which is stale. The
    contract does not mention the scratch page, its reservation rule, or the ownership rule the
    planner depends on. `capacity = page_table.shape[1] * block` treats every page-table entry as
    owned, so padded K/V rows go to any entry beyond the logical length. If a serving page table
    pads unused entries with a real page ID (block 0 or another request's page), masked prefill
    will write K/V into that page. The contract also does not list the chunk shapes a generator
    must warm before the first capture. Within one 1024-row input bucket, the plan picks 12
    (full) or 18 (linear) distinct chunk-size and mask sets by valid length (reviewer
    simulation).
  Why this matters: KV ownership and trace safety are stage-critical. The scratch page is new
  Stage-3 code. Later serving stages need these rules to allocate pages correctly.
  Required next step: (1) In `masked()`, add the condition `absolute + n <= max_seq_len` (or clamp
  padded RoPE positions), and add that invariant to `tests/test_prefill_plan.py` with a start near
  262144. (2) Add one guard-checked run with a scratch-page write and other rows' pages live. For
  example, use the reuse or positions gate with a tight per-row table and a length such as 1500,
  with a decode trace alive. Add one watcher run that writes the scratch page (for example, full
  1500 or 4000 with a tight table). (3) Update the contract: reserve the scratch page outside the
  serving allocator; every page-table entry up to the table width must be owned by the request or
  point at the scratch page; list the warm-up shape set (chunk sizes, masked and unmasked, MoE
  dispatch buckets, combine mode, extended table width).

## Other Concerns

- Planner constants. The step and group costs in `_PLAN_COST` (2.1 ms and 4.5 ms) are not
  calibrated, and decode-path steps are limited to remainders of 1 or 2 tokens. Measured increments
  are smaller: full 4097 versus 4096 +0.81 ms and 4098 versus 4097 +0.42 ms; linear +1.11 ms and
  +0.74 ms. A masked 32-row tail after 4096 costs +3.70 ms (full) and +3.29 ms (linear), because it
  forms its own MoE group. The model does not charge that group cost. Lengths of about
  4096k + 3 to 4096k + 6 likely pay 1 to 2 ms more than a decode-step plan (estimate, unverified).
- MoE bucket padding is not in the cost model. A group of 2176 rows runs MoE at 4096 rows. A
  single-call 2176 prompt took 24588 us (full) and 28480 us (linear) on intermediate source
  `c971cadb`. The README cites 23.6/27.4 ms for "a 2176-token prompt", but that number comes from
  the split run (128 tokens, then 2048) in `final_evidence13/prefill_*_2176_s128`. No single-call
  2176 run exists on the final source.
- Residual length steps are small and disclosed. Full: 1000 > 1024 (+0.42 ms), 1100 and 1500 >
  2048 (+0.30 / +0.53 ms), 2600 > 3072 (+0.03 ms), 4000 and 4095 > 4096 (+0.45 to +0.60 ms).
  Linear: 1000 > 1024 (+0.51 ms), 1500 > 2048 (+0.18 ms), 4095 > 4096 (+1.27 ms), 4095 > 4097
  (+0.16 ms). The review-3 steps were 24 to 51 percent; these are at most about 4 percent.
- Unaligned profile attribution. The README says linear 4095 adds "two more embedding gathers and
  the mask ops". In the raw rows, the largest extra item is one BF16 `ConcatDeviceOperation`
  (+358 us). The two embeddings add +81 us and +2 us. Untilize adds +84 us. Op-to-op gap is
  578 us versus 198 us
  (`final_evidence13/profile_unaligned/l0_4095_report.csv` versus
  `final_evidence13/profile/tracy/linear_attention/prefill_perf_report.csv`).
- The 40-layer estimate (59.73 GiB) uses the named-payload lower bound. Allocator totals give about
  60.6 GiB, with RoPE and module caches counted once. Label the figure as a lower bound.
- `_batched_prefill` output order depends on an indirect rule. Plans with leading alignment steps
  take the row-major path only because `count % 32 == 0` fails for count 1. The reuse cut case
  (1153 with cuts 3 and 513) passes, and I confirmed the rule by simulation. A direct condition or
  a unit test would make this less fragile.
- Carried from review 3, unchanged: the GDN prefill head split (reshape and transpose, about
  6.5 ms per 4096 tokens) has no alternative that keeps external normalization. Physical device 0
  replays traces 16 to 23 us slower. There is no chained two-layer unaligned prefill check.
- The calibration (`c971cadb`), default-plan check (`84b94bee`), routed-FFN screen (`61548568`) and
  tile-combine screen (`84b94bee`) ran on intermediate sources. Each records a snapshot and
  provenance. The final chain re-measures every default plan on `c974884f`. I accept this.

## Hard-Check Gaps

- No full-attention unaligned profile. The recorded blocker (16-bit zone-hash collision in
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp`,
  thrown at `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/impl/profiler/profiler.cpp:376`)
  matches the chain log (steps `prof_unaligned_3_*` exit 1).
- `tests/test_prefill_plan.py` checks plan invariants only, not output assembly order, RoPE range
  or MoE bucket choice.
- `prefill_repeat_identical` is recorded but not asserted (accepted by disclosure).
- No peak-DRAM measurement for the 262143-token prefill.

## Anomaly Ledger

- Observed anomaly: Linear token 950 at 0.9922 in 2600, 4097 and 4098 runs only.
  Evidence: `final_evidence13/prefill_linear_{2600,4097,4098}/default.json`; earlier chains and
  `review3/plan_calibration*`.
  Affected path: linear prefill, batched path; possibly the HF reference.
  Control or comparison: 4096, 4095 and 4100 runs (same first 4096-row group for 4097 and 4100).
  Likely subsystem: HF BF16 reference length sensitivity (unverified), or TT MoE routing near-tie.
  Investigation performed: Stage: none. Reviewer: raw-JSON scan and planner simulation.
  Resolution: more-work-needed (P2).

- Observed anomaly: Prefill repeat drift now covers 21 to 38 blocks.
  Evidence: `final_evidence13/control_grid_linear_2048/*.json`, `prefill_{full,linear}_1100`.
  Affected path: MoE combine at 2048-row groups (k-major tile combine).
  Control or comparison: review-3 source drift 1 to 3 blocks; the 4096 runs are identical.
  Likely subsystem: tile combine (`fast_reduce_nc`) or preceding op state; root cause open.
  Investigation performed: Stage: magnitude recorded per run. Reviewer: scan of all 80 results.
  Resolution: controlled in magnitude; documentation wrong (P2).

- Observed anomaly: `hybrid_routed_expert_moe` L1 out of memory by 8 KiB.
  Evidence: `review3/routed_ffn/*/hybrid_64_nd.log`; `hybrid_program_factory.cpp` arena sizing.
  Affected path: prefill routed experts (candidate only).
  Control or comparison: none.
  Likely subsystem: live L1 tensor in the layer at the MoE call.
  Investigation performed: Stage: none beyond the error. Reviewer: log and source reading.
  Resolution: more-work-needed (P2).

- Observed anomaly: Masked chunk padded positions can pass 262144.
  Evidence: reviewer simulation of `_chunk_plan` (start 258176, length 3952, scratch page bound).
  Affected path: full-attention RoPE gather in `_batched_prefill.step_input`.
  Control or comparison: none on device.
  Likely subsystem: planner `masked()` capacity rule.
  Investigation performed: Reviewer: CPU simulation and source reading.
  Resolution: more-work-needed (P2).

- Observed anomaly: Real-activation B32 positions, linear row 17 token 4 at 0.99393 (chain step
  `real_positions_0`, exit 1).
  Evidence: `final_evidence13/real_inputs/positions_0.json`.
  Affected path: GDN layer 0 decode at B32.
  Control or comparison: `review2/ref_precision/` (BF16 reference versus FP32 0.99484).
  Likely subsystem: BF16 reference sensitivity.
  Investigation performed: reference-precision control.
  Resolution: controlled.

- Observed anomaly: Linear token 1720 (0.99402, TT-side) and full token 911 (0.99137).
  Evidence: `final_evidence13/perf_{linear,full}/default.json`; `review2/prefill_ref/`.
  Affected path: prefill.
  Control or comparison: FP32 reference.
  Likely subsystem: BF16 rounding (1720); reference noise (911).
  Investigation performed: FP32 reference control.
  Resolution: controlled; the datatype sweep revisits the rule.

- Observed anomaly: Physical device 0 trace replay 16 to 23 us slower.
  Evidence: `final_evidence13/device_ab/*`.
  Affected path: host trace replay.
  Control or comparison: devices 1 to 3.
  Likely subsystem: unverified (host or PCIe, peripheral component interconnect express, path).
  Investigation performed: A/B only.
  Resolution: controlled for this stage; carry to multi-chip stages.

## Resolution Of Review-3 Findings

- P1 planner cost: resolved. MoE now runs once per group of up to 4096 planned rows (full 3072:
  25334 us, versus 27725 us before). A masked bucket is measured and adopted, the RM template
  copies are gone, the linear unaligned profile exists, all lengths are re-measured and the gates
  are rerun on the final source. Calibration winners re-derived from raw JSON: full 1100 / 1500
  masked, 2176 / 2600 / 3072 ladder; linear 1100 / 2176 / 2600 / 3072 ladder, 1500 masked. The
  final constants select each winner in reviewer simulation. The review-3 steps of 24 to 51
  percent are now at most about 4 percent. New planner and scratch-page issues are listed above.
- P2 sibling ops: partly resolved. Unified is measured and slower. The hybrid rejection is not
  earned (P2 above).
- P2 figures: partly resolved. GDN slot labels are fixed (32 slots, 65.5 MiB; census slot 0 under
  attention). The telemetry fields are renamed and hold the right values (whole-sequence 0.99972 /
  0.99952; per-token minimum 0.99402 / 0.99137). The drift block count is still wrong (P2 above).

## Re-derived Claims (match README unless noted)

- Headline (device 3): linear decode median 709.19 us (128 replays, range 700.4 to 714.0),
  prefill 32535.8 us; full 435.00 us, 25904.6 us. Device spans 812.5 / 32251.9 us (linear),
  481.6 / 25690.6 us (full). `decode_dram_pct` 21.08 / 22.39. Stack estimate 25.63 ms per token,
  1.24 s per 4096-token prefill.
- Prefill PCC 0.99972 / 0.99952. Minimum decode PCC 0.99874 / 0.99835 (equals the per-token
  minimum).
- Prefill length table: all 32 rows match the raw `warmed_prefill_host_us` and PCC values.
- Device A/B: device 0 is +16.4 us (full) and +23.3 us (linear) against the slowest other device.
- Positions random B32: linear 0.99923 / 0.99739, full 0.99855 / 0.99594 (guards clean).
  Reuse, 17 cases: linear 0.99900 / 0.99819, full 0.99852 / 0.99739, page 128 0.99851 / 0.99696.
  Recorded-activation reuse: linear minimum 0.99904, full 0.99657 / 0.99634.
- Context: linear 0.99939 / 0.99988 and 0.99955; full 0.99853 / 0.99837 and 0.99835.
- Watcher: 10 logs from the final source, 0 matches for error, assert, overflow, hang, corrupt or
  invalid.
- pytest: 8 passed with trace allocation tracking (program cache included). Planner pytest: 720
  passed, 0 failures.
- Dispatch maps: 27 of 27 pass in both combine modes.
- Unaligned profile: 4095 32.69 ms versus 4096 32.05 ms device; 1000 13.03 ms with
  `moe_fused_swiglu` 5488.6 us on 16 cores.
- Allocator per layer: 1577.0 MiB (linear), 1536.6 MiB (full).
- Chain: serial steps exit 0 except the controlled `real_positions_0` and the documented
  full-layer profiler blocker (`prof_unaligned_3_*`). The chain ends with `SOURCE_STABLE`.

## Scope Inspected

- Goal/skill paths: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/03-03-optimized-decoder.prompt.txt`
  (contract as given in the review brief);
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`;
  `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`; the three previous reviews.
- Artifact paths (under
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/`):
  `optimized_decoder/{README.md,work_log.md,checklist.md}`, `context_contract.json`,
  `optimized_decoder/final_evidence13/**` (summary, manifests, perf, prefill, noscratch, control
  grid, device A/B, positions, reuse, real inputs, context, watcher, watcher_long, pytest,
  plan_pytest.xml, dispatch maps, inventory, memory recheck, profile and profile_unaligned CSV),
  `optimized_decoder/final_evidence13_chain.log`, lane logs, `optimized_decoder/review3/**`
  (routed_ffn, plan_calibration, plan_calibration_default, scratch_check, planner_check1-4,
  tile_combine, tiles_check), result JSON in `final_evidence10`, `11` and `12` (token-950 scan);
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_claude/evidence_chain13.sh`;
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/telemetry/packets/9ba51029-cb68-4989-ae12-8f9ed3691bdd.json`.
- Code paths:
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py`
  (`OptimizationPolicy`, `DEFAULT_POLICIES`, `prefill_forward`, `_chunk_plan`, `_attention_step`,
  `_batched_prefill`, `_planned_prefill`, `bind_paged_kv_cache`, `_OptimizedAttention.forward`
  K/V fill, `_PLAN_COST`, `_LocalDispatch`);
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/functional_decoder.py`
  (`_indices`, `_rotary`);
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/{run_reuse.py,run_positions.py,run_layer.py,test_prefill_plan.py,hf_reference.py}`;
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/hybrid_routed_expert_ffn/device/hybrid_program_factory.cpp`.
- Commands run: `sha256sum`-equivalent Python manifest check; `git rev-parse`, `git status`,
  `git diff | sha256sum`; Python scripts over result JSON, inventory, perf-report CSV and pytest
  XML; CPU-only simulations of `OptimizedDecoder._chunk_plan` (imported with stub attributes, no
  device); a CPU PCC sensitivity check on the recorded layer-0 activations; `grep` over logs,
  docs and source. I did not run the CPU HF reference, to avoid CPU load during the owner's device
  timing runs. No device, server or hardware job.

## Residual Risk

- Decode is unchanged since review 2 and stays the accepted result.
- The fixes requested here are mostly documentation, one planner condition and two small coverage
  runs. If the planner condition changes plans near 262144, rerun the context gate.
- The 40-layer footprint is at least 59.7 GiB against 31.8 GiB per chip. The p150 profile depends
  on later expert placement and precision decisions.
- Prefill is not bitwise reproducible. The root cause is open, and the tile combine widens the
  affected rows.
