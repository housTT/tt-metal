# Stage Review

Stage 03, optimized decoder (single-device decoder-layer optimization), re-review 5.
Previous reviews:
`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/stage_review.md`,
`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/stage_review_2.md`,
`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/stage_review_3.md`,
`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/stage_review_4.md`
(review 4: more-work-needed, four P2).
Target `Qwen/Qwen3.5-35B-A3B` revision `59d61f3ce65a6d9863b86d2e96597125219dc754`. One Blackhole
ASIC (application-specific integrated circuit), 1x1 mesh. Reviewer: fresh independent subagent,
2026 Oct 11. I ran no device, server or hardware job. I ran one CPU-only pytest and small
artifact scripts.

Abbreviations: PCC (Pearson correlation coefficient), GDN (Gated DeltaNet, the linear-attention
layer), MoE (mixture of experts), KV (key/value cache), HF (Hugging Face reference), BF16/FP32
(16/32-bit float), BFP8/BFP4 (8/4-bit block floating point), RoPE (rotary position embedding),
L1 (per-core SRAM, static random-access memory), DRAM (device main memory), CPU (host
processor), FFN (feed-forward network), us/ms (micro/milliseconds), KiB/MiB/GiB (2^10/2^20/2^30
bytes), SHA-256 (secure hash algorithm, 256 bit).

Source identity recheck (reviewer): all 37 `tt/*.py` and `tests/*.py` entries of
`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/final_evidence14/source_manifest_start.json`
match the live files by SHA-256. No live file is missing or extra. Start and end manifests are
equal. `git rev-parse HEAD` is `1d857aa3`. The SHA-256 of `git diff HEAD` outside the autoport is
the empty-input hash. Final-chain results record `optimized_decoder.py` hash `834e84e5`.

Verdict: more-work-needed

## Review-4 findings: status

| Review-4 finding | Status | Reviewer evidence |
|---|---|---|
| P2 reproducibility wording | Resolved | Contract, README, work log and telemetry note 5 now separate the two combine paths. My scan of 88 final-chain results finds two non-identical repeats: `control_grid_linear_2048` (37 blocks) and `prefill_full_1500` (28 blocks, 794 rows, new and cited in the README). Both are 2048-row tile-combine groups, max abs at most 9.8e-4, repeat PCC 1.0. |
| P2 linear token 950 | Resolved | `review4/token950/tt_{4096,4097,4100}.pt`: I confirmed the first 4096 output rows are bitwise equal across 4096, 4097 and 4100; token 950 at 2600 has PCC 0.9999995 to 4096. `review4/token950_ref/token950_threads8.json`: router logits for experts 27 and 173 are both -4.59375 (8th and 9th); their order swaps at 4097 and 2600. TT versus the 4096 reference: 0.99975. Classification as reference tie noise is earned. |
| P2 `hybrid_routed_expert_moe` rejection | Partly resolved | Wrong cause and threshold claims are corrected. An adapted isolated retry exists. The second blocker is a configurable limit (new P2 below). |
| P2 scratch page | Resolved | `masked()` now checks `absolute + n <= max_seq_len`. CPU pytest of `tests/test_prefill_plan.py`: 1116 passed (my rerun, 148 s), starts up to 260000. `final_evidence14/reuse_3/full.json` and `full_page128.json`: case 2047 with cut 1152, B2, decode trace alive, program-cache misses forbidden, guards clean, `kv_scratch_written` true. `final_evidence14/watcher_long/layer3_1500.json`: scratch block 51 with a 48-page table (masked 2048 plan), watcher log with 0 errors. Contract `serving_contract` lists the scratch reservation, ownership rule, extended width and warm-up shapes. |
| Other: planner constants | Changed; new regression (P2 below) | |
| Other: single-call 2176 | Resolved | `final_evidence14/prefill_{full,linear}_2176`. |
| Other: unaligned attribution, 40-layer bound | Resolved | README text matches review-4 numbers. |
| Profile rerun justification | Earned | In `final_evidence14/profile/tracy/linear_attention/measurement.json`, replay 100 has `first_fw_cycle` 5046586572806, which is earlier than replay 99 `last_fw_cycle` 5050586230706. A real stall cannot start before the previous replay ends, so this is a timestamp error. The other 127 spans have mean 811.99 us. The rerun has 128 spans, mean 812.15 us, max 819.2 us, and its provenance records an equal source manifest. |

## Required Work

- P2: The review-4 planner recalibration makes short whole prompts slower. The stage measured and disclosed this, but did not fix it.
  Evidence:
  - `_PLAN_COST` in `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py`
    now allows decode-path steps for remainders up to 8 tokens. Step cost is 0.8 ms (full) and
    1.1 ms (linear). When the remainder is the whole plan (`p == offset`), the option also gets a
    3.0 ms group credit.
  - My simulation of `_chunk_plan` with the final source, joined with final-chain timings
    (`final_evidence14/prefill_{full,linear}_{2,3,4,5}/default.json`):

    | Layer | Tokens | Plan | Warmed prefill |
    |---|---:|---|---:|
    | full | 3 | 3 decode steps | 4644 us |
    | full | 4 | 4 decode steps | 5623 us |
    | full | 5 | masked 32/5 | 4773 us |
    | linear | 2 | 2 decode steps | 4399 us |
    | linear | 3 | 3 decode steps | 6005 us |
    | linear | 4 | masked 32/4 | 4889 us |

  - A masked 32-row chunk at 0 has the same programs and nearly the same cost for 3 to 5 valid
    tokens. The plan is therefore about 0.85 ms slower for full 4 tokens and about 1.1 ms slower
    for linear 3 tokens. These are 18 and 23 percent of the per-layer prefill. The review-3 rule
    (decode steps only for 1 or 2 tokens) planned masked 32 for these lengths. For comparison,
    `final_evidence13/prefill_full_5` took 4556 us and `prefill_linear_5` 5304 us.
  - README lines 94 to 97 state the cause: step cost is calibrated on tails after 4096 rows
    (0.4 to 0.9 ms per step), while a whole short prompt pays 1.0 to 1.6 ms per step.
  Why this matters: The stage contract requires non-aligned lengths to be fast as well as
  correct. This is a measured default-path regression from a change in this round, with a known
  cause and a small fix. The skill does not accept disclosure alone as a fix for a measured
  anomaly.
  Required next step: Use a separate decode-step cost for plans that start with no preceding
  chunk group, calibrated from the measured 2 to 4 token rows. Alternatively, do not give the
  group credit when `offset == 0`. Add the case to `tests/test_prefill_plan.py`, rerun
  prefill 2, 3, 4 and 5 tokens for both layer kinds, and update the README table.

- P2: The `hybrid_routed_expert_moe` rejection still rests on a configurable limit.
  Evidence:
  - `review4/hybrid/hybrid_l3_1024.json` and `hybrid_l3_4096.json`: with the router L1 tensors
    freed (`l1_allocated_after_free` 0), thresholds 32, 64, 128 and 256 all fail with
    `TT_FATAL ... program.cpp:3671: state.offset <= max_size`. The README states program sizes
    of 71264 to 74240 B against a 70656 B kernel config buffer. That is 0.6 to 3.5 KiB too big.
  - The limit is not fixed. `get_ringbuffer_size` in
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/impl/program/program.cpp` (line 112)
    returns `l1_unreserved_base - KERNEL_CONFIG base`. `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/impl/device/device.cpp`
    (line 586) sets the unreserved start to the L1 end minus `worker_l1_size`. The comment in
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h`
    (line 77) says that the kernel config buffer limit is "configurable via worker_l1_size".
    A device opened with `worker_l1_size` about 4 to 8 KiB smaller gives a large enough buffer.
    `grep` finds no `worker_l1_size` use in the autoport or in
    `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_claude/`.
  - The first blocker (8 KiB per core of router L1 tensors whose addresses the decode traces
    capture) is a real constraint in the layer. Because of it, a positive isolated result would
    still need a layer adaptation, for example moving those router tensors to DRAM.
  - The README defers the re-measurement to Stage 4. The goal contract
    (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/03-03-optimized-decoder.prompt.txt`) says "No remaining
    decoder optimization work is deferred to a later stage" and "A first TTNN/API error is not
    enough to reject a material op". The p150 (one chip) profile also keeps 256 local experts.
  Why this matters: `moe_fused_swiglu` is the largest prefill op (7.5 to 7.7 ms at 4096 tokens,
  23 to 29 percent of the layer). The skill accepts a rejection only after a measurement, an
  understood correctness failure, or a minimal repro that proves the op cannot express the
  contract. A device-configuration limit that is documented as adjustable is not such a proof.
  Required next step: Rerun `review4/hybrid/hybrid_isolated.py` on a mesh opened with a smaller
  `worker_l1_size` (record the value and the resulting buffer size) at two or more thresholds,
  for 1024 and 4096 tokens. If the hybrid is not faster than `moe_fused_swiglu`, record the
  timings as the rejection. If it is faster, record the router-L1 relocation it needs, and do it
  here or defer it to Stage 4 with that evidence.

- P2: An unclassified per-token dip in the stage's own device pytest. Full-layer token 684 is at 0.99306.
  Evidence:
  - `final_evidence14/pytest/pytest_artifacts/test_optimized_decoder_real_3_1/result.json`
    and `..._real_3_current/result.json` (real weights, random activations seed 1234, 1000
    tokens): `prefill_tokens_below_0995` is 2. Token 684 is at 0.99306 and token 592 at 0.99499.
  - The same two values appear bitwise in `final_evidence10`, `11`, `12` and `13`, across five
    source hashes. The output is deterministic.
  - `grep` finds no mention of token 684, token 592 or 0.9930 in the README, work log,
    checklist or contract. The README correctness table reports this pytest only as "8 passed".
  - 0.99306 is below the linear token 1720 (0.99402), which the stage classifies as a TT-side
    error. Linear layer 0 in the same pytest has no token below 0.995 (minimum 0.99751).
  Why this matters: The anomaly rule requires classification before a pass. Review 4 applied this
  rule to token 950. A deterministic per-token dip on real weights has the same kind of risk.
  Required next step: Use the same CPU-only control as for token 950. Compare HF BF16 and FP32
  reference outputs at tokens 684 and 592 for this input, and check the router top-8 for a near
  tie. If the reference moves, record the case as controlled. If it does not, classify it as a
  TT-side error with the same rule as token 1720.

## Other Concerns

- The README headline values 32566.8 us (linear) and 25893.6 us (full) are not in any JSON
  under the stage directory. The raw values in `final_evidence14/perf_{linear,full}/default.json`
  and `summary.json` are 32567.358 us and 25894.473 us. The difference is under 1 us and the
  speedups do not change. Use the raw values.
- The token-950 TT tensors and the hybrid isolation ran on intermediate source `c974884f`, not
  the final `834e84e5`. The final chain reproduces token 950 at 0.99220 for 2600 and 4097
  tokens (`final_evidence14/prefill_linear_{2600,4097}`), so I accept this.
- Linear 4100 tokens still plans a masked 32-row tail (36004 us, linear 4098 is 34401 us). An
  estimate from the 4097 and 4098 increments gives about 36.1 ms for four decode steps. That is a
  tie, so the choice is acceptable (estimate, unverified).
- `run_layer.py` still allocates guard pages without checking them. `run_reuse.py` and
  `run_positions.py` now check them, and they cover a scratch write, so I accept this.
- Carried from earlier reviews, unchanged and accepted by disclosure: the GDN prefill head split
  (about 6.5 ms per 4096 tokens), the 16 to 23 us slower trace replay on physical device 0, no
  chained two-layer unaligned prefill check, and the unresolved root cause of prefill repeat drift.
- Memory for later stages: both routed-expert representations stay resident (852.7 MiB
  prefill dispatch plus 560.0 MiB decode per layer). The 40-layer total is about 60 GiB against
  31.8 GiB per chip. The project contract requires a working p150 (one chip) profile. Stage 3
  states this, but it is the largest open risk for Stages 4 and later.

## Hard-Check Gaps

- No full-attention unaligned profile. The recorded blocker (16-bit zone-hash collision in
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp`)
  matches the chain log (four `rc=1`/`rc=2` lines at 23:51 to 23:52, steps
  `prof_unaligned_3_*` and `report_unaligned_3_*`).
- `prefill_repeat_identical` is recorded but not asserted (accepted by disclosure).
- `tests/test_prefill_plan.py` checks plan invariants only, not plan cost or output order.
- No peak-DRAM measurement for the 262143-token prefill.

## Anomaly Ledger

- Observed anomaly: Short whole prompts are slower than one token longer (full 4 versus 5, linear 3 versus 4).
  Evidence: `final_evidence14/prefill_{full,linear}_{3,4,5}/default.json`; README lines 94 to 97.
  Affected path: `_chunk_plan` decode-step option for whole short prompts.
  Control or comparison: masked 32 plans at 4 and 5 tokens; `final_evidence13` 5-token rows.
  Likely subsystem: `_PLAN_COST` step cost calibrated on post-4096 tails only.
  Investigation performed: Stage: measured and disclosed. Reviewer: plan simulation.
  Resolution: more-work-needed (P2).

- Observed anomaly: `hybrid_routed_expert_moe` program exceeds the kernel config buffer by 0.6 to 3.5 KiB.
  Evidence: `review4/hybrid/hybrid_l3_{1024,4096}.json`.
  Affected path: prefill routed experts (candidate only).
  Control or comparison: `moe_fused_swiglu` 5906 / 8033 us, `unified_routed_expert_moe` 7620 / 8251 us (same run).
  Likely subsystem: device `worker_l1_size` configuration.
  Investigation performed: Stage: isolated retry with router L1 freed. Reviewer: source reading.
  Resolution: more-work-needed (P2).

- Observed anomaly: Full-layer pytest token 684 at 0.99306 and token 592 at 0.99499.
  Evidence: `final_evidence14/pytest/pytest_artifacts/test_optimized_decoder_real_3_1/result.json`; same values in `final_evidence10` to `13`.
  Affected path: full-attention prefill, 1000 tokens, random activations, real weights.
  Control or comparison: none.
  Likely subsystem: HF BF16 reference or router near tie (unverified), or a TT-side error.
  Investigation performed: Stage: none. Reviewer: raw-JSON scan.
  Resolution: more-work-needed (P2).

- Observed anomaly: Linear token 950 at 0.9922 for 2600, 4097 and 4098 tokens.
  Evidence: `review4/token950/`, `review4/token950_ref/`.
  Affected path: HF reference router top-k.
  Control or comparison: TT outputs bitwise equal across lengths; exact BF16 router tie.
  Likely subsystem: HF CPU reference.
  Investigation performed: Stage: saved tensors and reference control. Reviewer: tensor equality recheck.
  Resolution: controlled.

- Observed anomaly: Prefill repeat drift over many blocks on 2048-row tile-combine groups (now also full 1500).
  Evidence: `final_evidence14/control_grid_linear_2048/default.json`, `final_evidence14/prefill_full_1500/default.json`.
  Affected path: MoE k-major tile combine.
  Control or comparison: 4096 runs identical; magnitude at most 9.8e-4, repeat PCC 1.0.
  Likely subsystem: combine op or state from preceding ops; root cause open.
  Investigation performed: magnitude recorded; documentation corrected.
  Resolution: controlled in magnitude.

- Observed anomaly: Linear profile replay 100 spans 3.18 s.
  Evidence: `final_evidence14/profile/tracy/linear_attention/measurement.json`.
  Affected path: profiler timestamps.
  Control or comparison: replay 100 starts before replay 99 ends; rerun `final_evidence14/profile_linear_rerun/` is clean.
  Likely subsystem: device profiler.
  Investigation performed: Stage: rerun on the same source. Reviewer: cycle-order check.
  Resolution: controlled.

- Observed anomaly: Real-activation B32 positions, linear row 17 token 4 at 0.99393 (chain step `real_positions_0`, rc=1).
  Evidence: `final_evidence14/real_inputs/positions_0.json`.
  Affected path: GDN decode at B32.
  Control or comparison: `review2/ref_precision/` (BF16 reference versus FP32 0.99484).
  Likely subsystem: BF16 reference sensitivity.
  Investigation performed: reference-precision control.
  Resolution: controlled.

- Observed anomaly: Linear token 1720 (0.99402, TT-side) and full token 911 (0.99137).
  Evidence: `final_evidence14/perf_{linear,full}/default.json`; `review2/prefill_ref/`.
  Affected path: prefill.
  Control or comparison: BF16 versus FP32 reference.
  Likely subsystem: TT precision (1720), reference (911).
  Investigation performed: reference-precision control; deferred to the datatype sweep with one rule.
  Resolution: controlled (as accepted in earlier reviews).

## Scope Inspected

- Goal/skill paths: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/03-03-optimized-decoder.prompt.txt` (read in full),
  `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/optimize/SKILL.md` (requirements as cited in the stage-review skill).
- Artifact paths (under `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/`):
  `optimized_decoder/stage_review_4.md`, `optimized_decoder/README.md`,
  `optimized_decoder/work_log.md` (section "Stage review 4 response"), `context_contract.json`
  (`optimized_decoder_validation`, `serving_contract`), `optimized_decoder/final_evidence14/`
  (`summary.json`, `source_manifest_{start,end}.json`, `perf_*`, `prefill_*`, `reuse_{0,3}`,
  `real_inputs/positions_0.json`, `watcher_*`, `watcher_long/layer3_1500.json` and all 11
  `watcher.log` files, `pytest/`, `plan_pytest.xml`, `profile/tracy/*/measurement.json`,
  `profile_linear_rerun/`, `memory_handoff_recheck.json`, `lane{A,B,C,D}.log`),
  `optimized_decoder/final_evidence14_chain.log`, `optimized_decoder/review4/` (`token950/`,
  `token950_ref/`, `hybrid/`, `checks/`), earlier `final_evidence10` to `13` pytest and prefill
  results, `/home/hous/dev/qwen3.5-35b-a3b/runs/main/telemetry/packets/9ba51029-cb68-4989-ae12-8f9ed3691bdd.json`.
- Code paths: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py`
  (`_chunk_plan`, `_batched_prefill`, `bind_paged_kv_cache`, `_PLAN_COST`),
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/test_prefill_plan.py`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_reuse.py`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_positions.py`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_layer.py`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/test_optimized_decoder_device.py`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/impl/program/program.cpp`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/impl/device/device.cpp`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/impl/context/metal_context.cpp`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/hybrid_routed_expert_ffn/hybrid_routed_expert_ffn.hpp`.
- Commands run: SHA-256 manifest comparison; `git rev-parse`, `git status`, `git diff HEAD`
  hash outside the autoport; chain and lane log exit-code scans; JSON scans for repeat drift and
  per-token dips; torch equality check of `review4/token950/tt_*.pt`; profile span statistics
  and cycle-order check; planner simulation through `OptimizedDecoder._chunk_plan` on CPU;
  `python -m pytest models/autoports/qwen_qwen3_5_35b_a3b/tests/test_prefill_plan.py`
  (1116 passed); source reading of the kernel config buffer sizing.

## Residual Risk

- The planner cost model is still a few constants. Lengths that were not measured (for example
  4101 to 4104, or starts that are not page aligned) can pick a plan that is 1 to 2 ms slower
  (unverified).
- Prefill repeat drift has no root cause. It is bounded at rounding level in every measured run.
- A serving page table that breaks the ownership rule in `serving_contract` will get padded K/V
  writes into another request's pages. Stage 6 and later must enforce this rule in the generator.
- The selected path keeps two routed-expert representations. The one-chip (p150) profile cannot
  hold 40 layers without a later change.
