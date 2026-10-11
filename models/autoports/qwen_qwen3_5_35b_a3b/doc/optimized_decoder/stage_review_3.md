# Stage Review

Stage 03, optimized decoder (single-device decoder-layer optimization), re-review 3.
Previous reviews: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/stage_review.md`
(review 1) and `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/stage_review_2.md`
(review 2, more-work-needed, 1 P1 and 3 P2). Target `Qwen/Qwen3.5-35B-A3B` revision
`59d61f3ce65a6d9863b86d2e96597125219dc754`, one Blackhole ASIC (application-specific integrated
circuit) in a 1x1 mesh. Reviewer: fresh independent subagent, 2026 Oct 10. No device, server or
hardware job was run.

Abbreviations: PCC (Pearson correlation coefficient), GDN (Gated DeltaNet, the linear-attention
layer), MoE (mixture of experts), KV (key/value cache), HF (Hugging Face reference), BF16/FP32
(16/32-bit float), BFP8/BFP4 (8/4-bit block floating point), SDPA (scaled dot-product attention),
RM (row-major layout), DRAM (device main memory), L1 (per-core SRAM, static random-access memory),
TTFT (time to first token), ULP (unit in the last place), us/ms (micro/milliseconds), MiB/GiB
(2^20/2^30 bytes), CPU (host processor), NoC (network on chip).

Source identity recheck (reviewer, SHA-256, secure hash algorithm 256): all 36 `tt/*.py` and
`tests/*.py` entries of
`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/final_evidence10/source_manifest_end.json`
match the live files. No live file is missing or extra. Start and end manifests are equal.
`git rev-parse HEAD` is `1d857aa3`. The SHA-256 of `git diff HEAD` outside the autoport is the
empty-input hash, as recorded. The extracted `source_snapshot.tar.gz` holds the same
`optimized_decoder.py` (`0d97af0e...`). Every final-chain result JSON that I opened records
`optimized_decoder.py` hash `0d97af0e`.

Verdict: more-work-needed

## Required Work

- P1: The prefill planner is not cost-monotonic. Unaligned and multi-chunk aligned lengths are much slower than the next 4096 or 1024 single chunk.
  Evidence (raw JSON in
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/final_evidence10/prefill_*/default.json`
  and `perf_*/default.json`, device 3, field `warmed_prefill_host_us`, final source):

  | Tokens | Path | full us | linear us |
  |---|---|---:|---:|
  | 1000 | batched | 16975 | 18906 |
  | 1024 | tiled, one chunk | 11747 | 12948 |
  | 3072 | tiled, 2048 + 1024 | 27725 | 32372 |
  | 4000 | batched | 37655 | 41077 |
  | 4095 | batched, masked tail | 39897 | 42593 |
  | 4096 | tiled, one chunk | 26344 | 33051 |
  | 4097 | batched, 4096 + decode-path token | 30483 | 37788 |

  - 1000 tokens cost 44 to 46 percent more than 1024 tokens. 4000 and 4095 tokens cost 24 to 51
    percent more than 4096 tokens. 4095 tokens cost 9.4 ms (full) and 4.8 ms (linear) more than
    4097 tokens. Aligned full 3072 costs more than aligned full 4096.
  - Cause 1, tiled ladder (reviewer source reading of `OptimizedDecoder.prefill_forward` in
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py`):
    each ladder chunk calls `self.layer.forward(..., mode="prefill")`, so each chunk runs its own
    MoE dispatch and `moe_fused_swiglu`. Full 3072 = 16039 (2048) + 11747 (1024) = 27786 us,
    which equals the measured 27725 us. `_batched_prefill` already runs one MoE call per group,
    but the tiled path does not.
  - Cause 2, batched path: attention runs once per ladder chunk. From 4095 versus 4097
    (same single MoE group of 4096 rows), 7 extra attention chunks cost about 1.3 ms each (full)
    and 0.7 ms each (linear). The intermediate-source profile
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/review2/smallprof/l3_1000_report_stacked.csv`
    shows SDPA at 1252 us over 6 calls for 1000 tokens versus 310 us for one 1024-token call
    (`l3_1024_report_stacked.csv`). Device time is 13.0 ms versus 9.8 ms. Host time adds about 2 ms
    more (395 versus 173 ops). No profile exists for 4000 or 4095 tokens.
  - Cause 3, fixed copies in `_batched_prefill`: it pads the whole input by `sizes[-1]` = 4096
    rows (`extended`), makes a zero tensor of that size, converts it to RM, pads the input again by
    32 rows and converts it to RM, and converts the full output back to tile layout. `extended`
    only provides a shape for `zeros_like`. At 262143 tokens each of these tensors is about 1 GiB
    (reviewer arithmetic, unverified on device).
  - The README and work log state "RM round trips cost about 4 ms at 4096 tokens versus the tiled
    path (4095 versus 4096 tokens)". The raw 4095 versus 4096 difference is 13.6 ms (full) and
    9.5 ms (linear). About 4 ms matches 4097 versus 4096, not 4095 versus 4096. The remaining
    9.4 / 4.8 ms is not attributed anywhere.
  - Not evaluated: a masked pad of the remainder to the next bucket. The GDN masked tail already
    zeroes beta and decay on padded rows, and `_OptimizedGDN.forward` only requires a 32-row
    multiple. The mask is hard-coded to 32 rows (`self._index_offsets[32]`). For full attention a
    masked pad is safe only while the padded rows stay in pages the request owns; the stage
    already has a page-coverage rule for the 32-row tail.
  Why this matters: Real prompts are mostly unaligned. TTFT for a 1000-token prompt is about 45
  percent higher than necessary on every layer. The stage contract says no decoder optimization is
  deferred, and the planner is new Stage-3 code. A documented attribution is contradicted by the
  raw data.
  Required next step: (1) Batch MoE across tiled ladder chunks, or route aligned multi-chunk
  lengths through a MoE-batched path without RM round trips. (2) Measure a masked pad to the next
  bucket for GDN layers, and for full attention where page coverage allows it; adopt it where it
  is faster, or record the exact blocker. (3) Remove the full-size template copies in
  `_batched_prefill` (allocate the zero output directly). (4) Profile one unaligned prefill (4000
  or 4095) on the final source with `tt-perf-report` and attribute the per-chunk attention cost
  (SDPA, projections, embedding gathers, host dispatch). (5) Re-measure 1000, 1024, 3072, 4000,
  4095, 4096 and 4097 and rerun the affected correctness gates (reuse, positions, context,
  watcher). (6) Correct the README and work-log attribution.

- P2: The blocker for the largest prefill op is not earned. Sibling expert ops in the same checkout were never measured.
  Evidence:
  - `moe_fused_swiglu` is the largest prefill op: 7650 us of the full-layer prefill (30.2 percent)
    and 7517 us of the linear-layer prefill (23.1 percent)
    (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/final_evidence10/profile/tracy/full_attention/prefill_perf_report_stacked.csv`,
    `.../linear_attention/prefill_perf_report_stacked.csv`). At 1024 tokens it is about 50
    percent of device time (`review2/smallprof/l3_1024_report_stacked.csv`, 4913 us).
  - README and work log: "Removing the expert-loop floor needs a change to the op (experts on
    disjoint core groups); this stage records it as the blocker."
  - The same checkout has `ttnn.experimental.deepseek_prefill.unified_routed_expert_moe` and
    `hybrid_routed_expert_moe`
    (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/`,
    `.../hybrid_routed_expert_ffn/`). Their bindings take the dispatched buffer, per-expert region
    offsets, per-expert token counts, an expert-ID table and per-expert gate/up/down lists. The
    unified op picks each expert's chunk geometry at run time from the device token count. The
    hybrid op sends low-count experts to the fused path and high-count experts to the unified path
    in one program. The stage's own
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/local_dispatch_diagnosis.md`
    (line 121) names them as "possible later FFN substitutions". A search of the autoport
    tests, `tt/` and `runs/main/stage03_claude` finds no call to either op. The unified op
    validation requires BFP8 tiled input in one branch (`TT_FATAL` at
    `unified_routed_expert_ffn/device/*device_operation.cpp:78`); the policy already has
    `dispatch_input_dtype: "bfp8"`. Whether either op is faster is unverified.
  Why this matters: The skill requires rejected material optimizations to be earned with a
  measurement or an exact op-contract blocker. The expert loop is the largest single prefill item
  and sets the fixed per-call floor.
  Required next step: Measure `unified_routed_expert_moe` and `hybrid_routed_expert_moe` (with a
  swept `hybrid_token_threshold`) on the existing dispatch maps at 1024, 2048 and 4096 tokens under
  the selected BFP8 weights, with recorded activations and per-token prefill PCC. Adopt a faster
  passing candidate, or record the exact `TT_FATAL` or measured result.

- P2: Determinism and memory figures in the stage contract are inconsistent with raw artifacts.
  Evidence:
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/context_contract.json`
    `optimized_decoder_validation.prefill_reproducibility` says "one or two 32-row blocks" and
    "maximum absolute difference below 1e-3". `final_evidence10/prefill_linear_2048/default.json`
    shows three blocks (20, 31, 54). `review2/bfp8_grid/full_1024/small4x4_large11x8.json` (4x4
    grid at 1024 tokens, the same grid the default uses at that size, intermediate source
    `5399f0c7`) shows maximum absolute difference 0.0078125. The telemetry packet
    `/home/hous/dev/qwen3.5-35b-a3b/runs/main/telemetry/packets/9ba51029-cb68-4989-ae12-8f9ed3691bdd.json`
    repeats "below 1e-3".
  - GDN slot accounting: the README row "GDN state, 32 configured slots ... 63.5 MiB" and the
    contract `gdn_recurrent_and_conv_state_slots: 63.45` count 31 slots. In
    `final_evidence10/inventory/layer0.json` the class holds 31 FP32 `[1,32,128,128]` roots and 31
    conv roots; slot 0 (`attention.recurrent_state`, `fused_conv_state`) and the zero templates are
    in "attention". The checklist says 65.5 MiB. 32 x (2 MiB + 48 KiB) = 65.5 MiB.
  - The telemetry field `prefill_pcc_min_vs_hf` holds the whole-sequence PCC (0.99971 linear),
    not the per-token minimum (0.99402).
  Why this matters: Later full-model and serving stages read the contract and telemetry as
  measured facts.
  Required next step: Correct the drift wording (block count range and the largest observed
  maximum absolute difference, with its source), the GDN slot row (32 slots, or label 31 plus
  slot 0 in attention), and the telemetry field name or value. Do this with the P1 rework.

## Other Concerns

- Returned tensor shape from `_batched_prefill`: the function returns
  `ttnn.reshape(tiled, (batch, length, hidden), tiled.padded_shape)`, where `tiled` has
  `physical_len + 4096` rows. A 100-token result therefore carries a padded shape of about 4224
  rows (reviewer source reading, unverified on device). Layer tests read the result through
  `to_torch` and do not see it. A chained full-model prefill can pass this tensor to the next
  layer, and tile-layout ops can process the padded volume. Add a chained two-layer unaligned
  prefill check, or return a compact tensor.
- Prefill nondeterminism (review-2 P2): the stage ran the requested experiments and disclosed the
  result. The probe logs show more than "device state": in
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/review2/drift/logs/E15_l0_d0.log`,
  `d_restored` (the reduce input) is identical across runs while `d_out` differs, and a later call
  in the same process (`d_out_late`) differs from `d_out`. Two different reduction kernels
  (`post_combine_reduce` and the `fast_reduce_nc` tile combine) both drift. This pattern is also
  consistent with a write-completion or ordering hazard in a preceding op. The `tt-noc-dump`
  barrier check was not tried. Root cause remains open.
- Precision rule consistency: BFP4 decode routed down was rejected for one token at 0.9946
  (`c5v_full/c5.json`, recorded activations, -7.9 us per full layer). The selected default has one
  TT-side prefill token at 0.99402 (linear token 1720, HF BF16 versus FP32 reference 0.99942),
  accepted because it is above the BF16-reference floor (0.99323 at a different token). The README
  calls both "the same per-token rule". They are not the same rule. BFP8 dispatch also adds 408 MiB
  per layer and raises the 40-layer estimate to 59.76 GiB. The datatype-sweep stage should revisit
  both choices with one rule and full-model accuracy.
- k-major tile combine was measured only under BFP4 dispatch with single samples
  (`review2/large_grid/*`). With the 8x8 grid it is faster in 5 of 6 rows at 2048 tokens and above
  (-104 to -513 us; linear 4095 +361 us). A size-dependent combine was not considered. Its drift is
  larger.
- GDN prefill head split: `ReshapeViewDeviceOperation` 5079 us (6 calls) and
  `TransposeDeviceOperation` 1412 us (4 calls) in the final linear prefill profile. Review 2 asked
  for an alternative head split that keeps external normalization. The review-2 response does not
  address it.
- `_ZERO_ROWS` keeps 33 MiB of zero tensors per mesh only as shape templates for `zeros_like`.
- The fixed-cost attribution (`review2/smallprof/`) is from the intermediate BFP4-dispatch source,
  not the final BFP8 source. The README discloses this.
- Each warmed prefill time is one sample after one warmup. Differences below about 1 percent
  (for example the tile-combine rows) are inside device-to-device spread
  (`final_evidence10/device_ab`, prefill 26228 to 26449 us full).
- Full-attention tails below 32 tokens use token-wise decode steps when the page size is above 32
  (`reuse_3/full_page128.json` passes). Every timed run uses page size 32. The serving page size is
  still open.
- Physical device 0 replays traces 16 to 21 us slower (`final_evidence10/device_ab`). Disclosed,
  not investigated. Later multi-chip stages must measure it.

## Hard-Check Gaps

- No profile of an unaligned prefill (4000 or 4095 tokens) on the final source.
- No chained-layer test that feeds a `_batched_prefill` output into another layer.
- No peak-DRAM measurement for the 262143-token batched prefill.
- Per-token prefill PCC is measured against the BF16 HF reference only; TT versus the FP32
  reference is not recorded.
- `prefill_repeat_identical` is recorded with magnitudes but not asserted (accepted by the
  disclosure).

## Anomaly Ledger

- Observed anomaly: Prefill time is non-monotonic in length (3072 full above 4096; 4095 above
  4097 by 9.4 / 4.8 ms; 1000 above 1024 by 44 to 46 percent).
  Evidence: `final_evidence10/prefill_*`, `perf_*` (P1 table).
  Affected path: `OptimizedDecoder.prefill_forward` tiled ladder and `_batched_prefill`.
  Control or comparison: 4096 and 1024 single chunks; 4097 (one MoE group plus one decode token).
  Likely subsystem: MoE per tiled chunk; per-chunk attention fixed cost; RM template copies.
  Investigation performed: Stage: none for these lengths; 4 ms attribution misassigned. Reviewer:
  source reading, arithmetic from raw JSON.
  Resolution: more-work-needed (P1).

- Observed anomaly: Repeated prefill differs in one to three 32-row blocks.
  Evidence: `final_evidence10/prefill_linear_{1024,2048,3072}`, `control_grid_linear_2048`;
  `review2/dispatch_dtype/full_1000/dispatch_bfp8.json` (batched path);
  `review2/bfp8_grid/full_1024/small4x4_large11x8.json` (0.0078 maximum absolute difference).
  Repeat PCC 0.9999998 to 1.0 in all 16 non-identical runs of 238 scanned.
  Affected path: MoE combine after `moe_fused_swiglu`.
  Control or comparison: back-to-back reduce calls agree; standalone DRAM-weight repro is
  deterministic; 4096 single-chunk runs identical in every chain.
  Likely subsystem: unresolved; device state or an ordering hazard before the reduce.
  Investigation performed: E1, E2, E5, E6, E3-lite (`review2/drift/`).
  Resolution: controlled by disclosure with bounded magnitude, per the review-2 requirement.
  Documentation figures still wrong (P2).

- Observed anomaly: Real-activation B32 positions, linear row 17 token 4 at 0.99395.
  Evidence: `final_evidence10/real_inputs/positions_0.json` (`failed_gate`; chain step rc=1).
  Affected path: GDN layer 0 decode at B32.
  Control or comparison: CPU FP32 reference (`review2/ref_precision/`): BF16 reference 0.99484 at
  that token.
  Likely subsystem: BF16 reference sensitivity.
  Investigation performed: reference-precision control.
  Resolution: controlled.

- Observed anomaly: Linear prefill token 1720 at 0.99402 (TT-side) and full token 911 at 0.99136.
  Evidence: `final_evidence10/perf_*/default.json` `prefill_token_pcc`;
  `review2/prefill_ref/prefill_token_ref_l0_4096.json` (reference 0.99942 at 1720; BF16 floor
  0.99323).
  Affected path: prefill, both layer kinds.
  Control or comparison: FP32 reference; full token 911 is reference noise (0.99156).
  Likely subsystem: BF16 rounding in prefill projections or MoE.
  Investigation performed: FP32 reference control.
  Resolution: controlled for gating; rule wording inconsistent (Other Concerns).

- Observed anomaly: Device 0 trace replay 16 to 21 us slower.
  Evidence: `final_evidence10/device_ab`, `final_evidence8/device_ab`.
  Affected path: host-side trace replay.
  Control or comparison: devices 1 to 3 with the same source.
  Likely subsystem: unverified (host or PCIe, peripheral component interconnect express, path).
  Investigation performed: A/B only.
  Resolution: controlled for this stage (headline on device 3); carry to multi-chip stages.

## Resolution Of Review-2 Findings

- P1 aligned ladder: resolved as specified. Aligned inputs now use `_planner_buckets`. Linear
  1024 / 2048 / 3072: 49168 / 98705 / 148815 us before, 12948 / 19358 / 32372 us now. Continuation
  (2048 after 128) 19233 us linear, 15838 us full, matching fresh 2048. Small-prefill profile and
  attribution exist (intermediate source). A new planner cost defect remains (P1 above).
- P2 nondeterminism: resolved per the required next step (magnitudes in every warmed run, three
  failing cases rerun, E1/E2/E5 and E3-lite, explicit disclosure). Contract numbers need
  correction (P2 above).
- P2 inventory: resolved. Root-buffer census for both layer kinds, per-layer bytes in README and
  contract, handoff recheck in `final_evidence10/memory_handoff_recheck.json` (handoff hashes
  still old, recheck states `hash_match: false`), 40-layer estimate recorded. Two unused copies
  released (16 MiB linear, 19.1 MiB full). Slot label error (P2 above).
- P2 documents: mostly resolved (single final pointer, 8x4/8x8/11x8 rows, reference control
  recorded). New inconsistencies listed above.

## Re-derived Claims (match README unless noted)

- Headline: linear 709.84 us decode median (128 replays) and 33050.8 us prefill; full 434.41 us
  and 26343.9 us. Device spans 810.6 / 32750.5 us (linear), 482.3 / 26076.3 us (full). Roofline
  171.3 / 107.8 us, decode DRAM 21.13 / 22.36 percent.
- Prefill PCC 0.99971 / 0.99952; minimum decode PCC 0.99874 / 0.99835; one token below 0.995 in
  each 4096-token prefill.
- Positions random B32: linear 0.99923 / 0.99770, full 0.99855 / 0.99594. Reuse 17 cases: linear
  0.99900 / 0.99821, full 0.99852 / 0.99739, page 128 passes. Recorded-activation reuse: linear
  0.99904 / 0.99892, full 0.99657 / 0.99634.
- Context: 262143 prefill and decode, 262144 prefill, both layer kinds pass (0.99939 / 0.99988,
  0.99955; 0.99848 / 0.99837, 0.99835).
- Watcher: 10 logs, 0 matches for error, assert, overflow, hang, corrupt or invalid.
- pytest: 8 passed with trace allocation tracking and program-cache allocations included.
- Dispatch maps: 27 of 27 pass at 32, 64, 128, 256, 512, 1024, 2048 and 4096 tokens.
- Grid control: selected 8x8 for groups of 2048 and above beats 4x4 by 0.23 to 1.93 ms.
- Profile rows: `moe_fused_swiglu` BF16 x BFP8 on 64 cores (110 available).
- Chain: one non-zero step (`real_positions_0`, the controlled diagnostic); `SOURCE_STABLE`.

## Scope Inspected

- Goal/skill paths: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/03-03-optimized-decoder.prompt.txt`;
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`;
  `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`;
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/stage_review_2.md`.
- Artifact paths (under
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/`):
  `optimized_decoder/{README.md,work_log.md,checklist.md,local_dispatch_diagnosis.md}`,
  `context_contract.json`, `optimized_decoder/final_evidence10/**` (summary, manifests, snapshot,
  perf, prefill, control_grid, device_ab, positions, reuse, real_inputs, context, watcher,
  watcher_long, pytest, dispatch maps, inventory, memory recheck, profile stacked CSV and provenance),
  `optimized_decoder/final_evidence10_chain.log`, lane logs,
  `optimized_decoder/review1/route_timing/*`,
  `optimized_decoder/review2/{smallprof,single_tail,dispatch_dtype,prefill_ref,large_grid,bfp8_grid,drift/logs}`,
  `optimized_decoder/{c5v_full,c5abl_full}`, `fused_decoder/README.md`;
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_claude/evidence_chain10.sh`;
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/telemetry/packets/9ba51029-cb68-4989-ae12-8f9ed3691bdd.json`.
- Code paths: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py`
  (policies, `prefill_forward`, `_chunk_plan`, `_attention_step`, `_batched_prefill`,
  `_planned_prefill`, `from_state_dict`, `_OptimizedGDN.forward`, `_LocalDispatch.__call__`,
  `_bucket_sizes`); `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/functional_decoder.py`
  (`_index_offsets`, `_indices`); `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/{run_layer.py,test_optimized_decoder_device.py,run_reuse.py,run_positions.py}`;
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/{unified_routed_expert_ffn,hybrid_routed_expert_ffn}` (bindings, validation).
- Commands run: `sha256sum` and a Python manifest check against live files; `git rev-parse`,
  `git status`, `git diff | sha256sum`; snapshot extraction to the scratchpad; Python scripts over
  result JSON, inventory JSON and perf-report CSV; `grep` over logs, watcher logs, docs and source.
  No device, server or hardware job.

## Residual Risk

- Decode is unchanged since review 2 and stays the accepted result.
- Fixing P1 changes the prefill planner again. Reuse, positions, context, watcher, pytest and
  prefill timing evidence must be regenerated on a new frozen source.
- The 40-layer footprint (59.76 GiB) is about 1.9 times one chip's DRAM. The p150 profile depends
  on later expert placement and precision decisions.
- Prefill is not bitwise reproducible; root cause open.
