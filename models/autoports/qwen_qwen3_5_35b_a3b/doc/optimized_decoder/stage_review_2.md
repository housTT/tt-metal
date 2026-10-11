# Stage Review

Stage 03, optimized decoder (single-device layer optimization), re-review after
`stage_review.md` (verdict more-work-needed, 1 P1 and 7 P2). Target `Qwen/Qwen3.5-35B-A3B`
revision `59d61f3ce65a6d9863b86d2e96597125219dc754`, one Blackhole ASIC (1x1 mesh).
Reviewer: fresh independent subagent, 2026 Oct 10. Live worktree at HEAD `1d857aa3` plus
uncommitted stage-3 changes. Reviewer recheck: all 37 entries of
`final_evidence6/source_manifest_end.json` match the live files (SHA-256), start and end
manifests are equal, `git_head` matches, and the diff outside the autoport is empty
(SHA-256 of the empty diff).

Abbreviations: PCC (Pearson correlation coefficient), GDN (Gated DeltaNet, the
linear-attention layer), MoE (mixture of experts), KV (key/value cache), HF (Hugging Face
reference), BF16 (bfloat16), FP32 (32-bit float), BFP8/BFP4 (8-bit/4-bit block floating
point), SDPA (scaled dot-product attention), ULP (unit in the last place), DRAM (device
main memory), L1 (per-core SRAM), CPU (host processor), us/ms (micro/milliseconds),
MiB/GiB (2^20/2^30 bytes), TTFT (time to first token).

Verdict: more-work-needed

## Required Work

- P1: Aligned prefill lengths bypass the new bucket ladder and run as 128-token chunks
  Evidence:
  - `tt/optimized_decoder.py` `OptimizedDecoder.prefill_forward` sends every input with
    `logical % 128 == 0` and `start_pos % 128 == 0` to the tiled path. That path builds its
    own bucket set `{512,128}` (full) or `{128}` (linear) plus `prefill_chunk` (4096). It
    never reads `prefill_buckets` (`[2048, 1024, 256, 64]`). Only `_planned_prefill` (via
    `_planner_buckets`/`_bucket_sizes`) uses the ladder. So a 2048-token linear prompt runs
    16 chunks of 128 tokens, and a 2048-token full prompt runs 4 chunks of 512 tokens.
  - Measured (stage-owner runs, recorded HF activations, warmed host prefill, linear
    layer 0, source `8fb3221b`; the tiled-path code is unchanged in the final source,
    reviewer diff of `review1/single_linear_2048/source_snapshot.tar.gz`):
    `review1/single_linear_1024` 49168/48048 us, `single_linear_2048` 98705/98305 us,
    `mix_linear_3072` 148815/148937 us, `mix_linear_2304` 110832 us. The same source
    prefills 4000 tokens through the ladder in 64023 us (`mix_linear_4000`), and the final
    source prefills 4096 tokens in 33991 us (`final_evidence6/perf_linear`). A 2048-token
    prompt is therefore about 2.9x slower than a 4096-token prompt, and 3072 tokens is
    about 4.4x slower.
  - `doc/context_contract.json` (`optimized_decoder_validation.prefill_chunk_buckets`) and
    README ("Chunk planner: aligned buckets 4096/2048/1024/512(full)/256/128/64/32") state
    the ladder applies. For aligned inputs this is false. None of these runs is cited in
    README, work log or checklist. `final_evidence6` measures no aligned length other than
    4096 (lengths 100, 1000, 4000, 4095, 4096, 4097).
  - Related, unexplained: per-chunk cost is large and non-monotonic in length. Final
    source: 1000 tokens take 33054 us (full) and 44149 us (linear), more than 4096 tokens
    (27281 / 33991 us). 100 tokens take 11456 / 13893 us. `single_linear_64` takes about
    6.1 to 6.6 ms for 64 tokens. One extra token-wise tail step costs about 0.93 ms (full)
    and 1.48 ms (linear) (4097 minus 4096). No profile exists for a ladder or small-chunk
    prefill, so the fixed per-chunk cost is not attributed.
  Why this matters: The bucket ladder is new stage-3 code and is documented as the chunk
  contract for both layer kinds. Common aligned prompt and chunked-prefill sizes (1024,
  2048, 3072) take several times longer than a 4096-token prompt. That directly sets TTFT
  in the later serving and latency-sweep stages, and the stage claims no deferred decoder
  optimization.
  Required next step: Make the aligned path use the same bucket ladder as the planner (or
  route aligned non-4096-multiple lengths through it), keeping trace/program-cache rules.
  Measure warmed prefill on the final source for aligned 1024, 2048 and 3072 and for an
  aligned continuation (`start_pos` 128) for both layer kinds, and rerun the affected
  correctness gates (reuse and positions exercise the changed planner). Profile one
  small-chunk or ladder prefill (for example 1000 tokens) with `tt-perf-report`, attribute
  the fixed per-chunk cost, and reduce it or record the exact blocker. Correct README and
  `context_contract.json`.

- P2: Prefill bitwise nondeterminism is classified from probes, not from the final failing runs
  Evidence:
  - `final_evidence6` itself records `prefill_repeat_identical: false` in 3 of 10
    unaligned runs (`unaligned_full_1000`, `unaligned_linear_1000`,
    `unaligned_linear_4095`). Earlier: `review1/flake_full_4095` 7 of 8 false,
    `flake_linear_4095` 5 of 8 false, `tail2_*_fill_ladder` 3 of 4 false at 4000/4095.
    Every 4096-token aligned run in all chains reports true. The failures correlate with the
    multi-size planned path (ladder plus tails), which is new stage-3 code.
  - `run_layer.py` stores only a boolean. The README and work log state the difference is
    "1-2 BF16 ULP in one 32-token block of the MoE combine output". That localization comes
    from `runs/main/stage03_claude/micro/prefill_probe.py` runs (`flake/PD_*`, `PS_*`,
    `TP_*`, `NT_*`) with probe-specific settings, not from the final failing runs. No
    magnitude, row set or layer-output PCC delta is recorded for them.
  - `review1/post_combine_autodebug/AUTODEBUG.md` finds no source defect and lists five
    discriminating experiments (E1 host diff of saved outputs, E2 L1-placed weights in the
    standalone repro, E3 bisect of preceding ops, E4 config-word readback, E5 back-to-back
    calls in model context). The work log reports none of them. The standalone repro still
    uses DRAM weights (`pcr_repro.py`), which AutoDebug section 7.3 identifies as not
    matching the model placement.
  - The k-major tile-combine option (-0.5 ms prefill) is rejected only because it shows
    "more combine drift", which uses this unexplained effect as a veto.
  Why this matters: Inputs that are bitwise identical give different reduction outputs.
  That points to state leaking between kernels or a placement-dependent defect, not to
  ordinary rounding. Repeated identical requests now produce different KV and recurrent
  state, so later full-model or serving determinism checks inherit an unexplained flake.
  The classification "rounding-level, PCC unaffected" is asserted for runs where it was
  not measured.
  Required next step: Record, for every repeated prefill, max absolute difference, number
  and positions of differing rows, and the PCC between the two runs, and rerun the three
  failing `final_evidence6` cases to show the drift matches the classification. Run the
  cheap AutoDebug experiments (at least E1, E2 and E5; E3 if E5 implicates preceding ops)
  and record the result. If the cause is still unresolved, state explicitly in README and
  `context_contract.json` that prefill is not bitwise reproducible for multi-chunk inputs,
  and do not use "more drift" as the sole veto for the tile-combine candidate.

- P2: Weight ownership inventory and memory-handoff recheck are still missing
  Evidence:
  - Stage review 1 required an ownership inventory (live root buffer, consumer, bytes),
    per-layer bytes in README and `doc/context_contract.json`, and a recheck of the
    memory-layout handoff. The response fixed same-dtype aliasing (`_as_dtype`) and reports
    allocator totals only (`final_evidence6/perf_*/default.json` `weight_memory`: linear
    1208.9 MiB, full 1170.5 MiB after cleanup; 3033.0 / 2994.6 MiB before cleanup).
    `context_contract.json` records no weight bytes. `memory-layout-handoff.json`
    `stage_3_*` sections still bind older `optimized_decoder.py` hashes
    (`321c9328...`, `67e16210...`), not the final `998a13e7...`.
  - Reviewer source reading (unverified on device): the default keeps two routed
    representations per layer. Sparse decode keeps the inherited packed weights (gate/up
    BFP4 about 288 MiB, down BFP8 about 272 MiB, aliased through `_as_dtype`). The
    prefill dispatch keeps separate per-expert BFP4 gate/up/down lists (`_LocalDispatch`,
    `ttnn.typecast` of the compact lists, about 432 MiB). The measured totals also include
    32 eagerly configured FP32 GDN state slots (2 MiB each, about 64 MiB per linear layer),
    which the handoff asks to record separately from active batch.
  - Arithmetic from the measured allocator values: 30 x 1208.9 MiB + 10 x 1170.5 MiB =
    46.8 GiB for the 40 decoder layers alone. The allocator reports 31.8 GiB total DRAM
    per chip (allocated plus free).
  Why this matters: AGENTS.md requires allocator ownership validation, removal or sharing
  of duplicated representations, and a qualified p150 single-chip profile. The stage
  introduced the dual representation; without an inventory the later stages cannot decide
  what to share or shard.
  Required next step: Produce the inventory for both selected defaults (each live device
  buffer, its consumer, bytes; configured GDN slots separate). Record per-layer bytes in
  README and `context_contract.json`. Recheck the memory-layout handoff against the final
  source hash. Note in the contract that the selected per-layer footprint does not fit 40
  layers on one chip, so the later profile work starts from measured numbers.

- P2: Stage documents are still stale or contradictory
  Evidence: `work_log.md` lines 243-246 name `final_evidence4/` as the final frozen
  evidence; line 331 names `final_evidence6/`. Line 239 still says `final_evidence2/`.
  The "duplicate weights" and "8x4" items cite `final_evidence4`. README and
  `context_contract.json` describe the ladder for aligned inputs (see P1). The README
  rejected-options table omits the 8x4 dispatch grid (stage review 1 asked for it). The
  real-activation B32 positions failure is still labeled "Stage-2 fused also fails"
  without a reference-precision control (see Anomaly Ledger; the reviewer control below
  closes it but the stage evidence does not record it).
  Why this matters: The goal requires README and work log to record exact artifacts and
  rejected options; AGENTS.md requires failed and superseded attempts to be explained.
  Required next step: Make the response section consistent with `final_evidence6`, add the
  8x4 dispatch row (re-measured, see Other Concerns), and record the reference-precision
  control for the real B32 positions diagnostic.

## Other Concerns

- 8x4 dispatch grid. `moe_fused_swiglu` is the largest prefill op (8681 us linear, 8667 us
  full, 16 of 110 cores; `final_evidence6/profile/tracy/*/prefill_perf_report.csv`). The
  8x4 grid was faster on the full layer (`rank_full/cumsum_8x4` 37734 versus 38321 us,
  older source) and was rejected only for a decode-identity failure. That failure is now
  classified (watcher-clean, deterministic, `final_evidence6/watcher_long/diag_8x4.json`;
  plausibly the uninitialized padding that `decode_input_fill` now fixes). Its speed was
  never re-measured on the final source. Re-measure 8x4 (and any larger legal grid) as part
  of P1 rework.
- Remaining prefill movement: one-core UINT32 scatter (1613 us) and UINT32 slot reshape
  (1663 to 1664 us) per 4096-token chunk, plus two BF16 q/k head-split reshapes (1539 and
  1536 us) in GDN prefill. The scatter has no measured alternative or recorded blocker.
  The flat q/k alternative was rejected on real activations (reviewer control below shows
  the HF reference is clean at that token, so the rejection is earned); an alternative
  head-split implementation that keeps external normalization was not tried.
- `_PREFIX_MATRICES` is a module-level cache keyed by `id(mesh)`. A process that closes a
  mesh and opens another can reuse the id and get tensors on a closed device. pytest
  passes now; this is a latent hazard for multi-mesh processes in later stages.
- Dispatch-map exactness (`dispatch_maps_matmul.json`) covers 32, 128, 512, 1024, 2048 and
  4096 tokens. The 64 and 256 buckets are covered only by PCC runs (100 and 1000 tokens).
- Full-attention chunks below the page size fall back to token-wise steps (page size 128
  allows no 32/64 bucket). Correct (`reuse_3/full_page128.json` passes), but slow; the
  serving page size is not yet fixed.
- `tests/run_layer.py` still keys position refresh on `args.layer == 3`, not on layer
  type (unchanged since review 1).

## Hard-Check Gaps

- No aligned non-4096-multiple prefill timing or profile in the final evidence.
- `prefill_repeat_identical` is recorded but not asserted and stores no magnitude.
- Real-activation B32 positions remains a non-gating diagnostic; it should gate against
  an FP32 reference or a tolerance derived from the BF16-versus-FP32 reference gap.
- pytest (`test_optimized_decoder_device.py`) covers 129/8 only.

## Anomaly Ledger

- Observed anomaly: Real-activation B32 positions fails for the default in every window.
  Evidence: `final_evidence6/real_inputs/positions_0.json` (row 17, start 813, token 4,
  0.99395); `review1/p1/default_realpos_517.json` (row 7, token 3, 0.99135);
  `default_realpos_1531.json` (row 1, token 3, 0.99036). Stage 2 fails identically.
  Affected path: GDN layer 0 decode at B32.
  Control or comparison: Reviewer CPU-only control (no device). HF layer 0 in BF16 versus
  the same layer in FP32, same windows and tokens (script
  `/tmp/claude-2005/-home-hous-dev-qwen3-5-35b-a3b/5bead60f-0645-4ca5-b27d-6b0f6acbc38f/scratchpad/ref_precision.py`):
  the BF16 reference itself has PCC 0.99484, 0.99188 and 0.99033 against FP32 at exactly
  these tokens, and above 0.9992 at every other token. TT versus the BF16 reference is
  0.99395, 0.99135 and 0.99036 at the same tokens.
  Likely subsystem: BF16 reference sensitivity (routing or near-tie behavior in the BF16
  reference), not the TT implementation.
  Investigation performed: Stage-2 control (stage owner); FP32 reference control (reviewer).
  Resolution: controlled. The stage evidence must record it (P2 documents).

- Observed anomaly: Many perturbations fail only at decode token 32 of the 4096/128 fixture.
  Evidence: `review1/p1/perf/*` (BFP8 in 0.9897-0.9898, BFP8 LoFi out 0.9888),
  `review1/flatqk/perf128/*` (0.9899-0.9901); default 0.99874 (minimum at token 21).
  Affected path: GDN layer 0.
  Control or comparison: Reviewer CPU-only control (script `ref_precision_b1.py` in the same
  scratchpad): HF BF16 versus FP32 PCC at token 32 is 0.999994; minimum over 128 tokens is
  0.99876 at token 18. The reference is clean at token 32, so these are TT-side errors.
  Likely subsystem: Reduced-precision GDN projections and prefill normalization.
  Investigation performed: Stage-owner token-energy correlation (partial); reviewer FP32
  reference control.
  Resolution: controlled for gating purposes. The rejections based on it are earned.

- Observed anomaly: Repeated prefill differs bitwise for multi-chunk inputs.
  Evidence: see P2 nondeterminism.
  Affected path: MoE combine reduction after the device dispatch (per probes).
  Control or comparison: Standalone repros deterministic (DRAM weights only); 4096 aligned
  runs always identical.
  Likely subsystem: Unknown; AutoDebug suggests state from preceding kernels or platform.
  Investigation performed: Probe localization, two reduction kernels, sync and neutral-op
  controls, AutoDebug source audit.
  Resolution: more-work-needed.

- Observed anomaly: Aligned 1024/2048/3072-token prefill is 1.4x to 4.4x slower than 4096.
  Evidence: see P1.
  Affected path: `OptimizedDecoder.prefill_forward` tiled path.
  Control or comparison: Ladder path (4000 tokens, 64023 us) and 4096 single chunk.
  Likely subsystem: Bucket selection in the tiled path; fixed per-chunk cost.
  Investigation performed: None by the stage; reviewer source reading.
  Resolution: more-work-needed.

- Observed anomaly: 8x4 dispatch decode-identity failure (review 1).
  Evidence: `rank_linear/cumsum_8x4.log`; final rerun `final_evidence6/watcher_long/diag_8x4.json`
  (deterministic, watcher-clean).
  Affected path: Prefill dispatch grid.
  Control or comparison: 4x4 default; same final source.
  Likely subsystem: Uninitialized tile padding fixed by `decode_input_fill` (plausible,
  not proven).
  Investigation performed: Watcher rerun on final source.
  Resolution: controlled; speed re-measurement still owed (Other Concerns).

## Resolution Of Review-1 Findings

- P1 BFP8 GDN projections: resolved. Clean single-variable rows on the final base with
  recorded activations (`review1/p1/`). Reviewer re-derived from raw JSON: default 727.69
  us, BFP8 LoFi in 672.69 us (min 0.98980 at token 32), BFP8 HiFi2 in 673.05 us (0.98974),
  BFP8 LoFi out 709.54 us (0.98881), BFP8 HiFi2 out 709.64 us (0.99874), both HiFi2
  655.09 us (0.98965). One rule applies to all rows (4096/128 real activations). BFP8 HiFi2
  out adopted; final 709.99 us reproduces it.
- P2 token sensitivity / real B32: controlled by the reviewer FP32 reference control
  above; stage documentation still owed.
- P2 duplicate weights: partially resolved (aliasing fixed, 1829 to 1170 MiB full); the
  inventory, contract record and handoff recheck remain (P2 above).
- P2 nondeterminism: 8x4 classified, watcher runs added at 4096 and 4095 (0 errors in 9
  logs, reviewer grep), repeated-prefill check added; new prefill drift unresolved (P2).
- P2 prefill overhead: dead cumsum removed (matmul counts and region starts), shared-down
  2D prefill config (-0.7 to -0.9 ms), `minimal_matmul` in-projections (full -0.86 ms,
  linear -0.2 ms) and full out-projection (-0.4 ms), all re-derived from `review1/reruns/*`.
  Scatter and slot reshape remain (Other Concerns).
- P2 unaligned prefill: tails reduced to 32-token granularity and measured; new defects
  found (P1).
- P2 decode geometry: resolved under frozen source (`review1/reruns/linear`: DRAM-sharded
  in-projection 743-785 us versus 730.7 us; residual16 454.5/448.5 versus 446.5 us;
  `gdn_state_l1` 725.8 versus 730.7 us, rejected for per-slot L1 capacity).
- P2 stale documents: partially resolved (P2 above).

## Scope Inspected

- Goal/skill paths: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/03-03-optimized-decoder.prompt.txt`;
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`;
  `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`; `/home/hous/dev/qwen3.5-35b-a3b/memory-layout-handoff.json`.
- Artifact paths (under `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/`):
  `optimized_decoder/{README.md,work_log.md,checklist.md,stage_review.md}`,
  `context_contract.json`, `optimized_decoder/final_evidence6/**` (summary, perf, unaligned,
  profile tracy CSV, positions, reuse, watcher and watcher_long logs, real_inputs, context,
  dispatch maps, manifests), `optimized_decoder/final_evidence6_chain.log`,
  `optimized_decoder/final_evidence{4,5}/**` (repeat-identity scan),
  `optimized_decoder/review1/{p1,reruns,flatqk,tail_*,tail2_*,single_linear_*,mix_linear_*,flake_*,combine_linear,post_combine_autodebug/AUTODEBUG.md}`,
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_claude/micro/{prefill_probe.py,pcr_repro.py,fred_repro.py,token_outlier.out,flake/PD_*}`,
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_claude/evidence_chain6.sh`.
- Code paths: `tt/optimized_decoder.py` (policies, `prefill_forward`, `_planned_prefill`,
  `from_state_dict` release logic, `_Projection`, `_OptimizedGDN.forward`/`_convolve`,
  `_OptimizedAttention.forward`, `_LocalDispatch`, `_ZeroPaddedDecodeLayer`, `_as_dtype`),
  `tt/fused_decoder.py` (weight construction), `tt/functional_decoder.py` (expert loading),
  `tests/run_layer.py`, `tests/run_positions.py`, `tests/hf_reference.py`.
- Commands run: read-only `cat`/`grep`/`git status`/`git diff`; SHA-256 manifest recheck;
  Python analysis of result JSON and perf-report CSV; snapshot diff of
  `review1/single_linear_2048`; two CPU-only HF scripts (layer 0, BF16 versus FP32
  reference, B32 windows and the 4096/128 fixture). No device, server or hardware job.

## Residual Risk

- Headline numbers re-derived from raw JSON match the README: linear 709.991 us decode
  median and 33990.997 us prefill; full 434.7935 us and 27281.294 us; speedups 1.77x/2.12x
  decode and 22.5x/25.6x prefill; stack 25.65 ms per token; roofline 171.3/107.8 us.
  Perf-report rows show the claimed dtypes and fidelities (GDN in BF16 HiFi4 at 128 us,
  QKVG BF16xBFP8 LoFi at 55 us).
- Trace evidence uses `TT_METAL_TRACE_ALLOC_TRACKING=1` with program-cache allocations
  included and acknowledges only the final trace output.
- Context contract remains 262144 with passing 262143/262144 runs on the final source.
- Fixing P1 changes the prefill planner. Reuse, positions, context, watcher and unaligned
  timing evidence must be regenerated on the new frozen source.
