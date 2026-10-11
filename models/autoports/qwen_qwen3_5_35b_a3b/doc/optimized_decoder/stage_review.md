# Stage Review

Stage 03, optimized decoder (single-device layer optimization), Qwen/Qwen3.5-35B-A3B revision
59d61f3ce65a6d9863b86d2e96597125219dc754. Reviewer: fresh independent subagent, 2026 Oct 10.
Reviewed live worktree at HEAD 1d857aa3 plus uncommitted stage-3 changes. Current `tt/` and
`tests/` bytes match `final_evidence3/source_manifest_end.json` (reviewer recheck), and the
diff outside the autoport is empty.

Abbreviations: PCC (Pearson correlation coefficient), GDN (Gated DeltaNet, the linear-attention
layer), MoE (mixture of experts), KV (key/value cache), BF16 (bfloat16), FP32 (32-bit float),
BFP8/BFP4 (8-bit/4-bit block floating point), HF (Hugging Face reference), SDPA (scaled
dot-product attention), TTFT (time to first token), DRAM (device main memory), L1 (per-core
SRAM), CPU (host processor), us (microseconds), MiB/GiB (2^20/2^30 bytes).

Verdict: more-work-needed

## Required Work

- P1: BFP8 GDN projection rejections are not earned under the final contract
  Evidence:
  - The README rejects BFP8 GDN in-projection on random-activation B32 positions (0.9932),
    random-activation 1153-token continuation reuse (0.9948) and real-activation B32 positions
    (0.9896). The selected default fails the same real-activation B32 positions gate
    (`final_evidence3/real_inputs/positions_0.json`, status `failed_gate`, 0.99188, two
    single-token dips: row 17 token 4, row 27 token 2). The BFP8 candidate also has exactly two
    single-token dips (`accept_ablation/c8_real_linear_positions.json`: row 17 token 4 0.9942,
    row 24 token 6 0.9896). The real gate cannot veto the candidate while it is only a
    diagnostic for the default.
  - On real activations the BFP8 in-projection candidate passes real reuse at the same minimum
    as the default (`accept_ablation/c8_real_linear_reuse.json` decode min 0.9989 versus
    `final_evidence3/real_inputs/reuse_0.json` 0.9989). The remaining blockers are
    random-activation (synthetic) tests. OPT-012 and the stage contract ("synthetic/random PCC
    cannot veto real-weight wins") forbid that veto without debugging the discrepancy.
  - The clean candidate "final linear default + BFP8 LoFi in-projection" was never measured at
    4096/128. Every 4096/128 BFP8 in-projection row also enabled the sharded decode norm without
    `decode_norm_fill` (for example `final_linear/default.json`, 625.3 us median, min PCC 0.9969,
    passes; `pcc32_linear/*`, `aofid_linear/*`). The work log itself shows that configuration
    reads uninitialized tile padding and is nondeterministic.
  - The BFP8 GDN output-projection rejection ("token 32 PCC 0.989-0.990", `pcc32_linear`,
    `aofid_linear`) comes from rows that all ran the unfilled sharded norm, except
    `pcc32_linear/norm_off.json`, which also changed the in-projection to BFP8 and the output to
    a DRAM-sharded r2 config. No clean single-variable row exists.
  - Expected value: in-projection about 55 us per GDN layer (README) plus output projection
    (BF16 HiFi4 42.5 us versus BFP8 LoFi 24.8 us for the same-shape full-attention row in
    `profile/tracy/*/decode_perf_report.csv`). Across 30 GDN layers that is up to about
    2 ms per token on a 26.08 ms decoder stack.
  Why this matters: The final linear-attention policy may keep a slower BF16 path on synthetic
  or confounded evidence. That is the exact pattern the stage-review skill and OPT-012 classify
  as required work.
  Required next step: Under frozen source, measure the final linear default with (a) BFP8 LoFi
  in-projection, (b) BFP8 output projection, (c) both. Use recorded activations at 4096/128,
  real B32 positions and real reuse, plus the random-activation gates as diagnostics. Apply one
  consistent acceptance rule to the default and to the candidates. If a random-activation
  failure remains the only blocker, debug the discrepancy per OPT-012 instead of using it as a
  veto. Update DEFAULT_POLICIES, README and work log with the result.

- P2: Token-level PCC sensitivity and the inherited real B32 positions failure are unexplained
  Evidence: At least nine independent linear-layer perturbations fail only at decode token 32 of
  the layer-0 4096/128 fixture (0.9886 to 0.9901: `pcc32_linear/*`, `aofid_linear/*`,
  `c10_linear/norm_fill.json` 0.9899, `c11_linear/norm_fp32_fill.json` 0.9898). The default
  passes that token at 0.9988. The real-activation B32 positions run fails for the default and
  for Stage 2 at the same rows and tokens. The README says "Its cause is not isolated
  (unverified)". Reviewer CPU-only check (HF layer 0 on CPU, script in the reviewer scratchpad,
  no device): decode token 32 is not a router near-tie. Its relative top-8 to top-9 probability
  gap is 2.8%, rank 55 of 128 decode tokens; the smallest gaps are at tokens 83, 49, 120, 8.
  The routing-flip explanation is therefore not supported for this token. Token 32 output
  standard deviation (0.0231) is near the median (0.0204).
  Why this matters: This sensitivity decides several precision rejections (sharded norm on GDN
  layers, BFP8 GDN output projection) and is the reason a failing real gate is labeled
  "diagnostic". Prose acknowledgment is not resolution.
  Required next step: Classify the anomaly with evidence. Suggested controls: HF BF16 versus
  HF FP32 reference PCC at the failing tokens (is the reference itself noise-limited?),
  per-channel TT-minus-HF error at token 32, and GDN state/output comparison at the failing
  decode positions. Then either fix the cause, or record a control that shows the failures are
  expected and apply the same rule to all candidates (see P1).

- P2: Persistent duplicated weights survive `release_unused_weights`
  Evidence: `tt/optimized_decoder.py` keeps `"weights"` in the `_OptimizedPrefillExperts` live
  set unconditionally. With `prefill_backend="dispatch"` no runtime path reads
  `experts.weights` (only the `sparse` and `packed` prefill backends do). Sparse decode reads
  its own `gate_up_weight`/`down_weight`, which `prepare` builds with same-dtype
  `ttnn.typecast` (BFP4 to BFP4, BFP8 to BFP8). `memory-layout-handoff.json`
  (`stage_3_indexed_sparse_recheck`) confirms that same-dtype typecast allocates new buffers
  (560 MiB per layer). `_Projection` also typecasts attention weights to the same dtype while
  the originals (`qkvg_weight`, `o_proj`, `mega_weight`, `output_weight`) stay referenced.
  Measured after cleanup (`final_evidence3/perf_*/default.json`, `weight_memory`): 1744 MiB
  (full) and 1811 MiB (linear) DRAM per layer. Reviewer estimate (unverified, from tile-aware
  sizes): sparse copies 560 MiB + dispatch BFP4 lists 432 MiB + dead base expert weights
  560 MiB + attention/shared/router/maps about 80 to 130 MiB, which matches the measurement.
  Without the dead copies the routed payload would be about 992 MiB per layer. Over 40 layers
  the dead base expert copy alone is about 22 GiB. The README and context contract do not
  report per-layer weight memory.
  Why this matters: AGENTS.md requires allocator-ownership validation and removal of unused
  duplicated representations before composition, and all three profiles (p150, p150x2,
  p150x4) must later fit. This stage introduced the dual routed representation and the
  same-dtype copies.
  Required next step: Write an ownership inventory for the selected default (each live root
  buffer, consumer, bytes). Drop or share every unused copy (for example share base tensors
  with sparse decode instead of same-dtype typecast, and drop `experts.weights` for the
  dispatch backend). Re-measure `weight_memory`, rerun the affected correctness gates, record
  per-layer bytes in README and `doc/context_contract.json`, and recheck the memory-layout
  handoff as AGENTS.md requires.

- P2: Unclassified nondeterminism signal in the routed prefill kernel family
  Evidence: `rank_linear/cumsum_8x4.log` fails `assert deterministic` in `tests/run_layer.py`
  (bitwise decode-replay identity after state restore) when only `dispatch_grid` changes from
  4x4 to 8x4. The same `moe_fused_swiglu` kernel runs in the default at 4x4. The work log
  records this only as "failed (assertion)". The README rejected-options table omits it. No
  test compares repeated prefill outputs bitwise (`run_layer.py`, `run_positions.py`,
  `run_reuse.py` check decode identity only). All watcher runs use lengths 65, 129 and 257,
  so the 512 and 4096 dispatch buckets, the 4096x4096 prefix-rank matmul and the
  40704-row capacity path never ran under watcher.
  Why this matters: A configuration-dependent race or out-of-bounds write in a kernel that is
  in the default path is a correctness risk. 8x4 was also faster for the full layer
  (37734 versus 38321 us, `rank_full`).
  Required next step: Classify the 8x4 failure (use `$autodebug`/`$autofix` if needed). Add a
  repeated-prefill bitwise identity check and a `TT_METAL_WATCHER=10` run for the selected
  default at a length that uses the 4096 bucket, with a hot-expert routing case. Record the
  outcome in the rejected-options table.

- P2: Measured prefill overhead and actionable perf-report advice left untried
  Evidence (full layer, `final_evidence3/profile/tracy/full_attention/ops.csv`, prefill window,
  29.4 ms device):
  - Dead code: `raw_start = cumsum(count) - count` in `_LocalDispatch.route_maps` is unused on
    the `matmul` rank path but still executes. Each 256-element cumsum runs on one core for
    293.7 us (two calls, 587 us).
  - `ttnn.reshape` of the UINT32 row-major slot map [4096,8] to [32768]: 1665 us.
  - One-core UINT32 `ttnn.scatter` building the dispatch map: 1613 us.
  - Shared-expert prefill down projection 4096x512x2048: 889 us at 1.6% FLOP use. The decode
    path passes `ttnn.L1_MEMORY_CONFIG` for a 16 MiB prefill output and the auto config picks
    `in0_block_w=1` (perf-report advice "in0_block_w=1 is small").
  - No prefill program-config sweep exists for the attention projections. `_Projection`
    supports `prefill_grid`/`prefill_k`, but no matrix uses them. QKVG prefill is 1341 us at
    18.9% FLOPs; GDN in-projection prefill is 1991 us.
  - `moe_fused_swiglu` runs on 16 of 110 cores for 8.67 ms (about 30% of prefill).
  The README names "gather, scatter and layout ops around the dispatch remain" but records no
  attempt.
  Why this matters: The optimize skill requires trying applicable `tt-perf-report` advice and a
  tuned 2D or `minimal_matmul` comparison for large prefill matmuls. About 5 ms of the 29.4 ms
  full-layer prefill sits in these rows.
  Required next step: Remove the dead cumsum and replace the remaining 256-element cumsum (for
  example with a small triangular matmul). Remove the UINT32 reshape and one-core scatter or
  measure alternatives. Give the shared expert phase-specific prefill memory and program
  configs. Sweep 2D and `minimal_matmul` configs for the prefill projections. Report
  before/after numbers.

- P2: Non-aligned prefill latency is unmeasured; tails run token-by-token
  Evidence: `prefill_forward` sends any length or start offset that is not a multiple of 128
  to `_planned_prefill`. Remainders after the largest aligned buckets run one token at a time
  in eager decode mode (`count` falls back to 1). Up to 127 head tokens (unaligned
  `start_pos`) plus up to 127 tail tokens can run this way. Every warmed prefill timing in the
  stage uses lengths 128, 512 or 4096 (reviewer scan of all result JSON). The 262143-token
  context run reports no timing.
  Why this matters: Real prompts are rarely multiples of 128. The optimize skill requires
  before/after prefill latency at chunk boundaries and tails when they are used. A 4095-token
  prompt may cost far more than a 4096-token prompt; the size of the gap is unknown.
  Required next step: Measure warmed prefill at, for example, 129, 4095, 4097 and an unaligned
  continuation start, for both layer kinds. Then either reduce the tail cost (for example a
  masked padded tail chunk that does not write padding into live state) or record the measured
  cost and the exact blocker.

- P2: Material decode geometry and layout rejections are not earned under the final policy
  Evidence: The GDN in-projection (BF16 HiFi4, 127.9 us, the largest decode op, about 17% of
  GDN decode device time) was tried DRAM-sharded only at 8 activation cores with k_block 8 or
  16 (L1 overflow for readers 1 and 2), at 16 cores k8 r1 (L1 overflow) and at r3 k8
  (970.8 versus 912.2 us). Smaller legal k_block (2 or 4) with 16 or 32 activation cores,
  which fit for the full-attention QKVG (`attention_real/attention_in_dram_c32k2r*`), was never
  tried for this role. A first L1 error is not a rejection. The sharded-residual family
  ("best 900 versus 873 us") and `gdn_state_l1` (-7 us) come from early Codex screens without
  `provenance.json` or source snapshots (`attention_real`, `gdn_l1`) and from a pre-indexed
  expert policy. The experiment-provenance handoff requires disputed or narrow rejections to be
  rerun under frozen source.
  Why this matters: The selected path keeps DRAM-interleaved decode residuals and an
  auto-config in-projection at `in0_block_w=2`. The rejections that justify this were measured
  under a different policy, or stopped at the first allocation error.
  Required next step: Under frozen final source, measure the GDN in-projection DRAM-sharded
  with readers 1, 2 and 3 at k_block 2 or 4 (16 or 32 activation cores) under the selected
  dtype (and under BFP8 if P1 adopts it). Re-measure the sharded-residual candidate and
  `gdn_state_l1` against the final default.

- P2: Stage documents are stale or incomplete
  Evidence: `work_log.md` ends with "Final frozen evidence: `final_evidence2/`". The README and
  checklist cite `final_evidence3/`. `final_evidence2_chain.log` stops during
  `accept_context_0` with no end manifest, and its source differs in `tt/optimized_decoder.py`,
  `tests/run_dispatch_maps.py` and `tests/test_optimized_decoder_device.py`. The work log does
  not explain this. The 8x4 nondeterminism is missing from the rejected table. The BFP8 GDN
  output rejection cites confounded rows (see P1).
  Why this matters: The goal requires README and work log to record commands, rejected options
  and exact artifacts. AGENTS.md requires failed or old attempts to be preserved and explained.
  Required next step: Update the work log with the chain-2 abort and reason, the chain-3
  evidence, and the items above. Correct the rejected-options table.

## Other Concerns

- Batched decode is a Python loop over rows (`functional_decoder.decode_forward`). Each request
  runs the whole layer with its own M=32-padded matmuls, so B32 decode re-reads every weight 32
  times. B32 correctness passes, and the stage headline is B1. The later Stage 13 sweep requires
  client concurrency up to 32, where this design will dominate. No B32 decode timing exists.
- Policies are selected per layer kind from one layer each (linear layer 0, full layer 3). The
  sharded RMSNorm passes on layer 3 and fails on layer 0. A deep layer of each kind (for example
  38 and 39) was never checked with the selected precision policy.
- `tests/run_layer.py` keys GDN state restore on `args.layer == 0` and position refresh on
  `args.layer == 3`, not on layer type. Running another layer index gives wrong replay inputs.
- Shared-expert down BFP4 was not tried separately (routed down BFP4 was, and was correctly
  rejected on real activations at 128 steps with four tokens below 0.997).
- The README names a fused recurrent GDN decode kernel as "the remaining large opportunity"
  and about 100 us of GDN decode reshapes/permutes plus three one-core FP32 norms (53 us)
  remain. That reads as deferred decoder work. It is documented as a limitation, not
  attempted.
- Stage-2 decode baselines are means, Stage-3 headlines are medians. The difference is under
  0.1 us here (reviewer recompute: linear mean 725.96, median 725.88).

## Hard-Check Gaps

- No bitwise repeated-prefill identity check for the new device dispatch path.
- No watcher run covering the 512 or 4096 prefill buckets.
- Real-activation fixtures exist only for layers 0 and 3.
- Real-activation B32 positions is run but treated as a non-gating diagnostic for the default.
- pytest (`test_optimized_decoder_device.py`) runs only 129/8 and checks policy fields, not the
  4096 bucket or a non-aligned continuation.

## Anomaly Ledger

- Observed anomaly: Real-activation B32 positions fails (min 0.99188) for the default.
  Evidence: `final_evidence3/real_inputs/positions_0.json`; Stage 2
  `accept_ablation/fused_real_linear_positions.json` 0.99186 at the same rows and tokens.
  Affected path: GDN layer decode at B32.
  Control or comparison: Stage-2 fused decoder fails identically, so it is not an optimization
  regression.
  Likely subsystem: Unknown. Router near-tie not supported for the related token-32 case
  (reviewer CPU check).
  Investigation performed: Stage-2 control only.
  Resolution: more-work-needed (P2 token sensitivity).

- Observed anomaly: Many unrelated perturbations fail only at layer-0 decode token 32.
  Evidence: `pcc32_linear/*`, `aofid_linear/*`, `c10_linear/norm_fill.json`,
  `c11_linear/norm_fp32_fill.json`.
  Affected path: GDN layer residual/output path.
  Control or comparison: Default passes at 0.9988.
  Likely subsystem: Unknown; reviewer CPU check: router gap at token 32 is 2.8% (rank 55/128).
  Investigation performed: None by the stage.
  Resolution: more-work-needed.

- Observed anomaly: 8x4 `moe_fused_swiglu` grid breaks decode-replay bitwise determinism.
  Evidence: `rank_linear/cumsum_8x4.log` (`assert deterministic`).
  Affected path: Routed prefill dispatch FFN (default uses the same kernel at 4x4).
  Control or comparison: 4x4 and 8x2 pass; 8x4 passes on the full layer.
  Likely subsystem: `deepseek_prefill.moe_fused_swiglu` work split or output bounds (unverified).
  Investigation performed: None.
  Resolution: more-work-needed.

- Observed anomaly: Sharded `rms_norm` output row 0 depends on uninitialized tile padding.
  Evidence: `accept_ablation/det*`, `normfill_*`, `runs/main/stage03_claude/micro/det_norm.out`.
  Affected path: Full-attention decode norms.
  Control or comparison: No-norm control 7/7 identical; zero-filled padding 4/4 identical.
  Likely subsystem: TTNN sharded LayerNorm.
  Investigation performed: Bisection and fix (`decode_norm_fill`).
  Resolution: fixed in the default. Earlier comparison rows that ran without the fill remain
  confounded (P1).

- Observed anomaly: Per-layer DRAM after cleanup is about 560 MiB above the reviewer's live-set
  estimate.
  Evidence: `final_evidence3/perf_*/default.json` `weight_memory`; source live sets.
  Affected path: Weight ownership for all later profiles.
  Control or comparison: Memory-layout handoff arithmetic.
  Likely subsystem: `release_unused_weights` live-key list and same-dtype typecasts.
  Investigation performed: Reviewer source and arithmetic only (unverified on device).
  Resolution: more-work-needed.

## Scope Inspected

- Goal/skill paths: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/03-03-optimized-decoder.prompt.txt`;
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/{stage-review,optimize}/SKILL.md`;
  `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`; `/home/hous/dev/qwen3.5-35b-a3b/experiment-provenance-handoff.json`;
  `/home/hous/dev/qwen3.5-35b-a3b/memory-layout-handoff.json`.
- Artifact paths (under `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/`):
  `optimized_decoder/{README.md,work_log.md,checklist.md}`, `context_contract.json`,
  `optimized_decoder/final_evidence3/**` (perf, profile tracy CSV/TXT/ops.csv, measurement,
  roofline, positions, reuse, watcher, pytest, real_inputs, context, dispatch maps, manifests),
  `optimized_decoder/final_evidence3_chain.log`, `optimized_decoder/final_evidence2_chain.log`,
  candidate matrices and rows (`c10_*`, `c11_*`, `pcc32_linear`, `aofid_linear`,
  `aobf16_linear`, `c5v_full`, `c5abl_full`, `shared_full`, `geom_*`, `inproj_*`,
  `rank_*`, `rankmm_*`, `gdn_real`, `attention_real`, `final_linear`, `final_full`,
  `accept_ablation/*`), `fused_decoder/README.md`, `fused_decoder/headline_final/*`,
  `fused_decoder/indexed_rope_native4096.json`, `fused_decoder/residual_linear4096/kda_tail.json`.
- Code paths: `tt/optimized_decoder.py` (all), `tt/functional_decoder.py` (decode_forward,
  router, norm), `tests/run_layer.py`, `tests/run_dispatch_maps.py`,
  `tests/test_optimized_decoder_device.py`, `tests/hf_reference.py`,
  `tests/record_optimized_inputs.py`, `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_claude/evidence_chain3.sh`.
- Commands run: read-only `cat`/`grep`/`git status`/`git log`; Python analysis of result JSON,
  perf-report CSV and `ops.csv`; SHA-256 recheck of current sources against the frozen
  manifest; two CPU-only HF scripts (layer 0, 4224 recorded tokens; router top-8 margin and
  per-token output statistics) in the reviewer scratchpad. No device, server or hardware job.

## Residual Risk

- Headline numbers re-derived from raw JSON match the README: linear 725.878 us decode median
  and 35990.046 us prefill; full 430.051 us and 29621.407 us; speedups 1.73x/2.14x decode and
  21.2x/23.5x prefill against the Stage-2 best correct baselines; roofline 186.6/107.8 us and
  22.66/22.71% decode DRAM fractions; perf-report rows show the claimed dtypes and fidelities
  (QKVG BF16xBFP8 LoFi, sparse gate/up BFP4 LoFi, sparse down BFP8 LoFi, shared down BFP8 LoFi
  DRAM-sharded, GDN in/out BF16 HiFi4, SDPA with BFP8 KV).
- Trace evidence uses `TT_METAL_TRACE_ALLOC_TRACKING=1` with program-cache allocations
  included and acknowledges only the final trace output.
- Context contract remains 262144 with passing 262143/262144 runs on the final source.
- The P1 re-qualification could reverse the GDN policy, which would change perf, roofline and
  profile artifacts. Those must be regenerated on the final source.
