# Stage Review

Stage 4, multichip decoder. Review 4, after `stage_review.md` (review 1), `stage_review2.md` and `stage_review3.md`.
Model: Qwen/Qwen3.5-35B-A3B, revision 59d61f3ce65a6d9863b86d2e96597125219dc754.
Reviewer: fresh independent subagent. Review date: 2026 Oct 11. I did not run any device job, server, build or test.

Abbreviations: TP (tensor parallel), EP (expert parallel), MoE (mixture of experts), GDN (Gated DeltaNet),
CCL (collective communication), PCC (Pearson correlation coefficient), HF (Hugging Face reference),
B1/B32 (batch 1 / batch 32), DRAM (device memory), BF16 (16-bit brain float), BFP4/BFP8 (4/8-bit block float),
RMSNorm (root mean square normalization), TTFT (time to first token).

Short names used below:
- `AP` = `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b`
- `MD` = `AP/doc/multichip_decoder`
- `E4` = `MD/evidence4`
- `ND` = `MD/evidence4/nondeterminism`

Verdict: more-work-needed

## Required Work

- P2: On p150x2 and p150x4, a 2176-token prefill is systematically slower than a 4096-token prefill. On p150x4 it
  is no faster than the single chip. The stage records this only as "timing varies between chains" and leaves it
  open. The artifacts show a systematic, mesh-specific cost, and the code shows a plausible stage-relevant cause.
  Evidence:
  - Warmed prefill host time (`prefill_host_us`) in `MD/evidence{1,2,3,4}/headline_*/commands.json` and
    `MD/evidence{1,2,3,4}/len_m*_*_2176/commands.json`, chains 1 to 4:

    | Profile, layer | 4096 tokens | 2176 tokens |
    |---|---|---|
    | p150x4 full | 20946, 19972, 20279, 19670 us | 24402, 21597, 22131, 24313 us |
    | p150x4 linear | 20882, 20780, 21969, 20848 us | 21666, 21638, 23159, 22409 us |
    | p150x2 full | 21820, 21921, 22204, 21817 us | 21707, 22207, 22481, 22142 us |
    | p150 full | 25422, 25521, 25562, 25488 us | 23608, 23594, 23668, 23573 us |
    | p150 linear | 32222, 32270, 32236, 32212 us | 27359, 27348, 27312, 27230 us |

    On p150x4, 2176 tokens is slower than 4096 tokens in all 8 (chain, layer) pairs. On p150, it is faster in all 8.
    This is not chain-to-chain noise.
  - The split control isolates the cost. `len_m*_*_2176_s128` runs the same 2176 tokens as two calls (128, then
    2048 at start 128). In chain 4 on p150x4, two calls take 20071 us (full) and 17630 us (linear). One call takes
    24313 us and 22409 us. One call costs 4.2 to 7.5 ms more than two calls on p150x4 full in every chain
    (3.6 to 5.4 ms on linear). On p150 the same difference is 0.75 to 0.84 ms (full) and 0.62 to 0.87 ms (linear).
    The extra cost is mesh-specific.
  - Plausible mechanism (from code reading; not verified on hardware). For 2176 tokens the planner runs attention
    chunks 2048 + 128 and then one MoE group. `AP/tt/optimized_decoder.py:432-434` pads the MoE group to the next
    size in `sizes`, which is 4096, with zero rows. A zero row stays zero after RMSNorm, so all 256 router logits
    are equal and top-k picks the same tie-broken 8 experts for all 1920 padding rows. On a single chip this costs
    the same total work as a real 4096-row group, which matches the p150 numbers. Under EP
    (`AP/tt/multichip_decoder.py::_ExpertParallelDispatch`) whole experts live on one rank. If the 8 tied experts
    sit on one rank, that rank gets about 15360 padding routes in addition to its share of real routes, while the
    other ranks wait at the MoE all-reduce. The route-map code already has a mask (`inside`, line 245) and a
    zeroed dummy slot (`capacity - 1`, lines 256 and 268). It could send padding-row routes there, but it does
    not.
  - README ledger row "Mesh prefill time varies between chains (one timed run per row)" lists some of these
    numbers but frames them as variance. It does not state the 2176 > 4096 inversion or the split-control
    result. Its resolution is "open, Stage 5". That is not one of the skill's allowed resolutions (fixed,
    controlled, more-work-needed). No investigation was run.
  Why this matters: p150x4 is the optimization target. Any prompt whose last MoE group is not a bucket size gets
  padding up to the next size, in the worst case almost twice the real rows. If the mechanism holds, every such
  prefill on p150x2 and p150x4 pays an EP load imbalance on one rank. This affects TTFT in the required
  short/medium latency cells, and it makes the multichip prefill slower than one chip at valid lengths. The
  goal asks the stage to use perf evidence to check the path before it accepts it, and it asks for "the best
  possible use of TT mesh". The stage-review standard treats a material anomaly that is dismissed with prose as
  required work.
  Required next step: classify the anomaly with evidence. For example, record per-rank route counts or a
  device profile for one 2176-token p150x4 prefill, and compare it with the 4096-token row. If padding routes
  cause the imbalance, fix it (for example, send padding-row routes to the zeroed dummy slot in
  `route_maps`, or stop padding the MoE group to the next bucket on meshes). Then measure 2176 tokens on p150x2
  and p150x4 again under a frozen source manifest, and rerun the affected PCC and length gates. If the cause is
  something else, record it with evidence and give the anomaly a resolution that the skill allows.

## Earlier Findings: Status

- Review 3 P2 (nondeterminism metric name, E2 target, "carried state" inference): resolved.
  - README "Trace, determinism, audit, watcher" now calls the metric "first prefill (compiles programs) against
    the warmed prefill". `AP/tests/run_layer.py:213-236` confirms this is what `prefill_repeat_identical` compares.
  - README reports the cold E0 result ("run 0 is still the odd run in 5 of 19 drifting (call, device) pairs").
  - New warm E0 run: `ND/e0e2/x4_full_b32_e0e2_warm.json`, reference warm run 1, warm runs 2 to 6 compared. I
    re-derived the numbers. Differing captures per run are 61, 84, 82, 69 and 53 (README: 53 to 84). Layer
    output (capture 544) differs by 151614, 255183, 201695, 198978 and 172091 elements, max abs 0.015625 (README:
    151614 to 255183). `moe_input_side_drift_captures` is 0.
  - The E2 replay target is now a warm-run alternate: capture 10, devices 1 and 2. In all 5 comparisons the
    element count and max abs at 10:1 and 10:2 are identical (3803451 / 0.00342 and 3085655 / 0.01172). So runs
    2 to 6 agree, and warm run 1 holds the alternate value. Back-to-back and interleaved replays give 1 distinct
    output per device, 200 of 200 equal to the model call (`replay` block). E2a on one chip gives the same result
    (`ND/e0e2/x4_full_b32_e2a_single_warm.json`).
  - True warm-only drift exists, for example capture 10 on device 0 differs in runs 3 and 5 with different
    element counts (3174925 and 3171447). README "Warm runs drift too" is correct.
  - The "carried state" inference is removed. README now says "The mechanism is unknown. Carried device state
    from preceding ops and execution timing both fit; E2 does not separate them." Root cause unknown, owner
    Stage 5. This matches the evidence.
  - `ND/e0e2/source_manifest_warm.json` equals `E4/source_manifest_start.json`. The probe diff to
    `probe_e0e2.py` only adds the reference-run argument.
- Review 3 other concerns:
  - Mesh prefill timing: a ledger row was added, but the systematic part is not classified. See the P2 above.
  - Reshape, typecast and slice in the p150x4 linear decode (112.6 of 522.6 us): now in Findings, the ledger and
    Stage 5 work, and named as inherited from Stage 3. Resolved as a documentation item.
  - E0 raw tensors for E1: README now says E0 and E2 ran with summary statistics and one saved call per target,
    and lists E1 as Stage 5 work. Resolved.
- Review 2 P2 and review 1 P2 findings (B32 reporting, memory handoff recheck, construction peaks, bound across
  chains, row counts): still resolved. I re-derived the chain-4 table rows below; nothing changed since review 3.

## Other Concerns

- The device pytest opens p150x2 on `TT_VISIBLE_DEVICES=0,1` by default
  (`AP/tests/test_multichip_decoder_device.py`, `QWEN35_X2_DEVICES`). The documented p150x2 profile uses PCIe 2,3.
  The test is still valid, but it qualifies the other card. Say so in README, or default to 2,3.
- The 128-token first call of every split row takes 5.5 to 7.6 ms on all profiles (`prefill_host_us` minus
  `continuation_host_us` in `E4/len_m*_*_2176_s128/commands.json`). The 3-token prefill takes 4.6 to 5.5 ms. Short
  prefills have a large fixed cost. This is inherited from Stage 3, but it will dominate short-prompt TTFT. Name it
  as a Stage 5 item.
- p150x2 qkvg: DRAM-sharded is 29.9 us against 34.1 us selected
  (`E4/micro/projection_geometry_sweep.json`). The rejection reason is a second padded weight copy of 9.6 MiB per
  full layer (about 96 MiB for 10 full layers). p150x2 has about 3.9 full-context sequences of headroom in
  `E4/capacity_model.json`, so memory does not force this. The cost is about 42 us per token over 40 layers.
  This is small and p150x2 is not the optimization target, so I do not count it as required work.
- `merge_stats.sharded_bytes` is 0 in every row, because `items[0]` has no `buffer_size` attribute
  (`AP/tt/multichip_decoder.py::merge_replicas`). The field is misleading. Remove it or fix it.
- The README nondeterminism table has a "repeat" entry for p150x2 1000 tokens in chain 2 and chain 3. It means
  "bitwise repeat". Spell it out.

## Hard-Check Gaps

- Each headline and length row is one process and one timing run. The 4096-token headline varies by up to 6%
  across chains (p150x4 linear 20780 to 21969 us). Decode is stable to under 1%.
- No mesh test drives an EP rank with zero local routes, or all routes on one rank, through the whole layer.
  From code reading the path is safe: `capacity >= routes + local*31 + 32` holds even if all routes land on one
  rank, and `restore_table` zeroes the dummy row.
- The source manifest hashes the ttnn patch text (723ed3a1), not the built `_ttnn.so`.
- The micro sweeps (`E4/micro/*.json`) have no source hashes. Chain 4 confirms the selected configs on the full
  layer.
- No p150 profile under Stage 4. p150 is bitwise equal to the Stage 3 baseline in all 18 rows, so the Stage 3
  profiles apply.
- The row-parallel topology table in README is compact. It does not list collective bytes, CCL dtype or the
  persistent-buffer plan per row as the multichip skill suggests. The measured rows (all-reduce, reduce-scatter +
  all-gather, sharded residual with distributed RMSNorm, bound for a fused all-gather + matmul) cover the
  required families. `ttnn.all_reduce` logs "Allocating semaphores in L1" once per run; no persistent semaphore
  plan is recorded.

## Anomaly Ledger

- Observed anomaly: 2176-token prefill slower than 4096-token prefill on p150x4 (all chains, both layer kinds)
  and on p150x2 full (3 of 4 chains); p150x4 full at 2176 tokens is no faster than p150 in chains 1 and 4.
  Evidence: `MD/evidence{1,2,3,4}/headline_*/commands.json`, `MD/evidence{1,2,3,4}/len_m*_*_2176*/commands.json`.
  Affected path: planned prefill with a padded MoE group (`AP/tt/optimized_decoder.py:432-434`) and EP dispatch
  (`AP/tt/multichip_decoder.py::_ExpertParallelDispatch.route_maps`).
  Control or comparison: p150 is faster at 2176 than at 4096. The two-call split control is 4.2 to 7.5 ms faster
  on p150x4 full and 0.75 to 0.84 ms faster on p150.
  Likely subsystem: EP load imbalance from padding rows that all route to the same tied experts (hypothesis, not
  verified).
  Investigation performed: none by the stage. This review re-derived the numbers and read the code.
  Resolution: more-work-needed (Required Work P2).
- Observed anomaly: prefill not bitwise repeatable (p150x4 B32 always, p150x2 1000 tokens sometimes, single-chip
  baseline rarely); warm runs also differ from each other.
  Evidence: `MD/evidence{1,2,3}/b32_m4_*`, `E4/b32_m4_*`, `E4/len_m2_*_1000`, `E4/len_base_linear_1000_commands.json`,
  `ND/*.json`, `ND/e0e2/*.json`.
  Affected path: prefill MoE (`moe_fused_swiglu`, `post_combine_reduce`) output; inputs bitwise identical.
  Control or comparison: p150 and baseline B32 repeat in every chain. For fixed inputs the op gives one value in
  600 replays per target, for a cold-run and a warm-run alternate, on one chip and on the mesh. Max abs 0.0156 in
  the layer output; control and HF gates pass; decode replays bitwise repeat.
  Likely subsystem: device or runtime state around the MoE kernels; mechanism unknown.
  Investigation performed: per-op probe, list-path control, standalone loops, AutoDebug source audit, E0 cold and
  warm, E2a and E2b cold and warm.
  Resolution: controlled by bound for Stage 4; root cause owned by Stage 5. Serving stages need a
  warm-against-warm repeatability check.
- Observed anomaly: layer 0 recorded-input positions, row 17 token 4, decode PCC 0.99393 to 0.99414 on all
  profiles (the three expected `rc=1` steps in `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage04/evidence_chain4.log`).
  Control or comparison: Stage 3 single chip 0.99393
  (`AP/doc/optimized_decoder/final_evidence15/real_inputs/positions_0.json`).
  Resolution: controlled.
- Observed anomaly: 1 to 3 prefill tokens per 4096 below 0.995 against HF.
  Control or comparison: same counts in the baseline; the tokens tie exactly at the 8th/9th router logit
  (`E4/micro/router_margin_layer0.json`).
  Resolution: controlled.
- Observed anomaly: B32 decode is 29 to 31 times the B1 step.
  Control or comparison: the Stage 3 baseline has the same per-row loop (`b32_base_*_commands.json`).
  Resolution: controlled; owner Stage 5 and Stage 7.
- Observed anomaly: p150x4 watcher build fails without `TT_METAL_WATCHER_NOINLINE=1`.
  Evidence: `MD/evidence1/accept_watcher_m4/linear.log`. All 16 chain-4 `watcher.log` files have no error,
  assert, sanitizer, hang or timeout line.
  Resolution: fixed.

## Verified Claims

- Chain 4: 125 steps; every step exits 0 except the three expected `real_positions_0_m*`. The log ends with
  `SOURCE_STABLE` and `CHAIN_DONE`. `E4/source_manifest_start.json` equals `source_manifest_end.json`. All 41 hashed
  files match the current worktree. `git diff HEAD` outside the autoport hashes to 744236997d8a, equal to the
  manifest, and is the same as `git diff HEAD -- ttnn tests/ttnn`. `E4/ttnn_patch.diff` hashes to 723ed3a1.
- Headline, length and B32 tables in README match the raw `commands.json` rows (prefill and decode host times,
  control PCC, HF PCC, replica identity, repeat flags). 42 length rows and 18 baseline rows exit 0.
- p150 is bitwise equal to the baseline in every p150 row (`control_equivalence` identical true).
- Reuse, positions (except layer 0 row 17), native context 262143 and 262144 on all three profiles, device pytest
  (12 passed), CPU tests (1154, 0 failures, 0 errors) pass. Positions rows record `trace_allocation_tracking` true;
  the device pytest sets `TT_METAL_TRACE_ALLOC_TRACKING=1` and `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`; the
  trace acknowledgment is exact-tensor (`acknowledge_corruptible(out)`, `AP/tests/run_layer.py:307`).
- Decode profile dtypes match the policy: routed gate/up BFP4 LoFi, routed down BFP8 LoFi, shared gate/up BFP4
  LoFi, GDN in-projection BF16 HiFi4 (Stage 3 real-weight blocker for BFP8), router BF16 HiFi4
  (`E4/profiles/m4_linear_attention/decode_perf_report.csv`).
- Geometry: the x4 GDN in-projection keeps `k_block` 2, but 4, 8, 16 and 32 were measured slower under the same
  BF16 HiFi4 policy. x4 qkvg (40 cores, k 8) is the best of 46 candidates.
- Profile family sums in README match `decode_replay_summary.txt` (for example p150x4 full reshape + typecast +
  slice = 19.3 us).
- Context contract: `capability_reduction` null, 262144 tokens on every profile, capacity per GDN slot count.

## Scope Inspected

- Goal/skill paths: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/04-04-multichip-decoder.prompt.txt`,
  `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/multichip/SKILL.md`.
- Artifact paths: `MD/README.md`, `MD/work_log.md`, `MD/stage_review.md`, `MD/stage_review2.md`,
  `MD/stage_review3.md`, `E4/` (manifests, headline, len, b32, accept, real_inputs, watcher logs, profiles,
  micro sweeps, residual contract, memory recheck, capacity model), `ND/e0e2/` (cold and warm JSON, probes,
  manifests), headline and length rows in `MD/evidence{1,2,3}/`, `AP/doc/context_contract.json`,
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage04/evidence_chain4.log`.
- Code paths: `AP/tt/multichip_decoder.py` (full), `AP/tt/optimized_decoder.py` (tiled and planned prefill,
  `_chunk_plan`, `_batched_prefill` MoE padding, router, policy defaults), `AP/tt/fused_decoder.py` (router),
  `AP/tests/run_layer.py` (prefill timing, split, trace), `AP/tests/test_multichip_decoder_device.py`.
- Commands run: `git status`, `git log`, `git diff | sha256sum`, `sha256sum`, `diff`, `grep`, `zcat`, and Python
  analysis of existing JSON, CSV and XML artifacts. No device, pytest, server or build.

## Residual Risk

- Mesh prefill is not bitwise repeatable and the cause is unknown. Greedy outputs can differ between identical
  requests. Stage 6 and the serving stages need a warm-against-warm repeatability check.
- TP and EP correctness rests on control PCC and replica identity, not on bitwise equality. Rare routing patterns
  (all routes on one rank) are not explicitly tested.
- p150 capacity depends on the configured GDN slot count (0.77 full-context sequences at 32 slots). The full
  model must set the slot count and measure capacity.
- Prefill timing evidence is one run per row; small differences between profiles at one length are not
  reliable.
