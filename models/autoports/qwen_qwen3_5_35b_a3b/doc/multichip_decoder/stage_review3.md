# Stage Review

Stage 4, multichip decoder. Review 3, after review 1 (`stage_review.md`) and review 2 (`stage_review2.md`).
Model: Qwen/Qwen3.5-35B-A3B, revision 59d61f3ce65a6d9863b86d2e96597125219dc754.
Reviewer: fresh independent subagent. Review date: 2026 Oct 11. I did not run any device job, server, build or test.

Abbreviations: TP (tensor parallel), EP (expert parallel), MoE (mixture of experts), GDN (Gated DeltaNet),
CCL (collective communication), PCC (Pearson correlation coefficient), HF (Hugging Face reference),
B1/B32 (batch 1 / batch 32), DRAM (device memory), BF16 (16-bit brain float), BFP4/BFP8 (4/8-bit block float),
ULP (unit in the last place).

Short names used below:
- `MD` = `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/multichip_decoder`
- `E4` = `MD/evidence4`
- `ND` = `MD/evidence4/nondeterminism`

Verdict: more-work-needed

## Required Work

- P2: The nondeterminism section calls the check "repeated warmed prefill", but it compares a cold first prefill
  with a warmed second prefill. The new E2 replay targets a call that drifts only in the cold run. So E2 cannot
  support the README's claim that the cause is state carried from preceding ops.
  Evidence:
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_layer.py:213-236`.
    The first `prefill()` runs under `audit()` with no warmup before it, so it compiles the programs. The
    `--warm-prefill` run is the second call. `prefill_repeat_identical` compares these two outputs
    (`torch.equal(pf, measured_host)`). The README says "Repeated warmed prefill is bitwise identical in 69 of 74
    rows", and its rate claim ("p150x4 B32 prefill never repeats, 8 of 8 rows") uses the same cold-against-warm
    metric.
  - `ND/e0e2/probe_e0e2.py` also has no warmup, so run 0 is the cold run. In `ND/e0e2/x4_full_b32_e0e2.json`
    the replay target is capture 27, device 3. Runs 1, 2 and 3 each differ from run 0 there with the same
    element count and the same max abs (3340987 elements, 0.00415). This means runs 1 to 3 agree, and only the
    cold run 0 has the alternate output. Four other (call, device) pairs show the same run-0-only pattern
    (`384:2`, `435:1`, `435:3`, `520:0`). The other 14 pairs drift in one warm run only.
  - E2a and E2b replay the call 200 times with the programs already compiled. They match the last model call,
    which is the warm value (`ND/e0e2/x4_full_b32_e2a_single.json`, `replay` in `x4_full_b32_e0e2.json`). A warm
    replay that cannot reproduce a cold-run-only output is the expected result if the cause is in the first
    execution. So E2 does not separate "state from preceding ops" from "first execution". README says: "The
    alternate output needs the preceding ops of the same layer prefill in the same run. Per AutoDebug E2, this
    points at state carried from preceding ops into the call". The selected target does not support this.
  - AutoDebug E0 (`MD/evidence2/moe_nondeterminism_autodebug/AUTODEBUG.md`, section 6) asks whether run 0 stops
    being the odd one after the probe fix. It is still the odd one for 5 of 19 drifting pairs. The README does
    not report this result.
  - Warm-against-warm drift does exist, so the anomaly is real and not only a compile effect. The layer output
    differs from run 0 by 124283, 130698 and 213670 elements in runs 1, 2 and 3, so the warm runs also differ
    from each other. Single-run drifts at captures such as `61:2` and `95:2` are warm-only events.
  Why this matters: review 2 required an accurate classification of this anomaly. The section now has a wrong
  name for the measured metric, and its only causal statement rests on an experiment whose target cannot test
  that statement. Stage 5 inherits the next experiments (E3, E1, E5). If they are designed around "carried state"
  and use cold-run targets, they test the wrong thing. The serving-relevant rate (warm against warm) is not
  measured by any chain row.
  Required next step: correct README "Trace, determinism, audit, watcher" and "Nondeterminism":
  - Describe the metric as first (compile) prefill against second (warmed) prefill.
  - Report the E0 result: run 0 is still the odd one for 5 of 19 drifting pairs, and the warm runs also differ
    from each other (layer output elements above).
  - Remove or qualify the "carried state" inference. E2 used a cold-run-only target, so it cannot separate
    carried state from first-execution effects.
  - Keep "root cause unknown" and the Stage 5 owner.
  If the stage wants to keep any causal statement, run E2 again on a call that drifts in a warm run, for example
  one of the single-run pairs. No new hardware run is required if the text is corrected.

## Earlier Findings: Status

- Review 1 P2, nondeterminism disclosure and isolation: disclosed, probed and controlled-by-bound, apart from the
  wording and inference in the P2 above.
- Review 1 P2, B32 reporting: resolved. I re-derived the `E4/b32_*` table in review 2 and it is unchanged. The
  expert-union figure is now 163.
- Review 1 P2, memory handoff recheck and construction peak: resolved. `E4/memory_handoff_recheck.json` now
  separates `replica_merge_peak_device_mib`, `setup_peak_before_cleanup_mib` and `construction_transient_peaks`,
  and documents them in `field_notes`. `released_after_merge` is null on p150, which is correct.
- Review 2 P2, bound and cause: mostly resolved.
  - Range across chains: correct. I re-derived it from all four chains (`*/b32_m4_*/commands.json`,
    `*/len_*_1000*`, `evidence2/len_base_full_2176_s128_commands.json`). B32 p150x4 full: 953, 709, 1305 and 1430
    rows, max abs 0.0156, 0.0117, 0.0156 and 0.0078. Minimum repeat PCC 0.9999982. All match the README table.
  - Row counts: correct. 74 repeat fields in chain 4, 5 false (56 mesh rows plus 18 baseline rows).
  - "Not caused by the Stage 4 changes" is gone. It now says "not specific to the stacked extension" and
    "root cause unknown". This is correct.
  - E0 and E2 ran under a source manifest equal to the chain 4 manifest
    (`ND/e0e2/source_manifest.json` equals `E4/source_manifest_start.json`, and every hashed file matches the
    worktree). Remaining issue: see the P2 above.
- Review 2 other concerns:
  - Baseline log overwrite: resolved. `E4/headline_base_logs/full_default.json` is layer 3 and
    `linear_default.json` is layer 0, and their timings match `E4/headline_base_*_commands.json`.
  - Work log: resolved. It names `evidence4/` as final and has entries for chain 3, chain 4 and review 2.
  - GiB units, the profiler overhead note and the placement note are fixed in README.
  - Micro sweep source hashes, built-library hash and single timing runs: still open as hard-check gaps (see
    below).

## Other Concerns

- Mesh prefill timing varies between chains, and the README does not classify this. Each row is one timed run.
  The variation is larger than some reported differences:
  - p150x4 full layer, 2176 tokens: 24402, 21597, 22131 and 24313 us in chains 1 to 4. In chains 1 and 4 this
    is equal to or slower than p150 (23608 and 23573 us). In chain 4 it is also slower than p150x2 (22142 us).
  - p150x4 full layer, 2176 = 128 + 2048: 16920, 17158, 17070 and 20071 us. Chain 4 is 18% slower than the
    other chains and slower than p150x2 in the same chain (18178 us).
  - p150x4 full layer, 1000 tokens: 10420 us in chain 4, slower than p150x2 (9645 us).
  - p150x4 full layer, 2176 tokens takes longer than 4096 tokens (19670 us) in every chain. Part of this comes
    from the Stage 3 ladder, which pads the MoE group to 4096. On p150, 2176 tokens is still faster than 4096.
  - Headline 4096 prefill varies by up to 6% (p150x4 linear 20780 to 21969 us). Decode is stable to 0.5 us.
  Add a ledger row: state the per-chain range for the mesh prefill rows, and say whether the length-row prefill
  times can be compared. If p150x4 full-layer prefill at 2176 tokens is really no faster than p150, name it as a
  Stage 5 prefill item.
- The largest decode op family on the p150x4 linear layer is not classified. Reshape, typecast and slice take
  112.6 of 522.6 us per step (`E4/profiles/m4_linear_attention/decode_replay_summary.json`: ReshapeView 45.6,
  Typecast 39.1, Slice 27.8). Linear layers are 30 of 40 layers. README lists it in the profile table, but it is
  not in "Findings" or in "Limitations and Stage 5 work". It is probably inherited from Stage 3. Say so and give
  it a Stage 5 owner.
- E0 in the AutoDebug plan asks for raw tensors for every differing capture. `ND/e0e2/probe_e0e2.py` saves only
  summary statistics, plus the inputs of one call
  (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage04/micro/nondeterminism/e0e2/x4_full_b32_e0e2.call.pt`, hashed in
  `ND/e0e2/sha256.txt`). E1 (column-band map) needs both outcomes of one call as raw tensors. This is
  Stage 5 work, but the README lists E0 as "done".
- The E0/E2 run has one source manifest, not a start and end pair. It is an investigation, not an acceptance
  gate, so I do not count this against the frozen-source rule in
  `/home/hous/dev/qwen3.5-35b-a3b/experiment-provenance-handoff.json`.
- p150 capacity depends on GDN slot count. With 32 configured slots, one 262144-token sequence does not fit
  (0.77 sequences, `E4/capacity_model.json`). README and `context_contract.json` say this. The full model must set
  the slot count and measure capacity. This is not a Stage 4 reduction.

## Hard-Check Gaps

- Each headline and length row is one process and one timing run (see the timing concern above).
- No mesh test drives an EP rank with zero local routes, or all routes on one rank, through the whole layer.
  From code reading, the path is safe. In `tt/multichip_decoder.py::route_maps`, non-local routes go to slot
  `capacity-1`, which is above every region end because `capacity >= routes + local*31 + 32`. `restore_table`
  zeroes that row.
- The source manifest hashes the ttnn patch text, not the built `_ttnn.so`. The lineage is consistent: the build
  is older than every chain, and the patch hash is 723ed3a1 in all chains.
- The micro sweeps (`E4/micro/*.json`) have no source hashes. Chain 4 confirms the selected configs on the
  full layer.
- No p150 profile under Stage 4. p150 is bitwise equal to the Stage 3 baseline in all 18 rows, so the Stage 3
  profiles apply.

## Anomaly Ledger

- Observed anomaly: mesh prefill not bitwise repeatable (B32 p150x4 in every chain; p150x2 at 1000 tokens
  sometimes; single-chip baseline rarely).
  Evidence: `MD/evidence{1,2,3}/b32_m4_*`, `E4/b32_m4_*`, `E4/len_m2_*_1000`, `ND/*.json`, `ND/e0e2/*.json`.
  Affected path: prefill MoE (`moe_fused_swiglu`, `post_combine_reduce`) on meshes.
  Control or comparison: p150 and baseline B32 repeat in every chain. Captured inputs are identical (E0). The op
  repeats on fixed inputs on one chip and on the mesh (E2a, E2b). Max abs 0.0156 over 4 chains. Control and HF
  gates pass.
  Likely subsystem: hardware or runtime state around the MoE kernels. The mechanism is unknown. A first-execution
  factor is likely for some calls (run 0 odd in 5 of 19 pairs). Warm runs also drift.
  Investigation performed: per-op probe, list-path control, standalone loops, AutoDebug source audit, E0, E2a,
  E2b.
  Resolution: controlled by bound for Stage 4. The documentation needs correction (Required Work P2). Owner
  Stage 5.
- Observed anomaly: layer 0 recorded-input positions, row 17 token 4, decode PCC 0.9939 to 0.9941 on all
  profiles (three expected `rc=1` steps in `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage04/evidence_chain4.log`).
  Evidence: `E4/real_inputs_m{1,2,4}/positions_0.json`.
  Control or comparison: Stage 3 single chip 0.99393
  (`../optimized_decoder/final_evidence15/real_inputs/positions_0.json`). The BF16 reference itself deviates
  there (Stage 3 README).
  Resolution: controlled.
- Observed anomaly: 1 to 3 prefill tokens per 4096 below 0.995 against HF.
  Control or comparison: same counts in the baseline. The tokens tie exactly at the 8th/9th router logit
  (`E4/micro/router_margin_layer0.json`).
  Resolution: controlled.
- Observed anomaly: B32 decode is 29 to 31 times the B1 step.
  Control or comparison: the Stage 3 baseline has the same per-row loop.
  Resolution: controlled. Owner Stage 5 and Stage 7.
- Observed anomaly: the p150x4 watcher build fails without `TT_METAL_WATCHER_NOINLINE=1`.
  Evidence: `MD/evidence1/accept_watcher_m4/linear.log`. The 16 chain 4 watcher logs are distinct files, the
  runs completed, and the logs contain no error, assert, sanitizer, hang or timeout line.
  Resolution: fixed.
- Observed anomaly: mesh prefill timing varies between chains by up to 18%. p150x4 full-layer prefill at 2176
  tokens is no faster than p150 in 2 of 4 chains.
  Evidence: `MD/evidence{1,2,3,4}/len_m{1,2,4}_full_{1000,2176,2176_s128}/commands.json`.
  Control or comparison: decode in the same rows is stable. The 4096-token headline varies by 6%.
  Likely subsystem: host dispatch on meshes or the prefill ladder plus EP dispatch bookkeeping. Not investigated.
  Resolution: not classified by the stage (Other Concerns).

## Verified Claims

- Chain 4: every step exits 0, except the three expected `real_positions_0_m*` gate failures. The log has
  `SOURCE_STABLE` and `CHAIN_DONE`. Start and end manifests are equal and match the current worktree.
  `git_diff_outside_autoport` (744236997d8a) equals the current `git diff HEAD` outside the autoport.
- Headline speedups, efficiencies and the 40-layer decode estimate match `E4/tables.md` and the raw rows.
- The headline and acceptance runners set `TT_METAL_TRACE_ALLOC_TRACKING=1` and
  `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0` (`tests/run_optimized_trials.py:72`, accept `commands.json`).
- Reuse, positions (except layer 0 row 17), native context 262143/262144 on all three profiles, device pytest
  (12 passed), CPU tests (1154 passed, 0 failures), stacked op test (3 passed).
- `tt/multichip_decoder.py`:
  - Head ownership is correct. Query heads `4r..4r+3` on x4 map to KV head `r*2//4`, which matches the 8:1 GQA
    grouping (GQA: grouped-query attention).
  - GDN key and value head slices are contiguous, so value head `j` keeps key head `j//2`.
  - There are no biases before the all-reduces. The shared-expert sigmoid gate and the routed score weights are
    linear in the partials.
  - The route-map arithmetic is exact. Per-expert tile counts are at most 128 at the 4096 chunk, so the BF16
    `tiles` tensor is exact. The FP32 prefix matmuls are exact below 2^24.
- The `moe_fused_swiglu` program hash includes `stacked_intermediate` and the two column offsets
  (`moe_fused_swiglu_types.hpp`), so list-mode and stacked-mode programs cannot share a cache entry.

## Scope Inspected

- Goal/skill paths: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/04-04-multichip-decoder.prompt.txt`,
  `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/multichip/SKILL.md`.
- Artifact paths: `MD/README.md`, `MD/work_log.md`, `MD/stage_review.md`, `MD/stage_review2.md`, `E4/` (manifests,
  headline, len, b32, accept, real_inputs, watcher logs, profiles, tables, capacity model, memory recheck,
  headline_base_logs), `ND/` and `ND/e0e2/`, repeat rows in `MD/evidence{1,2,3}/`,
  `MD/evidence2/moe_nondeterminism_autodebug/AUTODEBUG.md`,
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage04/evidence_chain4.{sh,log}`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/context_contract.json`.
- Code paths: `tt/multichip_decoder.py` (full), `tt/optimized_decoder.py` (`_LocalDispatch`, `_StackedDispatch`,
  dispatch `__call__`), `tests/run_layer.py` (prefill repeat logic), `tests/test_multichip_decoder_device.py`,
  the `moe_fused_swiglu` device operation diff.
- Commands run: `git status`, `git diff --stat`, `git diff`, `sha256sum`, `md5sum`, `grep`, `zcat`, and Python
  analysis of existing JSON and XML artifacts. No device, pytest, server or build.

## Residual Risk

- Mesh prefill is not repeatable, and the cause is unknown. Greedy outputs can differ between identical
  requests. Stage 6 and the serving stages need a warm-against-warm repeatability check at the full-model level.
- TP and EP correctness rests on control PCC and replica identity, not on bitwise equality. Rare routing patterns
  are not explicitly tested.
- p150 capacity comes from a model with a small margin, and it depends on the configured GDN slot count.
