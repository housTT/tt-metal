# Stage Review

Stage 4, multichip decoder, re-review after review 1
(`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/multichip_decoder/stage_review.md`).
Model: Qwen/Qwen3.5-35B-A3B, revision 59d61f3ce65a6d9863b86d2e96597125219dc754.
Reviewer: fresh independent subagent. Review date: 2026 Oct 11. No device, server, build or test was run.

Abbreviations: TP (tensor parallel), EP (expert parallel), MoE (mixture of experts), GDN (Gated DeltaNet),
CCL (collective communication), PCC (Pearson correlation coefficient), HF (Hugging Face reference),
B1/B32 (batch 1 / batch 32), DRAM (device memory), RSS (resident set size), ULP (unit in the last place),
BFP4/BFP8 (4/8-bit block float), BF16 (16-bit brain float).

Short names used below:
- `MD` = `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/multichip_decoder`
- `E4` = `MD/evidence4`

Verdict: more-work-needed

## Required Work

- P2: The nondeterminism classification states an effect bound that preserved artifacts contradict, and it makes a
  causal claim that its own AutoDebug report says not to make yet.
  Evidence:
  - README "Nondeterminism" says "Size: max abs 0.00024-0.0078" and "Effect size: max abs at most 0.0078 in the layer
    output". This is true only for chain 4. The same B32 p150x4 configuration, with the same ttnn patch
    (`ttnn_patch.diff` sha256 723ed3a1 in all four chains), gives:
    - `MD/evidence1/b32_m4_full/commands.json`: 953 rows, max abs 0.015625.
    - `MD/evidence2/b32_m4_full/commands.json`: 709 rows, max abs 0.01171875.
    - `MD/evidence3/b32_m4_full/commands.json`: 1305 rows, max abs 0.015625.
    - `E4/b32_m4_full/commands.json`: 1430 rows, max abs 0.0078125.
    Review 1 named 0.0156 as two times the Stage 3 maximum (0.0078). The README now reports only the most favorable
    run, so the comparison with the Stage 3 class looks closed when it is not.
  - Rate: the B32 p150x4 prefill is non-repeatable in every chain for both layer kinds. In chain 4, 1430 of 4128
    rows (35%) differ for the full layer and 1163 for the linear layer. The B32 baseline and p150 rows are
    repeatable in every chain. The README does not state this rate difference.
  - README says "Not caused by the Stage 4 changes". The evidence supports a narrower claim:
    - the BFP8-list path also drifts (`E4/nondeterminism/x4_full_b32_lists.json`), so the stacked kernel extension
      is not required;
    - unmodified `post_combine_reduce` drifts with identical captured inputs (`E4/nondeterminism/x4_full_b32_stacked.json`,
      run 1, first difference index 24);
    - the single-chip baseline is sometimes non-repeatable (`E4/len_base_linear_1000_commands.json`, 8 rows,
      0.00024; `MD/evidence2/len_base_full_2176_s128_commands.json`, 1090 rows, 0.0029; the README does not cite
      the second one).
    These do not exclude the EP context (64 local experts, concentrated routing, dummy slot at `capacity-1`) as the
    trigger of the much higher mesh rate. The only mesh standalone control
    (`MD/evidence2/nondeterminism/followup/moe_repeat_mesh*_x4.json`) uses uniform random routing and synthetic
    weights. The AutoDebug report calls this kind of control weak (`MD/evidence2/moe_nondeterminism_autodebug/AUTODEBUG.md`,
    I4 and 5.5) and says "Do not loosen tolerances or claim a fix until E1-E3 localize the stage" (section 7).
    None of E0-E6 ran. The chain 4 probe is byte-identical to the chain 2 probe, so the E0 probe defect
    (previous output kept alive, section 5.7) is still present.
  - The work log says the op outputs "drift 1-2 ULP". The layer-output `max_bf16_ulp` is 29568-29696 in the
    B32 p150x4 rows. That metric is probably dominated by values near zero, but the work log statement is not
    qualified.
  Why this matters: the prefill path of the layer-stack baseline is non-repeatable on p150x4 at B32 in 35% of rows,
  and the cause is unknown. Correctness gates pass (control PCC 0.99993 or more, HF PCC unchanged), so this does not
  block the stage by itself. But an anomaly classification must be accurate. A false bound and an unsupported causal
  claim become the basis for later stages (full model, serving determinism) to treat the issue as closed.
  Required next step: correct README and the anomaly ledger. State the full observed range across preserved chains
  (max abs up to 0.015625, up to 1430 of 4128 rows) and the mesh-only rate increase. Cite the evidence2 baseline row.
  Replace "Not caused by the Stage 4 changes" with what the evidence shows: the stacked extension is not required,
  the drifting ops are unmodified Stage 3 or upstream ops, and the mesh trigger is unknown. If the stage wants to keep
  the stronger claim, run AutoDebug E0 (fixed probe) and E2 (replay the captured inputs of one drifting call on a
  single chip and on the 1x4 mesh, at least 200 times) and record the result. Keep the issue open for Stage 5 with a
  named owner in either case.

## Review 1 Findings: Status

- Review 1 P2, nondeterminism: partly resolved.
  - Resolved: disclosure in README; per-op probe localizing the drift to `moe_fused_swiglu` and
    `post_combine_reduce` with identical captured inputs; BFP8-list control; p150 B32 rows added
    (`E4/b32_m1_{full,linear}`, bitwise equal to the baseline); AutoDebug inspection.
  - Not resolved: see the P2 above.
- Review 1 P2, B32 reporting: resolved.
  - README "Batch 32" table re-derived from `E4/b32_*`: decode 12615 / 10922 / 9978 us (full) and
    22227 / 17726 / 15216 us (linear); speedups 1.16, 1.26, 1.25, 1.46 match. Control PCC 0.99993 or more.
  - README "Batch dimension" classifies the per-row loop as inherited, states 2 x B collectives per layer per step,
    names Stage 5 and Stage 7 as owners, and estimates the B32 expert union. The B32 expert choice is deferred to a
    Stage 5 measurement. This is acceptable for Stage 4 because the Stage 3 control has the same per-row cost.
  - Small arithmetic point: 256 x (1 - (1 - 8/256)^32) = 163.3, not 162. The conclusion does not change.
- Review 1 P2, memory handoff recheck and construction peak: resolved.
  - `E4/memory_handoff_recheck.json` covers all 9 `required_future_work` items of
    `/home/hous/dev/qwen3.5-35b-a3b/memory-layout-handoff.json` (7 addressed, 2 open for the full model), has an
    ownership table, and its source hashes and ttnn patch hash equal the current files.
  - Construction samples (`construction_steps`) show device DRAM after each replica, after merge and after release.
    p150x4 linear: 364.2, 692.1, 1020.1, 1348.0, 1348.0, then 364.2 MiB. The replicas are released.
  - The higher setup transient inside each replica (`construction_transient_peaks`, from
    `E4/inventory/replica_peaks/`) is 1682 / 1835 MiB on p150x4 and 1976 / 1999 MiB on p150x2. README uses these.
  - Host peak RSS is recorded for p150x2 (6.51 GiB) and p150x4 (4.49 GiB).

## Other Concerns

- `E4/memory_handoff_recheck.json` is internally inconsistent. `profiles.*.construction_peak_device_mib` says 761.0
  (p150 linear) and 1348.0 (p150x4 linear). `construction_transient_peaks` says 2601.1 and 1681.9 for the same
  cases. README uses the second set. For p150, `construction_steps` is empty and `released_after_merge: true` is
  vacuous (no merge happens). p150 host RSS is not recorded. Label or remove the first set.
- The headline baseline log overwrite that review 1 reported is still present in chain 4.
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage04/evidence_chain4.sh` copies `$C/headline_base_$K/*.json` for
  both kinds into one directory. `E4/headline_base_logs/default.json` is the linear layer only (layer 0). The work log
  says this was fixed after chain 1. The full-layer files survive only in
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage04/evidence4_tensors/headline_base_full/` and in
  `E4/headline_base_full_commands.json`.
- `MD/work_log.md` is stale. Its header calls `evidence1/` the "final frozen chain", and it has no entry for the
  chain 4 run, its timestamps (07:32-08:45 UTC) or the `SOURCE_STABLE` result.
- README says "Repeated warmed prefill is bitwise identical in 70 of 75 rows (57 mesh-decoder rows, 18 baseline
  rows)". The artifacts give 56 mesh rows in headline, length, B32 and placement, or 64 with `watcher_long` and the
  profile runs. The 5 non-repeatable rows are correct. Fix the count or say how it was counted.
- README says host peak RSS "6.5 GB" and "4.5 GB". The JSON values are GiB.
- `E4/placement_p150x2_01_*/commands.json` and the headline p150x2 rows both record `device_ids` [1, 0] and no
  `TT_VISIBLE_DEVICES`. Only the chain script shows which card was used. The placement result (6.4 / 8.0 us slower
  on PCIe 0,1) depends on that script, not on the artifact.
- Profiled decode kernel sums are 2-5% larger than the traced host medians (p150x4 linear 522.6 against 498.3 us,
  `E4/profiles/m4_linear_attention/decode_replay_summary.txt`). This is consistent with profiler overhead. The
  README should say that profile times are not the headline times.
- Carried from review 1 and still open for later stages, correctly listed in README: p150 native context depends
  on per-mesh constant sharing and a measured reserve (`E4/capacity_model.json`, 1.32 sequences with 8 GDN slots,
  0.77 with 32); RoPE sharing; B32 batched decode; prefill dispatch bookkeeping; GDN in-projection and norms.

## Hard-Check Gaps

- No adversarial EP routing test. A prompt of one repeated token routes every token to the same 8 experts, so some
  ranks get zero local routes in `_ExpertParallelDispatch.route_maps`. The op skips experts with a zero count
  (`test_moe_fused_swiglu.py`, zero-count comment near line 249), so this is probably safe. No mesh test drives a
  whole rank with zero routes, or all routes on one rank, through the layer.
- The source manifest still hashes the patch text, not the built `_ttnn.so`. The build time (2026 Oct 11 02:25:47
  UTC) is before all four chains, and the ttnn patch is identical in all four, so lineage is consistent.
- Micro sweeps (`E4/micro/*.json`) do not carry source hashes. They are standalone op probes with their scripts
  copied next to them, and chain 4 confirms the selected configs on the full layer.
- Each headline row is one process and one timing run.

## Anomaly Ledger

- Observed anomaly: mesh prefill not bitwise repeatable; B32 p150x4 every chain, 709-1430 of 4128 rows, max abs
  0.0078-0.0156; p150x2 at 1000 tokens in some chains.
  Evidence: `MD/evidence{1,2,3}/b32_m4_*`, `E4/b32_m4_*`, `E4/len_m2_*_1000`, `E4/nondeterminism/*.json`.
  Affected path: multichip prefill, MoE expert compute and combine.
  Control or comparison: p150 and baseline B32 repeat in every chain; baseline single chip non-repeatable at
  1000 tokens (chain 4) and 2176 split 128 (chain 2); BFP8-list path drifts on the mesh.
  Likely subsystem: `moe_fused_swiglu` and `post_combine_reduce` behavior in mesh context; mechanism unknown.
  Investigation performed: per-op probe, list-path control, standalone op loops (single chip and mesh), AutoDebug
  source inspection. E0-E6 not run.
  Resolution: more-work-needed (classification must be corrected; see Required Work).
- Observed anomaly: B32 decode 29-31 x B1 per step.
  Evidence: `E4/b32_*`, `tt/functional_decoder.py` per-row `decode_forward`.
  Control or comparison: Stage 3 baseline has the same cost.
  Resolution: controlled (inherited, owner and plan recorded).
- Observed anomaly: layer 0 recorded-input positions, row 17 token 4, decode PCC 0.9939-0.9941 on all profiles
  (`real_positions_0_m*` exit 1 in the chain log).
  Evidence: `E4/real_inputs_m{1,2,4}/positions_0.json`.
  Control or comparison: Stage 3 single chip 0.99393.
  Resolution: controlled.
- Observed anomaly: 1-3 prefill tokens per 4096 below 0.995 against HF.
  Evidence: headline `prefill_tokens_below_0995`, `E4/micro/router_margin_layer0.json`.
  Control or comparison: identical counts in the baseline; exact 8th/9th router logit ties.
  Resolution: controlled.
- Observed anomaly: p150x4 watcher build fails without `TT_METAL_WATCHER_NOINLINE=1`.
  Evidence: `MD/evidence1/accept_watcher_m4/linear.log`.
  Resolution: fixed. 16 chain 4 watcher logs are distinct files (different hashes), all from completed runs, with
  no error, assert or sanitizer line.
- Observed anomaly: memory recheck JSON has two different "construction peak" values per profile.
  Evidence: `E4/memory_handoff_recheck.json`.
  Resolution: documentation fix (Other Concerns).

## Verified Claims

- Chain 4 (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage04/evidence_chain4.log`): every step exits 0 except the
  three expected `real_positions_0_m*` gate failures; `SOURCE_STABLE` and `CHAIN_DONE` are present.
- `E4/source_manifest_start.json` equals `source_manifest_end.json`. Every hashed `tt/` and `tests/` file matches
  the current worktree. `git_diff_outside_autoport` (744236997d8a) matches the current diff. `E4/ttnn_patch.diff`
  matches `git diff HEAD -- ttnn`.
- Headline table re-derived from `E4/headline_*`: prefill speedups 1.01 / 1.18 / 1.31 (full) and
  1.01 / 1.30 / 1.56 (linear); decode speedups 1.00 / 1.15 / 1.25 and 1.00 / 1.23 / 1.42; efficiency 31% and 36% at
  p150x4. 40-layer decode estimates 25.62, 21.01 and 18.41 ms are correct sums.
- p150 is bitwise equal to the Stage 3 baseline in all 18 rows (`prefill_identical` and `decode_identical` true,
  max abs 0).
- All 42 length rows exit 0 on three profiles. Minimum control PCC per group matches the README table.
- Native context 262143 prefill plus decode and 262144 prefill pass on all three profiles. `context_contract.json`
  only adds `multichip_decoder_validation`; earlier sections are unchanged in content. `capability_reduction` is null.
- Reuse (18 cases, page 32 and 128), positions, guards, device pytest (12 passed), CPU tests (1154 passed),
  stacked op test (3 passed) all pass in chain 4.
- Decode `sparse_matmul` gate and down are swept independently under the selected BFP4 / BFP8 LoFi policy, over all
  legal core counts and k blocks (`E4/micro/sparse_geometry_sweep.json`). Selections equal the best rows.
- Projection sweep (`E4/micro/projection_geometry_sweep.json`) uses the selected dtype and fidelity per row
  (BFP8 LoFi qkvg, BF16 HiFi4 GDN in-projection, BFP4 LoFi shared gate/up). Selected configs equal the fastest legal
  rows, except the x2 qkvg DRAM-sharded row, which is rejected with a recorded memory reason.
- No code comments were added by the Stage 4 diff (only `#ifndef` / `#define` preprocessor lines in kernels).

## Scope Inspected

- Goal/skill paths: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/04-04-multichip-decoder.prompt.txt`,
  `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`,
  `/home/hous/dev/qwen3.5-35b-a3b/memory-layout-handoff.json`, review 1 at `MD/stage_review.md`.
- Artifact paths: `MD/README.md`, `MD/work_log.md`, `E4/` (manifests, headline, len, b32, accept, real_inputs,
  watcher_long, placement, inventory, replica_peaks, memory_handoff_recheck, capacity_model, nondeterminism,
  profiles, micro sweeps, tables), `MD/evidence{1,2,3}/` repeat-difference rows, `MD/evidence2/nondeterminism/`,
  `MD/evidence2/moe_nondeterminism_autodebug/AUTODEBUG.md`,
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage04/evidence_chain4.{sh,log}`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/context_contract.json`.
- Code paths: `tt/multichip_decoder.py` (full), diffs of `tt/functional_decoder.py`, `tt/functional_gdn.py`,
  `tests/test_multichip_decoder_device.py`, `tests/run_optimized_trials.py::source_manifest`, attention_out policy in
  `tt/optimized_decoder.py`, ttnn diff for added comments.
- Commands run: `git status`, `git diff`, `git show`, `sha256sum`, `cmp`, `md5sum`, `grep`, and Python JSON
  analysis of existing artifacts. No device, pytest, server or build.

## Residual Risk

- The mesh-only prefill nondeterminism can make greedy outputs differ between identical requests once 40 layers
  are stacked. Stage 6 and the serving stages need a repeatability check at the full-model level.
- TP and EP correctness rests on control PCC and replica identity, not bitwise equality. Rare routing patterns
  (empty rank, one hot rank) are not explicitly tested.
- p150 capacity is model-based with a small margin. The full model must measure it.
