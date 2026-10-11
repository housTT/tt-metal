# Stage Review

Stage 4, multichip decoder. Qwen/Qwen3.5-35B-A3B at revision 59d61f3ce65a6d9863b86d2e96597125219dc754.
Reviewer: fresh independent subagent. Review date: 2026 Oct 11. No device or server was used.

Abbreviations: TP (tensor parallel), EP (expert parallel), MoE (mixture of experts), GDN (Gated DeltaNet),
CCL (collective communication), PCC (Pearson correlation coefficient), HF (Hugging Face reference),
B1/B32 (batch 1 / batch 32), DRAM (device memory), L1 (core SRAM), BFP4/BFP8 (4/8-bit block float).

Verdict: more-work-needed

## Required Work

- P2: Multichip prefill nondeterminism is not disclosed and not classified. A new p150x4 instance is larger than the Stage 3 class.
  Evidence:
  - `evidence1/b32_m4_full/commands.json`: `prefill_repeat_identical` false. 953 rows and 160274 elements differ between two
    warmed prefills. `max_abs` 0.015625. Repeat PCC 0.9999986.
  - `evidence1/b32_m4_linear/commands.json`: false, 838 rows, `max_abs` 0.0078125.
  - `evidence1/len_m2_full_1000`, `len_m2_linear_1000`: false (5 and 2 blocks of 32 rows).
  - The single-chip controls with the same inputs are bitwise repeatable: `b32_base_full_commands.json` and
    `b32_base_linear_commands.json` are true. `b32_m2_*` are true. No `b32_m1` row exists, so the stacked
    `moe_fused_swiglu` path on p150 is not tested at B32.
  - Stage 3 records a prefill nondeterminism with an unresolved root cause and a maximum of 0.0078 anywhere
    (`../optimized_decoder/README.md`, determinism paragraph). The b32_m4_full difference is two times that maximum and
    affects many whole 32-row blocks. Its baseline is repeatable.
  - README "Trace, determinism, audit, watcher" states only that headline prefill repeats are identical. It does not
    mention the four non-identical rows or the Stage 3 limitation.
  Why this matters: the p150x4 prefill path contains new code: the stacked and mixed-dtype `moe_fused_swiglu` kernels,
  the EP route maps (`_ExpertParallelDispatch.route_maps`), the dummy-slot `slice_write`, and ring all-reduce. A
  nondeterminism that appears only on the mesh, and is absent from the single-chip control with the same inputs, can
  come from a race in this new code. It is a correctness risk in the layer-stack baseline, even though the control PCC
  is still 0.99993 or more.
  Required next step: disclose the four rows in README and in the anomaly ledger. Isolate the source. Repeat the B32
  129-token prefill on p150x4 and compare per-op outputs between repeats: route maps, dispatch map, `moe_fused_swiglu`
  output, restore table, combine, and all-reduce. Show either the Stage 3 mechanism (`post_combine_reduce`
  state dependence) or the new-code cause, and fix the new-code cause if one exists. Use `$autodebug` or `$autofix`
  if the first isolation does not settle it. Add a p150 B32 row so that the stacked kernel path is also covered.

- P2: B32 decode latency is measured but not reported. The mesh plan is justified only at B1.
  Evidence:
  - Traced decode per layer and step, B32, 129 tokens (`b32_*`): baseline full 12606.7 us, linear 22254.5 us. p150x2
    12300.9 / 18765.1 us. p150x4 11308.5 / 16511.5 us. B1 values are 390.9 to 708.8 us. B32 is 29 to 31 times B1, and
    the p150x4 B32 speedup is 1.11x (full) and 1.35x (linear).
  - Root cause from code: `FunctionalDecoder.decode_forward` (`tt/functional_decoder.py:468-476`) runs the whole layer
    once per batch row. Each row also runs both `_AllReduceOutput` collectives, which is 64 all-reduces per layer per
    step at B32.
  - The MoE decode choice (expert-TP 69 us against EP 195-248 us, `micro/moe_ep_decode.json`) is made for one token.
    At B32 the union of selected experts approaches all 256. Expert-TP then costs about 32 x 61 us per layer, which
    reverses the comparison. The README does not analyze this.
  - The project contract requires client concurrency up to 32 in the final sweep (`/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`).
  Why this matters: this decoder is the baseline for the full layer stack. At about 0.6 s per B32 decode step for 40
  layers on p150x4, the plan does not match the required serving load. The B32 numbers exist in the evidence but are
  not in the README, and the anomaly is not classified.
  Required next step: report the B32 latencies, speedup and efficiency in README. Classify the per-row loop as
  inherited (the Stage 3 control has the same cost) and name the owner and plan for batched decode. Add the batch
  dimension to the mesh plan: compare expert-TP and EP at B32 with the existing micro numbers, and state how CCL count
  scales with batch.

- P2: The memory handoff recheck and the construction peak for the replica-merge path are missing.
  Evidence:
  - `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md` requires reading `memory-layout-handoff.json` before the multichip
    composition, and rechecking it after implementation changes. Stage 3 left
    `../optimized_decoder/final_evidence15/memory_handoff_recheck.json`. Stage 4 has no equivalent, and README and
    `work_log.md` never mention the handoff.
  - The handoff lists "record post-construction allocator DRAM and per-buffer identities, host RSS and setup peaks"
    and an ownership table as required future work. `evidence1/inventory/*.json` records only `after_merge` and
    post-construction allocator values.
  - `MultichipDecoder.from_state_dict` builds `tp` complete replicas, each replicated on every device, before
    `merge_replicas` (`tt/multichip_decoder.py`). The per-device peak during construction is therefore about `tp` times
    the local layer, and `merge_replicas` reads every tensor of every replica to host. Neither the device peak nor the
    host peak is measured.
  Why this matters: the full model builds 40 layers in sequence on top of the resident weights of earlier layers. The
  release of replicas 1..tp-1 depends on Python reference cleanup, and a past bug kept a replica alive
  (`work_log.md`, "loop variable kept last replica"). One layer's after-merge number does not prove that the
  construction peak and the release are bounded.
  Required next step: record a Stage 4 handoff recheck with current source hashes, the handoff items now addressed
  (local-expert sharding, single expert copy on p150), the items still open (per-mesh constant sharing, RoPE sharing),
  and an ownership table for the decode expert-TP copy and the prefill EP copy. Measure the device allocator high-water
  mark and the host RSS during construction for each profile, and confirm that replica buffers are released after
  the merge. This can be added to the next device run.

## Other Concerns

- p150 native context depends on full-model work that is not yet done. Per-layer constants that are not already cached
  globally are about 19.6 MiB per linear layer and 81.5 MiB per full layer on p150 (`inventory/m1_layer{0,3}.json`;
  `_PREFIX_MATRICES` is already shared). If the layers are stacked as built, the excess over the "counted once"
  assumption is about 1.25 GiB. The 8-slot margin above one 262144-token sequence is (345027 - 262144) x 10880 B, or
  about 0.84 GiB (`capacity_model.json`). The 2 GiB reserve is an estimate, and the 32-slot harness default gives 0.77
  sequences. `context_contract.json` records these conditions. Stage 6 must implement per-mesh constant sharing and
  measure the reserve before p150 can claim native context.
- The decode `sparse_matmul` geometry sweep at p150x4 is thin. Gate and down configurations are swept only as fixed
  pairs, and only two x4 pairs ran (`runs/main/stage04/micro/moe_split.json`). Down at 32 or 64 cores with gate at 8
  cores was never measured. The down N=2048 shape is the same as at x2, where 64 down cores won. The routed experts take
  61 us of each decode layer. Stage 5 must sweep gate and down independently under BFP4 and BFP8 LoFi.
- tt-perf-report rows not reviewed in README: shared-expert gate/up `MinimalMatmul 32x2048x256` takes 19 us on 110
  cores at 4% DRAM (SLOW), and the router `32x2048x288` HiFi4 on 9 cores takes 8-11 us (SLOW)
  (`profiles/m4_*/decode_perf_report.csv`). The GDN in-projection DRAM-sharded BF16 candidate is called "now fits at
  N=3136" but is not measured. These are Stage 5 items. Stage 4 README should list them.
- The p150x2 gain is small: full decode 1.03x, linear 1.17x. Per-device weights on p150x2 (596.5 / 578.8 MiB) are
  almost equal to p150 (629.9 / 594.8 MiB), because p150x2 holds two half-size expert copies. The README states this,
  but the capacity benefit of p150x2 comes only from KV and state.
- p150x4 full 2176-token prefill (24.4 ms) is slower than p150x2 (21.7 ms) and slower than p150x4 at 4097 tokens
  (21.6 ms). p150 shows the same pattern (23.6 ms at 2176 against 26.4 ms at 4097), which points to the inherited
  bucket planner. The README labels it "single sample, open for Stage 5". It is a performance anomaly, not a
  correctness one, and Stage 5 must reproduce it and explain it.
- Bug in an error path: `tt/functional_gdn.py` now raises `ValueError(f"... {actual} != {expected}")`, but this stage
  removed `expected`. A failing dimension check raises `NameError` and not the intended message.
- The new `moe_fused_swiglu` modes (`stacked_intermediate`, column offsets, separate down dtype) have no in-tree op
  test. The only evidence is `evidence1/micro/stacked_moe_test.py` and its JSON. The program hash includes the new
  attributes (`moe_fused_swiglu_types.hpp`), which is correct.
- README says "Watcher: 22 logs, no errors". Five of the 22 logs come from runs that failed at program load
  (`accept_watcher_m4/watcher_linear`, `watcher_long/logs_*`). They trivially have no errors. The 17 logs from completed
  runs are clean. A `-DWATCHER_NOINLINE` compile line in `watcher_m4_retry/group/full.log` and
  `noinline_linear.log` confirms the retry.
- README says "p150x2 on PCIe 2,3", but every p150x2 artifact records `device_ids` [1, 0] (logical IDs under
  `TT_VISIBLE_DEVICES`). Only the chain command shows the physical binding. The full model and the server must also
  set `TT_VISIBLE_DEVICES` for p150x2. This is an integration constraint for Stages 6 and 9.
- The p150x4 ring includes physical device 0. Stage 3 measured device 0 as 16-23 us slower on decode replay
  (`../optimized_decoder/README.md`, device placement), and no p150x4 placement control exists.
- The README cites `router_margin_layer0.json` for "913 of 4096 tokens tie". The JSON does not store the count. It
  stores `fraction_tokens_gap_below_1e-2` = 0.2229, and 0.2229 x 4096 = 913, so the claim can be derived.

## Hard-Check Gaps

- The source manifest hashes the patch text (`git_diff_outside_autoport`), not the built `_ttnn.so`. The build
  timestamp (2026 Oct 11 02:25:47 UTC) comes after the patched op tests and the restore build, and before the chain at
  02:55 UTC, so the lineage is consistent. It is not hash-bound.
- `tests/weight_inventory.py` changed after the chain (its current hash differs from `source_manifest_start.json`).
  The inventory JSONs record only the `optimized_decoder.py` hash, which matches. All other implementation files
  still match the chain manifest.
- The acceptance, positions and context gates compare against HF only. Single-chip equivalence for those cases comes
  from Stage 3 numbers quoted in README, not from a same-run control.
- Headline comparisons are one process per row, and timing repetition is one run per row.

## Anomaly Ledger

- Observed anomaly: prefill repeats not bitwise identical on meshes (b32_m4 max abs 0.0156, 953 rows).
  Evidence: `evidence1/b32_m4_{full,linear}/commands.json`, `len_m2_{full,linear}_1000/commands.json`.
  Affected path: multichip prefill (EP dispatch, stacked `moe_fused_swiglu`, all-reduce).
  Control or comparison: `b32_base_*` repeatable. Stage 3 known nondeterminism up to 0.0078.
  Likely subsystem: unknown. Either the inherited `post_combine_reduce` state dependence or the new Stage 4 code.
  Investigation performed: none in Stage 4.
  Resolution: more-work-needed.
- Observed anomaly: B32 decode 29-31 times B1 per step. Multichip gives only 1.11x / 1.35x at B32.
  Evidence: `evidence1/b32_*`. `tt/functional_decoder.py:468-476`.
  Affected path: decode for every batch above 1.
  Control or comparison: the Stage 3 single-chip B32 control has the same scale.
  Likely subsystem: serial per-row decode loop that repeats both CCLs per row.
  Investigation performed: none recorded. The reviewer found the cause by code inspection.
  Resolution: more-work-needed (report and plan). It is inherited, but it is not classified.
- Observed anomaly: layer 0 positions row 17 token 4 PCC 0.994.
  Evidence: `real_inputs_m{1,2,4}/positions_0.json`.
  Control or comparison: Stage 3 single chip 0.99393 (`final_evidence15/real_inputs/positions_0.json`). The BF16 HF
  reference itself deviates from FP32 (Stage 3 README).
  Resolution: controlled.
- Observed anomaly: 1-3 prefill tokens below 0.995 against HF.
  Evidence: headline `prefill_token_pcc_min`, `router_margin_layer0.json` (exact 8th/9th logit ties at tokens 950, 637,
  1720, 3407).
  Resolution: controlled.
- Observed anomaly: the p150x4 watcher build fails to load (ERISC router 29264 B is larger than 26624 B).
  Evidence: `accept_watcher_m4/linear.log`. Retry with `TT_METAL_WATCHER_NOINLINE=1` under the same manifest
  (`watcher_m4_retry/source_manifest.json` equals the chain start manifest).
  Resolution: fixed. All p150x4 watcher cases are clean.
- Observed anomaly: p150 against baseline, linear 1000 tokens, prefill not bitwise identical (max abs 0.00049).
  Evidence: `len_m1_linear_1000`. The baseline itself is not repeatable in that run
  (`len_base_linear_1000_commands.json`, `prefill_repeat_identical` false, the same 32-row block size).
  Resolution: controlled by the baseline. It is not a stacked-weight difference.
- Observed anomaly: p150x4 2176-token prefill slower than p150x2 and slower than 4097 tokens.
  Evidence: `len_m4_full_2176`, `len_m2_full_2176`, `len_m4_full_4097`.
  Control or comparison: p150 shows the same 2176 against 4097 pattern.
  Resolution: open for Stage 5 (performance only).

## Verified Claims

- Headline numbers re-derived from `headline_*/commands.json`: speedups 1.02/1.00, 1.19/1.03, 1.24/1.11 (full) and
  1.01/1.00, 1.30/1.17, 1.56/1.35 (linear). Control PCC 0.99996-1.0 for prefill and 0.99845 or more for decode. 40-layer
  estimates 25.61 ms and 19.70 ms. Efficiency 28% / 34% at p150x4.
- All length rows (3, 100, 1000, 2176, 2176 split 128, 4095, 4097) exit 0 on all three profiles. Control PCC is
  0.99935 or more. Output replicas are identical everywhere.
- The chain source manifest is stable from start to end. The current implementation files and the ttnn patch match it.
  The only exception is `tests/weight_inventory.py`.
- Trace replay uses `TT_METAL_TRACE_ALLOC_TRACKING=1` and `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`
  (`run_optimized_trials.py:72`, chain commands). Program-cache misses are forbidden during capture. Only the output is
  acknowledged as corruptible. Repeated replays are identical.
- TP slicing checked by code inspection: query heads `rank*16/tp`, KV head `rank*2//tp` (correct GQA group of 8),
  GDN key/value heads consistent with the repeat factor 2, a/b/z/dt/A_log/conv sliced by local heads, the per-head
  norm replicated, and o_proj/out_proj/shared down sliced on input columns. The router and shared gate are replicated.
  The sigmoid shared-gate scale is linear over partials.
- EP dummy slot: `capacity >= routes + local*31 + 32` keeps slot `capacity-1` outside every expert region. Its restore
  row is zeroed with `slice_write`, so non-local routes add zero. The integer arithmetic in route maps is exact
  (counts at most 4096 in FP32, tile counts at most 128 in BF16, slots below 65536).
- The fused-reduce `cluster_axis` bug was found, explained from kernel source, and fixed
  (`bringup/x2_layer*_before_cluster_axis.json`). Single-chip outputs are bitwise identical before and after the edits
  (`control_pre_edit`, `regression_post_edit`).
- `moe_fused_swiglu` op tests: 281 of 282 pass. The one failure also fails on the unmodified build
  (`op_tests/hybrid_ring_baseline.log`). The test patch equals `evidence1/ttnn_patch.diff`.
- The profiler runtime root `/tmp/q35-functional-profiler-root` is a symlink to the same checkout.

## Scope Inspected

- Goal/skill paths: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/04-04-multichip-decoder.prompt.txt`,
  `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`, stage-review and multichip `SKILL.md` under
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/`,
  `/home/hous/dev/qwen3.5-35b-a3b/memory-layout-handoff.json`,
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/05-05-optimized-multichip-decoder.prompt.txt` (stage boundary).
- Artifact paths: `doc/multichip_decoder/{README.md,work_log.md}`, `evidence1/` (manifests, headline, len, b32,
  placement, accept, real_inputs, watcher, watcher_m4_retry, profiles, inventory, capacity_model, micro,
  router_margin), `runs/main/stage04/{evidence_chain1.sh,evidence_chain1.log,micro,bringup,control_pre_edit,regression_post_edit,op_tests}`,
  `doc/context_contract.json`, `doc/optimized_decoder/README.md` and `work_log.md`.
- Code paths: `tt/multichip_decoder.py`, and the diffs of `tt/optimized_decoder.py`, `tt/fused_decoder.py`,
  `tt/functional_decoder.py`, `tt/functional_gdn.py`, `tests/run_layer.py`, `tests/test_multichip_plan.py`, and the
  `moe_fused_swiglu` device operation and types.
- Commands run: `git status`, `git diff`, `sha256sum`, `cmp`, and Python JSON/CSV analysis of existing artifacts. No
  device, pytest, server or build was run.

## Residual Risk

- The correctness of TP and EP rests on control PCC and on replica identity, not on bitwise equality. A rare
  routing-dependent error in the EP route maps would need adversarial routing (all tokens on one rank, empty ranks) to
  show. Neither case is tested explicitly.
- p150 capacity margin is small and model-based. The first full-model allocation on p150 may force a slot or reserve
  decision.
- Decode on p150x4 is latency-bound on replicated small ops (norms, eltwise, reshape/typecast: about 250 us of the
  553 us linear profile). The 1.30x 40-layer gain will not grow much without Stage 5 work on these ops.
