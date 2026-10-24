# Stage Review

Verdict: more-work-needed

Stage 7, optimized full model, third review. Reviewer: independent stage-review subagent, read-only (no device,
server, pytest or process action). Review time: 2026 Oct 24 10:13 Eastern.

Source under review: live worktree, branch `feat/qwen3-5-35b-a3b`, HEAD `25bbbe3c` plus the uncommitted diff in
`tt/functional_decoder.py`, `tt/model.py`, `tt/optimized_decoder.py`, docs and evidence. I recomputed the chain
`manifest()` over the autoport `tt/` and `tests/` trees. All 90 lines equal
`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence5/source_manifest_end_m1b.txt`.
The chain 5 and chain 5b snapshots differ only by fix5 (`PREFILL_PROGRAMS`). The fix5 trial snapshot and patch
equal chain 5b. No file outside the autoport is dirty.

Abbreviations: B (decode batch, active rows in one decode step), TTFT (time to first token), t/s/u (tokens per
second per user), HF (Hugging Face CPU reference), TT (Tenstorrent TTNN implementation), TTNN (Tenstorrent neural
network library), KV (key/value cache), GDN (Gated DeltaNet linear-attention layer), MoE (mixture of experts), EP
(expert parallel), TP (tensor parallel), LM head (final vocabulary projection), BF16 (16-bit brain float), FP32
(32-bit float), DRAM (device memory), L1 (per-core SRAM), SRAM (static RAM), PCC (Pearson correlation
coefficient), KL (Kullback-Leibler divergence, HF to TT, nats), TV (total variation distance), CCL (collective communication library), CPU (host
processor), ISL (input sequence length), KiB and MiB (2^10 and 2^20 bytes), MB and GB (10^6 and 10^9 bytes). P1 is
a correctness or contract gap that blocks the stage. P2 is required work with lower correctness risk. p150, p150x2
and p150x4 are the 1, 2 and 4 chip profiles.

Summary: both review 2 work items are resolved in code and evidence. I re-derived the headline table, the batched
table, the accounting table, the KL tables, the qualitative comparison, the HF gates and the context numbers from
raw JSON, CSV, logs and saved logits. They match the README. Two required artifacts contradict the final evidence.
Both fixes are documentation only and need no hardware run.

## Required Work

- P2: The memory handoff recheck omits a new persistent L1 allocation, and `perf_summary.json` carries stale
  evidence4 text
  Evidence:
  - L1 allocator after build and `prepare()` (`memory_after_build.l1.allocated_bytes`, per device): stage 6,
    evidence3 and evidence4 have 5800960 (p150x4), 3998720 (p150x2), 1802240 (p150). evidence5 has 9405440,
    7603200, 4505600. Delta: +3604480, +3604480, +2703360 bytes. Largest free L1 per bank: 1386112 -> 1351296
    (p150x4), 1438592 -> 1414016 (p150). Sources:
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence5/m4/full_evidence.json`,
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence4/m4/full_evidence.json`,
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence5/m4/full_evidence.json`
    and their m2 and m1 siblings (the `batch_perf_sampled0.json` files agree).
  - The step appears at fix2: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/projfix/m4_greedy.json` 5800960;
    `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/fix2/batch_perf_sampled0_m4.json` 9405440; fix2 p150 5406720;
    `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/fix4/batch_perf_m4.json` (buckets 1, 4, 32 only) 7603200;
    `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/fix5/batch_perf_m1.json` (buckets 1, 32 only; B32 uses the
    prefill top-k) 1802240.
  - Inferred source (exact byte match, not a buffer listing; unverified by an allocator dump): fix2 sends batched
    rows through the decode branch of `_OptimizedGateModule.route`
    (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py:981`).
    That branch caches 4 tensors per new width (`bias`, `indices`, `out`, `out_idx`), height sharded in L1 with one
    32x32 tile per core (`_sharded_mem_config`,
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/common/modules/moe/tt_moe_gate.py:458`). The L1 allocator
    reserves each sharded page on all 110 banks. Widths 4, 8, 16, 32: 16 x 2048 x 110 = 3604480. p150 (B32 uses
    the prefill top-k), widths 4, 8, 16: 12 x 2048 x 110 = 2703360. fix4 (widths 4, 32): 8 x 2048 x 110 = 1802240.
    All three match exactly. The census `layer_private` L1 grew by 491520 (60 cores x 4 x 2048) and 229376 (28 x 4 x
    2048): shard bytes only.
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/memory_handoff_recheck.json`
    item [1] says "Stage 7 adds only the batched-decode slot selector constants". `stage7_allocations` lists the
    selectors and "batched GDN decode: no persistent allocation". All deltas are DRAM. The README reports DRAM
    deltas only. The census does not reconcile L1 at all (stage 6 p150x4: allocator 5800960 against 1433600 named).
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/perf_summary.json`
    `named_limitations` is hard-coded in `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/write_perf_summary5.py`
    lines 33 to 40 with evidence4 values: "1.21x (p150x4), 1.45x (p150x2), 3.26x (p150)", "p150 B32 GDN-type layer
    6752 us", "device 2.1% above; profiler replay times rise from 9.26 to 10.01 ms". The final
    `evidence5/perf_accounting.json` gives 1.28x, 1.54x, 3.36x, 6612 us, device 2.5% above end to end, and replays
    9017 to 9776 us. The README uses the final values. The same list says "B1 reaches 17%" where the README says
    16% (0.1645).
  Why this matters: `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md` requires the memory handoffs to be rechecked after
  implementation changes and asks for whole-model persistent L1 placement per profile. The recheck now states a
  false invariant (only selectors added). Later stages size L1 circular buffers and serving buffers against this
  headroom. `perf_summary.json` is a required `$optimize` artifact. Its named limitations contradict the README and
  the numeric fields of the same file.
  Required next step (documentation only, from existing evidence):
  1. In `memory_handoff_recheck.json`, correct item [1] and add the router cached-slice allocation to
     `stage7_allocations`: widths per profile, 4 tensors per width, 2048 bytes per bank per tensor, 32 KiB per bank
     on p150x4 and p150x2 and 24 KiB on p150, shared across layers, created in `prepare()` before any capture, and
     its consumer (`generalized_moe_gate` at batched rows). Add per-profile L1 allocator deltas against stage 6 and
     state that the census does not reconcile sharded L1.
  2. Add one line for the L1 delta to the README memory row.
  3. Generate `perf_summary.json` `named_limitations` from `evidence5/perf_accounting.json`, or correct the text.

## Other Concerns

- Review 2 status, item by item:
  - Required item 1, sub-step 1 (router program config): done. p150x4 B32 router linear 7.0 us on 9 cores (was
    23.2 us); `GeneralizedMoeGate` 2.7 us; no `TopK` on the slot path. Raw ops CSV:
    `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/evidence5_work/named_profile_b32_m4/reports/2026_10_24_08_16_05/ops_perf_results_2026_10_24_08_16_05.csv`.
  - Sub-step 2 (p150 `attention_in`): done. GDN `32 x 2048 x 12352` 71.1 us on 97 cores, full `32 x 2048 x 9216`
    54.3 us on 96 cores (were 142.5 and 110.4 us `minimal_matmul`). Matmul total per replay rose 1761 -> 1889 us,
    so the net gain is about 125 us per 2-layer replay. The README cites only the `MinimalMatmul` drop
    (301 -> 49 us).
  - Sub-step 3 (index sticks once): done. `UntilizeWithUnpadding` 122 -> 0 us.
  - Sub-step 4 (combine without 32 rows per slot): partly. Scale before down removed the 92 us multiply
    (`BinaryNg` 229 -> 66 us). Down output and `fast_reduce_nc` still run over `[1,256,32,2048]` per layer
    (`FastReduceNC` 36 to 38 us per layer, about 1.5 ms of the 47.97 ms B32 step). The README records "kept" with
    no measured alternative. The residual cost is small, so I accept it.
  - Sub-step 5 (packed hidden): done. `ReshapeView` 175 -> 66 us.
  - Sub-step 6 (B > 1 topology table): done. I re-derived every p150x4 and p150x2 row from
    `evidence4/named_breakdown.json` and `evidence5/named_breakdown.json`.
  - Reruns: every affected check ran on every profile with the tracker (chain 5, 48 of 48 rc 0; chain 5b, 16 of
    16 rc 0; both `SOURCE_STABLE` with full-manifest `cmp`).
  - Required item 2: disposition table and recheck exist. The recheck source hashes equal the final source (12 of
    12). L1 omission above.
  - Review 2 other concerns: early-stop semantics documented; "bitwise" wording corrected; HF control cited;
    manifest compare defect fixed. Still open by design: force-argmax measured on p150x4 only; p150 per-row GDN;
    `generic_op` upstream issue not filed; CPU suite not rerun; DRAM-sharded LM head not measured in model on
    p150x2.
- Pre-fix5 identical-row divergence: the mechanism is not identified. Fix5 restores the evidence4 program
  selection inside the padded dispatch branch. I confirmed that `prefill_rows` equals the old `length > 1` test
  there, and that B1 decode and prefill take unchanged branches (B1 tokens, non-aligned tokens, long-prompt tokens
  and continuation PCC equal evidence3 bit for bit). Post-fix rows are identical and B32 KL is 0.0071. But why the
  decode gate inside the dispatch made identical rows differ, already in one decode step
  (`evidence5/m1_pre_fix5/sampling_audit.json`, `distribution.host_rows_identical` false), is not explained. One
  candidate (unverified): the decode gate returns `(T,1,1,8)` while `ttnn.topk` returns `[1,1,T,8]`, and a
  layout-dependent consumer in the dispatch reads them differently. The slot path consumes the decode gate through
  logical reshapes. Its rows are independent on current evidence: B16 row 0 logits equal B32 row 0 bit for bit on
  both meshes (my `torch.equal` on the saved logits), and B8 rows equal B32 rows. Record the mechanism when
  convenient. The p150 B17 to B32 dispatch branch now depends on a special program rule.
- p150 per-row GDN remains the dominant p150 batched cost. B32 GDN-type layer 6612 us against 1970 us for a
  full-type layer; 30 GDN layers are about 198 ms of 215 ms. The B8 to B16 increment (49.5 ms for 8 rows, 6.2 ms
  per row) equals 30 x 207 us of per-row GDN. Two 16-head calls (64 cores) are still untried (third review in a
  row). p150 is not the optimization target, so this is not required here. Stage 13 runs concurrency 32 on p150 at
  about 4.7 t/s/u.
- p150 `moe_slot_rows` 16 rests on a pre-fix2 measurement (slot 235.97 against dispatch 222.11 ms,
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/union/slot_perf_m1.json`). From the projection fix to fix3 the
  p150 slot path gained 12.1% at B16, and the dispatch path 3.8% at B32. Per-row GDN dominates, so the expected
  difference is a few ms. Re-measure when convenient. The slot path at B32 would also remove the dispatch special
  case above.
- `AllReduceAsync` "7.9 -> 13.4 us per call" is a mean of a bimodal distribution. In the evidence5 p150x4 B32
  replays most calls take 7.6 to 9 us, but the last all-reduce of the replay reaches 20 to 55 us on all four
  devices in sessions 1 to 3. evidence4 device 0 was flat at 7.7 us. This is peer skew before that call.
  Controlled by end-to-end time. Wording only.
- Calibration "100 teacher-forced decode positions" are 1 prefill position plus 99 decode positions
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_calibration_probe.py`
  lines 45 to 52). The B1 calibration counts the same way. `max_row_spread` is 0 by construction with
  `--distinct 1`, because only row 0 equals the reference sequence.
- p150 B32 profiled replay drift (9017 to 9776 us; layers (0,1,3) 14731 to 16904 us) is attributed to profiler
  overhead. The dispatch path is data-dependent (sampled B32 +3.3%), so step-dependent routing is an alternative
  explanation. It affects the device estimate only.
- p150x4 B32 host gap is -0.54 ms (device 1.1% above end to end) with no replay drift (2754 to 2730 us). It is
  inside the method error. The README explanation ("the profiler adds time") is shown only for p150.
- The batched "distinct" rows (calibration rows 1 to B-1, batch perf rows) are phases of the 293-byte readiness
  prompt repeated 400 times. The 262100-token long-context run uses the same repeated text. These runs test
  mechanics and capacity, not long-range retrieval. Inherited from stage 6.
- Checkpoint commit: `*.log`, `*.csv` and `*.pt` are ignored (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/.gitignore`
  lines 7, 8 and 99). Earlier checkpoints force-added CSV and log evidence; do the same for the evidence4 and
  evidence5 tables and logs. HEAD `25bbbe3c` holds a truncated
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence3/m1/full_evidence.log`
  (committed during chain 3c); the worktree copy is complete. Keep the 61 MB `batched_l2_watcher.pt` files and the
  untracked stage 6 `ops_perf_results.csv.gz` files out.

## Hard-Check Gaps

- The pre-fix5 divergence passed every scripted batched check: full evidence B32 (8 steps, KL to B1 maximum
  0.026, 255 of 256 argmax), sampling audit (48 steps, same-prompt check true) and calibration (row 0 only). Only
  the 128-token qualitative same-prompt comparison caught it. The sampling audit records
  `distribution.host_rows_identical` (false before fix5, true after, true on every evidence4 and evidence5 profile)
  but does not assert it. Make it a check.
- The census reconciles DRAM only. It does not see the all-bank reservation of sharded L1 buffers.
- The CPU suite last ran on the chain 3 source
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence3/cpu_tests.log`).
  Its four files do not cover `decode_forward` or `Qwen35Model.decode`, so the risk is low.
- No B4, B8 or B16 profile exists. Small-B costs are inferred from B32 rows.
- Batched HF calibration has no top-100 metric in the evidence. From the saved logits I computed HF top-1 inside TT
  top-100 at 100 of 100 positions for every B16 and B32 case (worst rank 3, p150 B16).

## Anomaly Ledger

- Observed anomaly: persistent L1 allocation grew 32 KiB per bank (p150x4, p150x2) and 24 KiB per bank (p150).
  Evidence: `memory_after_build.l1` in stage 6, evidence4 and evidence5 full evidence; fix2, fix4 and fix5 trial
  JSON.
  Affected path: whole model, L1 headroom for every program.
  Control or comparison: evidence4 and the fix5 trial (no batched decode gate) at the stage 6 value.
  Likely subsystem: router `cached_slices` at batched widths (fix2).
  Investigation performed: reviewer byte arithmetic; exact match on three configurations.
  Resolution: more-work-needed (documentation; Required Work).
- Observed anomaly: after fix2, p150 B32 rows with the same prompt diverged.
  Evidence: `evidence5/qualitative/pre_fix5/`, `evidence5/m1_pre_fix5/sampling_audit.json`
  (`host_rows_identical` false).
  Affected path: p150 B17 to B32 padded dispatch.
  Control or comparison: evidence4 and post-fix5 rows identical; slot path rows identical on all profiles.
  Likely subsystem: decode gate output consumed by the prefill dispatch.
  Investigation performed: stage fix5 and p150 requalification; reviewer source diff and saved-logit checks.
  Resolution: fixed (validated configuration restored); mechanism open (Other Concerns).
- Observed anomaly: batched KL to HF and to B1 moves with each program change.
  Evidence: `evidence5/calibration_compare.json`, `evidence5/m*/full_evidence.json`.
  Affected path: batched decode numerics.
  Control or comparison: I recomputed KL, top-1 and top-100 from
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/evidence5_work/calibration_b{16,32}_m{4,2,1}.pt`; all values
  match the README; HF top-1 in TT top-5 at every position.
  Likely subsystem: BF16 accumulation order.
  Investigation performed: stage TT-to-TT KL; reviewer recomputation.
  Resolution: controlled.
- Observed anomaly: `AllReduceAsync` mean time per call rose 7.9 -> 13.4 us on p150x4.
  Evidence: per-session raw ops CSV rows (most 7.6 to 9 us; last call 20 to 55 us in sessions 1 to 3).
  Affected path: B32 decode all-reduce.
  Control or comparison: replay total 3282 -> 2763 us; end to end 58.85 -> 47.97 ms.
  Likely subsystem: peer-device skew before the last all-reduce.
  Investigation performed: reviewer per-device, per-session extraction.
  Resolution: controlled.
- Observed anomaly: p150 B32 profiled replays rise 8 to 15% over 4 replays.
  Evidence: `evidence5/perf_accounting.json` (`m1_b32`).
  Affected path: p150 B32 device estimate only.
  Control or comparison: end-to-end runs flat (215.001, 215.002, 215.003 ms).
  Likely subsystem: profiler overhead or data-dependent dispatch (unverified).
  Investigation performed: stage comparison with end to end.
  Resolution: controlled.
- Observed anomaly: SKIP seed mode lets device token copies diverge on meshes.
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/seed_copies/m4_skip_b1.json`.
  Affected path: sampled decode with EP dispatch between draws.
  Control or comparison: host seeds, 0 mismatched steps in `evidence5/m*/seed_host.json` and all sampling audits.
  Likely subsystem: core PRNG (pseudo-random number generator) state shared with dispatch kernels.
  Investigation performed: decode-path matrix, audit check, negative control.
  Resolution: fixed (host default; SKIP raises for TP > 1).
- Observed anomaly: `generic_op` reused a cached program with shorter per-core runtime arguments.
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/batched/layer_check1/m4.json`.
  Affected path: batched GDN kernels.
  Control or comparison: `evidence5/m*/layer_check_gdn.json` at 4, 8, 16, 32 rows in one process.
  Likely subsystem: `generic_op` program hash.
  Investigation performed: source reading; compile-argument workaround.
  Resolution: fixed by workaround; upstream issue not filed.

## Scope Inspected

- Goal/skill paths: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/07-07-optimized-full-model.prompt.txt`,
  `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/optimize/SKILL.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/qualitative-check/SKILL.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/prompts/model_bringup_multigoal/07-optimized-full-model.check.sh`.
- Artifact paths: stage 7 `README.md`, `work_log.md`, `stage_review1.md`, `stage_review2.md`, `perf_summary.json`,
  `memory_handoff_recheck.json`, `batched_decode_audit.md`; every JSON in `evidence5/` (batch perf greedy and
  sampled, full evidence, inactive rows, sampling audit, seed test, layer checks, watcher run, calibration probes,
  accounting, named breakdown, check comparison, qualitative outputs and comparison, manifests, snapshots, patches)
  including `m1_pre_fix5/` and `qualitative/pre_fix5/`; `evidence3/` HF gate logs, autoregressive outputs,
  qualitative outputs, full evidence and profile summaries; `evidence4/` full evidence, layer checks, sampling audit
  and named breakdown; `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/context_contract.json`;
  stage 6 `full_model/evidence5/` batch perf, full evidence, teacher-forcing logs and qualitative outputs; stage 5
  `optimized_multichip_decoder/README.md`; `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/evidence_chain5.sh`,
  `evidence_chain5b.sh` and logs, `fix2/`, `fix3/`, `fix4/`, `fix5/`, `projfix/`, `union/`, watcher logs in
  `evidence5_work/watcher_m*/`, raw ops CSVs for p150x4 B32, p150x2 B32 and p150 B32 (evidence4 and evidence5),
  saved calibration logits in `evidence5_work/`, `pre_fix5/`, `fix5/` and `evidence4_work/`,
  `write_perf_summary5.py`, `named_breakdown5.py`, `calibration3/compare5.py`;
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/qualitative/hf_outputs.json` and
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/calibration/hf_aime24_logits.pt`.
- Code paths: `git diff HEAD` of the autoport `tt/`; snapshot diffs evidence3 chain 3c -> chain 5 -> chain 5b;
  `tt/optimized_decoder.py` (`_PackedGateMoE.forward`, `_OptimizedGateModule.route`, `_OptimizedShared.forward`,
  `_OptimizedSparseExperts.slotted`, `_Projection.__call__`, `_DecodeNorm`, `_LocalDispatch.route_maps`, mesh
  policies), `tt/expert_decode.py`, `tt/functional_decoder.py` (`decode_rows`, `decode_forward`,
  `_decode_batched`), `tt/model.py`, `tt/generator.py` (`decode_steps`, `prepare`, `_decode_trace`),
  `tt/multichip_decoder.py` (`MESH_POLICIES`, `_AllReduceOutput`), `tests/run_calibration_probe.py`,
  `tests/run_sampling_audit.py`, `tests/run_seed_advance.py`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/common/modules/moe/tt_moe_gate.py` (`_sharded_mem_config`).
- Commands run: `git status`, `git log`, `git diff`, `git ls-files`, `git check-ignore`; manifest recomputation
  and `cmp`; snapshot extraction and `diff`; the context-contract checker
  (`/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/scripts/check_context_contract.py
  --stage optimized_full_model --require-contract --strict-caps`: OK, 262144 supported); the CPU-only degenerate
  output checker (`check_degenerate_output.py --scope autoregressive --missing-artifacts critical`: no degenerate
  output); read-only Python over evidence JSON, raw ops CSVs and saved logits (KL, top-1, top-100, rank,
  `torch.equal`). Scratch files are in
  `/tmp/claude-2005/-home-hous-dev-qwen3-5-35b-a3b/5bead60f-0645-4ca5-b27d-6b0f6acbc38f/scratchpad/`. No device,
  server, pytest or process action.

## Residual Risk

- Verified as claimed: B1 TTFT 97.4 / 114.3 / 183.5 ms and token-out 9.11 / 11.30 / 15.80 ms (evidence3, and
  evidence5 repeats); traced teacher-forcing 99.83 / 82.88 / 60.85 t/s/u; batched table B4 to B32 on all profiles,
  sampled rows and run spreads; stage 6 before numbers; fix2, fix3, fix4 and fix5 trial numbers; accounting table
  (device, host gap, roofline, per-layer splits, sampler shares); profile table; B > 1 topology table; HF gates
  (AIME24 prefill and teacher forcing top-5 100% and top-100 100%; book 8193 top-5 at least 98.4%); batched KL to
  HF and to B1; layer delta PCC; inactive rows; sampling audit (TV 0.017 to 0.024); seed test; watcher clean;
  non-aligned prompts 5 to 12000; 262100-token prompt and 262143-token boundary; context-contract values; B1
  qualitative outputs identical to stage 6; batched qualitative rows coherent, same-prompt rows identical, B8 rows
  equal B32 rows on the meshes.
- B32 on p150x4 runs at 22% of its DRAM roofline. The slot MoE (gate_up gemv about 339 us, sparse down 314 us,
  reduce about 37 us per layer in one profiled replay, about 57% of the step) reads each selected expert once per
  row slot. There is no device-side de-duplication.
- p150x2 keeps both routed-expert copies (about 11.7 GB, estimate). Its KV pool holds 526336 tokens, 4 sequences at
  ISL 131072. p150 holds 2 and p150x4 holds 8. Stage 12 and stage 13 capacity planning depends on this choice.
- Free DRAM after all captures: 13.00 / 2.89 / 1.39 GB. Serving buffers on p150 have little room.
- Prefill is unchanged from stage 6 (262100 tokens in 62.1 / 90.8 / 139.2 s). The stage 13 ISL 131072 sweep depends
  on it.

## Rereview

Verdict: clean-pass

Rereview time: 2026 Oct 24 10:19 Eastern. Same read-only rules: I started no device job, server, pytest run or
process action.

Source identity: I recomputed the source manifest of the autoport `tt/` and `tests/` trees. It still equals
`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence5/source_manifest_end_m1b.txt`.
The 12 source hashes in the regenerated memory recheck equal the files on disk. This round changed only
documentation and post-processing output.

### Required Work

None. The review 3 P2 item is resolved:
- `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/memory_handoff_recheck.json`:
  - Item [1] now names both stage 7 allocations: the slot selectors (DRAM) and the router cached slices (L1).
  - `stage7_allocations` has a router cached-slice entry with widths, arithmetic, consumer, creation point
    (prepare, before capture), sharing and the attribution caveat (byte match, no allocator dump).
  - Each profile records L1 after build, the stage 6 value and the delta in total and per bank:
    - p150x4: +3604480 bytes, 32768 per bank.
    - p150x2: +3604480 bytes, 32768 per bank.
    - p150: +2703360 bytes, 24576 per bank.
  - I re-derived all values from stage 6 and evidence5 `full_evidence.json`, including free L1 and the largest
    free block per bank.
- `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/perf_summary.json`:
  - `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/write_perf_summary5.py` now computes `named_limitations`
    from `evidence5/perf_accounting.json`.
  - The text now gives 2.5%, 9017 to 9776 us, 16% to 36%, 1.28x / 1.54x / 3.36x, 6612 us, and 5.5% / 7.3% / 8.2%.
    All of these match the accounting file and the README.
  - The numeric fields equal the accounting file. No stale evidence4 value remains outside the labeled "Before"
    history line.
- README:
  - Context section: L1 lines added.
  - Disposition row: L1 deltas added.
  - p150 note: net Matmul change (about -125 us per replay).
  - All-reduce note: per-call spread (most about 8 us, last call 20 to 55 us).
  - Calibration wording: 1 prefill plus 99 decode positions.
  - Anomaly ledger: new L1 growth entry; the fix5 entry now says the exact op is not identified.

### Other Concerns (non-blocking)

- Required step 1 asked the recheck to state that the census does not reconcile sharded L1. Neither the recheck
  nor the README says so. The `method` text still says the census is compared "with the allocator total", which is
  true for DRAM only. Add one sentence when convenient. The new per-profile allocator L1 fields and the
  attribution caveat already carry the facts, so I do not block on this.
- The work log heading "Review 3 remediation (Oct 24 10:40)" is later than the file write time (10:17 Eastern by
  mtime). Correct the timestamp.
- `named_limitations` still states "(profiler overhead)" as the cause of the p150 B32 replay drift. The
  data-dependent dispatch path is an unexcluded alternative (review 3 Other Concerns). Wording only.
- All other review 3 Other Concerns and Hard-Check Gaps stay open as recorded; none blocks this stage:
  - Mechanism of the pre-fix5 divergence.
  - p150 per-row GDN.
  - p150 `moe_slot_rows` crossover.
  - `host_rows_identical` not asserted.
  - CPU suite not rerun.
  - Checkpoint force-add notes.

### Commands run

- Manifest recomputation and `cmp`.
- `git status`.
- Read-only Python over `memory_handoff_recheck.json`, `perf_summary.json`, `evidence5/perf_accounting.json`, and
  the stage 6 and evidence5 `full_evidence.json` files.
- `grep` for stale values and dash characters.
- Reading the two regenerated scripts, the changed README sections and the work log section.
