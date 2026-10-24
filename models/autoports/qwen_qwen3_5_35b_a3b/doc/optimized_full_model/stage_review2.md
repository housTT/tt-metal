# Stage Review

Verdict: more-work-needed

Stage 7, optimized full model, second review. Reviewer: independent stage-review subagent, read-only (no device,
server or pytest run). Review time: 2026 Oct 23 23:31 Eastern.

Source under review: live worktree, branch `feat/qwen3-5-35b-a3b`, HEAD `25bbbe3c` plus the uncommitted diff (the
projection fix at lines 1435 and 1438 of
`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py`, docs and
evidence). I recomputed the SHA-256 manifest of the autoport `tt/` and `tests/` trees. All 90 lines equal
`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence4/source_manifest_end.txt`.

Abbreviations: B (decode batch, active rows in one decode step), TTFT (time to first token), t/s/u (tokens per
second per user), HF (Hugging Face CPU reference), TT (Tenstorrent TTNN implementation), TTNN (Tenstorrent neural
network library), KV (key/value cache), GDN (Gated DeltaNet linear-attention layer), MoE (mixture of experts), EP
(expert parallel), TP (tensor parallel), LM head (final vocabulary projection), BF16 (16-bit brain float), BFP4 and
BFP8 (4-bit and 8-bit block float), FP32 (32-bit float), DRAM (device memory), L1 (core SRAM), SRAM (static RAM), RAM
(random-access memory), PCC (Pearson correlation coefficient), KL (Kullback-Leibler divergence, HF to TT), ULP (unit
in the last place), CPU (host processor), PRNG (pseudo-random number generator), SHA (secure hash algorithm). P1 is
a correctness or contract gap that blocks the stage. P2 is required work with lower correctness risk. p150, p150x2
and p150x4 are the 1, 2 and 4 chip profiles.

## Required Work

- P2: Batched decode (B > 1) still selects prefill programs in the router and on p150, and the selected slot path
  carries unaudited layout glue
  Evidence:
  - Router linear at decode rows has no program config:
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py:882`.
    The B1 branch of the same `_PackedGateMoE.forward` passes `program_config=self.program`. Both see the same
    padded shape (32 tile rows). p150x4 B32 row: `32 x 2048 x 288`, 23.2 us, 9 cores, `in0_block_w` 2, 9.9% DRAM
    (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence4/m4/profile_b32/decode_perf_report.csv`).
    p150x4 B1 row: 7.0 us, `in0_block_w` 32, 32.7% DRAM
    (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence3/m4/profile_b1/decode_perf_report.csv`).
  - Router top-k at decode rows takes the prefill branch `if total > 1 and getattr(self, "topk_prefill", False)`
    (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py:969`):
    a generic single-core `TopKDeviceOperation` of 33.4 us per layer plus a 1-core softmax of 5.0 us and 3
    typecasts. The B1 route uses `GeneralizedMoeGateDeviceOperation` (1.4 us), about 18 us for the whole route
    (B1 replay that I extracted from
    `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/evidence3_work/named_profile_b1_m4/reports/2026_10_23_20_39_28/ops_perf_results_2026_10_23_20_39_28.csv`).
    `TopKDeviceOperation` is 66.9 us per 2-layer replay on every profile
    (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence4/named_breakdown.json`).
  - p150: `attention_in` has `prefill_minimal` and no `decode_grid`
    (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py:171`
    and `:189`), so the projection fix (`pc is None`, `:1435`) does not apply. Batched decode on p150 runs
    `minimal_matmul` with the prefill grid: GDN `32 x 2048 x 12352` 142.5 us at 34.7% DRAM, full `32 x 2048 x 9216`
    110.4 us at 33.4% DRAM
    (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence4/m1/profile_b32/decode_perf_report.csv`).
    The B1 default matmul on the same padded shape is 70.2 us at 70.3% and 54.8 us at 67.2%
    (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence3/m1/profile_b1/decode_perf_report.csv`).
    That is about 2.7 ms per step on every p150 bucket from B4 to B32 (30 x 72 us + 10 x 56 us).
  - Slot MoE glue in one p150x4 B32 decode replay (session 4 of trace 2, device 0, that I extracted from
    `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/evidence4_work/named_profile_b32_m4/reports/2026_10_23_23_28_15/ops_perf_results_2026_10_23_23_28_15.csv`):
    - Each row chunk after the first re-lays out the whole index tensor (`stick(wide[:, :, start:start + count, :])`,
      `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py:1172`
      to `:1182`): untilize 9.5 us on 1 core, slice, tilize, reshape, typecast, untilize, about 20 us per chunk.
      B32 uses 6 chunks per layer, so about 100 us per layer.
    - The down output keeps 32 tile rows per slot. The score multiply runs over `[1,256,32,2048]` BF16 (33.5 MB):
      92 us, then `fast_reduce_nc` 36.6 us and the selector multiply 10 us (`:1203` to `:1205`).
    - The residual between layers stays `[B,1,2048]` (every row padded to its own tile). Each layer reshapes it to
      `[1,B,2048]` (16.1 us) and back (25 to 26 us)
      (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/functional_decoder.py:504`
      and `:510`). The embedding entry adds a tilize of 54.9 us and a reshape of 22.8 us once per step.
    - Layout and glue ops (untilize, tilize, reshape, typecast, slice, concat, transpose, interleaved/sharded
      conversions): 175 us in the GDN layer (1502 us, embedding entry excluded) and 222 us in the full layer
      (1253 us).
  - p150 B32 (EP dispatch path) still has the 1-core `32 x 2048 x 32` matmul (45.4 us per layer) and the 8-core
    `32 x 2048 x 256` router (47.8 us) that review 1 named.
  - The README "Operation topology" and "Checklist" tables list none of these. The checklist row says "MoE
    active-expert path at decode | slot path; union-mask rejected with measurements".
  Why this matters: the router and p150 cases are the same defect class as the projection wiring that this stage
  fixed: a prefill branch selected by `length > 1` inside batched decode. From the profile rows, the router costs up
  to about 1.9 ms per step at every bucket on every profile (about 11% of the p150x4 B4 step of 16.70 ms), if the
  B1 route cost holds at 32 rows. The listed glue sums to about 13 ms of the p150x4 B32 step of 58.85 ms. These are
  estimates from profile rows, not measured savings. `$optimize` requires explicit program configs for important
  ops, an operation-topology audit that lists layout conversions, and "no avoidable DRAM round trips through decode
  intermediates" for MoE. The stage 13 sweep runs client concurrency up to 32 on all three profiles.
  Required next step:
  1. Pass the B1 router program config at decode rows. Route B > 1 through the decode router path, or measure it
     slower and record the result.
  2. On p150, use the B1 `attention_in` program (or a decode program config) at decode rows.
  3. Build the slot index sticks once per layer (one row-major index tensor, row-major slices), or record why not.
  4. Try a combine that does not materialize 32 rows per slot (for example, select row `s // 8` of each slot before
     the score weighting), or record the measured result.
  5. Carry `[1,B,2048]` between layers, or record the exact blocker and its measured cost.
  6. Add a batched (B > 1) operation-topology table to the README.
  Then rerun the affected checks on every profile: batch perf (greedy and sampled), layer checks, HF calibration at
  B16 and B32, inactive rows, sampling audit and full evidence, with the trace allocation tracker on for the
  correctness runs. Report the final reproduced numbers.

- P2: Handoff items assigned to Stage 7 have no disposition, and the memory handoff was not rechecked after the
  implementation change
  Evidence:
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/README.md`,
    "Open items for later stages", assigns to Stage 7: "Routed experts stored twice on p150x2 and p150x4" and "DRAM
    remainder not itemized; construction peaks not measured".
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder/README.md`,
    "Work owned by later stages", assigns expert duplication to "Stage 7 memory" (stacked option measured at +4 to
    7% prefill).
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/work_log.md`
    lists both duplicated experts and long prefill (262100 tokens in 62 to 140 s) as candidate work at the start.
    Neither the README nor the work log mentions them again (no hits for duplicat, census, remainder, peak or
    handoff).
  - `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`: the memory handoffs "are source-hash-bound and must be rechecked
    after implementation changes". Stage 6 wrote
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/memory_handoff_recheck.json`.
    No Stage 7 recheck exists. Stage 7 adds persistent per-layer device memory: layer-private DRAM +37.5 MiB on
    p150x4 and p150x2, +17.5 MiB on p150 (census in
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence4/m4/full_evidence.json`
    and its m2 and m1 siblings, against
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence5/m4/full_evidence.json`).
  - Effect today: p150x2 keeps 24.4 GB of layer-private DRAM (the p150 level), 2.85 GB free after all captures and
    2.0 full-context sequences in the KV pool; p150 has 1.37 GB free.
  Why this matters: the project contract and two earlier stages assign these items to this stage. Silent omission
  hides a capacity decision that affects p150x2 serving at long context.
  Required next step: add a disposition table to the README (done, measured and rejected, or deferred to a named
  stage with the reason) for: duplicated routed experts, DRAM remainder and construction peaks, and long prefill.
  Write a Stage 7 memory handoff recheck from the existing evidence4 census: final source hashes, the new slot
  selector and batched GDN allocations, and the unnamed remainder by size.

## Other Concerns

- Stage review 1 status:
  - P1 (no B8 and B16 evidence): resolved. Inactive rows, sampling audit and full evidence cover buckets 1, 4, 8,
    16 and 32 with the tracker on; HF calibration with distinct rows at B16 and B32 on all profiles.
  - Qualitative at B > 1: resolved with a weaker design (see below).
  - Per-token blocking readback: resolved. `decode_steps` overlaps the token read with the next replay; host gap at
    most 0.65 ms per step.
  - Performance accounting and `perf_summary.json`: resolved.
  - Unrepresentative B32 profile and no decode-shaped MoE candidate: resolved (distinct rows, union mask measured
    and rejected, slot path selected).
  - Default did not reproduce 60.93 ms: resolved. The path was replaced; the final 58.85 ms reproduces in 3 runs
    within 0.01 ms.
  - DRAM-sharded LM head: resolved for p150x4 and p150 (in model, 0.105 to 0.11 ms slower). Not measured in model
    on p150x2.
  - Chain 1 provenance: resolved by disclosure; final claims use evidence3 and evidence4 only.
- `decode_steps` early stop
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/generator.py:232` to `:246`):
  when `stop` fires on token i, replay i+1 has already run with the EOS token as input. The KV cache, GDN state and
  device position advance one step past EOS, and the read of token i+1 is dropped without a wait. The runtime pins
  the host buffer of a non-blocking read (`MemoryPin`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/distributed/fd_mesh_command_queue.cpp:176` to `:183`), so this
  is not a use-after-free. `generate()` resets slot 0 at the next call, so there is no visible effect today. State
  the semantics in code or synchronize and document before a caller continues a slot after EOS. No run exercised
  this branch (every qualitative output is 128 tokens).
- Projection-fix anomaly wording: the README says single-layer batched outputs are "bitwise unchanged" and the
  probe outputs are "bitwise equal". The probe compares PCC values
  (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/projfix/layer_probe/layer_check_m4_l0.json` and siblings). A
  float32 PCC over 4 x 2048 values cannot resolve a 1-ULP BF16 change in a few elements, so equal PCC is not
  bitwise evidence. With `torch.equal` on the saved logits I confirmed the README numbers: mesh B16 logits first
  differ at decode position 3 (p150x4) and 2 (p150x2), maximum 9.47 and 10.68; p150 is bitwise unchanged; B16 equals
  B32 on both meshes. Reword the claim or compare raw layer outputs. After the fix, p150x2 batched KL maximum is
  0.255, above its B1 maximum 0.186 (top-1 99%, HF top-1 in TT top-5 at 100 of 100 positions).
- Qualitative B8 and B32 rows reuse the 6 prompts cyclically, not distinct fillers as review 1 asked. Rows r and
  r + 6 have the same neighbors, so "same-prompt rows identical" has little power against neighbor leakage. B8
  equals B32 row by row on p150x4 and p150x2 (I checked all 8 rows); that is strong row-independence evidence. On
  p150, B8 and B32 differ, which is expected (dispatch at B32). Every row I read is coherent. The README does not
  cite the HF control (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/qualitative/hf_outputs.json`). My
  comparison: the common prefix of B32 rows with HF is equal to or near the B1 prefix for every prompt (for
  example p150x4 B1 / B32: 66/39, 48/90, 119/119, 19/19, 44/46, 29/29 tokens).
- Greedy sampler choice was measured only on p150x4
  (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/sampler/`). On p150 the sampler trace is 1295 us (8.2% of B1
  token-out) and the local top-k over the unpadded 248320-entry vocabulary shard is about 1070 us per step.
  Force-argmax was not measured on p150 or p150x2 (repeat from review 1).
- p150 batched GDN runs per row because `per_head = 4` needs 128 cores
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/gdn_batch.py:83`). At p150
  B32 the 30 GDN layers take about 200 ms of the 220.85 ms step (6752 us per GDN layer,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence4/perf_accounting.json`).
  The head index is a runtime argument, so two calls of 16 heads (64 cores) look like a host-only change. Not tried
  (repeat from review 1). p150 is not the optimization target, but it must serve the concurrency sweep.
- The decoder-layer lower bound uses stage 5 host times that include a per-layer host sync. With the stage 5
  kernel sums (197.9 and 248.2 us per layer at 4096 context) the bound is 8.42 + 0.96 = 9.38 ms; B1 token-out
  9.11 ms is still below it. No action needed.
- Final norm is a 1-core FP32 `LayerNorm` of 24.6 us plus two typecasts per step (repeat from review 1, about 0.3%
  of B1).
- The `generic_op` program-hash limitation has a sound workaround; the upstream issue is still not filed.
- The CPU suite was not rerun after the projection fix (chain 3: 1219 passed). The fix is device-path only.
- Continuation prefill PCC of whole versus split prompt is 0.826 to 1.0 on p150x4 (0.876 to 1.0 on p150x2, 0.902
  to 1.0 on p150), bit-for-bit the same as stage 6. Stage 6 assigned the related over-confidence item to Stage 8.
- Checkpoint commit: keep the stage 6 untracked files
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence1/m1/profile/ops_perf_results.csv.gz`
  (and the m2, m4 and evidence2 siblings) out. The 61 MB `batched_l2_watcher.pt` files are ignored by
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/.gitignore` line 99.

## Hard-Check Gaps

- The chains' `SOURCE_STABLE` check is blind to `tt/optimized_decoder.py`. `manifest()` ends with
  `git diff --stat -- ttnn | tail -1`, which prints nothing when `ttnn` is clean. The compare uses `head -n -1`, so
  it drops the last real line, which is the `tt/optimized_decoder.py` hash (last in sort order). This affects
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/evidence_chain3c.sh`,
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/evidence_chain4.sh`,
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/profile_l037.sh` and
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/profile_named.sh`. I compared the full manifests: start equals
  end for chains 3c, 4, l037 and named, and the current worktree equals the chain 4 end manifest. The chain 3c to
  chain 4 source delta is exactly the two projection-fix lines (snapshot diff). No evidence is invalidated. Emit a
  fixed label line for the `ttnn` stat before the next chain.
- Layer checks (`run_layer_batch_check`) compare per-row and batched outputs by PCC and maximum absolute
  difference. They cannot show that a program change is numerically neutral.
- Device-time reconciliation sums kernel durations. It excludes device op-to-op gaps (tt-perf-report median 0.29
  us per op at B1 and 0.36 us at B32 on p150x4) and includes profiler inflation (p150 B32 device 225.49 ms against
  220.85 ms end to end; profiled replays rise 8 to 15% over 4 replays). Host-gap values of about 0.1 ms are inside
  the method error. The "device-bound" conclusion holds.
- No B4, B8 or B16 profile exists after the projection fix. Router and glue costs at small B are inferred from B32
  rows and op shapes.
- Batched HF calibration teacher-forces row 0 only (100 positions); there is no top-100 metric for batched decode;
  the batched-versus-B1 checks in full evidence run 8 steps.

## Anomaly Ledger

- Observed anomaly: batched decode selects prefill programs (router linear without program config, router
  prefill top-k on 1 core, p150 `attention_in` prefill `minimal_matmul`).
  Evidence: Required Work item 1.
  Affected path: every bucket B4 to B32 on every profile (router); every p150 bucket (`attention_in`).
  Control or comparison: B1 rows with the same padded shape (7.0 us router, 70.2 and 54.8 us `attention_in`).
  Likely subsystem: program selection by `length > 1` inside the decode batch context.
  Investigation performed: reviewer code reading and profile row extraction; the stage did not record it.
  Resolution: more-work-needed.
- Observed anomaly: after the projection fix, the single-layer PCC checks are unchanged but full-model batched
  logits change on the meshes.
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/evidence3_work/calibration_b16_m4.pt` against
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/evidence4_work/calibration_b16_m4.pt` (and m2): first
  difference at decode position 3 and 2.
  Affected path: batched decode numerics on p150x4 and p150x2.
  Control or comparison: p150 logits bitwise unchanged although its full-attention `attention_out` also moved from
  `minimal_matmul` to the 1D config; HF KL stays at the B1 level; HF top-1 in TT top-5 at all positions.
  Likely subsystem: different accumulation in the mesh `attention_in` programs. The PCC-based probe cannot detect
  small differences, so it does not support the "bitwise equal" wording.
  Investigation performed: stage probes with call counting and PCC; reviewer `torch.equal` on the saved logits.
  Resolution: controlled for model quality; wording correction needed (Other Concerns).
- Observed anomaly: `SOURCE_STABLE` compare excludes `tt/optimized_decoder.py`.
  Evidence: Hard-Check Gaps item 1.
  Affected path: provenance of chains 3c and 4 and the accounting profiles.
  Control or comparison: full-manifest compare by the reviewer (equal start and end; worktree equals chain 4 end).
  Likely subsystem: chain script `manifest()`.
  Investigation performed: reviewer recomputation and snapshot diff.
  Resolution: controlled; fix the script.
- Observed anomaly: `decode_steps` runs one replay past EOS and drops its read.
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/generator.py:232`
  to `:246`.
  Affected path: `generate(stop_on_eos=True)`.
  Control or comparison: runtime `MemoryPin` keeps the host buffer alive; `generate()` resets slot state on the
  next call.
  Likely subsystem: pipelined token readback.
  Investigation performed: reviewer code reading of the generator and the mesh command queue.
  Resolution: controlled; document or synchronize.
- Observed anomaly: p150 B32 profiled replays rise 8 to 15% over 4 replays.
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence4/perf_accounting.json`
  (`m1_b32`, `decode_per_replay_us`, layers (0,1,3): 15072 to 17328 us).
  Affected path: p150 B32 device-time estimate only.
  Control or comparison: end-to-end runs flat at 220.85 ms (3 runs); meshes flat within 1%.
  Likely subsystem: profiler overhead with 472 ops per replay.
  Investigation performed: stage comparison with end-to-end runs.
  Resolution: controlled.
- Observed anomaly: SKIP seed mode lets device token copies diverge on meshes.
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/seed_copies/m4_skip_b1.json`.
  Affected path: sampled decode with EP dispatch between draws.
  Control or comparison: host seeds, 0 mismatched copies in
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence4/m4/seed_host.json`
  and all 12 sampling-audit runs; negative control fails as expected.
  Likely subsystem: core PRNG state shared with dispatch kernels.
  Investigation performed: decode-path matrix, audit check, negative control.
  Resolution: fixed (host default; SKIP raises for TP > 1).
- Observed anomaly: continuation prefill PCC 0.826 (p150x4, start 128, length 300).
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence4/m4/full_evidence.json`
  (`continuation`).
  Affected path: chunked prefill continuation.
  Control or comparison: identical values in stage 6 evidence5; same argmax in every case; stage 6 HF probe.
  Likely subsystem: pre-existing, outside the Stage 7 change set.
  Investigation performed: stage 6 continuation probe.
  Resolution: controlled for Stage 7; tracked by Stage 8.

## Scope Inspected

- Goal/skill paths: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/07-07-optimized-full-model.prompt.txt`,
  `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`, `/home/hous/dev/qwen3.5-35b-a3b/experiment-provenance-handoff.json`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/optimize/SKILL.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/multichip/SKILL.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/qualitative-check/SKILL.md`.
- Artifact paths: stage 7 `README.md`, `work_log.md`, `stage_review1.md`, `perf_summary.json`,
  `batched_decode_audit.md` under
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/`;
  every JSON summary in its `evidence4/` (batch perf, full evidence, inactive rows, sampling audit, seed test, layer
  checks, watcher run, calibration, accounting, named breakdown, qualitative outputs and comparison, manifests,
  snapshot) and the B1 and prefill files in `evidence3/` (HF gate logs, autoregressive outputs, split-sampling
  trace, calibration, profiles, manifests, snapshots); B1 and B32 `decode_perf_report.csv` for all profiles;
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/context_contract.json`
  (diff and `optimized_full_model_validation`); stage 6 `full_model/README.md`, `evidence5` full evidence, batch
  perf and qualitative outputs; stage 5 `optimized_multichip_decoder/README.md`;
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/evidence_chain4.sh`, `evidence_chain4.log`, `check07_ev4.log`,
  `projfix/` and `projfix/layer_probe/`, `union/`, `sampler/`, watcher logs in `evidence4_work/watcher_m*/`;
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/qualitative/hf_outputs.json`.
- Code paths: diffs since `fb8ce438` of `tt/generator.py`, `tt/model.py`, `tt/optimized_decoder.py`,
  `tt/functional_decoder.py`, `tt/multichip_decoder.py`, `tt/gdn_decode.py`, full `tt/gdn_batch.py`, and the
  runners `tests/run_batch_perf.py`, `tests/run_full_evidence.py`, `tests/run_layer_batch_check.py`,
  `tests/run_inactive_rows.py`, `tests/run_qualitative.py`, `tests/runtime_audit.py` (all under
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/`); the non-blocking read path in
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/impl/tensor/distributed_tensor_apis.cpp` and
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/distributed/fd_mesh_command_queue.cpp`.
- Commands run: `git status`, `git log`, `git diff`; manifest recomputation and full and truncated manifest
  diffs; snapshot extraction and diff; the context-contract checker
  (`/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/scripts/check_context_contract.py
  --stage optimized_full_model --require-contract --strict-caps`: OK, 262144 supported); read-only Python over the
  evidence JSON, tt-perf-report CSVs and three raw ops CSVs (one decode replay each for p150x4 B1, p150x4 B32 and
  p150 B32); `torch.equal` over the saved calibration logits. Scratch outputs are in
  `/tmp/claude-2005/-home-hous-dev-qwen3-5-35b-a3b/5bead60f-0645-4ca5-b27d-6b0f6acbc38f/scratchpad/`. No device,
  server or pytest run.

## Residual Risk

- Verified as claimed: the B1 headline (TTFT 97.4 / 114.3 / 183.5 ms, token-out 9.11 / 11.30 / 15.80 ms), the
  batched table, sampled rows, HF gates (AIME24 top-5 100% and top-100 100% on every profile), KL tables, inactive
  rows, sampling audit, seed test, watcher, non-aligned prompts 5 to 12000, the 262100-token prompt and the
  262143-token boundary, native 262144 context and the context-contract values.
- B1 is device-bound at the stage 5 decoder level. Further B1 gains need decoder or terminal device work: LM head
  389 us, sampler 499 us, final norm 25 us on p150x4.
- p150x4 B32 runs at 18% of its DRAM roofline. Beyond the glue above, the slot path reads each selected expert
  once per slot (256 per layer at B32) against an expected union of about 165.
- p150 has 1.37 GB DRAM free after all captures; p150x2 has 2.85 GB. Stage 9 serving buffers may not fit without
  the expert de-duplication decision.
- Prefill and TTFT are unchanged from stage 6. ISL 131072 at concurrency 32 in stage 13 will depend on prefill
  throughput that this stage did not address.
