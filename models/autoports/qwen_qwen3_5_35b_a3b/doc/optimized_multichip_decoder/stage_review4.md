# Stage Review

Stage 05, optimized multichip decoder. Model: Qwen/Qwen3.5-35B-A3B, revision
59d61f3ce65a6d9863b86d2e96597125219dc754. Reviewer: fresh independent subagent, review 4. Review date: 2026 Oct 12,
about 11:10 Eastern. I did not open a device, start a server, build, or run a device test. I ran CPU-only reads of
existing artifacts, the ttnn source, and small Python scripts over JSON, CSV and log files.

Abbreviations: TP (tensor parallel), EP (expert parallel), MoE (mixture of experts), GDN (Gated DeltaNet, the
linear-attention layer), CCL (collective communication), AR (all-reduce), PCC (Pearson correlation coefficient), HF
(Hugging Face reference), BF16 (16-bit brain float), BFP8/BFP4 (8/4-bit block float), FP32 (32-bit float), L1 (core
SRAM), DRAM (device memory), KV (key/value cache), SDPA (scaled dot-product attention), TTFT (time to first token),
ISL (input sequence length), FLOP (floating-point operation), ET (Eastern Time), UTC (Coordinated Universal Time).

Short names:
- `AP` = `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b`
- `OD` = `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder`
- `E9` = `OD/evidence9`, `E10` = `OD/evidence10`
- `S5` = `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage05`
- `TM` = `/home/hous/dev/qwen3.5-35b-a3b/tt-metal`

Verdict: more-work-needed

Summary. The evidence chain is sound and the headline numbers are real: source identity, 243 chain steps, 191
audited harness rows, headline decode, prefill table, 40-layer estimates, accuracy minimums, profiles, capacity,
long context and context contract all recompute from raw artifacts (see Verified Claims). Review 3 items 1, 2 and 3
are resolved with measurements. Item 4 (prefill chunk size) is not: the rejection rests on a model-local
construction guard and on a bound that the stage's own chunk-size data contradicts. I also found one decode op that
no audit or review has identified (a redundant zero-fill before the routed down `sparse_matmul`, on every profile),
one op-removing optimization deferred without an owning skill (conv-state shift), and one open performance anomaly
(masked-chunk overhead).

## Required Work

- P2: Long-context prefill chunk size and SDPA query chunk: the "4096 kept" decision is not earned, and its stated
  bound contradicts the stage's own data.
  Evidence:
  - `S5/micro/chunk_size/l3_c2048.json` vs `l3_c4096.json` (p150x4, full layer, 131072 tokens): the last chunk
    (KV length about 131072) costs 64.02 ms for 2048 queries and 67.66 ms for 4096 queries. Halving the queries
    (and the FLOPs) leaves the per-call cost almost unchanged. The sum over chunks after the first falls from
    2270.2 ms (2048) to 1253.3 ms (4096): doubling the chunk nearly halved full-layer long-context prefill. Linear
    layer: 448.4 ms to 334.5 ms.
  - The only true 8192-row internal chunk (`l{0,3}_c8192_b8192`) fails at construction:
    `ValueError: Internal local-dispatch chunk must fit UINT16 route maps` (`run.log`: rc=1). That guard is model
    code, `AP/tt/multichip_decoder.py` lines 258-262 (and `AP/tt/optimized_decoder.py` line 2031). It builds a
    dispatch route map for every planner bucket length, including 8192, although meshes run the MoE in
    `prefill_moe_pieces: 2` pieces (4096 rows each for an 8192 chunk, 4096 x 8 routes = 32768 < 65536). This is a
    first-error rejection of the kind `$stage-review` and `$optimize` do not accept; no adapted attempt exists
    (for example, build route maps per MoE piece length, not per bucket).
  - The "8192" row in the README chunk table (`l{0,3}_c8192.json`) is an external 8192 call that the planner
    splits into two internal 4096 chunks (last chunk 133.0 ms = 2 x 67). It does not measure an 8192 chunk.
  - `OD/README.md` lines 433-435 bound the gain with "KV read is 71 MB (0.14 ms at 512 GB/s)" and "SDPA FLOPs
    (4.4e12 per device) take 14.5 ms at HiFi2 peak ... compute efficiency, not KV re-reads, sets the cost". Chunked
    SDPA streams the KV prefix once per query chunk and head, not once per call (32 query chunks x 4 local heads at
    q_chunk 128). By my calculation (unverified on hardware) the FLOPs at that point are 4 heads x 4096 x 131072 x
    256 x 2 (QK and PV) x 2 = 2.2e12 per device, half the stated value, so the op runs near 12-13% of the stated
    HiFi2 peak, not 25%. The 2048-vs-4096 data above shows the cost scales with calls, not FLOPs.
  - `S5/micro/sdpa_prefill/sdpa_prefill2.json` sweeps q_chunk 64 and 128 only. No q_chunk 256 or 512 candidate
    exists, although a larger query chunk halves the KV re-reads per call.
  - Impact: the layer-sum TTFT estimate at ISL 131072 (a required sweep point) is about 22.5 s on p150x4, of which
    10 full layers x 1.25 s is SDPA-dominated (`OD/README.md` line 416). If the per-call cost stays near 67 ms at
    8192 queries, as the 2048-to-4096 trend suggests, full-layer long-context prefill could drop by a large
    fraction. This is an estimate, not a measurement.
  Why this matters: `$optimize` "Prefill Latency And TTFT" requires a comparison of feasible chunk sizes for the same
  logical request, and rejections need an adapted attempt. Review 3 required "Adopt the faster legal size, or record
  the exact L1/CB allocation failure of the larger chunk". The recorded failure is a model-side guard, not an L1/CB
  limit.
  Required next step: (1) Sweep SDPA prefill `q_chunk_size` 256 and 512 (with k_chunk 256/512) at starts 65536 and
  126976 on p150x4 and p150, HiFi2, against the q128/k256 control. (2) Make an 8192-row internal chunk legal by
  sizing dispatch route maps per MoE piece (or record the exact L1/CB or op failure of the adapted path), and
  measure 131072 tokens on p150x4 for both layer kinds against the 4096 control. Adopt the faster legal setting,
  then rerun the long-context rows, the affected length rows and the context rows. Correct the KV-read and FLOP
  statements in `OD/README.md` and `OD/work_log.md`.

- P2: The routed down `sparse_matmul` is preceded by a full zero-fill every decode step on every profile. No audit
  identifies it, and two earlier reviews mislabeled it.
  Evidence:
  - `E10/profiles/m4_linear_attention/ops_perf_results.csv.gz`, device 0, last replay: op 29 is
    `UnaryDeviceOperation` with `op_chain` `UnaryOpType::FILL; param={0}`, input [1, 8, 32, 2048] BFP8, 110 cores,
    3.22 us, immediately before `SparseMatmulDeviceOperation` (output BFP8). In `E7` the same op is BF16 (it follows
    the `sparse_matmul` output dtype). Per-layer cost from `decode_step_ops.txt`: x4 linear 3.27 us, x4 full 3.24,
    x2 linear 3.14, p150 linear 3.31, p150 full 3.40. About 0.13 ms per token on p150x4 at 40 layers (1.3% of the
    9.67 ms layer-sum estimate).
  - Source: `TM/ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp` lines 491-541.
    `create_output_tensors` runs `ttnn::zeros_like` on the output unless the caller supplies an output tensor of
    the compact shape with `nnz` set. Lines 355-359 state that in indexed mode the writer fills all `num_active`
    slots. The model calls `ttnn.sparse_matmul(..., nnz=None, indices=ids)` with no output tensor
    (`AP/tt/optimized_decoder.py` lines 1124-1136), so the fill is redundant work.
  - The tt-perf-report advice on the same row reads "the CSV row has nnz=std::nullopt".
  - `OD/stage_review2.md` line 65 and `OD/stage_review3.md` line 78 call this op a "sigmoid `Unary`". `OD/README.md`
    and `OD/work_log.md` never name it; the operation-topology audit row "Expert down + weighted reduce" lists only
    the `sparse_matmul` and the fused reduce.
  Why this matters: the goal requires an operation-topology audit and evidence for each applicable op removal;
  `$optimize` requires "no avoidable DRAM round trips through decode intermediates" for MoE.
  Required next step: add the op to the audit. Try to remove it: pass a persistent preallocated output with a legal
  `nnz`/indexed shape that the op treats as compact, or make the minimal local ttnn change that skips the zero-fill
  in indexed mode when every slot is written. Measure traced decode (3 interleaved pairs, 6 rows) and PCC. If an op
  contract blocks it, record the exact error and the adaptation tried.

- P2: The conv-state shift (4 ops, 5.6 us per linear layer) is deferred to "Work owned by later stages" with no
  owning skill and no measurement.
  Evidence:
  - `E10/profiles/m4_linear_attention/decode_step_ops.txt` ops 7-10: slice 1.58, slice 0.67, concat 1.71, copy 1.67
    us (5.63 us; about 0.17 ms per token at 30 linear layers, 1.7% of the estimate).
  - `OD/README.md` lines 158-160: "the shortest traced alternative is 3 ops including a 35-row concat. Not measured
    ... this is kernel work, open (Work owned by later stages)"; line 568 lists it under later stages.
    `OD/work_log.md` line 970: "Not pursued".
  - The stage already owns a model-local GDN decode kernel that reads the conv outputs in place
    (`AP/tt/gdn_decode.py`, `AP/tt/kernels/gdn_decode/`), so a state write in that kernel is within reach.
  Why this matters: the goal states "No optimization is deferred", and `$optimize` says "Do not leave a known
  optimization for a later stage unless another skill explicitly owns it". No later stage skill owns decode conv
  kernel work.
  Required next step: measure the 3-op alternative in the traced layer, and attempt the state update inside the GDN
  decode kernel (or a conv variant with a state output). Adopt the faster correct path, or record the exact blocker.

- P2: Masked-chunk prefill overhead is an open anomaly with no investigation.
  Evidence: `E10/prefill_timing/*.json` (true median of 6 warm runs, ms): one masked 4096 chunk at 4095 tokens vs
  the unmasked 4096 chunk: x4 linear 11.27 vs 10.56, x4 full 11.51 vs 10.97, x2 linear 14.78 vs 14.14, x2 full 14.96
  vs 14.59, p150 linear 21.24 vs 20.44, p150 full 20.37 vs 20.07. At 2600-3968 tokens the masked chunk does less
  real work than 4096 rows (padding rows are excluded from MoE dispatch, `dispatch.valid_rows`) but costs more.
  `OD/README.md` line 588 resolves it as "open, about 0.3-0.7 ms; the masked-chunk overhead is not profiled".
  Why this matters: every non-aligned prompt ends with a masked chunk, so this cost applies to almost every real
  request (about 4-5% of a 3000-token prompt at 40 layers). The anomaly rule does not accept "open" with prose.
  Required next step: profile 4095 vs 4096 on p150x4 for both layer kinds with the existing profile harness,
  attribute the extra time to op rows, and fix it or record why it is required.

## Other Concerns

- p150x2 gets the BFP8 routed sum without a speed gain. `S5/trials/shared_addcmul_bfp8/` screens: x2 -0.06 / -0.68
  us (K block 8) and -0.5 / +0.02 us (K block 4), inside run noise, while random-activation decode minimums fall by
  0.0004-0.0010. A per-profile choice (keep the separate BF16 tail on p150x2) would remove the precision cost there.
  The datatype sweep must check this choice per profile.
- p150x2 planner: the mesh costs make 2600-3000 tokens 0.3-0.8 ms slower than chain 8 (recomputed from
  `E10` and `OD/evidence8` prefill tables). A length band for the 2048-row cap (cap only below 3300 tokens) was not
  tried; the CPU grid search covers only the two scalar costs.
- The "short lengths are noisy" note is not random noise: in all 15 cells with max/min above 1.2, the first warm
  run is the slowest (recomputed). It is a systematic first-warm penalty of up to 1 ms. Serving warmup must cover it,
  or the cause should be named.
- `S5/evidence_chain11.sh` and `S5/process_profiles11.sh` (written 14:39 UTC, 10:39 ET) point at an `evidence11/`
  that does not exist. Label them as unused or remove them to avoid a stale-artifact reference.
- The p150x2 placement gap (devices 0,1 about 6-7 us slower than devices 2,3) stays "open, not profiled". It does
  not affect the selected p150x2 profile, but a tt-smi clock/telemetry comparison of the two cards is cheap.
- Router LoFi and SDPA prefill LoFi remain handed to the datatype sweep (same as review 3). Both pass real-weight
  gates; Stage 8 must measure them.
- At ISL 131072 and 32 concurrent requests, the p150x4 capacity model gives 12.54 full-context sequences
  (about 3.29M KV tokens) against 4.19M needed; client concurrency 32 will queue. Stage 7/13 must record active
  server slots separately, as AGENTS.md requires.

## Hard-Check Gaps

- Watcher poll coverage. `TT_METAL_WATCHER=10` polls every 10 s, and `TM/tt_metal/impl/debug/watcher_server.cpp`
  (lines 163-200) detaches without a final dump. `E10/watcher_long/logs_m1_0_{1300,4096}` contain one dump (at
  about 0.5 s, before any model kernel ran); the other long logs contain 2-3. The "no error in 24 logs" claim
  therefore rests mostly on in-kernel checks that halt the core (all runs exited 0 with correct output), not on
  polled state during decode. A shorter interval (for example `TT_METAL_WATCHER=1`) on one row per profile would
  make the claim direct.
- OPT-006 cumulative contract table (projection topology, dtype/fidelity by group, KV dtype/layout, logical batch vs
  tile rows, SDPA config and row, residual/norm layout, op-to-op gap) is still spread across README sections (third
  review). The KV-cache dtype (BFP8) appears only in `AP/doc/context_contract.json`, not in the README (OPT-002).
- tt-perf-report advice "No program_config specified" on `MinimalMatmulDeviceOperation 32 x 128 x 2048` is false:
  the row attributes show `MinimalMatmulConfig(M_block_size=1;K_block_size=4;N_block_size=8;...)`. `$optimize` asks
  to record misleading advice as a candidate tt-perf-report improvement; the README does not.
- The CCL dtype family result (BFP8 `all_reduce_async` slower than BF16 even without conversions, 16.5-17.6 vs
  10.8-11.5 us, `S5/micro/direct_allreduce_x{2,4}.json`) is only in `OD/work_log.md` line 71. Add it to the README
  CCL family table.

## Anomaly Ledger

- Observed anomaly: SDPA prefill per-call cost at KV about 131072 is nearly independent of the query count (64.02 ms
  for 2048 queries, 67.66 ms for 4096).
  Evidence: `S5/micro/chunk_size/l3_c2048.json`, `l3_c4096.json`.
  Affected path: chunked full-layer prefill at long context.
  Control or comparison: FLOPs double between the two; stated roofline bound in `OD/README.md` lines 433-435.
  Likely subsystem: chunked SDPA work distribution and KV re-reads per query chunk (unverified).
  Investigation performed: none by the stage beyond the q_chunk 64/128 sweep.
  Resolution: more-work-needed (Required Work item 1).
- Observed anomaly: unidentified `UnaryOpType::FILL` (3.1-3.4 us) before every decode `sparse_matmul`.
  Evidence: `E10/profiles/m*_*/ops_perf_results.csv.gz`, `decode_step_ops.txt` op 29 (linear) / 43 (full).
  Affected path: routed expert down projection, all profiles.
  Control or comparison: `sparse_matmul_device_operation.cpp` compact-output path skips the fill.
  Likely subsystem: `sparse_matmul` output allocation with `nnz=None`.
  Investigation performed: none by the stage; mislabeled by reviews 2 and 3.
  Resolution: more-work-needed (Required Work item 2).
- Observed anomaly: masked 4096 chunk slower than the unmasked 4096 chunk on every profile and layer kind.
  Evidence: `E10/prefill_timing/*.json`.
  Affected path: prefill tail chunk of non-aligned prompts.
  Control or comparison: 4096-token rows (same chunk, no mask).
  Likely subsystem: mask construction and masked GDN/attention paths (unverified).
  Investigation performed: none.
  Resolution: more-work-needed (Required Work item 4).
- Observed anomaly: first warm prefill run is the slowest in all 15 high-variance cells.
  Evidence: `E10/prefill_timing/*.json` (index 0 holds the maximum in each of the 15 cells).
  Affected path: short-prompt prefill after the compile run.
  Control or comparison: runs 2-6 of the same cells.
  Likely subsystem: residual first-use allocation or program setup (unverified).
  Investigation performed: classified by the stage as noise.
  Resolution: controlled for the table (median excludes it); cause open, see Other Concerns.
- Observed anomaly: random-activation decode minimums 0.0004-0.0010 lower than chain 9.
  Evidence: `E10/b32_*`, `E10/accept_positions_m*`, `E10/accept_context_m*`, `E10/accept_watcher_m*`; `E9` same rows.
  Affected path: decode MoE tail (BFP8 routed sum required by the fused addcmul).
  Control or comparison: `E9` (source before the change), op micro `S5/micro/shared_addcmul/`, BF16-weight layer screen
  `S5/trials/shared_addcmul_bf16w/` (+2.3 to +5.4 us, recomputed).
  Likely subsystem: BFP8 addend precision.
  Investigation performed: op-level attribution and an alternative screen.
  Resolution: controlled; full-model check owed by the datatype sweep (see the p150x2 concern above).
- Observed anomaly: p150x2 on devices 0,1 is 6-7 us slower than on devices 2,3.
  Evidence: `E10/placement_p150x2_01_*`.
  Affected path: p150x2 placement only.
  Control or comparison: chain 7 shows the same gap; PCC identical.
  Likely subsystem: board or link placement (unverified).
  Investigation performed: none.
  Resolution: open; the selected profile uses devices 2,3. Not blocking.

## Review 3 Required Work: resolution

- Planner change and positions/reuse coverage: resolved. `E9` and `E10` rerun `accept_positions`, `accept_reuse` and
  `real_inputs` on all profiles on the final source; README text and `E10/source_delta.diff` (which now includes
  `tests/test_multichip_plan.py`) are correct.
- Mid-length mesh prefill: resolved on p150x4 (1300 tokens 7.95 -> 6.47 ms, recomputed from `OD/evidence8` and `E10`);
  p150x2 trade-off measured, a cap tried, residual loss disclosed (see Other Concerns).
- Shared-expert fold: resolved. Two fold variants measured (`S5/trials/shared_fold/`: interleaved pairs +7.5 / +7.0
  us; variant 2 smoke plus profile), BFP8 fused addcmul adopted with interleaved pairs (`S5/trials/shared_addcmul_bfp8/`,
  x4 -2.3 / -2.7 us recomputed), BF16-weight variant rejected with interleaved pairs.
- Prefill chunk size: not resolved (Required Work item 1).

## Verified Claims

- Source identity: the live `AP/tt` and `AP/tests` trees equal `E10/source_snapshot.tar.gz` (`diff -r`), every file
  hash equals `E10/source_manifest_end.json` (54 files), start and end manifests are identical, `git_head`
  5d4c0441 matches HEAD, and the out-of-autoport diff hash is the empty-input SHA-256. The BF16-weight screen ran
  after the chain (14:35-14:51 UTC) with stable source hashes and was reverted. `E10/memory_handoff_recheck.json`
  binds the same `tt/optimized_decoder.py` hash.
- Chain 10 (`S5/evidence_chain10.log`, 12:38-14:33 UTC = 08:38-10:33 ET): 243 top-level steps, all `rc=0`,
  SOURCE_STABLE, CHILD_FAILURES 0, CPU pytest 1207 passed (0 failed, 0 skipped), device pytest 12 passed.
- 191 harness rows (headline 18, aligned 9, lengths 126, 128-step 18, B32 6, long watcher 12, placement 2): all have
  `runtime_audit` true and bitwise prefill repeat.
- Headline decode medians recompute exactly: aligned x4 257.4, x2 289.8, p150 368.9 us; full 260.6 / 293.3 / 370.6;
  linear 236.4 / 277.5 / 369.8. Chain 9 values recompute (260.1 / 263.3 / 239.3 on x4). Stage 4 values recompute from
  `AP/doc/multichip_decoder/evidence7/headline_*`.
- 40-layer estimates recompute: decode 9.67 / 11.22 / 14.78 ms; prefill at 4096 about 427 ms on x4.
- Repeated prefill table (true median of 6 warm runs) recomputes cell for cell; all 13 Stage 4 lengths are faster on
  all 6 rows; chain 9 vs chain 10 per-length difference at most 0.38 ms.
- Length rows: min HF prefill 0.99931 (`len_m1_full_100`), min HF decode 0.99818 (full 1100), min control prefill
  0.99983, min control decode 0.99975, 78 rows with control tensors.
- Context contract: 12 native-context artifacts exist and match `AP/doc/context_contract.json`; prefill PCC
  0.99822-0.99949, decode 0.99785-0.99983; `capability_reduction` null.
- Profiles: kernel sums and op counts (208.7/38, 251.3/52, 247.3/38, 279.0/52, 341.6/35, 358.3/49) and the family
  table recompute; prefill x4 106 ops / 10.50 ms and 111 ops / 10.73 ms recompute. Fused addcmul row is LoFi
  BF16 x BFP8, `MinimalMatmulConfig(K_block_size=4, N_block_size=8)` on x4.
- Long-context table recomputes from `E10/long_context/*.json`; all chunks and decode finite.
- Capacity per-layer weights (608.6 / 596.1 MiB on p150 etc.) recompute from `E10/capacity_model.json`.
- No credential strings found in the stage evidence or run scripts (plain-text scan).

## Scope Inspected

- Goal/skill paths: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/05-05-optimized-multichip-decoder.prompt.txt`,
  `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`, `/home/hous/dev/qwen3.5-35b-a3b/experiment-provenance-handoff.json`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/optimize/SKILL.md` (sections
  through OPT-008). The multichip, tt-device-usage and tt-enable-tracing skills were not reread; Stage 5 did not change
  the trace contract after chain 7, and reviews 1-3 covered them.
- Artifact paths: `OD/README.md`, `OD/work_log.md` (review 3 follow-up onward in full, plus the review 2 follow-up and
  early CCL rows), `OD/stage_review3.md`, `E10/` (manifests, snapshot, source delta, headline, aligned SDPA, lengths,
  prefill timing, accept_*, watcher_long logs and kernel name files, long context, profiles, capacity, roofline,
  memory recheck), `E9/` headline, aligned SDPA and prefill timing, `OD/evidence8/prefill_timing/`,
  `AP/doc/multichip_decoder/evidence7/` headline and prefill timing, `AP/doc/context_contract.json`,
  `S5/evidence_chain10.sh`, `S5/evidence_chain10.log`, `S5/evidence_chain11.sh`, `S5/process_profiles1{0,1}.sh`,
  `S5/trials/{shared_addcmul_bfp8,shared_addcmul_bf16w,shared_fold}/`, `S5/micro/{shared_addcmul,ccl_flags,chunk_size,
  sdpa_prefill,direct_allreduce_x2.json,direct_allreduce_x4.json}`, checkpoint `config.json`.
- Code paths: `AP/tt/optimized_decoder.py` (policies, `_OptimizedShared.addcmul`, decode experts `sparse_matmul`, masked
  prefill), `AP/tt/multichip_decoder.py` (mesh policies, dispatch route-map guard),
  `TM/ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp`,
  `TM/tt_metal/impl/debug/watcher_server.cpp`.
- Commands run: `git status`, `git log`, `git rev-parse`, `ls`, `grep`, `awk`, `sed`, `diff -r`, `tar`, `sha256sum`,
  `zcat`, `date`, and Python over JSON, CSV, XML and logs (scratch in
  `/tmp/claude-2005/-home-hous-dev-qwen3-5-35b-a3b/5bead60f-0645-4ca5-b27d-6b0f6acbc38f/scratchpad/review4/`). No
  device, server, build or device test.

## Residual Risk

- The tie-aware gate accepts any expert set inside a 1-ULP band; a router bias smaller than one BF16 ULP is not
  detectable.
- MoE prefill drift has no root cause; `generalized_moe_gate` still runs in decode between serving prefills.
- Long-context accuracy beyond single-step 262143 rows is unmeasured; long-context timing uses random hidden input.
- B > 1 decode is a per-row loop (7.5 ms per layer step at B32 on x4); it dominates the concurrency-32 sweep and is
  owned by Stage 7.
- The BFP8 routed sum lowers random-activation decode minimums on every profile; full-model accuracy is unmeasured.
