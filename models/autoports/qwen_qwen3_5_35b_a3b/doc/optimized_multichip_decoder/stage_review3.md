# Stage Review

Stage 05, optimized multichip decoder. Model: Qwen/Qwen3.5-35B-A3B, revision
59d61f3ce65a6d9863b86d2e96597125219dc754. Reviewer: fresh independent subagent, review 3. Review date: 2026 Oct 12,
about 05:45 Eastern. I did not open a device, start a server, build, or run a device test. I ran CPU-only analysis of
existing artifacts and two CPU-only planner scripts that import the autoport modules without opening a device.

Abbreviations: TP (tensor parallel), EP (expert parallel), MoE (mixture of experts), GDN (Gated DeltaNet, the
linear-attention layer), CCL (collective communication), AR (all-reduce), PCC (Pearson correlation coefficient), HF
(Hugging Face reference), BF16 (16-bit brain float), BFP8/BFP4 (8/4-bit block float), FP32 (32-bit float), L1 (core
SRAM), DRAM (device memory), KV (key/value cache), SDPA (scaled dot-product attention), RoPE (rotary position
embedding), TTFT (time to first token), ISL (input sequence length), ET (Eastern Time), UTC (Coordinated Universal
Time).

Short names:
- `AP` = `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b`
- `OD` = `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder`
- `E7` = `OD/evidence7`, `E8` = `OD/evidence8`
- `S5` = `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage05`

Verdict: more-work-needed

The review 2 work is mostly done with real evidence: the full-attention decode audit and its candidates, the GDN
in-place reads, the router HiFi2 change, the real-weight SDPA LoFi and in-projection LoFi screens, the MoE merge rule,
the roofline, the aligned rows with audit and 3 runs. Headline numbers recompute exactly. Provenance is clean: the
live `AP/tt` and `AP/tests` trees equal `E8/source_snapshot.tar.gz` byte for byte, and the `E8` start, end and
supplement manifests are identical. The remaining required work is one false coverage claim after the planner change,
and three untried or weakly dismissed optimizations, two of which review 2 asked for directly.

## Required Work

- P2: The README claims that the planner change affects no chain 7 acceptance row except the 3-token full rows. That
  is false. Linear positions and reuse rows changed plan and were not rerun on the final source.
  Evidence:
  - `OD/README.md` line 18: "Plans for all chain 7 rows except the 3-token full-layer rows are unchanged ... Chain 7
    decode, profile and acceptance numbers therefore describe the final source."
  - `E7/accept_positions_m*/linear.json` and `E7/real_inputs_m*/positions_0.json` prefill per-slot prompts of 257 and
    513 tokens (`prompt_lengths`). `E7/accept_reuse_m*/linear.json` and `E7/real_inputs_m*/reuse_0.json` have a 513
    case and a 1153 case with cuts [1, 3, 127, 129, 160, 513] (segments start 160 length 353, start 513 length 640).
  - I ran the live planner (`_chunk_plan`) against the chain 7 costs (`E8/source_delta.diff`) for every chain 7 row
    shape on tp 1, 2 and 4 (script in the scratchpad, CPU only). Changed plans, all profiles, linear layer:
    257 (128 + 256 masked -> 256 + 1 step), 513 (128 + 256 + 256 masked -> 256 + 256 + 1 step), start 160 length 353
    (... 128 + 256 masked -> ... 256 + 1 step), start 513 length 640 (... 256 + 256 masked -> ... 256 + 256 + 1 step).
    The only other changed row is full 3 tokens, which chain 8 covers.
  - Chain 8 (`S5/evidence_chain8.sh`) reruns only batch-1, slot-0, start-0 length rows, 5 watcher runs and timing. The
    new "chunk plus one eager step" tail is not exercised on the final source at a request slot other than 0
    (positions B32), with two live slots (reuse B2), or after a non-zero cached prefix (reuse cuts).
  Why this matters: `$stage-review` requires coverage of changed chunking paths, awkward tails and sibling consumers;
  earlier evidence does not cover a later change. The README states the opposite.
  Required next step: rerun `accept_positions`, `accept_reuse` and the real-input positions/reuse rows for the linear
  layer on all three profiles on the final source (with `TT_METAL_TRACE_ALLOC_TRACKING=1` as before). Correct the
  README sentence and the `E8/source_delta.diff` scope note (the diff also omits the `tests/test_prefill_plan.py`
  change, which the snapshots show).

- P2: Medium-length prefill on meshes still loses 10-18% to a single masked chunk, because the planner keeps the
  single-chip chunk and row costs. Review 2 asked for a cost-based choice; only the MoE piece count changed.
  Evidence (`E8/prefill_timing/p150x4_layer0.json`, warm medians, linear layer, x4):
  - 1300 tokens = 7.96 ms, but 2049 tokens = 6.74 ms. 1300 runs 3 chunks (256 + 1024 + 32 masked); a 2048 masked
    chunk costs about 6.6 ms (2049 = 2048 + 1 step).
  - 2400 = 10.97, 2600 = 11.88, 3300 = 11.80 ms, but 4096 = 10.61 ms. These run 2-3 chunks.
  - From the same table, each extra chunk costs about 2.5-2.8 ms on x4 (2176 = 128 + 2048 is 9.48 ms vs 2049 at
    6.74 ms). The planner uses `chunk_cost` 0.5 and `row_cost` 0.003 for every profile (`_PLAN_COST`,
    `AP/tt/optimized_decoder.py` line 1819; `mesh_policy` overrides only `plan_start_step_cost`). With these costs
    the ladder always looks cheaper than one masked chunk.
  - `S5/micro/plan_cost/` screens 2, 3, 5, 8, 257, 1025, 2049, 3073, 4097, 4100, 4353. No forced single-chunk
    measurement exists for 1100-2047 or 2176-4095.
  - `OD/README.md` line 66 explains the 2176-4095 gap in prose and does not mention 1300 > 2049. p150 is monotonic.
  Why this matters: TTFT is an explicit `$optimize` target, and the deliverable sweep includes short and medium ISL.
  At 40 layers the x4 linear loss alone is about 30 x 1.2 ms = 36 ms per prompt near 1300 tokens.
  Required next step: fit per-profile chunk and row costs from the measured table (or add a mesh cost set), and
  measure default vs forced single masked chunk at, for example, 1300, 1500, 2400, 2600 and 3300 on x4 and x2 for
  both layer kinds. Adopt the faster plan, then rerun the CPU plan sweep, the affected length rows and the repeated
  prefill table.

- P2: The shared-expert decode tail (about 15 us per layer on x4, 40 layers) has no valid attempt at the fold that
  review 2 required. The rejection reasons do not hold on the target mesh.
  Evidence:
  - `E7/profiles/m4_linear_attention/decode_step_ops.txt`: shared down matmul 4.64 us (op 33), sigmoid `Unary` 3.40
    us on 110 cores (op 29), `BinaryNg` 4.80 and 2.45 us on 110 cores (ops 34-35). Total 15.3 us of 210.1 us, about
    0.6 ms per token at 40 layers (6% of the 9.77 ms estimate).
  - `OD/work_log.md` line 973: "Shared-expert fold into the routed `sparse_matmul` (9th slot) needs a 257-expert
    stacked down weight. On p150 ... the fold needs a full restack." That is a load-time cost, not a runtime blocker.
    On p150x2 and p150x4 the decode expert weights are a separate TP copy (`OD/README.md` line 443), so the fold does
    not touch the prefill tensor. The shared expert has the same shape as a routed expert
    (`shared_expert_intermediate_size` 512 = `moe_intermediate_size` 512, checkpoint `config.json`), and the gate/up
    kernel already runs it as an extra slot.
  - The only measured alternative, `dit_minimal_matmul_addcmul_fused` (+23 us), used a BF16 copy of the BFP8 shared
    down weight, no program config, and an extra multiply for the scale (`S5/trials/apply_shared_addcmul.py`). This
    is a dtype-incompatible comparison, not a rejection of the family.
  Why this matters: `$stage-review` does not accept a rejection without an adapted attempt for a candidate that
  removes ops. The goal says no optimization is deferred.
  Required next step: on x4 (and x2), append the shared down weight as slot 256 of the decode down tensor, add id 256
  with the sigmoid gate value as the 9th score to `deepseek_moe_fast_reduce_nc_fused`, and measure traced decode and
  PCC. If an op contract blocks 9 rows, record the exact error and try the next adaptation (for example a padded
  score row) or use `$autofix`. Also record the addcmul candidate with the BFP8 weight and a tuned config, or the
  exact dtype blocker.

- P2: Prefill chunk size was never compared. The stated bound refers to an op that is not in the final path.
  Evidence:
  - `OD/README.md` line 318: "Prefill chunk size stays 4096 ... A larger chunk is bounded by L1/CB sizes of the
    dispatch path (fused-combine L1 overflow at 4096 rows) and was not measured." The fused combine was rejected
    (`S5/trials/fused_combine/`) and is not in the final path. The policy already separates MoE pieces from the chunk
    (`prefill_moe_pieces`).
  - `E7/long_context/m4_layer0_131072.json`: 32 chunks of about 10.9 ms each. The prefill timing table shows about
    2.5 ms fixed cost per chunk on x4. At 131072 tokens that is about 80 ms per layer, about 3 s of the 22.6 s
    layer-sum TTFT estimate.
  - `$optimize` "Prefill Latency And TTFT": "Compare feasible chunk sizes against the same logical request." Review 2
    listed this as an open concern.
  Why this matters: ISL 131072 is a required sweep point and its TTFT is dominated by chunk count and full-layer SDPA.
  Required next step: measure 2048 (control) and at least one larger chunk (8192, with MoE kept in 4096-row pieces)
  at 131072 tokens on x4 for both layer kinds with the long-context harness. Adopt the faster legal size, or record
  the exact L1/CB allocation failure of the larger chunk.

## Other Concerns

- Router LoFi was rejected although it is faster (-1.5 to -2.3 us per layer) and passes every gate on real weights
  (control decode 0.99961 -> 0.99944 linear, 0.99957 -> 0.99918 full). "Routing change" is a reasonable quality
  concern but OPT-012 asks for a model-visible failure. Hand this candidate to the datatype sweep with full-model
  accuracy.
- SDPA prefill LoFi was rejected at 0.99534 (gate 0.995) on real weights at 262143 tokens. The "10 stacked layers"
  margin argument is a projection, not a measurement. It is about 13% of long-context full-layer prefill. Hand it to
  the datatype sweep with full-model long-context accuracy.
- Conv-state shift (4 ops, 5.7 us per linear layer, about 0.17 ms per token) is "Not pursued" without a measurement.
  The model already owns a custom GDN kernel; a state-output path in that kernel or a custom conv kernel would remove
  these ops.
- Time labels in `OD/work_log.md` mix ET and UTC. Chain 7 "02:54 to 04:34" is ET (log 06:54Z-08:34Z). The gate_l1,
  rope_joint and chain 8 times ("09:03", "09:15", "09:21 to 09:37") are UTC. "Review 2 follow-up: screens 01:24 to
  03:20" and "Defaults after review 2 (03:25)" contradict the chain 7 start at 02:54 ET; the files show the screens
  ended at 06:38Z (02:38 ET) and `evidence_chain7.sh` was written at 06:41Z (02:41 ET).
- `E7/roofline.json` lists the prefill router at HiFi4. The router is HiFi2 in decode and prefill since review 2.
- `timing.py` reports the upper middle of 6 warm runs as the median (for example x4 linear 2 tokens: reported
  2630.9 us, true median 2627.4 us). The effect is small; the label is not exact.
- p150x2 per-layer weights (585.8 MiB) stay close to p150 (608.5 MiB) because of the EP prefill plus TP decode
  expert copies. Concurrency 32 at ISL 131072 needs about 4.19M KV tokens; the x4 capacity model gives 3.29M at 32
  GDN slots. This is a Stage 7 decision, but it constrains the required sweep.
- Full-layer RoPE decode still costs 21.9 us in two `RotaryEmbeddingIndexed` ops on 110 cores each. Three
  alternatives were measured slower. The op split of those alternatives is blocked by a profiler source-location
  hash collision; that is a reasonable tt-metal issue to file.

## Hard-Check Gaps

- OPT-009 CCL table fields (`chunks_per_sync`, workers per link, buffers per channel) are still absent (third review).
- OPT-006 cumulative contract table with op-to-op gap per row is still spread over README sections.
- Chain 8 watcher covers x4 and p150 only; p150x2 has changed plans (full 2 and 3 tokens, linear 2 tokens, tails)
  without a watcher run on the final source.
- Device pytest (`E7/accept_pytest`) ran on the chain 7 source.

## Anomaly Ledger

- Observed anomaly: x4 linear prefill at 1300 tokens (7.96 ms) is slower than at 2049 tokens (6.74 ms); 2400-3300
  tokens are slower than 4096 tokens.
  Evidence: `E8/prefill_timing/p150x4_layer0.json`; planner output from the live `_chunk_plan`.
  Affected path: prefill chunk planner on meshes.
  Control or comparison: p150 is monotonic; 2049 and 4096 run one chunk.
  Likely subsystem: single-chip plan costs applied to meshes (fixed per-chunk cost about 5x underestimated).
  Investigation performed: MoE merge rule (piece count), step costs; chunk costs not refit.
  Resolution: more-work-needed.
- Observed anomaly: README coverage claim contradicts the planner diff for positions and reuse rows.
  Evidence: CPU planner check above; `E7/accept_positions_m*`, `E7/accept_reuse_m*`, `E7/real_inputs_m*`.
  Affected path: linear prefill tails at slot > 0, B2 reuse, non-zero cached prefix.
  Control or comparison: chain 8 B1 slot-0 rows of the same plan classes pass.
  Likely subsystem: documentation and evidence scope.
  Investigation performed: none by the stage.
  Resolution: more-work-needed.
- Observed anomaly: chain 8 `len_m*_linear_4353` failed (exit 1) with the recorded input.
  Evidence: `S5/evidence_chain8.log` lines 2589-2599; `E8/failed_input_too_short/`.
  Affected path: harness input length (4352 recorded tokens).
  Control or comparison: `E8/len_m*_linear_4353_random` with seeded random input pass (HF prefill 0.99955, decode
  min 0.99928-0.99935, audited, prefill repeat bitwise); `E8/source_manifest_supplement.json` equals the end
  manifest.
  Likely subsystem: test input, not model.
  Investigation performed: rerun with random input; failures kept.
  Resolution: controlled.
- Observed anomaly: watcher 8-step full-layer rows at 3 tokens show decode min 0.99662 (x4 and p150) while the 4-step
  length rows show 0.99863.
  Evidence: `E8/watcher/m{1,4}_layer3_3.json`; `E8/len_m*_full_3/default.json`.
  Affected path: full-layer decode after a 3-token masked prefill.
  Control or comparison: chain 7 positions rows show a similar 0.9964-0.9966 minimum; both profiles agree to 1e-6.
  Likely subsystem: router near tie at a later step (unverified; the watcher rows do not record tie data).
  Investigation performed: none.
  Resolution: above the 0.995 gate; record the tie data or state the step when the rows are rerun.
- Observed anomaly: MoE prefill nondeterminism (chains 2-5), not observed since the top-k prefill router.
  Evidence: `E7/nondeterminism/`.
  Resolution: controlled by measurement; root cause open; serving repeatability check remains owed (unchanged from
  review 2).

## Verified Claims

- Source identity: `E8/source_manifest_start.json` = `source_manifest_end.json` = `source_manifest_supplement.json`;
  live `AP/tt` and `AP/tests` equal `E8/source_snapshot.tar.gz`. `E7` start and end manifests are identical. The only
  `tt/` difference from `E7` is the planner cost change; `tests/test_prefill_plan.py` also changed. Git status shows
  no change outside the autoport.
- Chain 7: every `run_optimized_trials` END record has exit code 0 and `source_stable` true (122 records). Chain 8:
  30 records, 3 failures (the 4353 recorded-input rows, kept).
- Headline decode medians (x4 259.8 / 263.6 / 239.1, x2 291.2 / 294.4 / 278.6, p150 371.5 / 373.2 / 372.3 us),
  warm prefill at 4096, HF and control PCC ranges recompute from `E7/headline_*` and `E7/aligned_sdpa/`. All 27 rows
  have `runtime_audit` true.
- 40-layer estimates recompute: decode 9.77 / 11.27 / 14.88 ms; prefill 429 / 571 / 814 ms; Stage 4 18.37 ms and
  809 ms.
- Repeated prefill table recomputes from `E8/prefill_timing/`; every length that Stage 4 measured is faster on every
  row.
- 30 chain 8 length rows: plans as stated, prefill PCC 0.99950-0.99988, decode min at least 0.99863, audited,
  prefill repeat bitwise. 5 chain 8 watcher logs: no error, assert, hang or overflow line.
- Family table and kernel sums recompute from `E7/profiles/m*/decode_step_ops.txt` (210.1 / 253.6 / 246.6 / 278.3 /
  350.4 / 366.3 us).
- Long-context table recomputes from `E7/long_context/*.json`.
- Roofline bytes are consistent with the stored dtypes and TP split (for example x4 in-projection 2048 x 3136 BFP8 =
  6.72 MB; GDN state read plus write 1 MB per device).
- Capacity table recomputes from `E7/capacity_model.json`. `capability_reduction` is null; 262143/262144 rows exist
  on every profile and layer kind.
- CPU tests: `E8/cpu_pytest.xml` 1197 tests, 0 failures, 0 skipped.
- The in-projection sweep includes DRAM-sharded candidates with 1 and 2 readers per bank under HiFi2 and LoFi
  (`S5/micro/inproj_sweep.json`).

## Scope Inspected

- Goal/skill paths: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/05-05-optimized-multichip-decoder.prompt.txt`,
  `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/optimize/SKILL.md` (full).
  The multichip and tracing skills were not reread; review 2 covered them and Stage 5 did not change the trace
  contract after chain 7.
- Artifact paths: `OD/README.md`, `OD/work_log.md` (chain 6 onward in full), `OD/stage_review2.md`, `E7/` (headline,
  aligned_sdpa, accept_*, real_inputs, long_context, profiles step ops, roofline.json, capacity_model.json,
  decode_family_table.md, manifests, snapshot), `E8/` (all), `S5/evidence_chain7.log`, `S5/evidence_chain8.log`,
  `S5/evidence_chain8.sh`, `S5/micro/plan_cost/`, `S5/micro/inproj_sweep.json`, `S5/trials/{rope_joint,gate_l1,
  review2,review2b,review2c,review2d,topk_decode}/`, `S5/trials/apply_shared_addcmul.py`,
  `AP/doc/context_contract.json`, `AP/doc/multichip_decoder/evidence7/micro/prefill_timing7/`, checkpoint
  `config.json`.
- Code paths: `AP/tt/optimized_decoder.py` (`prefill_forward`, `_chunk_plan`, `_Projection` DRAM-sharded path, decode
  MoE and shared expert), `AP/tt/expert_decode.py`, `AP/tt/multichip_decoder.py` (mesh policy, dispatch),
  `AP/tests/run_positions.py`, `AP/tests/run_reuse.py`.
- Commands run: `git status`, `git log`, `ls`, `grep`, `awk`, `cmp`, `diff -r`, `tar`, `date`, Python over JSON and
  logs, and two CPU-only planner scripts (`plancheck.py`, `plan2.py` in the session scratchpad) that construct
  `OptimizedDecoder` without a device. No device, server, build or device test.

## Residual Risk

- The tie-aware gate accepts any expert set inside a 1-ULP band; a router bias smaller than one BF16 ULP is not
  detectable.
- MoE prefill drift has no root cause; the decode gate op still runs between serving prefills.
- Long-context accuracy beyond single-step 262143 rows is unmeasured; long-context timing uses random hidden input.
- Concurrency 32 at ISL 131072 does not fit the x4 capacity model with the EP plus TP expert layout.
- B > 1 decode is still a per-row loop (Stage 7 owner); it dominates the concurrency-32 sweep.
