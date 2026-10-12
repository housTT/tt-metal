# Stage Review

Stage 05, optimized multichip decoder. Model: Qwen/Qwen3.5-35B-A3B, revision
59d61f3ce65a6d9863b86d2e96597125219dc754. Reviewer: fresh independent subagent, review 5. Review date: 2026 Oct 12,
about 17:30 Eastern. I did not open a device, start a server, build, or run a device test. I ran CPU-only reads of
existing artifacts, the ttnn source and small Python scripts over JSON, XML and log files.

Abbreviations: TP (tensor parallel), MoE (mixture of experts), GDN (Gated DeltaNet, the linear-attention layer), CCL
(collective communication), PCC (Pearson correlation coefficient), HF (Hugging Face reference), BF16 (16-bit brain
float), BFP8 (8-bit block float), FP32 (32-bit float), L1 (core SRAM), DRAM (device memory), KV (key/value cache),
SDPA (scaled dot-product attention), TTFT (time to first token), ISL (input sequence length), FLOP (floating-point
operation), UTC (Coordinated Universal Time), ET (Eastern Time), CPU (host processor), JSON (JavaScript Object
Notation), XML (Extensible Markup Language).

Short names:
- `AP` = `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b`
- `OD` = `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder`
- `E12` = `OD/evidence12`, `E10` = `OD/evidence10`, `E11` = `OD/evidence11`
- `S5` = `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage05`
- `TM` = `/home/hous/dev/qwen3.5-35b-a3b/tt-metal`

Verdict: more-work-needed

Summary. All four review 4 Required Work items have real measurements now, and the headline numbers, source identity
and chain 12 results recompute from raw artifacts (see Verified Claims). Two gaps remain, both on the newest changes,
and both are small to close. (1) The new 8192-row full-attention step has no device accuracy evidence for a
continuation prefill (`start_pos` > 0). This is how serving will reach it, and it includes chunk starts that are not
multiples of the new `k_chunk_size` 512. The step also has no watcher run. (2) The ttnn `sparse_matmul` no-fill change
removes a safety zero-fill from a shared op. The op's own unit test, which has a 99.0-prefilled output case made for
exactly this contract, was not run after the change.

## Required Work

- P2: 8192-row attention step: no device evidence for continuation prefill or non-`k_chunk`-aligned chunk starts,
  and no watcher coverage.
  Evidence:
  - The merge rule (`AP/tt/optimized_decoder.py` lines 359-368) merges two unmasked 4096-row chunks when
    `start_pos + last[0] >= prefill_long_from`. `fits()` (line 318) requires only `absolute % 128 == 0` for a
    4096 chunk. A continuation that starts at a 128-aligned position, for example 8320, gives 8192-row steps at
    8320, 16512, and so on. 8320 is not a multiple of 512.
  - New SDPA configs for these steps: q128/k512 on p150 and p150x2, q64/k256 on p150x4 (`sdpa_prefill_long`, recorded
    in every `E12/headline_*/default.json` `optimization` dict).
  - `TM/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp` lines 355-370: the legacy
    scalar `chunk_start_idx` path rejects starts that are not a multiple of `q_chunk_size` and of `k_chunk_size`
    ("Workaround for https://github.com/tenstorrent/tt-metal/issues/35225"). The model uses the tensor
    (`chunk_start_idx_tensor`) path, which has no such check. The compute and writer kernels divide the start by
    `q_chunk_size` (`kernels/compute/sdpa.cpp` line 104, `kernels/dataflow/writer_interleaved.cpp` line 171).
    I did not verify whether the issue 35225 failure mode also applies to the tensor path.
  - Every device accuracy row that runs an 8192 step starts at position 0 in one call: `E12/long_chunk/` (16384 and
    20000 tokens, `run_layer` without `--split`) and `E12/accept_context_m*` (262143 and 262144 tokens,
    `AP/tests/run_fused_acceptance.py` lines 59-75). The continuation runs in `E12/long_context/` (8192-token
    external chunks, starts at multiples of 8192) are timing only, with random input and no reference. The reuse
    and positions acceptance rows do not reach position 8192. The CPU test `test_long_chunk_merge`
    (`AP/tests/test_prefill_plan.py` lines 113-119) has one nonzero start (4096) and checks the plan only.
  - Existing evidence for non-aligned starts covers only the old 4096-row path with k256 (for example the reuse
    segment that ends with a masked 1024 chunk at position 640, `E12/accept_reuse_m*`). It does not cover k512.
  - Watcher: `OD/README.md` "Trace, determinism, audit, watcher" states "Not exercised under the watcher: the
    8192-row attention step and the long SDPA configs (no watcher run exceeds 4096 tokens)."
  - Serving relevance: the required latency sweep (ISL 131072, `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`) sends
    long prompts that the vLLM scheduler splits into chunked prefill calls with `start_pos` > 0. Prefix caching can
    give block-aligned starts that are not multiples of 512.
  Why this matters: `$stage-review` requires coverage of affected paths, nearby boundary values and awkward tails for
  a changed chunking rule. Earlier evidence does not cover a later change. The goal requires watcher-clean evidence
  and stress coverage that matches the risk.
  Required next step: on p150 (k512) and p150x4 (q64/k256), full-attention layer 3, run `run_layer --trace --audit
  --sparse-reference` with `--split` so that the second call starts at a 128-aligned position that is not a multiple
  of 512 and contains at least two 8192-row steps (for example `--length 26000 --split 8320`; one more case with a
  512-aligned split such as 8192 is a useful control). Compare prefill and decode PCC with the same run under
  `prefill_long_chunk` 0. Run one of these cases per profile under `TT_METAL_WATCHER=10`. If a misaligned start
  fails, restrict the merge to `k_chunk_size`-aligned starts and rerun the affected rows.

- P2: ttnn `sparse_matmul` no-fill change: the op's existing unit test was not run after the change.
  Evidence:
  - `E12/ttnn_patch.diff` (equal to the live `git diff HEAD -- ttnn`, sha256 066fd7da...): `compact_output` is now
    true for every indexed call, so `create_output_tensors` skips `ttnn::zeros_like` for both a new output and a
    caller-supplied output.
  - `TM/tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul_indexed.py` exists (525 lines).
    `test_indexed_optional_output` (line 383) prefills the preallocated output with 99.0. Before the change the op
    zero-filled that tensor; now any tile that the writer does not reach keeps 99.0. The file also has a partial-N
    case, `n=96` on a (2, 1) grid with `per_core_n` 2 (line 422 parametrization).
  - No Stage 5 log, script or work-log entry runs this file (`grep -rl sparse_matmul_indexed
    /home/hous/dev/qwen3.5-35b-a3b/runs/main/` finds only Stage 3 transcripts). `S5/evidence_chain12.sh` defines a
    `kernels()` block for the model-local kernel tests but does not call it.
  - Model-side evidence is good: decode PCC is identical to chain 10 on all 18 headline runs, the FILL row is gone
    from `E12/profiles/*/decode_step_ops.txt`, and the writer source
    (`TM/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp`
    lines 316-326) writes every indexed slot. That covers this model's one shape, not the op contract.
  Why this matters: the stage changes the output-allocation contract of a shared ttnn op and will commit it. The op
  already has a direct regression test for that contract. `$stage-review` treats a plausible bug in a
  stage-critical subsystem as required work, and a cheap existing check that would confirm or refute it should run
  before the commit.
  Required next step: run `pytest TM/tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul_indexed.py` on one
  device with the current build (`_ttnncpp.so` sha256 b38c43f3...). Record the log under `S5/trials/indexed_nofill/`
  and the result in `OD/work_log.md`. If a case fails, fix the op or keep the fill for that case, then rerun the
  affected decode rows.

## Other Concerns

- README contradiction: `OD/README.md` lines 682-683 say "`../context_contract.json` does not list this table under
  `new_persistent_buffers`". The contract does list it ("full-attention index-offset table for the 8192-row long
  chunk ... 1048576 bytes per full layer"), and `OD/work_log.md` says the contract was regenerated with it. Remove the
  stale sentence.
- Long-chunk screen lineage. The x4 and p150 arms of `S5/micro/long_chunk/run.sh` ran on `tt/optimized_decoder.py`
  sha256 337b25ac... with the policy key `"sdpa": {"prefill_long": ...}`; the p150x2 arms (`run_x2.sh`) ran on
  a37bfd21... with the key `sdpa_prefill_long`; chain 12 runs 52e4df7a.... Neither intermediate source is preserved,
  and `run.log` ends `SOURCE_CHANGED` (only a `.pyc` hash moved; the `.py` hashes are equal). Under
  `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md` these arms are exploratory. The selected configs are reproduced on the
  final source: chain 12 chunks 3-16 equal the screen per chunk within 0.23 ms (0.25%) on x4, 0.95 ms (0.33%) on p150 and
  2.12 ms (1.94%) on p150x2 (recomputed).
  The rejections of the other arms (for example x4 k512, 40.45 vs 39.38 ms per 4096 tokens) rest on the exploratory
  runs only. The margins are 1-3% on x4, so this does not change the result materially.
- Long-context SDPA efficiency has no bound. The README withdrew the wrong bound but gives no replacement ("not
  derived here"). By my estimate (unverified on hardware) the x4 4096-row call at start 126976 does about 2.1e12
  FLOPs per device, which is about 7 ms at HiFi2 peak, against 57.2 ms measured (about 12%). Full-layer SDPA is about
  12 s of the 22 s layer-sum TTFT at ISL 131072. q64 being faster than q128 on x4 points at work distribution
  (4 local query heads over 110 cores), not DRAM. The q/k chunk sweep and exact L1 limits are recorded, so this is
  not required work here. The full-model stage should derive the bound and name the limitation for the ttnn backlog
  (`$optimize` "Name the limitations precisely").
- Masked-chunk cost is attributed, not removed: 12-18 extra planned-path ops, +0.26 to +0.48 ms kernel and up to
  +0.89 ms span per layer at 4095 tokens (`S5/micro/masked_chunk/`). The trace-safety reason is consistent with the
  `prefill_forward` routing (`AP/tt/optimized_decoder.py` lines 220-227). At 40 layers this is about 5-8% of a
  3000-token prefill. No reduced-crossing variant was tried. A later stage should try one, for example a fused
  gather/untilize for the input rows.
- The 40-layer TTFT estimate at ISL 131072 (22.0 s) includes the first-use build of the 8192-row program in chunk 2
  of every layer (x4: 73.8 ms cold vs 30.8 ms warm, `S5/micro/long_cold/`). The estimate is conservative by about
  0.4 s. Serving warmup must include the 8192-row step and the long SDPA configs; "Work owned by later stages" lists
  only the tail-length shapes.
- The index-offset table for count 8192 is per decoder instance (`AP/tt/optimized_decoder.py` lines 672-677,
  `result._index_offsets`), not shared. The capacity model counts it once. The full model must share it or count
  10 x 1 MiB.
- Carried from review 4 and still open: p150x2 keeps the BFP8 routed sum with no speed gain; the p150x2 planner loses
  0.3-0.9 ms at 2600-3000 tokens; the systematic first-warm-run penalty is not named; the BFP8 CCL dtype family
  result is only in `OD/work_log.md`, not in the README CCL table.

## Hard-Check Gaps

- The chain does not record the `_ttnncpp.so` hash. I checked it after the fact: `TM/build_Release/lib/_ttnncpp.so`
  sha256 b38c43f3... equals `S5/trials/indexed_nofill/newbuild_end.sha256`, and its modification time (15:40 UTC)
  is before chain 12 (19:07 UTC). Add the library hash to the start and end manifests in future chains.
- The model-local kernel tests (`kernels()` in `S5/evidence_chain12.sh`) are not run in chain 12. The gate-writer
  conv shift has layer-level coverage (126 length rows, B32, positions, reuse and 12 watcher logs that list
  `gate_writer_shift.cpp`) but no kernel-level test.
- tt-perf-report advice "No program_config specified" on the fused addcmul row is still not recorded as misleading
  advice (review 4).

## Anomaly Ledger

- Observed anomaly: expert-down `sparse_matmul` kernel time fell by 1.0-2.2 us (x4 13.0 -> 12.0, x2 21.0 -> 19.8,
  p150 30.0 -> 28.0 us) when only the preceding FILL op was removed.
  Evidence: `E10/profiles/*/decode_step_ops.txt` vs `E12/profiles/*/decode_step_ops.txt`.
  Affected path: decode routed expert down projection.
  Control or comparison: same op, program config and dtype in both chains.
  Likely subsystem: op start overlap with the preceding op in the profiler timeline (negative gaps appear in the op
  lists), unverified.
  Investigation performed: none by the stage.
  Resolution: open, not blocking (favorable direction; layer host time matches the kernel change).
- Observed anomaly: long-context chunk 2 (first 8192-row step) slower than chunk 3 on every full row.
  Evidence: `E12/long_context/m*_layer3_131072.json`.
  Affected path: first use of the 8192-row step.
  Control or comparison: two-pass run `S5/micro/long_cold/` (second pass 30.8 ms at position 8192).
  Likely subsystem: program build on first use.
  Investigation performed: two-pass control.
  Resolution: controlled.
- Observed anomaly: chain 11 linear reuse runs failed the trace allocation check.
  Evidence: `E11/accept_reuse_m4/linear.log` ("Found 1 device buffer(s) still alive before trace replay"),
  `S5/evidence_chain11.log` (6 steps with rc=1).
  Affected path: `conv_shift="gate"` decode under trace.
  Control or comparison: `S5/micro/conv_ref_fix/` (rc 0 on all profiles), chain 12 (all reuse runs pass).
  Likely subsystem: `_conv_current` reference held across capture.
  Investigation performed: root cause and fix; the chain 11 to chain 12 source delta is only that fix (verified by
  snapshot diff).
  Resolution: fixed.
- Observed anomaly: masked 4096 chunk slower than the unmasked chunk.
  Evidence: `S5/micro/masked_chunk/L{0,3}_{4096,4095,3000}*.json`.
  Affected path: last chunk of non-aligned prompts.
  Control or comparison: 4096-token unmasked chunk (tiled path).
  Likely subsystem: planned-path row-major crossings and GDN padding mask.
  Investigation performed: Tracy profiles with op attribution.
  Resolution: controlled (attributed); optimization left open (Other Concerns).
- Observed anomaly: p150x2 on devices 0,1 slower than on devices 2,3 by 5-7 us.
  Evidence: `E12/placement_p150x2_01_*`.
  Affected path: p150x2 placement only.
  Control or comparison: chains 7 and 10 show the same gap.
  Likely subsystem: board or link placement, unverified.
  Investigation performed: none.
  Resolution: open, not blocking (the profile uses devices 2,3).

## Review 4 Required Work: resolution

- Long-context chunk size and SDPA query chunk: resolved with measurements. SDPA op micro: q256 k512, q512 k256 and
  q512 k512 exceed L1 with an exact circular-buffer error; q256 k256 is about 2x slower. The 8192-row attention step
  was made legal by keeping MoE pieces at 4096 rows (`_moe_pieces`, `AP/tt/optimized_decoder.py` lines 1884-1888) and
  adopted per profile. Recomputed from `S5/micro/long_chunk/*.json`, ms per 4096 tokens after the first chunk: x4
  40.46 (control) / 41.48 / 39.38 / 40.45; x2 73.72 / 59.85 / 64.87 / 58.48; p150 108.36 / 95.11 / 115.59 / 92.43.
  The final default reproduces the selected configs (chain 12 per-chunk times). The wrong KV and FLOP bound was
  withdrawn. Remaining gap: continuation and watcher coverage (Required Work item 1).
- Expert-down zero-fill: resolved with a ttnn change, a sequential-build screen (-2.6 to -3.0 us on all 6 rows) and
  chain 12 profiles (the FILL row is gone; 38 -> 33 ops on x4 linear). Remaining gap: op unit test (Required Work
  item 2).
- Conv-state shift: resolved. The shift moved into `AP/tt/kernels/gdn_decode/gate_writer_shift.cpp`. I read the
  kernel: each gate core reads its own `2 * width / heads`-byte column span of state rows 1-2 and the new row, waits on a
  read barrier, then writes rows 0-2; spans are disjoint (`AP/tt/gdn_decode.py` lines 112-130). The 3-op
  alternative was measured and rejected (p150 +4.4 us). HF and control decode minimums are identical to chain 10.
- Masked-chunk overhead: resolved as attributed with profiles (Anomaly Ledger).

## Verified Claims

- Source identity: `E12/source_manifest_start.json` equals `source_manifest_end.json` (byte compare). Every manifest
  file hash equals the live tree now. `E12/source_snapshot.tar.gz` equals the live `AP/tt` and `AP/tests` trees
  (`diff -r`, excluding `__pycache__`). `git_head` 5d4c0441 equals HEAD. The live `git diff HEAD -- ttnn` hashes to
  the manifest `git_diff_outside_autoport` and to `E12/ttnn_patch.diff`. The only non-autoport changes are the two
  ttnn files. The chain 11 to chain 12 source delta is the conv-reference fix only.
- `S5/evidence_chain12.log`: 255 step starts, 255 `END rc=0`, `SOURCE_STABLE`, `CHILD_FAILURES 0`, `CHAIN_DONE`.
  `S5/evidence_chain12.sh` differs from `evidence_chain11.sh` only in the output paths.
- Headline decode (median of 3 runs, each the median of 128 `traced_replay_host_us`): x4 full 257.0, linear 228.1,
  aligned 254.4 us; x2 290.5 / 268.1 / 287.7; p150 367.9 / 358.7 / 365.9. Chain 10: 260.6 / 236.4 / 257.4, 293.3 /
  277.5 / 289.8, 370.6 / 369.8 / 368.9. Warm prefill at 4096: x4 11.55 / 10.74 ms, x2 15.05 / 14.37, p150 20.52 /
  20.51. All match the README.
- Headline accuracy: prefill PCC, HF tie-aware decode minimum, strict decode minimum and control decode minimum are
  identical to chain 10 on every profile and layer kind.
- 40-layer decode estimate: 10 x 254.4 + 30 x 228.1 us = 9.39 ms (x4), 10.92 ms (x2), 14.42 ms (p150); 1.96x over
  Stage 4 on x4.
- Policies in the runtime rows: `conv_shift` "gate" on linear and "ops" on full layers; `prefill_long_chunk` 8192
  and `prefill_long_from` 8192 on full layers; `sdpa_prefill_long` q64/k256 on x4 and q128/k512 on x2 and p150;
  `sparse_indexed` true; KV dtype BFP8.
- 162 length, 128-step, B32 and long-chunk rows: all `runtime_audit` true, no prefill-repeat failure, minimum prefill
  PCC 0.99842 (long-chunk, random input), minimum decode 0.99536 (`b32_m1_full`). Long-chunk rows match the
  `S5/micro/long_chunk/accuracy/` "long" arms exactly; the x4 control arm is bitwise equal on prefill, as expected
  for q64 vs q128 with the same k chunk.
- CPU pytest: 1213 tests, 0 failures, 0 errors, 0 skipped (`E12/cpu_pytest.xml`).
- Watcher: 24 logs with no error, assert, trip or overflow line; `gate_writer_shift.cpp` is in the kernel list of all
  6 long linear runs and the 3 acceptance linear runs.
- Built library: `_ttnncpp.so` sha256 b38c43f3... equals the screened new build and predates chain 12.
- `E12/memory_handoff_recheck.json` binds `tt/optimized_decoder.py` sha256 52e4df7a..., the chain 12 source.
- `AP/doc/context_contract.json` `optimized_multichip_decoder_validation`: capability reduction null, supported
  context 262144, 12 native-context artifacts under `evidence12/`.

## Scope Inspected

- Goal/skill paths: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/05-05-optimized-multichip-decoder.prompt.txt`,
  `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/optimize/SKILL.md` (roofline,
  prefill, watcher and checklist sections). I did not reread the multichip and tt-enable-tracing skills; the trace
  contract change in this round (conv reference release) is covered by trace allocation tracking in chain 12.
- Artifact paths: `OD/README.md`, `OD/work_log.md` (review 4 follow-up to the end), `OD/stage_review4.md`,
  `E12/` (manifests, snapshot, ttnn patch, headline, aligned SDPA, lengths, long128, B32, long-chunk, long-context,
  watcher logs and kernel lists, profiles, memory recheck, CPU pytest XML), `E10/` headline and profiles, `E11/`
  manifest, snapshot and reuse logs, `S5/evidence_chain11.sh`, `S5/evidence_chain12.sh`, `S5/evidence_chain11.log`,
  `S5/evidence_chain12.log`, `S5/process_profiles1{1,2}.sh`, `S5/trials/indexed_nofill/`,
  `S5/micro/{long_chunk,long_chunk/accuracy,long_cold,masked_chunk,conv_ref_fix}/`,
  `AP/doc/context_contract.json`.
- Code paths: `AP/tt/optimized_decoder.py` (planner, `fits`, merge, `_moe_pieces`, index offsets, `prefill_forward`
  routing, `_convolve`, `_decode_kernel`, `_decode_heads`, decode experts), `AP/tt/gdn_decode.py`,
  `AP/tt/kernels/gdn_decode/gate_writer_shift.cpp`, `AP/tests/test_prefill_plan.py`,
  `AP/tests/run_fused_acceptance.py`, `AP/tests/run_layer.py` (`--split`),
  `TM/ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp`,
  `TM/ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp`,
  `TM/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp`,
  `TM/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp` and its compute and dataflow
  kernels, `TM/tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul_indexed.py`.
- Commands run: `git status`, `git log`, `git diff`, `ls`, `grep`, `sed`, `diff`, `cmp`, `tar`, `sha256sum`, `find`,
  `date`, and Python over JSON, XML and logs (scratch under
  `/tmp/claude-2005/-home-hous-dev-qwen3-5-35b-a3b/5bead60f-0645-4ca5-b27d-6b0f6acbc38f/scratchpad/r5/`). No device,
  server, build or device test.

## Residual Risk

- The tie-aware gate accepts any expert set inside a 1-ULP band; a router bias smaller than one BF16 ULP (unit in the
  last place) is not detectable.
- MoE prefill drift has no root cause; `generalized_moe_gate` still runs in decode.
- Long-context accuracy beyond single-call rows is unmeasured until Required Work item 1 runs; long-context timing
  uses random input.
- B > 1 decode is a per-row loop (Stage 7).
- The BFP8 routed sum lowers random-activation decode minimums on every profile; full-model accuracy is owned by the
  datatype sweep.
