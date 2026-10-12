# Stage Review

Stage 05, optimized multichip decoder. Model: Qwen/Qwen3.5-35B-A3B, revision
59d61f3ce65a6d9863b86d2e96597125219dc754. Reviewer: fresh independent subagent, review 6. Review date: Oct 12, about
17:45 Eastern. I did not open a device, start a server, build, or run a device test. I ran CPU-only reads of existing
artifacts and the source, and small Python scripts over JSON, XML and log files.

Abbreviations: PCC (Pearson correlation coefficient), HF (Hugging Face reference), SDPA (scaled dot-product
attention), KV (key/value cache), MoE (mixture of experts), GDN (Gated DeltaNet, the linear-attention layer), CPU
(host processor), JSON (JavaScript Object Notation), XML (Extensible Markup Language), UTC (Coordinated Universal
Time), ET (Eastern Time). File modification times in the artifacts are UTC; times in prose are ET.

Short names:
- `AP` = `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b`
- `OD` = `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder`
- `E12` = `OD/evidence12`
- `S5` = `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage05`
- `R5` = `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage05/micro/review5`
- `TM` = `/home/hous/dev/qwen3.5-35b-a3b/tt-metal`

Verdict: clean-pass

Summary. Both review 5 Required Work items are closed by device evidence, not prose. (1) Continuation prefill with
the 8192-row attention step ran on all three profiles at a split (8320) whose 8192-row steps start at positions that
are not multiples of the SDPA `k_chunk_size`, with a `prefill_long_chunk` 0 control and a 512-aligned split control.
PCC matches the controls, and decode PCC is identical, which shows the KV cache content is equivalent. One watcher run
per profile on that case is clean. (2) The `sparse_matmul` op tests ran on the rebuilt library: 59 passed in
`test_sparse_matmul_indexed.py` (including the 99.0-prefilled optional output case) and 29 passed, 2 skipped (Quasar
only) in `test_sparse_matmul.py`. Source and library hashes are stable across these runs and still equal the chain 12
manifest. The headline numbers recompute from raw JSON. Remaining items are documentation or later-stage concerns.

## Required Work

None.

## Other Concerns

- Stale README sentence, carried from review 5 and still not fixed. `OD/README.md` line 707-708: "`../context_contract.json`
  does not list this table under `new_persistent_buffers`." `AP/doc/context_contract.json` line 684 does list it
  ("full-attention index-offset table for the 8192-row long chunk ... 1048576 bytes per full layer"). The contract
  is correct; the README is wrong. Delete the sentence before the checkpoint commit.
- The watcher reruns (`R5/m{4,2,1}_split8320_watcher2.*`) have no script. `R5/run_split.sh` launches only the first
  watcher attempt (without `TT_METAL_WATCHER_NOINLINE=1` and without `TT_METAL_LOGS_PATH`). The watcher2 command
  line is reconstructable only from the logs (`-DWATCHER_NOINLINE` in the compile lines, the watcher log path in
  `watcher_logs_m*/`). `R5/watcher_source_start.sha256` has no `_ttnncpp.so` line and there is no watcher end hash.
  The bracket still holds: the start hash, the runtime `implementation_sha256` in each watcher2 JSON (52e4df7a...)
  and the current tree (all 55 manifest files match `E12/source_manifest_end.json`) agree, and `_ttnncpp.so`
  modification time (15:40 UTC) predates the runs. Record the exact command in `OD/work_log.md`.
- Probe coverage of the continuation rows is thin. `--sparse-reference` checks 16 prefill rows
  (`prefill_probes` 0, 1, 30, 31, 32, 126, 127, 128, 510, 511, 512, 13000, 25871, 25967, 25998, 25999). Only row
  13000 falls inside the misaligned 8192-row step at 8320; no row falls inside the second step at 16512. The
  decode PCC identity with the control (all 4 tokens, all profiles) shows the KV written by both steps is
  equivalent, and step 2 runs the same program with a start that is also not 256- or 512-aligned. This is
  adequate for this stage. The full-model stage should add a probe inside every long step.
- Carried from review 5, still open, owned by later stages: serving warmup must include the 8192-row step and the
  long SDPA configs (`OD/README.md` line 495 says so, but "Work owned by later stages" lists only tail-length shapes);
  the per-decoder index-offset table (count 10 x 1 MiB or share it); no long-context SDPA efficiency bound (my
  unverified estimate is about 12% of HiFi2 peak on p150x4 at start 126976); masked-chunk overhead attributed but not
  reduced; p150x2 keeps the BFP8 (8-bit block float) routed sum with no speed gain; p150x2 planner loses 0.3-0.9 ms
  at 2600-3000 tokens.

## Hard-Check Gaps

- Evidence chains do not record the `_ttnncpp.so` hash in the manifests. `R5/source_{start,end}.sha256` now do
  (b38c43f3...); add it to the chain manifest in later stages.
- The model-local kernel tests (`kernels()` in `S5/evidence_chain12.sh`) are defined but not called in chain 12.
  `gate_writer_shift.cpp` has layer-level and watcher coverage only.
- The split JSON does not record the executed prefill plan; the 8192-row steps are inferred from the policy dict
  (`prefill_long_chunk` 8192) and the CPU plan (`R5/plans.txt`). A runtime plan record would make this direct.
- tt-perf-report advice "No program_config specified" on the fused addcmul row is still not marked as misleading
  advice.

## Anomaly Ledger

- Observed anomaly: p150x4 watcher run on the split case failed at mesh open.
  Evidence: `R5/failed_watcher_no_noinline/m4_split8320_watcher.log`: `TT_FATAL program.cpp:3671: state.offset <=
  max_size`, "Program size (29264) too large for kernel config buffer (26624) on ACTIVE_ETH".
  Affected path: watcher build of the fabric Ethernet kernel on the 4-chip ring only.
  Control or comparison: `S5/evidence_chain12.sh` lines 85 and 100 set `TT_METAL_WATCHER_NOINLINE=1` for p150x4 for
  the same reason; `R5/m4_split8320_watcher2` with that flag passes.
  Likely subsystem: watcher instrumentation size on the Ethernet core.
  Investigation performed: rerun with the documented workaround; failed attempt kept.
  Resolution: controlled.
- Observed anomaly: p150x2 continuation prefill PCC with the long chunk is 0.000012 below the control (0.998586 vs
  0.998598); probe row 13000 is 0.99752 vs 0.99769.
  Evidence: `R5/m2_split8320_long.json`, `R5/m2_split8320_ctl.json`.
  Affected path: 8192-row step with SDPA q128/k512 on p150x2.
  Control or comparison: the 512-aligned split (`R5/m2_split8192_long.json`) gives the same 0.998586 and the same
  0.99752 at row 13000, so the gap is not caused by the misaligned start. On p150x4 the long config keeps k256 and
  the result equals the control exactly; on p150 (k512) the gap is 0.000003.
  Likely subsystem: different key-chunk accumulation order (k512 vs k256).
  Investigation performed: aligned and misaligned splits plus control (stage); cross-profile comparison (reviewer).
  Resolution: controlled.
- Observed anomaly: the first p150x2 and p150 watcher runs wrote no watcher log under `R5/`.
  Evidence: `R5/run_split.sh` (no `TT_METAL_LOGS_PATH`), `OD/work_log.md` review 5 follow-up.
  Affected path: evidence capture only.
  Control or comparison: watcher2 reruns with logs in `R5/watcher_logs_m{2,1}/`; PCC identical to the first runs.
  Likely subsystem: harness configuration.
  Investigation performed: rerun.
  Resolution: fixed (first runs kept).
- Earlier ledger entries (review 5 and `OD/README.md` "Anomaly ledger") are unchanged by this round; no new source
  change since chain 12.

## Review 5 Required Work: resolution

- 8192-row step, continuation and watcher: resolved.
  - `R5/run_split.sh`: layer 3, 26000 tokens, 4 traced decode steps, `--trace --audit --sparse-reference`, on
    p150x4 (`STAGE5_MESH=4`), p150x2 (`TT_VISIBLE_DEVICES=2,3`) and p150 (`QDEV=3`). `R5/run_split.out`: all
    non-watcher runs rc=0, `SOURCE_STABLE`.
  - `R5/plans.txt`: split 8320 gives 8192-row steps at 8320 and 16512 and a 2048 masked tail at 24704 (1296 valid).
    8320 and 16512 are not multiples of 256 or 512. Split 8192 is the aligned control.
  - Control policy differs from the default only in `prefill_long_chunk` (8192 -> 0) on every profile (I diffed the
    recorded `optimization` dicts). The long rows carry the selected configs: q64/k256 on p150x4, q128/k512 on
    p150x2 and p150.
  - Recomputed, prefill PCC / decode PCC, long split 8320 vs control vs long split 8192: p150x4 0.998593 / 0.998254,
    0.998593 / 0.998254, 0.998594 / 0.998254; p150x2 0.998586 / 0.998267, 0.998598 / 0.998267, 0.998586 / 0.998267;
    p150 0.998589 / 0.998191, 0.998592 / 0.998191, 0.998589 / 0.998191. Decode minimum and strict minimum are
    identical across arms per profile. All rows: `runtime_audit` true, `repeated_identical` true,
    `output_replicas_identical` true, one recorded router tie at step 0 in every arm.
  - Watcher: `R5/watcher_logs_m{4,2,1}/generated/watcher/watcher.log` have 0 lines matching error, assert, trip,
    overflow or hang; 5, 4 and 3 dumps over 45, 39 and 23 s; `sdpa.cpp` and `sdpa_flash_decode.cpp` in the kernel
    lists. The watcher2 PCC equals the non-watcher long row on every profile.
- `sparse_matmul` no-fill unit tests: resolved.
  - `R5/test_sparse_matmul_indexed.log`: "59 passed in 23.05s", `test_indexed_optional_output` PASSED. That test
    fills the preallocated output with 99.0 and checks every slot against a torch reference, so any slot the writer
    skipped would fail.
  - `R5/test_sparse_matmul.log`: "29 passed, 2 skipped"; skips are "Quasar-only API".
  - Library: `TM/build` links to `build_Release`; `_ttnncpp.so` sha256 b38c43f3... in `R5/source_start.sha256` and
    `R5/source_end.sha256`, equal to the file now.
  - Writer source (`TM/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp`
    lines 113-114 and 316-319): in indexed mode the loop runs `num_active` slots with no sparsity skip, which matches
    the `use_indices` compact-output rule in `sparse_matmul_device_operation.cpp`.

## Verified Claims

- Live tree: all 55 files in `E12/source_manifest_end.json` hash to the manifest values now;
  `E12/source_manifest_start.json` equals the end manifest; `git diff HEAD -- ttnn` sha256 066fd7da... equals
  `E12/ttnn_patch.diff`. HEAD is 5d4c0441. The two ttnn changes are the indexed no-fill and the SDPA profiler zone
  label rename ("Reduce max" -> "Reduce_max", label only, documented in `OD/README.md` and `OD/work_log.md`).
- `S5/evidence_chain12.log`: 255 `END rc=0`, 0 nonzero, `SOURCE_STABLE`, `CHILD_FAILURES 0`, `CHAIN_DONE`.
- `E12/cpu_pytest.xml`: 1213 tests, 0 failures, 0 errors, 0 skipped.
- Headline decode, median of 3 per-run medians of `traced_replay_host_us`: p150x4 full 257.0, linear 228.1 us;
  p150x2 290.5 / 268.1; p150 367.9 / 358.7. Equal to `OD/README.md` and `OD/work_log.md`.
- Context: `AP/doc/context_contract.json` `optimized_multichip_decoder_validation` keeps 262144 tokens with no
  capability reduction and lists the 8192 index-offset table.

## Scope Inspected

- Goal/skill paths: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/05-05-optimized-multichip-decoder.prompt.txt`,
  `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`. I relied
  on reviews 1 to 5 for the full `$optimize`, `$multichip` and `$tt-enable-tracing` comparison; this round changed
  no source, so I did not reread those skills.
- Artifact paths: `OD/stage_review5.md`, `OD/README.md` (continuation, watcher, memory, ledger, code changes,
  reproduction sections), `OD/work_log.md` (chain 12 and review 5 follow-up), `R5/` (all files, including
  `failed_watcher_no_noinline/` and `watcher_logs_m*/`), `E12/` (manifests, ttnn patch, headline rows, CPU pytest
  XML), `S5/evidence_chain12.log`, `S5/evidence_chain12.sh`, `AP/doc/context_contract.json`.
- Code paths: `TM/ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp` (diff),
  `TM/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp` (diff),
  `TM/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp`,
  `TM/tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul_indexed.py` (lines 375-420).
- Commands run: `git log`, `git status`, `git diff`, `ls`, `cat`, `sed`, `grep`, `diff`, `cmp`, `sha256sum`, `find`,
  `date`, and Python over JSON, XML and logs. No device, server, build or device test.

## Residual Risk

- Continuation accuracy inside the second 8192-row step is checked only through its KV contribution (decode and
  tail probes), not through its own output rows.
- Long-context accuracy at ISL 131072 with external chunking is still single-layer, random-input evidence; the
  full-model stage owns end-to-end long-context quality.
- The tie-aware decode gate cannot detect a router bias smaller than one BF16 (16-bit brain float) unit in the last
  place.
- MoE prefill drift has no root cause; B > 1 decode is a per-row loop (Stage 7); the BFP8 routed sum is checked
  for full-model accuracy only in the datatype sweep.
