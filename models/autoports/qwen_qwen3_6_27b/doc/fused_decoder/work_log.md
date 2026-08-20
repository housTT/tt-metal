# Fused decoder work log

## Scope and provenance

Started from functional-decoder commit `2346a6fedb1` (`Record Qwen3.6 functional decoder handoff`). Stage-owned paths are only `tt/fused_decoder.py`, `tests/test_fused_decoder.py`, and `doc/fused_decoder/`. `doc/context_contract.json` is unchanged because dtype, cache layout, batch capacity, and maximum context remain unchanged.

Final code/test hashes are recorded in `source_manifest.sha256`. Those files were frozen before final correctness, profiler, and Watcher collection. Baselines from `doc/functional_decoder/perf/summary.csv` are linear prefill 6,046 us / 177 ops, linear traced decode 2,901 us / 88 ops, full prefill 2,594 us / 51 ops, and full traced decode 2,639 us / 50 ops.

## Hardware discipline

Preflight and postflight command:

```bash
timeout 60 tt-smi -ls --local
```

Four healthy Blackhole p300c chips were visible. Hardware commands were serialized. Tracy and Watcher were run separately. No reset or recovery was needed during the original frozen-final correctness, profiler, and Watcher collection described above.

The later split-conv remediation had a separate recovery event. Retained triage proves that the `config_tensors_in_dram=True` command stalled device 3 at zero ARC heartbeat, but the original candidate PID, termination command, list/reset/list output, lock decision, second-reset decision, and mesh-smoke terminal transcript were not retained. They cannot be reconstructed exactly and are not claimed.

Before the review-closing A1/B/A2, a fresh bounded ledger was captured in `candidates/split_conv1d/recovery/`. Process and lock inspection found no Qwen/pytest/Tracy-profiler hardware job and no owner on the selected chip-in-use locks. An unrelated long-running Laguna vLLM PID 512455 was neither killed nor reset; its separate P300 board was excluded. BDFs `0000:01:00.0` and `0000:02:00.0`, the complete free P300 board ID `000004613192404c`, were targeted with `timeout 180 tt-smi -r ...`; reset exited 0. Bounded lists before/after exited 0 with all four chips visible, so a second reset was unnecessary. No lock was cleared. An installed-runtime 1x1 mesh with both `trace_region_size=0` and `l1_small_size=16384` opened and closed on that isolated visibility pair and printed `MESH_SMOKE_OK 1`. Every remediation correctness/profiler run used the same `TT_VISIBLE_DEVICES=0000:01:00.0,0000:02:00.0` restriction.

## Final correctness command

```bash
cd /tmp
env PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal \
  pytest -q \
  /home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py \
  -k 'not perf' \
  --junitxml=/home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/correctness/final/junit.xml \
  -s
```

Result: 12 passed, 2 profiler tests deselected, in 102.91 seconds. The retained log and JUnit XML are in `correctness/final/`.

Coverage includes:

- real-weight full and linear layer kinds;
- 33- and 65-token non-aligned prefill/decode;
- forced 257-logical/384-physical chunked prefill;
- paged cache fill/update and trace replay;
- repeated-run determinism;
- 32 active users for both layer kinds;
- linear 262,144-token prefill plus traced decode;
- full 32,769- and 262,144-token prefill;
- full traced decode at position 262,143.

## Native-context autofix

The first review found that native linear context had not been executed. Adding that gate exposed nonfinite output at 262,144 tokens. Autofix isolated the failure by restoring one candidate family at a time:

1. restoring functional L2 normalization alone still failed;
2. restoring separate input projections still failed;
3. restoring material DeltaNet transposes still failed;
4. restoring reshape/concat head replication still failed;
5. restoring the unfused MLP still failed;
6. restoring only the linear core normalization's spelled mean-square/add/rsqrt/multiply passed.

The same dedicated RMSNorm failed native decode after the recurrent prefix. Enabling `fp32_dest_acc_en=True`, HiFi4, exact math, and L1 accumulation also failed with nonfinite output near BF16's upper range. The final graph therefore retains the stable core normalization in prefill and decode while keeping the adjacent SiLU input-activation multiply fusion. Both native linear gates pass in the complete final suite; representative short, trace, chunked, and batch-32 paths pass separately under Watcher.

## Profiler commands and final results

Each layer/phase used the following template with the matching test id, phase, and artifact directory:

```bash
cd /tmp
env QWEN36_PERF_PHASE=PHASE QWEN36_DECODE_REPLAYS=1 \
  PANDAS_FUTURE_INFER_STRING=0 \
  TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler \
  TT_METAL_RUNTIME_ROOT=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler \
  TT_METAL_CACHE=/tmp/qwen36_fused_tracy_cache \
  PYTHONPATH=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/ttnn:/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/tools:/home/ttuser/dev/qwen-perf/tt-metal \
  python -m tracy -p -r -v --check-exit-code --no-runtime-analysis \
  --tracy-tools-folder=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/build/tools/profiler/bin \
  -o ARTIFACT_DIR -m pytest -q TEST_NODE -s
```

Final signpost totals:

| Layer | Phase | Device us | Ops |
|---|---|---:|---:|
| Linear | prefill | 5,605.562 | 159 |
| Linear | traced decode | 2,728.252 | 71 |
| Full | prefill | 2,362.104 | 29 |
| Full | traced decode | 2,334.152 | 45 |

`tt-perf-report` was run for every raw CSV:

```bash
tt-perf-report RAW_OPS_CSV \
  --start-signpost START --end-signpost END \
  --arch blackhole --no-color --no-advice \
  --csv REPORT.csv --summary-file REPORT_summary
```

Raw CSVs are in `perf/raw_*/reports/`. Filtered operation tables, summary CSVs, and plots are in `perf/reports/`. `perf/summary.csv` contains the baseline comparison. The profiler's legacy CSV omits `DEVICE ARCH`, so `tt-perf-report` emits a roofline warning despite `--arch blackhole`; operation rows and device duration are valid, and the hardware identity is recorded by `tt-smi` and runtime logs.

## Candidate and pattern decisions

Exact candidate metrics, source hashes, PCC, disposition, and primary evidence paths are in `candidates/index.csv` and the per-candidate bundles. Important late candidates were:

- direct `repeat_interleave`: retained after frozen-base A/B/A showed reshape/concat 256.306 us slower in prefill and 15.605 us slower in traced decode;
- host-reordered `[Q,K,V,gate]` plus dedicated prefill/decode head creation: retained; final full prefill/decode improve to 2,362.104/2,334.152 us;
- tile-aligning the 48-wide beta/a packed tails: native-context correct and retained at 5,605.562 us / 159 prefill ops and 2,728.252 us / 71 decode ops;
- per-lane decode RoPE through the sequence axis: correct at decode PCC 0.997824526 and 39 ops, but frozen-base A/B/A put it 8.166 us slower than the primitive median control; rejected;
- sharded SDPA into `nlp_concat_heads_decode`: hardware fatal `Sharded output not supported for GQA`; rejected;
- split grouped causal `conv1d`: adapted through split-4/split-8 and flattened batch-32 experiments; the correct standalone-SiLU form passed native context. Exact 16-KiB A1/B/A2 made it 2.405% slower in prefill and 9.432% slower in traced decode; rejected;
- dedicated linear core RMSNorm and FP32-accumulating RMSNorm: native-context nonfinite; rejected;
- packed MLP: exact PCC pass, but slower in all four phases; rejected;
- dedicated prefill concat heads: exact PCC pass, but frozen-base A/B/A put it 8.406 us slower than the primitive median control; rejected;
- packed QKV/gate, dedicated QKV head creation, packed tile-aligned linear inputs, L2 RMSNorm, transpose attributes, and binary input activations: retained.

The README contains the complete graph-fusing pattern matrix and final data-movement inventory.

## Watcher

```bash
cd /tmp
env TT_METAL_WATCHER=10 TT_METAL_LOGGER_LEVEL=Info \
  TT_METAL_LOGS_PATH=/home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/watcher/final \
  PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal \
  TT_METAL_RUNTIME_ROOT=/home/ttuser/.local/lib/model-bringup/tt-metal \
  TT_METAL_CACHE=/tmp/qwen36_fused_watcher_cache \
  pytest -q \
  /home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py \
  -k 'full_attention_non_aligned_prefill_decode or full_attention_paged_decode_trace or full_attention_forced_chunked_prefill or linear_attention_non_aligned_prefill_decode or real_weight_batch32_prefill_decode' \
  --junitxml=/home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/watcher/final/junit.xml \
  -s
```

Result: six representative final fused-path tests passed, eight tests deselected, in 22.31 seconds. This scan returned no matches:

```bash
rg -ni 'watcher.*(error|assert|hang|timeout)|noc.*(error|timeout)|kernel.*assert|device.*hang|TEST FAILED' \
  doc/fused_decoder/watcher/final/generated/watcher/watcher.log
```

## Artifacts

- `source_manifest.sha256`: source/test provenance.
- `correctness/final/{pytest.log,junit.xml}`: final complete suite.
- `perf/summary.csv`: before/after latency and operation counts.
- `perf/candidates.csv`: selected/rejected candidates and evidence.
- `candidates/`: primary candidate source snapshots, exact commands/manifests, PCC/failure logs, and compact profiler reports.
- `perf/raw_*/reports/ops_perf_results.csv`: compact raw Tracy operation data.
- `perf/reports/*`: signpost-filtered `tt-perf-report` tables, summaries, and plots.
- `watcher/final/{pytest.log,junit.xml}` and `watcher/final/generated/watcher/watcher.log`: separate Watcher evidence.

Large intermediate Tracy databases and build logs were not retained; they are reproducible with the commands above.

## Review and commit

Initial stage review: `more-work-needed`; all findings above were addressed.

Fresh completion review: `more-work-needed`; its findings and anomaly ledger are retained in `stage_review.md`. A second review of the first remediation pass also returned `more-work-needed`; its exact provenance findings are retained in `stage_review_final.md`. AutoFix then produced the common-fixture 16-KiB split-conv A/B/A evidence and fresh hardware-recovery ledger described below.

Final independent rereview: `clean-pass` with no required work. The exact verdict, controlled anomalies, hard-check gaps, scope, and residual risk are retained in `stage_review_clean.md`.

Stage-owned implementation and evidence commit: `5d436142493e9a1085890b483cb86307877d9133`.

Initial documentation-only handoff commit: `48ae700db537b250ee349fe5856ffe2d13f96627`.

The post-review remediation checkpoint and successor SHA-record commit are reported in the final handoff; neither was pushed.

## AutoFix: split-channel depthwise conv1d

The completion review's grouped-convolution finding was tested rather than inferred. Split-4 BF16 HiFi4 with fused SiLU is valid for isolated B=1/L=67 prefill (Torch PCC 0.999771533; current TTNN-expression PCC 0.999769934), and direct DRAM output exactly matches explicit interleaving. Physical B=32/L=4 fails for split-4 and split-8 with a reader-index CB contract assertion; flattening the 32 independent states into physical B=1/L=128 and selecting every fourth output is exact, trace-safe, and reaches PCC 0.999760309 with determinism 1.0.

Integration exposed two separate constraints. The contract-preserving `config_tensors_in_dram=True` route stalled device 3; bounded triage recorded zero ARC heartbeat and all NoCs hung. Exact historical recovery process/status details were not retained; the fresh review-remediation safety ledger is described in Hardware discipline and retained under `candidates/split_conv1d/recovery/`. With the repository-standard 16 KiB L1_SMALL reservation, fused conv1d SiLU gave only PCC 0.957531573, and HiFi4 exact math with FP32 destination/L1 accumulation gave 0.956417713. A final control moved SiLU outside `Conv1dConfig`; that exact split-4 candidate passed the real 65-token test at PCC 0.997846637, with the remaining printed checks 0.999412803, 0.999644046, and 1.0. It also passed full 262,144-token prefill and native-prefix traced decode.

The first 5,737.134-us prefill and 2,983.133-us decode observations are now labeled historical exploration because the temporary 16-KiB profiler fixture was not retained and their quoted controls were zero-L1_SMALL final runs. Review remediation reconstructed the actual candidate implementation SHA256 `f69011093d5d28c1c3a8ca102aebf156cda1024a4a044eaef3f1661ae236a38d`; `dbe94bd3...` is only the retained patch SHA. A1, B, and A2 all used the identical test SHA256 `ea3c400790428be1f5c888103aa621914607274b741a10c381665fac356d9645`, whose profiler and correctness nodes both specify `trace_region_size=0` and `l1_small_size=16384`. The profiler runtime, isolated board, fixed cache, replay count, node, and signposts were unchanged across all six runs.

Like-for-like results are prefill 5,505.526 / 5,635.236 / 5,500.267 us (159/159/159 ops), making B 132.3395 us / 2.405% slower than the 5,502.8965-us median base; traced decode 2,625.831 / 2,873.108 / 2,625.095 us (71/94/71 ops), making B 247.645 us / 9.432% slower with 23 more ops. Because both phases lose, Watcher was not warranted. The implementation and tests were then restored byte-for-byte to `f0778a63a47a844621cffa3a251345333e563105cabf708e5abe0af0dbce3a14` and `05c38b764320bb682248df78d00ff950a1602b8aeee9b2e40cb25fd690a41dad`. Exact snapshots, fixture patch, source identities, correctness log, compact raw CSVs, `tt-perf-report` outputs, commands, arithmetic, and evidence hashes are under `candidates/split_conv1d/l1_16k_ab/`.

## AutoFix: frozen-base missing controls

Three one-change candidates were reconstructed on the same frozen source SHA `f0778a63a47a844621cffa3a251345333e563105cabf708e5abe0af0dbce3a14`. Every profile used `/home/ttuser/.local/lib/model-bringup/tt-metal-profiler`, `QWEN36_DECODE_REPLAYS=1`, fixed cache `/tmp/qwen36_missing_controls_tracy_cache` without clearing between A1/B/A2, the existing phase signposts, and the profiler/`tt-perf-report` commands above. Exact commands, source identities, compact raw CSVs, filtered reports, and summaries are in the three `_ab` candidate bundles.

- Functional reshape/concat Q/K repetition passed the real linear 65-token test at PCC 0.998477649 / 0.999461964 / 0.999649418 with trace determinism 1.0. Prefill A1/B/A2 was 5,604.755 / 5,863.601 / 5,609.836 us (159/167/159 ops); B is 256.306 us, 4.571% slower than the 5,607.296 us median base. Decode was 2,729.836 / 2,742.536 / 2,724.027 us (71/77/71 ops); B is 15.605 us, 0.572% slower than the 2,726.932 us median base. Direct `repeat_interleave` remains selected.
- Dedicated full-prefill `concatenate_heads` passed at PCC 0.998800717 / 0.997618106. A1/B/A2 was 2,358.497 / 2,365.869 / 2,356.430 us (29/28/29 ops); B is 8.406 us, 0.357% slower than the 2,357.464 us median base. Primitive permute/reshape remains selected.
- Lane-axis dedicated decode RoPE passed non-aligned and paged-trace tests at PCC 0.998800717 / 0.997824526 / 0.998387110 with determinism 1.0. A1/B/A2 was 2,339.681 / 2,346.163 / 2,336.314 us (45/39/45 ops); B is 8.166 us, 0.349% slower than the 2,337.998 us median base. Primitive partial decode RoPE remains selected.

No candidate won, so final correctness/performance/Watcher recollection was unnecessary. Source and tests were restored byte-for-byte to SHA256 `f0778a63a47a844621cffa3a251345333e563105cabf708e5abe0af0dbce3a14` and `05c38b764320bb682248df78d00ff950a1602b8aeee9b2e40cb25fd690a41dad`.
