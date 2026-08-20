# Fused decoder work log

## Scope and provenance

Started from functional-decoder commit `2346a6fedb1` (`Record Qwen3.6 functional decoder handoff`). Stage-owned paths are only `tt/fused_decoder.py`, `tests/test_fused_decoder.py`, and `doc/fused_decoder/`. `doc/context_contract.json` is unchanged because dtype, cache layout, batch capacity, and maximum context remain unchanged.

Final code/test hashes are recorded in `source_manifest.sha256`. Those files were frozen before final correctness, profiler, and Watcher collection. Baselines from `doc/functional_decoder/perf/summary.csv` are linear prefill 6,046 us / 177 ops, linear traced decode 2,901 us / 88 ops, full prefill 2,594 us / 51 ops, and full traced decode 2,639 us / 50 ops.

## Hardware discipline

Preflight and postflight command:

```bash
timeout 60 tt-smi -ls --local
```

Four healthy Blackhole p300c boards were visible. Hardware commands were serialized. Tracy and Watcher were run separately. No reset or recovery was needed.

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

- direct `repeat_interleave`: retained because the complete graph was faster despite internal composite layout work;
- host-reordered `[Q,K,V,gate]` plus dedicated prefill/decode head creation: retained; final full prefill/decode improve to 2,362.104/2,334.152 us;
- tile-aligning the 48-wide beta/a packed tails: native-context correct and retained at 5,605.562 us / 159 prefill ops and 2,728.252 us / 71 decode ops;
- per-lane decode RoPE through the sequence axis: correct at decode PCC 0.997824526 and 39 ops, but 2,346.213 us, 6.533 us slower than the primitive candidate on the identical base; rejected;
- sharded SDPA into `nlp_concat_heads_decode`: hardware fatal `Sharded output not supported for GQA`; rejected;
- grouped causal `conv1d`: the required 10,240 BF16 groups exceed the repository test's explicit >5,120-group OOM boundary; rejected without risking device OOM;
- dedicated linear core RMSNorm and FP32-accumulating RMSNorm: native-context nonfinite; rejected;
- packed MLP: exact PCC pass, but slower in all four phases; rejected;
- dedicated prefill concat heads: exact PCC pass, but 9.692 us slower; rejected;
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

Final stage review: `clean-pass`; no required work or hard-check gaps.

Stage-owned implementation and evidence commit: `5d436142493e9a1085890b483cb86307877d9133`.

The successor documentation-only commit records this SHA; its SHA is reported in the final handoff.
