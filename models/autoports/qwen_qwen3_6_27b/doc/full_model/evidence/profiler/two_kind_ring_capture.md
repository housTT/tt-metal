# Selected two-layer-kind Ring terminal profile

Date: 2026-08-20 EDT

This is the final reduced `tt-perf-report` for the selected full-model path. It
uses real checkpoint layer indices `[0,3]`: layer 0 is linear attention and
layer 3 is full attention. The capture also includes final RMSNorm, the TP4
vocabulary LM head, and canonical common Ring force argmax between
`QWEN36_FULL_MODEL_TOKEN_OUT_START` and
`QWEN36_FULL_MODEL_TOKEN_OUT_END`.

The profiler-runtime pytest passed. The signpost-bounded report contains:

- 195 merged operation rows and 4,361.4615 us summed device time;
- one `SdpaDecodeDeviceOperation` (6.674 us), proving the full-attention layer;
- two `PagedUpdateCacheDeviceOperation` rows and four async AllReduce rows;
- 29 matmuls totaling 1,221.205 us (28.00%);
- selected Ring force argmax: ArgMax 1,418.03 us (32.51%) plus Ring async
  all-gather 883.663 us (20.26%);
- no `TopKDeviceOperation`.

Command:

```bash
PANDAS_FUTURE_INFER_STRING=0 \
TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler \
TT_METAL_RUNTIME_ROOT=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler \
TT_METAL_CACHE=/tmp/qwen36_two_kind_ring_tracy_cache \
PYTHONPATH=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/ttnn:/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/tools:/home/ttuser/dev/qwen-perf/tt-metal \
QWEN36_RUN_TOKEN_OUT_BENCHMARK=1 \
QWEN36_BENCH_LAYER_INDICES=0,3 QWEN36_PROFILE_TOKEN_OUT=1 \
python -m tracy -p -r -v --check-exit-code --dump-device-data-mid-run \
  --tracy-tools-folder=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/build/tools/profiler/bin \
  -o /tmp/qwen36_two_kind_ring_raw \
  -m pytest -q -s models/autoports/qwen_qwen3_6_27b/tests/test_full_model.py \
  -k reduced_token_out_latency_breakdown

tt-perf-report /tmp/qwen36_two_kind_ring_raw/reports/2026_08_20_14_02_38/ops_perf_results_2026_08_20_14_02_38.csv \
  --start-signpost QWEN36_FULL_MODEL_TOKEN_OUT_START \
  --end-signpost QWEN36_FULL_MODEL_TOKEN_OUT_END \
  --csv evidence/profiler/two_kind_ring_token_out_report.csv \
  --summary-file evidence/profiler/two_kind_ring_token_out_summary
```

The 3.3 GiB raw Tracy directory and verbose compiler log remain under `/tmp`
for the current session and are intentionally excluded from repository
artifacts. The compact CSV, summary CSV, and summary PNG are authoritative.
