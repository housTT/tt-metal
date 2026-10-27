# Functional-decoder performance provenance

Warmed batch-1, sequence-1024 measurement of the real-weight layer-0
functional decoder on one Blackhole P300 chip (`TT_VISIBLE_DEVICES=1`,
1x1 mesh). Prefill is run once to compile and warm, then measured between
Tracy signposts. Decode is captured as one TTNN trace, replayed twice as
warmup, then measured as one blocking `ttnn.execute_trace` between
signposts. `ttnn.ReadDeviceProfiler` drains the device profiler buffers
between phases and outside the host-timing intervals.

Exact workload command (run from the checkout root with the stage
environment):

```bash
TT_VISIBLE_DEVICES=1 QWEN3_PERF=1 QWEN3_PERF_TRACE_WARMUPS=2 QWEN3_PERF_TRACE_ITERATIONS=1 \
timeout 1800 python -m tracy -r -p -v --check-exit-code \
  -o generated/profiler/qwen3_functional \
  -m pytest -q -s \
  'models/autoports/qwen_qwen3_8b/tests/test_functional_decoder.py::test_functional_decoder_perf_profile[blackhole-batch1-device_params0-mesh_device0]' \
  -p no:cacheprovider
```

Tracy wrote `generated/profiler/qwen3_functional/reports/2026_10_27_21_14_29/ops_perf_results_2026_10_27_21_14_29.csv`,
copied here as `full_attention/ops_perf_results.csv` (`ops_perf_results.source.txt`
records the source path). Report generation (tt-perf-report 1.2.9 in the
checkout `python_env`):

```bash
P=models/autoports/qwen_qwen3_8b/doc/functional_decoder/perf/full_attention
for phase in prefill decode; do
  S="PERF_$(echo $phase | tr a-z A-Z)_layer0_full_attention_seq1024_batch1"
  tt-perf-report $P/ops_perf_results.csv --start-signpost $S --end-signpost ${S}_END \
    --arch p150 --no-color --no-advice --csv $P/$phase.csv > $P/$phase.console.log
  tt-perf-report $P/ops_perf_results.csv --start-signpost $S --end-signpost ${S}_END \
    --arch p150 --no-color --no-advice --no-summary > $P/$phase.txt
done
```

`prefill.csv` and `decode.csv` are the signpost-filtered machine-readable
slices (every op in the measured phase); `prefill.txt` and `decode.txt` are
the human-readable tables. Device totals below are the sum of the
`Device Time` column of the filtered CSV, which tt-perf-report reports in
microseconds. Host values are from
`../layer0_full_attention_seq1024_batch1_host_timings.json`.

| Phase | Ops | Device time | Host wall |
| --- | ---: | ---: | ---: |
| warmed prefill, 1024 tokens | 21 | 10.965 ms | 11.391 ms |
| warmed traced decode, batch 1, position 1024 | 35 | 1.269 ms | 1.364 ms |

The compute configuration is HiFi4 with fp32 accumulation on every op
(correctness-first). The five largest prefill ops are the HiFi4 BF16
matmuls (down 3.21 ms, gate and up 1.94 ms each, QKV 1.61 ms, O 1.08 ms);
decode is DRAM-bound on the same matmuls (0.25 ms each for gate/up/down).
These are functional-stage baselines, not performance claims.
