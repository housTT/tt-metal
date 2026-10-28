# Functional-decoder performance provenance

Warmed batch-1, sequence-1024 measurement of the real-weight layer-0
functional decoder on one Blackhole P300 chip (`TT_VISIBLE_DEVICES=1`,
1x1 mesh). Prefill is run once to compile and warm, then measured between
Tracy signposts. A continuation prefill (100 tokens resumed at the
tile-unaligned start 37 over a 37-token prefix in a second cache) is warmed
once and measured the same way; it exercises the row-major front-pad detour
and the token-by-token cache writes for the leading rows. Decode is captured
as one TTNN trace, replayed twice as warmup, then measured as one blocking
`ttnn.execute_trace` between signposts. `ttnn.ReadDeviceProfiler` drains the
device profiler buffers between phases and outside the host-timing intervals.

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

Tracy wrote `generated/profiler/qwen3_functional/reports/2026_10_27_23_15_33/ops_perf_results_2026_10_27_23_15_33.csv`,
copied locally as `full_attention/ops_perf_results.csv` (1.4 MB, above the
repository's 500 KB limit, so it is not committed; `ops_perf_results.source.txt`
records the source path and the filtered per-window CSVs below are committed). Report generation (tt-perf-report 1.2.9 in the
checkout `python_env`):

```bash
P=models/autoports/qwen_qwen3_8b/doc/functional_decoder/perf/full_attention
for spec in prefill:PERF_PREFILL prefill_continuation:PERF_PREFILL_CONT decode:PERF_DECODE; do
  phase=${spec%%:*}; S="${spec#*:}_layer0_full_attention_seq1024_batch1"
  tt-perf-report $P/ops_perf_results.csv --start-signpost $S --end-signpost ${S}_END \
    --arch p150 --no-color --no-advice --csv $P/$phase.csv > $P/$phase.console.log
  tt-perf-report $P/ops_perf_results.csv --start-signpost $S --end-signpost ${S}_END \
    --arch p150 --no-color --no-advice --no-summary > $P/$phase.txt
done
```

`prefill.csv`, `prefill_continuation.csv` and `decode.csv` are the
signpost-filtered machine-readable slices (every op in the measured phase);
the `.txt` files are the human-readable tables. Device totals below are the
sum of the `Device Time` column of the filtered CSV, which tt-perf-report
reports in microseconds; every row in all three windows is a device op (no
host ops). Host values are from
`../layer0_full_attention_seq1024_batch1_host_timings.json`.

| Phase | Ops | Device time | Host wall |
| --- | ---: | ---: | ---: |
| warmed prefill, 1024 tokens | 21 | 10.956 ms | 11.349 ms |
| warmed continuation prefill, 100 tokens at start 37 | 381 | 2.873 ms | 5.529 ms |
| warmed traced decode, batch 1, position 1024 | 35 | 1.266 ms | 1.335 ms |

The compute configuration is HiFi4 with fp32 accumulation on every op
(correctness-first). The five largest prefill ops are the HiFi4 BF16
matmuls (down 3.21 ms, gate and up 1.94 ms each, QKV 1.61 ms, O 1.08 ms);
decode is DRAM-bound on the same matmuls (0.25 ms each for gate/up/down).
The continuation window is dominated by op count rather than device work:
27 leading rows are written one token at a time (slice, transpose, reshard,
`moreh_full`, two `paged_update_cache` calls each) before the block-aligned
fill, which is the functional baseline for tile-unaligned resumes. These are
functional-stage baselines, not performance claims.
