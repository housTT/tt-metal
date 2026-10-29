# Optimized-decoder performance provenance

Warmed layer-0 measurements of the real-weight decoder on one Blackhole P300
chip (`TT_VISIBLE_DEVICES=0`, 1x1 mesh, p150 mesh graph descriptor), with the
same harness, signposts and report commands as stages 1 and 2
(`../../functional_decoder/perf/README.md`): prefill of 1024 tokens warmed
once then measured between Tracy signposts; a 100-token continuation at
start 37 measured the same way; decode captured as one trace, replayed twice
as warmup, then one blocking `ttnn.execute_trace` measured. Batch 1 and
batch 32 are separate processes (the prefill windows do not depend on the
batch).

- `before/`: the stage-2 fused decoder with its BF16 weights and BF16 KV
  cache, run through this module's harness
  (`test_optimized_decoder.py::test_fused_baseline_perf_profile`,
  `QWEN3_EVIDENCE_SUFFIX=fused_before`) on 2026 Oct 28 15:56 ET; CSVs
  `generated/profiler/qwen3_opt_before/reports/2026_10_28_18_56_58/` (batch
  1) and `2026_10_28_18_57_13/` (batch 32).
- `full_attention/`: the committed optimized decoder (commit `18aac30c2f`,
  BFP8/BFP4 weights, BF16 KV cache, interleaved prefill copy for chunks up to
  256 rows), `test_optimized_decoder_perf_profile`, run on 2026 Oct 29 10:27
  ET; CSVs `generated/profiler/qwen3_opt_after/reports/2026_10_29_14_27_15/`
  (batch 1) and `2026_10_29_14_27_31/` (batch 32).
- `short_prompts/`: first prefill of 64, 128 and 256 tokens from position 0
  (`QWEN3_PERF_SEQ_LEN`), fused baseline (`before_*`, 09:55 ET) and
  optimized (`after_*`, 10:28 ET), batch 1; the source CSV paths are in its
  `ops_perf_results.source.txt`.

Workload command (optimized, batch 1; batch 32 replaces `batch1`; the
"before" run uses `test_fused_baseline_perf_profile` and
`QWEN3_EVIDENCE_SUFFIX=fused_before`), from the checkout root with the stage
environment (`TT_METAL_HOME`, `PYTHONPATH`, `TT_MESH_GRAPH_DESC_PATH` for the
p150 profile, `TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}'`):

```bash
TT_VISIBLE_DEVICES=0 QWEN3_PERF=1 QWEN3_PERF_TRACE_WARMUPS=2 QWEN3_PERF_TRACE_ITERATIONS=1 \
timeout 1800 python -m tracy -r -p -v --check-exit-code \
  -o generated/profiler/qwen3_opt_after \
  -m pytest -q -s \
  'models/autoports/qwen_qwen3_8b/tests/test_optimized_decoder.py::test_optimized_decoder_perf_profile[blackhole-batch1-device_params0-mesh_device0]' \
  -p no:cacheprovider
```

Report generation (tt-perf-report 1.4.1 from the checkout `python_env`; the
package has no console script on this host, so it runs as a module), for
`P` in `perf/before` and `perf/full_attention` and `b` in `1` and `32`:

```bash
for spec in prefill:PERF_PREFILL prefill_continuation:PERF_PREFILL_CONT decode:PERF_DECODE; do
  phase=${spec%%:*}; S="${spec#*:}_layer0_full_attention_seq1024_batch${b}"
  python -m tt_perf_report.perf_report <ops_perf_results.csv> --start-signpost $S --end-signpost ${S}_END \
    --arch p150 --no-color --csv $P/${phase}_batch${b}.csv > $P/${phase}_batch${b}.console.log
  python -m tt_perf_report.perf_report <ops_perf_results.csv> --start-signpost $S --end-signpost ${S}_END \
    --arch p150 --no-color --no-summary > $P/${phase}_batch${b}.txt
done
```

Advice stays enabled in every table (`*.txt` and `*.console.log`). The raw
`ops_perf_results.csv` files exceed the repository's 500 KB limit and are
not committed; `ops_perf_results.source.txt` records their paths. Device
totals are the sum of the `Device Time` column (microseconds); host values
come from `../layer0_full_attention_seq1024_batch{1,32}_host_timings*.json`.

| Window | Before: ops, device, host | After: ops, device, host |
| --- | --- | --- |
| warmed prefill, 1024 tokens | 19, 4,814 us, 5.19 ms | 21, 2,996 us, 3.25 ms |
| warmed first prefill, 64 / 128 / 256 tokens | 21, 1,359 us / 21, 1,587 us / 19, 2,082 us | 21, 1,062 us / 21, 1,310 us / 21, 2,105 us |
| warmed continuation prefill, 100 tokens at 37 (128-row chunk, 27 lead rows) | 381, 2,864 us, 5.64 ms | 122, 1,766 us, 3.47 to 6.46 ms (see note) |
| warmed traced decode, batch 1, position 1024 | 29, 1,257 us, 1.321 ms | 33, 543 us, 0.637 ms |
| warmed traced decode, batch 32, positions 1024 | 30, 1,818 us, 1.886 ms | 32, 1,096 us, 1.166 ms |

Note on host times: the final optimized profiles ran on chip 0 while the
test battery ran on chip 1 of the same P300 card; the single-sample host
wall values of the eager prefill windows carry that contention (the
batch-1 continuation sample is 6.46 ms with a 4.2 ms op-to-op gap, the
batch-32 process measured the same window at 3.47 ms). Device times are
unaffected. Decode replay host times are consistent across runs (0.60 to
0.64 ms at batch 1).

Conclusions from the tables: decode batch 1 is 2.31x faster on device
(2.07x host); the five weight matmuls take 393 us at 404 to 458 GB/s (79 to
89 percent of the 512 GB/s peak; the BFP4 up row is 368 GB/s and marked
`BOTH`), the two sharded RMSNorms 15 us (169 us before), the head split 17 us
from L1 interleaved (33 us before), SDPA 44 us (46 before), the SiLU-mul 23
us, and every other row is 8 us or less. Batch 32 is 1.66x faster; SDPA
(1,024-token contexts for 32 users) is 568 us, 52 percent of that window.
Prefill at 1024 tokens is 1.61x faster: the four `minimal_matmul` rows run at
LoFi on the reduced-precision weights, and the two residual adds returned
(57 us each) because the addcmul-fused kernel needs equal tile sizes. Short
prompts of 64 and 128 tokens are 1.28x and 1.21x faster on the interleaved
weight copy (`ttnn.linear`); 256 tokens ties the stage-2 kernel (2,105
versus 2,082 us). The continuation window has 122 ops instead of 381 (3 ops
per lead row instead of 13) and 1,766 us of device time instead of 2,864.
