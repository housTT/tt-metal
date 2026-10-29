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
- `full_attention/`: the committed optimized decoder (commit `20df38fd5e`,
  BFP8/BFP4 weights, BFP8 KV cache), `test_optimized_decoder_perf_profile`,
  run on 2026 Oct 29 08:41 ET; CSVs
  `generated/profiler/qwen3_opt_after/reports/2026_10_29_12_41_04/` (batch 1)
  and `2026_10_29_12_41_18/` (batch 32).

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
| warmed prefill, 1024 tokens | 19, 4,814 us, 5.19 ms | 23, 2,990 us, 3.23 ms |
| warmed continuation prefill, 100 tokens at 37 (128-row chunk, 27 lead rows) | 381, 2,864 us, 5.64 ms | 124, 2,282 us, 3.85 ms |
| warmed traced decode, batch 1, position 1024 | 29, 1,257 us, 1.321 ms | 33, 538 us, 0.646 ms |
| warmed traced decode, batch 32, positions 1024 | 30, 1,818 us, 1.886 ms | 32, 1,059 us, 1.126 ms |

Conclusions from the tables: decode batch 1 is 2.34x faster on device
(2.04x host); the five weight matmuls take 391 us at 404 to 458 GB/s (79 to
89 percent of the 512 GB/s peak; the BFP4 up row is 368 GB/s and marked
`BOTH`), the two sharded RMSNorms 15 us (169 us before), the head split 17 us
from L1 interleaved (33 us before), SDPA 44 us (46 before), the SiLU-mul 21
us, and every other row is 8 us or less. Batch 32 is 1.72x faster; SDPA
(1,024-token contexts for 32 users) is 54 percent of that window. Prefill at
1024 tokens is 1.61x faster: the four `minimal_matmul` rows run at LoFi on
the reduced-precision weights (gate 460, up 436, down 458, QKV 269 us versus
953, 953, 969 and 515 us at HiFi4 BF16), and the two residual adds returned
(57 us each) because the addcmul-fused kernel needs equal tile sizes. The
continuation window has 124 ops instead of 381 (3 ops per lead row instead
of 13) and 1.3 ms of host op-to-op gap instead of 2.6 ms; its device time is
dominated by the four 128-row `minimal_matmul` rows on the DRAM-sharded
weights (331 to 390 us each, see `../work_log.md` 3.10 for the rejected
interleaved-copy alternative).
