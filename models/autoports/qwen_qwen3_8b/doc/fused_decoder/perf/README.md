# Fused-decoder performance provenance

Warmed batch-1, sequence-1024 measurements of the real-weight layer-0 decoder
on one Blackhole P300 chip (`TT_VISIBLE_DEVICES=1`, 1x1 mesh), with the same
harness, signposts and report commands as the functional stage
(`../../functional_decoder/perf/README.md`): prefill warmed once then measured
between Tracy signposts; a 100-token continuation at start 37 measured the
same way; decode captured as one trace, replayed twice as warmup, then one
blocking `ttnn.execute_trace` measured.

- `before/`: the functional decoder (`test_functional_decoder.py::test_functional_decoder_perf_profile`,
  `QWEN3_EVIDENCE_SUFFIX=before_fused`), run on 2026 Oct 28 00:36 ET, CSV
  `generated/profiler/qwen3_functional_before_fused/reports/2026_10_28_04_36_51/`.
- `full_attention/`: the fused decoder (`test_fused_decoder.py::test_fused_decoder_perf_profile`),
  run on 2026 Oct 28 13:33 ET, CSV
  `generated/profiler/qwen3_fused/reports/2026_10_28_17_33_11/`.

Workload command (fused; the "before" run swaps the test file and function
name and adds `QWEN3_EVIDENCE_SUFFIX=before_fused`):

```bash
TT_VISIBLE_DEVICES=1 QWEN3_PERF=1 QWEN3_PERF_TRACE_WARMUPS=2 QWEN3_PERF_TRACE_ITERATIONS=1 \
timeout 1200 python -m tracy -r -p -v --check-exit-code \
  -o generated/profiler/qwen3_fused \
  -m pytest -q -s \
  'models/autoports/qwen_qwen3_8b/tests/test_fused_decoder.py::test_fused_decoder_perf_profile[blackhole-batch1-device_params0-mesh_device0]' \
  -p no:cacheprovider
```

Report generation (tt-perf-report in the checkout `python_env`), for `P` in
`perf/before` and `perf/full_attention`:

```bash
for spec in prefill:PERF_PREFILL prefill_continuation:PERF_PREFILL_CONT decode:PERF_DECODE; do
  phase=${spec%%:*}; S="${spec#*:}_layer0_full_attention_seq1024_batch1"
  tt-perf-report $P/ops_perf_results.csv --start-signpost $S --end-signpost ${S}_END \
    --arch p150 --no-color --no-advice --csv $P/$phase.csv > $P/$phase.console.log
  tt-perf-report $P/ops_perf_results.csv --start-signpost $S --end-signpost ${S}_END \
    --arch p150 --no-color --no-advice --no-summary > $P/$phase.txt
done
```

The raw `ops_perf_results.csv` files (1.4 MB) exceed the repository's 500 KB
limit and are not committed; `ops_perf_results.source.txt` records their
source paths and the signpost-filtered per-phase CSV and tables are committed.
Device totals are the sum of the `Device Time` column (microseconds); host
values come from `layer0_full_attention_seq1024_batch1_host_timings.json`
(fused) and `before/layer0_full_attention_seq1024_batch1_host_timings.json`.

| Phase | Before: ops, device, host | After: ops, device, host |
| --- | --- | --- |
| warmed prefill, 1024 tokens | 21, 10,966 us, 11.32 ms | 19, 4,803 us, 5.29 ms |
| warmed continuation prefill, 100 tokens at 37 | 381, 2,870 us, 5.45 ms | 381, 2,870 us, 5.77 ms |
| warmed traced decode, batch 1, position 1024 | 35, 1,265 us, 1.335 ms | 29, 1,253 us, 1.319 ms |

Conclusions from the tables: in prefill the four `minimal_matmul` rows run at
66 to 71 percent of peak FLOPs (vs 26 to 40 percent for the `ttnn.linear`
rows they replace) and the two residual adds are gone; the remaining
non-matmul time is the SiLU-gated multiply (209 us), SDPA (248 us) and the
norms. In decode every op is a device op; the four weight matmuls (948 us at
78 to 84 percent DRAM utilization) and the two single-core RMSNorms (169 us)
are 89 percent of the window and are not fusing targets; the fused graph
removed six small ops and 12 us. The continuation window is unchanged because
its 64-row chunk stays below the 256-row kernel crossover and it is dominated
by the 27 token-by-token leading-row writes. These are HiFi4/fp32-accumulate
measurements, not final performance claims.
