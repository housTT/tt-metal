# Focused full-model profiler evidence

Date: 2026-08-20 EDT

Both captures use runtime commit
`9b415f82002af5d9040eca389d703690e405d91f`, four Blackhole P300c devices,
physical `FABRIC_1D_RING`, real checkpoint layer 0 (linear attention) and layer
3 (full attention), final norm, TP LM head, and the generator terminal path.
Safe Watcher is not enabled in either profiler process.

## Token-out

The refreshed signpost-bounded report contains 228 merged rows and 3,380.81 us summed
device time. It includes:

- local reductions: 839.87 us, 24.84%;
- ArgMax: 467.58 us, 13.83%;
- compact candidate all-broadcast: 17.84 us, 0.53%;
- 29 matmuls: 1,222.28 us, 36.16%;
- four persistent async all-reduces: 66.72 us;
- two paged-cache updates: 7.32 us;
- one SDPA decode: 6.52 us;
- no `TopKDeviceOperation`, no all-gather, and no full-vocabulary collective.

The advice-enabled report models 141 GB/s overall DRAM traffic, 27.5% of the
Blackhole report roofline. The reduced capture is intentionally structural; the
uninstrumented 64-layer split trace is the authoritative 43.488361 ms /
22.994658 t/s/u measurement.

The 3,380.81 us figure is a `tt-perf-report` merged-row sum, not a serial
end-to-end latency: replicated device rows and mesh-level collective rows have
different aggregation semantics and some work overlaps. Percentages are used
only for operation/topology attribution. Wall-clock trace timing above is the
performance authority.

Raw source:

- `/tmp/qwen36_optimized_full_token_autofix/reports/2026_08_20_16_38_36/ops_perf_results_2026_08_20_16_38_36.csv`
- size 11,840,775 bytes
- SHA256 `378e69c8b06f4588e7862d08e9dcb18f512a19ce85206668789b8874a6586d68`

Compact artifacts: `token_out_report.csv` (SHA256
`c38a53ca46208f6decce6a5bbb8b5123754dd43d0840ec482421cb84a94b18c8`),
`token_out_summary.csv`, and `token_out_summary.png`.

## Prefill

The signpost-bounded report contains 532 operation rows plus its header and
5,746 us summed device time. Dominant groups are:

- 67 matmuls: 2,393.29 us, 41.65%;
- 12 DRAM-input norms: 543.03 us, 9.45%;
- six reduce-scatters plus six all-gathers: 408.72 us, 7.11%;
- untilize/unpadding: 365.71 us, 6.36%;
- reshape views: 309.79 us, 5.39%;
- both paged cache fills: 2.47 us.

The advice-enabled report models 102 GB/s overall DRAM traffic, 20.0% of the
Blackhole report roofline. The profile makes the expected collective, matmul,
norm, cache, embedding, final-norm/LM-head, and layout costs visible. The
uninstrumented final full64 prompt-128 TTFT is the authoritative 655.159903 ms;
its same-process inherited-policy control is 679.058899 ms.

Raw source:

- `/tmp/qwen36_optimized_full_prefill_final.cGdhMP/reports/2026_08_20_15_51_09/ops_perf_results_2026_08_20_15_51_09.csv`
- size 10,201,226 bytes
- SHA256 `50bb99988278c0e4023d76fcfe23c996358ba57988d5a66b004182abe192f1ec`

Compact artifacts: `prefill_report.csv`, `prefill_summary.csv`, and
`prefill_summary.png`.

## Commands

Both captures used this common environment:

```bash
PANDAS_FUTURE_INFER_STRING=0 \
TT_METAL_HOME=/home/ttuser/dev/tt-metal \
TT_METAL_RUNTIME_ROOT=/home/ttuser/dev/tt-metal \
TT_METAL_CACHE=/tmp/qwen36_optimized_full_token_cache \
PYTHONPATH=/home/ttuser/dev/tt-metal/ttnn:/home/ttuser/dev/tt-metal/tools:/home/ttuser/dev/qwen-perf/tt-metal \
QWEN36_RUN_TOKEN_OUT_BENCHMARK=1 QWEN36_BENCH_LAYER_INDICES=0,3 \
python -m tracy -p -r -v --check-exit-code --dump-device-data-mid-run \
  --tracy-tools-folder=/home/ttuser/dev/tt-metal/build/tools/profiler/bin \
  -o RAW_DIR -m pytest -q -s \
  models/autoports/qwen_qwen3_6_27b/tests/test_full_model.py::test_reduced_token_out_latency_breakdown
```

Token-out adds `QWEN36_PROFILE_TOKEN_OUT=1`; prefill adds
`QWEN36_PROFILE_PREFILL=1`. Reports were produced with:

```bash
tt-perf-report RAW_CSV --start-signpost SIGNPOST_START \
  --end-signpost SIGNPOST_END --no-color --csv REPORT.csv \
  --summary-file SUMMARY
```

Advice is enabled by default; `--no-advice` was not used.

## Advice triage

The report's fidelity suggestions are not adopted: switching selected LoFi
projection/MLP groups to HiFi2/HiFi4 would violate the inherited measured
dtype/fidelity policy, and the accuracy gates already pass top-5/top-100 at the
required threshold. Suggestions to put small recurrent matmul input 0 in L1 do
not account for the persistent DRAM recurrent-state contract and would add a
per-token transfer or a much larger persistent L1 reservation. The largest
projection matmuls already use inherited width-sharded L1 activations and
DRAM-sharded weights; their measured program configs and rejection ledger live
in the optimized-decoder and optimized-multichip-decoder stages. "No output
subblock size found" is a report metadata limitation for those explicit
configs, not evidence that the runtime used a generic fallback. No advice item
identifies an untested terminal sampler replacement or a remaining gap above
the 10-15% stack-bound threshold.

## Profiler repair record

The first two prefill Tracy captures executed the test successfully but failed
post-processing with a missing device-op row. Their logs showed profiler DRAM
buffers filling during the compile-heavy first warmup. The harness now calls
`ttnn.ReadDeviceProfiler` immediately after that compile warmup, again after an
ordinary warmup, and around the isolated signposted measurement. The final
capture has a complete host/device op ledger and generates both CSV tables.
