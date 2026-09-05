# Fused-decoder performance provenance

These reports measure `FusedDecoder` with real layer weights on one Blackhole
P300 chip (1x1 mesh, device id 1), batch 1, and sequence length 1,024. Prefill
is warmed once. Decode is captured with TTNN trace execution, warmed once, and
measured with one blocking replay in Tracy. The matching unprofiled comparison
uses 20 additional warmups and 200 measured trace replays.

Final hashes:

- `fused_decoder.py`: `a9f3d0b776674ecad286500fbd86a5d2cf6bb2fbc1d606312724d484c8aea4ea`
- `test_fused_decoder.py`: `fffff4af877208551844a2bd935c86dc4573a54c491c53de1716969db0aeb364`

## Tracy captures

Sliding attention:

```bash
source python_env/bin/activate && \
GEMMA4_FUSED_DECODER_PERF=1 \
GEMMA4_FUNCTIONAL_DECODER_TRACE_WARMUPS=1 \
GEMMA4_FUNCTIONAL_DECODER_TRACE_ITERATIONS=1 \
GEMMA4_FUSED_EXACT_COMMAND='definitive Tracy sliding command; see doc/fused_decoder/perf/README.md' \
timeout 1800 \
python -m tracy -r -p -v --check-exit-code \
  -o generated/profiler/gemma4_fused_sliding_definitive \
  -m pytest -q -s \
  'models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::test_fused_perf_profile[blackhole-sliding_attention_1024-device_params0-mesh_device0]'
```

Full attention:

```bash
source python_env/bin/activate && \
GEMMA4_FUSED_DECODER_PERF=1 \
GEMMA4_FUNCTIONAL_DECODER_TRACE_WARMUPS=1 \
GEMMA4_FUNCTIONAL_DECODER_TRACE_ITERATIONS=1 \
GEMMA4_FUSED_EXACT_COMMAND='definitive Tracy full command; see doc/fused_decoder/perf/README.md' \
timeout 1800 \
python -m tracy -r -p -v --check-exit-code \
  -o generated/profiler/gemma4_fused_full_definitive \
  -m pytest -q -s \
  'models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::test_fused_perf_profile[blackhole-full_attention_1024-device_params0-mesh_device0]'
```

Raw merged profiler inputs:

- `generated/profiler/gemma4_fused_sliding_definitive/reports/2026_09_05_05_48_35/ops_perf_results_2026_09_05_05_48_35.csv`
- `generated/profiler/gemma4_fused_full_definitive/reports/2026_09_05_05_48_59/ops_perf_results_2026_09_05_05_48_59.csv`

The canonical unprofiled timing JSONs were restored after Tracy with the same
two pytest nodes, replacing the Tracy variables with:

```bash
GEMMA4_FUSED_DECODER_PERF=1 \
GEMMA4_FUNCTIONAL_DECODER_TRACE_WARMUPS=20 \
GEMMA4_FUNCTIONAL_DECODER_TRACE_ITERATIONS=200 \
GEMMA4_FUSED_CANDIDATE_ID=definitive_final_v2_sliding
```

and `GEMMA4_FUSED_CANDIDATE_ID=definitive_final_v2_full`, respectively. The
durable copies are under `candidate_runs/`; the canonical JSONs at the stage
root contain the exact complete command and resolved policy.

## Report generation

Each `prefill.csv`/`decode.csv` was generated with `tt-perf-report` 1.2.9:

```bash
tt-perf-report RAW.csv \
  --start-signpost START \
  --end-signpost END \
  --arch p150 --no-color --no-advice --no-summary \
  --csv OUTPUT.csv
```

The matching `.txt` files use the same command without `--csv`. Signposts:

| Layer kind | Phase | Start/end signposts |
| --- | --- | --- |
| sliding | prefill | `PERF_PREFILL_layer0_sliding_attention_seq1024_batch1{,_END}` |
| sliding | decode | `PERF_DECODE_layer0_sliding_attention_seq1024_batch1{,_END}` |
| full | prefill | `PERF_PREFILL_layer5_full_attention_seq1024_batch1{,_END}` |
| full | decode | `PERF_DECODE_layer5_full_attention_seq1024_batch1{,_END}` |

| Layer kind | Prefill device total | Traced decode device total | Host prefill/decode |
| --- | ---: | ---: | ---: |
| sliding | 277.751 ms (456 ops) | 1.269 ms (65 ops) | 278.476 / 1.311 ms |
| full | 278.891 ms (456 ops) | 1.451 ms (67 ops) | 279.607 / 1.495 ms |

Functional-stage device baselines are 1,242.489/3.012 ms (sliding) and
1,243.618/3.207 ms (full). Matching functional host baselines are
1,243.063/3.052 ms and 1,244.202/3.230 ms.

## Table conclusions

- packed gate/up sparse matmul is 3.54 ms per prefill group and 222
  microseconds in decode, using 48 cores at about 323/318 GB/s;
- expert down sparse matmul is 4.43 ms per prefill group and 165 microseconds
  in decode, using 8 cores;
- prefill has 64 sparse matmuls (two per internal group) and one concat;
- decode has two sparse matmuls;
- the final source has no Torch/from/to host fallback or reshard;
- remaining typecasts and tilize/untilize operations surround router
  top-k/scatter and its required row-major sparse mask, while head layout
  changes belong to attention consumers. The reports expose no removable
  round-trip pair introduced by the fused stage.
