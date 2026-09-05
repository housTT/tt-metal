# Functional-decoder performance provenance

These reports measure the real-weight, batch-1 functional decoder at sequence
length 1024 on one Blackhole P300 chip (device id 1). Prefill is warmed once.
Decode is captured with TTNN trace execution, replayed once as a warmup, then
measured with one blocking trace replay. Profiler-buffer drains occur between
complete passes and outside the per-replay host-timing interval.

The exact sliding-attention workload command was:

```bash
source python_env/bin/activate && \
GEMMA4_FUNCTIONAL_DECODER_PERF=1 \
GEMMA4_FUNCTIONAL_DECODER_TRACE_WARMUPS=1 \
GEMMA4_FUNCTIONAL_DECODER_TRACE_ITERATIONS=1 \
timeout 1800 \
python -m tracy -r -p -v --check-exit-code \
  -o generated/profiler/gemma4_functional_sliding_v3 \
  -m pytest -q -s \
  'models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py::test_functional_decoder_perf_profile[blackhole-batch1-sliding_attention_1024-device_params0-mesh_device0]'
```

The exact full-attention command was:

```bash
source python_env/bin/activate && \
GEMMA4_FUNCTIONAL_DECODER_PERF=1 \
GEMMA4_FUNCTIONAL_DECODER_TRACE_WARMUPS=1 \
GEMMA4_FUNCTIONAL_DECODER_TRACE_ITERATIONS=1 \
timeout 1800 \
python -m tracy -r -p -v --check-exit-code \
  -o generated/profiler/gemma4_functional_full_v3 \
  -m pytest -q -s \
  'models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py::test_functional_decoder_perf_profile[blackhole-batch1-full_attention_1024-device_params0-mesh_device0]'
```

`prefill.csv` and `decode.csv` are machine-readable signpost slices produced
from Tracy's merged host/device operation reports by `tt-perf-report` 1.2.9.
They retain every operation in the measured phase. The matching `.txt` files
are the human-readable tables produced from the same slices.

Report generation used `--arch p150 --no-color --no-advice --no-summary` and
these signpost pairs:

| Layer kind | Phase | Start/end signposts |
| --- | --- | --- |
| sliding attention | prefill | `PERF_PREFILL_layer0_sliding_attention_seq1024_batch1{,_END}` |
| sliding attention | decode | `PERF_DECODE_layer0_sliding_attention_seq1024_batch1{,_END}` |
| full attention | prefill | `PERF_PREFILL_layer5_full_attention_seq1024_batch1{,_END}` |
| full attention | decode | `PERF_DECODE_layer5_full_attention_seq1024_batch1{,_END}` |

| Layer kind | Prefill device total | Traced decode device total | Host timing |
| --- | ---: | ---: | ---: |
| sliding attention | 1,242.489 ms (557 ops) | 3.012 ms (74 ops) | 1,243.350 / 3.112 ms |
| full attention | 1,243.618 ms (557 ops) | 3.207 ms (76 ops) | 1,244.512 / 3.304 ms |

Host values are `prefill_host_ms / decode_trace_host_ms`. These measurements
are functional-stage baselines, not performance-improvement claims.
