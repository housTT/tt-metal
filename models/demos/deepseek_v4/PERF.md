<!--
SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4-Flash on QuietBox (4× Blackhole p300c) Performance

This file contains the measured performance of DeepSeek-V4-Flash on a Blackhole QuietBox (4× p300c,
`p300x2`), coupled with the run commands we have used to generate these numbers.

Please note that using more recent versions of the software stack (TT-Metal and vLLM) might lead to different
performance numbers.

> **Scope of these numbers.** They are produced by the resident + sharded + Metal-Traced decode/prefill engine
> (`demo/decode_engine.py`), which runs the model's real op structure — 43 layers, top-6 MoE, MLA attention, KV
> cache — with weights resident and tensor-parallel-sharded across the 4 chips. The mHC 4-stream Sinkhorn residual
> is approximated by a residual add and the CSA/HCA compressor + lightning-indexer ops are not yet folded in, so a
> fully-faithful build will be somewhat slower (est. ~9–11 t/s/u). Per-module numerical correctness is validated to
> PCC ≥ 0.99 separately (`tests/test_*_pcc.py`). This is batch-1, single-user.

## 2026-07-03

- TT-Metal: `v0.75.0-dev20260703` (`1ca98673654`)
- Hardware: QuietBox, 4× Blackhole `p300c` (~28 GB usable DRAM/chip), TT-KMD 2.9.0, firmware 19.11.0
- Precision: bf16 activations, resident experts bf16 (bf8 also validated, PCC 0.99144)

### TT-Metal runs

```
pytest models/demos/deepseek_v4/tests/test_perf.py -m models_performance_bare_metal -s
```

To sweep sequence lengths directly:

```
python models/demos/deepseek_v4/demo/decode_engine.py --layers 43 --seq 128
python models/demos/deepseek_v4/demo/decode_engine.py --layers 43 --seq 2048
```

| Input length | Output length | Batch | TTFT (per user) | Token/s/u (avg of all decoded tokens) |
|--------------|---------------|-------|-----------------|----------------------------------------|
| 128          | 128           | 1     | 94.3 ms         | 61.4 ms, 16.29 t/s/u                    |
| 512          | 128           | 1     | 144.6 ms        | 61.9 ms, 16.15 t/s/u                    |
| 2K           | 128           | 1     | 322.8 ms        | 64.2 ms, 15.59 t/s/u                    |
| 8K           | 128           | 1     | 1,129.6 ms      | 73.4 ms, 13.62 t/s/u                    |

**Targets:** ≥ 5 t/s/u decode and < 5 s TTFT — met at every context length above (`test_perf.py` asserts these
and passes). Standard perf CSV emitted via `models/perf/prep_perf_report`; CI benchmark JSON via
`models/perf/benchmarking_utils` (`BenchmarkData.save_partial_run_json`).

### Reference point (same hardware)

For context, the tt-metal target for a comparable 120B MoE on this exact SKU (`models/model_targets.yaml`):

| Model | HW | Batch | Seq | TTFT | decode t/s/u |
|---|---|---|---|---|---|
| gpt-oss-120b | p300x2 (4× Blackhole) | 1 | 128 | 893 ms | 24.46 |
