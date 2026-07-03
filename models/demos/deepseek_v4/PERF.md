<!--
SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4-Flash on QuietBox (4× Blackhole p300c) Performance

This file contains the measured performance of DeepSeek-V4-Flash on a Blackhole QuietBox (4× p300c,
`p300x2`), coupled with the run commands we have used to generate these numbers.

Please note that using more recent versions of the software stack (TT-Metal and vLLM) might lead to different
performance numbers.

> **Scope of these numbers.** Produced by the resident + sharded + Metal-Traced decode/prefill engine
> (`demo/decode_engine.py`) running the model's **full, actual op structure — no approximations**: 43 layers with
> the real per-layer schedule (2 sliding + alternating CSA/HCA), the **mHC 4-stream residual with the full 20-iter
> Sinkhorn** (attn_hc + ffn_hc + hyper-head), the **CSA/HCA KV compressors and the lightning indexer** (q_b proj,
> scorer matmul, top-k), MLA attention with KV cache, and top-6 MoE — with weights resident and tensor-parallel-
> sharded across the 4 chips. It is an op-structure/shape/depth/sharding/precision-faithful perf harness (weights
> are random, so it measures latency, not the model's text output; per-token numerical correctness is validated to
> PCC ≥ 0.99 separately in `tests/test_*_pcc.py`). Batch-1, single-user.

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
| 128          | 128           | 1     | 411.5 ms        | 162.7 ms, 6.15 t/s/u                    |
| 2K           | 128           | 1     | 3,651.0 ms      | 177.5 ms, 5.63 t/s/u                    |

**Targets:** ≥ 5 t/s/u decode and < 5 s TTFT — met at both context lengths above with the full actual op
structure (`test_perf.py -m models_performance_bare_metal` asserts these and passes). Standard perf CSV emitted
via `models/perf/prep_perf_report`; CI benchmark JSON via `models/perf/benchmarking_utils`.

Notes: the full mHC Sinkhorn + CSA/HCA compressors + indexer take decode from ~16 t/s/u (their ops omitted) to
~6 t/s/u — the honest cost of the actual computation. TTFT grows with prompt length (the indexer/compressor and
prefill matmuls scale with sequence); beyond ~2–3K it will exceed 5 s and needs chunked prefill (not yet wired).

### Reference point (same hardware)

For context, the tt-metal target for a comparable 120B MoE on this exact SKU (`models/model_targets.yaml`):

| Model | HW | Batch | Seq | TTFT | decode t/s/u |
|---|---|---|---|---|---|
| gpt-oss-120b | p300x2 (4× Blackhole) | 1 | 128 | 893 ms | 24.46 |
