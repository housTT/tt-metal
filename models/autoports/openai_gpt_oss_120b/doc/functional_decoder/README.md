# OpenAI GPT-OSS 120B functional decoder

Status: functional-decoder stage complete for a single P150/Blackhole device.

The implementation is `tt/functional_decoder.py`. It covers both GPT-OSS decoder-layer kinds—sliding attention with window 128 and full attention—using the real 120B dimensions. P150x2 and P150x4 remain deployment targets, but their tensor/expert-parallel collectives belong to the later multichip-decoder stage and are intentionally outside this directory.

## Public contract

`FunctionalDecoder.from_state_dict` is the sole HF/host weight-conversion boundary. It accepts either model-prefixed keys such as `model.layers.0.self_attn.q_proj.weight` or a state dict local to one `GptOssDecoderLayer` and validates all 17 required tensors.

Prefill is paged-only:

```python
output = decoder.prefill_forward(
    hidden_states,                  # [1, batch, sequence, 2880]
    position_embeddings=[cos, sin],
    page_table=page_table,          # device int32
    kv_cache=optional_external_cache,
    user_id=0,
    batch_size=batch,
)
```

Decode is paged-only and trace-safe:

```python
output = decoder.decode_forward(
    hidden_states,                  # [1, 1, batch, 2880]
    position_embeddings=[cos, sin],
    current_position=current_pos,   # device int32, one value per user
    page_table=page_table,
    kv_cache=optional_external_cache,
    batch_size=batch,
)
```

The decoder owns tile padding and preserves the caller's public activation buffer. Logical prefill lengths need not align to a 32-token tile, 64-token page, 128-token sliding window, or 4096-token expert chunk. Both methods require a `1x1` mesh in this stage. `forward(..., mode="prefill"|"decode")` dispatches to the same contracts.

## Correctness evidence

The primary batch-one full-decoder model-specific acceptance bar is PCC `>= 0.95`; the seeded-random batch-two routing/position/page-table stress test uses `0.99`. Continuous components and fixed-route counterfactuals retain the default `0.995` bar. The reason for these exceptions is the model's discontinuous hard top-4 routing and is demonstrated in [precision_investigation.md](precision_investigation.md).

| Weights | Layer kind | Prefill PCC | Traced-decode PCC |
| --- | --- | ---: | ---: |
| Real checkpoint layer 0 | sliding | 0.977154173283 | 0.992163253293 |
| Real checkpoint layer 1 | full | 0.990045250294 | 0.965931676798 |
| Stats-derived synthetic layer 0 | sliding | 0.981616212808 | 0.980701267793 |
| Stats-derived synthetic layer 1 | full | 0.987222548868 | 0.993940590124 |

The stats-derived batch-two results were 0.983274679196/0.994583692307 for sliding prefill/decode and 0.990537791979/0.997145155509 for full prefill/decode. Decode used seeded-random distinct device positions `[8, 32]` and `[26, 16]`, respectively, disjoint page-table rows, and independently reconstructed HF prefixes. Both prefill cases pass `0.95`; both traced-decode cases pass the separately declared `0.99` batch-routing gate.

The test matrix additionally covers:

- permuted/disjoint page tables and device-resident current positions;
- HF-vs-TTNN batch-2 paged prefill/decode for both layer kinds, with distinct per-user current positions;
- a batch-32 full-context-per-user capacity test with complete trace replay;
- HF-valid length 1, plus 31/32/33, 63/64/65, 127/128/129, and 4095/4096/4097;
- prefill lengths 131071 and 131072 for both layer kinds;
- traced decode at position 131071, proving context 131072 addressability;
- bitwise-equal repeated trace replay and preserved public input addresses.

The checkpoint revision and exact population statistics for every layer-0/layer-1 tensor are in [real_weight_stats.json](real_weight_stats.json). Normal synthetic fixtures preserve each checkpoint-derived mean and use `min(real_std, initializer_range)` for independent noise; the cap avoids fabricating an ill-conditioned untrained model after learned cross-tensor correlations are removed. The opt-in real-weight test separately uses the exact dequantized tensors.

The retained opt-in `tests/test_routing_precision_analysis.py` re-derives the PCC exception with real weights, two users, both layer kinds, and actual traced-decode states. With identical TT routes, HF-vs-TT tail PCC is 0.998856913 (sliding) and 0.998906048 (full); raw output is under `correctness/`.

Losslessly compressed raw transcripts for the length-one/boundary, primary real-weight acceptance, chunk-boundary, advertised-context, default-suite, batch-two, batch-32, and routing-counterfactual gates are retained under `correctness/`; `work_log.md` records every original and compressed SHA-256.

## Context and cache

- HF advertised, configured, supported, and tested context: 131072.
- Capability reduction: none.
- Page size: 64; page tables and positions are device-resident `int32` tensors.
- Largest HF correctness batch: 2.
- Largest device capacity batch: 32, with 131072 context allocated per user.

The machine-readable record is [../context_contract.json](../context_contract.json).

## Warmed performance

Tracy measurements use sequence 128, a warmed prefill, and a warmed replay of the complete decode trace. Times below are summed device-op time from signpost-filtered `tt-perf-report` 1.2.8 output; they are functional-stage measurements, not optimization claims.

| Layer kind | Warmed prefill | Traced warmed decode | Device ops / host ops |
| --- | ---: | ---: | --- |
| sliding attention | 137.401 ms | 2.060 ms | 58/0 prefill, 64/0 decode |
| full attention | 137.418 ms | 2.047 ms | 58/0 prefill, 64/0 decode |

Gzip-compressed raw Tracy CSVs, filtered CSVs, and human-readable tables are under `tracy/sliding_attention/` and `tracy/full_attention/`. The original and compressed raw-CSV SHA-256 values are recorded in `work_log.md`.

## Runtime and device safety

[runtime_fallback_audit.md](runtime_fallback_audit.md) records the reachable call-tree audit: there is no PyTorch, `ttnn.from_torch`, `ttnn.to_torch`, or host fallback inside a measured decoder pass. The final watcher run covers both layer kinds separately from profiling; its compressed log and scan result are under `watcher/final_correctness/`.

See [work_log.md](work_log.md) for exact commands, all boundary/context results, profiler provenance, limitations, and commit/review records.
