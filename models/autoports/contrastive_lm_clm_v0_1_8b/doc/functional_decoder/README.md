# Stage 1: functional decoder layer (encoder mapping)

Goal mapping: the plugin's functional decoder stage asks for a tested single-layer TTNN decoder on a 1x1 mesh with
PCC >= 0.995 for prefill and decode. CLM-v0.1-8B uses Qwen3-8B as a frozen encoder with no decode step, so the
decode contract is recorded as not applicable and the prefill contract is tested at the lengths the serving path
uses. The layer implementation is `models.tt_transformers.tt.decoder.TransformerBlock` (the stock Qwen3 decoder
block: fused QKV projection, q/k RMSNorm, RoPE, paged SDPA, output projection, gated MLP), driven through the
autoport wrapper and test.

## Test

`tests/test_functional_decoder.py` (run with `pytest`, mesh `(1,1)`, `l1_small_size=32768`):

- real weights for layers 0, 17 and 35 (bfp8 weights, bf16 activations);
- real activations: the HF bf16 model's hidden state entering each layer for a 2048-token
  tale-of-two-cities excerpt (`reference/hf_layer_reference.py`, whose calling convention was verified on CPU:
  layer output equals the model's own next hidden state exactly, and layer 35 plus the final norm matches
  `last_hidden_state` to cosine 0.99997);
- lengths 32, 33, 127, 128, 129, 500, 1024, 2048 (boundaries of the 128 / 1024 / 2048 prefill buckets),
  each padded to its bucket the way the model does; users 0 to 3 at 128 tokens (four KV-cache slots);
- PCC of the first `n` positions against the HF layer run on the same input.

## Result

`layer_pcc.json`, 33 rows, gate 0.995, device P150:

| layer | min PCC | mean PCC |
|---|---|---|
| 0 | 0.99976 | 0.99978 |
| 17 | 1.00000 | 1.00000 |
| 35 | 0.99972 | 0.99978 |

All rows pass. Lowest rows are layer 35 at 32 and 33 tokens (0.99972, 0.99973). Non-aligned lengths (33, 127,
129, 500) match aligned ones.

- Plain run: `/home/hous/dev/clm-v0.1-8B/logs/stage1_functional_decoder_v3.log`, 1 passed in 26.4 s.
- Watcher run (`TT_METAL_WATCHER=10`): `/home/hous/dev/clm-v0.1-8B/logs/stage1_functional_decoder_watcher_v3.log`,
  1 passed in 38.9 s, no watcher errors.
- Determinism: the full-encoder probe and the trace replay check cover run-to-run determinism (cosine 1.0).

## Per-op device profile (Tracy)

`tracy/layer0/` holds the device profiler capture of layer 0 at 128 (four users) and 1024 tokens
(`python -m tracy -r -p`, then `tt-perf-report`): `prefill_perf_report.csv`, `prefill_perf_report_stacked.csv`
and `.png`, console log. Summary of the 130 ops (11.16 ms of device time over five layer passes):

| op | share of device time | utilization reported by tt-perf-report |
|---|---|---|
| MLP w1/w3 matmul 128x4096x12288 (bfp8, HiFi2) | 17.9 % | DRAM 43 %, FLOPs 58 %, 64 cores |
| wo matmul at 1024 tokens (bf16, HiFi4) | 13.1 % | FLOPs 79 %, 64 cores |
| QKV matmul 128x4096x6144 (bf16, HiFi4) | 9.5 % | DRAM 39 %, FLOPs 55 % |
| MLP w2 matmul 128x12288x4096 | 8.2 % | DRAM 45 %, FLOPs 64 % |
| RMSNorm (20 calls) | 7.2 % | |
| SDPA (5 calls) | 4.7 % | |

`tt-perf-report` advice on every large matmul: "Increase grid size (currently using 64)" (the chip has 110
worker cores) and "HiFi2 is sufficient for BFP8 multiplication" on the HiFi4 attention matmuls. These two
leads are the stage 3 optimization candidates; the second is what the `bfp8_attn_hifi2` precision policy tests.
Overall DRAM roofline for modeled ops: 23.6 % (121 GB/s).

## Not applicable

Decode path, paged decode cache updates, position tensors and decode traces: the encoder runs prefill only and
never reuses a KV cache across requests (`doc/context_contract.json`, `decode_contract`).
