# DeepSeek-V4-Flash on Tenstorrent QuietBox — Serving Bring-Up One-Pager

**Goal:** serve `nvidia/DeepSeek-V4-Flash-NVFP4` through tt-inference-server and improve decode throughput.
**Result:** correct serving + a **9× decode speedup**, measured end-to-end through the server:
**0.15 tok/s (recompute baseline) → 1.39 tok/s warm decode** (720 ms/token, via `/v1/completions`).

---

## Changes by repo

### 1. `tt-metal` — `/home/ttuser/.local/lib/model-bringup/tt-metal`
All model work lives in `models/demos/deepseek_v4/`.

- **`reference/real_weights.py`** — NVFP4 checkpoint support: `_dequant_nvfp4` + an NVFP4 branch in `deq()` (uint8 `.weight` + e4m3 `.weight_scale` + fp32 `.weight_scale_2`); `find_snapshot()` honors the `DEEPSEEK_V4_MODEL` env var. Validated lossless vs base (mse=0).
- **`tt/mla_v4_device.py` (NEW, core deliverable)** — the entire **on-device, traceable** decode, each op PCC-validated > 0.999 vs the host reference:
  - MLA-v4 attention (`mla_decode_device`): on-device interleaved partial RoPE (as a fixed [64,64] matmul), attention-sink softmax, −sin un-rotation, grouped o_lora.
  - CSA + HCA compressors (`csa_compress_device`, `hca_compress_device`) — windowed gated-softmax pools on device; the lightning-indexer top-k is a proven no-op within the 2048 context.
  - mHC hyperconnection (`mhc_device`) with a capped Sinkhorn.
  - MoE (`moe_device_fp4` + `Fp4ExpertCache`): resident router + shared expert, **streamed fp4 routed experts**, on-device fp32 accumulate.
  - `LayerDeviceWeights` + `decode_layer_device` assemble one full traceable layer.
- **`tt/fast_decode.py` (NEW)** — `FastDecoder`: 43 resident layer-weight sets + streamed fp4 experts; `prefill` / `decode_step`. This is the throughput path.
- **`tt/generator.py`** — added `prefill_fast` / `decode_fast` (and the earlier KV-cache path `prefill_kv` / `decode_kv`).
- **`tt/generator_vllm.py`** — vLLM adapter `prefill_forward` / `decode_forward` now drive the fast on-device path (was full-context recompute).
- **`tt/{attention,compressors,model,modules,kv_cache_decode}.py`** — host reference math + building blocks used to PCC-validate every device op.
- **vLLM plugin `platform.py`** — registers `DeepseekV4ForCausalLM` / `TTDeepseekV4ForCausalLM` → `generator_vllm`.
- **`demo/test_*_device.py`, `demo/fast_decode_bench.py`, `demo/micro_*.py`** — PCC-validation + perf micro-benchmarks.

### 2. `tt-inference-server` — `/home/ttuser/code/tt-inference-server`
- **`workflows/model_spec.py`** — `DeepSeek-V4-Flash` template + `P300X2`/`P300` device specs (max_context 2048, max_concurrency 1), `env_vars={"DEEPSEEK_V4_MODEL": "nvidia/DeepSeek-V4-Flash-NVFP4"}`, chat-template path.
- **Launch:** `run.py --model DeepSeek-V4-Flash --workflow server --local-server --tt-device p300x2 --no-auth --disable-trace-capture --disable-metal-timeout` (see key flags below). No source change beyond the model spec.

### Key operational flags / gotchas
- **`--tt-device p300x2`** (4 chips): the 2-chip `p300` path fails on the flaky chip 2-3 ethernet fabric link. The model still runs on a (1,1) submesh.
- **`--disable-trace-capture`**: the built-in warmup does cold ISL=128 prefills → hours at cold-tilize speed.
- **`--disable-metal-timeout`**, and warm the fp4 cache before measuring (first request per prompt is cold ~55 s/token).

---

## Performance today (measured)

| path | decode tok/s | notes |
|---|---|---|
| recompute (original vLLM) | 0.15 | re-ran whole context per token |
| bf16 KV-cache decode | ~0.15 | correct, still per-expert bf16 upload |
| **fp4 on-device fast decode (current)** | **1.39** | warm, through tt-inference-server |

Root cause of the ceiling: single-stream decode is **weight-movement-bound** — 3.25 GB of fp4 experts stream over one chip's PCIe per token (~9.5 GB/s), since the 157 GB model can't be DRAM-resident (128 GB).

---

## Next steps & expected peak

The single biggest lever is **multi-chip expert-parallel MoE**: distribute each token's routed experts across all 4 chips so each uploads ~1/4 of the weight over its own PCIe in parallel (~4× the effective host→DRAM bandwidth), with attention/mHC replicated and a single all-gather to combine — projected **~3 tok/s single-stream**. Secondary wins: persist the tilized `bfloat4_b` experts to disk (`ttnn.dump_tensor`) so the server starts warm instead of paying the ~40 min one-time tilize; dual-command-queue upload to overlap expert DMA behind matmul compute; and trimming the remaining ~160 ms/token of attention+mHC dispatch. **Expected peak single-stream on this 4-chip box is ~3 tok/s** (fundamentally DMA-bound). Substantially higher **aggregate** throughput is reachable with **large-batch continuous batching** (B ≥ ~64): once the routed union saturates all 256 experts/layer, the full model is uploaded once per forward and amortized across all users → **~10-20+ tok/s aggregate** (needs batched decode + raising `max_concurrency` from 1). The only route to ≥10 **single-stream** is more chips so the experts fit resident and streaming disappears entirely.
