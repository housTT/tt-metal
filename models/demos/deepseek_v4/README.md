<!--
SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4-Flash (TT-NN, Blackhole)

Hand-written TT-NN (Path B) bring-up of `deepseek-ai/DeepSeek-V4-Flash` on Tenstorrent Blackhole.

## What this is

DeepSeek-V4-Flash is a frontier MoE+MLA model (43 layers, 256 experts, ~160 GB fp8) with several mechanisms
that have no prior tt-metal implementation. This directory contains a hand-written TT-NN port validated bottom-up
against the HuggingFace `transformers.models.deepseek_v4` reference (PCC ≥ 0.99) on real Blackhole silicon,
covering:

- **MLA-v4 attention** — K==V shared-MQA, attention sinks, interleaved partial RoPE, conjugate −sin un-rotation,
  grouped `o_lora` output projection (`tt/attention.py`).
- **CSA / HCA KV compressors + lightning indexer** — windowed gated-softmax compression + top-512 sparse
  selection (`tt/compressors.py`).
- **mHC 4-stream Sinkhorn residual** — HyperConnection + HyperHead (`tt/modules.py`).
- **Hash-routed + top-k MoE** — sqrtsoftplus router, routed experts, shared expert, clamped SwiGLU (`tt/modules.py`, `tt/model.py`).

All learned matmuls/projections/experts/o_lora/LM-head run on device. A few intricate small-tensor control ops
(interleaved RoPE, sink softmax, 4×4 Sinkhorn, top-k/scatter, hash lookup, embedding gather) run on host as
documented fallbacks; migrating them to native TTNN ops is tracked in `docs/OPTIMIZATION_LOG.md`.

## Quick start

```bash
source models/demos/deepseek_v4/env.sh          # venv + TT_METAL_HOME (4x Blackhole)

# ACTUAL pretrained weights — full 43-layer model, streamed per layer, on device:
python models/demos/deepseek_v4/demo/demo.py --real --prompt "The capital of France is" --max-new-tokens 2

# Reduced-config functional demo (fast; verifies pipeline vs HF reference):
python models/demos/deepseek_v4/demo/demo.py --prompt "The capital of France is" --max-new-tokens 16 --verify

# Or via pytest (like other demos)
pytest models/demos/deepseek_v4/demo/test_demo.py
```

### `--real` — actual pretrained weights (the real model)

Runs the **full 43-layer DeepSeek-V4-Flash with its actual fp8/fp4 weights** on Blackhole. Real output:

```
Prompt: 'The capital of France is'
  ... layer 1/43 ... layer 43/43 (streamed per layer)
  token 1/2: id=11111  ' Paris'
  token 2/2: id=64465  '.",\n'
Continuation: ' Paris.",\n'
Full: 'The capital of France is Paris.'
DEMO_OK (actual weights)
```

**How it fits.** The checkpoint is ~149 GB fp8 (attention/shared-expert/LM-head, block-128 scale) + MXFP4 fp4
routed experts; bf16 would be ~320 GB (> host RAM). `reference/real_weights.py` reads the raw safetensors and
**dequantizes one layer at a time** into a 5-layer scratch module (covering all four (attention, MoE) structural
combos), and the MoE dequantizes **only the routed experts** — so peak memory is ~one layer. Loader validated
tensor-for-tensor (max|Δ|=0) vs the transformers-dequantized model; 4-layer end-to-end logits PCC 0.991 vs the HF
reference on device (`tests/test_real_weights.py`, `tests/test_real_e2e.py`).

**Timing** (measured, 4× Blackhole; prefill path, no KV cache): **startup ~10 s** (scratch built with
`_init_weights` no-op'd — otherwise ~4 min), first token ~65 s (cold fp4 dequant), **subsequent tokens ~35 s**
(dequantized experts cached in host RAM + fused gate/up matmul). A 2-token run is ~2 min end-to-end. This is
bandwidth-bound (it streams the touched weights through the host each token); real serving speed needs a KV cache
+ resident weights sharded across the chips + Metal Trace — the perf-scaling work beyond this bring-up.

### Reduced-config functional demo (default)

Runs a smaller config with the **real tokenizer/vocab** for a fast pipeline check; weights are a reproducible
random init so the *text* is gibberish, and `--verify` reports final-token logits PCC vs the HF reference
(≈0.9997) to prove functional correctness.

## Tests (correctness, on Blackhole)

```bash
python models/demos/deepseek_v4/tests/test_modules_pcc.py       # 8 modules, PCC >= 0.99
python models/demos/deepseek_v4/tests/test_compressors_pcc.py   # CSA/HCA compressors + indexer
python models/demos/deepseek_v4/tests/test_e2e_pcc.py           # 2-layer end-to-end (PCC 0.99971, greedy match)
python models/demos/deepseek_v4/tests/test_e2e_4layer_pcc.py    # 4-layer e2e with compressors (PCC 0.99922)
python models/demos/deepseek_v4/tests/test_optimizations.py     # precision sweep (bf16/bfp8/bfp4) + PCC re-check
```

## Layout

```
demo/demo.py          text-in -> generate-on-device -> text-out demo (--real actual weights; --verify)
demo/test_demo.py     pytest wrapper
reference/real_weights.py  stream + dequantize (fp8 + MXFP4) the actual checkpoint, layer by layer
demo/benchmark_tt.py  hardware benchmark harness ([hw] module table; refuses simulator)
demo/forge_demo.py    tt-forge (PJRT) compiler-path demo
tt/attention.py       MLA-v4 attention
tt/compressors.py     CSA/HCA compressors + lightning indexer
tt/modules.py         RMSNorm, SwiGLU, o_lora, router, mHC, MoE experts
tt/model.py           end-to-end forward (mHC decoder assembly)
reference/            reduced-config correctness oracle (HF DeepseekV4)
docs/                 HF math spec, TT reuse map, optimization log
```
