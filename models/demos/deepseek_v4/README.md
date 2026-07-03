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

# Run the demo: tokenize a prompt, generate on device, decode, and verify vs the HF reference
python models/demos/deepseek_v4/demo/demo.py --prompt "The capital of France is" --max-new-tokens 16 --verify

# Or via pytest (like other demos)
pytest models/demos/deepseek_v4/demo/test_demo.py
```

Example output (reduced config, so the *text* is not meaningful English — but the device tokens exactly match
the HF reference, i.e. the pipeline is functionally correct):

```
Prompt: 'The capital of France is'  ->  5 tokens
Device arch: Arch.BLACKHOLE. Generating 8 tokens on device...
Generated token ids: [48873, 113351, 16947, 6816, 63223, 116975, 5239, 25301]
[verify] device tokens == HF reference tokens: True
DEMO_OK
```

## Scope & how to run full weights

To keep a single laptop-free session tractable, the demo runs a **reduced-but-real-code-path** config (fewer
layers/hidden/experts) with the **real tokenizer and vocab** — the same code path as the full model, per the
bring-up guide's "reduce for correctness, never fake" rule. Weights are a reproducible reduced random init, so
the generated *text* is gibberish; `--verify` proves correctness by matching the HF reference token-for-token.

To run the full pretrained model, replace `build_reduced_model` (in `reference/reduced_config.py`) with a
full-weight loader — fp8 block-dequant (`models/demos/deepseek_v3/utils/hf_model_utils.py`) + the full
`config.json` — and keep the same `tt/model.py::tt_forward` loop. This needs a multi-chip Blackhole system to
hold the 256 experts; see `docs/TT_REUSE_MAP.md` and `../../../REPORT_5.md`.

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
demo/demo.py          text-in -> generate-on-device -> text-out demo (+ --verify)
demo/test_demo.py     pytest wrapper
demo/benchmark_tt.py  hardware benchmark harness ([hw] module table; refuses simulator)
demo/forge_demo.py    tt-forge (PJRT) compiler-path demo
tt/attention.py       MLA-v4 attention
tt/compressors.py     CSA/HCA compressors + lightning indexer
tt/modules.py         RMSNorm, SwiGLU, o_lora, router, mHC, MoE experts
tt/model.py           end-to-end forward (mHC decoder assembly)
reference/            reduced-config correctness oracle (HF DeepseekV4)
docs/                 HF math spec, TT reuse map, optimization log
```
