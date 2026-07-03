<!--
SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# TT-NN Reuse Map + Methodology for the DeepSeek-V4 port

Root: `/home/ttuser/.local/lib/model-bringup/tt-metal`.

## PCC helpers (use these)
- `from models.common.utility_functions import comp_pcc`  → `(passed, pcc_value)`, default 0.99, fp32/float64 accum, NaN/Inf-safe (`models/common/utility_functions.py:488`).
- `from tests.ttnn.utils_for_testing import assert_with_pcc`  → asserts, default 0.9999 (`tests/ttnn/utils_for_testing.py:94`).
- Thresholds (llms.md §4.4.1): submodule ~0.999, decoder layer ~0.998, full model ~0.99.

## What already exists for V4 (anchor)
- `models/common/modules/moe/configs/deepseek_v4_flash.yaml` — V4 MoE config (hidden 4096, 256 experts, top-6, sqrtsoftplus, noaux_tc bias, scale 1.5, shared=1, SILU, interm 2048). **Exercised** in `models/common/tests/modules/moe/test_tt_moe_decode.py`.
- `models/demos/deepseek_v3_d_p/reference/deepseek_v4_flash_config.py` — reference config builder.
- Generalized MoE = the only V4-aware TT code today: `TTMoEGate` (`tt_moe_gate.py:63`, `.forward→(scores,indices)`, `sqrtsoftplus` applied externally as `ttnn.sqrt(ttnn.softplus(x))`) + `TTMoEDecode` (`tt_moe_decode.py:671`, `.forward(tt_x,tt_scores,tt_indices,layer_id)`). Config `TTMoEDecodeConfig.from_yaml(text, topology)`; `.with_mesh_shape((r,c))` slices experts for a smaller mesh (reduced-config lever).

## Reuse tiers
**Strong reuse (standard modules):**
- RMSNorm: `deepseek_v3/tt/rms_norm/{rms_norm.py,distributed_rms_norm.py}`; ttnn.rms_norm / rms_norm_pre/post_all_gather.
- MLP/SwiGLU: `deepseek_v3/tt/mlp/mlp.py:45` (w1/w2/w3=gate/down/up), `mlp_dequant.py` for bf16-dequant loading.
- Embedding: `deepseek_v3/tt/embedding/{embedding1d,embedding2d}.py`; `ttnn.embedding`.
- LM head: `deepseek_v3/tt/lm_head1d.py:38`; DRAM-sharded matmul, column splits.
- RoPE primitives: `deepseek_v3/tt/rope.py` (`RotarySetup`, `get_cos_sin_matrix`, `get_rot_transformation_mat`); interleave perm `deepseek_v3_d_p/tt/mla/rope.py:79 interleaved_to_halfsplit_perm`.
- fp8 block dequant: `deepseek_v3/utils/hf_model_utils.py:201 dequantize_weight_tensor`, `:244 dequantize_state_dict`; CLI `scripts/dequantize_hf_checkpoint.py`.
- MoE (`moe` layers): generalized MoE above.
- Assembly patterns: `deepseek_v3/tt/model/row_batched_model.py:40` (AbstractModule + RunConfig, prefill chunking + decode loop) OR `deepseek_v3_d_p/tt/tt_prefill_transformer.py` (stateful).

**Pattern reference only (NOT drop-in — V4 shapes differ):**
- `deepseek_v3_d_p/tt/mla/ttMLA` (`mla.py:20`) + `TtIndexer` (`indexer.py:28`): this is V3.2 sparse MLA (kv_lora_rank based, `deepseek_v32`). V4 MLA is K==V shared MQA, no kv_lora_rank, adds grouped o_lora + sinks + CSA/HCA compressors. Use for: weight-cache pattern, paged KV, RoPE interleave, topk-indexer scaffolding — reimplement the math per HF_REFERENCE_SPEC.
- Blackhole kernels `deepseek_v3_b1/unified_kernels/{flash_mla,deepseek_moe_gate,rope,create_q_heads,sampling,argmax}.hpp` — for later optimization.

**Net-new TT-NN authoring (no prior art):**
- MLA-V4 attention: K==V MQA, partial interleaved RoPE + conjugate -sin un-rotation, attention sinks, grouped `o_lora` block-diagonal bmm, CSA/HCA windowed compressors + caches, lightning indexer top-512.
- **mHC 4-stream residual** (HyperConnection + HyperHead, Sinkhorn 20 iters) — pervasive, wraps every sublayer.
- Hash routing (frozen `tid2eid[vocab,6]` table) for first 3 layers.

## Reduced-config levers
- `deepseek_v3/conftest.py:193 hf_config_short` → `num_hidden_layers = request.param`.
- `deepseek_v3_d_p/conftest.py` → `download_model_weights(...num_layers=N)` (download fewer layers).
- Generalized MoE → `TTMoEDecodeConfig.with_mesh_shape((r,c))` slices experts.
- No in-yaml "truncate experts" flag — reduce at fixture/mesh level.

## PCC test idiom (AbstractModule, copyable)
`deepseek_v3/tests/test_mlp.py::run_test_forward_pass`: build torch ref module → torch_input/reference_output → `Cls.convert_weights(...)` → `get_model_config` → `Cls.create_state` → `create_run_config(mc,wc,state)` → `ttnn.from_torch(input, ShardTensor2dMesh)` → `run_module_forward` → `ttnn.to_torch(ConcatMesh2dToTensor)` → `assert_hidden_dim_pcc(out, ref, 0.975)`.

Fallback-for-missing-op (GOAL anti-BS + TTNN-bringup §3.1): keep the torch op inline (tt→torch→op→tt), file an issue, PCC-check the module, swap to TT-NN op later.

## Suggested V4 build path
1. Reference config: `deepseek_v3_d_p/reference/deepseek_v4_flash_config.py`.
2. Standard modules first (RMSNorm, MLP, embedding, LM head, RoPE) — reuse + per-module PCC.
3. MoE `moe` layers via generalized MoE (`TTMoEGate`+`TTMoEDecode` + v4_flash.yaml).
4. Net-new: MLA-V4, mHC, hash routing — per HF_REFERENCE_SPEC, torch-fallback where TTNN can't express yet.
5. Assemble decoder + model (mirror RowBatchedModel), reduced-config e2e logits PCC.
6. Optimize (Ckpt 3), benchmark on 4×BH (Ckpt 4).
