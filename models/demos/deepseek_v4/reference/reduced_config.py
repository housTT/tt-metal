# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Correctness-oracle helpers for the DeepSeek-V4 TT-NN bring-up.

The HF `transformers.models.deepseek_v4` implementation is the golden reference
for all PCC checks (per GOAL Checkpoint 2 methodology). Full DeepSeek-V4-Flash is
frontier-scale (~160 GB fp8, 43 layers, 256 experts) and impractical to run whole
on the correctness path. This module builds a *reduced but architecturally-faithful*
config through the SAME code path (GOAL rule 4: reduce for correctness, never fake).

Two modes:
  * `build_reduced_model(...)`  -> random-weight tiny model; identical weights feed
    both the HF reference and the TT-NN port, so PCC compares like-with-like WITHOUT
    the 160 GB download. This is the primary Checkpoint-2 oracle.
  * (real-weight reduced runs, which need the first few safetensors shards, are wired
    separately in the demo; not needed for per-module/per-layer PCC.)

Nothing here contains a simulator/hardware conditional (GOAL rule 2).
"""
from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForCausalLM

HF_MODEL = "deepseek-ai/DeepSeek-V4-Flash"


# Default reduced ("sim/correctness") config. Shrinks every dimension while keeping
# ALL structural knobs (MLA, mHC 4-stream residual, CSA/HCA compressors, indexer,
# hash+moe routing, shared expert, clamped SwiGLU) so the same code path is exercised.
# num_hidden_layers=4 => layers 0-2 hash_moe + 1 moe; layers 0-1 sliding, 2-3 CSA/HCA.
REDUCED = dict(
    num_hidden_layers=4,
    hidden_size=256,
    intermediate_size=512,
    moe_intermediate_size=128,
    n_routed_experts=8,
    num_experts_per_tok=2,
    n_shared_experts=1,
    num_attention_heads=8,
    head_dim=128,
    q_lora_rank=128,
    o_lora_rank=128,
    o_groups=8,
    index_n_heads=8,
    index_head_dim=64,
    index_topk=64,
    vocab_size=1024,
    num_nextn_predict_layers=0,  # drop MTP for base greedy decode
)


def make_reduced_config(overrides: dict | None = None):
    """Return a DeepseekV4Config with reduced dims (real config as the base)."""
    cfg = AutoConfig.from_pretrained(HF_MODEL)
    red = dict(REDUCED)
    if overrides:
        red.update(overrides)
    for k, v in red.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    # keep per-layer schedules consistent with the reduced layer count
    n = cfg.num_hidden_layers
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = cfg.layer_types[:n]
    if getattr(cfg, "mlp_layer_types", None):
        cfg.mlp_layer_types = cfg.mlp_layer_types[:n]
    return cfg


def build_reduced_model(seed: int = 0, dtype=torch.float32, overrides: dict | None = None):
    """Build a tiny random-weight DeepseekV4 on CPU (the PCC golden).

    Deterministic for a given seed so the TT-NN side can load the identical
    state_dict. Returns (model, config).
    """
    cfg = make_reduced_config(overrides)
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(cfg, dtype=dtype).eval()
    return model, cfg


def reference_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids).logits


if __name__ == "__main__":
    # Smoke test the oracle (CPU-only, no ttnn, no download).
    m, cfg = build_reduced_model(seed=0)
    n_params = sum(p.numel() for p in m.parameters())
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    logits = reference_logits(m, ids)
    layer0 = m.model.layers[0]
    print(f"reduced model OK: params={n_params/1e6:.2f}M layers={cfg.num_hidden_layers}")
    print(f"  layer0 attn={type(layer0.self_attn).__name__} mlp={type(layer0.mlp).__name__}")
    print(f"  layer_types={cfg.layer_types}")
    print(f"  mlp_layer_types={cfg.mlp_layer_types}")
    print(f"  logits shape={tuple(logits.shape)} dtype={logits.dtype}")
    # determinism check: same seed -> identical logits
    m2, _ = build_reduced_model(seed=0)
    logits2 = reference_logits(m2, ids)
    same = torch.equal(logits, logits2)
    print(f"  determinism (seed=0 twice identical): {same}")
