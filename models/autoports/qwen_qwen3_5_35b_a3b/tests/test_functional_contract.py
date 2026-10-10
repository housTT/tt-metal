# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from models.autoports.qwen_qwen3_5_35b_a3b.tests.layer_weights import layer_shards, layer_weight_map
from models.autoports.qwen_qwen3_5_35b_a3b.tt.functional_decoder import FunctionalDecoder, FunctionalDecoderConfig

CHECKPOINT = Path(__file__).resolve().parents[4].parent / "weights" / "Qwen3.5-35B-A3B"


@pytest.fixture(scope="module")
def hf_config():
    with Path(__file__).with_name("target_config.json").open() as stream:
        return json.load(stream)


def test_real_text_config_contract(hf_config):
    config = FunctionalDecoderConfig.from_hf_config(hf_config)
    assert (config.n_layers, config.dim, config.max_seq_len) == (40, 2048, 262144)
    assert (config.n_heads, config.n_kv_heads, config.head_dim) == (16, 2, 256)
    assert (config.linear_q_dim, config.linear_k_dim, config.linear_v_dim) == (2048, 2048, 4096)
    assert (config.moe_num_experts, config.moe_top_k) == (256, 8)
    assert (config.moe_intermediate_size, config.moe_shared_intermediate_size) == (512, 512)
    assert config.moe_norm_topk_prob is True
    assert all(config.is_moe_layer(i) for i in range(config.n_layers))
    assert config.layer_kind(0) == "linear_attention"
    assert config.layer_kind(3) == "full_attention"
    assert [i for i in range(config.n_layers) if config.is_full_attention_layer(i)] == list(range(3, 40, 4))


@pytest.mark.parametrize("logical,physical", [(1, 128), (127, 128), (128, 128), (129, 256), (2200, 2304)])
def test_prefill_physical_length(logical, physical):
    assert FunctionalDecoder.physical_prefill_length(logical) == physical


@pytest.mark.parametrize(
    "layer_idx,branch",
    [(0, "linear_attn.in_proj_qkv.weight"), (3, "self_attn.q_proj.weight")],
)
def test_checkpoint_has_complete_layer_kinds(layer_idx, branch):
    if not (CHECKPOINT / "model.safetensors.index.json").is_file():
        pytest.skip("Pinned checkpoint index is not installed")
    selected = layer_weight_map(CHECKPOINT / "model.safetensors.index.json", layer_idx)
    prefix = f"model.language_model.layers.{layer_idx}."
    required = {
        prefix + "input_layernorm.weight",
        prefix + "post_attention_layernorm.weight",
        prefix + branch,
        prefix + "mlp.gate.weight",
        prefix + "mlp.experts.gate_up_proj",
        prefix + "mlp.experts.down_proj",
        prefix + "mlp.shared_expert_gate.weight",
        prefix + "mlp.shared_expert.gate_proj.weight",
        prefix + "mlp.shared_expert.up_proj.weight",
        prefix + "mlp.shared_expert.down_proj.weight",
    }
    assert required <= selected.keys()
    assert layer_shards(CHECKPOINT / "model.safetensors.index.json", layer_idx)


def test_selective_shard_enumeration_ignores_other_layers():
    fake = {
        "model.language_model.layers.0.a": "a.safetensors",
        "model.language_model.layers.0.b": "b.safetensors",
        "model.language_model.layers.1.a": "c.safetensors",
        "lm_head.weight": "d.safetensors",
    }
    assert layer_shards(fake, 0) == ("a.safetensors", "b.safetensors")
