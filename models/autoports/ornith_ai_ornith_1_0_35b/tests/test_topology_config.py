# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only invariants for the P150, P150x2, and four-chip Ornith layouts."""

import pytest
import torch

from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import OrnithModel
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model_config import OrnithDecoderConfig
from models.autoports.ornith_ai_ornith_1_0_35b.tt.multichip_decoder import (
    _conv1d_host_weights_local,
    _global_to_local_expert_maps,
    kv_head_owner,
    local_decoder_config,
)


def _ornith_config() -> OrnithDecoderConfig:
    """The architecture fields pinned by the validated Ornith weight revision."""

    return OrnithDecoderConfig(
        dim=2048,
        norm_eps=1e-6,
        max_position_embeddings=262144,
        num_hidden_layers=40,
        layer_types=("linear_attention", "linear_attention", "linear_attention", "full_attention") * 10,
        n_heads=16,
        n_kv_heads=2,
        head_dim=256,
        rope_theta=10_000_000.0,
        partial_rotary_factor=0.25,
        mrope_section=(11, 11, 10),
        mrope_interleaved=True,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        hidden_act="silu",
    )


@pytest.mark.parametrize(
    ("tp", "expected"),
    [
        (1, (16, 2, 16, 32, 256, 512, 8192, 4096)),
        (2, (8, 1, 8, 16, 128, 256, 4096, 2048)),
        (4, (4, 1, 4, 8, 64, 128, 2048, 1024)),
    ],
)
def test_local_config_covers_release_topologies(tp, expected):
    cfg = local_decoder_config(_ornith_config(), tp)
    actual = (
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.linear_num_key_heads,
        cfg.linear_num_value_heads,
        cfg.num_experts,
        cfg.shared_expert_intermediate_size,
        cfg.conv_dim,
        cfg.linear_v_dim,
    )
    assert actual == expected
    assert cfg.n_heads % cfg.n_kv_heads == 0


@pytest.mark.parametrize(
    ("tp", "owners"),
    [(1, [0]), (2, [0, 1]), (4, [0, 0, 1, 1])],
)
def test_kv_head_ownership_preserves_gqa_groups(tp, owners):
    assert [kv_head_owner(2, tp, device) for device in range(tp)] == owners


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_expert_partition_is_disjoint_and_complete_for_release_topologies(tp):
    local_experts = 256 // tp
    mappings = _global_to_local_expert_maps(256, local_experts, tp)
    for device, mapping in enumerate(mappings):
        lo = device * local_experts
        hi = lo + local_experts
        assert torch.equal(mapping[lo:hi], torch.arange(local_experts, dtype=torch.int32))
        assert torch.count_nonzero(mapping == -1).item() == 256 - local_experts


@pytest.mark.parametrize(("tp", "blocks"), [(1, 8), (2, 4), (4, 2)])
def test_conv_weights_split_into_supported_1024_channel_calls(tp, blocks):
    cfg = local_decoder_config(_ornith_config(), tp)
    global_conv = torch.arange(8192 * 4, dtype=torch.float32).reshape(8192, 1, 4)
    split = _conv1d_host_weights_local(global_conv, cfg, tp, channels=1024)
    assert len(split) == blocks
    assert all(len(per_device) == tp for per_device in split)
    assert all(tuple(weight.shape) == (1024, 1, 1, 4) for per_device in split for weight in per_device)


def test_single_device_sampling_keeps_the_supported_two_half_reduction():
    model = object.__new__(OrnithModel)
    model.tp = 1
    model.padded_vocab_size = 249856
    assert model.best_topk_groups(max_top_k=32) == 1
