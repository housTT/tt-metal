# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@parametrize_mesh_tp()
def test_deltanet_decode_kernel_matches_torch(mesh_device, ensure_gc):
    """Validate the fused single-token recurrence and raw output contract."""
    num_k_heads, num_heads = 4, 12
    k_head_dim = v_head_dim = 128
    head_expand_ratio = num_heads // num_k_heads
    key_dim = num_k_heads * k_head_dim
    value_dim = num_heads * v_head_dim
    conv_dim = 2 * key_dim + value_dim

    torch.manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(num_k_heads, k_head_dim), dim=-1) * k_head_dim**-0.5
    k = torch.nn.functional.normalize(torch.randn(num_k_heads, k_head_dim), dim=-1)
    v = torch.randn(num_heads, v_head_dim) * 0.3
    beta = torch.rand(num_heads)
    decay = torch.exp(-torch.rand(num_heads) * 0.1)
    state = torch.randn(num_heads, k_head_dim, v_head_dim) * 0.1

    q_expanded = q.repeat_interleave(head_expand_ratio, dim=0)
    k_expanded = k.repeat_interleave(head_expand_ratio, dim=0)
    expected_state = state * decay[:, None, None]
    residual = v - torch.einsum("hk,hkv->hv", k_expanded, expected_state)
    expected_state += beta[:, None, None] * torch.einsum("hk,hv->hkv", k_expanded, residual)
    expected_output = torch.einsum("hk,hkv->hv", q_expanded, expected_state)

    replicate = ttnn.ReplicateTensorToMesh(mesh_device)

    def to_device(tensor):
        return ttnn.from_torch(
            tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=replicate,
        )

    qkv = torch.cat((q.flatten(), k.flatten(), v.flatten())).reshape(1, 1, 1, conv_dim)
    dummy_conv = to_device(torch.zeros(1, 1, conv_dim, 32))
    outputs = ttnn.experimental.deltanet_decode_full(
        to_device(qkv),
        to_device(torch.zeros(1, 1, 1, value_dim)),
        to_device(beta.reshape(1, 1, 1, num_heads)),
        to_device(decay.reshape(1, 1, 1, num_heads)),
        dummy_conv,
        to_device(state.unsqueeze(0)),
        dummy_conv,
        to_device(torch.zeros(1, 1, 1, num_heads)),
        to_device(torch.zeros(1, 1, 1, num_heads)),
        to_device(torch.ones(1, 1, 1, v_head_dim)),
        num_heads=num_heads,
        num_k_heads=num_k_heads,
        k_head_dim=k_head_dim,
        v_head_dim=v_head_dim,
        conv_dim=conv_dim,
        conv_kernel_size=4,
        head_expand_ratio=head_expand_ratio,
    )

    compose = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    actual_output = ttnn.to_torch(outputs[0], mesh_composer=compose)[0].reshape(num_heads, v_head_dim)
    actual_state = ttnn.to_torch(outputs[1], mesh_composer=compose)[0].reshape(num_heads, k_head_dim, v_head_dim)

    assert _pcc(actual_output, expected_output) > 0.999
    assert _pcc(actual_state, expected_state) > 0.999
