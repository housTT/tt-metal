# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

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
    torch.manual_seed(0)
    q = torch.randn(num_k_heads, k_head_dim)
    k = torch.randn(num_k_heads, k_head_dim)
    v = torch.randn(num_heads, v_head_dim) * 0.3
    beta = torch.rand(num_heads)
    decay = torch.exp(-torch.rand(num_heads) * 0.1)
    state = torch.randn(num_heads, k_head_dim, v_head_dim) * 0.1

    q_expanded = torch.nn.functional.normalize(q, dim=-1).repeat_interleave(head_expand_ratio, dim=0)
    q_expanded *= k_head_dim**-0.5
    k_expanded = torch.nn.functional.normalize(k, dim=-1).repeat_interleave(head_expand_ratio, dim=0)
    expected_state = state * decay[:, None, None]
    residual = v - torch.einsum("hk,hkv->hv", k_expanded, expected_state)
    expected_state += beta[:, None, None] * torch.einsum("hk,hv->hkv", k_expanded, residual)
    expected_output = torch.einsum("hk,hkv->hv", q_expanded, expected_state)

    replicate = ttnn.ReplicateTensorToMesh(mesh_device)

    def to_device(tensor, memory_config=ttnn.L1_MEMORY_CONFIG):
        return ttnn.from_torch(
            tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=replicate,
            memory_config=memory_config,
        )

    outputs = ttnn.experimental.deltanet_decode_full(
        to_device(q.unsqueeze(0)),
        to_device(k.unsqueeze(0)),
        to_device(v.unsqueeze(0)),
        to_device(beta.reshape(1, 1, num_heads)),
        to_device(decay.reshape(1, 1, num_heads)),
        to_device(state.unsqueeze(0), memory_config=ttnn.DRAM_MEMORY_CONFIG),
        num_heads=num_heads,
        num_k_heads=num_k_heads,
        k_head_dim=k_head_dim,
        v_head_dim=v_head_dim,
        head_expand_ratio=head_expand_ratio,
    )

    compose = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    actual_output = ttnn.to_torch(outputs[0], mesh_composer=compose)[0].reshape(num_heads, v_head_dim)
    actual_state = ttnn.to_torch(outputs[1], mesh_composer=compose)[0].reshape(num_heads, k_head_dim, v_head_dim)

    assert _pcc(actual_output, expected_output) > 0.999
    assert _pcc(actual_state, expected_state) > 0.999


@parametrize_mesh_tp()
def test_deltanet_decode_kernel_512_step_continuity(mesh_device, ensure_gc):
    """Gate accumulated BF16-state drift over changing inputs for a serving-length decode."""
    steps = 512
    num_k_heads, num_heads = 4, 12
    k_head_dim = v_head_dim = 128
    head_expand_ratio = num_heads // num_k_heads
    torch.manual_seed(17)

    def bf16(tensor):
        return tensor.to(torch.bfloat16).float()

    q = bf16(torch.randn(steps, num_k_heads, k_head_dim))
    k = bf16(torch.randn(steps, num_k_heads, k_head_dim))
    v = bf16(torch.randn(steps, num_heads, v_head_dim) * 0.2)
    beta = bf16(torch.sigmoid(torch.randn(steps, num_heads)))
    # Model-like near-one decays exercise the accumulation path without rapidly erasing history.
    decay = bf16(torch.exp(-torch.rand(steps, num_heads) * 0.05))
    initial_state = bf16(torch.randn(num_heads, k_head_dim, v_head_dim) * 0.05)
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)

    def to_device(tensor, memory_config):
        return ttnn.from_torch(
            tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=replicate,
            memory_config=memory_config,
        )

    q_all = to_device(q, ttnn.DRAM_MEMORY_CONFIG)
    k_all = to_device(k, ttnn.DRAM_MEMORY_CONFIG)
    v_all = to_device(v, ttnn.DRAM_MEMORY_CONFIG)
    beta_all = to_device(beta.reshape(steps, 1, num_heads), ttnn.DRAM_MEMORY_CONFIG)
    decay_all = to_device(decay.reshape(steps, 1, num_heads), ttnn.DRAM_MEMORY_CONFIG)
    state = to_device(initial_state.unsqueeze(0), ttnn.DRAM_MEMORY_CONFIG)

    state_ref = initial_state.clone()
    expected_output = None
    actual_output = None
    for step in range(steps):
        q_expanded = torch.nn.functional.normalize(q[step], dim=-1).repeat_interleave(head_expand_ratio, dim=0)
        q_expanded = bf16(q_expanded * k_head_dim**-0.5)
        k_expanded = bf16(torch.nn.functional.normalize(k[step], dim=-1).repeat_interleave(head_expand_ratio, dim=0))
        state_ref = state_ref * decay[step, :, None, None]
        residual = v[step] - torch.einsum("hk,hkv->hv", k_expanded, state_ref)
        state_ref += beta[step, :, None, None] * torch.einsum("hk,hv->hkv", k_expanded, residual)
        expected_output = torch.einsum("hk,hkv->hv", q_expanded, state_ref)

        q_step = ttnn.slice(
            q_all,
            (step, 0, 0),
            (step + 1, num_k_heads, k_head_dim),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        k_step = ttnn.slice(
            k_all,
            (step, 0, 0),
            (step + 1, num_k_heads, k_head_dim),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        v_step = ttnn.slice(
            v_all,
            (step, 0, 0),
            (step + 1, num_heads, v_head_dim),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        beta_step = ttnn.slice(
            beta_all,
            (step, 0, 0),
            (step + 1, 1, num_heads),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        decay_step = ttnn.slice(
            decay_all,
            (step, 0, 0),
            (step + 1, 1, num_heads),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        outputs = ttnn.experimental.deltanet_decode_full(
            q_step,
            k_step,
            v_step,
            beta_step,
            decay_step,
            state,
            num_heads=num_heads,
            num_k_heads=num_k_heads,
            k_head_dim=k_head_dim,
            v_head_dim=v_head_dim,
            head_expand_ratio=head_expand_ratio,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(state)
        state = outputs[1]
        if actual_output is not None:
            ttnn.deallocate(actual_output)
        actual_output = outputs[0]
        ttnn.deallocate(q_step)
        ttnn.deallocate(k_step)
        ttnn.deallocate(v_step)
        ttnn.deallocate(beta_step)
        ttnn.deallocate(decay_step)

    compose = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    actual_output_torch = ttnn.to_torch(actual_output, mesh_composer=compose)[0].reshape(num_heads, v_head_dim)
    actual_state_torch = ttnn.to_torch(state, mesh_composer=compose)[0].reshape(num_heads, k_head_dim, v_head_dim)
    output_pcc = _pcc(actual_output_torch, expected_output)
    state_pcc = _pcc(actual_state_torch, state_ref)
    logger.info(f"DeltaNet 512-step continuity: output PCC={output_pcc:.6f}, state PCC={state_pcc:.6f}")
    assert output_pcc > 0.999
    assert state_pcc > 0.999
