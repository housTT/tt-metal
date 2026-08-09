# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Second probe round: accuracy/perf of the fused candidates at the real layer's shapes.

* ``gdn``     - ``gated_delta_attn_seq`` accuracy vs HF at the real head geometry, and the wall
                time of the whole chunked delta rule for one 2048-token prefill chunk.
* ``mm``      - is the 1-core batched 32x32 matmul any faster out of L1 / at other ranks?
* ``sharded`` - do ``rms_norm`` and ``rotary_embedding_hf(is_decode_mode=True)`` accept the
                height-sharded tensors ``nlp_create_qkv_heads_decode`` produces?
"""

from __future__ import annotations

import sys
import time

import torch
import ttnn

from models.autoports.qwen_qwen3_6_27b.doc.fused_decoder.probes.probe_fused_ops import (  # noqa: E402
    expand_rope_mats,
    pcc,
    rope_permutation,
    torch_partial_rope,
    tt,
)

NV, DK, DV = 48, 128, 128  # real Qwen3.6-27B gated-delta-net geometry


def probe_gdn(device):
    from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq import (
        chunk_gated_delta_rule_seq_adapter,
        create_chunk_masks_seq,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    torch.manual_seed(0)
    seq = 2048
    q = torch.randn(1, seq, NV, DK)
    k = torch.randn(1, seq, NV, DK)
    v = torch.randn(1, seq, NV, DV)
    beta = torch.rand(1, seq, NV)
    g = -torch.rand(1, seq, NV) * 0.05
    state0 = torch.randn(1, NV, DK, DV) * 0.1

    golden, golden_state = torch_chunk_gated_delta_rule(
        q, k, v, g, beta, chunk_size=64, initial_state=state0.clone(),
        output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    masks = create_chunk_masks_seq(128, device)
    args = dict(chunk_size=128, device=device, cached_masks=masks)

    def run():
        return chunk_gated_delta_rule_seq_adapter(
            tt(q, device, ttnn.float32), tt(k, device, ttnn.float32), tt(v, device, ttnn.float32),
            tt(beta, device, ttnn.float32), tt(g, device, ttnn.float32),
            initial_state=tt(state0, device, ttnn.float32), **args,
        )

    out, state = run()
    print(f"  seq={seq} out pcc={pcc(golden, ttnn.to_torch(out).float()):.6f} "
          f"state pcc={pcc(golden_state, ttnn.to_torch(state).float()):.6f}")
    ttnn.deallocate(out)
    ttnn.deallocate(state)
    ttnn.synchronize_device(device)
    start = time.perf_counter()
    out, state = run()
    ttnn.synchronize_device(device)
    print(f"  seq={seq} whole chunked delta rule wall={1e3 * (time.perf_counter() - start):.1f} ms")


def probe_mm(device):
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )
    lofi = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=False,
    )
    cases = [
        ("dram f32 hifi4", [768, 32, 32], ttnn.float32, ttnn.DRAM_MEMORY_CONFIG, cfg),
        ("l1   f32 hifi2", [768, 32, 32], ttnn.float32, ttnn.L1_MEMORY_CONFIG, lofi),
        ("dram bf16 lofi", [768, 32, 32], ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, lofi),
        ("dram f32 b3072", [3072, 32, 32], ttnn.float32, ttnn.DRAM_MEMORY_CONFIG, cfg),
        ("dram f32 128sq", [768, 128, 128], ttnn.float32, ttnn.DRAM_MEMORY_CONFIG, cfg),
        ("dram f32 b48 128x2048x128", [48, 2048, 128], ttnn.float32, ttnn.DRAM_MEMORY_CONFIG, cfg),
    ]
    for name, shape, dtype, mem, kernel in cases:
        try:
            a = ttnn.from_torch(torch.randn(*shape), dtype=dtype, layout=ttnn.TILE_LAYOUT,
                                device=device, memory_config=mem)
            b_shape = shape if shape[1] == shape[2] else [shape[0], shape[2], shape[1]]
            b = ttnn.from_torch(torch.randn(*b_shape), dtype=dtype, layout=ttnn.TILE_LAYOUT,
                                device=device, memory_config=mem)
            ttnn.deallocate(ttnn.matmul(a, b, dtype=dtype, compute_kernel_config=kernel, memory_config=mem))
            ttnn.synchronize_device(device)
            start = time.perf_counter()
            for _ in range(5):
                ttnn.deallocate(ttnn.matmul(a, b, dtype=dtype, compute_kernel_config=kernel, memory_config=mem))
            ttnn.synchronize_device(device)
            print(f"  {name} {shape}: {1e3 * (time.perf_counter() - start) / 5:.3f} ms/call")
            ttnn.deallocate(a)
            ttnn.deallocate(b)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name} {shape}: FAILED {type(exc).__name__}: {str(exc)[:160]}")


def probe_sharded(device):
    batch, padded_heads, head_dim = 8, 32, 256
    grid = ttnn.num_cores_to_corerangeset(batch, ttnn.CoreCoord(8, 8), row_wise=True)
    head_cfg = ttnn.create_sharded_memory_config(
        shape=(padded_heads, head_dim), core_grid=grid, strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True,
    )
    rot_cfg = ttnn.create_sharded_memory_config(
        shape=(1, head_dim), core_grid=grid, strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True,
    )
    x = torch.randn(1, batch, padded_heads, head_dim)
    weight = torch.randn(1, 1, 1, head_dim)

    x_sh = ttnn.to_memory_config(tt(x, device), head_cfg)
    try:
        normed = ttnn.rms_norm(x_sh, epsilon=1e-6, weight=tt(weight, device))
        got = ttnn.to_torch(ttnn.to_memory_config(normed, ttnn.DRAM_MEMORY_CONFIG)).float()
        golden = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * weight
        print(f"  rms_norm on height-sharded input: pcc={pcc(golden, got):.6f} "
              f"sharded_out={normed.is_sharded()}")
    except Exception as exc:  # noqa: BLE001
        print(f"  rms_norm on height-sharded input: FAILED {type(exc).__name__}: {str(exc)[:200]}")

    perm = rope_permutation()
    cos = torch.randn(1, batch, 1, 64).clamp(-1, 1)
    sin = torch.randn(1, batch, 1, 64).clamp(-1, 1)
    cos_full, sin_full = expand_rope_mats(cos, sin)
    golden = torch_partial_rope(x, cos, sin)[..., perm]
    try:
        out = ttnn.experimental.rotary_embedding_hf(
            ttnn.to_memory_config(tt(x[..., perm].contiguous(), device), head_cfg),
            ttnn.to_memory_config(tt(cos_full, device), rot_cfg),
            ttnn.to_memory_config(tt(sin_full, device), rot_cfg),
            is_decode_mode=True,
        )
        got = ttnn.to_torch(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)).float()
        print(f"  rotary_embedding_hf decode sharded: pcc={pcc(golden, got):.6f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  rotary_embedding_hf decode sharded: FAILED {type(exc).__name__}: {str(exc)[:300]}")


PROBES = {"gdn": probe_gdn, "mm": probe_mm, "sharded": probe_sharded}


def main():
    names = sys.argv[1:] or list(PROBES)
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        for name in names:
            print(f"== {name}", flush=True)
            PROBES[name](device)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
