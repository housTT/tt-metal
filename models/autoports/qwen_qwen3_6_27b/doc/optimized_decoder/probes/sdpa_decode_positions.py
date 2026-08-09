# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Is there a decode-SDPA program config that is correct at *every* position?

``sdpa_decode_sweep.py`` found explicit ``SDPAProgramConfig`` candidates 4-9x faster than the
op's default, but most of them return garbage at some positions - the defect the functional
stage localised: the kernel's cross-core merge of partial softmax results is only valid when
``num_k_chunks == 1`` or ``num_k_chunks % (2 * cores_per_head) == 0``, and ``num_k_chunks``
depends on the runtime position while the program config is compile-time.

Hypothesis under test: with a grid of at most one core per query head there is no cross-core
k split and therefore no merge, so the defect cannot occur at any position.  24 query heads, so
the candidate grids are 3x8 = 24 cores (exactly one per head) and 4x8 = 32.

Golden is a float32 torch attention over the un-paged cache, not the op's own default - the
default is itself wrong at very long positions (PCC 0.55 at 262143, recorded in
``doc/context_contract.json``), so it cannot serve as a reference here.

Emits ``SDPAPOS `` JSON lines.
"""
from __future__ import annotations

import json
import math
import sys
import time

sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal")

import torch  # noqa: E402
import ttnn  # noqa: E402

N_HEADS, N_KV, HEAD_DIM = 24, 4, 256
BLOCK = 64
PADDED_HEADS = 32
ITERS = 10
MAX_CONTEXT = 262144
POSITIONS = [63, 127, 255, 511, 1023, 2047, 2048, 2049, 4095, 5003, 8191,
             12287, 16383, 32767, 65535, 131071, 261887, 262143]


def emit(**payload):
    print("SDPAPOS " + json.dumps(payload, sort_keys=True, default=str), flush=True)


def pcc(a, b):
    a = a.to(torch.float64).flatten(); a = a - a.mean()
    b = b.to(torch.float64).flatten(); b = b - b.mean()
    denom = a.norm() * b.norm()
    return 1.0 if denom == 0 else float((a @ b) / denom)


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        num_blocks = MAX_CONTEXT // BLOCK
        torch.manual_seed(0)
        k_host = torch.randn(num_blocks, N_KV, BLOCK, HEAD_DIM) * 0.05
        v_host = torch.randn(num_blocks, N_KV, BLOCK, HEAD_DIM) * 0.05
        k_cache = ttnn.from_torch(k_host, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                                  device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v_cache = ttnn.from_torch(v_host, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                                  device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Read the quantised cache back so the golden sees the same values the kernel does.
        k_q = ttnn.to_torch(k_cache).float()
        v_q = ttnn.to_torch(v_cache).float()
        table = torch.randperm(num_blocks).reshape(1, -1).to(torch.int32)
        page_table = ttnn.from_torch(table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                     device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q_host = torch.randn(1, 1, PADDED_HEADS, HEAD_DIM) * 0.05
        q = ttnn.from_torch(q_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)

        candidates = [("default", None)]
        for gx, gy, label in ((3, 8, "grid3x8_24cores"), (4, 8, "grid4x8_32cores"), (8, 8, "grid8x8_64cores")):
            candidates.append((label, ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                q_chunk_size=0, k_chunk_size=0, exp_approx_mode=False)))

        ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True)

        for position in POSITIONS:
            length = position + 1
            needed = math.ceil(length / BLOCK)
            gathered_k = k_q[table[0, :needed].long()].permute(1, 0, 2, 3).reshape(N_KV, needed * BLOCK, HEAD_DIM)[:, :length]
            gathered_v = v_q[table[0, :needed].long()].permute(1, 0, 2, 3).reshape(N_KV, needed * BLOCK, HEAD_DIM)[:, :length]
            # The decode SDPA infers its GQA grouping from the *padded* head count, so with 24
            # real heads padded to 32 and 4 KV heads the group size is 8, not 6: padded slot p
            # attends KV head p // 8.  ``nlp_create_qkv_heads_decode`` places real head r at
            # slot (r // 6) * 8 + (r % 6) so that the two agree.  The golden has to use the
            # padded layout, or it compares a different attention.
            groups = PADDED_HEADS // N_KV
            qh = q_host[0, 0].float()                                # [PADDED_HEADS, HEAD_DIM]
            kk = gathered_k.repeat_interleave(groups, dim=0)         # [PADDED_HEADS, L, D]
            vv = gathered_v.repeat_interleave(groups, dim=0)
            scores = torch.einsum("hd,hld->hl", qh, kk) * (HEAD_DIM ** -0.5)
            golden = torch.einsum("hl,hld->hd", torch.softmax(scores.double(), dim=-1).float(), vv)

            cur_pos = ttnn.from_torch(torch.tensor([position], dtype=torch.int32), dtype=ttnn.int32,
                                      layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                                      memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for label, pc in candidates:
                row = {"position": position, "candidate": label}
                try:
                    def run():
                        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
                            q, k_cache, v_cache, page_table, cur_pos_tensor=cur_pos,
                            scale=HEAD_DIM ** -0.5, program_config=pc,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckc)

                    got = ttnn.to_torch(run()).float().reshape(-1, HEAD_DIM)[:PADDED_HEADS]
                    row["pcc"] = pcc(golden, got)
                    row["scale_ratio"] = float((got.norm() / golden.norm()))
                    ttnn.synchronize_device(device)
                    start = time.perf_counter()
                    for _ in range(ITERS):
                        out = run()
                        ttnn.deallocate(out)
                    ttnn.synchronize_device(device)
                    row["us"] = (time.perf_counter() - start) * 1e6 / ITERS
                except Exception as exc:
                    lines = str(exc).splitlines()
                    row["error"] = lines[2] if len(lines) > 2 else str(exc)
                emit(**row)
            ttnn.deallocate(cur_pos)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
