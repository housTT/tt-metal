"""Deterministic reproducer: chunked SDPA reads page-table entries beyond the row when
round_up(chunk_start + Sq, k_chunk) exceeds the page-table extent; stale L1 from a previous
program then supplies the block ids."""

import struct

import torch

import ttnn
from models.common.utility_functions import comp_pcc

f32 = lambda x: struct.unpack("f", struct.pack("f", x))[0]
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
nq, nkv, hd, bs = 32, 8, 128, 32
grid = mesh.compute_with_storage_grid_size()


def tt(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(x, dtype=dtype, layout=layout, device=mesh)


def torch_ref(q, K, V, start):
    Sq = q.shape[2]
    L = start + Sq
    k = K[:L].repeat_interleave(nq // nkv, dim=1).transpose(0, 1).float()
    v = V[:L].repeat_interleave(nq // nkv, dim=1).transpose(0, 1).float()
    s = torch.matmul(q[0].float(), k.transpose(-1, -2)) * hd**-0.5
    s = s.masked_fill(~torch.ones(Sq, L, dtype=torch.bool).tril(diagonal=start), float("-inf"))
    return torch.matmul(torch.softmax(s, -1), v).unsqueeze(0)


try:
    torch.manual_seed(0)
    Sq, start, chunk, width = 64, 8192, 256, 258
    valid_blocks = width
    poison_blocks = 8
    total_blocks = valid_blocks + poison_blocks
    K = torch.randn(total_blocks * bs, nkv, hd)
    V = torch.randn(total_blocks * bs, nkv, hd)
    K[valid_blocks * bs :] = 3e38
    V[valid_blocks * bs :] = 3e38
    kc = tt(K.bfloat16().reshape(total_blocks, bs, nkv, hd).permute(0, 2, 1, 3).contiguous())
    vc = tt(V.bfloat16().reshape(total_blocks, bs, nkv, hd).permute(0, 2, 1, 3).contiguous())
    row = torch.arange(valid_blocks, dtype=torch.int32)
    poisoned_row = torch.cat([row, torch.arange(valid_blocks, total_blocks, dtype=torch.int32)]).reshape(
        1, total_blocks
    )
    pt_poison = tt(poisoned_row, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    pt_exact = tt(row.reshape(1, width), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    q = torch.randn(1, nq, Sq, hd).bfloat16()
    qt = tt(q)
    ref = torch_ref(q, K.bfloat16(), V.bfloat16(), start)

    def run(pt, c, label):
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=c, k_chunk_size=c, exp_approx_mode=False
        )
        o = ttnn.transformer.chunked_scaled_dot_product_attention(
            qt, kc, vc, pt, chunk_start_idx=start, scale=f32(hd**-0.5), program_config=pc
        )
        ot = ttnn.to_torch(o).float()
        print(f"{label}: pcc {comp_pcc(ref, ot, 0.0)[1]:.5f} max {float(ot.abs().max()):.3g}", flush=True)

    run(pt_exact, chunk, "A exact 258-entry row, chunk 256, fresh process")
    run(
        pt_poison,
        chunk,
        "B 266-entry row whose entries 258..265 point at 3e38 blocks (all masked by causality), chunk 256",
    )
    run(pt_exact, chunk, "C exact 258-entry row again, chunk 256 (stale L1 entries from B)")
    run(pt_exact, 64, "D exact 258-entry row, chunk 64 (padded extent 8256 fits)")
    run(pt_exact, chunk, "E exact 258-entry row, chunk 256 again")
finally:
    ttnn.close_mesh_device(mesh)
