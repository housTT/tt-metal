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


def shard_cfg(width):
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    return ttnn.create_sharded_memory_config(
        shape=(32, width),
        core_grid=core,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


try:
    torch.manual_seed(0)
    cur = 1024
    width = 33
    poison = 8
    total = width + poison
    K = torch.randn(total * bs, nkv, hd)
    V = torch.randn(total * bs, nkv, hd)
    K[width * bs :] = 3e38
    V[width * bs :] = 3e38
    kc = tt(K.bfloat16().reshape(total, bs, nkv, hd).permute(0, 2, 1, 3).contiguous())
    vc = tt(V.bfloat16().reshape(total, bs, nkv, hd).permute(0, 2, 1, 3).contiguous())
    row = torch.arange(width, dtype=torch.int32)
    pt_exact = tt(row.reshape(1, width), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    pt_poison = tt(torch.arange(total, dtype=torch.int32).reshape(1, total), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    q = torch.randn(1, 1, nq, hd).bfloat16()
    qt = ttnn.to_memory_config(tt(q), shard_cfg(hd))
    pos = tt(torch.tensor([cur], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    L = cur + 1
    k = K[:L].bfloat16().float().repeat_interleave(4, 1).transpose(0, 1)
    v = V[:L].bfloat16().float().repeat_interleave(4, 1).transpose(0, 1)
    s = torch.matmul(q[0].float().transpose(0, 1), k.transpose(-1, -2)) * hd**-0.5
    ref = torch.matmul(torch.softmax(s, -1), v).transpose(0, 1).unsqueeze(0)

    def run(pt, kchunk, label):
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            q_chunk_size=32,
            k_chunk_size=kchunk,
            exp_approx_mode=False,
        )
        o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            qt,
            kc,
            vc,
            page_table_tensor=pt,
            cur_pos_tensor=pos,
            scale=f32(hd**-0.5),
            program_config=pc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ot = ttnn.to_torch(o).float()[:, :, :nq]
        print(f"{label}: pcc {comp_pcc(ref, ot, 0.0)[1]:.5f} max {float(ot.abs().max()):.3g}", flush=True)

    run(pt_exact, 128, "A decode exact 33-entry row (1056 positions), cur_pos 1024, k_chunk 128")
    run(pt_poison, 128, "B decode 41-entry row, entries 33.. point at 3e38 blocks (all beyond cur_pos), k_chunk 128")
    run(pt_exact, 128, "C decode exact row again, k_chunk 128")
    run(pt_exact, 32, "D decode exact row, k_chunk 32 (divides 1056)")
    run(pt_poison, 32, "E decode poisoned row, k_chunk 32")
finally:
    ttnn.close_mesh_device(mesh)
