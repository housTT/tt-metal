"""Model-free sweep of ``paged_scaled_dot_product_attention_decode`` against a float32 golden.

Same shapes as the layer (24 q heads / 4 kv heads / head_dim 256 / page 64) and a shuffled page
table, sweeping the decode position over a cache big enough for the model's full advertised
context.  Reports both PCC and the scale factor ``alpha`` (device / golden), because a softmax
denominator defect is a pure scale and PCC cannot see it.

Env: POSITIONS (comma separated), CACHE (blocks*64 tokens, default 262144), KCHUNK (optional
decode k_chunk_size), FP32DEST (0/1).
"""

import os

import torch

import ttnn

N_Q_HEADS = 24
N_KV_HEADS = 4
HEAD_DIM = 256
BLOCK = 64

CACHE = int(os.environ.get("CACHE", "262144"))
POSITIONS = [int(x) for x in os.environ.get("POSITIONS", "4095,16383,65535,131071,262143").split(",")]
KCHUNK = int(os.environ.get("KCHUNK", "0"))
FP32DEST = os.environ.get("FP32DEST", "1") == "1"
SCALE = HEAD_DIM**-0.5


def pcc(a, b):
    a = a.detach().to(torch.float64).flatten()
    b = b.detach().to(torch.float64).flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main() -> None:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=FP32DEST,
        packer_l1_acc=True,
    )
    nb = CACHE // BLOCK
    gen = torch.Generator().manual_seed(99)
    perm = torch.randperm(nb, generator=gen)
    k = torch.randn(N_KV_HEADS, CACHE, HEAD_DIM, generator=gen).to(torch.bfloat16).to(torch.float32)
    v = torch.randn(N_KV_HEADS, CACHE, HEAD_DIM, generator=gen).to(torch.bfloat16).to(torch.float32)
    # A real cache is zero beyond what prefill wrote; ZEROTAIL reproduces that so a kernel that
    # reads past cur_pos shows up as a mild error instead of as garbage.
    if os.environ.get("ZEROTAIL"):
        valid = max(POSITIONS) + 1
        k[:, valid:] = 0.0
        v[:, valid:] = 0.0

    def paged(t):
        blocks = t.reshape(N_KV_HEADS, nb, BLOCK, HEAD_DIM).permute(1, 0, 2, 3).contiguous()
        out = torch.empty_like(blocks)
        out[perm] = blocks
        return out

    def dev(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    tt_k = dev(paged(k))
    tt_v = dev(paged(v))
    tt_pt = dev(perm.to(torch.int32).reshape(1, nb), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

    prog = None
    if KCHUNK:
        gx, gy = (int(x) for x in os.environ.get("GRID", "8,8").split(","))
        prog = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            q_chunk_size=int(os.environ.get("QCHUNK", "32")),
            k_chunk_size=KCHUNK,
            exp_approx_mode=os.environ.get("EXPAPPROX", "0") == "1",
            max_cores_per_head_batch=int(os.environ.get("MAXCORES", "16")),
        )

    for pos in POSITIONS:
        q = torch.randn(1, 1, N_Q_HEADS, HEAD_DIM, generator=torch.Generator().manual_seed(pos))
        q = q.to(torch.bfloat16).to(torch.float32)
        tt_q = dev(q)
        tt_pos = dev(torch.tensor([pos], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        kw = {"program_config": prog} if prog is not None else {}
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            tt_q,
            tt_k,
            tt_v,
            tt_pt,
            cur_pos_tensor=tt_pos,
            scale=SCALE,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=cfg,
            **kw,
        )
        got = ttnn.to_torch(out).to(torch.float32).reshape(N_Q_HEADS, HEAD_DIM)
        ttnn.deallocate(out)
        ttnn.deallocate(tt_q)
        ttnn.deallocate(tt_pos)

        n = pos + 1
        gold = torch.empty(N_Q_HEADS, HEAD_DIM)
        for h in range(N_Q_HEADS):
            kvh = h // (N_Q_HEADS // N_KV_HEADS)
            scores = (q[0, 0, h] @ k[kvh, :n].t()) * SCALE
            gold[h] = torch.softmax(scores, dim=-1) @ v[kvh, :n]
        a = gold.double()
        b = got.double()
        alpha = float((a * b).sum() / (a * a).sum())
        print(
            f"RESULT pos={pos} cache={CACHE} kchunk={KCHUNK or 'default'} fp32dest={FP32DEST} "
            f"pcc={pcc(gold, got):.6f} alpha={alpha:.5f} "
            f"rel_err={float((b - a).norm() / a.norm()):.5f}",
            flush=True,
        )

    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
