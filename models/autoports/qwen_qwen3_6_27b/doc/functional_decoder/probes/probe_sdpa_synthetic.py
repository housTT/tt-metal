"""Minimal, model-free reproducer for the long-context ``chunked_scaled_dot_product_attention``
accuracy cliff.

Synthetic Q/K/V only - no weights, no RoPE, no norms, no projections.  Q/K/V are rounded to
bfloat16 on the host *before* the golden is computed, so the torch golden and the device see
bit-identical inputs and the only difference left is the kernel's arithmetic.

Sweeps the total KV length with a fixed 2048-token query chunk at the end of the context and
reports the PCC of the last 256 query rows against a float32 torch attention.

Env knobs:
  LENGTHS   comma-separated kv lengths (default 32768,65536,131072,262144)
  Q_CHUNK   SDPA q chunk size (default 256)
  K_CHUNK   SDPA k chunk size (default 256)
  FP32DEST  0/1 (default 1)
  PACKL1    packer_l1_acc 0/1 (default 1)
  EXPAPPROX SDPAProgramConfig.exp_approx_mode 0/1 (default 0)
  SHUFFLE   0/1 shuffle the page table (default 1)
  SCALE     score scale, default head_dim**-0.5
"""

import os

import torch

import ttnn

N_Q_HEADS = 24
N_KV_HEADS = 4
HEAD_DIM = 256
BLOCK = 64
QLEN = 2048
TAIL = 256

LENGTHS = [int(x) for x in os.environ.get("LENGTHS", "32768,65536,131072,262144").split(",")]
Q_CHUNK = int(os.environ.get("Q_CHUNK", "256"))
K_CHUNK = int(os.environ.get("K_CHUNK", "256"))
FP32DEST = os.environ.get("FP32DEST", "1") == "1"
PACKL1 = os.environ.get("PACKL1", "1") == "1"
EXPAPPROX = os.environ.get("EXPAPPROX", "0") == "1"
SHUFFLE = os.environ.get("SHUFFLE", "1") == "1"
SCALE = float(os.environ.get("SCALE", HEAD_DIM**-0.5))


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().to(torch.float64).flatten()
    b = b.detach().to(torch.float64).flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def golden_tail(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, n: int) -> torch.Tensor:
    """float32 attention for the last TAIL query rows; returns [1, N_Q_HEADS, TAIL, HEAD_DIM]."""
    start = n - QLEN  # absolute position of query row 0
    out = torch.empty(1, N_Q_HEADS, TAIL, HEAD_DIM, dtype=torch.float32)
    qt = q[0, :, QLEN - TAIL :, :].to(torch.float32)  # [H, TAIL, D]
    positions = torch.arange(start + QLEN - TAIL, start + QLEN)  # absolute pos of each tail row
    for h in range(N_Q_HEADS):
        kvh = h // (N_Q_HEADS // N_KV_HEADS)
        kk = k[0, kvh].to(torch.float32)  # [n, D]
        vv = v[0, kvh].to(torch.float32)
        scores = (qt[h] @ kk.t()) * SCALE  # [TAIL, n]
        keep = torch.arange(n)[None, :] <= positions[:, None]
        scores.masked_fill_(~keep, float("-inf"))
        out[0, h] = torch.softmax(scores, dim=-1) @ vv
        del scores, keep
    return out


def build_paged(t: torch.Tensor, n: int, perm: torch.Tensor) -> torch.Tensor:
    """[1, n_kv, n, D] -> paged cache [num_blocks, n_kv, BLOCK, D] under ``perm``."""
    nb = n // BLOCK
    blocks = t[0].reshape(N_KV_HEADS, nb, BLOCK, HEAD_DIM).permute(1, 0, 2, 3).contiguous()
    cache = torch.empty_like(blocks)
    cache[perm] = blocks  # virtual block i lives at physical block perm[i]
    return cache


def main() -> None:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=FP32DEST,
        packer_l1_acc=PACKL1,
    )
    prog = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=Q_CHUNK,
        k_chunk_size=K_CHUNK,
        exp_approx_mode=EXPAPPROX,
    )
    for n in LENGTHS:
        gen = torch.Generator().manual_seed(1234)
        q = torch.randn(1, N_Q_HEADS, QLEN, HEAD_DIM, generator=gen).to(torch.bfloat16).to(torch.float32)
        k = torch.randn(1, N_KV_HEADS, n, HEAD_DIM, generator=gen).to(torch.bfloat16).to(torch.float32)
        v = torch.randn(1, N_KV_HEADS, n, HEAD_DIM, generator=gen).to(torch.bfloat16).to(torch.float32)

        nb = n // BLOCK
        if SHUFFLE:
            perm = torch.randperm(nb, generator=torch.Generator().manual_seed(7))
        else:
            perm = torch.arange(nb)
        page_table = torch.empty(nb, dtype=torch.int32)
        page_table[:] = perm.to(torch.int32)  # virtual block i -> physical perm[i]

        k_cache = build_paged(k, n, perm)
        v_cache = build_paged(v, n, perm)

        def dev(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        tt_q = dev(q)
        tt_k = dev(k_cache)
        tt_v = dev(v_cache)
        tt_pt = dev(page_table.reshape(1, nb), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        del k_cache, v_cache

        out = ttnn.transformer.chunked_scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            tt_pt,
            n - QLEN,
            scale=SCALE,
            program_config=prog,
            compute_kernel_config=cfg,
        )
        got = ttnn.to_torch(out).to(torch.float32)[:, :, QLEN - TAIL :, :]
        ttnn.deallocate(out)
        ttnn.deallocate(tt_q)
        ttnn.deallocate(tt_k)
        ttnn.deallocate(tt_v)
        ttnn.deallocate(tt_pt)

        gold = golden_tail(q, k, v, n)
        a = gold.double()
        b = got.double()
        alpha = float((a * b).sum() / (a * a).sum())
        rel = float((b - a).norm() / a.norm())
        rel_scaled = float((b - alpha * a).norm() / a.norm())
        print(
            f"RESULT n={n} qchunk={Q_CHUNK} kchunk={K_CHUNK} fp32dest={FP32DEST} "
            f"packl1={PACKL1} expapprox={EXPAPPROX} shuffle={SHUFFLE} "
            f"scale={SCALE:g} tail_pcc={pcc(gold, got):.6f} alpha={alpha:.5f} rel_err={rel:.5f} "
            f"rel_after_scale={rel_scaled:.5f} k_chunks={n // K_CHUNK}",
            flush=True,
        )
        del q, k, v, gold, got

    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
