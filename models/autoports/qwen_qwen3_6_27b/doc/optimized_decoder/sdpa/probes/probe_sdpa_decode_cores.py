"""Model-free sweep of ``paged_scaled_dot_product_attention_decode`` vs a float32 golden,
parameterised by ``max_cores_per_head_batch``.

Derived from ``doc/functional_decoder/probes/probe_sdpa_decode_synthetic.py`` (same shapes:
24 q heads / 4 kv heads / head_dim 256 / page 64, shuffled page table).  Adds:

* MAXCORES sweep in one process (one mesh open, one K/V upload) so a core-count x position
  table is cheap;
* the derived work split printed next to every row - ``nkc`` (num_k_chunks), ``cpc``
  (chunks per core) and ``rem`` (residual) - because the recorded defect is stated as a
  divisibility condition on those numbers;
* the ``alpha`` metric (device / float32-golden scale) is kept: the defect is a pure scale
  error and PCC is blind to it.

Env:
  POSITIONS  comma separated decode positions (default the 8 the functional logs use)
  CACHE      cache length in tokens (default 262144)
  KCHUNK     decode k_chunk_size; 0 = let the op choose (no program config at all)
  MAXCORES   comma separated max_cores_per_head_batch values (default 1)
  GRID       "x,y" compute grid (default 8,8)
  QCHUNK     q_chunk_size (default 32)
  FP32DEST   0/1 fp32_dest_acc_en (default 1)
  EXPAPPROX  0/1
  TAG        free-form label printed on every row
"""

import os
import time

import torch

import ttnn

N_Q_HEADS = 24
N_KV_HEADS = 4
HEAD_DIM = 256
BLOCK = 64

CACHE = int(os.environ.get("CACHE", "262144"))
POSITIONS = [int(x) for x in os.environ.get("POSITIONS", "1023,4095,12287,16383,65535,131071,261887,262143").split(",")]
KCHUNK = int(os.environ.get("KCHUNK", "0"))
MAXCORES = [int(x) for x in os.environ.get("MAXCORES", "1").split(",")]
FP32DEST = os.environ.get("FP32DEST", "1") == "1"
TAG = os.environ.get("TAG", "")
SCALE = HEAD_DIM**-0.5


def pcc(a, b):
    a = a.detach().to(torch.float64).flatten()
    b = b.detach().to(torch.float64).flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def split(nkc, ncores):
    """Mirror of ``get_workload_for_core`` (rt_args_common.hpp) chunk counts per core."""
    if ncores > nkc:
        return [1 if c < nkc else 0 for c in range(ncores)]
    cpc = nkc // ncores
    rem = nkc % ncores
    out = []
    for c in range(ncores):
        rev = ncores - c - 1
        out.append(cpc + (1 if rev < rem else 0))
    return out


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

    # goldens (host, float32) - computed once per position and reused across MAXCORES
    golds = {}
    qs = {}
    for pos in POSITIONS:
        q = torch.randn(1, 1, N_Q_HEADS, HEAD_DIM, generator=torch.Generator().manual_seed(pos))
        q = q.to(torch.bfloat16).to(torch.float32)
        qs[pos] = q
        n = pos + 1
        gold = torch.empty(N_Q_HEADS, HEAD_DIM)
        for h in range(N_Q_HEADS):
            kvh = h // (N_Q_HEADS // N_KV_HEADS)
            scores = (q[0, 0, h] @ k[kvh, :n].t()) * SCALE
            gold[h] = torch.softmax(scores, dim=-1) @ v[kvh, :n]
        golds[pos] = gold

    gx, gy = (int(x) for x in os.environ.get("GRID", "8,8").split(","))
    for mc in MAXCORES:
        prog = None
        if KCHUNK:
            prog = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                q_chunk_size=int(os.environ.get("QCHUNK", "32")),
                k_chunk_size=KCHUNK,
                exp_approx_mode=os.environ.get("EXPAPPROX", "0") == "1",
                max_cores_per_head_batch=mc,
            )
        # cores actually used per head: min(mc, grid/(B*kv_heads)) with B=1
        ncores = max(1, min(gx * gy, mc * N_KV_HEADS) // N_KV_HEADS)
        for pos in POSITIONS:
            tt_q = dev(qs[pos])
            tt_pos = dev(torch.tensor([pos], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            kw = {"program_config": prog} if prog is not None else {}
            try:
                t0 = time.perf_counter()
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
                dt = time.perf_counter() - t0
                ttnn.deallocate(out)
            except Exception as e:  # noqa: BLE001
                print(f"RESULT maxcores={mc} pos={pos} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
                ttnn.deallocate(tt_q)
                ttnn.deallocate(tt_pos)
                continue
            ttnn.deallocate(tt_q)
            ttnn.deallocate(tt_pos)

            kc = KCHUNK if KCHUNK else 0
            if kc:
                nkc = -(-(pos + 1) // kc)
                counts = split(nkc, ncores)
                desc = f"nkc={nkc} cpc={nkc // ncores} rem={nkc % ncores} counts={counts[:4]}{'..' if ncores > 4 else ''}"
            else:
                nkc = -1
                desc = "nkc=dynamic"
            gold = golds[pos]
            a = gold.double()
            b = got.double()
            alpha = float((a * b).sum() / (a * a).sum())
            print(
                f"RESULT tag={TAG} maxcores={mc} ncores={ncores} kchunk={kc or 'default'} pos={pos} "
                f"{desc} pcc={pcc(gold, got):.6f} alpha={alpha:.5f} "
                f"rel_err={float((b - a).norm() / a.norm()):.5f} wall={dt * 1e3:.1f}ms",
                flush=True,
            )

    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
