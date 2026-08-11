"""Solve for what the sdpa_decode tree reduction actually produced, instead of guessing formulas.

For a fixed (k_chunk, cores_per_head, position) with cores_per_head == 2 (root + one child):

  * compute both cores' exact local flash states (m, l, O_unnormalised) in float64, using the
    kernel's convention that the running max is the *unscaled* ``q.k`` and ``scale`` is folded
    into every exp;
  * check how well the device output aligns (cosine) with each candidate *direction*
    (O_root, O_child, O_root+O_child with/without max correction);
  * for the best-aligned direction, solve the least-squares scalar the device applied
    (``d_est``) and print it next to every candidate denominator (L, l_root, l_child,
    exp_max_diff, ...) as a ratio.

Env: KCHUNK, MAXCORES (must be 2 for the per-core breakdown), POSITIONS, CACHE, GRID, QCHUNK.
"""

import os

import torch

import ttnn

N_Q_HEADS = 24
N_KV_HEADS = 4
HEAD_DIM = 256
BLOCK = 64
SCALE = HEAD_DIM**-0.5

CACHE = int(os.environ.get("CACHE", "8192"))
POSITIONS = [int(x) for x in os.environ.get("POSITIONS", "1023").split(",")]
KCHUNK = int(os.environ.get("KCHUNK", "512"))
MAXCORES = int(os.environ.get("MAXCORES", "2"))
HEADS = [int(x) for x in os.environ.get("HEADS", "0,1,6,12").split(",")]


def workload(nkc, N, core):
    if N > nkc:
        cpc = 1 if core < nkc else 0
        return (nkc - core - 1) * cpc, (nkc - core) * cpc
    cpc, rem = nkc // N, nkc % N
    rev = N - core - 1
    s = rev * cpc + min(rem, rev)
    return s, s + cpc + (1 if rev < rem else 0)


def main() -> None:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=os.environ.get("FP32DEST", "1") == "1", packer_l1_acc=True,
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

    tt_k, tt_v = dev(paged(k)), dev(paged(v))
    tt_pt = dev(perm.to(torch.int32).reshape(1, nb), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    gx, gy = (int(x) for x in os.environ.get("GRID", "8,8").split(","))
    prog = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        q_chunk_size=int(os.environ.get("QCHUNK", "32")),
        k_chunk_size=KCHUNK, exp_approx_mode=False, max_cores_per_head_batch=MAXCORES,
    )
    N = max(1, min(gx * gy, MAXCORES * N_KV_HEADS) // N_KV_HEADS)

    for pos in POSITIONS:
        q = torch.randn(1, 1, N_Q_HEADS, HEAD_DIM, generator=torch.Generator().manual_seed(pos))
        q = q.to(torch.bfloat16).to(torch.float32)
        tt_q = dev(q)
        tt_pos = dev(torch.tensor([pos], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            tt_q, tt_k, tt_v, tt_pt, cur_pos_tensor=tt_pos, scale=SCALE,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=cfg, program_config=prog)
        got = ttnn.to_torch(out).to(torch.float64).reshape(N_Q_HEADS, HEAD_DIM)
        ttnn.deallocate(out); ttnn.deallocate(tt_q); ttnn.deallocate(tt_pos)

        n = pos + 1
        nkc = -(-n // KCHUNK)
        padded = nkc * KCHUNK
        ranges = [workload(nkc, N, c) for c in range(N)]
        print(f"\n=== pos={pos} kchunk={KCHUNK} N={N} nkc={nkc} ranges={ranges}", flush=True)

        for h in HEADS:
            kvh = h // (N_Q_HEADS // N_KV_HEADS)
            qk = (q[0, 0, h].double() @ k[kvh, :padded].double().t()).clone()
            qk[n:] = -float("inf")
            vv = v[kvh, :padded].double()
            states = []
            for (s, e) in ranges:
                lo, hi = s * KCHUNK, min(e * KCHUNK, n)
                if hi <= lo:
                    states.append(None)
                    continue
                sc = qk[lo:hi]
                m = float(sc.max())
                w = torch.exp((sc - m) * SCALE)
                states.append((m, float(w.sum()), (w[:, None] * vv[lo:hi]).sum(0)))
            m0, l0, O0 = states[0]
            m1, l1, O1 = states[1]
            M = max(m0, m1)
            e0, e1 = torch.exp(torch.tensor((m0 - M) * SCALE)).item(), torch.exp(torch.tensor((m1 - M) * SCALE)).item()
            L = l0 * e0 + l1 * e1
            Ocomb = O0 * e0 + O1 * e1
            d = got[h]
            cands = {
                "O_root+O_child(corrected)": Ocomb,
                "O_root(corrected)": O0 * e0,
                "O_child(corrected)": O1 * e1,
                "O_root+O_child(raw)": O0 + O1,
                "O_root(raw)": O0,
                "O_child(raw)": O1,
            }
            best, bestcos = None, -2.0
            for name, vecc in cands.items():
                c = float((vecc @ d) / (vecc.norm() * d.norm()))
                if c > bestcos:
                    best, bestcos = name, c
            num = cands[best]
            d_est = float((num @ d) / (d @ d))  # device = num / d_est
            print(
                f"  h={h:2d} m_root={m0:8.3f} m_child={m1:8.3f} l_root={l0:10.4f} l_child={l1:10.4f} "
                f"e_root={e0:.5f} e_child={e1:.5f} L={L:10.4f}",
                flush=True)
            allcos = " ".join(f"{nm.split('(')[0]}{'c' if 'corrected' in nm else 'r'}={float((vc @ d) / (vc.norm() * d.norm())):+.4f}" for nm, vc in cands.items())
            print(f"       cos: {allcos}", flush=True)
            print(
                f"       best={best} cos={bestcos:.5f} d_est={d_est:.5f} "
                f"| d_est/L={d_est / L:.5f} d_est/l_root={d_est / l0:.5f} d_est/l_child={d_est / l1:.5f} "
                f"d_est/e_root={d_est / e0:.5f} d_est/e_child={d_est / e1:.5f}",
                flush=True)

    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
