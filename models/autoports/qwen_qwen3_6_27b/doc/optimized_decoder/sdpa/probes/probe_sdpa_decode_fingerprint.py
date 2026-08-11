"""Identify *which* cross-core combination formula the sdpa_decode tree reduction actually
computes, by fingerprinting the device output against candidate host formulas.

Same shapes as the layer (24 q / 4 kv heads, head_dim 256, page 64).  For a fixed
(k_chunk, cores_per_head, position) it:

  1. reproduces ``get_workload_for_core`` to get each core's k-chunk range;
  2. computes each core's exact local flash state in float64 - the kernel keeps the *unscaled*
     ``q.k`` as the running max and folds ``scale`` into every exp (see
     ``sub_exp_block_bcast_cols_inplace`` / ``correction_block``), so
     ``m_c = max(q.k)``, ``l_c = sum exp((q.k - m_c) * scale)``,
     ``O_c = sum exp((q.k - m_c) * scale) v``;
  3. walks the *actual binary tree* (``get_tree_reduction_params``) and evaluates a family of
     candidate combination formulas, correct and deliberately-broken;
  4. prints rel_err(device, candidate) for each.  The candidate that matches to bf16 noise
     (~2e-2) is what the kernel computes.

Env: KCHUNK, MAXCORES, POSITIONS, CACHE, GRID, QCHUNK, FP32DEST.
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
FP32DEST = os.environ.get("FP32DEST", "1") == "1"

MAX_ROUNDS = 6


def tree_params(core_id, N):
    """Port of get_tree_reduction_params (sdpa_decode_device_operation.hpp)."""
    children = [None] * MAX_ROUNDS
    parent, send_at = None, None
    if N <= 1:
        return True, parent, send_at, children
    vid = (N - 1) - core_id
    num_rounds = N.bit_length() if (N & (N - 1)) else (N - 1).bit_length()
    num_rounds = (N - 1).bit_length()
    is_root = core_id == 0
    for r in range(num_rounds):
        mask = (2 << r) - 1
        if (vid & mask) == mask:
            child_vid = vid - (1 << r)
            if 0 <= child_vid < N:
                children[r] = (N - 1) - child_vid
    if not is_root:
        t = 0
        while (vid >> t) & 1:
            t += 1
        pvid = vid + (1 << t)
        parent = (N - 1) - pvid if pvid < N else 0
        send_at = t
    else:
        for c in range(1, N):
            cv = (N - 1) - c
            t = 0
            while (cv >> t) & 1:
                t += 1
            if cv + (1 << t) >= N and children[t] is None:
                children[t] = c
    return is_root, parent, send_at, children


def workload(nkc, N, core):
    """Port of get_workload_for_core chunk range (non-sliding-window)."""
    if N > nkc:
        cpc = 1 if core < nkc else 0
        return (nkc - core - 1) * cpc, (nkc - core) * cpc
    cpc = nkc // N
    rem = nkc % N
    rev = N - core - 1
    s = rev * cpc + min(rem, rev)
    e = s + cpc + (1 if rev < rem else 0)
    return s, e


def combine(a, b, mode):
    """Combine two flash states a=(m,l,O) and b=(m,l,O). ``mode`` selects the formula."""
    ma, la, Oa = a
    mb, lb, Ob = b
    if mode == "correct":
        M = torch.maximum(ma, mb)
        ea, eb = torch.exp((ma - M) * SCALE), torch.exp((mb - M) * SCALE)
        return M, la * ea + lb * eb, Oa * ea[:, None] + Ob * eb[:, None]
    if mode == "no_scale_in_emd":  # exp(m - M) instead of exp((m - M) * scale)
        M = torch.maximum(ma, mb)
        ea, eb = torch.exp(ma - M), torch.exp(mb - M)
        return M, la * ea + lb * eb, Oa * ea[:, None] + Ob * eb[:, None]
    if mode == "drop_local_l":  # local (parent) l lost from denominator
        M = torch.maximum(ma, mb)
        ea, eb = torch.exp((ma - M) * SCALE), torch.exp((mb - M) * SCALE)
        return M, lb * eb, Oa * ea[:, None] + Ob * eb[:, None]
    if mode == "drop_child_l":
        M = torch.maximum(ma, mb)
        ea, eb = torch.exp((ma - M) * SCALE), torch.exp((mb - M) * SCALE)
        return M, la * ea, Oa * ea[:, None] + Ob * eb[:, None]
    if mode == "l_no_correction":  # l added without the exp(m - M) rescale
        M = torch.maximum(ma, mb)
        ea, eb = torch.exp((ma - M) * SCALE), torch.exp((mb - M) * SCALE)
        return M, la + lb, Oa * ea[:, None] + Ob * eb[:, None]
    if mode == "swapped_emd":  # local O scaled by child's factor and vice versa
        M = torch.maximum(ma, mb)
        ea, eb = torch.exp((ma - M) * SCALE), torch.exp((mb - M) * SCALE)
        return M, la * eb + lb * ea, Oa * eb[:, None] + Ob * ea[:, None]
    if mode == "l_swapped_emd":  # only the denominator gets the swapped factors
        M = torch.maximum(ma, mb)
        ea, eb = torch.exp((ma - M) * SCALE), torch.exp((mb - M) * SCALE)
        return M, la * eb + lb * ea, Oa * ea[:, None] + Ob * eb[:, None]
    if mode == "local_l_is_first_chunk":
        raise NotImplementedError
    raise KeyError(mode)


def local_state(qk, v, s, e, kchunk, n_valid, first_chunk_only=False, last_chunk_only=False):
    """Exact local flash state over k chunks [s, e). qk: (nk,) scores (unscaled), v: (nk, D)."""
    lo, hi = s * kchunk, min(e * kchunk, n_valid)
    if hi <= lo:
        return None
    if first_chunk_only:
        hi = min((s + 1) * kchunk, hi)
    if last_chunk_only:
        lo = max((e - 1) * kchunk, lo)
    sc = qk[lo:hi]
    m = sc.max()
    w = torch.exp((sc - m) * SCALE)
    return m.reshape(1), w.sum().reshape(1), (w[:, None] * v[lo:hi]).sum(0)[None, :]


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

    tt_k, tt_v = dev(paged(k)), dev(paged(v))
    tt_pt = dev(perm.to(torch.int32).reshape(1, nb), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

    gx, gy = (int(x) for x in os.environ.get("GRID", "8,8").split(","))
    prog = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        q_chunk_size=int(os.environ.get("QCHUNK", "32")),
        k_chunk_size=KCHUNK,
        exp_approx_mode=False,
        max_cores_per_head_batch=MAXCORES,
    )
    N = max(1, min(gx * gy, MAXCORES * N_KV_HEADS) // N_KV_HEADS)

    modes = [
        "correct",
        "no_scale_in_emd",
        "drop_local_l",
        "drop_child_l",
        "l_no_correction",
        "swapped_emd",
        "l_swapped_emd",
    ]

    for pos in POSITIONS:
        q = torch.randn(1, 1, N_Q_HEADS, HEAD_DIM, generator=torch.Generator().manual_seed(pos))
        q = q.to(torch.bfloat16).to(torch.float32)
        tt_q = dev(q)
        tt_pos = dev(torch.tensor([pos], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            tt_q, tt_k, tt_v, tt_pt, cur_pos_tensor=tt_pos, scale=SCALE,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=cfg, program_config=prog,
        )
        got = ttnn.to_torch(out).to(torch.float64).reshape(N_Q_HEADS, HEAD_DIM)
        ttnn.deallocate(out)
        ttnn.deallocate(tt_q)
        ttnn.deallocate(tt_pos)

        n = pos + 1
        nkc = -(-n // KCHUNK)
        padded = nkc * KCHUNK
        ranges = [workload(nkc, N, c) for c in range(N)]
        print(f"\n=== pos={pos} kchunk={KCHUNK} N={N} nkc={nkc} ranges={ranges}", flush=True)

        results = {m: torch.empty(N_Q_HEADS, HEAD_DIM, dtype=torch.float64) for m in modes}
        gold = torch.empty(N_Q_HEADS, HEAD_DIM, dtype=torch.float64)
        for h in range(N_Q_HEADS):
            kvh = h // (N_Q_HEADS // N_KV_HEADS)
            qk = (q[0, 0, h].double() @ k[kvh, :padded].double().t())
            vv = v[kvh, :padded].double()
            # keys past cur_pos are masked out
            qk = qk.clone()
            qk[n:] = -float("inf")
            st = [local_state(qk, vv, s, e, KCHUNK, padded) for (s, e) in ranges]
            gsc = qk[:n]
            gw = torch.softmax(gsc * SCALE, dim=-1)
            gold[h] = gw @ vv[:n]
            for mode in modes:
                # walk the tree bottom-up in round order, exactly like the kernel
                acc = list(st)
                for rnd in range(MAX_ROUNDS):
                    for core in range(N):
                        is_root, parent, send_at, children = tree_params(core, N)
                        cid = children[rnd]
                        if cid is None or acc[core] is None or acc[cid] is None:
                            continue
                        acc[core] = combine(acc[core], acc[cid], mode)
                m, l, O = acc[0]
                results[mode][h] = (O / l[:, None])[0]

        def rel(a, b):
            return float((a - b).norm() / b.norm())

        print(f"  device vs golden      rel_err={rel(got, gold):.5f} alpha={float((gold * got).sum() / (gold * gold).sum()):.5f}")
        for mode in modes:
            print(f"  device vs {mode:22s} rel_err={rel(got, results[mode]):.5f}", flush=True)

    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
