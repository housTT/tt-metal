# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Stage 3: KV-cache prefill/decode attention, heads sharded across the 4 chips (resident).

Demonstrates the decode path that stops recomputing the whole sequence per token:
  prefill(prompt) computes K,V for all S tokens and KEEPS them resident (the cache);
  decode(new token) computes only its q,k,v, appends k,v to the resident cache, and attends
  over the cache — reusing the prompt's K,V instead of recomputing them.
Heads are sharded 1-per-chip (attention is per-head independent), weights resident. Validated:
decode output == full attention over the extended sequence (last position). Also captures the
decode step in a Metal Trace (all three techniques together)."""
import torch

import ttnn
from models.common.utility_functions import comp_pcc

torch.manual_seed(0)


def main():
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=90_000_000)
    try:
        C = mesh.get_num_devices()
        nh, hd = C, 128  # 1 head per chip
        H = nh * hd
        S = 32  # prompt length
        scale = hd**-0.5
        X = torch.randn(S, H, dtype=torch.bfloat16)  # prompt
        x1 = torch.randn(1, H, dtype=torch.bfloat16)  # the new (decode) token
        wq = torch.randn(H, H, dtype=torch.bfloat16) * 0.05
        wk = torch.randn(H, H, dtype=torch.bfloat16) * 0.05
        wv = torch.randn(H, H, dtype=torch.bfloat16) * 0.05
        wo = torch.randn(H, H, dtype=torch.bfloat16) * 0.05

        # ---- torch reference: full causal attention over [prompt + new], last position ----
        def heads(t):
            return t.view(-1, nh, hd).transpose(0, 1)  # [nh, L, hd]

        Xa = torch.cat([X, x1], 0).float()
        Q, K, V = heads(Xa @ wq.float()), heads(Xa @ wk.float()), heads(Xa @ wv.float())
        aw = (Q @ K.transpose(-1, -2)) * scale  # [nh, L, L]
        L = Xa.shape[0]
        aw = aw + torch.triu(torch.full((L, L), float("-inf")), 1)
        out = (torch.softmax(aw, -1) @ V).transpose(0, 1).reshape(L, H)  # [L, H]
        ref_last = out[-1:].to(torch.bfloat16).float() @ wo.float()  # [1, H]

        # ---- TT: heads sharded 1/chip; weights resident ----
        col = ttnn.ShardTensorToMesh(mesh, dim=-1)  # shard output (head) dim -> 1 head/chip
        row = ttnn.ShardTensorToMesh(mesh, dim=0)  # wo row-parallel (per head), sum partials
        rep = ttnn.ReplicateTensorToMesh(mesh)
        mk = lambda t, m: ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=m,
        )
        tq, tk, tv, two = mk(wq, col), mk(wk, col), mk(wv, col), mk(wo, row)

        def proj(x_dev, w):
            return ttnn.matmul(x_dev, w)  # [.,hd] per chip (1 head)

        # ---- prefill: compute + CACHE K,V for the prompt (resident, not recomputed at decode) ----
        tX = mk(X, rep)
        cacheK = proj(tX, tk)  # [S, hd] per chip  <-- the KV cache
        cacheV = proj(tX, tv)

        # ---- decode: only the new token's q,k,v; reuse the resident cache ----
        # tx1 is a RESIDENT input (allocated once); decode_step is pure on-device (no host I/O)
        # so it is Metal-Trace-capturable.
        tx1 = mk(x1, rep)

        def decode_step():
            q1 = proj(tx1, tq)  # [1, hd] per chip
            k1 = proj(tx1, tk)
            v1 = proj(tx1, tv)
            K = ttnn.concat([cacheK, k1], dim=0)  # [S+1, hd]  (append to resident cache)
            V = ttnn.concat([cacheV, v1], dim=0)
            aw = ttnn.multiply(ttnn.matmul(q1, ttnn.transpose(K, -2, -1)), scale)  # [1, S+1]
            p = ttnn.softmax(aw, dim=-1)
            ctx = ttnn.matmul(p, V)  # [1, hd] per chip
            return ttnn.matmul(ctx, two)  # [1, H] partial per chip (row-parallel)

        o = decode_step()
        parts = ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))  # [C, H] stacked
        got = parts.reshape(C, 1, H).sum(0)  # sum row-parallel partials -> [1, H]
        _, pcc = comp_pcc(ref_last, got, 0.99)
        print(
            f"[kv] decode (heads sharded {C} chips, KV cache reused): PCC vs full-attn last token = {float(pcc):.5f}",
            flush=True,
        )

        # ---- capture the decode step in a Metal Trace (all 3 techniques together) ----
        decode_step()  # warm program cache
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        o2 = decode_step()  # pure on-device -> capturable
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        got2 = ttnn.to_torch(o2, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).reshape(C, 1, H).sum(0)
        _, pcc2 = comp_pcc(ref_last, got2, 0.99)
        print(f"[kv] decode step captured + replayed via Metal Trace: PCC={float(pcc2):.5f}", flush=True)
        ttnn.release_trace(mesh, tid)
        print("KV_RESULT", "PASS" if pcc >= 0.99 and pcc2 >= 0.99 else "FAIL", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
