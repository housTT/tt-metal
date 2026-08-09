# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The landed head split (F17): two overlapping nlp_create_qkv_heads calls on the *narrow*
conv output vs the reshape/permute/repeat_interleave chain.  Checks bit-identity and time.
log: ../logs/probe_headsplit_narrow.log
"""
import time, torch, ttnn
CFG = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
d = ttnn.open_mesh_device(ttnn.MeshShape(1,1), trace_region_size=0)
def tt(t, dt=ttnn.float32): return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG)
def bench(fn, n=5):
    fn(); ttnn.synchronize_device(d); s=time.perf_counter()
    for _ in range(n): fn()
    ttnn.synchronize_device(d); return 1e3*(time.perf_counter()-s)/n
nk, nv, dk = 16, 48, 128
for L in (2048,):
    conv = tt(torch.randn(1,1,L,2*nk*dk+nv*dk))   # [q 2048 | k 2048 | v 6144] fp32
    ref_q = ttnn.to_torch(conv).float()
    def current():
        outs=[]
        for start, n_h in ((0, nk), (nk*dk, nk), (2*nk*dk, nv)):
            flat = ttnn.slice(conv, [0,0,0,start], [1,1,L,start+n_h*dk])
            t = ttnn.reshape(flat, (1, L, n_h, dk)); t = ttnn.permute(t, (0,2,1,3))
            if n_h != nv:
                t2 = ttnn.repeat_interleave(t, nv//n_h, dim=1); ttnn.deallocate(t); t = t2
            outs.append(t); ttnn.deallocate(flat)
        return outs
    def twocall():
        a = ttnn.slice(conv, [0,0,0,0], [1,1,L,6144])
        q,k,v0 = ttnn.experimental.nlp_create_qkv_heads(a, num_heads=nk, num_kv_heads=nk, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(a)
        b = ttnn.slice(conv, [0,0,0,4096], [1,1,L,10240])
        _v0,v1,v2 = ttnn.experimental.nlp_create_qkv_heads(b, num_heads=nk, num_kv_heads=nk, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(b); ttnn.deallocate(_v0)
        v = ttnn.concat([v0,v1,v2], dim=1)
        for t in (v0,v1,v2): ttnn.deallocate(t)
        qr = ttnn.repeat_interleave(q, 3, dim=1); ttnn.deallocate(q)
        kr = ttnn.repeat_interleave(k, 3, dim=1); ttnn.deallocate(k)
        return [qr, kr, v]
    # correctness: compare twocall v against current v
    cur = current(); two = twocall()
    for name, i in (("q",0),("k",1),("v",2)):
        a = ttnn.to_torch(cur[i]).float(); b = ttnn.to_torch(two[i]).float()
        print(f"  {name}: shapes {tuple(a.shape)} {tuple(b.shape)} maxdiff={float((a-b).abs().max()):.3g}")
    for t in cur+two: ttnn.deallocate(t)
    def run_current():
        for t in current(): ttnn.deallocate(t)
    def run_two():
        for t in twocall(): ttnn.deallocate(t)
    print(f"  L={L} current={bench(run_current):.2f} ms  two-call nlp_create_qkv_heads={bench(run_two):.2f} ms")
ttnn.close_mesh_device(d)
