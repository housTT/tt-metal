# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Follow-up measurements for the stage-review findings (work_log.md sections 6, 7 and 9.6).

headsplit - the widened-projection head split vs the current reshape chain (rejected: +205 us/token
            in decode and +189 MB of weights per layer).  log: ../logs/probe_review_followups.log
conv1d    - ttnn.conv1d depthwise retried with an explicit DRAM width-slice config after the
            auto-config failure.  Every slice count exhausts the allocator.
ba        - core-grid and output-width sweep for the SLOW 32 x 5120 x 128 fused b|a projection.

Run: python doc/fused_decoder/probes/probe_review_followups.py [headsplit|conv1d|ba ...]
"""
import time, sys, torch, ttnn
CFG = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
def tt(t, d, dt=ttnn.float32, mc=ttnn.DRAM_MEMORY_CONFIG, lay=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dt, layout=lay, device=d, memory_config=mc)
def bench(d, fn, n=5):
    o=fn()
    ttnn.synchronize_device(d); s=time.perf_counter()
    for _ in range(n): fn()
    ttnn.synchronize_device(d); return 1e3*(time.perf_counter()-s)/n

def headsplit(d):
    """Current reshape/permute/repeat_interleave head split vs a widened projection + nlp_create_qkv_heads."""
    for L, label in ((2048, "prefill"), (32, "decode-rows")):
        nk, nv, dk = 16, 48, 128
        hid = 5120
        x = tt(torch.randn(1,1,L,hid), d, ttnn.bfloat16)
        w_narrow = tt(torch.randn(1,1,hid, nk*dk*2 + nv*dk), d, ttnn.bfloat16)     # 10240
        w_wide   = tt(torch.randn(1,1,hid, nv*dk*3), d, ttnn.bfloat16)             # 18432
        def current():
            mixed = ttnn.linear(x, w_narrow, dtype=ttnn.float32, compute_kernel_config=CFG)
            outs=[]
            for start, n_h in ((0, nk), (nk*dk, nk), (2*nk*dk, nv)):
                flat = ttnn.slice(mixed, [0,0,0,start], [1,1,L,start+n_h*dk])
                t = ttnn.reshape(flat, (1, L, n_h, dk))
                t = ttnn.permute(t, (0,2,1,3))
                if n_h != nv:
                    t2 = ttnn.repeat_interleave(t, nv//n_h, dim=1); ttnn.deallocate(t); t = t2
                outs.append(t); ttnn.deallocate(flat)
            ttnn.deallocate(mixed)
            for t in outs: ttnn.deallocate(t)
        def widened():
            mixed = ttnn.linear(x, w_wide, dtype=ttnn.float32, compute_kernel_config=CFG)
            q,k,v = ttnn.experimental.nlp_create_qkv_heads(mixed, num_heads=nv, num_kv_heads=nv,
                        transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(mixed)
            for t in (q,k,v): ttnn.deallocate(t)
        print(f"  headsplit {label} L={L}: current={bench(d,current):.2f} ms  widened+nlp_create_qkv_heads={bench(d,widened):.2f} ms", flush=True)
        for t in (x, w_narrow, w_wide): ttnn.deallocate(t)

def conv1d_retry(d):
    """ttnn.conv1d depthwise with explicit DRAM width slicing instead of auto."""
    L, D, K = 2048, 10240, 4
    x = torch.randn(1,1,L+K-1,D); w = torch.randn(D,1,K)*0.1
    for nslices in (0, 2, 4, 8, 16, 32):
        try:
            xrm = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=d)
            wd = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            sc = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=nslices)
            o = ttnn.conv1d(input_tensor=xrm, weight_tensor=wd, device=d, in_channels=D, out_channels=D,
                batch_size=1, input_length=L+K-1, kernel_size=K, stride=1, padding=0, dilation=1,
                groups=D, dtype=ttnn.bfloat16, compute_config=CFG, slice_config=sc)
            print(f"  conv1d num_slices={nslices}: OK out={o.shape}", flush=True)
            ttnn.deallocate(o)
        except Exception as e:
            print(f"  conv1d num_slices={nslices}: FAIL {str(e)[:150]}", flush=True)

def ba_geometry(d):
    """The SLOW 32 x 5120 x 128 fused b|a projection: can a wider N or a core grid help?"""
    hid = 5120
    x = tt(torch.randn(1,1,32,hid), d, ttnn.bfloat16)
    for width in (128, 256, 512, 1024):
        w = tt(torch.randn(1,1,hid,width), d, ttnn.float32)
        b = tt(torch.randn(1,1,1,width), d, ttnn.float32)
        try:
            print(f"  ba N={width:5d} default : {1e3*bench(d, lambda: ttnn.deallocate(ttnn.linear(x, w, bias=b, dtype=ttnn.float32, compute_kernel_config=CFG)), 20):.1f} us", flush=True)
        except Exception as e: print(f"  ba N={width} default FAIL {str(e)[:120]}")
        for cg in (ttnn.CoreGrid(y=8,x=8), ttnn.CoreGrid(y=4,x=8)):
            try:
                print(f"  ba N={width:5d} grid {cg.y}x{cg.x}: {1e3*bench(d, lambda: ttnn.deallocate(ttnn.linear(x, w, bias=b, dtype=ttnn.float32, compute_kernel_config=CFG, core_grid=cg)), 20):.1f} us", flush=True)
            except Exception as e: print(f"  ba N={width} grid {cg.y}x{cg.x} FAIL {str(e)[:120]}", flush=True)
        ttnn.deallocate(w); ttnn.deallocate(b)
    # bf16 weights for comparison
    w = tt(torch.randn(1,1,hid,128), d, ttnn.bfloat16); b = tt(torch.randn(1,1,1,128), d, ttnn.bfloat16)
    print(f"  ba N=128 bf16 weights: {1e3*bench(d, lambda: ttnn.deallocate(ttnn.linear(x, w, bias=b, dtype=ttnn.float32, compute_kernel_config=CFG)), 20):.1f} us", flush=True)

P = {"headsplit": headsplit, "conv1d": conv1d_retry, "ba": ba_geometry}
d = ttnn.open_mesh_device(ttnn.MeshShape(1,1), trace_region_size=0)
try:
    for n in (sys.argv[1:] or list(P)):
        print("==", n, flush=True); P[n](d)
finally:
    ttnn.close_mesh_device(d)
