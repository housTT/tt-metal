import json, sys, time
import torch, ttnn

sys.path.insert(0, "/home/hous/dev/qwen3.5-35b-a3b/tt-metal")
from models.autoports.qwen_qwen3_5_35b_a3b.tt.expert_decode import expert_swiglu_decode

out = sys.argv[1]
torch.manual_seed(0)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=16 << 20, physical_device_ids=[3])
rows = []


def pcc(a, b):
    return float(torch.corrcoef(torch.stack((a.flatten().double(), b.flatten().double())))[0, 1])


def timed(fn):
    fn()
    t = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(50):
        fn()
    ttnn.end_trace_capture(mesh, t, cq_id=0)
    s = []
    for _ in range(7):
        t0 = time.perf_counter(); ttnn.execute_trace(mesh, t, cq_id=0, blocking=True)
        s.append((time.perf_counter() - t0) / 50 * 1e6)
    ttnn.release_trace(mesh, t)
    return sorted(s)[3]


try:
    for width in (256, 512, 1024):
        h = width // 2
        w = torch.randn(1, 256, 2048, width) * 0.02
        gate, up = torch.randn(2048, h) * 0.02, torch.randn(2048, h) * 0.02
        packed = torch.stack((gate.reshape(2048, -1, 32), up.reshape(2048, -1, 32)), dim=2).reshape(2048, -1)
        x = torch.randn(1, 1, 1, 2048)
        ids = torch.randperm(256)[:8].reshape(1, 1, 1, 8)
        tw = ttnn.from_torch(w, device=mesh, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
        ts = ttnn.from_torch(packed, device=mesh, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
        wref = ttnn.to_torch(tw).float()
        sref = ttnn.to_torch(ts).float().reshape(2048, -1, 2, 32)
        gref, uref = sref[:, :, 0].reshape(2048, h), sref[:, :, 1].reshape(2048, h)
        tx = ttnn.from_torch(x, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        xr = ttnn.to_torch(tx).float()[0, 0, 0]
        tid = ttnn.from_torch(ids.to(torch.int32), device=mesh, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT)
        both = torch.stack([xr @ wref[0, int(e)] for e in ids.flatten()])
        expected = both[:, :h] * torch.nn.functional.silu(both[:, h:])
        expected_shared = (xr @ uref) * torch.nn.functional.silu(xr @ gref)
        compute = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.LoFi,
                                                         fp32_dest_acc_en=False, packer_l1_acc=True)
        base = lambda: ttnn.experimental.minimal_matmul(tx, ts, fuse_swiglu=True, compute_kernel_config=compute,
                                                        memory_config=ttnn.L1_MEMORY_CONFIG)
        yb = ttnn.to_torch(base()).float()[0, 0, 0, :h] if len(ttnn.to_torch(base()).shape) == 4 else ttnn.to_torch(base()).float().reshape(-1)[:h]
        row = dict(width=width, minimal_shared_pcc=pcc(expected_shared, yb), minimal_shared_us=timed(base))
        for nper in (1, 2):
            if (h // 32) % nper or 9 * (h // 32) // nper > 110:
                continue
            run = lambda: expert_swiglu_decode(tx, tid, tw, shared=ts, nper=nper, kb=8)
            r, s = run()
            r = ttnn.to_torch(r).float()[0, :, 0, :]
            s = ttnn.to_torch(s).float()[0, 0, 0, :]
            row[f"n{nper}"] = dict(routed_pcc=pcc(expected, r), shared_pcc=pcc(expected_shared, s), us=timed(run))
            run0 = lambda: expert_swiglu_decode(tx, tid, tw, nper=nper, kb=8)
            row[f"n{nper}"]["routed_only_us"] = timed(run0)
        print(json.dumps(row), flush=True)
        rows.append(row)
    json.dump(rows, open(out, "w"), indent=1)
finally:
    ttnn.close_mesh_device(mesh)
