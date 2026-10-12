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
    for width, cores in ((256, 8), (512, 16), (1024, 32)):
        w = torch.randn(1, 256, 2048, width) * 0.02
        x = torch.randn(1, 1, 1, 2048)
        ids = torch.randperm(256)[:8].reshape(1, 1, 1, 8)
        tw = ttnn.from_torch(w, device=mesh, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
        wref = ttnn.to_torch(tw).float()
        tx = ttnn.from_torch(x, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        xr = ttnn.to_torch(tx).float()[0, 0, 0]
        tid = ttnn.from_torch(ids.to(torch.int32), device=mesh, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT)
        both = torch.stack([xr @ wref[0, int(e)] for e in ids.flatten()])
        h = width // 2
        expected = both[:, :h] * torch.nn.functional.silu(both[:, h:])
        placeholder = ttnn.from_torch(torch.zeros(1, 1, 1, 256), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        compute = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.LoFi,
                                                         fp32_dest_acc_en=False, packer_l1_acc=True)
        nt = width // 32
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(min(8, cores), (cores + 7) // 8), in0_block_w=64 if cores <= 16 else 32,
            out_subblock_h=1, out_subblock_w=1, out_block_h=1, out_block_w=nt // cores,
            per_core_M=1, per_core_N=nt // cores, fuse_batch=False, fused_activation=None, mcast_in0=True)

        def baseline():
            y = ttnn.sparse_matmul(tx, tw, sparsity=placeholder, nnz=None, indices=tid, program_config=pc,
                                   compute_kernel_config=compute, memory_config=ttnn.L1_MEMORY_CONFIG,
                                   dtype=ttnn.bfloat16, is_input_a_sparse=False, output_tile=ttnn.Tile([32, 32]))
            return ttnn.swiglu(ttnn.reshape(y, (1, 8, 1, width)), dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

        yb = ttnn.to_torch(baseline()).float().reshape(8, -1)[:, :h]
        row = dict(width=width, baseline_pcc=pcc(expected, yb), baseline_us=timed(baseline))
        for nper in (1, 2):
            for kb in (4, 8, 16):
                for fp32 in (True, False):
                    if (width // 64) % nper or 8 * (width // 64) // nper > 110:
                        continue
                    key = f"n{nper}_k{kb}_{'f32' if fp32 else 'bf16'}"
                    try:
                        run = lambda: expert_swiglu_decode(tx, tid, tw, nper=nper, kb=kb, fp32_dest=fp32)
                        y = ttnn.to_torch(run()).float()[0, :, 0, :]
                        row[key] = dict(pcc=pcc(expected, y), max_abs=float((expected - y).abs().max()), us=timed(run))
                    except Exception as e:
                        row[key] = dict(error=str(e).splitlines()[0][:300])
                    print(width, key, row[key], flush=True)
        print(json.dumps(row), flush=True)
        rows.append(row)
    json.dump(rows, open(out, "w"), indent=1)
finally:
    ttnn.close_mesh_device(mesh)
