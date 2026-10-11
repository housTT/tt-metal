import json
import math
import sys
import time

import torch
import ttnn

from models.autoports.qwen_qwen3_5_35b_a3b.tt.optimized_decoder import _minimal_config, _Projection

out = sys.argv[1]
only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20, physical_device_ids=[3])
torch.manual_seed(0)
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
BANKS = mesh.dram_grid_size().x


def put(x, dtype, memory=DRAM):
    return ttnn.from_torch(x.to(torch.bfloat16), device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT,
                           memory_config=memory)


def traced(fn, repeats=20):
    fn()
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(repeats):
        fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    samples = []
    for _ in range(15):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        samples.append((time.perf_counter() - t0) * 1e6 / repeats)
    ttnn.release_trace(mesh, tid)
    samples.sort()
    return samples[len(samples) // 2]


def mcast_candidates(k, n):
    kt, nt = k // 32, n // 32
    for cores in (8, 16, 20, 32, 40, 64, 80):
        gx = 8 if cores % 8 == 0 else 10
        if cores % gx or cores // gx > 10 or math.ceil(nt / cores) * (cores - 1) >= nt:
            continue
        for kb in (2, 4, 8, 16, 32):
            if kt % kb == 0:
                yield {"decode_grid": [gx, cores // gx], "k_block": kb, "per_n": math.ceil(nt / cores)}


def dram_candidates(k, n):
    for readers in (1, 2):
        for cores in (8, 16, 32, 64):
            if k % (32 * cores):
                continue
            per = k // cores // 32
            for kb in (1, 2, 4, 8, 16):
                if per % kb == 0:
                    yield {"dram": True, "readers": readers, "cores": cores, "k_block": kb}


def projection_rows(name, k, n, dtype, fidelity, source_memory, out_memory):
    x = put(torch.randn(1, 1, 1, k), ttnn.bfloat16, source_memory)
    w = put(torch.randn(1, 1, k, n) * 0.02, DT[dtype])
    base = {"dtype": dtype, "fidelity": fidelity}
    reference = None
    for extra in [{}] + list(mcast_candidates(k, n)) + list(dram_candidates(k, n)):
        options = {**base, **extra}
        row = dict(case=name, k=k, n=n, dtype=dtype, fidelity=fidelity, options=extra)
        try:
            projection = _Projection(w, mesh, options)
            result = ttnn.to_torch(projection(x, out_memory)).float().reshape(-1)[:n]
            if reference is None:
                reference = result
            row["max_abs_vs_auto"] = float((result - reference).abs().max())
            row["trace_us"] = traced(lambda: projection(x, out_memory))
            if getattr(projection, "decode_weight", None) is not None:
                projection.decode_weight.deallocate(True)
        except Exception as error:
            row["error"] = str(error).splitlines()[0][:200]
        print(json.dumps(row), flush=True)
        rows.append(row)
    w.deallocate(True)
    x.deallocate(True)


def shared_rows(name, width):
    k = 2048
    gate, up = torch.randn(k, width) * 0.02, torch.randn(k, width) * 0.02
    interleaved = torch.stack((gate.reshape(k, -1, 32), up.reshape(k, -1, 32)), dim=2).reshape(1, 1, k, 2 * width)
    packed_minimal = put(interleaved, ttnn.bfloat4_b)
    packed_split = put(torch.cat((up, gate), dim=-1).reshape(1, 1, k, 2 * width), ttnn.bfloat4_b)
    x = put(torch.randn(1, 1, 1, k), ttnn.bfloat16, DRAM)
    compute = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True,
                                               packer_l1_acc=True)
    nt = 2 * width // 32
    configs = [None]
    for gx, gy in ((1, 1), (2, 1), (4, 1), (8, 1), (4, 2), (8, 2)):
        for kb in (2, 4, 8, 16):
            for nb in (2, 4, 8, 16):
                if nb * gx * gy < nt or nt % nb:
                    continue
                for sw in (1, 2):
                    if nb % sw == 0:
                        configs.append({"grid": [gx, gy], "M_block_size": 1, "K_block_size": kb, "N_block_size": nb,
                                        "subblock_h": 1, "subblock_w": sw})
    for config in configs:
        row = dict(case=name, k=k, n=2 * width, variant="minimal_swiglu", options=config)
        try:
            fn = lambda: ttnn.experimental.minimal_matmul(x, packed_minimal, fuse_swiglu=True,
                                                          config=_minimal_config(config), compute_kernel_config=compute,
                                                          memory_config=L1)
            fn()
            row["trace_us"] = traced(fn)
        except Exception as error:
            row["error"] = str(error).splitlines()[0][:200]
        print(json.dumps(row), flush=True)
        rows.append(row)
    for extra in list(mcast_candidates(k, 2 * width)) + list(dram_candidates(k, 2 * width)):
        options = {"dtype": "bfp4", "fidelity": "LoFi", **extra}
        row = dict(case=name, k=k, n=2 * width, variant="projection_then_multiply", options=extra)
        try:
            projection = _Projection(packed_split, mesh, options)

            def fn():
                both = projection(x, L1)
                return ttnn.multiply(both[..., width:], both[..., :width],
                                     input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=L1)

            fn()
            row["trace_us"] = traced(fn)
        except Exception as error:
            row["error"] = str(error).splitlines()[0][:200]
        print(json.dumps(row), flush=True)
        rows.append(row)
    for t in (packed_minimal, packed_split, x):
        t.deallocate(True)


def router_rows():
    k, n = 2048, 288
    x = put(torch.randn(1, 1, 1, k), ttnn.bfloat16, DRAM)
    w = put(torch.randn(1, 1, k, n) * 0.02, ttnn.bfloat16)
    compute = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                     math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
    for gx, gy in ((1, 1), (3, 1), (9, 1), (8, 2), (3, 3)):
        for kb in (4, 8, 16, 32, 64):
            per_n = -(-9 // (gx * gy))
            row = dict(case="router", k=k, n=n, variant="mcast1d", options=dict(grid=[gx, gy], k_block=kb))
            try:
                program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(gx, gy), in0_block_w=kb, per_core_M=1, per_core_N=per_n,
                    out_subblock_h=1, out_subblock_w=per_n, out_block_h=1, out_block_w=per_n, fuse_batch=True,
                    fused_activation=None, mcast_in0=True)
                fn = lambda: ttnn.linear(x, w, compute_kernel_config=compute, program_config=program, memory_config=L1)
                fn()
                row["trace_us"] = traced(fn)
            except Exception as error:
                row["error"] = str(error).splitlines()[0][:200]
            print(json.dumps(row), flush=True)
            rows.append(row)
    for extra in dram_candidates(k, n):
        row = dict(case="router", k=k, n=n, variant="dram", options=extra)
        try:
            projection = _Projection(w, mesh, {"dtype": "bf16", "fidelity": "HiFi4", **extra})
            fn = lambda: projection(x, L1)
            fn()
            row["trace_us"] = traced(fn)
        except Exception as error:
            row["error"] = str(error).splitlines()[0][:200]
        print(json.dumps(row), flush=True)
        rows.append(row)


rows = []
cases = {
    "x4_attention_in_full": lambda: projection_rows("x4_attention_in_full", 2048, 2560, "bfp8", "LoFi", DRAM, L1),
    "x4_attention_in_linear": lambda: projection_rows("x4_attention_in_linear", 2048, 3136, "bf16", "HiFi4", DRAM, L1),
    "x2_attention_in_full": lambda: projection_rows("x2_attention_in_full", 2048, 4608, "bfp8", "LoFi", DRAM, L1),
    "x2_attention_in_linear": lambda: projection_rows("x2_attention_in_linear", 2048, 6208, "bf16", "HiFi4", DRAM, L1),
    "x4_attention_out_full": lambda: projection_rows("x4_attention_out_full", 1024, 2048, "bfp8", "LoFi", L1, L1),
    "x4_attention_out_linear": lambda: projection_rows("x4_attention_out_linear", 1024, 2048, "bfp8", "HiFi2", L1, L1),
    "x2_attention_out_full": lambda: projection_rows("x2_attention_out_full", 2048, 2048, "bfp8", "LoFi", L1, L1),
    "x2_attention_out_linear": lambda: projection_rows("x2_attention_out_linear", 2048, 2048, "bfp8", "HiFi2", L1, L1),
    "x4_shared_down": lambda: projection_rows("x4_shared_down", 128, 2048, "bfp8", "LoFi", L1, L1),
    "x2_shared_down": lambda: projection_rows("x2_shared_down", 256, 2048, "bfp8", "LoFi", L1, L1),
    "x4_shared_gate_up": lambda: shared_rows("x4_shared_gate_up", 128),
    "x2_shared_gate_up": lambda: shared_rows("x2_shared_gate_up", 256),
    "router": router_rows,
}
for name, fn in cases.items():
    if only is None or name in only:
        fn()
        json.dump(rows, open(out, "w"), indent=1)
ttnn.close_mesh_device(mesh)
