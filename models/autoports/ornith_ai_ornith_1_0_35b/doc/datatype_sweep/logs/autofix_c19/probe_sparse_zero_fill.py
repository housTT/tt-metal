"""The mechanism behind C19's trace-capture failure, isolated from the model.

`probe_routed_experts_trace.py` puts the failure on `ttnn.sparse_matmul` with `dtype=bfloat4_b`,
not on the widening typecast. `SparseMatmulDeviceOperation::create_output_tensors`
(ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp:311-336) zero-fills
its output with `ttnn::zeros_like` on every call, and `full_like_impl`
(ttnn/cpp/ttnn/operations/creation/creation.cpp:239-244) only takes the on-device `ttnn::fill` fast
path for BFLOAT8_B / BFLOAT16 / FLOAT32 — BFLOAT4_B falls through to the host-side `full_impl`,
which is a host->device write.

Each row below is run eagerly (warm) and then inside a trace capture.
"""

import traceback

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import close_ornith_mesh, open_ornith_mesh
from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import _sparse_matmul_config

E, M, K, N = 8, 32, 256, 256
MEM = ttnn.L1_MEMORY_CONFIG


def dev(mesh, host, dtype, mem=MEM):
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=mem,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def short(exc):
    return str(exc).replace("\n", " ")[:400]


def run(mesh, name, fn):
    """Warm once eagerly, then once under trace capture."""
    try:
        out = fn()
        ttnn.synchronize_device(mesh)
        eager = "ok" + (f" dtype={out.dtype}" if isinstance(out, ttnn.Tensor) else "")
    except Exception as exc:  # noqa: BLE001
        print(f"RESULT {name:34s} eager=FAIL {type(exc).__name__}: {short(exc)}", flush=True)
        return
    try:
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            fn()
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.release_trace(mesh, tid)
        traced = "CAPTURABLE"
    except Exception as exc:  # noqa: BLE001
        traced = f"NOT-CAPTURABLE {type(exc).__name__}: {short(exc)}"
    ttnn.synchronize_device(mesh)
    print(f"RESULT {name:34s} eager={eager} | traced={traced}", flush=True)


def main():
    mesh = open_ornith_mesh(trace_region_size=64 << 20)
    try:
        torch.manual_seed(0)
        t4 = dev(mesh, torch.randn(1, E, M, N), ttnn.bfloat4_b)
        t8 = dev(mesh, torch.randn(1, E, M, N), ttnn.bfloat8_b)
        t16 = dev(mesh, torch.randn(1, E, M, N), ttnn.bfloat16)

        run(mesh, "zeros_like bf16 -> new", lambda: ttnn.zeros_like(t16))
        run(mesh, "zeros_like bfp8 -> new", lambda: ttnn.zeros_like(t8))
        run(mesh, "zeros_like bfp4 -> new", lambda: ttnn.zeros_like(t4))
        run(mesh, "fill bfp8 in place", lambda: ttnn.fill(t8, 0.0))
        run(mesh, "fill bfp4 in place", lambda: ttnn.fill(t4, 0.0))

        # the sparse matmul itself, mode (is_input_a_sparse=True, is_input_b_sparse=True)
        a4 = dev(mesh, torch.randn(1, E, M, K) * 0.1, ttnn.bfloat4_b)
        b4 = dev(mesh, torch.randn(1, E, K, N) * 0.1, ttnn.bfloat4_b, ttnn.DRAM_MEMORY_CONFIG)
        sparsity = ttnn.from_torch(
            torch.ones(1, 1, 1, E),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        cfg = _sparse_matmul_config(M, N, K, cores=8, in0_block_w=1, grid=mesh.compute_with_storage_grid_size())

        def smm(dtype, optional_output=None):
            return ttnn.sparse_matmul(
                a4,
                b4,
                sparsity=sparsity,
                nnz=None,
                is_input_a_sparse=True,
                is_input_b_sparse=True,
                memory_config=MEM,
                program_config=cfg,
                dtype=dtype,
                optional_output_tensor=optional_output,
            )

        run(mesh, "sparse_matmul dtype=bf16", lambda: smm(ttnn.bfloat16))
        run(mesh, "sparse_matmul dtype=bfp8", lambda: smm(ttnn.bfloat8_b))
        run(mesh, "sparse_matmul dtype=bfp4", lambda: smm(ttnn.bfloat4_b))
        prealloc4 = dev(mesh, torch.zeros(1, E, M, N), ttnn.bfloat4_b)
        run(mesh, "sparse_matmul bfp4 + out=", lambda: smm(ttnn.bfloat4_b, prealloc4))

        # blocker 1: the reduction's own dtype contract, verbatim.
        for label, t in (("bfp8", t8), ("bfp4", t4)):
            try:
                out = ttnn.experimental.deepseek_moe_fast_reduce_nc(t, dim=1, split_size=N)[0]
                ttnn.synchronize_device(mesh)
                print(f"REDUCE {label}: ok dtype={out.dtype}", flush=True)
                ttnn.deallocate(out)
            except Exception as exc:  # noqa: BLE001
                print(f"REDUCE {label}: FAIL {type(exc).__name__}: {short(exc)}", flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    finally:
        close_ornith_mesh(mesh)
        print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
