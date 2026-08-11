"""Device time of ``paged_scaled_dot_product_attention_decode`` at the layer's real decode shape.

24 q / 4 kv heads, head_dim 256, page 64, 262144-token paged cache, one decode position.
Times a captured trace (ITERS ops per capture) so the number is device time, not host dispatch.
Falls back to loop-and-synchronize timing if trace capture is unavailable.

Env: POSITIONS, CACHE, KCHUNK (comma separated; 0 = default/auto), MAXCORES (comma separated),
     QCHUNK, FP32DEST, ITERS, REPLAYS.
"""

import os
import time

import torch

import ttnn

N_Q_HEADS = 24
N_KV_HEADS = 4
HEAD_DIM = 256
BLOCK = 64
SCALE = HEAD_DIM**-0.5

CACHE = int(os.environ.get("CACHE", "262144"))
POSITIONS = [int(x) for x in os.environ.get("POSITIONS", "262143").split(",")]
KCHUNKS = [int(x) for x in os.environ.get("KCHUNK", "512").split(",")]
MAXCORES = [int(x) for x in os.environ.get("MAXCORES", "1").split(",")]
FP32DEST = os.environ.get("FP32DEST", "1") == "1"
ITERS = int(os.environ.get("ITERS", "32"))
REPLAYS = int(os.environ.get("REPLAYS", "8"))


def main() -> None:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=FP32DEST, packer_l1_acc=True)
    nb = CACHE // BLOCK
    gen = torch.Generator().manual_seed(99)
    perm = torch.randperm(nb, generator=gen)
    k = torch.randn(nb, N_KV_HEADS, BLOCK, HEAD_DIM, generator=gen)
    v = torch.randn(nb, N_KV_HEADS, BLOCK, HEAD_DIM, generator=gen)

    def dev(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    tt_k, tt_v = dev(k), dev(v)
    tt_pt = dev(perm.to(torch.int32).reshape(1, nb), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    tt_q = dev(torch.randn(1, 1, N_Q_HEADS, HEAD_DIM, generator=gen))
    gx, gy = (int(x) for x in os.environ.get("GRID", "8,8").split(","))

    for pos in POSITIONS:
        tt_pos = dev(torch.tensor([pos], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        for kc in KCHUNKS:
            for mc in MAXCORES:
                prog = None
                if kc:
                    prog = ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                        q_chunk_size=int(os.environ.get("QCHUNK", "32")),
                        k_chunk_size=kc, exp_approx_mode=False, max_cores_per_head_batch=mc)
                kw = {"program_config": prog} if prog is not None else {}

                def run():
                    return ttnn.transformer.paged_scaled_dot_product_attention_decode(
                        tt_q, tt_k, tt_v, tt_pt, cur_pos_tensor=tt_pos, scale=SCALE,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=cfg, **kw)

                label = f"pos={pos} kchunk={kc or 'default'} maxcores={mc if kc else 'default'}"
                try:
                    o = run()
                    ttnn.deallocate(o)
                    ttnn.synchronize_device(mesh)
                except Exception as e:  # noqa: BLE001
                    print(f"PERF {label} FAILED {type(e).__name__}: {str(e)[:180]}", flush=True)
                    continue

                mode = "trace"
                try:
                    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                    outs = [run() for _ in range(ITERS)]
                    ttnn.end_trace_capture(mesh, tid, cq_id=0)
                    ttnn.synchronize_device(mesh)
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)  # warm
                    best = float("inf")
                    for _ in range(REPLAYS):
                        t0 = time.perf_counter()
                        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        best = min(best, (time.perf_counter() - t0) / ITERS)
                    ttnn.release_trace(mesh, tid)
                    for o in outs:
                        ttnn.deallocate(o)
                except Exception as e:  # noqa: BLE001
                    mode = f"loop({type(e).__name__})"
                    best = float("inf")
                    for _ in range(REPLAYS):
                        t0 = time.perf_counter()
                        os_ = [run() for _ in range(ITERS)]
                        ttnn.synchronize_device(mesh)
                        best = min(best, (time.perf_counter() - t0) / ITERS)
                        for o in os_:
                            ttnn.deallocate(o)
                print(f"PERF {label} mode={mode} per_op={best * 1e6:.1f}us", flush=True)
        ttnn.deallocate(tt_pos)

    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
