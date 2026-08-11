# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Op-level A/Bs behind three fused-decoder choices, at Ornith's exact shapes.

1. **Gated-DeltaNet head merge**: ``[1, 32, T, 128]`` head-major -> ``[1, T, 4096]`` token-major.
   Three ways to spell it: ``ttnn.experimental.nlp_concat_heads``; ``permute + reshape`` (the
   general equivalent); and a plain ``untilize -> reshape -> tilize``, which is only equivalent at
   ``T == 1`` (with one token there is nothing to transpose) — the printed PCC shows exactly that,
   1.000000 at T=1 and ~0 above it. So the flat relayout is a legal alternative for decode only,
   and the point of the measurement is which of the two legal decode spellings is faster.
2. **Recurrent-state matmuls.** ``[1, 32, 1, 128] x [1, 32, 128, 128]`` with and without an
   explicit ``core_grid``: ttnn's default heuristic picks a 4-core program for this batched shape.
3. **Outer product.** ``transpose(-1,-2) + matmul`` versus ``matmul(transpose_a=True)``.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/fused_decoder/logs/probe_decode_micro.py
"""

import time

import torch

import ttnn

NV, DK, DV = 32, 128, 128


def dev(mesh, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def pcc(a, b):
    a = a.double().flatten() - a.double().flatten().mean()
    b = b.double().flatten() - b.double().flatten().mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def same(a, b) -> str:
    """``bitwise-equal`` / ``differs``.

    A PCC printed to six decimals cannot establish bit equality — review round 12 found the work log
    claiming "bit-identical" on exactly that basis — so the comparison the documents make is the one
    the probe now performs.
    """
    return "bitwise-equal" if torch.equal(a, b) else "differs"


def bench(fn, mesh, reps=20, replays=20):
    """Per-call latency measured from a captured trace.

    A bare eager loop bottoms out at ~50 us of host dispatch per op, which is larger than every
    difference this probe is trying to resolve, so each candidate is captured as a trace of
    ``reps`` back-to-back calls and replayed.
    """
    ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(reps):
        fn()
    ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh)
    for _ in range(3):
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    start = time.time()
    for _ in range(replays):
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    per = (time.time() - start) / (replays * reps)
    ttnn.release_trace(mesh, trace_id)
    return per


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=200000000)
    grid = mesh.compute_with_storage_grid_size()
    full_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    gen = torch.Generator().manual_seed(17)
    try:
        # ---- 1. head merge -------------------------------------------------
        for seq in (1, 128, 2048):
            x = dev(mesh, torch.randn(1, NV, seq, DV, generator=gen).to(torch.bfloat16))

            def op_path(x=x, seq=seq):
                return ttnn.experimental.nlp_concat_heads(x)

            def manual(x=x, seq=seq):
                rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                r = ttnn.reshape(rm, [1, seq, NV * DV])
                ttnn.deallocate(rm)
                return ttnn.to_layout(r, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            def permute_reshape(x=x, seq=seq):
                t = ttnn.permute(x, (0, 2, 1, 3))
                out = ttnn.reshape(t, [1, seq, NV * DV])
                ttnn.deallocate(t)
                return out

            a = ttnn.to_torch(op_path()).float().reshape(1, seq, NV * DV)
            b = ttnn.to_torch(manual()).float().reshape(1, seq, NV * DV)
            c = ttnn.to_torch(permute_reshape()).float().reshape(1, seq, NV * DV)
            print(
                f"CONCATHEADS seq={seq:5d} nlp_concat_heads={bench(op_path, mesh) * 1e6:8.1f} us  "
                f"permute+reshape={bench(permute_reshape, mesh) * 1e6:8.1f} us "
                f"(pcc={pcc(a, c):.6f} {same(a, c)})  "
                f"untilize/reshape/tilize={bench(manual, mesh) * 1e6:8.1f} us "
                f"(pcc={pcc(a, b):.6f} {same(a, b)}; only equivalent at seq=1)",
                flush=True,
            )
            ttnn.deallocate(x)

        # ---- 2/3. recurrent-state matmuls ---------------------------------
        k_row = dev(mesh, torch.randn(1, NV, 1, DK, generator=gen), dtype=ttnn.float32)
        state = dev(mesh, torch.randn(1, NV, DK, DV, generator=gen), dtype=ttnn.float32)
        delta = dev(mesh, torch.randn(1, NV, 1, DV, generator=gen), dtype=ttnn.float32)

        def read_default():
            return ttnn.matmul(k_row, state, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckc)

        def read_grid():
            return ttnn.matmul(
                k_row,
                state,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=ckc,
                core_grid=full_grid,
            )

        print(
            f"STATEREAD  default={bench(read_default, mesh) * 1e6:8.1f} us  core_grid={bench(read_grid, mesh) * 1e6:8.1f} us  "
            f"pcc={pcc(ttnn.to_torch(read_default()).float(), ttnn.to_torch(read_grid()).float()):.6f}",
            flush=True,
        )

        def outer_transpose():
            kc = ttnn.transpose(k_row, -1, -2)
            out = ttnn.matmul(
                kc, delta, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckc, core_grid=full_grid
            )
            ttnn.deallocate(kc)
            return out

        def outer_fused():
            return ttnn.matmul(
                k_row,
                delta,
                transpose_a=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=ckc,
                core_grid=full_grid,
            )

        print(
            f"OUTER      transpose+matmul={bench(outer_transpose, mesh) * 1e6:8.1f} us  "
            f"matmul(transpose_a)={bench(outer_fused, mesh) * 1e6:8.1f} us  "
            f"pcc={pcc(ttnn.to_torch(outer_transpose()).float(), ttnn.to_torch(outer_fused()).float()):.6f}",
            flush=True,
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
