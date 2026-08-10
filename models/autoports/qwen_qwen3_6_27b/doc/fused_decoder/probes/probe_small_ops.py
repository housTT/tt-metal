# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Contract probes for the small dedicated ops the fused decoder folds work into.

Each block is a self-contained unfused-vs-fused comparison on the real Qwen3.6-27B shapes:

* ``rotary_embedding_hf``            vs the spelled-out slice/neg/concat partial RoPE
* sharded ``rms_norm``               vs the interleaved single-core one (PCC + latency)
* ``multiply(..., input_tensor_a_activations=[SILU])`` vs ``silu`` then ``multiply``
* ``paged_fused_update_cache``       vs two ``paged_update_cache`` calls (latency)

    python .../probes/probe_small_ops.py
"""

from __future__ import annotations

import time

import torch

import ttnn

HIDDEN = 5120
HEAD_DIM = 256
ROTARY_DIM = 64
N_HEADS = 24
INTER = 17408


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().to(torch.float64).flatten()
    b = b.detach().to(torch.float64).flatten()
    a, b = a - a.mean(), b - b.mean()
    if a.norm() == 0 or b.norm() == 0:
        return float(a.norm() == b.norm())
    return float((a @ b) / (a.norm() * b.norm()))


def timed(fn, iters=10):
    fn()
    ttnn.synchronize_device(fn.__self__ if hasattr(fn, "__self__") else DEV)
    best = None
    for _ in range(iters):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(DEV)
        dt = (time.perf_counter() - t0) * 1e3
        best = dt if best is None else min(best, dt)
    return best, out


DEV = None


def torch_partial_rope(x, cos, sin, rotary_dim):
    rot = x[..., :rotary_dim]
    half = rotary_dim // 2
    rotated = torch.cat([-rot[..., half:], rot[..., :half]], dim=-1)
    emb = rot * cos + rotated * sin
    return torch.cat([emb, x[..., rotary_dim:]], dim=-1)


def main() -> None:
    global DEV
    DEV = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        grid = DEV.compute_with_storage_grid_size()
        print(f"compute_with_storage_grid_size = {grid.x} x {grid.y} = {grid.x * grid.y} cores", flush=True)

        def dev(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=DEV)

        # ---------------------------------------------------------- rotary_embedding_hf
        torch.manual_seed(0)
        # Only prefill mode is probed here.  In decode mode the op requires a HEIGHT_SHARDED
        # input *and* sharded per-user cos/sin; prefill mode broadcasts cos/sin over dim 1,
        # which is the batch axis in the decode layout, so it cannot serve per-user positions.
        # See doc/fused_decoder/work_log.md for the decode RoPE decision.
        for tag, xshape, cshape in (("prefill", (1, N_HEADS, 2048, HEAD_DIM), (1, 1, 2048, ROTARY_DIM)),):
            x = torch.randn(*xshape)
            angle = torch.randn(*cshape[:-1], ROTARY_DIM // 2)
            cos = torch.cat([angle.cos(), angle.cos()], dim=-1)
            sin = torch.cat([angle.sin(), angle.sin()], dim=-1)
            ref = torch_partial_rope(x, cos, sin, ROTARY_DIM)
            tx, tc, ts = dev(x), dev(cos), dev(sin)
            rot = ttnn.slice(tx, [0, 0, 0, 0], [*xshape[:-1], ROTARY_DIM])
            emb = ttnn.experimental.rotary_embedding_hf(rot, tc, ts, is_decode_mode=(tag == "decode"))
            passthrough = ttnn.slice(tx, [0, 0, 0, ROTARY_DIM], list(xshape))
            out = ttnn.concat([emb, passthrough], dim=-1)
            print(f"rotary_embedding_hf {tag:8s} pcc={pcc(ref, ttnn.to_torch(out).float()):.6f}", flush=True)
            for t in (tx, tc, ts, rot, emb, passthrough, out):
                ttnn.deallocate(t)

        # ---------------------------------------------------------------- sharded rms_norm
        x = torch.randn(1, 1, 32, HIDDEN)
        w = torch.randn(1, 1, 1, HIDDEN) * 0.02 + 1.0
        ref = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)) * w
        tx, tw = dev(x), dev(w)
        base_ms, base_out = timed(lambda: ttnn.rms_norm(tx, epsilon=1e-6, weight=tw))
        print(f"rms_norm interleaved  ms={base_ms:.3f} pcc={pcc(ref, ttnn.to_torch(base_out).float()):.6f}", flush=True)
        for cores in (16, 20, 32, 40, 64, 80):
            if HIDDEN // 32 % cores:
                continue
            block_w = HIDDEN // cores // 32
            subblock_w = max(s for s in range(1, 5) if block_w % s == 0)
            mem = ttnn.create_sharded_memory_config(
                shape=(32, HIDDEN // cores),
                core_grid=ttnn.num_cores_to_corerangeset(cores, grid, row_wise=True),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[grid.x, grid.y],
                subblock_w=subblock_w,
                block_h=1,
                block_w=block_w,
                inplace=False,
            )

            def run():
                xs = ttnn.to_memory_config(tx, mem)
                out = ttnn.rms_norm(xs, epsilon=1e-6, weight=tw, program_config=cfg, memory_config=mem)
                ttnn.deallocate(xs)
                res = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(out)
                return res

            ms, out = timed(run)
            print(
                f"rms_norm sharded {cores:3d}c ms={ms:.3f} pcc={pcc(ref, ttnn.to_torch(out).float()):.6f} "
                f"block_w={block_w} subblock_w={subblock_w}",
                flush=True,
            )
            ttnn.deallocate(out)

        # ----------------------------------------------------- silu folded into multiply
        g = torch.randn(1, 1, 2048, INTER)
        u = torch.randn(1, 1, 2048, INTER)
        ref = torch.nn.functional.silu(g) * u
        tg, tu = dev(g), dev(u)
        ms_a, out_a = timed(lambda: ttnn.multiply(ttnn.silu(tg), tu), iters=5)
        ms_b, out_b = timed(lambda: ttnn.multiply(tg, tu, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]), iters=5)
        print(
            f"silu+mul  ms={ms_a:.3f} pcc={pcc(ref, ttnn.to_torch(out_a).float()):.6f} | "
            f"mul(act=SILU) ms={ms_b:.3f} pcc={pcc(ref, ttnn.to_torch(out_b).float()):.6f}",
            flush=True,
        )
    finally:
        ttnn.close_mesh_device(DEV)


if __name__ == "__main__":
    main()
