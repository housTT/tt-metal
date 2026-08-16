# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Why the sampler's local top-k had to change, measured at the real per-device logits shape.

No model build: this is `ttnn.topk` and friends on synthetic tensors of exactly the shape the LM
head produces on this mesh (``[1, 1, 32, 62080]`` bfloat16, one vocabulary shard of 248320/4). It
produces the ladder README section 4.3 quotes:

* cost against reduced width;
* cost against row count and leading dims (the point: it does not depend on them);
* cost against core count;
* the power-of-2 padding knob;
* ``stable=True`` vs ``False``;
* the grouped two-stage reduction at every tile-aligned group count, each checked for **exactness**
  against ``torch.topk`` on well-separated maxima.

    python .../doc/full_model/logs/probe_topk.py
"""

from __future__ import annotations

import sys
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import close_ornith_mesh, open_ornith_mesh

VOCAB, TP = 248320, 4
W = VOCAB // TP  # 62080, the per-device vocabulary shard
B, K = 32, 32  # sampler rows, max_top_k


def main():
    mesh = open_ornith_mesh()

    def timed(fn, iters=16):
        fn()
        ttnn.synchronize_device(mesh)
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        ttnn.synchronize_device(mesh)
        return (time.perf_counter() - start) / iters * 1e3

    def mk(shape, dtype=ttnn.bfloat16, t=None):
        if t is None:
            t = torch.randn(*shape).bfloat16()
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def scratch(shape):
        return mk(shape, dtype=ttnn.uint16, t=torch.zeros(*shape, dtype=torch.int32))

    def host(t):
        return ttnn.to_torch(t, mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh, dim=0))

    try:
        print(f"per-device vocabulary shard: {VOCAB} / {TP} = {W}; sampler rows {B}, max_top_k {K}\n")

        print("== ttnn.topk cost against the REDUCED WIDTH (rows fixed at 32) ==")
        for width in (1940, 3104, 3880, 7760, 15520, 31040, W):
            x, ix = mk((1, 1, B, width)), scratch((1, 1, B, width))

            def f(x=x, ix=ix):
                v, i = ttnn.topk(x, k=K, dim=-1, indices_tensor=ix, stable=False)
                ttnn.deallocate(v)
                ttnn.deallocate(i)

            print(f"   width {width:>6}: {timed(f):7.3f} ms")
            ttnn.deallocate(x)
            ttnn.deallocate(ix)

        print("\n== cost against ROWS and LEADING DIMS at a fixed width (the point: it does not depend on them) ==")
        for shape in ((1, 1, 32, 7760), (1, 1, 64, 7760), (1, 1, 128, 7760), (1, 4, 32, 1940), (1, 32, 32, 1940)):
            x, ix = mk(shape), scratch(shape)

            def f(x=x, ix=ix):
                v, i = ttnn.topk(x, k=K, dim=-1, indices_tensor=ix, stable=False)
                ttnn.deallocate(v)
                ttnn.deallocate(i)

            print(f"   shape {str(shape):>20} ({shape[-1]:>5} wide): {timed(f):7.3f} ms")
            ttnn.deallocate(x)
            ttnn.deallocate(ix)

        print("\n== cost against CORE COUNT at the real shard width ==")
        x, ix = mk((1, 1, B, W)), scratch((1, 1, B, W))
        for x1, y1 in ((3, 0), (7, 0), (7, 3), (7, 7), (10, 9)):
            crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x1, y1))})
            cores = (x1 + 1) * (y1 + 1)

            def f(crs=crs):
                v, i = ttnn.topk(x, k=K, dim=-1, indices_tensor=ix, sub_core_grids=crs, stable=False)
                ttnn.deallocate(v)
                ttnn.deallocate(i)

            try:
                print(f"   {cores:>4} cores: {timed(f):7.3f} ms")
            except Exception as exc:  # noqa: BLE001 - a refusal is the measurement
                print(f"   {cores:>4} cores: refused - {str(exc).splitlines()[0][:110]}")

        print("\n== stable, and the power-of-2 padding knob, at the real shard width ==")
        for stable in (False, True):

            def f(stable=stable):
                v, i = ttnn.topk(x, k=K, dim=-1, indices_tensor=ix, stable=stable)
                ttnn.deallocate(v)
                ttnn.deallocate(i)

            print(f"   width {W} stable={stable!s:<5}: {timed(f):7.3f} ms")
        xp, ixp = mk((1, 1, B, 65536)), scratch((1, 1, B, 65536))
        for stable in (False, True):

            def f(stable=stable):
                v, i = ttnn.topk(xp, k=K, dim=-1, indices_tensor=ixp, stable=stable)
                ttnn.deallocate(v)
                ttnn.deallocate(i)

            print(f"   width 65536 (pow2) stable={stable!s:<5}: {timed(f):7.3f} ms")

        def padf():
            p = ttnn.pad(x, [(0, 0), (0, 0), (0, 0), (0, 65536 - W)], value=-sys.float_info.max)
            ttnn.deallocate(p)

        print(f"   the pad itself ({W} -> 65536): {timed(padf):7.3f} ms")
        ttnn.deallocate(x)
        ttnn.deallocate(ix)
        ttnn.deallocate(xp)
        ttnn.deallocate(ixp)

        print("\n== the grouped two-stage reduction: cost AND exactness ==")
        # Well-separated maxima. A random bfloat16 tensor is dominated by exact ties, which both
        # spellings break arbitrarily, so it would compare tie-break policy rather than the reduction.
        torch.manual_seed(23)
        values = (torch.rand(1, 1, B, W) * 0.01).float()
        for row in range(B):
            positions = torch.randperm(W)[:K]
            values[0, 0, row, positions] = 10.0 + torch.arange(K).flip(0).float() * 0.5
        values = values.bfloat16()
        reference_values, reference_indices = torch.topk(values.float()[0, 0], k=K, dim=-1)
        logits = mk((1, 1, B, W), t=values)
        base_scratch = scratch((1, 1, B, W))
        v0, i0 = ttnn.topk(logits, k=K, dim=-1, indices_tensor=base_scratch, stable=True)
        print(
            f"   groups   1 (upstream): width {W:>5} -> --   "
            f"idx_exact={torch.equal(host(i0)[0, 0].to(torch.int64), reference_indices.to(torch.int64))}"
        )
        ttnn.deallocate(v0)
        ttnn.deallocate(i0)

        for groups in (2, 4, 5, 10, 20, 97, 194):
            if W % groups:
                continue
            per = W // groups
            if per % 32:
                print(f"   groups {groups:>3}: group width {per} is not tile aligned, skipped")
                continue
            offsets = mk((1, groups, 1, 1), dtype=ttnn.int32, t=(torch.arange(groups) * per).reshape(1, groups, 1, 1))
            g_scratch = scratch((1, groups, B, per))
            s2_scratch = scratch((1, 1, B, groups * K))

            def run():
                parts = [ttnn.slice(logits, [0, 0, 0, g * per], [1, 1, B, (g + 1) * per]) for g in range(groups)]
                stacked = ttnn.concat(parts, dim=1)
                for part in parts:
                    ttnn.deallocate(part)
                gv, gp = ttnn.topk(stacked, k=K, dim=-1, indices_tensor=g_scratch, stable=True)
                ttnn.deallocate(stacked)
                gi = ttnn.add(ttnn.typecast(gp, ttnn.int32), offsets, dtype=ttnn.int32)
                ttnn.deallocate(gp)
                vp = ttnn.permute(gv, [0, 2, 1, 3])
                ip = ttnn.permute(gi, [0, 2, 1, 3])
                ttnn.deallocate(gv)
                ttnn.deallocate(gi)
                cv = ttnn.reshape(vp, [1, 1, B, groups * K])
                ci = ttnn.reshape(ip, [1, 1, B, groups * K])
                tv, tp_ = ttnn.topk(cv, k=K, dim=-1, indices_tensor=s2_scratch, stable=True)
                ti = ttnn.gather(ci, dim=-1, index=tp_)
                ttnn.deallocate(cv)
                ttnn.deallocate(ci)
                ttnn.deallocate(tp_)
                return tv, ti

            tv, ti = run()
            exact_i = torch.equal(host(ti)[0, 0].to(torch.int64), reference_indices.to(torch.int64))
            exact_v = torch.allclose(host(tv)[0, 0].float(), reference_values.float(), atol=1e-2)
            ttnn.deallocate(tv)
            ttnn.deallocate(ti)

            def f():
                a, b = run()
                ttnn.deallocate(a)
                ttnn.deallocate(b)

            print(
                f"   groups {groups:>3}: width {per:>5} -> stage2 {groups * K:>5}  {timed(f):7.3f} ms  "
                f"idx_exact={exact_i} val_exact={exact_v}"
            )
            ttnn.deallocate(offsets)
            ttnn.deallocate(g_scratch)
            ttnn.deallocate(s2_scratch)

        print("\nPROBE_OK")
    finally:
        close_ornith_mesh(mesh)


if __name__ == "__main__":
    main()
