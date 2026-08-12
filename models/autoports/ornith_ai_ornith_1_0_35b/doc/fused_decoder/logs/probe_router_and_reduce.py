# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Two MoE candidates, measured for both accuracy and latency at Ornith's real decode shapes.

1. **Router dense-vector construction.** ``topk -> softmax(8) -> scatter`` (shipped) versus a
   threshold rewrite ``topk -> ge(kth) -> where -> softmax(256)`` that avoids the scatter's three
   untilize round trips. Accuracy is measured as the top-8 *set* agreement with a float64 torch
   reference on real layer-0 router weights, because expert selection is a discrete decision.

2. **Expert-axis reduction.** ``ttnn.experimental.fast_reduce_nc`` (shipped) versus
   ``ttnn.experimental.deepseek_moe_fast_reduce_nc``.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/fused_decoder/logs/probe_router_and_reduce.py
"""

import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R

TOKENS = 32
E = 256
K = 8
HIDDEN = 2048
INTER = 512
NEG = -1.0e30


def dev(mesh, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def timed(fn, iters=50):
    fn()
    ttnn.synchronize_device(fn.__self__ if hasattr(fn, "__self__") else None) if False else None
    start = time.time()
    for _ in range(iters):
        out = fn()
    return (time.time() - start) / iters, out


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0)
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    try:
        sd = R.load_layer_state_dict(0)
        gate_w = sd["mlp.gate.weight"].float()  # [E, hidden]
        gen = torch.Generator().manual_seed(4242)
        x = (torch.randn(1, 1, TOKENS, HIDDEN, generator=gen) * 0.5).to(torch.bfloat16)
        ref_logits = x.float().reshape(TOKENS, HIDDEN).double() @ gate_w.double().t()
        ref_vals, ref_idx = torch.topk(ref_logits, K, dim=-1)
        ref_dense = torch.zeros(TOKENS, E, dtype=torch.float64).scatter_(-1, ref_idx, torch.softmax(ref_vals, dim=-1))

        w = dev(mesh, gate_w.t().reshape(1, 1, HIDDEN, E))
        xt = dev(mesh, x)
        logits = ttnn.linear(xt, w, dtype=ttnn.float32, compute_kernel_config=ckc)

        def scatter_path():
            values, indices = ttnn.topk(logits, k=K, dim=-1, sorted=True)
            weights = ttnn.softmax(values, dim=-1, numeric_stable=True, compute_kernel_config=ckc)
            zeros = ttnn.typecast(ttnn.zeros_like(logits), ttnn.bfloat16)
            return ttnn.scatter(zeros, dim=-1, index=indices, src=ttnn.typecast(weights, ttnn.bfloat16))

        def where_path():
            values, _ = ttnn.topk(logits, k=K, dim=-1, sorted=True)
            thr = ttnn.slice(values, [0, 0, 0, K - 1], [1, 1, TOKENS, K])
            keep = ttnn.ge(logits, thr)
            masked = ttnn.where(keep, logits, NEG)
            dense = ttnn.softmax(masked, dim=-1, numeric_stable=True, compute_kernel_config=ckc)
            return ttnn.typecast(dense, ttnn.bfloat16)

        for name, fn in (("scatter", scatter_path), ("where", where_path)):
            out = fn()
            ttnn.synchronize_device(mesh)
            got = ttnn.to_torch(out).float().reshape(TOKENS, E)
            sel = set()
            match = 0
            for t in range(TOKENS):
                got_set = set(torch.topk(got[t], K).indices.tolist())
                ref_set = set(ref_idx[t].tolist())
                match += int(got_set == ref_set)
                sel |= got_set
            nonzero = int((got > 0).sum(-1).float().mean().item() * 100) / 100
            l1 = float((got.double() - ref_dense).abs().sum() / ref_dense.abs().sum())
            start = time.time()
            iters = 50
            for _ in range(iters):
                o = fn()
            ttnn.synchronize_device(mesh)
            per = (time.time() - start) / iters
            print(
                f"ROUTER {name:8s} set_match={match}/{TOKENS} mean_nonzero={nonzero} "
                f"score_L1_rel={l1:.6f} wall={per * 1e6:.1f} us/call",
                flush=True,
            )

        # ---- expert-axis reduction ----
        down = dev(mesh, torch.randn(1, E, TOKENS, HIDDEN, generator=gen).to(torch.bfloat16))
        ref_sum = ttnn.to_torch(down).float().sum(1, keepdim=True)
        for name in ("fast_reduce_nc", "deepseek_moe_fast_reduce_nc"):

            def run(name=name):
                if name == "fast_reduce_nc":
                    return ttnn.experimental.fast_reduce_nc(down, dims=[1])
                return ttnn.experimental.deepseek_moe_fast_reduce_nc(down, dim=1, split_size=HIDDEN)

            out = run()
            got = out[0] if isinstance(out, (list, tuple)) else out
            got_t = ttnn.to_torch(got).float().reshape(1, 1, TOKENS, HIDDEN)
            a = ref_sum.double().flatten() - ref_sum.double().mean()
            b = got_t.double().flatten() - got_t.double().mean()
            p = float((a * b).sum() / (a.norm() * b.norm() + 1e-12))
            ttnn.synchronize_device(mesh)
            start = time.time()
            iters = 30
            for _ in range(iters):
                run()
            ttnn.synchronize_device(mesh)
            per = (time.time() - start) / iters
            print(f"REDUCE {name:30s} pcc={p:.6f} wall={per * 1e6:.1f} us/call", flush=True)

        # ------------------------------------------------------------------ §4.17: the fused reduce
        # ttnn.experimental.deepseek_moe_fast_reduce_nc_fused folds mul(activation, expert_scores)
        # into the expert-axis reduction. Review round 23 found §4.13 had claimed no such fusion was
        # expressible without ever inventorying it. Its validation DOES accept this decoder's dense
        # [1, E, tokens, H] reduce at reduce_dim=1 with a [tokens, 1, 1, E] ROW_MAJOR score vector,
        # so the question is not expressibility but whether it is a good trade. Both arms are
        # faithful to the shipped graph: the shipped one scores the down projection's *input* at
        # moe_intermediate width (§3.2), the candidate scores its *output* inside the reduce.
        # NOTE the down projection here is a DENSE ttnn.matmul standing in for the shipped
        # sparse_matmul, which cannot be reproduced standalone without the sparsity mask. Both arms
        # pay it identically, so the ABSOLUTE times are not a proxy for the §5 windows but the
        # DIFFERENCE between the arms isolates exactly what the fusion changes: arm 1's permute and
        # moe_intermediate-wide multiply against arm 2's in-reduce scaling. Two things follow, and
        # §4.17 states both: the op accepts this decoder's dense shapes and is as accurate here as
        # the shipped pair, and it is not faster - because §3.2 already moved the multiply onto the
        # narrow input. The accuracy loss that adopting it would cause is separate, in-model, and
        # reproducible rather than committed (§4.17 gives the one-line reproduction).
        I = INTER
        gen2 = torch.Generator().manual_seed(77)
        act_i = dev(mesh, torch.randn(1, E, TOKENS, I, generator=gen2).to(torch.bfloat16))
        w_down = dev(mesh, torch.randn(1, E, I, HIDDEN, generator=gen2).to(torch.bfloat16))
        dense_t = torch.zeros(1, 1, TOKENS, E)
        for t in range(TOKENS):
            for e in torch.randperm(E, generator=gen2)[:K]:
                dense_t[0, 0, t, e] = float(torch.rand(1, generator=gen2))
        dense_tile = dev(mesh, dense_t.to(torch.bfloat16))
        dense_rm = dev(mesh, dense_t.reshape(TOKENS, 1, 1, E).to(torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT)
        zi = dev(mesh, torch.zeros(1, 1, 1, E, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        zm = dev(mesh, torch.zeros(1, 1, 1, E, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        # float64 reference: score, project, sum over the expert axis.
        a64 = ttnn.to_torch(act_i).double()
        w64 = ttnn.to_torch(w_down).double()
        s64 = dense_t.double().reshape(1, TOKENS, E).permute(0, 2, 1).reshape(1, E, TOKENS, 1)
        ref_fr = ((a64 * s64) @ w64).sum(1, keepdim=True)

        def shipped_score_input():
            sc = ttnn.permute(dense_tile, (0, 3, 2, 1))
            scaled = ttnn.multiply(act_i, sc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            dwn = ttnn.matmul(scaled, w_down, compute_kernel_config=ckc)
            out = ttnn.experimental.deepseek_moe_fast_reduce_nc(dwn, dim=1, split_size=HIDDEN)[0]
            ttnn.deallocate(sc)
            ttnn.deallocate(scaled)
            ttnn.deallocate(dwn)
            return out

        def fused_score_in_reduce():
            dwn = ttnn.matmul(act_i, w_down, compute_kernel_config=ckc)
            out = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
                dwn, zi, zm, 1, split_size=HIDDEN, cluster_axis=0, scores_tensor=dense_rm, compute_kernel_config=ckc
            )
            ttnn.deallocate(dwn)
            return out[0] if isinstance(out, (list, tuple)) else out

        for name, fn in (
            ("shipped score-input (dense stand-in)", shipped_score_input),
            ("fused score-in-reduce (dense stand-in)", fused_score_in_reduce),
        ):
            o = fn()
            g = ttnn.to_torch(o).double().reshape(1, 1, TOKENS, HIDDEN)
            aa = ref_fr.flatten() - ref_fr.mean()
            bb = g.flatten() - g.mean()
            p_fr = float((aa * bb).sum() / (aa.norm() * bb.norm() + 1e-12))
            ttnn.deallocate(o)
            ttnn.synchronize_device(mesh)
            st = time.time()
            iters = 20
            for _ in range(iters):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(mesh)
            print(
                f"FUSEDREDUCE {name:24s} pcc_vs_float64={p_fr:.6f} {(time.time()-st)/iters*1e6:8.1f} us/call",
                flush=True,
            )

        # ------------------------------------------------------------------ the §4.16 hoists
        # FusedMoE._routed_experts rebuilds two per-call quantities inside every 32-token expert
        # group: the sparse_matmul sparsity mask (reshape -> sum -> gtz -> to_layout) and the down
        # projection's score operand (permute). Both are hoistable to one whole-call computation
        # plus a per-group slice, exactly as the router above them already is. Whether that is
        # FASTER is the open question review rounds 19-21 kept raising, and this answers it at the
        # shipped prefill shape (2048 tokens, 64 groups of 32) rather than by argument.
        prefill_tokens, group = 2048, 32
        n_groups = prefill_tokens // group
        gen = torch.Generator().manual_seed(31)
        dense = dev(mesh, torch.rand(1, 1, prefill_tokens, E, generator=gen).to(torch.bfloat16))

        def per_group():
            """The superseded spelling: each group rebuilt its own mask and score operand."""
            outs = []
            for g in range(n_groups):
                lo = g * group
                scores_g = ttnn.slice(dense, [0, 0, lo, 0], [1, 1, lo + group, E])
                grouped = ttnn.reshape(scores_g, [1, 1, group, E])
                mask = ttnn.to_layout(ttnn.gtz(ttnn.sum(grouped, dim=-2, keepdim=True)), ttnn.ROW_MAJOR_LAYOUT)
                perm = ttnn.permute(scores_g, (0, 3, 2, 1))
                outs.append((mask, perm))
                ttnn.deallocate(scores_g)
            for mask, perm in outs:
                ttnn.deallocate(mask)
                ttnn.deallocate(perm)

        def hoisted():
            """What ships since §4.16: one whole-call mask and permute, then a per-group slice of each."""
            all_masks = ttnn.to_layout(
                ttnn.gtz(ttnn.sum(ttnn.reshape(dense, [1, n_groups, group, E]), dim=-2, keepdim=True)),
                ttnn.ROW_MAJOR_LAYOUT,
            )  # [1, n_groups, 1, E]
            all_scores = ttnn.permute(dense, (0, 3, 2, 1))  # [1, E, tokens, 1]
            outs = []
            for g in range(n_groups):
                lo = g * group
                outs.append(
                    (
                        ttnn.slice(all_masks, [0, g, 0, 0], [1, g + 1, 1, E]),
                        ttnn.slice(all_scores, [0, 0, lo, 0], [1, E, lo + group, 1]),
                    )
                )
            for mask, perm in outs:
                ttnn.deallocate(mask)
                ttnn.deallocate(perm)
            ttnn.deallocate(all_masks)
            ttnn.deallocate(all_scores)

        for name, fn in (("per-group (superseded)", per_group), ("hoisted whole-call (shipped)", hoisted)):
            fn()
            ttnn.synchronize_device(mesh)
            start = time.time()
            iters = 10
            for _ in range(iters):
                fn()
            ttnn.synchronize_device(mesh)
            per = (time.time() - start) / iters
            print(f"MASKHOIST {name:22s} {per * 1e3:8.3f} ms per 2048-token prefill MoE call", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
