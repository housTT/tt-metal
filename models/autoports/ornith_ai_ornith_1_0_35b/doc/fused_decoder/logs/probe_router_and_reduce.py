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
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
