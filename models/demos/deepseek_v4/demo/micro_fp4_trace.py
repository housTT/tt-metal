# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Does Metal-Trace remove the single-user MoE bottleneck? At batch=1 each routed expert is a
tiny vector-matrix matmul (~0.1ms compute) behind ~1.7ms host dispatch, so the MoE should be
DISPATCH-bound and trace-able (unlike attention, which trace_attn.py showed is compute-bound).

Measures one layer's fp4 MoE (6 routed experts + shared + router matmul) three ways on ONE chip:
  (1) eager-stream   : experts DMA'd per call (ttnn.to_device) + eager compute  = CURRENT path
  (2) eager-resident : experts resident (uploaded once) + eager compute          = isolates dispatch
  (3) traced-resident: experts resident + Metal-Trace the compute                = trace ceiling
Decomposes: DMA/layer = (1)-(2); dispatch removed by trace = (2)-(3). Realistic traced decode
keeps the data-dependent expert DMA OUTSIDE the trace, so per layer ~= DMA + traced-compute.
Projects single-user tok/s at 43 layers (+ measured attn/mHC that trace does NOT speed up)."""
import argparse
import time

import torch

import ttnn
from models.demos.deepseek_v4.reference import real_weights as RW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=6)
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--other-ms", type=float, default=170.0, help="measured attn+mHC+comp per token (untraced)")
    args = ap.parse_args()
    from transformers import AutoConfig

    snap = RW.find_snapshot()
    cfg = AutoConfig.from_pretrained(snap)
    I, H, limit = cfg.moe_intermediate_size, cfg.hidden_size, cfg.swiglu_limit
    n_exp = cfg.num_experts_per_tok
    E = args.experts
    store = RW.RealWeightStore(snap)

    # host-side pre-tilized bf4 experts (fused gate_up [H,2I], down [I,H]) — the Fp4ExpertCache format
    hgu, hdn = [], []
    for e in range(E):
        gu_T, dn_T = RW.expert_fused(store, 3, e)  # host bf16
        hgu.append(ttnn.from_torch(gu_T, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT))
        hdn.append(ttnn.from_torch(dn_T, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT))
    # shared expert + router (resident bf16), from a real layer-3 moe block
    scfg = AutoConfig.from_pretrained(snap)
    scfg.num_hidden_layers = 5
    scfg.num_nextn_predict_layers = 0
    scfg.layer_types = scfg.layer_types[:5]
    scfg.mlp_layer_types = scfg.mlp_layer_types[:5]
    scratch = RW.build_scratch(scfg)
    sl = None
    for L in scratch.model.layers:
        if not getattr(L.mlp, "is_hash", False):
            sl = L
            break
    RW.load_layer(sl, 3, store, skip_experts=True)
    se = sl.mlp.shared_experts
    w = [1.0 / E] * E  # fixed routing weights (timing only)

    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=100_000_000)
    try:
        tx = ttnn.from_torch(torch.randn(1, H) * 0.1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        se_g = ttnn.from_torch(se.gate_proj.weight.data.t().contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        se_u = ttnn.from_torch(se.up_proj.weight.data.t().contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        se_d = ttnn.from_torch(se.down_proj.weight.data.t().contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        gate_w = ttnn.from_torch(sl.mlp.gate.weight.data.t().contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

        # resident copies of the experts (uploaded once)
        rgu = [ttnn.to_device(t, dev) for t in hgu]
        rdn = [ttnn.to_device(t, dev) for t in hdn]

        def moe_compute(experts_gu, experts_dn):
            """One layer's MoE on device (router + shared + E routed, fp32 accumulate)."""
            sc = ttnn.matmul(tx, gate_w)  # router logits (on-device part of routing)
            scores = ttnn.sqrt(ttnn.softplus(sc))
            sg = ttnn.clamp(ttnn.matmul(tx, se_g), max=limit)
            su = ttnn.clamp(ttnn.matmul(tx, se_u), min=-limit, max=limit)
            sact = ttnn.multiply(ttnn.silu(sg), su)
            acc = ttnn.typecast(ttnn.matmul(sact, se_d), ttnn.float32)
            for k in range(E):
                gu = ttnn.matmul(tx, experts_gu[k])
                gt = ttnn.clamp(gu[..., :I], max=limit)
                up = ttnn.clamp(gu[..., I:], min=-limit, max=limit)
                act = ttnn.multiply(ttnn.silu(gt), up)
                y = ttnn.matmul(act, experts_dn[k])
                yf = ttnn.multiply(ttnn.typecast(y, ttnn.float32), float(w[k]))
                acc = ttnn.add(acc, yf)
                for t in (gu, gt, up, act, y, yf):
                    ttnn.deallocate(t)
            out = ttnn.reshape(acc, [1, 1, H])
            for t in (sc, scores, sg, su, sact):
                ttnn.deallocate(t)
            return out

        def eager_stream():
            tgu = [ttnn.to_device(t, dev) for t in hgu]
            tdn = [ttnn.to_device(t, dev) for t in hdn]
            out = moe_compute(tgu, tdn)
            for t in tgu + tdn:
                ttnn.deallocate(t)
            return out

        def eager_resident():
            return moe_compute(rgu, rdn)

        def timeit(fn, n):
            fn(); ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            outs = []
            for _ in range(n):
                o = fn()
                ttnn.deallocate(o)
            ttnn.synchronize_device(dev)
            return (time.perf_counter() - t0) / n

        t_stream = timeit(eager_stream, args.iters)
        t_resident = timeit(eager_resident, args.iters)

        # traced resident compute
        eager_resident(); ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        traced_out = moe_compute(rgu, rdn)  # captured; writes to fixed buffers
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        t_traced = (time.perf_counter() - t0) / args.iters
        ttnn.release_trace(dev, tid)

        dma = max(t_stream - t_resident, 0.0)
        disp_removed = max(t_resident - t_traced, 0.0)
        print(f"[MoE/layer, E={E}] eager-stream {t_stream*1000:.2f} | eager-resident {t_resident*1000:.2f} | traced-resident {t_traced*1000:.2f} ms", flush=True)
        print(f"  DMA/layer = stream-resident = {dma*1000:.2f} ms ; dispatch removed by trace = resident-traced = {disp_removed*1000:.2f} ms", flush=True)

        L = args.layers
        other = args.other_ms / 1000.0
        def tok(moe_layer, label):
            tokt = moe_layer * L + other
            print(f"  [{label}] MoE/token {moe_layer*L*1000:.0f} ms + other {args.other_ms:.0f} ms = {tokt*1000:.0f} ms -> {1.0/tokt:.2f} tok/s", flush=True)
        print(f"[projected single-user @ {L} layers, +{args.other_ms:.0f}ms attn/mHC]", flush=True)
        tok(t_stream, "current (eager stream)")
        tok(dma + t_traced, "realistic traced (DMA outside trace + traced compute)")
        tok(t_traced, "trace ceiling (resident, no DMA)")
        print("MICRO_FP4_TRACE_OK", flush=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
