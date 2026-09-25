# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""De-risk the multi-chip re-architecture: does Metal-Trace erase the 4-device SPMD dispatch
penalty on the REPLICATED attention+mHC? On the (1,4) mesh those ops ran ~1.74x slower than one
chip (170->296ms/token) — if that extra cost is per-op HOST dispatch (4-device coordination),
trace replays it from device-side command buffers and should collapse it back toward the raw
compute. If instead it's real device/fabric time, trace won't help (like single-chip attention,
which trace_attn.py showed is compute-bound at 1.0x).

Runs the attention half (mhc_device -> input_ln -> mla_decode_device) on a fixed context, EAGER vs
TRACED, on a (1,N) mesh. Compares per-call; projects x43. Decides whether to build the full
trace-the-mesh-attention + parallel-MoE-DMA path."""
import argparse
import time

import torch

import ttnn
from models.demos.deepseek_v4.reference import real_weights as RW
from models.demos.deepseek_v4.tt import mla_v4_device as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=int, default=4)
    ap.add_argument("--past", type=int, default=127)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--layers", type=int, default=43)
    args = ap.parse_args()
    from transformers import AutoConfig

    snap = RW.find_snapshot()
    cfg = AutoConfig.from_pretrained(snap)
    cfg.num_nextn_predict_layers = 0
    scfg = AutoConfig.from_pretrained(snap)
    scfg.num_hidden_layers = 5
    scfg.num_nextn_predict_layers = 0
    scfg.layer_types = scfg.layer_types[:5]
    scfg.mlp_layer_types = scfg.mlp_layer_types[:5]
    scratch = RW.build_scratch(scfg)
    layer = None
    for L in scratch.model.layers:
        if getattr(L.self_attn, "compressor", None) is None:
            layer = L
            break
    assert layer is not None, "no sliding layer"
    attn = layer.self_attn
    torch.manual_seed(0)
    for p in attn.parameters():
        p.data = (torch.randn_like(p.data) * 0.02).to(p.data.dtype)
    attn.sinks.data = torch.randn_like(attn.sinks.data) * 0.5

    H, hd, rd, HC = cfg.hidden_size, cfg.head_dim, cfg.qk_rope_head_dim, cfg.hc_mult
    past, eps = args.past, cfg.rms_norm_eps

    def run_on(n):
        if n > 1:
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        dev = ttnn.open_mesh_device(ttnn.MeshShape(1, n), trace_region_size=100_000_000)
        try:
            W = D.MLAv4DeviceWeights(attn, cfg, dev)
            Wm = D.MHCDevice(layer.attn_hc, cfg, dev)
            input_ln = D._dev(layer.input_layernorm.weight.data, dev, dtype=ttnn.bfloat16)
            streams = D._dev(torch.randn(1, 1, HC, H) * 0.1, dev)
            kv_d = D._dev(torch.randn(1, 1, past, hd) * 0.1, dev)
            cos_d = D._dev(torch.randn(1, 1, 1, rd) * 0.1, dev)
            sin_d = D._dev(torch.randn(1, 1, 1, rd) * 0.1, dev)

            def run():
                post, comb, collapsed = D.mhc_device(streams, Wm)
                collapsed_ln = ttnn.rms_norm(collapsed, epsilon=eps, weight=input_ln)
                out, mkv = D.mla_decode_device(collapsed_ln, kv_d, W, cos_d, sin_d, dev)
                s2 = D._stream_mix(post, comb, out, streams)
                for t in (post, comb, collapsed, collapsed_ln, out, mkv):
                    try: ttnn.deallocate(t)
                    except Exception: pass
                return s2

            run(); ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(args.iters):
                o = run(); ttnn.deallocate(o)
            ttnn.synchronize_device(dev)
            t_eager = (time.perf_counter() - t0) / args.iters

            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            traced_o = run()
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(args.iters):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            t_traced = (time.perf_counter() - t0) / args.iters
            ttnn.release_trace(dev, tid)
            return t_eager, t_traced
        finally:
            ttnn.close_mesh_device(dev)
            if n > 1:
                try: ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
                except Exception: pass

    n = args.mesh
    te, tt = run_on(n)
    L = args.layers
    print(f"[attn+mHC half on {n}-chip mesh] eager {te*1000:.2f} ms/call  traced {tt*1000:.2f} ms/call  -> {te/tt:.2f}x", flush=True)
    print(f"[projected x{L} layers] eager {te*L*1000:.0f} ms  traced {tt*L*1000:.0f} ms  (single-chip ref ~170 ms)", flush=True)
    print("MICRO_TRACE_MESH_OK", flush=True)


if __name__ == "__main__":
    main()
