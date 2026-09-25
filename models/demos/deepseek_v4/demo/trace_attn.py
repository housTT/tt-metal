# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Metal-Trace the on-device MLA-v4 attention decode (mla_decode_device) at a fixed context and
measure traced vs eager latency — the first real evidence of the trace dispatch-elimination win
on V4 code. Sliding layer (no compressor): all inputs are resident device tensors, so the op
sequence is fixed and directly traceable."""
import argparse
import time

import torch

import ttnn
from models.demos.deepseek_v4.reference import real_weights as RW
from models.demos.deepseek_v4.tt import mla_v4_device as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--past", type=int, default=127)
    ap.add_argument("--iters", type=int, default=50)
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
    attn = None
    for L in scratch.model.layers:
        if getattr(L.self_attn, "compressor", None) is None:
            attn = L.self_attn
            break
    assert attn is not None, "no sliding layer"
    torch.manual_seed(0)
    for p in attn.parameters():
        p.data = (torch.randn_like(p.data) * 0.02).to(p.data.dtype)
    attn.sinks.data = torch.randn_like(attn.sinks.data) * 0.5

    H, hd, rd = cfg.hidden_size, cfg.head_dim, cfg.qk_rope_head_dim
    past = args.past
    new_ln = torch.randn(1, 1, H) * 0.1
    kv_cache = torch.randn(1, 1, past, hd) * 0.1
    cos = torch.randn(1, 1, 1, rd) * 0.1
    sin = torch.randn(1, 1, 1, rd) * 0.1

    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=50_000_000)
    try:
        W = D.MLAv4DeviceWeights(attn, cfg, dev)
        new_ln_d = D._dev(new_ln, dev)
        kv_d = D._dev(kv_cache, dev)
        cos_d = D._dev(cos, dev)
        sin_d = D._dev(sin, dev)

        def run():
            out, mkv = D.mla_decode_device(new_ln_d, kv_d, W, cos_d, sin_d, dev)
            return out, mkv

        # compile
        run()
        ttnn.synchronize_device(dev)

        # eager timing
        t0 = time.perf_counter()
        for _ in range(args.iters):
            run()
        ttnn.synchronize_device(dev)
        t_eager = (time.perf_counter() - t0) / args.iters

        # capture trace
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        run()
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)

        # traced timing
        t0 = time.perf_counter()
        for _ in range(args.iters):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        t_traced = (time.perf_counter() - t0) / args.iters

        ttnn.release_trace(dev, tid)
        print(f"[attn decode past={past}] eager {t_eager*1000:.2f} ms/call  traced {t_traced*1000:.2f} ms/call  -> {t_eager/t_traced:.1f}x", flush=True)
        print(f"[projected 43-layer attn-only] eager {t_eager*43*1000:.0f} ms  traced {t_traced*43*1000:.0f} ms", flush=True)
        print("TRACE_ATTN_OK", flush=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
