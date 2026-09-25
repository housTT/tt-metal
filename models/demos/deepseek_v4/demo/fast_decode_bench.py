# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Measure real decode tok/s of the FastDecoder (43-layer on-device fp4 decode). Prefill a short
prompt, then run N decode steps; report per-step time (cold vs warm) and steady-state tok/s."""
import argparse
import time

import torch

import ttnn
from models.demos.deepseek_v4.reference import real_weights as RW
from models.demos.deepseek_v4.tt.fast_decode import FastDecoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--prompt", type=int, default=4)
    ap.add_argument("--decode", type=int, default=10)
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
    store = RW.RealWeightStore(snap)

    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        t0 = time.perf_counter()
        fd = FastDecoder(dev, store, cfg, scratch, num_layers=args.layers)
        print(f"[setup] built {args.layers} resident layers in {time.perf_counter()-t0:.1f}s", flush=True)

        # Fixed token sequence -> deterministic routing -> pass 2 reuses cached fp4 experts.
        prompt = torch.arange(2, 2 + args.prompt).reshape(1, -1)
        seq = [int(x) for x in torch.arange(100, 100 + args.decode).tolist()]

        def run_pass(tag):
            fd.prefill(prompt)
            dts = []
            for i, t in enumerate(seq):
                t0 = time.perf_counter()
                fd.decode_step(t, profile=(tag == "warm" and i == 0))
                dt = time.perf_counter() - t0
                dts.append(dt)
                if tag == "warm":
                    print(f"[{tag} decode {i+1}] {dt*1000:.0f} ms -> {1.0/dt:.2f} tok/s", flush=True)
            return dts

        t0 = time.perf_counter()
        run_pass("cold")  # populates the fp4 expert cache
        print(f"[pass1 cold] {time.perf_counter()-t0:.1f}s total (tilizes fp4 experts)", flush=True)
        # section profile on one warm token
        from models.demos.deepseek_v4.tt import mla_v4_device as _D
        fd.prefill(prompt)
        for k in _D.PROF:
            _D.PROF[k] = 0.0
        _D.PROF_ON = True
        fd.decode_step(seq[0])
        _D.PROF_ON = False
        print(f"[section prof/token] " + " ".join(f"{k}={v*1000:.0f}ms" for k, v in _D.PROF.items()), flush=True)
        dts = run_pass("warm")  # same routing -> cache hits -> true compute speed
        avg = sum(dts) / len(dts)
        print(f"\nWARM_DECODE_S_PER_TOK {avg:.3f}", flush=True)
        print(f"WARM_DECODE_TOK_S {1.0/avg:.3f}", flush=True)
        print("FAST_DECODE_OK", flush=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
