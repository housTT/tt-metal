# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Validate the real-weight KV-cache decode (tt/kv_cache_decode.py) against the full-recompute
reference, and report the per-token speedup.

Greedy-generates N tokens two ways and asserts identical token ids:
  reference: DeepSeekV4Generator.generate_ids (tt_forward_streaming, recompute per token)
  kv-cache : KVDecoder.prefill + decode_step (incremental, per-layer KV cache)

Usage:
  source models/demos/deepseek_v4/env.sh
  python models/demos/deepseek_v4/demo/kv_decode_check.py --layers 5 --new-tokens 2
  python models/demos/deepseek_v4/demo/kv_decode_check.py --layers 43 --new-tokens 3
"""
import argparse
import time

import ttnn
from models.demos.deepseek_v4.tt.generator import DeepSeekV4Generator
from models.demos.deepseek_v4.tt.kv_cache_decode import KVDecoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--new-tokens", type=int, default=2)
    ap.add_argument("--chips", type=int, default=1, help="mesh cols: 1 = single-chip non-resident MoE; 4 = resident sharded")
    args = ap.parse_args()

    # 1 chip -> non-resident MoE (fits); 4 chips -> resident tensor-parallel-sharded experts.
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, args.chips))
    dev = mesh
    try:
        gen = DeepSeekV4Generator(dev, num_layers=args.layers)
        input_ids = gen.tokenizer(args.prompt, return_tensors="pt").input_ids
        print(f"PROMPT: {args.prompt!r} ({input_ids.shape[1]} tokens), layers={args.layers}", flush=True)

        # reference: full-recompute greedy
        t0 = time.perf_counter()
        ref_ids = gen.generate_ids(input_ids, max_new_tokens=args.new_tokens, temperature=0.0, stop_at_eos=False)
        ref_dt = time.perf_counter() - t0
        print(f"REFERENCE ids={ref_ids} ({ref_dt:.1f}s, {ref_dt/max(1,args.new_tokens):.1f}s/tok)", flush=True)

        # kv-cache: prefill + incremental decode
        kd = KVDecoder(gen.scratch, gen.store, gen.layer_types, gen.mlp_types, dev, num_layers=args.layers)
        tp = time.perf_counter()
        logits = kd.prefill(input_ids)
        prefill_dt = time.perf_counter() - tp
        kv_ids = [int(logits.argmax(-1))]
        dec = []
        for _ in range(args.new_tokens - 1):
            td = time.perf_counter()
            logits = kd.decode_step(kv_ids[-1])
            dec.append(time.perf_counter() - td)
            kv_ids.append(int(logits.argmax(-1)))
        dec_s = f", decode {sum(dec)/len(dec):.1f}s/tok" if dec else ""
        print(f"KV-CACHE  ids={kv_ids} (prefill {prefill_dt:.1f}s{dec_s})", flush=True)
        print("KV-CACHE text:", repr(gen.tokenizer.decode(kv_ids)), flush=True)

        if ref_ids == kv_ids:
            print(f"KV_DECODE_OK tokens match {kv_ids}", flush=True)
        else:
            print(f"KV_DECODE_MISMATCH ref={ref_ids} kv={kv_ids}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
