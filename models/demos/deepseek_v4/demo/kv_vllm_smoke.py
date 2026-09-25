# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Smoke-test the vLLM-wired KV decode surface: generator.prefill_kv + decode_kv on real
weights, full 43 layers. Confirms correct tokens (" Paris") and measures true prefill/decode
tok/s exactly as the vLLM adapter drives it."""
import argparse
import time

import ttnn

from models.demos.deepseek_v4.tt.generator import DeepSeekV4Generator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--new-tokens", type=int, default=4)
    args = ap.parse_args()

    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        gen = DeepSeekV4Generator(dev, num_layers=args.layers)
        ids = gen.tokenizer(args.prompt, return_tensors="pt").input_ids
        print(f"[prompt] {args.prompt!r}  ({ids.shape[1]} tokens)", flush=True)

        t0 = time.perf_counter()
        logits = gen.prefill_kv(ids)  # [1, vocab]
        t_prefill = time.perf_counter() - t0
        nxt = int(logits[0].argmax(-1))
        toks = [nxt]
        print(f"[prefill] {t_prefill:.2f}s -> first token {nxt} {gen.tokenizer.decode([nxt])!r}", flush=True)

        dts = []
        for i in range(args.new_tokens):
            t0 = time.perf_counter()
            logits = gen.decode_kv(toks[-1], profile=(i == 0))
            dt = time.perf_counter() - t0
            dts.append(dt)
            nxt = int(logits[0].argmax(-1))
            toks.append(nxt)
            print(f"[decode {i+1}] {dt:.2f}s -> {nxt} {gen.tokenizer.decode([nxt])!r}", flush=True)

        completion = gen.tokenizer.decode(toks)
        avg = sum(dts) / len(dts)
        print(f"\nCOMPLETION {completion!r}")
        print(f"PREFILL_S {t_prefill:.2f}")
        print(f"DECODE_S_PER_TOK {avg:.2f}")
        print(f"DECODE_TOK_S {1.0/avg:.3f}")
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
