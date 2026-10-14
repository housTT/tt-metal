# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=int, choices=[1, 2, 4], required=True)
    p.add_argument("--prompt", default="Explain in three sentences why the sky is blue.")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--layers", default="")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import ttnn

    from ..tt.generator import build_generator
    from .mesh_util import open_mesh

    mesh = open_mesh(args.mesh)
    result = dict(vars(args), output=str(args.output))
    try:
        t0 = time.perf_counter()
        layer_ids = tuple(int(x) for x in args.layers.split(",")) if args.layers else None
        gen = build_generator(Path(__file__).parents[1], mesh, layer_ids=layer_ids)
        result["build_seconds"] = time.perf_counter() - t0
        result["load_seconds"] = gen.model.load_seconds
        text = gen.tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=False,
                                                 add_generation_prompt=True)
        ids = gen.tokenizer(text, add_special_tokens=False).input_ids
        result["rendered_prompt"] = text
        result["prompt_tokens"] = len(ids)
        times = []
        original = gen._replay

        def timed(batch, sample=True):
            start = time.perf_counter()
            out = original(batch, sample)
            times.append(start)
            return out

        gen._replay = timed
        start = time.perf_counter()
        tokens = gen.generate(ids, args.max_new_tokens)
        end = time.perf_counter()
        gen._replay = original
        result["tokens"] = tokens
        result["text"] = gen.tokenizer.decode(tokens)
        result["ttft_seconds"] = times[0] - start if times else None
        steps = [b - a for a, b in zip(times, times[1:] + [end])]
        result["decode_ms_per_token_median"] = sorted(steps)[len(steps) // 2] * 1e3 if steps else None
        result["counters"] = dict(gen.counters)
        result["pad_fallbacks"] = gen.model.pad_fallbacks
        print(json.dumps({k: v for k, v in result.items() if k not in ("tokens", "rendered_prompt")}), flush=True)
    finally:
        args.output.write_text(json.dumps(result, indent=1))
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
