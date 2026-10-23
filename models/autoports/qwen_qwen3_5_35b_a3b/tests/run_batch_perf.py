# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
from dataclasses import replace
import json
import os
import time
from pathlib import Path

import torch

from .run_full_evidence import BOOK, _median

ISL = 128
RUNS = 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=int, choices=[1, 2, 4], required=True)
    p.add_argument("--steps", type=int, default=127)
    p.add_argument("--batched", type=int, choices=[0, 1], default=1)
    p.add_argument("--batches", default="4,32")
    p.add_argument("--moe-policy", default="{}")
    p.add_argument("--greedy-sampler", default="topk")
    p.add_argument("--sampled", type=int, choices=[0, 1], default=0)
    p.add_argument("--lm-head-dram", type=int, choices=[0, 1], default=0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import ttnn

    from ..tt.generator import _greedy_params, build_generator
    from ..tt.optimized_decoder import _memory_bytes
    from .mesh_util import open_mesh

    mesh = open_mesh(args.mesh)
    result = dict(vars(args), output=str(args.output), isl=ISL,
                  env={k: os.environ.get(k) for k in ("TT_METAL_TRACE_ALLOC_TRACKING", "QWEN35_STRICT_CACHE",
                                                      "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE")})
    try:
        batches = tuple(int(b) for b in args.batches.split(","))
        gen = build_generator(Path(__file__).parents[1], mesh, decode_batches=(1, *batches),
                              batched_decode=bool(args.batched), moe_policy=json.loads(args.moe_policy),
                              greedy_sampler=args.greedy_sampler, lm_head_dram=bool(args.lm_head_dram))
        result["memory_after_build"] = _memory_bytes(mesh)
        tok = gen.tokenizer
        book = tok(BOOK.read_text().strip() * 400, add_special_tokens=False).input_ids
        greedy = _greedy_params()
        batch_params = replace(greedy, temperature=1.0, top_k=32, top_p=1.0) if args.sampled else greedy
        table = gen.default_page_table(32, ISL + args.steps + 1)

        ttft, decode = [], []
        for _ in range(RUNS):
            gen.reset()
            stamps = []
            original = gen._replay

            def timed(batch, sample=True, _o=original, _s=stamps):
                _s.append(time.perf_counter())
                return _o(batch, sample)

            gen._replay = timed
            t0 = time.perf_counter()
            tokens = gen.generate(book[:ISL], args.steps + 1)
            end = time.perf_counter()
            gen._replay = original
            ttft.append(stamps[0] - t0)
            decode.append((end - stamps[0]) / (len(tokens) - 1))
        result["b1_tokens"] = [int(t) for t in tokens]
        rows = {1: dict(ttft_ms=_median(ttft) * 1e3, step_ms=_median(decode) * 1e3,
                        runs_ttft_ms=[x * 1e3 for x in ttft], runs_step_ms=[x * 1e3 for x in decode])}

        for batch in batches:
            steps = []
            for _ in range(RUNS):
                gen.reset()
                ids = torch.tensor([book[r * ISL : (r + 1) * ISL] for r in range(batch)])
                first = gen.prefill_forward(ids, page_table=table[:batch], prompt_lens=[ISL] * batch)
                current = first[:, 0].argmax(-1).reshape(batch, 1)
                gen.decode_forward(current, [ISL] * batch, page_table=table[:batch], sampling_params=batch_params)
                t0 = time.perf_counter()
                gen.decode_steps(batch, args.steps - 1, batch_params)
                steps.append((time.perf_counter() - t0) / (args.steps - 1))
            rows[batch] = dict(step_ms=_median(steps) * 1e3, runs_step_ms=[x * 1e3 for x in steps])
        for batch, row in rows.items():
            row["tokens_per_second_per_user"] = 1e3 / row["step_ms"]
            row["aggregate_tokens_per_second"] = batch * 1e3 / row["step_ms"]
        result["token_out"] = {str(b): r for b, r in rows.items()}
        result["memory_after_all_captures"] = _memory_bytes(mesh)
        result["counters"] = dict(gen.counters)
        print(json.dumps({k: result[k] for k in ("env", "token_out")}), flush=True)
    finally:
        args.output.write_text(json.dumps(result, indent=1))
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
