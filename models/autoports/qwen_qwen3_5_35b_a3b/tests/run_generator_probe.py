# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import time
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=int, choices=[1, 2, 4], required=True)
    p.add_argument("--layers", default="0,3")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sequence", default="")
    p.add_argument("--random-lengths", type=int, default=0)
    args = p.parse_args()
    ref = torch.load(args.reference)
    ids, fed = ref["ids"][0].tolist(), ref["fed"]
    ref_top1 = [int(x.argmax()) for x in ref["logits"]]
    import ttnn

    from ..tt.generator import build_generator
    from .mesh_util import open_mesh

    mesh = open_mesh(args.mesh)
    result = dict(vars(args), output=str(args.output), reference=str(args.reference))
    try:
        t0 = time.perf_counter()
        gen = build_generator(Path(__file__).parents[1], mesh,
                              layer_ids=tuple(int(x) for x in args.layers.split(",")))
        result["build_seconds"] = time.perf_counter() - t0
        result["counters_after_build"] = dict(gen.counters)
        steps = len(fed) + 1
        if args.random_lengths:
            rng = torch.Generator().manual_seed(7)
            lengths = [int(x) for x in torch.randint(1, 8193, (args.random_lengths // 2,), generator=rng)]
            lengths += [8192 + int(x) for x in torch.randint(1, 8193, (args.random_lengths - len(lengths),), generator=rng)]
            done = []
            for length in lengths:
                prompt = torch.randint(0, gen.vocab_size, (length,), generator=rng).tolist()
                out = gen.generate(prompt, 2)
                done.append(dict(length=length, tokens=out))
                print(json.dumps(done[-1]), flush=True)
            result["random_lengths"] = done
        if args.sequence:
            runs = []
            for item in args.sequence.split(","):
                if item == "reset":
                    gen.reset()
                    continue
                out = gen.generate(ids, steps, host_sampling=item == "host")
                runs.append(dict(mode=item, tokens=out, matches_hf=out == ref_top1,
                                 first_miss=next((i for i, (a, b) in enumerate(zip(out, ref_top1)) if a != b), None)))
                print(json.dumps(runs[-1]), flush=True)
            result["sequence_runs"] = runs
            result["counters"] = dict(gen.counters)
            return
        feed = iter(fed)
        forced = gen.generate(ids, steps, next_input=lambda step, pred: next(feed, pred))
        result["teacher_forced_predictions"] = forced
        result["teacher_forced_top1_match"] = sum(int(a == b) for a, b in zip(forced, ref_top1)) / steps
        gen.reset()
        free = gen.generate(ids, steps)
        result["free_running"] = free
        result["free_running_matches_hf_greedy"] = free == ref_top1
        gen.reset()
        host = gen.generate(ids, steps, host_sampling=True)
        result["host_sampling"] = host
        result["host_equals_device"] = host == free
        gen.reset()
        again = gen.generate(ids, steps)
        result["repeat_identical"] = again == free
        result["counters"] = dict(gen.counters)
        result["decode_traces"] = sorted(gen._decode_traces)
        print(json.dumps({k: v for k, v in result.items() if k != "counters_after_build"}), flush=True)
    finally:
        args.output.write_text(json.dumps(result, indent=1))
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
