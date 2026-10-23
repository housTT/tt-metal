# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=int, choices=[1, 2, 4], required=True)
    p.add_argument("--layers", default="0,3")
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--batched", type=int, choices=[0, 1], default=1)
    p.add_argument("--distinct", type=int, choices=[0, 1], default=1)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import ttnn
    from tracy import signpost

    from ..tt.generator import _greedy_params, build_generator
    from .mesh_util import open_mesh

    mesh = open_mesh(args.mesh)
    result = dict(vars(args), output=str(args.output))
    try:
        from ..tt.model import Qwen35Model

        all_logits = Qwen35Model.all_logits
        Qwen35Model.all_logits = lambda self, hidden, valid_rows, block=None: None
        try:
            gen = build_generator(Path(__file__).parents[1], mesh,
                                  layer_ids=tuple(int(x) for x in args.layers.split(",")),
                                  warmup_first=(args.prompt_len,), warmup_continuation=(),
                                  decode_batches=tuple(sorted({1, args.batch})), batched_decode=bool(args.batched))
        finally:
            Qwen35Model.all_logits = all_logits
        ids = torch.randint(0, gen.vocab_size, (args.prompt_len,), generator=torch.Generator().manual_seed(3)).tolist()
        gen.generate(ids, args.steps + 1)
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        gen.reset()
        table = gen.default_page_table(1, args.prompt_len + args.steps + 1)
        ttnn.synchronize_device(mesh)
        signpost("PERF_PREFILL")
        (logits,) = gen._prefill_rows(torch.tensor([ids]), table, [args.prompt_len], [0], [0], False)
        gen._set_sampling(_greedy_params())
        gen.sampler.set_trace_bucket(None)
        gen.sampler.sample(gen._sampler_logits(logits, 1), enable_trace=False, tt_out_tok=gen.tt_tokens)
        ttnn.synchronize_device(mesh)
        signpost("PERF_PREFILL_END")
        logits.deallocate(True)
        batch = args.batch
        if batch > 1:
            table = gen.default_page_table(batch, args.prompt_len + args.steps + 1)
            rows = [ids] + [torch.randint(0, gen.vocab_size, (args.prompt_len,),
                                          generator=torch.Generator().manual_seed(100 + r)).tolist()
                            if args.distinct else ids for r in range(1, batch)]
            gen.prefill_forward(torch.tensor(rows), page_table=table, prompt_lens=[args.prompt_len] * batch)
        gen._write_page_table(batch, table)
        gen._write_positions(batch, [args.prompt_len] * batch)
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        signpost("PERF_DECODE")
        for _ in range(args.steps):
            gen._replay(batch, sample=True)
        ttnn.synchronize_device(mesh)
        signpost("PERF_DECODE_END")
        ttnn.ReadDeviceProfiler(mesh)
        result["tokens"] = gen.read_tokens(batch).tolist()
        result["counters"] = dict(gen.counters)
    finally:
        args.output.write_text(json.dumps(result, indent=1))
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
