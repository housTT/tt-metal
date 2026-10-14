# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch

VOCAB = 248320
ROWS = 32


def synthetic_logits(local_vocab, tp):
    gen = torch.Generator().manual_seed(0)
    logits = torch.full((ROWS, local_vocab * tp), -30.0)
    ids = torch.randperm(VOCAB, generator=gen)[:64]
    values = -torch.log(torch.arange(1, 65, dtype=torch.float32)) * 1.2
    logits[:, ids] = values
    return logits


def expected(host, top_k, top_p, temperature):
    out = torch.zeros_like(host)
    for r in range(host.shape[0]):
        top = host[r].topk(top_k)
        probs = (top.values / temperature).softmax(-1)
        keep = torch.cumsum(probs, -1) - probs < top_p
        keep[0] = True
        out[r, top.indices[keep]] = probs[keep] / probs[keep].sum()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=int, choices=[1, 2, 4], required=True)
    p.add_argument("--draws", type=int, default=64)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import ttnn

    from ..tt.generator import Qwen35Generator, _greedy_params
    from .mesh_util import open_mesh

    mesh = open_mesh(args.mesh, trace_region_size=64 << 20)
    result = dict(mesh=args.mesh, draws=args.draws, cases=[])
    try:
        tp = mesh.get_num_devices()
        gen = Qwen35Generator.__new__(Qwen35Generator)
        gen.mesh_device, gen.tp, gen.vocab_size = mesh, tp, VOCAB
        gen.local_vocab = -(-VOCAB // tp // 32) * 32
        gen.counters = Counter()
        gen._sampling_mode = None
        gen._replicate = ttnn.ReplicateTensorToMesh(mesh)
        gen.tt_tokens = gen._device(torch.zeros(1, 1, 1, ROWS, dtype=torch.int32), ttnn.uint32)
        gen.sampler = gen._build_sampler()
        host = synthetic_logits(gen.local_vocab, tp)
        logits = ttnn.from_torch(host.reshape(1, 1, ROWS, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
        bf16 = host.to(torch.bfloat16).float()

        def read():
            copies = [ttnn.to_torch(t).reshape(-1)[:ROWS] for t in ttnn.get_device_tensors(gen.tt_tokens)]
            return copies[0].long(), all(torch.equal(copies[0], c) for c in copies[1:])

        greedy = _greedy_params()
        for name, params in (("t1_k32_p1", replace(greedy, temperature=1.0, top_k=32, top_p=1.0)),
                             ("t1_k32_p095", replace(greedy, temperature=1.0, top_k=32, top_p=0.95)),
                             ("t07_k32_p09", replace(greedy, temperature=0.7, top_k=32, top_p=0.9))):
            ref = expected(bf16, params.top_k, params.top_p, params.temperature)
            gen._set_sampling(params)
            for mode in ("eager_fresh", "eager_frozen", "traced_fresh", "traced_frozen", "traced_skip"):
                draws, agree = [], True
                gen.sampler.set_trace_bucket(32 if mode.startswith("traced") else None)
                for _ in range(args.draws):
                    if mode.endswith("fresh"):
                        gen._sample(logits, enable_trace=mode.startswith("traced"), skip_precompile=True)
                    elif mode.endswith("skip"):
                        if not draws:
                            gen.sampler.seed_manager.reset_seed(None, list(range(ROWS)))
                        gen.sampler.seed_manager.get_new_values()
                        gen.sampler.sample(logits, enable_trace=True, tt_out_tok=gen.tt_tokens, skip_precompile=True)
                    else:
                        gen.sampler.sample(logits, enable_trace=mode.startswith("traced"), tt_out_tok=gen.tt_tokens,
                                           skip_precompile=True)
                    tokens, same = read()
                    draws.append(tokens)
                    agree = agree and same
                draws = torch.stack(draws)
                observed = torch.stack([torch.bincount(draws[:, r], minlength=ref.shape[1]).float() / len(draws)
                                        for r in range(ROWS)])
                tv = 0.5 * (observed - ref).abs().sum(-1)
                pooled = torch.bincount(draws.reshape(-1), minlength=ref.shape[1]).float() / draws.numel()
                row = dict(params=name, mode=mode, mean_tv=float(tv.mean()), max_tv=float(tv.max()),
                           pooled_tv=float(0.5 * (pooled - ref[0]).abs().sum()),
                           constant_rows=int((draws == draws[0]).all(0).sum()),
                           outside_support=int((ref.gather(1, draws.T) == 0).sum()), device_copies_agree=agree,
                           expected_top1=float(ref[0].max()), observed_top1_share=float((draws == ref[0].argmax()).float().mean()))
                result["cases"].append(row)
                print(json.dumps(row), flush=True)
    finally:
        args.output.write_text(json.dumps(result, indent=1))
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
