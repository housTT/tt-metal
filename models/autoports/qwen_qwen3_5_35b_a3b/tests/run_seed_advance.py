# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import torch

from .run_sampler_micro import ROWS, expected, synthetic_logits

PROMPT = "My favourite number between 1 and 100 is"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=int, choices=[1, 2, 4], required=True)
    p.add_argument("--mode", choices=["skip", "host"], required=True)
    p.add_argument("--layers", default="0,3")
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--batched", type=int, choices=[0, 1], default=1)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import ttnn

    from ..tt.generator import _greedy_params, build_generator
    from .mesh_util import open_mesh

    mesh = open_mesh(args.mesh)
    result = dict(vars(args), output=str(args.output),
                  env={k: os.environ.get(k) for k in ("TT_METAL_TRACE_ALLOC_TRACKING", "QWEN35_STRICT_CACHE")})
    try:
        gen = build_generator(Path(__file__).parents[1], mesh, decode_batches=(32,),
                              layer_ids=tuple(int(x) for x in args.layers.split(",")),
                              seed_mode=args.mode, batched_decode=bool(args.batched))
        tp = mesh.get_num_devices()
        ids = gen.tokenizer(PROMPT, add_special_tokens=False).input_ids
        table = gen.default_page_table(ROWS, len(ids) + args.steps + 2)
        prompt = torch.tensor([ids] * ROWS)
        first = gen.prefill_forward(prompt, page_table=table, prompt_lens=[len(ids)] * ROWS)
        params = replace(_greedy_params(), temperature=1.0, top_k=32, top_p=1.0)
        gen.decode_forward(first[:, 0].argmax(-1).reshape(ROWS, 1), [len(ids)] * ROWS, page_table=table,
                           sampling_params=params)
        _, logits = gen._decode_trace(ROWS)
        host = synthetic_logits(gen.local_vocab, tp)
        fixed = ttnn.from_torch(host.reshape(1, 1, ROWS, -1), dtype=logits.dtype, layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1))
        ref = expected(host.to(torch.bfloat16).float(), 32, 1.0, 1.0)
        gen.sampler.set_trace_bucket(ROWS)
        draws, agree, seed_values, mismatches = [], True, [], []
        for _ in range(args.steps):
            trace_id, _ = gen._decode_trace(ROWS)
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
            ttnn.copy_host_to_device_tensor(fixed, logits)
            gen._sample(logits, enable_trace=True, skip_precompile=True)
            copies = [ttnn.to_torch(t).reshape(-1)[:ROWS].long() for t in ttnn.get_device_tensors(gen.tt_tokens)]
            agree = agree and all(torch.equal(copies[0], c) for c in copies[1:])
            for d, c in enumerate(copies[1:], 1):
                rows = (copies[0] != c).nonzero().reshape(-1).tolist()
                if rows:
                    mismatches.append(dict(step=len(draws), device=d, rows=rows))
            draws.append(copies[0])
            if len(seed_values) < 4:
                seed_values.append([int(v) & 0xFFFFFFFF for v in
                                    ttnn.to_torch(ttnn.get_device_tensors(gen.sampler.tt_sampling.seeds_tt_tensor)[0])
                                    .reshape(-1)[:4].tolist()])
        draws = torch.stack(draws)
        probs = ref[0]
        collide = float((probs * probs).sum())
        observed = torch.stack([torch.bincount(draws[:, r], minlength=ref.shape[1]).float() / len(draws)
                                for r in range(ROWS)])
        tv = 0.5 * (observed - ref).abs().sum(-1)
        pooled = torch.bincount(draws.reshape(-1), minlength=ref.shape[1]).float() / draws.numel()
        lag = {k: float((draws[k:] == draws[:-k]).float().mean()) for k in (1, 2, 3, 4, 8, 16, 32)}
        same_step = torch.stack([(draws[:, i] == draws[:, j]).float().mean()
                                 for i in range(ROWS) for j in range(i + 1, ROWS)])
        stats = dict(mean_row_tv=float(tv.mean()), max_row_tv=float(tv.max()),
                     pooled_tv=float(0.5 * (pooled - probs).abs().sum()),
                     constant_rows=int((draws == draws[0]).all(0).sum()),
                     outside_support=int((ref.gather(1, draws.T) == 0).sum()),
                     expected_collision=collide, lag_match=lag, max_lag_excess=max(v - collide for v in lag.values()),
                     cross_row_match_mean=float(same_step.mean()), cross_row_match_max=float(same_step.max()),
                     device_copies_agree=agree, mismatch_steps=len({m["step"] for m in mismatches}),
                     mismatch_rows=sum(len(m["rows"]) for m in mismatches), mismatches=mismatches[:64],
                     seed_head_by_step=seed_values)
        result.update(stats=stats, counters=dict(gen.counters), draws=draws.T.tolist())
        print(json.dumps({k: v for k, v in stats.items() if k not in ("seed_head_by_step", "mismatches")}), flush=True)
    finally:
        args.output.write_text(json.dumps(result, indent=1))
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
