# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch

from .run_sampler_micro import ROWS, VOCAB, synthetic_logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=int, choices=[1, 2, 4], required=True)
    p.add_argument("--draws", type=int, default=32)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import ttnn

    from ..tt.generator import Qwen35Generator, _greedy_params
    from .mesh_util import open_mesh

    mesh = open_mesh(args.mesh)
    result = dict(mesh=args.mesh, draws=args.draws)
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
        gen._set_sampling(replace(_greedy_params(), temperature=1.0, top_k=32, top_p=1.0))
        captured = {}
        original = ttnn.sampling

        def capture(*a, **k):
            captured.update(args=tuple(ttnn.clone(t) for t in a), kwargs=dict(k))
            return original(*a, **k)

        ttnn.sampling = capture
        gen.sampler.set_trace_bucket(None)
        gen._sample(logits, enable_trace=False)
        ttnn.sampling = original
        values, indices = captured["args"]
        kw = {k: v for k, v in captured["kwargs"].items() if k != "output_tensor"}
        first = ttnn.get_device_tensors
        result["values_shape"] = list(values.shape)
        result["indices_shape"] = list(indices.shape)
        result["values_row0_head"] = ttnn.to_torch(first(values)[0]).float()[0, 0, 0, :8].tolist()
        result["values_row5_head"] = ttnn.to_torch(first(values)[0]).float()[0, 0, 5, :8].tolist()
        result["k"] = ttnn.to_torch(first(kw["k"])[0]).reshape(-1)[:4].tolist()
        result["p"] = ttnn.to_torch(first(kw["p"])[0]).float().reshape(-1)[:4].tolist()
        result["temp"] = ttnn.to_torch(first(kw["temp"])[0]).float().reshape(-1)[:4].tolist()
        result["seeds_after_write"] = ttnn.to_torch(first(gen.sampler.tt_sampling.seeds_tt_tensor)[0]).reshape(-1)[:4].tolist()
        user_ids = gen.sampler.tt_sampling.user_ids_tt_tensor

        def run(reseed_each):
            draws = []
            if not reseed_each:
                seeds = ttnn.from_torch(torch.arange(1, ROWS + 1, dtype=torch.int32) * 7919, dtype=ttnn.uint32,
                                        layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
                ttnn.manual_seed(seeds=seeds, user_ids=user_ids)
            for i in range(args.draws):
                if reseed_each:
                    seeds = ttnn.from_torch(torch.randint(1, 2**31, (ROWS,), dtype=torch.int64).to(torch.int32),
                                            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
                    ttnn.manual_seed(seeds=seeds, user_ids=user_ids)
                out = original(values, indices, **kw)
                draws.append(ttnn.to_torch(first(out)[0]).reshape(-1)[:ROWS].long())
            draws = torch.stack(draws)
            return dict(constant_rows=int((draws == draws[0]).all(0).sum()),
                        varying_rows=[r for r in range(ROWS) if not bool((draws[:, r] == draws[0, r]).all())],
                        distinct_per_row=[int(draws[:, r].unique().numel()) for r in range(ROWS)])

        result["seed_once"] = run(False)
        result["reseed_each"] = run(True)
        print(json.dumps(result), flush=True)
    finally:
        args.output.write_text(json.dumps(result, indent=1))
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
