# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import time
from pathlib import Path

import torch

from .run_full_evidence import BOOK, _median

STEPS = 16
TIMED = 32


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=int, choices=[1, 2, 4], required=True)
    p.add_argument("--batched", type=int, choices=[0, 1], required=True)
    p.add_argument("--layers", default="")
    p.add_argument("--eager-first", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import ttnn

    from ..tt.generator import _greedy_params, build_generator
    from .mesh_util import open_mesh

    mesh = open_mesh(args.mesh)
    result = dict(vars(args), output=str(args.output),
                  env={k: os.environ.get(k) for k in ("TT_METAL_TRACE_ALLOC_TRACKING", "QWEN35_STRICT_CACHE")})
    tensors = {}
    try:
        layer_ids = tuple(int(x) for x in args.layers.split(",")) if args.layers else None
        gen = build_generator(Path(__file__).parents[1], mesh, decode_batches=(1, 4, 8, 16, 32), layer_ids=layer_ids,
                              batched_decode=bool(args.batched))
        book = gen.tokenizer(BOOK.read_text().strip() * 400, add_special_tokens=False).input_ids
        greedy = _greedy_params()
        lengths = [5 + 37 * r for r in range(32)]
        prompts = [book[50 * r : 50 * r + n] for r, n in enumerate(lengths)]
        table = gen.default_page_table(32, max(lengths) + STEPS + TIMED + 4)

        def run(batch, active, traced=True):
            gen.reset()
            rows = sorted(active)
            ids = torch.zeros(len(rows), max(lengths[r] for r in rows), dtype=torch.int64)
            for i, r in enumerate(rows):
                ids[i, : lengths[r]] = torch.tensor(prompts[r])
            first = gen.prefill_forward(ids, page_table=table[rows], prompt_lens=[lengths[r] for r in rows],
                                        request_slots=rows)[:, 0]
            tokens = torch.zeros(batch, 1, dtype=torch.int64)
            positions = [-1] * batch
            for i, r in enumerate(rows):
                tokens[r, 0] = int(first[i].argmax())
                positions[r] = lengths[r]
            if args.eager_first:
                table_rows = table[:batch].clone()
                for r_, q_ in enumerate(positions):
                    if q_ < 0:
                        table_rows[r_] = gen.kv_pool.scratch_block
                gen._write_tokens(tokens.reshape(-1).tolist())
                gen._write_positions(batch, [max(q_, 0) for q_ in positions])
                gen._write_page_table(batch, table_rows)
                logits = gen._logits_to_host(gen._decode_device(batch), batch).float()
            else:
                logits = gen.decode_forward(tokens, positions, page_table=table[:batch]).float()
            out = {r: [int(tokens[r, 0]), int(logits[r].argmax())] for r in rows}
            tokens = logits.argmax(-1).reshape(batch, 1)
            positions = [q + 1 if q >= 0 else -1 for q in positions]
            chosen = gen.decode_forward(tokens, positions, page_table=table[:batch], sampling_params=greedy).reshape(-1)
            for r in rows:
                out[r].append(int(chosen[r]))
            for _ in range(STEPS - 2):
                gen._set_sampling(greedy)
                gen._replay(batch, sample=True)
                chosen = gen.read_tokens(batch)
                for r in rows:
                    out[r].append(int(chosen[r]))
            return out, logits[rows]

        cases = []
        for batch, active in ((4, list(range(4))), (8, list(range(8))), (16, list(range(16))), (32, list(range(32))), (4, [0, 2])):
            out, logits = run(batch, active)
            key = f"b{batch}_" + "_".join(map(str, active)) if len(active) < batch else f"b{batch}"
            tensors[key] = logits
            cases.append(dict(case=key, tokens={str(r): v for r, v in out.items()}))
            print(json.dumps(dict(case=key, first_tokens={str(r): v[:4] for r, v in list(out.items())[:4]})), flush=True)

        timing = {}
        for batch in (1, 4, 8, 16, 32):
            runs = []
            for _ in range(3):
                gen.reset()
                rows = list(range(batch))
                ids = torch.tensor([book[r * 128 : (r + 1) * 128] for r in rows])
                first = gen.prefill_forward(ids, page_table=table[:batch], prompt_lens=[128] * batch,
                                            request_slots=rows)[:, 0]
                gen.decode_forward(first.argmax(-1).reshape(batch, 1), [128] * batch, page_table=table[:batch],
                                   sampling_params=greedy)
                t0 = time.perf_counter()
                for _ in range(TIMED):
                    gen._set_sampling(greedy)
                    gen._replay(batch, sample=True)
                    gen.read_tokens(batch)
                runs.append((time.perf_counter() - t0) / TIMED * 1e3)
            timing[str(batch)] = dict(step_ms=_median(runs), runs_step_ms=runs,
                                      aggregate_tokens_per_second=batch * 1e3 / _median(runs))
        result.update(cases=cases, timing=timing, counters=dict(gen.counters))
        print(json.dumps(dict(timing={b: round(t["step_ms"], 2) for b, t in timing.items()})), flush=True)
    finally:
        args.output.write_text(json.dumps(result, indent=1))
        if tensors:
            torch.save(tensors, args.output.with_suffix(".pt"))
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
