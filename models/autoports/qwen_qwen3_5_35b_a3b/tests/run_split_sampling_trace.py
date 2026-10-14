# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=int, choices=[1, 2, 4], required=True)
    p.add_argument("--layers", default="0,3")
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import ttnn

    from ..tt.generator import _greedy_params, build_generator
    from .mesh_util import open_mesh

    mesh = open_mesh(args.mesh)
    result = dict(vars(args), output=str(args.output))
    checks = {}
    try:
        layer_ids = tuple(int(x) for x in args.layers.split(",")) if args.layers else None
        gen = build_generator(Path(__file__).parents[1], mesh, layer_ids=layer_ids)
        text = gen.tokenizer.apply_chat_template([{"role": "user", "content": "Name three prime numbers."}],
                                                 tokenize=False, add_generation_prompt=True)
        ids = gen.tokenizer(text, add_special_tokens=False).input_ids
        length = len(ids)

        def read_positions():
            return int(ttnn.to_torch(ttnn.get_device_tensors(gen.tt_positions[1])[0]).reshape(-1)[0])

        def stepwise(table, params):
            (logits,) = gen._prefill_rows(torch.tensor([ids]), table, [length], [0], [0], False)
            gen._set_sampling(params)
            gen.sampler.set_trace_bucket(None)
            gen.sampler.sample(gen._sampler_logits(logits, 1), enable_trace=False, tt_out_tok=gen.tt_tokens)
            logits.deallocate(True)
            gen._write_page_table(1, table)
            gen._write_positions(1, [length])
            tokens, positions = [int(gen.read_tokens(1)[0])], [read_positions()]
            before = {k: gen.counters[k] for k in ("token_refreshes", "position_refreshes", "page_table_refreshes")}
            for _ in range(args.steps):
                gen._set_sampling(params)
                gen._replay(1, sample=True)
                tokens.append(int(gen.read_tokens(1)[0]))
                positions.append(read_positions())
            after = {k: gen.counters[k] for k in before}
            return tokens, positions, {k: after[k] - before[k] for k in before}

        greedy = _greedy_params()
        table_a = gen.default_page_table(1, length + args.steps + 1)
        blocks = table_a.shape[1]
        table_b = torch.arange(gen.kv_pool.num_blocks - blocks, gen.kv_pool.num_blocks, dtype=torch.int32)
        table_b = table_b.flip(0).reshape(1, blocks)

        tokens_a, positions_a, refresh_a = stepwise(table_a, greedy)
        trace_ids = {b: str(t[0]) for b, t in gen._decode_traces.items()}
        checks["positions_advance_on_device"] = positions_a == list(range(length, length + args.steps + 1))
        checks["no_host_refresh_in_loop"] = all(v == 0 for v in refresh_a.values())
        reference = gen.generate(ids, args.steps + 1)
        checks["stepwise_matches_generate"] = tokens_a == reference
        before = gen.counters["page_table_refreshes"]
        tokens_same, _, _ = stepwise(table_a, greedy)
        checks["unchanged_table_no_refresh"] = gen.counters["page_table_refreshes"] == before
        checks["unchanged_table_same_tokens"] = tokens_same == tokens_a
        before = gen.counters["page_table_refreshes"]
        tokens_b, _, _ = stepwise(table_b, greedy)
        checks["changed_table_refreshed_once"] = gen.counters["page_table_refreshes"] == before + 1
        checks["changed_table_same_tokens"] = tokens_b == tokens_a
        sampled = replace(greedy, temperature=0.8, top_k=32, top_p=0.95)
        tokens_s, _, _ = stepwise(table_a, sampled)
        checks["sampled_tokens_valid"] = all(0 <= t < gen.vocab_size for t in tokens_s)
        tokens_g2, _, _ = stepwise(table_a, greedy)
        checks["greedy_after_sampled_identical"] = tokens_g2 == tokens_a
        checks["decode_trace_ids_stable"] = {b: str(t[0]) for b, t in gen._decode_traces.items()} == trace_ids
        checks["decode_captures_once"] = gen.counters["decode_captures"] == 1
        checks["sampler_trace_keys"] = len(gen.sampler._trace_states)
        result.update(tokens_greedy=tokens_a, tokens_sampled=tokens_s, positions=positions_a,
                      loop_refreshes=refresh_a, counters=dict(gen.counters), checks=checks)
        result["passed"] = all(v for k, v in checks.items() if isinstance(v, bool))
        print(json.dumps(dict(checks=checks, passed=result["passed"])), flush=True)
    finally:
        args.output.write_text(json.dumps(result, indent=1))
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
