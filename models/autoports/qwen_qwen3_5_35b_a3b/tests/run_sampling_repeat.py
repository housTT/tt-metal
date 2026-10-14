# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from .run_sampling_audit import PROMPTS


def nucleus_prob(row, token, temperature, top_k, top_p):
    top = row.float().topk(top_k)
    probs = (top.values / temperature).softmax(-1)
    keep = torch.cumsum(probs, -1) - probs < top_p
    keep[0] = True
    kept = probs * keep
    kept = kept / kept.sum()
    hit = (top.indices == token).nonzero()
    return float(kept[hit[0, 0]]) if len(hit) else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=int, choices=[1, 2, 4], required=True)
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import ttnn

    from ..tt.generator import _greedy_params, build_generator
    from .mesh_util import open_mesh

    mesh = open_mesh(args.mesh)
    result = dict(vars(args), output=str(args.output))
    try:
        gen = build_generator(Path(__file__).parents[1], mesh, decode_batches=(1, 4, 32))
        tok = gen.tokenizer
        prompts = [tok(tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False,
                                               add_generation_prompt=True), add_special_tokens=False).input_ids
                   for t in PROMPTS]
        greedy = _greedy_params()
        sampled = replace(greedy, temperature=1.0, top_k=32, top_p=0.95)

        def run(params):
            batch = len(prompts)
            gen.reset()
            lens = [len(r) for r in prompts]
            table = gen.default_page_table(batch, max(lens) + args.steps + 1)
            tokens = torch.zeros(batch, max(lens), dtype=torch.int64)
            for i, r in enumerate(prompts):
                tokens[i, : len(r)] = torch.tensor(r)
            first = gen.prefill_forward(tokens, page_table=table, prompt_lens=lens)[:, 0].argmax(-1).tolist()
            gen._write_tokens(first)
            gen._write_positions(batch, lens)
            gen._write_page_table(batch, table)
            out = [[t] for t in first]
            probs = [[] for _ in range(batch)]
            for _ in range(args.steps):
                gen._set_sampling(params)
                logits = gen._replay(batch, sample=True)
                chosen = gen.read_tokens(batch)
                host = gen._logits_to_host(logits, batch)
                for row in range(batch):
                    out[row].append(int(chosen[row]))
                    probs[row].append(nucleus_prob(host[row], int(chosen[row]), 1.0, 32, 0.95))
            return out, probs

        g, g_probs = run(greedy)
        p_repeat = [float(torch.tensor(r).prod()) for r in g_probs]
        result["tt_prob_sampled_row_equals_greedy"] = p_repeat
        result["tt_prob_all_rows_equal_greedy"] = float(torch.tensor(p_repeat).prod())
        runs = []
        for _ in range(args.repeats):
            s, _ = run(sampled)
            runs.append([s[r] == g[r] for r in range(len(prompts))])
        result["sampled_rows_equal_greedy"] = runs
        result["observed_row_equal_rate"] = [sum(r[i] for r in runs) / len(runs) for i in range(len(prompts))]
        result["runs_all_rows_equal_greedy"] = sum(all(r) for r in runs)
        result["counters"] = dict(gen.counters)
        print(json.dumps({k: v for k, v in result.items() if k != "counters"}), flush=True)
    finally:
        args.output.write_text(json.dumps(result, indent=1))
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
