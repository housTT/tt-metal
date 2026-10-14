# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import contextlib
import json
from dataclasses import replace
from pathlib import Path

import torch

PROMPTS = (
    "Write a short poem about the sea.",
    "Describe a walk through a forest in autumn.",
    "Invent a name and a backstory for a dragon.",
    "Tell me about your favourite kind of weather and why.",
)
DISTRIBUTION_PROMPT = "My favourite number between 1 and 100 is"
DISTRIBUTION_REPLAYS = 64


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=int, choices=[1, 2, 4], required=True)
    p.add_argument("--layers", default="")
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--freeze-seeds", action="store_true")
    args = p.parse_args()
    import ttnn
    from ttnn._ttnn.operations.trace import get_all_unsafe_tracked_ids, get_unsafe_tracked_ids, remove_unsafe_tracked_id
    from ttnn.tools import trace_allocation_tracker as tracker

    from ..tt.generator import _greedy_params, build_generator
    from .mesh_util import open_mesh

    mesh = open_mesh(args.mesh)
    result = dict(vars(args), output=str(args.output), tracking=tracker.TRACE_ALLOC_TRACKING,
                  tracebacks=tracker.TRACE_ALLOC_DIAGNOSTICS)
    checks = {}
    scopes = []
    gen = None

    @contextlib.contextmanager
    def recorded_scope(device):
        before = set(get_all_unsafe_tracked_ids())
        yield
        new = set(get_all_unsafe_tracked_ids()) - before
        context = {}
        for trace_id, _ in (gen._decode_traces.values() if gen else ()):
            context.update(get_unsafe_tracked_ids(device, trace_id))
        scopes.append(dict(
            bucket=gen.sampler._active_trace_bucket if gen else None,
            buffers=[dict(id=int(i), op=context.get(i, ""),
                          allocated_at=tracker.TraceAllocationTracker._tracebacks.get(i, "")[-1200:])
                     for i in sorted(new)]))
        for i in new:
            remove_unsafe_tracked_id(device, i)

    tracker.corruptible_allocation_scope = recorded_scope
    try:
        layer_ids = tuple(int(x) for x in args.layers.split(",")) if args.layers else None
        gen = build_generator(Path(__file__).parents[1], mesh, layer_ids=layer_ids, decode_batches=(1, 4, 32))
        if args.freeze_seeds:
            gen.sampler.seed_manager.write_device_seed_values = lambda *a, **k: None
        tok = gen.tokenizer

        def chat(text):
            rendered = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                               add_generation_prompt=True)
            return tok(rendered, add_special_tokens=False).input_ids

        prompts = [chat(t) for t in PROMPTS]

        def device_copies_agree(batch):
            copies = [ttnn.to_torch(t).reshape(-1)[:batch] for t in ttnn.get_device_tensors(gen.tt_tokens)]
            return all(torch.equal(copies[0], c) for c in copies[1:])

        def run(rows, params, steps):
            batch = len(rows)
            gen.reset()
            lens = [len(r) for r in rows]
            width = max(lens)
            table = gen.default_page_table(batch, width + steps + 1)
            tokens = torch.zeros(batch, width, dtype=torch.int64)
            for i, r in enumerate(rows):
                tokens[i, : len(r)] = torch.tensor(r)
            first = gen.prefill_forward(tokens, page_table=table, prompt_lens=lens)[:, 0].argmax(-1).tolist()
            gen._write_tokens(first)
            gen._write_positions(batch, lens)
            gen._write_page_table(batch, table)
            out = [[t] for t in first]
            worst_rank, agree = 0, True
            for _ in range(steps):
                gen._set_sampling(params)
                logits = gen._replay(batch, sample=True)
                chosen = gen.read_tokens(batch)
                host = gen._logits_to_host(logits, batch)
                for row in range(batch):
                    rank = int((host[row] > host[row, chosen[row]]).sum())
                    worst_rank = max(worst_rank, rank)
                    out[row].append(int(chosen[row]))
                agree = agree and device_copies_agree(batch)
            return out, worst_rank, agree

        greedy = _greedy_params()
        sampled = replace(greedy, temperature=1.0, top_k=32, top_p=0.95)
        reference = [gen.generate(prompts[0], args.steps + 1)]
        g1, g1_rank, g1_agree = run(prompts[:1], greedy, args.steps)
        checks["greedy_b1_matches_generate"] = g1 == reference
        checks["greedy_b1_rank0"] = g1_rank == 0
        s1, s1_rank, s1_agree = run(prompts[:1], sampled, args.steps)
        checks["sampled_b1_within_top_k"] = s1_rank < 32
        s4, s4_rank, s4_agree = run(prompts, sampled, args.steps)
        checks["sampled_b4_within_top_k"] = s4_rank < 32
        greedy4, g4_rank, g4_agree = run(prompts, greedy, args.steps)
        rows32 = [prompts[i % len(prompts)] for i in range(32)]
        g32, g32_rank, g32_agree = run(rows32, greedy, 16)
        checks["greedy_b32_rows_match_b4"] = all(g32[i] == greedy4[i % 4][:17] for i in range(32))
        again, _, again_agree = run(prompts[:1], greedy, args.steps)
        checks["greedy_b1_after_all_buckets_identical"] = again == g1
        sampled_generate = [gen.generate(prompts[0], args.steps + 1, sampling_params=sampled) for _ in range(2)]
        sampled_equals_greedy = dict(b1=[s1[0] == g1[0]], b4=[s4[i] == greedy4[i] for i in range(4)],
                                     generate=[r == reference[0] for r in sampled_generate],
                                     generate_runs_identical=sampled_generate[0] == sampled_generate[1])
        result["sampled_equals_greedy"] = sampled_equals_greedy
        checks["sampled_not_all_greedy"] = not all(sampled_equals_greedy["b1"] + sampled_equals_greedy["b4"]
                                                   + sampled_equals_greedy["generate"])
        checks["greedy_generate_after_sampled_identical"] = gen.generate(prompts[0], args.steps + 1) == reference[0]
        result["sampled_generate"] = [tok.decode(r) for r in sampled_generate]
        plain = tok(DISTRIBUTION_PROMPT, add_special_tokens=False).input_ids
        gen.reset()
        table = gen.default_page_table(32, len(plain) + 1)
        gen.prefill_forward(torch.tensor([plain[:-1]] * 32), page_table=table, prompt_lens=[len(plain) - 1] * 32)
        gen._write_tokens([plain[-1]] * 32)
        gen._write_positions(32, [len(plain) - 1] * 32)
        gen._write_page_table(32, table)
        logits = gen._replay(32, sample=False)
        host = gen._logits_to_host(logits, 32)
        top = host[0].topk(32)
        expected = torch.zeros(host.shape[-1])
        expected[top.indices] = top.values.softmax(-1)
        gen._set_sampling(replace(greedy, temperature=1.0, top_k=32, top_p=1.0))
        draws, draws_agree = [], True
        for _ in range(DISTRIBUTION_REPLAYS):
            gen.sampler.set_trace_bucket(32)
            gen._sample(logits, enable_trace=True, skip_precompile=True)
            draws.append(gen.read_tokens(32))
            draws_agree = draws_agree and device_copies_agree(32)
        draws = torch.stack(draws)
        observed = torch.bincount(draws.reshape(-1), minlength=host.shape[-1]).float() / draws.numel()
        tv = float(0.5 * (observed - expected).abs().sum())
        constant_rows = int((draws == draws[0]).all(0).sum())
        rows_identical = bool((host == host[0]).all())
        result["distribution"] = dict(
            prompt=DISTRIBUTION_PROMPT, draws=draws.numel(), replays=DISTRIBUTION_REPLAYS,
            expected_top1_prob=float(expected.max()), host_rows_identical=rows_identical,
            total_variation=tv, constant_rows=constant_rows,
            expected_top10={tok.decode([int(i)]): float(expected[i]) for i in top.indices[:10]},
            observed_top10={tok.decode([int(i)]): float(observed[i]) for i in observed.topk(10).indices})
        checks["distribution_rows_vary"] = constant_rows == 0
        checks["distribution_tv_below_0p1"] = tv < 0.1
        kth = float(top.values[-1])
        drawn = draws.reshape(-1).unique()
        outside = [dict(token=int(t), text=tok.decode([int(t)]), logit=float(host[0, t]),
                        rank=int((host[0] > host[0, t]).sum()), count=int((draws == t).sum()))
                   for t in drawn if expected[t] == 0]
        result["distribution"].update(kth_logit=kth, ties_at_kth=int((host[0] == kth).sum()), outside_top_k=outside,
                                      draws_by_row=draws.T.tolist())
        checks["distribution_within_top_k_ties"] = all(o["logit"] >= kth for o in outside)
        checks["device_token_copies_agree"] = all((g1_agree, s1_agree, s4_agree, g4_agree, g32_agree, again_agree,
                                                   draws_agree))
        checks["sampler_trace_keys"] = len(gen.sampler._trace_states)
        checks["decode_captures"] = gen.counters["decode_captures"]
        result.update(
            greedy_b1=tok.decode(g1[0]), sampled_b1=tok.decode(s1[0]),
            sampled_b4=[tok.decode(r) for r in s4], greedy_b4=[tok.decode(r) for r in greedy4],
            worst_rank=dict(greedy_b1=g1_rank, sampled_b1=s1_rank, sampled_b4=s4_rank, greedy_b4=g4_rank,
                            greedy_b32=g32_rank),
            sampler_scope_allocations=scopes, counters=dict(gen.counters), checks=checks)
        result["passed"] = all(v for v in checks.values() if isinstance(v, bool))
        print(json.dumps(dict(checks=checks, passed=result["passed"],
                              scope_buffers=[len(s["buffers"]) for s in scopes])), flush=True)
    finally:
        args.output.write_text(json.dumps(result, indent=1))
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
