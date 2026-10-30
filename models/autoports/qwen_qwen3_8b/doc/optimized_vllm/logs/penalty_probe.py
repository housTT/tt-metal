# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-penalties probe at the generator level, driving the plugin's call pattern.

usage: penalty_probe.py <output.json> [--layers 2] [--steps 24]

Prefill with device sampling, then decode steps with ``prompt_tokens`` / ``output_tokens`` passed the way the
plugin does (every step while penalties are active, ``reset_batch`` on the first), for: greedy, greedy plus a
strong repetition penalty, presence penalty, frequency penalty, and a mixed batch (rows with and without
penalties); records the token sequences, the sampler counters, the trace captures and the program-cache
entry count around the decode steps (after ``capture_decode_traces``).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8b.tt.generator import build_generator, greedy_sampling_params
from models.common.sampling import SamplingParams

MODEL_DIR = Path("models/autoports/qwen_qwen3_8b")


def run_case(gen, prompts, params, steps):
    gen.reset()
    rows = len(prompts)
    lengths = [len(p) for p in prompts]
    tokens = torch.zeros(rows, max(lengths), dtype=torch.int64)
    for i, ids in enumerate(prompts):
        tokens[i, : len(ids)] = torch.tensor(ids)
        gen._ensure_blocks(i, len(ids) + steps)
    table = gen._host_page_table(list(range(rows)))
    before = gen.counter_snapshot()
    pc0 = gen.mesh_device.num_program_cache_entries()
    first = gen.prefill_forward(tokens, page_table=table, kv_cache=gen.kv_cache, prompt_lens=lengths, sampling_params=params)
    generated = [[int(first[i])] for i in range(rows)]
    positions = torch.tensor(lengths, dtype=torch.int32)
    width = max(lengths)
    prompt_tokens = torch.full((32, width), -1, dtype=torch.int32)
    for i, ids in enumerate(prompts):
        prompt_tokens[i, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
    for step in range(1, steps):
        next_tokens = torch.tensor([[generated[i][-1]] for i in range(rows)])
        output_tokens = torch.full((32, step), -1, dtype=torch.int32)
        for i in range(rows):
            output_tokens[i] = torch.tensor(generated[i], dtype=torch.int32)
        out = gen.decode_forward(
            next_tokens, positions, page_table=table, kv_cache=gen.kv_cache, sampling_params=params,
            reset_batch=step == 1, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        )
        for i in range(rows):
            generated[i].append(int(out[i]))
        positions = positions + 1
    pc1 = gen.mesh_device.num_program_cache_entries()
    delta = {k: gen.counter_snapshot()[k] - before[k] for k in before}
    return {"generated": generated, "text": [gen.tokenizer.decode(g) for g in generated], "counters_delta": delta, "program_cache": [pc0, pc1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--layers", type=int, default=int(os.getenv("QWEN3_FULL_MODEL_LAYERS", "2")))
    ap.add_argument("--steps", type=int, default=24)
    args = ap.parse_args()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=512 * 1024 * 1024)
    kwargs = {"num_layers": args.layers}
    if args.layers != 36:
        kwargs.update(decode_buckets=(1, 8, 32), warmup_prefill_lengths=(32, 64, 128, 256), num_blocks=512)
    gen = build_generator(str(MODEL_DIR), mesh, **kwargs)
    captured = gen.capture_decode_traces(gen.kv_cache)
    tok = gen.tokenizer
    prompt = tok.encode("The quick brown fox jumps over the lazy dog. The quick brown fox", add_special_tokens=False)
    prompt_b = tok.encode("Write the alphabet: a a a a a a a a a a a", add_special_tokens=False)
    steps = args.steps
    cases = {
        "greedy": run_case(gen, [prompt], greedy_sampling_params(1), steps),
        "greedy_repetition_2": run_case(gen, [prompt], SamplingParams(temperature=[0.0], top_k=[1], top_p=[1.0], repetition_penalty=[2.0]), steps),
        "greedy_presence_2": run_case(gen, [prompt], SamplingParams(temperature=[0.0], top_k=[1], top_p=[1.0], presence_penalty=[2.0]), steps),
        "greedy_frequency_2": run_case(gen, [prompt_b], SamplingParams(temperature=[0.0], top_k=[1], top_p=[1.0], frequency_penalty=[2.0]), steps),
        "greedy_again": run_case(gen, [prompt], greedy_sampling_params(1), steps),
        "mixed_8": run_case(
            gen,
            [prompt, prompt] * 4,
            SamplingParams(temperature=[0.0] * 8, top_k=[1] * 8, top_p=[1.0] * 8, repetition_penalty=[1.0, 2.0] * 4),
            steps,
        ),
        "seeded_mixed_8": run_case(
            gen,
            [prompt, prompt] * 4,
            SamplingParams(temperature=[0.7] * 8, top_k=[20] * 8, top_p=[0.95] * 8, presence_penalty=[0.0, 2.0] * 4, seed=[42] * 8),
            steps,
        ),
    }
    summary = {
        "greedy_equals_greedy_again": cases["greedy"]["generated"] == cases["greedy_again"]["generated"],
        "repetition_changes_output": cases["greedy"]["generated"] != cases["greedy_repetition_2"]["generated"],
        "presence_changes_output": cases["greedy"]["generated"] != cases["greedy_presence_2"]["generated"],
        "mixed_no_penalty_rows_equal_greedy": all(cases["mixed_8"]["generated"][i] == cases["greedy"]["generated"][0] for i in (0, 2, 4, 6)),
        "mixed_penalty_rows_equal_solo_penalty": all(cases["mixed_8"]["generated"][i] == cases["greedy_repetition_2"]["generated"][0] for i in (1, 3, 5, 7)),
        "seeded_mixed_groups_deterministic": len({tuple(cases["seeded_mixed_8"]["generated"][i]) for i in (0, 2, 4, 6)}) == 1 and len({tuple(cases["seeded_mixed_8"]["generated"][i]) for i in (1, 3, 5, 7)}) == 1,
        "program_cache_flat_across_cases": len({tuple(c["program_cache"]) for c in cases.values()}) == 1,
    }
    payload = {"layers": args.layers, "steps": steps, "captured": {str(k): str(v) for k, v in captured.items()}, "cases": cases, "summary": summary, "visible_devices": os.getenv("TT_VISIBLE_DEVICES")}
    print(json.dumps(summary, indent=1), flush=True)
    for name, c in cases.items():
        print(name, c["program_cache"], {k: v for k, v in c["counters_delta"].items() if v}, repr(c["text"][0][:80]), flush=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    gen.teardown()
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
