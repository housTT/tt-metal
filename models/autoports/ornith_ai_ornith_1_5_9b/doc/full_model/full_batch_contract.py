# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""All32 layers, all32 active requests, mixed logical lengths and fixed slots."""

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import close_ornith_mesh, open_ornith_mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mesh = open_ornith_mesh()
    report = dict(layers=32, batch=32, cache_context=2048)
    try:
        gen = build_generator(Path(__file__).resolve().parents[2], mesh, max_batch_size=32, cache_context=2048)
        try:
            unique = [list(range(131)), list(range(127)), [31, 57, 88]]
            prompts = [unique[i % 3] for i in range(32)]
            tokens = gen.generate(prompts, 8, stop_on_eos=False)
            logits = gen.model.logits_to_host(gen._logits, 32)
            assert len(tokens) == 32 and all(len(row) == 8 for row in tokens)
            for row in range(32):
                assert tokens[row] == tokens[row % 3], (row, tokens[row], tokens[row % 3])
                assert torch.equal(logits[row], logits[row % 3]), (
                    row,
                    float((logits[row] - logits[row % 3]).abs().max()),
                    "cross-slot logits differ",
                )
            expected = [len(prompt) + 7 for prompt in prompts]
            positions = ttnn.to_torch(ttnn.get_device_tensors(gen._inputs[1])[0]).tolist()
            assert positions == expected
            counters = gen.perf["loop_counters"]
            for name in ("token_refreshes", "position_refreshes", "rope_refreshes", "page_table_refreshes"):
                assert counters[name] == 0
            assert counters["model_replays"] == counters["sampling_replays"] == 7
            gen.page_table = gen.page_table.flip(1).contiguous()
            repeated = gen.generate(prompts, 8, stop_on_eos=False)
            assert repeated == tokens, "page permutation or repeated full-batch generation changed tokens"
            repeated_logits = gen.model.logits_to_host(gen._logits, 32)
            assert torch.equal(logits, repeated_logits), "page permutation changed exact logits"
            report.update(
                tokens=tokens,
                positions=positions,
                loop_counters=counters,
                repeated_permuted_table_equal=True,
                exact_cross_slot_logits=True,
                exact_permuted_logits=True,
                perf=gen.perf,
                passed=True,
            )
        finally:
            gen.teardown()
    finally:
        close_ornith_mesh(mesh)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
