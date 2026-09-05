# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Verify stale finite KV isolation before omitting whole-cache clears at new requests."""

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import close_ornith_mesh, open_ornith_mesh

DOC = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--full", action="store_true")
args = parser.parse_args()
mesh = open_ornith_mesh()
rows = []
try:
    gen = build_generator(
        DOC.parents[1],
        mesh,
        layer_indices=None if args.full else [0, 3],
        max_batch_size=1 if args.full else 4,
        cache_context=None if args.full else 4096,
    )
    reset = gen.reset

    def clear_kv(**kwargs):
        reset(clear_kv=True)

    def preserve_kv(**kwargs):
        gen.model.reset_cache(gen.kv_cache, clear_kv=False)
        gen._previous_table = None
        gen._live = False

    def scores():
        return gen.model.logits_to_host(gen._logits, gen.max_batch_size)

    try:
        lengths = [128, 131, 63, 65, 127, 129, 2047, 2049, 3]
        gen.generate([100] * 128, 2, stop_on_eos=False)
        for length in lengths:
            prompts = [
                [(i * 17 + j * 53) % 10000 for i in range(max(1, length - j))] for j in range(gen.max_batch_size)
            ]
            gen.reset = clear_kv
            expected = gen.generate(prompts, 8, stop_on_eos=False)
            expected_logits = scores()
            baseline_perf = dict(gen.perf)
            # All future positions and physical pages start with stale finite KV.
            # Prefill must overwrite every live prefix before decode reads it.
            for pair in gen.kv_cache.kv:
                if pair:
                    for tensor in pair:
                        ttnn.add(tensor, 8192.0, output_tensor=tensor)
            gen.page_table = gen.page_table.flip(1).contiguous()
            gen.reset = preserve_kv
            actual = gen.generate(prompts, 8, stop_on_eos=False)
            assert actual == expected
            assert torch.equal(scores(), expected_logits), length
            rows.append(
                dict(
                    length=length,
                    batch=gen.max_batch_size,
                    exact_tokens=True,
                    exact_final_logits=True,
                    old_ttft_ms=baseline_perf["ttft_s"] * 1000,
                    keep_kv_ttft_ms=gen.perf["ttft_s"] * 1000,
                )
            )
            print(rows[-1], flush=True)
            args.output.write_text(json.dumps(rows, indent=2) + "\n")
    finally:
        gen.reset = reset
        gen.teardown()
finally:
    close_ornith_mesh(mesh)
