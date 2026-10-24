# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Find which batched-prefill inputs still compile programs after preparation (two-layer model)."""

from __future__ import annotations

import json
import os
import sys

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.test_full_model_contract import _config, _load_probe_state
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm import PRECISION_CONFIG
from models.autoports.google_gemma_4_26b_a4b_it.tt.model import Gemma4FullModel


def main() -> int:
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        model = Gemma4FullModel(
            mesh_device=mesh,
            hf_config=_config(),
            state_dict=_load_probe_state(full_stack=False),
            max_seq_len=4096,
            max_batch_size=16,
            layer_indices=[0, 5],
            tensor_cache_path=os.environ.get("GEMMA4_TENSOR_CACHE_PATH", "/tmp/gemma4_growth_cache"),
            create_kv_cache=True,
            precision_config_path=PRECISION_CONFIG,
        )
        gen = Gemma4Generator(model, tokenizer=None, sampling_mode="device")
        state = model.state
        gen.prepare_batched_prefill(state)
        print(
            json.dumps(
                {
                    "warmup_entries": gen.batched_prefill_warmup["program_cache_entries_after"],
                    "fill_steps": len(gen.batched_prefill_warmup["fill_length_steps"]),
                }
            ),
            flush=True,
        )
        rng = torch.Generator().manual_seed(0)

        def run(label, lens):
            gen.reset()
            width = max(((l + 31) // 32) * 32 for l in lens)
            tokens = torch.zeros((len(lens), width), dtype=torch.long)
            for row, l in enumerate(lens):
                tokens[row, :l] = torch.randint(1, 262_000, (l,), generator=rng)
            before = mesh.num_program_cache_entries()
            logits = gen.prefill_forward(tokens, page_table=state.page_tables, kv_cache=state, prompt_lens=lens)
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(logits)
            growth = mesh.num_program_cache_entries() - before
            groups = gen.batched_prefill_group_growth[-1] if gen.batched_prefill_group_growth else None
            print(json.dumps({"label": label, "lens": lens, "growth": growth, "last_group": groups}), flush=True)

        run("same_as_warm_4x128", [128, 128, 128, 128])
        run("rows01_len5", [5, 5])
        run("rows01_len5_repeat", [5, 5])
        run("rows0123_len5", [5, 5, 5, 5])
        run("rows0123_len5_repeat", [5, 5, 5, 5])
        run("rows0123_len21", [21, 21, 21, 21])
        run("rows0123_len64", [64, 64, 64, 64])
        run("rows0123_len65", [65, 65, 65, 65])
        run("rows0123_mixed", [5, 21, 32, 25])
        run("rows0123_mixed_desc", [33, 21, 31, 25])
        run("rows0123_mixed_desc_repeat", [33, 21, 31, 25])
        run("rows0to7_len5", [5] * 8)
        run("rows0to7_len100", [100] * 8)
    finally:
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
