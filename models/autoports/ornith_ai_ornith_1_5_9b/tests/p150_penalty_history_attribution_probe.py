# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reduced one-chip probe: which ttnn op compiles when penalized rows resume with host histories of new widths."""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig

from models.common.sampling import SamplingParams

from ..tt.functional_decoder import num_blocks_for_context
from ..tt.generator_vllm import TTOrnithForCausalLM
from ..tt.model import close_ornith_mesh, open_ornith_mesh
from .p150_prefill_program_attribution_probe import EVENTS, wrap_everything, wrap_methods, wrap_ttnn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("/home/hous/dev/ornith-1.5-9b/upstream"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["ORNITH_MODEL_PATH"] = str(args.model_path)
    os.environ["ORNITH_VLLM_LAYER_INDICES"] = os.environ.get("ORNITH_VLLM_LAYER_INDICES", "0,3")
    os.environ["ORNITH_VLLM_ALLOW_HOST_SAMPLING"] = "1"
    hf_config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    mesh = open_ornith_mesh(num_devices=1)
    report = {"status": "running", "steps": []}
    adapter = None
    B = 32
    try:
        adapter = TTOrnithForCausalLM.initialize_vllm_model(hf_config, mesh, max_batch_size=B, max_seq_len=262144)
        width = num_blocks_for_context(adapter.max_model_len, adapter.model.page_block_size)
        cache = adapter.allocate_kv_cache(
            (width + B * 40, 4, 64, adapter.model.hf_config.head_dim), torch.bfloat16, len(adapter.model.layers)
        )
        adapter.warmup_model_prefill(cache)
        adapter.warmup_model_decode(cache)
        wrap_ttnn(mesh)
        wrap_everything(mesh)
        gen = adapter.generator
        wrap_methods(
            gen,
            (
                "configure_sampling",
                "_configure_sampling",
                "remap_serving_slots",
                "_remap_serving_rows",
                "_remap_serving_vector",
                "refresh_serving_inputs",
                "_prepare_prompt_sampling",
                "_sample_prefill_device",
                "_capture",
            ),
            "generator",
        )
        wrap_methods(gen.sampling, ("reset_sampling_params", "reset_prompt_tokens", "reset_output_state"), "sampling")
        wrap_methods(
            gen.sampling.tt_penalties,
            ("reset_params", "reset_prompt_tokens", "reset_output_tokens", "update_output_tokens"),
            "penalties",
        )
        table = torch.zeros(B, width, dtype=torch.int32)
        for row in range(B):
            table[row, :40] = torch.arange(1 + row * 40, 1 + (row + 1) * 40)

        def params(**kw):
            base = dict(temperature=[0.0] * B, top_k=[1] * B, top_p=[1.0] * B, num_logprobs=[-2] * B)
            base.update(kw)
            return SamplingParams(**base)

        def run(label, rows, lengths, sampling, decode_kwargs=None, steps=2):
            EVENTS.clear()
            before = mesh.num_program_cache_entries()
            prompts = [torch.arange(17 + i * 11, 17 + i * 11 + n, dtype=torch.int64) for i, n in enumerate(lengths)]
            first = adapter.prefill_forward(
                tokens=prompts,
                prompt_lens=lengths,
                empty_slots=rows,
                page_table=table,
                kv_cache=cache,
                sampling_params=sampling,
            )
            first = first[0] if isinstance(first, tuple) else first
            tokens = torch.zeros(B, 1, dtype=torch.int64)
            positions = torch.full((B,), -1, dtype=torch.int64)
            for i, row in enumerate(rows):
                tokens[row, 0] = int(first.reshape(-1)[i])
                positions[row] = lengths[i]
            for step in range(steps):
                kw = dict(
                    tokens=tokens,
                    start_pos=positions,
                    page_table=table,
                    kv_cache=cache,
                    sampling_params=sampling,
                    reset_batch=step == 0,
                    read_from_device=True,
                )
                if decode_kwargs and step == 0:
                    kw.update(decode_kwargs)
                adapter.decode_forward(**kw)
            row = {
                "label": label,
                "new_programs": mesh.num_program_cache_entries() - before,
                "captures": gen.counters.get("captures"),
                "events": list(EVENTS),
            }
            print(json.dumps({k: row[k] for k in ("label", "new_programs", "captures")}), flush=True)
            for e in row["events"]:
                print("   ", json.dumps(e)[:300], flush=True)
            report["steps"].append(row)

        run("greedy_9rows", list(range(1, 10)), [6, 3, 3, 3, 3, 3, 7, 3, 4], params())
        pen = [0.0] * B
        for r in range(1, 10):
            pen[r] = 0.5 if r % 2 else 0.0
        run("mixed_penalties_9rows", list(range(1, 10)), [6, 3, 3, 3, 3, 3, 7, 3, 4], params(presence_penalty=pen))
        freq = [0.0] * B
        freq[2] = 2.0
        run("frequency_row2", [2], [5], params(frequency_penalty=freq), steps=6)
        for width_h in (3, 9, 17, 40):
            prompt_hist = [[1, 2, 3][: min(3, width_h)] for _ in range(B)]
            out_hist = [list(range(5, 5 + width_h)) for _ in range(B)]
            adapter._device_rows[:] = False
            run(
                f"resume_histories_width{width_h}",
                [3],
                [4],
                params(presence_penalty=pen),
                decode_kwargs=dict(prompt_tokens=prompt_hist, output_tokens=out_hist),
                steps=2,
            )
        remap = torch.tensor([*range(1, B), 0], dtype=torch.int64)
        run("remap_rotation", [4], [4], params(presence_penalty=pen), decode_kwargs=dict(slot_remap=remap), steps=2)
        report["status"] = "completed"
    finally:
        if adapter is not None:
            adapter.teardown()
        close_ornith_mesh(mesh)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
