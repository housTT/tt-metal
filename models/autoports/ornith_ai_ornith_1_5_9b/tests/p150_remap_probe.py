# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Slot remap probe: arbitrary permutations after warmup move recurrent/conv rows exactly with zero program growth."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn

from ..tt.generator_vllm import TTOrnithForCausalLM
from ..tt.model import close_ornith_mesh, open_ornith_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("/home/hous/dev/ornith-1.5-9b/upstream"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    os.environ["ORNITH_MODEL_PATH"] = str(args.model_path)
    hf_config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    mesh = open_ornith_mesh(num_devices=1)
    report = {"status": "running", "cases": []}
    adapter = None
    try:
        adapter = TTOrnithForCausalLM.initialize_vllm_model(
            hf_config, mesh, max_batch_size=args.batch, max_seq_len=262144
        )
        model = adapter.model
        blocks = int(os.environ.get("ORNITH_VLLM_POOL_TOKENS", "262208")) // model.page_block_size
        heads = max(1, model.hf_config.num_key_value_heads // mesh.get_num_devices())
        cache = adapter.allocate_kv_cache(
            (blocks, heads, model.page_block_size, model.hf_config.head_dim), torch.bfloat16, len(model.layers)
        )
        adapter.warmup_model_prefill(cache)
        adapter.warmup_model_decode(cache)
        gen = adapter.generator
        layers = [layer for layer in cache.decode_layers if not layer.is_full_attention]
        generator = torch.Generator().manual_seed(7)
        for layer in layers:
            state = torch.randn(tuple(int(d) for d in layer.recurrent_state.shape), generator=generator)
            ttnn.copy(model.upload(state, dtype=ttnn.float32), layer.recurrent_state)
            for buf in layer.conv_state:
                ttnn.copy(
                    model.upload(
                        torch.randn(tuple(int(d) for d in buf.shape), generator=generator).to(torch.bfloat16),
                        dtype=ttnn.bfloat16,
                    ),
                    buf,
                )
        b = args.batch
        cases = {
            "rotate_left": [*range(1, b), 0],
            "swap_3_17": [17 if i == 3 else 3 if i == 17 else i for i in range(b)],
            "reverse": list(range(b - 1, -1, -1)),
            "three_cycles_from_5": [i for i in range(b)],
            "random": torch.randperm(b, generator=generator).tolist(),
            "identity": list(range(b)),
        }
        c = cases["three_cycles_from_5"]
        for cycle in ([5, 9, 13], [20, 21], [30, 31, 29, 28]):
            for a_, b_ in zip(cycle, cycle[1:] + cycle[:1]):
                c[a_] = b_
        for name, values in cases.items():
            before = {
                id(layer): (
                    ttnn.to_torch(layer.recurrent_state).float(),
                    [ttnn.to_torch(x).float() for x in layer.conv_state],
                )
                for layer in layers
            }
            programs = mesh.num_program_cache_entries()
            captures = gen.counters.get("captures", 0)
            t0 = time.perf_counter()
            moved = gen.remap_serving_slots(torch.tensor(values, dtype=torch.int64))
            ttnn.synchronize_device(mesh)
            elapsed = time.perf_counter() - t0
            exact = True
            max_diff = 0.0
            for layer in layers:
                rec, conv = before[id(layer)]
                after_rec = ttnn.to_torch(layer.recurrent_state).float()
                expected = rec[values]
                exact &= bool(torch.equal(after_rec, expected))
                max_diff = max(max_diff, float((after_rec - expected).abs().max()))
                for buf, old in zip(layer.conv_state, conv):
                    got = ttnn.to_torch(buf).float()
                    exact &= bool(torch.equal(got, old[values]))
                    max_diff = max(max_diff, float((got - old[values]).abs().max()))
            row = {
                "case": name,
                "values": values,
                "layers_moved": moved,
                "elapsed_s": elapsed,
                "programs_added": mesh.num_program_cache_entries() - programs,
                "captures_added": gen.counters.get("captures", 0) - captures,
                "exact": exact,
                "max_abs_diff": max_diff,
            }
            print(json.dumps({k: v for k, v in row.items() if k != "values"}), flush=True)
            report["cases"].append(row)
        report["layer_indices"] = os.environ.get("ORNITH_VLLM_LAYER_INDICES")
        report["programs_end"] = int(mesh.num_program_cache_entries())
        report["status"] = "completed"
        report["all_exact_no_growth"] = all(
            r["exact"] and r["programs_added"] == 0 and r["captures_added"] == 0 for r in report["cases"]
        )
        print("all_exact_no_growth", report["all_exact_no_growth"], flush=True)
    finally:
        if adapter is not None:
            adapter.teardown()
        close_ornith_mesh(mesh)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
