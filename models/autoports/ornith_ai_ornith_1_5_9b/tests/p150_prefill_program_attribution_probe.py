# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reduced one-chip probe: attribute per-prompt-length program compiles to ttnn ops."""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.common.sampling import SamplingParams

from ..tt.functional_decoder import num_blocks_for_context
from ..tt.generator_vllm import TTOrnithForCausalLM
from ..tt.model import close_ornith_mesh, open_ornith_mesh

EVENTS = []
STACK = []
OPS = (
    "slice",
    "pad",
    "lt",
    "embedding",
    "typecast",
    "where",
    "reshape",
    "concat",
    "to_layout",
    "linear",
    "matmul",
    "rms_norm",
    "multiply",
    "copy",
    "clone",
)


def shape_of(value):
    if hasattr(value, "shape"):
        try:
            return list(value.shape)
        except Exception:
            return "?"
    if isinstance(value, (list, tuple)):
        return [shape_of(v) for v in value[:6]]
    if isinstance(value, (int, float, str, bool)):
        return value
    return type(value).__name__


def wrap_ttnn(mesh):
    for name in OPS:
        original = getattr(ttnn, name)

        def make(name=name, original=original):
            def wrapped(*args, **kwargs):
                before = mesh.num_program_cache_entries()
                result = original(*args, **kwargs)
                after = mesh.num_program_cache_entries()
                if after != before:
                    EVENTS.append(
                        {"op": name, "new_programs": after - before, "context": list(STACK), "args": shape_of(args)}
                    )
                return result

            return wrapped

        setattr(ttnn, name, make())


def wrap_everything(mesh):
    """Wrap every callable operation under ttnn, ttnn.experimental and ttnn.transformer."""
    import inspect

    for module, prefix in ((ttnn, ""), (ttnn.experimental, "experimental."), (ttnn.transformer, "transformer.")):
        for name in dir(module):
            if name.startswith("_") or (module is ttnn and name in OPS):
                continue
            try:
                obj = getattr(module, name)
            except Exception:
                continue
            if inspect.isclass(obj) or inspect.ismodule(obj) or not callable(obj):
                continue

            def make(name=prefix + name, original=obj):
                def wrapped(*args, **kwargs):
                    before = mesh.num_program_cache_entries()
                    result = original(*args, **kwargs)
                    after = mesh.num_program_cache_entries()
                    if after != before:
                        EVENTS.append(
                            {"op": name, "new_programs": after - before, "context": list(STACK), "args": shape_of(args)}
                        )
                    return result

                return wrapped

            try:
                setattr(module, name, make())
            except Exception:
                pass


def wrap_methods(obj, names, label):
    for name in names:
        if not hasattr(obj, name):
            continue
        original = getattr(obj, name)

        def make(name=name, original=original):
            def wrapped(*args, **kwargs):
                STACK.append(f"{label}.{name}")
                try:
                    return original(*args, **kwargs)
                finally:
                    STACK.pop()

            return wrapped

        setattr(obj, name, make())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("/home/hous/dev/ornith-1.5-9b/upstream"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", default="128,129,130,131,200,2305")
    args = parser.parse_args()
    os.environ["ORNITH_MODEL_PATH"] = str(args.model_path)
    os.environ["ORNITH_VLLM_LAYER_INDICES"] = os.environ.get("ORNITH_VLLM_LAYER_INDICES", "0,3")
    hf_config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    mesh = open_ornith_mesh(num_devices=1)
    report = {"status": "running", "steps": []}
    adapter = None
    try:
        adapter = TTOrnithForCausalLM.initialize_vllm_model(hf_config, mesh, max_batch_size=32, max_seq_len=262144)
        width = num_blocks_for_context(adapter.max_model_len, adapter.model.page_block_size)
        physical_blocks = width + 32 * 40
        heads = max(1, adapter.model.hf_config.num_key_value_heads // mesh.get_num_devices())
        cache = adapter.allocate_kv_cache(
            (physical_blocks, heads, adapter.model.page_block_size, adapter.model.hf_config.head_dim),
            torch.bfloat16,
            len(adapter.model.layers),
        )
        adapter.warmup_model_prefill(cache)
        adapter.warmup_model_decode(cache)
        wrap_ttnn(mesh)
        model = adapter.model
        wrap_methods(
            model, ("embed", "terminal", "_prefill_chunk_forward", "_prefill_validated", "logits_to_host"), "model"
        )
        for index, layer in enumerate(model.layers):
            wrap_methods(
                layer,
                (
                    "prefill_forward",
                    "_block",
                    "_norm",
                    "_gdn_prefill",
                    "_gdn_project",
                    "_gdn_prefill_conv_fields",
                    "_gdn_gates_projected",
                    "_chunk_delta_rule",
                    "_write_conv_state",
                    "_gdn_out_head_major",
                    "_attention_prefill",
                    "_project_qkv",
                    "_apply_partial_rope",
                    "_attention_output",
                    "_mlp",
                    "_mlp_prefill",
                ),
                f"layer{index}_{layer.kind}",
            )
        gen = adapter.generator
        wrap_methods(
            gen,
            ("_prepare_prompt_sampling", "_sample_prefill_device", "_read_tokens", "_ensure_replay_safe", "_capture"),
            "generator",
        )
        sampling = SamplingParams(temperature=[0.0] * 32, top_k=[1] * 32, top_p=[1.0] * 32, num_logprobs=[-2] * 32)
        table = torch.zeros(32, width, dtype=torch.int32)
        for row in range(32):
            table[row, :40] = torch.arange(1 + row * 40, 1 + (row + 1) * 40)
        for index, length in enumerate(int(v) for v in args.lengths.split(",")):
            EVENTS.clear()
            prompt = torch.arange(17 + index * 7, 17 + index * 7 + length, dtype=torch.int64)[None] % 20000
            before = mesh.num_program_cache_entries()
            captures = gen.counters.get("captures", 0)
            adapter.prefill_forward(
                tokens=prompt,
                prompt_lens=[length],
                empty_slots=[0],
                page_table=table,
                kv_cache=cache,
                sampling_params=sampling,
            )
            step = {
                "length": length,
                "new_programs": mesh.num_program_cache_entries() - before,
                "new_captures": gen.counters.get("captures", 0) - captures,
                "events": list(EVENTS),
            }
            print(json.dumps({k: v for k, v in step.items() if k != "events"}), flush=True)
            for event in step["events"]:
                print("   ", json.dumps(event), flush=True)
            report["steps"].append(step)
        report["status"] = "completed"
    finally:
        if adapter is not None:
            adapter.teardown()
        close_ornith_mesh(mesh)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
