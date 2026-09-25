# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Generate layer-boundary HF activations for the 125-token regression."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_flash_next.demo.generate_hf_reference import _load_oracle
from models.autoports.qwen_qwen3_8_flash_next.demo.generate_prefill_regression_reference import _render
from models.autoports.qwen_qwen3_8_flash_next.tests import harness as H
from models.autoports.qwen_qwen3_8_flash_next.tt.model import MODEL_ID, MODEL_REVISION


@torch.inference_mode()
def generate(output: Path, *, expert_cache_capacity: int = 32, threads: int | None = None) -> dict:
    if threads is not None:
        torch.set_num_threads(threads)
    H.import_target_transformers()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(H.MODEL_SNAPSHOT, local_files_only=True)
    messages, rendered, prompt = _render(tokenizer)
    load_started = time.perf_counter()
    model, ple_store, experts = _load_oracle(H.MODEL_SNAPSHOT, expert_cache_capacity=expert_cache_capacity)
    load_seconds = time.perf_counter() - load_started
    ple_store.reset_request("hf-aime24")
    layer_outputs: list[torch.Tensor | None] = [None] * int(model.config.num_hidden_layers)
    final_outputs: list[torch.Tensor] = []
    layer0_boundaries: dict[str, object] = {}

    def to_host(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().to(torch.bfloat16).contiguous()
        if isinstance(value, tuple):
            return tuple(to_host(item) for item in value)
        return value

    def capture_layer(index):
        def hook(_module, _inputs, output):
            layer_outputs[index] = output.detach().cpu().to(torch.bfloat16).contiguous()

        return hook

    hooks = [layer.register_forward_hook(capture_layer(index)) for index, layer in enumerate(model.model.layers)]
    hooks.append(
        model.model.hyper_connection_mixer.register_forward_hook(
            lambda _module, _inputs, output: final_outputs.append(
                output.detach().cpu().to(torch.bfloat16).contiguous()
            )
        )
    )
    layer0 = model.model.layers[0]
    for name, module in (
        ("attn_hyper_mix", layer0.attn_hyper_connection),
        ("gdn", layer0.linear_attn),
        ("mlp_hyper_mix", layer0.mlp_hyper_connection),
        ("router", layer0.mlp.gate),
        ("routed_experts", layer0.mlp.experts),
        ("shared_expert", layer0.mlp.shared_expert),
        ("moe", layer0.mlp),
    ):
        hooks.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, boundary=name: layer0_boundaries.__setitem__(
                    boundary, to_host(output)
                )
            )
        )
    started = time.perf_counter()
    try:
        result = model(input_ids=prompt, use_cache=False, logits_to_keep=1, return_dict=True)
    finally:
        for hook in hooks:
            hook.remove()
    prefill_seconds = time.perf_counter() - started
    embedding = model.model.embed_tokens(prompt).repeat(1, 1, int(model.config.hc_count))
    embedding = embedding.detach().cpu().to(torch.bfloat16).contiguous()
    if any(value is None for value in layer_outputs) or len(final_outputs) != 1:
        raise RuntimeError("HF boundary hooks did not capture every decoder layer and the final mixer")
    hidden_states = tuple(layer_outputs)
    final_hidden = final_outputs[0]
    embedding_shape = (1, int(prompt.shape[1]), int(model.config.hidden_size))
    layer_shape = embedding_shape[:-1] + (int(model.config.hidden_size) * int(model.config.hc_count),)
    actual_shapes = tuple(tuple(value.shape) for value in hidden_states)
    expected_shapes = (layer_shape,) * int(model.config.num_hidden_layers)
    if actual_shapes != expected_shapes:
        raise RuntimeError(f"unexpected hidden-state shapes; expected {expected_shapes}, got {actual_shapes}")
    if tuple(final_hidden.shape) != embedding_shape:
        raise RuntimeError(f"unexpected final mixer shape {tuple(final_hidden.shape)}; expected {embedding_shape}")

    artifact = {
        "metadata": {
            "schema_version": 1,
            "hf_model_id": MODEL_ID,
            "checkpoint_revision": MODEL_REVISION,
            "purpose": "first-divergent-layer localization for the 125-token prefill regression",
            "boundary_contract": (
                "embedding is the repeated four-stream embedding; hidden_states[i] is decoder layer i output; "
                "final_hidden is the final hyperconnection mixer output"
            ),
            "dtype": "bfloat16",
            "expert_cache_capacity_per_layer": expert_cache_capacity,
        },
        "messages": messages,
        "rendered_prompt": rendered,
        "prompt_tokens": prompt.reshape(-1).cpu(),
        "embedding": embedding,
        "hidden_states": hidden_states,
        "final_hidden": final_hidden,
        "layer0_boundaries": layer0_boundaries,
        "logits": result.logits[0, -1].detach().cpu().float(),
        "timing": {"model_load_seconds": load_seconds, "prefill_seconds": prefill_seconds},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "metadata": artifact["metadata"],
                "prompt_tokens": int(prompt.numel()),
                "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "hidden_boundaries": len(hidden_states) + 2,
                "embedding_shape": list(embedding_shape),
                "layer_shape": list(layer_shape),
                "first_token": int(artifact["logits"].argmax()),
                "timing": artifact["timing"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    ple_store.close()
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert-cache-capacity", type=int, default=32)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    artifact = generate(
        args.output,
        expert_cache_capacity=args.expert_cache_capacity,
        threads=args.threads,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "hidden_boundaries": len(artifact["hidden_states"]) + 2,
                "first_token": int(artifact["logits"].argmax()),
                "timing": artifact["timing"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
