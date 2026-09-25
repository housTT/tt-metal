# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Generate HF layer boundaries for the first teacher-forced decode token."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_flash_next.demo.generate_hf_reference import _load_oracle
from models.autoports.qwen_qwen3_8_flash_next.tests import harness as H
from models.autoports.qwen_qwen3_8_flash_next.tt.model import MODEL_ID, MODEL_REVISION


def _last_token(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim >= 2:
            value = value[:, -1:]
        return value.to(torch.bfloat16).contiguous()
    if isinstance(value, tuple):
        return tuple(_last_token(item) for item in value)
    return value


@torch.inference_mode()
def generate(
    output: Path,
    *,
    reference: Path,
    expert_cache_capacity: int = 32,
    threads: int | None = None,
) -> dict:
    if threads is not None:
        torch.set_num_threads(threads)
    readiness = torch.load(reference, map_location="cpu", weights_only=False)
    prompt = torch.as_tensor(readiness["prompt_tokens"], dtype=torch.int64).reshape(1, -1)
    teacher = torch.as_tensor(readiness["reference_tokens"][:1], dtype=torch.int64).reshape(1, 1)
    sequence = torch.cat([prompt, teacher], dim=1)

    load_started = time.perf_counter()
    model, ple_store, _experts = _load_oracle(H.MODEL_SNAPSHOT, expert_cache_capacity=expert_cache_capacity)
    load_seconds = time.perf_counter() - load_started
    ple_store.reset_request("hf-aime24")

    layer_outputs: list[torch.Tensor | None] = [None] * int(model.config.num_hidden_layers)
    final_outputs: list[torch.Tensor] = []
    layer0_boundaries: dict[str, object] = {}

    def capture_layer(index):
        def hook(_module, _inputs, value):
            layer_outputs[index] = _last_token(value)

        return hook

    hooks = [layer.register_forward_hook(capture_layer(index)) for index, layer in enumerate(model.model.layers)]
    hooks.append(
        model.model.hyper_connection_mixer.register_forward_hook(
            lambda _module, _inputs, value: final_outputs.append(_last_token(value))
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
                lambda _module, _inputs, value, boundary=name: layer0_boundaries.__setitem__(
                    boundary, _last_token(value)
                )
            )
        )

    started = time.perf_counter()
    try:
        result = model(input_ids=sequence, use_cache=False, logits_to_keep=1, return_dict=True)
    finally:
        for hook in hooks:
            hook.remove()
    forward_seconds = time.perf_counter() - started
    if any(value is None for value in layer_outputs) or len(final_outputs) != 1:
        raise RuntimeError("HF hooks did not capture every decode boundary")

    embedding = model.model.embed_tokens(teacher).repeat(1, 1, int(model.config.hc_count))
    artifact = {
        "metadata": {
            "schema_version": 1,
            "hf_model_id": MODEL_ID,
            "checkpoint_revision": MODEL_REVISION,
            "purpose": "first teacher-forced decode-token layer localization",
            "prompt_tokens": int(prompt.shape[1]),
            "decode_position": int(prompt.shape[1]),
            "dtype": "bfloat16",
            "expert_cache_capacity_per_layer": expert_cache_capacity,
        },
        "prompt_tokens": prompt.reshape(-1).cpu(),
        "teacher_token": teacher.reshape(-1).cpu(),
        "embedding": embedding.detach().cpu().to(torch.bfloat16).contiguous(),
        "hidden_states": tuple(layer_outputs),
        "final_hidden": final_outputs[0],
        "layer0_boundaries": layer0_boundaries,
        "logits": result.logits[0, -1].detach().cpu().float(),
        "expected_next_token": int(torch.as_tensor(readiness["reference_tokens"])[1]),
        "timing": {"model_load_seconds": load_seconds, "forward_seconds": forward_seconds},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "metadata": artifact["metadata"],
                "teacher_token": int(teacher.item()),
                "hf_top1": int(artifact["logits"].argmax()),
                "expected_next_token": artifact["expected_next_token"],
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
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).parents[1] / "doc/full_model/readiness_aime24_chat.refpt",
    )
    parser.add_argument("--expert-cache-capacity", type=int, default=32)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    artifact = generate(
        args.output,
        reference=args.reference,
        expert_cache_capacity=args.expert_cache_capacity,
        threads=args.threads,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "teacher_token": int(artifact["teacher_token"].item()),
                "hf_top1": int(artifact["logits"].argmax()),
                "timing": artifact["timing"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
