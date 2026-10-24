# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone-generator greedy control for the raw prompts of the mixed-shape serving check.

Usage (one chip, from the tt-metal worktree):
    TT_VISIBLE_DEVICES=1 ... python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_raw_prompt_control \
        --mixed-shape-json <readiness_vllm/trace_reuse/mixed_shape_check.json> --output <json>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.run_vllm_mixed_shape_check import _token_prompt
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import _padded_prefill_len, build_generator
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm import PRECISION_CONFIG

MODEL_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixed-shape-json", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=str,
        default="chunked_while_decoding:decode_b,chunked_while_decoding:decode_a,unseen_257,chunked_while_decoding:decode_d,unseen_130,unseen_129",
    )
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    served = {r["label"]: r for r in json.loads(args.mixed_shape_json.read_text())["requests_log"]}
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    report = []
    try:
        gen = build_generator(
            MODEL_ROOT,
            mesh,
            max_batch_size=1,
            precision_config_path=str(PRECISION_CONFIG),
            tensor_cache_path=os.environ.get("GEMMA4_TENSOR_CACHE_PATH", "/tmp/gemma4_raw_control_cache"),
        )
        tokenizer = gen.tokenizer
        state = gen.model.state
        for label in args.labels.split(","):
            record = served[label]
            ids = _token_prompt(tokenizer, record["prompt_len"], record["seed"])
            gen.reset()
            length = len(ids)
            tokens = torch.tensor(ids, dtype=torch.long).reshape(1, length)
            tokens = torch.nn.functional.pad(tokens, (0, _padded_prefill_len(length) - length))
            logits = gen.prefill_forward(tokens, page_table=state.page_tables, kv_cache=state, prompt_lens=[length])
            host = gen._gather_logits_to_torch(logits).reshape(-1, gen.model.vocab_size)[0]
            ttnn.deallocate(logits)
            generated = [int(host.argmax())]
            position = length
            for _ in range(args.steps - 1):
                sampled = gen.decode_forward(
                    torch.tensor([[generated[-1]]], dtype=torch.long),
                    torch.tensor([position], dtype=torch.int32),
                    page_table=state.page_tables,
                    kv_cache=state,
                    sampling_mode="device",
                    enable_trace=True,
                )
                generated.append(int(gen._read_tokens(sampled, 1)[0]))
                position += 1
            control_text = tokenizer.decode(generated)
            served_ids = tokenizer.encode(record["text"], add_special_tokens=False)
            common = 0
            for a, b in zip(generated, served_ids):
                if a != b:
                    break
                common += 1
            entry = {
                "label": label,
                "prompt_len": record["prompt_len"],
                "seed": record["seed"],
                "served_text_prefix": record["text"][:160],
                "control_text": control_text,
                "control_token_ids": generated,
                "served_token_ids_prefix": served_ids[: args.steps],
                "common_prefix_tokens": common,
                "served_temperature": record["temperature"],
            }
            report.append(entry)
            print(json.dumps(entry), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
