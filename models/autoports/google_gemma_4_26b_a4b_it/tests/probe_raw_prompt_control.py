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
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import (
    _padded_prefill_len,
    build_generator,
    plan_prefill_groups,
)
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
    parser.add_argument(
        "--group-batch",
        action="store_true",
        help="also prefill every label's prompt in one multi-row step through the standalone batched path "
        "and compare its first tokens with the solo controls",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    served = {r["label"]: r for r in json.loads(args.mixed_shape_json.read_text())["requests_log"]}
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    report = []
    try:
        gen = build_generator(
            MODEL_ROOT,
            mesh,
            max_batch_size=max(1, len(args.labels.split(","))) if args.group_batch else 1,
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
            top2 = torch.topk(host.float(), 2)
            served_first = tokenizer.encode(record["text"], add_special_tokens=False)[:1]
            served_first_logit = float(host[served_first[0]]) if served_first else None
            margin = {
                "control_top1": int(top2.indices[0]),
                "control_top2": int(top2.indices[1]),
                "top1_minus_top2_logit": float(top2.values[0] - top2.values[1]),
                "served_first_token": served_first[0] if served_first else None,
                "served_first_rank": int((host.float() > served_first_logit).sum()) if served_first else None,
                "top1_minus_served_first_logit": (float(top2.values[0]) - served_first_logit) if served_first else None,
            }
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
                "first_token_margin": margin,
            }
            report.append(entry)
            print(json.dumps(entry), flush=True)
        if args.group_batch:
            labels = args.labels.split(",")
            prompts = [_token_prompt(tokenizer, served[label]["prompt_len"], served[label]["seed"]) for label in labels]
            lens = [len(ids) for ids in prompts]
            width = max(_padded_prefill_len(length) for length in lens)
            tokens = torch.zeros((len(prompts), width), dtype=torch.long)
            for row, ids in enumerate(prompts):
                tokens[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            gen.reset()
            gen.batched_prefill_enabled = True
            logits = gen.prefill_forward(tokens, page_table=state.page_tables, kv_cache=state, prompt_lens=lens)
            host = gen._gather_logits_to_torch(logits).reshape(-1, gen.model.vocab_size)[: len(prompts)]
            ttnn.deallocate(logits)
            group_entry = {
                "label": "standalone_group_of_all_labels",
                "rows": [
                    {
                        "label": label,
                        "batched_first_token": int(host[row].argmax()),
                        "solo_control_first_token": report[row]["control_token_ids"][0],
                        "served_first_token": report[row]["first_token_margin"]["served_first_token"],
                        "batched_top1_minus_top2_logit": float(torch.topk(host[row].float(), 2).values.diff().abs()),
                    }
                    for row, label in enumerate(labels)
                ],
                "groups": [
                    [list(g.rows), g.segment_len, g.padded_batch]
                    for g in plan_prefill_groups(lens, [0] * len(lens), enabled=True)[0]
                ],
            }
            group_entry["batched_equals_solo_first_token"] = all(
                r["batched_first_token"] == r["solo_control_first_token"] for r in group_entry["rows"]
            )
            report.append(group_entry)
            print(json.dumps(group_entry), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
