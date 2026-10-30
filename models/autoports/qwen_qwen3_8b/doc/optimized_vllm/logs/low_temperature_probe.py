# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Characterize the device sampler against host sampling at low temperature (stage-9 carried item c).

usage: low_temperature_probe.py <output.json> [--layers 36] [--draws 64]

For the stage-9 prompt ("Letter: ") and two more prompts: the first-token logits (host fp32 through the
prefill-logits path), the host categorical distribution at temperature 0.01 / 0.7 with top_k 20, top_p 0.95
(vLLM's order: temperature, top-k, top-p), the device sampler's first token over --draws seeds at the same
parameters, and the bf16 view of the scaled logits (the device sampler scales in bf16).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8b.tt.generator import build_generator
from models.common.sampling import SamplingParams

MODEL_DIR = Path("models/autoports/qwen_qwen3_8b")


def host_distribution(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    scaled = logits.float() / temperature
    if top_k > 0:
        kth = torch.topk(scaled, top_k).values[-1]
        scaled = torch.where(scaled < kth, torch.full_like(scaled, float("-inf")), scaled)
    probs = torch.softmax(scaled, dim=-1)
    if top_p < 1.0:
        sorted_probs, idx = probs.sort(descending=True)
        cum = sorted_probs.cumsum(-1)
        keep = cum - sorted_probs < top_p
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask[idx[keep]] = True
        probs = torch.where(mask, probs, torch.zeros_like(probs))
        probs = probs / probs.sum()
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--draws", type=int, default=64)
    args = ap.parse_args()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=512 * 1024 * 1024)
    gen = build_generator(str(MODEL_DIR), mesh, num_layers=args.layers)
    tok = gen.tokenizer
    prompts = {"letter": "Letter: ", "random": "Random: ", "three_laws": "The three laws of thermodynamics are"}
    records = {}
    for name, text in prompts.items():
        ids = tok.encode(text, add_special_tokens=False)
        logits = gen.prefill_logits(ids)[0, -1].float()
        rec = {"prompt": text, "ids": ids, "top8": []}
        top = torch.topk(logits, 8)
        for v, i in zip(top.values.tolist(), top.indices.tolist()):
            rec["top8"].append({"id": i, "token": tok.decode([i]), "logit": v, "bf16_logit": float(torch.tensor(v).to(torch.bfloat16))})
        for temperature in (0.01, 0.7):
            probs = host_distribution(logits, temperature, 20, 0.95)
            host_top = torch.topk(probs, 5)
            device_counts = Counter()
            device_eager = 0
            for seed in range(args.draws):
                gen.reset()
                gen._ensure_blocks(0, len(ids))
                table = gen._host_page_table([0])
                params = SamplingParams(temperature=[temperature], top_k=[20], top_p=[0.95], seed=[seed])
                token = gen.prefill_forward(torch.tensor([ids]), page_table=table, kv_cache=gen.kv_cache, prompt_lens=[len(ids)], sampling_params=params)
                device_counts[int(token[0])] += 1
            rec[f"temperature_{temperature}"] = {
                "host_top5": [{"id": int(i), "token": tok.decode([int(i)]), "p": float(p)} for p, i in zip(host_top.values, host_top.indices)],
                "device_first_token_counts": {f"{tok.decode([i])!r} ({i})": c for i, c in device_counts.most_common()},
                "device_draws": args.draws,
            }
        records[name] = rec
        print(name, json.dumps({k: v for k, v in rec.items() if k.startswith("temperature")}, ensure_ascii=False), flush=True)
    Path(args.output).write_text(json.dumps({"records": records, "layers": args.layers, "visible_devices": os.getenv("TT_VISIBLE_DEVICES"), "note": "host distribution = temperature, top-k 20, top-p 0.95 on fp32 prefill logits; device = SamplingGenerator eager prefill sampling with explicit per-draw seeds"}, indent=2, ensure_ascii=False) + "\n")
    gen.teardown()
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
