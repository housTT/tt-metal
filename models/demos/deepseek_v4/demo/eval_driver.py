# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Offline eval driver for DeepSeek-V4-Flash — the entry an accuracy harness (lm-eval-style)
or tt-inference-server's eval step can call today, single-sequence, on real weights.

Reads a JSON list of prompts (or uses a default set), generates completions on the device via
`DeepSeekV4Generator`, and writes `{prompt, completion, token_ids}` records to a JSON file the
scorer consumes. Correct tokens (PCC ≥ 0.99 vs HF); throughput is the correctness path — see
PRODUCTION_STATUS.md.

Usage:
    source models/demos/deepseek_v4/env.sh
    python models/demos/deepseek_v4/demo/eval_driver.py --prompts prompts.json --out completions.json \
        --max-new-tokens 64
"""
import argparse
import json

import ttnn
from models.demos.deepseek_v4.tt.generator import DeepSeekV4Generator

DEFAULT_PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "The opposite of hot is",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=None, help="JSON list of prompt strings (default: built-in set)")
    ap.add_argument("--out", default="completions.json")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    prompts = json.load(open(args.prompts)) if args.prompts else DEFAULT_PROMPTS

    device = ttnn.CreateDevice(device_id=0)
    try:
        gen = DeepSeekV4Generator(device, num_layers=args.layers)
        records = []
        for i, p in enumerate(prompts):
            r = gen.generate(p, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
            print(f"[{i + 1}/{len(prompts)}] {p!r} -> {r['completion']!r}", flush=True)
            records.append(r)
    finally:
        ttnn.CloseDevice(device)

    json.dump(records, open(args.out, "w"), indent=2)
    print(f"EVAL_DRIVER_OK wrote {len(records)} completions -> {args.out}")


if __name__ == "__main__":
    main()
