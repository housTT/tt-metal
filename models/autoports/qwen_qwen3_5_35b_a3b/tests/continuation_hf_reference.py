# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import time
from pathlib import Path

import torch

BOOK = Path("/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/runtime/readiness_check/"
            "autoregressive_prompt.txt")
CHECKPOINT = Path("/home/hous/dev/qwen3.5-35b-a3b/weights/Qwen3.5-35B-A3B")
TOTALS = (428, 4301, 8193, 12417)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    book = tok(BOOK.read_text().strip() * 400, add_special_tokens=False).input_ids
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, dtype=torch.bfloat16, local_files_only=True).eval()
    result = dict(totals=TOTALS, ids=book[: max(TOTALS)], logits={}, seconds={})
    for total in TOTALS:
        t0 = time.perf_counter()
        out = model(torch.tensor([book[:total]]), logits_to_keep=1, use_cache=False)
        result["logits"][total] = out.logits[0, -1].float()
        result["seconds"][total] = time.perf_counter() - t0
        print(total, result["seconds"][total], int(result["logits"][total].argmax()), flush=True)
        torch.save(result, args.output)


if __name__ == "__main__":
    main()
