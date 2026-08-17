# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""What batch 4 changes about slot 0's greedy tokens, and what it does not.

Two separate questions the batch>1 path raises, answered against a batch-1 control on the reduced
two-layer probe (one real `linear_attention` layer, one real `full_attention` layer):

1. **does the prompt reach the decode slot?** `generate` prefills through the batch-1 state pack
   while the captured decode trace is bound to the batch-B pack. Without
   `OrnithModel.prefill_request_into_slot`'s merge the 30 recurrent layers decode from a zeroed
   state, and the **first decode token** is already wrong. That is the arm labelled `no merge`
   below, reproduced by monkey-patching the merge out;
2. **does another slot's content change slot 0?** Run the same prompt in slot 0 with three
   identical neighbours and again with three different ones. The answer is no: the two arms are
   token-identical. What *does* differ is batch 4 against batch **1** - a few tokens in, greedy
   picks a different near-tie, because the batch changes the matmul and MoE-grouping geometry and
   therefore the last bits of the logits. That is a property of the batch, not of the port, and
   this probe is the evidence for saying so.

    python .../doc/optimized_full_model/logs/probe_batch_slots.py [--steps 6]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from loguru import logger

from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import close_ornith_mesh, open_ornith_mesh

MODEL_DIR = Path(__file__).resolve().parents[3]
PROBE_LAYERS = [0, 3]
CACHE_CONTEXT = 4096

PROMPT = [6, 66, 666, 6666, 66]
NEIGHBOURS = [[9, 99, 999, 9999, 99], [1, 2, 3, 4, 5], [70000, 8, 90, 100, 3]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--output", default=str(MODEL_DIR / "doc" / "optimized_full_model" / "batch_slots.json"))
    args = ap.parse_args()

    mesh = open_ornith_mesh()
    report = {"prompt": PROMPT, "steps": args.steps, "arms": {}}
    try:
        one = build_generator(
            model_dir=str(MODEL_DIR),
            mesh_device=mesh,
            layer_indices=PROBE_LAYERS,
            max_batch_size=1,
            cache_context=CACHE_CONTEXT,
        )
        base = one.generate(prompt_token_ids=PROMPT, max_new_tokens=args.steps, enable_trace=True)
        report["batch1_generate"] = base
        logger.info(f"batch-1 generate: {base}")
        one.teardown()

        four = build_generator(
            model_dir=str(MODEL_DIR),
            mesh_device=mesh,
            layer_indices=PROBE_LAYERS,
            max_batch_size=4,
            cache_context=CACHE_CONTEXT,
        )

        def low_level(rows, label):
            four.reset()
            width = max(len(r) for r in rows)
            toks = torch.zeros(4, width, dtype=torch.int64)
            for user, row in enumerate(rows):
                toks[user, : len(row)] = torch.tensor(row)
            logits = four.prefill_forward(toks, page_table=None, kv_cache=None, prompt_lens=[len(r) for r in rows])
            current = torch.argmax(logits, dim=-1).reshape(-1)
            out = [int(current[0])]
            positions = torch.tensor([len(r) for r in rows], dtype=torch.int32)
            for _ in range(args.steps - 1):
                current = four.decode_forward(
                    current, positions, page_table=four.page_table, enable_trace=True, sample_on_device=True
                )
                out.append(int(current[0]))
                positions = positions + 1
            report["arms"][label] = out
            logger.info(f"{label}: {out}")
            return out

        same = low_level([PROMPT] * 4, "batch4_low_level_identical_neighbours")
        mixed = low_level([PROMPT] + NEIGHBOURS, "batch4_low_level_different_neighbours")
        gen4 = four.generate(prompt_token_ids=PROMPT, max_new_tokens=args.steps, enable_trace=True)
        report["arms"]["batch4_generate"] = gen4
        logger.info(f"batch4_generate: {gen4}")

        # The merge removed, on the same generator: this is the defect, not a hypothetical.
        original = type(four.model)._merge_prefill_state_into_slot
        try:
            type(four.model)._merge_prefill_state_into_slot = lambda self, slot: None
            no_merge = four.generate(prompt_token_ids=PROMPT, max_new_tokens=args.steps, enable_trace=True)
        finally:
            type(four.model)._merge_prefill_state_into_slot = original
        report["arms"]["batch4_generate_no_merge"] = no_merge
        logger.info(f"batch4_generate_no_merge: {no_merge}")

        report["findings"] = {
            "neighbour_content_changes_slot0": same != mixed,
            "first_decode_token_matches_batch1": len(base) > 1 and gen4[1] == base[1],
            "first_decode_token_matches_batch1_without_the_merge": len(base) > 1 and no_merge[1] == base[1],
            "batch4_greedy_first_divergence_from_batch1": next(
                (i for i, (a, b) in enumerate(zip(base, gen4)) if a != b), None
            ),
            "no_merge_first_divergence_from_batch1": next(
                (i for i, (a, b) in enumerate(zip(base, no_merge)) if a != b), None
            ),
        }
        four.teardown()
        Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps(report, indent=2))
        print("\nPROBE_OK")
    finally:
        close_ornith_mesh(mesh)


if __name__ == "__main__":
    main()
