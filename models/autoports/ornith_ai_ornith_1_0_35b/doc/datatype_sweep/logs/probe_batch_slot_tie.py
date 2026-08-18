# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Why one of four identical decode slots emits a different token under the selected policy.

`tests/test_full_model.py::test_the_batched_prefill_state_reaches_every_decode_slot` prefills the
same four-token prompt into four slots and asserts every slot decodes the batch-1 answer. It passes
on the pre-sweep policy and fails on the selected one, with slot 1 emitting 78562 where the other
three emit 45568.

Two hypotheses, and this probe separates them:

  A. near-tie - the top-2 logits at that position are within the extra quantisation noise bfloat4_b
     dense projections carry, and the per-slot reduction order decides which wins. Prediction: every
     slot's logit VECTOR agrees to high PCC, and the top-1/top-2 margin is tiny compared with the
     slot-to-slot spread of those same two logits.
  B. state bug - slot 1's prefill state is not reaching its decode pack. Prediction: slot 1's logit
     vector differs from slot 0's structurally, not just at the top of the ranking.

Run for both policies:

    python .../probe_batch_slot_tie.py --tag selected
    ORNITH_PRECISION_POLICY=optimized python .../probe_batch_slot_tie.py --tag baseline
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from loguru import logger

from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import close_ornith_mesh, open_ornith_mesh

MODEL_DIR = Path("models/autoports/ornith_ai_ornith_1_0_35b")
PROBE_LAYERS = [0, 3]  # the suite's reduced probe: one layer of each kind
PROMPT = [8, 88, 888, 8888]  # the failing test's prompt, verbatim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument(
        "--no-merge",
        action="store_true",
        help="the NEGATIVE CONTROL: disable OrnithModel._merge_prefill_state_into_slot, which is the "
        "failure this test exists to catch. It answers 'what do the cross-slot PCC and the shared "
        "top-5 look like when a slot's prefill state really does not arrive?', so the rewritten "
        "test's thresholds are set from a measurement rather than from a guess",
    )
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.no_merge:
        from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import OrnithModel

        OrnithModel._merge_prefill_state_into_slot = lambda self, slot: None
        logger.warning("NEGATIVE CONTROL: _merge_prefill_state_into_slot disabled")

    mesh = open_ornith_mesh()
    try:
        one = build_generator(
            model_dir=MODEL_DIR.resolve(),
            mesh_device=mesh,
            layer_indices=PROBE_LAYERS,
            max_batch_size=1,
            cache_context=4096,
        )
        expected = one.generate(prompt_token_ids=PROMPT, max_new_tokens=2, enable_trace=True)
        policy = one.model.policy
        one.teardown()

        gen = build_generator(
            model_dir=MODEL_DIR.resolve(),
            mesh_device=mesh,
            layer_indices=PROBE_LAYERS,
            max_batch_size=args.batch,
            cache_context=4096,
        )
        rounds = []
        try:
            for repeat in range(args.repeats):
                gen.reset()
                tokens = torch.tensor([PROMPT] * args.batch)
                prefill_logits = gen.prefill_forward(
                    tokens, page_table=None, kv_cache=None, prompt_lens=[len(PROMPT)] * args.batch
                )
                first = torch.argmax(prefill_logits, dim=-1).reshape(-1)
                # Host logits for the decode step: the same graph, read back instead of sampled, so
                # the full ranking per slot is visible rather than only the argmax.
                host = gen.decode_forward(
                    first,
                    torch.tensor([len(PROMPT)] * args.batch),
                    page_table=gen.page_table,
                    enable_trace=True,
                    sample_on_device=False,
                )
                logits = host.float() if isinstance(host, torch.Tensor) else torch.tensor(host).float()
                if logits.dim() == 3:
                    logits = logits[:, 0]
                top = torch.topk(logits, 5, dim=-1)
                slot_rows = []
                for slot in range(args.batch):
                    ids = top.indices[slot].tolist()
                    vals = top.values[slot].tolist()
                    slot_rows.append(
                        {
                            "slot": slot,
                            "argmax": ids[0],
                            "top5_ids": ids,
                            "top5_logits": vals,
                            "top1_minus_top2": vals[0] - vals[1],
                        }
                    )
                ref = logits[0]
                pcc = [float(torch.corrcoef(torch.stack([ref, logits[s]]))[0, 1]) for s in range(args.batch)]
                max_abs_diff = [float((logits[s] - ref).abs().max()) for s in range(args.batch)]
                # The two candidates the disagreement is between, across all slots.
                contenders = sorted({r["argmax"] for r in slot_rows})
                per_slot_contender_logits = {
                    int(c): [float(logits[s, c]) for s in range(args.batch)] for c in contenders
                }
                rounds.append(
                    {
                        "repeat": repeat,
                        "prefill_argmax_per_slot": first.tolist(),
                        "prefill_slots_agree": len(set(first.tolist())) == 1,
                        "decode_argmax_per_slot": [r["argmax"] for r in slot_rows],
                        "decode_slots_agree": len({r["argmax"] for r in slot_rows}) == 1,
                        "slots": slot_rows,
                        "logit_pcc_against_slot0": pcc,
                        "max_abs_logit_diff_against_slot0": max_abs_diff,
                        "contenders": contenders,
                        "per_slot_contender_logits": per_slot_contender_logits,
                    }
                )
                logger.info(f"repeat {repeat}: {rounds[-1]['decode_argmax_per_slot']}")
        finally:
            gen.teardown()
    finally:
        close_ornith_mesh(mesh)

    report = {
        "tag": args.tag,
        "negative_control_merge_disabled": bool(args.no_merge),
        "policy": policy.name,
        "policy_env": os.environ.get("ORNITH_PRECISION_POLICY"),
        "dense_projection_dtype": str(policy.proj_dtype),
        "dense_projection_fidelity": str(policy.proj_fidelity),
        "lm_head_dtype": str(policy.resolved_lm_head_dtype),
        "batch": args.batch,
        "prompt": PROMPT,
        "batch1_expected_tokens": [int(t) for t in expected],
        "rounds": rounds,
        "all_repeats_agree_across_slots": all(r["decode_slots_agree"] for r in rounds),
    }
    out = args.output or str(MODEL_DIR / "doc" / "datatype_sweep" / f"batch_slot_tie_{args.tag}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rounds"}, indent=2))
    for r in rounds:
        print(
            f"repeat {r['repeat']}: argmax {r['decode_argmax_per_slot']}  "
            f"pcc {['%.6f' % p for p in r['logit_pcc_against_slot0']]}  "
            f"maxdiff {['%.4f' % d for d in r['max_abs_logit_diff_against_slot0']]}"
        )
        for cid, vals in r["per_slot_contender_logits"].items():
            print(f"   token {cid}: per-slot logits {['%.4f' % v for v in vals]}")
        for s in r["slots"]:
            print(f"   slot {s['slot']}: top1-top2 margin {s['top1_minus_top2']:.5f}  top5 {s['top5_ids']}")
    print("PROBE_OK")


if __name__ == "__main__":
    main()
