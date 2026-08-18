# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full-stack teacher-forcing accuracy at **batch 4**, per slot, under a given precision config.

The sweep's accuracy gates are all batch 1, and the model advertises a batch bound of 32. The
selected config narrows the dense projections and the LM head to bfloat4_b, and
`probe_batch_slot_tie.py` showed that a batch-4 decode is not bit-identical across slots at *any*
precision (the per-row reduction order differs). That probe ran on the reduced two-layer variant;
this one answers the question the vLLM stage actually inherits: **does per-slot divergence cost
accuracy on the whole 40-layer stack?**

Every slot is given the *same* AIME24 prompt and the *same* forced continuation, so all four slots
should reproduce the batch-1 result. Per slot it reports top-1/top-5/top-100 against the readiness
reference, so a slot that drifts shows up as a lower number rather than as a token diff nobody scores.

    python .../doc/datatype_sweep/logs/probe_batch4_accuracy.py --batch 4 --output .../batch4_accuracy.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import close_ornith_mesh, open_ornith_mesh

MODEL_DIR = Path("models/autoports/ornith_ai_ornith_1_0_35b")
DEFAULT_REFERENCE = MODEL_DIR / "readiness_aime24_chat.refpt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    ap.add_argument("--cache-context", type=int, default=8192)
    ap.add_argument("--layers", default=None, help="comma-separated HF layer indices (default: all 40)")
    ap.add_argument("--config", default=None, help="precision config path (default: the selected one)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from models.common.readiness_check.schema import load_reference

    reference = load_reference(Path(args.reference).resolve())
    entry = reference.entries[0]
    prompt = entry.prompt_tokens[0].tolist()
    forced = entry.generated_tokens[0].tolist()
    topk = entry.topk_tokens  # [G, K]
    steps = len(forced)
    k = int(topk.shape[1])

    kwargs = {}
    if args.layers:
        kwargs["layer_indices"] = [int(v) for v in args.layers.split(",")]
    if args.config:
        kwargs["policy"] = args.config

    mesh = open_ornith_mesh()
    try:
        started = time.perf_counter()
        gen = build_generator(
            model_dir=MODEL_DIR.resolve(),
            mesh_device=mesh,
            max_batch_size=args.batch,
            cache_context=args.cache_context,
            **kwargs,
        )
        build_s = time.perf_counter() - started
        try:
            model = gen.model
            gen.reset()
            tokens = torch.tensor([prompt] * args.batch)
            logits = gen.prefill_forward(tokens, page_table=None, kv_cache=None, prompt_lens=[len(prompt)] * args.batch)
            # `prefill_forward` returns [batch, 1, vocab]; the first prediction is position 0 of the
            # reference's generated span.
            step_logits = logits[:, 0].float()
            hits = {slot: {"top1": 0, "top5": 0, f"top{k}": 0} for slot in range(args.batch)}
            per_slot_tokens = {slot: [] for slot in range(args.batch)}

            def score(step_idx, vecs):
                ref = topk[step_idx]
                ref1, ref5, refk = int(ref[0]), set(ref[:5].tolist()), set(ref.tolist())
                order = torch.topk(vecs, 1, dim=-1).indices.reshape(-1)
                for slot in range(args.batch):
                    tok = int(order[slot])
                    per_slot_tokens[slot].append(tok)
                    hits[slot]["top1"] += tok == ref1
                    hits[slot]["top5"] += tok in ref5
                    hits[slot][f"top{k}"] += tok in refk

            score(0, step_logits)
            positions = torch.tensor([len(prompt)] * args.batch)
            for step in range(1, steps):
                feed = torch.tensor([forced[step - 1]] * args.batch, dtype=torch.int32)
                host = gen.decode_forward(
                    feed, positions, page_table=gen.page_table, enable_trace=True, sample_on_device=False
                )
                vecs = host.float()
                if vecs.dim() == 3:
                    vecs = vecs[:, 0]
                score(step, vecs)
                positions = positions + 1

            per_slot = []
            for slot in range(args.batch):
                per_slot.append(
                    {
                        "slot": slot,
                        "top1": hits[slot]["top1"] / steps,
                        "top5": hits[slot]["top5"] / steps,
                        "top100": hits[slot][f"top{k}"] / steps,
                        "matches_top1": hits[slot]["top1"],
                        "matches_top5": hits[slot]["top5"],
                        "matches_top100": hits[slot][f"top{k}"],
                        "total": steps,
                        "k": k,
                    }
                )
            agreement = [
                sum(1 for a, b in zip(per_slot_tokens[0], per_slot_tokens[s]) if a == b) / steps
                for s in range(args.batch)
            ]
            report = {
                "policy": model.policy.name,
                "policy_env": os.environ.get("ORNITH_PRECISION_POLICY"),
                "config": args.config,
                "batch": args.batch,
                "layers": len(model.layers),
                "reduced": model.is_reduced,
                "reference": str(Path(args.reference).resolve()),
                "prompt_len": len(prompt),
                "steps": steps,
                "build_s": build_s,
                "regime": "teacher forcing at batch N with the SAME prompt and the SAME forced "
                "continuation in every slot, traced decode, logits read back per slot instead of "
                "sampled so every slot is scored rather than only compared",
                "per_slot": per_slot,
                "slot_token_agreement_with_slot0": agreement,
                "per_slot_top1": [p["top1"] for p in per_slot],
                "all_slots_equal_top1": len({p["top1"] for p in per_slot}) == 1,
                "min_top1": min(p["top1"] for p in per_slot),
                "min_top5": min(p["top5"] for p in per_slot),
                "min_top100": min(p["top100"] for p in per_slot),
            }
        finally:
            gen.teardown()
    finally:
        close_ornith_mesh(mesh)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k2: v for k2, v in report.items() if k2 != "per_slot"}, indent=2, default=str))
    for p in report["per_slot"]:
        print(f"slot {p['slot']}: top1={p['top1']:.3f} top5={p['top5']:.3f} top{p['k']}={p['top100']:.3f}")
    print("BATCH4_ACCURACY_OK")


if __name__ == "__main__":
    main()
