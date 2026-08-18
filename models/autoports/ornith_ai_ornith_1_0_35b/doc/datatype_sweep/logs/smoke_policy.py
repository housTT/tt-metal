# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""One-decoder smoke test for a precision policy, before it costs a full-model run.

``$datatype-sweep`` warns that a dtype change can need a small semantic change in the code (the
``paged_fill_cache`` / ``paged_update_cache`` dtype contract is the named example). So every
candidate goes through this first: build the **reduced** two-layer variant (one real
``linear_attention`` layer, one real ``full_attention`` layer, real weights, real cache and
page-table shapes, the real terminal path and the real traced decode), prefill a short non-aligned
prompt, run a few traced decode steps, and assert the tokens are finite and in vocabulary.

It also prints the built precision summary, so a policy field that the construction path ignores is
visible here rather than after a 4-minute full-model build.

    python .../doc/datatype_sweep/logs/smoke_policy.py --policy optimized
    python .../doc/datatype_sweep/logs/smoke_policy.py --config candidates/C05.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from loguru import logger

from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import close_ornith_mesh, open_ornith_mesh

MODEL_DIR = Path("models/autoports/ornith_ai_ornith_1_0_35b")
#: The reduced profiling variant: HF layer 0 is ``linear_attention`` and HF layer 3 is
#: ``full_attention``, so this pair exercises both kinds and both cache/state families.
REDUCED_LAYERS = [0, 3]


def smoke(mesh, policy, *, layers=REDUCED_LAYERS, prompt_len=87, steps=8, label="") -> dict:
    started = time.perf_counter()
    gen = build_generator(
        model_dir=MODEL_DIR.resolve(),
        mesh_device=mesh,
        max_batch_size=1,
        cache_context=8192,
        layer_indices=layers,
        policy=policy,
    )
    try:
        model = gen.model
        torch.manual_seed(0)
        prompt = torch.randint(0, model.vocab_size, (prompt_len,)).tolist()
        tokens = gen.generate(prompt_token_ids=prompt, max_new_tokens=steps, enable_trace=True, stop_on_eos=False)
        bad = [t for t in tokens if not (0 <= int(t) < model.vocab_size)]
        summary = model.precision_summary()
        result = {
            "label": label,
            "policy": model.policy.name,
            "layers": layers,
            "prompt_len": prompt_len,
            "tokens": [int(t) for t in tokens],
            "out_of_vocab": bad,
            "decode_t/s/u": gen.perf["decode_t/s/u"],
            "ttft_ms": gen.perf["ttft_s"] * 1e3,
            "trace_recaptures": gen.trace_recaptures,
            "pipelined_readback": gen.perf["pipelined_readback"],
            "precision": summary,
            "wall_s": time.perf_counter() - started,
            "ok": not bad and len(tokens) == steps,
        }
    finally:
        gen.teardown()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None, help="registered policy name, or a path to a JSON config")
    ap.add_argument("--config", default=None, help="path to a JSON precision config")
    ap.add_argument("--layers", default=",".join(str(i) for i in REDUCED_LAYERS))
    ap.add_argument("--prompt-len", type=int, default=87)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    policy = args.config or args.policy
    mesh = open_ornith_mesh()
    try:
        result = smoke(
            mesh,
            policy,
            layers=[int(v) for v in args.layers.split(",")],
            prompt_len=args.prompt_len,
            steps=args.steps,
            label=str(policy),
        )
    finally:
        close_ornith_mesh(mesh)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    logger.info(json.dumps({k: v for k, v in result.items() if k != "precision"}, indent=2, default=str))
    print(json.dumps(result["precision"]["built"], indent=2, default=str))
    print("SMOKE_OK" if result["ok"] else "SMOKE_FAILED")
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
