# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Where warmed TTFT goes, and how much of it is length-independent host work.

TTFT is 97 % prefill on this model, and prefill is the one part of the delivered path that is still
**eager** - 40 layers of ops dispatched one at a time from Python. This probe separates the two
terms that behaviour implies:

* a **slope**, the real per-token prefill work, from a warmed ladder over physical block lengths;
* an **intercept**, the length-independent cost of walking the stack once, which is what a captured
  prefill trace would remove.

It also A/Bs the loguru level, because the eager path emits a per-layer, per-expert-group DEBUG line
and a host-side log call inside a measured window is host work like any other.

    python .../doc/optimized_full_model/logs/probe_prefill.py --output .../prefill_profile.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import close_ornith_mesh, open_ornith_mesh

MODEL_DIR = Path("models/autoports/ornith_ai_ornith_1_0_35b")


def set_level(level: str):
    logger.remove()
    logger.add(sys.stderr, level=level)


def warmed_ttft(gen, length: int, repeats: int = 3) -> float:
    prompt = torch.randint(0, gen.model.vocab_size, (length,)).tolist()
    gen.generate(prompt_token_ids=prompt, max_new_tokens=2, enable_trace=True, stop_on_eos=False)
    best = None
    for _ in range(repeats):
        gen.generate(prompt_token_ids=prompt, max_new_tokens=2, enable_trace=True, stop_on_eos=False)
        ms = gen.perf["ttft_s"] * 1e3
        best = ms if best is None else min(best, ms)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="128,256,512,1024")
    ap.add_argument("--cache-context", type=int, default=8192)
    ap.add_argument("--output", default=str(MODEL_DIR / "doc" / "optimized_full_model" / "prefill_profile.json"))
    args = ap.parse_args()

    lengths = [int(v) for v in args.lengths.split(",")]
    mesh = open_ornith_mesh()
    out: dict = {"lengths": lengths}
    try:
        gen = build_generator(model_dir=MODEL_DIR, mesh_device=mesh, max_batch_size=1, cache_context=args.cache_context)

        for level in ("DEBUG", "WARNING"):
            set_level(level)
            out[level] = {str(n): warmed_ttft(gen, n) for n in lengths}

        set_level("WARNING")
        ladder = out["WARNING"]
        # Two fits, both reported, because they disagree by ~10 % on the intercept and the intercept
        # is the whole traced-prefill argument. The secant through the endpoints is the conservative
        # one (it attributes more of TTFT to per-token work); the least-squares fit over the whole
        # ladder is the smaller intercept. Neither is hand-typed anywhere downstream.
        lo, hi = str(lengths[0]), str(lengths[-1])
        slope = (ladder[hi] - ladder[lo]) / (lengths[-1] - lengths[0])
        intercept = ladder[lo] - slope * lengths[0]
        n = len(lengths)
        mx = sum(lengths) / n
        my = sum(ladder[str(v)] for v in lengths) / n
        sxx = sum((v - mx) ** 2 for v in lengths)
        sxy = sum((v - mx) * (ladder[str(v)] - my) for v in lengths)
        ls_slope = sxy / sxx
        ls_intercept = my - ls_slope * mx
        base = ladder[str(lengths[0])]
        out["fit"] = {
            "method": f"two-point secant through {lengths[0]} and {lengths[-1]} of the WARNING ladder",
            "slope_ms_per_token": slope,
            "intercept_ms": intercept,
            "intercept_share_at_128": intercept / ladder[str(128)] if "128" in ladder else None,
            "least_squares": {
                "method": f"least squares over all {n} WARNING points",
                "slope_ms_per_token": ls_slope,
                "intercept_ms": ls_intercept,
                f"intercept_share_at_{lengths[0]}": ls_intercept / base,
            },
            f"local_slope_{lengths[0]}_{lengths[1]}_ms_per_token": (
                (ladder[str(lengths[1])] - base) / (lengths[1] - lengths[0])
            ),
        }
        out["debug_logging_cost_ms"] = {k: out["DEBUG"][k] - out["WARNING"][k] for k in ladder}
        gen.teardown()
    finally:
        close_ornith_mesh(mesh)

    Path(args.output).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    print("PREFILL_PROBE_OK")


if __name__ == "__main__":
    main()
