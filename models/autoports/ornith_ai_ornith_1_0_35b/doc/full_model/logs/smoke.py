# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fast reduced-model probe: one real layer of each kind, real terminal path, short generation.

This is the debugging loop the $full-model skill asks for -- it exercises embeddings, the layer
stack, the final norm, the LM head, trace capture/replay and split sampling on real weights and
real cache/page-table shapes, in a couple of minutes rather than the tens of minutes a 40-layer
build costs. It is never evidence of accuracy: two layers do not make the model.
"""

from __future__ import annotations

import argparse
import time

import torch
from loguru import logger

from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import close_ornith_mesh, open_ornith_mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,3", help="HF layer indices to build ('all' for the full stack)")
    ap.add_argument("--mesh", default="1x4")
    ap.add_argument("--cache-context", type=int, default=4096)
    ap.add_argument("--prompt-len", type=int, default=37, help="deliberately non-aligned by default")
    ap.add_argument("--gen-len", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--sampling-mode", default="device")
    args = ap.parse_args()

    shape = tuple(int(v) for v in args.mesh.split("x"))
    mesh = open_ornith_mesh(shape, fabric=shape != (1, 1))
    try:
        kwargs = {}
        if args.layers != "all":
            kwargs["layer_indices"] = [int(v) for v in args.layers.split(",")]
        t0 = time.perf_counter()
        gen = build_generator(
            model_dir="models/autoports/ornith_ai_ornith_1_0_35b",
            mesh_device=mesh,
            max_batch_size=args.batch,
            cache_context=args.cache_context,
            sampling_mode=args.sampling_mode,
            **kwargs,
        )
        logger.info(f"build took {time.perf_counter() - t0:.1f} s")
        logger.info(f"capability: {gen.model.capability()}")

        torch.manual_seed(0)
        prompt = torch.randint(0, 200000, (args.prompt_len,)).tolist()

        logits = gen.prefill_forward(torch.tensor([prompt]), page_table=None, kv_cache=None, prompt_lens=[len(prompt)])
        logger.info(f"prefill logits {tuple(logits.shape)} finite={bool(torch.isfinite(logits).all())}")

        gen.reset()
        out = gen.generate(prompt_token_ids=prompt, max_new_tokens=args.gen_len, enable_trace=True)
        logger.info(f"generate -> {out}")
        logger.info(f"perf {gen.perf}")

        gen.reset()
        forced = []

        def next_input(step, predicted):
            forced.append(predicted)
            return (predicted + 1) % 200000

        out2 = gen.generate(
            prompt_token_ids=prompt, max_new_tokens=args.gen_len, next_input=next_input, enable_trace=True
        )
        logger.info(f"teacher-forced -> {out2} (callbacks={len(forced)})")
        logger.info(f"perf {gen.perf}")

        gen.teardown()
        print("SMOKE_OK")
    finally:
        close_ornith_mesh(mesh, fabric=shape != (1, 1))


if __name__ == "__main__":
    main()
