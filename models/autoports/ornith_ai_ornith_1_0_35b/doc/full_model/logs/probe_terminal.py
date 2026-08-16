# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Where a token-out decode step's non-decoder time goes.

Splits a warmed traced token-out step into: the model trace (embedding + layer stack + final norm
+ LM head + on-device position advance), the sampling trace, the host synchronize and the token
readback -- each measured on its own so the full-model-only costs the `$full-model` skill asks for
can be named rather than inferred from a difference.

Also A/Bs the two knobs that plausibly move the sampler: padding each vocab shard up to the next
power of two before ``ttnn.topk``, and the LM head's column split.
"""

from __future__ import annotations

import argparse
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import close_ornith_mesh, open_ornith_mesh


def timed(fn, iters, sync_mesh):
    fn()
    ttnn.synchronize_device(sync_mesh)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(sync_mesh)
    return (time.perf_counter() - start) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--iters", type=int, default=64)
    ap.add_argument("--cache-context", type=int, default=4096)
    ap.add_argument("--pad-pow2", action="store_true", help="pad each vocab shard to a power of 2 before topk")
    ap.add_argument("--topk-groups", type=int, default=None, help="grouped local top-k; 1 disables grouping")
    ap.add_argument("--lm-head-columns", type=int, default=None)
    args = ap.parse_args()

    mesh = open_ornith_mesh((1, 4))
    try:
        kwargs = {}
        if args.layers != "all":
            kwargs["layer_indices"] = [int(v) for v in args.layers.split(",")]
        if args.lm_head_columns:
            kwargs["lm_head_max_columns"] = args.lm_head_columns
        if args.topk_groups is not None:
            kwargs["topk_num_groups"] = args.topk_groups
        gen = build_generator(
            model_dir="models/autoports/ornith_ai_ornith_1_0_35b",
            mesh_device=mesh,
            max_batch_size=1,
            cache_context=args.cache_context,
            pad_logits_to_power_of_2=args.pad_pow2,
            **kwargs,
        )
        model = gen.model
        gen._ensure_decode_trace()
        gen.reset()
        torch.manual_seed(0)
        prompt = torch.randint(0, 200000, (128,)).tolist()
        gen.generate(prompt_token_ids=prompt, max_new_tokens=4, enable_trace=True)
        # Printed so the two arms of this probe (--topk-groups 1 and 20) can be compared for
        # OUTPUT as well as for latency: the grouped local top-k is meant to be exact, and a
        # committed pair of identical token lists is what says so at the model level.
        greedy_tokens = gen.generate(prompt_token_ids=prompt, max_new_tokens=16, enable_trace=True)
        print(f"greedy tokens (fixed seed-0 prompt, 16 steps): {greedy_tokens}", flush=True)

        iters = args.iters
        rows = {}
        rows["model trace replay"] = timed(
            lambda: ttnn.execute_trace(mesh, gen._trace_id, cq_id=0, blocking=False), iters, mesh
        )
        rows["sampling trace replay"] = timed(lambda: gen._sample_traced(), iters, mesh)

        def both():
            ttnn.execute_trace(mesh, gen._trace_id, cq_id=0, blocking=False)
            gen._sample_traced()

        rows["model + sampling"] = timed(both, iters, mesh)

        def full():
            ttnn.execute_trace(mesh, gen._trace_id, cq_id=0, blocking=False)
            gen._sample_traced()
            ttnn.synchronize_device(mesh)
            gen._read_tokens()

        start = time.perf_counter()
        for _ in range(iters):
            full()
        rows["token-out step (replay + sample + sync + readback)"] = (time.perf_counter() - start) / iters * 1e3

        # Terminal pieces measured on their own, outside any trace.
        hidden = ttnn.from_torch(
            torch.randn(1, 1, 32, model.dim).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        def lm_head_only():
            out = model._lm_head(model._final_norm(hidden))
            ttnn.deallocate(out)

        rows["final norm + LM head (eager)"] = timed(lm_head_only, iters, mesh)

        logits = model._lm_head(model._final_norm(hidden))

        def sampler_eager():
            gen.sampling.sample(logits=logits, tt_out_tok=gen._trace_inputs[0], enable_trace=False)

        rows["sampler (eager)"] = timed(sampler_eager, iters, mesh)

        def readback_only():
            gen._read_tokens()

        rows["token readback only"] = timed(readback_only, iters, mesh)

        print("\n=== terminal cost breakdown ===")
        print(
            f"layers={args.layers} iters={iters} pad_pow2={args.pad_pow2} lm_head_columns={args.lm_head_columns} topk_groups={args.topk_groups}"
        )
        for name, value in rows.items():
            print(f"{name:<56} {value:8.3f} ms")
        gen.teardown()
        print("PROBE_OK")
    finally:
        close_ornith_mesh(mesh)


if __name__ == "__main__":
    main()
