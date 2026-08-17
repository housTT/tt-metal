# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""One arm of the optimized-full-model terminal-path A/B, on the reduced two-layer variant.

Every arm is a separate process so a build cannot inherit the previous arm's device state, and
every arm prints one ``ARM_JSON {...}`` line that ``ab_terminal.sh`` collects. The rows measured
are the ones the full-model-only cost is made of:

* ``final_norm``      - the terminal RMSNorm on its own, eager, on a real decode-shaped activation;
* ``final_norm_head`` - that norm plus the LM head, eager, same activation;
* ``model_trace``     - the captured model trace replayed back to back (no sampling, no readback);
* ``sampling_trace``  - the captured sampling trace replayed back to back;
* ``token_out_serial``    - replay + sample + ``synchronize_device`` + token readback, per step;
* ``token_out_pipelined`` - the same work with the readback issued non-blocking behind the replay
  and waited on only after the next step has been enqueued.

Greedy tokens for a fixed prompt are printed as well, so an arm that changes the terminal
arithmetic (dtype, program config, vocabulary padding) is checked for *output*, not only latency.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import close_ornith_mesh, open_ornith_mesh

MODEL_DIR = "models/autoports/ornith_ai_ornith_1_0_35b"


def timed(fn, iters, mesh):
    fn()
    ttnn.synchronize_device(mesh)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - start) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--iters", type=int, default=64)
    ap.add_argument("--cache-context", type=int, default=4096)
    ap.add_argument("--lm-head-program", default="interleaved")
    ap.add_argument("--lm-head-cores", type=int, default=32)
    ap.add_argument("--lm-head-dtype", default=None)
    ap.add_argument("--lm-head-max-columns", type=int, default=None)
    ap.add_argument("--vocab-align-tiles", type=int, default=1)
    ap.add_argument("--terminal-norm-cores", type=int, default=None)
    ap.add_argument("--lm-head-in0-block-w", type=int, default=None)
    ap.add_argument("--lm-head-fidelity", default=None, choices=["lofi", "hifi2", "hifi4"])
    ap.add_argument("--terminal-norm-sharded", default="1")
    ap.add_argument("--topk-groups", type=int, default=None)
    args = ap.parse_args()

    kwargs = {
        "lm_head_program": args.lm_head_program,
        "lm_head_cores": args.lm_head_cores,
        "terminal_norm_sharded": args.terminal_norm_sharded == "1",
        "vocab_align_tiles": args.vocab_align_tiles,
    }
    if args.terminal_norm_cores:
        kwargs["terminal_norm_cores"] = args.terminal_norm_cores
    if args.lm_head_in0_block_w:
        kwargs["lm_head_in0_block_w"] = args.lm_head_in0_block_w
    if args.lm_head_fidelity:
        kwargs["lm_head_fidelity"] = args.lm_head_fidelity
    if args.layers != "all":
        kwargs["layer_indices"] = [int(v) for v in args.layers.split(",")]
    if args.lm_head_dtype:
        kwargs["lm_head_dtype"] = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "bf16": ttnn.bfloat16}[
            args.lm_head_dtype
        ]
    if args.lm_head_max_columns:
        kwargs["lm_head_max_columns"] = args.lm_head_max_columns
    if args.topk_groups is not None:
        kwargs["topk_num_groups"] = args.topk_groups

    row = {"arm": args.arm, **{k: str(v) for k, v in kwargs.items() if k != "layer_indices"}}
    mesh = open_ornith_mesh((1, 4))
    try:
        built = time.perf_counter()
        gen = build_generator(model_dir=MODEL_DIR, mesh_device=mesh, max_batch_size=1, cache_context=args.cache_context, **kwargs)
        row["build_s"] = round(time.perf_counter() - built, 1)
        model = gen.model
        row["padded_vocab_size"] = model.padded_vocab_size
        row["vocab_size"] = model.vocab_size
        row["topk_groups"] = int(gen.sampling.tt_sampling.topk_num_groups)
        cfg, act_mem = model._lm_head_cfg(32)
        row["in0_block_w"] = int(getattr(cfg, "in0_block_w", 0)) if cfg is not None else None
        row["per_core_N"] = int(getattr(cfg, "per_core_N", 0)) if cfg is not None else None
        row["out_subblock"] = (
            f"{int(getattr(cfg, 'out_subblock_h', 0))}x{int(getattr(cfg, 'out_subblock_w', 0))}" if cfg is not None else None
        )
        row["norm_cores"] = int(model._terminal_norm_cores)
        row["fidelity"] = str(model.lm_head_fidelity).split(".")[-1]

        torch.manual_seed(0)
        prompt = torch.randint(0, model.vocab_size, (128,)).tolist()
        gen.generate(prompt_token_ids=prompt, max_new_tokens=4, enable_trace=True)
        row["greedy16"] = [int(t) for t in gen.generate(prompt_token_ids=prompt, max_new_tokens=16, enable_trace=True)]

        iters = args.iters
        row["model_trace"] = timed(
            lambda: ttnn.execute_trace(mesh, gen._trace_id, cq_id=0, blocking=False), iters, mesh
        )
        row["sampling_trace"] = timed(lambda: gen._sample_traced(), iters, mesh)

        def serial():
            ttnn.execute_trace(mesh, gen._trace_id, cq_id=0, blocking=False)
            gen._sample_traced()
            ttnn.synchronize_device(mesh)
            gen._read_tokens()

        serial()
        start = time.perf_counter()
        for _ in range(iters):
            serial()
        row["token_out_serial"] = (time.perf_counter() - start) / iters * 1e3

        def pipelined(n):
            pending = None
            for _ in range(n):
                ttnn.execute_trace(mesh, gen._trace_id, cq_id=0, blocking=False)
                gen._sample_traced()
                nxt = gen._read_tokens_async()
                if pending is not None:
                    gen._finish_read(pending)
                pending = nxt
            if pending is not None:
                gen._finish_read(pending)
            ttnn.synchronize_device(mesh)

        pipelined(4)
        start = time.perf_counter()
        pipelined(iters)
        row["token_out_pipelined"] = (time.perf_counter() - start) / iters * 1e3

        hidden = ttnn.from_torch(
            torch.randn(1, 1, 32, model.dim).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        def norm_only():
            out = model._final_norm(hidden)
            ttnn.deallocate(out)

        def norm_head():
            normed = model._final_norm(hidden)
            out = model._lm_head(normed)
            ttnn.deallocate(normed)
            ttnn.deallocate(out)

        row["final_norm"] = timed(norm_only, iters, mesh)
        row["final_norm_head"] = timed(norm_head, iters, mesh)
        ttnn.deallocate(hidden)

        # Warmed TTFT through the public path, for the prefill half of the stage.
        ttfts = []
        for _ in range(3):
            gen.generate(prompt_token_ids=prompt, max_new_tokens=4, enable_trace=True)
            ttfts.append(gen.perf["ttft_s"] * 1e3)
        row["ttft_ms"] = min(ttfts)
        row["trace_recaptures"] = gen.trace_recaptures
    finally:
        close_ornith_mesh(mesh)

    print("ARM_JSON " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
