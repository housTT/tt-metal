# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Warmed full-model performance at the vLLM primary single-user profile (prompt 128 / generate 128).

Reports every figure the `$full-model` skill asks for, each measured on its own so none of them is a
difference of two others:

* **TTFT** — warmed prefill wall clock through the *public* generator path, including the embedding,
  the 40-layer stack, the final norm, the LM head and the on-device sampling of the first token.
  Warmed means the prompt length's programs are already compiled and the traces already re-captured
  for it, which is what a served request sees after the first one at that length;
* **token-out decode** — the delivered path: model trace replay + sampling trace replay +
  synchronize + the caller's token readback;
* **traced logits-only decode** — model trace replay alone. This is the PERF-style figure that is
  comparable with the decoder stage's per-layer traced decode, and it is *not* what a generator or a
  server sees;
* **layer-stack lower bound** — the decoder stage's own per-layer traced decode latencies times the
  layer counts, so the full-model-only cost is a named number rather than an impression;
* **host-work counters** for the steady-state decode loop.

    python .../doc/full_model/logs/bench_full_model.py --prompt-len 128 --gen-len 128
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import close_ornith_mesh, open_ornith_mesh

MODEL_DIR = Path("models/autoports/ornith_ai_ornith_1_0_35b")

#: Per-layer warmed traced decode from ``doc/optimized_multichip_decoder/README.md`` §1 (the `after`
#: column), in milliseconds. The lower bound for the whole stack is these times their layer counts;
#: anything the full model adds on top is embedding + final norm + LM head + sampling + loop.
DECODER_STAGE_MS = {"linear_attention": 0.564, "full_attention": 0.453}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--gen-len", type=int, default=128)
    ap.add_argument("--cache-context", type=int, default=8192)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--iters", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--output", default=str(MODEL_DIR / "doc" / "full_model" / "perf_summary.json"))
    args = ap.parse_args()

    mesh = open_ornith_mesh()
    try:
        kwargs = {}
        if args.layers:
            kwargs["layer_indices"] = [int(v) for v in args.layers.split(",")]
        build_started = time.perf_counter()
        gen = build_generator(
            model_dir=MODEL_DIR,
            mesh_device=mesh,
            max_batch_size=1,
            cache_context=args.cache_context,
            **kwargs,
        )
        build_s = time.perf_counter() - build_started
        model = gen.model
        capability = model.capability()

        torch.manual_seed(0)
        prompt = torch.randint(0, model.vocab_size, (args.prompt_len,)).tolist()

        # Warm: compile this prompt length's programs and let the traces settle, exactly as the
        # second and later requests at a given length see them.
        gen.generate(prompt_token_ids=prompt, max_new_tokens=4, enable_trace=True)
        warm_recaptures = gen.trace_recaptures

        runs = []
        for _ in range(args.repeats):
            gen.generate(prompt_token_ids=prompt, max_new_tokens=args.gen_len, enable_trace=True)
            runs.append(dict(gen.perf))
        assert gen.trace_recaptures == warm_recaptures, "a warmed run should never re-capture"

        best = min(runs, key=lambda r: r["decode_ms_per_token"])
        ttfts = sorted(r["ttft_s"] * 1e3 for r in runs)

        # Traced logits-only decode: the model trace on its own.
        def replay_only():
            ttnn.execute_trace(mesh, gen._trace_id, cq_id=0, blocking=False)

        replay_only()
        ttnn.synchronize_device(mesh)
        start = time.perf_counter()
        for _ in range(args.iters):
            replay_only()
        ttnn.synchronize_device(mesh)
        logits_only_ms = (time.perf_counter() - start) / args.iters * 1e3

        # Model trace + sampling trace, no readback.
        def replay_and_sample():
            ttnn.execute_trace(mesh, gen._trace_id, cq_id=0, blocking=False)
            gen._sample_traced()

        replay_and_sample()
        ttnn.synchronize_device(mesh)
        start = time.perf_counter()
        for _ in range(args.iters):
            replay_and_sample()
        ttnn.synchronize_device(mesh)
        sampled_no_readback_ms = (time.perf_counter() - start) / args.iters * 1e3

        kinds = [model.cfg.layer_kind(i) for i in model.layer_indices]
        lower_bound_ms = sum(DECODER_STAGE_MS[k] for k in kinds)

        summary = {
            "workload": {
                "prompt_len": args.prompt_len,
                "gen_len": args.gen_len,
                "batch": 1,
                "cache_context": args.cache_context,
                "profile": "vLLM primary single-user (prompt 128 / generate 128)",
            },
            "capability": capability,
            "build_s": build_s,
            "ttft_ms": {"min": ttfts[0], "median": ttfts[len(ttfts) // 2], "max": ttfts[-1]},
            "token_out_decode": {
                "ms_per_token": best["decode_ms_per_token"],
                "t/s/u": best["decode_t/s/u"],
                "steps": best["decode_steps"],
                "includes": "model trace replay + sampling trace replay + synchronize + token readback",
            },
            "traced_logits_only_decode": {
                "ms_per_token": logits_only_ms,
                "t/s/u": 1e3 / logits_only_ms,
                "includes": "model trace replay only (embedding, 40 layers, final norm, LM head, plus_one)",
            },
            "traced_decode_plus_sampling_no_readback": {
                "ms_per_token": sampled_no_readback_ms,
                "t/s/u": 1e3 / sampled_no_readback_ms,
            },
            "layer_stack_lower_bound": {
                "per_layer_ms": DECODER_STAGE_MS,
                "layer_counts": {k: kinds.count(k) for k in set(kinds)},
                "ms_per_token": lower_bound_ms,
                "t/s/u": 1e3 / lower_bound_ms,
                "source": "doc/optimized_multichip_decoder/README.md section 1, 'after' column",
            },
            "full_model_only_cost": {
                "logits_only_minus_lower_bound_ms": logits_only_ms - lower_bound_ms,
                "sampling_ms": sampled_no_readback_ms - logits_only_ms,
                "sync_and_readback_ms": best["decode_ms_per_token"] - sampled_no_readback_ms,
            },
            "trace_recaptures_total": gen.trace_recaptures,
            "steady_state_counters": best["counters"],
            "runs": runs,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, default=str))
        gen.teardown()
        print("BENCH_OK")
    finally:
        close_ornith_mesh(mesh)


if __name__ == "__main__":
    main()
