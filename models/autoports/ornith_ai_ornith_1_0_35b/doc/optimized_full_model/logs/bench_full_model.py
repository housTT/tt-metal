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
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import (
    TOPK_GROUP_MACHINERY_US_PER_GROUP,
    TOPK_US_PER_WIDTH_UNIT,
    close_ornith_mesh,
    open_ornith_mesh,
)

MODEL_DIR = Path("models/autoports/ornith_ai_ornith_1_0_35b")

#: Per-layer warmed traced decode from ``doc/optimized_multichip_decoder/README.md`` §1 (the `after`
#: column), in milliseconds. The lower bound for the whole stack is these times their layer counts;
#: anything the full model adds on top is embedding + final norm + LM head + sampling + loop.
DECODER_STAGE_MS = {"linear_attention": 0.564, "full_attention": 0.453}



#: Per-device DRAM bandwidth, in bytes/second, taken from the profiler's own denominator rather than a
#: datasheet: the previous stage's capture reports its LM-head row at 355 GB/s = 69.3 %, i.e. 512.3 GB/s
#: at 100 %. This stage's four LM-head rows read 353 GB/s = 69.0 %, which implies 511.6 - a 0.14 %
#: difference that would move the roofline estimate from 1.464 to 1.466 ms/token and leave the achieved
#: fraction at 6.3 %. The larger denominator is kept because it is the conservative one (it makes the
#: achieved fraction smaller) and because it keeps this stage comparable with the previous one.
DEVICE_DRAM_BYTES_PER_S = 512.3e9

#: Routed experts actually read per token, per device, out of the 64 each device owns under EP=4.
#: This is the profiled number, not a derivation from the gate's global top-k: the sparsity tensor is
#: the union over the call's valid rows plus `MOE_MASK_FLOOR`'s guaranteed expert, and the decoder
#: stage's captures and this stage's report both show `active=4/64` at batch-1 decode.
ACTIVE_EXPERTS_PER_DEVICE = 4

#: Bytes per element on device, including the shared exponent the block-float dtypes carry.
_DTYPE_BYTES = {"DataType.BFLOAT16": 2.0, "DataType.BFLOAT8_B": 1.0625, "DataType.BFLOAT4_B": 0.5625,
                "DataType.FLOAT32": 4.0, "DataType.UINT32": 4.0, "DataType.INT32": 4.0}


def _tensor_bytes(tensor) -> float:
    """Device bytes of a weight entry. Some entries are lists of per-role tensors."""
    if isinstance(tensor, (list, tuple)):
        return sum(_tensor_bytes(t) for t in tensor)
    if not hasattr(tensor, "shape"):
        return 0.0
    n = 1
    for d in tensor.shape:
        n *= int(d)
    return n * _DTYPE_BYTES.get(str(tensor.dtype), 2.0)


def _performance_accounting(model, *, ttft_ms, e2e_ms, device_scope_ms):
    """Roofline / device-time / end-to-end, per the `$optimize` performance-accounting contract."""
    cfg = model.cfg
    per_device_bytes = 0.0
    experts_per_device = max(1, cfg.num_experts // model.tp)
    active_fraction = ACTIVE_EXPERTS_PER_DEVICE / experts_per_device
    routed_keys = ("expert_gate_up", "expert_down")
    for layer in model.layers:
        for name, weight in layer.moe.w.items():
            share = active_fraction if name in routed_keys else 1.0
            per_device_bytes += _tensor_bytes(weight) * share
        for name, weight in layer.w.items():
            if name in layer.moe.w:
                continue
            per_device_bytes += _tensor_bytes(weight)
    # The terminal path: the LM-head shard, the final norm, and the one embedding row a token reads.
    for weight in model.lm_head_weights:
        per_device_bytes += _tensor_bytes(weight) / model.tp
    per_device_bytes += _tensor_bytes(model.norm_weight)
    per_device_bytes += model.dim * 2.0
    roofline_ms = per_device_bytes / DEVICE_DRAM_BYTES_PER_S * 1e3
    return {
        "workload": {"profile": "single_user_decode", "prompt_len": 128, "gen_len": 128, "batch": 1},
        "ttft_ms": ttft_ms,
        "decode_ms_per_token_e2e": e2e_ms,
        "decode_ms_per_token_device": device_scope_ms,
        "decode_ms_per_token_device_scope": (
            "not measurable for the 40-layer stack: `$optimize` and `$full-model` both forbid Tracy on the "
            "full stack, and the reduced two-layer capture's per-replay device time is inflated by the "
            "profiler itself (naively scaling its whole window by the layer ratio exceeds the un-profiled "
            "40-layer wall clock; a layer-scoped scaling would produce a number, but an estimate rather than "
            "a measurement of the delivered stack). "
            "tracy/decode_perf_report.summary.txt is the reduced window's device split; the ms/token figures "
            "here are un-profiled wall clock, which is the right basis for them"
        ),
        "roofline_ms_per_token_estimate": roofline_ms,
        "roofline_bytes_per_token_per_device": per_device_bytes,
        "roofline_bytes_note": (
            "summed from the live device tensors, so per-device by construction: every entry of each layer's "
            "weight dicts (mesh-sharded tensors report their per-device shape), the routed experts scaled to "
            "the profiled active fraction, plus the LM-head shard (its tensor carries the global width, hence "
            "the /tp), the final norm and one embedding row. It is deliberately SMALLER than "
            "footprint.json's measured resident set, which counts the whole paged KV cache, all 64 local "
            "experts and the RoPE tables - none of which a single decode step reads in full"
        ),
        "roofline_fraction_achieved": roofline_ms / e2e_ms,
        "roofline_note": (
            "weights at their stored dtypes, routed experts counted at the profiled active fraction "
            f"({ACTIVE_EXPERTS_PER_DEVICE}/{experts_per_device} per device), plus the LM-head shard, the final "
            "norm and one "
            "embedding row; KV-cache reads are excluded because at the 128-token benchmark context they are "
            "under 0.1 % of the total"
        ),
        "named_limitations": [
            "the decode step is launch-bound, not bandwidth-bound: ~100 device ops per layer at one tile of M, "
            f"which is why it sits at {roofline_ms / e2e_ms * 100:.1f} % of the DRAM roofline (the decoder stage "
            "measured ~7 % for the layer alone and named the same cause)",
            "ttnn.sparse_matmul parallelism is capped by the output tile count and it loops once per active "
            "expert at a single tile of M, so the two routed projections stay far below both rooflines "
            "(inherited limitation, doc/optimized_decoder/README.md)",
            "the sampler's grouped local top-k is bounded by two measured terms - 0.188 us per unit of reduced "
            "width and 5.52 us/replay per group of slice/concat/gather machinery "
            "(doc/optimized_full_model/logs/sampler_cost_model.md) - and the shipped 32 groups is the joint "
            "optimum of both. The remaining ~177 us/replay of machinery would need a reshape-based grouping in "
            "shared TTSampling code",
            "prefill is eager and most of a 128-token TTFT is length-independent - the exact share depends on "
            "whether the ladder is fitted by a two-point secant or by least squares, and "
            "doc/optimized_full_model/prefill_profile.json reports both rather than this string picking one - "
            "and a captured prefill trace is blocked by the decoder's logical-length-keyed prefill program set",
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--gen-len", type=int, default=128)
    ap.add_argument("--cache-context", type=int, default=8192)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--iters", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--output", default=str(MODEL_DIR / "doc" / "optimized_full_model" / "perf_summary.json"))
    ap.add_argument(
        "--arm",
        default="optimized",
        choices=["optimized", "inherited"],
        help=(
            "`inherited` rebuilds the pre-optimization path from constructor knobs - untuned interleaved LM "
            "head, unsharded terminal norm, no sampler-friendly vocabulary alignment, serial readback loop - "
            "so before/after can be measured in ONE session with the same repeat count. TTFT on this host has "
            "a spread far wider than the effect, so a 3-sample archive figure cannot be compared with a "
            "9-sample one."
        ),
    )
    args = ap.parse_args()

    mesh = open_ornith_mesh()
    try:
        kwargs = {}
        if args.layers:
            kwargs["layer_indices"] = [int(v) for v in args.layers.split(",")]
        if args.arm == "inherited":
            kwargs.update(
                lm_head_program="interleaved",
                terminal_norm_sharded=False,
                vocab_align_tiles=1,
                pipelined_readback=False,
            )
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

        # Serial (pre-optimization) token-out step, measured in the same build so the pipelined
        # loop's win is a difference of two rows of one run rather than of two runs.
        def serial_step():
            ttnn.execute_trace(mesh, gen._trace_id, cq_id=0, blocking=False)
            gen._sample_traced()
            ttnn.synchronize_device(mesh)
            gen._read_tokens()

        serial_step()
        start = time.perf_counter()
        for _ in range(args.iters):
            serial_step()
        serial_token_out_ms = (time.perf_counter() - start) / args.iters * 1e3

        # TTFT breakdown, warmed, through the same public pieces `generate` uses.
        ttft_parts = {}
        for _ in range(3):
            gen.reset()
            gen._ensure_decode_trace()
            t0 = time.perf_counter()
            page_row = gen._prefill_page_row(0)
            t1 = time.perf_counter()
            device_logits = model.prefill_request_into_slot(
                prompt, page_table=page_row, slot=0, start_pos=0, return_logits="device"
            )
            ttnn.synchronize_device(mesh)
            t2 = time.perf_counter()
            first = gen._first_token_after_prefill(device_logits)
            if page_row is not None:
                ttnn.deallocate(page_row)
            ttnn.synchronize_device(mesh)
            t3 = time.perf_counter()
            part = {
                "page_row_upload_ms": (t1 - t0) * 1e3,
                "prefill_ms": (t2 - t1) * 1e3,
                "first_token_sampling_ms": (t3 - t2) * 1e3,
                "total_ms": (t3 - t0) * 1e3,
                "first_token": int(first),
            }
            if not ttft_parts or part["total_ms"] < ttft_parts["total_ms"]:
                ttft_parts = part
        gen.reset()

        # What a *cold* prompt length costs, which is the term the full-model stage left unmeasured:
        # a length nothing has compiled yet pays the prefill compile plus one trace re-capture, and
        # neither lands in TTFT or in the decode window. `warmup` is the fix; this measures both
        # sides of it. The length is deliberately non-aligned.
        cold_len = args.prompt_len + 7
        cold_prompt = torch.randint(0, model.vocab_size, (cold_len,)).tolist()
        recaptures_before = gen.trace_recaptures
        t0 = time.perf_counter()
        gen.generate(prompt_token_ids=cold_prompt, max_new_tokens=2, enable_trace=True, stop_on_eos=False)
        cold_first_request_s = time.perf_counter() - t0
        cold_recaptures = gen.trace_recaptures - recaptures_before
        cold_ttft_ms = gen.perf["ttft_s"] * 1e3
        warm_after = []
        for _ in range(3):
            gen.generate(prompt_token_ids=cold_prompt, max_new_tokens=2, enable_trace=True, stop_on_eos=False)
            warm_after.append(gen.perf["ttft_s"] * 1e3)
        assert gen.trace_recaptures == recaptures_before + cold_recaptures, "a warmed length must not re-capture"
        cold_length_cost = {
            "length": cold_len,
            "first_request_wall_s": cold_first_request_s,
            "first_request_ttft_ms": cold_ttft_ms,
            "trace_recaptures": cold_recaptures,
            "warmed_ttft_ms": min(warm_after),
            "hidden_cost_ms": cold_first_request_s * 1e3 - cold_ttft_ms,
        }

        kinds = [model.cfg.layer_kind(i) for i in model.layer_indices]
        lower_bound_ms = sum(DECODER_STAGE_MS[k] for k in kinds)

        summary = {
            "workload": {
                "prompt_len": args.prompt_len,
                "gen_len": args.gen_len,
                "batch": 1,
                "cache_context": args.cache_context,
                "profile": "vLLM primary single-user (prompt 128 / generate 128)",
                "arm": args.arm,
                "repeats": args.repeats,
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
            "serial_token_out_decode": {
                "ms_per_token": serial_token_out_ms,
                "t/s/u": 1e3 / serial_token_out_ms,
                "includes": "replay + sample + synchronize_device + readback, i.e. the pre-optimization loop",
            },
            # Only meaningful for the optimized arm: in the `inherited` arm both loops are serial, so the
            # difference would be methodology noise between `serial_step()` and `generate()` rather than a
            # pipelining saving, and reporting a number there invites misreading.
            "pipelined_readback_saving_ms": (
                serial_token_out_ms - best["decode_ms_per_token"] if args.arm == "optimized" else None
            ),
            "ttft_breakdown_ms": ttft_parts,
            "cold_prompt_length_cost": cold_length_cost,
            # The canonical `$optimize` performance-accounting block, alongside the richer rows above.
            # `roofline_ms_per_token_estimate` is derived from the model's own tensors rather than
            # from a table: every dense projection weight in a layer, the *active* routed-expert
            # weights only (the gate selects `num_experts_per_tok` of `num_experts`, so a dense
            # all-expert count would be the wrong roofline for this model), the LM head, and the
            # embedding row, divided by the mesh's aggregate DRAM bandwidth.
            "performance_accounting": _performance_accounting(
                model,
                ttft_ms=ttfts[0],
                e2e_ms=best["decode_ms_per_token"],
                device_scope_ms=None,
            ),
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
