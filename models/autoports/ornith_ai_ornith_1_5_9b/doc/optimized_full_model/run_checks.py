# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Model-local TP4 entry points for the unchanged common readiness runners."""

import argparse
import json
import sys
import time
from pathlib import Path

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import close_ornith_mesh, open_ornith_mesh
from models.common.readiness_check.run_prefill_check import run_prefill_check
from models.common.readiness_check.run_teacher_forcing import run_teacher_forcing

ROOT = Path(__file__).resolve().parents[2]
DOC = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["smoke", "prefill", "teacher", "perf"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--cache-context", type=int)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--probe-request-reset", action="store_true")
    parser.add_argument("--warm-samples", type=int, default=5)
    parser.add_argument("--use-prefill-trace", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--head-dtype", choices=["bfloat16", "bfloat8_b", "bfloat4_b"], default="bfloat16")
    parser.add_argument("--head-fidelity", choices=["HiFi4", "HiFi2", "LoFi"], default="HiFi4")
    parser.add_argument("--head-columns", type=int, default=32768)
    parser.add_argument("--head-block-w", type=int, default=1)
    parser.add_argument("--head-readers", type=int, default=2)
    parser.add_argument("--sharded-final-norm", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    cache_context = args.cache_context if args.cache_context is not None else (None if args.mode == "perf" else 2048)
    head_kwargs = dict(
        use_prefill_trace=args.use_prefill_trace,
        lm_head_dtype=getattr(ttnn, args.head_dtype),
        lm_head_fidelity=getattr(ttnn.MathFidelity, args.head_fidelity),
        lm_head_columns=args.head_columns,
        lm_head_block_w=args.head_block_w,
        lm_head_readers=args.head_readers,
    )
    if args.sharded_final_norm is not None:
        head_kwargs["sharded_final_norm"] = args.sharded_final_norm
    mesh = open_ornith_mesh()
    result = dict(command=sys.argv, mode=args.mode, mesh=[1, 4], hardware="4 Blackhole chips on two P300c boards")
    start = time.perf_counter()
    try:
        if args.mode in ("prefill", "teacher"):
            run = run_prefill_check if args.mode == "prefill" else run_teacher_forcing
            result["metrics"] = run(
                model_dir=ROOT,
                reference_path=ROOT / "readiness_aime24_chat.refpt",
                mesh_device=mesh,
                build_kwargs={"cache_context": cache_context, **head_kwargs},
            )
            result["pass"] = all(
                row["k"] == 100 and row["top5"] >= 0.98 and row["top100"] == 1 for row in result["metrics"]
            )
        else:
            gen = build_generator(
                ROOT,
                mesh,
                layer_indices=[0, 3] if args.mode == "smoke" else None,
                cache_context=cache_context,
                max_batch_size=args.batch,
                **head_kwargs,
            )
            try:
                prompt = list(range(131)) if args.mode == "smoke" else [100] * args.prompt_length
                count = 8 if args.mode == "smoke" else 128
                first = gen.generate(prompt, count, stop_on_eos=False)
                first_perf = dict(gen.perf)
                warm_runs = []
                for _ in range(args.warm_samples if args.mode == "perf" else 1):
                    again = gen.generate(prompt, count, stop_on_eos=False)
                    assert first == again, (first, again)
                    if args.use_prefill_trace and args.batch == 1 and len(prompt) <= gen.model.prefill_chunk:
                        request = gen.perf["request_counters"]
                        assert request["prefill_replays"] == request["prefill_sampling_replays"] == 1, request
                        assert request["prefill_captures"] == request["prefill_eager_calls"] == 0, request
                    warm_runs.append(dict(gen.perf))
                selected_perf = sorted(warm_runs, key=lambda row: row["ttft_s"])[len(warm_runs) // 2]
                result.update(
                    tokens=again,
                    perf=selected_perf,
                    warm_runs=warm_runs,
                    selected_warm_run="median TTFT; decode metric from that same run",
                    first_perf=first_perf,
                    cache_context=gen.kv_cache.context,
                    prompt_length=len(prompt),
                )
                result["pass"] = True
                positions = ttnn.to_torch(ttnn.get_device_tensors(gen._inputs[1])[0]).tolist()
                result["positions"] = positions
                assert positions[0] == len(prompt) + count - 1
                counters = gen.perf["loop_counters"]
                for key in ("token_refreshes", "position_refreshes", "rope_refreshes", "page_table_refreshes"):
                    assert counters[key] == 0, counters
                assert counters["model_replays"] == counters["sampling_replays"] == count - 1
                if args.mode == "perf":
                    windows = []
                    for _ in range(3):
                        gen.generate(prompt, 1, stop_on_eos=False)
                        ttnn.synchronize_device(mesh)
                        before = dict(gen.counters)
                        begin = time.perf_counter()
                        for _ in range(127):
                            gen._replay()
                        ttnn.synchronize_device(mesh)
                        duration = time.perf_counter() - begin
                        counters = {k: gen.counters[k] - before[k] for k in before}
                        assert all(
                            counters[k] == 0
                            for k in (
                                "token_refreshes",
                                "position_refreshes",
                                "rope_refreshes",
                                "page_table_refreshes",
                                "readbacks",
                                "read_waits",
                                "synchronizations",
                            )
                        ), counters
                        assert gen._read_tokens()[0].item() == again[-1]
                        windows.append(
                            dict(
                                ms_per_token=duration * 1000 / 127,
                                tokens_per_second=127 / duration,
                                loop_counters=counters,
                                boundary_synchronizations=2,
                                final_validation_readbacks=1,
                            )
                        )
                    result["token_out_no_readback"] = windows
                    assert result["perf"]["decode_steps"] == count - 1
                    if args.probe_request_reset:
                        reset = gen.reset

                        def clear_kv(**kwargs):
                            reset(clear_kv=True)

                        def keep_kv(**kwargs):
                            gen.model.reset_cache(gen.kv_cache, clear_kv=False)
                            gen._previous_table = None
                            gen._live = False

                        reset_trials = []
                        try:
                            for preserve in (False, True, False, True, False, True):
                                gen.reset = keep_kv if preserve else clear_kv
                                ttnn.synchronize_device(mesh)
                                output = gen.generate(prompt, 8, stop_on_eos=False)
                                assert output == again[:8]
                                reset_trials.append(dict(preserve_kv=preserve, perf=dict(gen.perf), exact_tokens=True))
                        finally:
                            gen.reset = reset
                        result["request_reset_trials"] = reset_trials
                    # Separate PERF-style logits trace: fixed token, device position advance,
                    # no sampling, no token feedback, one boundary synchronization.
                    gen.reset()
                    logits = gen.prefill_forward(
                        [prompt],
                        page_table=gen.page_table,
                        kv_cache=gen.kv_cache,
                        prompt_lens=[len(prompt)],
                        return_device_logits=True,
                    )
                    ttnn.deallocate(logits)
                    gen._ensure_replay_safe()
                    gen._write_tokens([100])
                    gen._write_positions([len(prompt)])
                    gen._refresh_table(gen.page_table)
                    ttnn.synchronize_device(mesh)
                    begin = time.perf_counter()
                    for _ in range(128):
                        ttnn.execute_trace(mesh, gen._model_trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    duration = time.perf_counter() - begin
                    result["logits_only_trace"] = dict(
                        steps=128,
                        elapsed_s=duration,
                        ms_per_token=duration * 1000 / 128,
                        tokens_per_second=128 / duration,
                        sampling=False,
                        token_feedback=False,
                        host_refreshes_per_token=0,
                    )
            finally:
                gen.teardown()
    finally:
        close_ornith_mesh(mesh)
        result["elapsed_s"] = time.perf_counter() - start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert result["pass"], result


if __name__ == "__main__":
    main()
