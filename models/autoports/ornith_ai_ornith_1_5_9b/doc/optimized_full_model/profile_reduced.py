# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tracy collection uses only real layers 0/3 and the complete terminal/sampling path."""

import argparse
import json
import time
from pathlib import Path

import tracy

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import close_ornith_mesh, open_ornith_mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["decode", "prefill"], required=True)
    parser.add_argument("--plain-sampling", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    mesh = open_ornith_mesh()
    try:
        gen = build_generator(root, mesh, layer_indices=[0, 3], cache_context=None)
        try:
            gen.generate([100] * 128, 2)
            gen.generate([100] * 128, 2)
            ttnn.synchronize_device(mesh)
            before = dict(gen.counters)
            tracy.signpost("start")
            start = time.perf_counter()
            if args.mode == "decode":
                for _ in range(4):
                    gen._replay(collect_output=not args.plain_sampling)
            else:
                gen.generate([100] * 128, 1, stop_on_eos=False)
            ttnn.synchronize_device(mesh)
            elapsed = time.perf_counter() - start
            tracy.signpost("stop")
            counters = {key: gen.counters[key] - before[key] for key in before}
            if args.mode == "prefill":
                assert counters["prefill_replays"] == 1, counters
                assert counters["prefill_sampling_replays"] == 1, counters
                assert counters["prefill_captures"] == counters["prefill_eager_calls"] == 0, counters
            (args.output or Path(__file__).parent / f"profile_{args.mode}_host.json").write_text(
                json.dumps(
                    dict(
                        elapsed_s=elapsed,
                        steps=4 if args.mode == "decode" else 1,
                        layers=[0, 3],
                        collection=args.mode == "decode" and not args.plain_sampling,
                        profiler_enabled=True,
                        counters=gen.counters,
                        window_counters=counters,
                        boundary_synchronizations=2,
                        includes_first_token_readback=args.mode == "prefill",
                        prefill_request_perf=dict(gen.perf) if args.mode == "prefill" else None,
                        window_scope=(
                            "Warmed public generate128/1, including request reset/setup, first-token read and post-first-token position/page/history setup; ttft_s is retained separately."
                            if args.mode == "prefill"
                            else "Four warmed nonblocking model and split-sampler replays."
                        ),
                        head_program=str(gen.model.head_program),
                        head_compute=str(gen.model.head_compute),
                        sharded_final_norm=gen.model.sharded_final_norm,
                    ),
                    indent=2,
                )
                + "\n"
            )
        finally:
            gen.teardown()
    finally:
        close_ornith_mesh(mesh)


if __name__ == "__main__":
    main()
