# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""One signposted window of the full model's reduced profiling variant, for ``tt-perf-report``.

A standalone script rather than a pytest node, because the profiler crashes in
``close_mesh_device`` when pytest's fixture teardown closes the mesh while the model's device
tensors are still referenced by the test frame. Here the generator is released and collected before
the mesh is closed, which is the ordering the profiler survives.

``tracy/run_profiling.sh`` drives it once per phase.
"""

from __future__ import annotations

import argparse
import gc
import time

import torch
from loguru import logger

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import close_ornith_mesh, open_ornith_mesh

MODEL_DIR = "models/autoports/ornith_ai_ornith_1_0_35b"

#: One real layer of each kind: layer 0 is ``linear_attention``, layer 3 is ``full_attention``.
PROBE_LAYERS = [0, 3]

#: Setup-only. ``allocate_state`` prepares and probes one ``ttnn.conv1d`` program per 128-token block
#: length up to the prefill chunk, so 2048 costs sixteen probe runs per layer and 256 costs two.
#: Neither signposted window contains a program that differs: a 128-token prefill block is below
#: both chunk sizes and resolves the same SDPA config, MoE group size and conv1d length, and decode
#: never sees the chunk. What it buys is a capture ``process_ops_logs`` can reassemble - at 2048 the
#: setup probes alone push the op id past 400 000 and the post-processing fails to match a device row.
PROFILING_PREFILL_CHUNK = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["decode", "prefill"])
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--cache-context", type=int, default=4096)
    args = ap.parse_args()

    from tracy import signpost

    mesh = open_ornith_mesh()
    try:
        gen = build_generator(
            model_dir=MODEL_DIR,
            mesh_device=mesh,
            layer_indices=PROBE_LAYERS,
            max_batch_size=1,
            cache_context=args.cache_context,
            prefill_chunk=PROFILING_PREFILL_CHUNK,
        )
        torch.manual_seed(0)
        prompt = torch.randint(0, gen.model.vocab_size, (1, args.prompt_len))
        # Drain the device profiler after the build. The device-side buffer is finite and
        # `process_ops_logs` fails with *"Device data missing: Op N not present in
        # cpp_device_perf_report.csv"* once it overflows - which the model build alone does, several
        # times over. Flushing before the warm-up, and again immediately before the signpost, keeps
        # the profiled window's rows intact.
        ttnn.ReadDeviceProfiler(mesh)

        if args.phase == "decode":
            # No prefill in the decode phase: it only adds host ops before the window, and the
            # capture's cost is dominated by how many op ids the profiler has to carry.
            gen._ensure_decode_trace()
            ttnn.ReadDeviceProfiler(mesh)

            def step():
                ttnn.execute_trace(mesh, gen._trace_id, cq_id=0, blocking=False)
                gen._sample_traced()

            for _ in range(4):
                step()
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
            signpost("PERF_DECODE")
            start = time.perf_counter()
            for _ in range(args.iters):
                step()
            ttnn.synchronize_device(mesh)
            elapsed = time.perf_counter() - start
            signpost("PERF_DECODE_END")
            logger.info(
                f"FULL-MODEL PERF decode(traced, reduced {PROBE_LAYERS}) iters={args.iters} "
                f"wall/iter={elapsed / args.iters * 1e3:.3f} ms  t/s/u={args.iters / elapsed:.2f}"
            )
        else:
            gen.reset()
            gen.prefill_forward(prompt, page_table=None, kv_cache=None, prompt_lens=[args.prompt_len])
            gen.reset()
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
            signpost("PERF_PREFILL")
            start = time.perf_counter()
            gen.prefill_forward(prompt, page_table=None, kv_cache=None, prompt_lens=[args.prompt_len])
            ttnn.synchronize_device(mesh)
            elapsed = time.perf_counter() - start
            signpost("PERF_PREFILL_END")
            logger.info(
                f"FULL-MODEL PERF prefill({args.prompt_len}, reduced {PROBE_LAYERS}) " f"wall={elapsed * 1e3:.3f} ms"
            )

        gen.teardown()
        del gen
        gc.collect()
        ttnn.synchronize_device(mesh)
        print("PROFILE_OK", flush=True)
    finally:
        close_ornith_mesh(mesh)


if __name__ == "__main__":
    main()
