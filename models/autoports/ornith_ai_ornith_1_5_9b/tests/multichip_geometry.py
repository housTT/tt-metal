# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reader-one geometry search on actual TP-local decoder activations.

Every candidate is traced; whole-layer correctness and timing remain separate.
Reader two/three are excluded by the verified mesh API blocker in AutoDebug.
"""

import argparse
import json
import math
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch

import ttnn

from ..tt.multichip_decoder import MeshConfig, MultichipDecoder, fabric_router_config
from . import test_functional_decoder as H
from .test_optimization_experiments import recorded_activations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--roles", nargs="*")
    parser.add_argument("--max-block", type=int)
    parser.add_argument("--cores", nargs="+", type=int)
    parser.add_argument("--blocks", nargs="+", type=int)
    parser.add_argument("--grid", default="[8,4]")
    parser.add_argument("--family", choices=["dram", "interleaved"], default="dram")
    args = parser.parse_args()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, router_config=fabric_router_config())
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 * 1024**2)
    rows = []
    try:
        decoder = MultichipDecoder.from_state_dict(
            H.layer_state_dict(args.layer, "real"),
            hf_config=H.hf_config(),
            layer_idx=args.layer,
            mesh_device=mesh,
            max_context=4096,
            mesh_config=MeshConfig(
                decode_grid=None if args.family == "dram" else tuple(json.loads(args.grid)),
                decode_qkvg_dram=args.family == "dram",
            ),
        )
        decoder.allocate_state(1)
        decoder.allocate_kv_cache(64)
        table = H.to_device(
            mesh, torch.arange(64, dtype=torch.int32)[None], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        source = recorded_activations(args.layer)
        ttnn.deallocate(decoder.prefill_forward(H.to_device(mesh, source[:, :2048]), page_table=table))
        captured = {}
        original = decoder._linear

        def collect(x, role, **kwargs):
            saved = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            if x.memory_config() == ttnn.DRAM_MEMORY_CONFIG:
                saved = ttnn.clone(saved)
            captured[role] = saved, x.memory_config(), kwargs.copy()
            return original(x, role, **kwargs)

        decoder._linear = collect
        pos, rot = H.decode_inputs(mesh, torch.tensor([2048]))
        ttnn.deallocate(
            decoder.decode_forward(
                H.to_device(mesh, source[:, 2048:2049]), current_pos=pos, rot_idxs=rot, page_table=table
            )
        )
        decoder._linear = original
        initial = decoder.optimization

        def read(value):
            return [ttnn.to_torch(t) for t in ttnn.get_device_tensors(value)]

        for role, (saved_input, input_memory, kwargs) in captured.items():
            if args.roles and role not in args.roles:
                continue
            x = ttnn.to_memory_config(saved_input, input_memory)
            reference_tensor = original(x, role, **kwargs)
            reference = read(reference_tensor)
            ttnn.deallocate(reference_tensor)
            k, n = decoder.w[role].shape
            configurations = [
                (cores, block)
                for cores in (4, 8, 12, 16, 24, 32, 48, 64, 96)
                if k // 32 % cores == 0
                for block in range(1, k // 32 // cores + 1)
                if (k // 32 // cores) % block == 0
            ]
            if args.family == "interleaved":
                configurations = [
                    (math.prod(json.loads(args.grid)), block)
                    for block in range(1, k // 32 + 1)
                    if (k // 32) % block == 0
                ]
            for cores, block in configurations:
                if args.cores and cores not in args.cores:
                    continue
                if args.blocks and block not in args.blocks:
                    continue
                if args.max_block is not None and block > args.max_block:
                    continue
                decoder.optimization = replace(
                    initial, role_configs={**initial.role_configs, role: dict(cores=cores, block_w=block, readers=1)}
                )
                out = original(x, role, **kwargs)
                actual = read(out)
                ttnn.deallocate(out)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                traced = original(x, role, **kwargs)
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                times = []
                for repeat in range(5):
                    ttnn.synchronize_device(mesh)
                    start = time.perf_counter()
                    for _ in range(64):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    if repeat >= 2:
                        times.append((time.perf_counter() - start) * 1e6 / 64)
                exact = all(torch.equal(a, b) for a, b in zip(actual, read(traced)))
                record = dict(
                    family=args.family,
                    grid=json.loads(args.grid) if args.family == "interleaved" else None,
                    weight_dtype=str(decoder.decode_weights.get(role, decoder.w[role]).dtype),
                    layer=args.layer,
                    role=role,
                    local_shape=[1, k, n],
                    cores=cores,
                    block_w=block,
                    readers=1,
                    trace_us=statistics.median(times),
                    samples_us=times,
                    pcc=[H.pcc(a, b) for a, b in zip(reference, actual)],
                    exact_reference=all(torch.equal(a, b) for a, b in zip(reference, actual)),
                    trace_exact=exact,
                )
                print("GEOMETRY " + json.dumps(record), flush=True)
                rows.append(record)
                Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")
                ttnn.release_trace(mesh, trace)
                ttnn.deallocate(traced)
                assert exact and min(record["pcc"]) >= H.PCC_BAR
            decoder.optimization = initial
            ttnn.deallocate(x)
            if input_memory != ttnn.DRAM_MEMORY_CONFIG:
                ttnn.deallocate(saved_input)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
