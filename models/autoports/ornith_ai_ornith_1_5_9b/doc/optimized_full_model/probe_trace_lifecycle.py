# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reduced TP4 trace recapture with deferred public cleanup and memory accounting."""

import argparse
import json
from pathlib import Path

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import close_ornith_mesh, open_ornith_mesh


def trace_bytes(mesh):
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
    return int(view.num_banks * view.total_bytes_allocated_per_bank)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    mesh = open_ornith_mesh()
    report = dict(mesh=[1, 4], hardware="four Blackhole chips on P300c boards", layers=[0, 3], captures=[])
    try:
        gen = build_generator(root, mesh, layer_indices=[0, 3], cache_context=512, sharded_final_norm=True)
        saved_teardown = gen.teardown
        gen.teardown = lambda: None
        try:
            reference = gen.generate([17, 42], 2, stop_on_eos=False)
            initial_bytes = trace_bytes(mesh)
            assert 0 < initial_bytes < 100_000_000
            report["initial_trace_bytes_per_device"] = initial_bytes
            for iteration in range(8):
                gen._capture()
                tokens = gen.generate([17, 42], 2, stop_on_eos=False)
                allocated = trace_bytes(mesh)
                assert allocated == initial_bytes, (iteration, allocated, initial_bytes)
                assert tokens == reference, (iteration, tokens, reference)
                row = dict(iteration=iteration, trace_bytes_per_device=allocated, exact_tokens=True)
                report["captures"].append(row)
                print(json.dumps(row), flush=True)
        finally:
            saved_teardown()
        report["trace_bytes_after_cleanup_per_device"] = trace_bytes(mesh)
        assert report["trace_bytes_after_cleanup_per_device"] == 0
        report["pass"] = True
    finally:
        close_ornith_mesh(mesh)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
