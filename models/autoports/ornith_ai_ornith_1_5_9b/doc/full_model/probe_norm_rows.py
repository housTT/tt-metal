# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Frozen real Q/K same-op row control; alternatives are diagnostic only."""

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.probe_batch32 import difference, ranks
from models.autoports.ornith_ai_ornith_1_5_9b.tt.fused_decoder import _batch_grid, _height_memory
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import close_ornith_mesh, open_ornith_mesh

DOC = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-canonical", action="store_true")
    args = parser.parse_args()
    saved = torch.load(DOC / "batch32_boundaries_norm.pt", weights_only=True)["snapshots"]
    report = {}
    mesh = open_ornith_mesh()
    try:
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))])
        memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [1024, 32], ttnn.ShardOrientation.ROW_MAJOR),
        )
        program = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 1),
            block_h=32,
            block_w=1,
            subblock_w=1,
            inplace=False,
        )
        with safe_open("/home/hous/dev/ornith-1.5-9b/upstream/model-00001-of-00004.safetensors", framework="pt") as f:
            for label in ("q", "k"):
                count = 4 if label == "q" else 1
                raw = torch.stack([value[0].reshape(1, count, 256).repeat(32, 1, 1) for value in saved[label + "_raw"]])
                weight = (
                    f.get_tensor(f"model.language_model.layers.3.self_attn.{label}_norm.weight").float() + 1
                ).bfloat16()
                x = ttnn.from_torch(
                    raw,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=memory,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
                )
                w = ttnn.from_torch(
                    weight.reshape(1, 1, 1, 256),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
                for variant, config in [
                    ("original", None),
                    (
                        "fp32_acc_control",
                        ttnn.WormholeComputeKernelConfig(
                            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True, fp32_dest_acc_en=True
                        ),
                    ),
                ]:
                    kw = {} if config is None else {"compute_kernel_config": config}
                    out = ttnn.rms_norm(x, weight=w, epsilon=1e-6, program_config=program, memory_config=memory, **kw)
                    values = [v.reshape(32, count, 256) for v in ranks(out)]
                    result = report[label + "_" + variant] = {}
                    for rank, v in enumerate(values):
                        oracle = (
                            raw[rank] * torch.rsqrt(raw[rank].square().mean(-1, keepdim=True) + 1e-6) * weight.float()
                        )
                        result[f"rank{rank}_oracle"] = difference(oracle, v)
                        for row in (3, 6, 30):
                            result[f"rank{rank}_slot{row}"] = difference(v[0], v[row])
                        if variant == "original":
                            expected = saved[label + "_norm"][rank].reshape(4, count, 256)
                            result[f"rank{rank}_original_trace_rows"] = difference(expected, v[[0, 3, 6, 30]])
                            if args.expect_canonical:
                                assert torch.equal(v, v[0:1].expand_as(v)), (label, rank, "batch rows differ")
                                assert torch.equal(expected[0], v[0]), (label, rank, "canonical anchor changed")
                    ttnn.deallocate(out)
                height_memory = _height_memory(_batch_grid(mesh, 32), 256)
                height = ttnn.to_memory_config(x, height_memory)
                size = mesh.compute_with_storage_grid_size()
                height_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=(size.x, size.y),
                    block_h=1,
                    block_w=8,
                    subblock_w=8,
                    inplace=False,
                )
                for variant, source, mem, prog in [
                    ("height_control", height, height_memory, height_program),
                    ("width_timing", x, memory, program),
                ]:
                    try:
                        out = ttnn.rms_norm(source, weight=w, epsilon=1e-6, program_config=prog, memory_config=mem)
                    except RuntimeError as error:
                        if variant != "height_control" or "Height sharded inputs are not supported" not in str(error):
                            raise
                        report[label + "_" + variant] = {
                            "rejected": "Native validation: Height sharded inputs are not supported."
                        }
                        continue
                    vals = [v.reshape(32, count, 256) for v in ranks(out)]
                    result = report[label + "_" + variant] = {
                        f"rank{rank}_slot{row}": difference(v[0], v[row])
                        for rank, v in enumerate(vals)
                        for row in (3, 6, 30)
                    }
                    ttnn.deallocate(out)
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    out = ttnn.rms_norm(source, weight=w, epsilon=1e-6, program_config=prog, memory_config=mem)
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    ttnn.synchronize_device(mesh)
                    start = time.perf_counter()
                    for _ in range(128):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    result["trace_ms"] = (time.perf_counter() - start) * 1000 / 128
                    ttnn.release_trace(mesh, trace)
                    ttnn.deallocate(out)
                ttnn.deallocate(height)
                if args.expect_canonical:
                    one_memory = ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                        ttnn.BufferType.L1,
                        ttnn.ShardSpec(grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
                    )
                    one = ttnn.from_torch(
                        raw[:, :1].contiguous(),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh,
                        memory_config=one_memory,
                        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
                    )
                    one_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
                        compute_with_storage_grid_size=(8, 1),
                        block_h=1,
                        block_w=1,
                        subblock_w=1,
                        inplace=False,
                    )
                    one_out = ttnn.rms_norm(
                        one, weight=w, epsilon=1e-6, program_config=one_program, memory_config=one_memory
                    )
                    report[label + "_batch1_anchor"] = []
                    for rank, value in enumerate(ranks(one_out)):
                        expected = saved[label + "_norm"][rank][0].reshape(1, 1, count, 256)
                        metric = difference(expected, value)
                        report[label + "_batch1_anchor"].append(metric)
                        assert metric["equal"], (label, rank, "batch1 canonical anchor changed")
                    ttnn.deallocate(one_out)
                    ttnn.deallocate(one)
                ttnn.deallocate(x)
                ttnn.deallocate(w)
    finally:
        close_ornith_mesh(mesh)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
