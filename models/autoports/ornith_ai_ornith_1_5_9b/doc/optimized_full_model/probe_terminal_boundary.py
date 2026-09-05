# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Compare the removed DRAM hop with direct selected-layout terminal input."""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import OrnithModel, close_ornith_mesh, open_ornith_mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=32)
    args = parser.parse_args()
    doc = Path(__file__).resolve().parent
    mesh = open_ornith_mesh()
    results = []
    try:
        model = OrnithModel(None, mesh, layer_indices=[], max_context=2048)
        hidden = torch.load(doc.parent / "full_model/french_head_v1/hidden.pt", weights_only=True)["hidden"][0]
        extra = 221952 - ttnn.get_memory_view(mesh, ttnn.BufferType.L1).total_bytes_allocated_per_bank
        grid = mesh.compute_with_storage_grid_size()
        memory = ttnn.create_sharded_memory_config(
            shape=(1, extra // 2),
            core_grid=ttnn.CoreGrid(x=grid.x, y=grid.y),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        resident = model.upload(
            torch.zeros(grid.x * grid.y, extra // 2, dtype=torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT, memory=memory
        )
        for batch in (1, 4, 32):
            x = model.upload(hidden.repeat(1, batch, 1))
            if batch == 1:
                x = ttnn.to_memory_config(x, model.final_norm_memory)
            baseline = None
            for mode in ("old_dram_hop", "direct", "old_dram_hop", "direct"):

                def forward():
                    y = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG) if mode == "old_dram_hop" else x
                    return model.terminal(ttnn.reshape(y, [1, batch, model.dim]))

                eager = forward()
                scores = model.logits_to_host(eager, batch)
                if baseline is None:
                    baseline = scores
                assert torch.equal(scores, baseline)
                ttnn.deallocate(eager)
                ttnn.synchronize_device(mesh)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                output = forward()
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                start = time.perf_counter()
                for _ in range(args.repeats):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                elapsed = time.perf_counter() - start
                assert torch.equal(model.logits_to_host(output, batch), baseline)
                results.append(
                    dict(
                        batch=batch,
                        mode=mode,
                        eager_and_trace_exact=True,
                        trace_ms=elapsed * 1000 / args.repeats,
                        input_memory=str(x.memory_config()),
                    )
                )
                ttnn.release_trace(mesh, trace)
                ttnn.deallocate(output)
            ttnn.deallocate(x)
        ttnn.deallocate(resident)
    finally:
        close_ornith_mesh(mesh)
        args.output.write_text(json.dumps(dict(results=results, passed=len(results) == 12), indent=2) + "\n")


if __name__ == "__main__":
    main()
