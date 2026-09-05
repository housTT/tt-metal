# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real-input eager AG localization and isolated AG trace control; serialized lane only."""

import argparse
import json
from dataclasses import asdict

import torch

import ttnn

from ..tt.functional_decoder import num_blocks_for_context
from ..tt.multichip_decoder import MeshConfig, fabric_router_config
from . import test_functional_decoder as H
from .multichip_topology_candidates import GatherOutputProjection
from .multichip_z_diagnostic import metadata, read_report
from .test_optimization_experiments import recorded_activations


class InspectGather(GatherOutputProjection):
    """Read each eager boundary immediately; these reads deliberately synchronize."""

    def _gather_unobserved(self, tensor):
        extra = {}
        if self.gather_workers is not None:
            extra["num_workers_per_link"] = self.gather_workers
        if self.gather_core_grid:
            extra["sub_core_grids"] = self.ccl.ccl_cores
        return ttnn.experimental.all_gather_async(
            tensor,
            dim=3,
            persistent_output_buffer=None,
            multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            num_links=self.mesh_config.async_links,
            topology=ttnn.Topology.Ring,
            memory_config=tensor.memory_config(),
            **extra,
        )

    def _gather(self, tensor):
        index = len(self.gather_records)
        label = f"{self.phase}_ag{index}"
        source, _ = read_report(label + "_input", tensor)
        record = dict(source=source, dtype=tensor.dtype, metadata=metadata(tensor))
        gathered = self._gather_unobserved(tensor)
        reference = torch.cat(source, dim=-1)
        _, rows = read_report(label + "_output", gathered, [reference] * len(source))
        record["exact"] = all(row["exact"] for row in rows)
        self.gather_records.append(record)
        return gathered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--collective", choices=["native", "async"], default="native")
    parser.add_argument("--residual", choices=["replicated", "sharded"], default="replicated")
    parser.add_argument("--gather-workers", type=int, choices=[1, 2, 4])
    parser.add_argument("--gather-core-grid", action="store_true", help="Restrict AG workers to the semaphore grid")
    parser.add_argument(
        "--semaphore-full-grid", action="store_true", help="Allocate semaphores on the full device grid"
    )
    args = parser.parse_args()
    torch.set_num_threads(8)
    state_dict, config = H.layer_state_dict(args.layer, "real"), H.hf_config()
    recorded = recorded_activations(args.layer)
    if not 1 < args.length < min(recorded.shape[1], 4096):
        parser.error("length must leave one recorded token and fit the 4096-token probe capacity")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, router_config=fabric_router_config())
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 * 1024 * 1024, l1_small_size=24576)
    trace = None
    try:
        plan = MeshConfig(residual=args.residual, collective=args.collective)
        decoder = InspectGather.from_state_dict(
            state_dict, hf_config=config, layer_idx=args.layer, mesh_device=mesh, max_context=4096, mesh_config=plan
        )
        decoder.gather_workers, decoder.gather_core_grid = args.gather_workers, args.gather_core_grid
        if args.semaphore_full_grid:
            grid = mesh.compute_with_storage_grid_size()
            decoder.ccl.ccl_cores = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
            )
            decoder.ccl._init_semaphores()
        print(
            json.dumps(
                dict(
                    name="contract",
                    mesh_config=asdict(plan),
                    boundary_reads_synchronize=True,
                    gather_workers=args.gather_workers,
                    gather_core_grid=args.gather_core_grid,
                    semaphore_full_grid=args.semaphore_full_grid,
                    device_grid=str(mesh.compute_with_storage_grid_size()),
                    semaphore_grid=str(decoder.ccl.ccl_cores),
                )
            ),
            flush=True,
        )
        decoder.gather_records, decoder.phase = [], "prefill"
        mapper = (
            ttnn.ShardTensorToMesh(mesh, dim=-1) if args.residual == "sharded" else ttnn.ReplicateTensorToMesh(mesh)
        )

        def upload(value):
            return ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)

        prompt = upload(recorded[:, : args.length])
        token = upload(recorded[:, args.length : args.length + 1])
        blocks = num_blocks_for_context(4096)
        table = H.to_device(
            mesh,
            torch.arange(blocks, dtype=torch.int32).reshape(1, -1),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        positions, rotations = H.decode_inputs(mesh, torch.tensor([args.length]))
        decoder.allocate_state(1)
        decoder.allocate_kv_cache(blocks)
        decoder.reset_state()
        pref = decoder.prefill_forward(prompt, page_table=table)
        read_report("prefill_output", pref)
        ttnn.deallocate(pref)
        buffers = (
            [decoder.k_cache, decoder.v_cache]
            if decoder.is_full_attention
            else [decoder.recurrent_state, *decoder.conv_state]
        )
        snapshots = [ttnn.clone(value) for value in buffers]

        def restore():
            for source, target in zip(snapshots, buffers):
                ttnn.copy(source, target)

        first_output = None
        records = None
        eager_exact = None
        gather_checks = []
        for iteration in range(2):
            restore()
            decoder.gather_records, decoder.phase = [], f"eager{iteration + 1}"
            output = decoder.decode_forward(token, page_table=table, current_pos=positions, rot_idxs=rotations)
            values, rows = read_report(decoder.phase + "_layer_output", output, first_output)
            ttnn.deallocate(output)
            gather_checks.extend(row["exact"] for row in decoder.gather_records)
            if first_output is None:
                first_output, records = values, decoder.gather_records
            else:
                eager_exact = all(row["exact"] for row in rows)
        print(
            json.dumps(dict(name="layer_eager_summary", repeat_exact=eager_exact, gathers_exact=gather_checks)),
            flush=True,
        )

        # Recreate precisely the observed local logical shapes/dtypes. Persistent
        # inputs are allocated before capture; the trace has no host reads/writes.
        sources, references = [], []
        for index, record in enumerate(records):
            joined = torch.cat(record["source"], dim=-1)
            source = ttnn.from_torch(
                joined,
                device=mesh,
                dtype=record["dtype"],
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
            )
            sources.append(source)
            references.append([joined] * 4)
            print(
                json.dumps(dict(name=f"isolated_source{index}", original=record["metadata"], actual=metadata(source))),
                flush=True,
            )

        def sequence():
            # Call the production helper directly to bypass synchronizing reads.
            return [decoder._gather_unobserved(source) for source in sources]

        isolated_checks = []
        for iteration in range(2):
            outputs = sequence()
            for index, (output, reference) in enumerate(zip(outputs, references)):
                _, rows = read_report(f"isolated_eager{iteration + 1}_ag{index}", output, reference)
                isolated_checks.extend(row["exact"] for row in rows)
                ttnn.deallocate(output)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        outputs = sequence()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        for iteration in range(3):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            for index, (output, reference) in enumerate(zip(outputs, references)):
                _, rows = read_report(f"isolated_trace{iteration + 1}_ag{index}", output, reference)
                isolated_checks.extend(row["exact"] for row in rows)
        ttnn.release_trace(mesh, trace)
        trace = None
        for output in outputs:
            ttnn.deallocate(output)
        print(
            json.dumps(
                dict(
                    name="diagnostic_complete",
                    layer_eager_exact=eager_exact,
                    layer_gathers_exact=all(gather_checks),
                    isolated_gathers_exact=all(isolated_checks),
                )
            ),
            flush=True,
        )
        assert all(gather_checks), "synchronized eager model gather differs from host concat"
        assert eager_exact, "synchronized restored eager layer differs before trace"
        assert all(isolated_checks), "isolated gather sequence differs from host concat"
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
