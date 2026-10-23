# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Regression for absent outward fabric connections in two-device all-gather."""

import pytest
import torch

import ttnn


@pytest.mark.parametrize("rows,persistent", [(32, False), (1, True)], ids=["allocated", "persistent"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 0}],
    indirect=True,
)
def test_all_gather_async_non_mux_endpoints(bh_1d_mesh_device, rows, persistent):
    if bh_1d_mesh_device.get_num_devices() < 2:
        pytest.skip("All-gather endpoint regression requires two Blackhole devices")
    # The existing fixture closes its submeshes and parent device after the test.
    mesh = bh_1d_mesh_device.create_submesh(ttnn.MeshShape((2, 1)))
    grid = mesh.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    semaphores = [[ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2)] for _ in range(2)]
    barriers = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2)]
    kept_output = None
    try:
        for iteration in range(3):
            expected = (torch.arange(rows * 64).reshape(1, 1, rows, 64) + 17 * iteration) % 251 - 125
            expected = expected.to(torch.bfloat16)
            source = ttnn.from_torch(
                expected,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
            )
            output = None
            try:
                assert list(source.shape) == [1, 1, rows, 32]
                assert list(source.padded_shape) == [1, 1, 32, 32]
                output = ttnn.experimental.all_gather_async(
                    source,
                    dim=3,
                    persistent_output_buffer=kept_output,
                    multi_device_global_semaphore=semaphores[iteration % 2],
                    barrier_semaphore=barriers[iteration % 2],
                    num_links=1,
                    num_workers_per_link=1,
                    topology=ttnn.Topology.Ring,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                # One worker forces the non-mux writer. Ring on two devices
                # lowers to Linear, leaving one absent outward connection per rank.
                if persistent:
                    if kept_output is None:
                        kept_output = output
                    else:
                        assert output.buffer_address() == kept_output.buffer_address()
                devices = ttnn.get_device_tensors(output)
                assert len(devices) == 2
                for rank, device_output in enumerate(devices):
                    assert torch.equal(ttnn.to_torch(device_output), expected), (iteration, rank)
            finally:
                ttnn.deallocate(source)
                if output is not None and (
                    kept_output is None or output.buffer_address() != kept_output.buffer_address()
                ):
                    ttnn.deallocate(output)
    finally:
        if kept_output is not None:
            ttnn.deallocate(kept_output)
