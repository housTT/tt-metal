# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in, decoder-shape CCL probes retained as optimization evidence."""

from __future__ import annotations

import os
import time

import pytest
import torch
import ttnn

from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import _decode_l1_memory


MESH_DEVICE = [4]
DEVICE_PARAMS = [
    {
        "trace_region_size": 30_000_000,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    }
]


def _core_range_set(mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
    )


def _load_full_worker_subdevice(mesh_device):
    cores = _core_range_set(mesh_device)
    worker = ttnn.SubDevice([cores])
    worker_id = ttnn.SubDeviceId(0)
    manager = mesh_device.create_sub_device_manager([worker], 0)
    mesh_device.load_sub_device_manager(manager)
    mesh_device.set_sub_device_stall_group([worker_id])
    return cores, worker_id


def _trace_us(mesh_device, op, replays=100):
    op()
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = op()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    for _ in range(10):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    start = time.perf_counter()
    for _ in range(replays):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    elapsed_us = (time.perf_counter() - start) * 1e6 / replays
    ttnn.release_trace(mesh_device, trace_id)
    return output, elapsed_us


def _pcc(reference, actual):
    reference = reference.float().flatten()
    actual = actual.float().flatten()
    return torch.corrcoef(torch.stack((reference, actual)))[0, 1].item()


@pytest.mark.parametrize("mesh_device", MESH_DEVICE, indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("cores", [8, 16])
@pytest.mark.parametrize("payload", ["bf16", "bfp8"])
def test_decoder_shape_persistent_all_reduce(mesh_device, cores, payload):
    if os.environ.get("QWEN36_RUN_PERSISTENT_CCL_PROBE") != "1":
        pytest.skip("set QWEN36_RUN_PERSISTENT_CCL_PROBE=1")

    torch.manual_seed(0)
    ccl_cores, worker_id = _load_full_worker_subdevice(mesh_device)
    output_memory = _decode_l1_memory(32, 5120, cores)
    buffer_memory = _decode_l1_memory(32, 5120 * 4, cores)
    tt_dtype = ttnn.bfloat16 if payload == "bf16" else ttnn.bfloat8_b
    host = torch.randn((4, 1, 32, 5120), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=tt_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=output_memory,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(1, 4), dims=(None, 0)),
    )
    buffer_tensor = ttnn.from_torch(
        torch.zeros((1, 1, 32, 5120 * 4), dtype=torch.bfloat16),
        device=mesh_device,
        dtype=tt_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=buffer_memory,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    semaphore = ttnn.create_global_semaphore(mesh_device, ccl_cores, 0)

    def standard():
        return ttnn.all_reduce(
            input_tensor,
            num_links=2,
            topology=ttnn.Topology.Ring,
            memory_config=output_memory,
            subdevice_id=worker_id,
        )

    def persistent():
        return ttnn.experimental.all_reduce_async(
            input_tensor,
            buffer_tensor,
            cluster_axis=1,
            mesh_device=mesh_device,
            multi_device_global_semaphore=semaphore,
            dtype=tt_dtype,
            memory_config=output_memory,
            topology=ttnn.Topology.Ring,
            num_links=2,
            subdevice_id=worker_id,
        )

    standard_output, standard_us = _trace_us(mesh_device, standard)
    persistent_output, persistent_us = _trace_us(mesh_device, persistent)
    golden = host.sum(dim=0, keepdim=True)
    for name, output in (("standard", standard_output), ("persistent", persistent_output)):
        pccs = []
        for local in ttnn.get_device_tensors(output):
            pccs.append(_pcc(golden, ttnn.to_torch(local)))
        print(
            f"PERSISTENT_AR_CORRECTNESS payload={payload} cores={cores} "
            f"path={name} pccs={pccs} output_dtype={output.dtype} "
            f"buffer_dtype={buffer_tensor.dtype}"
        )
        assert min(pccs) >= 0.999
    print(
        f"PERSISTENT_AR_PERF payload={payload} cores={cores} standard_us={standard_us:.3f} "
        f"persistent_us={persistent_us:.3f} speedup={standard_us / persistent_us:.3f}"
    )
    mesh_device.reset_sub_device_stall_group()


@pytest.mark.parametrize("mesh_device", MESH_DEVICE, indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_decoder_shape_matmul_reduce_scatter(mesh_device):
    """Attention output projection: local [32,1536] @ [1536,5120]."""
    if os.environ.get("QWEN36_RUN_FUSED_CCL_PROBE") != "1":
        pytest.skip("set QWEN36_RUN_FUSED_CCL_PROBE=1")
    from tests.ttnn.unit_tests.operations.ccl.test_new_matmul_reduce_scatter import (
        run_reduce_scatter_impl,
    )

    dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    start = time.perf_counter()
    run_reduce_scatter_impl(
        mesh_device,
        4,
        [1, 1, 32, 5120],
        2,
        3,
        2,
        [1, 1, 1536 * 4, 5120],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.bfloat8_b,
        6,
        False,
        dram,
        dram,
        dram,
        ttnn.Topology.Ring,
        use_non_fused=False,
        num_iters=1,
        enable_trace=False,
    )
    print(
        "FUSED_RS_EXACT path=fused "
        f"wall_us={(time.perf_counter() - start) * 1e6:.3f}"
    )


@pytest.mark.parametrize("mesh_device", MESH_DEVICE, indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("local_n", [3584, 4352, 4608])
def test_decoder_shape_all_gather_matmul(mesh_device, local_n):
    """Fractured [32,1280] gather feeding one local QKV/MLP column matmul."""
    if os.environ.get("QWEN36_RUN_FUSED_CCL_PROBE") != "1":
        pytest.skip("set QWEN36_RUN_FUSED_CCL_PROBE=1")
    from models.tt_dit.tests.models.wan2_2.test_all_gather_minimal_matmul_async import (
        run_test_linear,
    )

    start = time.perf_counter()
    results = run_test_linear(
        mesh_device,
        32,
        5120,
        local_n,
        1,
        8,
        4,
        1,
        2,
        ttnn.Topology.Ring,
        core_grid=ttnn.CoreCoord(8, 8),
        num_workers_per_link=4,
        num_links=2,
        use_bias=False,
        use_non_fused=False,
        force_transpose=True,
        sp_axis=0,
        tp_axis=1,
        num_iters=1,
        enable_trace=False,
        cluster_axis=1,
        dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat8_b,
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_acc=False,
    )
    assert min(item["pcc"] for chunk in results[0] for item in chunk) > 0.995
    print(
        f"FUSED_AGMM_EXACT local_n={local_n} "
        "path=fused "
        f"wall_us={(time.perf_counter() - start) * 1e6:.3f}"
    )
