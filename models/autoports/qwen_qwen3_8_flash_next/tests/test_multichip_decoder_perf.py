# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Warmed fixed-P300 multichip prefill and traced-decode windows."""

from __future__ import annotations

import hashlib
import math
import os
import statistics
import time

import pytest
import torch
from tracy import signpost

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import harness as H
from models.autoports.qwen_qwen3_8_flash_next.tests.test_optimized_decoder_perf import _real_activations
from models.autoports.qwen_qwen3_8_flash_next.tt import functional_decoder as _functional_decoder
from models.autoports.qwen_qwen3_8_flash_next.tt.host_weight_cache import Qwen38PLEHostStore, SafetensorCheckpoint
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import (
    COLLECTIVE_NUM_LINKS as DEFAULT_COLLECTIVE_NUM_LINKS,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import (
    FABRIC_PACKET_BYTES as DEFAULT_FABRIC_PACKET_BYTES,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import (
    HostBackedSegmentedDecodeTrace,
    MultichipDecoder,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedDecoder

LAYER_KINDS = tuple(int(layer) for layer in os.environ.get("QWEN38_MC_PERF_LAYERS", "0,1,3").split(","))
DECODE_REPLAYS = int(os.environ.get("QWEN38_MC_PERF_DECODE_REPLAYS", "10"))
PROFILE_DIRECT_DECODE = os.environ.get("QWEN38_MC_PROFILE_DIRECT_DECODE", "0") == "1"
PROFILE_HOST_ONLY = os.environ.get("QWEN38_MC_PROFILE_HOST_ONLY", "0") == "1"
PREFILL_SEQ_LEN = 128
HOST_PREFILL_SEQ_LEN = int(os.environ.get("QWEN38_MC_HOST_PREFILL_SEQ_LEN", "33"))
RESIDUAL_TOPOLOGY_REPLAYS = int(os.environ.get("QWEN38_MC_RESIDUAL_TOPOLOGY_REPLAYS", "100"))
HOST_EXPERT_SLOTS = int(os.environ.get("QWEN38_MC_HOST_EXPERT_SLOTS", "10"))
HOST_PACKED_EXPERTS = int(os.environ.get("QWEN38_MC_HOST_PACKED_EXPERTS", "512"))
COLLECTIVE_NUM_LINKS = int(os.environ.get("QWEN38_MC_COLLECTIVE_NUM_LINKS", str(DEFAULT_COLLECTIVE_NUM_LINKS)))
FABRIC_PACKET_BYTES = int(os.environ.get("QWEN38_MC_FABRIC_PACKET_BYTES", str(DEFAULT_FABRIC_PACKET_BYTES)))


def _candidate_layer_kwargs():
    candidates = {
        "QWEN38_MC_DECODE_1D_CONFIG": "decode_1d_config",
        "QWEN38_MC_PREFILL_CONFIG": "prefill_config",
        "QWEN38_MC_DRAM_SHARDED_ROLE": "dram_sharded_role",
        "QWEN38_MC_OPTIMIZATION_POLICY": "optimization_policy",
        "QWEN38_MC_SHARED_PROJECTION_POLICY": "shared_projection_policy",
        "QWEN38_MC_GDN_PROJECTION_POLICY": "gdn_projection_policy",
        "QWEN38_MC_QSA_INPUT_POLICY": "qsa_input_policy",
        "QWEN38_MC_ATTENTION_OUTPUT_POLICY": "attention_output_policy",
        "QWEN38_MC_ROW_PARALLEL_DTYPE": "row_parallel_dtype",
    }
    return {argument: os.environ[name] for name, argument in candidates.items() if name in os.environ}


def _fabric_router(max_packet_payload_size_bytes):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = int(max_packet_payload_size_bytes)
    return config


PERF_DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D,
    "trace_region_size": 100_000_000,
}
if FABRIC_PACKET_BYTES != 4352:
    PERF_DEVICE_PARAMS["fabric_router_config"] = _fabric_router(FABRIC_PACKET_BYTES)


def _upload(tensor, mesh_device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=dtype,
        layout=layout,
    )


def _fractured_upload(tensor, mesh_device):
    grouped = tensor.reshape(1, 1, tensor.shape[-2] * 4, 2560)
    return ttnn.from_torch(
        grouped,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


def _fractured_host(tensor):
    shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]
    grouped = torch.cat(shards, dim=-1)
    return grouped.reshape(1, 1, grouped.shape[-2] // 4, 10240)


def _sha256_tensors(*tensors: torch.Tensor) -> str:
    """Hash exactly the logical BF16 inputs consumed by one measurement."""

    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _host_service_snapshot(layer) -> dict[str, int | float]:
    """Snapshot every exact host boundary counter with stable prefixes."""

    snapshot = {f"expert_{name}": value for name, value in layer.host_expert_cache.metrics().items()}
    snapshot.update({f"source_{name}": value for name, value in layer.host_expert_source.metrics().items()})
    if layer.host_ple_store is not None:
        snapshot.update({f"ple_{name}": value for name, value in layer.host_ple_store.metrics().items()})
    if layer.ple_staging is not None:
        snapshot.update({f"ple_stage_{name}": value for name, value in layer.ple_staging.metrics().items()})
    return snapshot


def _host_service_delta(before, after) -> dict[str, int | float]:
    return {name: after[name] - before.get(name, 0) for name in after if isinstance(after[name], (int, float))}


def _rate_gbps(byte_count: int | float, seconds: int | float) -> float:
    return float(byte_count) / float(seconds) / 1.0e9 if seconds else 0.0


def _print_host_service_evidence(*, layer_idx, phase, delta, end_to_end_seconds, service_seconds):
    """Emit auditable expert/PLE lookup and transfer accounting for one window."""

    source_pack = float(delta.get("expert_source_pack_seconds", 0.0))
    expert_h2d = float(delta.get("expert_h2d_seconds", 0.0))
    ple_lookup = float(delta.get("ple_lookup_seconds", 0.0))
    ple_h2d = float(delta.get("ple_stage_h2d_seconds", 0.0))
    # The current boundary is deliberately serialized.  Anything left in the
    # measured service window is route-id D2H, directory work, synchronization,
    # or host scheduling and is reported as unattributed stall.
    overlap = 0.0
    accounted = source_pack + expert_h2d + ple_lookup + ple_h2d
    stall = max(0.0, float(service_seconds) - accounted)
    print(
        "MC_HOST_SERVICE_EVIDENCE "
        f"mesh=1x2 layer={layer_idx} phase={phase} mode=exact-host-ep2 "
        f"end_to_end_ms={end_to_end_seconds * 1000.0:.6f} "
        f"service_window_ms={service_seconds * 1000.0:.6f} "
        f"expert_requests={int(delta.get('expert_requests', 0))} "
        f"expert_waves={int(delta.get('expert_waves', 0))} "
        f"expert_hits={int(delta.get('expert_hits', 0))} "
        f"expert_misses={int(delta.get('expert_misses', 0))} "
        f"expert_evictions={int(delta.get('expert_evictions', 0))} "
        f"packed_host_hits={int(delta.get('expert_packed_host_hits', 0))} "
        f"packed_host_misses={int(delta.get('expert_packed_host_misses', 0))} "
        f"checkpoint_reads={int(delta.get('source_checkpoint_expert_reads', 0))} "
        f"checkpoint_bytes={int(delta.get('source_checkpoint_bytes', 0))} "
        f"checkpoint_read_ms={float(delta.get('source_checkpoint_read_seconds', 0.0)) * 1000.0:.6f} "
        f"checkpoint_gbps={_rate_gbps(delta.get('source_checkpoint_bytes', 0), delta.get('source_checkpoint_read_seconds', 0.0)):.6f} "
        f"source_pack_ms={source_pack * 1000.0:.6f} "
        f"expert_h2d_bytes={int(delta.get('expert_h2d_bytes', 0))} "
        f"expert_h2d_ms={expert_h2d * 1000.0:.6f} "
        f"expert_h2d_gbps={_rate_gbps(delta.get('expert_h2d_bytes', 0), expert_h2d):.6f} "
        f"index_h2d_bytes={int(delta.get('expert_index_h2d_bytes', 0))} "
        f"index_upload_ms={float(delta.get('expert_index_upload_seconds', 0.0)) * 1000.0:.6f} "
        f"ple_selected_rows={int(delta.get('ple_selected_rows', 0))} "
        f"ple_unique_rows={int(delta.get('ple_unique_rows', 0))} "
        f"ple_table_rows_read={int(delta.get('ple_table_rows_read', 0))} "
        f"ple_table_bytes={int(delta.get('ple_table_bytes_read', 0))} "
        f"ple_lookup_ms={ple_lookup * 1000.0:.6f} "
        f"ple_stage_logical_h2d_bytes={int(delta.get('ple_stage_logical_h2d_bytes', 0))} "
        f"ple_stage_physical_h2d_bytes={int(delta.get('ple_stage_h2d_bytes', 0))} "
        f"ple_h2d_ms={ple_h2d * 1000.0:.6f} "
        f"ple_h2d_gbps={_rate_gbps(delta.get('ple_stage_h2d_bytes', 0), ple_h2d):.6f} "
        f"overlap_ms={overlap * 1000.0:.6f} stall_ms={stall * 1000.0:.6f}"
    )


def _capture_gdn_prefill_pipeline(layer):
    """Clone the narrow boundaries needed to localize the layer-0 TP miss."""

    captures = {name: [] for name in ("hyper_mix", "gdn", "hyper_inject", "routing", "routed", "moe")}

    def wrap_tuple(name, method_name):
        original = getattr(layer, method_name)

        def wrapped(*args, **kwargs):
            output = original(*args, **kwargs)
            captures[name].append(tuple(ttnn.clone(value, memory_config=ttnn.DRAM_MEMORY_CONFIG) for value in output))
            return output

        setattr(layer, method_name, wrapped)

    def wrap_tensor(name, method_name, *, capture_inputs=False):
        original = getattr(layer, method_name)

        def wrapped(*args, **kwargs):
            inputs = ()
            if capture_inputs:
                inputs = tuple(
                    ttnn.clone(value, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    for value in args
                    if isinstance(value, ttnn.Tensor)
                )
            output = original(*args, **kwargs)
            captures[name].append(
                (inputs, ttnn.clone(output, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                if capture_inputs
                else ttnn.clone(output, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            )
            return output

        setattr(layer, method_name, wrapped)

    wrap_tuple("hyper_mix", "_hyper_mix")
    wrap_tensor("gdn", "_gdn_prefill", capture_inputs=True)
    wrap_tensor("hyper_inject", "_hyper_inject", capture_inputs=True)
    wrap_tensor("routing", "_routing_from_logits", capture_inputs=True)
    wrap_tensor("routed", "_routed_experts", capture_inputs=True)
    wrap_tensor("moe", "_moe", capture_inputs=True)
    return captures


def _capture_host(tensor, *, reduce_partials=False):
    shards = [ttnn.to_torch(shard).float() for shard in ttnn.get_device_tensors(tensor)]
    return sum(shards[1:], shards[0]) if reduce_partials else shards[0]


def _print_gdn_prefill_pipeline_pcc(lhs, rhs):
    """Report the first drifting boundary without changing the asserted gate."""

    def report(label, left, right, *, reduce_partials=False):
        value = H.pcc(_capture_host(left), _capture_host(right, reduce_partials=reduce_partials))
        print(f"MC_GDN_PREFILL_BOUNDARY boundary={label} pcc={value:.8f}")

    for index, (left, right) in enumerate(zip(lhs["hyper_mix"], rhs["hyper_mix"])):
        for field, left_value, right_value in zip(("mixed", "hyper", "injection"), left, right):
            report(f"hyper_mix{index}.{field}", left_value, right_value)
    for family in ("gdn", "hyper_inject", "routing", "moe"):
        for index, (left, right) in enumerate(zip(lhs[family], rhs[family])):
            left_inputs, left_output = left
            right_inputs, right_output = right
            for input_index, (left_input, right_input) in enumerate(zip(left_inputs, right_inputs)):
                report(f"{family}{index}.input{input_index}", left_input, right_input)
            report(f"{family}{index}.output", left_output, right_output)
    for index, (left, right) in enumerate(zip(lhs["routed"], rhs["routed"])):
        left_inputs, left_output = left
        right_inputs, right_output = right
        for input_index, (left_input, right_input) in enumerate(zip(left_inputs, right_inputs)):
            report(f"routed{index}.input{input_index}", left_input, right_input)
        report(f"routed{index}.output_partial_sum", left_output, right_output, reduce_partials=True)


def _print_gdn_expert_weight_pcc(lhs, rhs):
    """Prove whether setup preserved the represented BFP4 expert shards."""

    baseline_gate_up = ttnn.to_torch(ttnn.get_device_tensors(lhs.expert_gate_up)[0]).float()
    local_gate_up = [ttnn.to_torch(shard).float() for shard in ttnn.get_device_tensors(rhs.expert_gate_up)]
    local_width = local_gate_up[0].shape[-1] // 2
    reconstructed_gate_up = torch.cat(
        (
            local_gate_up[0][..., :local_width],
            local_gate_up[1][..., :local_width],
            local_gate_up[0][..., local_width:],
            local_gate_up[1][..., local_width:],
        ),
        dim=-1,
    )
    baseline_down = ttnn.to_torch(ttnn.get_device_tensors(lhs.experts.down)[0]).float()
    local_down = [ttnn.to_torch(shard).float() for shard in ttnn.get_device_tensors(rhs.experts.down)]
    reconstructed_down = torch.cat(local_down, dim=-2)
    for name, baseline, reconstructed in (
        ("gate_up", baseline_gate_up, reconstructed_gate_up),
        ("down", baseline_down, reconstructed_down),
    ):
        print(
            "MC_GDN_EXPERT_WEIGHT "
            f"name={name} pcc={H.pcc(baseline, reconstructed):.8f} "
            f"max_abs={(baseline - reconstructed).abs().max().item():.8f} "
            f"exact={torch.equal(baseline, reconstructed)}"
        )


def _invoke_with_sparse_captures(invoke, *, down_dtype=None):
    """Capture the packed gate/up and down sparse-matmul outputs for one pass."""

    captures = []
    original = ttnn.sparse_matmul

    def wrapped(*args, **kwargs):
        if down_dtype is not None and len(captures) == 1:
            kwargs["dtype"] = down_dtype
        output = original(*args, **kwargs)
        captures.append(ttnn.clone(output, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        return output

    ttnn.sparse_matmul = wrapped
    try:
        return invoke(), captures
    finally:
        ttnn.sparse_matmul = original


def _print_gdn_sparse_stage_pcc(lhs, rhs):
    baseline_gate_up = ttnn.to_torch(ttnn.get_device_tensors(lhs[0])[0]).float()
    local_gate_up = [ttnn.to_torch(shard).float() for shard in ttnn.get_device_tensors(rhs[0])]
    local_width = local_gate_up[0].shape[-1] // 2
    reconstructed_gate_up = torch.cat(
        (
            local_gate_up[0][..., :local_width],
            local_gate_up[1][..., :local_width],
            local_gate_up[0][..., local_width:],
            local_gate_up[1][..., local_width:],
        ),
        dim=-1,
    )
    baseline_down = ttnn.to_torch(ttnn.get_device_tensors(lhs[1])[0]).float()
    local_down = [ttnn.to_torch(shard).float() for shard in ttnn.get_device_tensors(rhs[1])]
    reduced_down = sum(local_down[1:], local_down[0])
    for name, baseline, candidate in (
        ("gate_up", baseline_gate_up, reconstructed_gate_up),
        ("down_partial_sum", baseline_down, reduced_down),
    ):
        print(
            "MC_GDN_SPARSE_STAGE "
            f"name={name} pcc={H.pcc(baseline, candidate):.8f} "
            f"max_abs={(baseline - candidate).abs().max().item():.8f}"
        )


def _paged_inputs(layer, mesh_device, page_table_host, cos_host, sin_host, *, prefill_seq_len=PREFILL_SEQ_LEN):
    page_table = _upload(page_table_host, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    chunk_tables = []
    for start, _, padded in layer.prefill_chunk_plan(prefill_seq_len):
        first = start // layer.block_size
        last = (start + padded) // layer.block_size
        chunk_tables.append(
            _upload(
                page_table_host[:, first:last],
                mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        )
    cos = _upload(cos_host.reshape(1, 1, layer.max_seq_len, -1), mesh_device)
    sin = _upload(sin_host.reshape(1, 1, layer.max_seq_len, -1), mesh_device)
    return page_table, chunk_tables, (cos, sin)


def _observe_routing(layer, invoke):
    observation = {}
    original = layer._routed_experts

    def observe(x, routing):
        observation["routing"] = ttnn.clone(routing, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return original(x, routing)

    layer._routed_experts = observe
    try:
        output = invoke()
        ttnn.synchronize_device(layer.mesh_device)
        ttnn.deallocate(output)
    finally:
        del layer._routed_experts
    routing = observation.pop("routing")
    host = ttnn.to_torch(ttnn.get_device_tensors(routing)[0])
    ttnn.deallocate(routing)
    selected = host.ne(0).reshape(-1, host.shape[-1])
    if int(selected[0].sum()) != layer.shapes.num_experts_per_tok:
        raise AssertionError("decode/prefill routing did not preserve gate-selected top-k execution")
    tile_unions = selected.reshape(-1, 32, selected.shape[-1]).any(dim=1).sum(dim=1)
    return {
        "union": int(selected.any(dim=0).sum()),
        "tile_unions": tuple(int(value) for value in tile_unions),
        "hash": hashlib.sha256(selected.to(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def _profile_checkpoint(mesh_device):
    """Flush setup/warm-up markers before the next profiled phase."""

    if PROFILE_DIRECT_DECODE:
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)


def _hyper_mix_tail_from_normed(layer, normed, hyper_input, prefix):
    """Run the real fused hyper mixer after a precomputed grouped RMSNorm."""

    s = layer.shapes
    packed = layer._linear(normed, layer.w[f"{prefix}_down_inject"])
    low = layer._slice_last(packed, 0, s.hc_lowrank)
    injection = layer._slice_last(packed, s.hc_lowrank, s.hc_lowrank + s.hc_count)
    ttnn.deallocate(packed)
    low = ttnn.silu(low)
    mix = layer._linear(low, layer.w[f"{prefix}_up"])
    ttnn.deallocate(low)

    rows = math.prod(_functional_decoder._shape(normed)[:-1])
    norm_groups = ttnn.reshape(normed, (rows, s.hc_count, s.hidden_size))
    mix_groups = ttnn.reshape(mix, (rows, s.hc_count, s.hidden_size))
    mixed = ttnn.multiply(
        norm_groups,
        mix_groups,
        input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
    )
    _functional_decoder._free(norm_groups, normed, mixed)
    _functional_decoder._free(mix_groups, mix, mixed)
    ttnn.deallocate(mix)
    mixed = ttnn.mean(mixed, dim=1, keepdim=True)
    mixed = ttnn.reshape(mixed, (*_functional_decoder._shape(hyper_input)[:-1], s.hidden_size))
    ttnn.deallocate(normed)
    return mixed, injection


def _replicated_residual_consumer(layer, local_partial, hyper_input, attention_injection):
    """Current all-reduce boundary followed by the real next hyper mixer."""

    s = layer.shapes
    reduced = ttnn.all_reduce(
        local_partial,
        cluster_axis=layer.collective_axis,
        num_links=layer.collective_num_links,
        topology=layer.collective_topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if reduced.dtype == ttnn.float32:
        bf16 = ttnn.typecast(reduced, ttnn.bfloat16)
        ttnn.deallocate(reduced)
        reduced = bf16
    value = ttnn.reshape(reduced, (1, 1, 1, s.hidden_size))
    gate = ttnn.reshape(attention_injection, (1, 1, s.hc_count, 1))
    projected = ttnn.multiply(value, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    _functional_decoder._free(value, reduced, projected)
    _functional_decoder._free(gate, attention_injection, projected)
    ttnn.deallocate(reduced)
    hyper_groups = ttnn.reshape(hyper_input, (1, 1, s.hc_count, s.hidden_size))
    updated_groups = ttnn.mac(projected, 2.0, hyper_groups)
    ttnn.deallocate(projected)
    updated = ttnn.reshape(updated_groups, (1, 1, 1, s.hc_hidden_size))
    _functional_decoder._free(updated_groups, updated)
    normed = layer._rms_norm(
        updated,
        layer.w["mlp_hc_norm"],
        s.rms_norm_eps,
        group_count=s.hc_count,
    )
    mixed, injection = _hyper_mix_tail_from_normed(layer, normed, updated, "mlp_hc")
    router = layer._linear(mixed, layer.w["moe_input"])
    return mixed, injection, updated, router


def _fully_fractured_residual_consumer(
    layer,
    local_partial,
    local_hyper_groups,
    local_norm_weight,
    local_down_inject,
    local_up,
    attention_injection,
):
    """Keep the residual and both hyper projections fractured through router input."""

    s = layer.shapes
    reduced = ttnn.reduce_scatter(
        local_partial,
        dim=3,
        cluster_axis=layer.collective_axis,
        num_links=layer.collective_num_links,
        topology=layer.collective_topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if reduced.dtype == ttnn.float32:
        bf16 = ttnn.typecast(reduced, ttnn.bfloat16)
        ttnn.deallocate(reduced)
        reduced = bf16
    gate = ttnn.reshape(attention_injection, (1, 1, s.hc_count, 1))
    projected = ttnn.multiply(reduced, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    _functional_decoder._free(gate, attention_injection, projected)
    ttnn.deallocate(reduced)
    local_updated = ttnn.mac(projected, 2.0, local_hyper_groups)
    ttnn.deallocate(projected)

    stats = ttnn.rms_norm_pre_all_gather(
        local_updated,
        compute_kernel_config=_functional_decoder._hifi4(fp32=True),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gathered_stats = ttnn.all_gather(
        stats,
        dim=3,
        cluster_axis=layer.collective_axis,
        num_links=layer.collective_num_links,
        topology=layer.collective_topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(stats)
    local_normed = ttnn.rms_norm_post_all_gather(
        local_updated,
        gathered_stats,
        epsilon=s.rms_norm_eps,
        compute_kernel_config=_functional_decoder._hifi4(fp32=True),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(gathered_stats)
    weighted_local_normed = ttnn.multiply(local_normed, local_norm_weight)
    ttnn.deallocate(local_normed)
    local_flat = ttnn.reshape(weighted_local_normed, (1, 1, 1, s.hc_hidden_size // layer.tp_size))
    packed_partial = layer._linear_impl(local_flat, local_down_inject, dtype=ttnn.bfloat16)
    packed = ttnn.all_reduce(
        packed_partial,
        cluster_axis=layer.collective_axis,
        num_links=layer.collective_num_links,
        topology=layer.collective_topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(packed_partial)
    low = layer._slice_last(packed, 0, s.hc_lowrank)
    injection = layer._slice_last(packed, s.hc_lowrank, s.hc_lowrank + s.hc_count)
    ttnn.deallocate(packed)
    low = ttnn.silu(low)
    local_mix = layer._linear_impl(low, local_up, dtype=ttnn.bfloat16)
    ttnn.deallocate(low)
    local_mix_groups = ttnn.reshape(local_mix, (1, 1, s.hc_count, s.hidden_size // layer.tp_size))
    local_mixed = ttnn.multiply(
        weighted_local_normed,
        local_mix_groups,
        input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
    )
    _functional_decoder._free(local_mix_groups, local_mix, local_mixed)
    ttnn.deallocate(local_mix)
    ttnn.deallocate(weighted_local_normed)
    local_mixed = ttnn.mean(local_mixed, dim=2, keepdim=True)
    gathered_mixed = ttnn.all_gather(
        local_mixed,
        dim=3,
        cluster_axis=layer.collective_axis,
        num_links=layer.collective_num_links,
        topology=layer.collective_topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(local_mixed)
    mixed = ttnn.reshape(gathered_mixed, (1, 1, 1, s.hidden_size))
    _functional_decoder._free(gathered_mixed, mixed)
    router = layer._linear(mixed, layer.w["moe_input"])
    return mixed, injection, local_updated, router


def _warmed_trace_samples(mesh_device, invoke, *, replays):
    warm = invoke()
    ttnn.synchronize_device(mesh_device)
    for tensor in warm:
        if tensor.is_allocated():
            ttnn.deallocate(tensor)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    outputs = invoke()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    for _ in range(3):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    samples = []
    for _ in range(7):
        started = time.perf_counter()
        for _ in range(replays):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        samples.append((time.perf_counter() - started) * 1000.0 / replays)
    return trace_id, outputs, samples


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 64_000_000}],
    indirect=True,
)
def test_warmed_prefill_and_traced_decode(bh_1d_mesh_device, device_params, layer_idx):
    mode = os.environ.get("QWEN38_MC_PERF_MODE", "both")
    if mode not in {"both", "prefill", "decode"}:
        raise ValueError(f"unsupported QWEN38_MC_PERF_MODE={mode}")
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh_device = bh_1d_mesh_device
    config = H.target_config()
    max_seq_len = 4096 if layer_idx == 3 else 128
    state = H.load_real_layer_state(layer_idx)
    layer = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=max_seq_len,
    )
    del state
    _profile_checkpoint(mesh_device)
    hidden_host, ple_host, activation_hash = _real_activations(layer_idx)
    prefill_hidden = _fractured_upload(hidden_host[:, :, :PREFILL_SEQ_LEN], mesh_device)
    prefill_kwargs = {}
    if layer_idx == 1:
        prefill_kwargs["ple_embeddings"] = _upload(ple_host[:, :, :PREFILL_SEQ_LEN], mesh_device)
    if layer_idx == 3:
        cos, sin = H.rope_tables(max_seq_len)
        page_host = H.shuffled_page_table(max_seq_len)
        page, chunk_pages, rot = _paged_inputs(layer, mesh_device, page_host, cos, sin)
        prefill_kwargs.update(page_table=page, page_tables_per_chunk=chunk_pages, rot_mats=rot)

    routing = _observe_routing(layer, lambda: layer.prefill_forward_fractured(prefill_hidden, **prefill_kwargs))
    _profile_checkpoint(mesh_device)
    prefill_ms = None
    if mode in {"both", "prefill"}:
        signpost(f"MC_PERF_PREFILL_L{layer_idx}")
        start = time.perf_counter()
        with H.ForbidHostFallback():
            prefill_output = layer.prefill_forward_fractured(prefill_hidden, **prefill_kwargs)
        ttnn.synchronize_device(mesh_device)
        prefill_ms = (time.perf_counter() - start) * 1000.0
        signpost(f"MC_PERF_PREFILL_L{layer_idx}_END")
        ttnn.deallocate(prefill_output)
        _profile_checkpoint(mesh_device)
    if mode == "prefill":
        print(
            f"MC_PERFEVIDENCE mesh=1x2 layer={layer_idx} weights=real-checkpoint "
            f"activation_sha256={activation_hash} active_topk=10 routing_union={routing['union']} "
            f"routing_tile_unions={routing['tile_unions']} routing_hash={routing['hash']} "
            f"prefill_seq={PREFILL_SEQ_LEN} warmed_prefill_ms={prefill_ms:.6f}"
        )
        return

    layer.prepare_decode_state()
    decode_hidden = _fractured_upload(hidden_host[:, :, PREFILL_SEQ_LEN:], mesh_device)
    current_pos = _upload(
        torch.tensor([127], dtype=torch.int32),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    decode_kwargs = {"current_pos": current_pos}
    if layer_idx == 1:
        decode_kwargs["ple_embeddings"] = _upload(ple_host[:, :, PREFILL_SEQ_LEN:], mesh_device)
    if layer_idx == 3:
        decode_kwargs.update(page_table=page, rot_mats=rot)
    decode_routing = _observe_routing(layer, lambda: layer.decode_forward_fractured(decode_hidden, **decode_kwargs))
    _profile_checkpoint(mesh_device)

    if PROFILE_DIRECT_DECODE:
        with H.ForbidHostFallback():
            decode_output = layer.decode_forward_fractured(decode_hidden, **decode_kwargs)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(decode_output)
        _profile_checkpoint(mesh_device)
        direct_outputs = []
        signpost(f"MC_PERF_DECODE_L{layer_idx}")
        start = time.perf_counter()
        with H.ForbidHostFallback():
            for _ in range(DECODE_REPLAYS):
                direct_outputs.append(layer.decode_forward_fractured(decode_hidden, **decode_kwargs))
        ttnn.synchronize_device(mesh_device)
        decode_ms = (time.perf_counter() - start) * 1000.0 / DECODE_REPLAYS
        signpost(f"MC_PERF_DECODE_L{layer_idx}_END")
        decode_output = direct_outputs[-1]
        for previous_output in direct_outputs[:-1]:
            ttnn.deallocate(previous_output)
        decode_mode = "direct-profile"
    else:
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        with H.ForbidHostFallback():
            decode_output = layer.decode_forward_fractured(decode_hidden, **decode_kwargs)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        signpost(f"MC_PERF_DECODE_L{layer_idx}")
        start = time.perf_counter()
        for _ in range(DECODE_REPLAYS):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        decode_ms = (time.perf_counter() - start) * 1000.0 / DECODE_REPLAYS
        signpost(f"MC_PERF_DECODE_L{layer_idx}_END")
        decode_mode = "trace-replay"
    assert list(decode_output.shape) == [1, 1, 4, 1280]
    assert list(_fractured_host(decode_output).shape) == [1, 1, 1, 10240]
    ttnn.deallocate(decode_output)
    if not PROFILE_DIRECT_DECODE:
        ttnn.release_trace(mesh_device, trace_id)
    prefill_field = "not-profiled" if prefill_ms is None else f"{prefill_ms:.6f}"
    print(
        f"MC_PERFEVIDENCE mesh=1x2 layer={layer_idx} weights=real-checkpoint "
        f"activation_sha256={activation_hash} active_topk=10 routing_union={routing['union']} "
        f"routing_tile_unions={routing['tile_unions']} routing_hash={routing['hash']} "
        f"decode_routing_union={decode_routing['union']} "
        f"decode_routing_tile_unions={decode_routing['tile_unions']} "
        f"decode_routing_hash={decode_routing['hash']} prefill_seq={PREFILL_SEQ_LEN} "
        f"warmed_prefill_ms={prefill_field} decode_mode={decode_mode} decode_ms={decode_ms:.6f} "
        f"decode_replays={DECODE_REPLAYS}"
    )


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.parametrize(
    "device_params",
    [PERF_DEVICE_PARAMS],
    indirect=True,
)
def test_host_backed_warmed_prefill_and_segmented_decode(bh_1d_mesh_device, device_params, layer_idx):
    """Compare an exact single-chip graph with the final host-backed TP2 path."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh_device = bh_1d_mesh_device
    config = H.target_config()
    max_seq_len = 4096 if layer_idx == 3 else 128
    state = H.load_real_layer_state(layer_idx)
    baseline = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=max_seq_len,
    )
    del state
    host_backed = MultichipDecoder.from_checkpoint_host_backed(
        H.MODEL_SNAPSHOT,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=max_seq_len,
        expert_cache_slots=HOST_EXPERT_SLOTS,
        packed_host_experts=HOST_PACKED_EXPERTS,
        collective_num_links=COLLECTIVE_NUM_LINKS,
        **_candidate_layer_kwargs(),
    )
    prefill_seq_len = HOST_PREFILL_SEQ_LEN
    hidden_host, _, _ = _real_activations(layer_idx)
    prefill_hidden = _upload(hidden_host[:, :, :prefill_seq_len], mesh_device)
    host_prefill_hidden = _fractured_upload(hidden_host[:, :, :prefill_seq_len], mesh_device)
    token_ids = torch.arange(2026, 2026 + prefill_seq_len + 1, dtype=torch.int64).unsqueeze(0)
    reference_store = None
    activation_parts = [hidden_host[:, :, : prefill_seq_len + 1]]
    baseline_prefill_kwargs = {}
    host_prefill_kwargs = {}
    baseline_decode_kwargs = {}
    host_capture_kwargs = {}
    if layer_idx == 1:
        reference_store = Qwen38PLEHostStore(SafetensorCheckpoint(H.MODEL_SNAPSHOT), row_cache_capacity=256)
        prefill_ple = reference_store.prepare(("perf-reference",), token_ids[:, :prefill_seq_len], reset=True)
        activation_parts.append(prefill_ple)
        baseline_prefill_kwargs["ple_embeddings"] = _upload(prefill_ple.unsqueeze(0), mesh_device)
        host_prefill_kwargs.update(input_ids=token_ids[:, :prefill_seq_len], request_id="perf-host")
    if layer_idx == 3:
        cos, sin = H.rope_tables(max_seq_len)
        page_host = H.shuffled_page_table(max_seq_len)
        page, chunk_pages, rot = _paged_inputs(
            baseline, mesh_device, page_host, cos, sin, prefill_seq_len=prefill_seq_len
        )
        baseline_prefill_kwargs.update(page_table=page, page_tables_per_chunk=chunk_pages, rot_mats=rot)
        host_prefill_kwargs.update(page_table=page, page_tables_per_chunk=chunk_pages, rot_mats=rot)
        baseline_decode_kwargs.update(page_table=page, rot_mats=rot)
        host_capture_kwargs.update(page_table=page, rot_mats=rot)
    prefill_activation_hash = _sha256_tensors(*activation_parts)

    # Compile and fill the bounded host cache before the measured windows.
    baseline_warm = baseline.prefill_forward(prefill_hidden, **baseline_prefill_kwargs)
    if layer_idx == 1:
        host_warm = host_backed.prefill_forward_host_backed_fractured(host_prefill_hidden, **host_prefill_kwargs)
    else:
        host_warm = host_backed.prefill_forward_fractured(host_prefill_hidden, **host_prefill_kwargs)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(baseline_warm)
    ttnn.deallocate(host_warm)
    _profile_checkpoint(mesh_device)

    signpost(f"MC_BASELINE_PREFILL_L{layer_idx}")
    started = time.perf_counter()
    baseline_prefill = baseline.prefill_forward(prefill_hidden, **baseline_prefill_kwargs)
    ttnn.synchronize_device(mesh_device)
    baseline_prefill_ms = (time.perf_counter() - started) * 1000.0
    signpost(f"MC_BASELINE_PREFILL_L{layer_idx}_END")
    _profile_checkpoint(mesh_device)
    signpost(f"MC_HOST_PREFILL_L{layer_idx}")
    prefill_service_before = _host_service_snapshot(host_backed)
    started = time.perf_counter()
    if layer_idx == 1:
        host_prefill = host_backed.prefill_forward_host_backed_fractured(host_prefill_hidden, **host_prefill_kwargs)
    else:
        host_prefill = host_backed.prefill_forward_fractured(host_prefill_hidden, **host_prefill_kwargs)
    ttnn.synchronize_device(mesh_device)
    host_prefill_ms = (time.perf_counter() - started) * 1000.0
    prefill_service_after = _host_service_snapshot(host_backed)
    signpost(f"MC_HOST_PREFILL_L{layer_idx}_END")
    prefill_service_delta = _host_service_delta(prefill_service_before, prefill_service_after)
    prefill_accounted_seconds = sum(
        float(prefill_service_delta.get(name, 0.0))
        for name in (
            "expert_source_pack_seconds",
            "expert_h2d_seconds",
            "ple_lookup_seconds",
            "ple_stage_h2d_seconds",
        )
    )
    _print_host_service_evidence(
        layer_idx=layer_idx,
        phase="prefill",
        delta=prefill_service_delta,
        end_to_end_seconds=host_prefill_ms / 1000.0,
        service_seconds=prefill_accounted_seconds,
    )
    prefill_pcc = H.pcc(
        ttnn.to_torch(ttnn.get_device_tensors(baseline_prefill)[0]),
        _fractured_host(host_prefill),
    )
    print(
        "MC_HOST_PREFILL_PCC "
        f"mesh=1x2 layer={layer_idx} logical_seq={prefill_seq_len} "
        f"logical_input_sha256={prefill_activation_hash} pcc={prefill_pcc:.8f}"
    )
    assert prefill_pcc >= H.PCC_BAR
    ttnn.deallocate(baseline_prefill)
    ttnn.deallocate(host_prefill)

    host_backed.prepare_decode_state()
    decode_hidden = _upload(hidden_host[:, :, prefill_seq_len : prefill_seq_len + 1], mesh_device)
    host_decode_hidden = _fractured_upload(hidden_host[:, :, prefill_seq_len : prefill_seq_len + 1], mesh_device)
    current_pos = _upload(
        torch.tensor([prefill_seq_len], dtype=torch.int32),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    baseline_decode_kwargs["current_pos"] = current_pos
    host_capture_kwargs["current_pos"] = current_pos
    if layer_idx == 1:
        decode_ids = token_ids[:, prefill_seq_len:]
        decode_ple = reference_store.prepare(("perf-reference",), decode_ids)
        activation_parts.append(decode_ple)
        baseline_decode_kwargs["ple_embeddings"] = _upload(decode_ple.unsqueeze(0), mesh_device)
        host_capture_kwargs.update(ple_input_ids=decode_ids, request_ids=("perf-host",))
    activation_hash = _sha256_tensors(*activation_parts)

    if PROFILE_HOST_ONLY:
        if not PROFILE_DIRECT_DECODE:
            raise RuntimeError("QWEN38_MC_PROFILE_HOST_ONLY requires QWEN38_MC_PROFILE_DIRECT_DECODE=1")
        # Count-7 latency already measures the unchanged optimized baseline.
        # Host-only Tracy processes omit its decode setup so profiler capacity
        # is spent entirely on the final segmented path.
        baseline_decode_ms = float("nan")
        baseline_decode_host = None
        _profile_checkpoint(mesh_device)
    else:
        baseline.prepare_decode_state()
        baseline_warm = baseline.decode_forward(decode_hidden, **baseline_decode_kwargs)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(baseline_warm)
        baseline.prepare_decode_state()
        baseline_trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        baseline_output = baseline.decode_forward(decode_hidden, **baseline_decode_kwargs)
        ttnn.end_trace_capture(mesh_device, baseline_trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        baseline_decode_host = ttnn.to_torch(ttnn.get_device_tensors(baseline_output)[0])
        for _ in range(3):
            ttnn.execute_trace(mesh_device, baseline_trace_id, cq_id=0, blocking=True)
        _profile_checkpoint(mesh_device)
        signpost(f"MC_BASELINE_DECODE_L{layer_idx}")
        started = time.perf_counter()
        for _ in range(DECODE_REPLAYS):
            ttnn.execute_trace(mesh_device, baseline_trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        baseline_decode_ms = (time.perf_counter() - started) * 1000.0 / DECODE_REPLAYS
        signpost(f"MC_BASELINE_DECODE_L{layer_idx}_END")
        ttnn.release_trace(mesh_device, baseline_trace_id)
        ttnn.deallocate(baseline_output)

    segmented = HostBackedSegmentedDecodeTrace.capture(host_backed, host_decode_hidden, **host_capture_kwargs)
    host_decode_pcc = (
        float("nan") if baseline_decode_host is None else H.pcc(baseline_decode_host, _fractured_host(segmented.output))
    )
    if baseline_decode_host is not None:
        assert host_decode_pcc >= H.PCC_BAR
    replay_kwargs = {}
    if layer_idx == 1:
        replay_kwargs.update(ple_input_ids=token_ids[:, prefill_seq_len:], request_ids=("perf-host",))
    for _ in range(3):
        segmented.replay(**replay_kwargs)
    _profile_checkpoint(mesh_device)
    signpost(f"MC_HOST_DECODE_L{layer_idx}")
    decode_service_before = _host_service_snapshot(host_backed)
    samples = []
    segment_totals = {
        "ple_seconds": 0.0,
        "front_trace_seconds": 0.0,
        "expert_service_seconds": 0.0,
        "back_trace_seconds": 0.0,
    }
    for _ in range(DECODE_REPLAYS):
        started = time.perf_counter()
        segmented.replay(**replay_kwargs)
        samples.append((time.perf_counter() - started) * 1000.0)
        for name in segment_totals:
            segment_totals[name] += segmented.last_timing[name]
    signpost(f"MC_HOST_DECODE_L{layer_idx}_END")
    decode_service_after = _host_service_snapshot(host_backed)
    decode_service_delta = _host_service_delta(decode_service_before, decode_service_after)
    host_decode_ms = statistics.fmean(samples)
    host_decode_p50_ms = statistics.median(samples)
    speedup = baseline_decode_ms / host_decode_ms
    efficiency = speedup / MultichipDecoder.TP_SIZE
    assert list(segmented.output.shape) == [1, 1, 4, 1280]
    assert list(_fractured_host(segmented.output).shape) == [1, 1, 1, 10240]
    metrics = host_backed.host_expert_cache.metrics()
    timing_ms = {name: seconds * 1000.0 / DECODE_REPLAYS for name, seconds in segment_totals.items()}
    _print_host_service_evidence(
        layer_idx=layer_idx,
        phase="decode",
        delta=decode_service_delta,
        end_to_end_seconds=sum(samples) / 1000.0,
        service_seconds=segment_totals["ple_seconds"] + segment_totals["expert_service_seconds"],
    )
    print(
        f"MC_HOST_PERFEVIDENCE mesh=1x2 layer={layer_idx} weights=real-checkpoint "
        f"activation_sha256={activation_hash} prefill_pcc={prefill_pcc:.8f} decode_pcc={host_decode_pcc:.8f} "
        f"prefill_seq={prefill_seq_len} "
        f"baseline_prefill_ms={baseline_prefill_ms:.6f} host_prefill_ms={host_prefill_ms:.6f} "
        f"baseline_traced_decode_ms={baseline_decode_ms:.6f} host_segmented_decode_ms={host_decode_ms:.6f} "
        f"host_decode_p50_ms={host_decode_p50_ms:.6f} speedup={speedup:.6f} efficiency={efficiency:.6f} "
        f"ple_ms={timing_ms['ple_seconds']:.6f} front_trace_ms={timing_ms['front_trace_seconds']:.6f} "
        f"expert_service_ms={timing_ms['expert_service_seconds']:.6f} "
        f"back_trace_ms={timing_ms['back_trace_seconds']:.6f} expert_requests={metrics['requests']} "
        f"expert_hits={metrics['hits']} expert_misses={metrics['misses']} h2d_bytes={metrics['h2d_bytes']} "
        f"decode_replays={DECODE_REPLAYS}"
    )
    segmented.release()
    host_backed.close_host_backing()
    if reference_store is not None:
        reference_store.close()


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 100_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("edge", ("baseline_tp", "residual", "host", "ep_host_baseline"))
def test_gdn_prefill_residual_topology_diagnostic(bh_1d_mesh_device, device_params, edge):
    """AutoFix ladder for the current real-activation layer-0 prefill miss."""

    if os.environ.get("QWEN38_MC_RUN_GDN_RESIDUAL_DIAG", "0") != "1":
        pytest.skip("set QWEN38_MC_RUN_GDN_RESIDUAL_DIAG=1 for the focused AutoFix A/B")
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh_device = bh_1d_mesh_device
    config = H.target_config()
    state = H.load_real_layer_state(0)
    common = dict(
        hf_config=config,
        layer_idx=0,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=128,
    )
    hidden, _, _ = _real_activations(0)
    logical = HOST_PREFILL_SEQ_LEN
    logical_hidden = hidden[:, :, :logical].contiguous()
    activation_hash = _sha256_tensors(logical_hidden)
    replicated_hidden = _upload(logical_hidden, mesh_device)
    fractured_hidden = _fractured_upload(logical_hidden, mesh_device)
    lhs_captures = rhs_captures = None
    if edge == "baseline_tp":
        policy = os.environ.get("QWEN38_MC_DIAG_EXPERT_POLICY", "")
        policy_kwargs = {"optimization_policy": policy} if policy else {}
        lhs = OptimizedDecoder.from_state_dict(state, **common, **policy_kwargs)
        rhs = MultichipDecoder.from_state_dict(state, fractured_residual=False, **common, **policy_kwargs)
        if os.environ.get("QWEN38_MC_DIAG_EXPERT_FIDELITY", "") == "hifi2":
            rhs.expert_compute_cfg = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        lhs_captures = _capture_gdn_prefill_pipeline(lhs)
        if os.environ.get("QWEN38_MC_DIAG_BASELINE_ROUTING", "0") == "1":

            def baseline_routing(logits):
                ttnn.deallocate(logits)
                return ttnn.clone(lhs_captures["routing"][0][1], memory_config=ttnn.DRAM_MEMORY_CONFIG)

            rhs._routing_from_logits = baseline_routing
        rhs_captures = _capture_gdn_prefill_pipeline(rhs)
        lhs_output, lhs_sparse = _invoke_with_sparse_captures(lambda: lhs.prefill_forward(replicated_hidden))
        rhs_output, rhs_sparse = _invoke_with_sparse_captures(
            lambda: rhs.prefill_forward(replicated_hidden),
            down_dtype=(ttnn.float32 if os.environ.get("QWEN38_MC_DIAG_EXPERT_DOWN_DTYPE", "") == "fp32" else None),
        )
        lhs_host = ttnn.to_torch(ttnn.get_device_tensors(lhs_output)[0])
        rhs_host = ttnn.to_torch(ttnn.get_device_tensors(rhs_output)[0])
    elif edge == "ep_host_baseline":
        lhs = OptimizedDecoder.from_state_dict(state, **common)
        rhs = MultichipDecoder.from_checkpoint_host_backed(H.MODEL_SNAPSHOT, **common)
        lhs_captures = _capture_gdn_prefill_pipeline(lhs)
        rhs_captures = _capture_gdn_prefill_pipeline(rhs)
        lhs_output = lhs.prefill_forward(replicated_hidden)
        rhs_output = rhs.prefill_forward_fractured(fractured_hidden)
        lhs_host = ttnn.to_torch(ttnn.get_device_tensors(lhs_output)[0])
        rhs_host = _fractured_host(rhs_output)
    elif edge == "residual":
        lhs = MultichipDecoder.from_state_dict(state, fractured_residual=False, **common)
        rhs = MultichipDecoder.from_state_dict(state, fractured_residual=True, **common)
        lhs_output = lhs.prefill_forward(replicated_hidden)
        rhs_output = rhs.prefill_forward_fractured(fractured_hidden)
        lhs_host = ttnn.to_torch(ttnn.get_device_tensors(lhs_output)[0])
        rhs_host = _fractured_host(rhs_output)
    else:
        lhs = MultichipDecoder.from_state_dict(state, fractured_residual=True, **common)
        rhs = MultichipDecoder.from_checkpoint_host_backed(H.MODEL_SNAPSHOT, **common)
        lhs_output = lhs.prefill_forward_fractured(fractured_hidden)
        rhs_output = rhs.prefill_forward_fractured(fractured_hidden)
        lhs_host = _fractured_host(lhs_output)
        rhs_host = _fractured_host(rhs_output)
    del state
    ttnn.synchronize_device(mesh_device)
    if edge == "baseline_tp":
        _print_gdn_expert_weight_pcc(lhs, rhs)
        _print_gdn_sparse_stage_pcc(lhs_sparse, rhs_sparse)
        _print_gdn_prefill_pipeline_pcc(lhs_captures, rhs_captures)
    elif edge == "ep_host_baseline":
        baseline_routed = lhs_captures["routed"][0][1]
        ep_routed = rhs_captures["routed"][0][1]
        routed_pcc = H.pcc(_capture_host(baseline_routed), _capture_host(ep_routed, reduce_partials=True))
        print(f"MC_GDN_EP_BOUNDARY boundary=routed.output_owner_sum pcc={routed_pcc:.8f}")
    output_pcc = H.pcc(lhs_host, rhs_host)
    print(
        "MC_GDN_PREFILL_RESIDUAL_DIAG "
        f"layer=0 edge={edge} seq={logical} activation_sha256={activation_hash} output_pcc={output_pcc:.8f}"
    )
    assert output_pcc >= H.PCC_BAR


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 100_000_000}],
    indirect=True,
)
def test_qsa_residual_topology_through_real_hyper_consumer(bh_1d_mesh_device, device_params):
    """Compare the shipped fractured consumer with the optimized R baseline.

    The producer is the real zero-based layer-3 QSA output projection.  The
    baseline all-reduces its rank-local 2560-wide partial and runs the exact
    optimized replicated MLP hyper mixer.  The selected contender instead
    reduce-scatters that same partial into the production 4x1280 residual ABI
    and runs the landed distributed RMSNorm and sharded hyper projections.
    Its one gather for residual PCC is outside the measured trace.
    """

    if os.environ.get("QWEN38_MC_RUN_RESIDUAL_TOPOLOGY", "0") != "1":
        pytest.skip("set QWEN38_MC_RUN_RESIDUAL_TOPOLOGY=1 for the explicit topology experiment")

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh_device = bh_1d_mesh_device
    config = H.target_config()
    state = H.load_real_layer_state(3)
    baseline = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=3,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=4096,
    )
    del state
    layer = MultichipDecoder.from_checkpoint_host_backed(
        H.MODEL_SNAPSHOT,
        hf_config=config,
        layer_idx=3,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=4096,
    )
    hidden_host, _, activation_hash = _real_activations(3)
    hidden = _fractured_upload(hidden_host[:, :, :1], mesh_device)
    current_pos = _upload(
        torch.tensor([0], dtype=torch.int32),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cos, sin = H.rope_tables(4096)
    page_host = H.shuffled_page_table(4096)
    page_table, _, rot_mats = _paged_inputs(layer, mesh_device, page_host, cos, sin, prefill_seq_len=1)

    # Preserve the real QSA output-projection partial before the production
    # reduce-scatter.  Both measured consumers receive this identical tensor.
    original_reduce_scatter = layer._reduce_scatter_block
    layer._reduce_scatter_block = lambda partial: partial
    try:
        attention = layer._decode_attention_host(
            hidden,
            current_pos=current_pos,
            page_table=page_table,
            rot_mats=rot_mats,
        )
        ttnn.synchronize_device(mesh_device)
    finally:
        layer._reduce_scatter_block = original_reduce_scatter

    baseline.collective_axis = layer.collective_axis
    baseline.collective_num_links = layer.collective_num_links
    baseline.collective_topology = layer.collective_topology
    full_hyper = layer.gather_residual(attention.hyper)
    s = layer.shapes

    baseline_trace = selected_trace = None
    baseline_outputs = selected_outputs = None
    boundary_full = None
    try:
        layer._decode_active = True
        baseline_trace, baseline_outputs, baseline_samples = _warmed_trace_samples(
            mesh_device,
            lambda: _replicated_residual_consumer(
                baseline,
                attention.block,
                full_hyper,
                attention.injection,
            ),
            replays=RESIDUAL_TOPOLOGY_REPLAYS,
        )
        baseline_host = [ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]) for tensor in baseline_outputs]
        ttnn.release_trace(mesh_device, baseline_trace)
        baseline_trace = None
        for tensor in baseline_outputs:
            if tensor.is_allocated():
                ttnn.deallocate(tensor)
        baseline_outputs = None

        selected_trace, selected_outputs, selected_samples = _warmed_trace_samples(
            mesh_device,
            lambda: _fully_fractured_residual_consumer(
                layer,
                attention.block,
                attention.hyper,
                layer.w["mlp_hc_norm"],
                layer.w["mlp_hc_down_inject"],
                layer.w["mlp_hc_up"],
                attention.injection,
            ),
            replays=RESIDUAL_TOPOLOGY_REPLAYS,
        )
        selected_host = [ttnn.to_torch(ttnn.get_device_tensors(selected_outputs[index])[0]) for index in (0, 1, 3)]
        ttnn.release_trace(mesh_device, selected_trace)
        selected_trace = None
        boundary_full = ttnn.all_gather(
            selected_outputs[2],
            dim=3,
            cluster_axis=layer.collective_axis,
            num_links=layer.collective_num_links,
            topology=layer.collective_topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        selected_hidden_host = ttnn.to_torch(ttnn.get_device_tensors(boundary_full)[0])
        selected_hidden_host = selected_hidden_host.reshape(1, 1, 1, s.hc_hidden_size)

        mixed_pcc = H.pcc(baseline_host[0], selected_host[0])
        injection_pcc = H.pcc(baseline_host[1], selected_host[1])
        residual_pcc = H.pcc(baseline_host[2], selected_hidden_host)
        # ``moe_input`` also packs shared-expert gate/up columns.  Those are
        # full-width in the optimized baseline and TP2-local in production;
        # the leading router logits are replicated and directly comparable.
        router_pcc = H.pcc(
            baseline_host[3][..., : s.num_experts],
            selected_host[2][..., : s.num_experts],
        )
        assert mixed_pcc >= H.PCC_BAR
        assert injection_pcc >= H.PCC_BAR
        assert residual_pcc >= H.PCC_BAR
        assert router_pcc >= H.PCC_BAR
        baseline_ms = statistics.median(baseline_samples)
        selected_ms = statistics.median(selected_samples)
        print(
            "MC_RESIDUAL_TOPOLOGY_EVIDENCE "
            f"mesh=1x2 producer=qsa_output_projection layer=3 activation_sha256={activation_hash} "
            "baseline_layout=replicated baseline_collective=all_reduce_2560 "
            "selected_layout=fractured_4x1280 selected_collectives="
            "reduce_scatter_2560+all_gather_rms_stats+"
            "all_reduce_packed324+all_gather_mixed1280 "
            f"mixed_pcc={mixed_pcc:.8f} injection_pcc={injection_pcc:.8f} "
            f"residual_pcc={residual_pcc:.8f} router_pcc={router_pcc:.8f} "
            f"baseline_ms={baseline_ms:.6f} selected_ms={selected_ms:.6f} "
            f"selected_over_baseline={selected_ms / baseline_ms:.6f} "
            f"samples=7 replays_per_sample={RESIDUAL_TOPOLOGY_REPLAYS}"
        )
    finally:
        layer._decode_active = False
        if selected_trace is not None:
            ttnn.release_trace(mesh_device, selected_trace)
        if baseline_trace is not None:
            ttnn.release_trace(mesh_device, baseline_trace)
        if boundary_full is not None and boundary_full.is_allocated():
            ttnn.deallocate(boundary_full)
        for tensors in (baseline_outputs, selected_outputs):
            if tensors is not None:
                for tensor in tensors:
                    if tensor.is_allocated():
                        ttnn.deallocate(tensor)
        for tensor in (
            full_hyper,
            attention.block,
            attention.injection,
        ):
            if tensor.is_allocated():
                ttnn.deallocate(tensor)
        layer.close_host_backing()


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 100_000_000}],
    indirect=True,
)
def test_qsa_complete_layer_residual_topology(bh_1d_mesh_device, device_params):
    """Controlled same-source TP2 R/S A/B over a complete QSA decoder layer."""

    if os.environ.get("QWEN38_MC_RUN_RESIDUAL_TOPOLOGY", "0") != "1":
        pytest.skip("set QWEN38_MC_RUN_RESIDUAL_TOPOLOGY=1 for the explicit topology experiment")

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh_device = bh_1d_mesh_device
    config = H.target_config()
    state = H.load_real_layer_state(3)
    common = dict(
        hf_config=config,
        layer_idx=3,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=4096,
    )
    replicated = MultichipDecoder.from_state_dict(state, fractured_residual=False, **common)
    fractured = MultichipDecoder.from_state_dict(state, fractured_residual=True, **common)
    del state

    hidden_host, _, activation_hash = _real_activations(3)
    replicated_hidden = _upload(hidden_host[:, :, :1], mesh_device)
    fractured_hidden = _fractured_upload(hidden_host[:, :, :1], mesh_device)
    current_pos = _upload(
        torch.tensor([0], dtype=torch.int32),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cos, sin = H.rope_tables(4096)
    page_host = H.shuffled_page_table(4096)
    page_table, _, rot_mats = _paged_inputs(replicated, mesh_device, page_host, cos, sin, prefill_seq_len=1)
    kwargs = {"current_pos": current_pos, "page_table": page_table, "rot_mats": rot_mats}

    replicated_trace = fractured_trace = None
    replicated_outputs = fractured_outputs = None
    gathered = None
    try:
        replicated.prepare_decode_state()
        replicated_trace, replicated_outputs, replicated_samples = _warmed_trace_samples(
            mesh_device,
            lambda: (replicated.decode_forward(replicated_hidden, **kwargs),),
            replays=RESIDUAL_TOPOLOGY_REPLAYS,
        )
        replicated_host = ttnn.to_torch(ttnn.get_device_tensors(replicated_outputs[0])[0])
        ttnn.release_trace(mesh_device, replicated_trace)
        replicated_trace = None
        ttnn.deallocate(replicated_outputs[0])
        replicated_outputs = None

        fractured.prepare_decode_state()
        fractured_trace, fractured_outputs, fractured_samples = _warmed_trace_samples(
            mesh_device,
            lambda: (fractured.decode_forward_fractured(fractured_hidden, **kwargs),),
            replays=RESIDUAL_TOPOLOGY_REPLAYS,
        )
        ttnn.release_trace(mesh_device, fractured_trace)
        fractured_trace = None
        gathered = fractured.gather_residual(fractured_outputs[0])
        fractured_host = ttnn.to_torch(ttnn.get_device_tensors(gathered)[0])
        output_pcc = H.pcc(replicated_host, fractured_host)
        assert output_pcc >= H.PCC_BAR

        replicated_ms = statistics.median(replicated_samples)
        fractured_ms = statistics.median(fractured_samples)
        print(
            "MC_FULL_LAYER_RESIDUAL_TOPOLOGY_EVIDENCE "
            f"mesh=1x2 layer=3 weights=real-checkpoint activation_sha256={activation_hash} "
            f"output_pcc={output_pcc:.8f} replicated_ms={replicated_ms:.6f} "
            f"fractured_ms={fractured_ms:.6f} fractured_over_replicated={fractured_ms / replicated_ms:.6f} "
            f"samples=7 replays_per_sample={RESIDUAL_TOPOLOGY_REPLAYS}"
        )
    finally:
        if fractured_trace is not None:
            ttnn.release_trace(mesh_device, fractured_trace)
        if replicated_trace is not None:
            ttnn.release_trace(mesh_device, replicated_trace)
        if gathered is not None and gathered.is_allocated():
            ttnn.deallocate(gathered)
        for tensors in (replicated_outputs, fractured_outputs):
            if tensors is not None:
                for tensor in tensors:
                    if tensor.is_allocated():
                        ttnn.deallocate(tensor)
