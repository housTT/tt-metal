# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Warmed fixed-P300 multichip prefill and traced-decode windows."""

from __future__ import annotations

import hashlib
import os
import time

import pytest
import torch
from tracy import signpost

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import harness as H
from models.autoports.qwen_qwen3_8_flash_next.tests.test_optimized_decoder_perf import _real_activations
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MultichipDecoder

LAYER_KINDS = tuple(int(layer) for layer in os.environ.get("QWEN38_MC_PERF_LAYERS", "0,1,3").split(","))
DECODE_REPLAYS = int(os.environ.get("QWEN38_MC_PERF_DECODE_REPLAYS", "10"))
PROFILE_DIRECT_DECODE = os.environ.get("QWEN38_MC_PROFILE_DIRECT_DECODE", "0") == "1"
PREFILL_SEQ_LEN = 128


def _upload(tensor, mesh_device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=dtype,
        layout=layout,
    )


def _paged_inputs(layer, mesh_device, page_table_host, cos_host, sin_host):
    page_table = _upload(page_table_host, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    chunk_tables = []
    for start, _, padded in layer.prefill_chunk_plan(PREFILL_SEQ_LEN):
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
    prefill_hidden = _upload(hidden_host[:, :, :PREFILL_SEQ_LEN], mesh_device)
    prefill_kwargs = {}
    if layer_idx == 1:
        prefill_kwargs["ple_embeddings"] = _upload(ple_host[:, :, :PREFILL_SEQ_LEN], mesh_device)
    if layer_idx == 3:
        cos, sin = H.rope_tables(max_seq_len)
        page_host = H.shuffled_page_table(max_seq_len)
        page, chunk_pages, rot = _paged_inputs(layer, mesh_device, page_host, cos, sin)
        prefill_kwargs.update(page_table=page, page_tables_per_chunk=chunk_pages, rot_mats=rot)

    routing = _observe_routing(layer, lambda: layer.prefill_forward(prefill_hidden, **prefill_kwargs))
    _profile_checkpoint(mesh_device)
    prefill_ms = None
    if mode in {"both", "prefill"}:
        signpost(f"MC_PERF_PREFILL_L{layer_idx}")
        start = time.perf_counter()
        with H.ForbidHostFallback():
            prefill_output = layer.prefill_forward(prefill_hidden, **prefill_kwargs)
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
    decode_hidden = _upload(hidden_host[:, :, PREFILL_SEQ_LEN:], mesh_device)
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
    decode_routing = _observe_routing(layer, lambda: layer.decode_forward(decode_hidden, **decode_kwargs))
    _profile_checkpoint(mesh_device)

    if PROFILE_DIRECT_DECODE:
        with H.ForbidHostFallback():
            decode_output = layer.decode_forward(decode_hidden, **decode_kwargs)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(decode_output)
        _profile_checkpoint(mesh_device)
        direct_outputs = []
        signpost(f"MC_PERF_DECODE_L{layer_idx}")
        start = time.perf_counter()
        with H.ForbidHostFallback():
            for _ in range(DECODE_REPLAYS):
                direct_outputs.append(layer.decode_forward(decode_hidden, **decode_kwargs))
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
            decode_output = layer.decode_forward(decode_hidden, **decode_kwargs)
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
    rank_outputs = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(decode_output)]
    assert list(decode_output.shape) == [1, 1, 1, 10240]
    assert torch.equal(rank_outputs[0], rank_outputs[1])
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
