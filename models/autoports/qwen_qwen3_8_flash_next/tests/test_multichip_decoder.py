# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Structural and hardware gates for the fixed-P300 TP2 decoder."""

from __future__ import annotations

import inspect

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import harness as H
from models.autoports.qwen_qwen3_8_flash_next.tt.model_config import HF_ADVERTISED_CONTEXT
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import (
    MultichipDecoder,
    MultichipMemoryPlan,
    _rank_local_config,
    _rank_local_state,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedDecoder

LAYER_KINDS = (0, 1, 3)


def _replicated_upload(tensor, mesh_device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=dtype,
        layout=layout,
    )


def _rank_zero_host(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


def _paged_inputs(layer, mesh_device, page_table_host, cos_host, sin_host, seq_len):
    page_table = _replicated_upload(page_table_host, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    chunk_tables = []
    for start, _, padded in layer.prefill_chunk_plan(seq_len):
        first = start // layer.block_size
        last = (start + padded) // layer.block_size
        chunk_tables.append(
            _replicated_upload(
                page_table_host[:, first:last],
                mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        )
    cos = _replicated_upload(cos_host.reshape(1, 1, layer.max_seq_len, -1), mesh_device)
    sin = _replicated_upload(sin_host.reshape(1, 1, layer.max_seq_len, -1), mesh_device)
    return page_table, chunk_tables, (cos, sin)


def _capture_block(layer, method_name, captures):
    original = getattr(layer, method_name)

    def wrapped(*args, **kwargs):
        captures[f"{method_name}_input"] = ttnn.clone(args[0], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = original(*args, **kwargs)
        captures[f"{method_name}_output"] = ttnn.clone(output, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return output

    setattr(layer, method_name, wrapped)


def _capture_gdn_epilogue(layer, captures):
    original = layer._gdn_epilogue

    def wrapped(core, z, *, public_shape):
        captures["gdn_core"] = ttnn.clone(core, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        captures["gdn_z"] = ttnn.clone(z, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return original(core, z, public_shape=public_shape)

    layer._gdn_epilogue = wrapped


def _capture_routing(layer, captures):
    original = layer._routed_experts

    def wrapped(x, routing):
        captures["routing"] = ttnn.clone(routing, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return original(x, routing)

    layer._routed_experts = wrapped


def test_multichip_class_and_memory_contract():
    assert issubclass(MultichipDecoder, OptimizedDecoder)
    assert MultichipDecoder.TARGET_MESH == (1, 2)
    plan = MultichipMemoryPlan()
    assert plan.standard_bfp4_expert_bytes == 33_973_862_400
    assert plan.uniform_bfp2_expert_bytes == 18_874_368_000
    assert plan.standard_bfp4_fits is False
    assert plan.max_bfp4_fraction == pytest.approx(0.5308186848958333)
    assert plan.max_compressed_bfp4_fraction == pytest.approx(0.3213106043198529)


def test_rank_local_config_contract():
    config = H.target_config()
    gdn = _rank_local_config(config, 0).text_config
    qsa = _rank_local_config(config, 3).text_config
    assert gdn.linear_num_key_heads == 16
    assert gdn.linear_num_value_heads == 48
    assert qsa.num_attention_heads == 12
    assert qsa.num_key_value_heads == 1
    for local in (gdn, qsa):
        assert local.moe_intermediate_size == 320
        assert local.shared_expert_intermediate_size == 320
        assert local.num_experts == 512 and local.num_experts_per_tok == 10
        assert local.hidden_size == 2560 and local.hc_count == 4


def test_rank_local_checkpoint_shapes_without_allocating_weights():
    # Meta tensors exercise exact shape/slice/packing contracts without a 5 GiB
    # host allocation.  Runtime loading uses real checkpoint tensors.
    state = {
        "mlp.shared_expert.gate_proj.weight": torch.empty(640, 2560, device="meta"),
        "mlp.shared_expert.up_proj.weight": torch.empty(640, 2560, device="meta"),
        "mlp.shared_expert.down_proj.weight": torch.empty(2560, 640, device="meta"),
        "mlp.experts.gate_up_proj": torch.empty(512, 1280, 2560, device="meta"),
        "mlp.experts.down_proj": torch.empty(512, 2560, 640, device="meta"),
        "linear_attn.in_proj_qkv.weight": torch.empty(10240, 2560, device="meta"),
        "linear_attn.in_proj_z.weight": torch.empty(6144, 2560, device="meta"),
        "linear_attn.in_proj_b.weight": torch.empty(48, 2560, device="meta"),
        "linear_attn.in_proj_a.weight": torch.empty(48, 2560, device="meta"),
        "linear_attn.dt_bias": torch.empty(48, device="meta"),
        "linear_attn.A_log": torch.empty(48, device="meta"),
        "linear_attn.conv1d.weight": torch.empty(10240, 1, 4, device="meta"),
        "linear_attn.out_proj.weight": torch.empty(2560, 6144, device="meta"),
        "self_attn.q_proj.weight": torch.empty(12288, 2560, device="meta"),
        "self_attn.k_proj.weight": torch.empty(512, 2560, device="meta"),
        "self_attn.v_proj.weight": torch.empty(512, 2560, device="meta"),
        "self_attn.o_proj.weight": torch.empty(2560, 6144, device="meta"),
    }
    for rank in (0, 1):
        local = _rank_local_state(state, rank, shard_gdn=False)
        assert local["mlp.experts.gate_up_proj"].shape == (512, 640, 2560)
        assert local["mlp.experts.down_proj"].shape == (512, 2560, 320)
        assert local["linear_attn.in_proj_qkv.weight"].shape == (10240, 2560)
        assert local["linear_attn.in_proj_z.weight"].shape == (6144, 2560)
        assert local["linear_attn.conv1d.weight"].shape == (10240, 1, 4)
        assert local["linear_attn.out_proj.weight"].shape == (2560, 6144)
        assert local["self_attn.q_proj.weight"].shape == (6144, 2560)
        assert local["self_attn.k_proj.weight"].shape == (256, 2560)
        assert local["self_attn.o_proj.weight"].shape == (2560, 3072)


def test_runtime_collective_has_no_host_fallback():
    source = inspect.getsource(MultichipDecoder._all_reduce_block)
    assert "ttnn.all_reduce" in source
    for forbidden in ("torch", "to_torch", "from_torch", "as_tensor"):
        assert forbidden not in source


@pytest.mark.parametrize(
    "seq_len",
    [1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 2047, 2048, 2049, HF_ADVERTISED_CONTEXT - 1, HF_ADVERTISED_CONTEXT],
)
def test_multichip_prefill_plan_preserves_non_aligned_contract(seq_len):
    layer = object.__new__(MultichipDecoder)
    layer.max_seq_len = 262144
    plan = layer.prefill_chunk_plan(seq_len)
    assert sum(logical for _, logical, _ in plan) == seq_len
    assert plan[-1][1] == ((seq_len - 1) % 128) + 1


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_multichip_layer0_decode_structural_smoke(bh_1d_mesh_device, device_params):
    """Run local GDN/MoE graphs plus their collectives on the target P300."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh_device = bh_1d_mesh_device
    layer = MultichipDecoder.from_state_dict(
        None,
        hf_config=H.target_config(),
        layer_idx=0,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=128,
    )
    mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    hidden = ttnn.from_torch(
        torch.zeros(1, 1, 1, 10240, dtype=torch.bfloat16),
        device=mesh_device,
        mesh_mapper=mapper,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    current_pos = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32),
        device=mesh_device,
        mesh_mapper=mapper,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    with H.ForbidHostFallback():
        output = layer.decode_forward(hidden, current_pos=current_pos)
    ttnn.synchronize_device(mesh_device)

    rank_outputs = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(output)]
    assert len(rank_outputs) == 2
    assert list(output.shape) == [1, 1, 1, 10240]
    assert torch.equal(rank_outputs[0], rank_outputs[1])


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_multichip_real_weights_match_optimized_baseline(bh_1d_mesh_device, device_params, layer_idx):
    """Direct TTNN optimized-baseline PCC for GDN, PLE+GDN and QSA."""

    torch.manual_seed(20260827 + layer_idx)
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh_device = bh_1d_mesh_device
    config = H.target_config()
    state = H.load_real_layer_state(layer_idx)
    max_seq_len = 4096 if layer_idx == 3 else 128
    seq_len = 33
    hidden = (torch.randn(1, 1, seq_len + 1, 10240, dtype=torch.bfloat16) * 0.02).contiguous()
    ple = None
    if layer_idx == 1:
        ple = (torch.randn(1, 1, seq_len + 1, 2560, dtype=torch.bfloat16) * 0.02).contiguous()

    baseline = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=max_seq_len,
    )
    multichip = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=max_seq_len,
    )
    prefill_hidden = _replicated_upload(hidden[:, :, :seq_len], mesh_device)
    baseline_kwargs = {}
    multichip_kwargs = {}
    if ple is not None:
        prefill_ple = _replicated_upload(ple[:, :, :seq_len], mesh_device)
        baseline_kwargs["ple_embeddings"] = prefill_ple
        multichip_kwargs["ple_embeddings"] = prefill_ple
    if layer_idx == 3:
        cos, sin = H.rope_tables(max_seq_len)
        page_host = H.shuffled_page_table(max_seq_len)
        page, chunk_pages, rot = _paged_inputs(baseline, mesh_device, page_host, cos, sin, seq_len)
        baseline_kwargs.update(page_table=page, page_tables_per_chunk=chunk_pages, rot_mats=rot)
        multichip_kwargs.update(page_table=page, page_tables_per_chunk=chunk_pages, rot_mats=rot)

    with H.ForbidHostFallback():
        baseline_prefill = baseline.prefill_forward(prefill_hidden, **baseline_kwargs)
        multichip_prefill = multichip.prefill_forward(prefill_hidden, **multichip_kwargs)
    ttnn.synchronize_device(mesh_device)
    multichip_prefill_ranks = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(multichip_prefill)]
    assert torch.equal(multichip_prefill_ranks[0], multichip_prefill_ranks[1])
    prefill_pcc = H.pcc(_rank_zero_host(baseline_prefill), _rank_zero_host(multichip_prefill))
    print(f"MULTICHIPPCC layer={layer_idx} prefill={prefill_pcc:.8f}")
    assert prefill_pcc >= H.PCC_BAR
    if layer_idx == 3:
        for cache_name, baseline_cache, multichip_cache in (
            ("key", baseline.kv_cache[0], multichip.kv_cache[0]),
            ("value", baseline.kv_cache[1], multichip.kv_cache[1]),
        ):
            assert list(multichip_cache.shape) == [64, 1, 64, 256]
            local_caches = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(multichip_cache)]
            reconstructed = torch.cat(local_caches, dim=1)
            cache_pcc = H.pcc(_rank_zero_host(baseline_cache), reconstructed)
            print(f"MULTICHIPCACHEPCC phase=prefill cache={cache_name} pcc={cache_pcc:.8f}")
            assert cache_pcc >= H.PCC_BAR
        assert list(multichip.indexer_cache.shape) == [64, 1, 64, 128]
        index_pcc = H.pcc(_rank_zero_host(baseline.indexer_cache), _rank_zero_host(multichip.indexer_cache))
        print(f"MULTICHIPCACHEPCC phase=prefill cache=indexer pcc={index_pcc:.8f}")
        assert index_pcc >= H.PCC_BAR

    baseline.prepare_decode_state()
    multichip.prepare_decode_state()
    if layer_idx != 3:
        user_recurrent_shards = ttnn.get_device_tensors(multichip.user_recurrent_state[0])
        decode_recurrent_shards = ttnn.get_device_tensors(multichip.recurrent_state)
        for rank, (expected, actual) in enumerate(zip(user_recurrent_shards, decode_recurrent_shards)):
            copy_pcc = H.pcc(ttnn.to_torch(expected), ttnn.to_torch(actual))
            print(f"MULTICHIPSTATECOPY layer={layer_idx} state=recurrent rank={rank} pcc={copy_pcc:.8f}")
        user_conv_shards = ttnn.get_device_tensors(multichip.user_conv_state[0])
        for tap, target in enumerate(multichip.fused_conv_state):
            target_shards = ttnn.get_device_tensors(target)
            for rank, (source, destination) in enumerate(zip(user_conv_shards, target_shards)):
                expected = ttnn.to_torch(source)[:, :, tap : tap + 1, :]
                copy_pcc = H.pcc(expected, ttnn.to_torch(destination))
                print(f"MULTICHIPSTATECOPY layer={layer_idx} state=conv{tap} rank={rank} pcc={copy_pcc:.8f}")
    baseline_captures = {}
    multichip_captures = {}
    attention_method = "_qsa_decode" if layer_idx == 3 else "_gdn_decode"
    for layer, captures in ((baseline, baseline_captures), (multichip, multichip_captures)):
        _capture_block(layer, attention_method, captures)
        _capture_block(layer, "_moe", captures)
        _capture_routing(layer, captures)
        if layer_idx != 3:
            _capture_gdn_epilogue(layer, captures)
    decode_hidden = _replicated_upload(hidden[:, :, -1:], mesh_device)
    current_pos = _replicated_upload(
        torch.tensor([seq_len], dtype=torch.int32),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    baseline_decode_kwargs = {"current_pos": current_pos}
    multichip_decode_kwargs = {"current_pos": current_pos}
    if ple is not None:
        decode_ple = _replicated_upload(ple[:, :, -1:], mesh_device)
        baseline_decode_kwargs["ple_embeddings"] = decode_ple
        multichip_decode_kwargs["ple_embeddings"] = decode_ple
    if layer_idx == 3:
        baseline_decode_kwargs.update(page_table=page, rot_mats=rot)
        multichip_decode_kwargs.update(page_table=page, rot_mats=rot)
    with H.ForbidHostFallback():
        baseline_decode = baseline.decode_forward(decode_hidden, **baseline_decode_kwargs)
        multichip_decode = multichip.decode_forward(decode_hidden, **multichip_decode_kwargs)
    ttnn.synchronize_device(mesh_device)
    multichip_decode_ranks = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(multichip_decode)]
    assert torch.equal(multichip_decode_ranks[0], multichip_decode_ranks[1])
    for name in (f"{attention_method}_input", f"{attention_method}_output", "_moe_input", "_moe_output"):
        block_pcc = H.pcc(_rank_zero_host(baseline_captures[name]), _rank_zero_host(multichip_captures[name]))
        print(f"MULTICHIPBLOCKPCC layer={layer_idx} block={name} pcc={block_pcc:.8f}")
    if layer_idx != 3:
        for name in ("gdn_core", "gdn_z"):
            baseline_value = _rank_zero_host(baseline_captures[name])
            local_values = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(multichip_captures[name])]
            multichip_value = (
                local_values[0]
                if multichip.shapes.linear_num_value_heads == baseline.shapes.linear_num_value_heads
                else torch.cat(local_values, dim=-2)
            )
            value_pcc = H.pcc(baseline_value, multichip_value)
            print(f"MULTICHIPBLOCKPCC layer={layer_idx} block={name} pcc={value_pcc:.8f}")
    decode_pcc = H.pcc(_rank_zero_host(baseline_decode), _rank_zero_host(multichip_decode))
    print(f"MULTICHIPPCC layer={layer_idx} decode={decode_pcc:.8f}")
    assert decode_pcc >= H.PCC_BAR
    routing = _rank_zero_host(multichip_captures["routing"])
    assert int(routing[0, 0, 0].ne(0).sum()) == config.text_config.num_experts_per_tok
    if layer_idx == 3:
        for cache_name, baseline_cache, multichip_cache in (
            ("key", baseline.kv_cache[0], multichip.kv_cache[0]),
            ("value", baseline.kv_cache[1], multichip.kv_cache[1]),
        ):
            local_caches = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(multichip_cache)]
            cache_pcc = H.pcc(_rank_zero_host(baseline_cache), torch.cat(local_caches, dim=1))
            print(f"MULTICHIPCACHEPCC phase=decode cache={cache_name} pcc={cache_pcc:.8f}")
            assert cache_pcc >= H.PCC_BAR
        index_pcc = H.pcc(_rank_zero_host(baseline.indexer_cache), _rank_zero_host(multichip.indexer_cache))
        print(f"MULTICHIPCACHEPCC phase=decode cache=indexer pcc={index_pcc:.8f}")
        assert index_pcc >= H.PCC_BAR


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 64_000_000}],
    indirect=True,
)
def test_multichip_decode_trace_replay_determinism(bh_1d_mesh_device, device_params, layer_idx):
    """Capture/replay the real local graph and CCL with nonzero setup weights."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh_device = bh_1d_mesh_device
    config = H.target_config()
    max_seq_len = 4096 if layer_idx == 3 else 128
    layer = MultichipDecoder.from_state_dict(
        H.make_partial_state(config, layer_idx),
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=max_seq_len,
    )
    hidden = _replicated_upload(
        torch.randn(1, 1, 1, 10240, generator=torch.Generator().manual_seed(8800 + layer_idx)).bfloat16() * 0.02,
        mesh_device,
    )
    current_pos = _replicated_upload(
        torch.tensor([0], dtype=torch.int32),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    kwargs = {"current_pos": current_pos}
    if layer_idx == 1:
        kwargs["ple_embeddings"] = _replicated_upload(torch.zeros(1, 1, 1, 2560, dtype=torch.bfloat16), mesh_device)
    if layer_idx == 3:
        cos, sin = H.rope_tables(max_seq_len)
        page_host = H.shuffled_page_table(max_seq_len)
        page, _, rot = _paged_inputs(layer, mesh_device, page_host, cos, sin, 1)
        kwargs.update(page_table=page, rot_mats=rot)

    layer.prepare_decode_state()
    with H.ForbidHostFallback():
        warm = layer.decode_forward(hidden, **kwargs)
    ttnn.synchronize_device(mesh_device)
    warm_host = _rank_zero_host(warm)
    layer.prepare_decode_state()
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    with H.ForbidHostFallback():
        traced = layer.decode_forward(hidden, **kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    replay_outputs = []
    for _ in range(5):
        layer.prepare_decode_state()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        replay_outputs.append(_rank_zero_host(traced))
    ttnn.release_trace(mesh_device, trace_id)
    replay_pcc = H.pcc(warm_host, replay_outputs[0])
    print(f"MULTICHIPTRACE layer={layer_idx} warm_replay_pcc={replay_pcc:.8f}")
    assert replay_pcc >= H.PCC_BAR
    assert all(torch.equal(replay_outputs[0], output) for output in replay_outputs[1:])


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_multichip_stacked_decoder_layout_contract(bh_1d_mesh_device, device_params):
    """A replicated layer output is directly consumable by every layer kind."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh_device = bh_1d_mesh_device
    config = H.target_config()
    layers = [
        MultichipDecoder.from_state_dict(
            H.make_partial_state(config, layer_idx),
            hf_config=config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            max_batch=1,
            max_seq_len=4096 if layer_idx == 3 else 128,
        )
        for layer_idx in LAYER_KINDS
    ]
    for layer in layers:
        layer.prepare_decode_state()

    hidden = _replicated_upload(
        torch.randn(1, 1, 1, 10240, generator=torch.Generator().manual_seed(2401)).bfloat16() * 0.02,
        mesh_device,
    )
    current_pos = _replicated_upload(
        torch.tensor([0], dtype=torch.int32),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    ple = _replicated_upload(torch.zeros(1, 1, 1, 2560, dtype=torch.bfloat16), mesh_device)
    cos, sin = H.rope_tables(4096)
    page_host = H.shuffled_page_table(4096)
    page, _, rot = _paged_inputs(layers[-1], mesh_device, page_host, cos, sin, 1)
    kwargs = (
        {"current_pos": current_pos},
        {"current_pos": current_pos, "ple_embeddings": ple},
        {"current_pos": current_pos, "page_table": page, "rot_mats": rot},
    )

    with H.ForbidHostFallback():
        output = hidden
        for layer, layer_kwargs in zip(layers, kwargs):
            output = layer.decode_forward(output, **layer_kwargs)
    ttnn.synchronize_device(mesh_device)
    ranks = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(output)]
    assert list(output.shape) == [1, 1, 1, 10240]
    assert torch.equal(ranks[0], ranks[1])


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_multichip_batch32_decode_contract(bh_1d_mesh_device, device_params, layer_idx):
    """Preserve batch, per-user state/position and distinct QSA page tables."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh_device = bh_1d_mesh_device
    config = H.target_config()
    max_seq_len = 4096 if layer_idx == 3 else 128
    layer = MultichipDecoder.from_state_dict(
        None,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=32,
        max_seq_len=max_seq_len,
    )
    layer.prepare_decode_state()
    hidden = _replicated_upload(torch.zeros(1, 1, 32, 10240, dtype=torch.bfloat16), mesh_device)
    positions_host = (torch.arange(32, dtype=torch.int32) + 33).contiguous()
    current_pos = _replicated_upload(
        positions_host,
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    kwargs = {"current_pos": current_pos}
    if layer_idx == 1:
        kwargs["ple_embeddings"] = _replicated_upload(torch.zeros(1, 1, 32, 2560, dtype=torch.bfloat16), mesh_device)
    if layer_idx == 3:
        blocks_per_user = max_seq_len // layer.block_size
        page_host = torch.arange(32 * blocks_per_user, dtype=torch.int32).reshape(32, blocks_per_user)
        page_host[1::2] = page_host[1::2].flip(-1)
        page = _replicated_upload(
            page_host,
            mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        cos, sin = H.rope_tables(max_seq_len)
        rot = (
            _replicated_upload(cos.reshape(1, 1, max_seq_len, -1), mesh_device),
            _replicated_upload(sin.reshape(1, 1, max_seq_len, -1), mesh_device),
        )
        kwargs.update(page_table=page, rot_mats=rot)

    with H.ForbidHostFallback():
        output = layer.decode_forward(hidden, **kwargs)
    ttnn.synchronize_device(mesh_device)
    ranks = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(output)]
    assert list(output.shape) == [1, 1, 32, 10240]
    assert torch.equal(ranks[0], ranks[1])


@pytest.mark.long_context
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 64_000_000}],
    indirect=True,
)
def test_multichip_qsa_trace_at_advertised_context(bh_1d_mesh_device, device_params):
    """Exercise local KV heads and maximum page/position geometry in trace."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh_device = bh_1d_mesh_device
    layer = MultichipDecoder.from_state_dict(
        None,
        hf_config=H.target_config(),
        layer_idx=3,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=HF_ADVERTISED_CONTEXT,
    )
    assert layer.prefill_chunk_plan(HF_ADVERTISED_CONTEXT - 1)[-1] == (
        HF_ADVERTISED_CONTEXT - 128,
        127,
        128,
    )
    assert list(layer.kv_cache[0].shape) == [4096, 1, 64, 256]
    assert list(layer.indexer_cache.shape) == [4096, 1, 64, 128]
    hidden = _replicated_upload(torch.zeros(1, 1, 1, 10240, dtype=torch.bfloat16), mesh_device)
    current_pos = _replicated_upload(
        torch.tensor([HF_ADVERTISED_CONTEXT - 1], dtype=torch.int32),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    page_host = H.shuffled_page_table(HF_ADVERTISED_CONTEXT)
    page = _replicated_upload(
        page_host,
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cos, sin = H.rope_tables(HF_ADVERTISED_CONTEXT)
    rot = (
        _replicated_upload(cos.reshape(1, 1, HF_ADVERTISED_CONTEXT, -1), mesh_device),
        _replicated_upload(sin.reshape(1, 1, HF_ADVERTISED_CONTEXT, -1), mesh_device),
    )
    kwargs = {"current_pos": current_pos, "page_table": page, "rot_mats": rot}
    with H.ForbidHostFallback():
        layer.decode_forward(hidden, **kwargs)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    with H.ForbidHostFallback():
        traced = layer.decode_forward(hidden, **kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ranks = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(traced)]
    assert list(traced.shape) == [1, 1, 1, 10240]
    assert torch.equal(ranks[0], ranks[1])
    ttnn.release_trace(mesh_device, trace_id)
