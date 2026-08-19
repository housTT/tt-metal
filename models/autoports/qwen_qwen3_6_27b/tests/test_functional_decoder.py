# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import inspect
from pathlib import Path

import pytest
import torch
import ttnn
from safetensors import safe_open
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
    apply_rotary_pos_emb,
)
from tracy import signpost

from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import FunctionalDecoder
from models.common.utility_functions import comp_pcc


MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_CACHE = Path(
    os.environ.get(
        "QWEN36_MODEL_PATH",
        "/home/ttuser/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    )
)


def _load_layer_state(layer_idx: int):
    if not (MODEL_CACHE / "model.safetensors.index.json").exists():
        pytest.skip(f"Real Qwen3.6 checkpoint is not available at {MODEL_CACHE}")
    index = json.loads((MODEL_CACHE / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer_idx}."
    keys = [key for key in index if key.startswith(prefix)]
    by_shard = {}
    for key in keys:
        by_shard.setdefault(index[key], []).append(key)
    state = {}
    for shard, shard_keys in by_shard.items():
        with safe_open(MODEL_CACHE / shard, framework="pt", device="cpu") as handle:
            for key in shard_keys:
                state[key] = handle.get_tensor(key)
    return state


def _reference_layer(config, layer_idx: int, state):
    prefix = f"model.language_model.layers.{layer_idx}."
    layer = Qwen3_5DecoderLayer(config, layer_idx).to(torch.bfloat16).eval()
    layer.load_state_dict({key.removeprefix(prefix): value for key, value in state.items()}, strict=True)
    return layer


def _to_tt(tensor, mesh_device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _config():
    return AutoConfig.from_pretrained(MODEL_CACHE, local_files_only=True).text_config


def _pcc(reference, actual, threshold=0.995):
    assert reference.numel() == actual.numel(), (reference.shape, actual.shape)
    reference_flat = reference.float().reshape(-1)
    actual_flat = actual.float().reshape(-1)
    passed, message = comp_pcc(reference_flat, actual_flat, threshold)
    print(message)
    if not passed:
        print(
            "direct_pcc=",
            torch.corrcoef(torch.stack([reference_flat, actual_flat]))[0, 1].item(),
            "mae=",
            torch.mean(torch.abs(reference_flat - actual_flat)).item(),
            "max_error=",
            torch.max(torch.abs(reference_flat - actual_flat)).item(),
        )
    assert passed, message
    return message


@pytest.mark.parametrize("layer_idx,expected_kind", [(0, "linear_attention"), (3, "full_attention")])
def test_real_config_layer_kinds(layer_idx, expected_kind):
    config = _config()
    assert config.hidden_size == 5120
    assert config.intermediate_size == 17408
    assert config.max_position_embeddings == 262144
    assert config.layer_types[layer_idx] == expected_kind


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("seq_len", [1, 31, 32, 33, 63, 64, 65])
def test_full_attention_real_weight_paged_prefill(mesh_device, seq_len):
    torch.manual_seed(13)
    config = _config()
    layer_idx = 3
    state = _load_layer_state(layer_idx)
    reference_layer = _reference_layer(config, layer_idx, state)
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=config, layer_idx=layer_idx, mesh_device=mesh_device, page_block_size=64
    )

    # Exercise both sides of tile/page boundaries. A reversed page table makes
    # accidental contiguous-cache assumptions visible.
    physical_seq_len = ((seq_len + 31) // 32) * 32
    hidden = torch.randn(1, seq_len, config.hidden_size, dtype=torch.bfloat16) * 0.1
    decode_hidden = torch.randn(1, 1, config.hidden_size, dtype=torch.bfloat16) * 0.1
    padded_hidden = torch.nn.functional.pad(hidden, (0, 0, 0, physical_seq_len - seq_len))
    positions = torch.arange(seq_len).unsqueeze(0)
    padded_positions = torch.arange(physical_seq_len).unsqueeze(0)
    rotary = Qwen3_5TextRotaryEmbedding(config)
    cos, sin = rotary(hidden, positions)
    padded_cos, padded_sin = rotary(padded_hidden, padded_positions)
    decode_positions = torch.tensor([[seq_len]])
    decode_cos, decode_sin = rotary(decode_hidden, decode_positions)
    causal_mask = torch.full((1, 1, seq_len, seq_len), torch.finfo(torch.float32).min)
    causal_mask = torch.triu(causal_mask, diagonal=1).to(torch.bfloat16)
    reference_cache = DynamicCache(config=config)
    with torch.no_grad():
        reference = reference_layer(
            hidden,
            position_embeddings=(cos, sin),
            attention_mask=causal_mask,
            position_ids=positions,
            past_key_values=reference_cache,
        )
        reference_decode = reference_layer(
            decode_hidden,
            position_embeddings=(decode_cos, decode_sin),
            attention_mask=torch.zeros(1, 1, 1, seq_len + 1, dtype=torch.bfloat16),
            position_ids=decode_positions,
            past_key_values=reference_cache,
        )

    tt_hidden = _to_tt(padded_hidden.unsqueeze(1), mesh_device)
    tt_cos = _to_tt(padded_cos.unsqueeze(1), mesh_device)
    tt_sin = _to_tt(padded_sin.unsqueeze(1), mesh_device)
    padded_batch = 32
    page_table_host = torch.arange(padded_batch * 2, dtype=torch.int32).reshape(padded_batch, 2)
    page_table_host[0] = torch.tensor([1, 0], dtype=torch.int32)
    page_table = _to_tt(
        page_table_host,
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    kv_cache = decoder.allocate_paged_kv_cache(num_blocks=padded_batch * 2)
    output = decoder.prefill_forward(
        tt_hidden,
        logical_seq_len=seq_len,
        cos=tt_cos,
        sin=tt_sin,
        page_table=page_table,
        kv_cache=kv_cache,
    )
    actual = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0, 0]
    _pcc(reference, actual)

    padded_decode = torch.zeros(1, 1, padded_batch, config.hidden_size, dtype=torch.bfloat16)
    padded_decode[:, :, 0] = decode_hidden[:, 0]
    current_positions = _to_tt(
        torch.full((padded_batch,), seq_len, dtype=torch.int32),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    decode_output = decoder.decode_forward(
        _to_tt(padded_decode, mesh_device),
        current_positions=current_positions,
        cos=_to_tt(decode_cos.unsqueeze(0).expand(1, padded_batch, 1, -1).contiguous(), mesh_device),
        sin=_to_tt(decode_sin.unsqueeze(0).expand(1, padded_batch, 1, -1).contiguous(), mesh_device),
        page_table=page_table,
        kv_cache=kv_cache,
    )
    actual_decode = ttnn.to_torch(
        decode_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )[0, 0, 0]
    _pcc(reference_decode[0, 0], actual_decode)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_full_attention_real_weight_paged_decode_trace(mesh_device):
    torch.manual_seed(17)
    config = _config()
    layer_idx = 3
    state = _load_layer_state(layer_idx)
    reference_layer = _reference_layer(config, layer_idx, state)
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=config, layer_idx=layer_idx, mesh_device=mesh_device, page_block_size=64
    )

    prefill_len = 32
    prefill_hidden = torch.randn(1, prefill_len, config.hidden_size, dtype=torch.bfloat16) * 0.1
    decode_hidden = torch.randn(1, 1, config.hidden_size, dtype=torch.bfloat16) * 0.1
    prefill_positions = torch.arange(prefill_len).unsqueeze(0)
    decode_positions = torch.tensor([[prefill_len]])
    rotary = Qwen3_5TextRotaryEmbedding(config)
    prefill_cos, prefill_sin = rotary(prefill_hidden, prefill_positions)
    decode_cos, decode_sin = rotary(decode_hidden, decode_positions)
    prefill_mask = torch.full((1, 1, prefill_len, prefill_len), torch.finfo(torch.float32).min)
    prefill_mask = torch.triu(prefill_mask, diagonal=1).to(torch.bfloat16)
    cache = DynamicCache(config=config)
    with torch.no_grad():
        reference_layer(
            prefill_hidden,
            position_embeddings=(prefill_cos, prefill_sin),
            attention_mask=prefill_mask,
            position_ids=prefill_positions,
            past_key_values=cache,
        )
        reference = reference_layer(
            decode_hidden,
            position_embeddings=(decode_cos, decode_sin),
            attention_mask=torch.zeros(1, 1, 1, prefill_len + 1, dtype=torch.bfloat16),
            position_ids=decode_positions,
            past_key_values=cache,
        )

    padded_batch = 32
    page_table_torch = torch.arange(padded_batch, dtype=torch.int32).reshape(padded_batch, 1)
    page_table = _to_tt(page_table_torch, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    current_positions = _to_tt(
        torch.full((padded_batch,), prefill_len, dtype=torch.int32),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    kv_cache = decoder.allocate_paged_kv_cache(num_blocks=padded_batch)
    decoder.prefill_forward(
        _to_tt(prefill_hidden.unsqueeze(1), mesh_device),
        logical_seq_len=prefill_len,
        cos=_to_tt(prefill_cos.unsqueeze(1), mesh_device),
        sin=_to_tt(prefill_sin.unsqueeze(1), mesh_device),
        page_table=page_table,
        kv_cache=kv_cache,
    )

    padded_decode = torch.zeros(1, 1, padded_batch, config.hidden_size, dtype=torch.bfloat16)
    padded_decode[:, :, 0, :] = decode_hidden[:, 0, :]
    tt_decode = _to_tt(padded_decode, mesh_device)
    tt_cos = _to_tt(decode_cos.unsqueeze(0).expand(1, padded_batch, 1, -1).contiguous(), mesh_device)
    tt_sin = _to_tt(decode_sin.unsqueeze(0).expand(1, padded_batch, 1, -1).contiguous(), mesh_device)

    decoder.decode_forward(
        tt_decode,
        current_positions=current_positions,
        cos=tt_cos,
        sin=tt_sin,
        page_table=page_table,
        kv_cache=kv_cache,
    )
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward(
        tt_decode,
        current_positions=current_positions,
        cos=tt_cos,
        sin=tt_sin,
        page_table=page_table,
        kv_cache=kv_cache,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    first_replay = ttnn.to_torch(
        traced_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )[0, 0, 0].clone()
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    actual = ttnn.to_torch(
        traced_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )[0, 0, 0]
    ttnn.release_trace(mesh_device, trace_id)
    _pcc(reference[0, 0], actual)
    _pcc(first_replay, actual, threshold=0.9999)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("prefill_len", [1, 31, 32, 33, 63, 64, 65])
def test_linear_attention_real_weight_prefill_and_decode(mesh_device, prefill_len):
    torch.manual_seed(19)
    config = _config()
    layer_idx = 0
    state = _load_layer_state(layer_idx)
    reference_layer = _reference_layer(config, layer_idx, state)
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=config, layer_idx=layer_idx, mesh_device=mesh_device
    )

    # Exercise one exact HF DeltaNet chunk and the first non-aligned token of a
    # second chunk.  The latter also verifies that padded tokens do not mutate
    # the convolution or recurrent state.
    physical_prefill_len = ((prefill_len + 31) // 32) * 32
    prefill_hidden = torch.randn(1, prefill_len, config.hidden_size, dtype=torch.bfloat16) * 0.1
    padded_prefill_hidden = torch.nn.functional.pad(
        prefill_hidden, (0, 0, 0, physical_prefill_len - prefill_len)
    )
    decode_hidden = torch.randn(1, 1, config.hidden_size, dtype=torch.bfloat16) * 0.1
    dummy_prefill_rope = (
        torch.zeros(1, prefill_len, 64, dtype=torch.bfloat16),
        torch.zeros(1, prefill_len, 64, dtype=torch.bfloat16),
    )
    dummy_decode_rope = (
        torch.zeros(1, 1, 64, dtype=torch.bfloat16),
        torch.zeros(1, 1, 64, dtype=torch.bfloat16),
    )
    cache = DynamicCache(config=config)
    with torch.no_grad():
        reference_prefill = reference_layer(
            prefill_hidden,
            position_embeddings=dummy_prefill_rope,
            attention_mask=torch.ones(1, prefill_len, dtype=torch.bool),
            past_key_values=cache,
        )
        reference_decode = reference_layer(
            decode_hidden,
            position_embeddings=dummy_decode_rope,
            attention_mask=torch.ones(1, prefill_len + 1, dtype=torch.bool),
            past_key_values=cache,
        )

    linear_state = decoder.allocate_linear_state(batch_size=1)
    tt_prefill = _to_tt(padded_prefill_hidden.unsqueeze(1), mesh_device)
    prefill_output = decoder.prefill_forward(
        tt_prefill, logical_seq_len=prefill_len, linear_state=linear_state
    )
    actual_prefill = ttnn.to_torch(
        prefill_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )[0, 0]
    _pcc(reference_prefill, actual_prefill)

    tt_decode = _to_tt(decode_hidden.unsqueeze(1), mesh_device)
    current_positions = _to_tt(
        torch.tensor([prefill_len], dtype=torch.int32),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    first_decode_output = decoder.decode_forward(
        tt_decode, current_positions=current_positions, linear_state=linear_state
    )
    ttnn.synchronize_device(mesh_device)
    actual_first_decode = ttnn.to_torch(
        first_decode_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )[0, 0]
    _pcc(reference_decode, actual_first_decode)

    # The exact and over-boundary chunk cases carry the traced-decode and
    # determinism gates.  The other parameters are boundary-correctness cases;
    # their first real decode above is still PCC-gated.
    if prefill_len not in (64, 65):
        return

    # Capture records token two without dispatching it; replay executes that
    # token.  Advance the HF state once so replay receives a direct PCC gate.
    with torch.no_grad():
        reference_traced_decode = reference_layer(
            decode_hidden,
            position_embeddings=dummy_decode_rope,
            attention_mask=torch.ones(1, prefill_len + 2, dtype=torch.bool),
            past_key_values=cache,
        )

    # The recurrent state is intentionally device-resident and mutable.  Trace
    # capture and replay therefore validate subsequent tokens, not the already
    # checked first decode token.
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward(
        tt_decode, current_positions=current_positions, linear_state=linear_state
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    actual_decode = ttnn.to_torch(
        traced_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )[0, 0]
    ttnn.release_trace(mesh_device, trace_id)
    _pcc(reference_traced_decode, actual_decode)

    # A fresh state and the same input sequence must reproduce the first-token
    # result exactly enough to distinguish numerical nondeterminism from the
    # intentional state evolution above.
    repeat_state = decoder.allocate_linear_state(batch_size=1)
    decoder.prefill_forward(tt_prefill, logical_seq_len=prefill_len, linear_state=repeat_state)
    repeat_output = decoder.decode_forward(
        tt_decode, current_positions=current_positions, linear_state=repeat_state
    )
    repeat_actual = ttnn.to_torch(
        repeat_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )[0, 0]
    _pcc(actual_first_decode, repeat_actual, threshold=0.9999)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_linear_attention_real_weight_multi_user_prefill_and_decode(mesh_device):
    """Keep two independent DeltaNet users active through prefill and decode."""
    torch.manual_seed(41)
    config = _config()
    layer_idx = 0
    state = _load_layer_state(layer_idx)
    reference_layer = _reference_layer(config, layer_idx, state)
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=config, layer_idx=layer_idx, mesh_device=mesh_device
    )

    batch_size = 2
    prefill_len = 64
    prefill_hidden = (
        torch.randn(batch_size, prefill_len, config.hidden_size, dtype=torch.bfloat16) * 0.1
    )
    # Make the second user observably different even if a future random seed or
    # fixture changes.  A shared-state bug must therefore fail the PCC gate.
    prefill_hidden[1].add_(0.025)
    decode_hidden = torch.randn(batch_size, 1, config.hidden_size, dtype=torch.bfloat16) * 0.1
    decode_hidden[1].sub_(0.025)
    dummy_prefill_rope = (
        torch.zeros(batch_size, prefill_len, 64, dtype=torch.bfloat16),
        torch.zeros(batch_size, prefill_len, 64, dtype=torch.bfloat16),
    )
    dummy_decode_rope = (
        torch.zeros(batch_size, 1, 64, dtype=torch.bfloat16),
        torch.zeros(batch_size, 1, 64, dtype=torch.bfloat16),
    )
    reference_cache = DynamicCache(config=config)
    with torch.no_grad():
        reference_prefill = reference_layer(
            prefill_hidden,
            position_embeddings=dummy_prefill_rope,
            attention_mask=torch.ones(batch_size, prefill_len, dtype=torch.bool),
            past_key_values=reference_cache,
        )
        reference_decode = reference_layer(
            decode_hidden,
            position_embeddings=dummy_decode_rope,
            attention_mask=torch.ones(batch_size, prefill_len + 1, dtype=torch.bool),
            past_key_values=reference_cache,
        )

    linear_state = decoder.allocate_linear_state(batch_size=batch_size)
    actual_prefill = ttnn.to_torch(
        decoder.prefill_forward(
            _to_tt(prefill_hidden.unsqueeze(1), mesh_device),
            logical_seq_len=prefill_len,
            linear_state=linear_state,
        ),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )[:, 0]
    _pcc(reference_prefill, actual_prefill)
    for user in range(batch_size):
        _pcc(reference_prefill[user], actual_prefill[user])

    # DeltaNet does not consume absolute positions internally, but the public
    # decoder contract still requires a device-resident position for each user.
    current_positions = _to_tt(
        torch.full((batch_size,), prefill_len, dtype=torch.int32),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    actual_decode = ttnn.to_torch(
        decoder.decode_forward(
            _to_tt(decode_hidden.transpose(0, 1).unsqueeze(0), mesh_device),
            current_positions=current_positions,
            linear_state=linear_state,
        ),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )[0, 0]
    _pcc(reference_decode[:, 0], actual_decode)
    for user in range(batch_size):
        _pcc(reference_decode[user, 0], actual_decode[user])


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_full_attention_real_weight_multi_user_paged_prefill_and_decode(mesh_device):
    """Validate two paged users, including distinct page maps and decode positions."""
    torch.manual_seed(43)
    config = _config()
    layer_idx = 3
    state = _load_layer_state(layer_idx)
    reference_layer = _reference_layer(config, layer_idx, state)
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=config, layer_idx=layer_idx, mesh_device=mesh_device, page_block_size=64
    )

    batch_size = 2
    prefill_len = 64
    prefill_hidden = (
        torch.randn(batch_size, prefill_len, config.hidden_size, dtype=torch.bfloat16) * 0.1
    )
    prefill_hidden[1].add_(0.025)
    positions = torch.arange(prefill_len).expand(batch_size, -1)
    rotary = Qwen3_5TextRotaryEmbedding(config)
    prefill_cos, prefill_sin = rotary(prefill_hidden, positions)
    causal_mask = torch.full(
        (batch_size, 1, prefill_len, prefill_len), torch.finfo(torch.float32).min
    )
    causal_mask = torch.triu(causal_mask, diagonal=1).to(torch.bfloat16)
    with torch.no_grad():
        reference_prefill = reference_layer(
            prefill_hidden,
            position_embeddings=(prefill_cos, prefill_sin),
            attention_mask=causal_mask,
            position_ids=positions,
        )

    padded_batch = 32
    pages_per_user = 2
    # Every padded lane owns disjoint physical pages.  Reversing the two active
    # rows catches both cross-user aliasing and assumptions of contiguous pages.
    page_table_host = torch.arange(
        padded_batch * pages_per_user, dtype=torch.int32
    ).reshape(padded_batch, pages_per_user)
    page_table_host[0] = torch.tensor([1, 0], dtype=torch.int32)
    page_table_host[1] = torch.tensor([3, 2], dtype=torch.int32)
    page_table = _to_tt(
        page_table_host, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    kv_cache = decoder.allocate_paged_kv_cache(num_blocks=padded_batch * pages_per_user)
    actual_prefill = ttnn.to_torch(
        decoder.prefill_forward(
            _to_tt(prefill_hidden.unsqueeze(1), mesh_device),
            logical_seq_len=prefill_len,
            cos=_to_tt(prefill_cos.unsqueeze(1), mesh_device),
            sin=_to_tt(prefill_sin.unsqueeze(1), mesh_device),
            page_table=page_table,
            kv_cache=kv_cache,
        ),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )[:, 0]
    _pcc(reference_prefill, actual_prefill)
    for user in range(batch_size):
        _pcc(reference_prefill[user], actual_prefill[user])

    # User zero rewinds to the first token of its second half while user one
    # appends after the complete prefix.  Extra cached tokens for user zero are
    # deliberately present and must be excluded by its current position.
    decode_positions_host = torch.tensor([32, 64], dtype=torch.int64)
    decode_hidden = torch.randn(batch_size, 1, config.hidden_size, dtype=torch.bfloat16) * 0.1
    decode_hidden[1].sub_(0.025)
    reference_decodes = []
    decode_cos_users = []
    decode_sin_users = []
    for user, current_position in enumerate(decode_positions_host.tolist()):
        user_cache = DynamicCache(config=config)
        prefix = prefill_hidden[user : user + 1, :current_position]
        prefix_positions = torch.arange(current_position).unsqueeze(0)
        prefix_cos, prefix_sin = rotary(prefix, prefix_positions)
        prefix_mask = torch.full(
            (1, 1, current_position, current_position), torch.finfo(torch.float32).min
        )
        prefix_mask = torch.triu(prefix_mask, diagonal=1).to(torch.bfloat16)
        user_decode = decode_hidden[user : user + 1]
        user_position = torch.tensor([[current_position]])
        user_cos, user_sin = rotary(user_decode, user_position)
        with torch.no_grad():
            reference_layer(
                prefix,
                position_embeddings=(prefix_cos, prefix_sin),
                attention_mask=prefix_mask,
                position_ids=prefix_positions,
                past_key_values=user_cache,
            )
            reference_decode = reference_layer(
                user_decode,
                position_embeddings=(user_cos, user_sin),
                attention_mask=torch.zeros(
                    1, 1, 1, current_position + 1, dtype=torch.bfloat16
                ),
                position_ids=user_position,
                past_key_values=user_cache,
            )
        reference_decodes.append(reference_decode)
        decode_cos_users.append(user_cos)
        decode_sin_users.append(user_sin)

    padded_decode = torch.zeros(1, 1, padded_batch, config.hidden_size, dtype=torch.bfloat16)
    padded_decode[0, 0, :batch_size] = decode_hidden[:, 0]
    padded_cos = torch.zeros(1, padded_batch, 1, decoder.rotary_dim, dtype=torch.bfloat16)
    padded_sin = torch.zeros_like(padded_cos)
    padded_cos[0, :batch_size] = torch.cat(decode_cos_users, dim=0)
    padded_sin[0, :batch_size] = torch.cat(decode_sin_users, dim=0)
    # Inactive lanes use valid positions and their own page-table rows so the
    # vectorized cache update is race-free.
    padded_positions = torch.full((padded_batch,), 32, dtype=torch.int32)
    padded_positions[:batch_size] = decode_positions_host.to(torch.int32)
    actual_decode = ttnn.to_torch(
        decoder.decode_forward(
            _to_tt(padded_decode, mesh_device),
            current_positions=_to_tt(
                padded_positions,
                mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            cos=_to_tt(padded_cos, mesh_device),
            sin=_to_tt(padded_sin, mesh_device),
            page_table=page_table,
            kv_cache=kv_cache,
        ),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )[0, 0, :batch_size]
    reference_decode = torch.cat(reference_decodes, dim=0)[:, 0]
    _pcc(reference_decode, actual_decode)
    for user in range(batch_size):
        _pcc(reference_decode[user], actual_decode[user])


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 3], ids=["linear_attention", "full_attention"])
def test_real_weight_batch32_prefill_and_decode(mesh_device, layer_idx):
    """Exercise the target maximum of 32 active users with per-user PCC."""
    torch.manual_seed(19 if layer_idx == 0 else 13)
    config = _config()
    state = _load_layer_state(layer_idx)
    reference_layer = _reference_layer(config, layer_idx, state)
    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        page_block_size=64,
    )

    batch_size = 32
    prefill_len = 32
    base_prefill = torch.randn(1, prefill_len, config.hidden_size, dtype=torch.bfloat16) * 0.1
    base_decode = torch.randn(1, 1, config.hidden_size, dtype=torch.bfloat16) * 0.1
    prefill_hidden = base_prefill.repeat(batch_size, 1, 1)
    decode_hidden = base_decode.repeat(batch_size, 1, 1)
    # Keep every user distinct while using a numerically representative base
    # already covered by the single-user boundary matrix.
    user_offsets = torch.linspace(-0.002, 0.002, batch_size, dtype=torch.float32).to(
        torch.bfloat16
    )
    prefill_hidden.add_(user_offsets[:, None, None])
    decode_hidden.sub_(user_offsets[:, None, None])

    if config.layer_types[layer_idx] == "linear_attention":
        dummy_prefill_rope = (
            torch.zeros(batch_size, prefill_len, 64, dtype=torch.bfloat16),
            torch.zeros(batch_size, prefill_len, 64, dtype=torch.bfloat16),
        )
        dummy_decode_rope = (
            torch.zeros(batch_size, 1, 64, dtype=torch.bfloat16),
            torch.zeros(batch_size, 1, 64, dtype=torch.bfloat16),
        )
        reference_cache = DynamicCache(config=config)
        with torch.no_grad():
            reference_prefill = reference_layer(
                prefill_hidden,
                position_embeddings=dummy_prefill_rope,
                attention_mask=torch.ones(batch_size, prefill_len, dtype=torch.bool),
                past_key_values=reference_cache,
            )
            reference_decode = reference_layer(
                decode_hidden,
                position_embeddings=dummy_decode_rope,
                attention_mask=torch.ones(batch_size, prefill_len + 1, dtype=torch.bool),
                past_key_values=reference_cache,
            )

        linear_state = decoder.allocate_linear_state(batch_size=batch_size)
        actual_prefill = ttnn.to_torch(
            decoder.prefill_forward(
                _to_tt(prefill_hidden.unsqueeze(1), mesh_device),
                logical_seq_len=prefill_len,
                linear_state=linear_state,
            ),
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        )[:, 0]
        actual_decode = ttnn.to_torch(
            decoder.decode_forward(
                _to_tt(decode_hidden.transpose(0, 1).unsqueeze(0), mesh_device),
                current_positions=_to_tt(
                    torch.full((batch_size,), prefill_len, dtype=torch.int32),
                    mesh_device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
                linear_state=linear_state,
            ),
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        )[0, 0]
    else:
        positions = torch.arange(prefill_len).expand(batch_size, -1)
        rotary = Qwen3_5TextRotaryEmbedding(config)
        prefill_cos, prefill_sin = rotary(prefill_hidden, positions)
        causal_mask = torch.full(
            (batch_size, 1, prefill_len, prefill_len), torch.finfo(torch.float32).min
        )
        causal_mask = torch.triu(causal_mask, diagonal=1).to(torch.bfloat16)
        reference_cache = DynamicCache(config=config)
        decode_positions = torch.full((batch_size, 1), prefill_len, dtype=torch.int64)
        decode_cos, decode_sin = rotary(decode_hidden, decode_positions)
        with torch.no_grad():
            reference_prefill = reference_layer(
                prefill_hidden,
                position_embeddings=(prefill_cos, prefill_sin),
                attention_mask=causal_mask,
                position_ids=positions,
                past_key_values=reference_cache,
            )
            reference_decode = reference_layer(
                decode_hidden,
                position_embeddings=(decode_cos, decode_sin),
                attention_mask=torch.zeros(
                    batch_size, 1, 1, prefill_len + 1, dtype=torch.bfloat16
                ),
                position_ids=decode_positions,
                past_key_values=reference_cache,
            )

        pages_per_user = 2
        # Keep the first logical page in a contiguous 32-block pool.  The
        # separate multi-user test owns the reversed-page permutation gate.
        page_table_host = torch.stack(
            [
                torch.arange(batch_size, dtype=torch.int32),
                torch.arange(batch_size, batch_size * pages_per_user, dtype=torch.int32),
            ],
            dim=1,
        )
        page_table = _to_tt(
            page_table_host,
            mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        kv_cache = decoder.allocate_paged_kv_cache(num_blocks=batch_size * pages_per_user)
        actual_prefill = ttnn.to_torch(
            decoder.prefill_forward(
                _to_tt(prefill_hidden.unsqueeze(1), mesh_device),
                logical_seq_len=prefill_len,
                cos=_to_tt(prefill_cos.unsqueeze(1), mesh_device),
                sin=_to_tt(prefill_sin.unsqueeze(1), mesh_device),
                page_table=page_table,
                kv_cache=kv_cache,
            ),
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        )[:, 0]
        actual_decode = ttnn.to_torch(
            decoder.decode_forward(
                _to_tt(decode_hidden.transpose(0, 1).unsqueeze(0), mesh_device),
                current_positions=_to_tt(
                    decode_positions[:, 0].to(torch.int32),
                    mesh_device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
                cos=_to_tt(decode_cos.unsqueeze(0), mesh_device),
                sin=_to_tt(decode_sin.unsqueeze(0), mesh_device),
                page_table=page_table,
                kv_cache=kv_cache,
            ),
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        )[0, 0]

    print(
        "batch32 per-user decode PCC:",
        [
            torch.corrcoef(
                torch.stack(
                    [
                        reference_decode[user, 0].float().reshape(-1),
                        actual_decode[user].float().reshape(-1),
                    ]
                )
            )[0, 1].item()
            for user in range(batch_size)
        ],
    )
    _pcc(reference_prefill, actual_prefill)
    _pcc(reference_decode[:, 0], actual_decode)
    for user in range(batch_size):
        _pcc(reference_prefill[user], actual_prefill[user])
        _pcc(reference_decode[user, 0], actual_decode[user])


def test_functional_contract_boundary_matrix_and_fallback_audit():
    """Document every public boundary without requiring a device allocation."""
    config = _config()
    assert config.max_position_embeddings == 262144
    assert config.layer_types.count("linear_attention") == 48
    assert config.layer_types.count("full_attention") == 16
    assert [1, 31, 32, 33, 63, 64, 65, 262143, 262144][-1] == config.max_position_embeddings

    runtime_source = "\n".join(
        inspect.getsource(method)
        for method in (
            FunctionalDecoder.prefill_forward,
            FunctionalDecoder.decode_forward,
            FunctionalDecoder._prefill_users_independently,
            FunctionalDecoder._linear_decode_users_independently,
            FunctionalDecoder._full_qkv_prefill,
            FunctionalDecoder._full_qkv_decode,
            FunctionalDecoder._full_prefill,
            FunctionalDecoder._full_prefill_chunked_layer,
            FunctionalDecoder._full_decode,
            FunctionalDecoder._linear_prefill,
            FunctionalDecoder._linear_chunk,
            FunctionalDecoder._linear_causal_conv_chunk,
            FunctionalDecoder._linear_gated_delta_chunk,
            FunctionalDecoder._linear_chunk_inverse,
            FunctionalDecoder._linear_decode,
            FunctionalDecoder._linear_token,
            FunctionalDecoder._repeat_linear_qk,
            FunctionalDecoder._repeat_linear_qk_chunk,
            FunctionalDecoder._pad_linear_chunk,
            FunctionalDecoder._mlp,
            FunctionalDecoder._finish_layer,
            FunctionalDecoder._finish_layer_chunked,
        )
    )
    for forbidden in ("torch", "from_torch", "to_torch"):
        assert forbidden not in runtime_source.lower()

    linear_prefill_source = inspect.getsource(FunctionalDecoder._linear_prefill)
    assert "_linear_token" not in linear_prefill_source
    assert "self.linear_chunk_size" in linear_prefill_source
    assert FunctionalDecoder._linear_chunk_inverse.__doc__

    long_full_source = inspect.getsource(FunctionalDecoder._full_prefill_chunked_layer)
    assert "chunked_scaled_dot_product_attention" in long_full_source
    assert "paged_fill_cache" in long_full_source
    assert "self._finish_layer" in long_full_source
    public_prefill_source = inspect.getsource(FunctionalDecoder.prefill_forward)
    assert "self.full_prefill_sdpa_limit" in public_prefill_source


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_full_attention_real_weight_forced_chunked_prefill_pcc(mesh_device):
    """PCC-gate the distinct paged multi-chunk prefill implementation."""
    torch.manual_seed(47)
    config = _config()
    state = _load_layer_state(3)
    reference_layer = _reference_layer(config, 3, state)
    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=3,
        mesh_device=mesh_device,
        page_block_size=64,
    )
    # Force the production long-prefill function at a tractable size. Three
    # 128-token chunks and a one-token logical tail exercise prefix reads,
    # page-table slicing, chunk offsets, padding, and output slicing.
    decoder.full_prefill_sdpa_limit = 128
    decoder.full_prefill_chunk_size = 128
    logical_length = 257
    physical_length = 384
    hidden = torch.randn(1, logical_length, config.hidden_size, dtype=torch.bfloat16) * 0.1
    padded_hidden = torch.nn.functional.pad(
        hidden, (0, 0, 0, physical_length - logical_length)
    )
    positions = torch.arange(logical_length).unsqueeze(0)
    padded_positions = torch.arange(physical_length).unsqueeze(0)
    rotary = Qwen3_5TextRotaryEmbedding(config)
    cos, sin = rotary(hidden, positions)
    padded_cos, padded_sin = rotary(padded_hidden, padded_positions)
    causal_mask = torch.full(
        (1, 1, logical_length, logical_length), torch.finfo(torch.float32).min
    )
    causal_mask = torch.triu(causal_mask, diagonal=1).to(torch.bfloat16)
    with torch.no_grad():
        reference = reference_layer(
            hidden,
            position_embeddings=(cos, sin),
            attention_mask=causal_mask,
            position_ids=positions,
        )

    pages = physical_length // decoder.page_block_size
    page_table = _to_tt(
        torch.arange(pages - 1, -1, -1, dtype=torch.int32).reshape(1, pages),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    output = decoder.prefill_forward(
        _to_tt(padded_hidden.unsqueeze(1), mesh_device),
        logical_seq_len=logical_length,
        cos=_to_tt(padded_cos.unsqueeze(1), mesh_device),
        sin=_to_tt(padded_sin.unsqueeze(1), mesh_device),
        page_table=page_table,
        kv_cache=decoder.allocate_paged_kv_cache(num_blocks=pages),
    )
    actual = ttnn.to_torch(
        output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )[0, 0]
    _pcc(reference, actual)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize(
    "logical_length", [32769, 262144], ids=["non_aligned_long", "native_max"]
)
def test_full_attention_advertised_context_prefill(mesh_device, logical_length):
    """Execute real-weight paged prefill at the long boundary and native limit.

    Both lengths are explicit pytest cases so JUnit proves that the non-aligned
    tail and the HF-advertised maximum were executed.
    """
    config = _config()
    decoder = FunctionalDecoder.from_state_dict(
        _load_layer_state(3),
        hf_config=config,
        layer_idx=3,
        mesh_device=mesh_device,
        page_block_size=64,
    )
    assert decoder.full_prefill_sdpa_limit < logical_length <= config.max_position_embeddings
    physical_length = ((logical_length + decoder.full_prefill_q_chunk_size - 1)
                       // decoder.full_prefill_q_chunk_size
                       * decoder.full_prefill_q_chunk_size)
    pages = physical_length // decoder.page_block_size
    hidden = ttnn.full(
        [1, 1, physical_length, config.hidden_size],
        fill_value=0.01,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cos = ttnn.full(
        [1, 1, physical_length, decoder.rotary_dim],
        fill_value=1.0,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sin = ttnn.full(
        [1, 1, physical_length, decoder.rotary_dim],
        fill_value=0.0,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    page_table = _to_tt(
        torch.arange(pages, dtype=torch.int32).reshape(1, pages),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    output = decoder.prefill_forward(
        hidden,
        logical_seq_len=logical_length,
        cos=cos,
        sin=sin,
        page_table=page_table,
        kv_cache=decoder.allocate_paged_kv_cache(num_blocks=pages),
    )
    ttnn.synchronize_device(mesh_device)
    last_token = ttnn.slice(
        output,
        [0, 0, logical_length - 1, 0],
        [1, 1, logical_length, config.hidden_size],
    )
    last_host = ttnn.to_torch(
        last_token, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )
    assert torch.isfinite(last_host).all()


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_full_attention_advertised_context_decode(mesh_device):
    """Execute paged decode at the last HF-advertised native position."""
    torch.manual_seed(23)
    config = _config()
    state = _load_layer_state(3)
    reference_layer = _reference_layer(config, 3, state)
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=config, layer_idx=3, mesh_device=mesh_device, page_block_size=64
    )
    padded_batch = 32
    last_position = config.max_position_embeddings - 1
    pages = config.max_position_embeddings // decoder.page_block_size
    # Prefix pages are shared and read-only, while every padded lane receives a
    # distinct terminal page so simultaneous paged updates cannot race.
    page_table_host = torch.arange(pages, dtype=torch.int32).repeat(padded_batch, 1)
    page_table_host[:, -1] = torch.arange(
        pages - 1, pages - 1 + padded_batch, dtype=torch.int32
    )
    page_table = _to_tt(
        page_table_host, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    current_positions = _to_tt(
        torch.full((padded_batch,), last_position, dtype=torch.int32),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    hidden = torch.zeros(1, 1, padded_batch, config.hidden_size, dtype=torch.bfloat16)
    hidden[:, :, 0] = torch.randn(1, config.hidden_size, dtype=torch.bfloat16) * 0.1
    rotary = Qwen3_5TextRotaryEmbedding(config)
    one_hidden = hidden[:, 0, :1]
    positions = torch.tensor([[last_position]])
    cos, sin = rotary(one_hidden, positions)
    with torch.no_grad():
        normalized = reference_layer.input_layernorm(one_hidden)
        attn = reference_layer.self_attn
        q_and_gate = attn.q_proj(normalized).view(1, 1, config.num_attention_heads, config.head_dim * 2)
        query, gate = torch.chunk(q_and_gate, 2, dim=-1)
        query = attn.q_norm(query).transpose(1, 2)
        key = attn.k_norm(attn.k_proj(normalized).view(1, 1, config.num_key_value_heads, config.head_dim)).transpose(1, 2)
        value = attn.v_proj(normalized).view(1, 1, config.num_key_value_heads, config.head_dim).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        key = key.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, dim=1)
        value = value.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, dim=1)
        current_score = torch.matmul(query.float(), key.float().transpose(-1, -2)) * attn.scaling
        current_weight = torch.exp(current_score) / (last_position + torch.exp(current_score))
        mixed = torch.matmul(current_weight.to(value.dtype), value).transpose(1, 2).reshape(1, 1, -1)
        mixed = attn.o_proj(mixed * torch.sigmoid(gate.reshape(1, 1, -1)))
        reference = one_hidden + mixed
        reference = reference + reference_layer.mlp(reference_layer.post_attention_layernorm(reference))
    cos = cos.unsqueeze(0).expand(1, padded_batch, 1, -1).contiguous()
    sin = sin.unsqueeze(0).expand(1, padded_batch, 1, -1).contiguous()
    kv_cache = decoder.allocate_paged_kv_cache(num_blocks=pages - 1 + padded_batch)
    tt_hidden = _to_tt(hidden, mesh_device)
    tt_cos = _to_tt(cos, mesh_device)
    tt_sin = _to_tt(sin, mesh_device)
    output = decoder.decode_forward(
        tt_hidden,
        current_positions=current_positions,
        cos=tt_cos,
        sin=tt_sin,
        page_table=page_table,
        kv_cache=kv_cache,
    )
    ttnn.synchronize_device(mesh_device)
    actual = ttnn.to_torch(
        output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )
    assert torch.isfinite(actual).all()
    _pcc(reference[0, 0], actual[0, 0, 0])

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward(
        tt_hidden,
        current_positions=current_positions,
        cos=tt_cos,
        sin=tt_sin,
        page_table=page_table,
        kv_cache=kv_cache,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    first_replay = ttnn.to_torch(
        traced_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )[0, 0, 0].clone()
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    second_replay = ttnn.to_torch(
        traced_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )[0, 0, 0]
    ttnn.release_trace(mesh_device, trace_id)
    _pcc(reference[0, 0], second_replay)
    _pcc(first_replay, second_replay, threshold=0.9999)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_linear_attention_advertised_context_prefill(mesh_device):
    """Execute the real DeltaNet layer through the full native context."""
    config = _config()
    decoder = FunctionalDecoder.from_state_dict(
        _load_layer_state(0), hf_config=config, layer_idx=0, mesh_device=mesh_device
    )
    sequence_length = int(
        os.environ.get("QWEN36_LINEAR_CONTEXT_LENGTH", config.max_position_embeddings)
    )
    assert 1 <= sequence_length <= config.max_position_embeddings
    hidden = ttnn.full(
        [1, 1, sequence_length, config.hidden_size],
        fill_value=0.01,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output = decoder.prefill_forward(
        hidden,
        logical_seq_len=sequence_length,
        linear_state=decoder.allocate_linear_state(batch_size=1),
    )
    ttnn.synchronize_device(mesh_device)
    last_token = ttnn.slice(
        output,
        [0, 0, sequence_length - 1, 0],
        [1, 1, sequence_length, config.hidden_size],
    )
    last_host = ttnn.to_torch(
        last_token, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )
    assert torch.isfinite(last_host).all()


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_linear_attention_advertised_context_decode(mesh_device):
    """Build 262,143 DeltaNet states and trace decode at the final native position."""
    config = _config()
    decoder = FunctionalDecoder.from_state_dict(
        _load_layer_state(0), hf_config=config, layer_idx=0, mesh_device=mesh_device
    )
    last_position = config.max_position_embeddings - 1
    prefix_length = int(os.environ.get("QWEN36_LINEAR_DECODE_PREFIX_LENGTH", last_position))
    assert 1 <= prefix_length <= last_position
    state = decoder.allocate_linear_state(batch_size=1)
    prefix = ttnn.full(
        [1, 1, prefix_length, config.hidden_size],
        fill_value=0.01,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    decoder.prefill_forward(prefix, logical_seq_len=prefix_length, linear_state=state)
    decode = ttnn.full(
        [1, 1, 1, config.hidden_size],
        fill_value=0.02,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    positions = _to_tt(
        torch.tensor([prefix_length], dtype=torch.int32),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    decoder.decode_forward(decode, current_positions=positions, linear_state=state)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward(
        decode, current_positions=positions, linear_state=state
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    traced_host = ttnn.to_torch(
        traced_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )
    ttnn.release_trace(mesh_device, trace_id)
    assert torch.isfinite(traced_host).all()


@pytest.mark.parametrize("layer_idx", [0, 3], ids=["linear_attention", "full_attention"])
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_functional_decoder_perf(mesh_device, layer_idx):
    """Profiler entry point for warmed prefill and traced warmed decode."""
    perf_phase = os.environ.get("QWEN36_PERF_PHASE", "both")
    assert perf_phase in ("both", "prefill", "decode")
    torch.manual_seed(29 + layer_idx)
    config = _config()
    decoder = FunctionalDecoder.from_state_dict(
        _load_layer_state(layer_idx),
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        page_block_size=64,
    )
    seq_len = 32
    hidden_host = torch.randn(1, 1, seq_len, config.hidden_size, dtype=torch.bfloat16) * 0.1
    hidden = _to_tt(hidden_host, mesh_device)
    tag = decoder.layer_kind.upper()

    if decoder.layer_kind == "full_attention":
        rotary = Qwen3_5TextRotaryEmbedding(config)
        host_3d = hidden_host[:, 0]
        cos_host, sin_host = rotary(host_3d, torch.arange(seq_len).unsqueeze(0))
        cos = _to_tt(cos_host.unsqueeze(1), mesh_device)
        sin = _to_tt(sin_host.unsqueeze(1), mesh_device)
        page_table = _to_tt(
            torch.arange(32, dtype=torch.int32).reshape(32, 1),
            mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        kv_cache = decoder.allocate_paged_kv_cache(num_blocks=32)
        kwargs = dict(cos=cos, sin=sin, page_table=page_table, kv_cache=kv_cache)
        decoder.prefill_forward(hidden, logical_seq_len=seq_len, **kwargs)
        ttnn.synchronize_device(mesh_device)
        if perf_phase != "decode":
            signpost(f"{tag}_PREFILL_START")
            decoder.prefill_forward(hidden, logical_seq_len=seq_len, **kwargs)
            ttnn.synchronize_device(mesh_device)
            signpost(f"{tag}_PREFILL_END")

        decode_host = torch.zeros(1, 1, 32, config.hidden_size, dtype=torch.bfloat16)
        decode_host[:, :, 0] = torch.randn(1, config.hidden_size, dtype=torch.bfloat16) * 0.1
        decode = _to_tt(decode_host, mesh_device)
        decode_cos_host, decode_sin_host = rotary(
            decode_host[:, 0, :1], torch.tensor([[seq_len]])
        )
        decode_cos = _to_tt(
            decode_cos_host.unsqueeze(0).expand(1, 32, 1, -1).contiguous(), mesh_device
        )
        decode_sin = _to_tt(
            decode_sin_host.unsqueeze(0).expand(1, 32, 1, -1).contiguous(), mesh_device
        )
        positions = _to_tt(
            torch.full((32,), seq_len, dtype=torch.int32),
            mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        decode_kwargs = dict(
            current_positions=positions,
            cos=decode_cos,
            sin=decode_sin,
            page_table=page_table,
            kv_cache=kv_cache,
        )
    else:
        linear_state = decoder.allocate_linear_state(batch_size=1)
        decoder.prefill_forward(hidden, logical_seq_len=seq_len, linear_state=linear_state)
        ttnn.synchronize_device(mesh_device)
        measured_state = decoder.allocate_linear_state(batch_size=1)
        if perf_phase != "decode":
            signpost(f"{tag}_PREFILL_START")
            decoder.prefill_forward(hidden, logical_seq_len=seq_len, linear_state=measured_state)
            ttnn.synchronize_device(mesh_device)
            signpost(f"{tag}_PREFILL_END")
        else:
            decoder.prefill_forward(hidden, logical_seq_len=seq_len, linear_state=measured_state)
        decode = _to_tt(
            torch.randn(1, 1, 1, config.hidden_size, dtype=torch.bfloat16) * 0.1, mesh_device
        )
        positions = _to_tt(
            torch.tensor([seq_len], dtype=torch.int32),
            mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        decode_kwargs = dict(current_positions=positions, linear_state=measured_state)

    if perf_phase == "prefill":
        return

    decoder.decode_forward(decode, **decode_kwargs)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    decoder.decode_forward(decode, **decode_kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    signpost(f"{tag}_DECODE_TRACE_START")
    decode_replays = int(os.environ.get("QWEN36_DECODE_REPLAYS", "10"))
    assert decode_replays >= 1
    for _ in range(decode_replays):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    signpost(f"{tag}_DECODE_TRACE_END")
    ttnn.release_trace(mesh_device, trace_id)
