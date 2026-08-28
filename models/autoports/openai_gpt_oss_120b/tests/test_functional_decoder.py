# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time
from pathlib import Path

import pytest
import torch
from tracy import signpost
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer, GptOssRotaryEmbedding

import ttnn
from models.autoports.openai_gpt_oss_120b.tests.real_weight_utils import load_real_layer_state_dict
from models.autoports.openai_gpt_oss_120b.tt.functional_decoder import FunctionalDecoder
from models.common.utility_functions import comp_pcc
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric
from models.tt_transformers.tt.common import rope_scaling_model_factory
from models.tt_transformers.tt.rope import compute_gather_cos_sin

MODEL_CONFIG = Path(__file__).parents[3] / "demos/gpt_oss/configs/gpt-oss-120b"
REAL_WEIGHT_STATS = Path(__file__).parents[1] / "doc/functional_decoder/real_weight_stats.json"
DEFAULT_PCC_THRESHOLD = 0.995
# GPT-OSS hard top-k routing amplifies small upstream numeric differences into
# discontinuous expert changes. The model-specific full-layer acceptance bars
# below are stricter than the canonical repo gates (0.86 prefill / 0.90 decode)
# and are backed by the counterfactual evidence in doc/functional_decoder.
PREFILL_PCC_THRESHOLD = 0.95
DECODE_PCC_THRESHOLD = 0.95
# Hard-routing changes occur independently for each user.  Keep the primary
# batch-one/real-weight decode gate at 0.95. The separate seeded-random
# batch-two routing, position, and page-table stress still passes a 0.99 bar.
BATCH_DECODE_PCC_THRESHOLD = 0.99
PAGE_SIZE = 64
REAL_WEIGHT_SNAPSHOT = os.environ.get("GPT_OSS_120B_SNAPSHOT")


def _config():
    config = AutoConfig.from_pretrained(MODEL_CONFIG)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    return config


def _reference_layer(config, layer_idx, state_dict=None):
    if state_dict is not None:
        # A real dense GPT-OSS 120B expert layer is roughly 8.5 GiB in BF16.
        # Constructing on meta and assigning the loaded tensors prevents a
        # second equally large random initialization from existing at once.
        with torch.device("meta"):
            layer = GptOssDecoderLayer(config, layer_idx=layer_idx)
        layer.load_state_dict(state_dict, assign=True)
        return layer.eval()
    layer = GptOssDecoderLayer(config, layer_idx=layer_idx).eval()
    with REAL_WEIGHT_STATS.open(encoding="utf-8") as stats_file:
        layer_stats = json.load(stats_file)["layers"][str(layer_idx)]
    generator = torch.Generator().manual_seed(9472 + layer_idx)
    with torch.no_grad():
        for name, parameter in layer.named_parameters():
            stats = layer_stats[name]
            assert list(parameter.shape) == stats["shape"]
            # Exact trained moments without trained cross-tensor correlations
            # create an ill-conditioned random decoder (for example, expert
            # bias stddevs exceed 0.2). Preserve the checkpoint-derived mean
            # and scale while capping independent noise at the HF initializer
            # range; the real-weight gate separately exercises exact tensors.
            synthetic_std = min(stats["std"], config.initializer_range)
            if synthetic_std == 0.0:
                parameter.fill_(stats["mean"])
            else:
                parameter.normal_(mean=stats["mean"], std=synthetic_std, generator=generator)
    return layer


def _rope_tensors(config, mesh_device, positions, *, decode):
    rope_scaling = rope_scaling_model_factory(config.rope_scaling)
    theta = getattr(config, "rope_theta", None) or getattr(config, "default_theta", 150000.0)
    cos_cache, sin_cache = compute_gather_cos_sin(
        dhead=config.head_dim,
        end=2 * config.max_position_embeddings,
        theta=theta,
        rope_scaling=rope_scaling,
    )
    if decode:
        cos = cos_cache[:, :, positions.tolist(), :].permute(0, 2, 1, 3)
        sin = sin_cache[:, :, positions.tolist(), :].permute(0, 2, 1, 3)
    else:
        cos = cos_cache[:, :, : positions.numel(), :]
        sin = sin_cache[:, :, : positions.numel(), :]
    tt_cos = ttnn.from_torch(cos, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_sin = ttnn.from_torch(sin, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    if decode:
        grid = ttnn.num_cores_to_corerangeset(positions.numel(), ttnn.CoreCoord(8, 8), row_wise=True)
        memory_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, config.head_dim),
            core_grid=grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        tt_cos = ttnn.interleaved_to_sharded(tt_cos, memory_config)
        tt_sin = ttnn.interleaved_to_sharded(tt_sin, memory_config)
    return [tt_cos, tt_sin]


def _hf_position_embeddings(config, hidden_states, positions):
    return GptOssRotaryEmbedding(config)(hidden_states, positions.unsqueeze(0))


def _prefill_mask(config, layer_idx, sequence_length):
    mask = torch.triu(torch.full((1, 1, sequence_length, sequence_length), -float("inf")), diagonal=1)
    if config.layer_types[layer_idx] == "sliding_attention":
        mask += torch.tril(
            torch.full((1, 1, sequence_length, sequence_length), -float("inf")),
            diagonal=-config.sliding_window,
        )
    return mask


def _decode_mask(config, layer_idx, current_position):
    if config.layer_types[layer_idx] == "full_attention":
        return None
    total = current_position + 1
    mask = torch.zeros((1, 1, 1, total))
    first_visible = max(0, total - config.sliding_window)
    mask[..., :first_visible] = -float("inf")
    return mask


def _page_table(mesh_device, max_context, *, batch_size=1, seed=13):
    blocks_per_user = (max_context + PAGE_SIZE - 1) // PAGE_SIZE
    blocks = batch_size * blocks_per_user
    generator = torch.Generator().manual_seed(seed)
    host = (
        torch.randperm(blocks, generator=generator, dtype=torch.int64)
        .to(torch.int32)
        .reshape(batch_size, blocks_per_user)
    )
    return ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _to_host(tensor):
    return ttnn.to_torch(tensor)[0, 0]


def _to_host_batch(tensor):
    return ttnn.to_torch(tensor)[0]


def _assert_pcc(actual, expected, label, threshold=DEFAULT_PCC_THRESHOLD):
    passing, detail = comp_pcc(expected.float(), actual.float(), threshold)
    assert passing, f"{label} failed: {detail}"
    return detail


@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_paged_prefill_and_traced_decode(mesh_device, device_params, layer_idx, reset_seeds):
    """Real-shape synthetic gate for both meaningful layer kinds.

    The decode PCC is read after trace replay. The permuted page table and
    nonzero current position make cache-addressing mistakes observable.
    """
    config = _config()
    reference = _reference_layer(config, layer_idx)
    decoder = FunctionalDecoder.from_state_dict(
        reference.state_dict(),
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch_size=1,
        max_context_length=config.max_position_embeddings,
        page_size=PAGE_SIZE,
    )
    page_table = _page_table(mesh_device, config.max_position_embeddings, seed=31 + layer_idx)
    cache = DynamicCache()

    sequence_length = 129
    positions = torch.arange(sequence_length, dtype=torch.long)
    hidden = torch.randn(1, sequence_length, config.hidden_size) * 0.02
    hf_rope = _hf_position_embeddings(config, hidden, positions)
    with torch.no_grad():
        expected_prefill = reference(
            hidden,
            attention_mask=_prefill_mask(config, layer_idx, sequence_length),
            position_embeddings=hf_rope,
            position_ids=positions.unsqueeze(0),
            past_key_values=cache,
            use_cache=True,
        )
    tt_hidden = ttnn.from_torch(
        hidden.reshape(1, 1, sequence_length, config.hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_prefill = decoder.prefill_forward(
        tt_hidden,
        position_embeddings=_rope_tensors(config, mesh_device, positions, decode=False),
        page_table=page_table,
    )
    ttnn.synchronize_device(mesh_device)
    prefill_detail = _assert_pcc(
        _to_host(tt_prefill)[:sequence_length],
        expected_prefill[0],
        "prefill",
        threshold=PREFILL_PCC_THRESHOLD,
    )
    print(f"SYNTHETIC_PREFILL layer={layer_idx} type={config.layer_types[layer_idx]} {prefill_detail}")

    current = sequence_length
    decode_positions = torch.tensor([current], dtype=torch.long)
    decode_hidden = torch.randn(1, 1, config.hidden_size) * 0.02
    hf_decode_rope = _hf_position_embeddings(config, decode_hidden, decode_positions)
    with torch.no_grad():
        expected_decode = reference(
            decode_hidden,
            attention_mask=_decode_mask(config, layer_idx, current),
            position_embeddings=hf_decode_rope,
            position_ids=decode_positions.unsqueeze(0),
            past_key_values=cache,
            use_cache=True,
        )
    tt_decode_hidden = ttnn.from_torch(
        decode_hidden.reshape(1, 1, 1, config.hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_current = ttnn.from_torch(
        decode_positions.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    decode_rope = _rope_tensors(config, mesh_device, decode_positions, decode=True)

    # Compile once, then capture the complete decoder and compare replay output.
    _ = decoder.decode_forward(
        tt_decode_hidden,
        position_embeddings=decode_rope,
        current_position=tt_current,
        page_table=page_table,
    )
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    tt_decode = decoder.decode_forward(
        tt_decode_hidden,
        position_embeddings=decode_rope,
        current_position=tt_current,
        page_table=page_table,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    first = _to_host(tt_decode)[:1].clone()
    decode_detail = _assert_pcc(first, expected_decode[0], "traced decode", threshold=DECODE_PCC_THRESHOLD)
    print(f"SYNTHETIC_DECODE layer={layer_idx} type={config.layer_types[layer_idx]} {decode_detail}")
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    second = _to_host(tt_decode)[:1]
    assert torch.equal(second, first), "repeated traced decode was not bitwise deterministic"
    ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_PROFILE") != "1",
    reason="set GPT_OSS_120B_PROFILE=1 for warmed Tracy signpost collection",
)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_warmed_prefill_and_traced_decode_performance(mesh_device, device_params, layer_idx, reset_seeds):
    """Collect separate warmed prefill and traced-decode signpost windows."""
    config = _config()
    reference = _reference_layer(config, layer_idx)
    decoder = FunctionalDecoder.from_state_dict(
        reference.state_dict(),
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch_size=1,
        max_context_length=config.max_position_embeddings,
        page_size=PAGE_SIZE,
    )
    page_table = _page_table(mesh_device, config.max_position_embeddings, seed=53 + layer_idx)
    sequence_length = 128
    positions = torch.arange(sequence_length, dtype=torch.long)
    hidden = torch.randn(1, sequence_length, config.hidden_size) * 0.02
    tt_hidden = ttnn.from_torch(
        hidden.reshape(1, 1, sequence_length, config.hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    prefill_rope = _rope_tensors(config, mesh_device, positions, decode=False)

    warm_prefill = decoder.prefill_forward(tt_hidden, position_embeddings=prefill_rope, page_table=page_table)
    ttnn.synchronize_device(mesh_device)
    warm_prefill.deallocate(True)
    signpost("PERF_PREFILL")
    started = time.perf_counter()
    measured_prefill = decoder.prefill_forward(tt_hidden, position_embeddings=prefill_rope, page_table=page_table)
    ttnn.synchronize_device(mesh_device)
    prefill_seconds = time.perf_counter() - started
    signpost("PERF_PREFILL_END")

    decode_positions = torch.tensor([sequence_length], dtype=torch.long)
    decode_hidden = torch.randn(1, 1, config.hidden_size) * 0.02
    tt_decode_hidden = ttnn.from_torch(
        decode_hidden.reshape(1, 1, 1, config.hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_current = ttnn.from_torch(
        decode_positions.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    decode_rope = _rope_tensors(config, mesh_device, decode_positions, decode=True)
    _ = decoder.decode_forward(
        tt_decode_hidden,
        position_embeddings=decode_rope,
        current_position=tt_current,
        page_table=page_table,
    )
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward(
        tt_decode_hidden,
        position_embeddings=decode_rope,
        current_position=tt_current,
        page_table=page_table,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)

    signpost("PERF_DECODE")
    started = time.perf_counter()
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    decode_seconds = time.perf_counter() - started
    signpost("PERF_DECODE_END")
    assert torch.isfinite(_to_host(traced_output)).all()
    ttnn.release_trace(mesh_device, trace_id)
    print(
        f"WARMED_PERF layer={layer_idx} type={config.layer_types[layer_idx]} sequence={sequence_length} "
        f"prefill_wall_seconds={prefill_seconds:.9f} traced_decode_wall_seconds={decode_seconds:.9f}"
    )
    measured_prefill.deallocate(True)


@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_batch_two_paged_prefill_and_traced_decode(mesh_device, device_params, layer_idx, reset_seeds):
    """Batch-two gate with disjoint physical pages and per-user active experts."""
    config = _config()
    reference = _reference_layer(config, layer_idx)
    batch_size = 2
    decoder = FunctionalDecoder.from_state_dict(
        reference.state_dict(),
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch_size=batch_size,
        max_context_length=config.max_position_embeddings,
        page_size=PAGE_SIZE,
    )
    page_table = _page_table(
        mesh_device,
        config.max_position_embeddings,
        batch_size=batch_size,
        seed=71 + layer_idx,
    )
    cache = DynamicCache()

    sequence_length = 33
    positions = torch.arange(sequence_length, dtype=torch.long)
    position_ids = positions.unsqueeze(0).expand(batch_size, -1)
    hidden = torch.randn(batch_size, sequence_length, config.hidden_size) * 0.02
    hf_rope = GptOssRotaryEmbedding(config)(hidden, position_ids)
    with torch.no_grad():
        expected_prefill = reference(
            hidden,
            attention_mask=_prefill_mask(config, layer_idx, sequence_length),
            position_embeddings=hf_rope,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
    tt_hidden = ttnn.from_torch(
        hidden.reshape(1, batch_size, sequence_length, config.hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_prefill = decoder.prefill_forward(
        tt_hidden,
        position_embeddings=_rope_tensors(config, mesh_device, positions, decode=False),
        page_table=page_table,
        batch_size=batch_size,
    )
    ttnn.synchronize_device(mesh_device)
    prefill_detail = _assert_pcc(
        _to_host_batch(tt_prefill), expected_prefill, "batch-two prefill", threshold=PREFILL_PCC_THRESHOLD
    )
    print(f"BATCH_TWO_PREFILL layer={layer_idx} type={config.layer_types[layer_idx]} {prefill_detail}")

    # Use independent current positions so neither paged cache update nor SDPA
    # can accidentally broadcast one user's scalar position across the batch.
    position_generator = torch.Generator().manual_seed(211 + layer_idx)
    decode_positions = (torch.randperm(sequence_length - 1, generator=position_generator) + 2)[:batch_size]
    assert decode_positions.unique().numel() == batch_size
    decode_position_ids = decode_positions.unsqueeze(1)
    decode_hidden = torch.randn(batch_size, 1, config.hidden_size) * 0.02
    expected_decode = []
    with torch.no_grad():
        for user, current_position in enumerate(decode_positions.tolist()):
            # Rebuild an exact HF prefix per user. The TT cache was filled to
            # sequence_length for both users, but SDPA must ignore entries past
            # each user's current position; the shorter HF prefix is the
            # semantic reference for that behavior.
            user_cache = DynamicCache()
            user_positions = torch.arange(current_position, dtype=torch.long)
            user_hidden = hidden[user : user + 1, :current_position]
            user_rope = _hf_position_embeddings(config, user_hidden, user_positions)
            reference(
                user_hidden,
                attention_mask=_prefill_mask(config, layer_idx, current_position),
                position_embeddings=user_rope,
                position_ids=user_positions.unsqueeze(0),
                past_key_values=user_cache,
                use_cache=True,
            )
            user_decode_position_ids = decode_position_ids[user : user + 1]
            user_decode_hidden = decode_hidden[user : user + 1]
            user_decode_rope = GptOssRotaryEmbedding(config)(user_decode_hidden, user_decode_position_ids)
            user_expected = reference(
                user_decode_hidden,
                attention_mask=_decode_mask(config, layer_idx, current_position),
                position_embeddings=user_decode_rope,
                position_ids=user_decode_position_ids,
                past_key_values=user_cache,
                use_cache=True,
            )
            expected_decode.append(user_expected[0, 0])
    expected_decode = torch.stack(expected_decode)
    tt_decode_hidden = ttnn.from_torch(
        decode_hidden.reshape(1, 1, batch_size, config.hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_current = ttnn.from_torch(
        decode_positions.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    decode_rope = _rope_tensors(config, mesh_device, decode_positions, decode=True)

    _ = decoder.decode_forward(
        tt_decode_hidden,
        position_embeddings=decode_rope,
        current_position=tt_current,
        page_table=page_table,
        batch_size=batch_size,
    )
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    tt_decode = decoder.decode_forward(
        tt_decode_hidden,
        position_embeddings=decode_rope,
        current_position=tt_current,
        page_table=page_table,
        batch_size=batch_size,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    decode_detail = _assert_pcc(
        _to_host(tt_decode)[:batch_size],
        expected_decode,
        "batch-two traced decode",
        threshold=BATCH_DECODE_PCC_THRESHOLD,
    )
    print(f"BATCH_TWO_DECODE layer={layer_idx} type={config.layer_types[layer_idx]} {decode_detail}")
    ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_RUN_BATCH32") != "1",
    reason="set GPT_OSS_120B_RUN_BATCH32=1 for the full-context batch-32 capacity gate",
)
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_batch_32_paged_prefill_and_traced_decode_capacity(mesh_device, device_params, layer_idx, reset_seeds):
    """Prove batch-32 paged-cache allocation and complete trace replay.

    HF-vs-TTNN batch correctness is covered at batch 2. Running the 128-expert
    eager HF reference at batch 32 is prohibitively expensive on the host, so
    this opt-in hardware-capacity gate checks the largest intended functional
    batch with the full 131072-token-per-user page-table/cache allocation.
    """
    config = _config()
    reference = _reference_layer(config, layer_idx)
    batch_size = 32
    decoder = FunctionalDecoder.from_state_dict(
        reference.state_dict(),
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch_size=batch_size,
        max_context_length=config.max_position_embeddings,
        page_size=PAGE_SIZE,
    )
    page_table = _page_table(
        mesh_device,
        config.max_position_embeddings,
        batch_size=batch_size,
        seed=97 + layer_idx,
    )
    sequence_length = 33
    positions = torch.arange(sequence_length, dtype=torch.long)
    hidden = torch.randn(batch_size, sequence_length, config.hidden_size, dtype=torch.bfloat16) * 0.02
    tt_hidden = ttnn.from_torch(
        hidden.reshape(1, batch_size, sequence_length, config.hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    prefill = decoder.prefill_forward(
        tt_hidden,
        position_embeddings=_rope_tensors(config, mesh_device, positions, decode=False),
        page_table=page_table,
        batch_size=batch_size,
    )
    ttnn.synchronize_device(mesh_device)
    assert tuple(prefill.shape) == (1, batch_size, sequence_length, config.hidden_size)
    assert torch.isfinite(ttnn.to_torch(prefill)[0, :, -1]).all()

    position_generator = torch.Generator().manual_seed(307 + layer_idx)
    decode_positions = (torch.randperm(sequence_length, generator=position_generator) + 1)[:batch_size]
    assert decode_positions.unique().numel() == batch_size
    decode_hidden = torch.randn(batch_size, 1, config.hidden_size, dtype=torch.bfloat16) * 0.02
    tt_decode_hidden = ttnn.from_torch(
        decode_hidden.reshape(1, 1, batch_size, config.hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_current = ttnn.from_torch(
        decode_positions.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    decode_rope = _rope_tensors(config, mesh_device, decode_positions, decode=True)
    _ = decoder.decode_forward(
        tt_decode_hidden,
        position_embeddings=decode_rope,
        current_position=tt_current,
        page_table=page_table,
        batch_size=batch_size,
    )
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = decoder.decode_forward(
        tt_decode_hidden,
        position_embeddings=decode_rope,
        current_position=tt_current,
        page_table=page_table,
        batch_size=batch_size,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    first = _to_host(output)[:batch_size].clone()
    assert torch.isfinite(first).all()
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    assert torch.equal(_to_host(output)[:batch_size], first)
    ttnn.release_trace(mesh_device, trace_id)
    print(
        f"BATCH32_CAPACITY layer={layer_idx} type={config.layer_types[layer_idx]} "
        f"prefill_sequence={sequence_length} context_per_user={config.max_position_embeddings} "
        f"position_min={decode_positions.min().item()} position_max={decode_positions.max().item()}"
    )


@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@pytest.mark.skipif(
    not REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_SNAPSHOT to run the real-weight acceptance gate",
)
@parametrize_mesh_with_fabric([(1, 1)])
def test_real_weight_paged_prefill_and_traced_decode(mesh_device, device_params, layer_idx, reset_seeds):
    """Required real-checkpoint gate for both GPT-OSS decoder layer kinds.

    This test is opt-in because loading and dequantizing one 120B expert layer
    uses several GiB of host memory. Set ``GPT_OSS_120B_SNAPSHOT`` to the exact
    Hugging Face snapshot directory containing the index and layer shards.
    """
    config = _config()
    started = time.perf_counter()
    state_dict = load_real_layer_state_dict(REAL_WEIGHT_SNAPSHOT, layer_idx)
    dequant_seconds = time.perf_counter() - started
    print(f"REAL_WEIGHT_DEQUANT layer={layer_idx} seconds={dequant_seconds:.3f}")
    reference = _reference_layer(config, layer_idx, state_dict=state_dict)
    cache_root = (
        Path(os.environ.get("GPT_OSS_120B_TENSOR_CACHE", "/tmp/gpt_oss_120b_functional_decoder_tensor_cache"))
        / f"layer_{layer_idx}"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    decoder = FunctionalDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch_size=1,
        max_context_length=config.max_position_embeddings,
        page_size=PAGE_SIZE,
        tensor_cache_path=cache_root,
    )
    page_table = _page_table(mesh_device, config.max_position_embeddings, seed=113 + layer_idx)
    cache = DynamicCache()

    sequence_length = 129
    positions = torch.arange(sequence_length, dtype=torch.long)
    generator = torch.Generator().manual_seed(120_000 + layer_idx)
    hidden = (torch.randn((1, sequence_length, config.hidden_size), generator=generator) * 0.02).to(torch.bfloat16)
    hf_rope = _hf_position_embeddings(config, hidden, positions)
    with torch.no_grad():
        expected_prefill = reference(
            hidden,
            attention_mask=_prefill_mask(config, layer_idx, sequence_length),
            position_embeddings=hf_rope,
            position_ids=positions.unsqueeze(0),
            past_key_values=cache,
            use_cache=True,
        )
    tt_hidden = ttnn.from_torch(
        hidden.reshape(1, 1, sequence_length, config.hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_prefill = decoder.prefill_forward(
        tt_hidden,
        position_embeddings=_rope_tensors(config, mesh_device, positions, decode=False),
        page_table=page_table,
    )
    ttnn.synchronize_device(mesh_device)
    prefill_detail = _assert_pcc(
        _to_host(tt_prefill)[:sequence_length],
        expected_prefill[0],
        f"real-weight layer {layer_idx} prefill",
        threshold=PREFILL_PCC_THRESHOLD,
    )
    print(f"REAL_WEIGHT_PREFILL layer={layer_idx} type={config.layer_types[layer_idx]} {prefill_detail}")

    current = sequence_length
    decode_positions = torch.tensor([current], dtype=torch.long)
    decode_hidden = (torch.randn((1, 1, config.hidden_size), generator=generator) * 0.02).to(torch.bfloat16)
    hf_decode_rope = _hf_position_embeddings(config, decode_hidden, decode_positions)
    with torch.no_grad():
        expected_decode = reference(
            decode_hidden,
            attention_mask=_decode_mask(config, layer_idx, current),
            position_embeddings=hf_decode_rope,
            position_ids=decode_positions.unsqueeze(0),
            past_key_values=cache,
            use_cache=True,
        )
    tt_decode_hidden = ttnn.from_torch(
        decode_hidden.reshape(1, 1, 1, config.hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_current = ttnn.from_torch(
        decode_positions.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    decode_rope = _rope_tensors(config, mesh_device, decode_positions, decode=True)

    _ = decoder.decode_forward(
        tt_decode_hidden,
        position_embeddings=decode_rope,
        current_position=tt_current,
        page_table=page_table,
    )
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    tt_decode = decoder.decode_forward(
        tt_decode_hidden,
        position_embeddings=decode_rope,
        current_position=tt_current,
        page_table=page_table,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    first = _to_host(tt_decode)[:1].clone()
    decode_detail = _assert_pcc(
        first,
        expected_decode[0],
        f"real-weight layer {layer_idx} traced decode",
        threshold=DECODE_PCC_THRESHOLD,
    )
    print(f"REAL_WEIGHT_DECODE layer={layer_idx} type={config.layer_types[layer_idx]} {decode_detail}")
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    second = _to_host(tt_decode)[:1]
    assert torch.equal(second, first), f"real-weight layer {layer_idx} traced decode was not bitwise deterministic"
    ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_prefill_tile_page_and_window_boundaries(mesh_device, device_params, layer_idx, reset_seeds):
    """Execute real-shape prefill at length one and around tile/page/window boundaries."""
    config = _config()
    reference = _reference_layer(config, layer_idx)
    decoder = FunctionalDecoder.from_state_dict(
        reference.state_dict(),
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch_size=1,
        max_context_length=config.max_position_embeddings,
        page_size=PAGE_SIZE,
    )
    page_table = _page_table(mesh_device, config.max_position_embeddings, seed=173 + layer_idx)
    generator = torch.Generator().manual_seed(131_000 + layer_idx)

    for sequence_length in (1, 31, 32, 33, 63, 64, 65, 127, 128, 129):
        positions = torch.arange(sequence_length, dtype=torch.long)
        hidden = torch.randn((1, sequence_length, config.hidden_size), generator=generator) * 0.02
        with torch.no_grad():
            expected = reference(
                hidden,
                attention_mask=_prefill_mask(config, layer_idx, sequence_length),
                position_embeddings=_hf_position_embeddings(config, hidden, positions),
                position_ids=positions.unsqueeze(0),
                past_key_values=DynamicCache(),
                use_cache=True,
            )
        tt_hidden = ttnn.from_torch(
            hidden.reshape(1, 1, sequence_length, config.hidden_size),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        input_shard = ttnn.get_device_tensors(tt_hidden)[0]
        input_address = input_shard.buffer_address()
        rope = _rope_tensors(config, mesh_device, positions, decode=False)
        actual = decoder.prefill_forward(tt_hidden, position_embeddings=rope, page_table=page_table)
        ttnn.synchronize_device(mesh_device)

        assert input_shard.is_allocated(), f"public input was consumed at sequence_length={sequence_length}"
        assert input_shard.buffer_address() == input_address
        assert tuple(actual.shape) == (1, 1, sequence_length, config.hidden_size)
        detail = _assert_pcc(
            _to_host(actual),
            expected[0],
            f"boundary prefill sequence_length={sequence_length}",
            threshold=PREFILL_PCC_THRESHOLD,
        )
        print(
            f"BOUNDARY_PREFILL layer={layer_idx} type={config.layer_types[layer_idx]} "
            f"sequence={sequence_length} {detail}"
        )
        actual.deallocate(True)
        tt_hidden.deallocate(True)
        for rope_tensor in rope:
            rope_tensor.deallocate(True)


@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_traced_decode_at_advertised_context_limit(mesh_device, device_params, layer_idx, reset_seeds):
    """Trace decode at position 131071 to prove full page-table/cache addressability."""
    config = _config()
    reference = _reference_layer(config, layer_idx)
    decoder = FunctionalDecoder.from_state_dict(
        reference.state_dict(),
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch_size=1,
        max_context_length=config.max_position_embeddings,
        page_size=PAGE_SIZE,
    )
    page_table = _page_table(mesh_device, config.max_position_embeddings, seed=211 + layer_idx)
    current = config.max_position_embeddings - 1
    positions = torch.tensor([current], dtype=torch.long)
    hidden = torch.randn(1, 1, config.hidden_size) * 0.02
    tt_hidden = ttnn.from_torch(
        hidden.reshape(1, 1, 1, config.hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    input_shard = ttnn.get_device_tensors(tt_hidden)[0]
    input_address = input_shard.buffer_address()
    tt_current = ttnn.from_torch(
        positions.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    decode_rope = _rope_tensors(config, mesh_device, positions, decode=True)

    _ = decoder.decode_forward(
        tt_hidden,
        position_embeddings=decode_rope,
        current_position=tt_current,
        page_table=page_table,
    )
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = decoder.decode_forward(
        tt_hidden,
        position_embeddings=decode_rope,
        current_position=tt_current,
        page_table=page_table,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    first = _to_host(output)[:1].clone()
    assert torch.isfinite(first).all()
    assert input_shard.is_allocated()
    assert input_shard.buffer_address() == input_address
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    assert torch.equal(_to_host(output)[:1], first), "advertised-context traced decode was not bitwise deterministic"
    ttnn.release_trace(mesh_device, trace_id)
    print(
        f"MAX_CONTEXT_DECODE layer={layer_idx} type={config.layer_types[layer_idx]} "
        f"current_position={current} context={config.max_position_embeddings}"
    )


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_RUN_CHUNK_BOUNDARIES") != "1",
    reason="set GPT_OSS_120B_RUN_CHUNK_BOUNDARIES=1 for the expensive 4095/4096/4097-token gate",
)
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_prefill_chunk_boundaries(mesh_device, device_params, layer_idx, reset_seeds):
    """Execute the full decoder around the canonical 4096-token expert chunk."""
    config = _config()
    reference = _reference_layer(config, layer_idx)
    decoder = FunctionalDecoder.from_state_dict(
        reference.state_dict(),
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch_size=1,
        max_context_length=config.max_position_embeddings,
        page_size=PAGE_SIZE,
    )
    page_table = _page_table(mesh_device, config.max_position_embeddings, seed=251 + layer_idx)
    generator = torch.Generator().manual_seed(409_600 + layer_idx)
    for sequence_length in (4095, 4096, 4097):
        positions = torch.arange(sequence_length, dtype=torch.long)
        hidden = torch.randn((1, sequence_length, config.hidden_size), generator=generator) * 0.02
        tt_hidden = ttnn.from_torch(
            hidden.reshape(1, 1, sequence_length, config.hidden_size),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        rope = _rope_tensors(config, mesh_device, positions, decode=False)
        output = decoder.prefill_forward(tt_hidden, position_embeddings=rope, page_table=page_table)
        ttnn.synchronize_device(mesh_device)
        host_sample = _to_host(output)[-1]
        assert tuple(output.shape) == (1, 1, sequence_length, config.hidden_size)
        assert torch.isfinite(host_sample).all()
        print(f"CHUNK_BOUNDARY_PREFILL layer={layer_idx} sequence={sequence_length}")
        output.deallocate(True)
        tt_hidden.deallocate(True)
        for rope_tensor in rope:
            rope_tensor.deallocate(True)


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_RUN_MAX_PREFILL") != "1",
    reason="set GPT_OSS_120B_RUN_MAX_PREFILL=1 for the expensive 131071/131072-token gate",
)
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_prefill_at_advertised_context_limit(mesh_device, device_params, layer_idx, reset_seeds):
    """Run the full decoder at the non-aligned near-limit and advertised limit."""
    config = _config()
    reference = _reference_layer(config, layer_idx)
    decoder = FunctionalDecoder.from_state_dict(
        reference.state_dict(),
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch_size=1,
        max_context_length=config.max_position_embeddings,
        page_size=PAGE_SIZE,
    )
    page_table = _page_table(mesh_device, config.max_position_embeddings, seed=307 + layer_idx)
    generator = torch.Generator().manual_seed(13_107_200 + layer_idx)
    for sequence_length in (config.max_position_embeddings - 1, config.max_position_embeddings):
        positions = torch.arange(sequence_length, dtype=torch.long)
        hidden = torch.randn((1, sequence_length, config.hidden_size), generator=generator, dtype=torch.bfloat16) * 0.02
        tt_hidden = ttnn.from_torch(
            hidden.reshape(1, 1, sequence_length, config.hidden_size),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        input_shard = ttnn.get_device_tensors(tt_hidden)[0]
        input_address = input_shard.buffer_address()
        rope = _rope_tensors(config, mesh_device, positions, decode=False)
        started = time.perf_counter()
        output = decoder.prefill_forward(tt_hidden, position_embeddings=rope, page_table=page_table)
        ttnn.synchronize_device(mesh_device)
        elapsed = time.perf_counter() - started
        last_token = ttnn.slice(
            output,
            starts=[0, 0, sequence_length - 1, 0],
            ends=[1, 1, sequence_length, config.hidden_size],
            steps=[1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        assert torch.isfinite(ttnn.to_torch(last_token)).all()
        assert tuple(output.shape) == (1, 1, sequence_length, config.hidden_size)
        assert input_shard.is_allocated()
        assert input_shard.buffer_address() == input_address
        print(
            f"MAX_CONTEXT_PREFILL layer={layer_idx} type={config.layer_types[layer_idx]} "
            f"sequence={sequence_length} seconds={elapsed:.6f}"
        )
        last_token.deallocate(True)
        output.deallocate(True)
        tt_hidden.deallocate(True)
        for rope_tensor in rope:
            rope_tensor.deallocate(True)
