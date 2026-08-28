# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Opt-in counterfactual evidence for GPT-OSS hard-routing precision.

This analysis intentionally crosses the device/host boundary in the test.  It
does not change the production decoder contract.  Starting from the *actual*
TTNN post-attention and post-norm decode states, it compares:

* the natural HF and TT top-k routes;
* HF experts under the natural HF routes and the fixed TT routes/weights; and
* TT and HF expert tails under the same fixed TT routes/weights.

The fixed-route comparison must clear the normal functional-decoder PCC bar.
That counterfactual distinguishes continuous expert-kernel error from the
discontinuous change caused by a top-k route crossing.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
from transformers.cache_utils import DynamicCache

import ttnn
from models.autoports.openai_gpt_oss_120b.tests.real_weight_utils import load_real_layer_state_dict
from models.autoports.openai_gpt_oss_120b.tests.test_functional_decoder import (
    DEFAULT_PCC_THRESHOLD,
    PAGE_SIZE,
    _config,
    _decode_mask,
    _hf_position_embeddings,
    _page_table,
    _prefill_mask,
    _reference_layer,
    _rope_tensors,
    _to_host,
)
from models.autoports.openai_gpt_oss_120b.tt.functional_decoder import FunctionalDecoder
from models.common.utility_functions import comp_pcc
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric

RUN_ANALYSIS = os.environ.get("GPT_OSS_120B_ROUTING_ANALYSIS") == "1"
REAL_WEIGHT_SNAPSHOT = os.environ.get("GPT_OSS_120B_SNAPSHOT")
ROUTING_CANDIDATES = 256


def _pcc(expected: torch.Tensor, actual: torch.Tensor) -> float:
    """Return PCC without imposing an acceptance threshold."""
    _, value = comp_pcc(expected.float(), actual.float(), pcc=-1.0)
    return float(value)


def _select_low_margin_decode_inputs(reference, config, layer_idx: int, batch_size: int):
    """Choose deterministic probes near the HF fourth/fifth-router-logit boundary.

    Attention is residual-dominated at this stage, so post-normalizing raw
    candidate residuals is a cheap, reproducible proxy for finding route-boundary
    inputs.  The evidence itself always uses the later, actual TTNN post-attention
    and post-norm tensors.
    """
    generator = torch.Generator().manual_seed(240_000 + layer_idx)
    candidates = (torch.randn((ROUTING_CANDIDATES, 1, config.hidden_size), generator=generator) * 0.02).to(
        torch.bfloat16
    )
    with torch.no_grad():
        candidate_states = reference.post_attention_layernorm(candidates)[:, 0]
        logits = reference.mlp.router(candidate_states)[0]
        top_values = torch.topk(logits.float(), config.num_experts_per_tok + 1, dim=-1).values
        margins = top_values[:, config.num_experts_per_tok - 1] - top_values[:, config.num_experts_per_tok]
        selected = torch.topk(margins, batch_size, largest=False).indices
    return candidates[selected], selected, margins[selected]


def _hf_decode_intermediates(reference, config, layer_idx, prefix, decode_hidden, current_position):
    """Run one natural HF decode while retaining its two pre-MoE states."""
    cache = DynamicCache()
    prefix_positions = torch.arange(current_position, dtype=torch.long)
    with torch.no_grad():
        reference(
            prefix,
            attention_mask=_prefill_mask(config, layer_idx, current_position),
            position_embeddings=_hf_position_embeddings(config, prefix, prefix_positions),
            position_ids=prefix_positions.unsqueeze(0),
            past_key_values=cache,
            use_cache=True,
        )

        position = torch.tensor([current_position], dtype=torch.long)
        normed = reference.input_layernorm(decode_hidden)
        attention_out = reference.self_attn(
            hidden_states=normed,
            attention_mask=_decode_mask(config, layer_idx, current_position),
            position_embeddings=_hf_position_embeddings(config, decode_hidden, position),
            position_ids=position.unsqueeze(0),
            past_key_values=cache,
            use_cache=True,
        )[0]
        post_attention = decode_hidden + attention_out
        post_norm = reference.post_attention_layernorm(post_attention)
        mlp_out = reference.mlp(post_norm)[0]
        output = post_attention + mlp_out
    return output[0, 0], post_attention[0, 0], post_norm[0, 0]


def _tt_decode_intermediates(decoder, tt_hidden, decode_rope, tt_current, page_table, batch_size):
    """Re-run the device decode prefix and retain actual TT pre-MoE tensors."""
    residual = ttnn.clone(tt_hidden, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    normed = decoder.input_layernorm(residual)
    attention_out = decoder.self_attn(
        normed,
        rope_mats=decode_rope,
        position_idx=tt_current,
        page_table=page_table,
        kv_cache=decoder.kv_cache,
        is_decode=True,
        user_id=0,
        batch_size=batch_size,
    )
    normed.deallocate(True)
    post_attention = ttnn.add(residual, attention_out, output_tensor=attention_out)
    residual.deallocate(True)
    post_attention_host = _to_host(post_attention)[:batch_size].clone()
    post_norm = decoder.post_attention_layernorm(post_attention)
    post_norm_host = _to_host(post_norm)[:batch_size].clone()
    post_attention.deallocate(True)
    return post_norm, post_attention_host, post_norm_host


def _tt_fixed_route_experts(decoder, post_norm, config, batch_size):
    """Extract dense TT routes and execute TT experts with those exact routes."""
    user_states = ttnn.split(post_norm, 1, dim=2)
    route_indices = []
    route_weights = []
    expert_outputs = []

    for user_state in user_states:
        router_input = ttnn.clone(user_state, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        indices, dense_weights = decoder.mlp.router(router_input, use_throughput_experts=False)
        router_input.deallocate(True)

        indices_host = ttnn.to_torch(indices).reshape(-1)[: config.num_experts_per_tok].to(torch.long)
        dense_host = ttnn.to_torch(dense_weights).reshape(-1, config.num_local_experts)[0]
        route_indices.append(indices_host)
        route_weights.append(dense_host.gather(0, indices_host))

        expert_input = ttnn.clone(user_state, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        expert_dense_weights = ttnn.clone(dense_weights, memory_config=dense_weights.memory_config())
        expert_output = decoder.mlp.experts(
            expert_input,
            topk_expert_indices=None,
            topk_expert_weights=expert_dense_weights,
            is_decode=True,
        )
        expert_outputs.append(_to_host(expert_output)[:1].clone())
        expert_output.deallocate(True)
        indices.deallocate(True)
        dense_weights.deallocate(True)

    post_norm.deallocate(True)
    return torch.stack(route_indices), torch.stack(route_weights), torch.cat(expert_outputs)


@pytest.mark.skipif(
    not RUN_ANALYSIS,
    reason="set GPT_OSS_120B_ROUTING_ANALYSIS=1 for the real-weight routing counterfactual",
)
@pytest.mark.skipif(
    not REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_SNAPSHOT to the exact openai/gpt-oss-120b snapshot",
)
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@parametrize_mesh_with_fabric([(1, 1)])
def test_real_weight_batch_two_traced_decode_routing_counterfactual(mesh_device, device_params, layer_idx, reset_seeds):
    """Prove fixed-route parity on actual TT states for both decoder kinds."""
    config = _config()
    batch_size = 2
    started = time.perf_counter()
    state_dict = load_real_layer_state_dict(REAL_WEIGHT_SNAPSHOT, layer_idx)
    print(f"ROUTING_ANALYSIS_DEQUANT layer={layer_idx} seconds={time.perf_counter() - started:.3f}")
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
        max_batch_size=batch_size,
        max_context_length=config.max_position_embeddings,
        page_size=PAGE_SIZE,
        tensor_cache_path=cache_root,
    )
    page_table = _page_table(
        mesh_device,
        config.max_position_embeddings,
        batch_size=batch_size,
        seed=251 + layer_idx,
    )

    sequence_length = 33
    positions = torch.arange(sequence_length, dtype=torch.long)
    prefix_generator = torch.Generator().manual_seed(250_000 + layer_idx)
    prefix_hidden = (
        torch.randn((batch_size, sequence_length, config.hidden_size), generator=prefix_generator) * 0.02
    ).to(torch.bfloat16)
    tt_prefix = ttnn.from_torch(
        prefix_hidden.reshape(1, batch_size, sequence_length, config.hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_prefill = decoder.prefill_forward(
        tt_prefix,
        position_embeddings=_rope_tensors(config, mesh_device, positions, decode=False),
        page_table=page_table,
        batch_size=batch_size,
    )
    ttnn.synchronize_device(mesh_device)
    assert torch.isfinite(_to_host(tt_prefill)[:batch_size]).all()
    tt_prefill.deallocate(True)

    decode_hidden, candidate_indices, candidate_margins = _select_low_margin_decode_inputs(
        reference, config, layer_idx, batch_size
    )
    decode_positions = torch.tensor([sequence_length, sequence_length - 2], dtype=torch.long)
    expected_output = []
    expected_post_attention = []
    expected_post_norm = []
    for user, current_position in enumerate(decode_positions.tolist()):
        output, post_attention, post_norm = _hf_decode_intermediates(
            reference,
            config,
            layer_idx,
            prefix_hidden[user : user + 1, :current_position],
            decode_hidden[user : user + 1],
            current_position,
        )
        expected_output.append(output)
        expected_post_attention.append(post_attention)
        expected_post_norm.append(post_norm)
    expected_output = torch.stack(expected_output)
    expected_post_attention = torch.stack(expected_post_attention)
    expected_post_norm = torch.stack(expected_post_norm)

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

    # Compile, capture the complete production decoder, and measure replay output.
    decoder.decode_forward(
        tt_decode_hidden,
        position_embeddings=decode_rope,
        current_position=tt_current,
        page_table=page_table,
        batch_size=batch_size,
    )
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward(
        tt_decode_hidden,
        position_embeddings=decode_rope,
        current_position=tt_current,
        page_table=page_table,
        batch_size=batch_size,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    traced_output_host = _to_host(traced_output)[:batch_size].clone()
    ttnn.release_trace(mesh_device, trace_id)

    tt_post_norm, tt_post_attention_host, tt_post_norm_host = _tt_decode_intermediates(
        decoder,
        tt_decode_hidden,
        decode_rope,
        tt_current,
        page_table,
        batch_size,
    )
    tt_route_indices, tt_route_weights, tt_expert_output = _tt_fixed_route_experts(
        decoder, tt_post_norm, config, batch_size
    )

    # ttnn.to_torch materializes BF16 device tensors as FP32 on this build.
    # Cast back to the checkpoint dtype so the HF counterfactual uses the same
    # BF16 boundary as the device router and experts.
    hf_post_norm = tt_post_norm_host.to(reference.mlp.router.weight.dtype)
    tt_route_weights = tt_route_weights.to(hf_post_norm.dtype)
    with torch.no_grad():
        hf_router_logits, hf_route_weights, hf_route_indices = reference.mlp.router(hf_post_norm)
        del hf_router_logits
        hf_natural_expert_output = reference.mlp.experts(
            hf_post_norm,
            router_indices=hf_route_indices,
            routing_weights=hf_route_weights,
        )
        hf_fixed_expert_output = reference.mlp.experts(
            hf_post_norm,
            router_indices=tt_route_indices,
            routing_weights=tt_route_weights,
        )

    tt_fixed_tail = tt_post_attention_host + tt_expert_output
    hf_natural_tail = tt_post_attention_host + hf_natural_expert_output
    hf_fixed_tail = tt_post_attention_host + hf_fixed_expert_output

    fixed_route_pcc = _pcc(hf_fixed_tail, tt_fixed_tail)
    traced_vs_diagnostic_pcc = _pcc(traced_output_host, tt_fixed_tail)
    whole_decoder_pcc = _pcc(expected_output, traced_output_host)
    post_attention_pcc = _pcc(expected_post_attention, tt_post_attention_host)
    post_norm_pcc = _pcc(expected_post_norm, tt_post_norm_host)
    route_sensitivity_pcc = _pcc(hf_natural_tail, hf_fixed_tail)
    exact_route_agreement = (
        torch.sort(hf_route_indices, dim=-1).values == torch.sort(tt_route_indices, dim=-1).values
    ).all(dim=-1)
    route_overlap = (tt_route_indices.unsqueeze(-1) == hf_route_indices.unsqueeze(-2)).any(dim=-1).float().mean()

    assert fixed_route_pcc >= DEFAULT_PCC_THRESHOLD, (
        "TT-vs-HF fixed-route full-tail PCC did not clear the default functional-decoder bar: " f"{fixed_route_pcc:.9f}"
    )
    assert traced_vs_diagnostic_pcc >= DEFAULT_PCC_THRESHOLD, (
        "the fixed-route diagnostic does not represent the actual traced decoder output: "
        f"{traced_vs_diagnostic_pcc:.9f}"
    )
    assert not bool(exact_route_agreement.all()), "selected analysis probes did not cross a TT-vs-HF route boundary"
    assert route_sensitivity_pcc < DEFAULT_PCC_THRESHOLD, (
        "route changes did not produce the expected discontinuous full-tail sensitivity: "
        f"{route_sensitivity_pcc:.9f}"
    )
    print(
        "ROUTING_COUNTERFACTUAL "
        f"layer={layer_idx} type={config.layer_types[layer_idx]} batch={batch_size} "
        f"candidate_indices={candidate_indices.tolist()} candidate_margins={candidate_margins.tolist()} "
        f"whole_decoder_pcc={whole_decoder_pcc:.9f} "
        f"post_attention_pcc={post_attention_pcc:.9f} post_norm_pcc={post_norm_pcc:.9f} "
        f"exact_route_agreement={exact_route_agreement.float().mean().item():.6f} "
        f"topk_route_overlap={route_overlap.item():.6f} "
        f"natural_vs_fixed_tail_pcc={route_sensitivity_pcc:.9f} "
        f"fixed_route_hf_vs_tt_tail_pcc={fixed_route_pcc:.9f} "
        f"traced_vs_diagnostic_tail_pcc={traced_vs_diagnostic_pcc:.9f} "
        f"tt_routes={tt_route_indices.tolist()} hf_routes={hf_route_indices.tolist()} "
        f"tt_route_weights={tt_route_weights.float().tolist()}"
    )
