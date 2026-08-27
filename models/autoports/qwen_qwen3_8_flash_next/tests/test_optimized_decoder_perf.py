# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Warmed optimized-layer prefill and traced-decode performance windows."""

from __future__ import annotations

import hashlib
import json
import os
import time

import pytest
import torch
from safetensors import safe_open
from tracy import signpost

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import harness as H
from models.autoports.qwen_qwen3_8_flash_next.tests.test_functional_decoder import _paged_inputs, _upload
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedDecoder

LAYER_KINDS = tuple(int(layer) for layer in os.environ.get("QWEN38_OPT_PERF_LAYERS", "0,1,3").split(","))
DECODE_REPLAYS = int(os.environ.get("QWEN38_OPT_PERF_DECODE_REPLAYS", "10"))
DECODE_CONFIGS = tuple(spec for spec in os.environ.get("QWEN38_OPT_PERF_DECODE_CONFIGS", "").split(";") if spec) or (
    None,
)
IMPLEMENTATIONS = {"fused": FusedDecoder, "optimized": OptimizedDecoder}
EMBEDDING_KEY = "model.language_model.embed_tokens.weight"
EMBEDDING_ROW_START = 2026
PREFILL_SEQ_LEN = 128


def _implementation():
    name = os.environ.get("QWEN38_OPT_PERF_IMPLEMENTATION", "optimized").lower()
    try:
        return name, IMPLEMENTATIONS[name]
    except KeyError as exc:
        raise ValueError(f"unsupported QWEN38_OPT_PERF_IMPLEMENTATION={name}") from exc


def _checkpoint_embedding_rows(count: int) -> torch.Tensor:
    """Read only the deterministic token rows used as benchmark activations."""

    index = json.loads((H.MODEL_SNAPSHOT / "model.safetensors.index.json").read_text())
    shard = index["weight_map"][EMBEDDING_KEY]
    with safe_open(H.MODEL_SNAPSHOT / shard, framework="pt", device="cpu") as handle:
        rows = handle.get_slice(EMBEDDING_KEY)[EMBEDDING_ROW_START : EMBEDDING_ROW_START + count]
    if tuple(rows.shape) != (count, 2560):
        raise AssertionError(f"unexpected checkpoint embedding slice shape {tuple(rows.shape)}")
    return rows.contiguous()


def _real_activations(layer_idx: int):
    """Build four hyper streams and PLE inputs solely from real token rows."""

    rows_per_input = PREFILL_SEQ_LEN + 1
    rows = _checkpoint_embedding_rows(2 * rows_per_input)
    hidden_base = rows[:rows_per_input]
    hidden = hidden_base.repeat(1, 4).reshape(1, 1, rows_per_input, 10240).contiguous()
    ple = rows[rows_per_input:].reshape(1, 1, rows_per_input, 2560).contiguous() if layer_idx == 1 else None
    digest = hashlib.sha256()
    digest.update(hidden.view(torch.uint8).numpy().tobytes())
    if ple is not None:
        digest.update(ple.view(torch.uint8).numpy().tobytes())
    return hidden, ple, digest.hexdigest()


def _observe_routing(layer, invoke):
    """Observe active experts before timed signposts, then restore the runtime path."""

    observation = {}
    original = layer._routed_experts

    def observe(x, routing):
        routing_host = ttnn.to_torch(routing)
        selected = routing_host.reshape(-1, routing_host.shape[-1]).ne(0)
        if selected.shape[0] % 32:
            raise AssertionError(f"routing rows must be tile padded, got {selected.shape[0]}")
        tile_masks = selected.reshape(-1, 32, selected.shape[-1]).any(dim=1)
        global_mask = tile_masks.any(dim=0)
        observation["union"] = int(global_mask.sum())
        observation["tile_unions"] = tuple(int(value) for value in tile_masks.sum(dim=1))
        observation["expert_hash"] = hashlib.sha256(selected.to(torch.uint8).numpy().tobytes()).hexdigest()
        return original(x, routing)

    layer._routed_experts = observe
    try:
        output = invoke()
        ttnn.synchronize_device(layer.mesh_device)
        ttnn.deallocate(output)
    finally:
        del layer._routed_experts
    if not observation:
        raise AssertionError("unmeasured warmup did not execute routed experts")
    return observation


def _evidence_prefix(layer, implementation_name, decode_config, activation_hash, routing):
    policy = getattr(getattr(layer, "optimization_policy", None), "name", "fused-baseline")
    projection_names = getattr(layer, "projection_policy_names", {})
    tile_unions = ",".join(str(value) for value in routing["tile_unions"])
    return (
        f"OPT_PERFEVIDENCE implementation={implementation_name} weights=real-checkpoint "
        f"activation=checkpoint-token-embeddings activation_sha256={activation_hash} "
        f"embedding_row_start={EMBEDDING_ROW_START} policy={policy} "
        f"expert_topology={getattr(layer, 'expert_topology', 'packed')} "
        f"candidate={decode_config or 'default'} shared={projection_names.get('shared', 'bf16_hifi4')} "
        f"gdn={projection_names.get('gdn', 'bf16_hifi4')} "
        f"qsa_input={projection_names.get('qsa_input', 'bf16_hifi4')} "
        f"attention_output={projection_names.get('attention_output', 'bf16_hifi4')} "
        f"cache={getattr(layer, 'cache_policy', 'bf16')} "
        f"prefill_logical_batch=1 prefill_logical_rows={PREFILL_SEQ_LEN} "
        f"prefill_tile_rows={PREFILL_SEQ_LEN // 32} decode_logical_batch=1 decode_logical_rows=1 "
        f"decode_tile_rows=1 routing_union={routing['union']} routing_tile_unions={tile_unions} "
        f"expert_hash={routing['expert_hash']}"
    )


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.parametrize(
    "decode_config",
    DECODE_CONFIGS,
    ids=lambda spec: "default" if spec is None else spec.replace(",", "+"),
)
def test_warmed_prefill_and_traced_decode(mesh_device, layer_idx, decode_config):
    mode = os.environ.get("QWEN38_OPT_PERF_MODE", "both")
    if mode not in {"both", "prefill", "decode"}:
        raise ValueError(f"unsupported QWEN38_OPT_PERF_MODE={mode}")
    implementation_name, decoder_cls = _implementation()
    if implementation_name == "fused" and decode_config is not None:
        raise ValueError("QWEN38_OPT_PERF_DECODE_CONFIGS applies only to the optimized implementation")
    config = H.target_config()
    max_seq_len = 4096 if layer_idx == 3 else 128
    layer_kwargs = {}
    if decode_config is not None:
        layer_kwargs["decode_1d_config"] = decode_config
    state = H.load_real_layer_state(layer_idx)
    layer = decoder_cls.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=max_seq_len,
        **layer_kwargs,
    )
    del state
    assert type(layer) is decoder_cls
    if "QWEN38_OPT_PERF_IMPLEMENTATION" not in os.environ:
        assert type(layer) is OptimizedDecoder
    hidden_host, ple_host, activation_hash = _real_activations(layer_idx)
    prefill_hidden = _upload(hidden_host[:, :, :PREFILL_SEQ_LEN], mesh_device)
    prefill_kwargs = {}
    if layer_idx == 1:
        prefill_kwargs["ple_embeddings"] = _upload(ple_host[:, :, :PREFILL_SEQ_LEN], mesh_device)
    if layer_idx == 3:
        cos, sin = H.rope_tables(max_seq_len)
        page_host = H.shuffled_page_table(max_seq_len)
        page, chunk_pages, rot = _paged_inputs(layer, mesh_device, page_host, cos, sin, PREFILL_SEQ_LEN)
        prefill_kwargs.update(page_table=page, page_tables_per_chunk=chunk_pages, rot_mats=rot)

    routing = _observe_routing(layer, lambda: layer.prefill_forward(prefill_hidden, **prefill_kwargs))
    evidence_prefix = _evidence_prefix(layer, implementation_name, decode_config, activation_hash, routing)
    prefill_ms = None
    if mode in {"both", "prefill"}:
        signpost(f"OPT_PERF_PREFILL_L{layer_idx}")
        start = time.perf_counter()
        with H.ForbidHostFallback():
            layer.prefill_forward(prefill_hidden, **prefill_kwargs)
        ttnn.synchronize_device(mesh_device)
        prefill_ms = (time.perf_counter() - start) * 1000.0
        signpost(f"OPT_PERF_PREFILL_L{layer_idx}_END")
    if mode == "prefill":
        print(
            f"{evidence_prefix} prefill_configs={getattr(layer, 'prefill_configs', {})} "
            f"prefill_output={getattr(layer, 'prefill_output', 'dram')} layer={layer_idx} "
            f"prefill_seq={PREFILL_SEQ_LEN} warmed_prefill_ms={prefill_ms:.6f}"
        )
        return

    layer.prepare_decode_state()
    decode_hidden = _upload(hidden_host[:, :, PREFILL_SEQ_LEN:], mesh_device)
    current_pos = _upload(
        torch.tensor([127], dtype=torch.int32), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    decode_kwargs = {"current_pos": current_pos}
    if layer_idx == 1:
        decode_kwargs["ple_embeddings"] = _upload(ple_host[:, :, PREFILL_SEQ_LEN:], mesh_device)
    if layer_idx == 3:
        decode_kwargs.update(page_table=page, rot_mats=rot)

    decode_routing = _observe_routing(layer, lambda: layer.decode_forward(decode_hidden, **decode_kwargs))
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    with H.ForbidHostFallback():
        traced_output = layer.decode_forward(decode_hidden, **decode_kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)

    signpost(f"OPT_PERF_DECODE_L{layer_idx}")
    start = time.perf_counter()
    for _ in range(DECODE_REPLAYS):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    decode_ms = (time.perf_counter() - start) * 1000.0 / DECODE_REPLAYS
    signpost(f"OPT_PERF_DECODE_L{layer_idx}_END")
    assert list(traced_output.shape) == [1, 1, 1, 10240]
    ttnn.release_trace(mesh_device, trace_id)
    prefill_field = "not-profiled" if prefill_ms is None else f"{prefill_ms:.6f}"
    print(
        f"{evidence_prefix} decode_1d={getattr(layer, 'decode_1d_cores', {})} "
        f"decode_1d_output={getattr(layer, 'decode_1d_output', 'dram')} "
        f"dram_sharded={getattr(layer, 'dram_sharded_role', '') or 'none'} "
        f"prefill_configs={getattr(layer, 'prefill_configs', {})} "
        f"prefill_output={getattr(layer, 'prefill_output', 'dram')} "
        f"decode_routing_union={decode_routing['union']} "
        f"decode_routing_tile_unions={','.join(str(value) for value in decode_routing['tile_unions'])} "
        f"decode_expert_hash={decode_routing['expert_hash']} "
        f"layer={layer_idx} prefill_seq={PREFILL_SEQ_LEN} "
        f"warmed_prefill_ms={prefill_field} traced_decode_ms={decode_ms:.6f} "
        f"decode_replays={DECODE_REPLAYS}"
    )
