# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fast contract checks for the full-model stage (no device required)."""

from __future__ import annotations

import inspect
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from transformers import AutoConfig

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator
from models.autoports.google_gemma_4_26b_a4b_it.tt.model import (
    DEFAULT_MAX_CONTEXT,
    FULL_KIND,
    PROFILE_CONTEXT_LIMITS,
    SLIDING_CACHE_TOKENS,
    FullModelState,
    Gemma4FullModel,
    PagedCacheSpec,
)


def _config():
    return AutoConfig.from_pretrained(
        "models/demos/gemma4/configs/gemma-4-26B-A4B-it", local_files_only=True
    ).text_config


def test_full_model_architecture_and_cache_geometry_contract():
    config = _config()
    model = Gemma4FullModel.__new__(Gemma4FullModel)
    model.hf_config = config
    model.num_layers = config.num_hidden_layers
    model.layer_indices = list(range(config.num_hidden_layers))
    model.max_seq_len = DEFAULT_MAX_CONTEXT
    specs = model._make_cache_specs()

    assert len(specs) == 30
    assert sum(spec.layer_type == "sliding_attention" for spec in specs) == 25
    assert sum(spec.layer_type == "full_attention" for spec in specs) == 5
    for spec in specs:
        if spec.layer_type == "sliding_attention":
            assert (spec.block_size, spec.local_kv_heads, spec.head_dim) == (64, 2, 256)
            assert spec.capacity_tokens_per_slot == SLIDING_CACHE_TOKENS
        else:
            assert (spec.block_size, spec.local_kv_heads, spec.head_dim) == (128, 1, 512)
            assert spec.capacity_tokens_per_slot == DEFAULT_MAX_CONTEXT


@pytest.mark.parametrize(
    "tp_size,expected_context,sliding_heads,full_heads",
    [(1, 50_624, 8, 2), (2, 262_144, 4, 1), (4, 262_144, 2, 1)],
)
def test_full_model_profile_context_and_cache_geometry(tp_size, expected_context, sliding_heads, full_heads):
    config = _config()
    model = Gemma4FullModel.__new__(Gemma4FullModel)
    model.hf_config = config
    model.layer_indices = list(range(config.num_hidden_layers))
    model.max_seq_len = expected_context
    model.tp_size = tp_size
    specs = model._make_cache_specs()
    assert PROFILE_CONTEXT_LIMITS[tp_size] == expected_context
    assert {spec.local_kv_heads for spec in specs if spec.layer_type == "sliding_attention"} == {sliding_heads}
    assert {spec.local_kv_heads for spec in specs if spec.layer_type == "full_attention"} == {full_heads}


def test_generator_implements_readiness_contract_with_explicit_trace_keyword():
    assert not getattr(Gemma4Generator, "__abstractmethods__", set())
    generate = inspect.signature(Gemma4Generator.generate)
    assert "enable_trace" in generate.parameters
    assert generate.parameters["enable_trace"].kind is inspect.Parameter.KEYWORD_ONLY
    for method in (Gemma4Generator.prefill_forward, Gemma4Generator.decode_forward):
        signature = inspect.signature(method)
        assert "page_table" in signature.parameters
        assert "kv_cache" in signature.parameters
    assert "prompt_token_ids" in inspect.signature(Gemma4Generator.prefill_logits).parameters


def test_full_model_preserves_logical_prefill_length_at_decoder_boundary():
    source = inspect.getsource(Gemma4FullModel.prefill_forward)
    assert source.index("logical_seq_len = int(prompt_lens[0])") < source.index("for state_idx")
    assert source.index("hidden = ttnn.slice(") < source.index("for state_idx")
    assert source.index("position_ids = ttnn.slice(") < source.index("for state_idx")


def test_caller_owned_cache_wrapper_is_stable_and_allocation_free():
    def fail_allocate(**_kwargs):
        raise AssertionError("caller-owned KV must not allocate replacement state")

    specs = [
        PagedCacheSpec(0, "sliding_attention", 64, 2, 256, 1_024),
        PagedCacheSpec(5, "full_attention", 128, 1, 512, 128),
    ]
    generator = Gemma4Generator.__new__(Gemma4Generator)
    generator.model = SimpleNamespace(
        num_layers=2,
        max_seq_len=128,
        cache_specs=specs,
        allocate_state=fail_allocate,
    )
    generator._borrowed_states = {}
    cache_leaves = [object(), object(), object(), object()]
    page_tables = [torch.zeros((32, 16), dtype=torch.int32), torch.zeros((32, 1), dtype=torch.int32)]
    first = generator._state_from_args(
        kv_cache=[[cache_leaves[0], cache_leaves[1]], [cache_leaves[2], cache_leaves[3]]],
        page_table=list(page_tables),
        batch_size=2,
    )
    second = generator._state_from_args(
        kv_cache=[(cache_leaves[0], cache_leaves[1]), (cache_leaves[2], cache_leaves[3])],
        page_table=list(page_tables),
        batch_size=2,
    )
    assert first is second
    assert [tensor for pair in first.kv_cache for tensor in pair] == cache_leaves


def test_batched_return_all_logits_has_one_logical_host_tensor(monkeypatch):
    state = FullModelState(
        kv_cache=[],
        page_tables=[],
        cache_specs=[],
        max_batch_size=2,
        slot_context_lengths=[128, 128] + [0] * 30,
        prompt_lens=[0] * 32,
        positions=torch.full((32,), -1, dtype=torch.int32),
        active_mask=torch.zeros(32, dtype=torch.bool),
    )
    generator = Gemma4Generator.__new__(Gemma4Generator)
    generator.mesh_device = object()
    monkeypatch.setattr(ttnn, "corruptible_allocation_scope", lambda _device: nullcontext())
    generator.model = SimpleNamespace(
        max_seq_len=128,
        state=state,
        prefill_forward=lambda tokens, **kwargs: torch.full((1, tokens.shape[1], 4), float(kwargs["user_id"] + 1)),
    )
    generator._request_boundary = False
    generator._seeded_slots = torch.zeros(32, dtype=torch.bool)
    generator._sampling_key = None
    generator._state_from_args = lambda **_kwargs: state
    generator._host_tokens_to_device = lambda values: values
    generator._positions_to_device = lambda values, **_kwargs: values
    generator._gather_logits_to_torch = lambda values, logical_len=None: values[:, :logical_len]
    logits = generator.prefill_forward(
        torch.ones((2, 64), dtype=torch.long),
        page_table=[],
        kv_cache=state,
        prompt_lens=[33, 47],
        return_all_logits=True,
    )
    assert tuple(logits.shape) == (2, 47, 4)
    assert torch.equal(logits[0, :33], torch.ones((33, 4)))
    assert torch.equal(logits[0, 33:], torch.zeros((14, 4)))
    assert torch.equal(logits[1], torch.full((47, 4), 2.0))


def test_optimized_decode_source_has_no_host_logits_or_argmax_boundary():
    model_source = inspect.getsource(Gemma4FullModel.decode_forward)
    trace_source = inspect.getsource(Gemma4Generator._get_or_capture_decode_trace)
    forbidden = ("to_torch", "torch.argmax", ".argmax(", ".cpu(", ".numpy(")
    assert not any(token in model_source for token in forbidden)
    assert not any(token in trace_source for token in forbidden)
    assert "tt_out_tok=token_input" in trace_source
    decode_source = inspect.getsource(Gemma4Generator.decode_forward)
    assert "ttnn.plus_one(trace.current_pos" not in decode_source
    assert "ttnn.plus_one(trace.position_ids" not in decode_source
    assert "ttnn.plus_one(current_pos" in trace_source
    assert "ttnn.plus_one(position_ids" in trace_source


def test_sampling_specs_are_validated_and_trace_distinct(expect_error):
    generator = object.__new__(Gemma4Generator)
    greedy = generator._sampling_spec(1, temperature=0.0)
    sampled = generator._sampling_spec(1, top_k=8, top_p=0.95, temperature=0.8, seeds=42)
    assert greedy.greedy
    assert not sampled.greedy
    assert sampled.top_k == (8,)
    assert sampled.top_p == (0.95,)
    assert sampled.temperature == (0.8,)
    assert sampled.seeds == (42,)
    assert greedy.key != sampled.key
    with expect_error(ValueError, "top_k"):
        generator._sampling_spec(1, top_k=33, top_p=0.9, temperature=1.0)
    with expect_error(ValueError, "top_p"):
        generator._sampling_spec(1, top_k=8, top_p=1.1, temperature=1.0)
    with expect_error(ValueError, "seeds"):
        generator._sampling_spec(1, top_k=8, top_p=0.9, temperature=1.0, seeds=2**32 - 1)


def test_sampler_choice_is_semantically_greedy_split_topk():
    source = inspect.getsource(Gemma4Generator._sampling_params)
    init = inspect.getsource(Gemma4Generator.__init__)
    assert "return None, None, None" not in source
    assert "allow_force_argmax=False" in init
    trace_source = inspect.getsource(Gemma4Generator._get_or_capture_decode_trace)
    assert "ttnn.copy(input_a=seed_skip, input_b=seed_tensor)" in trace_source
    assert "MAX_UINT32" in trace_source
    assert "max_top_k=32" in init
    assert "ag_topology=ttnn.Topology.Ring if model.tp_size == 4 else ttnn.Topology.Linear" in init
    assert "Sampling1D" in init


def test_retained_trace_first_token_sampling_has_explicit_lifetime_scope():
    source = inspect.getsource(Gemma4Generator.generate)
    scope = source.index("with ttnn.corruptible_allocation_scope(self.mesh_device):")
    params = source.index("k, p, temp, seed_tensor = self._sampling_params(spec)", scope)
    readback = source.index("predicted = int(self._read_tokens(first_tt, 1)[0])", params)
    seeded = source.index("self._seeded_slots[0] = True", readback)
    assert scope < params < readback < seeded


def test_cache_spec_rounding():
    spec = PagedCacheSpec(0, "full_attention", 128, 1, FULL_KIND.head_dim, 129)
    assert spec.blocks_per_slot == 2


def _load_probe_state(*, full_stack: bool = False):
    roots = Path(os.environ.get("HF_HOME", "/huggingface")) / "hub/models--google--gemma-4-26B-A4B-it/snapshots"
    snapshots = [p for p in roots.iterdir() if (p / "model.safetensors.index.json").is_file()]
    if not snapshots:
        pytest.skip("local Gemma4 snapshot unavailable")
    snapshot = snapshots[0]
    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    prefixes = (
        ("model.language_model.layers.",)
        if full_stack
        else (
            "model.language_model.layers.0.",
            "model.language_model.layers.5.",
        )
    )
    terminal = {"model.language_model.embed_tokens.weight", "model.language_model.norm.weight"}
    wanted = {name for name in weight_map if name.startswith(prefixes) or name in terminal}
    state = {}
    for shard_name in sorted({weight_map[name] for name in wanted}):
        with safe_open(snapshot / shard_name, framework="pt", device="cpu") as shard:
            for name in wanted:
                if weight_map[name] == shard_name:
                    state[name] = shard.get_tensor(name)
    return state


@pytest.mark.parametrize(
    "mesh_device,device_params,profile_tp_size",
    [
        ((1, 1), {"trace_region_size": 67_108_864}, 1),
        ((2, 2), {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 67_108_864}, 2),
        # B32 decode records a 125.3 MiB full-stack model trace.  Reserve
        # 128 MiB so the same profile covers both the B1 latency target and
        # the fixed-slot B32 capability gate.
        ((1, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 134_217_728}, 4),
    ],
    ids=["p150", "p150x2", "p150x4"],
    indirect=["mesh_device", "device_params"],
)
def test_reduced_real_weight_full_model_probe(mesh_device, profile_tp_size, expect_error):
    if os.environ.get("GEMMA4_FULL_MODEL_PROBE") != "1":
        pytest.skip("set GEMMA4_FULL_MODEL_PROBE=1 for the serialized full-model hardware probe")
    full_stack = os.environ.get("GEMMA4_FULL_STACK_PROBE") == "1"
    batch32 = os.environ.get("GEMMA4_BATCH32_PROBE") == "1"
    long_context = os.environ.get("GEMMA4_LONG_CONTEXT_PROBE") == "1"
    capacity_only = os.environ.get("GEMMA4_CAPACITY_ONLY") == "1"
    state = _load_probe_state(full_stack=full_stack)
    tp_size = profile_tp_size
    target_mesh = (
        mesh_device
        if mesh_device.get_num_devices() == tp_size
        else mesh_device.create_submesh(ttnn.MeshShape((1, tp_size)), offset=ttnn.MeshCoordinate(0, 0))
    )
    prompt_len = (
        PROFILE_CONTEXT_LIMITS[tp_size] - 1 if long_context else int(os.environ.get("GEMMA4_PROBE_PROMPT_LEN", "32"))
    )
    model = Gemma4FullModel(
        mesh_device=target_mesh,
        hf_config=_config(),
        state_dict=state,
        max_seq_len=(
            PROFILE_CONTEXT_LIMITS[tp_size]
            if (long_context or capacity_only)
            else (
                2_048
                if batch32
                else (
                    512
                    if os.environ.get("GEMMA4_NO_HOST_TOKEN_OUT_BENCH") == "1"
                    or os.environ.get("GEMMA4_PREFILL_BENCH") == "1"
                    or os.environ.get("GEMMA4_GENERATE_BENCH") == "1"
                    else 128
                )
            )
        ),
        max_batch_size=32 if batch32 else 1,
        layer_indices=None if full_stack else [0, 5],
        tensor_cache_path=("/tmp/gemma4_full_model_cache" if full_stack else "/tmp/gemma4_full_model_probe_cache"),
    )
    if os.environ.get("GEMMA4_PRINT_PRECISION_SUMMARY") == "1":
        print("PRECISION_SUMMARY=" + json.dumps(model.precision_summary(), sort_keys=True))
    probe_batch = 32 if batch32 else 1
    # Prefill is intentionally scheduled one slot at a time; the public
    # generator owns mixed-prompt iteration.  Decode then exercises all 32
    # fixed slots together against the same fully allocated state.
    tokens = ttnn.from_torch(
        torch.ones((1, prompt_len), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=target_mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(target_mesh),
    )
    positions = ttnn.from_torch(
        torch.arange(prompt_len, dtype=torch.int32).reshape(1, prompt_len),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=target_mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(target_mesh),
    )
    if os.environ.get("GEMMA4_EMBED_TRACE_ONLY") == "1":
        decode_token = ttnn.from_torch(
            torch.ones((1, 1), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=target_mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(target_mesh),
        )
        model.embed_tokens(decode_token)
        ttnn.synchronize_device(target_mesh)
        trace_id = ttnn.begin_trace_capture(target_mesh, cq_id=0)
        model.embed_tokens(decode_token)
        ttnn.end_trace_capture(target_mesh, trace_id, cq_id=0)
        ttnn.release_trace(target_mesh, trace_id)
        return
    generator = Gemma4Generator(model, tokenizer=None, sampling_mode="device")
    original_page_tables = None
    replacement_page_tables = None
    second_replacement_page_tables = None
    if os.environ.get("GEMMA4_SAMPLED_TRACE_PROBE") == "1":
        # Allocate identity-distinct but value-identical tables before traces
        # pin allocator addresses. The sampled probe later uses them to prove
        # identity-only page-table adoption preserves token and RNG feedback.
        original_page_tables = list(model.state.page_tables)
        host_page_tables = [
            ttnn.to_torch(ttnn.get_device_tensors(table)[0]).to(torch.int32) for table in model.state.page_tables
        ]
        replacement_page_tables = [generator._positions_to_device(table) for table in host_page_tables]
        second_replacement_page_tables = [generator._positions_to_device(table) for table in host_page_tables]
    batch32_oracle_tokens = None
    batch32_oracle_logits = None
    batch32_decode_logits = None
    batch32_prompt_groups = None
    if capacity_only:
        assert full_stack, "capacity evidence must construct the complete 30-layer stack"
        assert model.max_seq_len == PROFILE_CONTEXT_LIMITS[tp_size]
        assert len(model.layers) == len(model.cache_specs) == 30
        if tp_size > 1:
            resources = model.layers[0].persistent_all_reduce_resources
            assert resources is not None
            assert all(layer.persistent_all_reduce_resources is resources for layer in model.layers)
            assert len(resources["buffers"]) == len(resources["semaphores"]) == 3
        full_specs = [spec for spec in model.cache_specs if spec.layer_type == "full_attention"]
        sliding_specs = [spec for spec in model.cache_specs if spec.layer_type == "sliding_attention"]
        assert len(full_specs) == 5 and len(sliding_specs) == 25
        assert all(spec.capacity_tokens_per_slot == PROFILE_CONTEXT_LIMITS[tp_size] for spec in full_specs)
        assert all(spec.capacity_tokens_per_slot == 1_024 for spec in sliding_specs)
        ttnn.synchronize_device(target_mesh)
        dram = ttnn.get_memory_view(target_mesh, ttnn.BufferType.DRAM)
        output_dir = Path(os.environ.get("GEMMA4_FULL_MODEL_PROBE_OUTPUT_DIR", "/tmp/gemma4_full_model_evidence"))
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "verdict": "pass",
            "profile": {1: "P150", 2: "P150x2", 4: "P150x4"}[tp_size],
            "tp_size": tp_size,
            "mesh_shape": list(target_mesh.shape),
            "checkpoint_revision": "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
            "real_weights": True,
            "num_layers": len(model.layers),
            "max_seq_len": model.max_seq_len,
            "max_batch_size": model.max_batch_size,
            "kv_cache_dtype": str(model.kv_cache_dtype),
            "terminal_weight_dtype": str(model.terminal_weight_dtype),
            "local_full_kv_heads": full_specs[0].local_kv_heads,
            "local_sliding_kv_heads": sliding_specs[0].local_kv_heads,
            "full_cache_shape": list(model.state.kv_cache[5][0].shape),
            "sliding_cache_shape": list(model.state.kv_cache[0][0].shape),
            "page_table_shapes": sorted({tuple(table.shape) for table in model.state.page_tables}),
            "persistent_all_reduce_buffers": (
                0 if tp_size == 1 else len(model.layers[0].persistent_all_reduce_resources["buffers"])
            ),
            "sampler_buffers_preloaded": generator.sampler._device_buffers_loaded,
            "dram": {
                "num_banks": dram.num_banks,
                "total_bytes": dram.total_bytes_per_bank * dram.num_banks,
                "allocated_bytes": dram.total_bytes_allocated_per_bank * dram.num_banks,
                "free_bytes": dram.total_bytes_free_per_bank * dram.num_banks,
                "largest_contiguous_free_bytes_per_bank": dram.largest_contiguous_bytes_free_per_bank,
            },
        }
        (output_dir / f"capacity_tp{tp_size}.json").write_text(json.dumps(report, indent=2, default=list) + "\n")
        print("GEMMA4_CAPACITY_RESULT=" + json.dumps(report, sort_keys=True, default=list))
        return
    if os.environ.get("GEMMA4_GENERATE_BENCH") == "1":
        assert full_stack and tp_size == 4
        prompt = [1 + (index % 255) for index in range(128)]
        generator.generate(prompt, 8, enable_trace=True, stop_on_eos=False)
        generator.reset()
        output = generator.generate(prompt, 128, enable_trace=True, stop_on_eos=False)
        assert len(output) == 128
        report = {
            "verdict": "pass",
            "profile": "P150x4",
            "boundary": "public Gemma4Generator.generate with host-visible token list",
            "full_stack": True,
            "prompt_len": 128,
            "generated_tokens": len(output),
            "stop_on_eos": False,
            **generator.last_perf,
        }
        output_dir = Path(os.environ.get("GEMMA4_FULL_MODEL_PROBE_OUTPUT_DIR", "/tmp/gemma4_full_model_evidence"))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "generator_generate_tp4.json").write_text(json.dumps(report, indent=2) + "\n")
        print("GEMMA4_GENERATE_RESULT=" + json.dumps(report, sort_keys=True))
        return
    if long_context:
        logits = generator.prefill_forward(
            torch.ones((1, prompt_len), dtype=torch.long),
            page_table=model.state.page_tables,
            kv_cache=model.state,
            prompt_lens=[prompt_len],
        )
    elif batch32:
        # Four distinct prompt families, repeated across all 32 rows, make the
        # B32 path check row/cache ownership and output ordering against four
        # independently prefetched B1 states rather than only checking shape.
        batch32_prompt_groups = torch.arange(probe_batch, dtype=torch.long) % 4
        token_columns = torch.arange(prompt_len, dtype=torch.long).reshape(1, -1)
        batch_tokens = 1 + (token_columns + 17 * batch32_prompt_groups.reshape(-1, 1)) % 255
        oracle_tokens = []
        oracle_logits = []
        for group in range(4):
            oracle_state = model.allocate_state(max_batch_size=1, slot_context_lengths=[model.max_seq_len])
            group_prompt = batch_tokens[group : group + 1]
            generator.prefill_forward(
                group_prompt,
                page_table=oracle_state.page_tables,
                kv_cache=oracle_state,
                prompt_lens=[prompt_len],
            )
            oracle_decode = generator.decode_forward(
                torch.ones((1, 1), dtype=torch.long),
                torch.tensor([prompt_len], dtype=torch.int32),
                page_table=oracle_state.page_tables,
                kv_cache=oracle_state,
                active_mask=torch.tensor([True]),
                enable_trace=False,
                sampling_mode="host",
            )
            oracle_host = generator._gather_logits_to_torch(oracle_decode).reshape(-1, model.vocab_size)[0].float()
            oracle_logits.append(oracle_host)
            oracle_tokens.append(int(oracle_host.argmax()))
            del oracle_decode, oracle_state
        batch32_oracle_tokens = oracle_tokens
        batch32_oracle_logits = oracle_logits
        logits = generator.prefill_forward(
            batch_tokens,
            page_table=model.state.page_tables,
            kv_cache=model.state,
            prompt_lens=[prompt_len] * probe_batch,
        )
        batch_decode = generator.decode_forward(
            torch.ones((probe_batch, 1), dtype=torch.long),
            torch.full((probe_batch,), prompt_len, dtype=torch.int32),
            page_table=model.state.page_tables,
            kv_cache=model.state,
            active_mask=torch.ones(probe_batch, dtype=torch.bool),
            enable_trace=False,
            sampling_mode="host",
        )
        batch32_decode_logits = (
            generator._gather_logits_to_torch(batch_decode).reshape(-1, model.vocab_size)[:probe_batch].float()
        )
    else:
        if os.environ.get("GEMMA4_PREFILL_BENCH") == "1":
            ttnn.synchronize_device(target_mesh)
            start_s = time.perf_counter()
            logits = model.prefill_forward(tokens, state=model.state, prompt_lens=[prompt_len], position_ids=positions)
            ttnn.synchronize_device(target_mesh)
            initial_s = time.perf_counter() - start_s
            start_s = time.perf_counter()
            logits = model.prefill_forward(tokens, state=model.state, prompt_lens=[prompt_len], position_ids=positions)
            ttnn.synchronize_device(target_mesh)
            warmed_s = time.perf_counter() - start_s
            report = {
                "profile": {1: "P150", 2: "P150x2", 4: "P150x4"}[tp_size],
                "boundary": "full model prefill through last-token sampler-ready logits",
                "full_stack": full_stack,
                "prompt_len": prompt_len,
                "initial_ttft_ms": initial_s * 1000.0,
                "warmed_ttft_ms": warmed_s * 1000.0,
            }
            output_dir = Path(os.environ.get("GEMMA4_FULL_MODEL_PROBE_OUTPUT_DIR", "/tmp/gemma4_full_model_evidence"))
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"prefill_tp{tp_size}.json").write_text(json.dumps(report, indent=2) + "\n")
            print("GEMMA4_PREFILL_RESULT=" + json.dumps(report, sort_keys=True))
        else:
            logits = model.prefill_forward(tokens, state=model.state, prompt_lens=[prompt_len], position_ids=positions)
    expected_prefill_rows = probe_batch if batch32 else 1
    assert tuple(logits.shape) == (1, 1, expected_prefill_rows, 262_144 // tp_size)

    if os.environ.get("GEMMA4_SAMPLER_AB_BENCH") == "1":
        sampling_logits = generator._pad_sampling_logits(logits)
        spec = generator._sampling_spec(1, top_k=1, top_p=0.0, temperature=0.0, seeds=0)
        k, p, temp, seed_tensor = generator._sampling_params(spec)
        topk_output, _ = generator.sampler.decode_forward(sampling_logits, k=k, p=p, temp=temp, seeds=seed_tensor)
        # Force-argmax is the other common greedy path in Sampling1D.  It
        # gathers the full vocabulary; the selected path instead preserves a
        # tile of local candidates and is semantically greedy via k=1.
        force_sampler = type(generator.sampler)(
            vocab_size=model.vocab_size,
            mesh_device=target_mesh,
            max_batch_size=32,
            max_top_k=32,
            allow_force_argmax=True,
            pad_to_power_of_2=True,
            num_gather_links=2 if tp_size == 4 else 1,
            ag_topology=ttnn.Topology.Ring if tp_size == 4 else ttnn.Topology.Linear,
        )
        force_sampler.load_device_buffers()
        argmax_output, _ = force_sampler.decode_forward(sampling_logits)
        topk_token = int(ttnn.to_torch(ttnn.get_device_tensors(topk_output)[0]).reshape(-1)[0])
        argmax_token = int(ttnn.to_torch(ttnn.get_device_tensors(argmax_output)[0]).reshape(-1)[0])
        local_logits = [ttnn.to_torch(shard).reshape(-1) for shard in ttnn.get_device_tensors(logits)]
        local_argmax = [int(shard.argmax()) for shard in local_logits]
        local_max = [float(shard[index]) for shard, index in zip(local_logits, local_argmax)]
        winning_shard = int(torch.tensor(local_max).argmax())
        trusted_host_argmax = winning_shard * local_logits[winning_shard].numel() + local_argmax[winning_shard]
        gathered_host_argmax = int(generator._gather_logits_to_torch(logits, logical_len=1).reshape(-1).argmax())
        local_width = local_logits[0].numel()
        topk_value = float(local_logits[topk_token // local_width][topk_token % local_width])
        argmax_value = float(local_logits[argmax_token // local_width][argmax_token % local_width])
        global_max = max(local_max)
        print(
            "GEMMA4_SAMPLER_ORACLE="
            + json.dumps(
                {
                    "local_argmax": local_argmax,
                    "local_max": local_max,
                    "winning_shard": winning_shard,
                    "shard_composed_argmax": trusted_host_argmax,
                    "gathered_argmax": gathered_host_argmax,
                    "topk_value": topk_value,
                    "argmax_value": argmax_value,
                    "global_max": global_max,
                },
                sort_keys=True,
            )
        )
        # Gemma's final soft-cap can produce exact ties at 30.0. Greedy means
        # choosing any global maximum; token-id equality would incorrectly
        # reject a valid top-k tie break in favor of argmax's first-index tie
        # break.
        assert topk_value == global_max
        assert argmax_value == global_max
        semantically_equivalent = topk_value == argmax_value == global_max
        warmups = int(os.environ.get("GEMMA4_SAMPLER_WARMUPS", "5"))
        iterations = int(os.environ.get("GEMMA4_SAMPLER_ITERATIONS", "128"))

        def measure(run):
            for _ in range(warmups):
                run()
            ttnn.synchronize_device(target_mesh)
            start_s = time.perf_counter()
            for _ in range(iterations):
                run()
            ttnn.synchronize_device(target_mesh)
            return time.perf_counter() - start_s

        topk_s = measure(
            lambda: generator.sampler.decode_forward(
                sampling_logits, k=k, p=p, temp=temp, seeds=seed_tensor, tt_out_tok=topk_output
            )
        )
        argmax_s = (
            measure(lambda: force_sampler.decode_forward(sampling_logits, tt_out_tok=argmax_output))
            if semantically_equivalent
            else None
        )
        report = {
            "profile": {1: "P150", 2: "P150x2", 4: "P150x4"}[tp_size],
            "semantic_contract": "greedy",
            "warmups": warmups,
            "iterations": iterations,
            "selected": "Sampling1D local top-32 candidates plus k=1,p=0,temp=1",
            "selected_ms": topk_s * 1000.0 / iterations,
            "rejected": "Sampling1D force-argmax full-vocabulary gather",
            "rejected_ms": None if argmax_s is None else argmax_s * 1000.0 / iterations,
            "rejection_reason": (
                "different active-row token on identical TP4 vocab-sharded logits; timing rejected as inequivalent"
                if not semantically_equivalent
                else "slower semantically equivalent path"
            ),
            "selected_token": topk_token,
            "rejected_token": argmax_token,
            "selected_logit": topk_value,
            "rejected_logit": argmax_value,
            "global_max_logit": global_max,
            "trusted_host_argmax": trusted_host_argmax,
            "gathered_host_argmax": gathered_host_argmax,
            "local_argmax": local_argmax,
            "local_max": local_max,
            "selected_matches_trusted_oracle": topk_value == global_max,
            "semantically_equivalent": semantically_equivalent,
        }
        output_dir = Path(os.environ.get("GEMMA4_FULL_MODEL_PROBE_OUTPUT_DIR", "/tmp/gemma4_full_model_evidence"))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"sampler_ab_tp{tp_size}.json").write_text(json.dumps(report, indent=2) + "\n")
        print("GEMMA4_SAMPLER_AB_RESULT=" + json.dumps(report, sort_keys=True))
        return

    if os.environ.get("GEMMA4_MODEL_TRACE_ONLY") == "1":
        decode_token = generator._host_tokens_to_device(torch.ones((1, 1), dtype=torch.long), rank4=True)
        current_pos = generator._positions_to_device(torch.tensor([32], dtype=torch.int32))
        position_ids = generator._positions_to_device(torch.tensor([32], dtype=torch.int32), dtype=ttnn.uint32)
        model.decode_forward(
            decode_token, state=model.state, current_pos=current_pos, position_ids=position_ids, batch_size=1
        )
        if os.environ.get("GEMMA4_TRACE_PLUS_ONE") == "1":
            ttnn.plus_one(current_pos, skip_negative_entries=True)
            ttnn.plus_one(position_ids, skip_negative_entries=True)
        elif os.environ.get("GEMMA4_TRACE_INPLACE_ADD") == "1":
            ttnn.add(current_pos, 1, output_tensor=current_pos)
            ttnn.add(position_ids, 1, output_tensor=position_ids)
        ttnn.synchronize_device(target_mesh)
        trace_id = ttnn.begin_trace_capture(target_mesh, cq_id=0)
        model.decode_forward(
            decode_token, state=model.state, current_pos=current_pos, position_ids=position_ids, batch_size=1
        )
        if os.environ.get("GEMMA4_TRACE_PLUS_ONE") == "1":
            ttnn.plus_one(current_pos, skip_negative_entries=True)
            ttnn.plus_one(position_ids, skip_negative_entries=True)
        elif os.environ.get("GEMMA4_TRACE_INPLACE_ADD") == "1":
            ttnn.add(current_pos, 1, output_tensor=current_pos)
            ttnn.add(position_ids, 1, output_tensor=position_ids)
        ttnn.end_trace_capture(target_mesh, trace_id, cq_id=0)
        if os.environ.get("GEMMA4_NO_SAMPLING_BENCH") == "1":
            warmups = int(os.environ.get("GEMMA4_NO_HOST_WARMUPS", "5"))
            iterations = int(os.environ.get("GEMMA4_NO_HOST_ITERATIONS", "128"))
            for _ in range(warmups):
                ttnn.execute_trace(target_mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(target_mesh)
            start_s = time.perf_counter()
            for _ in range(iterations):
                ttnn.execute_trace(target_mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(target_mesh)
            elapsed_s = time.perf_counter() - start_s
            report = {
                "profile": {1: "P150", 2: "P150x2", 4: "P150x4"}[tp_size],
                "boundary": "model decode through sampler-ready logits; sampling excluded",
                "full_stack": full_stack,
                "configured_initial_position": 32,
                "warmups": warmups,
                "iterations": iterations,
                "elapsed_s": elapsed_s,
                "ms_per_token": elapsed_s * 1000.0 / iterations,
                "decode_t_s_u": iterations / elapsed_s,
            }
            output_dir = Path(os.environ.get("GEMMA4_FULL_MODEL_PROBE_OUTPUT_DIR", "/tmp/gemma4_full_model_evidence"))
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"logits_only_trace_tp{tp_size}.json").write_text(json.dumps(report, indent=2) + "\n")
            print("GEMMA4_NO_SAMPLING_RESULT=" + json.dumps(report, sort_keys=True))
        ttnn.release_trace(target_mesh, trace_id)
        return
    sampled = generator.decode_forward(
        torch.ones((probe_batch, 1), dtype=torch.long),
        torch.full((probe_batch,), prompt_len, dtype=torch.int32),
        page_table=model.state.page_tables,
        kv_cache=model.state,
        enable_trace=True,
        active_mask=torch.ones(probe_batch, dtype=torch.bool) if batch32 else None,
    )
    assert tuple(sampled.shape) == (1, 1, 1, 32)
    assert generator.trace_counters.replays == 1
    trace = next(iter(generator._trace_cache.values()))
    assert int(ttnn.to_torch(ttnn.get_device_tensors(trace.current_pos)[0]).reshape(-1)[0]) == prompt_len + 1
    assert generator.trace_counters.token_refreshes == 1
    assert generator.trace_counters.position_refreshes == 1
    if batch32:
        sampled_rows = generator._read_tokens(sampled, probe_batch).tolist()
        batch_max = batch32_decode_logits.amax(dim=-1)
        sampled_values = batch32_decode_logits[torch.arange(probe_batch), torch.tensor(sampled_rows, dtype=torch.long)]
        assert torch.equal(sampled_values, batch_max)
        batch32_oracle_cosines = []
        for row, group in enumerate(batch32_prompt_groups.tolist()):
            cosine = torch.nn.functional.cosine_similarity(
                batch32_decode_logits[row], batch32_oracle_logits[group], dim=0
            )
            batch32_oracle_cosines.append(float(cosine))
        # B1 and B32 select different decode matmul programs; require strong
        # full-vocabulary agreement while the exact greedy invariant above
        # remains the correctness gate for every row.
        assert min(batch32_oracle_cosines) >= 0.99

    if os.environ.get("GEMMA4_NO_HOST_TOKEN_OUT_BENCH") == "1":
        warmups = int(os.environ.get("GEMMA4_NO_HOST_WARMUPS", "5"))
        iterations = int(os.environ.get("GEMMA4_NO_HOST_ITERATIONS", "128"))
        for _ in range(warmups):
            generator.decode_forward(
                torch.zeros((probe_batch, 1), dtype=torch.long),
                torch.full((probe_batch,), 999, dtype=torch.int32),
                page_table=model.state.page_tables,
                kv_cache=model.state,
                enable_trace=True,
            )
        ttnn.synchronize_device(target_mesh)
        start_s = time.perf_counter()
        for _ in range(iterations):
            generator.decode_forward(
                torch.zeros((probe_batch, 1), dtype=torch.long),
                torch.full((probe_batch,), 999, dtype=torch.int32),
                page_table=model.state.page_tables,
                kv_cache=model.state,
                enable_trace=True,
            )
        ttnn.synchronize_device(target_mesh)
        elapsed_s = time.perf_counter() - start_s
        final_position = int(ttnn.to_torch(ttnn.get_device_tensors(trace.current_pos)[0]).reshape(-1)[0])
        report = {
            "profile": {1: "P150", 2: "P150x2", 4: "P150x4"}[tp_size],
            "boundary": "split traced model decode plus on-device sampling and token feedback",
            "full_stack": full_stack,
            "prefill_prompt_len": prompt_len,
            "measurement_start_position": prompt_len + 1 + warmups,
            "measurement_end_position_exclusive": final_position,
            "warmups": warmups,
            "iterations": iterations,
            "elapsed_s": elapsed_s,
            "ms_per_token": elapsed_s * 1000.0 / iterations,
            "decode_t_s_u": iterations / elapsed_s,
            "trace_replays": generator.trace_counters.replays,
            "final_position": final_position,
            "token_refreshes": generator.trace_counters.token_refreshes,
            "position_refreshes": generator.trace_counters.position_refreshes,
            "rope_refreshes": generator.trace_counters.rope_refreshes,
            "page_table_refreshes": generator.trace_counters.page_table_refreshes,
            "token_readbacks": generator.trace_counters.token_readbacks,
            "synchronizations": generator.trace_counters.synchronizations,
        }
        output_dir = Path(os.environ.get("GEMMA4_FULL_MODEL_PROBE_OUTPUT_DIR", "/tmp/gemma4_full_model_evidence"))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"token_out_trace_tp{tp_size}.json").write_text(json.dumps(report, indent=2) + "\n")
        print("GEMMA4_NO_HOST_TOKEN_OUT_RESULT=" + json.dumps(report, sort_keys=True))
        return

    if batch32:
        original_trace_id = trace.model_trace_id
        token_address = trace.token_input.buffer_address()
        sampled_address = trace.sampled_tokens.buffer_address()
        repeated = generator.decode_forward(
            torch.ones((probe_batch, 1), dtype=torch.long),
            torch.full((probe_batch,), prompt_len, dtype=torch.int32),
            page_table=model.state.page_tables,
            kv_cache=model.state,
            active_mask=torch.ones(probe_batch, dtype=torch.bool),
            enable_trace=True,
        )
        repeated_rows = generator._read_tokens(repeated, probe_batch).tolist()
        repeated_values = batch32_decode_logits[
            torch.arange(probe_batch), torch.tensor(repeated_rows, dtype=torch.long)
        ]
        assert torch.equal(repeated_values, batch_max)
        assert trace.model_trace_id == original_trace_id
        assert trace.token_input.buffer_address() == token_address
        assert trace.sampled_tokens.buffer_address() == sampled_address
        expected_final_position = prompt_len + 1
    elif long_context:
        # The prompt already leaves exactly one legal decode position. A
        # second replay must be rejected on the host before touching device
        # RoPE or cache state.
        with expect_error(ValueError, "outside slot capacity"):
            generator.decode_forward(
                torch.zeros((probe_batch, 1), dtype=torch.long),
                torch.full((probe_batch,), 999, dtype=torch.int32),
                page_table=model.state.page_tables,
                kv_cache=model.state,
                enable_trace=True,
            )
        expected_final_position = PROFILE_CONTEXT_LIMITS[tp_size]
    else:
        # A steady-state replay consumes device token feedback and ignores the
        # deliberately stale host token/position without another refresh.
        if os.environ.get("GEMMA4_FULL_MODEL_DEVICE_PROFILE") == "1":
            from tracy import signpost

            signpost("FULL_MODEL_REDUCED_DECODE")
        generator.decode_forward(
            torch.zeros((probe_batch, 1), dtype=torch.long),
            torch.full((probe_batch,), 999, dtype=torch.int32),
            page_table=model.state.page_tables,
            kv_cache=model.state,
            enable_trace=True,
        )
        if os.environ.get("GEMMA4_FULL_MODEL_DEVICE_PROFILE") == "1":
            ttnn.synchronize_device(target_mesh)
            signpost("FULL_MODEL_REDUCED_DECODE_END")
        expected_final_position = prompt_len + 2
    expected_refreshes = 2 if batch32 else 1
    assert generator.trace_counters.token_refreshes == expected_refreshes
    final_positions = ttnn.to_torch(ttnn.get_device_tensors(trace.current_pos)[0]).reshape(-1)
    assert int(final_positions[0]) == expected_final_position
    if batch32:
        assert final_positions[:probe_batch].tolist() == [expected_final_position] * probe_batch

    output_dir = os.environ.get("GEMMA4_FULL_MODEL_PROBE_OUTPUT_DIR")
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        report = {
            "verdict": "pass",
            "profile": {1: "P150", 2: "P150x2", 4: "P150x4"}[tp_size],
            "tp_size": tp_size,
            "mesh_shape": list(target_mesh.shape),
            "checkpoint_revision": "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
            "real_weights": True,
            "layer_indices": model.layer_indices,
            "prompt_len": prompt_len,
            "max_seq_len": model.max_seq_len,
            "terminal_weight_dtype": str(model.terminal_weight_dtype),
            "kv_cache_dtype": str(model.kv_cache_dtype),
            "cache_local_heads": sorted({spec.local_kv_heads for spec in model.cache_specs}),
            "logits_local_shape": list(logits.shape),
            "sampled_shape": list(sampled.shape),
            "trace_counters": vars(generator.trace_counters),
            "final_position": int(ttnn.to_torch(ttnn.get_device_tensors(trace.current_pos)[0]).reshape(-1)[0]),
        }
        if batch32:
            report["batch_correctness"] = {
                "prompt_group_by_row": batch32_prompt_groups.tolist(),
                "b1_oracle_token_by_group": batch32_oracle_tokens,
                "first_b32_tokens": sampled_rows,
                "same_position_replay_tokens": repeated_rows,
                "all_rows_select_global_max": torch.equal(sampled_values, batch_max),
                "same_position_replay_selects_global_max": torch.equal(repeated_values, batch_max),
                "b1_b32_min_logits_cosine": min(batch32_oracle_cosines),
                "b1_b32_logits_cosine_by_row": batch32_oracle_cosines,
                "tie_break_ids_may_differ": True,
                "stable_trace_id": str(original_trace_id),
                "stable_token_input_address": token_address,
                "stable_sampled_output_address": sampled_address,
                "all_device_positions": final_positions[:probe_batch].tolist(),
            }
        evidence_kind = (
            "long_context_probe"
            if long_context
            else "batch32_probe"
            if batch32
            else "full_stack_probe"
            if full_stack
            else "reduced_probe"
        )
        (Path(output_dir) / f"{evidence_kind}_tp{tp_size}.json").write_text(json.dumps(report, indent=2) + "\n")

    if os.environ.get("GEMMA4_SAMPLED_TRACE_PROBE") == "1":
        prior_greedy_token = int(generator._read_tokens(trace.sampled_tokens, 1)[0])
        sampled = generator.decode_forward(
            torch.zeros((probe_batch, 1), dtype=torch.long),
            torch.full((probe_batch,), 34, dtype=torch.int32),
            page_table=model.state.page_tables,
            kv_cache=model.state,
            enable_trace=True,
            top_k=8,
            top_p=0.95,
            temperature=0.8,
            seeds=42,
        )
        assert tuple(sampled.shape) == (1, 1, 1, 32)
        sampled_traces = [entry for entry in generator._trace_cache.values() if not entry.sampling_spec.greedy]
        assert len(sampled_traces) == 1
        sampled_trace = sampled_traces[0]
        assert sampled_trace.sampling_trace_id is not None
        assert all(tensor is not None for tensor in sampled_trace.sampling_params)
        assert sampled_trace.sampled_tokens.buffer_address() == sampled_trace.token_input.buffer_address()
        stale_host_transition_logits = generator._gather_logits_to_torch(sampled_trace.logits, logical_len=1).clone()
        assert generator.trace_counters.device_feedback_reuses == 1
        stale_host_sampled_token = int(generator._read_tokens(sampled, 1)[0])
        generator.decode_forward(
            torch.zeros((probe_batch, 1), dtype=torch.long),
            torch.full((probe_batch,), 999, dtype=torch.int32),
            page_table=model.state.page_tables,
            kv_cache=model.state,
            enable_trace=True,
        )
        stale_host_greedy_trace = next(entry for entry in generator._trace_cache.values() if entry.sampling_spec.greedy)
        stale_host_reverse_logits = generator._gather_logits_to_torch(
            stale_host_greedy_trace.logits, logical_len=1
        ).clone()
        assert generator.trace_counters.device_feedback_reuses == 2

        # Re-run the same state transition from reset, this time supplying the
        # correct host tokens. Both transition logits must agree: the first
        # run's deliberately stale host zeros were replaced by prior device
        # feedback while k=1 -> k=8 -> k=1 recaptured each trace pair.
        generator.reset()
        model.prefill_forward(tokens, state=model.state, prompt_lens=[prompt_len], position_ids=positions)
        control_first = generator.decode_forward(
            torch.ones((probe_batch, 1), dtype=torch.long),
            torch.full((probe_batch,), prompt_len, dtype=torch.int32),
            page_table=model.state.page_tables,
            kv_cache=model.state,
            enable_trace=True,
        )
        control_first_token = int(generator._read_tokens(control_first, 1)[0])
        control_second = generator.decode_forward(
            torch.tensor([[control_first_token]], dtype=torch.long),
            torch.full((probe_batch,), prompt_len + 1, dtype=torch.int32),
            page_table=model.state.page_tables,
            kv_cache=model.state,
            enable_trace=True,
        )
        control_greedy_token = int(generator._read_tokens(control_second, 1)[0])
        assert control_greedy_token == prior_greedy_token
        control_sampled = generator.decode_forward(
            torch.tensor([[control_greedy_token]], dtype=torch.long),
            torch.full((probe_batch,), prompt_len + 2, dtype=torch.int32),
            page_table=model.state.page_tables,
            kv_cache=model.state,
            enable_trace=True,
            top_k=8,
            top_p=0.95,
            temperature=0.8,
            seeds=42,
        )
        sampled_trace = next(entry for entry in generator._trace_cache.values() if not entry.sampling_spec.greedy)
        correct_host_transition_logits = generator._gather_logits_to_torch(sampled_trace.logits, logical_len=1)
        control_sampled_token = int(generator._read_tokens(control_sampled, 1)[0])
        assert control_sampled_token == stale_host_sampled_token
        control_greedy = generator.decode_forward(
            torch.tensor([[control_sampled_token]], dtype=torch.long),
            torch.full((probe_batch,), prompt_len + 3, dtype=torch.int32),
            page_table=model.state.page_tables,
            kv_cache=model.state,
            enable_trace=True,
        )
        control_greedy_trace = next(entry for entry in generator._trace_cache.values() if entry.sampling_spec.greedy)
        correct_host_reverse_logits = generator._gather_logits_to_torch(control_greedy_trace.logits, logical_len=1)

        def transition_similarity(stale_logits, control_logits):
            stale_flat = stale_logits.float().reshape(-1)
            control_flat = control_logits.float().reshape(-1)
            return {
                "cosine": float(torch.nn.functional.cosine_similarity(stale_flat, control_flat, dim=0)),
                "top1_equal": int(stale_flat.argmax()) == int(control_flat.argmax()),
                "top100_equal": set(stale_flat.topk(100).indices.tolist())
                == set(control_flat.topk(100).indices.tolist()),
            }

        forward_similarity = transition_similarity(stale_host_transition_logits, correct_host_transition_logits)
        reverse_similarity = transition_similarity(stale_host_reverse_logits, correct_host_reverse_logits)
        for comparison in (forward_similarity, reverse_similarity):
            assert comparison["cosine"] >= 0.999
            assert comparison["top1_equal"]
            assert comparison["top100_equal"]
        transition_feedback_reuses = generator.trace_counters.device_feedback_reuses
        transition_feedback_restores = generator.trace_counters.device_feedback_restores
        assert transition_feedback_reuses == 2
        assert transition_feedback_restores == 2

        # Replace page-table tensor identities without changing greedy
        # sampling. The public stable-address path must consume the live token;
        # compare its logits and output with the same cache position reached
        # using correct host tokens and replacement tables from prefill.
        page_prior_token = int(generator._read_tokens(control_greedy, 1)[0])
        assert (
            replacement_page_tables is not None
            and second_replacement_page_tables is not None
            and original_page_tables is not None
        )
        page_trace_id_before = control_greedy_trace.model_trace_id
        page_sampling_trace_id_before = control_greedy_trace.sampling_trace_id
        page_table_addresses_before = [table.buffer_address() for table in model.state.page_tables]
        page_transition_greedy = generator.decode_forward(
            torch.zeros((probe_batch, 1), dtype=torch.long),
            torch.full((probe_batch,), 999, dtype=torch.int32),
            page_table=replacement_page_tables,
            kv_cache=model.state,
            enable_trace=True,
        )
        page_transition_token = int(generator._read_tokens(page_transition_greedy, 1)[0])
        page_trace = next(entry for entry in generator._trace_cache.values() if entry.sampling_spec.greedy)
        page_transition_logits = generator._gather_logits_to_torch(page_trace.logits, logical_len=1).clone()
        page_transition_feedback_reuses = generator.trace_counters.device_feedback_reuses
        page_transition_refreshes = generator.trace_counters.page_table_refreshes
        assert page_trace.model_trace_id == page_trace_id_before
        assert page_trace.sampling_trace_id == page_sampling_trace_id_before
        assert all(current is original for current, original in zip(model.state.page_tables, original_page_tables))
        assert [table.buffer_address() for table in model.state.page_tables] == page_table_addresses_before
        assert page_transition_feedback_reuses == 2
        assert page_transition_refreshes == 1

        # Reproduce the same five positions with correct host tokens.
        generator.reset()
        model.prefill_forward(tokens, state=model.state, prompt_lens=[prompt_len], position_ids=positions)
        page_control_first = generator.decode_forward(
            torch.ones((probe_batch, 1), dtype=torch.long),
            torch.full((probe_batch,), prompt_len, dtype=torch.int32),
            page_table=replacement_page_tables,
            kv_cache=model.state,
            enable_trace=True,
        )
        page_control_second = generator.decode_forward(
            torch.tensor([[int(generator._read_tokens(page_control_first, 1)[0])]], dtype=torch.long),
            torch.full((probe_batch,), prompt_len + 1, dtype=torch.int32),
            page_table=replacement_page_tables,
            kv_cache=model.state,
            enable_trace=True,
        )
        page_control_sampled = generator.decode_forward(
            torch.tensor([[int(generator._read_tokens(page_control_second, 1)[0])]], dtype=torch.long),
            torch.full((probe_batch,), prompt_len + 2, dtype=torch.int32),
            page_table=replacement_page_tables,
            kv_cache=model.state,
            enable_trace=True,
            top_k=8,
            top_p=0.95,
            temperature=0.8,
            seeds=42,
        )
        page_control_greedy = generator.decode_forward(
            torch.tensor([[int(generator._read_tokens(page_control_sampled, 1)[0])]], dtype=torch.long),
            torch.full((probe_batch,), prompt_len + 3, dtype=torch.int32),
            page_table=replacement_page_tables,
            kv_cache=model.state,
            enable_trace=True,
        )
        page_control_prior_token = int(generator._read_tokens(page_control_greedy, 1)[0])
        assert page_control_prior_token == page_prior_token
        page_control_output = generator.decode_forward(
            torch.tensor([[int(generator._read_tokens(page_control_greedy, 1)[0])]], dtype=torch.long),
            torch.full((probe_batch,), prompt_len + 4, dtype=torch.int32),
            page_table=replacement_page_tables,
            kv_cache=model.state,
            enable_trace=True,
        )
        page_control_token = int(generator._read_tokens(page_control_output, 1)[0])
        assert page_control_token == page_transition_token
        page_control_trace = next(entry for entry in generator._trace_cache.values() if entry.sampling_spec.greedy)
        page_control_logits = generator._gather_logits_to_torch(page_control_trace.logits, logical_len=1)
        page_similarity = transition_similarity(page_transition_logits, page_control_logits)
        assert page_similarity["cosine"] >= 0.999
        assert page_similarity["top1_equal"]
        assert page_similarity["top100_equal"]
        same_source_refreshes = generator.trace_counters.page_table_refreshes
        assert same_source_refreshes == 1

        # Switch to sampled mode, then use uniform fixed logits to prove that a
        # page-table-only stable-address adoption neither resets nor consumes a
        # random draw. The second token after adoption must be
        # exactly the second token from the uninterrupted control stream.
        generator.decode_forward(
            torch.tensor([[page_control_token]], dtype=torch.long),
            torch.full((probe_batch,), prompt_len + 5, dtype=torch.int32),
            page_table=replacement_page_tables,
            kv_cache=model.state,
            enable_trace=True,
            top_k=8,
            top_p=0.95,
            temperature=0.8,
            seeds=42,
        )
        sampled_trace = next(entry for entry in generator._trace_cache.values() if not entry.sampling_spec.greedy)
        sampled_seed_tensor = sampled_trace.sampling_params[3]
        seeded_values = generator._seed_values(sampled_trace.sampling_spec)
        ttnn.fill(
            sampled_trace.logits,
            0.0,
            memory_config=sampled_trace.logits.memory_config(),
            output_tensor=sampled_trace.logits,
        )
        ttnn.synchronize_device(target_mesh)

        generator._refresh_device_input(sampled_seed_tensor, seeded_values)
        uninterrupted_rng_tokens = []
        for _ in range(2):
            ttnn.execute_trace(target_mesh, sampled_trace.sampling_trace_id, cq_id=0, blocking=True)
            uninterrupted_rng_tokens.append(int(generator._read_tokens(sampled_trace.sampled_tokens, 1)[0]))
        generator._refresh_device_input(sampled_seed_tensor, seeded_values)
        ttnn.execute_trace(target_mesh, sampled_trace.sampling_trace_id, cq_id=0, blocking=True)
        recapture_first_token = int(generator._read_tokens(sampled_trace.sampled_tokens, 1)[0])
        assert recapture_first_token == uninterrupted_rng_tokens[0]

        sampled_model_trace_id = sampled_trace.model_trace_id
        sampled_sampling_trace_id = sampled_trace.sampling_trace_id
        generator._state_from_args(
            kv_cache=model.state,
            page_table=second_replacement_page_tables,
            batch_size=probe_batch,
        )
        assert sampled_trace.model_trace_id == sampled_model_trace_id
        assert sampled_trace.sampling_trace_id == sampled_sampling_trace_id
        page_adoption_refreshes = generator.trace_counters.page_table_refreshes
        assert page_adoption_refreshes == 2
        ttnn.execute_trace(target_mesh, sampled_trace.sampling_trace_id, cq_id=0, blocking=True)
        page_adoption_second_token = int(generator._read_tokens(sampled_trace.sampled_tokens, 1)[0])
        assert page_adoption_second_token == uninterrupted_rng_tokens[1]

        def fixed_logits_sequence(seed_values):
            generator._refresh_device_input(sampled_seed_tensor, seed_values)
            sequence = []
            for _ in range(16):
                ttnn.execute_trace(target_mesh, sampled_trace.sampling_trace_id, cq_id=0, blocking=True)
                sequence.append(int(generator._read_tokens(sampled_trace.sampled_tokens, 1)[0]))
            armed = ttnn.to_torch(ttnn.get_device_tensors(sampled_seed_tensor)[0]).reshape(-1)
            assert int(armed[0]) == 2**32 - 1
            return sequence

        sequence_a = fixed_logits_sequence(seeded_values)
        sequence_b = fixed_logits_sequence(seeded_values)
        different_seed_values = seeded_values.clone()
        different_seed_values[0] += 1
        sequence_c = fixed_logits_sequence(different_seed_values)
        assert sequence_a == sequence_b
        assert len(set(sequence_a)) > 1
        assert sequence_c != sequence_a
        output_dir = Path(os.environ.get("GEMMA4_FULL_MODEL_PROBE_OUTPUT_DIR", "/tmp/gemma4_full_model_evidence"))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"sampling_rng_tp{tp_size}.json").write_text(
            json.dumps(
                {
                    "verdict": "pass",
                    "profile": {1: "P150", 2: "P150x2", 4: "P150x4"}[tp_size],
                    "seed": int(seeded_values[0]),
                    "same_seed_first": sequence_a,
                    "same_seed_second": sequence_b,
                    "different_seed": sequence_c,
                    "same_seed_reproducible": sequence_a == sequence_b,
                    "stream_advances": len(set(sequence_a)) > 1,
                    "different_seed_differs": sequence_c != sequence_a,
                    "continuation_sentinel": 2**32 - 1,
                    "sampling_transition_device_feedback": {
                        "greedy_to_sampled_with_stale_host_token": 0,
                        "sampled_to_greedy_with_stale_host_token": 0,
                        "prior_device_token": prior_greedy_token,
                        "correct_host_control_token": control_greedy_token,
                        "sampled_device_token": stale_host_sampled_token,
                        "correct_host_control_sampled_token": control_sampled_token,
                        "greedy_to_sampled": forward_similarity,
                        "sampled_to_greedy": reverse_similarity,
                        "device_feedback_reuses": transition_feedback_reuses,
                        "device_feedback_restores": transition_feedback_restores,
                    },
                    "page_table_identity_adoption": {
                        "stale_host_token": 0,
                        "prior_device_token": page_prior_token,
                        "correct_host_control_prior_token": page_control_prior_token,
                        "greedy_token": page_transition_token,
                        "correct_host_control_greedy_token": page_control_token,
                        "logits": page_similarity,
                        "device_feedback_reuses_before_reset": page_transition_feedback_reuses,
                        "page_table_refreshes_before_reset": page_transition_refreshes,
                        "sampled_rng_uninterrupted": uninterrupted_rng_tokens,
                        "sampled_rng_before_adoption": recapture_first_token,
                        "sampled_rng_after_adoption": page_adoption_second_token,
                        "sampled_rng_continuity": page_adoption_second_token == uninterrupted_rng_tokens[1],
                        "model_trace_id_stable": str(page_trace.model_trace_id) == str(page_trace_id_before),
                        "sampling_trace_id_stable": str(page_trace.sampling_trace_id)
                        == str(page_sampling_trace_id_before),
                        "page_table_addresses_stable": [table.buffer_address() for table in model.state.page_tables]
                        == page_table_addresses_before,
                        "same_source_repeated_decode_copies": same_source_refreshes,
                        "different_source_total_copies": page_adoption_refreshes,
                    },
                },
                indent=2,
            )
            + "\n"
        )
        greedy_trace_id = trace.model_trace_id
        generator.decode_forward(
            torch.zeros((probe_batch, 1), dtype=torch.long),
            torch.full((probe_batch,), 34, dtype=torch.int32),
            page_table=model.state.page_tables,
            kv_cache=model.state,
            enable_trace=True,
        )
        assert len(generator._trace_cache) == 1
        recaptured_greedy = next(entry for entry in generator._trace_cache.values() if entry.sampling_spec.greedy)
        assert recaptured_greedy.model_trace_id != greedy_trace_id


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 67_108_864}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_reduced_mixed_prompt_and_inactive_slot_probe(mesh_device):
    if os.environ.get("GEMMA4_MIXED_PROBE") != "1":
        pytest.skip("set GEMMA4_MIXED_PROBE=1 for the serialized TP4 mixed-prompt probe")
    model = Gemma4FullModel(
        mesh_device=mesh_device,
        hf_config=_config(),
        state_dict=_load_probe_state(full_stack=False),
        max_seq_len=128,
        max_batch_size=2,
        layer_indices=[0, 5],
        tensor_cache_path="/tmp/gemma4_full_model_probe_cache",
    )
    generator = Gemma4Generator(model, tokenizer=None, sampling_mode="device")
    tokens = torch.ones((2, 64), dtype=torch.long)
    outputs = generator.prefill_forward(
        tokens,
        page_table=model.state.page_tables,
        kv_cache=model.state,
        prompt_lens=[33, 47],
    )
    assert tuple(outputs.shape) == (1, 1, 2, 65_536)
    table_addresses = [table.buffer_address() for table in model.state.page_tables]
    scheduler_tables = [ttnn.to_torch(ttnn.get_device_tensors(table)[0]).clone() for table in model.state.page_tables]
    changed_scheduler_tables = [table.clone() for table in scheduler_tables]
    for table in changed_scheduler_tables:
        table[0, 0], table[1, 0] = table[1, 0].clone(), table[0, 0].clone()

    cache_before_capture = [
        [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(cache)]
        for pair in model.state.kv_cache
        for cache in pair
    ]
    capture_spec = generator._sampling_spec(2)
    generator._get_or_capture_decode_trace(
        torch.ones((2, 1), dtype=torch.long),
        torch.tensor([33, -1], dtype=torch.int32),
        state=model.state,
        sampling_mode="device",
        sampling_spec=capture_spec,
    )
    ttnn.synchronize_device(mesh_device)
    cache_after_capture = [
        [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(cache)]
        for pair in model.state.kv_cache
        for cache in pair
    ]
    assert all(
        torch.equal(before, after)
        for before_shards, after_shards in zip(cache_before_capture, cache_after_capture)
        for before, after in zip(before_shards, after_shards)
    )

    sampled = generator.decode_forward(
        torch.ones((2, 1), dtype=torch.long),
        torch.tensor([33, -1], dtype=torch.int32),
        page_table=model.state.page_tables,
        kv_cache=model.state,
        active_mask=torch.tensor([True, False]),
        enable_trace=True,
    )
    assert tuple(sampled.shape) == (1, 1, 1, 32)
    trace = next(iter(generator._trace_cache.values()))
    original_trace_id = trace.model_trace_id
    sampled_after_first = int(ttnn.to_torch(ttnn.get_device_tensors(sampled)[0]).reshape(-1)[0])
    token_input_address = trace.token_input.buffer_address()
    sampled_output_address = trace.sampled_tokens.buffer_address()
    current = ttnn.to_torch(ttnn.get_device_tensors(trace.current_pos)[0]).reshape(-1)
    assert current[:3].tolist() == [34, -1, -1]
    generator.decode_forward(
        torch.zeros((2, 1), dtype=torch.long),
        torch.tensor([999, -1], dtype=torch.int32),
        page_table=model.state.page_tables,
        kv_cache=model.state,
        enable_trace=True,
    )
    current = ttnn.to_torch(ttnn.get_device_tensors(trace.current_pos)[0]).reshape(-1)
    assert current[:3].tolist() == [35, -1, -1]
    assert [table.buffer_address() for table in model.state.page_tables] == table_addresses
    assert generator.trace_counters.token_refreshes == 1
    assert generator.trace_counters.page_table_refreshes == 0

    # A scheduler-boundary mapping change is copied exactly once into the same
    # stable device tensors; subsequent tokens reuse those contents and trace.
    generator.refresh_page_tables(changed_scheduler_tables, state=model.state)
    changed_output = generator.decode_forward(
        torch.zeros((2, 1), dtype=torch.long),
        torch.tensor([35, -1], dtype=torch.int32),
        page_table=model.state.page_tables,
        kv_cache=model.state,
        enable_trace=True,
    )
    changed_trace = next(iter(generator._trace_cache.values()))
    assert [table.buffer_address() for table in model.state.page_tables] == table_addresses
    copied_tables = [ttnn.to_torch(ttnn.get_device_tensors(table)[0]) for table in model.state.page_tables]
    assert all(torch.equal(actual, expected) for actual, expected in zip(copied_tables, changed_scheduler_tables))
    assert generator.trace_counters.page_table_refreshes == 1
    assert changed_trace.model_trace_id == original_trace_id
    final_output = generator.decode_forward(
        torch.zeros((2, 1), dtype=torch.long),
        torch.tensor([999, -1], dtype=torch.int32),
        page_table=model.state.page_tables,
        kv_cache=model.state,
        enable_trace=True,
    )
    assert generator.trace_counters.page_table_refreshes == 1
    assert tuple(changed_output.shape) == tuple(final_output.shape) == (1, 1, 1, 32)

    # Scheduler activity changes reuse the same trace but refresh stable
    # token/position inputs once, including reactivation and all-inactive.
    switched = generator.decode_forward(
        torch.tensor([[0], [7]], dtype=torch.long),
        torch.tensor([-1, 47], dtype=torch.int32),
        page_table=model.state.page_tables,
        kv_cache=model.state,
        active_mask=torch.tensor([False, True]),
        enable_trace=True,
    )
    switched_trace = next(iter(generator._trace_cache.values()))
    assert switched_trace.model_trace_id == original_trace_id
    switched_positions = ttnn.to_torch(ttnn.get_device_tensors(switched_trace.current_pos)[0]).reshape(-1)
    assert switched_positions[:3].tolist() == [-1, 48, -1]
    assert generator.trace_counters.token_refreshes == 2
    generator.decode_forward(
        torch.zeros((2, 1), dtype=torch.long),
        torch.tensor([999, 999], dtype=torch.int32),
        page_table=model.state.page_tables,
        kv_cache=model.state,
        enable_trace=True,
    )
    steady_positions = ttnn.to_torch(ttnn.get_device_tensors(switched_trace.current_pos)[0]).reshape(-1)
    assert steady_positions[:3].tolist() == [-1, 49, -1]
    assert generator.trace_counters.token_refreshes == 2
    generator.decode_forward(
        torch.zeros((2, 1), dtype=torch.long),
        torch.tensor([-1, -1], dtype=torch.int32),
        page_table=model.state.page_tables,
        kv_cache=model.state,
        active_mask=torch.tensor([False, False]),
        enable_trace=True,
    )
    inactive_positions = ttnn.to_torch(ttnn.get_device_tensors(switched_trace.current_pos)[0]).reshape(-1)
    assert inactive_positions[:3].tolist() == [-1, -1, -1]
    device_transition_counters = vars(generator.trace_counters).copy()

    # A teacher trace retained across reset/prefill must restart from the new
    # request's positions rather than resume its previous device counters.
    generator.reset()
    generator.prefill_forward(
        tokens,
        page_table=model.state.page_tables,
        kv_cache=model.state,
        prompt_lens=[33, 47],
    )
    generator.decode_forward(
        torch.ones((2, 1), dtype=torch.long),
        torch.tensor([33, 47], dtype=torch.int32),
        page_table=model.state.page_tables,
        kv_cache=model.state,
        sampling_mode="teacher",
        enable_trace=True,
    )
    teacher_trace = next(iter(generator._trace_cache.values()))
    teacher_trace_id = teacher_trace.model_trace_id
    generator.reset()
    generator.prefill_forward(
        tokens,
        page_table=model.state.page_tables,
        kv_cache=model.state,
        prompt_lens=[35, 45],
    )
    generator.decode_forward(
        torch.ones((2, 1), dtype=torch.long),
        torch.tensor([35, 45], dtype=torch.int32),
        page_table=model.state.page_tables,
        kv_cache=model.state,
        sampling_mode="teacher",
        enable_trace=True,
    )
    reused_teacher_trace = next(iter(generator._trace_cache.values()))
    teacher_positions = ttnn.to_torch(ttnn.get_device_tensors(reused_teacher_trace.current_pos)[0]).reshape(-1)
    assert reused_teacher_trace.model_trace_id == teacher_trace_id
    assert teacher_positions[:3].tolist() == [36, 46, -1]
    assert generator.trace_counters.token_refreshes == 1
    assert generator.trace_counters.position_refreshes == 1

    output_path = os.environ.get("GEMMA4_MIXED_STATE_OUTPUT")
    if output_path:
        report = {
            "verdict": "pass",
            "path": "standalone traced Gemma4Generator decode",
            "stale_host_inputs": {
                "token_supplied_on_replay": 0,
                "position_supplied_on_replay": 999,
                "prior_sampled_device_token": sampled_after_first,
                "token_refreshes_before_scheduler_transitions": 1,
                "persistent_token_input_address": token_input_address,
                "persistent_sampled_output_address": sampled_output_address,
                "device_feedback_aliases_trace_input": token_input_address == sampled_output_address,
            },
            "positions": {
                "initial_active_position": 33,
                "after_first_replay": 34,
                "after_stale_position_replay": 35,
                "after_four_replays": 37,
                "after_active_row_switch": switched_positions[:2].tolist(),
                "after_steady_switched_replay": steady_positions[:2].tolist(),
                "after_all_inactive_replay": inactive_positions[:2].tolist(),
            },
            "page_tables": {
                "initial_bind_refresh_count": 0,
                "unchanged_replay_refresh_count": 0,
                "after_mutated_scheduler_table_refresh_count": 1,
                "stable_refresh_count_after_reuse": generator.trace_counters.page_table_refreshes,
                "stable_device_addresses_before": table_addresses,
                "stable_device_addresses_after": [table.buffer_address() for table in model.state.page_tables],
                "scheduler_first_entries_before": [int(table[0, 0]) for table in scheduler_tables],
                "scheduler_first_entries_after": [int(table[0, 0]) for table in changed_scheduler_tables],
                "device_first_entries_after": [int(table[0, 0]) for table in copied_tables],
                "device_contents_match_mutated_scheduler_tables": all(
                    torch.equal(actual, expected) for actual, expected in zip(copied_tables, changed_scheduler_tables)
                ),
                "generator_recaptures_from_table_identity": generator.trace_counters.page_table_refreshes,
                "changed_and_reused_output_shapes": [list(changed_output.shape), list(final_output.shape)],
            },
            "device_transition_counters": device_transition_counters,
            "teacher_trace_reuse": {
                "same_trace_id": reused_teacher_trace.model_trace_id == teacher_trace_id,
                "second_request_positions_after_replay": teacher_positions[:2].tolist(),
                "second_request_counters": vars(generator.trace_counters),
            },
        }
        Path(output_path).write_text(json.dumps(report, indent=2) + "\n")


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 67_108_864}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_public_nonaligned_prefill_preserves_logical_cache_tail(mesh_device):
    if os.environ.get("GEMMA4_LOGICAL_TAIL_PROBE") != "1":
        pytest.skip("set GEMMA4_LOGICAL_TAIL_PROBE=1 for the serialized TP4 logical-tail probe")
    model = Gemma4FullModel(
        mesh_device=mesh_device,
        hf_config=_config(),
        state_dict=_load_probe_state(full_stack=False),
        max_seq_len=2_048,
        max_batch_size=1,
        layer_indices=[0, 5],
        tensor_cache_path="/tmp/gemma4_full_model_probe_cache",
    )
    generator = Gemma4Generator(model, tokenizer=None, sampling_mode="device")
    results = []
    for logical_len in (1_025, 1_055):
        public_state = model.allocate_state(max_batch_size=1)
        direct_state = model.allocate_state(max_batch_size=1)
        logical_tokens = 1 + (torch.arange(logical_len, dtype=torch.long).reshape(1, -1) % 255)
        generator.prefill_forward(
            logical_tokens,
            page_table=public_state.page_tables,
            kv_cache=public_state,
            prompt_lens=[logical_len],
        )
        tt_tokens = generator._host_tokens_to_device(logical_tokens)
        tt_positions = generator._positions_to_device(
            torch.arange(logical_len, dtype=torch.int32).reshape(1, -1), dtype=ttnn.uint32
        )
        model.prefill_forward(
            tt_tokens,
            state=direct_state,
            prompt_lens=[logical_len],
            position_ids=tt_positions,
        )
        ttnn.synchronize_device(mesh_device)
        cache_equal = True
        for public_pair, direct_pair in zip(public_state.kv_cache, direct_state.kv_cache):
            for public_cache, direct_cache in zip(public_pair, direct_pair):
                public_shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(public_cache)]
                direct_shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(direct_cache)]
                cache_equal &= all(torch.equal(left, right) for left, right in zip(public_shards, direct_shards))
        assert cache_equal

        decode_tokens = generator._host_tokens_to_device(torch.ones((1, 1), dtype=torch.long), rank4=True)
        current_pos = generator._positions_to_device(torch.tensor([logical_len], dtype=torch.int32))
        position_ids = generator._positions_to_device(torch.tensor([logical_len], dtype=torch.int32), dtype=ttnn.uint32)
        public_decode = model.decode_forward(
            decode_tokens,
            state=public_state,
            current_pos=current_pos,
            position_ids=position_ids,
            batch_size=1,
        )
        direct_decode = model.decode_forward(
            decode_tokens,
            state=direct_state,
            current_pos=current_pos,
            position_ids=position_ids,
            batch_size=1,
        )
        public_host = generator._gather_logits_to_torch(public_decode, logical_len=1)
        direct_host = generator._gather_logits_to_torch(direct_decode, logical_len=1)
        decode_equal = torch.equal(public_host, direct_host)
        assert decode_equal
        results.append(
            {"logical_len": logical_len, "cache_bitwise_equal": cache_equal, "decode_bitwise_equal": decode_equal}
        )

    output_path = os.environ.get("GEMMA4_LOGICAL_TAIL_OUTPUT")
    if output_path:
        Path(output_path).write_text(
            json.dumps({"verdict": "pass", "profile": "P150x4", "results": results}, indent=2) + "\n"
        )
