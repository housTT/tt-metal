# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Contract and reduced real-weight gates for the Qwen3.8 full model."""

from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.demo.full_model import (
    run_autoregressive,
    run_prefill_check,
    run_qualitative_suite,
    run_teacher_forcing,
    write_autoregressive_artifacts,
    write_report,
)
from models.autoports.qwen_qwen3_8_flash_next.tests import harness as H
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import Qwen38Generator, build_generator
from models.autoports.qwen_qwen3_8_flash_next.tt.model import REQUIRED_L1_SMALL_SIZE, VOCAB_SIZE, Qwen38FullModel
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import FABRIC_PACKET_BYTES, MultichipDecoder


def _device_params(*, trace_region_size=1_073_741_824, l1_small_size=REQUIRED_L1_SMALL_SIZE):
    router = ttnn._ttnn.fabric.FabricRouterConfig()
    router.max_packet_payload_size_bytes = FABRIC_PACKET_BYTES
    return {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "fabric_router_config": router,
        "l1_small_size": l1_small_size,
        "trace_region_size": trace_region_size,
    }


def test_generator_interface_and_policy_are_explicit():
    assert tuple(inspect.signature(build_generator).parameters) == ("model_dir", "mesh_device", "kwargs")
    generate = inspect.signature(Qwen38Generator.generate).parameters
    assert generate["enable_trace"].kind is inspect.Parameter.KEYWORD_ONLY
    assert generate["next_input"].kind is inspect.Parameter.KEYWORD_ONLY
    assert generate["sampling_mode"].default == "device"
    source = inspect.getsource(Qwen38FullModel)
    assert "MultichipDecoder.from_checkpoint_host_backed" in source
    assert "HostBackedSegmentedDecodeTrace" in source
    assert "tt_out_tok=state.token_input" in source
    assert "ttnn.plus_one(state.current_pos" in source
    assert "Sampling1D" in source
    assert "argmax" not in inspect.getsource(Qwen38FullModel.decode_token_out_traced)
    assert Qwen38Generator.required_device_params["l1_small_size"] == REQUIRED_L1_SMALL_SIZE


def test_autoregressive_writer_emits_runner_contract(tmp_path):
    report = {
        "prompt_tokens": 3,
        "generation_tokens": 2,
        "hf_tokens": [7, 8],
        "tt_tokens": [7, 9],
        "hf_completion": "HF completion",
        "tt_completion": "TT completion",
    }
    write_autoregressive_artifacts(report, tmp_path)
    metadata = json.loads((tmp_path / "autoregressive_meta.json").read_text())
    assert metadata["hf"]["token_ids"] == [7, 8]
    assert metadata["tt"]["token_ids"] == [7, 9]
    assert (tmp_path / "hf_completion.txt").read_text() == "HF completion\n"
    assert (tmp_path / "tt_completion.txt").read_text() == "TT completion\n"


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_reduced_real_weight_embedding_to_terminal_gather_smoke(bh_1d_mesh_device, device_params):
    """Isolate endpoint collectives and check real checkpoint top-k numerics."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=128,
        layer_indices=(0,),
    )
    try:
        model.copy_tokens(
            model.new_batch_state([1], request_ids=("endpoint",)),
            torch.tensor([17], dtype=torch.int64),
        )
        residual = model.embed_tokens(model.decode_token_input)
        gathered = model.layers[-1].gather_residual(residual)
        ttnn.synchronize_device(bh_1d_mesh_device)
        assert tuple(gathered.shape) == (1, 1, 1, 10_240)
        ttnn.deallocate(gathered)

        logits = model.project_logits(residual)
        tt_logits = model.logits_to_torch(logits).reshape(-1)
        ttnn.deallocate(residual)
        ttnn.deallocate(logits)

        prefix = "model.language_model"
        hidden = model.checkpoint.tensor(f"{prefix}.embed_tokens.weight")[17].float()
        hyper = hidden.repeat(4)
        groups = hyper.reshape(4, 2_560)
        groups = groups * torch.rsqrt(groups.square().mean(-1, keepdim=True) + 1e-6)
        norm_weight = model.checkpoint.tensor(f"{prefix}.hyper_connection_mixer.hc_norm.weight").float()
        normed = groups.flatten() * (1.0 + norm_weight)
        down = model.checkpoint.tensor(f"{prefix}.hyper_connection_mixer.input_mix_weight_down.weight").float()
        up = model.checkpoint.tensor(f"{prefix}.hyper_connection_mixer.input_mix_weight_up.weight").float()
        mixing = torch.sigmoid(
            torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(normed, down) / 4.0), up)
        ).reshape(4, 2_560)
        final_hidden = (mixing * normed.reshape(4, 2_560)).mean(0)
        lm_head = model.checkpoint.tensor("lm_head.weight").float()
        reference = torch.mv(lm_head, final_hidden)
        reference_top100 = set(torch.topk(reference, 100).indices.tolist())
        tt_top100 = torch.topk(tt_logits, 100).indices.tolist()
        assert int(torch.argmax(tt_logits)) in reference_top100
        assert len(reference_top100.intersection(tt_top100)) >= 98
    finally:
        model.close(best_effort=True)


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_reduced_real_weight_full_model_endpoint_smoke(bh_1d_mesh_device, device_params):
    """One real optimized layer plus the real embedding/final-mix/LM head."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=128,
        layer_indices=(0,),
    )
    try:
        state = model.new_batch_state([1], request_ids=("reduced-real",))
        logits = model.prefill_forward(torch.tensor([[17]], dtype=torch.int64), state=state)
        ttnn.synchronize_device(bh_1d_mesh_device)
        host = model.logits_to_torch(logits)
        assert tuple(host.shape) == (1, 1, 1, VOCAB_SIZE)
        assert torch.isfinite(host).all()
        ttnn.deallocate(logits)
    finally:
        model.close(best_effort=True)


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_reduced_real_weight_split_greedy_trace_contract(bh_1d_mesh_device, device_params):
    """Prove model/sampler split traces, direct feedback, positions, and page refresh."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=128,
        layer_indices=(0,),
    )
    try:
        state = model.new_batch_state([3], request_ids=("split-trace",))
        model.set_sampling_params(top_k=1, top_p=0.0, temperature=1.0)
        prefill_logits = model.prefill_forward(torch.tensor([[17, 29, 31]], dtype=torch.int64), state=state)
        prefill_argmax = model.logits_to_torch(prefill_logits).reshape(-1).argmax().reshape(1)
        first = model.sample_logits(prefill_logits, state)
        first_host = model.sampled_tokens_to_torch(first, state)
        assert torch.equal(first_host, prefill_argmax)
        ttnn.deallocate(prefill_logits)

        _, trace_token = model.decode_token_out_traced(state, first_host.reshape(1, 1))
        ttnn.synchronize_device(bh_1d_mesh_device)
        capture_token = model.sampled_tokens_to_torch(trace_token, state)
        capture_position = ttnn.to_torch(ttnn.get_device_tensors(state.current_pos)[0]).reshape(-1)
        assert capture_position.tolist() == [4]
        copies_after_capture = (state.token_host_copies, state.position_host_copies)

        assert model.update_page_table(state, state.page_table_host.clone()) is False
        changed_pages = state.page_table_host.flip(-1)
        assert model.update_page_table(state, changed_pages) is True
        assert state.page_table_unchanged_skips == 1
        assert model.trace_page_table_changes == 1

        # Inspect the regenerated terminal tensor before its consumer trace is
        # allowed to reuse that corruptible address, then run sampling alone.
        trace_logits = model.replay_model_only_traced(state, capture_token.reshape(1, 1))
        ttnn.synchronize_device(bh_1d_mesh_device)
        replay_argmax = model.logits_to_torch(trace_logits).reshape(-1).argmax().reshape(1)
        ttnn.execute_trace(bh_1d_mesh_device, model.sampling_trace_id, cq_id=0, blocking=False)
        model.advance_positions_traced()
        ttnn.synchronize_device(bh_1d_mesh_device)
        trace_token = state.token_input
        replay_token = model.sampled_tokens_to_torch(trace_token, state)
        assert torch.equal(replay_token, replay_argmax)
        replay_position = ttnn.to_torch(ttnn.get_device_tensors(state.current_pos)[0]).reshape(-1)
        assert replay_position.tolist() == [5]
        assert (state.token_host_copies, state.position_host_copies) == copies_after_capture
        assert model.trace_replays == 0
    finally:
        model.close(best_effort=True)


@pytest.mark.skipif(os.getenv("RUN_QWEN38_SAMPLER_AB") != "1", reason="explicit sampler strategy benchmark")
@pytest.mark.parametrize("device_params", [_device_params(trace_region_size=268_435_456)], indirect=True)
def test_reduced_real_weight_split_greedy_sampler_strategy_ab(bh_1d_mesh_device, device_params, record_property):
    """Compare semantically greedy Sampling1D argmax and local-topk paths."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=128,
        layer_indices=(0,),
    )
    try:
        state = model.new_batch_state([3], request_ids=("sampler-ab",))
        logits = model.prefill_forward(torch.tensor([[17, 29, 31]], dtype=torch.int64), state=state)
        expected = model.logits_to_torch(logits).reshape(-1).argmax().reshape(1)
        model.set_sampling_params(top_k=1, top_p=0.0, temperature=1.0)
        evidence = {}
        for name, force_argmax in (("full_vocab_argmax", True), ("local_top32_k1", False)):
            model._sampling_force_argmax = force_argmax
            model.sample_logits(logits, state)
            ttnn.synchronize_device(bh_1d_mesh_device)
            samples = []
            started = time.perf_counter()
            for _ in range(7):
                sampled = model.sample_logits(logits, state)
                ttnn.synchronize_device(bh_1d_mesh_device)
                samples.append(model.sampled_tokens_to_torch(sampled, state))
            seconds = time.perf_counter() - started
            assert all(torch.equal(sample, expected) for sample in samples)
            evidence[name] = {"iterations": 7, "total_seconds": seconds, "mean_seconds": seconds / 7}
            record_property(f"{name}_mean_seconds", seconds / 7)
        record_property("selected_strategy", "full_vocab_argmax")
        print({"split_greedy_sampler_strategy_ab": evidence})
    finally:
        model.close(best_effort=True)


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_reduced_real_weight_non_greedy_split_trace_contract(bh_1d_mesh_device, device_params):
    """Top-k/top-p trace replay owns seeded sampling, feedback, and mode changes."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=4096,
        layer_indices=(0, 1, 3),
    )
    try:
        state = model.new_batch_state([3], request_ids=("non-greedy-trace",))

        def token_value(tokens):
            return int(model.sampled_tokens_to_torch(tokens, state)[0])

        def position_value():
            shard = ttnn.get_device_tensors(state.current_pos)[0]
            return int(ttnn.to_torch(shard).reshape(-1)[0])

        def topk_set(logits, k=4):
            host = model.logits_to_torch(logits).reshape(-1)
            return set(torch.topk(host, k).indices.tolist())

        def seed_snapshot():
            shard = ttnn.get_device_tensors(model.sampling._seeds)[0]
            return tuple(int(value) for value in ttnn.to_torch(shard).reshape(-1)[: model.max_batch])

        model.set_sampling_params(top_k=4, top_p=0.95, temperature=0.8, seeds=[12345])
        prefill_logits = model.prefill_forward(torch.tensor([[17, 18, 19]]), state=state)
        first_top4 = topk_set(prefill_logits)
        first = model.sample_logits(prefill_logits, state)
        first_host = token_value(first)
        first_seed = seed_snapshot()
        assert first_host in first_top4
        ttnn.deallocate(prefill_logits)

        logits0, tok0 = model.decode_token_out_traced(state, torch.tensor([[first_host]]))
        ttnn.synchronize_device(bh_1d_mesh_device)
        tok0_host = token_value(tok0)
        assert tok0 is state.token_input
        assert model._sampling_force_argmax is False
        assert tok0_host in topk_set(logits0)
        assert position_value() == 4
        capture_seed = seed_snapshot()
        assert capture_seed != first_seed
        copies_after_capture = (state.token_host_copies, state.position_host_copies)

        assert model.update_page_table(state, state.page_table_host.clone()) is False
        changed_pages = state.page_table_host.roll(shifts=1, dims=-1)
        assert model.update_page_table(state, changed_pages) is True
        assert state.page_table_unchanged_skips == 1
        assert model.trace_page_table_changes == 1

        logits1, tok1 = model.decode_token_out_traced(state, torch.tensor([[tok0_host]]))
        ttnn.synchronize_device(bh_1d_mesh_device)
        tok1_host = token_value(tok1)
        assert tok1 is state.token_input
        assert tok1_host in topk_set(logits1)
        assert position_value() == 5
        assert (state.token_host_copies, state.position_host_copies) == copies_after_capture
        assert model.trace_replays == 1
        assert seed_snapshot() != capture_seed

        model.set_sampling_params(top_k=1, top_p=0.0, temperature=1.0)
        assert model._trace_ready is False
        assert model.sampling_trace_id is None
        assert model._sampling_force_argmax is True
        logits2, tok2 = model.decode_token_out_traced(state, torch.tensor([[tok1_host]]))
        ttnn.synchronize_device(bh_1d_mesh_device)
        tok2_host = token_value(tok2)
        assert tok2_host == int(model.logits_to_torch(logits2).reshape(-1).argmax())
        assert position_value() == 6

        model.set_sampling_params(top_k=4, top_p=0.95, temperature=0.8, seeds=[12345])
        assert model._trace_ready is False
        assert model.sampling_trace_id is None
        assert model._sampling_force_argmax is False
        logits3, tok3 = model.decode_token_out_traced(state, torch.tensor([[tok2_host]]))
        ttnn.synchronize_device(bh_1d_mesh_device)
        assert token_value(tok3) in topk_set(logits3)
        assert position_value() == 7
        assert model.sampling_seed_host_copies == 4
    finally:
        model.close(best_effort=True)


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_reduced_real_weight_model_only_trace_compatibility(bh_1d_mesh_device, device_params):
    """Host-logit compatibility deploys model and position, never sampling."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=128,
        layer_indices=(0,),
    )
    generator = Qwen38Generator(model, object())
    try:
        state = generator.allocate_batch_state([3], request_ids=("model-only",))
        logits = model.prefill_forward(torch.tensor([[17, 29, 31]]), state=state)
        ttnn.deallocate(logits)
        host_logits, _ = generator.decode_forward(
            torch.tensor([41]),
            state=state,
            enable_trace=True,
            host_sampling_compatibility=True,
            read_from_device=True,
        )
        ttnn.synchronize_device(bh_1d_mesh_device)
        assert torch.isfinite(host_logits).all()
        assert model._trace_execution_mode == "model_only"
        assert model.sampled_tokens_to_torch(state.token_input).tolist() == [41]
        position = ttnn.to_torch(ttnn.get_device_tensors(state.current_pos)[0]).reshape(-1)
        assert position.tolist() == [4]

        generator.decode_forward(
            torch.tensor([42]),
            state=state,
            enable_trace=True,
            host_sampling_compatibility=True,
            read_from_device=True,
        )
        ttnn.synchronize_device(bh_1d_mesh_device)
        assert model.sampled_tokens_to_torch(state.token_input).tolist() == [42]
        position = ttnn.to_torch(ttnn.get_device_tensors(state.current_pos)[0]).reshape(-1)
        assert position.tolist() == [5]
    finally:
        generator.close()


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_reduced_mixed_prompts_inactive_row_and_eager_decode(bh_1d_mesh_device, device_params):
    """Mixed non-aligned prompts preserve fixed slots and inactive PLE history."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=3,
        max_seq_len=4096,
        layer_indices=(0, 1, 3),
    )
    generator = Qwen38Generator(model, object())
    try:
        prompts = torch.full((3, 33), model.pad_token_id, dtype=torch.int64)
        prompts[0, 0] = 17
        prompts[1] = torch.arange(33, dtype=torch.int64) + 29
        state = generator.allocate_batch_state(
            [1, 33, 0],
            active_mask=[True, True, False],
            request_ids=("short", "nonaligned", "inactive"),
        )
        inactive_history = model.ple_store._histories["inactive"].clone()
        logits = model.prefill_forward(prompts, state=state)
        host = model.logits_to_torch(logits)
        ttnn.deallocate(logits)
        assert tuple(host.shape) == (1, 1, 3, VOCAB_SIZE)
        assert torch.isfinite(host).all()
        assert torch.count_nonzero(host[..., 2, :]) == 0
        assert torch.equal(model.ple_store._histories["inactive"], inactive_history)

        output = model.decode_forward(
            torch.tensor([101, 102, model.pad_token_id]),
            state=state,
            enable_trace=False,
            on_device_sampling=True,
        )
        ttnn.synchronize_device(bh_1d_mesh_device)
        tokens = model.sampled_tokens_to_torch(output)
        assert tuple(tokens.shape) == (3,)
        positions = ttnn.to_torch(ttnn.get_device_tensors(state.current_pos)[0]).reshape(-1)
        assert positions.tolist() == [2, 34, -1]
        assert torch.equal(model.ple_store._histories["inactive"], inactive_history)
    finally:
        generator.close()


@pytest.mark.skipif(
    os.getenv("RUN_QWEN38_FULL_MODEL_BATCH32") != "1",
    reason="explicit all-48-layer batch-32 capability gate",
)
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_full_48_layer_batch32_eager_fixed_slots(bh_1d_mesh_device, device_params, record_property):
    """Prove the serving-facing eager cohort at the largest supported batch."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=32,
        max_seq_len=4096,
    )
    generator = Qwen38Generator(model, object())
    prompts = torch.full((32, 33), model.pad_token_id, dtype=torch.int64)
    prompts[0, 0] = 17
    prompts[31] = torch.arange(33, dtype=torch.int64) + 29
    prompt_lens = [1] + [0] * 30 + [33]
    active_mask = [True] + [False] * 30 + [True]
    pages = model._default_page_table_host.clone()
    pages[1::2] = pages[1::2].flip(-1)

    def histories(prefix):
        return {
            request_id: model.ple_store._histories[request_id].clone()
            for request_id in (f"{prefix}-{slot}" for slot in range(32))
        }

    def run(prefix):
        request_ids = tuple(f"{prefix}-{slot}" for slot in range(32))
        output = generator.generate_batch(
            prompts,
            max_new_tokens=2,
            enable_trace=False,
            sampling_mode="device",
            top_k=1,
            top_p=0.0,
            temperature=1.0,
            prompt_lens=prompt_lens,
            page_table=pages,
            request_ids=request_ids,
            active_mask=active_mask,
            stop_on_eos=False,
        )
        ttnn.synchronize_device(bh_1d_mesh_device)
        state = generator.state
        position_shard = ttnn.get_device_tensors(state.current_pos)[0]
        positions = ttnn.to_torch(position_shard).reshape(-1).tolist()
        assert tuple(output.shape) == (32, 2)
        assert torch.all((0 <= output) & (output < VOCAB_SIZE))
        assert state.active_slots == (0, 31)
        assert positions == [2] + [-1] * 30 + [34]
        assert torch.equal(state.page_table_host, pages)
        assert model.update_page_table(state, pages.clone()) is False
        assert state.page_table_unchanged_skips == 1
        assert state.token_host_copies == 1
        assert state.position_host_copies == 1
        assert state.page_table_host_copies == 1
        assert state.compact_token_readbacks == 2
        assert model._trace_ready is False
        assert model.trace_replays == 0
        current_histories = histories(prefix)
        assert current_histories[f"{prefix}-0"].tolist() == [17, int(output[0, 0])]
        assert current_histories[f"{prefix}-31"].tolist() == [61, int(output[31, 0])]
        for slot in range(1, 31):
            assert current_histories[f"{prefix}-{slot}"].tolist() == [model.ple_store.eos_token_id] * 2
        return output.clone(), current_histories

    try:
        assert model.is_full_stack
        assert len(model.layers) == 48
        assert len(model.kv_cache) == 12
        first_output, first_histories = run("batch32-first")
        second_output, second_histories = run("batch32-second")
        assert torch.equal(first_output[[0, 31]], second_output[[0, 31]])
        for request_id, expected in first_histories.items():
            assert torch.equal(model.ple_store._histories[request_id], expected)
        for slot in range(1, 31):
            assert second_histories[f"batch32-second-{slot}"].tolist() == [model.ple_store.eos_token_id] * 2
        audit = model.runtime_fallback_audit(generator.state)
        assert not any(audit["prohibited_host_work"].values())
        record_property("full_layers", len(model.layers))
        record_property("batch", model.max_batch)
        record_property("active_slots", "0,31")
        record_property("prompt_lens", "1,33")
        record_property("repeated_active_outputs", first_output[[0, 31]].tolist())
        record_property("planned_bytes_per_device", 23_558_946_904)
        record_property("headroom_bytes_per_device", 10_666_573_736)
    finally:
        generator.close()


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_reduced_real_weight_generator_token_out_trace_smoke(bh_1d_mesh_device, device_params):
    """Real GDN+PLE+QSA stack exercises split model/sampling/position traces."""

    class DummyTokenizer:
        def decode(self, ids, **_kwargs):
            return " ".join(str(int(value)) for value in ids)

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=4096,
        layer_indices=(0, 1, 3),
    )
    generator = Qwen38Generator(model, DummyTokenizer())
    try:
        prompt = torch.tensor([[17, 18, 19]], dtype=torch.int64)
        output = generator.generate_batch(
            prompt,
            max_new_tokens=3,
            enable_trace=True,
            sampling_mode="device",
            top_k=1,
            top_p=0.0,
            temperature=1.0,
            request_ids=("trace-smoke",),
            stop_on_eos=False,
        )
        assert tuple(output.shape) == (1, 3)
        assert torch.all((0 <= output) & (output < VOCAB_SIZE))
        audit = model.runtime_fallback_audit(generator.state)
        assert audit["prohibited_host_work"]["optimized_sampling_or_argmax"] is False
        assert audit["prohibited_host_work"]["token_feedback_reconstruction"] is False
        assert audit["counters"]["token_host_copies"] == 2
        assert audit["counters"]["position_host_copies"] == 2
        assert audit["counters"]["page_table_host_copies"] == 1
        assert audit["counters"]["compact_token_readbacks"] == 3
        assert model.trace_replays == 1
        assert model.sampling_trace_id is not None
        assert model.position_trace_id is not None
        assert model.last_decode_timing is not None
        assert "expert_service_seconds" in model.last_decode_timing
        assert "ple_service_seconds" in model.last_decode_timing
    finally:
        generator.close()


@pytest.mark.skipif(os.getenv("RUN_QWEN38_FULL_MODEL") != "1", reason="explicit full-stack hardware gate")
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_full_48_layer_token_out_trace_smoke(bh_1d_mesh_device, device_params):
    """All 48 real layers execute through the canonical token-out split trace."""

    class DummyTokenizer:
        def decode(self, ids, **_kwargs):
            return " ".join(str(int(value)) for value in ids)

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=4096,
    )
    slot_shard_counts = {
        layer.shapes.layer_idx: tuple(
            len(ttnn.get_device_tensors(slot.gate_up)) for slot in layer.host_expert_cache.slots
        )
        for layer in model.layers
        if layer.host_expert_cache is not None
    }
    assert all(count == 2 for counts in slot_shard_counts.values() for count in counts), slot_shard_counts
    generator = Qwen38Generator(model, DummyTokenizer())
    try:
        output = generator.generate_batch(
            torch.tensor([[17, 18, 19]], dtype=torch.int64),
            max_new_tokens=3,
            enable_trace=True,
            sampling_mode="device",
            top_k=1,
            top_p=0.0,
            temperature=1.0,
            request_ids=("full-stack-smoke",),
            stop_on_eos=False,
        )
        assert tuple(output.shape) == (1, 3)
        assert torch.all((0 <= output) & (output < VOCAB_SIZE))
        assert model.is_full_stack
        assert len(model.layers) == 48
        assert len(model.layer_traces) == 48
        assert model.trace_replays == 1
        assert model.last_decode_timing is not None
        print(
            {
                "tokens": output.tolist(),
                "metrics": generator.last_metrics,
                "audit": model.runtime_fallback_audit(generator.state),
            }
        )
    finally:
        generator.close()


@pytest.mark.skipif(
    os.getenv("RUN_QWEN38_CONTEXT_CAPACITY") != "1",
    reason="explicit advertised-context full-stack capacity gate",
)
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_full_model_advertised_context_construction(bh_1d_mesh_device, device_params):
    """Construct every weight/cache/endpoint at the HF 262,144-token limit."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=262_144,
    )
    try:
        assert model.is_full_stack
        assert len(model.layers) == 48
        assert len(model.kv_cache) == 12
        assert tuple(model._default_page_table_host.shape) == (1, 4096)
        assert tuple(model.rot_mats[0].shape) == (1, 1, 262_144, 64)
        assert tuple(model.rot_mats[1].shape) == (1, 1, 262_144, 64)
        assert all(layer.host_expert_cache.capacity == 10 for layer in model.layers)
        persistent = {
            "token": model.decode_token_input,
            "position": model.decode_current_pos,
            "page_table": model.decode_page_table,
            "sampling_k": model.sampling_k,
            "sampling_p": model.sampling_p,
            "sampling_temperature": model.sampling_temp,
            "sampling_index_offsets": model.sampling._index_offsets,
            "sampling_seeds": model.sampling._seeds,
            "sampling_user_ids": model.sampling._user_ids,
            "logprob_mask": model.sampling._log_probs_calculator.mask,
            "logprob_output": model.sampling._log_probs_calculator.output_tensor,
        }
        print(
            {
                "advertised_context_construction": {
                    name: {
                        "shape": tuple(tensor.shape),
                        "padded_shape": tuple(tensor.padded_shape),
                        "volume": tensor.volume(),
                        "dtype": str(tensor.dtype),
                    }
                    for name, tensor in persistent.items()
                }
            }
        )
    finally:
        model.close(best_effort=True)


REFERENCE = Path(__file__).parents[1] / "doc/full_model/readiness_aime24_chat.refpt"
QUALITATIVE_REFERENCE = Path(__file__).parents[1] / "doc/full_model/qualitative_shared_suite.refpt"


@pytest.mark.skipif(os.getenv("RUN_QWEN38_ACCURACY") != "1", reason="explicit full-stack accuracy gate")
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_full_model_aime24_prefill_accuracy(bh_1d_mesh_device, device_params, record_property):
    from transformers import AutoTokenizer

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=4096,
    )
    generator = Qwen38Generator(model, AutoTokenizer.from_pretrained(H.MODEL_SNAPSHOT, local_files_only=True))
    try:
        report = run_prefill_check(generator, REFERENCE)
        print({"prefill_accuracy": report})
        for key in ("top1_percent", "top5_percent", "top100_percent"):
            record_property(f"prefill_{key}", report[key])
        assert report["top5_percent"] >= 98.0
        assert report["top100_percent"] == 100.0
    finally:
        generator.close()


@pytest.mark.skipif(os.getenv("RUN_QWEN38_ACCURACY") != "1", reason="explicit full-stack accuracy gate")
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_full_model_aime24_teacher_forcing_accuracy(bh_1d_mesh_device, device_params, record_property):
    from transformers import AutoTokenizer

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=4096,
    )
    generator = Qwen38Generator(model, AutoTokenizer.from_pretrained(H.MODEL_SNAPSHOT, local_files_only=True))
    try:
        report = run_teacher_forcing(
            generator,
            REFERENCE,
            enable_trace=True,
            decode_rows=int(os.getenv("QWEN38_TEACHER_ROWS", "99")),
        )
        print({"teacher_forcing_accuracy": report})
        for key in ("top1_percent", "top5_percent", "top100_percent"):
            record_property(f"decode_{key}", report[key])
            record_property(f"prefill_{key}", report["prefill"][key])
        for key in (
            "trace_capture_seconds",
            "decode_measured_tokens",
            "decode_seconds",
            "decode_seconds_per_token",
            "decode_tokens_per_second_per_user",
        ):
            record_property(key, report[key])
        assert report["top5_percent"] >= 98.0
        assert report["top100_percent"] == 100.0
    finally:
        generator.close()


@pytest.mark.skipif(os.getenv("RUN_QWEN38_TRACE_DIAGNOSTIC") != "1", reason="explicit trace-state diagnostic")
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_full_model_eager_and_traced_teacher_prefix_match(bh_1d_mesh_device, device_params):
    """Localize early decode drift to eager model state or trace replay state."""

    from transformers import AutoTokenizer

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=4096,
    )
    generator = Qwen38Generator(model, AutoTokenizer.from_pretrained(H.MODEL_SNAPSHOT, local_files_only=True))
    try:
        eager = run_teacher_forcing(generator, REFERENCE, enable_trace=False, decode_rows=4)
        eager_position = ttnn.to_torch(ttnn.get_device_tensors(generator.state.current_pos)[0]).reshape(-1).tolist()
        traced = run_teacher_forcing(generator, REFERENCE, enable_trace=True, decode_rows=4)
        traced_position = ttnn.to_torch(ttnn.get_device_tensors(generator.state.current_pos)[0]).reshape(-1).tolist()
        evidence = {
            "eager": eager,
            "traced": traced,
            "eager_position": eager_position,
            "traced_position": traced_position,
        }
        print({"teacher_prefix_diagnostic": evidence})
        assert eager_position == traced_position == [205], evidence
        assert eager["tt_top1_tokens"] == traced["tt_top1_tokens"], evidence
    finally:
        generator.close()


@pytest.mark.skipif(os.getenv("RUN_QWEN38_ACCURACY") != "1", reason="explicit full-stack accuracy gate")
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_full_model_aime24_autoregressive_quality(bh_1d_mesh_device, device_params):
    from transformers import AutoTokenizer

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=4096,
    )
    generator = Qwen38Generator(model, AutoTokenizer.from_pretrained(H.MODEL_SNAPSHOT, local_files_only=True))
    try:
        report = run_autoregressive(generator, REFERENCE, enable_trace=True)
        output_dir = Path(__file__).parents[1] / "doc/full_model"
        write_report(report, output_dir / "aime24_autoregressive_100_report_final.json")
        write_autoregressive_artifacts(report, output_dir)
        print({"autoregressive": report})
        assert not report["tt_review"]["mechanically_degenerate"]
        assert report["generation_tokens"] == 100
    finally:
        generator.close()


@pytest.mark.skipif(os.getenv("RUN_QWEN38_QUALITATIVE_SUITE") != "1", reason="explicit shared qualitative gate")
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_full_model_shared_qualitative_suite(bh_1d_mesh_device, device_params, record_property):
    """Run three prompt-correct 128-token HF/TT chat comparisons."""

    from transformers import AutoTokenizer

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=4096,
    )
    generator = Qwen38Generator(model, AutoTokenizer.from_pretrained(H.MODEL_SNAPSHOT, local_files_only=True))
    try:
        report = run_qualitative_suite(generator, QUALITATIVE_REFERENCE, enable_trace=True)
        write_report(report, Path(__file__).parents[1] / "doc/full_model/qualitative_shared_suite_final.json")
        print({"qualitative_shared_suite": report})
        record_property("prompt_ids", ",".join(item["id"] for item in report["prompts"]))
        record_property("generation_length", report["metadata"]["generation_length"])
        for item in report["prompts"]:
            assert not item["hf_review"]["mechanically_degenerate"], item
            assert not item["tt_review"]["mechanically_degenerate"], item
            assert item["runtime_fallback_audit"]["prohibited_host_work"]["optimized_sampling_or_argmax"] is False
    finally:
        generator.close()


@pytest.mark.skipif(os.getenv("RUN_QWEN38_PERF") != "1", reason="explicit full-stack performance gate")
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_full_model_batch1_prompt128_generate128_performance(bh_1d_mesh_device, device_params, record_property):
    """Primary readiness workload: full 48-layer prompt128/generate128 token-out."""

    from transformers import AutoTokenizer

    reference = torch.load(REFERENCE, map_location="cpu", weights_only=False)
    prompt = torch.as_tensor(reference["prompt_tokens"], dtype=torch.int64).reshape(1, -1)[:, :128]
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=4096,
    )
    generator = Qwen38Generator(model, AutoTokenizer.from_pretrained(H.MODEL_SNAPSHOT, local_files_only=True))
    try:
        output = generator.generate_batch(
            prompt,
            max_new_tokens=128,
            enable_trace=True,
            sampling_mode="device",
            top_k=1,
            top_p=0.0,
            temperature=1.0,
            request_ids=("batch1-perf-128-128",),
            stop_on_eos=False,
        )
        assert tuple(output.shape) == (1, 128)
        assert model.trace_replays == 126
        report = {
            "workload": {"batch": 1, "prompt_tokens": 128, "generated_tokens": 128},
            "metrics": generator.last_metrics.report(),
            "last_decode_timing": model.last_decode_timing,
            "runtime_fallback_audit": model.runtime_fallback_audit(generator.state),
        }
        print({"full_model_performance": report})
        for key, value in report["metrics"].items():
            record_property(key, value)
        for key, value in report["runtime_fallback_audit"]["counters"].items():
            record_property(key, value)
        for key, value in report["runtime_fallback_audit"]["prohibited_host_work"].items():
            record_property(f"prohibited_{key}", value)
        record_property("generator_host_sampling_compatibility_calls", generator.host_sampling_compatibility_calls)
        assert report["metrics"]["traced"] is True
        assert report["metrics"]["sampling_mode"] == "device"
    finally:
        generator.close()


@pytest.mark.skipif(
    os.getenv("RUN_QWEN38_HOST_COLD_WARM") != "1",
    reason="explicit full-stack cold/warm host-store gate",
)
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_full_model_cold_and_warm_chunked_prefill(bh_1d_mesh_device, device_params, record_property):
    """Repeat prompt128 through real expert/PLE stores without full host residency."""

    reference = torch.load(REFERENCE, map_location="cpu", weights_only=False)
    prompt = torch.as_tensor(reference["prompt_tokens"], dtype=torch.int64).reshape(1, -1)[:, :128]
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=4096,
    )
    generator = Qwen38Generator(model, object())

    def totals():
        expert = [layer.host_expert_cache.metrics() for layer in model.layers]
        return {
            "packed_host_hits": sum(value["packed_host_hits"] for value in expert),
            "packed_host_misses": sum(value["packed_host_misses"] for value in expert),
            "expert_h2d_bytes": sum(value["h2d_bytes"] for value in expert),
            "source_pack_seconds": sum(value["source_pack_seconds"] for value in expert),
            "ple_table_rows_read": model.ple_store.metrics()["table_rows_read"],
            "ple_table_bytes_read": model.ple_store.metrics()["table_bytes_read"],
        }

    try:
        windows = []
        previous = totals()
        model.set_sampling_params(top_k=1, top_p=0.0, temperature=1.0)
        for label in ("cold", "warm"):
            started = time.perf_counter()
            logits = generator.prefill_forward(
                prompt,
                prompt_lens=[128],
                request_ids=(f"{label}-prefill",),
                read_from_device=False,
            )
            ttnn.synchronize_device(bh_1d_mesh_device)
            seconds = time.perf_counter() - started
            sampled = model.sample_logits(logits, generator.state)
            top1 = int(model.sampled_tokens_to_torch(sampled, generator.state)[0])
            ttnn.deallocate(logits)
            current = totals()
            windows.append(
                {
                    "label": label,
                    "seconds": seconds,
                    "top1": top1,
                    "delta": {key: current[key] - previous[key] for key in current},
                }
            )
            previous = current
        print({"full_model_cold_warm_prefill": windows})
        for window in windows:
            label = window["label"]
            record_property(f"{label}_seconds", window["seconds"])
            record_property(f"{label}_top1", window["top1"])
            for key, value in window["delta"].items():
                record_property(f"{label}_{key}", value)
        assert windows[0]["delta"]["packed_host_misses"] > 0
        assert windows[1]["delta"]["packed_host_hits"] > 0
        assert windows[1]["delta"]["packed_host_misses"] == 0
        assert windows[1]["delta"]["ple_table_rows_read"] == 0
        assert windows[0]["top1"] == windows[1]["top1"]
    finally:
        generator.close()
