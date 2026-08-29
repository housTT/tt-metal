# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-TT proof for two vLLM users over the physical-B1 trace."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import harness as H
from models.autoports.qwen_qwen3_8_flash_next.tests.test_full_model import _device_params
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import Qwen38Generator
from models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm import Qwen4ExpForConditionalGeneration
from models.autoports.qwen_qwen3_8_flash_next.tt.model import PAD_TOKEN_ID, VOCAB_SIZE, Qwen38FullModel
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MultichipDecoder

_REQUEST_A = "virtual-a"
_REQUEST_B = "virtual-b"
_PROMPT_LENS = {_REQUEST_A: 63, _REQUEST_B: 67}
_EXTERNAL_GENERATIONS = {_REQUEST_A: 11, _REQUEST_B: 23}


class _TokenOnlyTokenizer:
    @staticmethod
    def decode(ids, **_kwargs):
        return " ".join(str(int(value)) for value in ids)


def _greedy_sampling(rows: int):
    return SimpleNamespace(
        temperature=[1.0] * rows,
        top_p=[0.0] * rows,
        top_k=[1] * rows,
        presence_penalty=[0.0] * rows,
        frequency_penalty=[0.0] * rows,
        repetition_penalty=[1.0] * rows,
        seed=[None] * rows,
        enable_log_probs=[False] * rows,
        num_logprobs=[-2] * rows,
    )


def _mesh_tensor_to_torch(value) -> torch.Tensor:
    """Read one replicated shard; this gate checks state, not performance."""

    return ttnn.to_torch(ttnn.get_device_tensors(value)[0]).cpu()


def _slot_id(request: str) -> int:
    if request == _REQUEST_B:
        return 1
    # Independent controls deliberately reuse physical/virtual slot zero.
    return 0


def _prompt_len(request: str) -> int:
    return _PROMPT_LENS[request.removeprefix("control-")]


def _decode_compact(
    adapter: Qwen4ExpForConditionalGeneration,
    *,
    requests: tuple[str, ...],
    latest_tokens: dict[str, int],
    page_rows: dict[str, torch.Tensor],
    generations: dict[str, int],
    generated_counts: dict[str, int],
    released_state_slots=None,
) -> torch.Tensor:
    """Submit through the adapter and exercise its asynchronous output split."""

    rows = len(requests)
    submitted = adapter.decode_forward(
        tokens=torch.tensor([[latest_tokens[request]] for request in requests], dtype=torch.int64),
        start_pos=torch.tensor(
            [_prompt_len(request) + generated_counts[request] for request in requests],
            dtype=torch.int32,
        ),
        page_table=torch.stack([page_rows[request] for request in requests]),
        kv_cache=adapter._vllm_kv_cache,
        enable_trace=True,
        read_from_device=False,
        sampling_params=_greedy_sampling(rows),
        reset_batch=False,
        request_ids=list(requests),
        state_slot_ids=[_slot_id(request) for request in requests],
        state_slot_generations=[generations[request] for request in requests],
        unpadded_batch_size=rows,
        released_state_slots=released_state_slots,
    )
    host = adapter.read_decode_output(submitted, async_read=False)
    compact = adapter.process_decode_output_host(host, is_tokens=True)
    assert tuple(compact.shape) == (rows,)
    assert torch.all((0 <= compact) & (compact < VOCAB_SIZE))
    return compact


@pytest.mark.skipif(
    os.getenv("RUN_QWEN38_VLLM_VIRTUAL_B2_TT") != "1",
    reason="explicit real-weight physical-B1/virtual-B2 TT gate",
)
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_reduced_real_weight_vllm_virtual_b2_trace_isolation(
    bh_1d_mesh_device,
    device_params,
    monkeypatch,
    record_property,
    expect_error,
):
    """Prove two active logical rows retain independent traced decode state.

    The prompts are deliberately not aligned to a page, tile, or the 128-token
    prefill chunk.  Greedy sampling makes sequential physical-B1 controls an
    exact oracle for the time-multiplexed run.
    """

    del device_params
    monkeypatch.setenv("QWEN38_VLLM_LAYER_INDICES", "0,1,3")
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        virtual_slot_capacity=2,
        max_seq_len=4096,
        layer_indices=(0, 1, 3),
    )
    generator = Qwen38Generator(model, _TokenOnlyTokenizer())
    adapter = Qwen4ExpForConditionalGeneration(generator, max_batch_size=2, max_seq_len=4096)
    try:
        # The reduced stack contains one QSA layer.  Sixty-four physical pages
        # cover the advertised test context; only A:0 and B:32,33 are live for
        # these prompts, so their attention state cannot alias.
        cache = adapter.allocate_kv_cache((64, 1, 64, 256), torch.bfloat16, 1)
        assert cache is model.vllm_kv_cache
        assert model._attention_cache_owner == "vllm"

        page_a = torch.arange(64, dtype=torch.int32)
        page_b = torch.roll(page_a, shifts=-32)
        assert page_a[:2].tolist() == [0, 1]
        assert page_b[:2].tolist() == [32, 33]
        page_rows = {_REQUEST_A: page_a, _REQUEST_B: page_b}

        prompts = torch.full((2, 67), PAD_TOKEN_ID, dtype=torch.int64)
        prompts[0, :63] = (torch.arange(63, dtype=torch.int64) * 17 + 29) % VOCAB_SIZE
        prompts[1, :67] = (torch.arange(67, dtype=torch.int64) * 31 + 43) % VOCAB_SIZE
        prefill = adapter.prefill_forward(
            tokens=prompts,
            page_table=torch.stack((page_a, page_b)),
            kv_cache=cache,
            prompt_lens=[63, 67],
            sampling_params=_greedy_sampling(2),
            empty_slots=[0, 1],
            request_ids=[_REQUEST_A, _REQUEST_B],
            state_slot_ids=[0, 1],
            state_slot_generations=[_EXTERNAL_GENERATIONS[_REQUEST_A], _EXTERNAL_GENERATIONS[_REQUEST_B]],
            unpadded_batch_size=2,
        )
        assert tuple(prefill.shape) == (2,)

        observed = {
            _REQUEST_A: [int(prefill[0])],
            _REQUEST_B: [int(prefill[1])],
        }
        latest = {request: values[-1] for request, values in observed.items()}
        generated_counts = {_REQUEST_A: 0, _REQUEST_B: 0}

        # Alternate wire order as well as physical residency.  Slot IDs remain
        # stable, while compact outputs must follow the real-row wire order.
        for order in ((_REQUEST_A, _REQUEST_B), (_REQUEST_B, _REQUEST_A)):
            compact = _decode_compact(
                adapter,
                requests=order,
                latest_tokens=latest,
                page_rows=page_rows,
                generations=_EXTERNAL_GENERATIONS,
                generated_counts=generated_counts,
            )
            for request, token in zip(order, compact.tolist()):
                latest[request] = int(token)
                observed[request].append(int(token))
                generated_counts[request] += 1

        assert model._trace_ready is True
        assert model._trace_execution_mode == "token_out"
        assert model.sampling_trace_id is not None
        assert model.position_trace_id is not None
        assert model.trace_replays >= 3

        bank = model.virtual_decode_state_bank
        assert bank is not None
        ttnn.synchronize_device(bh_1d_mesh_device)
        assert _mesh_tensor_to_torch(bank.slot_tensor(0, "current_pos")).reshape(-1).tolist() == [65]
        assert _mesh_tensor_to_torch(bank.slot_tensor(1, "current_pos")).reshape(-1).tolist() == [69]
        assert torch.equal(_mesh_tensor_to_torch(bank.slot_tensor(0, "page_table")).reshape(-1), page_a)
        assert torch.equal(_mesh_tensor_to_torch(bank.slot_tensor(1, "page_table")).reshape(-1), page_b)

        # A stale scheduler generation must fail before a trace replay or any
        # device-bank mutation, even when the request name and slot still match.
        before_stale = dict(bank.metrics())
        replays_before_stale = model.trace_replays
        with expect_error(RuntimeError, "stale virtual state owner"):
            adapter.decode_forward(
                tokens=torch.tensor([[latest[_REQUEST_A]]], dtype=torch.int64),
                start_pos=torch.tensor([65], dtype=torch.int32),
                page_table=page_a.unsqueeze(0),
                kv_cache=cache,
                enable_trace=True,
                read_from_device=False,
                sampling_params=_greedy_sampling(1),
                reset_batch=False,
                request_ids=[_REQUEST_A],
                state_slot_ids=[0],
                state_slot_generations=[_EXTERNAL_GENERATIONS[_REQUEST_A] + 1],
                unpadded_batch_size=1,
            )
        assert model.trace_replays == replays_before_stale
        assert bank.metrics()["commits"] == before_stale["commits"]
        assert bank.metrics()["restores"] == before_stale["restores"]

        # The last interleaved row left A resident.  Cancelling B may zero B's
        # snapshot, but A's next two token-out steps require no restore/commit.
        before_cancel = dict(bank.metrics())
        for survivor_step in range(2):
            compact = _decode_compact(
                adapter,
                requests=(_REQUEST_A,),
                latest_tokens=latest,
                page_rows=page_rows,
                generations=_EXTERNAL_GENERATIONS,
                generated_counts=generated_counts,
                released_state_slots=(
                    [(_REQUEST_B, 1, _EXTERNAL_GENERATIONS[_REQUEST_B], "finished")] if survivor_step == 0 else None
                ),
            )
            latest[_REQUEST_A] = int(compact[0])
            observed[_REQUEST_A].append(int(compact[0]))
            generated_counts[_REQUEST_A] += 1
        after_survivor = dict(bank.metrics())
        assert after_survivor["resets"] == before_cancel["resets"] + 1
        assert after_survivor["commits"] == before_cancel["commits"]
        assert after_survivor["restores"] == before_cancel["restores"]
        assert model._virtual_slot_banked[0] is False
        ttnn.synchronize_device(bh_1d_mesh_device)
        assert _mesh_tensor_to_torch(model.decode_current_pos).reshape(-1).tolist() == [67]
        assert torch.equal(_mesh_tensor_to_torch(model.decode_page_table).reshape(-1), page_a)

        # Release the multiplexed request before reusing the same physical B1
        # runtime for independent deterministic controls.
        adapter.release_virtual_state_slots([(_REQUEST_A, 0, _EXTERNAL_GENERATIONS[_REQUEST_A], "finished")])

        def run_control(source_request: str, decode_steps: int, generation: int) -> list[int]:
            control_request = f"control-{source_request}"
            source_row = 0 if source_request == _REQUEST_A else 1
            control_prefill = adapter.prefill_forward(
                tokens=prompts[source_row : source_row + 1],
                page_table=page_rows[source_request].unsqueeze(0),
                kv_cache=cache,
                prompt_lens=[_PROMPT_LENS[source_request]],
                sampling_params=_greedy_sampling(1),
                empty_slots=[0],
                request_ids=[control_request],
                state_slot_ids=[0],
                state_slot_generations=[generation],
                unpadded_batch_size=1,
            )
            result = [int(control_prefill[0])]
            control_latest = {control_request: result[-1]}
            control_pages = {control_request: page_rows[source_request]}
            control_generations = {control_request: generation}
            control_counts = {control_request: 0}
            for _ in range(decode_steps):
                compact = _decode_compact(
                    adapter,
                    requests=(control_request,),
                    latest_tokens=control_latest,
                    page_rows=control_pages,
                    generations=control_generations,
                    generated_counts=control_counts,
                )
                control_latest[control_request] = int(compact[0])
                result.append(int(compact[0]))
                control_counts[control_request] += 1
            adapter.release_virtual_state_slots([(control_request, 0, generation, "finished")])
            return result

        control_a = run_control(_REQUEST_A, decode_steps=4, generation=101)
        control_b = run_control(_REQUEST_B, decode_steps=2, generation=102)
        assert observed[_REQUEST_A] == control_a
        assert observed[_REQUEST_B] == control_b

        audit = model.runtime_fallback_audit(generator.state)
        assert not any(audit["prohibited_host_work"].values())
        assert audit["counters"]["model_only_trace_replays"] == 0
        assert generator.host_sampling_compatibility_calls == 0
        assert audit["counters"]["compact_token_readbacks"] >= len(control_a) + len(control_b)

        record_property("physical_batch", 1)
        record_property("virtual_slot_capacity", 2)
        record_property("layer_indices", "0,1,3")
        record_property("prompt_lengths", "63,67")
        record_property("virtual_tokens", json.dumps(observed, sort_keys=True))
        record_property("control_tokens", json.dumps({_REQUEST_A: control_a, _REQUEST_B: control_b}, sort_keys=True))
        record_property("virtual_bank_metrics", json.dumps(bank.metrics(), sort_keys=True))
        record_property("runtime_fallback_audit", json.dumps(audit, sort_keys=True, default=str))
    finally:
        model.close(best_effort=True)


@pytest.mark.skipif(
    os.getenv("RUN_QWEN38_VLLM_REUSED_PROGRAM_TT") != "1",
    reason="explicit real-weight non-aligned reused-program TT gate",
)
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_reused_prefill_programs_alternating_one_and_sixty_three_tokens(
    bh_1d_mesh_device,
    device_params,
    monkeypatch,
    record_property,
):
    """Alternate exact input shapes and require deterministic completion.

    Lengths one and sixty-three exercise distinct embedding/reshape programs,
    both below a page, tile and 128-token prefill chunk boundary. Repeating the
    sequence while the physical-B1 token-out trace remains live detects stale
    program-cache runtime arguments or a shape-specific retained tensor.
    """

    del device_params
    monkeypatch.setenv("QWEN38_VLLM_LAYER_INDICES", "0,1,3")
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        virtual_slot_capacity=2,
        max_seq_len=4096,
        # One ordinary GDN, the host-backed PLE GDN, and one QSA layer cover
        # every decoder kind while keeping this regression substantially
        # smaller than a 48-layer server smoke.
        layer_indices=(0, 1, 3),
    )
    generator = Qwen38Generator(model, _TokenOnlyTokenizer())
    adapter = Qwen4ExpForConditionalGeneration(generator, max_batch_size=2, max_seq_len=4096)
    try:
        cache = adapter.allocate_kv_cache((64, 1, 64, 256), torch.bfloat16, 1)
        page_row = torch.arange(64, dtype=torch.int32)
        results: dict[int, list[tuple[int, int]]] = {1: [], 63: []}
        program_cache_entries = []

        for iteration, logical_len in enumerate((1, 63, 1, 63)):
            request_id = f"reused-program-{logical_len}-{iteration}"
            external_generation = 200 + iteration
            prompt = (torch.arange(logical_len, dtype=torch.int64) * 17 + 29) % VOCAB_SIZE
            prefill = adapter.prefill_forward(
                tokens=prompt.unsqueeze(0),
                page_table=page_row.unsqueeze(0),
                kv_cache=cache,
                prompt_lens=[logical_len],
                sampling_params=_greedy_sampling(1),
                empty_slots=[0],
                request_ids=[request_id],
                state_slot_ids=[0],
                state_slot_generations=[external_generation],
                unpadded_batch_size=1,
            )
            assert tuple(prefill.shape) == (1,)
            first_token = int(prefill[0])
            assert 0 <= first_token < VOCAB_SIZE

            submitted = adapter.decode_forward(
                tokens=torch.tensor([[first_token]], dtype=torch.int64),
                start_pos=torch.tensor([logical_len], dtype=torch.int32),
                page_table=page_row.unsqueeze(0),
                kv_cache=cache,
                enable_trace=True,
                read_from_device=False,
                sampling_params=_greedy_sampling(1),
                reset_batch=False,
                request_ids=[request_id],
                state_slot_ids=[0],
                state_slot_generations=[external_generation],
                unpadded_batch_size=1,
            )
            host = adapter.read_decode_output(submitted, async_read=False)
            compact = adapter.process_decode_output_host(host, is_tokens=True)
            assert tuple(compact.shape) == (1,)
            second_token = int(compact[0])
            assert 0 <= second_token < VOCAB_SIZE
            results[logical_len].append((first_token, second_token))
            program_cache_entries.append(bh_1d_mesh_device.num_program_cache_entries())

            adapter.release_virtual_state_slots([(request_id, 0, external_generation, "finished")])

        assert results[1][0] == results[1][1]
        assert results[63][0] == results[63][1]
        assert model._trace_ready is True
        assert model._trace_execution_mode == "token_out"
        # The first unseen len-63 workload grows the program cache beside the
        # len-1 trace, so admission releases that trace and the next decode
        # safely recaptures.  Repeated cache-hit shapes retain the trace.
        assert model.trace_replays == 2
        metrics = model.virtual_slot_metrics()
        assert metrics["prefill_admissions_while_trace_live"] >= 3
        assert metrics["prefill_trace_invalidations"] == 1

        record_property("logical_length_order", "1,63,1,63")
        record_property("completion_tokens", json.dumps(results, sort_keys=True))
        record_property("program_cache_entries", json.dumps(program_cache_entries))
        record_property("virtual_slot_metrics", json.dumps(metrics, sort_keys=True))
    finally:
        model.close(best_effort=True)
