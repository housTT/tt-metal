# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real-adapter request-boundary regression; run profiles serially."""

import json
import os
import traceback
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.test_full_model_contract import _config, _load_probe_state
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm import PRECISION_CONFIG, Gemma4ForCausalLM
from models.autoports.google_gemma_4_26b_a4b_it.tt.model import DECODE_SLOT_COUNT, Gemma4FullModel
from models.common.sampling.sampling_params import SamplingParams


def _host(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).clone()


def _identity(trace):
    return {
        "model_trace_id": str(trace.model_trace_id),
        "sampling_trace_id": str(trace.sampling_trace_id),
        "token": trace.token_input.buffer_address(),
        "sampled": trace.sampled_tokens.buffer_address(),
        "position": trace.current_pos.buffer_address(),
        "rope": trace.position_ids.buffer_address(),
        "tables": [table.buffer_address() for table in trace.state.page_tables],
        "cache": [tensor.buffer_address() for pair in trace.state.kv_cache for tensor in pair],
    }


@pytest.mark.skipif(
    os.environ.get("GEMMA4_VLLM_TRACE_REUSE_PROBE") != "1",
    reason="set GEMMA4_VLLM_TRACE_REUSE_PROBE=1 for the serialized hardware probe",
)
@pytest.mark.parametrize(
    "mesh_device,device_params,profile_tp_size",
    [
        ((1, 1), {"trace_region_size": 67_108_864}, 1),
        ((2, 2), {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 67_108_864}, 2),
        ((1, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 134_217_728}, 4),
    ],
    ids=["p150", "p150x2", "p150x4"],
    indirect=["mesh_device", "device_params"],
)
def test_adapter_request_trace_reuse(mesh_device, profile_tp_size, monkeypatch):
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1", "enable allocator checks before importing TTNN"
    target_mesh = (
        mesh_device
        if mesh_device.get_num_devices() == profile_tp_size
        else mesh_device.create_submesh(ttnn.MeshShape((1, profile_tp_size)), offset=ttnn.MeshCoordinate(0, 0))
    )
    model = Gemma4FullModel(
        mesh_device=target_mesh,
        hf_config=_config(),
        state_dict=_load_probe_state(full_stack=False),
        max_seq_len=128,
        max_batch_size=DECODE_SLOT_COUNT,
        layer_indices=[0, 5],
        tensor_cache_path="/tmp/gemma4_full_model_probe_cache",
        create_kv_cache=False,
        precision_config_path=PRECISION_CONFIG,
    )
    gen = Gemma4Generator(model, tokenizer=None, sampling_mode="device")
    adapter = Gemma4ForCausalLM(generator=gen)
    cache = adapter.allocate_kv_cache_per_layer(
        [
            (
                (DECODE_SLOT_COUNT * spec.blocks_per_slot, spec.local_kv_heads, spec.block_size, spec.head_dim),
                torch.bfloat16,
                i,
            )
            for i, spec in enumerate(model.cache_specs)
        ]
    )
    tables = [
        torch.arange(DECODE_SLOT_COUNT * spec.blocks_per_slot, dtype=torch.int32).reshape(
            DECODE_SLOT_COUNT, spec.blocks_per_slot
        )
        for spec in model.cache_specs
    ]
    # Change only the sliding layer's scheduler mapping; full-attention tables
    # keep their contents and must receive no upload at this boundary.
    changed_tables = [tables[0].roll(1, dims=0), tables[1]]
    page_uploads = []
    original_copy = ttnn.copy_host_to_device_tensor

    def record_page_copy(source, destination, *args, **kwargs):
        for index, table in enumerate(adapter._serving_state.page_tables):
            if destination is table:
                page_uploads.append(index)
        return original_copy(source, destination, *args, **kwargs)

    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", record_page_copy)
    greedy = SamplingParams(temperature=0.0, top_k=1, top_p=0.0)
    refreshed_inputs = {}
    original_refresh = gen._refresh_device_input

    def record_refresh(target, values, **kwargs):
        refreshed_inputs[target.buffer_address()] = values.clone()
        return original_refresh(target, values, **kwargs)

    monkeypatch.setattr(gen, "_refresh_device_input", record_refresh)
    operations = {"captures": 0, "releases": 0}
    trace_events = []

    def cached_traces():
        return [
            {"key": repr(key), "batch_size": trace.batch_size, "identity": _identity(trace)}
            for key, trace in gen._trace_cache.items()
        ]

    original_get_trace = gen._get_or_capture_decode_trace

    def record_get_trace(tokens, start_pos, *, state, sampling_mode, sampling_spec, **kwargs):
        requested_key = (
            tokens.shape[0],
            sampling_mode,
            sampling_spec.key,
            id(state),
            tuple(map(id, state.page_tables)),
        )
        trace_events.append(
            {
                "operation": "get_trace",
                "requested_key": repr(requested_key),
                "cache_hit": requested_key in gen._trace_cache,
                "cached_keys": [repr(key) for key in gen._trace_cache],
                "request_boundary": gen._request_boundary,
            }
        )
        return original_get_trace(
            tokens, start_pos, state=state, sampling_mode=sampling_mode, sampling_spec=sampling_spec, **kwargs
        )

    monkeypatch.setattr(gen, "_get_or_capture_decode_trace", record_get_trace)
    for name, key in (("begin_trace_capture", "captures"), ("release_trace", "releases")):
        original = getattr(ttnn, name)

        def counted(*args, _original=original, _key=key, **kwargs):
            operations[_key] += 1
            trace_events.append(
                {
                    "operation": _key,
                    "callers": [f"{frame.name}:{frame.lineno}" for frame in traceback.extract_stack(limit=4)[:-1]],
                    "decode_ready": adapter._decode_ready,
                    "last_execution_batch": adapter._last_execution_batch,
                }
            )
            return _original(*args, **kwargs)

        monkeypatch.setattr(ttnn, name, counted)

    def consume(output):
        host, events = adapter.read_decode_output(output, async_read=True)
        assert len(events) == 1
        for event in events:
            ttnn.event_synchronize(event)
        tokens, auxiliary = adapter.process_decode_output_host(host, is_tokens=True)
        assert auxiliary is None
        return tokens.clone()

    def run_request(lengths, token_offset, scheduler_tables, *, reset=False, force_release=False, phase="warm"):
        result = {
            "lengths": lengths,
            "reset_batch": reset,
            "role": "explicit_release_control" if force_release else "candidate",
            "phase": phase,
            "stage": "before_prefill",
            "traces_before_prefill": cached_traces(),
        }
        report["cases"].append(result)
        if force_release:
            adapter._release_decode_traces()
        before = operations.copy()
        before_refresh = adapter._page_table_refreshes
        before_upload = len(page_uploads)
        page_changed = adapter._page_tables_changed(scheduler_tables)
        expected_uploads = [
            index
            for index, table in enumerate(scheduler_tables)
            if adapter._last_page_tables is None or not torch.equal(table, adapter._last_page_tables[index])
        ]
        prompt = torch.stack(
            [(torch.arange(max(lengths)) + token_offset + row * 13) % 2000 + 1 for row in range(len(lengths))]
        )
        program_count_before = target_mesh.num_program_cache_entries()
        first = adapter.prefill_forward(
            prompt,
            page_table=None,
            page_tables_per_layer=scheduler_tables,
            kv_cache=cache,
            prompt_lens=lengths,
            sampling_params=greedy,
        )
        program_count_after = target_mesh.num_program_cache_entries()
        result.update(
            stage="after_prefill",
            traces_after_prefill=cached_traces(),
            program_count_before_prefill=program_count_before,
            program_count_after_prefill=program_count_after,
            prefill_releases=operations["releases"] - before["releases"],
            page_upload_layers=page_uploads[before_upload:],
        )
        assert page_uploads[before_upload:] == expected_uploads
        if result["traces_before_prefill"] and not force_release and program_count_after != program_count_before:
            assert not result["traces_after_prefill"]
            assert result["prefill_releases"] == 2
        assert adapter._page_table_refreshes - before_refresh == int(page_changed)
        counts_before_decode = vars(gen.trace_counters).copy()
        positions = torch.full((DECODE_SLOT_COUNT,), -1, dtype=torch.int32)
        positions[: len(lengths)] = torch.tensor(lengths, dtype=torch.int32)
        token_input = torch.zeros((DECODE_SLOT_COUNT, 1), dtype=torch.long)
        token_input[: len(lengths), 0] = first.reshape(-1)
        steps, logits = [first.reshape(-1).tolist()], []
        identity = None
        for step in range(3):
            output = adapter.decode_forward(
                token_input,
                positions,
                page_table=None,
                page_tables_per_layer=scheduler_tables,
                kv_cache=cache,
                sampling_params=greedy,
                reset_batch=reset and step == 0,
                read_from_device=False,
            )
            sampled = consume(output)
            trace = next(iter(gen._trace_cache.values()))
            current_identity = _identity(trace)
            result.update(stage=f"decode_step_{step}", identity=current_identity)
            assert current_identity["token"] == current_identity["sampled"]
            if identity is None:
                identity = current_identity
                assert (
                    refreshed_inputs[identity["token"]].reshape(-1)[: len(lengths)].tolist()
                    == first.reshape(-1).tolist()
                )
                assert refreshed_inputs[identity["position"]].reshape(-1)[: len(lengths)].tolist() == lengths
                assert refreshed_inputs[identity["rope"]].reshape(-1)[: len(lengths)].tolist() == lengths
            else:
                assert current_identity == identity
            expected = [length + step + 1 for length in lengths] + [-1] * (32 - len(lengths))
            assert _host(trace.current_pos).reshape(-1).tolist() == expected
            assert _host(trace.position_ids).reshape(-1)[: len(lengths)].tolist() == expected[: len(lengths)]
            assert _host(trace.token_input).reshape(-1)[: len(lengths)].tolist() == sampled[: len(lengths)].tolist()
            steps.append(sampled[: len(lengths)].tolist())
            # Read the actual terminal output, including all vocabulary shards;
            # keep only host tensors so no extra device allocations survive replay.
            logits.append(
                torch.cat(
                    [
                        ttnn.to_torch(shard).reshape(-1, shard.shape[-1])[: len(lengths)].clone()
                        for shard in ttnn.get_device_tensors(trace.logits)
                    ],
                    dim=-1,
                )
            )
            # Intentionally stale host feedback must be ignored on steady steps.
            token_input.zero_()
            positions[: len(lengths)] = 99
        for key in ("token_refreshes", "position_refreshes", "rope_refreshes"):
            assert getattr(gen.trace_counters, key) - counts_before_decode[key] == 1
        assert adapter._page_table_refreshes - before_refresh == int(page_changed)
        assert page_uploads[before_upload:] == expected_uploads
        assert gen.trace_counters.page_table_refreshes == 0
        for actual, supplied in zip(adapter._serving_state.page_tables, scheduler_tables):
            host_table = _host(actual)
            assert torch.equal(host_table[: supplied.shape[0], : supplied.shape[1]], supplied)
        result.update(
            {
                "stage": "request_complete",
                "page_changed": page_changed,
                "tokens": steps,
                "identity": identity,
                "captures": operations["captures"] - before["captures"],
                "releases": operations["releases"] - before["releases"],
                "refreshes": {
                    key: getattr(gen.trace_counters, key) - counts_before_decode[key]
                    for key in ("token_refreshes", "position_refreshes", "rope_refreshes")
                },
            }
        )
        return result, logits

    report = {
        "profile": model.precision_policy["resolved_profile"],
        "precision_config": model.precision_policy["config_id"],
        "allocator_tracking": True,
        "watcher": os.environ.get("TT_METAL_WATCHER"),
        "watcher_noinline": os.environ.get("TT_METAL_WATCHER_NOINLINE"),
        "watcher_disabled_features": {
            name: value for name, value in os.environ.items() if name.startswith("TT_METAL_WATCHER_DISABLE_")
        },
        "cases": [],
        "trace_events": trace_events,
    }
    try:
        request_groups = (
            [([33], 10, tables, False), ([47], 90, tables, False), ([33], 10, changed_tables, True)],
            [([33, 47], 10, tables, False), ([35, 45, 39], 90, tables, True), ([33, 47], 10, changed_tables, True)],
        )
        cold_groups = []
        # Keep the original cold B1/B2/B3 sequence: new prefill or first-token
        # programs must retire retained traces before replay. Shape changes
        # remain an independent reason to recapture.
        for requests in request_groups:
            cold = []
            for lengths, offset, mapping, reset in requests:
                result, logits = run_request(lengths, offset, mapping, reset=reset, phase="cold")
                previous = result["traces_before_prefill"]
                execution_batch = 1 if len(lengths) == 1 else DECODE_SLOT_COUNT
                programs_changed = result["program_count_after_prefill"] != result["program_count_before_prefill"]
                recapture = not previous or previous[0]["batch_size"] != execution_batch or programs_changed
                assert result["captures"] == (2 if recapture else 0)
                assert result["releases"] == (2 if previous and recapture else 0)
                cold.append((result, logits))
            cold_groups.append(cold)
        cold_three_rows = cold_groups[1][1][0]
        assert cold_three_rows["program_count_after_prefill"] > cold_three_rows["program_count_before_prefill"]
        assert cold_three_rows["prefill_releases"] == cold_three_rows["captures"] == 2

        # Repeat every warmed boundary with changed token/position/table data.
        # The first request creates the group's trace; every following request
        # must retain it with exactly zero captures and releases.
        for requests, cold in zip(request_groups, cold_groups):
            adapter._release_decode_traces()
            retained = []
            for i, (lengths, offset, mapping, reset) in enumerate(requests):
                result, logits = run_request(lengths, offset, mapping, reset=reset)
                retained.append((result, logits))
                assert result["program_count_after_prefill"] == result["program_count_before_prefill"]
                if i:
                    assert result["identity"] == retained[0][0]["identity"]
                    assert result["captures"] == result["releases"] == 0
                else:
                    assert result["captures"] == 2
            for (lengths, offset, mapping, reset), candidates in zip(requests, zip(cold, retained)):
                control, control_logits = run_request(lengths, offset, mapping, reset=reset, force_release=True)
                assert control["captures"] == 2
                for candidate, candidate_logits in candidates:
                    assert candidate["tokens"] == control["tokens"]
                    max_delta = 0.0
                    for actual, expected in zip(candidate_logits, control_logits):
                        torch.testing.assert_close(actual, expected, rtol=0.005, atol=0.0625)
                        max_delta = max(max_delta, float((actual - expected).abs().max()))
                    candidate["control_max_logit_delta"] = max_delta
                    candidate["explicit_release_control_tokens_match"] = True
        report["verdict"] = "pass"
    finally:
        adapter.teardown()
        output_dir = os.environ.get("GEMMA4_VLLM_TRACE_REUSE_OUTPUT_DIR")
        if output_dir:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / f"trace_reuse_tp{profile_tp_size}.json").write_text(json.dumps(report, indent=2) + "\n")
