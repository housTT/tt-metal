# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real-adapter trace retention probe; run profiles serially with allocation tracking on."""

import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.test_full_model_contract import _config, _load_probe_state
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm import PRECISION_CONFIG, Gemma4ForCausalLM
from models.autoports.google_gemma_4_26b_a4b_it.tt.model import DECODE_SLOT_COUNT, Gemma4FullModel
from models.common.sampling.sampling_params import SamplingParams

PROBE_MAX_SEQ_LEN = 512


def _host(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).clone()


def _trace_ids(gen):
    return sorted(
        (trace.batch_size, trace.model_trace_id, trace.sampling_trace_id) for trace in gen._trace_cache.values()
    )


def _spans_overlap(spans, start, end):
    return any(start < span_end and end > span_start for span_start, span_end in spans)


@pytest.mark.skipif(
    os.environ.get("GEMMA4_VLLM_TRACE_REUSE_PROBE") != "1",
    reason="set GEMMA4_VLLM_TRACE_REUSE_PROBE=1 for the serialized hardware probe",
)
@pytest.mark.parametrize(
    "mesh_device,device_params,profile_tp_size",
    [
        ((1, 1), {"trace_region_size": 134_217_728}, 1),
    ],
    ids=["p150"],
    indirect=["mesh_device", "device_params"],
)
def test_adapter_retains_decode_traces_across_mixed_shapes(mesh_device, profile_tp_size, monkeypatch):
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1", "enable allocator checks before importing TTNN"
    model = Gemma4FullModel(
        mesh_device=mesh_device,
        hf_config=_config(),
        state_dict=_load_probe_state(full_stack=False),
        max_seq_len=PROBE_MAX_SEQ_LEN,
        max_batch_size=DECODE_SLOT_COUNT,
        layer_indices=[0, 5],
        tensor_cache_path=os.environ.get("GEMMA4_TENSOR_CACHE_PATH", "/tmp/gemma4_full_model_probe_cache"),
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
    operations = {"captures": 0, "releases": 0}
    for name, key in (("begin_trace_capture", "captures"), ("release_trace", "releases")):
        original = getattr(ttnn, name)

        def counted(*args, _original=original, _key=key, **kwargs):
            operations[_key] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(ttnn, name, counted)

    report = {
        "profile": model.precision_policy["resolved_profile"],
        "precision_config": model.precision_policy["config_id"],
        "allocator_tracking": True,
        "requests": [],
    }
    greedy = SamplingParams(temperature=0.0, top_k=1, top_p=0.0)

    def consume(output):
        host, events = adapter.read_decode_output(output, async_read=True)
        for event in events:
            ttnn.event_synchronize(event)
        tokens, _ = adapter.process_decode_output_host(host, is_tokens=True)
        return tokens.clone()

    def prompt_for(lengths, offset):
        return torch.stack([(torch.arange(max(lengths)) + offset + row * 13) % 2000 + 1 for row in range(len(lengths))])

    def run_request(lengths, offset, *, sampling=greedy, steps=3, start=0, slot_remap=None, label=""):
        before = dict(operations)
        entries_before = mesh_device.num_program_cache_entries()
        prompt = prompt_for(lengths, offset)
        start_pos = None
        if start:
            start_pos = torch.full((len(lengths), 1), start, dtype=torch.int32)
        first = adapter.prefill_forward(
            prompt,
            page_table=None,
            page_tables_per_layer=tables,
            kv_cache=cache,
            prompt_lens=lengths,
            start_pos=start_pos,
            sampling_params=sampling,
        )
        positions = torch.full((DECODE_SLOT_COUNT,), -1, dtype=torch.int32)
        positions[: len(lengths)] = torch.tensor(lengths, dtype=torch.int32)
        token_input = torch.zeros((DECODE_SLOT_COUNT, 1), dtype=torch.long)
        if sampling is not None:
            token_input[: len(lengths), 0] = first.reshape(-1)
        else:
            token_input[: len(lengths), 0] = first.reshape(len(lengths), -1).argmax(dim=-1)
        tokens = [token_input[: len(lengths), 0].tolist()]
        for step in range(steps):
            output = adapter.decode_forward(
                token_input,
                positions,
                page_table=None,
                page_tables_per_layer=tables,
                kv_cache=cache,
                sampling_params=sampling,
                slot_remap=slot_remap if step == 0 else None,
                read_from_device=False,
            )
            if sampling is None:
                sampled = output.reshape(output.shape[0], -1).argmax(dim=-1)
                token_input[: len(lengths), 0] = sampled[: len(lengths)]
            else:
                sampled = consume(output)
                token_input.zero_()  # intentionally stale host feedback on steady steps
            tokens.append(sampled[: len(lengths)].tolist())
            positions[: len(lengths)] = 99
        record = {
            "label": label,
            "lengths": lengths,
            "start": start,
            "sampling": "host" if sampling is None else ("greedy" if sampling.temperature == 0.0 else "sampled"),
            "slot_remap": None if slot_remap is None else slot_remap.tolist()[: len(lengths)],
            "tokens": tokens,
            "captures": operations["captures"] - before["captures"],
            "releases": operations["releases"] - before["releases"],
            "program_cache_growth": mesh_device.num_program_cache_entries() - entries_before,
            "trace_ids": _trace_ids(gen),
        }
        report["requests"].append(record)
        return record

    try:
        adapter.warmup_model_prefill(enable_trace=False, kv_cache=cache)
        adapter.warmup_model_decode(enable_trace=False, kv_cache=cache)
        adapter.warmup_model_prefill(enable_trace=True, kv_cache=cache)
        adapter.warmup_model_decode(enable_trace=True, kv_cache=cache)
        prepared_ids = _trace_ids(gen)
        report["prepared_trace_ids"] = prepared_ids
        report["fence"] = {
            "verified": gen.trace_fence.verified,
            "spans": gen.trace_fence.spans,
            "bytes_per_bank": gen.trace_fence.bytes_per_bank,
            "attempts": gen.trace_fence.attempts,
            "detail": gen.trace_fence.detail,
            "addresses": [tensor.buffer_address() for tensor in gen.trace_fence.tensors],
        }
        assert operations == {"captures": 4, "releases": 0}, operations
        assert [batch for batch, _, _ in prepared_ids] == [1, DECODE_SLOT_COUNT]
        assert gen.trace_fence.verified, gen.trace_fence.detail
        assert report["fence"]["addresses"] == [start for start, _ in gen.trace_fence.spans]
        entries_after_prepare = mesh_device.num_program_cache_entries()
        report["program_cache_entries_after_prepare"] = entries_after_prepare

        a = run_request([33], 10, label="b1_warm")
        b = run_request([35, 47], 90, label="b2_concurrent")
        c = run_request([129], 7, label="b1_unseen_129")
        run_request([130], 8, label="b1_unseen_130")
        run_request([131], 9, label="b1_unseen_131")
        sampled = SamplingParams(temperature=[0.7, 0.7, 0.7], top_k=[32, 32, 32], top_p=[0.95, 0.95, 0.95])
        run_request([33, 129, 47], 10, sampling=sampled, label="b3_sampled_mixed_lengths")
        f = run_request([33], 10, label="b1_warm_repeat_after_sampled")
        g = run_request([33], 10, sampling=None, label="b1_host_mode_control")
        remap = torch.arange(DECODE_SLOT_COUNT)
        remap[:2] = torch.tensor([1, 0])
        run_request([35, 47], 90, slot_remap=remap, label="b2_slot_remap")
        run_request([65], 7, start=64, label="b1_continuation_start_64")
        run_request([35, 47], 90, label="b2_repeat")

        assert gen.trace_counters.unprepared_captures == 0
        assert operations == {"captures": 4, "releases": 0}, operations
        assert all(record["trace_ids"] == prepared_ids for record in report["requests"])
        assert c["program_cache_growth"] > 0, "an unseen prefill length should compile new programs"
        assert gen.trace_counters.program_cache_growth_events > 0
        assert gen.trace_counters.sampling_param_refreshes >= 2
        assert f["tokens"] == a["tokens"], "greedy repeat after sampled traffic must reproduce the warm request"
        assert g["tokens"] == a["tokens"], "eager host-mode greedy control must match the traced path"

        # Allocation-disjointness proof for the fence: every DRAM allocation of a
        # fresh eager prefill (new programs and all) lies outside the fenced spans.
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        try:
            run_request([257], 3, label="b1_unseen_257_under_graph_capture", steps=1)
        finally:
            nodes = ttnn.graph.end_graph_capture()
        intermediates, live = Gemma4Generator._dram_spans_from_graph(nodes, ttnn.get_device_ids(mesh_device)[0])
        overlaps = [span for span in intermediates + live if _spans_overlap(gen.trace_fence.spans, *span)]
        report["post_fence_allocations"] = {
            "dram_allocations": len(intermediates) + len(live),
            "overlapping_fence": overlaps,
        }
        assert not overlaps, overlaps
        report["counters"] = vars(gen.trace_counters)
        report["verdict"] = "pass"
    finally:
        adapter.teardown()
        report["teardown_releases"] = operations["releases"]
        output_dir = os.environ.get("GEMMA4_VLLM_TRACE_REUSE_OUTPUT_DIR")
        if output_dir:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / f"trace_reuse_tp{profile_tp_size}.json").write_text(json.dumps(report, indent=2) + "\n")
