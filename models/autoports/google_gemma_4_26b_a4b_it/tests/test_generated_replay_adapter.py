# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Execute the production adapter AST against a CPU cache-address recorder.

No TTNN import is needed for these row-routing and sampling-boundary controls.
"""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


def make_adapter():
    path = Path(__file__).parents[1] / "tt/generator_vllm.py"
    tree = ast.parse(path.read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Gemma4ForCausalLM")
    owner.bases = [ast.Name(id="object", ctx=ast.Load())]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), owner],
        type_ignores=[],
    )
    events, cache = [], {}
    runtime = NS(
        slice=lambda value, starts, ends, **_: value[tuple(slice(a, b) for a, b in zip(starts, ends))].clone(),
        concat=lambda values, dim, **_: torch.cat(values, dim=dim),
        release_trace=lambda *_args: events.append("release"),
    )
    namespace = dict(torch=torch, ttnn=runtime, DECODE_SLOT_COUNT=32, DEVICE_MAX_TOP_K=64)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    adapter = namespace["Gemma4ForCausalLM"]()
    tables = [torch.arange(10, 42, dtype=torch.int32)[:, None].repeat(1, 4)]
    tables = [tables[0], tables[0], tables[0] + 100]
    state = NS(
        kv_cache=object(),
        page_tables=[table.clone() for table in tables],
        host_page_tables=tables,
        positions=torch.full((32,), -1, dtype=torch.int32),
        active_mask=torch.zeros(32, dtype=torch.bool),
        prompt_lens=[0] * 32,
    )
    calls, draws, gathers = [], [], []

    def write(row, pos, token, phase):
        assert pos >= 0
        for layer, table in enumerate(state.page_tables):
            cache[(layer, int(table[row, 0]), pos)] = (token, phase)

    def prefill(tokens, *, prompt_lens, start_pos, **kwargs):
        events.append("prefill")
        calls.append(
            dict(mode="prefill", tokens=tokens.clone(), positions=start_pos.clone(), lengths=list(prompt_lens))
        )
        assert kwargs["page_table"] is state.page_tables
        assert all(torch.equal(a, b) for a, b in zip(kwargs["host_page_tables"], state.page_tables))
        logits = torch.zeros((1, 1, len(prompt_lens), 4))
        state.active_mask.zero_()
        state.active_mask[: len(prompt_lens)] = True
        state.positions.fill_(-1)
        for row, length in enumerate(prompt_lens):
            for col in range(length):
                write(row, int(start_pos[row, col]), int(tokens[row, col]), "prefill")
            state.positions[row] = int(start_pos[row, length - 1]) + 1
            logits[0, 0, row] = int(tokens[row, length - 1]) + 1000
        return logits

    def decode(tokens, positions, *, active_mask, sampling_mode, enable_trace, **kwargs):
        events.append("decode")
        assert sampling_mode == "host" and enable_trace is False
        assert kwargs["page_table"] is state.page_tables
        assert tokens.shape == (len(positions), 1)
        assert len(positions) in (1, 32)
        assert torch.equal(positions >= 0, active_mask)
        calls.append(
            dict(mode="decode", tokens=tokens.clone(), positions=positions.clone(), active=active_mask.clone())
        )
        logits = torch.full((1, 1, 32, 4), -999.0)
        state.active_mask.zero_()
        state.active_mask[: len(active_mask)] = active_mask
        state.positions.fill_(-1)
        for row in range(len(positions)):
            if not active_mask[row]:
                assert positions[row] == -1
                continue
            pos, token = int(positions[row]), int(tokens[row, 0])
            write(row, pos, token, "decode")
            logits[0, 0, row] = token + 2000
            state.positions[row] = pos + 1
        return logits

    def gather(logits):
        gathers.append(logits.clone())
        return logits

    def sample(logits, *, batch_size, **_):
        draws.append(logits.clone())
        return torch.zeros(batch_size, dtype=torch.long)

    gen = NS(
        model=NS(num_layers=3, vocab_size=4),
        _trace_cache={"retained": NS(model_trace_id=1, sampling_trace_id=2)},
        _seeded_slots=torch.zeros(32, dtype=torch.bool),
        prefill_forward=prefill,
        decode_forward=decode,
        _gather_logits_to_torch=gather,
        sample_device_logits=sample,
        _read_tokens=lambda output, batch: output,
    )
    adapter._require_generator = lambda: gen
    adapter.mesh_device = NS(num_program_cache_entries=lambda: 10)
    adapter.max_batch_size, adapter.max_seq_len = 32, 262144
    adapter._serving_state, adapter._last_page_tables = state, tables
    adapter._sampling_values = lambda *_args, **_kwargs: {}

    def refresh(sources):
        events.append("tables")
        for target, source in zip(state.page_tables, sources):
            target.zero_()
            target[: source.shape[0], : source.shape[1]] = source
        adapter._last_page_tables = [source.clone() for source in sources]
        state.host_page_tables = adapter._last_page_tables

    adapter._refresh_page_tables = refresh
    return NS(
        adapter=adapter,
        gen=gen,
        state=state,
        tables=tables,
        calls=calls,
        cache=cache,
        draws=draws,
        gathers=gathers,
        events=events,
    )


def run_case(starts, ends, boundaries, *, device_sampling=False, slots=None):
    case = make_adapter()
    tokens = torch.arange(max(ends))[None].repeat(len(starts), 1)
    result = case.adapter.prefill_forward(
        tokens,
        case.tables[0],
        case.state.kv_cache,
        ends,
        start_pos=torch.tensor(starts),
        original_prompt_lens=boundaries,
        sampling_params=object() if device_sampling else None,
        page_tables_per_layer=case.tables,
        empty_slots=slots,
    )
    expected = {}
    for row, (start, end) in enumerate(zip(starts, ends)):
        for position in range(start, end):
            phase = "prefill" if boundaries is None or position < boundaries[row] else "decode"
            for layer, table in enumerate(case.tables):
                expected[(layer, int(table[row, 0]), position)] = (position, phase)
    assert case.cache == expected
    assert all(torch.equal(original, actual) for original, actual in zip(case.tables, case.state.page_tables))
    assert all(torch.equal(original, actual) for original, actual in zip(case.tables, case.state.host_page_tables))
    assert case.state.positions[: len(ends)].tolist() == ends
    assert case.state.active_mask[: len(ends)].all()
    assert not case.state.active_mask[len(ends) :].any()
    expected_final = [
        end - 1 + (2000 if boundaries is not None and end > boundaries[row] else 1000) for row, end in enumerate(ends)
    ]
    final = case.draws[0][0, 0, :, 0] if device_sampling else result[:, 0, 0]
    assert final.tolist() == expected_final
    assert len(case.draws) == int(device_sampling)
    assert len(case.gathers) == int(not device_sampling)
    assert not case.adapter._decode_ready
    if boundaries is not None and any(end > boundary for end, boundary in zip(ends, boundaries)):
        assert case.events[:2] == ["release", "release"]
        assert not case.gen._trace_cache
    return case


@pytest.mark.parametrize("device_sampling", [False, True])
def test_generated_suffix_replays_decode_without_intermediate_draws(device_sampling):
    case = run_case([8192], [8197], [8192], device_sampling=device_sampling, slots=[17])
    assert [call["mode"] for call in case.calls] == ["decode"] * 5
    assert all(call["tokens"].shape == (1, 1) for call in case.calls)


@pytest.mark.parametrize("device_sampling", [False, True])
def test_mixed_prompt_crossing_and_replay_rows_preserve_cache_owners(device_sampling):
    case = run_case([40, 11, 8190], [43, 16, 8197], [35, 17, 8192], device_sampling=device_sampling, slots=[17, 2, 31])
    assert case.calls[0]["lengths"] == [5, 2]
    decodes = [call for call in case.calls if call["mode"] == "decode"]
    assert len(decodes) == 5 and all(call["tokens"].shape == (32, 1) for call in decodes)
    assert decodes[0]["positions"][:3].tolist() == [40, -1, 8192]
    assert decodes[3]["positions"][:3].tolist() == [-1, -1, 8195]


@pytest.mark.parametrize("length", [1, 5, 2048])
@pytest.mark.parametrize("metadata", [False, True])
def test_ordinary_prompt_retains_prefill_for_every_chunk_length(length, metadata):
    case = run_case([8192], [8192 + length], [12288] if metadata else None)
    assert len(case.calls) == 1 and case.calls[0]["mode"] == "prefill"


def test_all_32_rows_replay_with_different_remaining_lengths():
    starts = [8192] * 32
    ends = [8193 + row % 5 for row in range(32)]
    case = run_case(starts, ends, [8192] * 32, device_sampling=True, slots=list(reversed(range(32))))
    assert len(case.calls) == 5 and all(call["tokens"].shape == (32, 1) for call in case.calls)


def test_prompt_tables_restore_after_prefill_exception(expect_error):
    case = make_adapter()

    def fail(*_args, **_kwargs):
        raise RuntimeError("prefill failure")

    case.gen.prefill_forward = fail
    with expect_error(RuntimeError, "prefill failure"):
        case.adapter.prefill_forward(
            torch.zeros((2, 10), dtype=torch.long),
            case.tables[0],
            case.state.kv_cache,
            [10, 10],
            start_pos=torch.tensor([5, 5]),
            original_prompt_lens=[4, 8],
            page_tables_per_layer=case.tables,
        )
    assert all(torch.equal(a, b) for a, b in zip(case.tables, case.state.page_tables))
    assert all(torch.equal(a, b) for a, b in zip(case.tables, case.state.host_page_tables))


@pytest.mark.parametrize("boundaries", [[], [-1], [262145], [8192, 8192]])
def test_invalid_original_prompt_metadata_is_rejected(boundaries, expect_error):
    case = make_adapter()
    with expect_error(ValueError, "original_prompt_lens"):
        case.adapter.prefill_forward(
            torch.zeros((1, 8197), dtype=torch.long),
            case.tables[0],
            case.state.kv_cache,
            [8197],
            start_pos=torch.tensor([8192]),
            original_prompt_lens=boundaries,
            page_tables_per_layer=case.tables,
        )
    assert not case.calls


@pytest.mark.parametrize("start,length,boundary", [(0, 3, 0), (8192, 128, 8192), (262140, 4, 262140)])
def test_replay_respects_full_logical_length_and_context(start, length, boundary):
    case = run_case([start], [start + length], [boundary])
    assert len(case.calls) == length


def test_authoritative_prompt_boundary_capability_is_declared():
    assert make_adapter().adapter.supports_original_prompt_lens is True
