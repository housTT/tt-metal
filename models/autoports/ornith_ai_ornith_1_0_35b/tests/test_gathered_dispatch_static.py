# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only proofs and source-contract checks for Ornith's compact gathered MoE dispatch.

This file deliberately has no mesh fixture: it is safe to run while TT hardware is in use elsewhere.
The device-level permutation/PCC tests remain in ``test_multichip_decoder.py``.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

import pytest
import torch

from models.autoports.ornith_ai_ornith_1_0_35b.tt import optimized_decoder as OD


def _random_local_mask(tokens: int, experts: int, top_k: int, seed: int) -> torch.Tensor:
    """A legal per-device routing mask with anywhere from zero to ``top_k`` local experts/token."""
    generator = torch.Generator().manual_seed(seed)
    mask = torch.zeros(tokens, experts, dtype=torch.int64)
    degree = torch.randint(0, top_k + 1, (tokens,), generator=generator)
    for token, selected in enumerate(degree.tolist()):
        if selected:
            experts_for_token = torch.randperm(experts, generator=generator)[:selected]
            mask[token, experts_for_token] = 1
    return mask


def _host_compact_plan(mask: torch.Tensor, top_k: int):
    """Integer reference for the on-device count/cumsum/rank/scatter construction."""
    tokens, experts = mask.shape
    counts = mask.sum(dim=0)
    aligned = ((counts + OD.TILE - 1) // OD.TILE) * OD.TILE
    ends = torch.cumsum(aligned, dim=0)
    offsets = ends - aligned
    rank = torch.cumsum(mask, dim=0) - mask
    slots = offsets.reshape(1, experts) + rank
    capacity = OD._gather_compact_capacity(tokens, experts, top_k)
    return counts, aligned, offsets, slots, capacity


def test_compact_gather_capacity_is_tight_at_the_admitted_shape():
    tokens = OD.MOE_GATHER_SUB_CHUNK
    assert tokens == OD.MOE_GATHER_MIN_TOKENS == OD.MOE_GATHER_MAX_SUB_CHUNK == 1024
    experts, top_k = 64, 8
    capacity = OD._gather_compact_capacity(tokens, experts, top_k)
    assert capacity == tokens * top_k + (OD.TILE - 1) * experts
    assert capacity % OD.TILE == 0
    assert capacity < tokens * experts

    # This count vector reaches the bound exactly: every expert has remainder one modulo TILE, so
    # every one pays the full 31-row padding term while the total remains tokens * top_k.
    tile_units, remainder = divmod(tokens * top_k - experts, OD.TILE)
    assert remainder == 0
    counts = torch.ones(experts, dtype=torch.int64)
    whole_rounds, partial_round = divmod(tile_units, experts)
    counts += whole_rounds * OD.TILE
    counts[:partial_round] += OD.TILE
    assert int(counts.sum()) == tokens * top_k
    assert int(counts.max()) <= tokens
    aligned = ((counts + OD.TILE - 1) // OD.TILE) * OD.TILE
    assert int(aligned.sum()) == capacity


def test_compact_gather_host_dispatch_and_reverse_are_bijective_at_the_admitted_shape():
    tokens = OD.MOE_GATHER_SUB_CHUNK
    assert tokens == OD.MOE_GATHER_MIN_TOKENS == OD.MOE_GATHER_MAX_SUB_CHUNK == 1024
    experts, top_k = 64, 8
    masks = [
        _random_local_mask(tokens, experts, top_k, seed=1000 + tokens),
        torch.zeros(tokens, experts, dtype=torch.int64),
    ]
    # Worst local skew: every token chooses the same eight experts.
    skewed = torch.zeros(tokens, experts, dtype=torch.int64)
    skewed[:, :top_k] = 1
    masks.append(skewed)

    for mask in masks:
        counts, aligned, offsets, slots, capacity = _host_compact_plan(mask, top_k)
        selected = mask.bool()
        selected_slots = slots[selected]

        assert int(aligned.sum()) <= capacity
        assert torch.all(offsets % OD.TILE == 0)
        assert torch.all(counts <= tokens)
        assert torch.all(slots[selected] < capacity)
        assert selected_slots.unique().numel() == selected_slots.numel()

        # Emulate scatter(token_id -> packed_slot), then gather the same slot back. This is the exact
        # identity the device test wraps around an identity FFN; it also covers the zero-local case.
        dispatched = torch.zeros(capacity, dtype=torch.int64)
        token_ids = torch.arange(tokens).reshape(tokens, 1).expand(tokens, experts)
        dispatched[selected_slots] = token_ids[selected]
        assert torch.equal(dispatched[selected_slots], token_ids[selected])

        # Every occupied expert region is non-overlapping and exactly tile-rounded. Padding is never
        # selected by the reverse map even though it remains zero in the dispatch buffer.
        for expert in range(experts):
            start = int(offsets[expert])
            stop = start + int(aligned[expert])
            expert_slots = slots[:, expert][selected[:, expert]]
            assert torch.all(expert_slots >= start)
            assert torch.all(expert_slots < start + int(counts[expert]))
            assert stop <= capacity


def _qualified_call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def test_compact_gather_source_uses_one_scatter_one_reverse_sort():
    source = textwrap.dedent(inspect.getsource(OD.OptimizedMoE._gather_routed_experts))
    tree = ast.parse(source)
    calls = [_qualified_call_name(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]

    assert calls.count("ttnn.scatter") == 1
    assert calls.count("ttnn.sort") == 1, "only the expert-axis reverse sort should remain"
    assert calls.count("ttnn.embedding") == 2
    assert calls.count("ttnn.where") == 1, "uninitialized invalid FFN rows must be selected away"
    assert calls.count("ttnn.cumsum") == 2, "one expert-region cumsum plus one token-rank cumsum"
    assert calls.count("_gather_compact_capacity") == 1
    assert "ttnn.multiply(expert_sel, float(tokens))" not in source
    assert "region_offsets" in source and "packed_slot" in source


def test_compact_gather_invalid_rows_select_nonfinite_poison_to_zero():
    """A zero-local device must not turn unwritten FFN rows into NaNs via ``0 * poison``."""
    tokens, experts, top_k, hidden = OD.MOE_GATHER_SUB_CHUNK, 64, 8, 32
    assert tokens == 1024
    mask = torch.zeros(tokens, experts, dtype=torch.int64)
    _, _, _, slots, capacity = _host_compact_plan(mask, top_k)

    # The C++ composite allocates its shared TILE output with `empty` and writes no expert tiles when
    # all counts are zero. Poison that entire unwritten buffer, then emulate the production sentinel
    # path: every invalid expert clamps to E-1 and therefore gathers that expert's slot for the token.
    unwritten = torch.full((capacity, hidden), float("nan"))
    invalid_slots = slots[:, -1].clamp(0, capacity - 1).reshape(tokens, 1).expand(tokens, top_k)
    gathered = unwritten[invalid_slots]
    valid = torch.zeros(tokens, top_k, 1, dtype=torch.bool)

    assert torch.isnan(gathered * valid).all(), "the poison must reproduce the unsafe 0 * NaN failure"
    selected = torch.where(valid, gathered, 0.0)
    assert torch.isfinite(selected).all()
    assert torch.count_nonzero(selected) == 0


def test_compact_gather_cpp_op_contracts_admit_the_shapes():
    """Pin the non-obvious C++ contracts the Python construction relies on, without a device."""
    repo = Path(__file__).resolve().parents[4]
    scatter = (repo / "ttnn/cpp/ttnn/operations/data_movement/scatter/scatter.cpp").read_text()
    scatter_device = (
        repo / "ttnn/cpp/ttnn/operations/data_movement/scatter/device/scatter_device_operation.cpp"
    ).read_text()
    scatter_factory = (
        repo / "ttnn/cpp/ttnn/operations/data_movement/scatter/device/scatter_program_factory.cpp"
    ).read_text()
    unified = (
        repo / "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/"
        "unified_routed_expert_ffn_device_operation.cpp"
    ).read_text()

    # The 256-element integer limit is conditional on TILE layout; compact indices/source/base are
    # explicitly UINT32 ROW_MAJOR. The factory chunks long sticks, so capacity+1 need not fit in L1.
    assert "is_i32(index_dtype) && index_layout == Layout::TILE" in scatter
    assert "index_dtype == DataType::UINT32" in scatter_device
    assert "Sharded tensors are not supported" in scatter_device
    assert "input_and_output_chunk_size = std::min" in scatter_factory
    assert "index_chunk_size = std::min" in scatter_factory

    # The fused FFN accepts a shared BF16 ROW_MAJOR input, dynamic UINT32 ROW_MAJOR DRAM offsets, and
    # only requires its M dimension to be tile-aligned — exactly what _gather_compact_capacity pins.
    assert "x must be BFLOAT16 when x_is_row_major" in unified
    assert "expert_region_offsets must be UINT32" in unified
    assert "expert_region_offsets must be ROW_MAJOR layout" in unified
    assert "expert_region_offsets must be DRAM-interleaved" in unified
    assert "x M ({}) must be tile-aligned" in unified


@pytest.mark.parametrize(
    "args",
    [(0, 64, 8), (1024, 0, 8), (1024, 64, 0), (1024, 4, 8), (33, 64, 8)],
)
def test_compact_gather_capacity_rejects_illegal_geometry(args, expect_error):
    with expect_error(ValueError, r"requires positive|cannot exceed|must be tile-aligned"):
        OD._gather_compact_capacity(*args)


def test_gathered_shape_gate_admits_only_the_measured_1024_rows():
    """The production predicate must not broaden the one measured gathered sub-chunk."""
    experts = OD.MOE_GATHER_MAX_LOCAL_EXPERTS
    assert experts == 64
    assert OD.MOE_GATHER_MIN_TOKENS == OD.MOE_GATHER_SUB_CHUNK == OD.MOE_GATHER_MAX_SUB_CHUNK == 1024
    assert OD._gather_support_reason(experts, 1024) is None

    for rows in (OD.PREFILL_ALIGN, 256, 512, 1024 - OD.PREFILL_ALIGN):
        reason = OD._gather_support_reason(experts, rows)
        assert reason is not None
        assert "MOE_GATHER_MIN_TOKENS (1024)" in reason

    reason = OD._gather_support_reason(experts, 1024 + OD.PREFILL_ALIGN)
    assert reason is not None
    assert "MOE_GATHER_MAX_SUB_CHUNK (1024)" in reason
