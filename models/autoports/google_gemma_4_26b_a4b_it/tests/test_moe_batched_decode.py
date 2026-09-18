"""Row-level contract for the batched (union-sparsity) MoE decode chain.

The multi-user decode MoE runs one ``sparse_matmul`` chain over the 32-row tile with the
union of the rows' expert sets. Every row must equal the result of its own single-user
chain (the pre-existing batch-32-policy path), rows with all-zero routing must produce
zeros, and no row may leak into another. HF equivalence of the whole layer is covered by
``test_optimized_traced_decode_batch_contract`` / ``test_traced_decode_batch_contract``.
"""

from __future__ import annotations

import pytest
import torch

import models.autoports.google_gemma_4_26b_a4b_it.tests.test_functional_decoder as functional_tests
import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import (
    HIDDEN_SIZE,
    NUM_EXPERTS,
    TILE_SIZE,
    TOP_K_EXPERTS,
    union_expert_sparsity,
)
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder

BATCH = 32
ZERO_ROWS = (3, 17, 31)  # inactive decode slots: routing all zero


def _routing_scores(generator: torch.Generator) -> torch.Tensor:
    """Dense ``[1, 1, BATCH, NUM_EXPERTS]`` scores with a normalized top-k per active row."""
    logits = torch.randn(BATCH, NUM_EXPERTS, generator=generator)
    values, indices = torch.topk(torch.softmax(logits, dim=-1), TOP_K_EXPERTS, dim=-1)
    values = values / values.sum(dim=-1, keepdim=True)
    routing = torch.zeros(BATCH, NUM_EXPERTS).scatter(-1, indices, values)
    routing[list(ZERO_ROWS)] = 0.0
    return routing.reshape(1, 1, BATCH, NUM_EXPERTS)


def _to_device(tensor: torch.Tensor, mesh_device) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _rows(tensor: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(tensor).float().reshape(-1, HIDDEN_SIZE)[:BATCH]


def test_union_expert_sparsity_host_contract():
    generator = torch.Generator().manual_seed(7)
    routing = _routing_scores(generator)
    expected = (routing.abs().amax(dim=2, keepdim=True) > 0).reshape(-1)
    assert expected.sum() <= min(NUM_EXPERTS, TOP_K_EXPERTS * (BATCH - len(ZERO_ROWS)))
    for row in range(BATCH):
        active = routing[0, 0, row].nonzero().reshape(-1)
        assert expected[active].all()


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_batched_moe_decode_matches_per_row_chains(mesh_device, device_params, layer_idx):
    cfg = functional_tests._load_text_config()
    decoder = OptimizedDecoder.from_state_dict(
        functional_tests._load_layer_state(layer_idx),
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
    )
    generator = torch.Generator().manual_seed(1234)
    hidden = torch.randn(1, 1, BATCH, HIDDEN_SIZE, generator=generator)
    routing = _routing_scores(generator)
    hidden_tt = _to_device(hidden, mesh_device)
    routing_tt = _to_device(routing, mesh_device)

    union = union_expert_sparsity(routing_tt)
    assert union.layout == ttnn.ROW_MAJOR_LAYOUT
    assert list(union.shape) == [1, 1, 1, NUM_EXPERTS]
    union_host = ttnn.to_torch(union).reshape(-1)
    expected_union = routing.abs().amax(dim=2).reshape(-1) > 0
    assert torch.equal(union_host > 0, expected_union)

    before = dict(decoder.optimized_path_counters)
    batched = _rows(decoder._moe_decode(hidden_tt, routing_tt))
    after = decoder.optimized_path_counters
    assert after["expert_decode_batched"] == before["expert_decode_batched"] + 1
    assert after["expert_decode"] == before["expert_decode"] + 1, "the batch must run as one chain"

    per_row = []
    for row in range(BATCH):
        hidden_row = ttnn.slice(hidden_tt, [0, 0, row, 0], [1, 1, row + 1, HIDDEN_SIZE])
        routing_row = ttnn.slice(routing_tt, [0, 0, row, 0], [1, 1, row + 1, NUM_EXPERTS])
        per_row.append(_rows(decoder._moe_decode_single_user(hidden_row, routing_row, use_batch32_policy=True))[0])
    per_row = torch.stack(per_row)

    for row in ZERO_ROWS:
        assert torch.count_nonzero(batched[row]) == 0, f"inactive row {row} must stay zero"
    pcc_per_row = []
    for row in range(BATCH):
        if row in ZERO_ROWS:
            continue
        ok, pcc = functional_tests.comp_pcc(per_row[row : row + 1], batched[row : row + 1], 0.999)
        pcc_per_row.append(float(pcc))
        assert ok, f"row {row}: batched vs single-user PCC {pcc}"
    # Rows are independent: the best-matching single-user row of every batched row is itself.
    reference_rows = torch.nn.functional.normalize(per_row, dim=-1)
    batched_rows = torch.nn.functional.normalize(batched, dim=-1)
    best = (batched_rows @ reference_rows.T).argmax(dim=1).tolist()
    for row in range(BATCH):
        if row not in ZERO_ROWS:
            assert best[row] == row, (row, best[row])
    assert len(pcc_per_row) == BATCH - len(ZERO_ROWS)
    assert min(pcc_per_row) >= 0.999, pcc_per_row
    assert TILE_SIZE == BATCH
