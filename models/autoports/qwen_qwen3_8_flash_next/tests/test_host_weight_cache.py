# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for exact host-backed expert and PLE weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from models.autoports.qwen_qwen3_8_flash_next.tests import harness as H
from models.autoports.qwen_qwen3_8_flash_next.tt.host_weight_cache import (
    EXPERT_PACKED_BYTES_PER_RANK,
    PLE_EMBED_DIM,
    PLE_LOGICAL_ROWS,
    PLE_PADDED_ROWS,
    PLE_TABLE_BYTES,
    ExpertIdentity,
    ExpertSlotDirectory,
    Qwen38ExpertHostSource,
    Qwen38PLEHostStore,
    SafetensorCheckpoint,
)


@pytest.fixture(scope="module")
def checkpoint():
    return SafetensorCheckpoint(H.MODEL_SNAPSHOT)


@pytest.fixture(scope="module")
def ple_store(checkpoint):
    store = Qwen38PLEHostStore(checkpoint, row_cache_capacity=256)
    yield store
    store.close()


def test_checkpoint_capacity_constants_and_manifests(checkpoint, ple_store):
    assert checkpoint.total_size == 359_999_963_128
    assert PLE_LOGICAL_ROWS == 320_001_446
    assert PLE_PADDED_ROWS == 320_001_536
    assert PLE_TABLE_BYTES == 102_400_491_520
    assert EXPERT_PACKED_BYTES_PER_RANK == 1_382_400
    assert len(ple_store.manifest) == 33
    assert sum(entry["bytes"] for entry in ple_store.manifest) == 104_298_732_704
    assert all(len(entry["blob_sha256"]) == 64 for entry in ple_store.manifest)


def test_checkpoint_expert_tp_packing_is_exact(checkpoint):
    source = Qwen38ExpertHostSource(checkpoint, layer_idx=3)
    packed = source.load(17)
    fused = checkpoint.indexed_tensor(source.gate_up_key, 17)
    down = checkpoint.indexed_tensor(source.down_key, 17)
    for rank in range(2):
        start, stop = rank * 320, (rank + 1) * 320
        expected_gate_up = torch.cat(
            (fused[start:stop].T, fused[640 + start : 640 + stop].T),
            dim=-1,
        )
        assert torch.equal(packed.gate_up_by_rank[rank][0, 0], expected_gate_up)
        assert torch.equal(packed.down_by_rank[rank][0, 0], down[:, start:stop].T)
    assert source.metrics()["checkpoint_bytes"] == 9_830_400
    assert len(source.manifest) == 2


def test_slot_directory_cold_hit_partial_evict_reload_and_stale_guard(expect_error):
    directory = ExpertSlotDirectory(2)
    loads = []
    first = directory.ensure(4, [7, 11], lambda *args: loads.append(args))
    assert first.misses == (7, 11) and first.hits == ()
    second = directory.ensure(4, [11, 13], lambda *args: loads.append(args))
    assert second.hits == (11,) and second.misses == (13,)
    assert second.evictions == (ExpertIdentity(4, 7),)
    reloaded = directory.ensure(4, [7, 13], lambda *args: loads.append(args))
    assert reloaded.hits == (13,) and reloaded.misses == (7,)
    directory.validate(reloaded)
    stale = reloaded.__class__(**{**reloaded.__dict__, "generations": (0, 0)})
    with expect_error(RuntimeError, "stale expert slot"):
        directory.validate(stale)
    assert [identity.expert_id for _, identity, _ in loads] == [7, 11, 13, 7]


def test_slot_directory_capacity_one_thrash_duplicate_underfill_reset_and_failure(expect_error):
    directory = ExpertSlotDirectory(1)
    loads = []
    for expert in (3, 9, 3, 9):
        plan = directory.ensure(0, [expert, expert], lambda *args: loads.append(args))
        assert plan.requested == (expert,)
    assert len(loads) == 4
    before = directory.records[0].generation
    directory.reset()
    assert not directory.records[0].valid and directory.records[0].generation == before + 1

    def fail(*_args):
        raise OSError("rank-1 upload failed")

    with expect_error(OSError, "rank-1"):
        directory.ensure(0, [5], fail)
    assert not directory.records[0].valid


def test_slot_directory_prefill_wave_partition_is_exact():
    directory = ExpertSlotDirectory(3)
    route_ids = [8, 2, 8, 5, 7, 2, 11, 13]
    waves = directory.waves(route_ids)
    assert waves == ((8, 2, 5), (7, 11, 13))
    assert tuple(expert for wave in waves for expert in wave) == (8, 2, 5, 7, 11, 13)


def test_slot_directory_ordered_trace_replay_replaces_by_address(expect_error):
    directory = ExpertSlotDirectory(3)
    loads = []
    first = directory.ensure_ordered(2, [7, 11, 13], lambda *args: loads.append(args))
    assert first.slot_expert_ids == (7, 11, 13)
    assert first.misses == (7, 11, 13)
    second = directory.ensure_ordered(2, [11, 7, 17], lambda *args: loads.append(args))
    assert second.slot_expert_ids == (11, 7, 17)
    assert second.active_slots == (True, True, True)
    assert second.misses == (11, 7, 17)
    directory.validate(second)
    third = directory.ensure_ordered(2, [11, 7, 17], lambda *args: loads.append(args))
    assert third.hits == (11, 7, 17) and third.misses == ()
    with expect_error(ValueError, "duplicate"):
        directory.ensure_ordered(2, [1, 1], lambda *_: None)


def test_ple_row_ids_match_hf_embedding_boundary(ple_store):
    H.import_target_transformers()
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding

    config = H.target_config().text_config
    with torch.device("meta"):
        reference = Qwen4ExpTextNGramEmbedding(config, PLE_EMBED_DIM, layer_idx=1, ple_layer_index=0)
    reference.layer_multipliers = ple_store.layer_multipliers
    reference.ngram_heads_vocab_sizes = ple_store.head_vocab_sizes
    reference.ngram_heads_offsets = ple_store.head_offsets

    class CaptureEmbedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(1, 160), requires_grad=False)
            self.indices = None

        def forward(self, indices):
            self.indices = indices.clone()
            return torch.zeros(*indices.shape, 160, dtype=torch.bfloat16)

    capture = CaptureEmbedding()
    reference.ngram_embedding = capture
    tokens = torch.tensor([[12, 12, 248044, 17, 99, 3, 3]], dtype=torch.int64)
    reference(tokens, past_key_values=None)
    observed = ple_store.row_ids(["hf-parity"], tokens, reset=True)
    assert torch.equal(observed, capture.indices)


def test_ple_chunk_history_eos_reset_and_request_isolation(ple_store):
    tokens = torch.tensor([[4, 5, 248044, 6, 7, 8, 9]], dtype=torch.int64)
    whole = ple_store.row_ids(["whole"], tokens, reset=True)
    first = ple_store.row_ids(["chunk"], tokens[:, :3], reset=True)
    second = ple_store.row_ids(["chunk"], tokens[:, 3:5])
    third = ple_store.row_ids(["chunk"], tokens[:, 5:])
    assert torch.equal(whole, torch.cat((first, second, third), dim=1))

    other = ple_store.row_ids(["other"], torch.tensor([[100, 101]]), reset=True)
    ple_store.row_ids(["chunk"], torch.tensor([[10]]))
    other_next = ple_store.row_ids(["other"], torch.tensor([[102]]))
    isolated = ple_store.row_ids(["isolated"], torch.tensor([[100, 101, 102]]), reset=True)
    assert torch.equal(torch.cat((other, other_next), dim=1), isolated)
    ple_store.cancel_request("other")
    restarted = ple_store.row_ids(["other"], torch.tensor([[102]]))
    fresh = ple_store.row_ids(["fresh"], torch.tensor([[102]]), reset=True)
    assert torch.equal(restarted, fresh)


def test_ple_non_aligned_mask_and_real_checkpoint_rows(ple_store, checkpoint):
    tokens = torch.arange(35, dtype=torch.int64).reshape(1, 35)
    valid = torch.ones_like(tokens, dtype=torch.bool)
    valid[:, 17] = False
    row_ids = ple_store.row_ids(["nonaligned"], tokens, valid_mask=valid, reset=True)
    embeddings = ple_store.lookup_rows(row_ids)
    assert embeddings.shape == (1, 35, PLE_EMBED_DIM)
    for token_position, head in ((0, 0), (17, 9), (34, 15)):
        global_row = int(row_ids[0, token_position, head])
        shard, local = divmod(global_row, 2_500_012)
        key = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
        expected = checkpoint.indexed_tensor(key, local)
        start = head * 160
        assert torch.equal(embeddings[0, token_position, start : start + 160], expected)


def test_ple_duplicate_row_cache_and_metrics(ple_store):
    before = ple_store.metrics()
    ids = torch.zeros(2, 3, 16, dtype=torch.int64)
    first = ple_store.lookup_rows(ids)
    middle = ple_store.metrics()
    second = ple_store.lookup_rows(ids)
    after = ple_store.metrics()
    assert torch.equal(first, second)
    assert middle["table_rows_read"] - before["table_rows_read"] <= 1
    assert after["table_rows_read"] == middle["table_rows_read"]
    assert after["selected_rows"] - before["selected_rows"] == 2 * ids.numel()
    assert after["h2d_bytes"] - before["h2d_bytes"] == 2 * 2 * 3 * PLE_EMBED_DIM * 2


def test_host_contract_json_numbers_are_serializable(checkpoint, ple_store):
    payload = {
        "checkpoint_total_bytes": checkpoint.total_size,
        "ple_table_bytes": PLE_TABLE_BYTES,
        "ple_manifest": ple_store.manifest,
    }
    assert json.loads(json.dumps(payload))["ple_table_bytes"] == PLE_TABLE_BYTES
    contract_path = Path(__file__).resolve().parents[1] / "doc" / "host_weight_contract.json"
    contract = json.loads(contract_path.read_text())
    assert contract["expert_cache"]["device_bytes_per_rank_full_48_layer_stack"] == 729_907_200
    assert contract["full_stack_capacity"]["planned_total_bytes_per_device"] == 8_066_785_280
    assert contract["full_stack_capacity"]["fits"] is True
