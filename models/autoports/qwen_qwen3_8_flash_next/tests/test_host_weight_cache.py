# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for exact host-backed expert and PLE weights."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

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


_TILE_BYTES = {"bf16": 2_048, "bfp8": 1_088, "fp32": 4_096}

# Final persistent TT tensors after FunctionalDecoder loading, FusedDecoder
# packing, OptimizedDecoder typecasting, and MultichipDecoder TP2 slicing.
# Routed-expert slots, PLE rows, caches, recurrent state, and runtime constants
# are separate capacity categories.
_DECODER_NON_EXPERT_ROWS = (
    ("hc_norm", (4, 1_280), "bf16", 96),
    ("hc_down_inject", (5_120, 324), "bfp8", 96),
    ("hc_up", (320, 5_120), "bfp8", 96),
    ("moe_input", (2_560, 1_153), "bfp8", 48),
    ("shared_down", (320, 2_560), "bfp8", 48),
    ("gdn_qkv_base", (2_560, 10_336), "bfp8", 35),
    ("gdn_qkv_layer0", (2_560, 10_560), "bfp8", 1),
    ("gdn_bias_base", (1, 10_336), "fp32", 35),
    ("gdn_bias_layer0", (1, 10_560), "fp32", 1),
    ("gdn_z", (2_560, 6_144), "bfp8", 36),
    ("gdn_decode_taps", (1, 10_240), "fp32", 144),
    ("gdn_prefill_taps", (1, 10_240), "bf16", 144),
    ("gdn_neg_a", (1, 48), "fp32", 36),
    ("gdn_norm", (1, 128), "bf16", 36),
    ("gdn_out", (6_144, 1_280), "bfp8", 36),
    ("gdn_qk_norm_constants", (1, 128), "fp32", 72),
    ("ple_key_value", (2_560, 12_800), "bfp8", 1),
    ("ple_norms", (1, 10_240), "bf16", 3),
    ("ple_taps", (1, 10_240), "bf16", 4),
    ("qsa_input", (2_560, 7_296), "bf16", 12),
    ("qsa_out", (3_072, 2_560), "bf16", 12),
    ("qsa_qk_norms", (1, 256), "bf16", 24),
    ("qsa_index_norms", (1, 128), "bf16", 24),
)

_FULL_TEXT_ENDPOINT_ROWS = (
    ("embed_hidden_tp", (248_320, 1_280), "bf16", 1),
    ("final_hc_norm", (1, 10_240), "bf16", 1),
    ("final_hc_down", (10_240, 320), "bfp8", 1),
    ("final_hc_up", (320, 10_240), "bfp8", 1),
    ("lm_head_vocab_tp", (2_560, 124_160), "bfp8", 1),
)


def _tiled_bytes(shape, dtype):
    leading_elements = 1
    for dimension in shape[:-2]:
        leading_elements *= dimension
    height_tiles = (shape[-2] + 31) // 32
    width_tiles = (shape[-1] + 31) // 32
    return leading_elements * height_tiles * width_tiles * _TILE_BYTES[dtype]


def _inventory_bytes(rows):
    return sum(_tiled_bytes(shape, dtype) * count for _, shape, dtype, count in rows)


def _checkpoint_metadata(checkpoint, key):
    with safe_open(checkpoint.path_for(key), framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(key)
        return tuple(tensor_slice.get_shape()), str(tensor_slice.get_dtype())


def test_non_expert_weight_inventory_from_checkpoint_metadata(checkpoint):
    config = json.loads((H.MODEL_SNAPSHOT / "config.json").read_text())
    text_config = config["text_config"]
    layer_types = tuple(text_config["layer_types"])
    qsa_layers = tuple(index for index, layer_type in enumerate(layer_types) if layer_type == "full_attention")
    gdn_layers = tuple(index for index, layer_type in enumerate(layer_types) if layer_type == "linear_attention")
    assert len(layer_types) == 48
    assert qsa_layers == tuple(range(3, 48, 4))
    assert len(gdn_layers) == 36 and set(gdn_layers).isdisjoint(qsa_layers)
    assert text_config["ple_layer_ids"] == [2]
    assert text_config["vocab_size"] == 248_320
    assert config["tie_word_embeddings"] is False and text_config["tie_word_embeddings"] is False

    # Exercise every layer in the index without materializing any checkpoint
    # tensor.  Transformers canonicalizes raw ``full_attention`` to the
    # autoport's ``qwen_sparse_attention`` name.
    for layer_idx, layer_type in enumerate(layer_types):
        prefix = f"model.language_model.layers.{layer_idx}"
        assert _checkpoint_metadata(checkpoint, f"{prefix}.attn_hyper_connection.hc_norm.weight") == (
            (10_240,),
            "BF16",
        )
        assert _checkpoint_metadata(checkpoint, f"{prefix}.mlp.gate.weight") == ((512, 2_560), "BF16")
        if layer_type == "linear_attention":
            assert _checkpoint_metadata(checkpoint, f"{prefix}.linear_attn.in_proj_qkv.weight") == (
                (10_240, 2_560),
                "BF16",
            )
        else:
            assert _checkpoint_metadata(checkpoint, f"{prefix}.self_attn.q_proj.weight") == (
                (12_288, 2_560),
                "BF16",
            )

    representative_shapes = {
        # Common layer graph.
        "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight": (320, 10_240),
        "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up.weight": (10_240, 320),
        "model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight": (4, 10_240),
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": (640, 2_560),
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": (640, 2_560),
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": (2_560, 640),
        "model.language_model.layers.0.mlp.shared_expert_gate.weight": (1, 2_560),
        # Replicated GDN graph.
        "model.language_model.layers.0.linear_attn.in_proj_z.weight": (6_144, 2_560),
        "model.language_model.layers.0.linear_attn.in_proj_b.weight": (48, 2_560),
        "model.language_model.layers.0.linear_attn.in_proj_a.weight": (48, 2_560),
        "model.language_model.layers.0.linear_attn.conv1d.weight": (10_240, 1, 4),
        "model.language_model.layers.0.linear_attn.dt_bias": (48,),
        "model.language_model.layers.0.linear_attn.A_log": (48,),
        "model.language_model.layers.0.linear_attn.norm.weight": (128,),
        "model.language_model.layers.0.linear_attn.out_proj.weight": (2_560, 6_144),
        # PLE projection graph; the table row is metadata-only and excluded.
        "model.language_model.layers.1.ple.key_proj.weight": (10_240, 2_560),
        "model.language_model.layers.1.ple.value_proj.weight": (2_560, 2_560),
        "model.language_model.layers.1.ple.norm_key.weight": (10_240,),
        "model.language_model.layers.1.ple.norm_query.weight": (10_240,),
        "model.language_model.layers.1.ple.norm_conv.weight": (10_240,),
        "model.language_model.layers.1.ple.conv1d.weight": (10_240, 1, 4),
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": (2_500_012, 160),
        # TP2 QSA plus replicated indexer.
        "model.language_model.layers.3.self_attn.k_proj.weight": (512, 2_560),
        "model.language_model.layers.3.self_attn.v_proj.weight": (512, 2_560),
        "model.language_model.layers.3.self_attn.o_proj.weight": (2_560, 6_144),
        "model.language_model.layers.3.self_attn.q_norm.weight": (256,),
        "model.language_model.layers.3.self_attn.k_norm.weight": (256,),
        "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight": (640, 2_560),
        "model.language_model.layers.3.self_attn.indexer.q_layernorm.weight": (128,),
        "model.language_model.layers.3.self_attn.indexer.k_layernorm.weight": (128,),
        # Future full-text entry/exit tensors.
        "model.language_model.embed_tokens.weight": (248_320, 2_560),
        "model.language_model.hyper_connection_mixer.hc_norm.weight": (10_240,),
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight": (320, 10_240),
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight": (10_240, 320),
        "lm_head.weight": (248_320, 2_560),
    }
    for key, shape in representative_shapes.items():
        assert _checkpoint_metadata(checkpoint, key) == (shape, "BF16")

    decoder_bytes = _inventory_bytes(_DECODER_NON_EXPERT_ROWS)
    endpoint_bytes = _inventory_bytes(_FULL_TEXT_ENDPOINT_ROWS)
    assert decoder_bytes == 3_479_858_176
    assert endpoint_bytes == 981_032_960
    assert decoder_bytes + endpoint_bytes == 4_460_891_136


def test_checkpoint_capacity_constants_and_manifests(checkpoint, ple_store):
    assert checkpoint.total_size == 359_999_963_128
    assert PLE_LOGICAL_ROWS == 320_001_446
    assert PLE_PADDED_ROWS == 320_001_536
    assert PLE_TABLE_BYTES == 102_400_491_520
    assert EXPERT_PACKED_BYTES_PER_RANK == 2_764_800
    assert len(ple_store.manifest) == 33
    assert sum(entry["bytes"] for entry in ple_store.manifest) == 104_298_732_704
    assert all(len(entry["blob_sha256"]) == 64 for entry in ple_store.manifest)


def test_checkpoint_expert_ep2_packing_is_exact(checkpoint):
    source = Qwen38ExpertHostSource(checkpoint, layer_idx=3)
    packed = source.load(17)
    fused = checkpoint.indexed_tensor(source.gate_up_key, 17)
    down = checkpoint.indexed_tensor(source.down_key, 17)
    expected_gate_up = torch.cat((fused[:640].T, fused[640:].T), dim=-1)
    expected_down = down.T
    owner = 17 % 2
    for rank in range(2):
        if rank == owner:
            assert torch.equal(packed.gate_up_by_rank[rank][0, 0], expected_gate_up)
            assert torch.equal(packed.down_by_rank[rank][0, 0], expected_down)
        else:
            assert torch.count_nonzero(packed.gate_up_by_rank[rank]) == 0
            assert torch.count_nonzero(packed.down_by_rank[rank]) == 0
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


def test_slot_directory_batched_plan_is_serial_equivalent_and_failure_is_invalid(expect_error):
    serial = ExpertSlotDirectory(4)
    batched = ExpertSlotDirectory(4)
    serial_loads = []
    batched_loads = []
    for route in ([7, 11, 13, 17], [11, 19, 17, 23], [29, 19, 31, 23], [31, 29, 37, 41]):
        serial_plan = serial.ensure(6, route, lambda *load: serial_loads.append(load))

        def load_batch(loads):
            batched_loads.extend(loads)

        batched_plan = batched.ensure_batched(6, route, load_batch)
        assert batched_plan == serial_plan
        assert batched.records == serial.records
    assert batched_loads == serial_loads

    before = batched.ensure_batched(6, [29, 37, 43, 47], lambda _loads: None)
    assert before.hits == (29, 37)

    def fail(_loads):
        raise OSError("batched owner submission failed")

    with expect_error(OSError, "owner submission"):
        batched.ensure_batched(6, [29, 53, 37, 59], fail)
    records = batched.records
    assert any(record.valid and record.identity == ExpertIdentity(6, 29) for record in records)
    assert any(record.valid and record.identity == ExpertIdentity(6, 37) for record in records)
    assert sum(not record.valid for record in records) == 2


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
    assert contract["expert_cache"]["device_bytes_per_rank_full_48_layer_stack"] == 1_592_524_800
    capacity = contract["full_stack_capacity"]
    expected_total = sum(
        capacity[name]
        for name in (
            "runtime_trace_reserve_bytes_per_device",
            "max_context_cache_bytes_per_device",
            "non_expert_weight_allowance_bytes_per_device",
            "host_expert_cache_and_staging_bytes_per_device",
            "ple_staging_bytes_per_device",
            "all_runtime_state_bytes_per_device",
            "full_model_endpoint_runtime_bytes_per_device",
        )
    )
    assert expected_total == 10_004_550_744
    assert capacity["planned_total_bytes_per_device"] == expected_total
    assert capacity["headroom_bytes_per_device"] == capacity["dram_bytes_per_device"] - expected_total
    assert capacity["fits"] is True
    selected = contract["datatype_sweep_selected_policy"]
    assert selected["config_id"] == "qsa_bfp8_hifi2_lm_head_bf16_hifi2"
    assert selected["weight_groups"]["shared_projection"] == {
        "compute_fidelity": "lofi",
        "dtype": "bfp8",
        "policy": "bfp8_lofi",
    }
    assert selected["expert_representations"] == {
        "source": "bf16",
        "host_packed": "bfp4_tile",
        "device_staging": "bfp4_tile",
        "execution": "bfp4_tile",
    }
    assert selected["ple_representations"] == {
        "table": "bf16_row_major_mmap",
        "host_assembly": "bf16",
        "device_staging": "bf16_tile",
        "execution": "bf16",
    }
    assert selected["upload_policy_preserved"] is True


def test_datatype_sweep_candidate_matrix_is_reproducible_and_matches_results():
    sweep = Path(__file__).resolve().parents[1] / "doc" / "datatype_sweep"
    generator_path = sweep / "make_candidates.py"
    spec = importlib.util.spec_from_file_location("qwen38_make_candidates", generator_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rendered = module.render_outputs()
    assert len(rendered) == 17
    assert all(path.is_file() and path.read_text() == text for path, text in rendered.items())

    manifest = json.loads((sweep / "candidate_matrix.json").read_text())
    for entry in manifest:
        config_id = entry["config_id"]
        config = json.loads((sweep / entry["path"]).read_text())
        result = json.loads((sweep / "full_runs" / config_id / "candidate_result.json").read_text())
        propagation = result["precision_propagation"]
        assert result["config_id"] == propagation["config_id"] == config["config_id"] == config_id
        for dotted_path, check in propagation["checks"].items():
            value = config
            for part in dotted_path.split("."):
                value = value[part]
            assert value == check["expected"], dotted_path
