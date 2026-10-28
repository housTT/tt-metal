# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Functional-decoder tests for Qwen/Qwen3-8B on one Blackhole chip (1x1 mesh).

Default tests run with real weights when the local Hugging Face snapshot is
present and with synthetic weights otherwise. Opt-in environment switches:

* ``QWEN3_CONTEXT_TESTS=1``: advertised-context (40960) prefill and decode.
* ``QWEN3_PERF=1``: Tracy signposted prefill and traced-decode profile.
* ``QWEN3_EVIDENCE_COMMAND``: exact command recorded in evidence JSON.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_8b.tests import weights as weight_source
from models.autoports.qwen_qwen3_8b.tests.reference import HFLayerReference, chunked_layer_reference, load_config
from models.autoports.qwen_qwen3_8b.tt import functional_decoder as decoder_module
from models.autoports.qwen_qwen3_8b.tt.functional_decoder import (
    PREFILL_CHUNK_SIZE,
    TILE_SIZE,
    FunctionalDecoder,
    decode_sdpa_k_chunk_size,
    plan_cache_writes,
    plan_prefill_chunks,
    plan_prefill_window,
    prefill_sdpa_chunk_size,
    round_up,
    validate_hf_config,
)
from models.common.utility_functions import comp_pcc

MODEL_ID = weight_source.MODEL_ID
MODEL_DIR = weight_source.MODEL_DIR
ARTIFACT_DIR = MODEL_DIR / "doc" / "functional_decoder"
LAYER_KIND = "full_attention"
PCC_BAR = 0.995
DELTA_PCC_BAR = 0.99
REPLAY_PCC_BAR = 0.9999
REPRESENTATIVE_LAYERS = (0, 17)
BLOCK_SIZE = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = 4096
CONTEXT = 40960
DEVICE_PARAMS = [{"trace_region_size": 0}]


def _evidence_command(default: str) -> str:
    return os.getenv("QWEN3_EVIDENCE_COMMAND", default)


def _provenance(mesh_device, exact_command: str, weight_origin: str) -> dict:
    decoder_path = Path(decoder_module.__file__).resolve()
    test_path = Path(__file__).resolve()
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=test_path.parents[4], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "unavailable"
    extension = Path(ttnn._ttnn.__file__).resolve()
    try:
        device_ids = [int(d) for d in mesh_device.get_device_ids()]
        arch = str(mesh_device.arch())
    except (AttributeError, TypeError):
        device_ids, arch = [], os.getenv("ARCH_NAME", "unknown")
    return {
        "model_id": MODEL_ID,
        "layer_kind": LAYER_KIND,
        "weight_origin": weight_origin,
        "checkpoint_snapshot": str(weight_source.model_snapshot()),
        "functional_decoder_path": str(decoder_path),
        "functional_decoder_sha256": hashlib.sha256(decoder_path.read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "checkout_git_sha": git_sha,
        "ttnn_extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "hardware": {"arch": arch, "device_ids": device_ids, "mesh": "1x1", "platform": platform.platform()},
        "visible_devices": os.getenv("TT_VISIBLE_DEVICES"),
        "exact_command": exact_command,
        "environment": {
            k: v for k, v in sorted(os.environ.items()) if k.startswith("QWEN3_") or k.startswith("TT_METAL_")
        },
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _write_json(name: str, payload: dict) -> Path:
    """Write evidence JSON; ``QWEN3_EVIDENCE_SUFFIX`` keeps debug-gate runs from overwriting suite evidence."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = os.getenv("QWEN3_EVIDENCE_SUFFIX", "")
    if suffix:
        name = name.replace(".json", f".{suffix}.json")
    path = ARTIFACT_DIR / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _pcc(golden: torch.Tensor, actual: torch.Tensor) -> float:
    return float(comp_pcc(golden.to(torch.float32), actual.to(torch.float32), 0.0)[1])


def _delta_pcc(golden_out, actual_out, hidden_in) -> float:
    hidden_in = hidden_in.to(torch.float32)
    return _pcc(golden_out.to(torch.float32) - hidden_in, actual_out.to(torch.float32) - hidden_in)


def _tt(mesh_device, tensor: torch.Tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor, dtype=dtype, layout=layout, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _host(tensor: torch.Tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout)


def _to_torch(tensor: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(tensor)


def _page_table(mesh_device, num_users: int, blocks_per_user: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    table = torch.randperm(num_users * blocks_per_user, generator=generator).reshape(num_users, blocks_per_user)
    table = table.to(torch.int32)
    return table, _tt(mesh_device, table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)


def _blocks_for(tokens: int) -> int:
    return (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE


def _kv_cache(mesh_device, num_blocks: int, block_size: int = BLOCK_SIZE):
    return FunctionalDecoder.allocate_kv_cache(
        mesh_device, num_blocks=num_blocks, block_size=block_size, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM
    )


@pytest.fixture(scope="module")
def hf_config():
    return load_config()


class LayerFixture:
    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.state, self.weight_origin = weight_source.layer_state_dict(layer_idx)
        self.config = load_config()
        self._embedding = None
        self._token_ids = None
        self._captured = None
        self.input_origin = "unset"

    def reference(self) -> HFLayerReference:
        return HFLayerReference(self.config, self.layer_idx, self.state)

    def inputs(self, seq_len: int, seed: int) -> torch.Tensor:
        """Return ``[1, seq_len, HIDDEN]`` bfloat16 activations shaped like the real layer input."""
        generator = torch.Generator().manual_seed(seed)
        stats = weight_source.load_real_weight_stats()
        captured = stats.get("layer_inputs", {}).get(str(self.layer_idx))
        if self.layer_idx == 0 and self.weight_origin == "real":
            if self._embedding is None:
                self._embedding = weight_source.load_embedding()
                self._token_ids = torch.tensor(
                    json.loads((ARTIFACT_DIR / "layer_inputs/tale_token_ids.json").read_text())["tokens"]
                )
            if seq_len <= self._token_ids.shape[0]:
                offset = int(torch.randint(0, self._token_ids.shape[0] - seq_len + 1, (1,), generator=generator))
                self.input_origin = "real_text_tokens_through_real_embedding"
                return self._embedding[self._token_ids[offset : offset + seq_len]].unsqueeze(0)
        if captured is not None:
            artifact = ARTIFACT_DIR / captured["artifact"]
            if artifact.is_file() and seq_len <= captured["tokens"]:
                if self._captured is None:
                    self._captured = torch.load(artifact)["hidden"]
                offset = int(torch.randint(0, captured["tokens"] - seq_len + 1, (1,), generator=generator))
                self.input_origin = "real_captured_hidden_state"
                return self._captured[offset : offset + seq_len].unsqueeze(0).to(torch.bfloat16)
        if self.layer_idx == 0 and self.weight_origin == "real":
            if self._embedding is None:
                self._embedding = weight_source.load_embedding()
            tokens = torch.randint(0, self.config.vocab_size, (seq_len,), generator=generator)
            self.input_origin = "random_tokens_through_real_embedding"
            return self._embedding[tokens].unsqueeze(0)
        if captured is None:
            raise RuntimeError(f"no recorded input statistics for layer {self.layer_idx}")
        self.input_origin = "gaussian_with_recorded_statistics"
        return (torch.randn(1, seq_len, HIDDEN, generator=generator) * captured["std"] + captured["mean"]).to(
            torch.bfloat16
        )


_LAYER_FIXTURES: dict[int, LayerFixture] = {}


def _layer(layer_idx: int) -> LayerFixture:
    if layer_idx not in _LAYER_FIXTURES:
        _LAYER_FIXTURES[layer_idx] = LayerFixture(layer_idx)
    return _LAYER_FIXTURES[layer_idx]


def _decoder(mesh_device, fixture: LayerFixture) -> FunctionalDecoder:
    return FunctionalDecoder.from_state_dict(
        fixture.state, hf_config=fixture.config, layer_idx=fixture.layer_idx, mesh_device=mesh_device
    )


def test_hf_config_contract_host(hf_config, expect_error):
    geometry = validate_hf_config(hf_config, 0)
    assert geometry.hidden_size == HIDDEN
    assert geometry.intermediate_size == 12288
    assert geometry.num_heads == 32
    assert geometry.num_kv_heads == NUM_KV_HEADS
    assert geometry.head_dim == HEAD_DIM
    assert geometry.max_position_embeddings == CONTEXT
    assert geometry.rope_theta == 1000000.0
    assert geometry.rms_norm_eps == 1e-6
    assert set(hf_config.layer_types) == {LAYER_KIND}
    with expect_error(ValueError, "outside"):
        validate_hf_config(hf_config, hf_config.num_hidden_layers)


@pytest.mark.parametrize(
    "start_pos,seq_len", [(0, 1), (0, 32), (0, 33), (37, 5), (4095, 2), (4096, 4097), (1000, 9000), (40000, 960)]
)
def test_prefill_plan_host(start_pos, seq_len):
    phys_base, lead, phys_total = plan_prefill_window(start_pos, seq_len)
    assert phys_base % TILE_SIZE == 0 and phys_base + lead == start_pos
    assert phys_total % TILE_SIZE == 0 and phys_total >= lead + seq_len
    chunks = plan_prefill_chunks(start_pos, seq_len, PREFILL_CHUNK_SIZE)
    assert sum(c.phys_len for c in chunks) == phys_total
    assert sum(c.logical_len for c in chunks) == seq_len
    assert chunks[0].phys_start == phys_base and chunks[0].lead == lead
    for i, chunk in enumerate(chunks):
        assert chunk.phys_start % TILE_SIZE == 0 and chunk.phys_len % TILE_SIZE == 0
        assert chunk.phys_len <= PREFILL_CHUNK_SIZE
        assert chunk.abs_start >= chunk.phys_start
        assert chunk.abs_end <= chunk.phys_start + chunk.phys_len
        if i > 0:
            assert chunk.lead == 0 and chunk.phys_start == chunks[i - 1].phys_start + chunks[i - 1].phys_len
        kv_extent = round_up(start_pos + seq_len, TILE_SIZE)
        sdpa_chunk = prefill_sdpa_chunk_size(chunk.phys_start, chunk.phys_len, kv_extent)
        assert chunk.phys_start % sdpa_chunk == 0
        assert chunk.phys_start + round_up(chunk.phys_len, sdpa_chunk) <= kv_extent
        token_positions, fill_start = plan_cache_writes(chunk.abs_start, chunk.abs_end, BLOCK_SIZE)
        covered = set(token_positions) | (set(range(fill_start, chunk.abs_end)) if fill_start is not None else set())
        assert covered == set(range(chunk.abs_start, chunk.abs_end))
        assert fill_start is None or fill_start % BLOCK_SIZE == 0
        assert all(p % BLOCK_SIZE != 0 or p == chunk.abs_start for p in token_positions)
        assert all(p // BLOCK_SIZE == chunk.abs_start // BLOCK_SIZE for p in token_positions)


def test_sdpa_chunk_selection_host(expect_error):
    """The padded SDPA K window must never extend past the page-table extent."""
    assert prefill_sdpa_chunk_size(0, 32, 32) == 32
    assert prefill_sdpa_chunk_size(0, 1024, 1024) == 256
    assert prefill_sdpa_chunk_size(4096, 4096, 8256) == 256
    assert prefill_sdpa_chunk_size(8192, 64, 8256) == 64
    assert prefill_sdpa_chunk_size(8192, 64, 8448) == 256
    assert prefill_sdpa_chunk_size(36864, 4096, 40960) == 256
    assert prefill_sdpa_chunk_size(1000 - 1000 % 32, 32, 1024) == 32
    with expect_error(ValueError, "cannot hold chunk"):
        prefill_sdpa_chunk_size(8192, 96, 8256)
    assert decode_sdpa_k_chunk_size(40960) == 128
    assert decode_sdpa_k_chunk_size(1056) == 32
    assert decode_sdpa_k_chunk_size(8256) == 64
    with expect_error(ValueError, "not a multiple of 32"):
        decode_sdpa_k_chunk_size(1040)


@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_cache_write_plan_respects_block_size_host(block_size):
    positions, fill_start = plan_cache_writes(5, 300, block_size)
    assert positions == list(range(5, block_size))
    assert fill_start == block_size
    assert plan_cache_writes(block_size * 3, block_size * 3 + 10, block_size) == ([], block_size * 3)
    assert plan_cache_writes(7, 9, block_size) == ([7, 8], None)


def test_chunked_reference_matches_hf_host(hf_config):
    fixture = _layer(0)
    hidden = fixture.inputs(200, seed=11)
    hf_out = fixture.reference().forward(hidden, 0)
    chunked = chunked_layer_reference(hf_config, fixture.state, hidden, query_chunk=64)
    assert _pcc(hf_out, chunked) > 0.9999
    rows = chunked_layer_reference(hf_config, fixture.state, hidden, query_rows=slice(150, 200), query_chunk=16)
    assert _pcc(hf_out[:, 150:200], rows) > 0.9999


def test_synthetic_weights_host():
    stats = weight_source.load_real_weight_stats()
    for layer_idx in REPRESENTATIVE_LAYERS:
        recorded = stats["layers"][str(layer_idx)]["weights"]
        assert set(recorded) == set(weight_source.LAYER_KEYS)
        synthetic = weight_source.synthetic_layer_state_dict(layer_idx)
        again = weight_source.synthetic_layer_state_dict(layer_idx)
        for key in weight_source.LAYER_KEYS:
            assert list(synthetic[key].shape) == recorded[key]["shape"]
            assert torch.equal(synthetic[key], again[key])
    if weight_source.real_weights_available():
        real = weight_source.load_layer_state(0)
        for key in weight_source.LAYER_KEYS:
            assert tuple(real[key].shape) == tuple(stats["layers"]["0"]["weights"][key]["shape"])


class _GridStub:
    def __init__(self, x: int, y: int):
        self._grid = ttnn.CoreCoord(x, y)

    def compute_with_storage_grid_size(self):
        return self._grid


@pytest.mark.parametrize("grid", [(11, 10), (13, 10), (8, 8)], ids=["grid11x10", "grid13x10", "grid8x8"])
def test_decode_core_grid_for_every_batch_host(grid, expect_error):
    """Every decode batch in 1..32 gets exactly one core per user on each plausible grid."""
    stub = _GridStub(*grid)
    padded = {}
    for batch in range(1, 33):
        physical, x, y = decoder_module.decode_physical_batch(batch, *grid)
        assert batch <= physical <= 32 and x * y == physical and x <= grid[0] and y <= grid[1], (grid, batch)
        assert (x, y) == ((batch, 1) if batch <= 8 else (8, physical // 8)), (grid, batch, x, y)
        if physical != batch:
            padded[batch] = physical
        config = decoder_module._decode_height_sharded_memory_config(stub, physical, HEAD_DIM)
        assert config.shard_spec.grid.num_cores() == physical, (grid, batch)
        assert len(config.shard_spec.grid.ranges()) == 1
        assert tuple(config.shard_spec.shape) == (TILE_SIZE, HEAD_DIM)
    assert padded == {b: ((b + 7) // 8) * 8 for b in range(9, 33) if b % 8}, padded
    with expect_error(ValueError, "decode batch must be in"):
        decoder_module.decode_physical_batch(0, *grid)
    with expect_error(ValueError, "decode batch must be in"):
        decoder_module.decode_physical_batch(33, *grid)
    if padded:
        with expect_error(ValueError, "must be padded"):
            decoder_module._decode_height_sharded_memory_config(stub, next(iter(padded)), HEAD_DIM)


def test_hot_path_fallback_audit():
    forbidden = (
        "torch.",
        "import torch",
        "ttnn.from_torch",
        "ttnn.to_torch",
        "ttnn.full(",
        ".cpu()",
        "ttnn.to_device",
        "synchronize",
    )
    hot = [
        FunctionalDecoder.prefill_forward,
        FunctionalDecoder._prefill_row,
        FunctionalDecoder._pad_rows,
        FunctionalDecoder._pad_sequence,
        FunctionalDecoder._prefill_chunk,
        FunctionalDecoder._attention_prefill,
        FunctionalDecoder._write_prefill_cache,
        FunctionalDecoder._mlp,
        FunctionalDecoder._linear,
        FunctionalDecoder._rms_norm,
        FunctionalDecoder.decode_forward,
        FunctionalDecoder._decode_rope,
        FunctionalDecoder._attention_decode,
    ]
    for method in hot:
        source = inspect.getsource(method)
        for token in forbidden:
            assert token not in source, f"{method.__name__} uses {token}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("layer_idx,seq_len", [(0, 1024), (17, 1000)], ids=["layer0-seq1024", "layer17-seq1000"])
def test_real_weights_prefill_decode_pcc(mesh_device, device_params, layer_idx, seq_len):
    fixture = _layer(layer_idx)
    hidden = fixture.inputs(seq_len + 2, seed=100 + layer_idx)
    reference = fixture.reference()
    hf_prefill = reference.forward(hidden[:, :seq_len], 0)
    hf_decode = [reference.forward(hidden[:, seq_len + i : seq_len + i + 1], seq_len + i) for i in range(2)]

    decoder = _decoder(mesh_device, fixture)
    users = 4
    user_id = 3
    blocks = _blocks_for(seq_len + 2)
    table, page_table = _page_table(mesh_device, users, blocks, seed=1)
    kv_cache = _kv_cache(mesh_device, users * blocks)
    x = _tt(mesh_device, hidden[:, :seq_len].unsqueeze(1))
    out = decoder.prefill_forward(x, page_table=page_table, kv_cache=kv_cache, user_id=user_id)
    out_t = _to_torch(out)[:, 0]
    out_repeat = _to_torch(decoder.prefill_forward(x, page_table=page_table, kv_cache=kv_cache, user_id=user_id))[:, 0]
    prefill_pcc = _pcc(hf_prefill, out_t)
    prefill_delta_pcc = _delta_pcc(hf_prefill, out_t, hidden[:, :seq_len])
    prefill_repeat_bit_equal = bool(torch.equal(out_t, out_repeat))

    user_table = _tt(mesh_device, table[user_id : user_id + 1], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    decode_records = []
    for i in range(2):
        position = seq_len + i
        xd = _tt(mesh_device, hidden[:, position : position + 1].reshape(1, 1, 1, HIDDEN))
        current_pos = _tt(
            mesh_device, torch.tensor([position], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        od = decoder.decode_forward(xd, current_pos=current_pos, page_table=user_table, kv_cache=kv_cache)
        od_t = _to_torch(od).reshape(1, 1, HIDDEN)
        decode_records.append(
            {
                "position": position,
                "pcc": _pcc(hf_decode[i], od_t),
                "delta_pcc": _delta_pcc(hf_decode[i], od_t, hidden[:, position : position + 1]),
            }
        )
    payload = {
        "layer_idx": layer_idx,
        "sequence_length": seq_len,
        "page_table_user": user_id,
        "page_table_users": users,
        "page_table": table.tolist(),
        "pcc_bar": PCC_BAR,
        "input_origin": fixture.input_origin,
        "prefill": {"pcc": prefill_pcc, "delta_pcc": prefill_delta_pcc, "repeat_bit_equal": prefill_repeat_bit_equal},
        "decode": decode_records,
        "provenance": _provenance(
            mesh_device,
            _evidence_command(
                "python -m pytest -q models/autoports/qwen_qwen3_8b/tests/test_functional_decoder.py -k test_real_weights_prefill_decode_pcc"
            ),
            fixture.weight_origin,
        ),
    }
    _write_json(f"pcc_layer{layer_idx}_{LAYER_KIND}.json", payload)
    assert fixture.weight_origin == "real" or os.getenv("QWEN3_FORCE_SYNTHETIC_WEIGHTS") == "1"
    assert (
        fixture.input_origin.startswith("real_") or os.getenv("QWEN3_FORCE_SYNTHETIC_WEIGHTS") == "1"
    ), fixture.input_origin
    assert prefill_pcc >= PCC_BAR, prefill_pcc
    assert prefill_delta_pcc >= DELTA_PCC_BAR, prefill_delta_pcc
    assert prefill_repeat_bit_equal
    for record in decode_records:
        assert record["pcc"] >= PCC_BAR, record
        assert record["delta_pcc"] >= DELTA_PCC_BAR, record


BOUNDARY_LENGTHS = [1, 31, 32, 33, 63, 64, 65, 255, 256, 257, 1023, 1024, 1025, 4095, 4096, 4097, 8192, 8225, 33]


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_prefill_logical_boundary_lengths(mesh_device, device_params):
    fixture = _layer(0)
    decoder = _decoder(mesh_device, fixture)
    users = 4
    user_id = 2
    blocks = _blocks_for(max(BOUNDARY_LENGTHS) + 1)
    table, page_table = _page_table(mesh_device, users, blocks, seed=7)
    kv_cache = _kv_cache(mesh_device, users * blocks)
    results = []
    worst = 1.0
    worst_delta = 1.0
    for seq_len in BOUNDARY_LENGTHS:
        hidden = fixture.inputs(seq_len + 1, seed=200 + seq_len)
        reference = fixture.reference()
        hf_out = reference.forward(hidden[:, :seq_len], 0)
        hf_next = reference.forward(hidden[:, seq_len:], seq_len)
        x = _tt(mesh_device, hidden[:, :seq_len].unsqueeze(1))
        started = time.perf_counter()
        out = decoder.prefill_forward(x, page_table=page_table, kv_cache=kv_cache, user_id=user_id)
        ttnn.synchronize_device(mesh_device)
        elapsed = time.perf_counter() - started
        out_t = _to_torch(out)[:, 0]
        assert out_t.shape == (1, seq_len, HIDDEN)
        user_table = _tt(mesh_device, table[user_id : user_id + 1], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        xd = _tt(mesh_device, hidden[:, seq_len:].reshape(1, 1, 1, HIDDEN))
        current_pos = _tt(
            mesh_device, torch.tensor([seq_len], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        od = _to_torch(
            decoder.decode_forward(xd, current_pos=current_pos, page_table=user_table, kv_cache=kv_cache)
        ).reshape(1, 1, HIDDEN)
        record = {
            "seq_len": seq_len,
            "chunks": len(plan_prefill_chunks(0, seq_len, PREFILL_CHUNK_SIZE)),
            "prefill_pcc": _pcc(hf_out, out_t),
            "prefill_delta_pcc": _delta_pcc(hf_out, out_t, hidden[:, :seq_len]),
            "decode_after_prefill_pcc": _pcc(hf_next, od),
            "decode_after_prefill_delta_pcc": _delta_pcc(hf_next, od, hidden[:, seq_len:]),
            "prefill_wall_s": elapsed,
        }
        results.append(record)
        worst = min(worst, record["prefill_pcc"], record["decode_after_prefill_pcc"])
        worst_delta = min(worst_delta, record["prefill_delta_pcc"], record["decode_after_prefill_delta_pcc"])
    _write_json(
        f"prefill_boundaries_{LAYER_KIND}.json",
        {
            "layer_idx": 0,
            "block_size": BLOCK_SIZE,
            "prefill_chunk_size": PREFILL_CHUNK_SIZE,
            "page_table_user": user_id,
            "page_table": table.tolist(),
            "pcc_bar": PCC_BAR,
            "results": results,
            "worst_pcc": worst,
            "worst_delta_pcc": worst_delta,
            "provenance": _provenance(
                mesh_device,
                _evidence_command(
                    "python -m pytest -q models/autoports/qwen_qwen3_8b/tests/test_functional_decoder.py -k test_prefill_logical_boundary_lengths"
                ),
                fixture.weight_origin,
            ),
        },
    )
    assert worst >= PCC_BAR, results
    assert worst_delta >= DELTA_PCC_BAR, results


CONTINUATION_LENGTH = 4500
CONTINUATION_STARTS = [32, 37, 1000, 4096, 4101, 4499]
CONTINUATION_WINDOW = 256
ROW_DELTA_PCC_BAR = 0.98


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_prefill_continuation(mesh_device, device_params):
    """Prefix-then-continuation must match full prefill; dropping the prefix must fail."""
    fixture = _layer(0)
    decoder = _decoder(mesh_device, fixture)
    seq_len = CONTINUATION_LENGTH
    hidden = fixture.inputs(seq_len, seed=300)
    hf_full = fixture.reference().forward(hidden, 0)
    users = 2
    blocks = _blocks_for(seq_len)
    table, page_table = _page_table(mesh_device, users, blocks, seed=3)
    kv_full = _kv_cache(mesh_device, users * blocks)
    tt_full = _to_torch(
        decoder.prefill_forward(
            _tt(mesh_device, hidden.unsqueeze(1)), page_table=page_table, kv_cache=kv_full, user_id=1
        )
    )[:, 0]
    full_pcc = _pcc(hf_full, tt_full)
    results = []
    for start in CONTINUATION_STARTS:
        kv_cache = _kv_cache(mesh_device, users * blocks)
        decoder.prefill_forward(
            _tt(mesh_device, hidden[:, :start].unsqueeze(1)), page_table=page_table, kv_cache=kv_cache, user_id=1
        )
        cont = decoder.prefill_forward(
            _tt(mesh_device, hidden[:, start:].unsqueeze(1)),
            page_table=page_table,
            kv_cache=kv_cache,
            user_id=1,
            start_pos=start,
        )
        cont_t = _to_torch(cont)[:, 0]
        kv_empty = _kv_cache(mesh_device, users * blocks)
        control = decoder.prefill_forward(
            _tt(mesh_device, hidden[:, start:].unsqueeze(1)),
            page_table=page_table,
            kv_cache=kv_empty,
            user_id=1,
            start_pos=start,
        )
        control_t = _to_torch(control)[:, 0]
        kv_fresh = _kv_cache(mesh_device, users * blocks)
        fresh = decoder.prefill_forward(
            _tt(mesh_device, hidden[:, start:].unsqueeze(1)), page_table=page_table, kv_cache=kv_fresh, user_id=1
        )
        fresh_t = _to_torch(fresh)[:, 0]
        window_len = min(CONTINUATION_WINDOW, seq_len - start)
        rows = list(range(window_len)) + ([seq_len - start - 1] if seq_len - start - 1 >= window_len else [])
        resumed_in = hidden[:, start:]

        def metrics(actual: torch.Tensor) -> dict:
            ref_delta = hf_full[:, start:][:, rows].to(torch.float32) - resumed_in[:, rows].to(torch.float32)
            act_delta = actual[:, rows].to(torch.float32) - resumed_in[:, rows].to(torch.float32)
            per_row = [_pcc(ref_delta[0, r], act_delta[0, r]) for r in range(len(rows))]
            return {
                "window_rows": [start + r for r in rows],
                "window_row_delta_pcc_min_vs_hf": min(per_row),
                "window_row_delta_pcc_mean_vs_hf": sum(per_row) / len(per_row),
                "window_pcc_vs_hf": _pcc(hf_full[:, start:][:, rows], actual[:, rows]),
                "window_delta_pcc_vs_hf": _delta_pcc(hf_full[:, start:][:, rows], actual[:, rows], resumed_in[:, rows]),
                "window_delta_pcc_vs_tt_full": _delta_pcc(
                    tt_full[:, start:][:, rows], actual[:, rows], resumed_in[:, rows]
                ),
                "all_rows_pcc_vs_hf": _pcc(hf_full[:, start:], actual),
                "all_rows_delta_pcc_vs_hf": _delta_pcc(hf_full[:, start:], actual, resumed_in),
            }

        results.append(
            {
                "start": start,
                "tile_aligned": start % TILE_SIZE == 0,
                "resumed_tokens": seq_len - start,
                "continuation": metrics(cont_t),
                "control_no_history": metrics(control_t),
                "control_fresh_suffix": metrics(fresh_t),
            }
        )
    _write_json(
        f"prefill_continuation_{LAYER_KIND}.json",
        {
            "layer_idx": 0,
            "sequence_length": seq_len,
            "starts": CONTINUATION_STARTS,
            "window": CONTINUATION_WINDOW,
            "full_prefill_pcc_vs_hf": full_pcc,
            "delta_pcc_bar": DELTA_PCC_BAR,
            "row_delta_pcc_bar": ROW_DELTA_PCC_BAR,
            "results": results,
            "provenance": _provenance(
                mesh_device,
                _evidence_command(
                    "python -m pytest -q models/autoports/qwen_qwen3_8b/tests/test_functional_decoder.py -k test_prefill_continuation"
                ),
                fixture.weight_origin,
            ),
        },
    )
    assert full_pcc >= PCC_BAR
    for record in results:
        cont = record["continuation"]
        assert cont["all_rows_pcc_vs_hf"] >= PCC_BAR, record
        assert cont["window_delta_pcc_vs_hf"] >= DELTA_PCC_BAR, record
        assert cont["window_delta_pcc_vs_tt_full"] >= DELTA_PCC_BAR, record
        assert cont["window_row_delta_pcc_min_vs_hf"] >= ROW_DELTA_PCC_BAR, record
        assert record["control_no_history"]["window_row_delta_pcc_min_vs_hf"] < ROW_DELTA_PCC_BAR, record
        assert record["control_fresh_suffix"]["window_row_delta_pcc_min_vs_hf"] < ROW_DELTA_PCC_BAR, record


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_multi_row_prefill(mesh_device, device_params):
    """One prefill call with rows of different lengths, starts and page-table rows."""
    fixture = _layer(0)
    decoder = _decoder(mesh_device, fixture)
    lens = [5, 200, 4097, 33]
    starts = [0, 0, 0, 70]
    user_ids = [3, 0, 2, 1]
    users = 4
    blocks = _blocks_for(max(s + n for s, n in zip(starts, lens)))
    table, page_table = _page_table(mesh_device, users, blocks, seed=5)
    kv_cache = _kv_cache(mesh_device, users * blocks)
    s_max = max(lens)
    hiddens = []
    hf_outs = []
    for row, (n, start) in enumerate(zip(lens, starts)):
        hidden = fixture.inputs(start + n, seed=400 + row)
        reference = fixture.reference()
        if start:
            reference.forward(hidden[:, :start], 0)
            decoder.prefill_forward(
                _tt(mesh_device, hidden[:, :start].unsqueeze(1)),
                page_table=page_table,
                kv_cache=kv_cache,
                user_id=user_ids[row],
            )
        hf_outs.append(reference.forward(hidden[:, start:], start))
        hiddens.append(hidden[:, start:])
    batch = torch.zeros(len(lens), 1, s_max, HIDDEN, dtype=torch.bfloat16)
    for row, hidden in enumerate(hiddens):
        batch[row, 0, : hidden.shape[1]] = hidden[0]
    out = decoder.prefill_forward(
        _tt(mesh_device, batch),
        page_table=page_table,
        kv_cache=kv_cache,
        user_id=user_ids,
        start_pos=starts,
        seq_lens=lens,
    )
    out_t = _to_torch(out)
    assert out_t.shape == (len(lens), 1, s_max, HIDDEN)
    results = []
    for row, n in enumerate(lens):
        results.append(
            {
                "row": row,
                "seq_len": n,
                "start_pos": starts[row],
                "user_id": user_ids[row],
                "pcc": _pcc(hf_outs[row], out_t[row : row + 1, 0, :n]),
                "delta_pcc": _delta_pcc(hf_outs[row], out_t[row : row + 1, 0, :n], hiddens[row]),
                "padding_is_zero": bool((out_t[row, 0, n:] == 0).all()) if n < s_max else True,
            }
        )
    decode_records = []
    tokens = []
    positions = []
    for row, n in enumerate(lens):
        tokens.append(fixture.inputs(1, seed=500 + row))
        positions.append(starts[row] + n)
    order = user_ids
    decode_table = _tt(mesh_device, table[order], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    xd = _tt(mesh_device, torch.cat(tokens, dim=1).reshape(1, 1, len(lens), HIDDEN))
    current_pos = _tt(
        mesh_device, torch.tensor(positions, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    od = _to_torch(decoder.decode_forward(xd, current_pos=current_pos, page_table=decode_table, kv_cache=kv_cache))
    for row, n in enumerate(lens):
        reference = fixture.reference()
        full = torch.cat([fixture.inputs(starts[row] + n, seed=400 + row), tokens[row]], dim=1)
        reference.forward(full[:, : starts[row] + n], 0)
        hf_next = reference.forward(full[:, starts[row] + n :], starts[row] + n)
        decode_records.append(
            {"row": row, "position": positions[row], "pcc": _pcc(hf_next, od[0, 0, row].reshape(1, 1, HIDDEN))}
        )
    same_tile_lens = [33, 60]
    same_tile_users = [1, 3]
    kv_same_tile = _kv_cache(mesh_device, users * blocks)
    same_tile_hidden = [fixture.inputs(n, seed=550 + row) for row, n in enumerate(same_tile_lens)]
    same_tile_batch = torch.zeros(2, 1, max(same_tile_lens), HIDDEN, dtype=torch.bfloat16)
    for row, hidden in enumerate(same_tile_hidden):
        same_tile_batch[row, 0, : hidden.shape[1]] = hidden[0]
    same_tile_input = _tt(mesh_device, same_tile_batch)
    same_tile_records = []
    for repeat in range(2):
        out_same = _to_torch(
            decoder.prefill_forward(
                same_tile_input,
                page_table=page_table,
                kv_cache=kv_same_tile,
                user_id=same_tile_users,
                seq_lens=same_tile_lens,
            )
        )
        for row, n in enumerate(same_tile_lens):
            hf_row = fixture.reference().forward(same_tile_hidden[row], 0)
            same_tile_records.append(
                {
                    "repeat": repeat,
                    "row": row,
                    "seq_len": n,
                    "pcc": _pcc(hf_row, out_same[row : row + 1, 0, :n]),
                    "padding_is_zero": bool((out_same[row, 0, n:] == 0).all()) if n < max(same_tile_lens) else True,
                }
            )
    _write_json(
        f"prefill_multi_row_{LAYER_KIND}.json",
        {
            "layer_idx": 0,
            "rows": results,
            "decode_after_rows": decode_records,
            "same_tile_padding_rows": same_tile_records,
            "page_table": table.tolist(),
            "provenance": _provenance(
                mesh_device,
                _evidence_command(
                    "python -m pytest -q models/autoports/qwen_qwen3_8b/tests/test_functional_decoder.py -k test_multi_row_prefill"
                ),
                fixture.weight_origin,
            ),
        },
    )
    for record in results:
        assert record["pcc"] >= PCC_BAR and record["delta_pcc"] >= DELTA_PCC_BAR and record["padding_is_zero"], record
    for record in decode_records:
        assert record["pcc"] >= PCC_BAR, record
    for record in same_tile_records:
        assert record["pcc"] >= PCC_BAR and record["padding_is_zero"], record


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_block_size_64_prefill_continuation_decode(mesh_device, device_params):
    """A 64-token cache block: fresh prefill, continuation at a tile-unaligned start, then decode."""
    fixture = _layer(0)
    decoder = _decoder(mesh_device, fixture)
    block_size = 64
    seq_len = 300
    start = 37
    users = 2
    user_id = 1
    blocks = (seq_len + 1 + block_size - 1) // block_size
    table, page_table = _page_table(mesh_device, users, blocks, seed=23)
    hidden = fixture.inputs(seq_len + 1, seed=1100)
    reference = fixture.reference()
    hf_full = reference.forward(hidden[:, :seq_len], 0)
    hf_next = reference.forward(hidden[:, seq_len:], seq_len)
    kv_full = _kv_cache(mesh_device, users * blocks, block_size=block_size)
    tt_full = _to_torch(
        decoder.prefill_forward(
            _tt(mesh_device, hidden[:, :seq_len].unsqueeze(1)), page_table=page_table, kv_cache=kv_full, user_id=user_id
        )
    )[:, 0]
    kv_cont = _kv_cache(mesh_device, users * blocks, block_size=block_size)
    decoder.prefill_forward(
        _tt(mesh_device, hidden[:, :start].unsqueeze(1)), page_table=page_table, kv_cache=kv_cont, user_id=user_id
    )
    cont = _to_torch(
        decoder.prefill_forward(
            _tt(mesh_device, hidden[:, start:seq_len].unsqueeze(1)),
            page_table=page_table,
            kv_cache=kv_cont,
            user_id=user_id,
            start_pos=start,
        )
    )[:, 0]
    resumed_in = hidden[:, start:seq_len]
    ref_delta = hf_full[:, start:].to(torch.float32) - resumed_in.to(torch.float32)
    act_delta = cont.to(torch.float32) - resumed_in.to(torch.float32)
    per_row = [_pcc(ref_delta[0, r], act_delta[0, r]) for r in range(seq_len - start)]
    user_table = _tt(mesh_device, table[user_id : user_id + 1], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    xd = _tt(mesh_device, hidden[:, seq_len:].reshape(1, 1, 1, HIDDEN))
    current_pos = _tt(
        mesh_device, torch.tensor([seq_len], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    od = _to_torch(
        decoder.decode_forward(xd, current_pos=current_pos, page_table=user_table, kv_cache=kv_cont)
    ).reshape(1, 1, HIDDEN)
    token_positions, fill_start = plan_cache_writes(start, seq_len, block_size)
    payload = {
        "layer_idx": 0,
        "block_size": block_size,
        "sequence_length": seq_len,
        "continuation_start": start,
        "token_by_token_rows": token_positions,
        "fill_start": fill_start,
        "page_table": table.tolist(),
        "full_prefill_pcc_vs_hf": _pcc(hf_full, tt_full),
        "continuation_pcc_vs_hf": _pcc(hf_full[:, start:], cont),
        "continuation_delta_pcc_vs_tt_full": _delta_pcc(tt_full[:, start:], cont, resumed_in),
        "continuation_row_delta_pcc_min_vs_hf": min(per_row),
        "decode_after_continuation_pcc_vs_hf": _pcc(hf_next, od),
        "decode_k_chunk": decoder_module.decode_sdpa_k_chunk_size(blocks * block_size),
        "provenance": _provenance(
            mesh_device,
            _evidence_command(
                "python -m pytest -q models/autoports/qwen_qwen3_8b/tests/test_functional_decoder.py -k test_block_size_64_prefill_continuation_decode"
            ),
            fixture.weight_origin,
        ),
    }
    _write_json(f"block64_{LAYER_KIND}.json", payload)
    assert len(token_positions) == block_size - start and fill_start == block_size
    assert payload["full_prefill_pcc_vs_hf"] >= PCC_BAR
    assert payload["continuation_pcc_vs_hf"] >= PCC_BAR
    assert payload["continuation_delta_pcc_vs_tt_full"] >= DELTA_PCC_BAR
    assert payload["continuation_row_delta_pcc_min_vs_hf"] >= ROW_DELTA_PCC_BAR
    assert payload["decode_after_continuation_pcc_vs_hf"] >= PCC_BAR


DECODE_LENGTH_CHOICES = [1, 31, 32, 33, 100, 255, 256, 257, 600]


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("batch", [1, 13, 31, 32], ids=["batch1", "batch13", "batch31", "batch32"])
def test_traced_decode(mesh_device, device_params, batch):
    """Eager and traced paged decode for a batch of users with distinct positions and permuted pages."""
    fixture = _layer(0)
    decoder = _decoder(mesh_device, fixture)
    generator = torch.Generator().manual_seed(600 + batch)
    lens = [
        DECODE_LENGTH_CHOICES[int(torch.randint(0, len(DECODE_LENGTH_CHOICES), (1,), generator=generator))]
        for _ in range(batch)
    ]
    if batch == 1:
        lens = [257]
    blocks = _blocks_for(max(lens) + 3)
    table, page_table = _page_table(mesh_device, batch, blocks, seed=9)
    kv_cache = _kv_cache(mesh_device, batch * blocks)
    references = []
    steps = []
    for user in range(batch):
        hidden = fixture.inputs(lens[user] + 3, seed=700 + user)
        reference = fixture.reference()
        reference.forward(hidden[:, : lens[user]], 0)
        decoder.prefill_forward(
            _tt(mesh_device, hidden[:, : lens[user]].unsqueeze(1)),
            page_table=page_table,
            kv_cache=kv_cache,
            user_id=user,
        )
        references.append(reference)
        steps.append(hidden[:, lens[user] :])

    def host_inputs(step: int):
        tokens = torch.cat([steps[u][:, step : step + 1] for u in range(batch)], dim=1).reshape(1, 1, batch, HIDDEN)
        positions = torch.tensor([lens[u] + step for u in range(batch)], dtype=torch.int32)
        return _host(tokens), _host(positions, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    def hf_step(step: int) -> torch.Tensor:
        return torch.cat(
            [references[u].forward(steps[u][:, step : step + 1], lens[u] + step) for u in range(batch)], dim=1
        )

    tokens0, positions0 = host_inputs(0)
    xd = ttnn.to_device(tokens0, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    current_pos = ttnn.to_device(positions0, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    hf0 = hf_step(0)
    eager = _to_torch(decoder.decode_forward(xd, current_pos=current_pos, page_table=page_table, kv_cache=kv_cache))
    eager_pcc = [_pcc(hf0[:, u], eager[0, 0, u]) for u in range(batch)]

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_out = decoder.decode_forward(xd, current_pos=current_pos, page_table=page_table, kv_cache=kv_cache)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    replay0 = _to_torch(traced_out)
    replay_vs_eager = _pcc(eager, replay0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    repeat_bit_equal = bool(torch.equal(replay0, _to_torch(traced_out)))

    step_records = []
    for step in (1, 2):
        tokens, positions = host_inputs(step)
        ttnn.copy_host_to_device_tensor(tokens, xd)
        ttnn.copy_host_to_device_tensor(positions, current_pos)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        replay = _to_torch(traced_out)
        hf = hf_step(step)
        step_records.append(
            {
                "step": step,
                "positions": [lens[u] + step for u in range(batch)],
                "replay_pcc_vs_hf_per_user": [_pcc(hf[:, u], replay[0, 0, u]) for u in range(batch)],
                "replay_differs_from_previous_step": not torch.equal(replay, replay0),
            }
        )
        replay0 = replay
    ttnn.release_trace(mesh_device, trace_id)
    payload = {
        "layer_idx": 0,
        "batch": batch,
        "prefilled_lengths": lens,
        "page_table": table.tolist(),
        "eager_pcc_vs_hf_per_user": eager_pcc,
        "replay_vs_eager_pcc": replay_vs_eager,
        "repeat_replay_bit_equal": repeat_bit_equal,
        "updated_input_steps": step_records,
        "trace_region_size": device_params.get("trace_region_size"),
        "provenance": _provenance(
            mesh_device,
            _evidence_command(
                f"python -m pytest -q models/autoports/qwen_qwen3_8b/tests/test_functional_decoder.py -k 'test_traced_decode and batch{batch}'"
            ),
            fixture.weight_origin,
        ),
    }
    _write_json(f"trace_{LAYER_KIND}_batch{batch}.json", payload)
    assert min(eager_pcc) >= PCC_BAR, eager_pcc
    assert replay_vs_eager >= REPLAY_PCC_BAR
    assert repeat_bit_equal
    for record in step_records:
        assert min(record["replay_pcc_vs_hf_per_user"]) >= PCC_BAR, record
        assert record["replay_differs_from_previous_step"]


def _context_tests_enabled() -> bool:
    return os.getenv("QWEN3_CONTEXT_TESTS") == "1"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("seq_len", [CONTEXT, CONTEXT - 1], ids=["ctx40960", "ctx40959"])
def test_advertised_context_prefill_and_decode(mesh_device, device_params, seq_len):
    """Real-weight prefill at the advertised context, then traced decode at the last position."""
    if not _context_tests_enabled():
        pytest.skip("set QWEN3_CONTEXT_TESTS=1 to run the advertised-context tests")
    fixture = _layer(0)
    assert fixture.weight_origin == "real"
    decoder = _decoder(mesh_device, fixture)
    hidden = fixture.inputs(CONTEXT, seed=800)
    users = 2
    user_id = 1
    blocks = _blocks_for(CONTEXT)
    table, page_table = _page_table(mesh_device, users, blocks, seed=13)
    kv_cache = _kv_cache(mesh_device, users * blocks)
    x = _tt(mesh_device, hidden[:, :seq_len].unsqueeze(1))
    started = time.perf_counter()
    out = decoder.prefill_forward(x, page_table=page_table, kv_cache=kv_cache, user_id=user_id)
    ttnn.synchronize_device(mesh_device)
    prefill_wall = time.perf_counter() - started
    out_t = _to_torch(out)[:, 0]
    assert out_t.shape == (1, seq_len, HIDDEN)
    finite = bool(torch.isfinite(out_t.to(torch.float32)).all())
    windows = {
        "last256": slice(seq_len - 256, seq_len),
        "mid256": slice(seq_len // 2, seq_len // 2 + 256),
        "first256": slice(0, 256),
    }
    window_pcc = {}
    for name, rows in windows.items():
        reference = chunked_layer_reference(
            fixture.config, fixture.state, hidden[:, :seq_len], query_rows=rows, query_chunk=256
        )
        window_pcc[name] = {
            "pcc": _pcc(reference, out_t[:, rows]),
            "delta_pcc": _delta_pcc(reference, out_t[:, rows], hidden[:, :seq_len][:, rows]),
        }
    payload = {
        "layer_idx": 0,
        "sequence_length": seq_len,
        "chunks": len(plan_prefill_chunks(0, seq_len, PREFILL_CHUNK_SIZE)),
        "prefill_chunk_size": PREFILL_CHUNK_SIZE,
        "output_finite": finite,
        "prefill_wall_s": prefill_wall,
        "window_pcc": window_pcc,
        "cache_blocks_total": users * blocks,
        "kv_cache_bytes_per_user": 2 * blocks * NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM * 2,
        "page_table_user": user_id,
    }
    decode = None
    if seq_len < CONTEXT:
        position = seq_len
        reference = chunked_layer_reference(
            fixture.config, fixture.state, hidden, query_rows=slice(position, position + 1), query_chunk=1
        )
        user_table = _tt(mesh_device, table[user_id : user_id + 1], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        xd = _tt(mesh_device, hidden[:, position : position + 1].reshape(1, 1, 1, HIDDEN))
        current_pos = _tt(
            mesh_device, torch.tensor([position], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        eager = _to_torch(decoder.decode_forward(xd, current_pos=current_pos, page_table=user_table, kv_cache=kv_cache))
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        traced = decoder.decode_forward(xd, current_pos=current_pos, page_table=user_table, kv_cache=kv_cache)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        replay = _to_torch(traced)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        repeat = _to_torch(traced)
        ttnn.release_trace(mesh_device, trace_id)
        decode = {
            "position": position,
            "eager_pcc_vs_reference": _pcc(reference, eager.reshape(1, 1, HIDDEN)),
            "eager_delta_pcc_vs_reference": _delta_pcc(
                reference, eager.reshape(1, 1, HIDDEN), hidden[:, position : position + 1]
            ),
            "replay_vs_eager_pcc": _pcc(eager, replay),
            "repeat_replay_bit_equal": bool(torch.equal(replay, repeat)),
        }
        payload["decode"] = decode
    payload["provenance"] = _provenance(
        mesh_device,
        _evidence_command(
            f"QWEN3_CONTEXT_TESTS=1 python -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_functional_decoder.py -k 'test_advertised_context_prefill_and_decode and ctx{seq_len}'"
        ),
        fixture.weight_origin,
    )
    _write_json(f"prefill_capacity_{LAYER_KIND}_{seq_len}.json", payload)
    assert finite
    for name, record in window_pcc.items():
        assert record["pcc"] >= PCC_BAR and record["delta_pcc"] >= DELTA_PCC_BAR, (name, record)
    if decode is not None:
        assert (
            decode["eager_pcc_vs_reference"] >= PCC_BAR and decode["eager_delta_pcc_vs_reference"] >= DELTA_PCC_BAR
        ), decode
        assert decode["replay_vs_eager_pcc"] >= REPLAY_PCC_BAR and decode["repeat_replay_bit_equal"], decode


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_advertised_context_batch32_traced_decode(mesh_device, device_params):
    """Batch-32 traced decode at position 40959 with a full-context cache for every user."""
    if not _context_tests_enabled():
        pytest.skip("set QWEN3_CONTEXT_TESTS=1 to run the advertised-context tests")
    fixture = _layer(0)
    decoder = _decoder(mesh_device, fixture)
    batch = 32
    live_user = 5
    blocks = _blocks_for(CONTEXT)
    table, page_table = _page_table(mesh_device, batch, blocks, seed=17)
    kv_cache = _kv_cache(mesh_device, batch * blocks)
    hidden = fixture.inputs(CONTEXT, seed=800)
    position = CONTEXT - 1
    decoder.prefill_forward(
        _tt(mesh_device, hidden[:, :position].unsqueeze(1)), page_table=page_table, kv_cache=kv_cache, user_id=live_user
    )
    reference = chunked_layer_reference(
        fixture.config, fixture.state, hidden, query_rows=slice(position, position + 1), query_chunk=1
    )
    tokens = hidden[:, position : position + 1].reshape(1, 1, 1, HIDDEN).repeat(1, 1, batch, 1)
    xd = _tt(mesh_device, tokens)
    current_pos = _tt(
        mesh_device, torch.full((batch,), position, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    eager = _to_torch(decoder.decode_forward(xd, current_pos=current_pos, page_table=page_table, kv_cache=kv_cache))
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = decoder.decode_forward(xd, current_pos=current_pos, page_table=page_table, kv_cache=kv_cache)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    started = time.perf_counter()
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    replay_wall = time.perf_counter() - started
    replay = _to_torch(traced)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    repeat = _to_torch(traced)
    ttnn.release_trace(mesh_device, trace_id)
    payload = {
        "layer_idx": 0,
        "batch": batch,
        "position": position,
        "live_user": live_user,
        "cache_blocks_total": batch * blocks,
        "kv_cache_bytes_total": 2 * batch * blocks * NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM * 2,
        "live_user_eager_pcc_vs_reference": _pcc(reference, eager[0, 0, live_user].reshape(1, 1, HIDDEN)),
        "live_user_eager_delta_pcc_vs_reference": _delta_pcc(
            reference, eager[0, 0, live_user].reshape(1, 1, HIDDEN), hidden[:, position : position + 1]
        ),
        "all_users_finite": bool(torch.isfinite(eager.to(torch.float32)).all()),
        "replay_vs_eager_pcc": _pcc(eager, replay),
        "repeat_replay_bit_equal": bool(torch.equal(replay, repeat)),
        "replay_wall_s": replay_wall,
        "provenance": _provenance(
            mesh_device,
            _evidence_command(
                "QWEN3_CONTEXT_TESTS=1 python -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_functional_decoder.py -k test_advertised_context_batch32_traced_decode"
            ),
            fixture.weight_origin,
        ),
    }
    _write_json(f"advertised_context_decode_{LAYER_KIND}.json", payload)
    assert (
        payload["live_user_eager_pcc_vs_reference"] >= PCC_BAR
        and payload["live_user_eager_delta_pcc_vs_reference"] >= DELTA_PCC_BAR
    )
    assert (
        payload["all_users_finite"]
        and payload["replay_vs_eager_pcc"] >= REPLAY_PCC_BAR
        and payload["repeat_replay_bit_equal"]
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("batch", [1], ids=["batch1"])
def test_functional_decoder_perf_profile(mesh_device, device_params, batch):
    if os.getenv("QWEN3_PERF") != "1":
        pytest.skip("set QWEN3_PERF=1 to run the profiler harness")
    try:
        from tracy import signpost
    except ImportError:
        signpost = lambda *_, **__: None

    fixture = _layer(0)
    decoder = _decoder(mesh_device, fixture)
    seq_len = int(os.getenv("QWEN3_PERF_SEQ_LEN", "1024"))
    hidden = fixture.inputs(seq_len + 1, seed=900)
    blocks = _blocks_for(seq_len + 1)
    table, page_table = _page_table(mesh_device, batch, blocks, seed=19)
    kv_cache = _kv_cache(mesh_device, batch * blocks)
    prefill_args = dict(
        hidden_states=_tt(mesh_device, hidden[:, :seq_len].unsqueeze(1)),
        page_table=page_table,
        kv_cache=kv_cache,
        user_id=0,
    )
    decode_args = dict(
        hidden_states=_tt(mesh_device, hidden[:, seq_len:].reshape(1, 1, 1, HIDDEN).repeat(1, 1, batch, 1)),
        current_pos=_tt(
            mesh_device,
            torch.full((batch,), seq_len, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        page_table=page_table,
        kv_cache=kv_cache,
    )
    decoder.prefill_forward(**prefill_args)
    decoder.decode_forward(**decode_args)
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)

    case_id = f"layer0_{LAYER_KIND}_seq{seq_len}_batch{batch}"
    measured = {}
    signpost(f"PERF_PREFILL_{case_id}")
    started = time.perf_counter()
    prefill_out = decoder.prefill_forward(**prefill_args)
    ttnn.synchronize_device(mesh_device)
    measured["prefill_host_ms"] = (time.perf_counter() - started) * 1000
    signpost(f"PERF_PREFILL_{case_id}_END")
    ttnn.ReadDeviceProfiler(mesh_device)

    cont_start = 37
    cont_len = 100
    cont_table, cont_page_table = _page_table(mesh_device, 1, _blocks_for(cont_start + cont_len), seed=29)
    cont_cache = _kv_cache(mesh_device, cont_page_table.shape[1])
    cont_hidden = fixture.inputs(cont_start + cont_len, seed=901)
    decoder.prefill_forward(
        _tt(mesh_device, cont_hidden[:, :cont_start].unsqueeze(1)), page_table=cont_page_table, kv_cache=cont_cache
    )
    cont_args = dict(
        hidden_states=_tt(mesh_device, cont_hidden[:, cont_start:].unsqueeze(1)),
        page_table=cont_page_table,
        kv_cache=cont_cache,
        start_pos=cont_start,
    )
    decoder.prefill_forward(**cont_args)
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)
    signpost(f"PERF_PREFILL_CONT_{case_id}")
    started = time.perf_counter()
    cont_out = decoder.prefill_forward(**cont_args)
    ttnn.synchronize_device(mesh_device)
    measured["prefill_continuation_host_ms"] = (time.perf_counter() - started) * 1000
    measured["prefill_continuation"] = {"start_pos": cont_start, "tokens": cont_len}
    signpost(f"PERF_PREFILL_CONT_{case_id}_END")
    ttnn.ReadDeviceProfiler(mesh_device)
    assert cont_out.shape[-2] == cont_len

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_out = decoder.decode_forward(**decode_args)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    warmups = int(os.getenv("QWEN3_PERF_TRACE_WARMUPS", "2"))
    iterations = int(os.getenv("QWEN3_PERF_TRACE_ITERATIONS", "1"))
    for _ in range(warmups):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)
    signpost(f"PERF_DECODE_{case_id}")
    started = time.perf_counter()
    for _ in range(iterations):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    measured["decode_trace_host_ms"] = (time.perf_counter() - started) * 1000 / iterations
    signpost(f"PERF_DECODE_{case_id}_END")
    ttnn.ReadDeviceProfiler(mesh_device)
    measured["decode_trace_warmups"] = warmups
    measured["decode_trace_iterations"] = iterations
    ttnn.release_trace(mesh_device, trace_id)
    assert prefill_out.shape[-2] == seq_len and traced_out.shape[-2] == batch
    _write_json(
        f"{case_id}_host_timings.json",
        {
            "layer_idx": 0,
            "sequence_length": seq_len,
            "decode_position": seq_len,
            "decode_batch": batch,
            "signposts": {
                "prefill": [f"PERF_PREFILL_{case_id}", f"PERF_PREFILL_{case_id}_END"],
                "prefill_continuation": [f"PERF_PREFILL_CONT_{case_id}", f"PERF_PREFILL_CONT_{case_id}_END"],
                "decode": [f"PERF_DECODE_{case_id}", f"PERF_DECODE_{case_id}_END"],
            },
            "provenance": _provenance(
                mesh_device,
                _evidence_command(
                    "QWEN3_PERF=1 python -m tracy -r -p -v --check-exit-code -o generated/profiler/qwen3_functional -m pytest -q -s "
                    f"'models/autoports/qwen_qwen3_8b/tests/test_functional_decoder.py::test_functional_decoder_perf_profile[blackhole-batch{batch}-device_params0-mesh_device0]'"
                ),
                fixture.weight_origin,
            ),
            **measured,
        },
    )
