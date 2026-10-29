# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Optimized-decoder tests for Qwen/Qwen3-8B on one Blackhole chip (1x1 mesh).

The functional-decoder gates (real-weight PCC, boundary lengths, continuation,
multi-row prefill, block size 64, traced decode, advertised context, profiler
harness) are reused from ``test_functional_decoder`` and bound to
``OptimizedDecoder`` for the whole module; their evidence JSON lands under
``doc/optimized_decoder``. Optimized-specific tests cover the on-device
comparison against the fused decoder, the sharded residual contract, the
runtime precision policy, repeated-run stability and the hot-path audit.

Environment switches are the same as for the functional tests
(``QWEN3_CONTEXT_TESTS``, ``QWEN3_PERF``, ``QWEN3_EVIDENCE_COMMAND``,
``QWEN3_EVIDENCE_SUFFIX``).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_8b.tests import test_functional_decoder as base
from models.autoports.qwen_qwen3_8b.tests import test_fused_decoder as fused_tests
from models.autoports.qwen_qwen3_8b.tt import fused_decoder as fused_module
from models.autoports.qwen_qwen3_8b.tt import optimized_decoder as optimized_module
from models.autoports.qwen_qwen3_8b.tt.functional_decoder import decode_physical_batch
from models.autoports.qwen_qwen3_8b.tt.fused_decoder import FusedDecoder
from models.autoports.qwen_qwen3_8b.tt.optimized_decoder import (
    DEFAULT_POLICY,
    DTYPE_NAMES,
    OptimizedDecoder,
    dram_sharded_program_config,
    lead_update_core_sets,
    same_layout,
    width_sharded_memory_config,
)

ARTIFACT_DIR = base.MODEL_DIR / "doc" / "optimized_decoder"
KV_CACHE_DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}
KV_CACHE_DTYPE = os.getenv("QWEN3_KV_CACHE_DTYPE", "bfp8")
HIDDEN = base.HIDDEN
PCC_BAR = base.PCC_BAR
DELTA_PCC_BAR = base.DELTA_PCC_BAR
CACHE_PCC_BAR = 0.99
BATCH_USER_PCC_BAR = 0.99
DEVICE_PARAMS = base.DEVICE_PARAMS


def _optimized_decoder(mesh_device, fixture: base.LayerFixture) -> OptimizedDecoder:
    return OptimizedDecoder.from_state_dict(
        fixture.state, hf_config=fixture.config, layer_idx=fixture.layer_idx, mesh_device=mesh_device
    )


def _fused_decoder(mesh_device, fixture: base.LayerFixture) -> FusedDecoder:
    return FusedDecoder.from_state_dict(
        fixture.state, hf_config=fixture.config, layer_idx=fixture.layer_idx, mesh_device=mesh_device
    )


_original_provenance = base._provenance
_original_evidence_command = base._evidence_command
_original_kv_cache = base._kv_cache


def _optimized_kv_cache(mesh_device, num_blocks: int, block_size: int = base.BLOCK_SIZE):
    return OptimizedDecoder.allocate_kv_cache(
        mesh_device,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=base.NUM_KV_HEADS,
        head_dim=base.HEAD_DIM,
        dtype=KV_CACHE_DTYPES[KV_CACHE_DTYPE],
    )


def _optimized_provenance(mesh_device, exact_command: str, weight_origin: str) -> dict:
    record = _original_provenance(mesh_device, exact_command, weight_origin)
    for name, module in (("functional", base.decoder_module), ("fused", fused_module), ("optimized", optimized_module)):
        path = Path(module.__file__).resolve()
        record[f"{name}_decoder_path"] = str(path)
        record[f"{name}_decoder_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    record["decoder"] = "optimized"
    record["policy"] = DEFAULT_POLICY.describe()
    record["kv_cache_dtype"] = KV_CACHE_DTYPE
    record["test_sha256"] = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    record["functional_test_sha256"] = hashlib.sha256(Path(base.__file__).resolve().read_bytes()).hexdigest()
    return record


def _optimized_evidence_command(default: str) -> str:
    return _original_evidence_command(default.replace("test_functional_decoder", "test_optimized_decoder"))


def _write_optimized_json(name: str, payload: dict) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = os.getenv("QWEN3_EVIDENCE_SUFFIX", "")
    if suffix:
        name = name.replace(".json", f".{suffix}.json")
    path = ARTIFACT_DIR / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


@pytest.fixture(scope="module", autouse=True)
def _bind_base_tests_to_optimized_decoder():
    saved = {
        "_decoder": base._decoder,
        "_write_json": base._write_json,
        "_provenance": base._provenance,
        "_evidence_command": base._evidence_command,
        "_kv_cache": base._kv_cache,
    }
    base._decoder = _optimized_decoder
    base._kv_cache = _optimized_kv_cache
    base._write_json = _write_optimized_json
    base._provenance = _optimized_provenance
    base._evidence_command = _optimized_evidence_command
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(base, name, value)


test_real_weights_prefill_decode_pcc = base.test_real_weights_prefill_decode_pcc
test_prefill_logical_boundary_lengths = base.test_prefill_logical_boundary_lengths
test_prefill_continuation = base.test_prefill_continuation
test_multi_row_prefill = base.test_multi_row_prefill
test_block_size_64_prefill_continuation_decode = base.test_block_size_64_prefill_continuation_decode
test_traced_decode = base.test_traced_decode
test_advertised_context_prefill_and_decode = base.test_advertised_context_prefill_and_decode
test_advertised_context_batch32_traced_decode = base.test_advertised_context_batch32_traced_decode


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("batch", [1, 32], ids=["batch1", "batch32"])
def test_optimized_decoder_perf_profile(mesh_device, device_params, batch):
    base.test_functional_decoder_perf_profile(mesh_device, device_params, batch)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("batch", [1, 32], ids=["batch1", "batch32"])
def test_fused_baseline_perf_profile(mesh_device, device_params, batch):
    """The stage-2 fused decoder measured with this module's harness; the "before" row of the perf tables."""
    saved = (base._decoder, base._provenance, base._kv_cache)

    def fused_provenance(mesh, exact_command, weight_origin):
        record = _optimized_provenance(
            mesh,
            exact_command.replace("test_optimized_decoder_perf_profile", "test_fused_baseline_perf_profile"),
            weight_origin,
        )
        record["decoder"] = "fused_baseline"
        record["kv_cache_dtype"] = "bf16"
        record.pop("policy", None)
        return record

    base._decoder = _fused_decoder
    base._provenance = fused_provenance
    base._kv_cache = _original_kv_cache
    try:
        base.test_functional_decoder_perf_profile(mesh_device, device_params, batch)
    finally:
        base._decoder, base._provenance, base._kv_cache = saved


def test_optimized_hot_path_fallback_audit():
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
        OptimizedDecoder.prefill_forward,
        OptimizedDecoder._prefill_row,
        OptimizedDecoder._pad_rows,
        OptimizedDecoder._pad_sequence,
        OptimizedDecoder._prefill_chunk,
        OptimizedDecoder._prefill_matmul,
        OptimizedDecoder._prefill_matmul_residual,
        OptimizedDecoder._mlp_prefill,
        OptimizedDecoder._attention_prefill,
        OptimizedDecoder._write_prefill_cache,
        OptimizedDecoder._fill_prefill_cache,
        OptimizedDecoder._linear,
        OptimizedDecoder._rms_norm,
        OptimizedDecoder.decode_forward,
        OptimizedDecoder._decode_norm,
        OptimizedDecoder._decode_projection,
        OptimizedDecoder._mlp_decode,
        OptimizedDecoder._decode_head_norm,
        OptimizedDecoder._decode_rope,
        OptimizedDecoder._attention_decode,
        OptimizedDecoder._to,
    ]
    for method in hot:
        source = inspect.getsource(method)
        for token in forbidden:
            assert token not in source, f"{method.__name__} uses {token}"


def test_optimized_overrides_every_measured_path_host():
    for name in (
        "_linear",
        "_prefill_matmul",
        "_prefill_matmul_residual",
        "_write_prefill_cache",
        "decode_forward",
        "_attention_decode",
        "_mlp_decode",
    ):
        assert name in OptimizedDecoder.__dict__, name


def test_policy_geometry_host(expect_error):
    """Every default decode program config is legal for the real shapes and the DRAM bank count."""
    banks = 8
    shapes = {"qkv": (4096, 6144), "o_proj": (4096, 4096), "gate_up": (4096, 12288), "down": (12288, 4096)}
    for name, (k, n) in shapes.items():
        role = getattr(DEFAULT_POLICY, name)
        config = dram_sharded_program_config(role, k, n, banks)
        assert config.per_core_N * role.cores * 32 == n
        assert (k // 32 // role.cores) % role.in0_block_w == 0
        assert role.readers in (1, 2, 3)
        assert config.num_workers_per_dram_bank == role.readers
    stub = base._GridStub(11, 10)
    cfg = width_sharded_memory_config(HIDDEN, DEFAULT_POLICY.residual_cores, stub.compute_with_storage_grid_size())
    assert cfg.shard_spec.grid.num_cores() == DEFAULT_POLICY.residual_cores
    assert tuple(cfg.shard_spec.shape) == (32, HIDDEN // DEFAULT_POLICY.residual_cores)
    for count in (1, 5, 27, 31, 32):
        k_set, v_set = lead_update_core_sets(count, stub.compute_with_storage_grid_size())
        assert k_set.num_cores() == v_set.num_cores() == count
        assert not (fused_tests._cores(k_set) & fused_tests._cores(v_set)), count
    with expect_error(ValueError, "cores"):
        lead_update_core_sets(60, stub.compute_with_storage_grid_size())


def _batch_shard_config(decoder: OptimizedDecoder):
    return decoder.decode_residual_memory_config


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_optimized_matches_fused_on_device(mesh_device, device_params):
    """Optimized and fused graphs agree on real weights for prefill, continuation and decode within the precision policy."""
    fixture = base._layer(0)
    fused = _fused_decoder(mesh_device, fixture)
    optimized = _optimized_decoder(mesh_device, fixture)
    runtime = {
        name: {
            "dtype": str(getattr(optimized.weights, name).dtype),
            "memory_layout": str(getattr(optimized.weights, name).memory_config().memory_layout),
            "buffer_type": str(getattr(optimized.weights, name).memory_config().buffer_type),
            "shard_shape": list(getattr(optimized.weights, name).memory_config().shard_spec.shape),
        }
        for name in ("qkv", "o_proj", "gate_proj", "up_proj", "down_proj")
    }
    for name, dtype_name in optimized.weight_dtypes.items():
        assert getattr(optimized.weights, name).dtype == DTYPE_NAMES[dtype_name], (name, runtime[name])
        assert runtime[name]["memory_layout"].endswith("WIDTH_SHARDED") and runtime[name]["buffer_type"].endswith(
            "DRAM"
        )
    cases = []
    users = 4
    for seq_len, start in ((1024, 0), (300, 0), (37, 0), (4500, 37), (200, 4101)):
        hidden = fixture.inputs(start + seq_len + 4, seed=800 + seq_len)
        blocks = base._blocks_for(start + seq_len + 4)
        table, page_table = base._page_table(mesh_device, users, blocks, seed=11)
        outs = {}
        decodes = {}
        for name, decoder in (("fused", fused), ("optimized", optimized)):
            kv_cache = (_original_kv_cache if name == "fused" else base._kv_cache)(mesh_device, users * blocks)
            if start:
                decoder.prefill_forward(
                    base._tt(mesh_device, hidden[:, :start].unsqueeze(1)),
                    page_table=page_table,
                    kv_cache=kv_cache,
                    user_id=2,
                )
            out = decoder.prefill_forward(
                base._tt(mesh_device, hidden[:, start : start + seq_len].unsqueeze(1)),
                page_table=page_table,
                kv_cache=kv_cache,
                user_id=2,
                start_pos=start,
            )
            outs[name] = base._to_torch(out)[:, 0]
            user_table = base._tt(mesh_device, table[2:3], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            steps = []
            for i in range(2):
                position = start + seq_len + i
                xd = base._tt(mesh_device, hidden[:, position : position + 1].reshape(1, 1, 1, HIDDEN))
                current_pos = base._tt(
                    mesh_device,
                    torch.tensor([position], dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                od = decoder.decode_forward(xd, current_pos=current_pos, page_table=user_table, kv_cache=kv_cache)
                steps.append(base._to_torch(od).reshape(1, 1, HIDDEN))
            decodes[name] = steps
            key_cache, value_cache = kv_cache
            outs[name + "_kcache"] = base._to_torch(key_cache)[table[2].long()]
            outs[name + "_vcache"] = base._to_torch(value_cache)[table[2].long()]
        resumed_in = hidden[:, start : start + seq_len]
        cases.append(
            {
                "seq_len": seq_len,
                "start_pos": start,
                "prefill_pcc": base._pcc(outs["fused"], outs["optimized"]),
                "prefill_delta_pcc": base._delta_pcc(outs["fused"], outs["optimized"], resumed_in),
                "key_cache_pcc": base._pcc(outs["fused_kcache"], outs["optimized_kcache"]),
                "value_cache_pcc": base._pcc(outs["fused_vcache"], outs["optimized_vcache"]),
                "decode_pcc": [base._pcc(a, b) for a, b in zip(decodes["fused"], decodes["optimized"])],
                "decode_delta_pcc": [
                    base._delta_pcc(a, b, hidden[:, start + seq_len + i : start + seq_len + i + 1])
                    for i, (a, b) in enumerate(zip(decodes["fused"], decodes["optimized"]))
                ],
            }
        )
    batch = 32
    lens = [base.DECODE_LENGTH_CHOICES[u % len(base.DECODE_LENGTH_CHOICES)] for u in range(batch)]
    blocks = base._blocks_for(max(lens) + 2)
    table, page_table = base._page_table(mesh_device, batch, blocks, seed=13)
    batch_outs = {}
    hf_rows = []
    for user in range(batch):
        hidden = fixture.inputs(lens[user] + 1, seed=820 + user)
        reference = fixture.reference()
        reference.forward(hidden[:, : lens[user]], 0)
        hf_rows.append(reference.forward(hidden[:, lens[user] :], lens[user])[0])
    hf_batch = torch.cat(hf_rows, dim=0)
    for name, decoder in (("fused", fused), ("optimized", optimized)):
        kv_cache = (_original_kv_cache if name == "fused" else base._kv_cache)(mesh_device, batch * blocks)
        tokens = []
        for user in range(batch):
            hidden = fixture.inputs(lens[user] + 1, seed=820 + user)
            decoder.prefill_forward(
                base._tt(mesh_device, hidden[:, : lens[user]].unsqueeze(1)),
                page_table=page_table,
                kv_cache=kv_cache,
                user_id=user,
            )
            tokens.append(hidden[:, lens[user] :])
        xd = base._tt(mesh_device, torch.cat(tokens, dim=1).reshape(1, 1, batch, HIDDEN))
        current_pos = base._tt(
            mesh_device, torch.tensor(lens, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        batch_outs[name] = base._to_torch(
            decoder.decode_forward(xd, current_pos=current_pos, page_table=page_table, kv_cache=kv_cache)
        )[0, 0]
    batch_record = {
        "batch": batch,
        "prefilled_lengths": lens,
        "decode_pcc_per_user": [base._pcc(batch_outs["fused"][u], batch_outs["optimized"][u]) for u in range(batch)],
        "decode_delta_pcc_per_user": [
            base._delta_pcc(batch_outs["fused"][u], batch_outs["optimized"][u], torch.cat(tokens, dim=1)[0, u])
            for u in range(batch)
        ],
        "optimized_pcc_vs_hf_per_user": [base._pcc(hf_batch[u], batch_outs["optimized"][u]) for u in range(batch)],
        "fused_pcc_vs_hf_per_user": [base._pcc(hf_batch[u], batch_outs["fused"][u]) for u in range(batch)],
    }
    _write_optimized_json(
        f"optimized_vs_fused_{base.LAYER_KIND}.json",
        {
            "layer_idx": 0,
            "pcc_bar": PCC_BAR,
            "delta_pcc_bar": DELTA_PCC_BAR,
            "cache_pcc_bar": CACHE_PCC_BAR,
            "batch_user_pcc_bar": BATCH_USER_PCC_BAR,
            "runtime_weights": runtime,
            "kv_cache_dtype": KV_CACHE_DTYPE,
            "weight_bytes_per_token": optimized.weight_bytes_per_token(),
            "weight_bytes_resident": optimized.weight_bytes_resident(),
            "prefill_interleaved_copy_max_rows": optimized.policy.prefill_interleaved_copy_max_rows,
            "prefill_cases": cases,
            "decode_batch32": batch_record,
            "provenance": _optimized_provenance(
                mesh_device,
                _optimized_evidence_command(
                    "python -m pytest -q models/autoports/qwen_qwen3_8b/tests/test_optimized_decoder.py -k test_optimized_matches_fused_on_device"
                ),
                fixture.weight_origin,
            ),
        },
    )
    for record in cases:
        assert record["prefill_pcc"] >= PCC_BAR, record
        assert record["prefill_delta_pcc"] >= DELTA_PCC_BAR, record
        assert record["key_cache_pcc"] >= CACHE_PCC_BAR, record
        assert record["value_cache_pcc"] >= CACHE_PCC_BAR, record
        assert min(record["decode_pcc"]) >= PCC_BAR, record
        assert min(record["decode_delta_pcc"]) >= DELTA_PCC_BAR, record
    assert min(batch_record["decode_pcc_per_user"]) >= BATCH_USER_PCC_BAR, batch_record
    assert min(batch_record["decode_delta_pcc_per_user"]) >= DELTA_PCC_BAR, batch_record
    assert min(batch_record["optimized_pcc_vs_hf_per_user"]) >= PCC_BAR, batch_record


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_sharded_residual_contract(mesh_device, device_params):
    """A residual passed in ``decode_residual_memory_config`` gives the same output as the DRAM contract and stays sharded.

    Batch 13 runs as the physical batch 16 with padded users at position -1 and zero page-table rows, the
    layer-stack encoding of padded users; the same cache is reused after the DRAM-contract call wrote it.
    """
    fixture = base._layer(0)
    decoder = _optimized_decoder(mesh_device, fixture)
    records = []
    for batch in (1, 8, 13, 32):
        lens = [base.DECODE_LENGTH_CHOICES[u % len(base.DECODE_LENGTH_CHOICES)] for u in range(batch)]
        blocks = base._blocks_for(max(lens) + 3)
        table, page_table = base._page_table(mesh_device, batch, blocks, seed=21 + batch)
        kv_cache = base._kv_cache(mesh_device, batch * blocks)
        tokens = []
        for user in range(batch):
            hidden = fixture.inputs(lens[user] + 2, seed=860 + user)
            decoder.prefill_forward(
                base._tt(mesh_device, hidden[:, : lens[user]].unsqueeze(1)),
                page_table=page_table,
                kv_cache=kv_cache,
                user_id=user,
            )
            tokens.append(hidden[:, lens[user] : lens[user] + 1])
        x = torch.cat(tokens, dim=1).reshape(1, 1, batch, HIDDEN)
        current_pos = base._tt(
            mesh_device, torch.tensor(lens, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        dram_out = decoder.decode_forward(
            base._tt(mesh_device, x), current_pos=current_pos, page_table=page_table, kv_cache=kv_cache
        )
        assert not dram_out.is_sharded()
        dram_t = base._to_torch(dram_out)
        physical, _, _ = decode_physical_batch(batch, 11, 10)
        x_rows = torch.zeros(1, 1, 32, HIDDEN, dtype=x.dtype)
        x_rows[:, :, :batch] = x
        sharded_in = ttnn.to_memory_config(base._tt(mesh_device, x_rows), decoder.decode_residual_memory_config)
        padded_pos = torch.full((physical,), -1, dtype=torch.int32)
        padded_pos[:batch] = torch.tensor(lens, dtype=torch.int32)
        padded_table = torch.zeros(physical, blocks, dtype=torch.int32)
        padded_table[:batch] = table
        sharded_out = decoder.decode_forward(
            sharded_in,
            current_pos=base._tt(mesh_device, padded_pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
            page_table=base._tt(mesh_device, padded_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
            kv_cache=kv_cache,
        )
        assert sharded_out.is_sharded()
        assert same_layout(sharded_out.memory_config(), decoder.decode_residual_memory_config)
        sharded_t = base._to_torch(sharded_out)[:, :, :batch]
        records.append(
            {
                "batch": batch,
                "physical_batch": physical,
                "dram_output_shape": list(dram_out.shape),
                "sharded_output_shape": list(sharded_out.shape),
                "pcc_sharded_vs_dram": base._pcc(dram_t, sharded_t),
                "bit_equal": bool(torch.equal(dram_t, sharded_t)),
            }
        )
    _write_optimized_json(
        f"sharded_residual_{base.LAYER_KIND}.json",
        {
            "layer_idx": 0,
            "residual_memory_config": str(decoder.decode_residual_memory_config),
            "cases": records,
            "provenance": _optimized_provenance(
                mesh_device,
                _optimized_evidence_command(
                    "python -m pytest -q models/autoports/qwen_qwen3_8b/tests/test_optimized_decoder.py -k test_sharded_residual_contract"
                ),
                fixture.weight_origin,
            ),
        },
    )
    for record in records:
        assert record["pcc_sharded_vs_dram"] >= base.REPLAY_PCC_BAR, record


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_block64_long_lead_rows(mesh_device, device_params):
    """A 64-row block with a 1-token prefix: the continuation writes 63 lead rows in two pseudo-user groups."""
    fixture = base._layer(0)
    decoder = _optimized_decoder(mesh_device, fixture)
    block_size = 64
    prefix, length = 1, 200
    hidden = fixture.inputs(prefix + length + 2, seed=870)
    reference = fixture.reference()
    reference.forward(hidden[:, :prefix], 0)
    hf_cont = reference.forward(hidden[:, prefix : prefix + length], prefix)
    hf_decode = reference.forward(hidden[:, prefix + length : prefix + length + 1], prefix + length)
    blocks = (prefix + length + 2 + block_size - 1) // block_size
    table, page_table = base._page_table(mesh_device, 2, blocks, seed=31)
    kv_cache = base._kv_cache(mesh_device, 2 * blocks, block_size=block_size)
    decoder.prefill_forward(
        base._tt(mesh_device, hidden[:, :prefix].unsqueeze(1)), page_table=page_table, kv_cache=kv_cache, user_id=1
    )
    out = decoder.prefill_forward(
        base._tt(mesh_device, hidden[:, prefix : prefix + length].unsqueeze(1)),
        page_table=page_table,
        kv_cache=kv_cache,
        user_id=1,
        start_pos=prefix,
    )
    out_t = base._to_torch(out)[:, 0]
    user_table = base._tt(mesh_device, table[1:2], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    position = prefix + length
    od = decoder.decode_forward(
        base._tt(mesh_device, hidden[:, position : position + 1].reshape(1, 1, 1, HIDDEN)),
        current_pos=base._tt(
            mesh_device, torch.tensor([position], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        ),
        page_table=user_table,
        kv_cache=kv_cache,
    )
    od_t = base._to_torch(od).reshape(1, 1, HIDDEN)
    record = {
        "layer_idx": 0,
        "block_size": block_size,
        "prefix": prefix,
        "continuation_tokens": length,
        "lead_rows": block_size - prefix,
        "lead_groups": 2,
        "continuation_pcc_vs_hf": base._pcc(hf_cont, out_t),
        "continuation_delta_pcc_vs_hf": base._delta_pcc(hf_cont, out_t, hidden[:, prefix : prefix + length]),
        "decode_after_pcc_vs_hf": base._pcc(hf_decode, od_t),
        "decode_after_delta_pcc_vs_hf": base._delta_pcc(hf_decode, od_t, hidden[:, position : position + 1]),
        "provenance": _optimized_provenance(
            mesh_device,
            _optimized_evidence_command(
                "python -m pytest -q models/autoports/qwen_qwen3_8b/tests/test_optimized_decoder.py -k test_block64_long_lead_rows"
            ),
            fixture.weight_origin,
        ),
    }
    _write_optimized_json(f"block64_long_lead_{base.LAYER_KIND}.json", record)
    assert record["continuation_pcc_vs_hf"] >= PCC_BAR, record
    assert record["continuation_delta_pcc_vs_hf"] >= DELTA_PCC_BAR, record
    assert record["decode_after_pcc_vs_hf"] >= PCC_BAR, record
    assert record["decode_after_delta_pcc_vs_hf"] >= DELTA_PCC_BAR, record


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_repeated_runs_stress(mesh_device, device_params):
    """Repeated prefill is bit-stable and a long traced decode run tracks HF step by step."""
    fixture = base._layer(0)
    decoder = _optimized_decoder(mesh_device, fixture)
    prefill_repeats = int(os.getenv("QWEN3_STRESS_PREFILL_REPEATS", "10"))
    decode_steps = int(os.getenv("QWEN3_STRESS_DECODE_STEPS", "48"))
    replay_repeats = int(os.getenv("QWEN3_STRESS_REPLAY_REPEATS", "50"))
    batch = 8
    seq_len = 1024
    users = batch
    blocks = base._blocks_for(seq_len + decode_steps + 1)
    table, page_table = base._page_table(mesh_device, users, blocks, seed=17)
    kv_cache = base._kv_cache(mesh_device, users * blocks)
    hidden = fixture.inputs(seq_len + decode_steps, seed=830)
    x = base._tt(mesh_device, hidden[:, :seq_len].unsqueeze(1))
    first = None
    prefill_bit_equal = True
    started = time.perf_counter()
    for _ in range(prefill_repeats):
        out = base._to_torch(decoder.prefill_forward(x, page_table=page_table, kv_cache=kv_cache, user_id=0))
        if first is None:
            first = out
        elif not torch.equal(first, out):
            prefill_bit_equal = False
    prefill_wall = time.perf_counter() - started
    reference = fixture.reference()
    hf_prefill = reference.forward(hidden[:, :seq_len], 0)
    prefill_pcc = base._pcc(hf_prefill, first[:, 0])
    for user in range(1, batch):
        user_hidden = fixture.inputs(seq_len, seed=840 + user)
        decoder.prefill_forward(
            base._tt(mesh_device, user_hidden.unsqueeze(1)), page_table=page_table, kv_cache=kv_cache, user_id=user
        )

    def host_inputs(step: int):
        token = hidden[:, seq_len + step : seq_len + step + 1].reshape(1, 1, 1, HIDDEN).repeat(1, 1, batch, 1)
        positions = torch.full((batch,), seq_len + step, dtype=torch.int32)
        return base._host(token), base._host(positions, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    tokens0, positions0 = host_inputs(0)
    xd = ttnn.to_device(tokens0, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    current_pos = ttnn.to_device(positions0, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    decoder.decode_forward(xd, current_pos=current_pos, page_table=page_table, kv_cache=kv_cache)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_out = decoder.decode_forward(xd, current_pos=current_pos, page_table=page_table, kv_cache=kv_cache)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    replay0 = base._to_torch(traced_out)
    replay_bit_equal = True
    for _ in range(replay_repeats):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        if not torch.equal(replay0, base._to_torch(traced_out)):
            replay_bit_equal = False
    step_pcc = []
    worst_user0_pcc = 1.0
    for step in range(decode_steps):
        if step:
            tokens, positions = host_inputs(step)
            ttnn.copy_host_to_device_tensor(tokens, xd)
            ttnn.copy_host_to_device_tensor(positions, current_pos)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        replay = base._to_torch(traced_out)
        hf = reference.forward(hidden[:, seq_len + step : seq_len + step + 1], seq_len + step)
        pcc_user0 = base._pcc(hf, replay[0, 0, 0:1])
        worst_user0_pcc = min(worst_user0_pcc, pcc_user0)
        step_pcc.append(pcc_user0)
        assert torch.isfinite(replay.float()).all(), step
    ttnn.release_trace(mesh_device, trace_id)
    _write_optimized_json(
        f"stress_{base.LAYER_KIND}.json",
        {
            "layer_idx": 0,
            "sequence_length": seq_len,
            "decode_batch": batch,
            "prefill_repeats": prefill_repeats,
            "prefill_bit_equal_across_repeats": prefill_bit_equal,
            "prefill_pcc_vs_hf": prefill_pcc,
            "prefill_wall_s_total": prefill_wall,
            "replay_repeats_same_inputs": replay_repeats,
            "replay_bit_equal": replay_bit_equal,
            "decode_steps": decode_steps,
            "decode_user0_pcc_vs_hf_per_step": step_pcc,
            "decode_user0_worst_pcc": worst_user0_pcc,
            "provenance": _optimized_provenance(
                mesh_device,
                _optimized_evidence_command(
                    "python -m pytest -q models/autoports/qwen_qwen3_8b/tests/test_optimized_decoder.py -k test_repeated_runs_stress"
                ),
                fixture.weight_origin,
            ),
        },
    )
    assert prefill_bit_equal
    assert prefill_pcc >= PCC_BAR
    assert replay_bit_equal
    assert worst_user0_pcc >= PCC_BAR, step_pcc
