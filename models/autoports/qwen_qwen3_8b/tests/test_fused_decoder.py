# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fused-decoder tests for Qwen/Qwen3-8B on one Blackhole chip (1x1 mesh).

The functional-decoder gates (real-weight PCC, boundary lengths, continuation,
multi-row prefill, block size 64, traced decode, advertised context, profiler
harness) are reused from ``test_functional_decoder`` and bound to
``FusedDecoder`` for the whole module; their evidence JSON lands under
``doc/fused_decoder``. Fused-specific tests cover the on-device equivalence of
the fused and functional graphs, the prefill kernel selection, the decode V
core rectangle, repeated-run stability and the hot-path fallback audit.

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
from models.autoports.qwen_qwen3_8b.tt import fused_decoder as fused_module
from models.autoports.qwen_qwen3_8b.tt.functional_decoder import FunctionalDecoder, decode_physical_batch
from models.autoports.qwen_qwen3_8b.tt.fused_decoder import (
    PREFILL_MINIMAL_MATMUL_MIN_ROWS,
    FusedDecoder,
    _decode_v_update_memory_config,
)

ARTIFACT_DIR = base.MODEL_DIR / "doc" / "fused_decoder"
EQUIVALENCE_PCC_BAR = 0.9999
HIDDEN = base.HIDDEN
PCC_BAR = base.PCC_BAR
DEVICE_PARAMS = base.DEVICE_PARAMS


def _fused_decoder(mesh_device, fixture: base.LayerFixture) -> FusedDecoder:
    return FusedDecoder.from_state_dict(
        fixture.state, hf_config=fixture.config, layer_idx=fixture.layer_idx, mesh_device=mesh_device
    )


def _functional_decoder(mesh_device, fixture: base.LayerFixture) -> FunctionalDecoder:
    return FunctionalDecoder.from_state_dict(
        fixture.state, hf_config=fixture.config, layer_idx=fixture.layer_idx, mesh_device=mesh_device
    )


_original_provenance = base._provenance
_original_evidence_command = base._evidence_command


def _fused_provenance(mesh_device, exact_command: str, weight_origin: str) -> dict:
    record = _original_provenance(mesh_device, exact_command, weight_origin)
    functional_path = Path(base.decoder_module.__file__).resolve()
    fused_path = Path(fused_module.__file__).resolve()
    record["decoder"] = "fused"
    record["fused_decoder_path"] = str(fused_path)
    record["fused_decoder_sha256"] = hashlib.sha256(fused_path.read_bytes()).hexdigest()
    record["functional_decoder_path"] = str(functional_path)
    record["functional_decoder_sha256"] = hashlib.sha256(functional_path.read_bytes()).hexdigest()
    record["test_sha256"] = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    record["functional_test_sha256"] = hashlib.sha256(Path(base.__file__).resolve().read_bytes()).hexdigest()
    return record


def _fused_evidence_command(default: str) -> str:
    return _original_evidence_command(default.replace("test_functional_decoder", "test_fused_decoder"))


def _write_fused_json(name: str, payload: dict) -> Path:
    """Write evidence JSON under ``doc/fused_decoder``; ``QWEN3_EVIDENCE_SUFFIX`` works as in the base module."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = os.getenv("QWEN3_EVIDENCE_SUFFIX", "")
    if suffix:
        name = name.replace(".json", f".{suffix}.json")
    path = ARTIFACT_DIR / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


@pytest.fixture(scope="module", autouse=True)
def _bind_base_tests_to_fused_decoder():
    saved = {
        "_decoder": base._decoder,
        "_write_json": base._write_json,
        "_provenance": base._provenance,
        "_evidence_command": base._evidence_command,
    }
    base._decoder = _fused_decoder
    base._write_json = _write_fused_json
    base._provenance = _fused_provenance
    base._evidence_command = _fused_evidence_command
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
test_fused_decoder_perf_profile = base.test_functional_decoder_perf_profile


def test_fused_hot_path_fallback_audit():
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
        FusedDecoder.prefill_forward,
        FusedDecoder._prefill_row,
        FusedDecoder._pad_rows,
        FusedDecoder._pad_sequence,
        FusedDecoder._prefill_chunk,
        FusedDecoder._prefill_matmul,
        FusedDecoder._prefill_matmul_residual,
        FusedDecoder._mlp_prefill,
        FusedDecoder._attention_prefill,
        FusedDecoder._write_prefill_cache,
        FusedDecoder._linear,
        FusedDecoder._rms_norm,
        FusedDecoder.decode_forward,
        FusedDecoder._decode_matmul_residual,
        FusedDecoder._mlp_decode,
        FusedDecoder._decode_head_norm,
        FusedDecoder._decode_rope,
        FusedDecoder._attention_decode,
    ]
    for method in hot:
        source = inspect.getsource(method)
        for token in forbidden:
            assert token not in source, f"{method.__name__} uses {token}"


def test_fused_overrides_every_measured_path_host():
    """The fused class must own every method on the measured prefill and decode paths."""
    for name in ("_prefill_chunk", "_attention_prefill", "decode_forward", "_attention_decode", "_decode_rope"):
        assert name in FusedDecoder.__dict__, name
    assert "_mlp" not in FusedDecoder.__dict__
    assert PREFILL_MINIMAL_MATMUL_MIN_ROWS == 256


class _Shaped:
    def __init__(self, rows: int):
        self.shape = (1, 1, rows, HIDDEN)


@pytest.mark.parametrize("rows,expected", [(32, False), (224, False), (256, True), (1024, True), (4096, True)])
def test_prefill_kernel_selection_host(rows, expected):
    decoder = FusedDecoder.__new__(FusedDecoder)
    decoder.prefill_minimal_matmul_min_rows = PREFILL_MINIMAL_MATMUL_MIN_ROWS
    assert decoder._use_minimal_matmul(_Shaped(rows)) is expected


def _cores(core_range_set) -> set[tuple[int, int]]:
    return {
        (x, y)
        for r in core_range_set.ranges()
        for x in range(r.start.x, r.end.x + 1)
        for y in range(r.start.y, r.end.y + 1)
    }


@pytest.mark.parametrize("grid", [(11, 10), (13, 10), (8, 8)], ids=["grid11x10", "grid13x10", "grid8x8"])
def test_decode_v_update_rectangle_host(grid, expect_error):
    stub = base._GridStub(*grid)
    for batch in range(1, base.decoder_module.MAX_DECODE_BATCH + 1):
        physical, x, y = decode_physical_batch(batch, *grid)
        q_cfg = base.decoder_module._decode_height_sharded_memory_config(stub, physical, base.HEAD_DIM)
        v_cfg = _decode_v_update_memory_config(stub, physical, base.HEAD_DIM)
        q_cores = _cores(q_cfg.shard_spec.grid)
        v_cores = _cores(v_cfg.shard_spec.grid)
        assert len(v_cores) == len(q_cores) == physical
        assert not (q_cores & v_cores), (batch, grid)
        assert {c[1] for c in v_cores} == set(range(y, 2 * y)) and max(c[0] for c in v_cores) == x - 1
    with expect_error(ValueError, "core area"):
        _decode_v_update_memory_config(base._GridStub(8, 1), 1, base.HEAD_DIM)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_fused_matches_functional_on_device(mesh_device, device_params):
    """Fused and functional graphs agree on real weights for prefill, continuation and decode."""
    fixture = base._layer(0)
    functional = _functional_decoder(mesh_device, fixture)
    fused = _fused_decoder(mesh_device, fixture)
    cases = []
    users = 4
    for seq_len, start in ((1024, 0), (300, 0), (37, 0), (4500, 37)):
        hidden = fixture.inputs(start + seq_len + 4, seed=800 + seq_len)
        blocks = base._blocks_for(start + seq_len + 4)
        table, page_table = base._page_table(mesh_device, users, blocks, seed=11)
        outs = {}
        decodes = {}
        for name, decoder in (("functional", functional), ("fused", fused)):
            kv_cache = base._kv_cache(mesh_device, users * blocks)
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
        record = {
            "seq_len": seq_len,
            "start_pos": start,
            "prefill_pcc": base._pcc(outs["functional"], outs["fused"]),
            "prefill_delta_pcc": base._delta_pcc(outs["functional"], outs["fused"], resumed_in),
            "prefill_max_abs_diff": float((outs["functional"].float() - outs["fused"].float()).abs().max()),
            "key_cache_pcc": base._pcc(outs["functional_kcache"], outs["fused_kcache"]),
            "value_cache_pcc": base._pcc(outs["functional_vcache"], outs["fused_vcache"]),
            "decode_pcc": [base._pcc(a, b) for a, b in zip(decodes["functional"], decodes["fused"])],
            "decode_delta_pcc": [
                base._delta_pcc(a, b, hidden[:, start + seq_len + i : start + seq_len + i + 1])
                for i, (a, b) in enumerate(zip(decodes["functional"], decodes["fused"]))
            ],
        }
        cases.append(record)
    batch = 32
    lens = [base.DECODE_LENGTH_CHOICES[u % len(base.DECODE_LENGTH_CHOICES)] for u in range(batch)]
    blocks = base._blocks_for(max(lens) + 2)
    table, page_table = base._page_table(mesh_device, batch, blocks, seed=13)
    batch_outs = {}
    for name, decoder in (("functional", functional), ("fused", fused)):
        kv_cache = base._kv_cache(mesh_device, batch * blocks)
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
        "decode_pcc_per_user": [base._pcc(batch_outs["functional"][u], batch_outs["fused"][u]) for u in range(batch)],
        "decode_max_abs_diff": float((batch_outs["functional"].float() - batch_outs["fused"].float()).abs().max()),
    }
    _write_fused_json(
        f"fused_vs_functional_{base.LAYER_KIND}.json",
        {
            "layer_idx": 0,
            "equivalence_pcc_bar": EQUIVALENCE_PCC_BAR,
            "prefill_cases": cases,
            "decode_batch32": batch_record,
            "provenance": _fused_provenance(
                mesh_device,
                _fused_evidence_command(
                    "python -m pytest -q models/autoports/qwen_qwen3_8b/tests/test_fused_decoder.py -k test_fused_matches_functional_on_device"
                ),
                fixture.weight_origin,
            ),
        },
    )
    for record in cases:
        assert record["prefill_pcc"] >= EQUIVALENCE_PCC_BAR, record
        assert record["prefill_delta_pcc"] >= base.DELTA_PCC_BAR, record
        assert record["key_cache_pcc"] >= EQUIVALENCE_PCC_BAR, record
        assert record["value_cache_pcc"] >= EQUIVALENCE_PCC_BAR, record
        assert min(record["decode_pcc"]) >= EQUIVALENCE_PCC_BAR, record
        assert min(record["decode_delta_pcc"]) >= base.DELTA_PCC_BAR, record
    assert min(batch_record["decode_pcc_per_user"]) >= EQUIVALENCE_PCC_BAR, batch_record


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_repeated_runs_stress(mesh_device, device_params):
    """Repeated prefill is bit-stable and a long traced decode run tracks HF step by step."""
    fixture = base._layer(0)
    decoder = _fused_decoder(mesh_device, fixture)
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
    _write_fused_json(
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
            "provenance": _fused_provenance(
                mesh_device,
                _fused_evidence_command(
                    "python -m pytest -q models/autoports/qwen_qwen3_8b/tests/test_fused_decoder.py -k test_repeated_runs_stress"
                ),
                fixture.weight_origin,
            ),
        },
    )
    assert prefill_bit_equal
    assert prefill_pcc >= PCC_BAR
    assert replay_bit_equal
    assert worst_user0_pcc >= PCC_BAR, step_pcc
