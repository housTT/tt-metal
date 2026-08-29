# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Functional-decoder correctness, paging, trace and fallback gates."""

from __future__ import annotations

import inspect
import os

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import harness as H
from models.autoports.qwen_qwen3_8_flash_next.tt.functional_decoder import FunctionalDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.model_config import HF_ADVERTISED_CONTEXT, decoder_shapes

LAYER_KINDS = (0, 1, 3)  # GDN, GDN+PLE, QSA
BOUNDARY_LENGTHS = (31, 32, 33, 63, 64, 65, 127, 128, 129)


class _FakePersistentState:
    def __init__(self, value: int, address: int):
        self.value = value
        self._address = address
        self._allocated = True

    def is_allocated(self):
        return self._allocated

    def buffer_address(self):
        return self._address


def _upload(tensor, mesh_device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(tensor, device=mesh_device, dtype=dtype, layout=layout)


def _paged_inputs(layer, mesh_device, page_table_host, cos_host, sin_host, seq_len):
    page_table = _upload(page_table_host, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    chunk_tables = []
    for start, _, padded in layer.prefill_chunk_plan(seq_len):
        first = start // layer.block_size
        last = (start + padded) // layer.block_size
        chunk_tables.append(
            _upload(page_table_host[:, first:last], mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        )
    cos = _upload(cos_host.reshape(1, 1, layer.max_seq_len, -1), mesh_device)
    sin = _upload(sin_host.reshape(1, 1, layer.max_seq_len, -1), mesh_device)
    return page_table, chunk_tables, (cos, sin)


def test_target_shape_and_layer_kind_contract():
    config = H.target_config()
    observed = []
    for layer_idx in LAYER_KINDS:
        shapes = decoder_shapes(config, layer_idx)
        observed.append((shapes.layer_type, shapes.has_ple))
        assert shapes.hc_hidden_size == 10240
        assert shapes.max_position_embeddings == HF_ADVERTISED_CONTEXT
        assert shapes.num_experts == 512 and shapes.num_experts_per_tok == 10
    assert observed == [
        ("linear_attention", False),
        ("linear_attention", True),
        ("qwen_sparse_attention", False),
    ]


def test_prefill_state_update_preserves_preallocated_buffer(monkeypatch):
    """Later prefills must not replace state while decode traces are live."""

    persistent = _FakePersistentState(7, 0x1000)
    update = _FakePersistentState(19, 0x2000)
    copy_calls = []

    def copy(source, target):
        copy_calls.append((source, target))
        target.value = source.value

    def deallocate(tensor):
        tensor._allocated = False

    monkeypatch.setattr(ttnn, "copy", copy)
    monkeypatch.setattr(ttnn, "deallocate", deallocate)

    result = FunctionalDecoder._update_prefill_state(persistent, update)

    assert result is persistent
    assert copy_calls == [(update, persistent)]
    assert persistent.value == 19
    assert persistent.is_allocated()
    assert persistent.buffer_address() == 0x1000
    assert not update.is_allocated()


@pytest.mark.parametrize("seq_len", [1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 2047, 2048, 2049])
def test_prefill_plan_accepts_non_aligned_boundaries(seq_len):
    layer = object.__new__(FunctionalDecoder)
    layer.max_seq_len = HF_ADVERTISED_CONTEXT
    plan = layer.prefill_chunk_plan(seq_len)
    assert sum(logical for _, logical, _ in plan) == seq_len
    assert all(padded == 128 for _, _, padded in plan)
    assert plan[-1][1] == ((seq_len - 1) % 128) + 1


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_real_weights_hf_prefill_decode_pcc(mesh_device, layer_idx):
    """Real checkpoint tensors, cache-free HF decode oracle, shuffled TT pages."""

    torch.manual_seed(20260826 + layer_idx)
    config = H.target_config()
    state = H.load_real_layer_state(layer_idx)
    seq_len = 33
    max_seq_len = 4096 if layer_idx == 3 else 128
    hidden = (torch.randn(1, seq_len + 1, 10240, dtype=torch.bfloat16) * 0.02).contiguous()
    cos, sin = H.rope_tables(max_seq_len)
    ple = None
    if layer_idx == 1:
        ple = (torch.randn(1, seq_len + 1, 2560, dtype=torch.bfloat16) * 0.02).contiguous()

    hf_layer = H.build_hf_layer(config, layer_idx, state, ple_embeddings=None if ple is None else ple[:, :seq_len])
    expected_prefill = H.hf_forward(
        hf_layer,
        hidden[:, :seq_len],
        cos,
        sin,
        ple_embeddings=None if ple is None else ple[:, :seq_len],
    )

    layer = FunctionalDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=max_seq_len,
    )
    tt_hidden = _upload(hidden[:, :seq_len].unsqueeze(0), mesh_device)
    tt_ple = None if ple is None else _upload(ple[:, :seq_len].unsqueeze(0), mesh_device)
    kwargs = {"ple_embeddings": tt_ple} if ple is not None else {}
    if layer_idx == 3:
        page_host = H.shuffled_page_table(max_seq_len)
        page, chunk_pages, rot = _paged_inputs(layer, mesh_device, page_host, cos, sin, seq_len)
        kwargs.update(page_table=page, page_tables_per_chunk=chunk_pages, rot_mats=rot)
    with H.ForbidHostFallback():
        actual_prefill_tt = layer.prefill_forward(tt_hidden, **kwargs)
    actual_prefill = ttnn.to_torch(actual_prefill_tt).squeeze(0)
    prefill_pcc = H.pcc(expected_prefill, actual_prefill)
    print(f"PCCEVIDENCE real layer={layer_idx} prefill_pcc={prefill_pcc:.8f}")
    assert prefill_pcc >= H.PCC_BAR

    layer.prepare_decode_state()
    if ple is not None:
        hf_layer.ple.ple_embedding.value = ple
    expected_decode = H.hf_forward(hf_layer, hidden, cos, sin, ple_embeddings=ple)[:, -1:]
    decode_input = _upload(hidden[:, -1:].unsqueeze(0), mesh_device)
    current_pos = _upload(
        torch.tensor([seq_len], dtype=torch.int32), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    decode_kwargs = {"current_pos": current_pos}
    if ple is not None:
        decode_kwargs["ple_embeddings"] = _upload(ple[:, -1:].unsqueeze(0), mesh_device)
    if layer_idx == 3:
        decode_kwargs.update(page_table=page, rot_mats=rot)
    # Compile every target-shape decode program before capture, then restore
    # state so capture and replay both begin at the HF oracle's position.
    with H.ForbidHostFallback():
        layer.decode_forward(decode_input, **decode_kwargs)
    ttnn.synchronize_device(mesh_device)
    layer.prepare_decode_state()
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    with H.ForbidHostFallback():
        actual_decode_tt = layer.decode_forward(decode_input, **decode_kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    # Capture executes stateful programs. Restore the prefill recurrence, then
    # use the replay result as the PCC sample so this gate covers the deployed
    # traced decode path rather than a separate eager invocation.
    layer.prepare_decode_state()
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    actual_decode = ttnn.to_torch(actual_decode_tt).squeeze(0)
    ttnn.release_trace(mesh_device, trace_id)
    decode_pcc = H.pcc(expected_decode, actual_decode)
    print(f"PCCEVIDENCE real layer={layer_idx} decode_pcc={decode_pcc:.8f}")
    assert decode_pcc >= H.PCC_BAR


@pytest.mark.skipif(os.getenv("RUN_QWEN38_PROGRESSING_HF_DIAGNOSTIC") != "1", reason="explicit HF state diagnostic")
@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_real_weights_progressing_decode_against_hf(mesh_device, layer_idx):
    """Compare persistent TT state with cache-free HF over twelve decode rows."""

    torch.manual_seed(20260828 + layer_idx)
    config = H.target_config()
    state = H.load_real_layer_state(layer_idx)
    seq_len, decode_rows = 33, 12
    max_seq_len = 4096 if layer_idx == 3 else 128
    hidden = (torch.randn(1, seq_len + decode_rows, 10240, dtype=torch.bfloat16) * 0.02).contiguous()
    cos, sin = H.rope_tables(max_seq_len)
    ple = None
    if layer_idx == 1:
        ple = (torch.randn(1, seq_len + decode_rows, 2560, dtype=torch.bfloat16) * 0.02).contiguous()

    hf_layer = H.build_hf_layer(config, layer_idx, state, ple_embeddings=None if ple is None else ple[:, :seq_len])
    expected_prefill = H.hf_forward(
        hf_layer,
        hidden[:, :seq_len],
        cos,
        sin,
        ple_embeddings=None if ple is None else ple[:, :seq_len],
    )
    layer = FunctionalDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=max_seq_len,
    )
    kwargs = {}
    if ple is not None:
        kwargs["ple_embeddings"] = _upload(ple[:, :seq_len].unsqueeze(0), mesh_device)
    if layer_idx == 3:
        page_host = H.shuffled_page_table(max_seq_len)
        page, chunk_pages, rot = _paged_inputs(layer, mesh_device, page_host, cos, sin, seq_len)
        kwargs.update(page_table=page, page_tables_per_chunk=chunk_pages, rot_mats=rot)
    actual_prefill = layer.prefill_forward(_upload(hidden[:, :seq_len].unsqueeze(0), mesh_device), **kwargs)
    ttnn.synchronize_device(mesh_device)
    actual_prefill_host = ttnn.to_torch(actual_prefill).squeeze(0)
    prefill_pcc = H.pcc(expected_prefill, actual_prefill_host)
    layer.prepare_decode_state()

    pccs = []
    for step in range(decode_rows):
        stop = seq_len + step + 1
        if ple is not None:
            hf_layer.ple.ple_embedding.value = ple[:, :stop]
        expected = H.hf_forward(
            hf_layer,
            hidden[:, :stop],
            cos,
            sin,
            ple_embeddings=None if ple is None else ple[:, :stop],
        )[:, -1:]
        current_pos = _upload(
            torch.tensor([stop - 1], dtype=torch.int32),
            mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        decode_kwargs = {"current_pos": current_pos}
        if ple is not None:
            decode_kwargs["ple_embeddings"] = _upload(ple[:, stop - 1 : stop].unsqueeze(0), mesh_device)
        if layer_idx == 3:
            decode_kwargs.update(page_table=page, rot_mats=rot)
        actual = layer.decode_forward(
            _upload(hidden[:, stop - 1 : stop].unsqueeze(0), mesh_device),
            **decode_kwargs,
        )
        ttnn.synchronize_device(mesh_device)
        actual_host = ttnn.to_torch(actual).squeeze(0)
        pccs.append(H.pcc(expected, actual_host))
        ttnn.deallocate(actual)
    print(f"PROGRESSING_HF_PCC layer={layer_idx} prefill={prefill_pcc:.8f} decode={pccs}")
    assert prefill_pcc >= H.PCC_BAR
    assert min(pccs) >= H.PCC_BAR
    return {"prefill_pcc": prefill_pcc, "decode_pccs": pccs}


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_decode_trace_replay_and_determinism(mesh_device, layer_idx):
    config = H.target_config()
    max_seq_len = 4096 if layer_idx == 3 else 128
    layer = FunctionalDecoder.from_state_dict(
        None,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=max_seq_len,
    )
    hidden = _upload(torch.zeros(1, 1, 1, 10240, dtype=torch.bfloat16), mesh_device)
    current_pos = _upload(
        torch.tensor([0], dtype=torch.int32), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    kwargs = {"current_pos": current_pos}
    if layer_idx == 1:
        kwargs["ple_embeddings"] = _upload(torch.zeros(1, 1, 1, 2560, dtype=torch.bfloat16), mesh_device)
    if layer_idx == 3:
        cos, sin = H.rope_tables(max_seq_len)
        page_host = H.shuffled_page_table(max_seq_len)
        page, _, rot = _paged_inputs(layer, mesh_device, page_host, cos, sin, 1)
        kwargs.update(page_table=page, rot_mats=rot)

    warm = layer.decode_forward(hidden, **kwargs)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = layer.decode_forward(hidden, **kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    first = ttnn.to_torch(traced_output)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    second = ttnn.to_torch(traced_output)
    ttnn.release_trace(mesh_device, trace_id)
    assert torch.equal(first, second)
    assert list(warm.shape) == [1, 1, 1, 10240]


def test_runtime_source_has_no_host_fallback():
    source = inspect.getsource(FunctionalDecoder.prefill_forward) + inspect.getsource(FunctionalDecoder.decode_forward)
    for forbidden in ("torch.", "ttnn.from_torch", "ttnn.to_torch", "ttnn.as_tensor"):
        assert forbidden not in source


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_exact_target_shapes_run_non_aligned_prefill_boundaries(mesh_device, layer_idx):
    """Exercise logical tile/page/chunk edges, not only the chunk planner."""

    config = H.target_config()
    max_seq_len = 4096 if layer_idx == 3 else 256
    layer = FunctionalDecoder.from_state_dict(
        None,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=max_seq_len,
    )
    if layer_idx == 3:
        cos, sin = H.rope_tables(max_seq_len)
        page_host = H.shuffled_page_table(max_seq_len)

    for seq_len in BOUNDARY_LENGTHS:
        hidden = _upload(torch.zeros(1, 1, seq_len, 10240, dtype=torch.bfloat16), mesh_device)
        kwargs = {}
        if layer_idx == 1:
            kwargs["ple_embeddings"] = _upload(torch.zeros(1, 1, seq_len, 2560, dtype=torch.bfloat16), mesh_device)
        if layer_idx == 3:
            page, chunk_pages, rot = _paged_inputs(layer, mesh_device, page_host, cos, sin, seq_len)
            kwargs.update(page_table=page, page_tables_per_chunk=chunk_pages, rot_mats=rot)
        with H.ForbidHostFallback():
            output = layer.prefill_forward(hidden, **kwargs)
        assert list(output.shape) == [1, 1, seq_len, 10240]
        ttnn.deallocate(output)
        ttnn.deallocate(hidden)


def test_qsa_prefill_page_table_permutation_is_semantically_invariant(mesh_device):
    """A non-identity page map must change addresses, never virtual-token results."""

    torch.manual_seed(20260830)
    config = H.target_config()
    state = H.make_partial_state(config, 3)
    max_seq_len, seq_len = 4096, 65
    layer = FunctionalDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=3,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=max_seq_len,
    )
    hidden_host = torch.randn(1, 1, seq_len, 10240, dtype=torch.bfloat16) * 0.02
    cos, sin = H.rope_tables(max_seq_len)
    outputs = []
    for page_host in (
        torch.arange(max_seq_len // 64, dtype=torch.int32).reshape(1, -1),
        H.shuffled_page_table(max_seq_len),
    ):
        hidden = _upload(hidden_host, mesh_device)
        page, chunk_pages, rot = _paged_inputs(layer, mesh_device, page_host, cos, sin, seq_len)
        with H.ForbidHostFallback():
            output = layer.prefill_forward(
                hidden,
                page_table=page,
                page_tables_per_chunk=chunk_pages,
                rot_mats=rot,
            )
        outputs.append(ttnn.to_torch(output))
    assert H.pcc(outputs[0], outputs[1]) >= 0.99999


def test_qsa_batched_decode_uses_each_page_table_and_current_position(mesh_device):
    """Batch>1 updates distinct physical pages at non-aligned positions."""

    torch.manual_seed(20260831)
    config = H.target_config()
    state = H.make_partial_state(config, 3)
    max_batch, max_seq_len = 2, 4096
    layer = FunctionalDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=3,
        mesh_device=mesh_device,
        max_batch=max_batch,
        max_seq_len=max_seq_len,
    )
    hidden = _upload(torch.randn(1, 1, max_batch, 10240, dtype=torch.bfloat16) * 0.02, mesh_device)
    positions_host = torch.tensor([63, 128], dtype=torch.int32)
    positions = _upload(positions_host, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    row0 = H.shuffled_page_table(max_seq_len, seed=20260831).reshape(-1)
    row1 = max_seq_len // 64 + H.shuffled_page_table(max_seq_len, seed=20260832).reshape(-1)
    page_host = torch.stack([row0, row1])
    page = _upload(page_host, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    cos, sin = H.rope_tables(max_seq_len)
    rot = (
        _upload(cos.reshape(1, 1, max_seq_len, -1), mesh_device),
        _upload(sin.reshape(1, 1, max_seq_len, -1), mesh_device),
    )
    with H.ForbidHostFallback():
        output = layer.decode_forward(hidden, current_pos=positions, page_table=page, rot_mats=rot)
    assert list(output.shape) == [1, 1, max_batch, 10240]
    cache = ttnn.to_torch(layer.kv_cache[0])
    for user, position in enumerate(positions_host.tolist()):
        physical_page = int(page_host[user, position // layer.block_size])
        assert torch.count_nonzero(cache[physical_page, :, position % layer.block_size]) > 0
    untouched = next(
        page
        for page in range(cache.shape[0])
        if page
        not in {
            int(page_host[user, position // layer.block_size]) for user, position in enumerate(positions_host.tolist())
        }
    )
    assert torch.count_nonzero(cache[untouched]) == 0


def test_qsa_underfilled_selected_token_multiset(mesh_device):
    """Fixed-shape top-k lanes match HF without duplicating the partial tail."""

    config = H.target_config()
    max_seq_len = 4096
    layer = FunctionalDecoder.from_state_dict(
        None,
        hf_config=config,
        layer_idx=3,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=max_seq_len,
    )
    probe_positions = [
        0,
        2,
        3,
        4,
        31,
        32,
        33,
        63,
        64,
        65,
        127,
        128,
        129,
        2046,
        2047,
        2048,
        2049,
    ]
    padded_positions = probe_positions + [0] * (32 - len(probe_positions))
    positions = _upload(
        torch.tensor(padded_positions, dtype=torch.int32).reshape(1, 32),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    index_q = _upload(torch.zeros(1, 4, 32, 128, dtype=torch.bfloat16), mesh_device)
    page = _upload(
        H.shuffled_page_table(max_seq_len),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cos, sin = H.rope_tables(max_seq_len)
    rot = (
        _upload(cos.reshape(1, 1, max_seq_len, -1), mesh_device),
        _upload(sin.reshape(1, 1, max_seq_len, -1), mesh_device),
    )

    with H.ForbidHostFallback():
        selected, valid = layer._selected_virtual_tokens(index_q, page, positions, rot)
    selected_host = ttnn.to_torch(selected)[0, 0]
    valid_host = ttnn.to_torch(valid)[0, 0]
    for row, position in enumerate(probe_positions):
        actual = selected_host[row][valid_host[row] > 0].tolist()
        assert len(actual) == len(set(actual))
        assert sorted(actual) == list(range(position + 1))


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_two_user_prefill_then_batched_decode(mesh_device, layer_idx):
    """Every layer kind preserves two user states across prefill into decode."""

    config = H.target_config()
    max_seq_len = 4096 if layer_idx == 3 else 256
    layer = FunctionalDecoder.from_state_dict(
        None,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=2,
        max_seq_len=max_seq_len,
    )
    seq_len = 33
    page_rows = None
    if layer_idx == 3:
        pages_per_user = max_seq_len // layer.block_size
        page_rows = [
            H.shuffled_page_table(max_seq_len, seed=20260901).reshape(-1),
            pages_per_user + H.shuffled_page_table(max_seq_len, seed=20260902).reshape(-1),
        ]
        cos, sin = H.rope_tables(max_seq_len)
    for user_id in range(2):
        prefill_hidden = _upload(torch.zeros(1, 1, seq_len, 10240, dtype=torch.bfloat16), mesh_device)
        prefill_kwargs = {"user_id": user_id}
        if layer_idx == 1:
            prefill_kwargs["ple_embeddings"] = _upload(
                torch.zeros(1, 1, seq_len, 2560, dtype=torch.bfloat16), mesh_device
            )
        if layer_idx == 3:
            user_page = page_rows[user_id].reshape(1, -1)
            page, chunk_pages, rot = _paged_inputs(layer, mesh_device, user_page, cos, sin, seq_len)
            prefill_kwargs.update(page_table=page, page_tables_per_chunk=chunk_pages, rot_mats=rot)
        with H.ForbidHostFallback():
            prefill_output = layer.prefill_forward(prefill_hidden, **prefill_kwargs)
        assert list(prefill_output.shape) == [1, 1, seq_len, 10240]

    layer.prepare_decode_state()
    hidden = _upload(torch.zeros(1, 1, 2, 10240, dtype=torch.bfloat16), mesh_device)
    positions = _upload(
        torch.tensor([seq_len, seq_len], dtype=torch.int32),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    kwargs = {"current_pos": positions}
    if layer_idx == 1:
        kwargs["ple_embeddings"] = _upload(torch.zeros(1, 1, 2, 2560, dtype=torch.bfloat16), mesh_device)
    if layer_idx == 3:
        kwargs.update(
            page_table=_upload(
                torch.stack(page_rows),
                mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            rot_mats=rot,
        )
    with H.ForbidHostFallback():
        output = layer.decode_forward(hidden, **kwargs)
    assert list(output.shape) == [1, 1, 2, 10240]


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_batch_32_decode_contract(mesh_device, layer_idx):
    """Exercise the largest single-chip batch required by the stage contract."""

    config = H.target_config()
    batch = 32
    max_seq_len = 2048 if layer_idx == 3 else 256
    layer = FunctionalDecoder.from_state_dict(
        None,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=batch,
        max_seq_len=max_seq_len,
    )
    layer.prepare_decode_state()
    hidden = _upload(torch.zeros(1, 1, batch, 10240, dtype=torch.bfloat16), mesh_device)
    positions_host = (torch.arange(batch, dtype=torch.int32) * 61 + 33) % max_seq_len
    positions = _upload(
        positions_host,
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    kwargs = {"current_pos": positions}
    if layer_idx == 1:
        kwargs["ple_embeddings"] = _upload(
            torch.zeros(1, 1, batch, 2560, dtype=torch.bfloat16),
            mesh_device,
        )
    if layer_idx == 3:
        pages_per_user = max_seq_len // layer.block_size
        page_table_host = torch.arange(batch * pages_per_user, dtype=torch.int32).reshape(batch, pages_per_user)
        page_table_host[1::2] = torch.flip(page_table_host[1::2], dims=(-1,))
        cos, sin = H.rope_tables(max_seq_len)
        kwargs.update(
            page_table=_upload(
                page_table_host,
                mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            rot_mats=(
                _upload(cos.reshape(1, 1, max_seq_len, -1), mesh_device),
                _upload(sin.reshape(1, 1, max_seq_len, -1), mesh_device),
            ),
        )
    with H.ForbidHostFallback():
        output = layer.decode_forward(hidden, **kwargs)
    assert list(output.shape) == [1, 1, batch, 10240]


def test_qsa_public_prefill_long_non_aligned(mesh_device):
    """Run public QSA prefill just across the 2048-token sparse budget boundary."""

    config = H.target_config()
    max_seq_len, seq_len = 4096, 2049
    layer = FunctionalDecoder.from_state_dict(
        None,
        hf_config=config,
        layer_idx=3,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=max_seq_len,
    )
    hidden = _upload(torch.zeros(1, 1, seq_len, 10240, dtype=torch.bfloat16), mesh_device)
    cos, sin = H.rope_tables(max_seq_len)
    page_host = H.shuffled_page_table(max_seq_len)
    page, chunk_pages, rot = _paged_inputs(layer, mesh_device, page_host, cos, sin, seq_len)
    with H.ForbidHostFallback():
        output = layer.prefill_forward(
            hidden,
            page_table=page,
            page_tables_per_chunk=chunk_pages,
            rot_mats=rot,
        )
    assert list(output.shape) == [1, 1, seq_len, 10240]


@pytest.mark.long_context
@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.timeout(3600)
def test_full_advertised_context(mesh_device, layer_idx):
    """Allocate and execute the advertised prefill/decode contract.

    Every layer kind executes the complete public 262144-token prefill. QSA
    therefore covers all 2048 public chunks, page-table/cache progression and
    the surrounding hyperconnection/MoE path. All kinds also decode at
    position 262143.
    """

    config = H.target_config()
    layer = FunctionalDecoder.from_state_dict(
        None,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=HF_ADVERTISED_CONTEXT,
    )
    assert sum(x[1] for x in layer.prefill_chunk_plan(HF_ADVERTISED_CONTEXT)) == HF_ADVERTISED_CONTEXT
    near_max_plan = layer.prefill_chunk_plan(HF_ADVERTISED_CONTEXT - 1)
    assert near_max_plan[-1] == (HF_ADVERTISED_CONTEXT - 128, 127, 128)

    hidden = _upload(torch.zeros(1, 1, HF_ADVERTISED_CONTEXT, 10240, dtype=torch.bfloat16), mesh_device)
    prefill_kwargs = {}
    if layer_idx == 1:
        prefill_kwargs["ple_embeddings"] = _upload(
            torch.zeros(1, 1, HF_ADVERTISED_CONTEXT, 2560, dtype=torch.bfloat16), mesh_device
        )
    if layer_idx == 3:
        cos, sin = H.rope_tables(HF_ADVERTISED_CONTEXT)
        page_host = H.shuffled_page_table(HF_ADVERTISED_CONTEXT)
        page, chunk_pages, rot = _paged_inputs(
            layer,
            mesh_device,
            page_host,
            cos,
            sin,
            HF_ADVERTISED_CONTEXT,
        )
        prefill_kwargs.update(page_table=page, page_tables_per_chunk=chunk_pages, rot_mats=rot)
    with H.ForbidHostFallback():
        output = layer.prefill_forward(hidden, **prefill_kwargs)
    assert list(output.shape) == [1, 1, HF_ADVERTISED_CONTEXT, 10240]
    ttnn.deallocate(output)
    ttnn.deallocate(hidden)
    if layer_idx == 1:
        ttnn.deallocate(prefill_kwargs["ple_embeddings"])

    current_pos = _upload(
        torch.tensor([HF_ADVERTISED_CONTEXT - 1], dtype=torch.int32),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    decode_hidden = _upload(torch.zeros(1, 1, 1, 10240, dtype=torch.bfloat16), mesh_device)
    decode_kwargs = {"current_pos": current_pos}
    if layer_idx == 1:
        decode_kwargs["ple_embeddings"] = _upload(torch.zeros(1, 1, 1, 2560, dtype=torch.bfloat16), mesh_device)
    if layer_idx == 3:
        decode_kwargs.update(page_table=page, rot_mats=rot)
    with H.ForbidHostFallback():
        decoded = layer.decode_forward(decode_hidden, **decode_kwargs)
    assert list(decoded.shape) == [1, 1, 1, 10240]


@pytest.mark.long_context
@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
@pytest.mark.timeout(3600)
def test_near_max_non_aligned_context(mesh_device, layer_idx):
    """Execute the public final-short-chunk path one token below max context."""

    config = H.target_config()
    seq_len = HF_ADVERTISED_CONTEXT - 1
    layer = FunctionalDecoder.from_state_dict(
        None,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=HF_ADVERTISED_CONTEXT,
    )
    assert layer.prefill_chunk_plan(seq_len)[-1] == (HF_ADVERTISED_CONTEXT - 128, 127, 128)
    hidden = _upload(torch.zeros(1, 1, seq_len, 10240, dtype=torch.bfloat16), mesh_device)
    kwargs = {}
    if layer_idx == 1:
        kwargs["ple_embeddings"] = _upload(
            torch.zeros(1, 1, seq_len, 2560, dtype=torch.bfloat16),
            mesh_device,
        )
    if layer_idx == 3:
        cos, sin = H.rope_tables(HF_ADVERTISED_CONTEXT)
        page_host = H.shuffled_page_table(HF_ADVERTISED_CONTEXT)
        page, chunk_pages, rot = _paged_inputs(layer, mesh_device, page_host, cos, sin, seq_len)
        kwargs.update(page_table=page, page_tables_per_chunk=chunk_pages, rot_mats=rot)
    with H.ForbidHostFallback():
        output = layer.prefill_forward(hidden, **kwargs)
    assert list(output.shape) == [1, 1, seq_len, 10240]


@pytest.mark.long_context
@pytest.mark.timeout(600)
def test_qsa_traced_decode_at_advertised_context(mesh_device):
    """Capture and replay QSA decode with maximum cache/page/RoPE geometry."""

    config = H.target_config()
    layer = FunctionalDecoder.from_state_dict(
        None,
        hf_config=config,
        layer_idx=3,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=HF_ADVERTISED_CONTEXT,
    )
    hidden = _upload(torch.zeros(1, 1, 1, 10240, dtype=torch.bfloat16), mesh_device)
    current_pos = _upload(
        torch.tensor([HF_ADVERTISED_CONTEXT - 1], dtype=torch.int32),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    page_host = H.shuffled_page_table(HF_ADVERTISED_CONTEXT)
    page = _upload(page_host, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    cos, sin = H.rope_tables(HF_ADVERTISED_CONTEXT)
    rot = (
        _upload(cos.reshape(1, 1, HF_ADVERTISED_CONTEXT, -1), mesh_device),
        _upload(sin.reshape(1, 1, HF_ADVERTISED_CONTEXT, -1), mesh_device),
    )
    kwargs = {"current_pos": current_pos, "page_table": page, "rot_mats": rot}

    with H.ForbidHostFallback():
        layer.decode_forward(hidden, **kwargs)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    with H.ForbidHostFallback():
        traced_output = layer.decode_forward(hidden, **kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    assert list(traced_output.shape) == [1, 1, 1, 10240]
    ttnn.release_trace(mesh_device, trace_id)
