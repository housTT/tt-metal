# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""True-device acceptance gates for Ornith synchronized prefill batching.

These tests deliberately live apart from ``test_generator_vllm.py``.  They reuse that suite's
reduced two-layer construction (one DeltaNet layer and one full-attention layer), but allocate a
larger test KV pool so eight 2048-token prompts have disjoint physical pages.  Every test opens the
real 1x4 Blackhole mesh and is therefore collected, but never run, by host-only CI.

Run the five gates explicitly on an idle mesh::

    pytest -q models/autoports/ornith_ai_ornith_1_0_35b/tests/test_device_prefill_batching.py -x
"""

from __future__ import annotations

import copy
from contextlib import contextmanager

import pytest
import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tests import test_generator_vllm as serving_tests
from models.autoports.ornith_ai_ornith_1_0_35b.tt import model as M
from models.autoports.ornith_ai_ornith_1_0_35b.tt import multichip_decoder as MC
from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator_vllm import TTQwen3_5MoeForConditionalGeneration

TEST_BATCH = 8
TEST_CONTEXT = 4096
PREFILL_CHUNK = 2048
CONTINUATION_TAIL = 128
CONTINUED_LENGTH = PREFILL_CHUNK + CONTINUATION_TAIL
SLOTS4 = (7, 2, 6, 1)
SLOTS8 = (7, 2, 6, 1, 5, 3, 4, 0)
# B1 and B2/B4 select different low-precision matmul geometries, so comparing those two TTNN paths
# compounds their independent rounding error. Keep state/cache close to the repository's HF-golden
# bar, while logits get a separate cross-shape bound. Exact greedy-token checks are reserved for
# same-geometry comparisons, because nearly tied candidates can reverse across matmul geometries.
# Structural isolation below remains substantially tighter and is the gate that detects mixing.
CROSS_SHAPE_STATE_PCC_BAR = 0.995
CROSS_SHAPE_STATE_NRMSE_BAR = 0.10
CROSS_SHAPE_LOGITS_PCC_BAR = 0.99
CROSS_SHAPE_LOGITS_NRMSE_BAR = 0.20
# Fixed-shape paired controls exercise identical B2/B4 rows while another row, slot, or page-table
# entry changes. These are structural isolation checks, so retain a much tighter TT-vs-TT bar and
# require exact greedy agreement in every logits comparison.
ISOLATION_PCC_BAR = 0.999
ISOLATION_NRMSE_BAR = 0.02
# Decode isolation requires separate trace replays after resetting persistent state. The low-
# precision replay is noisier than a resident-pack comparison, so pair every perturbation with a
# fresh control and require the model-quality bar plus an identical greedy token.
REPLAY_ISOLATION_PCC_BAR = 0.99
REPLAY_ISOLATION_NRMSE_BAR = 0.20

pytestmark = [
    pytest.mark.parametrize("mesh_device", [MC.DEFAULT_MESH_SHAPE], indirect=True),
    pytest.mark.parametrize("device_params", serving_tests.DEVICE_PARAMS, indirect=True),
]


@pytest.fixture
def batching_adapter(mesh_device):
    """Build the plugin-facing reduced model with enough disjoint pages for eight users."""

    serving_tests._require_weights()
    hf_config = M.load_text_config(M.resolve_model_path())
    adapter = TTQwen3_5MoeForConditionalGeneration.initialize_vllm_model(
        hf_config,
        mesh_device,
        TEST_BATCH,
        max_seq_len=TEST_CONTEXT,
        tt_data_parallel=1,
        optimizations=None,
        layer_indices=serving_tests.PROBE_LAYERS,
    )
    model = adapter.model
    # Block zero is vLLM's null block. Every serving slot owns one complete 4096-token run after it.
    num_blocks = 1 + TEST_BATCH * adapter.page_table_blocks
    shape = (
        num_blocks,
        max(1, model.cfg.n_kv_heads // model.tp),
        model.page_block_size,
        model.cfg.head_dim,
    )
    kv_cache = adapter.allocate_kv_cache(shape, torch.bfloat16, len(model.layers))
    adapter.warmup_model_prefill(kv_cache=kv_cache, can_sample_on_device=True, enable_trace=False)
    adapter.warmup_model_decode(
        kv_cache=kv_cache,
        max_batch_size=TEST_BATCH,
        num_blocks=adapter.page_table_blocks,
        can_sample_on_device=True,
        enable_trace=False,
    )
    adapter.warmup_model_decode(
        kv_cache=kv_cache,
        max_batch_size=TEST_BATCH,
        num_blocks=adapter.page_table_blocks,
        can_sample_on_device=True,
        enable_trace=True,
    )
    adapter._test_kv_cache = kv_cache
    try:
        yield adapter
    finally:
        if adapter.generator is not None:
            adapter.generator.teardown()


def _page_table(adapter) -> torch.Tensor:
    """One complete, non-overlapping physical block run per serving slot."""

    width = adapter.page_table_blocks
    table = torch.empty(TEST_BATCH, width, dtype=torch.int32)
    for slot in range(TEST_BATCH):
        first = 1 + slot * width
        table[slot] = torch.arange(first, first + width, dtype=torch.int32)
    available = int(next(entry for entry in adapter._test_kv_cache if entry)[0].shape[0])
    assert int(table.max()) < available
    return table


def _prompts(rows: int, length: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(1, 100_000, (rows, length), generator=generator, dtype=torch.int64)
    # Pin row-distinct sentinels at both ends. A row permutation can no longer pass by feeding
    # identical random fixtures to two physical lanes.
    tokens[:, 0] = torch.arange(1001, 1001 + rows)
    tokens[:, -1] = torch.arange(20_001, 20_001 + rows)
    return tokens


def _reset(adapter) -> None:
    adapter.generator.reset()
    adapter._reset_serving_state()
    adapter.model.reset_prefill_batching_runtime()


def _prefill(adapter, tokens, lengths, slots, table, *, starts=0, sampling_params=None):
    result = adapter.prefill_forward(
        tokens=tokens,
        page_table=table[list(slots)],
        kv_cache=adapter._test_kv_cache,
        prompt_lens=list(lengths),
        start_pos=starts,
        sampling_params=sampling_params,
        empty_slots=list(slots),
    )
    return result[0] if adapter.uses_mrope else result


@contextmanager
def _record_physical_prefill_batches(model):
    """Spy immediately around the real embedding-to-layer-stack invocation."""

    calls = []
    original = model.ttnn_prefill_forward

    def wrapped(tokens_tt, *, start_pos, page_table=None):
        calls.append((int(tokens_tt.shape[0]), int(tokens_tt.shape[-1]), int(start_pos)))
        return original(tokens_tt, start_pos=start_pos, page_table=page_table)

    model.ttnn_prefill_forward = wrapped
    try:
        yield calls
    finally:
        # Restore the bound method, not merely the class descriptor: a failing device assertion must
        # not leave a spy installed for fixture teardown.
        model.ttnn_prefill_forward = original


@contextmanager
def _record_sample_rows(generator):
    calls = []
    original = generator._sample_prefill_row

    def wrapped(logits, *, local_row, user, slot, before_sample):
        value = original(
            logits,
            local_row=local_row,
            user=user,
            slot=slot,
            before_sample=before_sample,
        )
        calls.append((int(local_row), int(user), int(slot), int(value)))
        return value

    generator._sample_prefill_row = wrapped
    try:
        yield calls
    finally:
        generator._sample_prefill_row = original


@contextmanager
def _record_decode_refreshes(generator):
    """Capture the final host tensors copied into the persistent decode trace inputs."""

    calls = []
    original = generator._refresh_inputs

    def wrapped(device_inputs, tokens, positions, page_table, *, force=False, skip_tokens=False):
        calls.append(
            {
                "tokens": torch.as_tensor(tokens).clone(),
                "positions": torch.as_tensor(positions).clone(),
                "page_table": torch.as_tensor(page_table).clone(),
                "force": bool(force),
                "skip_tokens": bool(skip_tokens),
            }
        )
        return original(
            device_inputs,
            tokens,
            positions,
            page_table,
            force=force,
            skip_tokens=skip_tokens,
        )

    generator._refresh_inputs = wrapped
    try:
        yield calls
    finally:
        generator._refresh_inputs = original


def _single_serving_decode_refresh(calls, label: str) -> dict:
    staged = [call for call in calls if not call["force"]]
    assert len(staged) == 1, f"{label}: expected one non-capture refresh, got {len(staged)}"
    assert not staged[0]["skip_tokens"], f"{label}: host-authoritative decode must stage tokens"
    return staged[0]


def _state_snapshot(adapter, slots) -> dict:
    """All local-device recurrent and convolution-history rows for ``slots``."""

    model = adapter.model
    model._use_pack(adapter.max_batch_size, purpose="decode")
    snapshot = {}
    for layer in model.layers:
        if layer.is_full_attention or layer.recurrent_state is None:
            continue
        buffers = [("recurrent", layer.recurrent_state)] + [
            (f"conv{tap}", tensor) for tap, tensor in enumerate(layer.conv_state)
        ]
        for name, tensor in buffers:
            for device, shard in enumerate(ttnn.get_device_tensors(tensor)):
                host = serving_tests._slot_major(ttnn.to_torch(shard).float(), adapter.max_batch_size)
                snapshot[(int(layer.layer_idx), name, device)] = host[list(slots)].clone()
    assert snapshot, "the reduced target must include its real DeltaNet layer"
    return snapshot


def _used_blocks(adapter, table, slots, lengths) -> list[int]:
    blocks = set()
    for slot, length in zip(slots, lengths):
        physical = ((int(length) + 127) // 128) * 128
        count = (physical + adapter.model.page_block_size - 1) // adapter.model.page_block_size
        assert count <= int(table.shape[1]), f"length {length} needs {count} page-table entries"
        blocks.update(int(v) for v in table[int(slot), :count] if int(v) != 0)
    return sorted(blocks)


def _kv_snapshot(adapter, blocks) -> dict:
    """Every device-local K/V shard at the physical pages owned by this workload."""

    snapshot = {}
    for layer, entry in zip(adapter.model.layers, adapter._test_kv_cache):
        if not entry:
            continue
        for kind, tensor in zip(("k", "v"), entry):
            for device, shard in enumerate(ttnn.get_device_tensors(tensor)):
                snapshot[(int(layer.layer_idx), kind, device)] = ttnn.to_torch(shard).float()[blocks].clone()
    assert snapshot, "the reduced target must include its real paged-attention layer"
    return snapshot


def _assert_tensor_tracks(
    actual: torch.Tensor,
    expected: torch.Tensor,
    label: str,
    *,
    pcc_bar: float = CROSS_SHAPE_STATE_PCC_BAR,
    nrmse_bar: float = CROSS_SHAPE_STATE_NRMSE_BAR,
) -> None:
    """Catch row/state swaps while allowing normal B1-vs-B4 kernel rounding."""

    assert actual.shape == expected.shape, f"{label}: {tuple(actual.shape)} != {tuple(expected.shape)}"
    actual = actual.float().reshape(-1)
    expected = expected.float().reshape(-1)
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), f"{label}: non-finite value"
    reference_rms = torch.sqrt(torch.mean(expected.square()))
    error_rms = torch.sqrt(torch.mean((actual - expected).square()))
    if float(reference_rms) < 1e-8:
        assert float(error_rms) < 1e-5, f"{label}: expected zeros, RMS error {float(error_rms):.6g}"
        return
    nrmse = float(error_rms / reference_rms)
    gain = float(torch.dot(actual, expected) / torch.dot(expected, expected))
    pcc = float("nan")
    if actual.numel() > 1 and float(actual.std()) > 1e-8 and float(expected.std()) > 1e-8:
        pcc = float(torch.corrcoef(torch.stack((actual, expected)))[0, 1])
    detail = f"normalized RMS error {nrmse:.6f}, PCC {pcc:.6f}, least-squares gain {gain:.6f}"
    assert nrmse <= nrmse_bar, f"{label}: {detail}; NRMSE limit is {nrmse_bar}"
    if actual.numel() > 1 and float(actual.std()) > 1e-8 and float(expected.std()) > 1e-8:
        assert pcc >= pcc_bar, f"{label}: {detail}; PCC limit is {pcc_bar}"


def _assert_snapshot_tracks(
    actual: dict,
    expected: dict,
    label: str,
    *,
    pcc_bar: float = CROSS_SHAPE_STATE_PCC_BAR,
    nrmse_bar: float = CROSS_SHAPE_STATE_NRMSE_BAR,
) -> None:
    assert actual.keys() == expected.keys(), f"{label}: snapshot keys differ"
    for key in actual:
        _assert_tensor_tracks(
            actual[key],
            expected[key],
            f"{label} {key}",
            pcc_bar=pcc_bar,
            nrmse_bar=nrmse_bar,
        )


def _assert_logits_track(
    actual: torch.Tensor,
    expected: torch.Tensor,
    label: str,
    *,
    pcc_bar: float = CROSS_SHAPE_LOGITS_PCC_BAR,
    nrmse_bar: float = CROSS_SHAPE_LOGITS_NRMSE_BAR,
    require_exact_greedy: bool = False,
    aggregate: bool = False,
) -> None:
    assert tuple(actual.shape) == tuple(expected.shape)
    if aggregate:
        _assert_tensor_tracks(
            actual,
            expected,
            label,
            pcc_bar=pcc_bar,
            nrmse_bar=nrmse_bar,
        )
        if not require_exact_greedy:
            return
    for row in range(int(actual.shape[0])):
        got = actual[row].float().reshape(-1)
        want = expected[row].float().reshape(-1)
        _assert_tensor_tracks(
            got,
            want,
            f"{label} row {row}",
            pcc_bar=pcc_bar,
            nrmse_bar=nrmse_bar,
        )
        if require_exact_greedy:
            got_top = torch.topk(got, k=5)
            want_top = torch.topk(want, k=5)
            assert int(got_top.indices[0]) == int(want_top.indices[0]), (
                f"{label} row {row}: greedy token differs; "
                f"got top-5={list(zip(got_top.indices.tolist(), got_top.values.tolist()))}, "
                f"reference top-5={list(zip(want_top.indices.tolist(), want_top.values.tolist()))}"
            )


def _serialized_prefill(adapter, tokens, lengths, slots, table):
    """Accepted B1 path, preserving completed decode rows while processing users serially."""

    _reset(adapter)
    outputs = []
    for user, (length, slot) in enumerate(zip(lengths, slots)):
        outputs.append(
            _prefill(
                adapter,
                tokens[user : user + 1],
                [length],
                [slot],
                table,
                starts=[0],
            )
        )
    blocks = _used_blocks(adapter, table, slots, lengths)
    return (
        torch.cat(outputs, dim=0),
        _state_snapshot(adapter, slots),
        _kv_snapshot(adapter, blocks),
    )


def _assert_true_batch_matches_serial(adapter, physical_batch: int, *, seed: int) -> None:
    table = _page_table(adapter)
    slots = SLOTS4[:physical_batch]
    tokens = _prompts(physical_batch, PREFILL_CHUNK, seed)
    lengths = [PREFILL_CHUNK] * physical_batch

    _reset(adapter)
    with _record_physical_prefill_batches(adapter.model) as calls:
        grouped = _prefill(adapter, tokens, lengths, slots, table, starts=[0] * physical_batch)
    runtime = copy.deepcopy(adapter.model.prefill_batching_runtime)
    grouped_state = _state_snapshot(adapter, slots)
    grouped_kv = _kv_snapshot(adapter, _used_blocks(adapter, table, slots, lengths))

    assert calls == [(physical_batch, PREFILL_CHUNK, 0)], "one real model graph must receive the physical batch"
    assert runtime["device_invocations"] == 1
    assert runtime["physical_batch_histogram"] == {str(physical_batch): 1}
    assert runtime["logical_users"] == physical_batch
    assert runtime["logical_tokens"] == physical_batch * PREFILL_CHUNK
    assert runtime["fallback_invocations"] == 0 and runtime["fallback_reasons"] == {}
    assert runtime["batched_device_invocations"] == int(physical_batch > 1)

    serialized, serialized_state, serialized_kv = _serialized_prefill(adapter, tokens, lengths, slots, table)
    _assert_snapshot_tracks(grouped_state, serialized_state, f"B{physical_batch} recurrent state")
    _assert_snapshot_tracks(grouped_kv, serialized_kv, f"B{physical_batch} paged KV")
    _assert_logits_track(
        grouped,
        serialized,
        f"B{physical_batch} logits",
        require_exact_greedy=physical_batch == 1,
        aggregate=physical_batch > 1,
    )


def test_d01_batch1_control_matches_serial_reference(batching_adapter):
    _assert_true_batch_matches_serial(batching_adapter, 1, seed=101)


def _assert_fixed_shape_b2_row_isolation(adapter) -> None:
    """A second B2 lane must not affect the first lane's logits, state, or paged KV."""

    table = _page_table(adapter)
    slots = SLOTS4[:2]
    first = _prompts(2, PREFILL_CHUNK, 2202)
    second = first.clone()
    second[1] = _prompts(1, PREFILL_CHUNK, 2203)[0]

    def run(tokens):
        _reset(adapter)
        logits = _prefill(adapter, tokens, [PREFILL_CHUNK] * 2, slots, table, starts=[0, 0])
        state = _state_snapshot(adapter, slots[:1])
        blocks = _used_blocks(adapter, table, slots[:1], [PREFILL_CHUNK])
        return logits[:1], state, _kv_snapshot(adapter, blocks)

    first_logits, first_state, first_kv = run(first)
    second_logits, second_state, second_kv = run(second)
    _assert_logits_track(
        first_logits,
        second_logits,
        "fixed-B2 row isolation logits",
        pcc_bar=ISOLATION_PCC_BAR,
        nrmse_bar=ISOLATION_NRMSE_BAR,
        require_exact_greedy=True,
    )
    _assert_snapshot_tracks(
        first_state,
        second_state,
        "fixed-B2 row isolation recurrent state",
        pcc_bar=ISOLATION_PCC_BAR,
        nrmse_bar=ISOLATION_NRMSE_BAR,
    )
    _assert_snapshot_tracks(
        first_kv,
        second_kv,
        "fixed-B2 row isolation paged KV",
        pcc_bar=ISOLATION_PCC_BAR,
        nrmse_bar=ISOLATION_NRMSE_BAR,
    )


def test_d02_batch2_is_one_true_device_graph_and_matches_two_serial_calls(batching_adapter):
    _assert_true_batch_matches_serial(batching_adapter, 2, seed=202)
    _assert_fixed_shape_b2_row_isolation(batching_adapter)


def test_d03_batch4_is_true_and_count8_is_two_batch4_calls_with_per_row_device_sampling(
    batching_adapter,
):
    adapter = batching_adapter
    _assert_true_batch_matches_serial(adapter, 4, seed=303)

    table = _page_table(adapter)
    tokens = _prompts(8, CONTINUED_LENGTH, 308)
    lengths = [PREFILL_CHUNK] * 8
    _reset(adapter)
    host_logits = _prefill(adapter, tokens, lengths, SLOTS8, table, starts=[0] * 8)
    host_state = _state_snapshot(adapter, SLOTS8)
    host_kv = _kv_snapshot(adapter, _used_blocks(adapter, table, SLOTS8, lengths))
    c8_wave_states = []
    c8_wave_kvs = []
    for wave in range(2):
        users = slice(4 * wave, 4 * wave + 4)
        wave_slots = SLOTS8[users]
        c8_wave_states.append(_state_snapshot(adapter, wave_slots))
        c8_wave_kvs.append(_kv_snapshot(adapter, _used_blocks(adapter, table, wave_slots, lengths[users])))
    assert adapter.model._prefill_pack_slots[(4, 0)] == SLOTS8[:4]
    assert adapter.model._prefill_pack_slots[(4, 1)] == SLOTS8[4:]

    # Prove both waves are stable at fixed B4 geometry, especially the second resident lane. Running
    # either wave alone selects canonical lane zero but keeps the same kernels, rows, slots and KV
    # pages. These comparisons catch lane or staging corruption without using B1 as a structural
    # oracle for low-precision B4 logits.
    for wave in range(2):
        users = slice(4 * wave, 4 * wave + 4)
        wave_slots = SLOTS8[users]
        _reset(adapter)
        standalone_logits = _prefill(
            adapter,
            tokens[users],
            lengths[users],
            wave_slots,
            table,
            starts=[0] * 4,
        )
        standalone_state = _state_snapshot(adapter, wave_slots)
        standalone_kv = _kv_snapshot(adapter, _used_blocks(adapter, table, wave_slots, lengths[users]))
        label = f"C8 B4 wave {wave} vs standalone B4"
        _assert_logits_track(
            host_logits[users],
            standalone_logits,
            label,
            pcc_bar=ISOLATION_PCC_BAR,
            nrmse_bar=ISOLATION_NRMSE_BAR,
            require_exact_greedy=True,
        )
        _assert_snapshot_tracks(
            c8_wave_states[wave],
            standalone_state,
            f"{label} state",
            pcc_bar=ISOLATION_PCC_BAR,
            nrmse_bar=ISOLATION_NRMSE_BAR,
        )
        _assert_snapshot_tracks(
            c8_wave_kvs[wave],
            standalone_kv,
            f"{label} paged KV",
            pcc_bar=ISOLATION_PCC_BAR,
            nrmse_bar=ISOLATION_NRMSE_BAR,
        )

    _, serialized_state, serialized_kv = _serialized_prefill(adapter, tokens, lengths, SLOTS8, table)
    _assert_snapshot_tracks(host_state, serialized_state, "C8 B4+B4 recurrent state")
    _assert_snapshot_tracks(host_kv, serialized_kv, "C8 B4+B4 paged KV")
    # Sampling correctness is a same-geometry property.  B1 and B4 use
    # different low-precision matmul geometries, so nearly tied top candidates
    # may legitimately reverse even while the full logits track closely.
    greedy_reference = torch.argmax(host_logits[:, 0, :], dim=-1).to(torch.int32)

    _reset(adapter)
    with _record_physical_prefill_batches(adapter.model) as batches:
        with _record_sample_rows(adapter.generator) as sampled_rows:
            sampled = _prefill(
                adapter,
                tokens,
                lengths,
                SLOTS8,
                table,
                starts=[0] * 8,
                sampling_params=serving_tests.greedy_params(adapter.max_batch_size),
            )
    runtime = copy.deepcopy(adapter.model.prefill_batching_runtime)
    assert batches == [(4, PREFILL_CHUNK, 0), (4, PREFILL_CHUNK, 0)]
    assert [(row, user, slot) for row, user, slot, _ in sampled_rows] == [
        (user % 4, user, SLOTS8[user]) for user in range(8)
    ]
    assert torch.equal(sampled.to(torch.int32), greedy_reference), "each device-sampled row must use its own logits"
    assert [value for *_, value in sampled_rows] == [int(v) for v in sampled]
    assert runtime["device_invocations"] == runtime["batched_device_invocations"] == 2
    assert runtime["physical_batch_histogram"] == {"4": 2}
    assert runtime["logical_users"] == 8 and runtime["logical_tokens"] == 8 * PREFILL_CHUNK
    assert runtime["fallback_invocations"] == 0
    assert runtime["fallback_reasons"] == {}
    assert adapter.model._prefill_pack_slots[(4, 0)] == SLOTS8[:4]
    assert adapter.model._prefill_pack_slots[(4, 1)] == SLOTS8[4:]

    # Both resident C8 waves must retain their own B4 continuation authority. This closes the lane-1
    # cross-product that fresh-only C8 coverage cannot prove: no migration is needed and neither wave
    # may be restored from the fixed-batch decode pack.
    before = copy.deepcopy(adapter.model.prefill_batching_runtime)
    with _record_physical_prefill_batches(adapter.model) as continuation_batches:
        continued = _prefill(
            adapter,
            tokens,
            [CONTINUED_LENGTH] * 8,
            SLOTS8,
            table,
            starts=[PREFILL_CHUNK] * 8,
        )
    after = copy.deepcopy(adapter.model.prefill_batching_runtime)
    continued_state = _state_snapshot(adapter, SLOTS8)
    continued_kv = _kv_snapshot(adapter, _used_blocks(adapter, table, SLOTS8, [CONTINUED_LENGTH] * 8))
    assert continuation_batches == [
        (4, CONTINUATION_TAIL, PREFILL_CHUNK),
        (4, CONTINUATION_TAIL, PREFILL_CHUNK),
    ]
    assert after["device_invocations"] - before["device_invocations"] == 2
    assert after["batched_device_invocations"] - before["batched_device_invocations"] == 2
    assert after["grouped_chunks"] - before["grouped_chunks"] == 2
    assert after["grouped_users"] - before["grouped_users"] == 8
    assert after["logical_users"] - before["logical_users"] == 8
    assert after["logical_tokens"] - before["logical_tokens"] == 8 * CONTINUATION_TAIL
    assert after["physical_batch_histogram"]["4"] - before["physical_batch_histogram"]["4"] == 2
    assert after["migrated_continuations"] == before["migrated_continuations"]
    assert after["migrated_users"] == before["migrated_users"]
    assert after["fallback_invocations"] == before["fallback_invocations"]
    assert after["fallback_reasons"] == before["fallback_reasons"]
    assert adapter.model._prefill_pack_slots[(4, 0)] == SLOTS8[:4]
    assert adapter.model._prefill_pack_slots[(4, 1)] == SLOTS8[4:]

    serialized_continued, serialized_continued_state, serialized_continued_kv = _serialized_two_chunk_prompts(
        adapter,
        tokens,
        SLOTS8,
        table,
    )
    _assert_logits_track(
        continued,
        serialized_continued,
        "C8 B4+B4 continuation",
        aggregate=True,
    )
    _assert_snapshot_tracks(
        continued_state,
        serialized_continued_state,
        "C8 B4+B4 continuation state",
    )
    _assert_snapshot_tracks(
        continued_kv,
        serialized_continued_kv,
        "C8 B4+B4 continuation paged KV",
    )


def _decode_host(adapter, table, tokens_by_slot, positions_by_slot):
    return adapter.decode_forward(
        tokens=tokens_by_slot,
        page_table=table,
        kv_cache=adapter._test_kv_cache,
        start_pos=positions_by_slot,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        reset_batch=True,
    )


def _assert_inactive_state_is_zero(adapter, active_slots) -> None:
    inactive = [slot for slot in range(adapter.max_batch_size) if slot not in set(active_slots)]
    for key, tensor in _state_snapshot(adapter, inactive).items():
        assert torch.count_nonzero(tensor) == 0, f"prefill advanced inactive state {key}"


def _serialized_two_chunk_prompts(adapter, prompts, slots, table):
    """Complete one long request at a time so one B1 authority pack is sufficient."""

    _reset(adapter)
    outputs = []
    for user, slot in enumerate(slots):
        _prefill(
            adapter,
            prompts[user : user + 1, :PREFILL_CHUNK],
            [PREFILL_CHUNK],
            [slot],
            table,
            starts=[0],
        )
        outputs.append(
            _prefill(
                adapter,
                prompts[user : user + 1],
                [CONTINUED_LENGTH],
                [slot],
                table,
                starts=[PREFILL_CHUNK],
            )
        )
    blocks = _used_blocks(adapter, table, slots, [CONTINUED_LENGTH] * len(slots))
    return (
        torch.cat(outputs),
        _state_snapshot(adapter, slots),
        _kv_snapshot(adapter, blocks),
    )


def _advance_interloper_while_wave_is_paused(adapter, table, source_tokens, source_slots):
    """Pause ``source_slots`` in prefill packs, then run one real decode for slot zero."""

    short_len = 128  # included in the reduced adapter's bounded warm-up set
    short = _prompts(1, short_len, 40_400 + len(source_slots))
    interloper_logits = _prefill(adapter, short, [short_len], [0], table, starts=[0])
    _prefill(
        adapter,
        source_tokens[:, :PREFILL_CHUNK],
        [PREFILL_CHUNK] * len(source_slots),
        source_slots,
        table,
        starts=[0] * len(source_slots),
    )
    decode_tokens = torch.zeros(TEST_BATCH, dtype=torch.int64)
    positions = torch.full((TEST_BATCH,), -1, dtype=torch.int64)
    decode_tokens[0] = int(torch.argmax(interloper_logits[0, 0]))
    positions[0] = short_len
    _decode_host(adapter, table, decode_tokens, positions)


def _run_continuation_case(
    adapter,
    table,
    *,
    source_batch: int,
    survivor_order,
    expected_batches,
    expected_migrations: int,
    seed: int,
) -> None:
    source_slots = SLOTS4[:source_batch]
    prompts = _prompts(source_batch, CONTINUED_LENGTH, seed)
    _reset(adapter)
    _advance_interloper_while_wave_is_paused(adapter, table, prompts, source_slots)

    survivor_order = tuple(int(v) for v in survivor_order)
    survivor_slots = tuple(source_slots[index] for index in survivor_order)
    survivor_prompts = prompts[list(survivor_order)]
    before = copy.deepcopy(adapter.model.prefill_batching_runtime)
    with _record_physical_prefill_batches(adapter.model) as batches:
        actual = _prefill(
            adapter,
            survivor_prompts,
            [CONTINUED_LENGTH] * len(survivor_slots),
            survivor_slots,
            table,
            starts=[PREFILL_CHUNK] * len(survivor_slots),
        )
    after = copy.deepcopy(adapter.model.prefill_batching_runtime)
    actual_state = _state_snapshot(adapter, survivor_slots)
    actual_kv = _kv_snapshot(
        adapter,
        _used_blocks(adapter, table, survivor_slots, [CONTINUED_LENGTH] * len(survivor_slots)),
    )

    assert [batch for batch, length, start in batches] == list(expected_batches)
    assert all(length == CONTINUATION_TAIL and start == PREFILL_CHUNK for _, length, start in batches)
    assert after["migrated_continuations"] - before["migrated_continuations"] == expected_migrations
    assert after["migrated_users"] - before["migrated_users"] == (len(survivor_slots) if expected_migrations else 0)
    assert after["fallback_invocations"] == before["fallback_invocations"]
    assert after["fallback_reasons"] == before["fallback_reasons"]
    expected_invocations = len(expected_batches)
    expected_batched = sum(batch > 1 for batch in expected_batches)
    expected_users = sum(expected_batches)
    assert after["device_invocations"] - before["device_invocations"] == expected_invocations
    assert after["batched_device_invocations"] - before["batched_device_invocations"] == expected_batched
    assert after["grouped_chunks"] - before["grouped_chunks"] == expected_batched
    assert after["grouped_users"] - before["grouped_users"] == sum(batch for batch in expected_batches if batch > 1)
    assert after["logical_users"] - before["logical_users"] == expected_users
    assert after["logical_tokens"] - before["logical_tokens"] == expected_users * CONTINUATION_TAIL
    histogram_keys = set(before["physical_batch_histogram"]) | set(after["physical_batch_histogram"])
    histogram_delta = {
        key: after["physical_batch_histogram"].get(key, 0) - before["physical_batch_histogram"].get(key, 0)
        for key in histogram_keys
        if after["physical_batch_histogram"].get(key, 0) != before["physical_batch_histogram"].get(key, 0)
    }
    expected_histogram = {str(batch): list(expected_batches).count(batch) for batch in sorted(set(expected_batches))}
    assert histogram_delta == expected_histogram

    decode_tokens = torch.zeros(TEST_BATCH, dtype=torch.int64)
    decode_positions = torch.full((TEST_BATCH,), -1, dtype=torch.int64)
    for user, slot in enumerate(survivor_slots):
        decode_tokens[slot] = 60_001 + user
        decode_positions[slot] = CONTINUED_LENGTH + user
    actual_decode = _decode_host(adapter, table, decode_tokens, decode_positions)

    expected, expected_state, expected_kv = _serialized_two_chunk_prompts(
        adapter, survivor_prompts, survivor_slots, table
    )
    expected_decode = _decode_host(adapter, table, decode_tokens, decode_positions)
    _assert_logits_track(
        actual,
        expected,
        f"B{source_batch}->{len(survivor_slots)} continuation",
        aggregate=True,
    )
    _assert_snapshot_tracks(
        actual_state,
        expected_state,
        f"B{source_batch}->{len(survivor_slots)} continuation state",
    )
    _assert_snapshot_tracks(
        actual_kv,
        expected_kv,
        f"B{source_batch}->{len(survivor_slots)} continuation KV",
    )
    _assert_logits_track(
        actual_decode[list(survivor_slots)],
        expected_decode[list(survivor_slots)],
        f"B{source_batch}->{len(survivor_slots)} post-continuation decode",
        aggregate=True,
    )


def _run_split_b2_b1_authority_case(adapter, table) -> None:
    """A B2 survivor must not be greedily combined with the live B1 remainder."""

    source_slots = SLOTS4[:3]
    prompts = _prompts(3, CONTINUED_LENGTH, 446)
    _reset(adapter)
    _advance_interloper_while_wave_is_paused(adapter, table, prompts, source_slots)

    # Initial C3 admission is B2(rows 0,1)+B1(row 2). After row 0 aborts, caller order places the
    # surviving B2 row before the original B1 row. A fresh greedy decomposition would incorrectly
    # mix them into B2; authority-aware planning must use a collision-free B1 lane plus the original.
    survivor_order = (1, 2)
    survivor_slots = tuple(source_slots[index] for index in survivor_order)
    survivor_prompts = prompts[list(survivor_order)]
    before = copy.deepcopy(adapter.model.prefill_batching_runtime)
    with _record_physical_prefill_batches(adapter.model) as batches:
        actual = _prefill(
            adapter,
            survivor_prompts,
            [CONTINUED_LENGTH] * 2,
            survivor_slots,
            table,
            starts=[PREFILL_CHUNK] * 2,
        )
    after = copy.deepcopy(adapter.model.prefill_batching_runtime)
    actual_state = _state_snapshot(adapter, survivor_slots)
    actual_kv = _kv_snapshot(
        adapter,
        _used_blocks(adapter, table, survivor_slots, [CONTINUED_LENGTH] * len(survivor_slots)),
    )

    assert batches == [
        (1, CONTINUATION_TAIL, PREFILL_CHUNK),
        (1, CONTINUATION_TAIL, PREFILL_CHUNK),
    ]
    assert after["migrated_continuations"] - before["migrated_continuations"] == 1
    assert after["migrated_users"] - before["migrated_users"] == 1
    assert (
        after["fallback_invocations"] == before["fallback_invocations"]
    ), "the initial singleton remainder is already accounted; authority preservation is not a fallback"
    assert after["fallback_reasons"] == before["fallback_reasons"]
    assert after["device_invocations"] - before["device_invocations"] == 2
    assert after["batched_device_invocations"] == before["batched_device_invocations"]
    assert after["grouped_chunks"] == before["grouped_chunks"]
    assert after["grouped_users"] == before["grouped_users"]
    assert after["logical_users"] - before["logical_users"] == 2
    assert after["logical_tokens"] - before["logical_tokens"] == 2 * CONTINUATION_TAIL
    assert after["physical_batch_histogram"].get("1", 0) - before["physical_batch_histogram"].get("1", 0) == 2

    decode_tokens = torch.zeros(TEST_BATCH, dtype=torch.int64)
    decode_positions = torch.full((TEST_BATCH,), -1, dtype=torch.int64)
    for user, slot in enumerate(survivor_slots):
        decode_tokens[slot] = 61_001 + user
        decode_positions[slot] = CONTINUED_LENGTH + user
    actual_decode = _decode_host(adapter, table, decode_tokens, decode_positions)

    expected, expected_state, expected_kv = _serialized_two_chunk_prompts(
        adapter, survivor_prompts, survivor_slots, table
    )
    expected_decode = _decode_host(adapter, table, decode_tokens, decode_positions)
    _assert_logits_track(actual, expected, "B2 survivor plus original B1 continuation", aggregate=True)
    _assert_snapshot_tracks(actual_state, expected_state, "split B2/B1 continuation state")
    _assert_snapshot_tracks(actual_kv, expected_kv, "split B2/B1 continuation KV")
    _assert_logits_track(
        actual_decode[list(survivor_slots)],
        expected_decode[list(survivor_slots)],
        "split B2/B1 post-continuation decode",
        aggregate=True,
    )


@pytest.mark.timeout(900)
def test_d04_grouped_state_is_slot_isolated_and_continuations_survive_decode_shrink_and_reorder(
    batching_adapter,
):
    adapter = batching_adapter
    table = _page_table(adapter)
    prompts = _prompts(4, PREFILL_CHUNK, 404)
    slots = SLOTS4

    # A real B4 prefill may touch only the four assigned decode rows. Distinct decode tokens and
    # positions then have to agree with the same four prompts admitted through B1 one at a time.
    _reset(adapter)
    grouped_prefill = _prefill(adapter, prompts, [PREFILL_CHUNK] * 4, slots, table, starts=[0] * 4)
    _assert_inactive_state_is_zero(adapter, slots)
    decode_tokens = torch.zeros(TEST_BATCH, dtype=torch.int64)
    decode_positions = torch.full((TEST_BATCH,), -1, dtype=torch.int64)
    for user, slot in enumerate(slots):
        decode_tokens[slot] = 40_001 + user
        decode_positions[slot] = PREFILL_CHUNK + user
    grouped_decode = _decode_host(adapter, table, decode_tokens, decode_positions)

    serialized_prefill, _, _ = _serialized_prefill(
        adapter,
        prompts,
        [PREFILL_CHUNK] * 4,
        slots,
        table,
    )
    serialized_decode = _decode_host(adapter, table, decode_tokens, decode_positions)
    _assert_logits_track(grouped_prefill, serialized_prefill, "B4 isolated prefill", aggregate=True)
    _assert_logits_track(
        grouped_decode[list(slots)],
        serialized_decode[list(slots)],
        "B4 isolated decode",
        aggregate=True,
    )

    def fresh_b4_decode(*, decode_table=table, tokens=decode_tokens):
        """Pair each isolation perturbation with a control under the same warmed trace state."""

        _reset(adapter)
        _prefill(adapter, prompts, [PREFILL_CHUNK] * 4, slots, table, starts=[0] * 4)
        return _decode_host(adapter, decode_table, tokens, decode_positions)

    # Decode receives all eight page-table rows. Perturb two non-owner rows while leaving their
    # positions idle; none of the four owners may observe that unrelated scheduler metadata.
    perturbed_table = table.clone()
    perturbed_table[[0, 3]] = perturbed_table[[3, 0]]
    assert not torch.equal(perturbed_table[0], table[0])
    control_decode = fresh_b4_decode()
    perturbed_decode = fresh_b4_decode(decode_table=perturbed_table)
    for slot in slots:
        _assert_logits_track(
            perturbed_decode[slot : slot + 1],
            control_decode[slot : slot + 1],
            f"owner slot {slot} after non-owner page-row perturbation",
            pcc_bar=REPLAY_ISOLATION_PCC_BAR,
            nrmse_bar=REPLAY_ISOLATION_NRMSE_BAR,
            require_exact_greedy=True,
        )

    # Redirect exactly one active row to a valid foreign block run. Only that request may observe
    # the alternate paged-KV history; the other three decode rows must remain numerically invariant
    # and routed to their original owners.
    redirected_slot = slots[1]
    redirected_table = table.clone()
    redirected_table[redirected_slot] = table[0]
    control_decode = fresh_b4_decode()
    with _record_decode_refreshes(adapter.generator) as refreshes:
        redirected_decode = fresh_b4_decode(decode_table=redirected_table)
    staged = _single_serving_decode_refresh(refreshes, "active page-row redirection")
    assert torch.equal(staged["tokens"], decode_tokens)
    assert torch.equal(staged["positions"], decode_positions)
    assert torch.equal(staged["page_table"], redirected_table)
    changed_page_rows = torch.nonzero(torch.any(staged["page_table"] != table, dim=1)).flatten().tolist()
    assert changed_page_rows == [redirected_slot]
    assert torch.equal(staged["page_table"][redirected_slot], table[0])
    for slot in (slots[0], slots[2], slots[3]):
        _assert_logits_track(
            redirected_decode[slot : slot + 1],
            control_decode[slot : slot + 1],
            f"unchanged owner slot {slot} after active page-row redirection",
            pcc_bar=REPLAY_ISOLATION_PCC_BAR,
            nrmse_bar=REPLAY_ISOLATION_NRMSE_BAR,
            require_exact_greedy=True,
        )
    assert torch.isfinite(redirected_decode[redirected_slot]).all()

    # Changing one decode token must not perturb the other three request rows.
    changed_tokens = decode_tokens.clone()
    changed_tokens[slots[2]] += 1
    control_decode = fresh_b4_decode()
    with _record_decode_refreshes(adapter.generator) as refreshes:
        changed_decode = fresh_b4_decode(tokens=changed_tokens)
    staged = _single_serving_decode_refresh(refreshes, "single decode-token change")
    assert torch.equal(staged["positions"], decode_positions)
    assert torch.equal(staged["page_table"], table)
    changed_token_rows = torch.nonzero(staged["tokens"] != decode_tokens).flatten().tolist()
    assert changed_token_rows == [slots[2]]
    assert int(staged["tokens"][slots[2]]) == int(decode_tokens[slots[2]]) + 1
    for slot in (slots[0], slots[1], slots[3]):
        _assert_logits_track(
            changed_decode[slot : slot + 1],
            control_decode[slot : slot + 1],
            f"unchanged decode slot {slot}",
            pcc_bar=REPLAY_ISOLATION_PCC_BAR,
            nrmse_bar=REPLAY_ISOLATION_NRMSE_BAR,
            require_exact_greedy=True,
        )
    assert torch.isfinite(changed_decode[slots[2]]).all()

    # Full-wave reorder stays on its B4 authority. Shrinks migrate by source slot identity into
    # collision-free B2/B1 packs after an intervening fixed-batch decode has corrupted decode rows.
    cases = (
        # caller order is deliberately different from source row order
        (4, (3, 0, 2, 1), (4,), 0, 441),
        (4, (3, 0, 2), (2, 1), 2, 442),
        (4, (2, 0), (2,), 1, 443),
        (4, (3,), (1,), 1, 444),
        (2, (1,), (1,), 1, 445),
    )
    for source_batch, survivor_order, expected_batches, migrations, seed in cases:
        _run_continuation_case(
            adapter,
            table,
            source_batch=source_batch,
            survivor_order=survivor_order,
            expected_batches=expected_batches,
            expected_migrations=migrations,
            seed=seed,
        )
    _run_split_b2_b1_authority_case(adapter, table)


def test_d05_mixed_geometry_falls_back_exactly_while_equal_nonaligned_rows_stay_batched(
    batching_adapter,
    expect_error,
):
    adapter = batching_adapter
    table = _page_table(adapter)
    slots = (7, 2)

    # Unequal fresh rows are two accepted B1 calls, not a false B2 claim.
    mixed_tokens = _prompts(2, PREFILL_CHUNK, 505)
    mixed_lengths = (PREFILL_CHUNK, 1024)
    _reset(adapter)
    with _record_physical_prefill_batches(adapter.model) as calls:
        mixed = _prefill(adapter, mixed_tokens, mixed_lengths, slots, table, starts=[0, 0])
    runtime = copy.deepcopy(adapter.model.prefill_batching_runtime)
    mixed_state = _state_snapshot(adapter, slots)
    mixed_kv = _kv_snapshot(adapter, _used_blocks(adapter, table, slots, mixed_lengths))
    assert calls == [(1, PREFILL_CHUNK, 0), (1, 1024, 0)]
    assert runtime["physical_batch_histogram"] == {"1": 2}
    assert runtime["batched_device_invocations"] == 0
    assert runtime["logical_users"] == 2 and runtime["logical_tokens"] == 3072
    assert runtime["fallback_invocations"] == 2
    assert runtime["fallback_reasons"] == {"mixed_chunk_geometry": 2}
    serialized, serialized_state, serialized_kv = _serialized_prefill(
        adapter, mixed_tokens, mixed_lengths, slots, table
    )
    _assert_logits_track(mixed, serialized, "mixed-length fallback")
    _assert_snapshot_tracks(mixed_state, serialized_state, "mixed-length fallback state")
    _assert_snapshot_tracks(mixed_kv, serialized_kv, "mixed-length fallback paged KV")

    # Equal logical tails need not be 128-aligned. They share one physical padded shape and remain
    # a true B2 invocation; first use may compile, but it is not a semantic fallback.
    tail = 129
    tail_tokens = _prompts(2, tail, 506)
    _reset(adapter)
    with _record_physical_prefill_batches(adapter.model) as calls:
        grouped_tail = _prefill(adapter, tail_tokens, [tail, tail], slots, table, starts=[0, 0])
    runtime = copy.deepcopy(adapter.model.prefill_batching_runtime)
    grouped_tail_state = _state_snapshot(adapter, slots)
    grouped_tail_kv = _kv_snapshot(adapter, _used_blocks(adapter, table, slots, [tail, tail]))
    assert calls == [(2, tail, 0)]
    assert runtime["physical_batch_histogram"] == {"2": 1}
    assert runtime["logical_tokens"] == 2 * tail and runtime["fallback_invocations"] == 0
    serialized_tail, serialized_tail_state, serialized_tail_kv = _serialized_prefill(
        adapter, tail_tokens, [tail, tail], slots, table
    )
    _assert_logits_track(grouped_tail, serialized_tail, "equal non-aligned B2 tail", aggregate=True)
    _assert_snapshot_tracks(grouped_tail_state, serialized_tail_state, "equal non-aligned B2 tail state")
    _assert_snapshot_tracks(grouped_tail_kv, serialized_tail_kv, "equal non-aligned B2 tail paged KV")

    # A mixed continuation is unsafe: one row would have to restore from an advanced decode slot.
    _reset(adapter)
    ragged = _prompts(2, 2 * PREFILL_CHUNK, 507)
    with expect_error(RuntimeError, "must retain one synchronized start and chunk length"):
        _prefill(
            adapter,
            ragged,
            [2 * PREFILL_CHUNK, PREFILL_CHUNK],
            slots,
            table,
            starts=[PREFILL_CHUNK, 0],
        )
    runtime = adapter.model.prefill_batching_runtime
    assert runtime["device_invocations"] == 0
    assert runtime["fallback_invocations"] == 2
    assert runtime["fallback_reasons"] == {"unsynchronized_continuation_refused": 2}
