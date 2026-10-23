# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU lowered cache-address controls for the production continuation path."""

from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch

from models.autoports.google_gemma_4_26b_a4b_it.tt import multichip_decoder as module
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import _layer_kind


class Tensor:
    dtype = "bf16"

    def __init__(self, shape, *, rows=None, table=None, cache=None, value=None):
        self.shape = tuple(shape)
        self.rows, self.table, self.cache, self.value = rows, table, cache, value

    def memory_config(self):
        return None

    def deallocate(self, *_):
        pass


def recording_runtime(monkeypatch):
    def sliced(tensor, starts, ends, *args, **kwargs):
        shape = [end - start for start, end in zip(starts, ends)]
        if tensor.table is not None:
            return Tensor(shape, table=[row[starts[1] : ends[1]] for row in tensor.table[starts[0] : ends[0]]])
        if tensor.cache is not None:
            assert shape[0] == 1
            rows = [tensor.cache.get((starts[0], row)) for row in range(starts[2], ends[2])]
        else:
            rows = None if tensor.rows is None else tensor.rows[starts[2] : ends[2]]
        return Tensor(shape, rows=rows)

    def concat(items, dim, **kwargs):
        shape = list(items[0].shape)
        shape[dim] = sum(t.shape[dim] for t in items)
        if items[0].table is not None:
            return Tensor(shape, table=[sum([item.table[row] for item in items], []) for row in range(shape[0])])
        return Tensor(shape, rows=sum([item.rows for item in items], []))

    def transpose(tensor, a, b):
        shape = list(tensor.shape)
        shape[a], shape[b] = shape[b], shape[a]
        return Tensor(shape, rows=tensor.rows)

    def fill(cache, values, table, *, batch_idx, block_size=None, cache_position_modulo=None):
        block = block_size or cache.shape[2]
        for offset, value in enumerate(values.rows):
            position = offset if cache_position_modulo is None else offset % cache_position_modulo
            cache.cache[(table.table[batch_idx][position // block], position % block)] = value

    def update(
        cache, values, *, update_idxs_tensor, page_table, block_size=None, num_kv_heads=None, cache_position_modulo=None
    ):
        block = block_size or cache.shape[2]
        position = update_idxs_tensor.value
        if cache_position_modulo is not None:
            position %= cache_position_modulo
        cache.cache[(page_table.table[0][position // block], position % block)] = values.rows[0]

    def pad(tensor, padding, value):
        shape = [n + a + b for n, (a, b) in zip(tensor.shape, padding)]
        a, b = padding[2]
        return Tensor(shape, rows=[None] * a + tensor.rows + [None] * b)

    calls = []

    def sdpa(q, *args, chunk_start_idx, **kwargs):
        calls.append((chunk_start_idx, list(q.rows)))
        return Tensor(q.shape, rows=list(q.rows))

    tt = NS(
        Tensor=Tensor,
        bfloat16="bf16",
        int32="int32",
        ROW_MAJOR_LAYOUT=None,
        TILE_LAYOUT=None,
        zeros=lambda shape, **kwargs: Tensor(shape, rows=[None] * shape[2]),
        DRAM_MEMORY_CONFIG=None,
        slice=sliced,
        concat=concat,
        transpose=transpose,
        pad=pad,
        to_memory_config=lambda x, *a, **kw: x,
        full=lambda shape, value, **kw: Tensor(shape, value=value),
        SDPAProgramConfig=lambda **kw: kw,
        PagedCacheGeometryOverride=lambda **kw: kw,
        transformer=NS(chunked_scaled_dot_product_attention=sdpa),
        experimental=NS(paged_fill_cache=fill, paged_update_cache=update),
    )
    monkeypatch.setattr(module, "ttnn", tt)
    monkeypatch.setattr(module, "_make_single_user_cache_update_memory_config", lambda *args: None)
    return calls


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize(
    "kind,modulo", [("full_attention", None), ("sliding_attention", None), ("sliding_attention", 1024)]
)
@pytest.mark.parametrize(
    "start,length", [(0, 2048), (2048, 2048), (8192, 5), (8196, 1), (8196, 5), (8196, 2048), (8196, 8193)]
)
def test_actual_prefill_writes_absolute_rows_and_preserves_ring(monkeypatch, tp, kind, modulo, start, length):
    recording_runtime(monkeypatch)
    owner = module.MultichipDecoder.__new__(module.MultichipDecoder)
    owner.layer_kind = _layer_kind(kind)
    owner.local_full_kv_heads = max(1, 2 // tp)
    owner.local_sliding_kv_heads = 8 // tp
    owner.multichip_prefill_start = start
    owner.mesh_device = None
    block = owner.layer_kind.block_size
    heads = owner.local_full_kv_heads if kind == "full_attention" else owner.local_sliding_kv_heads
    columns = (modulo or (start + length + block - 1)) // block
    pages = list(range(columns - 1, -1, -1))
    table = Tensor((32, columns), table=[pages[:] for _ in range(32)])
    owner.multichip_prefill_page_table = pages
    previous = {}
    for position in range(max(0, start - (modulo or start)), start):
        local = position if modulo is None else position % modulo
        previous[(pages[local // block], local % block)] = position
    caches = [Tensor((columns, heads, block, owner.layer_kind.head_dim), cache=previous.copy()) for _ in range(2)]
    if kind == "sliding_attention" and start:
        history = owner._prefill_sliding_history(*caches, start, modulo)
        assert history[0].rows == list(range(max(0, start - 1023), start))
    physical = (length + 31) // 32 * 32
    values = [
        Tensor(
            (1, heads, physical, owner.layer_kind.head_dim),
            rows=list(range(start, start + length)) + [None] * (physical - length),
        )
        for _ in range(2)
    ]
    owner._fill_prefill_cache(*caches, *values, table, user_id=31, logical_seq_len=length, cache_position_modulo=modulo)
    expected = previous.copy()
    for position in range(start, start + length):
        local = position if modulo is None else position % modulo
        expected[(pages[local // block], local % block)] = position
    assert caches[0].cache == caches[1].cache == expected
    assert None not in caches[0].cache.values(), "Physical padding must never overwrite cache history"


@pytest.mark.parametrize("start,length", [(2048, 2048), (8192, 5), (8196, 5), (8196, 2048), (8196, 8193)])
def test_actual_full_sdpa_preserves_query_absolute_positions_and_shape(monkeypatch, start, length):
    calls = recording_runtime(monkeypatch)
    owner = module.MultichipDecoder.__new__(module.MultichipDecoder)
    owner.layer_kind = _layer_kind("full_attention")
    owner.local_q_heads = 8
    owner.local_full_kv_heads = 1
    owner.correctness_compute_config = None
    owner.mesh_device = NS(compute_with_storage_grid_size=lambda: (8, 8))
    physical = (length + 31) // 32 * 32
    q = Tensor((1, 8, physical, 512), rows=list(range(start, start + length)) + [None] * (physical - length))
    table = Tensor((32, 2048), table=[list(range(2048)) for _ in range(32)])
    output = owner._full_chunked_prefill_attention(
        q, None, None, table, user_id=31, prefill_start=start, logical_seq_len=length
    )
    assert output.shape == q.shape
    assert output.rows[:length] == list(range(start, start + length))
    for offset, rows in calls:
        assert offset % 32 == 0
        assert all(value is None or value == offset + row for row, value in enumerate(rows))


@pytest.mark.parametrize("explicit_metadata", [False, True])
def test_absolute_start_and_current_metadata_reach_actual_model_and_decoder_scope(monkeypatch, explicit_metadata):
    from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator
    from models.autoports.google_gemma_4_26b_a4b_it.tt.model import FullModelState, Gemma4FullModel
    from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder

    table = torch.arange(32 * 256, dtype=torch.int32).reshape(32, 256)
    table[0, 112] = 7

    class DeviceTable:
        shape = table.shape

        def __init__(self):
            self.data = table.clone()

    device_table = DeviceTable()
    downloads = []
    monkeypatch.setattr(module.ttnn, "Tensor", DeviceTable)
    monkeypatch.setattr(module.ttnn, "get_device_tensors", lambda value: [value])

    def download(value):
        downloads.append(value)
        return value.data

    monkeypatch.setattr(module.ttnn, "to_torch", download)
    state = FullModelState(
        kv_cache=[(None, None)],
        page_tables=[device_table],
        cache_specs=[],
        max_batch_size=32,
        slot_context_lengths=[262144] * 32,
        prompt_lens=[0] * 32,
        positions=torch.full((32,), -1),
        active_mask=torch.zeros(32, dtype=torch.bool),
        host_page_tables=[table],
        sliding_cache_position_modulo=None,
    )
    decoder = module.MultichipDecoder.__new__(module.MultichipDecoder)
    decoder.tp_size = 2
    observed = []

    def recorded_decoder(self, hidden, **kwargs):
        observed.append(
            (self.multichip_prefill_start, self.multichip_prefill_page_table, kwargs["cache_position_modulo"])
        )
        return hidden

    monkeypatch.setattr(OptimizedDecoder, "prefill_forward", recorded_decoder)
    monkeypatch.setattr(module.ttnn, "corruptible_allocation_scope", lambda *args: nullcontext())
    model = Gemma4FullModel.__new__(Gemma4FullModel)
    model.max_seq_len = 262144
    model.layer_indices = [0]
    model.layers = [decoder]
    model.hf_config = NS(layer_types=["sliding_attention"])
    model.embed_tokens = lambda tokens: Tensor((1, 1, 5, 2816))
    model._rope_rows = lambda *args: (None, None)
    model._terminal = lambda hidden: hidden
    generator = Gemma4Generator.__new__(Gemma4Generator)
    generator.model = model
    generator.mesh_device = None
    generator._state_from_args = lambda **kwargs: state
    generator._seeded_slots = torch.zeros(32, dtype=torch.bool)
    generator._host_tokens_to_device = lambda tokens: tokens
    generator._positions_to_device = lambda positions, **kwargs: positions
    generator._gather_logits_to_torch = lambda logits, **kwargs: logits
    for physical_page in (7, 99):
        device_table.data[0, 112] = physical_page
        metadata = {"host_page_tables": [device_table.data.clone()]} if explicit_metadata else {}
        generator.prefill_forward(
            torch.zeros((1, 5), dtype=torch.int32),
            page_table=state.page_tables,
            kv_cache=state,
            prompt_lens=[5],
            start_pos=torch.arange(8196, 8201),
            return_all_logits=True,
            **metadata,
        )
    assert observed[0][0] == 8196
    assert [int(item[1][112]) for item in observed] == [7, 99]
    assert len(downloads) == (0 if explicit_metadata else 2)
    assert observed[0][2] is None
    assert decoder.multichip_prefill_start == 0
    assert decoder.multichip_prefill_page_table is None
    assert decoder.multichip_execution_phase == "idle"


def test_replacing_device_table_invalidates_host_metadata(monkeypatch):
    from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator
    from models.autoports.google_gemma_4_26b_a4b_it.tt.model import FullModelState

    old = torch.zeros((32, 256), dtype=torch.int32)
    new = torch.ones_like(old)
    state = FullModelState(
        kv_cache=[(None, None)],
        page_tables=[old],
        cache_specs=[],
        max_batch_size=32,
        slot_context_lengths=[262144] * 32,
        prompt_lens=[0] * 32,
        positions=torch.full((32,), -1),
        active_mask=torch.zeros(32, dtype=torch.bool),
        host_page_tables=[old.clone()],
    )
    generator = Gemma4Generator.__new__(Gemma4Generator)
    generator.model = NS(num_layers=1)
    generator.mesh_device = None
    generator._page_table_sources = {}
    generator.trace_counters = NS(page_table_refreshes=0)
    generator._refresh_device_input = lambda target, source: target.copy_(source)
    monkeypatch.setattr(module.ttnn, "corruptible_allocation_scope", lambda *args: nullcontext())
    result = generator._state_from_args(kv_cache=state, page_table=[new], batch_size=1)
    assert result is state
    assert torch.equal(state.page_tables[0], new)
    assert state.host_page_tables is None


def test_adapter_passes_authoritative_current_host_metadata(monkeypatch):
    from models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm import Gemma4ForCausalLM

    current = [torch.zeros((1, 256), dtype=torch.int32)]
    observed = []
    generator = NS(model=NS(num_layers=1), _trace_cache={})
    generator.prefill_forward = lambda *args, **kwargs: observed.append(kwargs)
    adapter = Gemma4ForCausalLM()
    adapter._last_page_tables = current
    adapter._serving_state = NS(kv_cache=object(), page_tables=[object()])
    monkeypatch.setattr(adapter, "_require_generator", lambda: generator)
    monkeypatch.setattr(adapter, "_page_tables_changed", lambda _: False)
    monkeypatch.setattr(adapter, "_host_prefill_logits", lambda *args: torch.zeros((1, 1, 1)))
    adapter.prefill_forward(
        torch.zeros((1, 8201), dtype=torch.int32),
        page_table=current[0],
        kv_cache=adapter._serving_state.kv_cache,
        prompt_lens=[8201],
        start_pos=torch.tensor([[8196]]),
    )
    assert observed[0]["host_page_tables"] is current
