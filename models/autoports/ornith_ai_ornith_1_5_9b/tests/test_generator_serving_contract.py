# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of actual serving orchestration; never import or open TTNN."""

import ast
import copy
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_class(filename, name, namespace, methods=None):
    tree = ast.parse((ROOT / "tt" / filename).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    if methods is not None:
        cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    exec(compile(ast.Module(body=[cls], type_ignores=[]), filename, "exec"), namespace)
    return namespace[name]


@pytest.fixture
def tt():
    def select(mask, a, b, output_tensor=None):
        result = torch.where(mask.bool(), a, b)
        return result if output_tensor is None else output_tensor.copy_(result)

    return SimpleNamespace(
        int32=torch.int32,
        uint32=torch.int32,
        float32=torch.float32,
        bfloat16=torch.bfloat16,
        TILE_LAYOUT="tile",
        ROW_MAJOR_LAYOUT="row_major",
        clone=lambda value: value.clone(),
        copy=lambda value, target: target.copy_(value),
        deallocate=lambda value: None,
        reshape=lambda value, shape: value.reshape(shape),
        to_layout=lambda value, layout: value.clone(),
        slice=lambda value, begins, ends: value[tuple(slice(a, b) for a, b in zip(begins, ends))].clone(),
        repeat=lambda value, shape: value.repeat(shape),
        Shape=tuple,
        where=select,
        gather=lambda value, dim, index: value.gather(dim, index.long()),
        copy_host_to_device_tensor=lambda value, target: target.copy_(value),
    )


@pytest.fixture
def gen(tt):
    namespace = {"torch": torch, "ttnn": tt, "Generator": object}
    cls = load_class("generator.py", "OrnithGenerator", namespace)
    obj = cls.__new__(cls)
    obj.max_batch_size = 3
    obj.mesh_device = object()
    obj.counters = defaultdict(int)
    obj.sampling_mode = "device"
    obj._inputs = [
        torch.arange(100, 132, dtype=torch.int32).reshape(1, 1, 1, 32),
        torch.tensor([1000, 2000, 3000], dtype=torch.int32),
        torch.tensor([[1000, 2000, 3000]], dtype=torch.int32),
        torch.arange(6, dtype=torch.int32).reshape(3, 2),
    ]
    obj.model = SimpleNamespace(upload=lambda value, **kw: value.to(kw.get("dtype", value.dtype)).clone())
    obj.kv_cache = SimpleNamespace(
        active_recurrent=torch.ones(3, 1, 1, 1), active_conv=torch.ones(3, 1, 1), decode_layers=[]
    )
    obj._seed_values = torch.arange(32, dtype=torch.int32)
    obj._configured_seeds = list(range(32))
    obj.sampling = SimpleNamespace(
        seed_manager=SimpleNamespace(apply_slot_remap=lambda remap: None),
        tt_sampling=SimpleNamespace(seeds_tt_tensor=torch.arange(2**30, 2**30 + 32, dtype=torch.int32)),
        tt_penalties=SimpleNamespace(
            prompt_mask=torch.arange(128, dtype=torch.int32).reshape(32, 4),
            output_mask=torch.arange(128, dtype=torch.int32).reshape(32, 4) + 1000,
            output_counts=torch.arange(128, dtype=torch.int32).reshape(32, 4) + 2000,
            output_counts_gathered=torch.arange(128, dtype=torch.int32).reshape(32, 4) + 3000,
            _prompt_tokens_host=torch.arange(64).reshape(32, 2),
        ),
    )
    return obj


def test_explicit_pool_keeps_logical_context_and_exact_physical_blocks(tt):
    namespace = {"copy": copy, "torch": torch, "ttnn": tt, "dataclass": dataclass}
    namespace["DEFAULT_PAGE_BLOCK_SIZE"] = 64
    tree = ast.parse((ROOT / "tt/functional_decoder.py").read_text())
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("num_blocks_for_context", "_align_up")
    ]
    exec(compile(ast.Module(body=helpers, type_ignores=[]), "functional_decoder.py", "exec"), namespace)
    namespace["ModelCache"] = load_class("model.py", "ModelCache", namespace)
    cls = load_class("model.py", "OrnithModel", namespace, {"allocate_cache", "page_table"})
    model = cls.__new__(cls)
    model.max_context, model.page_block_size, model.dim = 262144, 64, 4096
    model.precision = {"kv_cache_dtype": "bfloat16", "residual_dtype": "bfloat16"}
    allocated = []
    model.layers = [
        SimpleNamespace(
            is_full_attention=True,
            allocate_state=lambda batch: None,
            allocate_kv_cache=lambda blocks, **kw: allocated.append(blocks) or (blocks, blocks),
        )
    ]
    model.upload = lambda value, **kw: value
    shared = model.allocate_cache(3, num_blocks=80)
    assert shared.context == 262144
    assert shared.num_blocks == 80
    assert allocated == [80]
    with pytest.raises(ValueError, match="page table"):  # allow-pytest.raises: CPU-only error.
        model.page_table(shared)
    standalone = model.allocate_cache(3, context=128)
    assert allocated == [80, 96]
    assert torch.equal(model.page_table(standalone), torch.arange(96, dtype=torch.int32).reshape(3, 32))
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="num_blocks"):  # allow-pytest.raises: CPU-only error.
            model.allocate_cache(3, num_blocks=bad)


def test_host_step_skips_sampling_without_replacing_traces(gen, tt):
    calls = []
    tt.execute_trace = lambda mesh, trace, **kw: calls.append((trace, kw["blocking"]))
    gen._model_trace, gen._sampling_trace, gen._sampling_history_trace = 10, 11, 12
    gen._history_rows = 0
    gen._replay(sample_on_device=False)
    assert calls == [(10, False)]
    assert gen._sampling_trace == 11
    gen._replay()
    assert calls == [(10, False), (10, False), (11, False)]
    assert gen.counters["sampling_replays"] == 1
    with pytest.raises(ValueError, match="sampl"):  # allow-pytest.raises: CPU-only error.
        gen._replay(sample_on_device=False, collect_output=True)


def test_slot_cycle_preserves_exact_values_and_stable_buffers(gen):
    recurrent = torch.arange(24, dtype=torch.float32).reshape(3, 2, 2, 2)
    recurrent[1, 0, 0, 0] = float("nan")
    conv = torch.arange(12, dtype=torch.bfloat16).reshape(3, 1, 4)
    gen.kv_cache.decode_layers = [
        SimpleNamespace(is_full_attention=False, recurrent_state=recurrent, conv_state=[conv]),
        SimpleNamespace(is_full_attention=True),
    ]
    buffers = [recurrent, conv] + gen._inputs[:3] + [gen.sampling.tt_sampling.seeds_tt_tensor]
    buffers += [
        getattr(gen.sampling.tt_penalties, name)
        for name in ("prompt_mask", "output_mask", "output_counts", "output_counts_gathered")
    ]
    before = [value.clone() for value in buffers]
    addresses = [value.data_ptr() for value in buffers]
    remap = torch.tensor([2, 0, 1])
    assert gen.remap_serving_slots(remap) == 1
    for index, (value, old) in enumerate(zip(buffers, before)):
        if index in (2, 3, 4, 5):
            expected = old.reshape(-1).clone()
            expected[:3] = old.reshape(-1)[remap]
            expected = expected.reshape(old.shape)
        else:
            expected = old.clone()
            expected[:3] = old[remap]
        torch.testing.assert_close(value, expected, rtol=0, atol=0, equal_nan=True)
    assert addresses == [value.data_ptr() for value in buffers]
    assert gen._configured_seeds[:3] == [2, 0, 1]
    assert gen._seed_values[:3].tolist() == [2, 0, 1]
    assert gen.sampling.tt_penalties._prompt_tokens_host[:3].tolist() == [[4, 5], [0, 1], [2, 3]]
    assert gen.remap_serving_slots([0, 1, 2]) == 0
    for bad in ([0, 0, 2], [0, 1], [0, 1, 3], [0.1, 1, 2]):
        with pytest.raises(ValueError, match="permutation"):  # allow-pytest.raises: CPU-only error.
            gen.remap_serving_slots(bad)


def test_vocabulary_history_remap_avoids_full_row_repeat(gen, tt):
    # The real reduced serving run exceeded 1.5 MiB L1 in repeat_upper_dims_rm
    # while moving common-sampler [32, 65536] INT32 penalty history rows.
    value = torch.arange(32, dtype=torch.int32)[:, None] * 1_000_000
    value = value + torch.arange(65536, dtype=torch.int32)[None, :]
    target = value.clone()
    tt.repeat = lambda *args, **kwargs: pytest.fail("full-width penalty repeat exceeds the hardware L1 budget")
    remap = [2, 0, 1] + list(range(3, 32))
    gen._remap_serving_rows(target, remap)
    assert torch.equal(target, value[remap])


def test_boundary_refresh_preserves_device_feedback_and_marks_idle_rows(gen):
    # Host scheduler values for row 0 lag the device; only row 1 is newly admitted.
    gen.refresh_serving_inputs([9, 250000, 8], [999, 7, -1], [False, True, False])
    assert gen._inputs[0].flatten()[:3].tolist() == [100, 250000, 102]
    assert gen._inputs[1].tolist() == [1000, 7, -1]
    assert gen._inputs[2].flatten().tolist() == [1000, 7, 3000]
    assert gen.kv_cache.active_recurrent.flatten().tolist() == [1, 1, 0]
    assert gen.kv_cache.active_conv.flatten().tolist() == [1, 1, 0]
    assert gen.counters["readbacks"] == 0
    with pytest.raises(ValueError, match="fixed slot"):  # allow-pytest.raises: CPU-only error.
        gen.refresh_serving_inputs([1], [1], [True])


def test_async_token_copy_snapshots_first_shard_before_replay(gen, tt):
    calls = []

    class Shard:
        def cpu(self, *, blocking):
            calls.append(("copy", blocking))
            return gen._inputs[0].clone()

    tt.get_device_tensors = lambda tensor: [Shard()] if tensor is gen._inputs[0] else [tensor]
    tt.record_event = lambda mesh, cq: calls.append(("event", cq)) or "event"
    tt.to_torch = lambda tensor, **kw: tensor
    pending, event = gen.read_output_async()
    gen._inputs[0].fill_(999)
    assert gen.tokens_from(pending).tolist() == [100, 101, 102]
    assert calls == [("copy", False), ("event", 0)]
    assert event == "event"


def test_async_logits_copy_keeps_all_shards_and_trims_vocabulary(gen, tt):
    values = torch.arange(24).reshape(1, 1, 3, 8)
    gen.model.vocab_size = 5

    class Logits:
        def cpu(self, *, blocking):
            assert blocking is False
            return values.clone()

    tt.get_device_tensors = lambda tensor: pytest.fail("Host logits must copy every vocabulary shard")
    tt.record_event = lambda mesh, cq: "event"
    tt.ConcatMeshToTensor = lambda mesh, dim: (mesh, dim)
    tt.to_torch = lambda tensor, **kw: tensor
    pending, _ = gen.read_output_async(Logits(), return_logits=True)
    assert torch.equal(gen.logits_from(pending), values[0, 0, :, :5].float())


@pytest.mark.parametrize("preserve_cache", [True, False])
def test_capture_can_skip_snapshot_only_for_explicit_empty_pool(gen, tt, preserve_cache):
    saved = torch.tensor([7.0])
    snapshots = []
    gen.owns_cache, gen._live, gen._model_trace = False, False, None
    gen._prefill_inputs, gen.page_table = None, None
    gen.sampling_mode = "host"
    gen.model.cache_buffers = lambda cache: snapshots.append(True) or [saved]
    gen.model.reset_cache = lambda cache, **kw: saved.zero_()
    gen._write_tokens = gen._write_positions = gen._refresh_table = lambda values: None
    gen._forward = lambda: torch.zeros(1)
    gen._capture = lambda: None
    tt.synchronize_device = lambda mesh: None
    gen.ensure_traces(preserve_cache=preserve_cache)
    assert snapshots == ([True] if preserve_cache else [])
    assert saved.item() == (7 if preserve_cache else 0)
