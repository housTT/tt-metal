# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only contracts for physical-B1 virtual decode state."""

from __future__ import annotations

import random
from types import SimpleNamespace

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder as multichip_module
from models.autoports.qwen_qwen3_8_flash_next.tt.model import Qwen38FullModel
from models.autoports.qwen_qwen3_8_flash_next.tt.model_config import LINEAR_ATTENTION
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MultichipVirtualDecodeStateBank


class _FakeTensor:
    def __init__(self, value, *, dtype=ttnn.float32, shape=(1,)):
        self.value = value
        self.dtype = dtype
        self.shape = tuple(shape)
        self.padded_shape = tuple(shape)
        self._allocated = True

    def get_layout(self):
        return ttnn.TILE_LAYOUT

    def is_allocated(self):
        return self._allocated


def _install_fake_tensor_ops(monkeypatch):
    zero_allocations = []

    def zeros_like(tensor):
        result = _FakeTensor(0, dtype=tensor.dtype, shape=tensor.shape)
        zero_allocations.append(result)
        return result

    monkeypatch.setattr(
        multichip_module.ttnn,
        "zeros_like",
        zeros_like,
    )
    monkeypatch.setattr(
        multichip_module.ttnn,
        "copy",
        lambda source, target: setattr(target, "value", source.value),
    )
    monkeypatch.setattr(
        multichip_module.ttnn,
        "deallocate",
        lambda tensor: setattr(tensor, "_allocated", False),
    )
    return zero_allocations


def test_device_bank_copies_only_request_local_state(monkeypatch):
    zero_allocations = _install_fake_tensor_ops(monkeypatch)
    recurrence = _FakeTensor(11, shape=(1, 2))
    conv = _FakeTensor(12)
    ple = _FakeTensor(13, dtype=ttnn.bfloat16)
    gdn = SimpleNamespace(
        shapes=SimpleNamespace(layer_type=LINEAR_ATTENTION, has_ple=True),
        recurrent_state=recurrence,
        fused_conv_state=(conv,),
        fused_ple_conv_state=(ple,),
        host_expert_cache=object(),
    )
    qsa_kv = _FakeTensor(999)
    qsa = SimpleNamespace(
        shapes=SimpleNamespace(layer_type="qwen_sparse_attention", has_ple=False),
        kv_cache=(qsa_kv,),
    )
    token = _FakeTensor(21, dtype=ttnn.uint32)
    position = _FakeTensor(22, dtype=ttnn.int32)
    page = _FakeTensor(23, dtype=ttnn.int32)
    bank = MultichipVirtualDecodeStateBank(
        object(),
        (gdn, qsa),
        capacity=2,
        io_tensors={"token": token, "current_pos": position, "page_table": page},
    )

    bank.commit_slot(0)
    for tensor, value in (
        (recurrence, 31),
        (conv, 32),
        (ple, 33),
        (token, 41),
        (position, 42),
        (page, 43),
    ):
        tensor.value = value
    bank.commit_slot(1)
    qsa_kv.value = 1000
    bank.restore_slot(0)

    assert (recurrence.value, conv.value, ple.value) == (11, 12, 13)
    assert (token.value, position.value, page.value) == (21, 22, 23)
    assert qsa_kv.value == 1000
    assert bank.slot_tensor(1, "token").value == 41
    metrics = bank.metrics()
    assert metrics["commits"] == 2
    assert metrics["restores"] == 1
    assert metrics["commit_logical_bytes"] == 2 * metrics["logical_bytes_per_slot"]
    allocations_before_reset = len(zero_allocations)
    bank.reset_slot(1)
    assert len(zero_allocations) == allocations_before_reset
    assert bank.slot_tensor(1, "token").value == 0
    assert bank.metrics()["allocated_logical_bytes"] == 3 * metrics["logical_bytes_per_slot"]
    bank.close()
    assert metrics["closed"] is False and bank.metrics()["closed"] is True
    assert all(not tensor.is_allocated() for tensor in zero_allocations)


class _FakeBank:
    def __init__(self):
        self.commits = []
        self.restores = []
        self.resets = []
        self.events = []
        self.physical_value = None
        self.slot_values = {0: None, 1: None}
        self.tokens = {0: object(), 1: object()}

    def commit_slot(self, slot):
        self.commits.append(slot)
        self.events.append(("commit", slot))
        self.slot_values[slot] = self.physical_value

    def restore_slot(self, slot):
        self.restores.append(slot)
        self.events.append(("restore", slot))
        self.physical_value = self.slot_values[slot]

    def reset_slot(self, slot):
        self.resets.append(slot)
        self.events.append(("reset", slot))
        self.slot_values[slot] = None

    def slot_tensor(self, slot, name):
        assert name == "token"
        return self.tokens[slot]

    def metrics(self):
        return {
            "enabled": True,
            "capacity": 2,
            "restores": len(self.restores),
            "commits": len(self.commits),
            "resets": len(self.resets),
        }


def _fake_model():
    model = object.__new__(Qwen38FullModel)
    model.max_batch = 1
    model.virtual_slot_capacity = 2
    model.virtual_decode_state_bank = _FakeBank()
    model._virtual_slot_owners = [None, None]
    model._virtual_slot_generations = [0, 0]
    model._virtual_slot_valid = [False, False]
    model._virtual_slot_banked = [False, False]
    model._virtual_slot_sampling = [None, None]
    model._virtual_slot_state_hosts = [None, None]
    model._virtual_resident_slot = None
    model._virtual_resident_committed = True
    model._virtual_assignments = 0
    model._virtual_releases = 0
    model._virtual_stale_rejections = 0
    model._virtual_prefill_admissions_while_trace_live = 0
    model._virtual_prefill_trace_invalidations = 0
    model._virtual_sampling_trace_mode_switches = 0
    model._trace_ready = False
    model._trace_execution_mode = None
    model._trace_sampling_force_argmax = None
    model._sampling_force_argmax = False
    model._sampling_seed_rngs = (random.Random(17),)
    model.decode_token_input = object()
    ple_histories = {"old": torch.tensor([1]), "b": torch.tensor([2])}
    model.ple_store = SimpleNamespace(
        resets=[],
        cancellations=[],
        histories=ple_histories,
        reset_request=lambda request: model.ple_store.resets.append(request),
        cancel_request=lambda request: (
            model.ple_store.cancellations.append(request),
            model.ple_store.histories.pop(request, None),
        ),
    )
    model._virtual_physical_state = SimpleNamespace(
        page_table_host=torch.tensor([[7, 8]], dtype=torch.int32),
        prompt_lens=torch.tensor([67], dtype=torch.int32),
        active_mask=torch.tensor([True]),
        request_ids=("a",),
    )
    return model


def test_single_live_slot_bypasses_bank_then_second_slot_materializes_state():
    model = _fake_model()
    a = model.assign_virtual_slot(0, "a")
    model.begin_virtual_prefill(0, "a", generation=a.generation)
    model.finish_virtual_prefill(0, "a", generation=a.generation)
    model.activate_virtual_slot(0, "a", generation=a.generation)
    model.commit_virtual_slot(0, "a", generation=a.generation)
    assert model.virtual_decode_state_bank.commits == []
    assert model.virtual_decode_state_bank.restores == []
    assert model.virtual_slot_token(0, "a", generation=a.generation) is model.decode_token_input

    b = model.assign_virtual_slot(1, "b")
    model._trace_ready = True
    model.begin_virtual_prefill(1, "b", generation=b.generation)
    # Transitioning from direct B1 to multi-active snapshots A exactly once.
    assert model.virtual_decode_state_bank.commits == [0]
    assert model._trace_ready is True
    assert model.virtual_slot_metrics()["prefill_admissions_while_trace_live"] == 1
    model._virtual_physical_state.request_ids = ("b",)
    model._virtual_physical_state.prompt_lens = torch.tensor([129], dtype=torch.int32)
    model._virtual_physical_state.page_table_host = torch.tensor([[9, 10]], dtype=torch.int32)
    model._sampling_seed_rngs = (random.Random(99),)
    model.finish_virtual_prefill(1, "b", generation=b.generation)
    assert model.virtual_decode_state_bank.commits == [0, 1]

    model.activate_virtual_slot(0, "a", generation=a.generation)
    assert model.virtual_decode_state_bank.restores == [0]
    assert model._virtual_physical_state.request_ids == ("a",)
    assert model._virtual_physical_state.page_table_host.tolist() == [[7, 8]]
    assert model._sampling_seed_rngs[0].randint(1, 1000) == random.Random(17).randint(1, 1000)
    model.commit_virtual_slot(0, "a", generation=a.generation)
    assert model.virtual_slot_token(0, "a", generation=a.generation) is model.virtual_decode_state_bank.tokens[0]

    # Dropping the other user returns A to direct mode. Its next update makes
    # the old bank snapshot stale; admitting C must materialize A again.
    model.release_virtual_slot(1, "b", generation=b.generation)
    assert model.virtual_decode_state_bank.resets == [1]
    model.activate_virtual_slot(0, "a", generation=a.generation)
    model.commit_virtual_slot(0, "a", generation=a.generation)
    assert model._virtual_slot_banked[0] is False
    c = model.assign_virtual_slot(1, "c")
    commits_before_c = len(model.virtual_decode_state_bank.commits)
    model.begin_virtual_prefill(1, "c", generation=c.generation)
    assert len(model.virtual_decode_state_bank.commits) == commits_before_c + 1
    assert model.virtual_decode_state_bank.commits[-1] == 0


def test_generation_guard_and_request_local_release(expect_error):
    model = _fake_model()
    lease = model.assign_virtual_slot(0, "old")
    model.begin_virtual_prefill(0, "old", generation=lease.generation)
    model.finish_virtual_prefill(0, "old", generation=lease.generation)
    model.release_virtual_slot(0, "old", generation=lease.generation)
    replacement = model.assign_virtual_slot(0, "new")
    assert replacement.generation > lease.generation
    assert model.ple_store.cancellations == ["old"]
    assert "old" not in model.ple_store.histories
    with expect_error(RuntimeError, "stale virtual slot lease"):
        model.activate_virtual_slot(0, "old", generation=lease.generation)
    assert model.virtual_slot_metrics()["stale_rejections"] == 1


def test_unbanked_release_skips_device_reset_but_cleans_request_lifecycle(expect_error):
    model = _fake_model()
    model.ple_store.histories["single-user"] = torch.tensor([3, 5])
    lease = model.assign_virtual_slot(0, "single-user")
    model.begin_virtual_prefill(0, lease.request_id, generation=lease.generation)
    model.finish_virtual_prefill(0, lease.request_id, generation=lease.generation)
    model.commit_virtual_slot(0, lease.request_id, generation=lease.generation)

    # The physical-B1 fast path is authoritative and has never materialized a
    # snapshot in slot zero.  Releasing it must not enqueue a full zero tree.
    assert model._virtual_slot_banked[0] is False
    assert model.virtual_slot_metrics()["bank"]["commits"] == 0
    model.release_virtual_slot(0, lease.request_id, generation=lease.generation)

    metrics = model.virtual_slot_metrics()
    assert metrics["active_slots"] == 0
    assert metrics["valid_slots"] == 0
    assert metrics["resident_slot"] is None
    assert metrics["releases"] == 1
    assert metrics["bank"]["resets"] == 0
    assert model.virtual_decode_state_bank.resets == []
    assert model.ple_store.cancellations == ["single-user"]
    assert "single-user" not in model.ple_store.histories
    assert model._virtual_slot_sampling[0] is None
    assert model._virtual_slot_state_hosts[0] is None

    replacement = model.assign_virtual_slot(0, "replacement")
    assert replacement.generation > lease.generation
    model.begin_virtual_prefill(0, replacement.request_id, generation=replacement.generation)
    model.finish_virtual_prefill(0, replacement.request_id, generation=replacement.generation)
    assert (
        model.virtual_slot_token(
            0,
            replacement.request_id,
            generation=replacement.generation,
        )
        is model.decode_token_input
    )
    assert model.virtual_decode_state_bank.restores == []
    with expect_error(RuntimeError, "stale virtual slot lease"):
        model.activate_virtual_slot(0, lease.request_id, generation=lease.generation)


def test_unbanked_in_place_reset_skips_device_reset_and_invalidates_state():
    model = _fake_model()
    model.ple_store.histories["retry"] = torch.tensor([8, 13])
    lease = model.assign_virtual_slot(0, "retry")
    model.begin_virtual_prefill(0, lease.request_id, generation=lease.generation)
    model.finish_virtual_prefill(0, lease.request_id, generation=lease.generation)
    model.commit_virtual_slot(0, lease.request_id, generation=lease.generation)

    replacement = model.reset_virtual_slot(0, lease.request_id, generation=lease.generation)

    assert replacement.request_id == lease.request_id
    assert replacement.generation > lease.generation
    assert model.virtual_decode_state_bank.resets == []
    assert model.virtual_slot_metrics()["bank"]["resets"] == 0
    assert model._virtual_slot_owners[0] == "retry"
    assert model._virtual_slot_valid[0] is False
    assert model._virtual_slot_banked[0] is False
    assert model._virtual_resident_slot == 0
    assert model._virtual_resident_committed is False
    assert model.ple_store.cancellations == ["retry"]
    assert "retry" not in model.ple_store.histories


def test_reused_unbanked_slot_is_fully_overwritten_before_restore():
    model = _fake_model()
    bank = model.virtual_decode_state_bank
    a = model.assign_virtual_slot(0, "a")
    model.begin_virtual_prefill(0, a.request_id, generation=a.generation)
    bank.physical_value = "a-prefill"
    model.finish_virtual_prefill(0, a.request_id, generation=a.generation)

    b = model.assign_virtual_slot(1, "b")
    model.begin_virtual_prefill(1, b.request_id, generation=b.generation)
    bank.physical_value = "b-prefill"
    model.finish_virtual_prefill(1, b.request_id, generation=b.generation)
    model.activate_virtual_slot(0, a.request_id, generation=a.generation)
    assert bank.physical_value == "a-prefill"
    bank.physical_value = "a-multi-decode"
    model.commit_virtual_slot(0, a.request_id, generation=a.generation)

    # Once B is released, A returns to direct physical-B1 mode. Its bank copy
    # becomes stale and is deliberately not reset when A itself is released.
    model.release_virtual_slot(1, b.request_id, generation=b.generation)
    bank.physical_value = "a-direct-decode"
    model.commit_virtual_slot(0, a.request_id, generation=a.generation)
    assert model._virtual_slot_banked[0] is False
    model.release_virtual_slot(0, a.request_id, generation=a.generation)
    assert bank.resets == [1]
    assert bank.slot_values[0] == "a-multi-decode"

    c = model.assign_virtual_slot(0, "c")
    model.begin_virtual_prefill(0, c.request_id, generation=c.generation)
    bank.physical_value = "c-prefill"
    model.finish_virtual_prefill(0, c.request_id, generation=c.generation)
    d = model.assign_virtual_slot(1, "d")
    model.begin_virtual_prefill(1, d.request_id, generation=d.generation)
    assert bank.slot_values[0] == "c-prefill"
    bank.physical_value = "d-prefill"
    model.finish_virtual_prefill(1, d.request_id, generation=d.generation)
    model.activate_virtual_slot(0, c.request_id, generation=c.generation)

    assert bank.physical_value == "c-prefill"
    assert bank.events[-3:] == [("commit", 0), ("commit", 1), ("restore", 0)]
    assert model.ple_store.cancellations[-2:] == ["b", "a"]


def test_mixed_sampler_slot_activation_invalidates_incompatible_trace():
    model = _fake_model()
    releases = []

    def release_trace():
        releases.append(model._trace_sampling_force_argmax)
        model._trace_ready = False
        model._trace_sampling_force_argmax = None

    model.release_decode_traces = release_trace
    a = model.assign_virtual_slot(0, "greedy")
    model._sampling_force_argmax = True
    model._sampling_seed_rngs = None
    model.begin_virtual_prefill(0, "greedy", generation=a.generation)
    model.finish_virtual_prefill(0, "greedy", generation=a.generation)
    b = model.assign_virtual_slot(1, "random")
    model.begin_virtual_prefill(1, "random", generation=b.generation)
    model._sampling_force_argmax = False
    model._sampling_seed_rngs = (random.Random(23),)
    model.finish_virtual_prefill(1, "random", generation=b.generation)

    model._trace_ready = True
    model._trace_execution_mode = "token_out"
    model._trace_sampling_force_argmax = False
    model.activate_virtual_slot(0, "greedy", generation=a.generation)
    assert releases == [False]
    assert model._sampling_force_argmax is True

    model.commit_virtual_slot(0, "greedy", generation=a.generation)
    model._trace_ready = True
    model._trace_sampling_force_argmax = True
    model.activate_virtual_slot(1, "random", generation=b.generation)
    assert releases == [False, True]
    assert model._sampling_force_argmax is False
    assert model.virtual_slot_metrics()["sampling_trace_mode_switches"] == 2


def test_mixed_sampler_slot_activation_keeps_model_only_trace():
    model = _fake_model()
    releases = []
    model.release_decode_traces = lambda: releases.append(True)
    lease = model.assign_virtual_slot(0, "greedy")
    model._sampling_force_argmax = True
    model.begin_virtual_prefill(0, lease.request_id, generation=lease.generation)
    model.finish_virtual_prefill(0, lease.request_id, generation=lease.generation)

    model._sampling_force_argmax = False
    model._trace_ready = True
    model._trace_execution_mode = "model_only"
    model._trace_sampling_force_argmax = False
    model._restore_virtual_host_metadata(0)

    assert releases == []
    assert model._sampling_force_argmax is True
    assert model.virtual_slot_metrics()["sampling_trace_mode_switches"] == 0


def test_export_restore_virtual_rng_preserves_exact_next_draw():
    model = _fake_model()
    lease = model.assign_virtual_slot(0, "seeded-or-entropy-seeded")
    model._sampling_seed_rngs = (random.Random(7341),)
    model.begin_virtual_prefill(0, lease.request_id, generation=lease.generation)
    model.finish_virtual_prefill(0, lease.request_id, generation=lease.generation)

    continuation = model.export_virtual_slot_sampling_rng_state(0, lease.request_id, generation=lease.generation)
    expected = random.Random()
    expected.setstate(continuation[0])
    expected_next = expected.randint(1, 0x7FFFFFFE)

    model._sampling_seed_rngs = (random.Random(1),)
    model.restore_virtual_slot_sampling_rng_state(0, lease.request_id, continuation, generation=lease.generation)
    assert model._sampling_seed_rngs[0].randint(1, 0x7FFFFFFE) == expected_next
