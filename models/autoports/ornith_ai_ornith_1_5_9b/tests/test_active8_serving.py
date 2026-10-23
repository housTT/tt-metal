# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Actual generator/adapter source with CPU-only boundaries; no TTNN import."""
import ast
import copy
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HELPERS = runpy.run_path(str(Path(__file__).with_name("test_generator_serving_contract.py")))
ROOT = Path(__file__).resolve().parents[1]
PREFIX = (True,) * 8 + (False,) * 24


@pytest.fixture
def case():
    tt = HELPERS["tt"].__wrapped__()
    gen = HELPERS["gen"].__wrapped__(tt)
    gen.max_batch_size = 32
    gen._inputs[1] = torch.zeros(32, dtype=torch.int32)
    gen._inputs[2] = torch.zeros(1, 32, dtype=torch.int32)
    gen.kv_cache.active_recurrent = torch.ones(32, 1, 1, 1)
    gen.kv_cache.active_conv = torch.ones(32, 1, 1)
    gen.selected_decode_variant.__func__.__globals__["PREFIX"] = PREFIX
    gen._active8_enabled, gen._active8_mode = True, "active8"
    gen._active8 = SimpleNamespace(cache=gen.kv_cache, owners=[], proxies={})
    return gen, tt


@pytest.mark.parametrize("invalid", [-0.5, 2**32, -(2**32)])
def test_invalid_positions_cannot_select_prefix_or_write_device(case, invalid, expect_error):
    gen, _ = case
    gen._active_slots = PREFIX
    before = [value.clone() for value in gen._inputs]
    values = [0] * 8 + [-1] * 24
    values[8] = invalid
    with expect_error(ValueError, "INT32"):
        gen._write_positions(values)
    assert gen._active_slots is None
    assert gen.selected_decode_variant() == "family_b32"
    assert all(torch.equal(a, b) for a, b in zip(before, gen._inputs))


def test_same_normalized_activity_drives_masks_cancel_and_remap(case, expect_error):
    gen, _ = case
    gen._write_positions([32768] * 8 + [-1] * 24)
    assert gen._active_slots == PREFIX and gen.selected_decode_variant() == "active8"
    assert tuple(gen.kv_cache.active_recurrent.flatten().bool().tolist()) == PREFIX
    gen.refresh_serving_inputs([0] * 32, [32769] * 7 + [-1] * 25, [False] * 32)
    assert gen.selected_decode_variant() == "family_b32"
    gen.refresh_serving_inputs([0] * 32, [32769] * 8 + [-1] * 24, [False] * 7 + [True] + [False] * 24)
    assert gen.selected_decode_variant() == "active8"
    swap = list(range(32))
    swap[7], swap[8] = swap[8], swap[7]
    gen.remap_serving_slots(swap)
    assert gen.selected_decode_variant() == "family_b32"
    gen.remap_serving_slots(swap)
    assert gen.selected_decode_variant() == "active8"
    with expect_error(ValueError, "Serving inputs require integer"):
        gen.refresh_serving_inputs([0] * 32, [0.5] * 32, [False] * 32)
    assert gen._active_slots is None


def trace_fixture(case):
    gen, tt = case
    calls = []
    next_id = iter(range(10, 30))
    tt.begin_trace_capture = lambda *a, **k: next(next_id)
    tt.end_trace_capture = lambda mesh, trace, **kw: calls.append(("end", trace))
    tt.release_trace = lambda mesh, trace: calls.append(("release", trace))
    tt.execute_trace = lambda mesh, trace, **kw: calls.append(("replay", trace))
    gen.mesh_device = SimpleNamespace(num_program_cache_entries=lambda: 7)
    gen.sampling._trace_states = {}
    gen._model_trace = gen._sampling_trace = gen._sampling_history_trace = gen._prefill_trace = None
    gen._prefill_inputs = None
    gen._logits = torch.tensor([99.0])
    gen._forward = lambda **kw: calls.append(("forward", kw.get("active8", False))) or torch.tensor(
        [7.0 if kw.get("active8") else 5.0]
    )
    gen._sample_device = lambda logits: calls.append(("sample", id(logits)))
    gen._append_output_history = lambda: calls.append(("history",))
    gen._history_rows = 0
    return gen, tt, calls


def test_two_resident_models_bind_one_canonical_sampler_and_route_without_capture(case):
    gen, _, calls = trace_fixture(case)
    canonical = gen._logits
    gen._capture()
    assert gen._logits is canonical
    assert [row for row in calls if row[0] == "forward"] == [("forward", False), ("forward", True)]
    assert [row for row in calls if row[0] == "sample"] == [("sample", id(canonical))] * 2
    captures = gen.counters["captures"]
    gen._active_slots = PREFIX
    gen._replay()
    assert gen._last_decode_variant == "active8"
    gen._active_slots = (True,) * 7 + (False,) * 25
    gen._replay(sample_on_device=False)
    assert gen._last_decode_variant == "family_b32"
    assert gen.counters["active8_replays"] == gen.counters["family_b32_replays"] == 1
    assert gen.counters["captures"] == captures


def test_original_mode_captures_without_new_output_copy(case):
    gen, tt, calls = trace_fixture(case)
    gen._active8_enabled = False
    gen._active8_mode = "original"

    def forbidden(*args):
        raise AssertionError("Original graph acquired a canonical logits copy")

    tt.copy = forbidden
    gen._capture()
    assert gen._active8_trace is None
    assert [row for row in calls if row[0] == "forward"] == [("forward", False)]
    assert gen.selected_decode_variant() == "original"


def test_public_device_and_host_logits_use_completed_canonical(case):
    gen, _, _ = trace_fixture(case)
    canonical = gen._logits
    gen._ensure_replay_safe = lambda: None
    gen._replay = lambda **kw: canonical.fill_(23)
    gen._live = True
    gen._model_trace = 1
    gen._previous_table = None
    gen.model.logits_to_host = lambda value, batch: (value, batch)
    out = gen.decode_forward(
        None, None, page_table=gen._inputs[3], kv_cache=gen.kv_cache, return_logits=True, read_from_device=False
    )
    assert out is canonical and out.item() == 23
    out = gen.decode_forward(
        None, None, page_table=gen._inputs[3], kv_cache=gen.kv_cache, sample_on_device=False, read_from_device=True
    )
    assert out == (canonical, 32)


def test_async_default_and_adapter_kind_use_output_identity_not_later_mode(case):
    gen, tt = case
    copies = []

    class Tensor:
        def __init__(self, name):
            self.name = name

        def cpu(self, blocking):
            assert blocking is False
            copies.append(self.name)
            return "host_" + self.name

    tokens, logits = Tensor("tokens"), Tensor("logits")
    gen._inputs[0], gen._logits = tokens, logits
    tt.get_device_tensors = lambda value: [value]
    tt.record_event = lambda *args: "event"
    assert gen.read_output_async(return_logits=True) == ("host_logits", "event")
    assert gen.read_output_async() == ("host_tokens", "event")
    adapter_type = HELPERS["load_class"]("generator_vllm.py", "TTOrnithForCausalLM", {}, methods={"read_decode_output"})
    adapter = adapter_type.__new__(adapter_type)
    adapter.generator, adapter.counters = gen, gen.counters
    adapter._last_device_sampling = True
    assert adapter.read_decode_output(logits, async_read=True) == (["host_logits"], ["event"])
    adapter._last_device_sampling = False
    assert adapter.read_decode_output(tokens, async_read=True) == (["host_tokens"], ["event"])


def test_cleanup_attempts_original_traces_prefill_and_scratch_after_failure(case, expect_error):
    gen, tt, calls = trace_fixture(case)
    gen._sampling_trace, gen._sampling_history_trace, gen._model_trace = "bad", "history", "model"
    gen._active8_trace = "active8"
    gen._prefill_inputs = ["input"]
    gen._prefill_sampling_saved = ["saved"]
    gen._prefill_sampling_masks = ["mask"]
    gen._prefill_sampling_logits, gen._logits = "prefill_logits", "canonical"
    gen._active8.owners = ["scratch"]
    first = ValueError("first native release")

    def release(mesh, trace):
        calls.append(("release", trace))
        if trace == "bad":
            raise first

    tt.release_trace = release
    tt.deallocate = lambda value: calls.append(("free", value))
    with expect_error(ValueError, "first native release") as caught:
        gen.teardown()
    assert caught.value is first
    for trace in ("bad", "history", "model", "active8"):
        assert ("release", trace) in calls
    for value in ("input", "saved", "mask", "prefill_logits", "canonical", "scratch"):
        assert ("free", value) in calls
    assert gen._sampling_trace == "bad"
    assert gen._active8.owners == [] and gen._logits is None


def test_capture_failure_records_only_current_open_trace(case, expect_error):
    gen, tt, calls = trace_fixture(case)
    original_end = tt.end_trace_capture
    count = 0

    def begin(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise ValueError("second begin failed")
        return "first_complete"

    tt.begin_trace_capture = begin
    with expect_error(ValueError, "second begin"):
        gen._capture()
    assert gen._capture_in_progress is None
    assert calls.count(("end", "first_complete")) == 1
    assert tt.end_trace_capture is original_end
    tt.begin_trace_capture = lambda *args, **kwargs: "actually_open"
    tt.end_trace_capture = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("end failed"))
    with expect_error(ValueError, "end failed"):
        gen._capture()
    assert gen._capture_in_progress == "actually_open"


def test_execution_info_uses_actual_sources_and_persistent_logits(case):
    import sys

    gen, _ = case

    class Model:
        pass

    gen.model = Model()
    gen.model.page_block_size = 64
    gen.mesh_device = SimpleNamespace(shape=(1, 4))
    gen.kv_cache.context, gen.kv_cache.num_blocks = 262144, 4129
    gen._logits = SimpleNamespace(buffer_address=lambda: 1234)
    gen._model_trace, gen._active8_trace = 12, 13
    namespace = gen.execution_info.__func__.__globals__
    namespace.update(sys=sys, __file__=str(ROOT / "tt/generator.py"))
    info = gen.execution_info()
    assert info["model_source"] == __file__
    assert info["generator_source"] == str(ROOT / "tt/generator.py")
    assert info["mesh"] == [1, 4] and info["logits_address"] == 1234
    assert info["effective_mode"] == "active8" and info["canonical_logits_owner"] == "generator"


def test_single_model_traversal_specializes_only_hybrid_boundary(expect_error):
    calls = []

    class Tensor:
        def __init__(self, label):
            self.label = label

    tt = SimpleNamespace(
        slice=lambda *args: Tensor("ids"),
        reshape=lambda value, shape: value,
        clone=lambda value: calls.append(("clone", value.label)) or Tensor("saved"),
        where=lambda *args, **kwargs: calls.append(("restore",)),
        deallocate=lambda value: None,
        plus_one=lambda *args, **kwargs: calls.append(("advance",)),
    )

    def layer(name, full):
        return SimpleNamespace(
            is_full_attention=full,
            recurrent_state=Tensor("state"),
            conv_state=[Tensor("conv")] * 3,
            decode_forward=lambda x, **kwargs: calls.append((name,)) or Tensor(name),
        )

    hybrid, attention, specialized = layer("hybrid", False), layer("attention", True), layer("active8", False)
    cache = SimpleNamespace(batch_size=32, decode_layers=[hybrid, attention], active_recurrent=None, active_conv=None)
    cls = HELPERS["load_class"]("model.py", "OrnithModel", {"ttnn": tt}, methods={"decode_forward"})
    model = cls.__new__(cls)
    model.dim, model.embed, model.terminal = 4096, lambda ids: Tensor("embedding"), lambda x: Tensor("logits")
    args = dict(current_pos=None, rot_idxs=None, page_table=None, kv_cache=cache)
    model.decode_forward(Tensor("tokens"), **args)
    assert calls.count(("restore",)) == 4 and ("hybrid",) in calls and ("attention",) in calls
    calls.clear()
    state = SimpleNamespace(cache=cache, proxies={id(hybrid): specialized})
    model.decode_forward(Tensor("tokens"), active8=state, **args)
    assert calls == [("active8",), ("attention",), ("advance",), ("advance",)]
    with expect_error(ValueError, "another canonical cache"):
        model.decode_forward(Tensor("tokens"), active8=SimpleNamespace(cache=object()), **args)


@pytest.mark.parametrize("mismatch", [None, "batch", "pool", "context", "precision", "layers", "mesh", "arch"])
@pytest.mark.parametrize("candidate_policy", ["selected", "mlp8"])
def test_eligibility_keeps_other_profiles_on_original(mismatch, candidate_policy):
    policy = runpy.run_path(str(ROOT / "tt/precision.py"))["load_precision"]()
    model = SimpleNamespace(
        layer_indices=list(range(32)),
        mesh_device=SimpleNamespace(shape=(1, 4), get_num_devices=lambda: 4),
        page_block_size=64,
        max_context=262144,
        dim=4096,
        vocab_size=248320,
        precision=copy.deepcopy(policy),
    )
    cache = SimpleNamespace(
        batch_size=32,
        context=262144,
        num_blocks=4129,
        decode_layers=[SimpleNamespace(mesh_config=SimpleNamespace(residual="replicated"))],
    )
    exported = arch_exports_from_current_source()
    model.mesh_device.arch = lambda: exported.Arch.BLACKHOLE if mismatch != "arch" else object()
    if candidate_policy == "mlp8":
        for group in ("attention", "mlp_gate_up", "mlp_down"):
            model.precision["weight_groups"][group] = "bfloat8_b"
    namespace = {"ttnn": exported, "load_precision": lambda: policy, "copy": copy}
    tree = ast.parse((ROOT / "tt/active8.py").read_text())
    methods = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("eligible", "_precision_eligible")
    ]
    exec(compile(ast.Module(body=methods, type_ignores=[]), "active8.py", "exec"), namespace)
    if mismatch == "batch":
        cache.batch_size = 8
    if mismatch == "pool":
        cache.num_blocks = 4096
    if mismatch == "context":
        cache.context = 32768
    if mismatch == "precision":
        model.precision = {"config_id": "other"}
    if mismatch == "layers":
        model.layer_indices = [0, 3]
    if mismatch == "mesh":
        model.mesh_device.shape = (1, 2)
    assert namespace["eligible"](model, cache) is (mismatch is None)


def test_partial_prefill_free_retains_only_failed_owners_for_retry(case, expect_error):
    gen, tt = case
    gen._prefill_inputs = ["first", "bad", "last"]
    freed = []

    def free(value):
        freed.append(value)
        if value == "bad":
            raise ValueError("native free")

    tt.deallocate = free
    for _ in range(2):
        with expect_error(ValueError, "native free"):
            gen._release_prefill_inputs()
        assert gen._prefill_inputs == ["bad"]
    assert freed == ["first", "bad", "last", "bad"]


def test_invalid_mode_rejected_before_model_or_cache_access(case, expect_error):
    gen, _ = case
    namespace = gen.__init__.__func__.__globals__
    namespace["os"] = SimpleNamespace(environ={"ORNITH_ACTIVE8_MODE": "invalid"})
    with expect_error(ValueError, "ORNITH_ACTIVE8_MODE"):
        type(gen)(object())


def test_family_warms_complete_operations_before_first_capture(case):
    gen, tt, calls = trace_fixture(case)
    gen._logits = None
    gen._active8 = None
    gen._live, gen.owns_cache = False, False
    gen.sampling_mode = "host"
    gen.page_table = gen._inputs[3]
    gen.model.cache_buffers = lambda cache: []
    gen.model.reset_cache = lambda cache, **kw: calls.append(("reset_cache",))
    gen._write_tokens = lambda values: None
    gen._refresh_table = lambda table: None
    tt.synchronize_device = lambda mesh: calls.append(("sync",))

    class State:
        def __init__(self, cache):
            self.cache, self.owners = cache, []

        def prepare(self, model):
            calls.append(("allocate_scratch",))

    gen.ensure_traces.__func__.__globals__["Active8State"] = State
    original_capture = gen._capture

    def capture():
        assert gen._logits is not None and gen._active8 is not None
        calls.append(("capture_start",))
        original_capture()

    gen._capture = capture
    gen.ensure_traces(preserve_cache=False)
    before_capture = calls[: calls.index(("capture_start",))]
    assert before_capture.index(("allocate_scratch",)) < before_capture.index(("forward", False))
    assert [row for row in before_capture if row[0] == "forward"] == [
        ("forward", False),
        ("forward", False),
        ("forward", True),
        ("forward", False),
        ("forward", True),
    ]
    assert gen._model_trace is not None and gen._active8_trace is not None
    assert gen.counters["captures"] == 1


def arch_exports_from_current_source():
    """Use actual Python export/alias ASTs; never invent a root convenience API."""
    metal = (ROOT / "doc").resolve().parents[3]
    exports = ast.parse((metal / "ttnn/ttnn/__init__.py").read_text())
    device = ast.parse((metal / "ttnn/ttnn/device.py").read_text())
    imports = [node for node in exports.body if isinstance(node, ast.ImportFrom) and node.module == "ttnn.device"]
    imported = {alias.asname or alias.name: alias.name for node in imports for alias in node.names}
    assert imported["Arch"] == "Arch"
    assert "is_blackhole" not in imported
    alias = next(
        node
        for node in device.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "Arch" for target in node.targets)
    )
    native_arch = SimpleNamespace(BLACKHOLE=object())
    namespace = {"ttnn": SimpleNamespace(_ttnn=SimpleNamespace(device=SimpleNamespace(Arch=native_arch)))}
    exec(compile(ast.Module(body=[alias], type_ignores=[]), "device.py", "exec"), namespace)
    return SimpleNamespace(Arch=namespace[imported["Arch"]])


def test_current_arch_exports_match_existing_model_and_native_mesh_binding():
    exported = arch_exports_from_current_source()
    assert not hasattr(exported, "is_blackhole")
    metal = (ROOT / "doc").resolve().parents[3]
    existing = ast.parse((ROOT / "tt/model.py").read_text())
    assert any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "arch"
        for node in ast.walk(existing)
    )
    binding = (metal / "ttnn/core/distributed/distributed_nanobind.cpp").read_text()
    assert '"arch"' in binding and "&MeshDevice::arch" in binding
    enum = (metal / "ttnn/cpp/ttnn-nanobind/device.cpp").read_text()
    assert '.value("BLACKHOLE", tt::ARCH::BLACKHOLE)' in enum


def test_prefix_snapshot_change_recaptures_both_resident_models(case):
    gen, _, calls = trace_fixture(case)
    cache = SimpleNamespace(generation=0)
    gen.model = SimpleNamespace(prefix_cache=cache)
    gen._capture()
    old_ids = gen._model_trace, gen._active8_trace, gen._sampling_trace
    captures = gen.counters["captures"]
    gen._ensure_replay_safe()
    assert gen.counters["captures"] == captures
    cache.generation += 1
    gen._ensure_replay_safe()
    assert gen.counters["captures"] == captures + 1
    assert all(old != new for old, new in zip(old_ids, (gen._model_trace, gen._active8_trace, gen._sampling_trace)))
    assert gen._captured_prefix_generation == cache.generation
    assert gen.model.prefix_cache is cache
    gen._ensure_replay_safe()
    assert gen.counters["captures"] == captures + 1
