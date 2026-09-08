# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm import Gemma4ForCausalLM
from models.common.sampling import SamplingParams


def test_capabilities_and_context_contract():
    assert Gemma4ForCausalLM.get_max_tokens_all_users(num_devices=1) == 50_624
    assert Gemma4ForCausalLM.get_max_tokens_all_users(num_devices=2) == 262_144
    assert Gemma4ForCausalLM.get_max_tokens_all_users(num_devices=4) == 262_144
    assert Gemma4ForCausalLM.model_capabilities == {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_async_decode_overlap": True,
        "supports_sample_on_device": True,
        "supports_device_penalties": False,
        "supports_device_seeded_sampling": False,
        "max_device_top_k": 32,
        "state_slots_are_stateless": True,
    }


def test_adapter_delegates_sampling_and_decode():
    source = inspect.getsource(Gemma4ForCausalLM)
    assert "gen.prefill_forward" in source
    assert "gen.decode_forward" in source
    assert "gen.sample_device_logits" in source
    assert "gen._read_tokens" in source
    forbidden = ("argmax", "topk", "top_k(logits", "full_logits")
    assert all(fragment not in source.lower() for fragment in forbidden)


def test_adapter_has_no_sampling_implementation():
    tree = ast.parse(inspect.getsource(Gemma4ForCausalLM))
    methods = {node.name for node in tree.body[0].body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "sample" not in methods
    assert "sampling" not in methods


def test_vllm_initialization_preserves_precision_policy_environment_override():
    source = inspect.getsource(Gemma4ForCausalLM.initialize_vllm_model)
    assert 'os.getenv("GEMMA4_PRECISION_CONFIG") or PRECISION_CONFIG' in source


def test_selected_precision_and_external_cache_are_explicit():
    source = inspect.getsource(Gemma4ForCausalLM.initialize_vllm_model)
    assert "precision_config_path=" in source and "PRECISION_CONFIG" in source
    assert "create_kv_cache=False" in source
    allocation = inspect.getsource(Gemma4ForCausalLM.allocate_kv_cache_per_layer)
    assert "gen.model.kv_cache_dtype" in allocation
    assert "FullModelState" in allocation
    assert "unique_buffers" in allocation
    assert "tensor_idx" in allocation
    assert "torch_dtype != torch.bfloat16" in allocation


def test_plugin_registration_targets_autoport():
    platform = Path("../vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py").read_text()
    assert '"models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm:"' in platform
    assert '"Gemma4ForCausalLM"' in platform
    assert 'os.getenv("TT_GEMMA4_TEXT_VER", "tt_transformers")' in platform
    assert 'gemma4_text_version == "google_gemma_4_26b_a4b_it_autoport"' in platform


def test_async_split_is_implemented():
    assert hasattr(Gemma4ForCausalLM, "read_decode_output")
    assert hasattr(Gemma4ForCausalLM, "process_decode_output_host")
    decode = inspect.getsource(Gemma4ForCausalLM.decode_forward)
    assert "read_from_device" in decode
    assert "enable_trace=True" in decode
    assert "self._page_table_refreshes" in inspect.getsource(Gemma4ForCausalLM._refresh_page_tables)
    process = inspect.getsource(Gemma4ForCausalLM.process_decode_output_host)
    assert "[: self.max_batch_size]" in process


def test_unseeded_greedy_request_uses_deterministic_device_seed_default():
    params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0, seed=None)
    assert Gemma4ForCausalLM._sampling_values(params, 1)["seeds"] == (0,)


def test_unseeded_stochastic_rows_use_distinct_device_streams():
    params = SamplingParams(
        temperature=[1.0] * 4,
        top_k=[10] * 4,
        top_p=[1.0] * 4,
        seed=[None] * 4,
    )
    assert Gemma4ForCausalLM._sampling_values(params, 4)["seeds"] == (0, 1, 2, 3)
    second = Gemma4ForCausalLM._sampling_values(params, 4, unseeded_epoch=1)["seeds"]
    assert second == (104729, 104730, 104731, 104732)


def test_formatted_greedy_requests_have_seed_independent_canonical_trace_key():
    from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator

    generator = Gemma4Generator.__new__(Gemma4Generator)
    params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0, seed=None)
    first = Gemma4ForCausalLM._sampling_values(params, 32, unseeded_epoch=1)
    second = Gemma4ForCausalLM._sampling_values(params, 32, unseeded_epoch=2)
    assert first["seeds"] != second["seeds"]
    assert first["temperature"] == second["temperature"] == (0.0,) * 32
    first_spec = generator._sampling_spec(32, **first)
    second_spec = generator._sampling_spec(32, **second)
    assert first_spec.greedy and second_spec.greedy
    assert first_spec.key == second_spec.key

    mixed = SamplingParams(temperature=[0.0, 0.5], top_k=[1, 8], top_p=[0.0, 0.9])
    mixed_values = Gemma4ForCausalLM._sampling_values(mixed, 2, unseeded_epoch=2)
    assert mixed_values["temperature"] == (1.0, 2.0)
    assert not generator._sampling_spec(2, **mixed_values).greedy
    positive_temperature = SamplingParams(temperature=1.0, top_k=1, top_p=0.0)
    assert Gemma4ForCausalLM._sampling_values(positive_temperature, 1)["temperature"] == (1.0,)


def test_host_sampling_compatibility_formats_rank2_logits():
    prefill = inspect.getsource(Gemma4ForCausalLM.prefill_forward)
    decode = inspect.getsource(Gemma4ForCausalLM.decode_forward)
    assert "_host_prefill_logits(logits, len(lengths))" in prefill
    assert ".reshape(-1, gen.model.vocab_size)[:execution_batch]" in decode
    helper = inspect.getsource(Gemma4ForCausalLM._host_prefill_logits)
    assert "reshape(batch_size, 1, -1)" in helper


def test_mixed_prefill_delegates_each_sampling_output_to_generator():
    source = inspect.getsource(Gemma4ForCausalLM.prefill_forward)
    assert "if isinstance(logits, list):" in source
    assert "gen.sample_device_logits(row_logits" in source


def test_full_model_multi_user_prefill_uses_single_user_kernel_rows():
    from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator

    source = inspect.getsource(Gemma4Generator.prefill_forward)
    assert "if len(prompt_lens) > 1:" in source
    assert "user_id=row" in source


def test_chunked_prefill_propagates_absolute_positions_and_scheduler_tables():
    from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator

    adapter = inspect.getsource(Gemma4ForCausalLM.prefill_forward)
    generator = inspect.getsource(Gemma4Generator.prefill_forward)
    assert "start_pos=positions" in adapter
    assert "chunk_page_tables=self._chunk_page_tables" in adapter
    assert "end - start" in adapter
    assert "tokens[row, start:end]" in adapter
    assert "torch.arange(start, end" in adapter
    assert "return None" in inspect.getsource(Gemma4ForCausalLM._chunk_page_tables)
    assert "position_rows[row]" in generator
    assert "chunk_page_tables=chunk_page_tables" in generator
    assert "physical_len = _padded_prefill_len(logical_len)" in generator
    assert "first_position + physical_len" in generator


def test_slot_remap_is_validated_and_recaptures_without_a_second_rng_path():
    source = inspect.getsource(Gemma4ForCausalLM.decode_forward)
    assert "slot_remap must be a permutation" in source
    assert "remap_changed" in source
    assert "gen.remap_sampling_slots" not in source
    assert "SeedManager" not in source
    assert "slot remapping is not yet supported" not in source


def test_unsupported_device_sampling_features_are_explicit_plugin_fallbacks():
    capabilities = Gemma4ForCausalLM.model_capabilities
    assert capabilities["supports_device_penalties"] is False
    assert capabilities["supports_device_seeded_sampling"] is False
    runner = Path("../vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py").read_text()
    assert "self.supports_device_penalties" in runner
    assert "presence_penalties_reqs" in runner
    assert "self.supports_device_seeded_sampling" in runner
    assert "SEED_NONE_SENTINEL" in runner


def test_decode_uses_batch_one_fast_path_and_padded_multi_request_width():
    source = inspect.getsource(Gemma4ForCausalLM.decode_forward)
    assert "logical_batch = int((flat_positions >= 0).sum().item())" in source
    assert "execution_batch = 1 if logical_batch == 1 else self.max_batch_size" in source
    assert "pack active requests before inactive slots" in source
    assert "tokens.reshape(-1)[:execution_batch].reshape(execution_batch, 1)" in source
    assert "active_rows_changed = not torch.equal(state.active_mask, desired_active)" in source
    assert "active if active_rows_changed or reset_batch or remap_changed or refresh_decode_inputs else None" in source


def test_decode_releases_pinned_traces_before_batch_shape_recapture():
    source = inspect.getsource(Gemma4ForCausalLM.decode_forward)
    assert "batch_shape_changed = self._last_execution_batch not in (None, execution_batch)" in source
    assert "or batch_shape_changed" in source
    assert source.index("self._release_decode_traces()") < source.index("gen.decode_forward(")


def test_adapter_steady_decode_ignores_stale_host_feedback_and_refreshes_changed_pages_once(monkeypatch):
    class FakeGenerator:
        def __init__(self, state):
            self.model = SimpleNamespace(num_layers=1, vocab_size=4)
            self.state = state
            self.calls = []

        def decode_forward(self, tokens, positions, *, active_mask=None, **kwargs):
            if active_mask is not None:
                self.state.active_mask.zero_()
                self.state.active_mask[: active_mask.numel()] = active_mask
            self.calls.append(
                {
                    "tokens": tokens.clone(),
                    "positions": positions.clone(),
                    "active_mask": None if active_mask is None else active_mask.clone(),
                    "page_table": kwargs["page_table"],
                }
            )
            return torch.zeros((32, 4))

        def _gather_logits_to_torch(self, output):
            return output

    state = SimpleNamespace(
        kv_cache=object(),
        active_mask=torch.zeros(32, dtype=torch.bool),
        page_tables=[object()],
    )
    gen = FakeGenerator(state)
    adapter = Gemma4ForCausalLM()
    adapter.max_batch_size = 32
    adapter._serving_state = state
    refreshes = []
    releases = []

    monkeypatch.setattr(adapter, "_require_generator", lambda: gen)
    monkeypatch.setattr(adapter, "_sampling_values", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(adapter, "_release_decode_traces", lambda: releases.append(True))

    def refresh(tables):
        refreshes.append([table.clone() for table in tables])
        adapter._last_page_tables = [table.clone() for table in tables]

    monkeypatch.setattr(adapter, "_refresh_page_tables", refresh)
    table_a = torch.tensor([[0, 1]], dtype=torch.int32)
    table_b = torch.tensor([[1, 0]], dtype=torch.int32)
    positions = torch.cat((torch.tensor([5]), torch.full((31,), -1)))

    adapter.decode_forward(
        torch.ones((32, 1), dtype=torch.int64),
        positions,
        table_a,
        state.kv_cache,
        sampling_params=object(),
        read_from_device=False,
    )
    adapter.decode_forward(
        torch.full((32, 1), 999, dtype=torch.int64),
        positions,
        table_a,
        state.kv_cache,
        sampling_params=object(),
        read_from_device=False,
    )
    adapter.decode_forward(
        torch.full((32, 1), 999, dtype=torch.int64),
        positions,
        table_b,
        state.kv_cache,
        sampling_params=object(),
        read_from_device=False,
    )

    assert not releases
    assert len(refreshes) == 2  # initial bind, then the one changed table
    assert gen.calls[0]["active_mask"].tolist() == [True]
    assert gen.calls[1]["active_mask"] is None
    assert gen.calls[2]["active_mask"] is None
    assert gen.calls[1]["tokens"].item() == 999
    assert gen.calls[1]["positions"].item() == 5

    def decode(**kwargs):
        return adapter.decode_forward(
            torch.ones((32, 1), dtype=torch.int64),
            positions,
            table_b,
            state.kv_cache,
            read_from_device=False,
            **kwargs,
        )

    decode(sampling_params=object(), reset_batch=True)
    assert not releases
    assert gen.calls[-1]["active_mask"].tolist() == [True]
    adapter._decode_ready = False  # A prefill boundary refreshes without release.
    decode(sampling_params=object())
    assert not releases
    assert gen.calls[-1]["active_mask"].tolist() == [True]
    decode(sampling_params=object())
    assert gen.calls[-1]["active_mask"] is None

    positions[1] = 7  # B1 -> padded execution requires a different trace.
    decode(sampling_params=object())
    assert len(releases) == 1
    remap = torch.arange(32)
    remap[:2] = torch.tensor([1, 0])
    decode(sampling_params=object(), slot_remap=remap)
    assert len(releases) == 2
    decode(sampling_params=None)
    assert len(releases) == 3
    assert not adapter._decode_ready
    decode(sampling_params=object())
    assert len(releases) == 3
    assert gen.calls[-1]["active_mask"] is not None
    decode(sampling_params=object())
    assert gen.calls[-1]["active_mask"] is None


def test_page_table_refresh_uses_host_staging_and_existing_device_storage(monkeypatch):
    target = SimpleNamespace(shape=(32, 4))
    adapter = Gemma4ForCausalLM()
    adapter.mesh_device = object()
    adapter._serving_state = SimpleNamespace(page_tables=[target])
    calls = []

    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: ("replicate", mesh))

    def from_torch(source, **kwargs):
        calls.append(("from_torch", source.clone(), kwargs))
        return "host-staging"

    monkeypatch.setattr(ttnn, "from_torch", from_torch)
    monkeypatch.setattr(
        ttnn,
        "copy_host_to_device_tensor",
        lambda source, destination: calls.append(("copy", source, destination)),
    )

    adapter._refresh_page_tables([torch.tensor([[7, 8]], dtype=torch.int32)])

    _, staged, kwargs = calls[0]
    assert "device" not in kwargs
    assert staged.shape == (32, 4)
    assert staged[0, :2].tolist() == [7, 8]
    assert calls[1] == ("copy", "host-staging", target)


def test_page_table_groups_copy_selectively_and_preserve_alias_mutation_semantics(monkeypatch):
    adapter = Gemma4ForCausalLM()
    adapter.mesh_device = object()
    targets = [SimpleNamespace(shape=(4, 4), layer=layer) for layer in range(30)]
    adapter._serving_state = SimpleNamespace(page_tables=targets)
    groups = [torch.full((2, 4), group, dtype=torch.int32) for group in range(6)]
    tables = [groups[layer % 6] for layer in range(30)]
    copies, comparisons = [], []
    device_contents = {}
    original_equal = torch.equal

    def equal(current, previous):
        comparisons.append((id(current), id(previous)))
        return original_equal(current, previous)

    def from_torch(source, **kwargs):
        assert "device" not in kwargs
        return source.clone()

    def copy(source, destination):
        copies.append(destination.layer)
        device_contents[destination.layer] = source.clone()

    monkeypatch.setattr(torch, "equal", equal)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: mesh)
    monkeypatch.setattr(ttnn, "from_torch", from_torch)
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", copy)

    def refresh(expected_layers):
        copies.clear()
        adapter._refresh_page_tables(tables)
        assert copies == expected_layers
        for source, target in zip(tables, targets):
            expected = torch.zeros(target.shape, dtype=torch.int32)
            rows, cols = min(source.shape[0], 4), min(source.shape[1], 4)
            expected[:rows, :cols] = source[:rows, :cols]
            assert original_equal(device_contents[target.layer], expected)

    refresh(list(range(30)))
    assert len({id(table) for table in adapter._last_page_tables}) == 6
    assert all(snapshot is not current for snapshot, current in zip(adapter._last_page_tables, tables))
    comparisons.clear()
    assert not adapter._page_tables_changed(tables)
    assert len(comparisons) == 6
    refresh([])
    assert adapter._page_table_refreshes == 1

    groups[0][0, 0] += 10  # One real HMA group owns five distinct device targets.
    assert adapter._page_tables_changed(tables)
    refresh([0, 6, 12, 18, 24])
    assert adapter._page_table_refreshes == 2

    tables[6] = tables[6].clone()  # Split one layer away from its old alias group.
    tables[6][0, 0] += 7
    refresh([6])
    groups[0][0, 0] += 20
    refresh([0, 12, 18, 24])
    tables[6] = groups[0]  # One current object now pairs with two old snapshots.
    assert adapter._page_tables_changed(tables)
    refresh([6])
    assert adapter._last_page_tables[0] is adapter._last_page_tables[6]

    tables[12] = tables[12].clone()  # Equal but distinct tables require no upload.
    assert not adapter._page_tables_changed(tables)
    refresh([])
    tables[12][0, 0] += 1
    refresh([12])
    tables[12] = torch.tensor([[9]], dtype=torch.int32)
    refresh([12])  # A shorter table must zero all previously populated trailing cells.
    assert device_contents[12].sum().item() == 9
    assert len({id(target) for target in adapter._serving_state.page_tables}) == 30


@pytest.mark.parametrize("compile_stage", [None, "prefill", "sampling", "read"])
@pytest.mark.parametrize("list_logits", [False, True])
def test_prefill_retires_traces_only_when_program_cache_changes(monkeypatch, compile_stage, list_logits):
    events = []
    program_count = [10]
    state = SimpleNamespace(kv_cache=object(), page_tables=[object()])

    def operation(stage, result):
        events.append(stage)
        if compile_stage == stage:
            program_count[0] += 1
        return result

    gen = SimpleNamespace(
        model=SimpleNamespace(num_layers=1),
        _trace_cache={"retained": object()},
        prefill_forward=lambda *_args, **_kwargs: operation("prefill", [object()] if list_logits else object()),
        sample_device_logits=lambda *_args, **_kwargs: operation("sampling", object()),
        _read_tokens=lambda *_args, **_kwargs: operation("read", torch.tensor([42])),
    )
    adapter = Gemma4ForCausalLM()
    adapter.mesh_device = SimpleNamespace(num_program_cache_entries=lambda: program_count[0])
    adapter._serving_state = state
    monkeypatch.setattr(adapter, "_require_generator", lambda: gen)
    monkeypatch.setattr(adapter, "_page_tables_changed", lambda _tables: False)

    def release():
        events.append("release")
        gen._trace_cache.clear()

    monkeypatch.setattr(adapter, "_release_decode_traces", release)
    result = adapter.prefill_forward(
        torch.ones((1, 33), dtype=torch.long),
        page_table=torch.zeros((1, 1), dtype=torch.int32),
        kv_cache=state.kv_cache,
        prompt_lens=[33],
        sampling_params=SamplingParams(temperature=0.0, top_k=1, top_p=0.0),
    )
    assert result.tolist() == [42]
    assert events == ["prefill", "sampling", "read"] + (["release"] if compile_stage is not None else [])
    assert bool(gen._trace_cache) == (compile_stage is None)


def test_state_slot_contract_keeps_external_cache_in_execution_row_order():
    prefill = inspect.getsource(Gemma4ForCausalLM.prefill_forward)
    capabilities = Gemma4ForCausalLM.model_capabilities
    assert capabilities["state_slots_are_stateless"] is True
    assert "one unique state slot per prefill row" in prefill
    assert "state slot outside the serving batch" in prefill
    assert "external page tables are already packed" in prefill
