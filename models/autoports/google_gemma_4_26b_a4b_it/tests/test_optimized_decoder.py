# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and path-identity gates for the Gemma-4 optimized decoder.

The functional stage's real-weight/HF oracles are intentionally reused so the
optimized stage cannot weaken thresholds or subtly change cache construction.
Each wrapper replaces the decoder constructor in the oracle module and checks
that optimized material methods were actually entered.

BF16 is the accepted cache policy. BFP8 numerical cases are opt-in candidate
checks and retain the full acceptance thresholds when explicitly enabled.
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
from types import SimpleNamespace

import pytest
import torch

import models.autoports.google_gemma_4_26b_a4b_it.tests.test_functional_decoder as functional_tests
import models.autoports.google_gemma_4_26b_a4b_it.tests.test_trace_mutable_buffers as mutable_tests
import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import FunctionalDecoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.fused_decoder import FusedDecoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import (
    _RESIDUAL_BOUNDARY_COUNTERS,
    OptimizedDecoder,
    _attention_candidate_options_from_env,
    _folded_tensor_cache_path,
    _prefill_attention_2d_geometry,
    _prefill_attention_options_from_env,
    _prepare_folded_state_dict,
    _residual_shard_cores_from_env,
    _residual_shard_geometry,
    _resolve_attention_candidate_geometry,
    _resolved_graph_fusion_policy,
)

ARTIFACT_DIR = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder")
OPTIMIZED_SOURCE = Path("models/autoports/google_gemma_4_26b_a4b_it/tt/optimized_decoder.py")
_KV_CACHE_EVIDENCE = {}
_KV_CACHE_INPUT_VALUES = {}
_CANDIDATE_PATH_EVIDENCE = {}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved_policy() -> dict:
    signature = inspect.signature(OptimizedDecoder.from_state_dict)
    defaults = {
        name: str(parameter.default)
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    overrides = {name: value for name, value in sorted(os.environ.items()) if name.startswith("GEMMA4_OPT")}
    attention_options = _attention_candidate_options_from_env()
    attention_runtime_geometry = {
        kind.name: _resolve_attention_candidate_geometry(
            layer_kind=kind.name,
            q_width=kind.q_width,
            qkv_width=kind.qkv_width,
            **attention_options,
        )
        for kind in (functional_tests.SLIDING_KIND, functional_tests.FULL_KIND)
    }
    attention_batch32_runtime_geometry = {
        kind.name: _resolve_attention_candidate_geometry(
            layer_kind=kind.name,
            q_width=kind.q_width,
            qkv_width=kind.qkv_width,
            qkv_working_cores=8,
            qkv_in0_block_w=1,
            qkv_out_subblock_w=4,
            full_o_working_cores=8,
            full_o_in0_block_w=32,
            full_o_out_subblock_w=1,
            attention_allow_padding=False,
        )
        for kind in (functional_tests.SLIDING_KIND, functional_tests.FULL_KIND)
    }
    return {
        "constructor_defaults": defaults,
        "environment_overrides": overrides,
        "attention_runtime_geometry": attention_runtime_geometry,
        "attention_batch32_runtime_geometry": attention_batch32_runtime_geometry,
        "batch32_expert_gate_up_dtype": str(ttnn.bfloat8_b),
    }


def _stamp_artifact(path: Path, *, exact_command: str | None = None) -> dict:
    contents = json.loads(path.read_text()) if path.exists() else {}
    contents["optimized_stage_provenance"] = {
        "checkout_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "exact_command": exact_command
        or os.getenv("GEMMA4_OPT_EXACT_COMMAND", f"pytest -q {os.getenv('PYTEST_CURRENT_TEST', '').split(' ')[0]}"),
        "hardware": {
            "arch": "Blackhole P300C",
            "platform": platform.platform(),
        },
        "optimized_decoder_sha256": _sha256(OPTIMIZED_SOURCE),
        "optimized_test_sha256": _sha256(Path(__file__)),
        "resolved_policy": _resolved_policy(),
        "exercised_kv_cache": _KV_CACHE_EVIDENCE,
        "exercised_candidate_paths": _CANDIDATE_PATH_EVIDENCE.get(
            os.getenv("PYTEST_CURRENT_TEST", "").split(" ")[0], {}
        ),
    }
    path.write_text(json.dumps(contents, indent=2, sort_keys=True) + "\n")
    candidate_id = os.getenv("GEMMA4_OPT_CANDIDATE_ID")
    if candidate_id:
        candidate_dir = ARTIFACT_DIR / "candidate_runs"
        candidate_dir.mkdir(exist_ok=True)
        candidate_path = candidate_dir / f"{candidate_id}.json"
        candidate = json.loads(candidate_path.read_text()) if candidate_path.exists() else {}
        candidate[path.name] = contents
        candidate_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    return contents


def _install_optimized_oracle(monkeypatch, module, *, required_methods):
    _install_requested_kv_cache_dtype(monkeypatch)
    calls = {name: 0 for name in required_methods}
    monkeypatch.setattr(module, "FunctionalDecoder", OptimizedDecoder)
    monkeypatch.setattr(module, "ARTIFACT_DIR", ARTIFACT_DIR)
    for name in required_methods:
        original = getattr(OptimizedDecoder, name)

        def wrapped(self, *args, __name=name, __original=original, **kwargs):
            calls[__name] += 1
            packed_counter = {
                "_moe_prefill_chunk": "packed_expert_prefill",
                "_moe_decode_single_user": "packed_expert_decode",
            }.get(__name)
            packed_before = self.optimized_path_counters.get(packed_counter, 0)
            boundary_before = {
                counter: self.optimized_path_counters[counter]
                for counter in ("residual_chain_decode", *_RESIDUAL_BOUNDARY_COUNTERS)
            }
            requested_r22_before = {}
            if __name == "decode_forward":
                hidden_states = args[0] if args else kwargs["hidden_states"]
                if hidden_states.shape[-2] == 1:
                    r22_request = os.getenv("GEMMA4_OPT_R22_DRAM_SHARDED")
                    if r22_request == "1":
                        requested_roles = {
                            role.strip()
                            for role in os.environ["GEMMA4_OPT_DRAM_SHARDED_ROLES"].split(",")
                            if role.strip()
                        }
                    elif r22_request is None:
                        attention_roles = {"o_proj"} if self.layer_kind.name == "sliding_attention" else set()
                        packed_default = os.getenv("GEMMA4_OPT_R22_PACKED_DENSE_GATE_UP", "1") == "1"
                        dense_roles = (
                            {"packed_mlp_gate_up", "mlp_down"} if packed_default else {"mlp_gate", "mlp_up", "mlp_down"}
                        )
                        requested_roles = attention_roles | dense_roles
                    else:
                        requested_roles = set()
                    if requested_roles:
                        requested_r22_before.update(
                            (f"r22_dram_{role}", self.optimized_path_counters[f"r22_dram_{role}"])
                            for role in requested_roles
                        )
                    packed_request = os.getenv("GEMMA4_OPT_R22_PACKED_DENSE_GATE_UP")
                    if packed_request == "1" or (packed_request is None and r22_request is None):
                        requested_r22_before["r22_packed_dense"] = self.optimized_path_counters["r22_packed_dense"]
            if __name == "_attention_decode" and self.residual_shard_cores:
                original_split = ttnn.experimental.nlp_create_qkv_heads_decode
                split_calls = 0

                def checked_split(xqkv, *split_args, **split_kwargs):
                    nonlocal split_calls
                    split_calls += 1
                    # The width-sharded splitter has a terminal runtime-arg
                    # read past its coordinate table. Assert the real R22
                    # call selects the watcher-safe interleaved factory.
                    assert not xqkv.is_sharded()
                    assert xqkv.memory_config() == ttnn.L1_MEMORY_CONFIG
                    return original_split(xqkv, *split_args, **split_kwargs)

                with monkeypatch.context() as split_patch:
                    split_patch.setattr(ttnn.experimental, "nlp_create_qkv_heads_decode", checked_split)
                    output = __original(self, *args, **kwargs)
                assert split_calls == 1
            else:
                output = __original(self, *args, **kwargs)
            packed_enabled = {
                "packed_expert_decode": self.packed_expert_decode_gate_up,
                "packed_expert_prefill": self.packed_expert_prefill_gate_up,
            }.get(packed_counter, False)
            if packed_enabled:
                assert self.optimized_path_counters[packed_counter] > packed_before
            if __name == "decode_forward" and self.residual_shard_cores:
                boundary_delta = {
                    counter: self.optimized_path_counters[counter] - boundary_before[counter]
                    for counter in boundary_before
                }
                assert all(count == 1 for count in boundary_delta.values()), boundary_delta
            if requested_r22_before:
                requested_r22_delta = {
                    counter: self.optimized_path_counters[counter] - before
                    for counter, before in requested_r22_before.items()
                }
                assert all(count > 0 for count in requested_r22_delta.values()), requested_r22_delta
            runtime = {
                name: getattr(self, name, {})
                for name in ("prefill_attention_runtime", "routing_runtime", "r22_projection_runtime")
            }
            if any(runtime.values()):
                runtime["host_dispatch_counters"] = dict(self.optimized_path_counters)
                node = os.getenv("PYTEST_CURRENT_TEST", "").split(" ")[0]
                _CANDIDATE_PATH_EVIDENCE.setdefault(node, {})[self.layer_kind.name] = runtime
            return output

        monkeypatch.setattr(OptimizedDecoder, name, wrapped)
    return calls


def _requested_kv_cache_dtype():
    name = os.getenv("GEMMA4_OPT_KV_CACHE_DTYPE", "bf16")
    choices = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}
    if name not in choices:
        raise ValueError(f"GEMMA4_OPT_KV_CACHE_DTYPE must be bf16 or bfp8, got {name!r}")
    return choices[name]


def _tensor_descriptor(tensor):
    return {
        "dtype": str(tensor.dtype),
        "layout": str(tensor.layout),
        "shape": list(tensor.shape),
        "padded_shape": list(tensor.padded_shape),
        "buffer_address": int(tensor.buffer_address()),
    }


def _stage_payload_to_stable(mesh_device, payload, stable):
    """Prepare all host transfer buffers before capturing a persistent trace."""
    staged = {}
    for name, source in payload.items():
        target = stable["kv_cache"][0 if name == "key_cache" else 1] if name.endswith("_cache") else stable[name]
        host = mutable_tests._host_tt(mesh_device, source, dtype=target.dtype, layout=target.layout)
        staged[name] = {"host": host}
        if name in ("current_pos", "page_table"):
            staged[name]["values"] = source.tolist()
    return staged


def _copy_staged_payload_to_stable(staged, stable):
    """Refresh captured addresses using already allocated host transfer buffers."""
    uploads = {}
    for name, source in staged.items():
        target = stable["kv_cache"][0 if name == "key_cache" else 1] if name.endswith("_cache") else stable[name]
        host = source["host"]
        assert host.dtype == target.dtype, name
        assert host.layout == target.layout, name
        assert host.shape == target.shape, name
        assert host.padded_shape == target.padded_shape, name
        uploads[name] = _tensor_descriptor(target)
        if "values" in source:
            uploads[name]["values"] = source["values"]
            _KV_CACHE_INPUT_VALUES[int(target.buffer_address())] = source["values"]
        ttnn.copy_host_to_device_tensor(host, target)
    if _KV_CACHE_EVIDENCE:
        _KV_CACHE_EVIDENCE["mutable_uploads"].append(uploads)


def _copy_payload_to_stable(mesh_device, payload, stable):
    """Stage mutable inputs in the actual target dtype and retain them through replay."""
    staged = _stage_payload_to_stable(mesh_device, payload, stable)
    _copy_staged_payload_to_stable(staged, stable)
    return [source["host"] for source in staged.values()]


def _install_requested_kv_cache_dtype(monkeypatch, *, decoder_cls=OptimizedDecoder) -> None:
    """Apply and observe the cache policy in both reused oracle namespaces."""
    cache_dtype = _requested_kv_cache_dtype()
    evidence = {"requested_dtype": str(cache_dtype), "cache_descriptors": [], "mutable_uploads": []}
    monkeypatch.setitem(globals(), "_KV_CACHE_EVIDENCE", evidence)
    integer_inputs = {}
    monkeypatch.setitem(globals(), "_KV_CACHE_INPUT_VALUES", integer_inputs)
    original_as_tt = functional_tests._as_tt

    def as_tt_with_cache_policy(mesh, tensor, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        # The inherited oracles have no separate cache allocator. These two
        # physical cache geometries exclude hidden states (width 2816), RoPE
        # (one head axis), and row-major metadata; observe the caller-owned
        # tensors again at decoder entry rather than trusting this selection.
        cache_geometry = tensor.ndim == 4 and tuple(tensor.shape[1:]) in {
            (8, 64, 256),
            (2, 128, 512),
        }
        if cache_geometry and layout == ttnn.TILE_LAYOUT and dtype in (ttnn.bfloat16, ttnn.bfloat8_b):
            dtype = cache_dtype
        result = original_as_tt(mesh, tensor, dtype=dtype, layout=layout)
        if dtype == ttnn.int32:
            integer_inputs[int(result.buffer_address())] = tensor.tolist()
        return result

    monkeypatch.setattr(functional_tests, "_as_tt", as_tt_with_cache_policy)
    monkeypatch.setattr(mutable_tests, "_as_tt", as_tt_with_cache_policy)
    monkeypatch.setattr(mutable_tests, "_copy_payload_to_stable", _copy_payload_to_stable)

    for name in ("prefill_forward", "decode_forward"):
        original = getattr(decoder_cls, name)

        def checked_cache(self, hidden_states, *, __name=name, __original=original, **kwargs):
            key, value = kwargs["kv_cache"]
            assert (
                key.dtype == value.dtype == cache_dtype
            ), f"KV cache dtype mismatch: {key.dtype}, {value.dtype}; expected {cache_dtype}"
            assert key.layout == value.layout == ttnn.TILE_LAYOUT
            assert key.shape == value.shape
            assert tuple(key.shape)[1:] in {(8, 64, 256), (2, 128, 512)}, key.shape
            assert key.buffer_address() != value.buffer_address()
            page_table = kwargs["page_table"]
            assert page_table.dtype == ttnn.int32 and page_table.layout == ttnn.ROW_MAJOR_LAYOUT
            descriptor = {
                "layer_kind": self.layer_kind.name,
                "key": _tensor_descriptor(key),
                "value": _tensor_descriptor(value),
                "logical_cache_view": [
                    int(key.shape[0]),
                    self.layer_kind.num_kv_heads,
                    self.layer_kind.block_size,
                    self.layer_kind.head_dim,
                ],
                "block_size": self.layer_kind.block_size,
                "cache_position_modulo": kwargs.get("cache_position_modulo"),
                "page_table": _tensor_descriptor(page_table),
                "page_table_values": integer_inputs.get(int(page_table.buffer_address())),
                "user_batch": int(hidden_states.shape[2] if __name == "decode_forward" else hidden_states.shape[0]),
            }
            # Keep long wrap-stress evidence compact while retaining every
            # observed eager/capture position. Replay refreshes are recorded
            # separately by the mutable shim or the owning oracle artifact.
            observations = evidence["cache_descriptors"]
            match = next((item for item in observations if item["descriptor"] == descriptor), None)
            if match is None:
                match = {"descriptor": descriptor, "prefill_calls": 0, "decode_calls": 0, "entry_positions": []}
                observations.append(match)
            match["prefill_calls" if __name == "prefill_forward" else "decode_calls"] += 1
            if __name == "decode_forward":
                positions = integer_inputs.get(int(kwargs["current_pos"].buffer_address()))
                if positions is not None and positions not in match["entry_positions"]:
                    match["entry_positions"].append(positions)
            return __original(self, hidden_states, **kwargs)

        monkeypatch.setattr(decoder_cls, name, checked_cache)


@pytest.mark.parametrize("value", [None, "bf16", "bfp8", "", "BFLOAT8_B", "BFP8", "float32"])
def test_optimized_kv_cache_dtype_policy_host(monkeypatch, expect_error, value):
    if value is None:
        monkeypatch.delenv("GEMMA4_OPT_KV_CACHE_DTYPE", raising=False)
    else:
        monkeypatch.setenv("GEMMA4_OPT_KV_CACHE_DTYPE", value)
    if value not in (None, "bf16", "bfp8"):
        with expect_error(ValueError, "must be bf16 or bfp8"):
            _requested_kv_cache_dtype()
    else:
        assert _requested_kv_cache_dtype() == (ttnn.bfloat8_b if value == "bfp8" else ttnn.bfloat16)


@pytest.mark.parametrize("dtype_name", ["bf16", "bfp8"])
def test_optimized_kv_cache_oracle_propagation_host(monkeypatch, expect_error, dtype_name):
    """Exercise allocation, entry validation and upload without opening a device."""
    monkeypatch.setenv("GEMMA4_OPT_KV_CACHE_DTYPE", dtype_name)
    cache_dtype = _requested_kv_cache_dtype()
    copies = []

    def fake_tensor(mesh, tensor, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return SimpleNamespace(
            shape=tuple(tensor.shape),
            padded_shape=tuple(tensor.shape),
            dtype=dtype,
            layout=layout,
            buffer_address=lambda: id(tensor),
        )

    class Oracle:
        layer_kind = functional_tests.SLIDING_KIND

        def prefill_forward(self, hidden_states, **kwargs):
            return hidden_states

        decode_forward = prefill_forward

    monkeypatch.setattr(functional_tests, "_as_tt", fake_tensor)
    monkeypatch.setattr(mutable_tests, "_host_tt", fake_tensor)
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda host, target: copies.append((host, target)))
    _install_requested_kv_cache_dtype(monkeypatch, decoder_cls=Oracle)
    assert functional_tests._as_tt is mutable_tests._as_tt
    payload = {
        "hidden_states": torch.zeros(1, 1, 1, functional_tests.HIDDEN_SIZE),
        "position_cos": torch.zeros(1, 1, 1, functional_tests.SLIDING_HEAD_DIM),
        "position_sin": torch.zeros(1, 1, 1, functional_tests.SLIDING_HEAD_DIM),
        "current_pos": torch.tensor([33], dtype=torch.int32),
        "page_table": torch.tensor([[1, 0]], dtype=torch.int32),
        "key_cache": torch.zeros(2, 8, 64, 256),
        "value_cache": torch.ones(2, 8, 64, 256),
    }
    stable = mutable_tests._device_args(None, payload)
    assert stable["hidden_states"].dtype == stable["position_cos"].dtype == ttnn.bfloat16
    assert all(cache.dtype == cache_dtype for cache in stable["kv_cache"])
    full_cache = functional_tests._as_tt(None, torch.zeros(2, 2, 128, 512))
    assert full_cache.dtype == cache_dtype
    Oracle().decode_forward(**stable)
    observation = _KV_CACHE_EVIDENCE["cache_descriptors"][0]
    assert observation["descriptor"]["key"]["dtype"] == str(cache_dtype)
    assert observation["entry_positions"] == [[33]]
    hosts = mutable_tests._copy_payload_to_stable(None, payload, stable)
    assert len(hosts) == len(payload) == len(copies)
    assert all(host.dtype == target.dtype and host.layout == target.layout for host, target in copies)
    assert hosts[-1].dtype == cache_dtype
    staged = _stage_payload_to_stable(None, payload, stable)
    with monkeypatch.context() as no_staging:

        def unexpected_staging(*args, **kwargs):
            raise AssertionError("replay must reuse preallocated host buffers")

        no_staging.setattr(mutable_tests, "_host_tt", unexpected_staging)
        _copy_staged_payload_to_stable(staged, stable)
    assert len(copies) == 2 * len(payload)
    stable["kv_cache"][0].dtype = ttnn.float32
    with expect_error(AssertionError, "KV cache dtype mismatch"):
        Oracle().decode_forward(**stable)


def test_optimized_material_paths_are_not_functional_fallbacks():
    material_methods = (
        "_prefill_forward_single_user",
        "decode_forward",
        "_attention_prefill",
        "_attention_decode",
        "_dense_mlp",
        "_moe_decode",
        "_moe_prefill",
        "_moe_prefill_chunk",
        "_moe_decode_single_user",
    )
    for name in material_methods:
        optimized_method = inspect.getattr_static(OptimizedDecoder, name)
        functional_method = inspect.getattr_static(FunctionalDecoder, name)
        assert optimized_method is not functional_method, name
        assert optimized_method.__module__.endswith(".optimized_decoder"), name


def test_optimized_hot_path_fallback_audit():
    forbidden = ("torch.", "import torch", "ttnn.from_torch", "ttnn.to_torch")
    methods = (
        OptimizedDecoder._attention_prefill,
        OptimizedDecoder._attention_decode,
        OptimizedDecoder._dense_mlp,
        OptimizedDecoder._prefill_forward_single_user,
        OptimizedDecoder.decode_forward,
        OptimizedDecoder._decode_forward_interleaved,
        OptimizedDecoder._residual_rms_norm,
        OptimizedDecoder._dense_mlp_residual_sharded,
        OptimizedDecoder._router_weights,
        OptimizedDecoder._router_weights_sharded,
        OptimizedDecoder._router_weights_from_normalized,
        OptimizedDecoder._shared_ffn_input,
        OptimizedDecoder._final_residual,
        OptimizedDecoder._moe_decode,
        OptimizedDecoder._moe_prefill,
        OptimizedDecoder._moe_prefill_chunk,
        OptimizedDecoder._moe_decode_single_user,
        OptimizedDecoder._packed_expert_activation,
    )
    source = "\n".join(inspect.getsource(method) for method in methods)
    for token in forbidden:
        assert token not in source


def test_optimized_decode_uses_distinct_paged_cache_view_contracts_host():
    source = inspect.getsource(OptimizedDecoder._attention_decode)

    # Cache writes accept the flat geometry fields returned by cache_view.
    assert source.count("**cache_view,") == 2
    # Paged SDPA accepts the atomic PagedCacheGeometryOverride form instead.
    assert source.count("**self._sdpa_cache_view_kwargs(cache_position_modulo=cache_position_modulo),") == 1


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_direct_fused_optimized_equivalence(mesh_device, device_params, layer_idx):
    """Compare distinct fused/optimized instances, caches, and traced decode."""

    cfg = functional_tests._load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    state = functional_tests._load_layer_state(layer_idx)
    seq_len = 32
    torch.manual_seed(7300 + layer_idx)
    hidden = torch.randn(1, seq_len, functional_tests.HIDDEN_SIZE, dtype=torch.bfloat16)
    decode_hidden = torch.randn(1, 1, functional_tests.HIDDEN_SIZE, dtype=torch.bfloat16)
    rotary = functional_tests.Gemma4TextRotaryEmbedding(cfg)
    cos, sin = rotary(hidden, torch.arange(seq_len).unsqueeze(0), layer_type=layer_type)
    decode_cos, decode_sin = rotary(decode_hidden, torch.tensor([[seq_len]]), layer_type=layer_type)

    def run(decoder_cls):
        decoder = decoder_cls.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
        page_table = functional_tests._as_tt(
            mesh_device,
            functional_tests._page_table(layer_type, shared_physical=False),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        cache_shape = functional_tests._cache_shape(layer_type, shared_physical=False)
        kv_cache = (
            functional_tests._as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
            functional_tests._as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
        )
        prefill = decoder.prefill_forward(
            functional_tests._as_tt(mesh_device, hidden.unsqueeze(1)),
            position_cos=functional_tests._as_tt(mesh_device, cos.unsqueeze(1)),
            position_sin=functional_tests._as_tt(mesh_device, sin.unsqueeze(1)),
            page_table=page_table,
            kv_cache=kv_cache,
        )
        ttnn.synchronize_device(mesh_device)
        prefill_host = functional_tests._to_torch(mesh_device, prefill).reshape(1, seq_len, -1).to(torch.bfloat16)
        decode_args = {
            "hidden_states": functional_tests._as_tt(mesh_device, decode_hidden.unsqueeze(1)),
            "position_cos": functional_tests._as_tt(mesh_device, decode_cos.unsqueeze(1)),
            "position_sin": functional_tests._as_tt(mesh_device, decode_sin.unsqueeze(1)),
            "current_pos": functional_tests._as_tt(
                mesh_device,
                torch.tensor([seq_len], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            "page_table": page_table,
            "kv_cache": kv_cache,
        }
        decoder.decode_forward(**decode_args)
        ttnn.synchronize_device(mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        traced_output = decoder.decode_forward(**decode_args)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        decode_host = functional_tests._to_torch(mesh_device, traced_output).reshape(1, 1, -1).to(torch.bfloat16)
        ttnn.release_trace(mesh_device, trace_id)
        return prefill_host, decode_host, getattr(decoder, "optimized_path_counters", {})

    fused_prefill, fused_decode, _ = run(FusedDecoder)
    optimized_prefill, optimized_decode, counters = run(OptimizedDecoder)
    prefill_ok, prefill_pcc = functional_tests.comp_pcc(fused_prefill, optimized_prefill, 0.995)
    decode_ok, decode_pcc = functional_tests.comp_pcc(fused_decode, optimized_decode, 0.995)
    assert counters["packed_expert_decode"] > 0
    assert counters["packed_expert_prefill"] == 0
    fusion_policy = _resolved_graph_fusion_policy()
    for counter, enabled in fusion_policy.items():
        if enabled:
            assert counters[counter] >= 2, (counter, counters)
    artifact = ARTIFACT_DIR / f"direct_fused_optimized_equivalence_{layer_type}.json"
    artifact.write_text(
        json.dumps(
            {
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "sequence_length": seq_len,
                "distinct_decoder_instances": True,
                "distinct_kv_caches": True,
                "decode_path": "trace_replay",
                "fused_vs_optimized_prefill_pcc": float(prefill_pcc),
                "fused_vs_optimized_decode_pcc": float(decode_pcc),
                "threshold": 0.995,
                "optimized_path_counters": counters,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _stamp_artifact(artifact)
    assert prefill_ok, prefill_pcc
    assert decode_ok, decode_pcc


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,shared_physical,decode_pcc",
    [
        pytest.param(0, True, 0.995, id="sliding_attention_shared_cache"),
        pytest.param(5, False, 0.995, id="full_attention_natural_cache"),
        pytest.param(5, True, 0.995, id="full_attention_shared_cache_view"),
    ],
)
def test_optimized_real_weights_prefill_decode(
    monkeypatch,
    mesh_device,
    device_params,
    layer_idx,
    shared_physical,
    decode_pcc,
):
    pcc_results = []
    original_comp_pcc = functional_tests.comp_pcc

    def comp_pcc_recording(reference, actual, threshold):
        ok, pcc = original_comp_pcc(reference, actual, threshold)
        pcc_results.append({"threshold": float(threshold), "pcc": float(pcc), "passed": bool(ok)})
        # Let the functional oracle finish writing its raw measurement so
        # rejected optimization candidates retain immutable provenance.
        return True, pcc

    monkeypatch.setattr(functional_tests, "comp_pcc", comp_pcc_recording)
    calls = _install_optimized_oracle(
        monkeypatch,
        functional_tests,
        required_methods=(
            "decode_forward",
            "_attention_prefill",
            "_attention_decode",
            "_dense_mlp",
            "_moe_prefill",
            "_moe_prefill_chunk",
            "_moe_decode_single_user",
        ),
    )
    functional_tests.test_functional_decoder_real_weights_prefill_decode(
        mesh_device,
        device_params,
        layer_idx,
        shared_physical,
        decode_pcc,
    )
    assert all(count > 0 for count in calls.values()), calls
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    artifact = _stamp_artifact(ARTIFACT_DIR / f"pcc_layer{layer_idx}_{layer_type}_shared{int(shared_physical)}.json")
    artifact["pcc_results"] = pcc_results
    artifact_path = ARTIFACT_DIR / f"pcc_layer{layer_idx}_{layer_type}_shared{int(shared_physical)}.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    candidate_id = os.getenv("GEMMA4_OPT_CANDIDATE_ID")
    if candidate_id:
        candidate_path = ARTIFACT_DIR / "candidate_runs" / f"{candidate_id}.json"
        candidate = json.loads(candidate_path.read_text())
        candidate[artifact_path.name] = artifact
        candidate_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    assert all(result["passed"] for result in pcc_results), pcc_results


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,shared_physical,logical_seq_len,cache_position_modulo",
    [
        pytest.param(0, True, 1025, 1024, id="sliding_attention_bounded_wrap"),
        pytest.param(5, False, 33, None, id="full_attention_natural_cache"),
        pytest.param(5, True, 33, None, id="full_attention_shared_cache_view"),
    ],
)
@pytest.mark.parametrize(
    "cache_dtype_name",
    [
        pytest.param("bf16", id="kv_bf16"),
        pytest.param(
            "bfp8",
            id="kv_bfp8",
            marks=pytest.mark.skipif(
                os.getenv("GEMMA4_OPT_BFP8_CACHE_CORRECTNESS") != "1",
                reason="BFP8 cache candidate: set GEMMA4_OPT_BFP8_CACHE_CORRECTNESS=1 to run its correctness gates",
            ),
        ),
    ],
)
def test_optimized_nonaligned_prefill_cache_consuming_decode(
    monkeypatch,
    mesh_device,
    device_params,
    layer_idx,
    shared_physical,
    logical_seq_len,
    cache_position_modulo,
    cache_dtype_name,
):
    """Gate the BF16 cache contract and explicitly requested BFP8 candidates.

    The numerical sliding transition starts from a 1025-token prefill. The
    separate bounded-modulo stress oracle covers the 1023/1024/1025 crossing.
    """
    _requested_kv_cache_dtype()
    monkeypatch.setenv("GEMMA4_OPT_KV_CACHE_DTYPE", cache_dtype_name)
    _install_requested_kv_cache_dtype(monkeypatch)
    cache_dtype = _requested_kv_cache_dtype()
    cfg = functional_tests._load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    state = functional_tests._load_layer_state(layer_idx)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
    )
    prefix = f"model.language_model.layers.{layer_idx}."
    reference = functional_tests.Gemma4TextDecoderLayer(cfg, layer_idx=layer_idx).eval().to(dtype=torch.bfloat16)
    reference.load_state_dict({name[len(prefix) :]: value for name, value in state.items()}, strict=True)
    rotary = functional_tests.Gemma4TextRotaryEmbedding(cfg)
    sliding_window = cfg.sliding_window if layer_type == "sliding_attention" else None
    positions = list(range(logical_seq_len, logical_seq_len + 3))
    # Keep at least two physical pages so the full-cache permutation changes
    # the page that is actually read, including for a 33-token prefix.
    cache_capacity = cache_position_modulo or 256
    cache_shape = functional_tests._cache_shape(
        layer_type,
        shared_physical=shared_physical,
        token_capacity=cache_capacity,
    )
    pages_a = torch.arange(cache_shape[0], dtype=torch.int32).view(1, -1)
    pages_b = pages_a.roll(1, dims=1)
    assert not torch.equal(pages_a, pages_b)
    zero_cache = torch.zeros(cache_shape, dtype=torch.bfloat16)
    # Retain the original failing sliding/shared-view inputs, and use the
    # same seed for the natural full-cache control and both cache dtypes.
    seed = 9101 + layer_idx
    torch.manual_seed(seed)
    decode_hidden = torch.randn(1, 3, functional_tests.HIDDEN_SIZE, dtype=torch.bfloat16)
    decode_cos, decode_sin = rotary(decode_hidden, torch.tensor([positions]), layer_type=layer_type)
    decode_payloads = [
        {
            "hidden_states": decode_hidden[:, step : step + 1].unsqueeze(1),
            "position_cos": decode_cos[:, step : step + 1].unsqueeze(1),
            "position_sin": decode_sin[:, step : step + 1].unsqueeze(1),
            "current_pos": torch.tensor([position], dtype=torch.int32),
        }
        for step, position in enumerate(positions)
    ]
    hidden = torch.randn(1, logical_seq_len, functional_tests.HIDDEN_SIZE, dtype=torch.bfloat16)
    prefix_positions = torch.arange(logical_seq_len).unsqueeze(0)
    cos, sin = rotary(hidden, prefix_positions, layer_type=layer_type)
    hf_cache = functional_tests.DynamicCache(config=cfg)
    with torch.no_grad():
        hf_prefill = reference(
            hidden,
            shared_kv_states={},
            position_embeddings=(cos, sin),
            attention_mask=functional_tests._causal_mask(logical_seq_len, sliding_window=sliding_window),
            position_ids=prefix_positions,
            past_key_values=hf_cache,
        )
        hf_decode = []
        for step, position in enumerate(positions):
            # HF evicts sliding history: size the mask for the keys its
            # next update returns, not the absolute sequence position.
            key_length, _ = hf_cache.get_mask_sizes(1, layer_idx)
            hf_decode.append(
                reference(
                    decode_hidden[:, step : step + 1],
                    shared_kv_states={},
                    position_embeddings=(decode_cos[:, step : step + 1], decode_sin[:, step : step + 1]),
                    attention_mask=functional_tests._decode_mask(key_length, sliding_window=sliding_window),
                    position_ids=torch.tensor([[position]]),
                    past_key_values=hf_cache,
                )
            )
    prefill_inputs = {
        "hidden_states": functional_tests._as_tt(mesh_device, hidden.unsqueeze(1)),
        "position_cos": functional_tests._as_tt(mesh_device, cos.unsqueeze(1)),
        "position_sin": functional_tests._as_tt(mesh_device, sin.unsqueeze(1)),
    }

    def fresh_decode_args(page_table):
        payload = {**decode_payloads[0], "page_table": page_table, "key_cache": zero_cache, "value_cache": zero_cache}
        return {**mutable_tests._device_args(mesh_device, payload), "cache_position_modulo": cache_position_modulo}

    stable = fresh_decode_args(pages_a)
    stable_tensors = {
        name: value for name, value in stable.items() if name not in ("kv_cache", "cache_position_modulo")
    }
    stable_tensors.update(key_cache=stable["kv_cache"][0], value_cache=stable["kv_cache"][1])
    addresses = {name: int(value.buffer_address()) for name, value in stable_tensors.items()}
    fill_source_dtypes = []
    tail_source_dtypes = []
    original_fill = ttnn.experimental.paged_fill_cache
    original_update = ttnn.experimental.paged_update_cache

    def checked_fill(cache, source, *args, **kwargs):
        fill_source_dtypes.append(source.dtype)
        assert source.dtype == cache.dtype == cache_dtype
        return original_fill(cache, source, *args, **kwargs)

    def checked_update(cache, source, *args, **kwargs):
        tail_source_dtypes.append(source.dtype)
        assert cache.dtype == cache_dtype
        assert source.dtype == ttnn.bfloat16
        return original_update(cache, source, *args, **kwargs)

    results, replay_outputs = [], []

    def compare(label, expected, actual, threshold, *, variant, position=None):
        finite = bool(torch.isfinite(actual).all())
        passed, pcc = functional_tests.comp_pcc(expected, actual, threshold)
        results.append(
            {
                "comparison": label,
                "variant": variant,
                "position": position,
                "pcc": float(pcc),
                "threshold": threshold,
                "passed": bool(passed) and finite,
                "finite": finite,
            }
        )

    # Build both physical cache mappings through the decoder's actual view,
    # then finish all eager work before any trace owns device buffers. Host
    # snapshots keep their original physical shape; no natural/shared host
    # reshape is used to manufacture a tiled cache reinterpretation.
    cases = []
    for variant, page_table in enumerate((pages_a, pages_b)):
        eager = fresh_decode_args(page_table)
        with monkeypatch.context() as cache_ops:
            cache_ops.setattr(ttnn.experimental, "paged_fill_cache", checked_fill)
            cache_ops.setattr(ttnn.experimental, "paged_update_cache", checked_update)
            prefill_output = decoder.prefill_forward(
                **prefill_inputs,
                page_table=eager["page_table"],
                kv_cache=eager["kv_cache"],
                cache_position_modulo=cache_position_modulo,
            )
        prefill_host = functional_tests._to_torch(mesh_device, prefill_output).reshape(
            1, logical_seq_len, functional_tests.HIDDEN_SIZE
        )
        compare("hf_vs_prefill", hf_prefill, prefill_host, 0.995, variant=variant)
        prefill_output.deallocate(True)
        physical_cache = [functional_tests._to_torch(mesh_device, tensor) for tensor in eager["kv_cache"]]
        assert all(tuple(tensor.shape) == cache_shape for tensor in physical_cache)
        cache_payload = {"page_table": page_table, "key_cache": physical_cache[0], "value_cache": physical_cache[1]}
        staged_cache = _stage_payload_to_stable(mesh_device, cache_payload, stable)
        # Verify the physical snapshot/upload round trip before using it as a
        # trace starting state, including BFP8 packing headers and tiled view.
        _copy_staged_payload_to_stable(staged_cache, stable)
        for role, target, expected in zip(("key", "value"), stable["kv_cache"], physical_cache):
            actual = functional_tests._to_torch(mesh_device, target)
            compare(f"{role}_snapshot_roundtrip", expected, actual, 0.9999, variant=variant)
        if variant == 1:
            compare("page_mapping_prefill", cases[0]["prefill"], prefill_host, 0.9999, variant=variant)
            for role, original, permuted in zip(("key", "value"), cases[0]["physical_cache"], physical_cache):
                compare(
                    f"{role}_logical_page_mapping",
                    original.index_select(0, pages_a.flatten().long()),
                    permuted.index_select(0, pages_b.flatten().long()),
                    0.9999,
                    variant=variant,
                )
        eager_outputs = []
        for step, position in enumerate(positions):
            host_copies = _copy_payload_to_stable(mesh_device, decode_payloads[step], eager)
            eager_output = decoder.decode_forward(**eager)
            eager_host = functional_tests._to_torch(mesh_device, eager_output).reshape(
                1, 1, functional_tests.HIDDEN_SIZE
            )
            eager_output.deallocate(True)
            eager_outputs.append(eager_host)
            compare("hf_vs_eager", hf_decode[step], eager_host, 0.995, variant=variant, position=position)
        cases.append(
            {
                "prefill": prefill_host,
                "physical_cache": physical_cache,
                "staged_cache": staged_cache,
                "eager": eager_outputs,
            }
        )
        for tensor in (
            *eager["kv_cache"],
            *(value for name, value in eager.items() if name not in ("kv_cache", "cache_position_modulo")),
        ):
            tensor.deallocate(True)

    staged_decode = [_stage_payload_to_stable(mesh_device, payload, stable) for payload in decode_payloads]
    _copy_staged_payload_to_stable(cases[0]["staged_cache"], stable)
    _copy_staged_payload_to_stable(staged_decode[0], stable)
    # HF was advanced exactly once per logical token above. Warm/capture
    # rewrite only the first decode row. All prefill, eager calls and host
    # staging allocations are complete before the trace becomes active.
    warm_output = decoder.decode_forward(**stable)
    ttnn.synchronize_device(mesh_device)
    warm_output.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward(**stable)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        for round_index, variant in enumerate((0, 1, 0)):
            _copy_staged_payload_to_stable(cases[variant]["staged_cache"], stable)
            round_outputs = []
            for step, position in enumerate(positions):
                _copy_staged_payload_to_stable(staged_decode[step], stable)
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
                replay = functional_tests._to_torch(mesh_device, traced_output).reshape_as(hf_decode[step])
                round_outputs.append(replay)
                compare("hf_vs_replay", hf_decode[step], replay, 0.995, variant=variant, position=position)
                compare(
                    "eager_vs_replay", cases[variant]["eager"][step], replay, 0.9999, variant=variant, position=position
                )
                if round_index > 0:
                    compare(
                        "repeat_a_replay" if round_index == 2 else "page_mapping_replay",
                        replay_outputs[0][step],
                        replay,
                        0.9999,
                        variant=variant,
                        position=position,
                    )
                assert {name: int(value.buffer_address()) for name, value in stable_tensors.items()} == addresses
            replay_outputs.append(round_outputs)
    finally:
        ttnn.release_trace(mesh_device, trace_id)

    artifact = (
        ARTIFACT_DIR
        / f"cache_{cache_dtype_name}_prefill_trace_{layer_type}_shared{int(shared_physical)}_seq{logical_seq_len}.json"
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "model_id": functional_tests.MODEL_ID,
                "layer_idx": layer_idx,
                "activation_source": "seeded random BF16 hidden states; real checkpoint weights",
                "activation_seed": seed,
                "cache_dtype": str(cache_dtype),
                "logical_history": "identical prefix and decode inputs for both physical page mappings",
                "logical_prefill_length": logical_seq_len,
                "positions": positions,
                "payload_order": ["A", "B", "A"],
                "page_tables": [pages_a.tolist(), pages_b.tolist()],
                "stable_addresses": addresses,
                "trace_capture_count": 1,
                "trace_replay_count": 9,
                "all_eager_and_staging_complete_before_trace": True,
                "bulk_fill_source_dtypes": [str(dtype) for dtype in fill_source_dtypes],
                "tail_update_source_dtypes": [str(dtype) for dtype in tail_source_dtypes],
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _stamp_artifact(artifact)
    assert fill_source_dtypes == [cache_dtype, cache_dtype] * 2
    expected_tail_dtypes = (
        [ttnn.bfloat16, ttnn.bfloat16] * (2 * (logical_seq_len % 32)) if cache_position_modulo else []
    )
    assert tail_source_dtypes == expected_tail_dtypes
    assert all(result["passed"] for result in results), [result for result in results if not result["passed"]]
    assert decoder.optimized_path_counters["prefill_attention"] > 0
    assert decoder.optimized_path_counters["decode_attention"] > 0


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
@pytest.mark.parametrize("batch", [1, 32], ids=["batch1", "batch32"])
def test_optimized_traced_decode_batch_contract(monkeypatch, mesh_device, device_params, layer_idx, batch):
    calls = _install_optimized_oracle(
        monkeypatch,
        functional_tests,
        required_methods=("decode_forward", "_attention_decode", "_dense_mlp", "_moe_decode_single_user"),
    )
    per_user_pcc = []
    best_replay_user = []
    pcc_results = []
    if batch == 32:
        original_comp_pcc = functional_tests.comp_pcc

        def comp_pcc_with_per_user_gate(reference, actual, threshold):
            ok, pcc = original_comp_pcc(reference, actual, threshold)
            pcc_results.append({"threshold": float(threshold), "pcc": float(pcc), "passed": bool(ok)})
            if threshold == 0.995 and reference.shape[0] == batch:
                per_user_pcc.extend(
                    float(original_comp_pcc(reference[index : index + 1], actual[index : index + 1], threshold)[1])
                    for index in range(batch)
                )
                reference_rows = torch.nn.functional.normalize(reference.float().reshape(batch, -1), dim=-1)
                actual_rows = torch.nn.functional.normalize(actual.float().reshape(batch, -1), dim=-1)
                best_replay_user.extend((reference_rows @ actual_rows.T).argmax(dim=1).tolist())
            return True, pcc

        monkeypatch.setattr(functional_tests, "comp_pcc", comp_pcc_with_per_user_gate)
    functional_tests.test_traced_decode_batch_contract(mesh_device, device_params, layer_idx, batch)
    assert all(count > 0 for count in calls.values()), calls
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    artifact = ARTIFACT_DIR / f"trace_{layer_type}_batch{batch}.json"
    if batch == 32:
        if layer_type == "sliding_attention":
            identity_mapping = best_replay_user == list(range(batch))
        else:
            identity_mapping = True
        contents = json.loads(artifact.read_text())
        contents["pcc_results"] = pcc_results
        contents["hf_vs_trace_replay_per_user_pcc"] = per_user_pcc
        contents["hf_vs_trace_replay_min_user_pcc"] = min(per_user_pcc)
        contents["best_replay_user"] = best_replay_user
        artifact.write_text(json.dumps(contents, indent=2, sort_keys=True) + "\n")
    _stamp_artifact(artifact)
    if batch == 32:
        assert all(result["passed"] for result in pcc_results), pcc_results
        assert min(per_user_pcc) >= 0.995, per_user_pcc
        assert identity_mapping, best_replay_user


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention_shared_hma"])
def test_optimized_trace_mutable_stable_buffers(monkeypatch, mesh_device, device_params, layer_idx):
    calls = _install_optimized_oracle(
        monkeypatch,
        mutable_tests,
        required_methods=("decode_forward", "_attention_decode", "_dense_mlp", "_moe_decode_single_user"),
    )
    mutable_tests.test_trace_mutable_stable_buffers(mesh_device, device_params, layer_idx)
    assert all(count > 0 for count in calls.values()), calls
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    _stamp_artifact(ARTIFACT_DIR / f"trace_mutable_buffers_{layer_type}_batch32.json")


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
def test_optimized_bounded_modulo_decode_stress(monkeypatch, mesh_device, device_params):
    """Run the optimized layer through 1,104 traced ring-cache replays."""
    calls = _install_optimized_oracle(
        monkeypatch,
        functional_tests,
        required_methods=("decode_forward", "_attention_decode", "_dense_mlp", "_moe_decode_single_user"),
    )
    functional_tests.test_bounded_modulo_decode_reads_across_wrap(mesh_device, device_params)
    assert all(count > 0 for count in calls.values()), calls
    _stamp_artifact(ARTIFACT_DIR / "bounded_modulo_decode_across_wrap.json")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_real_shape_batch2_prefill(monkeypatch, mesh_device, device_params, layer_idx):
    pcc_results = []
    original_comp_pcc = functional_tests.comp_pcc

    def comp_pcc_recording(reference, actual, threshold):
        ok, pcc = original_comp_pcc(reference, actual, threshold)
        pcc_results.append({"threshold": float(threshold), "pcc": float(pcc), "passed": bool(ok)})
        return True, pcc

    monkeypatch.setattr(functional_tests, "comp_pcc", comp_pcc_recording)
    calls = _install_optimized_oracle(
        monkeypatch,
        functional_tests,
        required_methods=("_attention_prefill", "_dense_mlp", "_moe_prefill", "_moe_prefill_chunk"),
    )
    functional_tests.test_functional_decoder_real_shape_batch2_prefill(mesh_device, device_params, layer_idx)
    assert all(count > 0 for count in calls.values()), calls
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    artifact_path = ARTIFACT_DIR / f"prefill_batch2_layer{layer_idx}_{layer_type}.json"
    artifact = _stamp_artifact(artifact_path)
    artifact["pcc_results"] = pcc_results
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    candidate_id = os.getenv("GEMMA4_OPT_CANDIDATE_ID")
    if candidate_id:
        candidate_path = ARTIFACT_DIR / "candidate_runs" / f"{candidate_id}.json"
        candidate = json.loads(candidate_path.read_text())
        candidate[artifact_path.name] = artifact
        candidate_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    assert all(result["passed"] for result in pcc_results), pcc_results


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_paged_prefill_logical_boundary_lengths(monkeypatch, mesh_device, device_params, layer_idx):
    calls = _install_optimized_oracle(
        monkeypatch,
        functional_tests,
        required_methods=("_attention_prefill", "_dense_mlp", "_moe_prefill", "_moe_prefill_chunk"),
    )
    recorded_pcc = []
    original_comp_pcc = functional_tests.comp_pcc

    def record_all_boundaries(reference, actual, threshold):
        passed, pcc = original_comp_pcc(reference, actual, threshold)
        recorded_pcc.append(float(pcc))
        return True, pcc

    monkeypatch.setattr(functional_tests, "comp_pcc", record_all_boundaries)
    functional_tests.test_paged_prefill_logical_boundary_lengths(mesh_device, device_params, layer_idx)
    assert all(count > 0 for count in calls.values()), calls
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    artifact = ARTIFACT_DIR / f"prefill_boundaries_{layer_type}.json"
    contents = _stamp_artifact(artifact)
    assert len(recorded_pcc) == len(contents["results"])
    assert min(recorded_pcc) >= 0.995, contents["results"]


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_advertised_context_traced_decode(monkeypatch, mesh_device, device_params, layer_idx):
    calls = _install_optimized_oracle(
        monkeypatch,
        functional_tests,
        required_methods=("decode_forward", "_attention_decode", "_dense_mlp", "_moe_decode_single_user"),
    )
    functional_tests.test_advertised_context_traced_decode(mesh_device, device_params, layer_idx)
    assert all(count > 0 for count in calls.values()), calls
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    _stamp_artifact(ARTIFACT_DIR / f"advertised_context_decode_{layer_type}.json")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_prefill_capacity_probe(monkeypatch, mesh_device, device_params, layer_idx):
    calls = _install_optimized_oracle(
        monkeypatch,
        functional_tests,
        required_methods=("_attention_prefill", "_dense_mlp", "_moe_prefill", "_moe_prefill_chunk"),
    )
    functional_tests.test_prefill_capacity_probe(mesh_device, device_params, layer_idx)
    assert all(count > 0 for count in calls.values()), calls
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    seq_len = int(os.getenv("GEMMA4_PREFILL_CAPACITY_LENGTH", "262143"))
    _stamp_artifact(ARTIFACT_DIR / f"prefill_capacity_{layer_type}_{seq_len}.json")


def test_graph_fusion_policy_and_state_folds_are_setup_only(monkeypatch, tmp_path, expect_error):
    selected_defaults = {
        "folded_router_projection": True,
        "shared_ffn_norm": True,
        "folded_expert_scale": True,
        "fused_final_scalar": True,
    }
    for env_name in (
        "GEMMA4_OPT_FOLDED_ROUTER_PROJECTION",
        "GEMMA4_OPT_SHARED_FFN_NORM",
        "GEMMA4_OPT_FOLDED_EXPERT_SCALE",
        "GEMMA4_OPT_FUSED_FINAL_SCALAR",
    ):
        monkeypatch.delenv(env_name, raising=False)
    assert all(_resolved_graph_fusion_policy().values())
    assert _resolved_graph_fusion_policy(**selected_defaults) == selected_defaults
    monkeypatch.setenv("GEMMA4_OPT_FOLDED_ROUTER_PROJECTION", "0")
    monkeypatch.setenv("GEMMA4_OPT_SHARED_FFN_NORM", "false")
    monkeypatch.setenv("GEMMA4_OPT_FOLDED_EXPERT_SCALE", "no")
    monkeypatch.setenv("GEMMA4_OPT_FUSED_FINAL_SCALAR", "off")
    assert not any(_resolved_graph_fusion_policy(**selected_defaults).values())
    monkeypatch.setenv("GEMMA4_OPT_FOLDED_ROUTER_PROJECTION", "yes")
    monkeypatch.setenv("GEMMA4_OPT_SHARED_FFN_NORM", "1")
    monkeypatch.setenv("GEMMA4_OPT_FOLDED_EXPERT_SCALE", "true")
    monkeypatch.setenv("GEMMA4_OPT_FUSED_FINAL_SCALAR", "on")
    assert all(_resolved_graph_fusion_policy(**selected_defaults).values())
    monkeypatch.setenv("GEMMA4_OPT_FUSED_FINAL_SCALAR", "sometimes")
    with expect_error(ValueError, "must be a boolean"):
        _resolved_graph_fusion_policy(**selected_defaults)

    prefix = "layers.0"
    state = {
        f"{prefix}.self_attn.q_proj.weight": torch.zeros(1),
        f"{prefix}.router.scale": torch.tensor([2.0, 3.0, 4.0], dtype=torch.bfloat16),
        f"{prefix}.router.proj.weight": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
        f"{prefix}.pre_feedforward_layernorm.weight": torch.tensor([0.5, 1.0, 1.5], dtype=torch.bfloat16),
        f"{prefix}.mlp.gate_proj.weight": torch.arange(12, dtype=torch.bfloat16).reshape(4, 3),
        f"{prefix}.mlp.up_proj.weight": torch.arange(12, 24, dtype=torch.bfloat16).reshape(4, 3),
        f"{prefix}.pre_feedforward_layernorm_2.weight": torch.tensor([1.5, 1.0, 0.5], dtype=torch.bfloat16),
        f"{prefix}.experts.gate_up_proj": torch.arange(48, dtype=torch.bfloat16).reshape(2, 8, 3),
        f"{prefix}.router.per_expert_scale": torch.tensor([0.5, 2.0], dtype=torch.bfloat16),
        f"{prefix}.experts.down_proj": torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4),
    }
    originals = {name: tensor.clone() for name, tensor in state.items()}
    prepared = _prepare_folded_state_dict(
        state,
        layer_idx=0,
        folded_router_projection=True,
        shared_ffn_norm=True,
        folded_expert_scale=True,
    )
    assert prepared is not state
    assert all(torch.equal(state[name], original) for name, original in originals.items())
    assert torch.equal(
        prepared[f"{prefix}.router.proj.weight"],
        originals[f"{prefix}.router.proj.weight"].float()
        * originals[f"{prefix}.router.scale"].float().unsqueeze(0)
        * (functional_tests.HIDDEN_SIZE**-0.5),
    )
    for suffix in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
        assert torch.equal(
            prepared[f"{prefix}.{suffix}"],
            originals[f"{prefix}.{suffix}"].float()
            * originals[f"{prefix}.pre_feedforward_layernorm.weight"].float().unsqueeze(0),
        )
    assert torch.equal(
        prepared[f"{prefix}.experts.gate_up_proj"],
        originals[f"{prefix}.experts.gate_up_proj"].float()
        * originals[f"{prefix}.pre_feedforward_layernorm_2.weight"].float().reshape(1, 1, -1),
    )
    assert torch.equal(
        prepared[f"{prefix}.experts.down_proj"],
        originals[f"{prefix}.experts.down_proj"].float()
        * originals[f"{prefix}.router.per_expert_scale"].float().reshape(-1, 1, 1),
    )

    # Exercise the algebra consumed by both the prefill and R22 decode graphs,
    # independently of TT hardware and upload-time dtype conversion.
    torch.manual_seed(4401)
    normalized = torch.randn(5, 3)
    hidden_scale = functional_tests.HIDDEN_SIZE**-0.5
    router_before = (normalized * originals[f"{prefix}.router.scale"].float() * hidden_scale) @ originals[
        f"{prefix}.router.proj.weight"
    ].float().T
    router_after = normalized @ prepared[f"{prefix}.router.proj.weight"].T
    torch.testing.assert_close(router_after, router_before)

    for suffix in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
        dense_before = (normalized * originals[f"{prefix}.pre_feedforward_layernorm.weight"].float()) @ originals[
            f"{prefix}.{suffix}"
        ].float().T
        dense_after = normalized @ prepared[f"{prefix}.{suffix}"].T
        torch.testing.assert_close(dense_after, dense_before)

    expert_before = torch.einsum(
        "th,eih->tei",
        normalized * originals[f"{prefix}.pre_feedforward_layernorm_2.weight"].float(),
        originals[f"{prefix}.experts.gate_up_proj"].float(),
    )
    expert_after = torch.einsum(
        "th,eih->tei",
        normalized,
        prepared[f"{prefix}.experts.gate_up_proj"],
    )
    torch.testing.assert_close(expert_after, expert_before)

    expert_activations = torch.randn(5, 2, 4)
    routes = torch.softmax(torch.randn(5, 2), dim=-1)
    down_before = torch.einsum(
        "te,teh->th",
        routes * originals[f"{prefix}.router.per_expert_scale"].float(),
        torch.einsum(
            "tei,ehi->teh",
            expert_activations,
            originals[f"{prefix}.experts.down_proj"].float(),
        ),
    )
    down_after = torch.einsum(
        "te,teh->th",
        routes,
        torch.einsum(
            "tei,ehi->teh",
            expert_activations,
            prepared[f"{prefix}.experts.down_proj"],
        ),
    )
    torch.testing.assert_close(down_after, down_before)

    residual = torch.randn(5, 3)
    branch = torch.randn(5, 3)
    layer_scalar = 0.125
    torch.testing.assert_close((residual + branch) * layer_scalar, torch.add(residual, branch, alpha=1) * layer_scalar)

    assert (
        _prepare_folded_state_dict(
            state,
            layer_idx=0,
            folded_router_projection=False,
            shared_ffn_norm=False,
            folded_expert_scale=False,
        )
        is state
    )
    assert (
        _folded_tensor_cache_path(
            tmp_path,
            folded_router_projection=False,
            shared_ffn_norm=False,
            folded_expert_scale=False,
        )
        == tmp_path
    )
    folded_cache = _folded_tensor_cache_path(
        tmp_path,
        folded_router_projection=True,
        shared_ffn_norm=True,
        folded_expert_scale=True,
    )
    assert folded_cache.parent == tmp_path
    assert folded_cache.name == "optimized_graph_folds_router1_sharedffn1_expertscale1"


def test_graph_fusion_hot_paths_are_owned_and_gated():
    assert OptimizedDecoder._prefill_forward_single_user is not FunctionalDecoder._prefill_forward_single_user
    prefill_source = inspect.getsource(OptimizedDecoder._prefill_forward_single_user)
    decode_source = inspect.getsource(OptimizedDecoder.decode_forward)
    router_source = inspect.getsource(OptimizedDecoder._router_weights_from_normalized)
    final_source = inspect.getsource(OptimizedDecoder._final_residual)

    assert "super()" not in prefill_source
    assert "self._shared_ffn_input" in prefill_source
    assert "self._shared_ffn_input" in decode_source
    assert "self.folded_router_projection" in router_source
    assert "self.folded_expert_scale" in router_source
    assert "MUL_UNARY_SFPU" in final_source
    assert "activations=" in final_source


def test_optimized_precision_defaults():
    signature = inspect.signature(OptimizedDecoder.from_state_dict)
    constructor_signature = inspect.signature(OptimizedDecoder)
    assert signature.parameters["weight_dtype"].default == ttnn.bfloat16
    # None selects the evidence-backed role policy in from_state_dict:
    # sliding=BF16, full=BFP8.
    assert signature.parameters["attention_weight_dtype"].default is None
    assert signature.parameters["qkv_weight_dtype"].default is None
    assert signature.parameters["o_proj_weight_dtype"].default is None
    assert signature.parameters["mlp_weight_dtype"].default == ttnn.bfloat8_b
    assert signature.parameters["mlp_down_weight_dtype"].default is None
    assert signature.parameters["prefill_expert_weight_dtype"].default == ttnn.bfloat8_b
    assert signature.parameters["expert_weight_dtype"].default == ttnn.bfloat8_b
    assert signature.parameters["expert_gate_weight_dtype"].default == ttnn.bfloat4_b
    assert signature.parameters["expert_down_weight_dtype"].default is None
    assert signature.parameters["activation_dtype"].default == ttnn.bfloat16
    assert signature.parameters["attention_math_fidelity"].default == ttnn.MathFidelity.HiFi2
    assert signature.parameters["prefill_full_attention_math_fidelity"].default == ttnn.MathFidelity.HiFi2
    assert signature.parameters["full_attention_math_fidelity"].default == ttnn.MathFidelity.LoFi
    assert signature.parameters["residual_full_attention_math_fidelity"].default == ttnn.MathFidelity.HiFi2
    assert signature.parameters["mlp_math_fidelity"].default == ttnn.MathFidelity.LoFi
    assert signature.parameters["expert_gate_per_core_n"].default == 1
    assert signature.parameters["expert_down_per_core_n"].default == 1
    assert signature.parameters["expert_gate_in0_block_w"].default == 22
    assert signature.parameters["expert_down_in0_block_w"].default == 11
    assert signature.parameters["expert_decode_input_l1"].default is True
    assert signature.parameters["prefill_expert_input_l1"].default is False
    assert signature.parameters["dense_decode_dram_sharded"].default is False
    assert signature.parameters["dram_in0_block_w"].default is None
    assert signature.parameters["dram_workers_per_bank"].default == 1
    assert signature.parameters["dram_sharded_roles"].default == ()
    assert signature.parameters["packed_dense_gate_up"].default is True
    assert signature.parameters["packed_expert_decode_gate_up"].default is True
    assert signature.parameters["packed_expert_prefill_gate_up"].default is False
    assert signature.parameters["folded_router_projection"].default is True
    assert signature.parameters["shared_ffn_norm"].default is True
    assert signature.parameters["folded_expert_scale"].default is True
    assert signature.parameters["fused_final_scalar"].default is True
    for candidate in (
        "folded_router_projection",
        "shared_ffn_norm",
        "folded_expert_scale",
        "fused_final_scalar",
    ):
        assert constructor_signature.parameters[candidate].default is True
    assert signature.parameters["residual_shard_cores"].default == 22
    assert signature.parameters["qkv_working_cores"].default == 22
    assert signature.parameters["qkv_in0_block_w"].default == 2
    assert signature.parameters["qkv_out_subblock_w"].default == 3
    assert signature.parameters["full_o_working_cores"].default == 22
    assert signature.parameters["full_o_in0_block_w"].default == 6
    assert signature.parameters["full_o_out_subblock_w"].default == 4
    assert signature.parameters["attention_allow_padding"].default is True
    assert constructor_signature.parameters["qkv_working_cores"].default == 22
    assert constructor_signature.parameters["full_o_working_cores"].default == 22
    assert constructor_signature.parameters["attention_allow_padding"].default is True
    assert signature.parameters["prefill_expert_chunk_size"].default == 32
    assert signature.parameters["prefill_routed_active"].default is True
    assert signature.parameters["prefill_expert_per_core_n"].default == 2
    assert signature.parameters["prefill_expert_gate_in0_block_w"].default == 44
    assert signature.parameters["prefill_expert_down_in0_block_w"].default == 11
    assert signature.parameters["prefill_expert_tail_per_core_n"].default == 11
    assert signature.parameters["prefill_expert_tail_in0_block_w"].default == 1

    phase_source = inspect.getsource(OptimizedDecoder._use_decode_dram_weight)
    assert 'getattr(self, "_in_decode_forward", False)' in phase_source
    assert "_in_decode_forward = False" in inspect.getsource(OptimizedDecoder.prefill_forward)
    assert "_in_decode_forward = True" in inspect.getsource(OptimizedDecoder.decode_forward)


def _persistent_weight_buffer_accounting(decoder):
    """Include batch-specific copies while counting aliased buffers only once."""
    groups = {
        "decoder_weights": vars(decoder.weights),
        "prefill_expert_weights": {
            "gate_proj": decoder.expert_weights.gate_proj,
            "up_proj": decoder.expert_weights.up_proj,
            "down_proj": decoder.expert_weights.down_proj,
        },
        "packed_weights": {
            "decode_expert_gate_up": decoder.decode_packed_expert_gate_up,
            "decode_expert_gate_up_batch32": decoder.decode_packed_expert_gate_up_batch32,
            "prefill_expert_gate_up": decoder.prefill_packed_expert_gate_up,
            "dense_gate_up": decoder.packed_mlp_gate_up,
        },
        "decode_batch32_expert_weights": {
            "gate_proj": decoder.batch32_expert_gate,
            "up_proj": decoder.batch32_expert_up,
        },
        "decode_attention_candidate_copies": decoder.decode_attention_weights,
        "decode_attention_batch32_copies": decoder.decode_attention_weights_batch32,
        "decode_dram_sharded_copies": decoder.decode_dram_weights,
        "runtime_buffers": {"decode_routing_zero_base": decoder.decode_routing_zero_base},
    }
    seen = set()
    buffers = {}
    group_totals = {}
    for group, tensors in groups.items():
        group_totals[group] = 0
        for name, tensor in tensors.items():
            if tensor is None or not isinstance(tensor, ttnn.Tensor) or not tensor.is_allocated():
                continue
            unique_id = int(tensor.buffer_unique_id())
            num_bytes = int(tensor.buffer_num_pages()) * int(tensor.buffer_page_size())
            assert num_bytes > 0
            duplicate = unique_id in seen
            seen.add(unique_id)
            buffers[f"{group}.{name}"] = {
                "buffer_unique_id": unique_id,
                "bytes": num_bytes,
                "counted_once": not duplicate,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
            if not duplicate:
                group_totals[group] += num_bytes
    return buffers, group_totals


@pytest.mark.parametrize("expert_policy", ["default_packed", "shared_bfp8_packed", "unpacked"])
def test_optimized_persistent_allocation_accounting_host(monkeypatch, expert_policy):
    class FakeTensor:
        def __init__(self, unique_id, num_bytes, dtype=ttnn.bfloat8_b, *, allocated=True):
            self.unique_id = unique_id
            self.num_bytes = num_bytes
            self.dtype = dtype
            self.shape = (1, 1, 32, 32)
            self.allocated = allocated

        def is_allocated(self):
            return self.allocated

        def buffer_unique_id(self):
            assert self.allocated
            return self.unique_id

        def buffer_num_pages(self):
            return self.num_bytes // 32

        def buffer_page_size(self):
            return 32

    monkeypatch.setattr(ttnn, "Tensor", FakeTensor)
    packed = expert_policy != "unpacked"
    shared = expert_policy == "shared_bfp8_packed"
    b1_bytes = (561 if shared else 297) * 1024**2
    b32_bytes = 561 * 1024**2
    b1_packed = FakeTensor(3, b1_bytes, ttnn.bfloat8_b if shared else ttnn.bfloat4_b) if packed else None
    b32_packed = FakeTensor(3 if shared else 4, b32_bytes) if packed else None
    decoder = SimpleNamespace(
        weights=SimpleNamespace(qkv=FakeTensor(1, 2048), o_proj=FakeTensor(2, 4096)),
        expert_weights=SimpleNamespace(gate_proj=None, up_proj=None, down_proj=None),
        decode_packed_expert_gate_up=b1_packed,
        decode_packed_expert_gate_up_batch32=b32_packed,
        prefill_packed_expert_gate_up=None,
        packed_mlp_gate_up=None,
        batch32_expert_gate=None if packed else FakeTensor(5, 8192),
        batch32_expert_up=None if packed else FakeTensor(6, 8192),
        decode_attention_weights={"qkv": FakeTensor(7, 8192), "o_proj": FakeTensor(2, 4096)},
        # Separate wrappers around the same buffer catch accidental object-id deduplication.
        decode_attention_weights_batch32={"qkv": FakeTensor(1, 2048), "o_proj": FakeTensor(2, 4096)},
        decode_dram_weights={},
        decode_routing_zero_base=FakeTensor(8, 256, ttnn.float32),
    )
    if shared:
        # Cumulative folds release these source buffers after B1/B32 packing.
        decoder.batch32_expert_gate = FakeTensor(5, 8192, allocated=False)
        decoder.batch32_expert_up = FakeTensor(6, 8192, allocated=False)
    buffers, totals = _persistent_weight_buffer_accounting(decoder)

    if packed:
        b1 = buffers["packed_weights.decode_expert_gate_up"]
        b32 = buffers["packed_weights.decode_expert_gate_up_batch32"]
        assert (b1["buffer_unique_id"] == b32["buffer_unique_id"]) is shared
        assert b1["counted_once"]
        assert b32["counted_once"] is not shared
        expert_bytes = b1_bytes if shared else b1_bytes + b32_bytes
        assert totals["packed_weights"] == expert_bytes
        assert totals["decode_batch32_expert_weights"] == 0
    else:
        expert_bytes = 2 * 8192
        assert totals["decode_batch32_expert_weights"] == expert_bytes
    for role in ("qkv", "o_proj"):
        assert buffers[f"decoder_weights.{role}"]["counted_once"]
        assert not buffers[f"decode_attention_batch32_copies.{role}"]["counted_once"]
    assert totals["decode_attention_batch32_copies"] == 0
    assert totals["decode_attention_candidate_copies"] == 8192
    assert totals["runtime_buffers"] == 256
    assert sum(totals.values()) == 2048 + 4096 + 8192 + 256 + expert_bytes


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_optimized_persistent_allocation_accounting(mesh_device, device_params, layer_idx):
    """Record every persistent weight buffer introduced or retained by this stage."""

    cfg = functional_tests._load_text_config()
    decoder = OptimizedDecoder.from_state_dict(
        functional_tests._load_layer_state(layer_idx),
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
    )
    obsolete_constants = {
        "router_scale": decoder.folded_router_projection,
        "pre_ff_ln": decoder.shared_ffn_norm,
        "pre_ff_ln_2": decoder.shared_ffn_norm,
        "router_per_expert_scale": decoder.folded_expert_scale,
        "layer_scalar": decoder.fused_final_scalar,
    }
    for name, folded in obsolete_constants.items():
        if folded:
            assert getattr(decoder.weights, name) is None
    cumulative_graph_folds = (
        decoder.folded_router_projection
        and decoder.shared_ffn_norm
        and decoder.folded_expert_scale
        and decoder.fused_final_scalar
    )
    if cumulative_graph_folds and decoder.packed_expert_decode_gate_up:
        assert decoder.decode_packed_expert_gate_up is not None
        assert decoder.weights.expert_gate is None
        assert decoder.weights.expert_up is None

    buffers, group_totals = _persistent_weight_buffer_accounting(decoder)
    if decoder.packed_expert_decode_gate_up:
        b1 = buffers["packed_weights.decode_expert_gate_up"]
        b32 = buffers["packed_weights.decode_expert_gate_up_batch32"]
        shared = decoder.decode_packed_expert_gate_up.dtype == ttnn.bfloat8_b
        assert (b1["buffer_unique_id"] == b32["buffer_unique_id"]) is shared
        assert b1["counted_once"]
        assert b32["counted_once"] is not shared
    for role in ("qkv", "o_proj"):
        b32 = buffers[f"decode_attention_batch32_copies.{role}"]
        assert b32["buffer_unique_id"] == buffers[f"decoder_weights.{role}"]["buffer_unique_id"]
        assert not b32["counted_once"]

    persistent_bytes = sum(group_totals.values())
    layer_type = cfg.layer_types[layer_idx]
    artifact = ARTIFACT_DIR / "persistent_allocation_accounting.json"
    contents = json.loads(artifact.read_text()) if artifact.exists() else {}
    contents[layer_type] = {
        "representative_layer": layer_idx,
        "buffers": buffers,
        "group_totals_bytes": group_totals,
        "persistent_bytes": persistent_bytes,
    }
    if all(layer_type in contents for layer_type in {"sliding_attention", "full_attention"}):
        counts = {name: cfg.layer_types.count(name) for name in {"sliding_attention", "full_attention"}}
        contents["projected_30_layer_decoder_weights"] = {
            "layer_kind_counts": counts,
            "persistent_bytes": sum(contents[name]["persistent_bytes"] * counts[name] for name in counts),
            "scope": "decoder-layer persistent weights only; excludes caller-owned KV cache and transient activations",
        }
    artifact.write_text(json.dumps(contents, indent=2, sort_keys=True) + "\n")
    _stamp_artifact(artifact)


def test_residual_shard_candidate_geometry_and_env(monkeypatch, expect_error):
    assert _residual_shard_geometry(0) is None
    assert _residual_shard_geometry(11) == (11, 1, 256, 8, 4)
    assert _residual_shard_geometry(22) == (11, 2, 128, 4, 4)
    with expect_error(ValueError, "must be one of"):
        _residual_shard_geometry(12)

    monkeypatch.delenv("GEMMA4_OPT_RESIDUAL_SHARD_CORES", raising=False)
    assert _residual_shard_cores_from_env() == 0
    for cores in (0, 11, 22):
        monkeypatch.setenv("GEMMA4_OPT_RESIDUAL_SHARD_CORES", str(cores))
        assert _residual_shard_cores_from_env() == cores
    monkeypatch.setenv("GEMMA4_OPT_RESIDUAL_SHARD_CORES", "33")
    with expect_error(ValueError, "must be one of"):
        _residual_shard_cores_from_env()
    monkeypatch.setenv("GEMMA4_OPT_RESIDUAL_SHARD_CORES", "r11")
    with expect_error(ValueError, "must be one of"):
        _residual_shard_cores_from_env()


def test_attention_candidate_geometry_and_env(monkeypatch, expect_error):
    env_names = (
        "GEMMA4_OPT_QKV_WORKING_CORES",
        "GEMMA4_OPT_QKV_BLOCK_W",
        "GEMMA4_OPT_QKV_OUT_SUBBLOCK_W",
        "GEMMA4_OPT_FULL_O_WORKING_CORES",
        "GEMMA4_OPT_FULL_O_BLOCK_W",
        "GEMMA4_OPT_FULL_O_OUT_SUBBLOCK_W",
        "GEMMA4_OPT_ATTENTION_ALLOW_PADDING",
    )
    for name in env_names:
        monkeypatch.delenv(name, raising=False)

    defaults = _attention_candidate_options_from_env()
    assert defaults == {
        "qkv_working_cores": 22,
        "qkv_in0_block_w": 2,
        "qkv_out_subblock_w": 3,
        "full_o_working_cores": 22,
        "full_o_in0_block_w": 6,
        "full_o_out_subblock_w": 4,
        "attention_allow_padding": True,
    }
    default_full = _resolve_attention_candidate_geometry(
        layer_kind="full_attention",
        q_width=8192,
        qkv_width=10240,
        **defaults,
    )
    assert default_full["qkv"] == {
        "role": "qkv",
        "grid": [11, 2],
        "cores": 22,
        "logical_k": 2816,
        "padded_k": 2816,
        "logical_n": 10240,
        "padded_n": 10560,
        "input_shard_tiles": 4,
        "in0_block_w": 2,
        "per_core_M": 1,
        "per_core_N": 15,
        "out_subblock_h": 1,
        "out_subblock_w": 3,
        "input_padding": 0,
        "output_padding": 320,
    }
    assert default_full["o_proj"]["grid"] == [11, 2]
    assert default_full["o_proj"]["input_shard_tiles"] == 12
    assert default_full["o_proj"]["in0_block_w"] == 6
    assert default_full["o_proj"]["per_core_N"] == 4

    env_values = {
        "GEMMA4_OPT_QKV_WORKING_CORES": "22",
        "GEMMA4_OPT_QKV_BLOCK_W": "2",
        "GEMMA4_OPT_QKV_OUT_SUBBLOCK_W": "3",
        "GEMMA4_OPT_FULL_O_WORKING_CORES": "22",
        "GEMMA4_OPT_FULL_O_BLOCK_W": "6",
        "GEMMA4_OPT_FULL_O_OUT_SUBBLOCK_W": "4",
        "GEMMA4_OPT_ATTENTION_ALLOW_PADDING": "1",
    }
    for name, value in env_values.items():
        monkeypatch.setenv(name, value)
    g22_options = _attention_candidate_options_from_env()
    g22_full = _resolve_attention_candidate_geometry(
        layer_kind="full_attention",
        q_width=8192,
        qkv_width=10240,
        **g22_options,
    )
    assert g22_full["qkv"]["grid"] == [11, 2]
    assert g22_full["qkv"]["padded_k"] == 2816
    assert g22_full["qkv"]["padded_n"] == 10560
    assert g22_full["qkv"]["input_shard_tiles"] == 4
    assert g22_full["qkv"]["in0_block_w"] == 2
    assert g22_full["qkv"]["per_core_N"] == 15
    assert g22_full["qkv"]["out_subblock_w"] == 3
    assert g22_full["o_proj"]["grid"] == [11, 2]
    assert g22_full["o_proj"]["padded_k"] == 8448
    assert g22_full["o_proj"]["padded_n"] == 2816
    assert g22_full["o_proj"]["input_shard_tiles"] == 12
    assert g22_full["o_proj"]["in0_block_w"] == 6
    assert g22_full["o_proj"]["per_core_N"] == 4
    assert g22_full["o_proj"]["out_subblock_w"] == 4

    g22_sliding = _resolve_attention_candidate_geometry(
        layer_kind="sliding_attention",
        q_width=4096,
        qkv_width=8192,
        **g22_options,
    )
    assert g22_sliding["qkv"]["padded_n"] == 8448
    assert g22_sliding["o_proj"]["grid"] == [8, 1]
    assert g22_sliding["o_proj"]["padded_k"] == 4096
    artifact_policy = _resolved_policy()
    assert artifact_policy["attention_runtime_geometry"]["full_attention"] == g22_full
    assert artifact_policy["attention_runtime_geometry"]["sliding_attention"] == g22_sliding
    json.dumps(artifact_policy)

    g32_full = _resolve_attention_candidate_geometry(
        layer_kind="full_attention",
        q_width=8192,
        qkv_width=10240,
        qkv_working_cores=32,
        qkv_in0_block_w=3,
        qkv_out_subblock_w=2,
        full_o_working_cores=32,
        full_o_in0_block_w=8,
        full_o_out_subblock_w=3,
        attention_allow_padding=True,
    )
    assert g32_full["qkv"]["grid"] == [8, 4]
    assert g32_full["qkv"]["padded_k"] == 3072
    assert g32_full["qkv"]["input_shard_tiles"] == 3
    assert g32_full["qkv"]["per_core_N"] == 10
    assert g32_full["o_proj"]["padded_n"] == 3072
    assert g32_full["o_proj"]["in0_block_w"] == 8
    assert g32_full["o_proj"]["per_core_N"] == 3

    with expect_error(ValueError, "needs padding"):
        _resolve_attention_candidate_geometry(
            layer_kind="full_attention",
            q_width=8192,
            qkv_width=10240,
            qkv_working_cores=22,
            attention_allow_padding=False,
        )
    with expect_error(ValueError, "working cores"):
        _resolve_attention_candidate_geometry(
            layer_kind="full_attention", q_width=8192, qkv_width=10240, qkv_working_cores=12
        )
    with expect_error(ValueError, "must divide"):
        _resolve_attention_candidate_geometry(
            layer_kind="full_attention",
            q_width=8192,
            qkv_width=10240,
            qkv_working_cores=22,
            qkv_in0_block_w=3,
            attention_allow_padding=True,
        )
    with expect_error(ValueError, "at most 8"):
        _resolve_attention_candidate_geometry(
            layer_kind="full_attention", q_width=8192, qkv_width=10240, qkv_out_subblock_w=10
        )
    monkeypatch.setenv("GEMMA4_OPT_ATTENTION_ALLOW_PADDING", "sometimes")
    with expect_error(ValueError, "must be a boolean"):
        _attention_candidate_options_from_env()
    monkeypatch.delenv("GEMMA4_OPT_ATTENTION_ALLOW_PADDING")
    monkeypatch.setenv("GEMMA4_OPT_QKV_WORKING_CORES", "g22")
    with expect_error(ValueError, "must be an integer"):
        _attention_candidate_options_from_env()


def test_residual_shard_candidate_has_coherent_static_path():
    decode_source = inspect.getsource(OptimizedDecoder.decode_forward)
    assert "return super().decode_forward" in decode_source
    assert "self._residual_rms_norm" in decode_source
    assert "self._rms_norm" not in decode_source

    attention_source = inspect.getsource(OptimizedDecoder._attention_decode)
    moe_source = inspect.getsource(OptimizedDecoder._moe_decode)
    single_user_source = inspect.getsource(OptimizedDecoder._moe_decode_single_user)
    assert "attention_batch32_runtime_geometry" in attention_source
    assert "use_batch32_policy=True" in moe_source
    assert "decode_packed_expert_gate_up_batch32" in single_user_source
    assert "memory_config=self.residual_memory_config" in decode_source
    assert "ttnn.L1_MEMORY_CONFIG" in decode_source

    norm_source = inspect.getsource(OptimizedDecoder._residual_rms_norm)
    assert "LayerNormShardedMultiCoreProgramConfig" not in norm_source
    assert "program_config=self.residual_norm_program_config" in norm_source
    assert "memory_config=self.residual_memory_config" in norm_source
    assert "DRAM_MEMORY_CONFIG" not in norm_source

    dense_source = inspect.getsource(OptimizedDecoder._dense_mlp_residual_sharded)
    assert "packed_mlp_gate_up" not in dense_source
    assert dense_source.count("ttnn.linear(") == 3
    assert "self.residual_intermediate_memory_config" in dense_source
    assert "memory_config=self.residual_memory_config" in dense_source
    assert "DRAM_MEMORY_CONFIG" not in dense_source

    attention_source = inspect.getsource(OptimizedDecoder._attention_decode)
    prefill_attention_source = inspect.getsource(OptimizedDecoder._attention_prefill)
    assert 'attention_program_configs["qkv"]' in attention_source
    assert 'attention_program_configs["o_proj"]' in attention_source
    assert 'attention_weights["qkv"]' in attention_source
    assert 'attention_weights["o_proj"]' in attention_source
    assert 'self._linear(x, "qkv"' in prefill_attention_source
    assert 'self._linear(attn_out, "o_proj"' in prefill_attention_source
    assert '"attention_qkv_input"' in attention_source
    assert '"attention_output"' in attention_source
    assert "self.attention_runtime_geometry" in attention_source
    assert "_prepare_padded_attention_input" in attention_source
    assert '"attention_qkv_output_slice"' in attention_source
    assert '"attention_o_output_slice"' in attention_source

    assert set(_RESIDUAL_BOUNDARY_COUNTERS) == {
        "residual_entry",
        "attention_qkv_input",
        "attention_sdpa_output",
        "attention_o_input",
        "attention_output",
        "router_input",
        "expert_input",
        "expert_output",
        "residual_exit",
    }


def test_prefill_attention_candidate_geometry_and_env(monkeypatch):
    monkeypatch.delenv("GEMMA4_OPT_PREFILL_ATTENTION_2D", raising=False)
    assert _prefill_attention_options_from_env() == {}

    monkeypatch.setenv("GEMMA4_OPT_PREFILL_ATTENTION_2D", "1")
    monkeypatch.setenv("GEMMA4_OPT_PREFILL_ATTENTION_ROLES", "qkv")
    monkeypatch.setenv("GEMMA4_OPT_PREFILL_QKV_GRID_X", "8")
    monkeypatch.setenv("GEMMA4_OPT_PREFILL_QKV_GRID_Y", "8")
    monkeypatch.setenv("GEMMA4_OPT_PREFILL_QKV_BLOCK_W", "4")
    options = _prefill_attention_options_from_env()
    assert set(options) == {"qkv"}
    assert options["qkv"]["max_rows"] == 1024
    geometry = _prefill_attention_2d_geometry(
        m=1024,
        k=2816,
        n=8192,
        grid_x=options["qkv"]["grid_x"],
        grid_y=options["qkv"]["grid_y"],
        in0_block_w=options["qkv"]["in0_block_w"],
        destination_tiles=4,
    )
    assert geometry["per_core_M"] == 4
    assert geometry["per_core_N"] == 32
    assert geometry["active_cores"] == 64


def test_selected_row_major_routing_is_trace_stable_and_default():
    source = OPTIMIZED_SOURCE.read_text()
    assert '_bool_from_env("GEMMA4_OPT_ROUTING_ROW_MAJOR", True)' in source
    assert "decoder.decode_routing_zero_base = ttnn.zeros(" in source
    assert "self.decode_routing_zero_base if row_major_routing" in source
    assert "routing_weights, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG" in source


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,shared_physical",
    [
        pytest.param(0, True, id="sliding_attention_1024"),
        pytest.param(5, False, id="full_attention_1024"),
    ],
)
def test_optimized_prefill_batch32_perf(monkeypatch, mesh_device, device_params, layer_idx, shared_physical):
    if os.getenv("GEMMA4_OPTIMIZED_PREFILL_BATCH32_PERF") != "1":
        pytest.skip("set GEMMA4_OPTIMIZED_PREFILL_BATCH32_PERF=1 to run batch-32 prefill")

    baseline = os.getenv("GEMMA4_OPTIMIZED_PREFILL_BASELINE") == "1"
    if baseline:
        _install_requested_kv_cache_dtype(monkeypatch, decoder_cls=FusedDecoder)
    calls = (
        {}
        if baseline
        else _install_optimized_oracle(
            monkeypatch,
            functional_tests,
            required_methods=("_attention_prefill", "_dense_mlp", "_moe_prefill", "_moe_prefill_chunk"),
        )
    )
    cfg = functional_tests._load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    state = functional_tests._load_layer_state(layer_idx)
    seq_len = int(os.getenv("GEMMA4_FUNCTIONAL_DECODER_SEQ_LEN", "1024"))
    batch = 32
    torch.manual_seed(1700 + layer_idx)
    one_hidden = torch.randn(1, seq_len, functional_tests.HIDDEN_SIZE, dtype=torch.bfloat16)
    positions = torch.arange(seq_len).unsqueeze(0)
    rotary = functional_tests.Gemma4TextRotaryEmbedding(cfg)
    one_cos, one_sin = rotary(one_hidden, positions, layer_type=layer_type)
    hidden = one_hidden.unsqueeze(1).expand(batch, 1, seq_len, -1)
    cos = one_cos.unsqueeze(1).expand(batch, 1, seq_len, -1)
    sin = one_sin.unsqueeze(1).expand(batch, 1, seq_len, -1)

    decoder_type = FusedDecoder if baseline else OptimizedDecoder
    decoder = decoder_type.from_state_dict(
        state,
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
    )
    one_user_cache_shape = functional_tests._cache_shape(
        layer_type,
        shared_physical=shared_physical,
        token_capacity=seq_len + 1,
    )
    blocks_per_user = one_user_cache_shape[0]
    page_table = functional_tests._as_tt(
        mesh_device,
        torch.arange(batch * blocks_per_user, dtype=torch.int32).view(batch, blocks_per_user),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cache_shape = (batch * blocks_per_user, *one_user_cache_shape[1:])
    kv_cache = (
        functional_tests._as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
        functional_tests._as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
    )
    kwargs = {
        "hidden_states": functional_tests._as_tt(mesh_device, hidden),
        "position_cos": functional_tests._as_tt(mesh_device, cos),
        "position_sin": functional_tests._as_tt(mesh_device, sin),
        "page_table": page_table,
        "kv_cache": kv_cache,
    }
    decoder.prefill_forward(**kwargs)
    ttnn.synchronize_device(mesh_device)
    start = time.perf_counter()
    output = decoder.prefill_forward(**kwargs)
    ttnn.synchronize_device(mesh_device)
    prefill_ms = (time.perf_counter() - start) * 1000
    assert output.shape[0] == batch
    if not baseline:
        assert all(count > 0 for count in calls.values()), calls

    artifact = ARTIFACT_DIR / f"layer{layer_idx}_{layer_type}_seq{seq_len}_batch32_host_timings.json"
    contents = json.loads(artifact.read_text()) if artifact.exists() else {}
    field = "fused_prefill_batch32_host_ms" if baseline else "prefill_batch32_host_ms"
    contents[field] = prefill_ms
    contents["prefill_batch"] = batch
    artifact.write_text(json.dumps(contents, indent=2, sort_keys=True) + "\n")
    _stamp_artifact(artifact)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,shared_physical",
    [
        pytest.param(0, True, id="sliding_attention_1024"),
        pytest.param(5, False, id="full_attention_1024"),
    ],
)
@pytest.mark.parametrize("batch", [1, 32], ids=["batch1", "batch32"])
def test_optimized_decoder_perf_profile(
    monkeypatch,
    mesh_device,
    device_params,
    layer_idx,
    shared_physical,
    batch,
):
    baseline = os.getenv("GEMMA4_OPTIMIZED_PERF_BASELINE") == "1"
    calls = {}
    if baseline:
        _install_requested_kv_cache_dtype(monkeypatch, decoder_cls=FusedDecoder)
        monkeypatch.setattr(functional_tests, "FunctionalDecoder", FusedDecoder)
        monkeypatch.setattr(functional_tests, "ARTIFACT_DIR", ARTIFACT_DIR)
    else:
        calls = _install_optimized_oracle(
            monkeypatch,
            functional_tests,
            required_methods=(
                "decode_forward",
                "_attention_prefill",
                "_attention_decode",
                "_dense_mlp",
                "_moe_prefill",
                "_moe_prefill_chunk",
                "_moe_decode_single_user",
            ),
        )
        if os.getenv("GEMMA4_OPT_DECODE_DEVICE_PROFILE") == "1":
            from tracy import signpost

            original_execute_trace = ttnn.execute_trace
            replay_calls = 0

            def profiled_execute_trace(*args, **kwargs):
                nonlocal replay_calls
                replay_calls += 1
                # The oracle compiles eagerly, captures with cached programs,
                # then executes once before all timing warmups. Delimit that
                # first real replay: it has the final traced topology and no
                # compile/dispatch work inside the measured device window.
                if replay_calls != 1:
                    return original_execute_trace(*args, **kwargs)
                signpost("OPTIMIZED_DECODE_TRACE_REPLAY")
                output = original_execute_trace(*args, **kwargs)
                ttnn.synchronize_device(mesh_device)
                signpost("OPTIMIZED_DECODE_TRACE_REPLAY_END")
                return output

            monkeypatch.setattr(ttnn, "execute_trace", profiled_execute_trace)
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    seq_len = int(os.getenv("GEMMA4_FUNCTIONAL_DECODER_SEQ_LEN", "1024"))
    artifact = ARTIFACT_DIR / f"layer{layer_idx}_{layer_type}_seq{seq_len}_batch{batch}_host_timings.json"
    previous = json.loads(artifact.read_text()) if artifact.exists() else {}
    functional_tests.test_functional_decoder_perf_profile(
        mesh_device,
        device_params,
        layer_idx,
        shared_physical,
        batch,
    )
    measured = json.loads(artifact.read_text())
    if baseline:
        if "prefill_host_ms" in measured:
            measured["fused_prefill_host_ms"] = measured.pop("prefill_host_ms")
        measured["fused_decode_trace_host_ms"] = measured.pop("decode_trace_host_ms")
        measured = {**previous, **measured}
        artifact.write_text(json.dumps(measured, indent=2, sort_keys=True) + "\n")
    else:
        assert all(count > 0 for count in calls.values()), calls
        measured = {**previous, **measured}
        artifact.write_text(json.dumps(measured, indent=2, sort_keys=True) + "\n")
    _stamp_artifact(artifact)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,shared_physical",
    [
        pytest.param(0, True, id="sliding_attention"),
        pytest.param(5, False, id="full_attention"),
    ],
)
def test_serving_batch32_prefill_perf(monkeypatch, mesh_device, device_params, layer_idx, shared_physical):
    """Measure warmed prefill at the context contract's serving batch."""
    if os.getenv("GEMMA4_OPT_SERVING_PREFILL_PERF") != "1":
        pytest.skip("set GEMMA4_OPT_SERVING_PREFILL_PERF=1 to run serving-batch prefill")

    baseline = os.getenv("GEMMA4_OPTIMIZED_PERF_BASELINE") == "1"
    decoder_cls = FusedDecoder if baseline else OptimizedDecoder
    _install_requested_kv_cache_dtype(monkeypatch, decoder_cls=decoder_cls)
    cfg = functional_tests._load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    state = functional_tests._load_layer_state(layer_idx)
    batch = 32
    seq_len = int(os.getenv("GEMMA4_FUNCTIONAL_DECODER_SEQ_LEN", "1024"))
    torch.manual_seed(3200 + layer_idx)
    hidden = torch.randn(batch, seq_len, functional_tests.HIDDEN_SIZE, dtype=torch.bfloat16)
    positions = torch.arange(seq_len).view(1, -1).expand(batch, -1)
    rotary = functional_tests.Gemma4TextRotaryEmbedding(cfg)
    cos, sin = rotary(hidden, positions, layer_type=layer_type)
    decoder = decoder_cls.from_state_dict(
        state,
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
    )
    one_user_cache_shape = functional_tests._cache_shape(
        layer_type,
        shared_physical=shared_physical,
        token_capacity=seq_len + 1,
    )
    blocks_per_user = one_user_cache_shape[0]
    page_table = functional_tests._as_tt(
        mesh_device,
        torch.arange(batch * blocks_per_user, dtype=torch.int32).view(batch, blocks_per_user),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cache_shape = (batch * blocks_per_user, *one_user_cache_shape[1:])
    kv_cache = (
        functional_tests._as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
        functional_tests._as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
    )
    prefill_args = {
        "hidden_states": functional_tests._as_tt(mesh_device, hidden.unsqueeze(1)),
        "position_cos": functional_tests._as_tt(mesh_device, cos.unsqueeze(1)),
        "position_sin": functional_tests._as_tt(mesh_device, sin.unsqueeze(1)),
        "page_table": page_table,
        "kv_cache": kv_cache,
    }
    decoder.prefill_forward(**prefill_args)
    ttnn.synchronize_device(mesh_device)
    start = time.perf_counter()
    output = decoder.prefill_forward(**prefill_args)
    ttnn.synchronize_device(mesh_device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert output.shape == [batch, 1, seq_len, functional_tests.HIDDEN_SIZE]
    if not baseline:
        for counter in ("prefill_attention", "dense_mlp", "expert_prefill"):
            assert decoder.optimized_path_counters[counter] > 0, decoder.optimized_path_counters

    artifact_path = ARTIFACT_DIR / f"layer{layer_idx}_{layer_type}_seq{seq_len}_batch32_prefill_host_timings.json"
    artifact = json.loads(artifact_path.read_text()) if artifact_path.exists() else {}
    key = "fused_prefill_host_ms" if baseline else "prefill_host_ms"
    artifact.update(
        {
            "model_id": functional_tests.MODEL_ID,
            "layer_idx": layer_idx,
            "layer_type": layer_type,
            "batch": batch,
            "sequence_length": seq_len,
            "cache_shape": list(cache_shape),
            key: elapsed_ms,
        }
    )
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    if not baseline:
        _stamp_artifact(artifact_path)
