# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness, path-identity, and profiling gates for the fused decoder.

The functional-stage HF/cache oracles are reused verbatim.  Each wrapper swaps
in :class:`FusedDecoder`, redirects artifacts, and proves that the material
fused methods and counters were entered, so a functional fallback cannot pass.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path

import pytest
import torch

import models.autoports.google_gemma_4_26b_a4b_it.tests.test_functional_decoder as functional_tests
import models.autoports.google_gemma_4_26b_a4b_it.tests.test_trace_mutable_buffers as mutable_tests
import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import FunctionalDecoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.fused_decoder import FusedDecoder, resolved_fusion_policy
from models.common.utility_functions import comp_pcc

ARTIFACT_DIR = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/fused_decoder")
FUSED_SOURCE = Path("models/autoports/google_gemma_4_26b_a4b_it/tt/fused_decoder.py")

_METHOD_COUNTER = {
    "_dense_mlp": "packed_dense_gate_up",
    "_router_weights": "folded_router_projection",
    "_moe_prefill_chunk": "packed_expert_gate_up_prefill",
    "_moe_decode_single_user": "packed_expert_gate_up_decode",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stamp(path: Path, exact_command: str | None = None) -> None:
    contents = json.loads(path.read_text())
    contents["fused_stage_provenance"] = {
        "checkout_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "exact_command": exact_command
        or os.getenv("GEMMA4_FUSED_EXACT_COMMAND", f"pytest -q {os.getenv('PYTEST_CURRENT_TEST', '').split(' ')[0]}"),
        "fused_decoder_sha256": _sha256(FUSED_SOURCE),
        "fused_test_sha256": _sha256(Path(__file__)),
        "environment_overrides": {
            name: value for name, value in sorted(os.environ.items()) if name.startswith("GEMMA4_FUSED_")
        },
        "resolved_fusion_policy": resolved_fusion_policy(),
    }
    path.write_text(json.dumps(contents, indent=2, sort_keys=True) + "\n")
    candidate_id = os.getenv("GEMMA4_FUSED_CANDIDATE_ID")
    if candidate_id:
        candidate_dir = ARTIFACT_DIR / "candidate_runs"
        candidate_dir.mkdir(exist_ok=True)
        candidate_path = candidate_dir / f"{candidate_id}_{path.name}"
        candidate_path.write_text(json.dumps(contents, indent=2, sort_keys=True) + "\n")


def _install_fused_oracle(monkeypatch, module, *, required_methods, require_selected=True):
    calls = {name: 0 for name in required_methods}
    monkeypatch.setattr(module, "FunctionalDecoder", FusedDecoder)
    monkeypatch.setattr(module, "ARTIFACT_DIR", ARTIFACT_DIR)
    for name in required_methods:
        original = getattr(FusedDecoder, name)

        def wrapped(self, *args, __name=name, __original=original, **kwargs):
            before = self.fusion_path_counts.get(_METHOD_COUNTER.get(__name, ""), 0)
            result = __original(self, *args, **kwargs)
            calls[__name] += 1
            counter = _METHOD_COUNTER.get(__name)
            if counter is not None and require_selected:
                assert self.fusion_path_counts[counter] > before, (counter, self.fusion_path_counts)
            if __name == "_dense_mlp" and require_selected and self.use_dense_geglu:
                assert self.fusion_path_counts["dense_geglu"] > 0, self.fusion_path_counts
            if __name == "_dense_mlp" and require_selected and self.use_binary_dense_geglu:
                assert self.fusion_path_counts["binary_dense_geglu"] > 0, self.fusion_path_counts
            if __name in {"_moe_prefill_chunk", "_moe_decode_single_user"} and require_selected:
                assert self.fusion_path_counts["binary_expert_geglu"] > 0, self.fusion_path_counts
            return result

        monkeypatch.setattr(FusedDecoder, name, wrapped)
    return calls


def test_fused_material_paths_are_not_functional_fallbacks():
    for name in (
        "decode_forward",
        "_prefill_forward_single_user",
        "_dense_mlp",
        "_router_weights",
        "_moe_prefill_chunk",
        "_moe_decode_single_user",
    ):
        fused_method = inspect.getattr_static(FusedDecoder, name)
        functional_method = inspect.getattr_static(FunctionalDecoder, name)
        assert fused_method is not functional_method, name
        assert fused_method.__module__.endswith(".fused_decoder"), name


def test_fused_hot_path_has_no_host_or_layout_fallback():
    forbidden = (
        "torch.",
        "import torch",
        "ttnn.from_torch",
        "ttnn.to_torch",
        "ttnn.to_memory_config",
        "ttnn.reshard",
    )
    methods = (
        FusedDecoder.decode_forward,
        FusedDecoder._prefill_forward_single_user,
        FusedDecoder._branch_sum_rms_norm,
        FusedDecoder._dense_mlp,
        FusedDecoder._router_weights,
        FusedDecoder._shared_ffn_input,
        FusedDecoder._final_residual,
        FusedDecoder._packed_expert_activation,
        FusedDecoder._moe_prefill_chunk,
        FusedDecoder._moe_decode_single_user,
    )
    source = "\n".join(inspect.getsource(method) for method in methods)
    for token in forbidden:
        assert token not in source
    # sparse_matmul requires its compact routing mask in ROW_MAJOR layout.
    # This single conversion is a consumer contract, not redundant churn.
    assert source.count("ttnn.to_layout") == 1


def test_selected_fusion_defaults():
    assert resolved_fusion_policy() == {
        "packed_dense_gate_up": True,
        "dense_geglu": False,
        "binary_dense_geglu": True,
        "folded_router_projection": True,
        "fused_branch_norm": False,
        "packed_expert_gate_up": True,
        "expert_geglu": False,
        "binary_expert_geglu": True,
        "folded_expert_scale": True,
        "shared_ffn_norm": True,
        "fused_final_scalar": True,
        "routed_reduction_matmul": False,
        "expert_padded_width": 1536,
        "expert_in0_block_w": 4,
    }


_FUSION_ENV_NAMES = (
    "GEMMA4_FUSED_PACKED_DENSE_GATE_UP",
    "GEMMA4_FUSED_DENSE_GEGLU",
    "GEMMA4_FUSED_BINARY_DENSE_GEGLU",
    "GEMMA4_FUSED_FOLDED_ROUTER",
    "GEMMA4_FUSED_BRANCH_NORM",
    "GEMMA4_FUSED_PACKED_EXPERT_GATE_UP",
    "GEMMA4_FUSED_EXPERT_GEGLU",
    "GEMMA4_FUSED_BINARY_EXPERT_GEGLU",
    "GEMMA4_FUSED_FOLDED_EXPERT_SCALE",
    "GEMMA4_FUSED_SHARED_FFN_NORM",
    "GEMMA4_FUSED_FINAL_SCALAR",
    "GEMMA4_FUSED_ROUTED_REDUCTION",
    "GEMMA4_FUSED_EXPERT_PADDED_WIDTH",
    "GEMMA4_FUSED_EXPERT_IN0_BLOCK_W",
)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
@pytest.mark.parametrize(
    "variant,enabled,required_counts",
    [
        pytest.param(
            "dense_only",
            {
                "GEMMA4_FUSED_PACKED_DENSE_GATE_UP": "1",
                "GEMMA4_FUSED_BINARY_DENSE_GEGLU": "1",
            },
            ("packed_dense_gate_up", "binary_dense_geglu"),
            id="dense_only",
        ),
        pytest.param(
            "router_only",
            {"GEMMA4_FUSED_FOLDED_ROUTER": "1"},
            ("folded_router_projection",),
            id="router_only",
        ),
        pytest.param(
            "expert_only",
            {
                "GEMMA4_FUSED_PACKED_EXPERT_GATE_UP": "1",
                "GEMMA4_FUSED_BINARY_EXPERT_GEGLU": "1",
                "GEMMA4_FUSED_EXPERT_PADDED_WIDTH": "1536",
                "GEMMA4_FUSED_EXPERT_IN0_BLOCK_W": "4",
            },
            ("packed_expert_gate_up_prefill", "packed_expert_gate_up_decode", "binary_expert_geglu"),
            id="expert_only",
        ),
        pytest.param(
            "selected",
            {
                "GEMMA4_FUSED_PACKED_DENSE_GATE_UP": "1",
                "GEMMA4_FUSED_BINARY_DENSE_GEGLU": "1",
                "GEMMA4_FUSED_FOLDED_ROUTER": "1",
                "GEMMA4_FUSED_PACKED_EXPERT_GATE_UP": "1",
                "GEMMA4_FUSED_BINARY_EXPERT_GEGLU": "1",
                "GEMMA4_FUSED_FOLDED_EXPERT_SCALE": "1",
                "GEMMA4_FUSED_SHARED_FFN_NORM": "1",
                "GEMMA4_FUSED_FINAL_SCALAR": "1",
                "GEMMA4_FUSED_EXPERT_PADDED_WIDTH": "1536",
                "GEMMA4_FUSED_EXPERT_IN0_BLOCK_W": "4",
            },
            (
                "packed_dense_gate_up",
                "binary_dense_geglu",
                "folded_router_projection",
                "packed_expert_gate_up_prefill",
                "packed_expert_gate_up_decode",
                "binary_expert_geglu",
                "folded_expert_scale",
                "shared_ffn_norm",
                "fused_final_scalar",
            ),
            id="selected",
        ),
    ],
)
def test_direct_functional_fused_equivalence(
    monkeypatch, mesh_device, device_params, layer_idx, variant, enabled, required_counts
):
    """Compare distinct functional/fused instances and caches, including traced decode."""

    for name in _FUSION_ENV_NAMES:
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("GEMMA4_FUSED_EXPERT_PADDED_WIDTH", "1408")
    monkeypatch.setenv("GEMMA4_FUSED_EXPERT_IN0_BLOCK_W", "1")
    for name, value in enabled.items():
        monkeypatch.setenv(name, value)

    cfg = functional_tests._load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    state = functional_tests._load_layer_state(layer_idx)
    seq_len = 32
    torch.manual_seed(4700 + layer_idx)
    hidden = torch.randn(1, seq_len, functional_tests.HIDDEN_SIZE, dtype=torch.bfloat16)
    decode_hidden = torch.randn(1, 1, functional_tests.HIDDEN_SIZE, dtype=torch.bfloat16)
    rotary = functional_tests.Gemma4TextRotaryEmbedding(cfg)
    cos, sin = rotary(hidden, torch.arange(seq_len).unsqueeze(0), layer_type=layer_type)
    decode_cos, decode_sin = rotary(decode_hidden, torch.tensor([[seq_len]]), layer_type=layer_type)

    def run(decoder_cls):
        decoder = decoder_cls.from_state_dict(
            state,
            hf_config=cfg,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
        )
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
        return prefill_host, decode_host, getattr(decoder, "fusion_path_counts", {})

    functional_prefill, functional_decode, _ = run(FunctionalDecoder)
    fused_prefill, fused_decode, counts = run(FusedDecoder)
    prefill_ok, prefill_pcc = comp_pcc(functional_prefill, fused_prefill, 0.995)
    decode_ok, decode_pcc = comp_pcc(functional_decode, fused_decode, 0.995)
    for counter in required_counts:
        assert counts[counter] > 0, (counter, counts)
    artifact = ARTIFACT_DIR / f"direct_equivalence_{variant}_{layer_type}.json"
    artifact.write_text(
        json.dumps(
            {
                "variant": variant,
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "sequence_length": seq_len,
                "distinct_decoder_instances": True,
                "distinct_kv_caches": True,
                "decode_path": "trace_replay",
                "functional_vs_fused_prefill_pcc": float(prefill_pcc),
                "functional_vs_fused_decode_pcc": float(decode_pcc),
                "threshold": 0.995,
                "required_fusion_counts": {counter: counts[counter] for counter in required_counts},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _stamp(artifact)
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
def test_fused_real_weights_prefill_decode(
    monkeypatch, mesh_device, device_params, layer_idx, shared_physical, decode_pcc
):
    calls = _install_fused_oracle(
        monkeypatch,
        functional_tests,
        required_methods=(
            "decode_forward",
            "_prefill_forward_single_user",
            "_dense_mlp",
            "_router_weights",
            "_branch_sum_rms_norm",
            "_moe_prefill_chunk",
            "_moe_decode_single_user",
        ),
    )
    functional_tests.test_functional_decoder_real_weights_prefill_decode(
        mesh_device, device_params, layer_idx, shared_physical, decode_pcc
    )
    assert all(count > 0 for count in calls.values()), calls
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    _stamp(ARTIFACT_DIR / f"pcc_layer{layer_idx}_{layer_type}_shared{int(shared_physical)}.json")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
@pytest.mark.parametrize("batch", [1, 32], ids=["batch1", "batch32"])
def test_fused_traced_decode_batch_contract(monkeypatch, mesh_device, device_params, layer_idx, batch):
    calls = _install_fused_oracle(
        monkeypatch,
        functional_tests,
        required_methods=(
            "decode_forward",
            "_dense_mlp",
            "_router_weights",
            "_branch_sum_rms_norm",
            "_moe_decode_single_user",
        ),
    )
    functional_tests.test_traced_decode_batch_contract(mesh_device, device_params, layer_idx, batch)
    assert all(count > 0 for count in calls.values()), calls
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    _stamp(ARTIFACT_DIR / f"trace_{layer_type}_batch{batch}.json")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention_shared_hma"])
def test_fused_trace_mutable_stable_buffers(monkeypatch, mesh_device, device_params, layer_idx):
    calls = _install_fused_oracle(
        monkeypatch,
        mutable_tests,
        required_methods=(
            "decode_forward",
            "_dense_mlp",
            "_router_weights",
            "_branch_sum_rms_norm",
            "_moe_decode_single_user",
        ),
    )
    mutable_tests.test_trace_mutable_stable_buffers(mesh_device, device_params, layer_idx)
    assert all(count > 0 for count in calls.values()), calls
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    _stamp(ARTIFACT_DIR / f"trace_mutable_buffers_{layer_type}_batch32.json")


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
def test_fused_bounded_modulo_decode_stress(monkeypatch, mesh_device, device_params):
    """Run the fused layer through 1104 traced ring-cache replays."""
    calls = _install_fused_oracle(
        monkeypatch,
        functional_tests,
        required_methods=("decode_forward", "_dense_mlp", "_router_weights", "_moe_decode_single_user"),
    )
    functional_tests.test_bounded_modulo_decode_reads_across_wrap(mesh_device, device_params)
    assert all(count > 0 for count in calls.values()), calls
    _stamp(ARTIFACT_DIR / "bounded_modulo_decode_across_wrap.json")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_fused_real_shape_batch2_prefill(monkeypatch, mesh_device, device_params, layer_idx):
    calls = _install_fused_oracle(
        monkeypatch,
        functional_tests,
        required_methods=(
            "_prefill_forward_single_user",
            "_dense_mlp",
            "_router_weights",
            "_branch_sum_rms_norm",
            "_moe_prefill_chunk",
        ),
    )
    functional_tests.test_functional_decoder_real_shape_batch2_prefill(mesh_device, device_params, layer_idx)
    assert all(count > 0 for count in calls.values()), calls
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    _stamp(ARTIFACT_DIR / f"prefill_batch2_layer{layer_idx}_{layer_type}.json")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_fused_paged_prefill_logical_boundary_lengths(monkeypatch, mesh_device, device_params, layer_idx):
    calls = _install_fused_oracle(
        monkeypatch,
        functional_tests,
        required_methods=(
            "_prefill_forward_single_user",
            "_dense_mlp",
            "_router_weights",
            "_branch_sum_rms_norm",
            "_moe_prefill_chunk",
        ),
    )
    functional_tests.test_paged_prefill_logical_boundary_lengths(mesh_device, device_params, layer_idx)
    assert all(count > 0 for count in calls.values()), calls
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    _stamp(ARTIFACT_DIR / f"prefill_boundaries_{layer_type}.json")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,shared_physical",
    [
        pytest.param(0, True, id="sliding_attention_1024"),
        pytest.param(5, False, id="full_attention_1024"),
    ],
)
def test_fused_perf_profile(monkeypatch, mesh_device, device_params, layer_idx, shared_physical):
    if os.getenv("GEMMA4_FUSED_DECODER_PERF") != "1":
        pytest.skip("set GEMMA4_FUSED_DECODER_PERF=1 to run the profiler harness")
    monkeypatch.setenv("GEMMA4_FUNCTIONAL_DECODER_PERF", "1")
    calls = _install_fused_oracle(
        monkeypatch,
        functional_tests,
        required_methods=(
            "decode_forward",
            "_prefill_forward_single_user",
            "_dense_mlp",
            "_router_weights",
            "_branch_sum_rms_norm",
            "_moe_prefill_chunk",
            "_moe_decode_single_user",
        ),
        require_selected=False,
    )
    functional_tests.test_functional_decoder_perf_profile(mesh_device, device_params, layer_idx, shared_physical, 1)
    assert all(count > 0 for count in calls.values()), calls
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    path = ARTIFACT_DIR / f"layer{layer_idx}_{layer_type}_seq1024_batch1_host_timings.json"
    _stamp(path)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_fused_advertised_context_traced_decode(monkeypatch, mesh_device, device_params, layer_idx):
    calls = _install_fused_oracle(
        monkeypatch,
        functional_tests,
        required_methods=(
            "decode_forward",
            "_dense_mlp",
            "_router_weights",
            "_branch_sum_rms_norm",
            "_moe_decode_single_user",
        ),
    )
    functional_tests.test_advertised_context_traced_decode(mesh_device, device_params, layer_idx)
    assert all(count > 0 for count in calls.values()), calls
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    _stamp(ARTIFACT_DIR / f"advertised_context_decode_{layer_type}.json")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_fused_prefill_capacity_probe(monkeypatch, mesh_device, device_params, layer_idx):
    calls = _install_fused_oracle(
        monkeypatch,
        functional_tests,
        required_methods=(
            "_prefill_forward_single_user",
            "_dense_mlp",
            "_router_weights",
            "_branch_sum_rms_norm",
            "_moe_prefill_chunk",
        ),
    )
    functional_tests.test_prefill_capacity_probe(mesh_device, device_params, layer_idx)
    assert all(count > 0 for count in calls.values()), calls
    layer_type = functional_tests._load_text_config().layer_types[layer_idx]
    length = int(os.getenv("GEMMA4_PREFILL_CAPACITY_LENGTH", "262143"))
    _stamp(ARTIFACT_DIR / f"prefill_capacity_{layer_type}_{length}.json")
