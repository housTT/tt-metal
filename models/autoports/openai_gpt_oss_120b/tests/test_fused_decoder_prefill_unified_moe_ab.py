# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Focused prefill A/B for the delivered unified routed-expert path.

The test uses one synthetic GPT-OSS-120B layer at the real M=192/E=128/H=2880
shape.  The accepted functional and final fused MLPs receive identical weights
and hidden states, including all GPT-OSS expert biases and SwiGLU-OAI semantics.
"""

import csv
import inspect
import os
import time
from pathlib import Path

import pytest
import torch
from tracy import signpost

import ttnn
from models.autoports.openai_gpt_oss_120b.tests import test_functional_decoder as accepted
from models.autoports.openai_gpt_oss_120b.tests.test_fused_decoder import _install_profile_constructor_drains
from models.autoports.openai_gpt_oss_120b.tt.functional_decoder import _FunctionalMLP
from models.autoports.openai_gpt_oss_120b.tt.fused_decoder import _FusedMLP
from models.common.utility_functions import comp_pcc
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gpt_oss.tt.ccl import CCLManager
from models.demos.gpt_oss.utils.general_utils import get_default_num_links


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_PREFILL_UNIFIED_MOE_AB") != "1",
    reason="set GPT_OSS_120B_PREFILL_UNIFIED_MOE_AB=1 for the focused prefill A/B",
)
@parametrize_mesh_with_fabric([(1, 1)])
def test_prefill_unified_moe_ab(monkeypatch, mesh_device, device_params, reset_seeds):
    """Prove functional equivalence, replay determinism, and warmed speed."""
    del device_params, reset_seeds
    config = accepted._config()
    assert (
        config.hidden_size,
        config.intermediate_size,
        config.num_local_experts,
        config.num_experts_per_tok,
    ) == (2880, 2880, 128, 4)

    reference = accepted._reference_layer(config, layer_idx=0)
    mlp_state = reference.mlp.state_dict()
    expert_state = reference.mlp.experts.state_dict()
    assert config.swiglu_limit == 7.0
    assert torch.count_nonzero(expert_state["gate_up_proj_bias"]) > 0
    assert torch.count_nonzero(expert_state["down_proj_bias"]) > 0
    assert "unified_routed_expert_moe" in inspect.getsource(_FusedMLP._run_chunk)

    mesh_config = MeshConfig(
        mesh_device.shape,
        decode=ModeConfig(tp=1, ep=1, sp=1),
        prefill=ModeConfig(tp=1, ep=1, sp=1),
    )
    ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device))
    functional = _FunctionalMLP(mesh_device, config, mlp_state, ccl_manager, mesh_config, None)

    if os.environ.get("GPT_OSS_120B_PREFILL_UNIFIED_MOE_PROFILE") == "1":
        _install_profile_constructor_drains(monkeypatch, mesh_device)
    fused = _FusedMLP(mesh_device, config, mlp_state, tensor_cache_path=None)

    logical_tokens = 192
    generator = torch.Generator().manual_seed(71_823)
    hidden = torch.randn((1, logical_tokens, config.hidden_size), generator=generator) * 0.02
    with torch.no_grad():
        torch_output = reference.mlp(hidden)[0]

    def make_hidden():
        return ttnn.from_torch(
            hidden.reshape(1, 1, logical_tokens, config.hidden_size),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    functional_output = functional(make_hidden(), is_decode=False)
    ttnn.synchronize_device(mesh_device)
    functional_host = ttnn.to_torch(functional_output)[0, 0, :logical_tokens].clone()
    functional_output.deallocate(True)

    fused_input = make_hidden()
    fused_output = fused(fused_input, is_decode=False)
    ttnn.synchronize_device(mesh_device)
    fused_host = ttnn.to_torch(fused_output)[0, 0, :logical_tokens].clone()
    fused_output.deallocate(True)
    fused_input.deallocate(True)

    direct_passed, direct_detail = comp_pcc(functional_host.float(), fused_host.float(), 0.995)
    functional_torch_passed, functional_torch_detail = comp_pcc(
        torch_output.float(), functional_host.float(), accepted.PREFILL_PCC_THRESHOLD
    )
    fused_torch_passed, fused_torch_detail = comp_pcc(
        torch_output.float(), fused_host.float(), accepted.PREFILL_PCC_THRESHOLD
    )

    # Capture only the final fused class after all lazy shape buffers and
    # programs are warm, then fully release it before any later allocations.
    trace_input = make_hidden()
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = fused(trace_input, is_decode=False)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    first_replay = ttnn.to_torch(traced_output)[0, 0, :logical_tokens].clone()
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    second_replay = ttnn.to_torch(traced_output)[0, 0, :logical_tokens].clone()
    replay_exact = torch.equal(first_replay, second_replay)
    eager_trace_exact = torch.equal(fused_host, first_replay)
    ttnn.release_trace(mesh_device, trace_id)
    traced_output.deallocate(True)
    trace_input.deallocate(True)
    ttnn.synchronize_device(mesh_device)

    repeats = int(os.environ.get("GPT_OSS_120B_PREFILL_UNIFIED_MOE_REPEATS", "3"))
    assert repeats > 0

    functional_inputs = [make_hidden() for _ in range(repeats)]
    functional_outputs = []
    signpost("PREFILL_FUNCTIONAL_MLP_M192")
    started = time.perf_counter()
    for functional_input in functional_inputs:
        functional_outputs.append(functional(functional_input, is_decode=False))
    ttnn.synchronize_device(mesh_device)
    functional_wall_ms = 1000.0 * (time.perf_counter() - started) / repeats
    signpost("PREFILL_FUNCTIONAL_MLP_M192_END")
    for output in functional_outputs:
        output.deallocate(True)

    fused_inputs = [make_hidden() for _ in range(repeats)]
    fused_outputs = []
    signpost("PREFILL_FUSED_UNIFIED_MOE_M192")
    started = time.perf_counter()
    for fused_input in fused_inputs:
        fused_outputs.append(fused(fused_input, is_decode=False))
    ttnn.synchronize_device(mesh_device)
    fused_wall_ms = 1000.0 * (time.perf_counter() - started) / repeats
    signpost("PREFILL_FUSED_UNIFIED_MOE_M192_END")
    for output in fused_outputs:
        output.deallocate(True)
    for fused_input in fused_inputs:
        fused_input.deallocate(True)

    speedup = functional_wall_ms / fused_wall_ms
    print(
        "PREFILL_UNIFIED_MOE_PCC "
        f"functional_torch={functional_torch_detail} fused_torch={fused_torch_detail} "
        f"functional_fused={direct_detail}"
    )
    print("PREFILL_UNIFIED_MOE_TRACE " f"eager_trace_exact={eager_trace_exact} replay_exact={replay_exact}")
    print(
        "PREFILL_UNIFIED_MOE_TIMING "
        f"shape=M{logical_tokens}_E{config.num_local_experts}_H{config.hidden_size}_K{config.num_experts_per_tok} "
        f"repeats={repeats} functional_wall_ms={functional_wall_ms:.6f} "
        f"fused_wall_ms={fused_wall_ms:.6f} speedup={speedup:.6f}"
    )

    artifact_dir = os.environ.get("GPT_OSS_120B_PREFILL_UNIFIED_MOE_ARTIFACT_DIR")
    if artifact_dir:
        path = Path(artifact_dir)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "timing_summary.csv").open("w", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=(
                    "logical_tokens",
                    "num_experts",
                    "hidden_size",
                    "top_k",
                    "repeats",
                    "functional_vs_torch_pcc",
                    "fused_vs_torch_pcc",
                    "functional_vs_fused_pcc",
                    "eager_trace_exact",
                    "replay_exact",
                    "functional_wall_ms",
                    "fused_wall_ms",
                    "speedup",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "logical_tokens": logical_tokens,
                    "num_experts": config.num_local_experts,
                    "hidden_size": config.hidden_size,
                    "top_k": config.num_experts_per_tok,
                    "repeats": repeats,
                    "functional_vs_torch_pcc": f"{functional_torch_detail:.10f}",
                    "fused_vs_torch_pcc": f"{fused_torch_detail:.10f}",
                    "functional_vs_fused_pcc": f"{direct_detail:.10f}",
                    "eager_trace_exact": eager_trace_exact,
                    "replay_exact": replay_exact,
                    "functional_wall_ms": f"{functional_wall_ms:.6f}",
                    "fused_wall_ms": f"{fused_wall_ms:.6f}",
                    "speedup": f"{speedup:.6f}",
                }
            )

    assert functional_torch_passed, f"functional MLP vs Torch failed: {functional_torch_detail}"
    assert fused_torch_passed, f"fused MLP vs Torch failed: {fused_torch_detail}"
    assert direct_passed, f"fused MLP changed functional output materially: {direct_detail}"
    assert eager_trace_exact, "fused eager and first trace replay differ"
    assert replay_exact, "fused prefill trace replay is not bitwise deterministic"
    assert (
        fused_wall_ms < functional_wall_ms
    ), f"fused prefill did not beat functional: {fused_wall_ms:.6f} vs {functional_wall_ms:.6f} ms"
