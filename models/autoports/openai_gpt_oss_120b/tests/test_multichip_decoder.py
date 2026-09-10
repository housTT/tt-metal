# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""P150-family tensor-parallel acceptance for the GPT-OSS 120B decoder.

The default tests are host-side plan and fallback audits. The real-checkpoint
hardware gate is opt-in because it dequantizes one 120B MoE layer and opens up
to four P150 devices. The P150 optimized baseline and P150x2/P150x4 target runs
are deliberately separate processes: overlapping parent/submesh workloads are
not a safe device-lifecycle contract on the four-device P150 system.
"""

from __future__ import annotations

import gc
import inspect
import json
import math
import os
import statistics
import time
import uuid
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from tracy import signpost
from transformers import AutoConfig

import ttnn
from models.autoports.openai_gpt_oss_120b.tests import test_functional_decoder as accepted
from models.autoports.openai_gpt_oss_120b.tests.real_weight_utils import load_real_layer_state_dict
from models.autoports.openai_gpt_oss_120b.tt import multichip_decoder as multichip_decoder_module
from models.autoports.openai_gpt_oss_120b.tt.fused_decoder import _FULL_LOCAL_CHECKPOINT_REVISION
from models.autoports.openai_gpt_oss_120b.tt.multichip_decoder import (
    _DOWN_SUBBLOCK_WIDTH_BY_TP,
    ATTENTION_BFP4_ACTIVATION_CCL_MULTICHIP_POLICY,
    ATTENTION_BFP4_MULTICHIP_POLICY,
    ATTENTION_BFP8_ACTIVATION_CCL_MULTICHIP_POLICY,
    ATTENTION_HIFI2_MULTICHIP_POLICY,
    ATTENTION_HIFI4_MULTICHIP_POLICY,
    ATTENTION_LOFI_MULTICHIP_POLICY,
    BF16_ACTIVATION_CCL_MULTICHIP_POLICY,
    BFP4_ACTIVATION_CCL_MULTICHIP_POLICY,
    BFP8_ACTIVATION_CCL_MULTICHIP_POLICY,
    DEFAULT_MULTICHIP_POLICY,
    DEFAULT_OPTIMIZED_POLICY,
    DENSE_PREFILL_MULTICHIP_POLICY,
    DRAM_SHARDED_OUTPUT_2_CORE_MULTICHIP_POLICY,
    DRAM_SHARDED_OUTPUT_4_CORE_MULTICHIP_POLICY,
    DRAM_SHARDED_OUTPUT_16_CORE_MULTICHIP_POLICY,
    DRAM_SHARDED_OUTPUT_MULTICHIP_POLICY,
    DRAM_SHARDED_QKV_MULTICHIP_POLICY,
    EXPERT_BF16_MULTICHIP_POLICY,
    EXPERT_BFP4_ACTIVATION_CCL_MULTICHIP_POLICY,
    EXPERT_BFP8_ACTIVATION_CCL_MULTICHIP_POLICY,
    EXPERT_BFP8_MULTICHIP_POLICY,
    EXPERT_DOWN_15_CORE_MULTICHIP_POLICY,
    EXPERT_DOWN_18_CORE_MULTICHIP_POLICY,
    EXPERT_DOWN_45_CORE_MULTICHIP_POLICY,
    EXPERT_DOWN_48_CORE_MULTICHIP_POLICY,
    EXPERT_GATE_UP_9_CORE_MULTICHIP_POLICY,
    EXPERT_GATE_UP_15_CORE_MULTICHIP_POLICY,
    EXPERT_GATE_UP_30_CORE_MULTICHIP_POLICY,
    EXPERT_GATE_UP_45_CORE_MULTICHIP_POLICY,
    EXPERT_GATE_UP_45_CORE_TP2_SUBBLOCK2_MULTICHIP_POLICY,
    EXPERT_GATE_UP_NARROW_SUBBLOCK_MULTICHIP_POLICY,
    EXPERT_GATE_UP_WIDE_SUBBLOCK_MULTICHIP_POLICY,
    EXPLICIT_OUTPUT_PROJECTION_MULTICHIP_POLICY,
    FUSED_OUTPUT_CCL_MULTICHIP_POLICY,
    FUSED_ROUTER_MULTICHIP_POLICY,
    INDEXED_PREFILL_MULTICHIP_POLICY,
    PACKED_GROUP_PREFILL_MULTICHIP_POLICY,
    PER_USER_LOOP_DECODE_MULTICHIP_POLICY,
    PREFILL_DOWN_45_CORE_MULTICHIP_POLICY,
    PREFILL_TOKEN_GROUP_SPARSITY_MULTICHIP_POLICY,
    ROUTER_BFP4_MULTICHIP_POLICY,
    ROUTER_BFP8_MULTICHIP_POLICY,
    ROUTER_PREFILL_EXPLICIT_MULTICHIP_POLICY,
    ROUTER_PREFILL_L1_EXPLICIT_MULTICHIP_POLICY,
    ROUTER_PREFILL_L1_MULTICHIP_POLICY,
    SELECTED_EXPERT_GEOMETRY_MULTICHIP_POLICY,
    SEPARATE_GATE_UP_MULTICHIP_POLICY,
    SEPARATE_QKV_MULTICHIP_POLICY,
    SUPPORTED_MESH_SHAPES,
    MultichipDecoder,
    _ActiveExpertTPMLP,
    _allreduce_physical_hidden,
    _PhysicalHiddenCollectiveAttention,
    _ReplicatedL1Router,
    tensor_plan,
)
from models.autoports.openai_gpt_oss_120b.tt.optimized_decoder import OptimizedDecoder
from models.common.utility_functions import comp_pcc
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.tt.ccl import CCLManager
from models.demos.gpt_oss.utils.general_utils import get_default_num_links
from models.demos.utils.trace_region_sizes import TRACE_MODEL_KEY_PARAM

MODEL_CONFIG = Path(__file__).parents[3] / "demos/gpt_oss/configs/gpt-oss-120b"
CONTEXT_CONTRACT = Path(__file__).parents[1] / "doc/context_contract.json"
REAL_WEIGHT_SNAPSHOT = os.environ.get("GPT_OSS_120B_SNAPSHOT")
RUN_ACCEPTANCE = os.environ.get("GPT_OSS_120B_MULTICHIP_ACCEPTANCE") == "1"
RUN_TOPOLOGY_PROBE = os.environ.get("GPT_OSS_120B_MULTICHIP_TOPOLOGY_PROBE") == "1"
RUN_FUSED_OUTPUT_PROBE = os.environ.get("GPT_OSS_120B_MULTICHIP_FUSED_OUTPUT_PROBE") == "1"
RUN_LONG_PREFILL_SDPA_SWEEP = os.environ.get("GPT_OSS_120B_LONG_PREFILL_SDPA_SWEEP") == "1"
RUN_LONG_PREFILL_LAYER_GATE = os.environ.get("GPT_OSS_120B_LONG_PREFILL_LAYER_GATE") == "1"
ARTIFACT_DIR = os.environ.get("GPT_OSS_120B_MULTICHIP_ARTIFACT_DIR")
ACCEPTANCE_RUN_ID = os.environ.get("GPT_OSS_120B_MULTICHIP_RUN_ID")
LONG_PREFILL_ARTIFACT_DIR = os.environ.get("GPT_OSS_120B_LONG_PREFILL_ARTIFACT_DIR")
WRITE_LONG_PREFILL_BASELINE = os.environ.get("GPT_OSS_120B_WRITE_LONG_PREFILL_BASELINE") == "1"
PROCESS_UUID = uuid.uuid4().hex
PREFILL_PCC_THRESHOLD = 0.95
DECODE_PCC_THRESHOLD = 0.95
CACHE_PCC_THRESHOLD = 0.99
ARTIFACT_SCHEMA_VERSION = 3
BATCH2_ARTIFACT_SCHEMA_VERSION = 3
TRACE_REFRESH_STEPS = 2


def _config():
    config = AutoConfig.from_pretrained(MODEL_CONFIG)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    return config


def _rank_to_torch(tensor, rank=0):
    device_tensors = ttnn.get_device_tensors(tensor)
    assert rank < len(device_tensors)
    return ttnn.to_torch(device_tensors[rank]).clone()


def _replicated_from_torch(host, mesh_device, *, dtype, layout):
    return ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _assert_pcc(actual, expected, threshold, label):
    passing, detail = comp_pcc(expected.float(), actual.float(), threshold)
    assert passing, f"{label} failed: {detail}"
    return detail


def _assert_replicated(tensor, logical_shape):
    shards = [_rank_to_torch(tensor, rank) for rank in range(len(ttnn.get_device_tensors(tensor)))]
    assert tuple(tensor.shape) == logical_shape
    reference = shards[0]
    for rank, shard in enumerate(shards[1:], start=1):
        if not torch.equal(shard, reference):
            difference = (shard.float() - reference.float()).abs()
            matching_pcc = comp_pcc(reference.float(), shard.float(), 0.0)[1]
            mismatch_count = int(torch.count_nonzero(difference).item())
            raise AssertionError(
                f"rank {rank} output differs from rank 0: mismatches={mismatch_count} "
                f"max_abs={float(difference.max().item())} pcc={matching_pcc}"
            )
    return reference


def _cache_block(cache, physical_block, rank=0):
    shard = ttnn.get_device_tensors(cache)[rank]
    block = ttnn.slice(
        shard,
        starts=[physical_block, 0, 0, 0],
        ends=[physical_block + 1, shard.shape[1], shard.shape[2], shard.shape[3]],
        steps=[1, 1, 1, 1],
    )
    host = ttnn.to_torch(block).clone()
    block.deallocate(True)
    return host


def _reconstruct_tp_cache_block(cache, physical_block):
    return torch.cat(
        [_cache_block(cache, physical_block, rank) for rank in range(len(ttnn.get_device_tensors(cache)))],
        dim=1,
    )


def _capture_decode(decoder, mesh_device, hidden, rope, current_position, page_table):
    batch_size = int(hidden.shape[-2])
    warm = decoder.decode_forward(
        hidden,
        position_embeddings=rope,
        current_position=current_position,
        page_table=page_table,
        batch_size=batch_size,
    )
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = decoder.decode_forward(
        hidden,
        position_embeddings=rope,
        current_position=current_position,
        page_table=page_table,
        batch_size=batch_size,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    return trace_id, output


def _warmed_trace_latency_ms(mesh_device, trace_id, repeats):
    started = time.perf_counter()
    for _ in range(repeats):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - started) * 1000 / repeats


def _warmed_trace_latency_samples(mesh_device, trace_id, *, signposted=False):
    repeats = int(os.environ.get("GPT_OSS_120B_MULTICHIP_TRACE_REPEATS", "20"))
    sample_count = int(os.environ.get("GPT_OSS_120B_MULTICHIP_TRACE_SAMPLES", "5"))
    if signposted:
        signpost("PERF_DECODE")
    samples = [_warmed_trace_latency_ms(mesh_device, trace_id, repeats) for _ in range(sample_count)]
    if signposted:
        signpost("PERF_DECODE_END")
    return repeats, samples, statistics.median(samples)


@pytest.mark.skipif(
    not RUN_LONG_PREFILL_SDPA_SWEEP,
    reason="set GPT_OSS_120B_LONG_PREFILL_SDPA_SWEEP=1 for the long-prefill SDPA tuning gate",
)
@pytest.mark.parametrize("sliding_window", [128, None], ids=["sliding", "full"])
@pytest.mark.parametrize(
    "q_chunk_size,k_chunk_size",
    [(128, 128), (256, 128), (128, 256), (256, 256), (512, 256), (256, 512)],
    ids=["q128-k128", "q256-k128", "q128-k256", "q256-k256", "q512-k256", "q256-k512"],
)
def test_long_prefill_sdpa_chunk_sweep(
    device,
    reset_seeds,
    q_chunk_size,
    k_chunk_size,
    sliding_window,
):
    """Measure the exact per-TP4-rank GPT-OSS SDPA shape at long context."""
    del reset_seeds
    sequence_length = int(os.environ.get("GPT_OSS_120B_LONG_PREFILL_SEQUENCE", "8192"))
    repeats = int(os.environ.get("GPT_OSS_120B_LONG_PREFILL_SDPA_REPEATS", "10"))
    sample_count = int(os.environ.get("GPT_OSS_120B_LONG_PREFILL_SDPA_SAMPLES", "3"))
    assert sequence_length % math.lcm(q_chunk_size, k_chunk_size, 256) == 0

    generator = torch.Generator().manual_seed(120_8192)
    q_host = torch.randn((1, 16, sequence_length, 64), generator=generator, dtype=torch.bfloat16)
    k_host = torch.randn((1, 2, sequence_length, 64), generator=generator, dtype=torch.bfloat16)
    v_host = torch.randn((1, 2, sequence_length, 64), generator=generator, dtype=torch.bfloat16)
    sink_host = torch.rand((1, 16, 1, 1), generator=generator, dtype=torch.bfloat16) * 32
    q = ttnn.from_torch(q_host, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    k = ttnn.from_torch(k_host, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    v = ttnn.from_torch(v_host, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sink = ttnn.from_torch(sink_host, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    def run(q_chunk, k_chunk):
        return ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            sliding_window_size=sliding_window,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
                exp_approx_mode=False,
            ),
            compute_kernel_config=compute_config,
            attention_sink=sink,
        )

    reference = run(256, 256)
    ttnn.synchronize_device(device)
    reference_host = ttnn.to_torch(reference).float()
    reference.deallocate(True)

    warm = run(q_chunk_size, k_chunk_size)
    ttnn.synchronize_device(device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    output = run(q_chunk_size, k_chunk_size)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    output_host = ttnn.to_torch(output).float()
    passing, detail = comp_pcc(reference_host, output_host, 0.99)
    assert passing, detail

    samples = []
    for _ in range(sample_count):
        started = time.perf_counter()
        for _ in range(repeats):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - started) * 1000 / repeats)
    ttnn.release_trace(device, trace_id)
    print(
        f"LONG_PREFILL_SDPA sequence={sequence_length} sliding_window={sliding_window} "
        f"q_chunk={q_chunk_size} k_chunk={k_chunk_size} median_ms={statistics.median(samples):.6f} "
        f"samples={samples} {detail}"
    )


def _long_prefill_artifact_path(layer_idx, sequence_length):
    assert LONG_PREFILL_ARTIFACT_DIR, "set GPT_OSS_120B_LONG_PREFILL_ARTIFACT_DIR"
    artifact_dir = Path(LONG_PREFILL_ARTIFACT_DIR).resolve()
    workspace_root = Path(__file__).resolve().parents[5]
    assert artifact_dir.is_relative_to(
        workspace_root
    ), f"long-prefill artifacts must remain inside {workspace_root}, got {artifact_dir}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir / f"layer_{layer_idx}_sequence_{sequence_length}.pt"


@pytest.mark.skipif(
    not RUN_LONG_PREFILL_LAYER_GATE or not REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_LONG_PREFILL_LAYER_GATE=1 and GPT_OSS_120B_SNAPSHOT for the long-prefill layer gate",
)
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (1, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "require_exact_physical_num_devices": True,
            },
            id="p150x4",
        )
    ],
    indirect=True,
)
def test_real_weight_long_prefill_layer_gate(mesh_device, device_params, layer_idx, reset_seeds):
    """Compare a long TP4 prefill candidate against a separate dense-policy process."""
    del device_params, reset_seeds
    sequence_length = int(os.environ.get("GPT_OSS_120B_LONG_PREFILL_SEQUENCE", "8192"))
    assert sequence_length >= 2048 and sequence_length % 256 == 0
    artifact_path = _long_prefill_artifact_path(layer_idx, sequence_length)
    candidate_name = os.environ.get("GPT_OSS_120B_MULTICHIP_CANDIDATE", "default")
    if WRITE_LONG_PREFILL_BASELINE:
        assert candidate_name == "dense_prefill", "the long-prefill baseline must use the dense-prefill policy"
    else:
        assert candidate_name in {
            "default",
            "prefill_token_group_sparsity",
        }, "the long-prefill comparison must use the promoted default or prefill_token_group_sparsity policy"

    config = _config()
    snapshot = Path(REAL_WEIGHT_SNAPSHOT)
    assert snapshot.name == _FULL_LOCAL_CHECKPOINT_REVISION
    state_dict = load_real_layer_state_dict(snapshot, layer_idx)
    decoder = _constructor(
        state_dict,
        config,
        layer_idx,
        mesh_device,
        _cache_root(layer_idx, "long_prefill_layer_gate_tp4"),
    )
    if config.layer_types[layer_idx] == "sliding_attention":
        assert decoder.self_attn.program_config.prefill_q_chunk_size_large == 128
        assert decoder.self_attn.program_config.prefill_k_chunk_size_large == 128
    else:
        assert decoder.self_attn.program_config.prefill_q_chunk_size_large == 256
        assert decoder.self_attn.program_config.prefill_k_chunk_size_large == 512

    generator = torch.Generator().manual_seed(120_000 + sequence_length + layer_idx)
    hidden_host = (torch.randn((1, 1, sequence_length, config.hidden_size), generator=generator) * 0.02).to(
        torch.bfloat16
    )
    page_table_host = _host_page_table(config, seed=121_000 + layer_idx)
    hidden, rope, page_table = _prefill_inputs(config, mesh_device, hidden_host, page_table_host)

    warm = decoder.prefill_forward(hidden, position_embeddings=rope, page_table=page_table)
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    started = time.perf_counter()
    output = decoder.prefill_forward(hidden, position_embeddings=rope, page_table=page_table)
    ttnn.synchronize_device(mesh_device)
    wall_ms = (time.perf_counter() - started) * 1000
    output_host = _assert_replicated(
        output,
        (1, 1, sequence_length, config.hidden_size),
    )[0, 0, :sequence_length]

    if WRITE_LONG_PREFILL_BASELINE:
        _save_artifact_atomic(
            {
                "schema_version": 1,
                "producer_process_uuid": PROCESS_UUID,
                "layer_idx": layer_idx,
                "layer_type": config.layer_types[layer_idx],
                "sequence_length": sequence_length,
                "policy": decoder.policy.name,
                "wall_ms": wall_ms,
                "output": output_host,
            },
            artifact_path,
        )
        print(
            f"LONG_PREFILL_LAYER_BASELINE layer={layer_idx} type={config.layer_types[layer_idx]} "
            f"sequence={sequence_length} wall_ms={wall_ms:.6f} artifact={artifact_path}"
        )
    else:
        assert artifact_path.is_file(), f"missing separate-process baseline artifact {artifact_path}"
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
        assert artifact["schema_version"] == 1
        assert artifact["producer_process_uuid"] != PROCESS_UUID
        assert artifact["layer_idx"] == layer_idx
        assert artifact["layer_type"] == config.layer_types[layer_idx]
        assert artifact["sequence_length"] == sequence_length
        detail = _assert_pcc(
            output_host,
            artifact["output"],
            PREFILL_PCC_THRESHOLD,
            "long-prefill candidate",
        )
        speedup = artifact["wall_ms"] / wall_ms
        print(
            f"LONG_PREFILL_LAYER_CANDIDATE layer={layer_idx} type={config.layer_types[layer_idx]} "
            f"sequence={sequence_length} baseline_ms={artifact['wall_ms']:.6f} candidate_ms={wall_ms:.6f} "
            f"speedup={speedup:.6f} {detail} artifact={artifact_path}"
        )
        assert speedup > 1.0, f"long-prefill candidate regressed: speedup={speedup:.6f}"

    output.deallocate(True)
    del decoder, state_dict
    gc.collect()


def _artifact_path(layer_idx):
    assert ARTIFACT_DIR, "set GPT_OSS_120B_MULTICHIP_ARTIFACT_DIR to a unique run directory"
    assert ACCEPTANCE_RUN_ID, "set GPT_OSS_120B_MULTICHIP_RUN_ID to a unique run identifier"
    return Path(ARTIFACT_DIR) / f"{ACCEPTANCE_RUN_ID}_layer_{layer_idx}.pt"


def _batch2_artifact_path(layer_idx):
    assert ARTIFACT_DIR, "set GPT_OSS_120B_MULTICHIP_ARTIFACT_DIR to a unique run directory"
    assert ACCEPTANCE_RUN_ID, "set GPT_OSS_120B_MULTICHIP_RUN_ID to a unique run identifier"
    return Path(ARTIFACT_DIR) / f"{ACCEPTANCE_RUN_ID}_batch2_high_position_layer_{layer_idx}.pt"


def _save_artifact_atomic(artifact, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{PROCESS_UUID}.tmp")
    torch.save(artifact, temporary_path)
    os.replace(temporary_path, path)


def _host_page_table(config, seed):
    blocks = (config.max_position_embeddings + accepted.PAGE_SIZE - 1) // accepted.PAGE_SIZE
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(blocks, generator=generator, dtype=torch.int64).to(torch.int32).reshape(1, blocks)


def _host_batch_page_table(config, batch_size, seed):
    """Return one seeded global allocation with disjoint pages for every user."""
    blocks_per_user = (config.max_position_embeddings + accepted.PAGE_SIZE - 1) // accepted.PAGE_SIZE
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randperm(batch_size * blocks_per_user, generator=generator, dtype=torch.int64)
        .reshape(batch_size, blocks_per_user)
        .to(torch.int32)
    )


def _batch2_high_context_inputs(config, layer_idx):
    """Build exact-limit positions and a page-table-only final trace refresh."""
    batch_size = 2
    blocks_per_user = config.max_position_embeddings // accepted.PAGE_SIZE
    generator = torch.Generator().manual_seed(811 + layer_idx)
    random_blocks = torch.randperm(blocks_per_user // 2 - 1, generator=generator)[:3] + blocks_per_user // 2
    logical_blocks = torch.cat((torch.tensor([blocks_per_user - 1]), random_blocks))
    offsets = torch.tensor([accepted.PAGE_SIZE - 1, 0, 17, 31])
    distinct_positions = (logical_blocks * accepted.PAGE_SIZE + offsets).reshape(batch_size, batch_size)
    position_sets = torch.stack((distinct_positions[0], distinct_positions[1], distinct_positions[1]))
    assert int(position_sets[0, 0]) == config.max_position_embeddings - 1
    assert (distinct_positions // accepted.PAGE_SIZE).unique().numel() == distinct_positions.numel()

    hidden_0 = (torch.randn((1, 1, batch_size, config.hidden_size), generator=generator) * 0.02).to(torch.bfloat16)
    hidden_1 = (torch.randn((1, 1, batch_size, config.hidden_size), generator=generator) * 0.02).to(torch.bfloat16)
    hidden_sets = [hidden_0, hidden_1, hidden_1.clone()]

    page_table_a = _host_batch_page_table(config, batch_size, seed=907 + layer_idx)
    table_a_entries = {int(page_table_a[user, 0]) for user in range(batch_size)} | {
        int(page_table_a[user, int(position // accepted.PAGE_SIZE)])
        for positions in position_sets[:2]
        for user, position in enumerate(positions)
    }
    for attempt in range(100):
        page_table_b = _host_batch_page_table(config, batch_size, seed=1907 + 101 * layer_idx + attempt)
        table_b_entries = {
            int(page_table_b[user, int(position // accepted.PAGE_SIZE)])
            for user, position in enumerate(position_sets[2])
        }
        if table_a_entries.isdisjoint(table_b_entries):
            break
    else:
        raise AssertionError("could not construct disjoint changed-page-table cache probes")
    page_tables = torch.stack((page_table_a, page_table_b))
    step_page_table_ids = torch.tensor([0, 0, 1], dtype=torch.int64)
    return position_sets, hidden_sets, page_tables, step_page_table_ids


def _constructor(
    state_dict,
    config,
    layer_idx,
    mesh_device,
    cache_root,
    *,
    max_batch_size=1,
    optimized_policy=None,
):
    candidate_name = os.environ.get("GPT_OSS_120B_MULTICHIP_CANDIDATE", "default")
    candidate_policies = {
        "default": DEFAULT_MULTICHIP_POLICY,
        "dense_prefill": DENSE_PREFILL_MULTICHIP_POLICY,
        "dram_sharded_qkv": DRAM_SHARDED_QKV_MULTICHIP_POLICY,
        "dram_sharded_output": DRAM_SHARDED_OUTPUT_MULTICHIP_POLICY,
        "dram_sharded_output_16_core": DRAM_SHARDED_OUTPUT_16_CORE_MULTICHIP_POLICY,
        "dram_sharded_output_4_core": DRAM_SHARDED_OUTPUT_4_CORE_MULTICHIP_POLICY,
        "dram_sharded_output_2_core": DRAM_SHARDED_OUTPUT_2_CORE_MULTICHIP_POLICY,
        "explicit_output_projection": EXPLICIT_OUTPUT_PROJECTION_MULTICHIP_POLICY,
        "fused_output_ccl": FUSED_OUTPUT_CCL_MULTICHIP_POLICY,
        "fused_router": FUSED_ROUTER_MULTICHIP_POLICY,
        "prefill_token_group_sparsity": PREFILL_TOKEN_GROUP_SPARSITY_MULTICHIP_POLICY,
        "per_user_loop_decode": PER_USER_LOOP_DECODE_MULTICHIP_POLICY,
        "indexed_prefill": INDEXED_PREFILL_MULTICHIP_POLICY,
        "packed_group_prefill": PACKED_GROUP_PREFILL_MULTICHIP_POLICY,
        "router_bfp8": ROUTER_BFP8_MULTICHIP_POLICY,
        "router_bfp4": ROUTER_BFP4_MULTICHIP_POLICY,
        "router_prefill_explicit": ROUTER_PREFILL_EXPLICIT_MULTICHIP_POLICY,
        "router_prefill_l1": ROUTER_PREFILL_L1_MULTICHIP_POLICY,
        "router_prefill_l1_explicit": ROUTER_PREFILL_L1_EXPLICIT_MULTICHIP_POLICY,
        "activation_ccl_bf16": BF16_ACTIVATION_CCL_MULTICHIP_POLICY,
        "activation_ccl_bfp8": BFP8_ACTIVATION_CCL_MULTICHIP_POLICY,
        "activation_ccl_bfp4": BFP4_ACTIVATION_CCL_MULTICHIP_POLICY,
        "attention_activation_ccl_bfp8": ATTENTION_BFP8_ACTIVATION_CCL_MULTICHIP_POLICY,
        "expert_activation_ccl_bfp8": EXPERT_BFP8_ACTIVATION_CCL_MULTICHIP_POLICY,
        "attention_activation_ccl_bfp4": ATTENTION_BFP4_ACTIVATION_CCL_MULTICHIP_POLICY,
        "expert_activation_ccl_bfp4": EXPERT_BFP4_ACTIVATION_CCL_MULTICHIP_POLICY,
        "attention_bfp4": ATTENTION_BFP4_MULTICHIP_POLICY,
        "attention_lofi": ATTENTION_LOFI_MULTICHIP_POLICY,
        "attention_hifi2": ATTENTION_HIFI2_MULTICHIP_POLICY,
        "attention_hifi4": ATTENTION_HIFI4_MULTICHIP_POLICY,
        "expert_bfp8": EXPERT_BFP8_MULTICHIP_POLICY,
        "expert_bf16": EXPERT_BF16_MULTICHIP_POLICY,
        "expert_gate_up_subblock2": EXPERT_GATE_UP_WIDE_SUBBLOCK_MULTICHIP_POLICY,
        "expert_gate_up_subblock1": EXPERT_GATE_UP_NARROW_SUBBLOCK_MULTICHIP_POLICY,
        "expert_gate_up_30_core": EXPERT_GATE_UP_30_CORE_MULTICHIP_POLICY,
        "expert_down_48_core": EXPERT_DOWN_48_CORE_MULTICHIP_POLICY,
        "expert_gate_up_15_core": EXPERT_GATE_UP_15_CORE_MULTICHIP_POLICY,
        "expert_down_45_core": EXPERT_DOWN_45_CORE_MULTICHIP_POLICY,
        "expert_gate_up_9_core": EXPERT_GATE_UP_9_CORE_MULTICHIP_POLICY,
        "expert_gate_up_45_core": EXPERT_GATE_UP_45_CORE_MULTICHIP_POLICY,
        "expert_gate_up_45_core_tp2_subblock2": EXPERT_GATE_UP_45_CORE_TP2_SUBBLOCK2_MULTICHIP_POLICY,
        "expert_down_15_core": EXPERT_DOWN_15_CORE_MULTICHIP_POLICY,
        "expert_down_18_core": EXPERT_DOWN_18_CORE_MULTICHIP_POLICY,
        "prefill_down_45_core": PREFILL_DOWN_45_CORE_MULTICHIP_POLICY,
        "selected_expert_geometry": SELECTED_EXPERT_GEOMETRY_MULTICHIP_POLICY,
        "separate_qkv": SEPARATE_QKV_MULTICHIP_POLICY,
        "separate_gate_up": SEPARATE_GATE_UP_MULTICHIP_POLICY,
    }
    if candidate_name not in candidate_policies:
        raise ValueError(f"unknown GPT_OSS_120B_MULTICHIP_CANDIDATE={candidate_name!r}")
    return MultichipDecoder.from_state_dict(
        state_dict=state_dict,
        hf_config=config,
        mesh_device=mesh_device,
        layer_idx=layer_idx,
        max_batch_size=max_batch_size,
        max_context_length=config.max_position_embeddings,
        page_size=accepted.PAGE_SIZE,
        tensor_cache_path=cache_root,
        calibrated_checkpoint_revision=_FULL_LOCAL_CHECKPOINT_REVISION,
        policy=candidate_policies[candidate_name],
        optimized_policy=optimized_policy,
    )


def _prefill_inputs(config, mesh_device, hidden_host, page_table_host):
    positions = torch.arange(hidden_host.shape[-2], dtype=torch.long)
    return (
        _replicated_from_torch(hidden_host, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        accepted._rope_tensors(config, mesh_device, positions, decode=False),
        _replicated_from_torch(
            page_table_host,
            mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
    )


def _decode_inputs(config, mesh_device, hidden_host, position_host):
    return (
        _replicated_from_torch(hidden_host, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        accepted._rope_tensors(config, mesh_device, position_host.to(torch.long), decode=True),
        _replicated_from_torch(
            position_host.to(torch.int32),
            mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
    )


def _decode_rope_host(config, position_host):
    rope_scaling = accepted.rope_scaling_model_factory(config.rope_scaling)
    theta = getattr(config, "rope_theta", None) or getattr(config, "default_theta", 150000.0)
    cos_cache, sin_cache = accepted.compute_gather_cos_sin(
        dhead=config.head_dim,
        end=2 * config.max_position_embeddings,
        theta=theta,
        rope_scaling=rope_scaling,
    )
    positions = position_host.to(torch.long).tolist()
    return (
        cos_cache[:, :, positions, :].permute(0, 2, 1, 3),
        sin_cache[:, :, positions, :].permute(0, 2, 1, 3),
    )


def _replicated_host_tensor(host, mesh_device, *, dtype, layout):
    return ttnn.from_torch(
        host,
        device=None,
        dtype=dtype,
        layout=layout,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _refresh_decode_trace_inputs(
    config,
    mesh_device,
    *,
    hidden,
    rope,
    current_position,
    page_table,
    hidden_host,
    position_host,
    page_table_host,
):
    """Refresh every mutable decode input while preserving captured addresses."""
    host_hidden = _replicated_host_tensor(
        hidden_host,
        mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    host_rope = tuple(
        _replicated_host_tensor(value, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        for value in _decode_rope_host(config, position_host)
    )
    host_position = _replicated_host_tensor(
        position_host.to(torch.int32),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    host_page_table = _replicated_host_tensor(
        page_table_host,
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    ttnn.copy_host_to_device_tensor(host_hidden, hidden)
    for source, destination in zip(host_rope, rope):
        ttnn.copy_host_to_device_tensor(source, destination)
    ttnn.copy_host_to_device_tensor(host_position, current_position)
    ttnn.copy_host_to_device_tensor(host_page_table, page_table)


def _cache_root(layer_idx, namespace):
    workspace_cache = Path(__file__).resolve().parents[5] / ".cache/gpt_oss_120b_tensor_cache"
    root = Path(os.environ.get("GPT_OSS_120B_TENSOR_CACHE", workspace_cache))
    root = root.resolve()
    workspace_root = Path(__file__).resolve().parents[5]
    assert root.is_relative_to(workspace_root), f"tensor cache must remain inside {workspace_root}, got {root}"
    path = root / f"layer_{layer_idx}" / namespace
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.mark.parametrize(
    "mesh_shape,expected",
    [
        ((1, 1), (2880, 2880, 2880, 64, 8, 5120)),
        ((1, 2), (1440, 2880, 1440, 32, 4, 2560)),
        ((1, 4), (736, 2944, 736, 16, 2, 1280)),
    ],
)
def test_tensor_plan(mesh_shape, expected):
    plan = tensor_plan(mesh_shape, _config())
    assert plan.mesh_shape == mesh_shape
    assert plan.tp == mesh_shape[1]
    assert (
        plan.padded_local_hidden,
        plan.padded_hidden_size,
        plan.padded_local_intermediate_size,
        plan.local_q_heads,
        plan.local_kv_heads,
        plan.local_qkv_width,
    ) == expected


def test_context_contract_covers_every_mesh_without_reducing_decoder_context():
    with CONTEXT_CONTRACT.open(encoding="utf-8") as contract_file:
        contract = json.load(contract_file)["multichip_decoder"]
    assert contract["target_meshes"] == {
        "P150": [1, 1],
        "P150x2": [1, 2],
        "P150x4": [1, 4],
    }
    assert contract["configured_context_length"] == 131072
    assert contract["supported_decoder_layer_context_length"] == 131072
    assert contract["capability_reduction"] is None
    assert contract["kv_cache"]["local_kv_heads"] == {"1": 8, "2": 4, "4": 2}


@pytest.mark.parametrize(
    "actual_tokens,expected_memory_config",
    [
        (32, None),
        (128, ttnn.L1_MEMORY_CONFIG),
        (129, ttnn.DRAM_MEMORY_CONFIG),
        (131072, ttnn.DRAM_MEMORY_CONFIG),
    ],
)
def test_replicated_l1_router_prefill_memory_policy(monkeypatch, actual_tokens, expected_memory_config):
    hidden_states = Mock()
    hidden_states.volume.return_value = actual_tokens * 2880
    router_input = Mock()
    router_logits = Mock()
    expert_indices = object()
    expert_weights = object()

    router = object.__new__(_ReplicatedL1Router)
    router.hidden_dim = 2880
    router.weight = object()
    router.bias = object()
    router.top_k = 4
    router.prefill_input_l1 = True
    router.prefill_program_config = object()
    router.compute_config = object()
    router.softmax_compute_config = object()
    router.use_fused_op = False

    base_call = Mock(return_value=(expert_indices, expert_weights))
    reshape = Mock(return_value=hidden_states)
    to_memory_config = Mock(return_value=router_input)
    linear = Mock(return_value=router_logits)
    topk_router = Mock(return_value=(expert_indices, expert_weights))
    monkeypatch.setattr(_ReplicatedL1Router.__mro__[1], "__call__", base_call)
    monkeypatch.setattr(ttnn, "reshape", reshape)
    monkeypatch.setattr(ttnn, "to_memory_config", to_memory_config)
    monkeypatch.setattr(ttnn, "linear", linear)
    monkeypatch.setattr(multichip_decoder_module, "topk_router", topk_router)

    use_throughput_experts = actual_tokens == 32
    assert router(hidden_states, use_throughput_experts) == (
        expert_indices,
        expert_weights,
    )
    if expected_memory_config is None:
        base_call.assert_called_once_with(hidden_states, use_throughput_experts)
        linear.assert_not_called()
        return

    linear.assert_called_once_with(
        router_input,
        router.weight,
        bias=router.bias,
        memory_config=expected_memory_config,
        program_config=router.prefill_program_config,
        compute_kernel_config=router.compute_config,
    )
    to_memory_config.assert_called_once_with(hidden_states, ttnn.L1_MEMORY_CONFIG)
    topk_router.assert_called_once_with(router_logits, 4, False, router.softmax_compute_config)
    router_input.deallocate.assert_called_once_with(True)
    router_logits.deallocate.assert_called_once_with(True)


def test_runtime_fallback_and_active_expert_audit():
    source = inspect.getsource(MultichipDecoder.from_state_dict)
    mlp_source = inspect.getsource(_ActiveExpertTPMLP)
    assert "OptimizedDecoder.from_state_dict" in source
    assert "FunctionalDecoder" not in source
    assert "self.use_throughput_experts = False" in mlp_source
    assert "self.router(hidden_states" in mlp_source
    assert "_run_indexed_decode" in mlp_source
    assert "is_input_b_sparse=True" in mlp_source
    assert "sparse_matmul" in mlp_source
    assert "super().__init__(" not in mlp_source
    assert "_PackedTPExpertsRuntime" in mlp_source
    assert "ThroughputExperts" not in mlp_source
    assert DEFAULT_MULTICHIP_POLICY.expert_weight_dtype == ttnn.bfloat4_b
    assert DEFAULT_MULTICHIP_POLICY.attention_activation_ccl_dtype == ttnn.bfloat8_b
    assert DEFAULT_MULTICHIP_POLICY.expert_activation_ccl_dtype is None
    assert DEFAULT_MULTICHIP_POLICY.activation_ccl_dtype == ttnn.bfloat16
    assert DEFAULT_MULTICHIP_POLICY.projection_math_fidelity == ttnn.MathFidelity.LoFi
    # 768-wide per-rank gate/up slices give 48 output tiles: a full (6, 8) grid.
    assert DEFAULT_MULTICHIP_POLICY.expert_gate_up_cores == (6, 8)
    assert DEFAULT_MULTICHIP_POLICY.expert_gate_up_subblock_w == 1
    assert DEFAULT_MULTICHIP_POLICY.expert_gate_up_subblock_w_tp2 == 2
    assert DEFAULT_MULTICHIP_POLICY.expert_down_cores == (5, 3)
    assert DEFAULT_MULTICHIP_POLICY.expert_prefill_down_cores == (5, 9)
    assert DEFAULT_MULTICHIP_POLICY.expert_prefill_down_subblock_w == 2
    assert DEFAULT_MULTICHIP_POLICY.expert_prefill_down_cores_tp2 is None
    assert DEFAULT_MULTICHIP_POLICY.expert_prefill_down_subblock_w_tp2 is None
    assert DEFAULT_MULTICHIP_POLICY.decode_dram_sharded_output
    assert not DEFAULT_MULTICHIP_POLICY.decode_dram_sharded_output_tp4
    assert DEFAULT_MULTICHIP_POLICY.decode_dram_sharded_output_input_cores == 16
    assert not DEFAULT_MULTICHIP_POLICY.decode_fused_output_projection_ccl
    assert DEFAULT_MULTICHIP_POLICY.decode_fused_router
    assert DEFAULT_MULTICHIP_POLICY.prefill_token_group_sparsity
    assert DEFAULT_MULTICHIP_POLICY.prefill_sliding_q_chunk_size_large == 128
    assert DEFAULT_MULTICHIP_POLICY.prefill_sliding_k_chunk_size_large == 128
    assert DEFAULT_MULTICHIP_POLICY.prefill_full_q_chunk_size_large == 256
    assert DEFAULT_MULTICHIP_POLICY.prefill_full_k_chunk_size_large == 512
    assert "expert_weight_dtype=policy.expert_weight_dtype" in source
    attention_source = inspect.getsource(_PhysicalHiddenCollectiveAttention)
    assert "_allreduce_physical_hidden" in attention_source
    assert "if not is_decode" in attention_source
    assert MultichipDecoder.optimization_manifest[-1] == "replicated_decode_l1_prefill_dram_stack_residual_contract"
    assert DEFAULT_MULTICHIP_POLICY.residual_layout == "replicated"
    assert SUPPORTED_MESH_SHAPES == ((1, 1), (1, 2), (1, 4))
    assert _DOWN_SUBBLOCK_WIDTH_BY_TP == {2: 1, 4: 3}


@pytest.mark.skipif(
    not RUN_TOPOLOGY_PROBE,
    reason="set GPT_OSS_120B_MULTICHIP_TOPOLOGY_PROBE=1 for the residual-boundary device probe",
)
@pytest.mark.timeout(900)
@pytest.mark.parametrize("logical_tp", [4], ids=["tp4"])
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (1, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "require_exact_physical_num_devices": True,
                TRACE_MODEL_KEY_PARAM: "gpt-oss-120b",
            },
            id="p150x4-parent",
        )
    ],
    indirect=True,
)
def test_residual_sharded_distributed_norm_fused_qkv_probe(mesh_device, device_params, logical_tp, reset_seeds):
    """Compare the exact TP4 attention and expert collective candidates.

    The attention comparison warms and times all three shape-faithful chains:

      preoptimized: slice 2944 -> 2880 -> DRAM -> L1 -> all-reduce -> norm/QKV;
      selected: all-reduce 2944 -> slice 2880 -> norm/QKV;
      explicit async: reduce-scatter -> all-gather -> slice 2880 -> norm/QKV;
      local row partial -> reduce-scatter -> distributed RMSNorm
      -> fused all-gather + local packed-QKV projection.

    TP4 uses the 2944 physical hidden width required by its 736-wide CCL shard;
    the last 64 zero columns are sliced before the logical comparison. A
    separate expert probe compares its current logical-width reduction against
    runtime zero-padding to 2944.
    """
    del device_params, reset_seeds
    target_mesh = (
        mesh_device
        if logical_tp == 4
        else mesh_device.create_submesh(ttnn.MeshShape(1, logical_tp), offset=ttnn.MeshCoordinate(0, 0))
    )
    config = _config()
    plan = tensor_plan((1, logical_tp), config)
    physical_hidden = plan.padded_hidden_size
    local_qkv_width = plan.local_qkv_width
    collective_dtype = (
        DEFAULT_MULTICHIP_POLICY.attention_activation_ccl_dtype or DEFAULT_MULTICHIP_POLICY.activation_ccl_dtype
    )
    generator = torch.Generator().manual_seed(818_000 + logical_tp)

    partials_host = torch.randn(
        (logical_tp, 1, ttnn.TILE_SIZE, physical_hidden),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    if physical_hidden != config.hidden_size:
        partials_host[..., config.hidden_size :] = 0

    def upload_partials(host):
        return ttnn.from_torch(
            host,
            device=target_mesh,
            dtype=collective_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.create_mesh_mapper(
                target_mesh,
                ttnn.MeshMapperConfig(
                    [ttnn.PlacementReplicate(), ttnn.PlacementShard(0)],
                    ttnn.MeshShape(1, logical_tp),
                ),
            ),
        )

    partials = upload_partials(partials_host)
    expert_partials = upload_partials(partials_host[..., : config.hidden_size].contiguous())
    gamma_current_host = torch.ones(
        (1, 1, config.hidden_size // ttnn.TILE_SIZE, ttnn.TILE_SIZE),
        dtype=torch.bfloat16,
    )
    gamma_current = ttnn.from_torch(
        gamma_current_host,
        device=target_mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(target_mesh),
    )
    gamma_host = torch.ones((1, 1, physical_hidden // ttnn.TILE_SIZE, ttnn.TILE_SIZE), dtype=torch.bfloat16)
    # Distributed RMSNorm observes the explicit TP4 CCL padding as real
    # columns. Scale valid gamma by sqrt(H/H_padded), the exact epsilon=0
    # correction; at GPT-OSS residual magnitudes the epsilon term is below
    # BF16 resolution. Padded gamma columns are zero and never reach QKV.
    gamma_sharded_host = torch.zeros_like(gamma_host)
    gamma_sharded_host.reshape(-1)[: config.hidden_size] = math.sqrt(config.hidden_size / physical_hidden)
    gamma_sharded = ttnn.from_torch(
        gamma_sharded_host,
        device=target_mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(target_mesh, target_mesh.shape, dims=(None, -2)),
    )
    qkv_host = (
        torch.randn(
            (1, 1, physical_hidden, local_qkv_width * logical_tp),
            generator=generator,
            dtype=torch.float32,
        )
        * 0.02
    ).to(torch.bfloat16)
    if physical_hidden != config.hidden_size:
        qkv_host[..., config.hidden_size :, :] = 0
    qkv_weight = ttnn.from_torch(
        qkv_host,
        device=target_mesh,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(target_mesh, target_mesh.shape, dims=(None, -1)),
    )
    qkv_current = ttnn.from_torch(
        qkv_host[..., : config.hidden_size, :].contiguous(),
        device=target_mesh,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(target_mesh, target_mesh.shape, dims=(None, -1)),
    )
    ccl = CCLManager(
        target_mesh,
        num_links=get_default_num_links(target_mesh),
        topology=ttnn.Topology.Ring,
    )
    rs_intermediate, rs_penultimate = ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(
        partials,
        dim=3,
        topology=ttnn.Topology.Ring,
        cluster_axis=1,
    )
    rs_output_shape = list(partials.shape)
    rs_output_shape[3] //= logical_tp
    rs_persistent_output = ttnn.from_torch(
        torch.zeros(rs_output_shape),
        device=target_mesh,
        dtype=collective_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(target_mesh),
    )
    ag_persistent_output = ttnn.from_torch(
        torch.zeros(tuple(int(dimension) for dimension in partials.shape)),
        device=target_mesh,
        dtype=collective_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(target_mesh),
    )
    rs_persistent_buffers = [rs_intermediate, rs_persistent_output]
    if rs_penultimate is not None:
        rs_persistent_buffers.append(rs_penultimate)
    qkv_tiles = local_qkv_width // ttnn.TILE_SIZE
    qkv_per_core_n = qkv_tiles // 8
    qkv_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 6),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=2 if qkv_per_core_n % 2 == 0 else 1,
        per_core_M=1,
        per_core_N=qkv_per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )
    qkv_compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def current_replicated_chain():
        # Match the canonical attention decode tail exactly: its O projection
        # produces the physical TP4 width, then it slices to the logical hidden
        # width and normalizes the sliced buffer through DRAM before all-reduce.
        sliced = ttnn.slice(
            partials,
            starts=[0, 0, 0, 0],
            ends=[
                partials.shape[0],
                partials.shape[1],
                partials.shape[2],
                config.hidden_size,
            ],
            steps=[1, 1, 1, 1],
        )
        sliced = ttnn.to_memory_config(sliced, ttnn.DRAM_MEMORY_CONFIG)
        sliced = ttnn.to_memory_config(sliced, ttnn.L1_MEMORY_CONFIG)
        reduced = ttnn.all_reduce(
            sliced,
            cluster_axis=1,
            num_links=ccl.num_links,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        normalized = ttnn.rms_norm(reduced, epsilon=config.rms_norm_eps, weight=gamma_current)
        projected = ttnn.linear(
            normalized,
            qkv_current,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        reduced.deallocate(True)
        normalized.deallocate(True)
        return projected

    def padded_replicated_chain():
        reduced = ttnn.all_reduce(
            partials,
            cluster_axis=1,
            num_links=ccl.num_links,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        sliced = ttnn.slice(
            reduced,
            starts=[0, 0, 0, 0],
            ends=[
                reduced.shape[0],
                reduced.shape[1],
                reduced.shape[2],
                config.hidden_size,
            ],
            steps=[1, 1, 1, 1],
        )
        normalized = ttnn.rms_norm(sliced, epsilon=config.rms_norm_eps, weight=gamma_current)
        projected = ttnn.linear(
            normalized,
            qkv_current,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        reduced.deallocate(True)
        sliced.deallocate(True)
        normalized.deallocate(True)
        return projected

    def async_replicated_chain(*, persistent=False):
        scattered = ttnn.experimental.reduce_scatter_minimal_async(
            partials,
            persistent_output_buffers=rs_persistent_buffers if persistent else None,
            dim=3,
            multi_device_global_semaphore=ccl.get_rs_ping_pong_semaphore(),
            barrier_semaphore=ccl.get_barrier_semaphore(),
            num_links=ccl.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
        )
        gathered = ttnn.experimental.all_gather_async(
            scattered,
            persistent_output_buffer=ag_persistent_output if persistent else None,
            dim=3,
            multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
            barrier_semaphore=ccl.get_barrier_semaphore(),
            num_links=ccl.num_links,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        sliced = ttnn.slice(
            gathered,
            starts=[0, 0, 0, 0],
            ends=[
                gathered.shape[0],
                gathered.shape[1],
                gathered.shape[2],
                config.hidden_size,
            ],
            steps=[1, 1, 1, 1],
        )
        normalized = ttnn.rms_norm(sliced, epsilon=config.rms_norm_eps, weight=gamma_current)
        projected = ttnn.linear(
            normalized,
            qkv_current,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if not persistent:
            scattered.deallocate(True)
            gathered.deallocate(True)
        sliced.deallocate(True)
        normalized.deallocate(True)
        return projected

    def sharded_chain(*, persistent=False):
        reduced = ttnn.experimental.reduce_scatter_minimal_async(
            partials,
            persistent_output_buffers=rs_persistent_buffers if persistent else None,
            dim=3,
            multi_device_global_semaphore=ccl.get_rs_ping_pong_semaphore(),
            barrier_semaphore=ccl.get_barrier_semaphore(),
            num_links=ccl.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
        )
        stats = ttnn.rms_norm_pre_all_gather(reduced, dtype=ttnn.bfloat16)
        gathered_stats = ttnn.all_gather(
            stats,
            dim=3,
            cluster_axis=1,
            num_links=ccl.num_links,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        normalized = ttnn.rms_norm_post_all_gather(
            reduced,
            gathered_stats,
            epsilon=config.rms_norm_eps,
            weight=gamma_sharded,
        )
        gathered, projected = ttnn.experimental.all_gather_matmul_async(
            normalized,
            qkv_weight,
            persistent_output_buffer=ag_persistent_output if persistent else None,
            dim=3,
            multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
            all_gather_core_grid_offset=(0, 6),
            barrier_semaphore=ccl.get_barrier_semaphore(),
            num_links=ccl.num_links,
            topology=ttnn.Topology.Ring,
            memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
            memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
            program_config=qkv_program_config,
            compute_kernel_config=qkv_compute_config,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        if not persistent:
            reduced.deallocate(True)
        stats.deallocate(True)
        gathered_stats.deallocate(True)
        normalized.deallocate(True)
        if not persistent:
            gathered.deallocate(True)
        return projected

    def expert_current_chain():
        return ttnn.all_reduce(
            expert_partials,
            cluster_axis=1,
            num_links=ccl.num_links,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def expert_runtime_padded_chain():
        padded = ttnn.pad(
            expert_partials,
            [(0, 0), (0, 0), (0, 0), (0, physical_hidden - config.hidden_size)],
            value=0.0,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        reduced = ttnn.all_reduce(
            padded,
            cluster_axis=1,
            num_links=ccl.num_links,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        padded.deallocate(True)
        logical = ttnn.slice(
            reduced,
            starts=[0, 0, 0, 0],
            ends=[
                reduced.shape[0],
                reduced.shape[1],
                reduced.shape[2],
                config.hidden_size,
            ],
            steps=[1, 1, 1, 1],
        )
        reduced.deallocate(True)
        return logical

    current_output = current_replicated_chain()
    padded_replicated_output = padded_replicated_chain()
    async_replicated_output = async_replicated_chain()
    async_replicated_persistent_output = async_replicated_chain(persistent=True)
    sharded_output = sharded_chain()
    sharded_persistent_output = sharded_chain(persistent=True)
    expert_current_output = expert_current_chain()
    expert_padded_output = expert_runtime_padded_chain()
    ttnn.synchronize_device(target_mesh)
    padded_pcc_details = []
    sharded_pcc_details = []
    async_pcc_details = []
    async_persistent_pcc_details = []
    sharded_persistent_pcc_details = []
    for rank, (
        current_rank,
        padded_rank,
        async_rank,
        async_persistent_rank,
        sharded_rank,
        sharded_persistent_rank,
    ) in enumerate(
        zip(
            ttnn.get_device_tensors(current_output),
            ttnn.get_device_tensors(padded_replicated_output),
            ttnn.get_device_tensors(async_replicated_output),
            ttnn.get_device_tensors(async_replicated_persistent_output),
            ttnn.get_device_tensors(sharded_output),
            ttnn.get_device_tensors(sharded_persistent_output),
        )
    ):
        current_host = ttnn.to_torch(current_rank)
        padded_pcc_details.append(
            _assert_pcc(
                ttnn.to_torch(padded_rank),
                current_host,
                0.99,
                f"TP{logical_tp} padded-replicated rank {rank}",
            )
        )
        sharded_pcc_details.append(
            _assert_pcc(
                ttnn.to_torch(sharded_rank),
                current_host,
                0.99,
                f"TP{logical_tp} residual-sharded fused-QKV rank {rank}",
            )
        )
        async_pcc_details.append(
            _assert_pcc(
                ttnn.to_torch(async_rank),
                ttnn.to_torch(padded_rank),
                0.999,
                f"TP{logical_tp} explicit-async replicated rank {rank}",
            )
        )
        async_persistent_pcc_details.append(
            _assert_pcc(
                ttnn.to_torch(async_persistent_rank),
                ttnn.to_torch(padded_rank),
                0.999,
                f"TP{logical_tp} persistent explicit-async replicated rank {rank}",
            )
        )
        sharded_persistent_pcc_details.append(
            _assert_pcc(
                ttnn.to_torch(sharded_persistent_rank),
                current_host,
                0.99,
                f"TP{logical_tp} persistent residual-sharded fused-QKV rank {rank}",
            )
        )
    current_output.deallocate(True)
    padded_replicated_output.deallocate(True)
    async_replicated_output.deallocate(True)
    async_replicated_persistent_output.deallocate(True)
    sharded_output.deallocate(True)
    sharded_persistent_output.deallocate(True)
    for rank, (current_rank, padded_rank) in enumerate(
        zip(
            ttnn.get_device_tensors(expert_current_output),
            ttnn.get_device_tensors(expert_padded_output),
        )
    ):
        _assert_pcc(
            ttnn.to_torch(padded_rank),
            ttnn.to_torch(current_rank),
            0.999,
            f"TP{logical_tp} runtime-padded expert collective rank {rank}",
        )
    expert_current_output.deallocate(True)
    expert_padded_output.deallocate(True)

    repeats = int(os.environ.get("GPT_OSS_120B_MULTICHIP_TOPOLOGY_REPEATS", "20"))
    current_started = time.perf_counter()
    for _ in range(repeats):
        output = current_replicated_chain()
        output.deallocate(True)
    ttnn.synchronize_device(target_mesh)
    current_ms = (time.perf_counter() - current_started) * 1000 / repeats

    padded_replicated_started = time.perf_counter()
    for _ in range(repeats):
        output = padded_replicated_chain()
        output.deallocate(True)
    ttnn.synchronize_device(target_mesh)
    padded_replicated_ms = (time.perf_counter() - padded_replicated_started) * 1000 / repeats

    async_replicated_started = time.perf_counter()
    for _ in range(repeats):
        output = async_replicated_chain()
        output.deallocate(True)
    ttnn.synchronize_device(target_mesh)
    async_replicated_ms = (time.perf_counter() - async_replicated_started) * 1000 / repeats

    async_persistent_started = time.perf_counter()
    for _ in range(repeats):
        output = async_replicated_chain(persistent=True)
        output.deallocate(True)
    ttnn.synchronize_device(target_mesh)
    async_persistent_ms = (time.perf_counter() - async_persistent_started) * 1000 / repeats

    sharded_started = time.perf_counter()
    for _ in range(repeats):
        output = sharded_chain()
        output.deallocate(True)
    ttnn.synchronize_device(target_mesh)
    sharded_ms = (time.perf_counter() - sharded_started) * 1000 / repeats

    sharded_persistent_started = time.perf_counter()
    for _ in range(repeats):
        output = sharded_chain(persistent=True)
        output.deallocate(True)
    ttnn.synchronize_device(target_mesh)
    sharded_persistent_ms = (time.perf_counter() - sharded_persistent_started) * 1000 / repeats

    expert_current_started = time.perf_counter()
    for _ in range(repeats):
        output = expert_current_chain()
        output.deallocate(True)
    ttnn.synchronize_device(target_mesh)
    expert_current_ms = (time.perf_counter() - expert_current_started) * 1000 / repeats

    expert_padded_started = time.perf_counter()
    for _ in range(repeats):
        output = expert_runtime_padded_chain()
        output.deallocate(True)
    ttnn.synchronize_device(target_mesh)
    expert_padded_ms = (time.perf_counter() - expert_padded_started) * 1000 / repeats
    print(
        "MULTICHIP_TOPOLOGY_PROBE "
        f"tp={logical_tp} dtype={collective_dtype} physical_hidden={physical_hidden} "
        f"local_hidden={plan.padded_local_hidden} "
        f"local_qkv={local_qkv_width} repeats={repeats} current_replicated_ms={current_ms:.9f} "
        f"padded_replicated_ms={padded_replicated_ms:.9f} sharded_fused_qkv_ms={sharded_ms:.9f} "
        f"async_replicated_ms={async_replicated_ms:.9f} async_replicated_persistent_ms={async_persistent_ms:.9f} "
        f"sharded_fused_qkv_persistent_ms={sharded_persistent_ms:.9f} "
        f"padded_ratio_vs_current={padded_replicated_ms / current_ms:.9f} "
        f"sharded_ratio_vs_current={sharded_ms / current_ms:.9f} "
        f"sharded_persistent_ratio_vs_current={sharded_persistent_ms / current_ms:.9f} "
        f"async_ratio_vs_selected={async_replicated_ms / padded_replicated_ms:.9f} "
        f"async_persistent_ratio_vs_selected={async_persistent_ms / padded_replicated_ms:.9f} "
        f"padded_pcc={' | '.join(map(str, padded_pcc_details))} "
        f"sharded_pcc={' | '.join(map(str, sharded_pcc_details))} "
        f"async_pcc={' | '.join(map(str, async_pcc_details))} "
        f"async_persistent_pcc={' | '.join(map(str, async_persistent_pcc_details))} "
        f"sharded_persistent_pcc={' | '.join(map(str, sharded_persistent_pcc_details))}"
    )
    print(
        "MULTICHIP_EXPERT_COLLECTIVE_PROBE "
        f"tp={logical_tp} logical_hidden={config.hidden_size} physical_hidden={physical_hidden} repeats={repeats} "
        f"current_ms={expert_current_ms:.9f} runtime_padded_ms={expert_padded_ms:.9f} "
        f"padded_ratio_vs_current={expert_padded_ms / expert_current_ms:.9f}"
    )


@pytest.mark.skipif(
    not RUN_FUSED_OUTPUT_PROBE,
    reason="set GPT_OSS_120B_MULTICHIP_FUSED_OUTPUT_PROBE=1 for the output-projection CCL probe",
)
@pytest.mark.timeout(900)
@pytest.mark.parametrize("logical_tp", [4], ids=["tp4"])
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (1, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "require_exact_physical_num_devices": True,
                TRACE_MODEL_KEY_PARAM: "gpt-oss-120b",
            },
            id="p150x4",
        )
    ],
    indirect=True,
)
def test_fused_output_projection_reduce_scatter_probe(mesh_device, device_params, logical_tp, reset_seeds):
    """Compare the exact TP4 decode O-projection family through replicated output.

    The selected chain is BF16 local matmul -> BFP8 physical-hidden all-reduce.
    The candidate is BFP8 fused matmul+reduce-scatter -> all-gather.  The fused
    family starts from the decoder's native physical width. Candidate retries
    may select a larger internal multiple, but both chains own any padding and
    slice back to the 2880 public residual width. TP2 is intentionally excluded:
    both adapted-3072 and native-2880 variants hung and required reset, so its
    preserved evidence is the captured triage/provenance log rather than a
    routinely runnable hardware test.
    """
    del device_params, reset_seeds
    config = _config()
    target_mesh = (
        mesh_device
        if logical_tp == 4
        else mesh_device.create_submesh(ttnn.MeshShape(1, logical_tp), offset=ttnn.MeshCoordinate(0, 0))
    )
    local_attention_width = config.num_attention_heads * config.head_dim // logical_tp
    physical_hidden = tensor_plan((1, logical_tp), config).padded_hidden_size
    batch = ttnn.TILE_SIZE
    generator = torch.Generator().manual_seed(919_004)
    input_host = (
        torch.randn(
            (1, 1, batch, local_attention_width * logical_tp),
            generator=generator,
            dtype=torch.float32,
        )
        * 0.02
    ).to(torch.bfloat16)
    weight_host = (
        torch.randn(
            (1, 1, local_attention_width * logical_tp, physical_hidden),
            generator=generator,
            dtype=torch.float32,
        )
        * 0.02
    ).to(torch.bfloat16)
    input_tensor = ttnn.from_torch(
        input_host,
        device=target_mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(target_mesh, dim=3),
    )
    weight = ttnn.from_torch(
        weight_host,
        device=target_mesh,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(target_mesh, dim=2),
    )
    ccl = CCLManager(
        target_mesh,
        num_links=get_default_num_links(target_mesh),
        topology=ttnn.Topology.Ring,
    )
    mesh_config = MeshConfig(
        target_mesh.shape,
        decode=ModeConfig(tp=logical_tp, ep=1, sp=1),
        prefill=ModeConfig(tp=logical_tp, ep=1, sp=1),
    )
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 6),
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=math.ceil((physical_hidden // ttnn.TILE_SIZE) / 8),
        out_block_w=max(1, math.ceil((physical_hidden // ttnn.TILE_SIZE) / 8) // 2),
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        target_mesh.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    persistent_intermediate = ttnn.from_torch(
        torch.zeros((1, 1, batch, physical_hidden)),
        device=target_mesh,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(target_mesh),
    )
    persistent_output = ttnn.from_torch(
        torch.zeros((1, 1, batch, physical_hidden // logical_tp)),
        device=target_mesh,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(target_mesh),
    )

    def selected_chain():
        partial = ttnn.linear(
            input_tensor,
            weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=compute_config,
        )
        partial_bfp8 = ttnn.typecast(partial, ttnn.bfloat8_b)
        partial.deallocate(True)
        return _allreduce_physical_hidden(
            partial_bfp8,
            hidden_size=config.hidden_size,
            padded_hidden_size=physical_hidden,
            mesh_config=mesh_config,
            ccl_manager=ccl,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def fused_chain():
        fused_input = ttnn.typecast(input_tensor, ttnn.bfloat8_b)
        matmul_output, scattered = ttnn.experimental.matmul_reduce_scatter_async(
            fused_input,
            weight,
            persistent_intermediate_buffer=persistent_intermediate,
            persistent_output_buffer=persistent_output,
            dim=3,
            multi_device_global_semaphore=ccl.get_rs_ping_pong_semaphore(),
            reduce_scatter_core_grid_offset=(0, 6),
            barrier_semaphore=ccl.get_barrier_semaphore(),
            num_links=ccl.num_links,
            memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            subdevice_id=None,
            memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
            program_config=program_config,
            compute_kernel_config=compute_config,
        )
        gathered = ttnn.experimental.all_gather_async(
            scattered,
            dim=3,
            multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
            barrier_semaphore=ccl.get_barrier_semaphore(),
            num_links=ccl.num_links,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        output = ttnn.slice(
            gathered,
            starts=[0, 0, 0, 0],
            ends=[1, 1, batch, config.hidden_size],
            steps=[1, 1, 1, 1],
        )
        fused_input.deallocate(True)
        matmul_output.deallocate(True)
        gathered.deallocate(True)
        return output

    selected = selected_chain()
    fused = fused_chain()
    ttnn.synchronize_device(target_mesh)
    pcc_details = []
    for rank, (selected_rank, fused_rank) in enumerate(
        zip(ttnn.get_device_tensors(selected), ttnn.get_device_tensors(fused))
    ):
        pcc_details.append(
            _assert_pcc(
                ttnn.to_torch(fused_rank),
                ttnn.to_torch(selected_rank),
                0.95,
                f"TP{logical_tp} fused matmul-reduce-scatter output rank {rank}",
            )
        )
    selected.deallocate(True)
    fused.deallocate(True)
    repeats = int(os.environ.get("GPT_OSS_120B_MULTICHIP_TOPOLOGY_REPEATS", "20"))

    def measure(chain):
        started = time.perf_counter()
        for _ in range(repeats):
            output = chain()
            output.deallocate(True)
        ttnn.synchronize_device(target_mesh)
        return (time.perf_counter() - started) * 1000 / repeats

    selected_ms = measure(selected_chain)
    fused_ms = measure(fused_chain)
    print(
        "MULTICHIP_FUSED_OUTPUT_PROBE "
        f"tp={logical_tp} logical_hidden={config.hidden_size} internal_hidden={physical_hidden} repeats={repeats} "
        f"selected_matmul_allreduce_ms={selected_ms:.9f} fused_mmrs_allgather_ms={fused_ms:.9f} "
        f"fused_ratio_vs_selected={fused_ms / selected_ms:.9f} pcc={' | '.join(map(str, pcc_details))}"
    )


@pytest.mark.skipif(
    not RUN_ACCEPTANCE or not REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_MULTICHIP_ACCEPTANCE=1 and GPT_OSS_120B_SNAPSHOT for the real-weight gate",
)
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (1, 1),
            {
                "fabric_config": None,
                "require_exact_physical_num_devices": False,
                TRACE_MODEL_KEY_PARAM: "gpt-oss-120b",
            },
            id="p150",
        )
    ],
    indirect=True,
)
def test_write_real_weight_optimized_baseline_artifact(mesh_device, device_params, layer_idx, reset_seeds):
    """Write a standalone P150 optimized-baseline artifact for a later process."""
    del device_params, reset_seeds
    snapshot = Path(REAL_WEIGHT_SNAPSHOT)
    assert snapshot.name == _FULL_LOCAL_CHECKPOINT_REVISION
    config = _config()
    state_dict = load_real_layer_state_dict(snapshot, layer_idx)
    baseline = _constructor(
        state_dict,
        config,
        layer_idx,
        mesh_device,
        _cache_root(layer_idx, "multichip_acceptance_tp1"),
    )
    assert baseline.is_single_chip_baseline
    assert isinstance(baseline.backend, OptimizedDecoder)
    assert baseline.tensor_plan == tensor_plan((1, 1), config)
    expected_cache_shape = (
        config.max_position_embeddings // accepted.PAGE_SIZE,
        config.num_key_value_heads,
        accepted.PAGE_SIZE,
        config.head_dim,
    )
    assert all(tuple(cache.shape) == expected_cache_shape for cache in baseline.kv_cache)
    assert all(cache.dtype == DEFAULT_MULTICHIP_POLICY.kv_cache_dtype for cache in baseline.kv_cache)

    # Public length is deliberately non-aligned.  Decode 127 followed by two
    # refreshed trace replays at 128 and 129 crosses both a page boundary and
    # the sliding-attention window boundary.
    sequence_length = 127
    generator = torch.Generator().manual_seed(470_000 + layer_idx)
    prefill_hidden_host = (torch.randn((1, 1, sequence_length, config.hidden_size), generator=generator) * 0.02).to(
        torch.bfloat16
    )
    decode_hidden_host = (torch.randn((1, 1, 1, config.hidden_size), generator=generator) * 0.02).to(torch.bfloat16)
    decode_position_host = torch.tensor([sequence_length], dtype=torch.int32)
    trace_refresh_hidden_hosts = [
        (torch.randn((1, 1, 1, config.hidden_size), generator=generator) * 0.02).to(torch.bfloat16)
        for _ in range(TRACE_REFRESH_STEPS)
    ]
    trace_refresh_position_hosts = [
        torch.tensor([sequence_length + step], dtype=torch.int32) for step in range(1, TRACE_REFRESH_STEPS + 1)
    ]
    page_table_host = _host_page_table(config, seed=601 + layer_idx)

    prefill_hidden, prefill_rope, page_table = _prefill_inputs(
        config, mesh_device, prefill_hidden_host, page_table_host
    )
    warm_prefill = baseline.prefill_forward(
        prefill_hidden,
        position_embeddings=prefill_rope,
        page_table=page_table,
    )
    ttnn.synchronize_device(mesh_device)
    warm_prefill.deallocate(True)
    ttnn.ReadDeviceProfiler(mesh_device)
    signpost("PERF_PREFILL")
    started = time.perf_counter()
    prefill_output = baseline.prefill_forward(
        prefill_hidden,
        position_embeddings=prefill_rope,
        page_table=page_table,
    )
    ttnn.synchronize_device(mesh_device)
    prefill_wall_ms = (time.perf_counter() - started) * 1000
    signpost("PERF_PREFILL_END")
    prefill_output_host = _assert_replicated(prefill_output, (1, 1, sequence_length, config.hidden_size))[
        0, 0, :sequence_length
    ]

    decode_hidden, decode_rope, current_position = _decode_inputs(
        config, mesh_device, decode_hidden_host, decode_position_host
    )
    trace_id, decode_output = _capture_decode(
        baseline,
        mesh_device,
        decode_hidden,
        decode_rope,
        current_position,
        page_table,
    )
    decode_output_host = _assert_replicated(decode_output, (1, 1, 1, config.hidden_size))[0, 0, :1]
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    assert torch.equal(
        _assert_replicated(decode_output, (1, 1, 1, config.hidden_size))[0, 0, :1],
        decode_output_host,
    )
    repeats, latency_samples_ms, latency_median_ms = _warmed_trace_latency_samples(
        mesh_device,
        trace_id,
        signposted=True,
    )

    trace_refresh_output_hosts = []
    previous_output_host = decode_output_host
    for refresh_hidden_host, refresh_position_host in zip(
        trace_refresh_hidden_hosts,
        trace_refresh_position_hosts,
    ):
        _refresh_decode_trace_inputs(
            config,
            mesh_device,
            hidden=decode_hidden,
            rope=decode_rope,
            current_position=current_position,
            page_table=page_table,
            hidden_host=refresh_hidden_host,
            position_host=refresh_position_host,
            page_table_host=page_table_host,
        )
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        refresh_output_host = _assert_replicated(decode_output, (1, 1, 1, config.hidden_size))[0, 0, :1]
        assert not torch.equal(refresh_output_host, previous_output_host)
        trace_refresh_output_hosts.append(refresh_output_host)
        previous_output_host = refresh_output_host

    # Cache inspection allocates slice outputs. Release the trace first so no
    # post-capture device allocation can overlap the trace's high-water mark.
    ttnn.release_trace(mesh_device, trace_id)

    logical_blocks = sorted(
        {
            sequence_length // accepted.PAGE_SIZE,
            (sequence_length + TRACE_REFRESH_STEPS) // accepted.PAGE_SIZE,
        }
    )
    physical_blocks = [int(page_table_host[0, logical_block].item()) for logical_block in logical_blocks]
    cache_blocks = {
        cache_name: [_cache_block(cache, physical_block) for physical_block in physical_blocks]
        for cache_name, cache in zip(("K", "V"), baseline.kv_cache)
    }
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": ACCEPTANCE_RUN_ID,
        "producer_process_uuid": PROCESS_UUID,
        "producer_pid": os.getpid(),
        "created_unix_ns": time.time_ns(),
        "checkpoint_revision": _FULL_LOCAL_CHECKPOINT_REVISION,
        "layer_idx": layer_idx,
        "layer_type": config.layer_types[layer_idx],
        "hidden_size": config.hidden_size,
        "max_context_length": config.max_position_embeddings,
        "page_size": accepted.PAGE_SIZE,
        "sequence_length": sequence_length,
        "prefill_hidden": prefill_hidden_host,
        "decode_hidden": decode_hidden_host,
        "decode_position": decode_position_host,
        "trace_refresh_hidden": trace_refresh_hidden_hosts,
        "trace_refresh_position": trace_refresh_position_hosts,
        "page_table": page_table_host,
        "prefill_output": prefill_output_host,
        "decode_output": decode_output_host,
        "trace_refresh_output": trace_refresh_output_hosts,
        "logical_blocks": logical_blocks,
        "physical_blocks": physical_blocks,
        "cache_blocks": cache_blocks,
        "prefill_wall_ms": prefill_wall_ms,
        "trace_repeats": repeats,
        "trace_latency_samples_ms": latency_samples_ms,
        "trace_latency_median_ms": latency_median_ms,
    }
    artifact_path = _artifact_path(layer_idx)
    _save_artifact_atomic(artifact, artifact_path)
    print(
        f"P150_BASELINE layer={layer_idx} type={config.layer_types[layer_idx]} "
        f"prefill_wall_ms={prefill_wall_ms:.6f} trace_median_ms={latency_median_ms:.9f} "
        f"samples={latency_samples_ms} repeats={repeats} artifact={artifact_path}"
    )
    del baseline, state_dict
    gc.collect()


@pytest.mark.skipif(
    not RUN_ACCEPTANCE or not REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_MULTICHIP_ACCEPTANCE=1 and GPT_OSS_120B_SNAPSHOT for the real-weight gate",
)
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (1, 1),
            {
                "fabric_config": None,
                "require_exact_physical_num_devices": False,
                TRACE_MODEL_KEY_PARAM: "gpt-oss-120b",
            },
            id="p150",
        )
    ],
    indirect=True,
)
def test_write_real_weight_batch2_high_position_baseline_artifact(mesh_device, device_params, layer_idx, reset_seeds):
    """Write P150 batch-2 prefill and exact-limit traced-decode evidence."""
    del device_params, reset_seeds
    snapshot = Path(REAL_WEIGHT_SNAPSHOT)
    assert snapshot.name == _FULL_LOCAL_CHECKPOINT_REVISION
    config = _config()
    batch_size = 2
    state_dict = load_real_layer_state_dict(snapshot, layer_idx)
    baseline = _constructor(
        state_dict,
        config,
        layer_idx,
        mesh_device,
        _cache_root(layer_idx, "multichip_batch2_high_position_tp1"),
        max_batch_size=batch_size,
        optimized_policy=DEFAULT_OPTIMIZED_POLICY,
    )
    assert baseline.is_single_chip_baseline
    assert baseline.single_chip_policy is DEFAULT_OPTIMIZED_POLICY
    blocks_per_user = config.max_position_embeddings // accepted.PAGE_SIZE
    expected_cache_shape = (
        batch_size * blocks_per_user,
        config.num_key_value_heads,
        accepted.PAGE_SIZE,
        config.head_dim,
    )
    assert all(tuple(cache.shape) == expected_cache_shape for cache in baseline.kv_cache)

    position_sets, hidden_sets, page_tables, step_page_table_ids = _batch2_high_context_inputs(config, layer_idx)
    generator = torch.Generator().manual_seed(180_000 + layer_idx)
    prefill_sequence_length = 33
    prefill_hidden_host = (
        torch.randn(
            (1, batch_size, prefill_sequence_length, config.hidden_size),
            generator=generator,
        )
        * 0.02
    ).to(torch.bfloat16)
    prefill_hidden, prefill_rope, page_table = _prefill_inputs(
        config,
        mesh_device,
        prefill_hidden_host,
        page_tables[int(step_page_table_ids[0])],
    )
    prefill_output = baseline.prefill_forward(
        prefill_hidden,
        position_embeddings=prefill_rope,
        page_table=page_table,
        batch_size=batch_size,
    )
    ttnn.synchronize_device(mesh_device)
    prefill_output_host = _assert_replicated(
        prefill_output,
        (1, batch_size, prefill_sequence_length, config.hidden_size),
    )
    hidden, rope, current_position = _decode_inputs(config, mesh_device, hidden_sets[0], position_sets[0])
    trace_id, output = _capture_decode(
        baseline,
        mesh_device,
        hidden,
        rope,
        current_position,
        page_table,
    )
    output_hosts = [_assert_replicated(output, (1, 1, batch_size, config.hidden_size))[0, 0, :batch_size]]
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    assert torch.equal(
        _assert_replicated(output, (1, 1, batch_size, config.hidden_size))[0, 0, :batch_size],
        output_hosts[0],
    )
    for step, (hidden_host, positions_host) in enumerate(zip(hidden_sets[1:], position_sets[1:]), start=1):
        page_table_host = page_tables[int(step_page_table_ids[step])]
        _refresh_decode_trace_inputs(
            config,
            mesh_device,
            hidden=hidden,
            rope=rope,
            current_position=current_position,
            page_table=page_table,
            hidden_host=hidden_host,
            position_host=positions_host,
            page_table_host=page_table_host,
        )
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        next_host = _assert_replicated(output, (1, 1, batch_size, config.hidden_size))[0, 0, :batch_size]
        if step == 1:
            assert not torch.equal(next_host, output_hosts[-1])
        output_hosts.append(next_host)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    assert torch.equal(
        _assert_replicated(output, (1, 1, batch_size, config.hidden_size))[0, 0, :batch_size],
        output_hosts[-1],
    )
    ttnn.release_trace(mesh_device, trace_id)

    cache_entries = [
        {
            "phase": "prefill",
            "step": -1,
            "page_table_id": 0,
            "user": user,
            "position": 0,
            "logical_block": 0,
            "physical_block": int(page_tables[0, user, 0]),
            "block_offset": 0,
        }
        for user in range(batch_size)
    ]
    for step, positions_host in enumerate(position_sets):
        page_table_id = int(step_page_table_ids[step])
        for user, position in enumerate(positions_host.tolist()):
            logical_block = position // accepted.PAGE_SIZE
            physical_block = int(page_tables[page_table_id, user, logical_block].item())
            cache_entries.append(
                {
                    "phase": "decode",
                    "step": step,
                    "page_table_id": page_table_id,
                    "user": user,
                    "position": position,
                    "logical_block": logical_block,
                    "physical_block": physical_block,
                    "block_offset": position % accepted.PAGE_SIZE,
                }
            )
    assert len({entry["physical_block"] for entry in cache_entries}) == len(cache_entries)
    cache_blocks = {
        cache_name: [_cache_block(cache, entry["physical_block"]) for entry in cache_entries]
        for cache_name, cache in zip(("K", "V"), baseline.kv_cache)
    }
    artifact = {
        "schema_version": BATCH2_ARTIFACT_SCHEMA_VERSION,
        "run_id": ACCEPTANCE_RUN_ID,
        "producer_process_uuid": PROCESS_UUID,
        "checkpoint_revision": _FULL_LOCAL_CHECKPOINT_REVISION,
        "layer_idx": layer_idx,
        "layer_type": config.layer_types[layer_idx],
        "batch_size": batch_size,
        "configured_max_batch_size": baseline.max_batch_size,
        "optimized_policy": baseline.single_chip_policy.name,
        "max_context_length": config.max_position_embeddings,
        "page_size": accepted.PAGE_SIZE,
        "prefill_sequence_length": prefill_sequence_length,
        "prefill_hidden": prefill_hidden_host,
        "prefill_output": prefill_output_host,
        "position_sets": position_sets,
        "hidden_sets": hidden_sets,
        "page_tables": page_tables,
        "step_page_table_ids": step_page_table_ids,
        "output_sets": output_hosts,
        "cache_shape": expected_cache_shape,
        "cache_entries": cache_entries,
        "cache_blocks": cache_blocks,
        "initial_replay_deterministic": True,
        "changed_table_replay_deterministic": True,
    }
    artifact_path = _batch2_artifact_path(layer_idx)
    _save_artifact_atomic(artifact, artifact_path)
    print(
        f"P150_BATCH2_HIGH_POSITION layer={layer_idx} type={config.layer_types[layer_idx]} "
        f"prefill_sequence={prefill_sequence_length} positions={position_sets.tolist()} "
        f"step_page_table_ids={step_page_table_ids.tolist()} artifact={artifact_path}"
    )
    del baseline, state_dict
    gc.collect()


@pytest.mark.skipif(
    not RUN_ACCEPTANCE or not REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_MULTICHIP_ACCEPTANCE=1 and GPT_OSS_120B_SNAPSHOT for the real-weight gate",
)
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@pytest.mark.parametrize("logical_tp", [2, 4], ids=["tp2", "tp4"])
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (1, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "require_exact_physical_num_devices": True,
                TRACE_MODEL_KEY_PARAM: "gpt-oss-120b",
            },
            id="p150x4-parent",
        )
    ],
    indirect=True,
)
def test_real_weight_multichip_against_baseline_artifact(
    mesh_device, device_params, logical_tp, layer_idx, reset_seeds
):
    """Validate TP2/TP4 from a full parent against a separate P150 process."""
    del device_params, reset_seeds
    artifact_path = _artifact_path(layer_idx)
    assert artifact_path.is_file(), (
        f"missing {artifact_path}; run test_write_real_weight_optimized_baseline_artifact "
        "as a separate safe-pytest process first"
    )
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    assert artifact["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert artifact["run_id"] == ACCEPTANCE_RUN_ID
    assert (
        artifact["producer_process_uuid"] != PROCESS_UUID
    ), "baseline and multichip target must run in separate processes; use two serialized safe-pytest commands"
    assert artifact["checkpoint_revision"] == _FULL_LOCAL_CHECKPOINT_REVISION
    assert artifact["layer_idx"] == layer_idx

    snapshot = Path(REAL_WEIGHT_SNAPSHOT)
    assert snapshot.name == _FULL_LOCAL_CHECKPOINT_REVISION
    config = _config()
    assert artifact["layer_type"] == config.layer_types[layer_idx]
    assert artifact["hidden_size"] == config.hidden_size
    assert artifact["max_context_length"] == config.max_position_embeddings
    assert artifact["page_size"] == accepted.PAGE_SIZE

    if logical_tp == 4:
        target_mesh = mesh_device
    else:
        target_mesh = mesh_device.create_submesh(ttnn.MeshShape(1, logical_tp), offset=ttnn.MeshCoordinate(0, 0))
    assert tuple(target_mesh.shape) == (1, logical_tp)
    state_dict = load_real_layer_state_dict(snapshot, layer_idx)
    multichip = _constructor(
        state_dict,
        config,
        layer_idx,
        target_mesh,
        _cache_root(layer_idx, f"multichip_acceptance_tp{logical_tp}"),
    )
    assert not multichip.is_single_chip_baseline
    assert multichip.mlp.decode_uses_gate_selected_sparse_experts
    assert not multichip.mlp.use_throughput_experts
    expected_plan = tensor_plan((1, logical_tp), config)
    assert multichip.tensor_plan == expected_plan
    expected_gate_up_shape = (
        1,
        config.num_local_experts,
        config.hidden_size,
        2 * multichip.mlp.padded_local_intermediate_size,
    )
    expected_down_shape = (
        1,
        config.num_local_experts,
        multichip.mlp.padded_local_intermediate_size,
        config.hidden_size,
    )
    if multichip.policy.decode_separate_gate_up:
        expected_separate_shape = (
            1,
            config.num_local_experts,
            config.hidden_size,
            multichip.mlp.padded_local_intermediate_size,
        )
        assert multichip.mlp.indexed_gate_up is None
        assert all(
            tuple(shard.shape) == expected_separate_shape
            for weight in (multichip.mlp.indexed_gate, multichip.mlp.indexed_up)
            for shard in ttnn.get_device_tensors(weight)
        )
    else:
        assert all(
            tuple(shard.shape) == expected_gate_up_shape
            for shard in ttnn.get_device_tensors(multichip.mlp.indexed_gate_up)
        )
    assert all(
        tuple(shard.shape) == expected_down_shape for shard in ttnn.get_device_tensors(multichip.mlp.indexed_down)
    )
    assert not hasattr(multichip.mlp.experts, "weights")
    expected_cache_shape = (
        config.max_position_embeddings // accepted.PAGE_SIZE,
        expected_plan.local_kv_heads,
        accepted.PAGE_SIZE,
        config.head_dim,
    )
    assert all(tuple(cache.shape) == expected_cache_shape for cache in multichip.kv_cache)
    assert all(cache.dtype == DEFAULT_MULTICHIP_POLICY.kv_cache_dtype for cache in multichip.kv_cache)

    sequence_length = artifact["sequence_length"]
    prefill_hidden, prefill_rope, page_table = _prefill_inputs(
        config,
        target_mesh,
        artifact["prefill_hidden"],
        artifact["page_table"],
    )
    warm_prefill = multichip.prefill_forward(
        prefill_hidden,
        position_embeddings=prefill_rope,
        page_table=page_table,
    )
    ttnn.synchronize_device(target_mesh)
    warm_prefill_host = _assert_replicated(warm_prefill, (1, 1, sequence_length, config.hidden_size))[
        0, 0, :sequence_length
    ]
    _assert_pcc(
        warm_prefill_host,
        artifact["prefill_output"],
        PREFILL_PCC_THRESHOLD,
        "warmed multichip prefill vs optimized baseline",
    )
    warm_prefill.deallocate(True)
    ttnn.ReadDeviceProfiler(target_mesh)

    signpost("PERF_PREFILL")
    started = time.perf_counter()
    prefill_output = multichip.prefill_forward(
        prefill_hidden,
        position_embeddings=prefill_rope,
        page_table=page_table,
    )
    ttnn.synchronize_device(target_mesh)
    prefill_wall_ms = (time.perf_counter() - started) * 1000
    signpost("PERF_PREFILL_END")
    prefill_output_host = _assert_replicated(prefill_output, (1, 1, sequence_length, config.hidden_size))[
        0, 0, :sequence_length
    ]
    prefill_detail = _assert_pcc(
        prefill_output_host,
        artifact["prefill_output"],
        PREFILL_PCC_THRESHOLD,
        "multichip prefill vs optimized baseline",
    )

    decode_hidden, decode_rope, current_position = _decode_inputs(
        config,
        target_mesh,
        artifact["decode_hidden"],
        artifact["decode_position"],
    )
    trace_id, decode_output = _capture_decode(
        multichip,
        target_mesh,
        decode_hidden,
        decode_rope,
        current_position,
        page_table,
    )
    decode_output_host = _assert_replicated(decode_output, (1, 1, 1, config.hidden_size))[0, 0, :1]
    decode_detail = _assert_pcc(
        decode_output_host,
        artifact["decode_output"],
        DECODE_PCC_THRESHOLD,
        "multichip traced decode vs optimized baseline",
    )
    ttnn.execute_trace(target_mesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(target_mesh)
    assert torch.equal(
        _assert_replicated(decode_output, (1, 1, 1, config.hidden_size))[0, 0, :1],
        decode_output_host,
    )
    ttnn.ReadDeviceProfiler(target_mesh)

    repeats, latency_samples_ms, latency_median_ms = _warmed_trace_latency_samples(
        target_mesh,
        trace_id,
        signposted=True,
    )

    trace_refresh_details = []
    previous_output_host = decode_output_host
    for refresh_hidden_host, refresh_position_host, expected_output_host in zip(
        artifact["trace_refresh_hidden"],
        artifact["trace_refresh_position"],
        artifact["trace_refresh_output"],
    ):
        _refresh_decode_trace_inputs(
            config,
            target_mesh,
            hidden=decode_hidden,
            rope=decode_rope,
            current_position=current_position,
            page_table=page_table,
            hidden_host=refresh_hidden_host,
            position_host=refresh_position_host,
            page_table_host=artifact["page_table"],
        )
        ttnn.execute_trace(target_mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(target_mesh)
        refresh_output_host = _assert_replicated(
            decode_output,
            (1, 1, 1, config.hidden_size),
        )[0, 0, :1]
        trace_refresh_details.append(
            _assert_pcc(
                refresh_output_host,
                expected_output_host,
                DECODE_PCC_THRESHOLD,
                "refreshed multichip trace vs optimized baseline",
            )
        )
        assert not torch.equal(refresh_output_host, previous_output_host)
        previous_output_host = refresh_output_host

    # Cache inspection below creates device-side slice outputs. The trace has
    # completed every replay and is no longer needed, so release it before any
    # such allocation. TT_METAL_TRACE_ALLOC_TRACKING=1 verifies this ordering.
    ttnn.release_trace(target_mesh, trace_id)

    for logical_block, physical_block in zip(artifact["logical_blocks"], artifact["physical_blocks"]):
        assert int(artifact["page_table"][0, logical_block].item()) == physical_block
    for cache_name, multichip_cache in zip(("K", "V"), multichip.kv_cache):
        for logical_block, physical_block, expected_cache_block in zip(
            artifact["logical_blocks"],
            artifact["physical_blocks"],
            artifact["cache_blocks"][cache_name],
        ):
            cache_detail = _assert_pcc(
                _reconstruct_tp_cache_block(multichip_cache, physical_block),
                expected_cache_block,
                CACHE_PCC_THRESHOLD,
                f"reconstructed local {cache_name} cache vs optimized baseline",
            )
            print(
                f"MULTICHIP_CACHE mesh=1x{logical_tp} layer={layer_idx} cache={cache_name} "
                f"logical_block={logical_block} physical_block={physical_block} {cache_detail}"
            )

    baseline_ms = artifact["trace_latency_median_ms"]
    speedup = baseline_ms / latency_median_ms
    efficiency = speedup / logical_tp
    print(
        f"MULTICHIP_PREFILL mesh=1x{logical_tp} layer={layer_idx} "
        f"type={config.layer_types[layer_idx]} sequence={sequence_length} "
        f"baseline_wall_ms={artifact['prefill_wall_ms']:.6f} multichip_wall_ms={prefill_wall_ms:.6f} "
        f"{prefill_detail}"
    )
    print(
        f"MULTICHIP_DECODE mesh=1x{logical_tp} layer={layer_idx} type={config.layer_types[layer_idx]} "
        f"baseline_ms={baseline_ms:.9f} multichip_ms={latency_median_ms:.9f} "
        f"speedup={speedup:.9f} efficiency={efficiency:.9f} repeats={repeats} "
        f"samples={latency_samples_ms} {decode_detail} refresh={trace_refresh_details}"
    )
    del multichip, state_dict
    gc.collect()


@pytest.mark.skipif(
    not RUN_ACCEPTANCE or not REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_MULTICHIP_ACCEPTANCE=1 and GPT_OSS_120B_SNAPSHOT for the real-weight gate",
)
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
@pytest.mark.parametrize("logical_tp", [2, 4], ids=["tp2", "tp4"])
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (1, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "require_exact_physical_num_devices": True,
                TRACE_MODEL_KEY_PARAM: "gpt-oss-120b",
            },
            id="p150x4-parent",
        )
    ],
    indirect=True,
)
def test_real_weight_multichip_batch2_high_position_against_baseline_artifact(
    mesh_device, device_params, logical_tp, layer_idx, reset_seeds
):
    """Validate batch-2 prefill, exact-limit decode, and page-table refresh."""
    del device_params, reset_seeds
    artifact_path = _batch2_artifact_path(layer_idx)
    assert artifact_path.is_file(), (
        f"missing {artifact_path}; run test_write_real_weight_batch2_high_position_baseline_artifact "
        "in a separate safe-pytest process first"
    )
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    assert artifact["schema_version"] == BATCH2_ARTIFACT_SCHEMA_VERSION
    assert artifact["run_id"] == ACCEPTANCE_RUN_ID
    assert artifact["producer_process_uuid"] != PROCESS_UUID
    assert artifact["checkpoint_revision"] == _FULL_LOCAL_CHECKPOINT_REVISION
    assert artifact["layer_idx"] == layer_idx
    batch_size = artifact["batch_size"]
    assert batch_size == 2

    snapshot = Path(REAL_WEIGHT_SNAPSHOT)
    assert snapshot.name == _FULL_LOCAL_CHECKPOINT_REVISION
    config = _config()
    assert artifact["layer_type"] == config.layer_types[layer_idx]
    assert artifact["max_context_length"] == config.max_position_embeddings
    assert artifact["page_size"] == accepted.PAGE_SIZE
    assert artifact["configured_max_batch_size"] == batch_size
    assert artifact["optimized_policy"] == DEFAULT_OPTIMIZED_POLICY.name
    assert artifact["initial_replay_deterministic"]
    assert artifact["changed_table_replay_deterministic"]
    if logical_tp == 4:
        target_mesh = mesh_device
    else:
        target_mesh = mesh_device.create_submesh(
            ttnn.MeshShape(1, logical_tp),
            offset=ttnn.MeshCoordinate(0, 0),
        )
    state_dict = load_real_layer_state_dict(snapshot, layer_idx)
    multichip = _constructor(
        state_dict,
        config,
        layer_idx,
        target_mesh,
        _cache_root(layer_idx, f"multichip_batch2_high_position_tp{logical_tp}"),
        max_batch_size=batch_size,
    )
    plan = tensor_plan((1, logical_tp), config)
    blocks_per_user = config.max_position_embeddings // accepted.PAGE_SIZE
    expected_cache_shape = (
        batch_size * blocks_per_user,
        plan.local_kv_heads,
        accepted.PAGE_SIZE,
        config.head_dim,
    )
    assert all(tuple(cache.shape) == expected_cache_shape for cache in multichip.kv_cache)
    assert tuple(artifact["cache_shape"]) == (
        batch_size * blocks_per_user,
        config.num_key_value_heads,
        accepted.PAGE_SIZE,
        config.head_dim,
    )
    assert multichip.mlp.decode_uses_gate_selected_sparse_experts

    prefill_hidden, prefill_rope, page_table = _prefill_inputs(
        config,
        target_mesh,
        artifact["prefill_hidden"],
        artifact["page_tables"][int(artifact["step_page_table_ids"][0])],
    )
    prefill_output = multichip.prefill_forward(
        prefill_hidden,
        position_embeddings=prefill_rope,
        page_table=page_table,
        batch_size=batch_size,
    )
    ttnn.synchronize_device(target_mesh)
    prefill_detail = _assert_pcc(
        _assert_replicated(
            prefill_output,
            (1, batch_size, artifact["prefill_sequence_length"], config.hidden_size),
        ),
        artifact["prefill_output"],
        PREFILL_PCC_THRESHOLD,
        f"TP{logical_tp} batch-two non-aligned prefill",
    )
    hidden, rope, current_position = _decode_inputs(
        config,
        target_mesh,
        artifact["hidden_sets"][0],
        artifact["position_sets"][0],
    )
    trace_id, output = _capture_decode(
        multichip,
        target_mesh,
        hidden,
        rope,
        current_position,
        page_table,
    )
    output_details = [
        _assert_pcc(
            _assert_replicated(output, (1, 1, batch_size, config.hidden_size))[0, 0, :batch_size],
            artifact["output_sets"][0],
            DECODE_PCC_THRESHOLD,
            f"TP{logical_tp} batch-two initial high-position decode",
        )
    ]
    initial_output_host = _assert_replicated(output, (1, 1, batch_size, config.hidden_size))[0, 0, :batch_size]
    ttnn.execute_trace(target_mesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(target_mesh)
    assert torch.equal(
        _assert_replicated(output, (1, 1, batch_size, config.hidden_size))[0, 0, :batch_size],
        initial_output_host,
    )
    for step, (hidden_host, positions_host, expected_output) in enumerate(
        zip(
            artifact["hidden_sets"][1:],
            artifact["position_sets"][1:],
            artifact["output_sets"][1:],
        ),
        start=1,
    ):
        page_table_host = artifact["page_tables"][int(artifact["step_page_table_ids"][step])]
        _refresh_decode_trace_inputs(
            config,
            target_mesh,
            hidden=hidden,
            rope=rope,
            current_position=current_position,
            page_table=page_table,
            hidden_host=hidden_host,
            position_host=positions_host,
            page_table_host=page_table_host,
        )
        ttnn.execute_trace(target_mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(target_mesh)
        output_details.append(
            _assert_pcc(
                _assert_replicated(output, (1, 1, batch_size, config.hidden_size))[0, 0, :batch_size],
                expected_output,
                DECODE_PCC_THRESHOLD,
                f"TP{logical_tp} batch-two refreshed high-position decode",
            )
        )
    changed_table_output_host = _assert_replicated(output, (1, 1, batch_size, config.hidden_size))[0, 0, :batch_size]
    ttnn.execute_trace(target_mesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(target_mesh)
    assert torch.equal(
        _assert_replicated(output, (1, 1, batch_size, config.hidden_size))[0, 0, :batch_size],
        changed_table_output_host,
    )
    ttnn.release_trace(target_mesh, trace_id)

    cache_details = []
    for cache_name, target_cache in zip(("K", "V"), multichip.kv_cache):
        for entry, expected_cache_block in zip(
            artifact["cache_entries"],
            artifact["cache_blocks"][cache_name],
        ):
            cache_details.append(
                _assert_pcc(
                    _reconstruct_tp_cache_block(target_cache, entry["physical_block"]),
                    expected_cache_block,
                    CACHE_PCC_THRESHOLD,
                    f"TP{logical_tp} batch-two {cache_name} user {entry['user']} position {entry['position']}",
                )
            )
    print(
        f"MULTICHIP_BATCH2_HIGH_POSITION mesh=1x{logical_tp} layer={layer_idx} "
        f"type={config.layer_types[layer_idx]} prefill={prefill_detail} "
        f"positions={artifact['position_sets'].tolist()} "
        f"step_page_table_ids={artifact['step_page_table_ids'].tolist()} "
        f"decode={output_details} cache_min={min(float(detail) for detail in cache_details):.12f}"
    )
    del multichip, state_dict
    gc.collect()
