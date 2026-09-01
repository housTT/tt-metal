# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in Tracy harnesses for the Qwen3.6 kernels targeted by issue #50475.

These tests load one real checkpoint layer, compile once, then place ``start`` / ``stop``
signposts around each measured production-path invocation.  They are skipped unless
``QWEN36_KERNEL_PROFILE`` selects ``gdn_prefill``, ``attention_prefill``, ``gdn_decode``,
``attention_decode``, ``mlp_decode``, or ``sampling_decode``.
"""

import os
import statistics
import time

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.modules.tt_ccl import get_tt_ccl
from models.common.sampling.generator import SamplingGenerator, SamplingParams, format_sampling_params
from models.demos.blackhole.qwen36.tests.test_factory import (
    load_attn_layer,
    load_gdn_layer,
    load_mlp_layer,
    model_path,
    parametrize_mesh_tp,
    replicate_to_device,
    shard_to_device,
)
from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_decode, rot_mats_prefill
from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention, load_attention_weights_tp
from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp
from models.demos.blackhole.qwen36.tt.mlp import Qwen36MLP
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs


def _selected(component):
    if os.environ.get("QWEN36_KERNEL_PROFILE") != component:
        pytest.skip(f"set QWEN36_KERNEL_PROFILE={component} to run this Tracy harness")


@torch.no_grad()
@parametrize_mesh_tp()
def test_gdn_prefill_profile(mesh_device, reset_seeds, ensure_gc):
    _selected("gdn_prefill")
    os.environ.setdefault("HF_MODEL", model_path())
    seq_len = int(os.environ.get("QWEN36_KERNEL_PROFILE_SEQ_LEN", "2048"))
    assert seq_len > 0 and seq_len % 128 == 0

    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=seq_len)
    layer_idx = next(i for i, kind in enumerate(args.attention_type_list) if kind == "linear_attention")
    weights = load_gdn_weights_tp(mesh_device, load_gdn_layer(args.CKPT_DIR, layer_idx), args)
    gdn = TPGatedDeltaNet(mesh_device, args, weights, get_tt_ccl(mesh_device))
    x = torch.randn(1, 1, seq_len, args.dim, dtype=torch.bfloat16)
    x_device = shard_to_device(mesh_device, x, dim=-1)
    fused_epilogue = os.environ.get("QWEN36_GDN_FUSED_EPILOGUE", "1") != "0"
    iterations = int(os.environ.get("QWEN36_KERNEL_PROFILE_ITERATIONS", "5"))
    assert iterations > 0
    gdn._gdn_fuse_out = fused_epilogue

    # Compile and populate the program cache outside the measured interval.
    out = gdn.forward_prefill(x_device, chunk_size=128, borrow_output=True)
    ttnn.synchronize_device(mesh_device)
    if gdn.rec_state is not None:
        ttnn.deallocate(gdn.rec_state)
        gdn.rec_state = None

    samples_ms = []
    for _ in range(iterations):
        signpost("start")
        begin = time.perf_counter()
        out = gdn.forward_prefill(x_device, chunk_size=128, borrow_output=True)
        ttnn.synchronize_device(mesh_device)
        samples_ms.append((time.perf_counter() - begin) * 1000.0)
        signpost("stop")
        if gdn.rec_state is not None:
            ttnn.deallocate(gdn.rec_state)
            gdn.rec_state = None
    logger.info(
        f"GDN_PREFILL_PROFILE_RESULT layer={layer_idx} seq_len={seq_len} fused_epilogue={fused_epilogue} "
        f"iterations={iterations} median_ms={statistics.median(samples_ms):.3f} "
        f"min_ms={min(samples_ms):.3f} max_ms={max(samples_ms):.3f}"
    )
    assert out.shape[-2] == seq_len


@torch.no_grad()
@parametrize_mesh_tp()
def test_attention_prefill_profile(mesh_device, reset_seeds, ensure_gc):
    _selected("attention_prefill")
    os.environ.setdefault("HF_MODEL", model_path())
    seq_len = int(os.environ.get("QWEN36_KERNEL_PROFILE_SEQ_LEN", "2048"))
    chunk_start = int(os.environ.get("QWEN36_KERNEL_PROFILE_CHUNK_START", "0"))
    block_size = 64
    assert seq_len > 0 and seq_len % block_size == 0
    assert chunk_start >= 0 and chunk_start % seq_len == 0
    max_seq_len = chunk_start + seq_len

    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=max_seq_len)
    layer_idx = next(i for i, kind in enumerate(args.attention_type_list) if kind == "full_attention")
    weights = load_attention_weights_tp(mesh_device, load_attn_layer(args.CKPT_DIR, layer_idx), args)
    attention = TPAttention(mesh_device, args, weights, get_tt_ccl(mesh_device))
    num_blocks = (max_seq_len + block_size - 1) // block_size
    k_cache_dtype = ttnn.bfloat8_b if attention._sdpa_k_bf8 else ttnn.bfloat16
    v_cache_dtype = ttnn.bfloat8_b if attention._sdpa_v_bf8 else ttnn.bfloat16

    def make_cache(cache_dtype):
        return ttnn.from_torch(
            torch.zeros(num_blocks, args.n_local_kv_heads, block_size, args.head_dim, dtype=torch.bfloat16),
            dtype=cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    attention.set_paged_kv_cache(make_cache(k_cache_dtype), make_cache(v_cache_dtype))
    full_page_table = ttnn.from_torch(
        torch.arange(num_blocks, dtype=torch.int32).reshape(1, -1),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
    )
    first_chunk_block = chunk_start // block_size
    chunk_blocks = seq_len // block_size
    chunk_page_table = ttnn.from_torch(
        torch.arange(first_chunk_block, first_chunk_block + chunk_blocks, dtype=torch.int32).reshape(1, -1),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
    )
    x = torch.randn(1, 1, seq_len, args.dim, dtype=torch.bfloat16)
    x_device = shard_to_device(mesh_device, x, dim=-1)
    cos, sin = rot_mats_prefill(mesh_device, args.rope_head_dim, seq_len, args.rope_theta)
    iterations = int(os.environ.get("QWEN36_KERNEL_PROFILE_ITERATIONS", "5"))
    flexible_offset = os.environ.get("QWEN36_KERNEL_PROFILE_FLEXIBLE_OFFSET", "0") == "1"
    assert iterations > 0
    chunk_start_tensor = (
        ttnn.from_torch(
            torch.tensor([chunk_start], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
        )
        if flexible_offset
        else None
    )

    def run_once():
        return attention.forward_prefill_paged(
            x_device,
            cos,
            sin,
            full_page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start,
            chunk_start_idx_tensor=chunk_start_tensor,
            borrow_output=True,
        )

    out = run_once()
    ttnn.synchronize_device(mesh_device)

    samples_ms = []
    for _ in range(iterations):
        signpost("start")
        begin = time.perf_counter()
        out = run_once()
        ttnn.synchronize_device(mesh_device)
        samples_ms.append((time.perf_counter() - begin) * 1000.0)
        signpost("stop")
    logger.info(
        f"ATTENTION_PREFILL_PROFILE_RESULT layer={layer_idx} seq_len={seq_len} "
        f"chunk_start={chunk_start} k_cache_dtype={k_cache_dtype} v_cache_dtype={v_cache_dtype} "
        f"flexible_offset={flexible_offset} "
        f"q_chunk={attention._sdpa_prefill_q_chunk or 'auto'} "
        f"k_chunk={attention._sdpa_prefill_k_chunk or 'auto'} "
        f"grid={attention._sdpa_prefill_grid or 'auto'} "
        f"iterations={iterations} median_ms={statistics.median(samples_ms):.3f} "
        f"min_ms={min(samples_ms):.3f} max_ms={max(samples_ms):.3f}"
    )
    assert out.shape[-2] == seq_len


@torch.no_grad()
@parametrize_mesh_tp()
def test_gdn_decode_profile(mesh_device, reset_seeds, ensure_gc):
    _selected("gdn_decode")
    os.environ.setdefault("HF_MODEL", model_path())
    batch = int(os.environ.get("QWEN36_KERNEL_PROFILE_BATCH", "32"))
    iterations = int(os.environ.get("QWEN36_KERNEL_PROFILE_ITERATIONS", "5"))
    assert batch in (1, 8, 32) and iterations > 0

    args = Qwen36ModelArgs(mesh_device, max_batch_size=batch, max_seq_len=256)
    layer_idx = next(i for i, kind in enumerate(args.attention_type_list) if kind == "linear_attention")
    weights = load_gdn_weights_tp(mesh_device, load_gdn_layer(args.CKPT_DIR, layer_idx), args)
    gdn = TPGatedDeltaNet(mesh_device, args, weights, get_tt_ccl(mesh_device))
    x = torch.randn(1, 1, batch, args.dim, dtype=torch.bfloat16)
    x_device = replicate_to_device(mesh_device, x)

    # Compile, allocate persistent state, and populate the program cache outside the measured interval.
    out = gdn.forward_decode(x_device)
    ttnn.synchronize_device(mesh_device)

    samples_ms = []
    for _ in range(iterations):
        signpost("start")
        begin = time.perf_counter()
        out = gdn.forward_decode(x_device)
        ttnn.synchronize_device(mesh_device)
        samples_ms.append((time.perf_counter() - begin) * 1000.0)
        signpost("stop")
    logger.info(
        f"GDN_DECODE_PROFILE_RESULT layer={layer_idx} batch={batch} iterations={iterations} "
        f"median_ms={statistics.median(samples_ms):.3f} min_ms={min(samples_ms):.3f} max_ms={max(samples_ms):.3f}"
    )
    assert out.shape[-2] == batch


@torch.no_grad()
@parametrize_mesh_tp()
def test_attention_decode_profile(mesh_device, reset_seeds, ensure_gc):
    _selected("attention_decode")
    os.environ.setdefault("HF_MODEL", model_path())
    batch = int(os.environ.get("QWEN36_KERNEL_PROFILE_BATCH", "1"))
    context = int(os.environ.get("QWEN36_KERNEL_PROFILE_CONTEXT", "131072"))
    iterations = int(os.environ.get("QWEN36_KERNEL_PROFILE_ITERATIONS", "5"))
    block_size = 64
    assert batch in (1, 8, 32) and context > 0 and context % block_size == 0 and iterations > 0

    args = Qwen36ModelArgs(mesh_device, max_batch_size=batch, max_seq_len=context + 512)
    layer_idx = next(i for i, kind in enumerate(args.attention_type_list) if kind == "full_attention")
    weights = load_attention_weights_tp(mesh_device, load_attn_layer(args.CKPT_DIR, layer_idx), args)
    attention = TPAttention(mesh_device, args, weights, get_tt_ccl(mesh_device))
    blocks_per_user = (context + 512 + block_size - 1) // block_size
    total_blocks = batch * blocks_per_user
    def make_cache():
        return ttnn.from_torch(
            torch.zeros(total_blocks, args.n_local_kv_heads, block_size, args.head_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    attention.set_paged_kv_cache(make_cache(), make_cache())
    page_table = ttnn.from_torch(
        torch.stack(
            [torch.arange(u * blocks_per_user, (u + 1) * blocks_per_user, dtype=torch.int32) for u in range(batch)]
        ),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
    )
    positions = torch.full((batch,), context - 1, dtype=torch.int32)
    positions_device = ttnn.from_torch(
        positions,
        dtype=ttnn.int32,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    cos, sin = rot_mats_decode(mesh_device, args.rope_head_dim, args.max_seq_len, args.rope_theta, positions)
    x_device = replicate_to_device(
        mesh_device, torch.randn(1, 1, batch, args.dim, dtype=torch.bfloat16)
    )

    def run_once():
        return attention.forward_decode(x_device, positions_device, cos, sin, page_table=page_table)

    out = run_once()
    ttnn.synchronize_device(mesh_device)
    samples_ms = []
    for _ in range(iterations):
        signpost("start")
        begin = time.perf_counter()
        out = run_once()
        ttnn.synchronize_device(mesh_device)
        samples_ms.append((time.perf_counter() - begin) * 1000.0)
        signpost("stop")
    logger.info(
        f"ATTENTION_DECODE_PROFILE_RESULT layer={layer_idx} batch={batch} context={context} "
        f"iterations={iterations} median_ms={statistics.median(samples_ms):.3f} "
        f"min_ms={min(samples_ms):.3f} max_ms={max(samples_ms):.3f}"
    )
    assert out.shape[-2] == batch


@torch.no_grad()
@parametrize_mesh_tp()
def test_mlp_decode_profile(mesh_device, reset_seeds, ensure_gc):
    _selected("mlp_decode")
    os.environ.setdefault("HF_MODEL", model_path())
    batch = int(os.environ.get("QWEN36_KERNEL_PROFILE_BATCH", "1"))
    iterations = int(os.environ.get("QWEN36_KERNEL_PROFILE_ITERATIONS", "10"))
    assert batch in (1, 8, 32) and iterations > 0

    args = Qwen36ModelArgs(mesh_device, max_batch_size=batch, max_seq_len=256)
    layer_idx = 0
    mlp = Qwen36MLP(
        mesh_device,
        load_mlp_layer(args.CKPT_DIR, layer_idx),
        args=args,
        tt_ccl=get_tt_ccl(mesh_device),
    )
    x_device = replicate_to_device(
        mesh_device, torch.randn(1, 1, batch, args.dim, dtype=torch.bfloat16)
    )

    out = mlp.forward(x_device)
    ttnn.synchronize_device(mesh_device)
    samples_ms = []
    for _ in range(iterations):
        signpost("start")
        begin = time.perf_counter()
        out = mlp.forward(x_device)
        ttnn.synchronize_device(mesh_device)
        samples_ms.append((time.perf_counter() - begin) * 1000.0)
        signpost("stop")
    logger.info(
        f"MLP_DECODE_PROFILE_RESULT layer={layer_idx} batch={batch} iterations={iterations} "
        f"median_ms={statistics.median(samples_ms):.3f} "
        f"min_ms={min(samples_ms):.3f} max_ms={max(samples_ms):.3f}"
    )
    assert out.shape[-2] == batch


@torch.no_grad()
@parametrize_mesh_tp()
def test_sampling_decode_profile(mesh_device, reset_seeds, ensure_gc):
    """Profile the production TP-sharded greedy sampler without loading model weights."""
    _selected("sampling_decode")
    os.environ.setdefault("HF_MODEL", model_path())
    force_argmax = os.environ.get("QWEN36_SAMPLING_FORCE_ARGMAX", "1") != "0"
    enable_trace = os.environ.get("QWEN36_SAMPLING_TRACE", "1") != "0"
    reset_params = os.environ.get("QWEN36_SAMPLING_RESET_PARAMS", "1") != "0"
    iterations = int(os.environ.get("QWEN36_KERNEL_PROFILE_ITERATIONS", "20"))
    assert iterations > 0

    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=128)
    args.model_config["SAMPLING_AG_CONFIG"]["allow_force_argmax"] = force_argmax
    sampling = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=get_tt_ccl(mesh_device))
    sampling_batch = sampling.tt_sampling.max_batch_size
    greedy_params = format_sampling_params(
        SamplingParams(temperature=0.0, top_k=1, top_p=1.0), sampling_batch
    )
    logits = torch.randn(1, 1, sampling_batch, args.padded_vocab_size, dtype=torch.bfloat16)
    logits[..., args.vocab_size :] = -float("inf")
    tt_logits = ttnn.from_torch(
        logits,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=args.cluster_shape),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Compile outside the timed interval. Explicit per-request seeds make serving run sampling eagerly;
    # QWEN36_SAMPLING_TRACE=0 measures that path, while the default measures internal trace replay.
    if reset_params:
        sampling.reset_sampling_params(greedy_params)
    sampling.sample(tt_logits, enable_trace=False)
    tt_tokens, _ = sampling.sample(tt_logits, enable_trace=enable_trace)
    ttnn.synchronize_device(mesh_device)
    samples_ms = []
    for _ in range(iterations):
        signpost("start")
        begin = time.perf_counter()
        if reset_params:
            sampling.reset_sampling_params(greedy_params)
        tt_tokens, _ = sampling.sample(tt_logits, enable_trace=enable_trace)
        ttnn.synchronize_device(mesh_device)
        samples_ms.append((time.perf_counter() - begin) * 1000.0)
        signpost("stop")

    token = int(ttnn.to_torch(ttnn.get_device_tensors(tt_tokens)[0]).reshape(-1)[0])
    expected_token = int(logits[0, 0, 0, : args.vocab_size].float().argmax())
    logger.info(
        f"SAMPLING_DECODE_PROFILE_RESULT force_argmax={force_argmax} enable_trace={enable_trace} "
        f"reset_params={reset_params} iterations={iterations} "
        f"median_ms={statistics.median(samples_ms):.3f} min_ms={min(samples_ms):.3f} "
        f"max_ms={max(samples_ms):.3f} token={token} expected_token={expected_token}"
    )
    sampling.reset_trace()
    assert token == expected_token
