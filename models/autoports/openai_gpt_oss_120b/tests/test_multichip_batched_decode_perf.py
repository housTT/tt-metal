# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Per-layer traced decode latency of the TP4 multichip decoder at batch > 1.

The acceptance tests in ``test_multichip_decoder.py`` only time decode at
batch 1. This module times one real-weight decoder layer at several decode
batch widths so the batched MoE path can be measured before and after it
changes. It prints one ``BATCHED_DECODE`` line per case.

Run (serialized on hardware):

    env GPT_OSS_120B_BATCHED_DECODE_PERF=1 \
        GPT_OSS_120B_SNAPSHOT=<snapshot dir> \
        scripts/run_safe_pytest.sh \
        models/autoports/openai_gpt_oss_120b/tests/test_multichip_batched_decode_perf.py -q -s

Optional: ``GPT_OSS_120B_BATCHED_DECODE_CHECK=1`` also compares every batched
row against a batch-1 decode of the same row through the same layer instance
and prints the PCC per row (slow: one extra traced decode per user).
"""

from __future__ import annotations

import gc
import os
import statistics
import time
from pathlib import Path

import pytest
import torch
from tracy import signpost

import ttnn
from models.autoports.openai_gpt_oss_120b.tests import test_multichip_decoder as tmd
from models.autoports.openai_gpt_oss_120b.tests.real_weight_utils import load_real_layer_state_dict
from models.common.utility_functions import comp_pcc
from models.demos.utils.trace_region_sizes import TRACE_MODEL_KEY_PARAM

RUN_PERF = os.environ.get("GPT_OSS_120B_BATCHED_DECODE_PERF") == "1"
RUN_CHECK = os.environ.get("GPT_OSS_120B_BATCHED_DECODE_CHECK") == "1"
REAL_WEIGHT_SNAPSHOT = os.environ.get("GPT_OSS_120B_SNAPSHOT")
ROW_PCC_THRESHOLD = 0.99


def _decode_case_inputs(config, batch_size, layer_idx, *, position_base=200):
    """Random decode rows for ``batch_size`` users at distinct short positions."""
    generator = torch.Generator().manual_seed(4_120_000 + 100 * layer_idx + batch_size)
    hidden = (torch.randn((1, 1, batch_size, config.hidden_size), generator=generator) * 0.02).to(torch.bfloat16)
    positions = torch.arange(batch_size, dtype=torch.long) * 7 + position_base
    page_table = tmd._host_batch_page_table(config, batch_size, seed=6_120_000 + 100 * layer_idx + batch_size)
    return hidden, positions, page_table


@pytest.mark.skipif(
    not RUN_PERF or not REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_BATCHED_DECODE_PERF=1 and GPT_OSS_120B_SNAPSHOT",
)
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("batch_size", [1, 2, 8, 32], ids=["b1", "b2", "b8", "b32"])
@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding", "full"])
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
def test_batched_decode_layer_latency(mesh_device, device_params, layer_idx, batch_size, reset_seeds):
    del device_params, reset_seeds
    snapshot = Path(REAL_WEIGHT_SNAPSHOT)
    config = tmd._config()
    logical_tp = int(mesh_device.shape[1])
    assert tuple(mesh_device.shape) == (1, 4)

    state_dict = load_real_layer_state_dict(snapshot, layer_idx)
    decoder = tmd._constructor(
        state_dict,
        config,
        layer_idx,
        mesh_device,
        tmd._cache_root(layer_idx, f"multichip_acceptance_tp{logical_tp}"),
        max_batch_size=batch_size,
    )
    assert not decoder.is_single_chip_baseline

    hidden_host, positions_host, page_table_host = _decode_case_inputs(config, batch_size, layer_idx)
    hidden, rope, current_position = tmd._decode_inputs(config, mesh_device, hidden_host, positions_host)
    page_table = tmd._replicated_from_torch(
        page_table_host,
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    trace_id, output = tmd._capture_decode(decoder, mesh_device, hidden, rope, current_position, page_table)
    output_host = tmd._assert_replicated(output, (1, 1, batch_size, config.hidden_size))[0, 0, :batch_size].clone()
    assert torch.isfinite(output_host.float()).all(), "batched decode produced non-finite values"
    ttnn.ReadDeviceProfiler(mesh_device)

    repeats, samples_ms, median_ms = tmd._warmed_trace_latency_samples(mesh_device, trace_id, signposted=True)
    ttnn.release_trace(mesh_device, trace_id)

    print(
        f"BATCHED_DECODE mesh=1x{logical_tp} layer={layer_idx} type={config.layer_types[layer_idx]} "
        f"batch={batch_size} median_ms={median_ms:.6f} per_user_ms={median_ms / batch_size:.6f} "
        f"repeats={repeats} samples={samples_ms}"
    )

    row_pcc = []
    if RUN_CHECK and batch_size > 1:
        # Isolate the MoE block: run the same [1, 1, B, H] activation through the
        # grouped path, the per-user loop, and one batch-1 indexed decode per row
        # on the same layer instance, then compare row by row.
        mlp = decoder.mlp
        generator = torch.Generator().manual_seed(9_120_000 + 100 * layer_idx + batch_size)
        moe_host = (torch.randn((1, 1, batch_size, config.hidden_size), generator=generator) * 0.5).to(torch.bfloat16)
        moe_input = tmd._replicated_from_torch(moe_host, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        def moe_rows(flag):
            previous = mlp.grouped_decode_batch
            mlp.grouped_decode_batch = flag
            try:
                out = mlp(moe_input, is_decode=True)
            finally:
                mlp.grouped_decode_batch = previous
            ttnn.synchronize_device(mesh_device)
            host = tmd._assert_replicated(out, (1, 1, batch_size, config.hidden_size))[0, 0, :batch_size].clone()
            out.deallocate(True)
            return host

        grouped_rows = moe_rows(True)
        loop_rows = moe_rows(False)
        for user in range(batch_size):
            single_input = tmd._replicated_from_torch(
                moe_host[:, :, user : user + 1, :], mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
            single_out = mlp(single_input, is_decode=True)
            ttnn.synchronize_device(mesh_device)
            single_row = tmd._assert_replicated(single_out, (1, 1, 1, config.hidden_size))[0, 0, :1].clone()
            single_out.deallocate(True)
            single_input.deallocate(True)
            pcc_loop = comp_pcc(single_row.float(), loop_rows[user : user + 1].float(), ROW_PCC_THRESHOLD)
            pcc_grouped = comp_pcc(single_row.float(), grouped_rows[user : user + 1].float(), ROW_PCC_THRESHOLD)
            row_pcc.append({"user": user, "loop": pcc_loop[1], "grouped": pcc_grouped[1], "ok": pcc_grouped[0]})
        moe_input.deallocate(True)
        print(f"BATCHED_DECODE_ROWS layer={layer_idx} batch={batch_size} row_pcc={row_pcc}")
        failing = [entry["user"] for entry in row_pcc if not entry["ok"]]
        assert not failing, f"grouped MoE rows differ from batch-1 indexed decode for users {failing}"

    del decoder, state_dict
    gc.collect()


def real_token_embeddings(snapshot, config, num_tokens, seed):
    """Return [1, 1, num_tokens, hidden] bf16 embedding rows of random real token ids."""
    import json

    from safetensors import safe_open

    index = json.load(open(Path(snapshot) / "model.safetensors.index.json", encoding="utf-8"))
    shard = index["weight_map"]["model.embed_tokens.weight"]
    with safe_open(str(Path(snapshot) / shard), framework="pt", device="cpu") as handle:
        table = handle.get_tensor("model.embed_tokens.weight")
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, int(config.vocab_size), (num_tokens,), generator=generator)
    return table[ids].to(torch.bfloat16).reshape(1, 1, num_tokens, config.hidden_size)


@pytest.mark.skipif(
    not RUN_PERF or not REAL_WEIGHT_SNAPSHOT,
    reason="set GPT_OSS_120B_BATCHED_DECODE_PERF=1 and GPT_OSS_120B_SNAPSHOT",
)
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("sequence_length", [128, 1024, 4096, 16384], ids=["s128", "s1024", "s4096", "s16384"])
@pytest.mark.parametrize("layer_idx", [0], ids=["sliding"])
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
def test_indexed_prefill_moe(mesh_device, device_params, layer_idx, sequence_length, reset_seeds):
    """Time and cross-check the MoE block prefill: packed group-sparse vs indexed per-expert."""
    del device_params, reset_seeds
    snapshot = Path(REAL_WEIGHT_SNAPSHOT)
    config = tmd._config()
    logical_tp = int(mesh_device.shape[1])
    state_dict = load_real_layer_state_dict(snapshot, layer_idx)
    decoder = tmd._constructor(
        state_dict,
        config,
        layer_idx,
        mesh_device,
        tmd._cache_root(layer_idx, f"multichip_acceptance_tp{logical_tp}"),
        max_batch_size=1,
    )
    mlp = decoder.mlp
    mlp.indexed_prefill_min_tokens = 0
    # Optional tuning overrides for stage sweeps.
    if os.environ.get("GPT_OSS_120B_IDX_GU"):
        mlp.indexed_prefill_gate_up_blocking = tuple(int(v) for v in os.environ["GPT_OSS_120B_IDX_GU"].split(","))
    if os.environ.get("GPT_OSS_120B_IDX_DN"):
        mlp.indexed_prefill_down_blocking = tuple(int(v) for v in os.environ["GPT_OSS_120B_IDX_DN"].split(","))
    if os.environ.get("GPT_OSS_120B_IDX_SLAB_BFP8") == "1":
        mlp.indexed_prefill_slab_dtype = ttnn.bfloat8_b
    if os.environ.get("GPT_OSS_120B_IDX_SLAB_ROWS"):
        type(mlp)._PREFILL_MAX_SLAB_ROWS = int(os.environ["GPT_OSS_120B_IDX_SLAB_ROWS"])
    host = real_token_embeddings(snapshot, config, sequence_length, 7_120_000 + sequence_length)
    page_table_host = tmd._host_batch_page_table(config, 1, seed=8_120_000 + sequence_length)
    prefill_hidden, prefill_rope, page_table = tmd._prefill_inputs(config, mesh_device, host, page_table_host)

    def run(flag, repeats):
        previous = mlp.indexed_prefill
        mlp.indexed_prefill = flag
        try:
            out = decoder.prefill_forward(prefill_hidden, position_embeddings=prefill_rope, page_table=page_table)
            ttnn.synchronize_device(mesh_device)
            result = tmd._assert_replicated(out, (1, 1, sequence_length, config.hidden_size))[0, 0].clone()
            out.deallocate(True)
            samples = []
            signpost("PERF_PREFILL_INDEXED" if flag else "PERF_PREFILL_PACKED")
            for _ in range(repeats):
                started = time.perf_counter()
                out = decoder.prefill_forward(prefill_hidden, position_embeddings=prefill_rope, page_table=page_table)
                ttnn.synchronize_device(mesh_device)
                samples.append((time.perf_counter() - started) * 1000)
                out.deallocate(True)
            signpost("PERF_PREFILL_INDEXED_END" if flag else "PERF_PREFILL_PACKED_END")
        finally:
            mlp.indexed_prefill = previous
        return result, samples

    repeats = 3 if sequence_length <= 1024 else 2
    packed_out, packed_ms = run(False, repeats)
    indexed_out, indexed_ms = run(True, repeats)
    passing, detail = comp_pcc(packed_out.float(), indexed_out.float(), 0.99)
    if os.environ.get("GPT_OSS_120B_PREFILL_STAGES") == "1":
        mlp.indexed_prefill = True
        mlp.indexed_prefill_min_tokens = 0
        for _ in range(2):  # the second pass is warm (programs compiled)
            mlp._prefill_timing = {}
            out = decoder.prefill_forward(prefill_hidden, position_embeddings=prefill_rope, page_table=page_table)
        ttnn.synchronize_device(mesh_device)
        out.deallocate(True)
        stages = {k: round(v * 1000, 2) for k, v in mlp._prefill_timing.items()}
        mlp._prefill_timing = None
        print(f"INDEXED_PREFILL_STAGES sequence={sequence_length} total_ms={sum(stages.values()):.2f} {stages}")
        print(f"INDEXED_PREFILL_LAYOUT sequence={sequence_length} {getattr(mlp, '_prefill_stats', None)}")
    print(
        f"INDEXED_PREFILL layer={layer_idx} sequence={sequence_length} "
        f"packed_ms={statistics.median(packed_ms):.3f} indexed_ms={statistics.median(indexed_ms):.3f} "
        f"speedup={statistics.median(packed_ms) / statistics.median(indexed_ms):.3f} pcc={detail}"
    )
    assert torch.isfinite(indexed_out.float()).all()
    assert passing, f"indexed prefill layer output differs from packed prefill: {detail}"
    del decoder, state_dict
    gc.collect()
