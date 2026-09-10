# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Check the indexed prefill routing intermediates against a torch reference."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.openai_gpt_oss_120b.tests import test_multichip_decoder as tmd
from models.autoports.openai_gpt_oss_120b.tests.real_weight_utils import load_real_layer_state_dict
from models.demos.utils.trace_region_sizes import TRACE_MODEL_KEY_PARAM

RUN = os.environ.get("GPT_OSS_120B_BATCHED_DECODE_PERF") == "1"
SNAPSHOT = os.environ.get("GPT_OSS_120B_SNAPSHOT")


@pytest.mark.skipif(not RUN or not SNAPSHOT, reason="set GPT_OSS_120B_BATCHED_DECODE_PERF=1 and GPT_OSS_120B_SNAPSHOT")
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("sequence_length", [128, 1024, 4096], ids=["s128", "s1024", "s4096"])
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
def test_indexed_prefill_routing_intermediates(mesh_device, device_params, sequence_length, reset_seeds):
    del device_params, reset_seeds
    config = tmd._config()
    state_dict = load_real_layer_state_dict(Path(SNAPSHOT), 0)
    decoder = tmd._constructor(state_dict, config, 0, mesh_device, tmd._cache_root(0, "multichip_acceptance_tp4"))
    mlp = decoder.mlp
    from models.autoports.openai_gpt_oss_120b.tests.test_multichip_batched_decode_perf import real_token_embeddings

    host = real_token_embeddings(SNAPSHOT, config, sequence_length, 7_120_000 + sequence_length)
    moe_input = tmd._replicated_from_torch(host, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    mlp._prefill_debug = {}
    mlp.indexed_prefill = True
    mlp.indexed_prefill_min_tokens = 0
    out = mlp(moe_input, is_decode=False)
    ttnn.synchronize_device(mesh_device)
    out.deallocate(True)
    d = mlp._prefill_debug
    mlp._prefill_debug = None
    mlp.indexed_prefill = False

    rows = sequence_length
    top_k = mlp.top_k
    experts = mlp.num_experts
    indices = d["expert_indices"].to(torch.int64)[:rows, :top_k]
    scores = d["routing_scores"].float()[:rows, :top_k]
    layout = d["layout"]
    capacity = d["capacity"]
    dispatch_host = d["dispatch_rows_host"].to(torch.int64)
    dispatch_dev = d["dispatch_rows"].reshape(-1)[:capacity].to(torch.int64)
    slot_columns = d["slot_columns"].to(torch.int64)  # [top_k, rows]
    print("indices min/max", int(indices.min()), int(indices.max()), "capacity", capacity, "groups", len(layout))
    assert torch.equal(dispatch_dev, dispatch_host), "uploaded dispatch rows differ from the host layout"
    assert capacity % 32 == 0

    # Which expert owns every slab row.
    owner = torch.full((capacity,), -1, dtype=torch.int64)
    used = torch.zeros(capacity, dtype=torch.bool)
    for height, members, start, _id_offset in layout:
        assert height % 32 == 0 and start % 32 == 0
        assert len(members) & (len(members) - 1) == 0, "group sizes must be powers of two"
        span = height * len(members)
        assert not used[start : start + span].any(), "overlapping slabs"
        used[start : start + span] = True
        owner[start : start + span] = torch.tensor(members, dtype=torch.int64).repeat_interleave(height)
    print("slab rows used", int(used.sum()), "of", capacity)

    # Every (token, slot) lands in a distinct row of a slab owned by its expert,
    # and that row's dispatch entry points back at the token.
    destinations = slot_columns.transpose(0, 1).reshape(-1)  # token-major, like indices
    assert destinations.numel() == torch.unique(destinations).numel(), "two slots share a slab row"
    assert torch.equal(owner[destinations], indices.reshape(-1)), "slot placed in another expert's slab"
    tokens = torch.arange(rows, dtype=torch.int64).repeat_interleave(top_k)
    assert torch.equal(dispatch_host[destinations], tokens), "dispatch row does not gather the slot's token"
    counts = torch.bincount(indices.reshape(-1), minlength=experts)
    slab_rows = torch.bincount(owner[owner >= 0], minlength=experts)
    assert bool((slab_rows >= counts).all()), "an expert has fewer slab rows than tokens"
    waste = (slab_rows - counts).sum().item()
    print("padded slab rows", waste, "of", int(slab_rows.sum()), "max count", int(counts.max()))
    assert abs(float(scores.sum(dim=-1).mean()) - 1.0) < 1e-2, "scores are not a softmax over the top-k"
    ttnn.synchronize_device(mesh_device)
