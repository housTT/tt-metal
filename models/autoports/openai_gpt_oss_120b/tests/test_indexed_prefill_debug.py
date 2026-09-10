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
    print("indices shape", tuple(d["expert_indices"].shape), "min/max", int(indices.min()), int(indices.max()))
    ref_counts = torch.bincount(indices.reshape(-1), minlength=experts)
    counts = d["counts"].reshape(-1)[:experts].to(torch.int64)
    print("counts match:", torch.equal(counts, ref_counts), "max", int(ref_counts.max()), "dev max", int(counts.max()))

    def slabs(c):
        r = max(32, ((c + 31) // 32) * 32)
        r = 1 << (r - 1).bit_length()
        if r <= 256:
            return r, 1
        return 256, (c + 255) // 256

    groups = {}
    for e, c in enumerate(ref_counts.tolist()):
        if c > 0:
            h, n = slabs(int(c))
            groups.setdefault(h, []).extend([e] * n)
    base = torch.zeros(experts, dtype=torch.int64)
    total = 0
    for h in sorted(groups):
        seen = set()
        for e in groups[h]:
            if e not in seen:
                base[e] = total
                seen.add(e)
            total += h
        padded = 1 << (len(groups[h]) - 1).bit_length()
        total += h * (padded - len(groups[h]))
    print(
        "slab heights (nonzero) distinct:",
        sorted(groups),
        "total capacity",
        total,
        "slots",
        indices.numel(),
        "max count",
        int(ref_counts.max()),
    )
    flat = indices.reshape(-1)
    slots = flat.numel()
    ref_sorted, ref_perm = torch.sort(flat, stable=True)
    sorted_dev = d["sorted_experts"].reshape(-1)[:slots].to(torch.int64)
    perm_dev = d["permutation"].reshape(-1)[:slots].to(torch.int64)
    print("sorted match:", torch.equal(sorted_dev, ref_sorted), "perm valid:", torch.equal(flat[perm_dev], sorted_dev))
    ref_offsets = torch.cumsum(ref_counts, 0) - ref_counts
    offsets_dev = d["token_offsets"].reshape(-1)[:experts].to(torch.int64)
    print("offsets match:", torch.equal(offsets_dev, ref_offsets))
    positions = torch.arange(slots)
    ref_dest = base[sorted_dev] + (positions - ref_offsets[sorted_dev])
    dest_dev = d["destinations"].reshape(-1)[:slots].to(torch.int64)
    print("dest match:", torch.equal(dest_dev, ref_dest), "dest max", int(dest_dev.max()), "capacity", total)
    assert torch.equal(dest_dev, ref_dest)
    capacity_total = 1 << (total - 1).bit_length()  # the path rounds the slab buffer to a power of two
    ref_dispatch = torch.zeros(capacity_total, dtype=torch.int64)
    ref_dispatch[ref_dest] = perm_dev // top_k
    dispatch_dev = d["dispatch_rows"].reshape(-1)[:capacity_total].to(torch.int64)
    print(
        "dispatch match:",
        torch.equal(dispatch_dev, ref_dispatch),
        "mismatches",
        int((dispatch_dev != ref_dispatch).sum()),
    )
    ref_slot_to_dest = torch.zeros(slots, dtype=torch.int64)
    ref_slot_to_dest[perm_dev] = dest_dev
    s2d_dev = d["slot_to_destination"].reshape(-1)[:slots].to(torch.int64)
    print("slot_to_dest match:", torch.equal(s2d_dev, ref_slot_to_dest))
    scores = d["routing_scores"].float()[:rows, :top_k].reshape(-1)
    ref_rw = torch.zeros(capacity_total)
    ref_rw[dest_dev] = scores[perm_dev]
    rw_flat = d["row_weights_flat"].float().reshape(-1)[:capacity_total]
    print("row_weights_flat max abs diff:", float((rw_flat - ref_rw).abs().max()))
    disp = d["dispatched"].float().reshape(capacity_total, -1)
    ref_disp = host[0, 0].float()[dispatch_dev]
    print("dispatched max abs diff:", float((disp - ref_disp).abs().max()))
    ttnn.synchronize_device(mesh_device)
