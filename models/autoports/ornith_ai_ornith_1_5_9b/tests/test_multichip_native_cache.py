# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Independent last-position oracle with distinct rank-local paged histories."""

import json

import pytest
import torch

import ttnn

from . import test_functional_decoder as H
from .test_multichip_decoder import multichip_contract, pytestmark  # noqa: F401


@pytest.mark.long
@pytest.mark.timeout(600)
def test_native_local_cache_oracle(mesh_device):
    cfg = H.hf_config()
    context, heads, dim, page = cfg.max_position_embeddings, cfg.num_key_value_heads, cfg.head_dim, 64
    blocks = H.num_blocks_for_context(context)
    decoder, _, _ = H.build_decoder(mesh_device, H.FULL_LAYER, "real", max_context=context)
    rng = torch.Generator().manual_seed(970)
    table_host = torch.randperm(blocks, generator=rng).to(torch.int32)[None]
    table = H.to_device(mesh_device, table_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    history = []
    for name in ("k_cache", "v_cache"):
        # Binary fractions are exactly representable in the production BFP8 cache.
        # Distinct global heads make rank replication/ownership errors observable.
        values = torch.randint(-8, 8, (1, heads, context, dim), generator=rng).to(torch.bfloat16) / 64
        history.append(values[:, :, : context - 1].float())
        logical = values.reshape(heads, blocks, page, dim).permute(1, 0, 2, 3).contiguous()
        physical = torch.empty_like(logical)
        physical[table_host[0].long()] = logical
        ttnn.deallocate(getattr(decoder, name))
        setattr(
            decoder,
            name,
            ttnn.from_torch(
                physical,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
            ),
        )
        assert list(getattr(decoder, name).shape) == [blocks, 1, page, dim]
        del values, logical, physical
    cache = H.R._new_cache(cfg)
    cache.update(history[0], history[1], H.FULL_LAYER)
    x = H.make_activations(1, 1, seed=971)
    position = torch.tensor([context - 1])
    with torch.no_grad():
        golden = H.R.reference_decode(H.reference_layer(H.FULL_LAYER, "real"), cfg, x.float(), position, cache)
    del history, cache
    x_buf = H.to_device(mesh_device, x)
    pos, rot = H.decode_inputs(mesh_device, position)

    def forward():
        return decoder.decode_forward(x_buf, current_pos=pos, rot_idxs=rot, page_table=table)

    eager = ttnn.to_torch(forward())
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = forward()
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
    actual = ttnn.to_torch(out)
    value = H.pcc(golden, actual)
    assert torch.equal(eager, actual)
    assert torch.isfinite(actual).all() and value >= H.PCC_BAR
    print(
        json.dumps(
            dict(
                native_cache_hf_pcc=value,
                position=context - 1,
                local_kv_heads=1,
                page_permutation=True,
                cache_dtype="BFP8",
                eager_trace_exact=True,
                historical_fixture_not_hf_prefill=True,
            )
        ),
        flush=True,
    )
    ttnn.release_trace(mesh_device, trace)
