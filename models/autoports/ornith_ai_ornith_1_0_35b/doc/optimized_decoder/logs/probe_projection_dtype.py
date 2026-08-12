# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""OPT-007: the BFP4 dense-projection candidate, decided on a real-weight PCC ladder.

`proj_dtype = bfloat4_b` (the packed attention in-projection and ``o_proj`` on ``full_attention``
layers, the packed DeltaNet in-projection and ``out_proj`` on ``linear_attention`` ones) is faster
than the selected BFP8 policy. OPT-007 requires the trial on **real target-model weights**, and it
requires the accept/reject decision to rest on model-visible correctness rather than on a synthetic
probe — so this runs the same HF-golden ladder the delivered suite runs, over the same prefill
lengths and the same four decode steps, for both dtypes, in one process on one device.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_decoder/logs/probe_projection_dtype.py

Every row is `PROJDTYPE <dtype> layer=<idx> <case> pcc=<value> bar=0.995`.
"""

from __future__ import annotations

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R
from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import (
    POLICIES,
    OptimizedDecoder,
    num_blocks_for_context,
)

CTX = 8192
BAR = 0.995
PREFILL_LENS = [1, 7, 32, 64, 128, 129, 250, 2048, 2049, 3000]
DECODE_PREFILL_LEN = 130
DECODE_STEPS = 4
LAYERS = {0: "linear_attention", 3: "full_attention"}


def pcc(golden: torch.Tensor, actual: torch.Tensor) -> float:
    a = golden.double().flatten()
    b = actual.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def activations(batch, seq_len, hidden, seed):
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(batch, seq_len, hidden, generator=gen) * 0.5).to(torch.bfloat16)


def dev(mesh, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def build(mesh, cfg, sd, layer_idx, policy):
    decoder = OptimizedDecoder.from_state_dict(
        sd, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh, max_context=CTX, policy=policy
    )
    blocks = num_blocks_for_context(CTX)
    decoder.allocate_kv_cache(blocks)
    decoder.allocate_state(1)
    page_table = None
    if decoder.is_full_attention:
        page_table = dev(
            mesh,
            torch.arange(blocks, dtype=torch.int32).reshape(1, blocks),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
    return decoder, page_table


def main():
    cfg = R.load_text_config()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0)
    base = POLICIES["optimized"]
    arms = [("bfloat8_b", base), ("bfloat4_b", base.replace(proj_dtype=ttnn.bfloat4_b))]
    try:
        for layer_idx, kind in LAYERS.items():
            sd = R.load_layer_state_dict(layer_idx)
            ref = R.build_reference_layer(cfg, layer_idx, sd)
            goldens = {}
            with torch.no_grad():
                for seq_len in PREFILL_LENS:
                    x = activations(1, seq_len, cfg.hidden_size, seed=seq_len)
                    goldens[("prefill", seq_len)] = (x, R.reference_prefill(ref, cfg, x.float())[0])
                x = activations(1, DECODE_PREFILL_LEN, cfg.hidden_size, seed=DECODE_PREFILL_LEN)
                _, cache = R.reference_prefill(ref, cfg, x.float())
                steps = []
                for step in range(DECODE_STEPS):
                    d = activations(1, 1, cfg.hidden_size, seed=1000 + step)
                    pos = torch.tensor([DECODE_PREFILL_LEN + step], dtype=torch.long)
                    steps.append((d, R.reference_decode(ref, cfg, d.float(), pos, cache)))
                goldens[("decode", DECODE_PREFILL_LEN)] = (x, steps)

            for name, policy in arms:
                for seq_len in PREFILL_LENS:
                    decoder, page_table = build(mesh, cfg, sd, layer_idx, policy)
                    x, golden = goldens[("prefill", seq_len)]
                    out = decoder.prefill_forward(dev(mesh, x), page_table=page_table)
                    value = pcc(golden, ttnn.to_torch(out).float().reshape(golden.shape))
                    ttnn.deallocate(out)
                    print(
                        f"PROJDTYPE {name} layer={layer_idx} ({kind}) prefill seq_len={seq_len} "
                        f"pcc={value:.6f} bar={BAR} {'PASS' if value > BAR else 'FAIL'}",
                        flush=True,
                    )
                    del decoder, page_table

                decoder, page_table = build(mesh, cfg, sd, layer_idx, policy)
                x, steps = goldens[("decode", DECODE_PREFILL_LEN)]
                ttnn.deallocate(decoder.prefill_forward(dev(mesh, x), page_table=page_table))
                for step, (d, golden) in enumerate(steps):
                    pos = torch.tensor([DECODE_PREFILL_LEN + step], dtype=torch.int32)
                    out = decoder.decode_forward(
                        dev(mesh, d),
                        current_pos=dev(mesh, pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
                        rot_idxs=dev(mesh, pos.reshape(1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
                        page_table=page_table,
                    )
                    value = pcc(golden, ttnn.to_torch(out).float().reshape(golden.shape))
                    ttnn.deallocate(out)
                    print(
                        f"PROJDTYPE {name} layer={layer_idx} ({kind}) decode step={step} "
                        f"pcc={value:.6f} bar={BAR} {'PASS' if value > BAR else 'FAIL'}",
                        flush=True,
                    )
                del decoder, page_table
            del ref, sd, goldens
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
