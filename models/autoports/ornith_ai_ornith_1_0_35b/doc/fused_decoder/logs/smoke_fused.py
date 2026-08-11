# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fast HF-vs-fused smoke check used while developing the rewrites (the durable coverage is
``tests/test_fused_decoder.py``). Run:

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/fused_decoder/logs/smoke_fused.py [seq_len ...]
"""

import sys

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R
from models.autoports.ornith_ai_ornith_1_0_35b.tt.fused_decoder import FusedDecoder, num_blocks_for_context

CTX = 8192


def pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def dev(mesh, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def main():
    seq_lens = [int(a) for a in sys.argv[1:]] or [128, 300]
    cfg = R.load_text_config()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0)
    try:
        for layer_idx in (0, 3):
            sd = R.load_layer_state_dict(layer_idx)
            ref = R.build_reference_layer(cfg, layer_idx, sd)
            decoder = FusedDecoder.from_state_dict(
                sd, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh, max_context=CTX
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
            for seq_len in seq_lens:
                decoder.reset_state()
                if decoder.is_full_attention:
                    decoder.allocate_kv_cache(blocks)
                gen = torch.Generator().manual_seed(seq_len)
                x = (torch.randn(1, seq_len, cfg.hidden_size, generator=gen) * 0.5).to(torch.bfloat16)
                dx = (torch.randn(1, 1, cfg.hidden_size, generator=gen) * 0.5).to(torch.bfloat16)
                with torch.no_grad():
                    ref_pre, cache = R.reference_prefill(ref, cfg, x.float())
                    ref_dec = R.reference_decode(ref, cfg, dx.float(), torch.tensor([seq_len]), cache)
                out = decoder.prefill_forward(dev(mesh, x), page_table=page_table)
                p = pcc(ref_pre, ttnn.to_torch(out))
                ttnn.deallocate(out)
                pos = torch.tensor([seq_len])
                cur = dev(mesh, pos.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
                rot = dev(mesh, pos.to(torch.int32).reshape(1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
                out = decoder.decode_forward(dev(mesh, dx), current_pos=cur, rot_idxs=rot, page_table=page_table)
                d = pcc(ref_dec, ttnn.to_torch(out))
                ttnn.deallocate(out)
                print(f"SMOKE layer={layer_idx} seq_len={seq_len} prefill_pcc={p:.6f} decode_pcc={d:.6f}", flush=True)
            del decoder, ref, sd
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
