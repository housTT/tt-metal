# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=int, choices=[1, 2, 4], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--rows", default="4,32")
    args = p.parse_args()
    import ttnn

    from ..tt.generator import build_generator
    from .mesh_util import ReplicaReader, open_mesh

    mesh = open_mesh(args.mesh)
    read = ReplicaReader()
    result = dict(vars(args), output=str(args.output), cases={})
    try:
        gen = build_generator(Path(__file__).parents[1], mesh, layer_ids=(0, 3), decode_batches=(1, *(int(r) for r in args.rows.split(","))))
        layer = gen.model.layers[args.layer]
        from ..tt import gdn_batch

        probes = []

        def finite(t):
            v = read(t).float()
            return dict(shape=list(v.shape), finite=bool(torch.isfinite(v).all()), absmax=float(v[torch.isfinite(v)].abs().max()) if torch.isfinite(v).any() else None)

        conv_original, decode_original = gdn_batch.conv_batch, gdn_batch.gdn_decode_batch

        def conv_spy(mega, taps, conv_states, width):
            out = conv_original(mega, taps, conv_states, width)
            probes.append(dict(op="conv", rows=len(conv_states), mega=finite(mega), out=finite(out),
                               mega_memory=str(mega.memory_config().memory_layout), mega_padded=str(mega.padded_shape)))
            return out

        def decode_spy(qkv, mega, states, *a, **kw):
            out = decode_original(qkv, mega, states, *a, **kw)
            probes.append(dict(op="decode", rows=len(states), out=finite(out), state0=finite(states[0])))
            return out

        gdn_batch.conv_batch, gdn_batch.gdn_decode_batch = conv_spy, decode_spy
        g = torch.Generator().manual_seed(7)

        def put(tensor, like):
            return ttnn.from_torch(tensor, dtype=like.dtype, layout=like.layout, device=mesh,
                                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh), memory_config=like.memory_config())

        for rows in (int(r) for r in args.rows.split(",")):
            states = layer._states[:rows] if hasattr(layer, "_states") else []
            init = [(torch.randn(tuple(s.shape), generator=g) * 0.05, torch.randn(tuple(c.shape), generator=g))
                    for s, c in states]
            hidden = torch.randn(rows, 1, 2048, generator=g).to(torch.bfloat16)

            def load():
                for (s, c), (hs, hc) in zip(states, init):
                    ttnn.copy(put(hs, s), s)
                    ttnn.copy(put(hc, c), c)
                return ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                       mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

            outs = {}
            for mode in (False, True):
                layer.batched_decode = mode
                x = load()
                kw = {}
                if not states:
                    positions = torch.full((rows,), 37, dtype=torch.int32)
                    kw = dict(position_tensor=ttnn.from_torch(positions, dtype=ttnn.int32, device=mesh,
                                                              mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)),
                              page_table=gen.tt_page_tables[rows])
                y = layer.decode_forward(x, request_slots=list(range(rows)), **kw)
                outs[mode] = read(y).float().reshape(rows, -1)
                outs[(mode, "state")] = [read(s).float() for s, _ in states]
            a, b = outs[False], outs[True]
            h = hidden.float().reshape(rows, -1)
            result["cases"][str(rows)] = dict(
                row_pcc=[float(torch.corrcoef(torch.stack([a[r], b[r]]))[0, 1]) for r in range(rows)],
                delta_pcc=[float(torch.corrcoef(torch.stack([a[r] - h[r], b[r] - h[r]]))[0, 1]) for r in range(rows)],
                batched_finite=bool(torch.isfinite(b).all()), batched_absmax=float(b.abs().max()),
                ref_absmax=float(a.abs().max()),
                state_max_abs=max((float((x - y).abs().max()) for x, y in zip(outs[(False, "state")], outs[(True, "state")])),
                                  default=0.0))
            print(json.dumps({k: (v if k not in ("row_pcc", "delta_pcc") else [round(x, 4) for x in v]) for k, v in result["cases"][str(rows)].items()}), flush=True)
            result["cases"][str(rows)]["probes"] = list(probes)
            print(json.dumps(probes), flush=True)
            probes.clear()
    finally:
        args.output.write_text(json.dumps(result, indent=1))
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
