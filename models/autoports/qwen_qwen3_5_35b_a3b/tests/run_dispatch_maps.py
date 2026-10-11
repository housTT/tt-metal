# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Validate every real route before a native expert reader consumes its offset."""

import argparse
import json

import torch
import ttnn

from ..tt.optimized_decoder import OptimizationPolicy, _LocalDispatch
from .runtime_audit import device_only


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--rank", choices=("sort", "cumsum", "matmul"), default="sort")
    p.add_argument("--combine", choices=("post_combine", "tiles"), default="post_combine")
    args = p.parse_args()
    torch.manual_seed(27)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    results = []
    try:
        for length in (32,64,128,256,512,1024,2048,4096):
            dispatcher = _LocalDispatch.__new__(_LocalDispatch)
            dispatcher.policy = OptimizationPolicy(prefill_chunk=length, dispatch_rank=args.rank,
                                                   dispatch_combine=args.combine)
            dispatcher.prepare_maps(mesh)
            cases = {"hot": torch.arange(8).repeat(length, 1),
                     "random": torch.rand(length, 256).argsort(-1)[:, :8]}
            boundary = torch.arange(8).repeat(length, 1)
            boundary[:, 0] = 255
            boundary[:31, 0], boundary[31:63, 0], boundary[63:96, 0] = 8, 9, 10
            cases["boundary"] = boundary
            if length >= 1024:
                native_boundary = torch.arange(8).repeat(length,1)
                native_boundary[:,0] = 255
                native_boundary[:255,0] = 8
                native_boundary[255:511,0] = 9
                native_boundary[511:768,0] = 10
                cases["native_block_boundary"] = native_boundary
            for name, indices in cases.items():
                ids = ttnn.from_torch(indices.reshape(1, 1, length, 8), device=mesh,
                    dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                with device_only():
                    counts, offsets, forward, inverse, capacity = dispatcher.route_maps((None, ids))
                actual = [ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).long().flatten()
                          for t in (counts, offsets, forward, inverse)]
                c, r, f, inv = actual
                expected_c = indices.flatten().bincount(minlength=256)
                aligned = ((expected_c + 31) // 32) * 32
                expected_r = aligned.cumsum(0) - aligned
                assert torch.equal(c, expected_c)
                assert torch.equal(r, expected_r)
                assert c.sum() == 8 * length
                assert ((r + aligned) <= capacity).all()
                assert ((inv >= 0) & (inv < capacity)).all()
                assert inv.unique().numel() == 8 * length
                kmajor = args.combine == "tiles"
                tokens = torch.arange(8 * length) % length if kmajor else torch.arange(8 * length) // 8
                assert torch.equal(f[inv], tokens)
                route_experts = indices.T.flatten() if kmajor else indices.flatten()
                assert ((inv >= r[route_experts]) & (inv < r[route_experts] + c[route_experts])).all()
                for expert in route_experts.unique():
                    mask = route_experts == expert
                    assert (inv[mask][tokens[mask].argsort()].diff() > 0).all()
                results.append(dict(length=length, case=name, routes=8 * length, capacity=capacity,
                                    occupied_rows=int(aligned.sum()), status="pass", runtime_audit=True,
                                    rank=args.rank, combine=args.combine, stable_token_order=True))
                print(results[-1], flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    with open(args.output, "w") as stream:
        json.dump(results, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
