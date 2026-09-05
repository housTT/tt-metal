# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Capture raw replicas and physical chunks without host fences in forward."""

import argparse
import json
from pathlib import Path

import torch

import ttnn

from ..tt.multichip_decoder import MultichipDecoder, fabric_router_config
from . import test_functional_decoder as H
from .test_optimization_experiments import recorded_activations


def summarize(parts):
    first = parts[0].float()
    rows = []
    for rank, raw in enumerate(parts):
        value = raw.float()
        valid = torch.isfinite(value) & torch.isfinite(first)
        different = (value != first) & valid
        coords = torch.nonzero(different)
        rows.append(
            dict(
                rank=rank,
                nan=int(torch.isnan(value).sum()),
                posinf=int(torch.isposinf(value).sum()),
                neginf=int(torch.isneginf(value).sum()),
                finite_differences=int(different.sum()),
                first_difference=coords[0].tolist() if len(coords) else None,
                last_difference=coords[-1].tolist() if len(coords) else None,
                max_finite_delta=float((value - first)[valid].abs().max()) if valid.any() else None,
                raw_equal=torch.equal(raw.view(torch.int16), parts[0].view(torch.int16)),
                blocks=[int(different[:, i : i + 2048].sum()) for i in range(0, value.shape[1], 2048)],
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--length", type=int, default=8001)
    parser.add_argument("--capture-chunks", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "doc/multichip_decoder"
    assert not (root / f"{args.name}.pt").exists()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, router_config=fabric_router_config())
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 * 1024 * 1024, l1_small_size=24576)
    try:
        H.FunctionalDecoder = MultichipDecoder
        decoder, table, _ = H.build_decoder(mesh, args.layer, "real", max_context=16384)
        source = recorded_activations(args.layer)[0]
        x = source[(torch.arange(args.length) + 61) % source.shape[0]].unsqueeze(0).clone()
        snapshots = []
        original_block = decoder._block

        def block(*a, **kw):
            out = original_block(*a, **kw)
            snapshots.append(ttnn.clone(out))
            return out

        if args.capture_chunks:
            decoder._block = block
        out = decoder.prefill_forward(H.to_device(mesh, x), page_table=table)
        print("FORWARD_DONE", flush=True)

        def read(v):
            return [ttnn.to_torch(t) for t in ttnn.get_device_tensors(v)]

        result = {"full": read(out)}
        print("FULL_READ_DONE", json.dumps(summarize(result["full"])), flush=True)
        for i, snapshot in enumerate(snapshots):
            result[f"physical_chunk{i}"] = read(snapshot)
            print(f"CHUNK{i}", json.dumps(summarize(result[f"physical_chunk{i}"])), flush=True)
        for offset in (0, 2048, 4096, 6144, args.length - 32):
            end = min(offset + 32, args.length)
            piece = ttnn.slice(out, [0, offset, 0], [1, end, 4096])
            result[f"slice{offset}"] = read(piece)
            print(f"SLICE{offset}", json.dumps(summarize(result[f"slice{offset}"])), flush=True)
        torch.save(result, root / f"{args.name}.pt")
        summary = {k: summarize(v) for k, v in result.items()}
        if snapshots:
            assembled = [
                torch.cat([result[f"physical_chunk{i}"][rank] for i in range(len(snapshots))], dim=1)[:, : args.length]
                for rank in range(4)
            ]
            summary["physical_concat_matches_full"] = [torch.equal(a, b) for a, b in zip(assembled, result["full"])]
            assert all(summary["physical_concat_matches_full"])
        (root / f"{args.name}.json").write_text(json.dumps(summary, indent=2) + "\n")
        assert all(torch.isfinite(p).all() and torch.equal(result["full"][0], p) for p in result["full"])
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
