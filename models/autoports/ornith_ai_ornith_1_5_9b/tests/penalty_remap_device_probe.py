# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact common-sampler penalty-row remap probe; supervising hardware lane only."""

import argparse
import json
from pathlib import Path

import torch

import ttnn

from ..reference.hf_reference import load_text_config
from ..tt.generator import OrnithGenerator
from ..tt.model import OrnithModel, SamplingCCL, close_ornith_mesh, open_ornith_mesh


def broadcast_rows(gen, target, values, *, use_repeat=False):
    shape = list(target.shape)
    assert len(shape) == 2
    moves = [(row, source) for row, source in enumerate(values) if row != source]
    saved = {source: ttnn.slice(target, [source, 0], [source + 1, shape[1]]) for _, source in moves}
    for row, source in moves:
        mask = torch.zeros(shape[0], 1, dtype=torch.int32)
        mask[row] = 1
        select = gen.model.upload(mask, dtype=ttnn.int32)
        # Keep the original repeat reproducer available after the fix lands.
        source_row = ttnn.repeat(saved[source], ttnn.Shape([shape[0], 1])) if use_repeat else saved[source]
        ttnn.where(select, source_row, target, output_tensor=target)
        if use_repeat:
            ttnn.deallocate(source_row)
        ttnn.deallocate(select)
    for tensor in saved.values():
        ttnn.deallocate(tensor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--method", choices=("repeat", "broadcast", "current"), default="broadcast")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mesh = open_ornith_mesh()
    try:
        model = OrnithModel.__new__(OrnithModel)
        model.mesh_device = mesh
        model.vocab_size = load_text_config(args.model_path).vocab_size
        model.padded_vocab_size = 262144  # Same LM-head padding as OrnithModel.__init__.
        model.ccl = SamplingCCL(mesh)
        gen = OrnithGenerator.__new__(OrnithGenerator)
        gen.model, gen.mesh_device = model, mesh
        gen.sampling = model.build_sampler()
        penalties = gen.sampling.tt_penalties
        names = ["prompt_mask", "output_mask", "output_counts", "output_counts_gathered"]
        remap = [2, 0, 3, 1] + list(range(4, 32))
        report = {"method": args.method, "remap": remap, "buffers": []}
        for name in names:
            target = getattr(penalties, name)
            shape = list(target.shape)
            value = torch.arange(shape[0], dtype=torch.int32)[:, None] * 1_000_000
            value = value + torch.arange(shape[1], dtype=torch.int32)[None, :]
            host = model.upload(value, dtype=target.dtype, layout=target.layout, device=False)
            ttnn.copy_host_to_device_tensor(host, target)
            address = target.buffer_address()
            print(f"PROBE {name} shape={shape} dtype={target.dtype} layout={target.layout}", flush=True)
            if args.method in ("broadcast", "repeat"):
                broadcast_rows(gen, target, remap, use_repeat=args.method == "repeat")
            else:
                gen._remap_serving_rows(target, remap)
            expected = value[remap]
            for shard in ttnn.get_device_tensors(target):
                actual = ttnn.to_torch(shard)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, check_dtype=False)
            assert target.buffer_address() == address
            report["buffers"].append({"name": name, "shape": shape, "exact_all_replicas": True})
            print(f"PASS {name}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        close_ornith_mesh(mesh)


if __name__ == "__main__":
    main()
