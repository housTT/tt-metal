# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path

import torch

CASES = ((128, 300), (4224, 77), (8192, 1), (8320, 4097), (12288, 129))


def pcc(a, b):
    return float(torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1])


def compare(ref, actual):
    choice = int(actual.argmax())
    return dict(pcc=pcc(ref, actual), argmax=choice, hf_argmax=int(ref.argmax()),
                hf_rank_of_argmax=int((ref > ref[choice]).sum()),
                top5_overlap=len(set(ref.topk(5).indices.tolist()) & set(actual.topk(5).indices.tolist())))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=int, choices=[1, 2, 4], required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ref = torch.load(args.reference)
    book = ref["ids"]
    import ttnn

    from ..tt.generator import build_generator
    from .mesh_util import open_mesh

    mesh = open_mesh(args.mesh)
    result = dict(mesh=args.mesh, reference=str(args.reference), cases=[])
    tensors = {}
    try:
        gen = build_generator(Path(__file__).parents[1], mesh)
        for start, length in CASES:
            total = start + length
            hf = ref["logits"][total]
            table = gen.default_page_table(1, total + 1)
            gen.reset()
            whole = gen.prefill_forward(torch.tensor([book[:total]]), page_table=table, prompt_lens=[total])[0, 0]
            gen.reset()
            gen.prefill_forward(torch.tensor([book[:start]]), page_table=table, prompt_lens=[start])
            split = gen.prefill_forward(torch.tensor([book[start:total]]), page_table=table, prompt_lens=[length],
                                        start_pos=[start])[0, 0]
            gen.reset()
            fed = gen.prefill_forward(torch.tensor([book[:start]]), page_table=table, prompt_lens=[start])[0, 0]
            for i in range(length):
                fed = gen.decode_forward(torch.tensor([[book[start + i]]]), [start + i], page_table=table)[0]
            tensors[total, start] = dict(whole=whole, split=split, decode_fed=fed)
            row = dict(start=start, length=length, total=total,
                       whole_vs_hf=compare(hf, whole), split_vs_hf=compare(hf, split),
                       decode_fed_vs_hf=compare(hf, fed), whole_vs_split_pcc=pcc(whole, split),
                       whole_vs_decode_fed_pcc=pcc(whole, fed), split_vs_decode_fed_pcc=pcc(split, fed))
            result["cases"].append(row)
            print(json.dumps(row), flush=True)
    finally:
        args.output.write_text(json.dumps(result, indent=1))
        torch.save(tensors, args.output.with_suffix(".pt"))
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
