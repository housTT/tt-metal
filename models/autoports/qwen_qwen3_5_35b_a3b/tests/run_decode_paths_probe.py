# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import time
from pathlib import Path

import torch


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def describe(ref, actual):
    top = actual.topk(2)
    choice = int(top.indices[0])
    hf_choice = int(ref.argmax())
    return dict(pcc=pcc(ref, actual), argmax=choice, hf_argmax=hf_choice, top1=choice == hf_choice,
                hf_rank_of_argmax=int((ref > ref[choice]).sum()), tt_margin=float(top.values[0] - top.values[1]),
                tt_gap_to_hf_argmax=float(actual[choice] - actual[hf_choice]),
                hf_gap_to_argmax=float(ref[hf_choice] - ref[choice]))


def eager(gen, ids, fed):
    import ttnn

    gen.model.zero_kv_cache()
    gen.model.reset_request_state()
    table = gen.default_page_table(1, len(ids) + len(fed) + 1)
    gen._write_prefill_table(table[0])
    logits = gen.model.prefill(torch.tensor([ids]), page_table=gen.tt_prefill_table, request_slots=(0,))
    out = [gen._logits_to_host(logits, 1)[0]]
    logits.deallocate(True)
    gen._write_page_table(1, table)
    tok = gen._device(torch.zeros(1, 1, dtype=torch.int32), ttnn.uint32)
    pos = gen._device(torch.zeros(1, dtype=torch.int32), ttnn.int32)
    for step, token in enumerate(fed):
        ttnn.copy_host_to_device_tensor(gen._host(torch.tensor([[token]], dtype=torch.int32), ttnn.uint32), tok)
        ttnn.copy_host_to_device_tensor(gen._host(torch.tensor([len(ids) + step], dtype=torch.int32), ttnn.int32),
                                        pos)
        logits = gen.model.decode(tok, pos, gen.tt_page_tables[1], request_slots=(0,))
        out.append(gen._logits_to_host(logits, 1)[0])
        logits.deallocate(True)
    return out


def traced(gen, ids, fed):
    gen.reset()
    table = gen.default_page_table(1, len(ids) + len(fed) + 1)
    out = [gen.prefill_forward(torch.tensor([ids]), page_table=table, prompt_lens=[len(ids)])[0, 0]]
    for step, token in enumerate(fed):
        out.append(gen.decode_forward(torch.tensor([[token]]), [len(ids) + step], page_table=table)[0])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=int, choices=[1, 2, 4], required=True)
    p.add_argument("--layers", default="0,3")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ref = torch.load(args.reference)
    ids, fed = ref["ids"][0].tolist(), ref["fed"]
    hf = [x.float() for x in ref["logits"]]
    import ttnn
    from transformers import AutoTokenizer

    from ..tt.generator import CHECKPOINT, DEFAULT_KV_BLOCKS, DEFAULT_REQUEST_SLOTS, Qwen35Generator
    from ..tt.model import Qwen35Model
    from .mesh_util import open_mesh

    mesh = open_mesh(args.mesh)
    result = dict(vars(args), output=str(args.output), reference=str(args.reference))
    phases = {}
    try:
        t0 = time.perf_counter()
        tp = mesh.get_num_devices()
        model = Qwen35Model.from_checkpoint(CHECKPOINT, mesh, layer_ids=tuple(int(x) for x in args.layers.split(",")),
                                            max_batch_size=DEFAULT_REQUEST_SLOTS[tp], kv_blocks=DEFAULT_KV_BLOCKS[tp])
        gen = Qwen35Generator(model, AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True))
        result["build_seconds"] = time.perf_counter() - t0
        phases["eager_before_prepare"] = eager(gen, ids, fed)
        gen.prepare()
        phases["traced_1"] = traced(gen, ids, fed)
        phases["eager_after_prepare"] = eager(gen, ids, fed)
        phases["traced_2"] = traced(gen, ids, fed)
        result["phases"] = {name: [describe(r, a) for r, a in zip(hf, out)] for name, out in phases.items()}
        names = list(phases)
        result["pairs"] = {
            f"{a}|{b}": [dict(pcc=pcc(x, y), max_abs=float((x - y).abs().max()), same_argmax=int(x.argmax()) == int(y.argmax()))
                         for x, y in zip(phases[a], phases[b])]
            for i, a in enumerate(names) for b in names[i + 1:]
        }
        for name, rows in result["phases"].items():
            print(name, [(r["argmax"], r["top1"], round(r["tt_margin"], 3)) for r in rows], flush=True)
        for name, rows in result["pairs"].items():
            print(name, [(round(r["max_abs"], 4), r["same_argmax"]) for r in rows], flush=True)
    finally:
        args.output.write_text(json.dumps(result, indent=1))
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
