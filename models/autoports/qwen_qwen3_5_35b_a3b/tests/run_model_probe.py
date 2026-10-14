# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import copy
import json
import time
from pathlib import Path

import torch

CHECKPOINT = Path("/home/hous/dev/qwen3.5-35b-a3b/weights/Qwen3.5-35B-A3B")
PROMPT = ("The history of the printing press begins in the fifteenth century, when movable type changed how "
          "books were made and read across Europe. ")


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def topk_overlap(ref, actual, k):
    return len(set(ref.topk(k).indices.tolist()) & set(actual.topk(k).indices.tolist())) / k


def hf_model(cfg, layer_ids):
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    from .layer_weights import load_layer_state_dict
    from ..tt.model import _load_keys

    local = copy.deepcopy(cfg)
    local.num_hidden_layers = len(layer_ids)
    local.layer_types = [cfg.layer_types[i] for i in layer_ids]
    local._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen3_5MoeForCausalLM(local)
    state = {}
    for new, old in enumerate(layer_ids):
        prefix = f"model.language_model.layers.{old}."
        for key, value in load_layer_state_dict(CHECKPOINT, old).items():
            state[f"model.layers.{new}." + key[len(prefix):]] = value
    top = _load_keys(CHECKPOINT, ("model.language_model.embed_tokens.weight", "model.language_model.norm.weight",
                                  "lm_head.weight"))
    state["model.embed_tokens.weight"] = top["model.language_model.embed_tokens.weight"]
    state["model.norm.weight"] = top["model.language_model.norm.weight"]
    state["lm_head.weight"] = top["lm_head.weight"]
    model.load_state_dict(state, assign=True, strict=True)
    model.model.rotary_emb = type(model.model.rotary_emb)(config=local)
    return model.to(torch.bfloat16).eval()


def hf_reference(cfg, layer_ids, ids, steps):
    from transformers.cache_utils import DynamicCache

    hf = hf_model(cfg, layer_ids)
    with torch.no_grad():
        cache = DynamicCache(config=hf.config)
        out = hf(input_ids=ids, past_key_values=cache, use_cache=True)
        ref_logits = [out.logits[0, -1].float()]
        fed = []
        token = int(ref_logits[-1].argmax())
        for _ in range(steps):
            fed.append(token)
            out = hf(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True)
            ref_logits.append(out.logits[0, -1].float())
            token = int(ref_logits[-1].argmax())
    return ref_logits, fed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=int, choices=[1, 2, 4], required=True)
    p.add_argument("--layers", default="0,3")
    p.add_argument("--prompt-len", type=int, default=300)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from transformers import AutoConfig, AutoTokenizer

    layer_ids = tuple(int(x) for x in args.layers.split(","))
    cfg = AutoConfig.from_pretrained(CHECKPOINT, local_files_only=True).text_config
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    ids = tokenizer(PROMPT * 40, return_tensors="pt").input_ids[:, : args.prompt_len]
    ref_path = args.output.with_name(f"hf_ref_{args.layers.replace(',', '_')}_{args.prompt_len}_{args.steps}.pt")
    if ref_path.exists():
        cached = torch.load(ref_path)
        ref_logits, fed = cached["logits"], cached["fed"]
    else:
        ref_logits, fed = hf_reference(cfg, layer_ids, ids, args.steps)
        torch.save(dict(logits=ref_logits, fed=fed, ids=ids), ref_path)
    print("HF_READY", flush=True)

    import ttnn

    from ..tt.model import Qwen35Model
    from .mesh_util import open_mesh

    mesh = open_mesh(args.mesh)
    result = dict(vars(args), output=str(args.output), layer_ids=layer_ids)
    try:
        t0 = time.perf_counter()
        model = Qwen35Model.from_checkpoint(CHECKPOINT, mesh, layer_ids=layer_ids)
        result["load_seconds"] = time.perf_counter() - t0
        blocks = -(-(args.prompt_len + args.steps + 1) // 32)
        model.allocate_kv_cache(blocks + 8)
        replicate = ttnn.ReplicateTensorToMesh(mesh)
        composer = ttnn.ConcatMeshToTensor(mesh, dim=-1)
        table = ttnn.from_torch(torch.arange(blocks, dtype=torch.int32).reshape(1, blocks), dtype=ttnn.int32,
                                layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=replicate)
        slots = (0,)
        model.reset_request_state(slots)
        logits = model.prefill(ids, page_table=table, request_slots=slots)
        actual = [ttnn.to_torch(logits, mesh_composer=composer)[0, -1, : cfg.vocab_size].float()]
        tok = ttnn.from_torch(torch.zeros(1, 1, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                              device=mesh, mesh_mapper=replicate)
        pos = ttnn.from_torch(torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                              device=mesh, mesh_mapper=replicate)
        for step, token in enumerate(fed):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.tensor([[token]], dtype=torch.int32),
                                                            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                                            mesh_mapper=replicate), tok)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.tensor([args.prompt_len + step], dtype=torch.int32),
                                                            dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                                            mesh_mapper=replicate), pos)
            logits = model.decode(tok, pos, table, request_slots=slots)
            actual.append(ttnn.to_torch(logits, mesh_composer=composer)[0, -1, : cfg.vocab_size].float())
        rows = []
        for i, (r, a) in enumerate(zip(ref_logits, actual)):
            rows.append(dict(step=i, pcc=pcc(r, a), top1=int(r.argmax()) == int(a.argmax()),
                             top5=topk_overlap(r, a, 5), top100=topk_overlap(r, a, 100)))
            print(json.dumps(rows[-1]), flush=True)
        result["rows"] = rows
        result["pcc_min"] = min(r["pcc"] for r in rows)
        result["top1_rate"] = sum(r["top1"] for r in rows) / len(rows)
    finally:
        args.output.write_text(json.dumps(result, indent=1))
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
