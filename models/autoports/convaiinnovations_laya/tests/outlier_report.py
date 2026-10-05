# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import torch

from models.autoports.convaiinnovations_laya import common as C
from models.autoports.convaiinnovations_laya.reference import laya_reference as R
from models.autoports.convaiinnovations_laya.tests.pcc_utils import outlier_report

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(os.path.dirname(HERE), "doc", "probe", "outliers.json")


def pick_sequences(tok, n: int) -> list:
    items = C.gate_items(tok)
    seen = set()
    chosen = []
    for it in sorted(items, key=lambda x: -len(x["ids"])):
        if it["case_index"] in seen:
            continue
        seen.add(it["case_index"])
        chosen.append(it)
        if len(chosen) == n:
            break
    return chosen


def tensor_stats(x: torch.Tensor, real: torch.Tensor) -> dict:
    v = x.detach().float()[real]
    per_ch = v.abs().max(dim=0).values
    top = torch.topk(per_ch, 5)
    l2 = v.norm(dim=-1)
    return {
        "max_abs": float(per_ch.max()),
        "median_channel_max": float(per_ch.median()),
        "top_channels": top.indices.tolist(),
        "top_values": [round(float(t), 3) for t in top.values],
        "mean_token_l2": float(l2.mean()),
        "max_token_l2": float(l2.max()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    ref = R.LayaReference()
    items = pick_sequences(ref.tok, a.n)
    b = ref.collate(items, seq_len=a.seq_len)
    real = b["attention_mask"].bool()
    valid = b["marker_mask"].bool()
    args = (b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
    norm_stats = {}
    handles = []

    def mask_for(x):
        if tuple(x.shape[:2]) == tuple(real.shape):
            return real
        if tuple(x.shape[:2]) == tuple(valid.shape):
            return valid
        return torch.ones(x.shape[:-1], dtype=torch.bool)

    def make_hook(name):
        def hook(mod, inputs, output):
            norm_stats[name] = {"input": tensor_stats(inputs[0], mask_for(inputs[0])), "output": tensor_stats(output, mask_for(output))}

        return hook

    for name, mod in ref.model.named_modules():
        if isinstance(mod, torch.nn.LayerNorm):
            handles.append(mod.register_forward_hook(make_hook(name)))
    t0 = time.perf_counter()
    logits_hooked, act_hooked = ref.forward(*args)
    t_hooked = time.perf_counter() - t0
    for h in handles:
        h.remove()
    t0 = time.perf_counter()
    logits, act, hidden = ref.forward_with_hidden(*args)
    t_hidden = time.perf_counter() - t0
    enc_stats = []
    for i, h in enumerate(hidden["encoder"]):
        label = "embeddings" if i == 0 else ("final_norm" if i == len(hidden["encoder"]) - 1 else f"layer_{i - 1}_out")
        s = tensor_stats(h, real)
        s["pcc_utils_outlier_report"] = outlier_report(h[real].unsqueeze(0))
        enc_stats.append({"index": i, "label": label, **s})
    head_stats = [{"label": "after_type_emb", **tensor_stats(hidden["after_type_emb"], real)}] + [{"label": f"head_layer_{j}_out", **tensor_stats(h, real)} for j, h in enumerate(hidden["head"])]
    table = []
    nl = ref.encoder_config.num_hidden_layers
    for i in range(nl):
        attn = norm_stats.get(f"encoder.layers.{i}.attn_norm")
        mlp = norm_stats.get(f"encoder.layers.{i}.mlp_norm")
        table.append(
            {
                "layer": i,
                "type": ref.encoder_config.layer_types[i],
                "pre_attn_norm_max_abs": None if attn is None else attn["input"]["max_abs"],
                "post_attn_norm_max_abs": None if attn is None else attn["output"]["max_abs"],
                "pre_mlp_norm_max_abs": mlp["input"]["max_abs"],
                "post_mlp_norm_max_abs": mlp["output"]["max_abs"],
                "residual_out_max_abs": enc_stats[i + 1]["max_abs"],
                "residual_out_mean_token_l2": enc_stats[i + 1]["mean_token_l2"],
                "residual_out_top_channel": enc_stats[i + 1]["top_channels"][0],
            }
        )
    out = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_dir": ref.model_dir,
        "revision": C.LAYA_REVISION,
        "dtype": str(ref.dtype),
        "attn_implementation": ref.attn_implementation,
        "seq_len": a.seq_len,
        "sequences": [{"case_id": it["case_id"], "case_index": it["case_index"], "workflow": it["workflow"], "qid": it["qid"], "type": R.QTYPE_NAMES[int(it["qtype"])], "k": it["k"], "len": len(it["ids"])} for it in items],
        "stats_over": "real token positions only (attention_mask == 1), all sequences pooled",
        "hooked_vs_plain_forward": {
            "max_abs_logit_delta": float((logits_hooked - logits)[b["marker_mask"]].abs().max()),
            "max_abs_act_logit_delta": float((act_hooked - act).abs().max()),
            "note": "forward hooks on the head LayerNorms switch nn.TransformerEncoderLayer from its fused fast path to the explicit module path",
        },
        "forward_seconds": {"hooked": round(t_hooked, 2), "with_hidden": round(t_hidden, 2)},
        "layer_table": table,
        "norms": norm_stats,
        "encoder_hidden_states": enc_stats,
        "head_hidden_states": head_stats,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}")
    print("| layer | type | pre attn_norm max abs | post attn_norm | pre mlp_norm | post mlp_norm | residual out max abs | residual mean token L2 | top channel |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in table:
        pa = "n/a (Identity)" if r["pre_attn_norm_max_abs"] is None else f"{r['pre_attn_norm_max_abs']:.1f}"
        po = "n/a" if r["post_attn_norm_max_abs"] is None else f"{r['post_attn_norm_max_abs']:.2f}"
        print(f"| {r['layer']} | {r['type'].split('_')[0]} | {pa} | {po} | {r['pre_mlp_norm_max_abs']:.1f} | {r['post_mlp_norm_max_abs']:.2f} | {r['residual_out_max_abs']:.1f} | {r['residual_out_mean_token_l2']:.1f} | {r['residual_out_top_channel']} |")
    for name in ("encoder.embeddings.norm", "encoder.final_norm", "head.layers.0.norm1", "head.layers.0.norm2", "head.layers.1.norm1", "head.layers.1.norm2", "scorer.0"):
        s = norm_stats[name]
        print(f"| {name} | | {s['input']['max_abs']:.1f} | {s['output']['max_abs']:.2f} | | | | {s['input']['mean_token_l2']:.1f} | {s['input']['top_channels'][0]} |")


if __name__ == "__main__":
    main()
