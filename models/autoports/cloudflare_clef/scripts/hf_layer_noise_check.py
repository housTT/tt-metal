"""Per-layer noise floor of the bf16 HF reference: run selected decoder layers in fp32 and in bf16
on the same bf16 hidden-state input and report the PCC of the layer delta (out - in) between the
two, per row, at the same positions the TT probe reports. Host only.

Usage: python hf_layer_noise_check.py --T 300 --layers 33,10,26,6,42,3,7 [--out REPORT.json]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import torch
from loguru import logger

SNAPSHOT = (
    "/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c"
)
os.environ.setdefault("CLEF_MODEL", SNAPSHOT)
os.environ.setdefault("HF_MODEL", SNAPSHOT)


def row_pcc(a, b):
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    return torch.nn.functional.cosine_similarity(a, b, dim=1)


def main():
    from models.autoports.cloudflare_clef.tests import test_engine as te
    from models.autoports.cloudflare_clef.tt import encode as clef_encode

    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--layers", default="33,10,26,6,42,3,7,62")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    out_path = Path(args.out or f"/home/hous/dev/clef/reports/stage1_hf_layer_noise_T{args.T}.json")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    tokenizer = clef_encode.load_tokenizer(SNAPSHOT)
    records = te.read_jsonl(te.RECORDS)
    ids = te.request_ids(tokenizer, records, args.T)
    positions = te.sample_positions(args.T, 8)
    hf = te.hf_model(64)
    text = hf.model.language_model
    t0 = time.perf_counter()
    with torch.no_grad():
        out = text(input_ids=ids, use_cache=False, output_hidden_states=True)
    hs = [h[0] for h in out.hidden_states]
    logger.info(f"HF bf16 forward {time.perf_counter() - t0:.1f} s")
    position_ids = torch.arange(args.T).view(1, 1, -1).expand(3, 1, -1).contiguous()
    report = {"T": args.T, "positions": positions, "layers": {}}
    for L in layers:
        layer_bf16 = text.layers[L]
        kind = text.config.layer_types[L]
        x_in = hs[L].unsqueeze(0)
        with torch.no_grad():
            pe16 = text.rotary_emb(x_in, position_ids)
            out16 = layer_bf16(
                hidden_states=x_in, position_embeddings=pe16, attention_mask=None, position_ids=position_ids
            )[0]
            layer_fp32 = copy.deepcopy(layer_bf16).float()
            x32 = x_in.float()
            rot32 = copy.deepcopy(text.rotary_emb).float()
            pe32 = rot32(x32, position_ids)
            out32 = layer_fp32(
                hidden_states=x32, position_embeddings=pe32, attention_mask=None, position_ids=position_ids
            )[0]
            del layer_fp32
        ref_next = hs[L + 1].float()
        x2 = x_in.float().reshape(args.T, -1)
        d16 = out16.float().reshape(args.T, -1) - x2
        d32 = out32.reshape(args.T, -1) - x32.reshape(args.T, -1)
        dnext = ref_next.reshape(args.T, -1) - x2
        p_16_32 = row_pcc(d16, d32)
        p_model_32 = row_pcc(dnext, d32)
        p_model_16 = row_pcc(dnext, d16)
        row = dict(
            kind=kind,
            bf16_vs_fp32_delta_min=round(float(p_16_32.min()), 6),
            bf16_vs_fp32_delta_mean=round(float(p_16_32.mean()), 6),
            bf16_vs_fp32_worst_pos=int(p_16_32.argmin()),
            bf16_vs_fp32_pos={q: round(float(p_16_32[q]), 6) for q in positions},
            model_next_vs_fp32_delta_min=round(float(p_model_32.min()), 6),
            model_next_vs_fp32_delta_mean=round(float(p_model_32.mean()), 6),
            model_next_vs_bf16_relayer_delta_min=round(float(p_model_16.min()), 6),
            model_next_vs_bf16_relayer_delta_mean=round(float(p_model_16.mean()), 6),
        )
        report["layers"][L] = row
        logger.info(f"layer {L} {kind}: {row}")
        out_path.write_text(json.dumps(report, indent=2))
    logger.info(f"HF_LAYER_NOISE_DONE {out_path}")


if __name__ == "__main__":
    main()
