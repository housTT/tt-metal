"""HF fp32 control for the per-row hidden-state gate (host only, no device).

Reads the hidden dumps written by tests/test_engine.py under CLEF_DUMP_HIDDEN (keys: T, ids,
tt, ref, positions; ref is the HF bf16 last_hidden_state of the same ids), runs the HF text model
in fp32 on the same ids, and reports the per-row PCC of TT vs fp32, bf16 vs fp32 and TT vs bf16
with the distribution (min, 1st percentile, mean, fraction below 0.99 / 0.95 / 0.80) and the
worst rows with their tokens. The fp32 hidden state is saved next to the report.

Usage:
  python hidden_fp32_control.py --dumps D1.pt[,D2.pt,...] [--out-prefix /path/stage1r_hidden_fp32_control]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

SNAPSHOT = (
    "/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c"
)


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), message, flush=True)


def row_pcc(a, b):
    a = a.float() - a.float().mean(dim=1, keepdim=True)
    b = b.float() - b.float().mean(dim=1, keepdim=True)
    return torch.nn.functional.cosine_similarity(a, b, dim=1)


def stats(p):
    q = torch.quantile(p, torch.tensor([0.01, 0.05, 0.5]))
    return dict(
        min=round(float(p.min()), 6),
        p01=round(float(q[0]), 6),
        p05=round(float(q[1]), 6),
        median=round(float(q[2]), 6),
        mean=round(float(p.mean()), 6),
        rows=int(p.numel()),
        rows_below_0_99=int((p < 0.99).sum()),
        rows_below_0_95=int((p < 0.95).sum()),
        rows_below_0_80=int((p < 0.80).sum()),
        frac_below_0_99=round(float((p < 0.99).float().mean()), 6),
        frac_below_0_95=round(float((p < 0.95).float().mean()), 6),
        frac_below_0_80=round(float((p < 0.80).float().mean()), 6),
        argmin=int(p.argmin()),
    )


def load_fp32(n_layers=64):
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    t0 = time.perf_counter()
    config = AutoConfig.from_pretrained(SNAPSHOT)
    config.text_config.num_hidden_layers = n_layers
    config.text_config.layer_types = config.text_config.layer_types[:n_layers]
    model = Qwen3_5ForConditionalGeneration.from_pretrained(SNAPSHOT, config=config, dtype=torch.float32).eval()
    log(f"fp32 model loaded in {time.perf_counter() - t0:.1f} s, {n_layers} layers")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dumps", required=True)
    parser.add_argument("--out-prefix", default="/home/hous/dev/clef/reports/stage1r_hidden_fp32_control")
    parser.add_argument("--n-layers", type=int, default=64)
    parser.add_argument("--worst", type=int, default=12)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    model = load_fp32(args.n_layers)
    for dump_path in args.dumps.split(","):
        d = torch.load(dump_path, weights_only=False)
        T, ids, tt, ref16, positions = d["T"], d["ids"], d["tt"].float(), d["ref"].float(), d["positions"]
        t0 = time.perf_counter()
        with torch.no_grad():
            ref32 = model.model.language_model(input_ids=ids, use_cache=False).last_hidden_state[0].float()
        t_fp32 = time.perf_counter() - t0
        log(f"T={T}: fp32 forward {t_fp32:.1f} s")
        torch.save({"T": T, "ids": ids, "ref_fp32": ref32}, f"{args.out_prefix}_T{T}_fp32.pt")
        p_tt32 = row_pcc(tt, ref32)
        p_1632 = row_pcc(ref16, ref32)
        p_tt16 = row_pcc(tt, ref16)
        toks = ids[0].tolist()

        def row(i):
            return dict(
                pos=int(i),
                token=tok.decode([toks[i]]),
                tt_vs_fp32=round(float(p_tt32[i]), 6),
                bf16_vs_fp32=round(float(p_1632[i]), 6),
                tt_vs_bf16=round(float(p_tt16[i]), 6),
                hf_fp32_norm=round(float(ref32[i].norm()), 2),
                hf_bf16_norm=round(float(ref16[i].norm()), 2),
                tt_norm=round(float(tt[i].norm()), 2),
            )

        worst_tt16 = [row(i) for i in p_tt16.argsort()[: args.worst].tolist()]
        worst_tt32 = [row(i) for i in p_tt32.argsort()[: args.worst].tolist()]
        worst_1632 = [row(i) for i in p_1632.argsort()[: args.worst].tolist()]
        low16 = (p_tt16 < 0.9).nonzero().flatten()
        low_rows = {
            "count": int(low16.numel()),
            "bf16_vs_fp32_at_these_rows": stats(p_1632[low16]) if low16.numel() else None,
            "tt_vs_fp32_at_these_rows": stats(p_tt32[low16]) if low16.numel() else None,
        }
        ok16 = (p_1632 >= 0.99).nonzero().flatten()
        report = dict(
            T=T,
            dump=dump_path,
            fp32_seconds=round(t_fp32, 1),
            n_layers=args.n_layers,
            tt_vs_fp32=stats(p_tt32),
            bf16_vs_fp32=stats(p_1632),
            tt_vs_bf16=stats(p_tt16),
            tt_vs_fp32_where_bf16_tracks_fp32=stats(p_tt32[ok16]) if ok16.numel() else None,
            sampled_positions={
                str(q): dict(
                    tt_vs_fp32=round(float(p_tt32[q]), 6),
                    bf16_vs_fp32=round(float(p_1632[q]), 6),
                    tt_vs_bf16=round(float(p_tt16[q]), 6),
                )
                for q in positions
            },
            rows_where_tt_vs_bf16_below_0_9=low_rows,
            worst_rows_tt_vs_bf16=worst_tt16,
            worst_rows_tt_vs_fp32=worst_tt32,
            worst_rows_bf16_vs_fp32=worst_1632,
            corr_of_row_pcc_tt32_vs_1632=round(float(torch.corrcoef(torch.stack([p_tt32, p_1632]))[0, 1]), 4),
        )
        out = Path(f"{args.out_prefix}_T{T}.json")
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        log(
            f"T={T}: tt_vs_fp32 {report['tt_vs_fp32']}; bf16_vs_fp32 {report['bf16_vs_fp32']}; "
            f"tt_vs_bf16 {report['tt_vs_bf16']}; low rows {low_rows}; worst tt_vs_fp32 {worst_tt32[:3]}"
        )
        log(f"FP32_CONTROL_DONE {out}")


if __name__ == "__main__":
    main()
