"""Row-level look at the sampled positions of a hidden-state dump (test_engine CLEF_DUMP_HIDDEN).

For each sampled position: HF row norm, the max-abs dimension and its share of the norm, the
per-row PCC, and the PCC after excluding the top 8 dimensions by HF magnitude. Host only.

Usage: python hidden_row_outliers.py DUMP.pt [--positions 215,1285]
"""

import json
import sys

import torch

SNAPSHOT = (
    "/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c"
)


def pcc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


def main():
    from transformers import AutoTokenizer

    d = torch.load(sys.argv[1])
    positions = (
        [int(p) for p in sys.argv[sys.argv.index("--positions") + 1].split(",")]
        if "--positions" in sys.argv
        else d["positions"]
    )
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    ids = d["ids"][0].tolist()
    tt, ref = d["tt"].float(), d["ref"].float()
    norms = ref.norm(dim=1)
    rows = {}
    for p in positions:
        r, t = ref[p], tt[p]
        top = r.abs().topk(8).indices
        keep = torch.ones(r.shape[0], dtype=torch.bool)
        keep[top] = False
        rows[p] = dict(
            token=tok.decode([ids[p]]),
            context=tok.decode(ids[max(0, p - 6) : p + 1]),
            hf_norm=round(float(norms[p]), 2),
            hf_norm_median_all_rows=round(float(norms.median()), 2),
            max_abs_dim=int(r.abs().argmax()),
            max_abs_value=round(float(r.abs().max()), 2),
            max_abs_share_of_norm=round(float(r.abs().max() / norms[p]), 4),
            top8_share_of_norm=round(float(r[top].norm() / norms[p]), 4),
            pcc=round(pcc(r, t), 6),
            pcc_excluding_top8=round(pcc(r[keep], t[keep]), 6),
            tt_norm=round(float(t.norm()), 2),
            max_abs_err_dim=int((r - t).abs().argmax()),
            max_abs_err=round(float((r - t).abs().max()), 3),
            tt_value_at_hf_max_dim=round(float(t[r.abs().argmax()]), 2),
            hf_value_at_hf_max_dim=round(float(r[r.abs().argmax()]), 2),
        )
    print(json.dumps({"T": d["T"], "rows": rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
