"""Worst rows of a hidden-state dump and their overlap with the worst rows of another dump (host only).

Both dumps are test_engine CLEF_DUMP_HIDDEN files for the same T (same ids, checked). For dump A
(for example the 4-layer run) it lists the rows below --bar with position, token, context and
per-row PCC, and for each of them the PCC of the same row in dump B (for example the 64-layer run).
It also reports the overlap between the bottom --k rows of A and of B and the rank correlation of
the two per-row PCC series.

Usage:
  python hidden_row_overlap.py --a A.pt --b B.pt [--bar 0.99] [--k 50] [--out OUT.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

SNAPSHOT = (
    "/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c"
)


def row_pcc(a, b):
    a = a.float() - a.float().mean(dim=1, keepdim=True)
    b = b.float() - b.float().mean(dim=1, keepdim=True)
    return torch.nn.functional.cosine_similarity(a, b, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--bar", type=float, default=0.99)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    A = torch.load(args.a, weights_only=False)
    B = torch.load(args.b, weights_only=False)
    assert A["T"] == B["T"] and torch.equal(A["ids"], B["ids"]), "dumps differ in ids"
    ids = A["ids"][0].tolist()
    pa = row_pcc(A["tt"], A["ref"])
    pb = row_pcc(B["tt"], B["ref"])
    low = (pa < args.bar).nonzero().flatten().tolist()
    rows = [
        dict(
            pos=p,
            token=tok.decode([ids[p]]),
            context=tok.decode(ids[max(0, p - 8) : p + 1]),
            pcc_a=round(float(pa[p]), 6),
            pcc_b=round(float(pb[p]), 6),
            offset_mod_128=p % 128,
            offset_mod_1024=p % 1024,
            ref_norm_a=round(float(A["ref"][p].norm()), 2),
            tt_norm_a=round(float(A["tt"][p].norm()), 2),
        )
        for p in sorted(low, key=lambda p: float(pa[p]))
    ]
    ka = set(pa.argsort()[: args.k].tolist())
    kb = set(pb.argsort()[: args.k].tolist())
    ra = pa.argsort().argsort().float()
    rb = pb.argsort().argsort().float()
    spearman = float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])
    report = dict(
        T=A["T"],
        a=args.a,
        b=args.b,
        bar=args.bar,
        a_min=round(float(pa.min()), 6),
        a_mean=round(float(pa.mean()), 6),
        a_rows_below_bar=len(low),
        b_min=round(float(pb.min()), 6),
        b_mean=round(float(pb.mean()), 6),
        b_rows_below_bar=int((pb < args.bar).sum()),
        rows_below_bar_in_a=rows,
        b_pcc_at_a_low_rows=dict(
            min=round(float(pb[low].min()), 6) if low else None,
            mean=round(float(pb[low].mean()), 6) if low else None,
            below_bar=int((pb[low] < args.bar).sum()) if low else None,
        ),
        bottom_k=args.k,
        bottom_k_overlap=len(ka & kb),
        bottom_k_overlap_positions=sorted(ka & kb),
        spearman_rank_corr=round(spearman, 4),
        pearson_corr=round(float(torch.corrcoef(torch.stack([pa, pb]))[0, 1]), 4),
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
