"""Per-record view of a parity_compare.py JSON summary with the candidate's latency (host only).

Joins the per-question rows of a parity_compare.py JSON on record id and prints one markdown row
per record: input tokens, questions, max dp, argmax flips, flips at margin, candidate seconds and
device seconds, cache mode and hit flag, and the reference seconds when the reference rows carry
them.

Usage:
  python per_record_parity.py --parity SUMMARY.json --candidate CAND.jsonl [--reference REF.jsonl] [--out OUT.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clef_paths import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    summary = json.loads(Path(args.parity).read_text())
    cand = {r["id"]: r for r in read_jsonl(args.candidate)}
    ref = {r["id"]: r for r in read_jsonl(args.reference)} if args.reference else {}
    per = {}
    for q in summary["per_question"]:
        row = per.setdefault(q["id"], dict(questions=0, max_dp=0.0, flips=0, flips_at_margin=0))
        row["questions"] += 1
        row["max_dp"] = max(row["max_dp"], q["dp"])
        row["flips"] += int(q["flip"])
        row["flips_at_margin"] += int(q["flip_at_margin"])
    lines = [
        "| record | input tokens | questions | max dp | argmax flips | flips at margin | TT seconds | TT device s | mode | cache hit | CPU bf16 s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rid, row in per.items():
        c = cand.get(rid, {})
        r = ref.get(rid, {})
        timing = c.get("timing", {})
        lines.append(
            f"| {rid} | {c.get('input_tokens', '')} | {row['questions']} | {row['max_dp']:.4f} | {row['flips']} | "
            f"{row['flips_at_margin']} | {c.get('seconds', '')} | {timing.get('device_s', '')} | {c.get('mode', '')} | "
            f"{c.get('cache_hit', '')} | {r.get('seconds', '')} |"
        )
    o = summary["overall"]
    lines.append("")
    lines.append(
        f"Overall: {o['questions']} questions, max dp {o['max_dp']:.4f}, mean dp {o['mean_dp']:.4f}, median dp "
        f"{o['median_dp']:.4f}, argmax flips {o['argmax_flips']}, flips at margin {o['flips_at_margin']}, "
        f"near-tie flips {o['near_tie_flips']}."
    )
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
