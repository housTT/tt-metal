"""Compare a candidate probability file against the CPU reference, question by question.

Both inputs are JSONL in the ref_text_bf16.jsonl row format written by cpu_reference.py:
one row per record with "id", "answers" (SystemOne answer objects, which carry the question
type), "probs" (question id -> option id -> probability) and, when the record is labelled,
"_label" (question id -> gold label). A candidate row may come from the TT server or the
engine; it only needs "id", "probs" and, if the reference row has no label, "_label".
Rows are joined on "id"; questions on question id; options on option id. A row with "error"
on either side is counted and skipped. A question whose option ids differ between the two
files is counted as a mismatch and skipped.

Per question: dp = max over options of |p_ref - p_cand| (the L-infinity distance), the
reference argmax, the candidate argmax, and the reference top-2 margin. An argmax flip is
counted twice: once unconditionally, and once only when the reference margin is at least
--margin (default 0.05, the rule of /home/hous/dev/kev/reports/FINAL_REPORT.md section 3.2).
Flips below the margin are reported as near-tie flips. Where a label exists, both sides get
label accuracy (argmax equals the gold option; noul maps true/false to the "true"/"false"
options, score maps the level index to its string), the Brier score (sum over options of
(p - onehot)^2, mean over questions) and the expected calibration error (ECE, --bins
equal-width bins on the top probability, default 10), the definitions of kev.metrics.

Outputs: a JSON summary (--out-json) with the aggregate metrics, a per-type breakdown, the
flip details and the per-question rows, and a markdown table (--out-md). Exit code 1 when
--max-dp-bar is given and max dp exceeds it, or when --no-margin-flips is given and a flip
at or above the margin exists; else 0.

Usage:
  python parity_compare.py --reference REF.jsonl --candidate CAND.jsonl \
      --out-json summary.json --out-md summary.md [--margin 0.05] [--bins 10] \
      [--max-dp-bar 0.10] [--no-margin-flips] [--name "TT bf16 vs CPU bf16"]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clef_paths import read_jsonl

DEFAULT_MARGIN = 0.05
DEFAULT_BINS = 10


def label_option(question_type: str, label) -> str:
    if question_type == "noul":
        return "true" if bool(label) else "false"
    return str(label)


def argmax(distribution: dict) -> str:
    return max(distribution, key=distribution.__getitem__)


def top2_margin(distribution: dict) -> float:
    values = sorted(distribution.values(), reverse=True)
    return values[0] - values[1] if len(values) > 1 else values[0]


def brier(distribution: dict, gold: str) -> float:
    return sum((p - (1.0 if option == gold else 0.0)) ** 2 for option, p in distribution.items())


def ece(confidences: list[float], corrects: list[bool], bins: int) -> float:
    if not confidences:
        return float("nan")
    total = 0.0
    count = len(confidences)
    for bin_index in range(bins):
        lo = bin_index / bins
        hi = (bin_index + 1) / bins
        members = [
            (c, ok) for c, ok in zip(confidences, corrects) if (lo <= c < hi) or (bin_index == bins - 1 and c == hi)
        ]
        if not members:
            continue
        mean_conf = sum(c for c, _ in members) / len(members)
        mean_acc = sum(1.0 for _, ok in members if ok) / len(members)
        total += len(members) / count * abs(mean_acc - mean_conf)
    return total


def index_rows(rows: list[dict]) -> tuple[dict, int]:
    by_id = {}
    errors = 0
    for row in rows:
        if "error" in row or "probs" not in row:
            errors += 1
            continue
        by_id[row["id"]] = row
    return by_id, errors


def question_type(row: dict, question_id: str, distribution: dict) -> str:
    answer = (row.get("answers") or {}).get(question_id) or {}
    if answer.get("type") in ("noul", "choice", "score"):
        return answer["type"]
    if set(distribution) == {"true", "false"}:
        return "noul"
    if all(option.isdigit() for option in distribution):
        return "score"
    return "choice"


def compare_rows(reference: dict, candidate: dict, margin: float) -> tuple[list[dict], int]:
    questions = []
    mismatches = 0
    for record_id, ref_row in reference.items():
        cand_row = candidate.get(record_id)
        if cand_row is None:
            continue
        labels = ref_row.get("_label") or cand_row.get("_label") or {}
        for question_id, ref_dist in ref_row["probs"].items():
            cand_dist = (cand_row.get("probs") or {}).get(question_id)
            if cand_dist is None or set(cand_dist) != set(ref_dist):
                mismatches += 1
                continue
            qtype = question_type(ref_row, question_id, ref_dist)
            ref_top = argmax(ref_dist)
            cand_top = argmax(cand_dist)
            ref_margin = top2_margin(ref_dist)
            entry = {
                "id": record_id,
                "question": question_id,
                "type": qtype,
                "options": len(ref_dist),
                "dp": max(abs(ref_dist[o] - cand_dist[o]) for o in ref_dist),
                "ref_argmax": ref_top,
                "cand_argmax": cand_top,
                "ref_margin": ref_margin,
                "flip": ref_top != cand_top,
                "flip_at_margin": ref_top != cand_top and ref_margin >= margin,
                "ref_top_prob": ref_dist[ref_top],
                "cand_top_prob": cand_dist[cand_top],
            }
            if question_id in labels:
                gold = label_option(qtype, labels[question_id])
                entry.update(
                    {
                        "label": gold,
                        "ref_correct": ref_top == gold,
                        "cand_correct": cand_top == gold,
                        "ref_brier": brier(ref_dist, gold),
                        "cand_brier": brier(cand_dist, gold),
                    }
                )
            questions.append(entry)
    return questions, mismatches


def aggregate(questions: list[dict], bins: int) -> dict:
    out = {"questions": len(questions)}
    if not questions:
        return out
    dps = [q["dp"] for q in questions]
    out.update(
        {
            "max_dp": max(dps),
            "mean_dp": statistics.mean(dps),
            "median_dp": statistics.median(dps),
            "argmax_flips": sum(q["flip"] for q in questions),
            "flips_at_margin": sum(q["flip_at_margin"] for q in questions),
            "near_tie_flips": sum(q["flip"] and not q["flip_at_margin"] for q in questions),
        }
    )
    labelled = [q for q in questions if "label" in q]
    out["labelled_questions"] = len(labelled)
    if labelled:
        for side in ("ref", "cand"):
            corrects = [q[f"{side}_correct"] for q in labelled]
            out[f"{side}_accuracy"] = sum(corrects) / len(labelled)
            out[f"{side}_brier"] = statistics.mean(q[f"{side}_brier"] for q in labelled)
            out[f"{side}_ece"] = ece([q[f"{side}_top_prob"] for q in labelled], corrects, bins)
        out["accuracy_delta_pp"] = 100.0 * (out["cand_accuracy"] - out["ref_accuracy"])
        out["ece_shift"] = out["cand_ece"] - out["ref_ece"]
    return out


def summarize(reference_rows: list[dict], candidate_rows: list[dict], margin: float, bins: int, name: str) -> dict:
    reference, ref_errors = index_rows(reference_rows)
    candidate, cand_errors = index_rows(candidate_rows)
    questions, mismatches = compare_rows(reference, candidate, margin)
    joined = sorted(set(reference) & set(candidate))
    summary = {
        "name": name,
        "margin": margin,
        "bins": bins,
        "records": {
            "reference": len(reference_rows),
            "candidate": len(candidate_rows),
            "reference_errors": ref_errors,
            "candidate_errors": cand_errors,
            "joined": len(joined),
            "missing_in_candidate": sorted(set(reference) - set(candidate)),
            "extra_in_candidate": sorted(set(candidate) - set(reference)),
            "option_mismatches": mismatches,
        },
        "overall": aggregate(questions, bins),
        "by_type": {
            qtype: aggregate([q for q in questions if q["type"] == qtype], bins)
            for qtype in ("noul", "choice", "score")
            if any(q["type"] == qtype for q in questions)
        },
        "flips": [
            {
                k: q[k]
                for k in ("id", "question", "type", "ref_argmax", "cand_argmax", "ref_margin", "dp", "label")
                if k in q
            }
            for q in questions
            if q["flip"]
        ],
        "per_question": questions,
    }
    return summary


def fmt(value, digits=4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def markdown(summary: dict) -> str:
    records = summary["records"]
    overall = summary["overall"]
    lines = [
        f"## {summary['name']}",
        "",
        f"Records: reference {records['reference']}, candidate {records['candidate']}, joined {records['joined']}, "
        f"reference errors {records['reference_errors']}, candidate errors {records['candidate_errors']}, "
        f"missing in candidate {len(records['missing_in_candidate'])}, option mismatches {records['option_mismatches']}. "
        f"Flip margin {summary['margin']}, ECE bins {summary['bins']}.",
        "",
        "| Set | questions | max dp | mean dp | median dp | argmax flips | flips at margin | near-tie flips | labelled | acc ref | acc cand | Brier ref | Brier cand | ECE ref | ECE cand |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def row(label: str, agg: dict) -> str:
        cells = [
            label,
            fmt(agg.get("questions")),
            fmt(agg.get("max_dp")),
            fmt(agg.get("mean_dp")),
            fmt(agg.get("median_dp")),
            fmt(agg.get("argmax_flips")),
            fmt(agg.get("flips_at_margin")),
            fmt(agg.get("near_tie_flips")),
            fmt(agg.get("labelled_questions")),
            fmt(agg.get("ref_accuracy")),
            fmt(agg.get("cand_accuracy")),
            fmt(agg.get("ref_brier")),
            fmt(agg.get("cand_brier")),
            fmt(agg.get("ref_ece")),
            fmt(agg.get("cand_ece")),
        ]
        return "| " + " | ".join(cells) + " |"

    lines.append(row("all", overall))
    for qtype, agg in summary["by_type"].items():
        lines.append(row(qtype, agg))
    lines.append("")
    if summary["flips"]:
        lines += [
            "| record | question | type | ref argmax | cand argmax | ref margin | dp | label |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for flip in summary["flips"]:
            lines.append(
                f"| {flip['id']} | {flip['question']} | {flip['type']} | {flip['ref_argmax']} | {flip['cand_argmax']} | "
                f"{fmt(flip['ref_margin'])} | {fmt(flip['dp'])} | {flip.get('label', 'n/a')} |"
            )
    else:
        lines.append("No argmax flips.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument("--max-dp-bar", type=float, default=None)
    parser.add_argument("--no-margin-flips", action="store_true")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    name = args.name or f"{Path(args.candidate).name} vs {Path(args.reference).name}"
    summary = summarize(read_jsonl(args.reference), read_jsonl(args.candidate), args.margin, args.bins, name)
    text = markdown(summary)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(text)
    print(text)
    overall = summary["overall"]
    failed = False
    if args.max_dp_bar is not None and overall.get("max_dp", 0.0) > args.max_dp_bar:
        print(f"FAIL max dp {overall['max_dp']:.4f} > bar {args.max_dp_bar}")
        failed = True
    if args.no_margin_flips and overall.get("flips_at_margin", 0) > 0:
        print(f"FAIL {overall['flips_at_margin']} flips at margin >= {args.margin}")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
