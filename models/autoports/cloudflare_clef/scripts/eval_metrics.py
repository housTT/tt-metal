"""Score a results JSONL against a labelled eval JSONL: accuracy, and macro-F1 for BANKING77.

The eval file is one of /home/hous/dev/clef/evals/*.jsonl (rows with "id", "questions" and
"_label": question id -> gold option id). The results file has one row per record with "id"
and "answers" (SystemOne answer objects, as cpu_reference.py or the TT server return them);
a row may carry "probs" instead, and a row with "error" is counted and skipped. Rows are
joined on "id". The predicted option of a choice question is answers[qid]["choice"]; of a
noul question "true" when answers[qid]["noul"] >= 0.5; of a score question the argmax of
answers[qid]["probabilities"]. When "answers" is absent the argmax of probs[qid] is used.

Metrics: accuracy over the joined labelled questions always; macro-F1 (scikit-learn
f1_score(average="macro") over the union of gold and predicted labels, zero_division=0)
when --metric is macro_f1, or when --metric is auto and the eval "_source" names banking77.
Exits 1 when any eval row has no result (coverage must be complete) unless --allow-missing.

Usage:
  python eval_metrics.py --results RESULTS.jsonl --eval EVAL.jsonl [--metric auto|accuracy|macro_f1]
                         [--question QID] [--out summary.json] [--allow-missing]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clef_paths import read_jsonl


def predicted_option(row: dict, question_id: str, question_type: str) -> str | None:
    answer = (row.get("answers") or {}).get(question_id)
    if answer is not None:
        if question_type == "noul" or answer.get("type") == "noul":
            return "true" if float(answer["noul"]) >= 0.5 else "false"
        if "choice" in answer:
            return str(answer["choice"])
        probabilities = answer.get("probabilities") or {}
        return max(probabilities, key=probabilities.__getitem__) if probabilities else None
    distribution = (row.get("probs") or {}).get(question_id)
    if distribution:
        return max(distribution, key=distribution.__getitem__)
    return None


def gold_option(question_type: str, label) -> str:
    if question_type == "noul":
        return "true" if bool(label) else "false"
    return str(label)


def score(eval_rows: list[dict], result_rows: list[dict], metric: str, question_filter: str | None) -> dict:
    results = {}
    errors = 0
    for row in result_rows:
        if "error" in row:
            errors += 1
            continue
        results[row["id"]] = row
    sources = collections.Counter(str(row.get("_source", "")) for row in eval_rows)
    if metric == "auto":
        metric = "macro_f1" if any("banking77" in source.lower() for source in sources) else "accuracy"
    gold, pred, missing, unanswered = [], [], [], []
    per_question = []
    for row in eval_rows:
        result = results.get(row["id"])
        if result is None:
            missing.append(row["id"])
            continue
        for question_id, label in (row.get("_label") or {}).items():
            if question_filter and question_id != question_filter:
                continue
            question_type = row["questions"][question_id]["type"]
            truth = gold_option(question_type, label)
            guess = predicted_option(result, question_id, question_type)
            if guess is None:
                unanswered.append((row["id"], question_id))
                continue
            gold.append(truth)
            pred.append(guess)
            per_question.append(
                {
                    "id": row["id"],
                    "question": question_id,
                    "label": truth,
                    "predicted": guess,
                    "correct": truth == guess,
                }
            )
    summary = {
        "metric": metric,
        "eval_rows": len(eval_rows),
        "result_rows": len(result_rows),
        "result_errors": errors,
        "missing_results": missing,
        "unanswered_questions": [list(pair) for pair in unanswered],
        "scored_questions": len(gold),
        "correct": sum(1 for g, p in zip(gold, pred) if g == p),
        "accuracy": (sum(1 for g, p in zip(gold, pred) if g == p) / len(gold)) if gold else None,
        "sources": dict(sources),
    }
    if metric == "macro_f1" and gold:
        from sklearn.metrics import f1_score

        labels = sorted(set(gold) | set(pred))
        summary["macro_f1"] = float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0))
        summary["macro_f1_labels"] = len(labels)
        summary["gold_labels"] = len(set(gold))
    summary["per_question"] = per_question
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--metric", choices=("auto", "accuracy", "macro_f1"), default="auto")
    parser.add_argument("--question", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    summary = score(read_jsonl(args.eval), read_jsonl(args.results), args.metric, args.question)
    summary["results_path"] = str(Path(args.results).resolve())
    summary["eval_path"] = str(Path(args.eval).resolve())
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    headline = (
        f"{Path(args.eval).name}: {summary['correct']}/{summary['scored_questions']} correct, accuracy {summary['accuracy']:.4f}"
        if summary["accuracy"] is not None
        else f"{Path(args.eval).name}: nothing scored"
    )
    if "macro_f1" in summary:
        headline += f", macro-F1 {summary['macro_f1']:.4f} over {summary['macro_f1_labels']} labels"
    headline += f"; missing {len(summary['missing_results'])}, errors {summary['result_errors']}, unanswered {len(summary['unanswered_questions'])}"
    print(headline)
    incomplete = summary["missing_results"] or summary["unanswered_questions"] or summary["result_errors"]
    return 1 if incomplete and not args.allow_missing else 0


if __name__ == "__main__":
    sys.exit(main())
