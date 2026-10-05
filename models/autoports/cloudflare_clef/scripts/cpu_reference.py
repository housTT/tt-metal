"""Run the Cloudflare/clef release on CPU as the parity reference.

Loads joint_schema_model.load_release_model(snapshot, device="cpu", dtype=torch.bfloat16)
from the Hugging Face snapshot and answers every record of a JSONL file the way
systemone() does. Each output row carries the input id, the systemone response
("model", "answers", "usage"), a "probs" mapping question id -> option id -> raw softmax
probability (not rounded), the token count, wall seconds, and, when the input row has
"_label", a "correct" mapping question id -> bool. A record that raises writes an
"error" row and the run continues. The summary line at the end gives median and max
seconds per record and the peak resident set size in MiB from resource.getrusage.

Image records list absolute PNG paths under "images"; they are opened with PIL and
converted to RGB before the request is passed to the model.

Usage:
  python cpu_reference.py --input records.jsonl --output ref.jsonl [--snapshot DIR]
                          [--threads 8] [--limit N] [--readme-examples OUT.json]
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clef_paths import add_snapshot_to_path, read_jsonl, snapshot_dir

REQUEST_KEYS = ("id", "model", "state", "questions", "images", "videos", "media_kwargs")


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), message, flush=True)


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def build_request(row: dict) -> dict:
    from PIL import Image

    request = {key: row[key] for key in REQUEST_KEYS if key in row}
    request.setdefault("model", "clef")
    if request.get("images"):
        request["images"] = [Image.open(path).convert("RGB") for path in request["images"]]
    return request


def answer_with_probs(model, processor, request: dict, jsm) -> tuple[dict, dict]:
    import torch

    questions = request["questions"]
    for question_id, question in questions.items():
        if question.get("type") not in jsm.QUESTION_TYPES:
            raise ValueError(f"{question_id}: type must be noul, choice, or score")
        if question["type"] != "noul" and not question.get("criteria"):
            raise ValueError(f"{question_id}: criteria must not be empty")
    encoded = jsm.encode_record(processor.tokenizer, request, max_length=16384, processor=processor)
    device = next(model.parameters()).device
    with torch.inference_mode():
        logits = model(jsm.collate_records([encoded], processor.tokenizer.pad_token_id, device))[0]
    probs = {}
    answers = {}
    for question, question_logits in zip(encoded.questions, logits):
        distribution = dict(zip(question.option_ids, question_logits.float().softmax(-1).tolist()))
        probs[question.question_id] = distribution
        answers[question.question_id] = jsm.systemone_answer(questions[question.question_id], distribution)
    response = {
        "model": request["model"],
        "answers": answers,
        "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0},
    }
    return response, probs


def correctness(questions: dict, probs: dict, labels: dict) -> dict:
    out = {}
    for question_id, label in labels.items():
        distribution = probs.get(question_id)
        if distribution is None:
            continue
        question_type = questions[question_id]["type"]
        if question_type == "noul":
            out[question_id] = (distribution["true"] >= 0.5) == bool(label)
        else:
            predicted = max(distribution, key=distribution.__getitem__)
            out[question_id] = predicted == str(label)
    return out


def run_records(model, processor, jsm, rows: list[dict], output: Path) -> list[float]:
    seconds = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            record_id = row.get("id", f"row{index}")
            result = {"id": record_id}
            if "_label" in row:
                result["_label"] = row["_label"]
            started = time.perf_counter()
            try:
                request = build_request(row)
                response, probs = answer_with_probs(model, processor, request, jsm)
                elapsed = time.perf_counter() - started
                result.update(response)
                result["probs"] = probs
                result["input_tokens"] = response["usage"]["input_tokens"]
                if "_label" in row:
                    result["correct"] = correctness(row["questions"], probs, row["_label"])
                seconds.append(elapsed)
                log(
                    f"{index + 1}/{len(rows)} {record_id} tokens={result['input_tokens']} {elapsed:.1f}s "
                    f"correct={result.get('correct', 'n/a')}"
                )
            except Exception as error:
                elapsed = time.perf_counter() - started
                result["error"] = f"{type(error).__name__}: {error}"
                log(f"{index + 1}/{len(rows)} {record_id} FAILED after {elapsed:.1f}s: {result['error']}")
            result["seconds"] = round(elapsed, 3)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
    return seconds


def run_readme_examples(model, processor, jsm, output: Path) -> None:
    import torch

    record = {
        "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
        "questions": {
            "status": {
                "type": "choice",
                "instructions": "What is the invoice status?",
                "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."},
            },
            "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"},
        },
    }
    started = time.perf_counter()
    encoded = jsm.encode_record(processor.tokenizer, record, processor=processor)
    batch = jsm.collate_records([encoded], processor.tokenizer.pad_token_id, torch.device("cpu"))
    with torch.inference_mode():
        logits = model(batch)[0]
    usage_example = {}
    for question, question_logits in zip(encoded.questions, logits):
        probabilities = question_logits.float().softmax(-1).tolist()
        printed = dict(zip(question.option_ids, probabilities))
        print(question.question_id, printed, flush=True)
        usage_example[question.question_id] = printed
    usage_seconds = time.perf_counter() - started

    request = {
        "model": "clef",
        "state": "Our checkout started returning errors and orders are blocked.",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle the message?",
                "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
            },
            "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
            "outage": {"type": "noul", "instructions": "Is a service down?"},
        },
    }
    started = time.perf_counter()
    response = jsm.systemone(model, processor, request)
    systemone_seconds = time.perf_counter() - started
    print(response["answers"], flush=True)
    _, systemone_probs = answer_with_probs(model, processor, request, jsm)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "usage_example": {
                    "record": record,
                    "input_tokens": len(encoded.input_ids),
                    "option_ids": {q.question_id: list(q.option_ids) for q in encoded.questions},
                    "printed_probabilities": usage_example,
                    "seconds": round(usage_seconds, 3),
                },
                "systemone_example": {
                    "request": request,
                    "response": response,
                    "raw_probs": systemone_probs,
                    "seconds": round(systemone_seconds, 3),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    log(f"readme examples -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--output", action="append", default=[])
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--readme-examples", default=None)
    args = parser.parse_args()
    if len(args.input) != len(args.output):
        raise SystemExit("--input and --output must be given the same number of times")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    import torch

    torch.set_num_threads(args.threads)
    snapshot = snapshot_dir(args.snapshot)
    add_snapshot_to_path(snapshot)
    import joint_schema_model as jsm

    log(f"snapshot {snapshot} torch {torch.__version__} threads {torch.get_num_threads()}")
    started = time.perf_counter()
    model, processor = jsm.load_release_model(snapshot, device="cpu", dtype=torch.bfloat16)
    log(f"model loaded in {time.perf_counter() - started:.1f}s, rss {peak_rss_mib():.0f} MiB")
    summary = {
        "snapshot": str(snapshot),
        "threads": args.threads,
        "load_seconds": round(time.perf_counter() - started, 1),
        "runs": {},
    }
    if args.readme_examples:
        run_readme_examples(model, processor, jsm, Path(args.readme_examples))
    for input_path, output_path in zip(args.input, args.output):
        rows = read_jsonl(input_path)
        if args.limit:
            rows = rows[: args.limit]
        log(f"running {len(rows)} records from {input_path}")
        seconds = run_records(model, processor, jsm, rows, Path(output_path))
        stats = {
            "records": len(rows),
            "succeeded": len(seconds),
            "failed": len(rows) - len(seconds),
            "median_seconds": round(statistics.median(seconds), 2) if seconds else None,
            "max_seconds": round(max(seconds), 2) if seconds else None,
            "min_seconds": round(min(seconds), 2) if seconds else None,
            "output": str(output_path),
        }
        summary["runs"][str(input_path)] = stats
        log(f"SUMMARY {input_path}: {json.dumps(stats)}")
    summary["peak_rss_mib"] = round(peak_rss_mib(), 1)
    log(f"PEAK_RSS_MIB {summary['peak_rss_mib']}")
    if args.output:
        summary_path = Path(args.output[-1]).with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        log(f"summary -> {summary_path}")
    log("CPU_REFERENCE_DONE")


if __name__ == "__main__":
    main()
