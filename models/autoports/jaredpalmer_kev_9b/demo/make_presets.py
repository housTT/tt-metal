import argparse
import ast
import json
import re
from pathlib import Path

KEV_REPO = Path("/home/hous/dev/kev/kev")
KEV_COMMIT = "952ce9d"
PLAYGROUND = KEV_REPO / "playground/src/lib/kev.ts"
BENCH = KEV_REPO / "scripts/serving_bench.py"
DOCUMENTS = KEV_REPO / "evals/documents-v1/development.jsonl"
HARD = KEV_REPO / "evals/hard-v1/development.jsonl"
DEVTOOLS = KEV_REPO / "evals/devtools-v1/development.jsonl"
DECISION = KEV_REPO / "evals/v7/decision-v7/development.jsonl"
OUT = Path(__file__).resolve().parent / "presets.json"

REPLAY_PLAN = [
    ("documents-v1", DOCUMENTS, 8, None),
    ("hard-v1", HARD, 8, None),
    ("devtools-v1", DEVTOOLS, 4, None),
    ("decision-v7 AG News", DECISION, 4, "agnews"),
]
REPLAY_SKIP = {"agnews/test/1080"}
SECOND_SET = ("tone", "escalate", "frustration")


def ts_block_to_json(text):
    start = text.index("export const PRESETS")
    start = text.index("= [", start) + 2
    depth, i = 0, start
    while True:
        c = text[i]
        if c in "\"'`":
            j = i + 1
            while text[j] != c:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if c in "[{":
            depth += 1
        elif c in "]}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    block = text[start : i + 1]
    out, i = [], 0
    while i < len(block):
        c = block[i]
        if c in "\"'":
            j = i + 1
            while block[j] != c:
                j += 2 if block[j] == "\\" else 1
            out.append(json.dumps(ast.literal_eval(block[i : j + 1])))
            i = j + 1
            continue
        m = re.match(r"[A-Za-z_$][A-Za-z0-9_$]*(?=\s*:)", block[i:])
        if m:
            out.append(json.dumps(m.group(0)))
            i += m.end()
            continue
        if c == ",":
            k = i + 1
            while k < len(block) and block[k] in " \t\r\n":
                k += 1
            if k < len(block) and block[k] in "]}":
                i += 1
                continue
        out.append(c)
        i += 1
    return json.loads("".join(out))


def bench_constants(path):
    tree = ast.parse(path.read_text())
    wanted = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("QUESTIONS", "TICKET", "PARAGRAPH"):
                wanted[name] = ast.literal_eval(node.value)
    missing = {"QUESTIONS", "TICKET", "PARAGRAPH"} - set(wanted)
    if missing:
        raise SystemExit(f"{path}: missing {sorted(missing)}")
    return wanted


def read_records(path):
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                yield line_no, json.loads(line)


def strip_record(rec):
    questions, labels = {}, {}
    for qid, q in rec["questions"].items():
        questions[qid] = {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
        if "label" in q:
            labels[qid] = q["label"]
    return questions, labels


def case_from_record(rec, line_no, path):
    questions, labels = strip_record(rec)
    rid = rec.get("_meta", {}).get("id", f"{path.name}:{line_no}")
    return {
        "name": rid,
        "source": f"{path.relative_to(KEV_REPO)} line {line_no}",
        "state": rec["state"],
        "questions": questions,
        "labels": labels,
    }


def cases_by_id(path, ids):
    found = {}
    for line_no, rec in read_records(path):
        rid = rec.get("_meta", {}).get("id")
        if rid in ids:
            found[rid] = case_from_record(rec, line_no, path)
    missing = [i for i in ids if i not in found]
    if missing:
        raise SystemExit(f"{path}: records not found: {missing}")
    return [found[i] for i in ids]


def replay_cases(path, count, source):
    out = []
    for line_no, rec in read_records(path):
        meta = rec.get("_meta", {})
        if source and meta.get("source") != source:
            continue
        if not isinstance(rec["state"], str) or meta.get("id") in REPLAY_SKIP:
            continue
        out.append(case_from_record(rec, line_no, path))
        if len(out) == count:
            break
    if len(out) < count:
        raise SystemExit(f"{path}: only {len(out)} usable records, wanted {count}")
    return out


def playground_case(presets, name):
    for p in presets:
        if p["name"] == name:
            return {
                "name": name,
                "source": f"playground/src/lib/kev.ts preset {name!r}",
                "state": p["state"],
                "questions": p["questions"],
                "blurb": p["blurb"],
            }
    raise SystemExit(f"playground preset {name!r} not found")


def build():
    pg = ts_block_to_json(PLAYGROUND.read_text())
    bench = bench_constants(BENCH)
    questions, ticket, paragraph = bench["QUESTIONS"], bench["TICKET"], bench["PARAGRAPH"]
    five = {k: questions[k] for k in ("department", "return_reason", "requested_resolution", "escalate", "frustration")}
    long_state = paragraph * 30
    triage = playground_case(pg, "Support triage")
    if triage["state"] != ticket or triage["questions"] != questions:
        raise SystemExit("playground Support triage and bench TICKET/QUESTIONS differ")
    presets = [
        {
            "title": "Support ticket triage",
            "what_it_shows": "Six questions of three types on one short ticket, answered in one request. The model card's quickstart case.",
            "cases": [
                {
                    "name": "ticket",
                    "source": "scripts/serving_bench.py TICKET and QUESTIONS (same as the playground Support triage preset)",
                    "state": ticket,
                    "questions": questions,
                }
            ],
        },
        {
            "title": "Long document, repeated state",
            "what_it_shows": "A 2,200-token state. The first run prefills it; a second run with the same state hits the prefix cache and skips the prefill. Case 2 asks a different question set on the same state and also hits the cache.",
            "cases": [
                {
                    "name": "5 questions",
                    "source": "scripts/serving_bench.py PARAGRAPH x30 with FIVE",
                    "state": long_state,
                    "questions": five,
                },
                {
                    "name": "second question set",
                    "source": "scripts/serving_bench.py PARAGRAPH x30 with tone, escalate, frustration",
                    "state": long_state,
                    "questions": {k: questions[k] for k in SECOND_SET},
                },
            ],
        },
        {
            "title": "Complaint routing",
            "what_it_shows": "Consumer complaints routed to a product and an issue, with the ground-truth labels next to the model's choice.",
            "cases": cases_by_id(DOCUMENTS, ["cfpb/9046322", "cfpb/2038700", "cfpb/8653073"]),
        },
        {
            "title": "Answer checking",
            "what_it_shows": "A yes/no judgement of a submitted arithmetic result plus a choice of the correct value, with labels.",
            "cases": cases_by_id(HARD, ["hard-v1/judge/development/00004", "hard-v1/judge/development/00053"]),
        },
        {
            "title": "Code review",
            "what_it_shows": "A git diff as the state: the change type and whether the commit message matches, with labels.",
            "cases": cases_by_id(DEVTOOLS, ["commitpackft/javascript/14553"]),
        },
        {
            "title": "News topic",
            "what_it_shows": "AG News topic as a choice plus two yes/no questions per article, with labels. Some options carry no description or a nested one.",
            "cases": cases_by_id(DECISION, ["agnews/test/1469", "agnews/test/7196", "agnews/test/5150"]),
        },
        {
            "title": "Review rating",
            "what_it_shows": "Score questions: ordered levels with an expected value between them, plus a yes/no with true and false criteria.",
            "cases": [playground_case(pg, "Review rating")],
        },
        {
            "title": "News article (object state)",
            "what_it_shows": "The state is a JSON object, not a string. Topic choice plus derived yes/no questions.",
            "cases": [playground_case(pg, "News article")],
        },
        {
            "title": "Isolation probe",
            "what_it_shows": "A secret placed in a sibling question must stay invisible to the probe question. Move it into the state and it becomes readable.",
            "cases": [playground_case(pg, "Isolation probe")],
        },
        {
            "title": "Boundary forgery",
            "what_it_shows": "An option text that tries to inject fake delimiters. The model must still see exactly three options.",
            "cases": [playground_case(pg, "Boundary forgery")],
        },
    ]
    replay = []
    for suite, path, count, source in REPLAY_PLAN:
        for c in replay_cases(path, count, source):
            replay.append({"suite": suite, **c})
    return {
        "generated_by": "make_presets.py",
        "kev_repo": str(KEV_REPO),
        "kev_commit": KEV_COMMIT,
        "model": "kev-latest",
        "presets": presets,
        "replay": replay,
    }


def main():
    ap = argparse.ArgumentParser(description="Rebuild presets.json from the kev checkout.")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    data = build()
    Path(args.out).write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    n_cases = sum(len(p["cases"]) for p in data["presets"])
    print(f"wrote {args.out}: {len(data['presets'])} presets, {n_cases} cases, {len(data['replay'])} replay records")


if __name__ == "__main__":
    main()
