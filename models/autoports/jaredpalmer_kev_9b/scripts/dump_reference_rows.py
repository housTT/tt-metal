import json
from pathlib import Path

from kev.api import SystemOneRequest, to_record
from kev.checkpoint import read_meta
from kev.data import api_request
from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE, encode, load_tokenizer, rows_of

ADAPTER = (
    "/home/hous/.cache/huggingface/hub/models--jaredpalmer--kev-9b/snapshots/db029f08b290afd9fee4aa4bbcd9ae48602d1eb0"
)
REF = Path("/home/hous/dev/kev/reports/reference")


def main():
    meta = read_meta(ADAPTER)
    tok = load_tokenizer(meta.base, revision=meta.base_revision)
    out = []
    with open(REF / "records.jsonl") as f:
        records = [json.loads(line) for line in f]
    for i, record in enumerate(records):
        rec, qmeta = to_record(SystemOneRequest.model_validate(api_request(record)))
        enc = encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH, strict=True)
        S, _, rows = rows_of(enc)
        for m, r in zip(qmeta, rows):
            out.append(
                {
                    "row_key": f"{i}:{m['id']}",
                    "record": i,
                    "qid": m["id"],
                    "type": m["type"],
                    "keys": m["keys"],
                    "legend": m.get("legend"),
                    "state_tokens": len(S),
                    "question_tokens": len(r["ids"]),
                    "row_tokens": len(S) + len(r["ids"]),
                    "ids": S + r["ids"],
                    "opt_positions": [len(S) + o for o in r["opts"]],
                    "decide_position": len(S) + r["decide"],
                }
            )
    with open(REF / "rows.json", "w") as f:
        json.dump({"adapter": ADAPTER, "base": meta.base, "base_revision": meta.base_revision, "rows": out}, f)
    print(len(records), "records,", len(out), "rows ->", REF / "rows.json")
    for r in out:
        print(
            r["row_key"],
            r["type"],
            "state",
            r["state_tokens"],
            "question",
            r["question_tokens"],
            "row",
            r["row_tokens"],
            "opts",
            len(r["opt_positions"]),
        )


if __name__ == "__main__":
    main()
