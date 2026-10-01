import gc
import json
import sys
import time
from pathlib import Path

import torch
from kev.api import SystemOneRequest, to_answers, to_record
from kev.checkpoint import LoadOptions
from kev.data import api_request
from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE, rows_of
from kev.predictors import LocalPredictor
from kev.suite import SERVING_CONTEXT

ADAPTER = (
    "/home/hous/.cache/huggingface/hub/models--jaredpalmer--kev-9b/snapshots/db029f08b290afd9fee4aa4bbcd9ae48602d1eb0"
)
REF = Path("/home/hous/dev/kev/reports/reference")
CHECK_RECORDS = (0, 4, 8)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def score(pred, record, hidden):
    rec, meta = to_record(SystemOneRequest.model_validate(api_request(record)))
    enc = pred.model.encode(pred.tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH, strict=True)
    captured = []
    hook = pred.model.lm.register_forward_hook(lambda m, i, o: captured.append(o.last_hidden_state.float()))
    t = time.perf_counter()
    try:
        logits = pred.model.forward(enc)
    finally:
        hook.remove()
    dt = 1000 * (time.perf_counter() - t)
    ps = [torch.softmax(z, -1) for z in logits]
    S, _, rows = rows_of(enc)
    if hidden is not None:
        flat = [h[i] for h in captured for i in range(h.shape[0])]
        assert len(flat) == len(rows), (len(flat), len(rows))
        for m, r, h in zip(meta, rows, flat):
            L = len(S) + len(r["ids"])
            h = h[:L]
            opts, dec = [len(S) + o for o in r["opts"]], len(S) + r["decide"]
            z = pred.model.head(h[dec], h[opts])
            assert torch.allclose(z, logits[meta.index(m)], atol=1e-4), (z, logits[meta.index(m)])
            hidden[m["id"]] = h[opts + [dec]].clone()
    answers = to_answers([p.tolist() for p in ps], meta)
    return {
        "input_tokens": len(enc["ids"]),
        "latency_ms": dt,
        "questions": {
            m["id"]: {
                "type": m["type"],
                "keys": m["keys"],
                "probabilities": dict(zip(m["keys"], p.tolist())),
                "logits": dict(zip(m["keys"], z.tolist())),
                "answer": answers[m["id"]],
            }
            for m, p, z in zip(meta, ps, logits)
        },
    }


def run(dtype, name, want_hidden):
    opts = LoadOptions() if dtype is None else LoadOptions(dtype=dtype)
    t = time.perf_counter()
    pred = LocalPredictor(ADAPTER, "cpu", opts=opts, context=SERVING_CONTEXT)
    log(
        name,
        "loaded in",
        round(time.perf_counter() - t),
        "s; dtype",
        pred.model.dtype,
        "temperature",
        pred.temperature,
        pred.environment,
    )
    with open(REF / "records.jsonl") as f:
        records = [json.loads(line) for line in f]
    out, hidden, checks = {}, {}, {}
    for i, record in enumerate(records):
        h = {}
        res = score(pred, record, h if want_hidden else None)
        for qid, v in h.items():
            hidden[f"{i}:{qid}"] = v
        out[str(i)] = {"source": record["_meta"].get("reference_source"), **res}
        log(
            name,
            "record",
            i,
            res["input_tokens"],
            "tokens",
            round(res["latency_ms"]),
            "ms",
            {
                q: v["answer"].get("choice", v["answer"].get("noul", v["answer"].get("score")))
                for q, v in res["questions"].items()
            },
        )
        if i in CHECK_RECORDS:
            theirs = pred(record)
            diff = max(
                abs(theirs["probabilities"][q][k] - out[str(i)]["questions"][q]["probabilities"][k])
                for q in theirs["probabilities"]
                for k in theirs["probabilities"][q]
            )
            checks[str(i)] = {"max_abs_dp_vs_LocalPredictor_call": diff, "latency_ms": theirs["latency_ms"]}
            log(name, "check record", i, "max |dp| vs LocalPredictor.__call__", diff)
    result = {
        "adapter": ADAPTER,
        "dtype": pred.model.dtype,
        "temperature": pred.temperature,
        "environment": pred.environment,
        "context": SERVING_CONTEXT,
        "checks": checks,
        "records": out,
    }
    with open(REF / f"probs_{name}.json", "w") as f:
        json.dump(result, f, indent=1)
    if want_hidden:
        torch.save(hidden, REF / f"hidden_{name}.pt")
    log(name, "done; wrote", REF / f"probs_{name}.json")
    del pred
    gc.collect()


def main():
    log("torch", torch.__version__, "threads", torch.get_num_threads())
    run(None, "fp32", True)
    if "--no-bf16" not in sys.argv:
        run(torch.bfloat16, "bf16", False)


if __name__ == "__main__":
    main()
