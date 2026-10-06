# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import socket
import time
from datetime import datetime, timezone

import torch

os.environ.setdefault("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES", "0")

from models.autoports.convaiinnovations_laya.tests.run_fidelity import call_tensors, index_for, load_corpus, loadavg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/home/hous/dev/laya/reference/parity_corpus_td.npz")
    ap.add_argument("--model-dir", default="/home/hous/dev/laya/state/laya_models/laya-typed-decisions")
    ap.add_argument("--rows", type=int, default=40, help="gate rows sent in one engine call")
    ap.add_argument("--seq-buckets", default="1024")
    ap.add_argument("--row-buckets", default="1,2,4,5,8,10,16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.environ["LAYA_MODEL_DIR"] = a.model_dir
    from models.autoports.convaiinnovations_laya.tt.engine import LayaEngine

    data = load_corpus(a.corpus, index_for(a.corpus))
    gate_rows = [r for r in range(len(data["items"])) if int(data["source"][r]) == 0][: a.rows]
    seqs = tuple(sorted(int(x) for x in a.seq_buckets.split(",")))
    rows = tuple(sorted(int(x) for x in a.row_buckets.split(",")))
    t0 = time.perf_counter()
    engine = LayaEngine(model_dir=a.model_dir, seq_buckets=seqs, row_buckets=rows, trace=True, threads=6)
    cap = engine.model.max_rows_for_seq(seqs[-1])
    t = call_tensors(data, gate_rows)
    one = engine.forward_detailed(t["input_ids"], t["attention_mask"], t["marker_pos"], t["marker_mask"], t["qtype"])
    parts = []
    for i in range(0, len(gate_rows), cap):
        tp = call_tensors(data, gate_rows[i : i + cap])
        L = t["input_ids"].shape[1]
        ids = torch.full((tp["input_ids"].shape[0], L), int(engine.model.pad_id), dtype=torch.long)
        att = torch.zeros((tp["input_ids"].shape[0], L), dtype=torch.long)
        ids[:, : tp["input_ids"].shape[1]] = tp["input_ids"]
        att[:, : tp["input_ids"].shape[1]] = tp["attention_mask"]
        kmax = t["marker_pos"].shape[1]
        mpos = torch.zeros((ids.shape[0], kmax), dtype=torch.long)
        mmask = torch.zeros((ids.shape[0], kmax), dtype=torch.bool)
        mpos[:, : tp["marker_pos"].shape[1]] = tp["marker_pos"]
        mmask[:, : tp["marker_mask"].shape[1]] = tp["marker_mask"]
        parts.append(engine.forward_detailed(ids, att, mpos, mmask, tp["qtype"]))
    logits_parts = torch.cat([p["logits"] for p in parts], 0)
    act_parts = torch.cat([p["act_logits"] for p in parts], 0)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "model_dir": a.model_dir,
        "engine_load_seconds": round(time.perf_counter() - t0, 1),
        "rows": len(gate_rows),
        "call_seq_len": int(t["input_ids"].shape[1]),
        "max_rows_for_seq": cap,
        "one_call_buckets": [list(b) for b in one["buckets"]],
        "one_call_device_ms": one["device_ms"],
        "explicit_call_buckets": [list(b) for p in parts for b in p["buckets"]],
        "explicit_calls_device_ms_sum": float(sum(p["device_ms"] for p in parts)),
        "logits_bit_identical": bool(torch.equal(one["logits"], logits_parts)),
        "act_logits_bit_identical": bool(torch.equal(one["act_logits"], act_parts)),
        "max_abs_dlogit": float((one["logits"] - logits_parts).abs().max()),
        "nan": bool(torch.isnan(one["logits"]).any()),
        "shapes": {
            k: engine.shapes()[k]
            for k in ("row_buckets", "row_buckets_by_seq", "max_rows_by_seq", "warm_shapes", "trace_bytes_total")
        },
        "loadavg": loadavg(),
    }
    report["pass"] = (
        report["logits_bit_identical"]
        and report["act_logits_bit_identical"]
        and not report["nan"]
        and len(one["buckets"]) == -(-len(gate_rows) // cap)
    )
    engine.close()
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print("CHUNK_RESULT", json.dumps({k: v for k, v in report.items() if k != "shapes"}))


if __name__ == "__main__":
    main()
