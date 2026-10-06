# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import socket
import time
from datetime import datetime, timezone

import numpy as np
import torch

os.environ.setdefault("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES", "0")

BENCH_LATENCY_DIR = "/home/hous/dev/laya/evals/latency"
DOC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "doc", "optimized_full_model")
MAX_LEN = 512
HEAD_MAX_LEN = 192
SHORT_STATE = {"m": "refund me"}


def loadavg():
    with open("/proc/loadavg") as f:
        return [float(x) for x in f.read().split()[:3]]


def stats(values_ms):
    arr = np.asarray(values_ms, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


STATE_EN = {
    "ticket": {
        "subject": "Payout failing",
        "messages": [
            {
                "from": "customer",
                "text": "Hi, my Stripe payouts have failed for 3 days and I am losing sales. Please help ASAP. " * 6,
            }
        ],
    }
}
Q_NOUL = {"type": "noul", "instructions": "Does `ticket.messages[0].text` express urgency?"}
Q_CHOICE = {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {"billing": "payments", "technical": "bugs and integrations", "sales": "pricing"},
}


def qs(n):
    return {("q%d" % i): (Q_NOUL if i % 2 else Q_CHOICE) for i in range(n)}


def speed_table_inputs():
    """STATE_EN, Q_NOUL, Q_CHOICE and qs(n) copied from the authors' research/scripts/bench_latency.py (the file at
    /home/hous/dev/laya/evals/latency/bench_latency.py imports pip laya, which the tt-metal venv does not have)."""
    return STATE_EN, Q_CHOICE, Q_NOUL, qs


def request_rows(tok, state, questions, max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN):
    """The server's request-to-rows path (server/engine.py encode_state on the vendored builder) collated to tensors."""
    from models.autoports.convaiinnovations_laya.server.engine import encode_state, to_internal
    from models.autoports.convaiinnovations_laya.vendor.rl_common import collate_items

    ids = list(questions.keys())
    internal = {qid: to_internal(questions[qid]) for qid in ids}
    items = encode_state(tok, state, ids, internal, max_len, head_max_len)
    b = collate_items([items], tok.pad_token_id)
    return {
        "input_ids": b["input_ids"],
        "attention_mask": b["attention_mask"],
        "marker_pos": b["marker_pos"],
        "marker_mask": b["marker_mask"],
        "qtype": b["qtype"],
        "lengths": [int(v) for v in b["attention_mask"].sum(-1)],
    }


def temperatures_for(engine, qtype, marker_mask):
    """temperature_by_options[temp_bucket] else temperature[type], clamped to [0.5, 5.0] (pip laya rule, amendment A6)."""
    from models.autoports.convaiinnovations_laya.server.decode import Temperatures

    temps = Temperatures(engine.rl_config, clamp=True)
    out = [temps.for_question(int(t), int(k)) for t, k in zip(qtype.tolist(), marker_mask.sum(-1).tolist())]
    return torch.tensor(out, dtype=torch.float32)


def decode_probs(logits, temps):
    z = logits / temps[:, None]
    return torch.softmax(z, -1)


def time_cell(engine, rows, warm, reps, split_reps, label):
    """Warm `warm` calls, then `reps` timed calls through LayaEngine.forward_detailed plus the temperature softmax; then a blocking split."""
    totals, device, tail, decode = [], [], [], []
    temps = None
    for i in range(warm + reps):
        t0 = time.perf_counter()
        res = engine.forward_detailed(
            rows["input_ids"], rows["attention_mask"], rows["marker_pos"], rows["marker_mask"], rows["qtype"]
        )
        t1 = time.perf_counter()
        if temps is None:
            temps = _temps(engine, rows)
        probs = decode_probs(res["logits"], temps)
        t2 = time.perf_counter()
        if i >= warm:
            totals.append((t2 - t0) * 1000.0)
            device.append(res["device_ms"])
            tail.append(res["host_tail_ms"])
            decode.append((t2 - t1) * 1000.0)
    bucket = list(res["bucket"])
    split = {}
    if split_reps > 0 and engine.runner is not None:
        parts = []
        for _ in range(split_reps):
            t = engine.runner.run_timed(rows["input_ids"], rows["attention_mask"], rows["qtype"])
            parts.append((t["write_ms"], t["replay_ms"], t["readback_ms"], t["device_ms"]))
        arr = np.array(parts)
        split = {
            "write_ms_p50": float(np.median(arr[:, 0])),
            "replay_ms_p50": float(np.median(arr[:, 1])),
            "readback_ms_p50": float(np.median(arr[:, 2])),
            "blocking_total_ms_p50": float(np.median(arr[:, 3])),
            "reps": split_reps,
        }
    n_rows = int(rows["input_ids"].shape[0])
    total = stats(totals)
    return {
        "label": label,
        "rows": n_rows,
        "row_lengths": rows["lengths"],
        "longest_row": int(max(rows["lengths"])),
        "bucket": bucket,
        "padded_tokens": int(bucket[0] * bucket[1]),
        "end_to_end_ms": total,
        "device_ms": stats(device),
        "host_tail_ms": stats(tail),
        "decode_ms": stats(decode),
        "ms_per_row_p50": total["p50"] / n_rows if total.get("p50") else None,
        "rows_per_s_p50": 1000.0 * n_rows / total["p50"] if total.get("p50") else None,
        "tokens_per_s_p50": 1000.0 * int(sum(rows["lengths"])) / total["p50"] if total.get("p50") else None,
        "padded_tokens_per_s_p50": 1000.0 * bucket[0] * bucket[1] / total["p50"] if total.get("p50") else None,
        "blocking_split": split,
        "probs_first_row": [round(float(v), 4) for v in probs[0][: int(rows["marker_mask"][0].sum())].tolist()],
        "loadavg": loadavg(),
        "warm": warm,
        "reps": reps,
    }


def _temps(engine, rows):
    try:
        return temperatures_for(engine, rows["qtype"], rows["marker_mask"])
    except Exception:
        k = rows["marker_mask"].sum(-1).clamp(min=1)
        return torch.ones(k.shape[0], dtype=torch.float32)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["deployment", "fresh", "throughput"], default="deployment")
    ap.add_argument("--cells", default="1,5,10,50", help="questions per call of the STATE_EN speed-table shape")
    ap.add_argument(
        "--throughput-cells", default="64x128,64x256,64x512", help="rows x seq cells filled with speed-table rows"
    )
    ap.add_argument(
        "--row-buckets", default=None, help="per-device row buckets (default: tt/model_config.py ROW_BUCKETS)"
    )
    ap.add_argument("--seq-buckets", default=None, help="seq buckets (default: tt/model_config.py SEQ_BUCKETS)")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--port", default="{}", help="JSON of PortConfig overrides")
    ap.add_argument("--mesh", default="1x1")
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--split-reps", type=int, default=10)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--trace-region", type=int, default=512 * 1024 * 1024)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    from models.autoports.convaiinnovations_laya.tests import laya_inputs as LI
    from models.autoports.convaiinnovations_laya.tt import model_config as mc
    from models.autoports.convaiinnovations_laya.tt.engine import LayaEngine, parse_mesh_shape

    policy = mc.policy_from_name(a.policy)
    port = mc.DEFAULT_PORT.with_(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in json.loads(a.port).items()})
    row_buckets = tuple(int(x) for x in a.row_buckets.split(",")) if a.row_buckets else mc.ROW_BUCKETS
    seq_buckets = tuple(int(x) for x in a.seq_buckets.split(",")) if a.seq_buckets else mc.SEQ_BUCKETS
    mesh = parse_mesh_shape(a.mesh)
    num_devices = mesh[0] * mesh[1]
    tok = LI.load_tokenizer()
    STATE_EN, Q_CHOICE, Q_NOUL, qs = speed_table_inputs()
    cells = [int(x) for x in a.cells.split(",") if x.strip()]
    tp_cells = mc.parse_buckets(a.throughput_cells) if a.mode == "throughput" else []

    inputs = {}
    for n in cells:
        inputs[("cell", n)] = request_rows(tok, STATE_EN, qs(n))
    for b, s in tp_cells:
        if s > 256:
            x = LI.build_inputs(batch_size=b, seq_len=s, fill=True)
            rows = {k: x[k] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}
            rows["lengths"] = [int(v) for v in x["lengths"]]
            rows["input_ids"] = rows["input_ids"][:, : max(rows["lengths"])]
            rows["attention_mask"] = rows["attention_mask"][:, : max(rows["lengths"])]
        else:
            rows = request_rows(tok, STATE_EN if s >= 256 else SHORT_STATE, qs(b))
        if rows["input_ids"].shape[1] > s:
            raise ValueError(f"throughput rows of {rows['input_ids'].shape[1]} tokens do not fit seq bucket {s}")
        inputs[("tp", b, s)] = rows

    def bucket_of(rows):
        n, L = rows["input_ids"].shape
        return mc.pick_bucket(-(-n // num_devices), row_buckets), mc.pick_bucket(int(L), seq_buckets)

    if a.mode == "deployment":
        warm_shapes = [(b, s) for s in seq_buckets for b in row_buckets]
    else:
        warm_shapes = sorted({bucket_of(r) for r in inputs.values()})
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "mode": a.mode,
        "mesh": a.mesh,
        "policy": policy.describe(),
        "port": port.describe(),
        "row_buckets": list(row_buckets),
        "seq_buckets": list(seq_buckets),
        "warm_shapes": [list(w) for w in warm_shapes],
        "protocol": {
            "inputs": "bench_latency.STATE_EN with qs(n) (alternating Q_CHOICE 3-way and Q_NOUL), tokenized by the server path at 512 / 192",
            "timing": f"{a.warm} warm calls, p50 of {a.reps} calls of LayaEngine.forward_detailed plus the temperature softmax; blocking split p50 of {a.split_reps}",
            "throughput_inputs": "STATE_EN rows (194 to 205 tokens) at seq 256; the short state {'m': 'refund me'} (about 40 tokens) at seq 128; typed-decisions rows with one row filled to 512 tokens at seq 512",
        },
        "loadavg_start": loadavg(),
        "cells": {},
        "throughput": {},
    }
    t0 = time.perf_counter()
    engine = LayaEngine(
        mesh_shape=mesh,
        policy=policy,
        port=port,
        row_buckets=row_buckets,
        seq_buckets=seq_buckets,
        trace=True,
        trace_region_size=a.trace_region,
        warmup_shapes=warm_shapes,
        threads=a.threads,
    )
    try:
        report["engine_load_seconds"] = round(time.perf_counter() - t0, 2)
        report["engine"] = engine.shapes()
        report["runner"] = engine.runner.describe()
        report["warmup_seconds"] = engine.warmup_seconds
        report["loadavg_after_warmup"] = loadavg()
        for n in cells:
            cell = time_cell(engine, inputs[("cell", n)], a.warm, a.reps, a.split_reps, f"{n} questions")
            report["cells"][str(n)] = cell
            print(
                "CELL",
                n,
                json.dumps(
                    {
                        k: cell[k]
                        for k in ("bucket", "end_to_end_ms", "device_ms", "host_tail_ms", "blocking_split", "loadavg")
                    }
                ),
            )
        for b, s in tp_cells:
            cell = time_cell(engine, inputs[("tp", b, s)], a.warm, a.reps, a.split_reps, f"{b} rows at seq {s}")
            report["throughput"][f"{b}x{s}"] = cell
            print(
                "THROUGHPUT",
                f"{b}x{s}",
                json.dumps(
                    {
                        k: cell[k]
                        for k in (
                            "bucket",
                            "end_to_end_ms",
                            "rows_per_s_p50",
                            "tokens_per_s_p50",
                            "blocking_split",
                            "loadavg",
                        )
                    }
                ),
            )
        report["buckets_described"] = engine.model.describe()["buckets"]
    finally:
        engine.close()
    report["loadavg_end"] = loadavg()
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print("BENCH_LATENCY_DONE", a.out)


if __name__ == "__main__":
    main()
