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


def p50(xs):
    return float(np.percentile(np.array(xs) * 1000.0, 50))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", default="1x512,8x512")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--port", default="{}")
    ap.add_argument("--trace-region", type=int, default=512 * 1024 * 1024)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    import ttnn
    from models.autoports.convaiinnovations_laya.tests import head_reference as HR
    from models.autoports.convaiinnovations_laya.tests import laya_inputs as LI
    from models.autoports.convaiinnovations_laya.tt import model_config as mc
    from models.autoports.convaiinnovations_laya.tt.laya_model import TtnnLayaModel
    from models.autoports.convaiinnovations_laya.tt.runner import LayaTraceRunner
    from models.autoports.convaiinnovations_laya.tt.weights import load_state_dict, split_state_dict

    buckets = [tuple(int(v) for v in s.split("x")) for s in a.buckets.split(",")]
    policy = mc.policy_from_name(a.policy)
    port = mc.DEFAULT_PORT.with_(**json.loads(a.port))
    config = LI.load_config()
    sd = load_state_dict()
    parts = split_state_dict(sd)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "policy": policy.describe(),
        "port": port.describe(),
        "TT_METAL_TRACE_ALLOC_TRACKING": os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING"),
        "trace_region_size": a.trace_region,
        "rounds": a.rounds,
        "repeats": a.repeats,
        "buckets": [],
        "unsafe_allocation_error": None,
    }
    device = ttnn.open_device(device_id=0, l1_small_size=79104, trace_region_size=a.trace_region)
    runner = None
    try:
        model = TtnnLayaModel(
            device,
            config,
            state_dict=sd,
            policy=policy,
            port=port,
            row_buckets=sorted({b for b, _ in buckets}),
            seq_buckets=sorted({s for _, s in buckets}),
        )
        inputs = {}
        for b, s in buckets:
            inputs[(b, s, "A")] = LI.build_inputs(batch_size=b, seq_len=s, offset=0, fill=b > 1)
            inputs[(b, s, "B")] = LI.build_inputs(batch_size=b, seq_len=s, offset=37, fill=False)
        eager = {}
        for b, s in buckets:
            times = []
            for key in ("A", "B"):
                x = inputs[(b, s, key)]
                for i in range(a.repeats if key == "A" else 1):
                    t0 = time.perf_counter()
                    out = model.forward(x["input_ids"], x["attention_mask"], x["qtype"], bucket=(b, s))
                    times.append(time.perf_counter() - t0)
                eager[(b, s, key)] = out
            eager[(b, s, "times")] = times
        runner = LayaTraceRunner(model, buckets)
        runner.warmup()
        report["warmup_seconds"] = runner.warmup_seconds
        first = {}
        traced_times = {k: [] for k in buckets}
        identical = {k: True for k in buckets}
        for r in range(a.rounds):
            order = buckets if r % 2 == 0 else list(reversed(buckets))
            for b, s in order:
                for key in ("A", "B"):
                    x = inputs[(b, s, key)]
                    reps = a.repeats if (r == 0 and key == "A") else 1
                    for _ in range(reps):
                        t0 = time.perf_counter()
                        out = runner.run(x["input_ids"], x["attention_mask"], x["qtype"], bucket=(b, s))
                        traced_times[(b, s)].append(time.perf_counter() - t0)
                    fk = (b, s, key)
                    if fk not in first:
                        first[fk] = out
                    else:
                        if not (torch.equal(out["logits"], first[fk]["logits"]) and torch.equal(out["cls"], first[fk]["cls"])):
                            identical[(b, s)] = False
        for b, s in buckets:
            x = inputs[(b, s, "A")]
            real = x["attention_mask"] == 1
            tr_a, tr_b = first[(b, s, "A")], first[(b, s, "B")]
            eg_a = eager[(b, s, "A")]
            delta_eager = float((tr_a["logits"] - eg_a["logits"]).abs().max())
            delta_cls = float((tr_a["cls"] - eg_a["cls"]).abs().max())
            mk = x["marker_mask"]
            m_tr = HR.gather_markers(tr_a["logits"], x["marker_pos"], mk)[mk]
            m_eg = HR.gather_markers(eg_a["logits"], x["marker_pos"], mk)[mk]
            la, lb = tr_a["logits"][:, :512], tr_b["logits"][:, :512]
            n = min(la.shape[1], lb.shape[1])
            a_vs_b = float((la[:, :n] - lb[:, :n]).abs().max())
            et = eager[(b, s, "times")][1:]
            tt = traced_times[(b, s)][1:]
            report["buckets"].append(
                {
                    "batch": b,
                    "seq": s,
                    "eager_ms_p50": p50(et),
                    "eager_ms_min": float(min(et)) * 1000.0,
                    "traced_ms_p50": p50(tt),
                    "traced_ms_min": float(min(tt)) * 1000.0,
                    "speedup_p50": p50(et) / p50(tt),
                    "traced_vs_eager_max_abs_logits": delta_eager,
                    "traced_vs_eager_max_abs_cls": delta_cls,
                    "traced_vs_eager_bit_identical": bool(delta_eager == 0.0 and delta_cls == 0.0),
                    "marker_logits_eager": m_eg.tolist()[:8],
                    "marker_logits_traced": m_tr.tolist()[:8],
                    "repeated_replay_identical": identical[(b, s)],
                    "input_change_max_abs_delta": a_vs_b,
                    "updated_input_changes_output": bool(a_vs_b > 1e-3),
                    "replays": len(traced_times[(b, s)]),
                    "nan": bool(torch.isnan(tr_a["logits"]).any() or torch.isnan(tr_b["logits"]).any()),
                }
            )
        report["runner"] = runner.describe()
        report["model"] = model.describe()
    except RuntimeError as exc:
        report["unsafe_allocation_error"] = str(exc)[:4000]
        raise
    finally:
        if runner is not None:
            runner.release()
        if "model" in dir():
            model.close()
        ttnn.close_device(device)
        report["pass"] = report["unsafe_allocation_error"] is None and all(
            v["repeated_replay_identical"] and v["updated_input_changes_output"] and v["traced_vs_eager_bit_identical"] and not v["nan"]
            for v in report["buckets"]
        )
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)
        print("REPLAY_RESULT", json.dumps({"pass": report["pass"], "error": report["unsafe_allocation_error"], "buckets": report["buckets"]}))


if __name__ == "__main__":
    main()
