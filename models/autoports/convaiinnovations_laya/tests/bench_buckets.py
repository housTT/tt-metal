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


def loadavg():
    with open("/proc/loadavg") as f:
        return [float(x) for x in f.read().split()[:3]]


def p50(xs):
    return float(np.percentile(np.array(xs) * 1000.0, 50))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", default="1x512,8x512,64x512")
    ap.add_argument("--variants", default='{"default": {}}', help="JSON {name: PortConfig overrides}")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--eager-reps", type=int, default=5)
    ap.add_argument("--trace-region", type=int, default=512 * 1024 * 1024)
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    import ttnn
    from models.autoports.convaiinnovations_laya.tests import head_reference as HR
    from models.autoports.convaiinnovations_laya.tests import laya_inputs as LI
    from models.autoports.convaiinnovations_laya.tt import model_config as mc
    from models.autoports.convaiinnovations_laya.tt.laya_model import TtnnLayaModel
    from models.autoports.convaiinnovations_laya.tt.runner import LayaTraceRunner
    from models.autoports.convaiinnovations_laya.tt.weights import load_state_dict

    buckets = [tuple(int(v) for v in s.split("x")) for s in a.buckets.split(",")]
    variants = json.loads(a.variants)
    policy = mc.policy_from_name(a.policy)
    config = LI.load_config()
    sd = load_state_dict()
    inputs = {(b, s): LI.build_inputs(batch_size=b, seq_len=s, fill=b > 1) for b, s in buckets}
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "policy": policy.describe(),
        "buckets": [list(b) for b in buckets],
        "warm": a.warm,
        "reps": a.reps,
        "loadavg_start": loadavg(),
        "variants": {},
    }
    device = ttnn.open_device(device_id=0, l1_small_size=79104, trace_region_size=a.trace_region)
    try:
        for name, overrides in variants.items():
            port = mc.DEFAULT_PORT.with_(**overrides)
            entry = {"port": port.describe(), "buckets": {}, "error": None}
            model = None
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
                for b, s in buckets:
                    x = inputs[(b, s)]
                    bk = model.build_bucket(b, s)
                    cell = {"plan": mc.describe_plan(bk.encoder.plan)}
                    eager = []
                    for _ in range(a.eager_reps):
                        t0 = time.perf_counter()
                        out = model.forward(x["input_ids"], x["attention_mask"], x["qtype"], bucket=(b, s))
                        eager.append(time.perf_counter() - t0)
                    cell["eager_ms"] = [round(v * 1000.0, 3) for v in eager]
                    cell["eager_ms_p50"] = p50(eager[1:]) if len(eager) > 1 else p50(eager)
                    cell["eager_out"] = out
                    entry["buckets"][f"{b}x{s}"] = cell
                if not a.no_trace:
                    runner = LayaTraceRunner(model, buckets)
                    runner.warmup()
                    entry["warmup_seconds"] = runner.warmup_seconds
                    for b, s in buckets:
                        x = inputs[(b, s)]
                        cell = entry["buckets"][f"{b}x{s}"]
                        times = []
                        for _ in range(a.warm + a.reps):
                            t0 = time.perf_counter()
                            out = runner.run(x["input_ids"], x["attention_mask"], x["qtype"], bucket=(b, s))
                            times.append(time.perf_counter() - t0)
                        times = times[a.warm :]
                        cell["traced_ms_p50"] = p50(times)
                        cell["traced_ms_min"] = float(min(times)) * 1000.0
                        cell["traced_ms_p95"] = float(np.percentile(np.array(times) * 1000.0, 95))
                        cell["rows_per_s"] = b / (cell["traced_ms_p50"] / 1000.0)
                        cell["traced_vs_eager_max_abs_logits"] = float((out["logits"] - cell["eager_out"]["logits"]).abs().max())
                        cell["loadavg"] = loadavg()
                        mk = x["marker_mask"]
                        cell["marker_logits"] = HR.gather_markers(out["logits"], x["marker_pos"], mk)[mk].tolist()[:8]
                for cell in entry["buckets"].values():
                    cell.pop("eager_out", None)
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {str(exc)[:1500]}"
                for cell in entry["buckets"].values():
                    cell.pop("eager_out", None)
            finally:
                if runner is not None:
                    try:
                        runner.release()
                    except Exception:
                        pass
                if model is not None:
                    try:
                        model.close()
                    except Exception:
                        pass
                ttnn.synchronize_device(device)
            report["variants"][name] = entry
            print("VARIANT", name, json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("plan",)} for k, v in entry["buckets"].items()}), entry["error"])
    finally:
        ttnn.close_device(device)
    report["loadavg_end"] = loadavg()
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print("BENCH_DONE", a.out)


if __name__ == "__main__":
    main()
