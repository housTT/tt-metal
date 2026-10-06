# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
AUTOPORT = os.path.dirname(HERE)
DOC_DIR = os.path.join(AUTOPORT, "doc", "datatype_sweep")
DEFAULT_POLICIES = (
    "bf8w_hifi3_erf",
    "bf16_hifi4",
    "bf8w_hifi3",
    "bf8w_hifi2",
    "bf8w_lofi_mlp",
    "bf8_act",
    "bf8w_hifi3_head_bf16",
    "bf8w_hifi2_erf",
    "bf8w_lofi_mlp_erf",
)
CELLS = ("1", "5", "10", "50")
TIE_PCT = 1.0
GATES = {"confident_agreement": 0.98, "median_max_abs_dp": 0.02, "scorer_pcc": 0.99, "hidden_pcc": 0.99}


def loadavg():
    with open("/proc/loadavg") as f:
        return [float(x) for x in f.read().split()[:3]]


def wait_quiet(limit=8.0, max_wait_s=3600):
    t0 = time.time()
    while loadavg()[0] >= limit:
        if time.time() - t0 > max_wait_s:
            return False
        print("SWEEP load", loadavg(), ">= limit, waiting", flush=True)
        time.sleep(30)
    return True


def run(cmd, log_path):
    print("SWEEP run", " ".join(cmd), flush=True)
    with open(log_path, "a") as log:
        log.write("=== " + " ".join(cmd) + "\n")
        log.flush()
        rc = subprocess.call(
            cmd, stdout=log, stderr=subprocess.STDOUT, cwd=os.environ.get("TT_METAL_HOME", os.getcwd())
        )
    print("SWEEP rc", rc, flush=True)
    return rc


def fidelity_row(fid):
    g = fid["gates"]
    gs = fid["gate_subset"]
    hidden = fid.get("hidden_states") or {}
    return {
        "pass": bool(fid.get("pass")),
        "confident_agree": g["confident_argmax_agreement"]["agree"],
        "confident_n": g["confident_argmax_agreement"]["of"],
        "confident_agree_rate": g["confident_argmax_agreement"]["value"],
        "median_max_abs_dp": g["median_max_abs_dp"]["value"],
        "scorer_logit_pcc": g["scorer_logit_pcc"]["value"],
        "hidden_encoder_pcc": hidden.get("encoder_pcc_pooled"),
        "hidden_head_pcc": hidden.get("head_pcc_pooled"),
        "hidden_encoder_pcc_min_call": hidden.get("encoder_pcc_min_call"),
        "nan_rows": g["no_nan"]["value"],
        "plain_argmax_agree_rate": gs["argmax_agree_rate"],
        "plain_argmax_agree": gs["argmax_agree"],
        "act_argmax_agree_rate": gs["act_argmax_agree_rate"],
        "p95_max_abs_dp": gs["max_abs_dp_p95"],
        "max_abs_dp": gs["max_abs_dp_max"],
        "max_abs_dlogit": gs["max_abs_dlogit"],
        "gate_detail": {k: v["pass"] for k, v in g.items()},
        "device_ms_per_call_p50": fid.get("device_ms_per_call_p50"),
        "bucket_histogram": fid.get("bucket_histogram"),
        "loadavg": fid.get("loadavg_after_device_pass"),
    }


def latency_row(lat):
    cells = {}
    for n in CELLS:
        c = lat["cells"].get(n)
        cells[n] = {
            "end_to_end_ms_p50": c["end_to_end_ms"]["p50"] if c else None,
            "device_ms_p50": c["device_ms"]["p50"] if c else None,
            "bucket": c["bucket"] if c else None,
            "loadavg": c["loadavg"] if c else None,
        }
    vals = [cells[n]["end_to_end_ms_p50"] for n in CELLS]
    return {
        "cells": cells,
        "latency_sum_ms": sum(vals) if all(v is not None for v in vals) else None,
        "warm_shapes": lat.get("warm_shapes"),
    }


GATE_WIDTHS = {
    "confident_agree_rate": (GATES["confident_agreement"], 1.0),
    "median_max_abs_dp": (GATES["median_max_abs_dp"], 0.0),
    "scorer_logit_pcc": (GATES["scorer_pcc"], 1.0),
    "hidden_encoder_pcc": (GATES["hidden_pcc"], 1.0),
    "hidden_head_pcc": (GATES["hidden_pcc"], 1.0),
}
THIN_FRACTION = 0.10
CONFIRM_E2 = {"accuracy": 0.010, "soft_accuracy": 0.015, "brier_vs_soft": 0.015, "ece": 0.015, "score_mae": 0.015}
CONFIRM_E1 = {"confident_agree_rate": 0.98, "argmax_agree_rate": 0.95}


def thin_margins(row):
    """Gates the row clears by less than THIN_FRACTION of the gate's width (threshold to the ideal value); amendment A11."""
    out = {}
    for key, (threshold, ideal) in GATE_WIDTHS.items():
        v = row.get(key)
        if v is None:
            continue
        width = abs(ideal - threshold)
        margin = (v - threshold) if ideal > threshold else (threshold - v)
        out[key] = {
            "value": v,
            "threshold": threshold,
            "margin": margin,
            "width": width,
            "thin": bool(margin < THIN_FRACTION * width),
        }
    return out


def confirmation(name, results_dir, reference_dir):
    """Amendment A11 confirmation gates on a served host_tt run: full E1 (488 items) and full E2 (400 cases)."""
    par = json.load(open(os.path.join(results_dir, "parity", "parity.json")))
    sc = json.load(open(os.path.join(results_dir, "typed_decisions", "score.json")))
    ref = json.load(open(os.path.join(reference_dir, "typed_decisions", "reference.json")))
    ref_row = ref.get("row") or ref.get("metrics") or ref
    row = sc["row"]
    e2 = {}
    for k, tol in CONFIRM_E2.items():
        delta = row[k] - float(ref_row[k])
        e2[k] = {
            "served": row[k],
            "cpu_fp32": float(ref_row[k]),
            "delta": delta,
            "tolerance": tol,
            "pass": bool(abs(delta) <= tol),
        }
    e1 = {}
    for path in ("tensor", "wire"):
        o = par["paths"][path]["overall"]
        e1[path] = {
            "n": o["n"],
            "confident_agree": o["confident_agree"],
            "confident_n": o["confident_n"],
            "confident_agree_rate": o["confident_agree_rate"],
            "argmax_agree": o["argmax_agree"],
            "argmax_agree_rate": o["argmax_agree_rate"],
            "median_max_abs_dp": o["median_max_abs_dp"],
            "max_abs_dp": o["max_abs_dp"],
            "pcc_scorer_logits": o.get("pcc_scorer_logits"),
            "pass": bool(
                o["confident_agree_rate"] >= CONFIRM_E1["confident_agree_rate"]
                and o["argmax_agree_rate"] >= CONFIRM_E1["argmax_agree_rate"]
            ),
        }
    passed = all(v["pass"] for v in e2.values()) and e1["tensor"]["pass"] and e1["wire"]["pass"]
    return {
        "policy": name,
        "results_dir": results_dir,
        "reference_dir": reference_dir,
        "e2": e2,
        "e2_cases": sc.get("cases_scored"),
        "e2_complete": sc.get("complete"),
        "e2_latency_client_p50_ms": (sc.get("latency") or {}).get("client_ms", {}).get("p50"),
        "e2_batch_histogram": sc.get("batch_histogram"),
        "e1": e1,
        "e1_batch_histogram": par["paths"]["tensor"].get("requests", {}).get("batch_histogram"),
        "gates": {"e2": CONFIRM_E2, "e1": CONFIRM_E1},
        "pass": bool(passed),
    }


def confirmation_md(conf):
    names = list(conf["runs"].keys())
    lines = ["| gate | " + " | ".join(names) + " | threshold |", "|---|" + "---|" * len(names) + "---|"]
    for k in CONFIRM_E2:
        cells = []
        for n in names:
            e = conf["runs"][n]["e2"][k]
            cells.append(f"{e['served']:.4f} (delta {e['delta']:+.4f}, {'pass' if e['pass'] else 'FAIL'})")
        ref = conf["runs"][names[0]]["e2"][k]["cpu_fp32"]
        lines.append(f"| E2 {k} | " + " | ".join(cells) + f" | within {CONFIRM_E2[k]} of CPU fp32 {ref:.4f} |")
    for path in ("tensor", "wire"):
        for k, label in (("confident_agree_rate", "confident agreement"), ("argmax_agree_rate", "argmax agreement")):
            cells = []
            for n in names:
                e = conf["runs"][n]["e1"][path]
                cnt = (
                    f"{e['confident_agree']} of {e['confident_n']}"
                    if k == "confident_agree_rate"
                    else f"{e['argmax_agree']} of {e['n']}"
                )
                cells.append(f"{100 * e[k]:.2f} percent ({cnt})")
            lines.append(
                f"| E1 {path} path {label} | " + " | ".join(cells) + f" | >= {100 * CONFIRM_E1[k]:.0f} percent |"
            )
    lines.append(
        "| confirmation | "
        + " | ".join("pass" if conf["runs"][n]["pass"] else "FAIL" for n in names)
        + " | all of the above |"
    )
    inv = conf.get("candidate_stage6_invariance")
    if inv:
        g = inv["max_abs_dp_alone_vs_in_batch"]
        lines.append(
            f"| stage 6 alone versus in batch (candidate {conf['candidate']}) | max abs delta p {g['value']:.4f}, same argmax {inv['same_argmax']['value']} of {inv['same_argmax']['of']} ({'pass' if inv['pass'] else 'FAIL'}) | | <= 0.01 and the same argmax |"
        )
    return "\n".join(lines) + "\n"


def cmd_confirm(a):
    out_dir = a.out_dir
    res = json.load(open(os.path.join(out_dir, "sweep_results.json")))
    by = {r["policy"]: r for r in res["rows"]}
    cand = by[a.candidate]
    thin = thin_margins(cand)
    needed = any(v["thin"] for v in thin.values())
    conf = {
        "rule": "amendment A11: when the fastest passing policy clears any gate by less than 10 percent of the gate's width, it is confirmed on the full served workload against the runner-up: served E2 accuracy within 0.010 of the CPU fp32 row and soft accuracy, Brier, ECE and score MAE within 0.015; E1 confident agreement over the 488 items at least 98 percent and argmax agreement at least 95 percent",
        "candidate": a.candidate,
        "runner_up": a.runner_up,
        "candidate_thin_margins": thin,
        "confirmation_needed": bool(needed),
        "runs": {},
    }
    for name, d in ((a.candidate, a.candidate_dir), (a.runner_up, a.runner_up_dir)):
        if d and os.path.isdir(d):
            conf["runs"][name] = confirmation(name, d, a.reference_dir)
    cand_ok = conf["runs"].get(a.candidate, {}).get("pass")
    inv_ok = None
    if a.invariance and os.path.exists(a.invariance):
        inv = json.load(open(a.invariance))
        g = inv["gates"]
        conf["candidate_stage6_invariance"] = {
            "file": os.path.relpath(a.invariance, AUTOPORT),
            "same_argmax": g["same_argmax"],
            "max_abs_dp_alone_vs_in_batch": g["max_abs_dp_alone_vs_in_batch"],
            "max_abs_dp_alone_vs": inv["summary"]["max_abs_dp_alone_vs"],
            "pcc_logits_alone_vs": inv["summary"]["pcc_logits_alone_vs"],
            "buckets": inv.get("buckets"),
            "pass": bool(inv.get("pass")),
        }
        inv_ok = bool(inv.get("pass"))
    if inv_ok is False:
        selected = a.runner_up
        why = (
            f"fastest, passes the A.7 gates and the A11 served confirmation, fails the stage 6 alone-versus-in-batch gate "
            f"(max abs delta p {conf['candidate_stage6_invariance']['max_abs_dp_alone_vs_in_batch']['value']:.4f} against 0.01); the runner-up ships"
        )
    elif not needed:
        selected, why = a.candidate, "no gate margin under 10 percent of its width; the sweep selection stands"
    elif cand_ok:
        selected, why = a.candidate, "the candidate passes the served confirmation gates"
    elif cand_ok is None:
        selected, why = a.runner_up, "the candidate's confirmation run is missing; the runner-up ships"
    else:
        selected, why = a.runner_up, "fastest, fails confirmation; the runner-up ships"
    conf["selected"] = selected
    conf["reason"] = why
    res["confirmation"] = conf
    res["selected_after_confirmation"] = selected
    with open(os.path.join(out_dir, "sweep_results.json"), "w") as f:
        json.dump(res, f, indent=1)
    sel_path = os.path.join(out_dir, "selected_precision_config.json")
    sel = json.load(open(sel_path)) if os.path.exists(sel_path) else {}
    chosen = by[selected]
    sel.update(
        {
            "selected_policy": selected,
            "sweep_fastest_passing": a.candidate,
            "default_changed": selected != "bf8w_hifi3_erf",
            "reason": res["selection_reason"] + "; " + why,
            "policy": chosen.get("policy_describe"),
            "gates": {
                k: chosen.get(k)
                for k in (
                    "confident_agree",
                    "confident_n",
                    "confident_agree_rate",
                    "median_max_abs_dp",
                    "scorer_logit_pcc",
                    "hidden_encoder_pcc",
                    "hidden_head_pcc",
                    "nan_rows",
                )
            },
            "latency_sum_ms": chosen.get("latency_sum_ms"),
            "cells": chosen.get("cells"),
            "env": {
                "LAYA_PRECISION": selected,
                "LAYA_SEQ_BUCKETS": ",".join(str(x) for x in res["seq_buckets"]),
                "LAYA_ROW_BUCKETS": ",".join(str(x) for x in res["row_buckets"]),
            },
            "confirmation": conf,
        }
    )
    with open(sel_path, "w") as f:
        json.dump(sel, f, indent=1)
    with open(os.path.join(out_dir, "confirmation.md"), "w") as f:
        f.write(confirmation_md(conf))
    print(
        "CONFIRM_SELECTED",
        json.dumps({"selected": selected, "reason": why, "needed": needed, "candidate_pass": cand_ok}),
    )


def select(rows):
    passing = [r for r in rows if r["pass"] and r.get("latency_sum_ms") is not None]
    if not passing:
        return None, "no policy passes every gate"
    best = min(passing, key=lambda r: r["latency_sum_ms"])
    tied = [r for r in passing if r["latency_sum_ms"] <= best["latency_sum_ms"] * (1.0 + TIE_PCT / 100.0)]
    chosen = sorted(tied, key=lambda r: (-r["confident_agree_rate"], r["latency_sum_ms"]))[0]
    reason = (
        f"fastest passing policy by the sum of p50 over the cells {', '.join(CELLS)}: {best['policy']} at {best['latency_sum_ms']:.1f} ms; "
        f"policies within {TIE_PCT:g} percent: {[r['policy'] for r in tied]}; the tie goes to the higher confident agreement: {chosen['policy']}"
    )
    return chosen, reason


def pareto_png(rows, selected, path, ykey, ylabel, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=120)
    for r in rows:
        if r.get("latency_sum_ms") is None or r.get(ykey) is None:
            continue
        ok = r["pass"]
        sel = selected is not None and r["policy"] == selected["policy"]
        ax.scatter(
            r["latency_sum_ms"],
            r[ykey],
            s=160 if sel else 70,
            marker="*" if sel else ("o" if ok else "x"),
            color="#1f77b4" if ok else "#d62728",
            zorder=3,
        )
        ax.annotate(r["policy"], (r["latency_sum_ms"], r[ykey]), textcoords="offset points", xytext=(6, 5), fontsize=8)
    if ykey == "confident_agree_rate":
        ax.axhline(GATES["confident_agreement"], color="grey", linestyle="--", linewidth=1)
    if ykey == "median_max_abs_dp":
        ax.axhline(GATES["median_max_abs_dp"], color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("sum of p50 over the 1, 5, 10 and 50 question cells (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    ap.add_argument("--row-buckets", required=True)
    ap.add_argument("--seq-buckets", required=True)
    ap.add_argument("--hidden-cases", type=int, default=40)
    ap.add_argument("--hidden-cache", default="/home/hous/dev/laya/state/tt_cache/fidelity_hidden_ref_gate40.pt")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--only", choices=["all", "fidelity", "latency", "aggregate", "confirm"], default="all")
    ap.add_argument("--log", default=None)
    ap.add_argument("--out-dir", default=DOC_DIR)
    ap.add_argument("--candidate", default=None)
    ap.add_argument("--candidate-dir", default=None)
    ap.add_argument("--runner-up", default="bf8w_hifi3_erf")
    ap.add_argument("--runner-up-dir", default=None)
    ap.add_argument(
        "--reference-dir", default="/home/hous/dev/laya/evals/results/cpu_reference_cpu_b0_20261005T210448Z"
    )
    ap.add_argument(
        "--invariance",
        default=None,
        help="decision_agreement JSON of the candidate (stage 6 alone-versus-in-batch gate)",
    )
    a = ap.parse_args(argv)
    if a.only == "confirm":
        cmd_confirm(a)
        return
    os.makedirs(a.out_dir, exist_ok=True)
    log_path = a.log or os.path.join(a.out_dir, "sweep_subprocess.log")
    policies = [p for p in a.policies.split(",") if p.strip()]
    py = sys.executable
    started = datetime.now(timezone.utc).isoformat()
    for p in policies:
        fid_out = os.path.join(a.out_dir, f"fidelity_{p}.json")
        lat_out = os.path.join(a.out_dir, f"latency_{p}.json")
        if a.only in ("all", "fidelity") and not (a.skip_existing and os.path.exists(fid_out)):
            cmd = [
                py,
                os.path.join(HERE, "run_fidelity.py"),
                "--policy",
                p,
                "--items",
                "gate",
                "--hidden-cases",
                str(a.hidden_cases),
                "--seq-buckets",
                a.seq_buckets,
                "--row-buckets",
                a.row_buckets,
                "--threads",
                str(a.threads),
                "--out",
                fid_out,
            ]
            if a.hidden_cache:
                cmd += ["--hidden-cache", a.hidden_cache]
            run(cmd, log_path)
        if a.only in ("all", "latency") and not (a.skip_existing and os.path.exists(lat_out)):
            wait_quiet()
            cmd = [
                py,
                os.path.join(HERE, "bench_latency.py"),
                "--mode",
                "fresh",
                "--cells",
                ",".join(CELLS),
                "--policy",
                p,
                "--row-buckets",
                a.row_buckets,
                "--seq-buckets",
                a.seq_buckets,
                "--warm",
                str(a.warm),
                "--reps",
                str(a.reps),
                "--threads",
                str(a.threads),
                "--out",
                lat_out,
            ]
            run(cmd, log_path)
    rows = []
    for p in policies:
        fid_out = os.path.join(a.out_dir, f"fidelity_{p}.json")
        lat_out = os.path.join(a.out_dir, f"latency_{p}.json")
        row = {
            "policy": p,
            "fidelity_file": os.path.relpath(fid_out, AUTOPORT) if os.path.exists(fid_out) else None,
            "latency_file": os.path.relpath(lat_out, AUTOPORT) if os.path.exists(lat_out) else None,
        }
        if os.path.exists(fid_out):
            fid = json.load(open(fid_out))
            row.update(fidelity_row(fid))
            row["policy_describe"] = fid["policy"]
        else:
            row["pass"] = False
            row["error"] = "fidelity run missing"
        if os.path.exists(lat_out):
            row.update(latency_row(json.load(open(lat_out))))
        else:
            row["latency_sum_ms"] = None
            row["error"] = (row.get("error") or "") + " latency run missing"
        rows.append(row)
    selected, reason = select(rows)
    for r in rows:
        if r.get("pass"):
            r["thin_margins"] = thin_margins(r)
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "started": started,
        "host": socket.gethostname(),
        "row_buckets": [int(x) for x in a.row_buckets.split(",")],
        "seq_buckets": [int(x) for x in a.seq_buckets.split(",")],
        "gates": GATES,
        "selection_rule": f"fastest passing policy by the sum of p50 over the cells {CELLS}; ties within {TIE_PCT:g} percent go to the higher confident agreement",
        "selected": selected["policy"] if selected else None,
        "selection_reason": reason,
        "rows": rows,
    }
    with open(os.path.join(a.out_dir, "sweep_results.json"), "w") as f:
        json.dump(result, f, indent=1)
    cols = [
        "policy",
        "pass",
        "confident_agree",
        "confident_n",
        "confident_agree_rate",
        "plain_argmax_agree_rate",
        "act_argmax_agree_rate",
        "median_max_abs_dp",
        "p95_max_abs_dp",
        "max_abs_dp",
        "scorer_logit_pcc",
        "hidden_encoder_pcc",
        "hidden_head_pcc",
        "nan_rows",
        "lat_1_ms",
        "lat_5_ms",
        "lat_10_ms",
        "lat_50_ms",
        "latency_sum_ms",
        "loadavg_fidelity",
        "loadavg_latency_cells",
    ]
    with open(os.path.join(a.out_dir, "sweep_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            cells = r.get("cells") or {}
            w.writerow(
                [
                    r["policy"],
                    r.get("pass"),
                    r.get("confident_agree"),
                    r.get("confident_n"),
                    r.get("confident_agree_rate"),
                    r.get("plain_argmax_agree_rate"),
                    r.get("act_argmax_agree_rate"),
                    r.get("median_max_abs_dp"),
                    r.get("p95_max_abs_dp"),
                    r.get("max_abs_dp"),
                    r.get("scorer_logit_pcc"),
                    r.get("hidden_encoder_pcc"),
                    r.get("hidden_head_pcc"),
                    r.get("nan_rows"),
                ]
                + [cells.get(n, {}).get("end_to_end_ms_p50") for n in CELLS]
                + [
                    r.get("latency_sum_ms"),
                    r.get("loadavg"),
                    [
                        cells.get(n, {}).get("loadavg", [None])[0] if cells.get(n, {}).get("loadavg") else None
                        for n in CELLS
                    ],
                ]
            )
    if selected:
        from models.autoports.convaiinnovations_laya.tt import model_config as mc

        sel_cfg = {
            "selected_policy": selected["policy"],
            "shipped_default_before_sweep": "bf8w_hifi3_erf",
            "default_changed": selected["policy"] != "bf8w_hifi3_erf",
            "current_default_policy_name": mc.DEFAULT_POLICY_NAME,
            "rule": result["selection_rule"],
            "reason": reason,
            "policy": selected.get("policy_describe"),
            "gates": {
                k: selected.get(k)
                for k in (
                    "confident_agree",
                    "confident_n",
                    "confident_agree_rate",
                    "median_max_abs_dp",
                    "scorer_logit_pcc",
                    "hidden_encoder_pcc",
                    "hidden_head_pcc",
                    "nan_rows",
                )
            },
            "latency_sum_ms": selected["latency_sum_ms"],
            "cells": selected["cells"],
            "env": {
                "LAYA_PRECISION": selected["policy"],
                "LAYA_SEQ_BUCKETS": a.seq_buckets,
                "LAYA_ROW_BUCKETS": a.row_buckets,
            },
            "named_profile_candidates": {
                "bf8w_hifi3_erf": "the stage 3 to 7 shipped policy, kept as a named profile candidate"
            },
        }
        with open(os.path.join(a.out_dir, "selected_precision_config.json"), "w") as f:
            json.dump(sel_cfg, f, indent=1)
    try:
        pareto_png(
            rows,
            selected,
            os.path.join(a.out_dir, "pareto_latency_vs_confident_agreement.png"),
            "confident_agree_rate",
            "confident argmax agreement (fraction of 149)",
            "Laya on p150: latency against confident agreement",
        )
        pareto_png(
            rows,
            selected,
            os.path.join(a.out_dir, "pareto_latency_vs_median_dp.png"),
            "median_max_abs_dp",
            "median over decisions of max |dp|",
            "Laya on p150: latency against median max |dp|",
        )
    except Exception as exc:
        print("SWEEP pareto failed:", exc)
    if selected:
        print("SWEEP_THIN_MARGINS", json.dumps({k: v["thin"] for k, v in thin_margins(selected).items()}))
    print("SWEEP_SELECTED", json.dumps({"selected": result["selected"], "reason": reason}))
    print(
        "SWEEP_TABLE",
        json.dumps(
            [
                {
                    k: r.get(k)
                    for k in (
                        "policy",
                        "pass",
                        "confident_agree_rate",
                        "median_max_abs_dp",
                        "scorer_logit_pcc",
                        "hidden_encoder_pcc",
                        "latency_sum_ms",
                    )
                }
                for r in rows
            ]
        ),
    )
    print("SWEEP_DONE", os.path.join(a.out_dir, "sweep_results.json"))


if __name__ == "__main__":
    main()
