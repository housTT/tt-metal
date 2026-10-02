import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

KEV_ROOT = Path("/home/hous/dev/kev")
SWEEP = KEV_ROOT / "reports" / "sweep"
CSV_PATH = SWEEP / "sweep_results.csv"
DOC = KEV_ROOT / "tt-metal" / "models" / "autoports" / "jaredpalmer_kev_9b" / "doc" / "datatype_sweep"
SELECTED_PATH = DOC / "selected_precision_config.json"
POLICY_PROBE = DOC.parent / "optimized" / "perf_probe_traced_bfp8_policy.json"
DEFAULT_ENGINE_LOG = KEV_ROOT / "logs" / "stage4r_reference_records.log"
PROBE_GLOB = "perf_probe_{variant}.json"
CPU_BF16_MAX_DP = 0.0189
FLIP_MARGIN = 0.05
RULE_AMENDMENT = (
    "Rule amendment from the orchestrator, record it verbatim in the work log and summary: an argmax flip counts against a variant "
    "only when the fp32 reference's top-2 probability margin on that question is at least 0.05. A flip on a near-tie row (margin under "
    "0.05, like the 0.2895 vs 0.3211 case) is reported but not disqualifying, because it measures the tie, not the variant. Keep max |dp| "
    "and mean |dp| as tie-breakers, and keep the 1.0 pp subset-accuracy window. Also report, for every variant, the count of "
    '"margin >= 0.05" flips and the count of near-tie flips separately. Continue the sweep.'
)
TRACED_COLS = [
    "traced_tail128_ms",
    "traced_tail256_ms",
    "traced_tail2048_ms",
    "traced_state2048_ms",
    "traced_build_s",
    "trace_mib",
]
EXTRA_COLS = TRACED_COLS + ["cache_gb"]

GROUPS = {
    "mlp.gate_proj": "gate_up_dtype",
    "mlp.up_proj": "gate_up_dtype",
    "mlp.down_proj": "down_dtype",
    "self_attn.q_proj": "proj_dtype",
    "self_attn.k_proj": "proj_dtype",
    "self_attn.v_proj": "proj_dtype",
    "self_attn.o_proj": "proj_dtype",
    "linear_attn.qkv_proj": "proj_dtype",
    "linear_attn.in_proj_a": "proj_dtype",
    "linear_attn.in_proj_b": "proj_dtype",
    "linear_attn.in_proj_z": "proj_dtype",
    "linear_attn.out_proj": "proj_dtype",
}
FIXED_DTYPES = {
    "tok_embeddings": "bf16",
    "norms (input, post_attention, final, q_norm, k_norm, linear_attn.norm)": "bf16",
    "linear_attn.A_log, dt_bias": "bf16",
    "linear_attn conv weights": "bf16 (host, row major)",
    "output (lm_head)": "bfp8 (loaded, unused by the kev pointer head)",
}
FIXED_FIDELITIES = {
    "sdpa (attention prefill and decode)": "HiFi2, fp32 acc (models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py:157-159)",
    "gdn chunk kernel (gated_delta_attn_seq)": "float32 kernel; preprocessing matmuls HiFi4 fp32 acc (ttnn_delta_rule_seq.py:248-251, 414-417)",
    "engine row select": "HiFi4, fp32 acc (models/autoports/jaredpalmer_kev_9b/tt/engine.py select_kernel_config)",
}


def dir_bytes(path):
    total = 0
    for root, _, files in os.walk(path, followlinks=True):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def probe(variant, path=None):
    path = SWEEP / PROBE_GLOB.format(variant=variant) if path is None else Path(path)
    if not path.exists():
        return {}
    d = json.load(open(path))
    tails = d.get("tail_ms", {})
    states = d.get("state_ms", {})
    out = {
        "traced_tail128_ms": tails.get("Q50", {}).get("total"),
        "traced_tail256_ms": tails.get("Q200", {}).get("total"),
        "traced_tail2048_ms": tails.get("Q2000", {}).get("total"),
        "traced_state2048_ms": states.get("S2048", {}).get("prefill_state"),
        "traced_build_s": d.get("build_s"),
        "trace_mib": round(d["trace_bytes_allocated_per_bank"] * d["trace_banks"] / 2**20, 1)
        if d.get("trace_bytes_allocated_per_bank")
        else None,
        "probe_json": str(path),
        "probe_mode": d.get("mode"),
        "probe_matmul_policy": d.get("matmul_policy"),
        "probe_card_ms": {
            k: {kk: vv for kk, vv in v.items() if kk != "probs"} for k, v in d.get("card_ms", {}).items()
        },
    }
    return out


def classify_flips(variant_json):
    margin, near, detail = 0, 0, []
    for row in variant_json.get("reference_rows", []):
        if row.get("argmax_agree", True):
            continue
        ref = sorted(row["probs_ref"], reverse=True)
        m = ref[0] - ref[1] if len(ref) > 1 else ref[0]
        kind = "margin" if m >= FLIP_MARGIN else "near_tie"
        if kind == "margin":
            margin += 1
        else:
            near += 1
        detail.append(
            {
                "row_key": row["row_key"],
                "ref_top2_margin": round(m, 4),
                "kind": kind,
                "probs_ref": [round(x, 4) for x in row["probs_ref"]],
                "probs_tt": [round(x, 4) for x in row["probs_tt"]],
            }
        )
    return margin, near, detail


def traced_block(p):
    policy = p.get("probe_matmul_policy")
    label = "on" if policy is True else "off" if policy is False else "unknown (no probe json)"
    flag = " --matmul-policy" if policy is True else ""
    return {
        "regime": f"stage 3 engine, traced=True, matmul policy {label} (matmul_policy field of probe_json), scripts/perf_probe.py --traced{flag}, median of 5 with device sync, 2048-token state in slot 0",
        "tail_bucket128_ms": num(p.get("traced_tail128_ms")),
        "tail_bucket256_ms": num(p.get("traced_tail256_ms")),
        "tail_bucket2048_ms": num(p.get("traced_tail2048_ms")),
        "state_2048_ms": num(p.get("traced_state2048_ms")),
        "build_s": num(p.get("traced_build_s")),
        "trace_mib": num(p.get("trace_mib")),
        "card_ms": p.get("probe_card_ms", {}),
        "probe_json": p.get("probe_json"),
    }


def default_engine_propagation(sweep_prop):
    if not DEFAULT_ENGINE_LOG.exists():
        return None
    text = DEFAULT_ENGINE_LOG.read_text(errors="replace")
    m = re.search(r"KevEngine args=\S+ .*? cache=(\S+) max_len=\d+ traced=(\w+) matmul_policy=(\w+)", text)
    if m is None:
        return None
    files = re.findall(
        r"Loaded cache for \S+/(mlp\.\w+|self_attn\.\w+|linear_attn\.\w+)\.weight_dtype_(\w+)_layout", text
    )
    out = {
        "source": f"default KevEngine (no QWEN36_* or KEV_* overrides): KevEngine args= line and the per-tensor 'Loaded cache' lines of {DEFAULT_ENGINE_LOG}; compute_kernel_config and kv_cache fields read back from the built engine by scripts/dtype_sweep.py under the same knob values (propagation_check_sweep_process)",
    }
    out.update(
        {k: v for k, v in sweep_prop.items() if "compute_kernel_config" in k or k in ("kv_cache.dtype", "n_layers")}
    )
    out["weight_cache_path"] = m.group(1)
    out["traced"] = m.group(2) == "True"
    out["matmul_policy"] = m.group(3) == "True"
    out["cached_weight_file_dtypes"] = {
        name: sorted({d for n, d in files if n == name}) for name in sorted({n for n, _ in files})
    }
    return out


def enrich(r):
    r = dict(r)
    for k, v in probe(r["variant"]).items():
        r[k] = "" if v is None else v
    vj = json.load(open(r["json"])) if r.get("json") and Path(r["json"]).exists() else {}
    if vj.get("reference_rows"):
        r["flips_margin"], r["flips_neartie"], r["flip_detail"] = classify_flips(vj)
    else:
        r["flips_margin"], r["flips_neartie"], r["flip_detail"] = "", "", []
    cache = r.get("cache_dir", "")
    if r.get("json") and Path(r["json"]).exists():
        cache = json.load(open(r["json"])).get("propagation", {}).get("weight_cache_path", cache)
    r["cache_dir_measured"] = cache
    r["cache_gb"] = round(dir_bytes(cache) / 1e9, 2) if cache and os.path.isdir(cache) else ""
    return r


def read_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    latest = {}
    for r in rows:
        latest[r["variant"]] = r
    return [enrich(r) for r in latest.values()]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def complete(r):
    return (
        r["status"] == "ok"
        and num(r["acc"]) is not None
        and num(r["warm_row_s"]) is not None
        and not r["variant"].endswith("_policy_on")
    )


TIME_NOISE = 0.005


def pareto(rows):
    front = set()
    for r in rows:
        dominated = any(
            num(o["warm_row_s"]) <= num(r["warm_row_s"])
            and num(o["acc"]) >= num(r["acc"])
            and (num(o["warm_row_s"]) < num(r["warm_row_s"]) * (1 - TIME_NOISE) or num(o["acc"]) > num(r["acc"]))
            for o in rows
            if o is not r
        )
        if not dominated:
            front.add(r["variant"])
    return front


def select(rows, acc_window_pp, tie_pct):
    best_acc = max(num(r["acc"]) for r in rows)
    eligible = [
        r for r in rows if num(r["acc"]) >= best_acc - acc_window_pp / 100 and int(float(r["flips_margin"])) == 0
    ]
    if not eligible:
        return None, best_acc, eligible, []
    fastest = min(num(r["warm_row_s"]) for r in eligible)
    ties = [r for r in eligible if num(r["warm_row_s"]) <= fastest * (1 + tie_pct / 100)]
    chosen = min(ties, key=lambda r: (num(r["max_dp"]), num(r["mean_dp"]), num(r["warm_row_s"])))
    return chosen, best_acc, eligible, ties


def table(rows, front, chosen):
    cols = [
        "variant",
        "gate_up_dtype",
        "down_dtype",
        "proj_dtype",
        "fidelity",
        "gdn_fp32_state",
        "matmul_policy",
        "warm_row_s",
        "warm_ms_per_token",
        "acc",
        "brier",
        "ece",
        "argmax_flips",
        "flips_margin",
        "flips_neartie",
        "max_dp",
        "mean_dp",
        "min_pcc",
        "traced_tail128_ms",
        "traced_state2048_ms",
        "cache_gb",
        "status",
    ]
    lines = ["| " + " | ".join(cols + ["pareto", "selected"]) + " |", "|" + "---|" * (len(cols) + 2)]
    for r in sorted(rows, key=lambda r: (num(r["warm_row_s"]) is None, num(r["warm_row_s"]) or 0)):
        vals = []
        for c in cols:
            v = r.get(c, "")
            f = num(v)
            if c in ("warm_row_s",) and f is not None:
                v = f"{f:.3f}"
            elif c in ("acc", "brier", "ece") and f is not None:
                v = f"{f:.4f}"
            elif c in ("max_dp", "mean_dp", "min_pcc") and f is not None:
                v = f"{f:.4f}"
            elif c == "warm_ms_per_token" and f is not None:
                v = f"{f:.3f}"
            elif c in ("traced_tail128_ms", "traced_state2048_ms") and f is not None:
                v = f"{f:.1f}"
            vals.append(str(v))
        vals.append("yes" if r["variant"] in front else "")
        vals.append("yes" if chosen is not None and r["variant"] == chosen["variant"] else "")
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def selected_config(r, all_rows, best_acc, eligible, ties, acc_window_pp, tie_pct):
    variant_json = json.load(open(r["json"])) if Path(r["json"]).exists() else {}
    weights = {g: r[col] for g, col in GROUPS.items()}
    weights.update(FIXED_DTYPES)
    fidelities = {
        "mlp (gate, up, down matmuls)": f"{r['fidelity']}, fp32 acc",
        "self_attn projections (q, k, v, o)": f"{r['fidelity']}, fp32 acc",
        "linear_attn projections (qkv, a, b, z, out)": f"{r['fidelity']}, fp32 acc",
        **FIXED_FIDELITIES,
    }
    return {
        "config_id": r["variant"],
        "model_id": "jaredpalmer/kev-9b",
        "base_model_id": "Qwen/Qwen3.5-9B-Base",
        "hardware": "Blackhole P150, single device (mesh 1x1)",
        "selected_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "weight_dtypes": weights,
        "layer_exceptions": [],
        "compute_fidelities": fidelities,
        "activation_dtype": "bf16",
        "residual_dtype": "bf16",
        "ccl_dtype": "n/a (single device, no CCL)",
        "kv_cache_dtype": "bf16 (KevEngine.allocate_kv_caches; QWEN_SDPA_BF8=0)",
        "gdn_recurrent_state_dtype": "fp32" if r["gdn_fp32_state"] == "1" else "bf16",
        "logits_dtype": "device hidden rows read back as fp32 (bf16 device tensor); pointer head and softmax on host in fp32 at the head temperature",
        "sampling_dtype": "n/a (no token sampling; option probabilities only)",
        "runtime_flags": {
            "QWEN36_MLP_GATE_UP_DTYPE": r["gate_up_dtype"],
            "QWEN36_MLP_DOWN_DTYPE": r["down_dtype"],
            "QWEN36_PROJ_DTYPE": r["proj_dtype"],
            "QWEN36_MATMUL_FIDELITY": r["fidelity"],
            "QWEN_GDN_FP32_STATE": r["gdn_fp32_state"],
            "QWEN_SDPA_BF8": "0",
            "KEV_MATMUL_POLICY": "1" if str(r.get("matmul_policy")) == "True" else "0",
            "TT_CACHE_PATH": r["cache_dir"],
        },
        "propagation_check": default_engine_propagation(variant_json.get("propagation", {}))
        or variant_json.get("propagation", {}),
        "propagation_check_sweep_process": variant_json.get("propagation", {}),
        "accuracy": {
            "subset_rows": int(float(r["subset_rows"])),
            "acc": num(r["acc"]),
            "brier": num(r["brier"]),
            "ece": num(r["ece"]),
            "nll": num(r["nll"]),
            "reference_rows": int(float(r["ref_rows"])),
            "argmax_flips": int(float(r["argmax_flips"])),
            "flips_margin_ge_0_05": int(float(r["flips_margin"])),
            "flips_near_tie": int(float(r["flips_neartie"])),
            "flip_detail": r.get("flip_detail", []),
            "max_dp": num(r["max_dp"]),
            "mean_dp": num(r["mean_dp"]),
            "min_pcc": num(r["min_pcc"]),
            "median_pcc": num(r["median_pcc"]),
        },
        "performance": {
            "regime": "eager full-row prefill_hidden, wall time per row including readback, program cache warm (bucket seen before)",
            "warm_row_s": num(r["warm_row_s"]),
            "warm_ms_per_token": num(r["warm_ms_per_token"]),
            "mean_row_s_all": num(r["mean_row_s"]),
            "ref_mean_row_s": num(r["ref_mean_row_s"]),
            "engine_load_s": num(r["engine_load_s"]),
        },
        "performance_traced": traced_block(r),
        "performance_traced_policy_on": traced_block(probe(r["variant"], POLICY_PROBE))
        if POLICY_PROBE.exists()
        else None,
        "weight_cache_gb": num(r.get("cache_gb")),
        "accuracy_by_suite": variant_json.get("subset", {}).get("by_suite", {}),
        "cpu_bf16_reference_max_dp": CPU_BF16_MAX_DP,
        "selection_rule": {
            "accuracy_window_pp": acc_window_pp,
            "best_acc": best_acc,
            "argmax_flips_required": f"0 flips on reference rows whose fp32 top-2 margin is >= {FLIP_MARGIN}; near-tie flips (margin < {FLIP_MARGIN}) are reported, not disqualifying",
            "flip_margin_threshold": FLIP_MARGIN,
            "rule_amendment_verbatim": RULE_AMENDMENT,
            "time_tie_pct": tie_pct,
            "tie_break": "lowest max_dp on the 29 reference rows, then lowest mean_dp, then time",
            "eligible": [e["variant"] for e in eligible],
            "ties": [t["variant"] for t in ties],
            "evaluated": [x["variant"] for x in all_rows],
        },
        "evidence": {
            "sweep_results_csv": str(CSV_PATH),
            "variant_json": r["json"],
            "timestamp": r["timestamp"],
        },
    }


INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SERIES = "#2a78d6"
SELECTED = "#d03b3b"


def plot(rows, front, chosen, xkey, xlabel, out, vline, vline_label, title, subtitle):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [(num(r[xkey]), 1.0 / num(r["warm_row_s"]), r) for r in rows]
    xs = [p[0] for p in pts] + ([vline] if vline is not None else [])
    ys = [p[1] for p in pts]
    xr = (max(xs) - min(xs)) or 1.0
    yr = (max(ys) - min(ys)) or 1.0
    groups = []
    for x, y, r in pts:
        for g in groups:
            if abs(g["x"] - x) < 0.002 * xr and abs(g["y"] - y) < 0.002 * yr:
                g["rows"].append(r)
                break
        else:
            groups.append({"x": x, "y": y, "rows": [r]})
    fig, ax = plt.subplots(figsize=(9, 5.8), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
        ax.spines[side].set_linewidth(1)
    ax.grid(True, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.margins(x=0.12, y=0.18)
    fr = sorted([p for p in pts if p[2]["variant"] in front], key=lambda p: p[0])
    if len(fr) > 1:
        ax.plot([p[0] for p in fr], [p[1] for p in fr], color="#c3c2b7", linewidth=2, solid_joinstyle="round", zorder=1)
    sel_name = chosen["variant"] if chosen is not None else None
    for g in groups:
        g["sel"] = any(r["variant"] == sel_name for r in g["rows"])
    for g in sorted(groups, key=lambda g: g["sel"]):
        ax.scatter(
            [g["x"]],
            [g["y"]],
            s=120 if g["sel"] else 80,
            color=SELECTED if g["sel"] else SERIES,
            edgecolors=SURFACE,
            linewidths=2,
            zorder=5 if g["sel"] else 3,
        )
        names = " + ".join(r["variant"] for r in g["rows"])
        tails = [num(r.get("traced_tail128_ms")) for r in g["rows"]]
        tails = [t for t in tails if t is not None]
        extra = f"\ntraced tail 128: {tails[0]:.0f} ms" if tails else ""
        below = any(
            o is not g and abs(o["x"] - g["x"]) < 0.12 * xr and 0 <= o["y"] - g["y"] < 0.08 * yr for o in groups
        )
        xytext = (8, -24 if below else 6)
        ax.annotate(
            f"{names}{extra}",
            (g["x"], g["y"]),
            textcoords="offset points",
            xytext=xytext,
            fontsize=8.5,
            color=INK if g["sel"] else "#52514e",
            va="bottom" if not below else "top",
            zorder=6,
        )
    if vline is not None:
        ax.axvline(vline, color=MUTED, linewidth=1, linestyle=(0, (1, 3)), zorder=2)
        ax.annotate(
            vline_label,
            (vline, 0),
            xycoords=("data", "axes fraction"),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            color=MUTED,
            ha="left",
            va="bottom",
            rotation=90,
        )
    ax.set_xlabel(xlabel, color="#52514e", fontsize=9.5)
    ax.set_ylabel("eager full-row prefill, rows / s (higher is better)", color="#52514e", fontsize=9.5)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=22)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom", wrap=True)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def plots(done, front, chosen, best_acc, acc_window_pp):
    sel = chosen["variant"] if chosen is not None else "none"
    plot(
        done,
        front,
        chosen,
        "acc",
        "option argmax accuracy on the 200-record development subset (291 rows)",
        DOC / "top1_perf_pareto.png",
        best_acc - acc_window_pp / 100,
        f"best - {acc_window_pp:g} pp",
        f"kev-9b precision sweep: accuracy versus prefill speed (selected: {sel})",
        "x: served-temperature argmax accuracy. y: eager warm full-row rows/s (policy off). Gray: non-dominated front. Red: selected.",
    )
    for r in done:
        r["_agree"] = 1.0 - num(r["max_dp"])
    plot(
        done,
        front,
        chosen,
        "_agree",
        "agreement with the fp32 CPU reference on 29 rows: 1 - max |dp| (higher is better)",
        DOC / "fp32_agreement_perf_pareto.png",
        1.0 - CPU_BF16_MAX_DP,
        f"CPU bf16 reference (max |dp| {CPU_BF16_MAX_DP})",
        f"kev-9b precision sweep: fp32 agreement versus prefill speed (selected: {sel})",
        "x: 1 - max |dp| over the 29 reference rows. y: eager warm full-row rows/s (policy off). Dotted: CPU bf16 model. Red: selected.",
    )
    for r in done:
        r.pop("_agree", None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(CSV_PATH))
    p.add_argument("--acc-window-pp", type=float, default=1.0)
    p.add_argument("--tie-pct", type=float, default=2.0)
    p.add_argument("--write", action="store_true")
    a = p.parse_args()
    rows = read_rows(a.csv)
    done = [r for r in rows if complete(r)]
    front = pareto(done) if done else set()
    chosen, best_acc, eligible, ties = select(done, a.acc_window_pp, a.tie_pct) if done else (None, None, [], [])
    print(table(rows, front, chosen))
    print()
    print(
        f"complete variants: {[r['variant'] for r in done]}; incomplete or rows-only: {[r['variant'] for r in rows if not complete(r)]}"
    )
    if best_acc is not None:
        print(
            f"best acc {best_acc:.4f}; eligible (acc >= best - {a.acc_window_pp} pp and 0 flips with fp32 margin >= {FLIP_MARGIN}): {[e['variant'] for e in eligible]}; time ties within {a.tie_pct}%: {[t['variant'] for t in ties]}"
        )
        for r in done:
            if r.get("flip_detail"):
                print(f"flips {r['variant']}: {r['flip_detail']}")
    if chosen is None:
        print("no eligible variant; nothing selected")
        return
    print(f"selected: {chosen['variant']}")
    if a.write:
        DOC.mkdir(parents=True, exist_ok=True)
        cfg = selected_config(chosen, done, best_acc, eligible, ties, a.acc_window_pp, a.tie_pct)
        json.dump(cfg, open(SELECTED_PATH, "w"), indent=1)
        fields = list(rows[0].keys()) if rows else []
        with open(DOC / "sweep_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[k for k in fields if k != "probe_card_ms"], extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(
                    {k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items() if k != "probe_card_ms"}
                )
        json.dump(
            {"rows": rows, "pareto_front": sorted(front), "selected": chosen["variant"]},
            open(DOC / "sweep_results.json", "w"),
            indent=1,
        )
        (DOC / "pareto_table.md").write_text(table(rows, front, chosen) + "\n")
        try:
            plots(done, front, chosen, best_acc, a.acc_window_pp)
            plotted = f", {DOC / 'top1_perf_pareto.png'}, {DOC / 'fp32_agreement_perf_pareto.png'}"
        except ImportError as e:
            plotted = f"; plots skipped ({e})"
        print(
            f"wrote {SELECTED_PATH}, {DOC / 'sweep_results.csv'}, {DOC / 'sweep_results.json'}, {DOC / 'pareto_table.md'}{plotted}"
        )


if __name__ == "__main__":
    main()
