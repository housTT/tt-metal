"""Build sweep_results.json, sweep_results.csv and the two Pareto charts from the per-candidate readiness sidecars.

Usage: collect_results.py <selected config id> [--bar-top1 0.90 --bar-top5 0.98]
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path("models/autoports/qwen_qwen3_8b/doc/datatype_sweep")
M = "models/autoports/qwen_qwen3_8b"
REF = "bringup/references/qwen3_8b_aime24_chat.refpt"

parser = argparse.ArgumentParser()
parser.add_argument("selected")
parser.add_argument("--bar-top1", type=float, default=0.90)
parser.add_argument("--bar-top5", type=float, default=0.98)
parser.add_argument("--bar-top100", type=float, default=1.0)
args = parser.parse_args()

HARDWARE = "one Blackhole P300 chip (TT_VISIBLE_DEVICES=2), 1x1 mesh, p150 mesh graph descriptor"
MESH = "N150 (1x1); stages 4 and 5 skipped by the operator, single chip is the target topology"


def summarize(cfg: dict) -> dict:
    groups = cfg["weight_groups"]
    exc = cfg.get("layer_exceptions", {})
    return {
        "weights": {g: f"{s['dtype']}/{s['fidelity']}" for g, s in groups.items()},
        "layer_exceptions": {
            layer: {
                g: f"{s.get('dtype', groups[g]['dtype'])}/{s.get('fidelity', groups[g]['fidelity'])}"
                for g, s in spec.items()
            }
            for layer, spec in exc.items()
        },
        "kv_cache_dtype": cfg["kv_cache_dtype"],
        "activation_dtype": cfg.get("activation_dtype", "bf16"),
        "ccl_dtype": cfg.get("ccl_dtype"),
        "lm_head": f"{cfg['lm_head']['dtype']}/{cfg['lm_head']['decode_fidelity']} (prefill {cfg['lm_head']['prefill_fidelity']})",
        "norm_fidelity": cfg.get("norm", {}).get("fidelity", "hifi4"),
        "sdpa_fidelity": cfg.get("sdpa", {}).get("fidelity", "hifi4"),
        "logits": cfg.get("logits", {}),
        "sampling": cfg.get("sampling", {}).get("dtype", "bf16"),
    }


rows = []
for cfg_file in sorted((D / "configs").glob("c*.json")):
    sidecar = D / f"readiness_{cfg_file.stem}_chip2.json"
    tf_only = False
    if not sidecar.exists():
        sidecar = D / f"readiness_{cfg_file.stem}_serial_chip2.json"
        tf_only = True
        if not sidecar.exists():
            continue
    data = json.loads(sidecar.read_text())
    runs = data["runs"]
    pc = runs.get("run_prefill_check", {})
    tf = runs.get("run_teacher_forcing", {})
    cid = tf.get("config_id") or pc.get("config_id") or sidecar.stem.replace("readiness_", "").replace("_chip2", "")
    cfg_path = D / "configs" / f"{cid}.json"
    cfg = json.loads(cfg_path.read_text())
    serial = D / f"readiness_{cid}_serial_chip2.json"
    tf_source = tf.get("log")
    if serial.exists():
        serial_tf = json.loads(serial.read_text())["runs"].get("run_teacher_forcing", {})
        if serial_tf.get("exit") == 0:
            tf = {
                **tf,
                **serial_tf,
                "round2_concurrent_row": {
                    k: tf.get(k)
                    for k in (
                        "top1",
                        "top5",
                        "top100",
                        "ttft_ms",
                        "decode_tps_per_user",
                        "e2e_tps_per_user",
                        "timing_regime",
                        "log",
                    )
                },
            }
            tf_source = serial_tf.get("log")
    counters = tf.get("runtime_counters", {})
    traced = (
        bool(counters) and counters.get("eager_decode_steps", 1) == 0 and counters.get("decode_trace_replays", 0) >= 99
    )
    ran = pc.get("exit") == 0 and tf.get("exit") == 0
    robustness = {}
    for tag in ("aime_p1", "book32"):
        for extra in sorted(D.glob(f"readiness_{cid}_{tag}_chip*.json")):
            extra_runs = json.loads(extra.read_text())["runs"]
            epc, etf = extra_runs.get("run_prefill_check", {}), extra_runs.get("run_teacher_forcing", {})
            robustness[tag] = {
                "reference": etf.get("reference") or epc.get("reference"),
                "chip": etf.get("chip"),
                "prefill_check": {k: epc.get(k) for k in ("top1", "top5", "top100", "exit", "log")},
                "teacher_forcing": {
                    k: etf.get(k) for k in ("top1", "top5", "top100", "exit", "log", "runtime_counters")
                },
                "timing_regime": etf.get("timing_regime"),
                "pass": epc.get("exit") == 0
                and etf.get("exit") == 0
                and all(v is not None for v in (epc.get("top1"), etf.get("top1")))
                and min(epc.get("top1", 0), etf.get("top1", 0)) >= args.bar_top1
                and min(epc.get("top5", 0), etf.get("top5", 0)) >= args.bar_top5
                and min(epc.get("top100", 0), etf.get("top100", 0)) >= args.bar_top100,
            }
    for extra in sorted(D.glob(f"natural_text_{cid}_chip*.json")):
        nt = json.loads(extra.read_text())
        robustness["natural_text"] = {
            "reference": f"tale-of-two-cities tokens [{nt['positions'][0]}, {nt['positions'][1]}): TT prefill logits (prefill path, LM head LoFi fp32 acc) against HF fp32 logits at {nt['num_positions']} real next-token positions",
            "chip": int(nt["visible_devices"]) if nt.get("visible_devices") else None,
            "top1": nt["top1_agreement_with_hf"],
            "top5": nt["top5_agreement_with_hf"],
            "pcc_mean": nt["pcc_mean"],
            "pcc_min": nt["pcc_min"],
            "artifact": str(extra),
            "pass": nt["top1_agreement_with_hf"] >= args.bar_top1 and nt["top5_agreement_with_hf"] >= args.bar_top5,
        }
    top1 = tf.get("top1")
    top5 = tf.get("top5")
    top100 = tf.get("top100")
    passed = (
        ran
        and traced
        and top1 is not None
        and top1 >= args.bar_top1
        and top5 >= args.bar_top5
        and top100 >= args.bar_top100
        and pc.get("top1", 0) >= args.bar_top1
        and pc.get("top5", 0) >= args.bar_top5
    )
    main_pass = passed
    passed = passed and all(r["pass"] for r in robustness.values())
    rows.append(
        {
            "config_id": cid,
            "description": cfg.get("description"),
            "precision_config_path": str(cfg_path),
            "policy": summarize(cfg),
            "compute_fidelity": {g: s["fidelity"] for g, s in cfg["weight_groups"].items()}
            | {
                "lm_head_decode": cfg["lm_head"]["decode_fidelity"],
                "norm": cfg.get("norm", {}).get("fidelity", "hifi4"),
                "sdpa": cfg.get("sdpa", {}).get("fidelity", "hifi4"),
            },
            "prefill_check": {
                "top1": pc.get("top1"),
                "top5": pc.get("top5"),
                "top100": pc.get("top100"),
                "exit": pc.get("exit"),
                "log": pc.get("log"),
            },
            "teacher_forcing": {
                "top1": top1,
                "top5": top5,
                "top100": top100,
                "ttft_ms": tf.get("ttft_ms"),
                "decode_t_s_u": tf.get("decode_tps_per_user"),
                "e2e_t_s_u": tf.get("e2e_tps_per_user"),
                "exit": tf.get("exit"),
                "log": tf_source,
                "timing_regime": tf.get("timing_regime"),
                "round2_concurrent_row": tf.get("round2_concurrent_row"),
                "runtime_counters": counters,
                "traced": traced,
            },
            "measurement_regime": "readiness run_teacher_forcing: AIME24 chat-template prompt (158 tokens), 100 forced tokens, batch 1, traced decode + traced split sampler (runtime counters: decode_trace_replays 99, eager_decode_steps 0); TTFT = prefill plus first sampled token readback; decode t/s/u from host wall time between token callbacks",
            "command": f"QWEN3_PRECISION_CONFIG={cfg_path} python -m readiness_check.run_prefill_check --model-dir {M} --reference {REF} --mesh-device N150; same for run_teacher_forcing (logs/sweep_chain_chip2.sh)",
            "hardware": HARDWARE,
            "mesh": MESH,
            "chip": 2,
            "checkout_git_sha": data.get("checkout_git_sha"),
            "checkout_dirty_tt": data.get("checkout_dirty"),
            "sidecar": str(sidecar),
            "robustness_references": robustness,
            "main_reference_pass": main_pass,
            "status": (
                "timing-only (teacher forcing only, no prefill check; a reader-geometry variant of c9)"
                if tf_only
                else ("pass" if passed else ("fail" if ran else "error"))
            ),
            "selected": cid == args.selected,
        }
    )

rows.sort(key=lambda r: (r["teacher_forcing"]["decode_t_s_u"] or 0), reverse=True)
result = {
    "model": "Qwen/Qwen3-8B",
    "reference": REF,
    "reference_note": "AIME24 prompt 0, chat template, 100 generated tokens, top-100 captured (readiness_check.generate)",
    "acceptance": {
        "top1_min": args.bar_top1,
        "top5_min": args.bar_top5,
        "top100_min": args.bar_top100,
        "requires_traced_decode": True,
        "rule": "pass requires the bar on the main AIME24 reference (prefill check and teacher forcing) and on every robustness reference the candidate was evaluated on (second AIME24 prompt, 32-token short-prefix book prompt, 1468 natural-text positions of prefill logits against HF fp32 logits); every candidate faster than the selected one was run on every robustness reference",
    },
    "ranking_metric": "trace-verified teacher-forcing decode t/s/u (batch 1)",
    "selected_config_id": args.selected,
    "candidates": rows,
}
(D / "sweep_results.json").write_text(json.dumps(result, indent=2) + "\n")
with (D / "sweep_results.csv").open("w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(
        [
            "config_id",
            "status",
            "main_reference_pass",
            "p1_tf_top1",
            "book32_pc_top1",
            "book32_tf_top1",
            "natural_top1",
            "natural_top5",
            "selected",
            "qkv",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "layer_exceptions",
            "kv_cache",
            "lm_head",
            "activation",
            "ccl",
            "tf_top1",
            "tf_top5",
            "tf_top100",
            "pc_top1",
            "pc_top5",
            "pc_top100",
            "ttft_ms",
            "tf_decode_t_s_u",
            "e2e_t_s_u",
            "traced",
            "decode_trace_replays",
            "eager_decode_steps",
            "commit",
            "sidecar",
        ]
    )
    for r in rows:
        p = r["policy"]
        tf = r["teacher_forcing"]
        pc = r["prefill_check"]
        writer.writerow(
            [
                r["config_id"],
                r["status"],
                r["main_reference_pass"],
                r["robustness_references"].get("aime_p1", {}).get("teacher_forcing", {}).get("top1"),
                r["robustness_references"].get("book32", {}).get("prefill_check", {}).get("top1"),
                r["robustness_references"].get("book32", {}).get("teacher_forcing", {}).get("top1"),
                r["robustness_references"].get("natural_text", {}).get("top1"),
                r["robustness_references"].get("natural_text", {}).get("top5"),
                r["selected"],
                p["weights"]["qkv"],
                p["weights"]["o_proj"],
                p["weights"]["gate_proj"],
                p["weights"]["up_proj"],
                p["weights"]["down_proj"],
                json.dumps(p["layer_exceptions"]) if p["layer_exceptions"] else "",
                p["kv_cache_dtype"],
                p["lm_head"],
                p["activation_dtype"],
                p["ccl_dtype"],
                tf["top1"],
                tf["top5"],
                tf["top100"],
                pc["top1"],
                pc["top5"],
                pc["top100"],
                tf["ttft_ms"],
                tf["decode_t_s_u"],
                tf["e2e_t_s_u"],
                tf["traced"],
                tf["runtime_counters"].get("decode_trace_replays"),
                tf["runtime_counters"].get("eager_decode_steps"),
                r["checkout_git_sha"],
                r["sidecar"],
            ]
        )


def pareto(points):
    best = []
    for x, y, cid in sorted(points, key=lambda p: (-p[0], -p[1])):
        if not best or y > best[-1][1]:
            best.append((x, y, cid))
    return sorted(best)


def chart(metric: str, bar: float, path: Path):
    pts = [
        (r["teacher_forcing"][metric], r["teacher_forcing"]["decode_t_s_u"], r["config_id"])
        for r in rows
        if r["teacher_forcing"]["decode_t_s_u"] is not None
        and r["teacher_forcing"][metric] is not None
        and r["teacher_forcing"]["traced"]
        and r["status"] in ("pass", "fail")
    ]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=140)
    front = pareto(pts)
    if len(front) > 1:
        ax.plot(
            [p[0] for p in front],
            [p[1] for p in front],
            color="#4c72b0",
            linewidth=1.4,
            alpha=0.8,
            zorder=1,
            label="Pareto frontier",
        )
    else:
        fx, fy, fcid = front[0]
        ax.scatter(
            [fx],
            [fy],
            s=320,
            facecolors="none",
            edgecolors="#4c72b0",
            linewidths=1.6,
            zorder=2,
            label=f"Pareto frontier: one non-dominated point ({fcid}); every candidate ties at {metric.replace('top', 'top-')} {fx:.3f}",
        )
    statuses = {r["config_id"]: r["status"] for r in rows}
    for x, y, cid in pts:
        selected = cid == args.selected
        ax.scatter(
            x,
            y,
            s=140 if selected else 70,
            color="#d62728" if selected else ("#2ca02c" if statuses[cid] == "pass" else "#7f7f7f"),
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
    groups = {}
    for x, y, cid in pts:
        groups.setdefault(round(x, 3), []).append((y, cid))
    offsets = [(7, 5), (7, -11), (-7, 5), (-7, -11), (7, 17), (-7, 17), (7, -23), (-7, -23)]
    for x, members in groups.items():
        members.sort(reverse=True)
        for n, (y, cid) in enumerate(members):
            dx, dy = offsets[n % len(offsets)]
            label = (
                f"{cid} ({x:.3f}, {y:.2f})"
                if cid == args.selected
                else cid.split("_", 1)[0] + " " + cid.split("_", 1)[1].replace("_", " ")
            )
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=6.8,
                ha="left" if dx > 0 else "right",
                color="#d62728" if cid == args.selected else "#222222",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#d62728", lw=0.8)
                if cid == args.selected
                else None,
                zorder=4,
            )
    ax.axvline(
        bar, linestyle=":", color="#d62728", linewidth=1.3, label=f"minimum {metric.replace('top', 'top-')} = {bar:.2f}"
    )
    ax.set_xlabel(f"teacher-forcing {metric.replace('top', 'top-')} accuracy (AIME24 chat, 100 tokens)")
    ax.set_ylabel("traced teacher-forcing decode t/s/u (batch 1, chip 2)")
    ax.set_title(
        f"Qwen3-8B datatype sweep: {metric.replace('top', 'top-')} versus traced decode throughput\n(green = passes the bar, grey = fails, red = selected)"
    )
    ax.grid(True, alpha=0.25)
    xs = [p[0] for p in pts] + [bar]
    lo, hi = min(xs), max(xs)
    pad = max(0.01, (hi - lo) * 0.15)
    ax.set_xlim(lo - pad, min(1.0, hi + pad) + 0.005)
    ax.legend(loc="lower left", fontsize=8)
    ax.text(
        0.01,
        0.99,
        "frontier through every evaluated point; admissibility (green) also needs the robustness references",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


chart("top1", args.bar_top1, D / "top1_perf_pareto.png")
chart("top5", args.bar_top5, D / "top5_perf_pareto.png")
for r in rows:
    tf = r["teacher_forcing"]
    rb = r["robustness_references"]
    print(
        f"{r['config_id']:28s} {r['status']:5s} top1={tf['top1']} top5={tf['top5']} top100={tf['top100']} ttft={tf['ttft_ms']} decode={tf['decode_t_s_u']} traced={tf['traced']} regime={str(tf.get('timing_regime'))[:6]} p1={rb.get('aime_p1', {}).get('teacher_forcing', {}).get('top1')} book32={rb.get('book32', {}).get('prefill_check', {}).get('top1')}/{rb.get('book32', {}).get('teacher_forcing', {}).get('top1')} natural={rb.get('natural_text', {}).get('top1')}/{rb.get('natural_text', {}).get('top5')}"
    )
