# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import argparse
import csv
import json
import os
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
AUTOPORT = os.path.dirname(HERE)
GATE = {"cosine_mean_min": 0.99, "cosine_min_min": 0.97, "head_cos_min_min": 0.95}
TIE_FRACTION = 0.01


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def latency_rows(bench):
    out = {}
    for r in bench["rows"]:
        out[(r["tokens"], r["batch"])] = r["p50_ms"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="accuracy,bfp8_attn,bfp8_attn_hifi2,performance")
    ap.add_argument("--outdir", default=os.path.join(AUTOPORT, "doc", "datatype_sweep"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rows = []
    for name in a.candidates.split(","):
        fid = load(os.path.join(a.outdir, f"fidelity_{name}.json")) or load(
            os.path.join(AUTOPORT, "doc", "full_model", f"fidelity_{name}.json")
        )
        bench = load(os.path.join(a.outdir, f"bench_{name}.json")) or load(
            os.path.join(AUTOPORT, "doc", "optimized_full_model", f"bench_{name}.json")
        )
        if fid is None or bench is None:
            rows.append({"policy": name, "status": "missing", "fidelity": fid is not None, "bench": bench is not None})
            continue
        lat = latency_rows(bench)
        head_min = min(v["min"] for v in fid["head_projection_cosine"].values())
        passed = (
            fid["cosine_single_vs_hf"]["mean"] >= GATE["cosine_mean_min"]
            and fid["cosine_single_vs_hf"]["min"] >= GATE["cosine_min_min"]
            and head_min >= GATE["head_cos_min_min"]
            and fid["nan_count"] == 0
        )
        rows.append(
            {
                "policy": name,
                "status": "pass" if passed else "fail",
                "cosine_mean": fid["cosine_single_vs_hf"]["mean"],
                "cosine_min": fid["cosine_single_vs_hf"]["min"],
                "cosine_p05": fid["cosine_single_vs_hf"]["p05"],
                "head_cos_state_mean": fid["head_projection_cosine"]["state"]["mean"],
                "head_cos_min": head_min,
                "batched_vs_single_min": fid["cosine_single_vs_batched"]["min"],
                "lat_128_b1_ms": lat.get((128, 1)),
                "lat_128_b8_ms": lat.get((128, 8)),
                "lat_1024_b1_ms": lat.get((1024, 1)),
                "lat_2048_b1_ms": lat.get((2048, 1)),
                "tokens_per_s_1024_b8": next(
                    (r["tokens_per_s"] for r in bench["rows"] if r["tokens"] == 1024 and r["batch"] == 8), None
                ),
                "fidelity_file": os.path.relpath(fid.get("_path", ""), AUTOPORT) if fid.get("_path") else None,
            }
        )
    passing = [r for r in rows if r.get("status") == "pass" and r.get("lat_128_b1_ms") is not None]
    selected = None
    if passing:
        fastest = min(passing, key=lambda r: r["lat_128_b1_ms"])
        near = [r for r in passing if r["lat_128_b1_ms"] <= fastest["lat_128_b1_ms"] * (1.0 + TIE_FRACTION)]
        selected = max(near, key=lambda r: r["cosine_min"])
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gate": GATE,
        "metric_mapping": "top-1/top-5 token accuracy replaced by embedding cosine vs HF fp32 reference and head-projection cosine (encoder-only model)",
        "selection_rule": f"fastest passing policy by 128-token batch-1 latency; policies within {TIE_FRACTION:.0%} of the fastest are a tie and the highest minimum cosine wins",
        "rows": rows,
        "selected": selected["policy"] if selected else None,
    }
    with open(os.path.join(a.outdir, "sweep_results.json"), "w") as f:
        json.dump(result, f, indent=1)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(a.outdir, "sweep_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    if selected:
        with open(os.path.join(a.outdir, "selected_precision_config.json"), "w") as f:
            json.dump(
                {
                    "precision": selected["policy"],
                    "env": {"CLM_PRECISION": selected["policy"]},
                    "gate": GATE,
                    "selected_row": selected,
                },
                f,
                indent=1,
            )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for metric, fname, gate in (
            ("cosine_mean", "cosine_mean_perf_pareto.png", GATE["cosine_mean_min"]),
            ("cosine_min", "cosine_min_perf_pareto.png", GATE["cosine_min_min"]),
        ):
            fig, ax = plt.subplots(figsize=(6, 4))
            for r in rows:
                if r.get("status") == "missing":
                    continue
                color = "red" if selected and r["policy"] == selected["policy"] else "tab:blue"
                ax.scatter(r["lat_128_b1_ms"], r[metric], color=color, s=60)
                ax.annotate(
                    r["policy"], (r["lat_128_b1_ms"], r[metric]), textcoords="offset points", xytext=(5, 5), fontsize=8
                )
            ax.axhline(gate, linestyle=":", color="gray")
            ax.set_xlabel("single-text latency, 128-token bucket (ms, p50)")
            ax.set_ylabel(metric + " vs HF fp32")
            ax.set_title("CLM-v0.1-8B encoder on p150: precision policy sweep")
            fig.tight_layout()
            fig.savefig(os.path.join(a.outdir, fname), dpi=120)
            plt.close(fig)
    except Exception as exc:
        result["plot_error"] = str(exc)
        with open(os.path.join(a.outdir, "sweep_results.json"), "w") as f:
            json.dump(result, f, indent=1)
    print(
        "SWEEP",
        json.dumps(
            {
                "selected": result["selected"],
                "rows": [
                    {
                        k: r.get(k)
                        for k in (
                            "policy",
                            "status",
                            "cosine_mean",
                            "cosine_min",
                            "head_cos_min",
                            "lat_128_b1_ms",
                            "lat_1024_b1_ms",
                        )
                    }
                    for r in rows
                ],
            }
        ),
    )


if __name__ == "__main__":
    main()
