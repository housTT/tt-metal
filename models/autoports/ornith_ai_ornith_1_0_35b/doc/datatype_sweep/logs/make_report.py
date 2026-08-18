# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Aggregate the sweep runs into ``sweep_results.{json,csv}`` and the two Pareto charts.

Reads every ``doc/datatype_sweep/runs/*.json`` written by ``sweep_one.py``, ranks the passing
candidates by trace-verified teacher-forcing decode t/s/u, and renders

    top1_perf_pareto.png    x = full-model top-1, y = traced teacher-forcing decode t/s/u
    top5_perf_pareto.png    x = full-model top-5, same y

with the non-dominated frontier drawn through the evaluated points, the selected config in red and a
dotted vertical line at the minimum allowed accuracy.

**The accuracy on the x axis is the binding one**: ``min(prefill-check, teacher-forcing)``, because
the acceptance gate is *both* gates, so a config's distance from the bar is set by its worse one.
Each point's tooltip-equivalent - the CSV and JSON rows - carries both separately.

    python .../doc/datatype_sweep/logs/make_report.py
    python .../doc/datatype_sweep/logs/make_report.py --select C05-proj-bfp4-hifi2
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SWEEP_DIR = Path("models/autoports/ornith_ai_ornith_1_0_35b/doc/datatype_sweep")
RUNS_DIR = SWEEP_DIR / "runs"

#: Why each candidate is in the matrix, from the file that wrote it.
RATIONALE = (
    {
        item["config_id"]: item["rationale"]
        for item in json.loads((SWEEP_DIR / "candidates" / "index.json").read_text(encoding="utf-8"))
    }
    if (SWEEP_DIR / "candidates" / "index.json").exists()
    else {}
)

# The reference palette's light-mode slots. Categorical identity here is
# passing / failing / selected - three slots, which is exactly the all-pairs-validated cap.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
BLUE = "#2a78d6"  # slot 1: evaluated and passing
ORANGE = "#eb6834"  # slot 2: evaluated and failing the gate
RED = "#e34948"  # slot 8, reserved here for the selected point, per the skill's brief
GRID = "#e6e5e1"


def load_rows() -> list[dict]:
    rows = []
    for path in sorted(RUNS_DIR.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        result = record["result"]
        policy = record["dtype_policy"]
        weights = policy["weight_groups"]
        fidelity = policy["compute_fidelities"]
        rows.append(
            {
                "config_id": record["config_id"],
                "precision_config_path": record.get("precision_config_path"),
                "run_artifact": str(path),
                "rationale": RATIONALE.get(record["config_id"], ""),
                # ---- dtype policy, flattened for the CSV ----
                "expert_gate_up_dtype": weights["routed_expert_gate_up"],
                "expert_down_dtype": weights["routed_expert_down"],
                "proj_dtype": weights["dense_projections"],
                "shared_dtype": weights["shared_expert"],
                "router_dtype": weights["router"],
                "lm_head_dtype": weights["lm_head"] or f"{weights['dense_projections']} (follows dense)",
                "expert_fidelity": fidelity["routed_experts"],
                "proj_fidelity": fidelity["dense_projections"],
                "shared_fidelity": fidelity["shared_expert"],
                "router_fidelity": fidelity["router"],
                "sdpa_fidelity": fidelity["sdpa"],
                "lm_head_fidelity": fidelity["lm_head"] or f"{fidelity['dense_projections']} (follows dense)",
                "residual_dtype": policy["activations"]["residual_stream"],
                "expert_act_dtype": policy["activations"]["routed_expert_output"],
                "ccl_dtype": policy["ccl"]["payload_dtype"] or "as produced (no cast)",
                "kv_cache_dtype": policy["kv_cache"]["dtype"],
                "logits_dtype": policy["logits_sampling"]["logits_dtype"],
                "layer_exceptions": json.dumps(policy["layer_exceptions"]),
                # ---- results ----
                "prefill_top1": result["prefill_top1"],
                "prefill_top5": result["prefill_top5"],
                "prefill_top100": result["prefill_top100"],
                "teacher_top1": result["teacher_top1"],
                "teacher_top5": result["teacher_top5"],
                "teacher_top100": result["teacher_top100"],
                "top1": min(result["prefill_top1"], result["teacher_top1"]),
                "top5": min(result["prefill_top5"], result["teacher_top5"]),
                "top100": min(result["prefill_top100"], result["teacher_top100"]),
                "ttft_ms": result["ttft_ms"],
                "teacher_decode_t_s_u": result["teacher_decode_t/s/u"],
                "teacher_decode_ms_per_token": result["teacher_decode_ms_per_token"],
                "teacher_decode_warm_repeats": json.dumps(result["teacher_decode_t/s/u_warm_repeats"]),
                "teacher_decode_spread_pct": result["teacher_decode_spread_pct"],
                "warm_repeats": result["warm_repeats"],
                "total_repeats": result["total_repeats"],
                "tokens": result["tokens"],
                "status": result["status"],
                "failures": "; ".join(result["failures"]),
                # ---- provenance ----
                "measurement_regime_accuracy": record["measurement_regime"]["accuracy"],
                "measurement_regime_performance": record["measurement_regime"]["performance"],
                "workload": record["measurement_regime"]["workload"],
                "reference": record["reference"],
                "command": record["command"],
                "branch": record["branch"],
                "commit": record["commit"],
                "hardware": f"{record['hardware']['board']} x{record['hardware']['num_devices']} "
                f"({record['hardware']['arch']}) on {record['hardware']['host']}",
                "mesh": "x".join(str(d) for d in record["hardware"]["mesh_shape"]),
                "build_s": record.get("build_s"),
                "trace_verified": all(
                    r["trace"]["enable_trace"] and r["trace"]["trace_id_present"]
                    for r in record["teacher_forcing_repeats"]
                ),
                "prefill_sdpa_chunk": record["precision_summary"]["built"]["prefill_sdpa_chunk"],
                "built_lm_head_weight_dtype": record["precision_summary"]["built"]["lm_head_weight_dtype"],
                "built_lm_head_math_fidelity": record["precision_summary"]["built"]["lm_head_math_fidelity"],
            }
        )
    return rows


def pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Non-dominated set for (accuracy, throughput), both higher-is-better, sorted by accuracy."""
    front = []
    for x, y in points:
        if not any((ox >= x and oy >= y) and (ox > x or oy > y) for ox, oy in points):
            front.append((x, y))
    # One point per accuracy value - the fastest - so the drawn line is a function of accuracy.
    best: dict[float, float] = {}
    for x, y in front:
        best[x] = max(best.get(x, float("-inf")), y)
    return sorted(best.items())


def draw(rows: list[dict], *, metric: str, gate: float, selected_id: str, path: Path, title: str) -> None:
    """One Pareto chart. ``metric`` is ``"top1"`` or ``"top5"``."""
    fig, ax = plt.subplots(figsize=(12.0, 8.0), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    xs = [r[metric] for r in rows]
    ys = [r["teacher_decode_t_s_u"] for r in rows]

    # --- frontier over every evaluated point -------------------------------------------------
    front = pareto_front(list(zip(xs, ys)))
    if len(front) > 1:
        ax.plot(
            [p[0] for p in front],
            [p[1] for p in front],
            color=MUTED,
            linewidth=2.0,
            zorder=2,
            solid_capstyle="round",
            label="_nolegend_",
        )
    ax.scatter(
        [p[0] for p in front],
        [p[1] for p in front],
        s=210,
        facecolors="none",
        edgecolors=MUTED,
        linewidths=1.6,
        zorder=3,
    )

    # --- the minimum allowed accuracy ---------------------------------------------------------
    ax.axvline(gate, color=INK_2, linestyle=":", linewidth=1.8, zorder=1)
    ax.annotate(
        f"minimum allowed {metric.replace('top', 'top-')} = {gate:.2f}",
        xy=(gate, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(6, -14),
        textcoords="offset points",
        color=INK_2,
        fontsize=9.5,
        rotation=0,
        ha="left",
        va="top",
    )

    # --- the points ---------------------------------------------------------------------------
    for row in rows:
        selected = row["config_id"] == selected_id
        passing = row["status"] == "pass"
        color = RED if selected else (BLUE if passing else ORANGE)
        ax.scatter(
            [row[metric]],
            [row["teacher_decode_t_s_u"]],
            s=150 if selected else 80,
            color=color,
            marker="o" if passing else "X",
            edgecolors=SURFACE,
            linewidths=2.0,
            zorder=6 if selected else 5,
        )

    lo, hi = min(xs), max(xs)
    span = max(hi - lo, 0.02)
    # Room on the left for the low-accuracy gutter and on the right for the high-accuracy one.
    ax.set_xlim(min(lo, gate) - span * 0.12, min(hi + span * 0.12, 1.005))
    ylo, yhi = min(ys), max(ys)
    yspan = max(yhi - ylo, 0.5)
    ax.set_ylim(ylo - yspan * 0.09, yhi + yspan * 0.09)
    lo_x, hi_x = ax.get_xlim()
    lo_y, hi_y = ax.get_ylim()

    # --- direct labels ------------------------------------------------------------------------
    # Every evaluated config is labelled, so the frontier and the rejected arms are readable without
    # opening sweep_results.csv. With 20+ points sitting on six distinct accuracies, labels beside
    # their marks collide, so they live in the figure margin *outside* the axes - one column, packed
    # in throughput order, joined to their mark by a thin leader. Because both the column and the
    # points are ordered by throughput, no two leaders cross. The selected row's leader is red, so
    # the eye lands on it before it reads any text.
    ordered_rows = sorted(rows, key=lambda r: -r["teacher_decode_t_s_u"])
    inset = (hi_y - lo_y) * 0.035
    top, bottom = hi_y - inset, lo_y + inset
    step = (top - bottom) / max(len(ordered_rows) - 1, 1)
    for slot, row in enumerate(ordered_rows):
        selected = row["config_id"] == selected_id
        head, tail = row["config_id"].split("-", 1)
        label = f"{head}  {tail}" + ("   \u2190 selected" if selected else "")
        ax.annotate(
            label,
            xy=(row[metric], row["teacher_decode_t_s_u"]),
            xycoords="data",
            xytext=(1.025, top - slot * step),
            textcoords=("axes fraction", "data"),
            fontsize=8.8,
            color=INK if selected else INK_2,
            fontweight="bold" if selected else "normal",
            va="center",
            ha="left",
            annotation_clip=False,
            arrowprops={
                "arrowstyle": "-",
                "color": RED if selected else GRID,
                "linewidth": 1.5 if selected else 0.9,
                "shrinkA": 2,
                "shrinkB": 5,
            },
            zorder=4,
        )

    ax.set_xlabel(
        f"full-model {metric.replace('top', 'top-')} agreement with the HF reference\n"
        "(the binding gate: min of run_prefill_check and run_teacher_forcing)",
        fontsize=10.5,
        color=INK_2,
    )
    ax.set_ylabel("trace-verified teacher-forcing decode  (tokens/s/user)", fontsize=10.5, color=INK_2)
    ax.set_title(title, fontsize=13.5, color=INK, pad=14, loc="left")
    # When every config lands on the same accuracy the frontier degenerates to one point, and saying
    # so is more useful than leaving the reader to work out why the line vanished.
    if len({round(x, 6) for x in xs}) == 1:
        ax.text(
            0.0,
            1.012,
            f"every evaluated config scores exactly {xs[0]:.3f}, so this metric constrains nothing here: "
            "the frontier is the single fastest point",
            transform=ax.transAxes,
            fontsize=9.4,
            color=INK_2,
            ha="left",
            va="bottom",
        )

    ax.grid(True, color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9.5)

    handles = [
        Line2D(
            [], [], marker="o", linestyle="", color=RED, markersize=9, markeredgecolor=SURFACE, label="selected config"
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=BLUE,
            markersize=8,
            markeredgecolor=SURFACE,
            label="evaluated, passes the gate",
        ),
        Line2D(
            [],
            [],
            marker="X",
            linestyle="",
            color=ORANGE,
            markersize=9,
            markeredgecolor=SURFACE,
            label="evaluated, fails the gate",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="-",
            color=MUTED,
            markersize=11,
            markerfacecolor="none",
            markeredgecolor=MUTED,
            linewidth=2.0,
            label="Pareto frontier (non-dominated)",
        ),
    ]
    legend = ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=9.2, labelcolor=INK_2, borderpad=0.7)
    legend.get_frame().set_facecolor(SURFACE)
    legend.get_frame().set_edgecolor(GRID)

    # The label column lives to the right of the axes, so the axes stop at 60 % of the canvas.
    fig.subplots_adjust(left=0.085, right=0.605, top=0.925, bottom=0.115)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", default=None, help="config id to mark as selected (default: fastest passing)")
    ap.add_argument("--gate-top1", type=float, default=0.90)
    ap.add_argument("--gate-top5", type=float, default=0.98)
    args = ap.parse_args()

    rows = load_rows()
    if not rows:
        raise SystemExit(f"no run artifacts in {RUNS_DIR}")

    passing = [r for r in rows if r["status"] == "pass"]
    fastest = max(passing, key=lambda r: r["teacher_decode_t_s_u"]) if passing else None
    selected_id = args.select or (fastest["config_id"] if fastest else rows[0]["config_id"])
    baseline = next((r for r in rows if r["config_id"].startswith("S00")), None)

    for row in rows:
        row["selected"] = row["config_id"] == selected_id
        if baseline:
            row["decode_speedup_vs_baseline_pct"] = (
                row["teacher_decode_t_s_u"] / baseline["teacher_decode_t_s_u"] - 1
            ) * 100

    ordered = sorted(rows, key=lambda r: (-r["teacher_decode_t_s_u"],))
    blocked = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((SWEEP_DIR / "blocked").glob("*.json"))
        if not path.name.endswith(".config.json")
    ]
    (SWEEP_DIR / "sweep_results.json").write_text(
        json.dumps(
            {
                "gate": {"top1": args.gate_top1, "top5": args.gate_top5, "top100": 1.00},
                "selection_rule": "the fastest evaluated config that satisfies both accuracy gates on "
                "both readiness checks, ranked by trace-verified teacher-forcing decode t/s/u",
                "selected": selected_id,
                "baseline": baseline["config_id"] if baseline else None,
                "results": ordered,
                "blocked": blocked,
                "blocked_note": "candidates that could not be measured at all. Each carries the exact "
                "TTNN/runtime blocker, the adaptation that was tried, and why the arm was not pursued "
                "further. They are absent from the Pareto charts because they have no measurement.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {SWEEP_DIR / 'sweep_results.json'}")

    fieldnames = list(ordered[0].keys())
    with (SWEEP_DIR / "sweep_results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)
    print(f"wrote {SWEEP_DIR / 'sweep_results.csv'}")

    draw(
        rows,
        metric="top1",
        gate=args.gate_top1,
        selected_id=selected_id,
        path=SWEEP_DIR / "top1_perf_pareto.png",
        title="Ornith-1.0-35B precision sweep — top-1 accuracy against traced decode throughput",
    )
    draw(
        rows,
        metric="top5",
        gate=args.gate_top5,
        selected_id=selected_id,
        path=SWEEP_DIR / "top5_perf_pareto.png",
        title="Ornith-1.0-35B precision sweep — top-5 accuracy against traced decode throughput",
    )

    print(f"\nselected: {selected_id}")
    for row in ordered:
        mark = "*" if row["selected"] else " "
        print(
            f"{mark} {row['config_id']:<26} {row['status']:>4}  top1={row['top1']:.3f} top5={row['top5']:.3f}  "
            f"{row['teacher_decode_t_s_u']:.3f} t/s/u  ({row.get('decode_speedup_vs_baseline_pct', 0):+.2f} %)"
        )


if __name__ == "__main__":
    main()
