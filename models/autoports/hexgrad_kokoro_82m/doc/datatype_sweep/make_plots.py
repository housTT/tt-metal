# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Pareto charts for the Kokoro-82M datatype sweep.

top1_perf_pareto.png : x = full-model teacher-forcing top-1, y = traced TF decode t/s/u
top5_perf_pareto.png : x = full-model teacher-forcing top-5, y = traced TF decode t/s/u

Each chart plots every evaluated config, draws the non-dominated Pareto frontier,
marks the selected config in red, and draws a vertical dotted line at the minimum
allowed accuracy for that chart. Performance metric = trace-verified teacher-forcing
decode t/s/u (the skill's mandated ranking metric); eager/untraced numbers are not
plotted.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
SELECTED = "baseline"
# accessible palette: pass = teal-blue, fail = neutral gray, selected = red
C_PASS = "#1f6f8b"
C_FAIL = "#9aa0a6"
C_SEL = "#d62728"
C_FRONT = "#2a2a2a"


def load():
    return json.loads((HERE / "sweep_results.json").read_text())


def frontier(points):
    """Non-dominated set for (accuracy up good, perf up good). Returns sorted by acc."""
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    front = []
    best_perf = -1e18
    # walk from highest accuracy downward; keep points whose perf exceeds all
    # already-kept (higher-accuracy) points -> non-dominated set
    for p in sorted(points, key=lambda p: -p[0]):
        if p[1] > best_perf:
            front.append(p)
            best_perf = p[1]
    return sorted(front, key=lambda p: p[0])


def make_chart(rows, acc_key, acc_min, title, xlabel, outfile):
    fig, ax = plt.subplots(figsize=(8.6, 5.8), dpi=130)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfc")

    all_pts = []
    for r in rows:
        tf = r.get("teacher_forcing") or {}
        acc = tf.get(acc_key)
        perf = tf.get("decode_t_s_u")
        if acc is None or perf is None:
            continue
        all_pts.append((acc, perf, r["id"], r["status"]))

    # Pareto frontier over ALL evaluated points (pass + fail)
    fr = frontier([(a, p) for a, p, _, _ in all_pts])
    if len(fr) >= 2:
        ax.plot(
            [a for a, _ in fr],
            [p for _, p in fr],
            "-",
            color=C_FRONT,
            lw=1.4,
            alpha=0.55,
            zorder=1,
            label="Pareto frontier",
        )

    # vertical dotted min-accuracy line
    ax.axvline(acc_min, ls=":", color="#b23b3b", lw=1.6, zorder=1)
    ax.text(
        acc_min,
        ax.get_ylim()[1],
        f" min {acc_key} = {acc_min:.2f}",
        color="#b23b3b",
        va="top",
        ha="left",
        fontsize=9,
        rotation=90,
    )

    for acc, perf, cid, status in all_pts:
        if cid == SELECTED:
            ax.scatter(
                [acc],
                [perf],
                s=190,
                color=C_SEL,
                edgecolor="black",
                lw=1.1,
                zorder=5,
                marker="*",
                label="selected (baseline)",
            )
        else:
            col = C_PASS if status == "pass" else C_FAIL
            ax.scatter([acc], [perf], s=70, color=col, edgecolor="white", lw=0.7, zorder=3)
        # annotate
        dy = 1.2 if cid != SELECTED else 2.2
        ax.annotate(cid, (acc, perf), textcoords="offset points", xytext=(5, dy), fontsize=7.4, color="#333")

    # legend proxies for pass/fail
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=C_SEL,
            markeredgecolor="black",
            markersize=15,
            label="selected (baseline)",
        ),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_PASS, markersize=9, label="passes gate"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_FAIL, markersize=9, label="fails gate"),
        Line2D([0], [0], ls="-", color=C_FRONT, alpha=0.55, label="Pareto frontier"),
        Line2D([0], [0], ls=":", color="#b23b3b", label=f"min accuracy ({acc_min:.2f})"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5, framealpha=0.95)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("traced teacher-forcing decode  (t/s/u, higher = better)", fontsize=11)
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=12)
    ax.grid(True, ls="--", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / outfile, bbox_inches="tight")
    plt.close(fig)
    print("wrote", outfile)


def main():
    rows = load()
    make_chart(
        rows,
        "top1",
        0.90,
        "Kokoro-82M datatype sweep — top-1 vs traced decode (Pareto)",
        "full-model teacher-forcing top-1 accuracy",
        "top1_perf_pareto.png",
    )
    make_chart(
        rows,
        "top5",
        0.98,
        "Kokoro-82M datatype sweep — top-5 vs traced decode (Pareto)",
        "full-model teacher-forcing top-5 accuracy",
        "top5_perf_pareto.png",
    )


if __name__ == "__main__":
    main()
