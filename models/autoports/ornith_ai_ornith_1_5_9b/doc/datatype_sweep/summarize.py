# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Collect full-model results and draw measured accuracy/throughput frontiers."""

import argparse
import copy
import csv
import json
import re
import statistics
from pathlib import Path

DOC = Path(__file__).resolve().parent


def collect():
    rows = []
    decisions_path = DOC / "quality_decisions.json"
    decisions = json.loads(decisions_path.read_text()) if decisions_path.exists() else {}
    for path in sorted(DOC.glob("*.json")):
        result = json.loads(path.read_text())
        if not isinstance(result, dict) or "teacher_forcing" not in result or not result.get("trace_verified"):
            continue
        metrics = result["teacher_forcing"][0]
        provenance = json.loads((DOC / "logs" / f"{path.stem}.provenance.json").read_text())
        policy = copy.deepcopy(result["dtype_policy"])
        geometry = policy.setdefault("head_geometry", {"cores": 64, "columns": 32768, "in0_block_w": 1, "readers": 2})
        command = result["command"]
        if "--head-block-w" in command:
            geometry["in0_block_w"] = int(command[command.index("--head-block-w") + 1])
        config_id = result["config_id"]
        if geometry != {"cores": 64, "columns": 32768, "in0_block_w": 1, "readers": 2} and "_c" not in config_id:
            config_id += f"_c{geometry['cores']}_k{geometry['in0_block_w']}_r{geometry['readers']}"
        policy["config_id"] = config_id
        rows.append(
            dict(
                config_id=config_id,
                run_id=path.stem,
                precision_config_path=str(DOC / "configs" / f"{config_id}.json"),
                dtype_policy=policy,
                compute_fidelity_policy=policy["compute_fidelities"],
                head_geometry=geometry,
                top1=metrics["top1"],
                top5=metrics["top5"],
                top100=metrics["top100"],
                token_count=metrics["total"],
                ttft_ms=metrics["ttft_ms"],
                traced_teacher_forcing_decode_t_s_u=metrics["decode_t/s/u"],
                prefill=result.get("prefill_check"),
                trace_verified=True,
                teacher_repeat_count=len(result.get("teacher_forcing_samples", [None])),
                teacher_samples=result.get("teacher_forcing_samples"),
                measurement_regime=result["measurement_regime"],
                command=provenance["command"],
                hardware=result["hardware"],
                clock_policy={
                    "requested_mhz": 1350,
                    "startup_advisories_mhz": [
                        int(v)
                        for v in re.findall(
                            r"AICLK settled at (\d+) MHz", (DOC / "logs" / f"{path.stem}.log").read_text()
                        )
                    ],
                    "timed_window_clock_sampled": False,
                },
                mesh=result["mesh"],
                pass_status=result["pass_status"],
                qualitative_decision=decisions.get(config_id, {"status": "not_evaluated"}),
                git_branch=provenance.get("git_branch", "hous/ornith-1.5-9b"),
                reference=result["reference"],
                git_head=provenance["git_head"],
                environment=provenance["environment"],
                evidence=str(path),
            )
        )
    return rows


def plot(rows, selected, metric, threshold):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    speed = "traced_teacher_forcing_decode_t_s_u"
    groups = {}
    for row in rows:
        groups.setdefault(row["config_id"], []).append(row)
    measured = []
    for name, samples in sorted(groups.items()):
        repeated = [r for r in samples if r["teacher_repeat_count"] > 1]
        row = dict(repeated[-1] if repeated else samples[0])
        observations = (
            [sample["metrics"][0]["decode_t/s/u"] for r in repeated for sample in r["teacher_samples"]]
            if repeated
            else [r[speed] for r in samples]
        )
        row[speed] = statistics.median(observations)
        measured.append(row)
    fig = plt.figure(figsize=(15, 8), facecolor="#f6f8fc")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.7, 1], wspace=0.12)
    ax = fig.add_subplot(grid[0])
    key = fig.add_subplot(grid[1])
    key.axis("off")
    ax.set_facecolor("#f6f8fc")
    points = sorted({(100 * r[metric], r[speed]) for r in measured})
    frontier = [(x, y) for x, y in points if not any(a >= x and b >= y and (a > x or b > y) for a, b in points)]
    if frontier:
        ax.plot(
            *zip(*frontier),
            color="#168b8a",
            linewidth=2,
            marker="D",
            markersize=10,
            markerfacecolor="none",
            markeredgewidth=2,
            label="Accuracy/performance frontier",
            zorder=6,
        )
    for i, row in enumerate(measured):
        chosen = row["config_id"] == selected
        rejected = row["qualitative_decision"]["status"] == "fail"
        color = "#d52d45" if chosen else "#929aa9" if rejected else "#306b9a"
        ax.scatter(
            100 * row[metric],
            row[speed],
            color=color,
            s=180 if chosen else 55,
            marker="*" if chosen else "o",
            zorder=5 if chosen else 3,
        )
        if (
            chosen
            or "baseline" in row["config_id"]
            or "canonical" in row["config_id"]
            or row[speed] == max(r[speed] for r in measured)
        ):
            ax.annotate(
                str(i + 1),
                (100 * row[metric], row[speed]),
                xytext=(-16, 9),
                textcoords="offset points",
                fontsize=9,
                color=color,
            )
        label = row["config_id"].replace("baseline_bfp4_lofi_qkvg8_lofi_head16_hifi4", "baseline")
        suffix = "  SELECTED" if chosen else "  quality fail" if rejected else ""
        key.text(
            0,
            0.98 - i * (0.9 / max(len(measured), 1)),
            f"{i+1:02d}  {label}  {row[speed]:.2f}{suffix}",
            fontsize=8,
            color=color,
            va="top",
            transform=key.transAxes,
        )
    ax.axvline(
        threshold * 100, color="#525b6c", linestyle=":", linewidth=1.8, label=f"Minimum {metric}: {threshold:.0%}"
    )
    ax.set(xlabel=f"Full-model {metric} accuracy (%)", ylabel="Traced teacher-forcing decode (tokens/s/user)")
    ax.set_xlim(min(threshold * 100, min(x for x, _ in points)) - 0.6, min(100.3, max(x for x, _ in points) + 1.2))
    ax.grid(alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.suptitle(f"Ornith-1.5-9B · {metric} / decode tradeoff", fontsize=19, x=0.12, ha="left")
    fig.text(
        0.12,
        0.918,
        "AIME24 chat100 · full32 layers · batch1 · TP4 on physical P300c boards",
        fontsize=11,
        color="#525b6c",
    )
    fig.text(
        0.12,
        0.025,
        "Each point is an evaluated full-model configuration; repeated configs use the median of all repeated traced samples. Gray marks controlled quality failures.",
        fontsize=9,
        color="#525b6c",
    )
    fig.subplots_adjust(top=0.87, bottom=0.12, left=0.08, right=0.98)
    fig.savefig(DOC / f"{metric}_perf_pareto.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected")
    args = parser.parse_args()
    rows = collect()
    for row in rows:
        row["accuracy_pass"] = row["pass_status"] == "pass"
        row["selection_status"] = (
            "selected"
            if row["config_id"] == args.selected
            else "rejected_quality"
            if row["qualitative_decision"]["status"] == "fail"
            else "not_selected_slower"
            if args.selected
            else "pending"
        )
    (DOC / "sweep_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    if rows:
        with (DOC / "sweep_results.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(
                {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in rows
            )
    if args.selected:
        plot(rows, args.selected, "top1", 0.90)
        plot(rows, args.selected, "top5", 0.98)
    for row in rows:
        print(
            row["run_id"],
            row["pass_status"],
            row["top1"],
            row["top5"],
            round(row["traced_teacher_forcing_decode_t_s_u"], 3),
        )


if __name__ == "__main__":
    main()
