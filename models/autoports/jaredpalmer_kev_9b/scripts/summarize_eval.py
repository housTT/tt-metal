import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_ROOT = "/home/hous/dev/kev/reports/eval"
SUITES = ("hard-v1", "devtools-v1", "documents-v1", "breadth-v1", "smoke-v1")
AUDIT_EXCLUDE_SOURCES = ("flakeflagger",)
AUDIT_EXCLUDE_TASKS = ("commitpackft_type",)
CARD = [
    ("hard-v1 + devtools-v1, audited", "development 0.821 / test 0.822 (accuracy)"),
    ("documents-v1", "development 0.902 / test 0.900 (accuracy)"),
    ("breadth-v1, all 14 datasets", "development 0.700 / test 0.698 (accuracy); test ECE 0.034"),
    ("transfer-v4 locked test", "ECE 0.034"),
]


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi) if hi < 1 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def scored(rows):
    acc, conf, brier = [], [], []
    for row in rows:
        p = np.asarray(row["p"], dtype=float)
        y = row["label"]
        acc.append(int(p.argmax() == y))
        conf.append(float(p.max()))
        brier.append(float(((p - np.eye(len(p))[y]) ** 2).sum()))
    return {"n": len(rows), "acc": float(np.mean(acc)), "brier": float(np.mean(brier)), "ece": ece(conf, acc)}


def knowable(rows):
    return [r for r in rows if r["variant"] == "clean" and r["source"] != "unknowable"]


def audited(rows):
    return [r for r in rows if r["source"] not in AUDIT_EXCLUDE_SOURCES and r["task"] not in AUDIT_EXCLUDE_TASKS]


def line(name, m, latency=None):
    lat = f"  latency p50 {latency['median']:.0f} ms p95 {latency['p95']:.0f} ms" if latency else ""
    return f"{name:44} n={m['n']:5d}  acc {m['acc']:.3f}  brier {m['brier']:.3f}  ece {m['ece']:.3f}{lat}"


def main():
    ap = argparse.ArgumentParser(
        description="Print accuracy, Brier and ECE of every kev.benchmark report under the eval root next to the model-card numbers."
    )
    ap.add_argument("--root", default=DEFAULT_ROOT)
    a = ap.parse_args()
    root = Path(a.root)
    pooled = {}
    print(f"reports under {root}")
    for split in ("development", "test"):
        for suite in SUITES:
            report = root / suite / split / "report.json"
            if not report.exists():
                print(f"{suite + '/' + split:44} not run")
                continue
            r = json.loads(report.read_text(encoding="utf-8"))
            print(line(f"{suite}/{split}", r["clean"], r.get("latency_ms")))
            rows_path = root / suite / split / "rows.json"
            if suite in ("hard-v1", "devtools-v1") and rows_path.exists():
                pooled.setdefault(split, []).extend(
                    audited(knowable(json.loads(rows_path.read_text(encoding="utf-8"))))
                )
        if split in pooled:
            print(line(f"hard-v1 + devtools-v1 audited/{split}", scored(pooled[split])))
    print()
    print("model card (jaredpalmer/kev-9b v2, docs/model-cards/kev-9b.md):")
    for name, value in CARD:
        print(f"  {name:40} {value}")
    print(
        "audited = devtools-v1 rows without source flakeflagger and task commitpackft_type (experiments/rounds/r27.json)"
    )


if __name__ == "__main__":
    main()
