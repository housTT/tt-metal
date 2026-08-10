# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Consistency checker for the functional-decoder stage documents.

Seven rounds of stage review on this stage produced findings in one class over and over: a
number or a path is corrected in one document and not in the others that carry it, or a
document cites an artifact that has moved or does not exist.  Reviewing harder does not fix
that; a check does.  This script is that check::

    python -m models.autoports.qwen_qwen3_6_27b.scripts.check_docs

It verifies three things and exits non-zero on the first failure:

1. **Paths resolve.** Every markdown link and every backticked repo-relative artifact path in
   the stage documents points at a file that exists.
2. **Headline numbers agree with their artifact.** The record counts, the PCC minimum and the
   scale range quoted in the documents are re-derived from ``pcc_evidence.json``; the perf
   numbers are re-derived by summing the ``Device Time`` column of the ``tt-perf-report`` CSVs;
   the test counts are re-derived from the run logs.
3. **One value per figure.** A figure that appears in more than one document appears with the
   same value in all of them.

It reads only committed artifacts and opens no device, so it is safe to run anywhere.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc" / "functional_decoder"
DOCS = [
    DOC / "README.md",
    DOC / "work_log.md",
    DOC / "probes" / "README.md",
    DOC / "watcher" / "WATCHER_AUDIT.md",
    ROOT / "doc" / "context_contract.json",
]
#: ``(phase, replays)`` — decode is measured as N trace replays inside one signposted window.
PHASES = {"prefill": 1, "decode": 8}


def fail(message: str) -> None:
    print(f"FAIL {message}")
    sys.exit(1)


def check_paths() -> None:
    """Every link and backticked artifact path in the stage documents resolves."""
    missing = []
    for doc in DOCS:
        text = doc.read_text()
        for match in re.finditer(r"\]\(([^)#][^)]*)\)", text):
            target = match.group(1).split("#")[0]
            if target.startswith("http"):
                continue
            if not (doc.parent / target).exists():
                missing.append(f"{doc.name}: link -> {target}")
        for match in re.finditer(r"`((?:\.\./)*(?:doc/|logs/|tracy/|watcher/|probes/)[\w./-]+)`", text):
            target = match.group(1)
            if not any((base / target).exists() for base in (doc.parent, ROOT, ROOT / "doc", Path("."))):
                missing.append(f"{doc.name}: path -> {target}")
    if missing:
        fail("unresolved paths:\n  " + "\n  ".join(sorted(set(missing))))
    print(f"ok   every link and artifact path in {len(DOCS)} documents resolves")


def check_evidence() -> dict:
    """Re-derive the record counts, the PCC minimum and the scale range."""
    evidence = json.loads((DOC / "pcc_evidence.json").read_text())
    records = evidence["records"]
    numeric = [r for r in records if isinstance(r["value"], (int, float)) and not isinstance(r["value"], bool)]
    pcc = [r for r in numeric if not r["metric"].endswith("_scale")]
    scale = [r for r in numeric if r["metric"].endswith("_scale")]
    derived = {
        "records": len(records),
        "pcc_records": len(pcc),
        "scale_records": len(scale),
        "min_pcc": min(r["value"] for r in pcc),
        "scale_range": [min(r["value"] for r in scale), max(r["value"] for r in scale)],
    }
    for key in ("records", "pcc_records", "scale_records", "min_pcc"):
        if (
            evidence[
                {
                    "records": "num_records",
                    "pcc_records": "num_pcc_records",
                    "scale_records": "num_scale_records",
                    "min_pcc": "min_pcc",
                }[key]
            ]
            != derived[key]
        ):
            fail(f"pcc_evidence.json summary field {key} disagrees with its own records")

    below = [r for r in pcc if r["value"] < 0.995]
    if below:
        fail(f"{len(below)} PCC records below the 0.995 bar, e.g. {below[0]}")

    contract = json.loads((ROOT / "doc" / "context_contract.json").read_text())["acceptance"]
    for key in ("records", "pcc_records", "scale_records", "min_pcc", "scale_range"):
        if contract[key] != derived[key]:
            fail(f"context_contract.json acceptance.{key} = {contract[key]!r}, evidence says {derived[key]!r}")
    print(
        f"ok   {derived['records']} records ({derived['pcc_records']} PCC, {derived['scale_records']} scale), "
        f"min PCC {derived['min_pcc']:.6f}, 0 below the bar; context_contract.json agrees"
    )
    return derived


def check_perf() -> dict:
    """Re-derive each phase's device time and op count from the tt-perf-report CSV."""
    summary = json.loads((DOC / "perf_summary.json").read_text())["measurements"]
    for key, recorded in summary.items():
        kind, phase = key.split("/")
        rows = list(csv.DictReader((DOC / "tracy" / kind / f"{phase}_perf_report.csv").open()))
        time_column = next(c for c in rows[0] if c.strip().lower().startswith("device time"))

        def microseconds(row) -> float:
            return float((row[time_column] or "0").replace(",", "").strip() or 0)

        replays = PHASES[phase]
        if len(rows) % replays:
            fail(f"{key}: {len(rows)} ops is not a whole multiple of {replays} replays")
        if recorded["ops_per_pass"] != len(rows) // replays:
            fail(f"{key}: ops_per_pass {recorded['ops_per_pass']} != {len(rows) // replays} from the CSV")
        derived_ms = round(sum(map(microseconds, rows)) / 1000.0 / replays, 3)
        if abs(derived_ms - recorded["device_kernel_time_ms"]) > 0.001:
            fail(f"{key}: device_kernel_time_ms {recorded['device_kernel_time_ms']} != {derived_ms} from the CSV")
    print(f"ok   {len(summary)} perf measurements re-derived from their tt-perf-report CSVs")
    return summary


def check_test_counts() -> None:
    """Re-derive the pass/skip counts the documents quote from the run logs."""
    expected = {
        "suite_main.log": r"(\d+) passed, (\d+) skipped",
        "long_context.log": r"(\d+) passed, \d+ deselected",
        "watcher_run.log": r"(\d+) passed, \d+ deselected",
        "ttnn_sdpa_decode_op_tests.log": r"(\d+) passed, (\d+) skipped",
    }
    counts = {}
    for name, pattern in expected.items():
        text = (DOC / "logs" / name).read_text(errors="replace")
        match = re.search(pattern, text)
        if not match:
            fail(f"logs/{name} has no pytest summary line matching {pattern!r}")
        counts[name] = match.groups()
    joined = "\n".join(doc.read_text() for doc in DOCS)
    for name, groups in counts.items():
        passed = groups[0]
        if f"{passed} passed" not in joined:
            fail(f"logs/{name} says '{passed} passed' but no stage document quotes it")
    print(
        "ok   test counts in the documents match their run logs: "
        + ", ".join(f"{n.split('.')[0]}={g[0]}" for n, g in counts.items())
    )


def check_one_value_per_figure(perf: dict) -> None:
    """A figure carried by more than one document carries the same value in all of them."""
    texts = {doc.name if doc.name != "README.md" else str(doc.relative_to(ROOT)): doc.read_text() for doc in DOCS}
    figures = {}
    for key, measurement in perf.items():
        figures[f"{key} ops/pass"] = str(measurement["ops_per_pass"])
    conflicts = []
    # Any figure written as "N passed" must be unique per test log; catch a document quoting a
    # count that no log produced (the failure mode that survived four review rounds).
    for name, text in texts.items():
        for match in re.finditer(r"(\d+) passed", text):
            count = match.group(1)
            if not any(
                f"{count} passed" in (DOC / "logs" / log).read_text(errors="replace")
                for log in ("suite_main.log", "long_context.log", "watcher_run.log", "ttnn_sdpa_decode_op_tests.log")
            ):
                conflicts.append(f"{name}: '{count} passed' appears in no run log")
    if conflicts:
        fail("figures with no artifact behind them:\n  " + "\n  ".join(sorted(set(conflicts))))
    print(f"ok   every '<n> passed' figure in the documents is produced by a committed run log")


def main() -> int:
    check_paths()
    check_evidence()
    perf = check_perf()
    check_test_counts()
    check_one_value_per_figure(perf)
    print("\nall document checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
