# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Consistency checker for the functional-decoder stage documents.

Rounds of stage review on this stage produced findings in one class over and over: a number is
corrected in one document and not in the others that carry it, or a document cites an artifact
that has moved.  Reviewing harder does not fix that; a check does.  This script is that check::

    python -m models.autoports.qwen_qwen3_6_27b.scripts.check_docs

Every figure it checks is **re-derived from a committed artifact** and then compared against
**every occurrence in the prose**, not just against another JSON summary.  Concretely:

1. *Paths resolve.*  Every markdown link, and every backticked path under a stage artifact
   directory, points at a file that exists.
2. *Evidence agrees with itself.*  ``pcc_evidence.json``'s summary fields agree with its own
   records, no PCC record is below the bar, and ``context_contract.json``'s acceptance block
   matches.
3. *Perf agrees with the profiler output.*  Each phase's device time and op count are re-derived
   by summing the ``Device Time`` column of the ``tt-perf-report`` CSV, and each decode window's
   op-code sequence is checked to repeat with an exact period, which is what makes "all replays
   were captured whole" true.
4. *Test counts come from the run logs*, and each count is bound to the log that produced it by
   requiring the sentence carrying it to name the right thing.
5. *One value per figure.*  For every figure derived above, every occurrence of that figure in
   the prose carries the derived value.  This is the check that closes the drift class: an
   edited headline PCC or a stale perf row fails here.

An earlier revision of this file advertised check 5 and did not implement it; a stage review
demonstrated it passed with a deliberately corrupted README.  The regression test for that is
``--self-test``, which mutates copies of the documents in a temporary directory and asserts the
checker rejects each mutation.

Reads only committed artifacts; opens no device.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: Decode is measured as N trace replays inside one signposted window; prefill is a single pass.
PHASES = {"prefill": 1, "decode": 8}
#: Each run log, the regex that finds its pytest summary, and a word the sentence quoting its
#: pass count must contain.  The keyword is what stops the watcher count being attributed to the
#: suite, or vice versa.
RUN_LOGS = {
    "suite_main.log": (r"(\d+) passed, (\d+) skipped", ("suite", "functional test")),
    "long_context.log": (r"(\d+) passed, \d+ deselected", ("context", "advertised")),
    "watcher_run.log": (r"(\d+) passed, \d+ deselected", ("watcher", "Watcher", "Result:")),
    "ttnn_sdpa_decode_op_tests.log": (
        r"(\d+) passed, (\d+) skipped",
        ("collected", "op suite", "unit-test file", "four files", "sdpa"),
    ),
}


class Failure(Exception):
    pass


def documents(root: Path) -> list[Path]:
    doc = root / "doc" / "functional_decoder"
    return [
        doc / "README.md",
        doc / "work_log.md",
        doc / "probes" / "README.md",
        doc / "watcher" / "WATCHER_AUDIT.md",
        root / "doc" / "context_contract.json",
    ]


def check_paths(root: Path) -> None:
    doc = root / "doc" / "functional_decoder"
    missing = []
    for path in documents(root):
        text = path.read_text()
        for match in re.finditer(r"\]\(([^)#][^)]*)\)", text):
            target = match.group(1).split("#")[0]
            if target.startswith("http"):
                continue
            if not (path.parent / target).exists():
                missing.append(f"{path.name}: link -> {target}")
        for match in re.finditer(r"`((?:\.\./)*(?:doc/|logs/|tracy/|watcher/|probes/)[\w./-]+)`", text):
            target = match.group(1)
            if not any((base / target).exists() for base in (path.parent, root, root / "doc", doc)):
                missing.append(f"{path.name}: path -> {target}")
    if missing:
        raise Failure("unresolved paths:\n  " + "\n  ".join(sorted(set(missing))))
    print(f"ok   every link and stage-artifact path in {len(documents(root))} documents resolves")


def derive_evidence(root: Path) -> dict:
    doc = root / "doc" / "functional_decoder"
    evidence = json.loads((doc / "pcc_evidence.json").read_text())
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
    summary_field = {
        "records": "num_records",
        "pcc_records": "num_pcc_records",
        "scale_records": "num_scale_records",
        "min_pcc": "min_pcc",
    }
    for key, field in summary_field.items():
        if evidence[field] != derived[key]:
            raise Failure(f"pcc_evidence.json {field} disagrees with its own records")
    below = [r for r in pcc if r["value"] < 0.995]
    if below:
        raise Failure(f"{len(below)} PCC records below the 0.995 bar, e.g. {below[0]}")
    contract = json.loads((root / "doc" / "context_contract.json").read_text())["acceptance"]
    for key in derived:
        if contract[key] != derived[key]:
            raise Failure(f"context_contract.json acceptance.{key} = {contract[key]!r}, evidence says {derived[key]!r}")
    print(
        f"ok   {derived['records']} records ({derived['pcc_records']} PCC, {derived['scale_records']} scale), "
        f"min PCC {derived['min_pcc']:.6f}, 0 below the bar; context_contract.json agrees"
    )
    return derived


def derive_perf(root: Path) -> dict:
    doc = root / "doc" / "functional_decoder"
    summary = json.loads((doc / "perf_summary.json").read_text())["measurements"]
    derived = {}
    for key, recorded in summary.items():
        kind, phase = key.split("/")
        rows = list(csv.DictReader((doc / "tracy" / kind / f"{phase}_perf_report.csv").open()))
        time_column = next(c for c in rows[0] if c.strip().lower().startswith("device time"))
        code_column = next(c for c in rows[0] if c.strip().lower() == "op code")
        replays = PHASES[phase]
        if len(rows) % replays:
            raise Failure(f"{key}: {len(rows)} ops is not a whole multiple of {replays} replays")
        period = len(rows) // replays
        codes = [r[code_column].strip() for r in rows]
        # Real periodicity, not just divisibility: op i must match op i % period. This is what
        # makes "all replays were captured whole" a checked statement.
        offenders = [i for i, code in enumerate(codes) if code != codes[i % period]]
        if offenders:
            raise Failure(f"{key}: op sequence is not periodic at {period}; first mismatch at index {offenders[0]}")
        milliseconds = round(
            sum(float((r[time_column] or "0").replace(",", "").strip() or 0) for r in rows) / 1000.0 / replays, 3
        )
        for field, value in (("ops_per_pass", period), ("device_kernel_time_ms", milliseconds)):
            if recorded[field] != value:
                raise Failure(f"{key}: perf_summary.json {field} = {recorded[field]}, CSV says {value}")
        derived[key] = {"ops_per_pass": period, "device_kernel_time_ms": milliseconds, "kind": kind, "phase": phase}
    print(f"ok   {len(derived)} perf measurements re-derived from their tt-perf-report CSVs, replays provably whole")
    return derived


def derive_test_counts(root: Path) -> dict:
    doc = root / "doc" / "functional_decoder"
    counts = {}
    for name, (pattern, _keywords) in RUN_LOGS.items():
        text = (doc / "logs" / name).read_text(errors="replace")
        match = re.search(pattern, text)
        if not match:
            raise Failure(f"logs/{name} has no pytest summary matching {pattern!r}")
        counts[name] = int(match.group(1))
    print(
        "ok   test counts re-derived from the run logs: "
        + ", ".join(f"{n.split('.')[0]}={c}" for n, c in counts.items())
    )
    return counts


def check_prose(root: Path, evidence: dict, perf: dict, counts: dict) -> None:
    """Every occurrence of a derived figure in the prose carries the derived value."""
    problems = []
    for path in documents(root):
        text = path.read_text()
        name = path.name

        # -- headline PCC minimum: "Minimum over ... : 0.998031" / "minimum PCC 0.998031"
        for match in re.finditer(r"[Mm]inimum(?:[^.\n]{0,60}?)([01]\.\d{4,})", text):
            if abs(float(match.group(1)) - evidence["min_pcc"]) > 5e-7:
                problems.append(f"{name}: quotes minimum {match.group(1)}, evidence says {evidence['min_pcc']:.6f}")

        # -- record counts: "N records", "N PCC", "N scale"
        for pattern, key in (
            # "268 records:" / "268 records (" - the evidence total, as the documents write it.
            # Deliberately not a bare "(\d+) records": "0 records below the bar" is a different
            # quantity, and a review section describing a superseded count must describe it
            # rather than quote it (see work_log.md section 13).
            (r"(\d+) records\s*[(:,]", "records"),
            (r"(\d+) PCC records?", "pcc_records"),
            (r"(\d+) scale records?", "scale_records"),
        ):
            for match in re.finditer(pattern, text):
                if int(match.group(1)) != evidence[key]:
                    problems.append(f"{name}: quotes '{match.group(0)}', evidence says {evidence[key]}")

        # -- perf table rows: | `kind` | phase ... | ops | **X ms** | wall |
        for match in re.finditer(
            r"\|\s*`(linear_attention|full_attention)`\s*\|\s*([a-z ,0-9]*?(?:prefill|decode)[^|]*)\|"
            r"\s*(\d+)\s*\|\s*\**([\d.]+) ms\**\s*\|",
            text,
        ):
            kind, phase_text, ops, milliseconds = match.groups()
            phase = "prefill" if "prefill" in phase_text else "decode"
            want = perf[f"{kind}/{phase}"]
            if int(ops) != want["ops_per_pass"]:
                problems.append(f"{name}: {kind}/{phase} row says {ops} ops, CSV says {want['ops_per_pass']}")
            if abs(float(milliseconds) - want["device_kernel_time_ms"]) > 0.011:
                problems.append(
                    f"{name}: {kind}/{phase} row says {milliseconds} ms, CSV says {want['device_kernel_time_ms']}"
                )

        # -- "<n> passed" must be a count some run log produced, and the sentence carrying it
        #    must name the right run.
        for match in re.finditer(r"(\d+) passed", text):
            value = int(match.group(1))
            line = text[text.rfind("\n", 0, match.start()) + 1 : text.find("\n", match.end())]
            owners = [log for log, count in counts.items() if count == value]
            if not owners:
                problems.append(f"{name}: '{match.group(0)}' matches no run log")
                continue
            if not any(any(word in line for word in RUN_LOGS[log][1]) for log in owners):
                problems.append(
                    f"{name}: '{match.group(0)}' is {'/'.join(o.split('.')[0] for o in owners)}'s count, but the "
                    f"sentence names none of {[w for o in owners for w in RUN_LOGS[o][1]]}"
                )
    if problems:
        raise Failure("prose disagrees with the artifacts:\n  " + "\n  ".join(sorted(set(problems))))
    print("ok   every derived figure quoted in the prose carries the derived value")


def run(root: Path) -> None:
    check_paths(root)
    evidence = derive_evidence(root)
    perf = derive_perf(root)
    counts = derive_test_counts(root)
    check_prose(root, evidence, perf, counts)


def self_test() -> int:
    """Prove the checker rejects the mutations it claims to catch."""
    doc = ROOT / "doc" / "functional_decoder"
    mutations = [
        ("work_log.md", lambda s: s.replace("| 151.24 ms |", "| 251.24 ms |", 1), "a wrong perf row in the work log"),
        ("README.md", lambda s: s.replace("0.998031.**", "0.999500.**", 1), "a wrong PCC minimum in the README"),
        ("README.md", lambda s: s.replace("268 records", "999 records", 1), "a wrong record count"),
        (
            "README.md",
            lambda s: s.replace("**9 passed**", "**57 passed**", 1),
            "the suite's count attributed to watcher",
        ),
        ("README.md", lambda s: s.replace("](work_log.md)", "](work_log_missing.md)", 1), "a dead link"),
    ]
    failures = []
    for filename, mutate, description in mutations:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "model"
            shutil.copytree(ROOT, copy, ignore=shutil.ignore_patterns("__pycache__"))
            target = copy / "doc" / "functional_decoder" / filename
            before = target.read_text()
            after = mutate(before)
            if after == before:
                failures.append(f"self-test could not apply mutation: {description}")
                continue
            target.write_text(after)
            try:
                run(copy)
            except Failure:
                print(f"ok   rejected: {description}")
                continue
            failures.append(f"NOT rejected: {description}")
    assert doc.exists()
    if failures:
        print("FAIL self-test:\n  " + "\n  ".join(failures))
        return 1
    print("\nself-test passed: the checker rejects every mutation it claims to catch")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="prove the checks are not vacuous")
    args = parser.parse_args()
    if args.self_test:
        print("== checking the committed tree ==")
        try:
            run(ROOT)
        except Failure as failure:
            print(f"FAIL {failure}")
            return 1
        print("\n== mutating copies, expecting each to be rejected ==")
        return self_test()
    try:
        run(ROOT)
    except Failure as failure:
        print(f"FAIL {failure}")
        return 1
    print("\nall document checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
