# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Consistency gate for the fused-decoder stage documents.

The functional stage learned this the hard way: review round after review round produced the
same class of finding - a number corrected in one document and not in the others that carry it,
or a document citing an artifact that has moved. Reviewing harder does not fix that; a check
does. ``scripts/check_docs.py`` is that check for the functional stage; this file is it for the
fused stage, as a test so it runs with the rest of the suite.

Every figure is **re-derived from a committed artifact** and then compared against **every**
occurrence of it in the prose:

1. *Paths resolve* - every markdown link, and every backticked path that looks like a stage
   artifact, points at a file that exists.
2. *Perf agrees with the profiler output* - each row of ``perf_summary.json`` is re-derived by
   summing the ``Device Time`` column of the matching ``tt-perf-report`` CSV, and each decode
   window's op-code sequence is checked to repeat with an exact period, which is what makes
   "all replays were captured whole" true rather than assumed.
3. *Prose agrees with the perf summary* - every device-time, op-count and speedup figure quoted
   in the README or the work log is the summary's value.
4. *Evidence agrees with itself* - ``pcc_evidence.json``'s summary fields agree with its own
   records, and no PCC record is below the stage bar.
5. *Test counts come from the run logs* they are attributed to.

Opens no device; reads only committed artifacts.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
from pathlib import Path

import pytest

from models.autoports.qwen_qwen3_6_27b.tests.harness import PCC_BAR

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc" / "fused_decoder"
DOCUMENTS = (DOC / "README.md", DOC / "work_log.md", DOC / "probes" / "README.md")
#: The implementation and its tests carry perf figures in comments and docstrings too, and a
#: stage review found four of them unsupported by any artifact.  They are held to the same rule as
#: the documents: a quoted figure must exist in a committed artifact.
SOURCES = (
    ROOT / "tt" / "fused_decoder.py",
    ROOT / "tests" / "test_fused_decoder.py",
    ROOT / "tests" / "test_fused_decoder_perf.py",
    ROOT / "tests" / "test_fused_decoder_docs.py",
)
IMPLS = ("functional", "fused")
KINDS = ("linear_attention", "full_attention")
PHASES = {"prefill": 1, "decode": 8}
#: Run log, the regex that finds its pytest summary, and words the sentence quoting its pass
#: count must contain. The keyword is what stops the watcher count being attributed to the main
#: suite, or vice versa.
RUN_LOGS = {
    "suite_main.log": (r"(\d+) passed, (\d+) skipped", ("suite", "fused test")),
    "long_context.log": (r"(\d+) passed, \d+ deselected", ("context", "advertised")),
    "watcher_run.log": (r"(\d+) passed, \d+ deselected", ("watcher", "Watcher")),
}


def _documents() -> dict[Path, str]:
    return {path: path.read_text() for path in DOCUMENTS}


def _watcher_log() -> str:
    """The watcher log, whether committed verbatim or gzipped past the repo's 500 KB limit."""
    plain = DOC / "watcher" / "generated" / "watcher" / "watcher.log"
    if plain.is_file():
        return plain.read_text(errors="replace")
    packed = plain.with_suffix(".log.gz")
    assert packed.is_file(), f"missing {plain} and {packed}"
    return gzip.decompress(packed.read_bytes()).decode(errors="replace")


def _perf_summary() -> dict:
    return json.loads((DOC / "perf_summary.json").read_text())


def test_documents_exist():
    for path in DOCUMENTS:
        assert path.is_file(), f"missing stage document {path}"
        assert path.read_text().strip(), f"empty stage document {path}"


def test_every_cited_path_resolves():
    """Markdown links and backticked artifact paths point at files that exist."""
    missing = []
    for path, text in _documents().items():
        base = path.parent
        for target in re.findall(r"\]\(([^)#]+)\)", text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (base / target).resolve().exists():
                missing.append(f"{path.name}: link {target}")
        for token in re.findall(r"`([^`\s]+)`", text):
            # Only paths that name a stage artifact directory, so op names and flags are skipped.
            if not re.search(r"(^|/)(logs|tracy|probes|watcher|doc|tests|tt)/", token):
                continue
            # Globs and <placeholders> name a family of files, not one file.
            if any(character in token for character in "*{}<>"):
                continue
            if token.endswith((".py", ".md", ".json", ".log", ".csv", ".sh", ".txt", ".gz", ".png")):
                candidates = [base / token, ROOT / token, ROOT.parents[2] / token]
                if not any(candidate.exists() for candidate in candidates):
                    missing.append(f"{path.name}: path {token}")
    assert not missing, "documents cite artifacts that do not exist:\n  " + "\n  ".join(missing)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("phase", sorted(PHASES))
def test_perf_summary_rederives_from_the_report(impl, kind, phase):
    """Device time and op counts in perf_summary.json come out of the tt-perf-report CSV."""
    summary = _perf_summary()["measurements"][f"{impl}/{kind}/{phase}"]
    report = DOC / "tracy" / impl / kind / f"{phase}_perf_report.csv"
    assert report.is_file(), f"missing {report}"
    with report.open() as handle:
        rows = list(csv.DictReader(handle))
    replays = PHASES[phase]
    total_us = sum(float(row["Device Time"] or 0) for row in rows)
    assert summary["ops_in_window"] == len(rows)
    assert summary["ops_per_pass"] == len(rows) // replays
    assert summary["device_kernel_time_ms"] == round(total_us / replays / 1000.0, 3)

    # The window is only comparable if every replay was captured whole, which is exactly the
    # claim that the op-code sequence repeats with the expected period.
    codes = [row["OP Code"] for row in rows]
    if replays > 1:
        assert len(codes) % replays == 0, f"{len(codes)} ops is not divisible by {replays} replays"
        per = len(codes) // replays
        for index, code in enumerate(codes):
            assert code == codes[index % per], f"op {index} ({code}) breaks the period of {per}"
    assert "BROKEN" not in summary["periodicity_check"]


def test_speedup_block_is_consistent():
    summary = _perf_summary()
    for key, row in summary["speedup"].items():
        kind, phase = key.split("/")
        before = summary["measurements"][f"functional/{kind}/{phase}"]
        after = summary["measurements"][f"fused/{kind}/{phase}"]
        assert row["device_ms_before"] == before["device_kernel_time_ms"]
        assert row["device_ms_after"] == after["device_kernel_time_ms"]
        assert row["ops_before"] == before["ops_per_pass"]
        assert row["ops_after"] == after["ops_per_pass"]
        assert row["speedup_x"] == round(row["device_ms_before"] / row["device_ms_after"], 3)
        # The stage's contract: the fused graph must be faster, not merely smaller.
        assert row["device_ms_after"] < row["device_ms_before"], f"{key} did not get faster"
        assert row["ops_after"] < row["ops_before"], f"{key} did not get smaller"


def _artifact_corpus() -> str:
    """Every committed artifact a fused-stage document may legitimately quote a number from.

    That spans both stages: the *before* half of every perf row is the functional stage's
    measurement, and the functional documents' own artifacts are where it lives.
    """
    parts = []
    for stage in ("fused_decoder", "functional_decoder"):
        root = ROOT / "doc" / stage
        for path in sorted(root.glob("logs/**/*.log")):
            parts.append(path.read_text(errors="replace"))
        for path in sorted(root.glob("*.json")):
            parts.append(path.read_text(errors="replace"))
        for path in sorted(root.glob("tracy/**/*_perf_report.csv")):
            parts.append(path.read_text(errors="replace"))
    return "\n".join(parts)


def test_prose_perf_figures_match_the_summary():
    """Every ms / speed-up / percentage figure in the prose is some committed artifact's number.

    Catches the drift class directly: a figure updated in ``perf_summary.json`` or re-measured in
    a probe log, and not updated in the document that quotes it.  The check is broad rather than
    positional - a value that exists somewhere in the corpus passes even if it is quoted in the
    wrong place - which is exactly what caught the stale probe figures this file was written for.
    """
    summary = _perf_summary()
    allowed = set()
    for row in summary["measurements"].values():
        allowed.add(f"{row['device_kernel_time_ms']:.3f}")
        allowed.add(f"{row['device_kernel_time_ms']:.2f}")
        for value in row["breakdown_ms"].values():
            allowed.add(f"{value:.3f}")
            allowed.add(f"{value:.2f}")
    for row in summary["speedup"].values():
        allowed.add(f"{row['speedup_x']:.2f}")
        allowed.add(f"{row['reduction_pct']:.1f}")
    corpus = _artifact_corpus()

    scanned = dict(_documents())
    scanned.update({path: path.read_text() for path in SOURCES})
    unexplained = []
    for path, text in scanned.items():
        # ms / x / % / GB/s figures, and microsecond figures whether or not they have a decimal
        # point: a stage review found the integer ones (an 8836-line watcher count, several `us`
        # timings) escaping an earlier decimals-only rule.
        quoted = re.findall(r"\*{0,2}(\d+(?:\.\d+)?)\*{0,2}\s*(ms|x|%|GB/s|us)\b", text)
        for value, unit in quoted:
            if value in allowed or value in corpus:
                continue
            unexplained.append(f"{path.name}: {value} {unit}")
    assert not unexplained, "prose quotes figures that are in no committed artifact:\n  " + "\n  ".join(unexplained)


def test_prose_op_counts_match_the_summary():
    """Every ``N -> M`` op-count claim in the prose is a pair the perf summary actually contains."""
    summary = _perf_summary()
    pairs = {(row["ops_before"], row["ops_after"]) for row in summary["speedup"].values()}
    quoted = set()
    for text in _documents().values():
        for before, after in re.findall(r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*$", text, flags=re.MULTILINE):
            quoted.add((int(before), int(after)))
    assert quoted, "no op-count pairs found in the documents - the table shape changed"
    assert quoted <= pairs, f"documents quote op-count pairs the summary does not have: {sorted(quoted - pairs)}"


def test_prose_pcc_figures_come_from_an_artifact():
    """Every 6-decimal correlation-shaped figure in the documents is some artifact's number.

    The perf gate covers ms/us/x/%; this covers the other half of what these documents quote.
    The corpus spans both stages' evidence, run logs and probe logs, because a fused-stage
    document legitimately quotes the functional stage's figures in its delta table.
    """
    corpus = _artifact_corpus()
    unexplained = []
    for path, text in _documents().items():
        for value in re.findall(r"(?<![\d.])([01]\.\d{6})(?![\d])", text):
            if value in corpus:
                continue
            # Also accept a rounded artifact value, which is how a table quotes 6 places.
            target = float(value)
            if any(abs(target - float(m)) < 5e-7 for m in re.findall(r"(?<![\d.])([01]\.\d{6,})", corpus)):
                continue
            unexplained.append(f"{path.name}: {value}")
    assert not unexplained, "documents quote PCC figures that are in no committed artifact:\n  " + "\n  ".join(
        unexplained
    )


def test_pcc_evidence_is_self_consistent():
    evidence = json.loads((DOC / "pcc_evidence.json").read_text())
    records = evidence["records"]
    numeric = [r for r in records if isinstance(r["value"], (int, float)) and not isinstance(r["value"], bool)]
    pcc_records = [r for r in numeric if not r["metric"].endswith("_scale")]
    scale_records = [r for r in numeric if r["metric"].endswith("_scale")]
    assert evidence["num_records"] == len(records)
    assert evidence["num_pcc_records"] == len(pcc_records)
    assert evidence["num_scale_records"] == len(scale_records)
    assert pcc_records, "no PCC records were collected"
    assert evidence["min_pcc"] == min(r["value"] for r in pcc_records)
    assert evidence["min_pcc"] >= PCC_BAR, f"a recorded PCC is below the {PCC_BAR} bar"
    if scale_records:
        low, high = evidence["scale_range"]
        assert low == min(r["value"] for r in scale_records)
        assert high == max(r["value"] for r in scale_records)
        assert 0.98 <= low <= high <= 1.02, "a recorded scale ratio is outside the stage tolerance"


def test_run_log_counts_match_the_prose():
    """Each pass count quoted in the documents comes from the log it is attributed to."""
    documents = _documents()
    for name, (pattern, keywords) in RUN_LOGS.items():
        log = DOC / "logs" / name
        assert log.is_file(), f"missing run log {log}"
        matches = re.findall(pattern, log.read_text(errors="replace"))
        assert matches, f"{name} has no pytest summary line"
        passed = int(matches[-1][0] if isinstance(matches[-1], tuple) else matches[-1])
        quoted = False
        for text in documents.values():
            for line in text.splitlines():
                if any(word in line for word in keywords) and re.search(rf"\b{passed}\b", line):
                    quoted = True
        assert quoted, f"no document states {name}'s pass count ({passed}) in a sentence naming {keywords}"


def test_context_contract_matches_the_evidence():
    """``doc/context_contract.json``'s fused block agrees with ``pcc_evidence.json`` field by field.

    The contract is the artifact later stages read, and it carries a copy of the acceptance
    summary; a copy is exactly what drifts.
    """
    contract = json.loads((ROOT / "doc" / "context_contract.json").read_text())["fused_decoder"]
    evidence = json.loads((DOC / "pcc_evidence.json").read_text())
    acceptance = contract["acceptance"]
    assert acceptance["pcc_bar"] == PCC_BAR
    assert acceptance["records"] == evidence["num_records"]
    assert acceptance["pcc_records"] == evidence["num_pcc_records"]
    assert acceptance["scale_records"] == evidence["num_scale_records"]
    assert acceptance["min_pcc"] == evidence["min_pcc"]
    assert acceptance["scale_range"] == evidence["scale_range"]
    assert acceptance["pcc_records_below_bar"] == 0

    # The long-context figures the contract quotes are records in the evidence, not prose.
    recorded = {
        (record["metric"], record.get("kind")): record["value"]
        for record in evidence["records"]
        if isinstance(record["value"], float)
    }
    for kind, results in contract["largest_context_tested"]["results"].items():
        for field, metric in (
            ("prefill_tail_8192_pcc", "fused_full_context_prefill_tail_pcc"),
            ("prefill_tail_256_pcc", "fused_full_context_prefill_tail_pcc"),
            ("decode_at_262143_pcc", "fused_full_context_decode_pcc"),
            ("prefill_tail_scale", "fused_full_context_prefill_tail_scale"),
            ("decode_scale", "fused_full_context_decode_scale"),
            ("conv_state_pcc", "fused_full_context_conv_state_pcc"),
            ("recurrent_state_pcc", "fused_full_context_recurrent_state_pcc"),
            ("paged_k_cache_pcc", "fused_full_context_paged_k_cache_pcc"),
            ("paged_v_cache_pcc", "fused_full_context_paged_v_cache_pcc"),
        ):
            if field not in results:
                continue
            assert recorded[(metric, kind)] == results[field], f"{kind}.{field} disagrees with pcc_evidence.json"

    # And the measured persistent-state delta.
    delta_records = {
        r.get("kind"): r["value"] for r in evidence["records"] if r["metric"] == "fused_persistent_dram_bytes"
    }
    for kind, block in contract["extra_persistent_device_bytes_per_layer"].items():
        if kind not in delta_records:
            continue
        for field in ("functional", "fused", "delta"):
            assert block[field] == delta_records[kind][field], f"{kind}.{field} disagrees with the measurement"


def test_watcher_audit_matches_its_artifacts():
    """Every quantity ``WATCHER_AUDIT.md`` states re-derives from the committed watcher artifacts.

    The audit claims a line count, a dump count, a category histogram and a pass/deselect count.
    Without this, only the "no offender lines" grep was pinned, and the audit could describe an
    earlier run while the committed log described a later one - which is exactly what happened
    once in this stage.
    """
    audit = (DOC / "watcher" / "WATCHER_AUDIT.md").read_text()
    lines = _watcher_log().splitlines()
    assert re.search(rf"\b{len(lines)} lines\b", audit), f"audit does not state the log's {len(lines)} lines"
    dumps = sum(1 for line in lines if line.startswith("Dump"))
    assert re.search(rf"\b{dumps}\b\s*`Dump`", audit), f"audit does not state the log's {dumps} Dump lines"

    histogram: dict[str, int] = {}
    for line in lines:
        token = line.split(" ")[0] if line else ""
        histogram[token] = histogram.get(token, 0) + 1
    for token, count in sorted(histogram.items(), key=lambda item: -item[1])[:6]:
        if f"{count} {token}" not in audit:
            continue  # the audit quotes the top of the histogram, not all of it
        assert f"{count} {token}" in audit

    run = (DOC / "logs" / "watcher_run.log").read_text(errors="replace")
    summary = re.findall(r"(\d+) passed, (\d+) deselected, \d+ warnings in ([\d.]+)s", run)
    assert summary, "watcher_run.log has no pytest summary"
    passed, deselected, seconds = summary[-1]
    for value in (passed, deselected, seconds):
        assert value in audit, f"audit does not state the run's {value}"

    # The command the audit prints must be the one the work log tells a reader to run.  Compare
    # the fenced block that carries it, not the first prose mention of the env var.
    marker = "TT_METAL_WATCHER=10"

    def fenced(text: str) -> str:
        blocks = [b for b in re.findall(r"```(?:bash)?\n(.*?)```", text, flags=re.DOTALL) if marker in b]
        return " ".join(" ".join(blocks).split())

    audit_cmd = fenced(audit)
    assert marker in audit_cmd, "the audit has no fenced watcher command"
    # The audit's block is only the watcher recipe; the documents' blocks list several commands,
    # so require containment rather than equality.
    for name in ("README.md", "work_log.md"):
        other = fenced((DOC / name).read_text())
        if marker not in other:
            continue
        assert audit_cmd in other, f"{name}'s watcher command differs from the audit's"


def test_readme_watcher_claims_match_the_audit():
    """Any watcher count the README quotes is the generated audit's, not an earlier run's."""
    audit = (DOC / "watcher" / "WATCHER_AUDIT.md").read_text()
    readme = (DOC / "README.md").read_text()
    for line in readme.splitlines():
        if "WATCHER_AUDIT" not in line:
            continue
        for number in re.findall(r"(?<![\w.])(\d{3,})(?![\w])", line):
            assert number in audit, f"README quotes {number} next to the watcher audit; the audit does not"


def test_watcher_log_is_clean():
    text = _watcher_log()
    pattern = re.compile(
        r"fatal|assert|corrupt|sanitiz|out of bounds|overflow|invalid|exception|error|fault|hang|unexpected",
        re.IGNORECASE,
    )
    offenders = [line for line in text.splitlines() if pattern.search(line) and "highest stack usage" not in line]
    assert not offenders, "watcher log is not clean:\n  " + "\n  ".join(offenders[:10])
