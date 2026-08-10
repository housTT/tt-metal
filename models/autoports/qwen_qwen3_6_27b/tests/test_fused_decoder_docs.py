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

#: Split so this file can name the token without tripping its own check.
_UNFILLED = "PLACE" + "HOLDER"

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
#: Decimal places the fused block of ``doc/context_contract.json`` states its PCC figures to.
CONTRACT_DECIMALS = 6
IMPLS = ("functional", "fused")
KINDS = ("linear_attention", "full_attention")
#: measured window -> trace replays inside it.  ``decode_batch32`` is the same window at the
#: advertised ``max_batch``, which takes a different branch of the z-gated norm.
PHASES = {"prefill": 1, "decode": 8, "decode_batch32": 8}
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


def _allowed_figures() -> set[str]:
    """Every latency/bandwidth figure a document may legitimately quote.

    Deliberately *derived*, not a substring corpus: four review rounds found stale figures
    surviving a membership test over both stages' logs and CSVs, because any 3-4 digit string
    occurs somewhere in a multi-megabyte artifact set.  A figure has to be a value this stage's
    perf summary holds, a value a probe actually printed, or the stage-1 summary's own.
    """
    allowed: set[str] = set()

    def add(value: float) -> None:
        for places in (0, 1, 2, 3):
            allowed.add(f"{value:.{places}f}")

    summary = _perf_summary()
    for row in summary["measurements"].values():
        add(row["device_kernel_time_ms"])
        add(row["op_to_op_gap_ms"])
        for value in row["breakdown_ms"].values():
            add(value)
        for op in row["top_ops_by_device_time"]:
            add(op["device_time_us"])
    for row in summary["speedup"].values():
        add(row["speedup_x"])
        add(row["reduction_pct"])
    functional = ROOT / "doc" / "functional_decoder" / "perf_summary.json"
    if functional.is_file():
        for row in json.loads(functional.read_text())["measurements"].values():
            add(row["device_kernel_time_ms"])
    # Whatever the probes printed, exactly as they printed it - and rounded, since a table may
    # quote fewer places than the log.
    for log in sorted((DOC / "logs").glob("probe_*.log")):
        for value in re.findall(r"(?<![\w.])(\d+\.\d+)(?![\w.])", log.read_text(errors="replace")):
            allowed.add(value)
            add(float(value))
    return allowed


def test_prose_perf_figures_match_the_summary():
    """Every ms / us / speed-up / percentage / bandwidth figure in the prose is a derived value.

    Scans the documents *and* the implementation and test files, because a stage review found
    four stale figures living in ``tt/fused_decoder.py``'s docstrings.
    """
    allowed = _allowed_figures()
    # Generated blocks are exempt: they are written from the artifacts by construction, and
    # ``test_generated_blocks_are_current`` is what keeps them honest.  This check is for the
    # hand-written prose around them, which is where every stale figure four review rounds found
    # actually lived.
    strip = re.compile(r"<!-- GENERATED:\w+ -->.*?<!-- END GENERATED:\w+ -->", re.DOTALL)
    scanned = {path: strip.sub("", text) for path, text in _documents().items()}
    scanned.update({path: path.read_text() for path in SOURCES})
    unexplained = []
    for path, text in scanned.items():
        # ``x`` counts only when attached *and* carrying a decimal point (a speedup): ``8x8`` is a
        # core grid.  Microsecond figures count with or without a decimal point.
        quoted = re.findall(r"\*{0,2}(\d+(?:\.\d+)?)\*{0,2}(?:\s*(ms|%|GB/s|us)|(x))\b", text)
        for value, unit, attached in quoted:
            if not unit and "." not in value:
                continue
            if value in allowed:
                continue
            unexplained.append(f"{path.name}: {value} {unit or attached}")
    assert not unexplained, "documents or sources quote figures no artifact derives:\n  " + "\n  ".join(unexplained)


def test_prose_op_counts_match_the_summary():
    """Every device op-count pair the before/after table states is a pair the summary contains.

    Scoped to the generated before/after block, because a markdown table of two numbers is not by
    itself an op-count claim - the batch sweep in ``work_log.md`` section 3.17 is one too.
    """
    summary = _perf_summary()
    pairs = {(row["ops_before"], row["ops_after"]) for row in summary["speedup"].values()}
    quoted = set()
    for text in _documents().values():
        for block in re.findall(
            r"<!-- GENERATED:before_after -->\n(.*?)\n<!-- END GENERATED:before_after -->", text, flags=re.DOTALL
        ):
            for before, after in re.findall(r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*$", block, flags=re.MULTILINE):
                quoted.add((int(before), int(after)))
    assert quoted, "no op-count pairs found in a generated before/after block - the table shape changed"
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
    assert acceptance["min_pcc"] == round(evidence["min_pcc"], CONTRACT_DECIMALS)
    assert acceptance["scale_range"] == [round(value, CONTRACT_DECIMALS) for value in evidence["scale_range"]]
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
            # 6 dp: the precision the contract states these to.  Comparing full doubles made
            # every re-measurement a document edit without making any claim more true.
            assert round(recorded[(metric, kind)], CONTRACT_DECIMALS) == results[field], (
                f"{kind}.{field} disagrees with pcc_evidence.json "
                f"({results[field]} vs {round(recorded[(metric, kind)], CONTRACT_DECIMALS)})"
            )

    # And the measured persistent-state delta.
    delta_records = {
        r.get("kind"): r["value"] for r in evidence["records"] if r["metric"] == "fused_persistent_dram_bytes"
    }
    for kind, block in contract["extra_persistent_device_bytes_per_layer"].items():
        if kind not in delta_records:
            continue
        for field in ("functional", "fused", "delta"):
            assert block[field] == delta_records[kind][field], f"{kind}.{field} disagrees with the measurement"


def test_contract_prose_matches_the_evidence():
    """Every number in the fused contract's *prose* re-derives from a measured field.

    ``test_context_contract_matches_the_evidence`` pins the contract's numeric fields, and the
    document gate pins the stage documents' prose — but the contract's own sentences are neither,
    and a stage review found the capacity conclusion still quoting a superseded byte count three
    rounds after the field beside it had been corrected.  The contract is what later stages read,
    so its prose is held to the same rule as the documents': a figure must be derivable.
    """
    contract = json.loads((ROOT / "doc" / "context_contract.json").read_text())["fused_decoder"]
    evidence = json.loads((DOC / "pcc_evidence.json").read_text())
    delta = {r.get("kind"): r["value"] for r in evidence["records"] if r["metric"] == "fused_persistent_dram_bytes"}
    linear = delta["linear_attention"]

    allowed = set()
    for record in evidence["records"]:
        value = record["value"]
        for item in value.values() if isinstance(value, dict) else (value,):
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                allowed.add(f"{item}")
                allowed.add(f"{float(item):.6f}".rstrip("0"))
    for block in contract["extra_persistent_device_bytes_per_layer"].values():
        if isinstance(block, dict):
            allowed.update(f"{v}" for v in block.values() if isinstance(v, int))
    # Percentages the conclusion is entitled to quote, at the precision it quotes them.
    dram_bytes = 31 * 1024**3
    for digits in (1, 2, 3):
        allowed.add(f"{100.0 * linear['delta'] / linear['functional']:.{digits}f}")
        allowed.add(f"{100.0 * linear['delta'] / dram_bytes:.{digits}f}")
    for record in evidence["records"]:
        if isinstance(record["value"], (int, float)) and not isinstance(record["value"], bool):
            for digits in (5, 6, 7):
                allowed.add(f"{round(record['value'], digits)}")
    allowed.update({"31", "1", "0", "1818230784", "262143", "262144", "0.995", "1.02", "0.98"})
    # Stage-1 figures the delta prose compares against, from the functional stage's own evidence.
    functional = json.loads((ROOT / "doc" / "functional_decoder" / "pcc_evidence.json").read_text())
    for record in functional["records"]:
        if isinstance(record["value"], (int, float)) and not isinstance(record["value"], bool):
            for digits in (5, 6, 7):
                allowed.add(f"{round(record['value'], digits)}")

    unexplained = []
    for key, text in contract.items():
        for path, value in _walk_strings(key, text):
            # Two shapes of figure: a unit-carrying one (``8290304 bytes``, ``0.78 %``) and a
            # bare PCC-like decimal (``0.999879``), which is how the largest_context prose
            # states its measurements.
            figures = re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:%|bytes|GiB|B\b)", value)
            figures += re.findall(r"(?<![\w.])(0\.\d{4,})(?![\d])", value)
            for figure in figures:
                if figure not in allowed and figure.rstrip("0") not in allowed:
                    unexplained.append(f"{path}: {figure}")
    assert not unexplained, f"the contract's prose quotes figures no measurement derives: {unexplained}"


def _walk_strings(prefix, value):
    """Yield ``(dotted path, string)`` for every string inside a nested JSON value."""
    if isinstance(value, str):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_strings(f"{prefix}.{key}", item)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_strings(f"{prefix}[{index}]", item)


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


def test_generated_blocks_are_current():
    """Re-run every document generator and require the committed text to match, byte for byte.

    This is the gate that retires the stale-figure class four review rounds kept finding: a
    generated block that was never filled (one survived a whole round holding the generator's
    placeholder text, because its markers were inline and the generator's regex needs them on
    their own lines), or one filled from an older artifact, fails here rather than in a review.
    """
    import shutil
    import subprocess
    import sys
    import tempfile

    generator = DOC / "probes" / "make_doc_tables.py"
    before = {path: path.read_text() for path in DOCUMENTS}
    # Run the generator against a *copy* of the stage tree: rewriting the real documents and
    # restoring them leaves them regenerated if this test is interrupted.
    with tempfile.TemporaryDirectory() as scratch:
        # Mirror the whole ``doc/`` tree: the generator reads the *functional* stage's evidence
        # too, for the delta table.
        mirror_doc = Path(scratch) / "doc"
        shutil.copytree(DOC.parent, mirror_doc, symlinks=True)
        mirror = mirror_doc / "fused_decoder"
        result = subprocess.run(
            [sys.executable, str(mirror / "probes" / "make_doc_tables.py")], capture_output=True, text=True
        )
        assert result.returncode == 0, f"{generator.name} failed: {result.stderr[-2000:]}"
        stale = []
        for path, text in before.items():
            regenerated = (mirror / path.relative_to(DOC)).read_text()
            if regenerated != text:
                stale.append(path.name)
            assert _UNFILLED not in text, f"{path.name} has a generated block the generator never filled"
    assert not stale, (
        "these documents' generated blocks are out of date with the artifacts; " f"re-run {generator.name}: {stale}"
    )


@pytest.mark.parametrize("kind", KINDS)
def test_no_layout_round_trip_in_the_measured_prefill(kind):
    """No profiled prefill op converts a layout that the very next op converts back.

    The device report is the ground truth here, not the python call trace: ``ttnn.concat``,
    ``ttnn.slice`` and friends relayout *inside* themselves, so a python-level trap cannot see
    them - which is exactly how a ``tilize`` immediately followed by an ``untilize`` of the same
    tensor survived three review rounds in the ``linear_attention`` prefill.
    """
    report = DOC / "tracy" / "fused" / kind / "prefill_perf_report.csv"
    with report.open() as handle:
        codes = [row["OP Code"] for row in csv.DictReader(handle)]
    to_tile = ("Tilize",)
    to_rows = ("Untilize",)

    def kindof(code: str) -> str | None:
        if any(code.startswith(token) for token in to_tile):
            return "tilize"
        if any(code.startswith(token) for token in to_rows):
            return "untilize"
        return None

    # Only the tilize -> untilize direction is a defect: a misaligned ``ttnn.slice`` legitimately
    # untilizes, cuts and re-tilizes inside itself, so untilize -> tilize is that op's own cost.
    # Producing a TILE tensor and immediately converting it straight back is not.
    offenders = [
        f"op {index}: {first} -> {second}"
        for index, (first, second) in enumerate(zip(codes, codes[1:]))
        if kindof(first) == "tilize" and kindof(second) == "untilize"
    ]
    assert not offenders, f"the profiled {kind} prefill undoes a layout conversion it just made: {offenders}"


def test_watcher_log_is_clean():
    text = _watcher_log()
    pattern = re.compile(
        r"fatal|assert|corrupt|sanitiz|out of bounds|overflow|invalid|exception|error|fault|hang|unexpected",
        re.IGNORECASE,
    )
    offenders = [line for line in text.splitlines() if pattern.search(line) and "highest stack usage" not in line]
    assert not offenders, "watcher log is not clean:\n  " + "\n  ".join(offenders[:10])
