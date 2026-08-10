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
import subprocess
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
    # Byte counts come from the evidence file's measured records, not from a perf table.
    evidence = json.loads((DOC / "pcc_evidence.json").read_text())
    for record in evidence["records"]:
        value = record["value"]
        for item in value.values() if isinstance(value, dict) else (value,):
            if isinstance(item, int) and not isinstance(item, bool):
                allowed.add(str(item))
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
        quoted = re.findall(r"\*{0,2}(\d+(?:\.\d+)?)\*{0,2}(?:\s*(ms|%|GB/s|us|bytes)|(x))\b", text)
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
            # Two shapes of figure: a unit-carrying one (a byte count, a percentage) and a
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


def test_every_cited_test_name_exists():
    """Every ``test_...`` identifier the documents or sources quote is a real, collected test.

    ``test_every_cited_path_resolves`` resolves file paths; nothing resolved test *names*, so a
    renamed test kept being cited in four places across two rounds.  The names are collected from
    the two test modules by parsing them, so this needs no pytest run.
    """
    defined = set()
    for module in ("test_fused_decoder.py", "test_fused_decoder_docs.py", "test_fused_decoder_perf.py"):
        source = (ROOT / "tests" / module).read_text()
        defined.update(re.findall(r"^def (test_\w+)", source, re.MULTILINE))
    # Stage 1's suite is cited too, and so is the shared harness's.
    for module in (ROOT / "tests").glob("test_*.py"):
        defined.update(re.findall(r"^def (test_\w+)", module.read_text(), re.MULTILINE))

    modules = {path.stem for path in (ROOT / "tests").glob("test_*.py")}
    unknown = {}
    for path, text in list(_documents().items()) + [(path, path.read_text()) for path in SOURCES]:
        for name in re.findall(r"\b(test_[a-z0-9_]+)\b", text):
            # A module stem (``test_fused_decoder``), a wildcarded family (``test_traced_decode_*``,
            # whose captured stem ends in ``_``) and the report directory are citations of things
            # other than one test.
            if name in defined or name in modules or name.endswith("_") or name == "test_reports":
                continue
            unknown.setdefault(path.name, set()).add(name)
    assert not unknown, f"these cited test names are not defined in tests/: {unknown}"


def test_every_artifact_the_gate_reads_is_tracked_by_git():
    """Every artifact this stage's documents and tests read is committed, not just on disk.

    The repository ignores ``*.log`` and ``*.csv``, so stage artifacts have to be force-added.
    Three rounds of evidence were regenerated and never added, and because the files were present
    in the working tree every gate stayed green while a fresh clone of the same commit could not
    even run the generators.  This is the check that makes "committed artifact" mean it.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", str(DOC.relative_to(ROOT.parents[2]))],
        cwd=ROOT.parents[2],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    tracked = {name for name in tracked if name}

    required = [DOC / "perf_summary.json", DOC / "pcc_evidence.json", DOC / "watcher" / "WATCHER_AUDIT.md"]
    required += [DOC / "logs" / f"{path.stem}.log" for path in (DOC / "probes").glob("probe_*.py")]
    required += [DOC / "logs" / f"{name}.log" for name in ("suite_main", "long_context", "watcher_run", "doc_gate")]
    for impl in IMPLS:
        for kind in KINDS:
            for phase in PHASES:
                base = DOC / "tracy" / impl / kind
                required += [
                    base / f"{phase}_perf_report.csv",
                    base / f"{phase}_perf_report.txt",
                    base / f"{phase}_ops.csv.gz",
                    base / f"{phase}_ops.csv.provenance",
                ]
                required.append(DOC / "logs" / f"tracy_{impl}_{kind}_{phase}.log")

    missing = [
        str(path.relative_to(ROOT.parents[2]))
        for path in required
        if str(path.relative_to(ROOT.parents[2])) not in tracked
    ]
    assert not missing, (
        "these artifacts are read by the documents or this gate but are not tracked by git "
        f"(force-add them; the repo ignores *.log and *.csv): {missing}"
    )


def _shipped_grids() -> dict[str, tuple[int, int] | None]:
    """The ``core_grid`` constants the layer ships, read out of its source.

    Parsed rather than imported so this test needs no device and no ``ttnn`` import; the names
    are asserted present, so a rename fails here instead of silently checking nothing.
    """
    source = (ROOT / "tt" / "fused_decoder.py").read_text()
    grids: dict[str, tuple[int, int] | None] = {}
    for name in ("_RECURRENCE_READ_GRID", "_RECURRENCE_OUTER_GRID"):
        match = re.search(rf"^{name} = \((\d+), (\d+)\)$", source, re.MULTILINE)
        assert match, f"{name} is not a (y, x) literal in tt/fused_decoder.py"
        grids[name] = (int(match.group(1)), int(match.group(2)))
    for name in ("_AB_MATMUL_GRID", "_GROUP_SUM_GRID", "_GROUP_EXPAND_GRID"):
        match = re.search(rf"^{name} = \{{(.+?)\}}$", source, re.MULTILINE)
        assert match, f"{name} is not a one-line dict literal in tt/fused_decoder.py"
        for phase, value in re.findall(r'"(\w+)": (\(\d+, \d+\)|None)', match.group(1)):
            grids[f"{name}[{phase}]"] = None if value == "None" else tuple(int(v) for v in re.findall(r"\d+", value))
    return grids


def test_selected_grids_are_the_measured_best():
    """Every shipped ``core_grid`` is within a stdev of the fastest one its probe measured.

    The stage picks core grids from probe sweeps and then *states* the winner in prose and in a
    generated table's "selected" column.  A stage review found one of those statements wrong -
    the shipped recurrence grid was 6.3 % and ~2.3 stdevs slower than the log's own minimum,
    inside a block labelled GENERATED, because the cell was a literal.  This re-derives the
    comparison from the logs, at every regime each probe measured.
    """
    shipped = _shipped_grids()
    tolerance_note = []

    def check(label, grid, rows):
        """``rows``: ``{(y, x) or None: (median, stdev)}`` for one measured regime."""
        assert grid in rows, f"{label}: the shipped grid {grid} is not in the sweep {sorted(k for k in rows if k)}"
        best = min((key for key in rows if key is not None), key=lambda key: rows[key][0])
        best_median, best_stdev = rows[best]
        median, stdev = rows[grid]
        tolerance_note.append(f"{label}: shipped {grid} {median:.1f} us, best {best} {best_median:.1f} us")
        # Tolerance is the two spreads added: "the shipped grid is not distinguishably slower
        # than the fastest one".  Tighter than that and a 40 us op's run-to-run noise fails the
        # gate; looser and the 6.3 %, ~2.3-stdev miss a stage review found would pass it.
        assert median <= best_median + best_stdev + stdev, (
            f"{label}: the shipped grid {grid} measures {median:.1f} +- {stdev:.1f} us, and {best} "
            f"measures {best_median:.1f} +- {best_stdev:.1f} us - distinguishably faster"
        )

    recurrence = (DOC / "logs" / "probe_decode_recurrence.log").read_text(errors="replace")
    for heads in (48, 48 * 32):
        for name, prefix in (("_RECURRENCE_READ_GRID", "read"), ("_RECURRENCE_OUTER_GRID", "outer")):
            token = "transpose_a " if prefix == "outer" else ""
            rows = {}
            for grid_y, grid_x, median, stdev in re.findall(
                rf"{prefix}\s+heads={heads}\s+{token}core_grid (\d+)x(\d+)\s*median_us=\s*([\d.]+) stdev_us=\s*([\d.]+)",
                recurrence,
            ):
                rows[(int(grid_y), int(grid_x))] = (float(median), float(stdev))
            assert rows, f"probe_decode_recurrence.log has no {prefix} sweep at {heads} heads"
            check(f"{name} at {heads} heads", shipped[name], rows)

    # The three small-N matmul grids, from probe_matmul_bound.log.  ``None`` means the default
    # program factory won, which the log records as its own row.
    bound = (DOC / "logs" / "probe_matmul_bound.log").read_text(errors="replace")
    rows_by_label: dict[str, dict[tuple[int, int] | None, tuple[float, float]]] = {}
    for label, median in re.findall(r"matmul (\S+\s+\S+)\s+\d+x\s*\d+x\s*\d+ out=\w+\+fp32dest_us=\s*([\d.]+)", bound):
        rows_by_label.setdefault(" ".join(label.split()), {})[None] = (float(median), 0.0)
    for label, grid_y, grid_x, median, stdev in re.findall(
        r"matmul (\S+\s+\S+)\s+core_grid\s+(\d+)x\s*(\d+)\s+us=\s*([\d.]+) \(\s*([\d.]+)\)", bound
    ):
        rows_by_label.setdefault(" ".join(label.split()), {})[(int(grid_y), int(grid_x))] = (
            float(median),
            float(stdev),
        )
    named = {
        ("_AB_MATMUL_GRID", "prefill"): "in_proj_ab prefill",
        ("_AB_MATMUL_GRID", "decode"): "in_proj_ab decode",
        ("_GROUP_SUM_GRID", "prefill"): "gated_norm_sum prefill",
        ("_GROUP_SUM_GRID", "decode"): "gated_norm_sum decode",
        ("_GROUP_EXPAND_GRID", "prefill"): "gated_norm_exp prefill",
        ("_GROUP_EXPAND_GRID", "decode"): "gated_norm_exp decode",
    }
    for (name, phase), label in named.items():
        rows = rows_by_label.get(label)
        assert rows, f"probe_matmul_bound.log has no sweep for {label}"
        # The default row has no spread of its own; give it the sweep's median spread so a tie
        # with the default is not judged more harshly than a tie between two explicit grids.
        spreads = [stdev for _, stdev in rows.values() if stdev]
        rows = {key: (median, stdev or (sum(spreads) / len(spreads))) for key, (median, stdev) in rows.items()}
        check(f"{name}[{phase}]", shipped[f"{name}[{phase}]"], rows)
    assert tolerance_note  # the comparison actually ran


def test_probe_readme_covers_every_probe():
    """``probes/README.md`` has a row for every probe, a log for every row, and states its own counts.

    Three review rounds in a row added probes and left this document describing the previous set;
    its opening sentence said "eight" when there were eleven.  The set of probes is a directory
    listing, so it is checkable.
    """
    readme = (DOC / "probes" / "README.md").read_text()
    probes = sorted(path.name for path in (DOC / "probes").glob("probe_*.py"))
    tooling = sorted(
        path.name
        for path in (DOC / "probes").iterdir()
        if path.is_file() and not path.name.startswith(("probe_", "README"))
    )
    missing = [name for name in probes if f"`{name}`" not in readme]
    assert not missing, f"probes/README.md has no row for {missing}"
    logs = [name for name in probes if not (DOC / "logs" / f"{name[:-3]}.log").is_file()]
    assert not logs, f"these probes have no committed log: {logs}"

    words = {
        8: "Eight",
        9: "Nine",
        10: "Ten",
        11: "Eleven",
        12: "Twelve",
        13: "Thirteen",
        14: "Fourteen",
        15: "Fifteen",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
    }
    opening = readme.splitlines()[2]
    assert (
        words.get(len(probes), "?") in opening
    ), f"probes/README.md opens with {opening!r}, which does not state the {len(probes)} probes present"
    assert (
        words.get(len(tooling), "?") in opening
    ), f"probes/README.md opens with {opening!r}, which does not state the {len(tooling)} tooling files present"


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
    # The generator writes the six most common first tokens, ``f"{count:7d} {token}"``.  This
    # used to ``continue`` when a line was missing, which made the assertion below unreachable
    # and the whole histogram claim unbound - a stage review caught it.  Six is the generator's
    # own number (``make_watcher_audit.top``); if that changes, this must too.
    top = sorted(histogram.items(), key=lambda item: -item[1])[:6]
    missing = [
        f"{count} {token}"
        for token, count in top
        if not re.search(rf"(?<!\d){count}\s+{re.escape(token)}(?=\s|$)", audit, re.MULTILINE)
    ]
    assert not missing, f"the audit does not state these histogram lines of the committed log: {missing}"

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
@pytest.mark.parametrize("phase", sorted(PHASES))
def test_no_layout_round_trip_in_the_measured_pass(kind, phase):
    """No profiled op converts a layout that the very next op converts back.

    The device report is the ground truth here, not the python call trace: ``ttnn.concat``,
    ``ttnn.slice`` and friends relayout *inside* themselves, so a python-level trap cannot see
    them - which is exactly how a ``tilize`` immediately followed by an ``untilize`` of the same
    tensor survived three review rounds in the ``linear_attention`` prefill.
    """
    report = DOC / "tracy" / "fused" / kind / f"{phase}_perf_report.csv"
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
    assert not offenders, f"the profiled {kind} {phase} undoes a layout conversion it just made: {offenders}"


def test_watcher_log_is_clean():
    text = _watcher_log()
    pattern = re.compile(
        r"fatal|assert|corrupt|sanitiz|out of bounds|overflow|invalid|exception|error|fault|hang|unexpected",
        re.IGNORECASE,
    )
    offenders = [line for line in text.splitlines() if pattern.search(line) and "highest stack usage" not in line]
    assert not offenders, "watcher log is not clean:\n  " + "\n  ".join(offenders[:10])
