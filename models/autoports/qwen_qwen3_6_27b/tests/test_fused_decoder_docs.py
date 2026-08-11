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
        # The stage's contract: the fused graph must be **faster**, not merely smaller.  Op
        # count is checked as "no larger" rather than "smaller" because section 3.22
        # deliberately trades six python-level ops for device time on the full_attention
        # decode path - the dedicated rotate-half is single-core by construction - and the
        # contract's own words are "fewer ops or cleaner topology is not enough".
        assert row["device_ms_after"] < row["device_ms_before"], f"{key} did not get faster"
        assert row["ops_after"] <= row["ops_before"], f"{key} got bigger"


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
        # ``x`` counts when it is a multiplier - a number, ``x``, then a non-digit.  ``8x8`` is a
        # core grid and does not match; a bare multiplier does.  The old rule exempted every integer
        # multiplier, which is how two stale ones sat next to the table they misquoted.  Note the
        # limit of this check for *small* integers: the allowed set is every figure any artifact
        # prints, and a one- or two-digit value is almost always in it, so the documents state
        # ratios in words or point at the generated table rather than typing them.
        quoted = re.findall(r"\*{0,2}(\d+(?:\.\d+)?)\*{0,2}(?:\s*(ms|%|GB/s|us|bytes)\b|(x)(?![\dx]))", text)
        for value, unit, attached in quoted:
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


def test_no_device_time_is_unclassified():
    """Every measured pass classifies all of its device time; the ``other`` bucket is empty.

    A stage review found an eighth of the advertised-batch decode sitting in ``other`` - the
    stage's own new ternary op, which the bucket predicates did not name - while both documents
    asserted the bucket was empty.  The breakdown table is what every "where the time goes"
    conclusion rests on, so nothing may hide in it.
    """
    unclassified = {
        key: row["breakdown_ms"]["other"]
        for key, row in _perf_summary()["measurements"].items()
        if row["breakdown_ms"].get("other")
    }
    assert not unclassified, (
        "these passes have device time in no named bucket; add a predicate to "
        f"probes/make_perf_summary.py::CATEGORIES: {unclassified}"
    )


def test_documented_dedicated_ops_are_the_ones_shipped():
    """Any TTNN op the documents say this layer *dispatches* is one the layer really dispatches.

    §3.22 reverted a dedicated-op substitution, and five artifacts - including the implementation's
    own module docstring and a README row describing what a test guarantees - went on claiming it
    for a round.  The claim is checkable: ``FusedDecoder.FUSED_OPS`` is the set the dispatch test
    pins, and the source is the ground truth for what is called.
    """
    source = (ROOT / "tt" / "fused_decoder.py").read_text()
    shipped = set(re.findall(r'"(ttnn\.[\w.]+)"', source.split("FUSED_OPS = (")[1].split(")")[0]))
    assert shipped, "FUSED_OPS did not parse"
    called = set(re.findall(r"\bttnn\.(?:experimental\.|transformer\.)?[a-z_0-9]+\(", source))
    called = {name.rstrip("(") for name in called}

    # Every op the documents name as one this layer dispatches must be in FUSED_OPS or called.
    dedicated = (
        "ttnn.experimental.rotate_half",
        "ttnn.experimental.rotary_embedding_hf",
        "ttnn.transformer.chunk_gated_delta_rule",
        "ttnn.addcmul",
    )
    # Scanned per paragraph, and with generated blocks and the section that records the revert
    # removed: those are statements *about* a rewrite that was measured and dropped, not claims
    # that the layer dispatches it.
    strip = re.compile(r"<!-- GENERATED:\w+ -->.*?<!-- END GENERATED:\w+ -->", re.DOTALL)
    reverted_section = re.compile(r"### 3\.22.*?(?=\n### |\n---)", re.DOTALL)
    claims = []
    for path, text in _documents().items():
        text = reverted_section.sub("", strip.sub("", text))
        for paragraph in re.split(r"\n\s*\n", text):
            for op in dedicated:
                if op.split(".")[-1] not in paragraph:
                    continue
                if any(
                    token in paragraph
                    for token in ("revert", "reject", "not a dedicated", "deliberately absent", "3.22")
                ):
                    continue
                if op in shipped or op in called:
                    continue
                claims.append(f"{path.name}: {' '.join(paragraph.split())[:110]}")
    assert not claims, "these document lines name a dedicated op the shipped layer does not dispatch: " + "; ".join(
        claims
    )


def test_every_run_was_made_against_the_shipped_build():
    """Every committed run log and profiler provenance names the *current* fused decoder source.

    A stage review found the committed suite, long-context and watcher runs had been produced
    before a shipped decode-configuration change: the perf windows were re-profiled afterwards and
    the correctness runs were not, so the watcher-clean claim was about a build that no longer
    existed.  Every gate reads artifacts, and none tied an artifact to the source - this is that
    tie.  ``tests/conftest.py`` prints the hash at session start and ``probes/run_perf.sh`` appends
    it to each provenance file, so a stale artifact fails here rather than in a review.

    The fingerprint is of the decoder's *code* - ``ast.unparse`` of the parsed module with
    docstrings stripped, see ``tt/build_fingerprint.py`` - so it changes when behaviour can change
    and not when a review round rewrites a comment.  A byte hash would make an hour of hardware
    evidence stale for a reworded sentence, which is how a gate becomes something to work around.
    """
    from models.autoports.qwen_qwen3_6_27b.tt.build_fingerprint import fingerprint as _fingerprint

    fingerprint = _fingerprint()
    stale = []
    probes = sorted(path.stem for path in (DOC / "probes").glob("probe_*.py"))
    for name in ("suite_main", "long_context", "watcher_run", *probes):
        text = (DOC / "logs" / f"{name}.log").read_text(errors="replace")
        stamps = re.findall(r"FUSED_BUILD tt/fused_decoder\.py code-sha256=([0-9a-f]{64})", text)
        if not stamps:
            stale.append(f"logs/{name}.log carries no FUSED_BUILD stamp")
        elif any(stamp != fingerprint for stamp in set(stamps)):
            stale.append(f"logs/{name}.log was run against {sorted(set(stamps))}, not {fingerprint[:12]}")
    for impl in IMPLS:
        for kind in KINDS:
            for phase in PHASES:
                path = DOC / "tracy" / impl / kind / f"{phase}_ops.csv.provenance"
                text = path.read_text(errors="replace")
                stamps = re.findall(r"FUSED_BUILD tt/fused_decoder\.py code-sha256=([0-9a-f]{64})", text)
                if not stamps:
                    stale.append(f"{path.relative_to(DOC)} carries no FUSED_BUILD stamp")
                elif any(stamp != fingerprint for stamp in set(stamps)):
                    stale.append(f"{path.relative_to(DOC)} is of {sorted(set(stamps))[0][:12]}")
    assert not stale, (
        "these committed artifacts were not produced by the shipped tt/fused_decoder.py "
        f"({fingerprint[:12]}): {stale}"
    )


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
    # The rejected alternative's traced reports (§3.22) are evidence too: without them the
    # rejection is an assertion.
    required += [
        DOC / "tracy" / "rejected" / "rotate_half_dedicated" / name
        for name in (
            "decode_perf_report.csv",
            "decode_batch32_perf_report.csv",
            "decode_ops.csv.provenance",
            "decode_batch32_ops.csv.provenance",
        )
    ]
    # The watcher evidence lives under a ``generated/`` path, which .gitignore also matches, and
    # one of its three files went untracked for a round while the audit claimed it was committed.
    required += [
        DOC / "watcher" / "generated" / "watcher" / name
        for name in ("watcher.log.gz", "kernel_names.txt.gz", "kernel_elf_paths.txt.gz")
    ]
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
    for name in ("_RECURRENCE_OUTER_GRID",):
        match = re.search(rf"^{name} = \((\d+), (\d+)\)$", source, re.MULTILINE)
        assert match, f"{name} is not a (y, x) literal in tt/fused_decoder.py"
        grids[name] = (int(match.group(1)), int(match.group(2)))
    for name in ("_AB_MATMUL_GRID", "_GROUP_SUM_GRID", "_GROUP_EXPAND_GRID", "_RECURRENCE_READ_GRID"):
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
        # than the fastest one".  Tighter than that and a small op's run-to-run noise fails the
        # gate; looser and the 6.3 %, ~2.3-stdev miss a stage review found would pass it.
        assert median <= best_median + best_stdev + stdev, (
            f"{label}: the shipped grid {grid} measures {median:.1f} +- {stdev:.1f} us, and {best} "
            f"measures {best_median:.1f} +- {best_stdev:.1f} us - distinguishably faster"
        )

    recurrence = (DOC / "logs" / "probe_decode_recurrence.log").read_text(errors="replace")
    # The state-read grid is keyed by regime because no single grid wins at both; the outer
    # product's one grid is checked at both.
    for heads, regime in ((48, "small"), (48 * 32, "large")):
        for name, prefix in ((f"_RECURRENCE_READ_GRID[{regime}]", "read"), ("_RECURRENCE_OUTER_GRID", "outer")):
            token = "transpose_a " if prefix == "outer" else ""
            rows = {}
            for grid_y, grid_x, median, stdev in re.findall(
                rf"{prefix}\s+heads={heads}\s+{token}core_grid (\d+)x(\d+)\s*median_us=\s*([\d.]+) stdev_us=\s*([\d.]+)",
                recurrence,
            ):
                rows[(int(grid_y), int(grid_x))] = (float(median), float(stdev))
            assert rows, f"probe_decode_recurrence.log has no {prefix} sweep at {heads} heads"
            check(f"{name} at {heads} heads", shipped[name], rows)
    # And the regime split itself is a claim about the code: the constant must be a two-entry dict.
    assert {"_RECURRENCE_READ_GRID[small]", "_RECURRENCE_READ_GRID[large]"} <= set(
        shipped
    ), "the state-read grid is no longer keyed by regime; this test's regime mapping is stale"

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


#: Comparative phrases that state a measured *ratio* in words.  Nineteen review rounds retired
#: every stale figure shape the gates can bind; this is the one left, because a wrong word is
#: invisible to a numeric check - and four of them were wrong at once by round 19 (a bucket that
#: "grew by an order of magnitude" had grown by less than two, a lever that "buys a few percent"
#: measured slower).  The documents say what the generated tables say, or point at them.
_UNBOUND_COMPARATIVES = (
    "a few percent",
    "well over half",
    "about a sixth",
    "about a fifth",
    "at half the cost",
    "an order of magnitude",
    "orders of magnitude",
    "nominally the faster",
    "twice as fast",
    "half as fast",
)


def test_no_unbound_comparatives_in_the_documents():
    """The documents state measured ratios as numbers from a table, not as words.

    ``test_prose_perf_figures_match_the_summary`` binds every figure that carries a unit; a ratio
    written in words carries none, so it is the one figure shape no gate could see - and it is
    where the last rounds' stale claims all lived.  The rule is mechanical: a sentence that wants
    to state a ratio takes it from a generated block or points at one.
    """
    offenders = []
    for path, text in _documents().items():
        stripped = re.sub(r"<!-- GENERATED:\w+ -->.*?<!-- END GENERATED:\w+ -->", "", text, flags=re.DOTALL)
        # §8 is the review log: its rows quote the wrong claims they record fixing, so a phrase
        # there is a citation rather than an assertion.
        stripped = re.split(r"\n## 8\. ", stripped)[0]
        for phrase in _UNBOUND_COMPARATIVES:
            for match in re.finditer(re.escape(phrase), stripped):
                offenders.append(f"{path.name}:{stripped[: match.start()].count(chr(10)) + 1}: {phrase!r}")
    assert not offenders, (
        "these documents state a measured ratio in words; quote the generated table's figure or "
        f"point at the table instead: {offenders}"
    )


def test_quoted_blockers_appear_in_a_committed_log():
    """Every exact blocker the documents quote is a string some committed log actually contains.

    §6 is where the stage's "nothing was rejected without evidence" claim lives, and its rows quote
    device errors verbatim.  A stage review found one of those quotes still stating a blocker §3.6
    had retracted two rounds earlier - a *contract* blocker where the real one is an L1 overflow,
    which is revisitable.  A quoted error is checkable: it either appears in a log or it does not.
    """
    logs = "\n".join(path.read_text(errors="replace") for path in sorted((DOC / "logs").glob("*.log")))
    # Quoted device errors and blocker phrases the documents use.  Each must be findable in some
    # committed log; a retracted one is not.
    quoted = set()
    for text in list(_documents().values()) + [(DOC / "probes" / "README.md").read_text()]:
        quoted.update(re.findall(r"`(TT_FATAL[^`]*)`", text))
        quoted.update(re.findall(r"`(TT_THROW[^`]*)`", text))
        quoted.update(re.findall(r"`(found_valid_config[^`]*)`", text))
        quoted.update(re.findall(r"`(shard_grid_fit_error[^`]*)`", text))
        quoted.update(re.findall(r"`(Num of users[^`]*)`", text))

    def grounded(phrase: str) -> bool:
        """A quoted blocker is grounded in a committed log *or* in the checkout's own source.

        Both are legitimate: some blockers were observed and captured in a probe log, others are
        citations of the assertion in the op's C++ - and a citation is checkable against the tree.
        What is *not* legitimate is a quote that is in neither, which is what a retracted or
        mis-transcribed blocker looks like.
        """
        # ``TT_FATAL @ path/file.cpp:161: !expr`` -> ``expr``; ``TT_FATAL(expr, ...)`` -> ``expr``;
        # ``TT_FATAL: message`` -> ``message``.  The leading ``!`` of an asserted-negation is
        # dropped because the source spells the condition, not the failure.
        needle = phrase
        if "@" in needle:
            needle = needle.rsplit(":", 1)[-1]
        else:
            needle = re.sub(r"^TT_(FATAL|THROW)\s*[:(]?", "", needle)
        needle = needle.split(",")[0].strip(" ().`!")
        if not needle:
            return True
        if needle in logs:
            return True
        found = subprocess.run(
            ["grep", "-rlF", needle, "ttnn/cpp", "tt_metal"],
            cwd=ROOT.parents[2],
            capture_output=True,
            text=True,
        )
        return bool(found.stdout.strip())

    missing = sorted(phrase for phrase in quoted if not grounded(phrase))
    assert not missing, (
        "these blocker strings are quoted in the documents but appear in neither a committed log "
        f"nor this checkout's source - a retracted or mis-transcribed blocker: {missing}"
    )


def test_selected_constants_are_the_measured_best():
    """The shipped *scalar* configuration constants agree with the probe logs that chose them.

    ``test_selected_grids_are_the_measured_best`` binds the ``core_grid`` constants.  The threshold
    constants were not bound to anything, and a stage review found ``_GATED_NORM_GROUP_BATCH``
    shipping the slower of two measured forms for a whole range of batches after a re-measurement
    moved the crossing.
    """
    source = (ROOT / "tt" / "fused_decoder.py").read_text()

    def constant(name):
        match = re.search(rf"^{name} = (\d+|None)$", source, re.MULTILINE)
        assert match, f"{name} is not a scalar literal in tt/fused_decoder.py"
        return None if match.group(1) == "None" else int(match.group(1))

    # 1. the z-gated norm threshold: the batch at which the group form first becomes
    #    distinguishably faster than the reshape form.
    rows = re.findall(
        r"gated_norm batch=\s*(\d+) reshape_us=\s*([\d.]+) \(\s*([\d.]+)\) group_us=\s*([\d.]+) \(\s*([\d.]+)\)",
        (DOC / "logs" / "probe_gated_norm_batch.log").read_text(errors="replace"),
    )
    assert rows, "probe_gated_norm_batch.log has no measurements"
    crossing = next(
        (
            int(batch)
            for batch, reshape, r_spread, group, g_spread in rows
            if float(group) + float(g_spread) < float(reshape)
        ),
        None,
    )
    assert crossing is not None, "the probe log shows no batch where the group form wins"
    assert constant("_GATED_NORM_GROUP_BATCH") == crossing, (
        f"_GATED_NORM_GROUP_BATCH is {constant('_GATED_NORM_GROUP_BATCH')} but the probe log puts the "
        f"crossing at {crossing}"
    )

    # 2. the decode RMS-norm shard width: within the combined spread of the fastest measured.
    norms = re.findall(
        r"rms_norm sharded\s+(\d+)c ms=([\d.]+) stdev_ms=([\d.]+)",
        (DOC / "logs" / "probe_small_ops.log").read_text(errors="replace"),
    )
    assert norms, "probe_small_ops.log has no sharded rms_norm sweep"
    measured = {int(cores): (float(ms), float(spread)) for cores, ms, spread in norms}
    shipped_cores = constant("NORM_SHARD_CORES")
    assert shipped_cores in measured, f"NORM_SHARD_CORES {shipped_cores} is not in the sweep"
    best = min(measured, key=lambda cores: measured[cores][0])
    shipped_ms, shipped_spread = measured[shipped_cores]
    best_ms, best_spread = measured[best]
    assert shipped_ms <= best_ms + best_spread + shipped_spread, (
        f"NORM_SHARD_CORES {shipped_cores} measures {shipped_ms:.3f} +- {shipped_spread:.3f} ms and "
        f"{best} measures {best_ms:.3f} +- {best_spread:.3f} ms - distinguishably faster"
    )

    # 3. the decode FIR dtype: ``None`` means float32 always, and §3.25 says why the faster
    #    bfloat16 form is not shipped, so the constant must *not* be a batch.
    assert constant("_DECODE_CONV_BF16_BATCH") is None, (
        "the bfloat16 decode FIR was measured faster and reverted for PCC (§3.25); shipping it "
        "behind a threshold again needs new correctness evidence"
    )


def test_qualitative_verdicts_match_their_logs():
    """A sentence calling a measured comparison a tie or a win agrees with the log it cites.

    ``test_prose_perf_figures_match_the_summary`` binds figures that carry a unit and
    ``test_no_unbound_comparatives_in_the_documents`` binds ratios written in words.  Neither can
    see the third shape: a *verdict* - "a tie", "wins outright" - about a pair of measured medians.
    That is the one figure shape with no gate, and round 21 restated a true sentence into a false
    one inside it, saying the two gated-norm forms tie at batch 16 where the log has them 16 us
    apart with 3.7 us of combined spread.

    The rule is the stage's own, the one every generated caption uses: a gap wider than the two
    spreads together is a win, anything narrower is a tie.  Its scope is deliberately narrow - a
    sentence that names a batch, in a section that cites exactly one probe log with a row at that
    batch.  A section citing several logs is ambiguous and is skipped rather than guessed at, which
    the ``ambiguous`` count below makes visible instead of silent.
    """
    two_form = re.compile(
        r"^(\w+) batch=\s*(\d+) (\w+)_us=\s*([\d.]+) \(\s*([\d.]+)\) (\w+)_us=\s*([\d.]+) \(\s*([\d.]+)\)",
        re.MULTILINE,
    )
    # probe log stem -> {batch: "tie" | "<name> wins"}
    verdicts: dict[str, dict[int, str]] = {}
    for path in sorted((DOC / "logs").glob("probe_*.log")):
        rows = {}
        for match in two_form.finditer(path.read_text(errors="replace")):
            _, batch, left_name, left, left_spread, right_name, right, right_spread = match.groups()
            gap = abs(float(left) - float(right))
            if gap <= float(left_spread) + float(right_spread):
                rows[int(batch)] = "tie"
            else:
                rows[int(batch)] = f"{left_name} wins" if float(left) < float(right) else f"{right_name} wins"
        if rows:
            verdicts[path.stem] = rows

    scanned = dict(_documents())
    scanned.update({path: path.read_text() for path in SOURCES})
    # Generated blocks are *not* stripped here, unlike the figure gates.  A caption that states a
    # verdict is exactly the thing being bound, and this test re-derives the verdict from the raw
    # log without going through ``make_doc_tables``, so a generator that classified a row wrongly
    # would fail here rather than agree with itself.
    #
    # A "section" is a markdown heading's span, or - in the sources - a whole ``#:`` comment block,
    # which is how the shipped constants carry their justification.
    offenders, ambiguous, checked = [], 0, 0
    for path, text in scanned.items():
        splitter = r"\n(?=#{2,4} )" if path.suffix == ".md" else r"\n(?=[^#\n])"
        for section in re.split(splitter, text):
            cited = sorted({name for name in verdicts if name in section})
            claims = [
                (match.group(0), int(match.group(1) or match.group(2)), "tie" in match.group(0))
                for match in re.finditer(
                    r"(?:at|batch)\s+(?:batch\s+)?(\d+)[^.;|]*?\b(?:tie|ties|wins)\b"
                    r"|\b(?:tie|ties|wins)\b[^.;|]*?(?:at|batch)\s+(?:batch\s+)?(\d+)",
                    section,
                )
            ]
            if not claims:
                continue
            if len(cited) != 1:
                ambiguous += len(claims)
                continue
            rows = verdicts[cited[0]]
            for sentence, batch, says_tie in claims:
                if batch not in rows:
                    continue
                checked += 1
                measured_tie = rows[batch] == "tie"
                if says_tie != measured_tie:
                    offenders.append(
                        f"{path.name}: {sentence.strip()!r} - {cited[0]}.log at batch {batch} "
                        f"measures {rows[batch]}"
                    )
    assert checked, "this gate matched no verdict claim at all; its patterns have gone stale"
    assert not offenders, (
        "these sentences call a measured comparison a tie or a win against the log they cite; "
        f"state what the log states, or point at the generated caption that derives it: {offenders}"
    )


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
        16: "Sixteen",
        17: "Seventeen",
        18: "Eighteen",
        19: "Nineteen",
        20: "Twenty",
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
    # A tooling row may spell its arguments (``run_perf.sh <kind> <phase> <impl>``), so match the
    # opening backtick and the name rather than the exact token.
    listed_tools = [name for name in tooling if f"`{name}" in readme]
    assert sorted(listed_tools) == sorted(
        tooling
    ), f"probes/README.md's tooling table is missing {sorted(set(tooling) - set(listed_tools))}"
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
        token = line.split()[0] if line.split() else ""
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
    """Any watcher count the README quotes is the generated audit's, not an earlier run's.

    The three-or-more-digits floor this used to carry made a two-digit pass count unfailable, and
    a stage review found the README still quoting the pre-round-8 count of 11 against a 21-test
    run.  Every integer on a line that names the audit is checked now, and the counts themselves
    are a generated block rather than prose.
    """
    audit = (DOC / "watcher" / "WATCHER_AUDIT.md").read_text()
    documents = {"README.md": (DOC / "README.md").read_text(), "work_log.md": (DOC / "work_log.md").read_text()}
    for name, text in documents.items():
        for line in text.splitlines():
            if "WATCHER_AUDIT" not in line and "watcher/WATCHER_AUDIT.md" not in line:
                continue
            # A *count*, not any integer: a §8 narrative row that names the audit and a review
            # round in the same sentence is not quoting a quantity of the run.
            counts = re.findall(
                r"(?<![\w.])(\d+)(?![\w])\s*(?:passed|selected|deselected|lines|dumps|tests|runs|cases)",
                line,
            )
            for number in counts:
                # Word-boundaried, not a substring: "11" is inside the audit's "11077 lines" and
                # a substring test would accept it.
                assert re.search(
                    rf"(?<![\d.]){number}(?![\d])", audit
                ), f"{name} quotes {number} as a watcher-run count; the audit does not state it"


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
        # too, for the delta table.  And ``tt/``, because the grid tables' "selected" cells are
        # read out of the shipped constants in ``tt/fused_decoder.py``.
        mirror_doc = Path(scratch) / "doc"
        shutil.copytree(DOC.parent, mirror_doc, symlinks=True)
        shutil.copytree(ROOT / "tt", Path(scratch) / "tt", symlinks=True)
        mirror = mirror_doc / "fused_decoder"
        result = subprocess.run(
            [sys.executable, str(mirror / "probes" / "make_doc_tables.py")], capture_output=True, text=True
        )
        assert result.returncode == 0, f"{generator.name} failed: {result.stderr[-2000:]}"
        # The watcher audit is generated too, and it counts things - a stage review found it a
        # round stale because nothing re-ran its generator.
        audit = subprocess.run(
            [sys.executable, str(mirror / "probes" / "make_watcher_audit.py")], capture_output=True, text=True
        )
        assert audit.returncode == 0, f"make_watcher_audit.py failed: {audit.stderr[-2000:]}"
        stale = []
        for path, text in before.items():
            regenerated = (mirror / path.relative_to(DOC)).read_text()
            if regenerated != text:
                stale.append(path.name)
            assert _UNFILLED not in text, f"{path.name} has a generated block the generator never filled"
        audit_path = DOC / "watcher" / "WATCHER_AUDIT.md"
        if (mirror / audit_path.relative_to(DOC)).read_text() != audit_path.read_text():
            stale.append(audit_path.name)
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


def _selects(expression: str, candidate: str) -> bool:
    """Evaluate the subset of pytest ``-k`` syntax this stage's watcher command uses.

    That is a disjunction of terms, each either a bare substring or ``(a and b)``.  Anything
    outside that grammar raises rather than being silently treated as a match.
    """
    for term in re.split(r"\bor\b", expression):
        term = term.strip()
        if term.startswith("(") and term.endswith(")"):
            parts = [part.strip() for part in re.split(r"\band\b", term[1:-1])]
        else:
            parts = [term]
        for part in parts:
            assert re.fullmatch(r"[\w.\[\]-]+", part), f"unsupported -k syntax in {expression!r}: {part!r}"
        if all(part in candidate for part in parts):
            return True
    return False


def test_watcher_command_selects_the_run_it_documents():
    """The published watcher ``-k`` selects every test the committed watcher log ran.

    The audit and the work log publish a command as the reproduction recipe.  It was a literal in
    the audit generator, and when the run grew to cover the advertised ``max_batch`` branch the
    literal did not: the documented command selected eleven of the seventeen tests the log
    records, quietly dropping exactly the coverage that had just been added.
    """
    audit = (DOC / "watcher" / "WATCHER_AUDIT.md").read_text()
    match = re.search(r'-k "([^"]+)"', audit)
    assert match, "WATCHER_AUDIT.md publishes no -k expression"
    expression = match.group(1)

    run = (DOC / "logs" / "watcher_run.log").read_text(errors="replace")
    # ``-v -s`` interleaves device logging between the test id and its PASSED, so the id is
    # matched on its own rather than by adjacency.
    ran = sorted({name for name in re.findall(r"test_fused_decoder\.py::(\w+(?:\[[^\]]*\])?)", run)})
    assert ran, "watcher_run.log records no PASSED tests"
    unselected = [name for name in ran if not _selects(expression, name)]
    assert (
        not unselected
    ), f"the published watcher command does not select these tests the committed run ran: {unselected}"
    # And the documents publish the same command the audit does.
    for path, text in _documents().items():
        for published in re.findall(r'-k "([^"]+)"', text):
            assert published == expression, f"{path.name} publishes a different watcher -k than the audit"


def test_watcher_log_is_clean():
    text = _watcher_log()
    pattern = re.compile(
        r"fatal|assert|corrupt|sanitiz|out of bounds|overflow|invalid|exception|error|fault|hang|unexpected",
        re.IGNORECASE,
    )
    offenders = [line for line in text.splitlines() if pattern.search(line) and "highest stack usage" not in line]
    assert not offenders, "watcher log is not clean:\n  " + "\n  ".join(offenders[:10])
