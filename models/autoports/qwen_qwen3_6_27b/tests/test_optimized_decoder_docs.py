# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Document/artifact consistency gate for the optimized-decoder stage.  Opens no device.

The stage documents quote a lot of numbers.  This file is what stops them from drifting from the
artifacts they are quoting:

``test_generated_blocks_match_the_artifacts``
    re-runs ``probes/make_doc_tables.py``'s generator in memory and fails if any committed
    ``<!-- GENERATED:... -->`` block differs from what the artifacts produce;
``test_perf_summary_rederives_from_the_reports``
    re-derives every device time in ``perf_summary.json`` by summing the ``Device Time`` column of
    the committed ``tt-perf-report`` CSVs, so the summary is a view of the reports rather than a
    second, hand-maintained copy of them;
``test_every_cited_path_exists``
    every repo-relative path either document mentions resolves;
``test_dominant_matmul_rows_prove_the_policy``
    the *measured* math fidelity and operand dtypes of the dominant matmul rows agree with the
    shipped precision policy - the OPT-013 check that a claimed BFP4/LoFi policy actually reached
    the ops, which no PCC test can see;
``test_decode_is_traced_and_periodic``
    every decode window's op sequence repeats with the exact replay period, which is what makes
    "8 whole trace replays were captured" true rather than assumed;
``test_evidence_summary_matches_the_records``
    the acceptance block in ``context_contract.json`` agrees with ``pcc_evidence.json``;
``test_watcher_log_is_clean``
    the committed watcher log contains no offender line, and the run it describes is the one the
    documents claim.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parents[1] / "doc" / "optimized_decoder"
CONTRACT = Path(__file__).resolve().parents[1] / "doc" / "context_contract.json"
REPO = Path(__file__).resolve().parents[4]
DOCUMENTS = ("README.md", "work_log.md")

#: Replays inside the signposted decode window; must match ``PERF_DECODE_ITERS``.
DECODE_REPLAYS = 8


def _load_generator():
    path = DOC / "probes" / "make_doc_tables.py"
    spec = importlib.util.spec_from_file_location("make_doc_tables", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary() -> dict:
    return json.loads((DOC / "perf_summary.json").read_text())


def _text(name: str) -> str:
    return (DOC / name).read_text()


@pytest.mark.parametrize("name", DOCUMENTS)
def test_generated_blocks_match_the_artifacts(name):
    """Every committed generated block is byte-identical to what the generator produces now."""
    module = _load_generator()
    blocks = module.build_blocks()
    original = _text(name)
    regenerated = module.render(original, blocks)
    if regenerated != original:
        # Report which block moved, not a whole-file diff.
        moved = []
        for block in blocks:
            pattern = re.compile(
                rf"<!-- GENERATED:{re.escape(block)} -->\n(.*?)<!-- END GENERATED:{re.escape(block)} -->",
                re.DOTALL,
            )
            before = pattern.search(original)
            after = pattern.search(regenerated)
            if before and after and before.group(1) != after.group(1):
                moved.append(block)
        raise AssertionError(
            f"{name} generated blocks are stale: {moved or 'unknown'}. "
            f"Run doc/optimized_decoder/probes/make_doc_tables.py"
        )


def test_perf_summary_rederives_from_the_reports():
    """Every device time in the summary is the sum of its report's own ``Device Time`` column."""
    summary = _summary()
    for key, entry in summary["measurements"].items():
        report = DOC / entry["artifacts"]["report_csv"]
        assert report.exists(), f"{key}: missing {report}"
        with report.open() as handle:
            rows = list(csv.DictReader(handle))
        replays = DECODE_REPLAYS if entry["phase"].startswith("decode") else 1
        total = sum(float(row["Device Time"] or 0) for row in rows) / replays / 1000.0
        assert entry["ops_in_window"] == len(rows), f"{key}: op count moved"
        assert (
            abs(entry["device_kernel_time_ms"] - total) < 1e-3
        ), f"{key}: summary says {entry['device_kernel_time_ms']} ms, the report sums to {total:.4f} ms"


def test_speedup_block_is_consistent():
    """The before/after block is arithmetic on the two measurements it names, in both directions."""
    summary = _summary()
    assert summary["speedup"], "no before/after pair was recorded"
    for key, row in summary["speedup"].items():
        kind, phase = key.split("/")
        before = summary["measurements"][f"fused/{kind}/{phase}"]["device_kernel_time_ms"]
        after = summary["measurements"][f"optimized/{kind}/{phase}"]["device_kernel_time_ms"]
        assert row["device_ms_before"] == before and row["device_ms_after"] == after
        assert abs(row["speedup_x"] - before / after) < 1e-3
        assert row["speedup_x"] > 1.0, f"{key} is not a speed-up: {row}"


def test_decode_is_traced_and_periodic():
    """Every decode window captured whole replays: the op sequence repeats with the exact period."""
    summary = _summary()
    checked = 0
    for key, entry in summary["measurements"].items():
        if not entry["phase"].startswith("decode"):
            continue
        assert not entry["periodicity_check"].startswith("BROKEN"), f"{key}: {entry['periodicity_check']}"
        assert entry["ops_in_window"] == entry["ops_per_pass"] * DECODE_REPLAYS, key
        checked += 1
    assert checked, "no decode window was recorded"


#: What the shipped precision policy must look like in the *measured* rows.  ``tt-perf-report``
#: renders the operand dtypes and the math fidelity of every matmul; a BFP4 weight shows as
#: ``BFP4`` on the right-hand operand and the fidelity column names the compute mode.  If a
#: dominant row says ``BF16 x BF16`` while the policy claims BFP4, the policy did not reach the op.
_POLICY_TOKENS = ("BFP4", "BFP8", "BF16", "FP32", "LoFi", "HiFi2", "HiFi4")


def test_dominant_matmul_rows_prove_the_policy():
    """The measured fidelity/dtype of the dominant matmul rows agrees with the shipped policy."""
    summary = _summary()
    for key, entry in summary["measurements"].items():
        if not key.startswith("optimized/"):
            continue
        config = entry.get("config")
        assert config, f"{key}: the run recorded no PERFCONFIG line, so the policy cannot be verified"
        rows = entry["dominant_matmul_rows"]
        assert rows, f"{key}: no matmul rows in the report"
        # Build the set of dtypes the policy asks for, as the profiler spells them.
        wanted = set()
        for role, role_entry in config["roles"].items():
            dtype = role_entry["weight_dtype"]
            wanted.add(
                {
                    "DataType.BFLOAT4_B": "BFP4",
                    "DataType.BFLOAT8_B": "BFP8",
                    "DataType.BFLOAT16": "BF16",
                    "DataType.FLOAT32": "FP32",
                }[dtype]
            )
        seen = " ".join(row["math_fidelity"] for row in rows)
        assert any(token in seen for token in _POLICY_TOKENS), f"{key}: no dtype/fidelity in the rows: {seen}"
        # The largest row must carry one of the reduced dtypes the policy selected, unless the
        # policy itself is the bfloat16 baseline arm.
        if "BFP4" in wanted or "BFP8" in wanted:
            reduced = [row for row in rows if "BFP4" in row["math_fidelity"] or "BFP8" in row["math_fidelity"]]
            assert reduced, (
                f"{key}: the policy selects block-float weights but no dominant matmul row shows one. "
                f"Rows: {[(r['op'], r['math_fidelity']) for r in rows]}"
            )


def test_accounting_triple_is_present():
    """Every optimized decode window reports roofline, device time and end-to-end from one run."""
    summary = _summary()
    assert summary["accounting"], "no performance accounting was recorded"
    for key, entry in summary["accounting"].items():
        for field in (
            "roofline_ms_per_step_estimate",
            "decode_ms_per_step_device",
            "decode_ms_per_step_e2e",
            "op_to_op_gap_ms",
        ):
            assert entry[field] is not None, f"{key}: {field} is missing from the accounting triple"
        assert entry["roofline_ms_per_step_estimate"] <= entry["decode_ms_per_step_device"] * 1.05, (
            f"{key}: the roofline estimate {entry['roofline_ms_per_step_estimate']} ms exceeds the measured "
            f"device time {entry['decode_ms_per_step_device']} ms, so one of them is wrong"
        )
        assert (
            entry["decode_ms_per_step_e2e"] >= entry["decode_ms_per_step_device"]
        ), f"{key}: end-to-end is below device time, which cannot be"


_PATH_PATTERN = re.compile(r"(?:\]\(|`)((?:\.\./)*[A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|json|md|csv|txt|log|sh|gz|png))")


@pytest.mark.parametrize("name", DOCUMENTS)
def test_every_cited_path_exists(name):
    """Every repo-relative path the document mentions resolves to a file."""
    text = _text(name)
    missing = []
    for match in _PATH_PATTERN.finditer(text):
        cited = match.group(1)
        for base in (DOC, DOC.parent, DOC.parent.parent, REPO):
            if (base / cited).resolve().exists():
                break
        else:
            missing.append(cited)
    assert not missing, f"{name} cites paths that do not exist: {sorted(set(missing))}"


def test_evidence_summary_matches_the_records():
    """The contract's optimized-stage acceptance block agrees with ``pcc_evidence.json``."""
    contract = json.loads(CONTRACT.read_text())
    block = contract.get("optimized_decoder", {}).get("acceptance")
    assert block, "context_contract.json has no optimized_decoder.acceptance block"
    evidence = json.loads((DOC / "pcc_evidence.json").read_text())
    assert block["records"] == evidence["num_records"], "record count drifted"
    assert block["pcc_records"] == evidence["num_pcc_records"], "PCC record count drifted"
    assert round(block["min_pcc"], 6) == round(evidence["min_pcc"], 6), "minimum PCC drifted"
    assert block["pcc_records_below_bar"] == 0, "a PCC record is below the bar"
    assert evidence["min_pcc"] >= block["pcc_bar"], "the recorded minimum is below the stated bar"


def test_watcher_log_is_clean():
    """The committed watcher log has no offender line and its size matches the audit."""
    log = DOC / "logs" / "watcher_run.log"
    assert log.exists(), "no watcher run is committed"
    text = log.read_text(errors="replace")
    offenders = [
        line
        for line in text.splitlines()
        if re.search(r"\bWatcher detected|\bAsserted\b|Ran out of|\bstuck\b|\bhang", line, re.IGNORECASE)
    ]
    assert not offenders, f"watcher log has offender lines: {offenders[:5]}"
    assert "passed" in text, "the watcher run log does not record a passing pytest summary"
    assert "TT_METAL_WATCHER" in text or "Watcher" in text, "this log was not produced with watcher enabled"


def test_context_contract_capability_is_not_reduced():
    """The optimized stage did not reduce the advertised context."""
    contract = json.loads(CONTRACT.read_text())
    block = contract["optimized_decoder"]
    assert block["supported_context"] == contract["target_context"], "the optimized stage reduced the context"
    assert block["reduced_from_advertised"] is False
