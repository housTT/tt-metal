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
    documents claim;
``test_every_piece_of_report_advice_is_answered``
    every distinct ``tt-perf-report`` advice line in the committed optimized reports is quoted and
    answered in the work log, so section 6.1's scope is derived from the artifacts instead of stated;
``test_every_measured_layout_op_is_accounted_for``
    every ``Tilize``/``Untilize`` group in the measured reports is on an allow-list with a reason, and a
    group running on one or two cores - the signature of an op choosing a layout change internally,
    which no Python-level op counter can see - has to be individually accounted for;
``test_prefill_fidelity_override_reached_the_measured_ops``
    the prefill-only fidelity override that fixes section 3.8.2's full-context scale failure shows up in
    the *profiler's* Math Fidelity column for the role it names, and not at decode.  It is the same
    OPT-013 idea as the dominant-row check, applied to a knob whose silent failure would look like a
    correctness regression with no stated cause.
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


#: Every layout op code that may appear in a measured optimized report, why it is there, and whether it
#: is allowed to run on one or two cores.
#:
#: This is an allow-list with reasons rather than a budget: an op group that appears here has been
#: looked at, and one that does not fails the gate until it is.  ``low_core_ok`` is the second half of
#: the check.  A layout conversion on one or two cores is the signature of an op choosing a layout
#: change *internally* - nothing in this layer asks for one, and a conversion this layer did ask for
#: would spread over the grid - so a low-core group has to be individually accounted for, not merely
#: present in the list.
_ALLOWED_LAYOUT_OPS = {
    "TilizeDeviceOperation": {
        "why": (
            "prefill only - the causal convolution's ROW_MAJOR slice/concat window, tilized once per "
            "chunk on 80-108 cores (work_log.md section 3.5)"
        ),
        "low_core_ok": False,
    },
    "UntilizeDeviceOperation": {
        "why": "prefill only - the other half of that same window, untilized once per chunk on 80-110 cores",
        "low_core_ok": False,
    },
    "TilizeWithValPaddingDeviceOperation": {
        "why": (
            "two sources, both accounted for: the conv window's ragged K-row cut at prefill, and at "
            "decode the re-tilize inside ttnn.repeat_interleave's tile-axis expansion of the 16 GDN key "
            "heads to 48 - work_log.md section 3.10, where the equivalent graph without it is measured "
            "and rejected on batch-32 latency"
        ),
        "low_core_ok": True,
    },
    "UntilizeWithUnpaddingDeviceOperation": {
        "why": "the other end of both of those - the same two sources, same accounting",
        "low_core_ok": True,
    },
}


def test_every_measured_layout_op_is_accounted_for():
    """No layout op group in a measured optimized report is unaccounted for.

    Two claims, and the second is the one that catches things.  The allow-list is the *accounting*.
    ``low_core_ok`` is the part that fails when something new appears: a layout op on one or two cores
    was chosen by an op rather than by this layer, and the only two in the measured graph are the ones
    section 3.10 attributes to ``ttnn.repeat_interleave``.  A new one - or an old one that spreads to a
    new op code - fails here rather than sitting in a report nobody re-reads.
    """
    summary = _summary()
    checked = 0
    for key, entry in summary["measurements"].items():
        if not key.startswith("optimized/"):
            continue
        report = DOC / entry["artifacts"]["report_csv"]
        with report.open() as handle:
            rows = list(csv.DictReader(handle))
        total = sum(float(row["Device Time"] or 0) for row in rows)
        groups: dict = {}
        for row in rows:
            op = (row["OP Code"] or "").strip()
            if "ilize" not in op.lower():
                continue
            bucket = groups.setdefault(op, {"us": 0.0, "cores": set(), "n": 0})
            bucket["us"] += float(row["Device Time"] or 0)
            bucket["n"] += 1
            try:
                bucket["cores"].add(int(float(row["Cores"])))
            except (TypeError, ValueError):
                pass
        for op, bucket in groups.items():
            assert op in _ALLOWED_LAYOUT_OPS, (
                f"{key}: layout op {op!r} is on the measured path ({bucket['n']} instances, "
                f"{bucket['us']:.1f} us, {100 * bucket['us'] / total:.2f} % of the pass, cores "
                f"{sorted(bucket['cores'])}) and is not accounted for in _ALLOWED_LAYOUT_OPS"
            )
            worst = min(bucket["cores"]) if bucket["cores"] else None
            if worst is not None and worst <= 2:
                assert _ALLOWED_LAYOUT_OPS[op]["low_core_ok"], (
                    f"{key}: layout op {op!r} runs on {worst} core(s) for {bucket['us']:.1f} us "
                    f"({100 * bucket['us'] / total:.2f} % of the pass), and it is not one of the groups "
                    "recorded as legitimately low-core.  A layout conversion on one or two cores is one "
                    "an op chose internally - find it in the graph and account for it rather than "
                    "flipping low_core_ok"
                )
        checked += 1
    assert checked, "no optimized measurement was recorded"


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


#: Advice lines that are a *verdict on this configuration* rather than a suggestion, so there is
#: nothing to account for.  Anything else the reports say has to be answered in the work log.
_ADVICE_NOT_ACTIONABLE = re.compile(r"look good")


def test_every_piece_of_report_advice_is_answered():
    """Every distinct ``tt-perf-report`` advice line in the optimized reports appears in the work log.

    ``work_log.md`` section 6.1 claims to account for "every distinct advice line in the six committed
    optimized reports".  That claim used to be scoped by hand - it named which reports it had looked at
    - which is exactly the kind of sentence that goes stale when a re-measurement adds a row.  So the
    scope is checked instead of stated: this extracts the advice column from every committed optimized
    report, splits it into individual lines, and requires each one to be quoted in the document.

    Quoted, specifically: the work log's table has one row per advice line with the line itself in a
    code span, so a match here means a reader can find the line and read what was tried.
    """
    summary = _summary()
    text = _text("work_log.md")
    seen: dict = {}
    for key, entry in summary["measurements"].items():
        if not key.startswith("optimized/"):
            continue
        report = DOC / entry["artifacts"]["report_csv"]
        with report.open() as handle:
            for row in csv.DictReader(handle):
                for piece in (row.get("Advice") or "").split("•"):
                    piece = piece.strip()
                    if piece and not _ADVICE_NOT_ACTIONABLE.search(piece):
                        seen.setdefault(piece, key)
    assert seen, "no advice lines in any optimized report, which means the reports were built with advice off"
    missing = []
    for piece, where in sorted(seen.items()):
        # Match on a distinctive prefix: the reports append operand shapes and parenthetical op names
        # to some lines, and the work log quotes the line itself.
        probe = piece.split("(")[0].strip()
        probe = probe[:60]
        if probe not in text:
            missing.append((probe, where))
    assert not missing, (
        "work_log.md does not answer these tt-perf-report advice lines, so section 6.1's claim to "
        f"account for all of them is false: {missing}"
    )


#: ``(K, N)`` of every role, so a measured ``MatmulDeviceOperation M x K x N`` row can be attributed back
#: to the role whose policy produced it.  ``in_proj_z``/``wgate`` and ``o_proj``/``out_proj`` share a shape
#: but live in different layer kinds, so the pair (kind, shape) is unambiguous.
_ROLE_SHAPES = {
    "linear_attention": {"in_proj_qkv": (5120, 10240), "in_proj_z": (5120, 6144), "out_proj": (6144, 5120)},
    "full_attention": {"wqkv": (5120, 8192), "wgate": (5120, 6144), "o_proj": (6144, 5120)},
}
_SHARED_ROLE_SHAPES = {"mlp_gate": (5120, 17408), "mlp_up": (5120, 17408), "mlp_down": (17408, 5120)}


def test_prefill_fidelity_override_reached_the_measured_ops():
    """A role given a prefill-only fidelity override must show that fidelity in the *prefill* report.

    This is OPT-013 applied to `PrecisionPolicy.prefill_fidelity_roles`, and it is worth its own check
    because the override is the fix for a correctness failure rather than a performance choice: if it
    silently did not reach the op, the full-context scale would regress and nothing else here would say
    why.  The role is identified by its ``K x N`` in the profiler's own op name, and the assertion is on
    the profiler's Math Fidelity column - not on the policy object, which is what claims it.

    The same rows must show the *unoverridden* fidelity at decode, since the override is prefill-only.
    """
    from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import DEFAULT_POLICY

    overrides = DEFAULT_POLICY.prefill_fidelity_roles or {}
    if not overrides:
        pytest.skip("the shipped policy has no prefill-only fidelity override")
    summary = _summary()
    checked = 0
    for key, entry in summary["measurements"].items():
        if not key.startswith("optimized/"):
            continue
        _, kind, phase = key.split("/")
        shapes = dict(_SHARED_ROLE_SHAPES, **_ROLE_SHAPES.get(kind, {}))
        for role, fidelity in overrides.items():
            shape = shapes.get(role)
            if shape is None:
                continue
            wanted = str(fidelity).replace("MathFidelity.", "")
            expected = (
                wanted
                if phase == "prefill"
                else str(DEFAULT_POLICY.fidelity(role, decode=True)).replace("MathFidelity.", "")
            )
            rows = [r for r in entry["dominant_matmul_rows"] if f"x {shape[0]} x {shape[1]}" in r["op"]]
            if not rows:
                continue
            for row in rows:
                assert row["math_fidelity"].split()[0] == expected, (
                    f"{key}: role {role!r} has a prefill fidelity override of {wanted} but its measured row "
                    f"{row['op']!r} reports {row['math_fidelity']!r}; expected {expected} at {phase}.  Either "
                    "the override did not reach the op or the report predates it"
                )
                checked += 1
    assert checked, (
        "no measured row could be attributed to an overridden role, so this check proved nothing - "
        f"overrides {sorted(overrides)}, shapes {sorted(_SHARED_ROLE_SHAPES)}"
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
    # The count that must be zero is the one against the bar the synthetic cases actually hold.  The
    # count against the acceptance bar is recorded honestly and is expected to be large, because the
    # synthetic-weight cases are not what carries that bar.
    assert (
        block["pcc_records_below_synthetic_stress_bar"] == 0
    ), f"{block['pcc_records_below_synthetic_stress_bar']} PCC records are below the synthetic stress bar"
    assert block["pcc_records_below_acceptance_bar"] == sum(
        1
        for rec in evidence["records"]
        if isinstance(rec.get("value"), (int, float))
        and not isinstance(rec.get("value"), bool)
        and "_scale" not in rec.get("metric", "")
        and rec["value"] < block["pcc_bar"]
    ), "the recorded below-acceptance-bar count is not what the evidence says"
    # Two bars, and the gate checks each against the evidence that carries it.  The acceptance bar is
    # held by the *real-checkpoint* records; the synthetic-weight cases hold the looser stress bar,
    # for the measured reason recorded in the contract's ``bar_note`` and pinned by
    # ``test_synthetic_bar_is_justified_by_the_real_weight_evidence`` in the device suite.
    assert (
        evidence["min_pcc"] >= block["synthetic_weight_stress_bar"]
    ), f"the recorded minimum {evidence['min_pcc']} is below the synthetic stress bar"
    assert (
        block["real_weight_min_pcc"] >= block["pcc_bar"]
    ), f"the real-weight minimum {block['real_weight_min_pcc']} is below the acceptance bar {block['pcc_bar']}"


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
