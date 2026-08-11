# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Assert that every measured figure quoted in the fused-decoder docs exists in a committed artifact.

Same guard the functional stage carries (`../functional_decoder/audit_figures.py`), pointed at this
stage's documents and evidence. Three passes over every document:

* **decimals** — every ``\\d+.\\d+`` must appear in some committed evidence file;
* **integers** — every 2+ digit integer must too, minus a declared set of shape/config constants;
* **labelled figures** — ``N passed`` and friends must appear in an artifact *together with their
  label*, because a bare integer will match something by accident in a large log.

Figures that are computed from other figures rather than read off a run are declared in ``DERIVED``
as an **expression**, which is evaluated and must reproduce the quoted value; every literal in the
expression must itself be sourced. That is what stops a document quoting a correct-looking ratio
that was derived from the wrong basis.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/fused_decoder/audit_figures.py

Exit code 0 means every quoted figure traces to an artifact. Run it after any doc edit and after any
re-run that changes the numbers.

Residual limits, so a reader does not over-trust it: substring matching means a figure can be
"sourced" by an unrelated occurrence of the same digits in a large log, and it checks figures, not
prose.
"""

import re
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parent
ROOT = DOC.parent.parent
CONTRACT = DOC.parent / "context_contract.json"

#: The implementation and the tests the artifacts below measure. Two rules apply to them: figures
#: quoted in their comments must be sourced like any other document's, and every artifact must be
#: **newer** than both — otherwise the evidence describes an earlier revision of the code being
#: shipped, which is exactly the failure this stage's review caught once.
SOURCES = [
    ROOT / "tt/fused_decoder.py",
    ROOT / "tests/test_fused_decoder.py",
    # The suite's conftest is part of what the artifacts measure — it registers the `long` marker the
    # advertised-context cases carry — so it belongs under the same hash and freshness gate.
    ROOT / "tests/conftest.py",
]

#: Documents whose figures must be sourced. The capability contract is here as well as in
#: ``ARTIFACTS``: review round 6 pointed out that being only an artifact made it a source of figures
#: that was never itself checked, so a stale number in it could never be caught.
DOCS = [
    CONTRACT,
    DOC / "README.md",
    DOC / "work_log.md",
    DOC / "tracy/PROVENANCE.md",
    DOC / "watcher/CLASSIFICATION.md",
    *SOURCES,
]

#: Evidence the figures may come from.
ARTIFACTS = [
    DOC / "logs/pcc_summary.txt",
    DOC / "logs/pytest_full_suite.txt",
    DOC / "logs/watcher_pytest.txt",
    DOC / "logs/commit_record.txt",
    DOC / "logs/source_manifest.txt",
    DOC / "logs/ab_functional_vs_fused.txt",
    DOC / "logs/ab_moe_group_tokens.txt",
    DOC / "logs/ab_rope_mode.txt",
    DOC / "logs/probe_fused_ops.txt",
    DOC / "logs/probe_router_and_reduce.txt",
    DOC / "logs/probe_gate_up_pack.txt",
    DOC / "logs/probe_conv1d_and_norm.txt",
    DOC / "logs/probe_decode_micro.txt",
    DOC / "logs/probe_conv_tail.txt",
    DOC / "logs/suite_criticals.txt",
    DOC / "tracy/perf_summary.txt",
    # Every percentage the README quotes that is computed rather than measured, recorded next to the
    # arithmetic and the operands that produced it. Written by logs/make_readme_perf.py, so those
    # figures are sourced by their own derivation instead of by a table maintained here by hand.
    DOC / "tracy/derived_figures.txt",
    # The UnaryDeviceOperation aggregate split into its FILL and non-FILL parts, recovered from the
    # raw capture's ATTRIBUTES column. Sources README §5.4's zero-fill bullet.
    DOC / "tracy/fill_summary.txt",
    DOC / "tracy/slow_ops_summary.txt",
    *sorted((DOC / "tracy").glob("*/*_tracy_run.txt")),
    *sorted((DOC / "tracy").glob("*/*_perf_report.txt")),
    DOC / "watcher/watcher_log.txt",
    DOC / "watcher/census_summary.txt",
    CONTRACT,
    # The functional stage is the measured baseline of this stage, so its evidence is evidence here.
    DOC.parent / "functional_decoder/tracy/perf_summary.txt",
    DOC.parent / "functional_decoder/logs/pcc_summary.txt",
    DOC.parent / "functional_decoder/logs/router_precision_ab.txt",
    DOC.parent / "functional_decoder/tracy/slow_ops_summary.txt",
    # Sources the measured allocatable-DRAM figure the contract's capacity evidence rests on.
    DOC.parent / "functional_decoder/logs/dram_capacity_probe.txt",
]

#: Values computed from sourced numbers rather than measured. Each maps the quoted string to
#: ``(expression, what it is)``; the expression is evaluated and must reproduce the quoted value at
#: the precision it is quoted, and every literal in it must itself be sourced.
DERIVED: dict[str, tuple[str, str]] = {
    # The RoPE cos/sin tables at the full 262144 context. The 67633152-byte figure is the
    # functional stage's footprint in context_contract.json; the "full" RoPE mode widens the table
    # from rope_dim (64) to head_dim (256), i.e. exactly 4x.
    "67.6": ("67633152 / 1000 ** 2", "rope_dim-wide cos/sin tables, MB"),
    "270.5": ("4 * 67633152 / 1000 ** 2", "head_dim-wide cos/sin tables, MB"),
    # Router score-vector L1 errors, quoted as percentages of the functional stage's measured
    # relative errors (doc/functional_decoder/logs/router_precision_ab.txt).
    "0.19": ("100 * 0.001913", "float32-router score-vector L1 relative error, %"),
    "1.15": ("100 * 0.01151", "bfloat16-router score-vector L1 relative error, %"),
    # context_contract.json's capacity evidence. These are byte counts computed from the model's
    # shapes, not measurements, so they are re-derived here on every audit rather than trusted:
    # hidden 2048, head_dim 256, 16 q-heads / 2 kv-heads, moe_intermediate 512, 256 experts,
    # 32 DeltaNet value heads of 128, conv kernel 4, context 262144 in 64-token blocks.
    "268435456": ("(262144 // 64) * 2 * 64 * 256 * 2", "paged K (or V) cache at the full context, B"),
    "264192": ("262144 + 2048", "RoPE table rows: the context plus one prefill chunk of headroom"),
    "67633152": ("2 * 264192 * 64 * 2", "rope_dim-wide cos/sin tables, B"),
    "1610612736": ("3 * 256 * 2048 * 512 * 2", "routed expert gate/up/down weights, B"),
    "7340032": ("2 * (3 * 2048 * 512 + 2048 * 256)", "shared expert + router weights, B"),
    "54525952": ("2 * (3 * 2048 * 4096 + 2 * 2048 * 512)", "full_attention q/k/v/o/gate weights, B"),
    "67436544": (
        "2 * (2048 * 8192 + 2048 * 4096 + 4096 * 2048 + 8192 * 4 + 2 * 2048 * 32)",
        "linear_attention in-projection/z/a/b/out/conv weights, B",
    ),
    "2097152": ("32 * 128 * 128 * 4", "DeltaNet recurrent state at batch 1, B (float32)"),
    "49152": ("3 * 8192 * 2", "DeltaNet conv state at batch 1, B (kernel-1 rows of conv_dim)"),
    # No fused-specific footprint figures: the packing is byte-neutral on device (the functional MoE
    # already stores the shared router tile-padded, tt/moe.py). Review round 15 retracted the earlier
    # "+131072 B per layer" claim and round 17 found these three still blessed here, which would have
    # let the retracted numbers reappear in any fused document without failing the audit.
    "2276982784": (
        "2 * 268435456 + 67633152 + 1610612736 + 7340032 + 54525952",
        "worst-case full_attention layer at the full context, B",
    ),
    "34091302912": ("31.75 * 1024 ** 3", "measured allocatable DRAM, B"),
    # NOTE: the fused-vs-functional percentages, the MoE shares and the SLOW-time figures used to be
    # listed here as hand-written expressions. They are now computed by logs/make_readme_perf.py,
    # which writes each one next to its arithmetic into tracy/derived_figures.txt and splices the
    # result into the README - so they are sourced by an artifact rather than by this table, and
    # they cannot drift from the profiler output. What remains here is the figures no generator
    # owns.
}

#: Figures the work log quotes because they belong to a *superseded* measurement round — the
#: intermediate A/B numbers that show how the graph evolved. They have no artifact by construction
#: (the committed A/B logs are the final round), so they are permitted only in ``work_log.md``.
HISTORICAL = {
    # Retracted by review round 15 and removed from DERIVED by round 17: the fused packing is
    # byte-neutral on device, so these two "fused equivalents" describe a footprint delta this stage
    # does not have. §7 quotes them to record the retraction, and they must not reappear anywhere else.
    "7471104",
    "2277113856",
    "338.13",  # round-1 A/B, functional linear prefill
    "319.35",  # round-1 A/B, fused linear prefill
    "316.66",  # round-1 A/B, functional full prefill
    "297.65",  # round-1 A/B, fused full prefill
    "2.615",  # round-1 A/B, functional linear traced decode
    "2.359",  # round-1 A/B, fused linear traced decode
    "2.395",  # round-1 A/B, functional full traced decode
    "1.944",  # round-1 A/B, fused full traced decode
    # The conv1d/FIR times as they stood when round 3's review read them. §7 quotes them to say what
    # the finding was; the shipped figures are the current probe's, in §3.1 and §4.4.
    "3.326",
    "0.539",
}

#: Constants and thresholds that are choices, not measurements.
ALLOWED = {
    "0.995",  # the PCC acceptance bar, inherited from the functional stage
    "0.9999",  # the fused-vs-functional equivalence bar and the RoPE assertion bar
    "0.999",  # the chunk-invariance assertion bar
    "1.0",
    "0.25",  # partial_rotary_factor
    "1.2",  # tt-perf-report major.minor in "1.2.8"
    "3.5",  # "Qwen3.5"
    "1.0.35",  # part of the model name
    "2.0",
    "20.0",  # the SOFTPLUS threshold in UnaryWithParam(SOFTPLUS, 1.0, 20.0)
    "0.84",  # quoted from models/demos/blackhole/qwen36 - another port's measurement, not ours
}

#: Integers that are architecture/config/shape constants or prose numbers rather than measured
#: results.
ALLOWED_INT = {
    "35",
    "40",
    "30",
    "10",
    "16",
    "32",
    "64",
    "96",
    "128",
    "192",
    "256",
    "512",
    "1024",
    "2048",
    "4096",
    "5120",
    "8192",
    "9216",
    "12352",
    "12544",
    "16384",
    "32768",
    "65536",
    "262143",
    "262144",
    "1056",
    "1312",
    "2049",
    "3000",
    "5000",
    "8000",
    "50000",
    "24576",
    "2026",
    "1e-6",
    "1e-5",
    "11",
    "12",
    "13",
    "14",
    "15",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "31",
    "33",
    "34",
    "36",
    "37",
    "38",
    "39",
    "41",
    "42",
    "43",
    "44",
    "45",
    "46",
    "47",
    "48",
    "49",
    "50",
    "55",
    "60",
    "70",
    "80",
    "90",
    "99",
    "100",
    "110",
    "130",
    "200",
    "250",
    "300",
    "330",
    "343",
    "355",
    "360",
    "391",
    "405",
    "429",
    "462",
    "1000",
    "1056",
}

#: Document *structure* rather than figures: markdown headings ("## 4. …", "### 4.11 …") and
#: cross-references ("§4.11", "§5.3"). Stripped before scanning, because a section number is not a
#: measurement and exempting each one by value would also exempt a real figure with those digits.
SECTION = re.compile(r"(?m)^#{1,6}\s+\d+(?:\.\d+)*\.?\s|§\s?\d+(?:\.\d+)*")

#: RNG seeds and source line references. Both are identifiers, not measurements: a seed exists so a
#: run is reproducible, and `file.cpp:123` is a citation. Stripped before scanning for the same
#: reason as section numbers — exempting them by value would also exempt a real figure.
IDENTIFIER = re.compile(r"(?:manual_)?seed\s*[=(]\s*\d+|\bseed=\d+|\.(?:cpp|hpp|py|cc|h):\d+(?:-\d+)?")

DECIMAL = re.compile(r"(?<![\d.])\d{1,12}\.\d{1,6}(?!\.?\d)")
INTEGER = re.compile(r"(?<![\d.\w])\d{2,}(?!\.?\d)(?!\w)")

LABELLED = [
    (re.compile(r"(\d[\d ,]*) passed"), "{} passed"),
    (re.compile(r"(\d[\d ,]*)[- ]line log"), "{} lines"),
    (re.compile(r"(\d[\d ,]*) fatal-class"), "fatal-class matches: {}"),
]


def load(paths):
    blobs = {}
    for path in paths:
        if path.is_file():
            blobs[path] = path.read_text(errors="replace")
    return blobs


def sourced(value: str, blobs) -> bool:
    plain = value.replace(",", "").replace(" ", "")
    for text in blobs.values():
        if value in text or plain in text:
            return True
    return False


def check_derived(blobs) -> list:
    problems = []
    for quoted, (expression, label) in DERIVED.items():
        for literal in DECIMAL.findall(expression) + INTEGER.findall(expression):
            if (
                literal in ALLOWED
                or literal in ALLOWED_INT
                or literal in DERIVED  # a derived figure may be built from other derived figures
                or sourced(literal, blobs)
            ):
                continue
            problems.append(f"DERIVED-OPERAND-UNSOURCED  {quoted} ({label}): {literal}")
        got = eval(expression)  # noqa: S307 - the table is source, not input
        digits = len(quoted.split(".")[1]) if "." in quoted else 0
        if f"{got:.{digits}f}" != quoted:
            problems.append(f"DERIVED-MISMATCH  {quoted} ({label}): {expression} = {got:.{digits + 2}f}")
    return problems


def check_source_manifest() -> list:
    """The recorded source hashes must match the files being shipped.

    Stronger than the mtime ordering below, and immune to a touch: ``run_evidence.sh`` records
    ``sha256sum`` of the implementation and the tests before it runs anything, so this proves the
    artifacts were produced by *these bytes*, not merely at a later wall-clock time.
    """
    import hashlib

    manifest = DOC / "logs/source_manifest.txt"
    if not manifest.is_file():
        return ["MISSING-ARTIFACT  logs/source_manifest.txt (run logs/run_evidence.sh)"]
    recorded = {}
    for line in manifest.read_text().split("\n"):
        if line.strip():
            digest, _, path = line.partition("  ")
            recorded[Path(path.strip()).name] = digest.strip()
    problems = []
    for source in SOURCES:
        if not source.is_file():
            continue
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if recorded.get(source.name) != actual:
            problems.append(f"SOURCE-CHANGED  {source.name} differs from the manifest the evidence was produced with")
    return problems


def check_readme_tables() -> list:
    """The README's generated blocks must equal what their generators produce right now.

    README §2.1-§2.3 come from ``logs/make_readme_tables.py``; the README headline table, §5.2, §5.3,
    §5.4's ``SLOW`` table and the percentage claims under it from ``logs/make_readme_perf.py``; and
    the work log's §4.12 probe table from ``logs/make_worklog_tables.py``. Round 3 of this stage's
    review found eleven PCC cells that disagreed with ``logs/pcc_summary.txt`` because they had been
    hand-transcribed from an earlier run, and the next re-run made twenty of the work log's inline
    probe figures stale the same way. The fix is to stop transcribing, so the agreement is asserted
    on every audit rather than assumed.
    """
    # Two generators claiming one block name silently fight: each --write undoes the other, so the
    # documents never settle and --check always reports stale. Caught once for real, so it is a gate.
    import re as _re
    import subprocess

    owners: dict[str, str] = {}
    problems = []
    for name in ("make_readme_tables.py", "make_readme_perf.py", "make_worklog_tables.py"):
        script = DOC / "logs" / name
        if not script.is_file():
            continue
        result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        for block in _re.findall(r"<!-- generated:([a-z0-9-]+) -->", result.stdout):
            if owners.setdefault(block, name) != name:
                problems.append(f"BLOCK-NAME-COLLISION  '{block}' is written by both {owners[block]} and {name}")

    for name in ("make_readme_tables.py", "make_readme_perf.py", "make_worklog_tables.py"):
        script = DOC / "logs" / name
        if not script.is_file():
            problems.append(f"MISSING-SCRIPT  logs/{name}")
            continue
        result = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        if result.returncode:
            problems += [ln for ln in (result.stdout + result.stderr).strip().split("\n") if ln.strip()]
    return problems


def check_freshness() -> list:
    """Every artifact must be newer than the code it measures.

    This is the guard for the one failure mode a figure audit cannot see: a source edit landing
    after the evidence run, so that every committed number describes a revision that is not the one
    being shipped. ``logs/commit_record.txt`` is written by hand after the commit and is exempt.
    """
    problems = []
    newest_source = max(((p.stat().st_mtime, p) for p in SOURCES if p.is_file()), default=None)
    if newest_source is None:
        return ["MISSING-SOURCE  no implementation file found to check freshness against"]
    stamp, source = newest_source
    for path in ARTIFACTS:
        exempt = (
            path.name in {"commit_record.txt", "context_contract.json", "source_manifest.txt"}
            or "functional_decoder" in path.parts
        )
        if not path.is_file() or exempt:
            continue
        if path.stat().st_mtime < stamp:
            problems.append(f"STALE-ARTIFACT  {path.name} predates {source.name}")
    return problems


def main() -> int:
    blobs = load(ARTIFACTS)
    missing = [p for p in ARTIFACTS if not p.is_file()]
    problems = [f"MISSING-ARTIFACT  {p}" for p in missing]
    problems += check_source_manifest()
    problems += check_freshness()
    problems += check_readme_tables()
    problems += check_derived(blobs)

    for doc in DOCS:
        if not doc.is_file():
            problems.append(f"MISSING-DOC  {doc}")
            continue
        # A document that is also an artifact must not source its own figures: that is circular, and
        # it would make the contract's numbers self-certifying now that it is audited.
        blobs = {path: body for path, body in load(ARTIFACTS).items() if path != doc}
        text = IDENTIFIER.sub(" ", SECTION.sub(" ", doc.read_text(errors="replace")))
        for value in sorted(set(DECIMAL.findall(text))):
            if value in ALLOWED or value in DERIVED or sourced(value, blobs):
                continue
            if value in HISTORICAL and doc.name == "work_log.md":
                continue
            problems.append(f"UNSOURCED  {doc.name}: {value}")
        for value in sorted(set(INTEGER.findall(text))):
            if value in ALLOWED_INT or value in DERIVED or sourced(value, blobs):
                continue
            # Same rule as the decimal pass above: a superseded figure may be quoted in the work log,
            # which is where this stage records what a review round found wrong, and nowhere else.
            if value in HISTORICAL and doc.name == "work_log.md":
                continue
            problems.append(f"UNSOURCED-INT  {doc.name}: {value}")
        for pattern, template in LABELLED:
            for value in sorted(set(pattern.findall(text))):
                phrase = template.format(value.strip())
                if not sourced(phrase, blobs):
                    problems.append(f"UNSOURCED-LABEL  {doc.name}: {phrase!r}")

    for problem in problems:
        print(problem)
    print(
        f"checked {len(DOCS)} documents (including {len(SOURCES)} source files) against "
        f"{len(blobs)} artifacts, evaluated {len(DERIVED)} derived figures, re-generated the "
        f"README's PCC tables, and asserted every artifact is newer than the code: "
        f"{len(problems)} problem(s)"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
