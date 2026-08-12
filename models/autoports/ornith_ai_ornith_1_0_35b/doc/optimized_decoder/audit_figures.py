# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Assert that every measured figure quoted in the optimized-decoder docs exists in a committed artifact.

The same guard the two preceding stages carry (``../fused_decoder/audit_figures.py``), pointed at this
stage's documents and evidence. Review rounds 2, 3 and 4 of this stage each closed on "the figures are
re-derived" and each next round found more that were not: round 4 found eight, including a whole
program-config search table quoting a superseded run of its own probe. Round 4's finding was that the
remedy already existed one directory over and this stage had not carried it. This is that port.

Passes over every document:

* **decimals** — every ``\\d+.\\d+`` must appear in some committed evidence file;
* **integers** — every 2+ digit integer must too, minus a declared set of shape/config constants;
* **labelled figures** — ``N passed`` and friends must appear in an artifact *together with their
  label*, because a bare integer will match something by accident in a large log.

Figures computed from other figures rather than read off a run are declared in ``DERIVED`` as an
**expression**, which is evaluated and must reproduce the quoted value; every literal in the
expression must itself be sourced. That is what stops a document quoting a correct-looking ratio that
was derived from the wrong basis.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_decoder/audit_figures.py

Exit code 0 means every quoted figure traces to an artifact. Run it after any doc edit and after any
re-run that changes the numbers; ``logs/run_evidence.sh`` runs it last, in-tree and again against a
``git archive HEAD`` extraction.

Residual limits, so a reader does not over-trust it: substring matching means a figure can be
"sourced" by an unrelated occurrence of the same digits in a large log, and it checks figures, not
prose. It cannot tell that a *correct* number is being used to support a wrong claim.
"""

from __future__ import annotations

import gzip
import hashlib
import re
import subprocess
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parent
ROOT = DOC.parent.parent
#: The tt-metal checkout, so generator subprocesses and error messages are repo-relative.
REPO = ROOT.parents[1]
FUSED = DOC.parent / "fused_decoder"
CONTRACT = DOC.parent / "context_contract.json"

#: The implementation and the tests the artifacts below measure. Two rules apply to them: figures
#: quoted in their comments must be sourced like any other document's, and every artifact must be
#: **newer** than all of them — otherwise the evidence describes an earlier revision of the code being
#: shipped. Round 4 flagged exactly that gap: a 50-line edit landed after six of the probe artifacts.
SOURCES = [
    ROOT / "tt/optimized_decoder.py",
    ROOT / "tests/test_optimized_decoder.py",
    ROOT / "tests/conftest.py",
]

#: Documents whose figures must be sourced. ``logs/make_readme.py`` is here because its
#: ``ADVICE_ACTIONS`` prose carries measured microsecond figures into the README's *generated* advice
#: block — so ``make_readme.py --check`` agrees with itself while quoting a stale number, which is
#: precisely how three of round 4's eight findings survived three rounds of review.
DOCS = [
    CONTRACT,
    DOC / "README.md",
    DOC / "work_log.md",
    DOC / "watcher/CLASSIFICATION.md",
    DOC / "logs/make_readme.py",
    # The probe and harness scripts too: their module docstrings quote measured figures to explain why
    # the probe exists, and one of them was still quoting a pre-round-4 pair of percentages that no
    # artifact contained. A script is a document when it makes a claim about a measurement.
    *sorted(DOC.glob("logs/probe_*.py")),
    DOC / "logs/bench.py",
    DOC / "tracy/perf_accounting.py",
    DOC / "watcher/census.py",
    *SOURCES,
]

#: Evidence the figures may come from. ``read_blob`` below reads ``x`` or ``x.gz``, because the repo's
#: 500 KB file-size hook and its blanket ``*.csv`` ignore rule mean the large artifacts are committed
#: gzipped.
ARTIFACTS = [
    DOC / "logs/pytest_full_suite.txt",
    DOC / "logs/watcher_pytest.txt",
    DOC / "logs/commit_record.txt",
    DOC / "logs/source_manifest.txt",
    DOC / "logs/ab_fused_vs_optimized.txt",
    DOC / "logs/ab_precision_policy.txt",
    DOC / "logs/ab_norm_shard_width.txt",
    DOC / "logs/ab_state_l1.txt",
    DOC / "logs/ab_sdpa_decode_contract.txt",
    DOC / "logs/ab_gdn_out_activation.txt",
    DOC / "logs/probe_sparse_matmul.txt",
    DOC / "logs/probe_dense_matmul.txt",
    DOC / "logs/probe_prefill_matmul.txt",
    DOC / "logs/probe_decode_micro.txt",
    DOC / "logs/probe_projection_dtype.txt",
    DOC / "logs/probe_footprint.txt",
    DOC / "tracy/perf_summary.json",
    DOC / "tracy/perf_accounting.txt",
    DOC / "watcher/census_summary.txt",
    DOC / "watcher/watcher_log.txt",
    DOC / "triage/tt-triage.txt",
    DOC / "triage/triage-summary.txt",
    CONTRACT,
    *sorted((DOC / "tracy").glob("*/*_perf_report.txt")),
    *sorted((DOC / "tracy").glob("*/*_perf_report.txt.gz")),
    *sorted((DOC / "tracy").glob("*/*_perf_report.summary.txt")),
    *sorted((DOC / "tracy").glob("*/*_perf_report.summary.txt.gz")),
    *sorted((DOC / "tracy").glob("*/*_perf_report.csv")),
    *sorted((DOC / "tracy").glob("*/*_perf_report.csv.gz")),
    *sorted((DOC / "tracy").glob("*/*_tracy_run.txt")),
    # The fused decoder is this stage's measured baseline, so its evidence is evidence here.
    FUSED / "tracy/perf_summary.txt",
    FUSED / "tracy/derived_figures.txt",
    FUSED / "tracy/fill_summary.txt",
    FUSED / "tracy/slow_ops_summary.txt",
    # The fused stage's own per-op decode/prefill reports: this stage's operation-topology audit
    # (work_log §2) is a table of the *baseline* it planned against, so those figures are only sourceable
    # from the previous stage's capture, not from this stage's optimized one.
    *sorted(FUSED.glob("tracy/*/*_perf_report.txt")),
    *sorted(FUSED.glob("tracy/*/*_perf_report.csv")),
    FUSED / "logs/pcc_summary.txt",
    FUSED / "logs/probe_decode_micro.txt",
    FUSED / "logs/probe_router_and_reduce.txt",
    FUSED / "logs/ab_moe_group_tokens.txt",
    DOC.parent / "functional_decoder/logs/dram_capacity_probe.txt",
    DOC.parent / "functional_decoder/logs/router_precision_ab.txt",
]

#: Values computed from sourced numbers rather than measured. Each maps the quoted string to
#: ``(expression, what it is)``; the expression is evaluated and must reproduce the quoted value at the
#: precision it is quoted, and every literal in it must itself be sourced.
DERIVED: dict[str, tuple[str, str]] = {
    # Every per-term byte count in context_contract.json's optimized-decoder footprint section is now
    # MEASURED by logs/probe_footprint.py and sourced from its artifact, so it does not need an
    # expression here. What remains is the arithmetic the contract states about shapes the probe cannot
    # see, and the one capacity figure that comes from an earlier stage's probe.
    #
    # bfloat8_b is 1.0625 B/elem and bfloat4_b 0.5625 B/elem (one shared exponent byte per 16-datum
    # face); hidden 2048, moe_intermediate 512, 256 experts, head_dim 256, 2 kv heads, context 262144
    # in 64-token blocks.
    "142606336": ("(262144 // 64) * 2 * 64 * 256 * 1.0625", "paged K (or V) cache at BFP8, full context, B"),
    "268435456": ("(262144 // 64) * 2 * 64 * 256 * 2", "paged K (or V) cache at bfloat16, full context, B"),
    "452984832": ("3 * 256 * 2048 * 512 * 0.5625", "routed expert gate/up/down weights at BFP4, B"),
    "1610612736": ("3 * 256 * 2048 * 512 * 2", "routed expert gate/up/down weights at bfloat16, B"),
    "4460544": (
        "(2048 * 1056 + 512 * 2048) * 1.0625 + 2048 * 256 * 2",
        "shared expert at BFP8 + the bfloat16 router, B",
    ),
    "67633152": ("2 * 264192 * 64 * 2", "rope_dim-wide cos/sin tables, B"),
    "264192": ("262144 + 2048", "RoPE table rows: the context plus one prefill chunk of headroom"),
    "34091302912": ("31.75 * 1024 ** 3", "measured allocatable DRAM, B"),
    # How far the contract's earlier hand-modelled worst-case layer was from the measured one.
    "285700": ("839553028 - 839267328", "the hand-modelled footprint's error, B"),
    # The *logical* size of the batch-1 DeltaNet conv state, which the contract quotes beside the
    # allocated padded size to explain the difference: 3 kernel-1 rows of conv_dim 8192 at bfloat16.
    "49152": ("3 * 8192 * 2", "DeltaNet conv state at batch 1, logical (unpadded), B"),
}

#: Constants and thresholds that are choices, not measurements.
ALLOWED = {
    "0.995",  # the PCC acceptance bar, inherited from the functional stage
    "0.998",  # the optimized-vs-fused equivalence bar
    "0.9999",
    "0.999",
    "1.0",
    "0.25",  # partial_rotary_factor
    "1.2",  # tt-perf-report major.minor in "1.2.8"
    "3.5",  # "Qwen3.5"
    "1.0.35",  # part of the model name
    "2.0",
    "20.0",  # the SOFTPLUS threshold in UnaryWithParam(SOFTPLUS, 1.0, 20.0)
    "1.0625",  # bfloat8_b bytes per element (shared exponent per 16-datum face)
    "0.5625",  # bfloat4_b bytes per element
    "0.32",  # EXPERT_L1_BUDGET_FRACTION
    "1.05",  # MATMUL_CB_MODEL_OVERHEAD
}

#: Integers that are architecture/config/shape constants, section/round numbers or prose numbers
#: rather than measured results.
ALLOWED_INT = {
    # model + device shape constants
    "10",
    "11",
    "16",
    "24",
    "32",
    "33",
    "48",
    "55",
    "56",
    "64",
    "80",
    "88",
    "96",
    "99",
    "110",
    "128",
    "129",
    "162",
    "192",
    "250",
    "256",
    "320",
    "512",
    "1024",
    "1056",
    "2048",
    "2049",
    "3000",
    "4096",
    "5120",
    "8192",
    "9216",
    "12352",
    "16384",
    "24576",
    "32768",
    "50000",
    "65536",
    "262143",
    "262144",
    # small integers used as counts, section numbers, review-round numbers and prose
    *(str(n) for n in range(12, 100)),
    "100",
    "130",
    "200",
    "300",
    "1000",
    "1500",
    "2026",
    "1e-6",
    "1e-5",
    # An upstream issue number, not a measurement: tt-metal #45943, cited by the sparse_matmul factory
    # itself at its `batch_nnz` runtime argument.
    "45943",
    # Byte counts the contract quotes for the FUSED policy, to say what each term was before. They are
    # either in DERIVED or measured by probe_footprint.py's `fused-parity` rows; these two are the
    # earlier hand-modelled values kept in a note and are exempt as such.
    "839267328",
}

#: A ``<!-- generated:NAME -->`` … ``<!-- /generated:NAME -->`` region of the README. Stripped before
#: scanning, because those regions are not hand-written: ``make_readme.py --check`` (run by
#: ``check_generators`` below) already proves the README equals what the generators produce from the
#: artifacts, and the generators derive figures — a per-step cost is a committed total divided by the
#: replay count, so the quotient is correct but is not a substring of any artifact. What this audit is
#: for is the prose *around* those blocks, which no generator owns. ``logs/make_readme.py`` is itself in
#: ``DOCS``, so a figure typed by hand into a generator's prose is still caught.
GENERATED_BLOCK = re.compile(r"<!-- generated:[a-z0-9-]+ -->.*?<!-- /generated:[a-z0-9-]+ -->", re.S)

#: Document *structure* rather than figures: markdown headings ("## 4. …", "### 4.11 …") and
#: cross-references ("§4.11", "§5.3"). Stripped before scanning, because a section number is not a
#: measurement and exempting each one by value would also exempt a real figure with those digits.
SECTION = re.compile(r"(?m)^#{1,6}\s+\d+(?:\.\d+)*\.?\s|§\s?\d+(?:\.\d+)*")

#: RNG seeds, source line references and git object names. All are identifiers, not measurements.
IDENTIFIER = re.compile(
    r"(?:manual_)?seed\s*[=(]\s*\d+|\bseed=\d+|\.(?:cpp|hpp|py|cc|h|sh|json|md|txt|csv):\d+(?:-\d+)?"
    # A git object name, which must contain at least one hex LETTER: `[0-9a-f]{7,40}` on its own also
    # matches a long decimal, which would silently exempt every byte count in the contract.
    r"|\b(?=[0-9a-f]{7,40}\b)[0-9a-f]*[a-f][0-9a-f]*\b"
)

DECIMAL = re.compile(r"(?<![\d.])\d{1,12}\.\d{1,6}(?!\.?\d)")
INTEGER = re.compile(r"(?<![\d.\w])\d{2,}(?!\.?\d)(?!\w)")

LABELLED = [
    (re.compile(r"(\d[\d ,]*) passed"), "{} passed"),
    (re.compile(r"(\d[\d ,]*) lines of the watcher"), "{}  TOTAL"),
    (re.compile(r"\*\*(\d[\d ,]*) matches\*\*"), "fatal-class matches: {}"),
]

#: Figures a document quotes because they belong to a *superseded* measurement round — the intermediate
#: A/B numbers that show how the graph evolved, and the wrong values a review round found. They have no
#: artifact by construction, so they are permitted only in ``work_log.md``, which is where this stage
#: records what each review round found.
HISTORICAL = {
    # --- Group 1: the development ladder in work_log §3 -------------------------------------------
    # One row per change, cumulative, from one harness during development. These are measurements of
    # *intermediate revisions of the code*, so by construction no artifact of the shipped revision can
    # contain them and re-running cannot reproduce them - the intermediate states no longer exist. They
    # are the record of the path taken, they are quoted only in the work log, and the section says so.
    # The SHIPPED level is README §5.2's generated table, always.
    "1.987",
    "1.691",
    "1.909",
    "1.624",
    "1.842",
    "1.297",
    "1.515",
    "1.253",
    "1.473",
    "0.976",
    "1.198",
    "0.983",
    "1.205",
    "0.875",
    "1.111",
    "0.855",
    "1.093",
    "1.094",
    "1.074",
    "243.44",
    "96.89",
    "257.73",
    "102.94",
    # The fused stage's published decode figure against this stage's re-measurement of the same
    # baseline, and the speedup that would follow from it. Quoted in §3 to say where the column starts.
    "2.06",
    "2.07",
    "1.86",
    # --- Group 2: figures a review round found WRONG -----------------------------------------------
    # Quoted in §6 to record what each finding was. They must never come back as a claim about a run.
    "21.4",
    "9.3",
    "20.4",
    "212.1",
    "288.3",
    "134.4",
    "119.5",
    "17.1",
    "262",
    "291",
    "528.5",
    "471.9",
    "260.3",
    "178.8",
    "695.3",
    "607.0",
    "111.9",
    "64.9",
    "73.4",
    "52.6",
    "1.093",
    "1.073",
    "0.859",
    "0.839",
    "1004.7",
    "13.9",
    "12.6",
    "52.4",
    "717.4",
    "150.0",
    "14.9",
    "2.063",
    "96.61",
    "102.62",
    "243.58",
    "257.67",
    "81.65",
    "1.07",
}


def read_blob(path: Path) -> str | None:
    """Text of ``path``, or of ``path.gz``. Returns ``None`` if neither exists.

    A ``.gz`` path is decompressed rather than read as text: the glob patterns above match both
    spellings, and reading compressed bytes with ``errors="replace"`` would add a few kilobytes of
    mojibake to the evidence pool, where it could "source" a figure by coincidence.
    """
    if path.suffix == ".gz":
        return gzip.decompress(path.read_bytes()).decode(errors="replace") if path.is_file() else None
    if path.is_file():
        return path.read_text(errors="replace")
    packed = path.with_suffix(path.suffix + ".gz")
    if packed.is_file():
        return gzip.decompress(packed.read_bytes()).decode(errors="replace")
    return None


def exists(path: Path) -> bool:
    return path.is_file() or (path.suffix != ".gz" and path.with_suffix(path.suffix + ".gz").is_file())


def artifact_path(path: Path) -> Path:
    """The file that actually holds ``path``'s bytes — itself or its ``.gz``."""
    return path if path.is_file() else path.with_suffix(path.suffix + ".gz")


def load(paths) -> dict:
    blobs = {}
    for path in paths:
        body = read_blob(path)
        if body is not None:
            blobs[path] = body
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
            if literal in ALLOWED or literal in ALLOWED_INT or literal in DERIVED or sourced(literal, blobs):
                continue
            problems.append(f"DERIVED-OPERAND-UNSOURCED  {quoted} ({label}): {literal}")
        got = eval(expression)  # noqa: S307 - the table is source, not input
        digits = len(quoted.split(".")[1]) if "." in quoted else 0
        if f"{got:.{digits}f}" != quoted:
            problems.append(f"DERIVED-MISMATCH  {quoted} ({label}): {expression} = {got:.{digits + 2}f}")
    return problems


def check_source_manifest() -> list:
    """The recorded source hashes must match the files being shipped.

    Stronger than the mtime ordering below and immune to a ``touch``: ``run_evidence.sh`` records
    ``sha256sum`` of the implementation and the tests before it runs anything, so this proves the
    artifacts were produced by *these bytes*, not merely at a later wall-clock time. Round 4's
    hard-check gap was exactly this: six probe artifacts predated a source edit and nothing proved
    whether the edit was inert.
    """
    manifest = DOC / "logs/source_manifest.txt"
    if not manifest.is_file():
        return ["MISSING-ARTIFACT  logs/source_manifest.txt (run logs/run_evidence.sh)"]
    recorded = {}
    for line in manifest.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
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


def check_generators() -> list:
    """The README's generated blocks must equal what ``make_readme.py`` produces right now, and the
    two summary artifacts must regenerate from their committed inputs."""
    problems = []
    checks = [
        (DOC / "logs/make_readme.py", ["--check"]),
        (DOC / "tracy/perf_accounting.py", []),
        (DOC / "watcher/census.py", []),
    ]
    for script, extra in checks:
        if not script.is_file():
            problems.append(f"MISSING-SCRIPT  {script.relative_to(DOC)}")
            continue
        result = subprocess.run([sys.executable, str(script), *extra], capture_output=True, text=True, cwd=str(REPO))
        if result.returncode:
            problems.append(f"GENERATOR-FAILED  {script.name}")
            problems += [f"    {ln}" for ln in (result.stdout + result.stderr).strip().splitlines()[-6:]]
    return problems


#: Artifacts the freshness rule does not apply to, and why.
#:
#: * ``commit_record.txt`` is written by hand after the commit, and ``source_manifest.txt`` is the
#:   freshness evidence itself.
#: * ``context_contract.json`` is a document that is also evidence; its figures are audited.
#: * ``triage/`` is a capture of a *hardware incident* that a re-run must not reproduce on purpose.
#: * The four ``ab_*.txt`` below are **one-off decision records**, not regenerable by
#:   ``run_evidence.sh``: each needed a deliberate variant of the implementation (a different constant,
#:   a different dtype on one multiply, a k-chunk the layer must not ship) that only exists long enough
#:   to measure it. They are A/B *pairs* measured back to back in one process, so what they establish is
#:   the sign and size of a **difference**, which does not go stale when unrelated code changes; their
#:   absolute level belongs to the revision that produced them, and the shipped default's absolute level
#:   is re-measured end to end on every run in ``ab_fused_vs_optimized.txt``. Documents quoting them must
#:   quote the pair, not the level.
EXEMPT_FROM_FRESHNESS = {
    "commit_record.txt",
    "context_contract.json",
    "source_manifest.txt",
    "tt-triage.txt",
    "triage-summary.txt",
    "ab_gdn_out_activation.txt",
    "ab_norm_shard_width.txt",
    "ab_state_l1.txt",
    "ab_sdpa_decode_contract.txt",
}


def check_freshness() -> list:
    """Every artifact must be newer than the code it measures.

    The guard for the one failure mode a figure audit cannot see: a source edit landing after the
    evidence run, so that every committed number describes a revision that is not the one being
    shipped. Files written by hand after the run, and the preceding stages' evidence, are exempt.
    """
    problems = []
    newest = max(((p.stat().st_mtime, p) for p in SOURCES if p.is_file()), default=None)
    if newest is None:
        return ["MISSING-SOURCE  no implementation file found to check freshness against"]
    stamp, source = newest
    for path in ARTIFACTS:
        exempt = path.name in EXEMPT_FROM_FRESHNESS or any(
            part in {"fused_decoder", "functional_decoder"} for part in path.parts
        )
        if exempt or not exists(path):
            continue
        if artifact_path(path).stat().st_mtime < stamp:
            problems.append(f"STALE-ARTIFACT  {path.name} predates {source.name}")
    return problems


def check_suite_log_complete() -> list:
    """The committed suite and watcher logs must record runs that *finished*.

    The fused stage shipped a suite log that had been stopped mid-test once, while every figure
    generated from it looked right. A log with no pytest summary line cannot back any figure.
    """
    problems = []
    for name in ("logs/pytest_full_suite.txt", "logs/watcher_pytest.txt"):
        body = read_blob(DOC / name)
        if body is None:
            continue  # the missing-artifact pass reports it
        if not [ln for ln in body.splitlines() if re.match(r"=+ .*(passed|failed)", ln)]:
            problems.append(
                f"INCOMPLETE-LOG  {name} has no pytest summary line: the run it records did not "
                f"finish, so no figure generated from it is backed by a completed suite"
            )
    return problems


def main() -> int:
    all_blobs = load(ARTIFACTS)
    problems = [f"MISSING-ARTIFACT  {p.relative_to(REPO)}" for p in ARTIFACTS if not exists(p)]
    problems += check_source_manifest()
    problems += check_suite_log_complete()
    problems += check_freshness()
    problems += check_generators()
    problems += check_derived(all_blobs)

    for doc in DOCS:
        if not doc.is_file():
            problems.append(f"MISSING-DOC  {doc}")
            continue
        # A document that is also an artifact must not source its own figures: that is circular, and it
        # would make the capability contract's numbers self-certifying now that it is audited.
        blobs = {path: body for path, body in all_blobs.items() if path != doc}
        body = doc.read_text(errors="replace")
        if doc == CONTRACT:
            # Only this stage's own section. The other stages' sections are their evidence's business,
            # and auditing them here would demand artifacts this stage does not own.
            import json as _json

            body = _json.dumps(_json.loads(body).get("optimized_decoder", {}), indent=1)
        text = IDENTIFIER.sub(" ", SECTION.sub(" ", GENERATED_BLOCK.sub(" ", body)))
        for value in sorted(set(DECIMAL.findall(text))):
            if value in ALLOWED or value in DERIVED or sourced(value, blobs):
                continue
            if value in HISTORICAL and doc.name == "work_log.md":
                continue
            problems.append(f"UNSOURCED  {doc.name}: {value}")
        for value in sorted(set(INTEGER.findall(text))):
            if value in ALLOWED_INT or value in DERIVED or sourced(value, blobs):
                continue
            if value in HISTORICAL and doc.name == "work_log.md":
                continue
            problems.append(f"UNSOURCED-INT  {doc.name}: {value}")
        for pattern, template in LABELLED:
            for value in sorted(set(pattern.findall(text))):
                phrase = template.format(value.strip().replace(",", "").replace(" ", ""))
                if not sourced(phrase, blobs):
                    problems.append(f"UNSOURCED-LABEL  {doc.name}: {phrase!r}")

    for problem in problems:
        print(problem)
    print(
        f"checked {len(DOCS)} documents (including {len(SOURCES)} source files) against "
        f"{len(all_blobs)} artifacts, evaluated {len(DERIVED)} derived figures, re-ran the README "
        f"generator and both summary generators, and asserted every artifact is newer than the code "
        f"and produced by its recorded hash: {len(problems)} problem(s)"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
