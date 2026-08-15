# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Assert that every measured figure quoted in the multichip-decoder docs exists in a committed artifact.

The guard the three preceding stages carry (``../optimized_decoder/audit_figures.py``), pointed at this
stage's documents and evidence. Review round 3 of this stage asked for it in as many words, and the
reason it gave is the record: rounds 1, 2 and 3 each found quoted-figure errors — a stale
``--active-experts 63``, "2048 extra columns", "~26.7 GiB" mislabelled from GB, a PCC pair
(``0.999938/0.999924``) that appears in no log, an advisory count attributed to the wrong op family.
Every one of those is a figure that does not trace to an artifact, which is exactly what this checks.

**This is the core rule set, not the whole single-chip file.** Carried over: the labelled-measurement
token model, paragraph-scoped citation checking, the ``DERIVED`` expression table, the
labelled-phrase rule, the artifact-freshness pass and the self-test. Deliberately **not** carried:
the single-chip stage's bespoke checks (`check_orientation_claims`, `check_sparse_block_rule`,
`check_mirrored_constants`, `check_model_facts`, `check_commit_record`, `check_generators`), each of
which validates a claim structure that stage has and this one does not. They are named here so their
absence is a recorded decision rather than an oversight.

Passes over every document:

* **decimals** — every ``\\d+.\\d+`` must appear in some committed evidence file, as a *labelled*
  measurement rather than anywhere in a large log;
* **integers** — every 4+ digit integer must too, minus a declared set of shape/config constants;
* **labelled figures** — ``N passed`` and friends must appear in an artifact *together with their
  label*, because a bare integer will match something by accident in a large log.

Figures computed from other figures rather than read off a run are declared in ``DERIVED`` as an
**expression**, which is evaluated and must reproduce the quoted value; every literal in the
expression must itself be sourced.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/multichip_decoder/audit_figures.py
    python .../audit_figures.py --selftest    # writes logs/audit_selftest.txt
    python .../audit_figures.py --stamp       # writes logs/source_stamp.json, at the end of a sweep

Exit code 0 means every quoted figure traces to an artifact. ``logs/run_evidence.sh`` runs it last.

Two things it cannot do, stated so a reader does not over-trust it: it checks figures, not prose, so
it cannot tell that a *correct* number is being used to support a wrong claim — which was round 2's
and round 3's dominant finding — and it cannot see a figure that is simply absent.
"""

from __future__ import annotations

import gzip
import json
import random
import re
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parent
ROOT = DOC.parent.parent
CONTRACT = DOC.parent / "context_contract.json"

#: The implementation and tests the artifacts measure. Figures in their comments are checked like any
#: other document's, and every artifact must be newer than all of them.
SOURCES = [
    ROOT / "tt" / "multichip_decoder.py",
    ROOT / "tests" / "test_multichip_decoder.py",
]

#: Not stage-owned and **not** scanned for figures — that is the optimized stage's own audit — but
#: every artifact here measures them: this stage subclasses `OptimizedDecoder` and inherits its
#: config wholesale, so a change under it invalidates these measurements exactly as a change to
#: `multichip_decoder.py` would. Review round 4 pointed out that the fingerprint covered only the two
#: files this stage writes. `optimized_decoder.py` imports nothing else from `tt/` but
#: `model_config.py`, so these two are the closure.
INHERITED_SOURCES = [
    ROOT / "tt" / "optimized_decoder.py",
    ROOT / "tt" / "model_config.py",
]

#: What the sweep fingerprints and what freshness is decided against.
MEASURED_SOURCES = SOURCES + INHERITED_SOURCES

#: Documents whose figures must be sourced. The probe and harness scripts are included for the reason
#: the single-chip port gives: a script that explains why it exists is making a claim about a
#: measurement, and round 3 found exactly such a claim (``run_profiling.sh``'s ``3.54``) stale.
DOCS = [
    CONTRACT,
    DOC / "README.md",
    DOC / "work_log.md",
    *sorted(DOC.glob("logs/probe_*.py")),
    DOC / "logs" / "ab_layer_knobs.py",
    DOC / "logs" / "bench.py",
    DOC / "logs" / "run_evidence.sh",
    DOC / "tracy" / "run_profiling.sh",
    DOC / "watcher" / "census.py",
    *SOURCES,
]

#: Evidence a hand-written figure may come from: the structured logs, where every measurement is
#: printed as a labelled field. The per-op ``tt-perf-report`` tables are **not** here, for the reason
#: the single-chip audit measured: they are a dense soup of thousands of durations, so almost any
#: plausible microsecond figure appears in one by coincidence, and including them would make the
#: check close to a no-op for the figure class that has actually gone wrong here.
ARTIFACTS = [
    DOC / "logs" / "pytest_full_suite.txt",
    DOC / "logs" / "watcher_pytest.txt",
    DOC / "logs" / "watcher_pytest_eth_enabled.txt",
    DOC / "logs" / "ab_single_vs_multichip.txt",
    DOC / "logs" / "ab_layer_knobs.txt",
    DOC / "logs" / "probe_ccl.txt",
    DOC / "logs" / "probe_fused_ccl.txt",
    DOC / "logs" / "probe_dense_matmul.txt",
    DOC / "logs" / "probe_sparse_matmul_local.txt",
    DOC / "logs" / "probe_expert_parallel.txt",
    DOC / "logs" / "probe_footprint_local.txt",
    DOC / "logs" / "probe_decode_batch.txt",
    DOC / "watcher" / "census_summary.txt",
    DOC / "watcher" / "watcher_log.txt",
    DOC / "triage" / "triage-summary.txt",
    DOC / "triage" / "tt-triage.txt",
    CONTRACT,
    DOC.parent / "optimized_decoder" / "logs" / "probe_footprint.txt",
    DOC.parent / "optimized_decoder" / "logs" / "ab_fused_vs_optimized.txt",
    *sorted((DOC / "tracy").glob("*/*_tracy_run.txt")),
    *sorted((DOC / "tracy").glob("*/*_perf_report.console.txt")),
    # The stacked-by-op-code CSVs ARE in the pool, unlike the per-op tables: ~40 rows of aggregate
    # per op code, which is what the README's share-of-time table quotes, and small enough that
    # accidental matching is not the problem it is for a multi-megabyte per-op report.
    *sorted((DOC / "tracy").glob("*/*_perf_report_stacked.csv.gz")),
]

#: Required to exist and to be fresh, excluded from the sourcing pool. See ``ARTIFACTS``.
PER_OP_REPORTS = [
    *sorted((DOC / "tracy").glob("*/*_perf_report.txt")),
    *sorted((DOC / "tracy").glob("*/*_perf_report.txt.gz")),
    *sorted((DOC / "tracy").glob("*/*_perf_report.summary.txt")),
    *sorted((DOC / "tracy").glob("*/*_perf_report.summary.txt.gz")),
    *sorted((DOC / "tracy").glob("*/*_perf_report.csv.gz")),
    *sorted((DOC / "tracy").glob("*/*_perf_report_stacked.csv.gz")),
]

#: Written by hand after the sweep, or inherited, so an mtime older than the sources is expected.
EXEMPT_FROM_FRESHNESS = {
    "watcher_pytest_eth_enabled.txt",
    "watcher_pytest_eth_enabled.txt.gz",
    "probe_footprint.txt",
    "ab_fused_vs_optimized.txt",
    "context_contract.json",
    "tt-triage.txt",
    "triage-summary.txt",
}

#: Values computed from sourced numbers rather than measured. ``quoted -> (expression, what it is)``.
DERIVED: dict[str, tuple[str, str]] = {
    # Per-device footprint in GiB, from the measured byte counts.
    "0.311": ("333565956 / 1024 ** 3", "per-device full_attention layer at full context, GiB"),
    "0.125": ("134680576 / 1024 ** 3", "per-device linear_attention layer at full context, GiB"),
    # The 40-layer projection: 10 full_attention (full_attention_interval 4) + 30 linear_attention.
    "7376076840": ("10 * 333565956 + 30 * 134680576", "all 40 layers, per device, B"),
    "24389222440": ("10 * 839553028 + 30 * 533123072", "all 40 layers, one chip, B"),
    "6.87": ("(10 * 333565956 + 30 * 134680576) / 1024 ** 3", "all 40 layers, per device, GiB"),
    "22.71": ("(10 * 839553028 + 30 * 533123072) / 1024 ** 3", "all 40 layers, one chip, GiB"),
    "4.62": ("34091302912 / (10 * 333565956 + 30 * 134680576)", "per-device DRAM headroom, x"),
    "1.40": ("34091302912 / (10 * 839553028 + 30 * 533123072)", "single-chip DRAM headroom, x"),
    "24.88": ("(34091302912 - (10 * 333565956 + 30 * 134680576)) / 1024 ** 3", "per-device DRAM left, GiB"),
    "9.04": ("(34091302912 - (10 * 839553028 + 30 * 533123072)) / 1024 ** 3", "single-chip DRAM left, GiB"),
    "1426063360": ("10 * (71303168 + 71303168)", "paged KV cache, all layers, per device, B"),
    "2852126720": ("10 * (142606336 + 142606336)", "paged KV cache, all layers, one chip, B"),
    "1.33": ("10 * (71303168 + 71303168) / 1024 ** 3", "paged KV cache, all layers, per device, GiB"),
    "2.66": ("10 * (142606336 + 142606336) / 1024 ** 3", "paged KV cache, all layers, one chip, GiB"),
    "1017118720": ("248320 * 2048 * 2", "embedding or lm_head at bfloat16, B"),
    "31.75": ("34091302912 / 1024 ** 3", "measured allocatable DRAM per chip, GiB"),
    # The kv duplication: 256 extra weight columns of 2048 rows at bfloat8_b.
    "557056": ("256 * 2048 * 1.0625", "duplicated k/v projection columns per device per layer, B"),
    # The DeltaNet a/b gate padding: 2 gates x 24 zero columns of 2048 rows at bfloat8_b.
    "104448": ("2 * 24 * 2048 * 1.0625", "DeltaNet a/b gate tile padding per device per layer, B"),
    # The router, replicated, at bfloat16; and the one-hot expert_select block.
    "1048576": ("2048 * 256 * 2", "replicated router weight, B"),
    "32768": ("256 * 64 * 2", "one-hot expert_select block, B"),
    # The corrected prefill --active-experts input: 32*top_k draws, of which 1/tp land on this device.
    "40.6": ("64 * (1 - (1 - 1 / 64) ** (32 * 8 / 4))", "expected per-device per-group active experts"),
    # The zero-local-expert probability at top-8 over 4 devices.
    "3.512": (
        "sum(max(c) * __import__('math').comb(64, c[0]) * __import__('math').comb(64, c[1]) * "
        "__import__('math').comb(64, c[2]) * __import__('math').comb(64, c[3]) "
        "for c in [(a, b, d, 8 - a - b - d) for a in range(9) for b in range(9 - a) "
        "for d in range(9 - a - b)]) / __import__('math').comb(256, 8)",
        "E[max local active experts] over 4 devices at top-8",
    ),
}

#: Figures from a **superseded sweep** that the documents quote as history — "4 cores measured
#: 8.44-8.62 us on three sweeps and 9.98-10.29 on another". They cannot appear in the committed
#: artifacts by construction, because the committed artifacts are the last sweep. Listed individually
#: rather than exempted by pattern, so adding one is a deliberate act.
HISTORICAL = {
    # Values a review round found *wrong* and the documents now quote as the wrong value, in the
    # round-by-round record of what was corrected. They must not resolve against today's artifacts.
    "0.9935",  # round 5: the pre-round-4 sampled expert-partition PCC
    "52538",  # round 5: the stale watcher line count, quoted in the finding that corrected it
    "8.44",
    "8.62",
    "9.98",
    "10.29",
    "8.50",
    "8.77",
    "28.14",
    "28.68",
    "3.328",
    "3.389",
    "0.56",
    "0.30",
    "0.58",
    "0.68",
    "0.51",
    "3.538",
    "3.54",
    "0.026",
}

#: Constants and thresholds that are choices, not measurements.
ALLOWED = {
    "0.995",  # PCC_BAR, inherited from the functional stage
    "0.999",  # BASELINE_BAR, this stage's bar against the single-chip TTNN baseline
    "0.9999",
    "1.000",
    "1.0",
    "0.5",
    "0.05",
    "0.02",
    "0.01",
    "0.1",
    "1.0625",  # bfloat8_b bytes per element
    "0.5625",  # bfloat4_b bytes per element
    "1.0.35",  # part of the model name
    "3.5",  # "Qwen3.5", and "~3.5 iterations" in prose
    "2.0",
    "1.011",  # vs_ideal, sourced from probe_footprint_local.txt but also quoted bare in prose
    "0.002",  # the ms resolution the whole-layer A/B can distinguish, a stated threshold
    "1.42",  # the uncommitted raw ops CSV's gzipped size in MB, an environmental fact
    "1.52",
    "1500",  # "~1500 us", a deliberate rounding of a per-op-report row (excluded from the pool)
    "0.999938",
    "0.999924",  # historical: a PCC pair round 3 found quoted with no run behind it
    "1.4",  # DECODE_SPEEDUP_BAR: a policy threshold in the suite, not a measurement
    "48.0",  # PREFILL_MS_BAR: the same
}

#: Integers that are architecture/config/shape constants, section/round numbers or prose numbers.
#: Only 4+ digit integers are checked at all, so this list stays short.
ALLOWED_INT = {
    "2048",  # hidden size, prefill chunk, and a tensor dim in half the shapes here
    "1024",  # per-device o_proj K, and the packed gate/up width
    "2560",  # per-device attn_in width
    "3136",  # per-device gdn_in width
    # K+V summed: the artifact prints the two 142 606 336 B halves separately and README section 6's
    # table rolls them into one row, on both the per-device and the single-chip side.
    "285212672",
    "142606336",
    "10000",  # "~10 000 rows", an approximation of the uncommitted raw ops CSV's size
    "15232",  # Blackhole max packet payload, a constant of ccl_common.cpp's arithmetic
    "9216",  # global attn_in width
    "12352",  # global gdn_in width
    "4096",  # global o_proj/gdn_out K
    "1056",  # global shared_in width
    "262144",  # the advertised context
    "262141",  # the non-aligned full-context case
    "262143",  # the largest legal decode position
    "248320",  # vocab size
    "6000",  # the chunk-size-invariance length
    "8000",  # the largest tractable HF-golden length
    "5000",  # the unaligned max_context case
    "1465",  # the BFP8 collective row, quoted from the per-op report (excluded from the pool)
    "25600",  # the ACTIVE_ETH kernel config buffer
    "29040",  # the watcher-instrumented ERISC program size
    "2026",  # the year in dates
    "1024",
    "1532032",  # worker L1 unreserved, from the footprint probe header
    "1500",  # "~1500 us", a deliberate rounding of a per-op-report row (excluded from the pool)
    # Harness constants: L1 small size, torch seeds, thresholds, an upstream issue number.
    "24576",
    "16384",
    "1234",
    "1492",
    "2300",
    "3100",
    "3600",
    "9000",
    "50000",
    "500000",
    "45943",
    "29040",
    "25600",
}

DECIMAL = re.compile(r"(?<![\w.])(\d+\.\d+)(?![\w])")
INTEGER = re.compile(r"(?<![\w.])(\d{4,})(?![\w.])")
#: The same, as both documents actually format large numbers: groups of three separated by spaces.
SPACED_INTEGER = re.compile(r"(?<![\w.])(\d{1,3}(?: \d{3})+)(?![\w.])")

#: Values quoted with their label, which must appear together in an artifact. Counts below the 4-digit
#: integer threshold are invisible to `INTEGER`, so anything quoted *with* its label is checked this
#: way instead. Review round 5 found the watcher census's dump and detail-line counts stale in four
#: places for exactly that reason.
LABELLED = [
    re.compile(r"\b(\d+ (?:passed|skipped|failed|deselected))\b"),
    re.compile(r"\b(fatal-class matches: \d+)"),
]

#: ``(pattern, label)`` where the pattern captures the *number* and ``label`` is the word the artifact
#: prints next to it. Checked as "this number appears on a line that mentions this label", which
#: survives the artifacts writing it the other way round (``dumps: 58``) or with ANSI codes in
#: between, and unlike a bare-substring test it can actually fail. Review round 6 found the first
#: version of these two patterns capturing only the digits, which made the check ask whether "20"
#: appeared anywhere in 31 artifacts.
LABELLED_PAIRS = [
    (re.compile(r"\b(\d+) dumps\b"), "dumps"),
    (re.compile(r"\bfree over (\d+) detail lines\b"), "detail"),
]

#: An artifact filename mentioned in prose.
CITATION = re.compile(r"\b([a-z0-9_]+\.(?:txt|json|csv|md|py|sh))\b")

#: Cross-references that look like decimals and are not figures: section numbers, headings, limitation
#: and round numbers. Stripped before scanning, because otherwise every "§5.7" would have to be
#: whitelisted as a constant and the whitelist would then also excuse a real 5.7.
REFERENCES = [
    re.compile(r"§\s*\d+(?:\.\d+)?[a-z]?"),
    re.compile(r"\bsections?\s+\d+(?:\.\d+)?[a-z]?", re.I),
    re.compile(r"^#{1,6}\s+\d+(?:\.\d+)?[a-z]?\b", re.M),
    re.compile(r"\b(?:limitation|round|item|step|figure|table)s?\s+\d+(?:\.\d+)?", re.I),
    re.compile(r"\bv?\d+\.\d+(?:\.\d+)+\b"),  # version strings like 2.1 in "v2.1", 1.0.35
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),  # dates
    re.compile(r"\b\d{2}:\d{2}\b"),  # times
]


def strip_references(text: str) -> str:
    for pattern in REFERENCES:
        text = pattern.sub(" ", text)
    return text


#: Every way a committed artifact prints a *measurement*. Copied from the single-chip audit unchanged:
#: a figure is "sourced" when it equals one of these captured values exactly, not when its digits
#: happen to occur somewhere in a large file.
MEASURED = [
    re.compile(r"(?:^|\s)[a-zA-Z_][\w/.]*=(-?[\d.]+)"),
    re.compile(r"(-?[\d.]+)\s*(?:us|µs|μs|ms|GB/s|B|%|bytes)(?![\w.])"),
    re.compile(r'":\s*(-?[\d.]+)'),
    re.compile(r"(?:^|\s)(\d[\d,]*)\s+(?:TOTAL|passed|failed|deselected|lines?)\b"),
    re.compile(r":\s*(\d+)\s*$"),
    re.compile(r"(?:^|[|\s])(-?\d+\.?\d*)(?=[|\s]|$)"),
]


def read_blob(path: Path) -> str | None:
    """Text of ``path``, or of ``path.gz``. ``None`` if neither exists."""
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
    return path if path.is_file() else path.with_suffix(path.suffix + ".gz")


def load(paths) -> dict:
    blobs = {}
    for path in paths:
        body = read_blob(path)
        if body is not None:
            blobs[path] = body
    return blobs


def measured_tokens(blobs) -> set:
    """Every value any artifact prints as a labelled measurement, as exact strings."""
    tokens = set()
    for text in blobs.values():
        for pattern in MEASURED:
            for hit in pattern.findall(text):
                tokens.add(hit)
                tokens.add(hit.replace(",", ""))
                whole, _, frac = hit.partition(".")
                # Only for the 5+ decimal class, i.e. a PCC at full precision, which a document
                # legitimately quotes truncated. Doing it for 3-decimal timings would register
                # `1.07` for `1.075` and widen accidental matching in the weakest class.
                if len(frac) >= 5:
                    for cut in range(2, len(frac)):
                        tokens.add(f"{whole}.{frac[:cut]}")
    return tokens


def scoped_tokens(paragraph: str, blobs, cache: dict):
    """The token set a paragraph's figures must come from, if it names an artifact."""
    named = set(CITATION.findall(paragraph))
    if not named:
        return None
    scope = {p: b for p, b in blobs.items() if p.name in named or p.name.removesuffix(".gz") in named}
    if not scope:
        return None
    key = tuple(sorted(p.name for p in scope))
    if key not in cache:
        cache[key] = measured_tokens(scope)
    return cache[key]


def sourced(value: str, tokens) -> bool:
    plain = value.replace(",", "").replace(" ", "")
    return value in tokens or plain in tokens


def labelled_pair_in_artifacts(value: str, label: str, blobs) -> bool:
    """Does some artifact print ``value`` on a line that also mentions ``label``?"""
    for text in blobs.values():
        for line in text.splitlines():
            if label in line and re.search(rf"(?<![\w.]){re.escape(value)}(?![\w.])", line):
                return True
    return False


def phrase_in_artifacts(phrase: str, blobs) -> bool:
    """Is this exact ``value label`` phrase printed by an artifact?

    The phrase must be captured *whole* by its `LABELLED` pattern: review round 6 found two patterns
    capturing only the digits, which made the check ask whether the substring "20" appeared anywhere
    in 31 artifacts. It always does, so those two patterns could not fail. Anything added here needs
    its capture group around the label as well as the number.
    """
    assert not phrase.strip().isdigit(), f"LABELLED pattern captured a bare number ({phrase!r}), not a phrase"
    return any(phrase in text for text in blobs.values())


def derived_from_artifacts() -> set:
    """Figures the documents legitimately quote that are *aggregates* of artifact rows.

    Computed here rather than whitelisted, so they stay checked. Two families:

    * the per-category share-of-time percentages README section 5.4 quotes, which are sums over the
      stacked-by-op-code CSVs (an op family spans several rows, one per ``in0`` layout);
    * the speedup and efficiency figures, which are ratios over ``ab_single_vs_multichip.txt``.

    Both used to be hand-transcribed, and both were wrong at least once across rounds 1-3.
    """
    import csv
    import io

    tokens: set[str] = set()
    for path in sorted((DOC / "tracy").glob("*/*_perf_report_stacked.csv.gz")):
        body = read_blob(path)
        if body is None:
            continue
        rows = list(csv.DictReader(io.StringIO(body)))
        groups = (
            lambda r: r["Op Code"].startswith("SparseMatmul"),
            lambda r: r["Op Code"].startswith("TopK"),
            lambda r: r["Op Code"].startswith("MatmulDeviceOperation"),
            lambda r: "AllGather" in r["Op Code"] or "ReduceScatter" in r["Op Code"],
            lambda r: r["Op Category"] == "DM",
        )
        for pick in groups:
            total = sum(float(r["Total % [%]"]) for r in rows if pick(r))
            for digits in (1, 2):
                tokens.add(f"{total:.{digits}f}")
        for r in rows:
            tokens.add(r["Total % [%]"])
            tokens.add(r["Device Time Sum [\u03bcs]"])

    # The two per-layer collectives as the *per-op* prefill reports see them. The per-op reports are
    # out of the sourcing pool (multi-megabyte, so any 3-digit figure would find a spurious match),
    # but section 8's block-float anomaly is stated in terms of exactly these rows, so they are
    # extracted by op code here: two rows a report, and they move on every re-sweep.
    for path in sorted((DOC / "tracy").glob("*/prefill_perf_report.summary.txt*")):
        body = read_blob(path)
        if body is None:
            continue
        for share, micros in re.findall(
            r"^\s*\d+\s+([0-9.]+) %\s+ReduceScatterDeviceOperation\s+\d+\s+([0-9,]+) ", body, re.M
        ):
            tokens.add(share)
            tokens.add(micros)
            tokens.add(micros.replace(",", ""))

    # Whole-window `Op Category` shares, for this stage's captures and for the single-chip stage's
    # (README section 5.4 quotes both, the second as the control that says a category is inherited).
    import csv as _csv

    category_shares: dict = {}
    for root in (DOC / "tracy", DOC.parent / "optimized_decoder" / "tracy"):
        for path in sorted(root.glob("*/*_perf_report_stacked.csv.gz")):
            body = read_blob(path)
            if body is None:
                continue
            totals: dict = {}
            for row in _csv.DictReader(io.StringIO(body)):
                totals[row["Op Category"]] = totals.get(row["Op Category"], 0.0) + float(row["Total % [%]"])
            for value in totals.values():
                for digits in (1, 2):
                    tokens.add(f"{value:.{digits}f}")
            # Keyed by root as well: both trees hold a `linear_attention/decode_...` file and the
            # cross-tree delta is exactly what this feeds.
            category_shares.setdefault(f"{root.parent.name}/{path.parent.name}/{path.name}", {}).update(totals)
            # Per-op-name aggregates, which is how README section 5.4's `TM` table names them: one
            # `Op Code` per `in0` layout, summed by the op.
            by_op: dict = {}
            for row in _csv.DictReader(io.StringIO(body)):
                by_op[row["Op Code"].split("DeviceOperation")[0]] = by_op.get(
                    row["Op Code"].split("DeviceOperation")[0], 0.0
                ) + float(row["Total % [%]"])
            for value in by_op.values():
                for digits in (1, 2):
                    tokens.add(f"{value:.{digits}f}")
            # Per-op *rates*: a row's device time divided by its op count, which is how the anomaly
            # ledger and README limitation 7 quote the two collectives (a per-call cost is the only
            # way to compare rows whose op counts differ).
            for row in _csv.DictReader(io.StringIO(body)):
                count = int(row["Op Count"])
                if count:
                    rate = float(row["Device Time Sum [\u03bcs]"]) / count
                    for digits in (1, 2):
                        tokens.add(f"{rate:.{digits}f}")

    # The multichip-minus-single-chip delta per category and phase, which is the number the prose
    # uses to say a category is or is not this stage's doing.
    for name, mine in category_shares.items():
        for other, theirs in category_shares.items():
            if name != other:
                for cat, value in mine.items():
                    if cat in theirs:
                        tokens.add(f"{abs(value - theirs[cat]):.1f}")

    dense = read_blob(DOC / "logs" / "probe_dense_matmul.txt")
    if dense:
        # The `repeatability` column of README section 5.6: the widest disagreement between repeated
        # measurements of one realised config. Computed here so the column stays checked rather than
        # whitelisted — it is the threshold the whole table's verdicts turn on.
        repeats: dict[tuple, list] = {}
        for line in dense.splitlines():
            parts = line.split()
            if len(parts) >= 11 and parts[0] == "DENSE" and parts[5] == "mcast1d" and parts[9][0].isdigit():
                repeats.setdefault((parts[1], parts[6], parts[7]), []).append(float(parts[9]))
        widest: dict[str, float] = {}
        for (role, _, _), values in repeats.items():
            if len(values) > 1:
                widest[role] = max(widest.get(role, 0.0), max(values) - min(values))
        for value in widest.values():
            tokens.add(f"{value:.2f}")
        # The `delta` column of README section 5.6: inherited realised point minus the local winner.
        ladders: dict[str, list] = {}
        for line in dense.splitlines():
            parts = line.split()
            if len(parts) >= 11 and parts[0] == "DENSE" and parts[5] == "mcast1d" and parts[9][0].isdigit():
                ladders.setdefault(parts[1], []).append((float(parts[9]), int(parts[6]), int(parts[7])))
        for values in ladders.values():
            best = min(v[0] for v in values)
            for value, _, _ in values:
                tokens.add(f"{value - best:.2f}")

    # Differences between A/B arms, which the prose quotes as "worth N us/step" and "N ms a layer".
    ab = read_blob(DOC / "logs" / "ab_layer_knobs.txt")
    if ab:
        arms: dict[tuple, list] = {}
        for line in ab.splitlines():
            parts = line.split()
            if parts and parts[0] == "ABLAYER":
                arms.setdefault((parts[1], parts[2], parts[4]), []).append((float(parts[6]), float(parts[7])))
        knobs = {k[0] for k in arms}
        for knob in knobs:
            names = sorted({k[1] for k in arms if k[0] == knob})
            for kind in ("linear_attention", "full_attention"):
                for i, a in enumerate(names):
                    for b in names[i + 1 :]:
                        left, right = arms.get((knob, a, kind)), arms.get((knob, b, kind))
                        if not left or not right:
                            continue
                        for index, scale, digits in ((0, 1000.0, 0), (1, 1.0, 1), (1, 1.0, 2)):
                            for x in (min(v[index] for v in left), max(v[index] for v in left)):
                                for y in (min(v[index] for v in right), max(v[index] for v in right)):
                                    tokens.add(f"{abs(x - y) * scale:.{digits}f}")

    # The reduce-scatter-only saving the sharded-residual budget is built from.
    ccl = read_blob(DOC / "logs" / "probe_ccl.txt")
    if ccl:
        rows_ccl: dict[tuple, float] = {}
        for shape, arm, value in re.findall(r"^CCL (\S+) (\S+) trace ([0-9.]+)", ccl, re.M):
            rows_ccl[(shape, arm)] = float(value)
        for shape in {s for s, _ in rows_ccl}:
            full = rows_ccl.get((shape, "all_reduce_ring"))
            half = rows_ccl.get((shape, "rs_only_ring"))
            if full and half:
                tokens.add(f"{full - half:.2f}")
                tokens.add(f"{2 * (full - half):.2f}")

    bench = read_blob(DOC / "logs" / "ab_single_vs_multichip.txt")
    if bench:
        rows = {}
        for line in bench.splitlines():
            if not line.startswith("BENCH"):
                continue
            parts = line.split()
            tag = next(x for x in parts if x.startswith("tag="))[4:]
            layer = next(x for x in parts if x.startswith("layer="))[6:]
            phase = "prefill" if "prefill" in line else "decode"
            value = (
                float(line.split("wall=")[1].split()[0])
                if phase == "prefill"
                else float(line.split("wall/iter=")[1].split()[0])
            )
            rows[(tag, layer, phase)] = value
        for layer in ("0", "3"):
            for phase in ("prefill", "decode"):
                base = rows.get(("single-chip-baseline", layer, phase))
                multi = rows.get(("multichip", layer, phase))
                control = rows.get(("replication-control", layer, phase))
                if not base or not multi:
                    continue
                for digits in (2, 3):
                    tokens.add(f"{base / multi:.{digits}f}")
                for digits in (0, 1):
                    tokens.add(f"{100 * base / multi / 4:.{digits}f}")
                if control:
                    tokens.add(f"{100 * (control / base - 1):.1f}")
    return tokens


def check_derived(tokens) -> list:
    problems = []
    for quoted, (expression, label) in DERIVED.items():
        for literal in DECIMAL.findall(expression) + INTEGER.findall(expression):
            if literal in ALLOWED or literal in ALLOWED_INT or literal in DERIVED or sourced(literal, tokens):
                continue
            problems.append(f"DERIVED-OPERAND-UNSOURCED  {quoted} ({label}): {literal}")
        got = eval(expression)  # noqa: S307 - the table is source, not input
        digits = len(quoted.split(".")[1]) if "." in quoted else 0
        if f"{got:.{digits}f}" != quoted:
            problems.append(f"DERIVED-MISMATCH  {quoted} ({label}): {expression} = {got:.{digits + 2}f}")
    return problems


#: Written by ``logs/run_evidence.sh`` at the end of a sweep: the *behavioural* fingerprint of every
#: source the artifacts measure. See :func:`code_fingerprint`.
SOURCE_STAMP = DOC / "logs" / "source_stamp.json"


def code_fingerprint(path) -> str:
    """A hash of ``path``'s code with comments and docstrings removed.

    ``ast.parse`` discards comments outright and docstrings are the only string-valued bare
    expressions this codebase writes, so dropping those leaves exactly the behaviour. Used to answer
    the question the mtime check gets wrong: an artifact older than its source is stale only if the
    source's *behaviour* changed. Documentation edits after a sweep — which are most of the edits a
    review round produces — do not invalidate a 90-minute measurement, and the difference is decided
    mechanically here rather than by whoever is holding the pen.
    """
    import ast
    import hashlib

    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        node.body = [
            stmt
            for stmt in body
            if not (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            )
        ] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def write_source_stamp() -> None:
    SOURCE_STAMP.write_text(
        json.dumps({p.name: code_fingerprint(p) for p in MEASURED_SOURCES if p.is_file()}, indent=1, sort_keys=True)
        + "\n"
    )


def unchanged_since_stamp() -> set:
    """Sources whose behaviour is byte-for-byte what the last sweep measured."""
    if not SOURCE_STAMP.is_file():
        return set()
    try:
        stamped = json.loads(SOURCE_STAMP.read_text())
    except ValueError:
        return set()
    return {p for p in MEASURED_SOURCES if p.is_file() and stamped.get(p.name) == code_fingerprint(p)}


def check_freshness() -> list:
    """Every artifact must be newer than the code it measures, unless only its prose moved."""
    problems = []
    fresh = unchanged_since_stamp()
    candidates = [p for p in MEASURED_SOURCES if p.is_file() and p not in fresh]
    if fresh and not candidates:
        return []
    newest = max(((p.stat().st_mtime, p) for p in candidates), default=None)
    if newest is None:
        return ["MISSING-SOURCE  no implementation file found to check freshness against"]
    stamp, source = newest
    stamps = {
        artifact_path(p).stat().st_mtime
        for p in ARTIFACTS + PER_OP_REPORTS
        if exists(p) and p.name not in EXEMPT_FROM_FRESHNESS
    }
    # A `git archive` extraction stamps every file with the commit time, making ordering vacuous.
    if len(stamps | {stamp}) <= 1:
        return []
    for path in ARTIFACTS + PER_OP_REPORTS:
        if path.name in EXEMPT_FROM_FRESHNESS or "optimized_decoder" in path.parts or not exists(path):
            continue
        if artifact_path(path).stat().st_mtime < stamp:
            problems.append(
                f"STALE-ARTIFACT  {path.name} predates {source.name} - re-run logs/run_evidence.sh, "
                f"or restore the mtime if the source was only touched"
            )
    return problems


def check_missing() -> list:
    return [f"MISSING-ARTIFACT  {p.relative_to(DOC.parent)}" for p in ARTIFACTS + PER_OP_REPORTS if not exists(p)]


def contract_section(text: str) -> str:
    """Only the part of ``context_contract.json`` this stage owns.

    The file also carries the functional, fused and optimized stages' sections, whose figures are
    sourced by *their* artifacts and are not this stage's to re-derive. Scanning the whole file would
    either fail on their numbers or force their evidence into this pool, and the second is worse: it
    would widen the token set that every multichip figure is checked against.
    """
    import json

    data = json.loads(text)
    owned = {k: data[k] for k in ("stage", "target", "multichip_decoder") if k in data}
    return json.dumps(owned, indent=1)


def scan_documents(blobs, tokens) -> list:
    """Every decimal and 4+ digit integer in every document must trace to an artifact."""
    problems, cache = [], {}
    for doc in DOCS:
        text = read_blob(doc)
        if text is None:
            problems.append(f"MISSING-DOC  {doc}")
            continue
        if doc == CONTRACT:
            text = contract_section(text)
        for raw in re.split(r"\n\s*\n", text):
            paragraph = strip_references(raw)
            scope = scoped_tokens(paragraph, blobs, cache)
            pool = scope if scope is not None else tokens
            for value in DECIMAL.findall(paragraph):
                if value in ALLOWED or value in HISTORICAL or value in DERIVED or sourced(value, pool):
                    continue
                # A paragraph that cites an artifact may still quote a figure from the global pool
                # (a cross-reference to another section); fall back rather than fail on that alone.
                if scope is not None and sourced(value, tokens):
                    continue
                problems.append(f"UNSOURCED-DECIMAL  {doc.name}: {value}")
            # Space-separated thousands (`52 538`, `333 565 956`) are how both documents format byte
            # counts and line counts; the raw `INTEGER` pattern cannot see them, which review round 5
            # found hiding a stale line count. Normalise them into the same scan.
            for value in SPACED_INTEGER.findall(paragraph):
                packed = value.replace(" ", "")
                if packed in ALLOWED_INT or packed in DERIVED or sourced(packed, pool) or sourced(packed, tokens):
                    continue
                problems.append(f"UNSOURCED-INT  {doc.name}: {value}")
            for value in INTEGER.findall(paragraph):
                if value in ALLOWED_INT or value in DERIVED or sourced(value, pool) or sourced(value, tokens):
                    continue
                problems.append(f"UNSOURCED-INT  {doc.name}: {value}")
            for pattern in LABELLED:
                for phrase in pattern.findall(paragraph):
                    if not phrase_in_artifacts(phrase, blobs):
                        problems.append(f"UNSOURCED-LABELLED  {doc.name}: {phrase!r}")
            for pattern, label in LABELLED_PAIRS:
                for value in pattern.findall(paragraph):
                    if not labelled_pair_in_artifacts(value, label, blobs):
                        problems.append(f"UNSOURCED-LABELLED  {doc.name}: {value} {label}")
    return problems


def selftest(tokens) -> str:
    """How often an *arbitrary* value comes back sourced, per figure class.

    The number a reader needs in order to know what a pass is worth. Written to
    ``logs/audit_selftest.txt`` so it is an artifact rather than a claim in this docstring.
    """
    rng = random.Random(11)
    lines = ["# audit_figures.py --selftest: false-positive rate per figure class", "# class trials hits rate"]
    classes = {
        "1-decimal-us": lambda: f"{rng.uniform(1, 800):.1f}",
        "2-decimal": lambda: f"{rng.uniform(0, 100):.2f}",
        "3-decimal-ms": lambda: f"{rng.uniform(0, 5):.3f}",
        "6-decimal-pcc": lambda: f"0.{rng.randint(999000, 999999):06d}"[:8],
        "byte-count": lambda: str(rng.randint(10**6, 10**9)),
    }
    for name, make in classes.items():
        trials = 2000
        hits = sum(1 for _ in range(trials) if sourced(make(), tokens))
        lines.append(f"{name} {trials} {hits} {hits / trials:.4f}")
    return "\n".join(lines) + "\n"


def main() -> int:
    blobs = load(ARTIFACTS)
    tokens = measured_tokens(blobs) | derived_from_artifacts()
    if "--selftest" in sys.argv[1:]:
        out = DOC / "logs" / "audit_selftest.txt"
        out.write_text(selftest(tokens))
        print(f"wrote {out}")
        return 0
    if "--stamp" in sys.argv[1:]:
        write_source_stamp()
        print(f"wrote {SOURCE_STAMP}")
        return 0
    problems = check_missing() + check_derived(tokens) + scan_documents(blobs, tokens) + check_freshness()
    for problem in problems:
        print(problem)
    print(f"{len(problems)} problem(s); {len(blobs)} artifacts, {len(tokens)} measured tokens")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
