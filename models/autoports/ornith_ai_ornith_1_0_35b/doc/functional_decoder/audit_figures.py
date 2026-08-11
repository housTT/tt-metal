# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Assert that every measured figure quoted in the stage docs exists in a committed artifact.

Four consecutive review rounds found numbers in the documentation that came from superseded runs.
This is the durable guard. It makes three passes over every stage document:

* **decimals** — every ``\\d+.\\d+`` must appear in some committed evidence file;
* **integers** — every 2+ digit integer must too, minus a declared set of shape/config constants
  (round 3's defect was an integer, so bare decimals are not enough);
* **labelled figures** — ``N passed``, ``N-line log``, ``N bytes free`` and friends must appear in an
  artifact *together with their label*, because a bare two- or three-digit integer will match
  something by accident somewhere in an 18 564-line watcher log.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/functional_decoder/audit_figures.py

Exit code 0 means every quoted figure traces to an artifact. Run it after any doc edit and after any
re-run that changes the numbers.

Known residual limits, so a reader does not over-trust it: substring matching means a figure can be
"sourced" by an unrelated occurrence of the same digits in a large log (the labelled pass exists to
cover the cases where that actually matters), and it checks figures, not prose — a sentence
describing what another document contains is outside its reach.
"""

import re
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parent
CONTRACT = DOC.parent / "context_contract.json"

#: Documents whose figures must be sourced.
DOCS = [
    DOC / "README.md",
    DOC / "work_log.md",
    DOC / "tracy/PROVENANCE.md",
    DOC / "watcher/CLASSIFICATION.md",
    CONTRACT,
]

#: Evidence the figures may come from.
ARTIFACTS = [
    DOC / "logs/pcc_summary.txt",
    DOC / "logs/summarise_pcc_run.txt",
    DOC / "logs/pytest_full_suite.txt",
    DOC / "logs/watcher_pytest.txt",
    DOC / "logs/dram_capacity_probe.txt",
    DOC / "logs/commit_record.txt",
    DOC / "logs/router_precision_ab.txt",
    DOC / "logs/router_setmatch_reconcile.txt",
    DOC / "logs/probe_gated_delta_rule_op.txt",
    DOC / "logs/probe_moe_vs_hf.txt",
    DOC / "tracy/perf_summary.txt",
    DOC / "tracy/slow_ops_summary.txt",
    *sorted((DOC / "tracy").glob("*/*_tracy_run.txt")),
    *sorted((DOC / "tracy").glob("*/*_perf_report.txt")),
    DOC / "watcher/watcher_log.txt",
    DOC / "watcher/census_summary.txt",
    DOC / "weight_stats_layer0.json",
    DOC / "weight_stats_layer3.json",
    CONTRACT,
]

#: Values computed from sourced numbers rather than measured, so they have no artifact of their own.
#:
#: Each maps the quoted string to ``(expression, what it is)``. The expression is **evaluated** and
#: must reproduce the quoted value to the precision it is quoted at, and every numeric literal in it
#: must itself be sourced in an artifact. A comment recording the operands is not enough: round 9
#: found README quoting a declared ratio for a *different* comparison than the one it was derived
#: from, and a comment cannot catch that — an evaluated expression names the basis, so quoting the
#: wrong basis now means the expression no longer reproduces the number.
DERIVED = {
    # Device kernel time vs the unprofiled wall clock (README §6, PROVENANCE).
    "0.14": ("100 * (338.22 / 337.759 - 1)", "linear prefill, unprofiled wall vs device"),
    "0.21": ("100 * (317.04 / 316.373 - 1)", "full prefill, unprofiled wall vs device"),
    "3.2": ("100 * (2.614 / 2.533 - 1)", "linear traced decode, unprofiled wall vs device"),
    "2.3": ("100 * (2.393 / 2.340 - 1)", "full traced decode, unprofiled wall vs device"),
    # Profiled wall clock vs device kernel time: host dispatch *plus* instrumentation.
    "5.6": ("100 * (2.676 / 2.533 - 1)", "linear traced decode, profiled wall vs device"),
    "4.7": ("100 * (2.451 / 2.340 - 1)", "full traced decode, profiled wall vs device"),
    # Profiled vs unprofiled wall clock: instrumentation alone, on a like-for-like basis. Both decode
    # windows round to the same 2.4 %; declared once, with both bases named in the label.
    "2.4": ("100 * (2.676 / 2.614 - 1)", "linear traced decode, profiler cost (full is 2.4 % too)"),
    # 0.21 also covers linear prefill's profiler cost, 338.94 vs 338.22 ms — same magnitude, and the
    # expression above for the device basis is the one README quotes it for first.
    # MoE share of a window: the two SparseMatmul rows of README §6 added together.
    "80.2": ("45.1 + 35.1", "MoE share of linear prefill device time"),
    "80.7": ("45.8 + 34.9", "MoE share of full prefill device time"),
    "34.9": ("21.3 + 13.6", "MoE share of linear decode device time"),
    "37.8": ("23.1 + 14.7", "MoE share of full decode device time"),
    "0.19": ("100 * 0.001913", "fp32-router score-vector L1 relative error as a percentage"),
    "1.15": ("100 * 0.01151", "bf16-router score-vector L1 relative error, work log §3"),
    "2.12": ("2276982784 / 1024 ** 3", "worst-case layer footprint in GiB"),
    "67.6": ("67633152 / 1000 ** 2", "RoPE cos/sin tables in MB"),
}

#: ``full_attention`` prefill is 316.98 ms profiled against 317.04 ms unprofiled, i.e. 0.02 % *faster*.
#: A negative deviation, kept in its own table so the sign is explicit rather than implied by prose,
#: and so a magnitude collision with a positive entry stays deliberate.
DERIVED_NEGATIVE = {
    "0.02": ("100 * (1 - 316.98 / 317.04)", "full prefill, profiler cost (negative: noise floor)"),
}

#: Figures the work log quotes *because they were wrong* — the superseded numbers a review round
#: found. They must not appear in README/PROVENANCE/CLASSIFICATION, and the audit enforces that by
#: only allowing them in work_log.md. Keep this set to values that are *both* superseded *and* still
#: quoted somewhere: a dead entry is noise, and a value that is also a live measurement (as
#: ``0.999902`` is — it is the real permuted-page-table decode PCC) must not be listed here at all,
#: because listing it would make the exemption inert and hide a genuine drift.
#: Entries are checked against two rules when this set changes: each must still be quoted somewhere
#: in ``work_log.md`` (a dead entry is noise), and none may also be a live measured value or a value
#: that occurs legitimately in an artifact (that would make the exemption inert — the bare ``1356``
#: was dropped for exactly that reason: the watcher log of the day had a genuine ``1356 bytes free``
#: line for one core. The re-measured log has no stack-usage block at all, so the bare value is kept
#: out and only the labelled phrases are exempted, in ``HISTORICAL_LABELLED``).
HISTORICAL = {
    "130.79",  # superseded watcher-run wall clock (work log §11 round 3)
    "22819",  # superseded watcher-log line count
    "0.0045",  # decode PCC under the pad-aliasing use-after-free, before the fix
    "2276458496",  # worst-case per-layer bytes before the RoPE table was lengthened (work log §9)
    # Superseded by the round-7 re-measurement (work log §14), quoted there to show what moved.
    "104.80",  # previous watcher-run wall clock
    "22891",  # previous watcher-log line count
    "337.638",  # device kernel time, linear prefill, measurement round 1
    "2.534",  # device kernel time, linear traced decode, round 1
    "316.178",  # device kernel time, full prefill, round 1
    # Superseded by the round-12 re-measurement after the pre-commit reformat (work log §18).
    "337.683",  # device kernel time, linear prefill, round 2
    "316.105",  # device kernel time, full prefill, round 2
    "2.339",  # device kernel time, full traced decode, round 2
    "2.392",  # unprofiled wall clock, full traced decode, round 2
    "2.678",  # profiled wall clock, linear traced decode, round 2
    "2.450",  # profiled wall clock, full traced decode, round 2
    # The previous minimum stack headroom, 1212, is deliberately NOT listed: the bare digits occur in
    # the perf reports as an op-row index, so a bare-integer exemption for it would be inert. The two
    # phrase entries in HISTORICAL_LABELLED are the real guard, and neither string is in any artifact.
}

#: Constants and thresholds that are choices, not measurements, so they have no artifact.
ALLOWED = {
    "0.995",  # the PCC acceptance bar
    "0.999",  # the chunk-invariance assertion bar
    "0.9999",  # the RoPE assertion bar
    "1.0",
    "0.25",  # partial_rotary_factor
    "1.2",  # tt-perf-report major.minor in "1.2.8"
    "3.5",  # "Qwen3.5"
    "3.6",
    "1.0.35",  # part of the model name
}

#: Integers that are architecture/config/shape constants or plain prose numbers rather than measured
#: results, so they have no artifact to trace to. Round 3's defect was an integer ("13 passed",
#: "22 819-line log"), so integers are audited too — but only with this set declared, otherwise every
#: hidden-size and tile count in the contract section would be a false positive.
ALLOWED_INT = {
    # model / config constants
    "35",
    "40",
    "30",
    "10",
    "16",
    "32",
    "2048",
    "256",
    "512",
    "8192",
    "4096",
    "1024",
    "128",
    "64",
    "262144",
    "262143",
    "131072",
    "12",
    "24",
    "48",
    "96",
    "192",
    "384",
    "768",
    "1536",
    "3072",
    "1e7",
    "10000000",
    "100",
    "1000",
    "10000",
    "50000",
    "20",
    "50",
    "200",
    "500",
    "5000",
    "8000",
    "2026",
    "4",
    "8",
    "2",
    "1",
    "3",
    "5",
    "6",
    "7",
    "9",
    "11",
    "13",
    "14",
    "15",
    "17",
    "18",
    "19",
    "21",
    "22",
    "23",
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
    "60",
    "70",
    "80",
    "90",
    "110",
    "120",
    "130",
    "160",
    "250",
    "300",
    "320",
    "400",
    "600",
    "640",
    "2049",
    "3000",
    "4095",
    "16384",
    "32768",
    "65536",
    "1e-6",
    "1e-5",
    "1e-9",
    "8064",  # align_up(8000, 128), the window end in the work log's bug-4a narrative
}

#: A decimal, possibly sentence-final. The trailing lookahead allows a `.` only when it is *not*
#: followed by another digit, so `419.37.` at the end of a sentence is still checked (an earlier cut
#: excluded `.` outright and silently skipped every sentence-final figure) while `1.0.35` is not
#: mistaken for a decimal.
DECIMAL = re.compile(r"(?<![\d.])\d{1,12}\.\d{1,6}(?!\.?\d)")
#: Integers of two digits or more, not part of a decimal, a version string or an identifier. No upper
#: length bound: the byte-level capacity figures in the context contract are 11 digits, and capping
#: the length is how the previous cut managed to skip them.
INTEGER = re.compile(r"(?<![\d.\w])\d{2,}(?!\.?\d)(?!\w)")

#: Figures the docs quote *with a unit or label*. A bare small integer will match something
#: somewhere in an 18 564-line watcher log by accident, so for these the **number and its label**
#: must appear together in an artifact. This is the shape of the round-3 defect ("13 passed",
#: "22 819-line log", "1 356 bytes free"), so it gets the strict check.
LABELLED = [
    (re.compile(r"(\d[\d ,]*) passed"), "{} passed"),
    (re.compile(r"(\d[\d ,]*)[- ]line log"), "{} lines"),
    (re.compile(r"(\d[\d ,]*) bytes free"), "{} bytes free"),
    # A bare `N bytes free` can match one core's headroom rather than the minimum, so the label above
    # cannot catch a wrong *minimum* on a log that reports per-core lines. Bind the word. (The
    # re-measured log has no stack-usage block, so this currently guards a shape no artifact has —
    # kept because the shape returns whenever watcher does emit the block.)
    (re.compile(r"minimum[^.]{0,80}?(\d[\d ,]*) bytes free"), "minimum stack headroom: {} bytes free"),
    (re.compile(r"headroom[^.]{0,80}?(\d[\d ,]*) bytes free"), "minimum stack headroom: {} bytes free"),
    (re.compile(r"(\d[\d ,]*) metric lines"), "{} metric lines"),
    (re.compile(r"(\d[\d ,]*) fatal-class"), "fatal-class matches: {}"),
]

#: Labelled figures the work log quotes because a review round found them wrong. Same rule as
#: ``HISTORICAL``, but keyed on the whole phrase, so exempting "13 passed" does not also exempt a
#: bare 13 (which is a tested batch size) anywhere in the docs.
HISTORICAL_LABELLED = {
    "13 passed",
    "60 passed",
    "1212 bytes free",  # the superseded watcher run's minimum headroom
    "minimum stack headroom: 1212 bytes free",
    "1356 bytes free",  # the round-2 defect's wrong headroom
    "minimum stack headroom: 1356 bytes free",
}

#: Values the work log quotes as the *deliberately fabricated* inputs of the mutation test that
#: proves this script fires. They are unsourced on purpose — that is the point of quoting them — and
#: like ``HISTORICAL`` they are permitted only in ``work_log.md``.
MUTATION_TEST = {
    "419.37",
    "87654",
    "1234567.89",
    "5.7244",  # the wrong-basis value the DERIVED mutation test produces, quoted in §16
    "71 passed",
    "84 passed",
}


#: ``context_contract.json``'s capacity figures are byte counts computed from the config, not values
#: read off a run, so grepping an artifact for them is the wrong check — the right one is arithmetic.
#: ``check_contract`` recomputes each of these from the formula the contract itself states, so they are
#: exempt from the grep pass and covered by a stronger check instead.
CONTRACT_COMPUTED = {
    "268435456",  # paged KV cache, per tensor: 4096 blocks x 2 kv heads x 64 tokens x 256 dim x 2 B
    "67633152",  # RoPE tables: 2 x 264192 rows x 64 rope dims x 2 B
    "264192",  # align_up(262144, 2048) + 2048
    "1610612736",  # routed experts: 3 x 256 experts x 2048 hidden x 512 intermediate x 2 B
    "7340032",  # shared expert 3 x 2048 x 512 x 2 B + router 256 x 2048 x 2 B
    "54525952",  # full_attention projections: q 2048x8192 + k,v 2048x512 each + o 4096x2048, x 2 B
    "67436544",  # linear_attention projections (in_proj q/k/v/b/a, conv, gate, out_proj), x 2 B
    "2097152",  # DeltaNet recurrent state, batch 1: 32 v heads x 128 x 128 x 4 B
    "49152",  # DeltaNet conv state, batch 1: 3 x 8192 x 2 B
    "2276982784",  # sum of the six components a full_attention layer needs at full context
    "34091302912",  # 31.75 GiB, the measured allocatable DRAM
    # The fused packing is byte-neutral on device, so it contributes no new figures here. An earlier
    # revision of the contract claimed a +131072 B per-layer delta and quoted two changed totals; that
    # was wrong. `moe_shared_and_router_weights`' formula below omits the shared router's tile padding
    # (tt/moe.py stores it as [1, 1, dim, 1] in TILE_LAYOUT, i.e. dim x 32 x 2 B), and recomputing the
    # fused packing from the formula alone made an omission look like growth. The formula is left as
    # the functional stage measured it; the contract now records the omission instead of a delta.
}


#: Width the shared expert's packed gate+up+router occupies in either spelling, in bf16 columns.
#: Read from the model config rather than written here, so the check below compares two independent
#: things: round 16 found the first version comparing a constant against the identical literal.
def shared_expert_packed_columns():
    """``2 * moe_intermediate + one padded tile``, from the model config the decoders actually use."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ornith_ai_ornith_1_0_35b.reference import hf_reference as R  # noqa: PLC0415

    # 32 is the tile width, a hardware constant; the intermediate size is the model's, read from the
    # checkpoint config. Neither is a figure this stage chose, which is the point of the check.
    # It is the *shared* expert's intermediate size that sizes this packing, not the routed experts'
    # - they happen to be equal here, which round 17 flagged as making the check pass by coincidence.
    return 2 * int(R.load_text_config().shared_expert_intermediate_size) + 32


def check_contract(contract: Path) -> list:
    """Recompute ``context_contract.json``'s byte figures from the formulas it states.

    The contract is both a document and an artifact, so its own figures cannot be sourced by grep
    without becoming self-sourcing. These are arithmetic, so check the arithmetic.
    """
    import json

    problems = []
    data = json.loads(contract.read_text())
    cap = data["capacity_evidence"]
    f = cap["full_context_footprint_per_layer_bytes"]
    ctx = data["supported"]["context_length"]
    chunk = data["internal_shape_policy"]["internal_prefill_chunk_tokens"]
    block = data["internal_shape_policy"]["paged_block_size_tokens"]

    def expect(name, got, want):
        if got != want:
            problems.append((contract.name, f"{name} = {got}, recomputes to {want}"))

    expect("measured_allocatable_dram_bytes", cap["measured_allocatable_dram_bytes"], int(31.75 * 1024**3))
    kv = (ctx // block) * 2 * block * 256 * 2
    expect("paged_kv_cache_k", f["paged_kv_cache_k"], kv)
    expect("paged_kv_cache_v", f["paged_kv_cache_v"], kv)
    rope_rows = -(-ctx // chunk) * chunk + chunk
    expect("rope_cos_sin_tables", f["rope_cos_sin_tables"], 2 * rope_rows * 64 * 2)
    expect("moe_routed_expert_weights", f["moe_routed_expert_weights"], 3 * 256 * 2048 * 512 * 2)
    expect("moe_shared_and_router_weights", f["moe_shared_and_router_weights"], 3 * 2048 * 512 * 2 + 256 * 2048 * 2)
    expect(
        "full_attention_projection_weights",
        f["full_attention_projection_weights"],
        (2048 * 2 * 16 * 256 + 2 * 2048 * 2 * 256 + 16 * 256 * 2048) * 2,
    )
    expect("deltanet_recurrent_state_batch1", f["deltanet_recurrent_state_batch1"], 32 * 128 * 128 * 4)
    expect("deltanet_conv_state_batch1", f["deltanet_conv_state_batch1"], 3 * 8192 * 2)
    total = (
        f["paged_kv_cache_k"]
        + f["paged_kv_cache_v"]
        + f["rope_cos_sin_tables"]
        + f["moe_routed_expert_weights"]
        + f["moe_shared_and_router_weights"]
        + f["full_attention_projection_weights"]
    )
    expect("worst_case_layer_total", f["worst_case_layer_total"], total)
    if f["worst_case_layer_total"] >= cap["measured_allocatable_dram_bytes"]:
        problems.append((contract.name, "worst_case_layer_total exceeds measured allocatable DRAM"))

    # The fusing stage packs the shared expert's gate, up and router into one tensor. That is only
    # byte-neutral if the packed width equals the unpacked widths plus the router's tile padding, so
    # assert the arithmetic rather than the prose: a future packing that actually grew the layer would
    # have to change this number, and `footprint_change` claims byte-neutrality on device.
    prose = data.get("fused_decoder", {}).get("footprint_change", "")
    if prose:
        # README §3.1 tabulates this width; the contract's byte-neutrality claim depends on it.
        want_cols = shared_expert_packed_columns()
        if str(want_cols) not in prose:
            problems.append(
                (contract.name, f"footprint_change does not quote the packed shared-expert width {want_cols}")
            )
        if "Byte-neutral on device" not in prose:
            problems.append((contract.name, "footprint_change no longer claims device byte-neutrality"))
        for stale in ("131072", "7471104", "2277113856"):
            if stale in prose:
                problems.append((contract.name, f"footprint_change quotes {stale}, a delta this stage does not have"))
    return problems


def check_derived(haystack: str) -> list:
    """Evaluate every ``DERIVED`` expression and confirm it reproduces the value it is declared for.

    Also confirms each operand is itself sourced, so a derived figure cannot rest on an invented one.
    """
    problems = []
    for table in (DERIVED, DERIVED_NEGATIVE):
        for value, (expression, label) in table.items():
            try:
                got = eval(expression, {"__builtins__": {}}, {})  # literals and arithmetic only
            except Exception as exc:  # a malformed declaration is itself a finding
                problems.append(("DERIVED", f"{value} ({label}): cannot evaluate {expression!r}: {exc}"))
                continue
            places = len(value.split(".")[1]) if "." in value else 0
            if f"{got:.{places}f}" != value:
                problems.append(("DERIVED", f"{value} ({label}): {expression} = {got:.4f}, not {value}"))
            for operand in re.findall(r"\d+\.\d+|\d{2,}", expression):
                if normalise(operand) not in haystack:
                    problems.append(("DERIVED", f"{value} ({label}): operand {operand} is unsourced"))
    return problems


def check_no_orphans() -> list:
    """Every exemption must still be earning its place in a document.

    An exemption is only safe while the figure it exempts is actually quoted. Once a document's number
    drifts away, the stale entry sits there silently exempting a value nothing claims — and worse, it
    exempts that magnitude for *any* future meaning. The two ``HISTORICAL`` rules used to be a comment
    asking a human to check this; this is the code.
    """
    docs = "\n".join(p.read_text(errors="replace") for p in DOCS if p.is_file())
    joined = normalise(docs)
    work_log = (DOC / "work_log.md").read_text(errors="replace") if (DOC / "work_log.md").is_file() else ""
    problems = []
    for name, table in (
        ("DERIVED", DERIVED),
        ("DERIVED_NEGATIVE", DERIVED_NEGATIVE),
        ("HISTORICAL", HISTORICAL),
        ("MUTATION_TEST", MUTATION_TEST),
    ):
        for value in table:
            # work-log-only sets must be quoted there specifically; the rest anywhere in the docs.
            where = normalise(work_log) if name in ("HISTORICAL", "MUTATION_TEST") else joined
            if normalise(value) not in where:
                problems.append((name, f"{value} is exempted but no longer quoted — drop it"))

    # HISTORICAL_LABELLED holds phrases the LABELLED templates *produce*, e.g. "1212 bytes free" from
    # a sentence reading "…is 1 212 bytes free". So the reachability test is whether some pattern
    # still yields the phrase from the work log, not whether the phrase appears verbatim.
    reachable = set()
    for pattern, template in LABELLED:
        for raw in pattern.findall(work_log):
            reachable.add(template.format(normalise(raw)))
    for phrase in HISTORICAL_LABELLED:
        if phrase not in reachable:
            problems.append(("HISTORICAL_LABELLED", f"{phrase!r} is exempted but unreachable — drop it"))

    # The second HISTORICAL rule, previously only a comment: an entry that also occurs in an artifact
    # is *inert*, because the value would have passed the grep anyway. Round 5 found exactly that
    # (``0.999902`` listed as historical while also being the real permuted-page-table decode PCC), so
    # the rule exists; this is it in code rather than in prose asking a human to check.
    artifacts = normalise("\n".join(p.read_text(errors="replace") for p in ARTIFACTS if p.is_file()))
    # MUTATION_TEST is included for the same reason: a fabricated value that starts occurring in an
    # artifact would be exempted for nothing, and would stop being a valid mutation-test input.
    for name, values in (
        ("HISTORICAL", HISTORICAL),
        ("HISTORICAL_LABELLED", HISTORICAL_LABELLED),
        ("MUTATION_TEST", MUTATION_TEST),
    ):
        for value in values:
            if normalise(value) in artifacts:
                problems.append((name, f"{value!r} also occurs in an artifact, so the exemption is inert — drop it"))
    return problems


def normalise(text: str) -> str:
    """Digits only, plus the separators, so `22 891` matches `22891` and `1 212` matches `1212`."""
    return text.replace(" ", "").replace(",", "").replace("_", "")


def haystack_for(doc: Path) -> str:
    """Every artifact except ``doc`` itself.

    ``context_contract.json`` is both a document (its figures must be sourced) and an artifact (the
    README quotes it). Auditing it against a haystack containing itself would make every figure in it
    trivially self-sourced, so the document under audit is always excluded.
    """
    return normalise("\n".join(p.read_text(errors="replace") for p in ARTIFACTS if p.is_file() and p != doc))


def main() -> int:
    failures = []
    for doc in DOCS:
        haystack = haystack_for(doc)
        if not doc.is_file():
            failures.append((doc.name, "MISSING DOCUMENT"))
            continue
        text = doc.read_text(errors="replace")
        # `22 891` and `1,212` are one figure written with separators; join them before matching so
        # the integer pass sees the number a reader sees.
        joined = re.sub(r"(?<=\d)[ ,](?=\d\d\d(?!\d))", "", text)
        candidates = [(v, ALLOWED) for v in sorted(set(DECIMAL.findall(text)))]
        candidates += [(v, ALLOWED_INT) for v in sorted(set(INTEGER.findall(joined)))]
        for value, allowed in candidates:
            if value in allowed or value in DERIVED or value in CONTRACT_COMPUTED:
                continue
            if doc.name == "work_log.md" and (value in HISTORICAL or value in MUTATION_TEST):
                continue
            if normalise(value) not in haystack:
                failures.append((doc.name, value))

        for pattern, template in LABELLED:
            for raw in set(pattern.findall(text)):
                value = normalise(raw)
                phrase = template.format(value)
                if doc.name == "work_log.md" and (
                    value in HISTORICAL or phrase in HISTORICAL_LABELLED or phrase in MUTATION_TEST
                ):
                    continue
                if normalise(phrase) not in haystack:
                    failures.append((doc.name, phrase))

    arithmetic = check_derived(haystack_for(Path("/nonexistent")))
    arithmetic += check_no_orphans()
    if CONTRACT.is_file():
        arithmetic += check_contract(CONTRACT)
    for name, value in failures:
        print(f"UNSOURCED  {name}: {value}")
    for name, message in arithmetic:
        print(f"ARITHMETIC {name}: {message}")
    failures += arithmetic
    print(
        f"checked {len(DOCS)} documents against {sum(p.is_file() for p in ARTIFACTS)} artifacts, "
        f"and recomputed {len(DERIVED) + len(DERIVED_NEGATIVE)} derived figures: "
        f"{len(failures)} problem(s)"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
