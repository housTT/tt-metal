# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Account for every ``critical``-level line in the passing suite log.

The suite passes, but its log is not silent: ``_prepare_conv1d_weights`` decides whether
``ttnn.conv1d`` can serve a given ``(batch, block length)`` by *executing* the program once at setup
and catching the failure, and a caught TT_FATAL/TT_THROW still prints at ``critical`` level. A reader
scanning the log finds those lines and has no way to tell a deliberately provoked, caught failure
from a real one — round 3 of this stage's review raised exactly that.

So they are classified rather than explained away: every ``critical`` line must match one of the two
patterns the conv probe can produce, and the script prints the resulting per-batch coverage. Anything
else is an unclassified critical and exits non-zero.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/fused_decoder/logs/classify_suite_criticals.py

Run by ``run_evidence.sh`` right after the suite, writing ``logs/suite_criticals.txt``.
"""

import re
import sys
from collections import Counter
from pathlib import Path

L = Path(__file__).resolve().parent
LOG = L / "pytest_full_suite.txt"
OUT = L / "suite_criticals.txt"

#: The failure classes the setup-time conv probe deliberately provokes and catches. Both are L1
#: capacity refusals for a (batch, length) whose conv program does not fit; the layer drops that
#: shape and uses the FIR form for it.
EXPECTED = {
    "conv-probe-l1-buffer": re.compile(r"TT_FATAL: Out of Memory: Not enough space to allocate .* L1 buffer"),
    "conv-probe-cb-overflow": re.compile(r"TT_THROW: Statically allocated circular buffers .* beyond max L1 size"),
}

#: Per-bank L1 capacity on this part, used to tell a hard overflow from a pressure-dependent refusal.
BANK_B = 1436800
NEED = re.compile(r"allocate (\d+) B .*?across (\d+) banks")

COVERAGE = re.compile(r"layer (?P<layer>\d+) batch (?P<batch>\d+): ttnn\.conv1d accepted (?P<n>\d+)/(?P<total>\d+)")


def main() -> int:
    if not LOG.is_file():
        print(f"MISSING {LOG}")
        return 1
    text = LOG.read_text(errors="replace")
    counts, unclassified = Counter(), []
    # Which refusal class each candidate shape hit, attributed to the sweep it belongs to: the probe
    # emits its refusals and then the decoder logs the resulting coverage, so the criticals since the
    # previous coverage line are that batch's. Review round 13 found the documents attributing the
    # whole coverage fall to circular-buffer overflow when that is the minority class at every batch
    # above 4 - so the split is measured here rather than asserted in prose.
    per_batch: dict[tuple[int, int], Counter] = {}
    per_batch_sub: dict[tuple[int, int], Counter] = {}
    pending: Counter = Counter()
    pending_sub: Counter = Counter()
    # Sub-classification of lines already in `counts`; kept apart so no total sums a line twice.
    sub_counts: Counter = Counter()
    for line in text.splitlines():
        if "critical" in line:
            for name, pattern in EXPECTED.items():
                if pattern.search(line):
                    counts[name] += 1
                    pending[name] += 1
                    # Bank-allocation refusals split further into "no amount of free space would have
                    # satisfied this" and "depends what else is resident", and *which* matters per
                    # batch: round 16 found README bounding the in-forward conv1d risk with a
                    # per-batch claim this script only computed globally.
                    need = NEED.search(line)
                    if name == "conv-probe-l1-buffer" and need:
                        hard = int(need[1]) / int(need[2]) > BANK_B
                        sub_counts["hard overflow" if hard else "pressure-dependent"] += 1
                        # Sub-classification of a line already counted above, so it must NOT go into
                        # `pending`, which is summed as the batch's refusal total. Round 17 found it
                        # there, double-counting every bank refusal (batch 32 printed "31 refusals of
                        # 16") and feeding that inflated denominator into README §2.
                        pending_sub["hard" if hard else "soft"] += 1
                    break
            else:
                unclassified.append(line.strip())
        m = COVERAGE.search(line)
        if m and pending:
            key = (int(m["batch"]), int(m["total"]))
            per_batch.setdefault(key, Counter()).update(pending)
            per_batch_sub.setdefault(key, Counter()).update(pending_sub)
            pending = Counter()
            pending_sub = Counter()

    # Keyed by (batch, candidate count): the suite builds decoders with more than one prefill_chunk,
    # and a chunk of 128 has one candidate block length rather than 16. Keying on the batch alone
    # made this last-write-wins and therefore log-order dependent.
    coverage = {}
    for m in COVERAGE.finditer(text):
        key = (int(m["batch"]), int(m["total"]))
        got = int(m["n"])
        assert coverage.get(key, got) == got, f"conflicting coverage for {key}: {coverage[key]} vs {got}"
        coverage[key] = got

    out = [
        "critical-level lines in the passing suite log, classified",
        f"log: {LOG.name}",
        f"total critical lines: {sum(counts.values()) + len(unclassified)}"
        f"  (each classified into exactly one class below)",
        "",
    ]
    for name in EXPECTED:
        out.append(f"  {counts[name]:4d}  {name}")
    out.append(f"  {len(unclassified):4d}  UNCLASSIFIED")
    # Tautological against the current code (`counts` only ever takes EXPECTED keys) and kept as a
    # regression guard: round 17's bug was a sub-classification leaking into a counter that was then
    # summed as a total, and this is what that would look like if it happened to `counts`.
    if sum(counts.values()) != sum(counts[n] for n in EXPECTED):
        raise SystemExit("a non-class key leaked into `counts`; the printed total would double-count")
    out.append("")
    out.append(
        f"  of the {counts['conv-probe-l1-buffer']} bank refusals: "
        f"{sub_counts['pressure-dependent']} pressure-dependent, {sub_counts['hard overflow']} hard overflow"
    )
    out.append("")
    out.append("Why each candidate shape was refused, by allocated batch (criticals attributed to the")
    out.append("coverage line that follows them). The classes are not interchangeable: conv-probe-l1-buffer")
    out.append("is a sharded-tensor bank allocation refusal, conv-probe-cb-overflow a program-build")
    out.append("circular-buffer set exceeding L1.")
    for batch, total in sorted(per_batch):
        got = coverage.get((batch, total))
        detail = "  ".join(f"{name}={per_batch[(batch, total)][name]}" for name in EXPECTED)
        counted = sum(per_batch[(batch, total)].values())
        # Two closure checks. Round 17 found the refusal total silently double-counting the
        # bank-allocation class, printing "31 refusals of 16" - self-refuting on its own line, and
        # spliced into README §2 as the denominator of the dominant-class ratio.
        parts = per_batch[(batch, total)]
        named = parts["conv-probe-l1-buffer"] + parts["conv-probe-cb-overflow"]
        # Also tautological today, and also kept deliberately: this is the exact shape of round 17's
        # double-count, so it fires the moment anything else is pushed into `pending`.
        if counted != named:
            raise SystemExit(f"batch {batch}: {counted} refusals counted but {named} classified")
        accepted = coverage.get((batch, total))
        if accepted is not None and counted != total - accepted:
            raise SystemExit(f"batch {batch}: {counted} refusals but {total} candidates with {accepted} accepted")
        sub = per_batch_sub[(batch, total)]
        if sub["hard"] + sub["soft"] != parts["conv-probe-l1-buffer"]:
            raise SystemExit(f"batch {batch}: hard/soft split does not cover the bank refusals")
        out.append(f"  batch {batch:3d}: {counted:2d} refusals of {total}   {detail}")
    out.append("")
    # Whether a bank-allocation refusal is pressure-dependent at all. If the per-bank share of the
    # request exceeds an entirely empty bank, no amount of free space would have satisfied it — so it
    # is a hard capacity overflow, not a "depends what else is resident" refusal. Round 15 found README
    # §8 characterising the whole class as pressure-dependent.
    hard = sum(c["hard"] for c in per_batch_sub.values())
    soft = sum(c["soft"] for c in per_batch_sub.values())
    out.append(f"Of the {hard + soft} bank-allocation refusals, {soft} are pressure-dependent (the per-bank")
    out.append(f"share would fit an empty {BANK_B} B bank) and {hard} are hard overflows that no amount of free")
    out.append("space would satisfy. Only the pressure-dependent subset can behave differently between the")
    out.append("setup probe and a forward pass, so the split matters per batch, not just in total:")
    for batch, total in sorted(per_batch_sub):
        c = per_batch_sub[(batch, total)]
        if c["hard"] or c["soft"]:
            out.append(f"  batch {batch:3d}: pressure-dependent={c['soft']:2d}  hard-overflow={c['hard']:2d}")
    out.append("")
    out.append("ttnn.conv1d prefill-block coverage, by allocated batch (from the decoder's own log).")
    out.append("Grouped by how many candidate block lengths the decoder's prefill_chunk produces:")
    for total in sorted({t for _, t in coverage}, reverse=True):
        out.append(f"  prefill_chunk with {total} candidate block length(s):")
        for batch, cand in sorted(coverage):
            if cand == total:
                out.append(f"    batch {batch:3d}: {coverage[(batch, cand)]:2d}/{total} block lengths accepted")
    if unclassified:
        out.append("")
        out.append("UNCLASSIFIED lines:")
        out += [f"  {line}" for line in unclassified]
    body = "\n".join(out) + "\n"
    OUT.write_text(body)
    print(body, end="")
    return 1 if unclassified else 0


if __name__ == "__main__":
    sys.exit(main())
