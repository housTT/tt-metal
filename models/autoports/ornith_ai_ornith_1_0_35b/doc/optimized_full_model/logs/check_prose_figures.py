# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Assert that the figures in README.md and work_log.md are the ones in the artifacts they name.

Six rounds of stage review on this stage found the same failure mode every time: prose numbers that
were true of a run which a later run overwrote. This is the gate that closes it, and it is deliberately
strict in three ways the first version was not:

* a check names the **document** it applies to, so a figure that is right in one file and stale in the
  other cannot pass by appearing "somewhere";
* a multi-value row must match **every** value in it, not any one of them;
* the before-arm rows, the generated `sampler_cost_model.md` figures, the `tt-perf-report` op shares and
  the ladder winners are all covered, not just the optimized arm's headline.

    python .../doc/optimized_full_model/logs/check_prose_figures.py

Exits non-zero and prints every mismatch. Also checks that every referenced repo path exists, and that
`logs/sampler_cost_model.md` is current with respect to the perf summaries it reads.

It is deliberately **not** a step of `logs/run_evidence.sh`: it checks the documents against that sweep's
own output, so inside the sweep it would always fail on the run that produces new numbers. Run it after
refreshing the documents from a sweep, and after any edit to either of them. Its committed output is
`logs/check_prose_figures.txt`.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import re
import statistics
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[6]
#: Whitespace-normalised, because a literal that a document wraps across two lines is the same claim.
def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


DOCS = {name: _flat((DOC / name).read_text()) for name in ("README.md", "work_log.md")}
RAW = {name: (DOC / name).read_text() for name in ("README.md", "work_log.md")}


def load(path: Path):
    return json.loads(path.read_text())


AFTER = load(DOC / "perf_summary.json")
BEFORE = load(DOC / "perf_summary_before.json")
ARCHIVE = load(ROOT / "models/autoports/ornith_ai_ornith_1_0_35b/doc/full_model/perf_summary.json")
PREFILL = load(DOC / "prefill_profile.json")
LONG = load(DOC / "long_prompt.json")
FOOT = load(DOC / "footprint.json")
GATE = {n: load(DOC / f"readiness_{n}.json")["per_entry"][0] for n in ("prefill", "teacher", "prefill_bfp4head", "teacher_bfp4head")}
COST_MODEL = (DOC / "logs" / "sampler_cost_model.md").read_text()
#: Pulled out of the generated cost model rather than retyped.
TOPK_PER_GROUP = float(re.search(r"\*\*([\d.]+) us/replay per group\*\*", COST_MODEL).group(1))
TOPK_MACHINERY_AT_G = float(re.search(r"\*\*([\d.]+) us/replay\*\* of grouping", COST_MODEL).group(1))
WALL_DELTA_US = float(re.search(r"\*\*([-+\d.]+) us/token\*\* wall delta", COST_MODEL).group(1))

ttft = {k: sorted(r["ttft_s"] * 1e3 for r in v["runs"]) for k, v in (("after", AFTER), ("before", BEFORE))}
dec = {k: [r["decode_ms_per_token"] for r in v["runs"]] for k, v in (("after", AFTER), ("before", BEFORE))}


def stacked_shares(path: Path) -> dict[str, float]:
    """`{op code: share of the window in %}` from a `tt-perf-report --group-by op` stacked CSV."""
    rows = list(csv.DictReader(io.StringIO(gzip.decompress(path.read_bytes()).decode())))
    tkey = next(k for k in rows[0] if "Device Time" in k)
    ckey = next(k for k in rows[0] if "OP Code" in k or "Op Code" in k)
    totals: dict[str, float] = {}
    for row in rows:
        code = row[ckey].strip().split(" (")[0]
        totals[code] = totals.get(code, 0.0) + float(row[tkey].replace(",", ""))
    grand = sum(totals.values())
    return {k: v / grand * 100 for k, v in totals.items()}


SH_AFTER = stacked_shares(DOC / "tracy" / "decode_perf_report_stacked.csv.gz")
SH_BEFORE = stacked_shares(
    ROOT / "models/autoports/ornith_ai_ornith_1_0_35b/doc/full_model/tracy/decode_perf_report_stacked.csv.gz"
)


def ladder(name: str) -> dict[str, dict]:
    """`{arm: row}` from one of the two A/B ladders' raw output."""
    path = DOC / "logs" / name
    text = gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()
    return {r["arm"]: r for r in (json.loads(x[9:]) for x in text.splitlines() if x.startswith("ARM_JSON "))}


L1 = ladder("ab_terminal.txt.gz")
L2 = ladder("ab_terminal_kblock.txt")

#: `(document, template, values, decimals)`. The expected string is **formatted from the artifacts**,
#: so it cannot go stale relative to them: if a run moves a number, this file's expectation moves with
#: it and the *document* is what fails. That is the whole point - four review rounds were lost to
#: expectations that were themselves hand-copied.
CHECKS: list[tuple[str, str, list[float], int]] = [
    # ---- README section 1: the result, both arms ----
    ("README.md", "The {} t/s/u in\n`readiness_teacher.json`", [GATE["teacher"]["decode_t/s/u"]], 2),
    ("README.md", "| traced teacher-forcing decode (`run_teacher_forcing`) | 37.01 t/s/u (archive) | {} t/s/u |",
     [GATE["teacher"]["decode_t/s/u"]], 2),
    ("README.md", "| **token-out decode** | {} ms/token — **{} t/s/u** | **{} ms/token — {} t/s/u** |",
     [BEFORE["token_out_decode"]["ms_per_token"], BEFORE["token_out_decode"]["t/s/u"],
      AFTER["token_out_decode"]["ms_per_token"], AFTER["token_out_decode"]["t/s/u"]], (3, 2, 3, 2)),
    ("README.md", "| traced logits-only decode (model trace alone) | {} ms/token | **{} ms/token** |",
     [BEFORE["traced_logits_only_decode"]["ms_per_token"], AFTER["traced_logits_only_decode"]["ms_per_token"]], 3),
    ("README.md", "| model trace + sampling trace, no readback | {} ms/token | **{} ms/token** |",
     [BEFORE["traced_decode_plus_sampling_no_readback"]["ms_per_token"],
      AFTER["traced_decode_plus_sampling_no_readback"]["ms_per_token"]], 3),
    ("README.md", "| decode run-to-run spread over the nine repeats | {}–{} ({} %) | **{}–{} ({} %)** |",
     [min(dec["before"]), max(dec["before"]), (max(dec["before"]) - min(dec["before"])) / min(dec["before"]) * 100,
      min(dec["after"]), max(dec["after"]), (max(dec["after"]) - min(dec["after"])) / min(dec["after"]) * 100],
     (3, 3, 2, 3, 3, 3)),
    ("README.md", "| **warmed TTFT** | {} min / **{} median** / {} max | {} min / **{} median** / {} max |",
     [ttft["before"][0], statistics.median(ttft["before"]), ttft["before"][-1],
      ttft["after"][0], statistics.median(ttft["after"]), ttft["after"][-1]], 1),
    ("README.md", "(`serial_token_out_decode`, {} ms/token / {} t/s/u)",
     [AFTER["serial_token_out_decode"]["ms_per_token"], AFTER["serial_token_out_decode"]["t/s/u"]], (3, 2)),
    ("README.md", "the {} ms the pipelined loop wins", [AFTER["pipelined_readback_saving_ms"]], 3),
    ("README.md", "## 2. The host stall: {} ms/step that was not work at all", [AFTER["pipelined_readback_saving_ms"]], 3),
    # ---- the decode decomposition, both arms ----
    ("README.md", "| embedding + final norm + LM head + device `plus_one` | {} | **{}** |",
     [BEFORE["full_model_only_cost"]["logits_only_minus_lower_bound_ms"],
      AFTER["full_model_only_cost"]["logits_only_minus_lower_bound_ms"]], 3),
    ("README.md", "| sampling trace | {} | **{}** |",
     [BEFORE["full_model_only_cost"]["sampling_ms"], AFTER["full_model_only_cost"]["sampling_ms"]], 3),
    ("README.md", "| host synchronize + caller token readback | {} | **{}** |",
     [BEFORE["full_model_only_cost"]["sync_and_readback_ms"], AFTER["full_model_only_cost"]["sync_and_readback_ms"]], 3),
    ("README.md", "| **token-out total** | **{}** | **{}** |",
     [BEFORE["token_out_decode"]["ms_per_token"], AFTER["token_out_decode"]["ms_per_token"]], 3),
    ("README.md", "{} ms of terminal arithmetic plus {} ms of sampling",
     [AFTER["full_model_only_cost"]["logits_only_minus_lower_bound_ms"], AFTER["full_model_only_cost"]["sampling_ms"]], 3),
    # ---- section 6: both TTFT distributions, and the breakdown ----
    ("README.md", "| inherited | {} {} {} {} **{}** {} {} {} {} | {} | **{}** | {} |",
     [*ttft["before"], ttft["before"][0], statistics.median(ttft["before"]), ttft["before"][-1]], 1),
    ("README.md", "| optimized | {} {} {} {} **{}** {} {} {} {} | {} | **{}** | {} |",
     [*ttft["after"], ttft["after"][0], statistics.median(ttft["after"]), ttft["after"][-1]], 1),
    ("README.md", "| page-table row upload | {} | {} | {} |",
     [BEFORE["ttft_breakdown_ms"]["page_row_upload_ms"], AFTER["ttft_breakdown_ms"]["page_row_upload_ms"],
      AFTER["ttft_breakdown_ms"]["page_row_upload_ms"] - BEFORE["ttft_breakdown_ms"]["page_row_upload_ms"]],
     (3, 3, "signed3")),
    ("README.md", "last-row norm/LM head) | {} | {} | {} |",
     [BEFORE["ttft_breakdown_ms"]["prefill_ms"], AFTER["ttft_breakdown_ms"]["prefill_ms"],
      AFTER["ttft_breakdown_ms"]["prefill_ms"] - BEFORE["ttft_breakdown_ms"]["prefill_ms"]], (3, 3, "signed3")),
    ("README.md", "| **first-token sampling, untraced** | **{}** | **{}** | **{}** |",
     [BEFORE["ttft_breakdown_ms"]["first_token_sampling_ms"], AFTER["ttft_breakdown_ms"]["first_token_sampling_ms"],
      AFTER["ttft_breakdown_ms"]["first_token_sampling_ms"] - BEFORE["ttft_breakdown_ms"]["first_token_sampling_ms"]],
     (3, 3, "signed3")),
    ("README.md", "| total | {} | {} | **{}** |",
     [BEFORE["ttft_breakdown_ms"]["total_ms"], AFTER["ttft_breakdown_ms"]["total_ms"],
      AFTER["ttft_breakdown_ms"]["total_ms"] - BEFORE["ttft_breakdown_ms"]["total_ms"]], (3, 3, "signed3")),
    # ---- cold-length cost, both arms ----
    ("README.md", "| first request, wall clock | **{} s** |", [AFTER["cold_prompt_length_cost"]["first_request_wall_s"]], 3),
    ("README.md", "| of which reported as TTFT | {} ms |", [AFTER["cold_prompt_length_cost"]["first_request_ttft_ms"]], 0),
    ("README.md", "| **hidden from both published metrics** | **{} ms** |", [AFTER["cold_prompt_length_cost"]["hidden_cost_ms"]], 0),
    ("README.md", "at {} s / {} ms / {} ms",
     [BEFORE["cold_prompt_length_cost"]["first_request_wall_s"], BEFORE["cold_prompt_length_cost"]["first_request_ttft_ms"],
      BEFORE["cold_prompt_length_cost"]["hidden_cost_ms"]], (3, 0, 0)),
    # ---- performance accounting ----
    ("README.md", "| roofline estimate | **{} ms/token** ({} B per device per token / 512.3 GB/s) |",
     [AFTER["performance_accounting"]["roofline_ms_per_token_estimate"],
      AFTER["performance_accounting"]["roofline_bytes_per_token_per_device"]], (3, "comma0")),
    ("README.md", "| end-to-end decode | {} ms/token |", [AFTER["performance_accounting"]["decode_ms_per_token_e2e"]], 3),
    ("README.md", "| fraction of roofline achieved | **{} %** |",
     [AFTER["performance_accounting"]["roofline_fraction_achieved"] * 100], 1),
    # ---- prefill ladder and both fits ----
    ("README.md", "| warmed TTFT (ms) | {} | {} | {} | {} |", [PREFILL["WARNING"][k] for k in ("128", "256", "512", "1024")], 1),
    ("README.md", "| two-point secant through 128 and 1024 | {} ms/token | **{} ms** | **{} %** |",
     [PREFILL["fit"]["slope_ms_per_token"], PREFILL["fit"]["intercept_ms"], PREFILL["fit"]["intercept_share_at_128"] * 100],
     (3, 1, 1)),
    ("README.md", "| least squares over all four points | {} ms/token | **{} ms** | **{} %** |",
     [PREFILL["fit"]["least_squares"]["slope_ms_per_token"], PREFILL["fit"]["least_squares"]["intercept_ms"],
      PREFILL["fit"]["least_squares"]["intercept_share_at_128"] * 100], (3, 1, 1)),
    ("README.md", "The local 128→256 slope is {} ms/token.", [PREFILL["fit"]["local_slope_128_256_ms_per_token"]], 3),
    ("README.md", "reads **{} / {} / {} / {} ms** across", [PREFILL["debug_logging_cost_ms"][k] for k in ("128", "256", "512", "1024")],
     ("signed1", "signed1", "signed1", "signed1")),
    # ---- capability ----
    ("README.md", "| prefill | {} s | {} s | {} s | {} s | {} s | {} s | **{} s** |",
     [r["prefill_s"] for r in LONG["results"]], 2),
    ("README.md", "| tokens/s | {} | {} | {} | {} | {} | {} | **{}** |",
     [r["prefill_tokens_per_s"] for r in LONG["results"]], 0),
    ("README.md", "| **total resident** | **{} B ({} GiB)** of {} GiB allocatable |",
     [FOOT["per_device_bytes"]["total_resident"], FOOT["per_device_bytes"]["total_resident"] / 2**30,
      FOOT["per_device_bytes"]["allocatable_total"] / 2**30], ("comma0", 2, 2)),
    ("README.md", "| **free for activations** | **{} GiB** |", [FOOT["per_device_bytes"]["free_for_activations"] / 2**30], 2),
    # ---- section 10 op shares, before -> after ----
    ("README.md", "| `TopKDeviceOperation` (sampler stages 1+2) | {} % | **{} %** |",
     [SH_BEFORE["TopKDeviceOperation"], SH_AFTER["TopKDeviceOperation"]], 2),
    ("README.md", "| `SliceDeviceOperation` | {} % | **{} %** |", [SH_BEFORE["SliceDeviceOperation"], SH_AFTER["SliceDeviceOperation"]], 2),
    ("README.md", "| `ConcatDeviceOperation` | {} % | **{} %** |", [SH_BEFORE["ConcatDeviceOperation"], SH_AFTER["ConcatDeviceOperation"]], 2),
    ("README.md", "| `LayerNorm` | {} % | **{} %** |", [SH_BEFORE["LayerNormDeviceOperation"], SH_AFTER["LayerNormDeviceOperation"]], 2),
    ("README.md", "| `GatherDeviceOperation` (stage-2 index recovery) | {} % | {} % |",
     [SH_BEFORE["GatherDeviceOperation"], SH_AFTER["GatherDeviceOperation"]], 2),
    # ---- ladder winners ----
    ("README.md", "vocab align 32** | **{}** | **{}** | **{}** |",
     [L1["mcast1d-c110-nsh-align32"]["model_trace"], L1["mcast1d-c110-nsh-align32"]["sampling_trace"],
      L1["mcast1d-c110-nsh-align32"]["token_out_pipelined"]], 3),
    ("README.md", "| `dram_sharded` 64 cores, sharded norm | {} | {} | {} |",
     [L1["dram-sharded-c64"]["model_trace"], L1["dram-sharded-c64"]["sampling_trace"],
      L1["dram-sharded-c64"]["token_out_pipelined"]], 3),
    ("README.md", "| **8** | **8** | **{}** | **shipped** |", [L2["k8-n8-hifi2"]["model_trace"]], 3),
    ("README.md", "| LoFi | {} | **slower.**", [L2["k8-n8-lofi"]["model_trace"]], 3),
    ("README.md", "| HiFi4 | {} | tie", [L2["k8-n8-hifi4"]["model_trace"]], 3),
    # ---- work_log ----
    ("work_log.md", "**{} t/s/u** token-out\ndecode ({} ms/token)",
     [ARCHIVE["token_out_decode"]["t/s/u"], ARCHIVE["token_out_decode"]["ms_per_token"]], (2, 3)),
    ("work_log.md", "| host synchronize + caller token readback | {} | 1.9 % |", [ARCHIVE["full_model_only_cost"]["sync_and_readback_ms"]], 3),
    ("work_log.md", "reads {} ms/token and TTFT {} min /\n{} median ms",
     [BEFORE["token_out_decode"]["ms_per_token"], ttft["before"][0], statistics.median(ttft["before"])], (3, 2, 2)),
    ("work_log.md", "| pipelined | {} | {} |", [AFTER["token_out_decode"]["ms_per_token"], AFTER["token_out_decode"]["t/s/u"]], (3, 2)),
    ("work_log.md", "| serial (`synchronize_device` + readback per token) | {} | {} |",
     [AFTER["serial_token_out_decode"]["ms_per_token"], AFTER["serial_token_out_decode"]["t/s/u"]], (3, 2)),
    ("work_log.md", "**{} ms/token**, and the serial arm reproduces the inherited arm's {}",
     [AFTER["pipelined_readback_saving_ms"], BEFORE["token_out_decode"]["ms_per_token"]], 3),
    # ---- work_log's own copies of the two TTFT distributions (review 6 found these stale) ----
    ("work_log.md", "| inherited | {} {} {} {} **{}** {} {} {} {} | {} | **{}** | {} |",
     [*ttft["before"], ttft["before"][0], statistics.median(ttft["before"]), ttft["before"][-1]], 1),
    ("work_log.md", "| optimized | {} {} {} {} **{}** {} {} {} {} | {} | **{}** | {} |",
     [*ttft["after"], ttft["after"][0], statistics.median(ttft["after"]), ttft["after"][-1]], 1),
    ("work_log.md", "| **first-token sampling, untraced** | **{}** | **{}** | **{}** |",
     [BEFORE["ttft_breakdown_ms"]["first_token_sampling_ms"], AFTER["ttft_breakdown_ms"]["first_token_sampling_ms"],
      AFTER["ttft_breakdown_ms"]["first_token_sampling_ms"] - BEFORE["ttft_breakdown_ms"]["first_token_sampling_ms"]],
     (3, 3, "signed3")),
    # ---- README's limitations section, which had no coverage at all ----
    ("README.md", "(intercept {} ms) or {} % by\n   four-point least squares ({} ms)",
     [PREFILL["fit"]["intercept_ms"], PREFILL["fit"]["least_squares"]["intercept_share_at_128"] * 100,
      PREFILL["fit"]["least_squares"]["intercept_ms"]], (1, 1, 1)),
    ("README.md", "length-independent share is {} % by a two-point secant", [PREFILL["fit"]["intercept_share_at_128"] * 100], 1),
    ("README.md", "sampled untraced**, {} ms inside TTFT, of which {} ms is",
     [AFTER["ttft_breakdown_ms"]["first_token_sampling_ms"],
      AFTER["ttft_breakdown_ms"]["first_token_sampling_ms"] - BEFORE["ttft_breakdown_ms"]["first_token_sampling_ms"]], 3),
    ("README.md", "~{} µs/replay of pure op-launch overhead** ({} µs/replay\n   per group",
     [TOPK_MACHINERY_AT_G, TOPK_PER_GROUP], (0, 2)),
    # ---- the delta columns of section 1, which review 6 found stale ----
    ("README.md", "**{} ms/token** | {} ms |",
     [AFTER["traced_logits_only_decode"]["ms_per_token"],
      AFTER["traced_logits_only_decode"]["ms_per_token"] - BEFORE["traced_logits_only_decode"]["ms_per_token"]],
     (3, "signed3")),
    ("README.md", "reproducible to {} % across the nine optimized repeats — against\n{} % for the inherited arm",
     [(max(dec["after"]) - min(dec["after"])) / min(dec["after"]) * 100,
      (max(dec["before"]) - min(dec["before"])) / min(dec["before"]) * 100], (3, 2)),
    ("README.md", "to **{} % on `traced_logits_only_decode`**, **{} % on\n`traced_decode_plus_sampling_no_readback`** and **{} % on `token_out_decode`**",
     [abs(BEFORE["traced_logits_only_decode"]["ms_per_token"] - ARCHIVE["traced_logits_only_decode"]["ms_per_token"]) / ARCHIVE["traced_logits_only_decode"]["ms_per_token"] * 100,
      abs(BEFORE["traced_decode_plus_sampling_no_readback"]["ms_per_token"] - ARCHIVE["traced_decode_plus_sampling_no_readback"]["ms_per_token"]) / ARCHIVE["traced_decode_plus_sampling_no_readback"]["ms_per_token"] * 100,
      abs(BEFORE["token_out_decode"]["ms_per_token"] - ARCHIVE["token_out_decode"]["ms_per_token"]) / ARCHIVE["token_out_decode"]["ms_per_token"] * 100], 3),
    ("README.md", "varying by {} %", [(max(dec["before"]) - min(dec["before"])) / min(dec["before"]) * 100], 2),
    # ---- the sampling pair and the trade arithmetic ----
    ("README.md", "**{} → {} ms** on the delivered 40-layer model, same sweep",
     [BEFORE["full_model_only_cost"]["sampling_ms"], AFTER["full_model_only_cost"]["sampling_ms"]], 3),
    ("README.md", "(20 → 32 groups, {} ms/token)",
     [AFTER["full_model_only_cost"]["sampling_ms"] - BEFORE["full_model_only_cost"]["sampling_ms"]], "signed3"),
    ("README.md", "the {} ms × 127 = {} ms the pipelined loop saves",
     [AFTER["pipelined_readback_saving_ms"], AFTER["pipelined_readback_saving_ms"] * 127], (3, 0)),
    # ---- the estimator disclosure: min-of-nine headline against the medians ----
    ("README.md", "the medians read {} → {} ms/token, a {} ms delta against\nthe {} ms reported",
     [statistics.median(dec["before"]), statistics.median(dec["after"]),
      statistics.median(dec["before"]) - statistics.median(dec["after"]),
      BEFORE["token_out_decode"]["ms_per_token"] - AFTER["token_out_decode"]["ms_per_token"]], 3),
]

#: `(document, literal, condition, what it asserts)`.
LITERALS: list[tuple[str, str, bool, str]] = [
    ("README.md", "| `run_prefill_check` | 0.940", GATE["prefill"]["top1"] == 0.94, "prefill top1"),
    ("README.md", "| `run_teacher_forcing` (traced decode) | 0.970", GATE["teacher"]["top1"] == 0.97, "teacher top1"),
    ("README.md", "| `run_prefill_check` top-1 | **0.940** | 0.920 |", GATE["prefill_bfp4head"]["top1"] == 0.92, "bfp4 prefill"),
    ("README.md", "| `run_teacher_forcing` top-1 | **0.970** | 0.940 |", GATE["teacher_bfp4head"]["top1"] == 0.94, "bfp4 teacher"),
    ("README.md", "`decode_syncs: 0`", AFTER["steady_state_counters"]["decode_syncs"] == 0, "decode_syncs"),
    ("README.md", "`decode_calls: 127`", AFTER["steady_state_counters"]["decode_calls"] == 127, "decode_calls"),
    ("README.md", "`token_refreshes: 0`", AFTER["steady_state_counters"]["token_refreshes"] == 0, "token_refreshes"),
    ("README.md", "`position_refreshes: 1`", AFTER["steady_state_counters"]["position_refreshes"] == 1, "position_refreshes"),
    ("README.md", "`token_readbacks: 128` for 127 steps", AFTER["steady_state_counters"]["token_readbacks"] == 128, "token_readbacks"),
    ("README.md", "262144. No reduction.", AFTER["capability"]["max_context"] == 262144, "advertised context"),
    ("README.md", "1536 padded LM-head columns", AFTER["capability"]["padded_vocab_size"] - AFTER["capability"]["vocab_size"] == 1536, "vocab padding"),
    ("README.md", "`null` — *not measurable for the 40-layer stack*", AFTER["performance_accounting"]["decode_ms_per_token_device"] is None, "device time null"),
    ("README.md", "nine repeats", len(AFTER["runs"]) == 9 and len(BEFORE["runs"]) == 9, "repeat count"),
    ("README.md", "unpadded 62080", BEFORE["capability"]["padded_vocab_size"] // BEFORE["capability"]["tp"] == 62080, "inherited arm's shard width"),
    # The generated cost model's own figures must be the ones both documents quote.
    ("README.md", "**−85.1 µs/replay**", "-85.1 us/replay" in COST_MODEL, "cost model window delta"),
    ("README.md", "5.52 µs/replay per group", "5.52 us/replay per group" in COST_MODEL, "cost model machinery slope"),
    ("work_log.md", "**5.52 us/replay per group**", "5.52 us/replay per group" in COST_MODEL, "cost model machinery slope"),
    ("README.md", "0.188 × (W/g + 32g)", "0.188 * (W/g + 32g)" in COST_MODEL, "cost model reduction slope"),
    ("README.md", "`SparseMatmul` 859.70 → 870.68 µs", "859.70 | 870.68" in COST_MODEL, "unchanged-op tracking"),
]


def numbers(text: str) -> list[float]:
    return [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", "").replace("−", "-"))]


def main() -> int:
    bad: list[str] = []

    for doc, template, values, spec in CHECKS:
        specs = spec if isinstance(spec, tuple) else (spec,) * len(values)
        rendered = []
        for value, how in zip(values, specs):
            if how == "comma0":
                rendered.append(f"{value:,.0f}")
            elif isinstance(how, str) and how.startswith("signed"):
                rendered.append(f"{value:+.{int(how[6:])}f}".replace("-", "\u2212"))
            else:
                # The documents use U+2212 MINUS SIGN in prose; render negatives the same way.
                rendered.append(f"{value:.{how}f}".replace("-", "\u2212"))
        expected = template.format(*rendered)
        if _flat(expected) not in DOCS[doc]:
            bad.append(f"{doc}: the artifacts say {expected!r} - not found in the document")

    for doc, literal, ok, what in LITERALS:
        if _flat(literal) not in DOCS[doc]:
            bad.append(f"{doc}: MISSING literal ({what}) {literal!r}")
        elif not ok:
            bad.append(f"{doc}: STALE ({what}) - the artifact no longer supports {literal!r}")

    # The generated cost model must be current with respect to the perf summaries it reads, and both
    # documents must quote its wall delta rather than a remembered one.
    fresh = (AFTER["traced_decode_plus_sampling_no_readback"]["ms_per_token"]
             - BEFORE["traced_decode_plus_sampling_no_readback"]["ms_per_token"]) * 1e3
    if abs(WALL_DELTA_US - fresh) > 0.05:
        bad.append(
            f"logs/sampler_cost_model.md is stale: it says {WALL_DELTA_US:+.1f} us/token but the committed "
            f"perf summaries say {fresh:+.1f}. Re-run logs/make_sampler_cost_model.py"
        )
    for doc in DOCS:
        want = f"{WALL_DELTA_US:.1f}".replace("-", "\u2212")
        if "wall delta" in DOCS[doc] and want not in DOCS[doc]:
            bad.append(f"{doc}: quotes a wall delta other than the generated {want} us/token")

    # The inherited arm must actually be the pre-optimization path.
    if BEFORE["capability"]["padded_vocab_size"] != BEFORE["capability"]["vocab_size"]:
        bad.append("perf_summary_before.json: the inherited arm must not pad the vocabulary")
    if BEFORE["pipelined_readback_saving_ms"] is not None:
        bad.append("perf_summary_before.json: the inherited arm must report a null pipelining saving")

    # Every referenced repo-relative path must exist.
    both = "\n".join(RAW.values())
    for m in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", both):
        t = m.group(2)
        if not t.startswith(("http", "#")) and not (DOC / t).resolve().exists():
            bad.append(f"MISSING path: {t}")
    for m in re.finditer(r"`((?:logs|tracy|watcher|triage)/[A-Za-z0-9_./-]+)`", both):
        t = m.group(1)
        if not (DOC / t).exists() and not (DOC / (t + ".gz")).exists():
            bad.append(f"MISSING path: {t}")

    if bad:
        print(f"{len(bad)} problem(s):")
        for line in bad:
            print("  " + line)
        return 1
    print(f"ok: {len(CHECKS)} numeric rows + {len(LITERALS)} literals, both documents, every path resolves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
