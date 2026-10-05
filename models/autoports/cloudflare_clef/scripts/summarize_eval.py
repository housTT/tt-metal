"""Stage 5 evaluation summary for Cloudflare/clef on Tenstorrent.

Reads the outputs of run_eval.sh under --out-root (eval_remote.py rows per benchmark, the kev.benchmark report
directories), scores them (eval_metrics.score: accuracy, macro-F1 for BANKING77), compares the 100-item TT
sample rows against the CPU bf16 controls (parity_compare.summarize: max dp, mean dp, flips at margin 0.05,
accuracy delta), reads the Kev suites' clean metrics (accuracy, Brier, ECE, coverage, latency), and writes
/home/hous/dev/clef/reports/final_numbers.json and doc/benchmark/EVAL.md (model-card-format table with the TT
number next to the card number and the CPU 100-item control, the coverage, the latency and the wall time, and
the wall-time estimate made before the run). Without --write it only prints.

Usage (host):
  python summarize_eval.py [--out-root /home/hous/dev/clef/reports/eval] [--write] [--final-json PATH] [--doc PATH]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_metrics
import parity_compare
from clef_paths import read_jsonl

AUTOPORT = Path(__file__).resolve().parents[1]
CLEF = Path("/home/hous/dev/clef")
EVALS = CLEF / "evals"
REFERENCE = CLEF / "reports" / "reference"
EVAL_CPU = CLEF / "reports" / "eval_cpu"
OUT_ROOT = CLEF / "reports" / "eval"
FINAL_JSON = CLEF / "reports" / "final_numbers.json"
DOC = AUTOPORT / "doc" / "benchmark" / "EVAL.md"
NOTES = AUTOPORT / "doc" / "benchmark" / "EVAL_findings.md"
KEV_FINAL = Path("/home/hous/dev/kev/reports/final_numbers.json")
FLIP_MARGIN = 0.05
ECE_BINS = 10
GAP_FINDING_PP = 1.0
BENCHMARKS = [
    {
        "key": "arc_challenge",
        "title": "ARC-Challenge test",
        "metric": "accuracy",
        "card": 97.7,
        "full": "arc_challenge_test",
        "sample": "arc_challenge_test_sample100",
        "records": 1172,
    },
    {
        "key": "banking77",
        "title": "BANKING77 test",
        "metric": "macro_f1",
        "card": 94.2,
        "full": "banking77_test",
        "sample": "banking77_test_sample100",
        "records": 3080,
    },
    {
        "key": "newyorker_matching",
        "title": "New Yorker caption matching test (images)",
        "metric": "accuracy",
        "card": 69.5,
        "full": "newyorker_matching_test",
        "sample": "newyorker_matching_test_sample100",
        "records": 528,
    },
]
KEV_SUITES = [("hard-v1", 700), ("devtools-v1", 900), ("documents-v1", 574)]
PARITY_SETS = [
    ("reference_text", REFERENCE / "ref_text_bf16.jsonl", REFERENCE / "ref_text_fp32.jsonl"),
    ("reference_image", REFERENCE / "ref_image_bf16.jsonl", REFERENCE / "ref_image_fp32.jsonl"),
    ("dev64_text", REFERENCE / "dev64_text.ref_bf16.jsonl", None),
    ("dev16_image", REFERENCE / "dev16_image.ref_bf16.jsonl", None),
]
ESTIMATE = [
    (
        "ARC-Challenge test",
        1172,
        0.20,
        0.35,
        "167 to 274 tokens: bucket 256, stage 3 server 395 ms median on the 16 text records",
    ),
    (
        "BANKING77 test",
        3080,
        0.70,
        1.00,
        "1,824 to 1,871 tokens (77-option schema): 1024 + 512 + 256 buckets; stage 1 uncontended 1,774-token record 0.69 s device",
    ),
    (
        "New Yorker matching test",
        528,
        0.45,
        0.70,
        "307 to 791 tokens with one image: stage 3 server 530 ms median on the 8 image records",
    ),
    ("three 100-item samples", 300, 0.45, 0.70, "one third each of the three rows above"),
    (
        "Kev suites test (hard-v1, devtools-v1, documents-v1)",
        2174,
        0.25,
        0.55,
        "145 to 2,672 tokens: sweep200 development records of the same suites, stage 1 0.19 to 0.69 s device",
    ),
]
ESTIMATE_OVERHEAD_S = 300


def num(x):
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def pct(x, digits=1):
    return "" if x is None else f"{100.0 * x:.{digits}f}"


def latency_block(rows, summary_path=None):
    ok = [r for r in rows if "error" not in r]
    out = {"records": len(rows), "ok": len(ok), "failed": len(rows) - len(ok)}
    if ok:
        lat = sorted(float(r["latency_ms"]) for r in ok)
        wall = sorted(float(r["wall_ms"]) for r in ok)
        tokens = [int(r["input_tokens"]) for r in ok]
        out.update(
            latency_ms_median=statistics.median(lat),
            latency_ms_p95=lat[min(len(lat) - 1, int(0.95 * len(lat)))],
            latency_ms_max=max(lat),
            latency_ms_sum_s=round(sum(lat) / 1000, 1),
            wall_ms_median=statistics.median(wall),
            tokens_median=statistics.median(tokens),
            tokens_max=max(tokens),
        )
    if summary_path and Path(summary_path).exists():
        s = json.loads(Path(summary_path).read_text())
        entry = next(iter(s.get("files", {}).values()), {})
        out["run_wall_s"] = entry.get("run_wall_s")
        out["records_per_s"] = entry.get("records_per_s_this_run")
        out["concurrency"] = s.get("concurrency")
        out["served_backend"] = (s.get("served") or {}).get("backend")
        out["served_precision"] = (s.get("served") or {}).get("precision")
    return out


def score_file(rows, eval_path, metric="auto"):
    if not rows or not Path(eval_path).exists():
        return None
    s = eval_metrics.score(read_jsonl(eval_path), rows, metric, None)
    return {
        k: s[k]
        for k in (
            "metric",
            "eval_rows",
            "result_rows",
            "result_errors",
            "scored_questions",
            "correct",
            "accuracy",
            "macro_f1",
        )
        if k in s
    } | {"missing_results": len(s["missing_results"]), "unanswered_questions": len(s["unanswered_questions"])}


def parity_block(reference_path, rows, name):
    if not rows or not Path(reference_path).exists():
        return None
    summary = parity_compare.summarize(read_jsonl(reference_path), rows, FLIP_MARGIN, ECE_BINS, name)
    overall = summary["overall"]
    keys = (
        "questions",
        "max_dp",
        "mean_dp",
        "median_dp",
        "argmax_flips",
        "flips_at_margin",
        "near_tie_flips",
        "labelled_questions",
        "ref_accuracy",
        "cand_accuracy",
        "accuracy_delta_pp",
        "ref_brier",
        "cand_brier",
        "ref_ece",
        "cand_ece",
        "ece_shift",
    )
    block = {k: overall[k] for k in keys if k in overall}
    block["records_joined"] = summary["records"]["joined"]
    block["flips"] = summary["flips"]
    block["within_stage1_bars"] = block.get("flips_at_margin", 0) == 0 and (block.get("max_dp") or 0.0) <= 0.10
    return block, summary


def flip_text(block):
    flips = (block or {}).get("flips") or []
    if not flips:
        return ""
    parts = [
        f"`{f['id']}` `{f['question']}` {f['ref_argmax']} to {f['cand_argmax']} (reference margin {f['ref_margin']:.4f}, dp {f['dp']:.4f}, label {f.get('label')})"
        for f in flips
    ]
    return "; flips at margin: " + ", ".join(parts)


def max_dp_text(summary):
    rows = sorted(summary.get("per_question") or [], key=lambda r: -r["dp"])[:1]
    if not rows:
        return ""
    r = rows[0]
    return (
        f"; largest dp: `{r['id']}` `{r['question']}` dp {r['dp']:.4f} (reference {r['ref_argmax']}, TT {r['cand_argmax']}, "
        f"reference margin {r['ref_margin']:.4f}, label {r.get('label')})"
    )


def kev_block(out_root, suite):
    report = Path(out_root) / "kev" / suite / "test" / "report.json"
    if not report.exists():
        return None
    r = json.loads(report.read_text())
    clean = r.get("clean") or {}
    return {
        "n": clean.get("n"),
        "acc": clean.get("acc"),
        "brier": clean.get("brier"),
        "ece": clean.get("ece"),
        "nll": clean.get("nll"),
        "coverage": r.get("coverage"),
        "latency_ms": r.get("latency_ms"),
        "remote": r.get("remote"),
        "report": str(report),
    }


def kev_card_rows():
    if not KEV_FINAL.exists():
        return {}
    ev = json.loads(KEV_FINAL.read_text()).get("eval") or {}
    return {s: ev.get(f"{s}/test") for s, _ in KEV_SUITES}


def estimate_table():
    rows = []
    low = high = 0.0
    for name, n, lo, hi, basis in ESTIMATE:
        rows.append(
            {
                "suite": name,
                "records": n,
                "s_per_request_low": lo,
                "s_per_request_high": hi,
                "low_s": n * lo,
                "high_s": n * hi,
                "basis": basis,
            }
        )
        low += n * lo
        high += n * hi
    return {
        "rows": rows,
        "total_low_s": low + ESTIMATE_OVERHEAD_S,
        "total_high_s": high + ESTIMATE_OVERHEAD_S,
        "overhead_s": ESTIMATE_OVERHEAD_S,
        "note": (
            "one TP=2 worker serves requests in series; client concurrency hides only the HTTP and encode time, "
            "so the wall time is the sum of the per-request model times plus the fp32 head on the host. "
            "Per-request figures are the stage 1 eager device times and the stage 3 traced server latencies "
            "(doc/optimized/perf_summary.json, logs/stage3_parity_remote.log)."
        ),
    }


def collect(out_root):
    out_root = Path(out_root)
    final = {
        "generated_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "out_root": str(out_root),
        "benchmarks": {},
        "parity": {},
        "kev": {},
        "kev_card_tt_p150x1": kev_card_rows(),
        "estimate": estimate_table(),
        "coverage_ok": True,
        "coverage_issues": [],
        "findings": [],
        "notes": [],
    }
    parity_md = {}
    for b in BENCHMARKS:
        entry = {"title": b["title"], "metric": b["metric"], "card": b["card"], "records": b["records"]}
        full_rows = read_jsonl(out_root / f"{b['full']}.jsonl") if (out_root / f"{b['full']}.jsonl").exists() else []
        entry["full"] = score_file(full_rows, EVALS / f"{b['full']}.jsonl", b["metric"])
        entry["full_latency"] = latency_block(full_rows, out_root / f"{b['full']}.summary.json") if full_rows else None
        sample_rows = (
            read_jsonl(out_root / f"{b['sample']}.jsonl") if (out_root / f"{b['sample']}.jsonl").exists() else []
        )
        entry["sample_tt"] = score_file(sample_rows, EVALS / f"{b['sample']}.jsonl", b["metric"])
        entry["sample_latency"] = (
            latency_block(sample_rows, out_root / f"{b['sample']}.summary.json") if sample_rows else None
        )
        cpu_rows_path = EVALS / f"{b['sample']}.ref_bf16.jsonl"
        entry["sample_cpu"] = (
            score_file(read_jsonl(cpu_rows_path), EVALS / f"{b['sample']}.jsonl", b["metric"])
            if cpu_rows_path.exists()
            else None
        )
        entry["sample_cpu_file"] = str(EVAL_CPU / f"{b['key']}_sample100_cpu_bf16.json")
        if sample_rows and cpu_rows_path.exists():
            block, summary = parity_block(cpu_rows_path, sample_rows, f"TT {b['sample']} vs CPU bf16")
            entry["sample_parity"] = block
            parity_md[b["sample"]] = summary
        for part in ("full", "sample_tt"):
            s = entry.get(part)
            if s and (s.get("result_errors") or s.get("missing_results") or s.get("unanswered_questions")):
                final["coverage_ok"] = False
                final["coverage_issues"].append(
                    f"{b['key']} {part}: {s.get('result_errors')} error rows, {s.get('missing_results')} missing, "
                    f"{s.get('unanswered_questions')} unanswered"
                )
        if entry["sample_tt"] and entry["sample_cpu"]:
            m = b["metric"]
            gap = 100.0 * (entry["sample_tt"][m] - entry["sample_cpu"][m])
            entry["sample_gap_pp"] = round(gap, 2)
            if abs(entry["sample_gap_pp"]) > GAP_FINDING_PP:
                final["findings"].append(
                    f"{b['title']}: TT sample {m} differs from the CPU control by {gap:+.2f} pp (finding above {GAP_FINDING_PP} pp)"
                    + flip_text(entry.get("sample_parity"))
                )
            elif abs(entry["sample_gap_pp"]) == GAP_FINDING_PP:
                final["notes"].append(
                    f"{b['title']}: TT sample {m} differs from the CPU control by exactly {gap:+.2f} pp, which is not above "
                    f"the {GAP_FINDING_PP} pp finding threshold" + flip_text(entry.get("sample_parity"))
                )
        final["benchmarks"][b["key"]] = entry
    for name, ref_bf16, ref_fp32 in PARITY_SETS:
        path = out_root / f"{name}.jsonl"
        if not path.exists():
            continue
        rows = read_jsonl(path)
        entry = {"rows": len(rows), "latency": latency_block(rows, out_root / f"{name}.summary.json")}
        block, summary = parity_block(ref_bf16, rows, f"TT server {name} vs CPU bf16")
        entry["vs_bf16"] = block
        parity_md[f"{name}_vs_bf16"] = summary
        if ref_fp32 and ref_fp32.exists():
            block32, summary32 = parity_block(ref_fp32, rows, f"TT server {name} vs CPU fp32")
            entry["vs_fp32"] = block32
            parity_md[f"{name}_vs_fp32"] = summary32
        if not block["within_stage1_bars"]:
            final["findings"].append(
                f"parity {name}: outside the stage 1 bars (max dp {block['max_dp']:.4f}, flips at margin {block['flips_at_margin']})"
                + flip_text(block)
                + max_dp_text(summary)
            )
        final["parity"][name] = entry
    for suite, n_records in KEV_SUITES:
        block = kev_block(out_root, suite)
        if block:
            cov = block.get("coverage") or {}
            if cov.get("rejected_records") or cov.get("truncated_records") or cov.get("evaluated_records") != n_records:
                final["coverage_ok"] = False
                final["coverage_issues"].append(
                    f"kev {suite}: {cov.get('evaluated_records')} of {n_records} test records evaluated, "
                    f"{cov.get('rejected_records')} rejected, {cov.get('truncated_records')} truncated"
                )
        final["kev"][suite] = block
    return final, parity_md


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(lines)


def fmt_s(seconds):
    if seconds is None:
        return ""
    return f"{seconds / 60:.0f} min" if seconds >= 120 else f"{seconds:.0f} s"


def render(final):
    L = []
    L.append("# Clef on Tenstorrent p150x2: evaluation (stage 5)")
    L.append("")
    L.append(
        f"Generated {final['generated_utc']} UTC by "
        "`/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/scripts/summarize_eval.py` from the outputs of "
        f"`scripts/run_eval.sh` under `{final['out_root']}`. Machine-readable copy: `/home/hous/dev/clef/reports/final_numbers.json`. "
        "The serving performance table of stage 3 is in `README.md` next to this file. Times in this file are UTC (the host clock); ET is UTC-4."
    )
    L.append("")
    L.append(
        "Acronyms: TT (Tenstorrent), TP (tensor parallel), CPU (host processor), dp (absolute difference of an option "
        "probability against the CPU reference), pp (percentage points), ECE (expected calibration error), p50 and p95 "
        "(percentiles), macro-F1 (unweighted mean of the per-class F1 scores)."
    )
    L.append("")
    L.append("## Method")
    L.append("")
    L.append(
        "1. Public benchmarks from Clef's model card, rendered once to SystemOne requests by `scripts/render_public_evals.py` "
        "(`/home/hous/dev/clef/evals/*.jsonl`, revisions in `/home/hous/dev/clef/evals/MANIFEST.json`; no prompt tuning, no option "
        "dropping), sent to the TT server (`POST /v1/systemone`) by `scripts/eval_remote.py` and scored by `scripts/eval_metrics.py` "
        "(accuracy; macro-F1 over the 77 intents for BANKING77). The card numbers come from the non-public Decision Index requests, "
        "so they are indicative only; the control that separates the rendering from the device is the CPU bf16 run of the author's "
        "own code on a stratified 100-item sample of each benchmark (`/home/hous/dev/clef/evals/*_sample100.ref_bf16.jsonl`, "
        "`/home/hous/dev/clef/reports/eval_cpu/`). The TT server runs the same 100 items; `scripts/parity_compare.py` reports max dp, "
        "mean dp and argmax flips at reference margin >= 0.05 between the two, and a TT-versus-CPU gap above 1.0 pp on any benchmark is a finding."
    )
    L.append(
        "2. Kev suites (`/home/hous/dev/kev/kev/evals/{hard-v1,devtools-v1,documents-v1}/test.jsonl`) through `kev.benchmark --remote` "
        "unchanged (`clean` block: accuracy, Brier, ECE over the clean knowable rows), which gives a second labelled corpus and the "
        "Clef-versus-Kev-on-the-same-silicon row."
    )
    L.append(
        "3. Parity against the author's reference on the stage 1 and 2 sets over HTTP (`run_eval.sh --steps parity`): the 16 text "
        "and 8 image reference records, dev64 text and dev16 image, against the CPU bf16 rows (and fp32 where it exists)."
    )
    L.append("")
    L.append("## Model-card table")
    L.append("")
    rows = []
    for key, b in final["benchmarks"].items():
        m = b["metric"]
        full = b.get("full") or {}
        tt_s = b.get("sample_tt") or {}
        cpu_s = b.get("sample_cpu") or {}
        par = b.get("sample_parity") or {}
        lat = b.get("full_latency") or {}
        rows.append(
            [
                b["title"],
                "macro-F1" if m == "macro_f1" else "accuracy",
                pct(full.get(m)) if full else "not run",
                f"{full.get('scored_questions', '')}/{b['records']}" if full else "",
                pct(tt_s.get(m)) if tt_s else "not run",
                pct(cpu_s.get(m)) if cpu_s else "",
                "" if b.get("sample_gap_pp") is None else f"{b['sample_gap_pp']:+.2f}",
                b["card"],
                (
                    ""
                    if not par
                    else f"{par['max_dp']:.4f} / {par['mean_dp']:.4f} / {par['flips_at_margin']} ({par['near_tie_flips']} near-tie)"
                ),
                "" if not lat else f"{lat.get('latency_ms_median', 0):.0f} / {lat.get('latency_ms_p95', 0):.0f}",
                "" if not lat or lat.get("run_wall_s") is None else fmt_s(lat["run_wall_s"]),
            ]
        )
    L.append(
        md_table(
            [
                "Benchmark",
                "Metric",
                "TT, full test set",
                "scored / records",
                "TT, 100-item sample",
                "CPU bf16, same 100 items",
                "TT minus CPU (pp)",
                "Model card",
                "TT vs CPU on the sample: max dp / mean dp / flips at margin 0.05",
                "TT latency_ms p50 / p95 (full set)",
                "Wall time (full set)",
            ],
            rows,
        )
    )
    L.append("")
    L.append(
        "Card values: ARC-Challenge 97.7, BANKING77 94.2, New Yorker 69.5 (Clef model card, non-public requests). The 100-item samples "
        "carry a binomial 95 percent interval of about plus or minus 4 points on ARC and plus or minus 10 on New Yorker, so the sample "
        "columns compare the TT server to the CPU control, not to the card. `latency_ms` is the server's model time per request "
        "(prefix-cache miss or hit, schema continuation, head on the host); the wall time is the client run at the concurrency of the "
        "summary file and includes the HTTP and encode time."
    )
    L.append("")
    L.append("## Kev suites (test splits, clean knowable rows)")
    L.append("")
    rows = []
    for suite, n_records in KEV_SUITES:
        k = final["kev"].get(suite)
        kev = (final.get("kev_card_tt_p150x1") or {}).get(suite) or {}
        cov = (k or {}).get("coverage") or {}
        lat = (k or {}).get("latency_ms") or {}
        rows.append(
            [
                suite,
                n_records,
                "not run" if not k else k.get("n"),
                "" if not k or k.get("acc") is None else f"{k['acc']:.3f}",
                "" if not k or k.get("brier") is None else f"{k['brier']:.3f}",
                "" if not k or k.get("ece") is None else f"{k['ece']:.3f}",
                (
                    ""
                    if not k
                    else f"{cov.get('evaluated_records', '')} evaluated, {cov.get('rejected_records', '')} rejected, {cov.get('truncated_records', '')} truncated"
                ),
                "" if not lat else f"{lat.get('median', 0):.0f} / {lat.get('p95', 0):.0f}",
                (
                    ""
                    if not kev
                    else f"{kev.get('acc', 0):.3f} / {kev.get('ece', 0):.3f}"
                    + (f" (n={kev['n']:,})" if k and kev.get("n") not in (None, k.get("n")) else "")
                ),
            ]
        )
    L.append(
        md_table(
            [
                "Suite / test",
                "records",
                "questions (clean knowable)",
                "Clef TT acc",
                "Brier",
                "ECE",
                "coverage",
                "client latency p50 / p95 ms",
                "Kev-9B on one P150 (stage 6): acc / ECE",
            ],
            rows,
        )
    )
    L.append("")
    L.append(
        "The Kev column is `jaredpalmer/kev-9b` served by its own TT engine on this box "
        "(`/home/hous/dev/kev/reports/final_numbers.json`, `eval` block, test splits). Where the Kev cell carries an `n`, Kev's clean "
        "question count differs from Clef's: Kev's devtools-v1 test number is over n=1,071 questions after two audited drops "
        "(`drop_ids` in that file: `codereviewer/cls-test/13657` and `19245`), Clef's over all 1,073; the difference moves accuracy by at most 0.2 pp. "
        "Clef and Kev are different models with "
        "different training data; the row shows two decision models on the same silicon and the same labelled requests, not a ranking of the ports."
    )
    L.append("")
    if final["parity"]:
        L.append("## Parity against the author's reference over HTTP")
        L.append("")
        rows = []
        for name, p in final["parity"].items():
            for ref in ("vs_bf16", "vs_fp32"):
                b = p.get(ref)
                if not b:
                    continue
                rows.append(
                    [
                        name,
                        ref[3:],
                        b["questions"],
                        f"{b['max_dp']:.4f}",
                        f"{b['mean_dp']:.4f}",
                        b["argmax_flips"],
                        b["flips_at_margin"],
                        b["near_tie_flips"],
                        "" if b.get("accuracy_delta_pp") is None else f"{b['accuracy_delta_pp']:+.2f}",
                        "" if b.get("ece_shift") is None else f"{b['ece_shift']:+.4f}",
                        "met" if b["within_stage1_bars"] else "not met",
                    ]
                )
        L.append(
            md_table(
                [
                    "Set",
                    "CPU reference",
                    "questions",
                    "max dp",
                    "mean dp",
                    "argmax flips",
                    "flips at margin 0.05",
                    "near-tie flips",
                    "accuracy delta pp",
                    "ECE shift",
                    "stage 1 bars (max dp <= 0.10, 0 flips at margin)",
                ],
                rows,
            )
        )
        L.append("")
    L.append("## Coverage and findings")
    L.append("")
    L.append(
        f"- Coverage complete (0 rejected, 0 truncated, 0 missing, 0 unanswered on every file): {'yes' if final['coverage_ok'] else 'NO'}."
    )
    if final["findings"]:
        for f in final["findings"]:
            L.append(f"- Finding: {f}")
    else:
        L.append(
            "- No TT-versus-CPU sample gap above 1.0 pp and no parity set outside the stage 1 bars (on the files present)."
        )
    for n in final.get("notes") or []:
        L.append(f"- Note: {n}")
    L.append("")
    if NOTES.exists():
        L.append(NOTES.read_text().rstrip())
        L.append("")
    L.append("## Wall-time estimate (made before the run, from the stage 1 and 3 latencies)")
    L.append("")
    est = final["estimate"]
    rows = [
        [
            r["suite"],
            r["records"],
            f"{r['s_per_request_low']:.2f} to {r['s_per_request_high']:.2f}",
            f"{fmt_s(r['low_s'])} to {fmt_s(r['high_s'])}",
            r["basis"],
        ]
        for r in est["rows"]
    ]
    rows.append(
        [
            "server start and warmup",
            "",
            "",
            fmt_s(est["overhead_s"]),
            "stage 3 server: engine load 103 to 126 s, traces, warmup request",
        ]
    )
    rows.append(
        [
            "total",
            sum(r["records"] for r in est["rows"]),
            "",
            f"{fmt_s(est['total_low_s'])} to {fmt_s(est['total_high_s'])}",
            est["note"],
        ]
    )
    L.append(md_table(["Step", "records", "s per request", "wall", "basis"], rows))
    L.append("")
    measured = [
        b.get("full_latency", {}).get("run_wall_s") for b in final["benchmarks"].values() if b.get("full_latency")
    ]
    measured += [
        b.get("sample_latency", {}).get("run_wall_s") for b in final["benchmarks"].values() if b.get("sample_latency")
    ]
    measured = [m for m in measured if m is not None]
    if measured:
        L.append(
            f"Measured client wall time of the eval_remote.py files present: {fmt_s(sum(measured))} in total ({len(measured)} files; the Kev suites are not included, see their report.json latency blocks)."
        )
        L.append("")
    L.append("## Commands")
    L.append("")
    L.append("```")
    L.append("cd /home/hous/dev/clef/tt-metal")
    L.append(
        "SNAP=/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c"
    )
    L.append(
        "nohup /home/hous/dev/clef/bin/devrun timeout 18000 env OMP_NUM_THREADS=8 CLEF_MODEL=$SNAP HF_MODEL=$SNAP HF_HUB_OFFLINE=1 "
        "CLEF_MESH_SHAPE=1x2 MESH_DEVICE=P150x2 CLEF_TRACED=0 CLEF_PLANNER=1 python -m uvicorn models.autoports.cloudflare_clef.tt.server:app "
        "--host 127.0.0.1 --port 8008 --lifespan on > /home/hous/dev/clef/logs/stage5_server.log 2>&1 &"
    )
    L.append(
        'bash models/autoports/cloudflare_clef/scripts/run_eval.sh --base-url http://127.0.0.1:8008 --concurrency 4 --steps "arc banking77 newyorker samples parity kev summarize"'
    )
    L.append(
        "/home/hous/dev/clef/bin/hostrun python models/autoports/cloudflare_clef/scripts/summarize_eval.py --write"
    )
    L.append("```")
    L.append("")
    L.append(
        "`run_eval.sh` writes one log per step under `/home/hous/dev/clef/logs/stage5_*.log`, the TT rows and `*.summary.json` under the out root, "
        "the Kev reports under `<out root>/kev/<suite>/test/` (`report.json`, `rows.json`, `predictions.jsonl`), and the metrics under `<out root>/metrics/`. "
        "`eval_remote.py` resumes a file (ids already answered are skipped), so an interrupted step is rerun with the same command."
    )
    L.append("")
    return "\n".join(L)


def print_summary(final):
    print(f"out_root {final['out_root']}")
    for key, b in final["benchmarks"].items():
        m = b["metric"]
        full = b.get("full")
        tt_s = b.get("sample_tt")
        cpu_s = b.get("sample_cpu")
        par = b.get("sample_parity")
        line = f"{b['title']:44} {m:9}"
        line += f" full {pct(full[m]) if full else 'not run':>7}"
        line += f" ({full['scored_questions']}/{b['records']})" if full else ""
        line += f"  sample TT {pct(tt_s[m]) if tt_s else 'not run':>7}  CPU {pct(cpu_s[m]) if cpu_s else '':>5}"
        line += f"  gap {b.get('sample_gap_pp'):+.2f} pp" if b.get("sample_gap_pp") is not None else ""
        line += f"  card {b['card']}"
        if par:
            line += f"  parity max dp {par['max_dp']:.4f} mean {par['mean_dp']:.4f} flips@margin {par['flips_at_margin']} near-tie {par['near_tie_flips']}"
        lat = b.get("full_latency") or b.get("sample_latency")
        if lat and lat.get("latency_ms_median") is not None:
            line += f"  latency p50 {lat['latency_ms_median']:.0f} ms p95 {lat['latency_ms_p95']:.0f} ms wall {fmt_s(lat.get('run_wall_s'))}"
        print(line)
    for name, p in final["parity"].items():
        for ref in ("vs_bf16", "vs_fp32"):
            b = p.get(ref)
            if b:
                print(
                    f"parity {name} {ref}: n={b['questions']} max dp {b['max_dp']:.4f} mean {b['mean_dp']:.4f} flips {b['argmax_flips']} at margin {b['flips_at_margin']} bars {'met' if b['within_stage1_bars'] else 'NOT met'}"
                )
    for suite, _ in KEV_SUITES:
        k = final["kev"].get(suite)
        if not k:
            print(f"kev {suite}/test: not run")
            continue
        cov = k.get("coverage") or {}
        lat = k.get("latency_ms") or {}
        print(
            f"kev {suite}/test: n={k['n']} acc {k['acc']:.3f} brier {k['brier']:.3f} ece {k['ece']:.3f} "
            f"coverage {cov.get('evaluated_records')}/{cov.get('requested_records')} rejected {cov.get('rejected_records')} "
            f"latency p50 {lat.get('median', 0):.0f} ms p95 {lat.get('p95', 0):.0f} ms"
        )
    est = final["estimate"]
    print(
        f"coverage ok: {final['coverage_ok']} {final.get('coverage_issues') or ''}; findings: {final['findings'] or 'none'}; "
        f"notes: {final.get('notes') or 'none'}"
    )
    print(
        f"wall-time estimate: {fmt_s(est['total_low_s'])} to {fmt_s(est['total_high_s'])} for {sum(r['records'] for r in est['rows'])} requests"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    parser.add_argument("--final-json", default=str(FINAL_JSON))
    parser.add_argument("--doc", default=str(DOC))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    final, parity_md = collect(args.out_root)
    print_summary(final)
    if not args.write:
        return
    metrics_dir = Path(args.out_root) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for name, summary in parity_md.items():
        (metrics_dir / f"parity_{name}.json").write_text(json.dumps(summary, indent=1))
        (metrics_dir / f"parity_{name}.md").write_text(parity_compare.markdown(summary))
    Path(args.final_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.final_json).write_text(json.dumps(final, indent=1) + "\n")
    Path(args.doc).parent.mkdir(parents=True, exist_ok=True)
    Path(args.doc).write_text(render(final))
    print(f"wrote {args.final_json}, {args.doc}, {metrics_dir}")


if __name__ == "__main__":
    main()
