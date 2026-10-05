"""Stage 4 sweep summary for Cloudflare/clef: Pareto table, selection rule, selected_precision_config.json.

Reads the per-variant JSON files written by dtype_sweep.py (one per variant under --sweep-root), applies the
selection rule of doc/datatype_sweep/README.md, prints the table and, with --write, writes into
doc/datatype_sweep/: sweep_results.json, sweep_results.csv, pareto_table.md, selected_precision_config.json,
top1_perf_pareto.png and agreement_perf_pareto.png. tt/precision_defaults.py reads the runtime_flags of
selected_precision_config.json, so the selected knobs become the engine default as soon as the file exists.

Selection rule (README, written before any run): among the complete text variants (ref16 and sweep200 present,
status ok), keep those with 0 argmax flips at reference margin >= 0.05 on the 16 reference records, sweep200
accuracy within 1.0 pp of baseline_stage1, and sweep200 mean dp within 0.005 of the best mean dp; take the
fastest by the warm median device time per sweep200 record; ties within 2 percent go to the fewer qwen36
changes, then the lower sweep200 mean dp, then the lower ref16 max dp.

Usage (host):
  python dtype_sweep_summary.py [--sweep-root /home/hous/dev/clef/reports/sweep] [--write] [--doc-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dtype_sweep import CSV_FIELDS, DEFAULT_RUN_ORDER, KNOB_SHORT, REF, STAGE1, VARIANTS, csv_row

AUTOPORT = Path(__file__).resolve().parents[1]
DOC = AUTOPORT / "doc" / "datatype_sweep"
SWEEP_ROOT = Path("/home/hous/dev/clef/reports/sweep")
BASELINE = "baseline_stage1"
FLIP_MARGIN = 0.05
ACC_WINDOW_PP = 1.0
MEAN_DP_WINDOW = 0.005
TIME_TIE_PCT = 2.0
REF16_MAX_DP_BAR = 0.10
MAX_STATE_DEFAULT = 16384
MAX_TAIL = 4096
BLOCK = 64
KV_RESERVE_BYTES = 2 << 30
RULE_TEXT = (
    "Eligible: complete text variants (status ok, ref16 and sweep200 rows present) with 0 argmax flips at "
    "reference top-2 margin >= 0.05 on the 16 reference records (CPU bf16 reference), sweep200 accuracy within "
    "1.0 pp of baseline_stage1, and sweep200 mean dp within 0.005 of the best mean dp among the complete "
    "variants. Selected: the fastest eligible variant by the warm median device seconds per sweep200 record "
    "(eager, full path, bucket sequence seen before in the process). Ties within 2 percent of that time go to "
    "the fewer qwen36 changes, then the lower sweep200 mean dp, then the lower ref16 max dp. Reported, not "
    "gated: ref16 max dp against the 0.10 bar, near-tie flips, the fp32 comparison, the image sets "
    "(ref_image8, dev16_image, the sensitive record 7b7ef383). vision_upstream is an image-only control "
    "and never a selection candidate; the vision knobs of the selected config are the stage 2 values."
)


def num(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def load_variants(root):
    out = {}
    for path in sorted(Path(root).glob("*.json")):
        name = path.stem
        if name not in VARIANTS:
            continue
        data = json.loads(path.read_text())
        row = csv_row(name, data.get("knobs", STAGE1), data)
        row["data"] = data
        out[name] = row
    return out


def complete_text(r):
    sets = r["data"].get("sets", {})
    return r["status"] == "ok" and "ref16" in sets and "sweep200" in sets and num(r["sweep200_acc"]) is not None


def speed(r):
    return num(r["sweep200_warm_median_device_s"]) or num(r["sweep200_median_device_s"])


def select(rows):
    text = [r for r in rows.values() if complete_text(r) and r["variant"] != "vision_upstream"]
    base = rows.get(BASELINE)
    report = {
        "rule": RULE_TEXT,
        "baseline": BASELINE,
        "baseline_complete": base is not None and complete_text(base),
        "complete_text_variants": [r["variant"] for r in text],
        "checks": {},
    }
    if not text or not report["baseline_complete"]:
        report["selected"] = None
        report["reason"] = "baseline_stage1 has no complete text run yet"
        return report
    base_acc = num(base["sweep200_acc"])
    best_mean_dp = min(num(r["sweep200_mean_dp"]) for r in text)
    report["baseline_sweep200_acc"] = base_acc
    report["best_sweep200_mean_dp"] = best_mean_dp
    eligible = []
    for r in text:
        checks = {
            "ref16_flips_at_margin_zero": int(num(r["ref16_flips_margin"]) or 0) == 0,
            "sweep200_acc_within_1pp_of_baseline": num(r["sweep200_acc"]) >= base_acc - ACC_WINDOW_PP / 100,
            "sweep200_mean_dp_within_0_005_of_best": num(r["sweep200_mean_dp"]) <= best_mean_dp + MEAN_DP_WINDOW,
            "ref16_max_dp_le_0_10_reported": (num(r["ref16_max_dp"]) or 0) <= REF16_MAX_DP_BAR,
            "speed_s": speed(r),
        }
        checks["eligible"] = all(
            checks[k]
            for k in (
                "ref16_flips_at_margin_zero",
                "sweep200_acc_within_1pp_of_baseline",
                "sweep200_mean_dp_within_0_005_of_best",
            )
        )
        report["checks"][r["variant"]] = checks
        if checks["eligible"]:
            eligible.append(r)
    report["eligible"] = [r["variant"] for r in eligible]
    if not eligible:
        report["selected"] = None
        report["reason"] = "no variant passes the three eligibility checks"
        return report
    fastest = min(speed(r) for r in eligible)
    ties = [r for r in eligible if speed(r) <= fastest * (1 + TIME_TIE_PCT / 100)]
    chosen = min(
        ties,
        key=lambda r: (int(r["qwen36_changes"]), num(r["sweep200_mean_dp"]), num(r["ref16_max_dp"]) or 0, speed(r)),
    )
    report["fastest_s"] = fastest
    report["ties"] = [r["variant"] for r in ties]
    report["selected"] = chosen["variant"]
    report["reason"] = (
        f"fastest eligible {fastest:.4f} s per sweep200 record; ties within {TIME_TIE_PCT:.0f} percent: "
        f"{', '.join(report['ties'])}; tie break by qwen36 changes, sweep200 mean dp, ref16 max dp"
    )
    return report


def pareto_front(rows, xkey):
    front = set()
    done = [r for r in rows if num(r[xkey]) is not None and speed(r) is not None]
    for r in done:
        dominated = any(
            speed(o) <= speed(r)
            and num(o[xkey]) >= num(r[xkey])
            and (speed(o) < speed(r) * (1 - TIME_TIE_PCT / 100) or num(o[xkey]) > num(r[xkey]))
            for o in done
            if o is not r
        )
        if not dominated:
            front.add(r["variant"])
    return front


TABLE_COLS = [
    ("variant", "variant"),
    ("status", "status"),
    ("gate_up", "gate/up"),
    ("down", "down"),
    ("proj", "proj"),
    ("fidelity", "MLP fidelity"),
    ("gdn_gate_fp32", "GDN gate fp32"),
    ("gdn_qkv_fp32", "GDN conv/beta fp32"),
    ("vision_precision", "vision"),
    ("qwen36_changes", "qwen36 changes"),
    ("sweep200_warm_median_device_s", "sweep200 warm median device s"),
    ("sweep200_warm_ms_per_token", "ms/token"),
    ("sweep200_acc", "sweep200 acc"),
    ("sweep200_mean_dp", "sweep200 mean dp"),
    ("sweep200_max_dp", "sweep200 max dp"),
    ("sweep200_flips_margin", "sweep200 flips at margin"),
    ("ref16_max_dp", "ref16 max dp"),
    ("ref16_mean_dp", "ref16 mean dp"),
    ("ref16_flips_margin", "ref16 flips at margin"),
    ("ref16_near_tie", "ref16 near-tie"),
    ("ref16_fp32_max_dp", "ref16 max dp vs fp32"),
    ("ref_image8_max_dp", "ref_image8 max dp"),
    ("dev16_max_dp", "dev16 max dp"),
    ("dev16_sensitive_dp", "7b7ef383 dp"),
    ("bucket_1024_s", "bucket 1024 s"),
    ("bucket_8192_s", "8192 tokens s"),
    ("dram_free_after_weights_gib", "DRAM free after weights GiB"),
    ("slots", "slots"),
    ("engine_load_s", "load s"),
]


def table(rows, front, selected):
    header = [label for _, label in TABLE_COLS] + ["pareto", "selected"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    ordered = sorted(rows.values(), key=lambda r: (speed(r) is None, speed(r) or 0.0, r["variant"]))
    for r in ordered:
        vals = [str(r.get(key, "")) for key, _ in TABLE_COLS]
        vals.append("yes" if r["variant"] in front else "")
        vals.append("yes" if r["variant"] == selected else "")
        lines.append("| " + " | ".join(vals) + " |")
    for name in VARIANTS:
        if name in rows:
            continue
        v = VARIANTS[name]
        vals = [name, v.status if v.status != "blocked" else f"blocked: {v.blocker}"] + [""] * (len(TABLE_COLS) - 2)
        knobs = {short: v.env.get(key, STAGE1.get(key, "")) for key, short in KNOB_SHORT.items()}
        for i, (key, _) in enumerate(TABLE_COLS):
            if key in knobs:
                vals[i] = knobs[key]
            if key == "qwen36_changes":
                vals[i] = str(v.qwen36_changes)
        lines.append("| " + " | ".join(vals + ["", ""]) + " |")
    return "\n".join(lines)


def context_contract(r):
    prop = r["data"].get("propagation", {})
    slot_bytes = prop.get("slot_bytes") or {}
    free_gib = num(prop.get("dram_free_after_weights_gib"))
    kv = num(slot_bytes.get("kv_per_slot"))
    gdn = num(slot_bytes.get("gdn_per_slot"))
    out = {
        "max_state_tokens": MAX_STATE_DEFAULT,
        "max_schema_tokens": MAX_TAIL,
        "snapshot_slots_requested": 4,
        "snapshot_slots_allocated": prop.get("snapshot_slots"),
        "slot_fit_at_max_state_16384": slot_bytes.get("fit"),
        "dram_free_after_weights_gib": free_gib,
        "dram_free_after_slots_gib": num(prop.get("dram_free_after_slots_gib")),
    }
    if free_gib and kv and gdn:
        max_len = MAX_STATE_DEFAULT + MAX_TAIL
        kv_per_token = kv / max_len
        budget_per_slot = (free_gib * 2**30 - KV_RESERVE_BYTES) / 4
        tokens = int((budget_per_slot - gdn) / kv_per_token) // BLOCK * BLOCK
        out["largest_max_len_with_4_slots_tokens"] = tokens
        out["largest_state_with_4_slots_tokens"] = max(0, tokens - MAX_TAIL)
        after_slots = num(prop.get("dram_free_after_slots_gib"))
        if after_slots:
            tower_gib = free_gib - after_slots - 4 * (kv + gdn) / 2**30
            out["vision_tower_dram_gib_excluded_from_bound"] = round(tower_gib, 3)
            out["bound_optimism_tokens_from_tower"] = int(tower_gib * 2**30 / 4 / kv_per_token) // BLOCK * BLOCK
        out["status"] = (
            "arithmetic only: derived from the DRAM free after the text weights; the vision tower's DRAM is not "
            "subtracted; no state above 16,384 tokens was executed"
        )
        out["method"] = (
            "ClefEngine._fit_slots: (DRAM free after weights - 2 GiB reserve) / 4 slots, minus the GDN snapshot "
            "per slot, divided by the KV bytes per token (bf16 paged KV, 16 attention layers), rounded down to the "
            "64-token block; state = that length minus the 4,096-token schema allowance"
        )
    return out


ENGINE_READOUT_FIELDS = (
    "mlp_gate_up_packed",
    "mlp_w1",
    "mlp_down",
    "mlp_fidelity",
    "gdn_qkvzab",
    "gdn_out_colpar",
    "gdn_fidelity",
    "gdn_gate_fp32",
    "gdn_dt_bias",
    "gdn_rec_state",
    "attn_qkv",
    "attn_wo",
    "attn_fidelity",
    "embedding",
)
SENSITIVITY_MARGIN = 0.06
DISAGREEMENTS = (
    ("hard-v1/multi_hop/development/00017", "paged"),
    ("hard-v1/probability/development/00084", "value"),
)
DECIDER = ("cfpb/5845071", "issue")


def read_rows(path):
    return {row["id"]: row for row in (json.loads(l) for l in Path(path).read_text().splitlines() if l.strip())}


def top2(dist):
    ordered = sorted(dist.items(), key=lambda kv: -kv[1])
    return (
        ordered[0][0],
        round(ordered[0][1], 4),
        ordered[1][0],
        round(ordered[1][1], 4),
        round(ordered[0][1] - ordered[1][1], 4),
    )


def flips_at(ref_rows, cand_rows, margin):
    import parity_compare

    questions, _ = parity_compare.compare_rows(ref_rows, cand_rows, margin)
    return [q for q in questions if q["flip_at_margin"]]


def eligibility_under(rows, text, base_acc, best_mean_dp, flip_counts):
    eligible = [
        r
        for r in text
        if flip_counts[r["variant"]] == 0
        and num(r["sweep200_acc"]) >= base_acc - ACC_WINDOW_PP / 100
        and num(r["sweep200_mean_dp"]) <= best_mean_dp + MEAN_DP_WINDOW
    ]
    if not eligible:
        return {"eligible": [], "selected": None}
    fastest = min(speed(r) for r in eligible)
    ties = [r for r in eligible if speed(r) <= fastest * (1 + TIME_TIE_PCT / 100)]
    chosen = min(
        ties,
        key=lambda r: (int(r["qwen36_changes"]), num(r["sweep200_mean_dp"]), num(r["ref16_max_dp"]) or 0, speed(r)),
    )
    return {
        "eligible": sorted(r["variant"] for r in eligible),
        "fastest_s": fastest,
        "ties": [r["variant"] for r in ties],
        "selected": chosen["variant"],
    }


def selection_sensitivity(rows, report, sweep_root):
    text = [r for r in rows.values() if complete_text(r) and r["variant"] != "vision_upstream"]
    if not text or not report.get("baseline_complete"):
        return None
    ref16 = read_rows(REF / "ref_text_bf16.jsonl")
    ref32 = read_rows(REF / "ref_text_fp32.jsonl")
    cand = {r["variant"]: read_rows(Path(sweep_root) / r["variant"] / "ref16.jsonl") for r in text}
    base_acc = report["baseline_sweep200_acc"]
    best = report["best_sweep200_mean_dp"]
    bf16_at_written = {v: len(flips_at(ref16, c, FLIP_MARGIN)) for v, c in cand.items()}
    bf16_at_wider = {v: len(flips_at(ref16, c, SENSITIVITY_MARGIN)) for v, c in cand.items()}
    fp32_at_written = {v: len(flips_at(ref32, c, FLIP_MARGIN)) for v, c in cand.items()}
    decider_id, decider_q = DECIDER
    decider = {
        "cpu_bf16_top2": top2(ref16[decider_id]["probs"][decider_q]),
        "cpu_fp32_top2": top2(ref32[decider_id]["probs"][decider_q]),
        "tt_top2_per_variant": {v: top2(c[decider_id]["probs"][decider_q]) for v, c in cand.items()},
    }
    return {
        "statement": (
            "The written rule (CPU bf16 reference, flips at margin >= 0.05) is the one applied; it was fixed before "
            "the sweep and the selection does not change. This block reports how the eligibility outcome moves "
            "under two perturbations of the rule, as the stage 4 review asked."
        ),
        "decider": {"record": decider_id, "question": decider_q, **decider},
        "ref16_flips_at_margin_0_05_vs_bf16": bf16_at_written,
        "ref16_flips_at_margin_0_06_vs_bf16": bf16_at_wider,
        "ref16_flips_at_margin_0_05_vs_fp32": fp32_at_written,
        "written_rule": eligibility_under(rows, text, base_acc, best, bf16_at_written),
        "margin_0_06_vs_bf16": eligibility_under(rows, text, base_acc, best, bf16_at_wider),
        "margin_0_05_vs_fp32": eligibility_under(rows, text, base_acc, best, fp32_at_written),
    }


def known_disagreements(rows, sweep_root, probes):
    ref16 = read_rows(REF / "sweep200_text.ref_bf16.jsonl")
    fp32_path = REF / "sweep200_disagree2.ref_fp32.jsonl"
    ref32 = read_rows(fp32_path) if fp32_path.exists() else {}
    text = [r["variant"] for r in rows.values() if complete_text(r) and r["variant"] != "vision_upstream"]
    out = []
    for record_id, question in DISAGREEMENTS:
        if record_id not in ref16:
            continue
        ref = ref16[record_id]["probs"][question]
        ref_top = max(ref, key=ref.get)
        entry = {
            "record": record_id,
            "question": question,
            "cpu_bf16": {o: round(p, 4) for o, p in ref.items()},
            "cpu_fp32": (
                {o: round(p, 4) for o, p in ref32[record_id]["probs"][question].items()} if record_id in ref32 else None
            ),
            "per_variant": {},
        }
        for v in sorted(text):
            path = Path(sweep_root) / v / "sweep200.jsonl"
            if not path.exists():
                continue
            dist = read_rows(path).get(record_id, {}).get("probs", {}).get(question)
            if dist is None:
                continue
            top = max(dist, key=dist.get)
            entry["per_variant"][v] = {
                "tt_argmax": top,
                "flip": top != ref_top,
                "dp_vs_bf16": round(max(abs(ref[o] - dist[o]) for o in ref), 4),
                "tt_top2_margin": top2(dist)[4],
            }
        for probe in probes:
            rec = probe.get("records", {}).get(record_id)
            if rec is None:
                continue
            entry.setdefault("layer_probe", {})[probe["variant"]] = {
                "teacher_gdn": rec.get("teacher_gdn"),
                "teacher_attn": rec.get("teacher_attn"),
                "final_norm_min_mean": [rec["final_norm"]["min_pcc"], rec["final_norm"]["mean_pcc"]],
                "splice_first_hf_prefix_with_ref_argmax": (rec.get("splice") or {}).get(
                    "first_hf_prefix_with_ref_argmax"
                ),
                "report": probe.get("path"),
            }
        out.append(entry)
    return out


def selected_config(name, rows, report, probes=None):
    r = rows[name]
    data = r["data"]
    knobs = data.get("knobs", {})
    prop = data.get("propagation", {})
    dtypes = {k: v for k, v in prop.get("device_dtypes", {}).items() if k in ENGINE_READOUT_FIELDS}
    precision_env = {k: v for k, v in (prop.get("precision_env") or {}).items() if k in STAGE1}
    sets = data.get("sets", {})
    base = rows.get(BASELINE, {}).get("data", {}).get("sets", {})
    vision_knobs = {k: STAGE1[k] for k in ("CLEF_VISION_PRECISION", "CLEF_VISION_ACT_BF16")}
    runtime_flags = {k: knobs.get(k, STAGE1[k]) for k in STAGE1}
    runtime_flags.update(vision_knobs)
    if "QWEN35_GDN_STATE_BF16" in knobs:
        runtime_flags["QWEN35_GDN_STATE_BF16"] = knobs["QWEN35_GDN_STATE_BF16"]
    return {
        "config_id": name,
        "model_id": "Cloudflare/clef",
        "revision": "2f3de3dd85f379784083b0814d997ab627200f0c",
        "backbone": "Qwen3.5-27B class (64 layers: 48 Gated DeltaNet + 16 full attention) on the qwen36 TP path",
        "hardware": "Blackhole p150x2: TP=2 on a (1,2) mesh (this box: submesh of a (1,4) parent, FABRIC_1D)",
        "selected_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "weight_dtypes": {
            "mlp gate/up (packed fused SwiGLU weight)": knobs.get("QWEN36_MLP_GATE_UP_DTYPE"),
            "mlp down": knobs.get("QWEN36_MLP_DOWN_DTYPE"),
            "attention q, k, v, o (TP shard_w through the engine tp_proj_dtype override)": knobs.get(
                "QWEN36_PROJ_DTYPE"
            ),
            "gdn qkvzab and out projections (same override)": knobs.get("QWEN36_PROJ_DTYPE"),
            "embedding": "bf16",
            "norms": "bf16",
            "lm_head": "never on device (32-row stub); lexical rows read on the host from the safetensors",
            "vision tower (wqkv, fc1, fc2, wo, merger)": "bf16 under CLEF_VISION_PRECISION=accuracy",
        },
        "layer_exceptions": [],
        "compute_fidelities": {
            "mlp (fused gate/up all-gather matmul, down)": f"{knobs.get('QWEN36_MATMUL_FIDELITY')}, fp32 acc (QWEN36_MATMUL_FIDELITY)",
            "attention projections (TP module, own compute config)": dtypes.get("attn_fidelity"),
            "gdn projections (TP module, own compute config)": dtypes.get("gdn_fidelity"),
            "gdn decay gate": "fp32 (QWEN36_GDN_GATE_FP32=1)" if knobs.get("QWEN36_GDN_GATE_FP32") == "1" else "bf16",
            "gdn chunk kernel": "fused chunk_gated_delta_rule op: bf16 q, k, v by its contract (the host wrapper casts any other dtype back to bf16), fp32 g, beta and state; internal precision not exposed (open item)",
            "gdn causal conv and beta sigmoid": (
                "bf16 FIR multiply-accumulate and bf16 sigmoid; the fp32 trial (sweep variant gdn_qkv_fp32, flag "
                "removed after the sweep) is recorded in the README"
            ),
            "gdn causal conv kernel": "FIR multiply-accumulate (QWEN_GDN_CONV=fir in traced mode; eager masked path always FIR)",
            "vision MLP and merger": "HiFi4, fp32 acc under CLEF_VISION_PRECISION=accuracy; SDPA and QKV HiFi4 (tt_transformers accuracy preset)",
            "head": "fp32 on the host",
        },
        "activation_dtype": "bf16",
        "residual_dtype": "bf16",
        "ccl_dtype": "bf16 (qwen36 TP all-gather and reduce-scatter on the fractured hidden; not swept)",
        "kv_cache_dtype": "bf16 (ClefEngine.allocate_kv_caches)",
        "gdn_recurrent_state_dtype": (
            "bf16" if runtime_flags.get("QWEN35_GDN_STATE_BF16") == "1" else "fp32 (QWEN35_GDN_STATE_BF16 unset)"
        ),
        "logits_dtype": "no logits; normalized hidden rows read back as fp32 from a bf16 device tensor; joint schema head on the host in fp32",
        "sampling_dtype": "n/a (option probabilities only)",
        "runtime_flags": runtime_flags,
        "propagation_check": {
            "source": "ClefEngine._device_dtypes() read back from the built model in the sweep process (dtype_sweep.py), plus vision.describe()",
            "device_dtypes": dtypes,
            "device_dtypes_note": (
                "the 14 fields the shipped ClefEngine._device_dtypes() logs at start plus gdn_rec_state, which the "
                "engine fills after the slots are allocated; readout fields of removed trial flags are not kept"
            ),
            "precision_env_in_process": precision_env,
            "vision": prop.get("vision"),
            "traced_in_sweep": prop.get("traced"),
            "gdn_conv_impl_in_sweep": prop.get("gdn_conv_impl"),
        },
        "accuracy": {
            "sweep200": {k: v for k, v in (sets.get("sweep200", {}).get("accuracy") or {}).items()},
            "sweep200_parity_bf16": {
                k: v for k, v in (sets.get("sweep200", {}).get("parity_bf16") or {}).items() if k not in ("worst",)
            },
            "ref16_parity_bf16": {
                k: v for k, v in (sets.get("ref16", {}).get("parity_bf16") or {}).items() if k != "worst"
            },
            "ref16_parity_fp32": {
                k: v for k, v in (sets.get("ref16", {}).get("parity_fp32") or {}).items() if k != "worst"
            },
            "ref16_max_dp_bar": REF16_MAX_DP_BAR,
            "image_sets_baseline_stage1": {
                s: {k: v for k, v in (base.get(s, {}).get("parity_bf16") or {}).items() if k != "worst"}
                for s in ("ref_image8", "dev16_image")
                if s in base
            },
            "image_sensitive_record_baseline": base.get("dev16_image", {}).get("sensitive_record_dp"),
        },
        "performance": {
            "regime": "eager (CLEF_TRACED=0) full-path probs_for_request, device part per record, warm = bucket sequence seen before in the process; 64 layers, TP=2",
            "sweep200_latency": sets.get("sweep200", {}).get("latency"),
            "ref16_latency": sets.get("ref16", {}).get("latency"),
            "bucket_timing_prefill_hidden": data.get("bucket_timing"),
            "engine_load_s": prop.get("timings"),
            "traced_note": "stage 3 measured traced within 2 percent of eager per bucket (doc/optimized/perf_summary.json); the served path is traced",
        },
        "context_contract": context_contract(r),
        "selection_rule": report,
        "selection_sensitivity": selection_sensitivity(rows, report, SWEEP_ROOT),
        "known_disagreements": known_disagreements(rows, SWEEP_ROOT, probes or []),
        "evidence": {
            "variant_json": data.get("json"),
            "rows_dir": str(Path(data.get("json", "")).with_suffix("")),
            "sweep_results_csv": str(SWEEP_ROOT / "sweep_results.csv"),
            "timestamp": data.get("timestamp"),
        },
    }


INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SERIES = "#2a78d6"
SELECTED = "#d03b3b"


def plot(rows, front, selected, xkey, xlabel, out, vline, vline_label, title, subtitle):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [(num(r[xkey]), 1.0 / speed(r), r) for r in rows if num(r[xkey]) is not None and speed(r)]
    if not pts:
        return False
    fig, ax = plt.subplots(figsize=(9, 5.8), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.margins(x=0.15, y=0.2)
    fr = sorted([p for p in pts if p[2]["variant"] in front], key=lambda p: p[0])
    if len(fr) > 1:
        ax.plot([p[0] for p in fr], [p[1] for p in fr], color="#c3c2b7", linewidth=2, zorder=1)
    for x, y, r in sorted(pts, key=lambda p: p[2]["variant"] == selected):
        sel = r["variant"] == selected
        ax.scatter(
            [x],
            [y],
            s=120 if sel else 80,
            color=SELECTED if sel else SERIES,
            edgecolors=SURFACE,
            linewidths=2,
            zorder=5 if sel else 3,
        )
        ax.annotate(
            r["variant"],
            (x, y),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=8.5,
            color=INK if sel else "#52514e",
        )
    if vline is not None:
        ax.axvline(vline, color=MUTED, linewidth=1, linestyle=(0, (1, 3)), zorder=2)
        ax.annotate(
            vline_label,
            (vline, 0),
            xycoords=("data", "axes fraction"),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            color=MUTED,
            rotation=90,
            va="bottom",
        )
    ax.set_xlabel(xlabel, color="#52514e", fontsize=9.5)
    ax.set_ylabel(
        "eager sweep200 records / s (warm median device time, higher is better)", color="#52514e", fontsize=9.5
    )
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=22)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom")
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", default=str(SWEEP_ROOT))
    parser.add_argument("--doc-dir", default=str(DOC))
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--anomaly-probe", action="append", default=[])
    args = parser.parse_args()
    probes = []
    for path in args.anomaly_probe:
        probe = json.loads(Path(path).read_text())
        probe["path"] = str(path)
        probes.append(probe)
    rows = load_variants(args.sweep_root)
    report = select(rows)
    selected = report.get("selected")
    text_rows = [r for r in rows.values() if complete_text(r) and r["variant"] != "vision_upstream"]
    front = pareto_front(text_rows, "sweep200_acc")
    print(f"sweep root {args.sweep_root}: {len(rows)} variant file(s): {', '.join(sorted(rows))}")
    print()
    print(table(rows, front, selected))
    print()
    print("selection rule: " + RULE_TEXT)
    print()
    for name, checks in report.get("checks", {}).items():
        print(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in checks.items()))
    print(f"eligible: {report.get('eligible')}; ties: {report.get('ties')}")
    print(f"selected: {selected} ({report.get('reason')})")
    if "vision_upstream" in rows and BASELINE in rows:
        up = rows["vision_upstream"]
        base = rows[BASELINE]
        print(
            "vision control (reported): vision_upstream ref_image8 max dp "
            f"{up['ref_image8_max_dp']} / dev16 {up['dev16_max_dp']} / 7b7ef383 {up['dev16_sensitive_dp']} "
            f"against baseline_stage1 {base['ref_image8_max_dp']} / {base['dev16_max_dp']} / {base['dev16_sensitive_dp']}; "
            f"image median device s {up['image_median_device_s']} vs {base['image_median_device_s']}"
        )
    if not args.write:
        return
    doc = Path(args.doc_dir)
    doc.mkdir(parents=True, exist_ok=True)
    (doc / "pareto_table.md").write_text(table(rows, front, selected) + "\n")
    with (doc / "sweep_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for name in list(DEFAULT_RUN_ORDER) + sorted(set(rows) - set(DEFAULT_RUN_ORDER)):
            if name in rows:
                writer.writerow({k: rows[name].get(k, "") for k in CSV_FIELDS})
    results = {
        "variants": {name: {k: v for k, v in r.items() if k != "data"} for name, r in rows.items()},
        "blocked": {
            name: {"status": v.status, "blocker": v.blocker, "note": v.note, "knobs": {**STAGE1, **v.env}}
            for name, v in VARIANTS.items()
            if v.status == "blocked"
        },
        "pareto_front": sorted(front),
        "selection": report,
    }
    (doc / "sweep_results.json").write_text(json.dumps(results, indent=1))
    if selected:
        target = doc / "selected_precision_config.json"
        config = selected_config(selected, rows, report, probes)
        if target.exists():
            previous = json.loads(target.read_text())
            if previous.get("config_id") == config["config_id"] and previous.get("selected_at"):
                config["selected_at"] = previous["selected_at"]
                config["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        target.write_text(json.dumps(config, indent=1) + "\n")
        print(f"wrote {target}")
        sens = config.get("selection_sensitivity") or {}
        print(
            "selection sensitivity: written rule "
            f"{sens.get('written_rule', {}).get('selected')}; margin 0.06 vs bf16 "
            f"{sens.get('margin_0_06_vs_bf16', {}).get('selected')}; margin 0.05 vs fp32 "
            f"{sens.get('margin_0_05_vs_fp32', {}).get('selected')}"
        )
    base_acc = report.get("baseline_sweep200_acc")
    plot(
        text_rows,
        front,
        selected,
        "sweep200_acc",
        "sweep200 accuracy (278 labelled questions, 200 development records)",
        doc / "top1_perf_pareto.png",
        None if base_acc is None else base_acc - ACC_WINDOW_PP / 100,
        "baseline accuracy - 1.0 pp",
        "Clef stage 4: accuracy versus eager speed per variant",
        "selected in red; dotted line = the accuracy window of the selection rule; non-dominated front in gray",
    )
    for r in text_rows:
        r["agreement"] = 1.0 - (num(r["sweep200_mean_dp"]) or 0.0)
    plot(
        text_rows,
        pareto_front(text_rows, "agreement"),
        selected,
        "agreement",
        "1 - sweep200 mean dp against the CPU bf16 reference",
        doc / "agreement_perf_pareto.png",
        None if not text_rows else 1.0 - (min(num(r["sweep200_mean_dp"]) for r in text_rows) + MEAN_DP_WINDOW),
        "best mean dp + 0.005",
        "Clef stage 4: reference agreement versus eager speed per variant",
        "selected in red; dotted line = the mean-dp window of the selection rule",
    )
    print(f"wrote {doc / 'pareto_table.md'}, sweep_results.csv, sweep_results.json, two pareto PNGs")


if __name__ == "__main__":
    main()
