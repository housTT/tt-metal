"""Stage 4 datatype sweep harness for Cloudflare/clef on the (1,2) submesh (TP=2).

One process runs one variant: the precision knobs are read by models/demos/blackhole/qwen36/tt/precision.py
at import time, so the environment is set here before the engine is imported. For each variant the harness
builds the eager engine once (CLEF_TRACED=0), runs the record sets of the variant through
ClefEngine.probs_for_request (full path), writes the rows per set, compares them against the CPU bf16
reference (and against the CPU fp32 reference where one exists) with parity_compare.summarize, scores the
labelled rows with eval_metrics.score, times the eager prefill per bucket, and records DRAM free and the
dtypes and fidelities read back from the built model. The per-variant JSON and the CSV row feed
dtype_sweep_summary.py, which applies the selection rule of doc/datatype_sweep/README.md.

Usage (device, through devrun; one variant per process):
  python dtype_sweep.py --variant baseline_stage1
  python dtype_sweep.py --run-all                 runs every runnable variant in sequence, each in a subprocess
  python dtype_sweep.py --dry-run                 prints the plan (no ttnn import)
  python dtype_sweep.py --fake --variant X        host only: synthetic engine over the CPU reference rows
  python dtype_sweep.py --print-env --variant X   the knob values of a variant, shell style
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clef_paths import read_jsonl, snapshot_dir, write_jsonl

AUTOPORT = Path(__file__).resolve().parents[1]
REF = Path("/home/hous/dev/clef/reports/reference")
OUT_ROOT = Path("/home/hous/dev/clef/reports/sweep")
DOC = AUTOPORT / "doc" / "datatype_sweep"
FLIP_MARGIN = 0.05
ECE_BINS = 10
TIMING_LENGTHS = (128, 256, 512, 1024, 2048, 4096, 8192)
TIMING_REPS = 3
SENSITIVE_IMAGE_RECORD = "7b7ef383"
STAGE1 = {
    "QWEN36_MLP_GATE_UP_DTYPE": "bfp8",
    "QWEN36_MLP_DOWN_DTYPE": "bf16",
    "QWEN36_PROJ_DTYPE": "bfp8",
    "QWEN36_MATMUL_FIDELITY": "HiFi2",
    "QWEN36_GDN_GATE_FP32": "1",
    "CLEF_VISION_PRECISION": "accuracy",
    "CLEF_VISION_ACT_BF16": "1",
}
KNOB_SHORT = {
    "QWEN36_MLP_GATE_UP_DTYPE": "gate_up",
    "QWEN36_MLP_DOWN_DTYPE": "down",
    "QWEN36_PROJ_DTYPE": "proj",
    "QWEN36_MATMUL_FIDELITY": "fidelity",
    "QWEN36_GDN_GATE_FP32": "gdn_gate_fp32",
    "QWEN36_GDN_QKV_FP32": "gdn_qkv_fp32",
    "CLEF_VISION_PRECISION": "vision_precision",
    "CLEF_VISION_ACT_BF16": "vision_act_bf16",
    "QWEN35_GDN_STATE_BF16": "gdn_state_bf16",
}
FIXED_ENV = {"CLEF_TRACED": "0", "CLEF_VISION": "1", "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "8"}
TEXT_SETS = ("ref16", "sweep200")
IMAGE_SETS = ("ref_image8", "dev16_image")
SETS = {
    "ref16": {
        "records": REF / "records_text.jsonl",
        "ref_bf16": REF / "ref_text_bf16.jsonl",
        "ref_fp32": REF / "ref_text_fp32.jsonl",
        "kind": "text",
    },
    "sweep200": {
        "records": REF / "sweep200_text.jsonl",
        "ref_bf16": REF / "sweep200_text.ref_bf16.jsonl",
        "ref_fp32": None,
        "kind": "text",
    },
    "ref_image8": {
        "records": REF / "records_image.jsonl",
        "ref_bf16": REF / "ref_image_bf16.jsonl",
        "ref_fp32": REF / "ref_image_fp32.jsonl",
        "kind": "image",
    },
    "dev16_image": {
        "records": REF / "dev16_image.jsonl",
        "ref_bf16": REF / "dev16_image.ref_bf16.jsonl",
        "ref_fp32": None,
        "kind": "image",
    },
}


@dataclass
class Variant:
    name: str
    env: dict
    sets: tuple
    qwen36_changes: int
    status: str
    note: str
    blocker: str = ""
    fake_noise: float = 0.01
    fake_speed: float = 1.0
    extra: dict = field(default_factory=dict)


VARIANTS = {
    "baseline_stage1": Variant(
        "baseline_stage1",
        {},
        TEXT_SETS + IMAGE_SETS,
        1,
        "runnable",
        "stage 1 to 3 production knobs: gate/up bfp8, down bf16, proj bfp8, HiFi2 MLP, GDN decay gate fp32, "
        "vision accuracy precision; runs the image sets too as the comparator of vision_upstream",
        fake_noise=0.100,
    ),
    "proj_bf16": Variant(
        "proj_bf16",
        {"QWEN36_PROJ_DTYPE": "bf16"},
        TEXT_SETS,
        1,
        "runnable",
        "bf16 attention and GDN projections through the engine's tp_proj_dtype override; stage 1 measured "
        "identical per-layer PCC, 2x load time, 20.4 GiB of weights per device",
        fake_noise=0.080,
        fake_speed=1.05,
    ),
    "all_hifi4": Variant(
        "all_hifi4",
        {"QWEN36_MATMUL_FIDELITY": "HiFi4"},
        TEXT_SETS,
        1,
        "runnable",
        "QWEN36_MATMUL_FIDELITY=HiFi4; on the TP path the knob reaches the MLP compute configs only "
        "(tt/mlp.py); the TP attention and GDN modules carry their own compute configs",
        fake_noise=0.090,
        fake_speed=1.15,
    ),
    "gate_fp32_off": Variant(
        "gate_fp32_off",
        {"QWEN36_GDN_GATE_FP32": "0"},
        TEXT_SETS,
        0,
        "runnable",
        "control: the stage 1 pre-fix engine (bf16 GDN decay gate); expected to fail the flip rule",
        fake_noise=0.600,
        fake_speed=0.99,
    ),
    "down_bfp8": Variant(
        "down_bfp8",
        {"QWEN36_MLP_DOWN_DTYPE": "bfp8"},
        TEXT_SETS,
        1,
        "runnable",
        "speed candidate: bfp8 MLP down projection (the qwen36 default); halves the down-projection weight bytes",
        fake_noise=0.120,
        fake_speed=0.93,
    ),
    "vision_upstream": Variant(
        "vision_upstream",
        {"CLEF_VISION_PRECISION": "upstream", "CLEF_VISION_ACT_BF16": "0"},
        IMAGE_SETS,
        1,
        "runnable",
        "image-only control: the upstream qwen36 vision tower precision (bfp8 wqkv, fc1, fc2, bfp8 q and v "
        "into the SDPA, HiFi2 MLP and merger); text knobs as baseline_stage1",
        fake_noise=0.300,
        fake_speed=0.97,
    ),
    "gdn_state_bf16": Variant(
        "gdn_state_bf16",
        {"QWEN35_GDN_STATE_BF16": "1"},
        TEXT_SETS,
        1,
        "optional",
        "optional speed candidate: bf16 GDN recurrent state (qwen36 flag QWEN35_GDN_STATE_BF16, read by "
        "gdn/tp.py); halves the per-slot GDN snapshot; not in the default run list",
        fake_noise=0.015,
        fake_speed=0.98,
    ),
    "gdn_qkv_fp32": Variant(
        "gdn_qkv_fp32",
        {"QWEN36_GDN_QKV_FP32": "1"},
        TEXT_SETS,
        2,
        "blocked",
        "the stage 4 qwen36 amendment, measured on 2026 Oct 05 and removed: QWEN36_GDN_QKV_FP32=1 ran the GDN "
        "causal FIR conv, its SiLU and the beta sigmoid in fp32 before the fused chunk_gated_delta_rule op (the op "
        "casts q, k, v back to bf16 by contract and takes beta in fp32); the stage 1 open item 1 candidate",
        blocker="not selected under the written rule (ref16 1 flip at margin, max dp 0.1243, 22 percent slower), so the "
        "flag was removed from gdn/tp.py per the stage 4 amendment; the measured results stay in "
        "/home/hous/dev/clef/reports/sweep/gdn_qkv_fp32.json and doc/datatype_sweep/README.md; the knob no longer "
        "exists in the code",
        fake_noise=0.007,
        fake_speed=1.10,
    ),
    "gdn_qkv_fp32_proj_bf16": Variant(
        "gdn_qkv_fp32_proj_bf16",
        {"QWEN36_GDN_QKV_FP32": "1", "QWEN36_PROJ_DTYPE": "bf16"},
        TEXT_SETS,
        2,
        "blocked",
        "gdn_qkv_fp32 combined with bf16 attention and GDN projections, measured on 2026 Oct 05 and removed with "
        "the flag",
        blocker="not selected (ref16 1 flip at margin, 27 percent slower); the flag no longer exists in the code; "
        "results in /home/hous/dev/clef/reports/sweep/gdn_qkv_fp32_proj_bf16.json",
        fake_noise=0.006,
        fake_speed=1.15,
    ),
    "mlp_gateup_bf16": Variant(
        "mlp_gateup_bf16",
        {"QWEN36_MLP_GATE_UP_DTYPE": "bf16"},
        TEXT_SETS,
        2,
        "blocked",
        "bf16 MLP gate/up weights",
        blocker="the fused SwiGLU all-gather matmul (tp_common.py MinimalMatmulConfig) clashes L1 circular "
        "buffers at bucket 512 with bf16 weights (stage 1, stage1_engine_l4_t300_debug.log); needs a "
        "tp_common block-size change keyed on dtype",
        fake_noise=0.006,
        fake_speed=1.12,
    ),
}
DEFAULT_RUN_ORDER = (
    "baseline_stage1",
    "down_bfp8",
    "proj_bf16",
    "all_hifi4",
    "gate_fp32_off",
    "vision_upstream",
)

CSV_FIELDS = [
    "variant",
    "status",
    "gate_up",
    "down",
    "proj",
    "fidelity",
    "gdn_gate_fp32",
    "gdn_qkv_fp32",
    "vision_precision",
    "vision_act_bf16",
    "gdn_state_bf16",
    "qwen36_changes",
    "ref16_max_dp",
    "ref16_mean_dp",
    "ref16_flips_margin",
    "ref16_near_tie",
    "ref16_fp32_max_dp",
    "ref16_fp32_flips_margin",
    "sweep200_max_dp",
    "sweep200_mean_dp",
    "sweep200_flips_margin",
    "sweep200_near_tie",
    "sweep200_acc",
    "sweep200_correct",
    "sweep200_questions",
    "sweep200_brier",
    "sweep200_ece",
    "sweep200_median_device_s",
    "sweep200_warm_median_device_s",
    "sweep200_warm_ms_per_token",
    "sweep200_median_seconds",
    "ref_image8_max_dp",
    "ref_image8_flips_margin",
    "ref_image8_fp32_max_dp",
    "dev16_max_dp",
    "dev16_flips_margin",
    "dev16_acc",
    "dev16_sensitive_dp",
    "image_median_device_s",
    "bucket_128_s",
    "bucket_1024_s",
    "bucket_2048_s",
    "bucket_8192_s",
    "dram_free_after_weights_gib",
    "dram_free_after_slots_gib",
    "slots",
    "slot_fit",
    "engine_load_s",
    "timestamp",
    "json",
]


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), message, flush=True)


def variant_env(name):
    v = VARIANTS[name]
    env = dict(STAGE1)
    env.update(v.env)
    return env


def short_knobs(env):
    return {short: env.get(key, "") for key, short in KNOB_SHORT.items()}


def plan(name, out_root):
    v = VARIANTS[name]
    env = variant_env(name)
    diff = {k: env[k] for k in v.env}
    return {
        "variant": name,
        "status": v.status,
        "sets": list(v.sets),
        "qwen36_changes": v.qwen36_changes,
        "knobs": env,
        "changed_from_stage1": diff,
        "note": v.note,
        "blocker": v.blocker,
        "json": str(Path(out_root) / f"{name}.json"),
        "rows": {s: str(Path(out_root) / name / f"{s}.jsonl") for s in v.sets},
    }


def print_dry_run(out_root):
    print(f"dtype sweep plan (no device): out_root={out_root}")
    print(f"record sets: " + ", ".join(f"{k} ({SETS[k]['kind']}, {SETS[k]['records'].name})" for k in SETS))
    print(f"default run order (--run-all): {', '.join(DEFAULT_RUN_ORDER)}")
    print()
    header = ("variant", "status", "qwen36 changes", "sets", "knobs changed from stage 1")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for name in VARIANTS:
        p = plan(name, out_root)
        changed = ", ".join(f"{k}={v}" for k, v in p["changed_from_stage1"].items()) or "(stage 1 values)"
        print(f"| {name} | {p['status']} | {p['qwen36_changes']} | {' '.join(p['sets'])} | {changed} |")
    print()
    for name in VARIANTS:
        p = plan(name, out_root)
        print(f"{name}: {p['note']}")
        if p["blocker"]:
            print(f"  blocked: {p['blocker']}")
        print(f"  json {p['json']}")
    print()
    print("full knob table:")
    for name in VARIANTS:
        env = variant_env(name)
        print(f"  {name}: " + " ".join(f"{k}={v}" for k, v in env.items()))
    missing = [
        str(p) for s in SETS.values() for p in (s["records"], s["ref_bf16"], s["ref_fp32"]) if p and not p.exists()
    ]
    print()
    print("reference files present: " + ("all" if not missing else "MISSING " + ", ".join(missing)))


def apply_env(name):
    env = variant_env(name)
    for key, value in FIXED_ENV.items():
        os.environ.setdefault(key, value)
    os.environ.update(env)
    snapshot = str(snapshot_dir(os.environ.get("CLEF_MODEL") or None))
    os.environ["CLEF_MODEL"] = snapshot
    os.environ["HF_MODEL"] = snapshot
    return env


def bucket_signature(T, chunk=1024):
    full, rem = divmod(T, chunk)
    last = None
    if rem:
        last = next(b for b in (128, 256, 512, 1024) if b >= rem)
    return (full, last)


def softmax_perturb(dist, sigma, rng):
    keys = list(dist)
    logits = [math.log(max(float(dist[k]), 1e-9)) + rng.gauss(0.0, sigma) for k in keys]
    top = max(logits)
    weights = [math.exp(x - top) for x in logits]
    total = sum(weights)
    return {k: w / total for k, w in zip(keys, weights)}


class FakeSweepEngine:
    def __init__(self, name, sets):
        v = VARIANTS[name]
        self.name = name
        self.noise = v.fake_noise
        self.speed = v.fake_speed
        self.reference = {}
        for s in sets:
            for row in read_jsonl(SETS[s]["ref_bf16"]):
                self.reference[row["id"]] = row
        self.snapshot_slots = 4
        self.device_dtypes = {"fake": True, "variant": name}
        self.precision = variant_env(name)
        self.timings = {"load_total_s": 0.0}
        self.dram_free_after_weights = int(14.5 * 2**30)
        self.dram_free_after_slots = int(11.5 * 2**30)
        self.slot_bytes = {"kv_per_slot": 671088640, "gdn_per_slot": 91226112, "fit": 17}
        self.cache_dir = "fake"
        self.traced = False
        self.gdn_conv_impl = "kda"
        self.vision = None

    def device_seconds(self, tokens):
        return (0.15 + 0.00019 * tokens) * self.speed

    def prefill_hidden(self, ids, slot=0, media=None):
        return None

    def probs_for_request(self, record, mode="full", slot=0):
        ref = self.reference[record["id"]]
        rng = random.Random(f"{self.name}:{record['id']}")
        probs = {qid: softmax_perturb(dist, self.noise, rng) for qid, dist in ref["probs"].items()}
        answers = {}
        for qid, dist in probs.items():
            kind = record["questions"][qid]["type"]
            if kind == "noul":
                answers[qid] = {"type": "noul", "noul": round(dist["true"], 4)}
            elif kind == "choice":
                top = max(dist, key=dist.__getitem__)
                answers[qid] = {
                    "type": "choice",
                    "choice": top,
                    "confidence": round(dist[top], 4),
                    "probabilities": {k: round(p, 4) for k, p in dist.items()},
                }
            else:
                levels = sorted(dist, key=int)
                answers[qid] = {
                    "type": "score",
                    "score": round(sum(int(k) * dist[k] for k in levels), 4),
                    "probabilities": {k: round(dist[k], 4) for k in levels},
                }
        tokens = int(ref["input_tokens"])
        device_s = self.device_seconds(tokens)
        return {
            "model": "clef",
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
            "probs": probs,
            "input_tokens": tokens,
            "seconds": round(device_s + 0.03, 3),
            "timing": {"encode_s": 0.01, "device_s": round(device_s, 3), "head_s": 0.02},
            "mode": mode,
            "cache_hit": None,
        }


def propagation(engine, fake):
    out = {
        "device_dtypes": dict(engine.device_dtypes),
        "precision_env": dict(engine.precision),
        "cache_dir": str(engine.cache_dir),
        "traced": engine.traced,
        "gdn_conv_impl": engine.gdn_conv_impl,
        "snapshot_slots": engine.snapshot_slots,
        "slot_bytes": dict(engine.slot_bytes),
        "dram_free_after_weights_gib": round(engine.dram_free_after_weights / 2**30, 3),
        "dram_free_after_slots_gib": round(engine.dram_free_after_slots / 2**30, 3),
        "timings": dict(engine.timings),
        "fake": fake,
    }
    text_free = getattr(engine, "dram_free_after_text_weights", None)
    if text_free is not None:
        out["dram_free_after_text_weights_gib"] = round(text_free / 2**30, 3)
    if engine.vision is not None:
        out["vision"] = engine.vision.describe()
    return out


def ids_for_length(engine, T, fake):
    import torch

    if fake:
        return torch.arange(T, dtype=torch.long).unsqueeze(0)
    from models.autoports.cloudflare_clef.tt import encode as clef_encode

    row = read_jsonl(SETS["ref16"]["records"])[0]
    request = {k: row[k] for k in ("model", "state", "questions") if k in row}
    base = torch.tensor(list(clef_encode.encode(engine.tokenizer, request).input_ids), dtype=torch.long)
    reps = math.ceil(T / base.numel())
    return base.repeat(reps)[:T].unsqueeze(0)


def bucket_timing(engine, fake, lengths=TIMING_LENGTHS, reps=TIMING_REPS):
    out = {}
    for T in lengths:
        ids = ids_for_length(engine, T, fake)
        times = []
        for rep in range(reps + 1):
            if fake:
                times.append(engine.device_seconds(T))
                continue
            import ttnn

            ttnn.synchronize_device(engine.mesh)
            t0 = time.perf_counter()
            engine.prefill_hidden(ids, slot=0)
            times.append(time.perf_counter() - t0)
        warm = times[1:]
        out[str(T)] = {
            "median_s": round(statistics.median(warm), 4),
            "min_s": round(min(warm), 4),
            "first_s": round(times[0], 4),
            "reps": reps,
            "ms_per_token": round(1000.0 * statistics.median(warm) / T, 4),
        }
        log(f"bucket timing T={T}: first {times[0]:.3f} s, warm median {out[str(T)]['median_s']:.3f} s")
    return out


def latency_stats(rows):
    ok = [r for r in rows if "error" not in r]
    if not ok:
        return {"ok": 0, "failed": len(rows)}
    seen = set()
    device, warm_device, warm_tokens, warm_ms_per_token, seconds = [], [], 0, [], []
    for r in ok:
        T = int(r["input_tokens"])
        sig = bucket_signature(T)
        d = float(r["timing"]["device_s"])
        device.append(d)
        seconds.append(float(r["seconds"]))
        if sig in seen:
            warm_device.append(d)
            warm_tokens += T
            warm_ms_per_token.append(1000.0 * d / T)
        seen.add(sig)
    out = {
        "ok": len(ok),
        "failed": len(rows) - len(ok),
        "tokens_min": min(int(r["input_tokens"]) for r in ok),
        "tokens_median": int(statistics.median(int(r["input_tokens"]) for r in ok)),
        "tokens_max": max(int(r["input_tokens"]) for r in ok),
        "median_device_s": round(statistics.median(device), 4),
        "mean_device_s": round(statistics.mean(device), 4),
        "max_device_s": round(max(device), 4),
        "median_seconds": round(statistics.median(seconds), 4),
        "sum_seconds": round(sum(seconds), 2),
        "warm_rows": len(warm_device),
    }
    if warm_device:
        out["warm_median_device_s"] = round(statistics.median(warm_device), 4)
        out["warm_ms_per_token"] = round(1000.0 * sum(warm_device) / warm_tokens, 4)
        out["warm_ms_per_token_median_row"] = round(statistics.median(warm_ms_per_token), 4)
    return out


def compact_parity(summary):
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
    out = {k: overall[k] for k in keys if k in overall}
    out["records_joined"] = summary["records"]["joined"]
    out["candidate_errors"] = summary["records"]["candidate_errors"]
    out["flips"] = summary["flips"]
    out["worst"] = sorted(
        (
            {"id": q["id"], "question": q["question"], "dp": round(q["dp"], 4), "ref_margin": round(q["ref_margin"], 4)}
            for q in summary["per_question"]
        ),
        key=lambda q: -q["dp"],
    )[:5]
    return out


def per_record_max_dp(summary):
    out = {}
    for q in summary["per_question"]:
        out[q["id"]] = max(out.get(q["id"], 0.0), float(q["dp"]))
    return out


def evaluate_set(name, set_name, rows, out_dir, parity_mod, eval_mod):
    spec = SETS[set_name]
    records = read_jsonl(spec["records"])
    result = {"records": len(records), "latency": latency_stats(rows)}
    for ref_key in ("ref_bf16", "ref_fp32"):
        path = spec[ref_key]
        if path is None:
            continue
        label = f"{name} {set_name} vs CPU {ref_key[4:]}"
        summary = parity_mod.summarize(read_jsonl(path), rows, FLIP_MARGIN, ECE_BINS, label)
        (out_dir / f"{set_name}_parity_{ref_key[4:]}.json").write_text(json.dumps(summary, indent=1))
        (out_dir / f"{set_name}_parity_{ref_key[4:]}.md").write_text(parity_mod.markdown(summary))
        result[f"parity_{ref_key[4:]}"] = compact_parity(summary)
        if ref_key == "ref_bf16":
            result["per_record_max_dp"] = per_record_max_dp(summary)
    if any(r.get("_label") for r in records):
        score = eval_mod.score(records, rows, "accuracy", None)
        result["accuracy"] = {
            k: score[k] for k in ("scored_questions", "correct", "accuracy", "result_errors", "missing_results")
        }
        (out_dir / f"{set_name}_accuracy.json").write_text(json.dumps(score, indent=1))
    if spec["kind"] == "image":
        sensitive = [k for k in result.get("per_record_max_dp", {}) if k.startswith(SENSITIVE_IMAGE_RECORD)]
        if sensitive:
            result["sensitive_record_dp"] = {k: result["per_record_max_dp"][k] for k in sensitive}
    return result


def load_tools():
    import eval_metrics
    import parity_compare

    return parity_compare, eval_metrics


def fmt(x, digits=4):
    if x is None or x == "":
        return ""
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def csv_row(name, env, result):
    v = VARIANTS[name]
    sets = result.get("sets", {})
    ref16 = sets.get("ref16", {})
    sweep = sets.get("sweep200", {})
    img8 = sets.get("ref_image8", {})
    dev16 = sets.get("dev16_image", {})
    prop = result.get("propagation", {})
    buckets = result.get("bucket_timing", {})
    knobs = short_knobs(env)

    def g(block, *keys, digits=4):
        cur = block
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return ""
            cur = cur[k]
        return fmt(cur, digits)

    sensitive = dev16.get("sensitive_record_dp") or {}
    image_device = [
        float(s["latency"]["median_device_s"])
        for s in (img8, dev16)
        if s.get("latency", {}).get("median_device_s") is not None
    ]
    return {
        "variant": name,
        "status": result["status"],
        "gate_up": knobs["gate_up"],
        "down": knobs["down"],
        "proj": knobs["proj"],
        "fidelity": knobs["fidelity"],
        "gdn_gate_fp32": knobs["gdn_gate_fp32"],
        "gdn_qkv_fp32": knobs["gdn_qkv_fp32"] or "0",
        "vision_precision": knobs["vision_precision"],
        "vision_act_bf16": knobs["vision_act_bf16"],
        "gdn_state_bf16": knobs["gdn_state_bf16"] or "0",
        "qwen36_changes": v.qwen36_changes,
        "ref16_max_dp": g(ref16, "parity_bf16", "max_dp"),
        "ref16_mean_dp": g(ref16, "parity_bf16", "mean_dp"),
        "ref16_flips_margin": g(ref16, "parity_bf16", "flips_at_margin"),
        "ref16_near_tie": g(ref16, "parity_bf16", "near_tie_flips"),
        "ref16_fp32_max_dp": g(ref16, "parity_fp32", "max_dp"),
        "ref16_fp32_flips_margin": g(ref16, "parity_fp32", "flips_at_margin"),
        "sweep200_max_dp": g(sweep, "parity_bf16", "max_dp"),
        "sweep200_mean_dp": g(sweep, "parity_bf16", "mean_dp"),
        "sweep200_flips_margin": g(sweep, "parity_bf16", "flips_at_margin"),
        "sweep200_near_tie": g(sweep, "parity_bf16", "near_tie_flips"),
        "sweep200_acc": g(sweep, "accuracy", "accuracy"),
        "sweep200_correct": g(sweep, "accuracy", "correct"),
        "sweep200_questions": g(sweep, "accuracy", "scored_questions"),
        "sweep200_brier": g(sweep, "parity_bf16", "cand_brier"),
        "sweep200_ece": g(sweep, "parity_bf16", "cand_ece"),
        "sweep200_median_device_s": g(sweep, "latency", "median_device_s"),
        "sweep200_warm_median_device_s": g(sweep, "latency", "warm_median_device_s"),
        "sweep200_warm_ms_per_token": g(sweep, "latency", "warm_ms_per_token"),
        "sweep200_median_seconds": g(sweep, "latency", "median_seconds"),
        "ref_image8_max_dp": g(img8, "parity_bf16", "max_dp"),
        "ref_image8_flips_margin": g(img8, "parity_bf16", "flips_at_margin"),
        "ref_image8_fp32_max_dp": g(img8, "parity_fp32", "max_dp"),
        "dev16_max_dp": g(dev16, "parity_bf16", "max_dp"),
        "dev16_flips_margin": g(dev16, "parity_bf16", "flips_at_margin"),
        "dev16_acc": g(dev16, "accuracy", "accuracy"),
        "dev16_sensitive_dp": fmt(max(sensitive.values())) if sensitive else "",
        "image_median_device_s": fmt(statistics.median(image_device)) if image_device else "",
        "bucket_128_s": g(buckets, "128", "median_s"),
        "bucket_1024_s": g(buckets, "1024", "median_s"),
        "bucket_2048_s": g(buckets, "2048", "median_s"),
        "bucket_8192_s": g(buckets, "8192", "median_s"),
        "dram_free_after_weights_gib": g(prop, "dram_free_after_weights_gib", digits=3),
        "dram_free_after_slots_gib": g(prop, "dram_free_after_slots_gib", digits=3),
        "slots": g(prop, "snapshot_slots"),
        "slot_fit": g(prop, "slot_bytes", "fit"),
        "engine_load_s": g(prop, "timings", "load_total_s", digits=1),
        "timestamp": result["timestamp"],
        "json": result["json"],
    }


def write_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def run(name, out_root, fake, sets, limit, timing, n_layers, max_state_len):
    v = VARIANTS[name]
    if v.status == "blocked" and not fake:
        raise SystemExit(f"{name} is blocked: {v.blocker}")
    env = variant_env(name)
    if not fake:
        env = apply_env(name)
    sets = tuple(sets) if sets else v.sets
    out_root = Path(out_root)
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{name}.json"
    result = {
        "variant": name,
        "status": "started",
        "fake": fake,
        "knobs": env,
        "changed_from_stage1": {k: env[k] for k in v.env},
        "qwen36_changes": v.qwen36_changes,
        "note": v.note,
        "sets_requested": list(sets),
        "limit": limit,
        "n_layers": n_layers,
        "argv": sys.argv,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "json": str(out_path),
        "sets": {},
    }
    log(f"variant {name}: knobs " + " ".join(f"{k}={val}" for k, val in env.items()))
    parity_mod, eval_mod = load_tools()
    mesh_ctx = None
    try:
        if fake:
            engine = FakeSweepEngine(name, sets)
        else:
            from reference_rows import run_rows as engine_run_rows

            from models.autoports.cloudflare_clef.tt.engine import ClefEngine, tp2_mesh

            mesh_ctx = tp2_mesh()
            mesh = mesh_ctx.__enter__()
            t0 = time.perf_counter()
            engine = ClefEngine(mesh, max_state_len=max_state_len, n_layers=n_layers, traced=False)
            log(f"engine built in {time.perf_counter() - t0:.1f} s")
        result["propagation"] = propagation(engine, fake)
        log(f"propagation: {json.dumps(result['propagation']['device_dtypes'])}")
        for set_name in sets:
            rows_in = read_jsonl(SETS[set_name]["records"])
            if limit:
                rows_in = rows_in[:limit]
            log(f"set {set_name}: {len(rows_in)} records")
            if fake:
                rows = []
                for row in rows_in:
                    out = {"id": row["id"]}
                    if "_label" in row:
                        out["_label"] = row["_label"]
                    out.update(engine.probs_for_request(row))
                    rows.append(out)
            else:
                rows = engine_run_rows(engine, rows_in, mode="full", slot=0)
            write_jsonl(out_dir / f"{set_name}.jsonl", rows)
            result["sets"][set_name] = evaluate_set(name, set_name, rows, out_dir, parity_mod, eval_mod)
            brief = {k: result["sets"][set_name].get(k) for k in ("latency",) if k in result["sets"][set_name]}
            for key in ("parity_bf16", "parity_fp32", "accuracy"):
                block = result["sets"][set_name].get(key)
                if block:
                    brief[key] = {k: block[k] for k in block if k not in ("flips", "worst")}
            log(f"set {set_name} result: {json.dumps(brief)}")
            (out_path).write_text(json.dumps(result, indent=1))
        if timing:
            result["bucket_timing"] = bucket_timing(engine, fake)
        result["status"] = "ok"
    except Exception as error:
        result["status"] = f"error: {type(error).__name__}: {str(error).splitlines()[0] if str(error) else ''}"
        result["error"] = f"{type(error).__name__}: {error}"
        log(result["error"])
        raise
    finally:
        result["finished"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        out_path.write_text(json.dumps(result, indent=1))
        log(f"wrote {out_path}")
        write_csv(out_root / "sweep_results.csv", csv_row(name, env, result))
        if mesh_ctx is not None:
            mesh_ctx.__exit__(None, None, None)
    return result


def run_all(args):
    names = args.variants.split(",") if args.variants else list(DEFAULT_RUN_ORDER)
    out_root = Path(args.out_root)
    for name in names:
        v = VARIANTS[name]
        if v.status == "blocked" and not args.fake:
            log(f"skip {name}: blocked ({v.blocker})")
            continue
        existing = out_root / f"{name}.json"
        if existing.exists() and not args.force:
            status = json.loads(existing.read_text()).get("status")
            if status == "ok":
                log(f"skip {name}: {existing} has status ok (use --force to rerun)")
                continue
        cmd = [sys.executable, str(Path(__file__).resolve()), "--variant", name, "--out-root", str(out_root)]
        if args.fake:
            cmd.append("--fake")
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.no_timing:
            cmd.append("--no-timing")
        if args.n_layers:
            cmd += ["--n-layers", str(args.n_layers)]
        if args.sets:
            cmd += ["--sets", args.sets]
        log(f"run {name}: {' '.join(cmd)}")
        started = time.perf_counter()
        proc = subprocess.run(cmd, timeout=args.variant_timeout)
        log(f"{name} exit {proc.returncode} after {time.perf_counter() - started:.0f} s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS))
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--variants", default=None, help="comma list for --run-all (default: the runnable order)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-env", action="store_true")
    parser.add_argument("--fake", action="store_true", help="host only: synthetic engine over the CPU reference rows")
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    parser.add_argument("--sets", default=None, help="comma list of record sets to run (default: the variant's sets)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--max-state-len", type=int, default=16384)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--variant-timeout", type=int, default=5400)
    args = parser.parse_args()
    if args.dry_run:
        print_dry_run(args.out_root)
        return
    if args.run_all:
        run_all(args)
        return
    if args.variant is None:
        parser.error("--variant, --run-all or --dry-run is required")
    if args.print_env:
        env = variant_env(args.variant)
        print(" ".join(f"{k}={v}" for k, v in {**FIXED_ENV, **env}.items()))
        return
    sets = args.sets.split(",") if args.sets else None
    run(
        args.variant,
        args.out_root,
        args.fake,
        sets,
        args.limit,
        not args.no_timing,
        args.n_layers,
        args.max_state_len,
    )


if __name__ == "__main__":
    main()
