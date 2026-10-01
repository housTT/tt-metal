import argparse
import csv
import importlib.util
import inspect
import json
import os
import statistics
import sys
import time
from pathlib import Path

KEV_ROOT = Path("/home/hous/dev/kev")
REF = KEV_ROOT / "reports" / "reference"
SWEEP = KEV_ROOT / "reports" / "sweep"
SUBSET_PATH = SWEEP / "subset200.jsonl"
CSV_PATH = SWEEP / "sweep_results.csv"
EVALS = KEV_ROOT / "kev" / "evals"
METRICS_PY = KEV_ROOT / "kev" / "kev" / "metrics.py"
SUBSET_SPEC = (("hard-v1", 70), ("devtools-v1", 70), ("documents-v1", 60))
DEVICE_PARAMS = dict(l1_small_size=24576, num_command_queues=2, trace_region_size=0)
DEFAULT_ENV = {
    "HF_MODEL": "/home/hous/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B-Base/snapshots/68c46c4b3498877f3ef123c856ecfde50c39f404",
    "KEV_RUN": "/home/hous/.cache/huggingface/hub/models--jaredpalmer--kev-9b/snapshots/db029f08b290afd9fee4aa4bbcd9ae48602d1eb0",
    "MESH_DEVICE": "P150",
    "HF_HUB_OFFLINE": "1",
    "QWEN_SDPA_BF8": "0",
}

VARIANTS = {
    "baseline": dict(gate_up="bfp4", down="bfp8", proj="bfp8", fidelity="LoFi", gdn_fp32_state="0", cache="baseline"),
    "mlp_bfp8": dict(gate_up="bfp8", down="bfp8", proj="bfp8", fidelity="LoFi", gdn_fp32_state="0", cache="mlp_bfp8"),
    "all_bfp8_hifi2": dict(
        gate_up="bfp8", down="bfp8", proj="bfp8", fidelity="HiFi2", gdn_fp32_state="0", cache="all_bfp8"
    ),
    "mlp_bf16": dict(gate_up="bf16", down="bf16", proj="bfp8", fidelity="HiFi2", gdn_fp32_state="0", cache="mlp_bf16"),
    "all_bf16": dict(gate_up="bf16", down="bf16", proj="bf16", fidelity="HiFi2", gdn_fp32_state="0", cache="all_bf16"),
    "all_bfp8_gdnfp32": dict(
        gate_up="bfp8", down="bfp8", proj="bfp8", fidelity="LoFi", gdn_fp32_state="1", cache="all_bfp8"
    ),
}

CSV_FIELDS = [
    "variant",
    "gate_up_dtype",
    "down_dtype",
    "proj_dtype",
    "fidelity",
    "gdn_fp32_state",
    "matmul_policy",
    "cache_dir",
    "ref_rows",
    "argmax_flips",
    "max_dp",
    "mean_dp",
    "min_pcc",
    "median_pcc",
    "ref_mean_row_s",
    "subset_rows",
    "acc",
    "brier",
    "ece",
    "nll",
    "mean_row_s",
    "warm_row_s",
    "warm_ms_per_token",
    "warm_rows",
    "engine_load_s",
    "status",
    "timestamp",
    "json",
]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def variant_env(name):
    v = VARIANTS[name]
    return {
        "QWEN36_MLP_GATE_UP_DTYPE": v["gate_up"],
        "QWEN36_MLP_DOWN_DTYPE": v["down"],
        "QWEN36_PROJ_DTYPE": v["proj"],
        "QWEN36_MATMUL_FIDELITY": v["fidelity"],
        "QWEN_GDN_FP32_STATE": v["gdn_fp32_state"],
        "TT_CACHE_PATH": str(KEV_ROOT / f"tt_cache_{v['cache']}"),
    }


def apply_env(name, matmul_policy=None, down_auto=False):
    for k, val in DEFAULT_ENV.items():
        os.environ.setdefault(k, val)
    env = variant_env(name)
    if matmul_policy is not None:
        env["KEV_MATMUL_POLICY"] = matmul_policy
    if down_auto:
        env["QWEN9B_MLP_DOWN_AUTO"] = "1"
    os.environ.update(env)
    return env


def load_metrics_module():
    spec = importlib.util.spec_from_file_location("kev_metrics", METRICS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def label_index(q):
    if q["type"] == "choice":
        return list(q["criteria"]).index(q["label"])
    return int(q["label"])


def materialize_subset(force=False):
    if SUBSET_PATH.exists() and not force:
        return [json.loads(line) for line in open(SUBSET_PATH)]
    from transformers import AutoTokenizer

    from models.autoports.jaredpalmer_kev_9b.tt.encode import rows_for_record

    ref = json.load(open(REF / "rows.json"))
    tok = AutoTokenizer.from_pretrained(ref["base"], revision=ref["base_revision"])
    out, dropped = [], []
    for suite, n in SUBSET_SPEC:
        with open(EVALS / suite / "development.jsonl") as f:
            records = [json.loads(line) for _, line in zip(range(n), f)]
        for i, rec in enumerate(records):
            meta = rec["_meta"]
            if meta.get("variant") != "clean" or meta.get("source") == "unknowable":
                dropped.append(meta["id"])
                continue
            rows = []
            for row in rows_for_record(tok, rec):
                q = rec["questions"][row.qid]
                if q["type"] == "choice":
                    assert row.option_keys == list(q["criteria"]), (meta["id"], row.qid)
                assert 0 <= label_index(q) < len(row.option_keys), (meta["id"], row.qid)
                rows.append(
                    {
                        "qid": row.qid,
                        "type": row.qtype,
                        "keys": row.option_keys,
                        "label": label_index(q),
                        "task": q["src"],
                        "ids": row.ids,
                        "opt_positions": row.opt_positions,
                        "decide_position": row.decide_position,
                        "row_tokens": len(row),
                    }
                )
            out.append(
                {
                    "suite": suite,
                    "index": i,
                    "id": meta["id"],
                    "source": meta["source"],
                    "group": meta.get("group_id"),
                    "variant": meta.get("variant"),
                    "rows": rows,
                }
            )
    SWEEP.mkdir(parents=True, exist_ok=True)
    with open(SUBSET_PATH, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_rows = sum(len(r["rows"]) for r in out)
    log(
        f"subset: {len(out)} records, {n_rows} rows, {sum(x['row_tokens'] for r in out for x in r['rows'])} tokens, dropped {dropped} -> {SUBSET_PATH}"
    )
    return out


def pcc(a, b):
    a = a.float().flatten() - a.float().mean()
    b = b.float().flatten() - b.float().mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-8)).item()


def build_engine(device, n_layers):
    from models.autoports.jaredpalmer_kev_9b.tt.engine import KevEngine
    from models.autoports.jaredpalmer_kev_9b.tt.loader import KevModelArgs

    params = inspect.signature(KevEngine.__init__).parameters
    kwargs = {}
    if "args_cls" in params:
        kwargs["args_cls"] = KevModelArgs
    if "max_state_len" in params:
        kwargs["max_state_len"] = 8192
    if "n_layers" in params and n_layers is not None:
        kwargs["n_layers"] = n_layers
    if "traced" in params:
        kwargs["traced"] = False
    log(f"KevEngine signature: {inspect.signature(KevEngine.__init__)}; using {sorted(kwargs)}")
    return KevEngine(device, **kwargs)


def propagation_check(engine):
    import ttnn

    model = engine.model
    mlp = model.layers[0].feed_forward
    attn = next(l.attention for l in model.layers if l.is_full_attention)
    gdn = next(l.attention for l in model.layers if not l.is_full_attention)
    return {
        "mlp.w1.dtype": str(mlp.weights.w1.dtype),
        "mlp.w2.dtype": str(mlp.weights.w2.dtype),
        "mlp.w3.dtype": str(mlp.weights.w3.dtype),
        "mlp.compute_kernel_config.math_fidelity": str(mlp.compute_kernel_config.math_fidelity),
        "mlp.compute_kernel_config.fp32_dest_acc_en": bool(mlp.compute_kernel_config.fp32_dest_acc_en),
        "attn.q_proj.dtype": str(attn.weights.q_proj.dtype),
        "attn.o_proj.dtype": str(attn.weights.o_proj.dtype),
        "attn.compute_kernel_config.math_fidelity": str(attn.compute_kernel_config.math_fidelity),
        "gdn.qkv_proj.dtype": str(gdn.weights.qkv_proj_weight.dtype),
        "gdn.out_proj.dtype": str(gdn.weights.o_proj_weight.dtype),
        "gdn.compute_kernel_config.math_fidelity": str(gdn.compute_kernel_config.math_fidelity),
        "kv_cache.dtype": str(ttnn.bfloat16),
        "weight_cache_path": str(engine.args.weight_cache_path()),
        "n_layers": engine.args.n_layers,
        "traced": bool(engine.traced),
        "matmul_policy": bool(engine.matmul_policy),
    }


def bucket_key(T, chunk_size):
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    return tuple(Qwen36Model._mask_bucket_for(min(chunk_size, T - cs)) for cs in range(0, T, chunk_size))


def run_rows(engine, head, rows, ref_hidden=None, ref_probs=None):
    import torch

    out, seen = [], set()
    for row in rows:
        ids = torch.tensor(row["ids"], dtype=torch.long).unsqueeze(0)
        positions = list(row["opt_positions"]) + [row["decide_position"]]
        key = bucket_key(ids.shape[1], engine.chunk_size)
        warm = key in seen
        seen.add(key)
        t0 = time.perf_counter()
        tt = engine.prefill_hidden(ids, positions)
        dt = time.perf_counter() - t0
        probs = head.probs(tt[-1], tt[:-1]).float()
        rec = {
            "row_key": row.get("row_key"),
            "type": row["type"],
            "row_tokens": int(ids.shape[1]),
            "n_options": len(row["opt_positions"]),
            "probs_tt": probs.tolist(),
            "argmax_tt": int(probs.argmax()),
            "seconds": dt,
            "warm": warm,
            "buckets": list(key),
        }
        if "label" in row:
            rec["label"] = row["label"]
            rec["correct"] = bool(probs.argmax() == row["label"])
        if ref_hidden is not None:
            ref = ref_hidden[row["row_key"]].float()
            rec["position_pcc"] = [pcc(ref[j], tt[j]) for j in range(len(positions))]
            rec["min_pcc"] = min(rec["position_pcc"])
        if ref_probs is not None:
            rq = ref_probs[str(row["record"])]["questions"][row["qid"]]["probabilities"]
            rp = torch.tensor([rq[k] for k in row["keys"]], dtype=torch.float32)
            rec["probs_ref"] = rp.tolist()
            rec["argmax_ref"] = int(rp.argmax())
            rec["argmax_agree"] = bool(probs.argmax() == rp.argmax())
            rec["max_dp"] = (probs - rp).abs().max().item()
        out.append(rec)
        extra = ""
        if "min_pcc" in rec:
            extra += f" min_pcc={rec['min_pcc']:.6f} agree={rec['argmax_agree']} max_dp={rec['max_dp']:.6f}"
        if "correct" in rec:
            extra += f" label={rec['label']} correct={rec['correct']}"
        log(f"row {rec['row_key'] or row.get('qid')} {rec['type']} T={rec['row_tokens']} {dt:.3f}s warm={warm}{extra}")
    return out


def timing(rows):
    warm = [r for r in rows if r["warm"]]
    return {
        "rows": len(rows),
        "mean_row_s": statistics.mean(r["seconds"] for r in rows) if rows else None,
        "warm_rows": len(warm),
        "warm_row_s": statistics.mean(r["seconds"] for r in warm) if warm else None,
        "warm_ms_per_token": 1000 * sum(r["seconds"] for r in warm) / sum(r["row_tokens"] for r in warm)
        if warm
        else None,
        "total_tokens": sum(r["row_tokens"] for r in rows),
        "total_s": sum(r["seconds"] for r in rows),
    }


def summarize_reference(rows):
    mins = [r["min_pcc"] for r in rows]
    max_dps = [r["max_dp"] for r in rows]
    return {
        "rows": len(rows),
        "argmax_agree": sum(r["argmax_agree"] for r in rows),
        "argmax_flips": sum(not r["argmax_agree"] for r in rows),
        "flipped_rows": [r["row_key"] for r in rows if not r["argmax_agree"]],
        "max_dp": max(max_dps),
        "max_dp_row": rows[max_dps.index(max(max_dps))]["row_key"],
        "mean_dp": statistics.mean(max_dps),
        "min_pcc": min(mins),
        "median_pcc": statistics.median(mins),
        "rows_below_0_99": sum(m < 0.99 for m in mins),
        **timing(rows),
    }


def summarize_subset(rows, metrics_mod):
    scored = [{"p": r["probs_tt"], "label": r["label"], "type": r["type"]} for r in rows]
    m = metrics_mod.metrics(scored)
    by_suite = {}
    for suite in sorted({r["suite"] for r in rows}):
        sub = [{"p": r["probs_tt"], "label": r["label"], "type": r["type"]} for r in rows if r["suite"] == suite]
        ms = metrics_mod.metrics(sub)
        by_suite[suite] = {k: ms[k] for k in ("n", "acc", "brier", "ece", "nll")}
    return {
        "rows": m["n"],
        "acc": m["acc"],
        "brier": m["brier"],
        "ece": m["ece"],
        "nll": m["nll"],
        "mean_conf": m["mean_conf"],
        "by_suite": by_suite,
        "metric_source": f"{METRICS_PY}: ece() lines 15-21, brier in metrics() lines 64-79 (sum((p - onehot)^2) per row, mean over rows), at the head temperature (served)",
        **timing(rows),
    }


def write_csv(row):
    SWEEP.mkdir(parents=True, exist_ok=True)
    new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def fmt(x, nd=6):
    return "" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else x)


def run(name, rows_only, device_id, n_layers, matmul_policy=None, down_auto=False, tag=""):
    env = apply_env(name, matmul_policy, down_auto)
    v = VARIANTS[name]
    log(f"variant {name}: {v}; env {env}")
    import torch

    import ttnn
    from models.autoports.jaredpalmer_kev_9b.tt.head import PointerHead

    metrics_mod = load_metrics_module()
    ref_rows = json.load(open(REF / "rows.json"))["rows"]
    ref_hidden = torch.load(REF / "hidden_fp32.pt")
    ref_probs = json.load(open(REF / "probs_fp32.json"))["records"]
    subset = None if rows_only else materialize_subset()
    head = PointerHead(os.environ["KEV_RUN"])
    result = {
        "variant": name,
        "settings": v,
        "env": env,
        "device_id": device_id,
        "device_params": DEVICE_PARAMS,
        "argv": sys.argv,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "head_temperature": head.temperature,
        "status": "started",
    }
    SWEEP.mkdir(parents=True, exist_ok=True)
    out_path = SWEEP / f"{name}{tag}{'_rows' if rows_only else ''}.json"
    device = ttnn.open_device(device_id=device_id, **DEVICE_PARAMS)
    device.enable_program_cache()
    try:
        t0 = time.perf_counter()
        engine = build_engine(device, n_layers)
        result["engine_load_s"] = time.perf_counter() - t0
        result["propagation"] = propagation_check(engine)
        log(f"engine loaded in {result['engine_load_s']:.1f}s; propagation {result['propagation']}")
        ref_out = run_rows(engine, head, ref_rows, ref_hidden, ref_probs)
        result["reference"] = summarize_reference(ref_out)
        result["reference_rows"] = ref_out
        log(f"reference: {json.dumps({k: v for k, v in result['reference'].items() if k != 'flipped_rows'})}")
        if subset is not None:
            flat = [
                {**row, "suite": rec["suite"], "record_id": rec["id"], "row_key": f"{rec['id']}:{row['qid']}"}
                for rec in subset
                for row in rec["rows"]
            ]
            sub_out = run_rows(engine, head, flat)
            for r, src in zip(sub_out, flat):
                r["suite"] = src["suite"]
                r["task"] = src["task"]
            result["subset"] = summarize_subset(sub_out, metrics_mod)
            result["subset_rows"] = sub_out
            log(f"subset: {json.dumps({k: v for k, v in result['subset'].items() if k != 'by_suite'})}")
            log(f"subset by suite: {json.dumps(result['subset']['by_suite'])}")
        result["status"] = "ok"
    except Exception as e:
        result["status"] = f"error: {type(e).__name__}: {str(e).splitlines()[0]}"
        result["error"] = f"{type(e).__name__}: {e}"
        log(result["error"])
        raise
    finally:
        json.dump(result, open(out_path, "w"), indent=1)
        log(f"wrote {out_path}")
        ttnn.close_device(device)
        ref = result.get("reference", {})
        sub = result.get("subset", {})
        write_csv(
            {
                "variant": name + tag,
                "gate_up_dtype": v["gate_up"],
                "down_dtype": v["down"],
                "proj_dtype": v["proj"],
                "fidelity": v["fidelity"],
                "gdn_fp32_state": v["gdn_fp32_state"],
                "matmul_policy": result.get("propagation", {}).get(
                    "matmul_policy", os.environ.get("KEV_MATMUL_POLICY", "1") == "1"
                ),
                "cache_dir": env["TT_CACHE_PATH"],
                "ref_rows": ref.get("rows", ""),
                "argmax_flips": ref.get("argmax_flips", ""),
                "max_dp": fmt(ref.get("max_dp")),
                "mean_dp": fmt(ref.get("mean_dp")),
                "min_pcc": fmt(ref.get("min_pcc")),
                "median_pcc": fmt(ref.get("median_pcc")),
                "ref_mean_row_s": fmt(ref.get("mean_row_s"), 4),
                "subset_rows": sub.get("rows", ""),
                "acc": fmt(sub.get("acc")),
                "brier": fmt(sub.get("brier")),
                "ece": fmt(sub.get("ece")),
                "nll": fmt(sub.get("nll")),
                "mean_row_s": fmt(sub.get("mean_row_s"), 4),
                "warm_row_s": fmt(sub.get("warm_row_s"), 4),
                "warm_ms_per_token": fmt(sub.get("warm_ms_per_token"), 4),
                "warm_rows": sub.get("warm_rows", ""),
                "engine_load_s": fmt(result.get("engine_load_s"), 1),
                "status": result["status"],
                "timestamp": result["timestamp"],
                "json": str(out_path),
            }
        )
        log(f"appended {CSV_PATH}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=sorted(VARIANTS))
    p.add_argument("--rows-only", action="store_true")
    p.add_argument("--materialize-only", action="store_true")
    p.add_argument("--force-subset", action="store_true")
    p.add_argument("--device-id", type=int, default=2)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--print-env", action="store_true")
    p.add_argument("--matmul-policy", choices=["0", "1"], default=None)
    p.add_argument("--down-auto", action="store_true")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    if a.materialize_only:
        materialize_subset(force=a.force_subset)
        return
    if a.variant is None:
        p.error("--variant is required")
    if a.print_env:
        env = variant_env(a.variant)
        if a.matmul_policy is not None:
            env["KEV_MATMUL_POLICY"] = a.matmul_policy
        print(" ".join(f"{k}={v}" for k, v in env.items()))
        return
    run(a.variant, a.rows_only, a.device_id, a.n_layers, a.matmul_policy, a.down_auto, a.tag)


if __name__ == "__main__":
    main()
