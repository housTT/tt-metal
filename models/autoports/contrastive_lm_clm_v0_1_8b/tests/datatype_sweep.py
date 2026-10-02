# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import argparse
import csv
import json
import os
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
AUTOPORT = os.path.dirname(HERE)
GATE = {
    "cosine_mean_min": 0.99,
    "cosine_min_min": 0.97,
    "head_cos_min_min": 0.95,
    "decision_agreement_margin_0p10_min": 0.98,
}
TIE_FRACTION = 0.01


STOCK_SPECS = {
    "accuracy": {
        "TensorPrecision": {"WQKV": "BF16", "KV_CACHE": "BF16", "WO": "BF16", "FF1_FF3": "BFP8", "FF2": "BFP8"},
        "OpFidelity": {
            "LI_QKV_PREFILL": "HIFI4",
            "LI_O_PREFILL": "HIFI4",
            "SDPA_PREFILL": "HIFI4",
            "LI_QKV_DECODE": "HIFI4",
            "LI_O_DECODE": "HIFI4",
            "SDPA_DECODE": "HIFI4",
            "LI_FF1_FF3": "HIFI2_FP16",
            "LI_FF2": "HIFI2_FP16",
        },
    },
    "performance": {
        "TensorPrecision": {"WQKV": "BFP8", "KV_CACHE": "BFP8", "WO": "BFP8", "FF1_FF3": "BFP4", "FF2": "BFP8"},
        "OpFidelity": {
            "LI_QKV_PREFILL": "HIFI2",
            "LI_O_PREFILL": "HIFI2",
            "SDPA_PREFILL": "HIFI4",
            "LI_FF1_FF3": "LOFI",
            "LI_FF2": "HIFI2_FP16",
        },
    },
}
DEFAULTS = {
    "TensorPrecision": {"WQKV": "BFP8", "KV_CACHE": "BFP8", "WO": "BFP8", "FF1_FF3": "BFP8", "FF2": "BFP8"},
    "OpFidelity": {
        "LI_QKV_PREFILL": "HIFI2",
        "LI_O_PREFILL": "HIFI2",
        "SDPA_PREFILL": "HIFI4",
        "LI_FF1_FF3": "HIFI2_FP16",
        "LI_FF2": "HIFI2_FP16",
    },
}


def policy_spec(name):
    name = name[: -len("_pc")] if name.endswith("_pc") else name
    if name in STOCK_SPECS:
        return STOCK_SPECS[name]
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("clm_encoder", os.path.join(AUTOPORT, "tt", "encoder.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        custom = mod.CUSTOM_POLICIES.get(name, {})
    except Exception:
        custom = {}
    merged = {k: dict(DEFAULTS[k]) for k in DEFAULTS}
    for k in merged:
        merged[k].update(custom.get(k, {}))
    return merged


def load(path):
    if not os.path.exists(path):
        return None
    data = json.load(open(path))
    if isinstance(data, dict):
        data["_path"] = path
    return data


WORKLOAD_CELLS = ((128, 1), (128, 8), (512, 8), (1024, 1), (2048, 1))


def latency_rows(bench):
    out = {}
    for r in bench["rows"]:
        out[(r["tokens"], r["batch"])] = r["p50_ms"]
    return out


def exact_bucket_rows(bench):
    return {(r["tokens"], r["batch"]): r["p50_ms"] for r in bench["rows"] if r["padded"] == r["tokens"]}


def workload_ms(bench):
    cells = exact_bucket_rows(bench)
    if all(c in cells for c in WORKLOAD_CELLS):
        return sum(cells[c] for c in WORKLOAD_CELLS)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidates",
        default="bf16_all,accuracy,accuracy_lofi_mlp,bfp8_attn,bfp8_attn_hifi2,bfp8_lofi_mlp,performance,accuracy_lofi_mlp_pc,accuracy_pc",
    )
    ap.add_argument("--outdir", default=os.path.join(AUTOPORT, "doc", "datatype_sweep"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rows = []
    for name in a.candidates.split(","):
        fid = (
            load(os.path.join(a.outdir, f"fidelity_{name}.json"))
            or load(os.path.join(AUTOPORT, "doc", "full_model", f"fidelity_{name}.json"))
            or load(os.path.join(AUTOPORT, "doc", "optimized_full_model", f"fidelity_{name}.json"))
        )
        bench = (
            load(os.path.join(a.outdir, f"bench_{name}_buckets5.json"))
            or load(os.path.join(AUTOPORT, "doc", "optimized_full_model", f"bench_{name}_buckets5.json"))
            or load(os.path.join(a.outdir, f"bench_{name}.json"))
            or load(os.path.join(AUTOPORT, "doc", "optimized_full_model", f"bench_{name}.json"))
        )
        if fid is None or bench is None:
            infeasible = load(os.path.join(a.outdir, f"infeasible_{name}.json"))
            if infeasible:
                rows.append(
                    {
                        "policy": name,
                        "status": "infeasible",
                        "error": infeasible.get("error"),
                        "log": infeasible.get("log"),
                    }
                )
            else:
                rows.append(
                    {"policy": name, "status": "missing", "fidelity": fid is not None, "bench": bench is not None}
                )
            continue
        lat = latency_rows(bench)
        head_min = min(v["min"] for v in fid["head_projection_cosine"].values())
        agr = load(os.path.join(a.outdir, f"agreement_{name}.json")) or load(
            os.path.join(AUTOPORT, "doc", "optimized_full_model", f"agreement_{name}.json")
        )
        agree_conf = agr["argmax_agreement_margin_ge_0p10"] if agr else None
        agree_all = agr["argmax_agreement"] if agr else None
        passed = (
            fid["cosine_single_vs_hf"]["mean"] >= GATE["cosine_mean_min"]
            and fid["cosine_single_vs_hf"]["min"] >= GATE["cosine_min_min"]
            and head_min >= GATE["head_cos_min_min"]
            and fid["nan_count"] == 0
            and agree_conf is not None
            and agree_conf >= GATE["decision_agreement_margin_0p10_min"]
        )
        rows.append(
            {
                "policy": name,
                "program_config_overrides": name.endswith("_pc"),
                "status": "pass" if passed else "fail",
                "cosine_mean": fid["cosine_single_vs_hf"]["mean"],
                "cosine_min": fid["cosine_single_vs_hf"]["min"],
                "cosine_p05": fid["cosine_single_vs_hf"]["p05"],
                "head_cos_state_mean": fid["head_projection_cosine"]["state"]["mean"],
                "head_cos_min": head_min,
                "batched_vs_single_min": fid["cosine_single_vs_batched"]["min"],
                "decision_agreement": agree_all,
                "decision_agreement_margin_0p10": agree_conf,
                "workload_ms": workload_ms(bench),
                "bench_file": os.path.relpath(bench.get("_path", ""), AUTOPORT) if bench.get("_path") else None,
                "lat_128_b1_ms": lat.get((128, 1)),
                "lat_256_b1_ms": exact_bucket_rows(bench).get((256, 1)),
                "lat_512_b1_ms": exact_bucket_rows(bench).get((512, 1)),
                "lat_512_b8_ms": exact_bucket_rows(bench).get((512, 8)),
                "lat_128_b8_ms": lat.get((128, 8)),
                "lat_1024_b1_ms": lat.get((1024, 1)),
                "lat_2048_b1_ms": lat.get((2048, 1)),
                "tokens_per_s_1024_b8": next(
                    (r["tokens_per_s"] for r in bench["rows"] if r["tokens"] == 1024 and r["batch"] == 8), None
                ),
                "fidelity_file": os.path.relpath(fid.get("_path", ""), AUTOPORT) if fid.get("_path") else None,
            }
        )
    passing = [r for r in rows if r.get("status") == "pass" and r.get("lat_128_b1_ms") is not None]
    selected = None
    selection_metric = None
    if passing:
        if all(r.get("workload_ms") is not None for r in passing):
            selection_metric = "workload_ms"
        else:
            selection_metric = "lat_128_b1_ms"
        fastest = min(passing, key=lambda r: r[selection_metric])
        near = [r for r in passing if r[selection_metric] <= fastest[selection_metric] * (1.0 + TIE_FRACTION)]
        selected = max(near, key=lambda r: r["cosine_min"])
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gate": GATE,
        "metric_mapping": "top-1/top-5 token accuracy replaced by embedding cosine vs HF fp32 reference, head-projection cosine, and Typed Decisions argmax agreement with the fp32 reference on the 40-case subset (decisions whose reference top-2 margin is at least 0.10)",
        "selection_rule": f"fastest passing policy by the served-workload latency sum over the cells {list(WORKLOAD_CELLS)} (tokens, batch) measured on the five-bucket encoder; falls back to 128-token batch-1 latency when a passing policy lacks a five-bucket bench; policies within {TIE_FRACTION:.0%} of the fastest are a tie and the highest minimum cosine wins (plan amendment 2026 Oct 2)",
        "selection_metric_used": selection_metric,
        "rows": rows,
        "selected": selected["policy"] if selected else None,
    }
    with open(os.path.join(a.outdir, "sweep_results.json"), "w") as f:
        json.dump(result, f, indent=1)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(a.outdir, "sweep_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    if selected:
        with open(os.path.join(a.outdir, "selected_precision_config.json"), "w") as f:
            json.dump(
                {
                    "precision": selected["policy"],
                    "env": {"CLM_PRECISION": selected["policy"]},
                    "gate": GATE,
                    "gate_vectors": "single-text vectors (fidelity_<policy>_tt_single.npy): the serving path for a request with one new text; batched vectors differ by batch composition (tt-metal 47238) and move the confident-decision count by up to five",
                    "gate_resolution": "one decision = 0.53 points on the 188 confident decisions",
                    "selected_row": selected,
                    "policy_spec": policy_spec(selected["policy"]),
                    "activation_dtype": "bf16 (TensorGroup.ACTIVATION unset; residual stream bf16 interleaved DRAM)",
                    "layer_exceptions": "none: DecodersPrecision applies the same configuration to all 36 decoders",
                    "ccl_dtype": "none on the 1x1 mesh; stock all-gather dtype on the 1x4 profile",
                    "weight_dtype_passed_to_create_tt_model": "ttnn.bfloat8_b (per-group dtypes above override it for WQKV, WO, KV_CACHE)",
                    "final_norm_and_heads": "host fp32: last-token pick and RMSNorm (eps 1e-6) in TtQwen3Encoder._pool_and_norm; CLM heads in torch fp32 (clm/heads.py)",
                    "runtime_flags": {"TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES": "0"},
                    "consumption_evidence": "models/tt_transformers/tt/attention.py lines 116 to 150 (wqkv, wo, kv dtypes from the policy), mlp.py lines 91 to 95 and 340 to 360 (MLP fidelity), doc/functional_decoder/tracy/layer0/prefill_perf_report.csv rows 15/39/63/87 (QKV HiFi4 BF16 x BF16), 28/52/76/100 (WO HiFi4), 31/32/34 (w1/w3/w2 HiFi2 BFP8)",
                },
                f,
                indent=1,
            )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for metric, fname, gate in (
            ("cosine_mean", "cosine_mean_perf_pareto.png", GATE["cosine_mean_min"]),
            ("cosine_min", "cosine_min_perf_pareto.png", GATE["cosine_min_min"]),
        ):
            fig, ax = plt.subplots(figsize=(6, 4))
            for r in rows:
                if r.get("status") in ("missing", "infeasible"):
                    continue
                color = "red" if selected and r["policy"] == selected["policy"] else "tab:blue"
                ax.scatter(r["lat_128_b1_ms"], r[metric], color=color, s=60)
                ax.annotate(
                    r["policy"], (r["lat_128_b1_ms"], r[metric]), textcoords="offset points", xytext=(5, 5), fontsize=8
                )
            ax.axhline(gate, linestyle=":", color="gray")
            ax.set_xlabel("single-text latency, 128-token bucket (ms, p50)")
            ax.set_ylabel(metric + " vs HF fp32")
            ax.set_title("CLM-v0.1-8B encoder on p150: precision policy sweep")
            fig.tight_layout()
            fig.savefig(os.path.join(a.outdir, fname), dpi=120)
            plt.close(fig)
    except Exception as exc:
        result["plot_error"] = str(exc)
        with open(os.path.join(a.outdir, "sweep_results.json"), "w") as f:
            json.dump(result, f, indent=1)
    print(
        "SWEEP",
        json.dumps(
            {
                "selected": result["selected"],
                "rows": [
                    {
                        k: r.get(k)
                        for k in (
                            "policy",
                            "status",
                            "cosine_mean",
                            "cosine_min",
                            "head_cos_min",
                            "lat_128_b1_ms",
                            "lat_1024_b1_ms",
                        )
                    }
                    for r in rows
                ],
            }
        ),
    )


if __name__ == "__main__":
    main()
