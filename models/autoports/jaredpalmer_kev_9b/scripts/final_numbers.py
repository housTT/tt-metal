import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_eval import DROP_IDS, SUITES, audited, knowable, scored

R = Path("/home/hous/dev/kev/reports")
AP = Path(__file__).resolve().parent.parent
TT_METAL = AP.parent.parent.parent
SHORT, LONG = "6 questions, short state", "5 questions, 2,200-token state"
TWO, MID = "2 questions, short state", "5 questions, 370-token state"
BENCH = {
    "p150x1_whole": "1 worker, KEV_FANOUT=0, full bench, idle host",
    "p150x1_whole_cpujob": "1 worker, KEV_FANOUT=0, full bench while the CPU fp32 parity job ran (host contention; not a card number)",
    "p150x1_whole_latency_idle": "1 worker, KEV_FANOUT=0, latency section only, idle host",
    "p150x2_whole": "2 workers, KEV_FANOUT=0, --quick",
    "p150x4_whole": "4 workers, KEV_FANOUT=0, full bench, before the readback fix (GIL-bound)",
    "p150x4_whole_fix_quick": "4 workers, KEV_FANOUT=0, --quick, with the GIL-releasing readback",
    "p150x4_whole_fix": "4 workers, KEV_FANOUT=0, full bench, with the GIL-releasing readback",
    "p150x4_fanout": "4 workers, KEV_FANOUT=1, full bench",
    "p150_stage4r": "stage 4 final, 1 chip (mesh 1x1, chip 0), --quick",
}
CARD_ROWS = {"P150 (1 chip)": "p150x1_whole", "P150 x4 (data parallel)": "p150x4_fanout"}
GIL = {
    "method": "sudo py-spy record --gil --threads --nonblocking --rate 200 on the 4-worker server during the 64-client short-state run; per-request latency_ms from the server log",
    "before_fix": {
        "profile": str(R / "bench" / "gil_A4" / "gil_012352.raw"),
        "gil_held_samples_of_6000": 5969,
        "gil_held_fraction": 0.995,
        "fraction_in_gather_to_torch": 0.986,
        "on_cpu_samples_of_6000": 182,
        "model_latency_ms_median": {"1 worker busy": 607, "2 workers busy": 1058, "4 workers busy": 1522},
        "requests_per_s_64_clients_short": {"1 worker": 1.6, "2 workers": 2.0, "4 workers": 2.5},
    },
    "fix": "tt/engine.py KevEngine._gather: ttnn.to_torch(B['rows'].cpu()); Tensor.cpu is bound with gil_scoped_release, the module-level from_device used by to_torch is not",
    "after_fix": {
        "profile": str(R / "bench" / "gil_A4fix" / "gil_014507.raw"),
        "gil_held_samples_of_4000": 58,
        "gil_held_fraction": 0.015,
        "model_latency_ms_median_4_workers_busy": 607,
        "requests_per_s_64_clients_short_4_workers": 6.6,
    },
}
MODEL_CARD = {
    "hard-v1": {"test_acc": 0.834, "test_ece": 0.054},
    "devtools-v1": {"test_acc": 0.791, "test_ece": 0.098},
    "documents-v1": {"test_acc": 0.900, "test_ece": 0.017},
    "hard-v1 + devtools-v1 audited": {"development_acc": 0.821, "test_acc": 0.822},
    "breadth-v1": "not available (private mirror)",
}


def read(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def git(*args):
    return subprocess.run(["git", "-C", str(TT_METAL), *args], capture_output=True, text=True).stdout.strip()


def bench_summary(rep):
    lat, thr = rep["latency"], rep.get("throughput", {})

    def level(name, c):
        row = thr.get(f"{name} @ {c} clients")
        return {k: row[k] for k in ("requests_per_s", "p50_ms", "p99_ms")} if row else None

    return {
        "label": rep["label"],
        "method": "quick (32 short / 32 decision-v7 / 8 long requests per level)"
        if rep.get("quick")
        else "full (256 / 256 / 64 requests per level)",
        "reps": rep["reps"],
        "device": rep["served"]["device"],
        "fanout": rep["served"]["dispatch"]["fanout"],
        "latency_ms": {
            case: {k: lat[case][k] for k in ("tokens", "new_ms", "cached_ms")} for case in (TWO, SHORT, MID, LONG)
        },
        "throughput": {
            name: {c: level(name, c) for c in rep["levels"]}
            for name in ("6 questions, new short state", "decision-v7 development", LONG)
        }
        if thr
        else None,
        "card_row": rep.get("card_row"),
    }


def card_entry(rep):
    s, l = rep["latency"][SHORT], rep["latency"][LONG]
    thr = rep.get("throughput", {}).get("6 questions, new short state @ 64 clients", {})
    return {
        "short_new_ms": s["new_ms"],
        "short_cached_ms": s["cached_ms"],
        "long_new_ms": l["new_ms"],
        "long_cached_ms": l["cached_ms"],
        "requests_per_s_64_clients": thr.get("requests_per_s"),
        "p50_ms_64_clients": thr.get("p50_ms"),
        "p99_ms_64_clients": thr.get("p99_ms"),
    }


def parity16():
    out = {}
    for name in (
        "stage4r_parity",
        "stage5_parity_chip0",
        "stage5_parity_chip1",
        "stage5_parity_chip2",
        "stage5_parity_chip3",
        "stage5_parity_x4",
    ):
        rep = read(R / f"{name}.json")
        if rep:
            out[name] = {
                "device": rep["served_model"]["device"],
                "fanout": rep["served_model"]["dispatch"]["fanout"],
                "vs_fp32": rep["summary_vs"]["fp32"],
                "vs_bf16": rep["summary_vs"]["bf16"],
                "revisits": {k: rep["revisits"][k] for k in ("total", "answers_equal_to_first_pass")},
                "latency_ms": rep["latency_ms"],
            }
    cmp = read(R / "stage5_parity_compare.json")
    if cmp:
        out["compare"] = {
            "base": cmp["base"],
            "questions": cmp["questions"],
            "all_identical": cmp["all_identical"],
            "runs": {
                p: {"identical": r["identical"], "differing": len(r["differing"])} for p, r in cmp["runs"].items()
            },
        }
    return out


def evals():
    out, pooled = {}, {}
    for split in ("development", "test"):
        for suite in SUITES:
            rep = read(R / "eval" / suite / split / "report.json")
            if rep is None:
                out[f"{suite}/{split}"] = None
                continue
            rows = read(R / "eval" / suite / split / "rows.json")
            c = {**rep["clean"], **scored(knowable(rows))} if rows else rep["clean"]
            out[f"{suite}/{split}"] = {
                "n": c["n"],
                "acc": c["acc"],
                "brier": c["brier"],
                "ece": c["ece"],
                "nll": c["nll"],
                "rows_dropped": rep["clean"]["n"] - c["n"],
                "latency_p50_ms": rep["latency_ms"]["median"],
                "latency_p95_ms": rep["latency_ms"]["p95"],
                "coverage": rep["coverage"],
                "concurrency": rep["remote"]["concurrency"],
            }
            if suite in ("hard-v1", "devtools-v1") and rows:
                pooled.setdefault(split, []).extend(audited(knowable(rows)))
        if split in pooled:
            out[f"hard-v1 + devtools-v1 audited/{split}"] = scored(pooled[split])
    out["drop_ids"] = sorted(DROP_IDS)
    return out


def main():
    benches = {name: read(R / "bench" / name / "report.json") for name in BENCH}
    precision = read(AP / "doc" / "datatype_sweep" / "selected_precision_config.json")
    x4 = benches.get("p150x4_fanout")
    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "commits": {
            "tt_metal_head": git("rev-parse", "HEAD"),
            "tt_metal_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "tt_metal_base": git("merge-base", "HEAD", "origin/main"),
            "tt_metal_uncommitted": git("diff", "--name-only").splitlines(),
            "kev": subprocess.run(
                ["git", "-C", "/home/hous/dev/kev/kev", "rev-parse", "HEAD"], capture_output=True, text=True
            ).stdout.strip(),
        },
        "precision": {
            "config_id": precision["config_id"],
            "runtime_flags": precision["runtime_flags"],
            "weight_cache": precision["default_construction"]["weight_cache_path"],
            "compute_fidelities": precision["compute_fidelities"],
        }
        if precision
        else None,
        "server": {
            "mesh": "ttnn.open_mesh_device(MeshShape(2,2)).create_submeshes(MeshShape(1,1)); submesh i -> physical chip [1],[0],[2],[3]",
            "workers": x4["served"]["workers"] if x4 else None,
            "dispatch": {k: v for k, v in x4["served"]["dispatch"].items() if k != "fanout_requests"} if x4 else None,
            "prefix_cache_slots_per_worker": 8,
            "traced": True,
            "max_state_tokens": 65536,
        },
        "card": {
            row: {**card_entry(benches[name]), "source": str(R / "bench" / name / "report.json"), "method": BENCH[name]}
            for row, name in CARD_ROWS.items()
            if benches.get(name)
        },
        "bench": {name: {**bench_summary(rep), "note": BENCH[name]} for name, rep in benches.items() if rep},
        "parity_16_records": parity16(),
        "parity_64_records": read(R / "parity64" / "compare.json"),
        "gil": GIL,
        "eval": evals(),
        "model_card": MODEL_CARD,
    }
    (R / "final_numbers.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("card", "eval")}, indent=1))


if __name__ == "__main__":
    main()
