"""Write doc/datatype_sweep/selected_precision_config.json from a candidate config plus the selection record.

Usage: write_selected.py <config id>
The file keeps every field build_generator consumes (weight_groups, layer_exceptions, norm, sdpa, prefill,
activation/residual dtype, ccl dtype, kv_cache_dtype, embedding dtype, lm_head, final_norm, logits, sampling) and
adds a `selection` block with the acceptance rule, the measured numbers and the evidence paths.
"""
import json
import subprocess
import sys
from pathlib import Path

D = Path("models/autoports/qwen_qwen3_8b/doc/datatype_sweep")
cid = sys.argv[1]
cfg = json.loads((D / "configs" / f"{cid}.json").read_text())
results = json.loads((D / "sweep_results.json").read_text())
row = next(r for r in results["candidates"] if r["config_id"] == cid)
baseline = next(r for r in results["candidates"] if r["config_id"] == "c0_stage7_baseline")
cfg["selection"] = {
    "selected_from": f"datatype_sweep/configs/{cid}.json",
    "stage": "datatype_sweep (stage 8)",
    "commit_of_sweep": row["checkout_git_sha"],
    "acceptance": results["acceptance"],
    "ranking_metric": results["ranking_metric"],
    "teacher_forcing": {
        k: row["teacher_forcing"][k]
        for k in ("top1", "top5", "top100", "ttft_ms", "decode_t_s_u", "timing_regime", "log")
    },
    "prefill_check": row["prefill_check"],
    "robustness_references": {
        k: (
            {
                "prefill_check": v["prefill_check"],
                "teacher_forcing": {kk: v["teacher_forcing"][kk] for kk in ("top1", "top5", "top100")},
                "pass": v["pass"],
            }
            if "prefill_check" in v
            else {kk: v[kk] for kk in ("reference", "top1", "top5", "pcc_mean", "pcc_min", "artifact", "pass")}
        )
        for k, v in row["robustness_references"].items()
    },
    "baseline_teacher_forcing": {
        k: baseline["teacher_forcing"][k] for k in ("top1", "top5", "top100", "ttft_ms", "decode_t_s_u")
    },
    "hardware": row["hardware"],
    "mesh": row["mesh"],
    "restore_baseline": "QWEN3_PRECISION_CONFIG=models/autoports/qwen_qwen3_8b/doc/datatype_sweep/configs/c0_stage7_baseline.json (or delete this file: build_generator then uses the built-in stage-7 baseline)",
    "consumed_by": "models/autoports/qwen_qwen3_8b/tt/generator.py::build_generator via tt/precision.py (default path when QWEN3_PRECISION_CONFIG is unset); propagation evidence in propagation_check_selected_36.json",
}
cfg["runtime_flags"] = {
    "QWEN3_PRECISION_CONFIG": "path of a precision config to use instead of this file",
    "QWEN3_LM_HEAD_DECODE_FIDELITY, QWEN3_LM_HEAD_PREFILL_FIDELITY, QWEN3_LM_HEAD_DTYPE, QWEN3_PREFILL_LOGITS_DTYPE": "terminal-policy overrides applied on top of the config",
    "QWEN3_PREFILL_KERNEL_TABLE, QWEN3_PREFILL_SHARDED_NORM_MAX_ROWS": "prefill kernel overrides applied on top of the config (not dtype policy)",
}
(D / "selected_precision_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
print(
    json.dumps(cfg["selection"]["teacher_forcing"]),
    subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
)
