#!/usr/bin/env python3
"""Count fatal patterns in the two reduced-model watcher logs (pytest battery on chip 1, penalty probe on chip 2) and write doc/optimized_vllm/watcher/summary.json."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path("models/autoports/qwen_qwen3_8b/doc/optimized_vllm")
PATTERNS = (
    "fatal",
    "exception",
    "assert",
    "overflow",
    "bad noc",
    "hung",
    "error",
    "stack overflow",
    "invalid",
    "corrupt",
)
FATAL = ("fatal", "assert", "bad noc", "hung", "stack overflow", "corrupt")
RUNS = {
    "pytest_batched_prefill_battery": {
        "chip": "1",
        "command": "TT_VISIBLE_DEVICES=1 TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=doc/optimized_vllm/watcher/pytest QWEN3_FULL_MODEL_LAYERS=2 QWEN3_EVIDENCE_SUFFIX=watcher python -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_full_model.py -k 'test_batched_prefill_equals_solo or test_batch_decode or test_split_sampling_trace' --timeout 1800 -p no:cacheprovider",
        "watcher_log": ROOT / "watcher" / "pytest" / "generated" / "watcher" / "watcher.log",
        "console_log": ROOT / "logs" / "pytest_watcher_reduced_chip1.log",
        "result_re": r"(\d+ passed.*?)\n",
    },
    "batched_prefill_probe_22_shapes": {
        "chip": "1",
        "command": "TT_VISIBLE_DEVICES=1 TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=doc/optimized_vllm/watcher/probe python doc/optimized_vllm/logs/batched_prefill_probe.py doc/optimized_vllm/logs/batched_prefill_probe_L2_watcher_chip1.json --layers 2 --reps 1",
        "watcher_log": ROOT / "watcher" / "probe" / "generated" / "watcher" / "watcher.log",
        "console_log": ROOT / "logs" / "batched_prefill_probe_watcher_chip1.log",
        "result_re": r"(pair 8x512: .*?)\n",
    },
    "penalty_probe": {
        "chip": "2",
        "command": "TT_VISIBLE_DEVICES=2 TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=doc/optimized_vllm/watcher/penalty python doc/optimized_vllm/logs/penalty_probe.py doc/optimized_vllm/logs/penalty_probe_L2_watcher_chip2.json --layers 2",
        "watcher_log": ROOT / "watcher" / "penalty" / "generated" / "watcher" / "watcher.log",
        "console_log": ROOT / "logs" / "penalty_probe_watcher_chip2.log",
        "result_re": r"(\"program_cache_flat_across_cases\": \w+)",
    },
}


def main() -> None:
    summary = {
        "model": "reduced full model: two real decoder layers, real embedding, final norm, block-sharded prefill norms, masked LM head, split sampling, decode and sampling traces, the 22 batched-prefill warmup shapes, penalties-on sampler variants",
        "paths_covered": [
            "pytest battery (reduced fixture, warmup lengths 32 to 256): batched shapes 2x32, 4x32, 8x32, 2x64, 4x64, 2x128 only (at most 256 dense rows, which run the stage10 solo dense kernels; the batched kernel table starts at 1024 rows), the mixed and reversed cases with batched groups beside single-row rows, uniform_4/8/32 on the single-row path (no prepared shape for them in this fixture), the stage10 1D O/down kernels at 32 to 256 rows, batched prefill followed by traced batch decode (buckets 8 and 32), split sampling traces",
            "22-shape probe (full warmup buckets 32 to 4096 on 2 layers): every prepared batched shape 2x32 .. 32x128, 16x256, 8x512, so the batched kernel table at 1024, 2048 and 4096 dense rows (2D/16 pieces for qkv, o, up, down; 1D (110, 8, 1x4) 512-row pieces for gate), batched attention and fill at 2 to 32 rows and buckets 32 to 512 (nlp_create_qkv_heads on [rows, 1, S, W], batched causal SDPA, paged_fill_cache with batch_idx_tensor over scratch-redirected tables, nlp_concat_heads), the serial per-row arm and the solo arm of every shape, reordered rows",
            "penalty probe: penalties-on sampler precompile, capture and replay; prompt-mask and output-count resets; seeded eager penalised sampling",
        ],
        "runs": {},
    }
    total_fatal = 0
    for name, run in RUNS.items():
        text = run["watcher_log"].read_text(errors="replace") if run["watcher_log"].exists() else ""
        counts = {p: len(re.findall(p, text, flags=re.IGNORECASE)) for p in PATTERNS}
        console = run["console_log"].read_text(errors="replace") if run["console_log"].exists() else ""
        found = re.findall(run["result_re"], console)
        fatal = sum(counts[p] for p in FATAL)
        total_fatal += fatal
        summary["runs"][name] = {
            "chip": run["chip"],
            "command": run["command"],
            "result": found[-1] if found else "see console log",
            "console_log": str(run["console_log"].relative_to(ROOT)),
            "watcher_log": str(run["watcher_log"].relative_to(ROOT)) if run["watcher_log"].exists() else None,
            "watcher_log_bytes": len(text),
            "dumps_completed": len(re.findall(r"Dump #\d+", text)),
            "pattern_counts": counts,
            "fatal_pattern_matches": fatal,
        }
    summary["fatal_pattern_matches_total"] = total_fatal
    summary["evidence_json"] = sorted(
        p.name
        for p in (ROOT.parent / "full_model").glob("*.watcher.json")
        if p.name.startswith(("batched_prefill_equals_solo_", "batch_decode_", "split_sampling_trace"))
    )
    summary[
        "evidence_json_note"
    ] = "this stage's watcher artifacts; host_work_counters, multi_row_prefill and tracked_replay_workload .watcher.json in the same directory are stage 7's"
    probe = ROOT / "logs" / "batched_prefill_probe_L2_watcher_chip1.json"
    if probe.exists():
        data = json.loads(probe.read_text())
        pairs = data["pairs"]
        summary["runs"]["batched_prefill_probe_22_shapes"]["probe_json"] = str(probe.relative_to(ROOT))
        summary["runs"]["batched_prefill_probe_22_shapes"]["shapes"] = [f"{p['rows']}x{p['seq_len']}" for p in pairs]
        summary["runs"]["batched_prefill_probe_22_shapes"]["all_logits_bit_equal"] = all(
            p["all_logits_bit_equal"] for p in pairs
        )
        summary["runs"]["batched_prefill_probe_22_shapes"]["all_kv_bit_equal"] = all(
            p["all_kv_bit_equal"] for p in pairs
        )
        summary["runs"]["batched_prefill_probe_22_shapes"]["program_cache"] = sorted(
            {(p["program_cache_before"], p["program_cache_after"]) for p in pairs}
        )
    (ROOT / "watcher" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "paths_covered"}, indent=1))


if __name__ == "__main__":
    main()
