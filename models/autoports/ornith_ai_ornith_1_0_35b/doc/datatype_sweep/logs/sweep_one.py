# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""One full-model datatype-sweep candidate: accuracy and traced teacher-forcing decode, in one build.

`$datatype-sweep`'s selection metric is **full-model** top-1/top-5 against the readiness reference,
ranked by **trace-verified teacher-forcing decode t/s/u**. Both come from the shared readiness
machinery: this driver calls `run_prefill_check`'s and `run_teacher_forcing`'s own per-entry
functions (`_run_one_entry_prefill` / `_run_one_entry`) against **one** generator instead of letting
each runner build its own, because a 40-layer build is ~3 minutes and the two checks are otherwise
identical to running the two CLIs back to back. `--verify-against-runners` runs the two official
programmatic entry points instead, which is the control that this shortcut reports the same numbers.

Teacher forcing is repeated (`--tf-repeats`, default 4) because one 99-token decode window is 2.4 s
and the ranking is on ~1 % differences. Every repeat gets a fresh `TokenAccuracy`, so each repeat's
accuracy is independently computed and they must agree. The reported decode t/s/u is the best **warm**
repeat - one with no trace re-capture inside its own timed window; the first repeat after a build is
normally not warm, because prefill compiles that prompt length's programs and the traces are
re-captured before the first replay, which costs ~9 %. Best-of-warm is the same estimator
`bench_full_model.py` uses for its token-out row.

    python .../doc/datatype_sweep/logs/sweep_one.py --config .../candidates/C03.json
    python .../doc/datatype_sweep/logs/sweep_one.py --policy optimized --verify-against-runners
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger

from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import close_ornith_mesh, open_ornith_mesh
from models.autoports.ornith_ai_ornith_1_0_35b.tt.precision_config import policy_to_dict, resolve_policy

MODEL_DIR = Path("models/autoports/ornith_ai_ornith_1_0_35b")
SWEEP_DIR = MODEL_DIR / "doc" / "datatype_sweep"
DEFAULT_REFERENCE = MODEL_DIR / "readiness_aime24_chat.refpt"

#: The acceptance bar. `$datatype-sweep`'s defaults, with top-100 held at the readiness expectation
#: the optimized full-model stage delivered (1.000 on both gates).
GATE = {"top1": 0.90, "top5": 0.98, "top100": 1.00}


def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:  # pragma: no cover - provenance only
        return "unknown"


def _hardware(mesh) -> dict:
    try:
        boards = subprocess.check_output(["tt-smi", "-ls", "--local"], text=True, timeout=60)
        board_type = "Blackhole p300c" if "p300c" in boards else "unknown"
    except Exception:  # pragma: no cover - provenance only
        board_type = "unknown"
    return {
        "board": board_type,
        "num_devices": mesh.get_num_devices(),
        "mesh_shape": list(mesh.shape),
        "arch": str(mesh.arch()),
        "host": platform.node(),
    }


def _trace_evidence(gen, perf) -> dict:
    """What proves the measured decode window replayed a captured trace.

    `generate(enable_trace=True)` is the only path that reaches `_decode_step_traced`; the eager path
    is a different function and never captures. So the evidence is: the trace id exists, capture
    happened before the timed window (`trace_recaptures` does not move across it), and the loop is
    the serial teacher-forcing one by construction (`pipelined_readback` False, one synchronize per
    token).
    """
    return {
        "enable_trace": True,
        "trace_id_present": gen._trace_id is not None,
        "sampling_trace_ready": bool(getattr(gen, "_sampling_trace_ready", False)),
        "trace_recaptures_total": gen.trace_recaptures,
        "recaptures_inside_timed_window": 0,
        "pipelined_readback": perf["pipelined_readback"],
        "teacher_forcing": perf["teacher_forcing"],
        "decode_calls": perf["counters"]["decode_calls"],
        "decode_syncs": perf["counters"]["decode_syncs"],
        "token_refreshes": perf["counters"]["token_refreshes"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="path to a JSON precision config")
    ap.add_argument("--policy", default=None, help="registered policy name (alternative to --config)")
    ap.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    ap.add_argument("--cache-context", type=int, default=8192)
    ap.add_argument("--tf-repeats", type=int, default=4)
    ap.add_argument("--output", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument(
        "--verify-against-runners",
        action="store_true",
        help="additionally run the official run_prefill_check / run_teacher_forcing entry points "
        "(each builds its own generator) as the control for this driver's shared-generator shortcut",
    )
    args = ap.parse_args()

    from models.common.readiness_check.run_prefill_check import _run_one_entry_prefill
    from models.common.readiness_check.run_teacher_forcing import _run_one_entry
    from models.common.readiness_check.schema import load_reference
    from models.common.readiness_check.teacher_forcing import TokenAccuracy

    policy_arg = args.config or args.policy
    policy = resolve_policy(policy_arg)
    reference_path = Path(args.reference).resolve()
    command = "python " + " ".join(sys.argv)

    mesh = open_ornith_mesh()
    record: dict = {
        "config_id": policy.name,
        "precision_config_path": str(args.config) if args.config else None,
        "policy_argument": str(policy_arg),
        "dtype_policy": policy_to_dict(policy),
        "reference": str(reference_path),
        "command": command,
        "note": args.note,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _git("rev-parse", "HEAD"),
        "hardware": _hardware(mesh),
        "measurement_regime": {
            "accuracy": "full-model top-1/top-5/top-100 against the AIME24 chat-template readiness "
            "reference (1 entry, 100 generated tokens, top-k=100), through run_prefill_check's and "
            "run_teacher_forcing's own per-entry functions",
            "performance": "trace-verified teacher-forcing decode t/s/u, batch 1, greedy, device "
            "sampling, serial loop by construction; best of --tf-repeats repeats in one build",
            "workload": "AIME24 prompt + 100 forced tokens",
        },
        "gate": GATE,
    }
    try:
        record["hardware"]["mesh_shape"] = list(mesh.shape)
        gen = None
        build_started = time.perf_counter()
        gen = build_generator(
            model_dir=MODEL_DIR.resolve(),
            mesh_device=mesh,
            max_batch_size=1,
            cache_context=args.cache_context,
            policy=policy,
        )
        record["build_s"] = time.perf_counter() - build_started
        try:
            model = gen.model
            record["capability"] = model.capability()
            record["precision_summary"] = model.precision_summary()

            # ---- prefill accuracy -------------------------------------------------------
            reference = load_reference(reference_path)
            prefill_stats = []
            for entry_idx, entry in enumerate(reference.entries):
                if entry_idx:
                    gen.reset()
                prefill_stats.append(_run_one_entry_prefill(generator=gen, entry=entry, reference=reference))
            record["prefill"] = prefill_stats
            logger.info(f"prefill: {prefill_stats}")

            # ---- traced teacher forcing -------------------------------------------------
            tf_repeats = []
            for repeat in range(args.tf_repeats):
                acc = TokenAccuracy(reference_path)
                per_entry = []
                recaptures_before = gen.trace_recaptures
                for entry_idx in range(acc.num_entries):
                    gen.reset()
                    stats = _run_one_entry(generator=gen, acc=acc, entry_idx=entry_idx)
                    per_entry.append(stats)
                perf = dict(gen.perf)
                tf_repeats.append(
                    {
                        "repeat": repeat,
                        "per_entry": per_entry,
                        "generator_perf": perf,
                        "trace": {
                            **_trace_evidence(gen, perf),
                            "recaptures_inside_timed_window": gen.trace_recaptures - recaptures_before,
                        },
                    }
                )
                logger.info(
                    f"teacher forcing repeat {repeat}: top1={per_entry[0]['top1']:.3f} "
                    f"decode={per_entry[0].get('decode_t/s/u', 0):.3f} t/s/u"
                )
            record["teacher_forcing_repeats"] = tf_repeats

            if args.verify_against_runners:
                from models.common.readiness_check.run_prefill_check import run_prefill_check
                from models.common.readiness_check.run_teacher_forcing import run_teacher_forcing

                gen.teardown()
                gen = None
                build_kwargs = {"max_batch_size": 1, "cache_context": args.cache_context, "policy": policy}
                record["official_runner_control"] = {
                    "prefill": run_prefill_check(
                        model_dir=MODEL_DIR.resolve(),
                        reference_path=reference_path,
                        mesh_device=mesh,
                        build_kwargs=build_kwargs,
                    ),
                    "teacher": run_teacher_forcing(
                        model_dir=MODEL_DIR.resolve(),
                        reference_path=reference_path,
                        mesh_device=mesh,
                        build_kwargs=build_kwargs,
                    ),
                }
        finally:
            if gen is not None:
                gen.teardown()
    finally:
        close_ornith_mesh(mesh)

    # ---- aggregate ----------------------------------------------------------------
    #
    # A repeat is **warm** only if no trace re-capture landed inside its timed window. The first
    # repeat after a build normally is not: `_ensure_traces_replay_safe` re-captures once, between
    # prefill and the decode loop, because prefill compiled this prompt length's programs while the
    # traces were live. That re-capture is inside `run_teacher_forcing`'s decode window (which starts
    # at the first `next_input` callback) and costs ~9 % of the measured rate here - it is the whole
    # difference between the archived 38.18 t/s/u and this build's 41.92. Ranking on it would rank
    # trace-capture cost, so the reported figure is the best **warm** repeat and cold repeats are
    # kept in the record but excluded.
    prefill = record["prefill"][0]
    repeats = record["teacher_forcing_repeats"]
    tf_entries = [r["per_entry"][0] for r in repeats]
    warm = [r["per_entry"][0] for r in repeats if r["trace"]["recaptures_inside_timed_window"] == 0]
    accuracies = {(e["top1"], e["top5"], e["top100"]) for e in tf_entries}
    ranked = warm or tf_entries
    best = max(ranked, key=lambda e: e.get("decode_t/s/u", 0.0))
    rates = [e["decode_t/s/u"] for e in ranked if e.get("decode_t/s/u")]
    record["result"] = {
        "prefill_top1": prefill["top1"],
        "prefill_top5": prefill["top5"],
        "prefill_top100": prefill["top100"],
        "teacher_top1": best["top1"],
        "teacher_top5": best["top5"],
        "teacher_top100": best["top100"],
        "teacher_accuracy_identical_across_repeats": len(accuracies) == 1,
        "teacher_accuracy_per_repeat": sorted(accuracies),
        "ttft_ms": min(e["ttft_ms"] for e in ranked),
        "ttft_ms_per_repeat": [e["ttft_ms"] for e in tf_entries],
        "teacher_decode_t/s/u": best.get("decode_t/s/u"),
        "teacher_decode_ms_per_token": 1e3 / best["decode_t/s/u"] if best.get("decode_t/s/u") else None,
        "teacher_decode_t/s/u_per_repeat": [e.get("decode_t/s/u") for e in tf_entries],
        "warm_repeats": len(warm),
        "total_repeats": len(tf_entries),
        "teacher_decode_t/s/u_warm_repeats": rates,
        "teacher_decode_spread_pct": ((max(rates) / min(rates) - 1) * 100) if len(rates) > 1 else 0.0,
        "tokens": best["total"],
    }
    # Both gates must pass: prefill exercises the prefill numerics, teacher forcing the decode ones.
    failures = []
    for gate_name, gate_value in GATE.items():
        for phase in ("prefill", "teacher"):
            got = record["result"][f"{phase}_{gate_name}"]
            if got < gate_value - 1e-9:
                failures.append(f"{phase} {gate_name} {got:.3f} < {gate_value:.3f}")
    record["result"]["failures"] = failures
    record["result"]["status"] = "pass" if not failures else "fail"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(record["result"], indent=2, default=str))
    print(f"SWEEP_ONE_OK {record['config_id']} {record['result']['status']}")


if __name__ == "__main__":
    main()
