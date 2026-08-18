# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Which of the two things ``prefill_requests_into_slots`` does breaks bit-identical prefill?

Two measurements disagree about whether repeating the *same* prefill returns the *same* logits:

* ``prefill_stability_with_traces.json`` drives it through ``model.prefill_request_into_slot`` with a
  **long-lived** page-table tensor and gets max |Δ| = 0.0 across rounds, with a captured decode trace
  live and replayed between rounds;
* ``logit_determinism.json`` arm ``prefill_rerun`` drives the same thing through
  ``OrnithGenerator.prefill_requests_into_slots`` and gets max |Δ| = 1.3-3.7.

``prefill_requests_into_slots`` adds exactly two things on top of the model call, and this probe puts
each one on its own arm, in one process, on one weight load, with the same prompt, the same slot, the
same traced decode step between rounds and the same host readback. Every arm is
reset -> prefill -> one traced decode replay, repeated, and compares **consecutive rounds' prefill
logits**, which nothing in any arm should change:

* ``cold_verbatim``            - the reproduce arm, run **first**:
  ``generator.prefill_requests_into_slots`` from a state where the traces have just been captured and
  no prefill has ever run, so the *first* round is what compiles the prefill programs. This is the
  ordering ``logit_determinism.json`` had, and the only ordering in which the prefill's kernel
  binaries are allocated after ``end_trace_capture`` handed the trace's intermediates back;
* ``long_row``                 - control: ``model.prefill_request_into_slot`` with the page row
  allocated **once, before trace capture**. This is the ``prefill_stability_with_traces`` arm;
* ``percall_row``             - identical, except the ``[1, blocks]`` page row is allocated with
  ``generator._page_row_tensor`` and deallocated again on **every** round, which is what
  ``prefill_requests_into_slots`` does. Isolates hypothesis (a): per-call allocation while a trace
  is live;
* ``long_row_forced_recapture`` - the control arm plus a forced ``_release_traces`` /
  ``_capture_traces`` before every prefill, which is what ``_ensure_traces_replay_safe`` does when a
  request has compiled a new program. Isolates hypothesis (b);
* ``verbatim``                 - ``generator.prefill_requests_into_slots(...)`` as serving calls it,
  i.e. (a) and (b) together. This is the arm that has to reproduce the failure;
* ``long_row_repeat``          - the control again, last, so an "everything after the first arm
  drifts" order effect cannot be mistaken for a verdict.

Program-cache size, ``trace_recaptures`` and the page row's device address are recorded per round, so
a reader can check that the replay-safety check was a no-op in the arms that were not meant to
re-capture, and can see whether the per-call allocation actually lands somewhere new each round.
Comparisons report logit PCC as well as max/mean |Δ|, so the size of any deviation is comparable to
``doc/datatype_sweep/README.md`` §9.1 (cross-slot max |Δ| 0.28-0.5 at PCC >= 0.9993).

A warm-up phase runs the verbatim path twice and then re-captures once, so every program the arms
need is already compiled: without it the first arm would pay a recapture that the later arms do not.

One caveat a reader needs, found afterwards and recorded here rather than hidden: this probe's
generator allocates its **own** KV cache, so ``_resolve_page_table`` substituted the generator's page
table for the one the ``verbatim`` arms passed. That substitution is harmless *here* only because the
two are element-wise identical (both ``arange(blocks)``), which is why these arms are bit-identical
while ``logit_determinism.json``'s were not - there the generator's table was all zeros.
``prefill_determinism_bisect.json`` is the probe that isolates that, and it is the one to read for the
cause; this one's value is the refutation of (a) and (b), which the substitution does not affect.

    python .../doc/vllm_integration/logs/probe_prefill_alloc_vs_recapture.py                  # 2-layer
    python .../doc/vllm_integration/logs/probe_prefill_alloc_vs_recapture.py --layer-indices all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator import OrnithGenerator
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import (
    OrnithModel,
    close_ornith_mesh,
    load_text_config,
    open_ornith_mesh,
    resolve_model_path,
)

MODEL_DIR = Path(__file__).resolve().parents[3]
PROBE_LAYERS = [0, 3]
PROMPT = [791, 6864, 315, 9822, 374]  # five arbitrary in-vocabulary ids: the length is what matters


def pcc(a, b):
    """Pearson correlation of the two logit vectors, the datatype-sweep stage's comparison metric."""
    x = a.float().reshape(-1)
    y = b.float().reshape(-1)
    if torch.equal(x, y):
        return 1.0
    return float(torch.corrcoef(torch.stack([x, y]))[0, 1])


def stats(a, b):
    diff = (a - b).abs()
    top_a = torch.topk(a, 2)
    top_b = torch.topk(b, 2)
    return {
        "bitwise_identical": bool(torch.equal(a, b)),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "nonzero_fraction": float((diff != 0).float().mean()),
        "pcc": pcc(a, b),
        "top1_left": int(top_a.indices[0]),
        "top1_right": int(top_b.indices[0]),
        "top1_agrees": int(top_a.indices[0]) == int(top_b.indices[0]),
        "top1_margin_left": float(top_a.values[0] - top_a.values[1]),
    }


def address_of(tensor):
    try:
        return int(ttnn.get_device_tensors(tensor)[0].buffer_address())
    except Exception:  # noqa: BLE001 - the address is diagnostic, not load-bearing
        return None


def summarise(rounds):
    comparisons = [stats(a, b) for a, b in zip(rounds[:-1], rounds[1:])]
    return {
        "comparisons": comparisons,
        "bit_stable": all(c["bitwise_identical"] for c in comparisons),
        "max_abs_diff": max((c["max_abs_diff"] for c in comparisons), default=None),
        "min_pcc": min((c["pcc"] for c in comparisons), default=None),
        "top1_stable": all(c["top1_agrees"] for c in comparisons),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--layer-indices", default=",".join(str(v) for v in PROBE_LAYERS))
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--output", default=str(MODEL_DIR / "doc" / "vllm_integration" / "prefill_alloc_vs_recapture.json"))
    args = ap.parse_args()
    layers = None if args.layer_indices.strip() == "all" else [int(v) for v in args.layer_indices.split(",")]
    report = {
        "layers": layers or "all",
        "rounds": args.rounds,
        "context": args.context,
        "prompt": PROMPT,
    }

    mesh = open_ornith_mesh()
    try:
        path = resolve_model_path()
        model = OrnithModel.from_pretrained(
            path,
            mesh_device=mesh,
            hf_config=load_text_config(path),
            layer_indices=layers,
            max_context=args.context,
        )
        generator = OrnithGenerator(model, max_batch_size=1, cache_context=args.context, sampling_mode="device")
        table = generator.page_table
        # Allocated BEFORE any capture, exactly like `_prefill_tokens`: this is the control's row.
        long_row = generator._page_row_tensor(table[0:1])
        generator.ensure_serving_traces()
        generator.ensure_sampling_trace()
        report["long_row_address"] = address_of(long_row)

        prompt_tt = torch.tensor(PROMPT, dtype=torch.int64).reshape(1, -1)

        def decode_step(last_logits):
            tokens = torch.tensor([int(torch.argmax(last_logits).item())], dtype=torch.int64)
            positions = torch.tensor([len(PROMPT)], dtype=torch.int64)
            generator.stage_serving_decode_inputs(tokens, positions, table, full_refresh=True)
            generator.submit_serving_decode(sample_on_device=True)
            generator.read_tokens()

        def model_prefill(page_row):
            out = model.prefill_request_into_slot(
                prompt_tt, page_table=page_row, slot=0, start_pos=0, return_logits=True
            )
            return out[0, -1].clone()

        def verbatim_prefill():
            out = generator.prefill_requests_into_slots(
                prompt_tt, [len(PROMPT)], [0], page_table=table[0:1], sample_on_device=False
            )
            return out[0, -1].clone()

        def run_arm(name, *, percall_row, forced_recapture, verbatim):
            logits = []
            trace = []
            for _ in range(args.rounds):
                generator.reset()
                before_entries = int(mesh.num_program_cache_entries())
                before_recaptures = generator.trace_recaptures
                if forced_recapture:
                    ttnn.synchronize_device(mesh)
                    generator._release_traces()
                    generator._capture_traces()
                    generator.trace_recaptures += 1
                if verbatim:
                    row_address = None
                    current = verbatim_prefill()
                elif percall_row:
                    row = generator._page_row_tensor(table[0:1])
                    row_address = address_of(row)
                    current = model_prefill(row)
                    ttnn.deallocate(row)
                else:
                    row_address = address_of(long_row)
                    current = model_prefill(long_row)
                logits.append(current)
                decode_step(current)
                trace.append(
                    {
                        "program_cache_entries": int(mesh.num_program_cache_entries()),
                        "program_cache_grew": int(mesh.num_program_cache_entries()) - before_entries,
                        "recaptures_this_round": generator.trace_recaptures - before_recaptures,
                        "page_row_address": row_address,
                    }
                )
            result = summarise(logits)
            result["rounds"] = trace
            report[name] = result
            logger.info(f"arm {name}: bit_stable={result['bit_stable']} max_abs_diff={result['max_abs_diff']}")

        # ------------------------------------------------------- the reproduce arm, first of all
        # `cold_verbatim` runs before anything has been prefilled, which is the ordering
        # `logit_determinism.json` had: the traces were captured by `ensure_serving_traces()` and the
        # *first* prefill is what compiles the prefill programs, so their kernel binaries land after
        # the capture. This is the arm that has to be non-bit-identical; every arm below it runs from
        # a program cache that is already complete.
        run_arm("cold_verbatim", percall_row=True, forced_recapture=False, verbatim=True)

        # ---------------------------------------------------------------- warm-up
        # Compile everything the arms need while a recapture is still free, then re-capture once so
        # the traces are already consistent with the final program cache.
        for _ in range(2):
            generator.reset()
            decode_step(verbatim_prefill())
        ttnn.synchronize_device(mesh)
        generator._release_traces()
        generator._capture_traces()
        report["program_cache_after_warmup"] = int(mesh.num_program_cache_entries())
        report["trace_recaptures_after_warmup"] = generator.trace_recaptures

        run_arm("long_row", percall_row=False, forced_recapture=False, verbatim=False)
        run_arm("percall_row", percall_row=True, forced_recapture=False, verbatim=False)
        run_arm("long_row_forced_recapture", percall_row=False, forced_recapture=True, verbatim=False)
        run_arm("verbatim", percall_row=True, forced_recapture=False, verbatim=True)
        run_arm("long_row_repeat", percall_row=False, forced_recapture=False, verbatim=False)

        ttnn.deallocate(long_row)
        generator.teardown()
    finally:
        close_ornith_mesh(mesh)

    report["summary"] = {
        name: {
            "bit_stable": report[name]["bit_stable"],
            "max_abs_diff": report[name]["max_abs_diff"],
            "min_pcc": report[name]["min_pcc"],
            "top1_stable": report[name]["top1_stable"],
            "recaptures": sum(r["recaptures_this_round"] for r in report[name]["rounds"]),
            "distinct_page_row_addresses": len({r["page_row_address"] for r in report[name]["rounds"]}),
        }
        for name in (
            "cold_verbatim",
            "long_row",
            "percall_row",
            "long_row_forced_recapture",
            "verbatim",
            "long_row_repeat",
        )
    }
    Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
    logger.info(f"wrote {args.output}")
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
