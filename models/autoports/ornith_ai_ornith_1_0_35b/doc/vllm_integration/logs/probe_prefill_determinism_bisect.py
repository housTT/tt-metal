# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""What, exactly, makes ``logit_determinism.json``'s ``prefill_rerun`` arm non-bit-identical?

``prefill_alloc_vs_recapture.json`` refuted the two suspects in ``prefill_requests_into_slots``: on the
full 40-layer model, per-call ``_page_row_tensor`` allocation and a forced trace re-capture are **both**
bit-identical across rounds, and so is the verbatim serving call, cold or warm. So the cause is one of
the remaining differences between that probe's loop and ``probe_logit_determinism.py``'s.

This probe rebuilds ``probe_logit_determinism.build()`` **verbatim** - the externally allocated
``1 + blocks_per_user`` KV cache, the all-zero page table handed to the generator, the ``arange`` table
starting at block 1 used for the calls - and then varies exactly one thing per arm. Every arm is
reset -> prefill -> one traced decode step, repeated, comparing **consecutive rounds' prefill logits**:

* ``exact``                    - the ``prefill_rerun`` loop verbatim: ``prefill_requests_into_slots``
  with ``page_table=`` but **no** ``kv_cache=``, and a decode step with ``sample_on_device=False``
  whose full logits are composed on host. This arm has to reproduce the reported instability;
* ``device_sampled_decode``    - the same prefill, but the *serving* decode step
  (``sample_on_device=True``, one sampling-trace replay, only the token id read back). Separates "the
  prefill differs" from "the decode step's host-logits readback differs";
* ``zeros_table_long_row``     - ``model.prefill_request_into_slot`` with a long-lived page row whose
  **contents are the generator's own all-zero page table**, which is what ``_resolve_page_table``
  substitutes when ``page_table`` arrives without ``kv_cache``. No per-call allocation and no
  generator call at all, so if this arm is unstable the cause is the *table contents*;
* ``real_table_long_row``      - the same call with the caller's real ``arange`` table, page row
  allocated once before capture;
* ``real_table_percall_row``   - the same, with the page row allocated and freed on every round, which
  is what ``prefill_requests_into_slots`` does. Together with the arm above this re-tests hypothesis
  (a) inside the construction that actually reproduces;
* ``verbatim_with_kv_cache``   - ``prefill_requests_into_slots`` with ``kv_cache=`` passed as well, so
  the caller's table is honoured instead of substituted. This is the one-line difference between the
  reproducing arm and a correct call;
* ``exact_repeat``             - ``exact`` again, last, so a first-arm/cold-cache order effect cannot
  be mistaken for a verdict.

Logit PCC is reported next to max/mean |Δ| so any deviation is comparable to
``doc/datatype_sweep/README.md`` §9.1 (cross-slot max |Δ| 0.28-0.5 at PCC >= 0.9993). The deviation
this probe finds is an order of magnitude outside that envelope, which is how a reader can tell it is
a wrong-addressing bug and not accumulated arithmetic noise.

What a reader should conclude, from the two artifacts this probe writes:

* ``prefill_determinism_bisect.json`` - **before** the ``_resolve_page_table`` fix (this probe with
  ``if kv_cache is None:`` restored in ``_resolve_page_table``, and an explicit ``--output``). ``exact``,
  ``device_sampled_decode`` and ``exact_repeat`` are not bit-identical (max |Δ| 3.3-3.9, PCC
  0.949-0.963 on the 40-layer model) and ``zeros_table_long_row`` reproduces the same deviation with
  no generator call at all, while ``real_table_long_row``, ``real_table_percall_row`` and
  ``verbatim_with_kv_cache`` are bit-identical. The cause is the *contents* of the substituted page
  table, not the per-call page-row allocation and not the trace re-capture;
* ``prefill_determinism_bisect_fixed.json`` - **after** it. A generator built on a caller-owned cache
  now honours the caller's table, so ``exact`` becomes bit-identical too, and
  ``zeros_table_long_row`` - which asks for the all-zero table explicitly - stays unstable as the
  negative control that the mechanism is still exactly what it was.

    python .../doc/vllm_integration/logs/probe_prefill_determinism_bisect.py --layer-indices all
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
from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import num_blocks_for_context

MODEL_DIR = Path(__file__).resolve().parents[3]
PROBE_LAYERS = [0, 3]
# The determinism probe's prompt is the five ids of "The capital of France is"; the ids themselves are
# irrelevant to a reproducibility measurement, the length is not.
PROMPT = [791, 6864, 315, 9822, 374]


def pcc(a, b):
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
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument(
        "--output",
        # The post-fix artifact by default, so a rerun cannot overwrite the pre-fix evidence: that one
        # is reproduced by reverting `_resolve_page_table`'s single condition and passing --output.
        default=str(MODEL_DIR / "doc" / "vllm_integration" / "prefill_determinism_bisect_fixed.json"),
    )
    args = ap.parse_args()
    layers = None if args.layer_indices.strip() == "all" else [int(v) for v in args.layer_indices.split(",")]
    report = {"layers": layers or "all", "rounds": args.rounds, "context": args.context, "prompt": PROMPT}

    mesh = open_ornith_mesh()
    try:
        path = resolve_model_path()
        model = OrnithModel.from_pretrained(
            path, mesh_device=mesh, hf_config=load_text_config(path), layer_indices=layers, max_context=args.context
        )
        # ---- probe_logit_determinism.build(mesh, 1, tokenizer), verbatim ----
        blocks_per_user = num_blocks_for_context(args.context, model.page_block_size)
        kv_cache = model.allocate_kv_cache(1 + blocks_per_user)
        generator = OrnithGenerator(
            model,
            max_batch_size=1,
            cache_context=args.context,
            sampling_mode="device",
            kv_cache=kv_cache,
            page_table=torch.zeros(1, blocks_per_user, dtype=torch.int32),
        )
        table = torch.arange(1, 1 + blocks_per_user, dtype=torch.int32).reshape(1, blocks_per_user)
        report["geometry"] = {
            "page_block_size": int(model.page_block_size),
            "blocks_per_user": int(blocks_per_user),
            "prefill_chunk": int(model.prefill_chunk),
            "kv_cache_blocks": 1 + int(blocks_per_user),
            "generator_page_table_is_all_zero": bool(int(generator.page_table.abs().sum()) == 0),
        }
        # Allocated before capture, so it is not a buffer the live trace could be sharing addresses with.
        long_row = generator._page_row_tensor(table[0:1])
        # The table `_resolve_page_table` substitutes for a caller table that arrives without a cache.
        zeros_row = generator._page_row_tensor(generator.page_table[0:1])
        generator.ensure_serving_traces()
        generator.ensure_sampling_trace()

        prompt_tt = torch.tensor(PROMPT, dtype=torch.int64).reshape(1, -1)

        def verbatim_prefill():
            out = generator.prefill_requests_into_slots(
                prompt_tt, [len(PROMPT)], [0], page_table=table[0:1], sample_on_device=False
            )
            return out[0, -1].clone()

        def with_kv_cache_prefill():
            out = generator.prefill_requests_into_slots(
                prompt_tt,
                [len(PROMPT)],
                [0],
                page_table=table[0:1],
                kv_cache=kv_cache,
                sample_on_device=False,
            )
            return out[0, -1].clone()

        def model_prefill(page_row):
            out = model.prefill_request_into_slot(
                prompt_tt, page_table=page_row, slot=0, start_pos=0, return_logits=True
            )
            return out[0, -1].clone()

        def real_table_long_row():
            return model_prefill(long_row)

        def zeros_table_long_row():
            return model_prefill(zeros_row)

        def real_table_percall_row():
            row = generator._page_row_tensor(table[0:1])
            try:
                return model_prefill(row)
            finally:
                ttnn.deallocate(row)

        def host_logits_decode(last):
            """The determinism probe's decode step: no device sampling, full logits composed on host."""
            tokens = torch.tensor([int(torch.argmax(last).item())], dtype=torch.int64)
            positions = torch.tensor([len(PROMPT)], dtype=torch.int64)
            generator.stage_serving_decode_inputs(tokens, positions, table, full_refresh=True)
            device_logits = generator.submit_serving_decode(sample_on_device=False)
            return generator.logits_from(device_logits)[0, 0].clone()

        def device_sampled_decode(last):
            """The serving decode step: sampling-trace replay, only the token id read back."""
            tokens = torch.tensor([int(torch.argmax(last).item())], dtype=torch.int64)
            positions = torch.tensor([len(PROMPT)], dtype=torch.int64)
            generator.stage_serving_decode_inputs(tokens, positions, table, full_refresh=True)
            generator.submit_serving_decode(sample_on_device=True)
            return generator.read_tokens()

        def run_arm(name, prefill, decode):
            prefills = []
            trace = []
            for _ in range(args.rounds):
                generator.reset()
                before_entries = int(mesh.num_program_cache_entries())
                before_recaptures = generator.trace_recaptures
                current = prefill()
                prefills.append(current)
                decode(current)
                trace.append(
                    {
                        "program_cache_entries": int(mesh.num_program_cache_entries()),
                        "program_cache_grew": int(mesh.num_program_cache_entries()) - before_entries,
                        "recaptures_this_round": generator.trace_recaptures - before_recaptures,
                    }
                )
            result = summarise(prefills)
            result["rounds"] = trace
            report[name] = result
            logger.info(f"arm {name}: bit_stable={result['bit_stable']} max_abs_diff={result['max_abs_diff']}")

        arms = (
            ("exact", verbatim_prefill, host_logits_decode),
            ("device_sampled_decode", verbatim_prefill, device_sampled_decode),
            ("zeros_table_long_row", zeros_table_long_row, host_logits_decode),
            ("real_table_long_row", real_table_long_row, host_logits_decode),
            ("real_table_percall_row", real_table_percall_row, host_logits_decode),
            ("verbatim_with_kv_cache", with_kv_cache_prefill, host_logits_decode),
            ("exact_repeat", verbatim_prefill, host_logits_decode),
        )
        for name, prefill, decode in arms:
            run_arm(name, prefill, decode)

        ttnn.deallocate(long_row)
        ttnn.deallocate(zeros_row)
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
        }
        for name, _, _ in arms
    }
    Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
    logger.info(f"wrote {args.output}")
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
