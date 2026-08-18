# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Where does batch>=8 run-to-run deviation enter the decode step: the trace, or the ops?

``slot_reproducibility.json`` establishes the shape of the effect on the full 40-layer model: two
identical greedy requests from a wiped state are bit-identical at batch 1, 2 and 4, and stop being
bit-identical at batch 8, 16 and 32. It also rules two things out — the readback path (``read_twice`` is
bit-identical at batch 32) and residue in unoccupied rows (the all-rows arm deviates just the same) — and
localises the entry point to the *first traced decode step*: index 0 of every arm is the prefill and it is
bit-identical everywhere, index 1 is the first decode step and at batch >= 8 it already deviates by
0.4-0.9 of a logit.

This probe asks the next question, which is the one that decides whether the finding belongs to this stage
at all:

* ``traced``      - two runs of prefill + N *replayed* decode steps (what serving runs);
* ``eager``       - two runs of the same steps through ``decode_forward(enable_trace=False)``, so every op
                    is dispatched individually and no captured graph is involved;
* ``one_step``    - two runs of prefill + exactly one traced step, comparing the step's logits *and* the
                    DeltaNet/conv state it writes. If the state written by the first step already differs,
                    the deviation is produced inside that step rather than carried into it.

Each arm runs at ``--batch`` (default 32, where the effect lives) and at ``--control-batch`` (default 4,
where run-to-run bit-identity still holds), on one weight load, so the only difference between the two is
the decode geometry.

    python .../doc/vllm_integration/logs/probe_decode_nondeterminism.py --layer-indices all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import close_ornith_mesh, open_ornith_mesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_slot_reproducibility import (  # noqa: E402  (path insert first, by design)
    PROMPT,
    build,
    compare,
    compare_state,
    kv_fingerprint,
    state_fingerprint,
)

from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator import OrnithGenerator  # noqa: E402
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import (  # noqa: E402
    OrnithModel,
    load_text_config,
    resolve_model_path,
)
from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import num_blocks_for_context  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[3]


def build_on(mesh, batch, layers, context, tp):
    """``build`` from the slot probe, with an explicit ``tp`` so a 1x1 mesh can be measured too.

    The 1x1 arm is the collectives control: on one device there is no all-gather, no reduce-scatter and
    no MoE traffic over the fabric, so a batch-32 decode that is bit-reproducible there and not on 1x4
    points at the collectives rather than at the local matmuls.
    """
    path = resolve_model_path()
    model = OrnithModel.from_pretrained(
        path,
        mesh_device=mesh,
        hf_config=load_text_config(path),
        layer_indices=layers,
        max_context=context,
        tp=tp,
    )
    blocks_per_user = num_blocks_for_context(context, model.page_block_size)
    kv_cache = model.allocate_kv_cache(1 + batch * blocks_per_user)
    table = torch.zeros(batch, blocks_per_user, dtype=torch.int32)
    for user in range(batch):
        base = 1 + user * blocks_per_user
        table[user] = torch.arange(base, base + blocks_per_user, dtype=torch.int32)
    # On one device the device sampler's grouped local top-k does not exist (it is the multi-device
    # vocabulary-shard path), so the 1x1 control builds in the host compatibility mode. The arms below
    # read *logits*, never a sampled token, so the decode graph they measure is the same one.
    generator = OrnithGenerator(
        model,
        max_batch_size=batch,
        cache_context=context,
        sampling_mode="device" if tp > 1 else "host",
        kv_cache=kv_cache,
        page_table=table,
    )
    generator.ensure_serving_traces()
    if tp > 1:
        generator.ensure_sampling_trace()
    return model, generator, table, kv_cache


def model_level_run(model, mesh, page_table, prompt_ids, batch, steps):
    """Prefill + ``steps`` eager decode steps through the **model**, with no generator involved.

    The 1x1 control cannot build a generator: this model's device sampler is the multi-device
    vocabulary-shard path and its constructor refuses a single device. The decode graph under test is
    ``ttnn_decode_forward`` either way, so the control drives it directly - the same thing the eager arm
    does one call further up.
    """
    model.reset_state()
    device_logits = model.prefill_request_into_slot(
        torch.tensor(prompt_ids, dtype=torch.int64).reshape(1, -1),
        page_table=ttnn.from_torch(
            page_table[:1],
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        ),
        slot=0,
        start_pos=0,
        return_logits="device",
    )
    out = [model._logits_to_host(device_logits)[0, -1].clone()]
    ttnn.deallocate(device_logits)
    tokens = torch.zeros(batch, dtype=torch.int32)
    positions = torch.full((batch,), -1, dtype=torch.int32)
    tokens[0] = int(torch.argmax(out[0]).item())
    positions[0] = len(prompt_ids)
    for _ in range(steps):
        host = model.prepare_decode_inputs_host(tokens, positions, page_table)
        device_inputs = [ttnn.to_device(t, device=mesh) if t is not None else None for t in host]
        decode_logits = model.ttnn_decode_forward(*device_inputs)
        row = model.decode_logits_to_host(decode_logits)[0].reshape(-1).clone()
        out.append(row)
        ttnn.deallocate(decode_logits)
        for tensor in device_inputs:
            if tensor is not None:
                ttnn.deallocate(tensor)
        tokens[0] = int(torch.argmax(row).item())
        positions[0] = int(positions[0]) + 1
    return out


def prefill(generator, table, kv_cache, prompt_ids, slot):
    return generator.prefill_requests_into_slots(
        torch.tensor(prompt_ids, dtype=torch.int64).reshape(1, -1),
        [len(prompt_ids)],
        [slot],
        page_table=table[slot : slot + 1],
        kv_cache=kv_cache,
        sample_on_device=False,
    )


def run_steps(generator, table, kv_cache, prompt_ids, slot, steps, *, traced):
    """Prefill into ``slot``, then ``steps`` decode steps, traced or eager.

    Returns ``[prefill_logits] + per_step_logits``, the same layout the slot probe uses, so ``compare``
    reads the same way: index 0 is the prefill, index 1 is the first decode step.
    """
    generator.reset()
    prefill_logits = prefill(generator, table, kv_cache, prompt_ids, slot)
    batch = generator.max_batch_size
    tokens = torch.zeros(batch, dtype=torch.int64)
    positions = torch.full((batch,), -1, dtype=torch.int64)
    tokens[slot] = int(torch.argmax(prefill_logits[0, -1]).item())
    positions[slot] = len(prompt_ids)
    out = [prefill_logits[0, -1].clone()]
    for _ in range(steps):
        if traced:
            generator.stage_serving_decode_inputs(tokens, positions, table, full_refresh=True)
            device_logits = generator.submit_serving_decode(sample_on_device=False)
            host = generator.logits_from(device_logits)[slot, 0]
        else:
            # Eager: every op dispatched on its own, no captured graph. Same inputs, same page table.
            host = generator.decode_forward(tokens, positions, page_table=table, kv_cache=kv_cache, enable_trace=False)[
                slot
            ]
        out.append(host.clone().reshape(-1))
        tokens[slot] = int(torch.argmax(out[-1]).item())
        positions[slot] = int(positions[slot]) + 1
    return out


def one_step_with_state(generator, model, table, kv_cache, prompt_ids, slot):
    """Prefill + exactly one traced decode step; return its logits, and the state that step left."""
    batch = generator.max_batch_size
    generator.reset()
    prefill_logits = prefill(generator, table, kv_cache, prompt_ids, slot)
    state_after_prefill = state_fingerprint(model, batch) + kv_fingerprint(model)
    tokens = torch.zeros(batch, dtype=torch.int64)
    positions = torch.full((batch,), -1, dtype=torch.int64)
    tokens[slot] = int(torch.argmax(prefill_logits[0, -1]).item())
    positions[slot] = len(prompt_ids)
    generator.stage_serving_decode_inputs(tokens, positions, table, full_refresh=True)
    device_logits = generator.submit_serving_decode(sample_on_device=False)
    logits = generator.logits_from(device_logits)[slot, 0].clone()
    state_after_step = state_fingerprint(model, batch) + kv_fingerprint(model)
    return prefill_logits[0, -1].clone(), logits, state_after_prefill, state_after_step


def pairs_mode(args, layers):
    """How *often* do two identical runs differ? Counted, on one mesh, at one batch.

    The effect is intermittent - a single non-deviating pair proves nothing, which is why the by-layer
    bisect below could not exclude anything. A count over N pairs can: 0/N on one mesh against k/N on
    another is a real difference even when a single pair is not.
    """
    shape = tuple(int(v) for v in args.mesh.split(","))
    tp = shape[0] * shape[1]
    report = {
        "mode": "pairs",
        "mesh": list(shape),
        "tp": tp,
        "batch": args.batch,
        "steps": args.steps,
        "pairs": args.pairs,
        "layers": layers or "all",
        "context": args.context,
        "traced": [],
        "eager": [],
        "complete": False,
    }

    def checkpoint():
        Path(args.output).write_text(json.dumps(report, indent=1) + "\n")

    mesh = open_ornith_mesh(shape, fabric=shape != (1, 1))
    # `--force-model-driver` runs the 1x4 mesh through the *same* model-level driver the 1x1 control
    # must use, so that comparison has one variable (the mesh) instead of two (the mesh and the driver).
    model_driver = args.force_model_driver or tp == 1
    arms = ("eager",) if model_driver else ("traced", "eager")
    report["arms"] = list(arms)
    report["driver"] = (
        "model (forced)"
        if args.force_model_driver and tp > 1
        else "generator"
        if tp > 1
        else "model (no generator: the device sampler needs >1 device)"
    )
    try:
        if not model_driver:
            model, generator, table, kv_cache = build_on(mesh, args.batch, layers, args.context, tp)
        else:
            path = resolve_model_path()
            model = OrnithModel.from_pretrained(
                path,
                mesh_device=mesh,
                hf_config=load_text_config(path),
                layer_indices=layers,
                max_context=args.context,
                tp=tp,
            )
            blocks = num_blocks_for_context(args.context, model.page_block_size)
            model.allocate_kv_cache(1 + args.batch * blocks)
            model.allocate_state(args.batch)
            table = torch.zeros(args.batch, blocks, dtype=torch.int32)
            for user in range(args.batch):
                base = 1 + user * blocks
                table[user] = torch.arange(base, base + blocks, dtype=torch.int32)
            generator = kv_cache = None
        for index in range(args.pairs):
            for arm in arms:
                if not model_driver:
                    traced = arm == "traced"
                    left = run_steps(generator, table, kv_cache, PROMPT, args.slot, args.steps, traced=traced)
                    right = run_steps(generator, table, kv_cache, PROMPT, args.slot, args.steps, traced=traced)
                else:
                    left = model_level_run(model, mesh, table, PROMPT, args.batch, args.steps)
                    right = model_level_run(model, mesh, table, PROMPT, args.batch, args.steps)
                result = compare(left, right)
                report[arm].append(
                    {
                        "pair": index,
                        "all_bitwise_identical": result["all_bitwise_identical"],
                        "max_abs_diff": result["max_abs_diff"],
                        "min_pcc": result["min_pcc"],
                        "top1_agreement": result["top1_agreement"],
                        "first_step_bitwise_identical": result["steps"][1]["bitwise_identical"],
                    }
                )
                checkpoint()
            logger.info(f"pair {index}: " + ", ".join(f"{arm}={report[arm][-1]}" for arm in arms))
        if generator is not None:
            report["counters"] = dict(generator.counters)
            generator.teardown()
        del generator, model
    finally:
        close_ornith_mesh(mesh, fabric=shape != (1, 1))

    report["summary"] = {
        arm: {
            "pairs": len(report[arm]),
            "pairs_that_deviated": sum(1 for r in report[arm] if not r["all_bitwise_identical"]),
            "pairs_whose_first_decode_step_deviated": sum(
                1 for r in report[arm] if not r["first_step_bitwise_identical"]
            ),
            "max_abs_diff": max((r["max_abs_diff"] for r in report[arm]), default=None),
        }
        for arm in arms
    }
    report["complete"] = True
    checkpoint()
    logger.info(f"wrote {args.output}")
    print(json.dumps(report["summary"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--control-batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--layer-indices", default="0,3")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--mesh", default="1,4", help="mesh shape for --pairs mode, e.g. '1,4' or '1,1'")
    ap.add_argument(
        "--force-model-driver",
        action="store_true",
        help="drive the model directly even on a multi-device mesh, so the 1x1 comparison varies only the mesh",
    )
    ap.add_argument(
        "--pairs",
        type=int,
        default=0,
        help="count how many of N identical run-pairs deviate, on --mesh at --batch (skips the arms above)",
    )
    ap.add_argument("--output", default=str(MODEL_DIR / "doc" / "vllm_integration" / "decode_nondeterminism.json"))
    args = ap.parse_args()
    layers = None if args.layer_indices.strip() == "all" else [int(v) for v in args.layer_indices.split(",")]
    if args.pairs:
        pairs_mode(args, layers)
        return
    report = {
        "layers": layers or "all",
        "steps": args.steps,
        "context": args.context,
        "slot": args.slot,
        "prompt": PROMPT,
        "complete": False,
    }

    def checkpoint():
        Path(args.output).write_text(json.dumps(report, indent=1) + "\n")

    for batch in (args.control_batch, args.batch):
        mesh = open_ornith_mesh()
        try:
            logger.info(f"building at batch {batch}")
            model, generator, table, kv_cache = build(mesh, batch, layers, args.context)
            key = f"batch{batch}"

            traced_a = run_steps(generator, table, kv_cache, PROMPT, args.slot, args.steps, traced=True)
            traced_b = run_steps(generator, table, kv_cache, PROMPT, args.slot, args.steps, traced=True)
            report[f"{key}_traced"] = compare(traced_a, traced_b)
            checkpoint()

            eager_a = run_steps(generator, table, kv_cache, PROMPT, args.slot, args.steps, traced=False)
            eager_b = run_steps(generator, table, kv_cache, PROMPT, args.slot, args.steps, traced=False)
            report[f"{key}_eager"] = compare(eager_a, eager_b)
            checkpoint()

            pre_a, step_a, st_pre_a, st_post_a = one_step_with_state(
                generator, model, table, kv_cache, PROMPT, args.slot
            )
            pre_b, step_b, st_pre_b, st_post_b = one_step_with_state(
                generator, model, table, kv_cache, PROMPT, args.slot
            )
            report[f"{key}_one_step"] = {
                "prefill_logits_identical": bool(torch.equal(pre_a, pre_b)),
                "state_after_prefill": compare_state(st_pre_a, st_pre_b, batch),
                "first_step_logits": compare([pre_a, step_a], [pre_b, step_b]),
                "state_after_first_step": compare_state(st_post_a, st_post_b, batch),
            }
            report[f"{key}_counters"] = dict(generator.counters)
            checkpoint()
            generator.teardown()
            del generator, model
        finally:
            close_ornith_mesh(mesh)

    report["summary"] = {
        "traced_bit_identical": {
            f"batch{b}": report[f"batch{b}_traced"]["all_bitwise_identical"] for b in (args.control_batch, args.batch)
        },
        "eager_bit_identical": {
            f"batch{b}": report[f"batch{b}_eager"]["all_bitwise_identical"] for b in (args.control_batch, args.batch)
        },
        "first_step_logits_bit_identical": {
            f"batch{b}": report[f"batch{b}_one_step"]["first_step_logits"]["all_bitwise_identical"]
            for b in (args.control_batch, args.batch)
        },
        "state_after_first_step_bit_identical": {
            f"batch{b}": report[f"batch{b}_one_step"]["state_after_first_step"]["bitwise_identical"]
            for b in (args.control_batch, args.batch)
        },
        "state_after_prefill_bit_identical": {
            f"batch{b}": report[f"batch{b}_one_step"]["state_after_prefill"]["bitwise_identical"]
            for b in (args.control_batch, args.batch)
        },
    }
    report["complete"] = True
    checkpoint()
    logger.info(f"wrote {args.output}")
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
