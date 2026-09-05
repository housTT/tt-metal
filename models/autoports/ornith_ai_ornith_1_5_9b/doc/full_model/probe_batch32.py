# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Blocking AutoFix localization; never a performance or autoregression result."""

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import close_ornith_mesh, open_ornith_mesh

DOC = Path(__file__).resolve().parent
SLOTS = [0, 3, 6, 27, 30, 31]


def ranks(tensor):
    return [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(tensor)]


def difference(a, b):
    delta = (a - b).abs()
    af, bf = a.flatten().double(), b.flatten().double()
    finite = bool(af.isfinite().all() and bf.isfinite().all())
    if finite and af.numel() > 1 and af.std() > 0 and bf.std() > 0:
        pcc = float(torch.corrcoef(torch.stack((af, bf)))[0, 1])
    else:
        pcc = None
    return dict(
        equal=torch.equal(a, b),
        count=int((a != b).sum()),
        max_abs=float(delta.nan_to_num().max()),
        finite_a=int(a.isfinite().sum()),
        finite_b=int(b.isfinite().sum()),
        elements=a.numel(),
        max_magnitude_a=float(a.abs().nan_to_num().max()),
        max_magnitude_b=float(b.abs().nan_to_num().max()),
        pcc=pcc,
    )


def comparisons(values):
    return {
        f"rank{rank}_slot{slot}": difference(value[0], value[slot])
        for rank, value in enumerate(values)
        for slot in SLOTS[1:-1]
    }


def state_snapshot(gen):
    return {
        f"layer{li}_{name}": ranks(buf)
        for li, layer in enumerate(gen.kv_cache.decode_layers)
        if not layer.is_full_attention
        for name, buf in [("recurrent", layer.recurrent_state)]
        + [(f"conv{i}", v) for i, v in enumerate(layer.conv_state)]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--where-only", action="store_true")
    args = parser.parse_args()
    report = {}
    mesh = open_ornith_mesh()
    gen = None
    try:
        gen = build_generator(DOC.parents[1], mesh, layer_indices=[0, 3], cache_context=2048, max_batch_size=32)
        if args.where_only:
            shape = list(gen.kv_cache.decode_layers[0].conv_state[0].shape)
            base = torch.arange(torch.tensor(shape).prod()).reshape(shape).remainder(257).bfloat16()
            source = (base + 512).bfloat16()
            mask = torch.zeros(32, 1, 1)
            mask[3] = 1
            expected = torch.where(mask.bool(), source, base).float()
            for dtype in (ttnn.float32, ttnn.bfloat16):
                dst = gen.model.upload(base, dtype=ttnn.bfloat16)
                src = gen.model.upload(source, dtype=ttnn.bfloat16)
                select = gen.model.upload(mask, dtype=dtype)
                ttnn.where(select, src, dst, output_tensor=dst)
                report[str(dtype)] = [difference(expected, value) for value in ranks(dst)]
                for value in (dst, src, select):
                    ttnn.deallocate(value)
            return
        gen.ensure_traces()
        prompts = [([*range(127)], [*range(131)], [31, 57, 88])[i % 3] for i in range(31)]
        original_table = gen.page_table.clone()
        baseline = {}
        for label, table in (("A", original_table), ("A2", original_table), ("B", original_table.flip(1).contiguous())):
            case = report[label] = {}
            gen.reset()
            gen.page_table = table
            original_transfer = gen.model._transfer_slot
            first_state = {}

            def transfer(source, target, slot, *, into_batch):
                original_transfer(source, target, slot, into_batch=into_batch)
                if into_batch and slot in (0, 3, 6, 30):
                    snapshot = state_snapshot(gen)
                    if slot == 0:
                        first_state.update({key: [v[0].clone() for v in value] for key, value in snapshot.items()})
                    case[f"handoff{slot}_preserve_slot0"] = {
                        f"{key}_rank{rank}": difference(first_state[key][rank], value[rank][0])
                        for key, value in snapshot.items()
                        for rank in range(4)
                    }

            original_terminal = gen.model.terminal

            def terminal(hidden):
                values = [v.reshape(-1, gen.model.dim) for v in ranks(hidden)]
                case["prefill_terminal_input"] = comparisons(values)
                return original_terminal(hidden)

            gen.model._transfer_slot = transfer
            gen.model.terminal = terminal
            logits = gen.prefill_forward(
                prompts,
                page_table=table,
                kv_cache=gen.kv_cache,
                prompt_lens=list(map(len, prompts)),
                return_device_logits=True,
            )
            gen.model._transfer_slot = original_transfer
            gen.model.terminal = original_terminal
            for step in range(3):
                if step:
                    forced = [([12, 1076, 220] if step == 1 else [13, 1076, 220])[i % 3] for i in range(32)]
                    positions = [len(row) + step - 1 for row in prompts] + [-1]
                    gen._write_tokens(forced)
                    gen._write_positions(positions)
                    gen._refresh_table(table)
                    gen._ensure_replay_safe()
                    ttnn.execute_trace(mesh, gen._model_trace, cq_id=0, blocking=True)
                    logits = gen._logits
                values = [v.reshape(32, -1) for v in ranks(logits)]
                state = state_snapshot(gen)
                record = case[f"step{step}"] = dict(
                    logits_equal_slots=comparisons(values),
                    state_equal_slots={key: comparisons(value) for key, value in state.items()},
                )
                full = torch.cat(values, -1)[:, : gen.model.vocab_size]
                top = full.topk(10, dim=-1)
                record["top10"] = {
                    str(slot): dict(
                        ids=top.indices[slot].tolist(),
                        values=top.values[slot].tolist(),
                        tied_max=int((full[slot] == full[slot].max()).sum()),
                    )
                    for slot in SLOTS
                }
                if label == "A":
                    baseline[step] = (
                        [v[SLOTS].clone() for v in values],
                        {key: [v[SLOTS].clone() for v in value] for key, value in state.items()},
                    )
                else:
                    record["logits_vs_A"] = [difference(a, b[SLOTS]) for a, b in zip(baseline[step][0], values)]
                    record["state_vs_A"] = {
                        key: [difference(a, b[SLOTS]) for a, b in zip(baseline[step][1][key], value)]
                        for key, value in state.items()
                    }
                if step:
                    ttnn.execute_trace(mesh, gen._sampling_trace, cq_id=0, blocking=True)
                    sampled = [v.flatten().long() for v in ranks(gen._inputs[0])]
                    record["sampled"] = [v.tolist() for v in sampled]
                    record["cpu_argmax"] = full.argmax(-1).tolist()
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    label,
                    step,
                    "logits equal",
                    all(v["equal"] for v in record["logits_equal_slots"].values()),
                    flush=True,
                )
                if step == 0:
                    ttnn.deallocate(logits)
                    del logits
            del logits
    finally:
        if gen:
            gen.teardown()
        close_ornith_mesh(mesh)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
