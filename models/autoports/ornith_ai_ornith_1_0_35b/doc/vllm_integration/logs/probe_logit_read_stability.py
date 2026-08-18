# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Is the logit *readback* stable, or is the *computation* what varies run to run?

``logit_determinism.json`` showed two identical prefills of the same prompt returning logits that
differ by ~1-4 (bfloat16 logit units). Two explanations fit that: the read is racy, or the multichip
forward pass is not bit-reproducible. (That artifact was later **withdrawn** - it had prefilled through a
substituted all-zero page table, see ``../work_log.md`` section 9 - but this probe's own answers stand on
their own and are what ruled the read path out.) This probe separates them on the reduced two-layer target, which
is cheap enough to repeat:

* ``read_twice``      - one forward, the resulting device logits composed to host **twice**. A racy read
  shows up here and nowhere else. This must be bit-identical;
* ``forward_twice``   - two forwards of the same tokens over the same wiped state, each read once. This
  is the computation, with the read already vindicated by the arm above;
* ``forward_twice_single_device`` - the same two forwards on a **1x1** mesh, where no collective runs at
  all. If the variation disappears, it is the fabric collectives' reduction order;
* ``forward_twice_after_decode`` - the same two forwards with a **traced decode step** in between, which
  is the sequence the full-model probe actually ran. The full model's prefill logits move by ~1-4 across
  such rounds while the two arms above are bit-identical, so this arm asks whether it is the decode step
  in between (a state or scratch buffer surviving the wipe) rather than the forward itself.

    python .../doc/vllm_integration/logs/probe_logit_read_stability.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from loguru import logger

import ttnn
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
CONTEXT = 2048
PROMPT = [6, 66, 666, 6666, 66, 6, 66]


def stats(a, b):
    diff = (a - b).abs()
    return {
        "bitwise_identical": bool(torch.equal(a, b)),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "nonzero_fraction": float((diff != 0).float().mean()),
    }


def one_mesh(shape, rounds, layer_indices=None, context=CONTEXT):
    mesh = open_ornith_mesh(shape, fabric=tuple(shape) != (1, 1))
    try:
        path = resolve_model_path()
        model = OrnithModel.from_pretrained(
            path,
            mesh_device=mesh,
            hf_config=load_text_config(path),
            layer_indices=layer_indices,
            max_context=context,
            tp=shape[0] * shape[1],
        )
        blocks = num_blocks_for_context(context, model.page_block_size)
        model.allocate_kv_cache(blocks)
        model.allocate_state(1)
        page_table = torch.arange(blocks, dtype=torch.int32).reshape(1, blocks)
        out = {"read_twice": [], "forward_twice": [], "forward_twice_after_decode": []}
        previous = None
        previous_after_decode = None
        page_table_tt = ttnn.from_torch(
            page_table,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        for _ in range(rounds):
            model.reset_state()
            device_logits = model.prefill_request_into_slot(
                torch.tensor(PROMPT, dtype=torch.int64).reshape(1, -1),
                page_table=ttnn.from_torch(
                    page_table,
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
            # Two independent compositions of the SAME device tensor.
            first_read = model._logits_to_host(device_logits)[0, -1].clone()
            second_read = model._logits_to_host(device_logits)[0, -1].clone()
            ttnn.deallocate(device_logits)
            out["read_twice"].append(stats(first_read, second_read))
            if previous is not None:
                out["forward_twice"].append(stats(previous, first_read))
            previous = first_read

            # Now the sequence the full-model probe ran: one decode step, then wipe and prefill again.
            host = model.prepare_decode_inputs_host(
                torch.tensor([int(torch.argmax(first_read).item())], dtype=torch.int32),
                torch.tensor([len(PROMPT)], dtype=torch.int32),
                page_table,
            )
            device_inputs = [ttnn.to_device(t, device=mesh) if t is not None else None for t in host]
            decode_logits = model.ttnn_decode_forward(*device_inputs)
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(decode_logits)
            for tensor in device_inputs:
                if tensor is not None:
                    ttnn.deallocate(tensor)
            model.reset_state()
            after = model.prefill_request_into_slot(
                torch.tensor(PROMPT, dtype=torch.int64).reshape(1, -1),
                page_table=page_table_tt,
                slot=0,
                start_pos=0,
                return_logits="device",
            )
            after_read = model._logits_to_host(after)[0, -1].clone()
            ttnn.deallocate(after)
            if previous_after_decode is not None:
                out["forward_twice_after_decode"].append(stats(previous_after_decode, after_read))
            previous_after_decode = after_read
            out["forward_twice_after_decode"].append(stats(first_read, after_read))
        ttnn.deallocate(page_table_tt)
        return out
    finally:
        close_ornith_mesh(mesh, fabric=tuple(shape) != (1, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument(
        "--layer-indices",
        default=",".join(str(v) for v in PROBE_LAYERS),
        help='comma-separated HF layer indices, or "all" for the whole 40-layer stack',
    )
    ap.add_argument("--context", type=int, default=CONTEXT)
    ap.add_argument("--single-device", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--output", default=str(MODEL_DIR / "doc" / "vllm_integration" / "logit_read_stability.json"))
    args = ap.parse_args()
    layers = None if args.layer_indices.strip() == "all" else [int(v) for v in args.layer_indices.split(",")]
    report = {
        "layers": layers or "all",
        "prompt": PROMPT,
        "rounds": args.rounds,
        "context": args.context,
    }

    multi = one_mesh((1, 4), args.rounds, layers, args.context)
    report["read_twice"] = multi["read_twice"]
    report["forward_twice"] = multi["forward_twice"]
    report["forward_twice_after_decode"] = multi["forward_twice_after_decode"]
    single = (
        one_mesh((1, 1), args.rounds, layers, args.context)
        if args.single_device
        else {"read_twice": [], "forward_twice": [], "forward_twice_after_decode": []}
    )
    report["read_twice_single_device"] = single["read_twice"]
    report["forward_twice_single_device"] = single["forward_twice"]
    report["forward_twice_after_decode_single_device"] = single["forward_twice_after_decode"]

    def stable(rows):
        """``None`` for an arm that did not run.

        ``all([])`` is ``True``, so summarising a skipped arm with a bare ``all`` reports the strongest
        possible claim from no measurement at all - and the 1x1 arms *are* skipped for the full model,
        which does not fit on one device. A reader must be able to tell "measured, stable" from
        "not measured".
        """
        return all(r["bitwise_identical"] for r in rows) if rows else None

    report["single_device_arms_ran"] = bool(args.single_device)
    report["summary"] = {
        "read_is_bit_stable_1x4": stable(report["read_twice"]),
        "read_is_bit_stable_1x1": stable(report["read_twice_single_device"]),
        "forward_is_bit_stable_1x4": stable(report["forward_twice"]),
        "forward_is_bit_stable_1x1": stable(report["forward_twice_single_device"]),
        "max_abs_diff_forward_1x4": max((r["max_abs_diff"] for r in report["forward_twice"]), default=None),
        "max_abs_diff_forward_1x1": max(
            (r["max_abs_diff"] for r in report["forward_twice_single_device"]), default=None
        ),
        "forward_after_decode_is_bit_stable_1x4": stable(report["forward_twice_after_decode"]),
        "forward_after_decode_is_bit_stable_1x1": stable(report["forward_twice_after_decode_single_device"]),
        "max_abs_diff_forward_after_decode_1x4": max(
            (r["max_abs_diff"] for r in report["forward_twice_after_decode"]), default=None
        ),
        "max_abs_diff_forward_after_decode_1x1": max(
            (r["max_abs_diff"] for r in report["forward_twice_after_decode_single_device"]), default=None
        ),
    }
    Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
    logger.info(f"wrote {args.output}")
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
