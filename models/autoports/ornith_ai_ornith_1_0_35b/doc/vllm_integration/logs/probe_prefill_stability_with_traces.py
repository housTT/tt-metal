# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Does a captured decode trace make repeated identical prefills stop being bit-identical?

``logit_read_stability_full_model.json`` shows the whole 40-layer prefill is bit-identical across
rounds when it is driven straight through the model, with an eager decode step in between. The
serving-path probe (``logit_determinism.json``) drives the *same* prefill through the generator - which
captures a decode trace and replays it - and there the prefill logits move by 1-4 between rounds. This
probe runs both in one process, on one weight load, so the only difference between the arms is the
trace:

* ``eager``     - model-only: reset, prefill, eager decode step, repeat;
* ``traced``    - the same sequence through the generator, whose decode step is a captured trace replay.

Every arm compares consecutive rounds' *prefill* logits, which nothing in either arm should change.

    python .../doc/vllm_integration/logs/probe_prefill_stability_with_traces.py --layer-indices all
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
PROMPT = [6, 66, 666, 6666, 66, 6, 66]


def stats(a, b):
    diff = (a - b).abs()
    return {
        "bitwise_identical": bool(torch.equal(a, b)),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "nonzero_fraction": float((diff != 0).float().mean()),
        "top1_left": int(torch.argmax(a)),
        "top1_right": int(torch.argmax(b)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--layer-indices", default="all")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument(
        "--output", default=str(MODEL_DIR / "doc" / "vllm_integration" / "prefill_stability_with_traces.json")
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
        blocks = num_blocks_for_context(args.context, model.page_block_size)
        kv_cache = model.allocate_kv_cache(blocks)
        model.allocate_state(1)
        host_table = torch.arange(blocks, dtype=torch.int32).reshape(1, blocks)
        table_tt = ttnn.from_torch(
            host_table,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        def prefill_logits():
            device_logits = model.prefill_request_into_slot(
                torch.tensor(PROMPT, dtype=torch.int64).reshape(1, -1),
                page_table=table_tt,
                slot=0,
                start_pos=0,
                return_logits="device",
            )
            host = model._logits_to_host(device_logits)[0, -1].clone()
            ttnn.deallocate(device_logits)
            return host

        # ------------------------------------------------------------------ eager arm
        eager = []
        previous = None
        for _ in range(args.rounds):
            model.reset_state()
            current = prefill_logits()
            if previous is not None:
                eager.append(stats(previous, current))
            previous = current
            host = model.prepare_decode_inputs_host(
                torch.tensor([int(torch.argmax(current).item())], dtype=torch.int32),
                torch.tensor([len(PROMPT)], dtype=torch.int32),
                host_table,
            )
            device_inputs = [ttnn.to_device(t, device=mesh) if t is not None else None for t in host]
            out = model.ttnn_decode_forward(*device_inputs)
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(out)
            for tensor in device_inputs:
                if tensor is not None:
                    ttnn.deallocate(tensor)
        report["eager"] = eager

        # ------------------------------------------------------------------ traced arm
        model.reset_state()
        generator = OrnithGenerator(
            model,
            max_batch_size=1,
            cache_context=args.context,
            sampling_mode="device",
            kv_cache=kv_cache,
            page_table=host_table,
        )
        generator.ensure_serving_traces()
        generator.ensure_sampling_trace()
        traced = []
        previous = None
        for _ in range(args.rounds):
            generator.reset()
            current = prefill_logits()
            if previous is not None:
                traced.append(stats(previous, current))
            previous = current
            tokens = torch.tensor([int(torch.argmax(current).item())], dtype=torch.int64)
            positions = torch.tensor([len(PROMPT)], dtype=torch.int64)
            generator.stage_serving_decode_inputs(tokens, positions, host_table, full_refresh=True)
            generator.submit_serving_decode(sample_on_device=True)
            generator.read_tokens()
        report["traced"] = traced
        report["trace_recaptures"] = generator.trace_recaptures
        report["summary"] = {
            "eager_bit_stable": all(r["bitwise_identical"] for r in eager),
            "traced_bit_stable": all(r["bitwise_identical"] for r in traced),
            "eager_max_abs_diff": max((r["max_abs_diff"] for r in eager), default=None),
            "traced_max_abs_diff": max((r["max_abs_diff"] for r in traced), default=None),
            "eager_top1_stable": all(r["top1_left"] == r["top1_right"] for r in eager),
            "traced_top1_stable": all(r["top1_left"] == r["top1_right"] for r in traced),
        }
        generator.teardown()
        ttnn.deallocate(table_tt)
    finally:
        close_ornith_mesh(mesh)

    Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
    logger.info(f"wrote {args.output}")
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
