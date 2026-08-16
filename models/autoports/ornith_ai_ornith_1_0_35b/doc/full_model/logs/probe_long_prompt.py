# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""How long a non-aligned prompt the delivered public path really accepts, measured.

`tests/test_full_model.py::test_full_stack_non_aligned_long_prompt` runs 5003 tokens through the
complete 40-layer stack, which proves the *shape* contract but is nowhere near the advertised
262144-token context. This probe walks non-aligned lengths up as far as a wall-clock budget allows,
on the full stack with the **full advertised cache allocated**, and reports the largest length that
completed. Each length is a fresh request through the public generator: prefill, then one traced
token-out decode step, then a check that the sampled token is a valid id and the logits are finite.

    python .../doc/full_model/logs/probe_long_prompt.py [--budget-s 2400] [--cache-context 262144]

Prefill cost grows superlinearly (the ten `full_attention` layers attend to the whole cache each
chunk), so the budget - not DRAM - is what stops this; the probe prints both so the limit is not
confused with a capacity limit.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import close_ornith_mesh, open_ornith_mesh

MODEL_DIR = Path(__file__).resolve().parents[3]

#: Deliberately awkward: prime or near-prime, none a multiple of 32, 128, 2048 or the page size.
LENGTHS = [5003, 8191, 16381, 32749, 65521, 131071, 262143]


def _dram(mesh):
    """(allocated, total) DRAM bytes for one device, the same accounting probe_footprint.py uses."""
    device = mesh.get_devices()[0] if hasattr(mesh, "get_devices") else mesh
    view = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    banks = int(view.num_banks)
    return int(view.total_bytes_allocated_per_bank) * banks, int(view.total_bytes_per_bank) * banks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-s", type=float, default=2400.0)
    parser.add_argument("--cache-context", type=int, default=262144)
    parser.add_argument("--output", default=str(MODEL_DIR / "doc" / "full_model" / "long_prompt.json"))
    args = parser.parse_args()

    mesh = open_ornith_mesh()
    started = time.perf_counter()
    results = []
    generator = None
    last = None  # (prompt_len, prefill_s), for the quadratic projection below
    try:
        generator = build_generator(
            model_dir=str(MODEL_DIR), mesh_device=mesh, max_batch_size=1, cache_context=args.cache_context
        )
        allocated, total = _dram(mesh)
        free_after_build = total - allocated
        logger.info(f"built with cache_context={args.cache_context}; {free_after_build / 2**30:.2f} GiB DRAM free")

        torch.manual_seed(11)
        for length in LENGTHS:
            if length > args.cache_context:
                results.append({"prompt_len": length, "status": "skipped", "why": "longer than the cache"})
                continue
            remaining = args.budget_s - (time.perf_counter() - started)
            # Prefill is quadratic in the prompt (each chunk's full_attention reads the whole cache
            # so far), so project from the last measured point rather than only checking the clock
            # before starting: a length that would overrun is skipped, not begun and abandoned.
            projected = None if last is None else last[1] * (length / last[0]) ** 2
            if remaining <= 0 or (projected is not None and projected > remaining):
                results.append(
                    {
                        "prompt_len": length,
                        "status": "not attempted",
                        "why": "wall-clock budget",
                        "projected_prefill_s": None if projected is None else round(projected, 1),
                        "remaining_budget_s": round(remaining, 1),
                    }
                )
                continue

            prompt = torch.randint(0, generator.model.vocab_size, (1, length))
            generator.reset()
            t0 = time.perf_counter()
            logits = generator.prefill_forward(prompt, page_table=None, kv_cache=None, prompt_lens=[length])
            prefill_s = time.perf_counter() - t0

            token = generator.decode_forward(
                torch.tensor([int(torch.argmax(logits[0, 0]))]),
                torch.tensor([length]),
                enable_trace=True,
                sample_on_device=True,
            )
            allocated, total = _dram(mesh)
            free_now = total - allocated
            row = {
                "prompt_len": length,
                "status": "ok",
                "prefill_s": round(prefill_s, 2),
                "prefill_tokens_per_s": round(length / prefill_s, 1),
                "finite_logits": bool(torch.isfinite(logits).all()),
                "decoded_token": int(token[0]),
                "token_in_vocab": 0 <= int(token[0]) < generator.model.vocab_size,
                "dram_free_gib": round(free_now / 2**30, 2),
            }
            assert row["finite_logits"], f"non-finite prefill logits at {length}"
            assert row["token_in_vocab"], f"decoded token out of range at {length}"
            results.append(row)
            last = (length, prefill_s)
            logger.info(f"prompt_len {length}: {row}")

        largest = max((r["prompt_len"] for r in results if r["status"] == "ok"), default=None)
        report = {
            "cache_context": args.cache_context,
            "budget_s": args.budget_s,
            "elapsed_s": round(time.perf_counter() - started, 1),
            "dram_free_after_build_gib": round(free_after_build / 2**30, 2),
            "largest_completed_non_aligned_prompt": largest,
            "results": results,
        }
        Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps(report, indent=2))
        print("\nPROBE_OK")
    finally:
        try:
            if generator is not None:
                generator.teardown()
        except Exception:  # noqa: BLE001 - teardown must not mask the probe's own result
            pass
        close_ornith_mesh(mesh)


if __name__ == "__main__":
    main()
