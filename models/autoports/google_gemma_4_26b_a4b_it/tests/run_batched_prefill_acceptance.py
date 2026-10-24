# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generator-level batched-equals-solo acceptance and bucket calibration for batched prefill.

Usage (one chip, from the tt-metal worktree):
    TT_VISIBLE_DEVICES=1 ... python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_batched_prefill_acceptance \
        --phase both --output-dir models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import (
    BATCHED_PREFILL_BUCKETS,
    _padded_prefill_len,
    batched_prefill_variants,
    build_generator,
    plan_prefill_groups,
)
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm import PRECISION_CONFIG
from models.autoports.google_gemma_4_26b_a4b_it.tt.model import DECODE_SLOT_COUNT

MODEL_ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_LENGTHS = (5, 31, 32, 33, 127, 128, 129, 257, 640, 1000, 1024)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten() - a.float().mean()
    b = b.float().flatten() - b.float().mean()
    denom = float(a.norm() * b.norm())
    return float(a @ b) / denom if denom else 1.0


def _host_logits(gen, logits) -> torch.Tensor:
    host = gen._gather_logits_to_torch(logits)
    ttnn.deallocate(logits)
    return host.reshape(-1, host.shape[-1]).float()


def _chat_prompts(tokenizer, prompts_file: Path, limit: int) -> list[list[int]]:
    pieces = [p.strip() for p in prompts_file.read_text().split("\n\n") if p.strip()]
    prompts = []
    for piece in pieces[:limit]:
        try:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": piece}], add_generation_prompt=True, tokenize=True
            )
            if hasattr(ids, "keys"):
                ids = ids["input_ids"]
            ids = list(ids[0]) if ids and isinstance(ids[0], (list, tuple)) else list(ids)
        except Exception:
            ids = list(tokenizer.encode(piece))
        prompts.append([int(v) for v in ids])
    return prompts


def _cache_rows(state, state_idx: int, rows: list[int], lens: list[int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    spec = state.cache_specs[state_idx]
    table = state.host_page_tables[state_idx]
    key_cache, value_cache = state.kv_cache[state_idx]
    k_host = ttnn.to_torch(ttnn.get_device_tensors(key_cache)[0])
    v_host = ttnn.to_torch(ttnn.get_device_tensors(value_cache)[0])
    result = []
    for row, length in zip(rows, lens):
        blocks = (length + spec.block_size - 1) // spec.block_size
        block_ids = [int(table[row][column]) for column in range(blocks)]
        k = torch.cat([k_host[b] for b in block_ids], dim=1)[:, :length, :].clone()
        v = torch.cat([v_host[b] for b in block_ids], dim=1)[:, :length, :].clone()
        result.append((k, v))
    return result


def _greedy_continuation(gen, state, row: int, batch: int, first_token: int, position: int, steps: int) -> list[int]:
    tokens = [first_token]
    for _ in range(steps):
        mask = torch.zeros(batch, dtype=torch.bool)
        mask[row] = True
        host_tokens = torch.zeros((batch, 1), dtype=torch.long)
        host_tokens[row, 0] = tokens[-1]
        positions = torch.full((batch,), -1, dtype=torch.int32)
        positions[row] = position + len(tokens) - 1
        sampled = gen.decode_forward(
            host_tokens,
            positions,
            page_table=state.page_tables,
            kv_cache=state,
            sampling_mode="device",
            enable_trace=True,
            active_mask=mask,
        )
        tokens.append(int(gen._read_tokens(sampled, batch)[row]))
    return tokens[1:]


def run_calibration(gen, state, *, repeats: int) -> dict:
    device = gen.mesh_device
    report = {"single_row_seconds": {}, "batched": [], "buckets": list(BATCHED_PREFILL_BUCKETS)}

    def timed(call):
        best = None
        for _ in range(repeats + 1):
            ttnn.synchronize_device(device)
            began = time.perf_counter()
            logits = call()
            ttnn.synchronize_device(device)
            elapsed = time.perf_counter() - began
            ttnn.deallocate(logits)
            best = elapsed if best is None else min(best, elapsed)
        return best

    for segment in BATCHED_PREFILL_BUCKETS:
        tokens = torch.randint(1, 262_000, (1, segment), dtype=torch.long)

        def single():
            gen.reset()
            gen.batched_prefill_enabled = False
            return gen.prefill_forward(tokens, page_table=state.page_tables, kv_cache=state, prompt_lens=[segment])

        report["single_row_seconds"][str(segment)] = round(timed(single), 4)
    for batch, segment in batched_prefill_variants():
        if batch > state.max_batch_size:
            continue
        tokens = torch.randint(1, 262_000, (batch, segment), dtype=torch.long)

        def batched():
            gen.reset()
            gen.batched_prefill_enabled = True
            return gen.prefill_forward(
                tokens, page_table=state.page_tables, kv_cache=state, prompt_lens=[segment] * batch
            )

        def serial():
            gen.reset()
            gen.batched_prefill_enabled = False
            return gen.prefill_forward(
                tokens, page_table=state.page_tables, kv_cache=state, prompt_lens=[segment] * batch
            )

        batched_seconds = timed(batched)
        serial_seconds = timed(serial)
        entry = {
            "batch": batch,
            "segment_len": segment,
            "tokens": batch * segment,
            "batched_seconds": round(batched_seconds, 4),
            "serial_rows_seconds": round(serial_seconds, 4),
            "ratio_batched_over_serial": round(batched_seconds / serial_seconds, 4) if serial_seconds else None,
        }
        report["batched"].append(entry)
        print(json.dumps(entry), flush=True)
    gen.batched_prefill_enabled = True
    return report


def run_acceptance(gen, state, *, batches: list[int], decode_steps: int, prompts_file: Path, seed: int) -> dict:
    device = gen.mesh_device
    rng = torch.Generator().manual_seed(seed)
    sliding_idx = next(i for i, spec in enumerate(state.cache_specs) if spec.layer_type == "sliding_attention")
    full_idx = next(i for i, spec in enumerate(state.cache_specs) if spec.layer_type == "full_attention")
    chat = [
        ids for ids in _chat_prompts(gen.tokenizer, prompts_file, 4) if 1 <= len(ids) <= max(BATCHED_PREFILL_BUCKETS)
    ]
    cases = []
    for batch in batches:
        for order in ("ascending", "descending"):
            lens = [ACCEPTANCE_LENGTHS[i % len(ACCEPTANCE_LENGTHS)] for i in range(batch)]
            lens = sorted(lens, reverse=(order == "descending"))
            prompts = [torch.randint(1, 262_000, (length,), generator=rng).tolist() for length in lens]
            sources = ["random"] * batch
            for slot, ids in zip(range(1, batch, max(1, batch // 2)), chat):
                prompts[slot] = ids
                lens[slot] = len(ids)
                sources[slot] = "chat"
            width = max(_padded_prefill_len(length) for length in lens)
            tokens = torch.zeros((batch, width), dtype=torch.long)
            for row, ids in enumerate(prompts):
                tokens[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            groups, singles = plan_prefill_groups(lens, [0] * batch, enabled=True)
            arms = {}
            for arm in ("solo", "batched"):
                gen.reset()
                gen.batched_prefill_enabled = arm == "batched"
                entries_before = device.num_program_cache_entries()
                group_records_before = len(gen.batched_prefill_group_growth)
                began = time.perf_counter()
                logits = gen.prefill_forward(tokens, page_table=state.page_tables, kv_cache=state, prompt_lens=lens)
                ttnn.synchronize_device(device)
                seconds = time.perf_counter() - began
                entries_after = device.num_program_cache_entries()
                group_growth = [
                    g["program_cache_growth"] for g in gen.batched_prefill_group_growth[group_records_before:]
                ]
                host = _host_logits(gen, logits)
                caches = {
                    "sliding": _cache_rows(state, sliding_idx, list(range(batch)), lens),
                    "full": _cache_rows(state, full_idx, list(range(batch)), lens),
                }
                continuations = [
                    _greedy_continuation(gen, state, row, batch, int(host[row].argmax()), lens[row], decode_steps)
                    for row in range(batch)
                ]
                arms[arm] = {
                    "logits": host,
                    "caches": caches,
                    "continuations": continuations,
                    "seconds": seconds,
                    "program_cache_growth": entries_after - entries_before,
                    "group_growth": group_growth,
                }
            rows = []
            for row in range(batch):
                solo_logits = arms["solo"]["logits"][row]
                batched_logits = arms["batched"]["logits"][row]
                kv_equal = {}
                kv_max_abs = {}
                for kind in ("sliding", "full"):
                    k_solo, v_solo = arms["solo"]["caches"][kind][row]
                    k_batched, v_batched = arms["batched"]["caches"][kind][row]
                    kv_equal[kind] = bool(torch.equal(k_solo, k_batched) and torch.equal(v_solo, v_batched))
                    kv_max_abs[kind] = float(
                        max(
                            (k_solo.float() - k_batched.float()).abs().max(),
                            (v_solo.float() - v_batched.float()).abs().max(),
                        )
                    )
                solo_cont = arms["solo"]["continuations"][row]
                batched_cont = arms["batched"]["continuations"][row]
                divergence = next((i for i, (a, b) in enumerate(zip(solo_cont, batched_cont)) if a != b), None)
                rows.append(
                    {
                        "row": row,
                        "length": lens[row],
                        "source": sources[row],
                        "top1_equal": bool(solo_logits.argmax() == batched_logits.argmax()),
                        "pcc": round(_pcc(solo_logits, batched_logits), 6),
                        "kv_equal": kv_equal,
                        "kv_max_abs": kv_max_abs,
                        "continuation_divergence_index": divergence,
                        "continuation_identical_through_8": divergence is None or divergence >= 8,
                    }
                )
            case = {
                "batch": batch,
                "order": order,
                "lengths": lens,
                "groups": [
                    {"rows": list(g.rows), "segment_len": g.segment_len, "padded_batch": g.padded_batch} for g in groups
                ],
                "single_rows": singles,
                "solo_seconds": round(arms["solo"]["seconds"], 4),
                "batched_seconds": round(arms["batched"]["seconds"], 4),
                "batched_program_cache_growth": arms["batched"]["program_cache_growth"],
                "batched_group_program_cache_growth": arms["batched"]["group_growth"],
                "single_row_program_cache_growth": arms["batched"]["program_cache_growth"]
                - sum(arms["batched"]["group_growth"]),
                "solo_program_cache_growth": arms["solo"]["program_cache_growth"],
                "rows": rows,
                "all_top1_equal": all(r["top1_equal"] for r in rows),
                "min_pcc": min(r["pcc"] for r in rows),
                "all_kv_equal": all(all(r["kv_equal"].values()) for r in rows),
                "all_continuations_identical_through_8": all(r["continuation_identical_through_8"] for r in rows),
            }
            cases.append(case)
            print(
                json.dumps({k: v for k, v in case.items() if k != "rows"}),
                flush=True,
            )
    status = (
        "pass"
        if all(
            c["all_top1_equal"]
            and c["min_pcc"] >= 0.999
            and c["all_kv_equal"]
            and c["all_continuations_identical_through_8"]
            and sum(c["batched_group_program_cache_growth"]) == 0
            for c in cases
        )
        else "fail"
    )
    return {"status": status, "decode_steps": decode_steps, "lengths": list(ACCEPTANCE_LENGTHS), "cases": cases}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("calibrate", "acceptance", "both"), default="both")
    parser.add_argument("--batches", type=str, default="2,4,8,32")
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=MODEL_ROOT / "readiness_vllm")
    parser.add_argument("--prompts-file", type=Path, default=None)
    parser.add_argument("--max-batch-size", type=int, default=DECODE_SLOT_COUNT)
    parser.add_argument("--max-seq-len", type=int, default=None)
    args = parser.parse_args()
    prompts_file = args.prompts_file
    if prompts_file is None:
        import readiness_check.run_prefill_resume_check as module

        prompts_file = Path(module.__file__).parent / "vllm_prompts.txt"
    layer_indices = None if args.layers == "all" else [int(v) for v in args.layers.split(",")]
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    report = {"layers": args.layers, "max_batch_size": args.max_batch_size, "max_seq_len": args.max_seq_len}
    try:
        began = time.perf_counter()
        gen = build_generator(
            MODEL_ROOT,
            mesh,
            layer_indices=layer_indices,
            max_batch_size=args.max_batch_size,
            max_seq_len=args.max_seq_len,
            precision_config_path=str(PRECISION_CONFIG),
            tensor_cache_path=os.environ.get("GEMMA4_TENSOR_CACHE_PATH", "/tmp/gemma4_full_model_cache"),
        )
        state = gen.model.state
        print(f"generator built in {time.perf_counter() - began:.1f}s", flush=True)
        warmup = gen.prepare_batched_prefill(state)
        report["warmup"] = warmup
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.phase in ("calibrate", "both"):
            calibration = run_calibration(gen, state, repeats=args.repeats)
            calibration["warmup"] = warmup
            (args.output_dir / "batched_prefill_calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
            report["calibration"] = calibration
        if args.phase in ("acceptance", "both"):
            acceptance = run_acceptance(
                gen,
                state,
                batches=[int(v) for v in args.batches.split(",")],
                decode_steps=args.decode_steps,
                prompts_file=prompts_file,
                seed=args.seed,
            )
            acceptance["warmup"] = warmup
            acceptance["program_cache_entries_after_workload"] = mesh.num_program_cache_entries()
            (args.output_dir / "vllm_batched_equals_solo.json").write_text(json.dumps(acceptance, indent=2) + "\n")
            report["acceptance_status"] = acceptance["status"]
            print(f"acceptance status: {acceptance['status']}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    print(json.dumps({k: v for k, v in report.items() if k not in ("calibration",)}, default=str)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
