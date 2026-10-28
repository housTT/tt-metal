# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""One-chip adapter probe: program-cache growth and trace captures across prompt lengths."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

import ttnn
from models.common.sampling import SamplingParams

from ..tt.functional_decoder import num_blocks_for_context
from ..tt.generator_vllm import TTOrnithForCausalLM
from ..tt.model import close_ornith_mesh, open_ornith_mesh


def snapshot(adapter, label, **extra):
    gen = adapter.generator
    row = {
        "label": label,
        "programs": int(adapter.mesh_device.num_program_cache_entries()),
        "captures": int(gen.counters.get("captures", 0)),
        "prefill_eager_calls": int(gen.counters.get("prefill_eager_calls", 0)),
        "prefill_replays": int(gen.counters.get("prefill_replays", 0)),
        "model_replays": int(gen.counters.get("model_replays", 0)),
        "model_trace": str(gen._model_trace),
        "active8_trace": str(gen._active8_trace),
        "sampling_trace": str(gen._sampling_trace),
    }
    row.update(extra)
    print(json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("/home/hous/dev/ornith-1.5-9b/upstream"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lengths", default="128,129,130,131,64,128,129,2305,2306,131")
    parser.add_argument(
        "--parity",
        action="store_true",
        help="Also run each prompt on the pre-change per-length path and compare tokens",
    )
    parser.add_argument("--decode-steps", type=int, default=1)
    args = parser.parse_args()
    os.environ["ORNITH_MODEL_PATH"] = str(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    import bz2

    book = Path(__file__).parents[4] / "models/tt_transformers/tests/tale-of-two-cities.txt.bz2"
    base = tokenizer(bz2.open(book, "rt").read()[:400000], add_special_tokens=False)["input_ids"]
    hf_config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    mesh = open_ornith_mesh(num_devices=1)
    report = {"status": "running", "batch": args.batch, "steps": []}
    adapter = None
    try:
        adapter = TTOrnithForCausalLM.initialize_vllm_model(
            hf_config, mesh, max_batch_size=args.batch, max_seq_len=262144
        )
        width = num_blocks_for_context(adapter.max_model_len, adapter.model.page_block_size)
        physical_blocks = int(os.environ.get("ORNITH_VLLM_POOL_TOKENS", "262208")) // adapter.model.page_block_size
        heads = max(1, adapter.model.hf_config.num_key_value_heads // mesh.get_num_devices())
        cache = adapter.allocate_kv_cache(
            (physical_blocks, heads, adapter.model.page_block_size, adapter.model.hf_config.head_dim),
            torch.bfloat16,
            len(adapter.model.layers),
        )
        report["physical_blocks"] = physical_blocks
        report["precision"] = adapter.model.precision
        t0 = time.perf_counter()
        adapter.warmup_model_prefill(cache)
        adapter.warmup_model_decode(cache)
        report["warmup_s"] = time.perf_counter() - t0
        report["steps"].append(snapshot(adapter, "after_warmup"))
        sampling = SamplingParams(
            temperature=[0.0] * args.batch,
            top_k=[1] * args.batch,
            top_p=[1.0] * args.batch,
            num_logprobs=[-2] * args.batch,
        )
        lengths = [int(v) for v in args.lengths.split(",")]
        qualified = adapter.model.padded_prefill_qualified
        blocks_per_row = 40
        table = torch.zeros(args.batch, width, dtype=torch.int32)
        for row in range(args.batch):
            table[row, :blocks_per_row] = torch.arange(1 + row * blocks_per_row, 1 + (row + 1) * blocks_per_row)
        for index, length in enumerate(lengths):
            rows = [index % args.batch]
            prompt = torch.tensor(base[index * 7 : index * 7 + length], dtype=torch.int64)[None]
            before = snapshot(adapter, f"before_{length}")
            t0 = time.perf_counter()
            first = adapter.prefill_forward(
                tokens=prompt,
                prompt_lens=[length],
                empty_slots=rows,
                page_table=table,
                kv_cache=cache,
                sampling_params=sampling,
            )
            prefill_s = time.perf_counter() - t0
            first = first[0] if isinstance(first, tuple) else first
            tokens = torch.zeros(args.batch, 1, dtype=torch.int64)
            tokens[rows[0], 0] = int(first.reshape(-1)[0])
            positions = torch.full((args.batch,), -1, dtype=torch.int64)
            positions[rows[0]] = length
            t0 = time.perf_counter()
            for step in range(args.decode_steps):
                out = adapter.decode_forward(
                    tokens=tokens,
                    start_pos=positions,
                    page_table=table,
                    kv_cache=cache,
                    sampling_params=sampling,
                    reset_batch=step == 0,
                    read_from_device=True,
                )
            decode_s = time.perf_counter() - t0
            after = snapshot(
                adapter,
                f"after_{length}",
                prompt_length=length,
                slot=rows[0],
                prefill_s=prefill_s,
                decode_s=decode_s,
                first_token=int(first.reshape(-1)[0]),
                decode_token=int(out.reshape(-1)[rows[0]]),
                new_programs=int(adapter.mesh_device.num_program_cache_entries()) - before["programs"],
                new_captures=int(adapter.generator.counters.get("captures", 0)) - before["captures"],
            )
            report["steps"].append(after)
            if args.parity:
                adapter.model.padded_prefill_qualified = lambda *a, **k: False
                try:
                    legacy = adapter.prefill_forward(
                        tokens=prompt,
                        prompt_lens=[length],
                        empty_slots=rows,
                        page_table=table,
                        kv_cache=cache,
                        sampling_params=sampling,
                    )
                    legacy = legacy[0] if isinstance(legacy, tuple) else legacy
                    legacy_out = adapter.decode_forward(
                        tokens=tokens,
                        start_pos=positions,
                        page_table=table,
                        kv_cache=cache,
                        sampling_params=sampling,
                        reset_batch=True,
                        read_from_device=True,
                    )
                finally:
                    adapter.model.padded_prefill_qualified = qualified
                after["legacy_first_token"] = int(legacy.reshape(-1)[0])
                after["legacy_decode_token"] = int(legacy_out.reshape(-1)[rows[0]])
                after["parity"] = (
                    after["legacy_first_token"] == after["first_token"]
                    and after["legacy_decode_token"] == after["decode_token"]
                )
                print(
                    json.dumps(
                        {
                            "parity": after["parity"],
                            "length": length,
                            "first": after["first_token"],
                            "legacy_first": after["legacy_first_token"],
                            "decode": after["decode_token"],
                            "legacy_decode": after["legacy_decode_token"],
                        }
                    ),
                    flush=True,
                )
        mixed = [131, 65, 131]
        rows = [3, 4, 5]
        prompts = [torch.tensor(base[100 + i * 11 : 100 + i * 11 + n], dtype=torch.int64) for i, n in enumerate(mixed)]
        before = snapshot(adapter, "before_mixed")
        t0 = time.perf_counter()
        first = adapter.prefill_forward(
            tokens=prompts,
            prompt_lens=mixed,
            empty_slots=rows,
            page_table=table,
            kv_cache=cache,
            sampling_params=sampling,
        )
        prefill_s = time.perf_counter() - t0
        first = first[0] if isinstance(first, tuple) else first
        tokens = torch.zeros(args.batch, 1, dtype=torch.int64)
        positions = torch.full((args.batch,), -1, dtype=torch.int64)
        for i, row in enumerate(rows):
            tokens[row, 0] = int(first.reshape(-1)[i])
            positions[row] = mixed[i]
        adapter.decode_forward(
            tokens=tokens,
            start_pos=positions,
            page_table=table,
            kv_cache=cache,
            sampling_params=sampling,
            reset_batch=True,
            read_from_device=True,
        )
        report["steps"].append(
            snapshot(
                adapter,
                "after_mixed",
                prompt_lengths=mixed,
                prefill_s=prefill_s,
                new_programs=int(adapter.mesh_device.num_program_cache_entries()) - before["programs"],
                new_captures=int(adapter.generator.counters.get("captures", 0)) - before["captures"],
            )
        )
        switch_prompt = torch.tensor(base[50000:50131], dtype=torch.int64)[None]
        modes = {
            "greedy": sampling,
            "sampled": SamplingParams(
                temperature=[0.8] * args.batch,
                top_k=[20] * args.batch,
                top_p=[0.95] * args.batch,
                seed=[7] * args.batch,
                num_logprobs=[-2] * args.batch,
            ),
            "penalized": SamplingParams(
                temperature=[0.0] * args.batch,
                top_k=[1] * args.batch,
                top_p=[1.0] * args.batch,
                presence_penalty=[0.5] * args.batch,
                num_logprobs=[-2] * args.batch,
            ),
        }
        switch_rows = []
        for name in ("greedy", "sampled", "penalized", "greedy", "sampled", "greedy"):
            before = snapshot(adapter, f"before_switch_{name}")
            first = adapter.prefill_forward(
                tokens=switch_prompt,
                prompt_lens=[131],
                empty_slots=[9],
                page_table=table,
                kv_cache=cache,
                sampling_params=modes[name],
            )
            first = first[0] if isinstance(first, tuple) else first
            tokens = torch.zeros(args.batch, 1, dtype=torch.int64)
            tokens[9, 0] = int(first.reshape(-1)[0])
            positions = torch.full((args.batch,), -1, dtype=torch.int64)
            positions[9] = 131
            outs = []
            for step in range(3):
                out = adapter.decode_forward(
                    tokens=tokens,
                    start_pos=positions,
                    page_table=table,
                    kv_cache=cache,
                    sampling_params=modes[name],
                    reset_batch=step == 0,
                    read_from_device=True,
                )
                outs.append(int(out.reshape(-1)[9]))
            row = snapshot(
                adapter,
                f"after_switch_{name}",
                mode=name,
                tokens=[int(first.reshape(-1)[0])] + outs,
                new_programs=int(adapter.mesh_device.num_program_cache_entries()) - before["programs"],
                new_captures=int(adapter.generator.counters.get("captures", 0)) - before["captures"],
                trace_releases=int(adapter.generator.counters.get("trace_releases", 0)),
            )
            switch_rows.append(row)
            report["steps"].append(row)
        greedy_runs = [row["tokens"] for row in switch_rows if row["mode"] == "greedy"]
        report["mode_switch"] = {
            "greedy_outputs_identical": all(run == greedy_runs[0] for run in greedy_runs),
            "captures_during_switches": sum(row["new_captures"] for row in switch_rows),
            "programs_during_switches": sum(row["new_programs"] for row in switch_rows),
        }
        print(json.dumps(report["mode_switch"]), flush=True)
        report["execution_info"] = adapter.generator.execution_info()
        import hashlib
        import subprocess

        root = Path(__file__).parents[1]
        report["source_sha256"] = {
            str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(root.glob("tt/*.py"))
        }
        report["git_head"] = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"]).decode().strip()
        report["git_dirty_files"] = (
            subprocess.check_output(["git", "-C", str(root), "status", "--short"]).decode().count("\n")
        )
        report["trace_allocation_tracking"] = bool(ttnn.TRACE_ALLOC_TRACKING)
        report["status"] = "completed"
    finally:
        if adapter is not None:
            adapter.teardown()
        close_ornith_mesh(mesh)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
