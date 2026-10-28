# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Batched-equals-solo prefill probe on one chip: logits, hybrid state, KV pages, greedy continuation, timing."""

import argparse
import bz2
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

import ttnn

from ..tt.functional_decoder import num_blocks_for_context
from ..tt.generator_vllm import TTOrnithForCausalLM
from ..tt.model import close_ornith_mesh, open_ornith_mesh


def pcc(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    a, b = a - a.mean(), b - b.mean()
    denominator = a.norm() * b.norm()
    return float((a @ b) / denominator) if denominator > 0 else 1.0


def read_whole(tensor):
    """Host copy of a whole device tensor; keeps probe reads free of per-slot slice programs."""
    return ttnn.to_torch(tensor).float()


class Arm:
    """One prefill arm over a group of rows; captures everything the acceptance compares."""

    def __init__(self, adapter, cache, table, gdn_layer, attn_layer):
        self.adapter, self.cache, self.table = adapter, cache, table
        self.gen, self.model = adapter.generator, adapter.model
        self.gdn_layer, self.attn_layer = gdn_layer, attn_layer

    def prefill(self, prompts, lengths, slots, *, batched):
        gen, model = self.gen, self.model
        model.batched_prefill["enabled"] = batched
        programs = self.adapter.mesh_device.num_program_cache_entries()
        captures = gen.counters.get("captures", 0)
        groups_before = model.counters.get("prefill_batched_groups", 0)
        t0 = time.perf_counter()
        logits = gen.prefill_forward(
            prompts,
            page_table=self.table,
            kv_cache=self.cache,
            prompt_lens=lengths,
            slots=slots,
            start_pos=[0] * len(slots),
            return_device_logits=True,
        )
        ttnn.synchronize_device(self.adapter.mesh_device)
        elapsed = time.perf_counter() - t0
        programs_added = self.adapter.mesh_device.num_program_cache_entries() - programs
        host = gen.logits_from(logits)
        if logits is not gen._logits:
            ttnn.deallocate(logits)
        rows = {}
        recurrent = read_whole(self.gdn_layer.recurrent_state)
        conv = [read_whole(buf) for buf in self.gdn_layer.conv_state]
        k_cache = read_whole(self.attn_layer.k_cache)
        v_cache = read_whole(self.attn_layer.v_cache)
        for slot, length in zip(slots, lengths):
            blocks = self.table[slot][: (length + 63) // 64].tolist()
            rows[slot] = {
                "logits": host[slot].clone(),
                "recurrent": recurrent[slot].clone(),
                "conv": [buf[slot].clone() for buf in conv],
                "k": k_cache[blocks].permute(0, 2, 1, 3).reshape(len(blocks) * 64, -1)[:length].clone(),
                "v": v_cache[blocks].permute(0, 2, 1, 3).reshape(len(blocks) * 64, -1)[:length].clone(),
            }
        del recurrent, conv, k_cache, v_cache
        return {
            "elapsed_s": elapsed,
            "programs_added": programs_added,
            "captures_added": gen.counters.get("captures", 0) - captures,
            "groups_added": model.counters.get("prefill_batched_groups", 0) - groups_before,
            "rows": rows,
        }

    def continuation(self, first_tokens, lengths, slots, steps):
        gen = self.gen
        batch = gen.max_batch_size
        tokens = torch.zeros(batch, dtype=torch.int64)
        positions = torch.full((batch,), -1, dtype=torch.int64)
        refresh = torch.zeros(batch, dtype=torch.bool)
        for slot, token, length in zip(slots, first_tokens, lengths):
            tokens[slot], positions[slot], refresh[slot] = token, length, True
        outputs = {slot: [int(token)] for slot, token in zip(slots, first_tokens)}
        for step in range(steps):
            gen.refresh_serving_inputs(tokens, positions, refresh)
            logits = gen.decode_forward(
                None, None, page_table=self.table, kv_cache=self.cache, return_logits=True, read_from_device=True
            )
            for slot in slots:
                token = int(logits[slot].argmax())
                outputs[slot].append(token)
                tokens[slot] = token
                positions[slot] += 1
        return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("/home/hous/dev/ornith-1.5-9b/upstream"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lengths", default="5,31,32,33,127,128,129,257,640,1000,1024")
    parser.add_argument("--group-sizes", default="2,4,8,32")
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--calibrate", default="", help="rows x seq shapes to time, e.g. 32x128,16x256")
    parser.add_argument("--chat-prompts", type=int, default=3)
    args = parser.parse_args()
    os.environ["ORNITH_MODEL_PATH"] = str(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    book = Path(__file__).parents[4] / "models/tt_transformers/tests/tale-of-two-cities.txt.bz2"
    base = tokenizer(bz2.open(book, "rt").read()[:400000], add_special_tokens=False)["input_ids"]
    hf_config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    generator = torch.Generator().manual_seed(1234)
    mesh = open_ornith_mesh(num_devices=1)
    report = {"status": "running", "batch": args.batch, "groups": [], "calibration": []}
    adapter = None
    try:
        adapter = TTOrnithForCausalLM.initialize_vllm_model(
            hf_config, mesh, max_batch_size=args.batch, max_seq_len=262144
        )
        model = adapter.model
        report["batched_policy"] = dict(model.batched_prefill)
        width = num_blocks_for_context(adapter.max_model_len, model.page_block_size)
        physical_blocks = int(os.environ.get("ORNITH_VLLM_POOL_TOKENS", "262208")) // model.page_block_size
        heads = max(1, model.hf_config.num_key_value_heads // mesh.get_num_devices())
        cache = adapter.allocate_kv_cache(
            (physical_blocks, heads, model.page_block_size, model.hf_config.head_dim), torch.bfloat16, len(model.layers)
        )
        t0 = time.perf_counter()
        adapter.warmup_model_prefill(cache)
        adapter.warmup_model_decode(cache)
        report["warmup_s"] = time.perf_counter() - t0
        report["prefill_warmup"] = adapter.prefill_warmup
        report["programs_after_warmup"] = int(mesh.num_program_cache_entries())
        gdn_layer = next(layer for layer in cache.decode_layers if not layer.is_full_attention)
        attn_layer = next(layer for layer in cache.decode_layers if layer.is_full_attention)
        blocks_per_row = 40
        table = torch.zeros(args.batch, width, dtype=torch.int32)
        for row in range(args.batch):
            table[row, :blocks_per_row] = torch.arange(1 + row * blocks_per_row, 1 + (row + 1) * blocks_per_row)
        arm = Arm(adapter, cache, table, gdn_layer, attn_layer)
        gen = adapter.generator
        lengths = [int(v) for v in args.lengths.split(",") if v]
        chat = []
        for index, text in enumerate(
            ["Explain why the sky is blue.", "Write a haiku about the sea.", "What is 17 times 23?"][
                : args.chat_prompts
            ]
        ):
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
            )
            chat.append(("chat", tokenizer(rendered, add_special_tokens=False)["input_ids"]))

        def prompt_for(kind, length, seed):
            if kind == "random":
                return torch.randint(0, model.vocab_size, (length,), generator=generator).tolist()
            return base[seed * 13 : seed * 13 + length]

        for group_size in [int(v) for v in args.group_sizes.split(",") if v]:
            if group_size > args.batch:
                continue
            for order in (0, 1):
                ordered = lengths if order == 0 else list(reversed(lengths))
                rows = []
                for index in range(group_size):
                    kind = "random" if index % 2 == 0 else "text"
                    length = ordered[index % len(ordered)]
                    rows.append((kind, prompt_for(kind, length, index + 7 * group_size)))
                if order == 0 and group_size >= len(chat):
                    for index, item in enumerate(chat):
                        rows[-1 - index] = item
                prompts = [row[1] for row in rows]
                lens = [len(prompt) for prompt in prompts]
                slots = list(range(group_size))
                gen.reset(clear_kv=True)
                solo_rows, solo_time = {}, 0.0
                solo_programs = 0
                for prompt, length, slot in zip(prompts, lens, slots):
                    result = arm.prefill([prompt], [length], [slot], batched=False)
                    solo_rows.update(result["rows"])
                    solo_time += result["elapsed_s"]
                    solo_programs += result["programs_added"]
                solo_first = [int(solo_rows[slot]["logits"].argmax()) for slot in slots]
                solo_tokens = arm.continuation(solo_first, lens, slots, args.decode_steps)
                gen.reset(clear_kv=True)
                result = arm.prefill(prompts, lens, slots, batched=True)
                batched_rows = result["rows"]
                batched_first = [int(batched_rows[slot]["logits"].argmax()) for slot in slots]
                batched_tokens = arm.continuation(batched_first, lens, slots, args.decode_steps)
                comparison = []
                for slot, length, (kind, _) in zip(slots, lens, rows):
                    a, b = solo_rows[slot], batched_rows[slot]
                    agree = next(
                        (i for i, (x, y) in enumerate(zip(solo_tokens[slot], batched_tokens[slot])) if x != y),
                        len(solo_tokens[slot]),
                    )
                    comparison.append(
                        {
                            "slot": slot,
                            "kind": kind,
                            "length": length,
                            "batched_path": length >= 3
                            and model.padded_prefill_length(length) <= model.batched_prefill["max_len"],
                            "top1_equal": int(a["logits"].argmax()) == int(b["logits"].argmax()),
                            "logits_pcc": pcc(a["logits"], b["logits"]),
                            "logits_max_abs_diff": float((a["logits"] - b["logits"]).abs().max()),
                            "recurrent_max_abs_diff": float((a["recurrent"] - b["recurrent"]).abs().max()),
                            "recurrent_exact": bool(torch.equal(a["recurrent"], b["recurrent"])),
                            "conv_exact": all(torch.equal(x, y) for x, y in zip(a["conv"], b["conv"])),
                            "conv_max_abs_diff": max(float((x - y).abs().max()) for x, y in zip(a["conv"], b["conv"])),
                            "k_exact": bool(torch.equal(a["k"], b["k"])),
                            "v_exact": bool(torch.equal(a["v"], b["v"])),
                            "k_max_abs_diff": float((a["k"] - b["k"]).abs().max()),
                            "v_max_abs_diff": float((a["v"] - b["v"]).abs().max()),
                            "greedy_agree_tokens": agree,
                            "greedy_solo": solo_tokens[slot],
                            "greedy_batched": batched_tokens[slot],
                        }
                    )
                entry = {
                    "group_size": group_size,
                    "order": order,
                    "lengths": lens,
                    "solo_prefill_s": solo_time,
                    "batched_prefill_s": result["elapsed_s"],
                    "solo_programs_added": solo_programs,
                    "batched_programs_added": result["programs_added"],
                    "batched_captures_added": result["captures_added"],
                    "batched_groups": result["groups_added"],
                    "rows": comparison,
                    "all_top1_equal": all(c["top1_equal"] for c in comparison),
                    "min_logits_pcc": min(c["logits_pcc"] for c in comparison),
                    "min_greedy_agree": min(c["greedy_agree_tokens"] for c in comparison),
                    "all_kv_exact": all(c["k_exact"] and c["v_exact"] for c in comparison),
                    "all_state_exact": all(c["recurrent_exact"] and c["conv_exact"] for c in comparison),
                }
                print(json.dumps({k: v for k, v in entry.items() if k != "rows"}), flush=True)
                report["groups"].append(entry)
        for shape in [v for v in args.calibrate.split(",") if v]:
            rows_n, seq = [int(v) for v in shape.split("x")]
            if rows_n > args.batch:
                continue
            prompts = [base[i * 17 : i * 17 + seq - 1] for i in range(rows_n)]
            lens = [len(p) for p in prompts]
            slots = list(range(rows_n))
            timings = {"shape": [rows_n, seq], "solo_s": [], "batched_s": []}
            for repeat in range(2):
                gen.reset(clear_kv=True)
                total = 0.0
                for prompt, length, slot in zip(prompts, lens, slots):
                    total += arm.prefill([prompt], [length], [slot], batched=False)["elapsed_s"]
                timings["solo_s"].append(total)
                gen.reset(clear_kv=True)
                result = arm.prefill(prompts, lens, slots, batched=True)
                timings["batched_s"].append(result["elapsed_s"])
                timings["batched_groups"] = result["groups_added"]
                timings["batched_programs_added"] = result["programs_added"]
            timings["ratio"] = min(timings["batched_s"]) / min(timings["solo_s"])
            print(json.dumps(timings), flush=True)
            report["calibration"].append(timings)
        model.batched_prefill["enabled"] = True
        report["programs_end"] = int(mesh.num_program_cache_entries())
        report["captures_end"] = int(gen.counters.get("captures", 0))
        root = Path(__file__).parents[1]
        report["source_sha256"] = {
            str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(root.glob("tt/*.py"))
        }
        report["git_head"] = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"]).decode().strip()
        report["trace_allocation_tracking"] = bool(ttnn.TRACE_ALLOC_TRACKING)
        report["layer_indices"] = os.environ.get("ORNITH_VLLM_LAYER_INDICES")
        report["status"] = "completed"
    finally:
        if adapter is not None:
            adapter.teardown()
        close_ornith_mesh(mesh)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
