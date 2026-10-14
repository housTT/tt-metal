# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

BOOK = Path("/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/runtime/readiness_check/"
            "autoregressive_prompt.txt")


def _median(values):
    return statistics.median(values) if values else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=int, choices=[1, 2, 4], required=True)
    p.add_argument("--batches", default="1,4,32")
    p.add_argument("--long-length", type=int, default=262100)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import ttnn

    from ..tt.generator import _greedy_params, build_generator
    from .mesh_util import open_mesh

    mesh = open_mesh(args.mesh)
    batches = tuple(int(x) for x in args.batches.split(","))
    result = dict(vars(args), output=str(args.output))

    def save():
        args.output.write_text(json.dumps(result, indent=1))

    try:
        start = time.perf_counter()
        gen = build_generator(Path(__file__).parents[1], mesh, decode_batches=batches, audit=True)
        result["build_seconds"] = time.perf_counter() - start
        result["load_seconds"] = gen.model.load_seconds
        from ..tt.optimized_decoder import _memory_bytes

        result["memory_after_build"] = _memory_bytes(mesh)
        from .model_census import census

        result["census_after_build"] = census(gen)
        result["kv_pool"] = dict(num_blocks=gen.kv_pool.num_blocks, block_size=gen.kv_pool.block_size,
                                 tokens=gen.kv_pool.num_blocks * gen.kv_pool.block_size,
                                 request_slots=gen.model.max_batch_size)
        tok = gen.tokenizer
        text = BOOK.read_text().strip()
        book = tok(text * 400, add_special_tokens=False).input_ids

        prompt128 = book[:128]
        ttft, decode = [], []
        for _ in range(3):
            gen.reset()
            stamps = []
            original = gen._replay

            def timed(batch, sample=True, _o=original, _s=stamps):
                _s.append(time.perf_counter())
                return _o(batch, sample)

            gen._replay = timed
            t0 = time.perf_counter()
            tokens = gen.generate(prompt128, 128)
            end = time.perf_counter()
            gen._replay = original
            ttft.append(stamps[0] - t0)
            decode.append((end - stamps[0]) / (len(tokens) - 1))
        result["perf_128_128"] = dict(ttft_ms=_median(ttft) * 1e3, token_out_ms_per_token=_median(decode) * 1e3,
                                      token_out_tokens_per_second_per_user=1.0 / _median(decode),
                                      runs_ttft_ms=[x * 1e3 for x in ttft], runs_decode_ms=[x * 1e3 for x in decode])
        trace_id, _ = gen._decode_traces[1]
        samples = []
        for _ in range(64):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            samples.append(time.perf_counter() - t0)
        result["logits_only_decode_ms"] = _median(samples) * 1e3
        save()

        gen.reset()
        first = gen.generate(prompt128, 32)
        gen.reset()
        second = gen.generate(prompt128, 32)
        table = gen.default_page_table(1, 300)
        logits_a = gen.prefill_forward(torch.tensor([book[:300]]), page_table=table, prompt_lens=[300])
        logits_b = gen.prefill_forward(torch.tensor([book[:300]]), page_table=table, prompt_lens=[300])
        result["determinism"] = dict(greedy_repeat_identical=first == second,
                                     prefill_logits_bitwise=bool(torch.equal(logits_a, logits_b)))
        save()

        lengths = [5, 127, 129, 300, 1000, 4095, 4097, 8191, 8193, 12000]
        singles = {}
        for length in lengths:
            gen.reset()
            singles[length] = gen.generate(book[:length], 8)
        result["nonaligned_single"] = {str(k): v for k, v in singles.items()}
        for batch in batches:
            rows = [lengths[i % len(lengths)] for i in range(batch)]
            if batch == 1:
                continue
            gen.reset()
            counts = [-(-(length + 8) // 32) for length in rows]
            table = torch.full((batch, max(counts)), gen.kv_pool.scratch_block, dtype=torch.int32)
            first_block = 0
            for r, count in enumerate(counts):
                table[r, :count] = torch.arange(first_block, first_block + count, dtype=torch.int32)
                first_block += count
            tokens = torch.zeros(batch, max(rows), dtype=torch.int64)
            for r, length in enumerate(rows):
                tokens[r, :length] = torch.tensor(book[:length])
            logits = gen.prefill_forward(tokens, page_table=table, prompt_lens=rows)
            current = logits[:, 0].argmax(-1)
            outs = [[int(t)] for t in current]
            positions = torch.tensor(rows)
            for _ in range(7):
                current = gen.decode_forward(current.reshape(batch, 1), positions, page_table=table,
                                             sampling_params=_greedy_params())
                positions = positions + 1
                for r in range(batch):
                    outs[r].append(int(current[r]))
            match = [outs[r] == singles[rows[r]] for r in range(batch)]
            result[f"batch_{batch}"] = dict(rows=rows, match_batch1=match, all_match=all(match))
            print(json.dumps({f"batch_{batch}": result[f"batch_{batch}"]}), flush=True)
            save()

        continuation = []
        for start, length in ((128, 300), (4224, 77), (8192, 1), (8320, 4097), (12288, 129)):
            total = start + length
            table = gen.default_page_table(1, total + 1)
            gen.reset()
            whole = gen.prefill_forward(torch.tensor([book[:total]]), page_table=table, prompt_lens=[total])[0, 0]
            gen.reset()
            gen.prefill_forward(torch.tensor([book[:start]]), page_table=table, prompt_lens=[start])
            split = gen.prefill_forward(torch.tensor([book[start:total]]), page_table=table, prompt_lens=[length],
                                        start_pos=[start])[0, 0]
            pcc = float(torch.corrcoef(torch.stack([whole.double(), split.double()]))[0, 1])
            continuation.append(dict(start=start, length=length, pcc=pcc,
                                     same_argmax=int(whole.argmax()) == int(split.argmax())))
        rejected = {}
        for start in (32, 8224):
            try:
                gen.prefill_forward(torch.tensor([book[:16]]), page_table=table, prompt_lens=[16], start_pos=[start])
                rejected[str(start)] = "accepted"
            except ValueError as error:
                rejected[str(start)] = f"ValueError: {error}"
        result["continuation"] = dict(cases=continuation, unaligned_start=rejected)
        print(json.dumps({"continuation": result["continuation"]}), flush=True)
        save()

        gen.reset()
        long_prompt = (book * (args.long_length // len(book) + 1))[: args.long_length]
        t0 = time.perf_counter()
        long_tokens = gen.generate(long_prompt, 16)
        result["long"] = dict(length=args.long_length, seconds=time.perf_counter() - t0, tokens=long_tokens,
                              text=tok.decode(long_tokens))
        gen.reset()
        boundary = (book * (262143 // len(book) + 1))[:262143]
        boundary_tokens = gen.generate(boundary, 16)
        result["boundary"] = dict(length=262143, requested=16, returned=len(boundary_tokens), tokens=boundary_tokens)
        result["counters"] = dict(gen.counters)
        result["pad_fallbacks"] = gen.model.pad_fallbacks
        print(json.dumps({k: v for k, v in result.items() if k not in ("nonaligned_single",)}), flush=True)
    finally:
        save()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
