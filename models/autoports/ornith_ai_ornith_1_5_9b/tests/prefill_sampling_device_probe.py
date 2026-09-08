# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reduced TP4 prefill sampler proof, run only in the supervising hardware lane.

Uses real layers 0/3, a native-context external cache, actual model prefill, and
exact production sampler shapes. Synthetic logits isolate sampling numerics.
Imports are deferred so --help and compilation do not touch TTNN or hardware.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, choices=(1, 32), required=True)
    args = parser.parse_args()

    import torch

    import ttnn
    from models.common.sampling import SamplingParams

    from ..tt.functional_decoder import num_blocks_for_context
    from ..tt.generator import OrnithGenerator
    from ..tt.model import OrnithModel, close_ornith_mesh, open_ornith_mesh

    if not ttnn.TRACE_ALLOC_TRACKING or os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "0":
        raise RuntimeError("Require TT_METAL_TRACE_ALLOC_TRACKING=1 with program-cache allocations included")

    def read_all(tensor):
        return [ttnn.to_torch(shard).to(torch.int64) for shard in ttnn.get_device_tensors(tensor)]

    def memory():
        result = {}
        for name in ("DRAM", "L1", "TRACE"):
            view = ttnn.get_memory_view(mesh, getattr(ttnn.BufferType, name))
            result[name] = {
                key: int(getattr(view, key))
                for key in (
                    "num_banks",
                    "total_bytes_per_bank",
                    "total_bytes_allocated_per_bank",
                    "total_bytes_free_per_bank",
                    "largest_contiguous_bytes_free_per_bank",
                )
            }
            result[name]["total_allocated_bytes_per_device"] = int(view.num_banks * view.total_bytes_allocated_per_bank)
        return result

    def buffer_receipt(name, tensor):
        item_bytes = {ttnn.bfloat16: 2, ttnn.int32: 4, ttnn.uint32: 4, ttnn.float32: 4}[tensor.dtype]
        return {
            "name": name,
            "shape": list(tensor.shape),
            "padded_shape": list(tensor.padded_shape),
            "dtype": str(tensor.dtype),
            "layout": str(tensor.layout),
            "address": tensor.buffer_address(),
            "logical_element_bytes_global": math.prod(tensor.shape) * item_bytes,
            "padded_element_bytes_global": math.prod(tensor.padded_shape) * item_bytes,
            "ranks": [
                {
                    "shape": list(shard.shape),
                    "padded_shape": list(shard.padded_shape),
                    "logical_element_bytes": math.prod(shard.shape) * item_bytes,
                    "padded_element_bytes": math.prod(shard.padded_shape) * item_bytes,
                }
                for shard in ttnn.get_device_tensors(tensor)
            ],
        }

    def trace_bytes():
        view = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
        return int(view.num_banks * view.total_bytes_allocated_per_bank)

    def report_now():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    report = {
        "batch": args.batch,
        "layers": [0, 3],
        "native_tracking_includes_program_cache": True,
        "scope": "Reduced correctness and lifetime check; no full-model accuracy or performance claim",
        "source_sha256": {
            name: hashlib.sha256((Path(__file__).parents[1] / "tt" / name).read_bytes()).hexdigest()
            for name in ("generator.py", "model.py")
        },
        "cases": [],
    }
    mesh, gen = None, None
    report_now()
    try:
        mesh = open_ornith_mesh()
        model = OrnithModel(args.model_path, mesh, layer_indices=[0, 3])
        width = num_blocks_for_context(model.max_context, model.page_block_size)
        request_blocks = num_blocks_for_context(256, model.page_block_size)
        pool_blocks = width + request_blocks * args.batch
        cache = model.allocate_cache(args.batch, num_blocks=pool_blocks)
        table = torch.zeros(args.batch, width, dtype=torch.int32)
        table[:, :request_blocks] = torch.arange(request_blocks * args.batch, dtype=torch.int32).reshape(args.batch, -1)
        gen = OrnithGenerator(model, kv_cache=cache, page_table=table, use_prefill_trace=False)
        original_prepare = gen._prepare_prefill_sampling

        def prepare(logits):
            report["memory_before_sampling_buffer_prepare"] = memory()
            original_prepare(logits)
            report["memory_after_sampling_buffer_prepare"] = memory()

        gen._prepare_prefill_sampling = prepare
        gen.ensure_traces(preserve_cache=False)
        gen._prepare_prefill_sampling = original_prepare
        report["memory_resident_traces_and_sampling_buffers"] = memory()
        report["buffer_memory_boundary"] = "prepare delta includes copy/merge program-cache warmup allocations"
        report.update(logical_context=cache.context, physical_blocks=pool_blocks)

        # Guard only the production first-token helper. Sampler warming/capture
        # and the explicit eager numerical oracle below remain necessary.
        original_sample, original_capture = gen._sample_device, gen._capture
        original_prefill_sample = gen._sample_prefill_device
        inside_helper, inside_capture = False, False
        captures = 0

        def capture():
            nonlocal inside_capture, captures
            inside_capture = True
            try:
                original_capture()
                captures += 1
            finally:
                inside_capture = False

        def sample(logits):
            assert not inside_helper or inside_capture, "Runtime prefill submitted eager sampling"
            return original_sample(logits)

        def prefill_sample(logits, rows):
            nonlocal inside_helper
            inside_helper = True
            try:
                return original_prefill_sample(logits, rows)
            finally:
                inside_helper = False

        gen._capture, gen._sample_device, gen._sample_prefill_device = capture, sample, prefill_sample
        persistent = gen._prefill_sampling_saved + gen._prefill_sampling_masks + [gen._prefill_sampling_logits]
        addresses = [tensor.buffer_address() for tensor in persistent]
        report["persistent_buffers"] = [
            buffer_receipt(name, tensor)
            for name, tensor in zip(
                (
                    "token_backup",
                    "seed_backup",
                    "output_mask_backup",
                    "output_counts_backup",
                    "output_counts_gathered_backup",
                    "lane_mask",
                    "row_mask",
                    "logits_staging",
                ),
                persistent,
            )
        ]

        candidates = [65, 66001, 132001, 198001]
        raw = torch.full((1, 1, 32, model.padded_vocab_size), -20.0, dtype=torch.bfloat16)
        raw[0, 0, :, candidates] = torch.tensor([4.0, 3.5, 3.0, 2.5], dtype=torch.bfloat16)
        host_logits = model.upload(raw, shard_dim=-1, device=False)
        histories = [[candidates[0], candidates[0]]] * args.batch
        prompts = [[candidates[2]]] * args.batch
        modes = [
            ("greedy", SamplingParams(temperature=0.0, top_k=1, top_p=1.0, seed=7)),
            ("seeded", SamplingParams(temperature=0.8, top_k=32, top_p=0.9, seed=7)),
            (
                "penalties",
                SamplingParams(
                    temperature=0.0,
                    top_k=1,
                    top_p=1.0,
                    seed=7,
                    presence_penalty=0.5,
                    frequency_penalty=0.5,
                    repetition_penalty=1.2,
                ),
            ),
            ("greedy_again", SamplingParams(temperature=0.0, top_k=1, top_p=1.0, seed=7)),
        ]

        def reset_sampler():
            gen.sampling.reset_prompt_tokens(torch.tensor([[candidates[2]]] * 32))
            gen.sampling.reset_output_state(torch.tensor([[candidates[0], candidates[0]]] * 32))
            for target, values in (
                (gen._inputs[0], torch.arange(240000, 240032, dtype=torch.int32).reshape(1, 1, 1, 32)),
                (
                    gen.sampling.tt_sampling.seeds_tt_tensor,
                    torch.arange(0xF1234500, 0xF1234520, dtype=torch.int64).to(torch.int32),
                ),
            ):
                host = model.upload(values, dtype=target.dtype, layout=target.layout, device=False)
                ttnn.copy_host_to_device_tensor(host, target)
            ttnn.copy_host_to_device_tensor(host_logits, gen._logits)

        for mode_index, (name, params) in enumerate(modes):
            gen.configure_sampling(params, prompt_token_ids=prompts, generated_token_ids=histories)
            rows = [0] if args.batch == 1 else ([31] if mode_index % 2 == 0 else [0, 7, 31])
            if args.batch == 32 and name == "greedy_again":
                rows = list(range(32))
            before_replays = gen.counters["prefill_sampling_replays"]
            # This normal public prefill creates actual terminal logits, performs
            # prompt admission and samples them through the guarded helper.
            tokens = gen.prefill_forward(
                [[10, 20, 30]] * len(rows),
                page_table=table,
                kv_cache=cache,
                prompt_lens=[3] * len(rows),
                slots=rows,
                start_pos=[0] * len(rows),
            )
            assert gen.counters["prefill_sampling_replays"] == before_replays + 1
            assert ((tokens >= 0) & (tokens < model.vocab_size)).all()
            targets = gen._prefill_sampling_targets()
            for external in (False, True):
                reset_sampler()
                gen._ensure_replay_safe()
                # Recapture can replace canonical logits, so populate afterwards.
                ttnn.copy_host_to_device_tensor(host_logits, gen._logits)
                before = [read_all(target) for target in targets]
                original_sample(gen._logits)
                eager = [read_all(target) for target in targets]
                reset_sampler()
                if external:
                    supplied = model.upload(raw, shard_dim=-1)
                else:
                    supplied = gen._logits
                    # Exercise populated canonical-logit preservation during a
                    # required recapture, independently of cache-hit accidents.
                    gen._programs = -1
                readbacks, replays = gen.counters["readbacks"], gen.counters["prefill_sampling_replays"]
                gen._sample_prefill_device(supplied, rows)
                assert gen.counters["readbacks"] == readbacks
                assert gen.counters["prefill_sampling_replays"] == replays + 1
                observed = [read_all(target) for target in targets]
                for index, (actual_ranks, eager_ranks, before_ranks) in enumerate(zip(observed, eager, before)):
                    for rank, (actual, control, old) in enumerate(zip(actual_ranks, eager_ranks, before_ranks)):
                        expected = old.clone()
                        if index < 2:
                            expected.reshape(-1)[rows] = control.reshape(-1)[rows]
                        elif gen.sampling._penalties_active:
                            expected[rows] = control[rows]
                        assert torch.equal(actual, expected), (name, external, index, rank)
                assert addresses == [tensor.buffer_address() for tensor in persistent]
                report["cases"].append(
                    {
                        "mode": name,
                        "external_logits": external,
                        "rows": rows,
                        "exact_eager_tokens_and_all_rank_state": True,
                        "real_prefill_tokens": tokens.tolist(),
                        "sampling_trace": str(gen._sampling_trace),
                    }
                )
                report_now()
        report.update(counters=dict(gen.counters), recaptures=captures, trace_bytes_before_teardown=trace_bytes())
        gen.teardown()
        gen = None
        report["memory_after_teardown"] = memory()
        report["trace_bytes_after_teardown"] = trace_bytes()
        assert report["trace_bytes_after_teardown"] == 0
        report["passed"] = True
        report_now()
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if gen is not None:
            gen.teardown()
        if mesh is not None:
            close_ornith_mesh(mesh)


if __name__ == "__main__":
    main()
