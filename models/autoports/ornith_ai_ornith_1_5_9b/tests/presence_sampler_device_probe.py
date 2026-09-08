# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact TP4 canonical sampler presence crossing; supervising hardware lane only.

No model weights are needed. Real model.build_sampler allocation, vocabulary,
output feedback, penalty history, and outer sampling trace are retained.
"""

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.common.sampling import SamplingParams, format_sampling_params

from ..reference.hf_reference import load_text_config
from ..tt.generator import OrnithGenerator
from ..tt.model import OrnithModel, SamplingCCL, close_ornith_mesh, open_ornith_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("presence", "combined"), default="presence")
    args = parser.parse_args()
    mesh = open_ornith_mesh()
    trace = None
    try:
        model = OrnithModel.__new__(OrnithModel)
        model.mesh_device, model.ccl = mesh, SamplingCCL(mesh)
        model.vocab_size = load_text_config(args.model_path).vocab_size
        model.padded_vocab_size = 262144
        gen = OrnithGenerator.__new__(OrnithGenerator)
        gen.model, gen.mesh_device, gen.max_batch_size = model, mesh, 32
        gen.counters = {"readbacks": 0}
        gen.sampling = model.build_sampler()
        gen._inputs = [gen._device(torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32)]
        candidates = [65, 66001, 132001, 198001]  # One candidate on each actual vocabulary shard.
        if args.case == "combined":
            # Repeat eight neutral/single/combined configurations across four
            # raw-score patterns, including negative and zero-crossing logits.
            settings = [
                (0.0, 0.0, 1.0),
                (2.0, 0.0, 1.0),
                (0.0, 0.5, 1.0),
                (0.0, 0.0, 2.0),
                (2.0, 0.0, 2.0),
                (0.0, 0.5, 2.0),
                (2.0, 0.5, 2.0),
                (-1.5, -0.5, 2.0),
            ] * 4
            score_rows = torch.tensor(
                [[4.0, 0.5, -0.5, -4.0]] * 8
                + [[-0.5, -1.0, -2.0, -4.0]] * 8
                + [[0.5, 0.25, -0.5, -4.0]] * 8
                + [[4.0, 1.5, 2.0, -4.0]] * 8
            )
            prompt_ids = torch.full((32, 1), candidates[2], dtype=torch.int64)
        else:
            settings = [(p, 0.0, 1.0) for p in [0.0, 2.0, -1.5, 0.5] * 8]
            score_rows = torch.tensor([4.0, 3.0, 2.5, -2.0]).repeat(32, 1)
            prompt_ids = torch.full((32, 1), -1, dtype=torch.int64)
        penalties, frequencies, repetitions = map(list, zip(*settings))
        params = SamplingParams(
            temperature=[0.0] * 32,
            top_k=[1] * 32,
            top_p=[1.0] * 32,
            presence_penalty=penalties,
            frequency_penalty=frequencies,
            repetition_penalty=repetitions,
            seed=[7] * 32,
        )
        gen.sampling.reset_sampling_params(format_sampling_params(params, 32))
        gen.sampling.reset_prompt_tokens(prompt_ids)
        raw = torch.full((1, 1, 32, model.padded_vocab_size), -20.0, dtype=torch.bfloat16)
        raw[0, 0, :, candidates] = score_rows.to(torch.bfloat16)
        logits = model.upload(raw, dtype=ttnn.bfloat16, shard_dim=-1)
        host_logits = model.upload(raw, dtype=ttnn.bfloat16, shard_dim=-1, device=False)

        def state_addresses():
            history = gen.sampling.tt_penalties
            return [
                tensor.buffer_address()
                for tensor in (
                    logits,
                    gen._inputs[0],
                    history.output_mask,
                    history.output_counts,
                    history.output_counts_gathered,
                    gen.sampling.tt_sampling.seeds_tt_tensor,
                )
            ]

        addresses = state_addresses()

        def reset(count):
            gen.sampling.reset_output_state(torch.full((32, count), candidates[0], dtype=torch.int64))
            ttnn.copy_host_to_device_tensor(host_logits, logits)
            assert state_addresses() == addresses, "Reset changed a trace-bound buffer address"

        # Warm both host-history widths before recording any device addresses.
        # reset_output_state currently expands them into the same dense device
        # shape; exercising both also protects this probe if that changes.
        for count in (1, 7):
            reset(count)
            gen._sample_device(logits)
        reset(1)
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        gen._sample_device(logits)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        programs = mesh.num_program_cache_entries()
        runs = []
        for count in (1, 7):
            reset(count)
            assert (
                mesh.num_program_cache_entries() == programs
            ), "Unwarmed reset changed the program cache after capture"
            counts = torch.zeros(32, 4, dtype=torch.int64)
            counts[:, 0] = count
            prompt_mask = torch.zeros(32, 4, dtype=torch.bool)
            prompt_mask[:, 2] = args.case == "combined"
            outputs = []
            for step in range(2):
                # In full decode, the model trace refreshes this same logits buffer.
                ttnn.copy_host_to_device_tensor(host_logits, logits)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                pending, event = gen.read_output_async()
                ttnn.event_synchronize(event)
                tokens = gen.tokens_from(pending)
                factor = torch.where(prompt_mask | (counts > 0), torch.tensor(repetitions)[:, None], 1.0)
                expected_scores = torch.where(score_rows > 0, score_rows / factor, score_rows * factor)
                expected_scores -= torch.tensor(frequencies)[:, None] * counts
                expected_scores -= torch.tensor(penalties)[:, None] * (counts > 0)
                expected = torch.tensor(candidates)[expected_scores.argmax(-1)]
                assert torch.equal(tokens, expected), (count, step, tokens.tolist(), expected.tolist())
                # Check exact subtraction as well as top-1, on all physical shards.
                for rank, shard in enumerate(ttnn.get_device_tensors(logits)):
                    observed = ttnn.to_torch(shard)[0, 0, :, candidates[rank] % (model.padded_vocab_size // 4)]
                    torch.testing.assert_close(observed.float(), expected_scores[:, rank], rtol=0, atol=0)
                for row, token in enumerate(tokens.tolist()):
                    counts[row, candidates.index(token)] += 1
                outputs.append(tokens.tolist())
            runs.append({"initial_generated_count": count, "step_tokens": outputs, "exact_penalty_scores": True})
        # Count must not affect neutral, presence-only, or repetition-only lanes.
        no_frequency = torch.tensor(frequencies) == 0
        assert torch.equal(
            torch.tensor(runs[0]["step_tokens"])[:, no_frequency],
            torch.tensor(runs[1]["step_tokens"])[:, no_frequency],
        ), "Without frequency, penalties depend on presence rather than count"
        report = {
            "hardware": "P300c, four Blackhole chips, TP4",
            "vocab_size": model.vocab_size,
            "padded_vocab_size": model.padded_vocab_size,
            "batch": 32,
            "penalties": penalties,
            "frequency_penalties": frequencies,
            "repetition_penalties": repetitions,
            "case": args.case,
            "raw_candidate_scores": score_rows.tolist(),
            "prompt_token_ids": prompt_ids.tolist(),
            "candidate_ids": candidates,
            "runs": runs,
            "state_addresses_stable": True,
            "reset_program_cache_stable": True,
            "path": "model.build_sampler + generator._sample_device with captured feedback/history",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        close_ornith_mesh(mesh)


if __name__ == "__main__":
    main()
