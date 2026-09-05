# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Focused real-layer feedback, page-table, fixed-slot and sampling trace tests."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import close_ornith_mesh, open_ornith_mesh
from models.common.sampling import SamplingParams

DOC = Path(__file__).resolve().parent
ROOT = DOC.parents[1]


def host(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    mesh = open_ornith_mesh()
    report = dict(batch=args.batch, checks={})
    try:
        gen = build_generator(ROOT, mesh, layer_indices=[0, 3], cache_context=2048, max_batch_size=args.batch)
        try:
            prompts = [list(range(127)), list(range(131)), [31, 57, 88]]
            if args.batch < 3:
                prompts = prompts[: args.batch]
            elif args.batch > 4:
                prompts = [prompts[i % 3] for i in range(args.batch - 1)]
            tokens = gen.generate(prompts, 8)
            positions = host(gen._inputs[1]).tolist()
            expected = [len(row) + 7 for row in prompts] + [-1] * (args.batch - len(prompts))
            assert positions == expected, (positions, expected)
            assert len(tokens) == len(prompts) and all(len(row) == 8 for row in tokens)
            for slot, prompt in enumerate(prompts):
                anchor = prompts.index(prompt)
                assert tokens[slot] == tokens[anchor], (slot, anchor, tokens[slot], tokens[anchor])
            report["checks"]["mixed_prompts"] = dict(tokens=tokens, positions=positions)
            original = gen.page_table.clone()
            gen.page_table = original.flip(1).contiguous()
            permuted = gen.generate(prompts, 8)
            assert tokens == permuted, (tokens, permuted)
            report["checks"]["permuted_page_table"] = True
            before = gen.counters["page_table_refreshes"]
            gen._refresh_table(gen.page_table)
            gen._refresh_table(gen.page_table.clone())
            assert gen.counters["page_table_refreshes"] == before
            changed = gen.page_table.clone()
            changed[:, -2:] = changed[:, -2:].flip(1)
            gen._refresh_table(changed)
            assert gen.counters["page_table_refreshes"] == before + 1
            assert torch.equal(host(gen._inputs[3]), changed)
            gen._refresh_table(changed)
            assert gen.counters["page_table_refreshes"] == before + 1
            report["checks"]["changed_unchanged_table"] = dict(copies_for_unchanged=0, copies_for_changed=1)
            address = gen._inputs[0].buffer_address()
            before = host(gen._inputs[0]).clone()
            step_records = []
            for step in range(3):
                pos = host(gen._inputs[1]).clone()
                with (
                    patch.object(ttnn, "from_torch", side_effect=AssertionError("host tensor creation during replay")),
                    patch.object(
                        ttnn, "copy_host_to_device_tensor", side_effect=AssertionError("host write during replay")
                    ),
                    patch.object(ttnn, "synchronize_device", side_effect=AssertionError("blocking sync during replay")),
                ):
                    gen._replay()
                after = host(gen._inputs[0]).clone()
                newpos = host(gen._inputs[1])
                assert gen._inputs[0].buffer_address() == address
                assert torch.equal(newpos, torch.where(pos >= 0, pos + 1, pos))
                step_records.append(
                    dict(
                        input_tokens=before.flatten().tolist(),
                        sampled_tokens=after.flatten().tolist(),
                        positions=newpos.tolist(),
                        token_buffer_address=address,
                    )
                )
                before = after
            report["checks"]["feedback"] = step_records
            for layer in gen.kv_cache.decode_layers:
                if layer.is_full_attention:
                    continue
                for tensor in [layer.recurrent_state] + layer.conv_state:
                    idle = host(tensor)[len(prompts) :]
                    assert bool((idle == 0).all()), "inactive recurrent/conv rows changed"
            report["checks"]["inactive_state_frozen"] = True
            scheduler_tokens = torch.arange(args.batch) + 71
            scheduler_pos = torch.tensor([len(row) + 12 for row in prompts] + [-1] * (args.batch - len(prompts)))
            a = gen.decode_forward(
                scheduler_tokens, scheduler_pos, page_table=changed, kv_cache=gen.kv_cache, return_logits=True
            )
            assert (
                host(gen._inputs[1]).tolist()
                == torch.where(scheduler_pos >= 0, scheduler_pos + 1, scheduler_pos).tolist()
            )
            b = gen.decode_forward(
                scheduler_tokens + 13,
                torch.where(scheduler_pos >= 0, scheduler_pos + 1, scheduler_pos),
                page_table=changed,
                kv_cache=gen.kv_cache,
                return_logits=True,
            )
            assert not torch.equal(a, b), "changed token/position replay returned stale logits"
            report["checks"]["changed_token_position_logits"] = True
            if args.batch > 1:
                all_prompts = [prompts[i % len(prompts)] for i in range(args.batch)]
                all_tokens = gen.generate(all_prompts, 4)
                assert len(all_tokens) == args.batch and all(len(row) == 4 for row in all_tokens)
                for slot, prompt in enumerate(all_prompts):
                    anchor = all_prompts.index(prompt)
                    assert all_tokens[slot] == all_tokens[anchor], (slot, anchor, all_tokens)
                report["checks"]["all_slots_active"] = dict(count=args.batch, tokens=all_tokens)
            greedy = gen.generate(prompts, 8)
            params = SamplingParams(temperature=0.7, top_k=16, top_p=0.9, seed=123)
            sample1 = gen.generate(prompts, 8, sampling_params=params)
            sample2 = gen.generate(prompts, 8, sampling_params=params)
            assert sample1 == sample2, (sample1, sample2)
            replicas = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(gen._inputs[0])]
            assert all(torch.equal(replicas[0], other) for other in replicas[1:])
            report["checks"]["seeded_replica_equality"] = True
            again = gen.generate(prompts, 8)
            assert greedy == again, (greedy, again)
            report["checks"]["alternate_seeded_greedy"] = dict(greedy=greedy, seeded=sample1, repeated_seeded=sample2)
            if args.batch == 4:
                card_params = SamplingParams(temperature=1.0, top_k=20, top_p=0.95, seed=77, presence_penalty=1.5)
                card1 = gen.generate(prompts, 8, sampling_params=card_params)
                card2 = gen.generate(prompts, 8, sampling_params=card_params)
                assert card1 == card2, (card1, card2)
                assert gen.generate(prompts, 8) == greedy
                report["checks"]["model_card_penalties_trace_reset"] = dict(tokens=card1, repeated=card2)
            report["pass"] = True
        finally:
            gen.teardown()
    finally:
        close_ornith_mesh(mesh)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
