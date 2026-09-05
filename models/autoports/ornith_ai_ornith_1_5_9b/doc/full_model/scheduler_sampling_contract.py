"""Reduced hardware probe for live scheduler sampling changes and partial prefill.

Run only under the stage owner's serialized TT hardware schedule.
"""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import close_ornith_mesh, open_ornith_mesh
from models.common.sampling import SamplingParams

ROOT = Path(__file__).resolve().parents[2]


def snapshots(tensors):
    return [[ttnn.to_torch(local).clone() for local in ttnn.get_device_tensors(tensor)] for tensor in tensors]


def assert_same(before, tensors):
    after = snapshots(tensors)
    assert len(before) == len(after)
    for index, (old_mesh, new_mesh) in enumerate(zip(before, after)):
        for chip, (old, new) in enumerate(zip(old_mesh, new_mesh)):
            assert torch.equal(old, new), (index, chip, old.shape)


def lane_merge_probe(mesh):
    """Separate tilize, WHERE dispatch, untilize and persistent copy effects."""

    def upload(value, dtype, layout):
        return ttnn.from_torch(
            value, dtype=dtype, layout=layout, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
        )

    old_host = (torch.arange(32, dtype=torch.int64) * 257 + 0x81230001).reshape(1, 1, 1, 32)
    new_host = (torch.arange(32, dtype=torch.int64) * 65537 + 0xFE000001).reshape(1, 1, 1, 32)
    condition = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
    condition[..., [2, 7, 31]] = 1
    expected = torch.where(condition.bool(), new_host, old_host)
    report = {}
    for name, dtype in [("uint32_predicate_control", ttnn.uint32), ("int32_predicate_fix", ttnn.int32)]:
        old = upload(old_host, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        new = upload(new_host, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        mask = upload(condition, dtype, ttnn.TILE_LAYOUT)
        tiled_old = ttnn.to_layout(old, ttnn.TILE_LAYOUT)
        tiled_new = ttnn.to_layout(new, ttnn.TILE_LAYOUT)
        for tensor, wanted in [(tiled_old, old_host), (tiled_new, new_host)]:
            assert all(
                torch.equal(value.to(torch.int64), wanted) for value in snapshots([tensor])[0]
            ), "tilize corrupted UINT32"
        merged = ttnn.where(mask, tiled_new, tiled_old)
        tiled_result = snapshots([merged])[0]
        restored = ttnn.to_layout(merged, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy(restored, old)
        final_result = snapshots([old])[0]
        report[name] = {
            "expected": expected.flatten().tolist(),
            "tiled": [value.flatten().tolist() for value in tiled_result],
            "copied": [value.flatten().tolist() for value in final_result],
            "exact": all(torch.equal(value.to(torch.int64), expected) for value in tiled_result + final_result),
        }
        print("LANE_MERGE_PROBE", json.dumps({name: report[name]}), flush=True)
        if name == "int32_predicate_fix":
            assert report[name]["exact"], report[name]
        for tensor in [old, new, mask, tiled_old, tiled_new, merged, restored]:
            ttnn.deallocate(tensor)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    assert args.batch >= 3
    report = {"batch": args.batch, "layer_indices": [0, 3], "checks": {}}
    mesh = open_ornith_mesh()
    try:
        report["checks"]["isolated_large_uint32_lane_merge"] = lane_merge_probe(mesh)
        if args.merge_only:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            return
        gen = build_generator(ROOT, mesh, layer_indices=[0, 3], max_batch_size=args.batch, cache_context=512)
        try:
            seed_trials = {}
            for name, seeds in (
                ("explicit_list", list(range(100, 100 + args.batch))),
                ("scalar", 123),
                ("unseeded", None),
            ):
                trials = []
                for _ in range(2):
                    seed_params = SamplingParams(temperature=[0.8] * args.batch, top_k=32, top_p=1.0, seed=seeds)
                    gen.configure_sampling(seed_params, reset_seed=True)
                    trials.append(snapshots([gen.sampling.tt_sampling.seeds_tt_tensor])[0][0].flatten().tolist())
                if name == "explicit_list":
                    assert trials[0][: args.batch] == trials[1][: args.batch]
                elif name == "scalar":
                    assert gen._configured_seeds == [123] + [None] * 31
                    assert trials[0][0] == trials[1][0]
                    assert trials[0][1:] != trials[1][1:]
                else:
                    assert trials[0] != trials[1]
                    assert all(len(set(values[: args.batch])) > 1 for values in trials)
                seed_trials[name] = trials
            report["checks"]["seed_initialization"] = seed_trials

            def params(temperature, top_k):
                return SamplingParams(
                    temperature=[temperature] * args.batch,
                    top_k=[top_k] * args.batch,
                    top_p=1.0,
                    presence_penalty=0.1,
                    frequency_penalty=0.1,
                    repetition_penalty=1.05,
                    seed=[42] * args.batch,
                )

            initial = params(0.8, 32)
            admission_prompt = [88, 99, 111]
            fresh_tokens = gen.generate([admission_prompt], 1, sampling_params=initial, stop_on_eos=False)
            fresh_seed = snapshots([gen.sampling.tt_sampling.seeds_tt_tensor])[0][0].flatten()[0].item()
            gen.generate([[11, 22, 33], [44, 55, 66, 77]], 4, sampling_params=initial, stop_on_eos=False)
            state = gen.model.cache_buffers(gen.kv_cache) + gen._inputs
            state += [gen.sampling.tt_sampling.seeds_tt_tensor] + gen._sampler_history_tensors()
            state += [gen.sampling.tt_penalties.prompt_mask]
            before = snapshots(state)
            addresses = [tensor.buffer_address() for tensor in state]
            for mode in (params(0.0, 1), initial):
                with patch.object(gen.model, "reset_cache", side_effect=AssertionError("live cache reset")):
                    gen.configure_sampling(mode)
                assert gen._live
                assert [tensor.buffer_address() for tensor in state] == addresses
                assert_same(before, state)
            report["checks"]["live_sampling_change_preserves_state"] = True
            # Run a replay with unchanged scheduler inputs: no host rebuild/copy/reset.
            old_pos = snapshots([gen._inputs[1]])[0]
            with patch.object(ttnn, "copy_host_to_device_tensor", side_effect=AssertionError("per-token host copy")):
                gen.decode_forward(None, None, page_table=gen.page_table, kv_cache=gen.kv_cache)
            new_pos = snapshots([gen._inputs[1]])[0]
            for old, new in zip(old_pos, new_pos):
                assert torch.equal(new, torch.where(old >= 0, old + 1, old))
            report["checks"]["replay_after_mode_change"] = True
            # Slot 2 starts while slots 0/1 remain live. Their histories and next
            # token/seed must survive the 32-lane prefill sampler unchanged.
            sampler = gen.sampling.tt_penalties
            ongoing = [sampler.prompt_mask] + gen._sampler_history_tensors()
            old_history = snapshots(ongoing)
            old_tokens = snapshots([gen._inputs[0]])[0]
            old_seeds = snapshots([gen.sampling.tt_sampling.seeds_tt_tensor])[0]
            joined_tokens = gen.prefill_forward(
                [admission_prompt], page_table=gen.page_table, kv_cache=gen.kv_cache, prompt_lens=[3], slots=[2]
            )
            joined_seed = snapshots([gen.sampling.tt_sampling.seeds_tt_tensor])[0][0].flatten()[2].item()
            assert joined_seed == fresh_seed, (joined_seed, fresh_seed)
            assert joined_tokens.tolist() == fresh_tokens[0], (joined_tokens, fresh_tokens)
            for old_mesh, new_mesh in zip(old_history, snapshots(ongoing)):
                for old, new in zip(old_mesh, new_mesh):
                    assert torch.equal(old[:2], new[:2]), "ongoing slot penalty history changed"
            for old, new in zip(old_tokens, snapshots([gen._inputs[0]])[0]):
                print(
                    "PARTIAL_PREFILL_TOKENS",
                    json.dumps({"before": old.flatten().tolist(), "after": new.flatten().tolist()}),
                    flush=True,
                )
                assert torch.equal(old.flatten()[:2], new.flatten()[:2]), (
                    "ongoing token feedback changed",
                    old.flatten().tolist(),
                    new.flatten().tolist(),
                )
            for old, new in zip(old_seeds, snapshots([gen.sampling.tt_sampling.seeds_tt_tensor])[0]):
                print(
                    "PARTIAL_PREFILL_SEEDS",
                    json.dumps({"before": old.flatten().tolist(), "after": new.flatten().tolist()}),
                    flush=True,
                )
                assert torch.equal(old.flatten()[:2], new.flatten()[:2]), (
                    "ongoing RNG state changed",
                    old.flatten().tolist(),
                    new.flatten().tolist(),
                )
            report["checks"]["partial_prefill_preserves_ongoing_sampler_rows"] = True
            before_continuation = snapshots([gen.sampling.tt_sampling.seeds_tt_tensor])[0]
            gen.prefill_forward(
                [[112]], page_table=gen.page_table, kv_cache=gen.kv_cache, prompt_lens=[1], slots=[2], start_pos=[3]
            )
            after_continuation = snapshots([gen.sampling.tt_sampling.seeds_tt_tensor])[0]
            for old, new in zip(before_continuation, after_continuation):
                expected = old.flatten().to(torch.int64).clone()
                expected[2] += 1
                assert torch.equal(new.flatten().to(torch.int64), expected), "continuation reset a request seed"
            reused_tokens = gen.prefill_forward(
                [admission_prompt], page_table=gen.page_table, kv_cache=gen.kv_cache, prompt_lens=[3], slots=[2]
            )
            reused_seed = snapshots([gen.sampling.tt_sampling.seeds_tt_tensor])[0][0].flatten()[2].item()
            assert reused_seed == fresh_seed
            assert reused_tokens.tolist() == fresh_tokens[0]
            report["checks"]["seeded_fresh_joined_reused_request"] = {
                "fresh_seed_after_prefill": fresh_seed,
                "joined_seed_after_prefill": joined_seed,
                "reused_seed_after_prefill": reused_seed,
                "tokens": fresh_tokens[0],
                "continuation_advanced_without_reset": True,
            }
            positions = snapshots([gen._inputs[1]])[0][0].flatten().tolist()
            positions[2] = 3
            gen.decode_forward(None, positions, page_table=gen.page_table, kv_cache=gen.kv_cache)
            report["checks"]["new_slot_joins_live_decode"] = True
        finally:
            gen.teardown()
    finally:
        close_ornith_mesh(mesh)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
