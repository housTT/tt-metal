# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serialized TP4 serving-boundary proof; run only under the supervising watchdog.

This imports TTNN and opens real hardware. It is a reduced real-weight check,
not a full-model accuracy/performance gate. The authoring subagent never ran it.
"""

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.common.sampling import SamplingParams

from ..tt.functional_decoder import num_blocks_for_context
from ..tt.generator import OrnithGenerator
from ..tt.model import OrnithModel, close_ornith_mesh, open_ornith_mesh


def read_all(tensor):
    return [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(tensor)]


def write(gen, target, values):
    host = gen.model.upload(values, dtype=target.dtype, layout=target.layout, device=False)
    ttnn.copy_host_to_device_tensor(host, target)


def check_remap(gen):
    """Exercise actual local heads, dtypes, layouts and common penalty buffers."""
    batch = gen.max_batch_size
    assert batch == 3
    remap = [2, 0, 1]
    state = []
    for layer in gen.kv_cache.decode_layers:
        if not layer.is_full_attention:
            state.extend([layer.recurrent_state] + layer.conv_state)
    for target in state:
        shape = list(target.shape)
        value = torch.arange(1, batch + 1).reshape(batch, *([1] * (len(shape) - 1))).expand(shape).float()
        value = value.clone()
        if target.dtype == ttnn.float32:
            value[1] = float("nan")
        write(gen, target, value)
    gen._write_tokens([70001, 140003, 240007])
    gen._write_positions([1000, 2000, 3000])
    seeds = gen.sampling.tt_sampling.seeds_tt_tensor
    write(gen, seeds, torch.arange(0xF1234500, 0xF1234520, dtype=torch.int64).to(torch.int32))
    penalties = [gen.sampling.tt_penalties.prompt_mask] + gen._sampler_history_tensors()
    for target in penalties:
        shape = list(target.shape)
        write(gen, target, torch.arange(32, dtype=torch.int32)[:, None].expand(shape).contiguous())
    targets = state + gen._inputs[:3] + [seeds] + penalties
    before = [read_all(target) for target in targets]
    addresses = [target.buffer_address() for target in targets]
    moved = gen.remap_serving_slots(remap)
    for index, target in enumerate(targets):
        for actual, old in zip(read_all(target), before[index]):
            expected = old.clone()
            if len(state) <= index < len(state) + 4:
                expected.reshape(-1)[:batch] = old.reshape(-1)[remap]
            else:
                expected[:batch] = old[remap]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert addresses == [target.buffer_address() for target in targets]
    gen.refresh_serving_inputs([11, 220009, 13], [999, 7, -1], [False, True, False])
    assert gen.tokens_from(gen._inputs[0]).tolist() == [240007, 220009, 140003]
    assert read_all(gen._inputs[1])[0].flatten().tolist() == [3000, 7, -1]
    assert read_all(gen._inputs[2])[0].flatten().tolist() == [3000, 7, 2000]
    assert read_all(gen.kv_cache.active_recurrent)[0].flatten().tolist() == [1, 1, 0]
    assert read_all(gen.kv_cache.active_conv)[0].flatten().tolist() == [1, 1, 0]
    return {"moved_linear_layers": moved, "exact_all_replicas": True, "stable_addresses": True}


def prefill(gen):
    gen.reset()
    gen.prefill_forward(
        [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
        page_table=gen.page_table,
        kv_cache=gen.kv_cache,
        prompt_lens=[3, 3, 3],
    )
    gen._write_positions([3, 3, 3])


def submit(gen, *, sample=True):
    return gen.decode_forward(
        None,
        None,
        page_table=gen.page_table,
        kv_cache=gen.kv_cache,
        read_from_device=False,
        return_logits=not sample,
        sample_on_device=sample,
    )


def check_async_and_host_mode(gen):
    prefill(gen)
    baseline = []
    for _ in range(2):
        host, event = gen.read_output_async(submit(gen))
        ttnn.event_synchronize(event)
        baseline.append(gen.tokens_from(host))
    prefill(gen)
    pending = []
    for _ in range(2):
        # Copy N is queued before replay N+1, but neither copy is formatted yet.
        pending.append(gen.read_output_async(submit(gen)))
    for expected, (host, event) in zip(baseline, pending):
        ttnn.event_synchronize(event)
        assert torch.equal(expected, gen.tokens_from(host))

    prefill(gen)
    gen._ensure_replay_safe()
    tokens_before = read_all(gen._inputs[0])
    seeds_before = read_all(gen.sampling.tt_sampling.seeds_tt_tensor)
    trace = gen._sampling_trace
    replays = gen.counters["sampling_replays"]
    host, event = gen.read_output_async(submit(gen, sample=False), return_logits=True)
    ttnn.event_synchronize(event)
    logits = gen.logits_from(host)
    assert logits.shape == (gen.max_batch_size, gen.model.vocab_size)
    assert torch.isfinite(logits).all()
    assert gen.counters["sampling_replays"] == replays
    assert gen._sampling_trace == trace
    for previous, current in zip(tokens_before, read_all(gen._inputs[0])):
        assert torch.equal(previous, current)
    for previous, current in zip(seeds_before, read_all(gen.sampling.tt_sampling.seeds_tt_tensor)):
        assert torch.equal(previous, current)
    # A host-sampled token must be explicitly refreshed before device mode resumes.
    gen.refresh_serving_inputs(logits.argmax(-1), [4, 4, 4], [True, True, True])
    submit(gen)
    assert gen.counters["sampling_replays"] == replays + 1
    ttnn.synchronize_device(gen.mesh_device)
    return {"async_tokens": [row.tolist() for row in baseline], "host_logits_shape": list(logits.shape)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0,3")
    args = parser.parse_args()
    mesh = open_ornith_mesh()
    gen = None
    try:
        model = OrnithModel(args.model_path, mesh, layer_indices=[int(value) for value in args.layers.split(",")])
        width = num_blocks_for_context(model.max_context, model.page_block_size)
        request_blocks = num_blocks_for_context(256, model.page_block_size)
        # paged_fused_update_cache requires table width <= physical blocks,
        # even when most logical pages map to an existing filler block.
        pool_blocks = width + request_blocks * 3
        cache = model.allocate_cache(3, num_blocks=pool_blocks)
        table = torch.zeros(3, width, dtype=torch.int32)
        table[:, :request_blocks] = torch.arange(request_blocks * 3, dtype=torch.int32).reshape(3, request_blocks)
        assert cache.context == 262144
        assert pool_blocks < 3 * width
        assert all(pair is None or pair[0].shape[0] == pool_blocks for pair in cache.kv)
        gen = OrnithGenerator(model, kv_cache=cache, page_table=table, use_prefill_trace=False)
        gen._configure_sampling(SamplingParams(temperature=0.0, top_k=1, top_p=1.0))
        gen.ensure_traces(preserve_cache=False)
        report = {
            "hardware": "P300c, four Blackhole chips, TP4",
            "layers": args.layers,
            "batch": 3,
            "logical_context": cache.context,
            "physical_blocks": pool_blocks,
            "remap": check_remap(gen),
            "decode": check_async_and_host_mode(gen),
            "counters": dict(gen.counters),
            "full_model_accuracy": "not assessed by this reduced contract probe",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if gen is not None:
            gen.teardown()
        close_ornith_mesh(mesh)


if __name__ == "__main__":
    main()
