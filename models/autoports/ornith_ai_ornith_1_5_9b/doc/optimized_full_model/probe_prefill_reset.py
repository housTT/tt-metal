# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Compare the existing double reset with skipping only its redundant second call."""

import argparse
import hashlib
import json
import os
import statistics
import traceback
from contextlib import contextmanager
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import close_ornith_mesh, open_ornith_mesh
from models.common.sampling import SamplingParams

DOC = Path(__file__).resolve().parent


@contextmanager
def duplicate_reset_baseline(gen):
    """Restore the pre-fix private call contract in both arms on promoted runtime."""
    missing = object()
    saved = gen.__dict__.get("_prefill", missing)
    original = gen._prefill

    def prefill(*args, **kwargs):
        kwargs["state_already_reset"] = False
        return original(*args, **kwargs)

    gen._prefill = prefill
    try:
        yield
    finally:
        if saved is missing:
            del gen._prefill
        else:
            gen._prefill = saved


@contextmanager
def second_reset_control(layers, *, skip_second):
    """Wrap one warmed request, preserving each layer's original method ownership."""
    counts = [0] * len(layers)
    originals = []
    missing = object()
    try:
        for index, layer in enumerate(layers):
            method = layer.reset_state
            originals.append((layer, layer.__dict__.get("reset_state", missing)))

            def reset(index=index, method=method):
                counts[index] += 1
                if skip_second and counts[index] == 2:
                    return
                method()

            layer.reset_state = reset
        yield counts
    finally:
        for layer, original in originals:
            if original is missing:
                del layer.reset_state
            else:
                layer.reset_state = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--full", action="store_true", help="Use all32 layers instead of the actual layers0/3")
    parser.add_argument("--lengths", nargs="+", type=int, default=[128, 131])
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    report = dict(
        layers="all32" if args.full else [0, 3],
        context=262144,
        batch=1,
        intervention="Both arms force private state_already_reset=False; candidate skips only second reset call",
        runtime_compatibility="Scoped wrappers support promoted runtime; immutable v2 used the original duplicate path",
        watcher=os.environ.get("TT_METAL_WATCHER"),
        allocation_tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING"),
        checks=[],
        trials=[],
        pass_=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def digest(tensor):
        return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    mesh = gen = None
    try:
        mesh = open_ornith_mesh()
        gen = build_generator(DOC.parents[1], mesh, layer_indices=None if args.full else [0, 3])
        cache = gen.kv_cache
        assert cache.batch_size == 1 and cache.context == 262144 and gen.use_prefill_trace
        assert all(a is b for a, b in zip(cache.decode_layers, cache.prefill_layers))
        writes_per_reset = sum(
            1 + len(layer.conv_state) for layer in cache.prefill_layers if not layer.is_full_attention
        )
        report["mesh_multiply_calls_per_reset"] = writes_per_reset
        report["warm_control_mesh_multiply_calls"] = 2 * writes_per_reset
        report["warm_candidate_mesh_multiply_calls"] = writes_per_reset
        canonical_table = gen.page_table.clone()
        seeds = [17 + index for index in range(32)]
        # Match list-valued seeds with list-valued temperatures. The v1 fixture
        # used scalar temperature, so the common formatter nested the seed list
        # and failed before either reset comparison arm ran.
        modes = {
            "greedy": SamplingParams(temperature=[0.0] * 32, top_k=1, top_p=1.0, seed=seeds),
            "sampled_penalties": SamplingParams(
                temperature=[0.8] * 32,
                top_k=20,
                top_p=0.95,
                seed=seeds,
                repetition_penalty=1.1,
                presence_penalty=0.1,
                frequency_penalty=0.1,
            ),
        }

        def run(tokens, params, *, skip_second):
            trace_ids = (gen._model_trace, gen._sampling_trace, gen._sampling_history_trace, gen._prefill_trace)
            assert all(trace is not None for trace in trace_ids)
            key = gen._prefill_key
            ttnn.synchronize_device(mesh)
            with (
                duplicate_reset_baseline(gen),
                second_reset_control(cache.prefill_layers, skip_second=skip_second) as counts,
            ):
                output = gen.generate(tokens, 3, sampling_params=params, stop_on_eos=False)
            assert counts == [2] * len(cache.prefill_layers), counts
            assert trace_ids == (gen._model_trace, gen._sampling_trace, gen._sampling_history_trace, gen._prefill_trace)
            assert key == gen._prefill_key
            request = gen.perf["request_counters"]
            assert request["prefill_captures"] == request["prefill_trace_misses"] == request["prefill_eager_calls"] == 0
            assert request["prefill_replays"] == request["prefill_sampling_replays"] == 1
            ttnn.synchronize_device(mesh)
            return output, dict(gen.perf)

        def snapshot():
            ranks = [hashlib.sha256() for _ in range(4)]
            for layer in cache.prefill_layers:
                if layer.is_full_attention:
                    continue
                for tensor in [layer.recurrent_state] + layer.conv_state:
                    for rank, shard in enumerate(ttnn.get_device_tensors(tensor)):
                        ranks[rank].update(ttnn.to_torch(shard).contiguous().view(torch.uint8).numpy().tobytes())
            return dict(
                hybrid_state=[value.hexdigest() for value in ranks],
                final_decode_logits=digest(gen.model.logits_to_host(gen._logits, 1)),
                token_feedback=[digest(ttnn.to_torch(shard)) for shard in ttnn.get_device_tensors(gen._inputs[0])],
                rng=[
                    digest(ttnn.to_torch(shard))
                    for shard in ttnn.get_device_tensors(gen.sampling.tt_sampling.seeds_tt_tensor)
                ],
                position=[digest(ttnn.to_torch(shard)) for shard in ttnn.get_device_tensors(gen._inputs[1])],
                penalties=(
                    [
                        digest(ttnn.to_torch(shard))
                        for tensor in gen._sampler_history_tensors()
                        for shard in ttnn.get_device_tensors(tensor)
                    ]
                    if gen.sampling._penalties_active
                    else []
                ),
            )

        for length in args.lengths:
            assert 1 <= length <= gen.model.prefill_chunk
            variants = [
                ([100] * length, canonical_table),
                ([101 + index % 11 for index in range(length)], canonical_table.flip(1).contiguous()),
            ]
            for mode, params in modes.items():
                for variant in (0, 1, 0):
                    tokens, table = variants[variant]
                    gen.page_table = table.clone()
                    # Warm any shape/mode/seed program before installing the
                    # wrappers; capture is explicitly forbidden in either arm.
                    gen.generate(tokens, 3, sampling_params=params, stop_on_eos=False)
                    gen.generate(tokens, 3, sampling_params=params, stop_on_eos=False)
                    expected, _ = run(tokens, params, skip_second=False)
                    before = snapshot()
                    actual, perf = run(tokens, params, skip_second=True)
                    after = snapshot()
                    check = dict(length=length, mode=mode, variant=variant, tokens_exact=expected == actual)
                    check.update({name + "_exact": before[name] == after[name] for name in before})
                    check["request_counters"] = perf["request_counters"]
                    report["checks"].append(check)
                    save()
                    assert all(value for name, value in check.items() if name.endswith("_exact")), check
                    print("PREFILL_RESET_EXACT", json.dumps(check), flush=True)

            tokens, table = variants[0]
            params = modes["greedy"]
            gen.page_table = table.clone()
            gen.generate(tokens, 3, sampling_params=params, stop_on_eos=False)
            gen.generate(tokens, 3, sampling_params=params, stop_on_eos=False)
            for trial in range(args.trials):
                order = (False, True) if trial % 2 == 0 else (True, False)
                for skip_second in order:
                    _, perf = run(tokens, params, skip_second=skip_second)
                    row = dict(
                        length=length,
                        trial=trial,
                        mode="skip_second" if skip_second else "control",
                        ttft_ms=perf["ttft_s"] * 1000,
                        request_setup_ms=perf["request_setup_s"] * 1000,
                        prefill_ms=perf["prefill_only_s"] * 1000,
                        request_counters=perf["request_counters"],
                    )
                    report["trials"].append(row)
                    print("PREFILL_RESET_TIMING", json.dumps(row), flush=True)
                save()
        report["medians"] = {
            str(length): {
                mode: {
                    metric: statistics.median(
                        row[metric] for row in report["trials"] if row["length"] == length and row["mode"] == mode
                    )
                    for metric in ("ttft_ms", "request_setup_ms", "prefill_ms")
                }
                for mode in ("control", "skip_second")
            }
            for length in args.lengths
        }
        report["timing_is_performance_evidence"] = not bool(report["watcher"])
        report["pass_"] = True
    except Exception as error:
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        if gen is not None:
            gen.teardown()
        if mesh is not None:
            close_ornith_mesh(mesh)
        save()
    print("PREFILL_RESET_OK", args.output, flush=True)


if __name__ == "__main__":
    main()
