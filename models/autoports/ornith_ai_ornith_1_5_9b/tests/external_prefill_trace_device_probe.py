# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact external B1 prefill comparison; run only in the supervising hardware lane.

Uses the production 4098-block pool, native context and selected precision.
Default layers0/3 are a correctness inner loop; --full32 uses every real layer.
Imports are deferred so --help and compilation do not import TTNN.
"""

import argparse
import gc
import hashlib
import json
import os
import traceback
from pathlib import Path

TRACE_FIELDS = ("_model_trace", "_sampling_trace", "_sampling_history_trace", "_prefill_trace")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full32", action="store_true")
    parser.add_argument(
        "--tensor-dir", type=Path, required=True, help="Persistent artifact directory outside source docs"
    )
    parser.add_argument("--require-trace-allocation-tracking", action="store_true")
    args = parser.parse_args()

    import torch

    import ttnn
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator_vllm import TTOrnithForCausalLM
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import OrnithModel, close_ornith_mesh, open_ornith_mesh
    from models.common.sampling import SamplingParams

    report = dict(
        layers=list(range(32)) if args.full32 else [0, 3],
        batch=1,
        context=262144,
        physical_blocks=4098,
        lanes={},
        comparisons=[],
        probe_execution_pass=False,
        trace_allocation_tracking=dict(
            enabled=bool(ttnn.TRACE_ALLOC_TRACKING),
            skip_program_cache=os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0"),
        ),
        source_sha256={
            name: hashlib.sha256((Path(__file__).parents[1] / "tt" / name).read_bytes()).hexdigest()
            for name in ("generator.py", "generator_vllm.py", "model.py")
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tensor_dir.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def sha(tensor):
        return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    def trace_bytes():
        view = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
        return int(view.num_banks * view.total_bytes_allocated_per_bank)

    def states():
        result = []
        for index, layer in zip(model.layer_indices, cache.prefill_layers):
            if layer.is_full_attention:
                continue
            for name, buffer in zip(
                ["recurrent"] + [f"conv_{i}" for i in range(len(layer.conv_state))],
                [layer.recurrent_state] + layer.conv_state,
            ):
                for rank, shard in enumerate(ttnn.get_device_tensors(buffer)):
                    result.append(dict(layer=index, buffer=name, rank=rank, sha256=sha(ttnn.to_torch(shard))))
        return result

    def sentinel():
        result = []
        for pair in cache.kv:
            if pair is None:
                continue
            for tensor in pair:
                page = ttnn.slice(tensor, [4097, 0, 0, 0], [4098, *list(tensor.shape)[1:]])
                try:
                    for shard in ttnn.get_device_tensors(page):
                        value = ttnn.to_torch(shard)
                        result.append(dict(sha256=sha(value), nonzero=int(torch.count_nonzero(value))))
                finally:
                    ttnn.deallocate(page)
        return result

    mesh = adapter = gen = cache = None
    try:
        if args.require_trace_allocation_tracking:
            assert ttnn.TRACE_ALLOC_TRACKING, "Set TT_METAL_TRACE_ALLOC_TRACKING=1 before starting Python"
            assert report["trace_allocation_tracking"]["skip_program_cache"] == "0"
        report["phase"] = "open"
        save()
        mesh = open_ornith_mesh()
        model = OrnithModel(None, mesh, layer_indices=None if args.full32 else [0, 3])
        report["precision"] = model.precision
        expected = {}
        # Only one external pool and trace family exists at a time. Each shape
        # begins with a fresh sampler/cache; both controls use the real adapter.
        for resident in (128, 131):
            for enabled in (True, False):
                name = f"resident{resident}_{'traced' if enabled else 'eager'}"
                lane = dict(cases=[], resident=resident, use_prefill_trace=enabled)
                report["lanes"][name] = lane
                adapter = TTOrnithForCausalLM(model, max_batch_size=1, max_model_len=262144, hf_config=model.hf_config)
                heads = max(1, model.hf_config.num_key_value_heads // mesh.get_num_devices())
                cache = adapter.allocate_kv_cache(
                    (4098, heads, model.page_block_size, model.hf_config.head_dim),
                    torch.bfloat16,
                    len(model.layers),
                )
                gen = adapter.generator
                gen.use_prefill_trace = enabled
                assert cache is gen.kv_cache and not gen.owns_cache
                assert cache.num_blocks == 4098 and cache.context == 262144 and model.page_block_size == 64
                base = torch.arange(adapter.page_table_blocks, dtype=torch.int32).reshape(1, -1)
                assert base.shape == (1, 4096)
                cache_addresses = [int(t.buffer_address()) for t in model.cache_buffers(cache)]
                snapshot_calls = []
                original_buffers = model.cache_buffers

                def counted_buffers(value):
                    snapshot_calls.append(id(value))
                    return original_buffers(value)

                model.cache_buffers = counted_buffers
                greedy = SamplingParams(temperature=[0.0], top_k=[1], top_p=[1.0], num_logprobs=[-2])
                report["phase"] = name + "_startup"
                save()
                if resident == 128:
                    adapter.warmup_model_prefill(cache, enable_trace=True)
                else:
                    # Fresh131 selection before an empty-pool capture, preserving
                    # exact logical length rather than padding the trace key.
                    gen._configure_sampling(greedy)
                    assert gen._prepare_prefill_trace([131], [0], [0], base) == enabled
                    gen.ensure_traces(preserve_cache=False)
                    gen.reset(clear_kv=False)
                assert not snapshot_calls, "empty startup unexpectedly snapshotted the external cache"
                assert not gen._live
                lane["startup_snapshot_calls"] = len(snapshot_calls)

                # The literal first request after startup must exercise the
                # non-live131 miss before any sentinel fixture or request runs.
                before = dict(gen.counters)
                output = gen.prefill_forward(
                    [[103 + i % 5 for i in range(131)]],
                    page_table=base,
                    kv_cache=cache,
                    prompt_lens=[131],
                    return_device_logits=True,
                )
                first_scores = model.logits_to_host(output, 1)
                ttnn.deallocate(output)
                assert gen.counters["prefill_replays"] - before["prefill_replays"] == int(enabled and resident == 131)
                assert not snapshot_calls
                first_record = dict(scores=first_scores, states=states())
                if enabled:
                    expected[(resident, "literal_first131")] = first_record
                else:
                    control = expected[(resident, "literal_first131")]
                    assert torch.equal(first_scores, control["scores"])
                    assert first_record["states"] == control["states"]
                lane["literal_first131"] = dict(
                    scores_sha256=sha(first_scores), state_hashes=first_record["states"], snapshot_calls=0
                )
                gen.reset(clear_kv=False)

                # Populate an unreferenced physical page with real nonzero KV.
                # The same-length operation also warms all sentinel-read kernels
                # before later replays; no diagnostic device tensor stays live.
                spare = base.clone()
                spare[0, 0] = 4097
                output = gen.prefill_forward(
                    [[111] * 64], page_table=spare, kv_cache=cache, prompt_lens=[64], return_device_logits=True
                )
                ttnn.deallocate(output)
                baseline_sentinel = sentinel()
                assert baseline_sentinel and all(row["nonzero"] > 0 for row in baseline_sentinel)
                gen.reset(clear_kv=False)
                # This is the startup reset boundary: a first131 must retain the
                # resident128 key even though the generator is not live.
                cases = [
                    ("first131_after_reset", [103 + i % 5 for i in range(131)], False, True, 0),
                    ("tokens_A", [100] * resident, False, True, 0),
                    ("tokens_B", [101 + i % 7 for i in range(resident)], False, False, 0),
                    ("pages_Q", [101 + i % 7 for i in range(resident)], True, False, 0),
                    ("pages_P", [101 + i % 7 for i in range(resident)], False, False, 0),
                    ("tokens_A_again", [100] * resident, False, False, 0),
                    ("live_miss", [103] * (131 if resident == 128 else 128), False, False, 0),
                    ("resident_reuse", [100] * resident, False, False, 0),
                    ("continuation", [106] * 32, False, False, resident + 1),
                ]
                records = {}
                for case, tokens, reverse, reset, start in cases:
                    report["phase"] = name + "_" + case
                    save()
                    print("EXTERNAL_PREFILL_START", name, case, flush=True)
                    if reset:
                        gen.reset(clear_kv=False)
                    table = base.flip(1).contiguous() if reverse else base
                    old_key = gen._prefill_key
                    addresses = (
                        None if gen._prefill_inputs is None else [int(t.buffer_address()) for t in gen._prefill_inputs]
                    )
                    before = dict(gen.counters)
                    snapshots_before = len(snapshot_calls)
                    output = gen.prefill_forward(
                        [tokens],
                        page_table=table,
                        kv_cache=cache,
                        prompt_lens=[len(tokens)],
                        start_pos=[start],
                        return_device_logits=True,
                    )
                    delta = {key: val - before[key] for key, val in gen.counters.items()}
                    traced = enabled and len(tokens) == resident and start == 0
                    assert delta["prefill_replays"] == int(traced), delta
                    assert delta["prefill_eager_calls"] == int(not traced), delta
                    assert gen._prefill_key == old_key
                    assert len(snapshot_calls) == snapshots_before, "shape miss rebuilt/snapshotted external pool"
                    if enabled:
                        assert [int(t.buffer_address()) for t in gen._prefill_inputs] == addresses
                        assert list(gen._prefill_inputs[0].shape) == [1, resident]
                        assert all(getattr(gen, field) is not None for field in TRACE_FIELDS)
                    scores = model.logits_to_host(output, 1)
                    state = states()
                    assert output is not gen._logits and output.buffer_address() != gen._logits.buffer_address()
                    ttnn.deallocate(output)
                    next_scores = gen.decode_forward(
                        [42], [start + len(tokens)], page_table=table, kv_cache=cache, return_logits=True
                    )
                    next_tokens = gen._read_tokens().tolist()
                    next_states = states()
                    assert sentinel() == baseline_sentinel, "unreferenced physical page changed"
                    assert cache is gen.kv_cache and not gen.owns_cache
                    assert [int(t.buffer_address()) for t in original_buffers(cache)] == cache_addresses
                    allocated = trace_bytes()
                    assert 0 < allocated <= 100_000_000
                    actual = dict(
                        scores=scores,
                        states=state,
                        next_scores=next_scores,
                        next_tokens=next_tokens,
                        next_states=next_states,
                    )
                    records[case] = actual
                    if enabled:
                        expected[(resident, case)] = actual
                    else:
                        control = expected[(resident, case)]
                        checks = {
                            key: (
                                bool(torch.equal(value, control[key]))
                                if isinstance(value, torch.Tensor)
                                else value == control[key]
                            )
                            for key, value in actual.items()
                        }
                        report["comparisons"].append(dict(resident=resident, case=case, **checks))
                        assert all(checks.values()), (resident, case, checks)
                    lane["cases"].append(
                        dict(
                            case=case,
                            logical_length=len(tokens),
                            start=start,
                            expected_trace=traced,
                            counters=delta,
                            scores_sha256=sha(scores),
                            state_hashes=state,
                            next_scores_sha256=sha(next_scores),
                            next_tokens=next_tokens,
                            next_state_hashes=next_states,
                            sentinel_preserved=True,
                            external_identity_preserved=True,
                            public_owned_and_deallocated=True,
                            trace_bytes=allocated,
                            prefill_addresses=addresses,
                        )
                    )
                    save()
                    print("EXTERNAL_PREFILL_PASS", name, case, json.dumps(delta), flush=True)
                assert not torch.equal(records["tokens_A"]["scores"], records["tokens_B"]["scores"])
                for case, reference in (
                    ("pages_Q", "tokens_B"),
                    ("pages_P", "tokens_B"),
                    ("tokens_A_again", "tokens_A"),
                    ("resident_reuse", "tokens_A"),
                ):
                    for key in ("scores", "next_scores"):
                        assert torch.equal(records[case][key], records[reference][key]), (case, key)
                    assert records[case]["states"] == records[reference]["states"]
                # Explicit default-preserving rebuild proves external sentinel and
                # hybrid buffers survive the native snapshot path as well.
                gen.reset(clear_kv=False)
                before_state = states()
                for _ in range(2):
                    gen._release_traces()
                    assert trace_bytes() == 0
                    before = len(snapshot_calls)
                    gen.ensure_traces()
                    assert len(snapshot_calls) == before + 1
                    assert states() == before_state and sentinel() == baseline_sentinel
                    gen._ensure_replay_safe()
                    gen.decode_forward([42], [resident], page_table=base, kv_cache=cache, return_logits=True)
                    gen.reset(clear_kv=False)
                    before_state = states()
                lane["snapshot_calls"] = len(snapshot_calls)
                tensor_path = args.tensor_dir / (args.output.stem + "_" + name + ".pt")
                torch.save(records, tensor_path)
                lane["tensor_artifact"] = str(tensor_path)
                model.cache_buffers = original_buffers
                adapter.teardown()
                assert trace_bytes() == 0
                assert all(getattr(gen, field) is None for field in TRACE_FIELDS)
                assert gen._prefill_inputs is None and gen._prefill_key is None
                lane["trace_bytes_after_teardown"] = 0
                # atexit holds adapters; teardown clears its generator reference.
                gen = cache = adapter = None
                model.cache = None
                gc.collect()
                save()
        report["phase"] = "complete"
        report["probe_execution_pass"] = True
    except Exception as error:
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        if adapter is not None:
            adapter.teardown()
        if mesh is not None:
            close_ornith_mesh(mesh)
        save()
    print("EXTERNAL_PREFILL_OK", args.output, flush=True)


if __name__ == "__main__":
    main()
