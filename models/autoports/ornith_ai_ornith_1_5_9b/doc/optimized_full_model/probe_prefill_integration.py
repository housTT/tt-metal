# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact prefill/decode integration gate against use_prefill_trace=False.

Quick mode covers ownership, shape/key lifecycle, greedy/sampled/penalty requests.
Normal mode adds length1/2048 and128/260-token output-history boundaries.
--full32 selects all32 real layers; every mode uses the native B1 cache.
Imports are deferred so --help and host compilation never access devices.
"""

import argparse
import gc
import hashlib
import json
import traceback
from pathlib import Path

DOC = Path(__file__).resolve().parent
TRACE_FIELDS = ("_model_trace", "_sampling_trace", "_sampling_history_trace", "_prefill_trace")
PREFILL_COUNTERS = (
    "prefill_replays",
    "prefill_captures",
    "prefill_trace_misses",
    "prefill_token_refreshes",
    "prefill_page_table_refreshes",
    "prefill_eager_calls",
    "prefill_sampling_replays",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--edges", action="store_true", help="Also test exact lengths1/2048 in quick mode")
    parser.add_argument("--full32", action="store_true", help="Use all32 layers; default is real layers0/3")
    args = parser.parse_args()

    import torch

    import ttnn
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import OrnithGenerator
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import OrnithModel, close_ornith_mesh, open_ornith_mesh
    from models.common.sampling import SamplingParams

    report = dict(
        layers=list(range(32)) if args.full32 else [0, 3],
        batch=1,
        context=262144,
        hardware="TP4, four Blackhole chips on physical P300c boards",
        quick=args.quick,
        edges=args.edges or not args.quick,
        lanes={},
        comparisons=[],
        probe_execution_pass=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def sha(tensor):
        data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        return hashlib.sha256(data).hexdigest()

    def trace_bytes(mesh):
        view = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
        return int(view.num_banks * view.total_bytes_allocated_per_bank)

    def traces(gen):
        return {name: None if getattr(gen, name) is None else str(getattr(gen, name)) for name in TRACE_FIELDS}

    def counters(gen, before):
        return {name: int(value - before.get(name, 0)) for name, value in gen.counters.items()}

    def state_hashes(gen):
        result = []
        for index, layer in zip(gen.model.layer_indices, gen.kv_cache.prefill_layers):
            if layer.is_full_attention:
                continue
            for name, buffer in zip(
                ["recurrent"] + [f"conv_{i}" for i in range(len(layer.conv_state))],
                [layer.recurrent_state] + layer.conv_state,
            ):
                for rank, tensor in enumerate(ttnn.get_device_tensors(buffer)):
                    result.append(dict(layer=index, buffer=name, rank=rank, sha256=sha(ttnn.to_torch(tensor))))
        return result

    mesh = None
    gen = None
    try:
        report["phase"] = "open"
        save()
        mesh = open_ornith_mesh()
        model = OrnithModel(None, mesh, layer_indices=None if args.full32 else [0, 3])
        reference_lanes = {}
        # Candidate first exposes real first-use/new-shape compilation. The
        # explicit eager control then starts with a fresh owned cache and sampler.
        # Model weights are shared, and no two generators/traces coexist.
        for enabled in (True, False):
            name = "traced" if enabled else "eager_control"
            gen = OrnithGenerator(model, cache_context=None, use_prefill_trace=enabled)
            assert gen.owns_cache and gen.kv_cache.context == 262144 and gen.max_batch_size == 1
            assert all(key in gen.counters for key in PREFILL_COUNTERS)
            lane = dict(use_prefill_trace=enabled, prefill=[], configure=[], generation=[], phase="prefill")
            report["lanes"][name] = lane
            records = dict(prefill={}, configure={}, generation={})
            base_table = gen.page_table.clone()
            plain = [100] * 128
            changed = [101 + i % 7 for i in range(128)]
            cases = [
                ("initial128", plain, False, True, True),
                ("changed_tokens128", changed, False, False, True),
                ("changed_pages128", changed, True, False, True),
                ("new131", [103 + i % 5 for i in range(131)], True, True, True),
                ("return128", plain, False, True, True),
                ("live_miss131", [103 + i % 5 for i in range(131)], False, False, False),
                ("reuse_old128", plain, False, False, True),
            ]
            if args.edges or not args.quick:
                cases += [
                    ("edge1", [107], False, True, True),
                    ("edge2048", [100 + i % 11 for i in range(2048)], True, True, True),
                    ("after_edges128", plain, False, True, True),
                ]
            gen._configure_sampling(SamplingParams(temperature=0.0, top_k=1, top_p=1.0, seed=1234))
            for case, tokens, reverse, reset, eligible in cases:
                report["phase"] = f"{name}_{case}"
                save()
                print("PREFILL_INTEGRATION_START", name, case, flush=True)
                if reset:
                    gen.reset(clear_kv=False)
                table = base_table.flip(1).contiguous() if reverse else base_table
                old_key = gen._prefill_key
                old_input_addresses = (
                    [int(t.buffer_address()) for t in gen._prefill_inputs] if gen._prefill_inputs is not None else None
                )
                before = dict(gen.counters)
                output = gen.prefill_forward(
                    [tokens],
                    page_table=table,
                    kv_cache=gen.kv_cache,
                    prompt_lens=[len(tokens)],
                    return_device_logits=True,
                )
                delta = counters(gen, before)
                scores = model.logits_to_host(output, 1)
                states = state_hashes(gen)
                assert output is not gen._logits
                assert output.buffer_address() != gen._logits.buffer_address(), "public result aliases canonical logits"
                if enabled and eligible:
                    assert delta["prefill_replays"] == 1 and delta["prefill_eager_calls"] == 0, delta
                    assert delta["prefill_token_refreshes"] == 1, delta
                    assert gen._prefill_key[1] == len(tokens), gen._prefill_key
                    assert list(gen._prefill_inputs[0].shape) == [1, len(tokens)], gen._prefill_inputs[0].shape
                    assert all(getattr(gen, field) is not None for field in TRACE_FIELDS)
                    if old_key == gen._prefill_key:
                        assert delta["prefill_trace_misses"] == 0, delta
                        assert [int(t.buffer_address()) for t in gen._prefill_inputs] == old_input_addresses
                    else:
                        assert delta["prefill_trace_misses"] == 1, delta
                    if case == "changed_tokens128":
                        assert delta["prefill_page_table_refreshes"] == 0, delta
                    if case == "changed_pages128":
                        assert delta["prefill_page_table_refreshes"] == 1, delta
                else:
                    assert delta["prefill_replays"] == 0 and delta["prefill_eager_calls"] == 1, delta
                    if enabled:
                        assert gen._prefill_key == old_key, "live miss replaced the resident trace spec"
                # Public ownership is meaningful: free the result, then execute
                # decode and sampler traces using their intact canonical output.
                ttnn.deallocate(output)
                next_scores = gen.decode_forward(
                    [42], [len(tokens)], page_table=table, kv_cache=gen.kv_cache, return_logits=True
                )
                next_tokens = gen._read_tokens().tolist()
                next_states = state_hashes(gen)
                allocated = trace_bytes(mesh)
                assert 0 < allocated <= 100_000_000, allocated
                row = dict(
                    case=case,
                    logical_length=len(tokens),
                    reversed_pages=reverse,
                    reset_before=reset,
                    expected_trace=enabled and eligible,
                    public_owned=True,
                    public_deallocated_before_decode=True,
                    counters=delta,
                    trace_ids=traces(gen),
                    trace_bytes_per_device=allocated,
                    prefill_key=repr(gen._prefill_key),
                    prefill_sha256=sha(scores),
                    state_hashes=states,
                    next_decode_sha256=sha(next_scores),
                    next_tokens=next_tokens,
                    next_state_hashes=next_states,
                )
                lane["prefill"].append(row)
                records["prefill"][case] = dict(
                    scores=scores, states=states, next_scores=next_scores, next_states=next_states, tokens=next_tokens
                )
                save()
                print("PREFILL_INTEGRATION_PASS", name, case, json.dumps(delta), flush=True)

            assert not torch.equal(
                records["prefill"]["initial128"]["scores"], records["prefill"]["changed_tokens128"]["scores"]
            )
            assert torch.equal(
                records["prefill"]["changed_tokens128"]["scores"], records["prefill"]["changed_pages128"]["scores"]
            )
            for case in ("return128", "reuse_old128"):
                for key in ("scores", "next_scores"):
                    assert torch.equal(records["prefill"]["initial128"][key], records["prefill"][case][key])
                assert records["prefill"]["initial128"]["states"] == records["prefill"][case]["states"]

            # Reconfigure a live sampler while the prefill spec remains resident.
            # This must preserve the populated model state and decode inputs while
            # rebuilding all traces with the new seeded/penalty sampling graph.
            for index, (case, params) in enumerate(
                (
                    ("seeded", SamplingParams(temperature=0.8, top_k=20, top_p=0.95, seed=1234)),
                    (
                        "penalties",
                        SamplingParams(
                            temperature=0.8,
                            top_k=20,
                            top_p=0.95,
                            seed=5678,
                            presence_penalty=0.25,
                            frequency_penalty=0.1,
                            repetition_penalty=1.1,
                        ),
                    ),
                )
            ):
                report["phase"] = f"{name}_configure_{case}"
                save()
                states_before = state_hashes(gen)
                feedback_before = ttnn.to_torch(ttnn.get_device_tensors(gen._inputs[0])[0])
                position_before = ttnn.to_torch(ttnn.get_device_tensors(gen._inputs[1])[0])
                old_key = gen._prefill_key
                before = dict(gen.counters)
                gen.configure_sampling(
                    params,
                    reset_seed=True,
                    prompt_token_ids=[plain],
                    generated_token_ids=[[42] + list(range(43, 43 + index))],
                )
                delta = counters(gen, before)
                assert state_hashes(gen) == states_before, "live configure mutated model state"
                assert torch.equal(feedback_before, ttnn.to_torch(ttnn.get_device_tensors(gen._inputs[0])[0]))
                assert torch.equal(position_before, ttnn.to_torch(ttnn.get_device_tensors(gen._inputs[1])[0]))
                assert gen._prefill_key == old_key
                assert delta["model_replays"] == 0 and delta["prefill_replays"] == 0, delta
                if enabled:
                    assert delta["prefill_captures"] == 1, delta
                    assert all(getattr(gen, field) is not None for field in TRACE_FIELDS)
                scores = gen.decode_forward(
                    [43 + index], [129 + index], page_table=base_table, kv_cache=gen.kv_cache, return_logits=True
                )
                tokens = gen._read_tokens().tolist()
                states = state_hashes(gen)
                records["configure"][case] = dict(scores=scores, tokens=tokens, states=states)
                lane["configure"].append(
                    dict(case=case, counters=delta, scores_sha256=sha(scores), tokens=tokens, state_hashes=states)
                )
                save()
                print("CONFIGURE_INTEGRATION_PASS", name, case, json.dumps(delta), flush=True)

            lane["phase"] = "generation"
            generation_cases = [
                ("greedy", plain, 4 if args.quick else 8, None, False),
                (
                    "sampled",
                    [103 + i % 5 for i in range(131)],
                    4 if args.quick else 128,
                    SamplingParams(temperature=0.8, top_k=20, top_p=0.95, seed=1234),
                    True,
                ),
                (
                    "penalties",
                    changed,
                    4 if args.quick else 128,
                    SamplingParams(
                        temperature=0.8,
                        top_k=20,
                        top_p=0.95,
                        seed=5678,
                        presence_penalty=0.25,
                        frequency_penalty=0.1,
                        repetition_penalty=1.1,
                    ),
                    False,
                ),
                ("greedy_after_penalties", plain, 4 if args.quick else 260, None, False),
            ]
            for case, tokens, count, params, reverse in generation_cases:
                report["phase"] = f"{name}_generation_{case}"
                save()
                gen.page_table = base_table.flip(1).contiguous() if reverse else base_table.clone()
                before = dict(gen.counters)
                generated = gen.generate(tokens, count, sampling_params=params, stop_on_eos=False)
                delta = counters(gen, before)
                assert len(generated) == count
                assert delta["prefill_replays"] == int(enabled), delta
                assert delta["prefill_eager_calls"] == int(not enabled), delta
                assert delta["model_replays"] == count - 1, delta
                assert delta["history_replays"] == count - 1, delta
                if enabled:
                    assert delta["prefill_sampling_replays"] == 1, delta
                    assert all(getattr(gen, field) is not None for field in TRACE_FIELDS)
                else:
                    assert delta["prefill_sampling_replays"] == 0, delta
                scores = model.logits_to_host(gen._logits, 1)
                states = state_hashes(gen)
                records["generation"][case] = dict(tokens=generated, scores=scores, states=states)
                lane["generation"].append(
                    dict(
                        case=case,
                        count=count,
                        tokens=generated,
                        counters=delta,
                        perf=dict(gen.perf),
                        logits_sha256=sha(scores),
                        state_hashes=states,
                        trace_ids=traces(gen),
                        trace_bytes_per_device=trace_bytes(mesh),
                    )
                )
                save()
                print("GENERATION_INTEGRATION_PASS", name, case, count, json.dumps(delta), flush=True)

            # Exercise explicit four-ID release and reconstruction while keeping
            # the same weights/cache, then prove public teardown frees all traces.
            gen.reset(clear_kv=False)
            gen._release_traces()
            assert all(getattr(gen, field) is None for field in TRACE_FIELDS)
            assert trace_bytes(mesh) == 0
            if enabled:
                assert gen._prefill_inputs is not None and gen._prefill_key is not None
            gen.ensure_traces()
            assert gen._model_trace is not None
            if enabled:
                assert all(getattr(gen, field) is not None for field in TRACE_FIELDS)
            gen.teardown()
            assert all(getattr(gen, field) is None for field in TRACE_FIELDS)
            assert gen._prefill_inputs is None and gen._prefill_key is None
            lane["trace_bytes_after_teardown_per_device"] = trace_bytes(mesh)
            assert lane["trace_bytes_after_teardown_per_device"] == 0
            lane["phase"] = "complete"
            tensor_path = args.output.with_name(args.output.stem + f"_{name}_expected.pt")
            torch.save(records, tensor_path)
            lane["tensor_artifact"] = str(tensor_path)
            reference_lanes[name] = records
            # Release the old owned cache before constructing the control lane.
            del gen
            gen = None
            model.cache = None
            gc.collect()
            save()

        candidate, control = reference_lanes["traced"], reference_lanes["eager_control"]
        for group in ("prefill", "configure", "generation"):
            assert candidate[group].keys() == control[group].keys()
            for case, actual in candidate[group].items():
                expected = control[group][case]
                checks = {
                    key: (
                        bool(torch.equal(value, expected[key]))
                        if isinstance(value, torch.Tensor)
                        else value == expected[key]
                    )
                    for key, value in actual.items()
                }
                report["comparisons"].append(dict(group=group, case=case, **checks))
                assert all(checks.values()), (group, case, checks)
        report["phase"] = "complete"
        report["probe_execution_pass"] = True
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
    print("PREFILL_INTEGRATION_OK", args.output, flush=True)


if __name__ == "__main__":
    main()
