# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""All32 native serving-capacity gate through public adapter device sampling.

Run only in the supervising hardware lane. Uses external4098 blocks, B1,
context262144 and the selected precision. No profiler or host logits.
Imports are deferred so authoring checks and --help cannot access devices.
"""

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

TRACE_FIELDS = ("_model_trace", "_sampling_trace", "_sampling_history_trace", "_prefill_trace")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-trace-allocation-tracking", action="store_true")
    args = parser.parse_args()

    import torch

    import ttnn
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator_vllm import TTOrnithForCausalLM
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import OrnithModel, close_ornith_mesh, open_ornith_mesh
    from models.common.sampling import SamplingParams

    report = dict(
        native_context=262144,
        physical_blocks=4098,
        batch=1,
        layers=list(range(32)),
        mesh=[1, 4],
        capability_reduction=None,
        measurements={},
        windows=[],
        probe_execution_pass=False,
        trace_allocation_tracking=dict(
            enabled=bool(ttnn.TRACE_ALLOC_TRACKING),
            skip_program_cache=os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0"),
        ),
        environment={
            name: os.environ.get(name)
            for name in (
                "TT_METAL_WATCHER",
                "TT_METAL_WATCHER_DISABLE_ETH",
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
            )
        },
        source_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((Path(__file__).parents[1] / "tt").glob("*.py"))
        },
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

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

    def sampler_buffers():
        result = {}

        def visit(name, value):
            if isinstance(value, ttnn.Tensor):
                if value.storage_type() == ttnn.StorageType.HOST:
                    return
                result[name] = dict(
                    shape=list(value.shape),
                    dtype=str(value.dtype),
                    layout=str(value.layout),
                    address=int(value.buffer_address()),
                    memory_config=str(value.memory_config()),
                )
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    visit(f"{name}[{index}]", item)
            elif isinstance(value, dict):
                for key, item in value.items():
                    visit(f"{name}.{key}", item)

        for name in (
            "_inputs",
            "_output_history",
            "_output_history_index",
            "_prefill_sampling_logits",
            "_prefill_sampling_saved",
            "_prefill_sampling_masks",
        ):
            visit("generator." + name, getattr(gen, name, None))
        for name in ("tt_sampling", "tt_penalties"):
            for key, value in vars(getattr(gen.sampling, name)).items():
                visit(f"sampling.{name}.{key}", value)
        return result

    def token(output):
        value = output[0] if isinstance(output, tuple) else output
        assert isinstance(value, torch.Tensor) and value.numel() == 1
        assert not value.is_floating_point(), "adapter returned logits instead of device-sampled IDs"
        result = int(value.reshape(-1)[0])
        assert 0 <= result < model.vocab_size
        return result

    mesh = adapter = gen = None
    try:
        if args.require_trace_allocation_tracking:
            assert ttnn.TRACE_ALLOC_TRACKING, "Enable native allocation tracking before Python starts"
            assert report["trace_allocation_tracking"]["skip_program_cache"] == "0"
        report["phase"] = "open"
        save()
        mesh = open_ornith_mesh()
        report["measurements"]["opened"] = memory()
        model = OrnithModel(None, mesh)
        assert model.layer_indices == list(range(32))
        report["precision"] = model.precision
        report["measurements"]["loaded_weights_constants"] = memory()
        adapter = TTOrnithForCausalLM(model, max_batch_size=1, max_model_len=262144, hf_config=model.hf_config)
        heads = max(1, model.hf_config.num_key_value_heads // mesh.get_num_devices())
        cache = adapter.allocate_kv_cache(
            (4098, heads, model.page_block_size, model.hf_config.head_dim), torch.bfloat16, len(model.layers)
        )
        gen = adapter.generator
        assert cache is gen.kv_cache and not gen.owns_cache
        assert cache.num_blocks == 4098 and cache.context == 262144 and model.page_block_size == 64
        table = torch.arange(adapter.page_table_blocks, dtype=torch.int32).reshape(1, -1)
        assert table.shape == (1, 4096)
        cache_addresses = [int(t.buffer_address()) for t in model.cache_buffers(cache)]
        report["measurements"]["allocated_external_native_pool"] = memory()
        report["phase"] = "startup"
        save()
        adapter.warmup_model_prefill(cache, enable_trace=True)
        assert gen._prefill_key[1] == 128 and all(getattr(gen, name) is not None for name in TRACE_FIELDS)
        # This gate must actually include the new persistent sampling allocation.
        assert len(getattr(gen, "_prefill_sampling_saved", [])) >= 2, "Apply the persistent prefill sampler candidate"
        assert gen._prefill_sampling_logits is not None
        resident_key = gen._prefill_key
        report["resident_prefill_key"] = repr(resident_key)
        report["measurements"]["startup_traces_sampler_ready"] = memory()
        report["sampler_buffers_after_startup"] = sampler_buffers()
        params = SamplingParams(temperature=[0.0], top_k=[1], top_p=[1.0], num_logprobs=[-2])
        for length in (262143, 262144):
            report["phase"] = f"prefill_{length}"
            save()
            print("NATIVE_SERVING_START", length, flush=True)
            before = dict(gen.counters)
            started = time.perf_counter()
            first = token(
                adapter.prefill_forward(
                    tokens=torch.full((1, length), 100, dtype=torch.int32),
                    page_table=table,
                    kv_cache=cache,
                    prompt_lens=[length],
                    empty_slots=[0],
                    sampling_params=params,
                    enable_trace=True,
                )
            )
            elapsed = time.perf_counter() - started
            delta = {name: value - before[name] for name, value in gen.counters.items()}
            assert delta["prefill_eager_calls"] == 1 and delta["prefill_replays"] == 0, delta
            assert delta["prefill_sampling_replays"] == 1, "first-token sampling must replay the canonical trace"
            assert gen._prefill_key == resident_key
            item = dict(logical_prompt_length=length, returned_token=first, prefill_s=elapsed, prefill_counters=delta)
            report["measurements"][f"after_prefill_{length}"] = memory()
            if length == 262143:
                before_decode = dict(gen.counters)
                out = adapter.decode_forward(
                    tokens=torch.tensor([[first]], dtype=torch.int64),
                    start_pos=torch.tensor([length]),
                    page_table=table,
                    kv_cache=cache,
                    sampling_params=params,
                    reset_batch=True,
                    enable_trace=True,
                    read_from_device=False,
                )
                hosts, events = adapter.read_decode_output(out, async_read=True)
                for event in events:
                    ttnn.event_synchronize(event)
                decoded = token(adapter.process_decode_output_host(hosts, is_tokens=True))
                positions = [
                    int(ttnn.to_torch(shard).reshape(-1)[0]) for shard in ttnn.get_device_tensors(gen._inputs[1])
                ]
                rope = [int(ttnn.to_torch(shard).reshape(-1)[0]) for shard in ttnn.get_device_tensors(gen._inputs[2])]
                assert positions == [262144] * 4 and rope == positions, (positions, rope)
                assert gen.counters["model_replays"] - before_decode["model_replays"] == 1
                assert gen.counters["sampling_replays"] - before_decode["sampling_replays"] == 1
                item.update(
                    last_decode_position=length, decode_token=decoded, advanced_positions=positions, rope_positions=rope
                )
                report["measurements"]["after_last_valid_decode"] = memory()
            assert (
                cache is gen.kv_cache
                and [int(t.buffer_address()) for t in model.cache_buffers(cache)] == cache_addresses
            )
            assert gen._prefill_key == resident_key and all(getattr(gen, name) is not None for name in TRACE_FIELDS)
            assert adapter.counters["host_decodes"] == 0
            report["windows"].append(item)
            report["sampler_buffers_after_windows"] = sampler_buffers()
            save()
            print("NATIVE_SERVING_WINDOW_OK", json.dumps(item), flush=True)
        report["generator_counters"] = dict(gen.counters)
        report["adapter_counters"] = dict(adapter.counters)
        adapter.teardown()
        report["measurements"]["after_teardown"] = memory()
        assert report["measurements"]["after_teardown"]["TRACE"]["total_allocated_bytes_per_device"] == 0
        assert all(getattr(gen, name) is None for name in TRACE_FIELDS)
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
    print("NATIVE_SERVING_CAPACITY_OK", args.output, flush=True)


if __name__ == "__main__":
    main()
