# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reduced real B1 admission timing; supervising serialized hardware lane only.

Measures production warmup and two normal requests without adding synchronization
inside submission. No profiler. This localizes first-use host work and is not a
full-model latency benchmark. Run fresh processes for before/after comparisons.
"""

import argparse
import functools
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

import ttnn
from models.common.sampling import SamplingParams

from ..doc.vllm_integration.logit_determinism_vllm import PROMPTS
from ..reference.hf_reference import HF_REVISION
from ..tt.functional_decoder import num_blocks_for_context
from ..tt.generator_vllm import TTOrnithForCausalLM
from ..tt.model import close_ornith_mesh, open_ornith_mesh


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def token_output(value):
    return (value[0] if isinstance(value, tuple) else value).reshape(-1).to(torch.int64).clone()


class HostSpans:
    """Instance-only observation: queries do not synchronize or mutate TTNN."""

    def __init__(self, adapter, records):
        self.adapter = adapter
        self.gen = adapter.generator
        self.records = records
        self.phase = "setup"

    def snapshot(self):
        return {
            "programs": self.gen.mesh_device.num_program_cache_entries(),
            "captured_programs": self.gen._programs,
            "trace_ids": {
                name: str(getattr(self.gen, name))
                for name in ("_model_trace", "_sampling_trace", "_sampling_history_trace")
            },
            "adapter_sampling_key_set": self.adapter._sampling_key is not None,
            "generator_sampling_key_set": self.gen._sampling_key is not None,
            "live": self.gen._live,
        }

    def wrap(self, owner, name):
        original = getattr(owner, name)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            before = self.snapshot()
            record = {"phase": self.phase, "method": name, "before": before}
            if name == "_merge_serving_vector":
                target = args[0]
                record["target"] = {"shape": list(target.shape), "dtype": str(target.dtype), "counter": args[3]}
            self.records.append(record)
            start = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                record["host_ms"] = (time.perf_counter_ns() - start) / 1e6
                record["after"] = self.snapshot()
                record["program_delta"] = record["after"]["programs"] - before["programs"]

        setattr(owner, name, measured)

    def install(self):
        for name in (
            "_merge_serving_vector",
            "refresh_serving_inputs",
            "_capture",
            "_ensure_replay_safe",
            "ensure_traces",
            "configure_sampling",
        ):
            self.wrap(self.gen, name)
        for name in ("prefill_forward", "decode_forward"):
            self.wrap(self.adapter, name)


def production_warmup(adapter, cache, spans):
    """Mirror pinned plugin model_runner.warmup_model's four-call ordering."""
    results = []
    for enable_trace in (False, True):
        for kind in ("prefill", "decode"):
            spans.phase = f"startup.{kind}.trace_{enable_trace}"
            before = spans.snapshot()
            start = time.perf_counter_ns()
            getattr(adapter, f"warmup_model_{kind}")(
                kv_cache=cache,
                enable_trace=enable_trace,
                can_sample_on_device=True,
                max_batch_size=1,
                num_blocks=adapter.page_table_blocks,
            )
            results.append(
                {
                    "phase": spans.phase,
                    "host_ms": (time.perf_counter_ns() - start) / 1e6,
                    "before": before,
                    "after": spans.snapshot(),
                }
            )
    return results


def clean_startup_state(adapter):
    """Small diagnostic reads only; no full KV or full-logit readback."""
    gen = adapter.generator
    vectors = {}
    for name, target in zip(("tokens", "positions", "rope"), gen._inputs):
        values = [ttnn.to_torch(shard).reshape(-1).tolist() for shard in ttnn.get_device_tensors(target)]
        assert all(value == values[0] for value in values)
        vectors[name] = values[0]
    state_buffers = 0
    for layer in gen.kv_cache.decode_layers:
        if not layer.is_full_attention:
            for target in [layer.recurrent_state, *layer.conv_state]:
                for shard in ttnn.get_device_tensors(target):
                    assert torch.count_nonzero(ttnn.to_torch(shard)) == 0
                state_buffers += 1
    shadow = gen.sampling.tt_penalties._prompt_tokens_host
    empty_prompt = shadow is None or bool((shadow < 0).all())
    history_index = ttnn.to_torch(ttnn.get_device_tensors(gen._output_history_index)[0]).reshape(-1).tolist()
    seed_values_match = all(
        torch.equal(ttnn.to_torch(shard).reshape(-1).to(torch.int32), gen._seed_values.reshape(-1).to(torch.int32))
        for shard in ttnn.get_device_tensors(gen.sampling.tt_sampling.seeds_tt_tensor)
    )
    clean = (
        not gen._live
        and not adapter._device_rows.any()
        and not adapter._prefilled_rows.any()
        and not adapter._pending_device_seeds.any()
        and adapter._last_device_sampling is None
        and empty_prompt
        and gen._history_rows == 0
        and history_index == [0]
        and seed_values_match
        and all(value == 0 for vector in vectors.values() for value in vector)
    )
    return {
        "clean": bool(clean),
        "vectors": vectors,
        "recurrent_conv_buffers_zero_all_replicas": state_buffers,
        "prompt_history_empty": empty_prompt,
        "generated_history_rows": gen._history_rows,
        "generated_history_index": history_index,
        "device_seeds_match_reset_host_values": seed_values_match,
        "last_device_sampling": adapter._last_device_sampling,
        "device_rows": adapter._device_rows.tolist(),
        "prefilled_rows": adapter._prefilled_rows.tolist(),
        "pending_device_seeds": adapter._pending_device_seeds.tolist(),
    }


def request(adapter, cache, table, prompts, sampling, spans, number):
    lengths = torch.tensor([prompts.shape[-1]], dtype=torch.int64)
    spans.phase = f"request{number}.prefill"
    before = spans.snapshot()
    start = time.perf_counter_ns()
    first = token_output(
        adapter.prefill_forward(
            tokens=prompts,
            prompt_lens=lengths.tolist(),
            empty_slots=[0],
            page_table=table,
            kv_cache=cache,
            sampling_params=sampling,
        )
    )
    result = {
        "prefill_ms": (time.perf_counter_ns() - start) / 1e6,
        "before_prefill": before,
        "after_prefill": spans.snapshot(),
        "tokens": [first.tolist()],
        "decode_steps": [],
    }
    last = first
    for step in range(3):
        spans.phase = f"request{number}.decode{step + 1}"
        before = spans.snapshot()
        start = time.perf_counter_ns()
        output = adapter.decode_forward(
            tokens=last[:, None],
            start_pos=lengths + step,
            page_table=table,
            kv_cache=cache,
            sampling_params=sampling,
            enable_trace=True,
            read_from_device=False,
            reset_batch=step == 0,
        )
        submitted = time.perf_counter_ns()
        hosts, events = adapter.read_decode_output(output, async_read=True)
        copied = time.perf_counter_ns()
        for event in events:
            ttnn.event_synchronize(event)
        last = adapter.process_decode_output_host(hosts, is_tokens=True).clone()
        finished = time.perf_counter_ns()
        result["tokens"].append(last.tolist())
        result["decode_steps"].append(
            {
                "phase": spans.phase,
                "submission_ms": (submitted - start) / 1e6,
                "copy_submission_ms": (copied - submitted) / 1e6,
                "finalize_ms": (finished - copied) / 1e6,
                "total_ms": (finished - start) / 1e6,
                "before": before,
                "after": spans.snapshot(),
            }
        )
    for value in result["tokens"]:
        assert len(value) == 1 and 0 <= value[0] < adapter.model.vocab_size
    for target in adapter.generator._inputs[1:3]:
        for shard in ttnn.get_device_tensors(target):
            assert ttnn.to_torch(shard).reshape(-1).tolist() == [131]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Before-fix probe JSON for exact first/repeated output parity")
    parser.add_argument("--require-trace-allocation-tracking", action="store_true")
    parser.add_argument("--require-clean-warmup", action="store_true")
    args = parser.parse_args()
    os.environ["ORNITH_MODEL_PATH"] = str(args.model_path.resolve())
    os.environ["ORNITH_VLLM_LAYER_INDICES"] = "0,3"
    os.environ["ORNITH_VLLM_ALLOW_HOST_SAMPLING"] = "0"
    model_root = Path(__file__).resolve().parents[1]
    report = {
        "scope": "reduced real-weight B1 production startup/admission host attribution; no full-model timing claim",
        "hardware": "four Blackhole chips on P300c, TP4",
        "layers": [0, 3],
        "batch": 1,
        "native_context": 262144,
        "checkpoint_revision": HF_REVISION,
        "trace_allocation_tracking": bool(ttnn.TRACE_ALLOC_TRACKING),
        "source_sha256": {
            str(path.relative_to(model_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__),
                model_root / "tt/generator_vllm.py",
                model_root / "tt/generator.py",
                model_root / "tt/model.py",
            ]
        },
        "host_spans": [],
        "requests": [],
        "cleanup_completed": False,
        "passed": False,
    }
    save(args.output, report)
    mesh = adapter = None
    try:
        if args.require_trace_allocation_tracking:
            assert ttnn.TRACE_ALLOC_TRACKING and os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
        config = AutoConfig.from_pretrained(args.model_path, local_files_only=True, revision=HF_REVISION)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, revision=HF_REVISION)
        ids = tokenizer.encode(PROMPTS["A"], add_special_tokens=False)[:128]
        assert len(ids) == 128
        prompts = torch.tensor([ids], dtype=torch.int64)
        report["prompt"] = {"token_ids": ids, "text": tokenizer.decode(ids), "length": len(ids)}
        mesh = open_ornith_mesh()
        adapter = TTOrnithForCausalLM.initialize_vllm_model(config, mesh, max_batch_size=1, max_seq_len=262144)
        assert adapter.model.layer_indices == [0, 3]
        width = num_blocks_for_context(adapter.max_model_len, adapter.model.page_block_size)
        physical_blocks = width + 1
        heads = max(1, adapter.model.hf_config.num_key_value_heads // mesh.get_num_devices())
        cache = adapter.allocate_kv_cache(
            (physical_blocks, heads, adapter.model.page_block_size, adapter.model.hf_config.head_dim),
            torch.bfloat16,
            len(adapter.model.layers),
        )
        gen = adapter.generator
        assert cache is gen.kv_cache and not gen.owns_cache
        assert cache.context == 262144 and cache.num_blocks == physical_blocks
        assert all(pair is None or pair[0].shape[0] == physical_blocks for pair in cache.kv)
        cache_id = id(cache)
        addresses = [value.buffer_address() for value in gen._inputs]
        table = torch.arange(1, width + 1, dtype=torch.int32).reshape(1, width)
        report.update(
            physical_blocks=physical_blocks,
            page_block_size=adapter.model.page_block_size,
            selected_precision=adapter.model.precision,
        )
        spans = HostSpans(adapter, report["host_spans"])
        spans.install()
        report["startup"] = production_warmup(adapter, cache, spans)
        report["after_startup"] = spans.snapshot()
        report["after_startup_state"] = clean_startup_state(adapter)
        if args.require_clean_warmup:
            assert report["after_startup_state"]["clean"], report["after_startup_state"]
        assert not gen._live and not adapter._device_rows.any() and not adapter._prefilled_rows.any()
        # Pinned plugin expands unrestricted top_k to vocab_size and represents
        # disabled logprobs with -2. Canonical formatting maps greedy k/p back
        # to 1/0, so raw-key differences must not rewarm the model trace.
        sampling = SamplingParams(temperature=0.0, top_k=adapter.model.vocab_size, top_p=1.0, seed=7, num_logprobs=-2)
        before_adapter = dict(adapter.counters)
        for number in (1, 2):
            # Real start_pos=0 admission resets its row. Retain the adapter and
            # sampling key across requests, just as the production server does.
            result = request(adapter, cache, table, prompts, sampling, spans, number)
            report["requests"].append(result)
            save(args.output, report)
            print(
                json.dumps(
                    {
                        "request": number,
                        "prefill_ms": result["prefill_ms"],
                        "decode_ms": [item["total_ms"] for item in result["decode_steps"]],
                        "tokens": result["tokens"],
                    }
                ),
                flush=True,
            )
        assert report["requests"][0]["tokens"] == report["requests"][1]["tokens"]
        if args.reference:
            reference = json.loads(args.reference.read_text())
            assert report["prompt"] == reference["prompt"]
            assert [item["tokens"] for item in report["requests"]] == [item["tokens"] for item in reference["requests"]]
            report["reference"] = {
                "path": str(args.reference),
                "sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                "exact_tokens": True,
            }
        assert id(gen.kv_cache) == cache_id
        assert [value.buffer_address() for value in gen._inputs] == addresses
        report["cache_identity_unchanged"] = report["persistent_input_addresses_unchanged"] = True
        report["adapter_counters"] = dict(adapter.counters)
        report["generator_counters"] = dict(gen.counters)
        report["request_adapter_counter_delta"] = {
            key: value - before_adapter.get(key, 0) for key, value in adapter.counters.items()
        }
        assert report["request_adapter_counter_delta"]["host_decodes"] == 0
        assert report["request_adapter_counter_delta"]["device_decodes"] == 6
        report["passed"] = True
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        save(args.output, report)
        try:
            if adapter is not None:
                adapter.teardown()
        finally:
            if mesh is not None:
                close_ornith_mesh(mesh)
        report["cleanup_completed"] = True
        save(args.output, report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "cleanup_completed": report["cleanup_completed"],
                "captures": [
                    {"phase": item["phase"], "host_ms": item["host_ms"], "program_delta": item["program_delta"]}
                    for item in report["host_spans"]
                    if item["method"] == "_capture"
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
