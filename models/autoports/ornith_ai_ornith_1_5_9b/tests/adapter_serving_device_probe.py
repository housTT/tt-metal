# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real-weight adapter async proof; the supervising hardware lane runs this.

This opens a TP4 mesh. Authoring/static checks must not import this module.
It checks reduced layers 0 and 3 with native logical context, a shared physical
pool, and device sampling. It is not a full-model accuracy or performance gate.
"""

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.common.sampling import SamplingParams

from ..reference.hf_reference import HF_REVISION
from ..tt.functional_decoder import num_blocks_for_context
from ..tt.generator_vllm import TTOrnithForCausalLM
from ..tt.model import close_ornith_mesh, open_ornith_mesh


def delta(before, after):
    return {key: after.get(key, 0) - before.get(key, 0) for key in set(before) | set(after)}


def shards(tensor):
    return [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(tensor)]


def replicated_vector(tensor, batch):
    values = [value.reshape(-1)[:batch].to(torch.int64) for value in shards(tensor)]
    for other in values[1:]:
        torch.testing.assert_close(other, values[0], rtol=0, atol=0)
    return values[0]


def state(adapter):
    gen = adapter.generator
    return {
        "tokens": replicated_vector(gen._inputs[0], adapter.max_batch_size),
        "positions": replicated_vector(gen._inputs[1], adapter.max_batch_size),
        "rope": replicated_vector(gen._inputs[2], adapter.max_batch_size),
        "addresses": [tensor.buffer_address() for tensor in gen._inputs],
    }


def page_snapshot(adapter, physical_block):
    result = []
    for pair in adapter.generator.kv_cache.kv:
        if pair is None:
            continue
        for tensor in pair:
            selected = ttnn.slice(tensor, [physical_block, 0, 0, 0], [physical_block + 1, *list(tensor.shape)[1:]])
            try:
                result.extend(shards(selected))
            finally:
                ttnn.deallocate(selected)
    return result


def tokens_only(output):
    # Prefill also returns zero RoPE deltas when the real HF config uses mRoPE.
    return (output[0] if isinstance(output, tuple) else output).reshape(-1).to(torch.int64).clone()


def prepare(adapter, prompts, lengths, table, sampling):
    """Reset only the A/B fixture; all inference goes through adapter methods."""
    gen = adapter.generator
    gen.reset(clear_kv=True)
    adapter._device_rows[:] = False
    adapter._prefilled_rows[:] = False
    adapter._last_device_sampling = None
    adapter._sampling_key = None
    first = tokens_only(
        adapter.prefill_forward(
            tokens=prompts,
            prompt_lens=lengths.tolist(),
            empty_slots=list(range(adapter.max_batch_size)),
            page_table=table,
            kv_cache=gen.kv_cache,
            sampling_params=sampling,
        )
    )
    output = adapter.decode_forward(
        tokens=first[:, None],
        start_pos=lengths,
        page_table=table,
        kv_cache=gen.kv_cache,
        sampling_params=sampling,
        reset_batch=True,
        read_from_device=True,
    ).clone()
    initial = state(adapter)
    torch.testing.assert_close(initial["positions"], lengths + 1, rtol=0, atol=0)
    torch.testing.assert_close(initial["rope"], lengths + 1, rtol=0, atol=0)
    torch.testing.assert_close(initial["tokens"], output, rtol=0, atol=0)
    assert gen._model_trace is not None and gen._sampling_trace is not None
    return output, initial


def finalize(adapter, pending):
    hosts, events = pending
    for event in events:
        ttnn.event_synchronize(event)
    result = adapter.process_decode_output_host(hosts, is_tokens=True).clone()
    assert result.shape == (adapter.max_batch_size,)
    assert bool(((result >= 0) & (result < adapter.model.vocab_size)).all())
    return result


def run_pair(adapter, prompts, lengths, initial_table, sampling, *, mode, spare_block):
    """Queue copy N before replay N+1, without formatting either deferred copy."""
    table = initial_table.clone()
    last, initial = prepare(adapter, prompts, lengths, table, sampling)
    batch = adapter.max_batch_size
    remap = torch.tensor([batch - 1, *range(batch - 1)], dtype=torch.int64) if mode == "remap" else None
    changed = mode == "changed_page"
    changed_row = 1
    logical_block = 1
    # Row 1 starts at length 62. Warm decode -> 63, first pair step -> 64.
    # The second pair step writes the first live token of logical page 1.
    assert int(lengths[changed_row]) + 2 == adapter.model.page_block_size
    original_block = int(table[changed_row, logical_block])
    old_page = page_snapshot(adapter, original_block) if changed else None
    spare_page = page_snapshot(adapter, spare_block) if changed else None
    before_gen, before_adapter = dict(adapter.generator.counters), dict(adapter.counters)
    pending, outputs = [], []
    for step in range(2):
        if remap is not None and step == 0:
            table = table[remap].clone()
        if changed and step == 1:
            table[changed_row, logical_block] = spare_block
        if mode == "synchronous":
            supplied_tokens = last[:, None]
            supplied_positions = lengths + 1 + step
        else:
            # Valid vocabulary IDs and valid, deliberately wrong positions keep
            # a broken refresh observable without an out-of-bounds device access.
            supplied_tokens = ((initial["tokens"] + 100003 + step) % adapter.model.vocab_size)[:, None]
            supplied_positions = torch.arange(batch, dtype=torch.int64) + 2 + step
            assert not torch.equal(supplied_tokens.flatten(), initial["tokens"])
            assert bool((supplied_positions != lengths + 1 + step).all())
        raw = adapter.decode_forward(
            tokens=supplied_tokens,
            start_pos=supplied_positions,
            page_table=table,
            kv_cache=adapter.generator.kv_cache,
            sampling_params=sampling,
            enable_trace=True,
            read_from_device=mode == "synchronous",
            reset_batch=remap is not None and step == 0,
            slot_remap=remap if step == 0 else None,
        )
        if mode == "synchronous":
            last = raw.clone()
            outputs.append(last)
        else:
            hosts, events = adapter.read_decode_output(raw, async_read=True)
            assert len(hosts) == len(events) == 1
            assert hosts[0] is not raw, "Async output must own a host copy, not the reused device tensor"
            if pending:
                assert hosts[0] is not pending[0][0][0], "Two decode steps must own distinct host copies"
            pending.append((hosts, events))
    # Both replays and output copies have been submitted before the first wait.
    outputs.extend(finalize(adapter, item) for item in pending)
    observed = state(adapter)
    expected_positions = lengths + 3
    if remap is not None:
        expected_positions = expected_positions[remap]
    torch.testing.assert_close(observed["positions"], expected_positions, rtol=0, atol=0)
    torch.testing.assert_close(observed["rope"], expected_positions, rtol=0, atol=0)
    torch.testing.assert_close(observed["tokens"], outputs[-1], rtol=0, atol=0)
    assert observed["addresses"] == initial["addresses"]
    for device_table in shards(adapter.generator._inputs[3]):
        torch.testing.assert_close(device_table, table, rtol=0, atol=0, check_dtype=False)

    counters = delta(before_gen, adapter.generator.counters)
    adapter_counters = delta(before_adapter, adapter.counters)
    assert counters["model_replays"] == counters["sampling_replays"] == 2
    for name in ("token_refreshes", "position_refreshes", "rope_refreshes"):
        assert counters[name] == 0, (mode, name, counters)
    assert counters["page_table_refreshes"] == int(changed or remap is not None)
    assert counters.get("slot_remaps", 0) == int(remap is not None)
    assert adapter_counters["device_decodes"] == 2 and adapter_counters["host_decodes"] == 0
    assert adapter_counters["async_reads"] == (0 if mode == "synchronous" else 2)
    page_proof = None
    if changed:
        for before, after in zip(old_page, page_snapshot(adapter, original_block)):
            torch.testing.assert_close(after, before, rtol=0, atol=0)
        for before, after in zip(spare_page, page_snapshot(adapter, spare_block)):
            assert not torch.equal(before, after), "Every K/V shard must update the new scheduler-owned page"
        page_proof = {
            "row": changed_row,
            "logical_block": logical_block,
            "old_physical_block": original_block,
            "new_physical_block": spare_block,
            "old_page_unchanged": True,
            "new_page_written_all_kv_shards": True,
        }
    return {
        "outputs": [value.tolist() for value in outputs],
        "initial_tokens": initial["tokens"].tolist(),
        "initial_positions": initial["positions"].tolist(),
        "final_positions": observed["positions"].tolist(),
        "final_rope_positions": observed["rope"].tolist(),
        "persistent_addresses_unchanged": True,
        "deferred_pair_submitted_before_wait": mode != "synchronous",
        "generator_counter_delta": counters,
        "adapter_counter_delta": adapter_counters,
        "page_boundary": page_proof,
        "slot_remap": None if remap is None else remap.tolist(),
    }


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, choices=(3, 4), default=3)
    parser.add_argument(
        "--require-trace-allocation-tracking",
        action="store_true",
        help="Require the native pre-replay guard; set TT_METAL_TRACE_ALLOC_TRACKING=1 before process startup.",
    )
    args = parser.parse_args()
    os.environ["ORNITH_MODEL_PATH"] = str(args.model_path.resolve())
    os.environ["ORNITH_VLLM_LAYER_INDICES"] = "0,3"
    os.environ["ORNITH_VLLM_ALLOW_HOST_SAMPLING"] = "0"
    report = {
        "status": "running",
        "hardware": "four Blackhole chips on two P300c boards, TP4",
        "layers": [0, 3],
        "batch": args.batch,
        "logical_context": 262144,
        "hf_revision": HF_REVISION,
        "scope": "reduced real-weight adapter contract; no full-model accuracy or performance claim",
        "trace_allocation_tracking": {
            "required": args.require_trace_allocation_tracking,
            "enabled_at_ttnn_import": bool(ttnn.TRACE_ALLOC_TRACKING),
            "skip_program_cache": os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0"),
            "tracebacks": os.environ.get("TT_METAL_TRACE_ALLOC_TRACEBACKS", "0"),
            "mechanism": "native ttnn.execute_trace verifies live tracked allocations before each replay",
            "scope": "conservative per-trace allocation lifetime accounting, not address-overlap measurement",
        },
        "cases": {},
        "source_sha256": {
            name: hashlib.sha256((Path(__file__).parents[1] / "tt" / name).read_bytes()).hexdigest()
            for name in ("generator_vllm.py", "generator.py", "model.py")
        },
    }
    mesh, adapter = None, None
    write_report(args.output, report)
    try:
        if args.require_trace_allocation_tracking:
            if not ttnn.TRACE_ALLOC_TRACKING:
                raise RuntimeError("Set TT_METAL_TRACE_ALLOC_TRACKING=1 before importing TTNN")
            if os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "0":
                raise RuntimeError("Program-cache allocations must remain included in the native tracker")
        hf_config = AutoConfig.from_pretrained(args.model_path, local_files_only=True, revision=HF_REVISION)
        mesh = open_ornith_mesh()
        adapter = TTOrnithForCausalLM.initialize_vllm_model(
            hf_config, mesh, max_batch_size=args.batch, max_seq_len=262144
        )
        assert adapter.model.layer_indices == [0, 3]
        assert {layer.is_full_attention for layer in adapter.model.layers} == {False, True}
        width = num_blocks_for_context(adapter.max_model_len, adapter.model.page_block_size)
        request_blocks = num_blocks_for_context(256, adapter.model.page_block_size)
        # paged_fused_update_cache requires physical blocks >= page-table width.
        # This stays below B full native allocations while satisfying that bound.
        physical_blocks = width + args.batch * request_blocks
        heads = max(1, adapter.model.hf_config.num_key_value_heads // mesh.get_num_devices())
        cache = adapter.allocate_kv_cache(
            (physical_blocks, heads, adapter.model.page_block_size, adapter.model.hf_config.head_dim),
            torch.bfloat16,
            len(adapter.model.layers),
        )
        assert cache is adapter.generator.kv_cache and not adapter.generator.owns_cache
        assert cache.context == 262144 and width <= physical_blocks < args.batch * width
        assert all(pair is None or pair[0].shape[0] == physical_blocks for pair in cache.kv)
        table = torch.zeros(args.batch, width, dtype=torch.int32)
        table[:, :request_blocks] = torch.arange(1, 1 + args.batch * request_blocks, dtype=torch.int32).reshape(
            args.batch, request_blocks
        )
        spare_block = 1 + args.batch * request_blocks
        assert spare_block < physical_blocks
        lengths = torch.arange(61, 61 + args.batch, dtype=torch.int64)
        prompts = [torch.arange(17 + 100 * row, 17 + 100 * row + int(length)) for row, length in enumerate(lengths)]
        sampling = SamplingParams(temperature=0.0, top_k=1, top_p=1.0, seed=42)
        adapter.warmup_model_decode(cache)
        report.update(
            physical_blocks=physical_blocks,
            native_page_table_width=width,
            kv_cache_dtype=adapter.model.precision["kv_cache_dtype"],
            prompt_lengths=lengths.tolist(),
            capabilities_at_start=dict(adapter.model_capabilities),
        )
        for mode in ("synchronous", "deferred_stale", "changed_page", "remap"):
            print(f"ADAPTER_PROBE_BEGIN {mode}", flush=True)
            result = run_pair(adapter, prompts, lengths, table, sampling, mode=mode, spare_block=spare_block)
            baseline = report["cases"].get("synchronous")
            if baseline is not None:
                assert result["initial_tokens"] == baseline["initial_tokens"]
                expected = torch.tensor(baseline["outputs"])
                if result["slot_remap"] is not None:
                    expected = expected[:, result["slot_remap"]]
                torch.testing.assert_close(torch.tensor(result["outputs"]), expected, rtol=0, atol=0)
                result["matches_synchronous_baseline"] = True
            report["cases"][mode] = result
            write_report(args.output, report)
            print(f"ADAPTER_PROBE_PASS {mode}", flush=True)
        baseline = report["cases"]["synchronous"]["outputs"]
        report["successive_token_vectors_differ"] = baseline[0] != baseline[1]
        assert report[
            "successive_token_vectors_differ"
        ], "Fixture produced identical successive token vectors; deferred-copy token isolation is inconclusive"
        report["status"] = "passed"
        report["advertised_supports_async_decode"] = adapter.model_capabilities["supports_async_decode"]
        report["probe_exercises_vllm_scheduler"] = False
        report["async_scope"] = "Adapter deferred-copy correctness; real plugin overlap is evidenced by serving runs"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        write_report(args.output, report)
        raise
    finally:
        cleanup_errors = []
        if adapter is not None:
            try:
                adapter.teardown()
            except Exception as exc:
                cleanup_errors.append(f"adapter.teardown: {type(exc).__name__}: {exc}")
        if mesh is not None:
            try:
                close_ornith_mesh(mesh)
            except Exception as exc:
                cleanup_errors.append(f"close_ornith_mesh: {type(exc).__name__}: {exc}")
        report["cleanup_completed"] = not cleanup_errors
        if cleanup_errors:
            report.update(status="failed", cleanup_errors=cleanup_errors)
        write_report(args.output, report)
        print(json.dumps(report, indent=2), flush=True)
        if cleanup_errors:
            raise RuntimeError("; ".join(cleanup_errors))


if __name__ == "__main__":
    main()
