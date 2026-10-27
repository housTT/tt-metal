# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real-weight adapter async proof; the supervising hardware lane runs this.

This opens a TP4 mesh. Authoring/static checks must not import this module.
It checks reduced layers 0 and 3 with native logical context, a shared physical
pool, and device sampling. It is not a full-model accuracy or performance gate.
The opt-in --active8-public-boundaries case uses all32 layers/B32/4129 blocks
and separately launched original/family_b32/active8 controls.
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import traceback
from dataclasses import replace
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

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
            selected = ttnn.slice(
                tensor,
                [physical_block, 0, 0, 0],
                [physical_block + 1, *list(tensor.shape)[1:]],
            )
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


def public_boundary_schedule():
    """Fixed cases shared by the three separately launched controls."""
    return [
        "prefix_admit",
        "prefix_deferred_1",
        "prefix_deferred_2",
        "sparse_swap",
        "sparse_steady",
        "prefix_swap_back",
        "cancel_without_admission",
        "seven_steady",
        "readmit_with_live_rows",
        "readmit_steady",
        "enable_seeded_penalties",
        "seeded_deferred",
        "host_after_drain",
        "host_continuation",
        "device_penalty_resume",
        "device_continuation",
    ]


def warm_public_state_reads(adapter):
    """Warm the fixed KV readback slice before any trace captures."""
    for pair in adapter.generator.kv_cache.kv:
        for tensor in pair or []:
            selected = ttnn.slice(tensor, [0, 0, 0, 0], [33, *list(tensor.shape)[1:]])
            ttnn.deallocate(selected)


def public_chat_prompts(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    texts = [
        f"Explain why astronomers record the time of an observation. Give a clear short answer for notebook {number}."
        for number in range(9)
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        for text in texts
    ]
    assert all(isinstance(row, list) and 1 < len(row) < 128 for row in prompts)
    return prompts


def public_boundary_probe(adapter, prompts, report, save, reference):
    """Short full-model public-boundary checks; no latency or long-cache claim."""
    gen, cache = adapter.generator, adapter.generator.kv_cache
    mode = os.environ.get("ORNITH_ACTIVE8_MODE", "original")
    assert mode in ("original", "family_b32", "active8")
    assert adapter.max_batch_size == 32 and cache.num_blocks == 4129 and cache.context == 262144
    assert adapter.model.layer_indices == list(range(32))
    assert adapter.mesh_device.get_num_devices() == 4 and adapter.model.page_block_size == 64
    assert adapter.model.vocab_size == 248320
    assert gen._active8_enabled == (mode != "original")
    assert gen._model_trace is not None and gen._sampling_trace is not None
    if gen._active8_enabled:
        assert gen._active8_trace is not None
    table = torch.zeros(32, 4096, dtype=torch.int32)
    table[:8, :4] = torch.arange(1, 33, dtype=torch.int32).reshape(8, 4)
    prompt_history = [list(row) for row in prompts[:8]] + [[] for _ in range(24)]
    output_history = [[] for _ in range(32)]
    request_ids = list(range(8)) + [None] * 24
    current_tokens = [0] * 32
    current_positions = [-1] * 32
    seeds = [111, 111, 222, 333, 444, 555, 666, 777] + [42] * 24
    greedy = SamplingParams(temperature=0.0, top_k=1, top_p=1.0, seed=seeds)
    stochastic = SamplingParams(
        temperature=0.7,
        top_k=32,
        top_p=0.95,
        presence_penalty=0.2,
        frequency_penalty=0.1,
        repetition_penalty=1.05,
        seed=seeds,
    )
    active_sampling = greedy
    result = {
        "mode": mode,
        "schedule": public_boundary_schedule(),
        "steps": [],
        "checkpoints": [],
        "prompt_lengths": list(map(len, prompts)),
        "prompt_sha256": [hashlib.sha256(json.dumps(row).encode()).hexdigest() for row in prompts],
        "cache_scope": "All ranks of every logical decode hybrid state; all ranks of KV blocks0..32 containing null and every request page. Full native pool exactness is the separate full-model component gate.",
        "long_full_model_gate_not_repeated": True,
    }
    report["public_boundary"] = result
    if reference is not None:
        assert reference["public_boundary"]["schedule"] == result["schedule"]
        assert reference["public_boundary"]["prompt_sha256"] == result["prompt_sha256"]

    def tensor_hash(value):
        checksum = hashlib.sha256()
        for row in value.detach():
            checksum.update(memoryview(row.reshape(-1).contiguous().view(torch.uint8).numpy()))
        return checksum.hexdigest()

    def checkpoint(name):
        evidence = {"name": name, "buffers": [], "completed": False}
        result["checkpoints"].append(evidence)
        for index, layer in enumerate(cache.decode_layers):
            values = (
                [("k", layer.k_cache), ("v", layer.v_cache)]
                if layer.is_full_attention
                else [("recurrent", layer.recurrent_state)]
                + [(f"conv{n}", value) for n, value in enumerate(layer.conv_state)]
            )
            for kind, value in values:
                selected = (
                    ttnn.slice(value, [0, 0, 0, 0], [33, *list(value.shape)[1:]]) if layer.is_full_attention else value
                )
                try:
                    device_shards = ttnn.get_device_tensors(selected)
                    assert len(device_shards) == 4
                    for rank, shard in enumerate(device_shards):
                        host = ttnn.to_torch(shard)
                        nonfinite = sum(int((~torch.isfinite(row)).sum()) for row in host)
                        evidence["buffers"].append(
                            {
                                "key": f"layer{index}/{kind}/rank{rank}",
                                "sha256": tensor_hash(host),
                                "nonfinite_count": nonfinite,
                            }
                        )
                        del host
                        assert nonfinite == 0
                finally:
                    if layer.is_full_attention:
                        ttnn.deallocate(selected)
        active = [slot for slot, position in enumerate(current_positions) if position >= 0]
        sampler_buffers = [
            ("seed_counters", gen.sampling.tt_sampling.seeds_tt_tensor),
            ("prompt_mask", gen.sampling.tt_penalties.prompt_mask),
        ]
        sampler_buffers += [
            (f"output_history{number}", value) for number, value in enumerate(gen._sampler_history_tensors())
        ]
        for kind, tensor in sampler_buffers:
            for rank, shard in enumerate(ttnn.get_device_tensors(tensor)):
                host = ttnn.to_torch(shard)
                selected = host.reshape(-1)[active] if kind == "seed_counters" else host[active]
                evidence["buffers"].append(
                    {
                        "key": f"sampler/{kind}/rank{rank}/active",
                        "sha256": tensor_hash(selected),
                        "nonfinite_count": 0,
                    }
                )
                del selected, host
        if reference is not None:
            expected = next(row for row in reference["public_boundary"]["checkpoints"] if row["name"] == name)
            actual_by_key = {row["key"]: row for row in evidence["buffers"]}
            expected_by_key = {row["key"]: row for row in expected["buffers"]}
            evidence["mismatches"] = [
                {
                    "key": key,
                    "actual": actual_by_key.get(key),
                    "expected": expected_by_key.get(key),
                }
                for key in sorted(actual_by_key.keys() | expected_by_key.keys())
                if actual_by_key.get(key) != expected_by_key.get(key)
            ]
            assert not evidence["mismatches"], name
        evidence["completed"] = True
        save()

    def admission(slot, prompt, request):
        first = tokens_only(
            adapter.prefill_forward(
                tokens=[prompt],
                prompt_lens=[len(prompt)],
                empty_slots=[slot],
                page_table=table,
                kv_cache=cache,
                sampling_params=replace(greedy, seed=888),
            )
        )
        current_tokens[slot], current_positions[slot] = int(first[0]), len(prompt)
        prompt_history[slot], output_history[slot], request_ids[slot] = (
            list(prompt),
            [int(first[0])],
            request,
        )

    gen.reset(clear_kv=True)
    adapter._device_rows[:] = False
    adapter._prefilled_rows[:] = False
    adapter._pending_device_seeds[:] = False
    adapter._last_device_sampling = adapter._sampling_key = None
    first = tokens_only(
        adapter.prefill_forward(
            tokens=prompts[:8],
            prompt_lens=list(map(len, prompts[:8])),
            empty_slots=list(range(8)),
            page_table=table,
            kv_cache=cache,
            sampling_params=greedy,
        )
    )
    for slot in range(8):
        current_tokens[slot], current_positions[slot] = int(first[slot]), len(prompts[slot])
        output_history[slot] = [int(first[slot])]
    canonical_address = int(gen._logits.buffer_address())
    input_addresses = [int(value.buffer_address()) for value in gen._inputs]
    scratch_addresses = [int(value.buffer_address()) for value in gen._active8.owners] if gen._active8_enabled else []
    result["persistent_resources"] = {
        "initial_logits": canonical_address,
        "inputs": input_addresses,
        "scratch": scratch_addresses,
    }
    pending = []

    def drain():
        if not pending:
            return
        for item in pending:
            row, token_pending, logit_pending, device, greedy_step = item
            for event in (token_pending[1] if token_pending is not None else []) + logit_pending[1]:
                ttnn.event_synchronize(event)
            logits = gen.logits_from(logit_pending[0][0])
            assert tuple(logits.shape) == (32, adapter.model.vocab_size)
            tokens = (
                adapter.process_decode_output_host(token_pending[0], is_tokens=True).tolist()
                if device
                else logits.argmax(dim=-1).tolist()
            )
            for slot in row["active_slots"]:
                nonfinite = int((~torch.isfinite(logits[slot])).sum())
                row["outputs"].append(
                    {
                        "slot": slot,
                        "request": row["request_ids"][slot],
                        "token": int(tokens[slot]),
                        "full_vocab_sha256": tensor_hash(logits[slot : slot + 1]),
                        "nonfinite_count": nonfinite,
                    }
                )
                assert nonfinite == 0
                if greedy_step:
                    assert int(tokens[slot]) == int(logits[slot].argmax())
                current_tokens[slot] = int(tokens[slot])
                output_history[slot].append(int(tokens[slot]))
            if reference is not None:
                expected = reference["public_boundary"]["steps"][row["index"]]
                if row["outputs"] != expected["outputs"]:
                    row["expected_outputs"] = expected["outputs"]
                assert row["outputs"] == expected["outputs"], row["name"]
            row["completed"] = True
            del logits
        observed = state(adapter)
        result["last_state"] = {
            name: value.tolist() if isinstance(value, torch.Tensor) else value for name, value in observed.items()
        }
        assert observed["addresses"] == input_addresses
        assert observed["positions"].tolist() == current_positions
        latest = pending[-1][0]
        expected_tokens = current_tokens if pending[-1][3] else latest["consumed_host_tokens"]
        for slot in latest["active_slots"]:
            assert int(observed["rope"][slot]) == current_positions[slot]
            assert int(observed["tokens"][slot]) == expected_tokens[slot]
        pending.clear()
        save()

    for index, name in enumerate(public_boundary_schedule()):
        if name not in ("prefix_deferred_2",):
            drain()
        remap = None
        reset = index in (0, 3, 5, 6, 8, 10, 12, 14)
        if index in (3, 5):
            remap = list(range(32))
            remap[7], remap[8] = remap[8], remap[7]
            table = table[remap].clone()
            current_tokens[:] = [current_tokens[old] for old in remap]
            current_positions[:] = [current_positions[old] for old in remap]
            prompt_history[:] = [prompt_history[old] for old in remap]
            output_history[:] = [output_history[old] for old in remap]
            request_ids[:] = [request_ids[old] for old in remap]
        if index == 6:
            current_positions[7], request_ids[7] = -1, None
        if index == 8:
            admission(7, prompts[8], 8)
        if index == 10:
            active_sampling = stochastic
        device = index not in (12, 13)
        step_sampling = replace(
            active_sampling,
            seed=[
                ([111, 111, 222, 333, 444, 555, 666, 777, 888][request] if request is not None else 42)
                for request in request_ids
            ],
        )
        active = [slot for slot, position in enumerate(current_positions) if position >= 0]
        expected_variant = (
            "original"
            if mode == "original"
            else ("active8" if mode == "active8" and active == list(range(8)) else "family_b32")
        )
        row = {
            "index": index,
            "name": name,
            "device_sampling": device,
            "active_slots": active,
            "request_ids": list(request_ids),
            "positions_before": list(current_positions),
            "consumed_host_tokens": list(current_tokens),
            "expected_variant": expected_variant,
            "outputs": [],
            "completed": False,
            "pending_before_boundary": len(pending),
        }
        result["steps"].append(row)
        before_programs, before_trace = int(adapter.mesh_device.num_program_cache_entries()), str(gen._model_trace)
        before_counters = dict(gen.counters)
        true_inputs = not device or index in (0, 8, 14)
        supplied_tokens = (
            current_tokens
            if true_inputs
            else [(token + 100003 + index) % adapter.model.vocab_size for token in current_tokens]
        )
        supplied_positions = (
            current_positions if true_inputs else [2 + slot if slot in active else -1 for slot in range(32)]
        )
        raw = adapter.decode_forward(
            tokens=torch.tensor(supplied_tokens)[:, None],
            start_pos=torch.tensor(supplied_positions),
            page_table=table,
            kv_cache=cache,
            sampling_params=step_sampling if device else None,
            reset_batch=reset,
            slot_remap=remap,
            read_from_device=False,
            prompt_tokens=prompt_history,
            output_tokens=output_history,
        )
        row["actual_variant"] = gen._last_decode_variant
        assert row["actual_variant"] == expected_variant
        row["physical_selector"] = list(gen._active_slots)
        assert row["physical_selector"] == [slot in active for slot in range(32)]
        assert gen.kv_cache is cache and cache.num_blocks == 4129
        if gen._active8_enabled:
            assert int(gen._logits.buffer_address()) == canonical_address
        row["counter_delta"] = delta(before_counters, gen.counters)
        assert row["counter_delta"]["model_replays"] == 1
        if expected_variant != "original":
            assert row["counter_delta"][expected_variant + "_replays"] == 1
        assert row["counter_delta"]["sampling_replays"] == int(device)
        row["traces"] = {
            "model": str(gen._model_trace),
            "active8": str(gen._active8_trace),
            "sampler": str(gen._sampling_trace),
        }
        if gen._active8_enabled:
            assert [int(value.buffer_address()) for value in gen._active8.owners] == scratch_addresses
        row["programs_before"], row["programs_after"] = before_programs, int(
            adapter.mesh_device.num_program_cache_entries()
        )
        if index in (1, 2, 4, 7, 9, 11, 15):
            for counter in ("token_refreshes", "position_refreshes", "rope_refreshes"):
                assert row["counter_delta"][counter] == 0, "Stale steady host inputs overwrote device feedback"
            assert (
                row["programs_after"] == before_programs and str(gen._model_trace) == before_trace
            ), "Steady activity recaptured or grew programs"
        if device:
            assert raw is gen._inputs[0]
            token_pending = adapter.read_decode_output(raw, async_read=True)
            host, event = gen.read_output_async(return_logits=True)
            logit_pending = ([host], [event])
        else:
            assert raw is gen._logits, "Public host-sampling path returned stale logits"
            token_pending = None
            logit_pending = adapter.read_decode_output(raw, async_read=True)
        pending.append(
            (
                row,
                token_pending,
                logit_pending,
                device,
                active_sampling is greedy or not device,
            )
        )
        for slot in active:
            current_positions[slot] += 1
        if index not in (1, 11):
            drain()
        if index in (5, 7, 9, 15):
            checkpoint(name)
    drain()
    before_sync = state(adapter)
    synchronous_logits = gen.decode_forward(
        None,
        None,
        page_table=table,
        kv_cache=cache,
        return_logits=True,
        read_from_device=True,
        sample_on_device=False,
    )
    assert tuple(synchronous_logits.shape) == (32, adapter.model.vocab_size)
    result["public_sync_logits"] = [
        {
            "slot": slot,
            "sha256": tensor_hash(synchronous_logits[slot : slot + 1]),
            "nonfinite_count": int((~torch.isfinite(synchronous_logits[slot])).sum()),
        }
        for slot in range(8)
    ]
    assert all(row["nonfinite_count"] == 0 for row in result["public_sync_logits"])
    if reference is not None:
        result["expected_public_sync_logits"] = reference["public_boundary"]["public_sync_logits"]
        assert result["public_sync_logits"] == result["expected_public_sync_logits"]
    after_sync = state(adapter)
    torch.testing.assert_close(after_sync["positions"][:8], before_sync["positions"][:8] + 1, rtol=0, atol=0)
    torch.testing.assert_close(after_sync["tokens"][:8], before_sync["tokens"][:8], rtol=0, atol=0)
    result["public_sync_logits_feedback_unchanged"] = True
    result["completed"] = True
    result["reference_comparison_exact"] = reference is not None
    report["status"] = "passed_public_boundaries"
    report["probe_exercises_vllm_scheduler"] = False
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, choices=(3, 4), default=3)
    parser.add_argument(
        "--num-devices",
        type=int,
        choices=(1, 2, 4),
        default=4,
        help="Reduced probe mesh size; the public-boundary case stays TP4.",
    )
    parser.add_argument(
        "--active8-public-boundaries",
        action="store_true",
        help="Opt in to full32/B32/native4129 public serving boundaries; default reduced probe is unchanged.",
    )
    parser.add_argument(
        "--reference-result",
        type=Path,
        help="Passing unchanged-original public-boundary JSON required for family_b32/active8 controls.",
    )
    parser.add_argument(
        "--require-trace-allocation-tracking",
        action="store_true",
        help="Require the native pre-replay guard; set TT_METAL_TRACE_ALLOC_TRACKING=1 before process startup.",
    )
    args = parser.parse_args()
    reference = None
    if args.active8_public_boundaries:
        args.batch = 32
        args.require_trace_allocation_tracking = True
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if args.output.exists():
            raise FileExistsError("Choose a fresh output; preserve all previous controls")
        mode = os.environ.get("ORNITH_ACTIVE8_MODE", "original")
        if mode not in ("original", "family_b32", "active8"):
            raise ValueError("Unknown active8 control mode")
        if mode != "original":
            if args.reference_result is None:
                raise ValueError("Candidate controls require the passing original result")
            reference = json.loads(args.reference_result.read_text())
            if not (
                reference.get("status") == "passed_public_boundaries"
                and reference.get("cleanup_completed") is True
                and reference.get("trace_bytes_after_teardown") == 0
                and reference.get("layers") == list(range(32))
                and reference.get("physical_blocks") == 4129
                and reference.get("public_boundary", {}).get("mode") == "original"
            ):
                raise ValueError("Reference is not a completed original full-model public-boundary control")
    os.environ["ORNITH_MODEL_PATH"] = str(args.model_path.resolve())
    if args.active8_public_boundaries:
        os.environ.pop("ORNITH_VLLM_LAYER_INDICES", None)
    else:
        os.environ["ORNITH_VLLM_LAYER_INDICES"] = "0,3"
    os.environ["ORNITH_VLLM_ALLOW_HOST_SAMPLING"] = "1" if args.active8_public_boundaries else "0"
    report = {
        "status": "running",
        "hardware": (
            "four Blackhole chips on two P300c boards, TP4"
            if args.num_devices == 4
            else f"{args.num_devices} Blackhole chip(s), TP{args.num_devices}"
        ),
        "layers": list(range(32)) if args.active8_public_boundaries else [0, 3],
        "batch": args.batch,
        "logical_context": 262144,
        "hf_revision": HF_REVISION,
        "scope": (
            "full-model short public serving boundaries; no performance or scheduler/release claim"
            if args.active8_public_boundaries
            else "reduced real-weight adapter contract; no full-model accuracy or performance claim"
        ),
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
    if args.active8_public_boundaries:
        helper = Path(__file__).parents[1] / "tt" / "active8.py"
        report["source_sha256"]["active8.py"] = hashlib.sha256(helper.read_bytes()).hexdigest()
    mesh, adapter = None, None
    frozen = False
    write_report(args.output, report)
    try:
        if args.active8_public_boundaries:
            expected_root = (Path(__file__).resolve().parents[1] / "tt").resolve()
            package = TTOrnithForCausalLM.__module__.rsplit(".", 1)[0]
            imported = {
                name: str(Path(sys.modules[package + "." + name].__file__).resolve())
                for name in ("generator_vllm", "generator", "model", "active8")
            }
            report["runtime_module_paths"] = imported
            for name, path in imported.items():
                assert Path(path) == expected_root / (name + ".py"), (
                    name,
                    path,
                    "Non-candidate import",
                )
            report["allow_host_sampling"] = os.environ["ORNITH_VLLM_ALLOW_HOST_SAMPLING"]
            write_report(args.output, report)
        if args.require_trace_allocation_tracking:
            if not ttnn.TRACE_ALLOC_TRACKING:
                raise RuntimeError("Set TT_METAL_TRACE_ALLOC_TRACKING=1 before importing TTNN")
            if os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "0":
                raise RuntimeError("Program-cache allocations must remain included in the native tracker")
        hf_config = AutoConfig.from_pretrained(args.model_path, local_files_only=True, revision=HF_REVISION)
        public_prompts = public_chat_prompts(args.model_path) if args.active8_public_boundaries else None
        if args.active8_public_boundaries and args.num_devices != 4:
            raise ValueError("The public-boundary control is defined for the TP4 mesh")
        mesh = open_ornith_mesh(num_devices=args.num_devices)
        adapter = TTOrnithForCausalLM.initialize_vllm_model(
            hf_config, mesh, max_batch_size=args.batch, max_seq_len=262144
        )
        assert adapter.model.layer_indices == report["layers"]
        assert {layer.is_full_attention for layer in adapter.model.layers} == {
            False,
            True,
        }
        width = num_blocks_for_context(adapter.max_model_len, adapter.model.page_block_size)
        request_blocks = num_blocks_for_context(256, adapter.model.page_block_size)
        # paged_fused_update_cache requires physical blocks >= page-table width.
        # This stays below B full native allocations while satisfying that bound.
        physical_blocks = 4129 if args.active8_public_boundaries else width + args.batch * request_blocks
        heads = max(1, adapter.model.hf_config.num_key_value_heads // mesh.get_num_devices())
        cache = adapter.allocate_kv_cache(
            (
                physical_blocks,
                heads,
                adapter.model.page_block_size,
                adapter.model.hf_config.head_dim,
            ),
            torch.bfloat16,
            len(adapter.model.layers),
        )
        assert cache is adapter.generator.kv_cache and not adapter.generator.owns_cache
        assert cache.context == 262144 and width <= physical_blocks < args.batch * width
        assert all(pair is None or pair[0].shape[0] == physical_blocks for pair in cache.kv)
        if args.active8_public_boundaries:
            report.update(
                physical_blocks=physical_blocks,
                native_page_table_width=width,
                precision=adapter.model.precision,
            )
            if reference is not None:
                assert report["precision"] == reference["precision"]
                assert report["source_sha256"] == reference["source_sha256"], "Source changed after original control"
                report["reference_result_sha256"] = hashlib.sha256(args.reference_result.read_bytes()).hexdigest()
            warm_public_state_reads(adapter)
            adapter.warmup_model_prefill(cache)
            adapter.warmup_model_decode(cache)
            from vllm.utils.gc_utils import freeze_gc_heap

            assert gc.isenabled() and gc.get_freeze_count() == 0
            frozen = True
            freeze_gc_heap()
            assert gc.isenabled() and gc.get_freeze_count() > 0
            public_boundary_probe(
                adapter,
                public_prompts,
                report,
                lambda: write_report(args.output, report),
                reference,
            )
            return
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
            result = run_pair(
                adapter,
                prompts,
                lengths,
                table,
                sampling,
                mode=mode,
                spare_block=spare_block,
            )
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
        report.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        write_report(args.output, report)
        raise
    finally:
        cleanup_errors = []
        if frozen:
            try:
                gc.unfreeze()
                gc.collect()
                assert gc.get_freeze_count() == 0
            except Exception as exc:
                cleanup_errors.append(f"gc cleanup: {type(exc).__name__}: {exc}")
        if adapter is not None:
            cleanup_generator = adapter.generator
            try:
                adapter.teardown()
                if args.active8_public_boundaries:
                    view = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
                    report["trace_bytes_after_teardown"] = int(view.num_banks * view.total_bytes_allocated_per_bank)
                    assert report["trace_bytes_after_teardown"] == 0
            except Exception as exc:
                cleanup_errors.append(f"adapter.teardown: {type(exc).__name__}: {exc}")
            if args.active8_public_boundaries:
                report["capture_cleanup_errors"] = getattr(cleanup_generator, "capture_cleanup_errors", [])
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
