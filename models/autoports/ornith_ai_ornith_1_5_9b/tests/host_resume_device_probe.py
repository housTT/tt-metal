# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real adapter host-step/resume versus uninterrupted device; supervisor only."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.common.sampling import SamplingParams
from models.common.sampling.generator import _hash_request_seed_to_device_seed

from ..reference.hf_reference import HF_REVISION
from ..tt.functional_decoder import num_blocks_for_context
from ..tt.generator_vllm import TTOrnithForCausalLM
from ..tt.model import close_ornith_mesh, open_ornith_mesh
from .adapter_serving_device_probe import delta, replicated_vector, state, tokens_only, write_report


def packed(rows):
    result = torch.full((len(rows), max(1, max(map(len, rows)))), -1, dtype=torch.int64)
    for row, values in enumerate(rows):
        result[row, : len(values)] = torch.as_tensor(values)
    return result


def host_sample(logits, prompts, histories):
    """Confirmed pinned host formula on absolute model logits, greedy only."""
    values = logits[:, 0, :].clone()
    prompt_mask = torch.zeros_like(values, dtype=torch.bool)
    output_counts = torch.zeros_like(values)
    for row, (prompt, history) in enumerate(zip(prompts, histories)):
        prompt_mask[row, torch.as_tensor(prompt)] = True
        output_counts[row].scatter_add_(0, torch.tensor(history), torch.ones(len(history)))
    factor = torch.where(prompt_mask | (output_counts > 0), 2.0, 1.0)
    values = torch.where(values > 0, values / factor, values * factor)
    values -= 0.5 * output_counts
    values -= 2.0 * (output_counts > 0)
    return values.argmax(-1)


def history_counts(adapter):
    tensor = adapter.generator.sampling.tt_penalties.output_counts
    return torch.cat(
        [ttnn.to_torch(shard)[: adapter.max_batch_size].clone() for shard in ttnn.get_device_tensors(tensor)], dim=-1
    )[:, : adapter.model.vocab_size]


def verify_histories(adapter, prompts, histories):
    batch, vocab = adapter.max_batch_size, adapter.model.vocab_size
    expected = torch.zeros(batch, vocab, dtype=torch.int32)
    expected_prompt = torch.zeros(batch, vocab, dtype=torch.int32)
    for row, (prompt, history) in enumerate(zip(prompts, histories)):
        expected[row].scatter_add_(0, torch.tensor(history), torch.ones(len(history), dtype=torch.int32))
        expected_prompt[row, torch.as_tensor(prompt)] = 1
    for name, wanted in (("output_counts", expected), ("output_mask", expected > 0), ("prompt_mask", expected_prompt)):
        tensor = getattr(adapter.generator.sampling.tt_penalties, name)
        observed = torch.cat(
            [ttnn.to_torch(shard)[:batch].clone() for shard in ttnn.get_device_tensors(tensor)], dim=-1
        )
        torch.testing.assert_close(observed[:, :vocab], wanted, rtol=0, atol=0, check_dtype=False)
    # Gathered counts feed the next scatter update; verify every replicated copy.
    for shard in ttnn.get_device_tensors(adapter.generator.sampling.tt_penalties.output_counts_gathered):
        observed = ttnn.to_torch(shard)[:batch, :vocab]
        torch.testing.assert_close(observed, expected, rtol=0, atol=0, check_dtype=False)


def run(adapter, prompts, lengths, table, sampling, *, host_detour):
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
    histories = [[int(value)] for value in first]

    def step(position, *, device):
        return adapter.decode_forward(
            tokens=torch.tensor([row[-1] for row in histories])[:, None],
            start_pos=position,
            page_table=table,
            kv_cache=gen.kv_cache,
            sampling_params=sampling if device else None,
            prompt_tokens=packed(prompts),
            output_tokens=packed(histories),
            read_from_device=True,
        )

    second = step(lengths, device=True)
    for row, value in zip(histories, second):
        row.append(int(value))
    verify_histories(adapter, prompts, histories)
    initial = state(adapter)
    before_gen, before_adapter = dict(gen.counters), dict(adapter.counters)
    key = adapter._sampling_key
    outputs = []
    for index in range(3):
        device = not (host_detour and index == 0)
        prior_counts = history_counts(adapter) if not device else None
        out = step(lengths + 1 + index, device=device)
        if not device:
            torch.testing.assert_close(history_counts(adapter), prior_counts, rtol=0, atol=0)
            out = host_sample(out, prompts, histories)
        for row, value in zip(histories, out):
            row.append(int(value))
        outputs.append(out.tolist())
        if device:
            verify_histories(adapter, prompts, histories)
        if host_detour and index == 1:
            assert adapter._sampling_key == key, "Resume must exercise an unchanged sampling key"
            assert adapter.counters["sampling_updates"] - before_adapter["sampling_updates"] == 1
    final = state(adapter)
    assert final["addresses"] == initial["addresses"]
    torch.testing.assert_close(final["positions"], lengths + 4, rtol=0, atol=0)
    torch.testing.assert_close(final["rope"], lengths + 4, rtol=0, atol=0)
    gen_delta, adapter_delta = delta(before_gen, gen.counters), delta(before_adapter, adapter.counters)
    assert gen_delta["model_replays"] == 3
    assert gen_delta["sampling_replays"] == (2 if host_detour else 3)
    assert adapter_delta["sampling_updates"] == int(host_detour)
    return {
        "first": first.tolist(),
        "second": second.tolist(),
        "continuation": outputs,
        "generated_histories": histories,
        "exact_prompt_output_histories_all_shards": True,
        "host_step_leaves_device_counts_unchanged": host_detour,
        "sampling_key_unchanged": adapter._sampling_key == key,
        "persistent_input_addresses_unchanged": True,
        "positions": final["positions"].tolist(),
        "generator_counters": gen_delta,
        "adapter_counters": adapter_delta,
    }


def seed_admission(adapter, prompts, lengths, table, *, remap):
    """Host-first request gets its own seed once; other rows keep their streams."""
    gen = adapter.generator
    gen.reset(clear_kv=True)
    adapter._device_rows[:] = False
    adapter._prefilled_rows[:] = False
    adapter._pending_device_seeds[:] = False
    adapter._last_device_sampling, adapter._sampling_key = None, None
    prior = SamplingParams(temperature=0.0, top_k=1, top_p=1.0, seed=[11, 11, 11])
    tokens = tokens_only(
        adapter.prefill_forward(
            tokens=prompts,
            prompt_lens=lengths.tolist(),
            empty_slots=[0, 1, 2],
            page_table=table,
            kv_cache=gen.kv_cache,
            sampling_params=prior,
        )
    )
    # Reuse only row 0 for a fresh request initially sampled on host.
    host_logits = adapter.prefill_forward(
        tokens=[prompts[0]],
        prompt_lens=[int(lengths[0])],
        empty_slots=[0],
        page_table=table[[0]],
        kv_cache=gen.kv_cache,
        sampling_params=None,
    )
    if isinstance(host_logits, tuple):
        host_logits = host_logits[0]
    tokens[0] = host_logits[0, 0].argmax()
    before = replicated_vector(gen.sampling.tt_sampling.seeds_tt_tensor, 32)
    assert int(before[0]) == _hash_request_seed_to_device_seed(11, 0)
    assert adapter._pending_device_seeds.tolist() == [True, False, False]
    permutation = [1, 2, 0] if remap else [0, 1, 2]
    order = permutation + list(range(3, 32))
    target = permutation.index(0)
    requested = [99 if old == 0 else 11 for old in permutation]
    params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0, seed=requested)
    before_updates = adapter.counters["sampling_updates"]
    output = adapter.decode_forward(
        tokens=tokens[permutation, None],
        start_pos=lengths[permutation],
        page_table=table[permutation],
        kv_cache=gen.kv_cache,
        sampling_params=params,
        slot_remap=permutation if remap else None,
        read_from_device=True,
    )
    after = replicated_vector(gen.sampling.tt_sampling.seeds_tt_tensor, 32)
    expected = before[order] + 1
    expected[target] = _hash_request_seed_to_device_seed(99, 0) + 1
    torch.testing.assert_close(after, expected, rtol=0, atol=0)
    assert gen.sampling.seed_manager.seeds[target] == 99
    assert not adapter._pending_device_seeds.any()
    assert adapter.counters["sampling_updates"] == before_updates + 1
    adapter.decode_forward(
        tokens=output[:, None],
        start_pos=lengths[permutation] + 1,
        page_table=table[permutation],
        kv_cache=gen.kv_cache,
        sampling_params=params,
        read_from_device=True,
    )
    second = replicated_vector(gen.sampling.tt_sampling.seeds_tt_tensor, 32)
    torch.testing.assert_close(second, after + 1, rtol=0, atol=0)
    assert adapter.counters["sampling_updates"] == before_updates + 1
    return {
        "slot_remap": permutation,
        "new_request_slot": target,
        "old_request_seed": 11,
        "new_request_seed": 99,
        "before": before.tolist(),
        "after_first_device_sample": after.tolist(),
        "after_steady_device_sample": second.tolist(),
        "exact_seed_all_replicas": True,
        "other_rows_advance_without_reset": True,
        "new_row_initialized_once": True,
        "stochastic_host_device_sequence_equivalence_claimed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["ORNITH_MODEL_PATH"] = str(args.model_path.resolve())
    os.environ["ORNITH_VLLM_LAYER_INDICES"] = "0,3"
    os.environ["ORNITH_VLLM_ALLOW_HOST_SAMPLING"] = "1"
    report = {
        "status": "running",
        "scope": "reduced layers0,3 real-weight adapter, greedy host/device history parity; no full-model or performance claim",
        "batch": 3,
        "logical_context": 262144,
        "hf_revision": HF_REVISION,
        "presence": 2.0,
        "frequency": 0.5,
        "repetition": 2.0,
        "source_sha256": {
            name: hashlib.sha256((Path(__file__).parents[1] / "tt" / name).read_bytes()).hexdigest()
            for name in ("generator_vllm.py", "generator.py", "model.py")
        },
    }
    mesh, adapter = None, None
    try:
        config = AutoConfig.from_pretrained(args.model_path, local_files_only=True, revision=HF_REVISION)
        mesh = open_ornith_mesh()
        adapter = TTOrnithForCausalLM.initialize_vllm_model(config, mesh, max_batch_size=3, max_seq_len=262144)
        width = num_blocks_for_context(adapter.max_model_len, adapter.model.page_block_size)
        request_blocks = num_blocks_for_context(256, adapter.model.page_block_size)
        blocks = width + 3 * request_blocks
        heads = max(1, adapter.model.hf_config.num_key_value_heads // mesh.get_num_devices())
        cache = adapter.allocate_kv_cache(
            (blocks, heads, adapter.model.page_block_size, adapter.model.hf_config.head_dim),
            torch.bfloat16,
            len(adapter.model.layers),
        )
        table = torch.zeros(3, width, dtype=torch.int32)
        table[:, :request_blocks] = torch.arange(1, 1 + 3 * request_blocks).reshape(3, request_blocks)
        lengths = torch.arange(61, 64)
        prompts = [torch.arange(17 + 100 * row, 17 + 100 * row + int(length)) for row, length in enumerate(lengths)]
        sampling = SamplingParams(
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            seed=42,
            presence_penalty=2.0,
            frequency_penalty=0.5,
            repetition_penalty=2.0,
        )
        adapter.warmup_model_decode(cache)
        for name, detour in (("unbroken_device", False), ("host_then_device", True)):
            report[name] = run(adapter, prompts, lengths, table, sampling, host_detour=detour)
            write_report(args.output, report)
        for key in ("first", "second", "continuation", "generated_histories", "positions"):
            assert report["unbroken_device"][key] == report["host_then_device"][key], key
        report["host_first_seed"] = seed_admission(adapter, prompts, lengths, table, remap=False)
        write_report(args.output, report)
        report["host_first_seed_remap"] = seed_admission(adapter, prompts, lengths, table, remap=True)
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if adapter is not None:
            adapter.teardown()
        if mesh is not None:
            close_ornith_mesh(mesh)
        write_report(args.output, report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
