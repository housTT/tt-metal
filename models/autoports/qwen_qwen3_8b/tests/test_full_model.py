# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full-model tests for Qwen/Qwen3-8B on one Blackhole chip (1x1 mesh).

The module builds one generator per process from the real checkpoint. ``QWEN3_FULL_MODEL_LAYERS``
selects the layer count (default 2: the reduced probe with one real layer of the only layer kind
plus one more so the inter-layer sharded residual contract is exercised; 36 is the full model).
Evidence JSON lands under ``doc/full_model``; ``QWEN3_EVIDENCE_SUFFIX`` keeps gate runs apart.

Tests:

* ``test_split_sampling_trace``: token feedback through ``tt_out_tok``, on-device position advance,
  unchanged and changed page tables, greedy and sampled parameters on the same traces.
* ``test_multi_row_prefill_equals_solo``: one ``prefill_forward`` call with rows of different lengths
  equals the single-row results; logits are deterministic across runs and batch positions.
* ``test_batch_decode``: batched token-out decode agrees with single-user decode for 8 and 32 users.
* ``test_host_work_counters``: a 64-token generation replays traces with no per-token host refresh.
* ``test_tracked_replay_workload``: the representative replay ordering for the allocation tracker.
* ``test_perf_batch1``: 128-token prompt, 128 generated tokens, TTFT and decode t/s/u (full model).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_8b.tt import generator as generator_module
from models.autoports.qwen_qwen3_8b.tt import model as model_module
from models.autoports.qwen_qwen3_8b.tt.generator import build_generator, greedy_sampling_params
from models.common.sampling import SamplingParams

MODEL_DIR = Path("models/autoports/qwen_qwen3_8b")
ARTIFACT_DIR = MODEL_DIR / "doc" / "full_model"
LAYERS = int(os.getenv("QWEN3_FULL_MODEL_LAYERS", "2"))
FULL = LAYERS == 36
TRACE_REGION = int(os.getenv("QWEN3_TRACE_REGION_SIZE", str(512 * 1024 * 1024)))
PCC_BAR = 0.999


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(a.norm() * b.norm())
    return 1.0 if denom == 0 else float(a @ b) / denom


def _provenance(gen, command: str) -> dict:
    files = {
        "model": Path(model_module.__file__),
        "generator": Path(generator_module.__file__),
        "test": Path(__file__),
    }
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[4], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "unavailable"
    return {
        "model_id": generator_module.MODEL_ID,
        "checkpoint_revision": gen.model.checkpoint_revision,
        "num_layers": gen.model.num_layers,
        "policy": gen.model.policy.describe(),
        "terminal_policy": gen.model.terminal_policy.describe(),
        "kv_cache_dtype": str(gen.kv_cache_dtype),
        "allow_force_argmax": gen.allow_force_argmax,
        "decode_buckets": list(gen.decode_buckets),
        "warmup_prefill_lengths": list(gen.warmup_prefill_lengths),
        "trace_region_size": TRACE_REGION,
        "checkout_git_sha": git_sha,
        "visible_devices": os.getenv("TT_VISIBLE_DEVICES"),
        "platform": platform.platform(),
        "exact_command": os.getenv("QWEN3_EVIDENCE_COMMAND", command),
        "environment": {
            k: v for k, v in sorted(os.environ.items()) if k.startswith("QWEN3_") or k.startswith("TT_METAL_")
        },
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **{f"{name}_sha256": hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()},
    }


def _write_json(name: str, payload: dict) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = os.getenv("QWEN3_EVIDENCE_SUFFIX", "")
    if suffix:
        name = name.replace(".json", f".{suffix}.json")
    path = ARTIFACT_DIR / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _chat_ids(tokenizer, text: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
    )
    return tokenizer.encode(rendered, add_special_tokens=False)


def _book_ids(tokenizer, length: int, offset: int = 0) -> list[int]:
    import bz2

    text = bz2.open("models/tt_transformers/tests/tale-of-two-cities.txt.bz2", "rt", encoding="utf-8").read(200000)
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[offset : offset + length]


@pytest.fixture(scope="module")
def generator():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=TRACE_REGION)
    kwargs = {"num_layers": LAYERS}
    if not FULL:
        kwargs.update(decode_buckets=(1, 8, 32), warmup_prefill_lengths=(32, 64, 128, 256), num_blocks=512)
    if os.getenv("QWEN3_ALLOW_FORCE_ARGMAX") == "1":
        kwargs["allow_force_argmax"] = True
    gen = build_generator(str(MODEL_DIR), mesh, **kwargs)
    yield gen
    gen.teardown()
    ttnn.close_mesh_device(mesh)


def _tensor(gen, name: str, bucket: int = 1) -> torch.Tensor:
    state = gen.buckets[bucket]
    return ttnn.to_torch(getattr(state, name)).reshape(-1)


def test_split_sampling_trace(generator):
    gen = generator
    tok = gen.tokenizer
    prompt = _chat_ids(tok, "Name three colours of the rainbow.")
    gen.reset()
    first = gen.generate_prefill_only(prompt)
    state = gen.buckets[1]
    before = gen.counter_snapshot()
    record = {"prompt_len": len(prompt), "first_token": first, "steps": []}
    assert int(_tensor(gen, "tokens")[0]) == first
    assert int(_tensor(gen, "current_pos")[0]) == len(prompt)
    logits_prev = None
    tokens_seen = [first]
    for step in range(4):
        consumed_token = int(_tensor(gen, "tokens")[0])
        consumed_pos = int(_tensor(gen, "current_pos")[0])
        out = gen._replay(state, "device", gen.kv_cache)
        sampled = int(gen._read_tokens(state, 1)[0])
        logits = ttnn.to_torch(out)[0, 0, 0, : gen.model.vocab_size].float()
        after_pos = int(_tensor(gen, "current_pos")[0])
        record["steps"].append(
            {
                "step": step,
                "token_consumed": consumed_token,
                "position_consumed": consumed_pos,
                "token_sampled_on_device": sampled,
                "position_after_replay": after_pos,
                "logits_argmax": int(logits.argmax()),
                "logits_changed": None if logits_prev is None else bool(not torch.equal(logits, logits_prev)),
            }
        )
        assert (
            consumed_token == tokens_seen[-1]
        ), "the token the trace consumed is not the one sampled by the previous replay"
        assert after_pos == consumed_pos + 1, "current_pos did not advance on device"
        assert sampled == int(logits.argmax()), "sampled token is not the greedy token of the sampler-ready logits"
        if logits_prev is not None:
            assert not torch.equal(logits, logits_prev)
        logits_prev = logits
        tokens_seen.append(sampled)
    after = gen.counter_snapshot()
    delta = {k: after[k] - before[k] for k in after}
    record["host_work_during_4_replays"] = delta
    assert delta["token_refreshes"] == 0 and delta["position_refreshes"] == 0 and delta["page_table_refreshes"] == 0
    assert delta["decode_trace_replays"] == 4 and delta["sampling_trace_replays"] == 4
    assert delta["decode_trace_captures"] <= 1 and delta["sampling_trace_captures"] <= 1

    table_before = state.host_table.clone()
    gen._write_page_table(state, state.host_table)
    assert gen.counter_snapshot()["page_table_refreshes"] == after["page_table_refreshes"]
    gen._ensure_blocks(
        0, len(state.host_table[0][state.host_table[0] != gen.scratch_block]) * gen.block_size + gen.block_size
    )
    changed = gen._host_page_table([0])
    gen._write_page_table(state, changed)
    assert gen.counter_snapshot()["page_table_refreshes"] == after["page_table_refreshes"] + 1
    assert not torch.equal(changed, table_before[: changed.shape[0]])
    assert torch.equal(ttnn.to_torch(state.page_table)[0], changed[0])
    gen._replay(state, "device", gen.kv_cache)
    sampled = int(gen._read_tokens(state, 1)[0])
    record["after_page_table_change"] = {"token": sampled, "position": int(_tensor(gen, "current_pos")[0])}

    sampled_params = SamplingParams(temperature=[0.7], top_k=[20], top_p=[0.95])
    captures = (gen.counter_snapshot()["decode_trace_captures"], gen.counter_snapshot()["sampling_trace_captures"])
    traced_before = gen.counter_snapshot()["sampling_trace_replays"]
    pos = torch.tensor([int(_tensor(gen, "current_pos")[0])])
    tok_a = gen.decode_forward(
        torch.tensor([[sampled]]), pos, page_table=changed, kv_cache=gen.kv_cache, sampling_params=sampled_params
    )
    tok_b = gen.decode_forward(
        torch.tensor([[int(tok_a[0])]]),
        pos + 1,
        page_table=changed,
        kv_cache=gen.kv_cache,
        sampling_params=greedy_sampling_params(1),
    )
    tok_c = gen.decode_forward(
        torch.tensor([[int(tok_b[0])]]),
        pos + 2,
        page_table=changed,
        kv_cache=gen.kv_cache,
        sampling_params=sampled_params,
    )
    record["mode_alternation"] = {
        "sampled_then_greedy_then_sampled": [int(tok_a[0]), int(tok_b[0]), int(tok_c[0])],
        "decode_trace_captures_added": gen.counter_snapshot()["decode_trace_captures"] - captures[0],
        "sampling_trace_captures_added": gen.counter_snapshot()["sampling_trace_captures"] - captures[1],
        "sampling_trace_replays_added": gen.counter_snapshot()["sampling_trace_replays"] - traced_before,
        "seeded_eager_sampling_steps_added": gen.counter_snapshot()["seeded_eager_sampling_steps"],
        "sampling_trace_slots": len(gen.sampling._trace_states),
    }
    assert record["mode_alternation"]["sampling_trace_replays_added"] == 3
    assert record["mode_alternation"]["seeded_eager_sampling_steps_added"] == 0
    assert record["mode_alternation"]["decode_trace_captures_added"] == 0
    record["counters"] = gen.counter_snapshot()
    record["provenance"] = _provenance(
        gen,
        "python -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_full_model.py -k test_split_sampling_trace",
    )
    _write_json(
        "split_sampling_trace_force_argmax.json" if gen.allow_force_argmax else "split_sampling_trace.json", record
    )


def test_multi_row_prefill_equals_solo(generator):
    gen = generator
    tok = gen.tokenizer
    lengths = [5, 31, 33, 127, 129, 200] if not FULL else [5, 31, 33, 127, 129, 257, 640, 1000]
    prompts = [_book_ids(tok, n, offset=137 * i) for i, n in enumerate(lengths)]
    gen.reset()
    solo = []
    for i, ids in enumerate(prompts):
        gen.reset()
        gen._ensure_blocks(0, len(ids))
        table = gen._host_page_table([0])
        out = gen.prefill_forward(torch.tensor([ids]), page_table=table, kv_cache=gen.kv_cache, prompt_lens=[len(ids)])
        solo.append(out[0, 0])
    gen.reset()
    rows = len(prompts)
    width = max(lengths)
    tokens = torch.zeros(rows, width, dtype=torch.int64)
    for i, ids in enumerate(prompts):
        tokens[i, : len(ids)] = torch.tensor(ids)
        gen._ensure_blocks(i, len(ids))
    table = gen._host_page_table(list(range(rows)))
    batched = gen.prefill_forward(tokens, page_table=table, kv_cache=gen.kv_cache, prompt_lens=lengths)
    rerun = gen.prefill_forward(tokens, page_table=table, kv_cache=gen.kv_cache, prompt_lens=lengths)
    order = list(reversed(range(rows)))
    reordered = gen.prefill_forward(
        tokens[order], page_table=table[order], kv_cache=gen.kv_cache, prompt_lens=[lengths[i] for i in order]
    )
    records = []
    for i in range(rows):
        records.append(
            {
                "length": lengths[i],
                "pcc_batched_vs_solo": _pcc(solo[i], batched[i, 0]),
                "top1_equal": int(solo[i].argmax()) == int(batched[i, 0].argmax()),
                "rerun_bit_equal": bool(torch.equal(batched[i, 0], rerun[i, 0])),
                "reordered_row_bit_equal": bool(torch.equal(batched[i, 0], reordered[order.index(i), 0])),
            }
        )
    payload = {
        "rows": rows,
        "lengths": lengths,
        "cases": records,
        "provenance": _provenance(
            gen,
            "python -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_full_model.py -k test_multi_row_prefill_equals_solo",
        ),
    }
    _write_json("multi_row_prefill.json", payload)
    for record in records:
        assert record["pcc_batched_vs_solo"] >= PCC_BAR, record
        assert record["top1_equal"], record
        assert record["rerun_bit_equal"], record
        assert record["reordered_row_bit_equal"], record


@pytest.mark.parametrize("batch", [8, 16, 24, 32])
def test_batch_decode(generator, batch):
    gen = generator
    tok = gen.tokenizer
    steps = 32
    lengths = [[5, 33, 127, 129, 31, 64, 200, 17][i % 8] for i in range(batch)]
    prompts = [_book_ids(tok, n, offset=53 * i) for i, n in enumerate(lengths)]
    solo = []
    for ids in prompts:
        gen.reset()
        gen._ensure_blocks(0, len(ids) + steps)
        table = gen._host_page_table([0])
        first = gen.prefill_forward(
            torch.tensor([ids]),
            page_table=table,
            kv_cache=gen.kv_cache,
            prompt_lens=[len(ids)],
            sampling_params=greedy_sampling_params(1),
        )
        run = [int(first[0])]
        rows = torch.zeros(batch, 1, dtype=torch.int64)
        pos = torch.full((batch,), -1, dtype=torch.int32)
        wide_table = torch.full((batch, gen.page_table_width), gen.scratch_block, dtype=torch.int32)
        wide_table[0] = table[0]
        for step in range(1, steps):
            rows[0, 0] = run[-1]
            pos[0] = len(ids) + step - 1
            out = gen.decode_forward(
                rows, pos, page_table=wide_table, kv_cache=gen.kv_cache, sampling_params=greedy_sampling_params(batch)
            )
            run.append(int(out[0]))
        solo.append(run)
    gen.reset()
    width = max(lengths)
    tokens = torch.zeros(batch, width, dtype=torch.int64)
    for i, ids in enumerate(prompts):
        tokens[i, : len(ids)] = torch.tensor(ids)
        gen._ensure_blocks(i, len(ids) + steps)
    table = gen._host_page_table(list(range(batch)))
    first = gen.prefill_forward(
        tokens,
        page_table=table,
        kv_cache=gen.kv_cache,
        prompt_lens=lengths,
        sampling_params=greedy_sampling_params(batch),
    )
    generated = [[int(first[i])] for i in range(batch)]
    positions = torch.tensor(lengths, dtype=torch.int32)
    before = gen.counter_snapshot()
    for step in range(1, steps):
        next_tokens = torch.tensor([[generated[i][-1]] for i in range(batch)])
        out = gen.decode_forward(
            next_tokens,
            positions,
            page_table=table,
            kv_cache=gen.kv_cache,
            sampling_params=greedy_sampling_params(batch),
        )
        for i in range(batch):
            generated[i].append(int(out[i]))
        positions = positions + 1
    delta = {k: gen.counter_snapshot()[k] - before[k] for k in before}
    matches = [sum(a == b for a, b in zip(solo[i], generated[i])) for i in range(batch)]
    first_divergence = [
        next((k for k, (a, b) in enumerate(zip(solo[i], generated[i])) if a != b), None) for i in range(batch)
    ]
    payload = {
        "solo_arm": f"one live row decoded in physical batch {batch} with {batch - 1} inactive rows at position -1",
        "first_divergence_per_user": first_divergence,
        "batch": batch,
        "steps": steps,
        "lengths": lengths,
        "solo": solo,
        "batched": generated,
        "matching_tokens_per_user": matches,
        "host_work_during_batched_decode": delta,
        "provenance": _provenance(
            gen,
            f"python -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_full_model.py -k 'test_batch_decode and {batch}'",
        ),
    }
    _write_json(f"batch_decode_{batch}.json", payload)
    assert delta["token_refreshes"] == 1 and delta["position_refreshes"] == 1
    assert delta["decode_trace_replays"] == steps - 1
    for i in range(batch):
        assert first_divergence[i] is None, (i, solo[i], generated[i])


def test_host_work_counters(generator):
    gen = generator
    tok = gen.tokenizer
    prompt = _chat_ids(tok, "Write one sentence about the sea.")
    gen.reset()
    before = gen.counter_snapshot()
    started = time.perf_counter()
    out = gen.generate(prompt, 64, next_input=None, enable_trace=True, stop_on_eos=False)
    elapsed = time.perf_counter() - started
    delta = {k: gen.counter_snapshot()[k] - before[k] for k in before}
    gen.reset()
    repeat = gen.generate(prompt, 64, next_input=None, enable_trace=True, stop_on_eos=False)
    payload = {
        "prompt_len": len(prompt),
        "generated": out,
        "text": tok.decode(out),
        "repeat_equal": out == repeat,
        "elapsed_s": elapsed,
        "host_work": delta,
        "provenance": _provenance(
            gen,
            "python -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_full_model.py -k test_host_work_counters",
        ),
    }
    _write_json("host_work_counters.json", payload)
    assert len(out) == 64
    assert delta["decode_trace_replays"] == 63 and delta["sampling_trace_replays"] == 63
    assert delta["seeded_eager_sampling_steps"] == 0
    assert delta["token_refreshes"] == 0
    assert delta["position_refreshes"] == 1
    assert delta["page_table_refreshes"] <= 1
    assert delta["eager_decode_steps"] == 0
    assert out == repeat


def test_tracked_replay_workload(generator):
    """Representative replay ordering: run it with TT_METAL_TRACE_ALLOC_TRACKING=1 for the allocation gate."""
    gen = generator
    tok = gen.tokenizer
    prompts = [_chat_ids(tok, "Count from one to five."), _book_ids(tok, 200, offset=900), _chat_ids(tok, "Say hello.")]
    log = []
    for ids in prompts:
        gen.reset()
        out = gen.generate(ids, 8, next_input=None, enable_trace=True, stop_on_eos=False)
        log.append({"kind": "greedy_generate", "prompt_len": len(ids), "tokens": out})
    gen.reset()
    sampled = gen.generate(
        prompts[0],
        8,
        next_input=None,
        enable_trace=True,
        stop_on_eos=False,
        sampling_params=SamplingParams(temperature=[0.8], top_k=[10], top_p=[0.9]),
    )
    log.append({"kind": "sampled_generate", "tokens": sampled})
    gen.reset()
    gen._ensure_blocks(0, len(prompts[0]) + 4)
    table = gen._host_page_table([0])
    last = gen.prefill_forward(
        torch.tensor([prompts[0]]), page_table=table, kv_cache=gen.kv_cache, prompt_lens=[len(prompts[0])]
    )
    host_logits = gen.decode_forward(
        torch.tensor([[int(last[0, 0].argmax())]]),
        torch.tensor([len(prompts[0])]),
        page_table=table,
        kv_cache=gen.kv_cache,
    )
    log.append({"kind": "host_logits_decode", "argmax": int(host_logits[0].argmax())})
    batch = 8
    gen.reset()
    lengths = [len(prompts[0])] * batch
    tokens = torch.tensor([prompts[0]] * batch)
    for i in range(batch):
        gen._ensure_blocks(i, lengths[i] + 4)
    table = gen._host_page_table(list(range(batch)))
    first = gen.prefill_forward(
        tokens,
        page_table=table,
        kv_cache=gen.kv_cache,
        prompt_lens=lengths,
        sampling_params=greedy_sampling_params(batch),
    )
    positions = torch.tensor(lengths, dtype=torch.int32)
    for _ in range(3):
        first = gen.decode_forward(
            first.reshape(batch, 1),
            positions,
            page_table=table,
            kv_cache=gen.kv_cache,
            sampling_params=greedy_sampling_params(batch),
        )
        positions = positions + 1
    log.append({"kind": "batch8_decode", "tokens": first.tolist()})
    entries_before = gen.mesh_device.num_program_cache_entries()
    for rows_n in (3, 32):
        gen.reset()
        lens = [len(prompts[0]) + (i % 3) for i in range(rows_n)]
        toks = torch.zeros(rows_n, max(lens), dtype=torch.int64)
        for i, n in enumerate(lens):
            toks[i, :n] = torch.tensor((prompts[0] + prompts[2])[:n])
            gen._ensure_blocks(i, n)
        tab = gen._host_page_table(list(range(rows_n)))
        out = gen.prefill_forward(
            toks,
            page_table=tab,
            kv_cache=gen.kv_cache,
            prompt_lens=lens,
            sampling_params=greedy_sampling_params(rows_n),
        )
        log.append({"kind": f"prefill_{rows_n}_rows_after_capture", "tokens": out.tolist()})
    if FULL:
        gen.reset()
        long_prompt = _book_ids(tok, 8192, offset=3000)
        out = gen.generate(long_prompt, 4, next_input=None, enable_trace=True, stop_on_eos=False)
        log.append({"kind": "generate_8192_after_capture", "tokens": out})
    entries_after = gen.mesh_device.num_program_cache_entries()
    log.append({"kind": "program_cache_entries", "before": entries_before, "after": entries_after})
    assert entries_after == entries_before, (entries_before, entries_after)
    gen.reset()
    again = gen.generate(prompts[0], 8, next_input=None, enable_trace=True, stop_on_eos=False)
    log.append({"kind": "greedy_generate_again", "tokens": again, "equal_to_first": again == log[0]["tokens"]})
    payload = {
        "tracking_env": {
            k: os.environ.get(k)
            for k in (
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "TT_METAL_TRACE_ALLOC_TRACEBACKS",
            )
        },
        "workload": log,
        "counters": gen.counter_snapshot(),
        "traces": {
            str(k): {"decode_traces": {m: str(t) for m, t in v["decode_traces"].items()}, "prepared": v["prepared"]}
            for k, v in gen.trace_summary().items()
        },
        "sampling_trace_slots": len(gen.sampling._trace_states),
        "program_cache_entries": gen.mesh_device.num_program_cache_entries(),
        "provenance": _provenance(
            gen,
            "TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_full_model.py -k test_tracked_replay_workload",
        ),
    }
    _write_json("tracked_replay_workload.json", payload)
    assert again == log[0]["tokens"]


def test_perf_batch1(generator):
    gen = generator
    tok = gen.tokenizer
    prompt = _book_ids(tok, 128, offset=2000)
    assert len(prompt) == 128
    gen.reset()
    gen.generate(prompt, 8, next_input=None, enable_trace=True, stop_on_eos=False)
    runs = []
    for _ in range(3):
        gen.reset()
        stamps = []

        def next_input(step, token):
            stamps.append(time.perf_counter())
            return token

        started = time.perf_counter()
        out = gen.generate(prompt, 128, next_input=next_input, enable_trace=True, stop_on_eos=False)
        ttft = stamps[0] - started
        decode = (stamps[-1] - stamps[0]) / (len(stamps) - 1)
        runs.append(
            {
                "ttft_ms": ttft * 1000,
                "decode_ms_per_token": decode * 1000,
                "decode_t_s_u": 1 / decode,
                "tokens": len(out),
            }
        )
    teacher = []
    for _ in range(2):
        gen.reset()
        stamps = []
        forced = _book_ids(tok, 128, offset=5000)

        def next_input_tf(step, token):
            stamps.append(time.perf_counter())
            return forced[step]

        started = time.perf_counter()
        gen.generate(prompt, 128, next_input=next_input_tf, enable_trace=True)
        teacher.append(
            {
                "ttft_ms": (stamps[0] - started) * 1000,
                "decode_ms_per_token": (stamps[-1] - stamps[0]) / (len(stamps) - 1) * 1000,
                "token_refreshes": gen.counter_snapshot()["token_refreshes"],
            }
        )
    payload = {
        "workload": {
            "prompt_tokens": 128,
            "generated_tokens": 128,
            "batch": 1,
            "sampling": "greedy split sampling (top-k path, k=1)",
        },
        "token_out_runs": runs,
        "teacher_forced_runs": teacher,
        "best_token_out": min(runs, key=lambda r: r["decode_ms_per_token"]),
        "counters": gen.counter_snapshot(),
        "provenance": _provenance(
            gen, "python -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_full_model.py -k test_perf_batch1"
        ),
    }
    payload["sampler_traced"] = gen.counter_snapshot()["seeded_eager_sampling_steps"] == 0
    suffix = "_force_argmax" if gen.allow_force_argmax else ""
    _write_json(f"perf_batch1_128_128{suffix}.json", payload)
    assert payload["sampler_traced"]


def test_seeded_sampling_last(generator):
    """Explicit request seeds run the sampler eagerly by the common sampler's contract and must not leak into
    later unseeded requests; this test runs last in the module so the earlier evidence stays on the traced sampler."""
    gen = generator
    tok = gen.tokenizer
    prompt = _chat_ids(tok, "Pick a random animal.")
    seeded = SamplingParams(temperature=[0.9], top_k=[20], top_p=[0.95], seed=[7])
    before = gen.counter_snapshot()
    gen.reset()
    first = gen.generate(prompt, 12, next_input=None, enable_trace=True, stop_on_eos=False, sampling_params=seeded)
    gen.reset()
    second = gen.generate(prompt, 12, next_input=None, enable_trace=True, stop_on_eos=False, sampling_params=seeded)
    seeded_delta = {k: gen.counter_snapshot()[k] - before[k] for k in before}
    mid = gen.counter_snapshot()
    gen.reset()
    greedy = gen.generate(prompt, 12, next_input=None, enable_trace=True, stop_on_eos=False)
    greedy_delta = {k: gen.counter_snapshot()[k] - mid[k] for k in mid}
    payload = {
        "seeded_runs": [first, second],
        "seeded_runs_equal": first == second,
        "seeded_distinct_tokens": len(set(first)),
        "host_work_seeded": seeded_delta,
        "greedy_after_seeded": greedy,
        "host_work_greedy_after_seeded": greedy_delta,
        "provenance": _provenance(
            gen,
            "python -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_full_model.py -k test_seeded_sampling_last",
        ),
    }
    _write_json("seeded_sampling.json", payload)
    assert first == second
    assert seeded_delta["seeded_eager_sampling_steps"] == 22 and seeded_delta["sampling_trace_replays"] == 0
    assert greedy_delta["sampling_trace_replays"] == 11 and greedy_delta["seeded_eager_sampling_steps"] == 0
