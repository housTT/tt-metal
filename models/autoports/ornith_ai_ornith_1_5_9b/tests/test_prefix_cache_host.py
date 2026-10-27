# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exercise actual model prefill control flow with CPU recurrent/KV boundaries."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from models.autoports.ornith_ai_ornith_1_5_9b.tt import model as model_module
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import OrnithGenerator
from models.autoports.ornith_ai_ornith_1_5_9b.tt.prefix_cache import HybridPrefixCache, PrefixSnapshot


class CPUStorage:
    """Independent oracle for snapshot contents; device codec needs silicon tests."""

    def __init__(self):
        self.released = []

    def estimate(self, cache, length):
        return 64

    def capture(self, cache, tokens, table):
        layer = cache.prefill_layers[0]
        states = {0: [tensor.clone() for tensor in [layer.recurrent_state] + layer.conv_state]}
        pages = table[: len(tokens) // 2]
        kv = {1: [tensor[pages].clone() for tensor in cache.kv[1]]}
        return PrefixSnapshot(tokens, states, kv, 64)

    def restore(self, cache, snapshot, table):
        layer = cache.prefill_layers[0]
        for dst, src in zip([layer.recurrent_state] + layer.conv_state, snapshot.states[0]):
            dst.copy_(src)
        pages = table[: len(snapshot.tokens) // 2]
        for dst, src in zip(cache.kv[1], snapshot.kv[1]):
            dst[pages] = src

    def release(self, snapshot):
        self.released.append(snapshot)


def linear(batch):
    layer = SimpleNamespace(
        is_full_attention=False,
        recurrent_state=torch.zeros(batch, 1, 1, dtype=torch.int64),
        conv_state=[torch.zeros(batch, 1, 1, dtype=torch.int64)],
    )

    def reset():
        layer.recurrent_state.zero_()
        for tensor in layer.conv_state:
            tensor.zero_()

    layer.reset_state = reset
    return layer


def fixture(*, batch=1, enabled=True, budget=256):
    model = model_module.OrnithModel.__new__(model_module.OrnithModel)
    model.dim, model.vocab_size, model.page_block_size, model.prefill_chunk = 1, 1000, 2, 4
    model.upload = lambda value, **kwargs: torch.as_tensor(value).clone()
    model.terminal = lambda value: value.clone()
    model.logits_to_host = lambda value, rows: value[0, :rows].clone()
    kv = (torch.zeros(192, 1, 2, 1, dtype=torch.int64), torch.zeros(192, 1, 2, 1, dtype=torch.int64))
    full = SimpleNamespace(is_full_attention=True, k_cache=kv[0], v_cache=kv[1], reset_state=lambda: None)
    decode = linear(batch)
    prefill = decode if batch == 1 else linear(1)
    cache = SimpleNamespace(
        batch_size=batch,
        context=32,
        num_blocks=192,
        kv=[None, kv],
        decode_layers=[decode, full],
        prefill_layers=[prefill, full],
        zero_hidden=torch.zeros(1, 1, 1, dtype=torch.int64),
    )
    model.prefix_cache = HybridPrefixCache(model, budget, 16, storage=CPUStorage()) if enabled else None
    model.computed = []

    def forward(ids, *, start_pos, page_table, kv_cache, **kwargs):
        outputs = []
        state = kv_cache.prefill_layers[0]
        for offset, token in enumerate(ids.flatten().tolist()):
            pos = start_pos + offset
            previous = int(state.conv_state[0].item())
            state.recurrent_state[:] = (state.recurrent_state * 3 + token + previous) % 10007
            state.conv_state[0][:] = token
            block = int(page_table[0, pos // 2])
            kv[0][block, 0, pos % 2, 0] = token
            kv[1][block, 0, pos % 2, 0] = state.recurrent_state.item()
            # Output depends on both recurrent state and every preceding KV value.
            attention = sum(int(kv[1][int(page_table[0, p // 2]), 0, p % 2, 0]) for p in range(pos + 1))
            outputs.append(int(state.recurrent_state.item()) + attention)
            model.computed.append((pos, token))
        return torch.tensor(outputs).reshape(1, -1, 1)

    model._prefill_chunk_forward = forward
    return model, cache


def where(condition, a, b, *, output_tensor):
    output_tensor.copy_(torch.where(condition.bool(), a, b))


FAKE_TT = SimpleNamespace(
    int32=torch.int32,
    uint32=torch.int32,
    ROW_MAJOR_LAYOUT="row_major",
    slice=lambda value, starts, ends: value[tuple(slice(a, b) for a, b in zip(starts, ends))].clone(),
    clone=lambda value: value.clone(),
    deallocate=lambda value: None,
    concat=lambda values, dim: torch.cat(values, dim=dim),
    Shape=lambda values: tuple(values),
    repeat=lambda value, shape: value.repeat(shape),
    where=where,
    copy=lambda src, dst: dst.copy_(src),
)


def run(model, cache, tokens, *, row=0, bank=0, start=0, all_logits=False):
    table = torch.arange(cache.batch_size * 32).reshape(cache.batch_size, 32) + bank
    return (
        model.prefill_forward(
            [tokens],
            page_table=table,
            kv_cache=cache,
            prompt_lens=[len(tokens)],
            slots=[row],
            start_pos=[start],
            return_all_logits=all_logits,
        ),
        table,
    )


class PrefixCacheTests(unittest.TestCase):
    def setUp(self):
        self.patch = patch.object(model_module, "ttnn", FAKE_TT)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def assert_same(self, got, expected):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_hit_restores_recurrence_conv_and_new_physical_pages(self):
        model, cache = fixture()
        prompt = list(range(1, 10))
        first, _ = run(model, cache, prompt)
        snapshot = next(iter(model.prefix_cache.entries.values()))
        saved = copy.deepcopy(snapshot)
        cache.decode_layers[0].recurrent_state.fill_(777)
        cache.decode_layers[0].conv_state[0].fill_(888)
        for tensor in cache.kv[1]:
            tensor.fill_(-123)
        model.computed.clear()
        second, table = run(model, cache, prompt, bank=32)
        self.assert_same(second, first)
        self.assertEqual(model.computed, [(8, 9)])
        self.assertEqual(model.prefix_cache.counters["reused_tokens"], 8)
        for actual, expected in zip(snapshot.buffers(), saved.buffers()):
            self.assert_same(actual, expected)
        for tensor in cache.kv[1]:
            self.assertTrue(bool((tensor[:32] == -123).all()))
        self.assertEqual(cache.kv[0], None)

    def test_divergence_and_longest_available_prefix(self):
        model, cache = fixture()
        run(model, cache, list(range(1, 10)))
        longer = list(range(1, 14))
        result, _ = run(model, cache, longer, bank=16)
        baseline, baseline_cache = fixture(enabled=False)
        expected, _ = run(baseline, baseline_cache, longer)
        self.assert_same(result, expected)
        model.computed.clear()
        result, _ = run(model, cache, longer, bank=32)
        self.assert_same(result, expected)
        self.assertEqual(model.computed, [(12, 13)])
        changed = [99, *longer[1:]]
        model.computed.clear()
        result, _ = run(model, cache, changed, bank=48)
        expected, _ = run(baseline, baseline_cache, changed)
        self.assert_same(result, expected)
        self.assertEqual(len(model.computed), len(changed))

    def test_eviction_then_recompute_is_correct(self):
        model, cache = fixture(budget=64)
        a, b = list(range(1, 10)), list(range(11, 20))
        expected, _ = run(model, cache, a)
        run(model, cache, b, bank=16)
        self.assertEqual(model.prefix_cache.counters["evictions"], 1)
        model.computed.clear()
        actual, _ = run(model, cache, a, bank=32)
        self.assert_same(actual, expected)
        self.assertEqual(len(model.computed), len(a))
        self.assertEqual(model.prefix_cache.size_bytes, 64)
        model.clear_prefix_cache()
        model.clear_prefix_cache()
        self.assertEqual(model.prefix_cache.size_bytes, 0)
        self.assertEqual(len(model.prefix_cache.storage.released), 3)

    def test_budget_rejects_oversized_snapshot_without_changing_outputs(self):
        model, cache = fixture(budget=63)
        expected, _ = run(model, cache, list(range(1, 10)))
        actual, _ = run(model, cache, list(range(1, 10)), bank=16)
        self.assert_same(actual, expected)
        self.assertEqual(model.prefix_cache.counters["oversized"], 2)
        self.assertFalse(model.prefix_cache.entries)

    def test_other_live_slot_and_pages_survive_reuse(self):
        model, cache = fixture(batch=3)
        a, b = list(range(1, 8)), list(range(11, 20))
        run(model, cache, a, row=0)
        state_before = cache.decode_layers[0].recurrent_state[0].clone()
        kv_before = [value[:16].clone() for value in cache.kv[1]]
        first, _ = run(model, cache, b, row=2)
        second, _ = run(model, cache, b, row=1, bank=48)
        self.assert_same(first[0, 2], second[0, 1])
        self.assert_same(cache.decode_layers[0].recurrent_state[0], state_before)
        for tensor, expected in zip(cache.kv[1], kv_before):
            self.assert_same(tensor[:16], expected)

    def test_all_logits_and_continuation_do_not_use_prefix_snapshots(self):
        model, cache = fixture()
        run(model, cache, list(range(1, 10)))
        before = dict(model.prefix_cache.counters)
        run(model, cache, [10, 11], start=9)
        run(model, cache, list(range(1, 10)), all_logits=True)
        self.assertEqual(model.prefix_cache.counters, before)

    def test_short_and_page_aligned_prompts_recompute_terminal_token(self):
        model, cache = fixture()
        for tokens in ([1], [1, 2], [1, 2, 3], list(range(1, 9))):
            self.assertLess(model.prefix_cache.checkpoint_length(len(tokens)), len(tokens))
            first, _ = run(model, cache, tokens)
            second, _ = run(model, cache, tokens, bank=16)
            self.assert_same(first, second)

    def test_prefix_lookup_stays_outside_prefill_trace(self):
        model, _ = fixture()
        gen = OrnithGenerator.__new__(OrnithGenerator)
        gen.use_prefill_trace = True
        gen.model = model
        self.assertFalse(gen._prepare_prefill_trace([128], [0], [0], torch.zeros(1, 16)))

    def test_new_snapshot_forces_recapture_even_without_new_programs(self):
        model, cache = fixture()
        gen = OrnithGenerator.__new__(OrnithGenerator)
        gen.model = model
        gen.mesh_device = SimpleNamespace(num_program_cache_entries=lambda: 42)
        gen._programs = 42
        gen._captured_prefix_generation = model.prefix_cache.generation
        gen._capture = Mock()
        gen._ensure_replay_safe()
        gen._capture.assert_not_called()
        run(model, cache, list(range(1, 10)))
        gen._ensure_replay_safe()
        gen._capture.assert_called_once()
        gen._captured_prefix_generation = model.prefix_cache.generation
        gen._capture.reset_mock()
        run(model, cache, list(range(1, 10)), bank=32)
        gen._ensure_replay_safe()
        gen._capture.assert_not_called()


if __name__ == "__main__":
    unittest.main()
