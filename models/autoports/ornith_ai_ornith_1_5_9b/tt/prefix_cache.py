# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bounded, model-owned snapshots of hybrid prefix state.

Active paged KV remains caller-owned. Snapshots own independent buffers and
restore packed KV into newly assigned pages; they never retain scheduler page
ownership. This module is an experimental bringup path until device validation.
"""

from collections import OrderedDict
from dataclasses import dataclass
from math import prod

import torch

import ttnn


def _bytes(tensor):
    """Largest rank's padded payload; the cache budget is per device."""
    sizes = []
    for shard in ttnn.get_device_tensors(tensor):
        elements = prod(shard.padded_shape)
        if shard.dtype == ttnn.bfloat8_b:
            sizes.append(elements // 1024 * 1088)
        elif shard.dtype == ttnn.bfloat4_b:
            sizes.append(elements // 1024 * 576)
        else:
            sizes.append(elements * (4 if shard.dtype == ttnn.float32 else 2))
    return max(sizes)


def _release(tensors):
    seen = set()
    unique = []
    for tensor in tensors:
        key = (tensor.memory_config().buffer_type, tensor.buffer_address())
        if key not in seen:
            unique.append(tensor)
            seen.add(key)
    for tensor in unique:
        ttnn.deallocate(tensor)


@dataclass
class PrefixSnapshot:
    tokens: tuple
    states: dict
    kv: dict
    size_bytes: int

    def buffers(self):
        return [tensor for group in (self.states, self.kv) for values in group.values() for tensor in values]


class DevicePrefixStorage:
    def __init__(self, model):
        self.model = model

    def estimate(self, cache, length):
        total = 0
        pages = length // self.model.page_block_size
        for layer in cache.prefill_layers:
            if layer.is_full_attention:
                total += sum(
                    _bytes(tensor) // int(tensor.shape[0]) * pages for tensor in (layer.k_cache, layer.v_cache)
                )
            else:
                total += sum(_bytes(tensor) for tensor in [layer.recurrent_state] + layer.conv_state)
        return total

    def _compact_kv(self, source, blocks):
        """Copy logical pages to [1, heads, tokens, dim], retaining packed dtype."""
        owned = []
        try:
            runs = []
            for block in blocks:
                if runs and block == runs[-1][1]:
                    runs[-1][1] += 1
                else:
                    runs.append([block, block + 1])
            pieces = []
            for first, end in runs:
                piece = ttnn.slice(source, [first, 0, 0, 0], [end, *list(source.shape)[1:]])
                # A full-range slice may alias the caller's cache. Own a clone
                # before any cleanup and never deallocate that source alias.
                if piece.buffer_address() != source.buffer_address():
                    owned.append(piece)
                piece = ttnn.clone(piece, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                owned.append(piece)
                pieces.append(piece)
            compact = pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=0)
            owned.append(compact)
            arranged = ttnn.permute(compact, (1, 0, 2, 3))
            owned.append(arranged)
            heads, pages, page_size, dim = map(int, arranged.shape)
            arranged = ttnn.reshape(arranged, (1, heads, pages * page_size, dim))
            owned.append(arranged)
            return ttnn.clone(arranged, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        finally:
            _release(owned)

    def capture(self, cache, tokens, table):
        states, kv, owned = {}, {}, []
        blocks = list(map(int, table[: len(tokens) // self.model.page_block_size]))
        try:
            for index, layer in enumerate(cache.prefill_layers):
                if layer.is_full_attention:
                    values = []
                    for source in (layer.k_cache, layer.v_cache):
                        tensor = self._compact_kv(source, blocks)
                        owned.append(tensor)
                        values.append(tensor)
                    kv[index] = values
                else:
                    values = []
                    for source in [layer.recurrent_state] + layer.conv_state:
                        tensor = ttnn.clone(source, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                        owned.append(tensor)
                        values.append(tensor)
                    states[index] = values
            return PrefixSnapshot(tuple(tokens), states, kv, sum(map(_bytes, owned)))
        except BaseException:
            _release(owned)
            raise

    def restore(self, cache, snapshot, table):
        mapping = self.model.upload(
            torch.as_tensor(table).reshape(1, -1).to(torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        try:
            for index, values in snapshot.kv.items():
                layer = cache.prefill_layers[index]
                for target, saved in zip((layer.k_cache, layer.v_cache), values):
                    ttnn.experimental.paged_fill_cache(target, saved, mapping, batch_idx=0)
            for index, values in snapshot.states.items():
                layer = cache.prefill_layers[index]
                for target, saved in zip([layer.recurrent_state] + layer.conv_state, values):
                    ttnn.copy(saved, target)
        finally:
            ttnn.deallocate(mapping)

    @staticmethod
    def release(snapshot):
        _release(snapshot.buffers())


class HybridPrefixCache:
    """LRU snapshots scoped to one immutable model/precision/RoPE instance."""

    def __init__(self, model, budget_bytes, max_tokens=4096, *, storage=None):
        if not isinstance(budget_bytes, int) or isinstance(budget_bytes, bool) or budget_bytes <= 0:
            raise ValueError("Prefix cache budget must be a positive integer per device")
        self.page_size = model.page_block_size
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < self.page_size:
            raise ValueError("Prefix cache max_tokens must cover at least one page")
        self.budget_bytes = budget_bytes
        self.max_tokens = max_tokens // self.page_size * self.page_size
        self.storage = storage if storage is not None else DevicePrefixStorage(model)
        self.entries = OrderedDict()
        self.size_bytes = 0
        self.generation = 0
        self.counters = dict(hits=0, misses=0, stores=0, evictions=0, oversized=0, reused_tokens=0)

    def checkpoint_length(self, length):
        # Always evaluate at least one suffix token to produce current logits.
        return min(self.max_tokens, (length - 1) // self.page_size * self.page_size)

    def find(self, tokens):
        limit = min(len(tokens) - 1, self.max_tokens)
        keys = [key for key in self.entries if len(key) <= limit and tuple(tokens[: len(key)]) == key]
        if not keys:
            self.counters["misses"] += 1
            return None
        key = max(keys, key=len)
        self.entries.move_to_end(key)
        return self.entries[key]

    def restore(self, cache, snapshot, table):
        self.storage.restore(cache, snapshot, table)
        self.counters["hits"] += 1
        self.counters["reused_tokens"] += len(snapshot.tokens)

    def save(self, cache, tokens, table):
        key = tuple(tokens)
        if not key or len(key) % self.page_size or len(key) > self.max_tokens:
            raise ValueError("Snapshot must end at a supported complete page")
        if key in self.entries:
            self.entries.move_to_end(key)
            return
        estimate = self.storage.estimate(cache, len(key))
        if estimate > self.budget_bytes:
            self.counters["oversized"] += 1
            return
        while self.entries and self.size_bytes + estimate > self.budget_bytes:
            _, old = self.entries.popitem(last=False)
            self.size_bytes -= old.size_bytes
            self.storage.release(old)
            self.counters["evictions"] += 1
        snapshot = self.storage.capture(cache, key, table)
        if snapshot.size_bytes > estimate:
            self.storage.release(snapshot)
            raise RuntimeError("Prefix snapshot allocation exceeded its pre-allocation estimate")
        self.entries[key] = snapshot
        self.size_bytes += snapshot.size_bytes
        self.generation += 1
        self.counters["stores"] += 1

    def clear(self):
        if self.entries:
            self.generation += 1
        for snapshot in self.entries.values():
            self.storage.release(snapshot)
        self.entries.clear()
        self.size_bytes = 0
