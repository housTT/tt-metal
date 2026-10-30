# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Generator for the Qwen3-8B full model on one Blackhole chip.

``Qwen3Generator`` implements ``readiness_check.contract.Generator``:

* low level: ``prefill_forward`` (several rows per call, absolute ``start_pos`` and ``prompt_lens``
  ends, caller-owned page table and KV cache, last-token logits or on-device sampled tokens) and
  ``decode_forward`` (one traced step over persistent device inputs; on-device split sampling or host
  logits);
* high level: ``prefill_logits``, ``prefill_logits_continuation``, ``generate`` and ``reset`` over a
  generator-owned paged cache and block pool.

Token-out decode is two traces per physical decode batch: the model decode trace (embedding to
sampler-ready logits, position vector advanced on device) and the ``SamplingGenerator`` trace of the
common sampler, which writes the sampled token straight into the decode token tensor (``tt_out_tok``).
Host work between replays is limited to the readback the caller asked for; every host refresh of a
persistent input is counted in ``counters``.

Two-phase warmup: ``prepare`` compiles every prefill bucket, every decode bucket in both output modes
and the sampler variants before the first capture; captures are deferred to first use and kept for
the lifetime of the generator.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, List, Optional, Sequence

import torch
from readiness_check.contract import Generator

import ttnn
from models.autoports.qwen_qwen3_8b.tt.hf_weights import MODEL_ID, CheckpointReader
from models.autoports.qwen_qwen3_8b.tt.model import (
    DECODE_BUCKETS,
    PREFILL_BUCKETS,
    PREFILL_CHUNK,
    PREFILL_CHUNK_STARTS,
    Qwen3Model,
    decode_bucket,
)
from models.autoports.qwen_qwen3_8b.tt.optimized_decoder import WEIGHT_BYTES_PER_ELEMENT
from models.autoports.qwen_qwen3_8b.tt.precision import (
    DTYPE_BY_TTNN,
    decoder_policy,
    kv_cache_dtype,
    layer_policies,
    load_precision_config,
    terminal_policy,
)
from models.common.sampling import SamplingGenerator, SamplingParams, format_sampling_params

SLOT_WIDTH = 32
MAX_TOP_K = 32
DEFAULT_WARMUP_PREFILL_LENGTHS = PREFILL_BUCKETS
DEFAULT_WARMUP_CHUNK_STARTS = PREFILL_CHUNK_STARTS[1:]
DEFAULT_WARMUP_CONTINUATION_STARTS = (32, 64, 128, 256, 33)
COUNTER_NAMES = (
    "decode_trace_replays",
    "sampling_trace_replays",
    "decode_trace_captures",
    "sampling_trace_captures",
    "token_refreshes",
    "position_refreshes",
    "page_table_refreshes",
    "sampling_param_refreshes",
    "token_readbacks",
    "logits_readbacks",
    "eager_decode_steps",
    "eager_sampling_steps",
    "seeded_eager_sampling_steps",
    "prefill_calls",
    "program_cache_entries_after_capture",
)


def greedy_sampling_params(batch: int = 1) -> SamplingParams:
    return SamplingParams(temperature=[0.0] * batch, top_k=[1] * batch, top_p=[1.0] * batch)


@dataclass
class DecodeBucket:
    bucket: int
    tokens: ttnn.Tensor
    current_pos: ttnn.Tensor
    page_table: ttnn.Tensor
    host_pos: torch.Tensor
    host_table: torch.Tensor
    host_tokens: torch.Tensor
    traces: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    prepared: set = field(default_factory=set)
    device_state_valid: bool = False
    last_mode: str | None = None


@dataclass
class DecodeOutput:
    """Device-resident result of ``decode_forward(read_from_device=False)``.

    ``device`` is the persistent decode token tensor (``mode == "device"``) or the decode trace's logits
    output (``mode == "host"``); ``host`` is filled by ``read_decode_output`` and formatted by
    ``process_decode_output_host``. ``eager`` marks a transient eager output that the host step frees.
    """

    state: DecodeBucket
    mode: str
    batch: int
    device: ttnn.Tensor
    host: ttnn.Tensor | None = None
    eager: bool = False


class Qwen3Generator(Generator):
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        model: Qwen3Model,
        tokenizer: Any,
        *,
        max_batch_size: int = SLOT_WIDTH,
        num_blocks: int | None = None,
        kv_cache_dtype: ttnn.DataType = ttnn.bfloat16,
        decode_buckets: Sequence[int] = DECODE_BUCKETS,
        warmup_prefill_lengths: Sequence[int] = DEFAULT_WARMUP_PREFILL_LENGTHS,
        warmup_chunk_starts: Sequence[int] = DEFAULT_WARMUP_CHUNK_STARTS,
        warmup_continuation_starts: Sequence[int] = DEFAULT_WARMUP_CONTINUATION_STARTS,
        warmup_multi_row: bool = True,
        allow_force_argmax: bool = False,
        prepare: bool = True,
        log: Callable[[str], None] = print,
    ) -> None:
        if not 1 <= max_batch_size <= SLOT_WIDTH:
            raise ValueError(f"max_batch_size must be in [1, {SLOT_WIDTH}]")
        self.mesh_device = mesh_device
        self.model = model
        self.tokenizer = tokenizer
        self.log = log
        self.max_batch_size = max_batch_size
        self.block_size = model.block_size
        self.max_context = model.max_context
        self.page_table_width = math.ceil(self.max_context / self.block_size)
        self.num_blocks = int(num_blocks if num_blocks is not None else self.page_table_width)
        self.scratch_block = self.num_blocks
        self.kv_cache_dtype = kv_cache_dtype
        self.owns_cache = True
        self._preparing = False
        self.kv_cache = model.allocate_kv_cache(self.num_blocks, dtype=kv_cache_dtype)
        self.free_blocks: list[int] = list(range(self.num_blocks))
        self.slot_blocks: dict[int, list[int]] = {slot: [] for slot in range(max_batch_size)}
        self.slot_len: dict[int, int] = {slot: 0 for slot in range(max_batch_size)}
        self.eos_ids = self._eos_ids(tokenizer)
        self.counters = {name: 0 for name in COUNTER_NAMES}
        self.allow_force_argmax = allow_force_argmax
        self.decode_buckets = tuple(sorted({decode_bucket(b, model.grid.x, model.grid.y) for b in decode_buckets}))
        self.warmup_prefill_lengths = tuple(warmup_prefill_lengths)
        self.warmup_chunk_starts = tuple(warmup_chunk_starts)
        self.warmup_continuation_starts = tuple(warmup_continuation_starts)
        self.warmup_multi_row = warmup_multi_row
        self._bound_cache_id = None
        self._sampling_signature = None
        self.sampling = SamplingGenerator(args=self._sampling_args(), mesh_device=mesh_device, tt_ccl=None)
        self.buckets: dict[int, DecodeBucket] = {b: self._allocate_bucket(b) for b in self.decode_buckets}
        self.prepared = False
        self._zero_rows = ttnn.from_torch(
            torch.zeros(1, model.geometry.num_kv_heads, PREFILL_CHUNK, model.geometry.head_dim, dtype=torch.bfloat16),
            dtype=kv_cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if prepare:
            self.prepare()

    # --------------------------------------------------------------- setup

    @staticmethod
    def _eos_ids(tokenizer) -> set[int]:
        ids = set()
        eos = getattr(tokenizer, "eos_token_id", None)
        for value in eos if isinstance(eos, (list, tuple)) else [eos]:
            if value is not None:
                ids.add(int(value))
        for token in ("<|im_end|>", "<|endoftext|>"):
            try:
                converted = tokenizer.convert_tokens_to_ids(token)
            except Exception:
                converted = None
            if isinstance(converted, int) and converted >= 0:
                ids.add(converted)
        return ids

    def _sampling_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            vocab_size=self.model.padded_vocab_size,
            padded_vocab_size=self.model.padded_vocab_size,
            cluster_shape=list(self.mesh_device.shape),
            max_batch_size=SLOT_WIDTH,
            max_top_k=MAX_TOP_K,
            sampling_dp=1,
            sub_core_grids=None,
            sub_core_grid_topk=None,
            start_core=ttnn.CoreCoord(0, 0),
            model_config={
                "SAMPLING_AG_CONFIG": {
                    "allow_force_argmax": self.allow_force_argmax,
                    "num_links": 1,
                    "topology": ttnn.Topology.Linear,
                    "chunks_per_sync": 10,
                }
            },
            pad_logits_to_power_of_2=False,
            salt_duplicate_seeds=False,
            use_unfused_ccl=False,
            use_topk_logprobs=False,
        )

    def _allocate_bucket(self, bucket: int) -> DecodeBucket:
        host_tokens = torch.zeros(1, 1, 1, SLOT_WIDTH, dtype=torch.int32)
        host_pos = torch.full((bucket,), -1, dtype=torch.int32)
        host_table = torch.full((bucket, self.page_table_width), self.scratch_block, dtype=torch.int32)
        tokens = ttnn.from_torch(host_tokens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device)
        current_pos = ttnn.from_torch(host_pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device)
        page_table = ttnn.from_torch(
            host_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device
        )
        return DecodeBucket(bucket, tokens, current_pos, page_table, host_pos.clone(), host_table.clone(), host_tokens)

    def prepare(self) -> None:
        """Phase 1 of the trace contract: compile every supported variant before any capture."""
        if self.prepared or self._preparing:
            return
        if self.kv_cache is None:
            raise RuntimeError("prepare() needs a cache: the owned pool or one adopted with adopt_cache()")
        self._preparing = True
        started = time.perf_counter()
        for length in self.warmup_prefill_lengths:
            self.log(f"warmup prefill {length}")
            self.reset()
            self.generate_prefill_only(list(range(1, length + 1)))
        self.reset()
        chunk_lengths = [b for b in self.warmup_prefill_lengths if b <= PREFILL_CHUNK]
        if self.warmup_chunk_starts:
            scratch_table = ttnn.from_torch(
                torch.full((1, self.page_table_width), self.scratch_block, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
            )
            host_scratch = torch.full((1, self.page_table_width + 1), self.scratch_block, dtype=torch.int32)
            for start in self.warmup_chunk_starts:
                self.log(f"warmup chunk pieces at start {start}: {chunk_lengths}")
                for length in chunk_lengths:
                    hidden = self.model.prefill_hidden(
                        list(range(1, length + 1)),
                        start_pos=start,
                        user_slot=0,
                        page_table=scratch_table,
                        kv_cache=self.kv_cache,
                        compute_len=length,
                        host_page_table=host_scratch,
                    )
                    hidden.deallocate(True)
            for start in self.warmup_continuation_starts:
                lead = start % SLOT_WIDTH
                self.log(f"warmup continuation at start {start} (lead {lead}): {chunk_lengths}")
                for length in chunk_lengths:
                    hidden = self.model.prefill_hidden(
                        list(range(1, length - lead + 1)),
                        start_pos=start,
                        user_slot=0,
                        page_table=scratch_table,
                        kv_cache=self.kv_cache,
                        host_page_table=host_scratch,
                    )
                    hidden.deallocate(True)
            scratch_table.deallocate(True)
        rows = SLOT_WIDTH
        if self.warmup_multi_row and rows in self.buckets:
            self.log(f"warmup multi-row prefill {rows} rows")
            self.reset()
            lengths = [PREFILL_BUCKETS[i % 2] for i in range(rows)]
            tokens = torch.zeros(rows, max(lengths), dtype=torch.int64)
            for i, n in enumerate(lengths):
                tokens[i, :n] = torch.arange(1, n + 1)
                self._ensure_blocks(i, n)
            table = self._host_page_table(list(range(rows)))
            self.prefill_forward(tokens, page_table=table, kv_cache=self.kv_cache, prompt_lens=lengths)
            self.prefill_forward(
                tokens,
                page_table=table,
                kv_cache=self.kv_cache,
                prompt_lens=lengths,
                sampling_params=greedy_sampling_params(rows),
            )
            self.reset()
        for bucket in self.decode_buckets:
            state = self.buckets[bucket]
            table = torch.full((bucket, self.page_table_width), self.scratch_block, dtype=torch.int32)
            pos = torch.zeros(bucket, dtype=torch.int32)
            self._write_positions(state, pos)
            self._write_page_table(state, table)
            for mode in ("device", "host"):
                self.log(f"warmup decode bucket {bucket} mode {mode}")
                out = self.model.ttnn_decode_forward(
                    state.tokens,
                    state.current_pos,
                    state.page_table,
                    self.kv_cache,
                    on_device_sampling=(mode == "device"),
                )
                if mode == "device":
                    self.sampling.set_trace_bucket(bucket)
                    self.sampling.precompile(logits=out, tt_out_tok=state.tokens)
                    self.sampling.set_trace_bucket(None)
                out.deallocate(True)
                state.prepared.add(mode)
            self._write_positions(state, torch.full((bucket,), -1, dtype=torch.int32))
            state.device_state_valid = False
        for state in self.buckets.values():
            ttnn.synchronize_device(self.mesh_device)
        self.reset()
        self.prepared = True
        self._preparing = False
        self.log(f"prepare done in {time.perf_counter() - started:.1f} s")

    def _require_prepared(self) -> None:
        """Generator-owned warmup enforcement: no forward pass runs before phase 1 completed."""
        if not self.prepared and not self._preparing:
            self.prepare()

    def release_owned_cache(self) -> None:
        """Free the generator-owned KV cache after ``prepare()`` when an external caller owns the cache.

        The high-level API (``generate``, ``prefill_logits``) is unavailable afterwards; traces bind to the
        caller's cache on their deferred first capture.
        """
        if self.kv_cache is None:
            return
        if self.owns_cache:
            for key_cache, value_cache in self.kv_cache:
                key_cache.deallocate(True)
                value_cache.deallocate(True)
        self.kv_cache = None
        self.free_blocks = []
        for slot in self.slot_blocks:
            self.slot_blocks[slot] = []
            self.slot_len[slot] = 0

    def adopt_cache(self, kv_cache: list[tuple], *, scratch_block: int) -> None:
        """Map the generator's block pool onto a caller-owned cache so ``prepare()`` and the deferred captures
        run over the tensors that serve requests (paged-attention program keys include the cache shape).

        The pool covers the first ``page_table_width`` blocks; ``scratch_block`` is the caller's extra block.
        ``release_owned_cache`` detaches the pool again without freeing the caller's tensors.
        """
        self.release_owned_cache()
        self.kv_cache = kv_cache
        self.owns_cache = False
        self.scratch_block = int(scratch_block)
        self.num_blocks = int(scratch_block)
        self.free_blocks = list(range(min(self.num_blocks, self.page_table_width)))
        self._bound_cache_id = None

    def teardown(self) -> None:
        self.log(f"runtime counters at teardown: {json.dumps(self.counter_snapshot(), sort_keys=True)}")
        for state in self.buckets.values():
            for trace_id in state.traces.values():
                ttnn.release_trace(self.mesh_device, trace_id)
            state.traces.clear()
            state.outputs.clear()
        self.sampling.reset_trace()

    # --------------------------------------------------------- block pool

    def _ensure_blocks(self, slot: int, tokens: int) -> list[int]:
        need = math.ceil(tokens / self.block_size)
        have = self.slot_blocks[slot]
        while len(have) < need:
            if not self.free_blocks:
                raise RuntimeError(f"KV block pool of {self.num_blocks} blocks is exhausted (slot {slot} needs {need})")
            have.append(self.free_blocks.pop(0))
        return have

    def _release_slot(self, slot: int, zero: bool = True) -> None:
        blocks = self.slot_blocks[slot]
        if blocks and zero:
            self._zero_blocks(blocks)
        self.free_blocks.extend(blocks)
        self.free_blocks.sort()
        self.slot_blocks[slot] = []
        self.slot_len[slot] = 0

    def _zero_blocks(self, blocks: list[int]) -> None:
        per_window = PREFILL_CHUNK // self.block_size
        for start in range(0, len(blocks), per_window):
            window = blocks[start : start + per_window]
            table = torch.full((1, per_window), self.scratch_block, dtype=torch.int32)
            table[0, : len(window)] = torch.tensor(window, dtype=torch.int32)
            tt_table = ttnn.from_torch(table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device)
            for key_cache, value_cache in self.kv_cache:
                ttnn.experimental.paged_fill_cache(key_cache, self._zero_rows, tt_table, batch_idx=0)
                ttnn.experimental.paged_fill_cache(value_cache, self._zero_rows, tt_table, batch_idx=0)
            tt_table.deallocate(True)

    def _host_page_table(self, rows: Sequence[int]) -> torch.Tensor:
        table = torch.full((len(rows), self.page_table_width), self.scratch_block, dtype=torch.int32)
        for i, slot in enumerate(rows):
            blocks = self.slot_blocks[slot]
            if blocks:
                table[i, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        return table

    def _scratch_for(self, kv_cache: Any, scratch_block: int | None) -> int:
        """The scratch block for ``kv_cache``: the generator's own for its cache, else the caller's explicit id."""
        if scratch_block is not None:
            return int(scratch_block)
        if self.kv_cache is not None and kv_cache is self.kv_cache:
            return self.scratch_block
        raise ValueError(
            "a caller-owned kv_cache needs scratch_block=<id of its extra block>; Qwen3Model.allocate_kv_cache "
            "allocates num_blocks + 1 blocks and the last one is that scratch block"
        )

    def _redirect_table(
        self, page_table: torch.Tensor, valid_tokens: Sequence[int], scratch_block: int | None = None
    ) -> torch.Tensor:
        """Full-width int32 copy of a caller page table whose columns past each row's ``valid_tokens`` name the
        scratch block (a caller's rows can carry stale block ids past the request's blocks, and the masked
        tail of a decode read may touch them, so the count must stay exact)."""
        rows = page_table.shape[0]
        scratch = self.scratch_block if scratch_block is None else int(scratch_block)
        table = torch.full((rows, self.page_table_width), scratch, dtype=torch.int32)
        width = min(page_table.shape[1], self.page_table_width)
        valid = torch.tensor(
            [min(width, math.ceil(int(v) / self.block_size)) for v in valid_tokens], dtype=torch.int64
        ).unsqueeze(1)
        keep = torch.arange(width, dtype=torch.int64).unsqueeze(0) < valid
        table[:, :width] = torch.where(keep, page_table[:, :width].to(torch.int32), table[:, :width])
        return table

    # ------------------------------------------------------- persistent I/O

    def _write_tokens(self, state: DecodeBucket, tokens: torch.Tensor) -> None:
        host = torch.zeros(1, 1, 1, SLOT_WIDTH, dtype=torch.int32)
        host[0, 0, 0, : tokens.numel()] = tokens.reshape(-1).to(torch.int32)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), state.tokens
        )
        state.host_tokens = host
        self.counters["token_refreshes"] += 1

    def _write_positions(self, state: DecodeBucket, positions: torch.Tensor) -> None:
        host = torch.full((state.bucket,), -1, dtype=torch.int32)
        host[: positions.numel()] = positions.reshape(-1).to(torch.int32)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), state.current_pos
        )
        state.host_pos = host
        self.counters["position_refreshes"] += 1

    def _write_page_table(self, state: DecodeBucket, table: torch.Tensor, scratch: int | None = None) -> None:
        pad_block = self.scratch_block if scratch is None else int(scratch)
        host = torch.full((state.bucket, self.page_table_width), pad_block, dtype=torch.int32)
        host[: table.shape[0]] = table.to(torch.int32)
        if torch.equal(host, state.host_table):
            return
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), state.page_table
        )
        state.host_table = host
        self.counters["page_table_refreshes"] += 1

    def _read_tokens(self, state: DecodeBucket, count: int) -> torch.Tensor:
        self.counters["token_readbacks"] += 1
        return ttnn.to_torch(state.tokens).reshape(-1)[:count].to(torch.int64)

    # ----------------------------------------------------------- sampling

    def _apply_sampling_params(
        self,
        sampling_params: SamplingParams,
        batch: int,
        *,
        admit: bool,
        positions: Sequence[int] | None = None,
        slot_remap: Any = None,
    ) -> None:
        """Refresh k/p/temp on change and keep the common sampler's seed manager in step with the batch.

        Prefill (``positions`` is None): the rows are freshly admitted requests, their seeds are registered
        with a fresh salt and nothing else is touched. Decode: the live slots are the rows at a position
        of 0 or more; ``slot_remap`` is applied first, slots that are no longer live are deactivated, seeds
        are re-registered for every live slot when ``admit`` (``reset_batch``, first decode after prefill,
        mode change) and only for slots whose seed changed otherwise, counters are aligned to the
        authoritative positions of the (re)registered slots, and ``get_new_values`` advances the seeded
        slots (a no-op for unseeded steady state). Mirrors ``models/tt_transformers/tt/generator.py``
        ``sample_decode_on_device``.
        """
        formatted = format_sampling_params(sampling_params, SLOT_WIDTH, MAX_TOP_K)
        signature = (
            tuple(formatted.temperature),
            tuple(formatted.top_k),
            tuple(formatted.top_p),
            tuple(formatted.presence_penalty),
            tuple(formatted.frequency_penalty),
            tuple(formatted.repetition_penalty),
            (
                tuple(formatted.enable_log_probs)
                if isinstance(formatted.enable_log_probs, list)
                else formatted.enable_log_probs
            ),
        )
        if signature != self._sampling_signature:
            self.sampling.reset_sampling_params(formatted)
            self._sampling_signature = signature
            self.counters["sampling_param_refreshes"] += 1
        manager = self.sampling.seed_manager
        if positions is None:
            slots = list(range(batch))
            manager.reset_seed(formatted.seed, slots)
            manager.get_new_values(slots)
            return
        position_list = [int(p) for p in positions]
        live = [slot for slot, p in enumerate(position_list[:SLOT_WIDTH]) if p >= 0]
        if slot_remap is not None:
            remap = torch.as_tensor(slot_remap).reshape(-1).to(torch.int64)
            full = torch.arange(SLOT_WIDTH, dtype=torch.int64)
            full[: min(SLOT_WIDTH, remap.numel())] = remap[:SLOT_WIDTH]
            manager.apply_slot_remap(full)
        manager.deactivate_slots_except(live)
        if live:
            if admit:
                manager.reset_seed_from_slots(formatted.seed, live)
                align = list(live)
            else:
                align = manager.reset_seed_from_slots_if_needed(formatted.seed, live)
            if align:
                manager.align_seed_counters_to_positions(formatted.seed, align, position_list)
        manager.get_new_values(live)

    def _sampler_is_traced(self) -> bool:
        """Mirror of ``SamplingGenerator.sample``: explicit request seeds run the sampler eagerly."""
        return not self.sampling.seed_manager.has_active_request_seed()

    def _sample_eager(self, logits: ttnn.Tensor, state: DecodeBucket) -> None:
        self.sampling.set_trace_bucket(None)
        self.sampling.sample(logits, enable_trace=False, tt_out_tok=state.tokens)
        self.counters["eager_sampling_steps"] += 1

    # ------------------------------------------------------------ traces

    def _bind_cache(self, kv_cache: list[tuple]) -> None:
        if self._bound_cache_id is None:
            self._bound_cache_id = id(kv_cache)
        elif self._bound_cache_id != id(kv_cache):
            raise RuntimeError("decode traces are bound to the KV cache they were captured with; pass the same cache")

    def _ensure_trace(self, state: DecodeBucket, mode: str, kv_cache: list[tuple]) -> ttnn.Tensor:
        if mode in state.traces:
            return state.outputs[mode]
        if mode not in state.prepared:
            raise RuntimeError(f"decode bucket {state.bucket} mode {mode} was not prepared before capture")
        self._bind_cache(kv_cache)
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        out = self.model.ttnn_decode_forward(
            state.tokens, state.current_pos, state.page_table, kv_cache, on_device_sampling=(mode == "device")
        )
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.mark_corruptible(out)
        state.traces[mode] = trace_id
        state.outputs[mode] = out
        self.counters["decode_trace_captures"] += 1
        if mode == "device":
            self.sampling.set_trace_bucket(state.bucket)
            self.sampling.capture_trace(logits=out, tt_out_tok=state.tokens, skip_precompile=True)
            self.sampling.set_trace_bucket(None)
            self.counters["sampling_trace_captures"] += 1
        self.counters["program_cache_entries_after_capture"] = self.mesh_device.num_program_cache_entries()
        return out

    def _replay(self, state: DecodeBucket, mode: str, kv_cache: list[tuple]) -> ttnn.Tensor:
        out = self._ensure_trace(state, mode, kv_cache)
        ttnn.execute_trace(self.mesh_device, state.traces[mode], cq_id=0, blocking=False)
        self.counters["decode_trace_replays"] += 1
        if mode == "device":
            traced = self._sampler_is_traced()
            self.sampling.set_trace_bucket(state.bucket)
            self.sampling.sample(out, enable_trace=True, tt_out_tok=state.tokens)
            self.sampling.set_trace_bucket(None)
            if traced:
                self.counters["sampling_trace_replays"] += 1
            else:
                self.counters["seeded_eager_sampling_steps"] += 1
        return out

    # ------------------------------------------------------------ prefill

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table: torch.Tensor,
        kv_cache: Any,
        prompt_lens: List[int],
        start_pos: Optional[torch.Tensor] = None,
        return_all_logits: bool = False,
        sampling_params: SamplingParams | None = None,
        empty_slots: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Prefill every row of ``tokens`` ``[B, T]``; row ``i`` computes ``tokens[i, start_pos[i]:prompt_lens[i]]``.

        ``page_table`` is ``[B, blocks]`` (row ``i`` is the request's block list); ``kv_cache`` is the
        per-layer list of ``(K, V)`` pairs. Returns ``[B, 1, vocab]`` last-token logits, ``[B, T, vocab]``
        with ``return_all_logits`` (single row only), or ``[B]`` sampled tokens with ``sampling_params``.
        """
        self._require_prepared()
        rows = tokens.shape[0]
        starts = [0] * rows if start_pos is None else [int(s) for s in torch.as_tensor(start_pos).reshape(-1)]
        ends = [int(e) for e in prompt_lens]
        if len(ends) != rows or len(starts) != rows:
            raise ValueError("prompt_lens and start_pos need one entry per row")
        for i in range(rows):
            if not 0 <= starts[i] < ends[i] <= self.max_context:
                raise ValueError(f"row {i}: start {starts[i]} and end {ends[i]} outside [0, {self.max_context}]")
        self.counters["prefill_calls"] += 1
        scratch = self._scratch_for(kv_cache, kwargs.get("scratch_block"))
        table = self._redirect_table(torch.as_tensor(page_table), ends, scratch)

        def row_table(i: int) -> ttnn.Tensor:
            return ttnn.from_torch(
                table[i : i + 1], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device
            )

        def host_row(i: int) -> torch.Tensor:
            return torch.cat([table[i : i + 1], torch.full((1, 1), scratch, dtype=torch.int32)], dim=1)

        if return_all_logits:
            if rows != 1:
                raise ValueError("return_all_logits supports one row per call")
            ids = tokens[0, starts[0] : ends[0]].tolist()
            tt_table = row_table(0)
            hidden = self.model.prefill_hidden(
                ids,
                start_pos=starts[0],
                user_slot=0,
                page_table=tt_table,
                kv_cache=kv_cache,
                host_page_table=host_row(0),
            )
            lead = self.model.lead_rows(starts[0], len(ids))
            logits = self.model.logits_all_rows(hidden)[lead : lead + len(ids)]
            hidden.deallocate(True)
            tt_table.deallocate(True)
            return logits.unsqueeze(0)
        pieces = []
        for i in range(rows):
            ids = tokens[i, starts[i] : ends[i]].tolist()
            tt_table = row_table(i)
            hidden = self.model.prefill_hidden(
                ids,
                start_pos=starts[i],
                user_slot=0,
                page_table=tt_table,
                kv_cache=kv_cache,
                keep_all_rows=False,
                host_page_table=host_row(i),
            )
            tt_table.deallocate(True)
            pieces.append(self.model.gather_rows(hidden, [self.model.last_row_index(len(ids), starts[i])]))
            hidden.deallocate(True)
        tile = self.model.rows_to_tile(pieces)
        for piece in pieces:
            piece.deallocate(True)
        logits = self.model.logits_from_tile(tile)
        tile.deallocate(True)
        if sampling_params is not None:
            bucket = self.buckets[decode_bucket(rows, self.model.grid.x, self.model.grid.y)]
            self._apply_sampling_params(sampling_params, rows, admit=True)
            self._sample_eager(logits, bucket)
            bucket.device_state_valid = False
            logits.deallocate(True)
            return self._read_tokens(bucket, rows)
        self.counters["logits_readbacks"] += 1
        host = ttnn.to_torch(logits)[0, 0, :rows, : self.model.vocab_size].to(torch.float32)
        logits.deallocate(True)
        return host.unsqueeze(1)

    def prefill_logits_rows(
        self, token_ids: Sequence[int], start: int, positions: Sequence[int], slot: int = 0
    ) -> torch.Tensor:
        """Continuation prefill of ``token_ids`` at absolute ``start`` on the generator cache; logits at ``positions``."""
        ids = list(token_ids)
        self._ensure_blocks(slot, start + len(ids))
        table = self._host_page_table([slot])
        tt_table = ttnn.from_torch(table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device)
        hidden = self.model.prefill_hidden(
            ids,
            start_pos=start,
            user_slot=0,
            page_table=tt_table,
            kv_cache=self.kv_cache,
            host_page_table=torch.cat([table, torch.full((1, 1), self.scratch_block, dtype=torch.int32)], dim=1),
        )
        tt_table.deallocate(True)
        lead = self.model.lead_rows(start, len(ids))
        if any(not 0 <= p - start < len(ids) for p in positions):
            raise ValueError("positions must lie inside the continuation window")
        local = [lead + p - start for p in positions]
        if len(local) == len(ids):
            logits = self.model.logits_all_rows(hidden)[lead : lead + len(ids)]
            hidden.deallocate(True)
            self.slot_len[slot] = start + len(ids)
            return logits.unsqueeze(0)
        rows = self.model.gather_rows(hidden, local)
        hidden.deallocate(True)
        tile = self.model.stack_rows_tile(
            [rows], pad_to=max(SLOT_WIDTH, math.ceil(len(local) / SLOT_WIDTH) * SLOT_WIDTH)
        )
        rows.deallocate(True)
        logits = self.model.logits_all_rows(tile)[: len(local)]
        tile.deallocate(True)
        self.slot_len[slot] = start + len(ids)
        return logits.unsqueeze(0)

    # ---------------------------------------------------------- decode API

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table: torch.Tensor,
        kv_cache: Any,
        enable_trace: bool = True,
        read_from_device: bool = True,
        sampling_params: SamplingParams | None = None,
        reset_batch: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """One decode step for ``tokens`` ``[B, 1]`` at ``start_pos`` ``[B]`` (``-1`` marks an inactive row).

        With ``sampling_params`` the step samples on device and returns ``[B]`` token ids; the device
        token and position state is authoritative across steady steps, so host tokens are only loaded
        on the first step after a prefill, after ``reset_batch`` or after a mode change. Without
        ``sampling_params`` the step returns ``[B, vocab]`` logits and reloads tokens and positions
        from the host every step (host-sampling compatibility mode). ``state.host_pos`` mirrors the
        device position (loaded on a reload, advanced once per step); the page-table redirect uses the
        mirror, because a caller that overlaps steps can pass a ``start_pos`` one behind the device.
        """
        self._require_prepared()
        batch = tokens.shape[0]
        mode = "device" if sampling_params is not None else "host"
        bucket = decode_bucket(batch, self.model.grid.x, self.model.grid.y)
        state = self.buckets[bucket]
        positions = torch.as_tensor(start_pos).reshape(-1).to(torch.int32)
        if positions.numel() != batch:
            raise ValueError(f"start_pos has {positions.numel()} entries for {batch} rows")
        if int(positions.max()) >= self.max_context:
            raise ValueError(f"decode position {int(positions.max())} is outside the {self.max_context}-token context")
        reload = (
            mode == "host" or reset_batch or not state.device_state_valid or state.last_mode != mode or not enable_trace
        )
        if reload:
            self._write_tokens(state, torch.as_tensor(tokens).reshape(-1))
            self._write_positions(state, positions)
        device_positions = state.host_pos[:batch]
        if int(device_positions.max()) >= self.max_context:
            raise ValueError(
                f"device position {int(device_positions.max())} is outside the {self.max_context}-token context"
            )
        scratch = self._scratch_for(kv_cache, kwargs.get("scratch_block"))
        self._write_page_table(
            state,
            self._redirect_table(torch.as_tensor(page_table), [int(p) + 1 for p in device_positions], scratch),
            scratch,
        )
        if mode == "device":
            self._apply_sampling_params(
                sampling_params,
                batch,
                admit=reload,
                positions=positions.tolist(),
                slot_remap=kwargs.get("slot_remap"),
            )
        if enable_trace:
            out = self._replay(state, mode, kv_cache)
        else:
            self._bind_cache(kv_cache)
            out = self.model.ttnn_decode_forward(
                state.tokens, state.current_pos, state.page_table, kv_cache, on_device_sampling=(mode == "device")
            )
            self.counters["eager_decode_steps"] += 1
            if mode == "device":
                self._sample_eager(out, state)
        state.last_mode = mode
        state.device_state_valid = mode == "device" and enable_trace
        state.host_pos = torch.where(state.host_pos >= 0, state.host_pos + 1, state.host_pos)
        if not read_from_device:
            return DecodeOutput(state, mode, batch, state.tokens if mode == "device" else out, eager=not enable_trace)
        if mode == "device":
            result = self._read_tokens(state, batch)
            if not enable_trace:
                out.deallocate(True)
            return result
        self.counters["logits_readbacks"] += 1
        host = ttnn.to_torch(out)[0, 0, :batch].to(torch.float32)
        if not enable_trace:
            out.deallocate(True)
        return host

    def read_decode_output(self, out: DecodeOutput, async_read: bool = False):
        """Start the host read of a deferred decode output.

        With ``async_read`` the read is enqueued without blocking and the call returns
        ``(out, [event])``; the caller synchronizes the event before ``process_decode_output_host``.
        The read is enqueued on the same command queue as the replay, so a later replay that overwrites
        the persistent token tensor is ordered after it.
        """
        if not async_read:
            out.host = out.device.cpu(blocking=True)
            return out
        out.host = out.device.cpu(blocking=False)
        return out, [ttnn.record_event(self.mesh_device, 0)]

    def process_decode_output_host(self, out: DecodeOutput, is_tokens: bool = False) -> torch.Tensor:
        """Host formatting of a read decode output: ``[B]`` token ids or ``[B, vocab]`` float32 logits."""
        host = out.host if out.host is not None else out.device.cpu(blocking=True)
        if out.mode == "device":
            self.counters["token_readbacks"] += 1
            result = ttnn.to_torch(host).reshape(-1)[: out.batch].to(torch.int64)
        else:
            self.counters["logits_readbacks"] += 1
            result = ttnn.to_torch(host)[0, 0, : out.batch].to(torch.float32)
        if out.eager:
            out.device.deallocate(True)
        out.host = None
        return result

    def capture_decode_traces(
        self, kv_cache: Any, *, modes: Sequence[str] = ("device",), scratch_block: int | None = None
    ) -> dict:
        """Phase 2 of the trace contract: capture the prepared decode (and sampling) traces on ``kv_cache``.

        Every bucket replays once over the scratch block at position 0 so the capture is executed before
        a request uses it; the bucket state is then marked invalid so the first real step reloads it.
        Returns ``{(bucket, mode): trace_id}`` for the traces captured by this call.
        """
        if not self.prepared:
            raise RuntimeError("prepare() must run before any trace capture")
        scratch = self._scratch_for(kv_cache, scratch_block)
        captured = {}
        for bucket in self.decode_buckets:
            state = self.buckets[bucket]
            for mode in modes:
                if mode in state.traces:
                    continue
                self._write_tokens(state, torch.zeros(bucket, dtype=torch.int32))
                self._write_positions(state, torch.zeros(bucket, dtype=torch.int32))
                self._write_page_table(
                    state, torch.full((bucket, self.page_table_width), scratch, dtype=torch.int32), scratch
                )
                if mode == "device":
                    self._apply_sampling_params(
                        greedy_sampling_params(bucket), bucket, admit=True, positions=[0] * bucket
                    )
                self._replay(state, mode, kv_cache)
                captured[(bucket, mode)] = state.traces[mode]
            self._write_positions(state, torch.full((bucket,), -1, dtype=torch.int32))
            state.device_state_valid = False
            state.last_mode = None
        ttnn.synchronize_device(self.mesh_device)
        self.sampling.seed_manager.deactivate_slots_except([])
        return captured

    # ------------------------------------------------------ high-level API

    def reset(self) -> None:
        for slot in range(self.max_batch_size):
            self._release_slot(slot)
        for state in self.buckets.values():
            state.device_state_valid = False
            state.last_mode = None

    def prefill_logits(self, prompt_token_ids: List[int]) -> torch.Tensor:
        self.reset()
        ids = list(prompt_token_ids)
        self._ensure_blocks(0, len(ids))
        table = self._host_page_table([0])
        logits = self.prefill_forward(
            torch.tensor([ids], dtype=torch.int64),
            page_table=table,
            kv_cache=self.kv_cache,
            prompt_lens=[len(ids)],
            return_all_logits=True,
        )
        self.slot_len[0] = len(ids)
        return logits

    def prefill_logits_continuation(
        self,
        prompt_token_ids: List[int],
        start: int,
        *,
        positions: Optional[Sequence[int]] = None,
        prefill_prefix: bool = True,
    ) -> torch.Tensor:
        self.reset()
        ids = list(prompt_token_ids)
        if not 0 <= start < len(ids):
            raise ValueError(f"start {start} outside [0, {len(ids)})")
        wanted = list(positions) if positions is not None else list(range(start, len(ids)))
        if start > 0 and prefill_prefix:
            self.prefill_logits_rows(ids[:start], 0, [start - 1])
        return self.prefill_logits_rows(ids[start:], start, wanted)

    def generate_prefill_only(self, prompt_token_ids: Sequence[int], slot: int = 0) -> int:
        """Prefill one prompt into ``slot`` with greedy on-device sampling; returns the first predicted token."""
        ids = list(prompt_token_ids)
        self._ensure_blocks(slot, len(ids))
        table = self._host_page_table([slot])
        bucket = self.buckets[1]
        token = self.prefill_forward(
            torch.tensor([ids], dtype=torch.int64),
            page_table=table,
            kv_cache=self.kv_cache,
            prompt_lens=[len(ids)],
            sampling_params=greedy_sampling_params(1),
        )
        self.slot_len[slot] = len(ids)
        self._write_positions(bucket, torch.tensor([len(ids)], dtype=torch.int32))
        self._write_page_table(bucket, table)
        bucket.device_state_valid = True
        bucket.last_mode = "device"
        return int(token[0])

    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
        *,
        next_input: Optional[Callable[[int, int], int]] = None,
        enable_trace: bool = True,
        sampling_params: SamplingParams | None = None,
        stop_on_eos: bool | None = None,
        **kwargs: Any,
    ) -> List[int]:
        """Token-in token-out generation for one prompt on slot 0 (greedy by default)."""
        if max_new_tokens < 1:
            return []
        ids = list(prompt_token_ids)
        if len(ids) + max_new_tokens > self.max_context:
            raise ValueError(
                f"prompt of {len(ids)} tokens plus {max_new_tokens} generated tokens exceeds the "
                f"{self.max_context}-token context (the last decode position must stay below it)"
            )
        self.reset()
        params = sampling_params if sampling_params is not None else greedy_sampling_params(1)
        self._ensure_blocks(0, len(ids) + max_new_tokens)
        table = self._host_page_table([0])
        state = self.buckets[1]
        first = self.prefill_forward(
            torch.tensor([ids], dtype=torch.int64),
            page_table=table,
            kv_cache=self.kv_cache,
            prompt_lens=[len(ids)],
            sampling_params=params,
        )
        self.slot_len[0] = len(ids)
        self._write_positions(state, torch.tensor([len(ids)], dtype=torch.int32))
        self._write_page_table(state, table)
        state.device_state_valid = True
        state.last_mode = "device"
        stop = stop_on_eos if stop_on_eos is not None else next_input is None
        predictions = [int(first[0])]
        forced = next_input(0, predictions[-1]) if next_input is not None else predictions[-1]
        for step in range(1, max_new_tokens):
            if stop and predictions[-1] in self.eos_ids:
                break
            if forced != predictions[-1]:
                self._write_tokens(state, torch.tensor([forced], dtype=torch.int32))
            self._apply_sampling_params(params, 1, admit=False, positions=[len(ids) + step - 1])
            if enable_trace:
                self._replay(state, "device", self.kv_cache)
            else:
                out = self.model.ttnn_decode_forward(
                    state.tokens, state.current_pos, state.page_table, self.kv_cache, on_device_sampling=True
                )
                self.counters["eager_decode_steps"] += 1
                self._sample_eager(out, state)
                out.deallocate(True)
            predictions.append(int(self._read_tokens(state, 1)[0]))
            self.slot_len[0] += 1
            forced = next_input(step, predictions[-1]) if next_input is not None else predictions[-1]
        return predictions

    # ----------------------------------------------------------- budgets

    DEVICE_DRAM_BYTES = 33_641_860_096
    WEIGHTS_RESIDENT_BYTES = 14_928_363_520
    ALLOCATOR_ROUNDING_BYTES = 14_928_363_520 - 14_899_046_400
    RUNTIME_RESERVE_BYTES = 2_147_483_648
    KV_BYTES_PER_TOKEN = 147_456

    @classmethod
    def max_tokens_all_users(
        cls,
        model: Qwen3Model | None = None,
        *,
        mesh_device=None,
        kv_cache_dtype: ttnn.DataType = ttnn.bfloat16,
        **_: Any,
    ) -> int:
        """KV-cache token budget on one chip: DRAM minus resident weights minus a runtime reserve, in blocks.

        Mirrors the plugin's ``get_max_tokens_all_users`` hook. With ``mesh_device`` the allocated and total
        bytes are read from ``ttnn.get_memory_view`` after the weights are on device; with only a model the
        resident bytes are computed from its dtype policy plus the stage-6 measured allocator rounding; with
        neither, the stage-6 measured constants recorded in ``doc/context_contract.json`` are used.
        ``kv_cache_dtype`` sizes the per-token cache bytes (BF16 by default; a BFP8 cache halves them).
        """
        per_token = (
            model.kv_bytes_per_token(kv_cache_dtype)
            if model is not None
            else int(cls.KV_BYTES_PER_TOKEN * WEIGHT_BYTES_PER_ELEMENT[DTYPE_BY_TTNN[kv_cache_dtype]] / 2)
        )
        if mesh_device is not None:
            view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
            free = int(view.total_bytes_free_per_bank) * int(view.num_banks)
        elif model is not None:
            free = cls.DEVICE_DRAM_BYTES - model.weight_bytes()["total_resident"] - cls.ALLOCATOR_ROUNDING_BYTES
        else:
            free = cls.DEVICE_DRAM_BYTES - cls.WEIGHTS_RESIDENT_BYTES
        tokens = (free - cls.RUNTIME_RESERVE_BYTES) // per_token
        return max(SLOT_WIDTH, (tokens // SLOT_WIDTH) * SLOT_WIDTH)

    # ----------------------------------------------------------- reports

    def counter_snapshot(self) -> dict:
        return dict(self.counters)

    def trace_summary(self) -> dict:
        return {
            bucket: {"decode_traces": dict(state.traces), "prepared": sorted(state.prepared)}
            for bucket, state in self.buckets.items()
        }


def build_generator(model_dir: str | Path, mesh_device, **kwargs) -> Qwen3Generator:
    """Readiness-contract factory: real Qwen3-8B weights from the local HF snapshot.

    Keyword overrides: ``num_layers`` (reduced probe), ``decode_buckets``, ``warmup_prefill_lengths``,
    ``warmup_chunk_starts``, ``warmup_multi_row``, ``num_blocks``, ``max_batch_size``,
    ``allow_force_argmax``, ``policy``, ``terminal_policy``, ``tensor_cache_path``, ``prepare``.
    """
    from transformers import AutoConfig, AutoTokenizer

    log = kwargs.pop("log", print)
    reader = CheckpointReader(kwargs.pop("snapshot", None))
    hf_config = AutoConfig.from_pretrained(str(reader.snapshot), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(reader.snapshot), local_files_only=True)
    model_kwargs = {
        key: kwargs.pop(key)
        for key in ("num_layers", "policy", "layer_policies", "terminal_policy", "tensor_cache_path")
        if key in kwargs
    }
    precision_config, precision_path = load_precision_config(kwargs.pop("precision_config", None))
    num_layers = int(model_kwargs.get("num_layers") or hf_config.num_hidden_layers)
    log(
        f"precision config {precision_config['config_id']} from "
        f"{precision_path if precision_path is not None else 'the built-in baseline'}: "
        f"weight groups {json.dumps(precision_config['weight_groups'], sort_keys=True)}, "
        f"layer exceptions {sorted(precision_config.get('layer_exceptions', {}))}, "
        f"kv cache {precision_config['kv_cache_dtype']}, lm head {json.dumps(precision_config['lm_head'], sort_keys=True)}"
    )
    if "policy" not in model_kwargs:
        policy_overrides = {}
        if "QWEN3_PREFILL_KERNEL_TABLE" in os.environ:
            policy_overrides["prefill_kernel_table"] = os.environ["QWEN3_PREFILL_KERNEL_TABLE"]
        if "QWEN3_PREFILL_SHARDED_NORM_MAX_ROWS" in os.environ:
            policy_overrides["prefill_sharded_norm_max_rows"] = int(os.environ["QWEN3_PREFILL_SHARDED_NORM_MAX_ROWS"])
        model_kwargs["policy"] = replace(decoder_policy(precision_config), **policy_overrides)
        if "layer_policies" not in model_kwargs:
            model_kwargs["layer_policies"] = {
                idx: replace(p, **policy_overrides) for idx, p in layer_policies(precision_config, num_layers).items()
            }
        if policy_overrides:
            log(f"decoder policy overrides from the environment: {policy_overrides}")
    if "terminal_policy" not in model_kwargs:
        overrides = {
            name: os.environ[env]
            for name, env in (
                ("lm_head_decode_fidelity", "QWEN3_LM_HEAD_DECODE_FIDELITY"),
                ("lm_head_prefill_fidelity", "QWEN3_LM_HEAD_PREFILL_FIDELITY"),
                ("lm_head_dtype", "QWEN3_LM_HEAD_DTYPE"),
                ("prefill_logits_dtype", "QWEN3_PREFILL_LOGITS_DTYPE"),
            )
            if env in os.environ
        }
        model_kwargs["terminal_policy"] = replace(terminal_policy(precision_config), **overrides)
        if overrides:
            log(f"terminal policy overrides from the environment: {overrides}")
    if "kv_cache_dtype" not in kwargs:
        kwargs["kv_cache_dtype"] = kv_cache_dtype(precision_config)
    started = time.perf_counter()
    model = Qwen3Model(mesh_device, hf_config=hf_config, reader=reader, log=log, **model_kwargs)
    log(f"model weights on device in {time.perf_counter() - started:.1f} s")
    generator = Qwen3Generator(mesh_device, model, tokenizer, log=log, **kwargs)
    generator.precision_config = precision_config
    generator.precision_config_path = precision_path
    return generator


__all__ = ["Qwen3Generator", "build_generator", "greedy_sampling_params", "MODEL_ID"]
