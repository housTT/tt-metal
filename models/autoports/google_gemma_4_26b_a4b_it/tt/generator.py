# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal-readiness generator for the TP=1/2/4 Gemma-4 26B A4B full model."""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Sequence

import torch
from loguru import logger
from transformers import AutoConfig, AutoTokenizer

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.model import DECODE_SLOT_COUNT, FullModelState, Gemma4FullModel
from models.common.modules.sampling.sampling_1d import Sampling1D
from models.common.modules.sampling.seed_manager_1d import MAX_UINT32

# Device sampler top-k ceiling. The model card's sampling recipe is top_k 64; the TTTv2
# sampler takes its top-k per vocabulary chunk, and 64 keeps the candidate row at two
# tiles. Requests above this ceiling fall back to host sampling in the vLLM plugin.
DEVICE_MAX_TOP_K = 64

# The autonomous readiness package was removed from the source tree after this
# autoport was seeded.  Keep the exact interface when that optional harness is
# on PYTHONPATH, while remaining importable in the current checkout.
try:
    from models.common.readiness_check.contract import Generator, NextInputFn
except ModuleNotFoundError:
    Generator = object
    NextInputFn = Callable[[int, int], int]
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs


@dataclass
class TraceCounters:
    replays: int = 0
    token_refreshes: int = 0
    device_feedback_reuses: int = 0
    device_feedback_restores: int = 0
    position_refreshes: int = 0
    rope_refreshes: int = 0
    page_table_refreshes: int = 0
    synchronizations: int = 0
    token_readbacks: int = 0
    captures: int = 0
    releases: int = 0
    unprepared_captures: int = 0
    sampling_param_refreshes: int = 0
    program_cache_growth_events: int = 0


DRAM_ALLOCATION_ALIGNMENT = 64
TRACE_FENCE_MAX_PLACEMENT_ATTEMPTS = 64


@dataclass
class TraceFence:
    """Persistent DRAM buffers that own the address ranges the decode traces write.

    Trace replay writes its intermediates to the addresses recorded at capture.
    Those addresses are free after capture, so any later allocation (a prefill
    temporary, a host-mode gather, or the kernel binaries of a newly compiled
    program) could land there and be corrupted by the next replay. The fence
    tensors occupy exactly the recorded intermediate ranges for the lifetime of
    the traces, so the allocator can never hand those addresses to anyone else.
    """

    spans: list[tuple[int, int]] = field(default_factory=list)
    tensors: list[ttnn.Tensor] = field(default_factory=list)
    verified: bool = False
    bytes_per_bank: int = 0
    attempts: int = 0
    detail: str = ""


@dataclass(frozen=True)
class SamplingSpec:
    """Semantic sampling configuration used to key captured sampler graphs."""

    greedy: bool
    top_k: tuple[int, ...] = ()
    top_p: tuple[float, ...] = ()
    temperature: tuple[float, ...] = ()
    seeds: tuple[int, ...] = ()

    @property
    def key(self) -> tuple[Any, ...]:
        return (self.greedy, self.top_k, self.top_p, self.temperature, self.seeds)


@dataclass
class DecodeTrace:
    model_trace_id: int | None
    sampling_trace_id: int | None
    token_input: ttnn.Tensor
    current_pos: ttnn.Tensor
    position_ids: ttnn.Tensor
    logits: ttnn.Tensor
    sampled_tokens: ttnn.Tensor
    state: FullModelState
    batch_size: int
    sampling_mode: str
    page_table_ids: tuple[int, ...]
    sampling_spec: SamplingSpec
    sampling_params: tuple[ttnn.Tensor | None, ttnn.Tensor | None, ttnn.Tensor | None, ttnn.Tensor | None]
    seed_skip: ttnn.Tensor | None
    active_mask: torch.Tensor
    prepared: bool = False
    intermediate_spans: list[tuple[int, int]] = field(default_factory=list)
    live_output_spans: list[tuple[int, int]] = field(default_factory=list)


def _padded_prefill_len(logical_len: int) -> int:
    if logical_len < 1:
        raise ValueError("prompt must contain at least one token")
    return max(ttnn.TILE_SIZE, ((logical_len + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE)


TERMINAL_ROW_CHUNK = 2_048


def _contiguous_ranges(rows: Sequence[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for row in rows:
        if ranges and ranges[-1][1] == row:
            ranges[-1] = (ranges[-1][0], row + 1)
        else:
            ranges.append((row, row + 1))
    return ranges


class Gemma4Generator(Generator):
    """Two-level generator with explicit state and split traced sampling.

    The optimized mode is ``sampling_mode='device'``.  ``'host'`` is an
    explicit compatibility mode for readiness checks that require logits.
    """

    model_capabilities = {
        "supports_prefix_caching": True,
        "supports_async_decode": True,
        "supports_mixed_prompt_lens": True,
        "supports_inactive_rows": True,
        "supports_on_device_sampling": True,
    }

    def __init__(self, model: Gemma4FullModel, tokenizer: Any, *, sampling_mode: str = "device") -> None:
        if sampling_mode not in ("device", "host"):
            raise ValueError("sampling_mode must be 'device' or 'host'")
        self.model = model
        self.mesh_device = model.mesh_device
        self.tokenizer = tokenizer
        self.sampling_mode = sampling_mode
        # TTTv2 was selected over the stateful TTTv1 SamplingGenerator: its
        # per-call parameters, native 1D topology, power-of-two local-vocab
        # padding, and direct tt_out_tok contract match this model exactly.
        self.sampler = Sampling1D(
            vocab_size=model.vocab_size,
            mesh_device=self.mesh_device,
            # Sampling always consumes the fixed 32-row terminal tile even
            # when model compute is sliced to a smaller logical batch.
            max_batch_size=DECODE_SLOT_COUNT,
            max_top_k=DEVICE_MAX_TOP_K,
            allow_force_argmax=False,
            pad_to_power_of_2=True,
            num_gather_links=2 if model.tp_size == 4 else 1,
            ag_topology=ttnn.Topology.Ring if model.tp_size == 4 else ttnn.Topology.Linear,
        )
        # Sampling1D owns lazy persistent index/seed buffers.  Materialize
        # them before any model trace pins allocator addresses; first-use
        # initialization between model and sampling trace capture is unsafe.
        self.sampler.load_device_buffers()
        self._trace_cache: dict[tuple[int, str, int, tuple[int, ...]], DecodeTrace] = {}
        self.trace_fence = TraceFence()
        self.trace_counters = TraceCounters()
        self.last_perf: dict[str, float] = {}
        self._request_boundary = True
        self._seeded_slots = torch.zeros(DECODE_SLOT_COUNT, dtype=torch.bool)
        self._sampling_key: tuple[Any, ...] | None = None
        self._borrowed_states: dict[tuple[Any, ...], FullModelState] = {}
        # Retain source objects, rather than only their Python IDs, so an ID
        # cannot be recycled and accidentally suppress a later scheduler copy.
        self._page_table_sources: dict[int, tuple[Any, ...]] = {}

    def _state_from_args(
        self,
        *,
        kv_cache: Any,
        page_table: Any,
        batch_size: int,
        prompt_lens: Sequence[int] | None = None,
    ) -> FullModelState:
        model = self.model
        owns_cache = kv_cache is None
        explicit_state = isinstance(kv_cache, FullModelState)
        if explicit_state:
            state = kv_cache
        elif kv_cache is None:
            state = model.state or model.allocate_state(max_batch_size=batch_size)
        else:
            cache_pairs = [tuple(pair) for pair in kv_cache]
            if len(cache_pairs) != model.num_layers or any(len(pair) != 2 for pair in cache_pairs):
                raise ValueError(f"expected {model.num_layers} caller-owned [K,V] cache pairs")
            if page_table is None:
                raise ValueError("caller-owned KV cache requires an explicit page table")
            page_tables = list(page_table) if isinstance(page_table, (list, tuple)) else [page_table] * model.num_layers
            if len(page_tables) != model.num_layers:
                raise ValueError(f"expected {model.num_layers} per-layer page tables")
            key = (
                batch_size,
                tuple(id(tensor) for pair in cache_pairs for tensor in pair),
                tuple(id(table) for table in page_tables),
            )
            state = self._borrowed_states.get(key)
            if state is None:
                full_capacities = [
                    int(table.shape[1]) * spec.block_size
                    for table, spec in zip(page_tables, model.cache_specs)
                    if spec.layer_type == "full_attention"
                ]
                external_capacity = min([model.max_seq_len, *full_capacities])
                state = FullModelState(
                    kv_cache=cache_pairs,
                    page_tables=page_tables,
                    cache_specs=model.cache_specs,
                    max_batch_size=batch_size,
                    slot_context_lengths=[external_capacity] * batch_size + [0] * (DECODE_SLOT_COUNT - batch_size),
                    prompt_lens=[0] * DECODE_SLOT_COUNT,
                    positions=torch.full((DECODE_SLOT_COUNT,), -1, dtype=torch.int32),
                    active_mask=torch.zeros(DECODE_SLOT_COUNT, dtype=torch.bool),
                )
                self._borrowed_states[key] = state
        if page_table is not None:
            if explicit_state:
                incoming_tables = (
                    list(page_table) if isinstance(page_table, (list, tuple)) else [page_table] * model.num_layers
                )
                if len(incoming_tables) != model.num_layers:
                    raise ValueError(f"expected {model.num_layers} per-layer page tables")
                source_tables = tuple(incoming_tables)
                target_tables = tuple(state.page_tables)
                previous_source_tables = self._page_table_sources.get(id(state), target_tables)
                source_is_target = all(source is target for source, target in zip(source_tables, target_tables))
                source_is_previous = len(source_tables) == len(previous_source_tables) and all(
                    source is previous for source, previous in zip(source_tables, previous_source_tables)
                )
                if not source_is_target and not source_is_previous:
                    for source, target in zip(incoming_tables, state.page_tables):
                        if tuple(source.shape) != tuple(target.shape):
                            raise ValueError(
                                f"page-table shape {tuple(source.shape)} does not match stable tensor "
                                f"{tuple(target.shape)}"
                            )
                        if isinstance(source, ttnn.Tensor):
                            ttnn.copy(input_a=source, input_b=target)
                        else:
                            self._refresh_device_input(target, torch.as_tensor(source, dtype=torch.int32))
                    state.host_page_tables = None
                    self.trace_counters.page_table_refreshes += 1
                self._page_table_sources[id(state)] = source_tables
            elif isinstance(page_table, (list, tuple)):
                if len(page_table) != model.num_layers:
                    raise ValueError(f"expected {model.num_layers} per-layer page tables")
                if any(source is not target for source, target in zip(page_table, state.page_tables)):
                    state.host_page_tables = None
                state.page_tables = list(page_table)
            elif model.num_layers > 0:
                # Internally allocated mixed-geometry state retains its exact
                # per-layer tables when a generic readiness placeholder is
                # supplied. Caller-owned caches may deliberately share one
                # sufficiently wide virtual-to-physical mapping.
                geometries = {(s.block_size, s.blocks_per_slot) for s in state.cache_specs}
                if len(geometries) != 1:
                    if not owns_cache:
                        state.host_page_tables = None
                        state.page_tables = [page_table] * model.num_layers
                        page_table = None
                    # The common readiness runner passes a geometry-agnostic
                    # placeholder table with kv_cache=None. In that case this
                    # generator owns cache allocation and must retain its
                    # matching per-layer page tables.
                    page_table = None
                if page_table is not None:
                    if any(source is not page_table for source in state.page_tables):
                        state.host_page_tables = None
                    state.page_tables = [page_table] * model.num_layers
        if prompt_lens is not None:
            state.prompt_lens[: len(prompt_lens)] = [int(x) for x in prompt_lens]
        return state

    def _host_tokens_to_device(self, tokens: torch.Tensor, *, rank4: bool = False) -> ttnn.Tensor:
        values = tokens.to(torch.int32)
        if rank4:
            flat = values.reshape(-1)
            if flat.numel() > DECODE_SLOT_COUNT:
                raise ValueError(f"decode batch exceeds fixed {DECODE_SLOT_COUNT}-slot contract")
            values = torch.zeros(DECODE_SLOT_COUNT, dtype=torch.int32)
            values[: flat.numel()] = flat
            values = values.reshape(1, 1, 1, DECODE_SLOT_COUNT)
        return ttnn.from_torch(
            values,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _positions_to_device(
        self, positions: torch.Tensor, *, dtype: Any = ttnn.int32, layout: Any = ttnn.ROW_MAJOR_LAYOUT
    ) -> ttnn.Tensor:
        return ttnn.from_torch(
            positions.to(torch.int32),
            dtype=dtype,
            layout=layout,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _refresh_device_input(self, target: ttnn.Tensor, values: torch.Tensor, *, rank4: bool = False) -> None:
        if target.dtype == ttnn.bfloat16:
            values = values.to(torch.bfloat16)
        else:
            values = values.to(torch.int64 if target.dtype == ttnn.uint32 else torch.int32)
        if rank4:
            flat = values.reshape(-1)
            values = torch.zeros(DECODE_SLOT_COUNT, dtype=torch.int32)
            values[: flat.numel()] = flat
            values = values.reshape(1, 1, 1, DECODE_SLOT_COUNT)
        host = ttnn.from_torch(
            values,
            dtype=target.dtype,
            layout=target.layout,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(host, target)

    def refresh_page_tables(self, page_tables: Sequence[torch.Tensor], *, state: FullModelState | None = None) -> None:
        """Copy a changed scheduler mapping into stable traced page-table tensors.

        External callers own the decision that a mapping changed.  Keeping the
        destination tensors stable lets the existing model and sampling traces
        consume the new mapping without recapture or per-token copies. Passing
        identity-distinct tables through prefill/decode similarly snapshots
        them once; subsequent in-place changes to the same source objects must
        call this method explicitly.
        """
        target_state = state or self.model.state
        if target_state is None:
            raise ValueError("page tables require an allocated full-model state")
        if len(page_tables) != len(target_state.page_tables):
            raise ValueError(f"expected {len(target_state.page_tables)} per-layer page tables")
        for source, target in zip(page_tables, target_state.page_tables):
            source_tensor = torch.as_tensor(source, dtype=torch.int32)
            if tuple(source_tensor.shape) != tuple(target.shape):
                raise ValueError(
                    f"page-table shape {tuple(source_tensor.shape)} does not match stable tensor {tuple(target.shape)}"
                )
            self._refresh_device_input(target, source_tensor)
        self.trace_counters.page_table_refreshes += 1
        self._page_table_sources[id(target_state)] = tuple(target_state.page_tables)

    @staticmethod
    def _expand_sampling_value(value: Any, batch_size: int, cast: Any, name: str) -> tuple[Any, ...]:
        values = list(value) if isinstance(value, (list, tuple, torch.Tensor)) else [value] * batch_size
        if len(values) != batch_size:
            raise ValueError(f"{name} must be scalar or have {batch_size} entries")
        return tuple(cast(item) for item in values)

    def _sampling_spec(
        self,
        batch_size: int,
        *,
        top_k: Any = 1,
        top_p: Any = 0.0,
        temperature: Any = 0.0,
        seeds: Any = 0,
    ) -> SamplingSpec:
        temperatures = self._expand_sampling_value(temperature, batch_size, float, "temperature")
        if all(value <= 0.0 for value in temperatures):
            return SamplingSpec(greedy=True)
        if any(value <= 0.0 for value in temperatures):
            raise ValueError("temperature must be positive for every sampled row")
        ks = self._expand_sampling_value(top_k, batch_size, int, "top_k")
        ps = self._expand_sampling_value(top_p, batch_size, float, "top_p")
        seed_values = self._expand_sampling_value(seeds, batch_size, int, "seeds")
        if any(value < 1 or value > DEVICE_MAX_TOP_K for value in ks):
            raise ValueError(f"top_k must be in [1, {DEVICE_MAX_TOP_K}]")
        if any(value < 0.0 or value > 1.0 for value in ps):
            raise ValueError("top_p must be in [0, 1]")
        if any(value < 0 or value >= MAX_UINT32 for value in seed_values):
            raise ValueError(f"seeds must be in [0, {MAX_UINT32 - 1}]")
        return SamplingSpec(False, ks, ps, temperatures, seed_values)

    @staticmethod
    def _seed_values(spec: SamplingSpec) -> torch.Tensor:
        values = torch.zeros(DECODE_SLOT_COUNT, dtype=torch.int64)
        if not spec.greedy:
            values[: len(spec.seeds)] = torch.tensor(spec.seeds, dtype=torch.int64)
        return values

    def _sampling_params(
        self, spec: SamplingSpec
    ) -> tuple[ttnn.Tensor | None, ttnn.Tensor | None, ttnn.Tensor | None, ttnn.Tensor | None]:
        padded_k = torch.ones(DECODE_SLOT_COUNT, dtype=torch.int32)
        padded_p = torch.zeros(DECODE_SLOT_COUNT, dtype=torch.bfloat16)
        padded_temp = torch.ones(DECODE_SLOT_COUNT, dtype=torch.bfloat16)
        padded_seeds = self._seed_values(spec)
        active = DECODE_SLOT_COUNT if spec.greedy else len(spec.top_k)
        if not spec.greedy:
            padded_k[:active] = torch.tensor(spec.top_k, dtype=torch.int32)
            padded_p[:active] = torch.tensor(spec.top_p, dtype=torch.bfloat16)
            padded_temp[:active] = torch.tensor(spec.temperature, dtype=torch.bfloat16)

        def make(values: torch.Tensor, dtype: Any) -> ttnn.Tensor:
            return ttnn.from_torch(
                values,
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, dims=(None, None), mesh_shape=tuple(self.mesh_device.shape)
                ),
            )

        return (
            make(padded_k, ttnn.uint32),
            make(padded_p, ttnn.bfloat16),
            make(padded_temp, ttnn.bfloat16),
            make(padded_seeds, ttnn.uint32),
        )

    def _request_seed_values(self, trace: DecodeTrace, active_mask: torch.Tensor) -> torch.Tensor:
        values = torch.full((DECODE_SLOT_COUNT,), MAX_UINT32, dtype=torch.int64)
        requested = self._seed_values(trace.sampling_spec)
        newly_seeded = active_mask & ~self._seeded_slots
        values[newly_seeded] = requested[newly_seeded]
        return values

    def _read_tokens(self, tokens: ttnn.Tensor, batch_size: int) -> torch.Tensor:
        shard = ttnn.get_device_tensors(tokens)[0] if isinstance(tokens.device(), ttnn.MeshDevice) else tokens
        self.trace_counters.token_readbacks += 1
        return ttnn.to_torch(shard).reshape(-1)[:batch_size].to(torch.long)

    def sample_device_logits(
        self,
        logits: ttnn.Tensor,
        *,
        batch_size: int,
        top_k: Any = 1,
        top_p: Any = 0.0,
        temperature: Any = 0.0,
        seeds: Any = 0,
        tt_out_tok: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Canonical on-device sampler boundary shared by standalone and vLLM.

        This deliberately returns the device token tensor.  Callers that need
        host-visible request state perform only the minimal token read after
        sampling; no caller reconstructs logits or implements another sampler.
        """
        spec = self._sampling_spec(batch_size, top_k=top_k, top_p=top_p, temperature=temperature, seeds=seeds)
        with self._track_program_cache_growth("prefill_sampling"):
            sampling_logits = self._pad_sampling_logits(logits)
            k, p, temp, seed_tensor = self._sampling_params(spec)
            sampled, _ = self.sampler.decode_forward(
                sampling_logits,
                k=k,
                p=p,
                temp=temp,
                seeds=seed_tensor,
                tt_out_tok=tt_out_tok,
            )
            for tensor in (k, p, temp, seed_tensor):
                ttnn.deallocate(tensor)
            if sampling_logits is not logits:
                ttnn.deallocate(sampling_logits)
        return sampled

    @staticmethod
    def _pad_sampling_logits(logits: ttnn.Tensor) -> ttnn.Tensor:
        if logits.shape[-2] == DECODE_SLOT_COUNT:
            return logits
        return ttnn.pad(
            logits,
            padding=[(0, 0), (0, 0), (0, DECODE_SLOT_COUNT - logits.shape[-2]), (0, 0)],
            value=0.0,
        )

    def _gather_logits_to_torch(self, logits: ttnn.Tensor, *, logical_len: int | None = None) -> torch.Tensor:
        """Explicit host-logits compatibility boundary for readiness checks."""
        gathered = logits
        if self.model.tp_size > 1:
            gathered = ttnn.all_gather(
                logits,
                dim=3,
                cluster_axis=1,
                num_links=2 if self.model.tp_size == 4 else 1,
                topology=ttnn.Topology.Ring if self.model.tp_size == 4 else ttnn.Topology.Linear,
            )
        shard = ttnn.get_device_tensors(gathered)[0]
        host = ttnn.to_torch(shard)
        if host.ndim == 4 and host.shape[1] == 1:
            host = host[:, 0]
        if logical_len is not None:
            host = host[:, :logical_len]
        return host

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table: Any,
        kv_cache: Any,
        prompt_lens: List[int],
        start_pos: torch.Tensor | None = None,
        chunk_page_tables: Sequence[ttnn.Tensor | None] | None = None,
        host_page_tables: Sequence[torch.Tensor] | None = None,
        return_all_logits: bool = False,
        logit_positions: Sequence[int] | None = None,
        return_hidden: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | ttnn.Tensor:
        del kwargs
        if (logit_positions is not None or return_hidden) and len(prompt_lens) != 1:
            raise ValueError("logit_positions and return_hidden select rows of one logical prompt")
        if tokens.ndim != 2 or tokens.shape[0] != len(prompt_lens):
            raise ValueError("tokens must be [batch,padded_prompt] with one prompt_lens entry per row")
        if any(n < 1 or n > self.model.max_seq_len for n in prompt_lens):
            raise ValueError(f"prompt lengths must be in [1, {self.model.max_seq_len}]")
        state = self._state_from_args(
            kv_cache=kv_cache,
            page_table=page_table,
            batch_size=tokens.shape[0],
            prompt_lens=prompt_lens,
        )
        state.active_mask.zero_()
        self._request_boundary = True
        self._seeded_slots.zero_()
        self._sampling_key = None
        state.active_mask[: len(prompt_lens)] = True
        state.positions.fill_(-1)
        if start_pos is None:
            position_rows = [torch.arange(n, dtype=torch.int32) for n in prompt_lens]
        else:
            positions = torch.as_tensor(start_pos, dtype=torch.int32)
            if positions.ndim == 1:
                positions = positions.reshape(len(prompt_lens), -1)
            position_rows = [positions[row, :n].contiguous() for row, n in enumerate(prompt_lens)]
        if host_page_tables is not None:
            if len(host_page_tables) != len(state.page_tables):
                raise ValueError("host_page_tables must contain the current metadata for every layer")
            state.host_page_tables = list(host_page_tables)
        elif any(int(row[0]) > 0 for row in position_rows):
            # Direct callers can mutate the same device table in place. Read
            # current TABLE metadata for every continuation unless the caller
            # supplies an authoritative host snapshot. K/V stays on device.
            state.host_page_tables = [
                (
                    ttnn.to_torch(ttnn.get_device_tensors(table)[0]).to(torch.int32).clone()
                    if isinstance(table, ttnn.Tensor)
                    else torch.as_tensor(table, dtype=torch.int32).clone()
                )
                for table in state.page_tables
            ]
        state.positions[: len(prompt_lens)] = torch.tensor(
            [int(row[-1]) + 1 for row in position_rows], dtype=torch.int32
        )
        for row, logical_len in enumerate(prompt_lens):
            if logical_len > state.slot_context_lengths[row]:
                raise ValueError(
                    f"prompt row {row} length {logical_len} exceeds allocated slot capacity "
                    f"{state.slot_context_lengths[row]}"
                )

        def run_model_prefill(*args: Any, **model_kwargs: Any) -> ttnn.Tensor:
            # Decode traces survive prefill. Prefill temporaries are released
            # before the next replay and the trace fence owns every address a
            # replay writes, so allocations made here (including the kernel
            # binaries of a newly compiled prefill program) cannot overlap a
            # trace. The scope only tells the temporal allocation tracker that;
            # program-cache growth is still counted and, without a verified
            # fence, releases the traces before the next replay.
            with self._track_program_cache_growth("prefill"), ttnn.corruptible_allocation_scope(self.mesh_device):
                return self.model.prefill_forward(*args, **model_kwargs)

        if len(prompt_lens) > 1:
            # The decoder prefill kernel is single-user: batching equal-length
            # prompts would place the user count in the kernel's sequence axis.
            # Execute every request through that canonical path; each logical
            # row still owns its cache/page-table row and outputs stay ordered.
            outputs = []
            for row, logical_len in enumerate(prompt_lens):
                physical_len = _padded_prefill_len(logical_len)
                row_tokens = tokens[row : row + 1, :physical_len]
                if row_tokens.shape[1] < physical_len:
                    row_tokens = torch.nn.functional.pad(row_tokens, (0, physical_len - row_tokens.shape[1]))
                tt_tokens = self._host_tokens_to_device(row_tokens)
                first_position = int(position_rows[row][0])
                physical_positions = torch.arange(
                    first_position, first_position + physical_len, dtype=torch.int32
                ).reshape(1, physical_len)
                tt_pos = self._positions_to_device(physical_positions, dtype=ttnn.uint32)
                outputs.append(
                    run_model_prefill(
                        tt_tokens,
                        state=state,
                        prompt_lens=[logical_len],
                        position_ids=tt_pos,
                        prefill_start=first_position,
                        user_id=row,
                        chunk_page_tables=chunk_page_tables,
                        return_hidden=return_all_logits,
                    )
                )
            if return_all_logits:
                max_logical_len = max(prompt_lens)
                logical_rows = []
                for output, logical_len in zip(outputs, prompt_lens):
                    row = self._hidden_rows_to_host_logits(output, range(logical_len))
                    logical_rows.append(torch.nn.functional.pad(row, (0, 0, 0, max_logical_len - logical_len)))
                return torch.cat(logical_rows, dim=0)
            return ttnn.concat(outputs, dim=2)

        logical_len = int(prompt_lens[0])
        physical_len = _padded_prefill_len(logical_len)
        physical_tokens = tokens[:, :physical_len]
        if physical_tokens.shape[1] < physical_len:
            physical_tokens = torch.nn.functional.pad(physical_tokens, (0, physical_len - physical_tokens.shape[1]))
        tt_tokens = self._host_tokens_to_device(physical_tokens)
        positions = torch.stack(
            [torch.arange(int(row[0]), int(row[0]) + physical_len, dtype=torch.int32) for row in position_rows]
        )
        tt_pos = self._positions_to_device(positions, dtype=ttnn.uint32)
        select_rows = return_all_logits or logit_positions is not None or return_hidden
        logits = run_model_prefill(
            tt_tokens,
            state=state,
            prompt_lens=prompt_lens,
            position_ids=tt_pos,
            prefill_start=int(position_rows[0][0]),
            chunk_page_tables=chunk_page_tables,
            return_hidden=select_rows,
        )
        if return_hidden:
            return logits
        if logit_positions is not None:
            rows = [int(row) for row in logit_positions]
            if any(row < 0 or row >= logical_len for row in rows):
                raise ValueError(f"logit_positions must lie in [0, {logical_len})")
            return self._hidden_rows_to_host_logits(logits, rows)
        if return_all_logits:
            return self._hidden_rows_to_host_logits(logits, range(logical_len))
        return logits

    def _hidden_rows_to_host_logits(self, hidden: ttnn.Tensor, rows: Sequence[int]) -> torch.Tensor:
        """Terminal norm/LM head over selected hidden rows, read back as [1, rows, vocab].

        The LM head over every row of a long prompt does not fit device DRAM
        (16K rows x 262,144 vocab), so rows are selected first and the terminal
        runs over bounded row chunks.
        """
        rows = list(rows)
        outputs = []
        for start, end in _contiguous_ranges(rows):
            for chunk_start in range(start, end, TERMINAL_ROW_CHUNK):
                chunk_end = min(end, chunk_start + TERMINAL_ROW_CHUNK)
                if chunk_start == 0 and chunk_end == hidden.shape[-2]:
                    selected = hidden
                else:
                    selected = ttnn.slice(
                        hidden,
                        (0, 0, chunk_start, 0),
                        (hidden.shape[0], hidden.shape[1], chunk_end, hidden.shape[3]),
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                logits = self.model.terminal_logits(selected)
                outputs.append(self._gather_logits_to_torch(logits, logical_len=chunk_end - chunk_start))
                ttnn.deallocate(logits)
                if selected is not hidden:
                    ttnn.deallocate(selected)
        ttnn.deallocate(hidden)
        return torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]

    def prefill_logits(self, prompt_token_ids: List[int]) -> torch.Tensor:
        """High-level logical-length prefill used by the readiness harness."""
        logical_len = len(prompt_token_ids)
        if logical_len < 1:
            raise ValueError("prompt must contain at least one token")
        state = self.model.state or self.model.allocate_state(max_batch_size=1)
        if logical_len > state.slot_context_lengths[0]:
            raise ValueError(f"prompt length {logical_len} exceeds slot capacity {state.slot_context_lengths[0]}")
        physical_len = _padded_prefill_len(logical_len)
        tokens = torch.tensor(prompt_token_ids, dtype=torch.long).reshape(1, logical_len)
        tokens = torch.nn.functional.pad(tokens, (0, physical_len - logical_len))
        return self.prefill_forward(
            tokens,
            page_table=state.page_tables,
            kv_cache=state,
            prompt_lens=[logical_len],
            return_all_logits=True,
        )

    def prefill_logits_continuation(
        self,
        prompt_token_ids: List[int],
        start: int,
        *,
        positions: Optional[Sequence[int]] = None,
        prefill_prefix: bool = True,
    ) -> torch.Tensor:
        """Resumed-prefill driver: prefix as a fresh request, suffix at absolute ``start``."""
        length = len(prompt_token_ids)
        if length < 1:
            raise ValueError("prompt must contain at least one token")
        if start < 0 or start >= length:
            raise ValueError(f"start must be in [0, {length}), got {start}")
        self.reset()
        state = self.model.state or self.model.allocate_state(max_batch_size=1)
        if length > state.slot_context_lengths[0]:
            raise ValueError(f"prompt length {length} exceeds slot capacity {state.slot_context_lengths[0]}")
        tokens = torch.tensor(prompt_token_ids, dtype=torch.long).reshape(1, length)
        if start > 0 and prefill_prefix:
            prefix = torch.nn.functional.pad(tokens[:, :start], (0, _padded_prefill_len(start) - start))
            prefix_logits = self.prefill_forward(
                prefix, page_table=state.page_tables, kv_cache=state, prompt_lens=[start]
            )
            ttnn.deallocate(prefix_logits)
        suffix_len = length - start
        suffix = torch.nn.functional.pad(tokens[:, start:], (0, _padded_prefill_len(suffix_len) - suffix_len))
        absolute = list(range(start, length)) if positions is None else [int(p) for p in positions]
        if any(p < start or p >= length for p in absolute):
            raise ValueError(f"positions must lie in [{start}, {length})")
        return self.prefill_forward(
            suffix,
            page_table=state.page_tables,
            kv_cache=state,
            prompt_lens=[suffix_len],
            start_pos=torch.arange(start, length, dtype=torch.int32).reshape(1, suffix_len),
            logit_positions=[p - start for p in absolute],
        )

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table: Any,
        kv_cache: Any,
        sampling_mode: str | None = None,
        enable_trace: bool = True,
        active_mask: torch.Tensor | None = None,
        top_k: Any = 1,
        top_p: Any = 0.0,
        temperature: Any = 0.0,
        seeds: Any = 0,
        **kwargs: Any,
    ) -> torch.Tensor | ttnn.Tensor:
        del kwargs
        if tokens.ndim != 2 or tokens.shape[1] != 1:
            raise ValueError("decode tokens must have shape [batch, 1]")
        batch = tokens.shape[0]
        if batch < 1 or batch > DECODE_SLOT_COUNT:
            raise ValueError(f"decode batch must be in [1, {DECODE_SLOT_COUNT}]")
        supplied_positions = torch.as_tensor(start_pos, dtype=torch.int32).reshape(-1)
        if supplied_positions.numel() != batch:
            raise ValueError(f"start_pos must contain {batch} entries")
        mode = sampling_mode or self.sampling_mode
        if mode not in ("device", "host", "teacher"):
            raise ValueError("sampling_mode must be 'device', 'host', or 'teacher'")
        if mode == "host":
            # Compatibility mode intentionally gathers logits on the host.
            # Keep it eager so those transient gather buffers can never
            # invalidate the stable-address optimized decode trace.
            enable_trace = False
        state = self._state_from_args(kv_cache=kv_cache, page_table=page_table, batch_size=batch)
        previous_mask = state.active_mask.clone()
        fresh_state = not bool(state.active_mask.any()) and not bool((state.positions >= 0).any())
        if active_mask is not None:
            normalized_mask = torch.as_tensor(active_mask, dtype=torch.bool).reshape(-1)
            if normalized_mask.numel() != batch:
                raise ValueError(f"active_mask must contain {batch} entries")
            state.active_mask.zero_()
            state.active_mask[:batch] = normalized_mask
        elif (self._request_boundary or fresh_state) and not bool(state.active_mask[:batch].any()):
            # Raw low-level callers do not have a separate admission method;
            # nonnegative positions identify active rows on their first call.
            state.active_mask[:batch] = supplied_positions >= 0

        scheduler_boundary = (
            self._request_boundary
            or fresh_state
            or active_mask is not None
            or not torch.equal(previous_mask, state.active_mask)
        )
        if scheduler_boundary or not enable_trace:
            state.positions.fill_(-1)
            state.positions[:batch] = torch.where(
                state.active_mask[:batch], supplied_positions, torch.full((batch,), -1, dtype=torch.int32)
            )
            newly_active = state.active_mask & ~previous_mask
            self._seeded_slots[newly_active] = False
            self._seeded_slots[~state.active_mask] = False

        active_rows = state.active_mask[:batch]
        current_positions = state.positions[:batch].clone()
        for row in torch.nonzero(active_rows, as_tuple=False).reshape(-1).tolist():
            position = int(current_positions[row])
            capacity = int(state.slot_context_lengths[row])
            if position < 0 or position >= capacity:
                raise ValueError(f"decode position {position} for row {row} is outside slot capacity {capacity}")
        spec = self._sampling_spec(batch, top_k=top_k, top_p=top_p, temperature=temperature, seeds=seeds)
        if mode in ("host", "teacher"):
            spec = SamplingSpec(greedy=True)
        sampling_changed = self._sampling_key is not None and self._sampling_key != spec.key
        if sampling_changed:
            self._seeded_slots[state.active_mask] = False
        if enable_trace:
            key = self._decode_trace_key(batch, mode, state)
            trace_was_cached = key in self._trace_cache
            # Sampling parameters are runtime tensor inputs of the captured
            # sampler, so a parameter change refreshes those tensors instead
            # of recapturing. Explicit state page-table replacements were
            # normalized into stable tensors above. During steady device
            # decode, ``tokens`` may intentionally be stale because the prior
            # sampling trace owns feedback. Keep that device tensor alive
            # across any cache miss and reuse it as the new stable input;
            # never fall back to the host argument.
            device_feedback_input = None
            if not trace_was_cached and not scheduler_boundary and mode == "device":
                device_feedback_input = next(
                    (
                        cached.token_input
                        for cached in self._trace_cache.values()
                        if cached.state is state and cached.batch_size == batch and cached.sampling_mode == "device"
                    ),
                    None,
                )
            trace = self._get_or_capture_decode_trace(
                tokens,
                current_positions,
                state=state,
                sampling_mode=mode,
                sampling_spec=spec,
                token_input_override=device_feedback_input,
            )
            if device_feedback_input is not None:
                self.trace_counters.device_feedback_reuses += 1
            if trace.sampling_trace_id is not None and trace.sampling_spec != spec:
                self._refresh_sampling_params(trace, spec)
            if mode == "teacher" or scheduler_boundary:
                # Teacher forcing deliberately replaces device feedback, but
                # still executes the full decode graph through its trace.
                self._refresh_device_input(trace.token_input, tokens, rank4=True)
                self.trace_counters.token_refreshes += 1
            if scheduler_boundary:
                padded_positions = torch.full((DECODE_SLOT_COUNT,), -1, dtype=torch.int32)
                padded_positions[:batch] = current_positions
                self._refresh_device_input(trace.current_pos, padded_positions)
                self._refresh_device_input(trace.position_ids, padded_positions.clamp_min(0))
                self.trace_counters.position_refreshes += 1
                self.trace_counters.rope_refreshes += 1
                trace.active_mask = state.active_mask.clone()
            if trace.sampling_trace_id is not None and (scheduler_boundary or sampling_changed or not trace_was_cached):
                self._refresh_device_input(
                    trace.sampling_params[3], self._request_seed_values(trace, state.active_mask)
                )
            ttnn.execute_trace(self.mesh_device, trace.model_trace_id, cq_id=0, blocking=False)
            if trace.sampling_trace_id is not None:
                ttnn.execute_trace(self.mesh_device, trace.sampling_trace_id, cq_id=0, blocking=False)
                self._seeded_slots[state.active_mask] = True
                self._sampling_key = spec.key
            self.trace_counters.replays += 1
            state.positions[state.active_mask] += 1
            self._request_boundary = False
            return trace.sampled_tokens if mode in ("device", "teacher") else trace.logits

        tt_tokens = self._host_tokens_to_device(tokens, rank4=True)
        tt_current = self._positions_to_device(current_positions)
        tt_position_ids = self._positions_to_device(current_positions.clamp_min(0), dtype=ttnn.uint32)
        logits = self.model.decode_forward(
            tt_tokens, state=state, current_pos=tt_current, position_ids=tt_position_ids, batch_size=batch
        )
        state.positions[state.active_mask] += 1
        self._request_boundary = False
        if mode == "host":
            return logits
        logits = self._pad_sampling_logits(logits)
        k, p, temp, seed_tensor = self._sampling_params(spec)
        sampled = self.sampler.decode_forward(logits, k=k, p=p, temp=temp, seeds=seed_tensor)[0]
        self._seeded_slots[state.active_mask] = True
        self._sampling_key = spec.key
        return sampled

    @staticmethod
    def _decode_trace_key(
        batch: int, sampling_mode: str, state: FullModelState
    ) -> tuple[int, str, int, tuple[int, ...]]:
        return (batch, sampling_mode, id(state), tuple(id(table) for table in state.page_tables))

    def _log_trace_event(self, kind: str, **fields: Any) -> None:
        payload = {"event": kind, **fields}
        logger.info("GEMMA4_TRACE_EVENT " + json.dumps(payload, sort_keys=True, default=str))

    @contextmanager
    def _track_program_cache_growth(self, phase: str) -> Iterator[None]:
        if not self._trace_cache:
            yield
            return
        before = self.mesh_device.num_program_cache_entries()
        yield
        after = self.mesh_device.num_program_cache_entries()
        if after != before:
            self.note_program_cache_growth(before, after, phase=phase)

    def note_program_cache_growth(self, before: int, after: int, *, phase: str) -> bool:
        """Programs compiled after capture allocate kernel binaries in DRAM.

        With a verified fence those binaries cannot overlap a trace, so the
        traces stay valid. Without one the only safe action is to release the
        traces before the next replay, which the counters make visible.
        """
        self.trace_counters.program_cache_growth_events += 1
        fenced = self.trace_fence.verified
        self._log_trace_event(
            "program_cache_growth",
            phase=phase,
            entries_before=before,
            entries_after=after,
            fence_verified=fenced,
            traces_retained=fenced,
        )
        if not fenced:
            self.release_decode_traces(reason=f"program_cache_growth_unfenced:{phase}")
        return fenced

    def release_decode_traces(self, *, reason: str) -> None:
        if not self._trace_cache and not self.trace_fence.tensors:
            return
        for key, trace in list(self._trace_cache.items()):
            if trace.model_trace_id is not None:
                ttnn.release_trace(self.mesh_device, trace.model_trace_id)
            if trace.sampling_trace_id is not None:
                ttnn.release_trace(self.mesh_device, trace.sampling_trace_id)
            self.trace_counters.releases += 1
            self._log_trace_event(
                "release",
                reason=reason,
                batch=trace.batch_size,
                mode=trace.sampling_mode,
                model_trace_id=trace.model_trace_id,
                sampling_trace_id=trace.sampling_trace_id,
                prepared=trace.prepared,
            )
        self._trace_cache.clear()
        for tensor in self.trace_fence.tensors:
            ttnn.deallocate(tensor)
        self.trace_fence = TraceFence()

    def _refresh_sampling_params(self, trace: DecodeTrace, spec: SamplingSpec) -> None:
        k, p, temp, _ = trace.sampling_params
        padded_k = torch.ones(DECODE_SLOT_COUNT, dtype=torch.int32)
        padded_p = torch.zeros(DECODE_SLOT_COUNT, dtype=torch.float32)
        padded_temp = torch.ones(DECODE_SLOT_COUNT, dtype=torch.float32)
        if not spec.greedy:
            active = len(spec.top_k)
            padded_k[:active] = torch.tensor(spec.top_k, dtype=torch.int32)
            padded_p[:active] = torch.tensor(spec.top_p, dtype=torch.float32)
            padded_temp[:active] = torch.tensor(spec.temperature, dtype=torch.float32)
        self._refresh_device_input(k, padded_k)
        self._refresh_device_input(p, padded_p)
        self._refresh_device_input(temp, padded_temp)
        trace.sampling_spec = spec
        self.trace_counters.sampling_param_refreshes += 1

    def _allocate_decode_variant(
        self,
        batch: int,
        sampling_mode: str,
        state: FullModelState,
        sampling_spec: SamplingSpec,
        *,
        token_input: ttnn.Tensor | None = None,
    ) -> DecodeTrace:
        """Allocate every persistent tensor of one decode variant before any capture."""
        if token_input is None:
            token_input = self._host_tokens_to_device(torch.zeros((batch, 1), dtype=torch.long), rank4=True)
        capture_positions = torch.full((DECODE_SLOT_COUNT,), -1, dtype=torch.int32)
        current_pos = self._positions_to_device(capture_positions)
        position_ids = self._positions_to_device(capture_positions.clamp_min(0), dtype=ttnn.uint32)
        sampled_mode = sampling_mode in ("device", "teacher")
        params = self._sampling_params(sampling_spec) if sampled_mode else (None, None, None, None)
        seed_skip = (
            ttnn.from_torch(
                torch.full((DECODE_SLOT_COUNT,), MAX_UINT32, dtype=torch.int64),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, dims=(None, None), mesh_shape=tuple(self.mesh_device.shape)
                ),
            )
            if sampled_mode
            else None
        )
        return DecodeTrace(
            model_trace_id=None,
            sampling_trace_id=None,
            token_input=token_input,
            current_pos=current_pos,
            position_ids=position_ids,
            logits=token_input,
            sampled_tokens=token_input,
            state=state,
            batch_size=batch,
            sampling_mode=sampling_mode,
            page_table_ids=tuple(id(table) for table in state.page_tables),
            sampling_spec=sampling_spec,
            sampling_params=params,
            seed_skip=seed_skip,
            active_mask=state.active_mask.clone(),
        )

    def _warm_decode_variant(self, trace: DecodeTrace) -> None:
        """Compile the exact inactive model, sampler, seed-arm, and position graph."""
        batch = trace.batch_size
        capture_positions = torch.full((DECODE_SLOT_COUNT,), -1, dtype=torch.int32)
        self._refresh_device_input(trace.current_pos, capture_positions)
        self._refresh_device_input(trace.position_ids, capture_positions.clamp_min(0))
        logits = self.model.decode_forward(
            trace.token_input,
            state=trace.state,
            current_pos=trace.current_pos,
            position_ids=trace.position_ids,
            batch_size=batch,
        )
        if trace.sampling_trace_id is None and trace.sampling_mode in ("device", "teacher"):
            k, p, temp, seed_tensor = trace.sampling_params
            sampling_logits = self._pad_sampling_logits(logits)
            # Compile the exact optional-output graph that capture will use.
            # Warming the allocator-output variant is insufficient: optional
            # output changes the program hash and would upload binaries during
            # capture, which the mesh command queue correctly rejects as a write.
            warm_sampled_tokens = self._host_tokens_to_device(torch.zeros((batch, 1), dtype=torch.long), rank4=True)
            _, _ = self.sampler.decode_forward(
                sampling_logits, k=k, p=p, temp=temp, seeds=seed_tensor, tt_out_tok=warm_sampled_tokens
            )
            ttnn.copy(input_a=trace.seed_skip, input_b=seed_tensor)
            ttnn.deallocate(warm_sampled_tokens)
            if sampling_logits is not logits:
                ttnn.deallocate(sampling_logits)
        ttnn.plus_one(trace.current_pos, skip_negative_entries=True)
        ttnn.plus_one(trace.position_ids, skip_negative_entries=True)
        ttnn.synchronize_device(self.mesh_device)
        ttnn.deallocate(logits)
        self._refresh_device_input(trace.current_pos, capture_positions)
        self._refresh_device_input(trace.position_ids, capture_positions.clamp_min(0))

    @staticmethod
    def _dram_spans_from_graph(
        nodes: Sequence[dict[str, Any]], device_id: int | None
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Return (intermediate spans, live output spans) of DRAM buffers allocated under capture.

        Spans are per-bank address intervals ``[address, address + aligned_size_per_bank)``.
        A buffer deallocated before capture ended is a trace intermediate; one still
        allocated is a trace output.
        """
        allocated: dict[int, tuple[int, int]] = {}
        intermediates: list[tuple[int, int]] = []
        for node in nodes:
            kind = node.get("node_type")
            if kind not in ("buffer_allocate", "buffer_deallocate"):
                continue
            params = node.get("params") or {}
            if params.get("type") != "DRAM":
                continue
            if device_id is not None and int(params.get("device_id", device_id)) != device_id:
                continue
            address = int(params["address"])
            size = int(params.get("max_size_per_bank") or 0)
            if size <= 0:
                continue
            if kind == "buffer_allocate":
                allocated[address] = (address, address + size)
            else:
                span = allocated.pop(address, (address, address + size))
                intermediates.append(span)
        return intermediates, list(allocated.values())

    @staticmethod
    def _merge_spans(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        for start, end in sorted(spans):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    def _capture_decode_variant(self, trace: DecodeTrace) -> None:
        """Capture the model and sampling traces of a prepared variant and record their footprint."""
        batch = trace.batch_size
        capture_positions = torch.full((DECODE_SLOT_COUNT,), -1, dtype=torch.int32)
        self._refresh_device_input(trace.current_pos, capture_positions)
        self._refresh_device_input(trace.position_ids, capture_positions.clamp_min(0))
        first_device = ttnn.get_device_ids(self.mesh_device)[0] if hasattr(ttnn, "get_device_ids") else None
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        try:
            model_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            logits = self.model.decode_forward(
                trace.token_input,
                state=trace.state,
                current_pos=trace.current_pos,
                position_ids=trace.position_ids,
                batch_size=batch,
            )
            ttnn.plus_one(trace.current_pos, skip_negative_entries=True)
            ttnn.plus_one(trace.position_ids, skip_negative_entries=True)
            ttnn.end_trace_capture(self.mesh_device, model_trace_id, cq_id=0)
        finally:
            model_nodes = ttnn.graph.end_graph_capture()
        intermediates, live = self._dram_spans_from_graph(model_nodes, first_device)
        sampling_trace_id = None
        sampled_tokens = trace.token_input
        sampled_mode = trace.sampling_mode in ("device", "teacher")
        if sampled_mode:
            k, p, temp, seed_tensor = trace.sampling_params
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            try:
                sampling_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
                sampling_logits = self._pad_sampling_logits(logits)
                sampled_tokens, _ = self.sampler.decode_forward(
                    sampling_logits, k=k, p=p, temp=temp, seeds=seed_tensor, tt_out_tok=trace.token_input
                )
                ttnn.copy(input_a=trace.seed_skip, input_b=seed_tensor)
                ttnn.end_trace_capture(self.mesh_device, sampling_trace_id, cq_id=0)
            finally:
                sampling_nodes = ttnn.graph.end_graph_capture()
            sampling_intermediates, sampling_live = self._dram_spans_from_graph(sampling_nodes, first_device)
            intermediates += sampling_intermediates
            live += sampling_live
            if sampling_logits is not logits:
                ttnn.mark_corruptible(sampling_logits)
        # The captured outputs are the only allocations that survive capture.
        # They are rewritten by every replay and consumed by the sampling
        # trace that follows immediately, before any other trace can run.
        ttnn.mark_corruptible(logits)
        trace.model_trace_id = model_trace_id
        trace.sampling_trace_id = sampling_trace_id
        trace.logits = logits
        trace.sampled_tokens = sampled_tokens
        trace.intermediate_spans = self._merge_spans(intermediates)
        trace.live_output_spans = self._merge_spans(live)
        trace.active_mask = trace.state.active_mask.clone()
        self.trace_counters.captures += 1
        self._log_trace_event(
            "capture",
            batch=batch,
            mode=trace.sampling_mode,
            model_trace_id=model_trace_id,
            sampling_trace_id=sampling_trace_id,
            prepared=trace.prepared,
            token_input=trace.token_input.buffer_address(),
            logits=logits.buffer_address(),
            intermediate_spans=trace.intermediate_spans,
            intermediate_bytes_per_bank=sum(end - start for start, end in trace.intermediate_spans),
            live_output_spans=trace.live_output_spans,
        )

    def _install_trace_fence(self) -> TraceFence:
        """Pin every recorded trace intermediate range with a persistent buffer."""
        fence = TraceFence()
        live = self._merge_spans([span for trace in self._trace_cache.values() for span in trace.live_output_spans])
        wanted: list[tuple[int, int]] = []
        for start, end in self._merge_spans(
            [span for trace in self._trace_cache.values() for span in trace.intermediate_spans]
        ):
            cursor = start
            for live_start, live_end in live:
                if live_end <= cursor or live_start >= end:
                    continue
                if live_start > cursor:
                    wanted.append((cursor, live_start))
                cursor = max(cursor, live_end)
            if cursor < end:
                wanted.append((cursor, end))
        try:
            num_banks = int(ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.DRAM).num_banks)
        except Exception as error:  # pragma: no cover - depends on the ttnn build
            fence.detail = f"memory view unavailable: {error}"
            self.trace_fence = fence
            self._log_trace_event("fence", verified=False, detail=fence.detail)
            return fence
        page_bytes = DRAM_ALLOCATION_ALIGNMENT
        padding: list[ttnn.Tensor] = []
        try:
            for start, end in wanted:
                length = ((end - start) // page_bytes) * page_bytes
                if length <= 0:
                    continue
                pages = (length // page_bytes) * num_banks
                shape = ttnn.Shape([pages, page_bytes // 4])
                placed = None
                for _ in range(TRACE_FENCE_MAX_PLACEMENT_ATTEMPTS):
                    candidate = ttnn.allocate_tensor_on_device(
                        shape, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.mesh_device, ttnn.DRAM_MEMORY_CONFIG
                    )
                    fence.attempts += 1
                    address = candidate.buffer_address()
                    per_bank = candidate.buffer_aligned_page_size() * math.ceil(
                        candidate.buffer_num_pages() / num_banks
                    )
                    if address == start and per_bank == length:
                        placed = candidate
                        break
                    if address > start:
                        ttnn.deallocate(candidate)
                        fence.detail = f"span {start}:{end} skipped: allocator placed {per_bank} bytes at {address}"
                        break
                    padding.append(candidate)
                if placed is None:
                    fence.detail = fence.detail or f"span {start}:{end} could not be placed"
                    break
                fence.tensors.append(placed)
                fence.spans.append((start, start + length))
                fence.bytes_per_bank += length
            else:
                fence.verified = bool(wanted)
        finally:
            for tensor in padding:
                ttnn.deallocate(tensor)
        if not fence.verified:
            for tensor in fence.tensors:
                ttnn.deallocate(tensor)
            fence.tensors = []
        self.trace_fence = fence
        self._log_trace_event(
            "fence",
            verified=fence.verified,
            spans=fence.spans,
            bytes_per_bank=fence.bytes_per_bank,
            attempts=fence.attempts,
            detail=fence.detail,
            live_output_spans=live,
        )
        return fence

    def prepare_decode_traces(
        self,
        state: FullModelState,
        *,
        batch_sizes: Sequence[int],
        sampling_mode: str = "device",
    ) -> dict[tuple[int, str, int, tuple[int, ...]], DecodeTrace]:
        """Generator-owned two-phase preparation of every serving decode variant.

        Phase one allocates the persistent inputs of every variant and compiles
        its programs. Phase two captures each variant; nothing is released.
        The fence is installed last so later allocations cannot reach any
        recorded intermediate range. Repeated calls reuse prepared variants.
        """
        wanted = []
        for batch in sorted({int(b) for b in batch_sizes}):
            key = self._decode_trace_key(batch, sampling_mode, state)
            if key not in self._trace_cache:
                wanted.append((key, batch))
        if not wanted:
            return self._trace_cache
        if self._trace_cache:
            # A variant added after another capture allocates after that
            # capture. Start over so every persistent input precedes every trace.
            self.release_decode_traces(reason="prepare_additional_variant")
            wanted = [
                (self._decode_trace_key(batch, sampling_mode, state), batch)
                for batch in sorted({int(b) for b in batch_sizes})
            ]
        began = time.perf_counter()
        variants = []
        for key, batch in wanted:
            variant = self._allocate_decode_variant(batch, sampling_mode, state, SamplingSpec(greedy=True))
            variant.prepared = True
            variants.append((key, variant))
        for _, variant in variants:
            self._warm_decode_variant(variant)
        warmed = time.perf_counter()
        for key, variant in variants:
            self._capture_decode_variant(variant)
            self._trace_cache[key] = variant
        self._install_trace_fence()
        self._log_trace_event(
            "prepare",
            batch_sizes=[batch for _, batch in wanted],
            mode=sampling_mode,
            warm_s=round(warmed - began, 3),
            capture_s=round(time.perf_counter() - warmed, 3),
            program_cache_entries=self.mesh_device.num_program_cache_entries(),
            fence_verified=self.trace_fence.verified,
        )
        return self._trace_cache

    def _get_or_capture_decode_trace(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        state: FullModelState,
        sampling_mode: str,
        sampling_spec: SamplingSpec,
        token_input_override: ttnn.Tensor | None = None,
    ) -> DecodeTrace:
        key = self._decode_trace_key(tokens.shape[0], sampling_mode, state)
        trace = self._trace_cache.get(key)
        if trace is not None:
            return trace
        # Unprepared variant: standalone generation and readiness modes reach
        # this path for their single variant. Allocating a new variant after
        # other captures would be unsafe, so release first; the counters make
        # any such capture during serving visible.
        if self._trace_cache:
            self.trace_counters.unprepared_captures += 1
            self.release_decode_traces(reason="unprepared_variant")
        batch = tokens.shape[0]
        feedback_backup = None
        if token_input_override is not None:
            # Trace capture executes the sampling graph and tt_out_tok aliases
            # token_input. Save the prior device token before capture and
            # restore it device-to-device afterward; no host token enters the
            # same-request transition.
            feedback_backup = self._host_tokens_to_device(torch.zeros_like(tokens), rank4=True)
            ttnn.copy(input_a=token_input_override, input_b=feedback_backup)
        trace = self._allocate_decode_variant(
            batch,
            sampling_mode,
            state,
            sampling_spec,
            token_input=token_input_override
            if token_input_override is not None
            else self._host_tokens_to_device(tokens, rank4=True),
        )
        self._warm_decode_variant(trace)
        self._capture_decode_variant(trace)
        if feedback_backup is not None:
            ttnn.copy(input_a=feedback_backup, input_b=trace.token_input)
            ttnn.synchronize_device(self.mesh_device)
            ttnn.deallocate(feedback_backup)
            self.trace_counters.device_feedback_restores += 1
            self.trace_counters.synchronizations += 1
        # Setup executed only with inactive positions, so KV is unchanged.
        # Restore the caller's logical first-step inputs and request seeds.
        runtime_positions = torch.full((DECODE_SLOT_COUNT,), -1, dtype=torch.int32)
        runtime_positions[:batch] = torch.where(
            state.active_mask[:batch],
            start_pos.reshape(-1).to(torch.int32),
            torch.full((batch,), -1, dtype=torch.int32),
        )
        self._refresh_device_input(trace.current_pos, runtime_positions)
        self._refresh_device_input(trace.position_ids, runtime_positions.clamp_min(0))
        if trace.sampling_params[3] is not None:
            self._refresh_device_input(trace.sampling_params[3], self._seed_values(sampling_spec))
        self._trace_cache[key] = trace
        return trace

    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
        *,
        next_input: Optional[NextInputFn] = None,
        enable_trace: bool = True,
        sampling_mode: str | None = None,
        top_k: Any = 1,
        top_p: Any = 0.0,
        temperature: Any = 0.0,
        seeds: Any = 0,
        stop_on_eos: bool = True,
        **kwargs: Any,
    ) -> List[int]:
        del kwargs
        if max_new_tokens < 1:
            return []
        start_s = time.perf_counter()
        mode = sampling_mode or ("teacher" if next_input is not None else self.sampling_mode)
        if next_input is not None and mode not in ("teacher", "host"):
            raise ValueError("teacher forcing requires sampling_mode='teacher' or explicit compatibility mode 'host'")
        state = self.model.state or self.model.allocate_state(max_batch_size=1)
        spec = self._sampling_spec(1, top_k=top_k, top_p=top_p, temperature=temperature, seeds=seeds)
        if mode in ("host", "teacher"):
            spec = SamplingSpec(greedy=True)
        counters_before = vars(self.trace_counters).copy()
        logical_len = len(prompt_token_ids)
        required_context = logical_len + max_new_tokens - 1
        if required_context > state.slot_context_lengths[0]:
            raise ValueError(
                f"prompt plus generation requires {required_context} cache positions, "
                f"but slot 0 owns {state.slot_context_lengths[0]}"
            )
        physical = _padded_prefill_len(logical_len)
        prompt = torch.tensor(prompt_token_ids, dtype=torch.long).reshape(1, logical_len)
        padded = torch.nn.functional.pad(prompt, (0, physical - logical_len))
        logits = self.prefill_forward(
            padded,
            page_table=state.page_tables,
            kv_cache=state,
            prompt_lens=[logical_len],
        )
        if mode in ("device", "teacher"):
            # A retained decode trace can be reused across reset() and the
            # next request. These first-token sampler tensors are consumed
            # and read here, then released explicitly so none of them is
            # alive at the next replay.
            with self._track_program_cache_growth("first_token_sampling"):
                k, p, temp, seed_tensor = self._sampling_params(spec)
                sampling_logits = self._pad_sampling_logits(logits)
                # Use the same explicit output contract as traced decode. The
                # allocator-output sampler variant can expose an uninitialized
                # asynchronous result on the first token on P300C.
                first_tt = self._host_tokens_to_device(torch.zeros((1, 1), dtype=torch.long), rank4=True)
                first_tt, _ = self.sampler.decode_forward(
                    sampling_logits, k=k, p=p, temp=temp, seeds=seed_tensor, tt_out_tok=first_tt
                )
                ttnn.synchronize_device(self.mesh_device)
                self.trace_counters.synchronizations += 1
                predicted = int(self._read_tokens(first_tt, 1)[0])
                for tensor in (k, p, temp, seed_tensor, first_tt):
                    ttnn.deallocate(tensor)
                if sampling_logits is not logits:
                    ttnn.deallocate(sampling_logits)
                ttnn.deallocate(logits)
            self._seeded_slots[0] = True
            self._sampling_key = spec.key
        else:
            predicted = int(self._gather_logits_to_torch(logits).reshape(-1).argmax())
        outputs = [predicted]
        first_token_s = time.perf_counter()
        feed = next_input(0, predicted) if next_input is not None else predicted
        eos_ids = set(self.model.eos_token_ids)
        decode_step_times = []
        for step in range(1, max_new_tokens):
            if stop_on_eos and next_input is None and predicted in eos_ids:
                break
            step_start_s = time.perf_counter()
            sampled = self.decode_forward(
                torch.tensor([[feed]], dtype=torch.long),
                torch.tensor([logical_len + step - 1], dtype=torch.int32),
                page_table=state.page_tables,
                kv_cache=state,
                sampling_mode=mode,
                enable_trace=enable_trace,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                seeds=seeds,
            )
            if mode in ("device", "teacher"):
                predicted = int(self._read_tokens(sampled, 1)[0])
            else:
                predicted = int(self._gather_logits_to_torch(sampled, logical_len=1).reshape(-1).argmax())
            outputs.append(predicted)
            feed = next_input(step, predicted) if next_input is not None else predicted
            decode_step_times.append(time.perf_counter() - step_start_s)
        end_s = time.perf_counter()
        decode_tokens = max(len(outputs) - 1, 0)
        decode_elapsed_s = max(end_s - first_token_s, 0.0)
        warm_decode_times = decode_step_times[min(5, len(decode_step_times)) :]
        counter_deltas = {
            name: int(value) - int(counters_before[name]) for name, value in vars(self.trace_counters).items()
        }
        self.last_perf = {
            "ttft_ms": (first_token_s - start_s) * 1000.0,
            "elapsed_s": end_s - start_s,
            "e2e_t/s/u": len(outputs) / max(end_s - start_s, 1e-12),
            "decode_tokens": float(decode_tokens),
            "decode_elapsed_s": decode_elapsed_s,
            "decode_t/s/u": decode_tokens / max(decode_elapsed_s, 1e-12),
            "warmed_decode_tokens": float(len(warm_decode_times)),
            "warmed_decode_elapsed_s": float(sum(warm_decode_times)),
            "warmed_decode_t/s/u": len(warm_decode_times) / max(sum(warm_decode_times), 1e-12),
            "trace_replays": float(counter_deltas.pop("replays")),
            **{name: float(value) for name, value in counter_deltas.items()},
        }
        return outputs

    def reset(self) -> None:
        state = self.model.state
        if state is not None:
            for cache_pair in state.kv_cache:
                for cache in cache_pair:
                    ttnn.fill(cache, 0.0, memory_config=cache.memory_config(), output_tensor=cache)
            state.prompt_lens[:] = [0] * DECODE_SLOT_COUNT
            state.positions.fill_(-1)
            state.active_mask.zero_()
        self.trace_counters = TraceCounters()
        self._request_boundary = True
        self._seeded_slots.zero_()
        self._sampling_key = None
        if state is not None:
            self._page_table_sources.pop(id(state), None)

    def teardown(self) -> None:
        self.release_decode_traces(reason="teardown")


def build_generator(model_dir: str | Path, mesh_device: Any, **kwargs: Any) -> Gemma4Generator:
    model_dir = Path(model_dir)
    model_path = kwargs.pop("model_path", None) or kwargs.pop("hf_model", None) or "google/gemma-4-26B-A4B-it"
    configured_max_seq_len = kwargs.pop("max_seq_len", None)
    max_seq_len = None if configured_max_seq_len is None else int(configured_max_seq_len)
    max_batch_size = int(kwargs.pop("max_batch_size", 1))
    num_layers = kwargs.pop("num_layers", None)
    layer_indices = kwargs.pop("layer_indices", None)
    sampling_mode = kwargs.pop("sampling_mode", "device")
    precision_config_path = kwargs.pop("precision_config_path", None)
    create_kv_cache = bool(kwargs.pop("create_kv_cache", True))
    tensor_cache_path = Path(
        kwargs.pop("tensor_cache_path", os.environ.get("GEMMA4_TENSOR_CACHE_PATH", "/tmp/gemma4_full_model_cache"))
    )
    state_dict = kwargs.pop("state_dict", None)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"unknown generator options: {unknown}")
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if state_dict is None:
        state_dict = Gemma4ModelArgs.load_state_dict(model_path, dummy_weights=False)
    model = Gemma4FullModel(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state_dict,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        num_layers=num_layers,
        layer_indices=layer_indices,
        tensor_cache_path=tensor_cache_path,
        precision_config_path=precision_config_path,
        create_kv_cache=create_kv_cache,
    )
    return Gemma4Generator(model, tokenizer, sampling_mode=sampling_mode)


__all__ = ["Gemma4Generator", "SamplingSpec", "TraceCounters", "build_generator"]
