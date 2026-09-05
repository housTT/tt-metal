# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Readiness generator with separate model and common-sampler decode traces."""

from __future__ import annotations

import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import OrnithModel
from models.common.readiness_check.contract import Generator
from models.common.sampling import SamplingParams, format_sampling_params


def _validate_host_sampling(params, host_sample):
    if host_sample is not None:
        if not callable(host_sample):
            raise TypeError("host_sample must be callable")
        return
    if params is None:
        return
    values = lambda value: value if isinstance(value, (list, tuple)) else [value]
    greedy = all(value == 0 for value in values(params.temperature)) or all(
        value == 1 for value in values(params.top_k)
    )
    plain = all(
        all(value == default for value in values(getattr(params, name, default)))
        for name, default in (
            ("presence_penalty", 0.0),
            ("frequency_penalty", 0.0),
            ("repetition_penalty", 1.0),
            ("enable_log_probs", False),
            ("num_logprobs", 0),
        )
    )
    if not greedy or not plain:
        raise ValueError("Host sampling parameters require an explicit host_sample callback")


def _host_predictions(logits, *, host_sample, sampling_params, step, prompts, predictions):
    if host_sample is None:
        tokens = logits.argmax(-1)
    else:
        tokens = host_sample(
            logits,
            sampling_params=sampling_params,
            step=step,
            prompt_token_ids=[list(prompt) for prompt in prompts],
            generated_token_ids=[list(row) for row in zip(*predictions)] if predictions else [[] for _ in prompts],
        )
    tokens = torch.as_tensor(tokens)
    if tokens.numel() != len(prompts) or tokens.is_floating_point() or tokens.dtype == torch.bool:
        raise ValueError("host_sample must return one integer token ID per prompt")
    tokens = tokens.reshape(-1)
    if bool(((tokens < 0) | (tokens >= logits.shape[-1])).any()):
        raise ValueError("host_sample returned a token ID outside the vocabulary")
    return tokens.tolist()


def _trim_eos(output, *stop_ids):
    stops = set()
    for value in stop_ids:
        if value is not None:
            stops.update(value if isinstance(value, (list, tuple, set)) else [value])
    return [row[: next((i + 1 for i, token in enumerate(row) if token in stops), len(row))] for row in output]


def _request_seed_values(seed_manager, configured_seeds, slots, *, reset_all=False):
    """Use common request/lane RNG initialization without any device access."""
    if reset_all:
        seed_manager.deactivate_slots_except([])
    seed_manager.reset_seed([configured_seeds[slot] for slot in slots], slots)
    return [seed_manager._next_device_seed_for_slot(slot) for slot in slots]


class OrnithGenerator(Generator):
    _output_history_capacity = 128

    def __init__(
        self,
        model,
        *,
        tokenizer=None,
        max_batch_size=1,
        cache_context=None,
        sampling_mode="device",
        kv_cache=None,
        page_table=None,
        force_argmax=False,
        host_sample=None,
        use_prefill_trace=True,
    ):
        if sampling_mode not in ("device", "host"):
            raise ValueError("sampling_mode must be device or host")
        self.model = model
        self.mesh_device = model.mesh_device
        self.tokenizer = tokenizer
        self.sampling_mode = sampling_mode
        self.use_prefill_trace = use_prefill_trace
        if host_sample is not None and sampling_mode != "host":
            raise ValueError('host_sample requires sampling_mode="host"')
        self.host_sample = host_sample
        self.owns_cache = kv_cache is None
        self.kv_cache = kv_cache or model.allocate_cache(max_batch_size, cache_context)
        self.max_batch_size = self.kv_cache.batch_size
        self.page_table = (
            model.page_table(self.kv_cache) if page_table is None else torch.as_tensor(page_table).to(torch.int32)
        )
        self.sampling = model.build_sampler(force_argmax=force_argmax)
        # Fixed slots are independent requests. Duplicate explicit seeds must
        # reproduce the same stream, not acquire salts from another slot's life.
        self.sampling.seed_manager.salt_duplicate_seeds = False
        self.sampling.reset_sampling_params(
            format_sampling_params(SamplingParams(temperature=0.0, top_k=1, top_p=1.0), 32)
        )
        self.counters = model.counters
        for name in (
            "history_replays",
            "history_index_refreshes",
            "history_readbacks",
            "prefill_replays",
            "prefill_captures",
            "prefill_trace_misses",
            "prefill_token_refreshes",
            "prefill_page_table_refreshes",
            "prefill_eager_calls",
            "prefill_sampling_replays",
        ):
            self.counters.setdefault(name, 0)
        self.perf = {}
        self._model_trace = None
        self._sampling_trace = None
        self._sampling_history_trace = None
        self._prefill_trace = None
        self._prefill_key = None
        self._prefill_inputs = None
        self._previous_prefill_table = None
        self._logits = None
        self._programs = None
        self._previous_table = None
        self._inputs = [
            self._device(torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32),
            self._device(torch.zeros(self.max_batch_size, dtype=torch.int32), ttnn.int32),
            self._device(torch.zeros(1, self.max_batch_size, dtype=torch.int32), ttnn.uint32),
            self._device(self.page_table, ttnn.int32),
        ]
        self._output_history = None
        self._output_history_index = None
        self._history_rows = 0
        if sampling_mode == "device":
            self._output_history = self._device(
                torch.zeros(self._output_history_capacity, 1, 1, 32, dtype=torch.int32), ttnn.uint32
            )
            self._output_history_index = self._device(torch.zeros(1, dtype=torch.int32), ttnn.int32)
        self._live = False
        self._sampling_key = None
        self._seed_values = torch.ones(32, dtype=torch.int32)
        self._configured_seeds = [None] * 32

    def _device(self, tensor, dtype):
        return self.model.upload(tensor, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    def _write(self, value, target, counter):
        host = self.model.upload(value, dtype=target.dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=False)
        ttnn.copy_host_to_device_tensor(host, target)
        self.counters[counter] += 1

    def _write_tokens(self, tokens):
        padded = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
        values = torch.as_tensor(tokens).reshape(-1)
        padded.reshape(-1)[: values.numel()] = values
        self._write(padded, self._inputs[0], "token_refreshes")

    def _write_positions(self, positions):
        pos = torch.as_tensor(positions).reshape(-1).to(torch.int32)
        if pos.numel() != self.max_batch_size:
            raise ValueError("positions must explicitly describe every fixed slot")
        if self.max_batch_size == 1 and int(pos[0]) < 0:
            raise ValueError("Batch-one high-level generation requires an active request")
        if self.max_batch_size > 1:
            active = (pos >= 0).float()
            for tensor, shape in (
                (self.kv_cache.active_recurrent, (-1, 1, 1, 1)),
                (self.kv_cache.active_conv, (-1, 1, 1)),
            ):
                host = self.model.upload(active.reshape(shape), dtype=tensor.dtype, device=False)
                ttnn.copy_host_to_device_tensor(host, tensor)
        self._write(pos, self._inputs[1], "position_refreshes")
        self._write(pos.clamp_min(0).reshape(1, -1), self._inputs[2], "rope_refreshes")

    def _refresh_table(self, page_table):
        table = torch.as_tensor(page_table).to(torch.int32)
        if list(table.shape) != list(self._inputs[3].shape):
            raise ValueError("page table shape must match the persistent trace table")
        if self._previous_table is None or not torch.equal(table, self._previous_table):
            self._write(table, self._inputs[3], "page_table_refreshes")
            self._previous_table = table.clone()

    def _forward(self):
        tokens, pos, rot, page = self._inputs
        return self.model.decode_forward(tokens, current_pos=pos, rot_idxs=rot, page_table=page, kv_cache=self.kv_cache)

    def _sample_device(self, logits):
        # The common implementation owns filtering, penalties, tie-breaking and RNG draw.
        # The outer trace owns RNG counter advance, including explicit seeded requests.
        output = self.sampling.sample(logits, tt_out_tok=self._inputs[0], enable_trace=False)
        ttnn.plus_one(self.sampling.tt_sampling.seeds_tt_tensor)
        return output

    def _sample_first_token(self, logits):
        """Use the common sampling trace when prefill populated its bound logits."""
        if logits is self._logits and self._programs == self.mesh_device.num_program_cache_entries():
            ttnn.execute_trace(self.mesh_device, self._sampling_trace, cq_id=0, blocking=False)
            self.counters["sampling_replays"] += 1
            self.counters["prefill_sampling_replays"] += 1
        else:
            self._sample_device(logits)

    def _append_output_history(self):
        # indexed_fill is out-of-place; replay must copy back to its stable input.
        updated = ttnn.indexed_fill(self._output_history_index, self._output_history, self._inputs[0], dim=0)
        ttnn.copy(updated, self._output_history)
        ttnn.plus_one(self._output_history_index)
        # Only history survives replay; later traces must not retain this scratch.
        ttnn.deallocate(updated)

    def _reset_output_history(self):
        self._write(torch.zeros(1, dtype=torch.int32), self._output_history_index, "history_index_refreshes")
        self._history_rows = 0

    def _read_output_history(self):
        self.counters["readbacks"] += 1
        self.counters["history_readbacks"] += 1
        pending = ttnn.get_device_tensors(self._output_history)[0].cpu(blocking=False)
        event = ttnn.record_event(self.mesh_device, 0)
        ttnn.event_synchronize(event)
        self.counters["read_waits"] += 1
        return ttnn.to_torch(pending).reshape(self._output_history_capacity, 32)[: self._history_rows].to(torch.int64)

    def _configure_sampling(self, params):
        params = params or SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
        formatted = format_sampling_params(params, 32)
        key = (
            tuple(formatted.presence_penalty),
            tuple(formatted.frequency_penalty),
            tuple(formatted.repetition_penalty),
            tuple(formatted.enable_log_probs),
            tuple(formatted.num_logprobs),
            tuple(formatted.top_k),
            tuple(formatted.top_p),
            tuple(formatted.temperature),
        )
        if self._model_trace is not None and key != self._sampling_key:
            self._release_traces()
        self.sampling.reset_sampling_params(formatted)
        self._sampling_key = key
        self._configured_seeds = list(formatted.seed)

    def _reset_seeds(self):
        self._seed_values = torch.tensor(
            _request_seed_values(self.sampling.seed_manager, self._configured_seeds, list(range(32)), reset_all=True),
            dtype=torch.int32,
        )
        target = self.sampling.tt_sampling.seeds_tt_tensor
        host = self.model.upload(self._seed_values, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=False)
        ttnn.copy_host_to_device_tensor(host, target)

    def _reset_request_seeds(self, rows):
        """Admit fresh requests in selected slots; preserve ongoing device counters."""
        values = _request_seed_values(self.sampling.seed_manager, self._configured_seeds, rows)
        self._seed_values[rows] = torch.tensor(values, dtype=torch.int32)
        keep = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
        keep[..., rows] = 1
        mask = self.model.upload(keep, dtype=ttnn.int32)
        fresh = self.model.upload(self._seed_values.reshape(1, 1, 1, 32), dtype=ttnn.uint32)
        target = self.sampling.tt_sampling.seeds_tt_tensor
        old = ttnn.to_layout(ttnn.reshape(target, (1, 1, 1, 32)), ttnn.TILE_LAYOUT)
        merged = ttnn.where(mask, fresh, old)
        restored = ttnn.reshape(ttnn.to_layout(merged, ttnn.ROW_MAJOR_LAYOUT), target.shape)
        ttnn.copy(restored, target)
        for tensor in (mask, fresh, old, merged, restored):
            ttnn.deallocate(tensor)

    def _sampler_history_tensors(self):
        penalties = self.sampling.tt_penalties
        return [penalties.output_mask, penalties.output_counts, penalties.output_counts_gathered]

    def configure_sampling(
        self, sampling_params=None, *, reset_seed=False, prompt_token_ids=None, generated_token_ids=None
    ):
        """Change low-level sampling at a scheduler boundary, preserving live state.

        Parameter lists describe fixed slots. Tokens, positions, pages, cache and
        penalty history survive recapture; seeds survive unless reset_seed=True.
        Enabling penalties on a live stream that did not track them requires
        caller-owned prompt/generated histories for every fixed slot.
        Call only when scheduler state changes, outside the steady-state loop.
        """
        if self.sampling_mode != "device":
            raise ValueError("configure_sampling controls the device sampler; host callers own host_sample")
        params = sampling_params or SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
        formatted = format_sampling_params(params, 32)
        penalties_on = any(
            any(value != default for value in getattr(formatted, name))
            for name, default in (("presence_penalty", 0.0), ("frequency_penalty", 0.0), ("repetition_penalty", 1.0))
        )
        if self._live and penalties_on and not self.sampling._penalties_active:
            if prompt_token_ids is None or generated_token_ids is None:
                raise ValueError("Enabling live penalties requires prompt_token_ids and generated_token_ids")
        histories = []
        for rows in (prompt_token_ids, generated_token_ids):
            if rows is None:
                histories.append(None)
                continue
            if len(rows) != self.max_batch_size:
                raise ValueError("Sampling histories must explicitly describe every fixed slot")
            packed = torch.full((32, max(1, max(map(len, rows)))), -1, dtype=torch.int64)
            for slot, row in enumerate(rows):
                packed[slot, : len(row)] = torch.tensor(row, dtype=torch.int64)
            histories.append(packed)
        if not self._live:
            self._configure_sampling(sampling_params)
            self.ensure_traces()
            self.sampling.reset_output_state()
            if histories[0] is not None:
                self.sampling.reset_prompt_tokens(histories[0])
            if histories[1] is not None:
                self.sampling.reset_output_state(histories[1])
            if reset_seed:
                self._reset_seeds()
            return
        if self._model_trace is None or self._logits is None:
            raise RuntimeError("Live sampling reconfiguration requires an existing decode trace")
        # Only warm the sampler: KV/recurrent state and position/page inputs are
        # never executed or reset. Restore every sampling mutation before capture.
        originals = [self._inputs[0], self._logits, self.sampling.tt_sampling.seeds_tt_tensor]
        originals += self._sampler_history_tensors()
        saved = [ttnn.clone(tensor) for tensor in originals]
        self._release_traces()
        try:
            self._configure_sampling(sampling_params)
            self._sample_device(originals[1])
        finally:
            for original, backup in zip(originals, saved):
                ttnn.copy(backup, original)
                ttnn.deallocate(backup)
        if reset_seed:
            self._reset_seeds()
        if histories[0] is not None:
            self.sampling.reset_prompt_tokens(histories[0])
        if histories[1] is not None:
            self.sampling.reset_output_state(histories[1])
        self._capture()

    def _sample_prefill_device(self, logits, rows):
        # The common sampler spans 32 lanes even for a partial prefill. Preserve
        # ongoing slots' feedback, RNG and penalty state entirely on device.
        targets = [self._inputs[0], self.sampling.tt_sampling.seeds_tt_tensor]
        targets += self._sampler_history_tensors() if self.sampling._penalties_active else []
        saved = [ttnn.clone(tensor) for tensor in targets]
        self._sample_device(logits)
        keep = torch.zeros(32, dtype=torch.int32)
        keep[rows] = 1
        # WHERE selects its LLK by predicate dtype. UINT32 currently selects
        # the BF16/LO16 implementation; INT32 selects exact 32-bit data moves.
        # Values and persistent outputs remain UINT32, including full RNG seeds.
        lane_mask = self.model.upload(keep.reshape(1, 1, 1, 32), dtype=ttnn.int32)
        row_mask = self.model.upload(keep.reshape(32, 1), dtype=ttnn.int32)
        for index, (target, backup) in enumerate(zip(targets, saved)):
            if index < 2:
                value = ttnn.to_layout(ttnn.reshape(target, (1, 1, 1, 32)), ttnn.TILE_LAYOUT)
                old = ttnn.to_layout(ttnn.reshape(backup, (1, 1, 1, 32)), ttnn.TILE_LAYOUT)
                merged = ttnn.where(lane_mask, value, old)
                restored = ttnn.reshape(ttnn.to_layout(merged, ttnn.ROW_MAJOR_LAYOUT), target.shape)
                ttnn.copy(restored, target)
                for tensor in (value, old, merged, restored):
                    ttnn.deallocate(tensor)
            else:
                ttnn.where(row_mask, target, backup, output_tensor=target)
            ttnn.deallocate(backup)
        ttnn.deallocate(lane_mask)
        ttnn.deallocate(row_mask)

    def _prepare_prompt_sampling(self, tokens, prompt_lens, rows, start_pos):
        if self.sampling_mode != "device":
            return
        starts = [0] * len(rows) if start_pos is None else list(start_pos)
        new_rows = [slot for slot, start in zip(rows, starts) if start == 0]
        if new_rows:
            self._reset_request_seeds(new_rows)
        if not self.sampling._penalties_active:
            return
        penalties = self.sampling.tt_penalties
        shadow = penalties._prompt_tokens_host
        prompts = []
        for tokens_row, length, slot, start in zip(tokens, prompt_lens, rows, starts):
            prefix = [] if not start or shadow is None else shadow[slot][shadow[slot] >= 0].tolist()
            prompts.append(prefix + torch.as_tensor(tokens_row)[:length].tolist())
        packed = torch.full((32, max(map(len, prompts))), -1, dtype=torch.int64)
        for slot, prompt in zip(rows, prompts):
            packed[slot, : len(prompt)] = torch.tensor(prompt)
        self.sampling.reset_prompt_tokens(packed, slots=rows)
        keep = torch.ones(32, 1, dtype=torch.int32)
        for slot, start in zip(rows, starts):
            if start == 0:
                keep[slot] = 0
        mask = self.model.upload(keep, dtype=ttnn.int32)
        for tensor in self._sampler_history_tensors():
            ttnn.mul(tensor, mask, output_tensor=tensor)
        ttnn.deallocate(mask)

    def _capture(self):
        self._release_traces()
        self._model_trace = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._logits = self._forward()
        ttnn.end_trace_capture(self.mesh_device, self._model_trace, cq_id=0)
        if self.sampling_mode == "device":
            self._sampling_trace = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            self._sample_device(self._logits)
            ttnn.end_trace_capture(self.mesh_device, self._sampling_trace, cq_id=0)
            self._sampling_history_trace = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            self._sample_device(self._logits)
            self._append_output_history()
            ttnn.end_trace_capture(self.mesh_device, self._sampling_history_trace, cq_id=0)
        if self._prefill_inputs is not None:
            # Capture last: every persistent buffer already predates this trace.
            # Its temporary output must not survive behind the decode traces.
            self._prefill_trace = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            self._prefill_trace_forward(self._logits)
            ttnn.end_trace_capture(self.mesh_device, self._prefill_trace, cq_id=0)
            self.counters["prefill_captures"] += 1
        self._programs = self.mesh_device.num_program_cache_entries()

    def _release_prefill_inputs(self):
        if self._prefill_inputs is not None:
            for tensor in self._prefill_inputs:
                ttnn.deallocate(tensor)
        self._prefill_inputs = None
        self._prefill_key = None
        self._previous_prefill_table = None

    def _prepare_prefill_trace(self, prompt_lens, rows, starts, table, *, return_all_logits=False):
        """Select one validated shape; live shape misses retain the eager path."""
        if not (
            self.use_prefill_trace
            and self.sampling_mode == "device"
            and self.owns_cache
            and self.max_batch_size == 1
            and rows == [0]
            and starts == [0]
            and not return_all_logits
            and 1 <= int(prompt_lens[0]) <= self.model.prefill_chunk
        ):
            return False
        length = int(prompt_lens[0])
        key = (id(self.kv_cache), length, starts[0], rows[0], tuple(table.shape))
        if key != self._prefill_key:
            if self._live:
                return False
            # Later persistent allocations invalidate every older trace.
            self._release_traces()
            self._release_prefill_inputs()
            self._prefill_inputs = [
                self._device(torch.zeros(1, length, dtype=torch.int32), ttnn.uint32),
                self._device(table.to(torch.int32), ttnn.int32),
            ]
            self._prefill_key = key
            self.counters["prefill_trace_misses"] += 1
        return True

    def _prefill_trace_forward(self, output):
        ids, page = self._prefill_inputs
        temporary = self.model.prefill_last_logits(ids, page_table=page, kv_cache=self.kv_cache)
        ttnn.copy(temporary, output)
        ttnn.deallocate(temporary)

    def ensure_traces(self):
        if self._model_trace is not None:
            return
        if self._live:
            raise RuntimeError("Capture must be warmed before writing live prompt state")
        originals = [] if self.owns_cache else self.model.cache_buffers(self.kv_cache)
        saved = [ttnn.clone(tensor) for tensor in originals]
        self.model.reset_cache(self.kv_cache, clear_kv=True)
        self._write_tokens([0] * self.max_batch_size)
        self._write_positions([0] * self.max_batch_size)
        self._refresh_table(self.page_table)
        warm = self._forward()
        if self._prefill_inputs is not None:
            for layer in self.kv_cache.prefill_layers:
                layer.reset_state()
            self._prefill_trace_forward(warm)
        if self.sampling_mode == "device":
            self._sample_device(warm)
            self._reset_output_history()
            self._append_output_history()
        ttnn.synchronize_device(self.mesh_device)
        ttnn.deallocate(warm)
        if self.sampling_mode == "device":
            self._reset_output_history()
        self.model.reset_cache(self.kv_cache, clear_kv=True)
        self._write_tokens([0] * self.max_batch_size)
        self._write_positions([0] * self.max_batch_size)
        for original, backup in zip(originals, saved):
            ttnn.copy(backup, original)
            ttnn.deallocate(backup)
        self._capture()

    def _ensure_replay_safe(self):
        if self._programs != self.mesh_device.num_program_cache_entries():
            # New prefill program binaries must not overlap temporary addresses recorded earlier.
            # Recapture records the already-warmed graph without executing a cache update.
            self._capture()

    def _replay(self, *, collect_output=False):
        if collect_output and (
            self._sampling_history_trace is None or self._history_rows >= self._output_history_capacity
        ):
            raise RuntimeError("Output collection requires a captured sampler and available history capacity")
        ttnn.execute_trace(self.mesh_device, self._model_trace, cq_id=0, blocking=False)
        self.counters["model_replays"] += 1
        if self.sampling_mode == "device":
            trace = self._sampling_history_trace if collect_output else self._sampling_trace
            ttnn.execute_trace(self.mesh_device, trace, cq_id=0, blocking=False)
            self.counters["sampling_replays"] += 1
            if collect_output:
                self.counters["history_replays"] += 1
                self._history_rows += 1

    def _read_tokens(self):
        self.counters["readbacks"] += 1
        return (
            ttnn.to_torch(ttnn.get_device_tensors(self._inputs[0])[0])
            .reshape(-1)[: self.max_batch_size]
            .to(torch.int64)
        )

    def prefill_forward(
        self,
        tokens,
        *,
        page_table,
        kv_cache,
        prompt_lens,
        return_all_logits=False,
        slots=None,
        start_pos=None,
        return_device_logits=False,
    ):
        return self._prefill(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            return_all_logits=return_all_logits,
            slots=slots,
            start_pos=start_pos,
            return_device_logits=return_device_logits,
        )

    def _prefill(
        self,
        tokens,
        *,
        page_table,
        kv_cache,
        prompt_lens,
        return_all_logits=False,
        slots=None,
        start_pos=None,
        return_device_logits=False,
        borrow_logits=False,
        state_already_reset=False,
    ):
        if kv_cache is not self.kv_cache:
            raise ValueError("Bind caller-owned cache at generator construction before capture")
        rows, starts, table = self.model.validate_prefill(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            slots=slots,
            start_pos=start_pos,
        )
        traced = self._prepare_prefill_trace(prompt_lens, rows, starts, table, return_all_logits=return_all_logits)
        self.ensure_traces()
        if traced:
            if not state_already_reset:
                for layer in kv_cache.prefill_layers:
                    layer.reset_state()
            # Seed/penalty preparation can compile programs. Finish it before
            # recapture, which replaces the canonical logits buffer.
            self._prepare_prompt_sampling(tokens, prompt_lens, rows, starts)
            ids, page = self._prefill_inputs
            self._write(
                torch.as_tensor(tokens[0])[: int(prompt_lens[0])].reshape(1, -1).to(torch.int32),
                ids,
                "prefill_token_refreshes",
            )
            if self._previous_prefill_table is None or not torch.equal(table, self._previous_prefill_table):
                self._write(table.to(torch.int32), page, "prefill_page_table_refreshes")
                self._previous_prefill_table = table.clone()
            self._ensure_replay_safe()
            ttnn.execute_trace(self.mesh_device, self._prefill_trace, cq_id=0, blocking=False)
            self.counters["prefill_replays"] += 1
            outputs = self._logits
        else:
            outputs = self.model._prefill_validated(
                tokens,
                page_table=table,
                kv_cache=kv_cache,
                prompt_lens=prompt_lens,
                slots=rows,
                start_pos=starts,
                return_all_logits=return_all_logits,
            )
            self.counters["prefill_eager_calls"] += 1
            self._prepare_prompt_sampling(tokens, prompt_lens, rows, starts)
        self._live = True
        if return_all_logits:
            max_len = max(prompt_lens)
            result = torch.zeros(len(outputs), max_len, self.model.vocab_size)
            for i, out in enumerate(outputs):
                result[i, : out.shape[0]] = out
            return result
        if return_device_logits:
            # Public callers own their result. Only generate borrows the buffer
            # that all four traces share; deallocating it would invalidate them.
            return ttnn.clone(outputs) if traced and not borrow_logits else outputs
        if self.sampling_mode == "host":
            return self.model.logits_to_host(outputs, self.max_batch_size)[rows, None, :]
        self._sample_prefill_device(outputs, rows)
        return self._read_tokens()[rows]

    def decode_forward(self, tokens, start_pos, *, page_table, kv_cache, return_logits=False, read_from_device=True):
        """Replay one step; callers may retain device output without synchronizing.

        The returned device tensor aliases persistent feedback and is overwritten
        by the next replay. The caller owns scheduling and context bounds.
        """
        if kv_cache is not self.kv_cache:
            raise ValueError("decode cache differs from the cache bound to this generator")
        self.ensure_traces()
        self._ensure_replay_safe()
        if tokens is not None:
            self._write_tokens(tokens)
        if start_pos is not None:
            self._write_positions(start_pos)
        self._refresh_table(page_table)
        self._replay()
        if not read_from_device:
            return self._logits if return_logits or self.sampling_mode == "host" else self._inputs[0]
        if return_logits or self.sampling_mode == "host":
            return self.model.logits_to_host(self._logits, self.max_batch_size)
        return self._read_tokens()

    def replay_decode(self, steps):
        """Submit fixed steps using device-owned token, position and page state.

        Bind scheduler changes through prefill_forward/decode_forward first.
        No output collection or host synchronization occurs here. The caller
        must keep every active position inside the allocated cache context.
        """
        if self.sampling_mode != "device":
            raise ValueError("Device token feedback requires device sampling")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        self.ensure_traces()
        self._ensure_replay_safe()
        for _ in range(steps):
            self._replay()
        return self._inputs[0]

    def prefill_logits(self, prompt_token_ids):
        self.reset()
        return self.prefill_forward(
            [prompt_token_ids],
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            prompt_lens=[len(prompt_token_ids)],
            return_all_logits=True,
        )

    def generate(
        self,
        prompt_token_ids,
        max_new_tokens,
        *,
        next_input=None,
        enable_trace=True,
        sampling_params=None,
        stop_on_eos=True,
        host_sample=None,
        **kwargs,
    ):
        """Generate tokens using traced decode.

        Host compatibility mode accepts ``host_sample(logits, *, sampling_params,
        step, prompt_token_ids, generated_token_ids)`` returning one token ID per
        user. The callback owns filtering, penalties, RNG and any logprob state;
        without it only plain greedy host sampling is accepted.

        Free-running output ends at the first tokenizer/model EOS by default.
        Device sampling reads one output history per window of up to 128 decode tokens.
        The device executes the requested generation length before EOS slicing;
        ``stop_on_eos=False`` returns that complete window for performance tests.
        Teacher forcing always returns every prediction and invokes every callback.
        """
        if not enable_trace:
            raise ValueError("This generator requires traced decode")
        host_sample = self.host_sample if host_sample is None else host_sample
        if self.sampling_mode == "host":
            _validate_host_sampling(sampling_params, host_sample)
        elif host_sample is not None:
            raise ValueError('host_sample requires sampling_mode="host"')
        if isinstance(prompt_token_ids, torch.Tensor):
            prompt_token_ids = prompt_token_ids.tolist()
        batched = bool(prompt_token_ids) and isinstance(prompt_token_ids[0], (list, tuple))
        prompts = prompt_token_ids if batched else [prompt_token_ids]
        users = len(prompts)
        lengths = [len(prompt) for prompt in prompts]
        if (
            users > self.max_batch_size
            or max_new_tokens < 0
            or any(n == 0 or n + max_new_tokens > self.kv_cache.context for n in lengths)
        ):
            raise ValueError("Invalid prompt/batch/generation length for cache context")
        if next_input is not None and users != 1:
            raise ValueError("The readiness next_input callback describes one request")
        if max_new_tokens == 0:
            return [[] for _ in prompts] if batched else []
        collect_output = self.sampling_mode == "device" and next_input is None
        request_before = dict(self.counters)
        request_start = time.perf_counter()
        # Prefill overwrites every live KV prefix; future positions are masked.
        # Keep explicit reset() clearing semantics for callers.
        self.reset(clear_kv=False)
        if self.sampling_mode == "device":
            self._configure_sampling(sampling_params)
            rows, starts, table = self.model.validate_prefill(
                prompts, page_table=self.page_table, kv_cache=self.kv_cache, prompt_lens=lengths
            )
            self._prepare_prefill_trace(lengths, rows, starts, table)
        self.ensure_traces()
        if self.sampling_mode == "device":
            self._reset_seeds()
            if self.sampling._penalties_active:
                self.sampling.reset_output_state()
        start = time.perf_counter()
        if self.sampling_mode == "device" and self.use_prefill_trace:
            logits = self._prefill(
                prompts,
                page_table=self.page_table,
                kv_cache=self.kv_cache,
                prompt_lens=lengths,
                return_device_logits=True,
                borrow_logits=True,
                state_already_reset=True,
            )
        else:
            logits = self.prefill_forward(
                prompts,
                page_table=self.page_table,
                kv_cache=self.kv_cache,
                prompt_lens=lengths,
                return_device_logits=True,
            )
        if self.sampling_mode == "device":
            self._sample_first_token(logits)
            first = self._read_tokens()[:users].tolist()
        else:
            first = _host_predictions(
                self.model.logits_to_host(logits, users),
                host_sample=host_sample,
                sampling_params=sampling_params,
                step=0,
                prompts=prompts,
                predictions=[],
            )
            self._write_tokens(first)
        if self.sampling_mode != "device" or logits is not self._logits:
            ttnn.deallocate(logits)
        first_token_time = time.perf_counter()
        self.perf["ttft_s"] = first_token_time - request_start
        self.perf["prefill_only_s"] = first_token_time - start
        self.perf["request_setup_s"] = start - request_start
        self.perf["request_counters"] = {name: self.counters[name] - value for name, value in request_before.items()}
        self._ensure_replay_safe()
        self._write_positions(lengths + [-1] * (self.max_batch_size - users))
        self._refresh_table(self.page_table)
        if collect_output:
            self._reset_output_history()
        predictions = [first]
        if next_input is not None:
            self._write_tokens([int(next_input(0, first[0]))])
        before = dict(self.counters)
        begin = time.perf_counter()
        for step in range(1, max_new_tokens):
            if collect_output:
                self._replay(collect_output=True)
                if self._history_rows == self._output_history_capacity and step + 1 < max_new_tokens:
                    predictions.extend(self._read_output_history()[:, :users].tolist())
                    self._reset_output_history()
            else:
                self._replay()
                token = (
                    self._read_tokens()[:users].tolist()
                    if self.sampling_mode == "device"
                    else _host_predictions(
                        self.model.logits_to_host(self._logits, users),
                        host_sample=host_sample,
                        sampling_params=sampling_params,
                        step=step,
                        prompts=prompts,
                        predictions=predictions,
                    )
                )
                predictions.append(token)
                if step + 1 < max_new_tokens:
                    self._write_tokens([int(next_input(step, token[0]))] if next_input is not None else token)
                elif next_input is not None:
                    next_input(step, token[0])
        if collect_output and self._history_rows:
            predictions.extend(self._read_output_history()[:, :users].tolist())
        elapsed = time.perf_counter() - begin
        self.perf.update(
            decode_s=elapsed,
            decode_steps=max_new_tokens - 1,
            tokens_per_second=(max_new_tokens - 1) / elapsed if elapsed else 0,
            loop_counters={k: self.counters[k] - before[k] for k in before},
        )
        output = [list(row) for row in zip(*predictions)]
        if stop_on_eos and next_input is None:
            output = _trim_eos(
                output,
                getattr(self.tokenizer, "eos_token_id", None),
                getattr(self.model.hf_config, "eos_token_id", None),
            )
        return output if batched else output[0]

    def reset(self, *, clear_kv=True):
        self.model.reset_cache(self.kv_cache, clear_kv=clear_kv)
        self._previous_table = None
        self._live = False

    def _release_traces(self):
        """Release internal trace state even when callers defer public cleanup."""
        self.sampling.reset_trace()
        if self._prefill_trace is not None:
            ttnn.release_trace(self.mesh_device, self._prefill_trace)
            self._prefill_trace = None
        if self._sampling_trace is not None:
            ttnn.release_trace(self.mesh_device, self._sampling_trace)
            self._sampling_trace = None
        if self._sampling_history_trace is not None:
            ttnn.release_trace(self.mesh_device, self._sampling_history_trace)
            self._sampling_history_trace = None
        if self._model_trace is not None:
            ttnn.release_trace(self.mesh_device, self._model_trace)
            self._model_trace = None

    def teardown(self):
        self._release_traces()
        self._release_prefill_inputs()


def build_generator(model_dir, mesh_device, *, use_prefill_trace=True, **kwargs):
    model_keys = (
        "precision_config",
        "layer_indices",
        "max_context",
        "prefill_chunk",
        "lm_head_dtype",
        "lm_head_fidelity",
        "lm_head_columns",
        "lm_head_block_w",
        "lm_head_readers",
        "lm_head_cores",
        "sharded_final_norm",
    )
    model = OrnithModel.from_pretrained(model_dir, mesh_device, **{k: kwargs.pop(k) for k in model_keys if k in kwargs})
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model.model_path, local_files_only=True)
    return OrnithGenerator(model, tokenizer=tokenizer, use_prefill_trace=use_prefill_trace, **kwargs)
