# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal-readiness generator for the Qwen/Qwen3.6-27B TP4 full model.

The measured autoregressive path replays two traces per token: the 64-layer
model trace advances cache and positions, then the common sampling trace writes
the selected token directly into the persistent model token input.  Host
sampling is retained only as an explicit compatibility mode for accuracy tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

import torch
import ttnn
from transformers import AutoTokenizer

from models.autoports.qwen_qwen3_6_27b.tt.model import (
    MAX_BATCH_SIZE,
    FullModelState,
    QwenFullModel,
)
from models.common.readiness_check.contract import Generator as ReadinessGenerator
from models.common.readiness_check.contract import NextInputFn
from models.common.sampling.generator import SamplingParams, format_sampling_params


class Generator(ReadinessGenerator):
    """Readiness and low-level serving adapter with explicit mutable state."""

    def __init__(
        self,
        *,
        mesh_device,
        checkpoint_path: str | Path | None = None,
        max_seq_len: int | None = None,
        override_num_layers: int | None = None,
        override_layer_indices: list[int] | tuple[int, ...] | None = None,
        precision_config_path: str | Path | None = None,
        host_sampling_compatibility: bool = False,
    ):
        self.mesh_device = mesh_device
        self.model = QwenFullModel(
            mesh_device=mesh_device,
            checkpoint_path=checkpoint_path,
            max_seq_len=max_seq_len,
            override_num_layers=override_num_layers,
            override_layer_indices=override_layer_indices,
            precision_config_path=precision_config_path,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model.checkpoint_path, local_files_only=True
        )
        self.host_sampling_compatibility = host_sampling_compatibility
        self._state: FullModelState | None = None
        self._model_trace_id: int | None = None
        self._trace_logits: ttnn.Tensor | None = None
        self._trace_state: FullModelState | None = None
        self._sampling_trace_output = None
        self._previous_page_table: torch.Tensor | None = None
        self._previous_page_table_object: object | None = None
        self._host_sampling_rng: torch.Generator | None = None
        self._device_sampling_slots: tuple[int, ...] = ()
        self._device_output_history: List[int] = []
        self._device_output_history_by_slot: List[List[int]] = []
        self._device_sampling_request_active = False
        self._last_decode_batch_size = MAX_BATCH_SIZE
        self._configure_greedy_sampling()

    @staticmethod
    def _first_sampling_value(value):
        """Select the active row from scalar or single-request list parameters."""
        if isinstance(value, list):
            if not value:
                raise ValueError("sampling parameter lists must not be empty")
            return value[0]
        return value

    def _reset_host_sampling_rng(self, params: SamplingParams) -> None:
        self._host_sampling_rng = torch.Generator(device="cpu")
        seed = self._first_sampling_value(params.seed)
        if seed is None:
            self._host_sampling_rng.seed()
        else:
            self._host_sampling_rng.manual_seed(int(seed))

    def _sample_host_logits(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        *,
        token_history: List[int],
    ) -> int:
        """Sample one request row with the public TT top-k/top-p contract.

        This is the compatibility sampler and the unavoidable host boundary
        for prefill logits.  Decode on the optimized greedy path continues to
        use the common on-device sampler and token-feedback trace.
        """
        scores = logits.detach().to(device="cpu", dtype=torch.float32).reshape(-1).clone()
        if scores.numel() < self.model.vocab_size:
            raise ValueError("logits do not cover the model vocabulary")
        scores = scores[: self.model.vocab_size]

        presence = float(self._first_sampling_value(params.presence_penalty))
        frequency = float(self._first_sampling_value(params.frequency_penalty))
        repetition = float(self._first_sampling_value(params.repetition_penalty))
        if repetition <= 0:
            repetition = 1.0
        if token_history and (presence != 0.0 or frequency != 0.0 or repetition != 1.0):
            history = torch.tensor(token_history, dtype=torch.long)
            history = history[(history >= 0) & (history < scores.numel())]
            if history.numel():
                token_ids, counts = torch.unique(history, return_counts=True)
                if repetition != 1.0:
                    selected = scores[token_ids]
                    scores[token_ids] = torch.where(
                        selected < 0, selected * repetition, selected / repetition
                    )
                scores[token_ids] -= presence + frequency * counts.to(scores.dtype)

        temperature = float(self._first_sampling_value(params.temperature))
        top_k = int(self._first_sampling_value(params.top_k))
        top_p = float(self._first_sampling_value(params.top_p))
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if temperature == 0 or (top_k == 1 and top_p in (0.0, 1.0)):
            return int(torch.argmax(scores).item())

        scores /= temperature
        # Match format_sampling_params: unrestricted k is represented by the
        # largest top-k supported by the common device sampler today.
        top_k = min(32, max(1, top_k if top_k >= 1 else 32), scores.numel())
        cutoff = torch.topk(scores, k=top_k).values[-1]
        scores[scores < cutoff] = -torch.inf

        top_p = min(1.0, max(0.0, top_p))
        if top_p == 0.0:
            return int(torch.argmax(scores).item())
        if top_p < 1.0:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            sorted_probs = torch.softmax(sorted_scores, dim=-1)
            remove = torch.cumsum(sorted_probs, dim=-1) - sorted_probs > top_p
            scores[sorted_indices[remove]] = -torch.inf

        probabilities = torch.softmax(scores, dim=-1)
        if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
            raise ValueError("sampling parameters produced an invalid probability distribution")
        if self._host_sampling_rng is None:
            self._reset_host_sampling_rng(params)
        return int(torch.multinomial(probabilities, 1, generator=self._host_sampling_rng).item())

    def _configure_greedy_sampling(self) -> None:
        params = SamplingParams(temperature=1.0, top_k=1, top_p=0.0, seed=None)
        formatted = self._configure_sampling(params)
        # This helper is also the explicit stochastic-to-greedy transition.
        # Clear any prior request seed so greedy remains on its captured common
        # sampler path rather than inheriting the seeded direct-sampling mode.
        self.model.sampling.seed_manager.reset_seed(formatted.seed, [0])
        self._device_sampling_slots = (0,)

    def _configure_sampling(self, params: SamplingParams) -> SamplingParams:
        # Parameter changes are request-boundary work.  Release resident model
        # and sampler traces before the common sampler can lazily allocate a
        # different greedy/stochastic program's buffers.
        if self._model_trace_id is not None:
            self._release_split_traces()
        # The common formatter pads inactive top-p rows with 1.0.  Broadcast
        # an explicitly scalar greedy request so every physical row uses the
        # compact p=0 representation and the common force-argmax path can be
        # selected for the fixed 32-row decode tensor.
        if (
            not isinstance(params.top_k, list)
            and params.top_k == 1
            and not isinstance(params.top_p, list)
            and params.top_p in (0.0, 1.0)
            and not isinstance(params.temperature, list)
            and params.temperature == 1.0
        ):
            params = SamplingParams(
                temperature=[1.0] * MAX_BATCH_SIZE,
                top_k=[1] * MAX_BATCH_SIZE,
                top_p=[0.0] * MAX_BATCH_SIZE,
                presence_penalty=params.presence_penalty,
                frequency_penalty=params.frequency_penalty,
                repetition_penalty=params.repetition_penalty,
                seed=params.seed,
                enable_log_probs=params.enable_log_probs,
                num_logprobs=params.num_logprobs,
            )
        formatted = format_sampling_params(params, MAX_BATCH_SIZE)
        self.model.sampling.reset_sampling_params(formatted)
        return formatted

    def _initialize_device_sampling_request(
        self,
        formatted_params: SamplingParams,
        *,
        prompt_token_ids: Sequence[int] | Sequence[Sequence[int]] | torch.Tensor,
        first_output_token: int | Sequence[int] | torch.Tensor | None = None,
        output_token_history: Sequence[int] | Sequence[Sequence[int]] | torch.Tensor | None = None,
        active_batch: int = 1,
    ) -> None:
        """Initialize request-owned seed and penalty state for active prefix rows."""
        if active_batch < 1 or active_batch > MAX_BATCH_SIZE:
            raise ValueError(f"active_batch must be in [1, {MAX_BATCH_SIZE}]")

        def rows(value, *, name: str) -> List[List[int]]:
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().tolist()
            value = list(value)
            if not value or not isinstance(value[0], (list, tuple)):
                if active_batch == 1:
                    value = [value]
                elif len(value) == active_batch:
                    value = [[item] for item in value]
                else:
                    raise ValueError(
                        f"flat {name} for a mixed batch must contain exactly one token per row"
                    )
            result = [list(map(int, row)) for row in value]
            if len(result) != active_batch:
                raise ValueError(f"{name} must contain one row per active request")
            return result

        prompt_rows = rows(prompt_token_ids, name="prompt_token_ids")
        if output_token_history is None:
            if first_output_token is None:
                output_rows = [[] for _ in range(active_batch)]
            elif isinstance(first_output_token, torch.Tensor):
                output_rows = [[int(token)] for token in first_output_token.reshape(-1).tolist()]
            elif isinstance(first_output_token, (list, tuple)):
                output_rows = [[int(token)] for token in first_output_token]
            else:
                output_rows = [[int(first_output_token)]]
        else:
            output_rows = rows(output_token_history, name="output_token_history")
        if len(output_rows) != active_batch:
            raise ValueError("output history must contain one row per active request")

        def padded_tensor(history_rows: List[List[int]]) -> torch.Tensor:
            width = max(1, max((len(row) for row in history_rows), default=0))
            padded = torch.full((active_batch, width), -1, dtype=torch.long)
            for slot, row in enumerate(history_rows):
                if row:
                    padded[slot, : len(row)] = torch.tensor(row, dtype=torch.long)
            return padded

        self._device_sampling_slots = tuple(range(active_batch))
        self._device_output_history_by_slot = output_rows
        # Preserve the existing single-request inspection field used by the
        # readiness evidence while the authoritative state remains per slot.
        self._device_output_history = list(output_rows[0]) if active_batch == 1 else []
        self.model.sampling.seed_manager.reset_seed(formatted_params.seed, self._device_sampling_slots)
        self.model.sampling.reset_prompt_tokens(padded_tensor(prompt_rows))
        self._restore_device_output_state()
        self._device_sampling_request_active = True

    def _restore_device_output_state(self) -> None:
        if not self._device_output_history_by_slot:
            self.model.sampling.reset_output_state()
            return
        width = max(1, max(len(row) for row in self._device_output_history_by_slot))
        padded = torch.full((len(self._device_output_history_by_slot), width), -1, dtype=torch.long)
        for slot, row in enumerate(self._device_output_history_by_slot):
            if row:
                padded[slot, : len(row)] = torch.tensor(row, dtype=torch.long)
        self.model.sampling.reset_output_state(padded)

    @staticmethod
    def _history_rows_from_vllm(
        output_tokens: torch.Tensor | None, active_batch: int
    ) -> List[List[int]]:
        """Mirror vLLM's reset-time output history for trace warmup restore."""

        if output_tokens is None:
            return [[] for _ in range(active_batch)]
        rows = output_tokens.detach().cpu().to(torch.long)
        if rows.ndim == 1:
            rows = rows.unsqueeze(1)
        if rows.ndim != 2 or rows.shape[0] < active_batch:
            raise ValueError("output_tokens must contain one row per active request")
        return [
            [int(token) for token in rows[row].tolist() if int(token) >= 0]
            for row in range(active_batch)
        ]

    def _advance_device_sampling_seed(self) -> None:
        if not self._device_sampling_request_active:
            raise RuntimeError(
                "device sampling request is not initialized; pass sampling_request_start=True"
            )
        self.model.sampling.seed_manager.get_new_values(list(self._device_sampling_slots))

    def _record_device_sampled_tokens(self, sampled: torch.Tensor) -> None:
        values = sampled.reshape(-1).to(torch.long).tolist()
        if len(values) != len(self._device_sampling_slots):
            raise ValueError("sampled token count must match active sampling slots")
        for slot, token in enumerate(values):
            self._device_output_history_by_slot[slot].append(int(token))
        if len(values) == 1:
            self._device_output_history = list(self._device_output_history_by_slot[0])

    def _release_split_traces(self) -> None:
        self.model.sampling.reset_trace()
        if self._model_trace_id is not None:
            ttnn.release_trace(self.mesh_device, self._model_trace_id)
        self._model_trace_id = None
        self._trace_logits = None
        self._trace_state = None
        self._sampling_trace_output = None

    @staticmethod
    def _first_device_tensor(tensor: ttnn.Tensor) -> ttnn.Tensor:
        tensors = ttnn.get_device_tensors(tensor)
        return tensors[0] if tensors else tensor

    def _tokens_to_host(self, tensor) -> torch.Tensor:
        if isinstance(tensor, tuple):
            tensor = tensor[0]
        host = ttnn.to_torch(self._first_device_tensor(tensor))
        return host.reshape(-1).to(torch.long)

    def _logits_to_host(self, logits: ttnn.Tensor) -> torch.Tensor:
        host = ttnn.to_torch(
            logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1)
        )
        return host[..., : self.model.vocab_size]

    def _copy_replicated(
        self,
        host: torch.Tensor,
        target: ttnn.Tensor,
        *,
        dtype,
        layout,
    ) -> None:
        source = ttnn.from_torch(
            host,
            dtype=dtype,
            layout=layout,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(source, target)

    def _page_table_to_host(self, page_table) -> torch.Tensor | None:
        if page_table is None:
            return None
        if isinstance(page_table, torch.Tensor):
            return page_table.detach().cpu().to(torch.int32)
        if isinstance(page_table, ttnn.Tensor):
            return ttnn.to_torch(self._first_device_tensor(page_table)).to(torch.int32)
        raise TypeError(f"Unsupported page-table type {type(page_table)}")

    def _resolve_state(
        self,
        *,
        batch_size: int,
        page_table,
        kv_cache,
        state: FullModelState | None,
    ) -> FullModelState:
        if state is not None:
            resolved = state
        elif isinstance(kv_cache, FullModelState):
            resolved = kv_cache
        elif self._state is not None and kv_cache is None:
            resolved = self._state
        else:
            resolved = self.model.allocate_state(
                batch_size=batch_size, page_table=self._page_table_to_host(page_table)
            )
            if kv_cache is not None:
                resolved.kv_cache = kv_cache
        self._state = resolved
        return resolved

    def _refresh_page_table(
        self,
        state: FullModelState,
        page_table,
        *,
        changed: bool | None = None,
    ) -> None:
        if page_table is None or changed is False:
            return
        # Serving reuses one stable device page-table tensor for many decode
        # tokens.  Identity is sufficient to avoid a per-token device read;
        # callers that mutate that tensor in place pass changed=True.
        if changed is None and page_table is self._previous_page_table_object:
            return
        host = self._page_table_to_host(page_table)
        assert host is not None
        if host.ndim != 2 or host.shape[0] > MAX_BATCH_SIZE:
            raise ValueError("page_table must have shape [active_batch, blocks]")
        if host.shape[1] > state.page_table_host.shape[1]:
            raise ValueError("page_table exceeds the allocated state width; allocate a new state")
        padded = torch.zeros_like(state.page_table_host)
        padded[: host.shape[0], : host.shape[1]] = host
        contents_changed = self._previous_page_table is None or not torch.equal(
            padded, self._previous_page_table
        )
        self._previous_page_table_object = page_table
        if not contents_changed:
            return
        state.page_table_host.copy_(padded)
        self._copy_replicated(
            padded,
            state.page_table,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self._previous_page_table = padded.clone()

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table: torch.Tensor,
        kv_cache: Any,
        prompt_lens: List[int],
        return_all_logits: bool = False,
        state: FullModelState | None = None,
        page_table_changed: bool | None = None,
        state_slots: Sequence[int] | None = None,
        sampling_mode: str = "host",
        sampling_params: SamplingParams | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, padded_prompt_len]")
        resolved = self._resolve_state(
            batch_size=tokens.shape[0], page_table=page_table, kv_cache=kv_cache, state=state
        )
        self._refresh_page_table(resolved, page_table, changed=page_table_changed)
        if sampling_mode not in ("device", "host"):
            raise ValueError("sampling_mode must be 'device' or 'host'")
        device_sampling = sampling_mode == "device"
        logits = self.model.prefill(
            tokens.to(torch.long),
            prompt_lens=prompt_lens,
            state=resolved,
            return_all_logits=return_all_logits,
            state_slots=state_slots,
            return_device_last_logits=device_sampling,
        )
        if not device_sampling:
            return logits
        if return_all_logits:
            raise ValueError("device prefill sampling requires last-token logits")
        if sampling_params is None:
            raise ValueError("device prefill sampling requires sampling_params")
        formatted = self._configure_sampling(sampling_params)
        active_batch = tokens.shape[0]
        slots = tuple(range(active_batch)) if state_slots is None else tuple(int(slot) for slot in state_slots)
        prompt_tokens = torch.full(
            (MAX_BATCH_SIZE, tokens.shape[1]), -1, dtype=torch.long
        )
        prompt_tokens[list(slots)] = tokens.to(torch.long)
        self.model.sampling.apply_prefill_state(
            sampling_params=formatted,
            prompt_tokens=prompt_tokens,
            empty_slots=list(slots),
            replicate_seeds=False,
        )
        self._device_sampling_slots = slots
        self._device_output_history_by_slot = [[] for _ in range(active_batch)]
        self._device_output_history = []
        self._device_sampling_request_active = True
        sampled = self.model.sampling.sample(logits, enable_trace=False)
        # Stateful prefills can occupy non-contiguous persistent slots while
        # vLLM consumes prefill results in request order.  Pack only the
        # already-sampled compact token tensor; logits never leave the device.
        def pack_slots(value):
            if value is None or slots == tuple(range(active_batch)):
                return value
            rank = len(value.shape)
            rows = [
                ttnn.slice(
                    value,
                    [0] * (rank - 1) + [slot],
                    list(value.shape[:-1]) + [slot + 1],
                )
                for slot in slots
            ]
            return rows[0] if len(rows) == 1 else ttnn.concat(rows, dim=-1)

        if isinstance(sampled, tuple):
            return pack_slots(sampled[0]), pack_slots(sampled[1])
        return pack_slots(sampled)

    def _snapshot_linear_state(self, state: FullModelState):
        snapshots = []
        for item in state.linear_state:
            snapshots.append(
                None
                if item is None
                else (
                    ttnn.clone(item[0], memory_config=item[0].memory_config()),
                    ttnn.clone(item[1], memory_config=item[1].memory_config()),
                )
            )
        return snapshots

    @staticmethod
    def _restore_linear_state(state: FullModelState, snapshots) -> None:
        for target, source in zip(state.linear_state, snapshots):
            if target is not None:
                ttnn.copy(source[0], target[0])
                ttnn.copy(source[1], target[1])

    def _capture_split_traces(self, state: FullModelState) -> None:
        if self._model_trace_id is not None:
            if self._trace_state is state:
                return
            raise RuntimeError(
                "A decode trace is already bound to another state object; call teardown before replacing state"
            )
        snapshots = self._snapshot_linear_state(state)
        current_host = torch.full((MAX_BATCH_SIZE,), -1, dtype=torch.int32)
        current_host[: len(state.prompt_lens)] = torch.tensor(state.prompt_lens, dtype=torch.int32)
        rotary_host = torch.zeros((1, MAX_BATCH_SIZE), dtype=torch.int32)
        rotary_host[0, : len(state.prompt_lens)] = current_host[: len(state.prompt_lens)]
        token_host = self._tokens_to_host(state.token_buffer).to(torch.int32).reshape(1, 1, 1, MAX_BATCH_SIZE)

        def restore_capture_inputs() -> None:
            self._restore_linear_state(state, snapshots)
            self._copy_replicated(
                current_host,
                state.current_positions,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            self._copy_replicated(
                rotary_host,
                state.rotary_positions,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            self._copy_replicated(
                token_host,
                state.token_buffer,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        warm_logits = self.model.decode_device(state)
        # Materialize every lazy sampler buffer and compile the selected path
        # before any trace exists.  Allocating ordinary device buffers after a
        # model trace is resident can overlap trace storage and is explicitly
        # rejected by Metal's trace-safety contract.
        self.model.sampling.sample(
            warm_logits,
            enable_trace=False,
            tt_out_tok=state.token_buffer,
        )
        ttnn.synchronize_device(self.mesh_device)
        # Sampling warmup is compile-only.  If penalties are enabled, sample()
        # records its throwaway token; restore the real request history before
        # capture/replay so token zero is the sole initial output token.
        self._restore_device_output_state()
        # Warmup mutates the exact persistent tensors that capture consumes:
        # decode advances both position tensors and recurrent state, while the
        # sampler overwrites the next-token input.  Restore before capture so
        # setup executes the same logical step.  This is mandatory at the last
        # context position, where a second un-restored decode would address
        # RoPE index max_seq_len and paged-KV block num_blocks.
        restore_capture_inputs()
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        trace_logits = self.model.decode_device(state)
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        self._sampling_trace_output = self.model.sampling.capture_trace(
            trace_logits,
            tt_out_tok=state.token_buffer,
            skip_precompile=True,
        )
        ttnn.synchronize_device(self.mesh_device)

        # Trace capture may execute the recorded program once.  Restore the
        # request-owned counts so capture itself can never become a generated
        # token in presence/frequency/repetition history.
        self._restore_device_output_state()
        restore_capture_inputs()
        self._model_trace_id = trace_id
        self._trace_logits = trace_logits
        self._trace_state = state

    def _decode_traced_device(self, state: FullModelState):
        self._capture_split_traces(state)
        self._advance_device_sampling_seed()
        ttnn.execute_trace(self.mesh_device, self._model_trace_id, cq_id=0, blocking=False)
        sampled = self.model.sampling.sample(
            self._trace_logits,
            enable_trace=True,
            tt_out_tok=state.token_buffer,
        )
        return sampled

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table: torch.Tensor,
        kv_cache: Any,
        state: FullModelState | None = None,
        enable_trace: bool = True,
        sampling_mode: str = "device",
        sampling_params: SamplingParams | None = None,
        sampling_request_start: bool = False,
        prompt_token_ids: Sequence[int] | Sequence[Sequence[int]] | torch.Tensor | None = None,
        output_token_history: Sequence[int] | Sequence[Sequence[int]] | torch.Tensor | None = None,
        page_table_changed: bool | None = None,
        read_from_device: bool = True,
        reset_batch: bool | None = None,
        slot_remap: torch.Tensor | Sequence[int] | None = None,
        prompt_tokens: torch.Tensor | None = None,
        output_tokens: torch.Tensor | None = None,
        serving_mode: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        resolved = self._resolve_state(
            batch_size=tokens.shape[0], page_table=page_table, kv_cache=kv_cache, state=state
        )
        self._refresh_page_table(resolved, page_table, changed=page_table_changed)
        if serving_mode:
            reset_batch = bool(reset_batch) or resolved.token_buffer is None
            if slot_remap is not None:
                self.model.remap_linear_state_slots(resolved, slot_remap)
                self.model.sampling.seed_manager.apply_slot_remap(slot_remap)
            # Under async scheduling the host token and position supplied for
            # a steady decode may be one step stale.  The sampler has already
            # written the authoritative next token into token_buffer and the
            # model trace advanced current/rotary positions in place, so only
            # a real layout reset is allowed to refresh those tensors.
            if bool(reset_batch):
                self.model.prepare_decode_state(
                    resolved,
                    tokens.reshape(-1),
                    positions=start_pos,
                )
        else:
            self.model.prepare_decode_state(resolved, tokens.reshape(-1), positions=start_pos)
        if sampling_mode not in ("device", "host"):
            raise ValueError("sampling_mode must be 'device' or 'host'")
        if serving_mode and sampling_mode == "device":
            if sampling_params is None:
                raise ValueError("serving device sampling requires sampling_params")
            self.model.sampling.apply_decode_state(
                [sampling_params],
                reset_batch=bool(reset_batch),
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
            active_batch = int((start_pos.reshape(-1) >= 0).sum().item())
            if active_batch < 1 or active_batch > MAX_BATCH_SIZE:
                raise ValueError("serving decode requires an active prefix of 1..32 rows")
            if not torch.all(start_pos.reshape(-1)[:active_batch] >= 0) or not torch.all(
                start_pos.reshape(-1)[active_batch:] < 0
            ):
                raise ValueError("serving decode active rows must be a contiguous prefix")
            self._device_sampling_slots = tuple(range(active_batch))
            if bool(reset_batch):
                # Trace warmup/capture restores sampler output history twice.
                # Mirror the authoritative vLLM history first so those
                # restores cannot erase presence/frequency/repetition state.
                self._device_output_history_by_slot = self._history_rows_from_vllm(
                    output_tokens, active_batch
                )
                self._device_output_history = (
                    list(self._device_output_history_by_slot[0])
                    if active_batch == 1
                    else []
                )
            self._device_sampling_request_active = True
        elif sampling_request_start:
            if sampling_mode != "device":
                raise ValueError("sampling request state applies only to device sampling")
            if sampling_params is None:
                raise ValueError("sampling_params are required when sampling_request_start=True")
            if prompt_token_ids is None:
                raise ValueError("prompt_token_ids are required when sampling_request_start=True")
            formatted = self._configure_sampling(sampling_params)
            self._initialize_device_sampling_request(
                formatted,
                prompt_token_ids=prompt_token_ids,
                output_token_history=(tokens.reshape(-1) if output_token_history is None else output_token_history),
                active_batch=tokens.shape[0],
            )
        elif sampling_params is not None:
            raise ValueError("sampling_params may only be changed at an explicit sampling request start")
        elif sampling_mode == "device" and not self._device_sampling_request_active:
            # Greedy is still explicit request state: initialize its seed and
            # history instead of silently inheriting construction-time state.
            raise RuntimeError("pass sampling_request_start=True for the first device decode")
        if enable_trace:
            self._capture_split_traces(resolved)
            ttnn.execute_trace(self.mesh_device, self._model_trace_id, cq_id=0, blocking=False)
            logits = self._trace_logits
        else:
            logits = self.model.decode_device(resolved)
        if sampling_mode == "host":
            return self._logits_to_host(logits)[0, 0, : tokens.shape[0]]
        self._advance_device_sampling_seed()
        sampled = self.model.sampling.sample(
            logits, enable_trace=enable_trace, tt_out_tok=resolved.token_buffer
        )
        self._last_decode_batch_size = (
            int((start_pos.reshape(-1) >= 0).sum().item()) if serving_mode else tokens.shape[0]
        )
        if not read_from_device:
            return sampled
        sampled_host = self._tokens_to_host(sampled)[: tokens.shape[0]]
        self._record_device_sampled_tokens(sampled_host)
        return sampled_host

    def read_decode_output(self, tt_out, async_read: bool = False):
        """Move a device token result to host storage, optionally nonblocking.

        This is the deferred half of the serving-ready low-level generator
        contract.  Sampling output is replicated, so one device tensor is the
        authoritative compact token result; no logits gather or host argmax is
        introduced here.
        """

        def read_one(value):
            if value is None:
                return None
            if isinstance(value, ttnn.Tensor):
                # Sampled tokens (and supported sampled-token logprobs) are
                # replicated.  Select one authoritative device before the
                # deferred CPU transfer so the host handle is not a
                # distributed MeshTensor requiring a logits-style composer.
                value = self._first_device_tensor(value)
            if hasattr(value, "cpu"):
                return value.cpu(blocking=not async_read)
            return value

        if isinstance(tt_out, tuple):
            host = tuple(read_one(value) for value in tt_out)
        else:
            host = read_one(tt_out)
        if async_read:
            return host, [ttnn.record_event(self.mesh_device, 0)]
        return host

    def process_decode_output_host(self, tt_out, is_tokens: bool = True) -> torch.Tensor:
        """Format a deferred compact-token read without submitting device work."""

        del is_tokens
        if (
            isinstance(tt_out, tuple)
            and len(tt_out) == 2
            and isinstance(tt_out[1], list)
        ):
            # AsyncDecodeManager normally removes this
            # ``(host_handles, read_events)`` envelope.  Accept it here too so
            # direct runner checks use the identical host formatter.
            tt_out, _read_events = tt_out
        log_probs = None
        if isinstance(tt_out, tuple):
            tt_out, log_probs = tt_out
        batch = getattr(self, "_last_decode_batch_size", MAX_BATCH_SIZE)
        tokens = ttnn.to_torch(tt_out).reshape(-1).to(torch.long)[:batch]
        if self._device_sampling_request_active:
            # The captured sampler already advanced its device-resident penalty
            # state.  Keep the host mirror coherent only when the scheduler
            # elects to consume the deferred compact-token read.
            self._record_device_sampled_tokens(tokens)
        if log_probs is None:
            return tokens
        if isinstance(log_probs, ttnn.Tensor):
            log_probs = ttnn.to_torch(log_probs).reshape(-1)[:batch]
        return tokens, log_probs

    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
        *,
        next_input: Optional[NextInputFn] = None,
        enable_trace: bool = True,
        host_sampling: bool | None = None,
        sampling_params: SamplingParams | None = None,
        token_observer: Callable[[int, int], None] | None = None,
        stop_on_eos: bool = True,
        **kwargs: Any,
    ) -> List[int]:
        del kwargs
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if max_new_tokens == 0:
            return []
        if len(prompt_token_ids) + max_new_tokens > self.model.max_seq_len:
            raise ValueError("prompt plus generated tokens exceed the supported context")
        use_host = self.host_sampling_compatibility if host_sampling is None else host_sampling
        if use_host:
            # Explicit compatibility only; optimized token-out remains the default.
            sampling_mode = "host"
        else:
            sampling_mode = "device"

        request_sampling_params = sampling_params or SamplingParams(
            temperature=1.0, top_k=1, top_p=0.0, seed=None
        )
        formatted_sampling_params = self._configure_sampling(request_sampling_params)
        self._reset_host_sampling_rng(request_sampling_params)

        if self._state is not None and self._state.prompt_lens:
            self.reset()
        if self._state is None:
            self._state = self.model.allocate_state(batch_size=1)
        logits = self.model.prefill(
            torch.tensor([prompt_token_ids], dtype=torch.long),
            prompt_lens=[len(prompt_token_ids)],
            state=self._state,
            return_all_logits=False,
        )
        predicted = self._sample_host_logits(
            logits[0, -1], request_sampling_params, token_history=prompt_token_ids
        )
        predictions = [predicted]
        if token_observer is not None:
            token_observer(0, predicted)
        if sampling_mode == "device":
            self._initialize_device_sampling_request(
                formatted_sampling_params,
                prompt_token_ids=prompt_token_ids,
                first_output_token=predicted,
            )
        forced = next_input(0, predicted) if next_input is not None else predicted
        self.model.prepare_decode_state(self._state, torch.tensor([forced]))

        for step in range(1, max_new_tokens):
            if enable_trace:
                if sampling_mode == "host":
                    self._capture_split_traces(self._state)
                    ttnn.execute_trace(
                        self.mesh_device, self._model_trace_id, cq_id=0, blocking=False
                    )
                    predicted = self._sample_host_logits(
                        self._logits_to_host(self._trace_logits)[0, 0, 0],
                        request_sampling_params,
                        token_history=prompt_token_ids + predictions,
                    )
                else:
                    sampled = self._decode_traced_device(self._state)
                    predicted = int(self._tokens_to_host(sampled)[0].item())
            else:
                logits_tt = self.model.decode_device(self._state)
                if sampling_mode == "host":
                    predicted = self._sample_host_logits(
                        self._logits_to_host(logits_tt)[0, 0, 0],
                        request_sampling_params,
                        token_history=prompt_token_ids + predictions,
                    )
                else:
                    self._advance_device_sampling_seed()
                    sampled = self.model.sampling.sample(
                        logits_tt, enable_trace=False, tt_out_tok=self._state.token_buffer
                    )
                    predicted = int(self._tokens_to_host(sampled)[0].item())
            predictions.append(predicted)
            if sampling_mode == "device":
                self._device_output_history_by_slot[0].append(predicted)
                self._device_output_history = list(self._device_output_history_by_slot[0])
            if token_observer is not None:
                token_observer(step, predicted)
            forced = next_input(step, predicted) if next_input is not None else predicted
            # Device sampling already writes its result into token_buffer.
            # Host compatibility (and teacher forcing) must explicitly feed
            # the chosen token back before the next model replay.
            if sampling_mode == "host" or next_input is not None:
                token_host = torch.zeros((1, 1, 1, MAX_BATCH_SIZE), dtype=torch.int32)
                token_host[0, 0, 0, 0] = forced
                self._copy_replicated(
                    token_host,
                    self._state.token_buffer,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            if next_input is None and stop_on_eos and predicted == self.tokenizer.eos_token_id:
                break
        return predictions

    def reset(self) -> None:
        # Sampling slots outlive the wrapper mirrors inside the common
        # SeedManager.  Clear them all before accepting another request so a
        # smaller unseeded batch cannot inherit an inactive row's explicit
        # seed and its per-token host seed update path.
        self.model.sampling.seed_manager.reset_request_state()
        self._device_sampling_slots = ()
        self._device_output_history = []
        self._device_output_history_by_slot = []
        self._device_sampling_request_active = False
        if self._state is None:
            return
        # Request boundaries invalidate both traces.  Their captured programs
        # bind cache/state tensors and must never survive reset or state reuse.
        self._release_split_traces()
        for cache in self._state.kv_cache:
            if cache is None:
                continue
            for tensor in cache:
                ttnn.fill(tensor, 0.0, memory_config=tensor.memory_config(), output_tensor=tensor)
        for state in self._state.linear_state:
            if state is None:
                continue
            ttnn.fill(state[0], 0.0, memory_config=state[0].memory_config(), output_tensor=state[0])
            ttnn.fill(state[1], 0.0, memory_config=state[1].memory_config(), output_tensor=state[1])
        self._state.prompt_lens = ()
        self._state.active_slots = ()
        self._previous_page_table = None
        self._previous_page_table_object = None

    def teardown(self) -> None:
        self._release_split_traces()


def build_generator(model_dir: str | Path, mesh_device, **kwargs) -> Generator:
    """Readiness discovery entry point; ``model_dir`` is intentionally repo-local."""

    del model_dir
    return Generator(mesh_device=mesh_device, **kwargs)


__all__ = ["Generator", "build_generator"]
