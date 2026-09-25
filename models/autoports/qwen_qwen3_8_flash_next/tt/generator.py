# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Standalone/readiness generator for Qwen3.8-Flash-Next on P300 TP4+EP4.

The optimized path uses the model's split traces: traced decoder segments,
traced terminal projection, traced ``Sampling1D`` candidate sampling, direct
``tt_out_tok`` feedback, and traced in-place position advancement.  Host-side
sampling is available only through the explicit compatibility mode used by
logit-based tests; it is never the measured token-out path.
"""

from __future__ import annotations

import dataclasses
import os
import secrets
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.model import (
    DEFAULT_SNAPSHOT,
    EOS_TOKEN_IDS,
    MODEL_ID,
    PAD_TOKEN_ID,
    REQUIRED_L1_SMALL_SIZE,
    Qwen38BatchState,
    Qwen38FullModel,
)
from models.common.llama_models import TokenResult
from models.common.model_capabilities import ModelCapabilitiesMixin


@dataclasses.dataclass
class GenerationMetrics:
    ttft_seconds: float = 0.0
    prefill_seconds: float = 0.0
    decode_capture_seconds: float = 0.0
    decode_step_seconds: list[float] = dataclasses.field(default_factory=list)
    decode_step_breakdown: list[dict[str, object]] = dataclasses.field(default_factory=list)
    sampling_mode: str = "device"
    traced: bool = True
    teacher_forced: bool = False

    @property
    def decode_tokens_per_second_per_user(self) -> float:
        steady = self.decode_step_seconds[1:] if len(self.decode_step_seconds) > 1 else self.decode_step_seconds
        return len(steady) / sum(steady) if steady and sum(steady) else 0.0

    def report(self) -> dict[str, object]:
        """Return the compact readiness/performance view of this generation."""

        steady = self.decode_step_seconds[1:] if len(self.decode_step_seconds) > 1 else self.decode_step_seconds
        total = sum(steady)
        report = {
            "ttft_seconds": self.ttft_seconds,
            "prefill_seconds": self.prefill_seconds,
            "decode_capture_seconds": self.decode_capture_seconds,
            "decode_measured_tokens": len(steady),
            "decode_seconds": total,
            "decode_seconds_per_token": total / len(steady) if steady else 0.0,
            "decode_tokens_per_second_per_user": self.decode_tokens_per_second_per_user,
            "sampling_mode": self.sampling_mode,
            "traced": self.traced,
            "teacher_forced": self.teacher_forced,
        }
        if self.decode_step_breakdown:
            timeline = (
                self.decode_step_breakdown[1:] if len(self.decode_step_breakdown) > 1 else self.decode_step_breakdown
            )

            def percentile(name: str, fraction: float) -> float:
                values = sorted(float(row[name]) for row in timeline)
                if not values:
                    return 0.0
                return values[min(len(values) - 1, int((len(values) - 1) * fraction + 0.5))]

            report["decode_completed_wall_p50_seconds"] = percentile("completed_wall_seconds", 0.50)
            report["decode_completed_wall_p95_seconds"] = percentile("completed_wall_seconds", 0.95)
            report["decode_timeline"] = timeline
        return report


@dataclasses.dataclass
class _ServingDecodeHost:
    """One serving decode output after its device-to-host copy."""

    kind: str
    host: Any
    rows: int
    state: Qwen38BatchState | None = None
    completion_event: Any = None
    _cached_tokens: torch.Tensor | None = dataclasses.field(default=None, init=False, repr=False)
    _cache_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock, init=False, repr=False)

    def to_torch(self, model: Qwen38FullModel, *, is_tokens: bool) -> torch.Tensor:
        if is_tokens != (self.kind == "tokens"):
            raise ValueError(f"decode produced {self.kind}, caller requested {'tokens' if is_tokens else 'logits'}")
        if self.kind == "tokens":
            with self._cache_lock:
                if self._cached_tokens is not None:
                    return self._cached_tokens
                if self.completion_event is not None:
                    ttnn.event_synchronize(self.completion_event)
                    self.completion_event = None
                hosts = self.host if isinstance(self.host, tuple) else (self.host,)
                values = []
                device_readbacks = 0
                for host in hosts:
                    if isinstance(host, torch.Tensor):
                        values.append(host.reshape(-1)[0].to(torch.int64))
                    else:
                        shard = ttnn.get_device_tensors(host)[0]
                        values.append(ttnn.to_torch(shard).reshape(-1)[0].to(torch.int64))
                        device_readbacks += 1
                self._cached_tokens = torch.stack(values)[: self.rows]
                if self.state is not None:
                    self.state.compact_token_readbacks += device_readbacks
                return self._cached_tokens
        if isinstance(self.host, torch.Tensor):
            return self.host.reshape(self.rows, 1, model.vocab_size)
        hosts = self.host if isinstance(self.host, tuple) else (self.host,)
        logits = [host if isinstance(host, torch.Tensor) else model.logits_to_torch(host) for host in hosts]
        return torch.cat(
            [value.reshape(1, 1, model.vocab_size) for value in logits],
            dim=0,
        )[: self.rows]


@dataclasses.dataclass
class _ServingDecodeOutput:
    """Opaque submitted output used by the shared asynchronous decode split."""

    kind: str
    device: Any
    rows: int
    state: Qwen38BatchState | None = None
    slot_keys: tuple[tuple[int, object, int], ...] = ()

    def read(self, *, blocking: bool) -> _ServingDecodeHost:
        devices = self.device if isinstance(self.device, tuple) else (self.device,)
        hosts = tuple(
            (device if isinstance(device, torch.Tensor) else device.cpu(blocking=blocking)) for device in devices
        )
        return _ServingDecodeHost(
            kind=self.kind,
            host=hosts if isinstance(self.device, tuple) else hosts[0],
            rows=self.rows,
            state=self.state,
        )


@dataclasses.dataclass
class _ServingVirtualSlot:
    """Generator-owned protocol state for one stable vLLM virtual slot."""

    request_id: object
    external_generation: int
    lease: Any
    device_feedback_current: bool = False
    sampling_signature: tuple | None = None
    resumed_sampling_rng_state: tuple[object, ...] | None = None
    pending_token_host: tuple[_ServingDecodeHost, int] | None = None


def _resolve_snapshot(model_dir: str | Path) -> Path:
    path = Path(model_dir).expanduser()
    if path.exists():
        return path.resolve()
    if str(model_dir) == MODEL_ID and DEFAULT_SNAPSHOT.exists():
        return DEFAULT_SNAPSHOT.resolve()
    raise FileNotFoundError(f"{model_dir!s} is neither a local snapshot nor the cached target {MODEL_ID} revision")


def _as_parameter(value, batch: int, *, dtype) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=dtype).reshape(-1)
    if result.numel() == 1:
        result = result.repeat(batch)
    if result.numel() != batch:
        raise ValueError(f"parameter must be scalar or contain {batch} values")
    return result


class Qwen38Generator(ModelCapabilitiesMixin):
    """Thin deterministic loop over the serving-facing full-model API."""

    required_device_params = {
        "fabric_config": "FABRIC_1D",
        "l1_small_size": REQUIRED_L1_SMALL_SIZE,
        "trace_region_size": 1_073_741_824,
    }

    def __init__(self, model: Qwen38FullModel, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.mesh_device = model.mesh_device
        self.state: Qwen38BatchState | None = None
        self.last_metrics: GenerationMetrics | None = None
        self.last_generated_tokens: torch.Tensor | None = None
        self.last_prompt_tokens: torch.Tensor | None = None
        self.host_sampling_compatibility_calls = 0
        self.async_feedback_host_reuses = 0
        self.async_feedback_device_fallbacks = 0
        self._serving_device_feedback_current = False
        self._serving_pending_token_host: _ServingDecodeHost | None = None
        self._serving_sampling_signature = None
        self._serving_virtual_slots: dict[int, _ServingVirtualSlot] = {}
        self._serving_preempted_sampling: dict[object, tuple[tuple | None, tuple[object, ...] | None]] = {}
        self._serving_virtual_decode_poisoned = False
        self._serving_program_cache_initialization_uncertain = False

    @classmethod
    def get_max_tokens_all_users(cls, **kwargs) -> int:
        return int(kwargs.get("max_seq_len", Qwen38FullModel.max_context_len))

    # ------------------------------------------------------------- low level

    def allocate_batch_state(
        self,
        prompt_lens,
        *,
        request_ids=None,
        page_table=None,
        active_mask=None,
    ) -> Qwen38BatchState:
        if self.model._trace_ready:
            self.model.release_decode_traces()
        self.state = self.model.new_batch_state(
            prompt_lens,
            request_ids=request_ids,
            page_table=page_table,
            active_mask=active_mask,
        )
        return self.state

    def _snapshot_live_trace_program_cache(self) -> int | None:
        """Snapshot exact compiled workloads before eager virtual prefill."""

        if not getattr(self.model, "_trace_ready", False):
            return None
        if getattr(self, "_serving_program_cache_initialization_uncertain", False):
            # A failed first enqueue can leave a cached MeshWorkload whose
            # kernel-binary allocation is still lazy.  Its later cache hit
            # would not change num_program_cache_entries(), so the delta alone
            # cannot protect a live trace.  This rare fail-stop mode releases
            # before eager work for the rest of this generator lifetime.
            self._invalidate_live_prefill_trace()
            return None
        return int(self.model.mesh_device.num_program_cache_entries())

    def _invalidate_live_prefill_trace(self) -> bool:
        if not getattr(self.model, "_trace_ready", False):
            return False
        self.model.release_decode_traces()
        self.model._virtual_prefill_trace_invalidations = (
            int(getattr(self.model, "_virtual_prefill_trace_invalidations", 0)) + 1
        )
        return True

    def _finish_live_trace_prefill(self, entries_before: int | None, *, failed: bool) -> bool:
        """Prevent a live decode trace from replaying over younger workloads.

        Prefill is synchronous at this API boundary and never replays a decode
        trace.  A new TTNN workload may allocate persistent program binaries
        while an older trace is live.  Keep a cache-hit trace, but release it
        before returning when the exact device program cache grew.  The next
        decode recaptures against the same stable physical-B1 state.
        """

        if failed:
            self._serving_program_cache_initialization_uncertain = True
        if entries_before is None or not getattr(self.model, "_trace_ready", False):
            return False
        entries_after = int(self.model.mesh_device.num_program_cache_entries())
        if not failed and entries_after == entries_before:
            return False
        return self._invalidate_live_prefill_trace()

    @staticmethod
    def _logical_rows(unpadded_batch_size, *values) -> int:
        if unpadded_batch_size is None:
            for value in values:
                if value is not None:
                    return len(value)
            raise ValueError("virtual serving requires an explicit real-row count")
        rows = int(unpadded_batch_size)
        if rows < 1:
            raise ValueError("unpadded_batch_size must be positive")
        return rows

    def _validate_virtual_rows(
        self,
        *,
        rows: int,
        request_ids,
        state_slot_ids,
        state_slot_generations,
    ) -> tuple[tuple[object, ...], tuple[int, ...], tuple[int, ...]]:
        if self.model.max_batch != 1:
            raise RuntimeError("virtual-slot microbatch requires the physical-B1 canonical trace")
        requests = tuple(request_ids or ())
        slots = tuple(int(value) for value in (state_slot_ids or ()))
        generations = tuple(int(value) for value in (state_slot_generations or (0,) * rows))
        if not (len(requests) == len(slots) == len(generations) == rows):
            raise ValueError("request_ids, state_slot_ids, and generations must name every real row exactly once")
        if len(set(requests)) != rows:
            raise ValueError("active request IDs must be unique")
        if len(set(slots)) != rows:
            raise ValueError("active virtual state slots must be unique")
        capacity = int(getattr(self.model, "virtual_slot_capacity", 1))
        if any(slot < 0 or slot >= capacity for slot in slots):
            raise ValueError(f"virtual state slot is outside allocated capacity {capacity}")
        return requests, slots, generations

    @staticmethod
    def _sampling_row(sampling_params, row: int):
        if sampling_params is None:
            return None

        def take(name: str, default):
            value = getattr(sampling_params, name, default)
            if value is None:
                return default
            if isinstance(value, torch.Tensor):
                flat = value.reshape(-1)
                return flat[0 if flat.numel() == 1 else row].item()
            if isinstance(value, (list, tuple)):
                return value[0 if len(value) == 1 else row]
            return value

        num_logprobs = take("num_logprobs", -2)
        return SimpleNamespace(
            temperature=[take("temperature", 1.0)],
            top_k=[take("top_k", 1)],
            top_p=[take("top_p", 0.0)],
            presence_penalty=[take("presence_penalty", 0.0)],
            frequency_penalty=[take("frequency_penalty", 0.0)],
            repetition_penalty=[take("repetition_penalty", 1.0)],
            seed=[take("seed", None)],
            num_logprobs=[num_logprobs],
            enable_log_probs=[take("enable_log_probs", num_logprobs >= 0)],
        )

    def _claim_virtual_slot(self, slot_id: int, request_id: object, external_generation: int) -> _ServingVirtualSlot:
        current = self._serving_virtual_slots.get(slot_id)
        if current is not None:
            if current.request_id == request_id and current.external_generation == external_generation:
                return current
            if external_generation <= current.external_generation:
                raise RuntimeError(
                    f"stale virtual prefill owner for slot {slot_id}: current "
                    f"({current.request_id!r}, {current.external_generation}), got "
                    f"({request_id!r}, {external_generation})"
                )
            self.model.release_virtual_slot(
                slot_id,
                current.request_id,
                generation=current.lease.generation,
            )
            current.pending_token_host = None
        lease = self.model.assign_virtual_slot(slot_id, request_id)
        resumed = getattr(self, "_serving_preempted_sampling", {}).pop(request_id, None)
        current = _ServingVirtualSlot(
            request_id=request_id,
            external_generation=external_generation,
            lease=lease,
            sampling_signature=None if resumed is None else resumed[0],
            resumed_sampling_rng_state=None if resumed is None else resumed[1],
        )
        self._serving_virtual_slots[slot_id] = current
        return current

    def _require_virtual_slot(self, slot_id: int, request_id: object, external_generation: int) -> _ServingVirtualSlot:
        current = self._serving_virtual_slots.get(slot_id)
        if current is None:
            raise RuntimeError(f"virtual state slot {slot_id} was not initialized by prefill")
        if current.request_id != request_id or current.external_generation != external_generation:
            raise RuntimeError(
                f"stale virtual state owner for slot {slot_id}: expected "
                f"({current.request_id!r}, {current.external_generation}), got "
                f"({request_id!r}, {external_generation})"
            )
        return current

    def _attach_virtual_token_host(self, output: _ServingDecodeOutput, host: _ServingDecodeHost) -> None:
        """Publish one already-enqueued plugin read for the next exact PLE lookup."""

        if output.kind != "tokens" or not output.slot_keys:
            return
        if len(output.slot_keys) != host.rows:
            raise RuntimeError("deferred token rows do not match virtual-slot ownership keys")
        for row, (slot_id, request_id, external_generation) in enumerate(output.slot_keys):
            current = self._serving_virtual_slots.get(slot_id)
            if current is None:
                continue
            if current.request_id == request_id and current.external_generation == external_generation:
                current.pending_token_host = (host, row)

    def _attach_direct_token_host(self, output: _ServingDecodeOutput, host: _ServingDecodeHost) -> None:
        """Publish the batch-one async read for the next PLE lookup.

        vLLM already enqueues a compact read of every sampled token.  Reusing
        that exact host object avoids a second synchronous D2H read from the
        persistent token tensor before the next trace replay.
        """

        if output.kind == "tokens" and not output.slot_keys and output.rows == 1:
            self._serving_pending_token_host = host

    def _consume_direct_ple_token(self, state: Qwen38BatchState) -> torch.Tensor:
        pending = self._serving_pending_token_host
        if pending is not None:
            self._serving_pending_token_host = None
            values = pending.to_torch(self.model, is_tokens=True)
            self.async_feedback_host_reuses = getattr(self, "async_feedback_host_reuses", 0) + 1
            return values.reshape(-1)[:1]
        self.async_feedback_device_fallbacks = getattr(self, "async_feedback_device_fallbacks", 0) + 1
        return self.model.sampled_tokens_to_torch(state.token_input, state)

    def _consume_virtual_ple_token(self, virtual: _ServingVirtualSlot, state: Qwen38BatchState) -> torch.Tensor:
        """Reuse the plugin's compact D2H instead of issuing a duplicate read."""

        pending = virtual.pending_token_host
        if pending is not None:
            virtual.pending_token_host = None
            host, row = pending
            values = host.to_torch(self.model, is_tokens=True)
            if not 0 <= row < values.numel():
                raise RuntimeError("deferred virtual token row is outside the compact host result")
            self.async_feedback_host_reuses = getattr(self, "async_feedback_host_reuses", 0) + 1
            return values.reshape(-1)[row : row + 1]
        self.async_feedback_device_fallbacks = getattr(self, "async_feedback_device_fallbacks", 0) + 1
        return self.model.sampled_tokens_to_torch(state.token_input, state)

    def _preflight_virtual_prefill_claims(
        self,
        requests: tuple[object, ...],
        slots: tuple[int, ...],
        generations: tuple[int, ...],
    ) -> None:
        """Reject every stale/conflicting claim before any physical state mutates."""

        by_request = {current.request_id: slot for slot, current in self._serving_virtual_slots.items()}
        for request_id, slot_id, generation in zip(requests, slots, generations):
            current = self._serving_virtual_slots.get(slot_id)
            if current is not None and not (
                current.request_id == request_id and current.external_generation == generation
            ):
                if generation <= current.external_generation:
                    raise RuntimeError(
                        f"stale virtual prefill owner for slot {slot_id}: current "
                        f"({current.request_id!r}, {current.external_generation}), got "
                        f"({request_id!r}, {generation})"
                    )
            other_slot = by_request.get(request_id)
            if other_slot is not None and other_slot != slot_id:
                raise RuntimeError(
                    f"request {request_id!r} already owns virtual slot {other_slot}, " f"not requested slot {slot_id}"
                )

    def release_virtual_slots(self, released_state_slots) -> None:
        """Apply scheduler releases inside the live-trace allocation guard."""

        trace_program_cache_entries = self._snapshot_live_trace_program_cache()
        try:
            self._release_virtual_slots_impl(released_state_slots)
        except Exception:
            self._finish_live_trace_prefill(trace_program_cache_entries, failed=True)
            raise
        self._finish_live_trace_prefill(trace_program_cache_entries, failed=False)

    def _release_virtual_slots_impl(self, released_state_slots) -> None:
        """Release scheduler slots without opening a second guard scope."""

        for release in released_state_slots or ():
            if len(release) not in (3, 4):
                raise ValueError("released virtual state slot must be (request_id, slot_id, generation[, reason])")
            request_id, raw_slot, raw_generation = release[:3]
            reason = release[3] if len(release) == 4 else "finished"
            slot_id = int(raw_slot)
            external_generation = int(raw_generation)
            current = self._serving_virtual_slots.get(slot_id)
            if current is None:
                continue
            # A deferred release from an older lifetime must never tear down a
            # slot already reused by a new request/generation.
            if external_generation < current.external_generation:
                continue
            if current.request_id != request_id or current.external_generation != external_generation:
                raise RuntimeError(
                    f"release does not match virtual slot {slot_id} owner: expected "
                    f"({current.request_id!r}, {current.external_generation}), got "
                    f"({request_id!r}, {external_generation})"
                )
            if reason == "preempted":
                export_rng = getattr(self.model, "export_virtual_slot_sampling_rng_state", None)
                rng_state = (
                    export_rng(
                        slot_id,
                        request_id,
                        generation=current.lease.generation,
                    )
                    if callable(export_rng)
                    else None
                )
                preempted = getattr(self, "_serving_preempted_sampling", None)
                if preempted is None:
                    preempted = {}
                    self._serving_preempted_sampling = preempted
                preempted[request_id] = (current.sampling_signature, rng_state)
            else:
                getattr(self, "_serving_preempted_sampling", {}).pop(request_id, None)
            self.model.release_virtual_slot(
                slot_id,
                request_id,
                generation=current.lease.generation,
            )
            current.pending_token_host = None
            del self._serving_virtual_slots[slot_id]
        if not self._serving_virtual_slots:
            self._serving_virtual_decode_poisoned = False

    def compile_prefill(
        self,
        *,
        tokens,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        start_pos=None,
        empty_slots=None,
        sampling_params=None,
        **_kwargs,
    ) -> None:
        """Readiness helper hook.

        This model keeps page/token buffers model-owned; the attention cache is
        either standalone model state or the exact object adopted from vLLM.
        Compilation occurs on the first concrete call.  The hook validates the
        same fixed-slot shape contract and records sampler parameters without
        running a hidden CPU or replicated fallback path.
        """

        del start_pos, empty_slots
        self.model._require_kv_cache_identity(kv_cache)
        token_tensor = torch.as_tensor(tokens, dtype=torch.int64, device="cpu")
        if token_tensor.ndim == 1:
            token_tensor = token_tensor.unsqueeze(0)
        if token_tensor.ndim != 2 or token_tensor.shape[0] != self.model.max_batch:
            raise ValueError(f"compile_prefill tokens must be [{self.model.max_batch}, seq]")
        if page_table is not None:
            pages = torch.as_tensor(page_table, dtype=torch.int32, device="cpu")
            if tuple(pages.shape) != tuple(self.model._default_page_table_host.shape):
                raise ValueError("compile_prefill page_table shape does not match active attention-cache blocks")
        if prompt_lens is not None:
            lengths = torch.as_tensor(prompt_lens, dtype=torch.int32).reshape(-1)
            if int(lengths.numel()) != self.model.max_batch:
                raise ValueError("compile_prefill prompt_lens must match fixed slots")
        if sampling_params is not None:
            # Same bridge as serving: greedy normalization, penalty/logprob
            # validation and entropy seeds for unseeded stochastic requests.
            self._apply_serving_sampling_params(sampling_params, reset_seed=True)

    def compile_decode(
        self,
        *,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        sampling_params=None,
        reset_batch: bool = False,
        **_kwargs,
    ) -> None:
        """Readiness helper hook for explicit one-token decode state."""

        del reset_batch
        self.model._require_kv_cache_identity(kv_cache)
        torch.as_tensor(tokens, dtype=torch.int64, device="cpu").reshape(self.model.max_batch)
        torch.as_tensor(start_pos, dtype=torch.int32, device="cpu").reshape(self.model.max_batch)
        if page_table is not None:
            pages = torch.as_tensor(page_table, dtype=torch.int32, device="cpu")
            if tuple(pages.shape) != tuple(self.model._default_page_table_host.shape):
                raise ValueError("compile_decode page_table shape does not match model-owned KV blocks")
        if sampling_params is not None:
            # Same bridge as serving: greedy normalization, penalty/logprob
            # validation and entropy seeds for unseeded stochastic requests.
            self._apply_serving_sampling_params(sampling_params, reset_seed=True)

    def prefill_forward(
        self,
        tokens,
        *,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        start_pos=None,
        intermediate_prefill_mask=None,
        state: Qwen38BatchState | None = None,
        request_ids=None,
        active_mask=None,
        empty_slots=None,
        state_slot_ids=None,
        state_slot_generations=None,
        unpadded_batch_size=None,
        released_state_slots=None,
        read_from_device: bool = True,
        return_all_logits: bool = False,
        on_device_sampling: bool = False,
        sampling_params=None,
        **_kwargs,
    ):
        """Low-level mixed-prompt prefill with explicit cache/page state.

        ``empty_slots`` is accepted for common harness compatibility.  This
        port uses a fixed cohort: supplied slots must be the active slot ids.
        """

        if state_slot_ids is not None:
            trace_program_cache_entries = self._snapshot_live_trace_program_cache()
            try:
                self._release_virtual_slots_impl(released_state_slots)
                result = self._prefill_forward_virtual(
                    tokens,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    prompt_lens=prompt_lens,
                    start_pos=start_pos,
                    intermediate_prefill_mask=intermediate_prefill_mask,
                    request_ids=request_ids,
                    state_slot_ids=(empty_slots if state_slot_ids is None else state_slot_ids),
                    state_slot_generations=state_slot_generations,
                    unpadded_batch_size=unpadded_batch_size,
                    read_from_device=read_from_device,
                    return_all_logits=return_all_logits,
                    on_device_sampling=on_device_sampling,
                    sampling_params=sampling_params,
                    _manage_trace_program_cache=False,
                )
            except Exception:
                self._finish_live_trace_prefill(trace_program_cache_entries, failed=True)
                raise
            self._finish_live_trace_prefill(trace_program_cache_entries, failed=False)
            return result

        token_tensor = torch.as_tensor(tokens, dtype=torch.int64, device="cpu")
        if token_tensor.ndim == 1:
            token_tensor = token_tensor.unsqueeze(0)
        if prompt_lens is None:
            prompt_lens = torch.full((token_tensor.shape[0],), token_tensor.shape[1], dtype=torch.int32)
        lengths = torch.as_tensor(prompt_lens, dtype=torch.int32).reshape(-1)
        if token_tensor.shape[0] != self.model.max_batch:
            raise ValueError(
                f"generator model owns {self.model.max_batch} fixed slots; got batch {token_tensor.shape[0]}"
            )
        starts = (
            torch.zeros_like(lengths)
            if start_pos is None
            else torch.as_tensor(start_pos, dtype=torch.int32, device="cpu").reshape(-1)
        )
        if starts.numel() != self.model.max_batch or lengths.numel() != self.model.max_batch:
            raise ValueError("prefill lengths and start_pos must match the physical batch")
        if state is None:
            # Reuse the trace-bound physical state across vLLM prefill steps
            # and across sequential batch-one requests.  Position zero below
            # performs the explicit reset for a new request.
            state = self.state
        intermediate = (
            torch.zeros_like(lengths, dtype=torch.bool)
            if intermediate_prefill_mask is None
            else torch.as_tensor(intermediate_prefill_mask, dtype=torch.bool, device="cpu").reshape(-1)
        )
        if intermediate.numel() != self.model.max_batch:
            raise ValueError("intermediate_prefill_mask must match the physical batch")
        requests = tuple(range(self.model.max_batch)) if request_ids is None else tuple(request_ids)
        if len(requests) != self.model.max_batch:
            raise ValueError("request_ids must match the physical batch")

        if bool(torch.all(starts == 0)):
            if state is None:
                state = self.allocate_batch_state(
                    lengths,
                    request_ids=requests,
                    page_table=page_table,
                    active_mask=active_mask,
                )
            else:
                # Keep the live decode trace across sequential batch-one
                # requests: the trace binds the fixed state buffers and caches,
                # which a new request only rewrites in place.  It is released
                # below only when this prefill compiles new programs (the same
                # guard the virtual-slot path uses); an unconditional release
                # cost a ~0.6 s recapture on every request's first decode.
                if os.environ.get("QWEN38_RELEASE_TRACE_PER_REQUEST", "0") == "1" and self.model._trace_ready:
                    self.model.release_decode_traces()
                state.prompt_lens = lengths.clone()
                state.active_mask = lengths > 0
                state.request_ids = requests
                if page_table is not None:
                    pages = torch.as_tensor(page_table, dtype=torch.int32, device="cpu")
                    if tuple(pages.shape) != tuple(state.page_table_host.shape):
                        raise ValueError("page-table shape cannot change across batch-one requests")
                    state.page_table_host = pages.clone()
                self.model.reset_batch_state(state)
            page_table = None
        else:
            if state is None:
                raise ValueError("chunked prefill continuation requires an existing request state")
            if state.request_ids != requests:
                raise ValueError("chunked prefill continuation changed request identity")
            committed = state.computed_lens
            if committed is None or not torch.equal(starts, committed):
                expected = None if committed is None else committed.tolist()
                raise ValueError(
                    f"chunked prefill starts {starts.tolist()} do not match committed offsets {expected}"
                )
            state.prompt_lens = lengths.clone()
            state.active_mask = lengths > 0

        if empty_slots is not None and tuple(int(value) for value in empty_slots) != state.active_slots:
            raise ValueError("empty_slots must match the active fixed-slot cohort")
        is_final_chunk = not bool(torch.any(intermediate))
        emit_logits = not (on_device_sampling and not is_final_chunk)
        if sampling_params is not None and is_final_chunk:
            self._apply_serving_sampling_params(sampling_params, reset_seed=True)
            on_device_sampling = True
        trace_program_cache_entries = self._snapshot_live_trace_program_cache()
        try:
            logits = self.model.prefill_forward(
                token_tensor,
                state=state,
                prompt_lens=lengths,
                start_pos=starts,
                is_final_chunk=is_final_chunk,
                emit_logits=emit_logits,
                page_table=page_table,
                kv_cache=kv_cache,
                return_all_logits=return_all_logits,
            )
        except Exception:
            self._finish_live_trace_prefill(trace_program_cache_entries, failed=True)
            raise
        self._finish_live_trace_prefill(trace_program_cache_entries, failed=False)
        self.state = state
        if not is_final_chunk and on_device_sampling:
            self._serving_device_feedback_current = False
            if not read_from_device:
                return state.token_input
            return torch.zeros(self.model.max_batch, dtype=torch.int64)
        if on_device_sampling:
            assert logits is not None
            sampled = self.model.sample_logits(logits, state)
            ttnn.deallocate(logits)
            self._serving_device_feedback_current = True
            if not read_from_device:
                return sampled
            return self.model.sampled_tokens_to_torch(sampled, state)
        assert logits is not None
        self._serving_device_feedback_current = False
        if not read_from_device:
            return logits
        host = self.model.logits_to_torch(logits)
        ttnn.deallocate(logits)
        return host.reshape(self.model.max_batch, 1, self.model.vocab_size)

    def _prefill_forward_virtual(
        self,
        tokens,
        *,
        page_table,
        kv_cache,
        prompt_lens,
        start_pos=None,
        intermediate_prefill_mask=None,
        request_ids,
        state_slot_ids,
        state_slot_generations,
        unpadded_batch_size,
        read_from_device: bool,
        return_all_logits: bool,
        on_device_sampling: bool,
        sampling_params,
        _manage_trace_program_cache: bool = True,
    ):
        """Prefill real rows into stable virtual slots through physical B1."""

        self.model._require_kv_cache_identity(kv_cache)
        if getattr(self, "_serving_virtual_decode_poisoned", False):
            raise RuntimeError(
                "virtual execution cohort is fail-stopped after a device execution "
                "error; release its request lifetimes before retrying"
            )
        token_tensor = torch.as_tensor(tokens, dtype=torch.int64, device="cpu")
        if token_tensor.ndim == 1:
            token_tensor = token_tensor.unsqueeze(0)
        if token_tensor.ndim != 2:
            raise ValueError("virtual prefill tokens must be [real_rows, padded_prompt_width]")
        rows = self._logical_rows(unpadded_batch_size, request_ids, state_slot_ids, prompt_lens)
        requests, slots, generations = self._validate_virtual_rows(
            rows=rows,
            request_ids=request_ids,
            state_slot_ids=state_slot_ids,
            state_slot_generations=state_slot_generations,
        )
        if token_tensor.shape[0] < rows:
            raise ValueError("virtual prefill token rows are shorter than unpadded_batch_size")
        lengths = torch.as_tensor(prompt_lens, dtype=torch.int32).reshape(-1)
        if lengths.numel() != rows:
            raise ValueError("virtual prefill prompt_lens must name every real row")
        starts = (
            torch.zeros(rows, dtype=torch.int32)
            if start_pos is None
            else torch.as_tensor(start_pos, dtype=torch.int32).reshape(-1)
        )
        if starts.numel() != rows:
            raise ValueError("virtual prefill start_pos must name every real row")
        if bool(torch.any(starts < 0)) or bool(torch.any(starts > lengths)):
            raise ValueError("virtual prefill requires 0 <= start_pos <= prompt_lens")
        intermediate = (
            torch.zeros(rows, dtype=torch.bool)
            if intermediate_prefill_mask is None
            else torch.as_tensor(intermediate_prefill_mask, dtype=torch.bool).reshape(-1)
        )
        if intermediate.numel() != rows:
            raise ValueError("intermediate_prefill_mask must name every real row")
        if int(lengths.max()) > token_tensor.shape[1]:
            raise ValueError("virtual prefill token width is shorter than the chunk end")
        pages = None if page_table is None else torch.as_tensor(page_table, dtype=torch.int32, device="cpu")
        if pages is not None:
            if pages.ndim != 2 or pages.shape[0] < rows:
                raise ValueError("virtual prefill page-table rows are shorter than unpadded_batch_size")
            expected_pages = self.model._default_page_table_host
            if tuple(pages.shape[1:]) != tuple(expected_pages.shape[1:]):
                raise ValueError("virtual prefill page-table width does not match the physical state")
        self._preflight_virtual_prefill_claims(requests, slots, generations)
        for request_id, slot_id, generation, start in zip(requests, slots, generations, starts.tolist()):
            if start > 0:
                self._require_virtual_slot(slot_id, request_id, generation)
        if on_device_sampling:
            for row in range(rows):
                if not bool(intermediate[row]):
                    self._validate_serving_sampling_params(self._sampling_row(sampling_params, row))

        multi_host_compatibility = rows > 1 and not on_device_sampling
        if multi_host_compatibility:
            # The plugin selects host sampling for the complete cohort when any
            # row needs logprobs or a host-only logits processor.  Each physical
            # B1 prefill reuses the same output storage, so materialize a row
            # before the next row can overwrite it.  The shared vLLM sampler,
            # not this generator, owns all host sampling policy.
            self.host_sampling_compatibility_calls += 1
        trace_program_cache_entries = self._snapshot_live_trace_program_cache() if _manage_trace_program_cache else None
        outputs = []
        started_rows = 0
        try:
            for row, (request_id, slot_id, generation) in enumerate(zip(requests, slots, generations)):
                started_rows += 1
                virtual = self._claim_virtual_slot(slot_id, request_id, generation)
                virtual.lease = self.model.begin_virtual_prefill(
                    slot_id,
                    request_id,
                    generation=virtual.lease.generation,
                )
                row_pages = (
                    self.model._default_page_table_host.clone() if pages is None else pages[row : row + 1].clone()
                )
                if self.state is None:
                    self.state = self.model.new_batch_state(
                        [int(lengths[row])],
                        request_ids=(request_id,),
                        page_table=row_pages,
                    )
                else:
                    # Keep the exact trace-bound object and fixed device addresses.
                    self.state.prompt_lens = lengths[row : row + 1].clone()
                    self.state.active_mask = torch.ones(1, dtype=torch.bool)
                    self.state.request_ids = (request_id,)
                    self.state.page_table_host = row_pages
                    self.model.reset_batch_state(self.state)
                logits = self.model.prefill_forward(
                    token_tensor[row : row + 1, : int(lengths[row])],
                    state=self.state,
                    prompt_lens=lengths[row : row + 1],
                    kv_cache=kv_cache,
                    return_all_logits=return_all_logits,
                )
                row_params = self._sampling_row(sampling_params, row)
                row_samples = bool(on_device_sampling and not intermediate[row])
                if row_samples:
                    self._apply_serving_sampling_params(row_params, reset_seed=True)
                    virtual.sampling_signature = self._serving_sampling_signature
                    restore_rng = getattr(self.model, "restore_virtual_slot_sampling_rng_state", None)
                    if virtual.resumed_sampling_rng_state is not None and callable(restore_rng):
                        restore_rng(
                            slot_id,
                            request_id,
                            virtual.resumed_sampling_rng_state,
                            generation=virtual.lease.generation,
                        )
                        virtual.resumed_sampling_rng_state = None
                    output = self.model.sample_logits(logits, self.state)
                    ttnn.deallocate(logits)
                    virtual.device_feedback_current = True
                elif on_device_sampling:
                    # Intermediate chunks update model/vLLM-owned state but emit no
                    # token.  Return the compact persistent token as a placeholder;
                    # the runner suppresses it and never reads full logits.
                    ttnn.deallocate(logits)
                    output = self.state.token_input
                    virtual.device_feedback_current = False
                else:
                    output = logits
                    virtual.device_feedback_current = False
                virtual.lease = self.model.finish_virtual_prefill(
                    slot_id,
                    request_id,
                    generation=virtual.lease.generation,
                )
                if multi_host_compatibility:
                    try:
                        output = self.model.logits_to_torch(output).reshape(1, 1, self.model.vocab_size)
                    finally:
                        ttnn.deallocate(logits)
                outputs.append(output)
        except Exception:
            # A partially executed admission must not leave an older trace
            # replayable beside any workload it may have compiled.
            if _manage_trace_program_cache:
                self._finish_live_trace_prefill(trace_program_cache_entries, failed=True)
            if started_rows:
                self._serving_virtual_decode_poisoned = True
                self._serving_program_cache_initialization_uncertain = True
            raise

        if _manage_trace_program_cache:
            self._finish_live_trace_prefill(trace_program_cache_entries, failed=False)
        self._serving_device_feedback_current = all(
            self._serving_virtual_slots[slot].device_feedback_current for slot in slots
        )
        if len(outputs) == 1:
            output = outputs[0]
            if not read_from_device:
                return output
            if on_device_sampling:
                if bool(intermediate[0]):
                    return torch.zeros(1, dtype=torch.int64)
                return self.model.sampled_tokens_to_torch(output, self.state)
            host = self.model.logits_to_torch(output)
            ttnn.deallocate(output)
            return host.reshape(1, 1, self.model.vocab_size)
        if multi_host_compatibility:
            return torch.cat(outputs, dim=0)[:rows]
        # Multi-row prefill is uncommon (the shared runner submits one prompt
        # at a time), but preserve deferred compact-token semantics if it occurs.
        devices = tuple(
            (
                torch.zeros(1, dtype=torch.int64)
                if bool(intermediate[row])
                else self.model.virtual_slot_token(
                    slot,
                    request,
                    generation=self._serving_virtual_slots[slot].lease.generation,
                )
            )
            for row, (slot, request) in enumerate(zip(slots, requests))
        )
        if read_from_device:
            return self.process_decode_output_host(
                _ServingDecodeOutput("tokens", devices, rows, self.state),
                is_tokens=True,
            )
        return _ServingDecodeOutput("tokens", devices, rows, self.state)

    @staticmethod
    def _sampling_values(value, name: str, default):
        raw = getattr(value, name, default)
        return default if raw is None else raw

    def _apply_serving_sampling_params(self, sampling_params, *, reset_seed: bool = False) -> None:
        """Apply the subset implemented by the canonical full-model sampler."""

        presence = torch.as_tensor(self._sampling_values(sampling_params, "presence_penalty", 0.0))
        frequency = torch.as_tensor(self._sampling_values(sampling_params, "frequency_penalty", 0.0))
        repetition = torch.as_tensor(self._sampling_values(sampling_params, "repetition_penalty", 1.0))
        enable_log_probs = torch.as_tensor(self._sampling_values(sampling_params, "enable_log_probs", False))
        if bool(torch.any(presence != 0)) or bool(torch.any(frequency != 0)) or bool(torch.any(repetition != 1)):
            raise ValueError("Qwen3.8 device sampling requires default penalties; use explicit host compatibility")
        if bool(torch.any(enable_log_probs)):
            raise ValueError("Qwen3.8 device sampling does not expose logprobs; use explicit host compatibility")

        top_k = self._sampling_values(sampling_params, "top_k", 1)
        top_p = self._sampling_values(sampling_params, "top_p", 0.0)
        temperature = self._sampling_values(sampling_params, "temperature", 1.0)
        # vLLM encodes greedy as temperature=0 with unrestricted top-k/top-p.
        # The canonical TT sampler encodes the same policy as forced argmax.
        # Normalize the representation without introducing an adapter sampler.
        temperature_tensor = torch.as_tensor(temperature)
        if bool(torch.all(temperature_tensor == 0)):
            top_k = torch.ones_like(torch.as_tensor(top_k), dtype=torch.int32)
            top_p = torch.zeros_like(torch.as_tensor(top_p), dtype=torch.float32)
            temperature = torch.ones_like(temperature_tensor, dtype=torch.float32)
        seed = self._sampling_values(sampling_params, "seed", None)
        if seed is None:
            raw_seed_values = [None] * self.model.max_batch
        elif isinstance(seed, (list, tuple)):
            raw_seed_values = [None if value is None else int(value) for value in seed]
        else:
            raw_seed_values = [int(seed)]
        signature = (
            tuple(torch.as_tensor(top_k).reshape(-1).tolist()),
            tuple(torch.as_tensor(top_p).reshape(-1).tolist()),
            tuple(torch.as_tensor(temperature).reshape(-1).tolist()),
            tuple(raw_seed_values),
        )
        if signature == self._serving_sampling_signature and not reset_seed:
            return
        force_argmax = bool(torch.all(torch.as_tensor(top_k) == 1))
        if force_argmax:
            seed_values = None
        else:
            seed_values = [secrets.randbelow(0x7FFFFFFE) + 1 if value is None else value for value in raw_seed_values]
        self.model.set_sampling_params(
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            seeds=seed_values,
        )
        self._serving_sampling_signature = signature

    def _validate_serving_sampling_params(self, sampling_params) -> None:
        """Pure validation used to make all row-local failures pre-execution."""

        presence = torch.as_tensor(self._sampling_values(sampling_params, "presence_penalty", 0.0))
        frequency = torch.as_tensor(self._sampling_values(sampling_params, "frequency_penalty", 0.0))
        repetition = torch.as_tensor(self._sampling_values(sampling_params, "repetition_penalty", 1.0))
        enable_log_probs = torch.as_tensor(self._sampling_values(sampling_params, "enable_log_probs", False))
        if bool(torch.any(presence != 0)) or bool(torch.any(frequency != 0)) or bool(torch.any(repetition != 1)):
            raise ValueError("Qwen3.8 device sampling requires default penalties; use explicit host compatibility")
        if bool(torch.any(enable_log_probs)):
            raise ValueError("Qwen3.8 device sampling does not expose logprobs; use explicit host compatibility")
        top_k = torch.as_tensor(self._sampling_values(sampling_params, "top_k", 1))
        top_p = torch.as_tensor(self._sampling_values(sampling_params, "top_p", 0.0))
        temperature = torch.as_tensor(self._sampling_values(sampling_params, "temperature", 1.0))
        greedy = temperature == 0
        normalized_k = torch.where(greedy, torch.ones_like(top_k), top_k)
        if bool(torch.any((normalized_k < 1) | (normalized_k > 32))):
            raise ValueError("top_k must be in [1, 32] for Qwen3.8 device sampling")
        if bool(torch.any((top_p < 0) | (top_p > 1))):
            raise ValueError("top_p must be in [0, 1]")
        if bool(torch.any((temperature < 0) | torch.isnan(temperature))):
            raise ValueError("temperature must be finite and non-negative")

    def decode_forward(
        self,
        tokens,
        start_pos=None,
        *,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        state: Qwen38BatchState | None = None,
        active_mask=None,
        request_ids=None,
        read_from_device: bool = True,
        enable_trace: bool = True,
        on_device_sampling: bool = False,
        host_sampling_compatibility: bool | None = None,
        sampling_params=None,
        reset_batch: bool = False,
        slot_remap=None,
        state_slot_ids=None,
        state_slot_generations=None,
        unpadded_batch_size=None,
        released_state_slots=None,
        serving_mode: bool = False,
        prompt_tokens=None,
        output_tokens=None,
        **_kwargs,
    ):
        """Low-level decode used by serving and common logit-based tests.

        Passing ``start_pos`` selects the explicit host-input compatibility
        path: token/position buffers are refreshed, but all model compute is
        still replayed from TT traces.  The optimized token-out path calls
        :meth:`decode_token_out` and never refreshes either buffer per token.
        """

        state = self.state if state is None else state
        if state is None:
            raise RuntimeError("prefill/allocate_batch_state must run before decode")
        if serving_mode:
            del prompt_tokens, output_tokens
            self.release_virtual_slots(released_state_slots)
            return self._decode_forward_serving(
                tokens,
                start_pos=start_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                state=state,
                read_from_device=read_from_device,
                enable_trace=enable_trace,
                sampling_params=sampling_params,
                reset_batch=reset_batch,
                slot_remap=slot_remap,
                request_ids=request_ids,
                state_slot_ids=state_slot_ids,
                state_slot_generations=state_slot_generations,
                unpadded_batch_size=unpadded_batch_size,
            )
        if page_table is not None:
            self.model.update_page_table(state, page_table)
            page_table = None
        values = torch.as_tensor(tokens, dtype=torch.int64, device="cpu").reshape(self.model.max_batch)
        compatibility = start_pos is not None if host_sampling_compatibility is None else host_sampling_compatibility
        if compatibility:
            self.host_sampling_compatibility_calls += 1
            self.model.copy_tokens(state, values)
            if start_pos is not None:
                positions = torch.as_tensor(start_pos, dtype=torch.int32).reshape(self.model.max_batch)
                host = ttnn.from_torch(positions, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
                ttnn.copy_host_to_device_tensor(host, state.current_pos)
                state.position_host_copies += 1
            if enable_trace:
                logits = self.model.replay_model_only_traced(state, values.reshape(-1, 1))
                self.model.advance_positions_traced()
            else:
                logits = self.model.decode_forward(
                    values,
                    state=state,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    prompt_lens=prompt_lens,
                    active_mask=active_mask,
                    request_ids=request_ids,
                    enable_trace=False,
                    on_device_sampling=False,
                )
                ttnn.plus_one(state.current_pos, skip_negative_entries=True)
            if not read_from_device:
                return logits, None
            host_logits = self.model.logits_to_torch(logits)
            if not enable_trace:
                ttnn.deallocate(logits)
            return host_logits, None

        if enable_trace:
            logits, sampled = self.model.decode_token_out_traced(state, values.reshape(-1, 1))
        else:
            sampled = self.model.decode_forward(
                values,
                state=state,
                page_table=page_table,
                kv_cache=kv_cache,
                prompt_lens=prompt_lens,
                active_mask=active_mask,
                request_ids=request_ids,
                enable_trace=False,
                on_device_sampling=True,
            )
            logits = None
        if on_device_sampling:
            if read_from_device:
                return self.model.sampled_tokens_to_torch(sampled, state), None
            return sampled, None
        if logits is None:
            raise ValueError("eager token-out decode does not retain logits; request on_device_sampling")
        if not read_from_device:
            return logits, None
        return self.model.logits_to_torch(logits), None

    def _decode_forward_serving(
        self,
        tokens,
        *,
        start_pos,
        page_table,
        kv_cache,
        state: Qwen38BatchState,
        read_from_device: bool,
        enable_trace: bool,
        sampling_params,
        reset_batch: bool,
        slot_remap,
        request_ids=None,
        state_slot_ids=None,
        state_slot_generations=None,
        unpadded_batch_size=None,
    ):
        """Canonical vLLM decode without a Python token-feedback writeback."""

        self.model._require_kv_cache_identity(kv_cache)
        if state_slot_ids is not None:
            if slot_remap is not None:
                raise ValueError("virtual state slots use stable IDs and never a physical gather/remap")
            return self._decode_virtual_slots(
                tokens,
                start_pos=start_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                state=state,
                read_from_device=read_from_device,
                enable_trace=enable_trace,
                sampling_params=sampling_params,
                reset_batch=reset_batch,
                request_ids=request_ids,
                state_slot_ids=state_slot_ids,
                state_slot_generations=state_slot_generations,
                unpadded_batch_size=unpadded_batch_size,
            )
        if slot_remap is not None:
            remap = torch.as_tensor(slot_remap, dtype=torch.int64).reshape(-1)
            identity = torch.arange(remap.numel(), dtype=remap.dtype)
            if not torch.equal(remap, identity):
                raise ValueError("the traced batch-one serving path cannot remap a live request slot")
        if page_table is not None:
            self.model.update_page_table(state, page_table)

        values = torch.as_tensor(tokens, dtype=torch.int64, device="cpu").reshape(self.model.max_batch)
        device_sampling = sampling_params is not None
        if device_sampling:
            self._apply_serving_sampling_params(sampling_params)
            # A reset/layout transition is drained by the runner, so its compact
            # host token is current and can feed the declared PLE lookup.  During
            # steady async decode the runner's token may be stale; read the
            # sampler's persistent token solely for that lookup.  It is never
            # copied back to the device.
            if self._serving_device_feedback_current and not reset_batch:
                ple_values = self._consume_direct_ple_token(state)
            else:
                self._serving_pending_token_host = None
                ple_values = values
            if enable_trace:
                _, output = self.model.decode_token_out_traced(state, ple_values.reshape(-1, 1))
            else:
                output = self.model.decode_forward(
                    ple_values,
                    state=state,
                    enable_trace=False,
                    on_device_sampling=True,
                )
            self._serving_device_feedback_current = True
            if read_from_device:
                return self.model.sampled_tokens_to_torch(output, state)
            return _ServingDecodeOutput("tokens", output, self.model.max_batch, state)

        # Explicit compatibility mode: vLLM samples full logits on the host,
        # making its token/position inputs authoritative for this step.
        self.host_sampling_compatibility_calls += 1
        self._serving_pending_token_host = None
        self.model.copy_tokens(state, values)
        if start_pos is not None:
            positions = torch.as_tensor(start_pos, dtype=torch.int32).reshape(self.model.max_batch)
            host = ttnn.from_torch(positions, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(host, state.current_pos)
            state.position_host_copies += 1
        if enable_trace:
            output = self.model.replay_model_only_traced(state, values.reshape(-1, 1))
            self.model.advance_positions_traced()
        else:
            output = self.model.decode_forward(
                values,
                state=state,
                enable_trace=False,
                on_device_sampling=False,
            )
            ttnn.plus_one(state.current_pos, skip_negative_entries=True)
        self._serving_device_feedback_current = False
        if read_from_device:
            return self.model.logits_to_torch(output).reshape(self.model.max_batch, 1, self.model.vocab_size)
        return _ServingDecodeOutput("logits", output, self.model.max_batch, state)

    def _decode_virtual_slots(
        self,
        tokens,
        *,
        start_pos,
        page_table,
        kv_cache,
        state: Qwen38BatchState,
        read_from_device: bool,
        enable_trace: bool,
        sampling_params,
        reset_batch: bool,
        request_ids,
        state_slot_ids,
        state_slot_generations,
        unpadded_batch_size,
    ):
        """Serialize logical rows through the canonical physical-B1 token-out trace."""

        del reset_batch
        if getattr(self, "_serving_virtual_decode_poisoned", False):
            raise RuntimeError(
                "virtual decode cohort is fail-stopped after a device execution error; "
                "release its request lifetimes before retrying"
            )
        values = torch.as_tensor(tokens, dtype=torch.int64, device="cpu")
        if values.ndim == 1:
            values = values.unsqueeze(1)
        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError("virtual decode tokens must be [wire_rows, 1]")
        rows = self._logical_rows(unpadded_batch_size, request_ids, state_slot_ids)
        requests, slots, generations = self._validate_virtual_rows(
            rows=rows,
            request_ids=request_ids,
            state_slot_ids=state_slot_ids,
            state_slot_generations=state_slot_generations,
        )
        if values.shape[0] < rows:
            raise ValueError("virtual decode token rows are shorter than unpadded_batch_size")
        pages = None if page_table is None else torch.as_tensor(page_table, dtype=torch.int32, device="cpu")
        if pages is not None and pages.shape[0] < rows:
            raise ValueError("virtual decode page-table rows are shorter than unpadded_batch_size")
        if sampling_params is None:
            # Explicit compatibility cohort: the plugin selects host sampling
            # for the whole step when any active request needs logprobs or a
            # host-only logits processor.  Serialize every logical row through
            # the same physical-B1 model-only trace and materialize its logits
            # before the next row can overwrite the trace output buffer.  The
            # all-device cohort below remains the canonical token-out path.
            positions = None
            if start_pos is not None:
                positions = torch.as_tensor(start_pos, dtype=torch.int32).reshape(-1)
                if positions.numel() < rows:
                    raise ValueError("virtual decode positions are shorter than unpadded_batch_size")
            state_pages = getattr(state, "page_table_host", None)
            if pages is not None and state_pages is not None and tuple(pages.shape[1:]) != tuple(state_pages.shape[1:]):
                raise ValueError("virtual decode page-table width does not match the trace-bound state")
            for request_id, slot_id, generation in zip(requests, slots, generations):
                self._require_virtual_slot(slot_id, request_id, generation)

            self.host_sampling_compatibility_calls += 1
            host_logits = []
            executed_rows = 0
            try:
                for row, (request_id, slot_id, generation) in enumerate(zip(requests, slots, generations)):
                    virtual = self._require_virtual_slot(slot_id, request_id, generation)
                    virtual.lease = self.model.activate_virtual_slot(
                        slot_id,
                        request_id,
                        generation=virtual.lease.generation,
                    )
                    state.request_ids = (request_id,)
                    if pages is not None:
                        self.model.update_page_table(state, pages[row : row + 1])
                    self.model.copy_tokens(state, values[row])
                    if positions is not None:
                        position = positions[row : row + 1]
                        host = ttnn.from_torch(position, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
                        ttnn.copy_host_to_device_tensor(host, state.current_pos)
                        state.position_host_copies += 1
                    if enable_trace:
                        output = self.model.replay_model_only_traced(state, values[row : row + 1])
                        self.model.advance_positions_traced()
                    else:
                        output = self.model.decode_forward(
                            values[row : row + 1],
                            state=state,
                            enable_trace=False,
                            on_device_sampling=False,
                        )
                        ttnn.plus_one(state.current_pos, skip_negative_entries=True)
                    executed_rows += 1
                    host_logits.append(self.model.logits_to_torch(output).reshape(1, 1, self.model.vocab_size))
                    virtual.device_feedback_current = False
                    virtual.lease = self.model.commit_virtual_slot(
                        slot_id,
                        request_id,
                        generation=virtual.lease.generation,
                    )
            except Exception:
                if executed_rows:
                    self._serving_virtual_decode_poisoned = True
                    self._serving_program_cache_initialization_uncertain = True
                raise

            self._serving_device_feedback_current = False
            wrapped = _ServingDecodeOutput("logits", torch.cat(host_logits, dim=0), rows, state)
            if read_from_device:
                return self.process_decode_output_host(wrapped, is_tokens=False)
            return wrapped

        # Validate the complete logical batch before the first row can advance
        # recurrent state or RNG. Runtime/device failures after this point are
        # fail-stopped: exact rollback would require N-1 extra device snapshots.
        for row, (request_id, slot_id, generation) in enumerate(zip(requests, slots, generations)):
            self._require_virtual_slot(slot_id, request_id, generation)
            self._validate_serving_sampling_params(self._sampling_row(sampling_params, row))
        state_pages = getattr(state, "page_table_host", None)
        if pages is not None and state_pages is not None and tuple(pages.shape[1:]) != tuple(state_pages.shape[1:]):
            raise ValueError("virtual decode page-table width does not match the trace-bound state")

        executed_rows = 0
        try:
            for row, (request_id, slot_id, generation) in enumerate(zip(requests, slots, generations)):
                virtual = self._require_virtual_slot(slot_id, request_id, generation)
                virtual.lease = self.model.activate_virtual_slot(
                    slot_id,
                    request_id,
                    generation=virtual.lease.generation,
                )
                # Activation restored this slot's device sampler tensors and the
                # canonical model's host RNG state; align the signature cache so
                # an A/B/A row order cannot skip the required parameter upload.
                self._serving_sampling_signature = virtual.sampling_signature
                state.request_ids = (request_id,)
                if pages is not None:
                    self.model.update_page_table(state, pages[row : row + 1])
                row_params = self._sampling_row(sampling_params, row)
                self._apply_serving_sampling_params(row_params)
                virtual.sampling_signature = self._serving_sampling_signature
                # The runner token can lag async device feedback. Read one compact
                # token solely for the declared host PLE lookup; feedback remains
                # in the restored device token buffer.
                if virtual.device_feedback_current:
                    ple_values = self._consume_virtual_ple_token(virtual, state)
                else:
                    ple_values = values[row : row + 1].reshape(1)
                if enable_trace:
                    self.model.decode_token_out_traced(state, ple_values.reshape(1, 1))
                else:
                    self.model.decode_forward(
                        ple_values,
                        state=state,
                        enable_trace=False,
                        on_device_sampling=True,
                    )
                executed_rows += 1
                virtual.device_feedback_current = True
                virtual.lease = self.model.commit_virtual_slot(
                    slot_id,
                    request_id,
                    generation=virtual.lease.generation,
                )
        except Exception:
            if executed_rows:
                self._serving_virtual_decode_poisoned = True
                self._serving_program_cache_initialization_uncertain = True
            raise

        devices = tuple(
            self.model.virtual_slot_token(
                slot,
                request,
                generation=self._serving_virtual_slots[slot].lease.generation,
            )
            for slot, request in zip(slots, requests)
        )
        self._serving_device_feedback_current = True
        output = _ServingDecodeOutput(
            "tokens",
            devices[0] if rows == 1 else devices,
            rows,
            state,
            tuple(zip(slots, requests, generations)),
        )
        if read_from_device:
            return self.process_decode_output_host(output, is_tokens=True)
        return output

    def read_decode_output(self, tt_out, *, async_read: bool = False):
        """Delegate the vLLM submit/read split to the canonical output."""

        if not isinstance(tt_out, _ServingDecodeOutput):
            return tt_out
        host = tt_out.read(blocking=not async_read)
        self._attach_direct_token_host(tt_out, host)
        if not async_read:
            self._attach_virtual_token_host(tt_out, host)
            return host
        completion = ttnn.record_event(self.mesh_device, 0)
        host.completion_event = completion
        self._attach_virtual_token_host(tt_out, host)
        return host, [completion]

    def process_decode_output_host(self, tt_out, *, is_tokens: bool = False) -> torch.Tensor:
        """Format only; sampling and device feedback have already completed."""

        if isinstance(tt_out, _ServingDecodeOutput):
            tt_out = self.read_decode_output(tt_out, async_read=False)
        if isinstance(tt_out, torch.Tensor):
            return tt_out
        if not isinstance(tt_out, _ServingDecodeHost):
            raise TypeError(f"cannot format serving decode output {type(tt_out).__name__}")
        return tt_out.to_torch(self.model, is_tokens=is_tokens)

    def decode_token_out(
        self,
        ple_input_ids,
        *,
        state: Qwen38BatchState | None = None,
        page_table=None,
        enable_trace: bool = True,
        read_from_device: bool = True,
    ):
        state = self.state if state is None else state
        if state is None:
            raise RuntimeError("batch state is not initialized")
        if page_table is not None:
            self.model.update_page_table(state, page_table)
        ids = torch.as_tensor(ple_input_ids, dtype=torch.int64).reshape(self.model.max_batch, 1)
        if enable_trace:
            _, output = self.model.decode_token_out_traced(state, ids)
        else:
            output = self.model.decode_forward(
                ids.reshape(-1),
                state=state,
                enable_trace=False,
                on_device_sampling=True,
            )
        return self.model.sampled_tokens_to_torch(output, state) if read_from_device else output

    # ------------------------------------------------------------ high level

    def _normalize_prompts(self, prompt_token_ids, prompt_lens=None):
        if hasattr(prompt_token_ids, "input_ids"):
            prompt_token_ids = prompt_token_ids.input_ids
        elif isinstance(prompt_token_ids, dict) and "input_ids" in prompt_token_ids:
            prompt_token_ids = prompt_token_ids["input_ids"]
        if isinstance(prompt_token_ids, torch.Tensor):
            tokens = prompt_token_ids.to(torch.int64).cpu()
            if tokens.ndim == 1:
                tokens = tokens.unsqueeze(0)
            if tokens.ndim != 2:
                raise ValueError("prompt_token_ids tensor must be rank one or two")
            lengths = (
                torch.full((tokens.shape[0],), tokens.shape[1], dtype=torch.int32)
                if prompt_lens is None
                else torch.as_tensor(prompt_lens, dtype=torch.int32).reshape(-1)
            )
            return tokens, lengths
        values = list(prompt_token_ids)
        if not values:
            raise ValueError("prompt_token_ids cannot be empty")
        if isinstance(values[0], int):
            values = [values]
        lengths = torch.tensor([len(row) for row in values], dtype=torch.int32)
        if prompt_lens is not None:
            supplied = torch.as_tensor(prompt_lens, dtype=torch.int32).reshape(-1)
            if not torch.equal(supplied, lengths):
                raise ValueError("prompt_lens disagrees with ragged prompt_token_ids")
        width = int(lengths.max())
        tokens = torch.full((len(values), width), PAD_TOKEN_ID, dtype=torch.int64)
        for row, sequence in enumerate(values):
            tokens[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.int64)
        return tokens, lengths

    @staticmethod
    def _host_sample(logits, *, top_k, top_p, temperature, seeds=None):
        """Explicit compatibility sampler; never used by optimized metrics."""

        # ``logits_to_torch`` returns the model's ``[1, 1, B, V]`` layout; the
        # teacher-forcing callers pass ``[B, S, V]`` and sample the last position.
        logits = torch.as_tensor(logits)
        rows = logits[:, -1, :].float() if logits.dim() == 3 else logits.float().reshape(-1, logits.shape[-1])
        batch = rows.shape[0]
        k_values = _as_parameter(top_k, batch, dtype=torch.int64)
        p_values = _as_parameter(top_p, batch, dtype=torch.float32)
        t_values = _as_parameter(temperature, batch, dtype=torch.float32)
        outputs = []
        for row in range(batch):
            k = int(k_values[row])
            if k == 1 and float(p_values[row]) == 0.0:
                outputs.append(int(torch.argmax(rows[row])))
                continue
            values, indices = torch.topk(rows[row] / float(t_values[row]), k=min(k, rows.shape[-1]))
            probabilities = torch.softmax(values, dim=-1)
            p = float(p_values[row])
            if 0.0 < p < 1.0:
                sorted_prob, order = probabilities.sort(descending=True)
                keep = sorted_prob.cumsum(0) <= p
                keep[0] = True
                probabilities = torch.where(keep[order.argsort()], probabilities, 0.0)
                probabilities /= probabilities.sum()
            generator = torch.Generator().manual_seed(int(seeds[row]) if seeds is not None else row)
            selected = int(torch.multinomial(probabilities, 1, generator=generator))
            outputs.append(int(indices[selected]))
        return torch.tensor(outputs, dtype=torch.int64)

    def generate(
        self,
        prompt_token_ids,
        max_new_tokens: int,
        *,
        next_input=None,
        enable_trace: bool = True,
        sampling_mode: str | None = None,
        top_k: int | Sequence[int] = 1,
        top_p: float | Sequence[float] = 0.0,
        temperature: float | Sequence[float] = 1.0,
        seeds: Sequence[int] | None = None,
        prompt_lens=None,
        page_table=None,
        request_ids=None,
        active_mask=None,
        stop_on_eos: bool = True,
    ) -> Iterable[TokenResult]:
        """Generate tokens; ``next_input`` enables traced teacher forcing.

        ``sampling_mode='device'`` is the optimized path.  It uses semantically
        greedy ``k=1,p=0,temp=1`` through full-vocabulary device argmax, or
        the same trace with caller-selected top-k/top-p parameters.
        ``sampling_mode='host'`` is an explicit compatibility mode that reads
        full logits and is excluded from performance evidence.
        """

        sampling_mode = self.model.selected_sampling_mode if sampling_mode is None else str(sampling_mode)
        if sampling_mode not in {"device", "host"}:
            raise ValueError("sampling_mode must be 'device' or explicit compatibility mode 'host'")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        prompts, lengths = self._normalize_prompts(prompt_token_ids, prompt_lens)
        if prompts.shape[0] != self.model.max_batch:
            raise ValueError(
                f"generator was built for {self.model.max_batch} fixed slots, got {prompts.shape[0]} prompts"
            )
        teacher = None
        if next_input is not None:
            teacher = torch.as_tensor(next_input, dtype=torch.int64, device="cpu")
            if teacher.ndim == 1:
                teacher = teacher.unsqueeze(0)
            if teacher.shape[0] != self.model.max_batch or teacher.shape[1] < max_new_tokens:
                raise ValueError("next_input must provide max_new_tokens teacher tokens for every fixed slot")

        state = self.allocate_batch_state(
            lengths,
            request_ids=request_ids,
            page_table=page_table,
            active_mask=active_mask,
        )
        self.model.set_sampling_params(top_k=top_k, top_p=top_p, temperature=temperature, seeds=seeds)
        metrics = GenerationMetrics(
            sampling_mode=sampling_mode,
            traced=enable_trace,
            teacher_forced=teacher is not None,
        )
        self.last_prompt_tokens = prompts.clone()
        generated = []

        started = time.perf_counter()
        prefill_logits_tt = self.model.prefill_forward(prompts, state=state)
        ttnn.synchronize_device(self.mesh_device)
        metrics.prefill_seconds = time.perf_counter() - started
        if sampling_mode == "device":
            first_tt = self.model.sample_logits(prefill_logits_tt, state)
            first = self.model.sampled_tokens_to_torch(first_tt, state)
        else:
            self.host_sampling_compatibility_calls += 1
            first_logits = self.model.logits_to_torch(prefill_logits_tt)
            first = self._host_sample(
                first_logits,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                seeds=seeds,
            )
            self.model.copy_tokens(state, first)
        ttnn.synchronize_device(self.mesh_device)
        metrics.ttft_seconds = time.perf_counter() - started
        ttnn.deallocate(prefill_logits_tt)
        generated.append(first.clone())
        if teacher is not None:
            self.model.copy_tokens(state, teacher[:, 0])
            feedback_host = teacher[:, 0]
        else:
            feedback_host = first
        if self.model.max_batch == 1:
            yield TokenResult(token=int(first[0]), text=self.tokenizer.decode([int(first[0])]))

        finished = torch.tensor([int(token) in EOS_TOKEN_IDS for token in first], dtype=torch.bool)
        collect_decode_timeline = os.getenv("QWEN38_COLLECT_DECODE_TIMELINE") == "1"
        for step in range(1, max_new_tokens):
            if stop_on_eos and teacher is None and bool(torch.all(finished)):
                break
            host_before = self.model.host_service_totals() if collect_decode_timeline else None
            step_started = time.perf_counter()
            if sampling_mode == "device":
                token_out_started = time.perf_counter()
                next_tokens = self.decode_token_out(
                    feedback_host,
                    state=state,
                    enable_trace=enable_trace,
                    read_from_device=True,
                )
                token_out_seconds = time.perf_counter() - token_out_started
            else:
                if enable_trace:
                    logits_tt = self.model.replay_model_only_traced(state, feedback_host.reshape(-1, 1))
                else:
                    logits_tt = self.model.decode_forward(
                        feedback_host,
                        state=state,
                        enable_trace=False,
                        on_device_sampling=False,
                    )
                logits = self.model.logits_to_torch(logits_tt)
                next_tokens = self._host_sample(
                    logits,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    seeds=(None if seeds is None else [int(seed) + step for seed in seeds]),
                )
                self.model.copy_tokens(state, next_tokens)
                if enable_trace:
                    self.model.advance_positions_traced()
                else:
                    ttnn.plus_one(state.current_pos, skip_negative_entries=True)
                    ttnn.deallocate(logits_tt)
                token_out_seconds = time.perf_counter() - step_started
            sync_started = time.perf_counter()
            ttnn.synchronize_device(self.mesh_device)
            post_read_sync_seconds = time.perf_counter() - sync_started
            completed_wall_seconds = time.perf_counter() - step_started
            metrics.decode_step_seconds.append(completed_wall_seconds)
            if collect_decode_timeline:
                host_after = self.model.host_service_totals()
                timing = dict(self.model.last_decode_timing or {})
                submit_seconds = float(timing.get("total_submit_seconds", 0.0))
                metrics.decode_step_breakdown.append(
                    {
                        "step": step,
                        "completed_wall_seconds": completed_wall_seconds,
                        "token_out_call_seconds": token_out_seconds,
                        "compact_read_and_completion_seconds": max(0.0, token_out_seconds - submit_seconds),
                        "post_read_sync_seconds": post_read_sync_seconds,
                        "model_submit": timing,
                        "host_service_delta": {
                            key: float(host_after[key]) - float(host_before[key]) for key in host_after
                        },
                    }
                )
            generated.append(next_tokens.clone())
            if teacher is not None:
                self.model.copy_tokens(state, teacher[:, step])
                feedback_host = teacher[:, step]
            else:
                feedback_host = next_tokens
            finished |= torch.tensor([int(token) in EOS_TOKEN_IDS for token in next_tokens], dtype=torch.bool)
            if self.model.max_batch == 1:
                token = int(next_tokens[0])
                yield TokenResult(token=token, text=self.tokenizer.decode([token]))

        metrics.decode_capture_seconds = self.model.trace_capture_seconds
        self.last_metrics = metrics
        self.last_generated_tokens = torch.stack(generated, dim=1)

    def generate_batch(self, *args, **kwargs) -> torch.Tensor:
        """Consume :meth:`generate` and return ``[batch, generated]`` tokens."""

        for _ in self.generate(*args, **kwargs):
            pass
        assert self.last_generated_tokens is not None
        return self.last_generated_tokens

    def chat_completion(
        self,
        messages,
        *,
        max_new_tokens: int = 100,
        enable_trace: bool = True,
        sampling_mode: str | None = None,
        **sampling,
    ) -> str:
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        tokens = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
        output = self.generate_batch(
            tokens,
            max_new_tokens,
            enable_trace=enable_trace,
            sampling_mode=sampling_mode,
            **sampling,
        )
        return self.tokenizer.decode(output[0].tolist(), skip_special_tokens=True)

    def close(self) -> None:
        self.model.close()


def build_generator(model_dir, mesh_device, **kwargs) -> Qwen38Generator:
    """Standard Metal readiness entry point."""

    from transformers import AutoTokenizer

    snapshot = _resolve_snapshot(model_dir)
    tokenizer = kwargs.pop("tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = Qwen38FullModel.from_pretrained(snapshot, mesh_device, **kwargs)
    return Qwen38Generator(model, tokenizer)


__all__ = ["GenerationMetrics", "Qwen38Generator", "build_generator"]
