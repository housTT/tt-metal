# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Thin vLLM adapter for the canonical Qwen3.8 full-model generator.

The adapter translates the shared TT plugin protocol only.  Model execution,
selected precision, split traced sampling, direct device token feedback,
linear-attention recurrence, expert/PLE host service, and output formatting all
remain owned by :mod:`tt.generator` and :mod:`tt.model`.

The serving cache object contains K, V, and the QSA indexer's paged key state
for each of the twelve QSA layers.  vLLM owns that attention object and its
block ids.  The model owns the 36 linear-attention recurrent states plus the
declared expert and PLE host stores; none of those are disguised as vLLM KV.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from models.autoports.qwen_qwen3_8_flash_next.tt.generator import Qwen38Generator, build_generator
from models.autoports.qwen_qwen3_8_flash_next.tt.model import DEFAULT_SNAPSHOT, MODEL_ID, Qwen38FullModel
from models.autoports.qwen_qwen3_8_flash_next.tt.precision_config import DEFAULT_PRECISION_CONFIG_PATH

MAX_MODEL_LEN = 262_144
MAX_NUM_SEQS = 2
PHYSICAL_TRACE_BATCH = 1
QSA_LAYER_INDICES = tuple(range(3, 48, 4))
_REDUCED_LAYERS_ENV = "QWEN38_VLLM_LAYER_INDICES"
_SERVING_METRICS_ENV = "QWEN38_VLLM_LOG_METRICS"


def _reduced_layer_indices() -> tuple[int, ...] | None:
    raw = os.getenv(_REDUCED_LAYERS_ENV, "").strip()
    if not raw:
        return None
    values = tuple(int(part) for part in raw.split(",") if part.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{_REDUCED_LAYERS_ENV} must contain unique comma-separated layer indices")
    return values


def _empty_text_placeholder(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (tuple, list)):
        return all(_empty_text_placeholder(item) for item in value)
    if isinstance(value, torch.Tensor):
        return value.numel() == 0
    return False


class Qwen4ExpForConditionalGeneration:
    """vLLM protocol bridge backed by :class:`Qwen38Generator`."""

    # Lets ModelConfig classify text_config.layer_types correctly: 12
    # full-attention blocks and 36 model-owned linear-attention blocks.
    is_hybrid = True
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
        "supports_request_specific_rope": False,
        "supports_device_sampling_penalties": False,
        "device_sampling_max_top_k": 32,
        # The TT and vLLM host samplers deliberately use different RNG
        # algorithms.  Keep explicit-seed stochastic requests on the optional
        # host compatibility path so cohort composition cannot switch algorithms.
        "force_host_seeded_sampling": True,
        "supports_virtual_state_slots": True,
        "supports_intermediate_prefill_device_sampling": True,
    }

    def __init__(
        self,
        generator: Qwen38Generator,
        *,
        max_batch_size: int,
        max_seq_len: int,
        vllm_config: Any = None,
    ) -> None:
        self.vllm_config = vllm_config
        self.generator = generator
        self.max_batch_size = int(max_batch_size)
        self.max_seq_len = int(max_seq_len)
        self._vllm_kv_cache = None
        self._closed = False
        self._completed_requests = 0

    @property
    def model(self) -> Qwen38FullModel:
        return self.generator.model

    @property
    def mesh_device(self):
        return self.generator.mesh_device

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=None,
        tt_data_parallel=1,
        optimizations=None,
        **kwargs,
    ) -> "Qwen4ExpForConditionalGeneration":
        if kwargs:
            raise TypeError(f"unsupported Qwen3.8 initializer arguments: {', '.join(sorted(kwargs))}")
        if optimizations is not None:
            raise ValueError("Qwen3.8 serving uses the datatype-sweep selection, not a vLLM preset")
        if int(tt_data_parallel) != 1:
            raise ValueError("Qwen3.8 serving requires the measured TP2 mesh with tt_data_parallel=1")
        if int(mesh_device.get_num_devices()) != 2:
            raise ValueError("Qwen3.8 serving requires exactly two P300 devices")
        if not 1 <= int(max_batch_size) <= MAX_NUM_SEQS:
            raise ValueError(
                f"Qwen3.8 supports between 1 and {MAX_NUM_SEQS} active virtual slots over its physical-B1 trace"
            )

        max_seq_len = MAX_MODEL_LEN if max_seq_len is None else int(max_seq_len)
        if not 1 <= max_seq_len <= MAX_MODEL_LEN:
            raise ValueError(f"max_seq_len must be in [1, {MAX_MODEL_LEN}]")
        served = str(getattr(hf_config, "_name_or_path", "") or "")
        snapshot = Path(served).expanduser() if served else DEFAULT_SNAPSHOT
        if not snapshot.exists():
            if served not in {"", MODEL_ID}:
                raise ValueError(f"vLLM requested {served!r}, but this adapter serves {MODEL_ID!r}")
            snapshot = DEFAULT_SNAPSHOT

        generator = build_generator(
            snapshot,
            mesh_device,
            max_batch=PHYSICAL_TRACE_BATCH,
            max_seq_len=max_seq_len,
            virtual_slot_capacity=int(max_batch_size),
            layer_indices=_reduced_layer_indices(),
            precision_config=DEFAULT_PRECISION_CONFIG_PATH,
        )
        if Path(generator.model.precision_config_path).resolve() != DEFAULT_PRECISION_CONFIG_PATH.resolve():
            generator.close()
            raise RuntimeError("vLLM did not load the selected datatype-sweep precision config")
        return cls(generator, max_batch_size=int(max_batch_size), max_seq_len=max_seq_len)

    @classmethod
    def get_max_tokens_all_users(cls, max_model_len=None, max_num_seqs=None, **_kwargs) -> int:
        if max_num_seqs is not None and not 1 <= int(max_num_seqs) <= MAX_NUM_SEQS:
            raise ValueError(f"Qwen3.8 traced serving supports at most {MAX_NUM_SEQS} active virtual slots")
        context = MAX_MODEL_LEN if max_model_len is None else int(max_model_len)
        if not 1 <= context <= MAX_MODEL_LEN:
            raise ValueError(f"max_model_len must be in [1, {MAX_MODEL_LEN}]")
        return context

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Allocate the exact vLLM-owned QSA attention object at selected BFP8."""

        del dtype  # Logical accounting dtype; selected TT precision is authoritative.
        expected_layers = len(QSA_LAYER_INDICES)
        if _reduced_layer_indices() is None and int(num_layers) != expected_layers:
            raise ValueError(f"vLLM requested {num_layers} attention layers; Qwen3.8 has {expected_layers}")
        if self._vllm_kv_cache is not None:
            raise RuntimeError("vLLM attention cache was already allocated")
        self._vllm_kv_cache = self.model.allocate_vllm_attention_cache(kv_cache_shape)
        return self._vllm_kv_cache

    def _require_cache(self, kv_cache) -> None:
        if self._vllm_kv_cache is None or kv_cache is not self._vllm_kv_cache:
            raise ValueError("vLLM must pass the exact attention object returned by allocate_kv_cache")

    @staticmethod
    def _reject_per_layer_page_tables(page_tables_per_layer) -> None:
        if page_tables_per_layer is not None:
            raise NotImplementedError("Qwen3.8 uses one uniform QSA page table, not hybrid per-layer groups")

    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        sampling_params=None,
        empty_slots=None,
        start_pos=None,
        intermediate_prefill_mask=None,
        request_ids=None,
        state_slot_ids=None,
        state_slot_generations=None,
        unpadded_batch_size=None,
        released_state_slots=None,
        page_tables_per_layer=None,
        **kwargs,
    ):
        self._log_serving_metrics("prefill_start")
        self._require_cache(kv_cache)
        self._reject_per_layer_page_tables(page_tables_per_layer)
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        if not _empty_text_placeholder(pixel_values) or not _empty_text_placeholder(image_grid_thw):
            raise ValueError("Qwen3.8 autoport serving is text-only")
        kwargs.pop("enable_trace", None)
        if kwargs:
            raise TypeError(f"unsupported Qwen3.8 prefill arguments: {', '.join(sorted(kwargs))}")
        slots = empty_slots if state_slot_ids is None else state_slot_ids
        result = self.generator.prefill_forward(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=[int(value) for value in prompt_lens],
            start_pos=start_pos,
            intermediate_prefill_mask=intermediate_prefill_mask,
            empty_slots=slots,
            request_ids=request_ids,
            state_slot_ids=slots,
            state_slot_generations=state_slot_generations,
            unpadded_batch_size=unpadded_batch_size,
            released_state_slots=released_state_slots,
            on_device_sampling=sampling_params is not None,
            sampling_params=sampling_params,
        )
        # Count logical requests, not adapter invocations.  A single physical
        # prefill call may contain several virtual rows, while intermediate
        # chunk continuations have not completed admission yet.  The next
        # request-boundary marker can therefore delimit B1 and burst windows by
        # exact request-count deltas even when vLLM groups prefills.
        if intermediate_prefill_mask is None:
            completed_rows = len(prompt_lens)
        else:
            intermediate = torch.as_tensor(intermediate_prefill_mask, dtype=torch.bool).reshape(-1)
            if intermediate.numel() != len(prompt_lens):
                raise ValueError("intermediate_prefill_mask must name every logical prompt row")
            completed_rows = int((~intermediate).sum().item())
        self._completed_requests += completed_rows
        return result

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        prompt_tokens=None,
        output_tokens=None,
        reset_batch=None,
        slot_remap=None,
        request_ids=None,
        state_slot_ids=None,
        state_slot_generations=None,
        unpadded_batch_size=None,
        released_state_slots=None,
        page_tables_per_layer=None,
        **kwargs,
    ):
        self._require_cache(kv_cache)
        self._reject_per_layer_page_tables(page_tables_per_layer)
        kwargs.pop("rope_deltas_all_users", None)
        if kwargs:
            raise TypeError(f"unsupported Qwen3.8 decode arguments: {', '.join(sorted(kwargs))}")
        result = self.generator.decode_forward(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            enable_trace=bool(enable_trace),
            read_from_device=bool(read_from_device),
            sampling_params=sampling_params,
            reset_batch=bool(reset_batch),
            slot_remap=slot_remap,
            request_ids=request_ids,
            state_slot_ids=state_slot_ids,
            state_slot_generations=state_slot_generations,
            unpadded_batch_size=unpadded_batch_size,
            released_state_slots=released_state_slots,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            serving_mode=True,
        )
        return result

    def read_decode_output(self, tt_out, async_read=False):
        return self.generator.read_decode_output(tt_out, async_read=async_read)

    def process_decode_output_host(self, tt_out, is_tokens=False):
        return self.generator.process_decode_output_host(tt_out, is_tokens=is_tokens)

    def release_virtual_state_slots(self, released_slots) -> None:
        """Release finished/preempted leases after async submissions drain."""

        self.generator.release_virtual_slots(released_slots)

    def warmup_model_prefill(self, **kwargs) -> None:
        # Request-bound prefill owns exact logical length and host-store history.
        del kwargs

    def warmup_model_decode(self, **kwargs) -> None:
        # Decode capture binds the exact vLLM cache/state buffers after prefill.
        del kwargs

    # vLLM inspects the plain architecture before TTPlatform prefixes it.  The
    # checkpoint has no upstream vLLM implementation, so expose the generative
    # protocol surface used by that inspection.  TT execution never calls these
    # methods; the shared TT runner calls prefill_forward/decode_forward.
    def embed_input_ids(self, input_ids):  # pragma: no cover - protocol shim
        raise NotImplementedError("Qwen3.8 embeds inside the TT prefill/decode path")

    def forward(self, input_ids, positions, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError("Qwen3.8 executes through the shared TT model runner")

    def compute_logits(self, hidden_states, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError("Qwen3.8 produces logits inside the TT generator")

    def lifecycle_metrics(self) -> dict[str, Any]:
        state = self.generator.state
        return {
            "attention_cache_owner": self.model._attention_cache_owner,
            "attention_cache": dict(self.model.attention_cache_lifecycle),
            "host_service": self.model.host_service_totals(),
            "host_gauges": self.model.host_service_gauges(),
            "decode_timing": dict(self.model.decode_timing_totals),
            "virtual_slots": self.model.virtual_slot_metrics(),
            "runtime_audit": None if state is None else self.model.runtime_fallback_audit(state),
        }

    def _log_serving_metrics(self, event: str) -> None:
        """Emit one compact cumulative snapshot at a request boundary.

        The opt-in marker is collected for readiness evidence only.  It runs
        before prefill, never in the token loop, so TPOT and traced decode are
        unaffected.  Consecutive snapshots make exact single-user and burst
        deltas recoverable from ``server.log``.
        """

        if os.getenv(_SERVING_METRICS_ENV) != "1":
            return
        state = self.generator.state
        runtime = None if state is None else self.model.runtime_fallback_audit(state)
        payload = {
            "event": event,
            "completed_requests": self._completed_requests,
            "attention_cache_owner": self.model._attention_cache_owner,
            "attention_cache": dict(self.model.attention_cache_lifecycle),
            "host_service": self.model.host_service_totals(),
            "host_gauges": self.model.host_service_gauges(),
            "decode_timing": dict(self.model.decode_timing_totals),
            "runtime_fallback": (
                None
                if runtime is None
                else {
                    "declared_host_work": runtime["declared_host_work"],
                    "prohibited_host_work": runtime["prohibited_host_work"],
                    "ownership": runtime["ownership"],
                    "counters": runtime["counters"]
                    | {
                        "async_feedback_host_reuses": self.generator.async_feedback_host_reuses,
                        "async_feedback_device_fallbacks": self.generator.async_feedback_device_fallbacks,
                    },
                    "host_sampling_compatibility_calls": self.generator.host_sampling_compatibility_calls,
                }
            ),
            "request_counters": (
                None
                if state is None
                else {
                    "token_host_copies": state.token_host_copies,
                    "position_host_copies": state.position_host_copies,
                    "page_table_host_copies": state.page_table_host_copies,
                    "page_table_unchanged_skips": state.page_table_unchanged_skips,
                    "compact_token_readbacks": state.compact_token_readbacks,
                    "async_feedback_host_reuses": self.generator.async_feedback_host_reuses,
                    "async_feedback_device_fallbacks": self.generator.async_feedback_device_fallbacks,
                }
            ),
        }
        # EngineCore disables third-party Python loggers, including WARNING,
        # so emit the opt-in readiness marker directly to its captured stdout.
        # This is request-boundary-only and remains disabled in normal serving.
        print(f"QWEN38_VLLM_METRICS {json.dumps(payload, sort_keys=True)}", flush=True)

    def close(self) -> None:
        if self._closed:
            return
        self.generator.close()
        self._closed = True


TTQwen4ExpForConditionalGeneration = Qwen4ExpForConditionalGeneration


__all__ = ["Qwen4ExpForConditionalGeneration", "TTQwen4ExpForConditionalGeneration"]
