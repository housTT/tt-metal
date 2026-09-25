# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Thin vLLM bridge for the Qwen/Qwen3.6-27B full-model generator.

The adapter translates the TT vLLM plugin protocol only.  Attention KV cache,
model execution, split traced sampling, token feedback, deferred reads, and
host compatibility all remain owned by :mod:`tt.generator` / :mod:`tt.model`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import ttnn

from models.autoports.qwen_qwen3_6_27b.tt.generator import Generator
from models.autoports.qwen_qwen3_6_27b.tt.model import (
    MAX_BATCH_SIZE,
    MODEL_ID,
    PAGE_BLOCK_SIZE,
)
from models.autoports.qwen_qwen3_6_27b.tt.precision import DEFAULT_PRECISION_CONFIG
from models.common.sampling.generator import SamplingParams


MAX_MODEL_LEN = 262_144
FULL_ATTENTION_LAYERS = tuple(range(3, 64, 4))


def _sampling_params(value: Any) -> SamplingParams:
    """Translate the plugin dataclass without inventing a sampling policy."""

    names = (
        "temperature",
        "top_p",
        "top_k",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "seed",
        "enable_log_probs",
        "num_logprobs",
    )
    return SamplingParams(**{name: getattr(value, name) for name in names})


def _empty_multimodal_placeholder(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return all(_empty_multimodal_placeholder(item) for item in value)
    if isinstance(value, torch.Tensor):
        return value.numel() == 0
    return False


class Qwen3_5ForConditionalGeneration:
    """vLLM protocol adapter backed by the canonical autoport ``Generator``."""

    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
    }

    def __init__(self, generator: Generator, *, max_batch_size: int, max_seq_len: int):
        self.generator = generator
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self._vllm_kv_cache = None
        self._expanded_kv_cache = None
        self._state = None
        self._decode_started = False

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
    ):
        del optimizations
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported Qwen3.6 vLLM initializer arguments: {unexpected}")
        max_seq_len = MAX_MODEL_LEN if max_seq_len is None else int(max_seq_len)
        if tt_data_parallel != 1:
            raise ValueError("Qwen3.6-27B autoport serving supports TP4 with tt_data_parallel=1")
        if not 1 <= int(max_batch_size) <= MAX_BATCH_SIZE:
            raise ValueError(f"max_batch_size must be in [1, {MAX_BATCH_SIZE}]")
        if not 1 <= max_seq_len <= MAX_MODEL_LEN:
            raise ValueError(f"max_seq_len must be in [1, {MAX_MODEL_LEN}]")
        if mesh_device.get_num_devices() != 4:
            raise ValueError("Qwen3.6-27B autoport serving requires the measured TP4 mesh")

        configured = getattr(hf_config, "_name_or_path", None)
        checkpoint = Path(configured).expanduser() if configured else None
        if checkpoint is not None and not checkpoint.exists():
            checkpoint = None
        generator = Generator(
            mesh_device=mesh_device,
            checkpoint_path=checkpoint,
            max_seq_len=max_seq_len,
            precision_config_path=DEFAULT_PRECISION_CONFIG,
        )
        if generator.model.precision_policy.path != DEFAULT_PRECISION_CONFIG.resolve():
            raise RuntimeError("vLLM did not load the selected datatype-sweep precision config")
        return cls(generator, max_batch_size=int(max_batch_size), max_seq_len=max_seq_len)

    @classmethod
    def get_max_tokens_all_users(cls, **kwargs) -> int:
        del kwargs
        # Shared physical token pool from doc/context_contract.json.  The TT
        # worker adds one page per admitted row for its worst-case heuristic.
        return MAX_MODEL_LEN

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        del dtype  # The selected datatype policy, not torch cache dtype, is authoritative.
        shape = tuple(int(value) for value in kv_cache_shape)
        if len(shape) != 4:
            raise ValueError("kv_cache_shape must be [blocks, local_kv_heads, block, head_dim]")
        num_blocks, local_heads, block_size, head_dim = shape
        if num_layers != len(FULL_ATTENTION_LAYERS):
            raise ValueError(f"expected {len(FULL_ATTENTION_LAYERS)} full-attention caches")
        if (local_heads, block_size, head_dim) != (1, PAGE_BLOCK_SIZE, 256):
            raise ValueError(
                "Qwen3.6 TP4 cache shape must use one local KV head, 64-token pages, head_dim=256"
            )

        cache_dtype = {"bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}[
            self.generator.model.precision_policy.kv_cache["dtype"]
        ]
        compact = []
        expanded = [None] * self.generator.model.num_layers
        for layer_idx in FULL_ATTENTION_LAYERS:
            pair = self.generator.model.layers[layer_idx].allocate_paged_kv_cache(
                num_blocks=num_blocks, dtype=cache_dtype
            )
            compact.append(pair)
            expanded[layer_idx] = pair

        page_table = torch.zeros(
            (MAX_BATCH_SIZE, math.ceil(self.max_seq_len / PAGE_BLOCK_SIZE)),
            dtype=torch.int32,
        )
        self._state = self.generator.model.allocate_state(
            batch_size=MAX_BATCH_SIZE,
            page_table=page_table,
            external_kv_cache=expanded,
            num_blocks=num_blocks,
            linear_state_capacity=MAX_BATCH_SIZE,
        )
        self.generator._state = self._state
        self._vllm_kv_cache = compact
        self._expanded_kv_cache = expanded
        return compact

    def _require_cache(self, kv_cache):
        if kv_cache is not self._vllm_kv_cache or self._state is None:
            raise ValueError("vLLM must pass through the cache object returned by allocate_kv_cache")

    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        sampling_params=None,
        empty_slots=None,
        **kwargs,
    ):
        self._require_cache(kv_cache)
        multimodal_protocol = (
            "pixel_values" in kwargs or "image_grid_thw" in kwargs
        )
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        if not _empty_multimodal_placeholder(
            pixel_values
        ) or not _empty_multimodal_placeholder(image_grid_thw):
            raise ValueError("Qwen3.6-27B autoport serving is text-only")
        kwargs.pop("enable_trace", None)
        kwargs.pop("start_pos", None)
        kwargs.pop("page_tables_per_layer", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported Qwen3.6 prefill arguments: {unexpected}")
        slots = list(range(tokens.shape[0])) if empty_slots is None else [int(x) for x in empty_slots]
        result = self.generator.prefill_forward(
            tokens,
            page_table=page_table,
            kv_cache=self._expanded_kv_cache,
            prompt_lens=[int(value) for value in prompt_lens],
            state=self._state,
            state_slots=slots,
            sampling_mode="device" if sampling_params is not None else "host",
            sampling_params=(None if sampling_params is None else _sampling_params(sampling_params)),
            page_table_changed=True,
        )
        if multimodal_protocol:
            # Qwen3.5's HF config advertises mRoPE even for text-only input,
            # so the shared runner persists one delta per request.  Text uses
            # ordinary absolute positions and therefore has a neutral delta.
            return result, torch.zeros(tokens.shape[0], dtype=torch.int64)
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
        **kwargs,
    ):
        self._require_cache(kv_cache)
        kwargs.pop("page_tables_per_layer", None)
        kwargs.pop("rope_deltas_all_users", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported Qwen3.6 decode arguments: {unexpected}")

        device_sampling = sampling_params is not None
        effective_reset = (not self._decode_started) or bool(reset_batch) or not device_sampling
        active_positions = start_pos.reshape(-1)
        page_boundary = bool(
            torch.any(
                (active_positions >= 0)
                & (active_positions.remainder(PAGE_BLOCK_SIZE) == 0)
            ).item()
        )
        result = self.generator.decode_forward(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=self._expanded_kv_cache,
            state=self._state,
            enable_trace=bool(enable_trace),
            sampling_mode="device" if device_sampling else "host",
            sampling_params=(None if sampling_params is None else _sampling_params(sampling_params)),
            # vLLM may append a physical page without changing the batch
            # layout. Refresh exactly on those page boundaries; copying the
            # full table every token would be avoidable serving overhead.
            page_table_changed=effective_reset or page_boundary,
            read_from_device=read_from_device,
            reset_batch=effective_reset,
            slot_remap=slot_remap,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            serving_mode=True,
        )
        self._decode_started = True
        if not device_sampling and isinstance(result, torch.Tensor) and result.ndim == 2:
            result = result.unsqueeze(1)
        return result

    def read_decode_output(self, tt_out, async_read=False):
        return self.generator.read_decode_output(tt_out, async_read=async_read)

    def process_decode_output_host(self, tt_out, is_tokens=True):
        return self.generator.process_decode_output_host(tt_out, is_tokens=is_tokens)

    def warmup_model_prefill(self, **kwargs):
        # Prefill supports arbitrary logical lengths and compiles before the
        # first request-bound decode trace is captured.
        del kwargs

    def warmup_model_decode(self, **kwargs):
        # Decode trace capture is deliberately request-bound: it must bind the
        # exact vLLM-owned cache/state buffers after the first real prefill.
        del kwargs


__all__ = ["Qwen3_5ForConditionalGeneration"]
