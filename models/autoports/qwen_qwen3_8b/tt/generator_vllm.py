# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""vLLM adapter for the Qwen3-8B autoport on one Blackhole chip.

``Qwen3ForCausalLM`` is registered with the TT vLLM plugin as ``TTQwen3ForCausalLM`` through an
``EXTRA_MODELS_DIR`` bundle (``models/autoports/qwen_qwen3_8b/vllm/qwen3_8b/vllm_metadata.json``). It
translates the plugin's calls to ``Qwen3Generator``'s low-level methods and owns nothing else:

* ``initialize_vllm_model`` builds the generator from the selected precision config, runs the generator's
  two-phase warmup (``prepare``) over a small owned block pool and then releases that pool;
* ``allocate_kv_cache`` allocates the vLLM-owned paged cache (plus the scratch block that page-table
  columns past a request name) and every later call passes that cache and scratch block through;
* ``prefill_forward`` and ``decode_forward`` delegate; ``read_decode_output`` and
  ``process_decode_output_host`` implement the plugin's async decode split over the generator's
  deferred read; ``warmup_model_decode(enable_trace=True)`` captures the decode and sampling traces on
  the vLLM cache before the first request.

The adapter logs one line per prefill call and per decode step (``qwen3-vllm ...``) with the sampling
mode, bucket, trace counters, program-cache entries and out-of-vocabulary sampled ids so the server log
is the serving-path evidence.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

import ttnn
from models.autoports.qwen_qwen3_8b.tt.generator import (
    MAX_TOP_K,
    SLOT_WIDTH,
    DecodeOutput,
    Qwen3Generator,
    build_generator,
)
from models.autoports.qwen_qwen3_8b.tt.model import decode_bucket
from models.autoports.qwen_qwen3_8b.tt.precision import check_propagation, kv_cache_dtype, load_precision_config

MODEL_DIR = Path(__file__).resolve().parent.parent
LOG_PREFIX = "qwen3-vllm"
NUM_LAYERS_ENV = "QWEN3_VLLM_NUM_LAYERS"
logger = logging.getLogger("vllm.autoport.qwen3_8b")


def _counter_delta(before: dict, after: dict, names: Sequence[str]) -> str:
    return " ".join(f"{name}={after[name] - before[name]}" for name in names)


class Qwen3ForCausalLM:
    model_capabilities = {
        "supports_prefix_caching": True,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
        "max_device_top_k": MAX_TOP_K,
        "supports_device_penalties": True,
        "supports_device_seeded_sampling": True,
    }

    _live: Optional["Qwen3ForCausalLM"] = None

    def __init__(self, generator: Qwen3Generator) -> None:
        self.generator = generator
        self.kv_cache_dtype = generator.kv_cache_dtype
        self.vocab_size = generator.model.vocab_size
        self.scratch_block: int | None = None
        self.tt_supported_decode_batch_sizes = tuple(generator.decode_buckets)
        self.prefill_calls = 0
        self.decode_steps = 0
        self.host_sampled_steps = 0
        self.device_sampled_steps = 0
        self.out_of_vocab_tokens = 0

    # ------------------------------------------------------------ plugin hooks

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config: Any,
        mesh_device: Any,
        max_batch_size: int,
        max_seq_len: int | None = None,
        tt_data_parallel: int = 1,
        optimizations: str | None = None,
        **kwargs: Any,
    ) -> "Qwen3ForCausalLM":
        if tt_data_parallel != 1:
            raise ValueError(f"{cls.__name__} serves one chip; tt_data_parallel={tt_data_parallel}")
        if mesh_device.get_num_devices() != 1:
            raise ValueError(f"{cls.__name__} serves one chip; mesh has {mesh_device.get_num_devices()} devices")
        if not 1 <= max_batch_size <= SLOT_WIDTH:
            raise ValueError(f"max_batch_size {max_batch_size} outside [1, {SLOT_WIDTH}]")
        build_kwargs: dict[str, Any] = {
            "max_batch_size": max_batch_size,
            "log": logger.info,
            "prepare": False,
            "num_blocks": 1,
        }
        layers = os.environ.get(NUM_LAYERS_ENV)
        if layers:
            build_kwargs["num_layers"] = int(layers)
            logger.warning(
                "%s reduced model: %s=%s (inner-loop target, not serving evidence)", LOG_PREFIX, NUM_LAYERS_ENV, layers
            )
        if optimizations is not None:
            logger.info(
                "%s ignoring optimizations=%r: the precision policy comes from the selected config",
                LOG_PREFIX,
                optimizations,
            )
        started = time.perf_counter()
        generator = build_generator(str(MODEL_DIR), mesh_device, **build_kwargs)
        if max_seq_len is not None and max_seq_len > generator.max_context:
            raise ValueError(f"max_seq_len {max_seq_len} exceeds the {generator.max_context}-token context")
        problems = check_propagation(generator)
        logger.info(
            "%s precision config %s from %s: propagation check %s",
            LOG_PREFIX,
            generator.precision_config["config_id"],
            generator.precision_config_path,
            "OK" if not problems else problems,
        )
        if problems:
            raise RuntimeError(f"selected precision config not consumed by the runtime: {problems}")
        instance = cls(generator)
        cls._live = instance
        try:
            from ttnn import trace_allocation_config

            tracking = bool(trace_allocation_config.TRACE_ALLOC_TRACKING)
        except Exception:
            tracking = None
        logger.info(
            "%s trace allocation tracking in this engine process: %s (TT_METAL_TRACE_ALLOC_TRACKING=%r, "
            "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=%r)",
            LOG_PREFIX,
            tracking,
            os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING"),
            os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"),
        )
        logger.info(
            "%s model ready in %.1f s: class %s.%s, max_batch_size %d, max_seq_len %s, decode buckets %s, "
            "kv dtype %s, program cache entries %d",
            LOG_PREFIX,
            time.perf_counter() - started,
            cls.__module__,
            cls.__name__,
            max_batch_size,
            max_seq_len,
            instance.tt_supported_decode_batch_sizes,
            instance.kv_cache_dtype,
            mesh_device.num_program_cache_entries(),
        )
        return instance

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        **kwargs: Any,
    ) -> int:
        config, path = load_precision_config(None)
        dtype = kv_cache_dtype(config)
        constants = Qwen3Generator.max_tokens_all_users(kv_cache_dtype=dtype)
        live = cls._live
        if live is None:
            budget = constants
            source = "stage-6 constants (no live model)"
        else:
            model = live.generator.model
            budget = Qwen3Generator.max_tokens_all_users(
                model, mesh_device=live.generator.mesh_device, kv_cache_dtype=dtype
            )
            computed = Qwen3Generator.max_tokens_all_users(model, kv_cache_dtype=dtype)
            source = (
                f"live free DRAM after the owned pool was released (model-computed {computed}, constants {constants})"
            )
        logger.info(
            "%s max_tokens_all_users=%d from %s; kv dtype %s (%s), max_model_len %s, max_num_seqs %s",
            LOG_PREFIX,
            budget,
            source,
            dtype,
            path,
            max_model_len,
            max_num_seqs,
        )
        return int(budget)

    def allocate_kv_cache(self, kv_cache_shape: Sequence[int], dtype: Any, num_layers: int) -> list[tuple]:
        num_blocks, kv_heads, block_size, head_dim = (int(v) for v in kv_cache_shape)
        model = self.generator.model
        if (kv_heads, block_size, head_dim) != (model.geometry.num_kv_heads, model.block_size, model.geometry.head_dim):
            raise ValueError(
                f"vLLM KV shape {tuple(kv_cache_shape)} does not match the model "
                f"({model.geometry.num_kv_heads}, {model.block_size}, {model.geometry.head_dim})"
            )
        if num_layers != model.num_layers:
            if not os.environ.get(NUM_LAYERS_ENV):
                raise ValueError(f"vLLM asked for {num_layers} layers; the model has {model.num_layers}")
            logger.warning(
                "%s reduced model: allocating %d layers of the %d vLLM asked for",
                LOG_PREFIX,
                model.num_layers,
                num_layers,
            )
        cache = model.allocate_kv_cache(num_blocks, dtype=self.kv_cache_dtype)
        self.scratch_block = num_blocks
        self.generator.adopt_cache(cache, scratch_block=num_blocks)
        logger.info(
            "%s vLLM KV cache: %d blocks of %d tokens plus scratch block %d, %d layers, dtype %s (vLLM spec dtype %s)",
            LOG_PREFIX,
            num_blocks,
            block_size,
            num_blocks,
            num_layers,
            self.kv_cache_dtype,
            dtype,
        )
        return cache

    def warmup_model_prefill(
        self, kv_cache: Any, enable_trace: bool, can_sample_on_device: bool, **kwargs: Any
    ) -> None:
        self.generator.prepare()
        logger.info(
            "%s warmup_model_prefill(enable_trace=%s): prefill is eager; prepare() is complete",
            LOG_PREFIX,
            enable_trace,
        )

    def warmup_model_decode(
        self,
        kv_cache: Any,
        enable_trace: bool,
        max_batch_size: int | None = None,
        num_blocks: int | None = None,
        can_sample_on_device: bool = True,
        **kwargs: Any,
    ) -> None:
        self.generator.prepare()
        if not enable_trace:
            logger.info("%s warmup_model_decode(enable_trace=False): prepare() is complete", LOG_PREFIX)
            return
        captured = self.generator.capture_decode_traces(kv_cache, scratch_block=self.scratch_block)
        self.generator.release_owned_cache()
        logger.info(
            "%s warmup_model_decode(enable_trace=True): captured %s; traces %s; program cache entries %d; counters %s",
            LOG_PREFIX,
            {f"{bucket}/{mode}": trace_id for (bucket, mode), trace_id in captured.items()},
            self.generator.trace_summary(),
            self.generator.mesh_device.num_program_cache_entries(),
            self.generator.counter_snapshot(),
        )

    # ------------------------------------------------------------ forward passes

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: Any,
        enable_trace: bool = False,
        prompt_lens: Sequence[int] | None = None,
        start_pos: Any = None,
        sampling_params: Any = None,
        empty_slots: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        before = self.generator.counter_snapshot()
        started = time.perf_counter()
        out = self.generator.prefill_forward(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=list(prompt_lens),
            start_pos=start_pos,
            sampling_params=sampling_params,
            empty_slots=empty_slots,
            scratch_block=self.scratch_block,
        )
        self.prefill_calls += 1
        starts = (
            [0] * tokens.shape[0] if start_pos is None else [int(s) for s in torch.as_tensor(start_pos).reshape(-1)]
        )
        logger.info(
            "%s prefill call=%d rows=%d starts=%s ends=%s mode=%s ms=%.1f program_cache=%d dram_alloc_mb=%.0f %s",
            LOG_PREFIX,
            self.prefill_calls,
            tokens.shape[0],
            starts,
            [int(e) for e in prompt_lens],
            "device-sampled" if sampling_params is not None else "logits",
            (time.perf_counter() - started) * 1000.0,
            self.generator.mesh_device.num_program_cache_entries(),
            self._dram_allocated_mb(),
            _counter_delta(
                before,
                self.generator.counter_snapshot(),
                (
                    "batched_prefill_groups",
                    "batched_prefill_rows",
                    "single_row_prefill_rows",
                    "decode_trace_captures",
                    "sampling_trace_captures",
                ),
            ),
        )
        return out

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: Any,
        enable_trace: bool = True,
        read_from_device: bool = True,
        sampling_params: Any = None,
        reset_batch: bool = False,
        slot_remap: Any = None,
        **kwargs: Any,
    ) -> Any:
        before = self.generator.counter_snapshot()
        out = self.generator.decode_forward(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            enable_trace=enable_trace,
            read_from_device=read_from_device,
            sampling_params=sampling_params,
            reset_batch=reset_batch,
            slot_remap=slot_remap,
            scratch_block=self.scratch_block,
            prompt_tokens=kwargs.get("prompt_tokens"),
            output_tokens=kwargs.get("output_tokens"),
        )
        self.decode_steps += 1
        device_mode = sampling_params is not None
        if device_mode:
            self.device_sampled_steps += 1
        else:
            self.host_sampled_steps += 1
        batch = int(tokens.shape[0])
        logger.info(
            "%s decode step=%d rows=%d bucket=%s mode=%s traced=%s reset_batch=%s deferred_read=%s %s program_cache=%d",
            LOG_PREFIX,
            self.decode_steps,
            batch,
            decode_bucket(batch, self.generator.model.grid.x, self.generator.model.grid.y),
            "device" if device_mode else "host",
            enable_trace,
            reset_batch,
            not read_from_device,
            _counter_delta(
                before,
                self.generator.counter_snapshot(),
                (
                    "decode_trace_replays",
                    "sampling_trace_replays",
                    "decode_trace_captures",
                    "sampling_trace_captures",
                    "token_refreshes",
                    "position_refreshes",
                    "page_table_refreshes",
                    "eager_decode_steps",
                    "seeded_eager_sampling_steps",
                    "penalized_sampling_steps",
                ),
            ),
            self.generator.mesh_device.num_program_cache_entries(),
        )
        if not read_from_device:
            return out
        return self._format_host(out, device_mode)

    def read_decode_output(self, tt_out: DecodeOutput, async_read: bool = False):
        return self.generator.read_decode_output(tt_out, async_read=async_read)

    def process_decode_output_host(self, tt_out: DecodeOutput, is_tokens: bool = False) -> torch.Tensor:
        result = self.generator.process_decode_output_host(tt_out, is_tokens=is_tokens)
        return self._format_host(result, is_tokens)

    def _dram_allocated_mb(self) -> float:
        view = ttnn.get_memory_view(self.generator.mesh_device, ttnn.BufferType.DRAM)
        return float(view.total_bytes_allocated_per_bank) * float(view.num_banks) / 1e6

    def _format_host(self, result: torch.Tensor, is_tokens: bool) -> torch.Tensor:
        if is_tokens:
            invalid = int((result >= self.vocab_size).sum()) + int((result < 0).sum())
            if invalid:
                self.out_of_vocab_tokens += invalid
                logger.error(
                    "%s %d sampled ids outside [0, %d): %s", LOG_PREFIX, invalid, self.vocab_size, result.tolist()
                )
            return result
        return result.unsqueeze(1)


__all__ = ["Qwen3ForCausalLM"]
