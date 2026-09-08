# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""vLLM interface for the selected Ornith generator and its split sampler."""

from __future__ import annotations

import atexit
import json
import os
from dataclasses import fields
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.common.sampling import SamplingParams

from ..reference.hf_reference import resolve_model_path
from .functional_decoder import num_blocks_for_context
from .generator import OrnithGenerator
from .model import OrnithModel


class TTOrnithForCausalLM:
    # Proven by tests/adapter_serving_device_probe.py with split trace replay.
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
        "supports_batched_prefill": True,
    }

    def __init__(self, model, *, max_batch_size, max_model_len, hf_config, vllm_config=None):
        self.model = model
        self.mesh_device = model.mesh_device
        self.max_batch_size = int(max_batch_size)
        self.max_model_len = int(max_model_len)
        self.hf_config = hf_config
        self.generator = None
        self.page_table_blocks = num_blocks_for_context(max_model_len, model.page_block_size)
        text = getattr(hf_config, "text_config", hf_config)
        self.uses_mrope = "mrope_section" in (getattr(text, "rope_parameters", None) or {})
        self.allow_host_sampling = os.environ.get("ORNITH_VLLM_ALLOW_HOST_SAMPLING", "0") == "1"
        self._device_rows = torch.zeros(self.max_batch_size, dtype=torch.bool)
        self._prefilled_rows = torch.zeros(self.max_batch_size, dtype=torch.bool)
        self._pending_device_seeds = torch.zeros(self.max_batch_size, dtype=torch.bool)
        self._last_device_sampling = None
        defaults = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
        self._params = {f.name: [getattr(defaults, f.name)] * self.max_batch_size for f in fields(defaults)}
        self._sampling_key = None
        self.counters = dict(prefills=0, device_decodes=0, host_decodes=0, async_reads=0, sampling_updates=0)
        self._warmed = False
        atexit.register(self.teardown)

    def embed_input_ids(self, input_ids):
        raise NotImplementedError("TT execution uses the generator prefill/decode interface")

    def forward(self, input_ids, positions, **kwargs):
        raise NotImplementedError("TT execution uses the generator prefill/decode interface")

    def compute_logits(self, hidden_states):
        raise NotImplementedError("TT execution uses the generator terminal projection")

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len=None, tt_data_parallel=1, optimizations=None
    ):
        if tt_data_parallel != 1 or optimizations is not None:
            raise ValueError("Ornith uses one tensor-parallel model and the datatype-sweep selected policy")
        if not 1 <= max_batch_size <= 32:
            raise ValueError("Ornith requires max_num_seqs in 1..32")
        contract = json.loads((Path(__file__).parents[1] / "doc/context_contract.json").read_text())
        context = contract["current_supported_context"]
        if max_seq_len is not None and not 1 <= max_seq_len <= context:
            raise ValueError(f"Requested context exceeds the validated {context}-token contract")
        indices = os.environ.get("ORNITH_VLLM_LAYER_INDICES")
        layer_indices = [int(i) for i in indices.split(",")] if indices else None
        if layer_indices is not None:
            logger.warning("Reduced serving target: layers {}; not full-model evidence", layer_indices)
        model = OrnithModel.from_pretrained(
            resolve_model_path(), mesh_device=mesh_device, max_context=context, layer_indices=layer_indices
        )
        logger.info("Serving selected precision: {}", json.dumps(model.precision, sort_keys=True))
        return cls(model, max_batch_size=max_batch_size, max_model_len=max_seq_len or context, hf_config=hf_config)

    @classmethod
    def get_max_tokens_all_users(
        cls, model_name="", num_devices=1, tt_data_parallel=1, max_model_len=None, max_num_seqs=None, **kwargs
    ):
        # Includes a null block so one native-length request fits in the pool.
        context = 262144 if max_model_len is None else int(max_model_len)
        return max(context + 64, int(os.environ.get("ORNITH_VLLM_POOL_TOKENS", "262208")))

    @property
    def cache_path(self):
        return None

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        if self.generator is not None:
            raise RuntimeError("A serving cache is already bound to this adapter")
        blocks, heads, page_size, head_dim = map(int, kv_cache_shape)
        cfg = self.model.hf_config
        expected_heads = max(1, cfg.num_key_value_heads // self.mesh_device.get_num_devices())
        if (heads, page_size, head_dim) != (expected_heads, self.model.page_block_size, cfg.head_dim):
            raise ValueError(f"Serving KV shape {kv_cache_shape} does not match the model")
        cache = self.model.allocate_cache(self.max_batch_size, self.max_model_len, num_blocks=blocks)
        self.generator = OrnithGenerator(
            self.model,
            kv_cache=cache,
            page_table=torch.zeros(self.max_batch_size, self.page_table_blocks, dtype=torch.int32),
            sampling_mode="device",
        )
        logger.info(
            "vLLM-owned cache: {} blocks, selected dtype {}, batch {}",
            blocks,
            self.model.precision["kv_cache_dtype"],
            self.max_batch_size,
        )
        return cache

    def _generator(self, kv_cache):
        if self.generator is None or kv_cache is not self.generator.kv_cache:
            raise ValueError("Pass the exact vLLM cache returned by allocate_kv_cache")
        return self.generator

    def _page_table(self, table, slots=None):
        table = torch.as_tensor(table, dtype=torch.int32)
        if table.ndim != 2 or table.shape[1] > self.page_table_blocks:
            raise ValueError("Invalid scheduler page-table shape")
        if slots is None and tuple(table.shape) == (self.max_batch_size, self.page_table_blocks):
            return table
        result = torch.zeros(self.max_batch_size, self.page_table_blocks, dtype=torch.int32)
        if slots is not None and table.shape[0] == len(slots):
            result[slots, : table.shape[1]] = table
        elif table.shape[0] == self.max_batch_size:
            result[:, : table.shape[1]] = table
        else:
            raise ValueError("Page table must describe the prefill rows or all decode slots")
        return result

    def _device_sampling(self, params):
        device = params is not None
        if not device and not self.allow_host_sampling:
            raise ValueError("Host sampling requires explicit ORNITH_VLLM_ALLOW_HOST_SAMPLING=1 compatibility mode")
        return device

    def _sampling(
        self,
        params,
        *,
        slots=None,
        prompt_tokens=None,
        output_tokens=None,
        fresh_slots=None,
        resume_slots=None,
        fresh_seed_slots=None,
    ):
        rows = list(range(self.max_batch_size)) if slots is None else list(slots)
        for f in fields(SamplingParams):
            value = getattr(params, f.name)
            if isinstance(value, torch.Tensor):
                value = value.reshape(-1).tolist()
            values = list(value) if isinstance(value, (tuple, list)) else [value] * len(rows)
            if len(values) < len(rows):
                raise ValueError(f"Sampling field {f.name} does not cover all scheduled rows")
            for row, val in zip(rows, values):
                self._params[f.name][row] = val
        key = tuple(tuple(self._params[f.name]) for f in fields(SamplingParams))
        restore_history = any(
            self._params[name][slot] != neutral
            for slot in resume_slots or []
            for name, neutral in (("presence_penalty", 0.0), ("frequency_penalty", 0.0), ("repetition_penalty", 1.0))
        )
        if restore_history and (prompt_tokens is None or output_tokens is None):
            raise ValueError("Resuming penalized host rows requires prompt and output histories")
        if key == self._sampling_key and not restore_history and not fresh_seed_slots:
            return
        self.generator.configure_sampling(
            SamplingParams(**self._params),
            prompt_token_ids=self._histories(prompt_tokens),
            generated_token_ids=self._histories(output_tokens),
            fresh_slots=fresh_slots,
            fresh_seed_slots=fresh_seed_slots,
        )
        self._sampling_key = key
        self.counters["sampling_updates"] += 1

    def _histories(self, tokens):
        if tokens is None:
            return None
        return [[int(t) for t in row if int(t) >= 0] for row in tokens][: self.max_batch_size]

    def prefill_forward(
        self,
        *,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        start_pos=None,
        sampling_params=None,
        empty_slots=None,
        enable_trace=True,
        page_tables_per_layer=None,
        **kwargs,
    ):
        gen = self._generator(kv_cache)
        if page_tables_per_layer is not None:
            raise ValueError("Ornith full-attention layers share one KV group; recurrent state is model-owned")
        if any(kwargs.get(k) is not None for k in ("pixel_values", "pixel_values_videos")):
            raise ValueError("This Ornith port serves text only")
        rows = list(range(len(prompt_lens))) if empty_slots is None else list(map(int, empty_slots))
        lengths = list(map(int, prompt_lens))
        starts = [0] * len(rows) if start_pos is None else torch.as_tensor(start_pos).reshape(-1).tolist()
        device = self._device_sampling(sampling_params)
        if device:
            self._sampling(
                sampling_params,
                slots=rows,
                fresh_slots=[row for row, start in zip(rows, starts) if start == 0],
                fresh_seed_slots=[
                    row for row, start in zip(rows, starts) if start > 0 and self._pending_device_seeds[row]
                ],
            )
        out = gen.prefill_forward(
            tokens,
            page_table=self._page_table(page_table, rows),
            kv_cache=kv_cache,
            prompt_lens=lengths,
            slots=rows,
            start_pos=starts,
            return_device_logits=not device,
        )
        if not device:
            out = gen.logits_from(out)[rows, None, :]
        self._prefilled_rows[rows] = True
        self._device_rows[rows] = device
        for row, start in zip(rows, starts):
            if device or start == 0:
                self._pending_device_seeds[row] = not device
        self.counters["prefills"] += len(rows)
        return (out, torch.zeros(len(rows), dtype=torch.int64)) if self.uses_mrope else out

    def decode_forward(
        self,
        *,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        sampling_params=None,
        enable_trace=True,
        read_from_device=True,
        reset_batch=False,
        slot_remap=None,
        prompt_tokens=None,
        output_tokens=None,
        page_tables_per_layer=None,
        **kwargs,
    ):
        gen = self._generator(kv_cache)
        if not enable_trace:
            raise ValueError("Ornith serving decode requires trace replay")
        if page_tables_per_layer is not None:
            raise ValueError("Unexpected per-layer page tables for the uniform full-attention group")
        device = self._device_sampling(sampling_params)
        if slot_remap is not None:
            remap = torch.as_tensor(slot_remap).reshape(-1).to(torch.int64)
            gen.remap_serving_slots(remap)
            self._device_rows = self._device_rows[remap]
            self._prefilled_rows = self._prefilled_rows[remap]
            self._pending_device_seeds = self._pending_device_seeds[remap]
            self._params = {name: [values[i] for i in remap.tolist()] for name, values in self._params.items()}
            self._sampling_key = None
        if device and (
            reset_batch
            or slot_remap is not None
            or self._prefilled_rows.any()
            or self._last_device_sampling is not True
        ):
            # Host steps do not update device penalty counts. Restore active
            # host-owned rows even when their sampling parameters are unchanged.
            # Fresh device-prefilled rows already include their first token.
            active = torch.as_tensor(start_pos).reshape(-1) >= 0
            resume_slots = torch.where(active & ~self._device_rows)[0].tolist()
            fresh_seed_slots = torch.where(active & self._pending_device_seeds)[0].tolist()
            self._sampling(
                sampling_params,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                resume_slots=resume_slots,
                fresh_seed_slots=fresh_seed_slots,
            )
            self._pending_device_seeds[fresh_seed_slots] = False
        boundary = bool(reset_batch or slot_remap is not None or self._prefilled_rows.any() or not device)
        if boundary or self._last_device_sampling != device:
            refresh = ~self._device_rows | self._prefilled_rows
            if not device:
                refresh[:] = True
            gen.refresh_serving_inputs(tokens, start_pos, refresh)
        # In steady state host token/position values may be stale. The canonical
        # sampling trace and position increments are authoritative device state.
        out = gen.decode_forward(
            None,
            None,
            page_table=self._page_table(page_table),
            kv_cache=kv_cache,
            return_logits=not device,
            read_from_device=False,
            sample_on_device=device,
        )
        if boundary or self._last_device_sampling != device:
            self._device_rows = (torch.as_tensor(start_pos).reshape(-1) >= 0) & device
        self._prefilled_rows[:] = False
        self._last_device_sampling = device
        self.counters["device_decodes" if device else "host_decodes"] += 1
        return self.process_decode_output_host(out, is_tokens=device) if read_from_device else out

    def read_decode_output(self, tt_out, async_read=False):
        if not async_read:
            return [tt_out]
        host, event = self.generator.read_output_async(tt_out, return_logits=self._last_device_sampling is False)
        self.counters["async_reads"] += 1
        return [host], [event]

    def process_decode_output_host(self, tt_out, is_tokens=False):
        out = tt_out[0] if isinstance(tt_out, (list, tuple)) else tt_out
        return self.generator.tokens_from(out) if is_tokens else self.generator.logits_from(out)[:, None, :]

    def warmup_model_prefill(self, kv_cache, enable_trace=False, **kwargs):
        gen = self._generator(kv_cache)
        if self._warmed:
            return
        # The pinned plugin represents disabled logprobs with num_logprobs=-2.
        # Prime the canonical key before capture so configuring the empty pool
        # does not release its trace and snapshot the full native KV allocation.
        sampling = SamplingParams(
            temperature=[0.0] * self.max_batch_size,
            top_k=[1] * self.max_batch_size,
            top_p=[1.0] * self.max_batch_size,
            num_logprobs=[-2] * self.max_batch_size,
        )
        gen._configure_sampling(sampling)
        length = min(128, self.max_model_len)
        table = torch.zeros(self.max_batch_size, self.page_table_blocks, dtype=torch.int32)
        table[0] = torch.arange(self.page_table_blocks, dtype=torch.int32)
        # Bind the B1 prefill inputs before capturing the complete trace family
        # over this empty pool; later shape misses preserve this resident shape.
        gen._prepare_prefill_trace([length], [0], [0], table)
        gen.ensure_traces(preserve_cache=False)
        self._sampling(sampling)
        first = self.prefill_forward(
            tokens=torch.zeros(1, length, dtype=torch.int32),
            page_table=table,
            kv_cache=kv_cache,
            prompt_lens=[length],
            empty_slots=[0],
            sampling_params=sampling,
        )
        first = first[0] if isinstance(first, tuple) else first
        tokens = torch.zeros(self.max_batch_size, 1, dtype=torch.int64)
        tokens[0, 0] = first.reshape(-1)[0]
        positions = torch.full((self.max_batch_size,), -1, dtype=torch.int64)
        positions[0] = length
        # Exercise real admission merges before serving. At B1 the width-1
        # RoPE/current-position kernels otherwise first compile after token 1,
        # forcing the complete decode/sampler trace to be recaptured there.
        output = self.decode_forward(
            tokens=tokens,
            start_pos=positions,
            page_table=table,
            kv_cache=kv_cache,
            sampling_params=sampling,
            reset_batch=True,
            read_from_device=False,
        )
        hosts, events = self.read_decode_output(output, async_read=True)
        for event in events:
            ttnn.event_synchronize(event)
        self.process_decode_output_host(hosts, is_tokens=True)
        gen.reset(clear_kv=True)
        gen._reset_seeds()
        gen._reset_output_history()
        gen._write_tokens([0] * self.max_batch_size)
        gen._write_positions([0] * self.max_batch_size)
        self._device_rows[:] = False
        self._prefilled_rows[:] = False
        self._pending_device_seeds[:] = False
        self._last_device_sampling = None
        self._warmed = True

    def warmup_model_decode(self, kv_cache, enable_trace=False, **kwargs):
        self._generator(kv_cache).ensure_traces()

    def teardown(self):
        if self.generator is not None:
            logger.info(
                "Serving execution counters: {}",
                json.dumps({"adapter": self.counters, "generator": self.generator.counters}, sort_keys=True),
            )
            self.generator.teardown()
            self.generator = None
