# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""vLLM integration surface for DeepSeek-V4-Flash (tt-inference-server / tt-metal vLLM plugin).

Mirrors the shape of the production adapters (`models/demos/deepseek_v3/tt/generator_vllm.py`,
`models/tt_transformers/tt/generator_vllm.py`): a `DeepseekV4ForCausalLM` the tt-metal vLLM
plugin instantiates via `initialize_vllm_model`, exposing `prefill_forward` / `decode_forward`
/ `allocate_kv_cache`. It wraps the correct real-weight `DeepSeekV4Generator` (tt/generator.py).

Status (see PRODUCTION_STATUS.md): this bridges to the CORRECTNESS forward, so it produces valid
tokens for accuracy evals driven single-sequence. Full production serving (paged KV attention +
continuous batching + the resident/sharded/traced throughput path from demo/decode_engine.py)
is the remaining work; the methods below raise a clear NotImplementedError where that path is
required rather than silently degrade.
"""
from __future__ import annotations

import os

import torch

from models.demos.deepseek_v4.tt.generator import DeepSeekV4Generator


class DeepseekV4ForCausalLM:
    """tt-metal vLLM bridge for DeepSeek-V4-Flash. Instantiated by the plugin via
    `initialize_vllm_model`; drives generation through `DeepSeekV4Generator`."""

    def __init__(self, generator: DeepSeekV4Generator, max_seq_len: int = 4096):
        self.generator = generator
        self.max_seq_len = max_seq_len
        self.tokenizer = generator.tokenizer

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations=None
    ):
        """Entry point the tt-metal vLLM plugin calls to build the model on the device mesh.
        Model path resolved from env (`DEEPSEEK_V4_HF_MODEL`) or the local HF cache snapshot."""
        num_layers = int(os.environ.get("DEEPSEEK_V4_NUM_LAYERS", hf_config.num_hidden_layers))
        gen = DeepSeekV4Generator(mesh_device, num_layers=num_layers)
        return cls(gen, max_seq_len=max_seq_len)

    # ---- generation (single-sequence, correctness path — drivable for accuracy evals) ----
    def prefill_forward(self, tokens, *args, **kwargs):
        """Prefill: prompt token ids [1, S] -> next-token logits [1, vocab]."""
        if not torch.is_tensor(tokens):
            tokens = torch.tensor(tokens).reshape(1, -1)
        return self.generator.prefill(tokens)

    def decode_forward(self, context_tokens, *args, **kwargs):
        """Decode one step given the full context [1, S] -> next-token logits [1, vocab].
        (Correctness path re-runs the sequence; paged single-token decode is the throughput upgrade.)"""
        if not torch.is_tensor(context_tokens):
            context_tokens = torch.tensor(context_tokens).reshape(1, -1)
        return self.generator.decode(context_tokens)

    def generate(self, prompt, max_new_tokens=32, temperature=0.0, top_p=1.0):
        """Convenience text-in/text-out used by offline eval drivers."""
        return self.generator.generate(prompt, max_new_tokens, temperature, top_p)

    def allocate_kv_cache(self, *args, **kwargs):
        """Paged KV-cache allocation for continuous batching. NOT yet implemented: the correctness
        forward recomputes the sequence per step. The paged/resident/sharded decode path lives in
        demo/decode_engine.py (perf-validated) and must be folded in for full vLLM serving."""
        raise NotImplementedError(
            "Paged KV cache / continuous batching not yet wired for DeepSeek-V4. Use the "
            "single-sequence generate()/prefill_forward()/decode_forward() path for accuracy evals; "
            "see PRODUCTION_STATUS.md for the throughput-integration plan."
        )
