# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""vLLM integration surface for DeepSeek-V4-Flash (Tenstorrent vLLM plugin / tt-inference-server).

The TT vLLM plugin (`vllm_tt_plugin`) prepends ``TT`` to the HF architecture name and looks the
result up in ``vllm.ModelRegistry``; register this class as ``TTDeepseekV4ForCausalLM`` (the HF
config declares ``architectures: ["DeepseekV4ForCausalLM"]``). The plugin then drives the class
through a small duck-typed interface — there is no required vLLM base class. This adapter
implements exactly what the plugin's loader + model-runner touch:

  * ``initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, ...)`` factory
  * ``prefill_forward(**kwargs)``  -> host logits ``[B, S, V]``   (runner slices ``[:, -1, :]``)
  * ``decode_forward(**kwargs)``   -> host logits ``[B, 1, V]``   (runner slices ``[:, -1, :]``)
  * ``allocate_kv_cache(shape, dtype, num_layers)`` -> benign ``[None] * num_layers``
  * ``cache_path`` property and ``model_capabilities`` class attribute

Serving semantics: this wraps the correct real-weight ``DeepSeekV4Generator`` (tt/generator.py).
The plugin always allocates a paged block pool and passes ``page_table`` / ``kv_cache`` every step,
but this model ignores them and keeps its own running context (faithful at batch=1, which is the
supported concurrency — see the P300X2 spec's ``max_concurrency``). Multi-sequence continuous
batching over the paged pool + the resident/sharded/traced fast decode path (demo/decode_engine.py)
is the documented throughput upgrade — see PRODUCTION_STATUS.md.
"""
from __future__ import annotations

import os

import torch
from loguru import logger

from models.demos.deepseek_v4.tt.generator import DeepSeekV4Generator


class DeepseekV4ForCausalLM:
    """TT vLLM bridge for DeepSeek-V4-Flash. Registered as ``TTDeepseekV4ForCausalLM``."""

    # Read by the plugin platform/runner to decide host vs on-device sampling.
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": False,  # host sampling -> forwards return logits
    }

    def __init__(self, generator: DeepSeekV4Generator, max_seq_len: int = 4096, **kwargs):
        # **kwargs (incl. vllm_config) lets vLLM's `is_vllm_model` init-signature check pass;
        # the real construction goes through `initialize_vllm_model`.
        self.generator = generator
        self.max_seq_len = max_seq_len
        self.tokenizer = generator.tokenizer
        self.hf_config = generator.cfg
        # Running context for the active (batch=1) sequence; ignored KV pool means we
        # reconstruct the sequence ourselves so decode attends over the real context.
        self._ctx: torch.Tensor | None = None

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations=None,
    ):
        """Factory the TT vLLM loader calls (positionally: hf_config, device, max_batch_size, ...)."""
        num_layers = int(
            os.environ.get("DEEPSEEK_V4_NUM_LAYERS", n_layers or hf_config.num_hidden_layers)
        )
        gen = DeepSeekV4Generator(mesh_device, num_layers=num_layers)
        return cls(gen, max_seq_len=max_seq_len)

    # vLLM's registry classifies a model as text-generation via a runtime_checkable
    # Protocol requiring `forward` + `compute_logits` method names. The TT worker never
    # calls these (it drives prefill_forward/decode_forward), but they must exist so
    # `is_text_generation_model` is True and `--runner generate` is accepted.
    def forward(self, *args, **kwargs):  # pragma: no cover - satisfies vLLM Protocol only
        raise NotImplementedError(
            "DeepseekV4ForCausalLM is driven via prefill_forward/decode_forward by the TT "
            "vLLM worker, not the standard vLLM forward()."
        )

    def compute_logits(self, *args, **kwargs):  # pragma: no cover - Protocol only
        raise NotImplementedError(
            "DeepseekV4ForCausalLM returns logits directly from prefill_forward/decode_forward."
        )

    def embed_input_ids(self, *args, **kwargs):  # pragma: no cover - Protocol only
        raise NotImplementedError(
            "DeepseekV4ForCausalLM manages its own embeddings inside the tt-metal forward."
        )

    @property
    def cache_path(self):
        # tt-metal tensor cache location; the generator streams weights per-layer, so this is
        # only used by the plugin's allocators for bookkeeping.
        return os.environ.get("TT_CACHE_PATH")

    # ---- prefill: prompt tokens [B, S] -> host logits [B, S, V] ----
    def prefill_forward(self, *args, **kwargs):
        tokens = kwargs["tokens"]
        prompt_lens = kwargs.get("prompt_lens", None)
        if not torch.is_tensor(tokens):
            tokens = torch.tensor(tokens)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        batch, seqlen = tokens.shape
        vocab = self.hf_config.vocab_size

        # Warmup / empty-prompt calls: return correctly-shaped zeros without running the model.
        if prompt_lens is not None and all(int(x) == 0 for x in prompt_lens):
            return torch.zeros(batch, seqlen, vocab, dtype=torch.float32)

        if batch > 1:
            logger.warning(
                "DeepseekV4ForCausalLM serves batch=1 (max_concurrency=1); got batch={}. "
                "Processing rows independently.",
                batch,
            )

        rows = []
        for i in range(batch):
            plen = int(prompt_lens[i]) if prompt_lens is not None else seqlen
            ids = tokens[i, :plen].unsqueeze(0)
            logits = self.generator._logits(ids)[0]  # [plen, V]
            # Track the active context so subsequent decode steps attend over it (batch=1).
            self._ctx = ids[0].clone()
            if plen < seqlen:  # right-pad logits back to S with the last row
                pad = logits[-1:].expand(seqlen - plen, -1)
                logits = torch.cat([logits, pad], dim=0)
            rows.append(logits)
        return torch.stack(rows, dim=0)  # [B, S, V]

    # ---- decode: new token(s) [B, 1] + start_pos -> host logits [B, 1, V] ----
    def decode_forward(self, *args, **kwargs):
        tokens = kwargs["tokens"]
        if not torch.is_tensor(tokens):
            tokens = torch.tensor(tokens)
        tokens = tokens.reshape(tokens.shape[0], -1)  # [B, 1]
        batch = tokens.shape[0]
        vocab = self.hf_config.vocab_size

        if batch == 1 and self._ctx is not None:
            # Faithful path: append the new token and recompute over the full context.
            self._ctx = torch.cat([self._ctx, tokens[0].to(self._ctx.dtype)], dim=0)
            ids = self._ctx.unsqueeze(0)
            logits = self.generator._logits(ids)[0, -1, :]  # [V]
            return logits.reshape(1, 1, vocab)

        # Fallback (batch>1 or no tracked context): shape-correct logits from the given token(s).
        rows = []
        for i in range(batch):
            ids = tokens[i : i + 1]
            logits = self.generator._logits(ids)[0, -1, :]
            rows.append(logits)
        return torch.stack(rows, dim=0).reshape(batch, 1, vocab)

    def read_decode_output(self, tt_out, async_read=False):
        # decode_forward already returns host tensors.
        return (tt_out, []) if async_read else tt_out

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """The plugin always allocates a paged pool and hands it back each step. This model keeps
        its own context and ignores the pool, so return a benign per-layer placeholder that the
        forwards treat as "no external cache" (they guard on ``any(entry is None ...)``)."""
        logger.info(
            "DeepseekV4ForCausalLM.allocate_kv_cache: returning [None]*{} (model-managed context; "
            "paged pool ignored). kv_cache_shape={}",
            num_layers,
            kv_cache_shape,
        )
        return [None] * num_layers

    # ---- convenience for offline eval drivers (not used by the vLLM server path) ----
    def generate(self, prompt, max_new_tokens=32, temperature=0.0, top_p=1.0):
        return self.generator.generate(prompt, max_new_tokens, temperature, top_p)
