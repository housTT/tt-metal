# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Metal-readiness generator for the optimized GPT-OSS 120B full model.

The default token-out path is canonical split sampling: the shared
``tt_transformers`` generator captures the model graph and the repository
``SamplingGenerator`` captures sampling over the vocab-sharded LM-head output.
The sampled token is written into the persistent decode token input, and the
model advances position and RoPE indices inside its trace.  Steady-state Python
only submits replays and reads the sampled token for the caller; it never feeds
that token back or reads full logits.

``sampling_mode="host"`` is an explicit compatibility path for accuracy tools
that require logits.  It is not the optimized or measured token-out path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

import torch

import ttnn
from models.autoports.openai_gpt_oss_120b.tt.model import HF_CONTEXT_LENGTH, MODEL_LAYERS, PAGE_SIZE, Model, build_model
from models.common.sampling.generator import SamplingParams
from models.tt_transformers.tt.generator import Generator as _TTGenerator

try:
    from models.common.readiness_check.contract import Generator as _ReadinessGenerator
except ModuleNotFoundError:
    # This checkout predates the runner-side ABC, while the autoport pipeline
    # loads generator.py by convention.  Keep the exact method signatures below;
    # when the readiness package is present this class inherits its real ABC.
    class _ReadinessGenerator:  # pragma: no cover - compatibility with old checkouts
        pass


GREEDY = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)

SAMPLER_DECISION = {
    "selected": "models.common.sampling.generator.SamplingGenerator",
    "selected_reason": (
        "It is already integrated with tt_transformers decode capture, persistent tt_out_tok feedback, "
        "per-slot sampling state, penalties, seeds, and log-prob handling."
    ),
    "rejected": "models.common.modules.sampling.sampling_1d.Sampling1D",
    "rejected_reason": (
        "Sampling1D is a useful stateless kernel wrapper but has no matching trace/token-feedback, seed, "
        "penalty, or request-state owner; selecting it would require a redundant custom sampling runtime."
    ),
    "greedy_semantics": "temperature=0 is canonically normalized to temperature=1, top_k=1, top_p=0",
    "force_argmax": False,
}


@dataclass
class TraceEvidence:
    decode_calls: int = 0
    trace_replays: int = 0
    full_input_refreshes: int = 0
    page_table_only_refreshes: int = 0
    page_table_reuses: int = 0
    sampled_token_readbacks: int = 0
    full_logits_readbacks: int = 0
    forced_token_refreshes: int = 0
    host_argmax_calls: int = 0
    token_input_host_refreshes: int = 0
    position_rope_host_refreshes: int = 0
    page_table_host_refreshes: int = 0
    steady_token_input_host_refreshes: int = 0
    steady_position_rope_host_refreshes: int = 0
    steady_page_table_host_refreshes: int = 0
    caller_visible_token_synchronizations: int = 0
    validation_full_logit_synchronizations: int = 0

    def to_dict(self):
        return dict(vars(self))


def _to_torch_page_table(page_table) -> torch.Tensor:
    if isinstance(page_table, torch.Tensor):
        return page_table.to(dtype=torch.int32, device="cpu")
    if isinstance(page_table, ttnn.Tensor):
        tensors = ttnn.get_device_tensors(page_table)
        source = tensors[0] if tensors else page_table
        return ttnn.to_torch(source).to(dtype=torch.int32, device="cpu")
    raise TypeError(f"page_table must be torch.Tensor or ttnn.Tensor, got {type(page_table).__name__}")


class Generator(_ReadinessGenerator):
    """Two-level readiness/serving generator with explicit cache ownership."""

    tokenizer: Any

    def __init__(self, model: Model, model_args, *, kv_cache=None):
        self.model = model
        self.model_args = model_args
        self.mesh_device = model.mesh_device
        self.tokenizer = model_args.tokenizer
        self._kv_cache = model.kv_cache if kv_cache is None else kv_cache
        self._inner = _TTGenerator(
            model=[model],
            model_args=[model_args],
            mesh_device=self.mesh_device,
            tokenizer=self.tokenizer,
        )
        self._page_table = self.allocate_page_table()
        self._dirty_cache = False
        self._last_page_table: torch.Tensor | None = None
        self._last_sampling_mode: str | None = None
        self._decode_started = False
        self._torn_down = False
        self.last_generation_metrics: dict[str, Any] = {}
        self.trace_evidence = TraceEvidence()

    @property
    def kv_cache(self):
        return self._kv_cache

    @property
    def page_table(self):
        return self._page_table

    def allocate_page_table(self) -> torch.Tensor:
        """Allocate fixed private page ranges for all configured request slots."""

        blocks_per_slot = (self.model_args.max_context_len + PAGE_SIZE - 1) // PAGE_SIZE
        return torch.arange(
            self.model_args.max_batch_size * blocks_per_slot,
            dtype=torch.int32,
        ).reshape(self.model_args.max_batch_size, blocks_per_slot)

    def _layer_cache(self, kv_cache):
        cache = self._kv_cache if kv_cache is None else kv_cache
        if len(cache) == 1 and isinstance(cache[0], (list, tuple)) and len(cache[0]) == self.model.n_layers:
            return cache[0]
        return cache

    def _outer_cache(self, kv_cache):
        cache = self._layer_cache(kv_cache)
        return [cache]

    def _validate_prefill(self, tokens, prompt_lens, page_table):
        if tokens.ndim != 2:
            raise ValueError(f"prefill tokens must be [batch, sequence], got {tuple(tokens.shape)}")
        batch, padded_len = tokens.shape
        if len(prompt_lens) != batch:
            raise ValueError(f"prompt_lens has {len(prompt_lens)} entries for batch {batch}")
        if batch > self.model_args.max_batch_size:
            raise ValueError(f"batch {batch} exceeds configured maximum {self.model_args.max_batch_size}")
        for prompt_len in prompt_lens:
            if not 1 <= int(prompt_len) <= min(padded_len, self.model_args.max_context_len):
                raise ValueError(
                    f"prompt length {prompt_len} is outside [1, {min(padded_len, self.model_args.max_context_len)}]"
                )
        if page_table.shape[0] < batch:
            raise ValueError(f"page_table has {page_table.shape[0]} rows for batch {batch}")
        needed_blocks = max((int(length) + PAGE_SIZE - 1) // PAGE_SIZE for length in prompt_lens)
        if page_table.shape[1] < needed_blocks:
            raise ValueError(f"page_table has {page_table.shape[1]} blocks; prompts require {needed_blocks}")

    def _gather_prefill_logits(self, logits) -> torch.Tensor:
        host = logits.cpu(blocking=True)
        shards = ttnn.get_device_tensors(host)
        tp = int(self.mesh_device.shape[1])
        gathered = torch.cat([ttnn.to_torch(shards[index]) for index in range(tp)], dim=-1)
        return gathered[..., : self.model.vocab_size]

    def _prefill_one(self, token_ids, page_table_row, layer_cache, *, return_all_logits):
        logical_len = int(token_ids.shape[-1])
        inputs = self.model.prepare_inputs_prefill(
            token_ids,
            page_table=page_table_row,
            batch_size=1,
            user_id=0,
        )
        embedded, rope_global, rope_local, tt_page_table, *_ = inputs
        logits = self.model.ttnn_prefill_forward(
            embedded,
            rot_mats_global=rope_global,
            rot_mats_local=rope_local,
            user_id=0,
            page_table=tt_page_table,
            get_last_token=-1 if return_all_logits else ((logical_len - 1) // 32) * 32,
            kv_cache=layer_cache,
            batch_size=1,
        )
        gathered = self._gather_prefill_logits(logits)
        self.trace_evidence.full_logits_readbacks += 1
        self.trace_evidence.validation_full_logit_synchronizations += 1
        if return_all_logits:
            return gathered[0, 0, :logical_len]
        row = (logical_len - 1) % 32
        return gathered[0, 0, row : row + 1]

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table,
        kv_cache,
        prompt_lens: List[int],
        return_all_logits: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Fill explicit paged cache state for mixed, non-aligned prompts."""

        del kwargs
        page_table_host = _to_torch_page_table(page_table)
        self._validate_prefill(tokens, prompt_lens, page_table_host)
        layer_cache = self._layer_cache(kv_cache)
        rows = []
        for user, prompt_len in enumerate(prompt_lens):
            prompt_len = int(prompt_len)
            rows.append(
                self._prefill_one(
                    tokens[user : user + 1, :prompt_len],
                    page_table_host[user : user + 1],
                    layer_cache,
                    return_all_logits=return_all_logits,
                )
            )
        self._dirty_cache = True
        self._inner.mode = None
        if return_all_logits:
            max_len = max(int(length) for length in prompt_lens)
            output = torch.zeros(len(rows), max_len, self.model.vocab_size, dtype=rows[0].dtype)
            for user, row in enumerate(rows):
                output[user, : row.shape[0]] = row
            return output
        return torch.stack(rows, dim=0)

    def _record_decode_staging(self, *, page_table, sampling_mode, enable_trace, reset_batch):
        current = page_table.clone()
        changed = self._last_page_table is None or not torch.equal(self._last_page_table, current)
        mode_changed = self._last_sampling_mode is not None and self._last_sampling_mode != sampling_mode
        was_started = self._decode_started
        full_refresh = sampling_mode == "host" or reset_batch or not was_started or mode_changed
        self.trace_evidence.decode_calls += 1
        if enable_trace:
            self.trace_evidence.trace_replays += 1
        if full_refresh:
            self.trace_evidence.full_input_refreshes += 1
            self.trace_evidence.token_input_host_refreshes += 1
            self.trace_evidence.position_rope_host_refreshes += 1
            self.trace_evidence.page_table_host_refreshes += 1
            if was_started:
                self.trace_evidence.steady_token_input_host_refreshes += 1
                self.trace_evidence.steady_position_rope_host_refreshes += 1
                self.trace_evidence.steady_page_table_host_refreshes += 1
        elif changed:
            self.trace_evidence.page_table_only_refreshes += 1
            self.trace_evidence.page_table_host_refreshes += 1
            self.trace_evidence.steady_page_table_host_refreshes += 1
        else:
            self.trace_evidence.page_table_reuses += 1
        self._last_page_table = current
        self._last_sampling_mode = sampling_mode
        self._decode_started = True

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table,
        kv_cache,
        enable_trace: bool = True,
        sampling_mode: str = "device",
        sampling_params: SamplingParams | None = None,
        reset_batch: bool = False,
        force_host_tokens: bool = False,
        prompt_tokens: torch.Tensor | None = None,
        output_tokens: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Decode active and inactive rows through the explicit serving state."""

        del kwargs
        if sampling_mode not in {"device", "host"}:
            raise ValueError(f"sampling_mode must be 'device' or 'host', got {sampling_mode!r}")
        if tokens.ndim != 2 or tokens.shape[1] != 1:
            raise ValueError(f"decode tokens must be [batch, 1], got {tuple(tokens.shape)}")
        if start_pos.ndim != 1 or start_pos.shape[0] != tokens.shape[0]:
            raise ValueError(f"start_pos must be [batch], got {tuple(start_pos.shape)}")
        if tokens.shape[0] > self.model_args.max_batch_size:
            raise ValueError(f"batch {tokens.shape[0]} exceeds configured maximum {self.model_args.max_batch_size}")
        active = start_pos >= 0
        if torch.any(start_pos[active] >= self.model_args.max_context_len):
            raise ValueError(f"decode position exceeds supported context {self.model_args.max_context_len}")
        page_table_host = _to_torch_page_table(page_table)
        if page_table_host.shape[0] < tokens.shape[0]:
            raise ValueError("page_table does not cover every decode row")
        self._record_decode_staging(
            page_table=page_table_host,
            sampling_mode=sampling_mode,
            enable_trace=enable_trace,
            reset_batch=reset_batch,
        )
        if force_host_tokens:
            # The shared generator normally preserves an async-ahead token that
            # device sampling wrote into the persistent decode input.  Teacher
            # forcing is the one case where the caller's token is authoritative.
            # Mark these fixed slots as freshly supplied so its reset path copies
            # the host token/position rather than silently continuing free-run.
            self._inner._slots_prefilled_since_decode.update(range(tokens.shape[0]))
        result = self._inner.decode_forward(
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table_host,
            kv_cache=self._outer_cache(kv_cache),
            enable_trace=enable_trace,
            read_from_device=True,
            sampling_params=(sampling_params or GREEDY) if sampling_mode == "device" else None,
            reset_batch=reset_batch,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )
        self._dirty_cache = True
        if sampling_mode == "device":
            self.trace_evidence.sampled_token_readbacks += 1
            self.trace_evidence.caller_visible_token_synchronizations += 1
            if isinstance(result, tuple):
                result = result[0]
            return result.reshape(-1)[: tokens.shape[0]].to(torch.int64)
        self.trace_evidence.full_logits_readbacks += 1
        self.trace_evidence.validation_full_logit_synchronizations += 1
        if isinstance(result, tuple):
            result = result[0]
        return result[:, 0, :]

    def _device_prefill_sample(self, prompt: torch.Tensor, page_table: torch.Tensor) -> int:
        result = self._inner.prefill_forward_text(
            prompt,
            page_table=page_table,
            kv_cache=[self._kv_cache],
            prompt_lens=[prompt.shape[1]],
            empty_slots=[0],
            enable_trace=False,
            sampling_params=GREEDY,
            warmup_prefill=False,
        )
        tokens = result[0] if isinstance(result, tuple) else result
        self.trace_evidence.sampled_token_readbacks += 1
        self.trace_evidence.caller_visible_token_synchronizations += 1
        self._dirty_cache = True
        return int(torch.as_tensor(tokens).reshape(-1)[0])

    def _eos_token_ids(self) -> set[int]:
        """Return the exact checkpoint generation stop set, with HF fallbacks."""

        values = getattr(getattr(self.model_args, "generation_config", None), "eos_token_id", None)
        if values is None:
            values = getattr(self.model.hf_config, "eos_token_id", None)
        if values is None:
            values = getattr(self.tokenizer, "eos_token_id", None)
        if values is None:
            return set()
        if isinstance(values, (list, tuple, set)):
            return {int(value) for value in values}
        return {int(values)}

    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
        *,
        next_input: Optional[Callable[[int, int], int]] = None,
        enable_trace: bool = True,
        sampling_mode: str = "device",
        stop_on_eos: bool = False,
        **kwargs: Any,
    ) -> List[int]:
        """Generate predictions; teacher forcing refreshes only when requested."""

        del kwargs
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must contain at least one token")
        if len(prompt_token_ids) > self.model_args.max_context_len:
            raise ValueError(
                f"prompt length {len(prompt_token_ids)} exceeds supported context {self.model_args.max_context_len}"
            )
        if max_new_tokens < 0 or len(prompt_token_ids) + max_new_tokens > self.model_args.max_context_len:
            raise ValueError("prompt plus requested generation exceeds the configured context")
        if max_new_tokens == 0:
            return []
        if sampling_mode not in {"device", "host"}:
            raise ValueError(f"sampling_mode must be 'device' or 'host', got {sampling_mode!r}")
        if next_input is None and sampling_mode == "device" and not enable_trace:
            raise ValueError("optimized token-out generation requires enable_trace=True")

        self.reset()
        generation_start = time.perf_counter()
        prompt = torch.tensor([prompt_token_ids], dtype=torch.long)
        page_table = self._page_table[:1]
        if sampling_mode == "device":
            predicted = self._device_prefill_sample(prompt, page_table)
        else:
            logits = self.prefill_forward(
                prompt,
                page_table=page_table,
                kv_cache=self._kv_cache,
                prompt_lens=[prompt.shape[1]],
            )
            predicted = int(torch.argmax(logits[0, 0]).item())
            self.trace_evidence.host_argmax_calls += 1
        first_token_time = time.perf_counter()

        predictions = [predicted]
        eos_ids = self._eos_token_ids()
        if stop_on_eos and predicted in eos_ids:
            return predictions

        feed = int(next_input(0, predicted)) if next_input is not None else predicted
        start_pos = torch.tensor([len(prompt_token_ids)], dtype=torch.int64)
        decode_start = time.perf_counter()
        for step in range(1, max_new_tokens):
            teacher_forced = next_input is not None
            result = self.decode_forward(
                torch.tensor([[feed]], dtype=torch.long),
                start_pos,
                page_table=page_table,
                kv_cache=self._kv_cache,
                enable_trace=enable_trace,
                sampling_mode=sampling_mode,
                sampling_params=GREEDY,
                reset_batch=(step == 1 or teacher_forced),
                force_host_tokens=teacher_forced,
                prompt_tokens=prompt if step == 1 else None,
            )
            if sampling_mode == "device":
                predicted = int(result[0])
            else:
                predicted = int(torch.argmax(result[0]).item())
                self.trace_evidence.host_argmax_calls += 1
            predictions.append(predicted)
            if stop_on_eos and predicted in eos_ids:
                break
            if teacher_forced:
                feed = int(next_input(step, predicted))
                self.trace_evidence.forced_token_refreshes += 1
            else:
                # Deliberately leave ``feed`` stale.  The sampling trace wrote the
                # authoritative token into the persistent device input, and the
                # next replay does not copy this Python value.
                feed = predicted if sampling_mode == "host" else feed
            start_pos += 1
        generation_end = time.perf_counter()
        decode_tokens = max(0, len(predictions) - 1)
        decode_seconds = max(0.0, generation_end - decode_start)
        self.last_generation_metrics = {
            "sampling_mode": sampling_mode,
            "teacher_forcing": next_input is not None,
            "enable_trace": enable_trace,
            "prompt_tokens": len(prompt_token_ids),
            "output_tokens": len(predictions),
            "ttft_seconds": first_token_time - generation_start,
            "decode_seconds": decode_seconds,
            "decode_tokens_per_second_per_user": decode_tokens / decode_seconds if decode_seconds else None,
            "total_seconds": generation_end - generation_start,
        }
        return predictions

    def reset(self) -> None:
        """Clear cache content and per-request state while retaining warmed traces."""

        if self._dirty_cache:
            self.model.clear_kv_caches()
        self._dirty_cache = False
        self._inner.mode = None
        self._inner.prev_page_table = None
        self._inner._prev_on_device_sampling = None
        self._inner._slots_prefilled_since_decode = set()
        self._last_page_table = None
        self._last_sampling_mode = None
        self._decode_started = False
        self.trace_evidence = TraceEvidence()

    def capability_report(self) -> dict[str, Any]:
        return {
            "model": "openai/gpt-oss-120b",
            "mesh_shape": [int(v) for v in self.mesh_device.shape],
            "max_batch_size": self.model_args.max_batch_size,
            "max_context_length": self.model_args.max_context_len,
            "page_size": PAGE_SIZE,
            "cache_owner": "generator unless explicit kv_cache is passed to the low-level API",
            "inactive_row_sentinel": -1,
            "sampling": dict(SAMPLER_DECISION),
            "trace_evidence": self.trace_evidence.to_dict(),
            "last_generation_metrics": dict(self.last_generation_metrics),
            "host_sampling_compatibility": True,
        }

    def teardown(self):
        if self._torn_down:
            return
        self._torn_down = True
        destructor = getattr(self._inner, "__del__", None)
        if callable(destructor):
            destructor()
        # ``__del__`` is not an ordinary teardown API and does not clear the
        # handles it released.  Clear them before dropping our last reference,
        # otherwise Python invokes it a second time and double-releases traces.
        for name in (
            "trace_id_prefill",
            "trace_id_prefill_sampling",
            "trace_ids_decode",
            "_bucket_trace_store",
            "trace_ids",
        ):
            store = getattr(self._inner, name, None)
            if hasattr(store, "clear"):
                store.clear()
        self._inner = None


def build_generator(model_dir: str | Path, mesh_device, **kwargs) -> Generator:
    """Readiness discovery entry point; this stage intentionally stops before vLLM."""

    del model_dir
    override_num_layers = kwargs.pop("override_num_layers", None)
    num_layers = int(override_num_layers or kwargs.pop("num_layers", MODEL_LAYERS))
    max_context_length = int(kwargs.pop("max_seq_len", kwargs.pop("max_context_length", HF_CONTEXT_LENGTH)))
    max_batch_size = int(kwargs.pop("max_batch_size", 1))
    snapshot_path = kwargs.pop("snapshot_path", None)
    tensor_cache_path = kwargs.pop("tensor_cache_path", None)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unknown build_generator arguments: {unknown}")
    model, args, kv_cache = build_model(
        mesh_device,
        snapshot_path=snapshot_path,
        tensor_cache_path=tensor_cache_path,
        max_batch_size=max_batch_size,
        max_context_length=max_context_length,
        num_layers=num_layers,
        allow_reduced_model=num_layers != MODEL_LAYERS,
    )
    return Generator(model, args, kv_cache=kv_cache)


__all__ = ["GREEDY", "Generator", "SAMPLER_DECISION", "TraceEvidence", "build_generator"]
