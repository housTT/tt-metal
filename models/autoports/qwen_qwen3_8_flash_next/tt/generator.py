# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Standalone/readiness generator for Qwen3.8-Flash-Next on P300 TP2.

The optimized path uses the model's split traces: traced decoder segments,
traced terminal projection, traced ``Sampling1D`` candidate sampling, direct
``tt_out_tok`` feedback, and traced in-place position advancement.  Host-side
sampling is available only through the explicit compatibility mode used by
logit-based tests; it is never the measured token-out path.
"""

from __future__ import annotations

import dataclasses
import os
import time
from pathlib import Path
from typing import Iterable, Sequence

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

        This model keeps cache/page/token buffers model-owned and compiles on
        the first concrete call.  The hook validates the same fixed-slot shape
        contract and records sampler parameters without running a hidden CPU or
        replicated fallback path.
        """

        del start_pos, empty_slots
        if kv_cache is not None and kv_cache is not self.model.kv_cache:
            raise ValueError("kv_cache must be the model-owned stable cache")
        token_tensor = torch.as_tensor(tokens, dtype=torch.int64, device="cpu")
        if token_tensor.ndim == 1:
            token_tensor = token_tensor.unsqueeze(0)
        if token_tensor.ndim != 2 or token_tensor.shape[0] != self.model.max_batch:
            raise ValueError(f"compile_prefill tokens must be [{self.model.max_batch}, seq]")
        if page_table is not None:
            pages = torch.as_tensor(page_table, dtype=torch.int32, device="cpu")
            if tuple(pages.shape) != tuple(self.model._default_page_table_host.shape):
                raise ValueError("compile_prefill page_table shape does not match model-owned KV blocks")
        if prompt_lens is not None:
            lengths = torch.as_tensor(prompt_lens, dtype=torch.int32).reshape(-1)
            if int(lengths.numel()) != self.model.max_batch:
                raise ValueError("compile_prefill prompt_lens must match fixed slots")
        if sampling_params is not None:
            self.model.set_sampling_params(
                top_k=getattr(sampling_params, "top_k", 1),
                top_p=getattr(sampling_params, "top_p", 0.0),
                temperature=getattr(sampling_params, "temperature", 1.0),
                seeds=getattr(sampling_params, "seed", None),
            )

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
        if kv_cache is not None and kv_cache is not self.model.kv_cache:
            raise ValueError("kv_cache must be the model-owned stable cache")
        torch.as_tensor(tokens, dtype=torch.int64, device="cpu").reshape(self.model.max_batch)
        torch.as_tensor(start_pos, dtype=torch.int32, device="cpu").reshape(self.model.max_batch)
        if page_table is not None:
            pages = torch.as_tensor(page_table, dtype=torch.int32, device="cpu")
            if tuple(pages.shape) != tuple(self.model._default_page_table_host.shape):
                raise ValueError("compile_decode page_table shape does not match model-owned KV blocks")
        if sampling_params is not None:
            self.model.set_sampling_params(
                top_k=getattr(sampling_params, "top_k", 1),
                top_p=getattr(sampling_params, "top_p", 0.0),
                temperature=getattr(sampling_params, "temperature", 1.0),
                seeds=getattr(sampling_params, "seed", None),
            )

    def prefill_forward(
        self,
        tokens,
        *,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        state: Qwen38BatchState | None = None,
        request_ids=None,
        active_mask=None,
        empty_slots=None,
        read_from_device: bool = True,
        return_all_logits: bool = False,
        **_kwargs,
    ):
        """Low-level mixed-prompt prefill with explicit cache/page state.

        ``empty_slots`` is accepted for common harness compatibility.  This
        port uses a fixed cohort: supplied slots must be the active slot ids.
        """

        token_tensor = torch.as_tensor(tokens, dtype=torch.int64, device="cpu")
        if token_tensor.ndim == 1:
            token_tensor = token_tensor.unsqueeze(0)
        if prompt_lens is None:
            prompt_lens = torch.full((token_tensor.shape[0],), token_tensor.shape[1], dtype=torch.int32)
        lengths = torch.as_tensor(prompt_lens, dtype=torch.int32).reshape(-1)
        if state is None:
            if token_tensor.shape[0] != self.model.max_batch:
                raise ValueError(
                    f"generator model owns {self.model.max_batch} fixed slots; got batch {token_tensor.shape[0]}"
                )
            state = self.allocate_batch_state(
                lengths,
                request_ids=request_ids,
                page_table=page_table,
                active_mask=active_mask,
            )
            page_table = None
        if empty_slots is not None and tuple(int(value) for value in empty_slots) != state.active_slots:
            raise ValueError("empty_slots must match the active fixed-slot cohort")
        logits = self.model.prefill_forward(
            token_tensor,
            state=state,
            prompt_lens=lengths,
            page_table=page_table,
            kv_cache=kv_cache,
            return_all_logits=return_all_logits,
        )
        self.state = state
        if not read_from_device:
            return logits
        host = self.model.logits_to_torch(logits)
        ttnn.deallocate(logits)
        return host

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

        rows = logits[:, -1, :].float()
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
                    seeds=None if seeds is None else [int(seed) + step for seed in seeds],
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
