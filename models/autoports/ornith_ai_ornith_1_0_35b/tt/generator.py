# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serving-shaped generator for ornith-ai/Ornith-1.0-35B on the 1x4 Blackhole ring.

Implements ``models.common.readiness_check.contract.Generator`` on top of
:class:`~models.autoports.ornith_ai_ornith_1_0_35b.tt.model.OrnithModel`, at the two levels the
contract asks for:

* **low level** — :meth:`OrnithGenerator.prefill_forward` and
  :meth:`OrnithGenerator.decode_forward` take explicit ``page_table`` / ``kv_cache`` /
  ``prompt_lens`` / ``start_pos`` state, accept mixed-length prompts, address fixed batch slots and
  leave inactive rows (position ``-1``) alone. This is the surface a serving adapter drives;
* **high level** — :meth:`OrnithGenerator.generate` owns the cache, the page table and the decode
  loop, and is what the readiness runners call.

Token-out decode is the canonical split-sampling path:

1. one captured **model** trace, token in -> vocab-sharded sampler-ready logits out, which also
   advances ``current_pos`` and the RoPE index on device with ``ttnn.plus_one``;
2. one captured **sampling** trace owned by ``models.common.sampling.SamplingGenerator``, called
   with ``tt_out_tok`` pointing at the persistent decode token buffer.

So the sampled token of replay ``N`` *is* the token input of replay ``N+1`` — there is no host
argmax, no full-logits readback, no untraced sampling and no Python token-feedback loop in the
measured path. ``sampling_mode="host"`` is the explicit compatibility mode for tests that need host
sampling; it is never the measured path.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, List, Optional

import torch
from loguru import logger

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import (
    DEFAULT_TOPK_GROUPS,
    MAX_SAMPLING_BATCH,
    TILE,
    OrnithModel,
    load_text_config,
    resolve_model_path,
)
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model_config import HF_MODEL_ID
from models.common.readiness_check.contract import Generator, NextInputFn

#: Default supported context the generator allocates a KV cache for.
#:
#: The **model** advertises the HF context (262144) and nothing here reduces it: this is the
#: *cache allocation* the default constructor makes, i.e. how many paged blocks are reserved, which
#: is a deployment choice exactly as it is in vLLM's ``--num-gpu-blocks``. Pass ``max_context`` to
#: build the cache for the whole advertised window; ``doc/full_model/README.md`` records the byte
#: arithmetic and the largest feasible context at each batch.
DEFAULT_CACHE_CONTEXT = 262144

_GREEDY = "greedy"


def _greedy_params():
    from models.common.sampling import SamplingParams

    # temperature 0 is the canonical greedy request; `format_sampling_params` turns it into the
    # device's compact greedy representation (temp=1, k=1, p=0) for every row.
    return SamplingParams(temperature=0.0, top_k=1, top_p=1.0)


class OrnithGenerator(Generator):
    """The readiness-check / serving generator for Ornith-1.0-35B."""

    def __init__(
        self,
        model: OrnithModel,
        *,
        tokenizer=None,
        max_batch_size: int = 1,
        cache_context: int = DEFAULT_CACHE_CONTEXT,
        sampling_mode: str = "device",
        max_top_k: int = 32,
        pad_logits_to_power_of_2: bool = False,
        topk_num_groups: int = DEFAULT_TOPK_GROUPS,
        kv_cache=None,
        page_table=None,
    ):
        if sampling_mode not in ("device", "host"):
            raise ValueError(f"sampling_mode must be 'device' or 'host', got {sampling_mode!r}")
        self.model = model
        self.mesh_device = model.mesh_device
        self.tokenizer = tokenizer
        self.max_batch_size = int(max_batch_size)
        self.sampling_mode = sampling_mode
        self.cache_context = min(int(cache_context), model.max_context)

        self.blocks_per_user = model.blocks_per_user(self.cache_context)
        self.total_blocks = self.blocks_per_user * self.max_batch_size

        model.allocate_state(self.max_batch_size)
        if kv_cache is None:
            self.kv_cache = model.allocate_kv_cache(self.total_blocks)
            self.owns_cache = True
        else:
            self.kv_cache = model.attach_kv_cache(kv_cache)
            self.owns_cache = False

        if page_table is None:
            page_table = torch.arange(self.total_blocks, dtype=torch.int32).reshape(
                self.max_batch_size, self.blocks_per_user
            )
        self.page_table = torch.as_tensor(page_table).to(torch.int32)

        self.sampling = model.build_sampler(
            max_batch_size=self.max_batch_size,
            max_top_k=max_top_k,
            pad_to_power_of_2=pad_logits_to_power_of_2,
            topk_num_groups=topk_num_groups,
        )
        self._sampling_params_key = None
        self._apply_sampling_params(_greedy_params())

        # Persistent decode trace state.
        self._trace_id = None
        self._trace_logits = None
        self._trace_inputs = None
        self._prev_page_table = None
        self._sampling_trace_ready = False
        self._kv_cache_at_capture = None
        self._warned_page_table_substitution = False
        #: Program-cache size when the traces were captured. See :meth:`_ensure_traces_replay_safe`.
        self._program_cache_at_capture = None
        #: How many times the traces had to be re-captured because something compiled after them.
        self.trace_recaptures = 0
        self.counters = model.counters
        self.perf: dict[str, Any] = {}
        self._eos_ids = self._resolve_eos_ids()

    # ------------------------------------------------------------------ setup helpers
    def _resolve_eos_ids(self) -> set:
        ids = set()
        cfg = self.model.hf_config
        for value in (getattr(cfg, "eos_token_id", None), getattr(self.tokenizer, "eos_token_id", None)):
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                ids.update(int(v) for v in value)
            else:
                ids.add(int(value))
        return ids

    def _apply_sampling_params(self, params):
        from models.common.sampling import format_sampling_params

        formatted = format_sampling_params(params, max(TILE, self.sampling.tt_sampling.max_batch_size))
        key = (tuple(formatted.top_k), tuple(formatted.top_p), tuple(formatted.temperature))
        if key == self._sampling_params_key:
            return
        self.sampling.reset_sampling_params(formatted)
        self._sampling_params_key = key

    def _resolve_page_table(self, page_table, kv_cache, where: str):
        """The page table a call should actually use, identically for prefill and decode.

        A caller-owned table without a caller-owned cache is refused (with one warning, then
        silently) because a foreign table addresses blocks the internal cache does not have. Honour
        it on one of the two calls only and the request prefills into one set of blocks and decodes
        out of another, reading pages nothing ever wrote - so both entry points resolve it here and
        cannot disagree.
        """
        if page_table is None:
            return self.page_table
        if kv_cache is None:
            if not self._warned_page_table_substitution:
                self._warned_page_table_substitution = True
                logger.warning(
                    f"{where} was given a page_table but no kv_cache, so the generator's own page table is "
                    "used instead: a foreign table can address blocks the internal cache does not have. Pass "
                    "kv_cache as well to drive caller-owned state."
                )
            return self.page_table
        return torch.as_tensor(page_table).to(torch.int32)

    # ------------------------------------------------------------------ trace management
    def _host_decode_inputs(self, tokens, positions, page_table):
        return self.model.prepare_decode_inputs_host(tokens, positions, page_table)

    def _ensure_decode_trace(self):
        """Warm-compile and capture the model decode trace and the sampling trace, once.

        **This destroys any live prompt state, so it must run before a prefill, never after.** The
        warm-compile pass really executes a decode step: it advances every DeltaNet recurrent row
        and writes one paged KV entry. That contamination has to be wiped, and the wipe
        (``model.reset_state()``) zeroes the recurrent/conv state and the whole paged cache. A
        generator that captured lazily on its first ``decode_forward`` would therefore answer the
        first request from an *empty* cache — coherent-looking output with the prompt thrown away.
        Both public entry points (:meth:`prefill_forward` and :meth:`generate`) call this before
        writing anything, and ``OrnithModel.state_is_live`` turns the remaining orderings into a
        loud error instead of silent wrong output. That flag lives on the **model** deliberately:
        every write path is there, including ``prefill_forward_single``, which is what the
        high-level ``generate`` uses, so a generator-side flag would have missed it.

        Order inside: wipe, warm both graphs, wipe again, restage the inputs, capture.

        * the **first** wipe is there to compile the wipe's own programs while compiling is still
          safe. Everything compiled after ``end_trace_capture`` lands in the window §5.1 of the
          README is about, and a wipe that first compiled there would be invisible to
          ``_ensure_traces_replay_safe``;
        * the **second** wipe removes what the warm passes did. It compiles nothing (the first wipe
          already did), so the program cache is unchanged between the warm passes and the capture,
          which is what the tracing skill asks for;
        * both graphs are warm-compiled before the model trace is captured, and the sampling capture
          then runs with ``skip_precompile=True``. Precompiling the sampler against the model
          trace's *output* tensor — ``SamplingGenerator.capture_trace``'s default — means executing a
          full sampling graph over a live trace-region buffer while a captured trace already exists,
          and on the 40-layer model that hung the mesh inside ``all_gather_async``.
        """
        if self._trace_id is not None:
            return
        if self.model.state_is_live:
            raise RuntimeError(
                "the decode traces have to be captured before a prefill, not after: capture warm-compiles a "
                "real decode step and then wipes the DeltaNet state and the paged KV cache, which would throw "
                "this prompt away. Call _ensure_decode_trace() (or generate()/prefill_forward(), which do it "
                "for you) before writing prompt state, or reset() first."
            )
        model = self.model
        batch = self.max_batch_size
        mesh = self.mesh_device

        dummy_tokens = torch.zeros(batch, dtype=torch.int32)
        dummy_pos = torch.zeros(batch, dtype=torch.int32)
        host = self._host_decode_inputs(dummy_tokens, dummy_pos, self.page_table)
        device_inputs = [ttnn.to_device(t, device=mesh) if t is not None else None for t in host]
        tok_buf, pos_buf, rot_buf, page_buf = device_inputs
        self._trace_inputs = device_inputs

        logger.info("compiling the state wipe before anything is captured")
        model.reset_state()

        logger.info("warm-compiling the decode graph before trace capture")
        warm = model.ttnn_decode_forward(tok_buf, pos_buf, rot_buf, page_buf)
        ttnn.synchronize_device(mesh)
        if self.sampling_mode == "device":
            logger.info("warm-compiling the sampling graph on a non-trace logits tensor")
            self.sampling.sample(logits=warm, tt_out_tok=tok_buf, enable_trace=False)
            ttnn.synchronize_device(mesh)
        ttnn.deallocate(warm)

        # Everything the warm passes touched is state, and it is all wiped here — before capture,
        # not after, and compiling nothing because the wipe above already did.
        model.reset_state()
        self._prev_page_table = None

        # The warm passes advanced the positions and overwrote the token buffer; put the
        # capture-time values back so the captured graph is recorded over exactly the state the
        # first replay will see.
        self._refresh_inputs(device_inputs, dummy_tokens, dummy_pos, self.page_table, force=True)
        ttnn.synchronize_device(mesh)

        logger.info("capturing the model decode trace and the sampling trace")
        self._capture_traces()

    def _capture_traces(self):
        """Capture the model decode trace and the sampling trace over the persistent inputs.

        Capture **records** without executing, and every program involved is already compiled, so
        this is state-preserving: it can be called between a prefill and its decode loop without
        disturbing the KV cache, the DeltaNet state or the positions.
        """
        mesh = self.mesh_device
        tok_buf, pos_buf, rot_buf, page_buf = self._trace_inputs
        # Capture *records* without executing, but it does run the Python body of
        # `ttnn_decode_forward`, which marks the model's state live. Recording is not writing, so
        # the flag is restored: otherwise a fresh capture would claim a prompt that does not exist
        # and the next `_ensure_decode_trace` would refuse for nothing.
        was_live = self.model.state_is_live
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        tt_logits = self.model.ttnn_decode_forward(tok_buf, pos_buf, rot_buf, page_buf)
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh)
        self._trace_id = trace_id
        self._trace_logits = tt_logits
        if self.sampling_mode == "device":
            self.sampling.capture_trace(logits=tt_logits, tt_out_tok=tok_buf, skip_precompile=True)
            self._sampling_trace_ready = True
        self._program_cache_at_capture = mesh.num_program_cache_entries()
        self.model.state_is_live = was_live
        # A caller-owned cache attached after capture would leave the trace bound to the previous
        # tensors, so the identity is part of what makes a capture still valid.
        self._kv_cache_at_capture = self.model.kv_cache

    def _release_traces(self):
        if self._sampling_trace_ready:
            self.sampling.reset_trace()
            self._sampling_trace_ready = False
        if self._trace_id is not None:
            ttnn.release_trace(self.mesh_device, self._trace_id)
            self._trace_id = None
        self._trace_logits = None

    def _ensure_traces_replay_safe(self):
        """Re-capture the traces if anything has been compiled since they were captured.

        This is the fix for a real, permanent corruption, not a precaution. tt-metal warns that
        *"allocating device buffers is unsafe due to the existence of an active trace; these buffers
        may be corrupted once a trace is executed"* (``allocator.cpp``) — the trace's intermediates
        are handed back to the allocator at ``end_trace_capture`` while the captured commands still
        write to those addresses. A **program's kernel binaries are such a buffer**, and unlike an
        activation they live in the program cache for the rest of the process. So:

        1. request 1 captures the decode traces;
        2. its prefill compiles the programs for *its* prompt length, and their binaries land on
           addresses the decode trace writes;
        3. the first trace replay overwrites those binaries;
        4. every later prefill that reuses those programs executes corrupt code — the model emits
           token 0 and then gibberish, for **every** prompt, permanently.

        ``doc/full_model/logs/probe_bisect.py`` isolates it: compiling the two prompt lengths before
        capture leaves both prefills stable across replays, and compiling the same two lengths after
        capture destroys both after a single replay.

        The check itself is a single integer comparison. A re-capture costs a few hundred
        milliseconds and only happens when a request actually introduced a new program — which,
        because the prefill program set is keyed by logical prompt length, means the first request
        at each new length and no request after that.
        """
        if self._trace_id is None:
            return
        entries = self.mesh_device.num_program_cache_entries()
        cache_changed = self.model.kv_cache is not getattr(self, "_kv_cache_at_capture", None)
        if entries == self._program_cache_at_capture and not cache_changed:
            return
        reason = (
            "the attached KV cache changed"
            if cache_changed
            else f"{entries - self._program_cache_at_capture} program(s) were compiled after capture, and their "
            "kernel binaries would be overwritten by the next replay"
        )
        logger.info(f"re-capturing the decode traces: {reason}")
        ttnn.synchronize_device(self.mesh_device)
        self._release_traces()
        self._capture_traces()
        self.trace_recaptures += 1

    def _refresh_inputs(self, device_inputs, tokens, positions, page_table, *, force=False, skip_tokens=False):
        """Copy host decode inputs into the persistent trace buffers.

        Only what actually changed is copied. In the steady state of a free-running traced decode
        this copies **nothing**: the token comes from ``tt_out_tok`` and the positions are advanced
        by ``ttnn.plus_one`` inside the trace.
        """
        host = self._host_decode_inputs(tokens, positions, page_table)
        tok_h, pos_h, rot_h, page_h = host
        tok_d, pos_d, rot_d, page_d = device_inputs
        if not skip_tokens:
            ttnn.copy_host_to_device_tensor(tok_h, tok_d)
            self.counters["token_refreshes"] += 1
        ttnn.copy_host_to_device_tensor(pos_h, pos_d)
        ttnn.copy_host_to_device_tensor(rot_h, rot_d)
        self.counters["position_refreshes"] += 1
        self.counters["rope_refreshes"] += 1
        if page_h is not None and page_d is not None:
            changed = force or self._prev_page_table is None or not torch.equal(self._prev_page_table, page_table)
            if changed:
                ttnn.copy_host_to_device_tensor(page_h, page_d)
                self._prev_page_table = page_table.clone()
                self.counters["page_table_refreshes"] += 1

    def _refresh_page_table_only(self, page_table):
        """Refresh just the page-table trace input, and only when it changed."""
        page_d = self._trace_inputs[3]
        if page_d is None:
            return False
        if self._prev_page_table is not None and torch.equal(self._prev_page_table, page_table):
            return False
        host = self._host_decode_inputs(
            torch.zeros(self.max_batch_size, dtype=torch.int32),
            torch.zeros(self.max_batch_size, dtype=torch.int32),
            page_table,
        )
        ttnn.copy_host_to_device_tensor(host[3], page_d)
        self._prev_page_table = torch.as_tensor(page_table).to(torch.int32).clone()
        self.counters["page_table_refreshes"] += 1
        return True

    def _write_tokens(self, tokens):
        """Host override of the decode token buffer (teacher forcing / host sampling)."""
        host = self._host_decode_inputs(tokens, torch.zeros(self.max_batch_size, dtype=torch.int32), None)
        ttnn.copy_host_to_device_tensor(host[0], self._trace_inputs[0])
        self.counters["token_refreshes"] += 1

    def _write_positions(self, positions):
        host = self._host_decode_inputs(torch.zeros(self.max_batch_size, dtype=torch.int32), positions, None)
        ttnn.copy_host_to_device_tensor(host[1], self._trace_inputs[1])
        ttnn.copy_host_to_device_tensor(host[2], self._trace_inputs[2])
        self.counters["position_refreshes"] += 1
        self.counters["rope_refreshes"] += 1

    def _read_tokens(self) -> torch.Tensor:
        """The caller-visible readback: the sampled tokens of the last replay."""
        whole = ttnn.to_torch(
            self._trace_inputs[0],
            mesh_composer=ttnn.concat_mesh_to_tensor_composer(self.mesh_device, dim=0),
        )
        return whole.reshape(-1)[: self.max_batch_size].to(torch.int64)

    def _decode_step_traced(self) -> None:
        ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=False)
        self.counters["decode_calls"] += 1

    def _sample_traced(self):
        self.sampling.sample(logits=self._trace_logits, tt_out_tok=self._trace_inputs[0], enable_trace=True)

    # ------------------------------------------------------------------ low-level API
    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table=None,
        kv_cache: Any = None,
        prompt_lens: Optional[List[int]] = None,
        return_all_logits: bool = False,
        start_pos=0,
        continue_from_state: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Low-level prefill. See ``models.common.readiness_check.contract.Generator``.

        ``continue_from_state`` chunks one long prompt across several calls: pass the chunk's
        absolute ``start_pos`` and ``continue_from_state=True`` for every chunk after the first, and
        the DeltaNet state carries over instead of being zeroed. Batch 1 only - see
        :meth:`OrnithModel.prefill_forward`.

        Cache ownership is explicit: pass ``kv_cache`` (and the matching ``page_table``) to drive the
        generator's model against caller-owned state, or leave both ``None`` to use the cache and
        page table this generator allocated. A caller that hands a page table without a cache gets
        the internal page table, because a foreign table can address blocks the internal cache does
        not have.

        The decode traces are captured **here**, before the prompt is written, not lazily on the
        first ``decode_forward``. Capture warm-compiles a real decode step and then wipes the state
        it touched, so a capture that happened after the prefill would throw the prompt's KV cache
        and DeltaNet state away - see :meth:`_ensure_decode_trace`.
        """
        del kwargs
        tokens = torch.as_tensor(tokens)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if not continue_from_state:
            # A continuation call is by definition preceded by the chunk that captured the traces
            # and wrote the state this one continues from; capturing here would wipe it.
            self._ensure_decode_trace()
        table = self._resolve_page_table(page_table, kv_cache, "prefill_forward")
        return self.model.prefill_forward(
            tokens,
            page_table=table,
            prompt_lens=prompt_lens,
            kv_cache=kv_cache,
            return_all_logits=return_all_logits,
            start_pos=start_pos,
            continue_from_state=continue_from_state,
        )

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table=None,
        kv_cache: Any = None,
        enable_trace: bool = True,
        sample_on_device: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Low-level decode. One step for every row of a fixed-slot batch.

        ``start_pos`` is each row's absolute KV position; a row whose position is ``-1`` is
        **inactive** and its cache is not written (``ttnn.plus_one(..., skip_negative_entries=True)``
        keeps it inactive across replays too).

        Returns torch logits ``[batch, vocab]``, or the sampled token ids ``[batch]`` when
        ``sample_on_device`` is set — the form a serving adapter with device sampling consumes.
        """
        del kwargs
        if sample_on_device and self.sampling_mode != "device":
            raise ValueError(
                "sample_on_device=True needs sampling_mode='device'; this generator was built in the host "
                "compatibility mode, which has no sampler graph and no sampling trace to run"
            )
        if kv_cache is not None:
            self.model.attach_kv_cache(kv_cache)
        table = self._resolve_page_table(page_table, kv_cache, "decode_forward")
        tokens = torch.as_tensor(tokens).reshape(-1)
        positions = torch.as_tensor(start_pos).reshape(-1)

        if not enable_trace:
            host = self._host_decode_inputs(tokens, positions, table)
            device_inputs = [ttnn.to_device(t, device=self.mesh_device) if t is not None else None for t in host]
            logits = self.model.ttnn_decode_forward(*device_inputs)
            self.counters["decode_calls"] += 1
            if sample_on_device:
                self.sampling.sample(logits=logits, tt_out_tok=device_inputs[0], enable_trace=False)
                out = ttnn.to_torch(
                    device_inputs[0], mesh_composer=ttnn.concat_mesh_to_tensor_composer(self.mesh_device, dim=0)
                ).reshape(-1)[: self.max_batch_size]
            else:
                out = self.model.decode_logits_to_host(logits)
            ttnn.deallocate(logits)
            return out

        self._ensure_decode_trace()
        self._ensure_traces_replay_safe()
        self._refresh_inputs(self._trace_inputs, tokens, positions, table)
        # A trace replay does not run the Python body that would mark the state live, so the
        # generator marks it here: after this step the cache and the DeltaNet rows hold a request.
        self.model.state_is_live = True
        self._decode_step_traced()
        if sample_on_device:
            self._sample_traced()
            return self._read_tokens()
        ttnn.synchronize_device(self.mesh_device)
        return self.model.decode_logits_to_host(self._trace_logits)

    # ------------------------------------------------------------------ high-level API
    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
        *,
        next_input: Optional[NextInputFn] = None,
        enable_trace: bool = True,
        stop_on_eos: Optional[bool] = None,
        sampling_params=None,
        reset: bool = True,
        **kwargs: Any,
    ) -> List[int]:
        """HF-style greedy generation over the low-level path.

        ``enable_trace`` is explicit and honoured: ``True`` runs every decode step through the
        captured trace (what the readiness teacher-forcing runner requires); ``False`` is a
        model-local eager debug path and is never used for evidence.

        ``reset`` (default ``True``) wipes the DeltaNet state and the paged KV cache before the
        prompt is written, which is what a fresh request wants. ``reset=False`` keeps the cache -
        useful for inspecting state across calls - but the prompt still starts at position 0 and the
        prefill pack is still zeroed for it, so it is not a continuation API; that is
        ``prefill_forward(..., continue_from_state=True)``.
        """
        del kwargs
        if not enable_trace:
            return self._generate_eager(prompt_token_ids, max_new_tokens, next_input=next_input)
        if stop_on_eos is None:
            stop_on_eos = next_input is None

        prompt = [int(t) for t in prompt_token_ids]
        if len(prompt) + max_new_tokens > self.cache_context:
            raise ValueError(
                f"prompt {len(prompt)} + {max_new_tokens} new tokens exceeds the allocated cache context "
                f"{self.cache_context}; build the generator with a larger cache_context"
            )
        # Reset FIRST: `_ensure_decode_trace` refuses to capture over live state, and a caller who
        # asked for a reset is entitled to have the stale state cleared rather than be refused for
        # it. (`reset()` is also what clears `model.state_is_live`.)
        if reset:
            self.reset()
        self._ensure_decode_trace()
        if sampling_params is not None:
            self._apply_sampling_params(sampling_params)

        start = time.perf_counter()
        # --- prefill -------------------------------------------------------------------
        # The page-table row is allocated *after* trace capture, so it must not outlive the prefill:
        # see `_ensure_traces_replay_safe` for why a long-lived post-capture buffer is unsafe.
        page_row = self._prefill_page_row(0)
        device_logits = self.model.prefill_request_into_slot(
            prompt, page_table=page_row, slot=0, start_pos=0, return_logits="device"
        )
        first = self._first_token_after_prefill(device_logits)
        if page_row is not None:
            ttnn.deallocate(page_row)
        ttnn.synchronize_device(self.mesh_device)
        ttft = time.perf_counter() - start

        predictions = [int(first)]
        forced = int(next_input(0, int(first))) if next_input is not None else int(first)
        on_device = int(first)

        # Prefill compiles the programs its prompt length needs. If any of them is new, their kernel
        # binaries were allocated while the decode traces were live and the first replay would
        # overwrite them, so the traces are re-captured before that replay ever happens. See
        # `_ensure_traces_replay_safe`. Capture does not execute, so the prefill state survives it.
        self._ensure_traces_replay_safe()

        # Positions are handed to the device once, here, at the request boundary; from this point
        # `ttnn.plus_one` inside the trace advances them.
        self._write_positions(torch.tensor([len(prompt)] * self.max_batch_size, dtype=torch.int32))
        self._refresh_page_table_only(self.page_table)
        programs_before_decode = self.mesh_device.num_program_cache_entries()

        decode_start = time.perf_counter()
        steps = 0
        for step in range(1, max_new_tokens):
            if forced != on_device:
                self._write_tokens(torch.tensor([forced] * self.max_batch_size, dtype=torch.int32))
                on_device = forced
            self._decode_step_traced()
            if self.sampling_mode == "device":
                self._sample_traced()
                ttnn.synchronize_device(self.mesh_device)
                predicted = int(self._read_tokens()[0])
            else:
                ttnn.synchronize_device(self.mesh_device)
                predicted = int(torch.argmax(self.model.decode_logits_to_host(self._trace_logits)[0]).item())
                self._write_tokens(torch.tensor([predicted] * self.max_batch_size, dtype=torch.int32))
            on_device = predicted
            predictions.append(predicted)
            steps += 1
            forced = int(next_input(step, predicted)) if next_input is not None else predicted
            if stop_on_eos and predicted in self._eos_ids:
                break
        decode_elapsed = time.perf_counter() - decode_start
        if self.mesh_device.num_program_cache_entries() != programs_before_decode:
            logger.warning(
                "a program was compiled inside the traced decode loop; its kernel binaries may have been "
                "overwritten by a replay. Re-capturing the traces before the next request."
            )

        self.perf = {
            "prompt_len": len(prompt),
            "requested_tokens": max_new_tokens,
            "generated_tokens": len(predictions),
            "ttft_s": ttft,
            "decode_steps": steps,
            "decode_elapsed_s": decode_elapsed,
            "decode_t/s/u": (steps / decode_elapsed) if steps and decode_elapsed > 0 else 0.0,
            "decode_ms_per_token": (decode_elapsed / steps * 1e3) if steps else 0.0,
            "sampling_mode": self.sampling_mode,
            "teacher_forcing": next_input is not None,
            "counters": dict(self.counters),
        }
        logger.info(
            f"generate: prompt={len(prompt)} tokens={len(predictions)} TTFT={ttft * 1e3:.1f} ms "
            f"decode={self.perf['decode_t/s/u']:.2f} t/s/u ({self.perf['decode_ms_per_token']:.2f} ms/token) "
            f"mode={self.sampling_mode}"
        )
        return predictions

    def _first_token_after_prefill(self, device_logits):
        """Sample the prompt's last-position logits **on device**, straight into the token buffer.

        The readiness contract describes this step as a host argmax of the prefill logits. Doing it
        on device instead removes the only host argmax the greedy path would otherwise contain and
        leaves the first decode replay reading a token the device produced. Prefill sampling runs
        untraced (``enable_trace=False``): its logits tensor is freshly allocated per request and so
        cannot be the tensor identity the decode-side sampling trace was captured against.
        """
        if self.sampling_mode == "device":
            self.sampling.sample(logits=device_logits, tt_out_tok=self._trace_inputs[0], enable_trace=False)
            ttnn.deallocate(device_logits)
            ttnn.synchronize_device(self.mesh_device)
            return int(self._read_tokens()[0])
        host = self.model.decode_logits_to_host(device_logits, batch=1)
        ttnn.deallocate(device_logits)
        token = int(torch.argmax(host[0]).item())
        self._write_tokens(torch.tensor([token] * self.max_batch_size, dtype=torch.int32))
        return token

    def _prefill_page_row(self, user: int):
        return ttnn.from_torch(
            self.page_table[user : user + 1].contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _generate_eager(self, prompt_token_ids, max_new_tokens, *, next_input=None):
        """Untraced debug loop. Not used for any evidence in this stage."""
        self.reset()
        prompt = [int(t) for t in prompt_token_ids]
        page_row = self._prefill_page_row(0)
        logits = self.model.prefill_request_into_slot(prompt, page_table=page_row, slot=0, start_pos=0)
        if page_row is not None:
            ttnn.deallocate(page_row)
        predicted = int(torch.argmax(logits[0, -1]).item())
        predictions = [predicted]
        nxt = int(next_input(0, predicted)) if next_input is not None else predicted
        for step in range(1, max_new_tokens):
            out = self.decode_forward(
                torch.tensor([nxt] * self.max_batch_size),
                torch.tensor([len(prompt) + step - 1] * self.max_batch_size),
                enable_trace=False,
            )
            predicted = int(torch.argmax(out[0]).item())
            predictions.append(predicted)
            nxt = int(next_input(step, predicted)) if next_input is not None else predicted
        return predictions

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """Wipe per-prompt state. Device buffers, traces and weights all survive."""
        self.model.reset_state()
        if self._trace_inputs is not None:
            self._refresh_inputs(
                self._trace_inputs,
                torch.zeros(self.max_batch_size, dtype=torch.int32),
                torch.zeros(self.max_batch_size, dtype=torch.int32),
                self.page_table,
                force=True,
            )
        self._prev_page_table = self.page_table.clone()
        for key in self.counters:
            self.counters[key] = 0
        ttnn.synchronize_device(self.mesh_device)

    def teardown(self) -> None:
        try:
            self._release_traces()
        except Exception as exc:  # noqa: BLE001 - teardown must not mask a test failure
            logger.warning(f"failed to release the decode/sampling traces: {exc}")


# ---------------------------------------------------------------------------- factory
def build_generator(model_dir=None, mesh_device=None, **kwargs) -> OrnithGenerator:
    """``models.common.readiness_check.contract`` factory.

    ``model_dir`` is the autoport directory (it only names the model; the weights come from the HF
    snapshot). Accepted keyword arguments:

    ``max_batch_size``        decode batch and number of serving slots (default 1)
    ``cache_context``         tokens of paged KV cache to reserve per user (default 262144)
    ``max_context``           the model's advertised context (default: the HF config's)
    ``override_num_layers``   build only the first N layers - debugging
    ``layer_indices``         build exactly these HF layer indices - the reduced profiling variant
    ``sampling_mode``         ``"device"`` (default) or ``"host"`` compatibility mode
    ``policy``                precision policy name; default is the decoder stage's ``optimized``
    ``lm_head_dtype``         override the LM head weight dtype
    """
    if mesh_device is None:
        raise ValueError("build_generator needs an open mesh_device")

    max_batch_size = int(kwargs.pop("max_batch_size", 1))
    cache_context = int(kwargs.pop("cache_context", DEFAULT_CACHE_CONTEXT))
    sampling_mode = kwargs.pop("sampling_mode", "device")
    max_top_k = int(kwargs.pop("max_top_k", 32))
    pad_logits_to_power_of_2 = bool(kwargs.pop("pad_logits_to_power_of_2", False))
    topk_num_groups = int(kwargs.pop("topk_num_groups", DEFAULT_TOPK_GROUPS))
    tokenizer = kwargs.pop("tokenizer", None)
    snapshot = kwargs.pop("snapshot_path", None)

    if max_batch_size > MAX_SAMPLING_BATCH:
        raise ValueError(f"max_batch_size {max_batch_size} exceeds the sampler's bound {MAX_SAMPLING_BATCH}")

    path = resolve_model_path(snapshot)
    hf_config = load_text_config(path)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)

    started = time.perf_counter()
    model = OrnithModel.from_pretrained(path, mesh_device=mesh_device, hf_config=hf_config, **kwargs)
    logger.info(f"model weights loaded in {time.perf_counter() - started:.1f} s")

    generator = OrnithGenerator(
        model,
        tokenizer=tokenizer,
        max_batch_size=max_batch_size,
        cache_context=cache_context,
        sampling_mode=sampling_mode,
        max_top_k=max_top_k,
        pad_logits_to_power_of_2=pad_logits_to_power_of_2,
        topk_num_groups=topk_num_groups,
    )
    if model_dir is not None:
        generator.model_dir = Path(model_dir)
    return generator
