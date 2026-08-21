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
        topk_num_groups: int | str = "auto",
        pipelined_readback: bool = True,
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
        #: Overlap the caller's token readback with the next decode replay (optimized-full-model).
        #:
        #: The steady-state free-running loop has no host->device dependency at all: the sampled
        #: token reaches the next step through ``tt_out_tok`` on device and the position advances
        #: with ``ttnn.plus_one`` inside the trace, so step ``N+1`` can be *enqueued* before the
        #: host has looked at token ``N``. With this off the loop calls
        #: ``ttnn.synchronize_device`` every step and the device idles for the whole readback plus
        #: the Python between replays; with it on the readback is issued non-blocking, an event is
        #: recorded behind it, the next replay is enqueued, and only then is the event waited on -
        #: so the wait overlaps device work that is already running.
        #:
        #: Teacher forcing keeps the serial loop by construction: ``next_input`` needs token ``N``
        #: on the host before step ``N+1``'s token input can be decided.
        self.pipelined_readback = bool(pipelined_readback)
        #: How many times a serving step had to capture a sampling trace for a parameter shape it
        #: had not seen (`ensure_sampling_trace`). Steady serving traffic drives this to 0.
        self.sampling_trace_captures = 0
        #: Scratch ``ttnn.sampling`` output for serving prefill, allocated **here** - before any
        #: trace is captured - because a long-lived buffer allocated while a trace is live can share
        #: addresses the trace writes (see :meth:`_ensure_traces_replay_safe`). Prefill sampling must
        #: not write the decode token buffer: that buffer carries the *other* slots' live tokens and
        #: ``ttnn.sampling`` writes all of its rows, not just the one the prompt occupies.
        self._prefill_tokens = ttnn.to_device(
            model.prepare_decode_inputs_host(
                torch.zeros(self.max_batch_size, dtype=torch.int32),
                torch.zeros(self.max_batch_size, dtype=torch.int32),
                None,
            )[0],
            device=self.mesh_device,
        )

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

        A table is *foreign* when it can address blocks the attached cache does not have, and that is
        decided by who allocated the cache, not by whether this one call happened to repeat the
        ``kv_cache`` handle. So:

        * the generator allocated its own cache (``owns_cache``) and the call names no cache - the
          table is foreign and the internal one is used, with one warning. This is the shared
          readiness runner's case: ``run_prefill_check`` passes a deliberately dummy
          ``arange(1024)`` table with ``kv_cache=None`` and its docstring says the generator should
          handle it, and substituting is what makes that work;
        * the cache the generator is driving is the **caller's** (``owns_cache`` false, i.e. a
          ``kv_cache=`` was passed to the constructor) - the caller's table addresses the caller's
          own blocks and is honoured, whether or not this particular call repeats the handle.

        Substituting in that second case is silently wrong rather than conservative, and it was
        measured: ``doc/vllm_integration/prefill_determinism_bisect.json`` shows a serving-shaped
        prefill driven that way writing every logical block of the prompt to physical block 0 (the
        substituted table was all zeros), which makes repeated identical prefills differ by up to
        4.2 logit units at PCC 0.886-0.963, while the same call with the caller's table is
        bit-identical. Honour it on one of the two entry points only and the request prefills into
        one set of blocks and decodes out of another, so both resolve it here and cannot disagree.
        """
        if page_table is None:
            return self.page_table
        if kv_cache is None and self.owns_cache:
            if not self._warned_page_table_substitution:
                self._warned_page_table_substitution = True
                logger.warning(
                    f"{where} was given a page_table but no kv_cache, and this generator allocated its own "
                    "cache, so its own page table is used instead: a foreign table can address blocks the "
                    "internal cache does not have. Pass kv_cache as well, or build the generator on the "
                    "caller-owned cache, to drive caller-owned state."
                )
            return self.page_table
        return torch.as_tensor(page_table).to(torch.int32)

    # ------------------------------------------------------------------ trace management
    def _host_decode_inputs(self, tokens, positions, page_table, *, page_table_only=False):
        return self.model.prepare_decode_inputs_host(tokens, positions, page_table, page_table_only=page_table_only)

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
            page_table_only=True,
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
        self.counters["token_readbacks"] += 1
        return whole.reshape(-1)[: self.max_batch_size].to(torch.int64)

    def _read_tokens_async(self):
        """Enqueue the token readback **behind** the replay that produced it, without waiting.

        The mechanism is :meth:`read_output_async`'s: ``cpu(blocking=False)`` puts the device->host copy
        on the same command queue as the model and sampling replays, so it observes exactly this step's
        sampled token (the queue is in-order and the *next* step's replay is enqueued after it), and the
        recorded event is what the host waits on later. This wrapper adds the ``token_readbacks``
        counter and is what the pipelined ``generate`` loop calls.
        """
        host, event = self.read_output_async()
        self.counters["token_readbacks"] += 1
        return host, event

    def _finish_read(self, pending) -> torch.Tensor:
        host, event = pending
        ttnn.event_synchronize(event)
        self.counters["read_waits"] += 1
        whole = ttnn.to_torch(host, mesh_composer=ttnn.concat_mesh_to_tensor_composer(self.mesh_device, dim=0))
        return whole.reshape(-1)[: self.max_batch_size].to(torch.int64)

    def _decode_step_traced(self) -> None:
        ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=False)
        self.counters["decode_calls"] += 1

    def _sample_traced(self):
        # `skip_precompile=True` matters only on the capture that `sample()` may still do lazily:
        # precompiling would execute a whole sampling graph over `_trace_logits`, which lives in the
        # trace region while a captured trace exists - the hazard `_ensure_decode_trace` documents.
        # Serving pre-captures through `ensure_sampling_trace()`; this is the belt to that brace.
        self.sampling.sample(
            logits=self._trace_logits,
            tt_out_tok=self._trace_inputs[0],
            enable_trace=True,
            skip_precompile=True,
        )

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
        the DeltaNet state carries over instead of being zeroed. This low-level batched surface accepts
        one continuing request at a time; serving supplies its explicit persistent slot through
        :meth:`OrnithModel.prefill_request_into_slot`.

        Cache ownership is explicit: pass ``kv_cache`` (and the matching ``page_table``) to drive the
        generator's model against caller-owned state, or leave both ``None`` to use the cache and
        page table this generator allocated. A caller that hands a page table without a cache to a
        generator that allocated its **own** cache gets the internal page table, because such a table
        can address blocks the internal cache does not have; a generator built on a caller-owned cache
        honours the caller's table either way (:meth:`_resolve_page_table`).

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

        # The free-running device-sampling loop has no host->device dependency: the sampled token
        # reaches the next replay through `tt_out_tok` and the position advances with
        # `ttnn.plus_one`, both inside the trace. So step N+1 is enqueued before token N is looked
        # at, and the host's wait overlaps device work instead of idling it. Teacher forcing and
        # host sampling keep the serial loop, because both decide step N+1's token input on the
        # host from token N.
        pipelined = self.pipelined_readback and self.sampling_mode == "device" and next_input is None

        decode_start = time.perf_counter()
        steps = 0
        if pipelined:
            pending = None
            for _ in range(1, max_new_tokens):
                self._decode_step_traced()
                self._sample_traced()
                in_flight = self._read_tokens_async()
                if pending is not None:
                    predicted = int(self._finish_read(pending)[0])
                    on_device = predicted
                    predictions.append(predicted)
                    steps += 1
                    if stop_on_eos and predicted in self._eos_ids:
                        # `in_flight` belongs to a step that was enqueued speculatively; its token is
                        # discarded and the synchronize below retires it before the next request.
                        pending = None
                        break
                pending = in_flight
            if pending is not None:
                predicted = int(self._finish_read(pending)[0])
                on_device = predicted
                predictions.append(predicted)
                steps += 1
            # Retire anything still in flight so the measured window covers all the work it issued.
            ttnn.synchronize_device(self.mesh_device)
        else:
            for step in range(1, max_new_tokens):
                if forced != on_device:
                    self._write_tokens(torch.tensor([forced] * self.max_batch_size, dtype=torch.int32))
                    on_device = forced
                self._decode_step_traced()
                if self.sampling_mode == "device":
                    self._sample_traced()
                    ttnn.synchronize_device(self.mesh_device)
                    self.counters["decode_syncs"] += 1
                    predicted = int(self._read_tokens()[0])
                else:
                    ttnn.synchronize_device(self.mesh_device)
                    self.counters["decode_syncs"] += 1
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
            "pipelined_readback": bool(pipelined),
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
            # This runs **untraced** on purpose, and the alternative was tried and reverted. Copying
            # the prefill logits into `self._trace_logits` and replaying the captured sampling trace
            # would turn 3.350 ms of a ~140 ms median TTFT into ~1.1 ms, but `_trace_logits` is allocated
            # inside the trace region, and writing to it from outside a replay wedged the mesh:
            # `doc/optimized_full_model/triage/` is the tt-triage capture (a stuck
            # `ReshapeViewDeviceOperation` on all four devices plus kernel `.text` mismatches), which
            # is the same trace-region hazard `SamplingGenerator.capture_trace(skip_precompile=True)`
            # already exists to avoid. See `doc/optimized_full_model/README.md` §Rejected.
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

    def warmup(self, prompt_lengths, *, max_new_tokens: int = 2) -> dict:
        """Pre-compile the prefill programs for the given prompt lengths, once, at startup.

        A prefill compiles programs keyed by its *logical* prompt length - the ``ttnn.slice``
        offsets, the MoE valid-token count and the conv1d length are all compile-time constants - and
        those kernel binaries are allocated while the decode traces are live, so
        :meth:`_ensure_traces_replay_safe` has to re-capture before the first replay can run
        (``doc/full_model/README.md`` §5.1). That re-capture is a few hundred milliseconds and it
        lands in *neither* reported metric, because ``generate`` stops the TTFT clock before it and
        starts the decode clock after it: a cold-length request is slower than any published number.

        This is the fix the full-model stage handed forward. Drive it once with the lengths a
        deployment expects (or the bucket boundaries it rounds to) and every later request at those
        lengths is warm, with ``trace_recaptures`` provably unchanged. Returns the per-length wall
        clock and the recapture count so the cost is measured rather than assumed.
        """
        lengths = [int(v) for v in prompt_lengths]
        report: dict[str, Any] = {"lengths": lengths, "seconds": {}, "recaptures_before": self.trace_recaptures}
        for length in lengths:
            if length < 1 or length > self.cache_context:
                raise ValueError(f"warmup length {length} is outside [1, {self.cache_context}]")
            started = time.perf_counter()
            self.generate(
                prompt_token_ids=[1] * length,
                max_new_tokens=max(1, int(max_new_tokens)),
                enable_trace=True,
                stop_on_eos=False,
            )
            report["seconds"][length] = time.perf_counter() - started
        self.reset()
        report["recaptures_after"] = self.trace_recaptures
        report["recaptures"] = report["recaptures_after"] - report["recaptures_before"]
        logger.info(
            f"warmup: {len(lengths)} prompt length(s) compiled in "
            f"{sum(report['seconds'].values()):.1f} s, {report['recaptures']} trace re-capture(s)"
        )
        return report

    # ------------------------------------------------------------------ serving (vLLM) API
    #
    # The primitives ``tt/generator_vllm.py`` drives. They live here, beside the persistent trace
    # inputs and the sampling traces, because that is the state they manipulate; the adapter owns the
    # vLLM-facing translation and nothing else. None of this is used by the readiness runners or by
    # :meth:`generate`, and the measured serving decode step is the *same* split-sampling path they
    # use: one model-trace replay, one sampling-trace replay, ``tt_out_tok`` feeding the next replay
    # on device with no host argmax, no logits readback and no Python token feedback.

    def ensure_serving_traces(self) -> None:
        """Capture the decode traces. Call from vLLM warmup, before any prompt has been written."""
        self._ensure_decode_trace()

    def ensure_replay_safe(self) -> None:
        """Re-capture the traces if anything has been compiled since they were captured."""
        self._ensure_traces_replay_safe()

    def invalidate_sampling_params_cache(self) -> None:
        """Forget what :meth:`_apply_sampling_params` last pushed to the device.

        A serving caller drives ``SamplingGenerator.apply_decode_state`` itself (per-row params,
        penalties, seeds), so this generator's own single-parameter-set cache no longer describes the
        device and must not be allowed to skip a later push.
        """
        self._sampling_params_key = None

    def ensure_sampling_trace(self) -> bool:
        """Capture the sampling trace for the **current** sampling parameters if it is not captured.

        ``SamplingGenerator`` keys its traces on (penalties, log-probs, force-argmax) and releases
        *all* of them whenever force-argmax flips - which a serving batch does whenever it stops or
        starts being all-greedy. Capturing here, before this step's model replay is enqueued, keeps
        two things true that ``sample()``'s own lazy capture would not:

        * capture never happens *after* a non-blocking replay has already been enqueued;
        * capture runs with ``skip_precompile=True``, so it records without executing a full sampling
          graph over the live trace-region logits buffer - the hazard :meth:`_ensure_decode_trace`
          documents, and the one that hung the mesh inside ``all_gather_async`` on the 40-layer model.

        Every program it records is compiled by the serving warm-up's eager phase, so the capture
        itself compiles nothing. Returns True when a trace was captured.
        """
        if self.sampling_mode != "device" or self._trace_logits is None:
            return False
        sampling = self.sampling
        if sampling.seed_manager.has_active_request_seed():
            # An explicit request seed rewrites a persistent seed tensor every token, so
            # ``SamplingGenerator.sample`` deliberately runs untraced. There is nothing to capture.
            return False
        _, slot = sampling._trace_slot(
            sampling._penalties_active,
            getattr(sampling, "_log_probs_active", False),
            sampling.tt_sampling.force_argmax_sampling,
        )
        if slot["id"] is not None:
            return False
        ttnn.synchronize_device(self.mesh_device)
        sampling.capture_trace(logits=self._trace_logits, tt_out_tok=self._trace_inputs[0], skip_precompile=True)
        self.sampling_trace_captures += 1
        return True

    def device_decode_state(self):
        """The tokens and positions the persistent decode trace inputs hold **right now**.

        Read from one shard, because both buffers are replicated across the mesh. This is the
        authority a serving refresh merges against: under async scheduling the host's view of a
        continuing row lags the device by one token, and the device's copy is the one the last replay
        actually produced.
        """
        tokens = ttnn.to_torch(ttnn.get_device_tensors(self._trace_inputs[0])[0]).reshape(-1)
        positions = ttnn.to_torch(ttnn.get_device_tensors(self._trace_inputs[1])[0]).reshape(-1)
        batch = self.max_batch_size
        return tokens[:batch].to(torch.int64), positions[:batch].to(torch.int64)

    def stage_serving_decode_inputs(
        self, tokens, positions, page_table, *, full_refresh: bool, device_token_rows=None
    ) -> dict:
        """Refresh the decode trace inputs for one serving step, and only as much as changed.

        ``full_refresh`` is the caller's statement that host token/position state is authoritative
        again for at least one row - a batch-layout change, a slot remap, a freshly prefilled slot, a
        switch to or from host sampling. It is *not* per-token: in the steady state of a traced
        device-sampling decode this method copies **nothing at all**, because the token arrives
        through ``tt_out_tok`` and the positions are advanced by ``ttnn.plus_one`` inside the trace.

        On a full refresh the values written are a merge, not the host's view: for every row the
        caller marks in ``device_token_rows`` whose device position is continuous with the host's
        (equal, or one ahead - the async-scheduling lag), the device's token *and* position win.
        Staging the lagging host pair for such a row would re-run a position that already has a
        token, which shows up as a doubled subword rather than as an error.

        ``page_table`` is refreshed whenever its contents changed, on both paths: new KV blocks are
        allocated as a request grows, and that is scheduler state the device cannot derive.
        """
        batch = self.max_batch_size
        tokens = torch.as_tensor(tokens).reshape(-1)[:batch].to(torch.int64)
        positions = torch.as_tensor(positions).reshape(-1)[:batch].to(torch.int64)
        table = torch.as_tensor(page_table).to(torch.int32)
        if table.dim() == 1:
            table = table.unsqueeze(0)
        before = self.counters["page_table_refreshes"]
        if not full_refresh:
            self._refresh_page_table_only(table)
            return {
                "tokens": False,
                "positions": False,
                "page_table": self.counters["page_table_refreshes"] != before,
                "device_rows": [],
            }
        dev_tokens, dev_positions = self.device_decode_state()
        if device_token_rows is None:
            trust = torch.zeros(batch, dtype=torch.bool)
        else:
            trust = torch.as_tensor(device_token_rows).reshape(-1)[:batch].to(torch.bool)
        continuous = (dev_positions == positions) | (dev_positions == positions + 1)
        use_device = trust & continuous & (positions >= 0)
        merged_tokens = torch.where(use_device, dev_tokens, tokens)
        merged_positions = torch.where(use_device, dev_positions, positions)
        self._refresh_inputs(self._trace_inputs, merged_tokens, merged_positions, table)
        return {
            "tokens": True,
            "positions": True,
            "page_table": self.counters["page_table_refreshes"] != before,
            "device_rows": use_device.nonzero().reshape(-1).tolist(),
        }

    def submit_serving_decode(self, *, sample_on_device: bool):
        """Replay the decode trace - and the sampling trace - without waiting, and return the tensor
        the caller should read.

        With ``sample_on_device`` that is the persistent decode **token** buffer, which the sampling
        trace has just written and which the next replay will read as its input; without it, the
        model's vocab-sharded logits, for the plugin's host sampler (log-probs, ``min_p``, structured
        output and the other host-only parameters). Both replays are ``blocking=False``: the caller
        reads behind them on the same command queue.

        The replay-safety check is here rather than left to the caller because forgetting it is
        silent: a prefill compiles programs for its own prompt length, their kernel binaries land on
        addresses the decode trace writes, and the *next* replay overwrites them - after which every
        request at that length emits gibberish, permanently (see
        :meth:`_ensure_traces_replay_safe`). It is one integer comparison on the steady path.
        """
        self._ensure_traces_replay_safe()
        self._decode_step_traced()
        # A replay does not run the Python body that would mark the state live, so mark it here:
        # after this step the paged cache and the DeltaNet rows hold a request.
        self.model.state_is_live = True
        if not sample_on_device:
            return self._trace_logits
        self._sample_traced()
        return self._trace_inputs[0]

    def read_tokens(self) -> torch.Tensor:
        """Blocking readback of the sampled tokens of the last replay."""
        return self._read_tokens()

    def read_tokens_async(self):
        """Enqueue the token readback behind the replay that produced it, without waiting.

        The token-buffer shorthand for :meth:`read_output_async`, plus the ``token_readbacks`` counter
        the standalone ``generate`` loop is measured by. One implementation, two entry points: the only
        difference between them is that counter.
        """
        return self._read_tokens_async()

    def read_output_async(self, tensor=None):
        """Enqueue a device->host copy of one decode step's output, behind the replays, without waiting.

        The serving adapter's async split needs this for *either* output tensor: the persistent decode
        token buffer on a device-sampled step, and the vocab-sharded logits on a host-sampled one. The
        two calls are the whole primitive - ``cpu(blocking=False)`` puts the copy on the same in-order
        command queue as the replays that produced it, so it observes exactly this step's result, and
        the recorded event is what the host waits on once the next step is already running. It lives
        here rather than in the adapter so no caller has to reason about queues or events, and
        :meth:`read_tokens_async` is the token-buffer-only shorthand the standalone ``generate`` loop
        uses (it also counts the readback, which the serving path counts for itself).

        ``tensor=None`` reads the token buffer, so the two behave identically on the sampled path.
        """
        target = self._trace_inputs[0] if tensor is None else tensor
        host = target.cpu(blocking=False)
        return host, ttnn.record_event(self.mesh_device, 0)

    def finish_token_read(self, pending) -> torch.Tensor:
        """Wait for a :meth:`read_tokens_async` and compose its result."""
        return self._finish_read(pending)

    def tokens_from(self, tensor) -> torch.Tensor:
        """Host token ids ``[batch]`` from a device **or** host copy of the decode token buffer."""
        whole = ttnn.to_torch(tensor, mesh_composer=ttnn.concat_mesh_to_tensor_composer(self.mesh_device, dim=0))
        return whole.reshape(-1)[: self.max_batch_size].to(torch.int64)

    def logits_from(self, tensor) -> torch.Tensor:
        """Host logits ``[batch, 1, vocab]`` from a device **or** host copy of the decode logits.

        This is the host-sampling compatibility boundary, and it is never on the measured path.
        """
        return self.model.decode_logits_to_host(tensor).unsqueeze(1)

    def remap_serving_slots(self, remap) -> int:
        """Apply a vLLM batch-condense permutation to the model's per-slot recurrent state.

        Returns the number of layers whose state moved. The paged KV half needs nothing: it follows
        the page table, which vLLM permutes itself.
        """
        return self.model.remap_state_slots(remap)

    def prefill_requests_into_slots(
        self,
        tokens,
        prompt_lens,
        slots,
        *,
        page_table,
        kv_cache=None,
        start_pos=None,
        sample_on_device: bool = False,
        before_sample=None,
        ensure_traces: bool = True,
    ):
        """Prefill one serving step's prompts, each into the decode slot vLLM assigned it.

        ``tokens`` is ``[N, P]`` in *request* order and so are ``prompt_lens``, ``start_pos`` and the
        rows of ``page_table``; ``slots[u]`` is the fixed device state slot request ``u`` will decode
        in, which is **not** ``u`` whenever an off-batch request still owns that row. Row ``u`` is
        prefilled from ``tokens[u, start_pos[u]:prompt_lens[u]]``, which is the chunk convention the
        TT plugin builds its inputs with.

        With ``sample_on_device`` the prompt's last-position logits are sampled by the on-device
        sampler and only the token id comes back, one int per request: no logits are composed on
        host and no host argmax runs. The sampler writes a **scratch** token buffer rather than the
        decode token buffer, because that buffer carries the other slots' live tokens and
        ``ttnn.sampling`` writes all 32 of its rows. ``before_sample(u, slot)`` is where the caller
        pushes that request's sampling parameters and seed.

        Without it, host logits ``[N, 1, vocab]`` come back for the plugin's host sampler.
        """
        if ensure_traces:
            # Capture before the prompt is written, never after: capture warm-compiles a real decode
            # step and then wipes the state it touched. `ensure_traces=False` is the serving warm-up's
            # compile phase, which deliberately runs before anything is captured.
            self._ensure_decode_trace()
            self._ensure_traces_replay_safe()
        if kv_cache is not None:
            self.model.attach_kv_cache(kv_cache)
        tokens = torch.as_tensor(tokens)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        count = int(tokens.shape[0])
        lens = [int(tokens.shape[1])] * count if prompt_lens is None else [int(v) for v in prompt_lens]
        rows = list(range(count)) if slots is None else [int(s) for s in slots]
        if len(lens) != count or len(rows) != count:
            raise ValueError(f"{count} prompt row(s) but {len(lens)} length(s) and {len(rows)} slot(s)")
        if start_pos is None:
            starts = [0] * count
        elif isinstance(start_pos, int):
            starts = [int(start_pos)] * count
        else:
            starts = [int(v) for v in torch.as_tensor(start_pos).reshape(-1)[:count]]
        table = self._resolve_page_table(page_table, kv_cache, "prefill_requests_into_slots")
        table = torch.as_tensor(table).to(torch.int32)
        if table.dim() == 1:
            table = table.unsqueeze(0)
        if table.shape[0] < count:
            raise ValueError(f"page table has {int(table.shape[0])} row(s) for {count} prompt(s)")

        out = []
        for user in range(count):
            slot = rows[user]
            if not 0 <= slot < self.max_batch_size:
                raise ValueError(f"state slot {slot} is outside [0, {self.max_batch_size})")
            end = lens[user]
            start = starts[user]
            if end <= start:
                raise ValueError(f"request {user} has an empty chunk [{start}, {end})")
            host_page_row = table[user : user + 1]
            page_row = self._page_row_tensor(host_page_row)
            prefill_inputs = self.model.prepare_prefill_chunk_inputs(
                page_table=page_row,
                host_page_table=host_page_row,
                start_pos=start,
                logical_len=end - start,
            )
            logits = self.model.prefill_request_into_slot(
                tokens[user : user + 1, start:end],
                page_table=prefill_inputs,
                slot=slot,
                start_pos=start,
                return_logits="device" if sample_on_device else True,
                continue_from_state=start > 0,
            )
            if page_row is not None:
                ttnn.deallocate(page_row)
            if not sample_on_device:
                out.append(logits)
                continue
            if before_sample is not None:
                before_sample(user, slot)
            self.sampling.sample(logits=logits, tt_out_tok=self._prefill_tokens, enable_trace=False)
            ttnn.deallocate(logits)
            ttnn.synchronize_device(self.mesh_device)
            sampled = ttnn.to_torch(ttnn.get_device_tensors(self._prefill_tokens)[0]).reshape(-1)[0]
            out.append(int(sampled))
        if sample_on_device:
            return torch.tensor(out, dtype=torch.int32)
        return torch.cat(out, dim=0)

    def _page_row_tensor(self, host_row):
        """One request's ``[1, blocks]`` int32 ROW_MAJOR page table, on device.

        Deallocated by the caller as soon as the prefill that needs it returns: a buffer allocated
        while the decode traces are live must not outlive the call that made it (see
        :meth:`_ensure_traces_replay_safe`).
        """
        return ttnn.from_torch(
            torch.as_tensor(host_row).to(torch.int32).contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

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
    ``policy``                precision policy. **Default: the selected precision config artifact**,
                              ``doc/datatype_sweep/selected_precision_config.json``, loaded by
                              ``tt/precision_config.py``. Also accepts a registered policy name
                              (``"optimized"`` is the pre-sweep decoder-stage policy,
                              ``"fused-parity"`` the bfloat16 floor), a path to another JSON config,
                              a dict in that schema, or a ``PrecisionPolicy``. The environment
                              variable ``ORNITH_PRECISION_POLICY`` overrides the default without
                              touching a call site
    ``lm_head_dtype``         override the LM head weight dtype (default: the policy's)
    ``lm_head_program``       terminal matmul spelling: ``"mcast1d"`` (default), ``"interleaved"`` or
                              ``"dram_sharded"``. ``"dram_sharded"`` requires ``lm_head_cores`` to
                              divide ``dim / 32`` and raises otherwise
    ``lm_head_cores``         compute cores for the tuned spellings (default 110, the whole grid)
    ``lm_head_in0_block_w``   override the terminal matmul's K block (default: the largest legal)
    ``lm_head_fidelity``      ``"lofi"`` / ``"hifi2"`` / ``"hifi4"`` for the terminal matmul only
    ``terminal_norm_cores``   cores the terminal RMSNorm width-shards ``dim`` over (default 8)
    ``vocab_align_tiles``     per-device vocabulary tile alignment (default 32; see the sampler)
    ``pipelined_readback``    overlap the caller token readback with the next replay (default True)
    """
    if mesh_device is None:
        raise ValueError("build_generator needs an open mesh_device")

    max_batch_size = int(kwargs.pop("max_batch_size", 1))
    cache_context = int(kwargs.pop("cache_context", DEFAULT_CACHE_CONTEXT))
    sampling_mode = kwargs.pop("sampling_mode", "device")
    max_top_k = int(kwargs.pop("max_top_k", 32))
    pad_logits_to_power_of_2 = bool(kwargs.pop("pad_logits_to_power_of_2", False))
    topk_num_groups = kwargs.pop("topk_num_groups", "auto")
    topk_num_groups = topk_num_groups if topk_num_groups == "auto" else int(topk_num_groups)
    pipelined_readback = bool(kwargs.pop("pipelined_readback", True))
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
        pipelined_readback=pipelined_readback,
    )
    if model_dir is not None:
        generator.model_dir = Path(model_dir)
    return generator
