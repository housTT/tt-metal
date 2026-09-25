# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Lossless k=1 speculative decoding with the checkpoint's MTP head (eager).

The target model is built with ``max_batch=2``; the two physical rows carry two
consecutive positions of one sequence.  Each step verifies the pending draft
and emits one or two tokens:

* rows = ``[b at P, c at P+1]`` where ``b`` is the last accepted token and ``c``
  the MTP draft for the token after it;
* the target's greedy token at row 0 is ``b'``; if ``b' == c`` the draft is
  accepted, row 1 yields the bonus token ``d`` and the new position is
  ``P+2``; otherwise the step emits ``b'`` and moves to ``P+1``;
* per layer, the GDN recurrence and the PLE conv run sequentially over the
  two rows (see ``FusedDecoder._gdn_decode_pair``); after the step row 0 holds
  the state after the first token and row 1 after both, and
  ``commit_speculative_pair(accept)`` makes row 0 the sequence's state;
* the QSA KV / indexer caches are position-addressed and overwrite-safe, the
  PLE n-gram history is snapshotted and rolled back on rejection;
* the MTP head (also two rows) then runs on the accepted positions and drafts
  the next token.

Greedy output equals the plain greedy output of the same kernels (row 0 is
bitwise independent of row 1).  ``force_reject=True`` runs the same graph but
never accepts, which is the reference for the lossless test.
"""

from __future__ import annotations

import os
import time

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt import functional_decoder as _functional_decoder
from models.autoports.qwen_qwen3_8_flash_next.tt.model import HC_COUNT, Qwen38FullModel, _copy_host_to_device
from models.autoports.qwen_qwen3_8_flash_next.tt.model_config import LINEAR_ATTENTION
from models.autoports.qwen_qwen3_8_flash_next.tt.mtp import Qwen38MTPDraftHead
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import RESIDUAL_SHARD_WIDTH


def _rows(residual, first: int, count: int = 1):
    start = HC_COUNT * first
    return ttnn.slice(residual, [0, 0, start, 0], [1, 1, start + HC_COUNT * count, RESIDUAL_SHARD_WIDTH])


class Qwen38SpeculativeDecoder:
    """Drive a ``max_batch=2`` target and its MTP head as one lossless k=1 decoder."""

    def __init__(
        self, model: Qwen38FullModel, head: Qwen38MTPDraftHead, *, force_reject: bool = False, traced: bool = False
    ):
        if model.max_batch != 2 or head.model is not model:
            raise ValueError("speculative decoding needs a max_batch=2 target and its MTP head")
        self.model = model
        self.head = head
        self.force_reject = bool(force_reject)
        self.traced = bool(traced)
        # Trace ids and their persistent outputs/inputs (captured on first use).
        self.target_trace_id = None
        self.trace_residual = None
        self.trace_logits = None
        self.commit_trace_id = None
        self.mtp_trace_id = None
        self.mtp_hidden = None
        self.mtp_logits = None
        self.trace_capture_seconds = 0.0
        self.target_tokens_out = None
        self.mtp_tokens_out = None
        self.device_argmax = os.environ.get("QWEN38_SPEC_DEVICE_ARGMAX", "1") == "1"
        self.request_id = "spec-main"
        self.state = None
        self.position = 0
        self.pending: tuple[int, int] | None = None
        self.tokens: list[int] = []
        self.steps = 0
        self.accepted = 0
        self.step_seconds = 0.0
        self.draft_seconds = 0.0
        # Per-row cache-update position registers (see FusedDecoder._qsa_decode
        # in speculative_pair mode): row-0-only, row-1-only, both-at-row-0,
        # both-at-row-1.
        mesh = model.mesh_device
        self._pos_regs = tuple(
            ttnn.from_torch(
                torch.zeros(2, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            for _ in range(4)
        )

    def _phase(self, name: str, started: float) -> float:
        now = time.perf_counter()
        phases = self.__dict__.setdefault("phase_seconds", {})
        phases[name] = phases.get(name, 0.0) + (now - started)
        return now

    def _set_pair_positions(self, layers, p0: int, p1: int) -> None:
        values = ([p0, -1], [-1, p1], [p0, p0], [p1, p1])
        for register, value in zip(self._pos_regs, values):
            _copy_host_to_device(
                torch.tensor(value, dtype=torch.int32), register, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
        for layer in layers:
            layer.pair_update_positions = self._pos_regs[:2]
            layer.pair_dup_positions = self._pos_regs[2:]

    # -------------------------------------------------------------- helpers

    def _ple_layer(self):
        return next((layer for layer in self.model.layers if layer.shapes.has_ple), None)

    def _target_rows(self, tokens, positions, ple_embeddings):
        """One eager pass of the target over two rows; returns the fractured residual."""

        model, state = self.model, self.state
        model.copy_tokens(state, torch.tensor(tokens, dtype=torch.int64))
        pos = torch.tensor(positions, dtype=torch.int32)
        _copy_host_to_device(pos, state.current_pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        state.host_positions = pos.clone()
        model._apply_qsa_variant(model._select_qsa_variant(state))
        ple_layer = self._ple_layer()
        staged = None if ple_layer is None else ple_layer.ple_staging.upload_decode(ple_embeddings)
        self._set_pair_positions(model.layers, positions[0], positions[1])
        for layer in model.layers:
            layer.speculative_pair = True
        try:
            residual = model.embed_tokens(state.token_input)
            for layer in model.layers:
                kwargs = {"current_pos": state.current_pos}
                if layer.shapes.layer_type != LINEAR_ATTENTION:
                    kwargs.update(page_table=state.page_table, rot_mats=model.rot_mats)
                if layer.shapes.has_ple:
                    kwargs["ple_embeddings"] = staged
                output = layer.decode_forward_fractured(residual, **kwargs)
                _functional_decoder._free(residual, output)
                residual = output
        finally:
            for layer in model.layers:
                layer.speculative_pair = False
        return residual

    # ------------------------------------------------------------- tracing

    def _target_graph(self, staged):
        """The two-row target graph on the persistent state registers."""

        model, state = self.model, self.state
        residual = model.embed_tokens(state.token_input)
        for layer in model.layers:
            kwargs = {"current_pos": state.current_pos}
            if layer.shapes.layer_type != LINEAR_ATTENTION:
                kwargs.update(page_table=state.page_table, rot_mats=model.rot_mats)
            if layer.shapes.has_ple:
                kwargs["ple_embeddings"] = staged
            output = layer.decode_forward_fractured(residual, **kwargs)
            _functional_decoder._free(residual, output)
            residual = output
        logits = model.project_logits(residual)
        if not self.device_argmax:
            return residual, logits
        tokens = self._device_argmax(logits, "target_tokens_out")
        ttnn.deallocate(logits)
        return residual, tokens

    def _device_argmax(self, logits, register_name: str):
        """Per-rank (max, argmax) of the vocab-sharded logits, gathered to ``[rows, ranks]``.

        The shared sampler's gather+argmax hangs inside this trace, and reading
        back the full two-row logits costs ~7.5 ms, so each rank reduces its
        62,080-wide shard and two tiny gathers carry the candidates; the host
        picks the winning rank (see :meth:`_read_tokens`).
        """

        del register_name
        layer = self.model.layers[0]
        local_max = ttnn.max(logits, dim=-1, keepdim=True)
        row_major = ttnn.untilize(logits, use_multicore=True)
        local_idx = ttnn.argmax(row_major, dim=-1, keepdim=True)
        ttnn.deallocate(row_major)
        idx_i32 = ttnn.typecast(local_idx, ttnn.int32)
        ttnn.deallocate(local_idx)
        idx_tiled = ttnn.to_layout(idx_i32, ttnn.TILE_LAYOUT)
        ttnn.deallocate(idx_i32)
        gather = dict(
            dim=3,
            cluster_axis=layer.collective_axis,
            num_links=layer.collective_num_links,
            topology=layer.collective_topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gathered_max = ttnn.all_gather(local_max, **gather)
        gathered_idx = ttnn.all_gather(idx_tiled, **gather)
        ttnn.deallocate(local_max)
        ttnn.deallocate(idx_tiled)
        ttnn.mark_corruptible(gathered_max)
        ttnn.mark_corruptible(gathered_idx)
        return (gathered_max, gathered_idx)

    def _read_tokens(self, candidates) -> list[int]:
        gathered_max, gathered_idx = candidates
        shard_width = int(self.model.lm_head_split_sizes[0]) if getattr(self.model, "lm_head_split_sizes", None) else 62080
        maxes = ttnn.to_torch(ttnn.get_device_tensors(gathered_max)[0]).float().reshape(-1, 4)
        idxs = ttnn.to_torch(ttnn.get_device_tensors(gathered_idx)[0]).to(torch.int64).reshape(-1, 4)
        out = []
        for row in range(int(self.model.max_batch)):
            rank = int(maxes[row].argmax())
            out.append(int(idxs[row, rank]) + rank * shard_width)
        return out

    def _run_target_traced(self, staged):
        """Replay (or capture on first use) the two-row target step."""

        model = self.model
        mesh = model.mesh_device
        if self.target_trace_id is None:
            started = time.perf_counter()
            for layer in model.layers:
                layer.speculative_pair = True
            try:
                self.target_trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
                self.trace_residual, self.trace_logits = self._target_graph(staged)
                ttnn.end_trace_capture(mesh, self.target_trace_id, cq_id=0)
            finally:
                for layer in model.layers:
                    layer.speculative_pair = False
            ttnn.mark_corruptible(self.trace_residual)
            if not self.device_argmax:
                ttnn.mark_corruptible(self.trace_logits)
            self.trace_capture_seconds += time.perf_counter() - started
        ttnn.execute_trace(mesh, self.target_trace_id, cq_id=0, blocking=True)
        return self.trace_residual, self.trace_logits

    def _commit_traced(self) -> None:
        model = self.model
        mesh = model.mesh_device
        if self.commit_trace_id is None:
            started = time.perf_counter()
            self.commit_trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
            for layer in model.layers:
                layer.commit_speculative_pair(True)
            ttnn.end_trace_capture(mesh, self.commit_trace_id, cq_id=0)
            self.trace_capture_seconds += time.perf_counter() - started
        ttnn.execute_trace(mesh, self.commit_trace_id, cq_id=0, blocking=True)

    def _mtp_traced(self):
        """Replay (or capture) the two-row MTP draft on ``self.mtp_hidden``."""

        head = self.head
        mesh = self.model.mesh_device
        if self.mtp_trace_id is None:
            started = time.perf_counter()
            head.layer.speculative_pair = True
            try:
                self.mtp_trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
                logits = head.draft_logits(
                    self.mtp_hidden, head.token_register, current_pos=head.scratch_pos, page_table=self.state.page_table
                )
                if self.device_argmax:
                    self.mtp_logits = self._device_argmax(logits, "mtp_tokens_out")
                    ttnn.deallocate(logits)
                else:
                    self.mtp_logits = logits
                    ttnn.mark_corruptible(self.mtp_logits)
                ttnn.end_trace_capture(mesh, self.mtp_trace_id, cq_id=0)
            finally:
                head.layer.speculative_pair = False
            self.trace_capture_seconds += time.perf_counter() - started
        ttnn.execute_trace(mesh, self.mtp_trace_id, cq_id=0, blocking=True)
        return self.mtp_logits

    def _fill_mtp_hidden(self, residual, accept: bool) -> None:
        if self.mtp_hidden is None:
            self.mtp_hidden = ttnn.clone(residual, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if accept:
            ttnn.copy(residual, self.mtp_hidden)
        else:
            row0 = _rows(residual, 0)
            both = ttnn.concat([row0, row0], dim=2)
            ttnn.copy(both, self.mtp_hidden)
            ttnn.deallocate(both)
            ttnn.deallocate(row0)

    def release_traces(self) -> None:
        mesh = self.model.mesh_device
        for name in ("target_trace_id", "commit_trace_id", "mtp_trace_id"):
            trace_id = getattr(self, name)
            if trace_id is not None:
                ttnn.release_trace(mesh, trace_id)
                setattr(self, name, None)

    def _greedy(self, logits) -> list[int]:
        host = self.model.logits_to_torch(logits)
        return [int(v) for v in host.reshape(-1, host.shape[-1]).argmax(dim=-1)]

    def _draft(self, residual, tokens, positions) -> int:
        """MTP over both rows; returns the draft of the last row."""

        started = time.perf_counter()
        head = self.head
        head.set_tokens(tokens)
        head.set_scratch_position(positions)
        self._set_pair_positions([head.layer], positions[0], positions[1])
        head.layer.speculative_pair = True
        try:
            logits = head.draft_logits(
                residual, head.token_register, current_pos=head.scratch_pos, page_table=self.state.page_table
            )
        finally:
            head.layer.speculative_pair = False
        drafts = self._read_tokens(self._device_argmax(logits, "mtp_tokens_out")) if self.device_argmax else self._greedy(logits)
        ttnn.deallocate(logits)
        self.draft_seconds += time.perf_counter() - started
        return drafts[-1]

    # ------------------------------------------------------------------ API

    def start(self, prompt: torch.Tensor) -> list[int]:
        """Prefill the prompt, seed the MTP head, return the first token."""

        model = self.model
        prompt = torch.as_tensor(prompt, dtype=torch.int64).reshape(-1)
        n = int(prompt.numel())
        state = model.new_batch_state([n, 0], request_ids=(self.request_id, "spec-shadow"))
        # Both rows address the same sequence: mirror slot 0's pages into slot 1.
        pages = state.page_table_host.clone()
        pages[1] = pages[0]
        model.update_page_table(state, pages)
        self.state = state
        ids = torch.zeros((2, n), dtype=torch.int64)
        ids[0] = prompt
        residual_all = model.prefill_forward(ids, state=state, return_residual=True)
        last = _rows(residual_all, n - 1)
        logits = model.project_logits(last)
        ttnn.deallocate(last)
        first = self._greedy(logits)[0]
        ttnn.deallocate(logits)
        # Seed the MTP head over the prompt two positions at a time:
        # rows (hidden_i, x_{i+1}) and (hidden_{i+1}, x_{i+2}); the last pair
        # uses the first generated token and yields the first draft.
        full = torch.cat([prompt, torch.tensor([first], dtype=torch.int64)])
        draft = None
        i = 0
        while i < n:
            j = min(i + 1, n - 1)
            rows = ttnn.concat([_rows(residual_all, i), _rows(residual_all, j)], dim=2) if j != i else None
            if rows is None:
                single = _rows(residual_all, i)
                rows = ttnn.concat([single, single], dim=2)
                ttnn.deallocate(single)
            draft = self._draft(rows, [int(full[i + 1]), int(full[j + 1])], [i, j])
            ttnn.deallocate(rows)
            i += 2
        ttnn.deallocate(residual_all)
        # Active rows for decode; state slot 1 is the speculative row.
        state.active_mask = torch.tensor([True, True])
        self.position = n
        self.pending = (first, int(draft))
        self.tokens = [first]
        return [first]

    def step(self) -> list[int]:
        """Verify the pending draft; return the newly emitted tokens (1 or 2)."""

        if self.pending is None:
            raise RuntimeError("call start() first")
        started = time.perf_counter()
        model, state = self.model, self.state
        b, c = self.pending
        p = self.position
        # The first step of a traced decoder runs eagerly so every program of
        # the two-row graph is compiled before capture.
        use_trace = self.traced and getattr(self, "_warmed", False)
        store = model.ple_store
        t = time.perf_counter()
        snapshot = store.history_snapshot(self.request_id)
        ple = store.prepare([self.request_id], torch.tensor([[b, c]], dtype=torch.int64))
        ple_rows = torch.as_tensor(ple, dtype=torch.bfloat16).reshape(2, 1, -1)
        t = self._phase("ple_lookup", t)
        if use_trace:
            model.copy_tokens(state, torch.tensor([b, c], dtype=torch.int64))
            pos = torch.tensor([p, p + 1], dtype=torch.int32)
            _copy_host_to_device(pos, state.current_pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            state.host_positions = pos.clone()
            model._apply_qsa_variant(model._select_qsa_variant(state))
            ple_layer = self._ple_layer()
            staged = None if ple_layer is None else ple_layer.ple_staging.upload_decode(ple_rows)
            self._set_pair_positions(model.layers, p, p + 1)
            t = self._phase("h2d", t)
            residual, tokens_out = self._run_target_traced(staged)
            t = self._phase("target_replay", t)
            b_target, d = self._read_tokens(tokens_out) if self.device_argmax else self._greedy(tokens_out)
            t = self._phase("logits_readback", t)
        else:
            residual = self._target_rows([b, c], [p, p + 1], ple_rows)
            logits = model.project_logits(residual)
            if self.device_argmax:
                b_target, d = self._read_tokens(self._device_argmax(logits, "target_tokens_out"))
            else:
                b_target, d = self._greedy(logits)
            ttnn.deallocate(logits)
        accept = (b_target == c) and not self.force_reject
        if use_trace:
            t = time.perf_counter()
            if accept:
                self._commit_traced()
            t = self._phase("commit", t)
        else:
            for layer in model.layers:
                layer.commit_speculative_pair(accept)
        if accept:
            emitted = [c, d]
            next_token = d
            draft_positions = [p, p + 1]
            draft_tokens = [c, d]
            self.position = p + 2
            self.accepted += 1
        else:
            emitted = [b_target]
            next_token = b_target
            store.history_restore(self.request_id, snapshot)
            store.row_ids([self.request_id], torch.tensor([[b]], dtype=torch.int64))
            # Row 1 duplicates row 0 so the two-row MTP step stays well formed.
            draft_positions = [p, p]
            draft_tokens = [b_target, b_target]
            self.position = p + 1
        if use_trace:
            draft_started = time.perf_counter()
            t = draft_started
            self._fill_mtp_hidden(residual, accept)
            self.head.set_tokens(draft_tokens)
            self.head.set_scratch_position(draft_positions)
            self._set_pair_positions([self.head.layer], draft_positions[0], draft_positions[1])
            t = self._phase("mtp_fill_h2d", t)
            mtp_tokens = self._mtp_traced()
            t = self._phase("mtp_replay", t)
            draft = (self._read_tokens(mtp_tokens) if self.device_argmax else self._greedy(mtp_tokens))[-1]
            t = self._phase("mtp_readback", t)
            self.draft_seconds += time.perf_counter() - draft_started
        else:
            if not accept:
                row0 = _rows(residual, 0)
                mtp_rows = ttnn.concat([row0, row0], dim=2)
                ttnn.deallocate(row0)
            else:
                mtp_rows = residual
            draft = self._draft(mtp_rows, draft_tokens, draft_positions)
            if mtp_rows is not residual:
                ttnn.deallocate(mtp_rows)
            ttnn.deallocate(residual)
        self.pending = (next_token, draft)
        self.tokens.extend(emitted)
        self.steps += 1
        self._warmed = True
        elapsed = time.perf_counter() - started
        self.step_seconds += elapsed
        if use_trace:
            self.traced_steps = getattr(self, "traced_steps", 0) + 1
            self.traced_step_seconds = getattr(self, "traced_step_seconds", 0.0) + elapsed
            self.traced_tokens = getattr(self, "traced_tokens", 0) + len(emitted)
        return emitted

    def generate(self, prompt: torch.Tensor, max_new_tokens: int) -> list[int]:
        out = list(self.start(prompt))
        while len(out) < max_new_tokens:
            out.extend(self.step())
        return out[:max_new_tokens]

    def report(self) -> dict[str, float]:
        return {
            "traced": self.traced,
            "trace_capture_seconds": self.trace_capture_seconds,
            "steps": self.steps,
            "accepted": self.accepted,
            "acceptance_rate": self.accepted / max(self.steps, 1),
            "tokens_per_step": (len(self.tokens) - 1) / max(self.steps, 1),
            "eager_step_seconds": self.step_seconds / max(self.steps, 1),
            "traced_steps": getattr(self, "traced_steps", 0),
            "traced_step_seconds": getattr(self, "traced_step_seconds", 0.0) / max(getattr(self, "traced_steps", 0), 1),
            "phase_ms_per_step": {
                k: round(1e3 * v / max(getattr(self, "traced_steps", 0), 1), 2) for k, v in self.__dict__.get("phase_seconds", {}).items()
            },
            "traced_ms_per_token": (
                1e3 * getattr(self, "traced_step_seconds", 0.0) / max(getattr(self, "traced_tokens", 0), 1)
                if getattr(self, "traced_steps", 0)
                else None
            ),
            "eager_draft_seconds": self.draft_seconds / max(self.steps + 1, 1),
        }
