# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full Qwen3-8B model for one Blackhole chip: embedding, 36 optimized decoder layers, final norm, LM head.

``Qwen3Model`` owns the device weights and the device-side prefill and decode graphs. It carries the
optimized decoder's policy unchanged (``OptimizedPolicy``: BFP8/BFP4 DRAM-sharded projections at LoFi,
BF16 activations and KV cache, L1 width-sharded decode residual over 32 cores) and adds the terminal
path: a sharded final RMSNorm on the residual layout and a DRAM-sharded BFP8 LM head split into
16384-column matmuls whose concatenated output is the sampler-ready ``[1, 1, 32, padded_vocab]``
logits tile.

Prefill runs one row at a time through every layer (the decoder chunks a row internally in 4096-token
physical chunks). The compute length of a fresh row is padded to a fixed bucket so the set of prefill
programs is finite and can be compiled before any trace is captured; the cache rows past the real
prompt land in the request's own last block or in a scratch block that the page table names for every
column past the request's allocation, never in another request's pages.

Decode is one device-only graph over persistent token, position and page-table tensors; the position
vector advances on device (``plus_one``) inside the graph when the caller samples on device, so the
sampled token written back into the token tensor is consumed at the right position by the next replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

import ttnn
from models.autoports.qwen_qwen3_8b.tt.functional_decoder import (
    TILE_SIZE,
    DecoderGeometry,
    decode_physical_batch,
    round_up,
    validate_hf_config,
)
from models.autoports.qwen_qwen3_8b.tt.hf_weights import CheckpointReader
from models.autoports.qwen_qwen3_8b.tt.optimized_decoder import (
    DEFAULT_POLICY,
    DTYPE_NAMES,
    FIDELITY_NAMES,
    WEIGHT_BYTES_PER_ELEMENT,
    DecodeMatmulRole,
    OptimizedDecoder,
    OptimizedPolicy,
    dram_sharded_program_config,
    dram_sharded_weight_memory_config,
    width_sharded_memory_config,
)

PREFILL_BUCKETS = (32, 64, 128, 256, 512, 1024, 2048, 4096)
PREFILL_CHUNK = 4096
LM_HEAD_SPLIT = 16384
VOCAB_ALIGNMENT = 1024
DECODE_BUCKETS = (1, 8, 16, 24, 32)
LOGITS_DTYPES = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32}


def prefill_compute_len(seq_len: int) -> int:
    """Physical row count a fresh prompt of ``seq_len`` tokens is computed at.

    Lengths up to one physical chunk round up to the next bucket; longer prompts round up to whole
    4096-token chunks so every chunk uses a program that the 40960-token warmup already compiled.
    """
    if seq_len < 1:
        raise ValueError("prefill needs at least one token")
    if seq_len <= PREFILL_CHUNK:
        return next(b for b in PREFILL_BUCKETS if b >= seq_len)
    return round_up(seq_len, PREFILL_CHUNK)


def padded_vocab(vocab_size: int) -> int:
    return round_up(vocab_size, VOCAB_ALIGNMENT)


def lm_head_splits(padded: int, split: int = LM_HEAD_SPLIT) -> list[int]:
    widths = [split] * (padded // split)
    if padded % split:
        widths.append(padded % split)
    return widths


@dataclass(frozen=True)
class TerminalPolicy:
    """Dtype and fidelity of the tensors the full model adds around the decoder stack."""

    embedding_dtype: str = "bf16"
    final_norm_dtype: str = "bf16"
    final_norm_fidelity: str = "hifi4"
    lm_head_dtype: str = "bfp8"
    lm_head_decode_fidelity: str = "hifi2"
    lm_head_prefill_fidelity: str = "hifi2"
    lm_head_prefill_fp32_acc: bool = True
    prefill_logits_dtype: str = "fp32"
    lm_head_split: int = LM_HEAD_SPLIT
    lm_head_in0_block_w: int = 4
    lm_head_readers: int = 1

    def describe(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


DEFAULT_TERMINAL_POLICY = TerminalPolicy()


class Qwen3Model:
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        *,
        hf_config: Any,
        reader: CheckpointReader,
        num_layers: int | None = None,
        policy: OptimizedPolicy = DEFAULT_POLICY,
        terminal_policy: TerminalPolicy = DEFAULT_TERMINAL_POLICY,
        tensor_cache_path: str | Path | None = None,
        log=print,
    ) -> None:
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.geometry: DecoderGeometry = validate_hf_config(hf_config, 0)
        self.num_layers = int(num_layers or hf_config.num_hidden_layers)
        self.vocab_size = int(hf_config.vocab_size)
        self.padded_vocab_size = padded_vocab(self.vocab_size)
        self.block_size = TILE_SIZE
        self.max_context = self.geometry.max_position_embeddings
        self.policy = policy
        self.terminal_policy = terminal_policy
        self.tied_embeddings = reader.tied_embeddings
        self.checkpoint_revision = reader.revision
        self.grid = mesh_device.compute_with_storage_grid_size()
        self.dram_banks = mesh_device.dram_grid_size().x
        g = self.geometry
        H = g.hidden_size
        arch = mesh_device.arch()

        self.layers: list[OptimizedDecoder] = []
        rope_tables = None
        for layer_idx in range(self.num_layers):
            log(f"loading layer {layer_idx}")
            state = reader.layer_state(layer_idx)
            layer = OptimizedDecoder.from_state_dict(
                state,
                hf_config=hf_config,
                layer_idx=layer_idx,
                mesh_device=mesh_device,
                policy=policy,
                tensor_cache_path=tensor_cache_path,
                rope_tables=rope_tables,
            )
            if rope_tables is None:
                w = layer.weights
                rope_tables = (w.rope_cos, w.rope_sin, w.rope_cos_table, w.rope_sin_table)
            self.layers.append(layer)
            del state
        log("loading embedding, final norm and LM head")
        self.embedding = ttnn.from_torch(
            reader.embedding().to(torch.bfloat16),
            dtype=DTYPE_NAMES[terminal_policy.embedding_dtype],
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.final_norm = ttnn.from_torch(
            reader.final_norm().to(torch.float32).reshape(1, 1, 1, H),
            dtype=DTYPE_NAMES[terminal_policy.final_norm_dtype],
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        lm = reader.lm_head().to(torch.float32).T.contiguous()
        if lm.shape[1] < self.padded_vocab_size:
            lm = torch.nn.functional.pad(lm, (0, self.padded_vocab_size - lm.shape[1]))
        self.lm_head_widths = lm_head_splits(self.padded_vocab_size, terminal_policy.lm_head_split)
        self.lm_head_weights: list[ttnn.Tensor] = []
        start = 0
        for width in self.lm_head_widths:
            self.lm_head_weights.append(
                ttnn.from_torch(
                    lm[:, start : start + width].unsqueeze(0).unsqueeze(0),
                    dtype=DTYPE_NAMES[terminal_policy.lm_head_dtype],
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=dram_sharded_weight_memory_config(H, width, self.dram_banks),
                )
            )
            start += width
        del lm

        def ckc(fidelity: str, fp32: bool, l1acc: bool):
            return ttnn.init_device_compute_kernel_config(
                arch,
                math_fidelity=FIDELITY_NAMES[fidelity],
                math_approx_mode=False,
                fp32_dest_acc_en=fp32,
                packer_l1_acc=l1acc,
            )

        first = self.layers[0]
        self.residual_memory_config = first.decode_residual_memory_config
        self.norm_program_config = first.norm_program_config
        self.norm_compute_config = ckc(terminal_policy.final_norm_fidelity, True, False)
        self.prefill_norm_compute_config = ckc(terminal_policy.final_norm_fidelity, True, False)
        self.lm_head_role = DecodeMatmulRole(
            terminal_policy.lm_head_dtype,
            terminal_policy.lm_head_decode_fidelity,
            False,
            policy.residual_cores,
            terminal_policy.lm_head_in0_block_w,
            terminal_policy.lm_head_readers,
        )
        self.lm_head_decode_compute = ckc(terminal_policy.lm_head_decode_fidelity, False, True)
        self.lm_head_prefill_compute = ckc(
            terminal_policy.lm_head_prefill_fidelity, terminal_policy.lm_head_prefill_fp32_acc, False
        )
        self.lm_head_program_configs = [
            dram_sharded_program_config(self.lm_head_role, H, width, self.dram_banks) for width in self.lm_head_widths
        ]
        self.lm_head_output_memory_configs = [
            width_sharded_memory_config(width, policy.residual_cores, self.grid) for width in self.lm_head_widths
        ]
        self.eps = g.rms_norm_eps

    # ------------------------------------------------------------------ sizes

    def weight_bytes(self) -> dict[str, int]:
        g = self.geometry
        per_layer = self.layers[0].weight_bytes_per_token()
        resident = self.layers[0].weight_bytes_resident()
        emb = self.vocab_size * g.hidden_size * 2
        lm = int(g.hidden_size * self.padded_vocab_size * WEIGHT_BYTES_PER_ELEMENT[self.terminal_policy.lm_head_dtype])
        rope = 4 * g.max_position_embeddings * g.head_dim * 2
        norms = self.num_layers * 2 * (g.hidden_size + g.head_dim) * 2 + g.hidden_size * 2
        return {
            "decoder_layer_weights_per_token": per_layer,
            "decoder_layer_weights_resident": resident,
            "decoder_stack_resident": resident * self.num_layers,
            "embedding": emb,
            "lm_head": lm,
            "rope_tables_shared": rope,
            "norm_weights": norms,
            "total_resident": resident * self.num_layers + emb + lm + rope + norms,
        }

    def kv_bytes_per_token(self) -> int:
        g = self.geometry
        return 2 * g.num_kv_heads * g.head_dim * 2 * self.num_layers

    # ---------------------------------------------------------------- caches

    def allocate_kv_cache(self, num_blocks: int, dtype: ttnn.DataType = ttnn.bfloat16) -> list[tuple]:
        """Per-layer ``(K, V)`` paged caches with ``num_blocks + 1`` blocks; block ``num_blocks`` is the scratch block.

        Page-table columns past a request's allocation name the scratch block, so padded prefill rows never
        touch another request's pages; callers that pass this cache to the generator pass
        ``scratch_block=num_blocks``.
        """
        g = self.geometry
        return [
            OptimizedDecoder.allocate_kv_cache(
                self.mesh_device,
                num_blocks=num_blocks + 1,
                block_size=self.block_size,
                num_kv_heads=g.num_kv_heads,
                head_dim=g.head_dim,
                dtype=dtype,
            )
            for _ in range(self.num_layers)
        ]

    # --------------------------------------------------------------- prefill

    def embed_ids(self, token_ids: Sequence[int], compute_len: int) -> ttnn.Tensor:
        """Embed one row padded to ``compute_len`` tokens; returns ``[1, 1, compute_len, H]`` TILE DRAM."""
        ids = torch.zeros(1, compute_len, dtype=torch.int32)
        ids[0, : len(token_ids)] = torch.tensor(list(token_ids), dtype=torch.int32)
        tt_ids = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device)
        emb = ttnn.embedding(tt_ids, self.embedding, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_ids.deallocate(True)
        return ttnn.unsqueeze_to_4D(emb)

    def prefill_hidden(
        self,
        token_ids: Sequence[int],
        *,
        start_pos: int,
        user_slot: int,
        page_table: ttnn.Tensor,
        kv_cache: list[tuple],
        compute_len: int | None = None,
    ) -> ttnn.Tensor:
        """Run one row through every layer; returns the final hidden state ``[1, 1, compute_len, H]``.

        Rows ``[0, len(token_ids))`` are the prompt; rows past it are padding whose cache writes stay in
        the row's last block or the scratch block the page table names past the allocation.
        """
        seq_len = len(token_ids)
        if compute_len is None:
            compute_len = prefill_compute_len(seq_len) if start_pos == 0 else seq_len
        if compute_len < seq_len:
            raise ValueError(f"compute_len {compute_len} < prompt length {seq_len}")
        x = self.embed_ids(token_ids, compute_len)
        for layer_idx, layer in enumerate(self.layers):
            y = layer.prefill_forward(
                x,
                page_table=page_table,
                kv_cache=kv_cache[layer_idx],
                user_id=user_slot,
                start_pos=start_pos,
                seq_lens=[compute_len],
            )
            x.deallocate(True)
            x = y
        return x

    def gather_rows(self, hidden: ttnn.Tensor, indices: Sequence[int]) -> ttnn.Tensor:
        """Pick rows of ``hidden`` ``[1, 1, L, H]`` by runtime index; returns ``[1, n, H]`` ROW_MAJOR DRAM."""
        rows = ttnn.untilize(hidden, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        table = ttnn.reshape(rows, (hidden.shape[-2], hidden.shape[-1]))
        idx = ttnn.from_torch(
            torch.tensor([list(indices)], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
        )
        picked = ttnn.embedding(idx, table, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        idx.deallocate(True)
        rows.deallocate(True)
        return picked

    def rows_to_tile(self, rows: list[ttnn.Tensor]) -> ttnn.Tensor:
        """Place ``[1, 1, H]`` row-major pieces at rows ``0..n-1`` of one ``[1, 1, 32, H]`` TILE DRAM tile.

        Row ``i`` is padded into its own 32-row tile and the tiles are summed, so the program set is the 32
        pad variants plus one add, independent of the row count (a concat over ``n`` inputs would compile per
        ``n``). Rows past ``n`` are zero.
        """
        if not 1 <= len(rows) <= TILE_SIZE:
            raise ValueError(f"rows_to_tile takes 1 to {TILE_SIZE} rows, got {len(rows)}")
        tile = None
        for i, piece in enumerate(rows):
            padded = ttnn.pad(piece, [(0, 0), (i, TILE_SIZE - 1 - i), (0, 0)], 0.0)
            tiled = ttnn.to_layout(padded, ttnn.TILE_LAYOUT)
            if padded.buffer_address() != piece.buffer_address():
                padded.deallocate(True)
            if tile is None:
                tile = tiled
                continue
            summed = ttnn.add(tile, tiled, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            tile.deallocate(True)
            tiled.deallocate(True)
            tile = summed
        return ttnn.unsqueeze_to_4D(tile)

    def stack_rows_tile(self, rows: list[ttnn.Tensor], pad_to: int = TILE_SIZE) -> ttnn.Tensor:
        """Stack ``[1, n_i, H]`` row-major pieces into ``[1, 1, pad_to, H]`` TILE DRAM (eager readiness paths)."""
        stacked = rows[0] if len(rows) == 1 else ttnn.concat(rows, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        count = stacked.shape[1]
        padded = stacked
        if count < pad_to:
            padded = ttnn.pad(stacked, [(0, 0), (0, pad_to - count), (0, 0)], 0.0)
        tiled = ttnn.to_layout(padded, ttnn.TILE_LAYOUT)
        if padded is not stacked and padded.buffer_address() != stacked.buffer_address():
            padded.deallocate(True)
        if stacked is not rows[0]:
            stacked.deallocate(True)
        return ttnn.unsqueeze_to_4D(tiled)

    # ---------------------------------------------------------- terminal path

    def _decode_norm(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=self.final_norm,
            program_config=self.norm_program_config,
            memory_config=self.residual_memory_config,
            compute_kernel_config=self.norm_compute_config,
        )

    def lm_head_decode(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Width-sharded ``[1, 1, 32, H]`` residual-layout input to ``[1, 1, 32, padded_vocab]`` DRAM logits."""
        pieces = []
        for weight, program_config, memory_config in zip(
            self.lm_head_weights, self.lm_head_program_configs, self.lm_head_output_memory_configs
        ):
            out = ttnn.linear(
                x,
                weight,
                program_config=program_config,
                compute_kernel_config=self.lm_head_decode_compute,
                dtype=ttnn.bfloat16,
                memory_config=memory_config,
            )
            piece = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
            out.deallocate(True)
            pieces.append(piece)
        logits = ttnn.concat(pieces, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for piece in pieces:
            piece.deallocate(True)
        return logits

    def logits_from_tile(self, tile: ttnn.Tensor) -> ttnn.Tensor:
        """Final norm plus LM head of a ``[1, 1, 32, H]`` TILE DRAM tile of last-token hidden states."""
        x = ttnn.to_memory_config(tile, self.residual_memory_config)
        normed = self._decode_norm(x)
        x.deallocate(True)
        logits = self.lm_head_decode(normed)
        normed.deallocate(True)
        return logits

    def logits_all_rows(self, hidden: ttnn.Tensor) -> torch.Tensor:
        """Final norm plus LM head over every row of ``[1, 1, L, H]`` DRAM; returns ``[L, vocab]`` float32 on host."""
        normed = ttnn.rms_norm(
            hidden,
            epsilon=self.eps,
            weight=self.final_norm,
            compute_kernel_config=self.prefill_norm_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        pieces = []
        logits_dtype = LOGITS_DTYPES[self.terminal_policy.prefill_logits_dtype]
        for weight in self.lm_head_weights:
            out = ttnn.experimental.minimal_matmul(
                normed,
                weight,
                dtype=logits_dtype,
                compute_kernel_config=self.lm_head_prefill_compute,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            pieces.append(ttnn.to_torch(out)[0, 0].to(torch.float32))
            out.deallocate(True)
        normed.deallocate(True)
        return torch.cat(pieces, dim=-1)[:, : self.vocab_size]

    # ---------------------------------------------------------------- decode

    def ttnn_decode_forward(
        self,
        tokens: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: list[tuple],
        *,
        on_device_sampling: bool,
    ) -> ttnn.Tensor:
        """Device-only decode step over persistent inputs.

        ``tokens`` is ``[1, 1, 1, 32]`` uint32 ROW_MAJOR (one slot per row of the residual tile),
        ``current_pos`` ``[B_phys]`` int32 (``-1`` marks an inactive row) and ``page_table``
        ``[B_phys, blocks]`` int32. With ``on_device_sampling`` the result is the full
        ``[1, 1, 32, padded_vocab]`` logits tile for the sampler and ``current_pos`` is advanced on
        device; otherwise the first ``B_phys`` rows of the real vocabulary are returned for host sampling.
        """
        users = current_pos.shape[0]
        physical, _, _ = decode_physical_batch(users, self.grid.x, self.grid.y)
        if physical != users:
            raise ValueError(f"decode rows {users} must be a physical batch (got physical {physical})")
        emb = ttnn.embedding(tokens, self.embedding, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        emb4 = ttnn.unsqueeze_to_4D(emb)
        x = ttnn.to_memory_config(emb4, self.residual_memory_config)
        emb.deallocate(True)
        for layer_idx, layer in enumerate(self.layers):
            y = layer.decode_forward(x, current_pos=current_pos, page_table=page_table, kv_cache=kv_cache[layer_idx])
            x.deallocate(True)
            x = y
        normed = self._decode_norm(x)
        x.deallocate(True)
        logits = self.lm_head_decode(normed)
        normed.deallocate(True)
        if on_device_sampling:
            ttnn.plus_one(current_pos, skip_negative_entries=True)
            return logits
        rows = ttnn.slice(logits, [0, 0, 0, 0], [1, 1, users, self.vocab_size], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        logits.deallocate(True)
        return rows


def decode_bucket(batch: int, grid_x: int = 11, grid_y: int = 10) -> int:
    """Smallest supported physical decode batch that holds ``batch`` users."""
    if not 1 <= batch <= DECODE_BUCKETS[-1]:
        raise ValueError(f"decode batch {batch} outside [1, {DECODE_BUCKETS[-1]}]")
    bucket = next(b for b in DECODE_BUCKETS if b >= batch)
    physical, _, _ = decode_physical_batch(bucket, grid_x, grid_y)
    if physical != bucket:
        raise ValueError(f"bucket {bucket} is not a physical batch on this grid")
    return bucket
