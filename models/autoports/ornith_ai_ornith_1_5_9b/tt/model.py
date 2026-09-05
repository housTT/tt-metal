# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Text autoregressive model preserving the optimized TP4 decoder contract."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import torch
from loguru import logger

import ttnn

from ..reference.hf_reference import CHECKPOINT_TEXT_PREFIX, load_layer_state_dict, load_text_config, resolve_model_path
from .functional_decoder import DEFAULT_PAGE_BLOCK_SIZE, DEFAULT_PREFILL_CHUNK, num_blocks_for_context
from .multichip_decoder import TP, MeshCCLManager, MeshConfig, MultichipDecoder, fabric_router_config


def open_ornith_mesh(*, trace_region_size=100_000_000):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, router_config=fabric_router_config())
    return ttnn.open_mesh_device(ttnn.MeshShape(1, TP), l1_small_size=32768, trace_region_size=trace_region_size)


def close_ornith_mesh(mesh):
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


class SamplingCCL:
    """Common sampler hook with full-grid, cycling TP4 collective semaphores."""

    def __init__(self, mesh):
        self.ccl = MeshCCLManager(mesh, 1)

    def get_and_cycle_ag_semaphore_handles(self, cluster_axis=None):
        return self.ccl.get_ag_ping_pong_semaphore()

    def get_and_cycle_barrier_semaphore_handle(self, cluster_axis=None):
        return self.ccl.get_barrier_semaphore()

    def line_all_gather(self, tensor, *, dim, cluster_axis=None, memory_config=None, num_links=None):
        return ttnn.experimental.all_gather_async(
            tensor,
            dim=dim,
            persistent_output_buffer=None,
            multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            num_links=1,
            topology=ttnn.Topology.Ring,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        )


@dataclass
class ModelCache:
    """Caller-ownable KV and hybrid recurrent state; buffers retain stable addresses."""

    batch_size: int
    context: int
    kv: list
    decode_layers: list
    prefill_layers: list
    active_recurrent: object
    active_conv: object
    zero_hidden: object


class OrnithModel:
    def __init__(
        self,
        model_path,
        mesh_device,
        *,
        layer_indices=None,
        max_context=None,
        prefill_chunk=DEFAULT_PREFILL_CHUNK,
        lm_head_dtype=ttnn.bfloat16,
        lm_head_fidelity=ttnn.MathFidelity.HiFi4,
        lm_head_columns=32768,
        lm_head_block_w=1,
        lm_head_readers=2,
    ):
        self.mesh_device = mesh_device
        if tuple(mesh_device.shape) != (1, TP):
            raise ValueError("The optimized Ornith full model requires a 1x4 Blackhole ring")
        candidate = Path(model_path) if model_path is not None else resolve_model_path()
        self.model_path = candidate if (candidate / "model.safetensors.index.json").is_file() else resolve_model_path()
        self.hf_config = load_text_config(self.model_path)
        self.dim = self.hf_config.hidden_size
        self.vocab_size = self.hf_config.vocab_size
        self.padded_vocab_size = 262144
        self.max_context = self.hf_config.max_position_embeddings if max_context is None else int(max_context)
        if not 1 <= self.max_context <= self.hf_config.max_position_embeddings:
            raise ValueError("max_context exceeds the HF contract")
        self.prefill_chunk = prefill_chunk
        self.page_block_size = DEFAULT_PAGE_BLOCK_SIZE
        self.ccl = SamplingCCL(mesh_device)
        self.layer_indices = (
            list(range(self.hf_config.num_hidden_layers)) if layer_indices is None else list(layer_indices)
        )
        with open(self.model_path / "model.safetensors.index.json") as f:
            weight_map = json.load(f)["weight_map"]

        def read(key):
            from safetensors import safe_open

            with safe_open(str(self.model_path / weight_map[key]), framework="pt", device="cpu") as f:
                return f.get_tensor(key)

        self.embed_weight = self.upload(
            read(CHECKPOINT_TEXT_PREFIX + "embed_tokens.weight"), layout=ttnn.ROW_MAJOR_LAYOUT, shard_dim=1
        )
        self.norm_weight = self.upload((read(CHECKPOINT_TEXT_PREFIX + "norm.weight").float() + 1).reshape(1, 1, 1, -1))
        head_key = (
            CHECKPOINT_TEXT_PREFIX + "embed_tokens.weight" if self.hf_config.tie_word_embeddings else "lm_head.weight"
        )
        head = read(head_key).T.contiguous()
        head = torch.nn.functional.pad(head, (0, self.padded_vocab_size - self.vocab_size))
        dram = mesh_device.dram_grid_size()
        banks = dram.x * dram.y
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))])
        self.head_columns = int(lm_head_columns)
        local_vocab = self.padded_vocab_size // TP
        if local_vocab % self.head_columns:
            raise ValueError("LM-head columns must divide the per-device padded vocabulary")
        head_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(grid, [self.dim, self.head_columns // banks], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.head_weights = []
        for start in range(0, local_vocab, self.head_columns):
            packed = torch.cat(
                [
                    head[:, rank * local_vocab + start : rank * local_vocab + start + self.head_columns]
                    for rank in range(TP)
                ],
                dim=1,
            )
            self.head_weights.append(self.upload(packed, dtype=lm_head_dtype, shard_dim=1, memory=head_mem))
        del head
        self.head_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=lm_head_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.head_input_memory = ttnn.create_sharded_memory_config(
            shape=(32, self.dim // 32),
            core_grid=ttnn.CoreGrid(x=8, y=4),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.head_program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=lm_head_block_w,
            per_core_M=1,
            per_core_N=self.head_columns // 32 // 32,
            num_workers_per_dram_bank=lm_head_readers,
            fused_activation=None,
        )
        self.layers = []
        mesh_config = MeshConfig()
        # Full-stack recurrent states share L1 with the FP32 GDN prefill matmul.
        # Splitting N preserves its K accumulation and exact projection output.
        mesh_config = replace(
            mesh_config,
            local=replace(mesh_config.local, large_prefill_role_configs={"gdn_out": {"out_block_w": 6}}),
        )
        for index in self.layer_indices:
            layer = MultichipDecoder.from_state_dict(
                load_layer_state_dict(index, self.model_path),
                hf_config=self.hf_config,
                layer_idx=index,
                mesh_device=mesh_device,
                max_context=self.max_context,
                prefill_chunk=prefill_chunk,
                mesh_config=mesh_config,
            )
            self.layers.append(layer)
            logger.info("Loaded layer {} ({})", index, layer.kind)
        self.cache = None
        self.counters = dict(
            model_replays=0,
            sampling_replays=0,
            token_refreshes=0,
            position_refreshes=0,
            rope_refreshes=0,
            page_table_refreshes=0,
            readbacks=0,
            synchronizations=0,
            read_waits=0,
        )

    @classmethod
    def from_pretrained(cls, model_dir=None, mesh_device=None, **kwargs):
        return cls(model_dir, mesh_device, **kwargs)

    def upload(
        self,
        tensor,
        *,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        shard_dim=None,
        memory=ttnn.DRAM_MEMORY_CONFIG,
        device=True,
    ):
        mapper = (
            ttnn.ReplicateTensorToMesh(self.mesh_device)
            if shard_dim is None
            else ttnn.ShardTensorToMesh(self.mesh_device, dim=shard_dim)
        )
        return ttnn.from_torch(
            tensor.contiguous(),
            dtype=dtype,
            layout=layout,
            mesh_mapper=mapper,
            device=self.mesh_device if device else None,
            memory_config=memory,
        )

    def allocate_cache(self, batch_size=1, context=None):
        context = self.max_context if context is None else int(context)
        if not 1 <= batch_size <= 32 or not 1 <= context <= self.max_context:
            raise ValueError("cache requires batch 1..32 and context within supported bounds")
        blocks = num_blocks_for_context(context, self.page_block_size) * batch_size
        decode, prefill, kv = [], [], []
        for original in self.layers:
            layer = copy.copy(original)
            layer.allocate_state(batch_size)
            pair = layer.allocate_kv_cache(blocks) if layer.is_full_attention else None
            decode.append(layer)
            kv.append(pair)
            if batch_size == 1:
                prefill.append(layer)
            else:
                single = copy.copy(layer)
                single.allocate_state(1)
                prefill.append(single)
        active_recurrent = self.upload(torch.ones(batch_size, 1, 1, 1), dtype=ttnn.float32)
        # where selects its SFPU data path from the condition dtype. Match the
        # BF16 convolution buffers; FP32 predicates reinterpret their elements.
        active_conv = self.upload(torch.ones(batch_size, 1, 1), dtype=ttnn.bfloat16)
        cache = ModelCache(
            batch_size,
            context,
            kv,
            decode,
            prefill,
            active_recurrent,
            active_conv,
            self.upload(torch.zeros(1, 1, self.dim)),
        )
        self.cache = cache
        return cache

    def page_table(self, cache):
        width = num_blocks_for_context(cache.context, self.page_block_size)
        return torch.arange(width * cache.batch_size, dtype=torch.int32).reshape(cache.batch_size, width)

    @staticmethod
    def cache_buffers(cache):
        buffers = []
        seen = set()
        for layer in cache.decode_layers + cache.prefill_layers:
            values = (
                [layer.k_cache, layer.v_cache]
                if layer.is_full_attention
                else [layer.recurrent_state] + layer.conv_state
            )
            for tensor in values:
                address = (tensor.memory_config().buffer_type, tensor.buffer_address())
                if address not in seen:
                    buffers.append(tensor)
                    seen.add(address)
        return buffers

    def reset_cache(self, cache, *, clear_kv=False):
        for layer in cache.decode_layers + ([] if cache.batch_size == 1 else cache.prefill_layers):
            layer.reset_state()
        if clear_kv:
            for pair in cache.kv:
                if pair:
                    for tensor in pair:
                        ttnn.multiply(tensor, 0, output_tensor=tensor)

    def build_sampler(self, *, max_top_k=32, force_argmax=False):
        from models.common.sampling import SamplingGenerator

        args = SimpleNamespace(
            vocab_size=self.vocab_size,
            padded_vocab_size=self.padded_vocab_size,
            cluster_shape=(1, TP),
            max_batch_size=32,
            max_top_k=max_top_k,
            sampling_dp=1,
            sub_core_grids=None,
            sub_core_grid_topk=None,
            start_core=ttnn.CoreCoord(0, 0),
            pad_logits_to_power_of_2=False,
            model_config={},
        )
        if force_argmax:
            args.model_config["SAMPLING_AG_CONFIG"] = {
                "allow_force_argmax": True,
                "num_links": 1,
                "topology": ttnn.Topology.Ring,
            }
        return SamplingGenerator(args=args, mesh_device=self.mesh_device, tt_ccl=self.ccl)

    def embed(self, ids):
        batch, seq = ids.shape
        local = ttnn.embedding(ids, self.embed_weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        local = ttnn.reshape(local, [1, 1, batch * seq, self.dim // TP])
        whole = self.ccl.line_all_gather(local, dim=3)
        ttnn.deallocate(local)
        return ttnn.reshape(whole, [batch, seq, self.dim])

    def terminal(self, hidden):
        """One <=32-row tile to padded vocabulary shards; the sampler masks vocabulary padding."""
        rows = int(hidden.shape[-2])
        flat = ttnn.reshape(hidden, [1, 1, rows, self.dim])
        flat = ttnn.to_memory_config(flat, ttnn.DRAM_MEMORY_CONFIG)
        if rows < 32:
            flat = ttnn.pad(flat, [(0, 0), (0, 0), (0, 32 - rows), (0, 0)], 0.0)
        normalized = ttnn.rms_norm(flat, weight=self.norm_weight, epsilon=self.hf_config.rms_norm_eps)
        sharded = ttnn.to_memory_config(normalized, self.head_input_memory)
        parts = []
        for weight in self.head_weights:
            out = ttnn.linear(
                sharded,
                weight,
                program_config=self.head_program,
                compute_kernel_config=self.head_compute,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            parts.append(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG))
            ttnn.deallocate(out)
        if len(parts) == 1:
            # concat of one input aliases it: ownership transfers to the caller.
            logits = parts[0]
        else:
            logits = ttnn.concat(parts, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for part in parts:
                ttnn.deallocate(part)
        ttnn.deallocate(sharded)
        ttnn.deallocate(normalized)
        return logits

    def logits_to_host(self, logits, rows):
        self.counters["readbacks"] += 1
        return ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1))[
            0, 0, :rows, : self.vocab_size
        ].float()

    def _transfer_slot(self, source, target, slot, *, into_batch):
        for a, b in zip(source, target):
            if a.is_full_attention:
                continue
            for src, dst in zip([a.recurrent_state] + a.conv_state, [b.recurrent_state] + b.conv_state):
                if into_batch:
                    # The decoder stores every hybrid state with batch on axis zero.
                    mask_shape = [int(dst.shape[0])] + [1] * (len(dst.shape) - 1)
                    mask = torch.zeros(mask_shape)
                    mask[slot] = 1
                    select = self.upload(mask, dtype=dst.dtype)
                    repeats = [int(dst.shape[0])] + [1] * (len(src.shape) - 1)
                    wide = ttnn.repeat(src, ttnn.Shape(repeats))
                    ttnn.where(select, wide, dst, output_tensor=dst)
                    ttnn.deallocate(wide)
                    ttnn.deallocate(select)
                else:
                    starts, ends = [0] * len(src.shape), list(src.shape)
                    starts[0], ends[0] = slot, slot + 1
                    row = ttnn.slice(src, starts, ends)
                    ttnn.copy(row, dst)
                    ttnn.deallocate(row)

    def prefill_forward(
        self, tokens, *, page_table, kv_cache, prompt_lens, slots=None, start_pos=None, return_all_logits=False
    ):
        """Ragged logical prompts into explicit fixed cache slots. Padding stays inside each decoder."""
        cache = kv_cache
        slots = list(range(len(prompt_lens))) if slots is None else list(slots)
        starts = [0] * len(slots) if start_pos is None else list(start_pos)
        if len(set(slots)) != len(slots) or any(s < 0 or s >= cache.batch_size for s in slots):
            raise ValueError("slots must be unique valid cache rows")
        if not (len(tokens) == len(slots) == len(prompt_lens) == len(starts)) or not slots:
            raise ValueError("tokens, slots, prompt_lens and start_pos must describe the same requests")
        table = torch.as_tensor(page_table)
        if table.ndim != 2 or table.shape[0] != cache.batch_size:
            raise ValueError("page table must describe every fixed slot")
        if table.shape[1] < num_blocks_for_context(cache.context, self.page_block_size):
            raise ValueError("page table does not cover the cache attention read window")
        physical_blocks = next(pair[0].shape[0] for pair in cache.kv if pair is not None)
        if bool((table < 0).any()) or bool((table >= physical_blocks).any()):
            raise ValueError("page table addresses an unallocated physical block")
        for row, length in zip(tokens, prompt_lens):
            ids = torch.as_tensor(row)
            if length > ids.numel() or bool((ids[:length] < 0).any()) or bool((ids[:length] >= self.vocab_size).any()):
                raise ValueError("invalid logical token IDs or prompt length")
        results = []
        for user, (slot, length, start) in enumerate(zip(slots, prompt_lens, starts)):
            length = int(length)
            if length < 1 or start < 0 or start + length > cache.context:
                raise ValueError("prompt window exceeds cache context")
            if cache.batch_size > 1 and start:
                self._transfer_slot(cache.decode_layers, cache.prefill_layers, slot, into_batch=False)
            elif start == 0:
                for layer in cache.prefill_layers:
                    layer.reset_state()
            pt = self.upload(
                torch.as_tensor(page_table)[slot : slot + 1].to(torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            chunks = []
            for offset in range(0, length, self.prefill_chunk):
                logical = min(self.prefill_chunk, length - offset)
                ids = self.upload(
                    torch.as_tensor(tokens[user][offset : offset + logical]).reshape(1, -1).to(torch.int32),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                x = self.embed(ids)
                for layer in cache.prefill_layers:
                    nxt = layer.prefill_forward(x, start_pos=start + offset, page_table=pt)
                    ttnn.deallocate(x)
                    x = nxt
                if return_all_logits:
                    for row in range(0, logical, 32):
                        n = min(32, logical - row)
                        hidden = ttnn.slice(x, [0, row, 0], [1, row + n, self.dim])
                        logits = self.terminal(hidden)
                        chunks.append(self.logits_to_host(logits, n))
                        ttnn.deallocate(logits)
                elif offset + logical == length:
                    hidden = ttnn.slice(x, [0, logical - 1, 0], [1, logical, self.dim])
                    chunks.append(ttnn.clone(hidden))
                ttnn.deallocate(x)
                ttnn.deallocate(ids)
            if cache.batch_size > 1:
                self._transfer_slot(cache.prefill_layers, cache.decode_layers, slot, into_batch=True)
            ttnn.deallocate(pt)
            results.append(torch.cat(chunks) if return_all_logits else chunks[-1])
        if return_all_logits:
            return results
        slot_rows = [cache.zero_hidden] * cache.batch_size
        for slot, hidden in zip(slots, results):
            slot_rows[slot] = hidden
        combined = slot_rows[0] if cache.batch_size == 1 else ttnn.concat(slot_rows, dim=1)
        return self.terminal(combined)

    def decode_forward(self, tokens, *, current_pos, rot_idxs, page_table, kv_cache, advance_positions=True):
        """Device-only token-to-logits path over caller-owned persistent state."""
        cache = kv_cache
        batch = cache.batch_size
        ids = ttnn.reshape(ttnn.slice(tokens, [0, 0, 0, 0], [1, 1, 1, batch]), [batch, 1])
        x = self.embed(ids)
        for layer in cache.decode_layers:
            saved = []
            if batch > 1 and not layer.is_full_attention:
                saved = [ttnn.clone(buf) for buf in [layer.recurrent_state] + layer.conv_state]
            nxt = layer.decode_forward(x, current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table)
            if saved:
                masks = [cache.active_recurrent] + [cache.active_conv] * len(layer.conv_state)
                for old, buf, mask in zip(saved, [layer.recurrent_state] + layer.conv_state, masks):
                    ttnn.where(mask, buf, old, output_tensor=buf)
                    ttnn.deallocate(old)
            ttnn.deallocate(x)
            x = nxt
        flat = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        flat = ttnn.reshape(flat, [1, batch, self.dim])
        logits = self.terminal(flat)
        if advance_positions:
            ttnn.plus_one(current_pos, skip_negative_entries=True)
            ttnn.plus_one(rot_idxs)
        return logits
