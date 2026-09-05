# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Four-chip Blackhole TP decoder; optimized decoder is the local compute baseline."""

import copy
import math
from dataclasses import dataclass, field

import ttnn
from models.demos.gpt_oss.tt.ccl import CCLManager

from .functional_decoder import PREFILL_ALIGN, FunctionalDecoder, _pad_dim
from .fused_decoder import _align_up, _field, _slice_owned
from .optimized_decoder import DecoderConfig, OptimizedDecoder

TP = 4


class MeshCCLManager(CCLManager):
    """Cover every core the Blackhole CCL worker planner can select."""

    def _init_subdevice(self):
        grid = self.mesh_device.compute_with_storage_grid_size()
        self.ccl_cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
        )
        self.ccl_sub_device_id = ttnn.SubDeviceId(0)


@dataclass(frozen=True)
class MeshConfig:
    pack_gdn: bool = True
    pack_mlp_decode: bool = True
    decode_dram_roles: tuple = ("gate_up", "down_proj")
    decode_grid: tuple | None = (8, 4)
    decode_qkvg_dtype: str | None = "bfloat8_b"
    decode_qkvg_dram: bool = True
    residual: str = "replicated"
    links: int = 2
    collective: str = "native"
    # Full-grid semaphores support both link counts; one link won the async
    # whole-layer comparison. Native all_reduce has its own measured policy.
    async_links: int = 1
    local: DecoderConfig = field(
        default_factory=lambda: DecoderConfig(
            cores=8,
            block_w=4,
            readers=1,
            # BFP8 QKVG preserves the real-input changed-trace PCC gate.
            # Packed gate/up and separate down use independently tuned readers.
            role_configs={
                "qkvg": {"cores": 32, "block_w": 4, "readers": 1},
                "gdn_all": {"cores": 4, "block_w": 8, "readers": 1},
                "gdn_packed": {"cores": 4, "block_w": 32, "readers": 1},
                "gdn_z_epilogue": {"cores": 4, "block_w": 32, "readers": 1},
                "gdn_out": {"cores": 4, "block_w": 8, "readers": 1},
                "o_proj": {"cores": 4, "block_w": 8, "readers": 1},
                "gate_proj": {"cores": 8, "block_w": 8, "readers": 2},
                "up_proj": {"cores": 8, "block_w": 8, "readers": 2},
                "gate_up": {"cores": 32, "block_w": 4, "readers": 3},
                "down_proj": {"cores": 8, "block_w": 6, "readers": 2},
            },
            conv_chunk=512,
        )
    )


def fabric_router_config():
    """Target-ring packet policy; pass to set_fabric_config before opening mesh."""
    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = 8192
    return router


def partition_state_dict(state, cfg, rank):
    """Setup-only exact head/channel partition in HF orientation."""
    import torch

    result = dict(state)
    for role in ("gate_proj", "up_proj"):
        result[f"mlp.{role}.weight"] = state[f"mlp.{role}.weight"].chunk(TP, dim=0)[rank]
    result["mlp.down_proj.weight"] = state["mlp.down_proj.weight"].chunk(TP, dim=1)[rank]
    if "self_attn.q_proj.weight" in state:
        for role in ("q_proj", "k_proj", "v_proj"):
            result[f"self_attn.{role}.weight"] = state[f"self_attn.{role}.weight"].chunk(TP, dim=0)[rank]
        result["self_attn.o_proj.weight"] = state["self_attn.o_proj.weight"].chunk(TP, dim=1)[rank]
    else:
        widths = (cfg.linear_num_key_heads * cfg.linear_key_head_dim,) * 2 + (
            cfg.linear_num_value_heads * cfg.linear_value_head_dim,
        )
        for name in ("in_proj_qkv.weight", "conv1d.weight"):
            parts = state[f"linear_attn.{name}"].split(widths, dim=0)
            result[f"linear_attn.{name}"] = torch.cat([v.chunk(TP, dim=0)[rank] for v in parts])
        for name in ("in_proj_z.weight", "in_proj_a.weight", "in_proj_b.weight", "A_log", "dt_bias"):
            result[f"linear_attn.{name}"] = state[f"linear_attn.{name}"].chunk(TP, dim=0)[rank]
        result["linear_attn.out_proj.weight"] = state["linear_attn.out_proj.weight"].chunk(TP, dim=1)[rank]
    return result


def _projection_weights(state, cfg):
    import torch

    result = {role: state[f"mlp.{role}.weight"].T for role in ("gate_proj", "up_proj", "down_proj")}
    if "self_attn.q_proj.weight" in state:
        qg = state["self_attn.q_proj.weight"].reshape(cfg.num_attention_heads, 2, cfg.head_dim, cfg.hidden_size)
        result["qkvg"] = torch.cat(
            [
                qg[:, 0].reshape(-1, cfg.hidden_size),
                state["self_attn.k_proj.weight"],
                state["self_attn.v_proj.weight"],
                qg[:, 1].reshape(-1, cfg.hidden_size),
            ]
        ).T
        result["o_proj"] = state["self_attn.o_proj.weight"].T
    else:
        # Tile-align A and B independently: slicing B must never start inside A's tile.
        fields = [state["linear_attn.in_proj_qkv.weight"]]
        for name in ("a", "b"):
            value = state[f"linear_attn.in_proj_{name}.weight"]
            fields.append(torch.nn.functional.pad(value, (0, 0, 0, 32 - value.shape[0])))
        result["gdn_packed"] = torch.cat(fields).T
        result["gdn_z_epilogue"] = state["linear_attn.in_proj_z.weight"].T
        result["gdn_out"] = state["linear_attn.out_proj.weight"].T
    return result


class MultichipDecoder(OptimizedDecoder):
    """TP4 weights, head-local state, and two residual reductions per layer."""

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, mesh_device, mesh_config=None, **kwargs):
        import torch

        plan = mesh_config or MeshConfig()
        if tuple(mesh_device.shape) != (1, TP):
            raise ValueError("Ornith multichip decoder requires the 1x4 Blackhole ring")
        if plan.residual not in ("replicated", "sharded"):
            raise ValueError("residual must be replicated or sharded")
        original = getattr(hf_config, "text_config", None) or hf_config
        local_cfg = copy.deepcopy(original)
        for attr in (
            "num_attention_heads",
            "num_key_value_heads",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "intermediate_size",
        ):
            value = getattr(original, attr)
            if value % TP:
                raise ValueError(f"{attr} must divide TP4")
            setattr(local_cfg, attr, value // TP)
        states = [partition_state_dict(state_dict, original, rank) for rank in range(TP)]
        # Reuse optimized setup for constants and op policies at local head/channel sizes.
        # Its provisional rank-zero weights are replaced below before any forward.
        decoder = super().from_state_dict(
            states[0],
            hf_config=local_cfg,
            mesh_device=mesh_device,
            config=plan.local,
            **kwargs,
        )
        decoder.mesh_config = plan
        decoder.global_hf_config = original
        decoder.residual_width = original.hidden_size // TP if plan.residual == "sharded" else original.hidden_size
        decoder.ccl = MeshCCLManager(mesh_device, plan.async_links)
        packed = [_projection_weights(s, local_cfg) for s in states]
        if plan.pack_gdn and not decoder.is_full_attention:
            for weights in packed:
                weights["gdn_all"] = torch.cat([weights.pop("gdn_packed"), weights.pop("gdn_z_epilogue")], dim=1)
            decoder.projection_compute["gdn_all"] = decoder.projection_compute["gdn_packed"]

        def upload(parts, axis, dtype, memory=ttnn.DRAM_MEMORY_CONFIG):
            return ttnn.from_torch(
                torch.cat(parts, dim=axis).contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=memory,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=axis),
            )

        dram = mesh_device.dram_grid_size()
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))])
        for role in packed[0]:
            axis = 0 if role in ("o_proj", "gdn_out", "down_proj") else 1
            parts = [p[role] for p in packed]
            old = decoder.w.get(role)
            dtype = old.dtype if old is not None else decoder.w["gdn_packed"].dtype
            decoder.w[role] = upload(parts, axis, dtype)
            if old is not None:
                ttnn.deallocate(old)
            k, n = parts[0].shape
            readers = decoder._role_config(role)[2]
            width = math.ceil(n / (32 * dram.x * dram.y * readers)) * 32 * readers
            memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(grid, [k, width], ttnn.ShardOrientation.ROW_MAJOR),
            )
            old = decoder.decode_weights.pop(role, None)
            decode_dtype = getattr(ttnn, plan.decode_qkvg_dtype) if role == "qkvg" and plan.decode_qkvg_dtype else dtype
            dram_decode = (
                plan.decode_grid is None or role in plan.decode_dram_roles or (role == "qkvg" and plan.decode_qkvg_dram)
            )
            if dram_decode or decode_dtype != dtype:
                decoder.decode_weights[role] = upload(
                    parts,
                    axis,
                    decode_dtype,
                    memory if dram_decode else ttnn.DRAM_MEMORY_CONFIG,
                )
            if old is not None:
                ttnn.deallocate(old)
        if plan.pack_mlp_decode:
            parts = [torch.cat([p["gate_proj"], p["up_proj"]], dim=1) for p in packed]
            k, n = parts[0].shape
            readers = decoder._role_config("gate_up")[2]
            width = math.ceil(n / (32 * dram.x * dram.y * readers)) * 32 * readers
            memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(grid, [k, width], ttnn.ShardOrientation.ROW_MAJOR),
            )
            weight = upload(parts, 1, decoder.w["gate_proj"].dtype, memory)
            # Both dictionaries reference one allocation. Prefill keeps separate
            # gate/up weights; the packed allocation is used only for decode.
            decoder.w["gate_up"] = decoder.decode_weights["gate_up"] = weight
            decoder.projection_compute["gate_up"] = decoder.projection_compute["gate_proj"]
            for role in ("gate_proj", "up_proj"):
                old = decoder.decode_weights.pop(role, None)
                if old is not None:
                    ttnn.deallocate(old)
        if plan.pack_gdn and not decoder.is_full_attention:
            for role in ("gdn_packed", "gdn_z_epilogue"):
                ttnn.deallocate(decoder.w.pop(role))
                ttnn.deallocate(decoder.decode_weights.pop(role))
                decoder.projection_compute.pop(role)
        if not decoder.is_full_attention:
            for index in range(4):
                parts = [s["linear_attn.conv1d.weight"][:, 0, index].reshape(1, 1, -1) for s in states]
                old = decoder.w["conv_taps"][index]
                decoder.w["conv_taps"][index] = upload(parts, -1, old.dtype)
                ttnn.deallocate(old)
            for role, source in (("A_neg", "A_log"), ("dt_bias", "dt_bias")):
                parts = [s[f"linear_attn.{source}"].float() for s in states]
                if role == "A_neg":
                    parts = [-p.exp() for p in parts]
                # Pad each local vector before mesh splitting, then retain its logical head width.
                padded = [torch.nn.functional.pad(p, (0, 32 - p.numel())).reshape(1, 1, -1) for p in parts]
                value = upload(padded, -1, ttnn.float32)
                old = decoder.w[role]
                decoder.w[role] = ttnn.slice(value, [0, 0, 0], [1, 1, local_cfg.linear_num_value_heads])
                ttnn.deallocate(value)
                ttnn.deallocate(old)
        decoder.distributed_norms = {}
        if plan.residual == "sharded":
            for name, source in (
                ("attn_norm", "input_layernorm.weight"),
                ("ff_norm", "post_attention_layernorm.weight"),
            ):
                parts = [(part.float() + 1).reshape(1, 1, 1, -1) for part in state_dict[source].chunk(TP)]
                decoder.distributed_norms[name] = upload(parts, -1, ttnn.bfloat16)
        return decoder

    def _gdn_project(self, x):
        cfg = self.cfg
        if self.mesh_config.pack_gdn:
            packed = self._linear(x, "gdn_all")
            qkv = _field(packed, 0, cfg.conv_dim)
            a = _field(packed, cfg.conv_dim, cfg.conv_dim + cfg.linear_num_value_heads)
            b = _field(packed, cfg.conv_dim + 32, cfg.conv_dim + 32 + cfg.linear_num_value_heads)
            z = _field(packed, cfg.conv_dim + 64, packed.shape[-1])
            ttnn.deallocate(packed)
            return qkv, z, a, b
        packed = self._linear(x, "gdn_packed")
        qkv = _field(packed, 0, cfg.conv_dim)
        a = _field(packed, cfg.conv_dim, cfg.conv_dim + cfg.linear_num_value_heads)
        b = _field(packed, cfg.conv_dim + 32, cfg.conv_dim + 32 + cfg.linear_num_value_heads)
        ttnn.deallocate(packed)
        kwargs = {"activation": "silu"} if x.shape[1] == 1 else {}
        z = self._linear(x, "gdn_z_epilogue", **kwargs)
        return qkv, z, a, b

    def _gdn_out_head_major(self, core, z, batch, seq):
        if not self.mesh_config.pack_gdn or seq > 1:
            return super()._gdn_out_head_major(core, z, batch, seq)
        normed = ttnn.rms_norm(core, weight=self.w["gdn_norm"], epsilon=self.cfg.norm_eps)
        ttnn.deallocate(core)
        heads = ttnn.reshape(normed, [batch, self.cfg.linear_num_value_heads, 1, self.cfg.linear_value_head_dim])
        combined = ttnn.permute(heads, (0, 2, 1, 3))
        ttnn.deallocate(normed)
        # Compact logical users into one tile row. Mixed BF16/FP32 BinaryNG
        # activation preprocessing must see BF16 on the left and one tile per
        # worker; separate padded batch planes violate that kernel contract.
        merged = ttnn.reshape(combined, [batch, 1, self.cfg.linear_v_dim])
        merged = ttnn.reshape(merged, [1, batch, self.cfg.linear_v_dim])
        gate_input = ttnn.reshape(z, [1, batch, self.cfg.linear_v_dim])
        gated = ttnn.multiply(
            gate_input, merged, dtype=ttnn.float32, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
        )
        gated = ttnn.reshape(gated, [batch, 1, self.cfg.linear_v_dim])
        ttnn.deallocate(combined)
        result = self._linear(gated, "gdn_out")
        ttnn.deallocate(gated)
        return result

    def _activate_mlp(self, x, mode):
        if self.mesh_config.pack_mlp_decode and mode == "decode":
            packed = self._linear(x, "gate_up")
            batch, _, width = packed.shape
            # Compact users before sharded slices: public [B,1,H] otherwise
            # has B separately padded tile rows rather than one shared row.
            if batch > 1:
                packed = ttnn.reshape(packed, [1, batch, width])
            rows = batch if batch > 1 else 1
            memory = self._width_memory(width // 2, self._role_config("down_proj")[0])
            gate = ttnn.slice(packed, [0, 0, 0], [1, rows, width // 2], memory_config=memory)
            up = ttnn.slice(packed, [0, 0, width // 2], [1, rows, width], memory_config=memory)
            result = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=memory)
            for value in (packed, gate, up):
                ttnn.deallocate(value)
            if batch > 1:
                result = ttnn.to_memory_config(result, ttnn.L1_MEMORY_CONFIG)
                result = ttnn.reshape(result, [batch, 1, width // 2])
            return result
        if not self.mesh_config.pack_gdn or mode != "decode" or "gate_proj" in self.mesh_config.decode_dram_roles:
            return super()._activate_mlp(x, mode)
        local = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        gate, up = self._linear(local, "gate_proj"), self._linear(local, "up_proj")
        result = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        if local.buffer_address() != x.buffer_address():
            ttnn.deallocate(local)
        return result

    def _linear(self, x, role, **kwargs):
        dram_role = (
            role in self.mesh_config.decode_dram_roles
            or (role == "gate_up" and self.mesh_config.pack_mlp_decode)
            or (role == "qkvg" and self.mesh_config.decode_qkvg_dram)
        )
        if x.shape[1] == 1 and self.mesh_config.decode_grid is not None and not dram_role:
            batch, _, k = x.shape
            n = self.w[role].shape[-1]
            grid = self.mesh_config.decode_grid
            local = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            local = ttnn.reshape(local, [1, 1, batch, k])
            weight = ttnn.reshape(self.decode_weights.get(role, self.w[role]), [1, 1, k, n])
            per_n = math.ceil(n / (32 * math.prod(grid)))
            program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=grid,
                in0_block_w=self._role_config(role)[1],
                out_subblock_h=1,
                out_subblock_w=max(v for v in range(1, 9) if per_n % v == 0),
                per_core_M=1,
                per_core_N=per_n,
                fuse_batch=True,
                mcast_in0=True,
            )
            kwargs.setdefault("dtype", ttnn.bfloat16)
            out = ttnn.linear(
                local,
                weight,
                program_config=program,
                compute_kernel_config=self.projection_compute[role],
                memory_config=ttnn.L1_MEMORY_CONFIG,
                **kwargs,
            )
            out = ttnn.reshape(out, [batch, 1, n])
        else:
            out = super()._linear(x, role, **kwargs)
        if role not in ("o_proj", "gdn_out", "down_proj"):
            return out
        shape = list(out.shape)
        local = ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG if shape[1] == 1 else ttnn.DRAM_MEMORY_CONFIG)
        folded = ttnn.reshape(local, [1, 1, shape[0] * shape[1], shape[2]])
        if self.mesh_config.residual == "replicated" and self.mesh_config.collective == "native":
            reduced = ttnn.all_reduce(folded, num_links=self.mesh_config.links, topology=ttnn.Topology.Ring)
        else:
            reduced = ttnn.experimental.reduce_scatter_minimal_async(
                folded,
                dim=3,
                persistent_output_buffers=None,
                multi_device_global_semaphore=self.ccl.get_rs_ping_pong_semaphore(),
                barrier_semaphore=self.ccl.get_barrier_semaphore(),
                num_links=self.mesh_config.async_links,
                topology=ttnn.Topology.Ring,
                memory_config=folded.memory_config(),
                intermediate_memory_config=folded.memory_config(),
            )
            if self.mesh_config.residual == "replicated":
                reduced = self._gather(reduced)
        shape[-1] = self.residual_width
        return ttnn.reshape(reduced, shape)

    def _gather(self, tensor):
        return ttnn.experimental.all_gather_async(
            tensor,
            dim=3,
            persistent_output_buffer=None,
            multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            num_links=self.mesh_config.async_links,
            topology=ttnn.Topology.Ring,
            memory_config=tensor.memory_config(),
        )

    def _residual_norm(self, x, name):
        shape = list(x.shape)
        memory = ttnn.L1_MEMORY_CONFIG if shape[1] == 1 else ttnn.DRAM_MEMORY_CONFIG
        local = ttnn.to_memory_config(x, memory)
        local = ttnn.reshape(local, [1, 1, shape[0] * shape[1], shape[2]])
        stats = ttnn.rms_norm_pre_all_gather(
            local, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_kernel_config
        )
        gathered_stats = self._gather(stats)
        normalized = ttnn.rms_norm_post_all_gather(
            local,
            gathered_stats,
            epsilon=self.cfg.norm_eps,
            weight=self.distributed_norms[name],
            compute_kernel_config=self.compute_kernel_config,
        )
        gathered = self._gather(normalized)
        ttnn.deallocate(stats)
        ttnn.deallocate(gathered_stats)
        ttnn.deallocate(normalized)
        return ttnn.reshape(gathered, [shape[0], shape[1], self.cfg.dim])

    def _block(self, x, *, mode, logical_len=None, page_table=None, chunk_start_idx=0, current_pos=None, rot_idxs=None):
        if self.mesh_config.residual == "replicated":
            return super()._block(
                x,
                mode=mode,
                logical_len=logical_len,
                page_table=page_table,
                chunk_start_idx=chunk_start_idx,
                current_pos=current_pos,
                rot_idxs=rot_idxs,
            )
        memory = self._width_memory(self.residual_width, 32) if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
        attn_in = self._residual_norm(x, "attn_norm")
        if self.is_full_attention:
            mixed = (
                self._attention_prefill(attn_in, page_table, chunk_start_idx)
                if mode == "prefill"
                else self._attention_decode(attn_in, current_pos, rot_idxs, page_table)
            )
        else:
            mixed = self._gdn_prefill(attn_in, logical_len) if mode == "prefill" else self._gdn_decode(attn_in)
        ttnn.deallocate(attn_in)
        h = self._residual_add(x, mixed, memory)
        ttnn.deallocate(mixed)
        ff_in = self._residual_norm(h, "ff_norm")
        activated = self._activate_mlp(ff_in, mode)
        ttnn.deallocate(ff_in)
        ff_out = self._linear(activated, "down_proj")
        ttnn.deallocate(activated)
        out = self._residual_add(h, ff_out, memory)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_out)
        return out

    def prefill_forward(self, x, *, start_pos: int = 0, page_table=None, chunk_size: int | None = None):
        """Preserve logical continuation semantics while collecting outputs in DRAM."""
        b, seq_len, dim = x.shape[0], x.shape[1], x.shape[2]
        if dim != self.residual_width:
            raise ValueError(f"hidden size {dim} != {self.residual_width}")
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if start_pos < 0 or start_pos + seq_len > self.max_context:
            raise ValueError(
                f"prefill window [{start_pos}, {start_pos + seq_len}) exceeds supported context {self.max_context}"
            )
        chunk_size = self.prefill_chunk if chunk_size is None else chunk_size
        if chunk_size < PREFILL_ALIGN or chunk_size % PREFILL_ALIGN:
            raise ValueError(f"chunk_size {chunk_size} must be a multiple of {PREFILL_ALIGN}")
        if not self.is_full_attention and chunk_size > self.w["pos_ramp"].shape[1]:
            raise ValueError(
                f"chunk_size {chunk_size} exceeds the position ramp built for {self.w['pos_ramp'].shape[1]}"
            )
        if self.batch_size is None:
            raise RuntimeError("call allocate_state(batch_size) before forward or trace capture")
        if b != self.batch_size:
            raise ValueError(f"batch {b} != allocated state batch {self.batch_size}")

        outputs = []
        leading = min(seq_len, (-start_pos) % PREFILL_ALIGN) if self.is_full_attention else 0
        for offset in range(leading):
            block, owned = _slice_owned(x, [0, offset, 0], [b, offset + 1, dim])
            position = start_pos + offset
            pos = ttnn.reshape(ttnn.slice(self.prefill_positions, [position, 0], [position + 1, b]), [b])
            rot = ttnn.slice(self.prefill_rot_idxs, [position, 0], [position + 1, b])
            out = self._block(block, mode="decode", current_pos=pos, rot_idxs=rot, page_table=page_table)
            # Up to 127 leading tokens use decode. Retaining their optimized
            # L1 residual shards exhausts the working space of later blocks.
            collected = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
            if collected.buffer_address() != out.buffer_address():
                ttnn.deallocate(out)
            outputs.append(collected)
            if owned:
                ttnn.deallocate(block)
            ttnn.deallocate(pos)
            ttnn.deallocate(rot)
        for offset in range(leading, seq_len, chunk_size):
            logical = min(chunk_size, seq_len - offset)
            phys = min(chunk_size, _align_up(logical, PREFILL_ALIGN))
            block, owned = _slice_owned(x, [0, offset, 0], [b, offset + logical, dim])
            if phys > logical:
                # Padding can alias the slice; retain its original ownership.
                block = _pad_dim(block, 1, phys - logical)
            out = self._block(
                block,
                mode="prefill",
                logical_len=logical,
                page_table=page_table,
                chunk_start_idx=start_pos + offset,
            )
            if owned:
                ttnn.deallocate(block)
            if phys > logical:
                trimmed = ttnn.slice(out, [0, 0, 0], [b, logical, dim])
                ttnn.deallocate(out)
                out = trimmed
            outputs.append(out)

        if len(outputs) == 1:
            return outputs[0]
        merged = ttnn.concat(outputs, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for out in outputs:
            ttnn.deallocate(out)
        return merged

    def decode_forward(self, x, *, current_pos=None, rot_idxs=None, page_table=None):
        """See the module docstring for the full contract."""
        b, t, dim = x.shape[0], x.shape[1], x.shape[2]
        if t != 1:
            raise ValueError(f"decode expects seq_len 1, got {t}")
        if dim != self.residual_width:
            raise ValueError(f"hidden size {dim} != {self.residual_width}")
        if self.is_full_attention and (current_pos is None or rot_idxs is None):
            raise ValueError("full_attention decode requires current_pos and rot_idxs device tensors")
        if self.batch_size is None:
            raise RuntimeError("call allocate_state(batch_size) before forward or trace capture")
        elif b != self.batch_size:
            raise ValueError(f"batch {b} != allocated state batch {self.batch_size}")
        return self._block(x, mode="decode", current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table)

    def allocate_state(self, batch_size):
        """Budget persistent L1 state for every linear layer in the future stack."""
        FunctionalDecoder.allocate_state(self, batch_size)
        if self.is_full_attention:
            return
        grid = self.device.compute_with_storage_grid_size()
        state_tiles = math.prod(self.recurrent_state.padded_shape) // 1024
        layer_bytes_per_bank = math.ceil(state_tiles / (grid.x * grid.y)) * 4096
        layers = self.cfg.layer_types.count("linear_attention")
        self.recurrent_l1_intermediates = self.requested_l1_intermediates and layer_bytes_per_bank <= 224 * 1024
        self.stack_state_bytes_per_bank = layers * layer_bytes_per_bank
        if self.stack_state_bytes_per_bank <= 300 * 1024:
            old = self.recurrent_state
            self.recurrent_state = ttnn.to_memory_config(old, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(old)
