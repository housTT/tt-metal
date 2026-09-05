# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Ornith decoder optimization with unchanged fused cache and sequence semantics."""

import math
from dataclasses import dataclass, field

import ttnn

from .functional_decoder import PREFILL_ALIGN, FunctionalDecoder, _pad_dim
from .fused_decoder import (
    FusedDecoder,
    _align_up,
    _batch_grid,
    _field,
    _height_memory,
    _rectangular_rope_grid,
    _slice_owned,
)


@dataclass(frozen=True)
class PrecisionPolicy:
    attention: str = "bfloat4_b"
    mlp_gate_up: str = "bfloat4_b"
    mlp_down: str = "bfloat4_b"
    attention_fidelity: str = "LoFi"
    mlp_fidelity: str = "LoFi"


@dataclass(frozen=True)
class DecoderConfig:
    residual_cores: int = 32
    dram_roles: tuple = (
        "gdn_packed",
        "gdn_z_epilogue",
        "gdn_out",
        "gate_proj",
        "up_proj",
        "down_proj",
        "qkvg",
        "o_proj",
    )
    cores: int = 8
    block_w: int = 8
    readers: int = 1
    role_configs: dict = field(
        default_factory=lambda: {
            "gdn_packed": {"cores": 32, "block_w": 4, "readers": 3},
            "gdn_z_epilogue": {"cores": 8, "block_w": 8, "readers": 2},
            "gdn_out": {"cores": 8, "block_w": 8, "readers": 2},
            "gate_proj": {"cores": 64, "block_w": 2, "readers": 3},
            "up_proj": {"cores": 64, "block_w": 2, "readers": 3},
            "down_proj": {"cores": 48, "block_w": 8, "readers": 2},
            "qkvg": {"cores": 32, "block_w": 2, "readers": 3},
            "o_proj": {"cores": 8, "block_w": 8, "readers": 2},
        }
    )
    sdpa_grid: tuple = (8, 8)
    sdpa_chunk: int = 256
    conv_chunk: int = 1024
    host_weight_packing: bool = True
    large_prefill_grid: tuple = (11, 10)
    large_prefill_block_w: int = 16
    large_prefill_min_seq: int = 2048
    large_prefill_subblock: tuple = (1, 7)


class OptimizedDecoder(FusedDecoder):
    """Per-role projections; setup-only precision materialization."""

    @classmethod
    def from_state_dict(cls, state_dict, *, policy=None, config=None, **kwargs):
        decoder = super().from_state_dict(state_dict, **kwargs)
        decoder.policy = policy or PrecisionPolicy()
        decoder.optimization = config or DecoderConfig()
        # The tuned separate gate/up projections beat the packed prefill path.
        ttnn.deallocate(decoder.w.pop("gate_up"))
        decoder.decode_weights = {}
        decoder.projection_compute = {}
        for name in ("qkvg", "o_proj", "gdn_packed", "gdn_z_epilogue", "gdn_out", "gate_proj", "up_proj", "down_proj"):
            if name not in decoder.w:
                continue
            mlp = name in ("gate_proj", "up_proj", "down_proj")
            group = "mlp_down" if name == "down_proj" else "mlp_gate_up" if mlp else "attention"
            dtype = getattr(ttnn, getattr(decoder.policy, group))
            if not decoder.optimization.host_weight_packing and decoder.w[name].dtype != dtype:
                old = decoder.w[name]
                decoder.w[name] = ttnn.typecast(old, dtype)
                ttnn.deallocate(old)
            fidelity = decoder.policy.mlp_fidelity if mlp else decoder.policy.attention_fidelity
            decoder.projection_compute[name] = ttnn.init_device_compute_kernel_config(
                decoder.device.arch(),
                math_fidelity=getattr(ttnn.MathFidelity, fidelity),
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
        import torch

        host_weights = {name: state_dict[f"mlp.{name}.weight"].T for name in ("gate_proj", "up_proj", "down_proj")}
        if decoder.is_full_attention:
            cfg = decoder.cfg
            qg = state_dict["self_attn.q_proj.weight"].reshape(cfg.n_heads, 2, cfg.head_dim, cfg.dim)
            host_weights["qkvg"] = torch.cat(
                [
                    qg[:, 0].reshape(-1, cfg.dim),
                    state_dict["self_attn.k_proj.weight"],
                    state_dict["self_attn.v_proj.weight"],
                    qg[:, 1].reshape(-1, cfg.dim),
                ]
            ).T
            host_weights["o_proj"] = state_dict["self_attn.o_proj.weight"].T
        else:
            host_weights["gdn_packed"] = torch.cat(
                [state_dict[f"linear_attn.in_proj_{n}.weight"] for n in ("qkv", "a", "b")]
            ).T
            host_weights["gdn_z_epilogue"] = state_dict["linear_attn.in_proj_z.weight"].T
            host_weights["gdn_out"] = state_dict["linear_attn.out_proj.weight"].T
        if decoder.optimization.host_weight_packing:
            for role, value in host_weights.items():
                old = decoder.w[role]
                group = (
                    "mlp_down"
                    if role == "down_proj"
                    else "mlp_gate_up"
                    if role in ("gate_proj", "up_proj", "gate_up")
                    else "attention"
                )
                decoder.w[role] = ttnn.from_torch(
                    value.contiguous(),
                    device=decoder.device,
                    dtype=getattr(ttnn, getattr(decoder.policy, group)),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(decoder.device),
                )
                ttnn.deallocate(old)
        dg = decoder.device.dram_grid_size()
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dg.x - 1, dg.y - 1))])
        for role in decoder.optimization.dram_roles:
            if role not in host_weights:
                continue
            k, n = host_weights[role].shape
            readers = decoder._role_config(role)[2]
            width = math.ceil(n / (32 * dg.x * dg.y * readers)) * 32 * readers
            mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(grid, [k, width], ttnn.ShardOrientation.ROW_MAJOR),
            )
            decoder.decode_weights[role] = ttnn.from_torch(
                host_weights[role].contiguous(),
                device=decoder.device,
                dtype=decoder.w[role].dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=ttnn.ReplicateTensorToMesh(decoder.device),
            )
        decoder.state_read_program = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=4,
        )
        decoder.state_compute = ttnn.init_device_compute_kernel_config(
            decoder.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        decoder.requested_l1_intermediates = True
        decoder.recurrent_l1_intermediates = True
        decoder.kda_conv_program = ttnn.QkvCausalConv1dSiluProgramConfig(
            channel_chunk_size=decoder.optimization.conv_chunk
        )
        decoder.decode_sdpa_compute = ttnn.init_device_compute_kernel_config(
            decoder.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        return decoder

    def _role_config(self, role):
        c = self.optimization
        r = c.role_configs.get(role, {})
        return r.get("cores", c.cores), r.get("block_w", c.block_w), r.get("readers", c.readers)

    def _width_memory(self, width, cores, rows=32):
        size = self.device.compute_with_storage_grid_size()
        rectangles = [x for x in range(1, size.x + 1) if cores % x == 0 and cores // x <= size.y]
        if not rectangles:
            grid = ttnn.num_cores_to_corerangeset(cores, size, row_wise=True)
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, [rows, math.ceil(width / (cores * 32)) * 32], ttnn.ShardOrientation.ROW_MAJOR),
            )
        cols = max(rectangles)
        grid = ttnn.CoreGrid(x=cols, y=cores // cols)
        return ttnn.create_sharded_memory_config(
            (rows, math.ceil(width / (cores * 32)) * 32),
            core_grid=grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _norm(self, x, weight):
        if x.is_sharded() and x.shape[-1] == self.cfg.head_dim:
            batch = int(x.shape[1])
            cores = 8
            mem = self._width_memory(self.cfg.head_dim, cores, rows=batch * 32)
            x = ttnn.to_memory_config(x, mem)
            program = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(cores, 1),
                block_h=batch,
                block_w=self.cfg.head_dim // 32 // cores,
                subblock_w=1,
                inplace=False,
            )
            return ttnn.rms_norm(x, weight=weight, epsilon=self.cfg.norm_eps, program_config=program, memory_config=mem)
        if self.optimization.residual_cores and x.shape[-1] == self.cfg.dim and x.shape[-2] == 1:
            cores = self.optimization.residual_cores
            public_shape = list(x.shape)
            if public_shape[0] > 1:
                # One tile row can hold every supported decode user. Keep the
                # public [B,1,H] layout only at the mixer/linear boundaries.
                x = ttnn.reshape(x, [1, public_shape[0], self.cfg.dim])
            rows = math.prod(list(x.padded_shape)[:-1])
            mem = self._width_memory(self.cfg.dim, cores, rows)
            size = self.device.compute_with_storage_grid_size()
            cols = max(c for c in range(1, size.x + 1) if cores % c == 0 and cores // c <= size.y)
            block = self.cfg.dim // 32 // cores
            config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(cols, cores // cols),
                block_h=rows // 32,
                block_w=block,
                subblock_w=max(i for i in range(1, 9) if block % i == 0),
                inplace=False,
            )
            x = ttnn.to_memory_config(x, mem)
            out = ttnn.rms_norm(x, weight=weight, epsilon=self.cfg.norm_eps, program_config=config, memory_config=mem)
            if public_shape[0] > 1:
                out = ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)
                out = ttnn.reshape(out, public_shape)
            return out
        return super()._norm(x, weight)

    def _linear(self, x, role, **kwargs):
        if role == "gdn_out":
            kwargs.setdefault("dtype", ttnn.bfloat16)
        if x.shape[1] != 1:
            return self._prefill_linear(x, role, **kwargs)
        if x.shape[1] == 1 and role in self.decode_weights:
            batch, _, k = x.shape
            cores, block_w, readers = self._role_config(role)
            n = self.w[role].shape[-1]
            folded = ttnn.reshape(x, [1, batch, k])
            mem = self._width_memory(k, cores)
            working = ttnn.to_memory_config(folded, mem)
            activation = kwargs.pop("activation", None)
            kwargs.pop("core_grid", None)
            program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=block_w,
                per_core_M=1,
                per_core_N=self.optimization.role_configs.get(role, {}).get("per_core_n", math.ceil(n / (32 * cores))),
                num_workers_per_dram_bank=readers,
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if activation == "silu" else None,
            )
            out = ttnn.linear(
                working,
                self.decode_weights[role],
                program_config=program,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                compute_kernel_config=self.projection_compute[role],
                **kwargs,
            )
            # Mixer helpers use head-shaped tensors. MLP consumes the shared width shards directly.
            if role not in ("gate_proj", "up_proj", "down_proj", "gate_up", "gdn_out", "o_proj") or batch > 1:
                out = ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)
            return ttnn.reshape(out, [batch, 1, n])
        if x.is_sharded():
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        return ttnn.linear(
            x,
            self.w[role],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.projection_compute[role],
            **kwargs,
        )

    def _residual_add(self, residual, update, memory_config):
        public_shape = list(residual.shape)
        folded = public_shape[0] > 1 and public_shape[-2] == 1 and memory_config.is_sharded()
        if folded:
            residual = ttnn.to_memory_config(
                ttnn.reshape(residual, [1, public_shape[0], public_shape[-1]]), memory_config
            )
            update = ttnn.to_memory_config(ttnn.reshape(update, [1, public_shape[0], public_shape[-1]]), memory_config)
        out = self._residual_sum(residual, update, memory_config)
        if folded:
            out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
            out = ttnn.reshape(out, public_shape)
        return out

    def _residual_sum(self, residual, update, memory_config):
        if residual.is_sharded() and residual.dtype != update.dtype:
            # Mixed BF16/FP32 width-sharded add is nondeterministic. Preserve the
            # FP32 update through the sum, then round once to the residual dtype.
            promoted = ttnn.typecast(residual, ttnn.float32)
            if update.dtype != ttnn.float32:
                update = ttnn.typecast(update, ttnn.float32)
            summed = ttnn.add(promoted, update, memory_config=memory_config)
            return ttnn.typecast(summed, residual.dtype)
        return ttnn.add(residual, update, memory_config=memory_config)

    def prefill_forward(self, x, *, start_pos: int = 0, page_table=None, chunk_size: int | None = None):
        """Preserve logical continuation semantics while collecting outputs in DRAM."""
        b, seq_len, dim = x.shape[0], x.shape[1], x.shape[2]
        if dim != self.cfg.dim:
            raise ValueError(f"hidden size {dim} != {self.cfg.dim}")
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

    def _block(self, x, *, mode, logical_len=None, page_table=None, chunk_start_idx=0, current_pos=None, rot_idxs=None):
        residual_mem = (
            self._width_memory(self.cfg.dim, self.optimization.residual_cores, 32)
            if mode == "decode" and self.optimization.residual_cores
            else ttnn.DRAM_MEMORY_CONFIG
        )
        compact_residual = mode == "decode" and x.shape[0] > 1 and self.optimization.residual_cores
        if mode == "decode" and not compact_residual:
            x = ttnn.to_memory_config(x, residual_mem)
        attn_in = self._norm(x, self.w["attn_norm"])
        if self.is_full_attention:
            if mode == "prefill":
                mixed = self._attention_prefill(attn_in, page_table, chunk_start_idx)
            else:
                mixed = self._attention_decode(attn_in, current_pos, rot_idxs, page_table)
        elif mode == "prefill":
            mixed = self._gdn_prefill(attn_in, logical_len)
        else:
            mixed = self._gdn_decode(attn_in)
        ttnn.deallocate(attn_in)
        if not compact_residual:
            mixed = ttnn.to_memory_config(mixed, residual_mem)
        h = self._residual_add(x, mixed, residual_mem)
        ttnn.deallocate(mixed)
        ff_in = self._norm(h, self.w["ff_norm"])
        activated = self._activate_mlp(ff_in, mode)
        ttnn.deallocate(ff_in)
        ff_out = self._linear(activated, "down_proj")
        ttnn.deallocate(activated)
        if not compact_residual:
            ff_out = ttnn.to_memory_config(ff_out, residual_mem)
        out = self._residual_add(h, ff_out, residual_mem)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_out)
        return out

    def _activate_mlp(self, ff_in, mode):
        if mode == "decode" and ff_in.shape[0] == 1:
            ff_in = ttnn.to_memory_config(ff_in, self._width_memory(self.cfg.dim, self._role_config("gate_proj")[0]))
        gate = self._linear(ff_in, "gate_proj")
        up = self._linear(ff_in, "up_proj")
        result = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        return result

    def _project_qkv(self, x):
        batch, seq, _ = list(x.shape)
        cfg = self.cfg
        qw = cfg.n_heads * cfg.head_dim
        end = qw + 2 * cfg.n_kv_heads * cfg.head_dim
        packed = self._linear(x, "qkvg")
        qkv = ttnn.slice(packed, [0, 0, 0], [batch, seq, end])
        gate = ttnn.slice(packed, [0, 0, end], [batch, seq, end + qw])
        ttnn.deallocate(packed)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=cfg.n_heads, num_kv_heads=cfg.n_kv_heads, transpose_key=False
        )
        ttnn.deallocate(qkv)
        q_norm, k_norm = self._norm(q, self.w["q_norm"]), self._norm(k, self.w["k_norm"])
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        return q_norm, k_norm, v, gate

    def _attention_output(self, attn, gate):
        gated = ttnn.multiply(gate, attn, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(attn)
        ttnn.deallocate(gate)
        result = self._linear(gated, "o_proj")
        ttnn.deallocate(gated)
        return result

    def _decode_rotary(self, q, k, rot_idxs, batch):
        grid, users_per_core = _rectangular_rope_grid(self.device, batch)
        if users_per_core == 1:
            return super()._decode_rotary(q, k, rot_idxs, batch)
        # Prime batches share one RoPE core. Release each owned partial before
        # constructing K, and retain completed outputs in interleaved DRAM.
        rd, dim = self.cfg.rope_dim, self.cfg.head_dim
        rotated_mem = _height_memory(grid, rd, users_per_core)
        cos, sin = self.rope.decode_forward(rot_idxs)
        cos, sin = [
            ttnn.transpose(ttnn.reshape(a, [1, 1, batch, rd]), 1, 2, memory_config=rotated_mem) for a in (cos, sin)
        ]
        outputs = []
        for value in (q, k):
            heads = int(value.shape[2])
            partial = ttnn.slice(value, [0, 0, 0, 0], [1, batch, heads, rd], memory_config=rotated_mem)
            rotated = ttnn.experimental.rotary_embedding_hf(
                partial,
                cos,
                sin,
                is_decode_mode=True,
                memory_config=rotated_mem,
                compute_kernel_config=self.compute_kernel_config,
            )
            ttnn.deallocate(partial)
            rotated_dram = ttnn.to_memory_config(rotated, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(rotated)
            # The untouched tail has no rotary-core requirement. Slice it
            # directly into DRAM; B31's otherwise-single-core tail is 372 KiB.
            tail = ttnn.slice(value, [0, 0, 0, rd], [1, batch, heads, dim], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            outputs.append(ttnn.concat([rotated_dram, tail], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            ttnn.deallocate(rotated_dram)
            ttnn.deallocate(tail)
        return tuple(outputs)

    def _sdpa_query(self, q, batch, program_grid):
        if not q.is_sharded():
            return q
        # SDPA's sharded reader addresses the first B cores of its own grid;
        # native RoPE may have produced a differently shaped rectangle.
        grid = ttnn.num_cores_to_corerangeset(batch, program_grid, row_wise=True)
        memory = _height_memory(grid, self.cfg.head_dim)
        return q if q.memory_config() == memory else ttnn.to_memory_config(q, memory)

    def _attention_decode(self, x, current_pos, rot_idxs, page_table):
        if self.k_cache is None or page_table is None:
            raise ValueError("full_attention requires allocated cache and page table")
        batch = int(x.shape[0])
        cfg = self.cfg
        qw = cfg.n_heads * cfg.head_dim
        kvw = 2 * cfg.n_kv_heads * cfg.head_dim
        packed = self._linear(x, "qkvg")
        packed = ttnn.reshape(packed, [1, 1, batch, 2 * qw + kvw])
        qkv = ttnn.slice(packed, [0, 0, 0, 0], [1, 1, batch, qw + kvw], memory_config=ttnn.L1_MEMORY_CONFIG)
        gate = ttnn.slice(packed, [0, 0, 0, qw + kvw], [1, 1, batch, 2 * qw + kvw])
        gate = ttnn.reshape(gate, [batch, 1, qw])
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv,
            num_heads=cfg.n_heads,
            num_kv_heads=cfg.n_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        q = self._norm(q, self.w["q_norm"])
        k = self._norm(k, self.w["k_norm"])
        q, k = self._decode_rotary(q, k, rot_idxs, batch)
        # V retains the head-split shard on the first B cores. K must be disjoint.
        kmem = _height_memory(_batch_grid(self.device, batch, offset=batch), cfg.head_dim)
        k = ttnn.to_memory_config(k, kmem)
        ttnn.experimental.paged_fused_update_cache(
            self.k_cache,
            k,
            self.v_cache,
            v,
            update_idxs_tensor=current_pos,
            page_table=page_table,
        )
        sdpa_grid = ttnn.CoreCoord(*self.optimization.sdpa_grid)
        q = self._sdpa_query(q, batch, sdpa_grid)
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            self.k_cache,
            self.v_cache,
            cur_pos_tensor=current_pos,
            page_table_tensor=page_table,
            is_causal=True,
            scale=cfg.head_dim**-0.5,
            compute_kernel_config=self.decode_sdpa_compute,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=sdpa_grid,
                q_chunk_size=32,
                k_chunk_size=self.optimization.sdpa_chunk,
                exp_approx_mode=False,
            ),
        )
        return self._decode_finish(attn, gate, batch)

    def _gdn_project(self, x):
        cfg = self.cfg
        # QKV/A/B share a projection; Z has its own activation epilogue.
        packed = self._linear(x, "gdn_packed")
        qkv = _field(packed, 0, cfg.conv_dim)
        a = _field(packed, cfg.conv_dim, cfg.conv_dim + cfg.linear_num_value_heads)
        b = _field(packed, cfg.conv_dim + cfg.linear_num_value_heads, cfg.conv_dim + 2 * cfg.linear_num_value_heads)
        ttnn.deallocate(packed)
        # Both modes use the same full-grid Z projection geometry.
        grid = self.device.compute_with_storage_grid_size()
        z_args = {"core_grid": ttnn.CoreGrid(x=grid.x, y=grid.y)}
        # Public prefill physically pads every chunk to at least128 tokens.
        # Only decode reaches this boundary with time1; no runtime data is read.
        if int(x.shape[1]) == 1:
            z_args["activation"] = "silu"
        z = self._linear(x, "gdn_z_epilogue", **z_args)
        return qkv, z, a, b

    def _gdn_out_head_major(self, core, z, batch, seq):
        if seq > 1:
            normed = ttnn.experimental.kda.sigmoid_gated_rms_norm(
                core,
                z,
                self.w["kda_norm_vector"],
                self.cfg.linear_num_value_heads,
                epsilon=self.cfg.norm_eps,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
                output_dtype=ttnn.float32,
            )
            ttnn.deallocate(core)
            gated = ttnn.multiply(normed, z)
            ttnn.deallocate(normed)
        else:
            normed = ttnn.rms_norm(core, weight=self.w["gdn_norm"], epsilon=self.cfg.norm_eps)
            ttnn.deallocate(core)
            heads = ttnn.reshape(normed, [batch, self.cfg.linear_num_value_heads, 1, self.cfg.linear_value_head_dim])
            combined = ttnn.permute(heads, (0, 2, 1, 3))
            ttnn.deallocate(normed)
            merged = ttnn.reshape(combined, [batch, 1, self.cfg.linear_v_dim])
            # Z already contains its matmul SiLU epilogue in decode.
            gated = ttnn.multiply(merged, z)
            ttnn.deallocate(combined)
        result = self._linear(gated, "gdn_out")
        ttnn.deallocate(gated)
        return result

    def allocate_state(self, batch_size):
        FunctionalDecoder.allocate_state(self, batch_size)
        if not self.is_full_attention:
            # Validated with borrowed width-sharded public inputs: persistent
            # state fits through 300 KiB/bank (B16 on 110-bank P300c), but the
            # outer-product working set needs a 224 KiB/bank bound (B12).
            # B13 L1 outer collides with its CBs; B31 L1 state collides with
            # prefill RMSNorm. Select both policies before any trace capture.
            grid = self.device.compute_with_storage_grid_size()
            state_tiles = math.prod(self.recurrent_state.padded_shape) // 1024
            state_bytes_per_bank = math.ceil(state_tiles / (grid.x * grid.y)) * 4096
            fits_l1 = state_bytes_per_bank <= 300 * 1024
            self.recurrent_l1_intermediates = self.requested_l1_intermediates and state_bytes_per_bank <= 224 * 1024
            if fits_l1:
                old = self.recurrent_state
                self.recurrent_state = ttnn.to_memory_config(old, ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(old)

    def allocate_kv_cache(self, num_blocks, dtype=ttnn.bfloat8_b):
        return super().allocate_kv_cache(num_blocks, dtype=dtype)

    def _delta_rule_step(self, q, k, v, beta, g):
        cfg = self.cfg
        batch, nv, dk = q.shape[0], cfg.linear_num_value_heads, cfg.linear_key_head_dim
        dram = ttnn.L1_MEMORY_CONFIG if self.recurrent_l1_intermediates else ttnn.DRAM_MEMORY_CONFIG
        # q/k already contain the same BF16 RMSNorm outputs produced by the
        # separate per-head norms; scalar/cast rounding points remain intact.
        q_row = ttnn.multiply(q, dk**-1.0, dtype=ttnn.float32, memory_config=dram)
        # Preserve BinaryNG's BF16 scalar and BF16 product rounding before FP32 output.
        k_row = ttnn.unary_chain(k, self.key_scale_chain, memory_config=ttnn.L1_MEMORY_CONFIG)
        beta_view = ttnn.reshape(beta, [batch, nv, 1, 1])
        g_view = ttnn.reshape(g, [batch, nv, 1, 1])
        state = self.recurrent_state
        ttnn.multiply(state, g_view, input_tensor_b_activations=[ttnn.UnaryOpType.EXP], output_tensor=state)
        read = ttnn.matmul(
            k_row,
            state,
            memory_config=dram,
            program_config=self.state_read_program,
            compute_kernel_config=self.state_compute,
        )
        difference = ttnn.subtract(v, read, dtype=ttnn.float32, memory_config=dram)
        delta = ttnn.multiply(difference, beta_view, memory_config=dram)
        for tensor in (read, difference):
            ttnn.deallocate(tensor)
        outer = ttnn.matmul(
            k_row,
            delta,
            transpose_a=True,
            program_config=self.outer_program_config,
            memory_config=dram,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(k_row)
        ttnn.deallocate(delta)
        ttnn.add(state, outer, output_tensor=state)
        ttnn.deallocate(outer)
        result = ttnn.matmul(
            q_row,
            state,
            memory_config=dram,
            program_config=self.state_read_program,
            compute_kernel_config=self.state_compute,
        )
        ttnn.deallocate(q_row)
        return result

    def _prefill_linear(self, x, role, **kwargs):
        large = x.shape[1] >= self.optimization.large_prefill_min_seq
        grid = self.optimization.large_prefill_grid if large else (8, 8)
        weight = self.w[role]
        k, n = weight.shape[-2], weight.shape[-1]
        m = math.prod(list(x.padded_shape)[:-1]) // 32
        pm, pn = math.ceil(m / grid[1]), math.ceil(n / 32 / grid[0])
        block_w = self.optimization.large_prefill_block_w if large else 8
        block_m = max(v for v in range(1, min(pm, 8) + 1) if pm % v == 0)
        block_n = max(v for v in range(1, min(pn, 32) + 1) if pn % v == 0)
        sub_h_max, sub_w_max = self.optimization.large_prefill_subblock if large else (2, 4)
        subblock_h = max(v for v in range(1, min(block_m, sub_h_max) + 1) if block_m % v == 0)
        subblock = max(v for v in range(1, min(8 // subblock_h, sub_w_max) + 1) if block_n % v == 0)
        kwargs.pop("core_grid", None)
        program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=block_w,
            per_core_M=pm,
            per_core_N=pn,
            out_block_h=block_m,
            out_block_w=block_n,
            out_subblock_h=subblock_h,
            out_subblock_w=subblock,
            transpose_mcast=False,
            fuse_batch=True,
        )
        out = ttnn.linear(
            x,
            weight,
            program_config=program,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.projection_compute[role],
            **kwargs,
        )
        return out

    def _attention_prefill(self, x, page_table, chunk_start_idx):
        if self.k_cache is None:
            raise RuntimeError("call allocate_kv_cache()/attach_kv_cache() before prefill")
        if page_table is None:
            raise ValueError("full_attention prefill requires a page_table")
        b, t = x.shape[0], x.shape[1]
        cos, sin = [
            ttnn.slice(self.w[key], [0, 0, chunk_start_idx, 0], [1, 1, chunk_start_idx + t, self.cfg.rope_dim])
            for key in ("rope_cos_tiled", "rope_sin_tiled")
        ]
        cos = ttnn.reshape(cos, [1, 1, t, self.cfg.rope_dim])
        sin = ttnn.reshape(sin, [1, 1, t, self.cfg.rope_dim])

        q, k, v, gate = self._project_qkv(x)
        q = self._apply_partial_rope(q, cos, sin)
        k = self._apply_partial_rope(k, cos, sin)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)

        blk0 = chunk_start_idx // self.page_block_size
        blk_n = _align_up(chunk_start_idx + t, self.page_block_size) // self.page_block_size
        chunk_page_table, pt_owned = _slice_owned(page_table, [0, blk0], [int(page_table.shape[0]), blk_n])
        indices, owned = _slice_owned(self.w["batch_idxs"], [0], [b])
        k = ttnn.typecast(k, self.k_cache.dtype)
        v = ttnn.typecast(v, self.v_cache.dtype)
        ttnn.experimental.paged_fill_cache(self.k_cache, k, chunk_page_table, batch_idx_tensor=indices)
        ttnn.experimental.paged_fill_cache(self.v_cache, v, chunk_page_table, batch_idx_tensor=indices)
        if owned:
            ttnn.deallocate(indices)
        if pt_owned:
            ttnn.deallocate(chunk_page_table)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            self.k_cache,
            self.v_cache,
            page_table,
            chunk_start_idx,
            scale=self.cfg.head_dim**-0.5,
            program_config=self._prefill_sdpa_config(chunk_start_idx, t),
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )
        ttnn.deallocate(q)
        attn = ttnn.reshape(ttnn.experimental.nlp_concat_heads(attn), [b, t, self.cfg.n_heads * self.cfg.head_dim])
        return self._attention_output(attn, gate)

    def _chunk_delta_rule(self, q, k, v, g, beta):
        cfg = self.cfg
        batch, seq = q.shape[0], q.shape[1]
        if seq % 32:
            raise ValueError("flat GDN requires a physically tile-aligned sequence")
        eye, tril, ones, masks = self.w["gdn_const_tiles"]

        def launch(q_, k_, v_, g_, beta_, state_):
            return ttnn.transformer.chunk_gated_delta_rule(
                q_,
                k_,
                v_,
                g_,
                beta_,
                initial_state=state_,
                output_final_state=True,
                chunk_size=32,
                use_qk_l2norm=False,
                output_head_major=True,
                eye=eye,
                tril=tril,
                ones=ones,
                masks=masks,
            )

        step = self.max_gdn_prefill_batch()
        if batch <= step:
            return launch(q, k, v, g, beta, self.recurrent_state)
        cores, states = [], []
        for start in range(0, batch, step):
            end = min(start + step, batch)
            args = [ttnn.slice(tensor, [start, 0, 0], [end, seq, tensor.shape[-1]]) for tensor in (q, k, v, g, beta)]
            state = ttnn.slice(
                self.recurrent_state,
                [start, 0, 0, 0],
                [end, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim],
            )
            core, final = launch(*args, state)
            for tensor in (*args, state):
                ttnn.deallocate(tensor)
            cores.append(core)
            states.append(final)
        output = ttnn.concat(cores, dim=0)
        final = ttnn.concat(states, dim=0)
        for tensor in cores + states:
            ttnn.deallocate(tensor)
        return output, final

    def _gdn_prefill_conv_fields(self, qkv, logical_len):
        cfg = self.cfg
        batch, seq = qkv.shape[0], qkv.shape[1]
        dram = ttnn.DRAM_MEMORY_CONFIG
        history_rows = [ttnn.to_layout(row, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram) for row in self.conv_state]
        history = ttnn.concat(history_rows, dim=1)
        for row in history_rows:
            ttnn.deallocate(row)
        tokens = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
        outputs = [[], [], []]
        for user in range(batch):
            user_tokens, tokens_owned = _slice_owned(tokens, [user, 0, 0], [user + 1, seq, cfg.conv_dim])
            user_history, history_owned = _slice_owned(history, [user, 0, 0], [user + 1, 3, cfg.conv_dim])
            fields = ttnn.experimental.kda.qkv_causal_conv1d_silu(
                user_tokens,
                user_history,
                *self.w["conv_taps"],
                cfg.linear_q_dim,
                cfg.linear_k_dim,
                cfg.linear_v_dim,
                program_config=self.kda_conv_program,
                memory_config=dram,
                compute_kernel_config=self.compute_kernel_config,
            )
            for destination, tensor in zip(outputs, fields):
                destination.append(tensor)
            if tokens_owned:
                ttnn.deallocate(user_tokens)
            if history_owned:
                ttnn.deallocate(user_history)
        if batch == 1:
            q, k, v = [parts[0] for parts in outputs]
        else:
            q, k, v = [ttnn.concat(parts, dim=0) for parts in outputs]
            for parts in outputs:
                for tensor in parts:
                    ttnn.deallocate(tensor)
        if logical_len >= 3:
            tail_rm = ttnn.slice(tokens, [0, logical_len - 3, 0], [batch, logical_len, cfg.conv_dim])
        else:
            old = ttnn.slice(history, [0, logical_len, 0], [batch, 3, cfg.conv_dim])
            new = ttnn.slice(tokens, [0, 0, 0], [batch, logical_len, cfg.conv_dim])
            tail_rm = ttnn.concat([old, new], dim=1)
            ttnn.deallocate(old)
            ttnn.deallocate(new)
        ttnn.deallocate(history)
        ttnn.deallocate(tokens)
        return q, k, v, tail_rm
