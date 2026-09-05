# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Ornith decoder optimization with unchanged fused cache and sequence semantics."""

import math
from dataclasses import dataclass, field

import ttnn

from .fused_decoder import FusedDecoder, _batch_grid, _field, _height_memory


@dataclass(frozen=True)
class PrecisionPolicy:
    attention: str = "bfloat8_b"
    mlp_gate_up: str = "bfloat8_b"
    mlp_down: str = "bfloat8_b"
    attention_fidelity: str = "HiFi2"
    mlp_fidelity: str = "HiFi2"


@dataclass(frozen=True)
class DecoderConfig:
    residual_cores: int = 0
    dram_roles: tuple = ()
    cores: int = 8
    block_w: int = 8
    readers: int = 1
    role_configs: dict = field(default_factory=dict)
    packed_mlp: bool = False


class OptimizedDecoder(FusedDecoder):
    """Per-role projections; setup-only precision materialization."""

    @classmethod
    def from_state_dict(cls, state_dict, *, policy=None, config=None, **kwargs):
        decoder = super().from_state_dict(state_dict, **kwargs)
        decoder.policy = policy or PrecisionPolicy()
        decoder.optimization = config or DecoderConfig()
        decoder.decode_weights = {}
        decoder.norm_configs = {}
        decoder.projection_compute = {}
        for name in (
            "qkvg",
            "o_proj",
            "gdn_packed",
            "gdn_z_epilogue",
            "gdn_out",
            "gate_proj",
            "up_proj",
            "gate_up",
            "down_proj",
        ):
            if name not in decoder.w:
                continue
            mlp = name in ("gate_proj", "up_proj", "gate_up", "down_proj")
            group = "mlp_down" if name == "down_proj" else "mlp_gate_up" if mlp else "attention"
            dtype = getattr(ttnn, getattr(decoder.policy, group))
            if decoder.w[name].dtype != dtype:
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
        host_weights["gate_up"] = torch.cat([state_dict["mlp.gate_proj.weight"], state_dict["mlp.up_proj.weight"]]).T
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
        return decoder

    def _role_config(self, role):
        c = self.optimization
        r = c.role_configs.get(role, {})
        return r.get("cores", c.cores), r.get("block_w", c.block_w), r.get("readers", c.readers)

    def _width_memory(self, width, cores, rows=32):
        size = self.device.compute_with_storage_grid_size()
        cols = max(x for x in range(1, size.x + 1) if cores % x == 0 and cores // x <= size.y)
        grid = ttnn.CoreGrid(x=cols, y=cores // cols)
        return ttnn.create_sharded_memory_config(
            (rows, math.ceil(width / (cores * 32)) * 32),
            core_grid=grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _norm(self, x, weight):
        if self.optimization.residual_cores and x.shape[-1] == self.cfg.dim and x.shape[-2] == 1:
            cores = self.optimization.residual_cores
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
            return ttnn.rms_norm(x, weight=weight, epsilon=self.cfg.norm_eps, program_config=config, memory_config=mem)
        return super()._norm(x, weight)

    def _linear(self, x, role, **kwargs):
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
                per_core_N=math.ceil(n / (32 * cores)),
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
            if role not in ("gate_proj", "up_proj", "down_proj", "gate_up") or batch > 1:
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

    def _block(self, x, *, mode, logical_len=None, page_table=None, chunk_start_idx=0, current_pos=None, rot_idxs=None):
        residual_mem = (
            self._width_memory(self.cfg.dim, self.optimization.residual_cores, math.prod(list(x.padded_shape)[:-1]))
            if mode == "decode" and self.optimization.residual_cores
            else ttnn.DRAM_MEMORY_CONFIG
        )
        if mode == "decode":
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
        mixed = ttnn.to_memory_config(mixed, residual_mem)
        h = ttnn.add(x, mixed, memory_config=residual_mem)
        ttnn.deallocate(mixed)
        ff_in = self._norm(h, self.w["ff_norm"])
        activated = self._activate_mlp(ff_in, mode)
        ttnn.deallocate(ff_in)
        ff_out = self._linear(activated, "down_proj")
        ttnn.deallocate(activated)
        ff_out = ttnn.to_memory_config(ff_out, residual_mem)
        out = ttnn.add(h, ff_out, memory_config=residual_mem)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_out)
        return ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)

    def _activate_mlp(self, ff_in, mode):
        if mode == "prefill" or self.optimization.packed_mlp:
            gate_up = self._linear(ff_in, "gate_up")
            gate, up = ttnn.chunk(gate_up, 2, dim=-1)
            ttnn.deallocate(gate_up)
        else:
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
        q, k = [ttnn.to_memory_config(a, ttnn.DRAM_MEMORY_CONFIG) for a in (q, k)]
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
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            self.k_cache,
            self.v_cache,
            cur_pos_tensor=current_pos,
            page_table_tensor=page_table,
            is_causal=True,
            scale=cfg.head_dim**-0.5,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                q_chunk_size=32,
                k_chunk_size=64,
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
