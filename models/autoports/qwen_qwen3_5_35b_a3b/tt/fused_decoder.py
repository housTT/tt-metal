# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-device graph fusion over the validated Qwen3.5 decoder contract.

Construction and request planning reuse the functional stage. Runtime rewrites
are installed on this instance only; imported functional code is never patched.
"""

from typing import Any, Optional, Sequence

from .functional_decoder import FunctionalDecoder, _FunctionalRMSNorm, _FunctionalRouter
from .functional_gdn import FunctionalGatedDeltaNet


class FusedDecoder(FunctionalDecoder):
    DEFAULT_FUSIONS = (
        "moe_gate",
        "shared_gate",
        "shared_minimal",
        "rope",
        "cache",
        "qkv",
        "heads",
        "indexed_rope",
        "gate",
        "kda_conv",
        "kda_tuple",
        "kda_tail",
        "kda_norm",
        "gdn_heads",
        "gdn_norm",
        "gdn_eltwise",
        "gdn_grouped",
        "gdn_state_ternary",
        "gdn_inplace",
        "gdn_flat_v",
        "gdn_head_major",
        "gdn_fused_chunk",
        "compact_reduce",
        "compact_routing",
        "prefill_split",
    )

    @classmethod
    def from_state_dict(cls, *args, fusions=None, **kwargs):
        selected = cls.DEFAULT_FUSIONS if fusions is None else tuple(fusions)
        attention_fusions = {"rope", "qkv", "cache", "heads", "gate", "decode_heads", "decode_concat", "indexed_rope"}
        gdn_fusions = {
            "kda_conv",
            "kda_tuple",
            "kda_tail",
            "kda_norm",
            "kda_norm_decode",
            "gdn_heads",
            "gdn_norm",
            "gdn_ternary",
            "gdn_eltwise",
            "native_recurrence",
            "gdn_grouped",
            "gdn_state_ternary",
            "gdn_conv_layout",
            "gdn_inplace",
            "gdn_flat_v",
            "gdn_head_major",
            "gdn_fused_chunk",
        }
        unknown = (
            set(selected)
            - {
                "norm",
                "router",
                "expert_layout",
                "moe_gate",
                "shared_pack",
                "shared_gate",
                "shared_sigmoid",
                "shared_minimal",
                "fused_ffn",
                "fused_ffn_hifi",
                "fused_ffn_grid2",
                "prefill_minimal",
                "compact_ffn",
                "compact_reduce",
                "prefill_split",
                "prefill_split32",
                "compact_routing",
                "projection_residual",
            }
            - attention_fusions
            - gdn_fusions
        )
        if unknown:
            raise ValueError(f"Unknown fusion candidates: {sorted(unknown)}")
        if "gdn_inplace" in selected and (
            not {"gdn_grouped", "gdn_state_ternary"}.intersection(selected) or "native_recurrence" in selected
        ):
            raise ValueError("gdn_inplace requires the explicit FP32 grouped/ternary state update")
        if "compact_routing" in selected and (
            "moe_gate" not in selected or not {"compact_ffn", "compact_reduce"}.intersection(selected)
        ):
            raise ValueError("compact_routing requires moe_gate and a compact expert consumer")
        if len({"prefill_minimal", "prefill_split", "prefill_split32"}.intersection(selected)) > 1:
            raise ValueError("Select one packed prefill adapter")
        if "indexed_rope" in selected and (not {"rope", "qkv", "heads"} <= set(selected) or "decode_heads" in selected):
            raise ValueError("indexed_rope requires packed prefill-layout heads and rope")
        result = super().from_state_dict(*args, **kwargs)
        ignored = gdn_fusions if result.config.is_full_attention_layer(result.layer_idx) else attention_fusions
        selected = tuple(flag for flag in selected if flag not in ignored)
        result.fusions = selected
        if "norm" in selected:
            for attr in ("attention_norm", "ffn_norm"):
                previous = getattr(result.layer, attr)
                fused = _FusedRMSNorm.__new__(_FusedRMSNorm)
                fused.__dict__.update(previous.__dict__)
                setattr(result.layer, attr, fused)
        moe = result.layer.feed_forward.module
        if "router" in selected:
            router = _FusedRouter.__new__(_FusedRouter)
            router.__dict__.update(moe.router.__dict__)
            moe.router = router
        if "expert_layout" in selected:
            previous = moe.experts
            moe.experts = _FusedExperts(previous)
        if "moe_gate" in selected:
            from models.demos.blackhole.qwen36.tt.weight_mapping import remap_qwen36_state_dict

            state = args[0] if args else kwargs["state_dict"]
            prefix = f"layers.{result.layer_idx}.mlp."
            if prefix + "gate.weight" not in state:
                state = remap_qwen36_state_dict(dict(state))
            moe.router = _DedicatedRouter(
                moe.router, kwargs["mesh_device"], state[prefix + "gate.weight"], "compact_routing" in selected
            )
        if {"shared_pack", "shared_gate", "shared_sigmoid", "shared_minimal"}.intersection(selected):
            state = args[0] if args else kwargs["state_dict"]
            from models.demos.blackhole.qwen36.tt.weight_mapping import remap_qwen36_state_dict

            prefix = f"layers.{result.layer_idx}.mlp.shared_expert."
            if prefix + "gate_proj.weight" not in state:
                state = remap_qwen36_state_dict(dict(state))
            moe.shared = _FusedShared(
                moe.shared,
                selected,
                kwargs["mesh_device"],
                {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)},
            )
        if {"fused_ffn", "fused_ffn_hifi", "fused_ffn_grid2"}.intersection(selected):
            moe.experts = _DedicatedExperts(
                moe.experts,
                kwargs["mesh_device"],
                "fused_ffn_hifi" in selected,
                2 if "fused_ffn_grid2" in selected else 8,
            )
        if {"compact_ffn", "compact_reduce"}.intersection(selected):
            moe.experts = _CompactExperts(moe.experts, kwargs["mesh_device"], "compact_reduce" in selected)
        if {"prefill_minimal", "prefill_split", "prefill_split32"}.intersection(selected):
            moe.experts = _PackedPrefillExperts(
                moe.experts, kwargs["mesh_device"], "prefill_split" in selected, "prefill_split32" in selected
            )
        if result.config.is_full_attention_layer(result.layer_idx) and (
            attention_fusions.intersection(selected) or "projection_residual" in selected
        ):
            from models.demos.blackhole.qwen36.tt.weight_mapping import remap_qwen36_state_dict

            state = args[0] if args else kwargs["state_dict"]
            prefix = f"layers.{result.layer_idx}.self_attn."
            if prefix + "q_proj.weight" not in state:
                state = remap_qwen36_state_dict(dict(state))
            result.layer.attention = _FusedAttention.from_existing(
                result.layer.attention,
                selected,
                {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)},
            )
        if result.config.is_deltanet_layer(result.layer_idx) and (
            gdn_fusions.intersection(selected) or "projection_residual" in selected
        ):
            result.layer.attention = _FusedGDN.from_existing(result.layer.attention, selected)
        if "projection_residual" in selected:
            result.layer = _ProjectionResidualLayer(result.layer)
        if "indexed_rope" in selected:
            _enable_indexed_rope(result, kwargs["mesh_device"])
        return result

    def prefill_forward(
        self,
        hidden_states: Any,
        *,
        valid_len: Optional[int] = None,
        page_table: Any = None,
        start_pos: int = 0,
        request_slots: Optional[Sequence[int]] = None,
    ) -> Any:
        """Consume [B,T,H] device activations; return all logical rows [B,T,H].

        start_pos is the absolute cached-prefix length; state is retained until an
        explicit reset. Full attention requires an owned page table [B,blocks].
        Bounded chunks and token-wise tails never write padding into live state.
        Partial text RoPE and chunk planning are owned by this decoder.
        """
        import ttnn

        if len(hidden_states.shape) != 3:
            raise ValueError("hidden_states must have shape [B,T,H]")
        batch, physical_len, hidden = hidden_states.shape
        length = physical_len if valid_len is None else int(valid_len)
        if not 0 < length <= physical_len or hidden != self.config.dim:
            raise ValueError("invalid logical length or hidden dimension")
        if start_pos < 0 or start_pos + length > self.config.max_seq_len:
            raise ValueError("prefill exceeds advertised context")
        full = self.config.is_full_attention_layer(self.layer_idx)
        if full and (page_table is None or not self.layer.attention.use_paged_attention):
            raise ValueError("full attention requires bound paged KV and page_table")
        if full and (
            page_table.shape[0] != batch
            or page_table.shape[1] * self.layer.attention.paged_kv_cache_key.shape[2] < start_pos + length
        ):
            raise ValueError("page table does not cover the requested batch/context")
        read_page_table = page_table
        if full and page_table.shape[1] % 8:
            # Chunk SDPA reads 32-byte sticks. Extra entries are outside the
            # logical causal prefix; pad on device without changing ownership.
            read_page_table = ttnn.pad(page_table, ((0, 0), (0, -page_table.shape[1] % 8)), 0)
        flat_input = ttnn.reshape(ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT), (batch * physical_len, hidden))
        # zeros(shape) uploads a host-created buffer even after warmup. Create
        # the output's zero padding on device, retaining the planner's RM writes.
        output = ttnn.to_layout(
            ttnn.zeros_like(ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT), memory_config=ttnn.DRAM_MEMORY_CONFIG),
            ttnn.ROW_MAJOR_LAYOUT,
        )
        for row, slot in enumerate(self._slots(batch, request_slots)):
            self._select_state(slot)
            pt = page_table[row : row + 1, :] if full else None
            read_pt = read_page_table[row : row + 1, :] if full else None
            block = self.layer.attention.paged_kv_cache_key.shape[2] if full else 128
            offset = 0
            while offset < length:
                absolute = start_pos + offset
                buckets = (512, 128) if full else (128,)
                count = next(
                    (
                        n
                        for n in buckets
                        if n % block == 0 and absolute % block == 0 and absolute % 128 == 0 and length - offset >= n
                    ),
                    1,
                )
                x = ttnn.embedding(
                    self._indices(row * physical_len + offset, count), flat_input, layout=ttnn.TILE_LAYOUT
                )
                kw = {}
                if full:
                    pos = self._indices(absolute, count)
                    c, s = self._rotary(pos, count)
                    if count == 1:
                        kw = dict(
                            cos=c,
                            sin=s,
                            position_tensor=ttnn.reshape(ttnn.typecast(pos, ttnn.int32), (1,)),
                            page_table=pt,
                        )
                    else:
                        kw = dict(
                            cos=ttnn.unsqueeze(c, 1),
                            sin=ttnn.unsqueeze(s, 1),
                            page_table=read_pt,
                            chunk_page_table=ttnn.gather(pt, 1, self._indices(absolute // block, count // block)),
                            chunk_start_idx_tensor=ttnn.reshape(
                                ttnn.typecast(self._indices(absolute, 1), ttnn.int32), (1,)
                            ),
                        )
                y = self.layer.forward(x, mode="decode" if count == 1 else "prefill", chunk_size=128, **kw)
                ttnn.experimental.slice_write(
                    ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT),
                    output,
                    (row, offset, 0),
                    (row + 1, offset + count, hidden),
                    (1, 1, 1),
                )
                offset += count
        tiled = ttnn.to_layout(output, ttnn.TILE_LAYOUT)
        return ttnn.reshape(tiled, (batch, length, hidden), tiled.padded_shape)


class _FusedRMSNorm(_FunctionalRMSNorm):
    def __call__(self, x, *, mode=None, norm_config=None):
        import ttnn

        normalized = ttnn.rms_norm(
            ttnn.typecast(x, ttnn.float32),
            epsilon=self.eps,
            weight=self.weight,
            compute_kernel_config=self.compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.typecast(normalized, ttnn.bfloat16)


class _FusedRouter(_FunctionalRouter):
    def __call__(self, hidden_states):
        import ttnn

        if not self.norm_topk_prob:
            raise ValueError("The fused router requires normalized top-k routing")
        logits = ttnn.linear(hidden_states, self.proj_weight, compute_kernel_config=self.compute_kernel_config)
        # softmax is monotone, and dividing selected probabilities by their sum
        # cancels the full-softmax denominator. Retain FP32 probability math.
        values, indices = ttnn.topk(ttnn.typecast(logits, ttnn.float32), k=self.top_k, dim=-1)
        values = ttnn.typecast(ttnn.softmax(values, dim=-1), ttnn.bfloat16)
        return ttnn.scatter(ttnn.zeros_like(logits), dim=-1, index=indices, src=values)


class _FusedExperts:
    def __init__(self, original):
        self.__dict__.update(original.__dict__)
        self.original = original

    def __call__(self, hidden_states, routing_weights, mode="decode"):
        if mode != "decode":
            return self.original(hidden_states, routing_weights, mode=mode)
        import ttnn
        from models.demos.blackhole.qwen36.tt.moe.decode import _build_sparse_matmul_config

        assert hidden_states.shape[2] == 1 and self.num_devices == 1
        experts, intermediate = self.config.num_experts, self.weights.intermediate_size_per_device
        sparsity = ttnn.to_layout(routing_weights, ttnn.ROW_MAJOR_LAYOUT)
        up_gate = ttnn.sparse_matmul(
            hidden_states,
            self.weights.gate_up_proj,
            sparsity=sparsity,
            nnz=self.config.top_k,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=ttnn.Tile([32, 32]),
            program_config=_build_sparse_matmul_config(1, 2 * intermediate),
            dtype=ttnn.bfloat16,
        )
        # Keep the expert-major layout through SwiGLU and the down projection.
        up_gate = ttnn.reshape(up_gate, (1, experts, 1, 2 * intermediate))
        down_input = ttnn.swiglu(up_gate, dim=-1)
        down = ttnn.sparse_matmul(
            down_input,
            self.weights.down_proj,
            sparsity=sparsity,
            nnz=self.config.top_k,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=ttnn.Tile([32, 32]),
            program_config=_build_sparse_matmul_config(1, self.config.hidden_size),
            is_input_a_sparse=True,
            dtype=ttnn.bfloat16,
        )
        routing = ttnn.permute(routing_weights, (0, 3, 2, 1))
        weighted = ttnn.multiply(down, routing)
        reduced = ttnn.experimental.fast_reduce_nc(weighted, dims=[1])
        return reduced[:, :, :1, :]


class _PackedPrefillExperts:
    """SwiGLU epilogue over horizontally packed expert columns; decode stays sparse."""

    def __init__(self, original, mesh, split=False, split_single=False):
        import torch

        import ttnn

        self.__dict__.update(original.__dict__)
        self.original = original
        self.split = split
        self.split_single = split_single
        width = self.weights.intermediate_size_per_device
        # Move complete tiles of already quantized values; preserve BFP4 groups.
        values = ttnn.to_torch(self.weights.gate_up_proj, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[0]
        experts, hidden = values.shape[:2]
        up = values[..., :width].reshape(experts, hidden, width // 32, 32)
        gate = values[..., width:].reshape(experts, hidden, width // 32, 32)
        packed = torch.stack((gate, up), dim=3).permute(1, 0, 2, 3, 4).contiguous().reshape(hidden, experts * width * 2)
        self.packed = ttnn.from_torch(
            packed,
            device=mesh,
            dtype=self.weights.gate_up_proj.dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        # Large split counts exhaust the NCRISC accessor tuple's stack. Keep
        # each call at 32 outputs; grouped calls win the measured prefill.
        self.packed_groups = (
            [self.packed[:, i * 32 * width * 2 : (i + 1) * 32 * width * 2] for i in range(experts // 32)]
            if split
            else []
        )
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    def __call__(self, hidden, routing, mode="decode"):
        if mode == "decode":
            return self.original(hidden, routing, mode=mode)
        import ttnn
        from models.demos.blackhole.qwen36.tt.moe.decode import _build_sparse_matmul_config

        length, experts, width = hidden.shape[2], self.config.num_experts, self.weights.intermediate_size_per_device
        if self.split_single:
            parts = ttnn.experimental.minimal_matmul_split(
                hidden,
                self.packed,
                chunks=32,
                fuse_swiglu=True,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.compute,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            groups = [ttnn.permute(ttnn.reshape(part, (1, length, 8, width)), (0, 2, 1, 3)) for part in parts]
            out = ttnn.concat(groups, dim=1)
        elif self.split:
            groups = []
            for weight in self.packed_groups:
                parts = ttnn.experimental.minimal_matmul_split(
                    hidden,
                    weight,
                    chunks=32,
                    fuse_swiglu=True,
                    dtype=ttnn.bfloat16,
                    compute_kernel_config=self.compute,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                groups.append(ttnn.concat(parts, dim=1))
            out = ttnn.concat(groups, dim=1)
        else:
            out = ttnn.experimental.minimal_matmul(
                hidden,
                self.packed,
                fuse_swiglu=True,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.compute,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            out = ttnn.permute(ttnn.reshape(out, (1, length, experts, width)), (0, 2, 1, 3))
        down = ttnn.sparse_matmul(
            out,
            self.weights.down_proj,
            sparsity=self.prefill_sparsity,
            nnz=experts,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=ttnn.Tile([32, 32]),
            program_config=_build_sparse_matmul_config(length, self.config.hidden_size),
            is_input_a_sparse=True,
            dtype=ttnn.bfloat16,
        )
        out = ttnn.multiply(down, ttnn.permute(routing, (0, 3, 2, 1)))
        return ttnn.experimental.fast_reduce_nc(out, dims=[1])


class _DedicatedExperts:
    """Try the fused routed FFN with device-resident activity counts.

    Each active expert sees the full bounded chunk; routing masks its contribution
    afterwards. This preserves the single-layer sparse semantics without dispatch."""

    def __init__(self, original, mesh, high_fidelity=False, grid_y=8):
        import torch

        import ttnn

        self.__dict__.update(original.__dict__)
        self.grid_y = grid_y
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4 if high_fidelity else ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=False,
        )
        up_gate = ttnn.typecast(self.weights.gate_up_proj, self.weights.down_proj.dtype)
        width = self.weights.intermediate_size_per_device
        self.up, self.gate, self.down = [], [], []
        for expert in range(self.config.num_experts):
            self.up.append(ttnn.reshape(up_gate[:, expert : expert + 1, :, :width], (2048, width)))
            self.gate.append(ttnn.reshape(up_gate[:, expert : expert + 1, :, width:], (2048, width)))
            self.down.append(ttnn.reshape(self.weights.down_proj[:, expert : expert + 1, :, :], (width, 2048)))

        def put(x):
            return ttnn.from_torch(
                x,
                device=mesh,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        self.expert_ids = put(torch.arange(256, dtype=torch.int32))
        self.offsets = {length: put(torch.arange(256, dtype=torch.int32) * length) for length in (32, 128, 512)}

    def __call__(self, hidden, routing, mode="decode"):
        import ttnn

        length = hidden.shape[2]
        physical = max(32, length)
        activity = ttnn.gt(ttnn.max(routing, dim=2, keepdim=True), 0.0)
        counts = ttnn.reshape(
            ttnn.to_layout(ttnn.typecast(ttnn.multiply(activity, float(length)), ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT),
            (256,),
        )
        inputs = ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT)
        # Reader validates output region offsets against input capacity even when
        # read_x_at_offset=False. Only the first physical rows are actually read.
        inputs = ttnn.pad(inputs, ((0, 0), (0, 0), (0, 256 * physical - length), (0, 0)), 0)
        output = ttnn.zeros_like(
            ttnn.empty(
                (1, 1, 256 * physical, 2048),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=hidden.device(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
        output = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
            inputs,
            self.gate,
            self.up,
            self.down,
            counts,
            self.expert_ids,
            input_m_tiles=physical // 32,
            core_grid=ttnn.CoreCoord(8, self.grid_y),
            dtype=ttnn.bfloat16,
            output=output,
            expert_region_offsets=self.offsets[physical],
            read_x_at_offset=False,
            compute_kernel_config=self.compute,
        )
        output = ttnn.reshape(output, (1, 256, physical, 2048))[:, :, :length, :]
        output = ttnn.multiply(output, ttnn.permute(routing, (0, 3, 2, 1)))
        return ttnn.experimental.fast_reduce_nc(output, dims=[1])[:, :, :length, :]


class _CompactExperts(_DedicatedExperts):
    """Map selected expert IDs to eight compact native FFN output regions."""

    def __init__(self, original, mesh, fused_reduce=False):
        super().__init__(original, mesh, grid_y=2)
        import torch

        import ttnn

        self.original = original
        self.fused_reduce = fused_reduce

        def put(x, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT):
            return ttnn.from_torch(
                x, device=mesh, dtype=dtype, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
            )

        self.compact_counts = put(torch.tensor([1] * 8 + [0], dtype=torch.int32))
        self.compact_offsets = put(torch.tensor([32 * i for i in range(8)] + [0], dtype=torch.int32))
        self.idx_template = put(torch.full((1, 1, 1, 256), 8, dtype=torch.int32), layout=ttnn.TILE_LAYOUT)
        self.slot_labels = put(torch.arange(8, dtype=torch.int32).reshape(1, 1, 1, 8), layout=ttnn.TILE_LAYOUT)
        self.mapping = put(torch.zeros((1, 256), dtype=torch.int32), dtype=ttnn.uint16)

    def __call__(self, hidden, routing, mode="decode"):
        if mode != "decode":
            return self.original(hidden, routing, mode=mode)
        import ttnn

        values, indices = routing if isinstance(routing, tuple) else ttnn.topk(routing, k=8, dim=-1)
        indices = ttnn.typecast(indices, ttnn.uint32)
        table = ttnn.scatter(self.idx_template, dim=-1, index=indices, src=self.slot_labels)
        table = ttnn.reshape(ttnn.to_layout(table, ttnn.ROW_MAJOR_LAYOUT), (256,))
        inputs = ttnn.pad(ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT), ((0, 0), (0, 0), (0, 255), (0, 0)), 0)
        out = ttnn.empty(
            (1, 1, 256, 2048),
            device=hidden.device(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
            inputs,
            self.gate,
            self.up,
            self.down,
            self.compact_counts,
            table,
            input_m_tiles=1,
            core_grid=ttnn.CoreCoord(8, 2),
            dtype=ttnn.bfloat16,
            output=out,
            expert_region_offsets=self.compact_offsets,
            read_x_at_offset=False,
            compute_kernel_config=self.compute,
        )
        out = ttnn.reshape(out, (8, 1, 32, 2048))[:, :, :1, :]
        if self.fused_reduce:
            return ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
                out,
                ttnn.to_layout(ttnn.typecast(indices, ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT),
                self.mapping,
                reduce_dim=0,
                split_size=2048,
                cluster_axis=0,
                output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                scores_tensor=ttnn.to_layout(values, ttnn.ROW_MAJOR_LAYOUT),
            )[0]
        return ttnn.experimental.fast_reduce_nc(ttnn.multiply(out, ttnn.permute(values, (3, 1, 2, 0))), dims=[0])[
            :, :, :1, :
        ]


class _DedicatedRouter:
    def __init__(self, original, mesh, weight, compact=False):
        import ttnn
        from models.common.modules.moe.tt_moe_gate import TTMoEGate
        from models.common.modules.moe.tt_moe_gate_config import TTMoEGateConfig

        self.compact = compact
        self.module = TTMoEGate(
            mesh,
            TTMoEGateConfig(
                num_routed_experts=256,
                select_experts_k=8,
                hidden_size=2048,
                gate_matmul_auto_program_config=False,
                gate_matmul_compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True, "math_approx_mode": False},
            ),
            weight.T.contiguous(),
        )
        self.zero_templates = {
            length: ttnn.zeros((1, 1, length, 256), dtype=ttnn.bfloat16, device=mesh, layout=ttnn.TILE_LAYOUT)
            for length in (1, 128, 512)
        }

    def __call__(self, x):
        import ttnn

        values, indices = self.module.forward(x)
        values = ttnn.permute(values, (1, 2, 0, 3))
        indices = ttnn.permute(ttnn.typecast(indices, ttnn.uint32), (1, 2, 0, 3))
        if self.compact and x.shape[2] == 1:
            return values, indices
        dense = ttnn.zeros_like(self.zero_templates[x.shape[2]])
        return ttnn.scatter(dense, dim=-1, index=indices, src=values)


class _FusedShared:
    def __init__(self, original, fusions, mesh, state):
        import ttnn

        self.__dict__.update(original.__dict__)
        self.fusions = fusions
        if "shared_pack" in fusions:
            self.packed = ttnn.concat((self.mlp.weights.w3, self.mlp.weights.w1), dim=-1)
        if "shared_minimal" in fusions:
            import torch

            gate, up = state["gate_proj.weight"].T, state["up_proj.weight"].T
            packed = torch.stack((gate.reshape(2048, 16, 32), up.reshape(2048, 16, 32)), dim=2)
            self.packed = ttnn.from_torch(
                packed.reshape(2048, 1024).contiguous(),
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat4_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

    def forward(self, x):
        import ttnn

        if "shared_sigmoid" in self.fusions:
            gate = ttnn.linear(x, self.gate_weight, activation="sigmoid")
        else:
            gate = ttnn.linear(x, self.gate_weight)
        if "shared_pack" in self.fusions or "shared_minimal" in self.fusions:
            # Match the inherited rank-four shared-expert matmul configuration.
            ckc = self.mlp.compute_kernel_config_decode
            if "shared_minimal" in self.fusions:
                hidden = ttnn.experimental.minimal_matmul(
                    x, self.packed, fuse_swiglu=True, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG
                )
            else:
                packed = ttnn.linear(x, self.packed, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG)
                hidden = ttnn.swiglu(packed, dim=-1)
            output = ttnn.linear(
                hidden, self.mlp.weights.w2, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            output = output[:, :, : x.shape[2], :]
        else:
            output = self.mlp.forward(x)
        if "shared_sigmoid" in self.fusions:
            return ttnn.multiply(output, gate)
        if "shared_gate" in self.fusions:
            return ttnn.multiply(gate, output, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        return ttnn.multiply(output, ttnn.sigmoid(gate))


class _FusedAttention:
    @classmethod
    def from_existing(cls, original, fusions, state):
        import ttnn

        self = cls()
        self.__dict__.update(original.__dict__)
        self.fusions = fusions
        if {"qkv", "heads", "decode_heads"}.intersection(fusions):
            import torch

            h, d = self.config.num_heads, self.config.head_dim
            qg = state["q_proj.weight"].reshape(h, 2 * d, -1)
            packed = torch.cat(
                (
                    qg[:, :d].reshape(h * d, -1),
                    state["k_proj.weight"],
                    state["v_proj.weight"],
                    qg[:, d:].reshape(h * d, -1),
                ),
                dim=0,
            ).T.contiguous()
            self.qkvg_weight = ttnn.from_torch(
                packed,
                device=self.device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )
        self.cache_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
                [32, self.config.head_dim],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        self.value_cache_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))}),
                [32, self.config.head_dim],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        return self

    def set_paged_kv_cache(self, key, value):
        self.paged_kv_cache_key, self.paged_kv_cache_value = key, value
        self.use_paged_attention = True

    def reset_cache(self):
        self.past_key = self.past_value = None

    def _rope(self, x, cos, sin, decode):
        import ttnn

        if "indexed_rope" in self.fusions:
            output = ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
                x,
                cos,
                sin,
                self.rotation,
                0,
                0,
                rotary_dim=64,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.L1_MEMORY_CONFIG if decode else ttnn.DRAM_MEMORY_CONFIG,
            )
            return ttnn.transpose(output, 1, 2) if decode else output
        width = cos.shape[-1]
        rotated = ttnn.experimental.rotary_embedding(
            x[..., :width],
            cos,
            sin,
            token_index=0 if decode else None,
        )
        if decode:
            rotated = rotated[:, :, : x.shape[2], :]
        return ttnn.concat((rotated, x[..., width:]), dim=-1)

    def forward(
        self,
        x,
        cos,
        sin,
        position_tensor=None,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        chunk_start_idx_tensor=None,
        residual=None,
    ):
        import ttnn
        from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention import (
            _get_flexible_sdpa_program_config,
            _get_paged_sdpa_decode_program_config,
            _get_sdpa_compute_kernel_config,
            apply_rotary_pos_emb_ttnn,
        )

        batch, length = x.shape[0], x.shape[1]
        assert batch == 1 and self.use_paged_attention
        h, kv, d = self.config.num_heads, self.config.num_kv_heads, self.config.head_dim
        decode = length == 1
        ckc = self.compute_kernel_config_decode if decode else self.compute_kernel_config
        mc = ttnn.L1_MEMORY_CONFIG if decode else ttnn.DRAM_MEMORY_CONFIG
        heads_in_decode = decode and "decode_heads" in self.fusions
        if {"qkv", "heads", "decode_heads"}.intersection(self.fusions):
            packed = ttnn.linear(x, self.qkvg_weight, compute_kernel_config=ckc, memory_config=mc)
            qw, kw = h * d, kv * d
            gate = packed[..., qw + 2 * kw :]
            if heads_in_decode:
                query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
                    ttnn.reshape(packed[..., : qw + 2 * kw], (1, 1, 1, qw + 2 * kw)),
                    num_heads=h,
                    num_kv_heads=kv,
                    memory_config=self.cache_mem,
                )
                query, key, value = (ttnn.to_memory_config(t, mc) for t in (query, key, value))
            elif "heads" in self.fusions:
                query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
                    packed[..., : qw + 2 * kw],
                    num_heads=h,
                    num_kv_heads=kv,
                    transpose_key=False,
                )
            else:
                query = ttnn.transpose(ttnn.reshape(packed[..., :qw], (1, length, h, d)), 1, 2)
                key = ttnn.transpose(ttnn.reshape(packed[..., qw : qw + kw], (1, length, kv, d)), 1, 2)
                value = ttnn.transpose(ttnn.reshape(packed[..., qw + kw : qw + 2 * kw], (1, length, kv, d)), 1, 2)
            query = ttnn.rms_norm(query, weight=self.weights.q_norm, epsilon=self.config.norm_eps)
            key = ttnn.rms_norm(key, weight=self.weights.k_norm, epsilon=self.config.norm_eps)
        else:
            qg = ttnn.linear(x, self.weights.q_proj, compute_kernel_config=ckc, memory_config=mc)
            query, gate = ttnn.chunk(ttnn.reshape(qg, (1, length, h, 2 * d)), 2, dim=-1)
            gate = ttnn.reshape(gate, (1, length, h * d))
            query = ttnn.transpose(ttnn.rms_norm(query, weight=self.weights.q_norm, epsilon=self.config.norm_eps), 1, 2)
            key = ttnn.linear(x, self.weights.k_proj, compute_kernel_config=ckc, memory_config=mc)
            key = ttnn.reshape(key, (1, length, kv, d))
            key = ttnn.transpose(ttnn.rms_norm(key, weight=self.weights.k_norm, epsilon=self.config.norm_eps), 1, 2)
            value = ttnn.linear(x, self.weights.v_proj, compute_kernel_config=ckc, memory_config=mc)
            value = ttnn.transpose(ttnn.reshape(value, (1, length, kv, d)), 1, 2)
        cos = ttnn.reshape(cos, (1, 1, length, cos.shape[-1]))
        sin = ttnn.reshape(sin, (1, 1, length, sin.shape[-1]))
        if "rope" in self.fusions:
            if decode and not heads_in_decode and "indexed_rope" not in self.fusions:
                query = ttnn.transpose(query, 1, 2)
                key = ttnn.transpose(key, 1, 2)
            query, key = self._rope(query, cos, sin, decode), self._rope(key, cos, sin, decode)
        else:
            query, key = apply_rotary_pos_emb_ttnn(query, key, cos, sin)
            if decode and not heads_in_decode:
                query = ttnn.transpose(query, 1, 2)
                key = ttnn.transpose(key, 1, 2)
        if decode:
            key = ttnn.to_memory_config(key, self.cache_mem)
            value = ttnn.to_memory_config(
                value if heads_in_decode else ttnn.transpose(value, 1, 2),
                self.value_cache_mem if "cache" in self.fusions else self.cache_mem,
            )
            if "cache" in self.fusions:
                ttnn.experimental.paged_fused_update_cache(
                    self.paged_kv_cache_key,
                    key,
                    self.paged_kv_cache_value,
                    value,
                    update_idxs_tensor=position_tensor,
                    page_table=page_table,
                )
            else:
                for cache, tensor in ((self.paged_kv_cache_key, key), (self.paged_kv_cache_value, value)):
                    ttnn.experimental.paged_update_cache(
                        cache, tensor, update_idxs_tensor=position_tensor, page_table=page_table
                    )
            pc = (
                self._tile_decode_program_config
                if (page_table.shape[1] * self.paged_kv_cache_key.shape[2]) % 64
                else _get_paged_sdpa_decode_program_config(
                    self.device, self.paged_kv_cache_key.shape[0] * self.paged_kv_cache_key.shape[2]
                )
            )
            output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG),
                self.paged_kv_cache_key,
                self.paged_kv_cache_value,
                cur_pos_tensor=position_tensor,
                page_table_tensor=page_table,
                is_causal=True,
                scale=d**-0.5,
                program_config=pc,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if "decode_concat" not in self.fusions:
                output = ttnn.transpose(output, 1, 2)
        else:
            ttnn.experimental.paged_fill_cache(self.paged_kv_cache_key, key, chunk_page_table, batch_idx=0)
            ttnn.experimental.paged_fill_cache(self.paged_kv_cache_value, value, chunk_page_table, batch_idx=0)
            output = ttnn.transformer.chunked_scaled_dot_product_attention(
                query,
                self.paged_kv_cache_key,
                self.paged_kv_cache_value,
                page_table,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
                scale=d**-0.5,
                program_config=_get_flexible_sdpa_program_config(self.device),
                compute_kernel_config=_get_sdpa_compute_kernel_config(),
            )
        if decode and "decode_concat" in self.fusions:
            output = ttnn.experimental.nlp_concat_heads_decode(
                ttnn.to_memory_config(output, self.cache_mem), num_heads=h
            )
            output = ttnn.to_memory_config(output, mc)[:, :, :1, :]
            output = ttnn.reshape(output, (1, 1, h * d))
        else:
            output = ttnn.transformer.concatenate_heads(output)
        if "gate" in self.fusions:
            output = ttnn.multiply(gate, output, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        else:
            output = ttnn.multiply(output, ttnn.sigmoid(gate))
        if residual is not None:
            return self.residual_projection(output, residual, ckc, mc)
        return ttnn.linear(output, self.weights.o_proj, compute_kernel_config=ckc, memory_config=mc)


class _FusedGDN(FunctionalGatedDeltaNet):
    @classmethod
    def from_existing(cls, original, fusions):
        import torch

        import ttnn

        self = cls()
        self.__dict__.update(original.__dict__)
        self.fusions = fusions
        if "gdn_fused_chunk" in fusions:
            self.chunk_program_config = ttnn.ChunkGdnFusedProgramConfig()
        if {"kda_norm", "kda_norm_decode"}.intersection(fusions):
            self.kda_norm_weight = ttnn.reshape(self.norm_weight, (self.head_v_dim,))
            self.kda_norm_compute = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
                math_approx_mode=False,
            )
        if "kda_conv" in fusions:
            self.kda_taps = tuple(ttnn.typecast(t, ttnn.bfloat16) for t in self.conv_taps)
            self.kda_start = ttnn.from_torch(
                torch.zeros((1,), dtype=torch.int32),
                device=self.device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )
        return self

    def _convolve(self, qkv):
        import ttnn

        if "kda_conv" in self.fusions:
            length = qkv.shape[1]
            history = ttnn.to_layout(self.fused_conv_state, ttnn.ROW_MAJOR_LAYOUT)
            current = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT)
            if length == 1:
                current = ttnn.pad(current, ((0, 0), (0, 31), (0, 0)), 0)
            q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
                current,
                history,
                *self.kda_taps,
                self.cfg.q_dim,
                self.cfg.k_dim,
                self.cfg.v_dim,
                actual_start=self.kda_start,
                predecessor_carry=history,
                program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=512),
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if length >= 3 and "kda_tail" in self.fusions:
                tail = current[:, length - 3 : length, :]
            else:
                actual = ttnn.concat((history, current[:, :length, :]), dim=1)
                tail = actual[:, length : length + 3, :]
            ttnn.copy(ttnn.to_layout(tail, ttnn.TILE_LAYOUT), self.fused_conv_state)
            parts = (q[:, :length, :], k[:, :length, :], v[:, :length, :])
            return parts if "kda_tuple" in self.fusions else ttnn.concat(parts, dim=-1)
        if not {"gdn_ternary", "gdn_conv_layout"}.intersection(self.fusions):
            return super()._convolve(qkv)
        length = qkv.shape[1]
        if "gdn_conv_layout" in self.fusions:
            history = ttnn.concat(
                (
                    ttnn.to_layout(self.fused_conv_state, ttnn.ROW_MAJOR_LAYOUT),
                    ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT),
                ),
                dim=1,
            )
        else:
            history = ttnn.concat((self.fused_conv_state, qkv), dim=1)
        accumulated = None
        for tap, weight in enumerate(self.conv_taps):
            x = (
                ttnn.to_layout(history[:, tap : tap + length, :], ttnn.TILE_LAYOUT, dtype=ttnn.float32)
                if "gdn_conv_layout" in self.fusions
                else ttnn.typecast(ttnn.to_layout(history[:, tap : tap + length, :], ttnn.TILE_LAYOUT), ttnn.float32)
            )
            accumulated = ttnn.multiply(x, weight) if accumulated is None else ttnn.addcmul(accumulated, x, weight)
        tail = ttnn.to_layout(history[:, length : length + 3, :], ttnn.TILE_LAYOUT)
        ttnn.copy(tail, self.fused_conv_state)
        rounded = ttnn.typecast(accumulated, ttnn.bfloat16)
        return ttnn.typecast(ttnn.silu(ttnn.typecast(rounded, ttnn.float32)), ttnn.bfloat16)

    def _heads(self, x, heads, dim):
        import ttnn

        if "gdn_heads" in self.fusions:
            return ttnn.reshape(x, (1, x.shape[1], heads, dim))
        return super()._heads(x, heads, dim)

    def _normalize_key(self, x):
        import ttnn

        if "gdn_norm" in self.fusions:
            return ttnn.multiply(
                ttnn.rms_norm(x, epsilon=1e-6 / self.head_k_dim, compute_kernel_config=self.compute_kernel_config),
                self.head_k_dim**-0.5,
            )
        return super()._normalize_key(x)

    def _recurrent_step_fp32(self, q, k, v, decay_log, beta):
        import ttnn

        if "native_recurrence" in self.fusions:
            return ttnn.transformer.fused_recurrent_gated_delta_rule(
                q,
                k,
                v,
                decay_log,
                beta,
                initial_state=self.recurrent_state,
                output_final_state=True,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if "gdn_grouped" in self.fusions or "gdn_state_ternary" in self.fusions:
            grouped = "gdn_grouped" in self.fusions
            heads = self.num_heads if grouped else self.num_v_heads
            groups = self.num_v_heads // heads

            def column(x):
                x = ttnn.typecast(x, ttnn.float32)
                if not grouped:
                    x = ttnn.repeat_interleave(x, self.num_v_heads // self.num_heads, dim=2)
                x = ttnn.permute(x, (0, 2, 3, 1))
                return ttnn.reshape(x, (heads, 1, self.head_k_dim, 1))

            def scalar(x):
                return ttnn.reshape(ttnn.typecast(x, ttnn.float32), (heads, groups, 1, 1))

            def reduce(x):
                return ttnn.sum(
                    x,
                    dim=-2,
                    keepdim=True,
                    compute_kernel_config=self.compute_kernel_config,
                    fast_and_approximate_mode=False,
                )

            q = ttnn.multiply(column(q), self.head_k_dim**-0.5)
            k = column(k)
            v = ttnn.reshape(ttnn.typecast(v, ttnn.float32), (heads, groups, 1, self.head_v_dim))
            decay = scalar(ttnn.exp(decay_log, fast_and_approximate_mode=False))
            beta = scalar(beta)
            state = ttnn.reshape(self.recurrent_state, (heads, groups, self.head_k_dim, self.head_v_dim))
            decayed = ttnn.multiply(state, decay)
            memory = reduce(ttnn.multiply(decayed, k))
            delta = ttnn.multiply(ttnn.subtract(v, memory), beta)
            target = {"output_tensor": state} if "gdn_inplace" in self.fusions else {}
            updated = (
                ttnn.addcmul(decayed, k, delta, **target)
                if "gdn_state_ternary" in self.fusions
                else ttnn.add(decayed, ttnn.multiply(k, delta), **target)
            )
            output = reduce(ttnn.multiply(updated, q))
            return (
                ttnn.reshape(output, (1, 1, self.num_v_heads, self.head_v_dim)),
                ttnn.reshape(updated, (1, self.num_v_heads, self.head_k_dim, self.head_v_dim)),
            )
        return super()._recurrent_step_fp32(q, k, v, decay_log, beta)

    def forward(self, x, mode="recurrent", chunk_size=None, valid_len=None, residual=None):
        """Consume one real token or an aligned real-token chunk, retaining state."""
        import ttnn

        if len(x.shape) != 3 or x.shape[0] != 1 or x.shape[2] != self.hidden_size:
            raise ValueError("GDN expects a selected request [1,T,2048]")
        length = x.shape[1]
        if mode not in ("recurrent", "chunk") or length < 1:
            raise ValueError("GDN mode must be recurrent or chunk, with at least one token")
        if valid_len is not None and valid_len != length:
            raise ValueError("FunctionalDecoder must remove padding before the GDN state update")
        if length > 1 and (mode != "chunk" or length % 32):
            raise ValueError("The outer decoder must plan aligned chunks and single-token tails")
        mc = ttnn.DRAM_MEMORY_CONFIG
        mega = ttnn.linear(x, self.mega_weight, memory_config=mc, compute_kernel_config=self.compute_kernel_config)
        qkv_width = self.cfg.q_dim + self.cfg.k_dim + self.cfg.v_dim
        a_start, b_start, z_start = qkv_width, qkv_width + self.num_v_heads, qkv_width + 2 * self.num_v_heads

        def columns(tensor, start, end):
            return ttnn.to_layout(tensor[:, :, start:end], ttnn.TILE_LAYOUT, memory_config=mc)

        qkv = self._convolve(columns(mega, 0, qkv_width))
        if isinstance(qkv, tuple):
            q, k, v = qkv
        else:
            q = columns(qkv, 0, self.cfg.q_dim)
            k = columns(qkv, self.cfg.q_dim, self.cfg.q_dim + self.cfg.k_dim)
            v = columns(qkv, self.cfg.q_dim + self.cfg.k_dim, qkv_width)
        fused_norm = "kda_norm_decode" in self.fusions or (length > 1 and "kda_norm" in self.fusions)
        a = ttnn.typecast(columns(mega, a_start, b_start), ttnn.float32)
        beta = ttnn.typecast(
            ttnn.sigmoid(ttnn.typecast(columns(mega, b_start, z_start), ttnn.float32), memory_config=mc), ttnn.bfloat16
        )
        decay_log = ttnn.multiply(
            self.a_neg,
            (
                ttnn.add(a, self.dt_bias, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)])
                if "gdn_eltwise" in self.fusions
                else ttnn.softplus(ttnn.add(a, self.dt_bias, memory_config=mc), memory_config=mc)
            ),
            memory_config=mc,
        )
        if length > 1:
            eye, tril, ones, masks = self.chunk_constants
            # Explicit head-split inputs bypass the native flat FP32 normalizer.
            # The exact chunk op expands the 16 key heads to 32 value heads.
            q = self._normalize_key(self._heads(q, self.num_heads, self.head_k_dim))
            k = self._normalize_key(self._heads(k, self.num_heads, self.head_k_dim))
            if "gdn_flat_v" not in self.fusions:
                v = self._heads(v, self.num_v_heads, self.head_v_dim)
            output, state = ttnn.transformer.chunk_gated_delta_rule(
                q,
                k,
                v,
                decay_log,
                beta,
                initial_state=self.recurrent_state,
                output_final_state=True,
                chunk_size=32,
                output_head_major="gdn_head_major" in self.fusions,
                program_config=self.chunk_program_config,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=mc,
                eye=eye,
                tril=tril,
                ones=ones,
                masks=masks,
            )
            if "gdn_head_major" in self.fusions and not fused_norm:
                output = ttnn.reshape(ttnn.permute(output, (1, 0, 2)), (1, length, self.num_v_heads, self.head_v_dim))
            else:
                output = ttnn.to_layout(output, ttnn.TILE_LAYOUT, memory_config=mc)
        else:
            q = self._normalize_key(self._heads(q, self.num_heads, self.head_k_dim))
            k = self._normalize_key(self._heads(k, self.num_heads, self.head_k_dim))
            v = self._heads(v, self.num_v_heads, self.head_v_dim)
            output, state = self._recurrent_step_fp32(q, k, v, decay_log, beta)
        if length > 1 or "gdn_inplace" not in self.fusions:
            ttnn.copy(state, self.recurrent_state)
        # HF recurrence returns its activation dtype before the gated RMSNorm.
        output = ttnn.typecast(output, ttnn.bfloat16)
        if fused_norm:
            gate_flat = columns(mega, z_start, z_start + self.cfg.v_dim)
            if length == 1:
                output = ttnn.reshape(output, (self.num_v_heads, 1, self.head_v_dim))
                output = ttnn.pad(output, ((0, 0), (0, 31), (0, 0)), 0)
                gate_padded = ttnn.pad(gate_flat, ((0, 0), (0, 31), (0, 0)), 0)
            else:
                if "gdn_head_major" not in self.fusions:
                    output = ttnn.reshape(
                        ttnn.permute(output, (0, 2, 1, 3)), (self.num_v_heads, length, self.head_v_dim)
                    )
                gate_padded = gate_flat
            gated = ttnn.experimental.kda.sigmoid_gated_rms_norm(
                output,
                gate_padded,
                self.kda_norm_weight,
                self.num_v_heads,
                epsilon=self.norm_eps,
                compute_kernel_config=self.kda_norm_compute,
                memory_config=mc,
                output_dtype=ttnn.float32,
            )
            gated = ttnn.multiply(gated[:, :length, :], ttnn.typecast(gate_flat, ttnn.float32))
            gated = ttnn.typecast(gated, ttnn.bfloat16)
            if residual is not None:
                return self.residual_projection(gated, residual, self.compute_kernel_config, mc)
            return ttnn.linear(
                gated, self.output_weight, memory_config=mc, compute_kernel_config=self.compute_kernel_config
            )
        normalized = ttnn.rms_norm(
            ttnn.typecast(output, ttnn.float32),
            epsilon=self.norm_eps,
            memory_config=mc,
            compute_kernel_config=self.compute_kernel_config,
        )
        normalized = ttnn.multiply(ttnn.typecast(normalized, ttnn.bfloat16), self.norm_weight, memory_config=mc)
        gate = self._heads(columns(mega, z_start, z_start + self.cfg.v_dim), self.num_v_heads, self.head_v_dim)
        if "gdn_eltwise" in self.fusions:
            gated = ttnn.multiply(
                ttnn.typecast(gate, ttnn.float32),
                ttnn.typecast(normalized, ttnn.float32),
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                memory_config=mc,
            )
        else:
            gated = ttnn.multiply(
                ttnn.typecast(normalized, ttnn.float32),
                ttnn.silu(ttnn.typecast(gate, ttnn.float32), memory_config=mc),
                memory_config=mc,
            )
        gated = ttnn.typecast(gated, ttnn.bfloat16)
        gated = ttnn.reshape(gated, (1, length, self.cfg.v_dim))
        if residual is not None:
            return self.residual_projection(gated, residual, self.compute_kernel_config, mc)
        return ttnn.linear(
            gated, self.output_weight, memory_config=mc, compute_kernel_config=self.compute_kernel_config
        )


def _enable_indexed_rope(model, mesh):
    """Pack complete16-value exponent groups; both cache readers use this basis."""
    from dataclasses import replace

    import torch

    import ttnn

    def host(tensor):
        return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))

    def put(value, like):
        return ttnn.from_torch(
            value.contiguous(),
            device=mesh,
            dtype=like.dtype,
            layout=like.layout,
            memory_config=like.memory_config(),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    partial = torch.cat((torch.arange(16), torch.arange(32, 48), torch.arange(16, 32), torch.arange(48, 64)))
    perm = torch.cat((partial, torch.arange(64, 256)))
    attention = model.layer.attention
    # Match the qualified packing lifetime. Earlier release can affect allocation
    # placement; retaining the inputs restores about42us per traced decode here.
    # They are released when construction finishes; no extra runtime state stays.
    previous_allocations = attention.weights, attention.qkvg_weight
    norm_weights = {}
    for name in ("q_norm", "k_norm"):
        old = getattr(attention.weights, name)
        norm_weights[name] = put(host(old)[..., perm], old)
    attention.weights = replace(attention.weights, **norm_weights)
    old = attention.qkvg_weight
    value = host(old)
    columns = torch.arange(value.shape[-1])
    for start, heads in ((0, 16), (4096, 2)):
        columns[start : start + heads * 256] = (start + torch.arange(heads)[:, None] * 256 + perm).reshape(-1)
    attention.qkvg_weight = put(value[..., columns], old)
    for name in ("_rope_cos", "_rope_sin"):
        old = getattr(model, name)
        setattr(model, name, put(host(old)[..., partial], old))
    rotation = torch.zeros(1, 1, 32, 32)
    rotation[..., torch.arange(16), torch.arange(16, 32)] = 1
    rotation[..., torch.arange(16, 32), torch.arange(16)] = -1
    attention.rotation = ttnn.from_torch(
        rotation,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    del previous_allocations


class _ResidualProjection:
    def __init__(self, weight, device):
        import ttnn

        # Native addcmul requires weight/residual formats to agree. Upcasting
        # quantized full-attention weights preserves their numerical values.
        self.weight = ttnn.typecast(weight, ttnn.bfloat16)
        self.ones = ttnn.ones((1, 1, 2048), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def __call__(self, activation, residual, compute, memory):
        import ttnn

        result = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
            activation,
            self.weight,
            1.0,
            residual,
            self.ones,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute,
            memory_config=memory,
        )
        return result[:, : residual.shape[1], :]


class _ProjectionResidualLayer:
    """Single-device layer composition retaining the fused first residual result."""

    def __init__(self, original):
        self.__dict__.update(original.__dict__)
        weight = self.attention.weights.o_proj if self.is_full_attention else self.attention.output_weight
        self.attention.residual_projection = _ResidualProjection(weight, self.device)

    def forward(
        self,
        x,
        cos=None,
        sin=None,
        mode="decode",
        chunk_size=128,
        position_tensor=None,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        chunk_start_idx_tensor=None,
        valid_len=None,
    ):
        import ttnn
        from models.tt_transformers.tt.common import Mode

        if mode not in ("prefill", "decode"):
            raise ValueError("mode must be prefill or decode")
        norm_mode = Mode.PREFILL if mode == "prefill" else Mode.DECODE
        normalized = self.attention_norm(x, mode=norm_mode)
        if self.is_full_attention:
            h = self.attention.forward(
                normalized,
                cos,
                sin,
                position_tensor=position_tensor,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
                residual=x,
            )
        else:
            h = self.attention.forward(
                normalized,
                mode="chunk" if mode == "prefill" else "recurrent",
                chunk_size=chunk_size,
                valid_len=valid_len,
                residual=x,
            )
        ttnn.deallocate(normalized)
        ff_input = self.ffn_norm(h, mode=norm_mode)
        ff_output = self.feed_forward.forward(ff_input, mode=mode)
        ttnn.deallocate(ff_input)
        output = ttnn.add(h, ff_output)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_output)
        return output
