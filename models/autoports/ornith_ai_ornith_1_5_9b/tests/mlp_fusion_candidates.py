# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MLP graph probes with a real matmul epilogue and a matched program control.

No geometry sweep: the paired 2048-token and decode programs are copied from
the existing functional profiler's automatic gate-projection configurations.
Other shapes use the normal full-device-grid program selector. Every matmul
retains BF16 output and the decoder's HiFi4/FP32 accumulation configuration.
"""

import ttnn

from .fusion_baseline import FusionBaseline as FusedDecoder


class _MLPCandidate(FusedDecoder):
    def _activate_mlp(self, ff_in, mode):
        gate = ttnn.linear(ff_in, self.w["gate_proj"], compute_kernel_config=self.compute_kernel_config)
        up = ttnn.linear(ff_in, self.w["up_proj"], compute_kernel_config=self.compute_kernel_config)
        return ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])

    def _block(self, x, *, mode, logical_len=None, page_table=None, chunk_start_idx=0, current_pos=None, rot_idxs=None):
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
        h = ttnn.add(x, mixed)
        ttnn.deallocate(mixed)
        ff_in = self._norm(h, self.w["ff_norm"])
        activated = self._activate_mlp(ff_in, mode)
        ttnn.deallocate(ff_in)
        ff_out = ttnn.linear(activated, self.w["down_proj"], compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(activated)
        out = ttnn.add(h, ff_out)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_out)
        return out


class MatmulSiluEpilogue(_MLPCandidate):
    """SiLU runs in the gate matmul's PACK epilogue, then one plain multiply."""

    fuse_silu = True

    def _gate_matmul_kwargs(self, ff_in):
        grid = self.device.compute_with_storage_grid_size()
        activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if self.fuse_silu else None
        shape = list(ff_in.shape)
        width = int(self.w["gate_proj"].shape[-1])
        # Exact automatic programs from functional_decoder/tracy/full_attention:
        # prefill_ops.csv and decode_ops.csv, BF16 [M,4096] @ [4096,12288].
        # Explicit program activation is sufficient: do not pass activation=
        # as well, which would add the wrapper's separate unary operation.
        if (grid.x, grid.y) == (11, 10) and int(shape[-1]) == 4096 and width == 12288:
            if int(shape[-2]) == 1:
                return {
                    "program_config": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=(11, 10),
                        in0_block_w=2,
                        out_subblock_h=1,
                        out_subblock_w=4,
                        out_block_h=1,
                        out_block_w=4,
                        per_core_M=1,
                        per_core_N=4,
                        fuse_batch=False,
                        fused_activation=activation,
                        mcast_in0=True,
                    )
                }
            if int(shape[-2]) == 2048:
                return {
                    "program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(11, 10),
                        in0_block_w=1,
                        out_subblock_h=1,
                        out_subblock_w=1,
                        out_block_h=7,
                        out_block_w=7,
                        per_core_M=7,
                        per_core_N=35,
                        transpose_mcast=False,
                        fused_activation=activation,
                        fuse_batch=False,
                    )
                }
        # A core_grid causes create_matmul_program_config to carry activation
        # into the selected multicast program. The control uses this same grid
        # and selector with activation disabled, so this is no tuning sweep.
        kwargs = {"core_grid": ttnn.CoreGrid(x=grid.x, y=grid.y)}
        if self.fuse_silu:
            kwargs["activation"] = "silu"
        return kwargs

    def _activate_mlp(self, ff_in, mode):
        gate = ttnn.linear(
            ff_in,
            self.w["gate_proj"],
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
            **self._gate_matmul_kwargs(ff_in),
        )
        up = ttnn.linear(ff_in, self.w["up_proj"], compute_kernel_config=self.compute_kernel_config)
        if self.fuse_silu:
            return ttnn.multiply(gate, up)
        return ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])


class MatmulSiluEpilogueControl(MatmulSiluEpilogue):
    """Identical gate matmul program; the binary op retains its SiLU activation."""

    fuse_silu = False


class PackedPrefillSeparateDecode(_MLPCandidate):
    """One gate/up projection for prefill; separate projections for decode."""

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        import torch

        decoder = super().from_state_dict(state_dict, **kwargs)
        packed = torch.cat([state_dict["mlp.gate_proj.weight"], state_dict["mlp.up_proj.weight"]])
        decoder.w["gate_up"] = ttnn.from_torch(
            packed.T.contiguous(),
            device=decoder.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(decoder.device),
        )
        return decoder

    def _activate_mlp(self, ff_in, mode):
        if mode != "prefill":
            return super()._activate_mlp(ff_in, mode)
        gate_up = ttnn.linear(ff_in, self.w["gate_up"], compute_kernel_config=self.compute_kernel_config)
        gate, up = ttnn.chunk(gate_up, 2, dim=-1)
        return ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])


class PackedPrefillEpilogueDecode(PackedPrefillSeparateDecode, MatmulSiluEpilogue):
    """Combine packed prefill with a true gate SiLU epilogue during decode."""
