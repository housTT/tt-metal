# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Mode-specific output-normalization probes with decode-only gate fusion."""

import ttnn

from . import combined_fusion_candidates as C
from . import gdn_joint_prefill_candidates as J
from . import linear_fusion_candidates as L
from .fir_fusion_candidates import SharedFIRRows


class PlainNormFIRGDN(C.JointGateGDN):
    """JointGateGDN with shared FIR rows; preserve its plain output norm."""

    _causal_conv = SharedFIRRows._causal_conv
    _write_conv_state = SharedFIRRows._write_conv_state


class ModeNormGDN(PlainNormFIRGDN):
    """KDA norm consumes raw prefill Z; decode consumes matmul-SiLU Z."""

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        decoder = super().from_state_dict(state_dict, **kwargs)
        if not decoder.is_full_attention:
            decoder.w["kda_norm_vector"] = ttnn.from_torch(
                state_dict["linear_attn.norm.weight"].float().reshape(-1).contiguous(),
                device=decoder.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(decoder.device),
            )
        return decoder

    def _gdn_project(self, x):
        cfg = self.cfg
        # SeparateZSiluGDN's cooperative setup already packed QKV/A/B into
        # gdn_packed and uploaded the independent gdn_z_epilogue matrix.
        packed = ttnn.linear(x, self.w["gdn_packed"], compute_kernel_config=self.compute_kernel_config)
        qkv = L._field(packed, 0, cfg.conv_dim)
        a = L._field(packed, cfg.conv_dim, cfg.conv_dim + cfg.linear_num_value_heads)
        b = L._field(packed, cfg.conv_dim + cfg.linear_num_value_heads, cfg.conv_dim + 2 * cfg.linear_num_value_heads)
        ttnn.deallocate(packed)
        # Keep SeparateZSiluControlGDN's full-grid program selection in both
        # modes; only the activation location changes with the output norm.
        grid = self.device.compute_with_storage_grid_size()
        z_args = {"core_grid": ttnn.CoreGrid(x=grid.x, y=grid.y)}
        # Public prefill physically pads every chunk to at least128 tokens.
        # Only decode reaches this boundary with time1; no runtime data is read.
        if int(x.shape[1]) == 1:
            z_args["activation"] = "silu"
        z = ttnn.linear(
            x,
            self.w["gdn_z_epilogue"],
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
            **z_args,
        )
        return qkv, z, a, b

    def _gdn_out_head_major(self, core, z, batch, seq):
        if seq > 1:
            # This branch directly computes RMSNorm(core)*sigmoid(raw_z)*raw_z.
            # Avoid KdaNormGDN's decode branch, whose super() expects its own MRO.
            return L.KdaNormGDN._gdn_out_head_major(self, core, z, batch, seq)
        # Decode Z already contains its SiLU epilogue. The inherited true
        # fuse_z_silu flag makes this method use plain multiply, exactly once.
        return J.SeparateZSiluGDN._gdn_out_head_major(self, core, z, batch, seq)
