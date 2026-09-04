# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Joint rank-four Q/K prefill normalization with HybridNorm's numeric policy.

The chunk launch and sub-batch geometry below match HybridNormGDN. Calling its
complete _chunk_delta_rule would normalize Q/K twice, so reuse that launch
structure after exactly one shared l2_norm_ttnn call instead.
"""

import ttnn
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import l2_norm_ttnn

from . import linear_fusion_candidates as L


def _launch_hybrid_chunks(decoder, q_norm, k_norm, v, g, beta):
    """HybridNormGDN's launch/sub-batch path, receiving already normalized Q/K."""
    cfg = decoder.cfg
    batch, seq = q_norm.shape[0], q_norm.shape[1]
    nk, dk = cfg.linear_num_key_heads, cfg.linear_key_head_dim
    nv, dv = cfg.linear_num_value_heads, cfg.linear_value_head_dim
    eye, tril, ones, masks = decoder.w["gdn_const_tiles"]

    def launch(q_, k_, v_, g_, beta_, state_):
        return ttnn.transformer.chunk_gated_delta_rule(
            q_,
            k_,
            v_,
            g_,
            beta_,
            initial_state=state_,
            output_final_state=True,
            chunk_size=decoder.w["gdn_chunk_size"],
            use_qk_l2norm=False,
            output_head_major=True,
            eye=eye,
            tril=tril,
            ones=ones,
            masks=masks,
        )

    step = decoder.max_gdn_prefill_batch()
    if batch <= step:
        return launch(q_norm, k_norm, v, g, beta, decoder.recurrent_state)
    cores, states = [], []
    for start in range(0, batch, step):
        end = min(start + step, batch)
        q_s = ttnn.slice(q_norm, [start, 0, 0, 0], [end, seq, nk, dk])
        k_s = ttnn.slice(k_norm, [start, 0, 0, 0], [end, seq, nk, dk])
        v_s = ttnn.slice(v, [start, 0, 0], [end, seq, nv * dv])
        g_s = ttnn.slice(g, [start, 0, 0], [end, seq, nv])
        beta_s = ttnn.slice(beta, [start, 0, 0], [end, seq, nv])
        state_s = ttnn.slice(decoder.recurrent_state, [start, 0, 0, 0], [end, nv, dk, dv])
        core_s, final_s = launch(q_s, k_s, v_s, g_s, beta_s, state_s)
        for tensor in (q_s, k_s, v_s, g_s, beta_s, state_s):
            ttnn.deallocate(tensor)
        cores.append(core_s)
        states.append(final_s)
    result = ttnn.concat(cores, dim=0), ttnn.concat(states, dim=0)
    for tensor in (*cores, *states):
        ttnn.deallocate(tensor)
    return result


class JointPrefillNormGDN(L.HybridNormGDN):
    """Join 16 Q + 16 K heads before one BF16 RMSNorm and BF16 L2 scale."""

    def _chunk_delta_rule(self, q, k, v, g, beta):
        cfg = self.cfg
        batch, seq = int(q.shape[0]), int(q.shape[1])
        nk, dk = cfg.linear_num_key_heads, cfg.linear_key_head_dim
        if len(q.shape) != 3 or list(q.shape) != list(k.shape):
            raise ValueError("joint prefill normalization requires equal rank-three Q/K")
        if 2 * nk % 32 or dk % 32 or seq % 32:
            raise ValueError("joint prefill normalization requires tile-aligned joined heads, head width and sequence")
        joined = ttnn.concat([q, k], dim=2)
        heads = ttnn.reshape(joined, [batch, seq, 2 * nk, dk])
        # This is the same RMSNorm(eps/dk) -> BF16 multiply(dk**-0.5)
        # as HybridNorm's two calls. The last dimension remains128, not4096.
        normalized = l2_norm_ttnn(heads)
        # Drop the temporary reshape reference before releasing its input owner;
        # this covers both materialized and view reshapes without a double free.
        del heads
        ttnn.deallocate(joined)
        q_norm = ttnn.slice(normalized, [0, 0, 0, 0], [batch, seq, nk, dk], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k_norm = ttnn.slice(normalized, [0, 0, nk, 0], [batch, seq, 2 * nk, dk], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(normalized)
        # Keep rank4 Q/K: rank3 would trigger the chunk kernel's implicit L2
        # normalization even with use_qk_l2norm=False. V intentionally stays flat.
        result = _launch_hybrid_chunks(self, q_norm, k_norm, v, g, beta)
        ttnn.deallocate(q_norm)
        ttnn.deallocate(k_norm)
        return result


class JointPrefillArithmetic(L.HybridArithmetic):
    """Joint prefill normalization with the existing Arithmetic decode graph."""

    _chunk_delta_rule = JointPrefillNormGDN._chunk_delta_rule


class JointPrefillCombined(L.HybridCombinedGDN):
    """Joint prefill normalization plus the combined convolution/output graph."""

    _chunk_delta_rule = JointPrefillNormGDN._chunk_delta_rule


class JointPrefillConv(L.HybridConvOnly):
    """Joint prefill normalization plus only the existing KDA prefill conv."""

    _chunk_delta_rule = JointPrefillNormGDN._chunk_delta_rule


class JointPrefillKdaNorm(L.HybridKdaNorm):
    """Joint prefill normalization plus only the existing KDA output norm."""

    _chunk_delta_rule = JointPrefillNormGDN._chunk_delta_rule


class SeparateZSiluGDN(L.HybridArithmetic):
    """QKV/A/B peer packing plus a separate Z projection with a SiLU epilogue."""

    fuse_z_silu = True

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        import torch

        decoder = super().from_state_dict(state_dict, **kwargs)
        if decoder.is_full_attention:
            return decoder
        if kwargs.get("dtype", ttnn.bfloat16) != ttnn.bfloat16:
            raise ValueError("separate Z SiLU probe preserves the BF16 projection policy")

        def upload(value):
            return ttnn.from_torch(
                value.T.contiguous(),
                device=decoder.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(decoder.device),
            )

        packed = torch.cat([state_dict[f"linear_attn.in_proj_{field}.weight"] for field in ("qkv", "a", "b")])
        old_packed = decoder.w["gdn_packed"]
        decoder.w["gdn_packed"] = upload(packed)
        ttnn.deallocate(old_packed)
        decoder.w["gdn_z_epilogue"] = upload(state_dict["linear_attn.in_proj_z.weight"])
        return decoder

    def _gdn_project(self, x):
        cfg = self.cfg
        packed = ttnn.linear(x, self.w["gdn_packed"], compute_kernel_config=self.compute_kernel_config)
        qkv = L._field(packed, 0, cfg.conv_dim)
        a = L._field(packed, cfg.conv_dim, cfg.conv_dim + cfg.linear_num_value_heads)
        b = L._field(packed, cfg.conv_dim + cfg.linear_num_value_heads, cfg.conv_dim + 2 * cfg.linear_num_value_heads)
        ttnn.deallocate(packed)
        grid = self.device.compute_with_storage_grid_size()
        z_args = {"core_grid": ttnn.CoreGrid(x=grid.x, y=grid.y)}
        if self.fuse_z_silu:
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
        cfg = self.cfg
        normed = ttnn.rms_norm(core, weight=self.w["gdn_norm"], epsilon=cfg.norm_eps)
        ttnn.deallocate(core)
        heads = ttnn.reshape(normed, [batch, cfg.linear_num_value_heads, seq, cfg.linear_value_head_dim])
        combined = ttnn.experimental.nlp_concat_heads(heads) if seq > 1 else ttnn.permute(heads, (0, 2, 1, 3))
        ttnn.deallocate(normed)
        merged = ttnn.reshape(combined, [batch, seq, cfg.linear_v_dim])
        if self.fuse_z_silu:
            gated = ttnn.multiply(merged, z)
        else:
            activated = ttnn.silu(z)
            gated = ttnn.multiply(merged, activated)
            ttnn.deallocate(activated)
        ttnn.deallocate(combined)
        result = ttnn.linear(gated, self.w["gdn_out"], compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(gated)
        return result


class SeparateZSiluControlGDN(SeparateZSiluGDN):
    """Same QKVAB/Z projections and Z core grid; keep standalone output SiLU."""

    fuse_z_silu = False


class JointPrefillSeparateZSiluGDN(SeparateZSiluGDN):
    """Compose the independent joint-QK normalization and separate-Z probes."""

    _chunk_delta_rule = JointPrefillNormGDN._chunk_delta_rule
