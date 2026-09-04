# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Coherent combinations after isolated controls; selected only by explicit test env."""

import ttnn

from ..tt.functional_decoder import _slice_owned
from . import gdn_joint_prefill_candidates as J
from . import linear_fusion_candidates as L
from .fir_fusion_candidates import SharedFIRRows


class JointGateGDN(J.SeparateZSiluGDN):
    _gdn_decode_heads = L.JointQKNormBeforeRepeat._gdn_decode_heads
    _delta_rule_step = L.JointQKNormGDN._delta_rule_step

    def _gdn_gates_projected(self, a_raw, b_raw, logical_len, seq_len):
        if seq_len != 1:
            return L.FlatGDN._gdn_gates_projected(self, a_raw, b_raw, logical_len, seq_len)
        return L._decode_gate_fusion(self, a_raw, b_raw, fold_a_cast=True, beta_chain=True)


class AllModeGateGDN(JointGateGDN):
    def _gdn_gates_projected(self, a_raw, b_raw, logical_len, seq_len):
        beta, g = L._decode_gate_fusion(self, a_raw, b_raw, fold_a_cast=True, beta_chain=True)
        if logical_len < seq_len:
            ramp, owned = _slice_owned(self.w["pos_ramp"], [0, 0, 0], [1, seq_len, 1])
            keep = ttnn.typecast(ttnn.lt(ramp, float(logical_len)), ttnn.float32)
            if owned:
                ttnn.deallocate(ramp)
            masked_beta, masked_g = ttnn.multiply(beta, keep), ttnn.multiply(g, keep)
            for tensor in (beta, g, keep):
                ttnn.deallocate(tensor)
            beta, g = masked_beta, masked_g
        return beta, g


class JointGateFIRGDN(AllModeGateGDN):
    _causal_conv = SharedFIRRows._causal_conv
    _write_conv_state = SharedFIRRows._write_conv_state
