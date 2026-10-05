# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.autoports.convaiinnovations_laya.tt.model_config import DEFAULT_POLICY, DEFAULT_PORT, FULL_ATTENTION
from models.autoports.convaiinnovations_laya.tt.modernbert_attention import TtnnModernBertAttention

LAYER_NORM_EPS = 1e-5


class TtnnHeadLayer:
    """nn.TransformerEncoderLayer(norm_first=True, ReLU): x + out_proj(SDPA(in_proj(LN1(x)))), x + linear2(relu(linear1(LN2(x))))."""

    def __init__(self, parameters, config, plan, device, policy=DEFAULT_POLICY):
        self.p = parameters
        self.compute = policy.compute_config(device, "head")
        self.norm_config = policy.compute_config(device, "norm")
        self.act_dtype = policy.act_dtype
        attn_params = {
            "Wqkv": parameters["in_proj"]["weight"],
            "Wqkv_bias": parameters["in_proj"]["bias"],
            "Wo": parameters["out_proj"]["weight"],
            "Wo_bias": parameters["out_proj"]["bias"],
        }
        self.attn = TtnnModernBertAttention(attn_params, config, FULL_ATTENTION, plan, device, policy, group="head")
        self._down = {"compute_kernel_config": self.compute}
        if plan.down_core_grid is not None:
            self._down["core_grid"] = plan.down_core_grid
        self.relu = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)

    def _norm(self, x, which):
        return ttnn.layer_norm(
            x,
            weight=self.p[which]["weight"],
            bias=self.p[which]["bias"],
            epsilon=LAYER_NORM_EPS,
            compute_kernel_config=self.norm_config,
        )

    def __call__(self, x, full_mask):
        normed = self._norm(x, "norm1")
        attn_out = self.attn(normed, None, full_mask)
        ttnn.deallocate(normed)
        h = ttnn.add(x, attn_out)
        ttnn.deallocate(attn_out)

        normed2 = self._norm(h, "norm2")
        hidden = ttnn.linear(
            normed2,
            self.p["linear1"]["weight"],
            bias=self.p["linear1"]["bias"],
            activation=self.relu,
            compute_kernel_config=self.compute,
            dtype=self.act_dtype,
        )
        ttnn.deallocate(normed2)
        ff = ttnn.linear(hidden, self.p["linear2"]["weight"], bias=self.p["linear2"]["bias"], **self._down)
        ttnn.deallocate(hidden)
        out = ttnn.add(h, ff)
        ttnn.deallocate(h)
        ttnn.deallocate(ff)
        return out


class TtnnLayaHead:
    """type_emb add, two head layers over the full pad mask, the scorer over every position, the CLS rows."""

    def __init__(self, parameters, config, plan, device, policy=DEFAULT_POLICY, port=DEFAULT_PORT):
        self.p = parameters
        self.config = config
        self.plan = plan
        self.policy = policy
        self.compute = policy.compute_config(device, "scorer")
        self.norm_config = policy.compute_config(device, "norm")
        self.layers = [TtnnHeadLayer(lp, config, plan, device, policy) for lp in parameters["layers"]]
        self.scorer_fp32 = policy.scorer_fp32_out
        self.gelu = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, int(policy.scorer_gelu_approx))
        self._dense = {"compute_kernel_config": self.compute}
        if plan.down_core_grid is not None:
            self._dense["core_grid"] = plan.down_core_grid

    def add_type_embedding(self, hidden, qtype_ids):
        emb = ttnn.embedding(qtype_ids, self.p["type_emb"], layout=ttnn.TILE_LAYOUT)
        out = ttnn.add(hidden, emb)
        ttnn.deallocate(emb)
        ttnn.deallocate(hidden)
        return out

    def run_layers(self, h, full_mask):
        for layer in self.layers:
            h = layer(h, full_mask)
        return h

    def scorer(self, h):
        sc = self.p["scorer"]
        normed = ttnn.layer_norm(
            h,
            weight=sc["norm"]["weight"],
            bias=sc["norm"]["bias"],
            epsilon=LAYER_NORM_EPS,
            compute_kernel_config=self.norm_config,
        )
        dense = ttnn.linear(
            normed, sc["dense"]["weight"], bias=sc["dense"]["bias"], activation=self.gelu, **self._dense
        )
        ttnn.deallocate(normed)
        if self.scorer_fp32:
            dense32 = ttnn.typecast(dense, ttnn.float32)
            ttnn.deallocate(dense)
            logits = ttnn.linear(
                dense32,
                sc["out"]["weight"],
                bias=sc["out"]["bias"],
                dtype=ttnn.float32,
                compute_kernel_config=self.compute,
            )
            ttnn.deallocate(dense32)
        else:
            logits = ttnn.linear(dense, sc["out"]["weight"], bias=sc["out"]["bias"], compute_kernel_config=self.compute)
            ttnn.deallocate(dense)
        return logits

    def cls_rows(self, h):
        b = self.plan.batch_size
        return ttnn.slice(h, [0, 0, 0], [b, 32, self.config.hidden_size])

    def __call__(self, hidden, qtype_ids, full_mask):
        """hidden: encoder output (B,S,H). Returns (logits (B,S,1) fp32 tiled, cls (B,32,H) bf16 tiled, h)."""
        h = self.add_type_embedding(hidden, qtype_ids)
        h = self.run_layers(h, full_mask)
        logits = self.scorer(h)
        cls = self.cls_rows(h)
        return logits, cls, h
