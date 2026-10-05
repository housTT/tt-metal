# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.autoports.convaiinnovations_laya.tt.model_config import DEFAULT_POLICY, DEFAULT_PORT, bucket_plan
from models.autoports.convaiinnovations_laya.tt.modernbert_embeddings import TtnnModernBertEmbeddings
from models.autoports.convaiinnovations_laya.tt.modernbert_layer import TtnnModernBertEncoderLayer
from models.autoports.convaiinnovations_laya.tt.modernbert_rope import TtnnModernBertRotary


class TtnnModernBertModel:
    """Encoder for one (batch, seq) bucket: embeddings, the layer stack and the final norm."""

    def __init__(
        self,
        parameters,
        config,
        device,
        seq_len,
        batch_size=1,
        policy=DEFAULT_POLICY,
        port=DEFAULT_PORT,
        mesh_mapper=None,
        layers=None,
    ):
        self.config = config
        self.device = device
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.eps = config.norm_eps
        self.policy = policy
        self.port = port
        self.plan = bucket_plan(device, config, batch_size, seq_len, policy, port)
        self.embeddings = TtnnModernBertEmbeddings(parameters["embeddings"], config, device, policy)
        self.rotary = TtnnModernBertRotary(
            config,
            device,
            seq_len,
            batch_size=batch_size,
            mesh_mapper=mesh_mapper,
            port=port,
            attention_memory=self.plan.attention_memory,
        )
        indices = list(range(config.num_hidden_layers)) if layers is None else list(layers)
        self.layers = [
            TtnnModernBertEncoderLayer(parameters["layers"][i], config, i, self.plan, device, policy, port)
            for i in indices
        ]
        self.final_norm = parameters["final_norm"]
        self.norm_config = policy.compute_config(device, "norm")
        shard = self.plan.mlp_shard
        self.residual_fp32 = policy.residual_fp32
        self.resident = shard is not None and shard.norm is not None and port.resident_residual and not self.residual_fp32

    def _check_shape(self, input_ids):
        got = tuple(input_ids.shape)[-2:]
        want = (self.batch_size, self.seq_len)
        if got != want:
            raise ValueError(f"model was built for (batch, seq) {want}, got {got}")

    def run_layers(self, hidden, masks, layer_hook=None):
        if self.resident:
            sharded = ttnn.to_memory_config(hidden, self.plan.mlp_shard.hidden_memory)
            ttnn.deallocate(hidden)
            hidden = sharded
        for layer in self.layers:
            hidden = layer(hidden, self.rotary, masks[layer.layer_type])
            if layer_hook is not None:
                layer_hook(layer.layer_idx, hidden)
        if self.resident:
            interleaved = ttnn.to_memory_config(hidden, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(hidden)
            hidden = interleaved
        return hidden

    def __call__(self, input_ids, masks, layer_hook=None, final_norm=True):
        """input_ids: ttnn uint32 (B, S) ROW_MAJOR on device. masks: {layer_type: (B,1,S,S) DRAM mask}."""
        self._check_shape(input_ids)
        hidden = self.embeddings(input_ids)
        if self.residual_fp32:
            h32 = ttnn.typecast(hidden, ttnn.float32)
            ttnn.deallocate(hidden)
            hidden = h32
        hidden = self.run_layers(hidden, masks, layer_hook)
        if not final_norm:
            return hidden
        out = ttnn.layer_norm(hidden, weight=self.final_norm, epsilon=self.eps, compute_kernel_config=self.norm_config)
        ttnn.deallocate(hidden)
        if self.residual_fp32:
            out16 = ttnn.typecast(out, ttnn.bfloat16)
            ttnn.deallocate(out)
            out = out16
        return out

    def deallocate(self):
        self.rotary.deallocate()
