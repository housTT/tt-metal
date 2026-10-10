# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated real-weight control for shared down + sigmoid gate + routed addition.

Only this experiment process changes its constructor. The production defaults
remain untouched until the candidate earns selection by correctness and latency.
"""

from ..tt.fused_decoder import FusedDecoder
from . import run_layer


class _SharedEpilogue:
    def __init__(self, original, mesh):
        import ttnn

        self.__dict__.update(original.__dict__)
        # The native epilogue requires base and weight formats to match.
        self.down = ttnn.typecast(self.mlp.weights.w2, ttnn.bfloat16)
        self.ones = ttnn.ones((1, 1, 1, 2048), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def forward(self, x, base):
        import ttnn

        length = x.shape[2]
        compute = self.mlp.compute_kernel_config_decode
        gate = ttnn.linear(x, self.gate_weight)
        gate = ttnn.multiply(gate, self.ones, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        hidden = ttnn.experimental.minimal_matmul(
            x, self.packed, fuse_swiglu=True, compute_kernel_config=compute, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        output = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
            ttnn.reshape(hidden, (1, length, 512)),
            self.down,
            1.0,
            ttnn.reshape(base, (1, length, 2048)),
            ttnn.reshape(gate, (1, length, 2048)),
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return ttnn.reshape(output[:, :length, :], (1, 1, length, 2048))


class _SharedEpilogueMoE:
    def __init__(self, original, mesh):
        self.__dict__.update(original.__dict__)
        self.shared = _SharedEpilogue(self.shared, mesh)

    def forward(self, x, mode="decode"):
        routing = self.router(x)
        routed = self.experts(x, routing, mode=mode)
        return self.shared.forward(x, routed)


def main():
    original = FusedDecoder.from_state_dict.__func__

    def build(cls, *args, **kwargs):
        kwargs["fusions"] = tuple(f for f in (kwargs.get("fusions") or cls.DEFAULT_FUSIONS) if f != "indexed_rope")
        model = original(cls, *args, **kwargs)
        model.layer.feed_forward.module = _SharedEpilogueMoE(model.layer.feed_forward.module, kwargs["mesh_device"])
        model.fusions = (*model.fusions, "shared_epilogue_trial")
        return model

    FusedDecoder.from_state_dict = classmethod(build)
    run_layer.main()


if __name__ == "__main__":
    main()
