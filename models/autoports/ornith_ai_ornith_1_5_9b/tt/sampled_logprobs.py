# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TT raw sampled-token logprobs for Ornith's vocabulary-sharded logits."""

import torch

import ttnn


class SampledRawLogprobs:
    """A stable 32-row output; call-scoped reductions never survive replay.

    The caller provides raw logits before penalties, temperature, or filtering,
    and the selected global indices from the unchanged device sampler. Only
    those 32 logprobs cross to the host. Positive top-logprob alternatives are
    deliberately outside this capability.
    """

    def __init__(self, mesh, ccl, padded_vocab_size):
        self.mesh = mesh
        self.ccl = ccl
        self.tp = mesh.get_num_devices()
        if tuple(mesh.shape) != (1, 4) or padded_vocab_size != 262144:
            raise ValueError("Sampled raw logprobs currently require Ornith TP4 vocabulary geometry")
        self.local_vocab = padded_vocab_size // self.tp
        self.output = ttnn.from_torch(
            torch.zeros(1, 1, 32, 1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.rank = ttnn.from_torch(
            torch.arange(self.tp).float().reshape(1, self.tp, 1, 1).expand(1, self.tp, 32, 1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 1), mesh_shape=(1, self.tp)),
        )

    def prepare(self, logits, sampler):
        """Mask the exact model vocabulary and copy before sampler mutation."""
        if tuple(logits.shape) != (1, 1, 32, self.local_vocab):
            raise ValueError("Raw logprob logits must use the canonical B32 vocabulary shard")
        masked = sampler._mask_invalid_vocab_logits(logits)
        try:
            # FP32 protects global sum-exp from BF16 accumulation rounding.
            # clone also makes already-FP32 inputs independent of the sampler.
            return ttnn.clone(masked, dtype=ttnn.float32)
        finally:
            if masked is not logits:
                ttnn.deallocate(masked)

    def calculate(self, raw, sampled_indices):
        temporaries = []

        def keep(value):
            temporaries.append(value)
            return value

        def gather(value):
            # No persistent buffer key: these collectives have call-scoped
            # ownership, just like their reductions and selected-logit scratch.
            return keep(self.ccl.line_all_gather(value, dim=1))

        try:
            local_max = keep(ttnn.max(raw, dim=3, keepdim=True))
            global_max = keep(ttnn.max(gather(local_max), dim=1, keepdim=True))
            centered = keep(ttnn.subtract(raw, global_max))
            exponential = keep(ttnn.exp(centered, fast_and_approximate_mode=False))
            local_sum = keep(ttnn.sum(exponential, dim=3, keepdim=True))
            global_sum = keep(ttnn.sum(gather(local_sum), dim=1, keepdim=True))
            normalizer = keep(ttnn.log(global_sum, fast_and_approximate_mode=False))

            # Convert before division: UINT32 division is not supported by the
            # shared calculator either. Global vocabulary indices fit FP32.
            indices_rm = keep(ttnn.to_layout(sampled_indices, ttnn.ROW_MAJOR_LAYOUT))
            indices_rows = keep(ttnn.reshape(indices_rm, (1, 1, 32, 1)))
            indices_tile = keep(ttnn.to_layout(indices_rows, ttnn.TILE_LAYOUT))
            indices_float = keep(ttnn.typecast(indices_tile, ttnn.float32))
            owners = keep(ttnn.div(indices_float, self.local_vocab, rounding_mode="floor"))
            local_float = keep(ttnn.remainder(indices_float, self.local_vocab))
            local_indices = keep(ttnn.typecast(local_float, ttnn.uint32))
            local_selected = keep(ttnn.gather(centered, dim=3, index=local_indices))
            owner_mask = keep(ttnn.eq(owners, self.rank))
            # where avoids 0 * -inf on the last chip's invalid padded tail.
            selected = keep(ttnn.where(owner_mask, local_selected, 0.0))
            global_selected = keep(ttnn.sum(gather(selected), dim=1, keepdim=True))
            ttnn.subtract(global_selected, normalizer, output_tensor=self.output)
            return self.output
        finally:
            # Views can share allocations, so explicit release is restricted to
            # distinct addresses and runs after the final consumer is queued.
            seen = set()
            borrowed = {raw.buffer_address(), sampled_indices.buffer_address()}
            owned = [(tensor, tensor.buffer_address()) for tensor in reversed(temporaries)]
            for tensor, address in owned:
                if address not in seen and address not in borrowed:
                    ttnn.deallocate(tensor)
                    seen.add(address)

    def release(self):
        for name in ("output", "rank"):
            tensor = getattr(self, name)
            if tensor is not None:
                ttnn.deallocate(tensor)
                setattr(self, name, None)
