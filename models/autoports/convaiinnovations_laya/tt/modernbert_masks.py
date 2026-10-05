# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.autoports.convaiinnovations_laya.tt.model_config import (
    FULL_ATTENTION,
    MASK_DTYPE,
    MASK_NEG,
    SLIDING_ATTENTION,
)


def band_mask(seq_len: int, half_window: int, neg: float = MASK_NEG) -> torch.Tensor:
    idx = torch.arange(seq_len)
    band = (idx[None, :] - idx[:, None]).abs() <= half_window
    return torch.zeros(seq_len, seq_len, dtype=torch.float32).masked_fill(~band, neg)[None, None]


def pad_row_host(attention_mask: torch.Tensor, neg: float = MASK_NEG) -> torch.Tensor:
    pad = torch.zeros(attention_mask.shape, dtype=torch.float32)
    return pad.masked_fill(attention_mask == 0, neg)[:, None, None, :]


def build_masks_host(config, attention_mask: torch.Tensor, seq_len: int, half_window=None, neg: float = MASK_NEG):
    """Reference (B,1,S,S) additive masks on host, both layer types, same arithmetic as the device path."""
    half = config.local_attention // 2 if half_window is None else half_window
    pad = pad_row_host(attention_mask, neg)
    batch = attention_mask.shape[0]
    sliding = band_mask(seq_len, half, neg) + pad
    full = torch.zeros(batch, 1, seq_len, seq_len, dtype=torch.float32) + pad
    return {SLIDING_ATTENTION: sliding, FULL_ATTENTION: full}


def _upload(t, device, mesh_mapper, dtype):
    kw = {"device": device, "memory_config": ttnn.DRAM_MEMORY_CONFIG}
    if mesh_mapper is not None:
        kw["mesh_mapper"] = mesh_mapper
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, **kw)


class TtnnMaskBuilder:
    """Static per-bucket band and zero tensors in DRAM; per call the pad row is added on device."""

    def __init__(self, config, device, seq_len, batch_size, mesh_mapper=None, dtype=MASK_DTYPE, half_window=None):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device
        self.mesh_mapper = mesh_mapper
        half = config.local_attention // 2 if half_window is None else half_window
        band = band_mask(seq_len, half).expand(batch_size, 1, seq_len, seq_len).contiguous()
        zeros = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.float32)
        self.band = _upload(band, device, mesh_mapper, dtype)
        self.zeros = _upload(zeros, device, mesh_mapper, dtype)

    def pad_row_tensor(self, attention_mask: torch.Tensor):
        """Host tensor (B,1,1,S) bf16 TILE, ready for copy_host_to_device_tensor or from_torch."""
        return ttnn.from_torch(pad_row_host(attention_mask), dtype=self.dtype, layout=ttnn.TILE_LAYOUT)

    def allocate_pad_row(self):
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape((self.batch_size, 1, 1, self.seq_len)),
            self.dtype,
            ttnn.TILE_LAYOUT,
            self.device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

    def upload_pad_row(self, attention_mask: torch.Tensor):
        return _upload(pad_row_host(attention_mask), self.device, self.mesh_mapper, self.dtype)

    def build(self, pad_row):
        """pad_row: device (B,1,1,S) tensor. Returns {layer_type: (B,1,S,S) DRAM mask}."""
        sliding = ttnn.add(self.band, pad_row, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        full = ttnn.add(self.zeros, pad_row, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return {SLIDING_ATTENTION: sliding, FULL_ATTENTION: full}

    def deallocate(self):
        for t in (self.band, self.zeros):
            if t.is_allocated():
                ttnn.deallocate(t)


def deallocate_masks(masks):
    for m in masks.values():
        if m is not None and m.is_allocated():
            ttnn.deallocate(m)
