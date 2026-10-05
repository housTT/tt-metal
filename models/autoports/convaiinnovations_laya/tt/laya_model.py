# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import torch

import ttnn
from models.autoports.convaiinnovations_laya.tt.model_config import (
    DEFAULT_POLICY,
    DEFAULT_PORT,
    FULL_ATTENTION,
    MASK_DTYPE,
    ROW_BUCKETS,
    SEQ_BUCKETS,
    describe_plan,
    pick_bucket,
)
from models.autoports.convaiinnovations_laya.tt.laya_head import TtnnLayaHead
from models.autoports.convaiinnovations_laya.tt.modernbert_masks import TtnnMaskBuilder, pad_row_host
from models.autoports.convaiinnovations_laya.tt.modernbert_model import TtnnModernBertModel
from models.autoports.convaiinnovations_laya.tt.weights import (
    deallocate_weights,
    load_state_dict,
    prepare_head_weights,
    prepare_weights,
)

os.environ.setdefault("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES", "0")


@dataclass
class HostInputs:
    """Host-side tensors for one bucket; the same objects are reused for every call of that bucket."""

    input_ids: object
    pad_row: object
    qtype: object
    n_real: int


@dataclass
class Bucket:
    batch_size: int
    seq_len: int
    encoder: TtnnModernBertModel
    head: TtnnLayaHead
    masks: TtnnMaskBuilder
    input_ids: object
    pad_row: object
    qtype: object
    trace_id: Optional[int] = None
    trace_outputs: Optional[Tuple[object, object]] = None
    addresses: Dict[str, int] = field(default_factory=dict)

    def input_addresses(self) -> Dict[str, int]:
        return {
            "input_ids": self.input_ids.buffer_address(),
            "pad_row": self.pad_row.buffer_address(),
            "qtype": self.qtype.buffer_address(),
        }


def open_device(device_id: int = 0, l1_small_size: int = 79104, trace_region_size: int = 0, num_command_queues=1):
    os.environ.setdefault("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES", "0")
    return ttnn.open_device(
        device_id=device_id,
        l1_small_size=l1_small_size,
        trace_region_size=trace_region_size,
        num_command_queues=num_command_queues,
    )


class TtnnLayaModel:
    """Device side of Laya: encoder plus head on one device, one sub-model per (rows, seq) bucket.

    Inputs per call: input_ids (n, L) long, attention_mask (n, L) long (1 real, 0 pad), qtype (n,) long.
    Rows are padded to the row bucket with pad_id, an all-zero attention row and qtype 0; the sequence is
    padded to the seq bucket. Outputs: scorer logits (n, L) float32 for every position (gather at the
    marker positions on host) and the CLS hidden state (n, hidden) float32 after the head layers.
    """

    def __init__(
        self,
        device,
        config,
        state_dict=None,
        weights_path=None,
        policy=DEFAULT_POLICY,
        port=DEFAULT_PORT,
        row_buckets: Sequence[int] = ROW_BUCKETS,
        seq_buckets: Sequence[int] = SEQ_BUCKETS,
        mesh_mapper=None,
        intermediate_pads=None,
    ):
        self.device = device
        self.config = config
        self.policy = policy
        self.port = port
        self.row_buckets = tuple(sorted(row_buckets))
        self.seq_buckets = tuple(sorted(seq_buckets))
        self.mesh_mapper = mesh_mapper
        self.pad_id = config.pad_token_id
        if state_dict is None:
            state_dict = load_state_dict(weights_path)
        if intermediate_pads is None:
            pads = {None}
            if port.geglu_plan == "sharded":
                pads.add(port.intermediate_pad)
            if port.interleaved_pad > 0:
                pads.add(port.interleaved_pad)
            intermediate_pads = tuple(pads)
        t0 = time.perf_counter()
        self.encoder_params = prepare_weights(
            state_dict, config, device, policy, mesh_mapper, intermediate_pads=intermediate_pads
        )
        self.head_params = prepare_head_weights(state_dict, config, device, policy, mesh_mapper)
        self.weight_load_seconds = time.perf_counter() - t0
        self.temperature = self.head_params["temperature"]
        self.act_head = self.head_params["act_head"]
        self.buckets: Dict[Tuple[int, int], Bucket] = {}

    def bucket_for(self, n_rows: int, seq_len: int) -> Tuple[int, int]:
        return pick_bucket(n_rows, self.row_buckets), pick_bucket(seq_len, self.seq_buckets)

    def build_bucket(self, batch_size: int, seq_len: int) -> Bucket:
        key = (batch_size, seq_len)
        if key in self.buckets:
            return self.buckets[key]
        encoder = TtnnModernBertModel(
            self.encoder_params, self.config, self.device, seq_len, batch_size, self.policy, self.port, self.mesh_mapper
        )
        head = TtnnLayaHead(self.head_params, self.config, encoder.plan, self.device, self.policy, self.port)
        masks = TtnnMaskBuilder(self.config, self.device, seq_len, batch_size, self.mesh_mapper)
        input_ids = ttnn.allocate_tensor_on_device(
            ttnn.Shape((batch_size, seq_len)), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG
        )
        qtype = ttnn.allocate_tensor_on_device(
            ttnn.Shape((batch_size, 1)), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG
        )
        pad_row = masks.allocate_pad_row()
        b = Bucket(batch_size, seq_len, encoder, head, masks, input_ids, pad_row, qtype)
        b.addresses = b.input_addresses()
        self.buckets[key] = b
        return b

    def host_inputs(self, input_ids, attention_mask, qtype, batch_size, seq_len) -> HostInputs:
        n, L = input_ids.shape
        if n > batch_size or L > seq_len:
            raise ValueError(f"inputs ({n}, {L}) do not fit bucket ({batch_size}, {seq_len})")
        ids = torch.full((batch_size, seq_len), self.pad_id, dtype=torch.int32)
        att = torch.zeros((batch_size, seq_len), dtype=torch.long)
        qt = torch.zeros((batch_size, 1), dtype=torch.int32)
        ids[:n, :L] = input_ids.to(torch.int32)
        att[:n, :L] = attention_mask
        qt[:n, 0] = qtype.to(torch.int32)
        return HostInputs(
            input_ids=ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
            pad_row=ttnn.from_torch(pad_row_host(att), dtype=MASK_DTYPE, layout=ttnn.TILE_LAYOUT),
            qtype=ttnn.from_torch(qt, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
            n_real=n,
        )

    def write_inputs(self, bucket: Bucket, host: HostInputs) -> None:
        ttnn.copy_host_to_device_tensor(host.input_ids, bucket.input_ids)
        ttnn.copy_host_to_device_tensor(host.pad_row, bucket.pad_row)
        ttnn.copy_host_to_device_tensor(host.qtype, bucket.qtype)

    def device_forward(self, bucket: Bucket, layer_hook=None):
        """The traced body: masks from pad_row, encoder, head. Returns (logits (B,S,1) fp32, cls (B,32,H))."""
        masks = bucket.masks.build(bucket.pad_row)
        hidden = bucket.encoder(bucket.input_ids, masks, layer_hook=layer_hook)
        logits, cls, h = bucket.head(hidden, bucket.qtype, masks[FULL_ATTENTION])
        ttnn.deallocate(h)
        for m in masks.values():
            ttnn.deallocate(m)
        return logits, cls

    def readback(self, logits, cls, n_real: int):
        lg = ttnn.to_torch(logits).float()
        lg = lg.reshape(lg.shape[0], lg.shape[1])[:n_real]
        cl = ttnn.to_torch(cls).float()[:n_real, 0, :]
        return lg, cl

    def forward(self, input_ids, attention_mask, qtype, bucket=None):
        """Eager path. Returns dict(logits (n, L) fp32, cls (n, H) fp32, bucket, device_ms)."""
        n, L = input_ids.shape
        if bucket is None:
            bucket = self.bucket_for(n, L)
        b = self.build_bucket(*bucket)
        host = self.host_inputs(input_ids, attention_mask, qtype, b.batch_size, b.seq_len)
        t0 = time.perf_counter()
        self.write_inputs(b, host)
        logits, cls = self.device_forward(b)
        lg, cl = self.readback(logits, cls, host.n_real)
        ttnn.deallocate(logits)
        ttnn.deallocate(cls)
        dt = time.perf_counter() - t0
        return {"logits": lg[:, :L], "cls": cl, "bucket": bucket, "device_ms": dt * 1000.0}

    def describe(self) -> dict:
        return {
            "policy": self.policy.describe(),
            "port": self.port.describe(),
            "row_buckets": list(self.row_buckets),
            "seq_buckets": list(self.seq_buckets),
            "buckets": {f"{b}x{s}": describe_plan(v.encoder.plan) for (b, s), v in self.buckets.items()},
            "weight_load_seconds": round(self.weight_load_seconds, 2),
        }

    def release_bucket(self, key) -> None:
        b = self.buckets.pop(key, None)
        if b is None:
            return
        if b.trace_id is not None:
            ttnn.release_trace(self.device, b.trace_id)
        b.encoder.deallocate()
        b.masks.deallocate()
        for t in (b.input_ids, b.pad_row, b.qtype):
            ttnn.deallocate(t)

    def close(self) -> None:
        for key in list(self.buckets):
            self.release_bucket(key)
        deallocate_weights(self.encoder_params)
        deallocate_weights({"type_emb": self.head_params["type_emb"], "layers": self.head_params["layers"], "scorer": self.head_params["scorer"]})
