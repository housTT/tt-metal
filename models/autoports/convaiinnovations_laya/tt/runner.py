# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

import ttnn
from models.autoports.convaiinnovations_laya.tt.laya_model import Bucket, TtnnLayaModel

os.environ.setdefault("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES", "0")
TRACE_REGION_SIZE = 512 * 1024 * 1024


def trace_bytes_allocated(device) -> int:
    """Bytes of the trace region in use on one device (per-bank allocation times the bank count)."""
    try:
        view = ttnn.get_memory_view(device, ttnn.BufferType.TRACE)
        return int(view.total_bytes_allocated_per_bank) * int(view.num_banks)
    except Exception:
        return -1


def trace_region_bytes(device) -> int:
    try:
        view = ttnn.get_memory_view(device, ttnn.BufferType.TRACE)
        return int(view.total_bytes_per_bank) * int(view.num_banks)
    except Exception:
        return -1


def mark_corruptible(tensor) -> None:
    try:
        from ttnn.unsafe_allocation_tracker import UnsafeAllocationTracker
    except ImportError:
        return
    try:
        UnsafeAllocationTracker.mark_corruptible(tensor)
    except Exception:
        pass


class LayaTraceRunner:
    """One Metal trace per (rows, seq) bucket around TtnnLayaModel.device_forward, two-phase warmup."""

    def __init__(self, model: TtnnLayaModel, buckets: Optional[Sequence[Tuple[int, int]]] = None, cq_id: int = 0):
        self.model = model
        self.device = model.device
        self.cq_id = cq_id
        if buckets is None:
            buckets = [(b, s) for s in model.seq_buckets for b in model.row_buckets]
        self.bucket_keys: List[Tuple[int, int]] = list(buckets)
        self.captured: Dict[Tuple[int, int], Bucket] = {}
        self.warmup_seconds = {"phase1": 0.0, "phase2": 0.0}
        self.build_seconds: Dict[Tuple[int, int], float] = {}
        self.eager_seconds: Dict[Tuple[int, int], float] = {}
        self.capture_seconds: Dict[Tuple[int, int], float] = {}
        self.trace_bytes: Dict[Tuple[int, int], int] = {}
        self.calls = 0

    def _dummy_host(self, b: Bucket):
        rows = self.model.rows_per_call(b.batch_size)
        ids = torch.full((rows, b.seq_len), self.model.pad_id, dtype=torch.long)
        ids[:, 0] = self.model.config.cls_token_id if hasattr(self.model.config, "cls_token_id") else self.model.pad_id
        att = torch.zeros((rows, b.seq_len), dtype=torch.long)
        att[:, :8] = 1
        qt = torch.zeros(rows, dtype=torch.long)
        return self.model.host_inputs(ids, att, qt, b.batch_size, b.seq_len)

    def warmup(self) -> None:
        """Phase 1: build every bucket and run it once eagerly. Phase 2: capture one trace per bucket, back to back."""
        t0 = time.perf_counter()
        built = []
        for key in self.bucket_keys:
            tb = time.perf_counter()
            b = self.model.build_bucket(*key)
            self.build_seconds[key] = time.perf_counter() - tb
            built.append(b)
        for b in built:
            te = time.perf_counter()
            self.model.write_inputs(b, self._dummy_host(b))
            logits, cls = self.model.device_forward(b)
            ttnn.synchronize_device(self.device)
            ttnn.deallocate(logits)
            ttnn.deallocate(cls)
            self.eager_seconds[(b.batch_size, b.seq_len)] = time.perf_counter() - te
        t1 = time.perf_counter()
        for b in built:
            self.capture(b)
        self.warmup_seconds = {"phase1": t1 - t0, "phase2": time.perf_counter() - t1}

    def capture(self, b: Bucket) -> None:
        if b.trace_id is not None:
            return
        key = (b.batch_size, b.seq_len)
        tc = time.perf_counter()
        self.model.write_inputs(b, self._dummy_host(b))
        ttnn.synchronize_device(self.device)
        before = b.input_addresses()
        used_before = trace_bytes_allocated(self.device)
        tid = ttnn.begin_trace_capture(self.device, cq_id=self.cq_id)
        logits, cls = self.model.device_forward(b)
        ttnn.end_trace_capture(self.device, tid, cq_id=self.cq_id)
        after = b.input_addresses()
        if before != after:
            raise RuntimeError(f"input addresses moved during capture of bucket {key}: {before} -> {after}")
        mark_corruptible(logits)
        mark_corruptible(cls)
        b.trace_id = tid
        b.trace_outputs = (logits, cls)
        self.captured[key] = b
        used_after = trace_bytes_allocated(self.device)
        self.trace_bytes[key] = (used_after - used_before) if (used_before >= 0 and used_after >= 0) else -1
        self.capture_seconds[key] = time.perf_counter() - tc

    def replay(self, b: Bucket, host) -> Tuple[torch.Tensor, torch.Tensor]:
        if b.input_addresses() != b.addresses:
            raise RuntimeError("input buffers moved since construction")
        self.model.write_inputs(b, host)
        ttnn.execute_trace(self.device, b.trace_id, cq_id=self.cq_id, blocking=False)
        logits, cls = b.trace_outputs
        return self.model.readback(logits, cls, host.n_real)

    def run(self, input_ids, attention_mask, qtype, bucket=None) -> dict:
        n, L = input_ids.shape
        if bucket is None:
            bucket = self.model.bucket_for(n, L)
        b = self.captured.get(tuple(bucket))
        if b is None:
            b = self.model.build_bucket(*bucket)
            self.capture(b)
        host = self.model.host_inputs(input_ids, attention_mask, qtype, b.batch_size, b.seq_len)
        t0 = time.perf_counter()
        lg, cl = self.replay(b, host)
        dt = time.perf_counter() - t0
        self.calls += 1
        return {"logits": lg[:, :L], "cls": cl, "bucket": tuple(bucket), "device_ms": dt * 1000.0}

    def run_timed(self, input_ids, attention_mask, qtype, bucket=None) -> dict:
        """Blocking replay so the write, the device replay and the host readback are timed apart (stage 5 report)."""
        n, L = input_ids.shape
        if bucket is None:
            bucket = self.model.bucket_for(n, L)
        b = self.captured.get(tuple(bucket))
        if b is None:
            b = self.model.build_bucket(*bucket)
            self.capture(b)
        host = self.model.host_inputs(input_ids, attention_mask, qtype, b.batch_size, b.seq_len)
        if b.input_addresses() != b.addresses:
            raise RuntimeError("input buffers moved since construction")
        t0 = time.perf_counter()
        self.model.write_inputs(b, host)
        ttnn.synchronize_device(self.device)
        t1 = time.perf_counter()
        ttnn.execute_trace(self.device, b.trace_id, cq_id=self.cq_id, blocking=True)
        t2 = time.perf_counter()
        logits, cls = b.trace_outputs
        lg, cl = self.model.readback(logits, cls, host.n_real)
        t3 = time.perf_counter()
        self.calls += 1
        return {
            "logits": lg[:, :L],
            "cls": cl,
            "bucket": tuple(bucket),
            "write_ms": (t1 - t0) * 1000.0,
            "replay_ms": (t2 - t1) * 1000.0,
            "readback_ms": (t3 - t2) * 1000.0,
            "device_ms": (t3 - t0) * 1000.0,
        }

    def run_eager(self, input_ids, attention_mask, qtype, bucket=None) -> dict:
        return self.model.forward(input_ids, attention_mask, qtype, bucket=bucket)

    def release(self) -> None:
        for key, b in list(self.captured.items()):
            if b.trace_id is not None:
                ttnn.release_trace(self.device, b.trace_id)
                b.trace_id = None
            b.trace_outputs = None
        self.captured = {}

    def trace_bytes_total(self) -> int:
        return sum(v for v in self.trace_bytes.values() if v > 0)

    def describe(self) -> dict:
        key = lambda k: f"{k[0]}x{k[1]}"
        return {
            "buckets": [list(k) for k in self.bucket_keys],
            "captured": [list(k) for k in self.captured],
            "warmup_seconds": {k: round(v, 3) for k, v in self.warmup_seconds.items()},
            "build_seconds": {key(k): round(v, 3) for k, v in self.build_seconds.items()},
            "eager_seconds": {key(k): round(v, 3) for k, v in self.eager_seconds.items()},
            "capture_seconds": {key(k): round(v, 3) for k, v in self.capture_seconds.items()},
            "trace_bytes": {key(k): v for k, v in self.trace_bytes.items()},
            "trace_bytes_total": self.trace_bytes_total(),
            "trace_bytes_allocated_now": trace_bytes_allocated(self.device),
            "trace_region_bytes": trace_region_bytes(self.device),
            "calls": self.calls,
            "num_devices": self.model.num_devices,
        }
