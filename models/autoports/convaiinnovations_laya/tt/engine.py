# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

os.environ.setdefault("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES", "0")

from models.autoports.convaiinnovations_laya.reference.laya_reference import NEG_LOGIT
from models.autoports.convaiinnovations_laya.tt.model_config import (
    DEFAULT_POLICY_NAME,
    DEFAULT_PORT,
    ROW_BUCKETS,
    SEQ_BUCKETS,
    PortConfig,
    PrecisionPolicy,
    policy_from_name,
)

log = logging.getLogger("laya.tt.engine")

DEFAULT_TRACE_REGION_SIZE = 512 * 1024 * 1024
DEFAULT_L1_SMALL_SIZE = 79104
DEFAULT_MODEL_DIR = "/home/hous/dev/laya/state/laya_models/laya"
ACT_HIDDEN = 256
ACT_FEATURES = 4


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    return int(str(v).strip())


def _env_int_list(name: str, default: Sequence[int]) -> Tuple[int, ...]:
    v = os.environ.get(name)
    if v in (None, ""):
        return tuple(default)
    return tuple(sorted({int(x) for x in v.replace(";", ",").split(",") if x.strip()}))


def parse_mesh_shape(text: Optional[str]) -> Tuple[int, int]:
    if not text:
        return (1, 1)
    parts = text.lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise ValueError(f"LAYA_MESH_SHAPE must look like 1x1 or 1x4, got {text!r}")
    return (int(parts[0]), int(parts[1]))


def parse_warmup_shapes(
    text: Optional[str], row_buckets: Sequence[int], seq_buckets: Sequence[int]
) -> List[Tuple[int, int]]:
    """LAYA_WARMUP_SHAPES: 'all' (default), 'none', or '8x512,64x512' in per-device rows x seq."""
    if text is None or text.strip() == "" or text.strip().lower() == "all":
        return [(b, s) for s in seq_buckets for b in row_buckets]
    if text.strip().lower() == "none":
        return []
    shapes = []
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        b, s = item.lower().split("x")
        shapes.append((int(b), int(s)))
    return shapes


def build_act_head(act_weights: Dict[str, torch.Tensor], hidden_size: int) -> torch.nn.Sequential:
    """nn.Sequential(Linear(d + 4, 256), GELU(), Linear(256, 2)) as in vendor/rl_common.py DecisionModel, fp32."""
    head = torch.nn.Sequential(
        torch.nn.Linear(hidden_size + ACT_FEATURES, ACT_HIDDEN),
        torch.nn.GELU(),
        torch.nn.Linear(ACT_HIDDEN, act_weights["2.weight"].shape[0]),
    )
    with torch.no_grad():
        head[0].weight.copy_(act_weights["0.weight"].float())
        head[0].bias.copy_(act_weights["0.bias"].float())
        head[2].weight.copy_(act_weights["2.weight"].float())
        head[2].bias.copy_(act_weights["2.bias"].float())
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


@torch.inference_mode()
def host_tail(
    logits_all: torch.Tensor, cls: torch.Tensor, marker_pos: torch.Tensor, marker_mask: torch.Tensor, act_head
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The host part of DecisionModel.forward after the scorer: gather at the markers, -1e4 fill, features, act head."""
    logits_all = logits_all.float()
    n, length = logits_all.shape
    marker_pos = marker_pos.long()
    marker_mask = marker_mask.bool()
    if marker_mask.shape[1] < 2:
        pad = marker_mask.shape[1]
        marker_pos = torch.cat([marker_pos, torch.zeros(n, 2 - pad, dtype=torch.long)], 1)
        marker_mask = torch.cat([marker_mask, torch.zeros(n, 2 - pad, dtype=torch.bool)], 1)
    idx = marker_pos.clamp(min=0, max=length - 1)
    logits = torch.gather(logits_all, 1, idx).masked_fill(~marker_mask, NEG_LOGIT)
    p = torch.softmax(logits, -1)
    k = marker_mask.sum(-1).clamp(min=2).float()
    ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
    top2 = p.topk(2, -1).values
    feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
    act = act_head(torch.cat([cls.float(), feats], -1))
    return logits, act.float()


class LayaEngine:
    """TT backend with the server contract: forward(input_ids, attention_mask, marker_pos, marker_mask, qtype) -> (logits, act_logits)."""

    name = "tt"

    def __init__(
        self,
        model_dir: Optional[str] = None,
        mesh_shape: Tuple[int, int] = (1, 1),
        device_id: int = 0,
        policy: Optional[PrecisionPolicy] = None,
        port: PortConfig = DEFAULT_PORT,
        seq_buckets: Sequence[int] = SEQ_BUCKETS,
        row_buckets: Sequence[int] = ROW_BUCKETS,
        trace: bool = True,
        trace_region_size: int = DEFAULT_TRACE_REGION_SIZE,
        l1_small_size: int = DEFAULT_L1_SMALL_SIZE,
        warmup_shapes: Optional[Sequence[Tuple[int, int]]] = None,
        device=None,
        threads: Optional[int] = None,
    ):
        from transformers import AutoConfig

        from models.autoports.convaiinnovations_laya.tt.laya_model import (
            TtnnLayaModel,
            close_device,
            mesh_size,
            open_mesh,
        )
        from models.autoports.convaiinnovations_laya.tt.runner import LayaTraceRunner
        from models.autoports.convaiinnovations_laya.tt.weights import load_state_dict

        if threads:
            torch.set_num_threads(int(threads))
        self.model_dir = model_dir or os.environ.get("LAYA_MODEL_DIR") or DEFAULT_MODEL_DIR
        self.policy = policy or policy_from_name(DEFAULT_POLICY_NAME)
        self.port = port
        self.seq_buckets = tuple(sorted(seq_buckets))
        self.row_buckets = tuple(sorted(row_buckets))
        self.trace = bool(trace)
        self.trace_region_size = int(trace_region_size)
        self.l1_small_size = int(l1_small_size)
        self.mesh_shape = tuple(mesh_shape)
        self.lock = threading.Lock()
        self.last_device_ms = 0.0
        self.last_host_tail_ms = 0.0
        self.last_bucket: Optional[Tuple[int, int]] = None
        self.calls = 0
        self._close_device = close_device
        t0 = time.perf_counter()
        self.config = AutoConfig.from_pretrained(os.path.join(self.model_dir, "encoder"))
        with open(os.path.join(self.model_dir, "rl_agent_config.json")) as f:
            self.rl_config = json.load(f)
        sd = load_state_dict(os.path.join(self.model_dir, "model.safetensors"))
        self.owns_device = device is None
        if device is None:
            if self.mesh_shape == (1, 1):
                from models.autoports.convaiinnovations_laya.tt.laya_model import open_device

                device = open_device(
                    device_id=device_id, l1_small_size=self.l1_small_size, trace_region_size=self.trace_region_size
                )
            else:
                device = open_mesh(
                    self.mesh_shape, l1_small_size=self.l1_small_size, trace_region_size=self.trace_region_size
                )
        self.device = device
        self.num_devices = mesh_size(device)
        self.device_ids = list(device.get_device_ids()) if hasattr(device, "get_device_ids") else [device_id]
        self.open_seconds = time.perf_counter() - t0
        self.model = TtnnLayaModel(
            device,
            self.config,
            state_dict=sd,
            policy=self.policy,
            port=self.port,
            row_buckets=self.row_buckets,
            seq_buckets=self.seq_buckets,
        )
        self.act_head = build_act_head(self.model.act_head, int(self.config.hidden_size))
        self.temperature = self.model.temperature.clone()
        del sd
        if warmup_shapes is None:
            warmup_shapes = [(b, s) for s in self.seq_buckets for b in self.row_buckets]
        self.warmup_shapes = [tuple(x) for x in warmup_shapes]
        self.runner = LayaTraceRunner(self.model, self.warmup_shapes) if self.trace else None
        t1 = time.perf_counter()
        if self.trace:
            self.runner.warmup()
            self.warmup_seconds = dict(self.runner.warmup_seconds)
        else:
            self.warmup_seconds = {"eager": 0.0}
            for b, s in self.warmup_shapes:
                bucket = self.model.build_bucket(b, s)
                host = self.model.host_inputs(
                    torch.full((1, s), self.model.pad_id, dtype=torch.long),
                    torch.ones((1, s), dtype=torch.long),
                    torch.zeros(1, dtype=torch.long),
                    bucket.batch_size,
                    bucket.seq_len,
                )
                self.model.write_inputs(bucket, host)
                lg, cl = self.model.device_forward(bucket)
                import ttnn

                ttnn.synchronize_device(self.device)
                ttnn.deallocate(lg)
                ttnn.deallocate(cl)
            self.warmup_seconds["eager"] = time.perf_counter() - t1
        self.load_seconds = time.perf_counter() - t0
        log.info(
            "LayaEngine ready: mesh %s devices %s policy %s trace %s buckets %s x %s warm %s in %.1f s",
            "x".join(str(v) for v in self.mesh_shape),
            self.device_ids,
            self.policy.name,
            self.trace,
            list(self.row_buckets_total()),
            list(self.seq_buckets),
            [list(w) for w in self.warm_shapes_total()],
            self.load_seconds,
        )

    @classmethod
    def from_env(cls) -> "LayaEngine":
        mesh_shape = parse_mesh_shape(os.environ.get("LAYA_MESH_SHAPE"))
        seq_buckets = _env_int_list("LAYA_SEQ_BUCKETS", SEQ_BUCKETS)
        row_buckets = _env_int_list("LAYA_ROW_BUCKETS", ROW_BUCKETS)
        policy = policy_from_name(os.environ.get("LAYA_PRECISION") or DEFAULT_POLICY_NAME)
        port = DEFAULT_PORT
        overrides = os.environ.get("LAYA_PORT_OVERRIDES")
        if overrides:
            raw = json.loads(overrides)
            port = DEFAULT_PORT.with_(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items()})
        warm = parse_warmup_shapes(os.environ.get("LAYA_WARMUP_SHAPES"), row_buckets, seq_buckets)
        return cls(
            model_dir=os.environ.get("LAYA_MODEL_DIR") or None,
            mesh_shape=mesh_shape,
            device_id=_env_int("LAYA_DEVICE_ID", 0),
            policy=policy,
            port=port,
            seq_buckets=seq_buckets,
            row_buckets=row_buckets,
            trace=_env_flag("LAYA_TRACE", True),
            trace_region_size=_env_int("LAYA_TRACE_REGION_SIZE", DEFAULT_TRACE_REGION_SIZE),
            l1_small_size=_env_int("LAYA_L1_SMALL_SIZE", DEFAULT_L1_SMALL_SIZE),
            warmup_shapes=warm,
            threads=_env_int("LAYA_CPU_THREADS", 0) or None,
        )

    def row_buckets_total(self) -> Tuple[int, ...]:
        return self.model.row_buckets_total()

    def warm_shapes_total(self) -> List[Tuple[int, int]]:
        return [(self.model.rows_per_call(b), s) for b, s in self.warmup_shapes]

    def bucket_for(self, n_rows: int, seq_len: int) -> Tuple[int, int]:
        return self.model.bucket_for(n_rows, seq_len)

    def _run_device(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, qtype: torch.Tensor) -> dict:
        if self.runner is not None:
            return self.runner.run(input_ids, attention_mask, qtype)
        return self.model.forward(input_ids, attention_mask, qtype)

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = torch.as_tensor(input_ids).long()
        attention_mask = torch.as_tensor(attention_mask).long()
        marker_pos = torch.as_tensor(marker_pos).long()
        marker_mask = torch.as_tensor(marker_mask).bool()
        qtype = torch.as_tensor(qtype).long()
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"input_ids {tuple(input_ids.shape)} and attention_mask {tuple(attention_mask.shape)} must be [B, S]"
            )
        n = input_ids.shape[0]
        if marker_pos.shape[0] != n or marker_mask.shape != marker_pos.shape or qtype.shape != (n,):
            raise ValueError("marker_pos, marker_mask and qtype must match the rows of input_ids")
        with self.lock:
            out = self._run_device(input_ids, attention_mask, qtype)
            self.last_device_ms = float(out["device_ms"])
            self.last_bucket = tuple(out["bucket"])
            self.calls += 1
        t0 = time.perf_counter()
        logits, act = host_tail(out["logits"], out["cls"], marker_pos, marker_mask, self.act_head)
        self.last_host_tail_ms = (time.perf_counter() - t0) * 1000.0
        return logits, act

    __call__ = forward

    def forward_detailed(self, input_ids, attention_mask, marker_pos, marker_mask, qtype) -> dict:
        """forward plus the all-position logits, the CLS rows and the bucket, for the fidelity scripts."""
        input_ids = torch.as_tensor(input_ids).long()
        attention_mask = torch.as_tensor(attention_mask).long()
        marker_pos = torch.as_tensor(marker_pos).long()
        marker_mask = torch.as_tensor(marker_mask).bool()
        qtype = torch.as_tensor(qtype).long()
        with self.lock:
            out = self._run_device(input_ids, attention_mask, qtype)
            self.last_device_ms = float(out["device_ms"])
            self.last_bucket = tuple(out["bucket"])
            self.calls += 1
        t0 = time.perf_counter()
        logits, act = host_tail(out["logits"], out["cls"], marker_pos, marker_mask, self.act_head)
        self.last_host_tail_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "logits": logits,
            "act_logits": act,
            "logits_all": out["logits"],
            "cls": out["cls"],
            "bucket": tuple(out["bucket"]),
            "device_ms": out["device_ms"],
            "host_tail_ms": self.last_host_tail_ms,
        }

    def shapes(self) -> Dict[str, Any]:
        total_rows = list(self.row_buckets_total())
        return {
            "backend": "tt",
            "device": f"blackhole x{self.num_devices} ids {self.device_ids}",
            "arch": str(self.device.arch()) if hasattr(self.device, "arch") else "blackhole",
            "mesh_shape": "x".join(str(v) for v in self.mesh_shape),
            "num_devices": self.num_devices,
            "precision": self.policy.name,
            "policy": self.policy.describe(),
            "port": self.port.describe(),
            "seq_buckets": list(self.seq_buckets),
            "row_buckets": total_rows,
            "row_buckets_per_device": list(self.row_buckets),
            "max_rows": max(total_rows),
            "warm_shapes": [list(w) for w in self.warm_shapes_total()],
            "trace": self.trace,
            "trace_region_size": self.trace_region_size,
            "l1_small_size": self.l1_small_size,
            "pad_rows_keep_one_token": False,
            "max_len": int(self.rl_config.get("max_len", 512)),
            "head_max_len": int(self.rl_config.get("head_max_len", 192)),
            "weight_load_seconds": round(self.model.weight_load_seconds, 2),
            "warmup_seconds": {k: round(float(v), 2) for k, v in self.warmup_seconds.items()},
            "load_seconds": round(self.load_seconds, 2),
            "trace_bytes": self.runner.describe().get("trace_bytes", {}) if self.runner is not None else {},
            "trace_bytes_total": self.runner.trace_bytes_total() if self.runner is not None else 0,
            "model_dir": self.model_dir,
            "calls": self.calls,
        }

    def close(self) -> None:
        with self.lock:
            if self.runner is not None:
                try:
                    self.runner.release()
                except Exception as e:
                    log.warning("trace release failed: %s", e)
                self.runner = None
            if self.model is not None:
                try:
                    self.model.close()
                except Exception as e:
                    log.warning("model close failed: %s", e)
                self.model = None
            if self.owns_device and self.device is not None:
                self._close_device(self.device)
            self.device = None
