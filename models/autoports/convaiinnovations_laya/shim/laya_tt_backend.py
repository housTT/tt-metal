# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Optional

import torch

try:
    from laya.backends.base import Backend as _LayaBackend
except ImportError:
    _LayaBackend = None


class _FallbackBackend:
    name = "eager"
    max_len: Optional[int] = None

    def __init__(self, agent):
        self.agent = agent
        self.model = agent.model
        self.installed = False
        self._stock_forward = None
        self._prior_forward = None

    def install(self) -> None:
        if self.installed:
            return
        self._prior_forward = self.model.__dict__.get("forward")
        self._stock_forward = self.model.forward
        self._install()
        self.model.forward = self.forward
        self.installed = True

    def uninstall(self) -> None:
        if not self.installed:
            return
        if self._prior_forward is None:
            self.model.__dict__.pop("forward", None)
        else:
            self.model.forward = self._prior_forward
        self._stock_forward = self._prior_forward = None
        self._uninstall()
        self.installed = False

    def _install(self) -> None:
        pass

    def _uninstall(self) -> None:
        pass

    def warmup(self, shapes=None) -> float:
        return 0.0

    def stock_forward(self, *args, **kwargs):
        fn = self._stock_forward if self._stock_forward is not None else type(self.model).forward.__get__(self.model)
        return fn(*args, **kwargs)

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=False):
        return self.stock_forward(input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder)

    def __repr__(self):
        return "%s(installed=%s)" % (type(self).__name__, self.installed)


Backend = _LayaBackend if _LayaBackend is not None else _FallbackBackend
BASE_FROM_PIP = _LayaBackend is not None


class TtBackend(Backend):
    name = "tt"

    def __init__(self, agent, url: Optional[str] = None, engine: Any = None, timeout: float = 120.0, api_key: Optional[str] = None):
        super().__init__(agent)
        self.url = (url or os.environ.get("LAYA_TT_URL") or "").rstrip("/")
        self.engine = engine
        self.timeout = float(timeout)
        self.api_key = api_key or os.environ.get("LAYA_API_KEY") or None
        if self.engine is None and not self.url:
            raise ValueError("TtBackend needs url=http://host:port of a laya-p150 server started with LAYA_RAW_FORWARD=1, or engine=LayaEngine")
        self.max_len = None
        self.last_batch = None
        self.last_device_ms = None

    @classmethod
    def in_process(cls, agent):
        from ..tt.engine import LayaEngine

        return cls(agent, engine=LayaEngine.from_env())

    def _post(self, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.url + "/v1/forward", data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", "Bearer " + self.api_key)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            self.last_batch = resp.headers.get("X-Laya-Batch")
            device_ms = resp.headers.get("X-Laya-Device-Ms")
            self.last_device_ms = float(device_ms) if device_ms else None
        return body

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=False):
        device = input_ids.device
        if self.engine is not None:
            logits, act = self.engine.forward(
                input_ids.detach().cpu().long(),
                attention_mask.detach().cpu().long(),
                marker_pos.detach().cpu().long(),
                marker_mask.detach().cpu().bool(),
                qtype.detach().cpu().long(),
            )
            n, k = input_ids.shape[0], marker_pos.shape[1]
            logits = torch.as_tensor(logits)[:n, :k]
            act = torch.as_tensor(act)[:n]
            self.last_batch = getattr(self.engine, "last_batch", None)
            self.last_device_ms = getattr(self.engine, "last_device_ms", None)
        else:
            body = self._post(
                {
                    "input_ids": input_ids.detach().cpu().long().tolist(),
                    "attention_mask": attention_mask.detach().cpu().long().tolist(),
                    "marker_pos": marker_pos.detach().cpu().long().tolist(),
                    "marker_mask": marker_mask.detach().cpu().bool().tolist(),
                    "qtype": qtype.detach().cpu().long().tolist(),
                }
            )
            logits = torch.tensor(body["logits"], dtype=torch.float32)
            act = torch.tensor(body["act_logits"], dtype=torch.float32)
        return logits.to(device=device, dtype=torch.float32), act.to(device=device, dtype=torch.float32)
