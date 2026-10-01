# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import threading
import time

import numpy as np

DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_DTYPE = "float32"
DEFAULT_THREADS = 8
HIDDEN = 4096
_DTYPES = {"float32": "float32", "fp32": "float32", "bfloat16": "bfloat16", "bf16": "bfloat16"}


def resolve_dtype(name: str | None):
    import torch

    key = (name or os.environ.get("CLM_HF_DTYPE") or DEFAULT_DTYPE).lower()
    if key not in _DTYPES:
        raise ValueError(f"CLM_HF_DTYPE must be float32 or bfloat16, got {name!r}")
    return getattr(torch, _DTYPES[key])


class HfQwen3Embedder:
    def __init__(
        self,
        model_id: str | None = None,
        max_tokens: int | None = None,
        dtype: str | None = None,
        threads: int | None = None,
        revision: str | None = None,
        load: bool = True,
    ):
        self.model_id = model_id or os.environ.get("HF_MODEL") or DEFAULT_MODEL_ID
        self.max_tokens = int(max_tokens or os.environ.get("CLM_MAX_TOKENS") or DEFAULT_MAX_TOKENS)
        self.dtype_name = (dtype or os.environ.get("CLM_HF_DTYPE") or DEFAULT_DTYPE).lower()
        self.threads = int(threads or os.environ.get("CLM_HF_THREADS") or DEFAULT_THREADS)
        self.revision = revision or os.environ.get("CLM_HF_REVISION") or None
        self.name = os.environ.get("CLM_EMB_MODEL", "qwen3-8b")
        self.hidden = HIDDEN
        self.tokenizer = None
        self.model = None
        self.load_seconds: float | None = None
        self._lock = threading.Lock()
        if load:
            self.load()

    def load(self) -> "HfQwen3Embedder":
        import torch
        from transformers import AutoTokenizer, Qwen3Model

        torch.set_num_threads(self.threads)
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        model = Qwen3Model.from_pretrained(self.model_id, dtype=resolve_dtype(self.dtype_name), revision=self.revision)
        self.model = model.eval()
        self.hidden = int(self.model.config.hidden_size)
        self.load_seconds = time.perf_counter() - t0
        return self

    @property
    def cap(self) -> int:
        return max(1, self.max_tokens - 1)

    def _ids(self, text: str) -> list[int]:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(t) for t in ids]

    def tokenize(self, text: str) -> list[int]:
        ids = self._ids(text)
        if not ids:
            ids = self._ids(" ")
        return ids

    def truncate(self, ids: list[int]) -> list[int]:
        ids = [int(t) for t in ids]
        if not ids:
            ids = self._ids(" ")
        return ids[-self.cap :]

    def _forward(self, ids: list[int]) -> np.ndarray:
        import torch

        x = torch.tensor([ids], dtype=torch.long)
        with torch.inference_mode():
            h = self.model(input_ids=x).last_hidden_state[0, -1]
        v = h.float().cpu().numpy().astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)

    def embed_ids(self, id_lists: list[list[int]]) -> tuple[np.ndarray, int]:
        if self.model is None:
            self.load()
        out: list[np.ndarray] = []
        tokens = 0
        with self._lock:
            for ids in id_lists:
                ids = self.truncate(ids)
                tokens += len(ids)
                out.append(self._forward(ids))
        if not out:
            return np.zeros((0, self.hidden), dtype=np.float32), 0
        return np.stack(out).astype(np.float32), tokens

    def embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        if self.model is None:
            self.load()
        return self.embed_ids([self.tokenize(t) for t in texts])

    def healthy(self) -> bool:
        return self.model is not None

    def info(self) -> dict:
        return {
            "kind": "hf",
            "model_id": self.model_id,
            "revision": self.revision,
            "dtype": self.dtype_name,
            "threads": self.threads,
            "max_tokens": self.max_tokens,
            "hidden": self.hidden,
            "load_seconds": self.load_seconds,
        }
