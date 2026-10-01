# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import numpy as np

from ..clm.heads import HIDDEN, PROJ_DIM, HeadPair

DEFAULT_CHECKPOINT = "/home/hous/dev/clm-v0.1-8B/checkpoints/CLM_v0.1-8B.pt"
MODEL_NAME = "clm-latest"


def checkpoint_path(path: str | None = None) -> str:
    p = path or os.environ.get("CLM_CKPT") or DEFAULT_CHECKPOINT
    if not os.path.isfile(p):
        raise FileNotFoundError(f"CLM head checkpoint not found: {p}")
    return p


def load_heads(path: str | None = None, device: str = "cpu", name: str = MODEL_NAME) -> HeadPair:
    return HeadPair(name, checkpoint_path(path), device).ensure()


def as_embeddings(x, hidden: int = HIDDEN) -> np.ndarray:
    a = np.ascontiguousarray(x, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != hidden:
        raise ValueError(f"expected embeddings of shape [n, {hidden}], got {tuple(a.shape)}")
    return a


def project_states(heads: HeadPair, embeddings) -> np.ndarray:
    return heads.project_states(as_embeddings(embeddings)).cpu().numpy()


def project_actions(heads: HeadPair, embeddings) -> np.ndarray:
    return heads.project_actions(as_embeddings(embeddings)).cpu().numpy()


def project(heads: HeadPair, states, actions) -> tuple[np.ndarray, np.ndarray]:
    return heads.project(as_embeddings(states), as_embeddings(actions))


def scores(heads: HeadPair, states, actions, temperature: float = 1.0) -> np.ndarray:
    zs, za = project(heads, states, actions)
    return (heads.scale / temperature) * (zs @ za.T)


def head_info(heads: HeadPair) -> dict:
    heads.ensure()
    return {
        "name": heads.name,
        "path": os.path.abspath(heads.path),
        "device": heads.device,
        "cfg": dict(heads.cfg),
        "hidden": int(heads.cfg.get("hidden_size", HIDDEN)),
        "proj_dim": int(heads.proj_dim or PROJ_DIM),
        "scale": float(heads.scale),
        "n_params": int(heads.n_params),
        "n_params_per_head": int(heads.n_params) // 2,
        "generation": heads.generation,
    }
