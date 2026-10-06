# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import pytest

CPU_MODEL_DIR = os.environ.get("LAYA_MODEL_DIR") or "/home/hous/dev/laya/state/laya_models/laya"


@pytest.fixture(scope="session")
def cpu_engine():
    if not os.path.isfile(os.path.join(CPU_MODEL_DIR, "model.safetensors")):
        pytest.skip("weights missing: %s" % CPU_MODEL_DIR)
    os.environ.setdefault("LAYA_CPU_THREADS", "4")
    os.environ["LAYA_BACKEND"] = "cpu"
    os.environ.setdefault("LAYA_MODEL_DIR", CPU_MODEL_DIR)
    os.environ.setdefault("LAYA_RAW_FORWARD", "1")
    from models.autoports.convaiinnovations_laya.server.engine import Engine

    return Engine.from_env()


@pytest.fixture(scope="session")
def client(cpu_engine):
    from fastapi.testclient import TestClient

    from models.autoports.convaiinnovations_laya.server.app import create_app

    with TestClient(create_app(engine=cpu_engine, raw_forward=True, demo=True, sanity=True)) as c:
        yield c
