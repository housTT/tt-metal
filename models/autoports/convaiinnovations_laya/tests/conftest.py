# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
import torch

from models.autoports.convaiinnovations_laya.tests import laya_inputs
from models.autoports.convaiinnovations_laya.tt.weights import load_state_dict, split_state_dict

DOC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "doc")
torch.set_num_threads(int(os.environ.get("LAYA_TORCH_THREADS", "6")))


def policy_from_env():
    from models.autoports.convaiinnovations_laya.tt.model_config import policy_from_name

    return policy_from_name(os.environ.get("LAYA_POLICY"))


def port_from_env():
    from models.autoports.convaiinnovations_laya.tt.model_config import DEFAULT_PORT, STAGE1_PORT

    base = STAGE1_PORT if os.environ.get("LAYA_PORT", "default") == "stage1" else DEFAULT_PORT
    overrides = json.loads(os.environ.get("LAYA_PORT_OVERRIDES", "{}"))
    overrides = {k: (tuple(v) if isinstance(v, list) else v) for k, v in overrides.items()}
    return base.with_(**overrides)


def port_label():
    label = "stage1port" if os.environ.get("LAYA_PORT", "default") == "stage1" else "shippedport"
    if os.environ.get("LAYA_PORT_OVERRIDES", "{}").strip() not in ("", "{}"):
        label += "_overrides"
    return label


@pytest.fixture(scope="module")
def module_device(_device_module_impl):
    return _device_module_impl


@pytest.fixture(scope="session")
def laya_config():
    return laya_inputs.load_config()


@pytest.fixture(scope="session")
def state_dict():
    return load_state_dict()


@pytest.fixture(scope="session")
def parts(state_dict):
    return split_state_dict(state_dict)


@pytest.fixture(scope="session")
def torch_encoder(laya_config, parts):
    from models.autoports.convaiinnovations_laya.reference.modernbert import ModernBertModel

    ref = ModernBertModel(laya_config)
    ref.load_state_dict(parts["encoder"], strict=True)
    ref.eval()
    return ref


@pytest.fixture(scope="session")
def pcc_log():
    rows = []
    yield rows
    path = os.environ.get("LAYA_PCC_LOG")
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        existing = []
        if os.path.exists(path):
            with open(path) as f:
                existing = json.load(f)
        with open(path, "w") as f:
            json.dump(existing + rows, f, indent=1)


def record(pcc_log, **row):
    pcc_log.append(row)
    print("\n[pcc] " + json.dumps(row))


@torch.no_grad()
def encoder_inputs(batch_size=1, seq_len=512, fill=False, offset=0):
    return laya_inputs.build_inputs(batch_size=batch_size, seq_len=seq_len, fill=fill, offset=offset)


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
