# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Self-contained fixtures for the Qwen3.6-27B functional-decoder tests.

The autoport tests deliberately do not depend on the tt-metal root ``conftest.py`` so they
can be run from any working directory (this host must run device jobs from outside a
tt-metal checkout; see ``doc/functional_decoder/work_log.md``).
"""

from __future__ import annotations

import pytest
import ttnn


def pytest_configure(config):
    config.addinivalue_line("markers", "long_context: full advertised-context capability run")


def pytest_addoption(parser):
    parser.addoption("--long-context", action="store_true", default=False, help="run full-context tests")


@pytest.fixture(autouse=True)
def release_layers():
    """Free the device weights/caches a test allocated so the session does not run out of DRAM."""
    from models.autoports.qwen_qwen3_6_27b.tests import harness

    yield
    harness.release_layers()


@pytest.fixture(scope="session")
def mesh_device():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    yield device
    ttnn.close_mesh_device(device)
