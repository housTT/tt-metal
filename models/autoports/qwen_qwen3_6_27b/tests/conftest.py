# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the Qwen3.6-27B functional-decoder tests.

This file supplies what the suite needs of its own - the 1x1 ``mesh_device``, the
``--long-context`` option and the per-test device-tensor release - rather than pulling in the
heavyweight tt-metal root ``conftest.py`` fixtures.  The root ``conftest.py`` is still on the
collection path when pytest runs from the repo root, which is where ``expect_error`` comes
from, and the root ``pytest.ini`` is where the 300 s per-test timeout comes from; see
``doc/functional_decoder/work_log.md`` section 2.3.
"""

from __future__ import annotations

import pytest

import ttnn


def pytest_configure(config):
    config.addinivalue_line("markers", "long_context: full advertised-context capability run")


def pytest_addoption(parser):
    parser.addoption("--long-context", action="store_true", default=False, help="run full-context tests")
    parser.addoption(
        "--impl",
        action="store",
        default="fused",
        choices=("fused", "functional"),
        help="decoder implementation the fused-stage perf runs measure (before/after pair)",
    )


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
