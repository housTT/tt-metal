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

import pathlib

import pytest

import ttnn

#: Printed once per run, into every log this stage commits.  A stage review found the committed
#: correctness, long-context and watcher runs had been produced by a build of
#: ``tt/fused_decoder.py`` that predated a shipped decode-configuration change, and no gate could
#: see it because every gate reads artifacts and none tied an artifact to the source.
#: ``test_fused_decoder_docs.py::test_every_run_was_made_against_the_shipped_build`` is that tie.
BUILD_STAMP = "FUSED_BUILD"


def fused_build_fingerprint() -> str:
    """SHA-256 of the fused decoder's source, which is what the stage's evidence is *of*."""
    import hashlib

    source = pathlib.Path(__file__).resolve().parents[1] / "tt" / "fused_decoder.py"
    return hashlib.sha256(source.read_bytes()).hexdigest()


def pytest_configure(config):
    config.addinivalue_line("markers", "long_context: full advertised-context capability run")
    print(f"\n{BUILD_STAMP} tt/fused_decoder.py sha256={fused_build_fingerprint()}")


def pytest_addoption(parser):
    parser.addoption("--long-context", action="store_true", default=False, help="run full-context tests")
    parser.addoption(
        "--impl",
        action="store",
        default="fused",
        choices=("fused", "functional"),
        help="decoder implementation the fused-stage perf runs measure (before/after pair)",
    )
    parser.addoption(
        "--perf-batch",
        action="store",
        type=int,
        default=1,
        help="users in the traced-decode perf window; 32 measures the advertised max_batch graph",
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
