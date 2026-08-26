# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn


def pytest_addoption(parser):
    parser.addoption("--long-context", action="store_true", default=False)


def pytest_configure(config):
    config.addinivalue_line("markers", "long_context: full advertised 262144-token context")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--long-context"):
        return
    skip = pytest.mark.skip(reason="pass --long-context for the full advertised-context run")
    for item in items:
        if "long_context" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def mesh_device():
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1),
        # TTNN auto-sizes trace storage from zero.  A fixed 768 MiB reservation
        # needlessly makes the exact 262143-token PLE concat miss DRAM by
        # roughly 171 MiB while providing no benefit to these layer traces.
        trace_region_size=0,
        physical_device_ids=[0],
    )
    yield mesh
    ttnn.close_mesh_device(mesh)
