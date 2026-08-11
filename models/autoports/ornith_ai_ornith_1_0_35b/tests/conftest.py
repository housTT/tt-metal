# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "long: advertised-context / long-running functional-decoder cases (opt in with -m long)",
    )
