# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "long: advertised-context / long-running decoder cases. The marker exists to narrow a run "
        "(`-m long` / `-m 'not long'`); it does not deselect anything by default, so a plain "
        "invocation of either decoder suite runs them.",
    )
