# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fail on host tensor computation/conversion inside one device-only forward."""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from torch.utils._python_dispatch import TorchDispatchMode


class _NoTorch(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise AssertionError(f"Runtime torch operation: {func}")


@contextmanager
def device_only():
    import ttnn

    def forbidden(*args, **kwargs):
        raise AssertionError("Runtime host/device tensor conversion")

    with ExitStack() as stack:
        for name in ("from_torch", "to_torch", "as_tensor"):
            stack.enter_context(patch.object(ttnn, name, forbidden))
        stack.enter_context(_NoTorch())
        yield
