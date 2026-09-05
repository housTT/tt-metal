# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Native decoder execution with conservative full-stack memory reservations.

Reservations model bytes, not additional executing layers. The separate stack
probe validates decoder composition. This is a capacity test, not a full model.
"""

import json
import math
from pathlib import Path

import pytest

import ttnn

from . import test_functional_decoder as H
from .test_multichip_decoder import multichip_contract, pytestmark  # noqa: F401


@pytest.mark.long
@pytest.mark.timeout(1800)
def test_native_context_with_stack_reservations(mesh_device):
    plan = json.loads(
        (Path(__file__).resolve().parents[1] / "doc/multichip_decoder/memory_capacity_plan.json").read_text()
    )
    # Reserve ALL planned persistent DRAM plus 2GiB scratch. The tested decoder's
    # own weights/cache/RoPE are additional, making this deliberately conservative.
    # Actual native input + chunk outputs + concat supply the other 6GiB peak.
    reserve_bytes = (
        plan["estimated_full_model_bytes_per_device"]
        - plan["trace_activation_and_small_constant_reserve_bytes_per_device"]
        + 2 * 1024**3
    )
    reservations = []
    reserve_bytes = math.ceil(reserve_bytes / (2 * 1024**2)) * (2 * 1024**2)
    remaining = reserve_bytes
    while remaining > 0:
        size = min(256 * 1024**2, remaining)
        reservations.append(
            ttnn.empty(
                [1, 1, size // 65536, 32768],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
        remaining -= size
    recurrent = [
        ttnn.empty(
            [1, 8, 128, 128],
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        for _ in range(24)
    ]
    print(
        json.dumps(
            dict(
                reserved_dram_bytes_per_device=reserve_bytes,
                persistent_recurrent_layers_in_l1=len(recurrent),
                native_context=H.ADVERTISED_CONTEXT,
                actual_layer="full_attention",
                reservation_is_not_execution=True,
            )
        ),
        flush=True,
    )
    H.test_full_context_prefill_and_decode(mesh_device, H.FULL_LAYER, H.ADVERTISED_CONTEXT - 1)
    ttnn.synchronize_device(mesh_device)
    for value in reservations + recurrent:
        ttnn.deallocate(value)
