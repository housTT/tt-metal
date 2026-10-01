# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import os

import ttnn

_DTYPES = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}
_FIDELITIES = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}


def _pick(table, env, default):
    value = os.environ.get(env, default)
    if value not in table:
        raise ValueError(f"{env}={value!r}; expected one of {sorted(table)}")
    return table[value]


MLP_GATE_UP_DTYPE = _pick(_DTYPES, "QWEN36_MLP_GATE_UP_DTYPE", "bfp4")
MLP_DOWN_DTYPE = _pick(_DTYPES, "QWEN36_MLP_DOWN_DTYPE", "bfp8")
PROJ_DTYPE = _pick(_DTYPES, "QWEN36_PROJ_DTYPE", "bfp8")
MATMUL_FIDELITY = _pick(_FIDELITIES, "QWEN36_MATMUL_FIDELITY", "LoFi")
