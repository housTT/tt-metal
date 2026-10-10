# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Control whether retaining construction inputs changes traced kernel placement."""

from ..tt import fused_decoder
from . import run_layer


def main():
    pack = fused_decoder._enable_indexed_rope

    def retain_inputs(model, mesh):
        attention = model.layer.attention
        held = attention.weights, attention.qkvg_weight
        pack(model, mesh)
        assert held[0] is not attention.weights and held[1] is not attention.qkvg_weight
        model.fusions = (*model.fusions, "indexed_placement_trial")

    fused_decoder._enable_indexed_rope = retain_inputs
    run_layer.main()


if __name__ == "__main__":
    main()
