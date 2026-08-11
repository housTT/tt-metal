# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Record real-weight statistics for the Ornith decoder layers used by the functional tests.

The real 77 GB checkpoint is the canonical key/shape contract, but CI must not depend on it.
This writes ``doc/functional_decoder/weight_stats_layer{idx}.json`` — name, shape, dtype, mean,
std, min and max for every tensor the TTNN layer consumes — from which
:func:`hf_reference.synthetic_state_dict` regenerates deterministic stand-ins with the real
shapes.

Usage::

    python -m models.autoports.ornith_ai_ornith_1_0_35b.reference.collect_weight_stats 0 3
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R

OUT_DIR = Path(__file__).resolve().parents[1] / "doc" / "functional_decoder"


def main(layer_indices):
    model_path = R.resolve_model_path()
    text_config = R.load_text_config(model_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for layer_idx in layer_indices:
        state_dict = R.load_layer_state_dict(layer_idx, model_path)
        payload = {
            "hf_model_id": R.HF_MODEL_ID,
            "checkpoint_path": str(model_path),
            "layer_idx": layer_idx,
            "layer_kind": R.layer_kind(text_config, layer_idx),
            "num_tensors": len(state_dict),
            "tensors": R.weight_stats(state_dict),
        }
        out = OUT_DIR / f"weight_stats_layer{layer_idx}.json"
        with open(out, "w") as f:
            json.dump(payload, f, indent=1, sort_keys=True)
        print(f"wrote {out} ({len(state_dict)} tensors, kind={payload['layer_kind']})")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [0, 3])
