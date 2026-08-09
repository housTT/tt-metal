# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Is the precision-independent structural gate of §20 still impossible?

§20 records that pinning every weight to BF16 and running the disputed lengths through the
optimized path dies in the prefill program-config search with a circular-buffer overflow: the
search is sized for the shipped BFP4/BFP8 weights and a BF16 weight is 3.6x a BFP4 one.

Two things changed after that was written.  ``_prefill_linear`` now re-searches when a *cached*
config fails, and its candidate list is ordered rather than computed from one calibrated model.
So the question is worth re-asking rather than inherited: for each layer kind and each disputed
length, build the layer at :data:`FUSED_BASELINE_PRECISION` (BF16 everywhere, HiFi4, fp32
accumulation) and report the PCC or the exact exception.
"""
from __future__ import annotations

import json
import sys
import traceback

sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal")

import torch  # noqa: E402
import ttnn  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref  # noqa: E402
from models.autoports.qwen_qwen3_6_27b.tests import harness as H  # noqa: E402
from models.autoports.qwen_qwen3_6_27b.tt import optimized_decoder as O  # noqa: E402

LENGTHS = [1, 17, 64, 743, 2049, 5000]
KINDS = {"linear_attention": 0, "full_attention": 3}


def main() -> None:
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=23887872)
    config = ref.load_text_config()
    stats = ref.load_weight_stats()
    try:
        for kind, layer_idx in KINDS.items():
            lut = H.build_layer(
                mesh_device, layer_idx, max_batch=1, max_seq_len=8192, real_weights=False,
                decoder_cls=O.OptimizedDecoder,
                decoder_kwargs={"precision": O.FUSED_BASELINE_PRECISION},
            )
            for seq_len in LENGTHS:
                row = {"kind": kind, "seq_len": seq_len}
                hidden = ref.synthetic_hidden_states(config, 1, seq_len, stats)
                cache = DynamicCache(config=config)
                try:
                    golden = H.reference_prefill(lut, hidden, cache)
                    row["prefill_pcc"] = H.pcc(golden, H.run_tt_prefill(lut, hidden))
                    H.prepare_decode(lut)
                    token = ref.synthetic_hidden_states(config, 1, 1, stats, seed=900 + seq_len)
                    golden_decode = H.reference_decode(lut, token, seq_len, cache)
                    got = H.run_tt_decode(lut, token, torch.tensor([seq_len]))
                    row["decode_pcc"] = H.pcc(golden_decode, got)
                except Exception as exc:  # noqa: BLE001 - the exception *is* the result here
                    row["error"] = "".join(traceback.format_exception_only(type(exc), exc))[:600]
                print("BF16STRUCT " + json.dumps(row), flush=True)
            H.release_layers()
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
