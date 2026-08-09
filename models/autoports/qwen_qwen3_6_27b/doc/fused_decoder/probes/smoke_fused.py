# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fast fused-decoder smoke: prefill + decode PCC against HF for both layer kinds.

Bring-up loop only — the delivered coverage is ``tests/test_fused_decoder.py``.
Usage: python -m models.autoports.qwen_qwen3_6_27b.doc.fused_decoder.probes.smoke_fused [seq_len ...]
"""

from __future__ import annotations

import sys
import time

import torch
import ttnn
from transformers.cache_utils import DynamicCache

from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tests import harness as H
from models.autoports.qwen_qwen3_6_27b.tt.fused_decoder import FusedDecoder


def main():
    seq_lens = [int(a) for a in sys.argv[1:]] or [17, 2049]
    H.DECODER_CLS = FusedDecoder
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    stats = ref.load_weight_stats()
    failures = 0
    try:
        for layer_idx in (H.LINEAR_LAYER_IDX, H.FULL_LAYER_IDX):
            for seq_len in seq_lens:
                lut = H.build_layer(device, layer_idx, max_batch=1, max_seq_len=8192)
                kind = lut.config.layer_types[layer_idx]
                hidden = ref.synthetic_hidden_states(lut.config, 1, seq_len, stats)
                cache = DynamicCache(config=lut.config)
                golden = H.reference_prefill(lut, hidden, cache)
                start = time.perf_counter()
                got = H.run_tt_prefill(lut, hidden)
                prefill_ms = 1e3 * (time.perf_counter() - start)
                value = H.pcc(golden, got)
                failures += value < H.PCC_BAR
                print(f"  {kind:17s} seq={seq_len:5d} prefill pcc={value:.6f} ({prefill_ms:.1f} ms)", flush=True)

                H.prepare_decode(lut)
                token = ref.synthetic_hidden_states(lut.config, 1, 1, stats, seed=101)
                golden_d = H.reference_decode(lut, token, seq_len, cache)
                got_d = H.run_tt_decode(lut, token, torch.tensor([seq_len]))
                value_d = H.pcc(golden_d, got_d)
                failures += value_d < H.PCC_BAR
                print(f"  {kind:17s} seq={seq_len:5d} decode  pcc={value_d:.6f}", flush=True)
                H.release_layers()
    finally:
        ttnn.close_mesh_device(device)
    print(f"SMOKE {'FAIL' if failures else 'OK'} ({failures} below bar)")


if __name__ == "__main__":
    main()
