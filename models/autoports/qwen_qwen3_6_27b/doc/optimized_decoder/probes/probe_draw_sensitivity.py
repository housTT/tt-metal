# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""How much does a real-weight PCC depend on which decode token you happen to draw?

Written because the sweep and the suite disagreed by 0.0045 about ``proj_fp32_acc=False`` -
``logs/sweep_v5_real_short_743.log`` reads 0.998899 for ``linear_attention`` decode at seq 743
and the shipped ``test_real_weight_pcc_at_disputed_lengths`` reads 0.994365 for the same field,
same length, same weights.  The two harnesses differ in exactly one input: the sweep draws its
decode token with ``seed=777``, the suite with ``seed=900 + seq_len``.

The general lesson is bigger than that one field.  A candidate whose worst measured PCC sits
within ~0.001 of the bar has not been shown to hold it, because the draw alone moves the number
by more than that.  This probe holds everything else fixed - same weights, same prompt, same
post-prefill state, same golden path - and sweeps only the draw, for any set of precision
candidates.

    python probe_draw_sensitivity.py --seq 743 --kinds linear,full --candidates default,no_fp32_acc
    python probe_draw_sensitivity.py --seq 17  --kinds full        --candidates default,bfp8_gate
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys

sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal")

import torch  # noqa: E402
import ttnn  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref  # noqa: E402
from models.autoports.qwen_qwen3_6_27b.tests import harness as H  # noqa: E402
from models.autoports.qwen_qwen3_6_27b.tt import optimized_decoder as O  # noqa: E402

#: 777 is the sweep's draw, ``900 + seq_len`` is the suite's; the rest are arbitrary and fixed.
EXTRA_SEEDS = [11, 12345, 2024, 4242]
LAYER_OF = {"linear_attention": 0, "full_attention": 3}

CANDIDATES = {
    "default": lambda: O.DEFAULT_PRECISION,
    "no_fp32_acc": lambda: dataclasses.replace(O.DEFAULT_PRECISION, proj_fp32_acc=False),
    #: The ``O14`` control: everything shipped except the BFP4 output gate.
    "bfp8_gate": lambda: dataclasses.replace(O.DEFAULT_PRECISION, attn_gate=ttnn.bfloat8_b),
    "bfp4_gate": lambda: dataclasses.replace(O.DEFAULT_PRECISION, attn_gate=ttnn.bfloat4_b),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=743)
    parser.add_argument("--kinds", default="linear,full")
    parser.add_argument("--candidates", default="default,no_fp32_acc")
    args = parser.parse_args()

    seq_len = args.seq
    seeds = [777, 900 + seq_len] + EXTRA_SEEDS
    kinds = {("linear_attention" if k.startswith("linear") else "full_attention"): None
             for k in args.kinds.split(",")}
    names = args.candidates.split(",")

    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=23887872)
    config = ref.load_text_config()
    stats = ref.load_weight_stats()
    hidden = ref.synthetic_hidden_states(config, 1, seq_len, stats)
    tokens = {s: ref.synthetic_hidden_states(config, 1, 1, stats, seed=s) for s in seeds}
    try:
        for kind in kinds:
            layer_idx = LAYER_OF[kind]
            lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=8192,
                                real_weights=True, decoder_cls=O.OptimizedDecoder)
            cache = DynamicCache(config=config)
            golden_prefill = H.reference_prefill(lut, hidden, cache)
            goldens = {}
            for seed in seeds:
                # ``reference_decode`` appends to the cache, so each draw needs its own
                # post-prefill cache rather than a shared one.
                draw_cache = DynamicCache(config=config)
                H.reference_prefill(lut, hidden, draw_cache)
                goldens[seed] = H.reference_decode(lut, tokens[seed], seq_len, draw_cache)
            H.release_layers()

            for name in names:
                lut = H.build_layer(mesh_device, layer_idx, max_batch=1, max_seq_len=8192,
                                    real_weights=True, decoder_cls=O.OptimizedDecoder,
                                    decoder_kwargs={"precision": CANDIDATES[name]()})
                row = {"kind": kind, "candidate": name, "seq_len": seq_len}
                row["prefill_pcc"] = H.pcc(golden_prefill, H.run_tt_prefill(lut, hidden))
                for seed in seeds:
                    # Every draw starts from the same post-prefill state, as the suite does.
                    H.prepare_decode(lut)
                    got = H.run_tt_decode(lut, tokens[seed], torch.tensor([seq_len]))
                    row[f"decode_pcc_seed{seed}"] = H.pcc(goldens[seed], got)
                worst = min(v for k, v in row.items() if k.startswith("decode_pcc_seed"))
                row["decode_worst"] = worst
                row["holds_995"] = bool(worst >= 0.995 and row["prefill_pcc"] >= 0.995)
                print("DRAWS " + json.dumps(row), flush=True)
                H.release_layers()
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
