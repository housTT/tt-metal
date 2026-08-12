# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Print the checkpoint's own shape constants as a labelled artifact, so documents can be checked against it.

Review round 9 found five documents — including a shipped source comment and the prose inside a *generated*
README block — describing Ornith-1.0-35B as a 48-layer model. It has 40 layers. The figure audit could not
see it: two-digit integers are exempt wholesale (a wrong `in0_block_w` would otherwise be unsourceable
noise), and a layer count spelled out in words is not a number at all. The claim was load-bearing — it is
the denominator of the compounding argument that rejects the BFP4 dense-projection candidate.

So the facts get an artifact of their own, and `audit_figures.check_model_facts` asserts every document's
layer/expert/head claims against it, spelled-out numerals included.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_decoder/logs/model_facts.py

Rows are ``FACT <name>=<value>``, read straight off the HF config of the snapshot the tests load.
"""

from __future__ import annotations

from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R


def facts() -> dict:
    cfg = R.load_text_config()
    n = int(cfg.num_hidden_layers)
    return {
        "num_hidden_layers": n,
        # The compounding argument's phrasing ("one layer of forty, a bet on the other thirty-nine") needs
        # this too, and deriving it in the checker would let the derivation drift from the sentence.
        "num_layers_minus_one": n - 1,
        "num_layer_types": len(cfg.layer_types),
        "num_experts": int(cfg.num_experts),
        "num_experts_per_tok": int(cfg.num_experts_per_tok),
        "moe_intermediate_size": int(cfg.moe_intermediate_size),
        "hidden_size": int(cfg.hidden_size),
        "num_attention_heads": int(cfg.num_attention_heads),
        "num_key_value_heads": int(cfg.num_key_value_heads),
        "head_dim": int(cfg.head_dim),
        "max_position_embeddings": int(cfg.max_position_embeddings),
    }


def main() -> None:
    print("# Shape constants read off the ornith-ai/Ornith-1.0-35B text config, for audit_figures.py.")
    for name, value in facts().items():
        print(f"FACT {name}={value}")


if __name__ == "__main__":
    main()
