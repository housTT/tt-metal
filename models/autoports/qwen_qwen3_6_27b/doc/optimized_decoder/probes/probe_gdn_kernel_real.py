# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Does the ``gated_delta_attn_seq`` path hold accuracy on the *real* layer's activations?

``probe_gdn_kernel.py`` measured PCC 0.9929 for the kernel path against HF on synthetic
N(0, 0.5) inputs - below this stage's 0.995 acceptance bar.  ``$optimize`` OPT-012 is explicit
that a synthetic distribution cannot by itself veto a candidate, so this probe repeats the
comparison at the real operating point: it monkeypatches HF's
``torch_chunk_gated_delta_rule`` to capture the exact ``q/k/v/g/beta`` the real Qwen3.6-27B
layer 0 feeds it during a prefill of real weights, then runs the kernel path on those.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal")
sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes")

import torch  # noqa: E402
import ttnn  # noqa: E402
from probe_gdn_kernel import load_reference, pcc  # noqa: E402

from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref  # noqa: E402

SEQ = 2048


def main():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    captured = {}
    original = m.torch_chunk_gated_delta_rule

    def spy(query, key, value, g, beta, **kwargs):
        captured.setdefault("args", (query.clone(), key.clone(), value.clone(), g.clone(), beta.clone()))
        return original(query, key, value, g, beta, **kwargs)

    config = ref.load_text_config()
    state_dict = ref.load_real_layer_state_dict(0)
    layer = ref.build_reference_layer(0, state_dict={k: v.clone() for k, v in state_dict.items()})
    hidden = ref.synthetic_hidden_states(config, 1, SEQ, ref.load_weight_stats())
    # ``Qwen3_5GatedDeltaNet.__init__`` binds the kernel to ``self.chunk_gated_delta_rule``,
    # so the instance attribute is the one that has to be replaced.
    layer.linear_attn.chunk_gated_delta_rule = spy
    from transformers.cache_utils import DynamicCache

    with torch.no_grad():
        positions = torch.arange(SEQ)
        cos, sin = ref.text_position_embeddings(ref.make_rotary(config), positions, batch=1)
        layer(hidden, position_embeddings=(cos, sin), attention_mask=None,
              position_ids=ref.build_text_position_ids(positions, 1)[0],
              past_key_values=DynamicCache(config=config), use_cache=True)
    layer.linear_attn.chunk_gated_delta_rule = original

    q, k, v, g, beta = captured["args"]
    stats = {"q_absmax": float(q.abs().max()), "k_absmax": float(k.abs().max()),
             "v_absmax": float(v.abs().max()), "g_min": float(g.min()), "g_max": float(g.max()),
             "beta_min": float(beta.min()), "beta_max": float(beta.max()),
             "shape": list(q.shape)}
    print(json.dumps({"captured": stats}), flush=True)

    golden_out, golden_state = original(
        q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
        chunk_size=64, output_final_state=True, use_qk_l2norm_in_kernel=True)

    reference = load_reference()
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        masks = reference.create_chunk_masks_seq(128, device)

        def to_dev(t):
            return ttnn.from_torch(t.float().contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                   device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        out_tt, state_tt = reference.chunk_gated_delta_rule_seq_adapter(
            to_dev(q), to_dev(k), to_dev(v), to_dev(beta), to_dev(g),
            chunk_size=128, initial_state=None, device=device, cached_masks=masks)
        got = ttnn.to_torch(out_tt).float().reshape(golden_out.shape)
        got_state = ttnn.to_torch(state_tt).float().reshape(golden_state.shape)
        print(json.dumps({"real_activations": True, "seq_len": SEQ,
                          "out_pcc": pcc(golden_out, got),
                          "state_pcc": pcc(golden_state, got_state)}), flush=True)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
