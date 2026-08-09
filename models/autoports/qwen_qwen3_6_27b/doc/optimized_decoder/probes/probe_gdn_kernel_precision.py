# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``gated_delta_attn_seq`` (``O5``) retried with the precision knobs that *are* reachable.

The first evaluation (``probe_gdn_kernel_real.py``) ran the built tree's canonical caller
unmodified and measured 0.9897 output / 0.9829 state PCC on the real layer's activations.  Two
things in that caller are Python-side choices rather than kernel properties, and both were in
the loop:

* every preprocessing matmul and the whole ``L_inv`` solve run at **HiFi2 without
  ``packer_l1_acc``** (``ttnn_delta_rule_seq.py`` lines 198-203, 330-334), while this decoder's
  own triangular inverse deliberately uses HiFi4 + ``fp32_dest_acc_en`` because the functional
  stage measured cancellation there;
* the adapter **typecasts the returned final state to bfloat16** before handing it back
  (lines 138-141), so the state PCC was measured through a downcast the decoder would not keep.

This probe removes both: it rewrites the loaded source's compute-kernel configs to
HiFi4 + fp32 accumulation, and calls the inner ``chunk_gated_delta_rule_seq`` directly so the
float32 final state is compared as float32.  Same real activations as the first probe.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types

sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal")
sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes")

import torch  # noqa: E402
import ttnn  # noqa: E402
from probe_gdn_kernel import pcc  # noqa: E402

from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref  # noqa: E402

BUILT = "/home/ttuser/.local/lib/model-bringup/tt-metal"
SRC = f"{BUILT}/models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_seq.py"
SEQ = 2048


def load_reference(high_precision: bool):
    """Load the built tree's seq-kernel module, optionally at HiFi4 + fp32 accumulation."""
    sys.path.insert(0, BUILT)
    source = open(SRC).read()
    if high_precision:
        source = source.replace("math_fidelity=ttnn.MathFidelity.HiFi2",
                                "math_fidelity=ttnn.MathFidelity.HiFi4")
        source = source.replace("packer_l1_acc=False", "packer_l1_acc=True")
        assert "HiFi4" in source
    name = "ref_seq_hi" if high_precision else "ref_seq_lo"
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = SRC
    module.__package__ = "models.experimental.gated_attention_gated_deltanet.tt"
    module.__dict__["__name__"] = name
    sys.modules[name] = module
    exec(compile(source, SRC, "exec"), module.__dict__)
    return module


def capture_real_activations():
    """The exact q/k/v/g/beta the real Qwen3.6-27B layer 0 feeds the delta rule."""
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    captured = {}
    original = m.torch_chunk_gated_delta_rule
    config = ref.load_text_config()
    state_dict = ref.load_real_layer_state_dict(0)
    layer = ref.build_reference_layer(0, state_dict={k: v.clone() for k, v in state_dict.items()})
    hidden = ref.synthetic_hidden_states(config, 1, SEQ, ref.load_weight_stats())

    def spy(query, key, value, g, beta, **kwargs):
        captured.setdefault("args", (query.clone(), key.clone(), value.clone(), g.clone(), beta.clone()))
        return original(query, key, value, g, beta, **kwargs)

    layer.linear_attn.chunk_gated_delta_rule = spy
    with torch.no_grad():
        positions = torch.arange(SEQ)
        cos, sin = ref.text_position_embeddings(ref.make_rotary(config), positions, batch=1)
        layer(hidden, position_embeddings=(cos, sin), attention_mask=None,
              position_ids=ref.build_text_position_ids(positions, 1)[0],
              past_key_values=DynamicCache(config=config), use_cache=True)
    return captured["args"], original


def run(module, device, q, k, v, beta, g, chunk_size=128):
    """The adapter's body, minus its bfloat16 downcast of the final state."""
    B, T, H, K = q.shape
    V = v.shape[3]
    BH = B * H
    dram = ttnn.DRAM_MEMORY_CONFIG

    def to_dev(t):
        return ttnn.from_torch(t.float().contiguous(), dtype=ttnn.float32,
                               layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)

    qd, kd, vd = to_dev(q), to_dev(k), to_dev(v)
    qd = module.l2_norm_ttnn(qd, dim=-1)
    kd = module.l2_norm_ttnn(kd, dim=-1)

    def to_bhtd(t, D):
        t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
        t = ttnn.reshape(t, [B, T, H, D])
        t = ttnn.permute(t, (0, 2, 1, 3))
        t = ttnn.reshape(t, [BH, T, D])
        t = ttnn.to_layout(t, ttnn.TILE_LAYOUT, memory_config=dram)
        return t if t.dtype == ttnn.float32 else ttnn.typecast(t, ttnn.float32, memory_config=dram)

    def to_bht(t):
        t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
        t = ttnn.reshape(t, [B, T, H])
        t = ttnn.permute(t, (0, 2, 1))
        t = ttnn.reshape(t, [BH, T])
        t = ttnn.to_layout(t, ttnn.TILE_LAYOUT, memory_config=dram)
        return t if t.dtype == ttnn.float32 else ttnn.typecast(t, ttnn.float32, memory_config=dram)

    masks = module.create_chunk_masks_seq(chunk_size, device)
    o_bh, final_state = module.chunk_gated_delta_rule_seq(
        to_bhtd(qd, K), to_bhtd(kd, K), to_bhtd(vd, V),
        ttnn.reshape(to_bht(to_dev(beta)), [BH, T, 1]), to_bht(to_dev(g)),
        chunk_size=chunk_size, initial_state=None, mesh_device=device, cached_masks=masks)
    o = ttnn.to_layout(o_bh, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
    o = ttnn.reshape(o, [B, H, T, V])
    o = ttnn.permute(o, (0, 2, 1, 3))
    o = ttnn.to_layout(o, ttnn.TILE_LAYOUT, memory_config=dram)
    return (ttnn.to_torch(o).float().reshape(B, T, H, V),
            ttnn.to_torch(final_state).float().reshape(B, H, K, V),
            str(final_state.dtype))


def main():
    (q, k, v, g, beta), original = capture_real_activations()
    golden_out, golden_state = original(
        q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
        chunk_size=64, output_final_state=True, use_qk_l2norm_in_kernel=True)

    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        for label, high in (("as_shipped_hifi2", False), ("hifi4_fp32acc", True)):
            module = load_reference(high)
            out, state, state_dtype = run(module, device, q, k, v, beta, g)
            print(json.dumps({
                "candidate": label, "seq_len": SEQ, "real_activations": True,
                "kernel_state_dtype_before_readback": state_dtype,
                "out_pcc": pcc(golden_out, out), "state_pcc": pcc(golden_state, state),
            }), flush=True)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
