"""Host-only: does the measured SDPA output error explain the layer output error?

Takes the tensors dumped by ``probe_sdpa_localise.py`` and pushes both the *device* attention
output and the *reference* attention output through the rest of the HF layer in float32.  If
substituting the device attention output reproduces the layer PCC, the layer error is pure
propagation of the attention error; if it does not, the error is downstream of SDPA.
"""

import os

import torch

from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tests import harness as H

DUMP_DIR = os.environ.get("DUMP_DIR", "/home/ttuser/dev/qwen/rundir/generated")
LENGTHS = [int(x) for x in os.environ.get("LENGTHS", "131071,262143").split(",")]
TAIL = 256


def rest_of_layer(layer, hidden_tail, attn_out):
    """``attn_out`` is [H, T, D] (SDPA output); returns the layer output [1, T, hidden]."""
    attn = layer.self_attn
    with torch.no_grad():
        normed = layer.input_layernorm(hidden_tail)
        t = attn_out.shape[1]
        gate = attn.q_proj(normed).view(1, t, -1, 2 * attn.head_dim)[..., attn.head_dim :]
        gate = gate.reshape(1, t, -1)
        merged = attn_out.permute(1, 0, 2).reshape(1, t, -1)
        gated = merged * torch.sigmoid(gate)
        out = hidden_tail + attn.o_proj(gated)
        return out + layer.mlp(layer.post_attention_layernorm(out))


def main() -> None:
    config = ref.load_text_config()
    stats = ref.load_weight_stats()
    state_dict = ref.synthetic_state_dict_from_stats(stats, H.FULL_LAYER_IDX, config, seed=0)
    layer = ref.build_reference_layer(H.FULL_LAYER_IDX, state_dict=state_dict)

    for length in LENGTHS:
        d = torch.load(f"{DUMP_DIR}/sdpa_localise_{length}.pt")
        hidden = ref.synthetic_hidden_states(config, 1, length, stats, seed=0)
        hidden_tail = hidden[:, -TAIL:, :].contiguous()
        del hidden

        dev_attn = d["dev_attn_tail"]
        ref_attn = d["ref_gold_attn"]
        golden = d["layer_golden_tail"]
        got = d["layer_got_tail"]

        def rel(a, b):
            return float((a - b).norm() / b.norm())

        sim_dev = rest_of_layer(layer, hidden_tail, dev_attn)
        sim_ref = rest_of_layer(layer, hidden_tail, ref_attn)

        print(
            f"RESULT len={length}\n"
            f"  attn:  pcc={H.pcc(ref_attn, dev_attn):.6f} rel_err={rel(dev_attn, ref_attn):.5f} "
            f"norm_ref={float(ref_attn.norm()):.3f}\n"
            f"  layer: pcc(device)={H.pcc(golden, got):.6f} rel_err={rel(got, golden):.5f} "
            f"norm_golden={float(golden.norm()):.3f}\n"
            f"  torch-rest-of-layer on REFERENCE attn: pcc={H.pcc(golden, sim_ref):.6f} "
            f"rel_err={rel(sim_ref, golden):.5f}\n"
            f"  torch-rest-of-layer on DEVICE    attn: pcc={H.pcc(golden, sim_dev):.6f} "
            f"rel_err={rel(sim_dev, golden):.5f}\n"
            f"  device layer out vs torch-rest-on-device-attn: pcc={H.pcc(sim_dev, got):.6f}\n"
            f"  ||o_proj path|| vs ||layer out||: "
            f"{float((sim_ref - hidden_tail).norm() / golden.norm()):.3f}",
            flush=True,
        )
        del d, dev_attn, ref_attn, golden, got, sim_dev, sim_ref


if __name__ == "__main__":
    main()
