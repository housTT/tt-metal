# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Validate the streaming real-weight loader against the transformers-loaded model (oracle).

Loads the ACTUAL DeepSeek-V4-Flash weights for a few layers via transformers (which dequantizes
fp8->bf16 and converts native->HF names), then checks our RealWeightStore mapping + dequant
reproduces the same tensors (allclose). Covers a sliding+hash layer, a CSA+hash layer, an
HCA+moe layer, and the globals. CPU-only (no device needed)."""
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from models.demos.deepseek_v4.reference import real_weights as RW

N = 4  # layers 0,1 sliding+hash ; 2 CSA+hash ; 3 HCA+moe
FAILS = []


def check(label, a, b, atol=2e-2, rtol=2e-2):
    ok = a.shape == b.shape and torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)
    maxdiff = (a.float() - b.float()).abs().max().item() if a.shape == b.shape else float("nan")
    print(f"  [{'OK ' if ok else 'BAD'}] {label:52s} max|Δ|={maxdiff:.4g} shape={tuple(a.shape)}")
    if not ok:
        FAILS.append(label)


def main():
    snap = RW.find_snapshot()
    store = RW.RealWeightStore(snap)
    cfg = AutoConfig.from_pretrained(snap)
    cfg.num_hidden_layers = N
    cfg.num_nextn_predict_layers = 0
    cfg.layer_types = cfg.layer_types[:N]
    cfg.mlp_layer_types = cfg.mlp_layer_types[:N]
    print(f"Loading {N}-layer transformers oracle (real weights)...")
    oracle = AutoModelForCausalLM.from_pretrained(snap, config=cfg, dtype="auto", low_cpu_mem_usage=True).eval()
    osd = oracle.state_dict()

    print("=== globals ===")
    for hf_k, nat in RW.GLOBAL_MAP.items():
        check(hf_k, store.deq(nat), osd[hf_k])

    for i in range(N):
        print(f"=== layer {i} ({cfg.layer_types[i]}, {cfg.mlp_layer_types[i]}) ===")
        p = f"layers.{i}."
        layer_sd = {k: v for k, v in osd.items() if k.startswith(f"model.layers.{i}.")}
        for full_k, ov in layer_sd.items():
            hf_k = full_k[len(f"model.layers.{i}.") :]
            if hf_k in RW.LAYER_MAP:
                check(hf_k, store.deq(p + RW.LAYER_MAP[hf_k]), ov)
            elif hf_k == "mlp.experts.gate_up_proj":
                # spot-check experts 0 and 7 (assembled cat(w1,w3))
                for e in (0, 7):
                    mine = torch.cat(
                        [store.deq(f"{p}ffn.experts.{e}.w1.weight"), store.deq(f"{p}ffn.experts.{e}.w3.weight")], dim=0
                    )
                    check(f"experts.gate_up_proj[{e}]", mine, ov[e])
            elif hf_k == "mlp.experts.down_proj":
                for e in (0, 7):
                    check(f"experts.down_proj[{e}]", store.deq(f"{p}ffn.experts.{e}.w2.weight"), ov[e])
            else:
                print(f"  [?? ] unmapped HF key: {hf_k}")

    print(f"\n{'ALL_MATCH' if not FAILS else 'MISMATCH: ' + str(FAILS)}")
    print("REAL_WEIGHTS_RESULT", "PASS" if not FAILS else "FAIL")


if __name__ == "__main__":
    main()
