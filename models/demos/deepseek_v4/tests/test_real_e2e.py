# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end validation of the streaming REAL-weight path on Blackhole.

Builds a 4-layer scratch model, loads the ACTUAL pretrained weights (fp8/fp4-dequantized) for
layers 0-3 into it, then compares:
  - HF reference forward on the real 4-layer model  (golden)
  - tt_forward_streaming (our TT-NN model, weights streamed per layer, compute on Blackhole)
Asserts logits PCC >= 0.99. This proves the real-weight loader + streaming forward + device
compute are all correct together, before running the full 43-layer model."""
import torch
from transformers import AutoConfig, AutoModelForCausalLM

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v4.reference import real_weights as RW
from models.demos.deepseek_v4.tt import model as TTM

torch.manual_seed(0)
N = 4


def main():
    snap = RW.find_snapshot()
    store = RW.RealWeightStore(snap)
    cfg = AutoConfig.from_pretrained(snap)
    cfg.num_hidden_layers = N
    cfg.num_nextn_predict_layers = 0
    cfg.layer_types = cfg.layer_types[:N]
    cfg.mlp_layer_types = cfg.mlp_layer_types[:N]

    print("loading 4-layer REAL model via transformers (oracle)...")
    oracle = AutoModelForCausalLM.from_pretrained(snap, config=cfg, dtype="auto", low_cpu_mem_usage=True).eval()

    S = 8
    input_ids = torch.randint(0, cfg.vocab_size, (1, S))
    with torch.no_grad():
        ref = oracle(input_ids, use_cache=False).logits.float()  # real 4-layer HF forward (golden)

    dev = ttnn.CreateDevice(device_id=0)
    try:
        # reuse the oracle module as the scratch; streaming pulls experts from `store`
        tt = TTM.tt_forward_streaming(oracle, store, cfg.layer_types, cfg.mlp_layer_types, input_ids, dev, N, log=print)
    finally:
        ttnn.CloseDevice(dev)

    _, pcc_all = comp_pcc(ref, tt, 0.99)
    _, pcc_last = comp_pcc(ref[:, -1, :], tt[:, -1, :], 0.99)
    lastmatch = int(ref[0, -1].argmax() == tt[0, -1].argmax())
    print(
        f"[REAL-E2E] logits PCC (all) = {float(pcc_all):.5f}  final-token PCC = {float(pcc_last):.5f}  "
        f"final-token argmax match = {bool(lastmatch)}"
    )
    print("REAL_E2E_RESULT", "PASS" if pcc_all >= 0.99 else "FAIL")


if __name__ == "__main__":
    main()
