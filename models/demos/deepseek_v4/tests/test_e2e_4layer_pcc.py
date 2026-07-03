# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end PCC for the FULL layer-type mix: 4 layers = 2 sliding + 1 CSA (compressor +
lightning indexer) + 1 HCA, so the compressed long-range KV path is exercised inside the
assembled TT-NN model. Compares logits to the HF DeepseekV4 reference. On Blackhole.

seq_len 32 > CSA window (m=4) so CSA emits 8 compressed entries + runs the indexer; HCA
(m=128) emits 0 at this length and degenerates to sliding (still exercises its code path)."""
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v4.reference.reduced_config import build_reduced_model
from models.demos.deepseek_v4.tt import model as TTM

torch.manual_seed(0)


def main():
    hf, cfg = build_reduced_model(seed=0, dtype=torch.bfloat16, overrides={"num_hidden_layers": 4})
    print("layer_types:", cfg.layer_types)
    S = 32
    input_ids = torch.randint(0, cfg.vocab_size, (1, S))
    with torch.no_grad():
        ref_logits = hf(input_ids, use_cache=False).logits.float()

    dev = ttnn.CreateDevice(device_id=0)
    try:
        tt_logits = TTM.tt_forward(hf, input_ids, dev)
    finally:
        ttnn.CloseDevice(dev)

    passed_all, pcc_all = comp_pcc(ref_logits, tt_logits, 0.99)
    passed_last, pcc_last = comp_pcc(ref_logits[:, -1, :], tt_logits[:, -1, :], 0.99)
    ref_tok = ref_logits.argmax(-1)
    tt_tok = tt_logits.argmax(-1)
    tok_match = int((ref_tok == tt_tok).sum())
    last_match = int(ref_tok[0, -1] == tt_tok[0, -1])

    print(f"[E2E-4L] logits PCC (all)        = {float(pcc_all):.5f}  {'PASS' if passed_all else 'FAIL'}")
    print(f"[E2E-4L] logits PCC (final token)= {float(pcc_last):.5f}  {'PASS' if passed_last else 'FAIL'}")
    print(f"[E2E-4L] teacher-forced token match = {tok_match}/{ref_tok.numel()}  final-token match={bool(last_match)}")
    ok = passed_all
    print("E2E_4L_RESULT", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    main()
