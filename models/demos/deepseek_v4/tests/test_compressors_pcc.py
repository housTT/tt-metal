# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Per-module PCC for the CSA/HCA KV compressors + lightning indexer vs the HF
DeepseekV4 reference (stateless single-shot mode, past_key_values=None). On Blackhole.

A 4-layer reduced config exposes layer 2 = compressed_sparse_attention (CSA, has indexer)
and layer 3 = heavily_compressed_attention (HCA). We call each HF compressor with
past_key_values=None and compare our ported TT-NN implementation."""
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v4.reference.reduced_config import build_reduced_model
from models.demos.deepseek_v4.tt import compressors as C

torch.manual_seed(0)
RESULTS = []


def rec(label, ok, detail=""):
    RESULTS.append((label, ok))
    print(f"[CMP] {label:44s} {'PASS' if ok else 'FAIL'}  {detail}")


def main():
    hf, cfg = build_reduced_model(seed=0, dtype=torch.float32, overrides={"num_hidden_layers": 4})
    print("layer_types:", cfg.layer_types)
    S = 160  # > HCA compress_rate (128) so HCA forms >=1 window; CSA (m=4) forms 40
    H = cfg.hidden_size
    hidden = torch.randn(1, S, H)
    q_residual = torch.randn(1, S, cfg.q_lora_rank)
    position_ids = torch.arange(S).unsqueeze(0)

    # locate a CSA layer (has compressor + indexer) and an HCA layer
    csa_layer = next(i for i, t in enumerate(cfg.layer_types) if t == "compressed_sparse_attention")
    hca_layer = next(i for i, t in enumerate(cfg.layer_types) if t == "heavily_compressed_attention")
    csa = hf.model.layers[csa_layer].self_attn.compressor
    hca = hf.model.layers[hca_layer].self_attn.compressor

    dev = ttnn.CreateDevice(device_id=0)
    try:
        # --- HCA compressor: compressed_kv PCC ---
        with torch.no_grad():
            ref_ckv, _ = hca(hidden, q_residual, position_ids, None, hca_layer)
        got_ckv = C.hca_compressor(hidden, hca, cfg, dev)
        _, pcc = comp_pcc(ref_ckv, got_ckv, 0.99)
        rec("HCA compressor: compressed_kv", pcc >= 0.99, f"PCC={float(pcc):.5f} shape={tuple(ref_ckv.shape)}")

        # --- CSA indexer: top-k index selection match ---
        with torch.no_grad():
            ref_tki = csa.indexer(hidden, q_residual, position_ids, None, csa_layer)
        got_tki = C.csa_indexer(hidden, q_residual, position_ids, csa.indexer, cfg, dev)
        # compare selected index SETS per query (topk order can differ on ties; sets should match)
        ref_sets = [set(r[r >= 0].tolist()) for r in ref_tki[0]]
        got_sets = [set(g[g >= 0].tolist()) for g in got_tki[0]]
        # empty-vs-empty (early queries below the causal threshold) counts as a match
        jac = sum(1.0 if not (a | b) else len(a & b) / len(a | b) for a, b in zip(ref_sets, got_sets)) / len(ref_sets)
        rec("CSA indexer: top-k selection", jac >= 0.99, f"mean set-Jaccard={jac:.4f}")

        # --- CSA compressor: compressed_kv PCC + block_bias structure ---
        with torch.no_grad():
            ref_ckv2, ref_bb = csa(hidden, q_residual, position_ids, None, csa_layer)
        got_ckv2, got_bb = C.csa_compressor(hidden, q_residual, position_ids, csa, cfg, dev)
        _, pcc2 = comp_pcc(ref_ckv2, got_ckv2, 0.99)
        rec("CSA compressor: compressed_kv", pcc2 >= 0.99, f"PCC={float(pcc2):.5f}")
        # block_bias: fraction of attendable (0.0) slots that agree
        ref_attend = ref_bb == 0.0
        got_attend = got_bb == 0.0
        agree = float((ref_attend == got_attend).float().mean())
        rec("CSA compressor: block_bias mask", agree >= 0.99, f"mask agreement={agree:.4f}")
    finally:
        ttnn.CloseDevice(dev)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} compressor/indexer checks pass")
    print("COMPRESSORS_RESULT", "ALL_PASS" if npass == len(RESULTS) else "SOME_FAIL")


if __name__ == "__main__":
    main()
