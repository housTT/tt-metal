# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint 2 — per-module PCC of the hand-written TT-NN modules vs the HF
DeepSeek-V4 reference (the correctness oracle). Runs on real Blackhole silicon.

Usage:
    source models/demos/deepseek_v4/env.sh
    python models/demos/deepseek_v4/tests/test_modules_pcc.py

Same weights feed both sides, so PCC measures the TT math against HF math directly.
Threshold: 0.99 (submodules target ~0.999 per llms.md §4.4.1).
"""
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v4.reference.reduced_config import build_reduced_model
from models.demos.deepseek_v4.tt import modules as M

torch.manual_seed(0)
RESULTS = []


def check(label, fn, thr=0.99):
    """fn() -> (ref, got); isolated so one module's failure doesn't abort the rest."""
    import traceback

    try:
        ref, got = fn()
        passed, pcc = comp_pcc(ref, got, thr)
        RESULTS.append((label, float(pcc), bool(passed)))
        print(f"[PCC] {label:40s} {float(pcc):.5f}  {'PASS' if passed else 'FAIL'} (>= {thr})")
    except Exception as e:
        RESULTS.append((label, 0.0, False))
        print(f"[PCC] {label:40s} ERROR: {type(e).__name__}: {str(e)[:80]}")
        traceback.print_exc()


def main():
    model, cfg = build_reduced_model(seed=0)
    L0 = model.model.layers[0]
    attn = L0.self_attn
    H = cfg.hidden_size

    dev = ttnn.CreateDevice(device_id=0)
    try:
        # 1) Weighted RMSNorm (input_layernorm)
        def t_rms_w():
            ln = L0.input_layernorm
            x = torch.randn(1, 32, H)
            with torch.no_grad():
                ref = ln(x)
            return ref, M.rms_norm(x, ln.weight.data, dev, eps=cfg.rms_norm_eps)

        check("RMSNorm (weighted)", t_rms_w)

        # 2) Unweighted RMSNorm (q_b_norm, over head_dim)
        def t_rms_u():
            xh = torch.randn(1, cfg.num_attention_heads, 32, cfg.head_dim)
            with torch.no_grad():
                ref = attn.q_b_norm(xh)
            return ref, M.rms_norm(xh, None, dev, eps=cfg.rms_norm_eps)

        check("RMSNorm (unweighted, per-head)", t_rms_u)

        # 3) Clamped SwiGLU (shared expert = DeepseekV4MLP)
        def t_swiglu():
            se = L0.mlp.shared_experts
            x = torch.randn(1, 32, H)
            with torch.no_grad():
                ref = se(x)
            return ref, M.clamped_swiglu_mlp(
                x,
                se.gate_proj.weight.data,
                se.up_proj.weight.data,
                se.down_proj.weight.data,
                dev,
                limit=cfg.swiglu_limit,
            )

        check("Clamped SwiGLU MLP (shared expert)", t_swiglu)

        # 4) Grouped output projection (o_lora, DeepseekV4GroupedLinear)
        def t_grouped():
            oa = attn.o_a_proj
            n_groups = cfg.o_groups
            in_per_group = oa.weight.shape[1]
            x = torch.randn(1, 32, n_groups, in_per_group)
            with torch.no_grad():
                ref = oa(x)
            return ref, M.grouped_linear(x.reshape(1, 32, -1), oa.weight.data, n_groups, dev)

        check("Grouped o_lora projection", t_grouped)

        # 5) sqrtsoftplus router score transform (V4 MoE gate front-end)
        def t_router():
            gate = L0.mlp.gate
            x = torch.randn(1, 32, H)
            with torch.no_grad():
                logits = torch.nn.functional.linear(x, gate.weight.data)
                ref = torch.sqrt(torch.nn.functional.softplus(logits))
            return ref, M.sqrtsoftplus_router_scores(x, gate.weight.data, dev)

        check("sqrtsoftplus router scores", t_router)

        # 6) mHC HyperConnection collapsed output (V4 defining novelty, Sinkhorn residual)
        def t_hc_collapsed():
            hc = L0.attn_hc
            streams = torch.randn(1, 8, cfg.hc_mult, H)
            with torch.no_grad():
                ref_post, ref_comb, ref_collapsed = hc(streams)
            post, comb, collapsed = M.hyperconnection(streams, hc, dev)
            return ref_collapsed, collapsed

        check("mHC HyperConnection: collapsed", t_hc_collapsed)

        # 6b) mHC comb matrix (Sinkhorn-normalized mixing)
        def t_hc_comb():
            hc = L0.attn_hc
            streams = torch.randn(1, 8, cfg.hc_mult, H)
            with torch.no_grad():
                _, ref_comb, _ = hc(streams)
            _, comb, _ = M.hyperconnection(streams, hc, dev)
            return ref_comb, comb

        check("mHC HyperConnection: Sinkhorn comb", t_hc_comb)

        # 7) MoE routed-expert compute (top-k clamped-SwiGLU experts + weighted combine)
        def t_moe_experts():
            experts = L0.mlp.experts
            gate = L0.mlp.gate
            x = torch.randn(1, 8, H)
            flat = x.reshape(-1, H)
            with torch.no_grad():
                # use the reference router to get indices+weights, then reference experts
                logits, weights, indices = gate(flat, input_ids=torch.zeros(flat.shape[0], dtype=torch.long))
                ref = experts(flat, indices, weights)
            got = M.moe_routed_experts(
                flat,
                experts.gate_up_proj.data,
                experts.down_proj.data,
                indices,
                weights,
                dev,
                limit=cfg.swiglu_limit,
            )
            return ref, got

        check("MoE routed experts (top-k SwiGLU)", t_moe_experts)

    finally:
        ttnn.CloseDevice(dev)

    print("\n=== SUMMARY ===")
    npass = sum(1 for _, _, p in RESULTS if p)
    for label, pcc, p in RESULTS:
        print(f"  {'PASS' if p else 'FAIL'}  {pcc:.5f}  {label}")
    print(f"{npass}/{len(RESULTS)} modules >= 0.99 PCC")
    return npass == len(RESULTS)


if __name__ == "__main__":
    ok = main()
    print("MODULE_PCC_RESULT", "ALL_PASS" if ok else "SOME_FAIL")
