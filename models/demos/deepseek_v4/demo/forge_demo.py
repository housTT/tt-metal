# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint 1 — tt-forge (tt-xla / PJRT) automatic compiler path.

Proves whether DeepSeek-V4's ops are supported by the automatic compiler path and
whether it computes the right answer, by running a REDUCED-config DeepseekV4 through
torch_xla → StableHLO → tt-mlir → TTNN on real Blackhole and comparing logits to a
CPU reference (PCC ≥ 0.99).

This is NOT a performance test. Runs in the isolated `.forge-venv` (torch 2.10 +
torch_xla 2.9 + pjrt-plugin-tt 1.3.0). Uses a tiny random-weight config so the SAME
weights feed both CPU ref and TT — PCC compares like-with-like without a 160 GB
download, while still exercising the full V4 op set through the compiler.
"""
import os
import sys
import traceback

import torch
from transformers import AutoConfig, AutoModelForCausalLM

HF_MODEL = "deepseek-ai/DeepSeek-V4-Flash"

# Reduced but architecturally-complete config (same knobs as reference/reduced_config.py).
REDUCED = dict(
    num_hidden_layers=2,  # keep tiny for the compiler; 2 sliding + hash_moe layers
    hidden_size=128,
    intermediate_size=256,
    moe_intermediate_size=64,
    n_routed_experts=4,
    num_experts_per_tok=2,
    n_shared_experts=1,
    num_attention_heads=4,
    head_dim=64,
    q_lora_rank=64,
    o_lora_rank=64,
    o_groups=8,
    index_n_heads=4,
    index_head_dim=32,
    index_topk=16,
    vocab_size=512,
    num_nextn_predict_layers=0,
)


def build_reduced():
    cfg = AutoConfig.from_pretrained(HF_MODEL)
    for k, v in REDUCED.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    n = cfg.num_hidden_layers
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = cfg.layer_types[:n]
    if getattr(cfg, "mlp_layer_types", None):
        cfg.mlp_layer_types = cfg.mlp_layer_types[:n]
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval()
    return model, cfg


def pcc(a, b):
    a = a.flatten().float() - a.flatten().float().mean()
    b = b.flatten().float() - b.flatten().float().mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-20))


def main():
    model, cfg = build_reduced()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        ref = model(ids, use_cache=False).logits
    print(f"[cpu-ref] logits {tuple(ref.shape)}  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # --- torch_xla / TT device path ---
    os.environ.setdefault("XLA_REGISTER_INSTALLED_PLUGINS", "1")
    os.environ.setdefault("PJRT_DEVICE", "TT")
    import torch_xla
    import torch_xla.core.xla_model as xm

    try:
        dev = torch_xla.device()
    except Exception:
        dev = xm.xla_device()
    print(f"[tt] xla device = {dev}")

    model_tt = model.to(dev)
    ids_tt = ids.to(dev)
    with torch.no_grad():
        out = model_tt(ids_tt, use_cache=False).logits
    xm.mark_step()
    tt_logits = out.cpu()
    v = pcc(ref[:, -1, :], tt_logits[:, -1, :])
    print(f"[RESULT] final-logits PCC (cpu vs TT) = {v:.5f}  (threshold 0.99)")
    ref_tok = ref[:, -1, :].argmax(-1).item()
    tt_tok = tt_logits[:, -1, :].argmax(-1).item()
    print(f"[RESULT] argmax token  cpu={ref_tok}  tt={tt_tok}  match={ref_tok == tt_tok}")
    print("CHECKPOINT_1_RESULT", "PASS" if v >= 0.99 and ref_tok == tt_tok else "PCC/token below bar")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("CHECKPOINT_1_RESULT COMPILE_OR_RUNTIME_ERROR")
        traceback.print_exc()
        sys.exit(0)  # non-fatal: the failure mode IS the Checkpoint-1 finding
