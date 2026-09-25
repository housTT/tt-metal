# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark the fp4-streaming MoE lever.

Isolates the dominant decode cost (per-expert weight upload) and compares:
  (A) CURRENT: host bf16 -> ttnn.from_torch(bfloat16, TILE) per token  (tilize + DMA每token)
  (B) FP4:     pre-tilize to a HOST bfloat4_b ttnn tensor ONCE, then ttnn.to_device per token
                (DMA-only, 1/4 the bytes, zero per-token tilize)

Reports s/expert for each + projected MoE/token (x258) + projected decode tok/s, and the
output PCC of (B) vs (A) so we know whether bf4 weights preserve the argmax. This proves (or
refutes) the path to ~6-7 tok/s WITHOUT needing the full 139 GB host cache — measured on a
sample of experts from one real MoE layer."""
import argparse
import time

import torch

import ttnn
from models.demos.deepseek_v4.reference import real_weights as RW
from models.demos.deepseek_v4.tt import modules as M


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def run_expert_bf16(x, gu_T, dn_T, interm, device, limit):
    """Current path: upload bf16 weights (tilize+DMA) + matmuls."""
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tgu = ttnn.from_torch(gu_T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tdn = ttnn.from_torch(dn_T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    gu = ttnn.matmul(tx, tgu)
    gate = ttnn.clamp(gu[..., :interm], max=limit)
    up = ttnn.clamp(gu[..., interm:], min=-limit, max=limit)
    act = ttnn.multiply(ttnn.silu(gate), up)
    y = ttnn.matmul(act, tdn)
    out = ttnn.to_torch(y)
    for t in (tx, tgu, tdn, gu, gate, up, act, y):
        ttnn.deallocate(t)
    return out


def prebuild_fp4_host(gu_T, dn_T):
    """One-time: tilize + pack to HOST bfloat4_b ttnn tensors (no device)."""
    hgu = ttnn.from_torch(gu_T, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
    hdn = ttnn.from_torch(dn_T, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
    return hgu, hdn


def run_expert_fp4(x, hgu, hdn, interm, device, limit):
    """FP4 path: DMA the pre-tilized bf4 host tensors to device (no tilize) + matmuls."""
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tgu = ttnn.to_device(hgu, device)
    tdn = ttnn.to_device(hdn, device)
    gu = ttnn.matmul(tx, tgu)
    gate = ttnn.clamp(gu[..., :interm], max=limit)
    up = ttnn.clamp(gu[..., interm:], min=-limit, max=limit)
    act = ttnn.multiply(ttnn.silu(gate), up)
    y = ttnn.matmul(act, tdn)
    out = ttnn.to_torch(y)
    for t in (tx, tgu, tdn, gu, gate, up, act, y):
        ttnn.deallocate(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=24)
    ap.add_argument("--layer", type=int, default=3)  # a MoE (non-hash) layer
    args = ap.parse_args()

    from transformers import AutoConfig

    snap = RW.find_snapshot()
    cfg = AutoConfig.from_pretrained(snap)
    interm = cfg.moe_intermediate_size
    H = cfg.hidden_size
    limit = cfg.swiglu_limit
    store = RW.RealWeightStore(snap)

    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        x = torch.randn(1, H) * 0.1  # single decode token activation
        E = args.experts
        print(f"[setup] layer={args.layer} experts={E} H={H} I={interm}", flush=True)

        # Warm the host dequant cache for all E experts (one-time, not counted in per-token cost)
        experts = []
        t0 = time.perf_counter()
        for e in range(E):
            gu_T, dn_T = RW.expert_fused(store, args.layer, e)  # host bf16 (cached dequant)
            experts.append((gu_T, dn_T))
        print(f"[dequant] {E} experts host-dequantized in {time.perf_counter()-t0:.2f}s", flush=True)

        # (A) bf16 per-token upload
        outs_bf16 = []
        t0 = time.perf_counter()
        for gu_T, dn_T in experts:
            outs_bf16.append(run_expert_bf16(x, gu_T, dn_T, interm, dev, limit))
        t_bf16 = (time.perf_counter() - t0) / E

        # (B) fp4: one-time prebuild, then per-token DMA-only
        t0 = time.perf_counter()
        fp4 = [prebuild_fp4_host(gu_T, dn_T) for gu_T, dn_T in experts]
        t_prebuild = (time.perf_counter() - t0) / E
        outs_fp4 = []
        t0 = time.perf_counter()
        for hgu, hdn in fp4:
            outs_fp4.append(run_expert_fp4(x, hgu, hdn, interm, dev, limit))
        t_fp4 = (time.perf_counter() - t0) / E

        pccs = [pcc(a, b) for a, b in zip(outs_bf16, outs_fp4)]
        mean_pcc = sum(pccs) / len(pccs)
        min_pcc = min(pccs)

        print(f"\n[bf16 per-expert]  {t_bf16*1000:.1f} ms  -> MoE/token(258) = {t_bf16*258:.2f}s", flush=True)
        print(f"[fp4  per-expert]  {t_fp4*1000:.1f} ms  -> MoE/token(258) = {t_fp4*258:.2f}s  (prebuild {t_prebuild*1000:.1f} ms/expert one-time)", flush=True)
        print(f"[speedup] {t_bf16/t_fp4:.2f}x per-expert upload", flush=True)
        print(f"[correctness] expert-output PCC fp4-vs-bf16: mean={mean_pcc:.4f} min={min_pcc:.4f}", flush=True)
        # rough decode projection: MoE + ~1s (attn+mhc+lm_head) per token from prior profiling
        for lbl, tmoe in [("bf16", t_bf16 * 258), ("fp4", t_fp4 * 258)]:
            dt = tmoe + 1.0
            print(f"[proj decode {lbl}] ~{dt:.2f}s/tok -> {1.0/dt:.2f} tok/s", flush=True)
        print("MICRO_FP4_OK", flush=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
