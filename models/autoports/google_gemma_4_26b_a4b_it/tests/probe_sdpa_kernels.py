# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Op-level isolation of fresh-versus-continuation attention numerics on one chip.

Usage (from the tt-metal worktree):
    TT_VISIBLE_DEVICES=3 ... python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_sdpa_kernels \
        --output /tmp/sdpa_kernels.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import (
    _prefill_sdpa_compute_config,
    _prefill_sdpa_program_config,
)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten() - a.float().mean()
    b = b.float().flatten() - b.float().mean()
    denom = float(a.norm() * b.norm())
    return float(a @ b) / denom if denom else 1.0


def _row_pcc_min(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1, a.shape[-2], a.shape[-1])
    b = b.float().reshape(-1, b.shape[-2], b.shape[-1])
    worst = 1.0
    for head in range(a.shape[0]):
        ac = a[head] - a[head].mean(dim=-1, keepdim=True)
        bc = b[head] - b[head].mean(dim=-1, keepdim=True)
        pcc = (ac * bc).sum(-1) / (ac.norm(dim=-1) * bc.norm(dim=-1)).clamp_min(1e-12)
        worst = min(worst, float(pcc.min()))
    return worst


def _device(t: torch.Tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _host(t: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(t).float()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=2560)
    parser.add_argument("--starts", type=str, default="64,1024,2432")
    parser.add_argument("--output", type=Path, default=Path("/tmp/gemma4_sdpa_kernels.json"))
    args = parser.parse_args()
    starts = [int(v) for v in args.starts.split(",")]
    length = args.length
    device = ttnn.open_device(device_id=0)
    report: dict = {"length": length, "cases": []}
    try:
        program_config = _prefill_sdpa_program_config(device)
        compute_config = _prefill_sdpa_compute_config(device)
        torch.manual_seed(0)
        for kind, q_heads, kv_heads, head_dim, window, block in (
            ("full_attention", 16, 2, 512, None, 128),
            ("sliding_attention", 16, 8, 256, 1024, 64),
        ):
            q = torch.randn(1, q_heads, length, head_dim, dtype=torch.bfloat16)
            k = torch.randn(1, kv_heads, length, head_dim, dtype=torch.bfloat16)
            v = torch.randn(1, kv_heads, length, head_dim, dtype=torch.bfloat16)
            tq, tk, tv = _device(q, device), _device(k, device), _device(v, device)
            reference = _host(
                ttnn.transformer.scaled_dot_product_attention(
                    tq,
                    tk,
                    tv,
                    is_causal=True,
                    sliding_window_size=window,
                    scale=1.0,
                    program_config=program_config,
                    compute_kernel_config=compute_config,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
            repeat = _host(
                ttnn.transformer.scaled_dot_product_attention(
                    tq,
                    tk,
                    tv,
                    is_causal=True,
                    sliding_window_size=window,
                    scale=1.0,
                    program_config=program_config,
                    compute_kernel_config=compute_config,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
            report["cases"].append(
                {
                    "kind": kind,
                    "case": "non_chunked_repeat",
                    "pcc": _pcc(reference, repeat),
                    "row_pcc_min": _row_pcc_min(reference, repeat),
                    "max_abs": float((reference - repeat).abs().max()),
                }
            )
            if window is None:
                blocks = length // block
                cache_k = ttnn.zeros(
                    (blocks + 1, kv_heads, block, head_dim),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                cache_v = ttnn.zeros(
                    (blocks + 1, kv_heads, block, head_dim),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                table = _device(
                    torch.arange(1, blocks + 1, dtype=torch.int32).reshape(1, blocks),
                    device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                ttnn.experimental.paged_fill_cache(cache_k, tk, table, batch_idx=0)
                ttnn.experimental.paged_fill_cache(cache_v, tv, table, batch_idx=0)
                geometry = ttnn.PagedCacheGeometryOverride(block_size=block, num_kv_heads=kv_heads)
                chunked0 = _host(
                    ttnn.transformer.chunked_scaled_dot_product_attention(
                        tq,
                        cache_k,
                        cache_v,
                        table,
                        chunk_start_idx=0,
                        scale=1.0,
                        program_config=program_config,
                        compute_kernel_config=compute_config,
                        paged_cache_geometry=geometry,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                )
                report["cases"].append(
                    {
                        "kind": kind,
                        "case": "chunked_start0_vs_non_chunked",
                        "pcc": _pcc(reference, chunked0),
                        "row_pcc_min": _row_pcc_min(reference, chunked0),
                        "max_abs": float((reference - chunked0).abs().max()),
                    }
                )
                for start in starts:
                    aligned = start - start % 32
                    q_suffix = _device(q[:, :, aligned:, :], device)
                    small = ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                        q_chunk_size=32,
                        k_chunk_size=32,
                    )
                    for label, config in (("prog128", program_config), ("prog32", small)):
                        if label == "prog128" and aligned % 128 != 0:
                            continue
                        out = _host(
                            ttnn.transformer.chunked_scaled_dot_product_attention(
                                q_suffix,
                                cache_k,
                                cache_v,
                                table,
                                chunk_start_idx=aligned,
                                scale=1.0,
                                program_config=config,
                                compute_kernel_config=compute_config,
                                paged_cache_geometry=geometry,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            )
                        )
                        ref_rows = reference[:, :, aligned:, :]
                        ch_rows = chunked0[:, :, aligned:, :]
                        report["cases"].append(
                            {
                                "kind": kind,
                                "case": f"chunked_start{aligned}_{label}",
                                "vs_non_chunked_pcc": _pcc(ref_rows, out),
                                "vs_non_chunked_row_pcc_min": _row_pcc_min(ref_rows, out),
                                "vs_chunked0_pcc": _pcc(ch_rows, out),
                                "vs_chunked0_row_pcc_min": _row_pcc_min(ch_rows, out),
                                "vs_chunked0_max_abs": float((ch_rows - out).abs().max()),
                            }
                        )
                    ttnn.deallocate(q_suffix)
            else:
                for start in starts:
                    history_len = window - 1
                    first = max(0, start - history_len)
                    for label, pad in (("unaligned", 0), ("aligned128", first % 128)):
                        rows = pad + (start - first)
                        q_in = torch.cat(
                            [torch.zeros(1, q_heads, rows, head_dim, dtype=torch.bfloat16), q[:, :, start:, :]], dim=2
                        )
                        k_in = torch.cat(
                            [torch.zeros(1, kv_heads, pad, head_dim, dtype=torch.bfloat16), k[:, :, first:, :]], dim=2
                        )
                        v_in = torch.cat(
                            [torch.zeros(1, kv_heads, pad, head_dim, dtype=torch.bfloat16), v[:, :, first:, :]], dim=2
                        )
                        total = q_in.shape[2]
                        padded = (total + 31) // 32 * 32
                        if padded != total:
                            q_in = torch.nn.functional.pad(q_in, (0, 0, 0, padded - total))
                            k_in = torch.nn.functional.pad(k_in, (0, 0, 0, padded - total))
                            v_in = torch.nn.functional.pad(v_in, (0, 0, 0, padded - total))
                        out = _host(
                            ttnn.transformer.scaled_dot_product_attention(
                                _device(q_in, device),
                                _device(k_in, device),
                                _device(v_in, device),
                                is_causal=True,
                                sliding_window_size=window,
                                scale=1.0,
                                program_config=program_config,
                                compute_kernel_config=compute_config,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            )
                        )[:, :, rows : rows + (length - start), :]
                        ref_rows = reference[:, :, start:, :]
                        report["cases"].append(
                            {
                                "kind": kind,
                                "case": f"history_start{start}_{label}",
                                "pad": pad,
                                "pcc": _pcc(ref_rows, out),
                                "row_pcc_min": _row_pcc_min(ref_rows, out),
                                "max_abs": float((ref_rows - out).abs().max()),
                            }
                        )
            for case in report["cases"]:
                if "printed" not in case:
                    case["printed"] = True
                    print(json.dumps({k: v for k, v in case.items() if k != "printed"}), flush=True)
    finally:
        ttnn.close_device(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
