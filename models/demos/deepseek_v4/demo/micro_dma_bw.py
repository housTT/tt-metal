# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Is the ~6.7 GB/s effective fp4 expert-upload rate a hardware floor or a transfer-pattern issue?
Measures raw host->device ttnn.to_device bandwidth on ONE chip as a function of transfer SIZE and
dtype (bf4 vs bf16 -> byte-rate vs element-rate), then the 4-chip SHARDED aggregate (does the host
feed 4 PCIe links in parallel?). This decides which single-user lever pays off:
  - if BW rises sharply with size  -> coalesce experts into one big transfer.
  - if bf4 and bf16 hit the same GB/s -> byte-bandwidth-bound (fp4 already optimal; need residency).
  - if 4-chip aggregate ~4x single -> multi-chip DMA is the lever (host isn't the bottleneck).
"""
import argparse
import time

import torch

import ttnn


def bench_to_device(dev, host_t, nbytes, iters, mesh_mapper=None):
    # upload the SAME pre-tilized host tensor `iters` times (fresh device buffer each), measure GB/s
    ttnn.to_device(host_t, dev)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        d = ttnn.to_device(host_t, dev)
        ttnn.deallocate(d)
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) / iters
    return nbytes / dt / 1e9, dt


def make_host(rows, cols, dtype, mesh_mapper=None):
    t = torch.randn(rows, cols) * 0.1
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh_mapper)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    # (label, rows, cols) — bf4 bytes = rows*cols*0.5
    sizes = [
        ("12.6MB (1 expert gate_up)", 4096, 6144),
        ("75MB (6 experts)", 4096, 36864),
        ("151MB", 8192, 36864),
        ("302MB", 8192, 73728),
        ("604MB", 16384, 73728),
    ]

    print("=== single chip: bf4 to_device bandwidth vs size ===", flush=True)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        for label, r, c in sizes:
            nbytes = r * c * 0.5  # bf4
            h = make_host(r, c, ttnn.bfloat4_b)
            gbps, dt = bench_to_device(dev, h, nbytes, args.iters)
            print(f"  bf4 {label:28s} {nbytes/1e6:7.1f} MB  {dt*1000:7.2f} ms  {gbps:6.2f} GB/s", flush=True)
        # dtype comparison at fixed element count (151M elem): bf4 vs bf16
        r, c = 8192, 36864
        for dt_lbl, dt_ttnn, bpe in [("bf4", ttnn.bfloat4_b, 0.5), ("bf16", ttnn.bfloat16, 2.0)]:
            nbytes = r * c * bpe
            h = make_host(r, c, dt_ttnn)
            gbps, dtt = bench_to_device(dev, h, nbytes, args.iters)
            print(f"  {dt_lbl:4s} @151M elem  {nbytes/1e6:7.1f} MB  {dtt*1000:7.2f} ms  {gbps:6.2f} GB/s", flush=True)
    finally:
        ttnn.close_mesh_device(dev)

    print("=== 4-chip: SHARDED to_device aggregate bandwidth (does host feed 4 links?) ===", flush=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        shard = ttnn.ShardTensorToMesh(mesh, dim=1)
        for label, r, c in [("302MB total (75MB/chip)", 8192, 73728), ("604MB total (151MB/chip)", 16384, 73728)]:
            nbytes = r * c * 0.5  # total bf4 across all chips
            h = make_host(r, c, ttnn.bfloat4_b, mesh_mapper=shard)
            gbps, dt = bench_to_device(mesh, h, nbytes, args.iters)
            print(f"  bf4 sharded/4 {label:28s} {nbytes/1e6:7.1f} MB total  {dt*1000:7.2f} ms  {gbps:6.2f} GB/s aggregate", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    print("MICRO_DMA_BW_OK", flush=True)


if __name__ == "__main__":
    main()
