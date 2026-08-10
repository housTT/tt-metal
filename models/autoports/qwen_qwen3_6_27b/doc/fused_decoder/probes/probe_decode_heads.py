# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""What ``nlp_create_qkv_heads_decode`` hands back, and whether the reshards around it are real.

The functional decoder converts q/k/v to DRAM straight after ``nlp_create_qkv_heads_decode``
and then converts K/V *back* to a hand-built height-sharded config for ``paged_update_cache``.
This probe prints the op's native memory config next to that hand-built one so the round trip
can be removed if they are the same, and checks that ``paged_fused_update_cache`` accepts the
native one.

    python .../probes/probe_decode_heads.py
"""

from __future__ import annotations

import torch

import ttnn

HIDDEN = 5120
HEAD_DIM = 256
N_HEADS = 24
N_KV = 4
PADDED_HEADS = 32
BLOCK = 64


def main() -> None:
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        for batch in (1, 4, 32):
            qkv = ttnn.from_torch(
                torch.randn(1, 1, batch, (N_HEADS + 2 * N_KV) * HEAD_DIM),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(qkv, num_heads=N_HEADS, num_kv_heads=N_KV)
            hand = ttnn.create_sharded_memory_config(
                shape=(PADDED_HEADS, HEAD_DIM),
                core_grid=ttnn.num_cores_to_corerangeset(batch, ttnn.CoreCoord(8, 8), row_wise=True),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            print(f"batch={batch}")
            print(f"  q shape={list(q.shape)} mem={q.memory_config()}")
            print(f"  k shape={list(k.shape)} mem={k.memory_config()}")
            print(f"  hand-built decode_head_mem_cfg = {hand}")
            print(f"  k.memory_config() == hand -> {k.memory_config() == hand}")

            blocks = batch * 8
            kc = ttnn.from_torch(
                torch.zeros(blocks, N_KV, BLOCK, HEAD_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            vc = ttnn.from_torch(
                torch.zeros(blocks, N_KV, BLOCK, HEAD_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            pt = ttnn.from_torch(
                torch.arange(blocks, dtype=torch.int32).reshape(batch, 8),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
            pos = ttnn.from_torch(
                torch.full((batch,), 5, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
            try:
                ttnn.experimental.paged_fused_update_cache(kc, k, vc, v, update_idxs_tensor=pos, page_table=pt)
                print("  paged_fused_update_cache(native sharded k/v) -> OK")
            except Exception as exc:  # noqa: BLE001
                print(f"  paged_fused_update_cache(native sharded k/v) -> FAILED: {str(exc)[:200]}")
            for t in (qkv, q, k, v, kc, vc, pt, pos):
                if t.is_allocated():
                    ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
