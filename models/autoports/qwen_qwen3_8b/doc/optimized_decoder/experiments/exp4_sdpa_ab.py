"""SDPA-decode A/B: two-op K/V cache update versus fused update, 5 measured traced replays each, under Tracy signposts."""
import json
import sys

import torch

import ttnn

sys.path.insert(0, "/home/hous/dev/tt-metal-wt-qwen3")
from models.autoports.qwen_qwen3_8b.tt.functional_decoder import (
    _decode_height_sharded_memory_config,
    decode_sdpa_k_chunk_size,
)
from models.autoports.qwen_qwen3_8b.tt.fused_decoder import _decode_v_update_memory_config

try:
    from tracy import signpost
except ImportError:
    signpost = lambda *_, **__: None

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 * 1024 * 1024)
ckc = ttnn.init_device_compute_kernel_config(
    mesh.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)
REPLAYS = 5
rows = []
for batch in (1, 32):
    blocks = 64
    extent = blocks * 32
    torch.manual_seed(1)
    kcache = ttnn.from_torch(
        torch.randn(batch * blocks, 8, 32, 128).to(torch.bfloat16) * 0.5,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    vcache = ttnn.from_torch(
        torch.randn(batch * blocks, 8, 32, 128).to(torch.bfloat16) * 0.5,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    table = torch.arange(batch * blocks, dtype=torch.int32).reshape(batch, blocks)
    page_table = ttnn.from_torch(
        table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    cur_pos = ttnn.from_torch(
        torch.full((batch,), 1024, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    heads_cfg = _decode_height_sharded_memory_config(mesh, batch, 128)
    v_cfg = _decode_v_update_memory_config(mesh, batch, 128)
    q = ttnn.from_torch(
        torch.randn(1, batch, 32, 128).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=heads_cfg,
    )
    k = ttnn.from_torch(
        torch.randn(1, batch, 8, 128).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=heads_cfg,
    )
    v_same = ttnn.from_torch(
        torch.randn(1, batch, 8, 128).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=heads_cfg,
    )
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=32,
        k_chunk_size=decode_sdpa_k_chunk_size(extent),
        exp_approx_mode=False,
    )

    def sdpa():
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            kcache,
            vcache,
            page_table_tensor=page_table,
            cur_pos_tensor=cur_pos,
            scale=128**-0.5,
            program_config=pc,
            compute_kernel_config=ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def two_op():
        ttnn.experimental.paged_update_cache(kcache, k, update_idxs_tensor=cur_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(vcache, v_same, update_idxs_tensor=cur_pos, page_table=page_table)
        return sdpa()

    def fused():
        v_moved = ttnn.to_memory_config(v_same, v_cfg)
        ttnn.experimental.paged_fused_update_cache(
            kcache, k, vcache, v_moved, update_idxs_tensor=cur_pos, page_table=page_table
        )
        v_moved.deallocate(True)
        return sdpa()

    def sdpa_only():
        return sdpa()

    for name, fn in (
        ("two_op", two_op),
        ("fused", fused),
        ("sdpa_only", sdpa_only),
        ("two_op", two_op),
        ("fused", fused),
    ):
        out = fn()
        ref = ttnn.to_torch(out)
        out.deallocate(True)
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = fn()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        tag = f"AB_{name}_batch{batch}"
        signpost(tag)
        for _ in range(REPLAYS):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)
        signpost(tag + "_END")
        ttnn.ReadDeviceProfiler(mesh)
        same = bool(torch.equal(ref, ttnn.to_torch(out)))
        ttnn.release_trace(mesh, tid)
        rows.append(dict(variant=name, batch=batch, replays=REPLAYS, replay_equals_eager=same, signpost=tag))
        print(json.dumps(rows[-1]), flush=True)
    for t in (kcache, vcache, q, k, v_same):
        t.deallocate(True)
json.dump(rows, open("/tmp/qwen3_opt/exp4.json", "w"), indent=1)
ttnn.close_mesh_device(mesh)
print("DONE")
