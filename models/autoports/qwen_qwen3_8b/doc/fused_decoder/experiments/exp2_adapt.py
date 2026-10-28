import os
import time

import torch

import ttnn
from models.autoports.qwen_qwen3_8b.tt import functional_decoder as fd

B = int(os.environ.get("EXP_BATCH", "1"))
HD, NH, NKV, HID, INTER = 128, 32, 8, 4096, 12288
BLOCK_SIZE = 32

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=128 * 1024 * 1024)
grid = mesh.compute_with_storage_grid_size()
print("GRID", grid.x, grid.y, "BATCH", B, flush=True)
compute = ttnn.init_device_compute_kernel_config(
    mesh.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


def tt(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, memory_config=mem)


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def timed_trace(fn, iters=10, reps=5):
    fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(iters):
        fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh)
    dt = (time.perf_counter() - t0) / (reps * iters) * 1e6
    ttnn.release_trace(mesh, tid)
    return dt


torch.manual_seed(0)
physical, gx, gy = fd.decode_physical_batch(B, grid.x, grid.y)
heads_mem = fd._decode_height_sharded_memory_config(mesh, physical, HD)
q_cores = heads_mem.shard_spec.grid
blocks_per_user = 8
num_blocks = physical * blocks_per_user
table_t = torch.randperm(num_blocks).reshape(physical, blocks_per_user).to(torch.int32)
page_table = tt(table_t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
kv_extent = blocks_per_user * BLOCK_SIZE
pos_t = torch.randint(64, kv_extent - 1, (physical,), dtype=torch.int32)
cur_pos = tt(pos_t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
x = tt(torch.randn(1, 1, physical, HID) * 0.5)
w_qkv = tt(torch.randn(HID, HD * (NH + 2 * NKV)) * 0.02)
w_o = tt(torch.randn(HD * NH, HID) * 0.02)
q_norm_w = tt(torch.rand(1, 1, 1, HD) + 0.5)
k_norm_w = tt(torch.rand(1, 1, 1, HD) + 0.5)
cos_t = torch.randn(40960, HD)
sin_t = torch.randn(40960, HD)
cos_rm = tt(cos_t, layout=ttnn.ROW_MAJOR_LAYOUT)
sin_rm = tt(sin_t, layout=ttnn.ROW_MAJOR_LAYOUT)
scale = fd._float32(HD**-0.5)
sdpa_cfg = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(8, min(8, grid.y)),
    q_chunk_size=32,
    k_chunk_size=fd.decode_sdpa_k_chunk_size(kv_extent),
    exp_approx_mode=False,
)

# two-grid memory config for overlap_qk_coregrid=False: 2*physical cores
two_cores = ttnn.num_cores_to_corerangeset(2 * physical, grid, row_wise=True)
two_mem = ttnn.create_sharded_memory_config(
    shape=(32, HD),
    core_grid=two_cores,
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
base_kv = torch.randn(num_blocks, NKV, BLOCK_SIZE, HD) * 0.1
caches = {name: (tt(base_kv.clone()), tt(base_kv.clone())) for name in ("base", "a", "b")}


def gather(table_rm, mem):
    idx = ttnn.reshape(cur_pos, (1, physical))
    idx = ttnn.typecast(idx, ttnn.uint32)
    emb = ttnn.embedding(idx, table_rm, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    emb = ttnn.reshape(emb, (1, physical, 1, HD))
    emb = ttnn.to_layout(emb, ttnn.TILE_LAYOUT)
    return ttnn.interleaved_to_sharded(emb, mem)


def norm_sharded(heads, w, mem):
    l1 = ttnn.to_memory_config(heads, ttnn.L1_MEMORY_CONFIG, dtype=heads.dtype)
    return ttnn.rms_norm(l1, epsilon=1e-6, weight=w, compute_kernel_config=compute, memory_config=mem)


def attention(variant):
    key_cache, value_cache = caches[variant]
    xqkv = ttnn.linear(x, w_qkv, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if variant == "a":
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv, num_heads=NH, num_kv_heads=NKV, memory_config=two_mem, overlap_qk_coregrid=False
        )
        k_mem = k.memory_config()
    else:
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv, num_heads=NH, num_kv_heads=NKV, memory_config=heads_mem
        )
        k_mem = heads_mem
    qn = norm_sharded(q, q_norm_w, heads_mem)
    kn = norm_sharded(k, k_norm_w, k_mem)
    cos = gather(cos_rm, heads_mem)
    sin = gather(sin_rm, heads_mem)
    q_rot = ttnn.experimental.rotary_embedding_hf(qn, cos, sin, is_decode_mode=True, compute_kernel_config=compute)
    if variant == "a":
        cos_k = ttnn.to_memory_config(cos, k_mem)
        sin_k = ttnn.to_memory_config(sin, k_mem)
        k_rot = ttnn.experimental.rotary_embedding_hf(
            kn, cos_k, sin_k, is_decode_mode=True, compute_kernel_config=compute
        )
        ttnn.experimental.paged_fused_update_cache(
            key_cache, k_rot, value_cache, v, update_idxs_tensor=cur_pos, page_table=page_table
        )
    elif variant == "b":
        k_rot = ttnn.experimental.rotary_embedding_hf(kn, cos, sin, is_decode_mode=True, compute_kernel_config=compute)
        v_cores = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            ttnn.CoreCoord(0, gy),
            physical,
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, gy), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]),
            row_wise=True,
        )
        v_mem = ttnn.create_sharded_memory_config(
            shape=(32, HD),
            core_grid=v_cores,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        v2 = ttnn.to_memory_config(v, v_mem)
        ttnn.experimental.paged_fused_update_cache(
            key_cache, k_rot, value_cache, v2, update_idxs_tensor=cur_pos, page_table=page_table
        )
    else:
        k_rot = ttnn.experimental.rotary_embedding_hf(kn, cos, sin, is_decode_mode=True, compute_kernel_config=compute)
        ttnn.experimental.paged_update_cache(key_cache, k_rot, update_idxs_tensor=cur_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(value_cache, v, update_idxs_tensor=cur_pos, page_table=page_table)
    attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_rot,
        key_cache,
        value_cache,
        page_table_tensor=page_table,
        cur_pos_tensor=cur_pos,
        scale=scale,
        program_config=sdpa_cfg,
        compute_kernel_config=compute,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    attn_sh = ttnn.to_memory_config(attn, heads_mem, dtype=attn.dtype)
    concat = ttnn.experimental.nlp_concat_heads_decode(attn_sh, num_heads=NH)
    return ttnn.linear(concat, w_o, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)


print("\n===== E7 adapted fused update cache variants =====", flush=True)
outs = {}
for variant in ("base", "a", "b"):
    try:
        o = attention(variant)
        outs[variant] = ttnn.to_torch(o)
        print(variant, "out abs mean", float(outs[variant].float().abs().mean()), flush=True)
    except Exception as e:
        print(variant, "FAIL:", str(e).splitlines()[0][:400], flush=True)
for variant in ("a", "b"):
    if variant in outs:
        kb, vb = caches["base"]
        kv_, vv_ = caches[variant]
        print(
            variant,
            "out equal base",
            torch.equal(outs[variant], outs["base"]),
            "pcc",
            pcc(outs[variant], outs["base"]),
            "kcache equal",
            torch.equal(ttnn.to_torch(kb), ttnn.to_torch(kv_)),
            "vcache equal",
            torch.equal(ttnn.to_torch(vb), ttnn.to_torch(vv_)),
        )
for rep in range(2):
    for variant in ("base", "a", "b"):
        if variant in outs:
            print(f"rep{rep} {variant} attention us {timed_trace(lambda: attention(variant)):.1f}", flush=True)

print("\n===== E9b gate/up packing A/B, same weights =====", flush=True)
for rows in (32, 1024):
    xm = tt(torch.randn(1, 1, rows, HID) * 0.5)
    wg_t = torch.randn(HID, INTER) * 0.02
    wu_t = torch.randn(HID, INTER) * 0.02
    wd_t = torch.randn(INTER, HID) * 0.02
    wg, wu, wd = tt(wg_t), tt(wu_t), tt(wd_t)
    wgu = tt(torch.cat([wg_t, wu_t], dim=-1))

    def mlp_base():
        g = ttnn.linear(xm, wg, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        u = ttnn.linear(xm, wu, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        a = ttnn.mul(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.linear(a, wd, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def mlp_packed():
        gu = ttnn.linear(xm, wgu, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        g = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, rows, INTER], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        u = ttnn.slice(gu, [0, 0, 0, INTER], [1, 1, rows, 2 * INTER], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        a = ttnn.mul(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.linear(a, wd, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def gu_only_base():
        ttnn.linear(xm, wg, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.linear(xm, wu, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def gu_only_packed():
        ttnn.linear(xm, wgu, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    b = ttnn.to_torch(mlp_base())
    p = ttnn.to_torch(mlp_packed())
    print(f"rows={rows} packed equal base {torch.equal(b, p)} pcc {pcc(b, p):.7f}", flush=True)
    iters = 4 if rows == 1024 else 10
    for rep in range(3):
        print(
            f"rows={rows} rep{rep} base {timed_trace(mlp_base, iters=iters, reps=3):.1f} packed {timed_trace(mlp_packed, iters=iters, reps=3):.1f} "
            f"gu_base {timed_trace(gu_only_base, iters=iters, reps=3):.1f} gu_packed {timed_trace(gu_only_packed, iters=iters, reps=3):.1f}",
            flush=True,
        )

ttnn.close_mesh_device(mesh)
print("DONE")
