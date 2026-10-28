import os
import time

import torch

import ttnn
from models.autoports.qwen_qwen3_8b.tt import functional_decoder as fd

B = int(os.environ.get("EXP_BATCH", "1"))
HD, NH, NKV, HID = 128, 32, 8, 4096
BLOCK_SIZE = 32
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 * 1024 * 1024)
grid = mesh.compute_with_storage_grid_size()
compute = ttnn.init_device_compute_kernel_config(
    mesh.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


def tt(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, memory_config=mem)


def stats(name, t):
    x = ttnn.to_torch(t).float()
    print(
        f"  {name}: shape {tuple(x.shape)} absmean {x.abs().mean():.4g} absmax {x.abs().max():.4g} finite {bool(torch.isfinite(x).all())}",
        flush=True,
    )


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
blocks_per_user = 8
num_blocks = physical * blocks_per_user
table_t = torch.randperm(num_blocks).reshape(physical, blocks_per_user).to(torch.int32)
page_table = tt(table_t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
kv_extent = blocks_per_user * BLOCK_SIZE
pos_t = torch.randint(64, kv_extent - 1, (physical,), dtype=torch.int32)
print("positions", pos_t.tolist()[:8], "extent", kv_extent, flush=True)
cur_pos = tt(pos_t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
x = tt(torch.randn(1, 1, physical, HID) * 0.5)
w_qkv = tt(torch.randn(HID, HD * (NH + 2 * NKV)) * 0.02)
w_o = tt(torch.randn(HD * NH, HID) * 0.02)
q_norm_w = tt(torch.rand(1, 1, 1, HD) + 0.5)
k_norm_w = tt(torch.rand(1, 1, 1, HD) + 0.5)
geometry = fd.DecoderGeometry(HID, 12288, NH, NKV, HD, 1e-6, 1e6, 40960)
cos_real, sin_real = fd.hf_rope_tables(geometry)
tables = {
    "real_rope": (tt(cos_real, layout=ttnn.ROW_MAJOR_LAYOUT), tt(sin_real, layout=ttnn.ROW_MAJOR_LAYOUT)),
    "randn_rope": (
        tt(torch.randn(40960, HD), layout=ttnn.ROW_MAJOR_LAYOUT),
        tt(torch.randn(40960, HD), layout=ttnn.ROW_MAJOR_LAYOUT),
    ),
}
scale = fd._float32(HD**-0.5)
sdpa_cfg = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(8, min(8, grid.y)),
    q_chunk_size=32,
    k_chunk_size=fd.decode_sdpa_k_chunk_size(kv_extent),
    exp_approx_mode=False,
)
two_cores = ttnn.num_cores_to_corerangeset(2 * physical, grid, row_wise=True)
two_mem = ttnn.create_sharded_memory_config(
    shape=(32, HD),
    core_grid=two_cores,
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
base_kv = torch.randn(num_blocks, NKV, BLOCK_SIZE, HD) * 0.1


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


def attention(variant, cos_rm, sin_rm, caches, verbose=False):
    key_cache, value_cache = caches
    xqkv = ttnn.linear(x, w_qkv, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if variant == "a":
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv, num_heads=NH, num_kv_heads=NKV, memory_config=two_mem, overlap_qk_coregrid=False
        )
        k_mem = k.memory_config()
        if verbose:
            print("  k grid", k_mem.shard_spec.grid, "v grid", v.memory_config().shard_spec.grid, flush=True)
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
            kn, cos_k, sin_k, is_decode_mode=True, compute_kernel_config=compute, memory_config=k_mem
        )
        if verbose:
            print("  k_rot grid", k_rot.memory_config().shard_spec.grid, flush=True)
        ttnn.experimental.paged_fused_update_cache(
            key_cache, k_rot, value_cache, v, update_idxs_tensor=cur_pos, page_table=page_table
        )
    else:
        k_rot = ttnn.experimental.rotary_embedding_hf(kn, cos, sin, is_decode_mode=True, compute_kernel_config=compute)
        ttnn.experimental.paged_update_cache(key_cache, k_rot, update_idxs_tensor=cur_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(value_cache, v, update_idxs_tensor=cur_pos, page_table=page_table)
    if verbose:
        stats("q_rot", q_rot)
        stats("k_rot", k_rot)
        stats("v", v)
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
    if verbose:
        stats("attn", attn)
    attn_sh = ttnn.to_memory_config(attn, heads_mem, dtype=attn.dtype)
    concat = ttnn.experimental.nlp_concat_heads_decode(attn_sh, num_heads=NH)
    if verbose:
        stats("concat", concat)
    out = ttnn.linear(concat, w_o, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if verbose:
        stats("out", out)
    return out


for tname, (cos_rm, sin_rm) in tables.items():
    print(f"\n===== tables={tname} =====", flush=True)
    outs = {}
    cache_sets = {}
    for variant in ("base", "a"):
        print(f"-- variant {variant}", flush=True)
        caches = (tt(base_kv.clone()), tt(base_kv.clone()))
        cache_sets[variant] = caches
        try:
            o = attention(variant, cos_rm, sin_rm, caches, verbose=True)
            outs[variant] = ttnn.to_torch(o)
        except Exception as e:
            print(variant, "FAIL:", str(e).splitlines()[0][:400], flush=True)
    if "a" in outs:
        kb, vb = cache_sets["base"]
        ka, va = cache_sets["a"]
        print(
            "a out equal base",
            torch.equal(outs["a"], outs["base"]),
            "kcache equal",
            torch.equal(ttnn.to_torch(kb), ttnn.to_torch(ka)),
            "vcache equal",
            torch.equal(ttnn.to_torch(vb), ttnn.to_torch(va)),
            flush=True,
        )
        for rep in range(2):
            for variant in ("base", "a"):
                print(
                    f"rep{rep} {variant} attention us {timed_trace(lambda: attention(variant, cos_rm, sin_rm, cache_sets[variant])):.1f}",
                    flush=True,
                )

ttnn.close_mesh_device(mesh)
print("DONE")
