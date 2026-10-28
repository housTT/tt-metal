import time
import traceback

import torch

import ttnn
from models.autoports.qwen_qwen3_8b.tt import functional_decoder as fd

B = int(__import__("os").environ.get("EXP_BATCH", "1"))
HD, NH, NKV, HID, INTER = 128, 32, 8, 4096, 12288
BLOCKS = 40
BLOCK_SIZE = 32

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 * 1024 * 1024)
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


def timed_trace(fn, iters=20):
    fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(iters):
        fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(3):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh)
    dt = (time.perf_counter() - t0) / (3 * iters) * 1e6
    ttnn.release_trace(mesh, tid)
    return dt


def section(name):
    print(f"\n===== {name} =====", flush=True)


torch.manual_seed(0)
physical, gx, gy = fd.decode_physical_batch(B, grid.x, grid.y)
heads_mem = fd._decode_height_sharded_memory_config(mesh, physical, HD)
x_t = torch.randn(1, 1, physical, HID) * 0.5
w_qkv_t = torch.randn(HID, HD * (NH + 2 * NKV)) * 0.02
w_o_t = torch.randn(HD * NH, HID) * 0.02
qn_t = torch.rand(1, 1, 1, HD) + 0.5
kn_t = torch.rand(1, 1, 1, HD) + 0.5
x = tt(x_t)
w_qkv = tt(w_qkv_t)
w_o = tt(w_o_t)
q_norm_w = tt(qn_t)
k_norm_w = tt(kn_t)
xqkv = ttnn.linear(x, w_qkv, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
    xqkv, num_heads=NH, num_kv_heads=NKV, memory_config=heads_mem
)
print("q_heads", q_heads.shape, q_heads.memory_config())


def baseline_norm(heads, w):
    l1 = ttnn.to_memory_config(heads, ttnn.L1_MEMORY_CONFIG, dtype=heads.dtype)
    n = ttnn.rms_norm(l1, epsilon=1e-6, weight=w, compute_kernel_config=compute)
    return ttnn.to_memory_config(n, heads_mem, dtype=n.dtype)


q_base = baseline_norm(q_heads, q_norm_w)
k_base = baseline_norm(k_heads, k_norm_w)

section("E1 rms_norm directly on height-sharded heads (sharded program config)")
try:
    pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy), subblock_w=4, block_h=1, block_w=4, inplace=False
    )
    out = ttnn.rms_norm(
        q_heads,
        epsilon=1e-6,
        weight=q_norm_w,
        program_config=pc,
        memory_config=heads_mem,
        compute_kernel_config=compute,
    )
    print("E1 OK pcc", pcc(ttnn.to_torch(out), ttnn.to_torch(q_base)))
except Exception as e:
    print("E1 FAIL:", str(e).splitlines()[0][:300])

section("E1b rms_norm on height-sharded heads with default program config")
try:
    out = ttnn.rms_norm(q_heads, epsilon=1e-6, weight=q_norm_w, compute_kernel_config=compute)
    print("E1b OK mem", out.memory_config(), "pcc", pcc(ttnn.to_torch(out), ttnn.to_torch(q_base)))
except Exception as e:
    print("E1b FAIL:", str(e).splitlines()[0][:300])

section("E2 nlp_create_qkv_heads_decode with interleaved L1 output")
try:
    outs = ttnn.experimental.nlp_create_qkv_heads_decode(
        xqkv, num_heads=NH, num_kv_heads=NKV, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    print("E2 OK", outs[0].memory_config())
except Exception as e:
    print("E2 FAIL:", str(e).splitlines()[0][:300])

section("E3 rms_norm interleaved input -> height-sharded output (skip the i2s)")
try:
    l1 = ttnn.to_memory_config(q_heads, ttnn.L1_MEMORY_CONFIG, dtype=q_heads.dtype)
    out = ttnn.rms_norm(l1, epsilon=1e-6, weight=q_norm_w, compute_kernel_config=compute, memory_config=heads_mem)
    print("E3 OK mem", out.memory_config().memory_layout, "pcc", pcc(ttnn.to_torch(out), ttnn.to_torch(q_base)))
except Exception as e:
    print("E3 FAIL:", str(e).splitlines()[0][:300])

section("E3b s2i into DRAM interleaved then rms_norm (skip L1)")
try:
    dr = ttnn.sharded_to_interleaved(q_heads, ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.rms_norm(dr, epsilon=1e-6, weight=q_norm_w, compute_kernel_config=compute)
    print("E3b OK pcc", pcc(ttnn.to_torch(out), ttnn.to_torch(q_base)))
except Exception as e:
    print("E3b FAIL:", str(e).splitlines()[0][:300])

# ---- RoPE gather
section("E6 RoPE gather without transpose")
cos_t = torch.randn(40960, HD)
cos_table_rm = tt(cos_t, layout=ttnn.ROW_MAJOR_LAYOUT)
pos_t = torch.randint(0, 2000, (physical,), dtype=torch.int32)
cur_pos = tt(pos_t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)


def rope_gather_base(cur_pos):
    idx = ttnn.reshape(cur_pos, (1, physical))
    idx = ttnn.typecast(idx, ttnn.uint32)
    emb = ttnn.embedding(idx, cos_table_rm, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    emb = ttnn.reshape(emb, (1, 1, physical, HD))
    emb = ttnn.transpose(emb, 1, 2)
    return ttnn.interleaved_to_sharded(emb, heads_mem)


def rope_gather_v1(cur_pos):
    idx = ttnn.reshape(cur_pos, (1, physical))
    idx = ttnn.typecast(idx, ttnn.uint32)
    emb = ttnn.embedding(idx, cos_table_rm, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    emb = ttnn.reshape(emb, (1, physical, 1, HD))
    emb = ttnn.to_layout(emb, ttnn.TILE_LAYOUT)
    return ttnn.interleaved_to_sharded(emb, heads_mem)


def rope_gather_v2(cur_pos):
    idx = ttnn.reshape(cur_pos, (1, physical))
    idx = ttnn.typecast(idx, ttnn.uint32)
    rm_sharded = ttnn.create_sharded_memory_config(
        shape=(1, HD),
        core_grid=heads_mem.shard_spec.grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    emb = ttnn.embedding(idx, cos_table_rm, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=rm_sharded)
    emb = ttnn.reshape(emb, (1, physical, 1, HD))
    return ttnn.to_layout(emb, ttnn.TILE_LAYOUT, memory_config=heads_mem)


cos_base = rope_gather_base(cur_pos)
ref = ttnn.to_torch(cos_base)
for name, fn in (("v1_rm_reshape_tilize", rope_gather_v1), ("v2_sharded_embedding", rope_gather_v2)):
    try:
        out = fn(cur_pos)
        o = ttnn.to_torch(out)
        print(
            name,
            "shape",
            out.shape,
            out.memory_config().memory_layout,
            "equal",
            torch.equal(o, ref),
            "pcc",
            pcc(o, ref),
        )
        q_rot = ttnn.experimental.rotary_embedding_hf(
            q_base, out, out, is_decode_mode=True, compute_kernel_config=compute
        )
        q_rot_ref = ttnn.experimental.rotary_embedding_hf(
            q_base, cos_base, cos_base, is_decode_mode=True, compute_kernel_config=compute
        )
        print(name, "rope equal", torch.equal(ttnn.to_torch(q_rot), ttnn.to_torch(q_rot_ref)))
        print(name, "us", timed_trace(lambda: fn(cur_pos)), "base us", timed_trace(lambda: rope_gather_base(cur_pos)))
    except Exception as e:
        print(name, "FAIL:", str(e).splitlines()[0][:300])

# ---- SDPA sharded output + concat + linear
section("E4/E5 SDPA output sharded, linear on width-sharded concat")
kc = tt(torch.randn(BLOCKS, NKV, BLOCK_SIZE, HD) * 0.1)
vc = tt(torch.randn(BLOCKS, NKV, BLOCK_SIZE, HD) * 0.1)
table_t = torch.arange(physical * (BLOCKS // max(physical, 1)), dtype=torch.int32)[
    : physical * (BLOCKS // physical)
].reshape(physical, -1)
page_table = tt(table_t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
kv_extent = page_table.shape[1] * BLOCK_SIZE
pos_small = tt(
    torch.randint(0, kv_extent - 1, (physical,), dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
)
sdpa_cfg = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(8, min(8, grid.y)),
    q_chunk_size=32,
    k_chunk_size=fd.decode_sdpa_k_chunk_size(kv_extent),
    exp_approx_mode=False,
)
q_rot = ttnn.experimental.rotary_embedding_hf(
    q_base, cos_base, cos_base, is_decode_mode=True, compute_kernel_config=compute
)
attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
    q_rot,
    kc,
    vc,
    page_table_tensor=page_table,
    cur_pos_tensor=pos_small,
    scale=fd._float32(HD**-0.5),
    program_config=sdpa_cfg,
    compute_kernel_config=compute,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
print("attn", attn.shape, attn.memory_config().memory_layout)
try:
    attn_sh = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_rot,
        kc,
        vc,
        page_table_tensor=page_table,
        cur_pos_tensor=pos_small,
        scale=fd._float32(HD**-0.5),
        program_config=sdpa_cfg,
        compute_kernel_config=compute,
        memory_config=heads_mem,
    )
    print("E4 OK", attn_sh.memory_config())
except Exception as e:
    print("E4 FAIL:", str(e).splitlines()[0][:300])

attn_sh = ttnn.to_memory_config(attn, heads_mem, dtype=attn.dtype)
concat = ttnn.experimental.nlp_concat_heads_decode(attn_sh, num_heads=NH)
print("concat", concat.shape, concat.memory_config())
concat_dram = ttnn.sharded_to_interleaved(concat, ttnn.DRAM_MEMORY_CONFIG)
o_base = ttnn.linear(concat_dram, w_o, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
try:
    o_direct = ttnn.linear(concat, w_o, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    print(
        "E5 OK pcc",
        pcc(ttnn.to_torch(o_direct), ttnn.to_torch(o_base)),
        "equal",
        torch.equal(ttnn.to_torch(o_direct), ttnn.to_torch(o_base)),
    )
    print(
        "E5 direct us",
        timed_trace(
            lambda: ttnn.linear(concat, w_o, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ),
    )
    print(
        "E5 base us",
        timed_trace(
            lambda: ttnn.linear(
                ttnn.sharded_to_interleaved(concat, ttnn.DRAM_MEMORY_CONFIG),
                w_o,
                compute_kernel_config=compute,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        ),
    )
except Exception as e:
    print("E5 FAIL:", str(e).splitlines()[0][:300])

section("E5b nlp_concat_heads_decode output straight to DRAM interleaved")
try:
    c2 = ttnn.experimental.nlp_concat_heads_decode(attn_sh, num_heads=NH, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    print("E5b OK", c2.memory_config())
except Exception as e:
    print("E5b FAIL:", str(e).splitlines()[0][:300])

# ---- fused update cache
section("E7 paged_fused_update_cache")
k_rot = ttnn.experimental.rotary_embedding_hf(
    k_base, cos_base, cos_base, is_decode_mode=True, compute_kernel_config=compute
)
kc1 = tt(torch.zeros(BLOCKS, NKV, BLOCK_SIZE, HD))
vc1 = tt(torch.zeros(BLOCKS, NKV, BLOCK_SIZE, HD))
kc2 = tt(torch.zeros(BLOCKS, NKV, BLOCK_SIZE, HD))
vc2 = tt(torch.zeros(BLOCKS, NKV, BLOCK_SIZE, HD))
ttnn.experimental.paged_update_cache(kc1, k_rot, update_idxs_tensor=pos_small, page_table=page_table)
ttnn.experimental.paged_update_cache(vc1, v_heads, update_idxs_tensor=pos_small, page_table=page_table)
try:
    ttnn.experimental.paged_fused_update_cache(
        kc2, k_rot, vc2, v_heads, update_idxs_tensor=pos_small, page_table=page_table
    )
    print(
        "E7 OK k equal",
        torch.equal(ttnn.to_torch(kc1), ttnn.to_torch(kc2)),
        "v equal",
        torch.equal(ttnn.to_torch(vc1), ttnn.to_torch(vc2)),
    )
    print(
        "E7 fused us",
        timed_trace(
            lambda: ttnn.experimental.paged_fused_update_cache(
                kc2, k_rot, vc2, v_heads, update_idxs_tensor=pos_small, page_table=page_table
            )
        ),
    )

    def two():
        ttnn.experimental.paged_update_cache(kc1, k_rot, update_idxs_tensor=pos_small, page_table=page_table)
        ttnn.experimental.paged_update_cache(vc1, v_heads, update_idxs_tensor=pos_small, page_table=page_table)

    print("E7 two us", timed_trace(two))
except Exception as e:
    print("E7 FAIL:", str(e).splitlines()[0][:300])
    traceback.print_exc()

# ---- matmul bias residual for logical batch 1
section("E8 matmul bias as residual (logical batch 1)")
x1 = tt(torch.randn(1, 1, 1, HID))
c1 = tt(torch.randn(1, 1, 32, HD * NH) * 0.1)
base = ttnn.linear(c1, w_o, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
base = ttnn.slice(base, [0, 0, 0, 0], [1, 1, 1, HID], memory_config=ttnn.DRAM_MEMORY_CONFIG)
h_base = ttnn.add(x1, base, memory_config=ttnn.DRAM_MEMORY_CONFIG)
try:
    fused = ttnn.linear(c1, w_o, bias=x1, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    fused_s = ttnn.slice(fused, [0, 0, 0, 0], [1, 1, 1, HID], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    print(
        "E8 shape",
        fused.shape,
        "pcc",
        pcc(ttnn.to_torch(fused_s), ttnn.to_torch(h_base)),
        "max abs diff",
        float((ttnn.to_torch(fused_s).float() - ttnn.to_torch(h_base).float()).abs().max()),
    )

    def fb():
        f = ttnn.linear(c1, w_o, bias=x1, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.slice(f, [0, 0, 0, 0], [1, 1, 1, HID], memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def bb():
        f = ttnn.linear(c1, w_o, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        f = ttnn.slice(f, [0, 0, 0, 0], [1, 1, 1, HID], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.add(x1, f, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    print("E8 fused us", timed_trace(fb), "base us", timed_trace(bb))
except Exception as e:
    print("E8 FAIL:", str(e).splitlines()[0][:300])

# ---- MLP variants
section("E9 MLP variants")
for rows in (32, 1024):
    xm = tt(torch.randn(1, 1, rows, HID) * 0.5)
    wg = tt(torch.randn(HID, INTER) * 0.02)
    wu = tt(torch.randn(HID, INTER) * 0.02)
    wd = tt(torch.randn(INTER, HID) * 0.02)
    wgu = tt(torch.cat([torch.randn(HID, INTER) * 0.02, torch.randn(HID, INTER) * 0.02], dim=-1))

    def mlp_base():
        g = ttnn.linear(xm, wg, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        u = ttnn.linear(xm, wu, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        a = ttnn.mul(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.linear(a, wd, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def mlp_silu_in_mm():
        g = ttnn.linear(xm, wg, activation="silu", compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        u = ttnn.linear(xm, wu, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        a = ttnn.mul(g, u, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.linear(a, wd, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def mlp_packed():
        gu = ttnn.linear(xm, wgu, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        g = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, rows, INTER], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        u = ttnn.slice(gu, [0, 0, 0, INTER], [1, 1, rows, 2 * INTER], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        a = ttnn.mul(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.linear(a, wd, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def mlp_packed_swiglu():
        gu = ttnn.linear(xm, wgu, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        a = ttnn.swiglu(gu, -1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.linear(a, wd, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    b = ttnn.to_torch(mlp_base())
    for name, fn in (
        ("silu_in_matmul", mlp_silu_in_mm),
        ("packed_gate_up_slices", mlp_packed),
        ("packed_swiglu", mlp_packed_swiglu),
    ):
        try:
            o = ttnn.to_torch(fn())
            note = "" if name.startswith("silu") else "(different weights: timing only)"
            print(f"rows={rows} {name} pcc_vs_base={pcc(o, b):.6f} {note} us={timed_trace(fn, iters=5):.1f}")
        except Exception as e:
            print(f"rows={rows} {name} FAIL:", str(e).splitlines()[0][:300])
    print(f"rows={rows} base us={timed_trace(mlp_base, iters=5):.1f}")

ttnn.close_mesh_device(mesh)
print("DONE")
