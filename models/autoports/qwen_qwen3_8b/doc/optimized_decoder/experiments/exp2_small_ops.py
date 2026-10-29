"""Decode small-op micro-benchmarks: sharded RMSNorm, residual add, SiLU-mul, head creation, SDPA configs."""
import json
import sys
import time

import torch

import ttnn

sys.path.insert(0, "/home/hous/dev/tt-metal-wt-qwen3")
from models.autoports.qwen_qwen3_8b.tests import weights as weight_source
from models.common.utility_functions import comp_pcc

REP = 10
REPLAYS = 5
OUT = "/tmp/qwen3_opt/exp2.json"
rows = []
state = weight_source.load_layer_state(0)
emb = weight_source.load_embedding()
tokens = torch.tensor(
    json.loads(
        open(
            "/home/hous/dev/tt-metal-wt-qwen3/models/autoports/qwen_qwen3_8b/doc/functional_decoder/layer_inputs/tale_token_ids.json"
        ).read()
    )["tokens"][100:132]
)
x = emb[tokens].float()
ln_w = state["input_layernorm.weight"].float()


def rms(v, w, eps=1e-6):
    return w * (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps))


mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 * 1024 * 1024)
grid = mesh.compute_with_storage_grid_size()
ckc_hifi4 = ttnn.init_device_compute_kernel_config(
    mesh.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)
ckc_hifi2 = ttnn.init_device_compute_kernel_config(
    mesh.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


def pcc(a, b):
    return float(comp_pcc(a.float(), b.float(), 0.0)[1])


def core_set(n):
    if n <= grid.x:
        return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n - 1, 0))]), (n, 1)
    for w in (8, 11, 10):
        if n % w == 0 and n // w <= grid.y and w <= grid.x:
            return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(w - 1, n // w - 1))]), (
                w,
                n // w,
            )
    raise ValueError(n)


def wcfg(width, n):
    cs, _ = core_set(n)
    return ttnn.create_sharded_memory_config(
        shape=(32, width // n),
        core_grid=cs,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def time_trace(fn):
    fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(REP):
        fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    best = 1e9
    for _ in range(REPLAYS):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        best = min(best, time.perf_counter() - t0)
    ttnn.release_trace(mesh, tid)
    return best * 1e6 / REP


def record(row):
    print(json.dumps(row), flush=True)
    rows.append(row)


x4 = x.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
ref_norm = rms(x, ln_w)
w_t = ttnn.from_torch(
    ln_w.reshape(1, 1, 1, -1).to(torch.bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# 1. RMSNorm variants
for mem_name, mem in (("dram_interleaved", ttnn.DRAM_MEMORY_CONFIG), ("l1_interleaved", ttnn.L1_MEMORY_CONFIG)):
    for ck_name, ck in (("hifi4_fp32", ckc_hifi4), ("hifi2", ckc_hifi2)):
        xt = ttnn.from_torch(x4, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=mem)
        row = dict(op="rms_norm", layout=mem_name, ckc=ck_name)
        try:

            def fn():
                y = ttnn.rms_norm(xt, epsilon=1e-6, weight=w_t, compute_kernel_config=ck, memory_config=mem)
                y.deallocate(True)

            y = ttnn.rms_norm(xt, epsilon=1e-6, weight=w_t, compute_kernel_config=ck, memory_config=mem)
            row["pcc"] = pcc(ref_norm, ttnn.to_torch(y)[0, 0])
            y.deallocate(True)
            row["us"] = time_trace(fn)
        except Exception as e:
            row["error"] = str(e).splitlines()[0][:300]
        record(row)
        xt.deallocate(True)

for n in (8, 16, 32, 64):
    cs, (gx, gy) = core_set(n)
    block_w = 4096 // 32 // n
    for sub_w in sorted({d for d in (1, 2, 4, 8) if block_w % d == 0}):
        for ck_name, ck in (("hifi4_fp32", ckc_hifi4), ("hifi2", ckc_hifi2)):
            cfg = wcfg(4096, n)
            xt = ttnn.from_torch(x4, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=cfg)
            pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(gx, gy), subblock_w=sub_w, block_h=1, block_w=block_w, inplace=False
            )
            row = dict(
                op="rms_norm",
                layout=f"width_sharded_{n}",
                grid=f"{gx}x{gy}",
                block_w=block_w,
                subblock_w=sub_w,
                ckc=ck_name,
            )
            try:

                def fn():
                    y = ttnn.rms_norm(
                        xt, epsilon=1e-6, weight=w_t, compute_kernel_config=ck, program_config=pc, memory_config=cfg
                    )
                    y.deallocate(True)

                y = ttnn.rms_norm(
                    xt, epsilon=1e-6, weight=w_t, compute_kernel_config=ck, program_config=pc, memory_config=cfg
                )
                row["pcc"] = pcc(ref_norm, ttnn.to_torch(y)[0, 0])
                y.deallocate(True)
                row["us"] = time_trace(fn)
            except Exception as e:
                row["error"] = str(e).splitlines()[0][:300]
            record(row)
            xt.deallocate(True)

# 2. residual add and 3. silu-mul on sharded tensors; 7. interleaved_to_sharded
for n in (8, 16, 32, 64):
    cfg = wcfg(4096, n)
    a = ttnn.from_torch(x4, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=cfg)
    b = ttnn.from_torch(x4, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=cfg)
    row = dict(op="add", layout=f"width_sharded_{n}")
    try:

        def fn():
            y = ttnn.add(a, b, memory_config=cfg)
            y.deallocate(True)

        row["us"] = time_trace(fn)
    except Exception as e:
        row["error"] = str(e).splitlines()[0][:300]
    record(row)
    d = ttnn.from_torch(
        x4, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    row = dict(op="interleaved_to_sharded", layout=f"dram->width_sharded_{n}")
    try:

        def fn():
            y = ttnn.to_memory_config(d, cfg)
            y.deallocate(True)

        row["us"] = time_trace(fn)
    except Exception as e:
        row["error"] = str(e).splitlines()[0][:300]
    record(row)
    row = dict(op="sharded_to_interleaved", layout=f"width_sharded_{n}->dram")
    try:

        def fn():
            y = ttnn.to_memory_config(a, ttnn.DRAM_MEMORY_CONFIG)
            y.deallocate(True)

        row["us"] = time_trace(fn)
    except Exception as e:
        row["error"] = str(e).splitlines()[0][:300]
    record(row)
    a.deallocate(True)
    b.deallocate(True)
    d.deallocate(True)
    cfg2 = wcfg(12288, n)
    g = ttnn.from_torch(
        torch.randn(1, 1, 32, 12288).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=cfg2,
    )
    u = ttnn.from_torch(
        torch.randn(1, 1, 32, 12288).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=cfg2,
    )
    row = dict(op="silu_mul", layout=f"width_sharded_{n}")
    try:

        def fn():
            y = ttnn.mul(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=cfg2)
            y.deallocate(True)

        row["us"] = time_trace(fn)
    except Exception as e:
        row["error"] = str(e).splitlines()[0][:300]
    record(row)
    g.deallocate(True)
    u.deallocate(True)

# 4. create heads from width-sharded xqkv
for batch in (1, 32):
    hx = batch if batch <= 8 else 8
    hy = 1 if batch <= 8 else batch // 8
    heads_cfg = ttnn.create_sharded_memory_config(
        shape=(32, 128),
        core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(hx - 1, hy - 1))]),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    for n in (8, 16, 32, 64):
        cfg = wcfg(6144, n)
        xq = ttnn.from_torch(
            torch.randn(1, 1, 32, 6144).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=cfg,
        )
        if batch != 32:
            xq_logical = ttnn.from_torch(
                torch.randn(1, 1, batch, 6144).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=cfg,
            )
            xq.deallocate(True)
            xq = xq_logical
        row = dict(op="create_qkv_heads_decode", batch=batch, layout=f"width_sharded_{n}")
        try:

            def fn():
                q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                    xq, num_heads=32, num_kv_heads=8, memory_config=heads_cfg
                )
                q.deallocate(True)
                k.deallocate(True)
                v.deallocate(True)

            row["us"] = time_trace(fn)
        except Exception as e:
            row["error"] = str(e).splitlines()[0][:300]
        record(row)
        xq.deallocate(True)
    xq = ttnn.from_torch(
        torch.randn(1, 1, batch, 6144).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    row = dict(op="create_qkv_heads_decode", batch=batch, layout="dram_interleaved")

    def fn():
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            xq, num_heads=32, num_kv_heads=8, memory_config=heads_cfg
        )
        q.deallocate(True)
        k.deallocate(True)
        v.deallocate(True)

    row["us"] = time_trace(fn)
    record(row)
    # concat heads output grid
    attn = ttnn.from_torch(
        torch.randn(1, batch, 32, 128).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=heads_cfg,
    )
    c = ttnn.experimental.nlp_concat_heads_decode(attn, num_heads=32)
    record(
        dict(
            op="concat_heads_decode",
            batch=batch,
            out_shape=list(c.shape),
            out_grid=str(c.memory_config().shard_spec.grid),
            shard=list(c.memory_config().shard_spec.shape),
        )
    )

    def fn():
        y = ttnn.experimental.nlp_concat_heads_decode(attn, num_heads=32)
        y.deallocate(True)

    record(dict(op="concat_heads_decode_time", batch=batch, us=time_trace(fn)))
    c.deallocate(True)
    attn.deallocate(True)
    xq.deallocate(True)

# 5. SDPA decode variants (batch 1 and 32, position ~1024, extent 2048)
for batch in (1, 32):
    blocks = 64
    extent = blocks * 32
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
    hx = batch if batch <= 8 else 8
    hy = 1 if batch <= 8 else batch // 8
    heads_cfg = ttnn.create_sharded_memory_config(
        shape=(32, 128),
        core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(hx - 1, hy - 1))]),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    q = ttnn.from_torch(
        torch.randn(1, batch, 32, 128).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=heads_cfg,
    )
    base = None
    for name, pc_args, ck_name, ck in (
        (
            "8x8_q32_k128_hifi4",
            dict(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                q_chunk_size=32,
                k_chunk_size=128,
                exp_approx_mode=False,
            ),
            "hifi4_fp32",
            ckc_hifi4,
        ),
        (
            "8x8_q32_k128_hifi2",
            dict(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                q_chunk_size=32,
                k_chunk_size=128,
                exp_approx_mode=False,
            ),
            "hifi2",
            ckc_hifi2,
        ),
        (
            "8x8_q32_k256_hifi4",
            dict(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                q_chunk_size=32,
                k_chunk_size=256,
                exp_approx_mode=False,
            ),
            "hifi4_fp32",
            ckc_hifi4,
        ),
        (
            "8x8_q32_k512_hifi4",
            dict(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                q_chunk_size=32,
                k_chunk_size=512,
                exp_approx_mode=False,
            ),
            "hifi4_fp32",
            ckc_hifi4,
        ),
        (
            "8x8_q32_k128_approx_hifi4",
            dict(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                q_chunk_size=32,
                k_chunk_size=128,
                exp_approx_mode=True,
            ),
            "hifi4_fp32",
            ckc_hifi4,
        ),
        (
            "11x10_q32_k128_hifi4",
            dict(
                compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
                q_chunk_size=32,
                k_chunk_size=128,
                exp_approx_mode=False,
            ),
            "hifi4_fp32",
            ckc_hifi4,
        ),
        (
            "11x10_q0_k0_hifi4",
            dict(
                compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
                q_chunk_size=0,
                k_chunk_size=0,
                exp_approx_mode=False,
            ),
            "hifi4_fp32",
            ckc_hifi4,
        ),
        (
            "8x8_q0_k0_hifi4",
            dict(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                q_chunk_size=0,
                k_chunk_size=0,
                exp_approx_mode=False,
            ),
            "hifi4_fp32",
            ckc_hifi4,
        ),
    ):
        pc = ttnn.SDPAProgramConfig(**pc_args)
        row = dict(op="sdpa_decode", batch=batch, config=name, ckc=ck_name, extent=extent)
        try:

            def fn():
                y = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                    q,
                    kcache,
                    vcache,
                    page_table_tensor=page_table,
                    cur_pos_tensor=cur_pos,
                    scale=128**-0.5,
                    program_config=pc,
                    compute_kernel_config=ck,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                y.deallocate(True)

            y = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                kcache,
                vcache,
                page_table_tensor=page_table,
                cur_pos_tensor=cur_pos,
                scale=128**-0.5,
                program_config=pc,
                compute_kernel_config=ck,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            out = ttnn.to_torch(y)
            if base is None:
                base = out
            row["pcc_vs_first"] = pcc(base, out)
            y.deallocate(True)
            row["us"] = time_trace(fn)
        except Exception as e:
            row["error"] = str(e).splitlines()[0][:300]
        record(row)
    q.deallocate(True)
    kcache.deallocate(True)
    vcache.deallocate(True)

json.dump(rows, open(OUT, "w"), indent=1)
ttnn.close_mesh_device(mesh)
print("DONE", OUT)
