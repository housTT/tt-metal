"""Op-level candidate probe for the optimized full model (stage 7). Random weights, timing by trace replay.

Sections (PROBE=name, comma separated): lmhead, sampler, topk, norm, heads, prefill_mm.
Each section prints one JSON line per candidate; the driver collects them into probe_candidates.<section>.json.
"""
import json
import math
import os
import time
from types import SimpleNamespace

import torch

import ttnn
from models.autoports.qwen_qwen3_8b.tt.optimized_decoder import (
    DecodeMatmulRole,
    dram_sharded_program_config,
    dram_sharded_weight_memory_config,
    width_sharded_memory_config,
)

H, V, VP = 4096, 151936, 152576
BANKS = 8
sections = os.environ.get("PROBE", "lmhead").split(",")
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=512 * 1024 * 1024)
grid = mesh.compute_with_storage_grid_size()
arch = mesh.arch()
results = []


def ckc(fidelity, fp32, l1acc):
    return ttnn.init_device_compute_kernel_config(
        arch, math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=l1acc
    )


def timed(name, fn, replays=10, **extra):
    """Capture ``fn`` as a trace after one eager warm call; return ms per blocking replay."""
    try:
        out = fn()
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = fn()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        for _ in range(3):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(replays):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ms = (time.perf_counter() - t0) * 1000 / replays
        ttnn.release_trace(mesh, tid)
        rec = {"candidate": name, "ms_per_replay": round(ms, 4), **extra}
    except Exception as exc:  # noqa: BLE001
        rec = {"candidate": name, "error": str(exc).splitlines()[0][:400], **extra}
        out = None
    print("RESULT " + json.dumps(rec), flush=True)
    results.append(rec)
    return out


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    d = float(a.norm() * b.norm())
    return 1.0 if d == 0 else float(a @ b) / d


def lm_head_section():
    splits = [16384] * 9 + [5120]
    x = ttnn.from_torch(torch.randn(1, 1, 32, H).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    x = ttnn.to_memory_config(x, width_sharded_memory_config(H, 32, grid))
    weights = [
        ttnn.from_torch(
            (torch.randn(1, 1, H, w) * 0.02).bfloat16(),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=dram_sharded_weight_memory_config(H, w, BANKS),
        )
        for w in splits
    ]
    mask = torch.zeros(1, 1, 32, 5120, dtype=torch.bfloat16)
    mask[..., 5120 - (VP - V) :] = torch.finfo(torch.bfloat16).min
    tail_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    tail_mask = ttnn.to_memory_config(tail_mask, width_sharded_memory_config(5120, 32, grid))

    def run(fidelity, readers, in0_block_w=4, cores=32, with_mask=False, l1acc=True):
        role = DecodeMatmulRole("bfp8", fidelity, False, cores, in0_block_w, readers)
        compute = ckc({"hifi2": ttnn.MathFidelity.HiFi2, "lofi": ttnn.MathFidelity.LoFi}[fidelity], False, l1acc)
        xin = x if cores == 32 else ttnn.to_memory_config(x, width_sharded_memory_config(H, cores, grid))

        def fn():
            pieces = []
            for i, (w, wt) in enumerate(zip(splits, weights)):
                o = ttnn.linear(
                    xin,
                    wt,
                    program_config=dram_sharded_program_config(role, H, w, BANKS),
                    compute_kernel_config=compute,
                    dtype=ttnn.bfloat16,
                    memory_config=width_sharded_memory_config(w, cores, grid),
                )
                if with_mask and i == len(splits) - 1:
                    o = ttnn.add(o, tail_mask, memory_config=o.memory_config())
                pieces.append(ttnn.sharded_to_interleaved(o, ttnn.DRAM_MEMORY_CONFIG))
            return ttnn.concat(pieces, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return timed(
            f"lm_head {fidelity} readers={readers} in0_block_w={in0_block_w} cores={cores} mask={with_mask} l1acc={l1acc}",
            fn,
            fidelity=fidelity,
            readers=readers,
            in0_block_w=in0_block_w,
            cores=cores,
            with_mask=with_mask,
        )

    base = run("hifi2", 1)
    lofi = run("lofi", 1)
    if base is not None and lofi is not None:
        print(
            "RESULT "
            + json.dumps({"candidate": "lofi_vs_hifi2_pcc", "pcc": pcc(ttnn.to_torch(base), ttnn.to_torch(lofi))}),
            flush=True,
        )
    run("lofi", 2)
    run("hifi2", 2)
    run("lofi", 1, with_mask=True)
    run("lofi", 1, l1acc=False)
    run("lofi", 2, in0_block_w=8, cores=16)
    run("lofi", 1, in0_block_w=8, cores=16)


def topk_section():
    def tensor(width):
        return ttnn.from_torch(
            torch.randn(1, 1, 32, width).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
        )

    full = tensor(VP)
    for k in (128, 64, 32):
        timed(f"topk full row {VP} k={k}", lambda: ttnn.topk(full, k=k, dim=-1, stable=False), width=VP, k=k)
    chunks4 = [tensor(VP // 4) for _ in range(4)]
    timed(
        "topk 4 x 38144 k=32", lambda: [ttnn.topk(c, k=32, dim=-1, stable=False) for c in chunks4], width=VP // 4, k=32
    )
    timed(
        "topk 4 x 38144 k=32 via split",
        lambda: [ttnn.topk(c, k=32, dim=-1, stable=False) for c in ttnn.split(full, VP // 4, dim=3)],
    )
    for width, n in ((16384, 10), (32768, 5), (65536, 3), (8192, 19)):
        chunks = [tensor(width) for _ in range(n)]
        timed(
            f"topk {n} x {width} k=32 (pow2)",
            lambda: [ttnn.topk(c, k=32, dim=-1, stable=False) for c in chunks],
            width=width,
            n=n,
            k=32,
        )
    big = tensor(163840)
    timed("topk full row 163840 k=128", lambda: ttnn.topk(big, k=128, dim=-1, stable=False), width=163840, k=128)
    pieces = [tensor(16384) for _ in range(10)]
    timed("concat 10 x 16384 bf16", lambda: ttnn.concat(pieces, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    x32 = tensor(32 * 10)
    timed("topk 320 wide k=32", lambda: ttnn.topk(x32, k=32, dim=-1, stable=False), width=320, k=32)


def sampler_section():
    from models.common.sampling import SamplingGenerator, SamplingParams, format_sampling_params

    logits = ttnn.from_torch(
        torch.randn(1, 1, 32, VP).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
    )
    tokens = ttnn.from_torch(
        torch.zeros(1, 1, 1, 32, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh
    )
    for vocab in (V, VP):
        args = SimpleNamespace(
            vocab_size=vocab,
            padded_vocab_size=VP,
            cluster_shape=[1, 1],
            max_batch_size=32,
            max_top_k=32,
            sampling_dp=1,
            sub_core_grids=None,
            sub_core_grid_topk=None,
            start_core=ttnn.CoreCoord(0, 0),
            model_config={
                "SAMPLING_AG_CONFIG": {
                    "allow_force_argmax": False,
                    "num_links": 1,
                    "topology": ttnn.Topology.Linear,
                    "chunks_per_sync": 10,
                }
            },
            pad_logits_to_power_of_2=False,
            salt_duplicate_seeds=False,
            use_unfused_ccl=False,
            use_topk_logprobs=False,
        )
        sampler = SamplingGenerator(args=args, mesh_device=mesh, tt_ccl=None)
        params = format_sampling_params(SamplingParams(temperature=[0.0], top_k=[1], top_p=[1.0]), 32, 32)
        sampler.reset_sampling_params(params)
        sampler.set_trace_bucket(1)
        sampler.precompile(logits=logits, tt_out_tok=tokens)
        sampler.capture_trace(logits=logits, tt_out_tok=tokens, skip_precompile=True)
        for _ in range(3):
            sampler.sample(logits, enable_trace=True, tt_out_tok=tokens)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        n = 20
        for _ in range(n):
            sampler.sample(logits, enable_trace=True, tt_out_tok=tokens)
        ttnn.synchronize_device(mesh)
        ms = (time.perf_counter() - t0) * 1000 / n
        ref = int(torch.argmax(ttnn.to_torch(logits).float()[0, 0, 0, :V]))
        got = int(ttnn.to_torch(tokens)[0, 0, 0, 0])
        rec = {
            "candidate": f"sampler greedy trace vocab_size={vocab} (mask {'on' if vocab < VP else 'off'})",
            "ms_per_step": round(ms, 4),
            "greedy_matches_torch": got == ref,
        }
        print("RESULT " + json.dumps(rec), flush=True)
        results.append(rec)
        sampler.reset_trace()
        sampler.set_trace_bucket(None)


def norm_section():
    weight = ttnn.from_torch(torch.ones(1, 1, 1, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    compute = ckc(ttnn.MathFidelity.HiFi4, True, False)
    for rows in (64, 128, 256, 512, 1024, 4096):
        x = ttnn.from_torch(
            torch.randn(1, 1, rows, H).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
        )
        base = timed(
            f"rms_norm interleaved rows={rows}",
            lambda: ttnn.rms_norm(
                x, epsilon=1e-6, weight=weight, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
            ),
            rows=rows,
        )
        mt = rows // 32
        for cores_c in (8, 11):
            kt = H // 32
            if kt % cores_c:
                continue
            cores_r = min(grid.y, mt)
            block_h = math.ceil(mt / cores_r)
            cores_r = math.ceil(mt / block_h)
            block_w = kt // cores_c
            core_grid = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores_c - 1, cores_r - 1))]
            )
            mem = ttnn.create_sharded_memory_config(
                shape=(block_h * 32, block_w * 32),
                core_grid=core_grid,
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            sub = max(w for w in (1, 2, 4) if block_w % w == 0)
            pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(cores_c, cores_r),
                subblock_w=sub,
                block_h=block_h,
                block_w=block_w,
                inplace=False,
            )

            def sharded_roundtrip():
                xs = ttnn.to_memory_config(x, mem)
                y = ttnn.rms_norm(
                    xs, epsilon=1e-6, weight=weight, program_config=pc, memory_config=mem, compute_kernel_config=compute
                )
                return ttnn.sharded_to_interleaved(y, ttnn.DRAM_MEMORY_CONFIG)

            def sharded_only():
                xs = ttnn.to_memory_config(x, mem)
                return ttnn.rms_norm(
                    xs, epsilon=1e-6, weight=weight, program_config=pc, memory_config=mem, compute_kernel_config=compute
                )

            out = timed(
                f"rms_norm block-sharded {cores_c}x{cores_r} block {block_h}x{block_w} sub {sub} rows={rows} (i2s+norm+s2i)",
                sharded_roundtrip,
                rows=rows,
                cores=cores_c * cores_r,
            )
            if out is not None and base is not None:
                print(
                    "RESULT "
                    + json.dumps(
                        {
                            "candidate": f"pcc sharded vs interleaved rows={rows} cores_c={cores_c}",
                            "pcc": pcc(ttnn.to_torch(base), ttnn.to_torch(out)),
                        }
                    ),
                    flush=True,
                )
            timed(
                f"rms_norm block-sharded {cores_c}x{cores_r} rows={rows} (i2s+norm, sharded out)",
                sharded_only,
                rows=rows,
            )
        x.deallocate(True)


def heads_section():
    nh, nkv, hd = 32, 8, 128
    for rows in (128, 256, 1024):
        xqkv = ttnn.from_torch(
            torch.randn(1, 1, rows, (nh + 2 * nkv) * hd).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
        )
        base = timed(
            f"nlp_create_qkv_heads interleaved rows={rows}",
            lambda: ttnn.experimental.nlp_create_qkv_heads(
                xqkv, num_heads=nh, num_kv_heads=nkv, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
            ),
            rows=rows,
        )

        def slice_reshape_transpose():
            q = ttnn.slice(xqkv, [0, 0, 0, 0], [1, 1, rows, nh * hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            k = ttnn.slice(
                xqkv, [0, 0, 0, nh * hd], [1, 1, rows, (nh + nkv) * hd], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            v = ttnn.slice(
                xqkv,
                [0, 0, 0, (nh + nkv) * hd],
                [1, 1, rows, (nh + 2 * nkv) * hd],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            outs = []
            for t, n in ((q, nh), (k, nkv), (v, nkv)):
                r = ttnn.reshape(t, (1, rows, n, hd))
                outs.append(ttnn.transpose(r, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            return outs

        alt = timed(f"slice+reshape+transpose heads rows={rows}", slice_reshape_transpose, rows=rows)
        if base is not None and alt is not None:
            print(
                "RESULT "
                + json.dumps(
                    {
                        "candidate": f"heads alt pcc rows={rows}",
                        "pcc_q": pcc(ttnn.to_torch(base[0]), ttnn.to_torch(alt[0])),
                        "pcc_k": pcc(ttnn.to_torch(base[1]), ttnn.to_torch(alt[1])),
                    }
                ),
                flush=True,
            )
        attn = ttnn.from_torch(
            torch.randn(1, nh, rows, hd).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
        )
        cbase = timed(
            f"nlp_concat_heads interleaved rows={rows}",
            lambda: ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            rows=rows,
        )

        def transpose_reshape():
            t = ttnn.transpose(attn, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return ttnn.reshape(t, (1, 1, rows, nh * hd))

        calt = timed(f"transpose+reshape concat rows={rows}", transpose_reshape, rows=rows)
        if cbase is not None and calt is not None:
            print(
                "RESULT "
                + json.dumps(
                    {"candidate": f"concat alt pcc rows={rows}", "pcc": pcc(ttnn.to_torch(cbase), ttnn.to_torch(calt))}
                ),
                flush=True,
            )
        xqkv.deallocate(True)
        attn.deallocate(True)


def prefill_mm_section():
    roles = {
        "qkv": (H, 6144, "bfp8", 4, 1),
        "o": (H, H, "bfp8", 4, 1),
        "gate": (H, 12288, "bfp8", 4, 1),
        "up": (H, 12288, "bfp4", 4, 3),
        "down": (12288, H, "bfp8", 12, 1),
    }
    lofi = ckc(ttnn.MathFidelity.LoFi, True, False)
    lofi_l1 = ckc(ttnn.MathFidelity.LoFi, False, True)
    for name, (k, n, dtype, in0_block_w, readers) in roles.items():
        wt = (torch.randn(1, 1, k, n) * 0.02).bfloat16()
        tt_dtype = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[dtype]
        w_sh = ttnn.from_torch(
            wt,
            dtype=tt_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=dram_sharded_weight_memory_config(k, n, BANKS),
        )
        w_il = ttnn.from_torch(
            wt, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        w_ref = ttnn.to_torch(w_il).float()
        for rows in (64, 128, 256, 512):
            xt = torch.randn(1, 1, rows, k).bfloat16()
            x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            ref = xt.float() @ w_ref[0, 0]
            base = timed(
                f"{name} ttnn.linear auto interleaved rows={rows}",
                lambda: ttnn.linear(
                    x, w_il, dtype=ttnn.bfloat16, compute_kernel_config=lofi, memory_config=ttnn.DRAM_MEMORY_CONFIG
                ),
                role=name,
                rows=rows,
            )
            if base is not None:
                print(
                    "RESULT "
                    + json.dumps({"candidate": f"{name} auto pcc rows={rows}", "pcc": pcc(ttnn.to_torch(base), ref)}),
                    flush=True,
                )
            timed(
                f"{name} minimal_matmul interleaved rows={rows}",
                lambda: ttnn.experimental.minimal_matmul(
                    x, w_il, dtype=ttnn.bfloat16, compute_kernel_config=lofi, memory_config=ttnn.DRAM_MEMORY_CONFIG
                ),
                role=name,
                rows=rows,
            )
            for blk2d in (16, 8):
                mt, nt = rows // 32, n // 32
                per_core_m = -(-mt // grid.y)
                per_core_n = -(-nt // grid.x)
                best = (1, 1)
                for hh in range(1, 5):
                    for ww in range(1, 5):
                        if (
                            hh * ww <= 4
                            and per_core_m % hh == 0
                            and per_core_n % ww == 0
                            and hh * ww > best[0] * best[1]
                        ):
                            best = (hh, ww)
                pc2d = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(grid.x, grid.y),
                    in0_block_w=blk2d,
                    out_subblock_h=best[0],
                    out_subblock_w=best[1],
                    per_core_M=per_core_m,
                    per_core_N=per_core_n,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=False,
                )
                out2d = timed(
                    f"{name} 2D config in0_block_w={blk2d} sub={best} per_core={per_core_m}x{per_core_n} rows={rows}",
                    lambda: ttnn.linear(
                        x,
                        w_il,
                        program_config=pc2d,
                        dtype=ttnn.bfloat16,
                        compute_kernel_config=lofi,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ),
                    role=name,
                    rows=rows,
                    in0_block_w=blk2d,
                )
                if out2d is not None:
                    print(
                        "RESULT "
                        + json.dumps(
                            {
                                "candidate": f"{name} 2D pcc rows={rows} blk={blk2d}",
                                "pcc": pcc(ttnn.to_torch(out2d), ref),
                            }
                        ),
                        flush=True,
                    )
            for cores, blk, rd in ((32, in0_block_w, readers), (32, in0_block_w, 1), (16, in0_block_w * 2, readers)):
                role = DecodeMatmulRole(dtype, "lofi", False, cores, blk, rd)
                try:
                    pc = dram_sharded_program_config(role, k, n, BANKS)
                    pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                        in0_block_w=pc.in0_block_w,
                        per_core_M=rows // 32,
                        per_core_N=pc.per_core_N,
                        fused_activation=None,
                        num_workers_per_dram_bank=rd,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        "RESULT "
                        + json.dumps(
                            {
                                "candidate": f"{name} dram-sharded rows={rows} cores={cores} blk={blk} rd={rd}",
                                "error": str(exc)[:200],
                            }
                        ),
                        flush=True,
                    )
                    continue
                in_mem = ttnn.create_sharded_memory_config(
                    shape=(rows, k // cores),
                    core_grid=ttnn.num_cores_to_corerangeset(cores, grid, True),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                out_mem = ttnn.create_sharded_memory_config(
                    shape=(rows, n // cores),
                    core_grid=ttnn.num_cores_to_corerangeset(cores, grid, True),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )

                def dram_sharded():
                    xs = ttnn.to_memory_config(x, in_mem)
                    o = ttnn.linear(
                        xs,
                        w_sh,
                        program_config=pc,
                        compute_kernel_config=lofi_l1,
                        dtype=ttnn.bfloat16,
                        memory_config=out_mem,
                    )
                    return ttnn.sharded_to_interleaved(o, ttnn.DRAM_MEMORY_CONFIG)

                out = timed(
                    f"{name} dram-sharded per_core_M={rows // 32} rows={rows} cores={cores} blk={blk} rd={rd} (i2s+mm+s2i)",
                    dram_sharded,
                    role=name,
                    rows=rows,
                    cores=cores,
                    in0_block_w=blk,
                    readers=rd,
                )
                if out is not None:
                    print(
                        "RESULT "
                        + json.dumps(
                            {
                                "candidate": f"{name} dram-sharded pcc rows={rows} cores={cores} blk={blk} rd={rd}",
                                "pcc": pcc(ttnn.to_torch(out), ref),
                            }
                        ),
                        flush=True,
                    )
            x.deallocate(True)
        w_sh.deallocate(True)
        w_il.deallocate(True)


def prefill_mm2_section():
    """Review-1 item: 1D multicast configs over the full grid for the rows still on the auto config."""
    roles = {"qkv": (H, 6144, "bfp8"), "gate": (H, 12288, "bfp8"), "up": (H, 12288, "bfp4")}
    lofi = ckc(ttnn.MathFidelity.LoFi, True, False)
    ncores = grid.x * grid.y
    for name, (k, n, dtype) in roles.items():
        wt = (torch.randn(1, 1, k, n) * 0.02).bfloat16()
        tt_dtype = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[dtype]
        w_il = ttnn.from_torch(
            wt, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        w_ref = ttnn.to_torch(w_il).float()
        for rows in (64, 128, 256):
            xt = torch.randn(1, 1, rows, k).bfloat16()
            x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            ref = xt.float() @ w_ref[0, 0]
            timed(
                f"{name} auto rows={rows}",
                lambda: ttnn.linear(
                    x, w_il, dtype=ttnn.bfloat16, compute_kernel_config=lofi, memory_config=ttnn.DRAM_MEMORY_CONFIG
                ),
                role=name,
                rows=rows,
            )
            mt, nt, kt = rows // 32, n // 32, k // 32
            for cores in (ncores, 96, 64):
                per_core_n = -(-nt // cores)
                for blk in (16, 8, 4):
                    if kt % blk:
                        continue
                    for sub_h, sub_w in ((1, 4), (2, 2), (1, 2), (2, 1), (1, 1)):
                        if mt % sub_h or per_core_n % sub_w or sub_h * sub_w > 4:
                            continue
                        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                            compute_with_storage_grid_size=(grid.x, grid.y),
                            in0_block_w=blk,
                            out_subblock_h=sub_h,
                            out_subblock_w=sub_w,
                            per_core_M=mt,
                            per_core_N=per_core_n,
                            fuse_batch=True,
                            fused_activation=None,
                            mcast_in0=True,
                        )
                        out = timed(
                            f"{name} 1D mcast_in0 cores={cores} in0_block_w={blk} sub={sub_h}x{sub_w} per_core={mt}x{per_core_n} rows={rows}",
                            lambda: ttnn.linear(
                                x,
                                w_il,
                                program_config=pc,
                                dtype=ttnn.bfloat16,
                                compute_kernel_config=lofi,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            ),
                            role=name,
                            rows=rows,
                            cores=cores,
                            in0_block_w=blk,
                            sub=[sub_h, sub_w],
                        )
                        if out is not None:
                            print(
                                "RESULT "
                                + json.dumps(
                                    {
                                        "candidate": f"{name} 1D pcc rows={rows} cores={cores} blk={blk} sub={sub_h}x{sub_w}",
                                        "pcc": pcc(ttnn.to_torch(out), ref),
                                    }
                                ),
                                flush=True,
                            )
                        break
            x.deallocate(True)
        w_il.deallocate(True)


def norm2_section():
    """Review-1 item: sharded RMSNorm at long and non-factoring row counts; two-piece 4096."""
    weight = ttnn.from_torch(torch.ones(1, 1, 1, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    compute = ckc(ttnn.MathFidelity.HiFi4, True, False)
    kt = H // 32
    cores_c, block_w = 8, kt // 8

    def config(tile_rows, block_h, cores_r):
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores_c - 1, cores_r - 1))])
        mem = ttnn.create_sharded_memory_config(
            shape=(block_h * 32, block_w * 32),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(cores_c, cores_r),
            subblock_w=4,
            block_h=block_h,
            block_w=block_w,
            inplace=False,
        )
        return mem, pc

    for rows in (1440, 1472, 1536, 2048, 3072, 4096):
        x = ttnn.from_torch(
            torch.randn(1, 1, rows, H).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
        )
        base = timed(
            f"rms_norm interleaved rows={rows}",
            lambda: ttnn.rms_norm(
                x, epsilon=1e-6, weight=weight, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
            ),
            rows=rows,
        )
        tile_rows = rows // 32
        candidates = []
        for block_h in range(1, tile_rows + 1):
            cores_r = -(-tile_rows // block_h)
            if cores_r <= grid.y:
                candidates.append((block_h, cores_r, cores_r * block_h == tile_rows))
        tried = 0
        for block_h, cores_r, exact in candidates:
            if tried >= 3:
                break
            tried += 1
            mem, pc = config(tile_rows, block_h, cores_r)

            def run():
                xs = ttnn.to_memory_config(x, mem)
                y = ttnn.rms_norm(
                    xs, epsilon=1e-6, weight=weight, program_config=pc, memory_config=mem, compute_kernel_config=compute
                )
                return ttnn.sharded_to_interleaved(y, ttnn.DRAM_MEMORY_CONFIG)

            out = timed(
                f"rms_norm block-sharded 8x{cores_r} block {block_h}x16 rows={rows} exact={exact}",
                run,
                rows=rows,
                block_h=block_h,
                cores_r=cores_r,
                exact_cover=exact,
            )
            if out is not None and base is not None:
                print(
                    "RESULT "
                    + json.dumps(
                        {
                            "candidate": f"pcc rows={rows} block_h={block_h}",
                            "pcc": pcc(ttnn.to_torch(base), ttnn.to_torch(out)),
                        }
                    ),
                    flush=True,
                )
        if rows == 4096:
            mem, pc = config(64, 8, 8)

            def two_pieces():
                outs = []
                for lo in (0, 2048):
                    piece = ttnn.slice(x, [0, 0, lo, 0], [1, 1, lo + 2048, H], memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    xs = ttnn.to_memory_config(piece, mem)
                    y = ttnn.rms_norm(
                        xs,
                        epsilon=1e-6,
                        weight=weight,
                        program_config=pc,
                        memory_config=mem,
                        compute_kernel_config=compute,
                    )
                    outs.append(ttnn.sharded_to_interleaved(y, ttnn.DRAM_MEMORY_CONFIG))
                return ttnn.concat(outs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            out = timed("rms_norm two 2048-row sharded pieces (slice, norm, concat) rows=4096", two_pieces, rows=4096)
            if out is not None and base is not None:
                print(
                    "RESULT "
                    + json.dumps(
                        {"candidate": "pcc rows=4096 two pieces", "pcc": pcc(ttnn.to_torch(base), ttnn.to_torch(out))}
                    ),
                    flush=True,
                )
        x.deallocate(True)


SECTIONS = {
    "lmhead": lm_head_section,
    "topk": topk_section,
    "sampler": sampler_section,
    "norm": norm_section,
    "heads": heads_section,
    "prefill_mm": prefill_mm_section,
    "prefill_mm2": prefill_mm2_section,
    "norm2": norm2_section,
}
for name in sections:
    print(f"SECTION {name}", flush=True)
    results.clear()
    SECTIONS[name]()
    out = f"models/autoports/qwen_qwen3_8b/doc/optimized_full_model/probe_candidates.{name}.json"
    with open(out, "w") as fh:
        json.dump({"section": name, "chip": os.environ.get("TT_VISIBLE_DEVICES"), "results": results}, fh, indent=2)
ttnn.close_mesh_device(mesh)
print("PROBE_OK", flush=True)
