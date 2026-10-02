# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import math
import os
import time
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AUTOPORT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--precision", default="accuracy")
    ap.add_argument("--only", default="", help="comma list of candidate names to run (default: all for the length)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or os.path.join(AUTOPORT, "doc", "optimized_decoder", f"program_config_experiment_{a.seq_len}.json")

    import ttnn
    from models.autoports.contrastive_lm_clm_v0_1_8b.reference.hf_layer_reference import (
        capture_hidden_states,
        load_hf,
        real_tokens,
        run_layer,
    )
    from models.autoports.contrastive_lm_clm_v0_1_8b.tt.encoder import precision_policy
    from models.common.utility_functions import comp_pcc
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.common import Mode, PagedAttentionConfig, get_rot_transformation_mat
    from models.tt_transformers.tt.decoder import TransformerBlock
    from models.tt_transformers.tt.model_config import ModelArgs
    from models.tt_transformers.tt.rope import get_rot_mats

    os.environ.setdefault("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES", "0")
    S = a.seq_len
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=0, num_command_queues=1)
    results = []
    try:
        tokenizer, hf = load_hf()
        tokens = real_tokens(tokenizer, S)
        hs, _ = capture_hidden_states(hf, tokens)
        x = hs[0].float()
        ref = run_layer(hf, 0, x, torch.arange(S)[None]).float()
        policy = precision_policy(a.precision)
        model_args = ModelArgs(
            mesh,
            max_batch_size=1,
            max_seq_len=1024,
            use_hf_rope=False,
            optimizations=lambda ma: policy(ma.n_layers, ma.model_name),
        )
        model_args.n_layers = 36
        state_dict = model_args.load_state_dict()
        rot_mats = get_rot_mats(
            head_dim=model_args.head_dim,
            device=mesh,
            seq_len=S,
            theta=model_args.rope_theta,
            rope_scaling=model_args.rope_scaling,
        )
        tmats = {
            "prefill": ttnn.as_tensor(
                get_rot_transformation_mat(model_args.head_dim),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
        }
        pac = PagedAttentionConfig(block_size=32, max_num_blocks=1024)
        page_table_tt = ttnn.from_torch(
            torch.arange(pac.max_num_blocks, dtype=torch.int32).reshape(1, -1),
            device=mesh,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        tt_ccl = TT_CCL(mesh)
        layer = TransformerBlock(
            mesh_device=mesh,
            tt_ccl=tt_ccl,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
            layer_num=0,
            dtype=ttnn.bfloat8_b,
            transformation_mats=tmats,
            args=model_args,
            paged_attention_config=pac,
        )
        x_pad = torch.zeros(1, S, model_args.dim)
        x_pad[0] = x
        tt_in = model_args.prepare_residual_tensor_prefill(x_pad)

        def run_layer_tt():
            return layer(
                tt_in,
                None,
                rot_mats_global=rot_mats,
                rot_mats_local=None,
                user_id=0,
                mode=Mode.PREFILL,
                page_table=page_table_tt,
            )

        stock = {
            "qkv": model_args.get_attn_qkv_program_config(Mode.PREFILL, S, None),
            "wo": model_args.get_attn_wo_program_config(Mode.PREFILL, S, None),
            "ff1_3": model_args.get_mlp_ff1_3_prg_config(Mode.PREFILL, S, None),
            "ff2": model_args.get_mlp_ff2_prg_config(Mode.PREFILL, S, None),
            "qkv_mem": model_args.get_attn_qkv_mm_mem_config(Mode.PREFILL, None),
            "use_minimal_qkv": model_args.use_minimal_qkv_prefill_matmul(S),
        }
        dim, hidden, qkv_n = model_args.dim, model_args.hidden_dim, model_args.qkv_size
        Mt = S // 32
        grid_cols = mesh.compute_with_storage_grid_size().x
        grid_rows = mesh.compute_with_storage_grid_size().y

        def mm2d(grid, in0_block_w, out_w, per_core_M, per_core_N, fuse_batch=True):
            return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=grid,
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=out_w,
                per_core_M=per_core_M,
                per_core_N=per_core_N,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=fuse_batch,
            )

        def minimal(mb, kb, nb, grid=(8, 10)):
            return ttnn.MinimalMatmulConfig(
                M_block_size=mb, K_block_size=kb, N_block_size=nb, compute_with_storage_grid_size=ttnn.CoreCoord(*grid)
            )

        def mc(m, k, n, grid, per_core_N, in0_block_w=None):
            return model_args.matmul_config(
                m=m, k=k, n=n, grid_size=grid, in0_block_w=in0_block_w, fuse_batch=S <= 1024, per_core_N=per_core_N
            )

        pcm = max(1, math.ceil(Mt / 8))
        qkv_pcn = math.ceil(qkv_n / 32 / 8)
        cands = []
        if S <= 128:
            cands += [
                ("qkv_b2_w2", {"qkv": mm2d((8, 10), 2, 2, 1, qkv_pcn)}),
                ("qkv_b4_w4", {"qkv": mm2d((8, 10), 4, 4, 1, qkv_pcn)}),
                ("qkv_b8_w4", {"qkv": mm2d((8, 10), 8, 4, 1, qkv_pcn)}),
                ("qkv_wide18_b4_w3", {"qkv": mm2d((11, 10), 4, 3, 1, 18)}),
                ("qkv_wide18_b8_w3", {"qkv": mm2d((11, 10), 8, 3, 1, 18)}),
                ("qkv_minimal_m8", {"qkv": minimal(8, 8, 8), "use_minimal_qkv": True}),
                ("qkv_minimal_m4", {"qkv": minimal(4, 8, 8), "use_minimal_qkv": True}),
                ("wo_wide12", {"wo": mc(S, dim, dim, (11, 8), 12)}),
                ("wo_k16", {"wo": mc(S, dim, dim, (8, 8), 16, in0_block_w=16)}),
                ("ff13_wide36", {"ff1_3": mc(S, dim, hidden, (11, 8), 36)}),
                ("ff13_k16", {"ff1_3": mc(S, dim, hidden, (8, 8), 48, in0_block_w=16)}),
                ("ff2_wide12", {"ff2": mc(S, hidden, dim, (11, 8), 12)}),
                ("ff2_k16", {"ff2": mc(S, hidden, dim, (8, 8), 16, in0_block_w=16)}),
                ("ff2_k16_wide12", {"ff2": mc(S, hidden, dim, (11, 8), 12, in0_block_w=16)}),
                ("norm_sharded_8x4", {"norm_sharded": (8, 4)}),
                ("qkv_1d_sharded_out_8", {"qkv_1d_sharded": 8}),
            ]
        else:
            cands += [
                ("qkv_minimal_grid11x10", {"qkv": minimal(8, 8, 8, grid=(11, 10)), "use_minimal_qkv": True}),
                ("ff2_minimal_grid11x10", {"ff2": minimal(8, 8, 8, grid=(11, 10))}),
                ("norm_sharded_8x8", {"norm_sharded": (8, 8)}),
                ("norm_sharded_8x4", {"norm_sharded": (8, 4)}),
            ]
            if S == 1024:
                cands += [
                    ("wo_wide12", {"wo": mc(S, dim, dim, (11, 8), 12)}),
                    ("wo_k16", {"wo": mc(S, dim, dim, (8, 8), 16, in0_block_w=16)}),
                    ("ff13_wide36", {"ff1_3": mc(S, dim, hidden, (11, 8), 36)}),
                    ("ff13_k16", {"ff1_3": mc(S, dim, hidden, (8, 8), 48, in0_block_w=16)}),
                ]
        if a.only:
            keep = set(a.only.split(","))
            cands = [c for c in cands if c[0] in keep]

        norm_originals = {}

        def apply(overrides):
            applied = []
            if "qkv" in overrides:
                cfg = overrides["qkv"]
                model_args.get_attn_qkv_program_config = lambda mode, seq_len=1, prefetcher=None, cfg=cfg: cfg
                applied.append("get_attn_qkv_program_config")
            if overrides.get("use_minimal_qkv"):
                model_args.use_minimal_qkv_prefill_matmul = lambda seq_len: True
                applied.append("use_minimal_qkv_prefill_matmul")
            if "wo" in overrides:
                cfg = overrides["wo"]
                model_args.get_attn_wo_program_config = lambda mode, seq_len=1, prefetcher=None, cfg=cfg: cfg
                applied.append("get_attn_wo_program_config")
            if "ff1_3" in overrides:
                cfg = overrides["ff1_3"]
                model_args.get_mlp_ff1_3_prg_config = lambda mode, seq_len=1, prefetcher=None, cfg=cfg: cfg
                applied.append("get_mlp_ff1_3_prg_config")
            if "ff2" in overrides:
                cfg = overrides["ff2"]
                model_args.get_mlp_ff2_prg_config = lambda mode, seq_len=1, prefetcher=None, cfg=cfg: cfg
                applied.append("get_mlp_ff2_prg_config")
            if "qkv_1d_sharded" in overrides:
                ncores = overrides["qkv_1d_sharded"]
                shard_w = qkv_n // ncores
                grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ncores - 1, 0))})
                memcfg = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(grid, (S, shard_w), ttnn.ShardOrientation.ROW_MAJOR),
                )
                cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(ncores, 1),
                    in0_block_w=8,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    per_core_M=Mt,
                    per_core_N=shard_w // 32,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
                model_args.get_attn_qkv_program_config = lambda mode, seq_len=1, prefetcher=None, cfg=cfg: cfg
                model_args.get_attn_qkv_mm_mem_config = lambda mode, prefetcher=None, memcfg=memcfg: memcfg
                applied += ["get_attn_qkv_program_config", "get_attn_qkv_mm_mem_config"]
            if "norm_sharded" in overrides:
                cols, rows = overrides["norm_sharded"]
                ncores = cols * rows
                shard_w = dim // cols
                grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))})
                memcfg = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(grid, (S // rows, shard_w), ttnn.ShardOrientation.ROW_MAJOR),
                )
                block_w = shard_w // 32
                sub_w = next(w for w in (4, 2, 1) if block_w % w == 0)
                prg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=[cols, rows],
                    subblock_w=sub_w,
                    block_h=S // rows // 32,
                    block_w=block_w,
                    inplace=False,
                )
                for name in ("attention_norm", "ff_norm"):
                    dn = getattr(layer, name)
                    norm_originals[name] = dn.__dict__.get("forward")

                    def fwd(x, mode, norm_config=None, dn=dn):
                        xs = ttnn.to_memory_config(x, memcfg)
                        y = ttnn.rms_norm(
                            xs,
                            epsilon=dn.norm.eps,
                            weight=dn.norm.weight,
                            program_config=prg,
                            compute_kernel_config=dn.norm.compute_kernel_config_hifi2,
                        )
                        ttnn.deallocate(xs)
                        out = ttnn.sharded_to_interleaved(y, ttnn.DRAM_MEMORY_CONFIG)
                        ttnn.deallocate(y)
                        return out

                    dn.forward = fwd
                applied.append("norm_forward")
            return applied

        def restore():
            for name in (
                "get_attn_qkv_program_config",
                "use_minimal_qkv_prefill_matmul",
                "get_attn_wo_program_config",
                "get_mlp_ff1_3_prg_config",
                "get_mlp_ff2_prg_config",
                "get_attn_qkv_mm_mem_config",
            ):
                model_args.__dict__.pop(name, None)
            for name in ("attention_norm", "ff_norm"):
                getattr(layer, name).__dict__.pop("forward", None)

        def measure(name, overrides):
            row = {"name": name, "seq_len": S, "precision": a.precision}
            try:
                row["applied"] = apply(overrides)
                row["consumed"] = {
                    "qkv": str(model_args.get_attn_qkv_program_config(Mode.PREFILL, S, None)),
                    "use_minimal_qkv": bool(model_args.use_minimal_qkv_prefill_matmul(S)),
                    "wo": str(model_args.get_attn_wo_program_config(Mode.PREFILL, S, None)),
                    "ff1_3": str(model_args.get_mlp_ff1_3_prg_config(Mode.PREFILL, S, None)),
                    "ff2": str(model_args.get_mlp_ff2_prg_config(Mode.PREFILL, S, None)),
                    "qkv_mem": str(model_args.get_attn_qkv_mm_mem_config(Mode.PREFILL, None)),
                }
                y = run_layer_tt()
                ttnn.synchronize_device(mesh)
                times = []
                for _ in range(a.repeats):
                    t0 = time.perf_counter()
                    y = run_layer_tt()
                    ttnn.synchronize_device(mesh)
                    times.append(time.perf_counter() - t0)
                yt = ttnn.to_torch(
                    y, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(1, 3), mesh_shape=model_args.cluster_shape)
                )[0, 0, :S, :dim]
                _, pcc = comp_pcc(ref, yt, 0.995)
                times.sort()
                row.update(
                    {
                        "status": "ok",
                        "pcc_vs_hf_bf16_layer": float(pcc),
                        "layer_ms_p50": round(1000 * times[len(times) // 2], 3),
                        "layer_ms_min": round(1000 * times[0], 3),
                    }
                )
            except Exception as exc:
                row.update(
                    {
                        "status": "error",
                        "error": str(exc).splitlines()[0][:400],
                        "traceback_tail": traceback.format_exc()[-1200:],
                    }
                )
            finally:
                restore()
            results.append(row)
            print(
                "PC_ROW",
                json.dumps({k: v for k, v in row.items() if k not in ("traceback_tail", "consumed")})[:600],
                flush=True,
            )
            return row

        baseline = measure("stock", {})
        for name, ov in cands:
            measure(name, ov)
        base_ms = baseline.get("layer_ms_p50")
        kept = {}
        for r in results[1:]:
            if (
                r.get("status") == "ok"
                and base_ms
                and r["layer_ms_p50"] < base_ms * 0.985
                and r["pcc_vs_hf_bf16_layer"] >= 0.995
            ):
                group = r["name"].split("_")[0]
                if group not in kept or r["layer_ms_p50"] < kept[group]["layer_ms_p50"]:
                    kept[group] = r
        if kept:
            combo = {}
            for g, r in kept.items():
                for name, ov in cands:
                    if name == r["name"]:
                        combo.update(ov)
            measure("combo:" + "+".join(sorted(r["name"] for r in kept.values())), combo)
        measure("stock_again", {})
    finally:
        ttnn.close_mesh_device(mesh)
    report = {
        "seq_len": S,
        "precision": a.precision,
        "repeats": a.repeats,
        "timing": "eager layer forward with synchronize_device after each call (includes dispatch); p50 and min of repeats",
        "reference": "HF Qwen3-8B layer 0 in bfloat16 on CPU (reference/hf_layer_reference.py)",
        "stock_configs": {k: str(v) for k, v in stock.items()},
        "rows": results,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w"), indent=1)
    print("PC_DONE", out)


if __name__ == "__main__":
    main()
