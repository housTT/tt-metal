# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
import time
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AUTOPORT = os.path.dirname(HERE)


def grid_override(max_rows, max_cols):
    def find_prefill_grid(row_tiles, col_tiles):
        rows = next((i for i in range(min(max_rows, row_tiles), 0, -1) if row_tiles % i == 0), 1)
        cols = next((i for i in range(min(max_cols, col_tiles), 0, -1) if col_tiles % i == 0), None)
        if cols is None:
            cols = min(max_cols, col_tiles)
        return rows, cols

    return find_prefill_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", default="8x8,8x10,8x11,10x11")
    ap.add_argument(
        "--force-grids",
        default="",
        help="comma list of rows x cols returned unconditionally by find_prefill_grid (uneven per-core N allowed)",
    )
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(AUTOPORT, "doc", "optimized_decoder", "geometry_experiment.json"))
    a = ap.parse_args()
    import ttnn
    from models.autoports.contrastive_lm_clm_v0_1_8b.reference.hf_layer_reference import (
        capture_hidden_states,
        load_hf,
        real_tokens,
        run_layer,
    )
    from models.common.utility_functions import comp_pcc
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.common import Mode, PagedAttentionConfig, get_rot_transformation_mat
    from models.tt_transformers.tt.decoder import TransformerBlock
    from models.tt_transformers.tt.model_config import ModelArgs
    from models.tt_transformers.tt.rope import get_rot_mats

    os.environ.setdefault("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES", "0")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=0, num_command_queues=1)
    results = []
    mesh_grid = mesh.compute_with_storage_grid_size()
    try:
        tokenizer, hf = load_hf()
        tokens = real_tokens(tokenizer, a.seq_len)
        hs, _ = capture_hidden_states(hf, tokens)
        x = hs[0].float()
        ref = run_layer(hf, 0, x, torch.arange(a.seq_len)[None]).float()
        original = ModelArgs.find_prefill_grid
        specs = [("cap", g) for g in a.grids.split(",") if g] + [("force", g) for g in a.force_grids.split(",") if g]
        for mode, spec in specs:
            rows, cols = (int(v) for v in spec.split("x"))
            row = {"grid_cap": spec, "mode": mode, "patch": "class method before ModelArgs construction"}
            try:
                if mode == "force":
                    ModelArgs.find_prefill_grid = lambda self, r, c, rows=rows, cols=cols: (min(rows, r), cols)
                elif spec == "8x8":
                    ModelArgs.find_prefill_grid = original
                else:
                    fn = grid_override(rows, cols)
                    ModelArgs.find_prefill_grid = lambda self, r, c, fn=fn: fn(r, c)
                model_args = ModelArgs(mesh, max_batch_size=1, max_seq_len=1024, use_hf_rope=False)
                model_args.n_layers = 36
                state_dict = model_args.load_state_dict()
                row["mlp13_grid"] = list(model_args.find_prefill_grid(a.seq_len // 32, model_args.hidden_dim // 32))
                row["qkv_grid"] = list(
                    model_args.find_prefill_grid(
                        a.seq_len // 32, (model_args.dim + 2 * model_args.n_kv_heads * model_args.head_dim) // 32
                    )
                )
                rot_mats = get_rot_mats(
                    head_dim=model_args.head_dim,
                    device=mesh,
                    seq_len=a.seq_len,
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
                x_pad = torch.zeros(1, a.seq_len, model_args.dim)
                x_pad[0] = x
                tt_in = model_args.prepare_residual_tensor_prefill(x_pad)
                out = layer(
                    tt_in,
                    None,
                    rot_mats_global=rot_mats,
                    rot_mats_local=None,
                    user_id=0,
                    mode=Mode.PREFILL,
                    page_table=page_table_tt,
                )
                ttnn.synchronize_device(mesh)
                times = []
                for _ in range(a.repeats):
                    t0 = time.perf_counter()
                    out = layer(
                        tt_in,
                        None,
                        rot_mats_global=rot_mats,
                        rot_mats_local=None,
                        user_id=0,
                        mode=Mode.PREFILL,
                        page_table=page_table_tt,
                    )
                    ttnn.synchronize_device(mesh)
                    times.append(time.perf_counter() - t0)
                y = ttnn.to_torch(
                    out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(1, 3), mesh_shape=model_args.cluster_shape)
                )[0, 0, : a.seq_len, : model_args.dim]
                passing, pcc = comp_pcc(ref, y, 0.995)
                times.sort()
                row.update(
                    {
                        "status": "ok",
                        "pcc": float(pcc),
                        "layer_ms_p50": round(1000 * times[len(times) // 2], 3),
                        "layer_ms_min": round(1000 * times[0], 3),
                    }
                )
                del layer, model_args, state_dict
            except Exception as exc:
                row.update(
                    {"status": "error", "error": str(exc)[:600], "traceback_tail": traceback.format_exc()[-800:]}
                )
            finally:
                ModelArgs.find_prefill_grid = original
            results.append(row)
            print("GRID_ROW", json.dumps(row)[:700], flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    report = {
        "seq_len": a.seq_len,
        "repeats": a.repeats,
        "rows": results,
        "note": "find_prefill_grid patched on the ModelArgs class before construction (cap mode keeps the divisibility rule with a larger cap; force mode returns the grid unconditionally); layer 0, bfp8 weights, accuracy policy, eager (untraced) layer forward timed with synchronize",
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print("GRID_DONE", a.out)


if __name__ == "__main__":
    main()
