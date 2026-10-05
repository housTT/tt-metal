"""Per-layer residual-stream PCC of the TP=2 engine against HF, one masked bucket.

Two readings per layer: "accumulated" feeds the TT residual stream through every layer and
compares the output of layer i with HF hidden_states[i+1]; "teacher" feeds HF hidden_states[i]
into TT layer i alone and compares its output with HF hidden_states[i+1], which isolates the
error each layer adds from the drift it inherits. The final row compares the engine's normalized
readout with HF last_hidden_state. Writes a JSON report and logs one line per layer.

Usage (device, through devrun):
  python layer_pcc_probe.py [--n-layers 64] [--T 300] [--out REPORT.json] [--teacher 0|1]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from loguru import logger

SNAPSHOT = (
    "/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c"
)
os.environ.setdefault("CLEF_MODEL", SNAPSHOT)
os.environ.setdefault("HF_MODEL", SNAPSHOT)


def row_pcc(a, b):
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    return torch.nn.functional.cosine_similarity(a, b, dim=1)


def main():
    from models.autoports.cloudflare_clef.tests import test_engine as te
    from models.autoports.cloudflare_clef.tt import encode as clef_encode
    from models.autoports.cloudflare_clef.tt.engine import ClefEngine, tp2_mesh

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-layers", type=int, default=64)
    parser.add_argument("--T", default="300,8192")
    parser.add_argument("--teacher", type=int, default=1)
    parser.add_argument("--out", default=None)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    lengths = [int(t) for t in args.T.split(",")]
    tokenizer = clef_encode.load_tokenizer(SNAPSHOT)
    records = te.read_jsonl(te.RECORDS)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    hf = te.hf_model(args.n_layers)
    with tp2_mesh() as mesh:
        engine = ClefEngine(mesh, max_state_len=te.MAX_STATE_LEN, n_layers=args.n_layers, snapshot_slots=1)
        for T in lengths:
            out_path = Path(
                args.out or f"/home/hous/dev/clef/reports/stage1_layer_pcc_l{args.n_layers}_T{T}{args.tag}.json"
            )
            probe_length(engine, mesh, hf, tokenizer, records, T, args.teacher, out_path)
            logger.info(f"LAYER_PCC_DONE {out_path}")


def probe_length(engine, mesh, hf, tokenizer, records, T, teacher, out_path):
    import ttnn
    from models.autoports.cloudflare_clef.tests import test_engine as te
    from models.tt_transformers.tt.common import Mode, num_blocks_in_seq

    ids = te.request_ids(tokenizer, records, T)
    positions = te.sample_positions(T, 8)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = hf.model.language_model(input_ids=ids, use_cache=False, output_hidden_states=True)
    hs = [h[0].float() for h in out.hidden_states]
    last = out.last_hidden_state[0].float()
    del out
    logger.info(f"HF T={T}: {len(hs)} hidden states, {time.perf_counter() - t0:.1f} s")
    model = engine.model
    margs = engine.args
    report = {"n_layers": margs.n_layers, "T": T, "positions": positions, "precision": engine.precision, "layers": []}
    rep = ttnn.ReplicateTensorToMesh(mesh)
    comp3 = ttnn.ConcatMeshToTensor(mesh, dim=3)
    comp0 = ttnn.ConcatMeshToTensor(mesh, dim=0)
    page_table = engine.page_tables[0]
    chunk = engine.chunk_size
    n_layers = len(model.layers)

    def inputs(cs, bucket):
        cos_t, sin_t = model._rope_tp_cos_sin_torch(cs, bucket)
        cos = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        sin = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        full_pt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
        blk0 = cs // 64
        blkN = num_blocks_in_seq(cs + valid, 64)
        chunk_pt = ttnn.from_torch(
            page_table[:, blk0:blkN].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh
        )
        csi = ttnn.from_torch(
            torch.tensor([cs], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh
        )
        return cos, sin, full_pt, chunk_pt, csi

    def run_layer(layer, x, cs, cos, sin, full_pt, chunk_pt, csi):
        if layer.is_full_attention:
            return layer.forward(
                x,
                cos=cos,
                sin=sin,
                mode="prefill",
                page_table=full_pt,
                chunk_page_table=chunk_pt,
                chunk_start_idx=cs,
                chunk_start_idx_tensor=csi,
                valid_len=valid,
            )
        return layer.forward(x, mode="prefill", chunk_size=margs.gdn_chunk_size, valid_len=valid)

    def read_residual(x):
        return ttnn.to_torch(x, mesh_composer=comp3)[0, 0, :valid].float()

    acc = [torch.empty(T, margs.dim) for _ in range(n_layers)]
    emb_all = torch.empty(T, margs.dim)
    final = torch.empty(T, margs.dim)
    model._reset_gdn_state_for_new_sequence()
    model._build_request_rope(ids, None)
    for cs in range(0, T, chunk):
        ce = min(cs + chunk, T)
        valid = ce - cs
        bucket = engine.bucket_for(valid)
        buf = torch.zeros(1, bucket, dtype=torch.int32)
        buf[:, :valid] = ids[:, cs:ce].to(torch.int32)
        cos, sin, full_pt, chunk_pt, csi = inputs(cs, bucket)
        tok = ttnn.from_torch(buf, dtype=ttnn.uint32, device=mesh, mesh_mapper=rep)
        x = model.embd(tok)
        x = ttnn.reshape(x, (1, 1, bucket, x.shape[-1]))
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        emb_all[cs:ce] = read_residual(x)
        for i, layer in enumerate(model.layers):
            x_new = run_layer(layer, x, cs, cos, sin, full_pt, chunk_pt, csi)
            ttnn.deallocate(x)
            x = x_new
            acc[i][cs:ce] = read_residual(x)
        normed = model.norm(x, mode=Mode.PREFILL)
        ttnn.deallocate(x)
        final[cs:ce] = ttnn.to_torch(normed, mesh_composer=comp0)[0].reshape(-1, margs.dim)[:valid].float()
        ttnn.deallocate(normed)
        for t in (cos, sin, full_pt, chunk_pt, csi):
            ttnn.deallocate(t)
        logger.info(f"T={T}: chunk {cs}:{ce} done")
    e = row_pcc(emb_all, hs[0])
    report["embedding"] = dict(min_pcc=float(e.min()), mean_pcc=float(e.mean()))
    logger.info(f"T={T} embedding: min {float(e.min()):.6f} mean {float(e.mean()):.6f}")
    for i, layer in enumerate(model.layers):
        ref = hs[i + 1]
        p = row_pcc(acc[i], ref)
        row = dict(
            layer=i,
            kind="attn" if layer.is_full_attention else "gdn",
            acc_min_pcc=round(float(p.min()), 6),
            acc_mean_pcc=round(float(p.mean()), 6),
            acc_pos_pcc={q: round(float(p[q]), 6) for q in positions},
            acc_worst_pos=int(p.argmin()),
            acc_rows_below_0_99=int((p < 0.99).sum()),
            ref_norm_mean=round(float(ref.norm(dim=1).mean()), 2),
            tt_norm_mean=round(float(acc[i].norm(dim=1).mean()), 2),
        )
        report["layers"].append(row)
        logger.info(
            f"T={T} layer {i:2d} {row['kind']}: accumulated min {row['acc_min_pcc']:.6f} mean {row['acc_mean_pcc']:.6f} "
            f"rows<0.99 {row['acc_rows_below_0_99']} worst pos {row['acc_worst_pos']} "
            f"norm tt/ref {row['tt_norm_mean']}/{row['ref_norm_mean']}"
        )
    p = row_pcc(final, last)
    report["final_norm"] = dict(
        min_pcc=float(p.min()),
        mean_pcc=float(p.mean()),
        pos={q: float(p[q]) for q in positions},
        worst_pos=int(p.argmin()),
    )
    logger.info(f"T={T} final norm: min {float(p.min()):.6f} mean {float(p.mean()):.6f} worst pos {int(p.argmin())}")
    out_path.write_text(json.dumps(report, indent=2))
    if not teacher:
        return

    def to_device_residual(h, bucket, valid):
        full = torch.zeros(1, 1, bucket, margs.dim, dtype=torch.bfloat16)
        full[0, 0, :valid] = h.to(torch.bfloat16)
        return ttnn.from_torch(
            full,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
        )

    blocks = [(0, min(1024, T))]
    if T > 1024:
        blocks.append((T - 1024, T))
    for i, layer in enumerate(model.layers):
        model._reset_gdn_state_for_new_sequence()
        got = torch.empty(T, margs.dim)
        for cs in range(0, T, chunk):
            ce = min(cs + chunk, T)
            valid = ce - cs
            bucket = engine.bucket_for(valid)
            cos, sin, full_pt, chunk_pt, csi = inputs(cs, bucket)
            x_in = to_device_residual(hs[i][cs:ce], bucket, valid)
            x_out = run_layer(layer, x_in, cs, cos, sin, full_pt, chunk_pt, csi)
            ttnn.deallocate(x_in)
            got[cs:ce] = read_residual(x_out)
            ttnn.deallocate(x_out)
            for t in (cos, sin, full_pt, chunk_pt, csi):
                ttnn.deallocate(t)
        ref = hs[i + 1]
        p = row_pcc(got, ref)
        delta_ref = ref - hs[i]
        delta_got = got - hs[i].to(torch.bfloat16).float()
        pd = row_pcc(delta_got, delta_ref)
        report["layers"][i].update(
            teacher_min_pcc=round(float(p.min()), 6),
            teacher_mean_pcc=round(float(p.mean()), 6),
            teacher_delta_min_pcc=round(float(pd.min()), 6),
            teacher_delta_mean_pcc=round(float(pd.mean()), 6),
            teacher_delta_worst_pos=int(pd.argmin()),
            teacher_delta_pos_pcc={q: round(float(pd[q]), 6) for q in positions},
            teacher_delta_block_mean={f"{a}:{b}": round(float(pd[a:b].mean()), 6) for a, b in blocks},
            teacher_delta_block_min={f"{a}:{b}": round(float(pd[a:b].min()), 6) for a, b in blocks},
        )
        logger.info(
            f"T={T} layer {i:2d} {report['layers'][i]['kind']}: teacher-forced out min {float(p.min()):.6f} "
            f"mean {float(p.mean()):.6f}; layer delta min {float(pd.min()):.6f} mean {float(pd.mean()):.6f} "
            f"worst pos {int(pd.argmin())} blocks {report['layers'][i]['teacher_delta_block_mean']}"
        )
        out_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
