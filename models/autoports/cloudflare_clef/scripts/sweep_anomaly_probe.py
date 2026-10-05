"""Per-layer probe of individual sweep200 records against HF (stage 4 remediation).

For each record (one chunk, T <= 1024) at the precision of the current environment:
  accumulated: the TT residual flows through every layer; PCC of layer i against HF hidden_states[i+1];
  teacher: HF hidden_states[i] into TT layer i alone; PCC of the layer delta (out - in) against HF's delta;
  head: the joint schema head on the TT normalized readout (reproduces the sweep row) and on the HF bf16
        last_hidden_state (reproduces the CPU bf16 reference), so the probe is anchored at both ends;
  splice: HF hidden_states[i] into TT layers i..63, normalized, through the head; the probability curve
        over i shows which layers' error moves the decision;
  dump: ids, tt, ref, positions in the CLEF_DUMP_HIDDEN format read by hidden_fp32_control.py.

Host step (no device): --hf-only computes and caches the HF bf16 hidden states per record.
Device step (devrun): reads the cache, opens the TP=2 engine, writes one JSON report and the dumps.

Usage:
  python sweep_anomaly_probe.py --hf-only --hf-cache DIR [--records R.jsonl]
  python sweep_anomaly_probe.py --variant selected --hf-cache DIR --out REPORT.json --dump-dir DIR \
      [--records R.jsonl] [--ref-bf16 REF.jsonl] [--ref-fp32 REF.jsonl] [--sweep-rows ROWS.jsonl] [--splice 1]
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
DEFAULT_RECORDS = "/home/hous/dev/clef/reports/reference/sweep200_disagree2.jsonl"
DEFAULT_REF_BF16 = "/home/hous/dev/clef/reports/reference/sweep200_text.ref_bf16.jsonl"
DEFAULT_REF_FP32 = "/home/hous/dev/clef/reports/reference/sweep200_disagree2.ref_fp32.jsonl"
_hf = {}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def row_pcc(a, b):
    a = a.float() - a.float().mean(dim=1, keepdim=True)
    b = b.float() - b.float().mean(dim=1, keepdim=True)
    return torch.nn.functional.cosine_similarity(a, b, dim=1)


def short_id(record_id):
    return record_id.replace("/", "_")


def cache_path(cache_dir, record_id):
    return Path(cache_dir) / f"hf_bf16_{short_id(record_id)}.pt"


def hf_model(n_layers=64):
    if n_layers in _hf:
        return _hf[n_layers]
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    t0 = time.perf_counter()
    config = AutoConfig.from_pretrained(SNAPSHOT)
    config.text_config.num_hidden_layers = n_layers
    config.text_config.layer_types = config.text_config.layer_types[:n_layers]
    model = Qwen3_5ForConditionalGeneration.from_pretrained(SNAPSHOT, config=config, dtype=torch.bfloat16).eval()
    logger.info(f"HF bf16 model loaded in {time.perf_counter() - t0:.1f} s, {n_layers} layers")
    _hf[n_layers] = model
    return model


def hf_states(ids, path, n_layers=64):
    path = Path(path)
    if path.exists():
        d = torch.load(path, weights_only=False)
        if torch.equal(d["ids"], ids):
            return d["hs"], d["last"]
        logger.warning(f"{path}: cached ids differ from the encoded ids; recomputing")
    model = hf_model(n_layers)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.model.language_model(input_ids=ids, use_cache=False, output_hidden_states=True)
    hs = torch.stack([h[0].float() for h in out.hidden_states])
    last = out.last_hidden_state[0].float()
    logger.info(f"HF forward T={ids.shape[1]}: {hs.shape[0]} hidden states in {time.perf_counter() - t0:.1f} s")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": ids, "hs": hs, "last": last}, path)
    return hs, last


def encode_record(tokenizer, record):
    from models.autoports.cloudflare_clef.tt import encode as clef_encode

    encoded = clef_encode.encode(tokenizer, record)
    ids = torch.tensor([list(encoded.input_ids)], dtype=torch.long)
    state_part, _, _ = clef_encode.split_for_cache(encoded, tokenizer, record)
    return encoded, ids, len(state_part)


def question_dp(a, b):
    return {q: max(abs(a[q][o] - b[q][o]) for o in a[q]) for q in a if q in b}


def argmax(dist):
    return max(dist, key=dist.get)


def rounded(dist):
    return {q: {o: round(p, 4) for o, p in d.items()} for q, d in dist.items()}


def ref_probs(path, record_id):
    if not path:
        return None
    for row in read_jsonl(path):
        if row.get("id") == record_id:
            return row.get("probs")
    return None


def spans_pcc(p, encoded, split):
    rows = {"state_rows": p[:split], "tail_rows": p[split:]}
    out = {k: dict(min=round(float(v.min()), 6), mean=round(float(v.mean()), 6)) for k, v in rows.items() if len(v)}
    for question in encoded.questions:
        a, b = question.question_span
        q = p[a:b]
        opt = torch.cat([p[x:y] for x, y in question.option_spans]) if question.option_spans else q[:0]
        out[f"question_{question.question_id}"] = dict(
            question_span=[int(a), int(b)],
            question_min=round(float(q.min()), 6),
            question_mean=round(float(q.mean()), 6),
            options_min=round(float(opt.min()), 6) if len(opt) else None,
            options_mean=round(float(opt.mean()), 6) if len(opt) else None,
        )
    return out


def worst_rows(p, ids, tokenizer, n=12):
    toks = ids[0].tolist()
    return [dict(pos=int(i), token=tokenizer.decode([toks[i]]), pcc=round(float(p[i]), 6)) for i in p.argsort()[:n]]


def probe_record(engine, mesh, tokenizer, record, encoded, ids, split, hs, last, args, report):
    import ttnn
    from models.autoports.cloudflare_clef.tt import head as clef_head
    from models.tt_transformers.tt.common import Mode, num_blocks_in_seq

    model = engine.model
    margs = engine.args
    T = ids.shape[1]
    valid = T
    bucket = engine.bucket_for(T)
    assert T <= engine.chunk_size, f"T={T} needs more than one chunk; this probe handles one chunk"
    rep = ttnn.ReplicateTensorToMesh(mesh)
    comp3 = ttnn.ConcatMeshToTensor(mesh, dim=3)
    comp0 = ttnn.ConcatMeshToTensor(mesh, dim=0)
    page_table = engine.page_tables[0]
    layers = model.layers
    n_layers = len(layers)
    head = engine.head
    rows_fn = margs.load_lm_head_rows

    def probs(hidden):
        return clef_head.probs_for_record(head, hidden, ids[0], encoded, rows_fn)

    model._build_request_rope(ids, None)
    cos_t, sin_t = model._rope_tp_cos_sin_torch(0, bucket)
    cos = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
    sin = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
    full_pt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
    blkN = num_blocks_in_seq(valid, 64)
    chunk_pt = ttnn.from_torch(
        page_table[:, 0:blkN].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh
    )
    csi = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh
    )

    def run_layer(layer, x):
        if layer.is_full_attention:
            return layer.forward(
                x,
                cos=cos,
                sin=sin,
                mode="prefill",
                page_table=full_pt,
                chunk_page_table=chunk_pt,
                chunk_start_idx=0,
                chunk_start_idx_tensor=csi,
                valid_len=valid,
            )
        return layer.forward(x, mode="prefill", chunk_size=margs.gdn_chunk_size, valid_len=valid)

    def read_residual(x):
        return ttnn.to_torch(x, mesh_composer=comp3)[0, 0, :valid].float()

    def normed_readout(x):
        normed = model.norm(x, mode=Mode.PREFILL)
        ttnn.deallocate(x)
        out = ttnn.to_torch(normed, mesh_composer=comp0)[0].reshape(-1, margs.dim)[:valid].float()
        ttnn.deallocate(normed)
        return out

    def to_device_residual(h):
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

    def run_stack(x, first):
        for j in range(first, n_layers):
            x_new = run_layer(layers[j], x)
            ttnn.deallocate(x)
            x = x_new
        return x

    t0 = time.perf_counter()
    model._reset_gdn_state_for_new_sequence()
    buf = torch.zeros(1, bucket, dtype=torch.int32)
    buf[:, :valid] = ids.to(torch.int32)
    tok = ttnn.from_torch(buf, dtype=ttnn.uint32, device=mesh, mesh_mapper=rep)
    x = model.embd(tok)
    x = ttnn.reshape(x, (1, 1, bucket, x.shape[-1]))
    x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    emb = read_residual(x)
    acc = []
    for layer in layers:
        x_new = run_layer(layer, x)
        ttnn.deallocate(x)
        x = x_new
        acc.append(read_residual(x))
    final = normed_readout(x)
    t_acc = time.perf_counter() - t0
    tt_probs = probs(final)
    hf_probs = probs(last)
    rec = report["records"][record["id"]] = {
        "T": T,
        "bucket": bucket,
        "tail_start": split,
        "questions": [q.question_id for q in encoded.questions],
        "tt_probs": rounded(tt_probs),
        "hf_bf16_head_probs": rounded(hf_probs),
        "ref_bf16_probs": rounded(args.ref_bf16_probs) if args.ref_bf16_probs else None,
        "ref_fp32_probs": rounded(args.ref_fp32_probs) if args.ref_fp32_probs else None,
        "sweep_row_probs": rounded(args.sweep_probs) if args.sweep_probs else None,
        "dp_tt_vs_hf_head": {q: round(v, 4) for q, v in question_dp(hf_probs, tt_probs).items()},
        "dp_hf_head_vs_ref_bf16": (
            {q: round(v, 4) for q, v in question_dp(args.ref_bf16_probs, hf_probs).items()}
            if args.ref_bf16_probs
            else None
        ),
        "dp_tt_vs_sweep_row": (
            {q: round(v, 4) for q, v in question_dp(args.sweep_probs, tt_probs).items()} if args.sweep_probs else None
        ),
        "accumulated_seconds": round(t_acc, 2),
    }
    focus = max(rec["dp_tt_vs_hf_head"], key=rec["dp_tt_vs_hf_head"].get)
    rec["focus_question"] = focus
    rec["focus_ref_argmax"] = argmax(hf_probs[focus])
    rec["focus_tt_argmax"] = argmax(tt_probs[focus])
    logger.info(
        f"{record['id']}: T={T} bucket={bucket} accumulated pass {t_acc:.1f} s; TT probs {rec['tt_probs']}; "
        f"HF-head probs {rec['hf_bf16_head_probs']}; dp TT vs HF head {rec['dp_tt_vs_hf_head']}"
    )
    e = row_pcc(emb, hs[0])
    rec["embedding"] = dict(min_pcc=round(float(e.min()), 6), mean_pcc=round(float(e.mean()), 6))
    per_layer = []
    for i, layer in enumerate(layers):
        p = row_pcc(acc[i], hs[i + 1])
        per_layer.append(
            dict(
                layer=i,
                kind="attn" if layer.is_full_attention else "gdn",
                acc_min_pcc=round(float(p.min()), 6),
                acc_mean_pcc=round(float(p.mean()), 6),
                acc_rows_below_0_99=int((p < 0.99).sum()),
                acc_worst_pos=int(p.argmin()),
                acc_tail_mean_pcc=round(float(p[split:].mean()), 6),
            )
        )
    rec["layers"] = per_layer
    pf = row_pcc(final, last)
    rec["final_norm"] = dict(
        min_pcc=round(float(pf.min()), 6),
        mean_pcc=round(float(pf.mean()), 6),
        rows_below_0_99=int((pf < 0.99).sum()),
        rows_below_0_95=int((pf < 0.95).sum()),
        spans=spans_pcc(pf, encoded, split),
        worst_rows=worst_rows(pf, ids, tokenizer),
    )
    logger.info(
        f"{record['id']}: final norm min {rec['final_norm']['min_pcc']:.6f} mean {rec['final_norm']['mean_pcc']:.6f} "
        f"rows<0.99 {rec['final_norm']['rows_below_0_99']} spans {rec['final_norm']['spans']}"
    )
    if args.dump_dir:
        dump_dir = Path(args.dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        positions = [int(q) for q in torch.linspace(1, T - 1, 8).round().long()]
        dump = dump_dir / f"stage4r_hidden_{args.variant}_{short_id(record['id'])}_T{T}.pt"
        torch.save({"T": T, "ids": ids, "tt": final, "ref": last, "positions": positions}, dump)
        rec["dump"] = str(dump)
    for i, layer in enumerate(layers):
        model._reset_gdn_state_for_new_sequence()
        x_in = to_device_residual(hs[i])
        x_out = run_layer(layer, x_in)
        ttnn.deallocate(x_in)
        if i == n_layers - 1:
            got = normed_readout(x_out)
            p = row_pcc(got, last)
            per_layer[i].update(
                teacher_min_pcc=round(float(p.min()), 6),
                teacher_mean_pcc=round(float(p.mean()), 6),
                teacher_note="layer 63 compared after the final norm against last_hidden_state",
            )
            continue
        got = read_residual(x_out)
        ttnn.deallocate(x_out)
        p = row_pcc(got, hs[i + 1])
        pd = row_pcc(got - hs[i].to(torch.bfloat16).float(), hs[i + 1] - hs[i])
        per_layer[i].update(
            teacher_min_pcc=round(float(p.min()), 6),
            teacher_mean_pcc=round(float(p.mean()), 6),
            teacher_delta_min_pcc=round(float(pd.min()), 6),
            teacher_delta_mean_pcc=round(float(pd.mean()), 6),
            teacher_delta_worst_pos=int(pd.argmin()),
            teacher_delta_worst_token=tokenizer.decode([int(ids[0, int(pd.argmin())])]),
            teacher_delta_tail_mean_pcc=round(float(pd[split:].mean()), 6),
        )
        logger.info(
            f"{record['id']} layer {i:2d} {per_layer[i]['kind']}: accumulated min {per_layer[i]['acc_min_pcc']:.6f} "
            f"mean {per_layer[i]['acc_mean_pcc']:.6f}; teacher delta min {float(pd.min()):.6f} mean {float(pd.mean()):.6f} "
            f"worst pos {int(pd.argmin())}"
        )
    for kind in ("gdn", "attn"):
        xs = [l["teacher_delta_mean_pcc"] for l in per_layer if l["kind"] == kind and "teacher_delta_mean_pcc" in l]
        mins = [l["teacher_delta_min_pcc"] for l in per_layer if l["kind"] == kind and "teacher_delta_min_pcc" in l]
        rec[f"teacher_{kind}"] = dict(
            layers=len(xs),
            mean_of_means=round(sum(xs) / len(xs), 6),
            worst_layer_mean=round(min(xs), 6),
            worst_layer=[l["layer"] for l in per_layer if l.get("teacher_delta_mean_pcc") == min(xs)][0],
            worst_row=round(min(mins), 6),
            layers_below_0_999=[
                l["layer"] for l in per_layer if l["kind"] == kind and 0 < l.get("teacher_delta_mean_pcc", 1) < 0.999
            ],
        )
    logger.info(f"{record['id']}: teacher gdn {rec['teacher_gdn']} attn {rec['teacher_attn']}")
    if args.splice:
        t1 = time.perf_counter()
        splice = []
        ref_top = rec["focus_ref_argmax"]
        tt_top = rec["focus_tt_argmax"]
        for i in range(n_layers):
            model._reset_gdn_state_for_new_sequence()
            x = run_stack(to_device_residual(hs[i]), i)
            p = probs(normed_readout(x))
            splice.append(
                dict(
                    hf_layers=i,
                    tt_layers_from=i,
                    focus_argmax=argmax(p[focus]),
                    p_ref_top=round(p[focus][ref_top], 4),
                    p_tt_top=round(p[focus][tt_top], 4),
                    dp_vs_hf_head=round(question_dp(hf_probs, p)[focus], 4),
                )
            )
        splice.append(
            dict(
                hf_layers=n_layers,
                tt_layers_from=None,
                focus_argmax=argmax(hf_probs[focus]),
                p_ref_top=round(hf_probs[focus][ref_top], 4),
                p_tt_top=round(hf_probs[focus][tt_top], 4),
                dp_vs_hf_head=0.0,
            )
        )
        rec["splice"] = dict(
            question=focus,
            ref_top=ref_top,
            tt_top=tt_top,
            seconds=round(time.perf_counter() - t1, 1),
            curve=splice,
            first_hf_prefix_with_ref_argmax=next(
                (s["hf_layers"] for s in splice if s["focus_argmax"] == ref_top), None
            ),
        )
        logger.info(
            f"{record['id']} splice ({focus}, ref top {ref_top}, TT top {tt_top}): "
            + " ".join(f"{s['hf_layers']}:{s['p_ref_top']:.3f}" for s in splice)
        )
    for t in (cos, sin, full_pt, chunk_pt, csi):
        ttnn.deallocate(t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--ids", default=None)
    parser.add_argument("--hf-cache", required=True)
    parser.add_argument("--hf-only", action="store_true")
    parser.add_argument("--variant", default="selected")
    parser.add_argument("--out", default=None)
    parser.add_argument("--dump-dir", default=None)
    parser.add_argument("--ref-bf16", default=DEFAULT_REF_BF16)
    parser.add_argument("--ref-fp32", default=DEFAULT_REF_FP32)
    parser.add_argument("--sweep-rows", default=None)
    parser.add_argument("--splice", type=int, default=1)
    parser.add_argument("--n-layers", type=int, default=64)
    args = parser.parse_args()
    from models.autoports.cloudflare_clef.tt import encode as clef_encode

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    tokenizer = clef_encode.load_tokenizer(SNAPSHOT)
    records = read_jsonl(args.records)
    if args.ids:
        wanted = set(args.ids.split(","))
        records = [r for r in records if r["id"] in wanted]
    prepared = []
    for record in records:
        encoded, ids, split = encode_record(tokenizer, record)
        hs, last = hf_states(ids, cache_path(args.hf_cache, record["id"]), args.n_layers)
        prepared.append((record, encoded, ids, split, hs, last))
        logger.info(f"{record['id']}: T={ids.shape[1]} tail starts at {split}; HF states ready")
    if args.hf_only:
        logger.info("HF_ONLY_DONE")
        return
    from models.autoports.cloudflare_clef.tt.engine import ClefEngine, tp2_mesh

    out_path = Path(args.out or f"/home/hous/dev/clef/reports/stage4r_anomaly_probe_{args.variant}.json")
    report = {"variant": args.variant, "n_layers": args.n_layers, "records": {}}
    with tp2_mesh() as mesh:
        engine = ClefEngine(mesh, n_layers=args.n_layers, snapshot_slots=1, vision=False)
        report["precision"] = engine.precision
        report["device_dtypes"] = engine.device_dtypes
        report["load_s"] = round(engine.timings["load_total_s"], 1)
        for record, encoded, ids, split, hs, last in prepared:
            args.ref_bf16_probs = ref_probs(args.ref_bf16, record["id"])
            args.ref_fp32_probs = ref_probs(args.ref_fp32, record["id"])
            args.sweep_probs = ref_probs(args.sweep_rows, record["id"])
            probe_record(engine, mesh, tokenizer, record, encoded, ids, split, hs, last, args, report)
            out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info(f"ANOMALY_PROBE_DONE {out_path}")


if __name__ == "__main__":
    main()
