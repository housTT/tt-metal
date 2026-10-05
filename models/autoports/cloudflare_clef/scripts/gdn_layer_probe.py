"""Isolated GDN layer experiment on the TP=2 path (stage 1 diagnosis).

Loads only the requested GDN layers (default 0 and 30) and, for each layer and length, feeds HF's
bf16 input of that layer and compares the layer's output with HF's output of that layer, as the
PCC of the layer delta (out minus in) per row, summarized per 1024-token block. Variants:
  a) stock: fused chunk kernel, 1024-token chunks with the carried GDN state (the engine path);
  b) seq: the non-fused seq adapter (fused_chunk_enabled patched to False, QWEN_GDN_FP32_STATE=1),
     same chunking;
  c) single: the fused kernel in one pass over the whole length (GDN conv forced to DRAM as in the
     stage 0 experiment), which removes the 1024-token carry.
Each variant is timed warm. A capture wrapper around both delta-rule adapters records the
per-device q/k/v/beta/g inputs and o/final_state outputs of device 0, and a torch fp32 reference
(transformers' torch_chunk_gated_delta_rule with its q/k L2 norm) is run on the concatenated
inputs so the delta-rule core is compared on its own inputs, per block and on the final state.

Usage (device, through devrun):
  python gdn_layer_probe.py [--layers 0,30] [--T 1024,8192] [--out REPORT.json]
"""

from __future__ import annotations

import argparse
import contextlib
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
os.environ.setdefault("QWEN_GDN_FP32_STATE", "1")
BLOCK = 1024


def row_pcc(a, b):
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    return torch.nn.functional.cosine_similarity(a, b, dim=1)


def block_stats(p, T):
    out = {}
    for a in range(0, T, BLOCK):
        b = min(a + BLOCK, T)
        out[f"{a}:{b}"] = (round(float(p[a:b].mean()), 6), round(float(p[a:b].min()), 6))
    return out


class Capture:
    def __init__(self, mesh):
        self.mesh = mesh
        self.calls = []
        self.proj = []
        self.comp = None

    def read_all(self, t):
        import ttnn

        if t is None:
            return None
        if self.comp is None:
            self.comp = ttnn.ConcatMeshToTensor(self.mesh, dim=0)
        full = ttnn.to_torch(t, mesh_composer=self.comp).float()
        return full.reshape(self.mesh.get_num_devices(), -1, *full.shape[1:])

    def read0(self, t):
        full = self.read_all(t)
        return None if full is None else full[0]

    def wrap(self, fn, name):
        def wrapped(q, k, v, beta, g, **kw):
            rec = dict(
                adapter=name,
                q_all=self.read_all(q),
                k_all=self.read_all(k),
                v_all=self.read_all(v),
                beta_all=self.read_all(beta),
                g_all=self.read_all(g),
                initial_state=self.read0(kw.get("initial_state")),
                scale=kw.get("scale"),
                qkv_head_dims=kw.get("qkv_head_dims"),
                return_o_bh=kw.get("return_o_bh"),
                valid_len=kw.get("valid_len"),
            )
            o, state = fn(q, k, v, beta, g, **kw)
            rec["o_all"] = self.read_all(o)
            rec["final_state"] = self.read0(state)
            for key in ("q", "k", "v", "beta", "g", "o"):
                rec[key] = rec[key + "_all"][0]
            rec["proj"] = self.proj.pop() if self.proj else None
            self.calls.append(rec)
            return o, state

        return wrapped

    def wrap_proj(self, fn):
        cap = self

        def wrapped(self_dn, x, S, out_mc=None):
            x_all = cap.read_all(x)
            qkv, z, a, b = fn(self_dn, x, S, out_mc=out_mc)
            cap.proj.append(
                dict(
                    x_all=x_all,
                    qkv_all=cap.read_all(qkv),
                    z_all=cap.read_all(z),
                    a_all=cap.read_all(a),
                    b_all=cap.read_all(b),
                )
            )
            return qkv, z, a, b

        return wrapped


@contextlib.contextmanager
def seq_adapter_forced():
    from models.demos.blackhole.qwen36.tt.gdn import fused_chunk

    original = fused_chunk.fused_chunk_enabled
    fused_chunk.fused_chunk_enabled = lambda: False
    try:
        yield
    finally:
        fused_chunk.fused_chunk_enabled = original


@contextlib.contextmanager
def captured(mesh):
    from models.demos.blackhole.qwen36.tt.gdn import fused_chunk
    from models.demos.blackhole.qwen36.tt.gdn import tp as gdn_tp

    cap = Capture(mesh)
    orig_fused = fused_chunk.chunk_gated_delta_rule_fused_adapter
    orig_seq = gdn_tp.chunk_gated_delta_rule_seq_adapter
    orig_proj = gdn_tp.TPGatedDeltaNet._project_qkvzab
    fused_chunk.chunk_gated_delta_rule_fused_adapter = cap.wrap(orig_fused, "fused")
    gdn_tp.chunk_gated_delta_rule_seq_adapter = cap.wrap(orig_seq, "seq")
    gdn_tp.TPGatedDeltaNet._project_qkvzab = cap.wrap_proj(orig_proj)
    try:
        yield cap
    finally:
        fused_chunk.chunk_gated_delta_rule_fused_adapter = orig_fused
        gdn_tp.chunk_gated_delta_rule_seq_adapter = orig_seq
        gdn_tp.TPGatedDeltaNet._project_qkvzab = orig_proj


def torch_reference(calls, Nk, Dk, Nv, Dv):
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    q = torch.cat([c["q"].reshape(-1, Nk, Dk) for c in calls]).unsqueeze(0)
    k = torch.cat([c["k"].reshape(-1, Nk, Dk) for c in calls]).unsqueeze(0)
    v = torch.cat([c["v"].reshape(-1, Nv, Dv) for c in calls]).unsqueeze(0)
    beta = torch.cat([c["beta"].reshape(-1, Nv) for c in calls]).unsqueeze(0)
    g = torch.cat([c["g"].reshape(-1, Nv) for c in calls]).unsqueeze(0)
    rep = Nv // Nk
    q = q.repeat_interleave(rep, dim=2)
    k = k.repeat_interleave(rep, dim=2)
    with torch.no_grad():
        o, state = torch_chunk_gated_delta_rule(
            q, k, v, g, beta, chunk_size=64, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True
        )
    return o[0].reshape(q.shape[1], Nv * Dv), state[0]


def tt_outputs(calls, Nv, Dv):
    outs = []
    for c in calls:
        o = c["o"]
        if c["return_o_bh"]:
            T = o.shape[-2]
            o = o.reshape(Nv, T, Dv).permute(1, 0, 2).reshape(T, Nv * Dv)
        else:
            o = o.reshape(-1, Nv * Dv)
        outs.append(o)
    return torch.cat(outs), calls[-1]["final_state"].reshape(Nv, -1)


def main():
    import ttnn
    from models.autoports.cloudflare_clef.tests import test_engine as te
    from models.autoports.cloudflare_clef.tests.test_tp2_sanity import gdn_conv_in_dram
    from models.autoports.cloudflare_clef.tt import encode as clef_encode
    from models.autoports.cloudflare_clef.tt.engine import ClefEngine, tp2_mesh
    from models.autoports.cloudflare_clef.tt.loader import ClefModelArgs

    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="0,30")
    parser.add_argument("--T", default="1024,8192")
    parser.add_argument("--out", default="/home/hous/dev/clef/reports/stage1_gdn_layer_probe.json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--variants", default="a,b,c")
    parser.add_argument("--save", default=None)
    args = parser.parse_args()
    wanted = set(args.variants.split(","))
    layers = [int(x) for x in args.layers.split(",")]
    lengths = [int(x) for x in args.T.split(",")]
    Tmax = max(lengths)
    out_path = Path(args.out)

    class SubsetArgs(ClefModelArgs):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.layer_indices = list(layers)
            self.n_layers = len(layers)

    tokenizer = clef_encode.load_tokenizer(SNAPSHOT)
    records = te.read_jsonl(te.RECORDS)
    ids = te.request_ids(tokenizer, records, Tmax)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    hf = te.hf_model(64)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = hf.model.language_model(input_ids=ids, use_cache=False, output_hidden_states=True)
    hs = {i: out.hidden_states[i][0].float() for L in layers for i in (L, L + 1)}
    del out
    logger.info(f"HF T={Tmax}: {time.perf_counter() - t0:.1f} s; kept hidden states {sorted(hs)}")
    report = {"layers": layers, "lengths": lengths, "results": {}, "core": {}}

    with tp2_mesh() as mesh:
        engine = ClefEngine(mesh, args_cls=SubsetArgs, max_state_len=te.MAX_STATE_LEN, snapshot_slots=1)
        model = engine.model
        margs = engine.args
        dim = margs.dim
        report["precision"] = engine.precision
        report["device_dtypes"] = engine.device_dtypes
        comp3 = ttnn.ConcatMeshToTensor(mesh, dim=3)
        tt_layers = {idx: layer for idx, layer in zip(model.layer_indices, model.layers)}

        def to_device_residual(h, bucket, valid):
            full = torch.zeros(1, 1, bucket, dim, dtype=torch.bfloat16)
            full[0, 0, :valid] = h.to(torch.bfloat16)
            return ttnn.from_torch(
                full,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
            )

        def run(layer, h_in, T, single):
            model._reset_gdn_state_for_new_sequence()
            got = torch.empty(T, dim)
            pieces = [(0, T)] if single else [(cs, min(cs + BLOCK, T)) for cs in range(0, T, BLOCK)]
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for cs, ce in pieces:
                valid = ce - cs
                bucket = valid if single else engine.bucket_for(valid)
                x_in = to_device_residual(h_in[cs:ce], bucket, valid)
                x_out = layer.forward(x_in, mode="prefill", chunk_size=margs.gdn_chunk_size, valid_len=valid)
                ttnn.deallocate(x_in)
                got[cs:ce] = ttnn.to_torch(x_out, mesh_composer=comp3)[0, 0, :valid].float()
                ttnn.deallocate(x_out)
            ttnn.synchronize_device(mesh)
            return got, time.perf_counter() - t0

        for L in layers:
            layer = tt_layers[L]
            assert not layer.is_full_attention
            dn = layer.attention
            Nk, Dk, Nv, Dv = dn.Nk, dn.Dk, dn.Nv, dn.Dv
            for T in lengths:
                h_in = hs[L][:T]
                delta_ref = hs[L + 1][:T] - h_in
                in_bf16 = h_in.to(torch.bfloat16).float()
                variants = [("a_stock_chunked", None, False), ("b_seq_chunked", seq_adapter_forced, False)]
                if T > BLOCK:
                    variants.append(("c_fused_single_pass", gdn_conv_in_dram, True))
                for name, ctx, single in variants:
                    if name[0] not in wanted:
                        continue
                    key = f"L{L}_T{T}_{name}"
                    row = {"layer": L, "T": T, "variant": name}
                    try:
                        with ctx() if ctx else contextlib.nullcontext():
                            if name == "a_stock_chunked" or name == "b_seq_chunked":
                                with captured(mesh) as cap:
                                    got, t_first = run(layer, h_in, T, single)
                                calls = cap.calls
                            else:
                                got, t_first = run(layer, h_in, T, single)
                                calls = None
                            times = [t_first]
                            for _ in range(args.repeats - 1):
                                _, t = run(layer, h_in, T, single)
                                times.append(t)
                        p = row_pcc(got - in_bf16, delta_ref)
                        po = row_pcc(got, hs[L + 1][:T])
                        row.update(
                            delta_mean=round(float(p.mean()), 6),
                            delta_min=round(float(p.min()), 6),
                            delta_worst_pos=int(p.argmin()),
                            out_mean=round(float(po.mean()), 6),
                            blocks_mean_min=block_stats(p, T),
                            seconds=[round(t, 4) for t in times],
                        )
                        if calls and args.save:
                            saved = dict(
                                layer=L,
                                T=T,
                                variant=name,
                                Nk=Nk,
                                Dk=Dk,
                                Nv=Nv,
                                Dv=Dv,
                                scale=calls[0]["scale"],
                                layer_out=got,
                                layer_in_bf16=in_bf16,
                                calls=[
                                    {kk: vv for kk, vv in c.items() if kk not in ("q", "k", "v", "beta", "g", "o")}
                                    for c in calls
                                ],
                            )
                            torch.save(saved, args.save.replace(".pt", f"_L{L}_T{T}_{name}.pt"))
                            logger.info(f"saved core inputs to {args.save.replace('.pt', f'_L{L}_T{T}_{name}.pt')}")
                        if calls:
                            ref_o, ref_state = torch_reference(calls, Nk, Dk, Nv, Dv)
                            tt_o, tt_state = tt_outputs(calls, Nv, Dv)
                            pc = row_pcc(tt_o, ref_o)
                            ps = row_pcc(tt_state, ref_state.reshape(Nv, -1))
                            row["core_vs_torch_fp32"] = dict(
                                adapter=calls[0]["adapter"],
                                calls=len(calls),
                                o_mean=round(float(pc.mean()), 6),
                                o_min=round(float(pc.min()), 6),
                                o_blocks_mean_min=block_stats(pc, T),
                                final_state_pcc_mean_over_heads=round(float(ps.mean()), 6),
                                final_state_pcc_min_over_heads=round(float(ps.min()), 6),
                                scale=calls[0]["scale"],
                                qkv_head_dims=list(calls[0]["qkv_head_dims"]) if calls[0]["qkv_head_dims"] else None,
                                initial_state_seen_on_chunk2=calls[1]["initial_state"] is not None
                                if len(calls) > 1
                                else None,
                            )
                    except Exception as error:
                        row["error"] = f"{type(error).__name__}: {str(error)[:400]}"
                        logger.error(f"{key}: {row['error']}")
                    report["results"][key] = row
                    logger.info(f"{key}: {json.dumps(row)}")
                    out_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info(f"GDN_LAYER_PROBE_DONE {out_path}")


if __name__ == "__main__":
    main()
