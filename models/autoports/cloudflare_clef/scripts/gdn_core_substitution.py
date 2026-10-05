"""Attribute the TP GDN layer error to its delta-rule core inputs by substitution (host only).

Loads the device capture written by gdn_layer_probe.py --save (per-device q/k/v/beta/g, the
attention-norm output x, the raw qkvzab projection outputs, o and the layer output), hooks HF's
own chunk_gated_delta_rule call for the same layer to get HF's q/k/v/beta/g and core output, and
runs transformers' torch fp32 core with every TT input, every HF input, and one TT input swapped
for the HF one at a time. Also compares each TT input with its HF counterpart (PCC per row over
heads and the max abs error) and emulates the output side (gated RMSNorm, SiLU(z) gate,
out projection) on host from the captured TT o and z.

Usage: python gdn_core_substitution.py --capture FILE.pt [--layer 30] [--T 8192] [--out REPORT.json]
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
BLOCK = 1024


def row_pcc(a, b):
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    return torch.nn.functional.cosine_similarity(a, b, dim=1)


def summary(p, T):
    return dict(
        mean=round(float(p.mean()), 6),
        min=round(float(p.min()), 6),
        worst_pos=int(p.argmin()),
        blocks={f"{a}:{min(a + BLOCK, T)}": round(float(p[a : a + BLOCK].mean()), 6) for a in range(0, T, BLOCK)},
    )


def compare_inputs(name, tt, hf):
    tt = tt.float().reshape(tt.shape[0], -1)
    hf = hf.float().reshape(hf.shape[0], -1)
    p = row_pcc(tt, hf)
    diff = (tt - hf).abs()
    rel = float(diff.max() / (hf.abs().max() + 1e-12))
    return dict(
        pcc_mean=round(float(p.mean()), 6),
        pcc_min=round(float(p.min()), 6),
        max_abs_err=float(diff.max()),
        mean_abs_err=float(diff.mean()),
        hf_max_abs=float(hf.abs().max()),
        hf_mean_abs=float(hf.abs().mean()),
        max_rel_err_to_hf_max=rel,
    )


def main():
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    from models.autoports.cloudflare_clef.tests import test_engine as te
    from models.autoports.cloudflare_clef.tt import encode as clef_encode

    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    L, T = args.layer, args.T
    out_path = Path(args.out or f"/home/hous/dev/clef/reports/stage1_gdn_substitution_L{L}_T{T}.json")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

    cap = torch.load(args.capture)
    Nk, Dk, Nv, Dv = cap["Nk"], cap["Dk"], cap["Nv"], cap["Dv"]
    nd = cap["calls"][0]["q_all"].shape[0]
    rep = Nv // Nk

    def cat_all(key, inner=None):
        parts = []
        for c in cap["calls"]:
            t = c[key] if inner is None else c[inner][key]
            parts.append(t)
        return torch.cat(parts, dim=1)

    tt_q = cat_all("q_all").reshape(nd, T, Nk, Dk)
    tt_k = cat_all("k_all").reshape(nd, T, Nk, Dk)
    tt_v = cat_all("v_all").reshape(nd, T, Nv, Dv)
    tt_beta = cat_all("beta_all").reshape(nd, T, Nv)
    tt_g = cat_all("g_all").reshape(nd, T, Nv)
    tt_o = torch.cat([c["o_all"].reshape(nd, Nv, -1, Dv).permute(0, 2, 1, 3) for c in cap["calls"]], dim=1)
    tt_x = cat_all("x_all", "proj").reshape(nd, T, -1)
    tt_qkv = cat_all("qkv_all", "proj").reshape(nd, T, -1)
    tt_z = cat_all("z_all", "proj").reshape(nd, T, -1)
    tt_a = cat_all("a_all", "proj").reshape(nd, T, -1)
    tt_b = cat_all("b_all", "proj").reshape(nd, T, -1)
    logger.info(
        f"capture: {nd} devices, q {tuple(tt_q.shape)} v {tuple(tt_v.shape)} o {tuple(tt_o.shape)} x {tuple(tt_x.shape)} z {tuple(tt_z.shape)}"
    )

    tokenizer = clef_encode.load_tokenizer(SNAPSHOT)
    records = te.read_jsonl(te.RECORDS)
    ids = te.request_ids(tokenizer, records, T)
    hf = te.hf_model(64)
    text = hf.model.language_model
    gdn = text.layers[L].linear_attn
    hooked = {}
    original = gdn.chunk_gated_delta_rule

    def hook(query, key, value, g, beta, **kw):
        hooked.update(
            q=query[0].float(),
            k=key[0].float(),
            v=value[0].float(),
            g=g[0].float(),
            beta=beta[0].float(),
            kw={k: v for k, v in kw.items() if not torch.is_tensor(v)},
        )
        o, state = original(query, key, value, g, beta, **kw)
        hooked["o"] = o[0].float()
        hooked["state"] = state[0].float() if state is not None else None
        return o, state

    gdn.chunk_gated_delta_rule = hook
    norm_in = {}
    ln = text.layers[L].input_layernorm
    ln.register_forward_hook(lambda m, i, o: norm_in.update(x=o[0].float()))
    gdn.register_forward_hook(lambda m, i, o: norm_in.update(attn_out=(o[0] if isinstance(o, tuple) else o)[0].float()))
    t0 = time.perf_counter()
    with torch.no_grad():
        out = text(input_ids=ids, use_cache=False, output_hidden_states=True)
    hs_in = out.hidden_states[L][0].float()
    hs_out = out.hidden_states[L + 1][0].float()
    del out
    gdn.chunk_gated_delta_rule = original
    logger.info(
        f"HF forward with hooks {time.perf_counter() - t0:.1f} s; hooked keys {sorted(hooked)} kw {hooked['kw']}"
    )

    hf_q, hf_k, hf_v, hf_g, hf_beta, hf_o = (hooked[k] for k in ("q", "k", "v", "g", "beta", "o"))
    logger.info(f"HF q {tuple(hf_q.shape)} v {tuple(hf_v.shape)} g {tuple(hf_g.shape)} o {tuple(hf_o.shape)}")
    report = {"layer": L, "T": T, "inputs": {}, "substitution": {}, "downstream": {}}

    hf_q_d = [hf_q[:, d * Nv : (d + 1) * Nv] for d in range(nd)]
    hf_k_d = [hf_k[:, d * Nv : (d + 1) * Nv] for d in range(nd)]
    hf_v_d = [hf_v[:, d * Nv : (d + 1) * Nv] for d in range(nd)]
    hf_g_d = [hf_g[:, d * Nv : (d + 1) * Nv] for d in range(nd)]
    hf_beta_d = [hf_beta[:, d * Nv : (d + 1) * Nv] for d in range(nd)]
    hf_o_d = [hf_o[:, d * Nv : (d + 1) * Nv] for d in range(nd)]
    tt_q_rep = tt_q.repeat_interleave(rep, dim=2)
    tt_k_rep = tt_k.repeat_interleave(rep, dim=2)
    for d in range(nd):
        report["inputs"][f"device{d}"] = dict(
            q=compare_inputs("q", tt_q_rep[d], hf_q_d[d]),
            k=compare_inputs("k", tt_k_rep[d], hf_k_d[d]),
            v=compare_inputs("v", tt_v[d], hf_v_d[d]),
            beta=compare_inputs("beta", tt_beta[d], hf_beta_d[d]),
            g=compare_inputs("g", tt_g[d], hf_g_d[d]),
            o_tt_core_vs_hf_core=compare_inputs("o", tt_o[d], hf_o_d[d]),
        )
    hdim = tt_x.shape[-1]
    report["inputs"]["attention_norm_output"] = {
        f"device{d}": compare_inputs("x", tt_x[d], norm_in["x"][:, d * hdim : (d + 1) * hdim]) for d in range(nd)
    }
    report["inputs"]["tt_values_bf16_representable"] = {
        name: bool(torch.equal(t.to(torch.bfloat16).float(), t))
        for name, t in (
            ("q", tt_q),
            ("k", tt_k),
            ("v", tt_v),
            ("beta", tt_beta),
            ("g", tt_g),
            ("a", tt_a),
            ("b", tt_b),
            ("z", tt_z),
            ("o", tt_o),
        )
    }
    report["inputs"]["hf_values_bf16_representable"] = {
        name: bool(torch.equal(t.to(torch.bfloat16).float(), t))
        for name, t in (("q", hf_q), ("k", hf_k), ("v", hf_v), ("beta", hf_beta), ("g", hf_g), ("o", hf_o))
    }
    report["inputs"][
        "hf_qkv_dtype_note"
    ] = "HF beta is sigmoid of a bf16 tensor; HF g is fp32 (-exp(A_log)*softplus(a.float()+dt_bias))"
    logger.info(json.dumps(report["inputs"], indent=1)[:3000])

    def core(q, k, v, g, beta):
        with torch.no_grad():
            o, _ = torch_chunk_gated_delta_rule(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                g.unsqueeze(0),
                beta.unsqueeze(0),
                chunk_size=64,
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        return o[0]

    d = 0
    tt_set = dict(q=tt_q_rep[d], k=tt_k_rep[d], v=tt_v[d], g=tt_g[d], beta=tt_beta[d])
    hf_set = dict(q=hf_q_d[d], k=hf_k_d[d], v=hf_v_d[d], g=hf_g_d[d], beta=hf_beta_d[d])
    ref_o = hf_o_d[d].reshape(T, -1)

    def run(name, **over):
        inputs = dict(tt_set)
        inputs.update(over)
        o = core(**inputs).reshape(T, -1)
        p = row_pcc(o, ref_o)
        report["substitution"][name] = summary(p, T)
        logger.info(f"substitution {name}: {report['substitution'][name]}")
        out_path.write_text(json.dumps(report, indent=2))
        return o

    o_all_tt = run("all_tt_inputs")
    run("all_hf_inputs", **hf_set)
    for key in ("g", "beta", "v", "k", "q"):
        run(f"hf_{key}_rest_tt", **{key: hf_set[key]})
    run("hf_g_and_beta_rest_tt", g=hf_set["g"], beta=hf_set["beta"])
    run("hf_q_k_v_rest_tt", q=hf_set["q"], k=hf_set["k"], v=hf_set["v"])
    p = row_pcc(tt_o[d].reshape(T, -1), ref_o)
    report["substitution"]["device_core_output_vs_hf_core"] = summary(p, T)
    p = row_pcc(tt_o[d].reshape(T, -1), o_all_tt)
    report["substitution"]["device_core_output_vs_torch_on_tt_inputs"] = summary(p, T)

    hf_layer_delta = hs_out - hs_in
    tt_layer_delta = cap["layer_out"] - cap["layer_in_bf16"]
    report["layer"] = dict(layer=L, T=T, layer_delta=summary(row_pcc(tt_layer_delta, hf_layer_delta), T))
    norm_w = gdn.norm.weight.float()
    out_w = gdn.out_proj.weight.float()
    eps = float(getattr(gdn.norm, "variance_epsilon", 1e-6))

    def downstream(o_dev, z_dev):
        o_full = torch.cat([o_dev[dd] for dd in range(nd)], dim=1).reshape(T, Nv * nd, Dv)
        z_full = torch.cat([z_dev[dd] for dd in range(nd)], dim=1).reshape(T, Nv * nd, Dv)
        n = o_full * torch.rsqrt(o_full.pow(2).mean(-1, keepdim=True) + eps) * norm_w.reshape(1, 1, Dv)
        gated = n * torch.nn.functional.silu(z_full)
        return gated.reshape(T, -1) @ out_w.T

    hf_core_o_dev = [hf_o[:, dd * Nv : (dd + 1) * Nv] for dd in range(nd)]
    hf_attn_out = norm_in["attn_out"]
    emu_tt = downstream(tt_o, tt_z)
    emu_hf_o_tt_z = downstream(torch.stack(hf_core_o_dev), tt_z)
    report["downstream"] = dict(
        tt_o_tt_z_emulated_vs_hf_attention_block_output=summary(row_pcc(emu_tt, hf_attn_out), T),
        hf_core_o_tt_z_emulated_vs_hf_attention_block_output=summary(row_pcc(emu_hf_o_tt_z, hf_attn_out), T),
        note="emulation in fp32 with HF weights: per-head RMSNorm(o) * norm_w, SiLU(z) gate, out_proj; compared with the HF linear_attn module output",
    )
    out_path.write_text(json.dumps(report, indent=2))
    logger.info(f"downstream: {json.dumps(report['downstream'], indent=1)}")
    logger.info(f"GDN_SUBSTITUTION_DONE {out_path}")


if __name__ == "__main__":
    main()
