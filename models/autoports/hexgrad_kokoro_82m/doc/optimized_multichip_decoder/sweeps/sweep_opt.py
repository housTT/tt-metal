# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Optimized-multichip-decoder sweep harness.

Opens the (1,4) ring mesh ONCE, builds the HF reference once, then for each named
OptConfig variant builds an OptimizedMultichipDecoder, measures warmed traced
decode latency at T=512 and T=128, and PCC vs HF at a max and a short length.

Usage:
  env $ENV python .../sweeps/sweep_opt.py <variant1> <variant2> ...
  (no args -> runs the default variant list)
"""
import json
import os
import random
import sys
import time

import torch

import ttnn
from models.autoports.hexgrad_kokoro_82m.tt.optimized_decoder import OptimizedDecoder, PrecisionPolicy
from models.autoports.hexgrad_kokoro_82m.tt.optimized_multichip_decoder import OptConfig
from models.autoports.hexgrad_kokoro_82m.tt.optimized_multichip_decoder import (
    OptimizedMultichipDecoder as MultichipDecoder,
)

MODEL_ID = "hexgrad/Kokoro-82M"
MESH_SHAPE = (1, 4)
ITERS = int(os.environ.get("KOKORO_SWEEP_ITERS", "50"))
_SYM = "ɐɚɛɜɪʊʌəɹɾɡ aioueɑɔbdfhjklmnprstvwzˈˌ "

# Named variants: name -> dict of OptConfig kwargs (explicit; do not rely on defaults)
_ORIG = {"norm_sharded": False, "fuse_residual": False, "persistent_ccl": False}  # stage-03 multichip behaviour
VARIANTS = {
    "orig": dict(_ORIG),  # no fusion, no sharding
    "fused": {"norm_sharded": False, "fuse_residual": True},  # residual-add fused into interleaved norm
    "sharded": {"norm_sharded": True, "fuse_residual": False},  # SELECTED: sharded norm only
    "fused_sharded": {"norm_sharded": True, "fuse_residual": True},
    "selected": {},  # OptConfig defaults == selected optimized policy
    "ag_bfp8": {**_ORIG, "ag_dtype": "bfp8"},
    "rs_bfp8": {**_ORIG, "rs_dtype": "bfp8"},
    "ccl_bfp8": {**_ORIG, "ag_dtype": "bfp8", "rs_dtype": "bfp8"},
    "sdpa_approx": {**_ORIG, "sdpa_exp_approx": True},
    "fs_ag8": {"norm_sharded": True, "fuse_residual": False, "ag_dtype": "bfp8"},
    "persist_ccl": {"norm_sharded": True, "fuse_residual": False, "persistent_ccl": True},
    # precision re-checks on the multichip FF1 (dominant matmul). "_policy" kwargs
    # override the inherited BFP8/HiFi2 policy.
    "lofi": {"norm_sharded": True, "fuse_residual": False, "_policy": {"matmul_fidelity": "LoFi"}},
    "bfp4_mlp": {"norm_sharded": True, "fuse_residual": False, "_policy": {"mlp_weight": "bfp4"}},
    "bfp4_attn": {"norm_sharded": True, "fuse_residual": False, "_policy": {"attn_weight": "bfp4"}},
}


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return round(torch.corrcoef(torch.stack([a, b]))[0, 1].item(), 5)


def main():
    from huggingface_hub import hf_hub_download
    from transformers import AlbertConfig, AlbertModel

    names = sys.argv[1:] or list(VARIANTS.keys())

    cfg = json.load(open(hf_hub_download(MODEL_ID, "config.json")))
    ac = AlbertConfig(vocab_size=cfg["n_token"], **cfg["plbert"])
    sd = torch.load(hf_hub_download(MODEL_ID, "kokoro-v1_0.pth"), map_location="cpu", weights_only=True)["bert"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    hf = AlbertModel(ac).eval()
    hf.load_state_dict(sd, strict=False)
    vocab = cfg["vocab"]
    pool = [vocab[c] for c in _SYM if c in vocab]
    H = ac.hidden_size

    def rep(L, seed):
        rng = random.Random(seed)
        body = [rng.choice(pool) for _ in range(max(L - 2, 0))]
        ids = ([0] + body + [0])[:L]
        while len(ids) < L:
            ids.append(0)
        return torch.tensor([ids], dtype=torch.long)

    def href(ids):
        with torch.no_grad():
            return hf(ids, attention_mask=torch.ones_like(ids)).last_hidden_state.float()

    # HF references (host, once)
    refs = {L: href(rep(L, 5000 + L)) for L in [512, 64]}

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
    )
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(MESH_SHAPE), trace_region_size=90000000)
    results = {}
    try:
        # Single-chip TTNN reference (replicated on device 0) for vs-single-chip PCC.
        sc_rep = OptimizedDecoder.from_state_dict(sd, hf_config=ac, mesh_device=mesh)

        def sc_prefill(ids):
            p = OptimizedDecoder.prepare_inputs(ids, mesh)
            o = sc_rep.prefill_forward(
                p["input_ids"],
                p["position_ids"],
                p["token_type_ids"],
                p["attention_mask"],
                batch=p["batch"],
                seq_len=p["padded_seq_len"],
            )
            return ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[
                : ids.shape[0], : p["seq_len"], :
            ].float()

        sc_refs = {L: sc_prefill(rep(L, 5000 + L)) for L in [512, 64]}
        sc_rep.release_traces()
        for name in names:
            kw = dict(VARIANTS[name])
            pol_kw = kw.pop("_policy", None)
            policy = PrecisionPolicy(**pol_kw) if pol_kw else None
            opt = OptConfig(**kw)
            dec = MultichipDecoder.from_state_dict(sd, hf_config=ac, mesh_device=mesh, opt=opt, policy=policy)

            def mc_decode(ids):
                p = dec.prepare_inputs(ids)
                o = dec.decode_forward(
                    p["input_ids"],
                    p["position_ids"],
                    p["token_type_ids"],
                    p["attention_mask"],
                    batch=p["batch"],
                    seq_len=p["padded_seq_len"],
                )
                return (
                    ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2))
                    .reshape(ids.shape[0], p["padded_seq_len"], H)[:, : p["seq_len"], :]
                    .float()
                )

            pcc = {
                str(L): {
                    "vs_hf": _pcc(mc_decode(rep(L, 5000 + L)), refs[L]),
                    "vs_sc": _pcc(mc_decode(rep(L, 5000 + L)), sc_refs[L]),
                }
                for L in [512, 64]
            }

            lat = {}
            for L in [512, 128]:
                ids = rep(L, 7)
                p = dec.prepare_inputs(ids)
                dec.decode_forward(
                    p["input_ids"],
                    p["position_ids"],
                    p["token_type_ids"],
                    p["attention_mask"],
                    batch=p["batch"],
                    seq_len=p["padded_seq_len"],
                )
                ttnn.synchronize_device(mesh)
                rec = dec._traces[(1, p["padded_seq_len"], p["attention_mask"] is not None)]
                t0 = time.perf_counter()
                for _ in range(ITERS):
                    ttnn.execute_trace(mesh, rec["trace_id"], cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                lat[str(L)] = round((time.perf_counter() - t0) / ITERS * 1e3, 4)
            dec.release_traces()
            results[name] = {"label": opt.label(), "decode_ms": lat, "pcc": pcc}
            print(f"[{name}] {opt.label()}  decode_ms={lat}  pcc={pcc}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    out = os.path.join(os.path.dirname(__file__), "sweep_results.json")
    prev = {}
    if os.path.exists(out):
        prev = json.load(open(out))
    prev.update(results)
    with open(out, "w") as f:
        json.dump(prev, f, indent=2)
    print("\n==== SUMMARY ====")
    for n, r in results.items():
        print(f"{n:16s} {r['decode_ms']}  pcc={r['pcc']}")


if __name__ == "__main__":
    main()
