# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Prove the selected precision config is CONSUMED by the default construction path.

Builds the model through ``build_generator`` with **no** explicit policy/opt (so it
loads ``selected_precision_config.json`` via ``precision_config.load_selected``),
then reads the ttnn ``.dtype`` off the real device weight tensors and the
``math_fidelity`` / ``fp32_dest_acc`` off the real compute-kernel configs the
forward path uses, and asserts they match the selected config's documented
policy. A JSON field the code ignored would surface here as a mismatch.

Writes propagation_check.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import ttnn
from models.autoports.hexgrad_kokoro_82m.tt.generator import build_generator
from models.autoports.hexgrad_kokoro_82m.tt.precision_config import load_selected

HERE = Path(__file__).parent
MODEL_DIR = HERE.parent.parent


def dt(t):
    return str(t.dtype).replace("DataType.", "")


def fid(cfg):
    return str(cfg.math_fidelity).replace("MathFidelity.", "")


def main():
    policy, opt, raw = load_selected(MODEL_DIR)
    assert raw is not None, "selected_precision_config.json must exist for the propagation check"

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
    )
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=90000000)
    out = {"selected_construct": {"policy": policy.__dict__, "opt": opt.__dict__}, "checks": {}, "mismatches": []}
    try:
        # NB: no policy=/opt= -> build_generator loads the selected config file.
        gen = build_generator(model_dir=str(MODEL_DIR), mesh_device=mesh)
        dec = gen.model.decoder
        w = dec.w
        observed = {
            "attn_qkv_w": dt(w["qkv_w"]),
            "attn_dense_w": dt(w["dense_w"]),
            "mlp_ffn_w": dt(w["ffn_w"]),
            "mlp_ffn_out_w": dt(w["ffn_out_w"]),
            "map_w": dt(w["map_w"]),
            "emb_word": dt(w["word_emb"]),
            "norm_attn_ln_w": dt(w["attn_ln_w"]),
            "readout_w": dt(gen.model.readout_w),
            "matmul_fidelity": fid(dec.matmul_kernel_config),
            "sdpa_fidelity": fid(dec.sdpa_kernel_config),
            "norm_fidelity": fid(dec.norm_kernel_config),
            "matmul_fp32_dest_acc": bool(dec.matmul_kernel_config.fp32_dest_acc_en),
            "ag_ccl_dtype": str(dec.ag_ccl_dtype).replace("DataType.", ""),
            "rs_ccl_dtype": str(dec.rs_ccl_dtype).replace("DataType.", ""),
            "activation_dtype": str(dec.activation_dtype).replace("DataType.", ""),
        }
        out["observed"] = observed

        dtmap = {"bf16": "BFLOAT16", "bfp8": "BFLOAT8_B", "bfp4": "BFLOAT4_B", "fp32": "FLOAT32"}
        exp = {
            "attn_qkv_w": dtmap[policy.attn_weight],
            "attn_dense_w": dtmap[policy.attn_weight],
            "mlp_ffn_w": dtmap[policy.mlp_weight],
            "mlp_ffn_out_w": dtmap[policy.mlp_weight],
            "map_w": dtmap[policy.map_weight],
            "emb_word": dtmap[policy.embedding],
            "norm_attn_ln_w": dtmap[policy.norm_weight],
            "readout_w": "BFLOAT16",  # logits/readout dtype fixed bf16
            "matmul_fidelity": policy.matmul_fidelity,
            "sdpa_fidelity": policy.sdpa_fidelity,
            "norm_fidelity": "HiFi4",
            "matmul_fp32_dest_acc": policy.fp32_dest_acc,
            "ag_ccl_dtype": dtmap[opt.ag_dtype],
            "rs_ccl_dtype": dtmap[opt.rs_dtype],
            "activation_dtype": dtmap[policy.activation],
        }
        for k, ev in exp.items():
            ok = observed[k] == ev
            out["checks"][k] = {"expected": ev, "observed": observed[k], "ok": ok}
            if not ok:
                out["mismatches"].append(k)
        out["all_consumed"] = len(out["mismatches"]) == 0
        print("all_consumed:", out["all_consumed"], "mismatches:", out["mismatches"], flush=True)
        gen.teardown()
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    (HERE / "propagation_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("PROP_OK" if out.get("all_consumed") else "PROP_MISMATCH", flush=True)


if __name__ == "__main__":
    main()
