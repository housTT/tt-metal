# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serial, independently logged graph candidates with a functional tensor control."""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

MATRIX = {
    "norm": "norm",
    "compact_routing": "moe_gate,rope,cache,qkv,heads,gate,shared_gate,shared_minimal,compact_reduce,compact_routing",
    "compact_ffn": "moe_gate,rope,cache,qkv,heads,gate,shared_gate,shared_minimal,compact_ffn",
    "compact_reduce": "moe_gate,rope,cache,qkv,heads,gate,shared_gate,shared_minimal,compact_reduce",
    "prefill_split": "moe_gate,rope,cache,qkv,heads,gate,shared_gate,shared_minimal,prefill_split",
    "prefill_minimal": "moe_gate,rope,cache,qkv,heads,gate,shared_gate,shared_minimal,prefill_minimal",
    "router": "router",
    "expert_layout": "expert_layout",
    "common": "norm,router,expert_layout",
    "norm_router": "norm,router",
    "kda_conv": "norm,router,kda_conv",
    "gdn_heads": "norm,router,gdn_heads",
    "gdn_norm": "norm,router,gdn_norm",
    "gdn_ternary": "norm,router,gdn_ternary",
    "gdn_eltwise": "norm,router,gdn_eltwise",
    "gdn_grouped": "norm,router,gdn_grouped",
    "gdn_state_ternary": "norm,router,gdn_state_ternary",
    "gdn_grouped_ternary": "norm,router,gdn_grouped,gdn_state_ternary",
    "gdn_exact": "router,gdn_heads,gdn_ternary,gdn_eltwise,gdn_grouped,gdn_state_ternary",
    "native_recurrence": "norm,router,native_recurrence",
    "gdn_all": "norm,router,kda_conv,gdn_heads,gdn_norm,gdn_eltwise",
    "native_all": "norm,router,kda_conv,gdn_heads,gdn_norm,gdn_eltwise,native_recurrence",
    "moe_gate": "moe_gate",
    "shared_pack": "router,shared_pack",
    "shared_gate": "router,shared_gate",
    "shared_sigmoid": "router,shared_sigmoid",
    "shared_minimal": "router,shared_minimal",
    "fused_ffn": "router,fused_ffn",
    "fused_ffn_hifi": "router,fused_ffn_hifi",
    "fused_ffn_grid2": "router,fused_ffn_grid2",
    "full_fast": "moe_gate,rope,cache,qkv,heads,gate,shared_gate",
    "full_minimal": "moe_gate,rope,cache,qkv,heads,gate,shared_gate,shared_minimal",
    "linear_fast": "moe_gate,kda_conv,gdn_heads,gdn_norm,gdn_eltwise,gdn_grouped,gdn_state_ternary,shared_gate,shared_minimal",
    "linear_exact": "moe_gate,gdn_heads,gdn_ternary,gdn_eltwise,gdn_grouped,gdn_state_ternary,shared_gate,shared_minimal",
    "decode_heads": "norm,router,rope,cache,qkv,heads,gate,decode_heads",
    "decode_concat": "norm,router,rope,cache,qkv,heads,gate,decode_concat",
    "attention_native": "norm,router,rope,cache,qkv,heads,gate,decode_heads,decode_concat",
    "rope": "norm,router,rope",
    "cache": "norm,router,cache",
    "qkv": "norm,router,qkv",
    "heads": "norm,router,qkv,heads",
    "gate": "norm,router,gate",
    "attention_all": "norm,router,rope,cache,qkv,heads,gate",
}
for name, extra in {
    "conv_layout": "gdn_conv_layout",
    "inplace": "gdn_inplace",
    "flat_v": "gdn_flat_v",
    "head_major": "gdn_head_major",
    "fused_chunk": "gdn_fused_chunk",
    "chunk_layout": "gdn_flat_v,gdn_head_major,gdn_fused_chunk",
    "linear_finalist": "gdn_inplace,gdn_flat_v,gdn_head_major,gdn_fused_chunk",
}.items():
    MATRIX[name] = MATRIX["linear_exact" if name == "conv_layout" else "linear_fast"] + "," + extra

MATRIX["linear_complete"] = MATRIX["linear_finalist"] + ",compact_reduce,compact_routing"
MATRIX["linear_packed"] = MATRIX["linear_complete"] + ",prefill_minimal"
MATRIX["full_packed"] = MATRIX["compact_routing"] + ",prefill_minimal"
MATRIX["full_split"] = MATRIX["compact_routing"] + ",prefill_split"
MATRIX["linear_split"] = MATRIX["linear_complete"] + ",prefill_split"
MATRIX["full_split32"] = MATRIX["compact_routing"] + ",prefill_split32"
MATRIX["linear_split32"] = MATRIX["linear_complete"] + ",prefill_split32"
MATRIX["kda_tuple"] = MATRIX["linear_split"] + ",kda_tuple"
MATRIX["kda_norm"] = MATRIX["kda_tuple"] + ",kda_norm"
MATRIX["kda_norm_decode"] = MATRIX["kda_tuple"] + ",kda_norm_decode"
MATRIX["full_residual"] = MATRIX["full_split"] + ",projection_residual"
MATRIX["linear_residual"] = MATRIX["kda_norm"] + ",projection_residual,kda_tail"
MATRIX["kda_tail"] = MATRIX["kda_norm"] + ",kda_tail"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, choices=(0, 3), required=True)
    p.add_argument("--length", type=int, default=4096)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--candidates", default="norm,router,expert_layout,common")
    p.add_argument("--directory", type=Path, required=True)
    a = p.parse_args()
    a.directory.mkdir(parents=True, exist_ok=False)
    control = a.directory / "functional_outputs.pt"
    records = []
    for name in ("functional", *a.candidates.split(",")):
        out = a.directory / (name + ".json")
        cmd = [
            sys.executable,
            "-m",
            "models.autoports.qwen_qwen3_5_35b_a3b.tests.run_layer",
            "--layer",
            str(a.layer),
            "--length",
            str(a.length),
            "--steps",
            str(a.steps),
            "--trace",
            "--audit",
            "--warm-prefill",
            "--output",
            str(out),
        ]
        if name == "functional":
            cmd.extend(("--save-tensors", str(control)))
        else:
            cmd.extend(("--decoder", "fused", "--fusions", MATRIX[name], "--compare-tensors", str(control)))
        env = dict(os.environ, TT_METAL_TRACE_ALLOC_TRACKING="1", TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0")
        start = time.time()
        print("START", name, flush=True)
        with (a.directory / (name + ".log")).open("w") as log:
            result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        row = dict(name=name, command=cmd, exit_code=result.returncode, elapsed_seconds=time.time() - start)
        if out.exists():
            d = json.loads(out.read_text())
            row.update(
                prefill_pcc=d["prefill_pcc"],
                decode_pcc_min=d["decode_pcc_min"],
                unfused_equivalence=d.get("unfused_equivalence"),
                median_traced_replay_host_us=statistics.median(d["traced_replay_host_us"]),
                warmed_prefill_host_us=d.get("warmed_prefill_host_us"),
            )
        records.append(row)
        (a.directory / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
        print("END", json.dumps(row), flush=True)
        if name == "functional" and result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
