# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact-shape distributed RMSNorm localization, independent of attention/state."""

import argparse
import json

import torch

import ttnn
from models.demos.gpt_oss.tt.ccl import CCLManager

from ..tt.multichip_decoder import MeshConfig, MultichipDecoder
from . import test_functional_decoder as H
from .test_optimization_experiments import recorded_activations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=1)
    parser.add_argument("--dram", action="store_true")
    parser.add_argument("--links", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(8)
    sd, cfg = H.layer_state_dict(0, "real"), H.hf_config()
    value = recorded_activations(0)[:, 128 : 128 + args.length].reshape(1, 1, args.length, 4096).bfloat16()
    weight = (sd["input_layernorm.weight"].float() + 1).bfloat16()
    ref = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + cfg.rms_norm_eps) * weight.float()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    try:
        decoder = object.__new__(MultichipDecoder)
        decoder.mesh_config = MeshConfig(residual="sharded", collective="async", async_links=args.links)
        decoder.ccl = CCLManager(mesh, args.links)
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        memory = ttnn.DRAM_MEMORY_CONFIG if args.dram or args.length > 1 else ttnn.L1_MEMORY_CONFIG

        def upload(v):
            return ttnn.from_torch(
                v,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
            )

        def read(t):
            return [ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(t)]

        def report(name, t):
            parts = read(t)
            print(
                json.dumps(
                    dict(
                        name=name,
                        shape=list(t.shape),
                        dtype=str(t.dtype),
                        parts=[
                            dict(
                                finite=bool(p.isfinite().all()),
                                minimum=float(p.min()),
                                maximum=float(p.max()),
                                first=p.flatten()[:8].tolist(),
                            )
                            for p in parts
                        ],
                    )
                ),
                flush=True,
            )
            return parts

        x = upload(value)
        w = upload(weight.reshape(1, 1, 1, -1))
        stats = ttnn.rms_norm_pre_all_gather(x, dtype=ttnn.bfloat16, compute_kernel_config=compute)
        stats_parts = report("stats", stats)
        stats_all = decoder._gather(stats)
        gathered_parts = report("gathered_stats", stats_all)
        stats_ref = torch.cat(stats_parts, dim=-1)
        print(
            json.dumps(dict(name="gather_exact", exact=[torch.equal(p, stats_ref) for p in gathered_parts])), flush=True
        )
        normalized = ttnn.rms_norm_post_all_gather(
            x, stats_all, epsilon=cfg.rms_norm_eps, weight=w, compute_kernel_config=compute
        )
        norm_parts = report("normalized", normalized)
        joined = torch.cat(norm_parts, dim=-1)
        print(
            json.dumps(dict(name="norm_accuracy", pcc=H.pcc(joined, ref), maxdiff=float((joined - ref).abs().max()))),
            flush=True,
        )
        gathered = decoder._gather(normalized)
        final = report("gathered_norm", gathered)
        print(json.dumps(dict(name="norm_gather_exact", exact=[torch.equal(p, joined) for p in final])), flush=True)
        assert all(torch.equal(p, stats_ref) for p in gathered_parts), "statistics gather differs from host concat"
        assert H.pcc(joined, ref) >= 0.995, "distributed norm differs from CPU"
        assert all(torch.equal(p, joined) for p in final), "activation gather differs from host concat"
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
