# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact failing batch-32 eager/trace case with immutable-state and boundary evidence."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch

import ttnn

from ..tt.multichip_decoder import MeshConfig, MultichipDecoder, fabric_router_config
from ..tt.optimized_decoder import OptimizedDecoder, PrecisionPolicy
from . import test_functional_decoder as H
from .test_optimization_experiments import recorded_activations


def read(value):
    return [ttnn.to_torch(t) for t in ttnn.get_device_tensors(value)]


def digest(values):
    return [hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for t in values]


def scores(ref, value):
    per_user = [H.pcc(ref[u], value[u]) for u in range(32)]
    return dict(
        pcc=H.pcc(ref, value),
        per_user=per_user,
        minimum=min(per_user),
        user=per_user.index(min(per_user)),
        finite=bool(torch.isfinite(value).all()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--variant", default="default")
    parser.add_argument("--policy", default="{}")
    parser.add_argument("--fp32-attention", action="store_true")
    parser.add_argument("--collective", default="native")
    parser.add_argument("--only", default="both")
    parser.add_argument("--qkvg-block", type=int, default=2)
    parser.add_argument("--role-configs", default="{}")
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / "doc/multichip_decoder" / f"{args.name}.pt"
    if path.exists():
        raise FileExistsError(path)
    source = recorded_activations(3)[0]

    def inputs(length, seed):
        indices = (torch.arange(length)[None, :] + seed + torch.arange(32)[:, None] * 137) % source.shape[0]
        return source[indices].clone()

    x = inputs(63, 31)
    tokens = [inputs(1, 3100 + i) for i in range(3)]
    _, refs = H.run_reference(3, "real", x, decode_x=tokens, decode_steps=3)
    artifact = {
        "ref": refs,
        "policy": args.policy,
        "collective": args.collective,
        "qkvg_block": args.qkvg_block,
        "role_configs": args.role_configs,
    }
    from .multichip_topology_candidates import CANDIDATES, fp32_attention_candidate

    target = MultichipDecoder if args.variant == "default" else CANDIDATES[args.variant]
    if args.fp32_attention:
        target = fp32_attention_candidate(target)
    for name, cls, size in [("baseline", OptimizedDecoder, 1), ("tp4", target, 4)]:
        if args.only != "both" and args.only != name:
            continue
        ttnn.set_fabric_config(
            ttnn.FabricConfig.DISABLED if size == 1 else ttnn.FabricConfig.FABRIC_1D_RING,
            router_config=fabric_router_config() if size == 4 else ttnn.FabricRouterConfig(),
        )
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, size), l1_small_size=24576, trace_region_size=32 * 1024 * 1024)
        try:
            H.FunctionalDecoder = cls
            decoder, table, blocks = H.build_decoder(mesh, 3, "real", batch=32, max_context=1024)
            if size == 4 and (
                args.policy != "{}" or args.collective != "native" or args.qkvg_block != 2 or args.role_configs != "{}"
            ):
                roles = dict(MeshConfig().local.role_configs)
                if args.qkvg_block != 2:
                    roles["qkvg"] = {"cores": 8, "block_w": args.qkvg_block, "readers": 1}
                roles.update(json.loads(args.role_configs))
                del decoder
                decoder = cls.from_state_dict(
                    H.layer_state_dict(3, "real"),
                    hf_config=H.hf_config(),
                    layer_idx=3,
                    mesh_device=mesh,
                    max_context=1024,
                    policy=PrecisionPolicy(**json.loads(args.policy)),
                    mesh_config=replace(
                        MeshConfig(collective=args.collective),
                        local=replace(
                            MeshConfig().local,
                            role_configs=roles,
                        ),
                    ),
                )
                decoder.allocate_state(32)
                decoder.allocate_kv_cache(blocks * 32)

            ttnn.deallocate(decoder.prefill_forward(H.to_device(mesh, x), page_table=table))
            buffers = [decoder.k_cache, decoder.v_cache]
            snapshots = [ttnn.clone(v) for v in buffers]
            expected = [digest(read(v)) for v in snapshots]

            def check(label, restored=False):
                hashes = [digest(read(v)) for v in snapshots]
                cache_hashes = [digest(read(v)) for v in buffers] if restored else None
                print(
                    json.dumps(
                        dict(
                            name=name,
                            check=label,
                            snapshot_immutable=hashes == expected,
                            restore_exact=cache_hashes == expected if restored else None,
                        )
                    ),
                    flush=True,
                )
                assert hashes == expected
                if restored:
                    assert cache_hashes == expected

            def restore():
                for s, b in zip(snapshots, buffers):
                    ttnn.copy(s, b)

            token = H.to_device(mesh, tokens[0])
            pos, rot = H.decode_inputs(mesh, torch.full((32,), 63))

            def forward():
                return decoder.decode_forward(token, page_table=table, current_pos=pos, rot_idxs=rot)

            boundaries = {}
            original = decoder._linear

            def linear(value, role, **kw):
                host_in = read(value)
                out = original(value, role, **kw)
                boundaries[role] = {"input": host_in, "output": read(out)}
                return out

            decoder._linear = linear
            eager = forward()
            eager_host = read(eager)
            ttnn.deallocate(eager)
            decoder._linear = original
            check("after_eager")
            restore()
            check("before_capture", True)
            repeated = forward()
            repeated_host = read(repeated)
            ttnn.deallocate(repeated)
            print(
                json.dumps(
                    dict(name=name, repeat_exact=[torch.equal(a, b) for a, b in zip(eager_host, repeated_host)])
                ),
                flush=True,
            )
            restore()
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            out = forward()
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            check("after_capture")
            restore()
            check("before_replay", True)
            trace_hosts = []
            for step in range(3):
                position = torch.full((32,), 63 + step, dtype=torch.int32)
                for host, dest, dtype, layout in [
                    (tokens[step], token, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                    (position, pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                    (position.reshape(1, -1), rot, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                ]:
                    ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=dtype, layout=layout), dest)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                got = read(out)
                trace_hosts.append(got)
                print(
                    json.dumps(
                        dict(
                            name=name,
                            step=step,
                            position=63 + step,
                            block_row=(63 + step) % 64,
                            chunk=0,
                            rank_scores=[scores(refs[step], v) for v in got],
                            trace_eager_exact=(
                                [torch.equal(a, b) for a, b in zip(eager_host, got)] if step == 0 else None
                            ),
                        )
                    ),
                    flush=True,
                )
            check("after_replay")
            artifact[name] = {"eager": eager_host, "trace": trace_hosts, "boundaries": boundaries}
            ttnn.release_trace(mesh, trace)
            del decoder
        finally:
            ttnn.close_mesh_device(mesh)
    path = Path(__file__).resolve().parents[1] / "doc/multichip_decoder" / f"{args.name}.pt"
    torch.save(artifact, path)
    print("ARTIFACT", path, flush=True)


if __name__ == "__main__":
    main()
