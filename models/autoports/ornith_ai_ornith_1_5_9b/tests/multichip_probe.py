# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serialized real-weight TP probe; invoke as a Python module."""

import argparse
import json
import statistics
import time
from dataclasses import asdict, replace

import torch

import ttnn

from ..tt.functional_decoder import num_blocks_for_context
from ..tt.multichip_decoder import MeshConfig, MultichipDecoder
from ..tt.optimized_decoder import OptimizedDecoder, PrecisionPolicy
from . import test_functional_decoder as H
from .test_optimization_experiments import recorded_activations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--prefill-iterations", type=int, default=4)
    parser.add_argument("--residual", default="replicated")
    parser.add_argument("--collective", default="native")
    parser.add_argument("--async-links", type=int, default=1)
    parser.add_argument("--local-config", default="{}")
    parser.add_argument("--mesh-config", default="{}")
    parser.add_argument("--role-configs", default="{}")
    parser.add_argument("--prefill-role-blocks", default="{}")
    parser.add_argument("--policy", default="{}")
    parser.add_argument("--fp32-attention", action="store_true")
    parser.add_argument("--variant", default="default")
    parser.add_argument("--activation-group", choices=["none", "attention", "mlp", "all"], default="none")
    parser.add_argument("--wide-grid", default="[8,8]")
    parser.add_argument(
        "--decode-qkvg-dtype", default=MeshConfig().decode_qkvg_dtype, choices=["bfloat4_b", "bfloat8_b", "baseline"]
    )
    parser.add_argument("--decode-grid", default="[8,4]", help="JSON grid or null for DRAM-sharded control")
    parser.add_argument("--interleaved-qkvg", action="store_true")
    parser.add_argument("--packet-size", type=int, default=8192)
    parser.add_argument("--l1-small-size", type=int, default=24576)
    args = parser.parse_args()
    if args.prefill_iterations < 2:
        parser.error("--prefill-iterations must include a warmup and a measured iteration")
    torch.set_num_threads(8)
    sd, cfg = H.layer_state_dict(args.layer, "real"), H.hf_config()
    values = recorded_activations(args.layer)
    outputs = {}
    from .multichip_topology_candidates import CANDIDATES, fp32_attention_candidate
    from .optimized_multichip_candidates import CANDIDATES as stage_candidates

    CANDIDATES = {**CANDIDATES, **stage_candidates}

    target = MultichipDecoder if args.variant == "default" else CANDIDATES[args.variant]
    if args.variant in ("wide_interleaved", "qkvg_grid", "packed_gdn_grid"):
        target.grid = tuple(json.loads(args.wide_grid))
    if args.activation_group != "none":
        base = target

        class ActivationCandidate(base):
            def _linear(self, x, role, **kwargs):
                mlp = role in ("gate_proj", "up_proj", "gate_up", "down_proj")
                selected = args.activation_group == "all" or (args.activation_group == "mlp") == mlp
                if not selected:
                    return super()._linear(x, role, **kwargs)
                reduced_input = ttnn.typecast(x, ttnn.bfloat8_b)
                kwargs["dtype"] = ttnn.bfloat8_b
                reduced_output = super()._linear(reduced_input, role, **kwargs)
                result = ttnn.typecast(reduced_output, ttnn.bfloat16)
                ttnn.deallocate(reduced_input)
                ttnn.deallocate(reduced_output)
                return result

        target = ActivationCandidate
    if args.fp32_attention:
        target = fp32_attention_candidate(target)
    prefill_blocks = json.loads(args.prefill_role_blocks)
    if prefill_blocks:
        base = target

        class PrefillRoleBlocks(base):
            def _prefill_linear(self, x, role, **kwargs):
                if role not in prefill_blocks:
                    return super()._prefill_linear(x, role, **kwargs)
                original = self.optimization
                self.optimization = replace(original, large_prefill_block_w=prefill_blocks[role])
                try:
                    return super()._prefill_linear(x, role, **kwargs)
                finally:
                    self.optimization = original

        target = PrefillRoleBlocks
    for name, cls, devices in [("baseline", OptimizedDecoder, 1), ("tp4", target, 4)]:
        fabric = ttnn.FabricConfig.DISABLED if devices == 1 else ttnn.FabricConfig.FABRIC_1D_RING
        router = ttnn.FabricRouterConfig()
        if devices == 4 and args.packet_size:
            router.max_packet_payload_size_bytes = args.packet_size
        ttnn.set_fabric_config(fabric, router_config=router)
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, devices),
            trace_region_size=32 * 1024 * 1024,
            l1_small_size=args.l1_small_size,
        )
        try:
            sharded = devices == 4 and args.residual == "sharded"

            def upload(value):
                return ttnn.from_torch(
                    value,
                    device=mesh,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1) if sharded else ttnn.ReplicateTensorToMesh(mesh),
                )

            def read(value):
                parts = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(value)]
                if not sharded:
                    assert all(torch.equal(parts[0], part) for part in parts[1:]), "residual replicas differ"
                return [torch.cat(parts, dim=-1)] if sharded else parts

            x = upload(values[:, : args.length])
            token = upload(values[:, args.length : args.length + 1])
            positions, rotations = H.decode_inputs(mesh, torch.tensor([args.length]))
            blocks = num_blocks_for_context(4096)
            table = H.to_device(
                mesh,
                torch.arange(blocks, dtype=torch.int32).reshape(1, -1),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            print("CONSTRUCT", name, flush=True)
            plan = MeshConfig(
                residual=args.residual,
                collective=args.collective,
                async_links=args.async_links,
                decode_grid=json.loads(args.decode_grid),
                decode_qkvg_dtype=None if args.decode_qkvg_dtype == "baseline" else args.decode_qkvg_dtype,
                decode_qkvg_dram=not args.interleaved_qkvg,
            )
            plan = replace(plan, local=replace(plan.local, **json.loads(args.local_config)))
            plan = replace(plan, **json.loads(args.mesh_config))
            plan = replace(
                plan,
                local=replace(plan.local, role_configs={**plan.local.role_configs, **json.loads(args.role_configs)}),
            )
            extra = {"mesh_config": plan, "policy": PrecisionPolicy(**json.loads(args.policy))} if devices == 4 else {}
            decoder = cls.from_state_dict(
                sd, hf_config=cfg, layer_idx=args.layer, mesh_device=mesh, max_context=4096, **extra
            )

            print(
                "MODEL_CONFIG "
                + json.dumps(
                    dict(
                        name=name,
                        l1_small_size=args.l1_small_size,
                        prefill_role_blocks=prefill_blocks if devices == 4 else {},
                        precision_policy=asdict(decoder.policy),
                        mesh_config=asdict(decoder.mesh_config) if devices == 4 else None,
                        prefill_weights={role: str(decoder.w[role].dtype) for role in decoder.projection_compute},
                        decode_weights={
                            role: str(decoder.decode_weights.get(role, decoder.w[role]).dtype)
                            for role in decoder.projection_compute
                        },
                        projection_compute={role: str(value) for role, value in decoder.projection_compute.items()},
                    )
                ),
                flush=True,
            )
            decoder.allocate_state(1)
            decoder.allocate_kv_cache(blocks)
            print("PREFILL", name, flush=True)
            durations = []
            for iteration in range(args.prefill_iterations):
                decoder.reset_state()
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                prefill = decoder.prefill_forward(x, page_table=table)
                ttnn.synchronize_device(mesh)
                if iteration:
                    durations.append((time.perf_counter() - start) * 1000)
                if iteration < args.prefill_iterations - 1:
                    ttnn.deallocate(prefill)
            print(
                json.dumps(dict(name=name, prefill_ms=statistics.median(durations), prefill_samples_ms=durations)),
                flush=True,
            )
            prefills = read(prefill)
            ttnn.deallocate(prefill)
            print("DECODE", name, flush=True)
            buffers = (
                [decoder.k_cache, decoder.v_cache]
                if decoder.is_full_attention
                else [decoder.recurrent_state, *decoder.conv_state]
            )
            snapshots = [ttnn.clone(t) for t in buffers]

            def restore():
                for source, target in zip(snapshots, buffers):
                    ttnn.copy(source, target)

            def decode():
                return decoder.decode_forward(token, page_table=table, current_pos=positions, rot_idxs=rotations)

            output = decode()
            state_host = [[ttnn.to_torch(part) for part in ttnn.get_device_tensors(buf)] for buf in buffers]
            outputs[name] = [prefills, read(output), state_host]
            ttnn.deallocate(output)
            restore()
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            traced = decode()
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            restore()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            trace_values = read(traced)
            print(
                json.dumps(
                    dict(
                        name=name,
                        trace_pcc=[H.pcc(a, b) for a, b in zip(trace_values, outputs[name][1])],
                        trace_maxdiff=[
                            float((a.float() - b.float()).abs().max()) for a, b in zip(trace_values, outputs[name][1])
                        ],
                    )
                ),
                flush=True,
            )
            restore()
            repeated = decode()
            repeat_values = read(repeated)
            ttnn.deallocate(repeated)
            assert all(torch.equal(a, b) for a, b in zip(repeat_values, outputs[name][1])), "restored eager differs"
            print(
                json.dumps(
                    dict(
                        name=name,
                        repeat_pcc=[H.pcc(a, b) for a, b in zip(repeat_values, outputs[name][1])],
                        repeat_maxdiff=[
                            float((a.float() - b.float()).abs().max()) for a, b in zip(repeat_values, outputs[name][1])
                        ],
                    )
                ),
                flush=True,
            )
            if not all(torch.equal(a, b) for a, b in zip(trace_values, outputs[name][1])):
                import datetime
                from pathlib import Path

                stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
                artifact = Path(__file__).resolve().parents[1] / f"doc/multichip_decoder/trace_mismatch_{stamp}.pt"
                torch.save(dict(eager=outputs[name][1], trace=trace_values, repeated=repeat_values), artifact)
                print(json.dumps(dict(trace_mismatch_artifact=str(artifact))), flush=True)
            assert all(
                torch.equal(a, b) for a, b in zip(trace_values, outputs[name][1])
            ), "restored trace differs from eager"
            restore()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            after_eager_trace = read(traced)
            assert all(
                torch.equal(a, b) for a, b in zip(after_eager_trace, outputs[name][1])
            ), "restored replay after eager differs"
            print(json.dumps(dict(name=name, trace_after_eager_exact=True)), flush=True)
            durations = []
            for repeat in range(5):
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                for _ in range(32):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                durations.append((time.perf_counter() - start) * 1000 / 32)
            print(json.dumps(dict(name=name, decode_ms=statistics.median(durations), trace_exact=True)), flush=True)
            ttnn.release_trace(mesh, trace)
            ttnn.deallocate(traced)
            del decoder
        finally:
            ttnn.close_mesh_device(mesh)
    for mode, index in [("prefill", 0), ("decode", 1)]:
        scores = [H.pcc(outputs["baseline"][index][0], out) for out in outputs["tp4"][index]]
        print(json.dumps(dict(mode=mode, pcc=scores)), flush=True)
        assert min(scores) >= 0.995
    state_scores = []
    for index, (single, shards) in enumerate(zip(outputs["baseline"][2], outputs["tp4"][2])):
        if args.layer % 4 == 3 or index == 0:
            combined = torch.cat(shards, dim=1)
        else:
            fields = [part.split((512, 512, 1024), dim=-1) for part in shards]
            combined = torch.cat([torch.cat([part[f] for part in fields], dim=-1) for f in range(3)], dim=-1)
        value = H.pcc(single[0], combined)
        state_scores.append(value)
        print(json.dumps(dict(state_index=index, state_pcc=value)), flush=True)
        assert torch.isfinite(combined.float()).all()
        assert value >= 0.99, "head-local state/cache partition mismatch"
    print(json.dumps(dict(state_pcc=state_scores, local_state_contract=True)), flush=True)


if __name__ == "__main__":
    main()
