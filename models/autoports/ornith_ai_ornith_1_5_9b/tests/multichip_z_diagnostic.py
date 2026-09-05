# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real-input TP4 Z-gating localization; run on the serialized device lane."""

import argparse
import hashlib
import json
import math
from dataclasses import asdict

import torch

import ttnn
from models.demos.gpt_oss.tt.ccl import CCLManager as LegacyCCLManager

from ..tt import multichip_decoder as mesh_impl
from ..tt.functional_decoder import num_blocks_for_context
from ..tt.multichip_decoder import MeshConfig, fabric_router_config
from . import test_functional_decoder as H
from .optimized_multichip_candidates import CANDIDATES, PackedGDNFusedZ
from .test_optimization_experiments import recorded_activations


def metadata(tensor):
    return dict(
        shape=list(tensor.shape),
        padded_shape=list(tensor.padded_shape),
        dtype=str(tensor.dtype),
        layout=str(tensor.layout),
        memory_config=str(tensor.memory_config()),
    )


def value_stats(value):
    flat = value.float().flatten()
    finite = bool(flat.isfinite().all())
    result = dict(finite=finite, nonfinite=int((~flat.isfinite()).sum()), nonzero=int(flat.count_nonzero()))
    if finite:
        result.update(
            minimum=float(flat.min()),
            maximum=float(flat.max()),
            std=float(flat.double().std(unbiased=False)),
            constant=bool((flat == flat[0]).all()),
        )
    return result


def comparison(target, value):
    actual_stats, target_stats = value_stats(value), value_stats(target)
    finite = actual_stats["finite"] and target_stats["finite"]
    exact = finite and torch.equal(target, value)
    defined = finite and not (actual_stats["constant"] or target_stats["constant"])
    pcc = H.pcc(target, value) if finite else None
    passed = exact or (defined and pcc >= 0.995)
    return dict(
        pcc=pcc,
        pcc_defined=defined,
        maxdiff=float((target - value).abs().max()) if finite else None,
        exact=exact,
        passed=passed,
        accepted_by="exact" if exact else ("pcc" if passed else None),
        actual=actual_stats,
        reference=target_stats,
    )


def read_report(name, tensor, reference=None, user_axis=0):
    parts = ttnn.get_device_tensors(tensor)
    values = [ttnn.to_torch(part).float() for part in parts]
    rows = []
    for rank, (part, value) in enumerate(zip(parts, values)):
        finite = bool(value.isfinite().all())
        row = dict(rank=rank, **metadata(part), finite=finite, nonfinite=int((~value.isfinite()).sum()))
        if finite:
            row.update(minimum=float(value.min()), maximum=float(value.max()))
        if reference is not None:
            target = reference[rank].float()
            row.update(comparison(target, value))
            users = [
                comparison(target.select(user_axis, user), value.select(user_axis, user))
                for user in range(value.shape[user_axis])
            ]
            row.update(
                user_pcc=[user["pcc"] for user in users],
                user_maxdiff=[user["maxdiff"] for user in users],
                user_nonfinite=[user["actual"]["nonfinite"] for user in users],
                user_exact=[user["exact"] for user in users],
                user_pass=[user["passed"] for user in users],
                user_pcc_defined=[user["pcc_defined"] for user in users],
                users=users,
            )
        else:
            row["user_stats"] = [value_stats(value.select(user_axis, user)) for user in range(value.shape[user_axis])]
        rows.append(row)
    print(json.dumps(dict(name=name, tensor=metadata(tensor), ranks=rows)), flush=True)
    return values, rows


def device_digests(tensor):
    """Hash logical tensor bytes returned by TTNN; unused padding is excluded."""
    return [
        hashlib.sha256(ttnn.to_torch(part).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        for part in ttnn.get_device_tensors(tensor)
    ]


class CaptureBoundary:
    """Keep device snapshots while executing the selected decoder boundary."""

    def _gdn_out_head_major(self, core, z, batch, seq):
        if seq == 1:
            self.boundary_metadata = dict(core=metadata(core), z=metadata(z), batch=batch, seq=seq)
            self.boundary_core = ttnn.clone(core, memory_config=core.memory_config())
            self.boundary_z = ttnn.clone(z, memory_config=z.memory_config())
        return super()._gdn_out_head_major(core, z, batch, seq)


class CaptureZ(CaptureBoundary, PackedGDNFusedZ):
    pass


def reference_user31(values, layer):
    """Run HF for the affected real window; the accelerator still runs all users."""
    reference = H.reference_layer(layer, "real")
    config = H.hf_config()
    window = values[31:32].float()
    with torch.no_grad():
        prefill, cache = H.R.reference_prefill(reference, config, window[:, :-1])
        del prefill
        boundary = {}

        def capture_norm(module, inputs):
            boundary["core"] = inputs[0].detach().clone()
            boundary["raw_z"] = inputs[1].detach().clone()

        hook = reference.linear_attn.norm.register_forward_pre_hook(capture_norm)
        try:
            output = H.R.reference_decode(reference, config, window[:, -1:], torch.tensor([window.shape[1] - 1]), cache)
        finally:
            hook.remove()
        boundary["norm_weight"] = reference.linear_attn.norm.weight.detach().clone()
    assert set(boundary) == {"core", "raw_z", "norm_weight"}, "HF norm boundary hook did not execute"
    return output, boundary


def distinct_user_offsets(source, batch, length):
    """Advance each real window until its decode row differs from earlier users."""
    offsets, decode_rows = [], []
    for user in range(batch):
        for advance in range(source.shape[0]):
            offset = (user * 137 + advance) % source.shape[0]
            row = source[(offset + length) % source.shape[0]]
            if all(not torch.equal(row, earlier) for earlier in decode_rows):
                offsets.append(offset)
                decode_rows.append(row)
                break
        else:
            raise ValueError(f"recorded source has fewer than {batch} distinct decode rows")
    return torch.tensor(offsets, dtype=torch.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--restored-repeats", type=int, default=1)
    parser.add_argument("--legacy-semaphore-grid", action="store_true")
    parser.add_argument(
        "--hf-user31", action="store_true", help="Check current production on the exact B32/T2048 workload"
    )
    parser.add_argument(
        "--hf-variant",
        choices=sorted(k for k in CANDIDATES if k.startswith("production_")),
        default="production_candidate",
    )
    args = parser.parse_args()
    if args.layer % 4 == 3:
        parser.error("Z gating requires a linear-attention layer")
    if not 1 <= args.batch <= 32:
        parser.error("batch must be within the decoder's supported 1–32 users")
    if not 0 <= args.restored_repeats <= 3:
        parser.error("restored repeats must be between 0 and 3")
    if args.hf_user31 and (args.layer, args.batch, args.length) != (0, 32, 2048):
        parser.error("--hf-user31 requires layer 0, batch 32, and length 2048")
    torch.set_num_threads(8)
    state_dict, config = H.layer_state_dict(args.layer, "real"), H.hf_config()
    recorded = recorded_activations(args.layer)
    if not 1 < args.length < min(recorded.shape[1], 4096):
        parser.error("length must leave one recorded decode token and fit the 4096-token probe capacity")
    source = recorded[0]
    offsets = distinct_user_offsets(source, args.batch, args.length)
    indices = (torch.arange(args.length + 1)[None, :] + offsets[:, None]) % source.shape[0]
    values = source[indices].clone()
    assert torch.unique(indices[:, 0]).numel() == args.batch, "recorded user offsets collided"
    assert torch.unique(values[:, -1], dim=0).shape[0] == args.batch, "duplicated real decode users"
    decoder_class = CaptureZ
    hf_output, hf_boundary = None, None
    if args.hf_user31:
        target = CANDIDATES[args.hf_variant]

        class CaptureProduction(CaptureBoundary, target):
            pass

        decoder_class = CaptureProduction
        print(
            json.dumps(dict(name="hf_user31_start", source_offset=int(offsets[31]), decode_row=int(indices[31, -1]))),
            flush=True,
        )
        hf_output, hf_boundary = reference_user31(values, args.layer)
        print(json.dumps(dict(name="hf_user31_reference_complete", output=value_stats(hf_output))), flush=True)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, router_config=fabric_router_config())
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=24576)
    try:
        plan = MeshConfig()
        manager_class = mesh_impl.MeshCCLManager
        if args.legacy_semaphore_grid:
            mesh_impl.MeshCCLManager = LegacyCCLManager
        try:
            decoder = decoder_class.from_state_dict(
                state_dict,
                hf_config=config,
                layer_idx=args.layer,
                mesh_device=mesh,
                max_context=4096,
                mesh_config=plan,
            )
        finally:
            mesh_impl.MeshCCLManager = manager_class
        print(
            json.dumps(
                dict(
                    name="contract",
                    layer=args.layer,
                    length=args.length,
                    batch=args.batch,
                    restored_repeats=args.restored_repeats,
                    legacy_semaphore_grid=args.legacy_semaphore_grid,
                    semaphore_grid=str(decoder.ccl.ccl_cores),
                    ccl_manager_class=type(decoder.ccl).__name__,
                    decoder_variant=args.hf_variant if args.hf_user31 else "packed_gdn_fused_z",
                    hf_reference_users=[31] if args.hf_user31 else [],
                    user_requested_source_offsets=[(user * 137) % source.shape[0] for user in range(args.batch)],
                    user_source_offsets=offsets.tolist(),
                    user_decode_source_rows=indices[:, -1].tolist(),
                    user_decode_input_stats=[value_stats(value[-1]) for value in values],
                    user_positions=[args.length] * args.batch,
                    mesh_config=asdict(decoder.mesh_config),
                    precision_policy=asdict(decoder.policy),
                    weights={role: str(decoder.w[role].dtype) for role in decoder.projection_compute},
                    compute_grid=str(mesh.compute_with_storage_grid_size()),
                    raw_tensor_artifacts=[],
                )
            ),
            flush=True,
        )

        def upload(value):
            return ttnn.from_torch(
                value,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        prompt = upload(values[:, : args.length])
        token = upload(values[:, args.length : args.length + 1])
        blocks = num_blocks_for_context(4096)
        table = H.to_device(
            mesh,
            torch.arange(args.batch * blocks, dtype=torch.int32).reshape(args.batch, -1),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        positions, rotations = H.decode_inputs(mesh, torch.full((args.batch,), args.length, dtype=torch.int32))
        decoder.allocate_state(args.batch)
        decoder.allocate_kv_cache(args.batch * blocks)
        print(
            json.dumps(
                dict(
                    name="allocated_state",
                    batch_size=decoder.batch_size,
                    recurrent_state=metadata(decoder.recurrent_state),
                    conv_state=[metadata(t) for t in decoder.conv_state],
                    recurrent_l1_intermediates=decoder.recurrent_l1_intermediates,
                    page_table=metadata(table),
                    positions=metadata(positions),
                    rotations=metadata(rotations),
                )
            ),
            flush=True,
        )
        decoder.reset_state()
        prefill = decoder.prefill_forward(prompt, page_table=table)
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(prefill)
        state_buffers = [decoder.recurrent_state, *decoder.conv_state]
        state_snapshots = (
            [ttnn.clone(t, memory_config=t.memory_config()) for t in state_buffers] if args.restored_repeats else []
        )
        state_digests = [device_digests(t) for t in state_buffers] if state_snapshots else []
        if state_snapshots:
            assert [device_digests(t) for t in state_snapshots] == state_digests, "state snapshot differs from prefill"
        decoded = decoder.decode_forward(token, page_table=table, current_pos=positions, rot_idxs=rotations)
        ttnn.synchronize_device(mesh)
        hf_rows = None
        if args.hf_user31:
            selected = ttnn.slice(decoded, [31, 0, 0], [32, 1, decoder.cfg.dim])
            _, hf_rows = read_report("hf_user31_layer_decode", selected, [hf_output] * 4)
            ttnn.deallocate(selected)
        else:
            read_report("original_fused_layer_output", decoded)
        ttnn.deallocate(decoded)
        print(json.dumps(dict(name="original_boundary", **decoder.boundary_metadata)), flush=True)

        core, z = decoder.boundary_core, decoder.boundary_z
        core_host, _ = read_report("captured_core", core)
        z_host, _ = read_report("captured_raw_z", z)
        normed = ttnn.rms_norm(core, weight=decoder.w["gdn_norm"], epsilon=decoder.cfg.norm_eps)
        heads = ttnn.reshape(
            normed, [args.batch, decoder.cfg.linear_num_value_heads, 1, decoder.cfg.linear_value_head_dim]
        )
        combined = ttnn.permute(heads, (0, 2, 1, 3))
        public_shape = [args.batch, 1, decoder.cfg.linear_v_dim]
        merged = ttnn.reshape(combined, public_shape)
        merged_host, _ = read_report("merged", merged)
        assert merged.dtype == ttnn.float32 and z.dtype == ttnn.bfloat16, "diagnosed dtype contract changed"
        assert list(merged.shape) == list(z.shape) == public_shape
        if args.hf_user31:
            rows = []
            local_heads = decoder.cfg.linear_num_value_heads
            dim = decoder.cfg.linear_value_head_dim
            assert hf_boundary["core"].shape == (4 * local_heads, dim)
            assert hf_boundary["raw_z"].shape == (4 * local_heads, dim)
            for rank, (actual_core, actual_z, actual_merged) in enumerate(zip(core_host, z_host, merged_host)):
                head_slice = slice(rank * local_heads, (rank + 1) * local_heads)
                expected_core = hf_boundary["core"][head_slice].reshape(1, local_heads, 1, dim)
                expected_z = hf_boundary["raw_z"][head_slice].reshape(1, 1, -1)
                expected_norm = expected_core * torch.rsqrt(
                    expected_core.square().mean(dim=-1, keepdim=True) + decoder.cfg.norm_eps
                )
                expected_norm = expected_norm * hf_boundary["norm_weight"].reshape(1, 1, 1, dim)
                expected_merged = expected_norm.permute(0, 2, 1, 3).reshape(1, 1, -1)
                rows.append(
                    dict(
                        rank=rank,
                        core=comparison(expected_core, actual_core[31:32]),
                        raw_z=comparison(expected_z, actual_z[31:32]),
                        merged=comparison(expected_merged, actual_merged[31:32]),
                    )
                )
            print(json.dumps(dict(name="hf_user31_boundary", ranks=rows)), flush=True)

        repeat_checks = []
        for repeat in range(args.restored_repeats):
            immutable = [device_digests(t) for t in state_snapshots] == state_digests
            assert immutable, "prefill state snapshot changed after decode"
            for snapshot, target in zip(state_snapshots, state_buffers):
                ttnn.copy(snapshot, target)
            restored = [device_digests(t) for t in state_buffers] == state_digests
            print(
                json.dumps(
                    dict(
                        name="state_restore",
                        repeat=repeat,
                        snapshot_immutable=immutable,
                        exact=restored,
                        digest_extent="logical_tensor_bytes",
                    )
                ),
                flush=True,
            )
            assert restored, "state restore differs from the original prefill state"
            repeated = decoder.decode_forward(token, page_table=table, current_pos=positions, rot_idxs=rotations)
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(repeated)
            repeat_core, repeat_z = decoder.boundary_core, decoder.boundary_z
            _, core_rows = read_report(f"restored_{repeat}_core_vs_first", repeat_core, core_host)
            _, z_rows = read_report(f"restored_{repeat}_raw_z_vs_first", repeat_z, z_host)
            repeat_norm = ttnn.rms_norm(repeat_core, weight=decoder.w["gdn_norm"], epsilon=decoder.cfg.norm_eps)
            repeat_heads = ttnn.reshape(
                repeat_norm, [args.batch, decoder.cfg.linear_num_value_heads, 1, decoder.cfg.linear_value_head_dim]
            )
            repeat_combined = ttnn.permute(repeat_heads, (0, 2, 1, 3))
            repeat_merged = ttnn.reshape(repeat_combined, public_shape)
            _, merged_rows = read_report(f"restored_{repeat}_merged_vs_first", repeat_merged, merged_host)
            captured_immutable = True
            for name, tensor, host in [
                ("core", core, core_host),
                ("raw_z", z, z_host),
                ("merged", merged, merged_host),
            ]:
                _, rows = read_report(f"restored_{repeat}_first_{name}_immutable", tensor, host)
                captured_immutable = captured_immutable and all(row["exact"] for row in rows)
            repeat_state_immutable = [device_digests(t) for t in state_snapshots] == state_digests
            check = dict(
                repeat=repeat,
                core_exact=all(row["exact"] for row in core_rows),
                z_exact=all(row["exact"] for row in z_rows),
                merged_exact=all(row["exact"] for row in merged_rows),
                captured_inputs_immutable=captured_immutable,
                state_snapshot_immutable=repeat_state_immutable,
            )
            repeat_checks.append(check)
            print(json.dumps(dict(name="restored_boundary_check", **check)), flush=True)
            for tensor in (repeat_combined, repeat_norm, repeat_core, repeat_z):
                ttnn.deallocate(tensor)
            assert all(
                value for key, value in check.items() if key != "repeat"
            ), "restored eager boundary changed; inspect per-user core/Z/merged evidence before trusting gate controls"
        for snapshot in state_snapshots:
            ttnn.deallocate(snapshot)

        if args.hf_user31:
            passed = all(row["passed"] and all(row["user_pass"]) for row in hf_rows)
            print(
                json.dumps(
                    dict(
                        name="hf_user31_complete",
                        passed=passed,
                        upstream_correctness_checked_users=[31],
                        accelerator_batch=args.batch,
                        prefill_length=args.length,
                        restored_boundary_checks=repeat_checks,
                        raw_tensor_artifacts=[],
                    )
                ),
                flush=True,
            )
            assert passed, "exact B32/T2048 user31 whole-layer decode failed HF comparison"
            return

        oracle = [a * torch.nn.functional.silu(b).bfloat16().float() for a, b in zip(merged_host, z_host)]
        activated = ttnn.silu(z, memory_config=z.memory_config())
        read_report("standalone_silu", activated, [torch.nn.functional.silu(b).bfloat16().float() for b in z_host])
        separated = ttnn.multiply(merged, activated)
        separated_host, control_rows = read_report("separate_gate_vs_cpu", separated, oracle)
        assert all(row["finite"] and all(row["user_pass"]) for row in control_rows), (
            "separate gate control failed for a logical user: expected finite exact equality "
            "or a defined PCC >= 0.995"
        )
        projected = decoder._linear(separated, "gdn_out")
        projected_host, _ = read_report("separate_gdn_out", projected)
        ttnn.deallocate(projected)
        ttnn.deallocate(separated)
        ttnn.deallocate(activated)

        z_fp32 = ttnn.typecast(z, ttnn.float32)
        read_report("raw_z_fp32", z_fp32, z_host)
        compact_shape = [1, args.batch, decoder.cfg.linear_v_dim]
        z_compact = ttnn.reshape(z, compact_shape)
        merged_compact = ttnn.reshape(merged, compact_shape)
        for name, compact, host in [
            ("compact_raw_z", z_compact, z_host),
            ("compact_merged", merged_compact, merged_host),
        ]:
            _, rows = read_report(name, compact, [t.reshape(compact_shape) for t in host], user_axis=1)
            assert all(row["exact"] for row in rows), "compacting changed logical user values or order"
        print(
            json.dumps(
                dict(
                    name="gate_work_counts",
                    public_tiles=math.prod(merged.padded_shape) // 1024,
                    compact_tiles=math.prod(merged_compact.padded_shape) // 1024,
                    worker_cores=mesh.compute_with_storage_grid_size().x * mesh.compute_with_storage_grid_size().y,
                )
            ),
            flush=True,
        )
        variants = [
            ("fused_rhs_bf16", merged, z, dict(input_tensor_b_activations=[ttnn.UnaryOpType.SILU]), False),
            ("fused_rhs_fp32", merged, z_fp32, dict(input_tensor_b_activations=[ttnn.UnaryOpType.SILU]), False),
            ("fused_lhs_bf16", z, merged, dict(input_tensor_a_activations=[ttnn.UnaryOpType.SILU]), False),
            (
                "compact_fused_lhs_bf16",
                z_compact,
                merged_compact,
                dict(input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
                True,
            ),
        ]
        summary = {}
        for name, lhs, rhs, activations, compact in variants:
            gated = ttnn.multiply(lhs, rhs, dtype=ttnn.float32, memory_config=merged.memory_config(), **activations)
            if compact:
                compact_gated = gated
                compact_values, _ = read_report(
                    name + "_compact_gate_vs_separate",
                    compact_gated,
                    [t.reshape(compact_shape) for t in separated_host],
                    user_axis=1,
                )
                gated = ttnn.reshape(compact_gated, public_shape)
                restored_values, _ = read_report(
                    name + "_restored_users", gated, [t.reshape(public_shape) for t in compact_values]
                )
                assert all(
                    torch.allclose(actual, expected.reshape(public_shape), rtol=0, atol=0, equal_nan=True)
                    for actual, expected in zip(restored_values, compact_values)
                ), "restoring changed logical user values or order"
                if gated.buffer_address() != compact_gated.buffer_address():
                    ttnn.deallocate(compact_gated)
            _, gate_rows = read_report(name + "_gate_vs_separate", gated, separated_host)
            out = decoder._linear(gated, "gdn_out")
            _, out_rows = read_report(name + "_gdn_out_vs_separate", out, projected_host)
            summary[name] = dict(
                gate_pcc=[row["pcc"] for row in gate_rows],
                output_pcc=[row["pcc"] for row in out_rows],
                all_finite=all(row["finite"] for row in gate_rows + out_rows),
                gate_user_pcc=[row["user_pcc"] for row in gate_rows],
                output_user_pcc=[row["user_pcc"] for row in out_rows],
                gate_user_pass=[row["user_pass"] for row in gate_rows],
                output_user_pass=[row["user_pass"] for row in out_rows],
                constant_gate_reference_users=[
                    [user for user, result in enumerate(row["users"]) if result["reference"].get("constant", False)]
                    for row in gate_rows
                ],
            )
            ttnn.deallocate(out)
            ttnn.deallocate(gated)
        ttnn.synchronize_device(mesh)
        print(
            json.dumps(
                dict(
                    name="diagnostic_complete",
                    comparisons=summary,
                    restored_boundary_checks=repeat_checks,
                    upstream_correctness_checked=False,
                )
            ),
            flush=True,
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
