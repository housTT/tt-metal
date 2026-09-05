# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Preserve every user/step of the exact real-input TP4 trace PCC contract.

Run only on the serialized device lane. This is an accuracy diagnostic, with
host readback and optional component substitutions, not a performance path.
"""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn

from ..tt.functional_decoder import num_blocks_for_context
from ..tt.multichip_decoder import (
    MeshConfig,
    MultichipDecoder,
    _projection_weights,
    fabric_router_config,
    partition_state_dict,
)
from . import test_functional_decoder as H
from .multichip_cache_diagnostic import digests, read_ranks
from .test_optimization_experiments import recorded_activations


def scores(reference, values, user_ids):
    rows = []
    for rank, value in enumerate(values):
        finite = bool(torch.isfinite(value).all())
        per_user = [H.pcc(reference[index], value[index]) for index in range(len(user_ids))]
        aggregate = H.pcc(reference, value)
        rows.append(
            dict(
                rank=rank,
                finite=finite,
                pcc=aggregate,
                users=[dict(user=user, pcc=pcc) for user, pcc in zip(user_ids, per_user)],
                passed=finite and min(per_user) >= H.PCC_BAR and aggregate > H.PCC_BAR,
            )
        )
    return dict(
        ranks=rows,
        replicated_exact=all(torch.equal(values[0], value) for value in values[1:]),
        passed=all(row["passed"] for row in rows),
    )


def exact_rank_outputs(reference, values):
    return [
        dict(rank=rank, exact=torch.equal(a, b), maxdiff=float((a.float() - b.float()).abs().max()))
        for rank, (a, b) in enumerate(zip(reference, values))
    ]


def install_projection_control(decoder, state, args, document):
    """Change only selected packed output fields; retain a raw-weight oracle."""
    cfg = H.hf_config()
    local = SimpleNamespace(
        num_attention_heads=decoder.cfg.n_heads, head_dim=decoder.cfg.head_dim, hidden_size=decoder.cfg.dim
    )
    weights = [_projection_weights(partition_state_dict(state, cfg, rank), local)["qkvg"] for rank in range(4)]
    qw = decoder.cfg.n_heads * decoder.cfg.head_dim
    kw = decoder.cfg.n_kv_heads * decoder.cfg.head_dim
    fields = dict(q=(0, qw), k=(qw, qw + kw), v=(qw + kw, qw + 2 * kw), gate=(qw + 2 * kw, 2 * qw + 2 * kw))
    restored = set(args.restore_fields.split(",")) if args.restore_fields else set()
    high = None
    if restored:
        high = ttnn.from_torch(
            torch.cat(weights, dim=1).contiguous(),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=decoder.device,
            memory_config=decoder.decode_weights["qkvg"].memory_config(),
            mesh_mapper=ttnn.ShardTensorToMesh(decoder.device, dim=1),
        )
    original = decoder._linear
    observer = dict(enabled=False, step=None)

    def linear(x, role, **kwargs):
        low = original(x, role, **kwargs)
        if role != "qkvg" or x.shape[1] != 1:
            return low
        result = low
        if restored:
            original_weight = decoder.decode_weights[role]
            decoder.decode_weights[role] = high
            try:
                upper = original(x, role, **kwargs)
            finally:
                decoder.decode_weights[role] = original_weight
            batch = x.shape[0]
            pieces = [
                ttnn.slice(
                    upper if field in restored else low,
                    [0, 0, start],
                    [batch, 1, end],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                for field, (start, end) in fields.items()
            ]
            result = ttnn.concat(pieces, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
            for value in (*pieces, low, upper):
                ttnn.deallocate(value)
        if observer["enabled"]:
            inputs, outputs = read_ranks(x), read_ranks(result)
            rows = []
            for rank, (value, actual, weight) in enumerate(zip(inputs, outputs, weights)):
                golden = value.float() @ weight.float()
                rows.append(
                    dict(
                        rank=rank,
                        fields={
                            field: [
                                H.pcc(golden[user, :, start:end], actual[user, :, start:end])
                                for user in range(value.shape[0])
                            ]
                            for field, (start, end) in fields.items()
                        },
                    )
                )
            document.setdefault("projection_boundaries", []).append(dict(step=observer["step"], ranks=rows))
        return result

    decoder._linear = linear
    return observer, high


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=[1, 4, 32], default=32)
    parser.add_argument("--users", help="original B32 user IDs, e.g. 8 or 8,26; overrides --batch")
    parser.add_argument("--qkv-dtype", choices=["bfloat4_b", "bfloat8_b", "bfloat16"], default="bfloat4_b")
    parser.add_argument("--qkv-cores", type=int, default=8)
    parser.add_argument("--qkv-block", type=int, default=16)
    parser.add_argument("--qkv-readers", type=int, default=2)
    parser.add_argument("--qkv-fidelity", choices=["LoFi", "HiFi2", "HiFi4"], default="LoFi")
    parser.add_argument("--qkv-fp32", action="store_true")
    parser.add_argument(
        "--restore-fields", default="", help="comma-separated QKVG fields restored from BFP8: q,k,v,gate"
    )
    parser.add_argument("--boundaries", action="store_true", help="additional eager raw-Torch projection comparison")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".pt").exists():
        parser.error("refusing to overwrite diagnostic evidence")
    user_ids = list(map(int, args.users.split(","))) if args.users else list(range(args.batch))
    if not user_ids or len(set(user_ids)) != len(user_ids) or not all(0 <= user < 32 for user in user_ids):
        parser.error("users must be distinct original batch IDs in [0,31]")
    if args.restore_fields and not set(args.restore_fields.split(",")) <= {"q", "k", "v", "gate"}:
        parser.error("unknown projection field")
    torch.set_num_threads(8)
    source = recorded_activations(H.FULL_LAYER)[0]
    offsets = torch.tensor(user_ids)[:, None] * 137
    prefix = source[(offsets + torch.arange(63)[None, :] + 31) % len(source)].clone()
    tokens = [source[(offsets + 3100 + step) % len(source)].clone() for step in range(3)]
    golden_prefill, golden = H.run_reference(H.FULL_LAYER, "real", prefix, decode_x=tokens, decode_steps=3)
    raw = dict(user_ids=user_ids, prefix=prefix, tokens=tokens, golden_prefill=golden_prefill, golden=golden)
    document = dict(
        status="running",
        hardware="four Blackhole chips on physical P300c boards",
        mesh=[1, 4],
        batch=len(user_ids),
        user_ids=user_ids,
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        input_sha256=digests([prefix, *tokens]),
        output_gate=H.PCC_BAR,
        phases={},
        snapshot_digests={},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(document, indent=2) + "\n")

    save()
    plan = MeshConfig()
    roles = dict(plan.local.role_configs)
    roles["qkvg"] = dict(cores=args.qkv_cores, block_w=args.qkv_block, readers=args.qkv_readers)
    plan = replace(plan, decode_qkvg_dtype=args.qkv_dtype, local=replace(plan.local, role_configs=roles))
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, router_config=fabric_router_config())
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 * 1024 * 1024, l1_small_size=24576)
    trace_id = None
    try:
        state = H.layer_state_dict(H.FULL_LAYER, "real")
        decoder = MultichipDecoder.from_state_dict(
            state, hf_config=H.hf_config(), layer_idx=H.FULL_LAYER, mesh_device=mesh, mesh_config=plan, max_context=1024
        )
        blocks = num_blocks_for_context(1024)
        decoder.allocate_kv_cache(len(user_ids) * blocks)
        decoder.allocate_state(len(user_ids))
        table_host = torch.arange(len(user_ids) * blocks, dtype=torch.int32).reshape(len(user_ids), blocks)
        table = H.to_device(mesh, table_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        prefix_tt = H.to_device(mesh, prefix)
        prefill_out = decoder.prefill_forward(prefix_tt, page_table=table)
        raw["prefill"] = read_ranks(prefill_out)
        document["prefill"] = scores(golden_prefill, raw["prefill"], user_ids)
        ttnn.deallocate(prefill_out)
        ttnn.deallocate(prefix_tt)
        # Apply this compute control only after identical prefill producers exist.
        decoder.projection_compute["qkvg"] = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, args.qkv_fidelity),
            math_approx_mode=False,
            fp32_dest_acc_en=args.qkv_fp32,
            packer_l1_acc=True,
        )
        observer, high_weight = install_projection_control(decoder, state, args, document)
        caches = [decoder.k_cache, decoder.v_cache]
        saved = [ttnn.clone(value) for value in caches]

        def snapshot_hash():
            return [digests(read_ranks(value)) for value in saved]

        def restore():
            for value, cache in zip(saved, caches):
                ttnn.copy(value, cache)

        document["snapshot_digests"]["prefill"] = snapshot_hash()
        document.update(
            mesh_config=asdict(plan),
            policy=asdict(decoder.policy),
            qkv_prefill_dtype=str(decoder.w["qkvg"].dtype),
            qkv_decode_dtype=str(decoder.decode_weights["qkvg"].dtype),
            qkv_decode_compute=str(decoder.projection_compute["qkvg"]),
            cache_dtype=[str(value.dtype) for value in caches],
            cache_shape=list(decoder.k_cache.shape),
            cache_memory=str(decoder.k_cache.memory_config()),
            page_table=table_host.tolist(),
            page_block_size=decoder.page_block_size,
            local_heads=dict(q=decoder.cfg.n_heads, kv=decoder.cfg.n_kv_heads, dim=decoder.cfg.head_dim),
        )
        x_buf = H.to_device(mesh, tokens[0])
        pos_buf, rot_buf = H.decode_inputs(mesh, torch.full((len(user_ids),), 63))

        def forward():
            return decoder.decode_forward(x_buf, current_pos=pos_buf, rot_idxs=rot_buf, page_table=table)

        def refresh(step):
            position = torch.full((len(user_ids),), 63 + step, dtype=torch.int32)
            for value, destination, dtype, layout in (
                (tokens[step], x_buf, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                (position, pos_buf, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                (position.reshape(1, -1), rot_buf, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            ):
                ttnn.copy_host_to_device_tensor(ttnn.from_torch(value, dtype=dtype, layout=layout), destination)

        ttnn.deallocate(forward())
        ttnn.synchronize_device(mesh)
        document["snapshot_digests"]["warm"] = snapshot_hash()
        restore()
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        trace_out = forward()
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        document["snapshot_digests"]["capture"] = snapshot_hash()
        restore()
        for phase in ("trace", "eager", "trace_after_eager"):
            if phase != "trace":
                restore()
            rows, outputs = [], []
            for step in range(3):
                refresh(step)
                if phase == "eager":
                    out = forward()
                else:
                    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
                    out = trace_out
                ttnn.synchronize_device(mesh)
                values = read_ranks(out)
                outputs.append(values)
                row = dict(
                    step=step,
                    position=63 + step,
                    page=(63 + step) // decoder.page_block_size,
                    tile=(63 + step) // 32,
                    sdpa_chunk=(63 + step) // decoder.optimization.sdpa_chunk,
                    **scores(golden[step], values, user_ids),
                )
                if phase != "trace":
                    row["versus_trace"] = exact_rank_outputs(raw["trace"][step], values)
                rows.append(row)
                document["phases"][phase] = rows
                print(json.dumps(dict(phase=phase, **row)), flush=True)
                save()
                if phase == "eager":
                    ttnn.deallocate(out)
            raw[phase] = outputs
            document["snapshot_digests"][phase] = snapshot_hash()
            document.setdefault("final_cache_digests", {})[phase] = [digests(read_ranks(value)) for value in caches]
            save()
        if args.boundaries:
            restore()
            observer["enabled"] = True
            for step in range(3):
                observer["step"] = step
                refresh(step)
                ttnn.deallocate(forward())
            observer["enabled"] = False
        document["snapshot_unchanged"] = all(
            value == document["snapshot_digests"]["prefill"] for value in document["snapshot_digests"].values()
        )
        document["eager_trace_exact"] = all(
            rank["exact"]
            for phase in ("eager", "trace_after_eager")
            for row in document["phases"][phase]
            for rank in row["versus_trace"]
        )
        document["final_cache_exact"] = all(
            value == document["final_cache_digests"]["trace"] for value in document["final_cache_digests"].values()
        )
        failed = [
            dict(phase=phase, step=row["step"], position=row["position"], rank=rank["rank"], **user)
            for phase, rows in document["phases"].items()
            for row in rows
            for rank in row["ranks"]
            for user in rank["users"]
            if not rank["finite"] or not user["pcc"] >= H.PCC_BAR
        ]
        document["failed_users"] = failed
        document["status"] = (
            "passed"
            if not failed
            and document["snapshot_unchanged"]
            and document["eager_trace_exact"]
            and document["final_cache_exact"]
            and all(row["passed"] for rows in document["phases"].values() for row in rows)
            and all(row["replicated_exact"] for rows in document["phases"].values() for row in rows)
            else "failed"
        )
        torch.save(raw, args.output.with_suffix(".pt"))
        save()
        assert document["status"] == "passed", f"complete diagnostic evidence: {args.output}; failures={failed}"
    except BaseException as error:
        document["exception"] = repr(error)
        document["status"] = "failed"
        torch.save(raw, args.output.with_suffix(".pt"))
        save()
        raise
    finally:
        if trace_id is not None:
            ttnn.release_trace(mesh, trace_id)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
