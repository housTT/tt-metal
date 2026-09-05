# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TP4 same-producer cache precision and exact physical-row diagnostic.

This intentionally reads device tensors to CPU; it is not a performance path.
All original output/state acceptance bars remain enforced, including 0.99
between the matched BFP4/BFP8 cache states. A precision failure therefore
leaves a complete JSON diagnosis and exits nonzero.
"""

import argparse
import gc
import hashlib
import json
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn

from ..tt.functional_decoder import num_blocks_for_context
from ..tt.multichip_decoder import fabric_router_config
from . import test_functional_decoder as H
from .optimized_multichip_candidates import CANDIDATES
from .test_optimization_experiments import recorded_activations


def read_ranks(value):
    parts = [ttnn.to_torch(part) for part in ttnn.get_device_tensors(value)]
    assert len(parts) == 4, "the diagnostic requires real TP4 tensors"
    return parts


def digests(values):
    return [hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for value in values]


def exact(reference, actual):
    return [
        dict(rank=rank, exact=torch.equal(a, b), mismatch_elements=int(torch.count_nonzero(a != b)))
        for rank, (a, b) in enumerate(zip(reference, actual))
    ]


def logical_rows(cache, table, user, length):
    heads, block, width = cache.shape[1:]
    return cache[table[user].long()].permute(1, 0, 2, 3).reshape(heads, -1, width)[:, :length]


def run_policy(mesh, decoder, table, table_host, prefix, token, golden, dtype, position):
    decoder.allocate_kv_cache(table_host.numel(), dtype=dtype)
    caches = {"k": decoder.k_cache, "v": decoder.v_cache}
    cache_names = {id(value): key for key, value in caches.items()}
    expected_fill = {key: read_ranks(value) for key, value in caches.items()}
    assert all(torch.count_nonzero(part) == 0 for values in expected_fill.values() for part in values)
    report = dict(
        dtype=str(dtype),
        cache_shape=list(decoder.k_cache.shape),
        cache_memory=str(decoder.k_cache.memory_config()),
        page_block_size=decoder.page_block_size,
        producer_digests=[],
        fill_calls=[],
        sdpa=[],
    )
    pending_casts, update_controls = {}, {}
    original_cast = ttnn.typecast
    original_fill = ttnn.experimental.paged_fill_cache
    original_fused = ttnn.experimental.paged_fused_update_cache
    original_sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode

    def cast(value, target_dtype, *args, **kwargs):
        result = original_cast(value, target_dtype, *args, **kwargs)
        if target_dtype == dtype and len(value.shape) == 4 and value.shape[1] == 1 and value.shape[-1] == 256:
            # The fill wrapper consumes this record only for its actual input.
            pending_casts[id(result)] = digests(read_ranks(value))
        return result

    def fill(cache, value, page_table, **kwargs):
        kind = cache_names[id(cache)]
        raw_digests = pending_casts.pop(id(value))
        report["producer_digests"].append(dict(phase="prefill", kind=kind, ranks=raw_digests))
        sources = read_ranks(value)
        pages = read_ranks(page_table)
        assert all(torch.equal(pages[0], part) for part in pages[1:])
        users = read_ranks(kwargs["batch_idx_tensor"])[0].flatten().long().tolist()
        block = decoder.page_block_size
        for rank, source in enumerate(sources):
            for source_user, cache_user in enumerate(users):
                for page, start in enumerate(range(0, source.shape[2], block)):
                    count = min(block, source.shape[2] - start)
                    physical = int(pages[0][cache_user, page])
                    expected_fill[kind][rank][physical, :, :count] = source[source_user, :, start : start + count]
        report["fill_calls"].append(dict(kind=kind, shape=list(value.shape), pages=list(page_table.shape)))
        return original_fill(cache, value, page_table, **kwargs)

    def update(k_cache, k, v_cache, v, **kwargs):
        for kind, cache, value in (("k", k_cache, k), ("v", v_cache, v)):
            report["producer_digests"].append(dict(phase="decode", kind=kind, ranks=digests(read_ranks(value))))
            before = read_ranks(cache)
            scratch = ttnn.clone(cache, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            try:
                ttnn.experimental.paged_update_cache(scratch, value, **kwargs)
                independently_updated = read_ranks(scratch)
            finally:
                ttnn.deallocate(scratch)
            expected = [part.clone() for part in before]
            for user in range(token.shape[0]):
                physical = int(table_host[user, position // decoder.page_block_size])
                offset = position % decoder.page_block_size
                for rank in range(4):
                    expected[rank][physical, :, offset] = independently_updated[rank][physical, :, offset]
            update_controls[kind] = dict(expected=expected, independently_updated=independently_updated)
        return original_fused(k_cache, k, v_cache, v, **kwargs)

    def sdpa(q, k, v, **kwargs):
        result = original_sdpa(q, k, v, **kwargs)
        queries, keys, values, actuals = (read_ranks(value) for value in (q, k, v, result))
        for rank, (query, key, value, actual) in enumerate(zip(queries, keys, values, actuals)):
            users = []
            for user in range(token.shape[0]):
                group = query.shape[2] // key.shape[1]
                kr = logical_rows(key, table_host, user, position + 1).float().repeat_interleave(group, dim=0)
                vr = logical_rows(value, table_host, user, position + 1).float().repeat_interleave(group, dim=0)
                scores = (query[0, user].float().unsqueeze(1) @ kr.transpose(-1, -2)) * decoder.cfg.head_dim**-0.5
                oracle = (scores.softmax(dim=-1) @ vr).squeeze(1)
                users.append(H.pcc(oracle, actual[0, user]))
            report["sdpa"].append(dict(rank=rank, per_user_pcc=users, passed=min(users) >= 0.999))
        return result

    current_pos, rot_idxs = H.decode_inputs(mesh, torch.full((token.shape[0],), position))
    x, d = H.to_device(mesh, prefix), H.to_device(mesh, token)
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(ttnn, "typecast", cast))
            stack.enter_context(patch.object(ttnn.experimental, "paged_fill_cache", fill))
            stack.enter_context(patch.object(ttnn.experimental, "paged_fused_update_cache", update))
            stack.enter_context(patch.object(ttnn.transformer, "paged_scaled_dot_product_attention_decode", sdpa))
            out = decoder.prefill_forward(x, page_table=table)
            prefill_outputs = read_ranks(out)
            ttnn.deallocate(out)
            report["prefill_exact_rows"] = {
                key: exact(expected_fill[key], read_ranks(value)) for key, value in caches.items()
            }
            out = decoder.decode_forward(d, current_pos=current_pos, rot_idxs=rot_idxs, page_table=table)
            decode_outputs = read_ranks(out)
            ttnn.deallocate(out)
        state = {key: read_ranks(value) for key, value in caches.items()}
        report["decode_exact_rows"] = {
            key: dict(
                mapped_row_only=exact(update_controls[key]["expected"], values),
                unfused_control=exact(update_controls[key]["independently_updated"], values),
            )
            for key, values in state.items()
        }
        report["output"] = {}
        for name, reference, outputs in zip(("prefill", "decode"), golden, (prefill_outputs, decode_outputs)):
            scores = [[H.pcc(reference[user], value[user]) for user in range(token.shape[0])] for value in outputs]
            report["output"][name] = dict(
                all_rank_user_pcc=scores,
                replicated_exact=all(torch.equal(outputs[0], value) for value in outputs[1:]),
                passed=min(min(row) for row in scores) >= H.PCC_BAR,
            )
        return report, state
    finally:
        for value in (x, d, current_pos, rot_idxs, *caches.values()):
            ttnn.deallocate(value)
        decoder.k_cache = decoder.v_cache = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["production_cache4", "production_qkv4_cache4"], required=True)
    parser.add_argument("--batch", type=int, choices=[1, 4, 32], default=1)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--permuted", action="store_true")
    parser.add_argument(
        "--legacy-batch-inputs", action="store_true", help="exact earlier cache-regression B32 distribution"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite evidence: {args.output}")
    if not 1 <= args.length <= 4095:
        parser.error("length must leave a decode token within the 4096-token diagnostic capacity")
    if args.legacy_batch_inputs and (args.batch != 32 or args.length != 96):
        parser.error("legacy inputs require --batch 32 --length 96")
    torch.set_num_threads(8)
    source = recorded_activations(H.FULL_LAYER)[0]
    offsets = torch.arange(args.batch)[:, None] * 137
    prefix_indices = offsets + torch.arange(args.length)[None, :] + (39 if args.legacy_batch_inputs else 0)
    token_indices = offsets + (131 if args.legacy_batch_inputs else args.length)
    prefix, token = source[prefix_indices % len(source)].clone(), source[token_indices % len(source)].clone()
    reference = H.reference_layer(H.FULL_LAYER, "real")
    with torch.no_grad():
        golden_prefix, reference_cache = H.R.reference_prefill(reference, H.hf_config(), prefix.float(), start_pos=0)
        golden_decode = H.R.reference_decode(
            reference, H.hf_config(), token.float(), torch.full((args.batch,), args.length), reference_cache
        )
    del reference_cache, reference
    gc.collect()
    document = dict(
        status="running",
        hardware="four Blackhole chips on physical P300c boards",
        mesh=[1, 4],
        variant=args.variant,
        batch=args.batch,
        logical_length=args.length,
        permuted=args.permuted,
        legacy_batch_inputs=args.legacy_batch_inputs,
        thresholds=dict(output=H.PCC_BAR, state=0.99, sdpa_oracle=0.999),
        input_sha256=digests([prefix, token]),
        policies=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(document, indent=2) + "\n")

    save()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, router_config=fabric_router_config())
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=24576)
    try:
        context = max(1024, args.length + 1)
        decoder = CANDIDATES[args.variant].from_state_dict(
            H.layer_state_dict(H.FULL_LAYER, "real"),
            hf_config=H.hf_config(),
            layer_idx=H.FULL_LAYER,
            mesh_device=mesh,
            max_context=context,
        )
        decoder.allocate_state(args.batch)
        blocks = num_blocks_for_context(context)
        table_host = torch.arange(args.batch * blocks, dtype=torch.int32)
        if args.permuted:
            table_host = torch.randperm(args.batch * blocks, generator=torch.Generator().manual_seed(819)).to(
                torch.int32
            )
        table_host = table_host.reshape(args.batch, blocks)
        table = H.to_device(mesh, table_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        document.update(
            mesh_config=asdict(decoder.mesh_config), policy=asdict(decoder.policy), page_table=table_host.tolist()
        )
        states = []
        for dtype in (ttnn.bfloat4_b, ttnn.bfloat8_b):
            report, state = run_policy(
                mesh, decoder, table, table_host, prefix, token, (golden_prefix, golden_decode), dtype, args.length
            )
            document["policies"].append(report)
            states.append(state)
            save()
        document["identical_producers"] = (
            document["policies"][0]["producer_digests"] == document["policies"][1]["producer_digests"]
        )
        document["state_precision"] = {}
        for kind in ("k", "v"):
            low, high = (state[kind] for state in states)
            per_rank_user = [
                [
                    H.pcc(
                        logical_rows(a, table_host, user, args.length + 1),
                        logical_rows(b, table_host, user, args.length + 1),
                    )
                    for user in range(args.batch)
                ]
                for a, b in zip(low, high)
            ]
            score = H.pcc(torch.cat(low, dim=1), torch.cat(high, dim=1))
            document["state_precision"][kind] = dict(
                full_cache_pcc=score,
                used_rows_per_rank_user_pcc=per_rank_user,
                passed=score >= 0.99,
            )
        implementation_checks = [document["identical_producers"]]
        output_checks = []
        for policy in document["policies"]:
            implementation_checks.extend(
                row["exact"] for ranks in policy["prefill_exact_rows"].values() for row in ranks
            )
            implementation_checks.extend(
                row["exact"]
                for groups in policy["decode_exact_rows"].values()
                for ranks in groups.values()
                for row in ranks
            )
            implementation_checks.extend(row["passed"] for row in policy["sdpa"])
            output_checks.extend(row["passed"] and row["replicated_exact"] for row in policy["output"].values())
        document["cache_implementation_checks_passed"] = all(implementation_checks)
        document["output_accuracy_gate_passed"] = all(output_checks)
        document["state_precision_gate_passed"] = all(row["passed"] for row in document["state_precision"].values())
        document["status"] = (
            "passed"
            if all(implementation_checks + output_checks) and document["state_precision_gate_passed"]
            else "failed"
        )
        save()
        print("CACHE_PRECISION_DIAGNOSTIC " + json.dumps(document), flush=True)
        assert document["status"] == "passed", f"unchanged cache diagnostic gate failed; see {args.output}"
    except Exception as exc:
        document["status"] = "failed"
        document["exception"] = f"{type(exc).__name__}: {exc}"
        save()
        raise
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
