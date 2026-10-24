# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-level probe of the Sampling1D top-k sampler as the generator configures it.

Builds logits in which every user has a disjoint, known candidate set, samples many times at the
requested k / temperature, and reports out-of-vocabulary ids, ids outside the user's own candidate
set, and the rank histogram of the sampled candidates per user. No model is loaded.

Usage:
    TT_VISIBLE_DEVICES=1 python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_sampling1d_topk \
        --output <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import DEVICE_MAX_TOP_K
from models.autoports.google_gemma_4_26b_a4b_it.tt.model import DECODE_SLOT_COUNT
from models.common.modules.sampling.sampling_1d import Sampling1D

VOCAB = 262144


def _row_tensor(mesh: ttnn.MeshDevice, values: torch.Tensor, dtype: Any) -> ttnn.Tensor:
    return ttnn.from_torch(
        values,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, None), mesh_shape=tuple(mesh.shape)),
    )


def _case(
    mesh: ttnn.MeshDevice,
    sampler: Sampling1D,
    *,
    k: int,
    temperature: float,
    top_p: float,
    slope: float,
    calls: int,
    base: int,
    users: int,
    dirty_l1: bool = False,
) -> dict[str, Any]:
    logits = torch.full((1, 1, DECODE_SLOT_COUNT, VOCAB), -20.0, dtype=torch.bfloat16)
    sets = []
    for user in range(DECODE_SLOT_COUNT):
        start = base + user * k
        ids = list(range(start, start + k))
        sets.append(ids)
        logits[0, 0, user, start : start + k] = torch.tensor([8.0 - slope * j for j in range(k)], dtype=torch.bfloat16)
    tt_logits = ttnn.from_torch(
        logits,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    k_t = _row_tensor(mesh, torch.full((DECODE_SLOT_COUNT,), k, dtype=torch.int32), ttnn.uint32)
    p_t = _row_tensor(mesh, torch.full((DECODE_SLOT_COUNT,), top_p, dtype=torch.bfloat16), ttnn.bfloat16)
    temp_t = _row_tensor(mesh, torch.full((DECODE_SLOT_COUNT,), 1.0 / temperature, dtype=torch.bfloat16), ttnn.bfloat16)
    out = ttnn.from_torch(
        torch.zeros((1, 1, 1, DECODE_SLOT_COUNT), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    invalid: list[tuple[int, int, int]] = []
    foreign: list[tuple[int, int, int]] = []
    ranks = [[0] * k for _ in range(DECODE_SLOT_COUNT)]
    dirty_a = dirty_b = None
    if dirty_l1:
        dirty_a = ttnn.from_torch(
            torch.randn(1, 1, 2048, 2048, dtype=torch.bfloat16) * 1000.0,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        dirty_b = ttnn.from_torch(
            torch.randn(1, 1, 2048, 2048, dtype=torch.bfloat16) * 1000.0,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
    for call in range(calls):
        if dirty_l1:
            product = ttnn.matmul(dirty_a, dirty_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(product)
        seeds = _row_tensor(mesh, torch.arange(DECODE_SLOT_COUNT, dtype=torch.int64) + 1000 * (call + 1), ttnn.uint32)
        sampled, _ = sampler.decode_forward(tt_logits, k=k_t, p=p_t, temp=temp_t, seeds=seeds, tt_out_tok=out)
        host = ttnn.to_torch(ttnn.get_device_tensors(sampled)[0]).reshape(-1)[:DECODE_SLOT_COUNT]
        for user in range(users):
            token = int(host[user].item())
            signed = token - (1 << 32) if token >= (1 << 31) else token
            if token < 0 or token >= VOCAB:
                invalid.append((call, user, signed))
                continue
            if sets[user][0] <= token < sets[user][0] + k:
                ranks[user][token - sets[user][0]] += 1
            else:
                foreign.append((call, user, token))
        ttnn.deallocate(seeds)
    samples = calls * users
    per_user = {
        str(user): {
            "invalid": sum(1 for _, u, _ in invalid if u == user),
            "foreign": sum(1 for _, u, _ in foreign if u == user),
            "rank_mass_first_16": sum(ranks[user][:16]) / max(1, sum(ranks[user])),
            "rank_mass_first_32": sum(ranks[user][:32]) / max(1, sum(ranks[user])),
            "max_rank_hit": max((r for r, c in enumerate(ranks[user]) if c), default=None),
        }
        for user in range(users)
    }
    for tensor in (tt_logits, k_t, p_t, temp_t, out, dirty_a, dirty_b):
        if tensor is not None:
            ttnn.deallocate(tensor)
    return {
        "dirty_l1": dirty_l1,
        "k": k,
        "temperature": temperature,
        "top_p": top_p,
        "logit_slope": slope,
        "calls": calls,
        "users": users,
        "samples": samples,
        "invalid_count": len(invalid),
        "foreign_count": len(foreign),
        "invalid_examples": invalid[:12],
        "foreign_examples": foreign[:12],
        "per_user": per_user,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument(
        "--dirty-l1",
        action="store_true",
        help="run a large random matmul before every sampling call so the cores' L1 holds nonzero data "
        "beyond the sampler's buffers; cases are then k=64 and k=32 at temperature 5",
    )
    args = parser.parse_args()
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        sampler = Sampling1D(
            vocab_size=VOCAB,
            mesh_device=mesh,
            max_batch_size=DECODE_SLOT_COUNT,
            max_top_k=DEVICE_MAX_TOP_K,
            allow_force_argmax=False,
            pad_to_power_of_2=True,
            num_gather_links=1,
            ag_topology=ttnn.Topology.Linear,
        )
        sampler.load_device_buffers()
        cases = []
        case_list = (
            ((64, 5.0, 1.0, 0.05), (32, 5.0, 1.0, 0.05), (64, 5.0, 1.0, 0.05), (32, 5.0, 1.0, 0.05))
            if args.dirty_l1
            else (
                (64, 5.0, 1.0, 0.05),
                (64, 1.0, 1.0, 0.05),
                (32, 5.0, 1.0, 0.05),
                (50, 5.0, 1.0, 0.05),
                (64, 5.0, 1.0, 0.5),
            )
        )
        for k, temperature, top_p, slope in case_list:
            case = _case(
                mesh,
                sampler,
                k=k,
                temperature=temperature,
                top_p=top_p,
                slope=slope,
                calls=args.calls,
                base=70000,
                users=DECODE_SLOT_COUNT,
                dirty_l1=args.dirty_l1,
            )
            cases.append(case)
            print(
                json.dumps(
                    {
                        key: case[key]
                        for key in (
                            "dirty_l1",
                            "k",
                            "temperature",
                            "logit_slope",
                            "samples",
                            "invalid_count",
                            "foreign_count",
                        )
                    }
                ),
                flush=True,
            )
            print(
                "   users 29..31:",
                {u: case["per_user"][str(u)] for u in (0, 15, 16, 29, 30, 31)},
                flush=True,
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"vocab": VOCAB, "max_top_k": DEVICE_MAX_TOP_K, "cases": cases}, indent=2) + "\n"
        )
        print(f"sampling1d probe -> {args.output}")
    finally:
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
