# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact UINT32 TP4 history probe; run only in the serialized hardware lane."""

import argparse
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import close_ornith_mesh, open_ornith_mesh

CAPACITY = 128
LANES = 32
MASK = (1 << 32) - 1
SENTINEL = 0xCAFEBA00


def host_tensor(value, dtype, mesh):
    # INT32 host representation preserves all UINT32 bits, including >2^31.
    return ttnn.from_torch(
        value.to(torch.int32).contiguous(),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def upload(value, dtype, mesh):
    return ttnn.to_device(host_tensor(value, dtype, mesh), mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def write(value, target, mesh):
    ttnn.copy_host_to_device_tensor(host_tensor(value, target.dtype, mesh), target)


def addresses(tensor):
    return [part.buffer_address() for part in ttnn.get_device_tensors(tensor)]


def append(history, cursor, tokens):
    updated = ttnn.indexed_fill(cursor, history, tokens, dim=0)
    copied = ttnn.copy(updated, history)
    assert addresses(copied) == addresses(history), "copy did not return the persistent output allocation"
    ttnn.plus_one(cursor)
    # Probe-only input producer: production reads the sampler's token feedback.
    ttnn.plus_one(tokens)
    return updated


def initial_values(offset=0):
    edges = [
        0,
        1,
        255,
        256,
        65535,
        65536,
        65537,
        248319,
        0x3FFFFFFF,
        0x40000000,
        0x7FFFFFFF,
        0x80000000,
        0xBABECAFE,
        0xFEDCBA00,
        0xFFFFFFF0,
        0xFFFFFFFF,
    ]
    values = torch.tensor(edges + list(reversed(edges)), dtype=torch.int64)
    return ((values + offset) & MASK).reshape(1, 1, 1, LANES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        command=sys.argv,
        hardware="four Blackhole chips on P300c boards",
        mesh=[1, 4],
        capacity=CAPACITY,
        native_batch="1..32; independent of the 32 physical history lanes",
        collector="indexed_fill + copy + plus_one(cursor)",
        probe_only_input_producer="plus_one(tokens)",
        timing_scope="collector plus probe input producer plus one final mesh transfer; no model",
        windows=[],
        status="running",
    )
    mesh = None
    trace = None
    scratch = None
    try:
        mesh = open_ornith_mesh()
        blank = torch.full((CAPACITY, 1, 1, LANES), SENTINEL, dtype=torch.int64)
        history = upload(blank, ttnn.uint32, mesh)
        cursor = upload(torch.tensor([3]), ttnn.int32, mesh)
        tokens = upload(initial_values(700), ttnn.uint32, mesh)
        stable = dict(history=addresses(history), cursor=addresses(cursor), tokens=addresses(tokens))
        assert all(len(value) == 4 for value in stable.values()), stable
        report["buffer_addresses"] = stable

        print("OUTPUT_HISTORY_WARM", flush=True)
        warm = append(history, cursor, tokens)
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(warm)
        # Capture-time values differ from both warm and replay. Tensor values must
        # remain runtime inputs; only addresses and specs belong in the trace.
        write(blank, history, mesh)
        write(torch.tensor([5]), cursor, mesh)
        write(initial_values(4000), tokens, mesh)
        ttnn.synchronize_device(mesh)
        programs = mesh.num_program_cache_entries()
        mesh.set_program_cache_misses_allowed(False)
        print("OUTPUT_HISTORY_CAPTURE", flush=True)
        try:
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            scratch = append(history, cursor, tokens)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
        finally:
            mesh.set_program_cache_misses_allowed(True)
        assert mesh.num_program_cache_entries() == programs, "capture compiled a new program"
        report["program_cache_entries"] = programs

        cases = [
            ("batch_one", 1, 0, [0]),
            ("sparse_fixed_slots", 7, 37, [0, 3, 17, 31]),
            ("full_capacity_batch32", CAPACITY, 0, list(range(LANES))),
            ("reset_a", 33, 991, [0, 2, 5, 17, 31]),
            ("reset_a_repeat", 33, 991, [0, 2, 5, 17, 31]),
        ]
        prior_reset = None
        for name, steps, offset, slots in cases:
            initial = initial_values(offset)
            write(blank, history, mesh)
            write(torch.tensor([0]), cursor, mesh)
            write(initial, tokens, mesh)
            # Request-boundary synchronization isolates timing from initialization.
            ttnn.synchronize_device(mesh)
            print(f"OUTPUT_HISTORY_REPLAY {name} steps={steps}", flush=True)
            start = time.perf_counter()
            with ExitStack() as guard:
                for operation in (
                    "from_torch",
                    "to_torch",
                    "copy_host_to_device_tensor",
                    "synchronize_device",
                    "event_synchronize",
                ):
                    guard.enter_context(
                        patch.object(ttnn, operation, side_effect=AssertionError(f"{operation} in replay loop"))
                    )
                for _ in range(steps):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            submit_s = time.perf_counter() - start
            # Exactly one final mesh read; all conversions below consume host shards.
            pending = history.cpu(blocking=False)
            event = ttnn.record_event(mesh, 0)
            ttnn.event_synchronize(event)
            elapsed_s = time.perf_counter() - start
            got = [ttnn.to_torch(part).to(torch.int64) & MASK for part in ttnn.get_device_tensors(pending)]
            assert len(got) == 4, f"expected every TP rank, got {len(got)}"
            expected = blank.clone()
            expected[:steps] = (initial + torch.arange(steps).reshape(steps, 1, 1, 1)) & MASK
            for rank, actual in enumerate(got):
                assert torch.equal(actual, expected), (
                    name,
                    rank,
                    "UINT32 history mismatch",
                    (actual != expected).nonzero()[:12].tolist(),
                )
                # The host selects fixed slots without compaction in the device path.
                selected = actual[:steps, 0, 0, slots].transpose(0, 1)
                assert torch.equal(selected, expected[:steps, 0, 0, slots].transpose(0, 1))
            if name == "reset_a":
                prior_reset = got
            elif name == "reset_a_repeat":
                assert all(torch.equal(old, new) for old, new in zip(prior_reset, got))
            assert addresses(history) == stable["history"]
            assert addresses(tokens) == stable["tokens"]
            assert addresses(cursor) == stable["cursor"]
            assert mesh.num_program_cache_entries() == programs, "replay created new programs"
            result = dict(
                name=name,
                steps=steps,
                active_fixed_slots=slots,
                checked_ranks=[0, 1, 2, 3],
                exact_uint32=True,
                untouched_tail=True,
                buffer_identity_preserved=True,
                loop_host_reads=0,
                loop_host_writes=0,
                loop_host_waits=0,
                final_mesh_reads=1,
                final_waits=1,
                trace_replays=steps,
                blocking_replay=False,
                submit_s=submit_s,
                completion_with_final_read_s=elapsed_s,
                amortized_completion_ms=elapsed_s * 1000 / steps,
            )
            report["windows"].append(result)
            print(json.dumps(result), flush=True)
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if mesh is not None:
            if trace is not None:
                ttnn.release_trace(mesh, trace)
            if scratch is not None:
                ttnn.deallocate(scratch)
            close_ornith_mesh(mesh)


if __name__ == "__main__":
    main()
