# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reduced real-model uninstrumented prefill and exact prepared-body trace control."""

import argparse
import hashlib
import json
import statistics
import time
import traceback
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import close_ornith_mesh, open_ornith_mesh

DOC = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 131])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--watcher", action="store_true")
    args = parser.parse_args()
    report = dict(
        layers=[0, 3],
        batch=1,
        context=262144,
        profiler_enabled=False,
        public_window_ms=[],
        cases=[],
        probe_execution_pass=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def sha(tensor):
        return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    mesh = None
    gen = None
    trace = None
    try:
        report["phase"] = "open"
        save()
        mesh = open_ornith_mesh()
        gen = build_generator(DOC.parents[1], mesh, layer_indices=[0, 3], cache_context=None)
        model, cache = gen.model, gen.kv_cache
        assert cache.context == 262144 and cache.batch_size == 1
        gen.generate([100] * 128, 2)
        gen.generate([100] * 128, 2)
        ttnn.synchronize_device(mesh)

        def public(tokens, table):
            logits = gen.prefill_forward(
                [tokens],
                page_table=table,
                kv_cache=cache,
                prompt_lens=[len(tokens)],
                return_device_logits=True,
            )
            gen._sample_device(logits)
            return logits

        for _ in range(args.trials):
            ttnn.synchronize_device(mesh)
            start = time.perf_counter()
            logits = public([100] * 128, gen.page_table)
            ttnn.synchronize_device(mesh)
            report["public_window_ms"].append((time.perf_counter() - start) * 1000)
            ttnn.deallocate(logits)
        report["public_window_median_ms"] = statistics.median(report["public_window_ms"])
        report["recorded_profiled_window_ms"] = (
            json.loads((DOC / "profile_prefill_host.json").read_text())["elapsed_s"] * 1000
        )
        print("PUBLIC_UNINSTRUMENTED", json.dumps(report["public_window_ms"]), flush=True)
        # The prepared prefill experiment owns one trace at a time. Never replay
        # a decode trace whose workspace predates new prefill allocations.
        gen._release_traces()

        def snapshot(logits, length, table):
            scores = model.logits_to_host(logits, 1)
            states = []
            for layer in cache.prefill_layers:
                if not layer.is_full_attention:
                    for buffer in [layer.recurrent_state] + layer.conv_state:
                        states.extend(sha(ttnn.to_torch(t)) for t in ttnn.get_device_tensors(buffer))
            sampled = gen._read_tokens().tolist()
            # One next-token decode checks the populated KV and hybrid-state
            # contract, including nonaligned prompts and permuted physical pages.
            gen._write_positions([length])
            gen._refresh_table(table)
            decoded = gen._forward()
            next_scores = model.logits_to_host(decoded, 1)
            ttnn.deallocate(decoded)
            return dict(scores=scores, state_hashes=states, sampled=sampled, next_scores=next_scores)

        for length in args.lengths:
            assert 1 <= length <= model.prefill_chunk
            item = dict(length=length, start_pos=0, trials=[], checks=[], phase="prepare")
            report["cases"].append(item)
            save()
            variants = [
                ([100] * length, gen.page_table.clone()),
                ([101 + index % 7 for index in range(length)], gen.page_table.flip(1).contiguous()),
            ]
            ids = model.upload(
                torch.tensor(variants[0][0], dtype=torch.int32).reshape(1, -1),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            pt = model.upload(variants[0][1], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

            def prepare(tokens, table):
                for layer in cache.prefill_layers:
                    layer.reset_state()
                for value, target in (
                    (torch.tensor(tokens, dtype=torch.int32).reshape(1, -1), ids),
                    (table.to(torch.int32), pt),
                ):
                    host = model.upload(value, dtype=target.dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=False)
                    ttnn.copy_host_to_device_tensor(host, target)
                gen._prepare_prompt_sampling([tokens], [length], [0], None)

            def body():
                x = model.embed(ids)
                for layer in cache.prefill_layers:
                    nxt = layer.prefill_forward(x, start_pos=0, page_table=pt)
                    ttnn.deallocate(x)
                    x = nxt
                hidden = ttnn.slice(x, [0, length - 1, 0], [1, length, model.dim])
                last = ttnn.clone(hidden)
                ttnn.deallocate(x)
                logits = model.terminal(last)
                gen._sample_device(logits)
                return logits

            references = []
            for tokens, table in variants:
                # Exact public model path plus the same generator preparation,
                # without rebuilding decode traces already intentionally released.
                original = model.prefill_forward([tokens], page_table=table, kv_cache=cache, prompt_lens=[length])
                gen._prepare_prompt_sampling([tokens], [length], [0], None)
                gen._sample_device(original)
                references.append(snapshot(original, length, table))
                ttnn.deallocate(original)
                prepare(tokens, table)
                eager = body()
                actual = snapshot(eager, length, table)
                expected = references[-1]
                assert torch.equal(actual["scores"], expected["scores"])
                assert actual["state_hashes"] == expected["state_hashes"]
                assert actual["sampled"] == expected["sampled"]
                assert torch.equal(actual["next_scores"], expected["next_scores"])
                ttnn.deallocate(eager)
            item["changed_tokens_change_logits"] = not torch.equal(references[0]["scores"], references[1]["scores"])
            assert item["changed_tokens_change_logits"]
            prepare(*variants[0])
            warm = body()
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(warm)
            prepare(*variants[0])
            ttnn.synchronize_device(mesh)
            item["phase"] = "capture"
            save()
            start = time.perf_counter()
            print("PREFILL_CAPTURE_BEGIN", length, flush=True)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            traced = body()
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            item["capture_ms"] = (time.perf_counter() - start) * 1000
            print("PREFILL_CAPTURE_END", length, flush=True)
            item["phase"] = "verify"
            save()
            for index in [0, 1, 0]:
                prepare(*variants[index])
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                actual = snapshot(traced, length, variants[index][1])
                expected = references[index]
                check = dict(
                    variant=index,
                    prefill_exact=bool(torch.equal(actual["scores"], expected["scores"])),
                    state_exact=actual["state_hashes"] == expected["state_hashes"],
                    token_exact=actual["sampled"] == expected["sampled"],
                    next_decode_exact=bool(torch.equal(actual["next_scores"], expected["next_scores"])),
                    scores_sha256=sha(actual["scores"]),
                )
                item["checks"].append(check)
                print("PREFILL_TRACE_CHECK", length, json.dumps(check), flush=True)
                assert all(check[k] for k in ("prefill_exact", "state_exact", "token_exact", "next_decode_exact"))
            item["phase"] = "timing"
            save()
            for trial in range(args.trials):
                # Watcher mode exercises only the established trace; normal
                # measurements alternate same-body eager and replay submissions.
                order = ["traced"] if args.watcher else (["eager", "traced"] if trial % 2 == 0 else ["traced", "eager"])
                for mode in order:
                    ttnn.synchronize_device(mesh)
                    start = time.perf_counter()
                    prepare(*variants[trial % 2])
                    prepared_at = time.perf_counter()
                    if mode == "eager":
                        output = body()
                    else:
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        output = traced
                    submitted_at = time.perf_counter()
                    ttnn.synchronize_device(mesh)
                    finished = time.perf_counter()
                    row = dict(
                        trial=trial,
                        mode=mode,
                        prepare_host_ms=(prepared_at - start) * 1000,
                        submit_host_ms=(submitted_at - prepared_at) * 1000,
                        synchronized_ms=(finished - start) * 1000,
                    )
                    item["trials"].append(row)
                    assert torch.equal(model.logits_to_host(output, 1), references[trial % 2]["scores"])
                    if mode == "eager":
                        ttnn.deallocate(output)
                    print("PREFILL_TIMING", length, json.dumps(row), flush=True)
            item["medians"] = {
                mode: {
                    key: statistics.median(row[key] for row in item["trials"] if row["mode"] == mode)
                    for key in ("prepare_host_ms", "submit_host_ms", "synchronized_ms")
                }
                for mode in sorted(set(row["mode"] for row in item["trials"]))
            }
            view = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
            item["trace_bytes_per_bank"] = int(view.total_bytes_allocated_per_bank)
            ttnn.release_trace(mesh, trace)
            trace = None
            ttnn.deallocate(traced)
            ttnn.deallocate(ids)
            ttnn.deallocate(pt)
            item["phase"] = "complete"
            save()
        report["phase"] = "complete"
        report["probe_execution_pass"] = True
    except Exception as error:
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        if mesh is not None:
            if trace is not None:
                ttnn.release_trace(mesh, trace)
            if gen is not None:
                gen.teardown()
            close_ornith_mesh(mesh)
        save()
    print("PREFILL_GAPS_OK", args.output, flush=True)


if __name__ == "__main__":
    main()
