# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Native long-context execution while the bounded prefill trace family remains resident."""

import argparse
import json
import time
from pathlib import Path

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import OrnithGenerator
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import OrnithModel, close_ornith_mesh, open_ornith_mesh

DOC = Path(__file__).resolve().parent


def memory(mesh):
    result = {}
    for name in ("DRAM", "L1", "TRACE"):
        view = ttnn.get_memory_view(mesh, getattr(ttnn.BufferType, name))
        result[name] = {
            key: int(getattr(view, key))
            for key in (
                "num_banks",
                "total_bytes_per_bank",
                "total_bytes_allocated_per_bank",
                "total_bytes_free_per_bank",
                "largest_contiguous_bytes_free_per_bank",
            )
        }
        result[name]["total_allocated_bytes_per_device"] = view.num_banks * view.total_bytes_allocated_per_bank
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision-config")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = dict(native_context=262144, batch=1, mesh=[1, 4], capability_reduction=None, measurements={})
    mesh = open_ornith_mesh()
    try:
        report["measurements"]["opened"] = memory(mesh)
        model = OrnithModel(None, mesh, precision_config=args.precision_config)
        report["precision"] = model.precision
        report["measurements"]["loaded_weights_constants"] = memory(mesh)
        gen = OrnithGenerator(model)
        from .run_candidate import runtime_summary

        report["runtime"] = runtime_summary(gen)
        report["measurements"]["allocated_full_native_cache"] = memory(mesh)
        try:
            for trace_length in (131, 2048):
                gen.generate([100] * trace_length, 2, stop_on_eos=False)
                assert gen._prefill_trace is not None
                report["measurements"][f"resident_prefill_trace_{trace_length}"] = memory(mesh)
            report["resident_prefill_key"] = list(gen._prefill_key)
            report["prefill_counters"] = dict(gen.counters)
            report["measurements"]["traces_sampler_ready"] = memory(mesh)
            report["windows"] = []
            for length in (262143, 262144):
                gen.reset()
                prompt = [100] * length
                start = time.perf_counter()
                logits = gen.prefill_forward(
                    [prompt],
                    page_table=gen.page_table,
                    kv_cache=gen.kv_cache,
                    prompt_lens=[length],
                    return_device_logits=True,
                )
                gen._sample_device(logits)
                token = int(gen._read_tokens()[0])
                ttnn.deallocate(logits)
                elapsed = time.perf_counter() - start
                item = dict(logical_prompt_length=length, returned_token=token, prefill_s=elapsed)
                if length < model.max_context:
                    gen._ensure_replay_safe()
                    gen._write_positions([length])
                    gen._refresh_table(gen.page_table)
                    output = gen.decode_forward(None, None, page_table=gen.page_table, kv_cache=gen.kv_cache)
                    position = ttnn.to_torch(ttnn.get_device_tensors(gen._inputs[1])[0]).item()
                    assert position == 262144
                    item.update(last_decode_position=length, decode_token=int(output[0]), advanced_position=position)
                report["windows"].append(item)
                report["measurements"][f"after_prefill_{length}"] = memory(mesh)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print("CONTEXT_WINDOW_OK", item, flush=True)
            assert gen._prefill_trace is not None
            report["pass"] = True
        finally:
            gen.teardown()
    finally:
        close_ornith_mesh(mesh)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
