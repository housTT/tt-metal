# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Standard tt-metal performance benchmark for the DeepSeek-V4 build.

Uses tt-metal's canonical perf harness — `BenchmarkProfiler` + `BenchmarkData`
(the CI benchmark JSON) and `prep_perf_report` (the perf CSV) — and the
`models_performance_bare_metal` marker, exactly like the other models' perf tests.
Measures decode tok/s and prefill TTFT of the resident+sharded+traced decode engine
on the 4-chip Blackhole mesh and asserts against the targets (>=5 tok/s, <5s TTFT).

Run:
    pytest models/demos/deepseek_v4/tests/test_perf.py -m models_performance_bare_metal -s
"""
import pytest

from models.demos.deepseek_v4.demo.decode_engine import measure
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.perf_utils import prep_perf_report

TARGET_TOKS = 5.0  # decode tok/s target
TARGET_TTFT_S = 5.0  # time-to-first-token target


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("seq", [128, 2048])
def test_deepseek_v4_perf(seq):
    profiler = BenchmarkProfiler()
    profiler.start("run")
    with profiler("inference", iteration=0):
        m = measure(layers=43, seq=seq, topk=6, iters=20)
    profiler.end("run")

    decode_toks = m["decode_toks"]
    decode_s = m["decode_ms"] / 1000.0
    ttft_s = m["ttft_s"]
    print(
        f"\n[perf] DeepSeek-V4-Flash seq={seq}: decode {decode_toks:.2f} tok/s "
        f"({decode_s*1000:.1f} ms/token), TTFT {ttft_s*1000:.1f} ms, {m['num_devices']}x Blackhole"
    )

    # canonical perf CSV (models/perf) — treat per-token decode as the inference time
    prep_perf_report(
        model_name=f"DeepSeek-V4-Flash_decode_seq{seq}",
        batch_size=1,
        inference_and_compile_time=m["compile_s"] + decode_s,
        inference_time=decode_s,
        expected_compile_time=180.0,
        expected_inference_time=1.0 / TARGET_TOKS,  # 0.2 s == 5 tok/s
        comments=f"resident+sharded+traced_4xBH_seq{seq}",
    )

    # CI benchmark JSON (no-op off-CI, but the standard reporting path)
    bench = BenchmarkData()
    bench.add_measurement(profiler, 0, "inference", "decode_t/s/u", decode_toks)
    bench.add_measurement(profiler, 0, "inference", "prefill_time_to_first_token_ms", ttft_s * 1000)
    bench.save_partial_run_json(
        profiler,
        run_type="deepseek_v4_decode",
        ml_model_name="DeepSeek-V4-Flash",
        ml_model_type="text-generation",
        num_layers=43,
        batch_size=1,
        input_sequence_length=seq,
        precision="bf16-resident/bf4-experts",
    )

    # target assertions (this is what models_performance tests enforce)
    assert decode_toks >= TARGET_TOKS, f"decode {decode_toks:.2f} tok/s < target {TARGET_TOKS}"
    assert ttft_s < TARGET_TTFT_S, f"TTFT {ttft_s:.2f}s >= target {TARGET_TTFT_S}s"
