# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reduced full-path collector and no-read API parity with an explicit readback control."""
import argparse
import json
from pathlib import Path

from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import close_ornith_mesh, open_ornith_mesh
from models.common.sampling import SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
mesh = open_ornith_mesh()
report = dict(checks=[])
try:
    gen = build_generator(root, mesh, layer_indices=[0, 3], cache_context=2048)
    try:
        rendered = gen.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Count from one to ten."}], tokenize=False, add_generation_prompt=True
        )
        prompt = gen.tokenizer.encode(rendered, add_special_tokens=False)
        cases = [
            (8, None),
            (128, None),
            (260, None),
            (128, SamplingParams(temperature=0.8, top_k=20, top_p=0.95, seed=1234)),
            (8, None),
        ]
        if args.quick:
            cases = [(8, None), (8, cases[3][1]), (8, None)]
        for count, params in cases:
            print(f"PARITY_START count={count} sampled={params is not None}", flush=True)
            collected = gen.generate(prompt, count, sampling_params=params, stop_on_eos=False)
            perf = dict(gen.perf)
            control = gen.generate(
                prompt, count, sampling_params=params, stop_on_eos=False, next_input=lambda step, token: token
            )
            assert collected == control, "history and per-token readback control differ"
            assert len(collected) == count
            report["checks"].append(
                dict(count=count, sampled=params is not None, exact_readback_control=True, tokens=collected, perf=perf)
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"PARITY_PASS count={count} sampled={params is not None}", flush=True)
        first = gen.generate(prompt, 1, stop_on_eos=False)
        result = gen.decode_forward(
            None, None, page_table=gen.page_table, kv_cache=gen.kv_cache, read_from_device=False
        )
        assert result is gen._inputs[0]
        before = dict(gen.counters)
        result = gen.replay_decode(6)
        assert result is gen._inputs[0]
        counters = {k: gen.counters[k] - before[k] for k in before}
        for name in [
            "token_refreshes",
            "position_refreshes",
            "rope_refreshes",
            "page_table_refreshes",
            "readbacks",
            "read_waits",
            "synchronizations",
        ]:
            assert counters[name] == 0, counters
        last = gen._read_tokens()[0].item()
        control = gen.generate(prompt, 8, stop_on_eos=False, next_input=lambda step, token: token)
        assert first[0] == control[0] and last == control[-1]
        report["device_output_api"] = dict(exact_control=True, same_feedback_tensor=True, loop_counters=counters)
        report["pass"] = True
    finally:
        gen.teardown()
finally:
    close_ornith_mesh(mesh)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
