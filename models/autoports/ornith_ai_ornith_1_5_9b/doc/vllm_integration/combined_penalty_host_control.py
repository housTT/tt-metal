# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Source-execute pinned vLLM's actual Torch penalty path without TTNN imports."""

import argparse
import builtins
import hashlib
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-difference", action="store_true")
    args = parser.parse_args()
    test_path = Path(__file__).resolve().parents[2] / "tests/test_sampling_penalty_order.py"
    helpers = runpy.run_path(str(test_path))
    definitions = helpers["definitions"]
    ops_path = args.vllm_root / "vllm/_custom_ops.py"
    host_path = args.vllm_root / "vllm/model_executor/layers/utils.py"
    ops = {"torch": torch}
    definitions(ops_path, {"apply_repetition_penalties_torch", "apply_repetition_penalties"}, ops)

    def import_actual_ops(name, *positional, **kwargs):
        if name == "vllm._custom_ops":
            return SimpleNamespace(apply_repetition_penalties=ops["apply_repetition_penalties"])
        return builtins.__import__(name, *positional, **kwargs)

    host = {"torch": torch, "__builtins__": dict(vars(builtins), __import__=import_actual_ops)}
    definitions(host_path, {"get_token_bin_counts_and_mask", "apply_penalties"}, host)
    logits = torch.tensor([[4.0, -4.0, 0.5, -0.5, 0.0], [-0.5, 0.5, -4.0, 4.0, 0.5]])
    prompt_tokens = torch.tensor([[2, 3]] * 2)
    output_tokens = torch.tensor([[0, 1, 1, 1, 3, 3]] * 2)
    prompt_mask = torch.tensor([[False, False, True, True, False]] * 2)
    counts = torch.tensor([[1, 3, 0, 2, 0]] * 2)
    cases = []
    for p, f, r in (
        (0.0, 0.0, 1.0),
        (2.0, 0.0, 1.0),
        (-1.5, 0.0, 1.0),
        (0.0, 0.5, 1.0),
        (0.0, -0.5, 1.0),
        (0.0, 0.0, 2.0),
        (0.0, 0.0, 0.5),
        (2.0, 0.0, 2.0),
        (0.0, 0.5, 2.0),
        (2.0, 0.5, 2.0),
        (-1.5, -0.5, 2.0),
        (2.0, 0.5, 0.5),
    ):
        params = [torch.full((2,), value) for value in (p, f, r)]
        actual_host = host["apply_penalties"](logits.clone(), prompt_tokens, output_tokens, *params)
        reference = helpers["reference_penalties"](logits, prompt_mask, counts, *params)
        torch.testing.assert_close(actual_host, reference, rtol=0, atol=0)
        common = helpers["common_penalties"](logits, prompt_mask, counts, *params)
        cases.append(
            {
                "presence": p,
                "frequency": f,
                "repetition": r,
                "host_equals_reference": True,
                "common_equals_host": torch.equal(common, actual_host),
                "host_scores": actual_host.tolist(),
                "common_scores": common.tolist(),
            }
        )
    report = {
        "path": "actual vLLM utils.apply_penalties -> actual _custom_ops.apply_repetition_penalties -> actual Torch implementation",
        "scope": "CPU source execution, including prompt/generated bin counts; no TTNN or vLLM initialization",
        "host_source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ops_path, host_path)},
        "cases": cases,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    differs = any(not case["common_equals_host"] for case in cases)
    assert differs == args.expect_difference, report
    print(
        json.dumps(
            {"host_contract_cases": len(cases), "common_mismatches": sum(not c["common_equals_host"] for c in cases)}
        )
    )


if __name__ == "__main__":
    main()
