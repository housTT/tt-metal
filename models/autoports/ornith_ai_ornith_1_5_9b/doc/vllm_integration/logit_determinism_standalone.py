# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""All-32-layer selected-policy raw-logit control; supervising hardware lane only.

Uses the standalone low-level generator with explicit raw-logits/host-greedy
diagnostic output, native context and an exact shared physical pool. This is
not the serving adapter, a device-sampling benchmark, or an HF accuracy test.
Run as a module only after stopping the server and closing its device mesh.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

import ttnn

from ...tt.functional_decoder import num_blocks_for_context
from ...tt.generator import OrnithGenerator
from ...tt.model import OrnithModel, close_ornith_mesh, open_ornith_mesh
from .logit_determinism_vllm import CASES, REVISION, compare, comparisons, prompt_manifest, save, signature


def logits_signature(logits, prompt_ids, top_count):
    """Pinned vLLM sampler's raw-logprob definition is FP32 log_softmax."""
    values = logits.log_softmax(dim=-1, dtype=torch.float32)
    tokens = logits.argmax(dim=-1).tolist()
    top_values, top_ids = values.topk(top_count, dim=-1)
    choice = {
        "prompt_token_ids": prompt_ids,
        "token_ids": tokens,
        "logprobs": {
            "tokens": [f"token_id:{value}" for value in tokens],
            "token_logprobs": values.gather(-1, torch.tensor(tokens)[:, None]).flatten().tolist(),
            "top_logprobs": [
                {f"token_id:{key}": value for key, value in zip(keys, row)}
                for keys, row in zip(top_ids.tolist(), top_values.tolist())
            ],
        },
    }
    return signature(choice, prompt_ids, len(tokens), top_count)


def run_case(gen, rows, prompt_ids, steps):
    # A new prefill overwrites every causal KV position that this short request
    # reads. Reset recurrent/conv state; leave masked, unused physical KV alone.
    gen.reset(clear_kv=False)
    lengths = list(map(len, prompt_ids))
    output = gen.prefill_forward(
        prompt_ids,
        page_table=gen.page_table,
        kv_cache=gen.kv_cache,
        prompt_lens=lengths,
        slots=rows,
        return_device_logits=True,
    )
    history = [gen.logits_from(output)[rows].clone()]
    ttnn.deallocate(output)  # The public prefill output is caller-owned.
    for step in range(1, steps):
        tokens = torch.zeros(gen.max_batch_size, dtype=torch.int64)
        positions = torch.full((gen.max_batch_size,), -1, dtype=torch.int32)
        tokens[rows] = history[-1].argmax(-1)
        positions[rows] = torch.tensor(lengths, dtype=torch.int32) + step - 1
        output = gen.decode_forward(
            tokens,
            positions,
            page_table=gen.page_table,
            kv_cache=gen.kv_cache,
            return_logits=True,
            read_from_device=False,
            sample_on_device=False,
        )
        history.append(gen.logits_from(output)[rows].clone())
    # Decode output aliases persistent traced storage; it is not deallocated.
    return torch.stack(history, dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("../upstream"))
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--vllm-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--raw-logits-output", type=Path, help="Optional external .pt destination for the 42 MiB raw logits"
    )
    args = parser.parse_args()
    assert 3 <= args.batch <= 32 and args.max_tokens > 0 and 1 <= args.top_logprobs <= 20
    prompts = prompt_manifest(args.model_path)
    serving = json.loads(args.vllm_result.read_text())
    assert prompts == serving["prompts"]
    assert serving["server_max_num_seqs"] == args.batch
    assert len(serving["cases"]) == len(CASES)
    logits_path = args.raw_logits_output or args.output.with_suffix(".pt")
    logits_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "standalone all32-layer selected-policy raw logits; host-greedy diagnostic, no performance claim",
        "hardware": "P300c, four Blackhole chips, TP4",
        "checkpoint_revision": REVISION,
        "prompts": prompts,
        "batch": args.batch,
        "vllm_result": str(args.vllm_result),
        "vllm_result_sha256": hashlib.sha256(args.vllm_result.read_bytes()).hexdigest(),
        "cases": [],
        "cleanup_completed": False,
    }
    save(args.output, report)
    mesh = open_ornith_mesh()
    gen = None
    full_logits, references, full_comparisons = {}, {}, []
    try:
        model = OrnithModel(args.model_path, mesh)
        assert model.layer_indices == list(range(32)) and model.max_context == 262144
        report.update(layers=model.layer_indices, precision=model.precision, logical_context=model.max_context)
        root = Path(__file__).resolve().parents[2]
        report["source_sha256"] = {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), root / "tt/model.py", root / "tt/generator.py", root / "tt/precision.py"]
        }
        width = num_blocks_for_context(model.max_context, model.page_block_size)
        used = num_blocks_for_context(
            max(item["token_count"] for item in prompts.values()) + args.max_tokens, model.page_block_size
        )
        pool_blocks = width + args.batch * used
        cache = model.allocate_cache(args.batch, num_blocks=pool_blocks)
        table = torch.zeros(args.batch, width, dtype=torch.int32)
        table[:, :used] = torch.arange(args.batch * used, dtype=torch.int32).reshape(args.batch, used)
        gen = OrnithGenerator(model, kv_cache=cache, page_table=table)
        gen.ensure_traces(preserve_cache=False)
        report["physical_blocks"] = pool_blocks
        for name, labels in CASES:
            rows = [0] if len(labels) == 1 else [0, 1, args.batch - 1]
            ids = [prompts[label]["token_ids"] for label in labels]
            logits = run_case(gen, rows, ids, args.max_tokens)
            assert torch.isfinite(logits).all()
            full_logits[name] = logits
            case = {
                "name": name,
                "labels": list(labels),
                "device_rows": rows,
                "signatures": [
                    logits_signature(row, prompt_ids, args.top_logprobs) for row, prompt_ids in zip(logits, ids)
                ],
                "full_logits_shape": list(logits.shape),
                "full_logits_sha256": hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
            }
            for position, (label, values) in enumerate(zip(labels, logits)):
                location = f"{name}[{position}]/device_row{rows[position]}"
                if label not in references:
                    references[label] = (location, values.clone())
                else:
                    source, previous = references[label]
                    full_comparisons.append(
                        {
                            "reference": source,
                            "candidate": location,
                            "exact": torch.equal(previous, values),
                            "max_abs_difference": float((previous - values).abs().max()),
                        }
                    )
            report["cases"].append(case)
            torch.save(full_logits, logits_path)
            save(args.output, report)
            print(json.dumps({"case": name, "device_rows": rows, "tokens": logits.argmax(-1).tolist()}), flush=True)
        report["full_logits_comparisons"] = full_comparisons
        report["logprob_comparisons"] = comparisons(report["cases"])
        report["vllm_comparisons"] = []
        for standalone, api in zip(report["cases"], serving["cases"]):
            assert standalone["name"] == api["name"] and standalone["labels"] == api["labels"]
            for position, (a, b) in enumerate(zip(standalone["signatures"], api["signatures"])):
                report["vllm_comparisons"].append(
                    {
                        "case": standalone["name"],
                        "api_position": position,
                        "device_row": standalone["device_rows"][position],
                        **compare(a, b),
                    }
                )
        report["counters"] = dict(gen.counters)
        report["full_logits_artifact"] = str(logits_path)
        report["full_logits_artifact_sha256"] = hashlib.sha256(logits_path.read_bytes()).hexdigest()
        report["full_logits_artifact_bytes"] = logits_path.stat().st_size
        report["passed"] = all(
            item["exact"]
            for key in ("full_logits_comparisons", "logprob_comparisons", "vllm_comparisons")
            for item in report[key]
        )
        save(args.output, report)
        print(
            json.dumps(
                {
                    "passed": report["passed"],
                    "full_logits_comparisons": full_comparisons,
                    "vllm_comparisons": report["vllm_comparisons"],
                }
            ),
            flush=True,
        )
        assert report["passed"], "Numerical control mismatch; inspect full logits and saved comparisons"
    finally:
        try:
            if gen is not None:
                gen.teardown()
        finally:
            close_ornith_mesh(mesh)
        report["cleanup_completed"] = True
        save(args.output, report)


if __name__ == "__main__":
    main()
