# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Localize the exact French branch to the final projection policy."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import OrnithGenerator
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import OrnithModel, close_ornith_mesh, open_ornith_mesh

DOC = Path(__file__).resolve().parent
SNAPSHOT = "/home/hous/dev/ornith-1.5-9b/upstream"
# Each candidate obeys the full-stack resident L1 budget. BF16 uses the previously proven geometry.
POLICIES = {
    "bf4_lofi": ("bfloat4_b", "LoFi", 32768, 4, 2),
    "bf4_hifi2": ("bfloat4_b", "HiFi2", 32768, 4, 2),
    "bf4_hifi4": ("bfloat4_b", "HiFi4", 32768, 4, 2),
    "bf8_lofi": ("bfloat8_b", "LoFi", 32768, 2, 2),
    "bf8_hifi2": ("bfloat8_b", "HiFi2", 32768, 2, 2),
    "bf16_hifi4": ("bfloat16", "HiFi4", 8192, 4, 1),
    "bf16_hifi4_n16k_r2": ("bfloat16", "HiFi4", 16384, 4, 2),
    "bf16_hifi4_n32k_k1_r2": ("bfloat16", "HiFi4", 32768, 1, 2),
    "bf16_hifi4_n32k_k2_r2": ("bfloat16", "HiFi4", 32768, 2, 2),
    "bf16_hifi4_n32k_k4_r2": ("bfloat16", "HiFi4", 32768, 4, 2),
}


def kwargs(name):
    dtype, fidelity, columns, block, readers = POLICIES[name]
    return dict(
        lm_head_dtype=getattr(ttnn, dtype),
        lm_head_fidelity=getattr(ttnn.MathFidelity, fidelity),
        lm_head_columns=columns,
        lm_head_block_w=block,
        lm_head_readers=readers,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["capture", "terminal", "french"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=list(POLICIES), default="bf4_lofi")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    control_path = DOC / "qualitative_final_v1/french_branch_control.json"
    control = json.loads(control_path.read_text())
    prompt, prefix = control["prompt_token_ids"], control["tt_prefix_token_ids"]
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    report = dict(
        mode=args.mode, policy=args.policy, control_sha256=hashlib.sha256(control_path.read_bytes()).hexdigest()
    )
    mesh = open_ornith_mesh()
    try:
        model = OrnithModel(
            None, mesh, layer_indices=[] if args.mode == "terminal" else None, max_context=2048, **kwargs(args.policy)
        )
        if args.mode == "capture":
            captured = {}
            original = model.terminal

            def terminal(hidden):
                captured["hidden"] = ttnn.clone(hidden)
                return original(hidden)

            model.terminal = terminal
            gen = OrnithGenerator(model, cache_context=2048)
            try:
                tokens = gen.generate(
                    prompt,
                    len(prefix) + 1,
                    next_input=lambda step, token: prefix[step] if step < len(prefix) else token,
                    stop_on_eos=False,
                )
                assert tokens[-1] == 39102, tokens
                hidden = [ttnn.to_torch(t).clone() for t in ttnn.get_device_tensors(captured["hidden"])]
                torch.save(
                    dict(hidden=hidden, prompt=prompt, prefix=prefix, prediction=tokens[-1]), args.output / "hidden.pt"
                )
                report.update(
                    predictions=tokens,
                    prediction_text=tokenizer.decode(tokens[-1:]),
                    hidden_shape=list(hidden[0].shape),
                    replicas_exact=all(torch.equal(hidden[0], h) for h in hidden[1:]),
                    hidden_sha256=hashlib.sha256((args.output / "hidden.pt").read_bytes()).hexdigest(),
                )
            finally:
                gen.teardown()
        elif args.mode == "terminal":
            captured = torch.load(args.output / "hidden.pt", weights_only=True)
            hidden = model.upload(torch.cat(captured["hidden"], dim=0), shard_dim=0)
            view = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
            extra = 221952 - int(view.total_bytes_allocated_per_bank)
            assert extra > 0 and extra % 64 == 0, extra
            grid = mesh.compute_with_storage_grid_size()
            memory = ttnn.create_sharded_memory_config(
                shape=(1, extra // 2),
                core_grid=ttnn.CoreGrid(x=grid.x, y=grid.y),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            resident = model.upload(
                torch.zeros(grid.x * grid.y, extra // 2, dtype=torch.bfloat16),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory=memory,
            )
            report["resident_l1_bytes_per_bank"] = int(
                ttnn.get_memory_view(mesh, ttnn.BufferType.L1).total_bytes_allocated_per_bank
            )
            logits = model.terminal(hidden)
            scores = model.logits_to_host(logits, 1)[0]
            score_path = args.output / f"scores_{args.policy}.pt"
            torch.save(scores, score_path)
            report["scores_sha256"] = hashlib.sha256(scores.contiguous().numpy().tobytes()).hexdigest()
            baseline_path = args.output / "scores_bf16_hifi4.pt"
            if baseline_path.exists():
                report["exact_bf16_baseline"] = bool(torch.equal(scores, torch.load(baseline_path, weights_only=True)))
            values, indices = scores.topk(10)
            report.update(
                top10_ids=indices.tolist(),
                top10_text=[tokenizer.decode([i]) for i in indices.tolist()],
                top10_logits=values.tolist(),
                inform_rank=int((scores > scores[39102]).sum()) + 1,
            )
            ttnn.synchronize_device(mesh)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            traced = model.terminal(hidden)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            assert torch.equal(scores, model.logits_to_host(traced, 1)[0])
            start = time.perf_counter()
            for _ in range(64):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            report["trace_ms"] = (time.perf_counter() - start) * 1000 / 64
            ttnn.release_trace(mesh, trace)
            print("TERMINAL_BRANCH", report, flush=True)
        else:
            gen = OrnithGenerator(model, cache_context=2048)
            try:
                tokens = gen.generate(prompt, 128, stop_on_eos=False)
                text = tokenizer.decode(tokens, skip_special_tokens=False)
                report.update(tokens=tokens, completion=text, perf=gen.perf)
                (args.output / f"french_{args.policy}.txt").write_text(text)
                print("FRENCH_COMPLETION", text, flush=True)
            finally:
                gen.teardown()
        report["probe_execution_pass"] = True
        if args.mode == "french":
            report["qualitative_correct"] = None  # Requires reading the saved completion.
    finally:
        close_ornith_mesh(mesh)
        (args.output / f"{args.mode}_{args.policy}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
