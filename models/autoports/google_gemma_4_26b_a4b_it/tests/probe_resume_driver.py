# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run the readiness resume-check case logic through the real driver on a reduced layer set.

Usage (chip 1, from the tt-metal worktree):
    TT_VISIBLE_DEVICES=1 ... python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_resume_driver \
        --layers 0,5 --length 2560 --starts 64,2432 --output /tmp/resume_driver.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from readiness_check.run_prefill_resume_check import DEFAULT_THRESHOLDS, natural_tokens, run_case

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.test_full_model_contract import _load_probe_state
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import build_generator
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm import PRECISION_CONFIG


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=str, default="0,5", help="comma list, or 'all' for the full stack")
    parser.add_argument("--length", type=int, default=2560)
    parser.add_argument("--starts", type=str, default="64,2432")
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--controls", type=str, default="")
    parser.add_argument("--prompts-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("/tmp/gemma4_resume_driver.json"))
    parser.add_argument("--force-full-chunked", action="store_true")
    parser.add_argument("--dump-positions", action="store_true")
    parser.add_argument("--compare-cache", action="store_true")
    args = parser.parse_args()
    if args.force_full_chunked:
        from models.autoports.google_gemma_4_26b_a4b_it.tt import multichip_decoder

        original_path = multichip_decoder._prefill_attention_path

        def forced_path(seq_len, *, is_sliding, has_paged_cache, **kwargs):
            if not is_sliding and has_paged_cache:
                print(f"FORCED_FULL_CHUNKED seq_len={seq_len}", flush=True)
                return "full_chunked"
            return original_path(seq_len, is_sliding=is_sliding, has_paged_cache=has_paged_cache, **kwargs)

        multichip_decoder._prefill_attention_path = forced_path
    starts = [int(v) for v in args.starts.split(",")]
    full_stack = args.layers == "all"
    layer_indices = None if full_stack else [int(v) for v in args.layers.split(",")]
    load_full_state = full_stack or any(layer not in (0, 5) for layer in (layer_indices or []))
    controls = [c for c in args.controls.split(",") if c] if args.controls else []
    prompts_file = args.prompts_file
    if prompts_file is None:
        import readiness_check.run_prefill_resume_check as module

        prompts_file = Path(module.__file__).parent / "vllm_prompts.txt"
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), trace_region_size=0)
    report = {"length": args.length, "layers": args.layers, "max_seq_len": args.max_seq_len, "cases": []}
    try:
        began = time.perf_counter()
        state_dict = _load_probe_state(full_stack=load_full_state)
        gen = build_generator(
            Path(__file__).resolve().parents[1],
            mesh,
            layer_indices=layer_indices,
            max_seq_len=args.max_seq_len,
            max_batch_size=1,
            state_dict=state_dict,
            precision_config_path=str(PRECISION_CONFIG),
            tensor_cache_path=os.environ.get("GEMMA4_TENSOR_CACHE_PATH", "/tmp/gemma4_resume_driver_cache"),
        )
        print(f"generator built in {time.perf_counter() - began:.1f}s", flush=True)
        pieces = [p.strip() for p in prompts_file.read_text().split("\n\n") if p.strip()]
        tokens = natural_tokens(gen.tokenizer, pieces, args.length)
        for start in starts:
            if args.compare_cache:
                state = gen.model.state
                table = state.host_page_tables[0][0].tolist()
                spec = state.cache_specs[0]

                def cache_rows(cache, upto):
                    host = ttnn.to_torch(ttnn.get_device_tensors(cache)[0]).float()
                    blocks = [host[table[i]] for i in range((upto + spec.block_size - 1) // spec.block_size)]
                    joined = torch.cat(blocks, dim=1)
                    return joined[:, :upto, :]

                gen.reset()
                gen.prefill_logits_continuation(tokens, 0, positions=[args.length - 1])
                ref_k = cache_rows(state.kv_cache[0][0], args.length)
                gen.reset()
                gen.prefill_logits_continuation(tokens, start, positions=[args.length - 1])
                cand_k = cache_rows(state.kv_cache[0][0], args.length)
                diff = (ref_k - cand_k).abs()
                per_position = diff.amax(dim=(0, 2))
                rel = diff / ref_k.abs().clamp_min(1e-3)
                print(
                    json.dumps(
                        {
                            "start": start,
                            "k_cache_compare": {
                                "elements": int(diff.numel()),
                                "nonzero_fraction": round(float((diff > 0).float().mean()), 6),
                                "max_abs": float(diff.max()),
                                "max_rel": round(float(rel.max()), 5),
                                "positions_with_diff": int((per_position > 0).sum()),
                                "first_diff_positions": [
                                    int(i) for i in torch.nonzero(per_position > 0).reshape(-1)[:12]
                                ],
                                "prefix_nonzero_fraction": round(float((diff[:, :start] > 0).float().mean()), 6)
                                if start
                                else 0.0,
                                "suffix_nonzero_fraction": round(float((diff[:, start:] > 0).float().mean()), 6),
                            },
                        }
                    ),
                    flush=True,
                )
            if args.dump_positions:
                from readiness_check.run_prefill_resume_check import _select_positions, window_positions

                positions = window_positions(start, args.length, args.window)
                gen.reset()
                reference = _select_positions(
                    gen.prefill_logits_continuation(tokens, 0, positions=positions), 0, positions, args.length
                )
                gen.reset()
                candidate = _select_positions(
                    gen.prefill_logits_continuation(tokens, start, positions=positions), start, positions, args.length
                )
                ref = reference[0].float()
                cand = candidate[0].float()
                ref_c = ref - ref.mean(dim=-1, keepdim=True)
                cand_c = cand - cand.mean(dim=-1, keepdim=True)
                pcc = (ref_c * cand_c).sum(-1) / (ref_c.norm(dim=-1) * cand_c.norm(dim=-1)).clamp_min(1e-12)
                top1 = ref.argmax(-1) == cand.argmax(-1)
                worst = sorted(range(len(positions)), key=lambda i: float(pcc[i]))[:12]
                print(
                    json.dumps(
                        {
                            "start": start,
                            "worst_positions": [
                                {
                                    "position": positions[i],
                                    "rel": positions[i] - start,
                                    "pcc": round(float(pcc[i]), 5),
                                    "top1": bool(top1[i]),
                                }
                                for i in worst
                            ],
                            "pcc_by_rel_mod_32_mean": [
                                round(
                                    float(
                                        pcc[
                                            [i for i in range(len(positions)) if (positions[i] - start) % 32 == r]
                                        ].mean()
                                    ),
                                    5,
                                )
                                for r in range(32)
                            ],
                            "top1_mismatch_rel": [positions[i] - start for i in range(len(positions)) if not top1[i]],
                        }
                    ),
                    flush=True,
                )
            case = run_case(
                generator=gen,
                tokens=tokens,
                start=start,
                kind="probe",
                window=args.window,
                thresholds=dict(DEFAULT_THRESHOLDS),
                controls=controls,
            )
            case_summary = {k: v for k, v in case.items() if k != "controls"}
            case_summary["controls"] = {
                name: {"top1_agreement": c.get("top1_agreement"), "failed_as_expected": c.get("failed_as_expected")}
                for name, c in case["controls"].items()
            }
            report["cases"].append(case_summary)
            print(json.dumps(case_summary), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
