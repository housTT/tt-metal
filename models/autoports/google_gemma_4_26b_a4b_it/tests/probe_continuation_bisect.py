# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bisect the resumed-prefill mismatch by layer kind on real weights (one chip).

Usage (chip 1, from the tt-metal worktree):
    TT_VISIBLE_DEVICES=1 ... python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_continuation_bisect \
        --layers 0 --layers 5 --length 2560 --starts 128,2432 --output /tmp/bisect.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.test_full_model_contract import _config, _load_probe_state
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator, _padded_prefill_len
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm import PRECISION_CONFIG
from models.autoports.google_gemma_4_26b_a4b_it.tt.model import Gemma4FullModel


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten() - a.float().mean()
    b = b.float().flatten() - b.float().mean()
    denom = float(a.norm() * b.norm())
    return float(a @ b) / denom if denom else 1.0


def _hidden_to_host(gen: Gemma4Generator, hidden: ttnn.Tensor) -> torch.Tensor:
    host = ttnn.to_torch(ttnn.get_device_tensors(hidden)[0]).float()
    ttnn.deallocate(hidden)
    return host.reshape(-1, host.shape[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", action="append", type=int, required=True)
    parser.add_argument("--length", type=int, default=2560)
    parser.add_argument("--starts", type=str, default="128,2432")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=Path("/tmp/gemma4_continuation_bisect.json"))
    args = parser.parse_args()
    starts = [int(v) for v in args.starts.split(",")]
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), trace_region_size=0)
    report = {"length": args.length, "cases": []}
    try:
        state_dict = _load_probe_state(full_stack=False)
        rng = torch.Generator().manual_seed(0)
        tokens = torch.randint(1, 262_000, (args.length,), generator=rng).tolist()
        for layer in args.layers:
            model = Gemma4FullModel(
                mesh_device=mesh,
                hf_config=_config(),
                state_dict=state_dict,
                max_seq_len=args.max_seq_len,
                max_batch_size=1,
                layer_indices=[layer],
                tensor_cache_path=os.environ.get("GEMMA4_TENSOR_CACHE_PATH", "/tmp/gemma4_bisect_cache"),
                create_kv_cache=True,
                precision_config_path=PRECISION_CONFIG,
            )
            gen = Gemma4Generator(model, tokenizer=None, sampling_mode="device")
            state = model.state
            full_tokens = torch.tensor(tokens, dtype=torch.long).reshape(1, -1)
            padded = torch.nn.functional.pad(full_tokens, (0, _padded_prefill_len(args.length) - args.length))
            gen.reset()
            reference = _hidden_to_host(
                gen,
                gen.prefill_forward(
                    padded, page_table=state.page_tables, kv_cache=state, prompt_lens=[args.length], return_hidden=True
                ),
            )
            kind = model.hf_config.layer_types[layer]
            for start in starts:
                gen.reset()
                prefix = torch.nn.functional.pad(full_tokens[:, :start], (0, _padded_prefill_len(start) - start))
                prefix_hidden = gen.prefill_forward(
                    prefix, page_table=state.page_tables, kv_cache=state, prompt_lens=[start], return_hidden=True
                )
                prefix_host = _hidden_to_host(gen, prefix_hidden)
                suffix_len = args.length - start
                suffix = torch.nn.functional.pad(
                    full_tokens[:, start:], (0, _padded_prefill_len(suffix_len) - suffix_len)
                )
                continuation = _hidden_to_host(
                    gen,
                    gen.prefill_forward(
                        suffix,
                        page_table=state.page_tables,
                        kv_cache=state,
                        prompt_lens=[suffix_len],
                        start_pos=torch.arange(start, args.length, dtype=torch.int32).reshape(1, suffix_len),
                        return_hidden=True,
                    ),
                )
                rows = sorted(
                    {0, 1, 2, 3, 7, 15, 31, 32, 63, 64, 127, 128, 255, 256, 511, 1023, 1024, suffix_len - 1}
                    & set(range(suffix_len))
                )
                per_row = {int(r): round(_pcc(reference[start + r], continuation[r]), 4) for r in rows}
                prefix_pcc = round(_pcc(reference[:start], prefix_host[:start]), 4)
                whole = round(_pcc(reference[start:], continuation), 4)
                case = {
                    "layer": layer,
                    "kind": kind,
                    "start": start,
                    "suffix_len": suffix_len,
                    "prefix_rows_pcc_vs_reference": prefix_pcc,
                    "continuation_pcc_all_rows": whole,
                    "continuation_pcc_by_row": per_row,
                }
                report["cases"].append(case)
                print(json.dumps(case))
            del gen, model
    finally:
        ttnn.close_mesh_device(mesh)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
