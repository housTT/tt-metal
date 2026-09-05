# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Persistent trace-boundary copies for exact duplicate-slot localization."""

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.probe_batch32 import difference, ranks
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import close_ornith_mesh, open_ornith_mesh

DOC = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mesh = open_ornith_mesh()
    gen = None
    report = {}
    original_norm = ttnn.rms_norm
    original_sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode
    try:
        gen = build_generator(DOC.parents[1], mesh, layer_indices=[0, 3], cache_context=2048, max_batch_size=32)
        snapshots = {
            name: gen.model.upload(torch.zeros(shape))
            for name, shape in (
                ("embedding", [32, 1, 4096]),
                ("layer0", [32, 1, 4096]),
                ("layer3", [32, 1, 4096]),
                ("terminal_input", [1, 32, 4096]),
                ("terminal_norm", [1, 1, 32, 4096]),
            )
        }
        active = [False, False]
        full = gen.kv_cache.decode_layers[1]
        heads, dim = full.cfg.n_heads, full.cfg.head_dim
        for name, count in (("q_raw", heads), ("q_norm", heads), ("k_raw", 1), ("k_norm", 1)):
            snapshots[name] = gen.model.upload(torch.zeros(1, 32, count, dim))
        for name, shape in (
            ("sdpa_q", [1, 32, heads, dim]),
            ("sdpa_out", [1, 32, heads, dim]),
            ("qkvg", [32, 1, 2 * heads * dim + 2 * dim]),
        ):
            snapshots[name] = gen.model.upload(torch.zeros(shape))
        original_linear = full._linear

        def linear(x, name, **kw):
            result = original_linear(x, name, **kw)
            if active[0] and name == "qkvg":
                ttnn.copy(result, snapshots["qkvg"])
            return result

        full._linear = linear

        def sdpa(q, *a, **kw):
            result = original_sdpa(q, *a, **kw)
            if active[0]:
                ttnn.copy(q, snapshots["sdpa_q"])
                ttnn.copy(result, snapshots["sdpa_out"])
            return result

        ttnn.transformer.paged_scaled_dot_product_attention_decode = sdpa
        original_forward = gen._forward

        def forward():
            active[0] = True
            try:
                return original_forward()
            finally:
                active[0] = False

        gen._forward = forward
        original_embed = gen.model.embed

        def embed(ids):
            result = original_embed(ids)
            if active[0]:
                ttnn.copy(result, snapshots["embedding"])
            return result

        gen.model.embed = embed
        for name, layer in zip(("layer0", "layer3"), gen.kv_cache.decode_layers):
            original = layer.decode_forward

            def decode(*a, _original=original, _name=name, **kw):
                result = _original(*a, **kw)
                if active[0]:
                    ttnn.copy(result, snapshots[_name])
                return result

            layer.decode_forward = decode
        original_terminal = gen.model.terminal

        def terminal(hidden):
            if active[0]:
                ttnn.copy(hidden, snapshots["terminal_input"])
                active[1] = True
            try:
                return original_terminal(hidden)
            finally:
                active[1] = False

        gen.model.terminal = terminal

        def norm(*a, **kw):
            result = original_norm(*a, **kw)
            if active[1]:
                ttnn.copy(result, snapshots["terminal_norm"])
            if active[0]:
                for label in ("q", "k"):
                    if kw.get("weight") is full.w[label + "_norm"]:
                        ttnn.copy(a[0], snapshots[label + "_raw"])
                        ttnn.copy(result, snapshots[label + "_norm"])
            return result

        ttnn.rms_norm = norm
        gen.ensure_traces()
        prompts = [([*range(127)], [*range(131)], [31, 57, 88])[i % 3] for i in range(31)]
        gen.reset()
        prefill = gen.prefill_forward(
            prompts,
            page_table=gen.page_table,
            kv_cache=gen.kv_cache,
            prompt_lens=list(map(len, prompts)),
            return_device_logits=True,
        )
        ttnn.deallocate(prefill)
        gen._write_tokens([[12, 1076, 220][i % 3] for i in range(32)])
        gen._write_positions([len(row) for row in prompts] + [-1])
        gen._refresh_table(gen.page_table)
        gen._ensure_replay_safe()
        ttnn.execute_trace(mesh, gen._model_trace, cq_id=0, blocking=True)
        host_snapshots = {}
        for name, tensor in list(snapshots.items()) + [("logits", gen._logits)]:
            values = [value.reshape(32, -1) for value in ranks(tensor)]
            host_snapshots[name] = [value[[0, 3, 6, 30]].clone() for value in values]
            report[name] = {
                f"rank{rank}_slot{slot}": difference(value[0], value[slot])
                for rank, value in enumerate(values)
                for slot in (3, 6, 30)
            }
            print(
                name, [(key, value["max_abs"]) for key, value in report[name].items() if not value["equal"]], flush=True
            )
        torch.save(host_snapshots, args.output.with_suffix(".pt"))
        caches = {}
        for name, tensor in (("k", full.k_cache), ("v", full.v_cache)):
            caches[name] = []
            for slot in (0, 3, 6, 30):
                pages = ttnn.slice(tensor, [slot * 32, 0, 0, 0], [slot * 32 + 3, 1, 64, dim])
                caches[name].append([v.reshape(-1, dim)[:128] for v in ranks(pages)])
                ttnn.deallocate(pages)
            report[f"cache_{name}"] = {
                f"rank{rank}_slot{slot}": difference(caches[name][0][rank], caches[name][i][rank])
                for rank in range(4)
                for i, slot in enumerate((0, 3, 6, 30))
                if i
            }
        oracle = {}
        for rank in range(4):
            for index, slot in enumerate((0, 3, 6, 30)):
                q = host_snapshots["sdpa_q"][rank][index].reshape(heads, dim).float()
                k, v = caches["k"][index][rank], caches["v"][index][rank]
                expected = (q @ k.T / dim**0.5).softmax(-1) @ v
                actual = host_snapshots["sdpa_out"][rank][index].reshape(heads, dim)
                oracle[f"rank{rank}_slot{slot}"] = difference(expected, actual)
        report["sdpa_cpu_oracle"] = oracle
        torch.save({"snapshots": host_snapshots, "caches": caches}, args.output.with_suffix(".pt"))
        report["snapshot_artifact"] = str(args.output.with_suffix(".pt"))
        report["device_positions"] = ranks(gen._inputs[1])[0].tolist()
    finally:
        ttnn.rms_norm = original_norm
        ttnn.transformer.paged_scaled_dot_product_attention_decode = original_sdpa
        if gen:
            gen.teardown()
        close_ornith_mesh(mesh)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
