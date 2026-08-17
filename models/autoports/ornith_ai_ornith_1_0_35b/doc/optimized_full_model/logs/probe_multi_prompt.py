# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Second-request corruption probe: several prompts through one generator, with state instrumented.

The qualitative suite runs six prompts through a single generator with ``reset()`` between them and
only the first came out clean. This walks the same sequence in controlled arms so the discriminator
can be isolated:

``plain``     exactly the qualitative loop: ``reset(); generate(...)``.
``prefill``   the same, with one extra standalone ``prefill_forward_single`` before each generate.
``state``     the same as ``plain``, but reading the DeltaNet / KV / gate state to host each round,
              which also inserts a device synchronize.

Each arm prints the first tokens per prompt, so a clean arm next to a broken one names the cause.
"""

from __future__ import annotations

import argparse

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import close_ornith_mesh, open_ornith_mesh

QUALITATIVE_PROMPTS = [
    "Write a haiku about machine learning.",
    "Explain the difference between supervised and unsupervised learning in simple terms.",
    "Complete this story: Once upon a time, in a faraway kingdom, there lived a curious young inventor who discovered",
    "What are the three laws of thermodynamics?",
    'Translate the following to French: "Hello, how are you today?"',
    "Write a Python function to calculate the Fibonacci sequence.",
]


def host(mesh, tensor):
    return ttnn.to_torch(tensor, mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh, dim=0)).float()


def state_report(mesh, model):
    recurrent, conv, kv = [], [], []
    for layer in model.layers:
        if layer.is_full_attention:
            kv.append(float(host(mesh, layer.k_cache).abs().max()))
        else:
            recurrent.append(float(host(mesh, layer.recurrent_state).abs().max()))
            conv.append(max(float(host(mesh, b).abs().max()) for b in layer.conv_state))
    return {
        "recurrent_absmax": max(recurrent) if recurrent else 0.0,
        "conv_absmax": max(conv) if conv else 0.0,
        "kv_absmax": max(kv) if kv else 0.0,
    }


def run_arm(mesh, gen, tok, prompts, *, arm, gen_len):
    print(f"\n########## arm={arm} ##########", flush=True)
    outputs = []
    for index, prompt in enumerate(prompts):
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
        )
        ids = tok.encode(rendered, add_special_tokens=False)
        if arm == "state":
            print(f"  [{index}] state before reset: {state_report(mesh, gen.model)}", flush=True)
        gen.reset()
        if arm == "prefill":
            logits = gen.model.prefill_forward_single(
                ids, page_table=gen._prefill_page_row(0), start_pos=0, return_logits=True
            )
            print(f"  [{index}] standalone prefill top1={int(torch.argmax(logits[0, 0]))}", flush=True)
            gen.reset()
        elif arm == "sync":
            ttnn.synchronize_device(mesh)
        out = gen.generate(prompt_token_ids=ids, max_new_tokens=gen_len, enable_trace=True)
        text = tok.decode(out, skip_special_tokens=False)
        outputs.append(out)
        print(f"  [{index}] {prompt[:44]!r} -> {text[:110]!r}", flush=True)
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default=None)
    ap.add_argument("--gen-len", type=int, default=24)
    ap.add_argument("--cache-context", type=int, default=8192)
    ap.add_argument("--arms", default="plain,prefill,state,plain")
    ap.add_argument("--prompts", type=int, default=len(QUALITATIVE_PROMPTS))
    args = ap.parse_args()

    mesh = open_ornith_mesh()
    try:
        kwargs = {}
        if args.layers:
            kwargs["layer_indices"] = [int(v) for v in args.layers.split(",")]
        gen = build_generator(
            model_dir="models/autoports/ornith_ai_ornith_1_0_35b",
            mesh_device=mesh,
            max_batch_size=1,
            cache_context=args.cache_context,
            **kwargs,
        )
        tok = gen.tokenizer
        prompts = QUALITATIVE_PROMPTS[: args.prompts]
        for arm in args.arms.split(","):
            run_arm(mesh, gen, tok, prompts, arm=arm, gen_len=args.gen_len)
        gen.teardown()
        print("PROBE_OK")
    finally:
        close_ornith_mesh(mesh)


if __name__ == "__main__":
    main()
