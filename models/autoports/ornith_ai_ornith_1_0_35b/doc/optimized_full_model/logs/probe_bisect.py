# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Which step permanently corrupts the model: trace capture, sampling, or a post-capture compile?

The multi-prompt probe showed the corruption is **permanent** — once the second request goes wrong,
even a bare prefill of the first prompt is wrong afterwards — and that repeating *one* prompt never
corrupts anything. The remaining difference is that a second prompt of a different length compiles
new programs, and tt-metal warns that *"allocating device buffers is unsafe due to the existence of
an active trace; these buffers may be corrupted once a trace is executed"* — a program's kernel
binaries are such a buffer, and unlike an activation they live in the program cache forever.

This walks that hypothesis:

``after``  capture the decode trace first, then compile a second prefill length, then replay -> the
           first length's prefill is re-checked. Corruption here means post-capture compilation.
``before`` compile both prefill lengths *before* capture, then replay -> both re-checked. Clean here
           means "no program-cache miss after capture" is the property that matters.

``num_program_cache_entries`` is printed at every step, so the compiles are visible rather than
inferred.
"""

from __future__ import annotations

import argparse

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import close_ornith_mesh, open_ornith_mesh

PROMPT_A = "Write a haiku about machine learning."
PROMPT_B = "Explain the difference between supervised and unsupervised learning in simple terms."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default=None)
    ap.add_argument("--cache-context", type=int, default=8192)
    ap.add_argument("--order", default="after", choices=["after", "before"])
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

        def encode(prompt):
            rendered = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
            )
            return tok.encode(rendered, add_special_tokens=False)

        ids_a, ids_b = encode(PROMPT_A), encode(PROMPT_B)

        def bare_prefill(label, ids):
            before = mesh.num_program_cache_entries()
            gen.reset()
            logits = gen.model.prefill_forward_single(
                ids, page_table=gen._prefill_page_row(0), start_pos=0, return_logits=True
            )
            top1 = int(torch.argmax(logits[0, 0]))
            after = mesh.num_program_cache_entries()
            print(
                f"CHECK {label:<34} len={len(ids):<4} top1={top1:>7} ({tok.decode([top1])!r:>14}) "
                f"programs {before}->{after} (+{after - before})",
                flush=True,
            )
            return top1

        def replay(label):
            before = mesh.num_program_cache_entries()
            gen._decode_step_traced()
            gen._sample_traced()
            ttnn.synchronize_device(mesh)
            print(f"REPLAY {label:<33} programs {before}->{mesh.num_program_cache_entries()}", flush=True)

        if args.order == "before":
            base_a = bare_prefill("pre-capture prefill A", ids_a)
            base_b = bare_prefill("pre-capture prefill B", ids_b)
            # These two prefills exist to *compile* their programs; their state is not wanted. The
            # generator refuses to capture over live prompt state (README §5.3) - correctly, since
            # capture wipes it - so say explicitly that it is disposable.
            gen.reset()
            gen._ensure_decode_trace()
            print(f"TRACE captured; programs={mesh.num_program_cache_entries()}", flush=True)
        else:
            gen._ensure_decode_trace()
            print(f"TRACE captured; programs={mesh.num_program_cache_entries()}", flush=True)
            base_a = bare_prefill("post-capture prefill A", ids_a)
            base_b = bare_prefill("post-capture prefill B", ids_b)

        replay("after both prefills")
        again_a = bare_prefill("prefill A again", ids_a)
        again_b = bare_prefill("prefill B again", ids_b)
        replay("again")
        final_a = bare_prefill("prefill A once more", ids_a)
        print(
            f"RESULT order={args.order} A stable={base_a == again_a == final_a} "
            f"B stable={base_b == again_b} (A={base_a}/{again_a}/{final_a} B={base_b}/{again_b})"
        )
        gen.teardown()
        print("PROBE_OK")
    finally:
        close_ornith_mesh(mesh)


if __name__ == "__main__":
    main()
