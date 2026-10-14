# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import time
from pathlib import Path

import torch

CHECKPOINT = Path("/home/hous/dev/qwen3.5-35b-a3b/weights/Qwen3.5-35B-A3B")
REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
SUITE = Path("/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/runtime/readiness_check/"
             "vllm_prompts.txt")


def prompts(tokenizer):
    items = [p.strip() for p in SUITE.read_text().split("\n\n") if p.strip()]
    rendered = []
    for index, text in enumerate(items):
        chat = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                             add_generation_prompt=True)
        ids = tokenizer(chat, add_special_tokens=False).input_ids
        rendered.append(dict(id=f"p{index}", text=text, rendered=chat, token_ids=ids))
    return rendered


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["hf", "tt"], required=True)
    p.add_argument("--mesh", type=int, choices=[1, 2, 4])
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    items = prompts(tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(hf_model="Qwen/Qwen3.5-35B-A3B", revision=REVISION, checkpoint=str(CHECKPOINT),
                tokenizer_class=type(tokenizer).__name__, chat_template_present=bool(tokenizer.chat_template),
                prompt_mode="chat", rendering="tokenizer.apply_chat_template(tokenize=False, "
                "add_generation_prompt=True), default template arguments", prompt_source=str(SUITE),
                generation=dict(greedy=True, max_new_tokens=args.max_new_tokens, stop="eos"))
    (args.output_dir / "qualitative_prompt_format.json").write_text(json.dumps(meta, indent=1))
    (args.output_dir / "rendered_prompts.json").write_text(json.dumps(items, indent=1))
    eos = tokenizer.eos_token_id
    outputs = []
    if args.phase == "hf":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, local_files_only=True).eval()
        for item in items:
            start = time.perf_counter()
            with torch.no_grad():
                out = model.generate(torch.tensor([item["token_ids"]]), max_new_tokens=args.max_new_tokens,
                                     do_sample=False, num_beams=1, pad_token_id=eos)
            tokens = out[0, len(item["token_ids"]):].tolist()
            outputs.append(dict(id=item["id"], tokens=tokens, text=tokenizer.decode(tokens),
                                seconds=time.perf_counter() - start))
            print(json.dumps(dict(id=item["id"], text=outputs[-1]["text"][:200])), flush=True)
        (args.output_dir / "hf_outputs.json").write_text(json.dumps(outputs, indent=1))
        return
    import ttnn

    from ..tt.generator import build_generator
    from .mesh_util import open_mesh

    mesh = open_mesh(args.mesh)
    try:
        gen = build_generator(Path(__file__).parents[1], mesh)
        for item in items:
            gen.reset()
            tokens = gen.generate(item["token_ids"], args.max_new_tokens, stop_on_eos=True)
            if eos in tokens:
                tokens = tokens[: tokens.index(eos) + 1]
            outputs.append(dict(id=item["id"], tokens=tokens, text=tokenizer.decode(tokens)))
            print(json.dumps(dict(id=item["id"], text=outputs[-1]["text"][:200])), flush=True)
        (args.output_dir / f"tt_m{args.mesh}_outputs.json").write_text(json.dumps(outputs, indent=1))
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
