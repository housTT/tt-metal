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
    p.add_argument("--batch", type=int, default=1)
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
        if args.batch > 1:
            from ..tt.generator import _greedy_params

            gen = build_generator(Path(__file__).parents[1], mesh, decode_batches=(1, args.batch))
            rows = [items[r % len(items)] for r in range(args.batch)]
            lens = [len(item["token_ids"]) for item in rows]
            table = gen.default_page_table(args.batch, max(lens) + args.max_new_tokens + 1)
            ids = torch.zeros(args.batch, max(lens), dtype=torch.int64)
            for r, item in enumerate(rows):
                ids[r, : lens[r]] = torch.tensor(item["token_ids"])
            first = gen.prefill_forward(ids, page_table=table, prompt_lens=lens)[:, 0].argmax(-1)
            second = gen.decode_forward(first.reshape(-1, 1), lens, page_table=table, sampling_params=_greedy_params())
            steps = gen.decode_steps(args.batch, args.max_new_tokens - 2, _greedy_params())
            columns = torch.stack([first, second, *steps], dim=1).tolist()
            reference_path = args.output_dir / f"tt_m{args.mesh}_outputs.json"
            reference = {o["id"]: o["tokens"] for o in json.loads(reference_path.read_text())} if reference_path.exists() else {}
            for r, (item, tokens) in enumerate(zip(rows, columns)):
                if eos in tokens:
                    tokens = tokens[: tokens.index(eos) + 1]
                ref = reference.get(item["id"])
                prefix = next((i for i, (a, b) in enumerate(zip(tokens, ref)) if a != b), min(len(tokens), len(ref))) if ref else None
                outputs.append(dict(row=r, id=item["id"], tokens=tokens, text=tokenizer.decode(tokens),
                                    b1_common_prefix=prefix, b1_tokens=len(ref) if ref else None,
                                    same_prompt_rows_identical=columns[r] == columns[r % len(items)]))
                if r < len(items):
                    print(json.dumps(dict(id=item["id"], b1_common_prefix=prefix, text=outputs[-1]["text"][:200])), flush=True)
            (args.output_dir / f"tt_m{args.mesh}_b{args.batch}_outputs.json").write_text(json.dumps(outputs, indent=1))
            return
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
