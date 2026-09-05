# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Run the shared prompt suite with exact HF chat formatting and one reused TT generator."""

import argparse
import hashlib
import importlib
import json
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import close_ornith_mesh, open_ornith_mesh

DOC = Path(__file__).resolve().parent
ROOT = DOC.parents[1]
SNAPSHOT = "/home/hous/dev/ornith-1.5-9b/upstream"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-index", type=int)
    parser.add_argument("--reuse-hf", type=Path)
    args = parser.parse_args()
    suite = Path("models/common/readiness_check/vllm_prompts.txt")
    prompts = [line.strip() for line in suite.read_text().splitlines() if line.strip()]
    runner = importlib.import_module("models.common.readiness_check.run_autoregressive")
    mesh = open_ornith_mesh()
    try:
        gen = build_generator(ROOT, mesh, cache_context=2048)
        if args.reuse_hf:
            previous = json.loads((args.reuse_hf / "qualitative_prompt_format.json").read_text())
            assert previous["revision"] == "489cb97981b8654bcfcf30ce1f94ed1b62e07b53"
            assert previous["template_sha256"] == hashlib.sha256(gen.tokenizer.chat_template.encode()).hexdigest()
            assert previous["source_sha256"] == hashlib.sha256(suite.read_bytes()).hexdigest()
            assert previous["max_new_tokens"] == 128 and previous["chat_template"]
        teardown = gen.teardown
        gen.teardown = lambda: None
        # Reuse weights, fixed caches and compiled traces across the entire suite.
        # Every prompt still uses the standard runner and the real generator.generate.
        try:
            for index, prompt in enumerate(prompts):
                if args.prompt_index is not None and index != args.prompt_index:
                    continue
                output = args.output / f"prompt_{index}"
                output.mkdir(parents=True, exist_ok=True)
                source = output / "user_prompt.txt"
                source.write_text(prompt)
                control_context = nullcontext()
                if args.reuse_hf:
                    cached = json.loads((args.reuse_hf / f"prompt_{index}" / "autoregressive_meta.json").read_text())

                    def cached_hf(**kwargs):
                        assert kwargs["hf_model_id"] == cached["hf_model_id"] == SNAPSHOT
                        assert kwargs["prompt_token_ids"] == cached["prompt_token_ids"]
                        assert kwargs["max_new_tokens"] == cached["max_new_tokens"] == 128
                        assert cached["chat_template"] and cached["prompt_text"] == prompt
                        return list(cached["hf"]["token_ids"])

                    control_context = patch.object(runner, "_hf_generate_greedy", side_effect=cached_hf)
                with (
                    control_context,
                    patch.object(runner, "_import_build_generator", return_value=lambda **kwargs: gen),
                ):
                    runner.run_autoregressive(
                        model_dir=ROOT,
                        hf_model_id=SNAPSHOT,
                        prompt_file=source,
                        mesh_device=mesh,
                        output_dir=output,
                        max_new_tokens=128,
                        chat_template=True,
                    )
                (output / "generator_perf.json").write_text(json.dumps(gen.perf, indent=2) + "\n")
            metadata = dict(
                hf_model_id="ornith-ai/Ornith-1.5-9B",
                revision="489cb97981b8654bcfcf30ce1f94ed1b62e07b53",
                snapshot=SNAPSHOT,
                prompt_mode="chat",
                chat_template=True,
                tokenizer_class=type(gen.tokenizer).__name__,
                template_sha256=hashlib.sha256(gen.tokenizer.chat_template.encode()).hexdigest(),
                source=str(suite),
                source_sha256=hashlib.sha256(suite.read_bytes()).hexdigest(),
                max_new_tokens=128,
                reused_generator=True,
                reused_hf_control=str(args.reuse_hf) if args.reuse_hf else None,
                prompt_indices=list(range(len(prompts))) if args.prompt_index is None else [args.prompt_index],
            )
            (args.output / "qualitative_prompt_format.json").write_text(json.dumps(metadata, indent=2) + "\n")
        finally:
            teardown()
    finally:
        close_ornith_mesh(mesh)


if __name__ == "__main__":
    main()
