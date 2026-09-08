# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Supervising hardware lane: selected-policy standalone shared haiku continuation."""

import argparse
import hashlib
import json
import shlex
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

DOC = Path(__file__).resolve().parent
MODEL = DOC.parents[1]
REVISION = "489cb97981b8654bcfcf30ce1f94ed1b62e07b53"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("../upstream"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    assert 256 <= args.max_new_tokens <= 1024
    controls = json.loads((DOC / "qualitative_controls.json").read_text())
    control = controls["prompts"][0]
    previous_path = Path(control["source"]) / "autoregressive_meta.json"
    previous = json.loads(previous_path.read_text())
    serving_path = MODEL / "readiness_vllm/vllm_qualitative_outputs.json"
    serving = json.loads(serving_path.read_text())[0]
    assert serving["prompt"] == control["prompt"] == previous["prompt_text"]
    precision_path = MODEL / "doc/datatype_sweep/selected_precision_config.json"
    selected = json.loads(precision_path.read_text())
    assert selected == controls["provenance"]["selected_precision"]
    snapshot = args.snapshot.resolve()
    metadata_files = sorted((snapshot / ".cache/huggingface/download").glob("*.metadata"))
    assert metadata_files
    for path in metadata_files:
        assert path.read_text().splitlines()[0] == REVISION
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "starting",
        "command": shlex.join([sys.executable, *sys.argv]),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "revision": REVISION,
        "snapshot": str(snapshot),
        "scope": "full32 standalone device-sampled haiku quality control, no performance claim",
        "batch": 1,
        "cache_context_requested": 2048,
        "logical_max_context": 262144,
        "generation_kwargs": {"max_new_tokens": args.max_new_tokens, "stop_on_eos": True},
        "previous_selected128_source": str(previous_path),
        "serving256_source": str(serving_path),
        "source_sha256": {
            str(path): sha(path)
            for path in [
                Path(__file__),
                precision_path,
                MODEL / "tt/generator.py",
                MODEL / "tt/model.py",
                serving_path,
                previous_path,
            ]
        },
        "cleanup_completed": False,
    }

    def save():
        (args.output / "metadata.json").write_text(json.dumps(result, indent=2, default=str) + "\n")

    save()
    # Imports remain inside main so --help and source checks do not import TTNN.
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import close_ornith_mesh, open_ornith_mesh

    mesh = None
    gen = None
    try:
        print("STANDALONE_HAIKU_BEGIN", flush=True)
        mesh = open_ornith_mesh()
        gen = build_generator(snapshot, mesh, max_batch_size=1, cache_context=2048, precision_config=precision_path)
        assert gen.model.layer_indices == list(range(32))
        assert gen.model.precision == selected
        assert gen.sampling_mode == "device"
        assert gen.model.model_path.resolve() == snapshot
        rendered = gen.tokenizer.apply_chat_template(
            [{"role": "user", "content": control["prompt"]}], tokenize=False, add_generation_prompt=True
        )
        prompt_ids = gen.tokenizer.encode(rendered, add_special_tokens=False)
        assert rendered == control["rendered_prompt"] == previous["rendered_prompt"]
        assert prompt_ids == control["prompt_token_ids"] == previous["prompt_token_ids"]
        template_sha = hashlib.sha256(gen.tokenizer.chat_template.encode()).hexdigest()
        assert template_sha == controls["provenance"]["template_sha256"]
        result.update(
            {
                "status": "generating",
                "prompt": control["prompt"],
                "rendered_prompt": rendered,
                "prompt_token_ids": prompt_ids,
                "prompt_mode": "chat",
                "template_sha256": template_sha,
                "tokenizer_class": type(gen.tokenizer).__name__,
                "selected_precision": gen.model.precision,
                "cache_context_actual": gen.kv_cache.context,
                "physical_blocks": gen.kv_cache.num_blocks,
                "layer_indices": gen.model.layer_indices,
            }
        )
        save()
        start = time.monotonic()
        tokens = [int(token) for token in gen.generate(prompt_ids, args.max_new_tokens)]
        result.update(
            {
                "status": "completed",
                "generation_wall_s": time.monotonic() - start,
                "num_generated_tokens": len(tokens),
                "selected128_exact_token_match": tokens[:128] == previous["tt"]["token_ids"],
                "serving256_exact_text_match": gen.tokenizer.decode(tokens[:256], skip_special_tokens=True)
                == serving["greedy_completion"],
                "generator_perf": gen.perf,
            }
        )
        text = gen.tokenizer.decode(tokens, skip_special_tokens=False)
        (args.output / "tt_token_ids.json").write_text(json.dumps(tokens) + "\n")
        (args.output / "tt_completion.txt").write_text(text)
        (args.output / "tt_256_completion.txt").write_text(
            gen.tokenizer.decode(tokens[:256], skip_special_tokens=False)
        )
        save()
        print(
            "STANDALONE_HAIKU_COMPLETED",
            json.dumps(
                {
                    key: result[key]
                    for key in ["num_generated_tokens", "selected128_exact_token_match", "serving256_exact_text_match"]
                }
            ),
            flush=True,
        )
        print(text, flush=True)
    except BaseException:
        result.update({"status": "failed", "error": traceback.format_exc()})
        save()
        raise
    finally:
        errors = []
        if gen is not None:
            try:
                gen.teardown()
            except BaseException:
                errors.append(traceback.format_exc())
        if mesh is not None:
            try:
                close_ornith_mesh(mesh)
            except BaseException:
                errors.append(traceback.format_exc())
        result.update(
            {
                "cleanup_completed": not errors,
                "cleanup_errors": errors,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        if errors:
            result["status"] = "failed"
        save()
        if errors:
            raise RuntimeError("Standalone haiku cleanup failed: " + "\n".join(errors))


if __name__ == "__main__":
    main()
