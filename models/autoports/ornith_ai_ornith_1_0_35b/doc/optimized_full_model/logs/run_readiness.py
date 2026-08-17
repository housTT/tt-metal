# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Driver for the shared readiness runners on the 1x4 Blackhole ring.

``models.common.readiness_check``'s CLIs only know the ``N150 / N300 / T3K / TG`` mesh labels, and
their opener passes neither ``l1_small_size`` nor a trace region — neither of which suits a ``1x4``
Blackhole ring running a 40-layer traced decode. Every runner also has a **programmatic** entry
point that takes an already-open ``mesh_device``, so this driver opens the mesh the way the decoder
stage measured, runs the requested checks against it, and writes a machine-readable summary.

    python .../doc/optimized_full_model/logs/run_readiness.py --check prefill teacher
    python .../doc/optimized_full_model/logs/run_readiness.py --check autoregressive --max-new-tokens 128
    python .../doc/optimized_full_model/logs/run_readiness.py --check qualitative
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import close_ornith_mesh, open_ornith_mesh
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model_config import HF_MODEL_ID

MODEL_DIR = Path("models/autoports/ornith_ai_ornith_1_0_35b")
DOC_DIR = MODEL_DIR / "doc" / "optimized_full_model"
DEFAULT_REFERENCE = MODEL_DIR / "readiness_aime24_chat.refpt"
QUALITATIVE_PROMPTS = Path("models/common/readiness_check/vllm_prompts.txt")


def _build_kwargs(args) -> dict:
    kwargs = {"max_batch_size": args.batch, "cache_context": args.cache_context}
    if args.layers:
        kwargs["layer_indices"] = [int(v) for v in args.layers.split(",")]
    if args.sampling_mode:
        kwargs["sampling_mode"] = args.sampling_mode
    if getattr(args, "lm_head_dtype", None):
        import ttnn

        kwargs["lm_head_dtype"] = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "bf16": ttnn.bfloat16}[
            args.lm_head_dtype
        ]
    if getattr(args, "suffix_json", None):
        pass
    return kwargs


JSON_SUFFIX = ""


def _write(name: str, payload) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    if JSON_SUFFIX:
        name = name.replace(".json", f"{JSON_SUFFIX}.json")
    path = DOC_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    logger.info(f"wrote {path}")


def run_prefill(mesh, args) -> dict:
    from models.common.readiness_check.run_prefill_check import run_prefill_check

    started = time.perf_counter()
    per_entry = run_prefill_check(
        model_dir=MODEL_DIR.resolve(),
        reference_path=Path(args.reference).resolve(),
        mesh_device=mesh,
        build_kwargs=_build_kwargs(args),
    )
    return {"per_entry": per_entry, "elapsed_s": time.perf_counter() - started}


def run_teacher(mesh, args) -> dict:
    from models.common.readiness_check.run_teacher_forcing import run_teacher_forcing

    started = time.perf_counter()
    per_entry = run_teacher_forcing(
        model_dir=MODEL_DIR.resolve(),
        reference_path=Path(args.reference).resolve(),
        mesh_device=mesh,
        build_kwargs=_build_kwargs(args),
    )
    return {"per_entry": per_entry, "elapsed_s": time.perf_counter() - started}


def run_autoregressive(mesh, args) -> dict:
    from models.common.readiness_check.run_autoregressive import run_autoregressive as runner

    output_dir = Path(args.output_dir) if args.output_dir else MODEL_DIR / "readiness_autoregressive"
    started = time.perf_counter()
    paths = runner(
        model_dir=MODEL_DIR.resolve(),
        hf_model_id=HF_MODEL_ID,
        prompt_file=Path(args.prompt_file).resolve(),
        mesh_device=mesh,
        output_dir=output_dir.resolve(),
        max_new_tokens=args.max_new_tokens,
        build_kwargs=_build_kwargs(args),
    )
    return {"paths": {k: str(v) for k, v in paths.items()}, "elapsed_s": time.perf_counter() - started}


def run_qualitative(mesh, args) -> dict:
    """The shared qualitative prompt suite, rendered with the checkpoint's own chat template.

    Both sides see exactly the same rendered prompt: the HF control runs first and is freed before
    the TT generator opens the device, mirroring ``run_autoregressive``.
    """
    from transformers import AutoTokenizer

    from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator import build_generator
    from models.common.readiness_check.hf_model import load_hf_reference_model

    prompts = [p.strip() for p in QUALITATIVE_PROMPTS.read_text(encoding="utf-8").split("\n\n") if p.strip()]
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    chat_template = bool(getattr(tokenizer, "chat_template", None))
    rendered = []
    for prompt in prompts:
        if chat_template:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
            )
        else:
            text = prompt
        rendered.append(
            {"prompt": prompt, "rendered": text, "token_ids": tokenizer.encode(text, add_special_tokens=False)}
        )

    results = {
        "prompt_format": {
            "hf_model_id": HF_MODEL_ID,
            "tokenizer_class": type(tokenizer).__name__,
            "chat_template_present": chat_template,
            "prompt_mode": "chat" if chat_template else "completion",
            "rendering": "tokenizer.apply_chat_template(add_generation_prompt=True)",
            "prompt_source": str(QUALITATIVE_PROMPTS),
            "generation": {"greedy": True, "max_new_tokens": args.max_new_tokens},
        },
        "prompts": rendered,
        "hf": [],
        "tt": [],
    }

    if not args.skip_hf:
        logger.info("running the HF control for the qualitative suite")
        model = load_hf_reference_model(HF_MODEL_ID, trust_remote_code=True).eval()
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        for item in rendered:
            ids = torch.tensor([item["token_ids"]], dtype=torch.long)
            with torch.no_grad():
                out = model.generate(
                    ids, max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1, pad_token_id=pad_id
                )
            text = tokenizer.decode(out[0, ids.shape[1] :].tolist(), skip_special_tokens=True)
            results["hf"].append({"prompt": item["prompt"], "completion": text})
            logger.info(f"HF  | {item['prompt'][:48]!r} -> {text[:120]!r}")
        del model
        import gc

        gc.collect()

    generator = build_generator(model_dir=MODEL_DIR.resolve(), mesh_device=mesh, **_build_kwargs(args))
    try:
        for item in rendered:
            generator.reset()
            out = generator.generate(
                prompt_token_ids=item["token_ids"], max_new_tokens=args.max_new_tokens, enable_trace=True
            )
            text = generator.tokenizer.decode(out, skip_special_tokens=True)
            results["tt"].append({"prompt": item["prompt"], "completion": text, "num_tokens": len(out)})
            logger.info(f"TT  | {item['prompt'][:48]!r} -> {text[:120]!r}")
    finally:
        generator.teardown()
    return results


CHECKS = {
    "prefill": run_prefill,
    "teacher": run_teacher,
    "autoregressive": run_autoregressive,
    "qualitative": run_qualitative,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", nargs="+", choices=sorted(CHECKS), required=True)
    ap.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--cache-context", type=int, default=8192)
    ap.add_argument("--layers", default=None, help="comma-separated HF layer indices (default: the whole stack)")
    ap.add_argument("--sampling-mode", default=None, choices=[None, "device", "host"])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--prompt-file", default="models/common/readiness_check/autoregressive_prompt.txt")
    ap.add_argument("--skip-hf", action="store_true", help="qualitative: skip the HF control")
    ap.add_argument("--output-dir", default=None, help="autoregressive: where hf/tt completions go")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--lm-head-dtype", default=None, choices=["bfp8", "bfp4", "bf16"])
    ap.add_argument("--json-suffix", default="", help="appended to the written JSON names, for A/B arms")
    args = ap.parse_args()
    global JSON_SUFFIX
    JSON_SUFFIX = args.json_suffix

    mesh = open_ornith_mesh()
    summary = {}
    try:
        for name in args.check:
            logger.info(f"=== readiness check: {name} ===")
            summary[name] = CHECKS[name](mesh, args)
            _write(f"readiness_{name}{args.suffix}.json", summary[name])
            # Each runner builds its own generator; drop it before the next one so two copies of a
            # 7 GB-per-device weight set never coexist.
            gc.collect()
            ttnn.synchronize_device(mesh)
    finally:
        close_ornith_mesh(mesh)
    print(json.dumps({k: (v.get("per_entry") or "see json") for k, v in summary.items()}, indent=2, default=str))
    print("READINESS_OK")


if __name__ == "__main__":
    main()
