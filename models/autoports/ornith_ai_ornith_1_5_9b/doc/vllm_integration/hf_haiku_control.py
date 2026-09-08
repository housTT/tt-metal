# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only continuation of the exact pinned shared haiku control."""

import argparse
import hashlib
import json
import os
import platform
import shlex
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

DOC = Path(__file__).resolve().parent
MODEL = DOC.parents[1]
REVISION = "489cb97981b8654bcfcf30ce1f94ed1b62e07b53"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_cpu_only():
    assert not any(name == "ttnn" or name.startswith("ttnn.") for name in sys.modules)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("../upstream"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    assert args.max_new_tokens >= 256
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert_cpu_only()
    torch.set_num_threads(8)
    snapshot = args.snapshot.resolve()
    controls = json.loads((DOC / "qualitative_controls.json").read_text())
    control = controls["prompts"][0]
    previous_path = MODEL / "doc/datatype_sweep/qualitative_first8_haiku256_v1/prompt_0/autoregressive_meta.json"
    previous = json.loads(previous_path.read_text())
    assert previous["prompt_token_ids"] == control["prompt_token_ids"]
    assert previous["prompt_text"] == control["prompt"]
    assert previous["max_new_tokens"] == 256
    prior_model = json.loads((MODEL / "readiness_aime24_chat.meta.json").read_text())
    assert prior_model["model_class"] == "Qwen3_5ForCausalLM"
    assert prior_model["model_dtype"] == "torch.bfloat16"
    assert prior_model["torch_version"] == torch.__version__
    assert prior_model["transformers_version"] == transformers.__version__
    metadata_files = sorted((snapshot / ".cache/huggingface/download").glob("*.metadata"))
    assert metadata_files
    snapshot_downloads = {}
    for path in metadata_files:
        revision, etag, *_ = path.read_text().splitlines()
        assert revision == REVISION, (path, revision)
        snapshot_downloads[path.name] = {"revision": revision, "etag": etag}
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(meminfo["MemAvailable"].split()[0]) * 1024
    cgroup_max = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    cgroup_current = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
    headroom = available if cgroup_max == "max" else min(available, int(cgroup_max) - cgroup_current)
    assert headroom > 36 * 1024**3, f"Insufficient CPU memory headroom: {headroom}"
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "status": "loading",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": "HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 " + shlex.join([sys.executable, *sys.argv]),
        "hf_model_id": "ornith-ai/Ornith-1.5-9B",
        "revision": REVISION,
        "snapshot": str(snapshot),
        "snapshot_download_metadata": snapshot_downloads,
        "snapshot_metadata_sha256": {
            name: sha(snapshot / name)
            for name in [
                "config.json",
                "chat_template.jinja",
                "tokenizer.json",
                "tokenizer_config.json",
                "model.safetensors.index.json",
            ]
        },
        "script_sha256": sha(__file__),
        "previous_hf256_metadata": str(previous_path),
        "previous_hf256_metadata_sha256": sha(previous_path),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "device": "cpu",
        "dtype_requested": "torch.bfloat16",
        "memory_before_load": {
            "available_bytes": available,
            "cgroup_max": cgroup_max,
            "cgroup_current_bytes": cgroup_current,
        },
    }

    def save():
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")

    save()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": control["prompt"]}], tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
    assert rendered == previous["rendered_prompt"] == control["rendered_prompt"]
    assert prompt_ids == previous["prompt_token_ids"] == control["prompt_token_ids"]
    template_sha = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    assert template_sha == controls["provenance"]["template_sha256"]
    metadata.update(
        {
            "prompt": control["prompt"],
            "rendered_prompt": rendered,
            "prompt_token_ids": prompt_ids,
            "prompt_mode": "chat",
            "template_sha256": template_sha,
            "tokenizer_class": type(tokenizer).__name__,
        }
    )
    started_load = time.monotonic()
    model, diagnostics = AutoModelForCausalLM.from_pretrained(
        snapshot, trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16, output_loading_info=True
    )
    model = model.eval().to("cpu")
    metadata.update(
        {
            "load_s": time.monotonic() - started_load,
            "model_class": type(model).__name__,
            "model_config_class": type(model.config).__name__,
            "model_dtype": str(model.dtype),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "parameter_dtypes": dict(Counter(str(p.dtype) for p in model.parameters())),
            "loading_diagnostics": diagnostics,
            "generation_config": model.generation_config.to_dict(),
            "attention_implementation": model.config._attn_implementation,
        }
    )
    save()
    assert type(model).__name__ == prior_model["model_class"]
    assert str(model.dtype) == prior_model["model_dtype"]
    assert metadata["parameter_count"] == prior_model["parameter_count"]
    assert all(p.device.type == "cpu" and p.dtype == torch.bfloat16 for p in model.parameters())
    assert not any(diagnostics.get(key) for key in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
    assert_cpu_only()
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        eos = tokenizer.eos_token_id
        pad_id = eos[0] if isinstance(eos, (list, tuple)) else eos
    # Match run_autoregressive._hf_generate_greedy, including inherited cache/EOS settings.
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "pad_token_id": pad_id,
    }
    metadata.update({"status": "generating", "generation_kwargs": generation_kwargs})
    save()
    print(
        "HF_CONTROL_LOADED",
        json.dumps({key: metadata[key] for key in ["model_class", "model_dtype", "load_s", "parameter_count"]}),
        flush=True,
    )
    started_generation = time.monotonic()

    class Progress(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            generated = input_ids[0, len(prompt_ids) :].tolist()
            if len(generated) % 32 == 0:
                metadata.update(
                    {"generated_so_far": len(generated), "generation_s": time.monotonic() - started_generation}
                )
                (args.output / "partial_completion.txt").write_text(
                    tokenizer.decode(generated, skip_special_tokens=False)
                )
                save()
                print("HF_CONTROL_PROGRESS", len(generated), round(metadata["generation_s"], 3), flush=True)
            if len(generated) == 256:
                metadata["hf256_exact_previous_token_match"] = generated == previous["hf"]["token_ids"]
                (args.output / "hf_256_token_ids.json").write_text(json.dumps(generated) + "\n")
                (args.output / "hf_256_completion.txt").write_text(
                    tokenizer.decode(generated, skip_special_tokens=False)
                )
                save()
            return False

    with torch.no_grad():
        output = model.generate(
            torch.tensor([prompt_ids], dtype=torch.long, device="cpu"),
            **generation_kwargs,
            stopping_criteria=StoppingCriteriaList([Progress()]),
        )
    generated = output[0, len(prompt_ids) :].tolist()
    text = tokenizer.decode(generated, skip_special_tokens=False)
    metadata.update(
        {
            "status": "completed",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "generation_s": time.monotonic() - started_generation,
            "num_generated_tokens": len(generated),
            "hf256_exact_previous_token_match": generated[:256] == previous["hf"]["token_ids"],
            "hf128_exact_saved_text_match": tokenizer.decode(generated[:128], skip_special_tokens=True)
            == control["hf_completion"],
            "ttnn_imported": False,
        }
    )
    assert_cpu_only()
    (args.output / "hf_token_ids.json").write_text(json.dumps(generated) + "\n")
    (args.output / "hf_completion.txt").write_text(text)
    save()
    print(
        "HF_CONTROL_COMPLETED",
        json.dumps(
            {
                key: metadata[key]
                for key in [
                    "num_generated_tokens",
                    "generation_s",
                    "hf256_exact_previous_token_match",
                    "hf128_exact_saved_text_match",
                ]
            }
        ),
        flush=True,
    )
    print(text, flush=True)


if __name__ == "__main__":
    main()
