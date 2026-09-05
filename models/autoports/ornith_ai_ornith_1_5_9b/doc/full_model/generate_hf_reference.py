"""Run the common readiness generator with a pinned local CPU control and provenance."""

import hashlib
import importlib
import json
import os
import platform
import shlex
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer

readiness = importlib.import_module("models.common.readiness_check.generate")
from models.common.readiness_check.schema import load_reference

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = Path("/home/hous/dev/ornith-1.5-9b/upstream")
OUT = ROOT / "readiness_aime24_chat.refpt"
META = ROOT / "readiness_aime24_chat.meta.json"
STAGE = ROOT / "doc/full_model"
assert os.environ.get("HF_HUB_OFFLINE") == "1"
torch.set_num_threads(8)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


metadata = {
    "hf_model_id": "ornith-ai/Ornith-1.5-9B",
    "revision": "489cb97981b8654bcfcf30ce1f94ed1b62e07b53",
    "snapshot_path": str(SNAPSHOT),
    "prompt_source": "aime24",
    "aime24_prompt_index": 0,
    "chat_template": True,
    "prompt_mode": "chat",
    "generation_length_requested": 100,
    "top_k": 100,
    "generation_command": "HF_HUB_OFFLINE=1 TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache OMP_NUM_THREADS=8 PYTHONPATH=. "
    + shlex.join([sys.executable, str(Path(__file__).resolve())]),
    "generate_reference_arguments": {
        "hf_model_id": str(SNAPSHOT),
        "prompt_source": "aime24",
        "aime24_prompt_index": 0,
        "chat_template": True,
        "gen_len": 100,
        "top_k": 100,
        "device": "cpu",
        "output_path": str(OUT),
    },
    "transformers_version": transformers.__version__,
    "torch_version": torch.__version__,
    "python_version": platform.python_version(),
    "device": "cpu",
    "torch_num_threads": torch.get_num_threads(),
    "source_hashes": {
        str(path): sha(path)
        for path in [Path(readiness.__file__), Path(__file__), readiness.DEFAULT_AIME24_PROMPTS_FILE]
    },
    "snapshot_metadata_hashes": {
        name: sha(SNAPSHOT / name)
        for name in [
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
        ]
        if (SNAPSHOT / name).exists()
    },
}
tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=True)
assert tokenizer.chat_template
prompt = readiness._load_aime24_prompt(readiness.DEFAULT_AIME24_PROMPTS_FILE, 0)
rendered = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
)
ids = readiness._chat_or_plain_prompt_tokens(tokenizer, prompt, chat_template=True)
assert tokenizer.encode(rendered, add_special_tokens=False) == ids
metadata.update(
    {
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_identity": str(SNAPSHOT),
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "user_prompt": prompt,
        "rendered_prompt": rendered,
        "prompt_token_ids": ids,
        "prompt_length": len(ids),
        "generation_parameters": {"do_sample": False, "max_new_tokens": 100, "use_cache": True},
    }
)
(STAGE / "aime24_chat_prompt.txt").write_text(rendered)
META.write_text(json.dumps(metadata, indent=2) + "\n")

# Observe the same loader selected by the common generator; do not substitute a model.
original_loader = readiness.AutoModelForCausalLM.from_pretrained


def observed_loader(*args, **kwargs):
    model, info = original_loader(*args, **kwargs, output_loading_info=True)
    metadata["model_class"] = type(model).__name__
    metadata["model_dtype"] = str(model.dtype)
    metadata["model_config_class"] = type(model.config).__name__
    metadata["parameter_count"] = sum(p.numel() for p in model.parameters())
    metadata["loading_diagnostics"] = info
    metadata["generation_stop_ids"] = readiness._generation_stop_ids(tokenizer, model)
    META.write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    print(
        "HF_LOAD_DIAGNOSTICS",
        json.dumps(
            {k: metadata[k] for k in ["model_class", "model_dtype", "parameter_count", "loading_diagnostics"]},
            default=str,
        ),
        flush=True,
    )
    assert not info.get("missing_keys"), info
    assert not info.get("mismatched_keys"), info
    assert not info.get("error_msgs"), info
    return model


readiness.AutoModelForCausalLM.from_pretrained = observed_loader
start = time.monotonic()
readiness.generate_reference(
    hf_model_id=str(SNAPSHOT),
    prompt_source="aime24",
    aime24_prompt_index=0,
    chat_template=True,
    gen_len=100,
    top_k=100,
    device=torch.device("cpu"),
    output_path=OUT,
)
ref = load_reference(OUT)
assert ref.k == 100 and len(ref.entries) == 1
entry = ref.entries[0]
assert entry.num_generated == 100, entry.num_generated
assert entry.prompt_tokens[0].tolist() == ids
completion = tokenizer.decode(entry.generated_tokens[0].tolist(), skip_special_tokens=False)
(STAGE / "aime24_hf_completion.txt").write_text(completion)
metadata.update(
    {
        "generation_length_actual": entry.num_generated,
        "generated_token_ids": entry.generated_tokens[0].tolist(),
        "completion": completion,
        "reference_sha256": sha(OUT),
        "wall_seconds": time.monotonic() - start,
        "ttnn_imported": "ttnn" in sys.modules,
        "status": "complete",
    }
)
assert not metadata["ttnn_imported"]
META.write_text(json.dumps(metadata, indent=2, default=str) + "\n")
print("HF_COMPLETION", completion, sep="\n", flush=True)
print("REFERENCE_COMPLETE", OUT, "tokens", entry.num_generated, "seconds", metadata["wall_seconds"], flush=True)
