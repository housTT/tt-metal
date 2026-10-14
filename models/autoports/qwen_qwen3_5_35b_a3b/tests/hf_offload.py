# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch

MAX_CPU_GIB = int(os.environ.get("QWEN35_HF_MAX_CPU_GIB", "34"))
OFFLOAD_DIR = Path(os.environ.get("QWEN35_HF_OFFLOAD_DIR", "/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/hf_offload"))


class _OffloadedLoader:
    def __init__(self, original):
        self.original = original

    def from_pretrained(self, *args, **kwargs):
        kwargs.setdefault("dtype", torch.bfloat16)
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {"cpu": f"{MAX_CPU_GIB}GiB"}
        kwargs["offload_folder"] = str(OFFLOAD_DIR)
        model = self.original.from_pretrained(*args, **kwargs)
        model.to = lambda *a, **k: model
        return model


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _token_list(original):
    def tokens(tokenizer, prompt_text, *, chat_template):
        if not chat_template:
            return original(tokenizer, prompt_text, chat_template=chat_template)
        encoded = tokenizer.apply_chat_template([{"role": "user", "content": prompt_text}],
                                                add_generation_prompt=True, tokenize=True)
        ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        return [int(token_id) for token_id in ids]
    return tokens


def main():
    runner, *argv = sys.argv[1:]
    module = importlib.import_module(runner)
    offload = os.environ.get("QWEN35_HF_OFFLOAD", "1") == "1"
    if offload:
        module.AutoModelForCausalLM = _OffloadedLoader(module.AutoModelForCausalLM)
    if hasattr(module, "_chat_or_plain_prompt_tokens"):
        module._chat_or_plain_prompt_tokens = _token_list(module._chat_or_plain_prompt_tokens)
    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(argv[argv.index("--hf-model") + 1]) if "--hf-model" in argv else None
    meta = dict(runner=runner, argv=argv, offload=offload, max_cpu_gib=MAX_CPU_GIB, offload_dir=str(OFFLOAD_DIR),
                torch=torch.__version__, python=platform.python_version(), started=time.time())
    if checkpoint is not None and checkpoint.is_dir():
        meta["tokenizer_sha256"] = {name: _sha256(checkpoint / name)
                                    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
                                    if (checkpoint / name).exists()}
        meta["config_sha256"] = _sha256(checkpoint / "config.json")
    output = Path(argv[argv.index("--output") + 1]) if "--output" in argv else None
    sys.argv = [runner] + argv
    module._main() if hasattr(module, "_main") else module.main()
    meta["finished"] = time.time()
    if output is not None:
        output.with_suffix(output.suffix + ".meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
