#!/usr/bin/env bash
set -Eeuo pipefail

readonly IMAGE="${1:-ghcr.io/houstt/ornith-1.0-35b:qb2}"
readonly WEIGHTS_REVISION="5df2ed3f675c7beaa490328cc70bb573b65fb660"
readonly SNAPSHOT="${HOME}/.cache/huggingface/hub/models--ornith-ai--Ornith-1.0-35B/snapshots/${WEIGHTS_REVISION}"

docker image inspect "${IMAGE}" >/dev/null

docker run --rm -i \
    --device /dev/tenstorrent \
    --ipc=host \
    --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
    --entrypoint /opt/venv/bin/python \
    "${IMAGE}" - <<'PY'
from pathlib import Path

import torch
import ttnn
import vllm
import vllm_tt_plugin

checks = {
    "ttnn": (Path(ttnn.__file__).resolve(), Path("/opt/tt-metal")),
    "vllm": (Path(vllm.__file__).resolve(), Path("/opt/vllm")),
    "vllm_tt_plugin": (Path(vllm_tt_plugin.__file__).resolve(), Path("/opt/vllm")),
}
for package, (origin, root) in checks.items():
    assert origin.is_relative_to(root), f"{package}: unexpected origin {origin}"
assert torch.version.cuda is None, torch.version.cuda
assert hasattr(ttnn.experimental.deepseek_prefill, "topk_routed_expert_moe")
print({package: str(origin) for package, (origin, _) in checks.items()})
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
PY

test -r "${SNAPSHOT}/model.safetensors.index.json" || {
    printf 'Weights are missing. Run: hf download ornith-ai/Ornith-1.0-35B --revision %s\n' "${WEIGHTS_REVISION}" >&2
    exit 1
}

printf 'Image and native QB2 import checks passed. Start the server with the documented docker run command.\n'
