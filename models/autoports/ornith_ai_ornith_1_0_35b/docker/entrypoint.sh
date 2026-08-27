#!/usr/bin/env bash
set -Eeuo pipefail

readonly MODEL_ID="ornith-ai/Ornith-1.0-35B"
readonly MODEL_REVISION="5df2ed3f675c7beaa490328cc70bb573b65fb660"
readonly MODEL_CACHE_NAME="models--ornith-ai--Ornith-1.0-35B"
readonly MODEL_DIR="${HF_HUB_CACHE}/${MODEL_CACHE_NAME}/snapshots/${MODEL_REVISION}"
export MODEL_DIR

fail() {
    printf 'Ornith-1.0-35b: %s\n' "$*" >&2
    exit 1
}

if [[ ! -r "${MODEL_DIR}/model.safetensors.index.json" ]]; then
    fail "the pinned weights are not present at ${MODEL_DIR}. Run: hf download ${MODEL_ID} --revision ${MODEL_REVISION}"
fi

python - <<'PY'
import json
import os
from pathlib import Path

model_dir = Path(os.environ["MODEL_DIR"])
required_metadata = ("config.json", "tokenizer.json", "tokenizer_config.json")
missing = [name for name in required_metadata if not (model_dir / name).is_file()]
with (model_dir / "model.safetensors.index.json").open() as handle:
    index = json.load(handle)
shards = sorted(set(index.get("weight_map", {}).values()))
if not shards:
    missing.append("model shards listed by model.safetensors.index.json")
missing.extend(name for name in shards if not (model_dir / name).is_file())
if missing:
    rendered = "\n  - ".join(missing)
    raise SystemExit(f"Ornith-1.0-35b: the pinned HF snapshot is incomplete:\n  - {rendered}")
print(f"Ornith-1.0-35b: found {len(shards)} weight shards in the default Hugging Face cache")
PY

if [[ ! -d /dev/tenstorrent ]]; then
    fail "/dev/tenstorrent is unavailable; add --device /dev/tenstorrent to docker run"
fi

device_count="$(find /dev/tenstorrent -maxdepth 1 -type c -printf '.' | wc -c)"
if [[ "${device_count}" -ne 4 ]]; then
    fail "QuietBox2 requires exactly four visible Tenstorrent devices; found ${device_count}"
fi

if [[ ! -d /dev/hugepages-1G ]] || [[ "$(stat -f -c '%T' /dev/hugepages-1G)" != "hugetlbfs" ]]; then
    fail "/dev/hugepages-1G is not a hugetlbfs mount; bind-mount it into the container"
fi

if (( $# > 0 )); then
    exec "$@"
fi

export HF_MODEL="${MODEL_DIR}"
export MODEL_WEIGHTS_DIR="${MODEL_DIR}"

printf 'Ornith-1.0-35b: starting %s on http://0.0.0.0:7890\n' "${MODEL_ID}"
printf 'Ornith-1.0-35b: first launch compiles kernels; the named Docker volume preserves them\n'

exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_DIR}" \
    --served-model-name "${MODEL_ID}" \
    --host 0.0.0.0 \
    --port 7890 \
    --block-size 64 \
    --max-num-seqs 1 \
    --max-model-len 262144 \
    --additional-config '{"tt":{"sample_on_device_mode":"all","trace_region_size":200000000,"l1_small_size":32768,"fabric_config":"FABRIC_1D_RING","fabric_router_max_packet_bytes":8192}}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml
