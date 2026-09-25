#!/usr/bin/env bash
set -Eeuo pipefail

readonly MODEL_ID="meta-models/Muse-Glimmer-30B"
readonly MODEL_REVISION="f84ecc3a0ea984a4c04542a84269e3d065350a6e"
readonly MODEL_CACHE_NAME="models--meta-models--Muse-Glimmer-30B"
readonly MODEL_DIR="${HF_HUB_CACHE}/${MODEL_CACHE_NAME}/snapshots/${MODEL_REVISION}"
readonly EXPECTED_KMD_VERSION="2.11.0"
export MODEL_DIR

fail() {
    printf 'Muse-Glimmer-30B: %s\n' "$*" >&2
    exit 1
}

if [[ ! -r "${MODEL_DIR}/model.safetensors.index.json" ]]; then
    fail "the pinned weights are absent at ${MODEL_DIR}. Run: hf download ${MODEL_ID} --revision ${MODEL_REVISION}"
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
if len(shards) != 2:
    raise SystemExit(
        f"Muse-Glimmer-30B: expected exactly 2 indexed weight shards, found {len(shards)}"
    )
missing.extend(name for name in shards if not (model_dir / name).is_file())
if missing:
    rendered = "\n  - ".join(missing)
    raise SystemExit(f"Muse-Glimmer-30B: the pinned HF snapshot is incomplete:\n  - {rendered}")
print(f"Muse-Glimmer-30B: found {len(shards)} weight shards in the default Hugging Face cache")
PY

if [[ ! -d /dev/tenstorrent ]]; then
    fail "/dev/tenstorrent is unavailable; add --device /dev/tenstorrent to docker run"
fi

device_count="$(find /dev/tenstorrent -maxdepth 1 -type c -printf '.' | wc -c)"
if [[ "${device_count}" -ne 4 ]]; then
    fail "P300x2 requires exactly four visible Tenstorrent devices; found ${device_count}"
fi

sysfs_count=0
for sysfs_device in /sys/class/tenstorrent/tenstorrent\!*; do
    [[ -e "${sysfs_device}" ]] || continue
    ((sysfs_count += 1))
    pci_vendor="$(<"${sysfs_device}/device/vendor")"
    pci_device="$(<"${sysfs_device}/device/device")"
    if [[ "${pci_vendor}" != "0x1e52" || "${pci_device}" != "0xb140" ]]; then
        fail "expected Blackhole PCI ID 1e52:b140; found ${pci_vendor}:${pci_device} at ${sysfs_device}"
    fi
done
if [[ "${sysfs_count}" -ne 4 ]]; then
    fail "expected four Blackhole sysfs devices; found ${sysfs_count}"
fi

if [[ ! -d /dev/hugepages ]] || [[ "$(stat -f -c '%T' /dev/hugepages)" != "hugetlbfs" ]]; then
    fail "/dev/hugepages is not a hugetlbfs mount; bind-mount it into the container"
fi

if [[ -r /sys/module/tenstorrent/version ]]; then
    kmd_version="$(</sys/module/tenstorrent/version)"
else
    kmd_version=""
fi
if [[ "${kmd_version}" != "${EXPECTED_KMD_VERSION}" ]]; then
    fail "this image was validated with Tenstorrent KMD ${EXPECTED_KMD_VERSION}; found ${kmd_version:-unknown}"
fi

if (( $# > 0 )); then
    exec "$@"
fi

export HF_MODEL="${MODEL_DIR}"
export MODEL_WEIGHTS_DIR="${MODEL_DIR}"

printf 'Muse-Glimmer-30B: starting %s on http://0.0.0.0:8000\n' "${MODEL_ID}"
printf 'Muse-Glimmer-30B: first launch compiles kernels; the named Docker volume preserves them\n'

exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_DIR}" \
    --served-model-name "${MODEL_ID}" \
    --host 0.0.0.0 \
    --port 8000 \
    --block-size 64 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 131072 \
    --max-model-len 131072 \
    --max-log-len 32 \
    --seed 9472 \
    --additional-config '{"tt":{"sample_on_device_mode":"all","trace_region_size":400000000,"fabric_config":"FABRIC_1D_RING","fabric_packet_payload_bytes":8192,"l1_small_size":6144,"trace_mode":"decode_only"}}'
