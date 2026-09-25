#!/usr/bin/env bash
set -Eeuo pipefail

readonly TT_METAL_REVISION="0dd37ce6ee33826ebb8ce23a5d83a45bca7d6b29"
readonly VLLM_REVISION="ee0da84ab9e04ac7610e28580af62c365e898389"
readonly VLLM_TT_PLUGIN_REVISION="106744c01de96825ed8c81226f4fa043e8e929f4"
readonly TRANSFORMERS_REVISION="5eddc12edfaf8cafde8c9bae4ccb12f8a139b4f9"
readonly DEFAULT_IMAGE="muse-glimmer-30b:qb2"
readonly IMMUTABLE_TAG="ttmetal-0dd37ce-vllm-ee0da84-plugin-106744c"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tt_metal_root="${TT_METAL_SOURCE:-$(git -C "${script_dir}" rev-parse --show-toplevel)}"
workspace_root="$(dirname -- "${tt_metal_root}")"
vllm_root="${VLLM_SOURCE:-${workspace_root}/vllm}"
plugin_root="${VLLM_TT_PLUGIN_SOURCE:-${workspace_root}/vllm-tt-plugin}"
transformers_root="${TRANSFORMERS_SOURCE:-${workspace_root}/transformers}"
image="${DEFAULT_IMAGE}"
publish=0

usage() {
    printf 'Usage: %s [--push] [--image IMAGE:TAG]\n' "${0##*/}"
}

while (( $# > 0 )); do
    case "$1" in
        --push)
            publish=1
            shift
            ;;
        --image)
            (( $# >= 2 )) || { usage >&2; exit 2; }
            image="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "${image}" != *:* ]]; then
    printf 'Image must include a tag: %s\n' "${image}" >&2
    exit 2
fi
image_repository="${image%:*}"
immutable_image="${image_repository}:${IMMUTABLE_TAG}"

declare -A revisions=(
    ["${tt_metal_root}"]="${TT_METAL_REVISION}"
    ["${vllm_root}"]="${VLLM_REVISION}"
    ["${plugin_root}"]="${VLLM_TT_PLUGIN_REVISION}"
    ["${transformers_root}"]="${TRANSFORMERS_REVISION}"
)
for repo in "${!revisions[@]}"; do
    git -C "${repo}" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
        printf 'Not a Git checkout: %s\n' "${repo}" >&2
        exit 1
    }
    revision="${revisions[${repo}]}"
    git -C "${repo}" cat-file -e "${revision}^{commit}"
    if [[ "$(git -C "${repo}" rev-parse HEAD)" != "${revision}" ]]; then
        printf '%s must be checked out at %s\n' "${repo}" "${revision}" >&2
        exit 1
    fi
done

context_root="$(mktemp -d /tmp/muse-glimmer-30b-build.XXXXXX)"
cleanup() {
    case "${context_root}" in
        /tmp/muse-glimmer-30b-build.*) rm -rf -- "${context_root}" ;;
        *) printf 'Refusing to remove unexpected build context: %s\n' "${context_root}" >&2 ;;
    esac
}
trap cleanup EXIT

clone_local_revision() {
    local source_repo="$1"
    local destination="$2"
    local revision="$3"
    git init --quiet "${destination}"
    git -C "${destination}" remote add source "file://${source_repo}"
    git -C "${destination}" fetch --quiet --depth 1 --no-tags source "${revision}"
    git -C "${destination}" -c advice.detachedHead=false checkout --quiet --detach FETCH_HEAD
    git -C "${destination}" remote remove source

    git -C "${destination}" -c pack.threads=1 repack -qadf --window=0 --depth=0
    find "${destination}/.git/logs" -depth -delete
    for volatile_file in FETCH_HEAD ORIG_HEAD index; do
        volatile_path="${destination}/.git/${volatile_file}"
        [[ ! -e "${volatile_path}" ]] || unlink "${volatile_path}"
    done
    git -C "${destination}" read-tree HEAD
}

clone_local_revision "${tt_metal_root}" "${context_root}/tt-metal" "${TT_METAL_REVISION}"
clone_local_revision "${vllm_root}" "${context_root}/vllm" "${VLLM_REVISION}"
clone_local_revision "${plugin_root}" "${context_root}/vllm-tt-plugin" "${VLLM_TT_PLUGIN_REVISION}"
clone_local_revision "${transformers_root}" "${context_root}/transformers" "${TRANSFORMERS_REVISION}"

while IFS= read -r submodule_line; do
    [[ -n "${submodule_line}" ]] || continue
    if [[ "${submodule_line:0:1}" != " " ]]; then
        submodule_path="$(cut -d' ' -f3 <<<"${submodule_line}")"
        printf 'Submodule is not at its recorded revision: %s\n' "${submodule_path}" >&2
        exit 1
    fi
    read -r submodule_revision submodule_path _ <<<"${submodule_line:1}"
    source_submodule="${tt_metal_root}/${submodule_path}"
    destination_submodule="${context_root}/tt-metal/${submodule_path}"
    mkdir -p "$(dirname -- "${destination_submodule}")"
    clone_local_revision "${source_submodule}" "${destination_submodule}" "${submodule_revision}"
done < <(git -C "${tt_metal_root}" submodule status --recursive)

output_args=(--load)
if (( publish )); then
    output_args=(--push --provenance=mode=max --sbom=true)
fi

printf 'Building %s and %s from clean local revisions:\n' "${image}" "${immutable_image}"
printf '  tt-metal        %s\n' "${TT_METAL_REVISION}"
printf '  vLLM            %s\n' "${VLLM_REVISION}"
printf '  vLLM TT plugin  %s\n' "${VLLM_TT_PLUGIN_REVISION}"
printf '  Transformers    %s\n' "${TRANSFORMERS_REVISION}"

docker buildx build \
    --progress=plain \
    --platform linux/amd64 \
    --build-context "ttmetal=${context_root}/tt-metal" \
    --build-context "vllm=${context_root}/vllm" \
    --build-context "vllm_tt_plugin=${context_root}/vllm-tt-plugin" \
    --build-context "transformers=${context_root}/transformers" \
    --build-arg "TT_METAL_REVISION=${TT_METAL_REVISION}" \
    --build-arg "VLLM_REVISION=${VLLM_REVISION}" \
    --build-arg "VLLM_TT_PLUGIN_REVISION=${VLLM_TT_PLUGIN_REVISION}" \
    --build-arg "TRANSFORMERS_REVISION=${TRANSFORMERS_REVISION}" \
    --tag "${image}" \
    --tag "${immutable_image}" \
    "${output_args[@]}" \
    "${script_dir}"
