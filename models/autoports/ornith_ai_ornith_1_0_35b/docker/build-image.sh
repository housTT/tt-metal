#!/usr/bin/env bash
set -Eeuo pipefail

readonly TT_METAL_REVISION="b33529c1a9575b134b063ebdfe9de068c88554a9"
readonly VLLM_REVISION="a887998646dc4e6f192bce8d485bf89f4596ca2f"
readonly DEFAULT_IMAGE="ghcr.io/houstt/ornith-1.0-35b:qb2"
readonly IMMUTABLE_IMAGE="ghcr.io/houstt/ornith-1.0-35b:ttmetal-b33529c-vllm-a887998"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tt_metal_root="${TT_METAL_SOURCE:-$(git -C "${script_dir}" rev-parse --show-toplevel)}"
workspace_root="$(dirname -- "${tt_metal_root}")"
vllm_root="${VLLM_SOURCE:-${workspace_root}/vllm}"
image="${DEFAULT_IMAGE}"
publish=0

usage() {
    printf 'Usage: %s [--push] [--image IMAGE]\n' "${0##*/}"
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

for repo in "${tt_metal_root}" "${vllm_root}"; do
    git -C "${repo}" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
        printf 'Not a Git checkout: %s\n' "${repo}" >&2
        exit 1
    }
done

git -C "${tt_metal_root}" cat-file -e "${TT_METAL_REVISION}^{commit}"
git -C "${vllm_root}" cat-file -e "${VLLM_REVISION}^{commit}"

if [[ "$(git -C "${tt_metal_root}" rev-parse HEAD)" != "${TT_METAL_REVISION}" ]]; then
    printf 'tt-metal must be checked out at %s\n' "${TT_METAL_REVISION}" >&2
    exit 1
fi
if [[ "$(git -C "${vllm_root}" rev-parse HEAD)" != "${VLLM_REVISION}" ]]; then
    printf 'vLLM must be checked out at %s\n' "${VLLM_REVISION}" >&2
    exit 1
fi

context_root="$(mktemp -d /tmp/ornith-1.0-35b-build.XXXXXX)"
cleanup() {
    case "${context_root}" in
        /tmp/ornith-1.0-35b-build.*) rm -rf -- "${context_root}" ;;
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

    # Fetch packfiles and checkout indexes contain nondeterministic ordering and
    # filesystem timestamps. Normalize both so exact-revision source contexts
    # are byte-identical and BuildKit can reuse the expensive native layer.
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

# Recreate each checked-out tt-metal submodule from the local worktree. This
# avoids both network clones and any ignored/untracked host files.
while IFS= read -r submodule_line; do
    [[ -n "${submodule_line}" ]] || continue
    if [[ "${submodule_line:0:1}" != " " ]]; then
        submodule_path="$(cut -d' ' -f3 <<<"${submodule_line}")"
        printf 'Submodule is not at the recorded revision: %s\n' "${submodule_path}" >&2
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

printf 'Building %s from local revisions:\n  tt-metal %s\n  vLLM    %s\n' \
    "${image}" "${TT_METAL_REVISION}" "${VLLM_REVISION}"

docker buildx build \
    --progress=plain \
    --platform linux/amd64 \
    --build-context "ttmetal=${context_root}/tt-metal" \
    --build-context "vllm=${context_root}/vllm" \
    --build-arg "TT_METAL_REVISION=${TT_METAL_REVISION}" \
    --build-arg "VLLM_REVISION=${VLLM_REVISION}" \
    --tag "${image}" \
    --tag "${IMMUTABLE_IMAGE}" \
    "${output_args[@]}" \
    "${script_dir}"
