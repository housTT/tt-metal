#!/usr/bin/env bash
set -Eeuo pipefail

readonly IMAGE="${1:-muse-glimmer-30b:qb2}"
readonly TT_METAL_REVISION="0dd37ce6ee33826ebb8ce23a5d83a45bca7d6b29"
readonly VLLM_REVISION="ee0da84ab9e04ac7610e28580af62c365e898389"
readonly VLLM_TT_PLUGIN_REVISION="106744c01de96825ed8c81226f4fa043e8e929f4"
readonly TRANSFORMERS_REVISION="5eddc12edfaf8cafde8c9bae4ccb12f8a139b4f9"
readonly WEIGHTS_REVISION="f84ecc3a0ea984a4c04542a84269e3d065350a6e"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

docker image inspect "${IMAGE}" >/dev/null

python3 - "${IMAGE}" <<'PY'
import json
import subprocess
import sys

image = sys.argv[1]
data = json.loads(subprocess.check_output(["docker", "image", "inspect", image]))[0]
config = data["Config"]
labels = config["Labels"]
expected = {
    "ai.muse-glimmer.tt-metal.revision": "0dd37ce6ee33826ebb8ce23a5d83a45bca7d6b29",
    "ai.muse-glimmer.vllm.revision": "ee0da84ab9e04ac7610e28580af62c365e898389",
    "ai.muse-glimmer.vllm-tt-plugin.revision": "106744c01de96825ed8c81226f4fa043e8e929f4",
    "ai.muse-glimmer.transformers.revision": "5eddc12edfaf8cafde8c9bae4ccb12f8a139b4f9",
    "ai.muse-glimmer.weights.revision": "f84ecc3a0ea984a4c04542a84269e3d065350a6e",
}
for key, value in expected.items():
    assert labels.get(key) == value, (key, labels.get(key), value)
assert config["User"] == "muse", config["User"]
assert config["Entrypoint"] == ["/usr/bin/tini", "--", "/usr/local/bin/muse-glimmer-30b"]
assert "8000/tcp" in config["ExposedPorts"]
assert config["Healthcheck"]["Test"][0] == "CMD-SHELL"
print("OCI config and exact provenance labels passed")
PY

docker run --rm -i \
    --entrypoint /opt/venv/bin/python \
    "${IMAGE}" - <<'PY'
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
import json
import torch

expected = {
    "ttnn": Path("/opt/tt-metal"),
    "vllm": Path("/opt/vllm"),
    "vllm_tt_plugin": Path("/opt/vllm-tt-plugin"),
    "transformers": Path("/opt/transformers"),
}
for package, root in expected.items():
    spec = find_spec(package)
    assert spec is not None and spec.origin is not None, package
    origin = Path(spec.origin).resolve()
    assert origin.is_relative_to(root), (package, origin, root)
assert find_spec("transformers.models.muse_glimmer") is not None
manifest = json.loads(Path("/usr/share/muse-glimmer-30b/source-revisions.json").read_text())
assert manifest == {
    "tt_metal": "0dd37ce6ee33826ebb8ce23a5d83a45bca7d6b29",
    "vllm": "ee0da84ab9e04ac7610e28580af62c365e898389",
    "vllm_tt_plugin": "106744c01de96825ed8c81226f4fa043e8e929f4",
    "transformers": "5eddc12edfaf8cafde8c9bae4ccb12f8a139b4f9",
    "weights": "f84ecc3a0ea984a4c04542a84269e3d065350a6e",
}
assert Path("/usr/share/muse-glimmer-30b/python-packages.txt").stat().st_size > 0
assert torch.__version__ == "2.11.0+cpu", torch.__version__
assert torch.version.cuda is None, torch.version.cuda
print({name: version(name.replace("_", "-")) for name in expected})
print("Package origins, CPU PyTorch, Muse-Glimmer registration, and embedded manifests passed")
PY

expected_packages="$(grep -Ev '^(#|$)' "${script_dir}/runtime-lock.txt" | sort)"
actual_packages="$(docker run --rm --entrypoint /bin/bash "${IMAGE}" -c '
    sed \
        -e "s|^-e file:///opt/transformers$|transformers==5.15.0|" \
        -e "s|^-e file:///opt/tt-metal$|ttnn==0.1.dev1+g0dd37ce6ee|" \
        -e "s|^-e file:///opt/vllm$|vllm==0.24.0+gee0da84ab9.empty|" \
        -e "s|^-e file:///opt/vllm-tt-plugin$|vllm-tt-plugin==0.1.0|" \
        /usr/share/muse-glimmer-30b/python-packages.txt | sort
')"
if ! diff -u <(printf '%s\n' "${expected_packages}") <(printf '%s\n' "${actual_packages}"); then
    printf 'Embedded Python environment does not match runtime-lock.txt\n' >&2
    exit 1
fi

history="$(docker history --no-trunc --format '{{.CreatedBy}}' "${IMAGE}")"
if grep -Eiq '(^|[^[:alnum:]_])(hf_token|hugging_face_hub_token|authorization:[[:space:]]*bearer|aws_secret_access_key)([^[:alnum:]_]|$)' <<<"${history}"; then
    printf 'Possible credential material found in image history\n' >&2
    exit 1
fi

docker run --rm --user 0:0 --entrypoint /bin/bash "${IMAGE}" -c '
    set -euo pipefail
    ! find /opt/tt-metal /opt/vllm /opt/vllm-tt-plugin /opt/transformers \
        -name .git -print -quit | grep -q .
    expected_pems=$'"'"'/opt/venv/_python/lib/python3.10/site-packages/pip/_vendor/certifi/cacert.pem\n/opt/venv/lib/python3.10/site-packages/certifi/cacert.pem\n/opt/venv/lib/python3.10/site-packages/grpc/_cython/_credentials/roots.pem'"'"'
    actual_pems="$(find / -xdev -type f -name "*.pem" -print 2>/dev/null | sort)"
    test "${actual_pems}" = "${expected_pems}"
    ! find / -xdev -type f \
        \( -name ".netrc" -o -name "token" -o -name "credentials" \) \
        -print -quit 2>/dev/null | grep -q .
'

printf 'Device-free image validation passed for %s.\n' "${IMAGE}"
printf 'Hardware startup/API/cache validation remains a separate release gate.\n'
