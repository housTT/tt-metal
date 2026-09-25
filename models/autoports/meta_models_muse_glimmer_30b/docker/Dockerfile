# syntax=docker/dockerfile:1.7

ARG BUILDER_BASE="ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64@sha256:bf26594246b39f12514698baf17a18d70252013169f57bab540ae6b0f1574d13"
ARG RUNTIME_BASE="docker.io/library/ubuntu@sha256:79676deb51ebb02885b0b9d33788e78a37cf1045ad79d1bb04c6a222c3556b3d"

FROM ${BUILDER_BASE} AS builder

USER root
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ARG TT_METAL_REVISION
ARG VLLM_REVISION
ARG VLLM_TT_PLUGIN_REVISION
ARG TRANSFORMERS_REVISION

# These named contexts are clean exact-revision local clones prepared by
# build-image.sh. Discard every inherited source tree and Python environment.
RUN rm -rf /opt/venv /opt/tt-metal /opt/vllm /opt/vllm-tt-plugin /opt/transformers /tt-metal
COPY --from=ttmetal / /opt/tt-metal/
COPY --from=vllm / /opt/vllm/
COPY --from=vllm_tt_plugin / /opt/vllm-tt-plugin/
COPY --from=transformers / /opt/transformers/
COPY runtime-constraints.txt /opt/muse-glimmer-runtime-constraints.txt

RUN test "$(git -C /opt/tt-metal rev-parse HEAD)" = "${TT_METAL_REVISION}" && \
    test "$(git -C /opt/vllm rev-parse HEAD)" = "${VLLM_REVISION}" && \
    test "$(git -C /opt/vllm-tt-plugin rev-parse HEAD)" = "${VLLM_TT_PLUGIN_REVISION}" && \
    test "$(git -C /opt/transformers rev-parse HEAD)" = "${TRANSFORMERS_REVISION}" && \
    test -f /opt/tt-metal/tt_metal/third_party/umd/CMakeLists.txt && \
    test -f /opt/vllm/requirements/common.txt && \
    test -f /opt/vllm-tt-plugin/docs/vllm-overrides.txt && \
    test -f /opt/transformers/src/transformers/models/muse_glimmer/modeling_muse_glimmer.py

WORKDIR /opt/tt-metal
RUN ./build_metal.sh --release
COPY runtime-lock.txt /opt/muse-glimmer-runtime-lock.txt

# Build a serving-only, relocatable environment. The TT-Metal developer venv is
# intentionally not used because it includes test/docs packages and host state.
RUN test "$(uv --version | cut -d' ' -f2)" = "0.9.26" && \
    uv python install 3.10.12 && \
    uv venv --link-mode copy --relocatable --managed-python --python 3.10.12 /opt/venv && \
    source /opt/venv/bin/activate && \
    UV_CONSTRAINT=/opt/muse-glimmer-runtime-lock.txt \
        uv pip install uv==0.12.5 setuptools==80.10.2 setuptools-scm==8.1.0 \
        setuptools-rust==1.13.0 semantic-version==2.10.0 wheel==0.45.1 && \
    test "$(/opt/venv/bin/uv --version | cut -d' ' -f2)" = "0.12.5" && \
    UV_CONSTRAINT=/opt/muse-glimmer-runtime-lock.txt \
        /opt/venv/bin/uv pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match torch==2.11.0+cpu && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TTNN="0.1.dev1+g${TT_METAL_REVISION:0:10}" \
        UV_CONSTRAINT=/opt/muse-glimmer-runtime-lock.txt \
        /opt/venv/bin/uv pip install --no-build-isolation --editable /opt/tt-metal && \
    /opt/tt-metal/scripts/bundle_python_into_venv.sh /opt/venv --force && \
    /opt/venv/bin/uv pip check

# Install the exact source revision that supplies Muse-Glimmer support, then
# vLLM's empty (non-CUDA) target and the matching Tenstorrent plugin.
RUN source /opt/venv/bin/activate && \
    export UV_CONSTRAINT=/opt/muse-glimmer-runtime-lock.txt && \
    /opt/venv/bin/uv pip install --no-build-isolation --editable /opt/transformers && \
    /opt/venv/bin/uv pip install --override /opt/vllm-tt-plugin/docs/vllm-overrides.txt \
        -r /opt/vllm/requirements/common.txt && \
    VLLM_TARGET_DEVICE=empty \
        VLLM_VERSION_OVERRIDE="0.24.0+g${VLLM_REVISION:0:10}.empty" \
        /opt/venv/bin/uv pip install --no-deps --no-build-isolation --editable /opt/vllm && \
    /opt/venv/bin/uv pip install --no-build-isolation --editable /opt/vllm-tt-plugin && \
    if check_output="$(/opt/venv/bin/uv pip check 2>&1)"; then \
        printf 'Expected the documented OpenCV/NumPy metadata override\n' >&2; \
        exit 1; \
    else \
        check_status=$?; \
    fi && \
    printf '%s\n' "${check_output}" && \
    test "${check_status}" -eq 1 && \
    test "$(grep -c '^The package `' <<<"${check_output}")" -eq 1 && \
    grep -Fqx 'Found 1 incompatibility' <<<"${check_output}" && \
    grep -Fqx 'The package `vllm` requires `opencv-python-headless>=4.13.0`, but `4.11.0.86` is installed' <<<"${check_output}"

# Prove project packages resolve only from supplied source contexts and that the
# environment retained CPU PyTorch. find_spec avoids initializing TT hardware.
RUN source /opt/venv/bin/activate && \
    PYTHONPATH=/opt/tt-metal python - <<'PY'
from importlib.util import find_spec
from pathlib import Path

import torch

expected = {
    "ttnn": Path("/opt/tt-metal"),
    "vllm": Path("/opt/vllm"),
    "vllm_tt_plugin": Path("/opt/vllm-tt-plugin"),
    "transformers": Path("/opt/transformers"),
}
origins = {}
for package, root in expected.items():
    spec = find_spec(package)
    if spec is None or spec.origin is None:
        raise SystemExit(f"missing local package: {package}")
    origin = Path(spec.origin).resolve()
    if not origin.is_relative_to(root):
        raise SystemExit(f"{package} resolved outside {root}: {origin}")
    origins[package] = str(origin)
if find_spec("transformers.models.muse_glimmer") is None:
    raise SystemExit("the pinned Transformers source lacks Muse-Glimmer support")
if torch.version.cuda is not None:
    raise SystemExit(f"CUDA PyTorch is unsupported: torch.version.cuda={torch.version.cuda}")
print(origins)
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
PY

RUN test -s /opt/tt-metal/ttnn/ttnn/_ttnn.so && \
    test -f /opt/tt-metal/models/autoports/meta_models_muse_glimmer_30b/tt/generator_vllm.py && \
    grep -q 'meta_models_muse_glimmer_30b' /opt/vllm-tt-plugin/src/vllm_tt_plugin/platform.py

RUN mkdir -p /opt/muse-glimmer-release && \
    /opt/venv/bin/uv pip freeze --python /opt/venv/bin/python > /opt/muse-glimmer-release/python-packages.txt && \
    printf '{\n  "tt_metal": "%s",\n  "vllm": "%s",\n  "vllm_tt_plugin": "%s",\n  "transformers": "%s",\n  "weights": "%s"\n}\n' \
        "${TT_METAL_REVISION}" "${VLLM_REVISION}" \
        "${VLLM_TT_PLUGIN_REVISION}" "${TRANSFORMERS_REVISION}" \
        "f84ecc3a0ea984a4c04542a84269e3d065350a6e" \
        > /opt/muse-glimmer-release/source-revisions.json && \
    find /opt/tt-metal /opt/vllm /opt/vllm-tt-plugin /opt/transformers -name .git -prune -exec rm -rf -- {} + && \
    rm -rf /root/.cache /opt/tt-metal/.cpmcache /opt/vllm/.deps

FROM ${RUNTIME_BASE} AS runtime

ARG TT_METAL_REVISION
ARG VLLM_REVISION
ARG VLLM_TT_PLUGIN_REVISION
ARG TRANSFORMERS_REVISION

LABEL org.opencontainers.image.title="Muse-Glimmer-30B for QuietBox2" \
      org.opencontainers.image.description="OpenAI-compatible Muse-Glimmer-30B vLLM server for four-chip Blackhole QuietBox2 systems" \
      org.opencontainers.image.source="https://github.com/tenstorrent/tt-metal" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="ttmetal-0dd37ce-vllm-ee0da84-plugin-106744c" \
      ai.muse-glimmer.tt-metal.revision="${TT_METAL_REVISION}" \
      ai.muse-glimmer.vllm.revision="${VLLM_REVISION}" \
      ai.muse-glimmer.vllm-tt-plugin.revision="${VLLM_TT_PLUGIN_REVISION}" \
      ai.muse-glimmer.transformers.revision="${TRANSFORMERS_REVISION}" \
      ai.muse-glimmer.weights.revision="f84ecc3a0ea984a4c04542a84269e3d065350a6e" \
      ai.muse-glimmer.hardware="QuietBox2 / P300x2 / 4x Blackhole"

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/muse \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/opt/openmpi-v5.0.7-ulfm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH=/opt/tt-metal \
    TT_METAL_HOME=/opt/tt-metal \
    TT_METAL_RUNTIME_ROOT=/opt/tt-metal \
    TT_METAL_CACHE=/cache \
    HF_HOME=/home/muse/.cache/huggingface \
    HF_HUB_CACHE=/home/muse/.cache/huggingface/hub \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MESH_DEVICE=P300x2 \
    ARCH_NAME=blackhole \
    VLLM_CONFIGURE_LOGGING=1 \
    VLLM_RPC_TIMEOUT=900000 \
    TORCHDYNAMO_DISABLE=1 \
    LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:/opt/tt-metal/build/lib

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libatomic1 \
        libcap2 \
        libevent-core-2.1-7 \
        libevent-pthreads-2.1-7 \
        libgl1 \
        libgmp10 \
        libhwloc15 \
        libmpc3 \
        libmpfr6 \
        libnuma1 \
        libsndfile1 \
        libudev1 \
        libzstd1 \
        tini \
        zlib1g && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 1000 muse && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash muse && \
    install -d -o muse -g muse \
        /cache \
        /home/muse/.cache \
        /home/muse/.cache/huggingface \
        /home/muse/.cache/huggingface/hub \
        /dev/hugepages

COPY --from=builder --chown=muse:muse /opt/tt-metal /opt/tt-metal
COPY --from=builder --chown=muse:muse /opt/vllm /opt/vllm
COPY --from=builder --chown=muse:muse /opt/vllm-tt-plugin /opt/vllm-tt-plugin
COPY --from=builder --chown=muse:muse /opt/transformers /opt/transformers
COPY --from=builder --chown=muse:muse /opt/venv /opt/venv
COPY --from=builder /opt/openmpi-v5.0.7-ulfm /opt/openmpi-v5.0.7-ulfm
COPY --from=builder /opt/tenstorrent /opt/tenstorrent
COPY --from=builder /opt/muse-glimmer-release /usr/share/muse-glimmer-30b
COPY --chmod=755 entrypoint.sh /usr/local/bin/muse-glimmer-30b

WORKDIR /opt/tt-metal
USER muse

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=25m --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8000/health >/dev/null || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/muse-glimmer-30b"]
