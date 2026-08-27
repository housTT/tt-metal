# syntax=docker/dockerfile:1.7

ARG BUILDER_BASE="ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64@sha256:bf26594246b39f12514698baf17a18d70252013169f57bab540ae6b0f1574d13"
ARG RUNTIME_BASE="docker.io/library/ubuntu@sha256:79676deb51ebb02885b0b9d33788e78a37cf1045ad79d1bb04c6a222c3556b3d"

FROM ${BUILDER_BASE} AS builder

USER root
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ARG TT_METAL_REVISION
ARG VLLM_REVISION

# The named contexts are clean local clones prepared by build-image.sh. Remove
# any environment inherited from the builder image before creating ours.
RUN rm -rf /opt/venv /opt/tt-metal /opt/vllm /tt-metal
COPY --from=ttmetal / /opt/tt-metal/
COPY --from=vllm / /opt/vllm/
COPY runtime-constraints.txt /opt/ornith-runtime-constraints.txt

RUN test "$(git -C /opt/tt-metal rev-parse HEAD)" = "${TT_METAL_REVISION}" && \
    test "$(git -C /opt/vllm rev-parse HEAD)" = "${VLLM_REVISION}" && \
    test -f /opt/tt-metal/tt_metal/third_party/umd/CMakeLists.txt

WORKDIR /opt/tt-metal
RUN ./build_metal.sh --release

# Build a serving-only environment. create_venv.sh intentionally installs the
# full tt-metal developer/test/docs stack; none of that belongs in this image.
RUN test "$(uv --version)" = "uv 0.9.26" && \
    uv python install 3.10.12 && \
    uv venv --link-mode copy --relocatable --managed-python --python 3.10.12 /opt/venv && \
    source /opt/venv/bin/activate && \
    uv pip install uv==0.9.26 setuptools==80.10.2 setuptools-scm==8.1.0 wheel==0.45.1 && \
    UV_CONSTRAINT=/opt/ornith-runtime-constraints.txt \
        uv pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match torch==2.11.0+cpu && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TTNN="0.1.dev1+g${TT_METAL_REVISION:0:10}" \
        UV_CONSTRAINT=/opt/ornith-runtime-constraints.txt \
        uv pip install --no-build-isolation --editable /opt/tt-metal && \
    /opt/tt-metal/scripts/bundle_python_into_venv.sh /opt/venv --force && \
    uv pip check

WORKDIR /opt/vllm
RUN source /opt/venv/bin/activate && \
    export UV_CONSTRAINT=/opt/ornith-runtime-constraints.txt && \
    export VLLM_VERSION_OVERRIDE="0.1.dev1+g${VLLM_REVISION:0:10}.empty" && \
    source plugins/vllm-tt-plugin/docs/install-vllm-tt.sh && \
    uv pip check

# Prove that every project package came from the supplied source contexts and
# that no CUDA runtime replaced the tt-metal CPU PyTorch build. Do not import
# TTNN here: BuildKit has no Tenstorrent devices, and native initialization is
# exercised later by verify-image.sh with the four QB2 devices attached.
RUN source /opt/venv/bin/activate && \
    PYTHONPATH=/opt/tt-metal python - <<'PY'
from importlib.util import find_spec
from pathlib import Path

import torch

origins = {
    package: Path(spec.origin).resolve()
    for package in ("ttnn", "vllm", "vllm_tt_plugin")
    if (spec := find_spec(package)) is not None and spec.origin is not None
}
expected = {
    "ttnn": Path("/opt/tt-metal"),
    "vllm": Path("/opt/vllm"),
    "vllm_tt_plugin": Path("/opt/vllm"),
}
if origins.keys() != expected.keys():
    raise SystemExit(f"missing local packages: {expected.keys() - origins.keys()}")
for package, origin in origins.items():
    if not origin.is_relative_to(expected[package]):
        raise SystemExit(f"{package} resolved outside local source: {origin}")
if torch.version.cuda is not None:
    raise SystemExit(f"CUDA PyTorch is not supported: torch.version.cuda={torch.version.cuda}")
print({name: str(origin) for name, origin in origins.items()})
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
PY

RUN test -f /opt/tt-metal/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/topk_routed_expert_moe/topk_routed_expert_moe.cpp && \
    test -s /opt/tt-metal/ttnn/ttnn/_ttnn.so

RUN mkdir -p /opt/ornith-release && \
    uv pip freeze --python /opt/venv/bin/python > /opt/ornith-release/python-packages.txt && \
    printf '{\n  "tt_metal": "%s",\n  "vllm": "%s",\n  "weights": "%s"\n}\n' \
        "${TT_METAL_REVISION}" "${VLLM_REVISION}" \
        "5df2ed3f675c7beaa490328cc70bb573b65fb660" \
        > /opt/ornith-release/source-revisions.json && \
    find /opt/tt-metal /opt/vllm -name .git -prune -exec rm -rf -- {} + && \
    rm -rf /root/.cache /opt/tt-metal/.cpmcache /opt/vllm/.deps

FROM ${RUNTIME_BASE} AS runtime

ARG TT_METAL_REVISION
ARG VLLM_REVISION

LABEL org.opencontainers.image.title="Ornith-1.0-35b for QuietBox2" \
      org.opencontainers.image.description="OpenAI-compatible Ornith-1.0-35B vLLM server for four-chip Blackhole QuietBox2 systems" \
      org.opencontainers.image.source="https://github.com/housTT/tt-metal" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="ttmetal-b33529c-vllm-a887998" \
      ai.ornith.tt-metal.revision="${TT_METAL_REVISION}" \
      ai.ornith.vllm.revision="${VLLM_REVISION}" \
      ai.ornith.weights.revision="5df2ed3f675c7beaa490328cc70bb573b65fb660" \
      ai.ornith.hardware="QuietBox2 / 4x Blackhole p300c"

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/ornith \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/opt/openmpi-v5.0.7-ulfm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH=/opt/tt-metal \
    TT_METAL_HOME=/opt/tt-metal \
    TT_METAL_RUNTIME_ROOT=/opt/tt-metal \
    TT_METAL_CACHE=/cache \
    HF_HOME=/home/ornith/.cache/huggingface \
    HF_HUB_CACHE=/home/ornith/.cache/huggingface/hub \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MESH_DEVICE="(1, 4)" \
    VLLM_RPC_TIMEOUT=300000 \
    ORNITH_MOE_TOPK_NATIVE=1 \
    ORNITH_MOE_TOPK_NATIVE_SUB_CHUNK=2048 \
    ORNITH_VLLM_PREFILL_WARMUP=all \
    TT_MAX_PREFILLS_PER_STEP=4 \
    TT_INTERLEAVE_PREFILL_CHUNKS=1 \
    OMPI_VERSION=v5.0.7 \
    OMPI_PREFIX=/opt/openmpi-v5.0.7-ulfm \
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
    groupadd --gid 1000 ornith && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash ornith && \
    install -d -o ornith -g ornith \
        /cache \
        /home/ornith/.cache \
        /home/ornith/.cache/huggingface \
        /home/ornith/.cache/huggingface/hub \
        /dev/hugepages-1G

COPY --from=builder --chown=ornith:ornith /opt/tt-metal /opt/tt-metal
COPY --from=builder --chown=ornith:ornith /opt/vllm /opt/vllm
COPY --from=builder --chown=ornith:ornith /opt/venv /opt/venv
COPY --from=builder /opt/openmpi-v5.0.7-ulfm /opt/openmpi-v5.0.7-ulfm
COPY --from=builder /opt/tenstorrent /opt/tenstorrent
COPY --from=builder /opt/ornith-release /usr/share/ornith-1.0-35b
COPY --chmod=755 entrypoint.sh /usr/local/bin/ornith-1.0-35b

WORKDIR /opt/tt-metal
USER ornith

EXPOSE 7890
HEALTHCHECK --interval=30s --timeout=5s --start-period=25m --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:7890/health >/dev/null || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/ornith-1.0-35b"]
