# Ornith-1.0-35B case study

Read this only when a concrete example helps implement or review another
Tenstorrent model container. These values are evidence from Ornith, not defaults
for other models.

## Release contract

- Model: `ornith-ai/Ornith-1.0-35B`
- Weights: `5df2ed3f675c7beaa490328cc70bb573b65fb660`
- tt-metal: `b33529c1a9575b134b063ebdfe9de068c88554a9`
- vLLM: `a887998646dc4e6f192bce8d485bf89f4596ca2f`
- Hardware: QuietBox2, four Blackhole `p300c`, mesh `(1,4)`
- API: `127.0.0.1:7890`, same port inside the container
- Image: `ghcr.io/houstt/ornith-1.0-35b:qb2`
- Immutable tag: `ttmetal-b33529c-vllm-a887998`
- OCI digest: `sha256:9b2c318a56a2ff1003d0e0fba648df006336c3981b0965af90ed78f55aa599e0`

The model snapshot stayed in the host's default HF cache. Only
`~/.cache/huggingface/hub` was mounted read-only, so the container could not
read the host token. A named `/cache` volume retained TT JIT artifacts.

## Implementation

The complete implementation is pinned at commit
[`72e00230fb574680119c7e77a780e6f0ca056562`](https://github.com/housTT/tt-metal/commit/72e00230fb574680119c7e77a780e6f0ca056562).

- [Dockerfile](https://github.com/housTT/tt-metal/blob/72e00230fb574680119c7e77a780e6f0ca056562/models/autoports/ornith_ai_ornith_1_0_35b/docker/Dockerfile)
- [build-image.sh](https://github.com/housTT/tt-metal/blob/72e00230fb574680119c7e77a780e6f0ca056562/models/autoports/ornith_ai_ornith_1_0_35b/docker/build-image.sh)
- [entrypoint.sh](https://github.com/housTT/tt-metal/blob/72e00230fb574680119c7e77a780e6f0ca056562/models/autoports/ornith_ai_ornith_1_0_35b/docker/entrypoint.sh)
- [verify-image.sh](https://github.com/housTT/tt-metal/blob/72e00230fb574680119c7e77a780e6f0ca056562/models/autoports/ornith_ai_ornith_1_0_35b/docker/verify-image.sh)
- [consumer quickstart](https://github.com/housTT/tt-metal/blob/72e00230fb574680119c7e77a780e6f0ca056562/models/autoports/ornith_ai_ornith_1_0_35b/doc/prefill_optimization/QUICKSTART.md)

The build wrapper constructed normalized exact-revision local Git contexts, so
repeated builds reused the expensive native layers without accepting dirty host
files. The runtime used an embedded CPython 3.10.12 environment, CPU PyTorch,
local editable tt-metal/vLLM sources, direct vLLM API launch, a non-root user,
`tini`, offline HF settings, provenance labels, and baked dependency manifests.

## Validation evidence

- Native TTNN, vLLM and TT plugin imports resolved under `/opt/tt-metal` and
  `/opt/vllm` on the four attached devices.
- The exact two-line consumer command loaded all 40 layers and exposed the
  official model name with a 262,144-token server limit.
- `/health`, `/v1/models`, and chat completion passed. The chat response
  contained `QB2 container ready`.
- First engine warmup took 269.46 seconds. Reusing the named volume reduced it
  to 20.36 seconds with 2,662/2,662 JIT cache hits.
- Both public tags resolved anonymously to the tested OCI digest, and a clean
  Docker configuration pulled that digest without credentials.

## Lessons

- `--device /dev/tenstorrent` exposed all four numbered devices on this host;
  verify this behavior on other Docker/driver versions.
- A direct vLLM API process gave correct Docker log and signal behavior.
- The vLLM source version required the local `.empty` suffix in its override.
- An immediate restart after shutdown encountered a TT fabric timeout once;
  resetting the otherwise idle board with `tt-smi` restored it. This remained a
  troubleshooting action, not container startup behavior.
- GHCR created the new package private. Public visibility required an explicit
  package-settings action before anonymous verification.
- The working image was correct but not minimal: 2.03 GB compressed, 6.39 GB
  visible filesystem content and 8.42 GB in Docker layer accounting. It retained
  full source/build trees, generic vLLM dependencies, and multiple SFPI compiler
  trees; slimming requires cold-cache hardware revalidation.
