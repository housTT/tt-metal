---
name: tenstorrent-model-container
description: Package Hugging Face models for Tenstorrent QuietBox hardware as standalone, reproducible Docker images that run OpenAI-compatible vLLM servers. Use when containerizing a TT model, replacing host setup or tt-model-manager with Docker, creating a minimal HF-download plus docker-run quickstart, or validating and publishing the resulting image. Do not use for ordinary Docker applications without Tenstorrent hardware.
---

# Tenstorrent Model Container

Build a consumer image whose host contract is limited to Docker, compatible
Tenstorrent devices and hugepages, and model weights in Hugging Face's default
cache. The image owns Python, PyTorch, tt-metal, TTNN, the Tenstorrent vLLM
fork and plugin, model code, native runtime libraries, and serving settings.

Preserve the user's explicit choices. Treat the Ornith implementation as a
proven pattern, not a universal set of model IDs, ports, revisions, topology,
or launch flags.

## Establish the release contract

Before editing, inspect repository instructions, the existing launch path,
source checkouts, dirty worktrees, hardware state, and current model cache.
Resolve rather than guess:

- public model ID, exact weight revision, shard index, and expected cache path;
- served model name, reasoning/tool parsers, context limit, and launch flags;
- Tenstorrent architecture, device count/topology, device nodes, hugepage mount,
  firmware/KMD constraints, and kernel-cache requirements;
- exact tt-metal and vLLM source revisions and submodules;
- builder/runtime base images pinned by digest;
- registry namespace, friendly tag, immutable tag, and host/container port.

Use `hf`, not deprecated `huggingface-cli`. Keep the standard Hub cache layout
unless the user explicitly requests another layout. A weight revision is a
reproducibility and compatibility pin; do not invent one or use a moving branch
without explaining the loss of reproducibility.

For the detailed implementation sequence, read
[references/workflow.md](references/workflow.md). For a concrete, measured
implementation, read [references/ornith-case-study.md](references/ornith-case-study.md).

## Preserve hermeticity

- Build only from the repository sources the user placed in scope. Do not copy
  host pyenvs, virtualenvs, `site-packages`, random cached wheels, build trees,
  credentials, or model weights.
- Create clean, exact-revision build contexts. Validate every source HEAD and
  submodule inside the build. Normalize generated Git metadata when stable
  BuildKit caching matters.
- Remove any inherited builder environment before constructing the release.
  Install a dedicated relocatable Python environment from pinned constraints.
- Prove `ttnn`, `vllm`, and the TT plugin resolve from the supplied source
  trees. Prove PyTorch has the backend expected by tt-metal and was not silently
  replaced by a CUDA build.
- Bake source revisions and a complete Python freeze into the image. Remove
  `.git`, package caches, and build-only credentials from the runtime stage.

## Preserve the runtime contract

- Use a multi-stage image and a small runtime base. Retain TT native libraries,
  firmware data, kernel sources, and the compiler/toolchain actually needed by
  first-launch JIT; test before pruning any of them.
- Run as a non-root user under `tini`. Keep vLLM as the long-lived process and
  stream logs to Docker.
- Mount only the host's default Hugging Face `hub` directory read-only. Enable
  offline Hub/Transformers behavior in the container so a validated snapshot is
  the only model input. Do not mount the host token file.
- Resolve the exact snapshot directory and validate required metadata, the
  safetensors index, and every referenced shard before opening devices.
- Validate the expected character devices and that the hugepage path is
  `hugetlbfs`. Fail with a command the user can act on.
- Put TT JIT artifacts in a named Docker volume. The image must also work with
  an empty volume so a new user can compile on first launch.
- Listen on the user-selected port inside the container and normally publish
  the same port on host loopback. Do not introduce a second internal port
  without a demonstrated requirement.

## Deliver a consumer flow

The end-user quickstart should lead with exactly the shortest supported flow:

```bash
hf download OWNER/MODEL --revision WEIGHT_COMMIT
docker run ... --mount type=bind,src="$HOME/.cache/huggingface/hub",dst=CONTAINER_HF_HUB,readonly ... IMAGE:FRIENDLY_TAG
```

State first-start expectations, kernel-volume reuse, the local API URL, health
and chat probes, clean shutdown behavior, and an OpenCode provider example when
requested. Keep maintainer build/publish instructions in a separate README so
users cannot mistake them for prerequisites.

## Require evidence before release

Do not call the package complete after `docker build`. Run the exact documented
consumer command on idle target hardware and require:

1. syntax, formatting, dependency, package-origin, and native import checks;
2. exact OCI provenance labels and an embedded dependency manifest;
3. successful four-stage startup: devices, weights, kernel warmup, API ready;
4. `/health`, `/v1/models`, and a chat completion with non-empty content;
5. a clean restart using the same named volume and observed kernel-cache hits;
6. immutable and friendly registry tags resolving to one tested digest;
7. a credential-free manifest inspection and digest-pinned pull if the image is
   intended to be public.

Read [references/release-checklist.md](references/release-checklist.md) before
publication or handoff.

Do not reset hardware unless it is idle, the target is exact, and the action is
within the user's scope. Do not make a registry package public, overwrite a
moving tag, or push source changes unless the user authorized that external
mutation. If credentials, visibility, source revisions, weights, devices, or
hugepages block completion, report the precise failed gate rather than weakening
the contract.
