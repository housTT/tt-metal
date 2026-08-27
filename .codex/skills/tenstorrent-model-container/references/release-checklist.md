# Release checklist

Use this checklist before publishing or declaring a Tenstorrent model container
ready for other users.

## Contract

- Model ID, case, weight revision, served name, topology and port are exact.
- tt-metal, vLLM, submodules, base images and critical dependencies are pinned.
- Consumer prerequisites do not include source builds, host Python, cached
  wheels, or a model manager unless the user explicitly selected one.

## Image

- Builder inputs are clean repository revisions.
- Runtime contains no credentials, weights, `.git`, host environments or caches.
- Python packages resolve from intended sources and dependency checks pass.
- Expected native TT operations and plugin imports work on target hardware.
- Non-root user, `tini`, direct server process and healthcheck are configured.
- Source revisions and complete Python freeze are embedded.
- Compressed, expanded and unique disk sizes are reported.

## Runtime

- Default HF `hub` mount is read-only and token files are not exposed.
- Exact snapshot metadata, index and all shards are validated offline.
- Device count/types, hugepages, mesh and TT settings are validated.
- An empty named kernel volume supports first-launch compilation.
- `/health`, `/v1/models` and chat content pass through the documented port.
- Reasoning and tool calls pass when promised by the quickstart.
- Clean shutdown releases devices.
- Cached restart shows real compiler-cache hits and materially shorter warmup.

## Publication

- Immutable and friendly tags point to the tested digest.
- Package visibility matches the user's requested audience.
- A credential-free manifest inspection succeeds.
- A credential-free, digest-pinned pull succeeds.
- Registry credentials and temporary Docker configurations are cleaned up.

## Documentation

- The user quickstart leads with `hf download` and one `docker run` command.
- The quickstart says what the image owns and what remains on the host.
- Ports, mounts, first-start timing, cache reuse, API probes and OpenCode config
  match the tested command.
- Maintainer build and publication steps are separate from user prerequisites.
- GitHub links point to the branch or commit that actually contains the files.
