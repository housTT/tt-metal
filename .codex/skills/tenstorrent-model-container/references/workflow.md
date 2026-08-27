# Implementation workflow

Use this reference when creating or materially changing a Tenstorrent model
container.

## 1. Discover the measured launch

Find the current quickstart, launch scripts, environment variables, plugin
installation path, model implementation, evaluation results, and expected
hardware. Prefer the already measured configuration over generic vLLM defaults.

Record a release contract before writing the Dockerfile:

| Field | Required value |
|---|---|
| HF model and revision | Public ID plus immutable commit |
| Served model name | Exact case-sensitive API identifier |
| Hardware | Architecture, count, mesh, firmware and KMD |
| Host mounts | Devices, hugepages, HF hub, named kernel cache |
| API | Bind address, one internal/host port, parsers and limits |
| Sources | tt-metal, vLLM and submodule commits |
| Bases | Builder and runtime image digests |
| Registry | Friendly and source-revision tags |

Verify the weight revision with Hub metadata and inspect the local standard
snapshot if it is already downloaded. Validate the index and count its shards.

## 2. Create clean build inputs

Use the repository checkouts only as object sources. A robust build wrapper:

1. requires each checkout to be at the selected commit;
2. creates a temporary shallow checkout of that exact commit;
3. recreates submodules at recorded revisions from local checkouts;
4. excludes untracked, ignored, generated, and user-modified host files;
5. normalizes Git pack/index metadata when reproducible context hashes are
   needed for BuildKit cache reuse;
6. passes the clean trees as named Docker build contexts.

Fail if a source revision or submodule cannot be reproduced locally. Do not
fall back to an unrelated network branch, environment, or wheel.

## 3. Build in stages

The builder should:

- start from a digest-pinned architecture-compatible TT builder;
- delete inherited `/opt/venv`, `/opt/tt-metal`, `/opt/vllm`, or equivalents;
- copy the clean named contexts and revalidate their commits;
- compile tt-metal in the measured release mode;
- create a pinned relocatable Python environment;
- install the expected PyTorch build, local tt-metal, local vLLM, and local TT
  plugin under constraints;
- run dependency and package-origin checks;
- generate `python-packages.txt` and `source-revisions.json`;
- remove Git metadata and caches.

The runtime should contain only what the server and TT JIT need. Start with a
correct image, validate it, then prune iteratively. TT compiler trees, firmware,
kernel source, dynamic libraries, and apparently build-oriented paths may be
runtime requirements; prove removal with a cold-cache hardware launch.

## 4. Implement the entrypoint

Run validation before vLLM:

1. derive the exact snapshot path beneath the container's default HF hub;
2. require configuration/tokenizer metadata and the safetensors index;
3. require every unique shard referenced by the index;
4. require the expected number and type of TT device nodes;
5. require the configured hugepage mount to report `hugetlbfs`;
6. export only the model and TT environment needed by the measured launch;
7. `exec` the OpenAI-compatible vLLM API server directly.

Use actionable errors that include the missing `hf download` or Docker flag.
Do not fetch weights at container startup: downloading remains an explicit host
step and the model cache remains read-only.

## 5. Validate progressively

Run inexpensive gates first:

- shell syntax, Dockerfile parse/build, HTML/Markdown checks, and repository
  hooks;
- image labels, user, entrypoint, healthcheck, platform, size, `.git` absence,
  and embedded manifests;
- native imports with target devices attached;
- exact documented startup with an empty named kernel cache;
- API health, model listing, chat content, reasoning/tools if part of the
  contract, and clean shutdown;
- restart using the same cache and telemetry showing actual cache hits.

A startup after an unclean termination may require an explicit board reset.
Confirm no other process uses the hardware and use the repository-supported
device tool. Never hide a reset inside the consumer entrypoint.

## 6. Publish and prove consumption

Push an immutable source-revision tag first, then the friendly hardware tag.
Require both to resolve to the same OCI digest. Registry publication often
creates a private package by default; change visibility only when public access
is part of the user's request and warn if the transition is irreversible.

Prove public consumption with a newly created empty Docker configuration:

```bash
DOCKER_CONFIG=EMPTY_DIRECTORY docker buildx imagetools inspect IMAGE:TAG
DOCKER_CONFIG=EMPTY_DIRECTORY docker pull IMAGE:TAG@sha256:DIGEST
```

Clean the temporary configuration and any transient registry credentials. Do
not use the normal logged-in Docker configuration as evidence of anonymity.

## 7. Hand off

Report the public image, immutable tag, digest, compressed download size,
expanded disk size, exact two-line quickstart, validation evidence, first-start
expectations, and source/documentation links. Distinguish completed gates from
external blockers.
