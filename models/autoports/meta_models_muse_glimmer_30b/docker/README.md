# Muse-Glimmer-30B container maintenance

This directory builds a standalone OpenAI-compatible serving image from clean,
exact-revision local Git contexts. It never copies host virtual environments,
model weights, credentials, build trees, or dirty worktree files.
`runtime-lock.txt` constrains the complete resolved Python environment, while
`runtime-constraints.txt` records the smaller ABI compatibility boundary.

## Release contract

| Field | Pinned value |
|---|---|
| Weights | `meta-models/Muse-Glimmer-30B` at `f84ecc3a0ea984a4c04542a84269e3d065350a6e` |
| Served name | `meta-models/Muse-Glimmer-30B` |
| tt-metal | `0dd37ce6ee33826ebb8ce23a5d83a45bca7d6b29` |
| vLLM | `ee0da84ab9e04ac7610e28580af62c365e898389` (`v0.24.0`) |
| vLLM TT plugin | `106744c01de96825ed8c81226f4fa043e8e929f4` |
| Transformers | `5eddc12edfaf8cafde8c9bae4ccb12f8a139b4f9` (`v5.15.0`) |
| Hardware | P300x2 QuietBox2, four Blackhole `1e52:b140`, mesh 1x4 |
| KMD | 2.11.0 |
| API | `127.0.0.1:8000`, 131,072 tokens, batch capacity 32 |
| Builder | `tt-metalium/ubuntu-22.04-dev-amd64@sha256:bf265942…` |
| Runtime | `ubuntu@sha256:79676deb…` |

The Hugging Face `main` ref currently points to a metadata-only snapshot. The
weight revision above is deliberate and must not be replaced by `main`.

The plugin intentionally overrides vLLM's OpenCV floor with
`opencv-python-headless==4.11.0.86`: TTNN requires NumPy 1.x, while OpenCV 4.13
requires NumPy 2.x, and Muse-Glimmer does not use vLLM's lazy video path. The
build asserts that this is the sole package-metadata incompatibility; any other
`uv pip check` failure stops the image build.

## Build

Prepare local checkouts at the revisions in the table. By default the script
expects sibling directories named `vllm`, `vllm-tt-plugin`, and `transformers`
beside the `tt-metal` checkout. Alternate paths can be supplied with
`VLLM_SOURCE`, `VLLM_TT_PLUGIN_SOURCE`, and `TRANSFORMERS_SOURCE`.

```bash
./build-image.sh
./verify-image.sh muse-glimmer-30b:qb2
```

The first command produces both `muse-glimmer-30b:qb2` and the immutable local
tag `muse-glimmer-30b:ttmetal-0dd37ce-vllm-ee0da84-plugin-106744c`.
`verify-image.sh` is intentionally device-free: it checks OCI configuration,
provenance labels, package origins, embedded manifests, CPU PyTorch, and the
absence of Git metadata and obvious credentials without disturbing TT devices.

## Hardware release gate

Do not attach this image to hardware while any other process is using the TT
devices. A release still requires the exact consumer command in QUICKSTART.md
on idle hardware, followed by:

1. native TTNN, vLLM, plugin, and model import checks with all four devices;
2. cold startup through device discovery, weight loading, kernel warmup, and API readiness;
3. `/health`, `/v1/models`, and a non-empty chat completion;
4. clean shutdown and restart with the same named cache volume;
5. measured restart time and observed compiler-cache hits.

Do not reset hardware from an entrypoint or validation script.

## Publish

Publication is opt-in and requires an explicit registry-qualified image name:

```bash
./build-image.sh --push --image ghcr.io/OWNER/muse-glimmer-30b:qb2
```

The build pushes the friendly and immutable tags with OCI provenance and SBOM.
Before handoff, confirm both tags resolve to the same digest and prove an
anonymous digest-pinned pull with a new empty Docker configuration. Registry
visibility changes are separate operator actions.
