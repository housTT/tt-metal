# Validation record

Date: 2026-08-27

## Local artifact

- Friendly tag: `muse-glimmer-30b:qb2`
- Immutable tag: `muse-glimmer-30b:ttmetal-0dd37ce-vllm-ee0da84-plugin-106744c`
- Shared local image ID: `sha256:4aeed6191f1cfe0be8e752c1509ca6ca31b9d56af49dca73b144d015f196ed89`
- Docker-reported content size: 2,083,253,829 bytes
- Unpacked size: 8.7 GB total, 8.58 GB unique
- Weights embedded: no

## Passed without TT devices

- Multi-stage build from clean, exact-revision Git contexts
- Digest-pinned builder and runtime bases
- Full 168-package Python resolution constrained by `runtime-lock.txt`
- CPU-only PyTorch 2.11.0 and exact local package origins
- Muse-Glimmer Transformers registration and pinned source manifests
- Native `_ttnn.so` shared-library resolution with no missing libraries
- Non-root UID 1000 runtime, `tini`, health check, loopback-oriented quickstart
- No Git metadata, unexpected PEM files, credential filenames, or secrets in history
- Read-only pinned HF snapshot preflight: index, metadata, and exactly two shards
- Entrypoint stops at the device gate when `/dev/tenstorrent` is not mounted
- Bash syntax and trailing-whitespace checks

`shellcheck` was not available on the host, so that optional lint was not run.

## Intentionally pending

Hardware validation is paused because another bring-up owns the TT devices. On
idle hardware the release still needs cold startup, OpenAI API checks, a clean
restart with the same named compiler-cache volume, and observed cache reuse.

No registry push or visibility change has been performed. Publication requires
an explicitly authorized registry-qualified image name, followed by digest and
anonymous-pull verification.
