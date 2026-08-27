# Ornith-1.0-35b container

This directory builds the standalone QuietBox2 runtime published as
`ghcr.io/houstt/ornith-1.0-35b:qb2`. It does not use `tt-model-manager`, embed
model weights, or consume a host Python environment.

## Build locally

The sibling `tt-metal` and `vllm` repositories must contain the revisions
pinned in `build-image.sh`. The script makes clean local clones of those Git
objects and their checked-out submodules before invoking Docker BuildKit.

```bash
./models/autoports/ornith_ai_ornith_1_0_35b/docker/build-image.sh
```

Pass `--push` to publish both the `qb2` tag and the immutable source-revision
tag. GHCR authentication must have `write:packages`.

## Consumer quickstart

```bash
hf download ornith-ai/Ornith-1.0-35B --revision 5df2ed3f675c7beaa490328cc70bb573b65fb660
docker run --rm --name ornith-1.0-35b --device /dev/tenstorrent --ipc=host --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G --mount type=bind,src="$HOME/.cache/huggingface/hub",dst=/home/ornith/.cache/huggingface/hub,readonly --mount source=ornith-1.0-35b-kernels,target=/cache --publish 127.0.0.1:7890:7890 ghcr.io/houstt/ornith-1.0-35b:qb2
```
