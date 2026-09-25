# Muse-Glimmer-30B on QuietBox2

Run the text-only `meta-models/Muse-Glimmer-30B` model as a local,
OpenAI-compatible vLLM server on a P300x2 QuietBox2 with four Blackhole chips.
The image owns Python, CPU PyTorch, tt-metal, TTNN, vLLM, the Tenstorrent vLLM
plugin, Transformers, the model implementation, native runtime libraries, and
the measured serving settings. The host needs only Docker, compatible devices
and KMD, hugepages, and the weights in Hugging Face's default cache.

## Two-line quickstart

The public model is approximately 60 GB and does not require a Hugging Face
token. This image was built for Tenstorrent KMD 2.11.0.

```bash
hf download meta-models/Muse-Glimmer-30B --revision f84ecc3a0ea984a4c04542a84269e3d065350a6e
docker run --rm --name muse-glimmer-30b --device /dev/tenstorrent --ipc=host --mount type=bind,src=/dev/hugepages,dst=/dev/hugepages --mount type=bind,src="$HOME/.cache/huggingface/hub",dst=/home/muse/.cache/huggingface/hub,readonly --mount source=muse-glimmer-30b-kernels,target=/cache --publish 127.0.0.1:8000:8000 muse-glimmer-30b:qb2
```

Only the cache's `hub` directory is mounted read-only, so the container cannot
read the host's Hugging Face token. The exact snapshot is checked for its
metadata, safetensors index, and both shards before any device is opened.

The first start loads about 60 GB of weights and compiles TT kernels. Allow up
to 25 minutes for a completely empty cache; later starts reuse the named
`muse-glimmer-30b-kernels` volume. Keep the foreground process open and press
Ctrl-C for a clean shutdown.

## Verify the API

In another shell:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models | python3 -m json.tool
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"meta-models/Muse-Glimmer-30B","messages":[{"role":"user","content":"Merge two sorted lists in Python."}],"max_tokens":256,"temperature":0}' \
  | python3 -m json.tool
```

The model list should advertise `meta-models/Muse-Glimmer-30B` with a
131,072-token context limit. The unauthenticated endpoint is published only on
host loopback.

## Troubleshooting

- Pinned weights missing: rerun the first quickstart command without
  `--local-dir`; the image requires the standard Hub cache layout.
- Device validation failure: confirm exactly four Blackhole `1e52:b140`
  devices are present and the Docker command includes `/dev/tenstorrent`.
- Hugepages failure: `stat -f -c %T /dev/hugepages` must print `hugetlbfs`.
- Busy hardware: stop and wait. The container never resets devices or evicts
  another process.
- An unclean prior shutdown can require a board reset, but only reset a known
  idle system as an explicit operator action.

Maintainer build and verification instructions are in [README.md](README.md).

