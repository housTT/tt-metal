# Ornith-1.0-35b on QuietBox2

Run the text-only `ornith-ai/Ornith-1.0-35B` model as a local,
OpenAI-compatible vLLM server on a four-chip Blackhole QuietBox2. The image
contains and configures Python, PyTorch, tt-metal, TTNN, the Tenstorrent vLLM
fork, the TT plugin, the Ornith model code, and the measured serving settings.
Users do not clone or build tt-metal, create a virtual environment, install
wheels, or use `tt-model-manager` on the host.

## Two-line quickstart

Prerequisites are Docker, the current `hf` CLI, approximately 71 GB for the
public model weights, and an idle QB2 whose 1 GiB hugepages are mounted at
`/dev/hugepages-1G`.

```bash
hf download ornith-ai/Ornith-1.0-35B --revision 5df2ed3f675c7beaa490328cc70bb573b65fb660
docker run --rm --name ornith-1.0-35b --device /dev/tenstorrent --ipc=host --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G --mount type=bind,src="$HOME/.cache/huggingface/hub",dst=/home/ornith/.cache/huggingface/hub,readonly --mount source=ornith-1.0-35b-kernels,target=/cache --publish 127.0.0.1:7890:7890 ghcr.io/houstt/ornith-1.0-35b:qb2
```

The first command stores the exact tested snapshot in Hugging Face's default
cache. The container mounts only the `hub` subdirectory read-only, so it does
not receive the host's Hugging Face token. The public weights do not require a
token.

The first launch compiles TT kernels and can take 15–25 minutes. Later launches
reuse the `ornith-1.0-35b-kernels` Docker volume. Keep the foreground process
open; press Ctrl-C for a clean shutdown.

## Verify the server

In another shell:

```bash
curl -fsS http://127.0.0.1:7890/health
curl -fsS http://127.0.0.1:7890/v1/models | python3 -m json.tool
curl -fsS http://127.0.0.1:7890/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ornith-ai/Ornith-1.0-35B","messages":[{"role":"user","content":"What is 17 multiplied by 19?"}],"max_tokens":2048,"temperature":0}' \
  | python3 -m json.tool
```

The model list should advertise `ornith-ai/Ornith-1.0-35B` with a 262,144-token
server limit. The container listens on port 7890 internally and Docker exposes
the same port only on host loopback. The server is intentionally
unauthenticated and is not reachable from other hosts.

## Connect OpenCode

Add this provider to `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ornith-1-0-35b": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ornith-1.0-35b",
      "options": {
        "baseURL": "http://127.0.0.1:7890/v1",
        "apiKey": "unused"
      },
      "models": {
        "ornith-ai/Ornith-1.0-35B": {
          "name": "Ornith-1.0-35b",
          "limit": {
            "context": 32768,
            "output": 16384
          }
        }
      }
    }
  },
  "model": "ornith-1-0-35b/ornith-ai/Ornith-1.0-35B"
}
```

`apiKey` is a client-side placeholder; the local vLLM server does not validate
it. OpenCode uses a conservative 32K context by default even though the server
supports longer requests.

## Troubleshooting

- **Pinned weights missing:** rerun the first quickstart command. Do not use
  `--local-dir`; the container expects the standard HF cache layout.
- **Not exactly four devices:** confirm `tt-smi -ls` reports four Blackhole
  `p300c` devices and that the Docker command includes `/dev/tenstorrent`.
- **Hugepages error:** confirm `stat -f -c %T /dev/hugepages-1G` prints
  `hugetlbfs`.
- **Port already allocated:** stop the process using host port 7890 before
  starting the container.
- **A previous process was killed:** on an otherwise idle host, reset the
  devices with `tt-smi -r`, then retry. The container never resets hardware
  automatically.

Maintainer build and publication instructions live separately in
[`docker/README.md`](../../docker/README.md).
