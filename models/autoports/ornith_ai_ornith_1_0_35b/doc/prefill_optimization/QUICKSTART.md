# Ornith-1.0-35B vLLM quickstart

This starts the validated OpenAI-compatible server for `ornith-ai/Ornith-1.0-35B` on four
Blackhole `p300c` chips in a `(1, 4)` mesh (P300_X2). The throughput profile supports a 262,144-token
context, up to eight active sequences, reasoning parsing, and XML tool calls.

## 1. Prepare the paired checkouts

Start from a configured Tenstorrent host where `tt-smi -ls` shows four Blackhole `p300c` devices.
Allow roughly 80 GB for the model snapshot plus build and kernel-cache headroom.

The measured runtime revisions are tt-metal `824072e81e99af0cacb36adb6a33e271cd66c47f` and vLLM
`a887998646dc4e6f192bce8d485bf89f4596ca2f`. Use the normal tt-metal build and environment setup,
then install the TT vLLM plugin from the paired vLLM checkout:

```bash
export ORNITH_ROOT=/path/to/ornith-stack

git clone --recurse-submodules https://github.com/housTT/tt-metal.git "$ORNITH_ROOT/tt-metal"
git -C "$ORNITH_ROOT/tt-metal" checkout 824072e81e99af0cacb36adb6a33e271cd66c47f
git -C "$ORNITH_ROOT/tt-metal" submodule update --init --recursive

cd "$ORNITH_ROOT/tt-metal"
sudo ./install_dependencies.sh
./build_metal.sh
./create_venv.sh --python-version 3.12 --env-dir "$ORNITH_ROOT/venv"
source "$ORNITH_ROOT/venv/bin/activate"

git clone https://github.com/housTT/vllm.git "$ORNITH_ROOT/vllm"
git -C "$ORNITH_ROOT/vllm" checkout a887998646dc4e6f192bce8d485bf89f4596ca2f
cd "$ORNITH_ROOT/vllm"
source "$ORNITH_ROOT/venv/bin/activate"
source plugins/vllm-tt-plugin/docs/install-vllm-tt.sh

export TT_METAL_HOME="$ORNITH_ROOT/tt-metal"
export PYTHONPATH="$TT_METAL_HOME${PYTHONPATH:+:$PYTHONPATH}"
python -c "import ttnn, vllm_tt_plugin; print('ttnn and vLLM TT plugin available')"
```

Use the current Hugging Face CLI to pin the evaluated weights:

```bash
hf auth login
hf download ornith-ai/Ornith-1.0-35B \
  --revision 5df2ed3f675c7beaa490328cc70bb573b65fb660
```

## 2. Select the measured throughput profile

```bash
export ORNITH_MAX_NUM_SEQS=8
export ORNITH_MAX_TOKENS_ALL_USERS=1052672
export ORNITH_API_SERVER_COUNT=4
export ORNITH_VLLM_PREFILL_PROFILE=throughput
export ORNITH_MOE_TOPK_NATIVE=1
export ORNITH_MOE_TOPK_NATIVE_SUB_CHUNK=2048
export ORNITH_VLLM_PREFILL_WARMUP=all
export TT_MAX_PREFILLS_PER_STEP=4
export TT_INTERLEAVE_PREFILL_CHUNKS=1
unset ORNITH_PRECISION_POLICY ORNITH_MOE_GATHER

export ORNITH_TT_CONFIG='{"trace_region_size": 200000000, "l1_small_size": 32768, "fabric_config": "FABRIC_1D_RING", "fabric_router_max_packet_bytes": 8192, "input_queue_batching_delay": 2.0}'
```

The token pool covers eight requests of 131,072 input tokens plus 512 output tokens. For a
single-user latency profile, set `ORNITH_MAX_NUM_SEQS=1`, `ORNITH_API_SERVER_COUNT=1`, unset
`ORNITH_MAX_TOKENS_ALL_USERS` and `ORNITH_VLLM_PREFILL_PROFILE`, and remove
`input_queue_batching_delay` from `ORNITH_TT_CONFIG`.

## 3. Launch

Launch from tt-metal so an adjacent vLLM source tree does not shadow the installed plugin. Resetting
with `tt-smi -r` affects every visible TT device, so do it only on an idle host.

```bash
export TT_METAL_HOME="$ORNITH_ROOT/tt-metal"
export PYTHONPATH="$TT_METAL_HOME${PYTHONPATH:+:$PYTHONPATH}"
cd "$TT_METAL_HOME"

tt-smi -r
python -m models.common.readiness_check.run_vllm_server \
  --stages serve \
  --model-dir models/autoports/ornith_ai_ornith_1_0_35b \
  --hf-model ornith-ai/Ornith-1.0-35B \
  --mesh-device "(1, 4)" \
  --max-num-seqs "$ORNITH_MAX_NUM_SEQS" \
  --api-server-count "$ORNITH_API_SERVER_COUNT" \
  --max-model-len 262144 \
  --block-size 64 \
  --server-timeout 3600 \
  --port 8100 \
  --tt-config "$ORNITH_TT_CONFIG" \
  --additional-server-args="--host 127.0.0.1 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml"
```

A quiet startup of several minutes is normal. Wait for `Server ready at http://localhost:8100`.

## 4. Verify the endpoint

```bash
curl -fsS http://127.0.0.1:8100/health

curl -fsS http://127.0.0.1:8100/v1/models | python -c \
  'import json,sys; m=json.load(sys.stdin)["data"][0]; print(m["id"], m["max_model_len"])'

curl -fsS http://127.0.0.1:8100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ornith-ai/Ornith-1.0-35B",
    "messages": [{"role": "user", "content": "What is 17 multiplied by 19?"}],
    "max_tokens": 2048,
    "temperature": 0
  }' | python -m json.tool
```

The post-optimization [evaluation summary](../post_optimization_eval/RESULTS.md) records the current
API, device, quality-subset, and finalized performance checks. Its sibling `results.json` is the
canonical compact artifact for downstream release tooling.
