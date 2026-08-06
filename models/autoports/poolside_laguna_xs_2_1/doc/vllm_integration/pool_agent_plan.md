# `pool` coding agent on Laguna-XS-2.1 — plan (not yet executed)

**Status: PLAN ONLY, written 2026-08-06. Nothing here has been run.** No parser has been vendored,
no launcher or serve script changed, no server booted, no `pool` installed. Every step below is
future work. Findings recorded here were established by reading code/config on disk and querying
the GitHub API; the device was not touched.

Goal: drive agentic coding on the P150×4 Laguna-XS-2.1 vLLM server with **`pool`**, poolside's own
terminal coding agent, by bringing our TT plugin in line with upstream vLLM main — rather than
patching the fork or working around it with launch flags.

---

## Why it does not work today

`pool` has a first-class standalone mode for any OpenAI-compatible server
(`POOLSIDE_STANDALONE_BASE_URL` / `_MODEL` / `_CONTEXT_LENGTH` + `POOLSIDE_API_KEY`), so no adapter
or proxy is needed. The blocker is our serving config, not the agent.

Poolside ships the intended serving contract in-band, in the HF repo we already serve —
`~/.cache/huggingface/hub/models--poolside--Laguna-XS-2.1/snapshots/e9df9a5…/generation_config.json`:

```json
"temperature": 1.0, "top_p": 1.0, "top_k": 20, "min_p": 0.0, "max_new_tokens": 32768,
"tool_call_parser": "poolside_v1",
"reasoning_parser":  "poolside_v1",
"default_chat_template_kwargs": { "enable_thinking": true }
```

Our stack honours the *sampling* keys — `ModelConfig.generation_config` defaults to `"auto"`, so
temp 1.0 / top_p 1.0 / top_k 20 / max_tokens 32768 already apply as request defaults — but has
neither `poolside_v1` parser, and it ignores the declared `default_chat_template_kwargs`
(`ModelConfig.get_diff_sampling_param` consumes only the sampling keys). We substitute `glm47` +
`deepseek_r1`, and `enable_thinking` therefore defaults to the template's `false`.

The chat template (`laguna_glm_thinking_v8` lineage) gates the think block on that flag:

```jinja
{%- if add_generation_prompt -%}{{- "<assistant>" -}}
  {%- if enable_thinking -%}{{- '<think>' -}}{%- else -%}{{- '</think>' -}}{%- endif -%}
```

With thinking off the **prompt** closes the think block, so the model's output contains no
`</think>`. `--reasoning-parser deepseek_r1` assumes it starts *inside* reasoning and switches on
`</think>`, so it files the **entire** output — `<tool_call>` included — under `reasoning` and
leaves `content` empty. `glm47` then parses nothing and every agent turn returns `tool_calls: []`.

**Interim unblock (one flag, if someone needs pool working before this plan lands):** add
`--default-chat-template-kwargs '{"enable_thinking":true}'` to the server args. The flag exists in
our fork (`vllm/entrypoints/openai/cli_args.py:120`; request-level values take precedence, so
`scripts/humaneval_served.py`, which sends `enable_thinking:false`, is unaffected). Quoting gotcha:
the runner does `shlex.split(args.additional_server_args)` (`run_vllm_server.py:952`), which strips
quotes — bare `{"enable_thinking":true}` arrives as invalid `{enable_thinking:true}` and
`json.loads` fails. The JSON must reach shlex still single-quoted; hoist the arg string into a
variable passed through `env` rather than nesting quotes inside `bash -c "… '…'"`.

## Where the real code lives

`tenstorrent/vllm` does **not** have these parsers — checked `dev` and `main` via the GitHub API;
both carry only `glm4_moe` / `glm47_moe` and `deepseek_r1` / `deepseek_v3`. So syncing our fork to
its own remote buys nothing here. They exist only in `vllm-project/vllm` main.

Our plugin is already the established home for exactly this. `vllm_tt_plugin/` ships
`gemma4_tool_parser.py` (362 ln) and `gemma4_reasoning_parser.py` (135 ln), registered lazily in
`entrypoints.py::register()` via `ToolParserManager.register_lazy_module` /
`ReasoningParserManager.register_lazy_module`, with the docstring stating the rationale: *"Kept in
the plugin (rather than patched into `vllm.reasoning`) so it carries over unchanged when switching
to upstream vLLM."* `register()` is the `vllm.general_plugins` entrypoint and
`load_general_plugins()` runs in the frontend path (`engine/arg_utils.py:615`), which is where
parsers resolve. This plan is a second instance of a proven pattern.

### Provenance to vendor (pinned)

From **`vllm-project/vllm` @ `8543522ca792de824c026e1b9e3eb51ca809550d`** (main, 2026-08-05),
Apache-2.0 — same licence and SPDX header style as the existing plugin files:

| upstream file | size | destination |
| --- | --- | --- |
| `vllm/tool_parsers/poolside_v1_tool_parser.py` | 24,313 B (~570 ln) | `vllm_tt_plugin/poolside_v1_tool_parser.py` |
| `vllm/reasoning/poolside_v1_reasoning_parser.py` | 3,073 B (~80 ln) | `vllm_tt_plugin/poolside_v1_reasoning_parser.py` |

`PoolsideV1ToolParser` subclasses `ToolParser`; its own docstring reads *"GLM-4 Tool Call Parser
with incremental string streaming support"* and it exists to stream long string args (4000+ chars of
code) incrementally instead of buffering until complete — directly relevant to a coding agent, and
confirmation that our `glm47` stand-in was at least the right lineage. `PoolsideV1ReasoningParser`
subclasses `DeepSeekV3ReasoningParser` and decides reasoning-vs-content by walking token ids
backward, terminating at the `<assistant>` start-of-message token.

### Import compatibility — checked against our fork (`cd64666bcf77`, 2026-07-24)

| symbol | status |
| --- | --- |
| `DeepSeekV3ReasoningParser`, `ToolParser` / `ToolParserManager` | present |
| `make_tool_call_id`, `ChatCompletionNamedToolChoiceParam`, `ResponsesRequest`, `TokenizerLike` | present |
| `vllm.entrypoints.openai.engine.protocol` delta/tool types | present (in-plugin `gemma4` uses the same module) |
| `openai.types.responses.ToolChoiceFunction` | present (openai 2.44.0 in `.tenstorrent-venv`) |
| `tool_parsers.utils.safe_literal_eval` | **missing — vendor (4 ln)** |
| `tool_parsers.utils.partial_tag_overlap` | **missing — vendor (11 ln)** |

Laguna's structural tokens are real single tokens, which the reasoning parser's backward walk
requires: `<think>`=18, `</think>`=19, `<assistant>`=23, `<tool_call>`=25, `</tool_call>`=26.

**Hypothesis worth testing early:** because `is_reasoning_end` is also called with
`prompt_token_ids` (`entrypoints/openai/chat_completion/serving.py:837`), the backward walk should
find the `</think>` that thinking-*off* puts in the prompt and correctly report reasoning-already-
ended — making tool calling work in **both** template modes with no client-side kwargs, and
demoting the interim flag to a quality choice. Not a given; the offline test below settles it.

**Explicitly not doing:** syncing the `.local` vLLM checkout to `origin/dev` (behind: `cd64666bc`
vs `e93a9d6e`) — it does not contain the parsers, and rebasing the pinned known-good serving tree
is a separate, riskier change. Also not monkeypatching `FrontendArgs` defaults from the plugin;
that is reachable but buys nothing over the launcher change in step 4.

---

## Steps

Steps 1–2 land in the installed plugin tree
`/home/ttuser/.local/lib/model-bringup/tt-metal/vllm` (its own git repo — commit there separately
from this repo). Note STATUS.md already records an **uncommitted plugin diff** there
(`reset_batch=True`); don't clobber it.

### 1. Vendor the two parsers + the two missing helpers

- `src/vllm_tt_plugin/poolside_v1_tool_parser.py`, `src/vllm_tt_plugin/poolside_v1_reasoning_parser.py`
  — verbatim from the pinned SHA, upstream SPDX headers kept, plus a one-line provenance comment
  naming the source path and commit so the next sync is mechanical.
- `safe_literal_eval` + `partial_tag_overlap` into `src/vllm_tt_plugin/poolside_v1_utils.py`, also
  verbatim from upstream `vllm/tool_parsers/utils.py`; repoint those imports.
- Change **only** import paths. Any further edit is a finding to record in the commit message — it
  is what the next upstream sync has to re-apply.

### 2. Register both, mirroring the `gemma4` precedent

In `src/vllm_tt_plugin/entrypoints.py`:

```python
ToolParserManager.register_lazy_module(
    "poolside_v1", "vllm_tt_plugin.poolside_v1_tool_parser", "PoolsideV1ToolParser")
ReasoningParserManager.register_lazy_module(
    "poolside_v1", "vllm_tt_plugin.poolside_v1_reasoning_parser", "PoolsideV1ReasoningParser")
```

Lazy registration keeps a broken import from affecting boots that don't select these parsers.

### 3. Offline pytest first — this is the inner loop

Pure CPU, no device: load the real tokenizer from the HF snapshot on disk and assert against canned
outputs.

- non-streaming tool call, thinking-on shape (`reasoning… </think>` then `<tool_call>…`)
- non-streaming tool call, thinking-off shape (no `</think>` in output; pass prompt ids ending in
  `</think>` to `is_reasoning_end`) — **this is the both-modes hypothesis**
- multi-tool-call in one message; a tool call whose arg is a 4 KB code string
- streaming: feed deltas token-by-token, assert incremental string streaming and that the assembled
  `tool_calls` match the non-streaming result
- reasoning-only output with no tool call

Write this **before** booting anything: it turns a 15–20 min device cycle into seconds and is what
catches upstream API drift across ~570 vendored lines.

### 4. Make it OOTB: launcher reads `generation_config.json`

`models/common/readiness_check/run_vllm_server.py` is ours (already patched once for
`--additional-config`). Have `_launch_server` read the model's `generation_config.json` and pass
`--tool-call-parser`, `--reasoning-parser` and `--default-chat-template-kwargs` when the caller
hasn't set them — mirroring what newer upstream vLLM does natively. Then no launch line needs
parser flags, for Laguna or any future model shipping them. Building the list in Python also side-
steps the `shlex` quoting trap described above.

### 5. Boot and verify on device

Reuse `scripts/stage_ce_serve.sh`, swapping `--reasoning-parser deepseek_r1 --tool-call-parser
glm47` for the `poolside_v1` pair (or dropping both once step 4 lands). Keep the `lsof` holder
sweep, `tt-smi -r all`, `env_passthrough`, and the `: > readiness_vllm/server.log` stale-fatal-
marker truncation — all load-bearing. ~15–20 min warmup.

Before touching pool, confirm a request carrying **no** `chat_template_kwargs` gets a parsed tool
call — exactly what pool sends:

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"poolside/Laguna-XS-2.1","max_tokens":2048,
  "messages":[{"role":"user","content":"What files are in /tmp? Use the tool."}],
  "tools":[{"type":"function","function":{"name":"run_bash","description":"Run a bash command",
    "parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}]}' \
| python3 -c 'import sys,json;c=json.load(sys.stdin)["choices"][0];print("finish:",c["finish_reason"]);print("tool_calls:",c["message"].get("tool_calls"))'
```

Expect `finish: tool_calls` and non-empty `tool_calls`. Repeat with `"stream":true` — streaming is
the path pool uses and where tool-parser bugs hide.

### 6. Install and run `pool`

```bash
curl -fsSL https://downloads.poolside.ai/pool/install.sh | sh
```

Lands in `~/.local/bin/pool`, no sudo (already on PATH). The installer prompts for **EULA
acceptance** and offers to edit the shell rc; `POOL_INSTALL_ACCEPT_EULA=1
POOL_INSTALL_UPDATE_PATH=0` makes it non-interactive. No `pool login` needed — standalone mode
authenticates from the env.

```bash
POOLSIDE_STANDALONE_BASE_URL="http://127.0.0.1:8000" \
POOLSIDE_API_KEY="EMPTY" \
POOLSIDE_STANDALONE_MODEL="poolside/Laguna-XS-2.1" \
POOLSIDE_STANDALONE_CONTEXT_LENGTH=131072 \
pool -C /path/to/repo
```

- **Base URL without `/v1`** — pool appends the path (its docs key the model-list override off "the
  provider's `/v1/models` response"; the README's llama.cpp example is a bare host:port). If it
  can't see the model, retry with `.../v1`.
- **131072, not 262144** — the servable ceiling; pool uses it only to size auto-compaction.
- **One session at a time.** Concurrent long-context agent decode has crashed EngineCore with
  `Bus error (7)` → `EngineDeadError`. No second `pool` or bench against the same server.
- No sampling flags needed — temp 1.0 / top_p 1.0 / top_k 20 come from `generation_config.json`.
- `--sandbox disabled` if pool's sandbox balks here; `--mode accept-edits` cuts approval prompts.
- Expect ~20 t/s/u single-user with always-on verbose reasoning: minutes per turn. Usable, not snappy.

### 7. Commit + record

Two scoped commits — plugin repo (vendored parsers, registration, tests) and this repo (launcher
auto-config, serve script, STATUS.md). Update STATUS.md's serving section to name `poolside_v1`
instead of `glm47`/`deepseek_r1` once validated. Per repo convention, no run outputs or agent
trajectories get committed.

## Verification

1. `pytest` on the offline parser tests — green, including the thinking-off case.
2. `python -c "import vllm_tt_plugin; print(vllm_tt_plugin.__file__)"` resolves under `.local/`
   before trusting any on-device result — the editable install can silently point at another
   checkout, and then you are testing a file the server never loads.
3. Server log shows `tool_call_parser='poolside_v1'` and `reasoning_parser='poolside_v1'`.
4. Bare-tool-call curl (step 5) non-streaming **and** streaming → `finish_reason: tool_calls`,
   non-empty `tool_calls`, no `<think>` leak in `content`.
5. `pool exec -p "…" -o json --unsafe-auto-allow` exits 0 with executed tool calls in the trajectory.
6. Regression: `scripts/humaneval_served.py --chat` (sends `enable_thinking:false`) behaves as
   before on a couple of problems; the plugin's `gemma4` parsers still register.

## Estimate

≈ **1 day** — vendor + import fixups 2–4 h (all the risk: ~570 lines written against a newer vLLM
than our fork, though only two util helpers are known missing); registration 15 min; offline tests
1–2 h; launcher auto-config 1–2 h; on-device pass 1–2 h mostly boot wait; doc/commit 30 min.
~0.5 day if the vendored files drop in clean. No fork diff, and it retires the `glm47` +
`deepseek_r1` stand-ins.

## Follow-ups (out of scope)

- **Upstream the vendored parsers into `tenstorrent/vllm`** so the plugin copy can eventually be
  deleted; or revisit when the `.local` tree rebases onto a vLLM that already has them.
- **`dflash` speculative decoding** — `generation_config.json` declares
  `poolside/Laguna-XS-2.1-DFlash`, 15 speculative tokens. Separate from the ngram spec-decode work
  (see STATUS.md) and a far bigger lever on agent latency than anything in this plan.
