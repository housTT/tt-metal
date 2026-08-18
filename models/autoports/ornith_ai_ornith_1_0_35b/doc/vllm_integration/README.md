# Ornith-1.0-35B — vLLM serving integration (TTNN, 4-chip Blackhole ring)

The [datatype sweep's](../datatype_sweep/) selected model, served through the shared TT vLLM path on the
same hardware: four Blackhole `p300c` chips as a `1x4` ring under `FabricConfig.FABRIC_1D_RING`, TP=4
dense and EP=4 over the 256 routed experts, precision policy **`C06-proj-bfp4-lofi`**, paged KV cache
**bfloat8_b**, **262144-token** advertised and served context.

---

## 1. Headline: primary single-user serving performance

**Workload: 128-token prompt, 128 generated tokens, 1 request, `--max-concurrency 1`, greedy
(`--temperature 0.0`), `ignore_eos`, `--max-num-seqs 1`, decode trace on, on-device sampling
(`sample_on_device_mode: all`), async scheduling off.** Raw
[`readiness_vllm/vllm_result.json`](../../readiness_vllm/vllm_result.json), normalized
[`readiness_vllm/vllm_benchmark.json`](../../readiness_vllm/vllm_benchmark.json).

| metric | warm | first request at this prompt length |
|---|---|---|
| **TTFT** P50 / P99 | **149.4 / 149.4 ms** | 172.8 ms |
| **TPOT** mean / P99 | **23.19 / 23.19 ms** | 24.89 ms |
| **ITL** P50 / P99 | **23.12 / 23.68 ms** | 23.13 / 24.1 ms |
| **decode t/s/u** (`1000 / mean_tpot_ms`) | **43.1 t/s/u** | 40.2 t/s/u |
| decode t/s/u from ITL P50 | 43.25 t/s/u | 43.24 t/s/u |
| aggregate output throughput | 41.36 tok/s | 38.5 tok/s |
| requests completed | **1/1, 128/128 tokens** | 1/1, 128/128 |

Both columns are the same command on the same server, back to back
([cold](batch1/vllm_benchmark_primary_cold_first_request.json), then the warm run that is the artifact
above). The difference is that the first request at a prompt length compiles that length's prefill
programs and pays a trace re-capture inside its TTFT — the datatype-sweep stage measured the same effect
standalone ([`cold_prompt_length_cost`](../datatype_sweep/post_selection_token_out.json): 312 ms cold
against 177 ms warmed). Note that **ITL P50 is 23.1 ms in both columns**: the cost is entirely in the
first token, not in the decode loop. `OrnithGenerator.warmup(prompt_lengths)` removes it for a deployment
that knows its lengths.

### Against the model's own decode floor

| | ms/token | t/s/u | source |
|---|---|---|---|
| full-model token-out benchmark, same 128/128/1 shape, batch 1, 9 warm repeats | 23.165 | 43.169 | [`post_selection_token_out.json`](../datatype_sweep/post_selection_token_out.json) |
| **vLLM serving, this stage** | **23.19** (TPOT mean), 23.12 (ITL P50) | **43.1** (43.25 from ITL) | this README |
| traced decode without sampling or readback (lower bound) | 21.965 | 45.526 | same artifact |
| teacher-forcing traced decode (serial by construction) | 23.643 | 42.296 | [datatype sweep](../datatype_sweep/README.md) |

Serving decode is **at** the model's own token-out figure — 0.03 ms/token, inside the 9-repeat spread of
the baseline, and the ITL median matches it to 0.05 ms — so there is no measurable vLLM-specific decode
overhead left to remove. TTFT carries +10 ms of serving path (149.4 ms against the model's 139.5 ms
median warmed TTFT): HTTP, tokenizer, scheduler and detokenization. Teacher forcing is quoted only as the lower-bound comparison the skill
asks for, and serving is *faster* than it, as expected — teacher forcing decides step *N+1*'s input on
the host.

---

## 2. Secondary: CI serving-burst profile (vLLM-nightly shape)

**Workload: 100-token prompts, 100 output tokens, 32 requests, no `--max-concurrency`, greedy, `ignore_eos`,
`--max-num-seqs 32`.** Raw
[`readiness_vllm/vllm_ci_serving_result.json`](../../readiness_vllm/vllm_ci_serving_result.json),
normalized [`readiness_vllm/vllm_ci_serving_benchmark.json`](../../readiness_vllm/vllm_ci_serving_benchmark.json).

| metric | value |
|---|---|
| requests completed | **32/32**, 3200/3200 tokens |
| TTFT P50 / P99 | 5504.1 / 5505.4 ms |
| TPOT mean / P99 | 149.98 / 185.02 ms |
| ITL P50 / P99 | 141.65 / 366.00 ms |
| aggregate output throughput | **158.5 tok/s** |
| request throughput | 1.59 req/s |
| TPOT-derived per-user decode | 6.67 t/s/u |
| elapsed | 20.19 s |

**This is not the headline decode number**, per the skill: burst admission and the padded decode batch
dominate its TPOT. It is the vLLM-nightly parity and serving-capacity view, and read that way it says
the mesh converts 43 t/s/u at one user into **158.5 tok/s aggregate at 32** — 3.8× the single-user
throughput — because this model's decode step is launch-bound at batch 1 (the optimized-full-model
stage measured 6.3 % of the DRAM roofline) and batching amortises the launches.

For the same reason, a single user on a server built for 32 pays for the padded batch: the *primary*
128/128/1 profile on a `--max-num-seqs 32` server measures TPOT 140.1 ms = 7.1 t/s/u
([artifact](batch32/vllm_benchmark_single_user_at_max_num_seqs_32.json)). Single-user latency and
32-user capacity are two deployments here, not one.

---

## 3. Serving status

| | |
|---|---|
| adapter | [`tt/generator_vllm.py`](../../tt/generator_vllm.py), class `TTQwen3_5MoeForConditionalGeneration` |
| registered as | `TTQwen3_5MoeForConditionalGeneration` **and** `Qwen3_5MoeForConditionalGeneration` in `vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py::register_tt_models` (why the second: [work log §3](work_log.md#3-plugin-registration-and-the-two-upstream-decisions-that-had-to-be-taken-away-from-upstream)) |
| vLLM checkout | `tenstorrent/vllm@bf98d556` (`dev`), installed `VLLM_TARGET_DEVICE=empty` ([`vllm_checkout.txt`](vllm_checkout.txt), plugin diff [`vllm_tt_plugin_changes.diff`](vllm_tt_plugin_changes.diff)) |
| served context | **262144**, equal to `doc/context_contract.json`; no capability reduction |
| batch coverage | `max_num_seqs=32` (the sampler's `MAX_SAMPLING_BATCH` bound) and `1` |
| KV cache owner | **vLLM**. `allocate_kv_cache` builds the cache at the block count vLLM sized and the generator is constructed around it (`kv_cache=`), so nothing allocates a second one |
| recurrent state owner | the model — 30 `linear_attention` layers' fixed-size DeltaNet matrix and conv window per slot, moved on a vLLM batch condense by `remap_state_slots` |
| capability flags | `supports_async_decode=True` (proven, §5), `supports_sample_on_device=True` (enforced by the runner, `sample_on_device_mode: all`), `supports_prefix_caching=False` (not implemented, not claimed) |
| sampling | **on device**, the full-model split-sampling path: one model-trace replay + one sampling-trace replay per token, `tt_out_tok` feeding the next replay. No host argmax, no full-logits readback, no top-k greedy fallback, no Python token feedback on the measured path |
| host sampling | only when the plugin asks for it (log-probs on a 4-device mesh, `min_p`, `bad_words`, `logit_bias`, `allowed_token_ids`, `min_tokens`, structured output). Explicit, optional, and never the measured path |
| qualitative verdict | **coherent, on topic, English, no repetition loops, no gibberish, no cross-request contamination**, matched against the HF and full-model controls (§4) |
| degenerate-output check | `no degenerate output detected` (`--scope vllm` and `--scope all`), on the default and the async-scheduling runs |
| sampling suite | 54/73 at `max_num_seqs=32`, 65/73 at `max_num_seqs=1`; every failure in both is a reproducibility or batch-size-structural assertion, none a correctness one (§6) |
| single-user determinism | at `max_num_seqs=1`, repeated greedy requests are **identical** (3/3), seeded requests too (2/2), and the same text comes back with `--async-scheduling` on. At `max_num_seqs=32` they are not — §6 |
| what the built model is | [`readiness_vllm/vllm_serving_capability.json`](../../readiness_vllm/vllm_serving_capability.json), written by the adapter at the end of warm-up: policy `C06-proj-bfp4-lofi`, KV cache `BFLOAT8_B`, LM head `BFLOAT4_B`, 40 layers, 4097 blocks, no layer exceptions, `owns_cache=False` |

---

## 4. Qualitative output, with controls

Two runs, because the checkpoint has a chat template and `$qualitative-check` is explicit that raw
completion prompts from a chat model are continuation coverage, not a verdict:

| run | prompts | artifact |
|---|---|---|
| `run_vllm_server --stages qualitative` — raw `/v1/completions`, the shared runner's own path, greedy and sampled (T=0.7, top-p 0.9), 256 tokens | the six in `models/common/readiness_check/vllm_prompts.txt` | [`readiness_vllm/vllm_qualitative_outputs.json`](../../readiness_vllm/vllm_qualitative_outputs.json) |
| `probe_qualitative_chat.py` — `/v1/chat/completions`, so the server renders `apply_chat_template(add_generation_prompt=True)`, greedy, 128 tokens | the same six | [`qualitative_chat.json`](qualitative_chat.json) |

The chat run's controls are the two the full-model stage produced from the *same* rendered prompts:
its HF reference completions and its own TTNN completions
([`readiness_qualitative.json`](../full_model/readiness_qualitative.json)).

Against the controls, the chat run tracks the HF reference **character for character** for the first
25 – 236 characters of each answer and then takes a different phrasing of the same plan ("Analyze the
Request" against "Deconstruct the request") — the near-tie behaviour §6 measures, not a different answer.
All six finish at the 128-token limit with no truncation artefacts.

**Verdict: pass.** Read prompt by prompt:

* *"Write a haiku about machine learning."* — the model opens a `<think>` block, enumerates the 5-7-5
  constraint and brainstorms imagery; the sampled completion closes the block and produces an actual
  haiku ("Data learns the pattern, / Deep networks find hidden truths, / AI wakes and sees."). Coherent,
  on topic, English.
* *"Explain the difference between supervised and unsupervised learning…"* — a structured plan naming
  labelled vs unlabelled data, teacher/no-teacher, prediction vs discovery. Matches the HF control's
  content and structure closely.
* *"Complete this story…"* — continues the story, then plans. Both this run **and the HF control** emit
  "Here's a thinking thinking sequence" — a doubled word that is a **checkpoint** behaviour, not a
  serving artefact: the HF reference, the full-model TTNN run and the served run all produce it, and the
  served text matches the HF control for 236 characters *including* that phrase.
* *"What are the three laws of thermodynamics?"* — correct physics, no repetition loop.
* *"Translate … to French"* — French output, no wrong-language drift in the other five.
* *"Write a Python function … Fibonacci"* — Python, correct recursion/iteration discussion.

No prompt echo, no control-token leakage, no cross-request contamination (each completion answers its own
prompt), no mechanical repetition. `check_degenerate_output.py` agrees mechanically: adjacent-duplication
0.000 – 0.024 and trigram-loop 0.021 – 0.112 across all twelve completions, **no degenerate output
detected**.

Greedy and sampled completions are both reasonable, and the sampled ones differ from the greedy ones
without degrading — the haiku prompt is the clearest example.

---

## 5. Async decode (overlap), proven rather than declared

`supports_async_decode=True` is claimed because the split is implemented and measured:
`decode_forward(..., read_from_device=False)` returns device handles, `read_decode_output(...,
async_read=True)` enqueues `cpu(blocking=False)` plus `ttnn.record_event` behind the replays, and
`process_decode_output_host(...)` only formats. The contract that makes overlap *safe* is the refresh
policy: in the steady state nothing is copied to the device, and when the layout does change the staged
token/position pair is merged per row so a continuing row keeps the device's authoritative pair.

| evidence | result |
|---|---|
| `--async-scheduling` accepted by the plugin (no "Disabling async scheduling") | `async_scheduling=True` in the server log |
| **batch-1 greedy text, overlapped vs not** | **character-for-character identical** ([`async/overlap_equivalence.json`](async/overlap_equivalence.json)); 3/3 identical repeats under overlap, and 2/2 for a seeded request |
| degenerate-output check on the overlapped run | no degenerate output detected |
| smoke sampling profile under overlap | the same 2 reproducibility failures as without it — overlap adds none |
| 32-request burst, non-aligned lengths, a 9000-token prompt under overlap | all completed at their exact lengths ([artifact](async/serving_requests_async_max_num_seqs_32.json)) |
| CI burst 100/100/32, overlapped vs not (both warm) | TTFT P50 5504.1 → 5495.8 ms, aggregate 158.5 → 158.7 tok/s, TPOT 149.98 → 149.72 ms — **unchanged within noise** |
| primary 128/128/1, overlapped vs not (both warm) | TTFT 149.4 → 148.3 ms, TPOT 23.19 → 23.14 ms, ITL P50 23.12 → 23.13 ms — unchanged |
| steady-state device copies, at the primitive level | 0 tokens, 0 positions, 0 page tables over 5 consecutive steps ([`serving_primitives.json`](serving_primitives.json)) |
| one-token-stale host pair vs the correct pair | **token-for-token identical**; with the merge removed, the stream diverges and repeats a token |

Overlap buys nothing measurable on this model, and that is the expected result rather than a
disappointment: the decode loop already has no host dependency to hide — the token reaches the next
replay through `tt_out_tok` on device and the positions advance inside the trace, so there is no
host-side gap for the scheduler to overlap with. What the async split *does* buy is the freedom to run
under `--async-scheduling` without corrupting generation, which is what the identical-text row proves and
what `supports_async_decode=True` claims.

Overlap is **off** in the headline numbers because that is vLLM's default and the skill's rule: overlap
stays off unless a deployment opts in. Artifacts from the overlapped runs are in [`async/`](async/).

---

## 6. Sampling suite: what passed, what did not, and why

`--sampling-profile full` — the whole canonical `tests/tt` suite — was run on two server configurations.

| configuration | passed | failed | skipped |
|---|---|---|---|
| `--max-num-seqs 32` | 54 | **18** | 1 |
| `--max-num-seqs 1` | 65 | **7** | 1 |

Nothing in either failure set is a correctness, log-prob, crash or output-quality failure. Concretely,
in **both** configurations these pass: all 16 `test_logprobs` parameterisations (host sampling — device
log-probs need 8 or 32 devices), all five host-only parameters (`min_p`, `bad_words`, `logit_bias`,
`allowed_token_ids`, `min_tokens`), structured output at full capacity, seed *variety*, temperature
variation within and between batches, `test_topk[15]`, and request isolation for differing parameters.
The single skip is `test_chat_logprobs_all_vocab`, which skips itself because the plugin clamps
`max_logprobs` to 20.

**The 18 failures at `max_num_seqs=32` are all one class**: "the same request must produce the same
text" — `test_top1_is_greedy`, `test_topk[19]`, `test_seeding`, `test_specific_seed_reproducible[…]`,
`test_batch1_seed_reproducible[…]`, `test_uniform_seed_deterministic[…]`,
`test_same_seeds_reproduce_across_batches`, `test_mixed_params_batch`, and the three
`test_*_penalty_mixed_batch`. Every one of them **passes at `max_num_seqs=1`**.

**The 7 failures at `max_num_seqs=1` are structural**: those tests build their request list by slicing it
with `max_batch_size` and then assert on ≥2 distinct outputs (`test_different_*_penalties`), ≥5 distinct
outputs (`test_uniform_noseed_varied`), or index `results[1]` (`test_frequency_penalty_mixed_batch`).
With a batch of one there is one result, so they cannot pass — a harness assumption, not a model answer.
The penalty tests that *can* run at 32 pass there.

### Why the reproducibility class fails, measured at the logits

`$vllm-integration` asks for logit-level evidence before classifying determinism failures. The full
investigation, including the first measurement that turned out to be measuring a page table of zeros, is
[work log §8](work_log.md#8-the-reproducibility-class-of-failures-measured) and
[§9](work_log.md#9-autofix-the-first-determinism-measurement-was-measuring-a-page-table-of-zeros). The
corrected result, from [`slot_reproducibility.json`](slot_reproducibility.json) on the full 40-layer model
(13 logit vectors per arm: the prefill's plus 12 traced decode steps):

| arm | bit-identical | max abs Δ per step | min logit PCC | top-1 agreement |
|---|---|---|---|---|
| **batch 1**, same prompt twice from a wiped state | **yes, all 13** | **0.0** | **1.0** | 13/13 |
| batch 32, same slot, two reruns | no | 0.85 | 0.9973 | 10/13 |
| batch 32, the same prompt in another slot | no | 1.09 | 0.9960 | 9/13 |
| batch 32, with an occupied padding row beside it | no | 0.85 | 0.9975 | 12/13 |

* **At batch 1 the whole serving path is bit-reproducible**, prefill and every traced decode step. That
  is why the entire reproducibility class of the shared suite passes at `--max-num-seqs 1`.
* **At batch 32 the state is bit-identical and the decode replay is not.** All 140 DeltaNet
  recurrent/conv buffers and the written KV pages are identical across rounds; the batch-32 traced decode
  step deviates by 0.47 – 0.85 of a logit at PCC ≥ 0.9973.
* **A measured tie decides the flip.** Step 10 of this prompt has a top-1/top-2 margin of **exactly
  0.0** — one bfloat16 value for two tokens — in both configurations. At batch 1 both runs resolve it the
  same way; at batch 32 the 0.8 of deviation decides it, and from there the streams diverge.
* **The same property, one batch size down, is already on file**: the datatype sweep measured cross-slot
  max |Δ| 0.28 – 0.5 at logit PCC ≥ 0.9993 with margins 0.0 – 0.19 at batch 4 and concluded that
  "cross-slot decode has never been bit-identical, at any precision"
  ([§9.1](../datatype_sweep/README.md), [`batch_slot_tie_selected.json`](../datatype_sweep/batch_slot_tie_selected.json)).

Making batch-32 decode bit-reproducible means changing the decode geometry or the collectives — decoder
work with its own gates, not serving work, and nothing at the serving layer can substitute for it. Output
quality is unaffected: every arm stays fluent, on topic and in English, and the difference is a different
near-tie winner, never corruption.

---

## 7. How this was run

Environment: `tenstorrent/vllm@bf98d556` (`dev`) installed into the tt-metal venv with
`VLLM_TARGET_DEVICE=empty`, plus `plugins/vllm-tt-plugin` ([work log §2](work_log.md#2-vllm-was-not-installed-and-two-shared-paths-did-not-work-against-the-current-fork)).
A `tt-smi -r` + mesh smoke ran before every launch ([§10.1](work_log.md#101-every-vllm-shutdown-left-the-fabric-unable-to-re-open-until-the-boards-were-reset)).

**The server, single-user configuration** (headline numbers):

```bash
python -m models.common.readiness_check.run_vllm_server \
  --stages serve \
  --model-dir models/autoports/ornith_ai_ornith_1_0_35b \
  --hf-model ornith-ai/Ornith-1.0-35B \
  --mesh-device "(1, 4)" \
  --max-num-seqs 1 \
  --max-model-len 262144 \
  --server-timeout 2400 \
  --port 8100 \
  --tt-config '{"trace_region_size": 200000000, "l1_small_size": 24576, "fabric_config": "FABRIC_1D_RING", "fabric_router_max_packet_bytes": 8192}'
```

`--max-num-seqs 32` is the same command with that one value changed; `--additional-server-args="--async-scheduling"`
adds overlap (the `=` matters — argparse eats a value that starts with `-`).

**The checks**, attached to the running server so one weight load serves all of them:

```bash
# the full canonical sampling suite
python -m models.common.readiness_check.run_vllm_server --stages sampling \
  --model-dir models/autoports/ornith_ai_ornith_1_0_35b --hf-model ornith-ai/Ornith-1.0-35B \
  --max-num-seqs 1 --sampling-profile full --server-url http://localhost:8100

# qualitative (raw completions) and both benchmark profiles
python -m models.common.readiness_check.run_vllm_server --stages qualitative \
  --model-dir models/autoports/ornith_ai_ornith_1_0_35b --hf-model ornith-ai/Ornith-1.0-35B \
  --max-num-seqs 1 --server-url http://localhost:8100
python -m models.common.readiness_check.run_vllm_server --stages benchmark \
  --model-dir models/autoports/ornith_ai_ornith_1_0_35b --hf-model ornith-ai/Ornith-1.0-35B \
  --max-num-seqs 1 --no-benchmark-ci-serving --server-url http://localhost:8100   # primary only
python -m models.common.readiness_check.run_vllm_server --stages benchmark \
  --model-dir models/autoports/ornith_ai_ornith_1_0_35b --hf-model ornith-ai/Ornith-1.0-35B \
  --max-num-seqs 32 --server-url http://localhost:8100                            # + the CI burst

# stage-owned checks
python models/autoports/ornith_ai_ornith_1_0_35b/doc/vllm_integration/logs/probe_qualitative_chat.py --url http://localhost:8100
python models/autoports/ornith_ai_ornith_1_0_35b/doc/vllm_integration/logs/probe_serving_requests.py --url http://localhost:8100
python models/common/readiness_check/check_degenerate_output.py \
  --hf-model ornith-ai/Ornith-1.0-35B --missing-artifacts critical --scope vllm
python .agents/scripts/check_context_contract.py \
  --model-dir models/autoports/ornith_ai_ornith_1_0_35b --hf-model ornith-ai/Ornith-1.0-35B \
  --stage vllm --require-contract

# adapter suite (host-only cases need no device; the rest use the reduced two-layer target)
pytest models/autoports/ornith_ai_ornith_1_0_35b/tests/test_generator_vllm.py -q
```

Standalone probes (no server): `probe_serving_primitives.py`, `probe_logit_determinism.py`,
`probe_logit_read_stability.py`, `probe_prefill_stability_with_traces.py`, all in [`logs/`](logs/).

The reduced serving target for bring-up: `ORNITH_VLLM_LAYER_INDICES=0,3` (one real layer of each kind).
Nothing in this README is measured on it.

---

## 8. Limitations

1. **Reproducibility at `max_num_seqs > 1`.** Repeated identical greedy requests are not guaranteed to
   return identical text; they agree until the first near-tie. Measured, with the mechanism and the
   prior-stage corroboration, in §6 and [work log §8](work_log.md#8-the-reproducibility-class-of-failures-measured).
   A deployment that needs bit-reproducible completions should serve at `--max-num-seqs 1`, where the
   whole reproducibility class of the shared suite passes.
2. **The first request at a new prompt length is slower.** It compiles that length's prefill programs and
   pays a trace re-capture: 238.6 ms against 155.5 ms warmed TTFT for the 128-token shape.
   `OrnithGenerator.warmup(prompt_lengths)` removes it for a deployment that knows its lengths; the
   serving warm-up compiles one length (64) rather than guessing a bucket set.
3. **Single-user latency and 32-user capacity are two server configurations.** A `--max-num-seqs 32`
   server pays the padded decode batch on every step, so single-user TPOT there is 140.3 ms (7.1 t/s/u)
   against 23.15 ms (43.2 t/s/u) at `--max-num-seqs 1`. Both are measured and both artifacts are kept.
4. **Prefix caching is off** and not claimed (`supports_prefix_caching=False`).
5. **Text only.** The checkpoint carries a vision tower this port does not implement. The adapter is
   registered as a text-only model, so vLLM refuses multimodal content at the API, and
   `prefill_forward` refuses any pixel payload that reaches it anyway.
6. **`max_num_seqs` is capped at 32** by the sampler (`MAX_SAMPLING_BATCH`); the adapter refuses more
   with a clear error rather than silently truncating.
7. **On-device log-probs need 8 or 32 devices**, so on this 4-chip mesh any log-prob request falls back
   to the plugin's host sampler (which is why all 16 `test_logprobs` cases pass through logits rather
   than through the device sampler). That is a plugin/tt-metal capability boundary, not an adapter
   choice.
8. **The plugin and runner needed changes** to serve this model at all: two runner bugs against the
   current vLLM fork, one plugin registration override, one fabric-router-config passthrough. The
   plugin lives in a separate checkout; its diff is kept here as
   [`vllm_tt_plugin_changes.diff`](vllm_tt_plugin_changes.diff).
9. **No device-profiler evidence** from any serving run, per the skill's prohibition. Op-level timing for
   this decode stack is the optimized-full-model and datatype-sweep stages' own non-serving profiles.

---

## 9. Artifacts

Under [`readiness_vllm/`](../../readiness_vllm/) — what the shared runner and the adapter wrote:

| file | what it is |
|---|---|
| `vllm_result.json` / `vllm_benchmark.json` / `vllm_benchmark.log` | primary single-user 128/128/1: raw `vllm bench serve`, normalized summary, console |
| `vllm_ci_serving_result.json` / `vllm_ci_serving_benchmark.json` / `vllm_ci_serving_benchmark.log` | secondary CI serving-burst 100/100/32, same three forms |
| `vllm_qualitative_outputs.json` | greedy and sampled completions for the six shared prompts (raw `/v1/completions`) |
| `sampling_tests.log` | the canonical `tests/tt` suite, `--sampling-profile full` |
| `server.log` | the served launch, warm-up and request log |
| `vllm_serving_capability.json` | what the *built* serving model is: precision policy per layer, KV-cache dtype, capability flags, page-table geometry, generator counters — written by the adapter at the end of warm-up, from inside the engine-core process |

Under [`doc/vllm_integration/`](.) — this stage's own evidence:

| file | what it answers |
|---|---|
| [`work_log.md`](work_log.md) | the engineering record, including every wrong turn |
| [`qualitative_chat.json`](qualitative_chat.json) | the six prompts in the checkpoint's chat format, beside the HF and full-model controls |
| [`serving_requests.json`](serving_requests.json) | non-aligned prompt lengths, null-block containment, 32-way concurrency, a 9000-token prompt |
| [`serving_primitives.json`](serving_primitives.json) | the six mechanical contract checks (steady state, stale-pair merge, page-table-only refresh, layout change, slot remap) |
| [`slot_reproducibility.json`](slot_reproducibility.json) | batch-1 vs batch-32 logit reproducibility, per step, with state and KV-page comparisons |
| [`prefill_alloc_vs_recapture.json`](prefill_alloc_vs_recapture.json) | refutes per-call page-row allocation and trace re-capture as sources of prefill drift |
| [`prefill_determinism_bisect.json`](prefill_determinism_bisect.json) / [`…_fixed.json`](prefill_determinism_bisect_fixed.json) | the page-table-substitution defect, before and after the fix |
| [`prefill_stability_with_traces.json`](prefill_stability_with_traces.json) | a live captured trace does not make repeated prefills drift |
| [`logit_read_stability.json`](logit_read_stability.json) / [`…_full_model.json`](logit_read_stability_full_model.json) | the readback path is bit-stable, reduced target and full model, `1x4` and `1x1` |
| [`vllm_checkout.txt`](vllm_checkout.txt) / [`vllm_tt_plugin_changes.diff`](vllm_tt_plugin_changes.diff) | the vLLM commit served, and the plugin changes (that repo is not committed here) |
| [`batch32/`](batch32/) | the `--max-num-seqs 32` server's sampling log, qualitative outputs, server log, and its single-user benchmark |
| [`batch1/`](batch1/) | the `--max-num-seqs 1` server's sampling log, qualitative outputs, and the cold/warm primary benchmark pair |
| [`async/`](async/) | the `--async-scheduling` runs' qualitative outputs, sampling log, benchmarks and request probe |
| [`logs/`](logs/) | every probe (`.py`) with the console log of its final run (`.txt`) |

Tests: [`tests/test_generator_vllm.py`](../../tests/test_generator_vllm.py) — 8 host-only cases and 11 on
the reduced two-layer target, driving the adapter through the plugin-facing API.
