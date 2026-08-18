# Ornith-1.0-35B — vLLM serving integration (TTNN, 4-chip Blackhole ring)

The [datatype sweep's](../datatype_sweep/) selected model, served through the shared TT vLLM path on the
same hardware: four Blackhole `p300c` chips as a `1x4` ring under `FabricConfig.FABRIC_1D_RING`, TP=4
dense and EP=4 over the 256 routed experts, precision policy **`C06-proj-bfp4-lofi`**, paged KV cache
**bfloat8_b**, **262144-token** advertised and served context.

---

## 1. Headline: primary single-user serving performance

**Workload: 128-token prompt, 128 generated tokens, 1 request, `--max-concurrency 1`, greedy
(`--temperature 0.0`), `ignore_eos`, `--max-num-seqs 1`, decode trace on, on-device sampling
(`sample_on_device_mode: all`), async scheduling ON — which is this vLLM's default for a server that
passes no scheduling flag, and is what the server behind these numbers ran ([§5](#5-async-decode-overlap-on-by-default-here-and-worth-6-)).** Raw
[`readiness_vllm/vllm_result.json`](../../readiness_vllm/vllm_result.json), normalized
[`readiness_vllm/vllm_benchmark.json`](../../readiness_vllm/vllm_benchmark.json).

| metric | **warm** (the committed artifact) | first request at this prompt length, same server |
|---|---|---|
| **TTFT** P50 / P99 | **151.7 / 151.7 ms** | 228.3 ms |
| **TPOT** mean / P99 | **23.174 / 23.174 ms** | 24.894 ms |
| **ITL** P50 / P99 | **23.133 / 23.72 ms** | 23.134 / 25.47 ms |
| **decode t/s/u** (`1000 / mean_tpot_ms`) | **43.15 t/s/u** | 40.17 t/s/u |
| decode t/s/u from ITL P50 | 43.23 t/s/u | 43.23 t/s/u |
| aggregate output throughput | 41.36 tok/s | 37.76 tok/s |
| requests completed | **1/1, 128/128 tokens** | 1/1, 128/128 |

Both columns are the same command on the same server, back to back: the cold run
([`batch1/vllm_benchmark_repeat_cold_first_request.json`](batch1/vllm_benchmark_repeat_cold_first_request.json)),
then the warm run, which is the committed
[`readiness_vllm/vllm_benchmark.json`](../../readiness_vllm/vllm_benchmark.json) — byte-identical to
[`batch1/vllm_benchmark_repeat_warm.json`](batch1/vllm_benchmark_repeat_warm.json), and produced by the same
server as the committed `server.log.gz`, `sampling_tests.log.gz`, qualitative outputs and both capability
reports (§9 explains why that attribution is checked explicitly).

**The warm figure repeats; the cold one does not.** Three independent overlapped batch-1 servers, all warm:

| server | TTFT P50 | TPOT | ITL P50 | t/s/u |
|---|---|---|---|---|
| the committed one ([artifact](batch1/vllm_benchmark_repeat_warm.json)) | 151.7 ms | 23.174 ms | 23.133 ms | 43.15 |
| an earlier one ([artifact](batch1/vllm_benchmark_second_server_warm.json)) | 145.3 ms | 23.170 ms | 23.140 ms | 43.16 |
| another ([artifact](async/async_max_num_seqs_1_vllm_benchmark_warm.json)) | 148.3 ms | 23.146 ms | 23.139 ms | 43.20 |

TPOT spans 0.028 ms (0.12 %) and ITL P50 0.007 ms across the three; warm TTFT spans 6.4 ms. The *first*
request at a prompt length is much looser — 172.8, 228.3 and 236.3 ms on three servers — because it
compiles that length's prefill programs and pays a trace re-capture inside its TTFT, and how much of that
work is already done depends on what the process has served before.

**Where that cost actually lands, because the ITL median hides it.** ITL P50 is 23.13 ms in *both* columns,
which is why an earlier version of this section said the cost was entirely in the first token. It is not.
`vllm bench serve` excludes TTFT from TPOT (`TPOT = (e2el − ttft)/(n−1)`), so the cold column's own numbers
put **218 ms inside the inter-token intervals**: e2el 3389.9 ms − TTFT 228.3 ms = 3161.6 ms over 127
intervals, against 127 × 23.174 ms = 2943.1 ms warm. And it is one interval, not a uniform shift — ITL P50
is unchanged at 23.134 ms while ITL P99 is 25.47 ms, so at most one interval of the 127 is long, and it
carries the whole ~220 ms. The second server's cold run gives ~231 ms the same way.

The mechanism is in the committed server log: the cold request enters prefill at `18:55:38.314` and the
decode traces are re-captured at `18:55:38.494`. That re-capture is not part of the prefill — it comes from
`submit_serving_decode`'s replay-safety check, which runs on the **first decode step**, so it lands after the
first token has been returned and is charged to the first inter-token interval rather than to TTFT. The warm
request at `18:55:52.562` has no re-capture line. The datatype-sweep stage measured the same thing standalone and
called it by name: [`cold_prompt_length_cost`](../datatype_sweep/post_selection_token_out.json) records
`hidden_cost_ms` 250.4 with `trace_recaptures` 1 (312 ms cold TTFT against 177 ms warmed, on its own
harness where the re-capture fell inside the TTFT window instead).

So the first request at a new prompt length costs ~+77 ms of TTFT **and** one ~220 ms stall a few tokens in.
Every later request at that length pays neither. `OrnithGenerator.warmup(prompt_lengths)` removes both for a
deployment that knows its lengths; the serving warm-up compiles one length (64) rather than guessing a
bucket set.

### Against the model's own decode floor

| | ms/token | t/s/u | source |
|---|---|---|---|
| full-model token-out benchmark, same 128/128/1 shape, batch 1, 9 warm repeats | 23.165 | 43.169 | [`post_selection_token_out.json`](../datatype_sweep/post_selection_token_out.json) |
| **vLLM serving, this stage** | **23.174** (TPOT mean), 23.133 (ITL P50) | **43.15** (43.23 from ITL) | this README |
| traced decode without sampling or readback (lower bound) | 21.965 | 45.526 | same artifact |
| teacher-forcing traced decode (serial by construction) | 23.643 | 42.296 | [datatype sweep](../datatype_sweep/README.md) |

Serving decode is **at** the model's own token-out figure — 0.009 ms/token above it on TPOT and 0.032 below
it on the ITL median, both well inside the 9-repeat spread of the baseline — so there is no measurable
vLLM-specific decode overhead left to remove. That is true of the *overlapped* configuration, which is
this vLLM's default and the one measured here; with `--no-async-scheduling` the same server sits at
24.591 ms ITL, 1.4 ms above the floor ([§5](#5-async-decode-overlap-on-by-default-here-and-worth-6-)).
TTFT carries +6 to +12 ms of serving path across the three warm runs (145.3–151.7 ms against the model's
139.5 ms median warmed TTFT): HTTP, tokenizer, scheduler and detokenization. Teacher forcing is quoted
only as the lower-bound comparison the skill asks for, and serving is *faster* than it, as expected —
teacher forcing decides step *N+1*'s input on the host.

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
| degenerate-output check | `no degenerate output detected` (`--scope vllm` and `--scope all`), on the default (overlapped) runs at both batch sizes and on the `--no-async-scheduling` control |
| sampling suite | 54/73 at `max_num_seqs=32`, 65/73 at `max_num_seqs=1`; every failure in both is a reproducibility or batch-size-structural assertion, none a correctness one (§6). The batch-1 result was seen three times on three servers, but only the last run's log is committed — `sampling_tests.log` is truncated per launch ([work log §7.5](work_log.md#75-sampling-suite)) |
| single-user determinism | at `max_num_seqs=1`, repeated greedy requests are **identical** (3/3, and again after ~90 intervening requests), and the same greedy text comes back with overlap turned off. Seeded requests repeat within a scheduling mode but differ between the two (§5, limitation 10). At `max_num_seqs=32` greedy repeats are not identical — §6 |
| what the built model is | [`readiness_vllm/vllm_serving_capability.json`](../../readiness_vllm/vllm_serving_capability.json), written by the adapter at the end of warm-up: policy `C06-proj-bfp4-lofi`, KV cache `BFLOAT8_B`, LM head `BFLOAT4_B`, 40 layers, 4097 blocks, no layer exceptions, `owns_cache=False`. [`…_final.json`](../../readiness_vllm/vllm_serving_capability_final.json) is the same report at engine-core exit, with the counters of the traffic that server served (10831 decode steps, 9431 of them copying nothing to the device) |

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
25 – 236 characters of each answer (25/34/236/214/213/130 by prompt) and then takes a different phrasing
of the same plan ("Analyze the Request" against "Deconstruct the request") — the near-tie behaviour §6
measures, not a different answer.

**Where the completions stop.** The chat run's six all end at the 128-token cap
(`finish_reason: length`, 128 completion tokens, in the artifact). The raw run is committed without
`finish_reason` — the shared runner stores only the two texts — and four of its six sampled completions run
to the 256-token cap while **two self-terminate**: the haiku (13 words) and the French translation (8
words) skip the reasoning block entirely (`<think>\n\n</think>`) and answer, which is the checkpoint
deciding it does not need to think aloud. Every greedy raw completion runs to the cap. Where a completion
caps *inside* a `<think>` block, that is this checkpoint's format rather than truncation by the serving
path: the HF control and the full-model TTNN control, generated at the same caps, stop in the same kind of
place.

**Verdict: pass.** Prompt by prompt, read from the committed artifacts:

* *"Write a haiku about machine learning."* — the greedy raw completion opens a `<think>` block, restates
  the 5-7-5 constraint, brainstorms keywords ("Data, code, neural, nets, learn, predict, train, model, AI,
  brain, silicon, patterns, future, hidden, deep"), drafts candidate lines with syllable counts ("Data
  flows through wires (5) -> Da-ta flows through wi-res (5). Good."), critiques its own attempt and starts
  another ("Silicon minds wake (5)") — and hits the 256-token cap mid-plan, as does the HF control. The
  **sampled** completion takes the other route: empty `<think></think>`, then a finished haiku — *"Silicon
  dreams deep, / Data flows through hidden layers, / Patterns come alive."* Coherent, on topic, English.
* *"Explain the difference between supervised and unsupervised learning…"* — both completions plan the
  answer: labelled vs unlabelled data, teacher-guided vs finding patterns alone, prediction vs discovery,
  with analogies (greedy: "learning to identify fruits with a teacher"; sampled: "a child learning animals
  with a parent pointing and naming each one", "exploring a new city without a map"). Matches the HF
  control's content and structure closely.
* *"Complete this story…"* — the greedy completion continues the sentence ("a strange device that could
  manipulate the fabric of reality"), then plans the story and names the device ("the Aether Loom"). The
  **sampled** completion writes the story instead, after an empty `<think></think>`: a pocket-watch-sized
  device "crafted from a metal that seemed to drink in the light", an inventor called Elian, found beneath
  the roots of the Ancient Oak in the Royal Gardens, and it reaches the 256-token cap mid-scene, a few words
  after time snaps "back to normal with a deafening *crack*". Fluent prose, consistent within itself. The greedy one also contains "Here's a
  thinking thinking sequence" — a doubled word that is a **checkpoint** behaviour, not a serving artefact:
  the HF reference and the full-model TTNN run produce it too, and the chat run matches the HF control for
  236 characters *including* that phrase.
* *"What are the three laws of thermodynamics?"* — answered directly and correctly in both (conservation of
  energy; entropy of the universe increasing; entropy of a perfect crystal zero at absolute zero). Being a
  raw completion rather than a chat turn, both then pose and answer their own follow-ups ("What does the
  first law of thermodynamics state?") — continuation behaviour, expected of a chat model prompted without
  its template.
* *"Translate … to French"* — French, correct, and in the sampled case complete and nothing else:
  *"Bonjour, comment allez-vous aujourd'hui ?"*. The greedy completion plans the register (formal
  `allez-vous` against informal `ça va`) and caps inside that. No wrong-language drift in any of the twelve.
* *"Write a Python function … Fibonacci"* — the greedy completion emits working iterative Python
  (`sequence = [0, 1]` then a loop) and then **loops**, repeating
  `"What is the output of the following Python code? … def greet(name) … The output is: Hello, World!"`
  four times with the same 12-gram. The sampled completion instead plans the implementation options
  (naive recursion, memoised, iterative) and caps inside that plan.

  That loop is the one piece of mechanical repetition in the twelve raw completions. It is *not* produced by
  the serving path: the `--no-async-scheduling` server returns the **byte-identical** greedy completion,
  loop included, and the *chat* run of the same prompt — the verdict path — has no repetition at all and
  answers the one question it was asked. (The `max_num_seqs=32` server's greedy completion also
  self-quizzes, but with three *different* blocks rather than a repeat, so it is a weaker control and is
  not counted as one.) Mechanically it stays inside the gate: adjacent-duplication 0.000 and trigram-loop
  0.101 against thresholds of 0.10 and 0.50. Worth knowing before serving raw completions from this
  checkpoint without its chat template; not a defect in the serving path.

No prompt echo, no control-token leakage, no cross-request contamination (each completion answers its own
prompt). One completion does loop — the greedy Fibonacci one above — and it is reproduced byte-for-byte on
the non-overlapped server, so it belongs to the checkpoint's raw-continuation behaviour rather than to
serving. `check_degenerate_output.py` agrees mechanically — over the ten raw completions long enough for it to
measure (its `MIN_WORDS_FOR_DUPLICATION` is 20 words, and the two self-terminating sampled completions are
13 and 8): adjacent-duplication **0.0000 – 0.0140** against a 0.10 threshold and trigram-loop
**0.0149 – 0.1304** against 0.50 — so **no degenerate output detected**, on `--scope vllm` and on
`--scope all`. Console log: [`logs/check_degenerate_output.txt`](logs/check_degenerate_output.txt).

Greedy and sampled completions are both reasonable, and the sampled ones differ from the greedy ones
without degrading — the haiku and the story are the clearest examples: greedy plans, sampled delivers.

---

## 5. Async decode (overlap): on by default here, and worth 6 %

**Read this first if you are comparing configurations.** This vLLM enables async scheduling unless
something incompatible is configured (`vllm/config/vllm.py`), so the headline numbers in §1 — and every
other served number in this stage that does not say otherwise — were measured with **overlap on**. Passing
`--async-scheduling` changes nothing here; the flag that changes something is `--no-async-scheduling`.
An earlier version of this README had that backwards, and its "overlap buys nothing" conclusion came from
comparing two overlapped servers ([work log §7.6](work_log.md#76-async-scheduling-decode-overlap-and-the-default-this-stage-first-got-wrong)).

`supports_async_decode=True` is claimed because the split is implemented and measured:
`decode_forward(..., read_from_device=False)` returns device handles, `read_decode_output(...,
async_read=True)` enqueues `cpu(blocking=False)` plus `ttnn.record_event` behind the replays, and
`process_decode_output_host(...)` only formats. The contract that makes overlap *safe* is the refresh
policy: in the steady state nothing is copied to the device, and when the layout does change the staged
token/position pair is merged per row so a continuing row keeps the device's authoritative pair.

**What overlap costs and buys**, primary single-user 128/128/1, warm, `max_num_seqs=1`:

| warm, 128/128/1 | overlap **on** (default): 3 servers | overlap **off** (`--no-async-scheduling`): 2 servers, 3 warm runs |
|---|---|---|
| TPOT | **23.146 – 23.174 ms** | 24.711 – 24.821 ms |
| ITL P50 | **23.133 – 23.140 ms** | 24.591 – 24.595 ms |
| decode t/s/u (TPOT-derived) | **43.15 – 43.20** | 40.29 – 40.47 |
| TTFT P50 | 145.3 – 151.7 ms | 145.7 – 157.9 ms |
| `async_reads` counter over the process | **10831** (= its decode-call count) | **0** |

The ranges do not overlap on any decode metric and do not separate at all on TTFT: overlap is worth
**~1.45 ms per token, 6.3 %**, in the decode loop and nothing measurable in time-to-first-token. The
`async_reads` counter says which path each server actually took — exactly one async read per decode call
with overlap on, none at all with it off.

It also settles the decode-floor question in §1: the model's own traced token-out floor is 23.165 ms, the
overlapped servers sit at 23.133 – 23.140 ms — *at* the floor — and the non-overlapped ones at 24.59 ms.
The 1.45 ms the async split hides is real host-side time, not a modelling artefact.

**Equivalence at equal request length** ([`logs/probe_overlap_equivalence.py`](logs/probe_overlap_equivalence.py),
both arms 32 tokens):

| evidence | result |
|---|---|
| **greedy text, overlap on vs off** | **character-for-character identical** ([`async/overlap_texts_no_async.json`](async/overlap_texts_no_async.json)) |
| greedy repeats within each mode, fresh and after ~90 requests | identical, 4 arms |
| one-token-stale host pair vs the correct pair, at the primitive level | **token-for-token identical**; with the merge removed the stream diverges and repeats a token ([`serving_primitives.json`](serving_primitives.json)) |
| steady-state device copies | 0 tokens, 0 positions, 0 page tables over 5 consecutive steps; **9431 of 10831** served decode steps copied nothing ([`readiness_vllm/vllm_serving_capability_final.json`](../../readiness_vllm/vllm_serving_capability_final.json)) |
| smoke sampling profile, overlap off | same result as with overlap on (3 passed, 1 skipped) |
| request shapes and null-block containment, overlap off | unchanged ([`async/serving_requests_no_async_max_num_seqs_1.json`](async/serving_requests_no_async_max_num_seqs_1.json)) |
| degenerate-output check, overlap off | no degenerate output detected |
| **seeded** (temp 0.8, top-p 0.9, seed 4242) text, overlap on vs off | **differs** — reproducible within each mode, across traffic; see §8 limitation 10 and [work log §7.6.1](work_log.md#761-a-seeded-request-is-reproducible-within-a-server-not-across-two-of-them) |

The greedy row is the serving-level stale-input proof: with overlap on, vLLM submits step *N+1* before
token *N* has been applied to its host state, so a row that read the host's lagging pair would double a
subword. It does not, because the refresh policy merges per row.

The CI burst was measured twice, both times overlapped (158.5 and 158.7 tok/s aggregate) — a repeatability
check, not an overlap comparison; no non-overlapped burst was run. Artifacts for both scheduling modes are
in [`async/`](async/).

---

## 6. Sampling suite: what passed, what did not, and why

`--sampling-profile full` — the whole canonical `tests/tt` suite — was run on two server configurations.

| configuration | passed | failed | skipped |
|---|---|---|---|
| `--max-num-seqs 32` | 54 | **18** | 1 |
| `--max-num-seqs 1` | 65 | **7** | 1 |

Nothing in either failure set is a correctness, log-prob, crash or output-quality failure. **50 of the 73
cases pass in both configurations**, and they are the ones that matter for correctness — read straight out
of the two committed logs: all **20** `test_logprobs` parameterisations (host sampling; device log-probs
need 8 or 32 devices) and all 8 `test_build_logprobs_from_topk` cases, all five host-only parameters
(`min_p`, `bad_words`, `logit_bias`, `allowed_token_ids`, `min_tokens`), structured output at full capacity,
the three plugin-config cases, and 13 of the 28 `test_seeding_and_variety` cases: seed *variety*
(`test_batch1_no_seed_varied`, `test_different_seeds_produce_different_outputs`), a negative seed not
crashing, temperature variation within and between batches (all six), `test_topk[19]`,
`test_specific_seed_reproducible[42]` and `test_uniform_seed_deterministic[1-0/1-1]`. The single skip is
`test_chat_logprobs_all_vocab`, which skips itself because the plugin clamps `max_logprobs` to 20.

Which *parameterisations* of the reproducibility cases pass is itself the near-tie property measured below,
so it moves between runs: `test_topk[19]` and `test_specific_seed_reproducible[42]` pass at both batch sizes
while `test_topk[15]`, `[32]` and the other three seeds fail at 32, and an earlier run had that the other way
round.

**The 18 failures at `max_num_seqs=32` are all one class**: "the same request must produce the same
text". Exactly, from [the log](batch32/sampling_tests_max_num_seqs_32.log.gz): `test_mixed_params_batch`,
`test_seeding`, `test_same_seeds_reproduce_across_batches`, `test_specific_seed_reproducible[0/123/999]`,
`test_batch1_seed_reproducible[0/1]`, `test_uniform_seed_deterministic[10-0/10-1/32-0/32-1]`,
`test_top1_is_greedy`, `test_topk[15]`, `test_topk[32]`, and the three `test_*_penalty_mixed_batch`.
`test_topk[19]` and `test_specific_seed_reproducible[42]` passed — which batch shape or seed lands on a
near-tie is itself the property measured below, so the exact membership moves between runs while the class
does not.

**The 7 failures at `max_num_seqs=1` are structural**, from
[its log](batch1/sampling_tests_max_num_seqs_1.log.gz): the three `test_different_*_penalties`, the three
`test_*_penalty_mixed_batch`, and `test_uniform_noseed_varied`. Each builds its request list by slicing it
with `max_batch_size` and then asserts on ≥2 distinct outputs, ≥5 distinct outputs, or a comparison across
rows. At a capacity of one, the `test_different_*_penalties` slice leaves one request and the
`test_*_penalty_mixed_batch` slice leaves **none** — two of them report `Got 0 unique results out of 0.`
and the third, `test_frequency_penalty_mixed_batch`, raises `IndexError: list index out of range` indexing
that empty list. A harness assumption, not a model answer.

So **15 of the 18** batch-32 failures pass at `max_num_seqs=1`, and the three `test_*_penalty_mixed_batch`
fail in both configurations: for the reproducibility reason at 32 and for the empty-slice reason at 1.

### Why the reproducibility class fails, measured at the logits

`$vllm-integration` asks for logit-level evidence before classifying determinism failures. The full
investigation — including the first measurement that turned out to be measuring a page table of zeros — is
[work log §8](work_log.md#8-the-reproducibility-class-of-failures-measured) and
[§9](work_log.md#9-autofix-the-first-determinism-measurement-was-measuring-a-page-table-of-zeros). The
corrected result, from [`slot_reproducibility.json`](slot_reproducibility.json) on the full 40-layer model,
13 logit vectors per arm (the prefill's plus 12 traced decode steps):

| arm, two identical greedy runs from a wiped state | bit-identical | max abs Δ per step while the streams still agree | min PCC there | top-1 agreement | first flip (margin there) |
|---|---|---|---|---|---|
| **batch 1** | **yes, all 13** | **0.0** | **1.0** | 13/13 | none |
| **batch 2** | **yes, all 13** | **0.0** | **1.0** | 13/13 | none |
| **batch 4** | **yes, all 13** | **0.0** | **1.0** | 13/13 | none |
| batch 8 | no | 0.78 | 0.9976 | 9/13 | step 9 (0.0625) |
| batch 16 | no | 0.84 | 0.9976 | 10/13 | step 10 (0.0625) |
| batch 32, same slot, runs 1 vs 2 | no | 0.72 | 0.9979 | 10/13 | step 10 (0.0625) |
| batch 32, same slot, runs 2 vs 3 | no | 0.81 | 0.9978 | 12/13 | step 12 (0.125) |
| batch 32, every one of the 32 rows occupied | no | 0.84 | 0.9983 | 10/13 | step 10 (**0.0**) |
| batch 32, the same step's device logits read to host twice | **yes** | **0.0** | **1.0** | — | none |

The deviation column stops before the flip on purpose: once the two streams pick different tokens they are
different text, and the per-step numbers jump to 8–16 for that reason alone.

* **Bit-reproducible to batch 4, not at batch 8 and above.** Batch 1, 2 and 4 are bit-identical across
  runs — prefill and every traced decode step, PCC 1.0. The property is lost between batch 4 and batch 8
  and stays lost at 16 and 32. That is why the whole reproducibility class of the shared suite passes at
  `--max-num-seqs 1` and why `--max-num-seqs 32` fails part of it.
* **It is the collectives, and nothing else on the path.** Ruled out by measurement rather than argument
  ([`decode_nondeterminism.json`](decode_nondeterminism.json), [`reduced_target/`](reduced_target/),
  [work log §8.3](work_log.md#83-where-it-enters-measured-the-multi-device-collectives)): the readback path
  (the same device result composed to host twice is bit-identical), the prefill and the state it writes
  (all 140 DeltaNet buffers plus the written KV pages, identical and finite at both batch sizes), residue
  in unoccupied rows (the all-rows arm deviates just the same), and trace replay (the *eager* path deviates
  too, on the full model as well as the reduced one). The control that names it: the same driver, same
  batch 32, on a **1x1** mesh where no collective runs — **0 of 5** run-pairs deviate, against **3 of 5**
  on `1x4`, and 0/5 against 4/5 for batch 4 against batch 8 with mesh and driver held fixed.
* **A near-tie decides every flip.** While the streams agree the deviation is 0.70–0.84 of a logit, and
  every first flip in the table happens where the top-1/top-2 margin is **0.0–0.125** — an exact tie (one
  bfloat16 value for two tokens) or one to two bfloat16 steps at that magnitude. Where the margin is
  comfortable the deviation changes nothing: the slot-7 arm deviates by up to 1.25 across all 13 vectors
  without changing a single token.
* **The datatype sweep's §9.1 is a control that holds, not corroboration.** It measured a different axis:
  four *slots inside one run* at batch 4, and it concluded the effect there was "fully deterministic — the
  same slot flips on all three repeats … a fixed reduction order, not nondeterminism"
  ([§9.1](../datatype_sweep/README.md#91-one-of-four-identical-decode-slots-emits-a-different-token)). This
  stage's batch-4 arm agrees with it: run-to-run bit-identical at batch 4. Cross-slot difference at a fixed
  reduction order and run-to-run nondeterminism are two different properties, and only the second one is
  what fails above batch 4.

Making batch ≥ 8 decode bit-reproducible means changing the collectives or the decode geometry — decoder
work with its own accuracy and performance gates, not serving work, and nothing at the serving layer can
substitute for it. It is recorded as an open, named decoder-stage defect in
[work log §8.5](work_log.md#85-what-is-and-is-not-fixable-here-and-the-defect-this-stage-hands-on). Output
quality is unaffected: every arm stays fluent, on topic and in English, the difference is a different
near-tie winner rather than corruption, and the degenerate-output gate passes on both configurations.

---

## 7. How this was run

Environment: `tenstorrent/vllm@bf98d556` (`dev`) installed into the tt-metal venv with
`VLLM_TARGET_DEVICE=empty`, plus `plugins/vllm-tt-plugin` ([work log §2](work_log.md#2-vllm-was-not-installed-and-two-shared-paths-did-not-work-against-the-current-fork)).
A `tt-smi -r` + mesh smoke ran before every launch ([§10.1](work_log.md#101-every-vllm-shutdown-left-the-fabric-unable-to-re-open-until-the-boards-were-reset)).

**One environment change to be aware of when reproducing these numbers**: installing vLLM moved `numpy`
from 1.26.4 to **2.3.5**, which is outside tt-metal's own `numpy>=1.24.4,<2` pin
(`tt_metal/python_env/requirements-dev.txt`, `pyproject.toml`). Every device run in this stage — probes,
servers, benchmarks and both test suites — ran under numpy 2.3.5, and the model's own suite passes there
([work log §2](work_log.md#2-vllm-was-not-installed-and-two-shared-paths-did-not-work-against-the-current-fork));
the pin is nevertheless still 1.x, so a reader reproducing this on a clean env pinned to numpy 1.x is not
running quite the same environment.

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

`--max-num-seqs 32` is the same command with that one value changed. Overlap is already on in the command
above (this vLLM's default, [§5](#5-async-decode-overlap-on-by-default-here-and-worth-6-)); the
non-overlapped control is `--additional-server-args="--no-async-scheduling"`. The `=` matters either way —
without it argparse eats a value that starts with `-`.

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
python models/autoports/ornith_ai_ornith_1_0_35b/doc/vllm_integration/logs/probe_serving_requests.py \
  --url http://localhost:8100 --server-label "max_num_seqs=1, default flags (async scheduling on)"
python models/autoports/ornith_ai_ornith_1_0_35b/doc/vllm_integration/logs/probe_overlap_equivalence.py \
  --url http://localhost:8100 --tag sync     # and again with --tag async against the overlap server
python models/common/readiness_check/check_degenerate_output.py \
  --hf-model ornith-ai/Ornith-1.0-35B --missing-artifacts critical --scope vllm
python .agents/scripts/check_context_contract.py \
  --model-dir models/autoports/ornith_ai_ornith_1_0_35b --hf-model ornith-ai/Ornith-1.0-35B \
  --stage vllm --require-contract

# adapter suite (host-only cases need no device; the rest use the reduced two-layer target).
# It builds reduced adapters, and a reduced build deliberately does not write readiness_vllm/ (§9).
pytest models/autoports/ornith_ai_ornith_1_0_35b/tests/test_generator_vllm.py -q
```

Ten probes, all in [`logs/`](logs/). Each writes its JSON next to this README and its console log
beside itself.

*Against a running server* (three): `probe_qualitative_chat.py`, `probe_serving_requests.py`,
`probe_overlap_equivalence.py`.

*Standalone, driving the generator or the model directly* (seven): `probe_serving_primitives.py`,
`probe_slot_reproducibility.py`, `probe_decode_nondeterminism.py`, `probe_prefill_determinism_bisect.py`,
`probe_prefill_alloc_vs_recapture.py`, `probe_prefill_stability_with_traces.py`,
`probe_logit_read_stability.py`.

An eleventh, `probe_logit_determinism.py`, was **withdrawn**: it passed a page table without the cache it
indexed, so the generator substituted its own all-zeros table and every logical block aliased onto
physical block 0. Its "nondeterminism" was that aliasing. Both the probe and its JSON are deleted rather
than kept with a caveat, because the measurement never measured what it claimed
([work log §9](work_log.md#9-autofix-the-first-determinism-measurement-was-measuring-a-page-table-of-zeros));
`probe_slot_reproducibility.py` replaces it.

The reduced serving target for bring-up: `ORNITH_VLLM_LAYER_INDICES=0,3` (one real layer of each kind).
Nothing in this README is measured on it.

---

## 8. Limitations

1. **Reproducibility above `max_num_seqs 4`.** Repeated identical greedy requests are not guaranteed to
   return identical text above a padded decode batch of 4; they agree until the first near-tie. The
   boundary is measured (bit-identical at batch 1, 2 and 4; not at 8, 16, 32) and so is where it enters
   (the multi-device collectives — a 1x1 control is bit-identical at batch 32): §6 and
   [work log §8](work_log.md#8-the-reproducibility-class-of-failures-measured). It is an open
   decoder-stage defect, named with a reproducer in
   [work log §8.5](work_log.md#85-what-is-and-is-not-fixable-here-and-the-defect-this-stage-hands-on);
   nothing at the serving layer can fix it. A deployment that needs bit-reproducible completions should
   serve at `--max-num-seqs 1`, where the whole reproducibility class of the shared suite passes.
2. **The first request at a new prompt length is slower, in two places.** It compiles that length's prefill
   programs — **228.3 ms TTFT against 151.7 ms** warmed for the 128-token shape on the committed server, and
   172.8 and 236.3 ms on two others — *and* it pays the decode-trace re-capture as a single ~220 ms stall a
   few tokens into the stream, which is why its TPOT is 24.894 ms against 23.174 warm while its ITL median
   is unchanged (§1 derives both from the artifacts). Every later request at that length pays neither.
   `OrnithGenerator.warmup(prompt_lengths)` removes both for a deployment that knows its lengths; the
   serving warm-up compiles one length (64) rather than guessing a bucket set.
3. **Single-user latency and 32-user capacity are two server configurations.** A `--max-num-seqs 32`
   server pays the padded decode batch on every step, so single-user TPOT there is 140.069 ms (7.14 t/s/u)
   against 23.174 ms (43.15 t/s/u) at `--max-num-seqs 1`. Both are measured and both artifacts are kept.
4. **Prefix caching is off** and not claimed (`supports_prefix_caching=False`).
5. **Text only.** The checkpoint carries a vision tower this port does not implement. The adapter is
   registered as a text-only model, so vLLM refuses multimodal content at the API, and
   `prefill_forward` refuses any pixel payload that reaches it anyway.
6. **`max_num_seqs` is capped at 32** by the sampler (`MAX_SAMPLING_BATCH`); the adapter refuses more
   with a clear error rather than silently truncating.
7. **On-device log-probs need 8 or 32 devices**, so on this 4-chip mesh any log-prob request falls back
   to the plugin's host sampler (which is why all 20 `test_logprobs` cases pass through logits rather
   than through the device sampler). That is a plugin/tt-metal capability boundary, not an adapter
   choice.
8. **The plugin and runner needed changes** to serve this model at all: two runner bugs against the
   current vLLM fork, one plugin registration override, one fabric-router-config passthrough. The
   plugin lives in a separate checkout; its diff is kept here as
   [`vllm_tt_plugin_changes.diff`](vllm_tt_plugin_changes.diff).
9. **No device-profiler evidence** from any serving run, per the skill's prohibition. Op-level timing for
   this decode stack is the optimized-full-model and datatype-sweep stages' own non-serving profiles.
10. **A seed reproduces a completion within one scheduling mode, not across both.** The same seeded request
   (temperature 0.8, top-p 0.9, seed 4242) returns one text on an overlapped server and a different one
   with `--no-async-scheduling` — each reproducibly, including after ~90 intervening requests, and greedy
   output is identical across the two. It is the sampler's draw sequence landing differently under the two
   schedulers, characterised in
   [work log §7.6.1](work_log.md#761-a-seeded-request-is-reproducible-within-a-server-not-across-two-of-them);
   pinning it to a line is plugin work, not adapter work.
11. **Async scheduling is on unless you turn it off.** It is worth ~6 % of decode here ([§5](#5-async-decode-overlap-on-by-default-here-and-worth-6-)),
   so that default is the right one for this model — but it is a default, not a choice this stage made, and
   the flag that changes it is `--no-async-scheduling`.

---

## 9. Artifacts

Under [`readiness_vllm/`](../../readiness_vllm/) — what the shared runner and the adapter wrote. **This
directory is last-writer-wins**: the runner truncates `server.log`, `sampling_tests.log` and the benchmark
files on every launch, so what is committed there is one server's output. Which server is not left to trust:
every one of `vllm_benchmark.json`, `vllm_result.json`, `vllm_qualitative_outputs.json`,
`vllm_serving_capability.json` and `…_final.json` is **byte-identical** to its `batch1/` copy on this tree,
and `server.log.gz` is that same server's log (`Asynchronous scheduling is enabled`, 518 completion + 13 chat
requests, 28 trace re-captures). Nothing after that server writes into the directory either: the adapter
suite builds *reduced* two-layer adapters, and the capability writer refuses to write for a reduced build
(a regression test asserts the committed bytes survive it). This was wrong twice before the check existed — see
[work log §7.7](work_log.md#77-benchmarks) and [§12](work_log.md#12-runtime-fallback-audit). Every console
log is committed **gzipped**: the repo's `.gitignore` excludes `*.log`, so an uncompressed `server.log`
silently would not be in the commit at all; the `.gz` beside it is the committed copy — the headline single-user
configuration (`--max-num-seqs 1`, default flags, i.e. overlap on), which ran the full sampling suite,
qualitative, both request probes and the benchmark pair. The exceptions are the three `vllm_ci_serving_*`
files, which only a `--max-num-seqs 32` server produces — and those now have `batch32/` copies too
(`vllm_ci_serving_benchmark_max_num_seqs_32.json`, `…_result_…json`, `…_benchmark_…log.gz`, byte-identical to
the `readiness_vllm/` originals), so every committed serving number is attributable to a configuration
directory rather than to the last writer. Per-configuration copies of everything live in
[`batch1/`](batch1/), [`batch32/`](batch32/) and [`async/`](async/), and those are the authoritative sets
when a number needs attribution to a server.

| file | what it is |
|---|---|
| `vllm_result.json` / `vllm_benchmark.json` / `vllm_benchmark.log.gz` | primary single-user 128/128/1: raw `vllm bench serve`, normalized summary, console |
| `vllm_ci_serving_result.json` / `vllm_ci_serving_benchmark.json` / `vllm_ci_serving_benchmark.log.gz` | secondary CI serving-burst 100/100/32, same three forms |
| `vllm_qualitative_outputs.json` | greedy and sampled completions for the six shared prompts (raw `/v1/completions`) |
| `sampling_tests.log.gz` | the canonical `tests/tt` suite, `--sampling-profile full` |
| `server.log.gz` | the served launch, warm-up and request log |
| `vllm_serving_capability.json` | what the *built* serving model is: precision policy per layer, KV-cache dtype, capability flags, page-table geometry, generator counters — written by the adapter at the end of warm-up, from inside the engine-core process |
| `vllm_serving_capability_final.json` | the same report re-written at engine-core exit, so its counters (refreshes, no-refresh steps, slot remaps, async reads, device- against host-sampled decodes) describe the traffic the server actually served rather than the warm-up |

Under [`doc/vllm_integration/`](.) — this stage's own evidence:

| file | what it answers |
|---|---|
| [`work_log.md`](work_log.md) | the engineering record, including every wrong turn |
| [`qualitative_chat.json`](qualitative_chat.json) | the six prompts in the checkpoint's chat format, beside the HF and full-model controls |
| [`serving_requests.json`](serving_requests.json) | non-aligned prompt lengths, null-block containment, 32-way concurrency, a 9000-token prompt |
| [`serving_primitives.json`](serving_primitives.json) | the six mechanical contract checks (steady state, stale-pair merge, page-table-only refresh, layout change, slot remap) |
| [`slot_reproducibility.json`](slot_reproducibility.json) | run-to-run logit reproducibility at batch 1, 2, 4, 8, 16 and 32, per step, with state and KV-page comparisons |
| [`decode_nondeterminism.json`](decode_nondeterminism.json) + [`reduced_target/`](reduced_target/) | where the batch ≥ 8 deviation enters: traced against eager, generator against model driver, `1x4` against `1x1`, batch 4 against batch 8, counted over five run-pairs per arm |
| [`prefill_alloc_vs_recapture.json`](prefill_alloc_vs_recapture.json) | refutes per-call page-row allocation and trace re-capture as sources of prefill drift |
| [`prefill_determinism_bisect.json`](prefill_determinism_bisect.json) / [`…_fixed.json`](prefill_determinism_bisect_fixed.json) | the page-table-substitution defect, before and after the fix |
| [`prefill_stability_with_traces.json`](prefill_stability_with_traces.json) | a live captured trace does not make repeated prefills drift — and, since the review's round-2 finding, each comparison records the compared row's own min/max/nonzero fraction, because the first version of this probe was comparing tile padding ([work log §9.1](work_log.md#91-the-same-class-again-two-probes-were-comparing-the-tile-padding)) |
| [`logit_read_stability.json`](logit_read_stability.json) / [`…_full_model.json`](logit_read_stability_full_model.json) | the readback path is bit-stable over **non-degenerate** rows: reduced target on `1x4` and `1x1`, full model on `1x4` (the full model does not fit on one device, so its `1x1` fields are `null`) |
| [`vllm_checkout.txt`](vllm_checkout.txt) / [`vllm_tt_plugin_changes.diff`](vllm_tt_plugin_changes.diff) | the vLLM commit served, and the plugin changes (that repo is not committed here) |
| [`batch32/`](batch32/) | the `--max-num-seqs 32` server's sampling log, qualitative outputs, server log, capability report, its single-user benchmark, and the CI serving-burst set (§2's numbers) |
| [`batch1/`](batch1/) | the `--max-num-seqs 1` server's sampling log, qualitative outputs, and the cold/warm primary benchmark pair |
| [`async/`](async/) | the decode-overlap comparison, **both** arms: the default (overlapped) servers' artifacts (including `async_max_num_seqs_32_sampling_tests.log.gz`, the smoke profile on an overlapped batch-32 server — 2 failed of 3, both from the batch-32 reproducibility class of §6) and the `--no-async-scheduling` control's server log, benchmarks, qualitative outputs, sampling smoke log and request probe, plus the four `overlap_texts_*.json` arms behind §5 |
| [`reduced_target/`](reduced_target/) | the run-pair counts that localise the batch ≥ 8 nondeterminism to the collectives (mechanical, on the two-layer bring-up target — see [work log §8.3](work_log.md#83-where-it-enters-measured-the-multi-device-collectives)) |
| [`logs/`](logs/) | every probe (`.py`) with the console log of its final run (`.txt`, gzipped where it is large), both test-suite logs, the two gate console logs ([`check_degenerate_output.txt`](logs/check_degenerate_output.txt), [`check_context_contract.txt`](logs/check_context_contract.txt)), and the device reset/mesh-smoke record |

Tests: [`tests/test_generator_vllm.py`](../../tests/test_generator_vllm.py) — 9 host-only cases and 11 on
the reduced two-layer target, driving the adapter through the plugin-facing API. **20 passed** on the
committed tree; console log [`logs/pytest_generator_vllm.txt`](logs/pytest_generator_vllm.txt).
