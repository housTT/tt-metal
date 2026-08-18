# vLLM integration — work log

Stage: `$vllm-integration` + `$tt-device-usage` for `ornith-ai/Ornith-1.0-35B`, starting from the
completed [datatype sweep](../datatype_sweep/). Branch `agentic-research/hous/ornith-1.0-35B`, `1x4`
Blackhole `p300c` ring under `FABRIC_1D_RING`.

Every device job ran **one at a time**, in its own process, with the mesh opened and closed by that
process — `$tt-device-usage`'s first rule. Two device incidents and their recovery are in §10.

---

## 1. What the stage started from

| | |
|---|---|
| model / generator | [`tt/model.py`](../../tt/model.py), [`tt/generator.py`](../../tt/generator.py), full 40-layer stack, TP=4 dense + EP=4 over 256 routed experts |
| precision policy | the datatype sweep's selection, `C06-proj-bfp4-lofi`, loaded by `tt/precision_config.py` when `policy` is left unset — so a serving caller that names nothing gets it |
| decode path | one captured **model** trace (token in → vocab-sharded sampler-ready logits out, `ttnn.plus_one` advancing `current_pos`/RoPE on device) + one captured **sampling** trace writing `tt_out_tok` into the persistent decode token buffer |
| archived token-out decode | **23.165 ms/token = 43.169 t/s/u**, prompt 128 / generate 128, batch 1, 9 warm repeats ([`post_selection_token_out.json`](../datatype_sweep/post_selection_token_out.json)) |
| archived warmed TTFT | 139.5 ms median (133.9 min), same shape |
| context contract | [`doc/context_contract.json`](../context_contract.json): HF-advertised **262144**, supported 262144, no capability reduction |
| KV cache cost | 5440 B per token per device, all 10 `full_attention` layers, bfloat8_b ([`capacity/selected.json`](../datatype_sweep/capacity/selected.json)) |

The generator already had what this stage needs and nothing was bolted on for serving: split
sampling with device token feedback, `-1`-position inactive slots, per-slot prefill with a state
merge, an explicit `kv_cache=` constructor argument for caller-owned cache, and a re-capture guard
for the post-capture-compile hazard.

---

## 2. vLLM was not installed, and two shared paths did not work against the current fork

`import vllm` failed and no `vllm-tt-plugin` existed anywhere on the machine, so the stage began by
installing the fork the runner's own docstring names:

```bash
cd /home/ttuser/dev/ornith && git clone --branch dev --depth 1 https://github.com/tenstorrent/vllm.git
cd vllm && VLLM_TARGET_DEVICE=empty uv pip install -e . \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv pip install -e plugins/vllm-tt-plugin
```

`tenstorrent/vllm@bf98d556` (`dev`), installed as `vllm 0.1.dev1+gbf98d556b`. `VLLM_TARGET_DEVICE=empty`
takes `requirements/common.txt`, which pins no torch, so the tt-metal `torch 2.11.0+cpu` survived.

**One environment consequence worth recording**: the install moved `numpy` from 1.26.4 to **2.3.5**,
which is outside tt-metal's own `numpy>=1.24.4,<2` pin (`tt_metal/python_env/requirements-dev.txt`).
Reverting it would have conflicted with the `opencv-python-headless 5.x` vLLM pulls in, so instead the
combination was **tested**: `import ttnn`, a `1x4` mesh open/close smoke, and
`tests/test_precision_config.py` all pass, and every device run in this stage ran under numpy 2.3.5.

Then two shared paths had to be fixed before the runner could launch this model at all:

| what broke | why | fix |
|---|---|---|
| `api_server.py: error: unrecognized arguments: --plugin-config` | the runner passes the TT plugin config on `--plugin-config`; current vLLM populates `additional_config` from `--additional-config`, and the TT plugin reads `vllm_config.additional_config` (`vllm_tt_plugin/config.py`) | `run_vllm_server._tt_config_flag()` picks the flag from the installed `EngineArgs` fields instead of hard-coding it |
| `--mesh-device` rejected everything but `N150/N300/T3K/TG` | a `1x4` Blackhole ring is none of those; the TT plugin's own table has `P150x4`/`P300x2` for `(1, 4)` and also accepts a literal `"(rows, cols)"` | `run_vllm_server._mesh_device()` accepts a preset, any plugin-known name, or an explicit grid, and forwards it verbatim |

Both are runner bugs against the current fork, not model-specific workarounds: no TT model on a mesh
outside those four names could have used the shared runner.

---

## 3. Plugin registration, and the two upstream decisions that had to be taken away from upstream

`register_tt_models()` gained:

```python
_ornith_target = "models.autoports.ornith_ai_ornith_1_0_35b.tt.generator_vllm:TTQwen3_5MoeForConditionalGeneration"
_register_model_if_missing(ModelRegistry, "TTQwen3_5MoeForConditionalGeneration", _ornith_target)
ModelRegistry.register_model("Qwen3_5MoeForConditionalGeneration", _ornith_target)
```

The `TT`-prefixed line is the plugin's convention and is what `check_and_update_config` validates. The
second line — **replacing** upstream's class for the plain architecture — was not the first attempt;
it is what two startup failures forced, and each is worth recording because the same trap waits for
any TT port of a hybrid multimodal checkpoint:

1. **`AssertionError` in `MultiModalRegistry._get_model_cls`.** `ModelConfig.__post_init__` resolves the
   architecture *before* `check_and_update_config` prepends `TT`, finds upstream's multimodal
   `Qwen3_5MoeForConditionalGeneration`, and populates `multimodal_config`. `InputProcessor` then asks the
   multimodal registry for a processor and asserts `_processor_factory` on the *resolved* (TT) class. This
   is the trap the plugin's Gemma4 block documents; the difference here is that the architecture name is
   already taken, so it had to be replaced rather than filled in.
2. **`ValueError: vLLM block_size=1072 but this model's paged cache uses 64-token blocks`.** Upstream's
   class is `IsHybrid`, so `verify_and_update_config` raises `cache_config.block_size` until an attention
   page can hold a whole GDN state — 1072 tokens for this config. That is a GPU-side constraint about
   sharing one tensor pool. This port keeps its recurrent state inside the model, so its paged attention
   blocks are its own 64-token blocks, and a rewritten block size simply does not describe the cache it
   allocates. The adapter refuses it loudly (that message is the adapter's) rather than allocating a cache
   vLLM will index wrongly.

An intermediate attempt — keeping upstream's class and registering upstream's Qwen3.5 multimodal
processor on the adapter — got past (1) and died on (2), and would also have let the API accept image
requests a text-only port cannot serve. Replacing the architecture fixes both: `is_multimodal_model`
and `is_hybrid` are then computed from the TT class, which is text-only and not hybrid. The cost is
four members on the adapter (`vllm_config` on `__init__`, `embed_input_ids`, `forward`,
`compute_logits`) that exist only because `interfaces_base.is_text_generation_model` is a structural
check and `ModelConfig` validates `--runner generate` against it. They raise; nothing on TT calls them.

One more plugin change, in `vllm_tt_plugin/worker.py`: `set_fabric` now honours a
`fabric_router_max_packet_bytes` key from the TT config and passes it to
`ttnn.set_fabric_config(..., router_config=...)`. Fabric configuration happens before
`ttnn.open_mesh_device`, so a model cannot set it later; this stack's collectives were tuned at 8192 B
(`multichip_decoder.DEFAULT_FABRIC_PACKET_BYTES`, worth up to 18 % on the bf16 collective rows) and the
build default is 4352 B. Without the passthrough every served token would pay a slower CCL than the
model's own benchmarks.

---

## 4. The adapter, and the seven primitives it drives

[`tt/generator_vllm.py`](../../tt/generator_vllm.py) is interface translation only. Everything that
touches the device is a generator method, and the serving decode step is the *same* split-sampling path
the datatype sweep measured.

| plugin entry point | what the adapter does |
|---|---|
| `initialize_vllm_model` | resolves the snapshot, builds `OrnithModel` with `max_context` = the advertised 262144, refuses `tt_data_parallel != 1` and `optimizations=...` (this port's precision is the sweep's artifact, selected by `ORNITH_PRECISION_POLICY`, not by a performance/accuracy switch it would silently ignore) |
| `get_max_tokens_all_users` | the shared paged-KV token pool: `max(max_model_len, 262144)`, overridable with `ORNITH_MAX_TOKENS_ALL_USERS`. Not a context cap — every request may still use the whole advertised window |
| `allocate_kv_cache` | validates vLLM's `(num_blocks, kv_heads, block_size, head_size)` against the model's own per-device geometry, allocates the cache at the **policy's** `kv_cache_dtype` rather than vLLM's torch view, and then builds the generator *around* that cache (`kv_cache=`, `page_table=` of the serving shape) so the traces are captured against vLLM's tensors from the start and nothing allocates a second cache |
| `warmup_model_prefill` | phase 1 only: compiles the serving prefill path (terminal norm, LM head, prefill sampling, state merge) at one length so the first real request does not compile them while the traces are live. This port's prefill is eager in every stage, so phase 2 has nothing to capture |
| `warmup_model_decode` | phase 1 compiles the eager decode path and **all four** sampling shapes a serving batch can take (greedy/sampled × penalties on/off); phase 2 captures the model trace and the greedy sampling trace. That is what makes every later sampling-trace capture record-only |
| `prefill_forward` | one call per scheduled prompt into the slot vLLM assigned (`empty_slots`), device-sampled token out (or host logits when the plugin wants host sampling), plus the zero `rope_deltas` the mrope-declaring config makes vLLM unpack |
| `decode_forward` | slot remap → sampling params/penalties/seeds → replay-safety → sampling-trace readiness → input staging → submit; returns device tensors when `read_from_device=False` |
| `read_decode_output` / `process_decode_output_host` | the async split: `cpu(blocking=False)` + `ttnn.record_event` behind the replays, then host formatting only |

The generator gained seven serving primitives and the model one method. Each closes a real contract
gap rather than duplicating model logic:

| addition | why the adapter could not do it |
|---|---|
| `ensure_serving_traces`, `ensure_replay_safe` | the capture/re-capture rules live with the trace state |
| `ensure_sampling_trace` | `SamplingGenerator` releases *all* sampling traces when force-argmax flips, which a serving batch does whenever it stops or starts being all-greedy. Capturing here, before the step's replay is enqueued and with `skip_precompile=True`, is what keeps a re-capture from executing a sampling graph over the live trace-region logits buffer — the hazard that hung the mesh in the optimized-full-model stage |
| `device_decode_state` | reads the token/position buffers from one shard; this is the authority a refresh merges against |
| `stage_serving_decode_inputs` | owns the refresh policy (§5) over the persistent trace inputs |
| `submit_serving_decode` | replays model + sampling traces non-blocking, and calls the replay-safety check itself so no caller can forget it |
| `prefill_requests_into_slots` | per-request prefill into an arbitrary slot with device sampling into a **scratch** token buffer — the decode token buffer carries the other slots' live tokens and `ttnn.sampling` writes all 32 of its rows |
| `remap_state_slots` (model) | a vLLM batch condense moves a request's recurrent state to another row; nothing outside the model can move a `linear_attention` layer's DeltaNet matrix and conv window |

`invalidate_sampling_params_cache` is a one-line honesty fix: the serving path drives
`SamplingGenerator.apply_decode_state` itself, so the generator's own single-parameter cache no longer
describes the device and must not be allowed to skip a later push.

---

## 5. The refresh policy is the async-decode contract

In the steady state of a traced device-sampling decode, `stage_serving_decode_inputs` copies
**nothing**: the token arrives through `tt_out_tok` and the positions advance with `ttnn.plus_one`,
both inside the trace. Three cases, and the third is the one that makes overlap safe:

* **page table changed** → copy only the page table. A growing request gets new blocks, and that is
  scheduler state the device cannot derive.
* **layout changed** (`reset_batch`, a slot remap, a freshly prefilled slot, a switch to or from host
  sampling) → re-stage tokens and positions, **merged per row** against what the device holds: for a
  row whose device position is continuous with the host's (equal, or one ahead — the async-scheduling
  lag), the device's token *and* position win.
* **otherwise** → nothing at all.

Staging the lagging host pair for a continuing row would re-run a position that already has a token,
which shows up as a doubled subword rather than as an error. `supports_async_decode=True` is claimed
only because that merge is implemented and measured (§7.1, §7.6).

---

## 6. Minimum-surface bring-up: a two-layer serving target

Per `$vllm-integration`, none of the plumbing was debugged on the 40-layer model. `ORNITH_VLLM_LAYER_INDICES=0,3`
builds one real layer of each kind (the same reduced target
[`doc/full_model/logs/probe_batch_slots.py`](../full_model/logs/probe_batch_slots.py) used) with the same
generator, adapter, plugin registration, cache and page-table shapes, terminal path, sampling path and
trace behaviour. It loads in ~40 s instead of ~4 min, and it caught, in order:

1. `ttnn.Shape` does not support slicing (`buf.shape[1:]`) — `remap_state_slots`;
2. the runner's `--plugin-config`/`--mesh-device` problems (§2);
3. both plugin-registration traps (§3);
4. `--additional-server-args "--async-scheduling"` needs `=` or argparse eats the value.

Then, on the reduced target: a first served completion, non-aligned prompt lengths 130/1000/2049/4097,
logprobs (host sampling), seeded reproducibility, penalties, greedy determinism, `min_p`, and the smoke
sampling profile at `max_num_seqs=1` and `32`. Only after all of that did the 40-layer model run.

The reduced target is a bring-up tool. No number or output from it is reported as this model's accuracy
or performance, and its own artifacts live in `/tmp`, not here — except
[`serving_primitives.json`](serving_primitives.json), which is a *mechanical* contract check (§7.1) and
says so.

---

## 7. Evidence

### 7.1 The serving primitives, against controls
[`probe_serving_primitives.py`](logs/probe_serving_primitives.py) →
[`serving_primitives.json`](serving_primitives.json), reduced two-layer target, batch 4. Six mechanical
contract checks, each with the control that makes the answer observable:

| arm | result |
|---|---|
| per-slot prefill, slots `[1, 3]` (not row order) | both prompts prefilled and decoded from their own slot |
| steady state, 5 steps after the first | `tokens/positions/page_table` copied: **none**. Counters over 6 decode calls: 1 token refresh, 1 position refresh, 1 page-table refresh, 0 device syncs |
| stale host inputs, three arms over the same prompt from an identical wiped batch | `fresh` (correct pair each step) and `stale-merged` (one-token-behind pair, row marked device-authoritative) are **token-for-token identical**; `stale-host` (same stale pair, host wins) **diverges** and repeats a token |
| page-table-only refresh | page table copied, token/position refreshes unchanged |
| batch-layout change (a second request prefilled mid-stream) | the continuing row's recurrent state is **bit-identical** across the interruption, its device position advanced by exactly 1, and its device token is the one vLLM was handed |
| slot remap `[0, 1, 3, 2]` | row 2 took row 3's state and row 3 took row 2's, **bit-identically on every device shard**; rows 0 and 1 untouched |

The stale-input arms are the async-decode proof at the primitive level: the merge reproduces the
non-stale stream exactly, and removing it corrupts. The two state arms are compared as *tensors* rather
than as tokens on purpose — at batch > 1 a neighbouring row changes the last bits of every row's logits
(§8), so token equality is the wrong instrument for a state question.

### 7.2 Non-aligned prompt lengths, and the block vLLM pads with
[`probe_serving_requests.py`](logs/probe_serving_requests.py) →
[`serving_requests.json`](serving_requests.json), full model, `max_num_seqs=32`.

Lengths 1, 3, 17, 65, **130**, 257, 999, **2049**, 4097 all served at their exact length
(`prompt_tokens` equals the request length for every one). 130 and 2049 are the interesting ones: 130
divides neither the 64-token page nor the 128-token prefill-block alignment nor the 32-row tile, and
2049 crosses the 2048-token internal prefill chunk.

130 also answers a question the padded prefill raises. The prefill's KV write covers
`align_up(len, 128)` = 256 tokens, i.e. four 64-token pages, while vLLM allocated `ceil(130/64)` = three
— so the fourth page id comes from the block table's zero padding. That is not an arbitrary block:
vLLM v1's `BlockPool` pops block 0 as `null_block` at construction and never allocates it to a request
(`vllm/v1/core/block_pool.py`), and `block_tables_for_rows` pads with zeros. The padded write therefore
lands where no request reads.

The served evidence for that is on the **`max_num_seqs=1`** server, and deliberately so: there the decode
path is bit-reproducible (§8.2), which is what makes an exact-match comparison mean something. A baseline
greedy request is answered alone, then answered again while eight 130-token requests are in flight
(serialised by the scheduler at that capacity, so this is interleaving rather than simultaneity), and the
two answers are **character-for-character identical**
([`batch1/serving_requests_max_num_seqs_1.json`](batch1/serving_requests_max_num_seqs_1.json)). Had a
padded write reached a block another request reads, or had a recycled block leaked stale content past
`cur_pos`, that comparison would have moved. The same probe at `max_num_seqs=32`
([`batch32/serving_requests_max_num_seqs_32.json`](batch32/serving_requests_max_num_seqs_32.json)) keeps
all 32 requests completing at their exact lengths, but its exact-match rows are not evidence either way,
because batch-32 decode is not bit-reproducible for this model at all (§8).

### 7.3 Batch and concurrency coverage
At `max_num_seqs=32`: 32 concurrent requests all completed, non-aligned lengths all preserved, and a
9000-token prompt served at its full length — with and without `--async-scheduling`. The CI serving-burst
benchmark is the other half of that coverage: 32/32 requests, 3200/3200 tokens, 158.5 tok/s aggregate.
`max_num_seqs=32` is the sampler's own bound (`MAX_SAMPLING_BATCH`) and the adapter refuses more with a
clear error rather than truncating, which
`test_the_serving_token_pool_holds_one_full_context_request` pins.

`max_num_seqs=1` was used for the primary single-user benchmark and for the reproducibility comparison in
§8 — both cases the skill names — and it is also the configuration a deployment that needs reproducible
completions should use (README §8).

### 7.4 Qualitative output
Two runs, because this checkpoint has a chat template and `$qualitative-check` is explicit that raw
completion prompts are continuation coverage, not a verdict:

* `run_vllm_server --stages qualitative` (raw `/v1/completions`, the shared runner's own path) →
  [`readiness_vllm/vllm_qualitative_outputs.json`](../../readiness_vllm/vllm_qualitative_outputs.json);
* [`probe_qualitative_chat.py`](logs/probe_qualitative_chat.py) → [`qualitative_chat.json`](qualitative_chat.json),
  the same six prompts through `/v1/chat/completions` (so the server renders
  `apply_chat_template(add_generation_prompt=True)`), greedy, 128 new tokens — the same shape as the two
  controls the full-model stage left in [`readiness_qualitative.json`](../full_model/readiness_qualitative.json):
  its HF reference completions and its own TTNN completions.

`check_degenerate_output.py --scope vllm` and `--scope all`: **no degenerate output detected**, on the
served outputs and on the async-scheduling run's (§7.6).

### 7.5 Sampling suite
`--sampling-profile full` (the whole `tests/tt` suite) was run twice, on two server configurations:

| configuration | result | failures |
|---|---|---|
| `max_num_seqs=32` | **54 passed, 18 failed, 1 skipped** in 623 s ([log](batch32/sampling_tests_max_num_seqs_32.log)) | every failure is a "same request must produce the same text" assertion: `test_top1_is_greedy`, `test_topk[…]`, `test_seeding`, `test_specific_seed_reproducible[42/123/999/0]`, `test_batch1_seed_reproducible[0/1]`, `test_uniform_seed_deterministic[10/32-0/1]`, `test_same_seeds_reproduce_across_batches`, `test_mixed_params_batch`, `test_{repetition,presence,frequency}_penalty_mixed_batch` |
| `max_num_seqs=1` | **65 passed, 7 failed, 1 skipped** in 289 s ([log](batch1/sampling_tests_max_num_seqs_1.log)) | every reproducibility test above now **passes**; the 7 failures are tests that cannot pass with a batch of one — they slice their config list by `max_batch_size` and then assert on ≥2 or ≥5 distinct outputs, or index `results[1]` |

Both runs are on the **final** code (§9 explains why every measurement was re-run). The 18/7 split
reproduced exactly across the before-fix and after-fix runs; only *which* `test_topk[…]`
parameterisations land on a near-tie moved, which is itself the property §8 measures.

The one skip in both runs is `test_chat_logprobs_all_vocab`, which skips itself: the plugin clamps
`max_logprobs` to 20, so `logprobs=-1` is rejected and the test's own `pytest.skip` fires.

Everything that is not a reproducibility assertion passes in both configurations: 16 `test_logprobs`
parameterisations (host sampling, since on-device log-probs need 8 or 32 devices), all five host-only
parameters (`min_p`, `bad_words`, `logit_bias`, `allowed_token_ids`, `min_tokens`), seed *variety*,
temperature variation within and between batches, `test_topk[15]`, request isolation for differing
parameters, structured output at full capacity, and the penalties-differ tests at `max_num_seqs=32`.
§8 is the investigation of the reproducibility class.

### 7.6 Async scheduling (decode overlap)
`--additional-server-args="--async-scheduling"` with `sample_on_device_mode=all` and decode tracing, at
`max_num_seqs=32` and at `1`. The server log confirms `async_scheduling=True` with no "Disabling async
scheduling" warning, i.e. the plugin accepted `supports_async_decode=True` and overlap really ran.

| check | result |
|---|---|
| degenerate-output check on the overlapped run's qualitative outputs | no degenerate output detected ([artifact](async/async_max_num_seqs_32_vllm_qualitative_outputs.json)) |
| smoke sampling profile under overlap | the same 2 reproducibility failures as the non-overlapped run at `max_num_seqs=32` — overlap adds none |
| 32-request burst, non-aligned lengths, 9000-token prompt under overlap | all completed at their exact lengths ([artifact](async/serving_requests_async_max_num_seqs_32.json)) |
| **batch-1 greedy text, overlapped vs not** | **character-for-character identical** ([`async/overlap_equivalence.json`](async/overlap_equivalence.json)); 3/3 identical repeats under overlap, 2/2 for a seeded request |
| CI serving burst 100/100/32, overlapped vs not (both warm) | TTFT P50 5504.1 → 5495.8 ms, aggregate 158.5 → 158.7 tok/s, TPOT mean 149.98 → 149.72 ms — unchanged within noise |
| primary single-user 128/128/1, overlapped vs not (both warm) | TTFT 149.4 → 148.3 ms, TPOT 23.19 → 23.14 ms, ITL P50 23.12 → 23.13 ms — unchanged |

The batch-1 row is the serving-level stale-input proof: with overlap on, vLLM submits step *N+1* before
token *N* has been applied to its host state, and the emitted text is the same text the non-overlapped
server produced. A staged stale pair would have doubled a subword.

An earlier pass at this table reported overlap *improving* the burst TTFT by 15 % (6518 → 5520 ms). That
was wrong, and the re-run on the final code is why it is not in the table: the 6518 ms measurement was
the **first** burst on a freshly started server, so it included compiling the 100-token prefill program
set, while the async number was measured on a server that had already served the sampling suite. With
both warm the two are within 0.2 %. Overlap buys nothing measurable here, and it should not: the decode
loop has no host-side gap to hide, because the token reaches the next replay through ``tt_out_tok`` on
device and the positions advance inside the trace. What the async split buys is the *freedom* to run
under `--async-scheduling` without corrupting generation.

The headline numbers in [README.md](README.md) are from the **non-overlapped** server, because that is
vLLM's default and the skill's rule is that overlap stays off unless a deployment opts in.

### 7.7 Benchmarks
Both profiles, on the final code, with the workload beside every number:

| profile | workload | TTFT P50/P99 | TPOT mean/P99 | ITL P50/P99 | aggregate | t/s/u |
|---|---|---|---|---|---|---|
| primary single-user, **warm** | 128 in / 128 out, 1 request, `--max-concurrency 1`, greedy, `max_num_seqs=1` | 149.4 / 149.4 ms | 23.19 / 23.19 ms | 23.12 / 23.68 ms | 41.36 tok/s | **43.1** (43.25 from ITL) |
| primary single-user, first request at that length | same | 172.8 ms | 24.89 ms | 23.13 / 24.1 ms | 38.5 tok/s | 40.2 |
| CI serving burst | 100 in / 100 out, 32 requests, no concurrency cap, greedy, `max_num_seqs=32` | 5504.1 / 5505.4 ms | 149.98 / 185.02 ms | 141.65 / 366.0 ms | **158.5 tok/s** | 6.67 |
| primary single-user on a `max_num_seqs=32` server | 128/128/1 | 227.9 ms | 140.1 ms | 138.2 / 139.7 ms | 7.1 tok/s | 7.1 |

Raw JSON: [`readiness_vllm/vllm_result.json`](../../readiness_vllm/vllm_result.json) and
[`vllm_ci_serving_result.json`](../../readiness_vllm/vllm_ci_serving_result.json); normalized:
[`vllm_benchmark.json`](../../readiness_vllm/vllm_benchmark.json) and
[`vllm_ci_serving_benchmark.json`](../../readiness_vllm/vllm_ci_serving_benchmark.json). The comparison
against the model's own decode floor, and what the 32-row difference means, are in
[README §1](README.md#1-headline-primary-single-user-serving-performance) and
[§2](README.md#2-secondary-ci-serving-burst-profile-vllm-nightly-shape).

### 7.8 Tests
```
pytest models/autoports/ornith_ai_ornith_1_0_35b/tests/test_generator_vllm.py -q   # 19 passed
pytest models/autoports/ornith_ai_ornith_1_0_35b/tests/test_full_model.py -q -m "not long"   # 50 passed, 5 deselected
```
Console log: [`logs/pytest_final_sweep.txt.gz`](logs/pytest_final_sweep.txt.gz). The second command is the
regression check for this stage's generator/model additions: the full-model stage's own suite, unchanged,
on the same reduced target it uses.

---

## 8. The reproducibility class of failures, measured

`$vllm-integration` asks for logit-level evidence when determinism tests fail, standalone and through
vLLM, before classifying anything. That investigation is the largest single piece of work in this stage,
and its first answer was wrong — §9 is how that was found and fixed. What follows is the corrected
measurement.

### 8.1 Through vLLM
Repeated identical greedy requests, `"The capital of France is"`, temperature 0:

| server | result |
|---|---|
| `max_num_seqs=1` | 4 of 4 completions identical (32 tokens); seeded sampling 3 of 3 identical; the overlapped (`--async-scheduling`) server produced the *same* text |
| `max_num_seqs=32` | 3 completions, 2 distinct: they agree for 10 tokens and then split — `"…political and cultural heart for centuries"` against `"…since the Middle Ages"` |

Both remain fluent, on topic and in English. A near-tie flip, not corruption.

### 8.2 Standalone, at the logits
[`probe_slot_reproducibility.py`](logs/probe_slot_reproducibility.py) →
[`slot_reproducibility.json`](slot_reproducibility.json), full 40-layer model, greedy, 13 logit vectors
per arm (the prefill's plus 12 traced decode steps), every call passing the caller-owned cache:

| arm | bit-identical | max abs Δ | min logit PCC | top-1 agreement | first flip |
|---|---|---|---|---|---|
| **batch 1**, same prompt twice from a wiped state | **yes, all 13** | **0.0** | **1.0** | 13/13 | none |
| batch 32, same slot, runs 1 vs 2 | no | 0.85 per step | 0.9973 | 10/13 | step 10 |
| batch 32, same slot, runs 2 vs 3 | no | 0.84 per step | 0.9975 | 10/13 | step 10 |
| batch 32, an occupied padding row beside it | no | 0.85 | 0.9975 | 12/13 | step 10 |
| batch 32, the same prompt in slot 7 instead of slot 0 | no | 1.09 | 0.9960 | 9/13 | step 1 |
| batch 1 against batch 32 | no | — | — | 10/13 | step 10 |

Three things follow, and together they settle the class:

* **At batch 1 the whole path is bit-reproducible** — prefill and every traced decode step, PCC 1.0.
  That is why every reproducibility test passes at `max_num_seqs=1`.
* **At batch 32 the state is also bit-identical, and the decode replay is not.** The probe compares all
  140 DeltaNet recurrent/conv buffers and the written KV pages after prefill: identical in every round.
  So the prefill, the per-slot state merge, the cache fill and the staged inputs are all deterministic;
  what deviates is the batch-32 traced decode step itself, by 0.47 – 0.85 of a logit at PCC ≥ 0.9973.
* **A tie decides the flip, and the tie is visible.** Step 10 of this prompt has a top-1/top-2 margin of
  **exactly 0.0** — the same bfloat16 value for two tokens — in *both* configurations. At batch 1 both
  runs resolve it identically (bit-identical logits, deterministic tie-break) and pick 71100; at batch 32
  the 0.8 of per-step deviation decides it, one run picking 2548 and the other 71100. From there the two
  streams feed different tokens and PCC drops to 0.62 — the large numbers in the table are the
  *consequence* of a flip, not its cause.

### 8.3 The same property, one batch size down, already on file
The datatype sweep measured this across *slots* rather than runs and wrote it up as
[§9.1](../datatype_sweep/README.md) with [`batch_slot_tie_selected.json`](../datatype_sweep/batch_slot_tie_selected.json):
at batch 4 on this exact precision policy, cross-slot max |Δ| 0.28 – 0.5, logit PCC ≥ 0.9993, and
top-1/top-2 margins of **0.0 – 0.19**, with the conclusion "cross-slot decode has never been
bit-identical, at any precision … a batch-4 decode gives each row a different position in the sharded
matmuls and collectives, so the reduction order differs per row". The numbers here — 0.47 – 0.85 at
PCC ≥ 0.9973 for batch 32 — are the same class one batch size up.

### 8.4 What is and is not fixable here
Not fixable at this layer, and not attempted: making the batch-32 decode bit-reproducible means changing
the decode geometry or the collectives, which is decoder-stage work with its own accuracy and performance
gates. Nothing at the serving layer can do it — in particular a persistent per-slot page-table row cannot,
because per-call allocation is already bit-identical and lands at the same device address every round
([`prefill_alloc_vs_recapture.json`](prefill_alloc_vs_recapture.json)).

What that means for a deployment is in [README §8](README.md#8-limitations): serve at `--max-num-seqs 1`
for bit-reproducible completions, and expect near-tie variation above it. Quality is unaffected — every
arm stays fluent and on topic, and the degenerate-output check passes on both configurations.

---

## 9. `$autofix`: the first determinism measurement was measuring a page table of zeros

The first pass at §8 reported that repeated identical *prefills* differ by 1.28 – 3.72 logit units. That
was a real measurement of the wrong thing, and `$autofix` (a forked subagent, `general-purpose`, xhigh
effort — the environment has no `fork` agent type, so the skill's loop ran with that instead) found it.

**Starting evidence.** The probe's `prefill_rerun` arm deviated; the same prefill driven straight through
`model.prefill_request_into_slot` with a long-lived page table was bit-identical. Two candidates: the
per-call page-row allocation, and the trace re-capture.

**Both refuted, with numbers.** [`probe_prefill_alloc_vs_recapture.py`](logs/probe_prefill_alloc_vs_recapture.py)
→ [`prefill_alloc_vs_recapture.json`](prefill_alloc_vs_recapture.json): six arms on one weight load —
cold capture with a genuine re-capture (107 programs compiled after capture), a long-lived row, a
per-call row, four forced re-captures — **all bit-identical, max |Δ| 0.0**, and the per-call page row
landed at the *same* device address (132992448) in all four rounds, so "the allocator layout differs
round to round" was simply false.

**The cause.** [`probe_prefill_determinism_bisect.py`](logs/probe_prefill_determinism_bisect.py) →
[`prefill_determinism_bisect.json`](prefill_determinism_bisect.json) rebuilt the failing probe verbatim
and varied one thing per arm:

| arm | bit-stable | max abs Δ | min logit PCC |
|---|---|---|---|
| `exact` (the original probe's loop) | no | 3.30 | 0.949 |
| `zeros_table_long_row` | no | 4.21 | 0.886 |
| `real_table_long_row` | **yes** | 0.0 | 1.0 |
| `real_table_percall_row` | **yes** | 0.0 | 1.0 |
| `verbatim_with_kv_cache` | **yes** | 0.0 | 1.0 |

The probe had called `prefill_requests_into_slots(page_table=…, kv_cache=None)`.
`OrnithGenerator._resolve_page_table` then **substituted the generator's own page table**, which that
probe had constructed as `torch.zeros(...)` — so every logical block of every prompt was written to
physical block 0 while the decode step read the blocks the caller's table named. The substitution warning
is in the probe's own console log, twice, and PCC 0.886 – 0.963 is an order of magnitude outside the
envelope §8.3 accepts: that was wrong *addressing*, not arithmetic.

**The fix, and why it is the right boundary.** A caller table is only foreign when it can address blocks
the attached cache does not have, and that is decided by who allocated the cache — not by whether one
call repeated the `kv_cache` handle. `_resolve_page_table` now substitutes only when the generator
allocated its own cache (`owns_cache`); a generator built on a caller-owned cache honours the caller's
table. The `owns_cache=True` branch is unchanged, which is what the shared readiness runner needs:
`run_prefill_check` deliberately passes a dummy `arange(1024)` table with `kv_cache=None` and relies on
the substitution. Verified by [`prefill_determinism_bisect_fixed.json`](prefill_determinism_bisect_fixed.json):
`exact` and `device_sampled_decode` become bit-identical (0.0), while `zeros_table_long_row` — which asks
for the zero table explicitly — stays unstable at 5.21 as the negative control.

**Consequences for this stage's evidence.**

* `logit_determinism.json` and its probe were **withdrawn**, not corrected: every arm had prefilled
  through the substituted zero table. [`probe_slot_reproducibility.py`](logs/probe_slot_reproducibility.py)
  is the replacement and is what §8.2 reports.
* The **serving path was never affected**: the adapter passes `kv_cache` on every prefill and decode call,
  so the substitution never fired in any served request. The audit in §12 shows zero substitution warnings
  in both servers' logs.
* [`probe_serving_primitives.py`](logs/probe_serving_primitives.py) had the same missing kwarg in its
  prefill calls. It was re-run after the fix — unchanged, because it now honours the table it passes — and
  §7.1 reports that re-run.
* One bug the subagent found in **this stage's own new code** while verifying against the adapter suite:
  `_refuse_device_log_probs` tested `bool(enable_log_probs)`, and a plain `[False, False, …]` list is a
  truthy object, so every device-sampled prefill raised. It never reached a served request (it was added
  after the serving runs and caught before the next one), the fix asks each row, and
  `test_the_log_probs_refusal_reads_the_rows_not_the_container` pins it. This is also why every serving
  measurement in this log was **re-run after the fix**: §7's numbers are the final code's.

## 10. Device incidents and recovery

Two, both recorded here because both cost real time and neither is a model fault.

### 10.1 Every vLLM shutdown left the fabric unable to re-open, until the boards were reset

Signature, on the *next* mesh open after a vLLM server exited cleanly:

```
TT_THROW: Device 0: Timed out while waiting for active ethernet core 29-25 to become active again.
Try resetting the board. Minimum tt-firmware version is 18.10.0 (assert.hpp:104)
 --- tt::tt_metal::RiscFirmwareInitializer::assert_active_ethernet_cores_to_reset(int)
```

It reproduced three times: after the reduced server's clean shutdown, after the reduced smoke run's, and
after the batch-32 server's. Non-vLLM jobs (the probes in this stage, which open and close the same
`1x4` ring with the same fabric config through `open_ornith_mesh`) never triggered it, so the difference
is the plugin's own teardown — `close_mesh_device` closes submeshes then the mesh and calls
`reset_fabric`, which sets `FabricConfig.DISABLED` (`vllm_tt_plugin/worker.py`, whose own comment notes
that even setting DISABLED can be unstable).

Recovery, per `$tt-device-usage`, and it worked every time (six resets over the stage, all first-attempt):

```bash
pkill -f "vllm.entrypoints"; pkill -f "EngineCore"
timeout 60  tt-smi -ls --local     # 4 Blackhole p300c chips
timeout 240 tt-smi -r              # "Resetting all PCI devices: [0, 1, 2, 3]"
timeout 60  tt-smi -ls --local     # 4 chips again
python -c "import ttnn; m=ttnn.open_mesh_device(ttnn.MeshShape(1,4), trace_region_size=0); ttnn.close_mesh_device(m)"
```

Scripted as `reset_devices.sh` and run before **every** server launch after the first failure. Infrastructure
recovery, not a model or performance result.

### 10.2 A harness tool-call timeout killed a server's process group

The first 40-layer launch stopped mid-weight-load with no error in any log. The cause was mine: the
server had been started as a background subshell of the same shell that then polled for readiness, and
when a polling call hit its 120 s tool timeout the harness killed that process group — taking the server
with it. Every later launch used `setsid nohup … &` so the server has its own session, and polling was
split into bounded chunks. Recorded because the failure mode looks exactly like a silent device crash.

---

## 11. Process hygiene

* No vLLM/EngineCore process was left holding a device: after each server the runner's SIGTERM path was
  used (`_hold_until_signal` → `terminate`), then `ps -eo pid,args | awk '/EngineCore|vllm.entrypoints/'`
  was checked, then the reset script ran. The final state of the stage was verified the same way.
* Liveness waits in this stage never used a `pgrep -f <pattern>` that could match the checking shell —
  the trap `$tt-device-usage` warns about. Waits keyed on the launched PID (`kill -0`), on an artifact
  appearing, or on `/health` returning 200.
* No Tracy, `tt-perf-report`, `TT_METAL_DEVICE_PROFILER` or `ttnn.ReadDeviceProfiler` run was made
  around any serving job, per the skill's prohibition. The device-op-level evidence for this decode stack
  is the optimized-full-model and datatype-sweep stages' own non-serving profiles.

---

## 12. Runtime-fallback audit

What the served logs say about paths that would silently cost correctness or speed:

| thing that would be wrong | how it shows up | batch-1 server | batch-32 server |
|---|---|---|---|
| a foreign page table without a caller-owned cache (the generator substitutes its own and warns) | `"was given a page_table but no kv_cache"` | **0** | **0** |
| a program compiled *inside* the traced decode loop, whose binaries a replay may have overwritten | `"a program was compiled inside the traced decode loop"` | **0** | **0** |
| trace re-captures | `"re-capturing the decode traces"` | 13 | 25 |
| the KV cache falling back to a default dtype | `"No dtype specified for the model KV cache"` | **0** | **0** |
| async scheduling silently disabled | `"Disabling async scheduling"` | **0** | **0** |
| the sampler taking the force-argmax path | `"Forcing argmax sampling"` | **0** | **0** |

The re-captures are the designed mechanism, not a fallback: a prompt length the process has not seen
compiles programs whose kernel binaries were allocated while the traces were live, and
`_ensure_traces_replay_safe` re-captures before the next replay can overwrite them. 13 and 25 are the
number of *distinct* prompt lengths those servers saw (the sampling suite alone sends many), and
`test_traces_are_recaptured_when_a_new_program_is_compiled` pins that a repeat length re-captures
nothing.

The zero force-argmax count is also the intended state, and it is a *measured* decision rather than an
oversight: the full-model stage rejected `allow_force_argmax` because it would all-gather the whole
249856-wide logits row and run a global `ttnn.argmax` — "precisely the sampler-op-dominates-decode shape
the goal forbids" ([`doc/full_model/README.md` §4.2](../full_model/README.md),
[`doc/optimized_full_model/README.md` §9](../optimized_full_model/README.md)). Greedy therefore runs the
same captured split-sampling graph as every other mode (local top-32 per vocabulary shard, gather 4×32
candidates, `ttnn.sampling` with `k=1, p=0, temp=1`), which is the fastest strategy measured for this
mesh. Serving inherits it unchanged: the serving benchmark's 23.15 ms/token against the model's own
23.165 ms is the proof that no other sampling path crept in.

Two more fallbacks that are *reachable by request* rather than by accident, and are supposed to be:

* **host sampling.** The plugin routes a step to its own host sampler for log-probs (on a 4-device mesh),
  `min_p`, `bad_words`, `logit_bias`, `allowed_token_ids`, `min_tokens` and structured output. The
  adapter then returns logits — `[slots, 1, vocab]` — instead of tokens, and its
  `serving_counters["host_sampled_decodes"]` counts those steps separately from
  `device_sampled_decodes`. It is never entered for an ordinary greedy or sampled request.
* **eager decode.** `decode_forward(enable_trace=False)` exists for the warm-up's compile phase and for
  `trace_mode=none`. Serving runs with `trace_mode=all`, so every served token is a trace replay; the
  `decode_syncs` counter staying at 0 across a request is the other half of that proof (an eager step
  synchronizes).

---

## 13. The two runner-side gates, and the advisories they print

```bash
python models/common/readiness_check/check_degenerate_output.py \
  --model-dir models/autoports/ornith_ai_ornith_1_0_35b --missing-artifacts critical --scope all
# -> "No degenerate output detected."   exit 0

python .agents/scripts/check_context_contract.py \
  --model-dir models/autoports/ornith_ai_ornith_1_0_35b --hf-model ornith-ai/Ornith-1.0-35B \
  --stage vllm --require-contract
# -> "Context contract OK ... target=262144, supported=262144 (full HF context)."   exit 0
```

The context gate also prints eleven **advisory** lines, and they are worth naming so nobody reads them as
a served cap. Ten are one line each from a probe's own console log — every probe logs
`building OrnithModel: … max_context=N` at startup, and a probe that only needs a 2048-, 4096- or
8192-token window says so. The eleventh is a pre-existing comment in `tt/functional_decoder.py` that uses
`max_context=8000` as an illustration. Advisories are not failures (the gate returns 2 only for a JSON
*key* below the supported context, and `--strict-caps` is not used by the stage gate), and nothing in
`tt/`, `readiness_vllm/` or this directory's JSON caps the served window: `doc/context_contract.json`'s new
`vllm_integration` block records 262144 served against 262144 advertised, and
`readiness_vllm/vllm_serving_capability.json` shows the built model carrying it.

---

## 14. Files changed

In this repo:

| file | change |
|---|---|
| `tt/generator_vllm.py` | **new.** The vLLM adapter: `TTQwen3_5MoeForConditionalGeneration` |
| `tt/generator.py` | one new `serving (vLLM) API` section (the seven primitives of §4) plus four small changes elsewhere: `_sample_traced` passes `skip_precompile=True`; the constructor allocates the prefill sampling scratch buffer (before any capture) and a `sampling_trace_captures` counter; `submit_serving_decode` calls the replay-safety check itself; and `_resolve_page_table` substitutes only when the generator owns its cache (§9) |
| `tt/model.py` | **+57 lines**: `remap_state_slots` and its `_remap_rows` helper |
| `tests/test_generator_vllm.py` | **new.** 8 host-only cases (registration, the flags the plugin reads, the interface vLLM introspects, the shared adapter contract, no sampling path of its own, the token-pool bound, the log-probs refusal reading rows rather than the container, visual-payload refusal) + 11 device cases on the reduced target (cache ownership, block-size refusal, per-slot prefill, steady-state refresh, stale-pair merge, page-table-only refresh, slot remap and its identity skip, host-sampling logits, the precision-config propagation, the capability report) |
| `models/common/readiness_check/run_vllm_server.py` | **+73 lines**: `_tt_config_flag()` picks `--additional-config` / `--plugin-config` from the installed engine, and `_mesh_device()` accepts a mesh name or an explicit `(rows, cols)` grid. Both are fixes against the current vLLM fork, not model-specific |
| `doc/vllm_integration/**` | **new.** This log, the README, nine probes with the console log of their final run, the evidence JSON, and the archived per-configuration artifact sets (`batch1/`, `batch32/`, `async/`) |
| `doc/context_contract.json` | **+1 block**: `vllm_integration`, recording 262144 served against 262144 advertised, the KV-pool sizing and its cost, the non-aligned-length evidence, the 64-token block size, and the tested batch coverage |
| `readiness_vllm/**` | **new.** The shared runner's artifacts: `server.log`, `sampling_tests.log`, `vllm_qualitative_outputs.json`, `vllm_result.json`, `vllm_benchmark.json`, `vllm_benchmark.log`, `vllm_ci_serving_result.json`, `vllm_ci_serving_benchmark.json`, `vllm_ci_serving_benchmark.log`, and `vllm_serving_capability.json` (written by the adapter at the end of warm-up, from inside the engine-core process) |

Nothing in `tt/model.py`, `tt/optimized_decoder.py`, `tt/multichip_decoder.py` or `tt/precision_config.py`
changed behaviour for the earlier stages: the only model addition is a new method nothing else calls, and
the generator's existing entry points are untouched apart from `skip_precompile=True` on the lazy
sampling-trace capture (which is strictly safer) and one new buffer allocated at construction.

Outside this repo, in the `tenstorrent/vllm` checkout (kept here as
[`vllm_tt_plugin_changes.diff`](vllm_tt_plugin_changes.diff), not committed to tt-metal):

| file | change |
|---|---|
| `plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py` | register this adapter for `TTQwen3_5MoeForConditionalGeneration` and, replacing upstream's class, for `Qwen3_5MoeForConditionalGeneration` (§3) |
| `plugins/vllm-tt-plugin/src/vllm_tt_plugin/worker.py` | `get_fabric_router_config()` + `set_fabric` honouring a `fabric_router_max_packet_bytes` TT-config key, because fabric configuration has to happen before the mesh is opened (§3) |
