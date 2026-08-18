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

**What replacing the architecture costs, and the scope it takes.** `ModelRegistry.register_model` is
unconditional: it replaces the *architecture*, not one checkpoint, so in a TT process every checkpoint
declaring `Qwen3_5MoeForConditionalGeneration` now resolves to this port. Registration runs at
plugin-import time, before any model config exists, so there is nothing available to scope it against.
The consequences, in order of how likely they are to bite:

* a **different Qwen3.5-MoE checkpoint** served on TT would be built by this port's code rather than
  upstream's. It fails loudly rather than serving something wrong — `OrnithConfig.from_hf` raises on
  `layer_types` it does not implement, and the weight loader raises `no checkpoint entries under
  'model.language_model.'` for a checkpoint that does not nest the way this one does — and
  `initialize_vllm_model` now logs a warning naming the override whenever the resolved checkpoint is not
  `ornith-ai/Ornith-1.0-35B`. What it will *not* do is fall back to upstream;
* a **second TT port of the same architecture** would collide on this line. There is none today; the
  comment at the registration says so, and that line is where the collision must be resolved;
* **image requests** are refused by the API for any checkpoint of this architecture in a TT process,
  because the class vLLM introspects is text-only. For this checkpoint that is correct — the port has no
  vision half — but it is a property of the architecture name now, not of the checkpoint.

Only `ornith-ai/Ornith-1.0-35B` was validated by this stage.

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
or performance, and its own artifacts live in `/tmp`, not here — with two deliberate exceptions, both
*mechanical* rather than quality or speed claims, and both labelled as reduced-target at the top of their
files: [`serving_primitives.json`](serving_primitives.json), the contract checks of §7.1, and
[`reduced_target/`](reduced_target/), the run-pair counts that localise the batch ≥ 8 nondeterminism to
the collectives (§8.3). The property they measure — "are two identical runs bit-identical" — is not a
number that depends on having all 40 layers, and every one of their conclusions is re-confirmed on the
full model in [`decode_nondeterminism.json`](decode_nondeterminism.json) and
[`slot_reproducibility.json`](slot_reproducibility.json).

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
[`probe_serving_requests.py`](logs/probe_serving_requests.py), full model, run on three server
configurations. The top-level [`serving_requests.json`](serving_requests.json) is the `max_num_seqs=1`
run on the fixed probe, and [`batch1/serving_requests_max_num_seqs_1.json`](batch1/serving_requests_max_num_seqs_1.json)
is the same file archived per configuration (byte-identical; the pre-fix copy it replaced is gone rather
than kept, because its `ignore_eos` arm was measuring nothing — see below). The other two are
[`batch32/serving_requests_max_num_seqs_32.json`](batch32/serving_requests_max_num_seqs_32.json) and, on
the non-overlapped control server,
[`async/serving_requests_no_async_max_num_seqs_1.json`](async/serving_requests_no_async_max_num_seqs_1.json).

Lengths 1, 3, 17, 65, **130**, 257, 999, **2049**, 4097 all served at their exact length
(`prompt_tokens` equals the request length for every one), and a 9000-token prompt too. 130 and 2049 are
the interesting ones: 130 divides neither the 64-token page nor the 128-token prefill-block alignment nor
the 32-row tile, and 2049 crosses the 2048-token internal prefill chunk.

**Why `all_completed` was false, and what fixed the measurement rather than the model.** Every length is
asked for 8 output tokens, and in the first version of this arm the 2049- and 4097-token prompts came back
with 3 and 1 tokens. That is not truncation: these prompts are *random token ids*, and the model answered
some of them with an immediate end-of-text. The probe now records `finish_reason` and asks each length
twice, once with `ignore_eos`:

| length | with `ignore_eos` | plain request |
|---|---|---|
| 1, 3, 17, 65, 130, 257, 999 | 8 tokens, `finish_reason: length` | 8 tokens, `length` |
| 2049 | **8 tokens**, `length` | 3 tokens, **`stop`** |
| 4097 | **8 tokens**, `length` | 1 token, **`stop`** |
| 9000 | **8 tokens**, `length` | — |

So with `ignore_eos` the arm is `all_completed: true` at every length, and the short plain completions are
the model choosing to stop, which the recorded `finish_reason` now says outright. (The first attempt at
this passed `ignore_eos` inside `extra_body`, which is an OpenAI *client* concept — a raw POST nests it
into a field the server ignores, and both arms came back identical. That is why the flag now goes at the
top level, and why this artifact was re-measured on a fresh server rather than edited.)

130 also answers a question the padded prefill raises. The prefill's KV write covers
`align_up(len, 128)` = 256 tokens, i.e. four 64-token pages, while vLLM allocated `ceil(130/64)` = three
— so the fourth page id comes from the block table's zero padding. That is not an arbitrary block:
vLLM v1's `BlockPool` pops block 0 as `null_block` at construction and never allocates it to a request
(`vllm/v1/core/block_pool.py`), and `block_tables_for_rows` pads with zeros. The padded write therefore
lands where no request reads.

The served evidence for that is on a **`max_num_seqs=1`** server, and deliberately so: there the decode
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
committed served outputs, on the `--max-num-seqs 32` server's and on the `--no-async-scheduling` control's
(§7.6). Console log: [`logs/check_degenerate_output.txt`](logs/check_degenerate_output.txt). Over the ten
committed raw completions the gate measures — its `MIN_WORDS_FOR_DUPLICATION` is 20 words, and two sampled
completions self-terminate at 13 and 8 — adjacent-duplication is 0.0000 – 0.0140 against a 0.10 threshold
and trigram-loop 0.0149 – 0.1304 against 0.50.

The per-prompt reading, against the HF and full-model controls, is in
[README §4](README.md#4-qualitative-output-with-controls). It is written from the *committed* artifacts, and
that mattered: the qualitative set was regenerated twice while the batch-1 evidence was re-run (§7.7), and
round 3 of the review found §4 still describing the **previous** generation's sampled completions. The chat
run's divergence points against the HF control (25/34/236/214/213/130 characters by prompt) are unchanged
across all of those runs.

### 7.5 Sampling suite
`--sampling-profile full` (the whole `tests/tt` suite) was run twice, on two server configurations:

| configuration | result | failures |
|---|---|---|
| `max_num_seqs=32` | **54 passed, 18 failed, 1 skipped** in 623 s ([log](batch32/sampling_tests_max_num_seqs_32.log.gz)) | every failure is a "same request must produce the same text" assertion. Exactly: `test_mixed_params_batch`, `test_seeding`, `test_same_seeds_reproduce_across_batches`, `test_specific_seed_reproducible[0/123/999]`, `test_batch1_seed_reproducible[0/1]`, `test_uniform_seed_deterministic[10-0/10-1/32-0/32-1]`, `test_top1_is_greedy`, `test_topk[15]`, `test_topk[32]`, `test_{repetition,presence,frequency}_penalty_mixed_batch`. `test_topk[19]` and `test_specific_seed_reproducible[42]` passed |
| `max_num_seqs=1` | **65 passed, 7 failed, 1 skipped** in 292 s ([log](batch1/sampling_tests_max_num_seqs_1.log.gz)) | **15 of the 18 above now pass.** The 7 failures are the three `test_different_*_penalties`, the three `test_*_penalty_mixed_batch` and `test_uniform_noseed_varied` — all of which slice their request list by `max_batch_size` and then assert on ≥2 distinct outputs, ≥5 distinct outputs, or a cross-row comparison. At capacity 1 the `mixed_batch` slice is **empty**, which two of them report as `Got 0 unique results out of 0.` and the third (`test_frequency_penalty_mixed_batch`) as an outright `IndexError: list index out of range` when it indexes that empty list |

The three `test_*_penalty_mixed_batch` cases therefore fail in *both* configurations, for different
reasons: the reproducibility reason at 32, the empty-slice reason at 1. Neither is a correctness failure,
and both are visible in the logs above.

Both runs are on the **final** code (§9 explains why every measurement was re-run). The 18/7 split
reproduced across the before-fix and after-fix runs; only *which* `test_topk[…]` and
`test_specific_seed_reproducible[…]` parameterisations land on a near-tie moved, which is itself the
property §8 measures — the earlier run failed `test_topk[19]` and `test_specific_seed_reproducible[42]`
and passed `test_topk[15]`/`[32]`, and the final run does the opposite.

The `max_num_seqs=1` result then reproduced **three** times, on three different servers: 7 failed / 65
passed / 1 skipped each time, with the same seven names every time (`test_uniform_noseed_varied`, the three
`test_different_*_penalties`, the three `test_*_penalty_mixed_batch`). Only the last of those runs is
committed — `server.log` and `sampling_tests.log` are truncated per launch and only one server's copy is kept
(§7.7) — so the committed evidence is that run: **292.00 s**, and the failure list above reads straight out
of it. The two earlier repeats are recorded here in prose only. That is a stronger statement than the batch-32 side, where membership moves
between runs because it depends on which requests land on a near-tie.

The one skip in both runs is `test_chat_logprobs_all_vocab`, which skips itself: the plugin clamps
`max_logprobs` to 20, so `logprobs=-1` is rejected and the test's own `pytest.skip` fires.

**Exactly which cases pass in both**, extracted from the two committed logs rather than described: **50 of
73**. By file: 20 `test_logprobs` parameterisations (host sampling, since on-device log-probs need 8 or 32
devices) and 8 `test_build_logprobs_from_topk`, 5 `test_host_only_params` (`min_p`, `bad_words`,
`logit_bias`, `allowed_token_ids`, `min_tokens`), 3 `test_config`, 1 `test_structured_output_dp1` (full
capacity, mixed structured and plain), and 13 of 28 `test_seeding_and_variety` — the two seed-*variety*
cases, a negative seed not crashing, all six temperature-variation cases, `test_topk[19]`,
`test_specific_seed_reproducible[42]`, and `test_uniform_seed_deterministic[1-0]`/`[1-1]`.

The rest split cleanly: **15** pass at 1 and fail at 32 (the reproducibility class, including
`test_topk[15]`, `test_topk[32]` and `test_mixed_params_batch` — so neither of those belongs in a
"passes in both" list, which an earlier version of this section and README §6 both claimed), **3** fail in
both (`test_*_penalty_mixed_batch`), **4** fail at 1 and pass at 32 (`test_uniform_noseed_varied` and the
three `test_different_*_penalties`, all of them empty- or single-row slices at capacity 1), and **1** skips
itself. 50 + 15 + 3 + 4 + 1 = 73, and the two totals follow: 65/7 at capacity 1, 54/18 at 32.

§8 is the investigation of the reproducibility class.

### 7.6 Async scheduling (decode overlap), and the default this stage first got wrong
**The correction first, because it changes what every other number in this stage means.** This vLLM
resolves an unset `async_scheduling` to **True**: `vllm/config/vllm.py` says "Enable async scheduling
unless there is an incompatible option" and only turns it off for speculative decoding, an unsupported
executor backend or Mamba prefix caching. So *every* server this stage ran with no scheduling flag — the
headline single-user server included — ran with **overlap enabled**, and passing
`--additional-server-args="--async-scheduling"` changed nothing. Every preserved server log says
`Asynchronous scheduling is enabled` and `Scheduler class: TTScheduler, async_scheduling=True`, including
the ones an earlier version of this log called "non-overlapped".

The earlier "sync vs async" comparison was therefore two overlapped servers, and its conclusion — "overlap
buys nothing measurable" — was an artefact of that.

The default is resolved *before* the plugin sees the config, and the plugin still gates it on the model's
declared capability: `VllmConfig.__post_init__` resolves `async_scheduling=None` to `True` around
`vllm/config/vllm.py:657-699` and only calls `current_platform.check_and_update_config` at line 917, where
`vllm_tt_plugin/platform.py:760-768` turns overlap back off for a model whose `model_capabilities` do not
declare `supports_async_decode`. So a TT model without the capability is not silently overlapped — it gets
the `Disabling async scheduling` line the audit in §12 counts (zero on every server here, because this
adapter *does* declare it). The only rough edge is the wording: that warning says overlap "was requested"
when in fact it was defaulted on, which is what made this default hard to notice in the first place. The genuine control is `--no-async-scheduling`
(`--async-scheduling` is a `BooleanOptionalAction`, so the negative form exists), and it was run:
[`async/server_no_async_max_num_seqs_1.log.gz`](async/server_no_async_max_num_seqs_1.log.gz) logs
`Asynchronous scheduling is disabled` and `async_scheduling=False`.

| primary single-user 128/128/1, warm, `max_num_seqs=1` | overlap **on** (the default): 3 servers | overlap **off** (`--no-async-scheduling`): 2 servers, 3 warm runs |
|---|---|---|
| TPOT | **23.146 – 23.174 ms** | 24.711 – 24.821 ms |
| ITL P50 | **23.133 – 23.140 ms** | 24.591 – 24.595 ms |
| decode t/s/u (TPOT-derived) | **43.15 – 43.20** | 40.29 – 40.47 |
| TTFT P50 | 145.3 – 151.7 ms | 145.7 – 157.9 ms |
| `serving_counters["async_reads"]` for the whole process | **10831**, exactly its `decode_calls` | **0** |
| artifacts | [`batch1/vllm_benchmark_repeat_warm.json`](batch1/vllm_benchmark_repeat_warm.json) (the committed one), [`…_second_server_warm.json`](batch1/vllm_benchmark_second_server_warm.json), [`async/async_max_num_seqs_1_vllm_benchmark_warm.json`](async/async_max_num_seqs_1_vllm_benchmark_warm.json) | [`async/no_async_vllm_benchmark_warm.json`](async/no_async_vllm_benchmark_warm.json), [`…_second_server_warm.json`](async/no_async_second_server_warm.json), [`…_second_server_warm_run2.json`](async/no_async_second_server_warm_run2.json) |

So overlap is worth **~1.45 ms per token, 6.3 %**, at batch 1 on this model — in the decode loop only, since
the TTFT ranges overlap completely — and the decode ranges do not overlap at all across five warm runs.
The `async_reads` counter is the mechanical proof of which path each server took: it equals the decode-call
count with overlap on, and the non-overlapped servers never entered `read_decode_output(async_read=True)`
at all. It also explains the decode-floor comparison in §7.7: the
model's own traced token-out floor is 23.165 ms, the overlapped servers' ITL is 23.133 – 23.140 ms (at the floor),
and the non-overlapped servers' is 24.59 ms — the ~1.45 ms of host-side gap that the async split is there
to hide.

**Equivalence, at equal request length.** [`probe_overlap_equivalence.py`](logs/probe_overlap_equivalence.py)
asks the same greedy request on both servers with the same `max_tokens=32`:

| check | result |
|---|---|
| greedy text, overlap on vs off | **character-for-character identical** ([`async/overlap_texts_no_async.json`](async/overlap_texts_no_async.json) carries both arms) |
| greedy repeats within each arm | 3/3 identical on each |
| seeded (temp 0.8, top-p 0.9, seed 4242) repeats within each arm | 2/2 identical on each |
| seeded text across the two arms | **differs** — investigated in §7.6.1 |
| smoke sampling profile, overlap off | 3 passed, 1 skipped — the same result as with overlap on ([`async/sampling_tests_smoke_no_async.log.gz`](async/sampling_tests_smoke_no_async.log.gz)) |
| request shapes, overlap off | every non-aligned length served exactly, 32-way burst complete, null-block containment identical ([`async/serving_requests_no_async_max_num_seqs_1.json`](async/serving_requests_no_async_max_num_seqs_1.json)) |
| degenerate-output check, overlap off | no degenerate output detected ([artifact](async/vllm_qualitative_outputs_no_async_max_num_seqs_1.json)) |

The greedy row is the serving-level stale-input proof: with overlap on, vLLM submits step *N+1* before
token *N* has been applied to its host state, so a row that kept the host's lagging pair would double a
subword. The text is identical to the non-overlapped server's, which is what the refresh policy's per-row
merge (§5) is for.

An earlier artifact, `async/overlap_equivalence.json`, is **withdrawn**: its two arms were both overlapped
servers *and* it compared unequal completion lengths (one arm requested fewer tokens), so its
"identical on the compared prefix" claim measured neither overlap nor a full completion.
`probe_overlap_equivalence.py` replaces it — a script rather than a hand-assembled file, with the request
length fixed on both arms and the tag recorded in the artifact.

**One more correction to the burst row.** The CI serving burst was measured twice, and both runs were
overlapped (158.5 and 158.7 tok/s aggregate, TTFT P50 5504.1 and 5495.8 ms) — that pair is a repeatability
check, not an overlap comparison. No non-overlapped burst was measured; the overlap comparison in this
stage is the batch-1 pair above.

#### 7.6.1 A seeded request is reproducible within a server, not across two of them
The seeded arm differs between the two servers while being 2/2 reproducible inside each. Two candidate
causes, and they are not the same thing: overlap itself, or the sampler's RNG state depending on how much
traffic the server has already served (the overlapped arm ran after the full sampling suite and ~500
requests; the non-overlapped arm ran on a fresh server).

The control keeps the server fixed — non-overlapped, one process — and varies only history: the same seeded
request before and after the smoke sampling profile.

The control was run on **both** scheduling modes, one server each, the same seeded request before and
after the smoke sampling profile (~90 requests of mixed parameters):

| control | greedy, fresh vs after traffic | seeded, fresh vs after traffic |
|---|---|---|
| `--no-async-scheduling` server ([fresh](async/overlap_texts_no_async_fresh.json), [after](async/overlap_texts_no_async_after_traffic.json)) | identical | **identical** |
| default (overlapped) server ([fresh](async/overlap_texts_async_fresh.json), [after](async/overlap_texts_async_after_traffic.json)) | identical | **identical** |

So history is not the cause: a seeded completion survives ~90 intervening requests unchanged, in both
modes. What differs is the mode itself, and it differs *reproducibly* — four arms, two per mode, and each
mode returns its own text every time:

* overlap **on**: `" Paris, a city of approximately 2.2 million inhabitants, which serves as the country's economic hub…"`
* overlap **off**: `" Paris, a city renowned for its rich history and iconic landmarks such as the Eiffel Tower…"`

Both completions are fluent, on topic and in English, and greedy output is identical across the modes, so
this is not a correctness failure — it is the sampler's draw sequence landing differently under the two
schedulers, most plausibly because the async path applies per-slot sampling state at a different point
relative to the decode steps it submits (the async scheduler runs a request's step *N+1* before token *N*
is applied on the host). Pinning that down is plugin work in `vllm_tt_plugin`'s async decode path, not
adapter work: the adapter is handed the sampling parameters per step and applies them where it is told.

**What a deployment should take from it**: a seed reproduces a completion on a given server configuration,
including across other traffic — but the same seed on the same model with `--async-scheduling` toggled is a
different sample. It is recorded in [README §8](README.md#8-limitations) as such. Reproducer:
`probe_overlap_equivalence.py --tag a` on one server, then `--tag b --compare-with <a's file>` on the other.

### 7.7 Benchmarks
Both profiles, on the final code, with the workload beside every number:

| profile | workload | TTFT P50/P99 | TPOT mean/P99 | ITL P50/P99 | aggregate | t/s/u |
|---|---|---|---|---|---|---|
| **primary single-user, warm — the committed artifact** | 128 in / 128 out, 1 request, `--max-concurrency 1`, greedy, `max_num_seqs=1`, overlap on (default) | 151.7 / 151.7 ms | 23.174 / 23.174 ms | 23.133 / 23.72 ms | 41.36 tok/s | **43.15** (43.23 from ITL) |
| primary single-user, first request at that length, same server | same | 228.3 ms | 24.894 ms | 23.134 / 25.47 ms | 37.76 tok/s | 40.17 |
| primary single-user, warm, an earlier server | same | 145.3 ms | 23.170 ms | 23.140 / 23.84 ms | 41.45 tok/s | 43.16 |
| primary single-user, warm, another server | same | 148.3 ms | 23.146 ms | 23.139 / 23.76 ms | 41.45 tok/s | 43.20 |
| primary single-user, warm, `--no-async-scheduling` | same, overlap **off** | 157.9 ms | 24.742 ms | 24.591 / 27.66 ms | 38.78 tok/s | 40.42 |
| primary single-user, warm, `--no-async-scheduling`, a second server | same, overlap **off** | 149.0 ms | 24.711 ms | 24.595 / 27.53 ms | 38.93 tok/s | 40.47 |
| primary single-user, first request, `--no-async-scheduling` | same, overlap **off** | 240.5 ms | 26.493 ms | 24.576 / 26.69 ms | 35.50 tok/s | 37.75 |
| CI serving burst | 100 in / 100 out, 32 requests, no concurrency cap, greedy, `max_num_seqs=32` | 5504.1 / 5505.4 ms | 149.98 / 185.02 ms | 141.65 / 366.0 ms | **158.5 tok/s** | 6.67 |
| primary single-user on a `max_num_seqs=32` server | 128/128/1 | 227.9 ms | 140.069 ms | 138.211 / 139.7 ms | 7.1 tok/s | 7.14 |

Raw JSON: [`readiness_vllm/vllm_result.json`](../../readiness_vllm/vllm_result.json) and
[`vllm_ci_serving_result.json`](../../readiness_vllm/vllm_ci_serving_result.json); normalized:
[`vllm_benchmark.json`](../../readiness_vllm/vllm_benchmark.json) and
[`vllm_ci_serving_benchmark.json`](../../readiness_vllm/vllm_ci_serving_benchmark.json). The comparison
against the model's own decode floor, and what the 32-row difference means, are in
[README §1](README.md#1-headline-primary-single-user-serving-performance) and
[§2](README.md#2-secondary-ci-serving-burst-profile-vllm-nightly-shape).

**Where the first-request cost lands.** Not only in TTFT, which is what an earlier version of this section
and README §1 both said. The cold row's TPOT is 24.894 ms against 23.174 warm while its ITL *median* is
unchanged at 23.134, and `vllm bench serve` excludes TTFT from TPOT — so 3389.9 − 228.3 = 3161.6 ms of
decode window over 127 intervals carries **~218 ms** more than 127 × 23.174, concentrated in at most one
interval (ITL P99 25.47). The committed server log shows why. The cold request *enters* prefill at `18:55:38.314` (that
`prefill_forward` INFO is logged before `prefill_requests_into_slots` runs), and `_ensure_traces_replay_safe`
logs its re-capture at `18:55:38.494` — but not from the prefill: the re-capture is triggered by
`submit_serving_decode`'s `ensure_replay_safe()` (`tt/generator.py`), which runs on the **first decode
step**. So it lands after the first token has been returned, and is charged to the first inter-token
interval rather than to TTFT. The warm request at `18:55:52.562` logs no re-capture at all. The
datatype-sweep stage measured the same quantity standalone as
[`cold_prompt_length_cost.hidden_cost_ms`](../datatype_sweep/post_selection_token_out.json) = 250.4 ms with
one re-capture. README §1 and its limitation 2 now say both halves; round 3 of the review caught the
single-token framing.

**Two things about how this table was assembled, both of which changed it.** First, the warm single-user
figure reproduces across three servers to within 0.028 ms of TPOT (23.146, 23.170, 23.174) and 0.007 ms of
ITL P50, while the *first-request* TTFT does not (172.8, 228.3, 236.3 ms on three servers), because it
depends on what that process had already compiled. That is why the first request is a separate row rather
than folded into the headline.

Second, an earlier version of this table quoted a warm run of **149.4 ms TTFT / 23.19 ms TPOT** whose
artifact no longer existed: `readiness_vllm/vllm_benchmark.json` is truncated per launch, and the two files
archived under `batch1/` at that point (`vllm_benchmark_primary_canonical_run.json`, 162.3 ms / 24.970 ms,
and `…_cold_first_request.json`, 172.8 ms / 24.888 ms) are both *first-request* runs — neither is the number
that was published. Rather than keep a headline no file supports, the whole batch-1 set was re-run on one
server, its artifacts archived before shutdown, and the headline is now that server's warm run, which is
also what `readiness_vllm/` holds. The published figure moves by 2.3 ms of TTFT and 0.02 ms of TPOT; the
conclusion — serving decode sits at the model's own token-out floor — does not.

That took **two** attempts, and the second failure is worth recording because it is the same trap: the first
re-run did produce a coherent `readiness_vllm/` set, and then a later `--no-async-scheduling` benchmark cycle
— run to give the overlap comparison a second server — overwrote the benchmark JSON, the raw result, the
server log and both capability reports with the *control's* numbers, while the README still described them
as the headline server's. Round 2 of the review caught it. The fix is not only "re-run last": the final
cycle now ends by `md5sum`-comparing the runner's set against the archived per-configuration copies and
printing the server log's scheduling state, request count and re-capture count, so the attribution is
checked rather than assumed. Every one of those five files is byte-identical to its `batch1/` copy on the
committed tree.

### 7.8 The precision policy the served build actually carries
[`readiness_vllm/vllm_serving_capability.json`](../../readiness_vllm/vllm_serving_capability.json) is
written by the adapter at the end of warm-up, from inside the engine-core process, and it reads the
*built* objects rather than a config file (`OrnithModel.precision_summary()`). What it records for the
served model:

| | |
|---|---|
| policy on every one of the 40 layers | `C06-proj-bfp4-lofi`, `reduced: false` |
| weight groups | routed expert gate/up + down `bfloat4_b`, dense projections `bfloat4_b`, shared expert `bfloat8_b`, router `bfloat16`, LM head `bfloat4_b` |
| math fidelity | dense projections **LoFi**, routed experts **LoFi**, shared expert HiFi2, router HiFi4, SDPA HiFi2, DeltaNet state HiFi4 — one identical set on all 40 layers |
| KV-cache dtype, read off the live cache tensors | `DataType.BFLOAT8_B` on all 10 `full_attention` layers — i.e. the cache **vLLM owns** was allocated at the policy's dtype, not at vLLM's `torch.bfloat16` view |
| residual / CCL activation dtype | `bfloat16` / `None` (no cast), on every layer |
| layer exceptions | `[]` |

`policy_to_dict(load_selected_policy())` and this report's `selected_config` block are equal field for
field; `tests/test_generator_vllm.py::test_the_serving_build_carries_the_selected_precision_config`
asserts exactly that against
[`doc/datatype_sweep/selected_precision_config.json`](../datatype_sweep/selected_precision_config.json).
The committed artifact file additionally carries nine prose and provenance keys the sweep put there for
humans — `stage`, `status`, `selected_from_run`, `selection_rule`, `measured`, `kv_cache_update_contract`,
`lm_head_note`, `sampling_note`, `override_note`; those are not policy fields, `policy_to_dict` does not
emit them, and their absence from the served report is not a policy difference. Every field that *is* a
policy field is equal.

**One way this report could lie, closed in round 3.** The writer fired for *any* built adapter, including
the two-layer bring-up targets the adapter suite builds — so running the documented
`pytest tests/test_generator_vllm.py` replaced both committed reports with a `reduced: true,
layer_indices: [0, 3]` description of a model nothing served, and silently broke the md5 attribution the
evidence rests on. It only survived because a real server happened to run afterwards. `_write_serving_capability`
now refuses to write when `capability["reduced"]` is set, logging the reduced report instead, and
`test_a_reduced_target_does_not_overwrite_the_served_capability_report` asserts the committed bytes are
unchanged after a reduced build calls it. Verified end to end: the suite's 24 "not writing" log lines, and
`md5sum -c` on both artifacts after the run.

**One thing that report cannot tell you, and the second copy that can.** Its `serving_counters` block is
all zeros, and necessarily so: warm-up drives the generator's primitives directly rather than going
through `decode_forward`/`prefill_forward`, so the adapter's own counters have nothing to count yet when
that copy is written, and no request has arrived either. Reading those zeros as "no refresh, no remap and
no host-sampled decode ever happened while serving" would be wrong in the other direction. So the adapter now registers an `atexit` hook that
writes the same report again at engine-core exit, as
[`readiness_vllm/vllm_serving_capability_final.json`](../../readiness_vllm/vllm_serving_capability_final.json),
whose counters *do* describe the traffic that server handled — no-refresh steady-state steps, page-table
refreshes, slot remaps, async reads, and device- against host-sampled decodes. The warm-up copy is kept
because it is the one a crashed process still leaves behind.

### 7.9 Which console log belongs to which run
Every probe keeps the console log of the run whose artifact is committed, as `logs/<probe>.txt`. Three
files need a word of explanation, because a probe that was re-run leaves more than one:

* [`logs/probe_slot_reproducibility_first_run.txt.gz`](logs/probe_slot_reproducibility_first_run.txt.gz) —
  the *first* run of that probe, before the batch sweep, the all-rows arm and the read-twice arm were
  added. Kept because it is the log behind the numbers the stage first reported;
* [`logs/probe_slot_reproducibility_extended_attempt_crash_tail.txt`](logs/probe_slot_reproducibility_extended_attempt_crash_tail.txt) —
  the tail of the first *extended* attempt, which crashed in the new batch-sweep arm: `state_is_live` is a
  model-level flag and the arms before it had left a prompt in the model, so capturing the sweep
  generator's traces was refused. Kept because the fix (wipe first, as `generate(reset=True)` does) and the
  probe's new per-arm checkpointing are both consequences of it;
* [`logs/probe_logit_read_stability_full.txt`](logs/probe_logit_read_stability_full.txt) — the full-model
  run of the read-stability probe, whose artifact is `logit_read_stability_full_model.json`; the
  reduced-target run's log is `probe_logit_read_stability.txt`.

**Everything in `logs/`, `batch1/`, `batch32/` and `async/` that is a console log is committed gzipped.**
The repo's `.gitignore` has `*.log` at line 7, so a `server.log` copied into the evidence tree is silently
*not* in the commit — which is how the first commit of this stage ended up citing five server and sampling
logs it did not contain. Two probe logs are also over the repo's 500 KB per-file pre-commit limit
uncompressed. Both problems have the same fix, and `gzip` output is what the links point at.

Server logs follow the same rule, one per server, and there are six: the three the fallback audit reads
(§12) plus [`batch1/server_max_num_seqs_1.log.gz`](batch1/server_max_num_seqs_1.log.gz) (the benchmark-only
server behind README §1's earlier cold/warm pair) and two gzipped ones kept for completeness —
[`batch1/server_max_num_seqs_1_requests_probe.log.gz`](batch1/server_max_num_seqs_1_requests_probe.log.gz)
(the server that re-ran the request-shape probe after the `ignore_eos` fix, §7.2) and the two
`async/server_*_history_control.log.gz` files (the seeded-history controls of §7.6.1).

`logs/pytest_final_sweep.txt.gz` and `logs/pytest_generator_vllm.txt` are the two test-suite logs (§7.10).

### 7.10 Tests
```
pytest models/autoports/ornith_ai_ornith_1_0_35b/tests/test_generator_vllm.py -q   # 20 passed in 175 s
pytest models/autoports/ornith_ai_ornith_1_0_35b/tests/test_full_model.py -q -m "not long"   # 50 passed, 5 deselected
```
Console logs: the adapter suite's own run on the committed tree is
[`logs/pytest_generator_vllm.txt`](logs/pytest_generator_vllm.txt) — **20 passed** (9 host-only cases and 11
on the reduced two-layer target), `PYTEST_EXIT=0` at the end of the file; the full-model regression run is in
[`logs/pytest_final_sweep.txt.gz`](logs/pytest_final_sweep.txt.gz).

The suite's first attempt after the round-3 fixes errored 11 device cases on the ethernet 29-25 timeout
(§10.1's recoverable fault, this time left behind by the final server), recovered with the bounded
reset + mesh smoke, and passed on the retry. That is also why this log is the *second* run: round 3 noticed
the previous log's loguru line numbers no longer matched the committed `tt/generator_vllm.py`.

The second command is the regression check for this stage's generator/model additions: the full-model
stage's own suite, unchanged, on the same reduced target it uses. It was **not** re-run after the last
round of changes, and deliberately: `git diff` over `tt/` since that run touches only
`tt/generator_vllm.py` (the `atexit` capability dump and the foreign-checkpoint warning), which nothing in
`test_full_model.py` imports. The adapter suite, which does exercise it, was re-run and is the log above.

---

## 8. The reproducibility class of failures, measured

`$vllm-integration` asks for logit-level evidence when determinism tests fail, standalone and through
vLLM, before classifying anything. That investigation is the largest single piece of work in this stage,
and its first answer was wrong — §9 is how that was found and fixed. What follows is the corrected
measurement.

### 8.1 Through vLLM
Repeated identical greedy requests, `"The capital of France is"`, temperature 0, read out of the committed
artifacts (an earlier version of this table quoted a 4-of-4 arm that no committed file carries, and a
batch-32 pair of phrases stitched from two different files — round 4 of the review caught both):

| server | arm | result |
|---|---|---|
| `max_num_seqs=1`, overlap on | 3 greedy repeats, 24 tokens ([`serving_requests.json`](serving_requests.json)) | **all 3 identical** |
| `max_num_seqs=1`, overlap on | 3 greedy + 2 seeded repeats, 32 tokens, on a fresh server and again after ~90 requests ([`async/overlap_texts_async_fresh.json`](async/overlap_texts_async_fresh.json), [`…_after_traffic.json`](async/overlap_texts_async_after_traffic.json)) | **identical in every arm** |
| `max_num_seqs=1`, overlap **off** | the same two arms ([`async/overlap_texts_no_async_fresh.json`](async/overlap_texts_no_async_fresh.json), [`…_after_traffic.json`](async/overlap_texts_no_async_after_traffic.json)) | **identical in every arm**, and the greedy text is identical to the overlapped server's |
| `max_num_seqs=32`, overlap on | 3 greedy repeats, 24 tokens ([`batch32/serving_requests_max_num_seqs_32.json`](batch32/serving_requests_max_num_seqs_32.json)) | **3 distinct texts.** All three open `" Paris, a city that has "` — 5 words, 24 characters — and then split: `"…served as the country's political and cultural heart since the Middle Ages"`, `"…been the country's capital since the 10th century"`, `"…served as the nation's political and cultural heart since the Middle Ages"` |
| `max_num_seqs=32`, overlap on, a second server | the same arm ([`async/serving_requests_async_max_num_seqs_32.json`](async/serving_requests_async_max_num_seqs_32.json)) | 3 texts, 2 distinct — the same shape of split |

Every one of those completions is fluent, on topic and in English: what changes is which of several equally
good continuations the model takes when a near-tie is decided differently. A flip, not corruption. §8.2
measures the same thing at the logits.

### 8.2 Standalone, at the logits
[`probe_slot_reproducibility.py`](logs/probe_slot_reproducibility.py) →
[`slot_reproducibility.json`](slot_reproducibility.json), full 40-layer model, greedy, 13 logit vectors
per arm (the prefill's plus 12 traced decode steps), every call passing the caller-owned cache:

The deviation columns are *per step, while the two streams still agree* — up to but not including the flip.
Once two greedy streams emit different tokens they are different text; and the step *at* a flip sometimes
jumps by ~10 logits, which is a separate phenomenon (below) rather than the cause of the flip. The artifact
carries every step, so both readings are available.

| arm, two identical runs from a wiped state | bit-identical | max abs Δ per step, pre-flip | min PCC there | top-1 agreement | first flip (margin there) |
|---|---|---|---|---|---|
| **batch 1**, same prompt twice | **yes, all 13** | **0.0** | **1.0** | 13/13 | none |
| **batch 2**, same prompt twice | **yes, all 13** | **0.0** | **1.0** | 13/13 | none |
| **batch 4**, same prompt twice | **yes, all 13** | **0.0** | **1.0** | 13/13 | none |
| batch 8, same prompt twice | no | 0.78 | 0.99757 | 9/13 | step 9 (0.0625) |
| batch 16, same prompt twice | no | 0.84 | 0.99761 | 10/13 | step 10 (0.0625) |
| batch 32, same slot, runs 1 vs 2 | no | 0.72 | 0.99794 | 10/13 | step 10 (0.0625) |
| batch 32, same slot, runs 2 vs 3 | no | 0.81 | 0.99783 | 12/13 | step 12 (0.125) |
| batch 32, all 32 rows occupied, watching slot 0 | no | 0.84 | 0.99831 | 10/13 | step 10 (**0.0**) |
| batch 32, an occupied ghost row beside it | no | 0.70 | 0.99817 | 10/13 | step 10 (0.0625) |
| batch 32, the same prompt in slot 7 | no | 1.25 | 0.99356 | 13/13 | none |
| batch 1 against batch 32 | no | 0.91 | 0.99730 | 10/13 | step 10 (**0.0**) |
| batch 32, one step's device logits composed to host twice | **yes** | **0.0** | **1.0** | — | none |

Four things follow, and the batch sweep is what makes them sharp:

* **Bit-reproducibility survives to batch 4 and is lost at batch 8.** Batch 1, 2 and 4 are bit-identical
  run to run — prefill and all 12 traced decode steps, PCC 1.0, max |Δ| exactly 0.0. Batch 8, 16 and 32
  are not. The boundary is between 4 and 8, and it is reproducible: an independent five-pair count on the
  reduced target puts batch 4 at 0/5 deviating pairs and batch 8 at 4/5 (§8.3).
* **The prefill and the state it writes are deterministic at every batch size.** The probe compares all
  **140** DeltaNet recurrent/conv buffers and the written KV pages after prefill: bit-identical in every
  round at batch 1 and at batch 32 (`finite_max_abs_diff` 0.0, `rows_that_differ` empty, nothing non-finite), and the prefill logits
  are identical too (three rounds at batch 32). So the prefill, the per-slot merge, the cache fill and the
  staged inputs are all exonerated; what deviates is the decode step.
* **The readback path is exonerated too.** `batch32_read_twice` composes one step's device logits to host
  twice: bit-identical. A deviation that appeared only on the second composition would have been a read
  bug rather than a compute one.
* **Every flip sits at a margin the noise can cover.** The pre-flip deviation is 0.70–0.84 of a logit and
  every first flip in the table happens at a top-1/top-2 margin of **0.0 to 0.125** — an exact tie (two
  tokens holding the same bfloat16 value) or one to two bfloat16 steps at that magnitude. Nothing flips
  where the margin is comfortable: the `slot 7` arm deviates by up to 1.25 across all 13 vectors and never
  changes a token, and so does `batch32_same_slot_second_pair` until step 12. After a flip the per-step
  numbers jump to 8–16, which is simply two different texts being compared and says nothing about the
  cause — that is why the table's deviation columns stop before the flip.

### 8.3 Where it enters, measured: the multi-device collectives
[`probe_decode_nondeterminism.py`](logs/probe_decode_nondeterminism.py) exists to answer the next
question, because "the batch-32 decode step deviates" is not yet a mechanism. It runs the same
prefill-plus-decode sequence twice and compares, with one variable changed per arm, and it counts *pairs*
rather than trusting one comparison — the effect is intermittent, so a single non-deviating pair proves
nothing (a first attempt at a per-layer bisect learned that the hard way:
[`reduced_target/decode_nondeterminism_single_layer0.json`](reduced_target/decode_nondeterminism_single_layer0.json)
and [`…layer3.json`](reduced_target/decode_nondeterminism_single_layer3.json) each show one clean pair and
therefore exclude nothing).

On the reduced two-layer target (one `full_attention` + one `linear_attention`), context 2048, 4 decode
steps, five pairs of identical runs per arm:

| arm | driver | mesh | batch | pairs that deviated |
|---|---|---|---|---|
| traced replay, through the generator | generator | `1x4` | 32 | **5/5** |
| eager dispatch, through the generator | generator | `1x4` | 32 | **5/5** |
| eager dispatch, model level | model | `1x4` | 32 | **3/5** |
| eager dispatch, model level | model | **`1x1`** | 32 | **0/5** |
| eager dispatch, model level | model | `1x4` | 4 | **0/5** |
| eager dispatch, model level | model | `1x4` | 8 | **4/5** |

Artifacts: [`reduced_target/decode_nondeterminism_pairs_1x4.json`](reduced_target/decode_nondeterminism_pairs_1x4.json),
[`…_1x4_model_driver.json`](reduced_target/decode_nondeterminism_pairs_1x4_model_driver.json),
[`…_1x1.json`](reduced_target/decode_nondeterminism_pairs_1x1.json),
[`…_1x4_batch4.json`](reduced_target/decode_nondeterminism_pairs_1x4_batch4.json),
[`…_1x4_batch8.json`](reduced_target/decode_nondeterminism_pairs_1x4_batch8.json). The full-model arms
(traced, eager and a one-step arm that also compares the state the first decode step *writes*) are
[`decode_nondeterminism.json`](decode_nondeterminism.json).

What each row buys:

* **Not the trace.** The eager path — every op dispatched individually, no captured graph — deviates as
  often as trace replay. A trace-replay artefact (a stale kernel binary, an input buffer read before it is
  written) would not survive that arm.
* **Not the driver.** The `1x1` control cannot build a generator at all, because this model's device
  sampler is the multi-device vocabulary-shard path and its constructor refuses a single device. So the
  comparison is run *twice* on `1x4`: once through the generator and once through the same model-level
  driver the control uses. Both deviate. The mesh is the variable, not the driver.
* **The collectives.** On one device — no all-gather, no reduce-scatter, no MoE traffic over the fabric,
  same weights, same driver, same batch 32 — five pairs are bit-identical. On the `1x4` ring, three of five
  deviate. That is what names the mechanism: **run-to-run variation enters through the multi-device
  collectives**, not through the local matmuls, the sampler, the trace or the read path.
* **The batch threshold is a property of those collectives, not of the driver.** With the driver and the
  mesh held fixed, batch 4 gives 0/5 and batch 8 gives 4/5 — the same boundary the full-model sweep in
  §8.2 shows. A plausible reading is a payload that fits one fabric transfer at small batch and splits at
  batch ≥ 8, so the per-chunk accumulation order stops being fixed; the tuned router packet size is 8192 B
  (§3). This stage did not instrument the fabric, so that mapping stays a hypothesis — what is *measured*
  is that the collectives are where it enters and that the boundary sits between batch 4 and batch 8.
* **Where it first appears inside a request.** Index 0 of every arm is the prefill and it is bit-identical
  everywhere; index 1 is the *first* decode step, and at batch ≥ 8 it already deviates. Nothing has to
  accumulate: the very first multi-row decode step is already not reproducible.

On the **full 40-layer model** the same arms run at batch 4 and batch 32
([`decode_nondeterminism.json`](decode_nondeterminism.json), 4 decode steps per run):

| arm | batch 4 | batch 32 |
|---|---|---|
| traced replay, two runs | **bit-identical**, max abs Δ 0.0 | differs, max abs Δ **0.6875** over 4 steps, min PCC 0.99811 |
| eager dispatch, two runs | **bit-identical**, max abs Δ 0.0 | differs, max abs Δ **0.625**, min PCC 0.99866 |
| prefill logits, two runs | identical | identical |
| state after prefill (140 buffers + the written KV pages) | **bit-identical**, all finite | **bit-identical**, all finite |
| the first decode step's logits | **bit-identical** | differs, max abs Δ **0.671875** |
| the state that first decode step writes | **bit-identical**, all finite | **differs**, the active row included |

The last row carries a second observation worth recording on its own. At batch 32 with one active row, the
31 rows no prefill wrote are still carried through every batched op — the recurrent update is a dense
matmul over all rows — and, fed token 0 at position -1, their recurrent state runs away to float32
saturation, with a few entries going non-finite (1.8e-6 of entries in one run against 1.2e-7 in the other,
in rows 18 and 29, *and not in the same places*). Three things about that:

* **it does not cause the deviation under investigation.** The all-rows arm of §8.2 — all 32 rows holding a
  real prompt, so no runaway rows at all — deviates just the same;
* **it does not corrupt served output.** These ops are row-independent, the active row's logits stay finite
  and fluent, and the degenerate-output gate passes on both served configurations. vLLM also never decodes
  a row it never prefilled: the padding rows exist because the graph is a fixed 32-row batch, not because
  any request reads them;
* **it is a latent hazard all the same**, and it belongs with the defect in §8.5 — a non-finite value in a
  row that *is* read would not stay local, and it is why `compare_state` now separates non-finite entries
  from real differences (the first version of this measurement reported an *infinite* state difference,
  which was this garbage rather than a real deviation).

### 8.4 What the datatype sweep's §9.1 does and does not say
[`doc/datatype_sweep/README.md` §9.1](../datatype_sweep/README.md#91-one-of-four-identical-decode-slots-emits-a-different-token)
is the nearest earlier measurement, and it is a **control that holds** rather than corroboration — an
earlier draft of this section cited it the wrong way round, which the stage review caught.

What §9.1 measured: four slots *inside one run* at batch 4, on this exact precision policy — cross-slot
max |Δ| 0.28 – 0.5, logit PCC ≥ 0.9993, top-1/top-2 margins 0.0 – 0.19, one slot picking a different token
at an exact tie. Its conclusion about repeats is explicit: "It is fully deterministic — the same slot flips
on all three repeats — so this is a fixed reduction order, not nondeterminism."

That is a different axis from the one that fails here, and the two agree:

| | axis | batch 4 | batch ≥ 8 |
|---|---|---|---|
| datatype sweep §9.1 | slot against slot, within one run | differs (0.28–0.5), **stable across repeats** | not measured |
| this stage §8.2 | run against run, same slot | **bit-identical** | differs, 0.70–0.84 per step before the first flip |

So §9.1's determinism claim is confirmed here at batch 4, and the property that breaks above batch 4 is
run-to-run reproducibility, which §9.1 never measured. Its cross-slot finding also still holds at batch 32
in this stage's own data — the `slot 7` arm deviates from the `slot 0` arm by up to **1.25** logits (§8.2's
table) while never flipping a token.

### 8.5 What is and is not fixable here, and the defect this stage hands on
Not fixable at this layer, and not attempted: making the batch ≥ 8 decode bit-reproducible means changing
the collectives or the decode geometry, which is decoder-stage work with its own accuracy and performance
gates. Nothing at the serving layer can substitute — in particular a persistent per-slot page-table row
cannot, because per-call allocation is already bit-identical and lands at the same device address every
round ([`prefill_alloc_vs_recapture.json`](prefill_alloc_vs_recapture.json)).

**Open defect, named and handed on:** *multi-device decode is not run-to-run bit-reproducible at batch ≥ 8*
— it enters through the 1x4 fabric collectives (§8.3), the boundary is between batch 4 and batch 8, the
per-step magnitude is **0.70–0.84** of a logit before the first flip on the full model at 12 steps
(§8.2) and **0.47–0.69** over the shorter 4-step arms of §8.3, at PCC ≥ 0.997 throughout, and it flips a
greedy token only where the top-1/top-2 margin is inside that noise (every observed first flip sits at
0.0–0.125). Reproducer:
`probe_decode_nondeterminism.py --pairs 5 --mesh 1,4 --batch 8 --force-model-driver` against
`--mesh 1,1` as the control, on the two-layer target, which needs neither vLLM nor the full weight load.
The two things a decoder-stage fix would need to settle, which this stage deliberately did not: **which
collective** (a per-op comparison under the same pair-count method — the ops that run per decode step are
the TP all-gather/reduce-scatter and the EP expert dispatch/combine), and whether the padding rows'
recurrent state should be clamped or masked rather than left to run away to float32 saturation (§8.3) —
harmless where it sits today, and one shared reduction away from not being.

What that means for a deployment is in [README §8](README.md#8-limitations): serve at `--max-num-seqs 1`
(or up to 4) for bit-reproducible completions, and expect near-tie variation above it. Quality is
unaffected — every arm stays fluent and on topic, and the degenerate-output check passes on both served
configurations.

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
landed at the *same* device address in all four rounds — 173233536 for the per-call arm, 142904448 for the
long-lived one, `distinct_page_row_addresses: 1` in every arm — so "the allocator layout differs round to
round" was simply false.

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

### 9.1 The same class again: two probes were comparing the tile padding
Round 2 of the stage review asked why
[`prefill_stability_with_traces.json`](prefill_stability_with_traces.json) reported `top1 = 0` on both
sides of every comparison. Adding the non-degeneracy fields answered it, and the answer was worse than a
cosmetic oddity: **the rows being compared were all zeros** — `row_min = row_max = row_mean = 0.0`,
`row_nonzero_fraction = 0.0`, every top-5 value `0.0`. "Bit-identical" over two zero vectors is not a
measurement.

The cause is one index. On the `return_logits="device"` path, `prefill_forward_single` runs the LM head
over a **single** row (`_sampler_rows(last, 1)`) and the result is tile-padded to 32 rows, which is why the
model's own host path returns `[:, :1, :]`. Both
[`probe_prefill_stability_with_traces.py`](logs/probe_prefill_stability_with_traces.py) and
[`probe_logit_read_stability.py`](logs/probe_logit_read_stability.py) read `_logits_to_host(...)[0, -1]` —
row 31, the padding.

Fixed to `[0, 0]` in both, with the compared row's own shape recorded in every comparison so it cannot
recur silently, and both probes re-run on the full model. The conclusions survive on real rows:

| artifact | rows non-degenerate | conclusion |
|---|---|---|
| [`prefill_stability_with_traces.json`](prefill_stability_with_traces.json) | yes — 99.999 % of entries nonzero | repeated prefills are bit-identical, eager and with a live captured decode trace (max abs Δ 0.0) |
| [`logit_read_stability.json`](logit_read_stability.json) (reduced target) | yes, `1x4` and `1x1` | the read path, repeated forwards, and forwards after a decode step are all bit-stable on both meshes |
| [`logit_read_stability_full_model.json`](logit_read_stability_full_model.json) | yes, `1x4` | the same on the full 40-layer model; the `1x1` fields stay `null` because it does not fit on one device |

This is the **third** measurement in this stage that read the wrong tensor — the page table of zeros above,
and now the tile padding in two probes — and all three had the same tell: a suspiciously clean result that
nothing else corroborated. The countermeasure that works is the one now in every comparison helper here:
record what the compared data *was*, not only how it compared. `all([])` and `torch.equal(zeros, zeros)`
are both true, and neither means anything.

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
  used (`_hold_until_signal` → `terminate`), then the process table was checked, then the reset script ran.
* One reading note on the committed server log: its loguru lines name
  `generator_vllm:prefill_forward:670`, while the committed adapter has that call at 684. The 14-line
  difference is the round-3 reduced-build guard (§7.8), added after the last server ran. Nothing in the log
  is stale in *content* — a server cannot be re-run without replacing the artifact set it is the attribution
  anchor for (§7.7) — but the line numbers are from the pre-guard file. The **pytest** log is from after the
  guard and its line numbers match the committed adapter exactly.
* The **final** state is [`logs/final_device_reset_and_mesh_smoke.txt`](logs/final_device_reset_and_mesh_smoke.txt),
  captured after the last device job of the stage (the adapter suite, itself the last thing to touch a
  device after the last server): no device-owning process, 8 board lines before and after a `tt-smi -r`, and
  `MESH_SMOKE_OK`. Round 3 of the review caught that this file was three hours older than the last server —
  it recorded an *intermediate* cleanup, not the final one.
* Liveness waits in this stage never used a `pgrep -f <pattern>` that could match the checking shell —
  the trap `$tt-device-usage` warns about. Waits keyed on the launched PID (`kill -0`), on an artifact
  appearing, or on `/health` returning 200. The final-state check above is keyed on the *executable* for the
  same reason, and it excludes the bringup orchestrator, whose own command line names the vLLM stage
  prompts and would otherwise look like a device holder forever.
* No Tracy, `tt-perf-report`, `TT_METAL_DEVICE_PROFILER` or `ttnn.ReadDeviceProfiler` run was made
  around any serving job, per the skill's prohibition. The device-op-level evidence for this decode stack
  is the optimized-full-model and datatype-sweep stages' own non-serving profiles.

---

## 12. Runtime-fallback audit

What the served logs say about paths that would silently cost correctness or speed. Each column is one
**preserved server log**, and the traffic each one served is named, because a count only means something
next to the requests that produced it:

| | batch-1, default flags | batch-32, default flags | batch-1, `--no-async-scheduling` |
|---|---|---|---|
| log | [`batch1/server_max_num_seqs_1_all_checks.log.gz`](batch1/server_max_num_seqs_1_all_checks.log.gz) | [`batch32/server_max_num_seqs_32.log.gz`](batch32/server_max_num_seqs_32.log.gz) | [`async/server_no_async_max_num_seqs_1.log.gz`](async/server_no_async_max_num_seqs_1.log.gz) |
| traffic it served | the full sampling suite, qualitative, the chat probe, the request-shape probe, the overlap arm, two primary benchmarks | the full sampling suite, qualitative, the request-shape probe, single-user and CI-burst benchmarks | the overlap control arm, the request-shape probe, the smoke sampling profile, qualitative, two primary benchmarks |
| `POST /v1/completions` + `/v1/chat/completions` | 518 + 13 | 1565 + 38 | 103 + 1 |
| adapter `prefill_calls` (one below the request count: the `logprobs=-1` chat request is rejected `400` by the plugin's clamp and never prefills — it is the case that skips `test_chat_logprobs_all_vocab`) | 530 | — | 103 |
| **a foreign page table without a caller-owned cache** (`"was given a page_table but no kv_cache"`) | **0** | **0** | **0** |
| **a program compiled inside the traced decode loop** (`"a program was compiled inside the traced decode loop"`) | **0** | **0** | **0** |
| **KV cache falling back to a default dtype** (`"No dtype specified for the model KV cache"`) | **0** | **0** | **0** |
| **async scheduling silently disabled** (`"Disabling async scheduling"`) | **0** | **0** | **0** |
| **the sampler taking the force-argmax path** (`"Forcing argmax sampling"`) | **0** | **0** | **0** |
| **a reduced serving target** (`"REDUCED serving target"`) | **0** | **0** | **0** |
| trace re-captures (`"re-capturing the decode traces"`) — the designed mechanism, not a fallback | 28 | 26 | 19 |

This table has been wrong twice, in the same way, and both times the review caught it. Round 1: the batch-1
column quoted counts from a server whose log had been overwritten (`server.log` is truncated per launch),
while the log actually archived was a *benchmark-only* server with 2 POSTs. Round 2: the re-derived counts
were correct for the log they were taken from, but a later re-run of the batch-1 set replaced that log, and
the numbers were not re-derived against the new one. The counts above are read from the three logs this
commit contains, and the re-capture count is cross-checked against the capability report's own
`trace_recaptures`. The benchmark-only log is still kept, as
[`batch1/server_max_num_seqs_1.log.gz`](batch1/server_max_num_seqs_1.log.gz), because it is the log behind
the earlier cold/warm benchmark pair.

The re-captures are the designed mechanism: a prompt length the process has not seen compiles programs
whose kernel binaries were allocated while the traces were live, and `_ensure_traces_replay_safe`
re-captures before the next replay can overwrite them. The count tracks the number of *distinct* prompt
lengths a server saw rather than its request count — 531 requests and 28 re-captures on one server, 1603 and
26 on another, 104 and 19 on the third — and
`test_traces_are_recaptured_when_a_new_program_is_compiled` pins that a repeat length re-captures nothing.
The archived capability report agrees independently: `trace_recaptures` in
[`batch1/vllm_serving_capability_final_max_num_seqs_1.json`](batch1/vllm_serving_capability_final_max_num_seqs_1.json)
is the same 28.

**The adapter's own counters, after real traffic.** `readiness_vllm/vllm_serving_capability_final.json`
(§7.8) is written at engine-core exit, so it reports what the process did rather than what it was
configured to do:

| counter | batch-1, default flags ([artifact](batch1/vllm_serving_capability_final_max_num_seqs_1.json)) | batch-1, `--no-async-scheduling` ([artifact](async/vllm_serving_capability_final_no_async.json)) |
|---|---|---|
| `prefill_calls` / `decode_calls` | 530 / 10831 | 103 / 5051 |
| `device_sampled_decodes` / `host_sampled_decodes` | 9969 / 862 | 5042 / 9 |
| `no_refresh_steps` (the steady state copies nothing) | **9431** of 10831 | **4891** of 5051 |
| `full_refreshes` / `page_table_only_refreshes` | 1345 / 55 | 110 / 50 |
| `slot_remaps` | 0 | 0 |
| `async_reads` | **10831** | **0** |

Three things that table settles that a config dump cannot: the overwhelming majority of served decode
steps copied **nothing** to the device — 9431 of 10831 (**87 %**) on the overlapped server and 4891 of 5051
(**97 %**) on the non-overlapped one, the difference being how much prefill traffic each server interleaved
— so the refresh policy's steady state is the common case, not a special case; the host sampler was entered 862 times on the server that ran the log-probs tests and 9
times on the one that did not, so that fallback is request-driven rather than accidental; and `async_reads`
is exactly the decode-call count with overlap on and exactly **zero** with it off, which is the mechanical
proof of which path each server took (§7.6). `slot_remaps` stays 0 because `max_num_seqs=1` has no second row to condense into — the remap
path's evidence is the primitive-level arm in §7.1 and the two adapter tests, not this counter.

The zero force-argmax count is also the intended state, and it is a *measured* decision rather than an
oversight: the full-model stage rejected `allow_force_argmax` because it would all-gather the whole
249856-wide logits row and run a global `ttnn.argmax` — "precisely the sampler-op-dominates-decode shape
the goal forbids" ([`doc/full_model/README.md` §4.2](../full_model/README.md),
[`doc/optimized_full_model/README.md` §9](../optimized_full_model/README.md)). Greedy therefore runs the
same captured split-sampling graph as every other mode (local top-32 per vocabulary shard, gather 4×32
candidates, `ttnn.sampling` with `k=1, p=0, temp=1`), which is the fastest strategy measured for this
mesh. Serving inherits it unchanged: the serving benchmark's 23.140 ms ITL against the model's own
23.165 ms token-out figure is the proof that no other sampling path crept in.

**One log signature that looks alarming and is not.** The `max_num_seqs=32` server's log carries **480**
`critical` lines — 180 `TT_THROW … Statically allocated circular buffers … beyond max L1 size` and 300
`TT_FATAL: Out of Memory … L1 buffer` — grouped as **16 per layer**, each group closed by one
`layer N batch 32: ttnn.conv1d accepted 0/16 prefill block lengths []` summary (30 of those, one per
`linear_attention` layer). That is `allocate_state`'s own capability probe: it *tries* all 16 prefill block
lengths through `ttnn.conv1d` at that batch size, catches the ones that do not fit L1, and records which
lengths the conv path can serve. At batch 32 none can, so the DeltaNet conv falls back to the FIR path — and
prefill runs at batch 1 anyway, where the same probe logs `accepted 16/16` for all 30 layers in every server
log. The full-model stage classified this signature already
([`doc/full_model/work_log.md`](../full_model/work_log.md), `doc/full_model/README.md`); it is listed here
so a reader of the batch-32 log does not have to re-discover it.

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

The context gate also prints **advisory** lines on stderr — 33 of them on the last run — and they are
worth naming so nobody reads them as a served cap. Exactly:

* **30 lines, one per model build in a probe or test console log** (the number moved from 34 when two probe
  logs were gzipped, which the gate does not read, and two probes were re-run)**.** Every build logs
  `building OrnithModel: … max_context=N` at startup, and a run that only needs a 2048-, 4096- or
  8192-token window says so. This is one line per *build*, not per file:
  `probe_logit_read_stability.txt` contributes two (it builds the reduced target at `tp=4` and `tp=1`),
  and `logs/pytest_generator_vllm.txt` contributes eleven, one per device test. The count moves whenever a
  probe is added, re-run, or gzipped, which is why the breakdown matters more than the total;
* **1 line from `tt/functional_decoder.py:239`**, a pre-existing comment that uses `max_context=8000`
  as an illustration of a prefill block ending past `max_context`;
* **2 self-referential lines from this section**, because the text above quotes `max_context=8000` twice
  and the gate greps text, not meaning.

Advisories are not failures (the gate returns 2 only for a JSON
*key* below the supported context, and `--strict-caps` is not used by the stage gate), and nothing in
`tt/`, `readiness_vllm/` or this directory's JSON caps the served window: `doc/context_contract.json`'s new
`vllm_integration` block records 262144 served against 262144 advertised, and
`readiness_vllm/vllm_serving_capability.json` shows the built model carrying it.

---

## 14. Files changed

In this repo:

| file | change |
|---|---|
| `tt/generator_vllm.py` | **new.** The vLLM adapter: `TTQwen3_5MoeForConditionalGeneration`. Includes the `atexit` capability dump (§7.8) and the warning that fires when a checkpoint other than this one resolves to this class (§3) |
| `tt/generator.py` | one new `serving (vLLM) API` section (the seven primitives of §4) plus four small changes elsewhere: `_sample_traced` passes `skip_precompile=True`; the constructor allocates the prefill sampling scratch buffer (before any capture) and a `sampling_trace_captures` counter; `submit_serving_decode` calls the replay-safety check itself; and `_resolve_page_table` substitutes only when the generator owns its cache (§9) |
| `tt/model.py` | **+57 lines**: `remap_state_slots` and its `_remap_rows` helper |
| `tests/test_generator_vllm.py` | **new.** 9 host-only cases (registration, the flags the plugin reads, the interface vLLM introspects, the shared adapter contract, no sampling path of its own, the token-pool bound, the log-probs refusal reading rows rather than the container, visual-payload refusal, a reduced build not overwriting the served capability report) + 11 device cases on the reduced target (cache ownership, block-size refusal, per-slot prefill, steady-state refresh, stale-pair merge, page-table-only refresh, slot remap and its identity skip, host-sampling logits, the precision-config propagation, the capability report) |
| `models/common/readiness_check/run_vllm_server.py` | **+65 / -4 lines**: `_tt_config_flag()` picks `--additional-config` / `--plugin-config` from the installed engine, and `_mesh_device()` accepts a mesh name or an explicit `(rows, cols)` grid. Both are fixes against the current vLLM fork, not model-specific |
| `doc/vllm_integration/**` | **new.** This log, the README, ten probes with the console log of their final run, the evidence JSON, the reduced-target localisation set (`reduced_target/`), and the archived per-configuration artifact sets (`batch1/`, `batch32/`, `async/`) |
| `doc/context_contract.json` | **+1 block**: `vllm_integration`, recording 262144 served against 262144 advertised, the KV-pool sizing and its cost, the non-aligned-length evidence, the 64-token block size, and the tested batch coverage |
| `readiness_vllm/**` | **new.** The shared runner's artifacts, last-writer-wins per launch, with every console log committed **gzipped** because the repo's `.gitignore` excludes `*.log`: `server.log.gz`, `sampling_tests.log.gz`, `vllm_qualitative_outputs.json`, `vllm_result.json`, `vllm_benchmark.json`, `vllm_benchmark.log.gz`, `vllm_ci_serving_result.json`, `vllm_ci_serving_benchmark.json`, `vllm_ci_serving_benchmark.log.gz`, and `vllm_serving_capability.json` plus `vllm_serving_capability_final.json` (written by the adapter at the end of warm-up and again at engine-core exit, from inside that process). What is committed there is the headline single-user server's output; the per-configuration archives under `doc/vllm_integration/{batch1,batch32,async}/` are the attributable copies |

Nothing in `tt/model.py`, `tt/optimized_decoder.py`, `tt/multichip_decoder.py` or `tt/precision_config.py`
changed behaviour for the earlier stages: the only model addition is a new method nothing else calls, and
the generator's existing entry points are untouched apart from `skip_precompile=True` on the lazy
sampling-trace capture (which is strictly safer) and one new buffer allocated at construction.

Outside this repo, in the `tenstorrent/vllm` checkout (kept here as
[`vllm_tt_plugin_changes.diff`](vllm_tt_plugin_changes.diff), not committed to tt-metal):

| file | change |
|---|---|
| `plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py` | register this adapter for `TTQwen3_5MoeForConditionalGeneration` and, replacing upstream's class, for `Qwen3_5MoeForConditionalGeneration`, with a comment recording that the override is architecture-wide and where a future collision would have to be resolved (§3) |
| `plugins/vllm-tt-plugin/src/vllm_tt_plugin/worker.py` | `get_fabric_router_config()` + `set_fabric` honouring a `fabric_router_max_packet_bytes` TT-config key, because fabric configuration has to happen before the mesh is opened (§3) |

---

## 15. Stage review and commits

### Review rounds, and what they changed

`$stage-review` returned `more-work-needed` four times. All four rounds' findings are listed below with the
measurement or correction each one produced, because several of them changed published numbers.

**Round 1.**

| finding | what it turned into |
|---|---|
| **P1** — §8's mechanism claim ("a fixed reduction order cannot produce run-to-run variance") was unsupported, and it cited datatype-sweep §9.1, which concluded the opposite | three new probe arms plus a new probe (`probe_decode_nondeterminism.py`): the batch boundary is measured (bit-identical at 1/2/4, not at 8/16/32), four candidate causes are each ruled out by their own arm, and a 1x1 control names the collectives. §8.3–§8.5 are the result; §8.4 corrects the §9.1 citation to what it actually says |
| **P2** — the adapter suite had not been run on the committed tree, and no console log existed | re-run: 19 passed, [`logs/pytest_generator_vllm.txt`](logs/pytest_generator_vllm.txt) (§7.10) |
| **P2** — §12's fallback audit quoted counts no preserved log supported | the whole batch-1 set re-run on one server whose log is archived before shutdown, plus a third column for the no-overlap server, plus the traffic named per column (§12) |
| **P2** — `serving_requests.json` was unlabelled and `all_completed` was false | the probe now requires `--server-label`, records `finish_reason`, and asks each length with and without `ignore_eos` (§7.2). `all_completed` is now true, and the short plain completions are shown to be end-of-text stops |
| `overlap_equivalence.json` had no producer script and compared unequal lengths | [`probe_overlap_equivalence.py`](logs/probe_overlap_equivalence.py). Running it surfaced the bigger problem: this vLLM enables async scheduling by default, so both of that artifact's arms were overlapped. The artifact is withdrawn and the real control was run (§7.6) |
| `logit_read_stability_full_model.json` reported `1x1` stability from an arm that never ran (`all([])`) | the probe reports `null` for a skipped arm, and the artifact's summary was recomputed from its own recorded arms |
| `prefill_stability_with_traces.json` compared rows without showing they were non-degenerate | the probe records each row's own shape — which is how round 2 discovered the rows were **zeros** (§9.1) |
| the plain-architecture `ModelRegistry.register_model` override was unscoped and undocumented | scope, failure modes and collision point recorded at the line and in §3; the adapter warns when a foreign checkpoint resolves to it |
| probe/console-log bookkeeping (a wrong probe count, a missing log, a superseded one) | §7.9 names which log belongs to which run; §14 and README §7 have the corrected inventory |
| `numpy` was outside tt-metal's pin without the README saying so | disclosed in README §7 |
| `serving_counters` were all zero in the capability artifact | the `atexit` second copy, with the counters of served traffic (§7.8, §12) |
| `doc/context_contract.json` still said `"stage": "datatype-sweep"` at the top level | updated, with a note that per-stage blocks keep their own |
| §13's advisory count was wrong | recounted and broken down by source (§13) |

**Round 2.** Round 1's remediation introduced one new instance of the very failure it fixed, and the review
found it plus three more:

| finding | what it turned into |
|---|---|
| **P1** — `readiness_vllm/`'s benchmark JSON, raw result, server log and both capability reports belonged to the `--no-async-scheduling` control, because that cycle ran *after* the coherent batch-1 re-run, while README §1 attributed them to the headline server | the batch-1 set re-run once more, as the **last** server of the stage, and the cycle now ends by `md5sum`-comparing the runner's set against the archived copies and printing the server log's scheduling state, request count and re-capture count. All five files are byte-identical to their `batch1/` copies (§7.7, README §9) |
| **P2** — the serving-counter numbers in README §3/§5 and §12 existed in no committed artifact | recomputed from [`batch1/vllm_serving_capability_final_max_num_seqs_1.json`](batch1/vllm_serving_capability_final_max_num_seqs_1.json), which is now also the committed `readiness_vllm/` copy (§12) |
| **P2** — §12's batch-1 POST and re-capture counts did not match the log it cited | recounted from the three committed logs, and the re-capture count cross-checked against the capability report's own `trace_recaptures` (§12) |
| **P2** — §15 claimed the `prefill_stability_with_traces.json` finding was closed, but the artifact predated the fix | the probe was re-run — and its new fields showed the compared rows were **all zeros**, a tile-padding index bug in two probes. Both fixed and re-run; conclusions survive on real rows (§9.1) |
| **P3** — `batch1/serving_requests_max_num_seqs_1.json` was the pre-fix copy | replaced with the fixed run (byte-identical to the top-level file), and §7.2's attribution corrected |
| **P3** — README §4 described the wrong completion and claimed "no mechanical repetition" | the greedy Fibonacci completion's 4× loop is now described, quantified, and controlled against two other server configurations (README §4) |
| **P3** — `probe_overlap_equivalence.py` had no console log | captured on the final server: [`logs/probe_overlap_equivalence.txt`](logs/probe_overlap_equivalence.txt) |
| the review's carry-forward that the plugin overlaps *any* TT model regardless of the capability flag | checked against the code: the default is resolved before `check_and_update_config`, and the plugin does disable overlap for a model that does not declare `supports_async_decode`. Recorded with line references in §7.6, along with the one real rough edge (the warning says "requested" when it was defaulted) |
| the review's note that the batch-32 log's L1 `TT_THROW`/`TT_FATAL` lines (480 of them) are unexplained here | classified in §12 as `allocate_state`'s conv1d capability probe, already characterised by the full-model stage |

**Round 3.**

| finding | what it turned into |
|---|---|
| **P2** — README §1 and limitation 2 said the first-request cost was "entirely in the first token"; the artifact shows ~220 ms of it inside one inter-token interval | derived from the committed numbers (cold TPOT 24.894 against 23.174 warm with an unchanged ITL median, e2el − TTFT over 127 intervals) and tied to the log line that proves it (prefill returns at `18:55:38.314`, the re-capture runs at `18:55:38.494`), with the datatype sweep's `hidden_cost_ms` = 250.4 as the prior-stage control (§7.7, README §1, limitation 2) |
| **P2** — README §4's sampled column, its "every completion hits the cap" claim, its degeneracy ranges and one of its two loop controls described the *previous* qualitative generation | §4 rewritten from the committed artifacts: two sampled completions self-terminate (haiku and translation, with an empty `<think></think>`), the raw artifact carries no `finish_reason` so the cap claim is made only for the chat run, the batch-32 "control" is downgraded to what that file actually shows (three *different* self-quiz blocks, not a repeat), and the degeneracy figures are the gate's own over the ten completions it measures (dup 0.0000 – 0.0140, loop 0.0149 – 0.1304) with its console log now committed |
| **P2** — the documented adapter-suite command overwrote two committed `readiness_vllm/` capability artifacts with a reduced two-layer report | `_write_serving_capability` refuses to write for a reduced build, a regression test asserts the committed bytes survive it, and the suite was re-run: 20 passed, both artifacts `md5sum -c` clean (§7.8) |
| **P3** — `logs/final_device_reset_and_mesh_smoke.txt` predated the last three servers | re-captured after the last device job of the stage, with a process check keyed on the executable rather than the command line (§11) |
| the pytest log's line numbers no longer matched the committed adapter | closed by the re-run above |
| small unverifiable figures (`289 s` against the log's 292 s, `+73 lines` against 65/−4, the third `mixed_batch` failure's real `IndexError`) | corrected in §6, §7.5 and §14 |
| `async/async_max_num_seqs_32_sampling_tests.log.gz` was committed but cited nowhere | described in README §9's `async/` row |

**Round 4.** No new device work: every finding was a number or a description that did not re-derive from a
committed file.

| finding | what it turned into |
|---|---|
| **P2** — README §6 and §7.5 listed `test_topk[15]` and request isolation as passing in *both* configurations while the same sections listed them among the batch-32 failures | both lists rebuilt by extracting every per-test result from the two committed logs: 50 pass in both, 15 pass at 1 and fail at 32, 3 fail in both, 4 fail at 1 and pass at 32, 1 skips — which sums to 73 and reproduces both totals (§7.5, README §6) |
| **P2** — §8.1's through-vLLM row quoted a 4-of-4 greedy arm and a 3-of-3 seeded arm that no committed file carries, and a batch-32 pair of phrases stitched from two different files | §8.1 rebuilt from the committed arms: 3 greedy repeats identical at `max_num_seqs=1` (plus the six 3-greedy/2-seeded overlap arms), and the batch-32 split quoted from **one** file with its own three texts and their 24-character common prefix |
| **P3** — `test_logprobs` is 20 parameterisations, not 16 | corrected in README §6, limitation 7 and §7.5 |
| **P3** — §12 said "~360" L1 lines, "each followed by" a summary | recounted: 180 + 300 = 480, grouped 16 per layer with 30 summaries |
| **P3** — §12 generalised 87 % of no-refresh steps to "either server" | both figures given: 87 % overlapped, 97 % non-overlapped, with why they differ |
| **P3** — limitation 2 repeated a sentence; §15 said the review returned `more-work-needed` "twice" | fixed |
| **P3** — the "289 – 292 s" range still contained an uncited endpoint | only the committed run's 292.00 s is quoted; the two earlier repeats are named as prose-only |
| the CI serving-burst artifacts had no per-configuration archive copy | copied into `batch32/`, byte-identical, so every committed serving number is attributable to a configuration directory (README §9) |
| the committed server log's loguru line numbers predate the round-3 guard | noted in §11, with why the log cannot be re-run without breaking the attribution it anchors |

**Round 5.** Documentation again, and all of it inside the sections earlier rounds had not touched:

| finding | what it turned into |
|---|---|
| **P2** — §8.3's full-model table quoted an eager magnitude of 0.94 and a first-step magnitude of 0.72 that `decode_nondeterminism.json` does not carry, and §8.4/§8.5 stated the per-step range as 0.47–0.92 while §8.2 and README §6 said 0.70–0.84 | the table rebuilt from the artifact (traced 0.6875 / eager 0.625 / first step 0.671875, each with its min PCC), and the range in §8.4/§8.5 now names both arms and their lengths — 0.70–0.84 over §8.2's 12 steps, 0.47–0.69 over §8.3's 4-step arms — so no two sections disagree |
| **P3** — §8.4 said the `slot 7` arm deviates by up to 1.0 logit; the artifact and §8.2's own table say 1.25 | corrected to 1.25 with the pointer to §8.2 |
| **P3** — §9 quoted a per-call page-row address (132992448) no artifact carries | replaced with the recorded addresses (173233536 per-call, 142904448 long-lived) and `distinct_page_row_addresses: 1` |
| **P3** — README §1 and §7.7 read the `prefill_forward` log line as prefill *returning*, making the 180 ms look like a gap after the first token; the line is logged on prefill **entry** | both re-anchored to the call site that actually produces the re-capture — `submit_serving_decode`'s replay-safety check on the first decode step — which is what puts the stall in the first inter-token interval |
| **P3** — §15 said "three times" over four round tables; the commits table was out of chronological order and missing the current HEAD | fixed, and the table is now in `git log` order |
| **P3** — §7.8 enumerated the sweep artifact's prose-only keys wrongly (four `*_note` fields, two keys missing) | replaced with the nine actual extras |
| §12's request count (518 + 13) sat beside `prefill_calls: 530` with no explanation | the table now says why: the `logprobs=-1` chat request is rejected `400` by the plugin's clamp and never prefills — the same case that skips `test_chat_logprobs_all_vocab` |
| README's serving-status row asserted the batch-1 sampling result "reproduced exactly on a second server" without the caveat §7.5 carries | the row now says three servers saw it and only the last run's log is committed |

Three things no review asked for came out of doing all of the above, and all three changed published numbers
or claims: the async-scheduling default (§7.6), the headline benchmark's overwritten artifact (§7.7), and two
probes comparing tile padding (§9.1). Each is recorded where its numbers are, not only here.

### Commits

| repo | branch | commit | contents |
|---|---|---|---|
| `tt-metal` | `agentic-research/hous/ornith-1.0-35B` | `71f8bab9ff0` | the adapter, the generator/model serving primitives and the `_resolve_page_table` fix, the adapter suite, this stage's evidence, the `readiness_vllm/` artifacts, the context-contract block, and the two shared-runner fixes |
| `tt-metal` | same | `2e5675d669c` | review round 1 remediation: the localisation probe and the extended reproducibility arms, the no-overlap control and its artifacts, the re-run batch-1 evidence set, the `atexit` capability dump and the foreign-checkpoint warning, the corrected docs, and every console log gzipped so `.gitignore`'s `*.log` cannot silently drop it |
| `tt-metal` | same | `ef5e63409dc` | the round-1 review record in §15 |
| `tt-metal` | same | `a92706dbc4f` | review round 2 remediation: the re-run headline set with its attribution check, the tile-padding fix in two probes and their re-measured artifacts, the recomputed counters and audit counts, and the README/work-log corrections |
| `tt-metal` | same | `7168b79045d` | the round-2 review record in §15 |
| `tt-metal` | same | `5303a7a7a1d` | review round 3 remediation: the reduced-build guard on the capability writer with its regression test and the re-run suite, the corrected first-request cost accounting, README §4 rewritten from the committed qualitative artifacts, the two gate console logs, the re-captured final device state, and the small figure corrections |
| `tt-metal` | same | `0f2d3aa310c` | the round-3 review record in §15 |
| `tt-metal` | same | `b7347f123d1` | review round 4 remediation: the per-test pass/fail inventory rebuilt from the sampling logs, §8.1 rebuilt from the committed determinism arms, the recounted L1 and no-refresh figures, the `batch32/` CI-burst copies, and the smaller figure corrections |
| `tt-metal` | same | `b90a91e05b9` | the round-4 review record in §15 |
| `tt-metal` | same | *this round's commits* | review round 5 remediation: §8.3–§8.5's magnitudes rebuilt from `decode_nondeterminism.json` and `slot_reproducibility.json`, §9's page-row address corrected, the first-request-stall argument re-anchored to the call site that produces it, and §7.8/§12/§15's enumerations fixed |
| `vllm` (separate checkout, `tenstorrent/vllm@bf98d556` + these) | `dev` | `a8a5a4c` | the plugin registration and the fabric-router-config passthrough |
| `vllm` | same | `5380fd4` | the comment recording the architecture-override's scope |

No commit is pushed. The `tt-metal` commits contain only stage-owned paths; the two files that were
already dirty when the stage started (`.agents/skills/tt-device-usage/SKILL.md`, modified, and
`.agents/fast-models-fast-feedback.md`, untracked) were left untouched and out of them.
