# Serving plugin and vLLM changes

Pinned serving pair for this run (set by the run environment, not chosen by
this stage): the `tenstorrent/vllm` checkout at `/home/hous/dev/vllm`, commit
`f96662a7d239c077e1435b658c0e7d316f9c4799`, with its in-tree plugin
`plugins/vllm-tt-plugin` installed editable in `/home/hous/dev/tt-metal/python_env`
(`vllm 0.1.dev14202+gf96662a7d`). The checkout's working tree is clean at that
commit; this stage made no new change to the plugin or to vLLM.

The pinned commit is not an upstream release. Relative to the last upstream
commit of that fork (`5ffebf4128f`, 2026 Aug 27) it carries nine local commits
authored 2026 Sep 08 to Sep 23 that earlier stages of this bringup relied on.
Each is recorded below as the plugin-change rules require. None of these
records was written when the change was made; the records are reconstructed
from the commit messages, the diffs, and the plugin tests that ship with them.
Where a rule of `references/plugin-changes.md` is not met, the record says so.

Pre-existing records are labelled `implemented` with the commit SHA. The
per-item human authorization that the rules require for new shared interface
is not on record for any of them; the operator selected this exact checkout
as the pinned pair for this run (goal prompt, Environment section), which is
the only authorization evidence available. The stage treats that as
authorization of the pair, not of each item, and lists it as a residual risk.

## 1. `2f81f493b` feat(tt-plugin): add Gemma 4 serving support

1. **Requirement and failure.** Gemma 4 is not in vLLM's upstream model
   registry; without a registry entry the nested text/vision config resolves
   to `TransformersMultiModalForCausalLM` and the server crashes in the
   multimodal processor factory before the TT plugin's `TT` prefix logic
   runs. Reproducer: launch the server for `google/gemma-4-26B-A4B-it` on the
   base commit.
2. **Owner.** Case 2, missing shared behavior: the plugin owns registry
   resolution for TT architectures, and a registry entry for a model class is
   not something the model checkout can express. Parts of the same commit are
   not plugin-owned: the `TT_GEMMA4_TEXT_VER` environment switch selects
   between `models.demos.gemma4` and this autoport in plugin code, which is a
   model-specific branch in shared code and a built-in registration for an
   autoport adapter. The pinned plugin also offers `EXTRA_MODELS_DIR` bundles
   with `vllm_metadata.json`; that mechanism could have selected the adapter
   without plugin code. This stage keeps the pinned behavior (the
   environment fixes the pair) and records the deviation.
3. **Model-side alternative.** None for registry resolution. For adapter
   selection, an `EXTRA_MODELS_DIR` bundle naming `Gemma4ForCausalLM` and
   `models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm:Gemma4ForCausalLM`.
4. **Existing mechanisms.** `EXTRA_MODELS_DIR` (platform.py
   `_iter_extra_model_bundles`): applies to adapter selection, not to the
   missing upstream architecture entry. `TT_MODEL_CLASS_OVERRIDES` does not
   exist in this pinned plugin.
5. **Effect on other models.** Other models are unaffected except for the
   device-sampling eligibility changes in the same commit
   (`supports_device_penalties`, `supports_device_seeded_sampling`,
   `max_device_top_k`), which default to the previous behavior when a model
   declares nothing.
6. **Authorization.** Adds shared interface (three `model_capabilities`
   keys and one environment variable). Per-item authorization: not recorded.
   Pair authorization: goal prompt Environment section (operator).
7. **Validation.** Plugin host tests added in the commit:
   `tests/test_device_sampling_eligibility.py`, `tests/test_gemma4_reasoning_parser.py`,
   `tests/test_gemma4_tool_parser.py`, `tests/test_worker_parent_mesh.py`;
   device tests `tests/tt/test_host_only_params.py`, `tests/tt/test_tt_penalties.py`
   pass in `readiness_vllm/sampling_tests.log`.
8. **Status.** `implemented`, commit `2f81f493b969da7ce3cd64c0f6bc1895b8f229cb`.

## 2. `7b24b0e59` Fix TT worker mesh shutdown and require successful serving warmups

1. **Requirement and failure.** Worker shutdown left the parent mesh open and
   a failed `warmup_model` was ignored, so a broken warmup served requests.
2. **Owner.** Case 1, plugin defect in worker lifecycle.
3. **Model-side alternative.** None; the worker owns mesh lifetime.
4. **Existing mechanisms.** None apply.
5. **Effect on other models.** Every model: warmup failures now fail startup;
   shutdown closes submeshes before the parent.
6. **Authorization.** No shared interface added.
7. **Validation.** `tests/test_worker_parent_mesh.py` (extended in the commit).
8. **Status.** `implemented`, commit `7b24b0e5904dac2f3859d9ab36577dbdfb5b7d55`.

## 3. `2a4543da3` Keep data-parallel TT ranks in step across empty prefill and preemption

1. **Requirement and failure.** Data-parallel ranks diverged when one rank's
   prefill produced no tokens or when the decode path dropped scheduler
   notifications.
2. **Owner.** Case 1, plugin defect in the gathered-DP step protocol.
3. **Model-side alternative.** None.
4. **Existing mechanisms.** None apply.
5. **Effect on other models.** DP models only; this run serves one rank.
6. **Authorization.** No shared interface added.
7. **Validation.** `tests/test_dp_engine.py`.
8. **Status.** `implemented`, commit `2a4543da3e5cc85533cf24f08267f56e9df17ae6`.

## 4. `f555870a9` Checkpoint per-request sampling RNG so recompute reproduces accepted history

1. **Requirement and failure.** A preempted request redrew host samples from
   an advanced generator state and its regenerated tail disagreed with
   history the scheduler had accepted.
2. **Owner.** Case 1, plugin defect (host sampler state across recompute).
3. **Model-side alternative.** None; host sampling RNG is plugin state.
4. **Existing mechanisms.** None apply.
5. **Effect on other models.** Every model using the host sampler.
6. **Authorization.** No shared interface added.
7. **Validation.** `tests/test_accepted_history_rng.py`, `tests/test_recompute_tail.py`.
8. **Status.** `implemented`, commit `f555870a9bb8cef8ffb963288f2c56c2c97c195e`.

## 5. `cf983b0f5`, `87874eb24`, `c87df030e` Gemma 4 reasoning and tool parsers

1. **Requirement and failure.** Gemma 4 thought-channel markers leaked into
   chat output, the reasoning parser was never given the chance to adjust a
   request (`87874eb24` adds the hook in upstream vLLM
   `vllm/reasoning/abs_reasoning_parsers.py` and `vllm/entrypoints/openai/engine/serving.py`),
   and a thought block before a tool call broke tool parsing.
2. **Owner.** vLLM parsers are per-model by upstream design (registered by
   name), so model-specific parser modules are the upstream pattern, not a
   model branch in shared execution code. The request-adjust hook is a shared
   vLLM interface change (case 2).
3. **Model-side alternative.** None; parsers run in the API server.
4. **Existing mechanisms.** Upstream parser registration (used).
5. **Effect on other models.** Only requests that select `gemma4` parsers;
   the reasoning-parser hook is a no-op for parsers that do not implement it.
6. **Authorization.** `87874eb24` adds shared interface in vLLM core.
   Per-item authorization: not recorded.
7. **Validation.** `tests/test_gemma4_reasoning_parser.py`,
   `tests/test_gemma4_tool_parser.py`; served tool and reasoning requests in
   `readiness_vllm/openai_feature_checks.json`.
8. **Status.** `implemented`, commits `cf983b0f5`, `87874eb24f84925df5a09ea0f9aa030e9ca3be6e`, `c87df030e`.

## 6. `cd25e6602` Full block-table width for sliding windows, KV admission reserve, prefix caching opt-in

1. **Requirement and failure.** (a) The runner trimmed the per-request block
   table to the pool's block count; a sliding-window continuation then
   addressed a column past the table ("sliding continuation page table does
   not cover its history" on a 131k prompt over a 1,509-block pool). (b) A
   pool filled to the last block by same-length prompts entered a
   preempt/replay cycle. (c) Prefix caching was vetoed for every
   sliding-window model.
2. **Owner.** (a) Case 1: `references/prefill-features.md` names this exact
   trim as a plugin defect for sliding-window models. (b) Case 2: admission
   policy is scheduler-owned. (c) Case 1: the veto contradicts the model
   interface the plugin otherwise documents (`supports_prefix_caching`).
3. **Model-side alternative.** (a) A position modulo in the model would hide
   vLLM's absolute block addressing and break reclaim; rejected by the
   prefill-features contract (invariant 5). (b) None. (c) None.
4. **Existing mechanisms.** No `--additional-config` option controlled the
   table width or admission; `supports_prefix_caching` existed but was
   overridden by the sliding-window veto.
5. **Effect on other models.** (a) Only models with a `SlidingWindowSpec`
   group. (b) Every model: `kv_admission_reserve_blocks_per_seq` (default 4)
   holds prompts back near the pool limit. (c) Sliding-window models that
   declare `supports_prefix_caching_sliding_window`.
6. **Authorization.** Adds shared interface: the `tt` config option
   `kv_admission_reserve_blocks_per_seq` and the capability key
   `supports_prefix_caching_sliding_window`. Per-item authorization: not
   recorded.
7. **Validation.** `tests/test_block_table_width.py`,
   `tests/test_admission_reserve.py`, `tests/test_admission_reserve_cpu.py`,
   `tests/test_scheduler_cache_pressure_cpu.py`; served
   `readiness_vllm/prefix_cache_check.json` (status pass).
8. **Status.** `implemented`, commit `cd25e660210f1e9132e6539f222b6580fa194e8f`.

## 7. `f96662a7d` Size the KV admission reserve per KV group and count in-flight prefills

1. **Requirement and failure.** The reserve from record 6 still admitted a
   16th resident with five free blocks and preempted; in-flight prefills were
   not counted and each KV group can take a block per step.
2. **Owner.** Case 2 (scheduler-owned admission policy), refinement of 6(b).
3. **Model-side alternative.** None.
4. **Existing mechanisms.** The option added in record 6.
5. **Effect on other models.** Every model: reserve scales with KV groups.
6. **Authorization.** No new interface; changes the default behavior of the
   option from record 6.
7. **Validation.** `tests/test_admission_reserve.py`, `tests/test_num_available_blocks.py`.
8. **Status.** `implemented`, commit `f96662a7d239c077e1435b658c0e7d316f9c4799`.

## Chunked prefill for Gemma 4 (`platform.py` `_CHUNKED_PREFILL_MODEL_TYPES`)

The pinned plugin enables scheduler chunked prefill only for
`model_type in {"gemma4", "gemma4_unified"}` (`platform.py:46`, from commit
`2f81f493b`). This is a model-type branch in shared plugin code, which the
rules do not allow. A model-independent form would be a documented
capability (`supports_chunked_prefill`) that the model declares. That is a
new `model_capabilities` key and therefore needs human authorization; it is
recorded here as `proposed` and is not implemented in this stage.

- Requirement: chunked prefill for prompts over `max_num_batched_tokens`.
- Owner: case 2 (scheduler policy) for the behavior; the selection mechanism
  belongs to the model interface.
- Model-side alternative: none; scheduling is plugin-owned.
- Existing mechanisms: the model-type set above (pinned behavior).
- Effect on other models: none until a model declares the key.
- Authorization: `pending`; exact item: new key `supports_chunked_prefill`.
- Planned validation: `tests/test_chunked_prefill_policy.py` extended to
  select by capability; rerun `readiness_vllm/prefix_cache_check.json` and
  the mixed-shape check.
- Status: `proposed`.

## Verification of the pinned state in this run

```bash
git -C /home/hous/dev/vllm rev-parse HEAD        # f96662a7d239c077e1435b658c0e7d316f9c4799
git -C /home/hous/dev/vllm status --short        # clean
python -m pip show vllm vllm-tt-plugin           # editable installs from /home/hous/dev/vllm
```
