# Stage Review

Verdict: more-work-needed

Stage: 6, full model, third review. Reviewer: independent stage-review subagent, read-only. Review date: Oct 14
(Eastern Time). Source under review: live worktree at HEAD `fb2e3b5a` plus uncommitted stage changes. The live tree
equals the chain 4 end manifest
(`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence4/source_manifest_end.txt`),
and the start and end manifests are equal.

Abbreviations: HF (Hugging Face CPU reference), TT (Tenstorrent TTNN implementation), KV (key/value cache), GDN
(Gated DeltaNet linear-attention layer), MoE (mixture of experts), LM head (final vocabulary projection), RoPE
(rotary position embedding), CCL (collective communication library), DRAM (device memory), L1 (per-core static
RAM), CPU (host processor), API (application programming interface), KL (Kullback-Leibler divergence), TV (total
variation distance), PCC (Pearson correlation coefficient), TTFT (time to first token), BF16 (16-bit brain float),
BFP8 (8-bit block float), vLLM (the upstream serving engine), MiB (2^20 bytes). B1, B4 and B32 mean decode batches of
1, 4 and 32 rows.

## Status of review 2 Required Work

| Review 2 item | Status | Basis |
|---|---|---|
| 1. Inactive rows and fixed slots | Partly resolved | Steps 1 and 3 are done. Step 2 is only partly done: `request_slots` is checked, but other slot arguments are still dropped without an error. See Required Work 1. |
| 2. Sampling parameter handling | Resolved | Penalties, log-probabilities and seeds raise `ValueError`. The refresh key covers every `SamplingParams` field. 11 CPU tests pass (I reran them). |
| 3. TT against HF quality deviations | Resolved | Calibration probe on all three profiles. I recomputed `compare.json` from the saved tensors and got an identical result. HF teacher forcing controls the p150x2 story error. |

I verified these claims against raw artifacts. Each one holds:

- Headline TTFT, token-out and logits-only numbers, all three runs per profile, KV pool sizes, DRAM free, largest
  free block, census categories and the unnamed remainder (`evidence4/m{4,2,1}/full_evidence.json`).
- Readiness top-1, top-5 and top-100 for AIME24 teacher forcing, book teacher forcing and prefill check, on all
  profiles (`evidence4/m*/teacher_forcing_*.log`, `prefill_check_aime24.log`).
- All 46 boolean checks in each `full_evidence.json` are true. Every token result equals chain 3, and the
  continuation PCC values agree to 6 decimals.
- Autoregressive and qualitative outputs are byte-identical to chain 2. The HF prefix-match table is correct.
- Sampling audit, sampler micro test, split-sampling trace and frozen-seed control match the README tables.
- The chain 4 calibration logits on p150x4 and p150x2 are bitwise equal to the review 2 fix runs.
- Inactive-row runs: 7 cases pass on every profile, 3 decode captures.
- The stage gate passes: no degenerate output, and the context contract reports target 262144 and supported
  262144.
- The memory recheck source hashes match the current `tt/` files.
- No unsafe-allocation, program-cache-miss or trace warning appears in any chain 4 log. No credentials appear in
  the artifacts.

The delivered greedy B1 path is correct and well evidenced on all three profiles. The two items below are
serving-contract defects in the generator API. Both can silently corrupt another request's GDN state under the
pinned vLLM plugin.

## Required Work

- P2: Slot arguments that the pinned plugin sends are still silently ignored.
  Evidence:
  - `prefill_forward(..., request_slots=None, **kwargs)` and `decode_forward(..., request_slots=None, **kwargs)`
    accept and ignore unknown keywords
    (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/generator.py:291-292`,
    `:309-310`). Only `request_slots` is checked (`:315-316`).
  - The pinned plugin sends `empty_slots` to prefill
    (`/home/hous/dev/qwen3.5-35b-a3b/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py:3348`) and `slot_remap` to
    decode (`/home/hous/dev/qwen3.5-35b-a3b/vllm-tt-plugin/src/vllm_tt_plugin/async_decode.py:1269`). The plugin
    comment at `model_runner.py:2321` says that without the slot argument a stateful model falls back to `range(N)`
    and overwrites a decoding request's state. The reusable qwen36 adapter reads `empty_slots`
    (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/demos/blackhole/qwen36/tt/qwen36_vllm.py:221`).
  - Reviewer CPU probe (stubbed device calls, same pattern as
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/test_generator_sampling.py`):
    `prefill_forward(..., empty_slots=[5])` prefilled slot 0 with no error.
    `decode_forward(..., slot_remap=[5, 1, 2, 3])` was accepted with no error.
  - Review 2 step 2 offered "reject unknown kwargs" as one remedy. The stage implemented a check for one name only.
  Why this matters: a dropped slot argument writes one request's GDN state into another request's slot. Output is
  then wrong with no error. The goal requires serving-ready fixed slots.
  Required next step:
  1. In `prefill_forward` and `decode_forward`, raise `ValueError` for slot arguments that the generator does not
     honor: `empty_slots` and `prefill_empty_slots` that differ from the slots used, and any non-identity
     `slot_remap`. Mapping `empty_slots` onto `request_slots` is also acceptable.
  2. Add CPU tests for each name in
     `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/test_generator_sampling.py`.
  3. This is a host-only change. Rerun the CPU tests. Record the old and new `tt/generator.py` hashes as the
     provenance handoff requires.

- P2: An inactive decode row overwrites the GDN state of its slot. Nothing guards a live slot, and the README rule
  for Stage 9 misses a case that the pinned plugin uses.
  Evidence:
  - Decode runs every row, inactive rows included. It selects the row's slot state and runs the full layer
    forward, which updates the recurrent and convolution state in place
    (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/functional_decoder.py:468-476`).
  - The README states this (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/README.md`
    lines 237 and 452-453). The only hazard it names is a partly prefilled request (chunked prefill).
  - The pinned plugin keeps model state for running requests that are not scheduled in a step: "Temporary
    unscheduling does not [invalidate model-owned request state]: a request may simply be hidden for this step"
    (`/home/hous/dev/qwen3.5-35b-a3b/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py:968-969`, `:978-990`).
    The plugin pads decode rows with position -1 (`:2069`) and moves non-decoding slots to the tail rows
    (`:1268`). Under the generator contract, a hidden request's slot becomes an inactive row, and its GDN state is
    overwritten.
  - The reusable qwen36 implementation preserves inactive GDN rows and tests it
    (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/demos/blackhole/qwen36/tests/test_decode_bucketing.py:683-739`).
  - `run_inactive_rows.py` checks a fresh prefill after inactivity. It does not check that a live slot survives an
    inactive step, because by design it does not survive.
  Why this matters: a live request that sits out one decode step resumes from corrupted recurrent state and gives
  wrong text with no error. The goal requires serving-ready inactive rows. GDN state is the per-request cache for
  30 of the 40 layers.
  Required next step. Choose one option:
  - (a) Preserve state on device: skip or mask the GDN state update for rows with a negative position, as qwen36
    does. Add a device case to
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_inactive_rows.py`:
    prefill slot s, run several decode steps with s inactive, then resume s and match its B1 tokens. Rerun it with
    the tracker and strict cache on p150x4 and p150. Rerun the affected decode gates, because the decode trace
    changes.
  - (b) Make it loud: track which slots hold prefilled state, clear a slot on `reset()`, on prefill at start 0 and
    on an explicit release API (for example `release_request(slot)`). Raise `ValueError` when `decode_forward`
    marks a live slot inactive. Add CPU tests.
  For both options, correct the README Stage 9 note. It must name the temporary unscheduling case and say that
  prefix caching is unsupported (see Other Concerns).

## Other Concerns

- B32 decode runs the whole decoder layer once per row: attention or GDN, MoE and both all-reduces
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/functional_decoder.py:468-476`).
  README line 468 says "loops attention and GDN per row", which understates it.
  - Estimate, not measured: B32 costs about 32 B1 steps, so batching gives little aggregate gain.
  - The pinned plugin pads plain decode to the wire capacity
    (`/home/hous/dev/qwen3.5-35b-a3b/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py:2056-2079`). So a
    single-request decode can land on the B32 trace unless the Stage 9 adapter selects a smaller bucket.
  - Stage 13 needs client concurrency up to 32. Measure B4 and B32 token-out once now to give Stage 7 a baseline,
    and correct the README wording.
- Trace command buffers are not counted in any memory or context number.
  - The model scripts open the mesh with `trace_region_size=0`
    (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/mesh_util.py:7`). In
    that mode tt-metal allocates trace buffers dynamically in ordinary DRAM, top-down
    (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/distributed/mesh_trace.cpp:65-71`).
  - `memory_after_build` is taken after `prepare()` and before any capture.
  - The earlier context contract says the 2 GiB reserve for activations, trace and CCL buffers is "an estimate
    until the full model measures it"
    (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/context_contract.json`
    lines 567 and 654). That reserve is still not measured.
  - This is not a capability failure. The 262100-token run completes on every profile with all six traces live.
  - p150 has 1.62 GB free before capture. Record allocator totals and the largest free block after all captures,
    for Stage 9 headroom.
- README line 73 says the HF margin at the token-10 divergence is not recorded. The data exists in
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/review2_fixes/calibration/hf_story_m2_teacher_forced.json`
  at step 10:
  - The HF batched forward ranks the TT token " move" first (probability 0.264).
  - The HF incremental reference chose " stretch" (token 14084).
  - So the identical divergence on all three profiles comes from noise in the HF reference, not from a
    systematic TT difference. Record this in the README.
- README lines 455-456 imply that prefix caching works if starts align to 128. GDN state belongs to a slot, not to
  a prefix. A cached-prefix request that starts at `start_pos > 0` in a new slot does not reset that slot
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/generator.py:283-284`). It
  runs from another request's state or from zeros. Unless GDN state snapshots exist, state that prefix caching is
  unsupported.
- The pinned plugin uses `num_logprobs = -2` to mean no log-probabilities
  (`/home/hous/dev/qwen3.5-35b-a3b/vllm-tt-plugin/src/vllm_tt_plugin/input_batch.py:54`,
  `/home/hous/dev/qwen3.5-35b-a3b/vllm-tt-plugin/src/vllm_tt_plugin/model_input.py:91`). `_set_sampling` raises on
  any nonzero `num_logprobs` (`generator.py:116`). Every plugin request would raise unless the Stage 9 adapter
  normalizes the sentinel. Add this to the Stage 9 notes.
- Two smaller API issues:
  - `decode_forward` with an unprepared batch size fails with `KeyError` in `_write_positions` (`generator.py:156`)
    before it reaches the clear `ValueError` in `_decode_trace` (`:253-254`).
  - `build_generator` prepares only B1 by default (`decode_batches=(1,)`, `generator.py:39`). Serving must pass
    `(1, 4, 32)`. Document this in the Stage 9 notes.
- Carried from review 2, recorded and correctly assigned to later stages:
  - headline measured with the tracker on;
  - host seed write on every sampled step, and the SKIP protocol not tested across decode replays;
  - routed experts stored twice on p150x2 and p150x4;
  - unnamed DRAM remainder of 187 to 313 MiB, and construction peaks not measured;
  - LM head program config, full-vocabulary top-k cost and force-argmax;
  - log-probabilities on 1, 2 and 4 devices;
  - explicit request seeds.
- Provenance: the stage source is uncommitted. Chain 4 records HEAD and file hashes, but no source archive. Before
  you record the checkpoint SHA, check that the committed files reproduce the chain 4 end manifest
  (`/home/hous/dev/qwen3.5-35b-a3b/experiment-provenance-handoff.json`, `required_before_acceptance`).

## Hard-Check Gaps

- No check confirms that a live slot keeps its GDN state across a decode step where its row is inactive (Required
  Work 2).
- No test runs a batch that mixes greedy and sampled rows, or different per-row parameters, in one decode step.
- The qualitative suite stops at 128 tokens. Only p5 reaches a final answer; p0 to p4 end inside the thinking
  section on every profile and in the HF control.
- The watcher ran only on the 2-layer model. `generate(host_sampling=True)` ran only in the 2-layer generator
  probe on an earlier source.
- Accuracy beyond 8193 tokens uses only periodic text.
- The calibration probe has no pass threshold, so a later regression in calibration would not fail a gate.
- Logit determinism across batch positions is checked only at the token level.
- The sampled distribution is checked only at bucket 32 with fixed logits.
- No B4 or B32 decode timing, no sampled token-out timing, no tracker-off timing, and no DRAM measurement after
  trace capture.

## Anomaly Ledger

- Observed anomaly: An inactive row overwrites its slot's GDN state, so a hidden live request is corrupted.
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/functional_decoder.py:468-476`;
  README lines 237 and 452-453; plugin `model_runner.py:968-969`.
  Affected path: `decode_forward` and the traced loop with inactive rows.
  Control or comparison: qwen36 preserves inactive GDN rows (`test_decode_bucketing.py:683-739`).
  Likely subsystem: GDN slot state ownership.
  Investigation performed: code reading of the generator, the decoder and the pinned plugin.
  Resolution: more-work-needed (Required Work 2).

- Observed anomaly: `empty_slots` and `slot_remap` are accepted and ignored.
  Evidence: reviewer CPU probe; `generator.py:291-292`, `:309-316`.
  Affected path: low-level prefill and decode API.
  Control or comparison: the plugin sends both names (`model_runner.py:3348`, `async_decode.py:1269`).
  Likely subsystem: generator argument validation.
  Investigation performed: stubbed CPU call.
  Resolution: more-work-needed (Required Work 1).

- Observed anomaly: All three profiles leave HF at token 10 with the same choice (" move faster than" against
  " stretch and shrink").
  Evidence: `evidence4/m{4,2,1}/autoregressive/autoregressive_meta.json` (prefix 10 on every profile).
  Affected path: greedy token-out decode, story prompt without chat template.
  Control or comparison: the HF batched forward over the same prefix ranks " move" first with probability 0.264
  (`review2_fixes/calibration/hf_story_m2_teacher_forced.json`, step 10). Only the HF incremental reference picks
  " stretch".
  Likely subsystem: noise in the HF reference at a near-tie.
  Investigation performed: reviewer read the existing HF teacher-forced artifact.
  Resolution: controlled. The README still needs to record this (Other Concerns).

- Observed anomaly: TT is sharper than HF at the 428-token continuation position (KL 0.50, 0.20 and 0.42).
  Evidence: `evidence2/continuation/m4_with_kl.json`.
  Affected path: prefill and decode logits in periodic text.
  Control or comparison: calibration over 100 AIME24 positions on all profiles. KL mean is 0.0055 to 0.0074 and
  KL max is 0.19 or less. On the 40 unsaturated positions, TT probability of the HF top-1 matches HF within 0.01.
  The median slope of the top-100 logits is 1.00.
  Likely subsystem: precision margin in copy-heavy periodic text (hypothesis).
  Investigation performed: calibration probe on all three profiles; I recomputed it.
  Resolution: controlled as isolated. The cause is not isolated and is assigned to Stage 8.

- Observed anomaly: The p150x2 story accepts "Seventy-two" hours per week as "Correct!".
  Evidence: `evidence4/m2/autoregressive/tt_completion.txt`.
  Affected path: p150x2 greedy token-out decode.
  Control or comparison: HF teacher-forced on the TT text ranks every error token first. 243 of 256 tokens are HF
  top-1, and the maximum rank is 2. I verified this.
  Likely subsystem: model behavior.
  Investigation performed: HF control.
  Resolution: controlled.

- Observed anomaly: On p150x4, 3 of 7 sampled sequences equal greedy.
  Evidence: `evidence4/m4/sampling_audit.json` `sampled_equals_greedy`.
  Affected path: sampled B4 and sampled `generate()`.
  Control or comparison: the p150x2 repeat test measured TT probabilities of 0.33 and 0.46 for the same rows. The
  distribution TV is 0.018, there are no constant rows, and the two `generate()` runs differ from each other.
  Likely subsystem: confident prompts.
  Investigation performed: audit checks and repeat test.
  Resolution: controlled.

- Observed anomaly: Whole and split prefill logits differ (PCC 0.83 to 0.96).
  Evidence: `evidence4/m*/full_evidence.json` `continuation`.
  Affected path: continuation prefill.
  Control or comparison: HF KL of 0.0002 or less at 4301, 8193 and 12417 tokens; same argmax in every case.
  Likely subsystem: tail noise.
  Investigation performed: HF KL probe in chain 2.
  Resolution: controlled.

- Observed anomaly: The 2-layer probe diverges from HF on p150x2 and p150 at TT near-ties.
  Evidence: `evidence2/m*/watcher_probe.json`, `probe_control/`.
  Affected path: reduced 2-layer build.
  Control or comparison: eager, traced and post-warmup logits are bitwise equal.
  Likely subsystem: numerics that depend on the build.
  Investigation performed: path and watcher controls.
  Resolution: controlled for trace and warmup. Residual: the build dependence is not isolated.

- Observed anomaly: 187 to 313 MiB of DRAM per device is not reachable from the generator object graph.
  Evidence: `memory_handoff_recheck.json` `unnamed_dram_mib`.
  Affected path: memory ownership.
  Control or comparison: none.
  Likely subsystem: program-cache or CCL buffers (unverified).
  Investigation performed: none.
  Resolution: residual, assigned to Stage 7.

## Scope Inspected

- Goal and skill paths:
  - `/home/hous/dev/qwen3.5-35b-a3b/runs/main/06-06-full-model.prompt.txt`
  - `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`
  - `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`
  - `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/full-model/SKILL.md`
  - `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/tt-enable-tracing/SKILL.md`
  - `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/qualitative-check/SKILL.md`
  - `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/runtime/readiness_check/contract.py`,
    `contract_vllm.py`
  - `/home/hous/dev/qwen3.5-35b-a3b/experiment-provenance-handoff.json`
- Artifact paths:
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/README.md`,
    `work_log.md`, `stage_review1.md`, `stage_review2.md`, `memory_handoff_recheck.json`
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/context_contract.json`
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence4/`:
    chain script and log, source manifests, `source_provenance.txt`, `cpu_tests.log`, `degenerate_check.*`,
    `control_frozen_seeds_l2_m4.json`, `calibration/`, `qualitative/`, and `m{4,2,1}/` (full evidence, readiness
    logs, autoregressive, inactive rows, split-sampling trace, sampling audit, sampler micro, calibration)
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/review2_fixes/`
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence2/`
    (`SUPERSEDED.md`, `m*/profile/summary.json`, `decode_perf_report.csv`, completions, qualitative)
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/`
    (manifests, `SUPERSEDED.md`) and `evidence1/SUPERSEDED.md`
  - `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/references/` (refpt metadata),
    `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/calibration/`,
    `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/evidence4_work/`,
    `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/qualitative/hf_outputs.json`,
    `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/inactive/`
- Code paths:
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/generator.py` (all),
    `tt/model.py` (all), the `tt/optimized_decoder.py` diff against HEAD, the router slice cache,
    `tt/functional_decoder.py` (slots, RoPE, decode loop, state reset), `tt/multichip_decoder.py` (class
    structure)
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/test_generator_sampling.py`,
    `test_generator_warmup.py`, `run_inactive_rows.py`, `run_sampling_audit.py`, `run_split_sampling_trace.py`,
    `run_full_evidence.py`, `run_calibration_probe.py`, `runtime_audit.py`, `mesh_util.py`
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/common/sampling/generator.py` (`SamplingParams`,
    `reset_sampling_params`, `sample`, `format_sampling_params`, `SeedManager`),
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/common/sampling/tt_log_probs.py`
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/tt_metal/distributed/mesh_trace.cpp` (dynamic trace buffer mode)
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/demos/blackhole/qwen36/tt/qwen36_vllm.py`,
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/demos/blackhole/qwen36/tests/test_decode_bucketing.py`
  - `/home/hous/dev/qwen3.5-35b-a3b/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py`, `async_decode.py`,
    `input_batch.py`, `model_input.py`, `platform.py` (slot remap, empty slots, unscheduling, decode padding,
    sentinels, chunked prefill policy)
- Commands run (none touched a device):
  - Manifest recompute over `tt/` and `tests/` and diff against the chain 4 end manifest (equal); diffs of the
    chain 2, 3 and 4 manifests.
  - `evidence4/calibration/compare.py 4 2 1` rerun on CPU (output identical to `compare.json`). `torch.equal`
    of the review 2 and chain 4 calibration tensors (equal).
  - Python comparisons of `full_evidence.json` across chains 2, 3 and 4; of the sampling audit, micro, trace and
    inactive-row JSON; of HF and TT qualitative tokens; and of the HF story teacher-forced JSON.
  - `pytest tests/test_generator_sampling.py` with `PYTHONDONTWRITEBYTECODE=1` (11 passed).
  - A stubbed CPU probe of `prefill_forward(empty_slots=...)` and `decode_forward(slot_remap=...)` in the session
    scratchpad.
  - Stage gate `06-full-model.check.sh` with `MODEL_DIR` set (no degenerate output; context contract OK).
  - `grep` over the chain 4 logs for trace, allocation and cache warnings (none), and over the artifacts for
    credentials (none).

## Residual Risk

- Serving correctness depends on the Stage 9 adapter keeping every live slot in an active row on every decode step,
  until Required Work 2 is done.
- B32 decode throughput is likely close to B1 throughput, because every layer runs once per row. This is
  unmeasured.
- DRAM headroom after trace capture is unknown. p150 has the least margin.
- The cause of the 428-token sharpening in periodic text is not isolated.
- Decode numerics of the truncated 2-layer model depend on the build, and the cause is not isolated.
