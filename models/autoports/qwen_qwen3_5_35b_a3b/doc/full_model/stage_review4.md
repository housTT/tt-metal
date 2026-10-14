# Stage Review

Verdict: clean-pass

Stage: 6, full model, fourth review. Reviewer: independent stage-review subagent, read-only. Review date: 2026 Oct 14.
Source under review: live worktree at HEAD `fb2e3b5a604e064033aba073d015be1a1869d282` plus uncommitted stage changes.
I recomputed the source manifest over `tt/` and `tests/` with the chain 5 `manifest()` function. It is equal to
`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence5/source_manifest_end.txt`,
and the chain 5 start and end manifests are equal.

Abbreviations: HF (Hugging Face CPU reference), TT (Tenstorrent TTNN implementation), KV (key/value cache), GDN
(Gated DeltaNet linear-attention layer), MoE (mixture of experts), LM head (final vocabulary projection), CCL
(collective communication library), DRAM (device memory), CPU (host processor), API (application programming
interface), KL (Kullback-Leibler divergence), TV (total variation distance), TTFT (time to first token), t/s/u
(tokens per second per user), vLLM (the upstream serving engine), MiB (2^20 bytes), MB (10^6 bytes). B1, B4 and B32
mean decode batches of 1, 4 and 32 rows. p150, p150x2 and p150x4 are the 1, 2 and 4 chip profiles.

## Status of review 3 Required Work

| Review 3 item | Status | Basis |
|---|---|---|
| 1. Slot arguments that the pinned plugin sends are silently ignored | Resolved | `prefill_forward` and `decode_forward` have no catch-all keyword argument (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/generator.py:292-293`, `:320-321`). `empty_slots` selects the prefill slots and must agree with `request_slots` (`:296-305`). A non-identity `request_slots` or `slot_remap` raises `ValueError` (`:328-330`). An unknown keyword raises `TypeError`. CPU tests `test_decode_rejects_slot_map_other_than_row_order` and `test_prefill_empty_slots_select_request_slots` cover each name. I reran `tests/test_generator_sampling.py`: 14 passed. Old and new `tt/generator.py` hashes are in the chain 4 and chain 5 manifests (`d06b0443...` and `4140f40c...`). |
| 2. An inactive decode row overwrites the GDN state of a live slot | Resolved (option b, loud) | The generator tracks live slots (`generator.py:68`, `:310`, `:381`), clears them in `release_slots` and `reset` (`:414-418`), and raises `ValueError` before any device work when a decode call marks a live slot inactive (`:332-335`). Device evidence: `evidence5/m{4,2,1}/inactive_rows.json`, case 8 (B4, slot 2 live, hide attempt at step 5): `hide_rejected` true, and slot 2 still equals its B1 tokens on all three profiles. All 8 cases pass, 3 decode captures. README Stage 9 notes now name the temporary unscheduling case, `release_slots`, and say prefix caching is unsupported (README lines 485-503). |

Review 3 Other Concerns that the stage closed with evidence:
- B4 and B32 token-out are measured (`evidence5/m*/batch_perf.json`). I re-derived every number in the README table.
- DRAM after all captures is measured. Capture deltas are 249.6, 274.5 and 298.0 MB per device (allocator bytes in
  `batch_perf.json`), as the README states.
- The HF margin at the token 10 story divergence is recorded (README lines 90-94).
- `num_logprobs = -2`, `decode_batches=(1, 4, 32)`, and the `KeyError` to `ValueError` change are recorded or
  fixed (`generator.py:326-327`).

Review 1 and review 2 items stay closed: sampler scope accounting (0 surviving buffers per bucket on all profiles),
unaligned continuation starts rejected (`continuation.unaligned_start` in each `full_evidence.json`), sampled mode
samples (`sampled_not_all_greedy` true; distribution TV 0.013 to 0.019), 32 request slots on p150, sampling
parameter rejection and refresh key (CPU tests), and the HF controls for the quality deviations.

Claims I re-derived from raw artifacts (all hold):
- Headline TTFT, token-out and logits-only numbers and the 3 runs per profile (`evidence5/m*/full_evidence.json`
  `perf_128_128`, `logits_only_decode_ms`).
- Tracker-off B1, B4 and B32 numbers, DRAM after build, DRAM after all captures and largest free blocks
  (`evidence5/m*/batch_perf.json`).
- Readiness top-1, top-5 and top-100, TTFT and decode t/s/u for AIME24 teacher forcing, book teacher forcing and
  prefill check (`evidence5/m*/teacher_forcing_*.log`, `prefill_check_aime24.log`). Bar met on every profile; the
  tightest case is book top-5 63 of 64 on p150x2 and p150.
- 46 of 46 boolean checks per profile in `full_evidence.json`. Determinism, non-aligned lengths, B4, B32 and
  boundary results equal chain 4.
- Chain 5 calibration tensors are bitwise equal to chain 4 (`torch.equal` over `prefill` and `decode` for all
  three profiles).
- Autoregressive prefix match with HF is 10 tokens on every profile (HF token 14084 " stretch" against TT 3166).
- Qualitative prefix-match table (README lines 134-141) matches the token comparison against
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/qualitative/hf_outputs.json`.
- Census remainder of 271.2, 312.6 and 187.3 MiB equals allocator total minus census sum in
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/context_contract.json`.
- `memory_handoff_recheck.json` `source_sha256` matches every live `tt/*.py` file.
- Chain 5 log: 39 `END` lines, all `rc=0`, `SOURCE_STABLE`, `CHAIN_DONE`, 10:42Z to 16:37Z (06:42 to 12:37
  Eastern Time). CPU tests 1218 passed.
- Stage gate `06-full-model.check.sh` with `MODEL_DIR` set: no degenerate output; context contract target 262144,
  supported 262144.
- No credentials in the docs tree.

## Required Work

None.

## Other Concerns

These do not block the stage. Fix the documentation items before the checkpoint commit if convenient.

- `batch_perf` logs contain the Metal warning "Allocating device buffers is potentially unsafe due to the existence
  of an active trace" (`evidence5/m4/batch_perf.log:40`, `evidence5/m1/batch_perf.log:27`, also in m2). README
  lines 344-346 classify this warning only for the watcher and profiler runs, and README line 328 says "Every
  evidence run uses `TT_METAL_TRACE_ALLOC_TRACKING=1`", which is not true for `run_batch_perf` (tracker off by
  design, README line 51) or `run_sampler_micro` (tracking only). I classified it below as controlled: the same
  sequence (B1 capture, then prefill of new requests, then deferred B4 and B32 capture) passes under the tracker in
  `run_inactive_rows` and `run_full_evidence`. Add one sentence to the README and correct line 328.
- `batch_perf` does not check its B4 and B32 tokens. Correctness for those buckets comes from the tracker-on runs,
  not from this run. State that next to the table.
- README line 412 labels the 250 to 298 MB capture delta as "Decode traces B1, B4, B32". The delta also contains
  the three sampler traces and any allocations made between the two snapshots. Relabel it as "all six traces and
  allocations after build".
- The context contract `full_model_validation` does not record `memory_after_all_captures`. The older sections of
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/context_contract.json` (lines
  567, 654, 853, 857) still say the 2 GiB reserve for trace and CCL buffers is unmeasured. The measurement now
  exists. Add it to `full_model_validation` so Stage 9 reads one source.
- Stale small numbers: README line 464 gives token-out minus logits-only as 0.87 and 1.33 ms for p150x2 and p150.
  Chain 5 gives 0.88 and 1.34 ms. README line 472 gives logits-only as 8.58 ms; the headline gives 8.59 ms.
- `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence2/SUPERSEDED.md`
  still says it is superseded by `evidence4/`, which is itself superseded by `evidence5/`.
- An active decode row whose slot was never prefilled (not live) is accepted with no error, and decodes from zero or
  stale GDN state. The live-slot set already exists, so rejecting this case is a cheap guard. The Stage 9 adapter
  must also map the plugin hooks `release_request` and `note_state_slots_moved`
  (`/home/hous/dev/qwen3.5-35b-a3b/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py:196-215`, `:919-935`) onto
  `release_slots` and the identity slot contract.
- With option (b), a request whose prefill is split into chunks holds a live slot. Any decode step that marks its
  row inactive raises. So chunked prefill that overlaps decode in the same row range is not possible with this
  generator. This is loud, not silent, but Stage 9 needs a plan. Name this case in the README Stage 9 notes next to
  hidden requests.
- The low-level `decode_forward` writes tokens and positions and builds a [batch, 8192] host page table on every
  call (README line 504). The standalone traced loop has zero per-token refreshes, so the goal item holds, but the
  serving path is host-stepped until Stage 9 or 10 feeds tokens on device.
- Carried and correctly assigned to later stages: B4 and B32 run the whole decoder layer once per row, so aggregate
  throughput is flat with batch (113.3, 93.0 and 68.7 t/s at B32); host seed write per sampled step; routed experts
  stored twice on p150x2 and p150x4; unnamed DRAM remainder and construction peaks; LM head program config,
  full-vocabulary top-k and force-argmax; log-probabilities on 1, 2 and 4 devices; 128-token continuation start
  alignment.
- Provenance: the stage source is uncommitted and there is no source archive or dirty diff for chain 5
  (`/home/hous/dev/qwen3.5-35b-a3b/experiment-provenance-handoff.json`, `required_before_acceptance` item 1). The
  live tree equals the chain 5 end manifest now. Before recording the checkpoint SHA, check that the committed
  `tt/` and `tests/` files reproduce `evidence5/source_manifest_end.txt`, and record the result in the work log.

## Hard-Check Gaps

- No test runs a decode step that mixes greedy and sampled rows, or different per-row sampling parameters.
- The p150x4 sampled `generate()` prompt has no repeat test. In chain 5 both sampled `generate()` runs equal greedy
  and equal each other. Across chains 3b, 4 and 5 the p150x4 rate is 3 of 6 runs, against 1 of 6 on p150x2 and 0 of
  6 on p150. The check `sampled_not_all_greedy` passes, and earlier chains show this prompt does sample on p150x4.
- The qualitative suite stops at 128 tokens; only p5 reaches the final answer on any profile and in the HF control.
- p150x2 qualitative p3 differs from HF at the first token. Review 1 classified it as a near-tie between two native
  openings, but no artifact records the HF probability of " Here" at that position.
- The watcher ran only on the 2-layer model. `generate(host_sampling=True)` ran only in the 2-layer generator probe.
- Accuracy beyond 8193 tokens uses only periodic text.
- The calibration probe has no pass threshold.
- No sampled token-out timing and no force-argmax comparison.

## Anomaly Ledger

- Observed anomaly: Metal warns about buffer allocation while a trace is active in `run_batch_perf`.
  Evidence: `evidence5/m4/batch_perf.log:40`, `evidence5/m1/batch_perf.log:27`, and the m2 log.
  Affected path: B1 `generate()` repeats and B4 and B32 prefill after the B1 capture, tracker off.
  Control or comparison: the same order of prefill after capture and deferred B4 and B32 capture runs with
  `TT_METAL_TRACE_ALLOC_TRACKING=1` and `QWEN35_STRICT_CACHE=1` in `evidence5/m*/inactive_rows.json` and
  `full_evidence.json`. Both pass, and B4 and B32 rows equal B1.
  Likely subsystem: expected Metal warning when the tracker is off.
  Investigation performed: read `tests/run_batch_perf.py` and `tests/run_inactive_rows.py`; compared call order.
  Resolution: controlled. The README should classify it (Other Concerns).

- Observed anomaly: On p150x4 both sampled `generate()` runs equal greedy and each other.
  Evidence: `evidence5/m4/sampling_audit.json` `sampled_equals_greedy.generate` [true, true].
  Affected path: sampled `generate()`.
  Control or comparison: chain 3b rerun 2 and chain 4 on the same prompt and mesh gave [false, false] and
  [false, true]. Sampled B1 and B4 rows differ from greedy in chain 5; the distribution test has TV 0.019 and 0
  constant rows. `generate()` changed between chains 4 and 5 only by `self._live_slots.add(0)`.
  Likely subsystem: a confident prompt; chance event.
  Investigation performed: cross-chain comparison; code diff of `generate()`.
  Resolution: controlled.

- Observed anomaly: All profiles leave HF at story token 10 (" move" against " stretch").
  Evidence: `evidence5/m*/autoregressive/autoregressive_meta.json`.
  Control or comparison: HF batched forward ranks " move" first with probability 0.264
  (`review2_fixes/calibration/hf_story_m2_teacher_forced.json`).
  Likely subsystem: HF reference noise at a near-tie.
  Investigation performed: read the HF control; recounted the prefix.
  Resolution: controlled.

- Observed anomaly: p150x2 story accepts "Seventy-two" hours per week as "Correct!".
  Evidence: `evidence5/m2/autoregressive/tt_completion.txt`.
  Control or comparison: HF teacher-forced on the TT text ranks every error token first; 243 of 256 tokens are HF
  top-1.
  Likely subsystem: model behavior.
  Investigation performed: earlier reviews verified the control; chain 5 text equals chain 4.
  Resolution: controlled.

- Observed anomaly: p150x2 qualitative p3 starts "Here's a thinking process" where HF starts "Thinking Process:".
  Evidence: `evidence5/qualitative/tt_m2_outputs.json`.
  Control or comparison: HF uses the "Here's a thinking process" opening for p2 and for the AIME24 reference.
  Likely subsystem: near-tie between two native openings.
  Investigation performed: read all 18 outputs against HF.
  Resolution: controlled (no HF margin artifact; Hard-Check Gaps).

- Observed anomaly: TT is sharper than HF at the 428-token periodic continuation (KL 0.50, 0.20, 0.42).
  Evidence: `evidence2/continuation/m4_with_kl.json`.
  Control or comparison: calibration over 100 AIME24 positions, KL mean 0.0055 to 0.0074 on all profiles.
  Likely subsystem: precision margin in periodic text (hypothesis).
  Investigation performed: calibration probe; tensors bitwise equal to chain 4.
  Resolution: controlled as isolated; cause assigned to Stage 8.

- Observed anomaly: Whole and split prefill logits differ (PCC 0.83 to 0.96).
  Evidence: `evidence5/m*/full_evidence.json` `continuation`.
  Control or comparison: HF KL 0.0002 or less at 4301, 8193 and 12417 tokens; same argmax.
  Likely subsystem: low-probability tail noise.
  Investigation performed: chain 2 HF probe.
  Resolution: controlled.

- Observed anomaly: The 2-layer probe diverges from HF on p150x2 and p150 at TT near-ties.
  Evidence: `evidence2/m*/watcher_probe.json`, `probe_control/`.
  Control or comparison: eager, traced and post-warmup logits are bitwise equal.
  Likely subsystem: build-dependent rounding.
  Investigation performed: path and watcher controls.
  Resolution: controlled for trace and warmup; residual build dependence not isolated.

- Observed anomaly: B4 and B32 step time grows linearly with batch.
  Evidence: `evidence5/m*/batch_perf.json` (p150x4: 9.18, 35.87, 282.31 ms).
  Control or comparison: `tt/functional_decoder.py:468-476` runs the layer once per row.
  Likely subsystem: decoder row loop inherited from earlier stages.
  Investigation performed: code reading and measurement.
  Resolution: controlled as a known performance limit; assigned to Stage 7 and Stage 9.

- Observed anomaly: 187 to 313 MiB of DRAM per device is not reachable from the generator object graph.
  Evidence: `memory_handoff_recheck.json` `unnamed_dram_mib`.
  Control or comparison: none.
  Likely subsystem: program-cache or CCL buffers (unverified).
  Investigation performed: recomputed the remainder.
  Resolution: residual, assigned to Stage 7.

## Scope Inspected

- Goal and skill paths:
  - `/home/hous/dev/qwen3.5-35b-a3b/runs/main/06-06-full-model.prompt.txt`
  - `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`
  - `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`
  - `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/full-model/SKILL.md`
  - `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/tt-enable-tracing/SKILL.md`
    (capture timing rules)
  - `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/qualitative-check/SKILL.md`
  - `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/prompts/model_bringup_multigoal/06-full-model.check.sh`
  - `/home/hous/dev/qwen3.5-35b-a3b/experiment-provenance-handoff.json`
- Artifact paths:
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/README.md`,
    `work_log.md`, `stage_review1.md`, `stage_review2.md`, `stage_review3.md`, `memory_handoff_recheck.json`
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/context_contract.json`
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence5/`:
    chain script and log, manifests, `source_provenance.txt`, `cpu_tests.log`, `degenerate_check.json`, and
    `m{4,2,1}/` (`full_evidence.json`, `batch_perf.json` and logs, readiness logs, `autoregressive/`,
    `inactive_rows.json`, `split_sampling_trace.json`, `sampling_audit.json`), `qualitative/`
  - `evidence4/SUPERSEDED.md`, `evidence2/SUPERSEDED.md`, `evidence2/m4/profile/`,
    `evidence3/audit_rerun2/m*/sampling_audit.json`, `evidence4/m*/sampling_audit.json`,
    `evidence4/m*/full_evidence.json`
  - `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/references/` (refpt metadata),
    `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/qualitative/hf_outputs.json`,
    `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/evidence4_work/` and `evidence5_work/` (calibration tensors)
- Code paths:
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/generator.py` (all)
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/model.py` (slot and decode
    entry points), `tt/functional_decoder.py` (decode row loop)
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_inactive_rows.py`,
    `tests/run_batch_perf.py`, `tests/test_generator_sampling.py`
  - `/home/hous/dev/qwen3.5-35b-a3b/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py` (`empty_slots`, slot moves,
    release hooks)
- Commands run (none touched a device):
  - Manifest recompute with the chain 5 `manifest()` function and `diff` against the end manifest (equal).
  - `pytest -q -p no:cacheprovider tests/test_generator_sampling.py` with `PYTHONDONTWRITEBYTECODE=1` (14 passed).
  - `06-full-model.check.sh` with `MODEL_DIR` set (exit 0).
  - Python comparisons over chain 5 JSON and logs, chain 4 JSON, the HF qualitative control, and the context
    contract; `torch.equal` over chain 4 and chain 5 calibration tensors.
  - `grep` over chain 5 logs for trace, cache and fatal messages, and over the docs tree for credentials.

## Residual Risk

- Serving under the pinned plugin depends on Stage 9 keeping every live slot active, mapping slot moves to the
  identity contract, and releasing slots. Violations now raise instead of corrupting state, except an active row on
  a never-prefilled slot.
- B32 decode gives almost no aggregate gain over B1 until Stage 7 removes the per-row layer loop. This limits the
  Stage 13 sweep at client concurrency above 1.
- p150 has 1.32 GB DRAM free and a 147 MB largest block per bank after all captures.
- The cause of the 428-token sharpening in periodic text is not isolated.
- Decode numerics of the truncated 2-layer model depend on the build; the cause is not isolated.
