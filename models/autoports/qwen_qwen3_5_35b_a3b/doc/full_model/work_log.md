# Stage 6 full model: work log

Times are Eastern (EDT). Raw run directories are under `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/`.
Preserved failed attempts are in `failed_*` subdirectories next to the runs that replaced them.

## Start (2026 Oct 12, 17:50)

- Base: Stage 5 commit `1f9fb518` (optimized multichip decoder), record commit `fb2e3b5a`.
- Decoder used: `MultichipDecoder` from `tt/multichip_decoder.py`, all 40 layers, Stage 5 default policies.
- Read: readiness `Generator` contract, `tt-enable-tracing`, `full-model`, `qualitative-check` skills.
- Sampler survey (subagent, read-only): `models/common/sampling` `SamplingGenerator` chosen.
  - Owns trace capture/replay; writes `tt_out_tok` in place; params refresh without recapture.
  - Mesh path: local top-k per device, all-gather of candidates only.
- Rejected: `models/common/modules/sampling/sampling_1d.py` (`Sampling1D`): no trace ownership; stateless.
- Logprobs: `LogProbsCalculator` supports only 8 or 32 devices; not available on 1, 2 or 4 devices.

## Reduced probe (2 layers: 0 linear, 3 full; real embedding, norm, LM head)

- `tests/run_model_probe.py`: eager prefill and decode vs an HF model built from the same two layers.
- Failure 1 (18:42): load stalled for 25 min. Cause: safetensors returns mmap-backed tensors;
  device upload of >= 8192 rows from such a tensor never completes. Random or cloned data uploads in 0.44 s.
  Fix: `_load_keys` clones tensors. Evidence: `probe/embed_*.log`, `probe/up_*.log`, `probe/failed_load_hang/`.
- Failure 2 (18:45): `ttnn.pad` asked for an 8.8 TB buffer. Cause: decoder prefill returns a view
  (logical `[1,300,H]`, padded `[1,1024,H]`); the next layer pads to 512 rows, below the padded size.
  Repro on one chip: clone, slice and `to_memory_config` keep the padded shape; a plain reshape resets it.
  Fix: `Qwen35Model._physical_view` (two view reshapes to the physical bucket, no copy).
- Failure 3: final norm rejected the L1-sharded decode output. Fix: `sharded_to_interleaved` before the norm.
- Result: top-1 matches HF on all 9 positions (prefill + 8 decode steps); logits PCC 0.989 to 0.998.

## Generator (2-layer probe, `tests/run_generator_probe.py`)

- First runs: free-running and host-sampled runs diverged from HF after the first request.
- Eager decode had the same divergence, so the trace was not the cause.
- Cause: unused page-table entries were 0. Masked prefill rows 320..511 (300-token prompt, 512-row chunk)
  were written into block 0, overwriting the prompt's first 32 tokens.
- Fix: unused entries point at the KV scratch block (`kv_pool.scratch_block`).
- After the fix: 6 requests (host and device sampling, with and without `reset()`) match HF greedy
  exactly, with `TT_METAL_TRACE_ALLOC_TRACKING=1` and program-cache misses disallowed.
- Failed runs kept: `probe/failed_page_table_alias/`.

## Full 40-layer stack: L1 and program-cache issues

- L1 clash 1 (19:23): `TTMoEGate` (models/common) keeps 4 L1 height-sharded tensors per instance
  (8 KB per bank per layer). They are read-only templates. Fix: `_share_gate_templates` shares layer 0's
  tensors after an on-device equality check. L1 flat at 41984 B/bank for 4 layers (`l1/l1_shared_m4.log`).
- Program-cache miss 1 (19:34): `EmbeddingsDeviceOperation` in the decoder planner for a new length.
  The Stage 5 planner builds per-length chunk compositions (tail pads, piece concats, `slice_write` offsets).
- L1 clash 2 (20:00): 1-token prefill steps create 4 router slices per layer (`cached_slices`).
  Fix: one shared slice cache for all layers. L1 flat at 50176 B/bank over all warmup lengths.
- Shape-stable prefill (20:17): new decoder policy `prefill_bucket_plan` (default False; the model sets it).
  Plan: 4096-row pieces plus one masked chunk that reaches the model's power-of-two bucket.
  Warmup: 15 first-chunk and 15 continuation lengths (`WARMUP_FIRST`); CPU test
  `tests/test_generator_warmup.py` proves the cover for every length 1..8192 at start 0 and 8192.
- Program-cache miss 2 (20:37): length 1 uses a separate per-token decoder path; length 2 added to warmup.
- 2-layer strict check: 40 random lengths (20 above 8192) with no program-cache miss and HF match exact.

## Full model on p150x4 (20:37)

- Smoke (`smoke/m4_full.json`): coherent thinking text for the sky prompt; load 474 s; TTFT 111 ms at
  21 tokens; token-out 9.15 ms/token; no per-token token, position or page-table refreshes.

## HF references (CPU)

- Disk offload (accelerate 1.7 with transformers 5.12) produced "Here!!!!...". Rejected; kept in
  `references/failed_disk_offload_garbage/`.
- Plain load works: BF16 safetensors stay memory-mapped page cache (no anonymous copy).
- `readiness_check.generate` breaks on transformers 5.12: `apply_chat_template(tokenize=True)` returns a
  dict-like encoding. Local shim in `tests/hf_offload.py` (`_token_list`); plugin fix candidate.
- AIME24 chat reference: `references/readiness_aime24_chat.refpt` (161-token prompt, 100 tokens, top-100).
- Book reference: `references/readiness_book_8193.refpt` (8193-token prompt, 64 tokens, top-100).

## Readiness (p150x4, before the final chain)

- Teacher forcing (20:53): top-1 99%, top-5 100%, top-100 100%; TTFT 124.9 ms; 99.9 t/s/u.
  Fix needed first: the runner loads `generator.py` by path, so the model import is absolute.
- Prefill check (21:14): top-1 99%, top-5 100%, top-100 100%.
- Split-sampling trace test, full model: 11 of 11 checks pass (`trace/m4_full.json`).

## Evidence runs before the final chain

- p150x4 (21:23): TTFT 100 ms at 128 tokens; token-out 9.16 ms; logits-only 8.58 ms; determinism and
  batch 4 and 32 match batch 1. The 262143-token prompt failed: generate asked for 16 tokens past the
  context. Fix: `generate` clips decode steps at `max_seq_len`; the run now uses 262100 tokens plus a
  separate 262143 boundary case. Kept in `evidence/failed_m4_context_overrun/`.
- p150x2 (21:38): all checks pass; 262100-token prompt prefill 91 s with the correct story continuation.
- p150, three DRAM failures (all kept in `evidence/failed_m1_dram_oom/`):
  1. KV pool did not fit after load (32.6 GB used). Cause: per-layer constants (RoPE tables, dispatch
     buffers, zero templates, index tables) were allocated per layer; the Stage 5 capacity model counted
     them once. Fix: `_share_layer_constants` (constants checked equal on device; scratch shared) and the
     KV pool is allocated before the layers.
  2. OOM while loading layer 39: 32 GDN request slots cost 2.06 GB. Fix: p150 uses 8 request slots.
  3. OOM in the all-position logits warmup (`[1,4096,V]` = 2.03 GB). Fix: `Qwen35Model.all_logits`
     computes 128-row blocks and reads each block to host.
- p150 (22:15): TTFT 185 ms; token-out 15.85 ms; batch 4 and 8 match batch 1; 262100-token prompt
  prefill 140 s with the correct continuation.

## Host reboot (about 22:30 Oct 12)

- The host rebooted (kernel 7.0.0-34 to 7.0.0-38). The in-flight p150x4 profile was lost
  (`profile_work/failed_m4_reboot/`). After the reboot host RAM shows 229 GB available; the earlier
  unexplained 178 GB is gone.
- An earlier profile attempt was killed by host OOM: Tracy captured 74.6 M zones from the full warmup.
  Fix: the profile script warms only its own prompt (`warmup_first` override). Kept in
  `profile_work/failed_m4_host_oom/`.
- Sampler traced calls pass `skip_precompile=True`; all sampling modes are warmed eagerly in `prepare()`.

## Reduced-model profile, p150x4 (10:38 Oct 13)

- `tests/run_model_profile.py` under Tracy; `tt-perf-report` over `PERF_PREFILL` and `PERF_DECODE`.
- Decode (4 token-out steps, layers 0 and 3 plus terminal path): LM-head matmul 32 x 2048 x 62080
  388 us per step (DRAM-bound); sampler `TopkLargeIndices` 286 us per step; sampler pad 31 us.
- Full model terminal cost is about 0.7 ms of 9.16 ms token-out. Sampling does not dominate.
  The top-k cost is a Stage 7 target.

## Final evidence chain 1 (from 10:39 Oct 13)

- Script: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/evidence_chain1.sh`;
  log `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/evidence_chain1.log`.
- Output: `doc/full_model/evidence1/`. Source manifests at start and end.
- Result: 29 steps, all exit code 0, `SOURCE_STABLE`; finished 14:32.
- Headline (ISL 128 / OSL 128, batch 1): TTFT 99.6 / 113.9 / 185.2 ms; token-out 9.15 / 11.34 / 15.85 ms
  (x4 / x2 / p150). Accuracy and the full tables: `README.md`.
- Book teacher forcing top-5 on x2 and p150: 63 of 64 (98.4%), at the bar.

## 2-layer probe mismatch on p150x2 and p150 (14:34 to 14:50)

- Chain watcher probe: tokens differ from HF greedy at step 2 (x2) and step 8 (p150); x4 matches.
- Control without the watcher (`probe_control/gen_m{1,2}.json`): same tokens. Not caused by the watcher.
- Eager probe with forced HF tokens (`probe_control/model_m{1,2}.json`): top-1 matches at all 9 positions.
- New diagnostic `tests/run_decode_paths_probe.py`: eager before warmup, traced, eager after warmup, traced
  again, all on the generator build. Logits bitwise equal in all four paths on x2 and p150
  (`probe_control/paths_m{1,2}.json`).
- Differing steps are ties: TT top-two margin 0.0625 (x2, step 2) and 0.125 (p150, step 8); HF ranks the TT
  choice 4th and 3rd. The eager probe builds the model differently (default request slots, KV pool allocated
  after the layers, arange page table) and breaks the same ties the other way. Which setting changes the
  rounding is not isolated.
- Conclusion: no trace, warmup or path defect; truncated 2-layer model has near-ties.

## Profiles (14:33)

- `evidence1/process_profiles.sh` on all three. Per decode step: LM head 388 us (x4), 1478 us (p150), both
  DRAM-bound at 66% and 69%; sampler top-k 287 / 567 / 1074 us. Stage 7 targets; not dominant.
- Context contract updated (`update_contract.py`, `full_model_validation`).

## Stage review 1 (Oct 13, 14:55)

- Verdict more-work-needed (`stage_review1.md`). Four P2 items:
  sampler scope accounting, unwarmed continuation starts, no real sampled-mode check, p150 8 slots.
- Correction to the 2-layer probe section above: these are not ties. HF margins at those steps are 0.625
  (x2) and 0.4375 (p150). The near-tie is on the TT side only (0.0625 and 0.125).

## p150 request slots (Oct 13, 15:00 to 15:30)

- Slot pool allocated before the layers: OOM at layer 38. The 570 MB load-time typecast needed 71 MB
  per bank; largest free block 59 MB (`evidence2/slots/failed_typecast_frag/`).
- Typecast skip when dtypes match: the dtypes differ, so the copy still runs; same OOM. Reverted
  (`evidence2/slots/failed_typecast_skip/`).
- Slots 1 to 31 allocated after all layers (`tt/model.py` `late_slots`, `_adopt_request_slots`): build
  438 s, 1.62 GB free, largest free block 178 MB per bank, " Paris." correct (`evidence2/slots/m1_32.json`).
- `DEFAULT_REQUEST_SLOTS` is 32 on every profile.

## Continuation starts (Oct 13, 15:10)

- `Qwen35Model.prefill` rejects a start that is not a multiple of 128 (`ValueError`).
- `tests/test_generator_warmup.py`: every aligned start 128 to 16256 in the test set maps to warmed
  signatures; unaligned start rejected. 6 passed.
- `run_full_evidence.py` adds whole vs split prefill cases and rejection of starts 32 and 8224.

## Sampler scope and census tools (Oct 13, 15:15)

- `tests/run_sampling_audit.py`: records every unsafe buffer created inside the sampler's
  `corruptible_allocation_scope`, with tracebacks, per bucket.
- `tests/model_census.py`: groups live device buffers by owner after build.

## Final evidence chain 2 (Oct 13, 15:34 to 20:01)

- Script `evidence2/evidence_chain2.sh`; log `evidence2/evidence_chain2.log`. 32 steps, all exit code 0,
  `SOURCE_STABLE`.
- Greedy results equal chain 1 within 1 ms. p150 now 32 slots.
- Sampler scope: 0 surviving unsafe buffers per bucket (1, 4, 32) on all profiles.

## Split vs whole prefill PCC (Oct 13, 20:05 to 20:30)

- Chain 2 continuation cases: same argmax, but full-vocabulary PCC 0.83 to 0.96 at some cases.
- HF CPU last-token logits for totals 428, 4301, 8193, 12417
  (`tests/continuation_hf_reference.py`, 3 to 97 s each).
- `tests/run_continuation_probe.py` on p150x4: whole, split and decode-fed against HF
  (`evidence2/continuation/m4_with_kl.json`).
- 4301, 8193, 12417 tokens: KL(HF || TT) <= 0.0002 on every path. Low full PCC is tail noise.
- 428 tokens: HF top-1 probability 0.926; every TT path is sharper. KL whole 0.50, split 0.20,
  decode-fed 0.42. Decode-fed is the path validated by teacher forcing, so this is precision at that
  position, not a continuation defect. Recorded for the Stage 8 datatype sweep.

## Sampling defects (Oct 13, 20:30 to 21:00)

- Chain 2 sampled audit: rows 1 to 3 of the sampled B4 run equal greedy for 48 tokens; all three meshes
  give identical sampled text.
- HF probability that sampling reproduces greedy on those rows: about 3% (`runs/main/stage06/sampling_hf/`).
- New distribution check (fixed logits, 64 sampler replays x 32 rows) and frozen-seed control.
- Defect 1, device seed never advanced: the generator never wrote the seed tensor. `manual_seed` runs
  before every draw, so each draw restarted the PRNG from the same seed.
- First fix (SeedManager entropy then SKIP): still 31 constant rows. Kept in
  `runs/main/stage06/seed_fix/failed_skip_seed/`.
- Second attempt (fresh seeds per step): still 31 constant rows. Kept in `seed_fix/failed_scalar_params/`.
- `ttnn.sampling` op test on Blackhole passes (`seed_fix/op_distribution_test.log`).
- `tests/run_sampler_op_probe.py` captured the op inputs: k = [32, 1, 1, ...], p = [1.0, 0, 0, ...].
- Defect 2, lanes: `format_sampling_params` treats a scalar temperature as one active lane and pads the
  other lanes with greedy defaults. `_set_sampling` passed scalars, so rows 1 to 31 never sampled.
- Fix in `tt/generator.py`: broadcast a scalar temperature to all 32 lanes; write fresh device seeds before
  each sampled draw (skipped when all lanes are greedy); explicit request seeds raise `ValueError`
  (they would run the sampler eagerly after capture).
- `tests/run_sampler_micro.py` (synthetic logits, eager and traced): 0 constant rows, top-1 share matches
  the expected probability on p150 and p150x4; frozen seeds give 32 constant rows. The SKIP protocol also
  works once lanes are fixed (`traced_skip`); fresh per-step seeds are kept because they do not depend on
  PRNG state surviving the decode kernels between draws.
- `tests/test_generator_sampling.py`: 3 CPU tests (lane broadcast, seed writes, seed rejection).
- Effect on earlier evidence: greedy batch-1 runs use lane 0 only, whose parameters did not change.
  Sampled evidence in chains 1 and 2 is invalid and superseded by chain 3.

## Evidence chain 3 (from Oct 13, 20:40)

- Script `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/evidence_chain3.sh`. Steps: CPU tests, 2-layer
  frozen-seed control (p150x4), and per profile sampler micro, split-sampling trace, sampling audit, full
  evidence.
- Result: 14 steps, all exit code 0, `SOURCE_STABLE`; finished Oct 13 22:27. Copied to `evidence3/`.
- Greedy outputs equal chain 2 on every profile (lengths, B4, B32, long, boundary, continuation PCC,
  audit greedy text). Only timings differ.
- Sampler micro on all profiles: fresh seeds 0 constant rows, frozen seeds 32.
- Sampling audit: p150x2 passes. p150x4 and p150 fail only `distribution_within_top_k`; TV 0.015 and
  0.020, 0 constant rows, worst sampled rank 2 and 3. Suspected BF16 ties at the 32nd candidate; the
  draws were not saved, so not verified.
- Split-sampling trace sampled leg equals greedy again. HF CPU check
  (`evidence3/hf_trace_prompt_margin.json`): at T 0.8, top-p 0.95 the nucleus holds one token at all
  7 steps; probability 1.0. Expected.

## Audit rerun, chain 3b (Oct 13, from 22:27)

- `tests/run_sampling_audit.py`: support check now tie-aware (an out-of-top-32 draw passes only if its
  logit equals the 32nd); draws, 32nd logit, tie count and out-of-support draws saved.
- Added two sampled `generate()` runs: they must differ from each other and from greedy; greedy after
  them must be identical.
- `tests/run_sampler_micro.py`: pooled TV over all 32 rows (rows share the logits).
- First launch stopped by PID at 22:29 during the CPU tests, before any device step, to add the
  `generate()` checks. Its partial directory held only a CPU test log and was removed.
- Relaunch `evidence3/evidence_chain3b.sh` into `evidence3/audit_rerun/`. `tt/` unchanged from chain 3.
- Result (Oct 13 22:29 to 23:04): all steps exit code 0, `SOURCE_STABLE`. p150x4 and p150 pass every
  check. Tie confirmed: the one out-of-top-32 draw has logit 13.125, equal to the 32nd (3 and 5 tied).
- p150x2 failed `sampled_b4_differs_from_greedy`: all 4 sampled B4 rows equal greedy for 48 steps.
  Chain 3 passed the same check on the same `tt/` source. Kept in `evidence3/failed_audit_rerun1/`.

## p150x2 sampled B4 equal to greedy (Oct 13, 23:05 to 23:20)

- New probe `tests/run_sampling_repeat.py`: greedy B4 once, then sampled B4 (T 1.0, top-k 32,
  top-p 0.95) 8 times. TT probability that a sampled row reproduces greedy, from TT logits.
- TT probability per row: 0.127, 0.334, 0.458, 0.176. All four rows: 0.0034.
- Observed per-row rate over 8 repeats: 0.25, 0.375, 0.25, 0.125. No repeat had all rows greedy.
- Conclusion: sampling works; the failure was a 0.34% chance event. The check design was weak:
  B1 alone fails with 12.7%, both `generate()` runs with about 24%.
- Fix (test only): one check `sampled_not_all_greedy` over all 7 sampled sequences (B1, B4 rows, two
  `generate()` runs). False-failure probability about 7e-6. Per-sequence results are recorded.
- Evidence: `evidence3/sampling_repeat/`. Rerun 2 of chain 3b into `evidence3/audit_rerun2/`.
- Rerun 2 result (Oct 13 23:18 to 23:53): 7 steps exit code 0, `SOURCE_STABLE`, every check passes on
  all three profiles. Sampled sequences equal to greedy: 0, 1, 1 of 7 (x4, x2, p150). Distribution TV
  0.026, 0.022, 0.015; 0 constant rows; sampler scope 0 buffers per bucket. CPU tests 1207 passed.

## Stage review 2 (Oct 13 23:54 to Oct 14 00:13)

- Verdict more-work-needed (`stage_review2.md`). Review 1 items closed. Three P2 items:
  inactive rows and fixed slots, sampling-parameter validation, two HF deviations without control.

## Review 2 fixes (Oct 14, 00:13 to 00:51)

- Sampling parameters (`tt/generator.py` `_set_sampling`): non-default presence, frequency or
  repetition penalty raises `ValueError`; so does any log-probability request. The refresh key covers
  every `SamplingParams` field, normalized to the field defaults. CPU tests: 11 passed.
- Inactive rows (`decode_forward`): a negative position marks an inactive row. The generator writes
  position 0 and an all-scratch page-table row for it, so its KV writes go to the scratch block.
- Fixed slots: decode row r is request slot r. A `request_slots` argument other than row order raises
  `ValueError`. An inactive row still updates its own GDN slot state; prefill at start 0 resets the
  slot, so a slot must not hold a partly prefilled request while its row is inactive (Stage 9 note).
- `tests/run_inactive_rows.py` on p150x4, tracker and strict cache on: 7 cases pass. B4 and B32 with
  inactive rows, through `decode_forward` and the traced loop; later prefill into a previously
  inactive slot; a request in slot 5 of B32. Every active row equals its B1 tokens
  (`review2_fixes/inactive/m4.json`).
- Calibration (`tests/run_calibration_probe.py`, HF CPU forward over AIME24 prompt plus 100 tokens,
  `review2_fixes/calibration/compare.json`):

| Profile, path | KL mean | KL max | TT minus HF top-1 logit, mean | HF top-1 prob (40 unsaturated) | TT prob of HF top-1 | Top-100 slope median |
|---|---:|---:|---:|---:|---:|---:|
| p150x4 prefill | 0.0072 | 0.122 | -0.13 | 0.746 | 0.742 | 1.00 |
| p150x4 decode | 0.0070 | 0.155 | -0.19 | 0.746 | 0.749 | 1.00 |
| p150x2 prefill | 0.0063 | 0.134 | -0.16 | 0.746 | 0.740 | 1.00 |
| p150x2 decode | 0.0074 | 0.186 | -0.18 | 0.746 | 0.742 | 1.00 |

- Conclusion: no systematic over-confidence or logit scale error. The 428-token case is isolated.
- p150x2 story ("Seventy-two" hours per week, "Correct!"): HF teacher-forced on the TT completion
  ranks every token of the error first ("Se" 0.29, "-two" 0.98, "Correct" 0.25, "Not" 0.43). Over
  the whole completion 243 of 256 tokens are HF top-1, max rank 2. Model behavior, not a TT defect.
- Final chain 4 (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/evidence_chain4.sh`) launched
  00:51 into `evidence4/`: all affected gates on the final source, all profiles. Watcher and
  profiler are not rerun: the changes are host-side only.

## Chain 4 results (Oct 14, 00:51 to 06:11)

- 36 steps, all exit code 0, `SOURCE_STABLE`. CPU tests: 1215 passed.
- Accuracy, autoregressive and qualitative outputs equal chain 2 on all profiles (byte-identical text).
- Full evidence: 46 boolean checks pass per profile; token results equal chain 3; census and allocator equal.
- Headline: p150x4 100.8 ms TTFT, 9.19 ms per token; p150x2 113.8, 11.38; p150 183.4, 15.84.
- Sampling audit, micro, split-sampling trace and frozen-seed control pass on all profiles.
- Inactive rows: 7 cases pass on all profiles; 3 decode captures.
- Calibration now includes p150: KL mean 0.0059 (prefill), 0.0055 (decode). p150x4 and p150x2 logits are
  bitwise equal to the review 2 fix runs.
- Contract and memory recheck regenerated from evidence4 (`evidence4/update_contract4.py`,
  `evidence4/memory_handoff_recheck4.py`). Only artifact paths, long-run seconds and the generator hash changed.
- Superseded directories labeled in place (`evidence1/`, `evidence2/`, `evidence3/` `SUPERSEDED.md`).
- `evidence4/source_provenance.txt`: HEAD and clean status of reused modules.

## Review 3 and fixes (Oct 14, 06:14 to 06:42)

- Review 3 (`stage_review3.md`): more-work-needed, two P2 items.
- P2 1: `prefill_forward` and `decode_forward` dropped unknown keywords. The plugin sends
  `empty_slots` (prefill) and `slot_remap` (decode). Fix: no catch-all keywords; `empty_slots`
  selects the prefill slots; `slot_remap` must be the identity; unknown keywords raise `TypeError`.
- P2 2: an inactive row overwrites its slot's GDN state. Fix: the generator tracks live slots;
  a decode that marks a live slot inactive raises `ValueError` before device work;
  `release_slots` and `reset` clear slots.
- Also: an unprepared decode batch raises `ValueError` (was `KeyError`).
- CPU tests: 1218 passed.
- `tests/run_inactive_rows.py` releases slots after each case and adds a case that tries to hide
  a live slot at step 5, expects `ValueError`, then resumes and must match B1.
- New `tests/run_batch_perf.py`: B1, B4, B32 token-out with the tracker off, and allocator
  memory after all six trace captures.
- Chain 5 (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/evidence_chain5.sh`, PID 1031623)
  launched 06:42 into `evidence5/`: chain 4 gates plus batch perf, on the final source.

## Chain 5 results (Oct 14, 06:42 to 12:37)

- 39 steps, all exit code 0, `SOURCE_STABLE`. Source: `tt/generator.py` sha256 `4140f40c...`.
- CPU tests: 1218 passed.
- Full evidence: 46 of 46 boolean checks per profile. Every token result equals chain 4.
- Accuracy, autoregressive, qualitative, sampler micro, split-sampling trace: equal to chain 4.
- Calibration logits bitwise equal to chain 4 on all profiles; `compare.json` identical.
- Sampling audit: all checks pass. Sampled counts differ from chain 4 (fresh seeds).
- Inactive rows: 8 cases pass; `hidden_live_slot_rejected` true on all profiles.
- Batch perf (tracker off): B32 aggregate 113.3, 93.0 and 68.7 t/s. Flat with batch.
- Trace capture DRAM: 250, 275 and 298 MB per device.
- Contract and memory recheck regenerated from evidence5; evidence4 marked superseded.

## Review 4 and commit (Oct 14)

- Review 4 (`stage_review4.md`): clean-pass, no required work.
- Doc fixes from its concerns: tracker-off note, trace label, stale numbers.
- Contract now records DRAM after all trace captures.
- Stage 9 notes add chunked prefill and never-prefilled slot cases.
- Local commit `dfe0c6bd3c71f39853bdfc052787d5db67c465b0` on `feat/qwen3-5-35b-a3b`; not pushed.
- Committed tt/ and tests/ hashes equal `evidence5/source_manifest_end.txt`.
- Raw `ops_perf_results.csv.gz` dumps left out; perf reports and sha256 kept.
