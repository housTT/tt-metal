# Stage Review

Verdict: more-work-needed

Stage: 6, full model. Reviewer: independent stage-review subagent, read-only. Review date: 2026 Oct 13.

Abbreviations: HF (Hugging Face CPU reference), TT (Tenstorrent TTNN implementation), KV (key/value cache), GDN
(Gated DeltaNet linear-attention layer), TP (tensor parallel, mesh width), LM head (final vocabulary projection),
TF (teacher forcing), DRAM (device memory), RNG (random number generator), API (application programming
interface), CPU (host processor).

The core full-model path is in good shape. All 40 layers run on p150x4, p150x2 and p150. The AIME24 and book
references are fresh. Teacher forcing and the prefill check pass the top-5 >= 98% and top-100 = 100% bar on all
three profiles. The qualitative suite runs with the chat template and an HF control. The split-sampling trace is
real: tokens feed back through `tt_out_tok`, positions advance through `plus_one` inside the trace, and the
page table is written only on change. The tracker runs keep program-cache allocations in scope, and the source
manifests prove the chain ran on stable source. The stage gate passes.

Four contract items are still open. Each one is a gap in trace-safety accounting, public-API preparation,
sampling evidence or capacity evidence. None of them shows wrong output in the delivered greedy path.

## Required Work

- P2: The sampler's trace-allocation exemptions are not accounted for. The README lists only one acknowledgment.
  Evidence: The chosen common sampler wraps its whole capture window (begin capture, run sampling, end capture,
  synchronize) in `trace_allocation_tracker.corruptible_allocation_scope(self.mesh_device)`
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/common/sampling/generator.py:601`). When a trace bucket is
  set, it also acknowledges `(logits, output)` as corruptible (line 624). `Qwen35Generator._replay` calls
  `sampler.set_trace_bucket(batch)` before every traced sample, so this branch always runs. `output` contains
  `tt_out_tok`, which is the persistent decode token input `generator.tt_tokens`
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/generator.py`, `_replay`).
  The README section "Trace allocation safety" names one acknowledged tensor only: the decode trace logits.
  Why this matters: `$tt-enable-tracing` requires the stage to list each acknowledged backing allocation with
  its aliases, consumers and lifetime invariant against every live trace. It also requires each allocation scope
  to be explained and its enclosed allocations identified. The stage-review standard treats a scope over an
  entire capture as more-work-needed. The delivered path keeps six live traces: decode for B1, B4 and B32, and
  one sampler trace per bucket. Nothing in the evidence shows what the sampler scope hides.
  Required next step: Run one tracker run with `TT_METAL_TRACE_ALLOC_TRACEBACKS=1`, or take an allocator
  snapshot before and after each sampler capture (buckets 1, 4 and 32). List every allocation that survives the
  scope. If none survive apart from trace-owned outputs, record that. Then record the lifetime invariants for
  `tt_tokens` and for each bucket's logits across all six traces, in the generator's replay order (decode_b,
  then sampler_b, then token read). Include the `decode_forward(sampling_params=None)` logits-readback path.

- P2: The public `prefill_forward(start_pos=...)` accepts start positions that warmup does not prepare.
  Evidence: The new `prefill_bucket_plan` branch in `_chunk_plan`
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py`)
  runs only when `start_pos % 128 == 0`. Any other start uses the Stage 5 planner, which builds 32-row and
  64-row piece compositions. `Qwen35Generator.prefill_forward` passes any caller `start_pos` through without a
  check. `tests/test_generator_warmup.py` enumerates only starts 0 and 8192. I ran a CPU-only probe with the
  stage's own planner and warmup helpers (`_chunk_plan`, `WARMUP_FIRST`). For block-aligned starts 32, 64, 96
  and 8224, 1156 to 1165 of about 1160 sampled lengths map to compositions outside the warmed set. For 661 to
  935 of those lengths, the planned rows are also below the model's physical bucket, so the run reaches the
  unwarmed `ttnn.pad` fallback in `Qwen35Model._physical_view`. Starts 4096 and 16384 have 0 unwarmed
  signatures. In strict mode (`QWEN35_STRICT_CACHE=1`) these inputs raise a program-cache miss. Without strict
  mode, they compile and allocate after trace capture. The evidence has no caller-driven continuation chunk.
  Why this matters: The goal contract requires serving-ready low-level prefill with explicit position state.
  `$tt-enable-tracing` requires warmup of every supported prefill variant before the first capture. It also
  requires explicit coverage of continuation chunks at different start positions. With a 32-token KV block,
  chunked or prefix-cached serving produces starts that are not multiples of 128. The stage changed the
  chunking rule, so the sibling branch needs coverage.
  Required next step: Preferred fix: extend the bucket plan to block-aligned starts, for example with a warmed
  alignment piece followed by buckets. Then add those starts to warmup and to the CPU cover test. Run one device
  check under the tracker and strict mode: a first chunk, continuations at starts that are not multiples of 128,
  and a partial final chunk. Each must match the single-call result. If you do not take that route, reject
  unsupported starts with a clear error at the public API and record the contract for Stage 9.

- P2: There is no evidence that sampled (top-k/top-p) token-out actually samples, and on x2 and x4 there is no
  check that every device gets the same sampled token.
  Evidence: In `evidence1/m{4,2,1}/split_sampling_trace.json`, `tokens_sampled` equals `tokens_greedy`
  (`[90700, 8340, 25, 271, 16, 13, 220]`, "Thinking Process:\n\n1.  ") on all three profiles. The check
  `sampled_tokens_valid` only tests `0 <= t < vocab`. The settings are temperature 0.8, top_p 0.95 and top_k 32.
  On these high-confidence opening tokens the 0.95 nucleus can collapse to one token, so a greedy-only path also
  passes. No other evidence run uses non-greedy sampling. `read_tokens` reads the device-0 replica only.
  Why this matters: `$full-model` requires the same traced path to support top-k/top-p sampling. It forbids
  leaving sampled serving to the vLLM stage. On TP meshes, each device embeds its own copy of the sampled token
  for the next step. Replicas that disagree under non-greedy sampling would corrupt decode without any visible
  error.
  Required next step: On each profile, add a sampled-mode check with an open-ended chat prompt. Use temperature
  1.0, top_k 32, top_p 1.0 (or 0.9) and at least 32 tokens, over two runs. Assert that the output differs from
  greedy. Assert that every sampled token is inside the TT top-32 at its step. On x2 and x4, assert that all
  device replicas of `tt_tokens` agree after every step. Then confirm that greedy returns the original greedy
  sequence.

- P2: The p150 reduction to 8 request slots is not proven to be a hard physical limit, and the full-model
  memory-handoff recheck is missing.
  Evidence:
  - The note in `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/context_contract.json`
    (`full_model_validation.notes`) says that 32 slots, costing 2.06 GB, "do not fit" in 34.18 GB.
  - With 8 slots, `evidence1/m1/full_evidence.json` measures 30,946,798,080 B allocated and 3,231,932,928 B free
    per device after build.
  - Going from 8 to 32 slots adds 24 x 30 x 2,146,304 B = 1.545 GB. That gives about 32.49 GB, which is below
    34.18 GB.
  - The measured failure
    (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/evidence/failed_m1_dram_oom/m1_after_sharing.log`) is a
    302 MB load-time `ttnn.concat` while loading layer 39. At that moment each bank had 228 MB free (1.82 GB
    total), but the largest free block was 22.8 MB per bank. That is fragmentation from a transient during load,
    not steady-state capacity.
  - The Stage 5 capacity model (`doc/optimized_multichip_decoder/README.md`, "Memory and capacity") predicted
    1.01 full-context sequences with 32 slots on p150.
  - No 16-slot or 24-slot probe exists.
  - `AGENTS.md` and the `memory-layout-handoff.json` "required_before_full_model_selection" list require a recheck
    after implementation changes, including per-buffer root ownership after construction. Stage 6 adds
    `_share_layer_constants`, which replaces references through `gc` and frees duplicates. Stage 5 has
    `memory_handoff_recheck.json`, but Stage 6 has no recheck artifact. Allocator totals are the only proof that
    the per-layer copies were freed.
  Why this matters: GDN request slots are cache/state capacity. The stage-review standard requires a hard
  physical limit and the largest feasible value before such a capacity is reduced. `$full-model` requires the
  largest tested batch plus the hard limit. Stage 13 has to disclose the real server slots. The contract note's
  arithmetic contradicts the stage's own measurements.
  Required next step: Allocate the GDN slot state before the layer weights, as the stage already does for the KV
  pool, or remove the load-time fragmentation another way. Then try 32 slots on p150: build, run the
  262100-token prefill, and decode at B8 and B32. If 32 slots fail, search for the largest slot count that works
  and record the failing peak. Correct the contract note. Add a Stage 6 memory-handoff recheck for all three
  profiles. It should cover per-buffer ownership after construction, confirmation that superseded per-layer
  constants were freed, and construction peaks.

## Other Concerns

- Log-probabilities are not available on 1, 2 or 4 devices. `LogProbsCalculator._is_supported` returns False
  unless the mesh has 8 or 32 devices (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/common/sampling/tt_log_probs.py:422`).
  The Stage 13 latency methodology (`/home/hous/dev/qwen3.5-35b-a3b/evaluation/latency-methodology.json`)
  requires `/v1/completions` with `logprobs=0`. The README lists the limitation but does not connect it to that
  downstream requirement. Decide on a path before Stage 9.
- Inactive rows are untested. The decode trace uses `plus_one(..., skip_negative_entries=True)`, but no run
  passes a position of -1. GDN layers update the slot state for every row, padded rows included.
  `decode_forward` has no `request_slots` argument, so row r is always slot r. In Stage 9, chunked prefill that
  runs between decode steps would corrupt a slot whose row is padded. Record this contract now.
- Penalty programs are not warmed. Enabling presence, frequency or repetition penalties after capture would
  compile and allocate late.
- The `prepare()` warmup with top-k at temperature 0.7 does not restore sampler RNG state, which
  `$tt-enable-tracing` asks for.
- Force-argmax was not benchmarked. Local top-k costs 287, 567 and 1074 us per step on x4, x2 and p150 (3.1%,
  5.0% and 6.8% of token-out). This is not dominant. It is reasonable to defer it to Stage 7, but record the
  measurement plan.
- The rejection rationale for `Sampling1D` is partly wrong. `sampling_1d.py` accepts `tt_out_tok` and takes
  k/p/temperature per call. The real difference is trace and state ownership. Correct the record; the selection
  can stay.
- The README defines top-k accuracy backwards. In `readiness_check/teacher_forcing.py` `compute_accuracy`, top-k
  is "the TT predicted token is inside the HF top-k", not "the HF token is inside the TT top k".
- The README says the p150x2 LM head is "not in the device-0 CSV". This is a reporting artifact: the
  `tt-perf-report` CSVs mix rows from several devices (x4 decode has rows from devices 0 to 3). The x2 LM head
  `32 x 2048 x 124160` takes 738 us per step in the device-1 rows. The device-0 `summary.json` matmul total is
  834 us per step. Report the x2 LM head and its share.
- The qualitative verdict overstates the outputs. At 128 tokens most outputs end inside the thinking section.
  For p5, the x4 and x2 outputs end before any code, and p150 stops at "def fibonacci(n". The text is coherent
  and matches the HF style, so the pass verdict holds, but correct the claim.
- For the 2-layer probe, the README calls the differing steps "ties". The HF margins there are 0.625 (x2, step 2)
  and 0.4375 (p150, step 8). The near-tie is in TT only. The eager model probe and the generator build give
  different logits for the same inputs, and the cause (page-table width, KV pool size or allocation order) is not
  isolated.
- `decode_forward` builds a host page table of shape [batch, 8192] and compares it on every call. This is a
  per-step host cost on the serving path (Stage 9 and 10).
- `_check_kv` rejects any KV cache the caller allocates. The vLLM adapter must return `generator.kv_pool.caches`
  from its KV allocation. Record this for Stage 9.
- `STAGE6_RUNNER=1` in `evidence1/evidence_chain1.sh` has no effect. Readiness runs on p150 use the default
  single device, while model scripts use physical device 3. Both are single-chip p150, so this is only a
  labeling issue.

## Hard-Check Gaps

- The stage gate `06-full-model.check.sh` checks only degenerate output and the context contract. It exits 0 here.
  It does not check trace-safety accounting, sampled mode, prefill start coverage or slot capacity.
- `sampled_tokens_valid` is only a range check.
- `test_generator_warmup.py` asserts warmup cover only for starts 0 and 8192.
- Logit determinism across batch positions is checked only by token equality over 8 tokens.
- HF accuracy is checked up to 8193 tokens. The 262100-token check uses periodic text, and its next tokens appear
  in the last ~60 tokens, so it shows the path runs, not long-range attention accuracy.
- The HF references are noisy. I compared `generated_tokens` (incremental HF generate) with `topk_tokens[:, 0]`
  (batched HF forward) in the reference files. They disagree at 12 of 64 book positions and 1 of 100 AIME24
  positions. BF16 HF numerics therefore limit how well top-1 can discriminate on the book reference. An FP32 HF
  reference would be stronger.
- The watcher ran only on the 2-layer model. That is acceptable, but the 40-layer stack has no watcher pass.

## Anomaly Ledger

- Observed anomaly: Free-running output diverges from HF at token 10 on all three profiles ("move faster than"
  versus "stretch and shrink").
  Evidence: `evidence1/m{4,2,1}/autoregressive/autoregressive_meta.json`; the TT text is coherent.
  Affected path: token-out decode, story prompt without chat template.
  Control or comparison: Book TF top-1 is 89 to 92%. The HF reference itself disagrees between incremental and
  batched modes at 12 of 64 book positions (reviewer analysis), so narrative text has many near-ties.
  Likely subsystem: precision margin, not feedback. Feedback is checked separately by the split-sampling trace
  test.
  Investigation performed: The stage read the outputs and did not record the HF logit margin. The reviewer
  analyzed the reference files.
  Resolution: controlled. Residual: the exact HF margin at that position is not recorded.

- Observed anomaly: The 2-layer probe differs from HF greedy at step 2 (x2) and step 8 (p150).
  Evidence: `evidence1/m{2,1}/watcher_probe.json`, `probe_control/{gen,paths,model}_m{1,2}.json`.
  Affected path: reduced 2-layer generator build.
  Control or comparison: A run without the watcher gives the same tokens. Host and device sampling give the same
  tokens. Eager and traced logits are bitwise equal before and after `prepare()`. The eager model probe (a
  different build) matches HF top-1 at every step.
  Likely subsystem: numerics that depend on the build (page-table width, KV pool, allocation), amplified by MoE
  routing.
  Investigation performed: path, watcher and sampling controls.
  Resolution: controlled for trace, warmup, watcher and sampling. The cause of the build dependence is not
  isolated (Other Concerns).

- Observed anomaly: The p150x2 qualitative p3 output differs from HF at the first token ("Here's a thinking
  process" versus "Thinking Process:").
  Evidence: `evidence1/qualitative/tt_m2_outputs.json`; HF control
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/qualitative/hf_outputs.json`.
  Affected path: chat-template greedy generation.
  Control or comparison: HF itself opens p2 and the AIME24 reference with "Here's a thinking process", so both
  openings are native to the model. The rest of the x2 output is correct.
  Likely subsystem: a near-tie between two learned openings.
  Investigation performed: the reviewer read all outputs against the HF control.
  Resolution: controlled.

- Observed anomaly: Sampled-mode tokens are identical to greedy tokens.
  Evidence: `evidence1/m*/split_sampling_trace.json`.
  Affected path: traced sampler with non-greedy parameters.
  Control or comparison: none.
  Likely subsystem: test design, because the 0.95 nucleus collapses on high-confidence tokens. A parameter
  refresh failure is not excluded.
  Investigation performed: none.
  Resolution: more-work-needed (Required Work P2).

- Observed anomaly: The p150 build with 32 slots fails with OOM even though about 1.8 GB is free.
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/evidence/failed_m1_dram_oom/m1_after_sharing.log`.
  Affected path: model construction on p150.
  Control or comparison: 8-slot build, 3.23 GB free. Stage 5 capacity model.
  Likely subsystem: DRAM fragmentation during load.
  Investigation performed: the stage reduced slots to 8.
  Resolution: more-work-needed (Required Work P2).

- Observed anomaly: "Allocating device buffers is potentially unsafe due to the existence of an active trace"
  appears in the watcher, profile and probe-control logs.
  Evidence: `evidence1/m*/watcher_probe.log`, `evidence1/m*/profile.log`, `probe_control/*.log`.
  Affected path: prefill after the first capture, in runs without the tracker.
  Control or comparison: Runs with the tracker on and the same multi-request patterns (evidence, trace and
  readiness runs) pass, and their logs contain no such warning.
  Likely subsystem: transient prefill allocations.
  Investigation performed: the stage compared runs with and without the tracker.
  Resolution: controlled, subject to the sampler-scope accounting in Required Work.

- Observed anomaly: On x2, AIME24 TF top-1 is 96% versus 99% (x4) and 98% (p150), and the 2-layer probe
  diverges earliest on x2.
  Evidence: `evidence1/m2/teacher_forcing_aime24.log`, `evidence1/m2/watcher_probe.json`.
  Affected path: p150x2 numerics.
  Control or comparison: top-5 is 100% and the prefill check is 97%. The difference is a few positions out of 100.
  Likely subsystem: collective reduction order or precision margin.
  Investigation performed: none.
  Resolution: residual risk. This is not a blocker on its own.

## Scope Inspected

- Goal and skill paths:
  - `/home/hous/dev/qwen3.5-35b-a3b/runs/main/06-06-full-model.prompt.txt`
  - `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`
  - `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/{stage-review,full-model,tt-enable-tracing,qualitative-check}/SKILL.md`
  - `/home/hous/dev/qwen3.5-35b-a3b/memory-layout-handoff.json`
  - `/home/hous/dev/qwen3.5-35b-a3b/evaluation/latency-methodology.json`
- Artifact paths:
  - `doc/full_model/README.md`, `doc/full_model/work_log.md`
  - `doc/full_model/evidence1/`: chain script and log, source manifests, cpu_tests, degenerate check,
    `m{4,2,1}` readiness logs, autoregressive completions, `full_evidence.json`, `split_sampling_trace.json`,
    watcher probes, profile summaries and perf-report CSVs
  - `doc/full_model/evidence1/qualitative/`
  - `doc/full_model/probe_control/`
  - `doc/context_contract.json` (diff)
  - `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/{references,qualitative,evidence/failed_m1_dram_oom,probe,probe_control}/`
  - `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/evidence1_work/watcher_m*/generated/watcher/watcher.log`
- Code paths:
  - `tt/model.py`, `tt/generator.py`, and the `tt/optimized_decoder.py` diff against HEAD
  - `tt/functional_decoder.py` (slots, indices)
  - `tests/run_full_evidence.py`, `tests/run_split_sampling_trace.py`, `tests/run_qualitative.py`,
    `tests/hf_offload.py`, `tests/test_generator_warmup.py`, `tests/test_prefill_plan.py`,
    `tests/runtime_audit.py`, `tests/run_decode_paths_probe.py`, `tests/run_model_probe.py`,
    `tests/mesh_util.py`, `tests/process_profiles.py`
  - `models/common/sampling/generator.py`, `models/common/sampling/tt_log_probs.py`,
    `models/common/modules/sampling/sampling_1d.py`
  - `tt_metal/distributed/mesh_trace.cpp` (dynamic trace region), `ttnn/ttnn/tools/trace_allocation_tracker.py`
  - readiness `contract.py`, `generate.py`, `teacher_forcing.py`
- Commands run (none touched a device):
  - Stage gate: `MODEL_DIR=models/autoports/qwen_qwen3_5_35b_a3b PYTHONPATH=<plugin runtime> bash .../06-full-model.check.sh`
    from `/home/hous/dev/qwen3.5-35b-a3b/tt-metal`. Exit code 0. Output: "No degenerate output detected." and
    "Context contract OK ... target=262144, supported=262144 (full HF context)."
  - Comparison of the current source manifest with `source_manifest_end.txt`. Only `tests/run_decode_paths_probe.py`
    was added after the chain, as the README states.
  - Python reads of the evidence JSON, the reference `.refpt` files and the HF references.
  - CPU-only planner and warmup-cover probes using the stage's test helpers (scripts in the session scratchpad).
  - `grep` and `awk` over logs and perf-report CSVs.

## Residual Risk

- Long-context accuracy beyond 8193 tokens is shown only by a continuation of periodic text.
- Decode numerics depend on the build (2-layer probe), and the cause is not isolated.
- Serving contracts are not yet defined: inactive rows, the slot-to-row mapping, penalties, log-probabilities, and
  KV ownership through the vLLM adapter.
- The x2 top-1 rate is slightly lower than the other profiles, within noise at this sample size.
