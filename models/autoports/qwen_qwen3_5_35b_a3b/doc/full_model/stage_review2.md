# Stage Review

Verdict: more-work-needed

Stage: 6, full model, second review. Reviewer: independent stage-review subagent, read-only. Review date: Oct 14.

Abbreviations: HF (Hugging Face CPU reference), TT (Tenstorrent TTNN implementation), KV (key/value cache), GDN
(Gated DeltaNet linear-attention layer), MoE (mixture of experts), TP (tensor parallel), EP (expert parallel), LM
head (final vocabulary projection), RoPE (rotary position embedding), TF (teacher forcing), DRAM (device memory), L1
(per-core static RAM), CCL (collective communication library), BF16 (16-bit brain float), KL (Kullback-Leibler
divergence), PCC (Pearson correlation coefficient), TV (total variation distance), PRNG (pseudo-random number
generator), CPU (host processor), RSS (resident set size of the host process), API (application programming
interface), MiB (2^20 bytes). B1, B4 and B32 mean a decode batch of 1, 4 and 32 rows.

All four required items from review 1 are resolved by evidence, not prose:

- Sampler scope accounting. The audit replaces `corruptible_allocation_scope` with a recorder, so allocations are
  tracked and not suppressed. It finds 0 surviving buffers for buckets 1, 4 and 32 on all profiles
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/audit_rerun2/m{4,2,1}/sampling_audit.json`).
  The README lifetime table matches the code (`logits_b`, `tt_tokens`, seed tensor, sampler scope).
- Continuation starts. `Qwen35Model.prefill` rejects starts that are not multiples of 128. Strict-cache and
  tracker runs pass continuation cases at starts 128 to 12288.
- Sampled mode. The stage found and fixed two real sampler defects (frozen device seed, greedy lanes 1 to 31). The
  final audit shows varied, coherent sampled text, TV 0.015 to 0.026 against the host softmax, ranks below 32, and
  device copies that agree. A frozen-seed control fails the same checks.
- p150 slots and memory recheck. All profiles now have 32 GDN slots, full 262144-token context, and a census-based
  handoff recheck whose source hashes match the current `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/` files.

I also verified these claims directly:

- Chain 2 and chain 3 greedy outputs are identical on all profiles. Only `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/generator.py` changed between the two
  chains. The greedy lane parameters are identical before and after the fix: temperature 1.0, k 1, p 0.0, and no
  seed write. So the chain 2 accuracy evidence applies to the final source.
- Every README headline, accuracy, sampling, memory and profile number that I checked matches its JSON or log.
- The stage gate passes. CPU tests pass (1207). Source manifests are stable and equal to the current tree.

Three new or carried-forward items still block the stage. Each one is a serving-path or quality contract gap. None
of them shows wrong output in the delivered greedy B1 path.

## Required Work

- P2: The inactive-row and fixed-slot decode contract is untested, and the public decode API silently drops a slot
  mapping.
  Evidence:
  - The goal requires serving-ready low-level decode with "fixed slots, and inactive rows where used".
  - `decode_forward` accepts only the prepared decode buckets (`decode_batches`: 1, 4 and 32 in the evidence runs;
    the default is 1 only)
    (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/generator.py:246-248`).
    Any other active count must be padded with inactive rows. So inactive rows are part of the design, not optional.
  - `_decode_device` hard-codes `request_slots=tuple(range(batch))` (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/generator.py:134-135`). Row r is always GDN
    slot r. A request prefilled in slot 5 can decode only in row 5 of a B32 batch.
  - `decode_forward(..., **kwargs)` (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/generator.py:303-304`) silently swallows a `request_slots` argument. A caller
    that passes one gets slot 0..b-1 state with no error.
  - The trace advances positions with `plus_one(..., skip_negative_entries=True)` (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/generator.py:136`), which implies
    -1 marks an inactive row. But no run passes -1.
  - In the decode path, each full-attention row computes RoPE through `_rotary`, which typecasts the position to
    uint32 and indexes an embedding table with no clamp
    (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/functional_decoder.py:301-308`,
    called at line 473). What -1 does there is unverified.
  - Linear-attention rows update their slot state for every row, inactive rows included
    (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/functional_decoder.py:468-476`).
  - The README lists "Inactive rows are untested" as a Stage 9 note. Review 1 raised the same gap.
  Why this matters: `$full-model` treats the generator as a serving component whose callers own fixed request slots
  and inactive rows. The stage-review standard requires the goal's implementation paths to be present and checked.
  An inactive row that reads out of range, or that corrupts a live slot's GDN state, breaks serving without any
  visible error.
  Required next step:
  1. Choose the inactive-row convention. Either guard the -1 path (clamp the RoPE index, and make sure the paged
     update and the decode attention skip that row), or document "position 0 plus an all-scratch page-table row" and
     reject negative positions at the API.
  2. Either add an explicit row-to-slot map to `decode_forward` and the decode trace, or reject unknown kwargs and
     document "row == slot" as the fixed-slot contract.
  3. Run one device check under the tracker and strict cache on p150x4 and p150. It must cover:
     - B4 and B32 decode, both through `decode_forward` and through the traced token-out loop, with some rows
       inactive;
     - active rows that equal their B1 tokens;
     - no device fault;
     - a later prefill into a previously inactive slot that reproduces its B1 output;
     - a request in a non-zero slot (for example slot 5 in B32) that decodes correctly while the other rows are
       inactive.

- P2: Sampling-parameter handling silently accepts sampler variants that are not prepared, and the mode cache key
  ignores them.
  Evidence:
  - `_set_sampling` keys its refresh only on temperature, top-k and top-p (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/generator.py:112-119`).
  - Presence, frequency and repetition penalties, `enable_log_probs` and `num_logprobs` pass straight through to
    `SamplingGenerator.reset_sampling_params`
    (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/common/sampling/generator.py:354-386`).
  - Consequence 1: a request with penalties, followed by a request with the same temperature, top-k and top-p but
    default penalties, does not refresh the sampler. Penalties stay active for the second request.
  - Consequence 2: the reverse order ignores the new penalties.
  - When penalties or log-probabilities do turn on, the sampler selects a new trace key (`_TraceKey.penalties_on` or
    `log_probs_on`). `_replay` then captures that key at request time with `skip_precompile=True`
    (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/generator.py:267`). `prepare()` never compiled it (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/generator.py:224-233` warms only greedy and top-k). The
    result is a program-cache miss during capture: an error in strict mode, and a compile inside capture otherwise.
  - The generator never sets the penalty prompt or output token state (`reset_prompt_tokens`,
    `reset_output_state`), so penalties would also be applied to empty counts.
  - Log-probabilities are unsupported on 1, 2 and 4 devices. The flag still changes the trace key.
  - Only explicit seeds are rejected today (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/generator.py:107-109`).
  Why this matters: sampling is a stage-critical subsystem, and the generator is the serving component. `$tt-enable-tracing`
  forbids compiling or capturing unprepared variants during requests. A silently stale parameter state gives wrong
  sampled output with no error. The README calls the penalty gap "not warmed", which understates it.
  Required next step:
  - Preferred: in `_set_sampling`, reject non-default penalties and any log-probability request with a clear
    `ValueError`, as is done for seeds. Make the mode key cover every sampler-affecting field.
  - Alternative: wire the penalty token state, warm the penalty and log-probability variants in `prepare()`, and test
    them on device.
  - Add CPU tests in
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/test_generator_sampling.py`
    for: rejection or refresh when only a penalty field changes; rejection of log-probabilities; and no stale state
    across requests.

- P2: Two TT-versus-HF quality deviations are not classified by a control.
  Evidence:
  1. Over-confidence at the 428-token position. I read the saved logits
     (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/continuation/m4.pt` and `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/continuation/hf_last_logits.pt`). On p150x4:
     - TT puts 0.996 to 0.9999 on the top token on all three paths. HF puts 0.926.
     - KL(HF, TT) is 0.50 (whole prefill), 0.20 (split) and 0.42 (decode-fed).
     - The TT top-1 logit is 3.4 to 5.3 above HF (25.4, 23.9 and 25.8 against 20.5).
     - The slope of TT against HF over the HF top-100 logits is 1.01 to 1.02. On the split and decode-fed paths the
       second logit also matches HF (17.6 and 17.1 against 17.25). So mainly the winning token is inflated.
     - At the other three positions the TT top-1 logit is also above HF in every path (+0.4 to +1.9). Those
       positions are saturated (HF top-1 at least 0.9998), so KL cannot show the effect there.
     - The README calls this "a precision effect at that position" and defers it to Stage 8. No precision control
       or wider measurement supports that claim. Every accuracy gate measures rank only, so it cannot see this.
  2. A visible content error in p150x2 free-running text. The p150x2 completion
     (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence2/m2/autoregressive/tt_completion.txt`)
     contains: "how many hours are in a week?" / "Seventy-two" / "Correct!". Just before that, the elder answers
     "Not quite" to the correct "seven times". HF diverged at token 10, so the HF control does not cover this text.
     p150x2 also has the lowest AIME24 TF top-1 (96%). The README calls all outputs coherent and does not mention
     the error.
  Why this matters:
  - The stage-review anomaly rule says a deviation visible in model behavior needs a control or a fix.
  - Item 1 directly changes sampled serving: at temperature 1.0, HF picks a non-top token 7.4% of the time here and
    TT 0.01% to 0.4%, depending on the path. The later datatype sweep selects precision by top-k accuracy only, so it
    will not catch this either.
  - Item 2 is cheap to classify.
  Required next step:
  1. On p150x4 and p150x2, run `prefill_logits` over a non-periodic chat sequence, for example the AIME24 reference
     prompt plus its 100 tokens. Run one HF CPU forward over the same tokens. Report per-position KL(HF, TT), the
     TT-minus-HF top-1 logit, and top-1 probability quantiles.
  2. If the over-confidence is systematic, run one precision control before naming the cause, for example a BF16 LM
     head or BF16 KV on the 428-token case. If it is isolated, record the distribution and close the item.
  3. Run an HF CPU forward teacher-forced on the p150x2 completion. Record HF's rank and probability for " Seventy",
     "Correct" and "Not quite". If HF ranks them at or near the top, record the error as model behavior.

## Other Concerns

- The headline performance was measured with the tracker and strict cache on. `run_full_evidence` runs under
  `TT_METAL_TRACE_ALLOC_TRACKING=1 QWEN35_STRICT_CACHE=1`
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/evidence_chain3.sh`).
  `ttnn.execute_trace` then calls `verify_before_replay` on every replay, and token-out has two replays per token.
  `$tt-enable-tracing` asks for performance numbers taken separately with tracking off, with the settings recorded.
  The README does not state the settings. The effect is likely small and conservative. Record it, or remeasure in
  the next chain.
- Sampled mode writes 32 fresh seeds from the host on every step (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/generator.py:97-102`). The common sampler's own
  unseeded protocol (init, then SKIP, then no copy) avoids that per-step write
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/common/sampling/generator.py:1453-1506`). The work log keeps the
  per-step write because PRNG state "may not survive the decode kernels". That is untested. `traced_skip` passes
  the micro test, but only with sampler-only replays in between, not decode replays. Sampled token-out speed is
  also unmeasured. Test SKIP across real decode replays, then adopt it or record the measured reason.
- Superseded artifacts still claim a pass:
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence2/m*/sampling_audit.json` reports `"passed": true`, but it ran with the defective sampler.
  - All of `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence1/` is superseded.
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/m{1,4}/sampling_audit.json` reports `"passed": false`.
  The README explains this, but the files are not labeled in place. A later stage could copy them. Label them where
  they are.
- The memory recheck is partly inferred:
  - The DRAM remainder that the census does not reach (271, 313 and 187 MiB per device) is not itemized. A leaked
    copy of a per-layer constant would hide there.
  - Review 1 asked for construction peaks. The recheck says host RSS and setup peaks were "not re-measured".
  - p150 has 1.62 GB free with a 179 MB largest block per bank, so these numbers matter for Stage 9 headroom.
  - Next time, sample the allocator after each layer load and itemize the remainder by buffer size.
- On p150x2 and p150x4 the routed experts are stored twice: the EP prefill copy and the TP decode copy. The
  per-device layer-private DRAM on p150x2 (24359 MiB) roughly equals p150 (24425 MiB), and p150x2 has only
  2.99 GB free. This work is assigned to Stage 7. Keep it visible, because it caps the p150x2 KV pool at 2.0
  full-context sequences.
- Two serving contracts remain open for Stage 9. Log-probabilities are unavailable on 1, 2 and 4 devices, while the
  Stage 13 recorder requires `/v1/completions` with `logprobs=0`
  (`/home/hous/dev/qwen3.5-35b-a3b/evaluation/latency-methodology.json`). Explicit request seeds raise an error.
  Both block the packaged sweep unless solved.
- Several full-model-only costs are deferred to Stage 7. The LM head runs DRAM-interleaved with inner block 2 at
  66% DRAM use, and the perf report suggests a DRAM-sharded program config. Local top-k over the full vocabulary
  shard costs 285 to 1070 us per step. Force-argmax is not benchmarked.
- Decode at B32 loops attention and GDN per row (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/functional_decoder.py:468-476`). No B4 or B32 decode timing exists.
  Batch scaling is a Stage 7 and Stage 9 risk.
- Source manifests hash only `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/` and `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/` plus a `ttnn` diff stat. They record no
  HEAD commit and no status for the reused modules: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/common/sampling`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/common/modules`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/demos/blackhole/qwen36`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/experimental/gated_attention_gated_deltanet` and `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/tt_transformers`.
  These are clean at HEAD `fb2e3b5a` today. The provenance handoff asks for the starting commit and for the
  relevant modules outside `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/`.
- On p150, the readiness runners (`--mesh-device N150`) open the default device, while the model scripts use
  physical device 3 (`QDEV=3`). The p150 evidence therefore spans two physical chips. Label this.
- The work log says the first chain 3b launch directory was removed (it held a CPU log only). That departs slightly
  from "preserve failed/old attempts".

## Hard-Check Gaps

- The stage gate (`/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/prompts/model_bringup_multigoal/06-full-model.check.sh`) checks only degenerate output and the context contract. It passed here
  (exit 0).
- No gate measures calibration (KL against HF). Top-1, top-5 and top-100 cannot detect over-confidence.
- The sampled-distribution check runs only at bucket 32 with fixed logits. B1 and B4 sampled checks are rank < 32
  and "at least one of seven sequences differs from greedy".
- Logit determinism across batch positions is checked only at token level.
- Accuracy beyond 8193 tokens uses periodic text only (the 12417-token KL probe and the 262100-token run). Native
  context correctness rests on the layer-level evidence from earlier stages.
- The watcher ran only on the 2-layer model.
- Host-sampling compatibility mode ran only in the 2-layer generator probe.
- The HF references are noisy. Incremental generate disagrees with batched top-1 at 12 of 64 book positions and 1 of
  100 AIME24 positions (reverified).
- Penalty, log-probability and inactive-row variants have no CPU or device test (see Required Work).

## Anomaly Ledger

- Observed anomaly: TT is over-confident against HF at the 428-token position on every path.
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence2/continuation/m4_with_kl.json`; reviewer analysis of
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/continuation/m4.pt`.
  Affected path: prefill and decode logits, terminal distribution.
  Control or comparison: HF CPU logits at the same position. The three saturated positions also show a positive TT
  top-1 bias.
  Likely subsystem: precision policy margin, or an amplification in MoE routing or attention (hypothesis).
  Investigation performed: whole, split and decode-fed paths against HF at 4 positions. No precision control.
  Resolution: more-work-needed (Required Work P2, item 1).

- Observed anomaly: The p150x2 free-running story contains an arithmetic and logic error ("Seventy-two" hours per
  week, "Correct!").
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence2/m2/autoregressive/tt_completion.txt`.
  Affected path: p150x2 greedy token-out decode.
  Control or comparison: none. HF diverged at token 10.
  Likely subsystem: model behavior in a raw-completion story (hypothesis).
  Investigation performed: none.
  Resolution: more-work-needed (Required Work P2, item 2).

- Observed anomaly: All three profiles leave the HF free-running path at token 10, with the same alternative ("move
  faster than" against "stretch and shrink").
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence2/m{4,2,1}/autoregressive/autoregressive_meta.json` (prefix 10 on every profile).
  Affected path: token-out decode, story prompt without chat template.
  Control or comparison: the HF reference self-disagrees on narrative text; book TF top-5 is at least 98.4%.
  Likely subsystem: the shared precision policy at a narrative near-tie.
  Investigation performed: outputs read; the HF margin is not recorded.
  Resolution: controlled. Residual: the identical divergence on all profiles points to a systematic TT difference.
  The calibration item above may explain it.

- Observed anomaly: The sampled leg of the split-sampling trace equals greedy.
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/m*/split_sampling_trace.json`.
  Affected path: traced sampler at temperature 0.8, top-p 0.95.
  Control or comparison: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/hf_trace_prompt_margin.json`. The HF nucleus holds one token at all 7 steps
  (probability 1.0).
  Likely subsystem: test prompt choice.
  Investigation performed: HF nucleus check.
  Resolution: controlled.

- Observed anomaly: In chain 3b rerun 1, all four p150x2 sampled B4 rows equal greedy.
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/failed_audit_rerun1/m2/sampling_audit.json`.
  Affected path: sampled B4 decode.
  Control or comparison: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/sampling_repeat/m2.json`. Over 8 repeats the per-row rates (0.25, 0.375, 0.25,
  0.125) match the TT probabilities (0.127, 0.334, 0.458, 0.176). No repeat had all rows greedy.
  Likely subsystem: a chance event in a weak check.
  Investigation performed: repeat test against TT probabilities; the check was redesigned.
  Resolution: controlled.

- Observed anomaly: Some sampled draws fall outside the host top 32.
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/audit_rerun2/m{4,1}/sampling_audit.json` (one draw each, logit 13.125 equal to the 32nd, 3 and
  5 tokens tied).
  Affected path: device top-k at a BF16 tie.
  Control or comparison: tie-aware support check.
  Likely subsystem: BF16 tie order.
  Investigation performed: draws saved; ties counted.
  Resolution: controlled.

- Observed anomaly: Chain 2 sampled mode did not sample, yet the chain 2 audit passed.
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence2/m*/sampling_audit.json` (`passed: true`); the work log.
  Affected path: sampler seed and lane parameters.
  Control or comparison: frozen-seed control (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/control_frozen_seeds_l2_m4.json`) and the op distribution
  test.
  Likely subsystem: generator sampler wiring.
  Investigation performed: op probe; two defects fixed; checks redesigned; failed attempts kept under
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/seed_fix/`.
  Resolution: fixed. The stale pass files still need labels (Other Concerns).

- Observed anomaly: Whole and split prefill logits differ (PCC 0.83 to 0.96).
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/m*/full_evidence.json` `continuation`.
  Affected path: continuation prefill.
  Control or comparison: HF KL of at most 0.0002 at 4301, 8193 and 12417 tokens on every path.
  Likely subsystem: tail noise; MoE shape-dependent rounding.
  Investigation performed: HF KL probe.
  Resolution: controlled for the continuation contract. The 428-token case is covered by the first entry.

- Observed anomaly: The 2-layer probe diverges from HF on p150x2 (step 2) and p150 (step 8), at TT near-ties.
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence2/m*/watcher_probe.json`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/probe_control/`.
  Affected path: reduced 2-layer generator build.
  Control or comparison: eager, traced and post-warmup logits are bitwise equal; the run without the watcher gives
  the same tokens.
  Likely subsystem: numerics that depend on the build.
  Investigation performed: path and watcher controls.
  Resolution: controlled for trace and warmup. Residual: the build dependence is not isolated.

- Observed anomaly: 187 to 313 MiB of allocated DRAM per device is not reachable from the generator object graph.
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/memory_handoff_recheck.json` `unnamed_dram_mib`.
  Affected path: model construction and memory ownership.
  Control or comparison: none.
  Likely subsystem: program-cache or CCL-owned buffers, or unreleased per-layer copies (unverified).
  Investigation performed: none.
  Resolution: residual (Other Concerns).

## Scope Inspected

- Goal and skill paths:
  - `/home/hous/dev/qwen3.5-35b-a3b/runs/main/06-06-full-model.prompt.txt`
  - `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`
  - `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`
  - `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/full-model/SKILL.md`
  - `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/tt-enable-tracing/SKILL.md`
  - `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/qualitative-check/SKILL.md`
  - `/home/hous/dev/qwen3.5-35b-a3b/memory-layout-handoff.json` (required_future_work, GDN residency audit, setup
    peaks)
  - `/home/hous/dev/qwen3.5-35b-a3b/experiment-provenance-handoff.json` (required_before_acceptance)
- Artifact paths:
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/README.md`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/work_log.md`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/stage_review1.md`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/memory_handoff_recheck.json`,
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/context_contract.json`
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/evidence_chain3.sh`,
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/evidence_chain3.log`,
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/source_manifest_start.txt`,
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/source_manifest_end.txt`
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/m{4,2,1}/full_evidence.json`,
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/m{4,2,1}/split_sampling_trace.json`,
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/m{4,2,1}/sampling_audit.json` (superseded)
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/control_frozen_seeds_l2_m4.json`,
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/hf_trace_prompt_margin.json`,
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/hf_trace_prompt_margin.py`,
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/sampling_repeat/m2.json`
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/audit_rerun2/evidence_chain3b.sh`,
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/audit_rerun2/evidence_chain3b.log`,
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/audit_rerun2/m{4,2,1}/sampler_micro.json`,
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/audit_rerun2/m{4,2,1}/sampling_audit.json`,
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/audit_rerun2/cpu_tests.log`
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/failed_audit_rerun1/m{4,2,1}/sampling_audit.json`
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence2/evidence_chain2.sh`,
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence2/evidence_chain2.log`,
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence2/m{4,2,1}/teacher_forcing_aime24.log`,
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence2/m{4,2,1}/teacher_forcing_book8193.log`,
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence2/m{4,2,1}/prefill_check_aime24.log`
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence2/m{4,2,1}/autoregressive/`
    (completions and meta),
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence2/qualitative/`,
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence2/continuation/m4_with_kl.json`,
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence2/slots/` (with failed attempts)
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence2/m{4,2,1}/profile/summary.json`,
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence2/m4/profile/decode_perf_report.csv`
  - `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/references/` (refpt and meta), `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/qualitative/hf_outputs.json`, `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/sampling_hf/hf_sampling_margin.json`,
    `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/sampling_hf/hf_sampling_margin.py`, `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/continuation/m4.pt`, `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/continuation/hf_last_logits.pt`, `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/slots/`
- Code paths:
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/generator.py`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/model.py`, and the `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py` diff (`prefill_bucket_plan`,
    `_chunk_plan`) plus its attention decode path
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/functional_decoder.py` (`decode_forward`, `_rotary`, `_indices`), `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/fused_decoder.py` (indexed RoPE)
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_sampling_audit.py`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_sampler_micro.py`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_split_sampling_trace.py`,
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_full_evidence.py`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_continuation_probe.py`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/runtime_audit.py`,
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/test_generator_warmup.py`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/test_generator_sampling.py`
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/common/sampling/generator.py` (capture scope, trace keys, `format_sampling_params`, SeedManager),
    `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/common/sampling/tt_log_probs.py`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/common/sampling/tt_sampling.py` (return values)
  - `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/ttnn/tools/trace_allocation_tracker.py`
- Commands run (none touched a device):
  - Stage gate: `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/runtime/readiness_check/check_degenerate_output.py --scope autoregressive`
    (exit 0, "No degenerate output detected") and
    `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/scripts/check_context_contract.py --stage full-model --require-contract`
    ("target=262144, supported=262144").
  - Manifest diffs across chains 2, 3 and 3b, and against the current tree (equal).
  - Python comparisons of the chain 2 and chain 3 full-evidence JSON files (greedy outputs identical).
  - Reads of the audit, micro, trace, HF-margin and repeat JSON files; qualitative and autoregressive prefix matches.
  - CPU-only `torch.load` analysis of the saved continuation logits and the HF reference `.refpt` files (scripts in
    the session scratchpad).
  - `grep` over the final evidence logs (no trace, unsafe-allocation or cache-miss warnings) and over the artifacts
    for credentials (none found).

## Residual Risk

- TT may be systematically sharper than HF in non-saturated positions. This is unmeasured beyond one position.
- Serving contracts (inactive rows, slot mapping, penalties, log-probabilities, seeds, KV ownership through the
  adapter, 128-aligned continuation starts) are documented or partly enforced, but not exercised on device.
- p150 runs with 1.62 GB free DRAM and a 179 MB largest block per bank. Any later persistent allocation (vLLM, CCL
  buffers) has little room.
- Decode numerics depend on the build (2-layer probe), and the cause is not isolated.
- B32 decode cost and sampled-mode token-out speed are unmeasured.
