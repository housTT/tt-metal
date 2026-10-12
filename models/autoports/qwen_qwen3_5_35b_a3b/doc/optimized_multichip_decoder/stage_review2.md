# Stage Review

Stage 05, optimized multichip decoder. Model: Qwen/Qwen3.5-35B-A3B, revision
59d61f3ce65a6d9863b86d2e96597125219dc754. Reviewer: fresh independent subagent, review 2. Review date: 2026 Oct 12,
about 01:10 Eastern. I did not open a device, start a server, build, or run a device test. I ran CPU-only analysis of
existing artifacts.

Abbreviations: TP (tensor parallel), EP (expert parallel), MoE (mixture of experts), GDN (Gated DeltaNet, the
linear-attention layer), CCL (collective communication), AR (all-reduce), PCC (Pearson correlation coefficient), HF
(Hugging Face reference), BF16 (16-bit brain float), BFP8/BFP4 (8/4-bit block float), FP32/FP64 (32/64-bit float), L1
(core SRAM), DRAM (device memory), KV (key/value cache), SDPA (scaled dot-product attention), RoPE (rotary position
embedding), I2S/S2I (interleaved-to-sharded / sharded-to-interleaved), ULP (unit in the last place), TTFT (time to
first token), ISL (input sequence length), B32 (batch 32).

Short names:
- `AP` = `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b`
- `OD` = `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder`
- `E6` = `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder/evidence6`
- `S5` = `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage05`
- `E7` = `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/multichip_decoder/evidence7` (Stage 4 final)

Verdict: more-work-needed

Review 1 work is mostly addressed with real evidence (residual/CCL family micro, padding probes, TT router capture,
repeated prefill table, AutoDebug report, stacked-expert trial, local-capacity trial, DRAM row-hotspot fix, four
prefill knobs). Provenance is clean. The remaining required work is untried decode and prefill items that are
visible in the current `tt-perf-report` and step-op tables, two precision decisions that lack real-weight
evidence, and a missing performance-accounting item from `$optimize`.

## Required Work

- P1: The full-attention decode path has no operation-topology audit, and about 54 us of its 254 us kernel time is
  an untried cluster of small attention ops.
  Evidence:
  - `OD/README.md` "Operation-topology audit / Decode" lists "Segment (linear layer)" rows only. No row covers the
    full-attention decode segments (head creation, q/k norm, RoPE, transposes, SDPA input/output layout, output
    gate). 10 of 40 layers are full attention.
  - `E6/profiles/m4_full_attention/decode_step_ops.txt` (p150x4, device 0, one replay, us): RoPE gathers and tilizes
    ops 0-4 = 9.4 (owned by Stage 6, accepted); `NlpCreateHeads` 2.08; q-norm `LayerNorm` 5.27 (4 cores); k-norm
    `LayerNorm` 5.13 (1 core); `RotaryEmbeddingIndexed` 8.85 and 12.95 (110 cores each); `Transpose` 1.84, 1.75,
    1.66, 2.34; I2S 0.63 x 2; `Copy` 0.87; `NLPConcatHeads` 2.98; output-gate `BinaryNg` 7.26 (110 cores). Sum
    without the Stage 6 gathers: about 54 us per full-layer step.
  - `AP/tt/optimized_decoder.py` lines 1384-1446: decode RoPE uses `deepseek_prefill.rotary_embedding_indexed` with
    a transpose after each call; the query goes to `ttnn.DRAM_MEMORY_CONFIG` before SDPA decode, and the SDPA
    output is `ttnn.DRAM_MEMORY_CONFIG`, then transposed. The `$optimize` skill calls explicit DRAM on decode
    intermediates a performance smell that needs a failed L1 attempt and before/after evidence. None exists.
  - The only Stage 5 full-layer attention screens are the SDPA config, the combined cos/sin gather (rejected, -0.5
    us) and fast sigmoid (rejected, PCC). `OD/work_log.md` "Rejected: combined cos/sin RoPE table" addresses the
    gather, not the two 8.9 / 13.0 us rotary ops.
  Why this matters: the goal requires the pass to start with an operation-topology audit and to try every
  applicable item visible in the current `tt-perf-report`. At 10 layers, this cluster is about 0.54 ms per token on
  p150x4 (about 5% of the 10.00 ms layer-sum decode estimate).
  Required next step: add the full-layer decode audit table. Try, on p150x4 with the final defaults: decode RoPE with
  a decode-mode rotary op on the head layout that SDPA consumes (for example `rotary_embedding_llama` decode mode,
  with identity cos/sin padding for the 192 non-rotary dims, or a 64-dim rotary slice), head creation that removes
  the four transposes, L1 (or sharded L1) query and SDPA output, and the output gate multiply in L1. Record
  before/after decode, PCC, or an exact op-contract blocker for each. Use `$autofix` if a first API error blocks a
  candidate that removes ops.

- P2: Linear-layer and MoE decode small-op clusters remain without an attempt.
  Evidence (`E6/profiles/m4_linear_attention/decode_step_ops.txt`, p150x4, us):
  - GDN input slices before the fused kernel: ops 11-14 = 1.59 + 5.72 + 1.73 + 1.42 = 10.5. The kernel already reads
    `a`/`b` directly from the in-projection output (`gdn_decode_gates`); q/k/v/z offsets were not folded the same way.
  - Conv-state shift: ops 7-10 (slice, slice, concat, copy) = 5.6.
  - Shared-expert tail: shared down matmul 4.61 (op 37), `Unary` 3.38 (op 33, 110 cores), `BinaryNg` 4.85 and 2.53
    (ops 38-39, 110 cores) = 15.4. The shared expert is already a slot of the expert gate/up kernel; folding its down
    projection into the routed `sparse_matmul` as an extra slot, with the sigmoid gate as its score in
    `DeepseekMoEFastReduceNCFused`, was not tried.
  - Residual-boundary conversions around matmuls: S2I after each norm (ops 2, 22) and I2S before each AR (ops 18,
    40) = 5.5. A matmul that reads the width-sharded norm output, or writes the AR input shard spec directly, was not
    tried.
  - These items add up to about 37 us per linear-layer step (16% of 222.9 us kernel time), x 30 layers.
  Why this matters: `$optimize` "Final Audit Checks" and OPT-003 require removal of unnecessary I2S/S2I and layout
  ops, or measured evidence. The README lists "layout ops" as a largest item but records no action.
  Required next step: try the GDN-kernel offset read for q/k/v/z, the conv-state shift fold, the shared-down fold,
  and sharded matmul input/output at the residual boundary. Keep the ones that win; record before/after or an exact
  blocker for the others.

- P2: Two precision/fidelity decisions lack real-weight evidence.
  Evidence:
  - Router plus shared gate: 32 x 2048 x 288 BF16 x BF16 HiFi4, 7.70 us, `SLOW`, 30% DRAM, 40 layers
    (`E6/profiles/m4_*_attention/decode_perf_report.csv`). Advice: "HiFi2 may also work". `OD/README.md` precision
    table "Tried" column for this group is empty. No HiFi2, LoFi or BFP8-weight router trial exists in Stage 3, 4 or 5
    work logs. "Router ties" is a reason to measure with the tie-aware gate and router capture, not a measurement.
  - SDPA prefill LoFi: `S5/micro/sdpa_prefill/sdpa_prefill2.json` shows q128/k512 LoFi 49.7 ms vs selected
    q128/k256 HiFi2 57.2 ms per 4096-token chunk at start 126976 on x4 (-13%), and 149.1 vs 173.0 ms on p150. It was
    rejected on "PCC vs FP64 0.99682". `S5/micro/sdpa_prefill/sdpa_prefill.py` lines 32-38 build Q/K/V with
    `torch.randn`. This is a synthetic-only veto, which `$stage-review` and OPT-012 do not accept for a faster
    candidate. Long-context TTFT is dominated by this op (README: full-layer chunks grow to 67.6 ms on x4).
  - GDN in-projection BFP8 LoFi was measured only as an op micro (22.0 vs 22.3 us, `S5/micro/inproj_sweep.json`).
    The README says "BFP8 HiFi2 fastest passing"; the faster LoFi candidate never ran the real-weight layer gate.
  Required next step: run router HiFi2 (and BFP8-weight HiFi2/LoFi if legal) with the tie-aware gate plus router
  capture on real weights; run SDPA prefill LoFi on real weights at a long chunk start (for example inside the
  262143-token acceptance harness or with recorded activations) and decide on that evidence; run GDN in-projection
  BFP8 LoFi in the layer screen. Adopt any candidate that passes and wins; otherwise record the model-visible failure.

- P2: Mid-length prefill is slower than 4096-token prefill on meshes, and nobody examined it.
  Evidence: `E6/prefill_timing/p150x4_layer0.json` warm medians: 2400 = 10.91 ms, 2600 = 12.12, 3000 = 11.23,
  3300 = 11.81, 4095 = 11.29, 4096 = 10.60. Full layer 2600 = 11.90 vs 4096 = 11.11. p150x2 linear 2600 = 15.45 vs
  4096 = 14.18. p150 is monotonic. `E6/len_m4_linear_2600/default.json` `moe_pieces` = [[2048, 2048, 2048],
  [1024, 576, 552]]: two MoE calls, while 4096 uses one. The planner ladder (`prefill_buckets` [2048, 1024, 256, 64])
  costs more than one masked 4096 piece for these lengths.
  Why this matters: TTFT is an explicit `$optimize` target, and the required sweep includes short and medium ISL.
  Required next step: compare the current ladder with one padded/masked 4096 MoE piece (or a cost-based choice) for
  lengths between 2048 and 4095 on all profiles, keep non-aligned correctness, and adopt the faster plan.

- P2: Performance accounting has no roofline.
  Evidence: `OD/README.md` reports kernel time and traced host time, but no roofline. `$optimize` "Performance
  Accounting" requires roofline, device time and end-to-end from the same run, and the checklist item "Performance
  accounting reconciled" is mandatory. Stage 4 reported a modeled roofline
  (`AP/doc/multichip_decoder/README.md` line 412) and `AP/tests/estimate_roofline.py` exists.
  Required next step: compute the per-token compulsory bytes (weights at stored dtypes, active experts, GDN state
  read/write, KV read at 4096 and 131072) per profile from `E6` and report roofline, device time and host time with
  the gaps named. This needs no device run.

## Other Concerns

- `E6/aligned_sdpa/m{1,2,4}.json` have `runtime_audit` false (no `--audit` in `S5/evidence_chain6.sh` line 106) and
  are single runs. `OD/README.md` says the audit is clean "on every trial row" and the decode table header says
  "median of 3 runs"; the aligned column (used for the 40-layer estimate) is neither. Correct the text or rerun with
  `--audit`.
- Top-k decode router: +40 us for a change that removes about 11.6 us of ops (`S5/trials/topk_decode/`). No
  profile row explains it. A single-core `TopKDeviceOperation` fallback is the likely cause; it is unverified.
- p150 SDPA prefill: q128/k512 HiFi2 is 166.5 ms vs selected k256 173.0 ms at start 126976 (87.4 vs 90.9 at 65536)
  in `S5/micro/sdpa_prefill/sdpa_prefill2.json`. The profile-wide k256 choice is best only on x4.
- Linear-layer prefill per-token PCC: tokens below 0.995 went from 1-3 (Stage 4) and 4-6 (chain 5) to 5-7 (chain 6);
  minimum token 0.9928 -> 0.9910 (token 3515). Aggregate PCC is unchanged (0.99972 -> 0.99969). Not reported in the
  README. Likely router near ties under the top-k prefill router; unverified.
- Padding probe (`E6/probes/padding_state_x*.json`) checks GDN states only. For the full layer `tensors` is 0, so KV
  cache contents after padding injection are not checked. The README statement is accurate but the coverage gap
  should be stated.
- MoE prefill drift: not observed in chain 6, mechanism not isolated. `generalized_moe_gate` still runs in decode.
  The 4097-length rows cover one gate call before a repeated prefill; a decode-then-prefill bitwise repeat on p150x4
  is not directly tested. Serving-stage repeatability remains the stated owner.
- No prefill chunk-size comparison (4096 fixed) for the 131072-token TTFT target, which `$optimize` "Prefill
  Latency And TTFT" asks for.
- B > 1 decode is still a per-row loop (B32 x4 7.5 ms per layer step). Owner Stage 7. It dominates the required
  concurrency-32 sweep.
- p150x2 per-layer weights (614 MB) are close to p150 (638 MB) because of the EP prefill plus TP decode expert
  copies (`E6/capacity_model.json`). The stacked option is measured (+4-7% prefill) and selectable; the capacity
  decision is deferred to Stage 7.

## Hard-Check Gaps

- OPT-009 CCL table fields (`chunks_per_sync`, workers per link, buffers per channel) are still absent (review 1 gap).
- OPT-006 cumulative contract table with op-to-op gap per row is still spread over README sections (review 1 gap).
- Headline and length rows run without `TT_METAL_TRACE_ALLOC_TRACKING`; positions, reuse, watcher-long and device
  pytest rows use it with program-cache allocations included.
- Watcher long runs cover p150x4 and p150 only; p150x2 has the four acceptance watcher cases.
- Long-context timing uses random hidden input; accuracy at long context rests on the 262143-token acceptance rows.

## Anomaly Ledger

- Observed anomaly: full-attention decode spends about 54 us in small attention ops (two rotary ops at 8.9 and 13.0
  us on 110 cores).
  Evidence: `E6/profiles/m4_full_attention/decode_step_ops.txt`.
  Affected path: full-attention decode head/RoPE/SDPA layout.
  Control or comparison: none in Stage 5.
  Likely subsystem: prefill rotary op used for decode, transposes, DRAM intermediates.
  Investigation performed: none recorded.
  Resolution: more-work-needed (P1).
- Observed anomaly: prefill at 2400-4095 tokens is slower than at 4096 on p150x4 and p150x2.
  Evidence: `E6/prefill_timing/*.json`, `E6/len_m4_linear_2600/default.json` `moe_pieces`.
  Affected path: prefill MoE piece planner.
  Control or comparison: p150 is monotonic; 4096 runs one MoE piece.
  Likely subsystem: bucket ladder plus per-call EP dispatch overhead.
  Investigation performed: none recorded.
  Resolution: more-work-needed (P2).
- Observed anomaly: top-k decode router costs +40 us.
  Evidence: `S5/trials/topk_decode/screen_m4_{linear,full}/`.
  Affected path: decode router candidate.
  Control or comparison: 3 alternating pairs.
  Likely subsystem: `ttnn.topk` decode kernel selection (unverified).
  Investigation performed: timing only.
  Resolution: controlled as a rejection; cause not named.
- Observed anomaly: linear prefill per-token PCC tail widened after chain 6.
  Evidence: `E6/headline_m*_linear/default.json` `prefill_token_pcc`; `OD/evidence5/headline_m*_linear/default.json`.
  Affected path: prefill router/norm/q-k knobs.
  Control or comparison: aggregate PCC unchanged; Stage 4 minimum token 0.9928.
  Likely subsystem: router near ties (unverified).
  Investigation performed: none recorded.
  Resolution: open; record it, check against the prefill router tie set.
- Observed anomaly: MoE prefill drift on p150x4 in chains 2-5.
  Evidence: `E6/nondeterminism/`, `OD/evidence3/nondeterminism/followup/AUTODEBUG.md`.
  Affected path: `moe_fused_swiglu`, `post_combine_reduce`.
  Control or comparison: 0 differing captures in 3 runs per profile in chain 6; 133 of 133 chain 6 rows repeat.
  Likely subsystem: per-core state left by an earlier kernel (open).
  Investigation performed: E3-E6 probes, AutoDebug; gate-loop router removed.
  Resolution: controlled by measurement; root cause open; serving repeatability check owed.
- Observed anomaly: strict decode PCC 0.990 at router near ties.
  Evidence: `E6/probes/router_l0_x4.json`, `router_l3_2600_x4.json`.
  Control or comparison: HF logits. Reviewer recomputation: every flip in the layer 0 positions and headline cases
  is inside the 1-ULP window. In the layer 3, 2600-token probe (final BFP8 attention-out), step 6 routes expert 163
  at 2 ULP below the HF boundary (-4.5625 vs -4.5); strict output PCC there is 0.99896, so no gate is affected.
  Resolution: controlled. The README sentence is accurate for its stated scope; a 2-ULP flip still occurs on full
  layers and should be stated next to the gate limit.

## Verified Claims

- Provenance: `E6/source_manifest_start.json` equals `source_manifest_end.json`. All 54 hashed `tt/` and `tests/`
  files, including the 10 kernel sources, match the live tree. No live file is unhashed. `git_head`
  `5d4c0441` equals `HEAD`; the outside-autoport diff hash equals the empty-diff hash. `E6/source_snapshot.tar.gz`
  matches live `AP/tt` and `AP/tests` byte for byte. `optimized_decoder.py` and `multichip_decoder.py` mtimes
  (05:07 UTC) are after the chain, but their content equals the manifest (post-chain trial source removed).
- `S5/evidence_chain6.log`: 190 `rc=` lines, all `rc=0`; `SOURCE_STABLE`, `CHILD_FAILURES 0`, `CHAIN_DONE`.
- Headline decode medians (246.4 / 263.9, 287.0 / 294.0, 380.3 / 373.7 us) and prefill medians recompute from
  `E6/headline_*`. Stage 4 values match `E7/headline_p150*`. 40-layer sums recompute (10.00 ms, 429 ms on p150x4).
- Repeated prefill table matches `E6/prefill_timing/*.json` and `E7/micro/prefill_timing7/`; only p150x2 full at 3
  tokens is slower than Stage 4.
- 157 result rows scanned: every prefill and decode PCC is at or above 0.995; `runtime_audit` true except the 3
  aligned rows; 133 prefill-repeat rows, 0 differ.
- Padding probes: 36 of 36 row-0 bitwise cases; non-finite padding reaches the device.
- 20 watcher logs: no error, assert, hang or overflow line. Device pytest 12 passed; CPU tests 1179 passed.
- Native context 262143/262144 rows on all profiles; `capability_reduction` null.
- `tt-perf-report` rows confirm the README dtypes: GDN in-projection HiFi2 BF16 x BFP8, qkvg and full attention-out
  LoFi BF16 x BFP8, linear attention-out HiFi2 BF16 x BFP8, router HiFi4 BF16 x BF16.
- Review 1 P1 residual family: `S5/micro/residual_family/` lower bound (b1) 47.9 / 61.9 us exceeds the adopted
  boundary 26.6 / 27.1 us; the bound also covers the untried fused all-gather-matmul, because that path still runs
  the in-projection.

## Scope Inspected

- Goal/skill paths: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/05-05-optimized-multichip-decoder.prompt.txt`,
  `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/optimize/SKILL.md` (full),
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/multichip/SKILL.md` (full).
  `tt-enable-tracing` was not reread; Stage 5 did not change the trace contract beyond pre-capture state.
- Artifact paths: `OD/README.md`, `OD/work_log.md`, `OD/stage_review1.md`, `E6/` (manifests, snapshot, headline,
  aligned_sdpa, len, long128, b32, accept, real_inputs, prefill_timing, profiles m1/m2/m4 step ops, replay summaries,
  perf CSV, prefill raw summaries, probes, nondeterminism, long_context, capacity_model.json, watcher logs,
  cpu_pytest.xml), `AP/doc/context_contract.json`, `S5/evidence_chain6.sh`, `S5/evidence_chain6.log`,
  `S5/trials/{topk_decode,fused_combine,prefill}/`, `S5/micro/{geometry_full_sweep.json,inproj_sweep.json,
  geometry_out_*.json,sdpa_prefill/,sdpa_long/}`, `E7/headline_*`, `E7/b32_*`, `E7/micro/prefill_timing7/`,
  `OD/evidence5/headline_*_linear/default.json`.
- Code paths: `AP/tt/optimized_decoder.py` (full-attention decode path 1335-1450, GDN decode 1530-1700),
  `AP/tt/fused_decoder.py` `_rope`, `AP/tests/run_layer.py` warm-prefill block, `E6/probes/padding_state_probe.py`.
- Commands run: `git status`, `git log`, `git diff`, `git rev-parse`, `sha256sum`, `tar`, `diff -r`, `ls`, `grep`,
  `zgrep`, `wc`, `date`, and Python over JSON, CSV and logs. No device, server, build or device test.

## Residual Risk

- The tie-aware gate accepts any expert set inside a 1-ULP band; a router bias smaller than one BF16 ULP is not
  detectable.
- MoE prefill drift has no root cause; the decode gate op still runs between serving prefills.
- Full-model padding behavior over 40 layers is covered by single-layer and two-layer probes only.
- Long-context accuracy beyond single-step 262143 rows is unmeasured; SDPA prefill precision at long chunk starts has
  only synthetic evidence.
- Concurrency 32 at ISL 131072 does not fit the p150x4 capacity model with the EP plus TP expert layout; the stacked
  layout would free memory at a prefill cost.
