# Stage Review

Stage 05, optimized multichip decoder. Model: Qwen/Qwen3.5-35B-A3B, revision
59d61f3ce65a6d9863b86d2e96597125219dc754. Reviewer: fresh independent subagent, review 1. Review date: 2026 Oct 11
(20:30 to 21:10 UTC). I did not open a device, start a server, build, or run a device test. I ran CPU-only analysis
of existing artifacts.

Abbreviations: TP (tensor parallel), EP (expert parallel), MoE (mixture of experts), GDN (Gated DeltaNet, the
linear-attention layer), CCL (collective communication), AR (all-reduce), RS (reduce-scatter), AG (all-gather), PCC
(Pearson correlation coefficient), HF (Hugging Face reference), BF16 (16-bit brain float), BFP8/BFP4 (8/4-bit block
float), ULP (unit in the last place), L1 (core SRAM), DRAM (device memory), KV (key/value cache), CB (circular
buffer), B1/B32 (batch 1/32), ISL (input sequence length).

Short names:
- `AP` = `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b`
- `OD` = `AP/doc/optimized_multichip_decoder`
- `E3` = `OD/evidence3`
- `S5` = `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage05`
- `E7` = `AP/doc/multichip_decoder/evidence7` (Stage 4 final)

Note: `OD/README.md` and `OD/work_log.md` changed during this review (mtime 20:42:37 UTC; new section "BFP4
re-screen on the final defaults", `S5/trials/b4_{full,linear}_x4`). I reviewed that version.

Verdict: more-work-needed

## Required Work

- P1: Fused matmul+CCL and fractured-residual families are dismissed by prose, not measured on the Stage 5 baseline.
  Evidence:
  - `OD/README.md` "CCL and residual families": the fused row says "not applicable: the producers (out-projection
    9.6 us, shared down, expert reduce) are not one matmul followed by one CCL". This is false for the attention
    side. In `AP/tt/multichip_decoder.py` `_AllReduceOutput.forward`, the attention module output (GDN or full
    out-projection, one matmul) feeds exactly one `all_reduce_async`. Only the MoE boundary sums several producers.
  - The fractured row cites only Stage 4 micro data (`E7/micro/residual_contract_x4.json`: replicated 40.7 us,
    RS + distributed norm + AG 45.3 us, without output gather 36.5 us). That micro measured the Stage 4 replicated
    path (RS + AG, 1-core FP32 norm). Stage 5 changed the replicated baseline (direct AR 12.0 vs 16.8 us, sharded
    residual, 16-core norm). No Stage 5 micro or layer screen measures RS (or matmul+RS) plus distributed norm, and
    no `all_gather_matmul`-style consumer was tried. `S5/micro/` has no residual-family artifact.
  - `OD/work_log.md` candidate 8 says "recheck with direct all-reduce: planned". The freeze table replaces it with
    the width-sharded replicated residual, which is a different family.
  Why this matters: the goal contract requires coherent CCL/residual families and evidence for fused matmul-CCL
  paths. The optimize skill (OPT topology audit, "if a matmul is immediately followed by a collective ... that is
  optimization work") and stage-review rules reject prose dismissals and stale cross-baseline bounds.
  Required next step: on p150x4 (and p150x2), measure in one harness at the Stage 5 defaults: (a) adopted path
  (direct AR into the sharded residual, sharded add, 16-core norm), (b) out-projection + RS (fused
  matmul-RS where the API allows) + distributed RMSNorm with no output gather, as a lower bound, and (c) the
  same with the gather fused into, or placed before, the next consumer. If (b) is slower than (a), record that as
  the bound-based rejection. Correct the "not applicable" text.

- P1: Material prefill optimizations are deferred, contrary to "No optimization is deferred".
  Evidence:
  - `OD/README.md` "Deferred": EP prefill dispatch with local-only capacity deferred to Stage 7; work log: "Estimated
    gain about 10% ... (not measured)".
  - `E3/profiles/m4_linear_attention/prefill_raw_summary.json`: `EmbeddingsDeviceOperation` (dispatch and restore
    gathers) 4278 us of 19154 us kernel time (22%), the largest prefill op family. README says about 3/4 of these
    rows are non-local.
  - The same file shows about 5.4 ms of layout and elementwise work (BinaryNg 2218, ReshapeView 1345, Untilize
    852 + 731, Typecast 771, Concat 733, Tilize 692 us). The operation-topology audit in `OD/README.md` covers
    decode only. There is no prefill topology table or action for these rows.
  - The required service has a 131072-token ISL. The layer-sum TTFT estimate is about 33 s on p150x4
    (`OD/README.md` "Long context").
  Why this matters: the goal says every applicable item is tried and nothing is deferred. The optimize skill says
  "Do not leave a known optimization for a later stage unless another skill explicitly owns it" and names prefill
  and TTFT as explicit targets. No later skill owns the EP dispatch capacity.
  Required next step: implement and measure local-only (or bounded) EP dispatch capacity, for example with a
  device-side count and capacity bucket or one host count read per MoE call, and add a prefill operation-topology
  audit with actions for the layout rows. If an op blocks it, keep a minimal repro or use `$autofix`. Report
  before/after prefill at 4096 and at a long chunk start.

- P1: Precision coverage has one untried dominant group and one rejection on a reason the optimize skill forbids.
  Evidence:
  - GDN in-projection (32 x 2048 x 3136 BFP8 HiFi2 on p150x4, 23.6 to 23.9 us, the largest decode matmul and a
    `SLOW` row in `E3/profiles/m4_linear_attention/decode_perf_report.csv`, 30 of 40 layers) has no BFP4 trial.
    `S5/trials/prec_linear_x4/` and `S5/micro/inproj_sweep.json` contain BF16 and BFP8 rows only.
  - Attention-out BFP4 LoFi (full layer) passes on real weights and recorded activations (HF decode min 0.99622,
    gate 0.995) and saves 3.4 us decode (`S5/trials/b4_full_x4`). `OD/README.md` rejects it as "net slower for
    prefill, large margin loss". The optimize skill says: "Do not reject BFP4 attention ... because a prefill row
    got slower when the optimization target is decode." The +1.06 ms prefill figure is one warm run. Stage 4
    measured single-run prefill spread up to 3.8 ms and repeated spread up to 2.1 ms (`AP/doc/multichip_decoder/README.md`
    "Prefill latency by length"). A decode-only BFP4 copy with the BFP8 prefill weight kept was not tried.
  Why this matters: the goal requires evidence for precision/fidelity reductions on model-applicable groups, and
  stage-review treats a slower higher-precision selection without an allowed blocker as required work.
  Required next step: run GDN in-projection BFP4 (LoFi and HiFi2) at the selected geometry family on real weights
  on all three profiles. For attention-out BFP4, either adopt a decode-only BFP4 copy (measure memory and decode)
  or reject it on an allowed reason (model-visible loss, trace failure, or repeated-timing evidence of
  unacceptable latency with no decode-only option).

- P2: The padding-fill removal contradicts a Stage 3 finding, and the residual-contract claim is untested.
  Evidence:
  - Stage 3 (`AP/doc/optimized_decoder/work_log.md` lines 225-231 and 282-285): "The sharded `rms_norm` output
    row 0 depends on uninitialized tile padding rows. This is a TTNN finding; the model now fills padding before
    this op." and "decode-mode steps inside prefill ... carry uninitialized tile padding rows".
  - Stage 5 sets `decode_input_fill` and `decode_norm_fill` to false (`git diff HEAD -- AP/tt/optimized_decoder.py`).
    `OD/work_log.md` line 82-83 still says the norm fill "remains required for the sharded norm (Stage 3
    determinism finding)". Line 218-221 then removes it: "norm, matmul, add, all-reduce are row-independent ...
    Open check: padding rows over 40 layers (Stage 6 full model)".
  - The inter-layer contract in `OD/README.md` states "rows 1-31 are tile padding, read only by row-local ops".
    The single-layer harness feeds zero padding from the host. No test injects non-zero, large or non-finite
    values into padding rows of the layer input, and no test stacks two layers.
  Why this matters: full-model bring-up inherits this contract. If padding grows over 40 layers, a padding-sensitive
  op corrupts row 0.
  Required next step: add a decode test on all profiles that writes large and non-finite values into rows 1-31 of
  the layer input and of the sharded norm inputs, and shows row 0 output bitwise unchanged. Alternatively restore
  the fills at layer entry. Reconcile work log line 82-83.

- P2: The tie-aware gate is justified, but its documentation is wrong and TT-side evidence is missing.
  Evidence (reviewer analysis of all `E3` rows):
  - On non-tie tokens, Stage 5 strict PCC matches Stage 4. Headline: full non-tie min 0.99903 vs Stage 4 0.99906;
    linear 0.99971 vs 0.99966. Real-input positions: the largest non-tie drop against Stage 4 is 0.00018
    (`E3/real_inputs_m*/positions_{0,3}.json` against `E7/real_inputs_m*`). The gate does not hide a broad
    regression.
  - `OD/README.md` says "up to 15 candidates per token". The measured maximum is 924 candidates
    (`E3/headline_m{1,2,4}_full/default.json` step 86: 12 experts within the window, 6 open slots). 120 to 165
    candidates occur in `real_inputs_m*/positions_3.json`, `len_m*_full_2400` and `headline_*_linear`.
  - The window is |logit - boundary| <= 1 ULP on both sides, so two swapped experts can differ by 2 ULP.
  - "TT picks the rank-10 expert" is inferred from two different comparisons (HF-with-rank-10 vs HF 0.98954; TT vs
    HF 0.9900). No TT router logits or top-8 indices were captured. Stage 3 did capture them for its tie
    (`AP/doc/optimized_decoder/README.md`, `review5/token684_tt/`).
  - The flips come from a Stage 5 precision change: linear-layer decode norm moved from the FP32 1-core norm to
    the 16-core BF16-input norm (`S5/trials/tie_check_x4`, strict 0.98962 at step 32; control 0.99874). The
    README attributes it only to "TT may round the router logit one ULP differently" and the precision table does
    not list the norm input change.
  Why this matters: the gate is the stage's accuracy acceptance rule. A wrong bound in the README misleads later
  stages. Without a TT-side capture, the near-tie classification is inference.
  Required next step: correct the candidate count and window description. Capture TT top-8 indices and router
  logits on decode for the flagged tokens (positions row 24 step 6, row 17 step 4, headline linear step 32) on one
  mesh and show that TT's set equals the chosen candidate and that TT boundary logits are within 1 ULP of HF. Add
  the norm input precision change to the precision table.

- P2: Prefill before/after numbers are single runs inside the documented noise.
  Evidence: README Results prefill p150x4 full 19.9 -> 18.39 ms, linear 21.0 -> 19.68 ms, and the 40-layer
  829 -> 774 ms estimate. All Stage 5 values are single warm runs (`prefill_host_us` in `E3/headline_*/commands.json`).
  Stage 4 declared its repeated table the latency reference (p150x4 4096: full 18.8, linear 20.7 ms; spread up to
  2.1 ms). Against that reference the Stage 5 gain is 0.4 / 1.0 ms. The scatter A/B in `S5/trials/sc_*` is a
  same-session pair, which is more credible, but the final default has no repeated table.
  Why this matters: the goal requires before/after warmed prefill, and stage-review requires like-for-like
  comparisons.
  Required next step: run a repeated warm-prefill table (compile run plus at least 6 warm runs) on the final
  default for the Stage 4 length set on all profiles, with frozen sources. Report against Stage 4's repeated table.

- P2: The MoE prefill nondeterminism record is inconsistent and cites a missing report.
  Evidence:
  - `OD/README.md` cites `followup/AUTODEBUG.md`. It does not exist. `S5/autodebug_nondet.log` is 0 bytes. An
    AutoDebug process was still running at review time.
  - The README conclusion excludes candidates using "p150 (no fabric, no CCL) repeats bitwise". The chain shows a
    single-chip drift: `E3/len_m1_linear_1000/default.json` `prefill_repeat_difference` = 32 rows (one 32-row
    block, rows 256-287), max abs 0.00049. Stage 3 also saw single-chip combine drift. The fabric hypothesis must
    account for this.
  - Ownership came from Stage 4 (`AP/doc/multichip_decoder/stage_review7.md`: "owner Stage 5"). The root cause is
    still unknown.
  Why this matters: repeatability of greedy output is a serving contract. A run-level state dependence that the op
  reads but does not write can be an uninitialized read.
  Required next step: finish AutoDebug and record its result. Reconcile the single-chip drift. If no fix, keep
  "controlled by bound" with the corrected scope and a named owner, and remove the dangling citation until the
  file exists.

- P2: Expert weights are duplicated on meshes, and this limits the required concurrency.
  Evidence: `OD/README.md` "Expert weights on meshes: an expert-TP copy (decode) and an EP copy (prefill)", deferred
  to Stage 7. `E3/capacity_model.json`: 40-layer weights per device p150 25.34 GB, p150x2 24.45 GB (3.5% less on
  two chips). p150x4 KV capacity with 32 GDN slots is 3,298,438 tokens. 32 concurrent requests at 131072 + 1024
  tokens need 4,227,072 tokens. The AGENTS.md sweep requires concurrency 32 at ISL 131072.
  Why this matters: AGENTS.md asks for real local-expert sharding and removal of duplicated representations.
  This duplication is the main memory lever for the required profiles.
  Required next step: measure a single expert representation (EP layout used by decode too, or TP layout used by
  prefill) on p150x4 and p150x2, with decode and prefill timing. If it is slower, record the memory/latency
  tradeoff and the resulting active-slot limit per profile.

## Other Concerns

- Batched decode is deferred to Stage 7. B32 decode is a per-row loop: p150x4 full 7.52 ms per layer step
  (`E3/b32_m4_full/commands.json`), the same per-user cost as B1. The optimize skill makes B1 the primary target,
  so this is not a stage blocker, but it dominates the required concurrency-32 sweep. The decode CCL path still
  selects by `partial.shape[-2] == 1` (`AP/tt/multichip_decoder.py`).
- `OD/README.md` precision table: "BFP8 HiFi2 passes; LoFi fails the BF16 trial". The BF16 LoFi trial passed the
  gate (HF 0.99671, control 0.99672, `OD/work_log.md` precision matrix, "no gain"). BFP8 LoFi was measured only as
  an op micro (22.0 vs 22.3 us). State the actual reason.
- `S5/trials/statel1_x4` (`gdn_state_l1`, 2 slots: 454.3 us) and `S5/trials/norm_fill_x4/norm16_fill_statel1`
  (exit 1, `dataflow_buffer.cpp:2704`) are not in the work log. The README does not state where the fused GDN
  kernel keeps the recurrent state. AGENTS.md asks for GDN state residency per profile.
- Geometry: x2 qkvg op 34.3 -> 30.6 us and x2 full layer 295.2 -> 293.7 us were "within run-to-run spread; not
  adopted". The spread is not quantified.
- The SLOW in-projection row reports output subblock 1x1 (per_n 5). The sweep tried per_n values with larger
  subblocks; say so beside the SLOW row.
- `ttnn::to_layout: memory_config is specified but the tensor is already in the requested layout` appears in
  acceptance logs (48 times in `E3/accept_pytest/device.log`; also in Stage 4). A requested memory config change
  is silently ignored somewhere. Identify the call site.
- `E3/source_manifest_*.json` hashes Python sources only. The device kernel sources under `AP/tt/kernels/` are not
  hashed. `E3/source_snapshot.tar.gz` contains them and matches the live tree (reviewer diff), and their mtimes
  (15:30 UTC latest) precede the chain (18:52-20:18 UTC). Add the kernel files to the manifest.
- `OD/work_log.md` says times are Eastern, but "15:30:40" matches the UTC file mtime of
  `AP/tt/kernels/expert_gemv/reader.cpp`.
- The README 40-layer decode estimate uses the aligned-table full decode (261.0 us) against Stage 4 numbers taken
  with the 132-block table. The difference is about 0.04 ms per token.

## Hard-Check Gaps

- Optimize OPT-009 asks for a per-row CCL table with payload dtype, memory configs, axis, topology,
  `chunks_per_sync`, workers per link, buffers per channel, persistent buffer status and time. The README CCL table
  lacks the tuning fields. `num_links` is fixed at 2 (`CCL_LINKS`) without a Stage 5 sweep.
- The optimize skill asks for a cumulative contract table (projection topology, dtypes and fidelity, KV layout,
  logical batch versus padded rows, SDPA config, residual/norm layout, geometry rows, total device time, op-to-op
  gap). The data is spread over README sections; op-to-op gap is not reported per row.
- No stacked two-layer decode test proves that the layer-entry `to_memory_config` is a no-op when the previous
  layer produced the contract.
- The GDN kernel unit test is single-step (`E3/kernels/gdn_kernel_test.py`). Multi-step state drift rests on the
  128-step headline and positions runs.
- Headline and length rows run without `TT_METAL_TRACE_ALLOC_TRACKING`. Positions, reuse, watcher-long and device
  pytest rows use it (`S5/evidence_chain3.sh`), which covers the new persistent state.
- The p150x4 KV capacity shortfall for 32 x 131072 is a model estimate (2 GiB reserve assumed), not measured.

## Anomaly Ledger

- Observed anomaly: strict decode PCC 0.9900 (positions row 24 step 6, all profiles) and 0.9903-0.9906 (headline
  linear step 32).
  Evidence: `E3/real_inputs_m*/positions_0.json`, `E3/headline_m*_linear/default.json`, `S5/trials/tie_check_x4`.
  Affected path: linear-layer decode norm feeding the router.
  Control or comparison: Stage 4 0.99982 at the same token; non-tie tokens unchanged (max drop 0.00018).
  Likely subsystem: BF16 router near ties after the 16-core BF16-input norm change.
  Investigation performed: CPU HF logits by rank (`S5/micro/positions_ties/`); tie-aware candidates.
  Resolution: more-work-needed (documentation fix plus TT-side top-8 capture; see P2).
- Observed anomaly: MoE prefill not bitwise repeatable, including one single-chip row.
  Evidence: `E3/nondeterminism/`, `E3/nondeterminism/followup/`, `E3/len_m1_linear_1000/default.json`.
  Affected path: `moe_fused_swiglu`, `post_combine_reduce`.
  Control or comparison: within-run repeats identical; inputs hashed identical (E4).
  Likely subsystem: run-level device state read by the ops; mechanism unknown.
  Investigation performed: E3, E3b, E4, E5, E6; AutoDebug running.
  Resolution: more-work-needed (finish AutoDebug, reconcile p150 drift, fix citation).
- Observed anomaly: padding fills removed although Stage 3 found sharded norm row 0 depends on padding.
  Evidence: `AP/doc/optimized_decoder/work_log.md` 225-231; `OD/work_log.md` 82-83 and 218-221.
  Affected path: decode norms, residual contract over stacked layers.
  Control or comparison: single-layer PCC identical with and without fills (zero host padding).
  Likely subsystem: sharded `rms_norm` padding handling.
  Investigation performed: single-layer screen only.
  Resolution: more-work-needed (padding-injection test or restore fills).
- Observed anomaly: attention-out BFP4 prefill +1.06 ms.
  Evidence: `S5/trials/b4_full_x4/commands.json` (18354 -> 19413 us, one run each).
  Affected path: prefill out-projection with BFP4 weight.
  Control or comparison: none repeated; Stage 4 single-run spread up to 3.8 ms.
  Likely subsystem: timing noise or prefill matmul config with BFP4.
  Investigation performed: none.
  Resolution: more-work-needed (see P1 precision).
- Observed anomaly: SDPA decode k512 fell back to k64 on a 1026-block table.
  Evidence: `S5/micro/long_context/x4_full_32k*.json`.
  Control or comparison: 16-block aligned table 310.6 vs 500 us.
  Resolution: controlled (documented page-table contract in `AP/doc/context_contract.json`).
- Observed anomaly: chain 1 sharded residual broke one-token prefill groups; chain 2 harness defects; duplicate
  BF16 GDN weight.
  Evidence: `OD/evidence1_aborted/`, `OD/evidence2/`, `OD/evidence2/acceptance_rerun/`.
  Resolution: fixed (chain 3 `CHILD_FAILURES 0`, `SOURCE_STABLE`; inventory shows the release).
- Observed anomaly: aborted FillPad screen after a kernel edit during a device run.
  Evidence: `S5/trials/fill_*_x4_aborted_source_edit`.
  Resolution: controlled (kept, not used as evidence; rerun in `S5/trials/sf_*_x4`).

## Verified Claims

- `E3/source_manifest_start.json` equals `source_manifest_end.json`. All 43 hashed files match the live worktree.
  `E3/source_snapshot.tar.gz` matches the live `AP/tt` and `AP/tests` byte for byte, including `AP/tt/kernels/`.
- `S5/evidence_chain3.log`: 159 `rc=` lines, none non-zero; `SOURCE_STABLE`, `CHILD_FAILURES 0`, `CHAIN_DONE`.
- Post-chain trials `S5/trials/b4_*_x4`, `geo_linear_x4`, `geo_full_x2` record source hashes equal to the chain.
- `E3/memory_handoff_recheck.json` source hashes equal the live files.
- Headline decode and prefill values in `OD/README.md` equal `E3/headline_*/commands.json`. The 40-layer sums
  recompute (10.81 / 774 ms on p150x4).
- Prefill repeat 85 of 92 rows: the 7 drifting rows match the README list.
- `runtime_audit` is true in all 151 inspected rows. 20 watcher logs contain no error, assert or hang line.
- Native context 262143/262144 rows exist on all profiles; `capability_reduction` is null.
- CPU tests 1179 pass, 0 fail (`E3/cpu_pytest.xml`). Device pytest 12 passed (`E3/accept_pytest/device.log`).
- MoE decode keeps gate-selected experts: the expert kernel reads expert ids on device.

## Scope Inspected

- Goal/skill paths: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/05-05-optimized-multichip-decoder.prompt.txt`,
  `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`,
  `.../skills/optimize/SKILL.md` and `.../skills/multichip/SKILL.md` (targeted reads of batch, deferral,
  topology, precision, CCL and geometry rules). I did not reread `tt-device-usage` in full; no device use was
  needed.
- Artifact paths: `OD/README.md`, `OD/work_log.md`, `E3/` (manifests, snapshot, headline, `len_*`, `b32_*`,
  `real_inputs_m*`, `accept_*`, `watcher_long`, `long_context`, `profiles/*` summaries and perf CSV,
  `capacity_model.json`, `memory_handoff_recheck.json`, `nondeterminism/`, `kernels/`), `AP/doc/context_contract.json`,
  `S5/evidence_chain3.sh`, `S5/evidence_chain3.log`, `S5/trials/{prec_*,b4_*,norm_*,tie_check_x4,statel1_x4,geo_*}`,
  `S5/micro/{inproj_sweep.json,geometry_full_sweep.json,direct_allreduce_x{2,4}.json}`, `E7/` headline,
  positions, b32 and `micro/residual_contract_x4.json`, `AP/doc/multichip_decoder/README.md`,
  `AP/doc/multichip_decoder/stage_review7.md`, `AP/doc/optimized_decoder/README.md` and `work_log.md` (padding
  and tie findings).
- Code paths: `git diff HEAD` of `AP/tests/hf_reference.py`, `AP/tests/run_layer.py`, `AP/tests/run_positions.py`,
  `AP/tt/optimized_decoder.py` (fill and policy lines); `AP/tt/multichip_decoder.py` (`_decode_allreduce`,
  `_ShardedResidualLayer`, `_AllReduceOutput`, `_release_gdn_input_weight`, construction).
- Commands run: `git status`, `git log`, `git diff`, `sha256sum`, `cmp`, `diff -r` against the extracted snapshot,
  `ls`, `grep`, `zgrep`, `ps`, and Python over JSON and logs. No device, server, build or device test.

## Residual Risk

- The tie-aware gate accepts any expert set inside a 2-ULP band. A real router bias smaller than one BF16 ULP at
  the boundary is indistinguishable from rounding.
- Full-model behavior with padding fills off and the sharded residual over 40 layers is unmeasured.
- MoE prefill nondeterminism has no root cause; greedy outputs of identical requests can differ.
- Long-context numbers use random hidden inputs and check finiteness only; accuracy at 131072 rests on the
  262143-token single-step acceptance rows.
- Concurrency-32 serving at ISL 131072 does not fit the p150x4 capacity model with the current expert layout.
