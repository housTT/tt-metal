# Stage Review

Verdict: more-work-needed

## Required Work

- P1: Prove or remove the decode `nnz=8` invariant before using the fixed-count sparse kernel.
  Evidence: `tt/multichip_decoder.py:1547-1550` passes `nnz=TOP_K_EXPERTS` to every packed decode sparse matmul. The sparsity tensor is not an independently constructed eight-entry mask: `tt/optimized_decoder.py:2020-2053` typecasts router input and logits, applies top-k and softmax, scatters BF16 scores, and may multiply/typecast them by per-expert scales before returning the tensor used as sparsity. No code or test asserts `count_nonzero(sparsity) == 8` for every token or batch item. The stacked test at `tests/test_multichip_decoder.py:3719-3724` calls `torch.topk(..., k=8)`, which always returns eight indices and therefore does not establish eight nonzero values. The work log itself listed exact `nnz=8` as an action at line 26, but no validating artifact was retained.
  Why this matters: The optimize contract says the explicit sparse count is exact, not an upper bound; router dtype conversion, scaling, and Blackhole zero flush are named reasons that can change it. A receiver that loops eight times while the sender skips fewer than eight entries can hang or wedge the device. Passing the current random-input tests only establishes that those samples did not trigger the hazard.
  Required next step: Either omit `nnz` and measure the robust runtime-inference path, or construct a semantically equivalent sparsity mask whose exact count is guaranteed independently of routing-score magnitude. Add an assertion/test that observes the actual post-conversion tensor for real recorded activations plus adversarial score/scale cases, and rerun B1/B32 eager, trace replay, PCC, timing, and watcher evidence on TP1/TP2/TP4.

- P1: Finish the dominant packed-expert geometry and current profiler-advice search.
  Evidence: `tt/multichip_decoder.py:423-431` derives expert `per_core_N` from the old unpacked local width; TP2's 352-wide/11-tile projection consequently forces `per_core_N=1`. This stage enables packing by default at lines 435-442 and uses a 704-wide/22-tile packed result at lines 1530-1539 and 668, but retains that old value. The legality test compounds the stale contract by testing `n=profile.local_moe_intermediate_size`, not the packed width, at `tests/test_multichip_decoder.py:117-148`. The current helper (`tt/optimized_decoder.py:760-800` and `803-833`) makes `per_core_N=2`, 11 projection cores, and a 1x2 subblock legal for a 22-tile output. Instead, the final TP2 profiler rows are marked `SLOW` and use 22 cores, `in0_block_w=44`, `per_core_N=1`, and output subblock 1x1. Across the 32 S=1024 chunks this one row consumes 22.575 ms for sliding prefill and 25.375 ms for full prefill; decode consumes 181.612/180.739 us across three replays. Its advice explicitly requests output-subblock area at least two. No directory under `candidates/` tests a packed TP2 `per_core_N`/subblock alternative. Separately, TP2 sliding QKV is a 183.436-us/three-replay row whose advice requests a DRAM-sharded config, but the only TP2 DRAM candidate is `candidates/tp2_dram_o/`; no QKV candidate or exact blocker exists. This contradicts README lines 85-107 and work-log lines 248-269, which claim all advice is closed and nothing is deferred.
  Why this matters: Packed experts are the stage's headline change and the dominant prefill family. The stage cannot call that family optimized while it reuses geometry chosen for a different N dimension and leaves an obviously legal advised geometry unmeasured. The explicit QKV recommendation is another current, material recommendation that was neither tried nor precisely blocked.
  Required next step: Run a precision-locked TP2 packed expert sweep including `per_core_N=2`/1x2 and any other legal subblock/core choices for both prefill and traced decode, with full routed-chain latency and PCC. Try TP2 sliding QKV DRAM sharding under the selected residual/precision contract or retain exact capacity/divisibility/L1/op-contract evidence. Add the optimize-required dominant-matmul search table with shape, dtype/fidelity, grid/core count, shard geometry, block/subblock values, memory configs, row and whole-layer latency, correctness, and decision.

- P1: Complete precision/topology coverage for the release-blocking TP profiles instead of extrapolating a global TP4 failure.
  Evidence: `candidates/ccl_bfp8/` contains only P150x4 evidence. A TP4 BFP8 all-reduce miss does not reject TP2, whose rank count/topology and numerical reduction differ. `candidates/activation_bfp8/` contains only one hand-authored `pcc_result.json` for a global P150x4 B32 switch: there is no runner log, JUnit, timing, source hash, or recorded-activation artifact. It does not isolate attention/CCL, MLP, MoE, and residual/norm activation policies, and neither candidate is crossed with TP2's selected coherent-R22 topology. The performance harness itself creates random hidden tensors (`tests/test_functional_decoder.py:1614-1616`), so the label "real-weight" is not evidence of recorded target-model activations. README lines 104-107 and work-log line 121 nonetheless treat the activation/CCL precision family as closed for the stage.
  Why this matters: P150x2 is independently release-blocking and has three repeated all-reduces per layer. It can pass a lower-precision CCL policy that TP4 fails, and a mixed role-specific activation policy can pass even when the blunt global switch fails. The optimize contract explicitly requires per-group precision trials using real weights and recorded activations, plus a small combined matrix crossing activation/CCL dtype, material weight dtype/fidelity, and the selected topology.
  Required next step: At minimum measure BF16 versus BFP8 CCL on TP2 for both layer kinds under coherent R22, with real recorded activations, B32/PCC, and traced B1 latency. Isolate BFP8 attention/CCL-facing activation from MLP/MoE and residual/norm activation on the best TP2 and TP4 topologies. Retain raw runner/JUnit/timing/provenance artifacts and summarize the crossed matrix; do not use the unbound JSON as the sole rejection evidence.

- P2: Repair profiler, performance-accounting, and command provenance artifacts.
  Evidence: Every retained `profiler/*/{prefill,decode}_report.txt` is CSV-mode console chatter, not the required human-readable table; for example `profiler/p150x2_sliding/decode_report.txt:1-15` says "Writing CSV output" and contains warnings rather than rows. This is the exact artifact error prohibited by `optimize/SKILL.md:515-536`. `perf_summary.json:15-33` publishes final 30-replay host latency from `final_post_format/` beside device totals from an independently captured three-replay profiler run; work-log lines 151-188 explicitly describe them as independent. It reports modeled DRAM percentages but no theoretical bytes/bandwidth roofline in ms/token, so it does not reconcile theoretical roofline, device time, and end-to-end time from the same run as required by `optimize/SKILL.md:97-107`. Finally, every final timing JSON embeds an `exact_command` for `tests/test_functional_decoder.py::test_functional_decoder_perf_profile`; e.g. `final_post_format/p150x2/layer0_sliding_attention_seq1024_batch1_host_timings.json:17-40`. The actual JUnit path is the multichip wrapper (`test_required_profile_perf_profile` for TP1/2 or `test_multichip_perf_profile` for TP4). The JSON's measured-decoder path/hash proves the multichip implementation was exercised, but the purported exact command is not the command that reproduces it and does not show the claimed fallback-throw setting.
  Why this matters: The original contract requires human-readable `tt-perf-report` tables and CSVs, reproducible final-default measurements, and roofline/device/end-to-end accounting. The CSVs support useful row analysis, but the missing tables, mixed runs, absent theoretical roofline, and wrong commands make the evidence package non-reproducible as written.
  Required next step: Regenerate advice-enabled human tables from the retained raw CSVs without `--csv`, keeping CSV-mode chatter under a console-log name. Retain source/test/build/hardware provenance for each profile capture. Produce same-run theoretical roofline ms/token, signposted device time, and host end-to-end decode with named gaps/limitations. Fix the delegated harness to set `GEMMA4_EVIDENCE_COMMAND` (or otherwise record the real wrapper command and fallback policy), then rerun the final default measurements rather than hand-editing old provenance.

- P2: Close the Ethernet-watcher limitation for the multi-chip CCL path.
  Evidence: The only final watcher command, at work-log lines 221-232, sets `TT_METAL_WATCHER_DISABLE_ETH=1` while validating a stage whose core path uses persistent asynchronous inter-chip all-reduces. README lines 161-162 call the result watcher-clean, but neither README, work log, nor `final/watcher.log` retains the prerequisite ACTIVE_ETH config-buffer overflow, an attempted watcher run with Ethernet enabled, or a scoped limitation. `optimize/SKILL.md:508-513` permits this retry only after that overflow signature and requires the limitation to be recorded.
  Why this matters: The ten passing JUnit cases and clean worker-core log are useful, but they do not support an unqualified watcher-clean claim for the Ethernet kernels carrying the stage's three per-layer reductions.
  Required next step: Prefer a bounded selected-profile watcher run with Ethernet watching enabled. If the documented ACTIVE_ETH buffer limit reproduces, retain that failure log, use the allowed disabled-ETH retry, explicitly scope the limitation, and preserve the strongest targeted CCL correctness/health control available without overstating Ethernet watcher coverage.

## Other Concerns

- The packed tensor is resident-capacity-neutral, but construction uploads separate gate/up tensors (`tt/multichip_decoder.py:593-601`), then uploads the packed tensor (lines 659-678), and only then deallocates the separate tensors (lines 679-685). The capacity JSON models persistent bytes, not this transient duplicate. This may be harmless if all decoder weights are built before large KV allocations, but TP1's reported full-stack headroom is only 1,037,824 bytes; full-model bringup must account for construction order or eliminate the transient.
- All prefill reports were generated with `--active-experts 8`, while prefill constructs the union of per-token routes over each 32-token group and uses `nnz=None` (`tt/optimized_decoder.py:3013-3026`). Device durations remain usable, but the report's `active=8/128` labels, modeled FLOPs/bytes, and prefill roofline/advice model are not proven to represent the actual group union.
- Final TP4 full decode is 0.12% slower than baseline (0.985535 versus 0.984365 ms). The final number is disclosed and correctly used, but the claim that the delta is noise has no retained repeated-run variance or alternating-order stability evidence.

## Hard-Check Gaps

- The final report does not include the optimize-required cumulative contract tables for dominant matmuls, SDPA/cache program configuration and row, norm/residual rows and layouts, and every material CCL row's full program/buffer fields. The CSVs expose some row fields, but not a complete kept/rejected configuration record.
- The activation-BFP8 rejection has only summary JSON, so no hard check binds it to source/test/build/hardware or proves the command passed with fallback throwing.
- The stage reuses TP1 50,623/50,624 capacity evidence on the assertion that both persistent bytes and activation peak are unchanged, but no check covers the new temporary separate-plus-packed construction lifetime.
- No retained stability distribution supports the TP4 full decode "measurement noise" classification.

## Anomaly Ledger

- Observed anomaly: Fixed sparse `nnz=8` is supplied without an exact post-conversion nonzero invariant.
  Evidence: `tt/multichip_decoder.py:1547-1550`; router transformations at `tt/optimized_decoder.py:2020-2053`; no relevant `count_nonzero` test.
  Affected path: Packed expert decode on TP1/TP2/TP4, B1 and per-row B32.
  Control or comparison: Prefill uses runtime inference (`nnz=None`) for routed groups; current random real-weight samples pass.
  Likely subsystem: Sparse routing metadata / `ttnn.sparse_matmul` fixed-count sender-receiver contract.
  Investigation performed: Source and test inspection, artifact search for exact-nnz validation, and comparison with the optimize sparse-matmul contract.
  Resolution: more-work-needed.

- Observed anomaly: TP2 packed experts use geometry chosen for the unpacked 352-wide output.
  Evidence: Source defaults and test at `tt/multichip_decoder.py:423-442`, `tests/test_multichip_decoder.py:117-148`; final 704-wide TP2 rows are 22-core/1x1 `SLOW` rows with untried advice.
  Affected path: TP2 packed expert gate/up prefill and decode for sliding and full attention.
  Control or comparison: The config helper makes 11-core, `per_core_N=2`, 1x2 legal for 22 output tiles; no measured candidate exists.
  Likely subsystem: Sparse matmul program geometry after projection packing.
  Investigation performed: Re-derived tile/core divisibility from source and aggregated final CSV row times across calls/replays.
  Resolution: more-work-needed.

- Observed anomaly: Precision-family closure is inferred from TP4-only global candidates.
  Evidence: Only `candidates/ccl_bfp8/p150x4/` and one `candidates/activation_bfp8/pcc_result.json` exist; no TP2 or role-isolated artifacts.
  Affected path: TP2 CCL and TP2/TP4 attention/MLP/MoE activation policies.
  Control or comparison: TP4 global BFP8 activation and CCL both miss PCC; BF16 defaults pass.
  Likely subsystem: Precision/topology policy selection and evidence coverage.
  Investigation performed: Enumerated all candidate artifacts and environment/provenance fields; compared them with the profile-specific contract.
  Resolution: more-work-needed.

- Observed anomaly: Profiler table files and exact-command provenance do not describe the retained runs.
  Evidence: `profiler/*/*_report.txt` contains CSV console boilerplate; final timing JSON commands name the functional test while profiler/final JUnit names the multichip wrappers; roofline/device/e2e values are from different runs.
  Affected path: All six headline profile/layer-kind measurements.
  Control or comparison: Measured-decoder hashes match current `multichip_decoder.py`, and CSV device-time sums reproduce `perf_summary.json`.
  Likely subsystem: Evidence harness delegation and profiler post-processing.
  Investigation performed: Parsed JSON/JUnit, compared hashes and commands, inspected report text, and independently summed CSV device rows.
  Resolution: more-work-needed.

- Observed anomaly: The final watcher run disables Ethernet checking.
  Evidence: Work-log command uses `TT_METAL_WATCHER_DISABLE_ETH=1`; no retained overflow/control artifact or scoped limitation.
  Affected path: Persistent TP2/TP4 asynchronous all-reduces.
  Control or comparison: Ten worker-watcher cases pass; `tt_smi_post.json` reports healthy devices and zero GDDR errors.
  Likely subsystem: Watcher evidence coverage for CCL/Ethernet kernels.
  Investigation performed: Read watcher command, JUnit, log, health JSON, and searched stage evidence for the allowed overflow rationale.
  Resolution: more-work-needed.

- Observed anomaly: TP2 stacked layer 5 selects seven of eight reference experts after consuming approximate layer-0 output.
  Evidence: Final stacked artifact and `tests/test_multichip_decoder.py:3681-3729` distinguish chained inputs from same-input controls.
  Affected path: Two-layer TP2 chained decode stress only.
  Control or comparison: Same-input layer-5 PCC passes, chained output passes its declared 0.98 stress threshold, and 20 trace replays are bit-exact.
  Likely subsystem: Expected MoE routing sensitivity to an approximate upstream activation.
  Investigation performed: Compared same-input and chained checks and read the test's explicit control rationale.
  Resolution: controlled.

- Observed anomaly: Final TP4 full decode is slightly slower than the baseline.
  Evidence: `perf_summary.json:31-32` reports speedup 0.998812; README lines 70-75 discloses the regression as noise.
  Affected path: P150x4 full-attention B1 traced decode.
  Control or comparison: Final post-format run reproduces the pre-format run within 0.16%, PCC passes, prefill improves 24.9%, and retained storage is reduced.
  Likely subsystem: Small-run latency variance versus the packed-expert/storage tradeoff.
  Investigation performed: Recomputed baseline/final ratios and compared pre/post-format results.
  Resolution: controlled for headline reporting, with variance evidence still a hard-check gap.

## Scope Inspected

- Goal/skill paths: supplied optimized-multichip-decoder contract; `.agents/skills/stage-review/SKILL.md`; `.agents/skills/optimize/SKILL.md`; `.agents/skills/tt-device-usage/SKILL.md`; repository `AGENTS.md` instructions supplied with the task.
- Artifact paths: all files under `doc/optimized_multichip_decoder/` relevant to README/work log, topology, performance, capacity, baseline/final/final-post-format correctness and timing, stress/watcher/health, candidate metadata, and profiler CSV/report/JUnit provenance; `doc/context_contract.json`; inherited `doc/multichip_decoder/AUTODEBUG_FUSED_RS.md` and `AUTOFIX_FRACTURED_RESIDUAL.md` plus referenced controls.
- Code paths: `tt/multichip_decoder.py`, relevant routing/MoE/config sections of `tt/optimized_decoder.py`, and the multichip/functional harness sections of `tests/test_multichip_decoder.py` and `tests/test_functional_decoder.py`.
- Commands run: read-only `git status/diff/diff --check/rev-parse`, `sha256sum`, `rg`, `find`, `sed`, `nl`, `stat`, JSON/JUnit inspection, and small Python CSV/JSON aggregation scripts. No server, device, reset, profiler, watcher, test, or hardware command was launched by this reviewer.

## Residual Risk

- The retained real-mesh IDs, all-profile real-weight PCC, S=33 nonalignment, TP2/TP4 stacked trace replay, current decoder hash, no-fallback correctness JUnit, post-run health, and inherited adapted fused/fractured-residual rejection are credible positive evidence. They do not close the fixed-count sparse hazard or the missing optimization/evidence work above.
- This review was artifact- and source-based by design. Hardware confirmation for each remediation must be performed by the stage owner under `$tt-device-usage`, with watcher and profiler kept separate.
- Until Required Work is closed and a fresh independent review returns `clean-pass`, this stage must not advance to full-model bringup.
