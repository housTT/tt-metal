# AutoFix: persistent allocations after trace capture

**Status: H1 verified and fixed in the matched-test harness.** Hardware experiments were run serially by the coordinator on a single Blackhole chip on P300c hardware. This report was prepared from source and saved artifacts without device access.

## Starting evidence

- Diagnosis: [AUTODEBUG_transpose_trace.md](AUTODEBUG_transpose_trace.md).
- Original failure: [reuse_outer_matched_v1.log](logs/reuse_outer_matched_v1.log), with [provenance](logs/reuse_outer_matched_v1.provenance.json) and the frozen source archive it identifies. Eager comparison passed; final stress PCC was **0.25064918168434985**, below 0.995. The allocator warned about device allocations after a live trace at 23:15:36.174.
- The harness constructed/captured runtime A, then allocated decoder B's weights, state, and inputs. The allocator does not retain reservations for deallocated intermediates whose addresses remain baked into A's trace (`tt_metal/impl/allocator/allocator.hpp:174–175`).

Original failing command, in the recorded task environment:

```bash
OMP_NUM_THREADS=8 ORNITH_WEIGHTS=real pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_fusion_equivalence.py -k reuse_outer_matched_timing -x -v -s
```

## Hypothesis experiment

**H1:** A replay overwrites later persistent allocations belonging to B. It predicts observable corruption even when both decoders use the identical runtime and B never executes a replay.

The opt-in control in `tests/test_fusion_trace_allocations.py:28` preserves the bad setup order with two identical `FusedDecoder` instances. It takes host snapshots of B's token, position tensors, small weights, and recurrent/convolution state, then executes A exactly once and never replays B. It uses a reserved 32 MiB trace region, so reserving trace command storage alone does not prevent this hazard.

```bash
ORNITH_TRACE_ALLOCATION_REPRO=1 OMP_NUM_THREADS=8 ORNITH_WEIGHTS=real pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_fusion_trace_allocations.py -x -v -s
```

**Result: verified, 1 passed.** [trace_allocation_repro_v1.log](logs/trace_allocation_repro_v1.log) records eager PCC **0.9999999999999812** and A's replay output exactly equal to its eager output. **Twelve persistent B buffers changed despite B never replaying.** Selected measurements:

| B buffer | Buffer ID | Address | Changed elements | Nonfinite elements afterward |
| --- | ---: | ---: | ---: | ---: |
| recurrent state | 1472 | 15654528 | 425981 / 524288 | 0 |
| convolution state 0 | 1474 | 15261312 | 8186 / 8192 | 0 |
| attention norm weight | 1370 | 15163008 | 4096 / 4096 | 0 |
| feed-forward norm weight | 1374 | 15195776 | 4096 / 4096 | 0 |
| convolution tap 2 | 1410 | 15490688 | 8190 / 8192 | 1 |
| convolution tap 3 | 1414 | 15556224 | 8192 / 8192 | 6 |

The remaining changed buffers were GDN norm, A_neg, dt_bias, KDA norm vector, and convolution taps 0 and 1. The token and position probes remained unchanged. The log contains every probe's ID, address, changed-element count, and maximum error. Restoring mutable state could not repair the overwritten weights.

The control releases both traces and is skipped unless `ORNITH_TRACE_ALLOCATION_REPRO=1` (`test_fusion_trace_allocations.py:21–24`). A pass deliberately means corruption was reproduced; it is diagnostic evidence, not an ordinary model acceptance test.

## Fix and verification

The coordinator changed only the matched harness setup order for the repair: build, prefill, warm, and retain **both** decoders and all persistent inputs before capturing either trace. The separate capture loop is at current `tests/test_fusion_equivalence.py:149–155`. An immediate one-replay/eager exact-equality check at lines 159–162 catches corruption before timing. State restore remains a host-to-existing-device-buffer copy; no precision, kernel, or state-update changes were required.

The original selected test command above then passed twice:

| Run | Measured windows × replays | Eager PCC | Stress PCC | Result |
| --- | ---: | ---: | ---: | --- |
| [reuse_outer_matched_v2](logs/reuse_outer_matched_v2.provenance.json) | 15 × 64 | 0.9999999999999812 | 0.9999999999999881 | 1 passed |
| [reuse_outer_matched_v3](logs/reuse_outer_matched_v3.provenance.json) | 31 × 256 | 0.9999999999999812 | 0.999999999999993 | 1 passed |

Both runs also passed exact one-replay/eager equality before and after timing and the recurrent/convolution state comparisons. Each run includes two additional warmup windows. These results verify the harness repair without weakening the failing PCC gate.

**Production code was untouched by the H1 repair.** The original failed run, positive corruption control, and both passing matched runs all record the same `tt/fused_decoder.py` SHA-256:

```text
3cf319b63a169574a20d318b7cdc78ecff03aba5498e07de862ec1de5fb84083
```

Relevant experiment test SHA-256 values, with complete source hashes and commands in the linked provenance:

| Run | Recorded test file | SHA-256 |
| --- | --- | --- |
| failed matched v1 | test_fusion_equivalence.py | `94f377c4f8cffc6878ae0cc11709f753c63bbf73573ad8e69291e8d1fb26f189` |
| [corruption control v1](logs/trace_allocation_repro_v1.provenance.json) | test_fusion_trace_allocations.py | `c362ae8b4684177fd24ebb236a0d2b40dc2173d7de6db120a897bec25970550f` |
| corrected matched v2 | test_fusion_equivalence.py | `5e6e023382792961e48c8dda9b436c9ed95adf93c0877b2b371a2e4044896943` |
| corrected matched v3 | test_fusion_equivalence.py | `8f43b57c07f777e2a760835c521039686c5c2e63d70d901df0fd7df1e32eb4e9` |

The report preparation verified all four log and compressed-source-archive hashes against their provenance. Frozen archives define each experiment's source; current source has subsequently advanced.

## Subsequent selection and remaining scope

The passing v3 timing data showed native reuse faster in **25/31** windows: mean native-minus-default **−0.248252 µs**, sample standard error **0.040963 µs**, paired median **−0.250328 µs** per replay. These are small, single-run paired timing differences, not an end-to-end serving claim.

After H1 was fixed, the coordinator separately selected whole-head native transpose for the runtime. The current matched test consequently compares `T.DefaultOuterGDN` (the frozen prior default graph) with `FusedDecoder` (`test_fusion_equivalence.py:134`); `DefaultOuterGDN` is defined at `tests/transpose_fusion_candidates.py:82`. The verified allocation-before-capture ordering remains intact. Later runtime-selection validation belongs to the final stage artifacts; the H1 result requires no kernel fix.

The alternative theories of native-kernel failure, 64-replay numerical collapse, or missing state restoration were unnecessary to explain this failure: identical runtime graphs reproduced corruption after one replay, and correcting allocation order passed longer stress with the runtime source unchanged.
