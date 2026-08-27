# Host-backed segmented trace AutoFix

Date: 2026-08-27

## Symptom

Layer 1 was compared against the same eager host-backed TP2 layer for four
changing hidden inputs and token ids `(23, 91, 248044, 7)`.  The test preserved
PLE history and GDN recurrence across tokens.  Capture plus the first two
external replays matched exactly.  On the fourth token:

| Check | Result |
| --- | ---: |
| route ids, tokens 0/1/2/3 | exact / exact / exact / different |
| output PCC, tokens 0/1/2/3 | 1.00000012 / 1.00000036 / 1.00000000 / 0.92648160 |
| final recurrent-state PCC | 0.82660490 |
| final oldest FIR-tap PCC | 0.99999964 |
| final middle FIR-tap PCC | -0.01665942 |
| final newest FIR-tap PCC | 1.00000036 |
| all nine PLE state taps | exact (PCC >= 0.99999964) |

The newest projected GDN row staged in fixed DRAM was bit-exact, while a
separately staged recurrent result inherited the same 0.82660490 divergence as
the live recurrence.  The fourth-token failure therefore follows a corrupted
previous recurrent input rather than PLE, expert service, or the new hidden
input.

## Isolated variants

Every variant was tested alone against the same eager oracle.  Failed variants
were removed from the delivered source.

| Variant | Observation | Conclusion |
| --- | --- | --- |
| original front/back segmented traces | newest FP32 L1 tap bad immediately after front | back trace and final output are not producer |
| fixed pre-capture DRAM source for newest row | same failure | transient source lifetime refuted |
| `add(..., output_tensor=L1)` instead of `copy` | same failure | copy-op-only bug refuted |
| newest/all convolution taps resident in DRAM | changed optimized GDN numerical/kernel contract, PCC about 0.8 | not an acceptable baseline-preserving fix |
| attention and router split traces | later replay still fails | monolithic front lifetime refuted |
| separate tiny captured L1 state commit | identical token-four/state failure | captured commit ordering refuted |
| state commit after back trace | identical failure | back overwrite refuted |
| warmed eager D2D commit between live traces | identical failure | dispatch versus capture of commit refuted |
| canonical DRAM shadows hydrated into original L1 compute buffers per front | identical failure | cross-replay shadow storage alone is insufficient |

Allocation tracking found the dynamic final `mac` output and was clean after
that explicitly disposable output was marked corruptible.  That tracker does
not prove that a pre-existing persistent L1 address is absent from captured
scratch footprints.  The first localized bad tap had per-rank address 1412800
and unique id 6205; it did not alias any live retained front tensor or DRAM
output at the Python tensor level.

## Surviving trace control

The cleaned layer-3 QSA path captures one warmed front trace, performs declared
compact route-id D2H and exact fixed-slot expert service outside capture, then
captures/replays one warmed back trace.  Four changing hidden inputs and
positions `0..3` pass route, output PCC >= 0.995, shuffled page-table, local
KV/index cache, and final exact-cache checks at `max_seq_len=4096`.

## Verdict

`$autofix` failed to produce correct progressing GDN trace replay on the
current P300 runtime without changing the optimized baseline's numerical
contract.  `HostBackedSegmentedDecodeTrace.capture` therefore rejects linear
attention layers explicitly.  This is the remaining stage blocker: 36 of 48
layers are GDN, so the host-backed decoder cannot be accepted as a traced
full-stack baseline even though direct eager TTNN correctness and QSA segmented
trace replay pass.
