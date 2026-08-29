# AutoFix result: focused layer PCC regression is outside this change

## Verdict

AutoFix failed to produce a justified source fix for
`test_host_backed_layer0_decode_matches_optimized_reference`: the test returns
the same deterministic decode PCC, `0.8637402653694153`, with the optimized
peer-reset policy, with peer-zero copies forcibly restored, with an explicit
device synchronization after `ensure_indexed`, and from a clean detached copy
of the stage's starting HEAD (`adcfaa2191584bdb6e56c1d2e8b49ecc278a9a51`).

This is a pre-existing focused-test/current-HEAD regression, not evidence that
direct persistent-slot H2D or conditional peer-zero reset is incorrect. No
speculative fix was retained. The vLLM serving stage instead keeps the exact
all-slot cache controls, stale-input/isolation gates, sampling tests, and real
server output/benchmark evidence that exercise the changed path.

## Isolated experiments

| Experiment | Changed variable | Result | Artifact |
|---|---|---|---|
| Optimized cache path | Direct owner H2D; skip peer reset while owner is unchanged | PCC `0.8637402653694153` | `focused_owner_zero_tt.xml` |
| Forced peer-zero control | Force `reset_non_owner=True` on every miss | Identical PCC and displayed tensors | `autofix_force_peer_zero_control.xml` |
| Clean-HEAD control | No optimized-vLLM source changes; exact starting HEAD | Identical PCC and displayed tensors | `autofix_clean_head_control.xml` |
| General ordering A/B | Synchronize the mesh after `ensure_indexed` and before expert consumption | Identical PCC and displayed tensors | `autofix_sync_after_indexed_tt.xml` |

The synchronization experiment refutes the remaining cache-service/consumer
ordering hypothesis proposed by AutoDebug for this symptom. The forced-reset
and clean-HEAD controls refute peer-zero omission and direct-slot H2D as its
cause. The diagnostic synchronization was removed after the A/B.

## Positive exactness evidence retained

`direct_target_completed_cache_tt.xml` passes 20 completed ten-expert waves,
checks every one of the ten persistent slots on both ranks and for both packed
projections, forces all ten owner flips, and rechecks exact placement. It
records `12.496113 GB/s` aggregate completed owner H2D, `2.171385 ms` p50 and
`2.699363 ms` p95 service latency, zero owner D2D, correct peer-zero skips, and
correct owner-flip resets.

The real reduced vLLM virtual-batch-2 test in
`async_feedback_virtual_b2_tt.xml` additionally passes changed tokens,
current positions, changed and unchanged page tables, stale-generation
rejection, cancellation, request isolation, non-aligned lengths 63 and 67,
and the async feedback split on the direct-slot implementation.

## Remaining diagnostic scope

Historical artifacts show the focused layer test passing at the earlier
multichip stage, so the unresolved regression interval begins after commit
`5b8898664b3` and ends no later than the starting HEAD. Isolating that broader
model regression requires a source bisect and stage-level intermediate tensor
comparison; it is not a valid reason to mutate the optimized vLLM adapter or
cache path without causal evidence. `AUTODEBUG.md` contains the source audit
and ranked follow-ups.

AutoFix is therefore exhausted for the stage-relevant suspected bug. Its
failure and the unresolved pre-existing test are carried as a limitation for
independent stage review.
